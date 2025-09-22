# workflow.py

# Importa as bibliotecas e módulos necessários
from typing import Union
import pandas as pd
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Importa as ferramentas personalizadas do nosso módulo
from tools.custom_tools import criar_ferramentas_analise

# NOVA CLASSE: Parser customizado para lidar com erros de formatação.
class CustomReactOutputParser(ReActSingleInputOutputParser):
    """
    Parser customizado que tenta extrair a resposta final em caso de erro de formatação.
    """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Analisa a saída do LLM. Se o parser padrão falhar, tenta um fallback.
        """
        try:
            # Tenta usar o parser padrão primeiro.
            return super().parse(text)
        except OutputParserException as e:
            # Se falhar, executa nossa lógica de fallback para extrair a resposta final.
            if "Resposta Final:" in text:
                texto_apos_keyword = text.split("Resposta Final:")[-1].strip()
                return AgentFinish({"output": texto_apos_keyword}, log=text)
            # Se o fallback também falhar, retorna uma mensagem de erro dentro do AgentFinish.
            return AgentFinish({"output": f"Desculpe, ocorreu um erro de formatação e não consegui extrair uma resposta final. Detalhes: {e}"}, log=str(e))


def criar_fluxo_agente(df: pd.DataFrame, llm_provider: str, api_key: str, model_name: str):
    """
    Cria e compila o workflow do agente ReAct para análise de dados.

    Args:
        df: O DataFrame do Pandas que o agente irá analisar.
        llm_provider: O provedor de LLM selecionado (Gemini, OpenAI, Groq, Anthropic).
        api_key: A chave de API para o provedor selecionado.
        model_name: O nome do modelo a ser usado.

    Returns:
        Um AgentExecutor configurado e pronto para ser usado, ou uma exceção em caso de erro.
    """
    # 1. Instância do modelo de linguagem com base na seleção do usuário
    llm = None
    try:
        if llm_provider == "Gemini":
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, convert_system_message_to_human=True)
        elif llm_provider == "OpenAI":
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0)
        elif llm_provider == "Groq":
            llm = ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=0)
        elif llm_provider == "Anthropic":
            llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=0)
        else:
            raise ValueError(f"Provedor de LLM desconhecido: {llm_provider}")
    except Exception as e:
        # Retorna a exceção para ser exibida na interface do usuário
        return e

    # 2. Cria a lista de ferramentas que o agente pode usar
    ferramentas = criar_ferramentas_analise(df)

    # 3. Puxa o prompt base para um agente ReAct que funciona com chat
    prompt = hub.pull("hwchase17/react-chat")

    # 4. Adiciona instruções específicas ao prompt para o nosso caso de uso
    # PROMPT ATUALIZADO: Usa "Resposta Final:" para manter a consistência em português.
    prompt.template = """
Você é um analista de dados especialista. Sua tarefa é responder à pergunta do usuário sobre um conjunto de dados.

**REGRAS IMPORTANTES:**
1.  **SEMPRE** use a ferramenta `python_code_executor` para executar código Python e inspecionar o dataframe `df` para encontrar a resposta. 
NÃO tente responder com base no seu conhecimento prévio.
2.  O DataFrame pandas com os dados já está carregado e disponível na variável `df`.
3.  O código que você escreve para a ferramenta DEVE usar `print()` para que o resultado seja visível.
4.  Você tem um limite de 3 passos (Pensamento/Ação). Se você não conseguir a resposta final em 3 passos, resuma suas descobertas na "Resposta Final".

Você tem acesso às seguintes ferramentas:
{tools}

Use o seguinte formato:

Pergunta: a pergunta de entrada que você deve responder
Pensamento: você deve sempre pensar sobre o que fazer. O seu pensamento deve ser em português.
Ação: a ação a ser tomada, deve ser uma das [{tool_names}]
Entrada da Ação: a entrada para a ação
Observação: o resultado da ação
... (este Pensamento/Ação/Entrada da Ação/Observação pode se repetir N vezes)
Pensamento: Agora eu sei a resposta final.
Resposta Final: a resposta final para a pergunta original, em português.

Comece!

Histórico do Chat:
{chat_history}

Pergunta: {input}
{agent_scratchpad}
"""
    # 5. Preenche as variáveis de ferramentas no prompt.
    # Este passo é crucial para que o LLM saiba quais ferramentas ele pode usar.
    prompt = prompt.partial(
        tools=render_text_description(ferramentas),
        tool_names=", ".join([t.name for t in ferramentas]),
    )

    # 6. Cria o agente ReAct manualmente para injetar nosso parser customizado
    llm_with_stop = llm.bind(stop=["\nObservation:"])
    
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | CustomReactOutputParser()
    )

    # 7. Cria o Executor do Agente
    # EXECUTOR ATUALIZADO: Removemos o 'handle_parsing_errors' pois o parser já trata disso.
    agente_executor = AgentExecutor(
        agent=agent,
        tools=ferramentas,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=3,
        early_stopping_method="force",
    )

    return agente_executor