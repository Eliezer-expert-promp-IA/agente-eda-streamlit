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
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Importa as ferramentas personalizadas do nosso módulo
from tools.custom_tools import criar_ferramentas_analise

class CustomOutputParser(ReActJsonSingleInputOutputParser):
    """
    Parser customizado que tenta extrair a resposta final em caso de erro de formatação.
    """
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Analisa a saída do LLM. Se o parser JSON padrão falhar, tenta um fallback.
        """
        try:
            return super().parse(text)
        except OutputParserException as e:
            # Se falhar, executa nossa lógica de fallback para extrair a resposta final.
            if "Final Answer:" in text:
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
    # Configurações de segurança para o Gemini, para evitar bloqueios em dados de exemplo
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    try:
        if llm_provider == "Gemini":
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, convert_system_message_to_human=True, safety_settings=safety_settings)
        elif llm_provider == "OpenAI":
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0)
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
    # PROMPT ATUALIZADO: Adiciona instruções para usar a nova ferramenta de gráficos.
    prompt.template = """
Você é um analista de dados sênior. Sua tarefa é responder à pergunta do usuário sobre um conjunto de dados que ele vai disponibilizar em arquivo.

**REGRAS IMPORTANTES:**
1.  **SEMPRE** use as ferramentas disponíveis para inspecionar o dataframe `df` e encontrar a resposta.
2.  Para análise, cálculo ou manipulação de dados, use a ferramenta `python_code_executor`.
3.  Se o usuário pedir um gráfico ou visualização, **SEMPRE** use a ferramenta `chart_generator`.
4.  NÃO tente responder com base no seu conhecimento prévio.
5.  O DataFrame pandas com os dados já está carregado e disponível na variável `df`.
6.  O código que você escreve para a ferramenta `python_code_executor` DEVE usar `print()` para que o resultado seja visível.
7.  Você tem um limite de 5 passos (Pensamento/Ação) para chegar a uma resposta final. Se você não conseguir a resposta final em 5 passos, resuma suas descobertas na "".
8.  Ao usar `chart_generator`, inclua a string de saída (que contém o gráfico) diretamente na sua "Resposta Final". O gráfico DEVE ser a última coisa na resposta.

Você tem acesso às seguintes ferramentas:
{tools}

Use o seguinte formato:

Pergunta: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar sobre o que fazer. O seu pensamento deve ser em português.
Action:
```json
{{
  "action": "nome da ferramenta",
  "action_input": "a entrada para a ferramenta"
}}
```

Comece!

Histórico do Chat:
{chat_history}

Pergunta: {input}
{agent_scratchpad}
"""

    # Adicionei um exemplo de uso da ferramenta de gráfico no prompt para guiar melhor o LLM
    prompt.template = prompt.template.replace(
        "{agent_scratchpad}",
        """{agent_scratchpad}
{agent_scratchpad}
"""
    )
    
    # 5. Preenche as variáveis de ferramentas no prompt.
    # Este passo é crucial para que o LLM saiba quais ferramentas ele pode usar.
    prompt = prompt.partial(
        tools=render_text_description(ferramentas),
        tool_names=", ".join([t.name for t in ferramentas]),
    )

    # 6. Cria o agente ReAct
    agent = create_react_agent(llm, ferramentas, prompt)

    # 7. Cria o Executor do Agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=ferramentas,
        verbose=True,
        return_intermediate_steps=True, # Retorna os passos intermediários
        max_iterations=5, # Aumenta o número de passos para tarefas mais complexas
        early_stopping_method="generate", # Método mais robusto para parar a execução
        handle_parsing_errors=True, # Lida com erros de formatação do LLM
    )
    return agent_executor