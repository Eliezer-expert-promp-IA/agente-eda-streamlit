# workflow.py

# Importa as bibliotecas e módulos necessários
import pandas as pd
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
# from langchain_core.agents import AgentFinish # Correção: Importado de 'langchain_core'
# from langchain.schema import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI # Correção: 'GenerativeAI'
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

# Importa as ferramentas personalizadas do nosso módulo
from tools.custom_tools import criar_ferramentas_analise

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
    prompt.template = """
Você é um analista de dados experiente e domina a linguagem de programação Python. Sua tarefa é responder à pergunta do usuário sobre um conjunto de dados. Seu pensamento e sua resposta final devem ser sempre em português.

**REGRAS IMPORTANTES:**
1.  Para perguntas sobre os dados, **SEMPRE** use a ferramenta `python_code_executor` para inspecionar o dataframe `df`. NÃO tente responder com base no seu conhecimento prévio.
2.  Se a pergunta do usuário não for sobre os dados (ex: uma saudação como "oi"), responda diretamente sem usar ferramentas, usando o formato "Final Answer".
3.  O DataFrame pandas com os dados já está carregado e disponível na variável `df`.
4.  O código que você escreve para a ferramenta DEVE usar `print()` para que o resultado seja visível.
5.  Você tem um limite de 3 passos (Pensamento/Ação). Se você não conseguir a resposta final em 3 passos, resuma suas descobertas na "Final Answer".

Use o seguinte formato. As palavras-chave do formato (Question, Thought, Action, Action Input, Final Answer) DEVEM ser em inglês:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar sobre o que fazer. O seu pensamento deve ser em português.
Action: a ação a ser tomada, deve ser uma das [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Agora eu sei a resposta final.
Final Answer: a resposta final para a pergunta original, em português.

Comece!

Histórico do Chat:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}
"""

    # 5. Cria o agente ReAct
    agente = create_react_agent(llm, ferramentas, prompt)

    # 6. Cria o Executor do Agente
    agente_executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True,
        handle_parsing_errors=True, # Trata erros de parsing, dando ao agente a chance de se corrigir.
        return_intermediate_steps=True,
        max_iterations=3,
        early_stopping_method="force",
    )

    return agente_executor