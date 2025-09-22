# workflow.py

# Importa as bibliotecas e módulos necessários
import pandas as pd
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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
    #    Injetamos o DataFrame diretamente na ferramenta Python REPL
    ferramentas = criar_ferramentas_analise(df)

    # 3. Puxa o prompt base para um agente ReAct que funciona com chat
    #    Este prompt já define o formato de Pensamento/Ação/Observação
    prompt = hub.pull("hwchase17/react-chat")

    # 4. Adiciona instruções específicas ao prompt para o nosso caso de uso
    #    É crucial informar ao agente sobre o DataFrame 'df' e como usar a ferramenta.
    prompt.template = """
Você é um analista de dados especialista. Para responder à pergunta do usuário, você DEVE usar a ferramenta 'python_code_executor' para executar código Python e inspecionar o dataframe `df`.
NÃO tente responder com base no seu conhecimento prévio.

Siga estritamente este processo:
1.  **Pensamento**: Pense em qual código Python você precisa executar para responder à pergunta.
2.  **Ação**: Use a ferramenta `python_code_executor`.
3.  **Entrada da Ação**: Escreva o código Python que você pensou. O código DEVE usar `print()` para que o resultado seja visível.
4.  **Observação**: Analise o resultado do código.
5.  Repita os passos 1-4 até ter informações suficientes. Se você atingir o limite de 3 passos, resuma suas descobertas e forneça uma resposta final.

**Importante**: O DataFrame pandas com os dados já está carregado e disponível na variável `df`.
 
Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato:

Pergunta: a pergunta de entrada que você deve responder
Pensamento: você deve sempre pensar sobre o que fazer.
Ação: a ação a ser tomada, deve ser uma das [{tool_names}]
Entrada da Ação: a entrada para a ação
Observação: o resultado da ação
... (este Pensamento/Ação/Entrada da Ação/Observação pode se repetir N vezes)
Pensamento: Agora eu sei a resposta final.
Resposta Final: a resposta final para a pergunta original

Comece!

Histórico do Chat:
{chat_history}

Pergunta: {input}
Pensamento:{agent_scratchpad}
"""

    # 5. Cria o agente ReAct
    #    Este agente combina o LLM, as ferramentas e o prompt para tomar decisões.
    agente = create_react_agent(llm, ferramentas, prompt)

    # 6. Cria o Executor do Agente
    #    O executor é o que de fato executa o loop ReAct (Pensamento -> Ação -> Observação).
    agente_executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True,  # Exibe os passos do agente no console, ótimo para depuração
        handle_parsing_errors=True, # Lida com erros de formatação da saída do LLM
        return_intermediate_steps=True, # Retorna os pensamentos e ações do agente
        max_iterations=3, # Define um limite de 3 ciclos de pensamento/ação
        early_stopping_method="generate", # Força o agente a gerar uma resposta final se atingir o limite
    )

    return agente_executor