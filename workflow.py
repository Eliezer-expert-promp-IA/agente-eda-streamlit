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
import os
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
            # Adiciona configurações de segurança para evitar bloqueios de resposta da API do Gemini.
            # Isso previne o erro "ValueError: No generation chunks were returned" e o "ModuleNotFoundError".
            try:
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            except ImportError:
                # Fallback para ambientes onde 'google-generativeai' não está instalado corretamente.
                # Usa valores brutos que são menos robustos a mudanças na biblioteca.
                # A chave '10' corresponde a HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT.
                # O valor '4' corresponde a HarmBlockThreshold.BLOCK_NONE.
                safety_settings = { 10: 4 }

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0,
                convert_system_message_to_human=True,
                safety_settings=safety_settings
            )
        elif llm_provider == "LLM de Teste (Gemini)":
            test_api_key = os.getenv("TEST_GEMINI_API_KEY")
            test_model_name = os.getenv("TEST_GEMINI_MODEL_NAME")
            if not test_api_key or not test_model_name:
                raise ValueError("As variáveis TEST_GEMINI_API_KEY e TEST_GEMINI_MODEL_NAME devem ser definidas no arquivo .env.")
            try:
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                safety_settings = { HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE }
            except ImportError:
                safety_settings = { 10: 4 }
            
            llm = ChatGoogleGenerativeAI(
                model=test_model_name,
                google_api_key=test_api_key,
                temperature=0,
                convert_system_message_to_human=True,
                safety_settings=safety_settings
            )
        elif llm_provider == "OpenAI":
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0)
        elif llm_provider == "Groq":
            llm = ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=0)
        elif llm_provider == "Anthropic":
            llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=0)
        else:
            raise ValueError(f"Provedor de LLM desconhecido: {llm_provider}")
    except Exception as e:
        # Imprime o erro completo no console para depuração
        print(f"ERRO ao instanciar o LLM: {e}")
        # Retorna uma mensagem de erro segura para a interface do usuário
        return "Falha ao criar o agente. Verifique se sua chave de API e o nome do modelo estão corretos e válidos."

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
5.  Você tem um limite de 5 passos (Pensamento/Ação). Se você não conseguir a resposta final em 5 passos, resuma suas descobertas na "Final Answer".

6.  **PARA GERAR GRÁFICOS:**
    - Use a biblioteca `matplotlib.pyplot` (disponível como `plt`) ou `seaborn` (disponível como `sns`).
    - **SEMPRE** salve o gráfico em um arquivo na pasta `temp_charts/` usando `plt.savefig('temp_charts/nome_do_grafico.png')`. Use um nome de arquivo único e descritivo.
    - **NUNCA** use `plt.show()`, pois isso causará um erro no ambiente de execução.
    - Após salvar, inclua a tag especial `[CHART_PATH:caminho/do/arquivo.png]` na sua "Final Answer" para que o gráfico possa ser exibido. Exemplo: `Final Answer: Aqui está o gráfico de barras solicitado. [CHART_PATH:temp_charts/fraudes_por_classe.png]`

Use o seguinte formato. As palavras-chave do formato (Question, Thought, Action, Action Input, Final Answer) DEVEM ser em inglês:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar sobre o que fazer. O seu pensamento deve ser em português.
Action: a ação a ser tomada, deve ser uma das [{tool_names}]
Action Input: o código Python puro para a ação. **IMPORTANTE**: NÃO inclua formatação de markdown como ```python ou ```.
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
        max_iterations=5,
        early_stopping_method="force",
    )

    return agente_executor