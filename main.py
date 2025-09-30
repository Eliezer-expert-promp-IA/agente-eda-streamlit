# main.py

import streamlit as st
from dotenv import load_dotenv
from workflow import criar_fluxo_agente
import os
import pandas as pd
import re

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração da Página e Estado da Sessão ---

st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")

# Inicializa o estado da sessão se não existir
def inicializar_estado_sessao():
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []
    if "agente_analise" not in st.session_state:
        st.session_state.agente_analise = None
    if "df" not in st.session_state:
        st.session_state.df = None

inicializar_estado_sessao()


# --- Layout da Aplicação (Sidebar e Chat) ---

st.title("👨‍🔬 Agente Autônomo de Análise de Dados")

with st.sidebar:
    st.header("⚙️ Configuração do LLM")

    # Seleção do provedor de LLM
    llm_provider = st.selectbox(
        "Escolha o Provedor de LLM:",
        ("LLM de Teste (Gemini)", "Gemini", "OpenAI", "Groq", "Anthropic")
    )

    api_key = ""
    model_name = ""

    # Oculta os campos de API e modelo se o LLM de teste for selecionado
    if llm_provider != "LLM de Teste (Gemini)":
        # Campo para a chave de API
        api_key = st.text_input("Insira sua Chave de API:", type="password")

        # Campo para o nome do modelo com placeholders
        model_name_placeholder = {
            "Gemini": "gemini-1.5-flash-latest",
            "OpenAI": "gpt-4-turbo",
            "Groq": "llama3-70b-8192",
            "Anthropic": "claude-3-opus-20240229"
        }.get(llm_provider, "Insira o nome do modelo")
        
        model_name = st.text_input("Nome do Modelo:", placeholder=model_name_placeholder)

    st.header("📂 Upload de Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xls", "xlsx"])

    # Botão para criar/atualizar o agente
    if st.button("Criar/Atualizar Agente"):
        # Validação condicional da chave de API
        if llm_provider != "LLM de Teste (Gemini)" and not api_key:
            st.warning("Por favor, insira sua chave de API.")
        elif llm_provider != "LLM de Teste (Gemini)" and not model_name:
            st.warning(f"Por favor, insira o nome do modelo. Sugestão: `{model_name_placeholder}`")
        elif uploaded_file is None:
            st.warning("Por favor, carregue um arquivo CSV ou Excel.")
        else:
            with st.spinner("Processando arquivo e configurando agente..."):
                try:
                    # Carrega o DataFrame com base na extensão do arquivo
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_extension == ".csv":
                        df = pd.read_csv(uploaded_file)
                    elif file_extension in [".xls", ".xlsx"]:
                        # Para ler arquivos Excel, a biblioteca 'openpyxl' pode ser necessária
                        df = pd.read_excel(uploaded_file)
                    else:
                        # Esta parte é redundante por causa do 'type' no file_uploader, mas é uma boa prática
                        st.error("Formato de arquivo não suportado. Use CSV ou Excel.")
                        st.stop()
                    st.session_state.df = df
                    
                    # Cria o agente com as configurações fornecidas
                    agente = criar_fluxo_agente(df, llm_provider, api_key, model_name)
                    
                    # Verifica se a criação do agente retornou um erro
                    if isinstance(agente, Exception):
                        st.error(f"Erro ao criar o agente: {agente}")
                        st.session_state.agente_analise = None
                    else:
                        st.session_state.agente_analise = agente
                        # Reseta o chat para a nova análise
                        st.session_state.mensagens = [{"role": "assistant", "content": f"Agente configurado com `{llm_provider} ({model_name})` e arquivo `{uploaded_file.name}` carregado. O que gostaria de saber?"}]
                        st.success("Agente pronto!")
                        st.dataframe(df.head())

                except Exception as e:
                    st.error(f"Erro ao processar: {e}")
                    st.session_state.df = None
                    st.session_state.agente_analise = None

# --- Exibição da Interface Principal ---

# Se o agente ainda não foi criado, exibe a tela de boas-vindas.
if st.session_state.agente_analise is None:
    st.info("👋 **Bem-vindo ao Agente de Análise de Dados!** Configure o LLM e carregue seu arquivo na barra lateral para começar.")
    
    st.markdown("### 🎯 Funcionalidades Principais")
    st.markdown("""
    - **Análise Exploratória Automática:** Descubra as características, tipos de dados e estatísticas descritivas do seu dataset.
    - **Visualizações Inteligentes:** Gere gráficos de barras, histogramas, gráficos de dispersão e mais, de forma automática.
    - **Insights sob Demanda:** Faça perguntas em linguagem natural para extrair informações, resumos e realizar cálculos.
    - **Portabilidade de LLMs:** Escolha seu provedor preferido (Gemini, OpenAI, Groq, Anthropic) com sua própria chave de API.
    """)

    st.markdown("### 🚀 Como Começar")
    st.markdown("""
    1.  **Configure o LLM:** Na barra lateral, escolha o provedor, insira sua chave de API e o nome do modelo.
    2.  **Faça o Upload:** Carregue um arquivo CSV ou Excel (limite de 200MB).
    3.  **Crie o Agente:** Clique no botão "Criar/Atualizar Agente".
    4.  **Inicie a Análise:** Após a confirmação, a interface de chat aparecerá. Faça sua primeira pergunta!
    """)

    st.markdown("### 💡 Exemplos de Perguntas")
    st.markdown("""
    - "Faça uma análise exploratória inicial dos dados."
    - "Qual a quantidade de fraudes?"
    - "Gere um gráfico de barras mostrando a contagem de fraudes."
    """)
    st.warning("**Aviso:** O agente tem um limite de 5 interações por pergunta. Para análises complexas, divida sua pergunta em partes menores.")

# Se o agente já foi criado, exibe a interface de chat.
else:
    for mensagem in st.session_state.mensagens:
        with st.chat_message(mensagem["role"]):
            st.write(mensagem["content"])

# Exibe o histórico do chat
for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.write(mensagem["content"])

# --- Lógica do Chat ---

if prompt := st.chat_input("Qual sua pergunta sobre os dados?"):
    if st.session_state.agente_analise is None:
        st.warning("Por favor, carregue um arquivo CSV ou Excel na barra lateral primeiro.")
        st.stop()

    # Adiciona a mensagem do usuário ao histórico e à tela
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Invoca o agente e obtém a resposta
    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            # Prepara a entrada para o agente, incluindo o histórico do chat
            # Simplificado para usar a estrutura de mensagens diretamente,
            # que já é limpa e não contém dados de UI.
            entrada_agente = {
                "input": prompt,
                "chat_history": st.session_state.mensagens
            }

            try:
                # Invoca o agente
                resposta = st.session_state.agente_analise.invoke(entrada_agente)

                # Variável para armazenar a resposta limpa para o histórico
                resposta_limpa_para_historico = ""

                # Exibe os passos intermediários do agente de forma estruturada
                with st.expander("Ver Raciocínio do Agente", expanded=False):
                    intermediate_steps = resposta.get("intermediate_steps", [])
                    if not intermediate_steps:
                        st.write("Nenhum passo intermediário foi executado (ex: o agente respondeu diretamente).")
                    else:
                        for i, passo in enumerate(intermediate_steps):
                            st.subheader(f"🔄 Ciclo {i+1}")
                            acao, observacao = passo

                            # 1. Pensamento do Agente
                            st.markdown("##### 1. Pensamento")
                            # O atributo 'log' contém a cadeia de pensamento completa que o LLM gerou.
                            st.text(acao.log.strip())

                            # 2. Ação Executada
                            st.markdown("##### 2. Ação")
                            st.markdown(f"**Ferramenta:** `{acao.tool}`")
                            st.markdown("**Entrada da Ação (código executado):**")
                            st.code(acao.tool_input, language="python")

                            # 3. Observação (Resultado da Ação)
                            st.markdown("##### 3. Observação")
                            # A observação já vem formatada da nossa ferramenta customizada
                            st.markdown(observacao)
                            
                            if i < len(intermediate_steps) - 1:
                                st.divider()

                # Exibe a resposta final do agente
                resposta_final = resposta.get("output", "Desculpe, não consegui obter uma resposta.")                

                # Lógica para lidar com o limite de iteração atingido de forma mais amigável
                if "Agent stopped due to iteration limit" in resposta_final:
                    st.warning("A análise se tornou muito complexa e não foi concluída no tempo limite. Aqui está o último passo do raciocínio:")
                    intermediate_steps = resposta.get("intermediate_steps", [])
                    if intermediate_steps:
                        # Pega a última observação, que é o resultado mais recente que o agente viu
                        ultima_acao, ultima_observacao = intermediate_steps[-1]
                        # Exibe o último passo diretamente na interface
                        st.markdown("##### Pensamento")
                        st.text(ultima_acao.log.strip())
                        st.markdown("##### Observação")
                        st.markdown(ultima_observacao)
                    # Adiciona a mensagem de erro ao histórico
                    resposta_limpa_para_historico = "A análise não foi concluída no tempo limite."
                else:
                    # Se o agente concluiu, processa a resposta final normalmente
                    resposta_limpa_para_historico = resposta_final
                    # Lógica robusta para extrair e exibir um ou mais gráficos
                    chart_tag = "[CHART_PATH:"
                    if chart_tag in resposta_final:
                        resposta_limpa_para_historico = re.sub(r'\[CHART_PATH:.*?\]', '', resposta_final).strip()
                        parts = resposta_final.split(chart_tag)
                        if parts[0].strip():
                            st.write(parts[0])
                        for part in parts[1:]:
                            if ']' in part:
                                caminho_imagem, texto_depois = part.split(']', 1)
                                caminho_imagem = caminho_imagem.strip()
                                try:
                                    st.image(caminho_imagem, caption="Gráfico gerado pelo agente.", use_column_width=True)
                                except Exception as img_e:
                                    st.error(f"Erro ao exibir o gráfico em '{caminho_imagem}': {img_e}")
                                if texto_depois.strip():
                                    st.write(texto_depois)
                            else:
                                st.write(f"{chart_tag}{part}")
                    else:
                        st.write(resposta_final)
                
                # Adiciona a resposta limpa (sem tags de gráfico) do agente ao histórico
                if resposta_limpa_para_historico:
                    st.session_state.mensagens.append({"role": "assistant", "content": resposta_limpa_para_historico})
            except Exception as e:
                st.error("Ocorreu um erro durante a execução do agente. Veja os detalhes abaixo:")
                st.exception(e)
                st.session_state.mensagens.append({"role": "assistant", "content": f"Erro na execução: {e}"})