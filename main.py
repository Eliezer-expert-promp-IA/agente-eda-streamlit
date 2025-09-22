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
        st.session_state.mensagens = [{"role": "assistant", "content": "Olá! Configure o LLM na barra lateral e carregue um arquivo CSV para começar."}]
    if "agente_analise" not in st.session_state:
        st.session_state.agente_analise = None
    if "df" not in st.session_state:
        st.session_state.df = None
    # Cria um diretório para armazenar os gráficos temporariamente
    if not os.path.exists("temp_charts"):
        os.makedirs("temp_charts")

inicializar_estado_sessao()


# --- Layout da Aplicação (Sidebar e Chat) ---

st.title("👨‍🔬 Agente Autônomo de Análise de Dados")
st.write("Configure seu LLM, carregue um arquivo CSV e faça perguntas para analisá-lo.")

with st.sidebar:
    st.header("⚙️ Configuração do LLM")

    # Seleção do provedor de LLM
    llm_provider = st.selectbox(
        "Escolha o Provedor de LLM:",
        ("Gemini", "OpenAI", "Groq", "Anthropic")
    )

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
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    # Botão para criar/atualizar o agente
    if st.button("Criar/Atualizar Agente"):
        if not api_key:
            st.warning("Por favor, insira sua chave de API.")
        elif not model_name:
            st.warning(f"Por favor, insira o nome do modelo. Sugestão: `{model_name_placeholder}`")
        elif uploaded_file is None:
            st.warning("Por favor, carregue um arquivo CSV.")
        else:
            with st.spinner("Processando arquivo e configurando agente..."):
                try:
                    # Carrega o DataFrame
                    df = pd.read_csv(uploaded_file)
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

# Exibe o histórico do chat
for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.write(mensagem["content"])

# --- Lógica do Chat ---

if prompt := st.chat_input("Qual sua pergunta sobre os dados?"):
    if st.session_state.agente_analise is None:
        st.warning("Por favor, carregue um arquivo CSV na barra lateral primeiro.")
        st.stop()

    # Adiciona a mensagem do usuário ao histórico e à tela
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Invoca o agente e obtém a resposta
    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            # Prepara a entrada para o agente, incluindo o histórico do chat
            entrada_agente = {
                "input": prompt,
                "chat_history": [msg for msg in st.session_state.mensagens if msg['role'] != 'assistant' or 'passos:' not in msg['content']]
            }

            try:
                # Invoca o agente
                resposta = st.session_state.agente_analise.invoke(entrada_agente)

                # Exibe os passos intermediários do agente (pensamento e ações)
                with st.expander("Ver Passos do Agente"):
                    passos_formatados = []
                    for passo in resposta.get("intermediate_steps", []):
                        acao, observacao = passo
                        passos_formatados.append(f"**Pensamento:**\n{acao.log}\n\n**Observação:**\n```\n{observacao}\n```")
                    st.markdown("\n---\n".join(passos_formatados))

                # Exibe a resposta final do agente
                resposta_final = resposta.get("output", "Desculpe, não consegui obter uma resposta.")                
                
                # Lógica para extrair e exibir o gráfico, se houver
                chart_tag = "[CHART_PATH:"
                if chart_tag in resposta_final:
                    # Separa o texto da string do gráfico
                    texto, caminho_imagem = resposta_final.split(chart_tag)
                    caminho_imagem = caminho_imagem.strip()[:-1] # Remove o ']' final

                    # Exibe o texto
                    if texto.strip():
                        st.write(texto)
                    
                    # Exibe a imagem do arquivo
                    try:
                        st.image(caminho_imagem, caption="Gráfico gerado pelo agente.", use_column_width=True)
                    except Exception as img_e:
                        st.error(f"Erro ao decodificar e exibir o gráfico: {img_e}")
                else:
                    # Se não houver gráfico, apenas exibe a resposta
                    st.write(resposta_final)
                
                # Adiciona a resposta COMPLETA do agente ao histórico para consistência
                st.session_state.mensagens.append({"role": "assistant", "content": resposta_final})
            except Exception as e:
                st.error("Ocorreu um erro durante a execução do agente. Veja os detalhes abaixo:")
                st.exception(e)
                st.session_state.mensagens.append({"role": "assistant", "content": f"Erro na execução: {e}"})