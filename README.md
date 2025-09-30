# 👨‍🔬 Agente Autônomo de Análise de Dados

Este projeto é um agente autônomo construído com Python, Streamlit e LangChain, projetado para realizar Análise Exploratória de Dados (EDA) em arquivos CSV e Excel. O usuário pode interagir com seus dados através de uma interface de chat, fazendo perguntas em linguagem natural para obter insights, gerar visualizações

## 🎯 Funcionalidades Principais

- **Análise via Chat**: Faça perguntas em linguagem natural sobre seu conjunto de dados.
- **Visualizações Inteligentes**: Gere gráficos (barras, dispersão, histogramas, etc.) automaticamente.
- **Análise Exploratória de Dados (EDA)**: Obtenha estatísticas descritivas, informações sobre colunas e tipos de dados.
- **Portabilidade de LLMs**: Suporte para múltiplos provedores de LLM, incluindo:
  - Google Gemini
  - OpenAI
  - Groq
  - Anthropic
- **Modo de Teste**: Inclui uma opção "LLM de Teste" pré-configurada com o Gemini, permitindo o uso imediato sem a necessidade de uma chave de API.
- **Transparência do Agente**: Visualize o "raciocínio" do agente, incluindo o código Python executado e os resultados obtidos em cada passo.

---

## 🚀 Exemplos de Uso

> **[EDITAR]** - *Substitua o texto e as imagens abaixo pelos seus próprios exemplos de perguntas e respostas.*

#### Exemplo 1: Análise Exploratória Inicial

**Pergunta do Usuário:**
```
Faça uma análise exploratória inicial dos dados.
```

**Resposta do Agente:**
```
Claro. O conjunto de dados possui 1000 linhas e 5 colunas. As colunas são 'ID', 'Idade', 'Salário', 'Produto' e 'Fraude'. A média de idade é 38.5 anos e o salário médio é de R$ 55.000. A coluna 'Fraude' é do tipo booleano, indicando a ocorrência de fraude.
```

---

#### Exemplo 2: Geração de Gráfico

**Pergunta do Usuário:**
```
Gere um gráfico de barras mostrando a contagem de fraudes.
```

**Resposta do Agente:**

Aqui está o gráfico de barras mostrando a contagem de fraudes e não fraudes.

*(Cole a imagem do seu gráfico aqui. Ex: `!Gráfico de Fraudes`)*

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Interface**: Streamlit
- **Framework de Agente**: LangChain
- **Manipulação de Dados**: Pandas
- **Visualização**: Matplotlib & Seaborn
- **Variáveis de Ambiente**: python-dotenv

## 🏗️ Arquitetura do Projeto

O agente utiliza o padrão **ReAct (Reasoning and Acting)**, onde o LLM raciocina sobre qual ferramenta usar para responder a uma pergunta e então age executando essa ferramenta.

1.  **Interface do Usuário (`main.py`)**: Construída com Streamlit, a interface permite ao usuário configurar o LLM, fazer upload de um arquivo e interagir com o agente através de um chat.
2.  **Criação do Workflow (`workflow.py`)**: Este módulo é o coração do agente. Ele:
    - Instancia o LLM escolhido pelo usuário (Gemini, OpenAI, etc.).
    - Cria as ferramentas disponíveis para o agente (atualmente, uma ferramenta `python_code_executor`).
    - Carrega e personaliza um prompt ReAct, instruindo o LLM sobre como se comportar, como usar as ferramentas e o formato da resposta.
    - Monta o `AgentExecutor`, que orquestra a interação entre o LLM, as ferramentas e o prompt.
3.  **Ferramentas Customizadas (`tools/custom_tools.py`)**: Define as capacidades do agente. A principal ferramenta é o `python_code_executor`, que executa código Python em um ambiente seguro para analisar o DataFrame `df`.
4.  **Fluxo de Execução**:
    - O usuário envia uma pergunta.
    - O `AgentExecutor` passa a pergunta para o LLM.
    - O LLM gera um "Pensamento" (Thought) e decide qual "Ação" (Action) tomar.
    - A ação (ex: executar um código Python) é executada pela ferramenta correspondente.
    - O resultado ("Observação") é retornado ao LLM.
    - O ciclo se repete até que o LLM tenha uma "Resposta Final" (Final Answer) ou atinja o limite de iterações.
    - A resposta final e os gráficos são exibidos na interface.

## ⚙️ Instalação e Configuração

Siga os passos abaixo para executar o projeto localmente.

**1. Clone o Repositório**
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd agente-eda-eliezer
```

**2. Crie e Ative um Ambiente Virtual**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Instale as Dependências**
```bash
pip install -r requirements.txt
```

**4. Configure as Variáveis de Ambiente**

Crie um arquivo chamado `.env` na raiz do projeto, copiando o conteúdo de `.env.example` (se houver) ou usando o modelo abaixo.

```properties
# .env

# Chave e modelo para o "LLM de Teste (Gemini)"
TEST_GEMINI_API_KEY="sua_chave_api_do_google_aqui"
TEST_GEMINI_MODEL_NAME="gemini-.5-flash"

# Você pode adicionar outras chaves se quiser usar outros provedores
# OPENAI_API_KEY="sk-..."
# GROQ_API_KEY="gsk_..."
```

> Para obter uma chave de API do Gemini, acesse o Google AI Studio.

**5. Crie a Pasta para Gráficos**

Crie uma pasta chamada `temp_charts` na raiz do projeto. O agente salvará os gráficos gerados aqui.

```bash
mkdir temp_charts
```

**6. Execute a Aplicação**

Com o ambiente virtual ativado, execute o seguinte comando:
```bash
streamlit run main.py
```

A aplicação será aberta no seu navegador padrão.