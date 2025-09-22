# tools/custom_tools.py

# Importa as bibliotecas e módulos necessários
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool

# A ferramenta Python REPL é a única necessária para a análise de dados,
# desde que o DataFrame seja injetado em seu escopo.

def criar_ferramentas_analise(df: pd.DataFrame):
    """
    Função auxiliar que cria as ferramentas para o agente.
    A principal ferramenta é um Python REPL que já tem o DataFrame
    carregado na variável 'df'.

    Args:
        df: O DataFrame do Pandas a ser analisado.

    Returns:
        Uma lista contendo as ferramentas configuradas.
    """
    # Define o escopo local para o REPL, disponibilizando o DataFrame 'df'
    python_repl = PythonAstREPLTool(locals={"df": df})

    # Damos um nome e uma descrição clara para a ferramenta, instruindo o LLM a usá-la.
    python_repl.name = "python_code_executor"
    python_repl.description = """Executa código Python para explorar e analisar os dados em um DataFrame do pandas.
Use esta ferramenta para responder a QUALQUER pergunta sobre os dados.
O DataFrame do pandas já está carregado e disponível na variável `df`.
O código que você escreve DEVE usar a função `print()` para que o resultado seja retornado como uma observação.
Exemplo de uso: print(df.describe())
    """

    return [python_repl]