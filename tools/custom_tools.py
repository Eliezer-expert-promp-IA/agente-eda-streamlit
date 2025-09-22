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
    
    # Você pode adicionar descrições mais detalhadas para as ferramentas se necessário
    python_repl.description = """
    Executa código Python. Use esta ferramenta para explorar e analisar os dados.
    O DataFrame do pandas contendo os dados do usuário já está carregado na variável 'df'.
    Você pode executar qualquer código pandas em 'df', como 'print(df.head())' ou 'print(df.describe())'.
    Sempre use 'print()' para retornar os resultados da sua análise como uma observação.
    """
    
    return [python_repl]