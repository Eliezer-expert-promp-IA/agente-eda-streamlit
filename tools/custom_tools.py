# tools/custom_tools.py

import pandas as pd
from langchain.tools import tool
from io import StringIO
import sys

# Contexto de execução para o código Python
_globals = {}
_locals = {}

@tool
def python_code_executor(code: str) -> str:
    """
    Executa código Python para explorar e analisar os dados em um DataFrame do pandas.
    Use esta ferramenta para responder a QUALQUER pergunta sobre os dados.
    O DataFrame do pandas já está carregado e disponível na variável `df`.
    O código que você escreve DEVE usar a função `print()` para que o resultado seja retornado como uma observação.
    Exemplo de uso: print(df.describe())
    """
    try:
        # Redireciona a saída padrão (stdout) para capturar os prints
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Executa o código enviado pelo agente
        exec(code, _globals, _locals)

        # Restaura a saída padrão
        sys.stdout = old_stdout

        # Pega o que foi capturado no 'print'
        output = captured_output.getvalue()
        if not output:
            return "O código foi executado com sucesso, mas não produziu nenhuma saída. Lembre-se de usar a função `print()`."
        return output

    except Exception as e:
        # Restaura a saída padrão em caso de erro
        sys.stdout = old_stdout
        # Retorna a mensagem de erro como observação para o agente
        return f"Erro ao executar o código: {str(e)}"

def criar_ferramentas_analise(df: pd.DataFrame):
    """
    Função auxiliar que cria e configura as ferramentas para o agente.
    """
    # Disponibiliza o DataFrame no escopo local da ferramenta
    _locals["df"] = df
    _locals["pd"] = pd
    return [python_code_executor]