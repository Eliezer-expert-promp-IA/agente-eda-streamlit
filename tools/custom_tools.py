# tools/custom_tools.py

import pandas as pd
from langchain.tools import tool
from io import StringIO
import sys
import os
import matplotlib
# Define o backend do Matplotlib como 'Agg' para evitar que ele tente abrir uma GUI.
# Isso é crucial para rodar em ambientes de servidor/não-interativos como o Streamlit.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Define um escopo persistente para a execução do código
_persistent_scope = {}

@tool
def python_code_executor(code: str) -> str:
    """
    Executa código Python para explorar e analisar dados em um DataFrame pandas.
    O DataFrame está na variável `df`. Use `print()` para retornar resultados.
    Exemplo: print(df.head())
    """
    global _persistent_scope

    # Limpa o estado de qualquer gráfico anterior para evitar sobreposição de plots
    if 'plt' in _persistent_scope:
        _persistent_scope['plt'].clf()

    # Limpa o código de entrada para remover possíveis formatações de markdown
    # que o LLM possa ter incluído, causando um SyntaxError.
    cleaned_code = code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[9:]
    elif cleaned_code.startswith("```"):
        cleaned_code = cleaned_code[3:]

    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-3]

    cleaned_code = cleaned_code.strip()

    try:
        # Redireciona a saída padrão (stdout) para capturar os prints
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        # Executa o código no escopo persistente
        exec(cleaned_code, _persistent_scope)

        # Restaura a saída padrão
        sys.stdout = old_stdout
        output = captured_output.getvalue()

        if output:
            return f"Execução bem-sucedida. Saída:\n```\n{output}\n```"
        else:
            return "O código foi executado com sucesso, mas não produziu nenhuma saída. Use a função `print()` para ver os resultados."

    except Exception:
        # Restaura a saída padrão em caso de erro
        sys.stdout = old_stdout
        # Captura o traceback completo do erro e o retorna como uma string formatada
        # Isso é crucial para o agente entender o erro sem quebrar
        error_trace = traceback.format_exc()
        return f"Erro ao executar o código. Detalhes:\n```\n{error_trace}\n```"

def criar_ferramentas_analise(df: pd.DataFrame):
    """
    Função auxiliar que cria e configura as ferramentas para o agente.
    """
    # Inicializa o escopo com o DataFrame e o pandas
    global _persistent_scope
    _persistent_scope = {
        'df': df,
        'pd': pd,
        'plt': plt,
        'sns': sns
    }
    # Garante que o diretório para salvar os gráficos exista.
    # É uma boa prática ter a ferramenta responsável por suas próprias dependências de diretório.
    if not os.path.exists("temp_charts"):
        os.makedirs("temp_charts")

    return [python_code_executor]