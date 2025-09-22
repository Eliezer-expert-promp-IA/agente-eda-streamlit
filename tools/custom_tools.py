# tools/custom_tools.py

import pandas as pd
from langchain.tools import tool
import io
import base64
import matplotlib.pyplot as plt
import traceback
import seaborn as sns

# Define o backend do matplotlib como 'Agg' para evitar que ele tente abrir uma janela de GUI no servidor
plt.switch_backend('Agg')

def criar_ferramentas_analise(df: pd.DataFrame):
    """
    Cria e retorna uma lista de ferramentas de análise de dados para o agente.
    As ferramentas são definidas dentro desta função para terem acesso ao DataFrame `df` via closure.
    """

    @tool
    def python_code_executor(code: str) -> str:
        """
        Executa código Python para realizar análise de dados em um DataFrame pandas.
        O DataFrame está disponível como `df`.
        O código DEVE usar `print()` para retornar qualquer resultado.
        Esta ferramenta deve ser usada para cálculos, manipulação de dados e inspeção.
        Não pode renderizar gráficos. Para gráficos, use a ferramenta 'chart_generator'.
        """
        local_vars = {"df": df, "pd": pd, "io": io, "sns": sns, "plt": plt}
        buffer = io.StringIO()
        
        try:
            import sys
            original_stdout = sys.stdout
            sys.stdout = buffer

            exec(code, {}, local_vars)

            sys.stdout = original_stdout
            output = buffer.getvalue()
            
            if not output:
                return "Código executado com sucesso, mas não produziu saída. Certifique-se de usar print() para ver os resultados."
            return output
        except Exception as e:
            if 'original_stdout' in locals() and original_stdout:
                sys.stdout = original_stdout
            return f"Erro ao executar o código: {e}\n{traceback.format_exc()}"

    @tool
    def chart_generator(code: str) -> str:
        """
        Executa código Python para gerar um único gráfico usando matplotlib/seaborn.
        O DataFrame está disponível como `df`. O código deve gerar um gráfico e
        esta ferramenta irá capturá-lo. NÃO chame `plt.show()` no código.
        Use esta ferramenta sempre que o usuário pedir um gráfico ou visualização.
        A saída será uma string de imagem codificada em base64 dentro de uma tag especial.
        Você deve incluir esta string diretamente em sua 'Resposta Final' após um texto descritivo,
        e ela DEVE ser a última parte da resposta.
        Exemplo de uso:
        Pensamento: O usuário quer um gráfico de barras. Vou usar a ferramenta chart_generator.
        Ação: chart_generator
        Entrada da Ação: import seaborn as sns\nimport matplotlib.pyplot as plt\nsns.barplot(data=df, x='categoria', y='valor')\nplt.title('Gráfico de Barras')
        Observação: [CHART_BASE64]...
        Pensamento: Eu tenho o gráfico. Agora vou apresentar a resposta final.
        Resposta Final: Aqui está o gráfico de barras que você solicitou:\n[CHART_BASE64]...
        """
        local_vars = {"df": df, "pd": pd, "plt": plt, "sns": sns, "io": io}
        
        # Cria uma nova figura para cada gráfico para evitar sobreposição
        fig = plt.figure()
        
        try:
            exec(code, {}, local_vars)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            plt.close(fig)

            return f"[CHART_BASE64]{image_base64}"
        except Exception as e:
            plt.close(fig)
            return f"Erro ao gerar o gráfico: {e}\n{traceback.format_exc()}"

    return [python_code_executor, chart_generator]