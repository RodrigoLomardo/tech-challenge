"""Estatísticas descritivas e diagnósticos da exploração de dados.

As funções aqui apenas *calculam*; a renderização fica em :mod:`medai.viz` e a
narrativa no notebook ``01_exploracao_dados.ipynb``. Essa separação é o que
permite reaproveitar os mesmos números no relatório e nos testes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from medai.config import Config


def resumo_estrutural(df: pd.DataFrame) -> dict[str, object]:
    """Caracterização de alto nível da base: forma, tipos, memória, nulos."""
    return {
        "linhas": len(df),
        "colunas": df.shape[1],
        "memoria_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "tipos": df.dtypes.value_counts().to_dict(),
        "colunas_com_nulos": int((df.isna().sum() > 0).sum()),
        "total_nulos": int(df.isna().sum().sum()),
        "linhas_duplicadas": int(df.duplicated().sum()),
    }


def distribuicao_classes(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Contagem e proporção por classe, com a positiva identificada."""
    alvo = config.dataset.coluna_alvo
    contagem = df[alvo].value_counts()
    return pd.DataFrame(
        {
            "contagem": contagem,
            "proporcao": (contagem / len(df)).round(4),
            "e_positiva": contagem.index == config.dataset.valor_positivo,
        }
    )


def estatisticas_descritivas(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """``describe()`` acrescido de assimetria e curtose.

    Assimetria e curtose não aparecem no ``describe()`` padrão, mas são o que
    sustenta a discussão sobre escalonamento: uma feature com skew 5 tem cauda
    longa o bastante para que ``StandardScaler`` não a torne simétrica, e
    modelos sensíveis a escala sofrem com isso.
    """
    base = df[features].describe().T
    base["skew"] = df[features].skew()
    base["kurtosis"] = df[features].kurtosis()
    # Coeficiente de variação: dispersão comparável entre features de
    # unidades diferentes (área em px² vs. suavidade adimensional).
    base["cv"] = base["std"] / base["mean"].replace(0, np.nan)
    return base


def separabilidade_cohen_d(df: pd.DataFrame, config: Config, features: list[str]) -> pd.Series:
    """``d`` de Cohen entre as classes, por feature.

    Diferença das médias em unidades de desvio padrão combinado. Mede quanto
    cada feature separa maligno de benigno isoladamente, antes de qualquer
    modelo. Convenção: |d| > 0,8 é efeito grande; perto de 0, não separa.
    """
    alvo = config.dataset.coluna_alvo
    positivos = df[df[alvo] == config.dataset.valor_positivo][features]
    negativos = df[df[alvo] != config.dataset.valor_positivo][features]

    desvio_combinado = np.sqrt((positivos.var() + negativos.var()) / 2)
    d = (positivos.mean() - negativos.mean()) / desvio_combinado
    return d.sort_values(ascending=False).rename("cohen_d")


def detectar_outliers(df: pd.DataFrame, features: list[str], z_limite: float = 3.0) -> pd.DataFrame:
    """Conta outliers por feature pelos critérios IQR e z-score.

    **Detectar não é remover.** Em dados clínicos o valor extremo costuma ser
    justamente o caso patológico de interesse: um tumor com área muito acima
    do normal é o sinal que o modelo precisa aprender, não ruído a descartar.
    Esta função existe para quantificar e discutir, conforme ADR-007.
    """
    q1 = df[features].quantile(0.25)
    q3 = df[features].quantile(0.75)
    iqr = q3 - q1
    fora_iqr = ((df[features] < q1 - 1.5 * iqr) | (df[features] > q3 + 1.5 * iqr)).sum()

    z = (df[features] - df[features].mean()) / df[features].std()
    fora_z = (z.abs() > z_limite).sum()

    resultado = pd.DataFrame(
        {
            "outliers_iqr": fora_iqr,
            "pct_iqr": (fora_iqr / len(df) * 100).round(2),
            f"outliers_z{z_limite:g}": fora_z,
            "maximo": df[features].max(),
            "p99": df[features].quantile(0.99),
        }
    )
    return resultado.sort_values("outliers_iqr", ascending=False)


def proporcao_outliers_por_classe(
    df: pd.DataFrame, config: Config, features: list[str]
) -> pd.DataFrame:
    """Em qual classe se concentram os outliers.

    É o argumento empírico a favor de mantê-los: se os valores extremos se
    concentram na classe maligna, removê-los apagaria sinal diagnóstico.
    """
    alvo = config.dataset.coluna_alvo
    q1, q3 = df[features].quantile(0.25), df[features].quantile(0.75)
    iqr = q3 - q1
    marcador = (df[features] < q1 - 1.5 * iqr) | (df[features] > q3 + 1.5 * iqr)

    # Uma linha é "atípica" se tem ao menos uma feature fora da cerca.
    linhas_atipicas = marcador.any(axis=1)
    tabela = pd.crosstab(df[alvo], linhas_atipicas)
    tabela.columns = ["tipica", "atipica"]
    tabela["pct_atipica"] = (tabela["atipica"] / tabela.sum(axis=1) * 100).round(1)
    return tabela
