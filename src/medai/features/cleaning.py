"""Limpeza do dataset bruto e preparação do alvo.

A limpeza aqui é **anterior ao split** e deliberadamente restrita a operações
que não aprendem nada dos dados: descartar colunas, mapear o rótulo, remover
duplicatas, converter zeros impossíveis em ausentes.

Tudo que *estima* algo a partir dos dados — imputação, escalonamento,
codificação — fica no pipeline do scikit-learn (:mod:`medai.features.build_pipeline`),
que é ajustado somente no conjunto de treino. Essa fronteira é o que impede
vazamento, e é fácil de borrar sem perceber: imputar a mediana antes do split
já contamina o teste com informação do conjunto inteiro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class RelatorioLimpeza:
    """O que a limpeza fez, para registro no relatório e no notebook."""

    colunas_descartadas: list[str] = field(default_factory=list)
    colunas_inexistentes: list[str] = field(default_factory=list)
    duplicatas_removidas: int = 0
    zeros_convertidos: dict[str, int] = field(default_factory=dict)
    linhas_antes: int = 0
    linhas_depois: int = 0
    nulos_apos_limpeza: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        linhas = ["Limpeza:"]
        linhas.append(f"  linhas: {self.linhas_antes} → {self.linhas_depois}")
        if self.colunas_descartadas:
            linhas.append(f"  colunas descartadas: {self.colunas_descartadas}")
        if self.colunas_inexistentes:
            linhas.append(
                f"  descartes não aplicáveis (ausentes na fonte): {self.colunas_inexistentes}"
            )
        if self.duplicatas_removidas:
            linhas.append(f"  duplicatas removidas: {self.duplicatas_removidas}")
        if self.zeros_convertidos:
            linhas.append(f"  zeros impossíveis → NaN: {self.zeros_convertidos}")
        total_nulos = sum(self.nulos_apos_limpeza.values())
        linhas.append(f"  nulos a imputar no pipeline: {total_nulos}")
        return "\n".join(linhas)


def limpar(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, RelatorioLimpeza]:
    """Aplica as regras de limpeza configuradas.

    Returns:
        O dataset limpo e o relatório do que foi feito.
    """
    relatorio = RelatorioLimpeza(linhas_antes=len(df))
    df = df.copy()

    # --- Descarte de colunas -----------------------------------------------
    # `id` jamais pode virar feature: não tem significado clínico e, se
    # correlacionar com o alvo por ordem de coleta, o modelo aprende o
    # vazamento em vez da patologia.
    for coluna in config.dataset.colunas_descartar:
        if coluna in df.columns:
            df = df.drop(columns=[coluna])
            relatorio.colunas_descartadas.append(coluna)
        else:
            # A rota sklearn não tem 'id' nem 'Unnamed: 32'. Ausência é
            # esperada, não erro.
            relatorio.colunas_inexistentes.append(coluna)

    # --- Zeros fisiologicamente impossíveis --------------------------------
    # Em bases como o Pima, Glucose=0 ou BMI=0 não são medições: são ausentes
    # gravados como zero. Tratá-los como valores reais distorce a mediana e
    # ensina ao modelo uma fisiologia inexistente.
    for coluna in config.limpeza.zeros_impossiveis:
        if coluna not in df.columns:
            continue
        n_zeros = int((df[coluna] == 0).sum())
        if n_zeros:
            df[coluna] = df[coluna].replace(0, np.nan)
            relatorio.zeros_convertidos[coluna] = n_zeros

    # --- Duplicatas ---------------------------------------------------------
    if config.limpeza.remover_duplicatas:
        antes = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        relatorio.duplicatas_removidas = antes - len(df)

    # --- Outliers: deliberadamente intocados -------------------------------
    if config.limpeza.remover_outliers:  # pragma: no cover - desligado por ADR-007
        raise NotImplementedError(
            "Remoção de outliers não é implementada por decisão de projeto (ADR-007): "
            "53,8% dos casos malignos são atípicos pelo critério IQR, contra 16,0% dos "
            "benignos. Removê-los descartaria preferencialmente a classe de interesse."
        )

    relatorio.linhas_depois = len(df)
    nulos = df.isna().sum()
    relatorio.nulos_apos_limpeza = {k: int(v) for k, v in nulos[nulos > 0].items()}

    logger.info(
        "Limpeza concluída",
        extra={
            "linhas": relatorio.linhas_depois,
            "descartadas": relatorio.colunas_descartadas,
            "duplicatas": relatorio.duplicatas_removidas,
        },
    )
    return df, relatorio


def separar_alvo(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo, codificando o alvo como 0/1.

    **A classe positiva (1) é a de interesse clínico** — maligno, no dataset
    principal. Inverter isso inverte a leitura de precision e recall sem gerar
    nenhum erro visível.
    """
    alvo = config.dataset.coluna_alvo
    if alvo not in df.columns:
        raise KeyError(f"Coluna alvo '{alvo}' ausente do dataset")

    y = (df[alvo] == config.dataset.valor_positivo).astype(int)
    y.name = alvo

    if y.nunique() != 2:
        raise ValueError(
            f"Alvo não ficou binário após codificação. Valores originais: "
            f"{sorted(df[alvo].unique(), key=str)}; classe positiva configurada: "
            f"'{config.dataset.valor_positivo}'"
        )

    x = df.drop(columns=[alvo])
    logger.info(
        "Alvo separado",
        extra={"n_features": x.shape[1], "positivos": int(y.sum()), "taxa": round(y.mean(), 4)},
    )
    return x, y
