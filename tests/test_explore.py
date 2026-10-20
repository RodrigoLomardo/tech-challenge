"""Testes das estatísticas descritivas e diagnósticos da EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medai.config import Config
from medai.data import explore
from medai.data.loader import FEATURES_CANONICAS, carregar_breast_cancer_sklearn


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return carregar_breast_cancer_sklearn()


@pytest.fixture(scope="module")
def features() -> list[str]:
    return list(FEATURES_CANONICAS)


def test_resumo_estrutural(df: pd.DataFrame) -> None:
    resumo = explore.resumo_estrutural(df)
    assert resumo["linhas"] == 569
    assert resumo["colunas"] == 31
    assert resumo["total_nulos"] == 0
    assert resumo["linhas_duplicadas"] == 0


def test_distribuicao_classes_marca_a_positiva(
    df: pd.DataFrame, config_breast_cancer: Config
) -> None:
    dist = explore.distribuicao_classes(df, config_breast_cancer)
    assert dist.loc["M", "contagem"] == 212
    assert dist.loc["M", "e_positiva"]
    assert not dist.loc["B", "e_positiva"]
    assert dist["proporcao"].sum() == pytest.approx(1.0)


def test_descritivas_incluem_skew_e_kurtosis(df: pd.DataFrame, features: list[str]) -> None:
    desc = explore.estatisticas_descritivas(df, features)
    assert {"skew", "kurtosis", "cv"} <= set(desc.columns)
    assert len(desc) == 30
    # area_se é a feature mais assimétrica do dataset.
    assert desc.loc["area_se", "skew"] > 5


def test_cohen_d_ordenado_e_coerente_com_o_dominio(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    d = explore.separabilidade_cohen_d(df, config_breast_cancer, features)
    assert d.is_monotonic_decreasing
    # concave points_worst é a feature mais discriminante conhecida.
    assert d.index[0] == "concave points_worst"
    assert d.iloc[0] > 2.0


def test_cohen_d_e_zero_para_feature_sem_sinal(config_breast_cancer: Config) -> None:
    """Feature idêntica nas duas classes deve ter d ≈ 0."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "diagnosis": ["M"] * 100 + ["B"] * 100,
            "ruido": rng.normal(size=200),
        }
    )
    d = explore.separabilidade_cohen_d(df, config_breast_cancer, ["ruido"])
    assert abs(d["ruido"]) < 0.4


def test_cohen_d_detecta_direcao(config_breast_cancer: Config) -> None:
    """d positivo significa valor maior na classe positiva."""
    df = pd.DataFrame(
        {
            "diagnosis": ["M"] * 50 + ["B"] * 50,
            "maior_em_maligno": [10.0] * 50 + [1.0] * 50,
        }
    )
    d = explore.separabilidade_cohen_d(df, config_breast_cancer, ["maior_em_maligno"])
    assert d["maior_em_maligno"] > 0


def test_deteccao_de_outliers(df: pd.DataFrame, features: list[str]) -> None:
    out = explore.detectar_outliers(df, features)
    assert len(out) == 30
    assert out["outliers_iqr"].is_monotonic_decreasing
    # area_se concentra o maior número de valores fora da cerca.
    assert out.index[0] == "area_se"
    assert (out["pct_iqr"] <= 100).all()


def test_outliers_concentram_se_na_classe_maligna(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    """Evidência empírica do ADR-007: remover outliers apagaria sinal.

    Se esta relação se inverter em algum refactor, a decisão de manter os
    valores atípicos perde sua justificativa e precisa ser reavaliada.
    """
    tabela = explore.proporcao_outliers_por_classe(df, config_breast_cancer, features)
    assert tabela.loc["M", "pct_atipica"] > tabela.loc["B", "pct_atipica"]
    assert tabela.loc["M", "pct_atipica"] > 50
    assert tabela.loc["B", "pct_atipica"] < 20
