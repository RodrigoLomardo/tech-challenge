"""Testes das figuras.

Não verificamos aparência — isso é inspeção visual. Verificamos que cada
função produz uma figura válida, que a paleta respeita a classe positiva e
que os guardas contra uso indevido funcionam.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

from medai import viz
from medai.config import Config
from medai.data import explore
from medai.data.loader import FEATURES_CANONICAS, carregar_breast_cancer_sklearn


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return carregar_breast_cancer_sklearn()


@pytest.fixture(scope="module")
def features() -> list[str]:
    return list(FEATURES_CANONICAS)


@pytest.fixture(autouse=True)
def _fecha_figuras():
    yield
    plt.close("all")


def test_paleta_associa_laranja_a_classe_positiva(config_breast_cancer: Config) -> None:
    paleta = viz.paleta_classes(config_breast_cancer)
    assert paleta["M"] == viz.COR_MALIGNO
    assert paleta["B"] == viz.COR_BENIGNO


def test_distribuicao_classes_marca_o_classificador_trivial(
    df: pd.DataFrame, config_breast_cancer: Config
) -> None:
    fig = viz.plot_distribuicao_classes(df, config_breast_cancer)
    ax = fig.axes[0]
    # A linha tracejada do baseline é o argumento visual contra accuracy.
    legenda = ax.get_legend().get_texts()[0].get_text()
    assert "62.7%" in legenda


def test_grade_de_distribuicoes_cobre_todas_as_features(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    fig = viz.plot_grade_distribuicoes(df, config_breast_cancer, features, titulo="t")
    visiveis = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visiveis) >= len(features)


@pytest.mark.parametrize("funcao", ["plot_boxplots", "plot_violinos"])
def test_graficos_por_classe(
    funcao: str, df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    fig = getattr(viz, funcao)(df, config_breast_cancer, features[:4], titulo="t")
    assert len(fig.axes) >= 4


def test_pairplot_recusa_features_demais(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    """Um pairplot com 30 features geraria 900 painéis ilegíveis."""
    with pytest.raises(ValueError, match="ilegível"):
        viz.plot_pairplot(df, config_breast_cancer, features)


def test_pairplot_aceita_subconjunto(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    fig = viz.plot_pairplot(df, config_breast_cancer, features[:4])
    assert fig is not None


@pytest.mark.parametrize("metodo", ["pearson", "spearman"])
def test_heatmap_correlacao(metodo: str, df: pd.DataFrame, features: list[str]) -> None:
    fig = viz.plot_heatmap_correlacao(df, features, metodo=metodo)
    assert metodo in fig.axes[0].get_title().lower()


def test_pca_reporta_variancia_explicada(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    fig = viz.plot_pca_2d(df, config_breast_cancer, features)
    ax = fig.axes[0]
    assert "PC1" in ax.get_xlabel()
    assert "%" in ax.get_xlabel()


def test_salvar_figura_grava_arquivo(
    tmp_path, df: pd.DataFrame, config_breast_cancer: Config
) -> None:
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.figuras = tmp_path / "figs"
    fig = viz.plot_distribuicao_classes(df, cfg)
    caminho = viz.salvar_figura(fig, "teste", cfg)
    assert caminho.exists()
    assert caminho.stat().st_size > 1000


def test_separabilidade_e_outliers(
    df: pd.DataFrame, config_breast_cancer: Config, features: list[str]
) -> None:
    d = explore.separabilidade_cohen_d(df, config_breast_cancer, features)
    assert viz.plot_separabilidade(d) is not None

    tabela = explore.proporcao_outliers_por_classe(df, config_breast_cancer, features)
    assert viz.plot_outliers_por_classe(tabela, config_breast_cancer) is not None
