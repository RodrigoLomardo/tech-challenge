"""Testes de interpretabilidade e robustez."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from medai.config import Config
from medai.features.build_pipeline import construir_pipeline
from medai.models import explain, robustez
from medai.models.train import preparar_dados


@pytest.fixture(scope="module")
def dados(config_breast_cancer: Config):
    return preparar_dados(config_breast_cancer)


@pytest.fixture(scope="module")
def pipeline_logistico(dados, config_breast_cancer: Config):
    x, y = dados.treino_mais_validacao()
    pipe = construir_pipeline(
        LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced"),
        x,
        config_breast_cancer,
    )
    pipe.fit(x, y)
    return pipe


# --- Coeficientes ------------------------------------------------------------


def test_coeficientes_cobrem_todas_as_features(pipeline_logistico) -> None:
    tabela = explain.coeficientes(pipeline_logistico)
    assert len(tabela) == 30
    assert tabela["abs_coeficiente"].is_monotonic_decreasing


def test_razao_de_chances_coerente_com_o_sinal(pipeline_logistico) -> None:
    tabela = explain.coeficientes(pipeline_logistico)
    positivos = tabela[tabela["coeficiente"] > 0]
    negativos = tabela[tabela["coeficiente"] < 0]
    assert (positivos["razao_de_chances"] > 1).all()
    assert (negativos["razao_de_chances"] < 1).all()


def test_modelo_sem_coeficientes_falha(dados, config_breast_cancer: Config) -> None:
    x, y = dados.x_treino, dados.y_treino
    pipe = construir_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42), x, config_breast_cancer
    )
    pipe.fit(x, y)
    with pytest.raises(TypeError, match="coeficientes"):
        explain.coeficientes(pipe)


def test_multicolinearidade_inverte_sinais(
    pipeline_logistico, dados, config_breast_cancer: Config
) -> None:
    """Com VIF acima de 3.900, coeficientes contradizem a associação univariada.

    Este teste documenta um comportamento conhecido e indesejado, não uma
    propriedade desejável: é o que fundamenta a decisão de não exibir
    coeficientes ao usuário clínico.
    """
    from medai.data import explore

    x, y = dados.treino_mais_validacao()
    df = x.copy()
    df["diagnosis"] = y.map({1: "M", 0: "B"})
    d = explore.separabilidade_cohen_d(df, config_breast_cancer, x.columns.tolist())

    coef = explain.coeficientes(pipeline_logistico)["coeficiente"]
    discordantes = ((coef * d) < 0).sum()
    assert discordantes > 0, (
        "Sem sinais invertidos, a justificativa para esconder os coeficientes "
        "da UI precisaria ser reavaliada"
    )


# --- Permutation importance --------------------------------------------------


@pytest.mark.slow
def test_permutation_importance(pipeline_logistico, dados) -> None:
    tabela = explain.importancia_por_permutacao(
        pipeline_logistico, dados.x_teste, dados.y_teste, n_repeticoes=5
    )
    assert len(tabela) == 30
    assert tabela["queda_media"].is_monotonic_decreasing
    assert {"desvio", "significativa"} <= set(tabela.columns)


# --- SHAP --------------------------------------------------------------------


@pytest.fixture(scope="module")
def explicacao(pipeline_logistico, dados):
    x_fundo, _ = dados.treino_mais_validacao()
    return explain.explicar(pipeline_logistico, dados.x_teste, x_fundo=x_fundo)


def test_shap_tem_a_forma_esperada(explicacao, dados) -> None:
    assert explicacao.valores.shape == (len(dados.x_teste), 30)
    assert len(explicacao.nomes) == 30


def test_shap_soma_ao_log_odds_predito(explicacao, pipeline_logistico, dados) -> None:
    """Propriedade da aditividade: base + contribuições = saída do modelo.

    Se isso quebrar, as explicações mostradas ao clínico não correspondem ao
    que o modelo de fato calculou.
    """
    proba = pipeline_logistico.predict_proba(dados.x_teste)[:, 1]
    log_odds = np.log(proba / (1 - proba))
    reconstruido = explicacao.valor_base + explicacao.valores.sum(axis=1)
    np.testing.assert_allclose(reconstruido, log_odds, atol=1e-6)


def test_importancia_global_ordenada(explicacao) -> None:
    importancia = explicacao.importancia_global
    assert len(importancia) == 30
    assert importancia.is_monotonic_decreasing
    assert (importancia >= 0).all()


def test_contribuicoes_de_caso_individual(explicacao) -> None:
    tabela = explicacao.contribuicoes_do_caso(0, top=5)
    assert len(tabela) == 5
    assert {"contribuicao", "valor_medido", "efeito"} <= set(tabela.columns)
    # Ordenado por magnitude absoluta.
    assert tabela["contribuicao"].abs().is_monotonic_decreasing


def test_agrupamento_corrige_a_diluicao(explicacao) -> None:
    """A leitura individual subestima grupos de features redundantes.

    Somar por grandeza deve mostrar que 'tamanho' — nove features
    correlacionadas — supera a feature individual de maior peso.
    """
    individual = explicacao.importancia_global
    agrupado = explain.agrupar_por_medida(individual)

    assert agrupado.is_monotonic_decreasing
    assert agrupado.index[0].startswith("tamanho")
    # O grupo supera com folga qualquer feature isolada.
    assert agrupado.iloc[0] > individual.iloc[0] * 2


def test_agrupamento_preserva_a_soma_total(explicacao) -> None:
    individual = explicacao.importancia_global
    agrupado = explain.agrupar_por_medida(individual)
    assert agrupado.sum() == pytest.approx(individual.sum(), rel=1e-6)


# --- Robustez ----------------------------------------------------------------


@pytest.mark.slow
def test_robustez_sob_multiplas_sementes(config_breast_cancer: Config, dados) -> None:
    tabela = robustez.avaliar_sob_sementes(
        config_breast_cancer,
        "logistic_regression",
        {"modelo__C": 0.5, "modelo__class_weight": "balanced"},
        dados.x_treino.columns.tolist(),
        sementes=[42, 7, 13],
    )
    assert len(tabela) == 3
    assert (tabela["recall"] > 0.9).all()
    # Sementes diferentes produzem conjuntos de teste diferentes.
    assert tabela["recall"].nunique() > 1


def test_resumo_de_robustez() -> None:
    tabela = pd.DataFrame(
        {"recall": [0.95, 0.98, 1.00], "f1": [0.90, 0.92, 0.94]}, index=[42, 7, 13]
    )
    resumo = robustez.resumir(tabela, metricas=["recall", "f1"])
    assert resumo.loc["recall", "minimo"] == 0.95
    assert resumo.loc["recall", "maximo"] == 1.00
    assert resumo.loc["recall", "amplitude"] == pytest.approx(0.05)
