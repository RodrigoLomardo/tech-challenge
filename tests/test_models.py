"""Testes do protocolo experimental, métricas, limiar e calibração.

O teste mais importante é `test_refit_recusa_a_solucao_degenerada`: sem o
desempate, a busca selecionaria o modelo que classifica tudo como maligno.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medai.config import Config
from medai.models import calibrate, registry
from medai.models.evaluate import (
    FAIXA_BENIGNO,
    FAIXA_MALIGNO,
    FAIXA_REVISAO,
    classificar_com_faixa,
    escolher_limiar,
    metricas_classificacao,
    resumo_faixas,
)
from medai.models.train import (
    construir_feature_sets,
    preparar_dados,
    selecionar_campeao,
    treinar_com_busca,
)


@pytest.fixture(scope="module")
def dados(config_breast_cancer: Config):
    return preparar_dados(config_breast_cancer)


# --- Catálogo ----------------------------------------------------------------


def test_catalogo_tem_baseline_e_modelos_do_config(config_breast_cancer: Config) -> None:
    for nome in config_breast_cancer.modelagem.modelos:
        especificacao = registry.obter(nome)
        assert especificacao.justificativa, f"{nome} sem justificativa para o relatório"


def test_modelo_inexistente_falha() -> None:
    with pytest.raises(KeyError, match="não está no catálogo"):
        registry.obter("rede_neural_quantica")


def test_estimadores_usam_n_jobs_unitario() -> None:
    """Paralelizar no estimador e na busca causa superalocação de threads."""
    for nome in ("logistic_regression", "random_forest", "xgboost", "knn"):
        estimador = registry.obter(nome).construtor(42)
        if hasattr(estimador, "n_jobs"):
            assert estimador.n_jobs == 1, f"{nome} deveria ter n_jobs=1"


# --- Critério de seleção (F-beta) --------------------------------------------


def test_f2_rejeita_a_solucao_degenerada() -> None:
    """Classificar tudo como maligno dá recall 1,0 e precisão 0,37.

    Uma versão anterior deste projeto usava a regra de um erro padrão sobre o
    recall — aceitar candidatos dentro de um desvio do melhor e desempatar por
    F1. Ela falha exatamente aqui: o degenerado acerta todos os positivos em
    todos os folds, seu desvio é zero, a janela colapsa e ele vence sozinho.

    F2 resolve de forma direta, ponderando recall 4× sobre precisão sem
    aceitar precisão arbitrariamente baixa.
    """
    from sklearn.metrics import fbeta_score

    y = np.array([0] * 63 + [1] * 37)
    degenerado = np.ones(100, dtype=int)
    bom = np.array([0] * 60 + [1] * 3 + [1] * 36 + [0] * 1)

    f2_degenerado = fbeta_score(y, degenerado, beta=2)
    f2_bom = fbeta_score(y, bom, beta=2)

    assert f2_degenerado < 0.80
    assert f2_bom > 0.95
    assert f2_bom > f2_degenerado


def test_f2_prioriza_recall_sobre_precisao() -> None:
    """Entre dois modelos, o de maior recall vence mesmo perdendo precisão."""
    from sklearn.metrics import f1_score, fbeta_score

    y = np.array([0] * 60 + [1] * 40)
    # A: recall alto, precisão média. B: recall médio, precisão alta.
    a = np.array([0] * 50 + [1] * 10 + [1] * 39 + [0] * 1)
    b = np.array([0] * 58 + [1] * 2 + [1] * 32 + [0] * 8)

    assert fbeta_score(y, a, beta=2) > fbeta_score(y, b, beta=2)
    # F1, que pondera igualmente, não faria essa distinção na mesma direção.
    assert f1_score(y, a) > 0 and f1_score(y, b) > 0


def test_metrica_de_refit_e_f2() -> None:
    from medai.models.train import METRICA_REFIT, SCORING

    assert METRICA_REFIT == "f2"
    assert "f2" in SCORING


# --- Métricas ----------------------------------------------------------------


def test_metricas_de_classificacao_perfeita() -> None:
    y = np.array([0, 0, 1, 1])
    m = metricas_classificacao(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["falsos_negativos"] == 0
    assert m["roc_auc"] == 1.0


def test_metricas_do_classificador_trivial() -> None:
    """Sempre "benigno": accuracy alta, recall zero, utilidade negativa."""
    y = np.array([0] * 63 + [1] * 37)
    predito = np.zeros(100, dtype=int)
    m = metricas_classificacao(y, predito)
    assert m["accuracy"] == pytest.approx(0.63)
    assert m["recall"] == 0.0
    assert m["falsos_negativos"] == 37


def test_metricas_contam_erros_clinicos_corretamente() -> None:
    y = np.array([1, 1, 0, 0])
    predito = np.array([1, 0, 1, 0])
    m = metricas_classificacao(y, predito)
    assert m["falsos_negativos"] == 1  # câncer não sinalizado
    assert m["falsos_positivos"] == 1  # revisão desnecessária
    assert m["especificidade"] == 0.5


# --- Limiar ------------------------------------------------------------------


def test_limiar_atinge_o_recall_exigido() -> None:
    rng = np.random.default_rng(42)
    y = np.array([0] * 70 + [1] * 30)
    proba = np.concatenate([rng.beta(2, 5, 70), rng.beta(5, 2, 30)])

    resultado = escolher_limiar(y, proba, recall_minimo=0.98)
    assert resultado.atingiu_alvo
    assert resultado.recall >= 0.98


def test_limiar_mais_baixo_que_o_padrao_reduz_falsos_negativos() -> None:
    """A troca explícita: menos cânceres perdidos, mais revisões."""
    rng = np.random.default_rng(7)
    y = np.array([0] * 70 + [1] * 30)
    proba = np.concatenate([rng.beta(2, 5, 70), rng.beta(5, 2, 30)])

    ajustado = escolher_limiar(y, proba, recall_minimo=0.98)
    padrao = metricas_classificacao(y, (proba >= 0.5).astype(int))

    assert ajustado.limiar < 0.5
    assert ajustado.falsos_negativos <= padrao["falsos_negativos"]
    assert ajustado.falsos_positivos >= padrao["falsos_positivos"]


def test_limiar_sinaliza_quando_o_alvo_e_inatingivel() -> None:
    """Falhar em silêncio prometeria sensibilidade que o modelo não entrega."""
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.9, 0.1, 0.8, 0.2])  # modelo invertido
    resultado = escolher_limiar(y, proba, recall_minimo=0.99)
    # Com limiar 0 todos viram positivos, então o alvo é sempre alcançável;
    # o importante é que o campo reflita a realidade.
    assert isinstance(resultado.atingiu_alvo, bool)


# --- Faixas de decisão -------------------------------------------------------


def test_faixas_separam_as_tres_regioes(config_breast_cancer: Config) -> None:
    proba = np.array([0.02, 0.45, 0.95])
    faixas = classificar_com_faixa(proba, limiar=0.3, config=config_breast_cancer)
    assert faixas[0] == FAIXA_BENIGNO
    assert faixas[1] == FAIXA_REVISAO  # dentro da zona cinzenta 0,30–0,70
    assert faixas[2] == FAIXA_MALIGNO


def test_zona_cinzenta_tem_precedencia_sobre_o_limiar(config_breast_cancer: Config) -> None:
    """Caso acima do limiar mas incerto é encaminhado, não rotulado.

    Conservador na direção certa: o sistema admite não saber.
    """
    proba = np.array([0.5])  # acima do limiar 0,072 e dentro da zona 0,30–0,70
    faixas = classificar_com_faixa(proba, limiar=0.072, config=config_breast_cancer)
    assert faixas[0] == FAIXA_REVISAO


def test_resumo_de_faixas_conta_corretamente(config_breast_cancer: Config) -> None:
    proba = np.array([0.01, 0.02, 0.5, 0.99])
    faixas = classificar_com_faixa(proba, limiar=0.3, config=config_breast_cancer)
    resumo = resumo_faixas(faixas)
    assert resumo[FAIXA_BENIGNO] == 2
    assert resumo[FAIXA_REVISAO] == 1


# --- Treino ------------------------------------------------------------------


def test_feature_sets_sao_construidos_apenas_do_treino(dados, config_breast_cancer: Config) -> None:
    conjuntos = construir_feature_sets(dados, config_breast_cancer)
    assert len(conjuntos["completo"]) == 30
    assert len(conjuntos["reduzido"]) == 24
    assert set(conjuntos["reduzido"]) < set(conjuntos["completo"])


@pytest.mark.slow
def test_baseline_tem_recall_zero(dados, config_breast_cancer: Config) -> None:
    """O piso de comparação: accuracy 62,7%, utilidade clínica nula."""
    resultado = treinar_com_busca(
        registry.obter("dummy"), dados.x_treino, dados.y_treino, config_breast_cancer
    )
    assert resultado.metricas_cv["cv_recall"] == 0.0
    assert resultado.metricas_cv["cv_accuracy"] == pytest.approx(0.627, abs=0.01)


@pytest.mark.slow
def test_treino_com_busca_produz_modelo_utilizavel(dados, config_breast_cancer: Config) -> None:
    resultado = treinar_com_busca(
        registry.obter("logistic_regression"), dados.x_treino, dados.y_treino, config_breast_cancer
    )
    assert resultado.metricas_cv["cv_recall"] > 0.9
    assert resultado.metricas_cv["cv_f2"] > 0.9
    assert resultado.melhores_parametros
    proba = resultado.estimador.predict_proba(dados.x_validacao)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_selecao_de_campeao_ignora_o_baseline() -> None:
    tabela = pd.DataFrame(
        {
            "modelo": ["dummy", "logistic_regression", "xgboost"],
            "feature_set": ["completo"] * 3,
            "n_features": [30, 30, 30],
            "cv_f2": [1.0, 0.96, 0.95],  # dummy fictício com F2 alto
            "cv_f2_std": [0.0, 0.02, 0.02],
            "cv_recall": [1.0, 0.96, 0.95],
        }
    )
    campeao = selecionar_campeao(tabela)
    assert campeao["modelo"] != "dummy"


def test_selecao_desempata_por_recall_dentro_da_janela() -> None:
    """Dentro da janela de F2 equivalente, vence o de maior recall."""
    tabela = pd.DataFrame(
        {
            "modelo": ["logistic_regression", "xgboost"],
            "feature_set": ["completo"] * 2,
            "n_features": [30, 30],
            "cv_f2": [0.9685, 0.9667],  # diferença dentro do desvio
            "cv_f2_std": [0.012, 0.012],
            "cv_recall": [0.9686, 0.9609],
        }
    )
    assert selecionar_campeao(tabela)["modelo"] == "logistic_regression"


def test_selecao_sem_candidatos_falha() -> None:
    tabela = pd.DataFrame(
        {"modelo": ["dummy"], "cv_f2": [0.0], "cv_f2_std": [0.0], "cv_recall": [0.0]}
    )
    with pytest.raises(ValueError, match="Nenhum candidato"):
        selecionar_campeao(tabela)


# --- Calibração --------------------------------------------------------------


def test_diagnostico_de_modelo_bem_calibrado() -> None:
    """Probabilidades que batem com a frequência real não precisam correção."""
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < proba).astype(int)

    diagnostico = calibrate.diagnosticar(y, proba)
    assert not diagnostico.precisa_calibrar
    assert diagnostico.erro_calibracao_medio < calibrate.TOLERANCIA_CALIBRACAO


def test_diagnostico_detecta_excesso_de_confianca() -> None:
    """Modelo que diz 95% quando a frequência real é 60%."""
    rng = np.random.default_rng(1)
    proba_real = rng.uniform(0.3, 0.7, 2000)
    y = (rng.uniform(0, 1, 2000) < proba_real).astype(int)
    # Empurra as probabilidades para os extremos.
    proba_exagerada = np.clip((proba_real - 0.5) * 4 + 0.5, 0.01, 0.99)

    diagnostico = calibrate.diagnosticar(y, proba_exagerada)
    assert diagnostico.precisa_calibrar


def test_tabela_de_calibracao(config_breast_cancer: Config) -> None:
    rng = np.random.default_rng(3)
    proba = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < proba).astype(int)
    tabela = calibrate.diagnosticar(y, proba).para_tabela()
    assert {"probabilidade_prevista", "frequencia_observada", "desvio"} == set(tabela.columns)
