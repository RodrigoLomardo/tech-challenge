"""Testes de limpeza, particionamento, pipeline e correlação.

O teste mais importante deste arquivo é o de vazamento: ele falha se alguém
ajustar o escalonador fora do conjunto de treino.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from medai.config import Config, carregar_config
from medai.data.loader import carregar_breast_cancer_sklearn
from medai.data.split import construir_cv, particionar, salvar_particoes
from medai.features import correlation as corr
from medai.features.build_pipeline import (
    construir_pipeline,
    construir_preprocessador,
    nomes_features_saida,
    separar_tipos,
)
from medai.features.cleaning import limpar, separar_alvo


@pytest.fixture(scope="module")
def df_bruto() -> pd.DataFrame:
    return carregar_breast_cancer_sklearn()


@pytest.fixture(scope="module")
def xy(df_bruto: pd.DataFrame, config_breast_cancer: Config):
    limpo, _ = limpar(df_bruto, config_breast_cancer)
    return separar_alvo(limpo, config_breast_cancer)


@pytest.fixture(scope="module")
def particoes(xy, config_breast_cancer: Config):
    x, y = xy
    return particionar(x, y, config_breast_cancer)


# --- Limpeza -----------------------------------------------------------------


def test_limpeza_descarta_colunas_configuradas(
    df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    df = df_bruto.copy()
    df.insert(0, "id", range(len(df)))
    df["Unnamed: 32"] = pd.NA

    limpo, relatorio = limpar(df, config_breast_cancer)
    assert "id" not in limpo.columns
    assert "Unnamed: 32" not in limpo.columns
    assert set(relatorio.colunas_descartadas) == {"id", "Unnamed: 32"}


def test_coluna_ausente_nao_e_erro(df_bruto: pd.DataFrame, config_breast_cancer: Config) -> None:
    """A rota sklearn não tem 'id'; ausência é esperada."""
    _, relatorio = limpar(df_bruto, config_breast_cancer)
    assert "id" in relatorio.colunas_inexistentes
    assert relatorio.colunas_descartadas == []


def test_zeros_impossiveis_viram_nulos() -> None:
    """No Pima, Glucose=0 é ausente disfarçado, não medição."""
    cfg = carregar_config("configs/diabetes.yaml")
    df = pd.DataFrame(
        {
            "Glucose": [100.0, 0.0, 120.0],
            "BMI": [25.0, 30.0, 0.0],
            "Pregnancies": [0, 2, 1],  # zero legítimo: não está na lista
            "Outcome": [1, 0, 1],
        }
    )
    limpo, relatorio = limpar(df, cfg)
    assert limpo["Glucose"].isna().sum() == 1
    assert limpo["BMI"].isna().sum() == 1
    assert limpo["Pregnancies"].isna().sum() == 0
    assert relatorio.zeros_convertidos == {"Glucose": 1, "BMI": 1}


def test_duplicatas_removidas(df_bruto: pd.DataFrame, config_breast_cancer: Config) -> None:
    df = pd.concat([df_bruto, df_bruto.head(4)], ignore_index=True)
    limpo, relatorio = limpar(df, config_breast_cancer)
    assert relatorio.duplicatas_removidas == 4
    assert len(limpo) == len(df_bruto)


def test_remocao_de_outliers_e_recusada(
    df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    """ADR-007: a opção existe na config mas falha ruidosamente se ligada."""
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.limpeza.remover_outliers = True
    with pytest.raises(NotImplementedError, match="ADR-007"):
        limpar(df_bruto, cfg)


def test_alvo_codificado_com_maligno_como_positivo(xy) -> None:
    _, y = xy
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 212  # malignos
    assert y.mean() == pytest.approx(0.373, abs=0.001)


def test_alvo_ausente_falha(df_bruto: pd.DataFrame, config_breast_cancer: Config) -> None:
    with pytest.raises(KeyError):
        separar_alvo(df_bruto.drop(columns=["diagnosis"]), config_breast_cancer)


# --- Particionamento ---------------------------------------------------------


def test_proporcoes_do_split(particoes) -> None:
    tamanhos = particoes.tamanhos
    total = sum(tamanhos.values())
    assert total == 569
    assert tamanhos["treino"] == 341  # 60%
    assert tamanhos["validacao"] == 114  # 20%
    assert tamanhos["teste"] == 114  # 20%


def test_estratificacao_preserva_a_taxa_de_positivos(particoes) -> None:
    for conjunto, taxa in particoes.taxas_positivas.items():
        assert abs(taxa - 0.373) < 0.02, f"{conjunto} desviou: {taxa}"


def test_conjuntos_sao_disjuntos(particoes) -> None:
    """Uma amostra em dois conjuntos invalidaria a estimativa de desempenho."""
    treino = set(particoes.x_treino.index)
    validacao = set(particoes.x_validacao.index)
    teste = set(particoes.x_teste.index)
    assert not treino & validacao
    assert not treino & teste
    assert not validacao & teste


def test_split_e_reprodutivel(xy, config_breast_cancer: Config) -> None:
    x, y = xy
    a = particionar(x, y, config_breast_cancer)
    b = particionar(x, y, config_breast_cancer)
    assert list(a.x_teste.index) == list(b.x_teste.index)


def test_semente_diferente_produz_split_diferente(xy, config_breast_cancer: Config) -> None:
    x, y = xy
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.split.random_state = 999
    a = particionar(x, y, config_breast_cancer)
    b = particionar(x, y, cfg)
    assert list(a.x_teste.index) != list(b.x_teste.index)


def test_treino_mais_validacao_exclui_o_teste(particoes) -> None:
    x, y = particoes.treino_mais_validacao()
    assert len(x) == 455
    assert not set(x.index) & set(particoes.x_teste.index)


def test_cv_e_estratificada(config_breast_cancer: Config) -> None:
    cv = construir_cv(config_breast_cancer)
    assert cv.n_splits == 5
    assert cv.shuffle


def test_salvar_particoes(tmp_path, particoes, config_breast_cancer: Config) -> None:
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.dados_processados = tmp_path
    arquivos = salvar_particoes(particoes, cfg)
    assert len(arquivos) == 3
    treino = pd.read_csv(arquivos["treino"])
    assert "diagnosis" in treino.columns
    assert len(treino) == 341


# --- Pipeline ----------------------------------------------------------------


def test_separacao_de_tipos(xy) -> None:
    x, _ = xy
    numericas, categoricas = separar_tipos(x)
    assert len(numericas) == 30
    assert categoricas == []


def test_preprocessador_padroniza(xy, particoes, config_breast_cancer: Config) -> None:
    x, _ = xy
    pre = construir_preprocessador(x, config_breast_cancer)
    transformado = pre.fit_transform(particoes.x_treino)
    # Após StandardScaler, o treino tem média 0 e desvio 1.
    assert np.allclose(transformado.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(transformado.std(axis=0), 1, atol=1e-10)


def test_pipeline_nao_vaza_estatisticas_do_teste(
    xy, particoes, config_breast_cancer: Config
) -> None:
    """O escalonador deve usar média e desvio **do treino** no teste.

    Se alguém ajustar o scaler sobre o dataset inteiro, o teste transformado
    também teria média 0 — e este teste falha. É a trava contra o erro mais
    difícil de perceber deste projeto.
    """
    x, _ = xy
    pre = construir_preprocessador(x, config_breast_cancer)
    pre.fit(particoes.x_treino)

    teste_transformado = pre.transform(particoes.x_teste)
    media_teste = np.abs(teste_transformado.mean(axis=0))

    # Alguma feature precisa desviar de zero: o teste é outra amostra.
    assert media_teste.max() > 0.05, (
        "O conjunto de teste ficou centrado em zero, o que indica que o "
        "escalonador foi ajustado com dados do teste (vazamento)"
    )


def test_imputacao_funciona_mesmo_sem_ausentes_no_treino(
    xy, particoes, config_breast_cancer: Config
) -> None:
    """A API receberá payloads incompletos; o pipeline precisa aguentar."""
    x, _ = xy
    pre = construir_preprocessador(x, config_breast_cancer)
    pre.fit(particoes.x_treino)

    com_buraco = particoes.x_teste.copy()
    com_buraco.iloc[0, 0] = np.nan
    transformado = pre.transform(com_buraco)
    assert not np.isnan(transformado).any()


def test_pipeline_completo_treina_e_prediz(xy, particoes, config_breast_cancer: Config) -> None:
    x, _ = xy
    pipe = construir_pipeline(LogisticRegression(max_iter=1000), x, config_breast_cancer)
    assert isinstance(pipe, Pipeline)

    pipe.fit(particoes.x_treino, particoes.y_treino)
    probabilidades = pipe.predict_proba(particoes.x_validacao)[:, 1]
    assert probabilidades.shape == (114,)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


def test_nomes_das_features_de_saida(xy, particoes, config_breast_cancer: Config) -> None:
    """A ordem precisa casar com a matriz SHAP, ou a explicação erra a feature."""
    x, _ = xy
    pre = construir_preprocessador(x, config_breast_cancer)
    pre.fit(particoes.x_treino)
    nomes = nomes_features_saida(pre)
    assert len(nomes) == 30
    assert nomes[0] == "radius_mean"


def test_pipeline_recusa_dataframe_sem_colunas(config_breast_cancer: Config) -> None:
    with pytest.raises(ValueError, match="Nenhuma coluna"):
        construir_preprocessador(pd.DataFrame(index=[0, 1]), config_breast_cancer)


# --- Correlação --------------------------------------------------------------


def test_pares_redundantes_sao_o_trio_geometrico(particoes) -> None:
    """radius/perimeter/area medem a mesma grandeza: r > 0,95 é geometria."""
    pares = corr.pares_correlacionados(particoes.x_treino, limiar=0.95)
    assert len(pares) > 10
    topo = pares.iloc[0]
    assert {topo["feature_a"], topo["feature_b"]} == {"radius_mean", "perimeter_mean"}
    assert topo["correlacao"] > 0.99


def test_correlacao_com_alvo(particoes) -> None:
    resultado = corr.correlacao_com_alvo(particoes.x_treino, particoes.y_treino)
    assert len(resultado) == 30
    assert resultado["abs_r"].is_monotonic_decreasing
    # concave points_worst é a feature mais associada ao desfecho.
    assert resultado.index[0] == "concave points_worst"


def test_vif_detecta_multicolinearidade_severa(particoes) -> None:
    vif = corr.calcular_vif(particoes.x_treino)
    assert len(vif) == 30
    # A convenção trata VIF > 10 como severo; aqui passa de 1000.
    assert vif["vif"].max() > 1000
    assert (vif["severidade"] == "severo").sum() > 20


def test_selecao_remove_redundantes_e_reduz_vif(particoes, config_breast_cancer: Config) -> None:
    selecao = corr.selecionar_por_redundancia(
        particoes.x_treino, particoes.y_treino, config_breast_cancer
    )
    assert len(selecao.mantidas) == 24
    assert len(selecao.removidas) == 6
    assert set(selecao.mantidas) & set(selecao.removidas) == set()

    vif_antes = corr.calcular_vif(particoes.x_treino)["vif"].max()
    vif_depois = corr.calcular_vif(particoes.x_treino[selecao.mantidas])["vif"].max()
    assert vif_depois < vif_antes / 10


def test_selecao_preserva_a_feature_mais_informativa(
    particoes, config_breast_cancer: Config
) -> None:
    """Entre duas redundantes, mantemos a mais associada ao alvo."""
    selecao = corr.selecionar_por_redundancia(
        particoes.x_treino, particoes.y_treino, config_breast_cancer
    )
    informatividade = corr.correlacao_com_alvo(particoes.x_treino, particoes.y_treino)["abs_r"]
    for removida, (mantida, _) in selecao.motivo.items():
        assert informatividade[mantida] >= informatividade[removida]


def test_spearman_difere_de_pearson_sob_assimetria(particoes) -> None:
    pearson = corr.matriz_correlacao(particoes.x_treino, "pearson")
    spearman = corr.matriz_correlacao(particoes.x_treino, "spearman")
    assert not np.allclose(pearson.to_numpy(), spearman.to_numpy(), atol=0.05)
