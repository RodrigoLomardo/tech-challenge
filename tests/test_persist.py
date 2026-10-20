"""Testes de serialização e do contrato de entrada do artefato.

O teste de round-trip é a trava contra o erro mais caro possível em
produção: um artefato que carrega mas prediz diferente do que predizia no
treino.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from medai.config import Config
from medai.data.loader import carregar_breast_cancer_sklearn
from medai.features.build_pipeline import construir_pipeline
from medai.features.cleaning import limpar, separar_alvo
from medai.models import persist
from medai.models.persist import Metadados


@pytest.fixture(scope="module")
def xy(config_breast_cancer: Config):
    limpo, _ = limpar(carregar_breast_cancer_sklearn(), config_breast_cancer)
    return separar_alvo(limpo, config_breast_cancer)


@pytest.fixture(scope="module")
def pipeline_treinado(xy, config_breast_cancer: Config):
    x, y = xy
    pipe = construir_pipeline(
        LogisticRegression(max_iter=1000, random_state=42), x, config_breast_cancer
    )
    pipe.fit(x, y)
    return pipe


@pytest.fixture
def metadados_exemplo(xy) -> Metadados:
    x, _ = xy
    return Metadados(
        versao_modelo="teste_v1",
        nome_arquivo="teste.joblib",
        treinado_em="2026-10-20T00:00:00+00:00",
        dataset="breast_cancer_wisconsin",
        hash_config="abc123",
        hash_dados="def456",
        features_esperadas=x.columns.tolist(),
        feature_set="completo",
        limiar_decisao=0.158,
        limiar_origem="teste",
        zona_cinzenta=(0.30, 0.70),
        classe_positiva="M",
        rotulo_positivo="maligno",
        metricas_cv={"cv_f2": 0.97},
        metricas_validacao={"recall": 1.0},
        estatisticas_features={
            coluna: {
                "minimo": float(x[coluna].min()),
                "p01": float(x[coluna].quantile(0.01)),
                "mediana": float(x[coluna].median()),
                "p99": float(x[coluna].quantile(0.99)),
                "maximo": float(x[coluna].max()),
            }
            for coluna in x.columns
        },
        algoritmo="logistic_regression",
        hiperparametros={"modelo__C": "0.5"},
        random_state=42,
        n_amostras_treino=455,
    )


# --- Round-trip --------------------------------------------------------------


def test_round_trip_preserva_as_probabilidades(
    tmp_path, pipeline_treinado, metadados_exemplo, xy, config_breast_cancer: Config
) -> None:
    """Salvar e recarregar não pode alterar nenhuma predição.

    Qualquer divergência aqui significaria que o modelo servido em produção
    não é o modelo avaliado no teste.
    """
    x, _ = xy
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.modelos = tmp_path

    antes = pipeline_treinado.predict_proba(x)[:, 1]
    persist.salvar(pipeline_treinado, metadados_exemplo, cfg)
    recarregado, _ = persist.carregar(cfg)
    depois = recarregado.predict_proba(x)[:, 1]

    np.testing.assert_array_equal(antes, depois)


def test_round_trip_preserva_os_metadados(
    tmp_path, pipeline_treinado, metadados_exemplo, config_breast_cancer: Config
) -> None:
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.modelos = tmp_path

    persist.salvar(pipeline_treinado, metadados_exemplo, cfg)
    _, lidos = persist.carregar(cfg)

    assert lidos.limiar_decisao == metadados_exemplo.limiar_decisao
    assert lidos.features_esperadas == metadados_exemplo.features_esperadas
    assert lidos.zona_cinzenta == (0.30, 0.70)
    assert lidos.classe_positiva == "M"


def test_artefato_contem_o_pipeline_inteiro(
    tmp_path, pipeline_treinado, metadados_exemplo, config_breast_cancer: Config
) -> None:
    """ADR-005: salvar só o estimador obrigaria a reimplementar o
    pré-processamento na API, com risco de divergência silenciosa."""
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.modelos = tmp_path
    persist.salvar(pipeline_treinado, metadados_exemplo, cfg)
    recarregado, _ = persist.carregar(cfg)

    assert "preprocessamento" in recarregado.named_steps
    assert "modelo" in recarregado.named_steps


def test_pipeline_recarregado_aceita_dados_crus(
    tmp_path, pipeline_treinado, metadados_exemplo, xy, config_breast_cancer: Config
) -> None:
    """O artefato precisa escalonar sozinho — a API envia dados não tratados."""
    x, _ = xy
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.modelos = tmp_path
    persist.salvar(pipeline_treinado, metadados_exemplo, cfg)
    recarregado, _ = persist.carregar(cfg)

    # Valores na escala original, incluindo area_worst na casa dos milhares.
    proba = recarregado.predict_proba(x.head(3))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


# --- Metadados ---------------------------------------------------------------


def test_versoes_preenchidas_automaticamente(metadados_exemplo: Metadados) -> None:
    for pacote in ("python", "scikit-learn", "numpy", "pandas", "joblib"):
        assert pacote in metadados_exemplo.versoes


def test_serializacao_json_ida_e_volta(metadados_exemplo: Metadados) -> None:
    texto = metadados_exemplo.para_json()
    assert json.loads(texto)
    recuperado = Metadados.de_json(texto)
    assert recuperado.zona_cinzenta == metadados_exemplo.zona_cinzenta
    assert recuperado.features_esperadas == metadados_exemplo.features_esperadas


def test_divergencia_de_versao_e_sinalizada(metadados_exemplo: Metadados, caplog) -> None:
    """Incompatibilidade é aviso, não erro: o artefato pode funcionar, mas o
    operador precisa saber antes de servir predições clínicas."""
    metadados_exemplo.versoes["scikit-learn"] = "0.24.0"
    divergencias = persist._verificar_versoes(metadados_exemplo)
    assert any("scikit-learn" in d for d in divergencias)


def test_sem_divergencia_quando_versoes_batem(metadados_exemplo: Metadados) -> None:
    assert persist._verificar_versoes(metadados_exemplo) == []


def test_hash_de_arquivo_e_estavel(tmp_path) -> None:
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("a,b\n1,2\n")
    assert persist.hash_arquivo(arquivo) == persist.hash_arquivo(arquivo)

    arquivo.write_text("a,b\n1,3\n")
    assert persist.hash_arquivo(arquivo) != persist.hash_arquivo(tmp_path / "inexistente.csv")


def test_hash_de_arquivo_ausente() -> None:
    from pathlib import Path

    assert persist.hash_arquivo(Path("/nao/existe/jamais.csv")) == "indisponivel"


# --- Contrato de entrada -----------------------------------------------------


def test_validacao_reordena_colunas(xy, metadados_exemplo: Metadados) -> None:
    """Cliente que envia as features em outra ordem deve funcionar.

    O que não pode acontecer é o pipeline receber colunas desalinhadas — e a
    reordenação explícita é justamente o que impede isso.
    """
    x, _ = xy
    embaralhado = x[list(reversed(x.columns))]
    reordenado = persist.validar_entrada(embaralhado, metadados_exemplo)
    assert list(reordenado.columns) == metadados_exemplo.features_esperadas
    pd.testing.assert_frame_equal(reordenado, x[metadados_exemplo.features_esperadas])


def test_validacao_recusa_feature_faltante(xy, metadados_exemplo: Metadados) -> None:
    x, _ = xy
    with pytest.raises(ValueError, match="Features ausentes"):
        persist.validar_entrada(x.drop(columns=["radius_mean"]), metadados_exemplo)


def test_validacao_ignora_colunas_extras(xy, metadados_exemplo: Metadados) -> None:
    x, _ = xy
    com_extra = x.assign(id_paciente=range(len(x)), observacoes="nota")
    reordenado = persist.validar_entrada(com_extra, metadados_exemplo)
    assert "id_paciente" not in reordenado.columns
    assert len(reordenado.columns) == 30


# --- Erros -------------------------------------------------------------------


def test_carregar_sem_metadados(tmp_path, config_breast_cancer: Config) -> None:
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.caminhos.modelos = tmp_path
    with pytest.raises(FileNotFoundError, match="make train"):
        persist.carregar(cfg)


def test_carregar_sem_argumentos() -> None:
    with pytest.raises(ValueError, match="config"):
        persist.carregar()


# --- Artefato real (integração) ----------------------------------------------


@pytest.mark.integration
def test_artefato_versionado_carrega_e_prediz(xy, config_breast_cancer: Config) -> None:
    """Roda contra o artefato real em models/, se ele existir."""
    destino = config_breast_cancer.caminhos.absoluto("modelos")
    if not (destino / persist.NOME_METADADOS).exists():
        pytest.skip("Artefato ainda não treinado (make train)")

    pipeline, metadados = persist.carregar(config_breast_cancer)
    x, y = xy

    entrada = persist.validar_entrada(x, metadados)
    proba = pipeline.predict_proba(entrada)[:, 1]

    assert len(proba) == len(y)
    assert metadados.limiar_decisao < 0.5, "O limiar deve ser mais conservador que o padrão"
    assert metadados.n_amostras_treino == 455, "Treino final usa treino + validação"
    assert len(metadados.features_esperadas) == 30


def test_estatisticas_cobrem_todas_as_features(metadados_exemplo: Metadados) -> None:
    """A API valida a faixa de entrada e a UI dimensiona o formulário com isso."""
    assert len(metadados_exemplo.estatisticas_features) == 30
    for limites in metadados_exemplo.estatisticas_features.values():
        assert limites["minimo"] <= limites["mediana"] <= limites["maximo"]
