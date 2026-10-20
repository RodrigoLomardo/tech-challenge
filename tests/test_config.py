"""Testes do carregamento e da validação de configuração."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medai.config import Config, carregar_config


def test_carrega_config_principal(config_breast_cancer: Config) -> None:
    cfg = config_breast_cancer
    assert cfg.dataset.nome == "breast_cancer_wisconsin"
    assert cfg.dataset.coluna_alvo == "diagnosis"
    # Regra inviolável: a classe positiva é a maligna.
    assert cfg.dataset.valor_positivo == "M"
    assert "id" in cfg.dataset.colunas_descartar


def test_carrega_config_bonus() -> None:
    cfg = carregar_config("configs/diabetes.yaml")
    # No Pima, zeros impossíveis são ausentes disfarçados.
    assert "Glucose" in cfg.limpeza.zeros_impossiveis
    assert cfg.dataset.valor_positivo == 1


def test_hash_config_e_estavel_e_distingue_arquivos() -> None:
    a = carregar_config("configs/breast_cancer.yaml")
    b = carregar_config("configs/breast_cancer.yaml")
    c = carregar_config("configs/diabetes.yaml")
    assert a.hash_config == b.hash_config
    assert a.hash_config != c.hash_config


def test_metrica_primaria_e_recall(config_breast_cancer: Config) -> None:
    """ADR-003: accuracy nunca seleciona modelo neste projeto."""
    assert config_breast_cancer.modelagem.metrica_primaria == "recall"


def test_outliers_nao_sao_removidos(config_breast_cancer: Config) -> None:
    """ADR-007: em dados clínicos o outlier costuma ser o caso de interesse."""
    assert config_breast_cancer.limpeza.remover_outliers is False


def test_proporcoes_de_split_somam_menos_que_um() -> None:
    with pytest.raises(ValidationError):
        Config(
            dataset={
                "nome": "x",
                "loader": "sklearn_breast_cancer",
                "coluna_alvo": "y",
                "valor_positivo": 1,
            },
            split={"proporcao_teste": 0.6, "proporcao_validacao": 0.5},
        )


def test_zona_cinzenta_invalida_e_rejeitada() -> None:
    with pytest.raises(ValidationError):
        Config(
            dataset={
                "nome": "x",
                "loader": "sklearn_breast_cancer",
                "coluna_alvo": "y",
                "valor_positivo": 1,
            },
            modelagem={"zona_cinzenta": (0.8, 0.2)},
        )


def test_loader_csv_exige_caminho() -> None:
    with pytest.raises(ValidationError):
        Config(
            dataset={
                "nome": "x",
                "loader": "csv",
                "coluna_alvo": "y",
                "valor_positivo": 1,
            }
        )


def test_config_rejeita_chave_desconhecida() -> None:
    """extra='forbid' evita que um typo no YAML passe silenciosamente."""
    with pytest.raises(ValidationError):
        Config(
            dataset={
                "nome": "x",
                "loader": "sklearn_breast_cancer",
                "coluna_alvo": "y",
                "valor_positivo": 1,
            },
            chave_inexistente=123,
        )


def test_config_arquivo_ausente() -> None:
    with pytest.raises(FileNotFoundError):
        carregar_config("configs/nao_existe.yaml")
