"""Fixtures compartilhadas pela suíte de testes."""

from __future__ import annotations

from pathlib import Path

import pytest

from medai.config import RAIZ_PROJETO, Config, carregar_config


@pytest.fixture(scope="session")
def raiz_projeto() -> Path:
    return RAIZ_PROJETO


@pytest.fixture(scope="session")
def config_breast_cancer() -> Config:
    """Configuração do dataset principal, usada na maioria dos testes."""
    return carregar_config("configs/breast_cancer.yaml")
