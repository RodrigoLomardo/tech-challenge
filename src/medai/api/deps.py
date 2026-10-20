"""Dependências compartilhadas da API.

O serviço de predição é um singleton criado no ciclo de vida da aplicação.
Carregar o artefato por requisição acrescentaria dezenas de milissegundos a
cada chamada sem nenhum ganho — o modelo é imutável.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, status

from medai.api.service import ServicoPredicao
from medai.config import carregar_config

_servico: ServicoPredicao | None = None


def inicializar() -> ServicoPredicao:
    """Carrega o modelo. Chamado uma vez, no startup da aplicação."""
    global _servico
    caminho_config = os.getenv("MEDAI_CONFIG", "configs/breast_cancer.yaml")
    _servico = ServicoPredicao(carregar_config(caminho_config))
    return _servico


def encerrar() -> None:
    global _servico
    _servico = None


def obter_servico() -> ServicoPredicao:
    """Dependência do FastAPI. Erro 503 se o modelo não estiver carregado."""
    if _servico is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("Modelo não carregado. Treine-o com 'make train' e reinicie o serviço."),
        )
    return _servico


def servico_opcional() -> ServicoPredicao | None:
    """Versão que não levanta erro — usada pelo health check."""
    return _servico
