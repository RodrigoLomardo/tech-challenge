"""Testes da configuração de logging."""

from __future__ import annotations

import json
import logging

from medai import AVISO_CLINICO
from medai.logging_conf import configurar_logging, get_logger


def test_formato_json_produz_linha_parseavel(capsys) -> None:
    configurar_logging(nivel="INFO", formato="json")
    get_logger("teste").info("predicao realizada", extra={"versao_modelo": "v1"})

    registro = json.loads(capsys.readouterr().out.strip())
    assert registro["mensagem"] == "predicao realizada"
    assert registro["nivel"] == "INFO"
    # Campos extras viram contexto estruturado — base da auditoria de predições.
    assert registro["contexto"]["versao_modelo"] == "v1"


def test_formato_texto(capsys) -> None:
    configurar_logging(nivel="INFO", formato="text")
    get_logger("teste").info("mensagem simples")
    assert "mensagem simples" in capsys.readouterr().out


def test_configuracao_repetida_nao_duplica_handlers() -> None:
    """Notebooks reexecutam células de setup; handlers não podem acumular."""
    for _ in range(3):
        configurar_logging()
    assert len(logging.getLogger().handlers) == 1


def test_aviso_clinico_centralizado() -> None:
    assert "não constitui diagnóstico" in AVISO_CLINICO.lower()
