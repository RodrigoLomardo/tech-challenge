"""Configuração centralizada de logging.

Um único ponto de configuração evita que notebooks, scripts de treino e a API
emitam logs em formatos divergentes. O formato ``json`` existe porque a API
registra toda predição para auditoria, e log estruturado é o que torna esse
registro consultável.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

_FORMATO_TEXTO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURADO = False


class FormatadorJSON(logging.Formatter):
    """Serializa cada registro como uma linha JSON.

    Campos extras passados via ``logger.info(..., extra={...})`` são incluídos
    no objeto, o que permite anexar metadados de predição (versão do modelo,
    probabilidade, latência) sem poluir a mensagem de texto.
    """

    _PADRAO = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "nivel": record.levelname,
            "logger": record.name,
            "mensagem": record.getMessage(),
        }
        if record.exc_info:
            payload["excecao"] = self.formatException(record.exc_info)

        extras = {k: v for k, v in record.__dict__.items() if k not in self._PADRAO}
        if extras:
            payload["contexto"] = extras

        return json.dumps(payload, ensure_ascii=False, default=str)


def configurar_logging(nivel: str | None = None, formato: str | None = None) -> None:
    """Configura o logger raiz. Idempotente — chamadas repetidas não duplicam handlers.

    Args:
        nivel: ``DEBUG``/``INFO``/``WARNING``/``ERROR``. Sem valor, lê
            ``MEDAI_LOG_LEVEL`` (padrão ``INFO``).
        formato: ``text`` ou ``json``. Sem valor, lê ``MEDAI_LOG_FORMAT``
            (padrão ``text``).
    """
    global _CONFIGURADO

    nivel = (nivel or os.getenv("MEDAI_LOG_LEVEL", "INFO")).upper()
    formato = (formato or os.getenv("MEDAI_LOG_FORMAT", "text")).lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        FormatadorJSON() if formato == "json" else logging.Formatter(_FORMATO_TEXTO)
    )

    raiz = logging.getLogger()
    # Substitui handlers anteriores em vez de acumular: notebooks reexecutam
    # células de setup o tempo todo, e handlers duplicados geram log repetido.
    for existente in list(raiz.handlers):
        raiz.removeHandler(existente)
    raiz.addHandler(handler)
    raiz.setLevel(nivel)

    # Bibliotecas verbosas que não acrescentam informação ao nosso fluxo.
    for ruidoso in ("matplotlib", "PIL", "urllib3", "numexpr"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    _CONFIGURADO = True


def get_logger(nome: str) -> logging.Logger:
    """Devolve um logger, configurando o logging raiz na primeira chamada."""
    if not _CONFIGURADO:
        configurar_logging()
    return logging.getLogger(nome)
