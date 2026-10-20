"""API REST do MedAI Triage.

Toda resposta carrega o aviso clínico. Toda predição é registrada em log
estruturado, sem dados identificáveis, para auditoria posterior — medir a
taxa de concordância com o laudo final é parte do monitoramento que um uso
real exigiria.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from medai import AVISO_CLINICO, __version__
from medai.api import deps, schemas
from medai.api.service import RECOMENDACOES, ServicoPredicao
from medai.logging_conf import configurar_logging, get_logger

logger = get_logger(__name__)

# Preenchido no startup, quando os metadados do modelo são conhecidos.
EsquemaEntrada: type | None = None


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    """Carrega modelo e constrói o schema de entrada na inicialização."""
    global EsquemaEntrada
    configurar_logging()
    try:
        servico = deps.inicializar()
        EsquemaEntrada = schemas.construir_schema_entrada(servico.metadados)
        logger.info("API pronta", extra={"modelo": servico.metadados.versao_modelo})
    except FileNotFoundError as erro:
        # Subir sem modelo é deliberado: /health precisa responder para que o
        # orquestrador relate "não pronto" em vez de reiniciar em laço.
        logger.error("Modelo indisponível no startup", extra={"erro": str(erro)})
    yield
    deps.encerrar()


app = FastAPI(
    title="MedAI Triage",
    description=(
        "Sistema de apoio à triagem diagnóstica — Hospital Universitário Santa Clara "
        "(instituição fictícia).\n\n"
        f"**{AVISO_CLINICO}**\n\n"
        "Projeto educacional. Não é dispositivo médico, não possui validação clínica "
        "nem aprovação regulatória."
    ),
    version=__version__,
    lifespan=ciclo_de_vida,
)


def _para_dataframe(entrada, servico: ServicoPredicao) -> pd.DataFrame:
    """Converte o corpo da requisição em DataFrame com os nomes originais."""
    bruto = entrada.model_dump(by_alias=True)
    return pd.DataFrame([bruto])[servico.metadados.features_esperadas]


def _montar_resposta(resultado, servico: ServicoPredicao) -> dict:
    return {
        "predicao": resultado.predicao,
        "rotulo": servico.rotulo(resultado.predicao),
        "probabilidade_malignidade": round(resultado.probabilidade, 4),
        "faixa_decisao": resultado.faixa,
        "recomendacao": RECOMENDACOES[resultado.faixa],
        "limiar_usado": servico.metadados.limiar_decisao,
        "versao_modelo": servico.metadados.versao_modelo,
        "avisos": resultado.avisos,
        "aviso_clinico": AVISO_CLINICO,
    }


# -----------------------------------------------------------------------------
# Saúde e metadados
# -----------------------------------------------------------------------------


@app.get("/health", response_model=schemas.RespostaSaude, tags=["serviço"])
def health() -> schemas.RespostaSaude:
    """Liveness. Responde mesmo sem modelo carregado, indicando o estado."""
    servico = deps.servico_opcional()
    return schemas.RespostaSaude(
        status="ok" if servico else "degradado",
        modelo_carregado=servico is not None,
        versao_modelo=servico.metadados.versao_modelo if servico else None,
        versao_medai=__version__,
    )


@app.get("/model/info", response_model=schemas.InfoModelo, tags=["serviço"])
def info_modelo(
    servico: Annotated[ServicoPredicao, Depends(deps.obter_servico)],
) -> schemas.InfoModelo:
    """Metadados do artefato servido, incluindo suas limitações conhecidas.

    Expor as limitações pela própria API é deliberado: quem integra o
    sistema precisa delas tanto quanto das métricas.
    """
    return schemas.InfoModelo(**servico.info())


# -----------------------------------------------------------------------------
# Predição
# -----------------------------------------------------------------------------


@app.post("/predict", response_model=schemas.RespostaPredicao, tags=["predição"])
def prever(
    entrada: dict,
    servico: Annotated[ServicoPredicao, Depends(deps.obter_servico)],
) -> schemas.RespostaPredicao:
    """Analisa um exame e devolve probabilidade, faixa de decisão e recomendação.

    Aceita os nomes originais das colunas (``concave points_mean``) ou a
    versão saneada (``concave_points_mean``).
    """
    if EsquemaEntrada is None:  # pragma: no cover - só sem modelo
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Schema não inicializado")

    validado = EsquemaEntrada(**entrada)
    inicio = time.perf_counter()
    resultado = servico.prever(_para_dataframe(validado, servico))[0]
    duracao = (time.perf_counter() - inicio) * 1000

    # Auditoria: sem dados identificáveis, apenas o que permite medir
    # concordância e detectar deriva.
    logger.info(
        "predicao",
        extra={
            "versao_modelo": servico.metadados.versao_modelo,
            "probabilidade": round(resultado.probabilidade, 4),
            "faixa": resultado.faixa,
            "n_avisos": len(resultado.avisos),
            "latencia_ms": round(duracao, 2),
        },
    )
    return schemas.RespostaPredicao(**_montar_resposta(resultado, servico))


@app.post("/explain", response_model=schemas.RespostaExplicacao, tags=["predição"])
def explicar(
    entrada: dict,
    servico: Annotated[ServicoPredicao, Depends(deps.obter_servico)],
    top: int = 10,
) -> schemas.RespostaExplicacao:
    """Predição acompanhada da explicação SHAP.

    As contribuições vêm em duas leituras. A individual mostra cada feature;
    a **agrupada** soma as que medem a mesma grandeza, e é a clinicamente
    correta — features redundantes dividem crédito entre si, fazendo cada uma
    parecer menos importante do que o grupo é.
    """
    if EsquemaEntrada is None:  # pragma: no cover
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Schema não inicializado")

    validado = EsquemaEntrada(**entrada)
    try:
        resultado, detalhe = servico.explicar(_para_dataframe(validado, servico), top=top)
    except RuntimeError as erro:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(erro)) from erro

    return schemas.RespostaExplicacao(**_montar_resposta(resultado, servico), **detalhe)


@app.post("/predict/batch", response_model=schemas.RespostaLote, tags=["predição"])
def prever_lote(
    entradas: list[dict],
    servico: Annotated[ServicoPredicao, Depends(deps.obter_servico)],
) -> schemas.RespostaLote:
    """Analisa vários exames e devolve a fila ordenada por suspeição."""
    if EsquemaEntrada is None:  # pragma: no cover
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Schema não inicializado")
    if not entradas:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Lista de exames vazia")

    validados = [EsquemaEntrada(**item) for item in entradas]
    quadro = pd.DataFrame([v.model_dump(by_alias=True) for v in validados])
    return _responder_lote(quadro, servico)


@app.post("/predict/csv", response_model=schemas.RespostaLote, tags=["predição"])
async def prever_csv(
    servico: Annotated[ServicoPredicao, Depends(deps.obter_servico)],
    arquivo: Annotated[UploadFile, File(description="CSV com uma linha por exame")],
) -> schemas.RespostaLote:
    """Triagem de fila a partir de um CSV."""
    conteudo = await arquivo.read()
    try:
        quadro = pd.read_csv(io.BytesIO(conteudo))
    except Exception as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"CSV ilegível: {erro}") from erro

    faltantes = [c for c in servico.metadados.features_esperadas if c not in quadro.columns]
    if faltantes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Colunas ausentes no CSV: {faltantes[:10]}",
        )
    return _responder_lote(quadro, servico)


def _responder_lote(quadro: pd.DataFrame, servico: ServicoPredicao) -> schemas.RespostaLote:
    """Ordena por probabilidade decrescente — é a fila de triagem."""
    resultados = servico.prever(quadro)
    respostas = [schemas.RespostaPredicao(**_montar_resposta(r, servico)) for r in resultados]
    respostas.sort(key=lambda r: r.probabilidade_malignidade, reverse=True)

    resumo: dict[str, int] = {}
    for resposta in respostas:
        resumo[resposta.faixa_decisao] = resumo.get(resposta.faixa_decisao, 0) + 1

    logger.info("predicao_lote", extra={"n": len(respostas), "faixas": resumo})
    return schemas.RespostaLote(total=len(respostas), resumo_faixas=resumo, resultados=respostas)


# -----------------------------------------------------------------------------
# Erros
# -----------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def tratar_valor_invalido(request, exc: ValueError) -> JSONResponse:
    """Converte erro de contrato em 422 legível, não em 500."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc), "aviso_clinico": AVISO_CLINICO},
    )
