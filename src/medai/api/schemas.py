"""Contratos de entrada e saída da API.

O schema de entrada é **construído em tempo de execução a partir dos
metadados do modelo**, não escrito à mão. A razão é direta: o modelo declara
quais features espera e em que ordem, e duplicar essa lista aqui criaria duas
fontes da verdade que divergiriam no primeiro retreino com feature set
diferente.

Os nomes das colunas contêm espaço (``concave points_mean``), o que não é um
identificador Python válido. Cada campo recebe um nome saneado e um *alias*
com o nome original, de modo que ambos funcionem na requisição.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from medai import AVISO_CLINICO
from medai.models.persist import Metadados


def _sanear(nome: str) -> str:
    """Converte o nome da coluna em identificador Python válido."""
    return nome.replace(" ", "_").replace("-", "_")


def construir_schema_entrada(metadados: Metadados) -> type[BaseModel]:
    """Gera o modelo pydantic de entrada a partir do contrato do artefato.

    Cada feature vira um campo obrigatório ``float``, com:

    - ``ge=0`` — medidas morfológicas são grandezas físicas; negativo indica
      corrupção de dados, não um caso atípico;
    - ``alias`` com o nome original da coluna;
    - descrição com a faixa observada no treino, visível no Swagger.

    Note que **não há limite superior**. Valores acima do máximo observado
    são sinalizados na resposta, não recusados: um tumor excepcionalmente
    grande é exatamente o caso que o sistema precisa processar, não rejeitar.
    """
    campos: dict[str, Any] = {}
    for coluna in metadados.features_esperadas:
        estatisticas = metadados.estatisticas_features.get(coluna, {})
        descricao = (
            f"Faixa observada no treino: {estatisticas.get('minimo', 0):.4g} a "
            f"{estatisticas.get('maximo', 0):.4g} (mediana {estatisticas.get('mediana', 0):.4g})"
        )
        campos[_sanear(coluna)] = (
            float,
            Field(..., ge=0, alias=coluna, description=descricao),
        )

    exemplo = {
        coluna: round(metadados.estatisticas_features[coluna]["mediana"], 4)
        for coluna in metadados.features_esperadas
        if coluna in metadados.estatisticas_features
    }

    return create_model(
        "ExameEntrada",
        __config__=ConfigDict(
            populate_by_name=True,
            json_schema_extra={"example": exemplo},
        ),
        **campos,
    )


# --- Saída -------------------------------------------------------------------


class ContribuicaoFeature(BaseModel):
    """Contribuição de uma feature para a predição, em log-odds."""

    feature: str
    contribuicao: float = Field(description="Valor SHAP, em log-odds")
    valor_medido: float
    efeito: str = Field(description="'aumenta suspeição' ou 'reduz suspeição'")


class ContribuicaoGrupo(BaseModel):
    """Contribuição somada das features que medem a mesma grandeza.

    É a leitura clinicamente correta: features redundantes dividem crédito
    individualmente, e só a soma revela o peso real do grupo.
    """

    grupo: str
    contribuicao_absoluta: float


class RespostaPredicao(BaseModel):
    """Resultado de uma análise."""

    model_config = ConfigDict(protected_namespaces=())

    predicao: int = Field(description="1 = maligno, 0 = benigno, no limiar operacional")
    rotulo: str
    probabilidade_malignidade: float = Field(ge=0, le=1)
    faixa_decisao: str = Field(description="'benigno', 'zona_cinzenta_revisao_medica' ou 'maligno'")
    recomendacao: str
    limiar_usado: float
    versao_modelo: str
    avisos: list[str] = Field(
        default_factory=list,
        description="Entradas fora da faixa observada no treino, entre outros alertas",
    )
    aviso_clinico: str = AVISO_CLINICO


class RespostaExplicacao(RespostaPredicao):
    """Predição acompanhada da explicação SHAP."""

    contribuicoes: list[ContribuicaoFeature]
    contribuicoes_agrupadas: list[ContribuicaoGrupo]
    valor_base: float = Field(
        description="Log-odds de referência antes das contribuições das features"
    )


class RespostaLote(BaseModel):
    """Fila de triagem: casos ordenados por probabilidade decrescente."""

    total: int
    resumo_faixas: dict[str, int]
    resultados: list[RespostaPredicao]
    aviso_clinico: str = AVISO_CLINICO


class RespostaSaude(BaseModel):
    """Liveness e prontidão do serviço."""

    status: str
    modelo_carregado: bool
    versao_modelo: str | None = None
    versao_medai: str


class InfoModelo(BaseModel):
    """O que a API sabe sobre o artefato que está servindo."""

    model_config = ConfigDict(protected_namespaces=())

    versao_modelo: str
    algoritmo: str
    treinado_em: str
    dataset: str
    n_amostras_treino: int
    features_esperadas: list[str]
    # Faixa observada no treino por feature. A UI usa para dimensionar o
    # formulário; expor aqui evita que ela precise do modelo em memória.
    estatisticas_features: dict[str, dict[str, float]]
    limiar_decisao: float
    limiar_origem: str
    zona_cinzenta: tuple[float, float]
    metricas_validacao: dict[str, float]
    versoes: dict[str, str]
    limitacoes: list[str]
