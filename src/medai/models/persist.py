"""Serialização do modelo e dos metadados que o tornam utilizável.

O que é salvo é o **pipeline completo** — imputador, escalonador e estimador
em um único objeto (ADR-005). Salvar apenas o estimador obrigaria a
reimplementar o pré-processamento no servidor, e qualquer divergência sutil
produziria predições erradas sem erro aparente.

Ao lado do artefato vai um ``metadata.json`` que responde às perguntas que a
API precisa fazer antes de servir: quais features ele espera e em que ordem,
qual o limiar de decisão, sob quais versões de biblioteca foi treinado, e de
qual arquivo de dados ele veio.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from medai import __version__
from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)

NOME_METADADOS = "metadata.json"


@dataclass
class Metadados:
    """Tudo que a API precisa saber sobre o artefato que está carregando."""

    # --- Identificação ------------------------------------------------------
    versao_modelo: str
    nome_arquivo: str
    treinado_em: str
    dataset: str
    hash_config: str
    hash_dados: str

    # --- Contrato de entrada ------------------------------------------------
    # A ORDEM importa: o pipeline espera as colunas nesta sequência, e a API
    # valida contra esta lista. Colunas fora de ordem produziriam predições
    # silenciosamente erradas.
    features_esperadas: list[str]
    feature_set: str

    # --- Ponto de operação --------------------------------------------------
    limiar_decisao: float
    limiar_origem: str
    zona_cinzenta: tuple[float, float]
    classe_positiva: str
    rotulo_positivo: str

    # --- Desempenho conhecido -----------------------------------------------
    # Métricas de treino e validação. O teste NÃO entra aqui: na Fase 5 ele
    # ainda não foi tocado, e registrá-lo depois exigiria nova versão do
    # artefato — o que é o comportamento correto.
    metricas_cv: dict[str, float]
    metricas_validacao: dict[str, float]

    # --- Distribuição observada no treino -----------------------------------
    # Mínimo, percentis e mediana por feature. A API usa para sinalizar
    # entradas fora da faixa observada (sem recusá-las — um tumor atípico
    # real não deve ser rejeitado) e a UI usa para dimensionar os controles
    # do formulário.
    estatisticas_features: dict[str, dict[str, float]]

    # --- Reprodutibilidade --------------------------------------------------
    algoritmo: str
    hiperparametros: dict[str, Any]
    random_state: int
    n_amostras_treino: int
    versoes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.versoes:
            self.versoes = {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
                "medai": __version__,
            }

    def para_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)

    @classmethod
    def de_json(cls, texto: str) -> Metadados:
        dados = json.loads(texto)
        dados["zona_cinzenta"] = tuple(dados["zona_cinzenta"])
        return cls(**dados)


def hash_arquivo(caminho: Path) -> str:
    """SHA-256 abreviado de um arquivo, para rastrear qual dado gerou o modelo."""
    if not caminho.exists():
        return "indisponivel"
    digest = hashlib.sha256(caminho.read_bytes()).hexdigest()
    return digest[:16]


def salvar(pipeline: Pipeline, metadados: Metadados, config: Config) -> tuple[Path, Path]:
    """Grava o pipeline e seus metadados em ``models/``.

    Returns:
        Os caminhos do artefato e do arquivo de metadados.
    """
    destino = config.caminhos.absoluto("modelos")
    destino.mkdir(parents=True, exist_ok=True)

    caminho_modelo = destino / metadados.nome_arquivo
    joblib.dump(pipeline, caminho_modelo, compress=3)

    caminho_metadados = destino / NOME_METADADOS
    caminho_metadados.write_text(metadados.para_json(), encoding="utf-8")

    logger.info(
        "Modelo serializado",
        extra={
            "arquivo": caminho_modelo.name,
            "tamanho_kb": round(caminho_modelo.stat().st_size / 1024, 1),
            "algoritmo": metadados.algoritmo,
            "limiar": metadados.limiar_decisao,
        },
    )
    return caminho_modelo, caminho_metadados


def carregar(
    config: Config | None = None, caminho_modelo: Path | None = None
) -> tuple[Pipeline, Metadados]:
    """Carrega pipeline e metadados, verificando compatibilidade de versões.

    A incompatibilidade de versão é **aviso, não erro**: o artefato pode
    carregar e funcionar. Mas um pipeline do scikit-learn 1.5 carregado sob
    1.6 pode produzir resultados sutilmente diferentes, ou falhar em um
    atributo interno renomeado — e o operador precisa saber disso antes de
    servir predições clínicas.
    """
    if caminho_modelo is None:
        if config is None:
            raise ValueError("Informe `config` ou `caminho_modelo`")
        destino = config.caminhos.absoluto("modelos")
        caminho_metadados = destino / NOME_METADADOS
        if not caminho_metadados.exists():
            raise FileNotFoundError(
                f"Metadados não encontrados em {caminho_metadados}. "
                "Treine o modelo com: make train"
            )
        metadados = Metadados.de_json(caminho_metadados.read_text(encoding="utf-8"))
        caminho_modelo = destino / metadados.nome_arquivo
    else:
        caminho_modelo = Path(caminho_modelo)
        caminho_metadados = caminho_modelo.parent / NOME_METADADOS
        metadados = Metadados.de_json(caminho_metadados.read_text(encoding="utf-8"))

    if not caminho_modelo.exists():
        raise FileNotFoundError(f"Artefato não encontrado: {caminho_modelo}")

    pipeline = joblib.load(caminho_modelo)
    _verificar_versoes(metadados)

    logger.info(
        "Modelo carregado",
        extra={"versao": metadados.versao_modelo, "algoritmo": metadados.algoritmo},
    )
    return pipeline, metadados


def _verificar_versoes(metadados: Metadados) -> list[str]:
    """Compara as versões de treino com as de execução."""
    atuais = {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    divergencias = [
        f"{nome}: treinado com {metadados.versoes.get(nome)}, executando com {atual}"
        for nome, atual in atuais.items()
        if metadados.versoes.get(nome) and metadados.versoes[nome] != atual
    ]
    if divergencias:
        logger.warning(
            "Divergência de versões entre treino e execução",
            extra={"divergencias": divergencias},
        )
    return divergencias


def validar_entrada(dados: pd.DataFrame, metadados: Metadados) -> pd.DataFrame:
    """Confere e reordena as colunas conforme o contrato do modelo.

    Reordenar em vez de apenas validar é deliberado: um cliente que envie as
    features corretas em ordem diferente deve funcionar, não falhar. O que
    não pode acontecer é o pipeline receber colunas desalinhadas — e é
    justamente isso que uma reordenação explícita impede.
    """
    esperadas = metadados.features_esperadas
    faltantes = [c for c in esperadas if c not in dados.columns]
    if faltantes:
        raise ValueError(
            f"Features ausentes na entrada: {faltantes}. "
            f"O modelo '{metadados.versao_modelo}' espera {len(esperadas)} features."
        )

    extras = [c for c in dados.columns if c not in esperadas]
    if extras:
        logger.info("Colunas extras ignoradas", extra={"colunas": extras[:10]})

    return dados[esperadas]
