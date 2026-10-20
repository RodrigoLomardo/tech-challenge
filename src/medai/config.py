"""Carregamento e validação da configuração em YAML.

Todo caminho, hiperparâmetro e regra de limpeza vem de ``configs/*.yaml``.
Isso é o que permite rodar o mesmo pipeline sobre outro dataset (p. ex. Pima
Diabetes) sem alterar código, e o que torna cada execução auditável: o hash da
configuração é registrado junto das métricas.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# Raiz do repositório: três níveis acima de src/medai/config.py.
RAIZ_PROJETO = Path(__file__).resolve().parents[2]


class ConfigDataset(BaseModel):
    """Origem dos dados e identificação das colunas."""

    nome: str
    descricao: str = ""
    # Rota A: carregador embutido do scikit-learn (sem credencial).
    # Rota B: CSV baixado do Kaggle. Ver ADR-002.
    loader: Literal["sklearn_breast_cancer", "csv"]
    caminho_csv: Path | None = None
    kaggle_id: str | None = None
    url_referencia: str = ""

    coluna_alvo: str
    # Rótulo tratado como classe positiva (1). Para câncer de mama: "M".
    # Errar isso inverte a leitura de precision/recall. Ver regra 3 do CLAUDE.md.
    valor_positivo: str | int
    colunas_descartar: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exige_caminho_para_csv(self) -> ConfigDataset:
        if self.loader == "csv" and self.caminho_csv is None:
            raise ValueError("loader='csv' exige 'caminho_csv'")
        return self


class ConfigLimpeza(BaseModel):
    """Regras de limpeza aplicadas antes do split."""

    remover_duplicatas: bool = True
    # Colunas onde o valor zero é fisiologicamente impossível e representa
    # ausência disfarçada (ex.: Glucose=0 no Pima). Convertidas em NaN.
    zeros_impossiveis: list[str] = Field(default_factory=list)
    estrategia_imputacao_numerica: Literal["median", "mean"] = "median"
    estrategia_imputacao_categorica: Literal["most_frequent", "constant"] = "most_frequent"
    # Outliers são identificados e discutidos, nunca removidos. Ver ADR-007.
    remover_outliers: bool = False


class ConfigSplit(BaseModel):
    """Particionamento treino / validação / teste."""

    proporcao_teste: float = Field(0.20, gt=0, lt=1)
    proporcao_validacao: float = Field(0.20, gt=0, lt=1)
    estratificar: bool = True
    random_state: int = 42
    n_folds_cv: int = Field(5, ge=2)

    @model_validator(mode="after")
    def _soma_valida(self) -> ConfigSplit:
        if self.proporcao_teste + self.proporcao_validacao >= 1.0:
            raise ValueError("teste + validação devem somar menos que 1.0")
        return self


class ConfigCorrelacao(BaseModel):
    """Parâmetros da análise de multicolinearidade."""

    metodos: list[Literal["pearson", "spearman"]] = Field(default=["pearson", "spearman"])
    limiar_redundancia: float = Field(0.95, gt=0, le=1)
    calcular_vif: bool = True


class ConfigModelagem(BaseModel):
    """Catálogo de modelos e critério de seleção."""

    modelos: list[str] = Field(default_factory=list)
    # Recall da classe maligna é a prioridade *clínica*: falso negativo é
    # câncer não sinalizado (ADR-003). Mas recall puro como critério de
    # seleção escolhe o modelo degenerado que classifica tudo como maligno —
    # recall 1,0, precisão 0,37, utilidade nula. A seleção usa F-beta, que
    # pondera recall beta² vezes mais que precisão e por isso operacionaliza
    # a prioridade clínica sem aceitar a solução trivial (ADR-014).
    metrica_primaria: Literal["recall", "f1", "roc_auc", "average_precision"] = "recall"
    # beta=2 → recall pesa 4× precisão na seleção de modelo.
    beta_fscore: float = Field(2.0, gt=0)
    metricas_secundarias: list[str] = Field(default=["f1", "roc_auc", "average_precision"])
    recall_minimo_alvo: float = Field(0.98, gt=0, le=1)
    # Faixa de probabilidade encaminhada para revisão médica em vez de receber
    # rótulo binário. Ver ADR-008.
    zona_cinzenta: tuple[float, float] = (0.30, 0.70)
    calibrar_probabilidades: bool = True
    sementes_robustez: list[int] = Field(default=[42, 7, 13, 2024, 99])

    @field_validator("zona_cinzenta")
    @classmethod
    def _faixa_coerente(cls, v: tuple[float, float]) -> tuple[float, float]:
        inferior, superior = v
        if not 0 < inferior < superior < 1:
            raise ValueError("zona_cinzenta deve satisfazer 0 < inferior < superior < 1")
        return v


class ConfigCaminhos(BaseModel):
    """Diretórios de saída, relativos à raiz do projeto."""

    dados_brutos: Path = Path("data/raw")
    dados_processados: Path = Path("data/processed")
    modelos: Path = Path("models")
    figuras: Path = Path("reports/figures")
    metricas: Path = Path("reports/metrics")
    experimentos: Path = Path("reports/experiments.csv")

    def absoluto(self, campo: str) -> Path:
        """Resolve um dos caminhos contra a raiz do projeto."""
        return RAIZ_PROJETO / getattr(self, campo)


class Config(BaseModel):
    """Configuração completa de uma execução do pipeline."""

    dataset: ConfigDataset
    limpeza: ConfigLimpeza = Field(default_factory=ConfigLimpeza)
    split: ConfigSplit = Field(default_factory=ConfigSplit)
    correlacao: ConfigCorrelacao = Field(default_factory=ConfigCorrelacao)
    modelagem: ConfigModelagem = Field(default_factory=ConfigModelagem)
    caminhos: ConfigCaminhos = Field(default_factory=ConfigCaminhos)

    # Preenchido no carregamento; entra em experiments.csv para rastrear qual
    # configuração produziu qual resultado.
    hash_config: str = ""

    model_config = {"extra": "forbid"}


def carregar_config(caminho: str | Path) -> Config:
    """Lê um YAML, valida contra o schema e calcula seu hash.

    Args:
        caminho: Caminho do YAML, absoluto ou relativo à raiz do projeto.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        pydantic.ValidationError: Se o conteúdo violar o schema.
    """
    caminho = Path(caminho)
    if not caminho.is_absolute():
        caminho = RAIZ_PROJETO / caminho
    if not caminho.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {caminho}")

    bruto = caminho.read_text(encoding="utf-8")
    dados: dict[str, Any] = yaml.safe_load(bruto) or {}

    config = Config(**dados)
    config.hash_config = hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:12]
    return config
