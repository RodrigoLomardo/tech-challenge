"""Carregamento do dataset bruto, com duas rotas de aquisição.

Rota A — ``sklearn``: ``sklearn.datasets.load_breast_cancer()``. Mesma origem
UCI do dataset do Kaggle, sem exigir credencial. Garante que o projeto roda em
qualquer máquina e dentro do Docker.

Rota B — ``csv``: o CSV baixado do Kaggle. Traz a coluna ``id`` e a coluna
fantasma ``Unnamed: 32`` (100% nula, artefato de vírgula final), que dão
trabalho real à etapa de limpeza.

Ver ADR-002 em docs/DECISOES_TECNICAS.md.

As duas rotas produzem **o mesmo schema canônico** (nomenclatura do Kaggle),
porque o restante do pipeline não pode depender de qual rota foi usada.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from medai.config import RAIZ_PROJETO, Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)

# As 10 medidas morfológicas de base. Cada uma aparece três vezes no dataset:
# média (_mean), erro padrão (_se) e pior valor observado (_worst).
MEDIDAS_BASE: tuple[str, ...] = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave points",
    "symmetry",
    "fractal_dimension",
)

SUFIXOS: tuple[str, ...] = ("mean", "se", "worst")

# Schema canônico: 30 features na nomenclatura do Kaggle.
FEATURES_CANONICAS: tuple[str, ...] = tuple(
    f"{medida}_{sufixo}" for sufixo in SUFIXOS for medida in MEDIDAS_BASE
)

# Tradução dos nomes do scikit-learn ("mean radius", "radius error",
# "worst radius") para a nomenclatura canônica. Note que o Kaggle mantém o
# espaço em "concave points" mas usa underscore em "fractal_dimension" —
# inconsistência da fonte que precisamos reproduzir fielmente.
_BASE_SKLEARN_PARA_CANONICA: dict[str, str] = {
    "radius": "radius",
    "texture": "texture",
    "perimeter": "perimeter",
    "area": "area",
    "smoothness": "smoothness",
    "compactness": "compactness",
    "concavity": "concavity",
    "concave points": "concave points",
    "symmetry": "symmetry",
    "fractal dimension": "fractal_dimension",
}


def _traduzir_nome_sklearn(nome: str) -> str:
    """Converte um nome de feature do scikit-learn para o schema canônico.

    >>> _traduzir_nome_sklearn("mean radius")
    'radius_mean'
    >>> _traduzir_nome_sklearn("fractal dimension error")
    'fractal_dimension_se'
    >>> _traduzir_nome_sklearn("worst concave points")
    'concave points_worst'
    """
    if nome.startswith("mean "):
        base, sufixo = nome[len("mean ") :], "mean"
    elif nome.endswith(" error"):
        base, sufixo = nome[: -len(" error")], "se"
    elif nome.startswith("worst "):
        base, sufixo = nome[len("worst ") :], "worst"
    else:  # pragma: no cover - não ocorre no dataset atual
        raise ValueError(f"Nome de feature inesperado no scikit-learn: {nome!r}")

    try:
        return f"{_BASE_SKLEARN_PARA_CANONICA[base]}_{sufixo}"
    except KeyError as erro:  # pragma: no cover
        raise ValueError(f"Medida desconhecida: {base!r}") from erro


def carregar_breast_cancer_sklearn() -> pd.DataFrame:
    """Rota A: carrega o dataset embutido no scikit-learn no schema canônico.

    **Armadilha tratada aqui:** no scikit-learn, ``target == 0`` é *maligno* e
    ``target == 1`` é *benigno* — o inverso da convenção deste projeto, em que
    a classe positiva é a maligna. Usar ``target`` diretamente como rótulo
    inverteria silenciosamente precision e recall, e o modelo pareceria estar
    funcionando enquanto otimizaria exatamente a métrica errada.
    """
    from sklearn.datasets import load_breast_cancer

    bruto = load_breast_cancer()

    df = pd.DataFrame(bruto.data, columns=[_traduzir_nome_sklearn(n) for n in bruto.feature_names])

    # bruto.target_names == ['malignant', 'benign'], portanto índice 0 = maligno.
    nomes_classes = list(bruto.target_names)
    indice_maligno = nomes_classes.index("malignant")
    df.insert(0, "diagnosis", ["M" if alvo == indice_maligno else "B" for alvo in bruto.target])

    # Não sintetizamos a coluna 'id': ela não existe nesta fonte, e inventar
    # identificadores seria fabricar dado. O pipeline de limpeza descarta
    # colunas ausentes sem reclamar.
    logger.info(
        "Dataset carregado via scikit-learn",
        extra={"linhas": len(df), "colunas": df.shape[1], "rota": "sklearn"},
    )
    return df


def carregar_csv(caminho: str | Path) -> pd.DataFrame:
    """Rota B: lê o CSV baixado do Kaggle, preservando suas peculiaridades."""
    caminho = Path(caminho)
    if not caminho.is_absolute():
        caminho = RAIZ_PROJETO / caminho
    if not caminho.exists():
        raise FileNotFoundError(f"CSV não encontrado: {caminho}")

    df = pd.read_csv(caminho)
    logger.info(
        "Dataset carregado via CSV",
        extra={"linhas": len(df), "colunas": df.shape[1], "arquivo": str(caminho), "rota": "csv"},
    )
    return df


def carregar_dados_brutos(config: Config, *, permitir_fallback: bool = True) -> pd.DataFrame:
    """Carrega o dataset bruto conforme a configuração.

    Quando ``loader='csv'`` e o arquivo não existe, cai para a rota do
    scikit-learn — mas apenas para o dataset de câncer de mama, que é o único
    com equivalente embutido. Para qualquer outro dataset o erro é propagado,
    porque um fallback silencioso para a base errada seria pior que falhar.

    Args:
        config: Configuração carregada de ``configs/*.yaml``.
        permitir_fallback: Desative em testes que precisam garantir qual rota
            foi exercitada.
    """
    if config.dataset.loader == "sklearn_breast_cancer":
        return carregar_breast_cancer_sklearn()

    caminho = config.dataset.caminho_csv
    assert caminho is not None  # garantido pelo validador de ConfigDataset

    caminho_absoluto = caminho if caminho.is_absolute() else RAIZ_PROJETO / caminho
    if caminho_absoluto.exists():
        return carregar_csv(caminho_absoluto)

    if permitir_fallback and config.dataset.nome == "breast_cancer_wisconsin":
        logger.warning(
            "CSV ausente; usando o carregador do scikit-learn (mesma origem UCI). "
            "Para exercitar a limpeza das colunas 'id' e 'Unnamed: 32', baixe o CSV "
            "do Kaggle com: make data",
            extra={"caminho_esperado": str(caminho_absoluto)},
        )
        return carregar_breast_cancer_sklearn()

    raise FileNotFoundError(
        f"CSV não encontrado em {caminho_absoluto} e não há fallback para "
        f"o dataset '{config.dataset.nome}'. Baixe-o com: make data CONFIG=<seu.yaml>"
    )
