"""Construção do pipeline de pré-processamento.

O pipeline é a **fronteira contra vazamento**: tudo que estima algo a partir
dos dados (mediana da imputação, média e desvio do escalonamento, categorias
do encoder) vive aqui dentro, e o ``fit`` acontece exclusivamente sobre o
conjunto de treino. Dentro da validação cruzada, o scikit-learn reajusta o
pipeline inteiro a cada fold — o que não aconteceria se o escalonamento
tivesse sido aplicado antes, manualmente.

Uma segunda razão, igualmente prática: o objeto serializado para a API é este
pipeline completo, não só o estimador. Salvar apenas o modelo obrigaria a
reimplementar o pré-processamento no servidor, e qualquer divergência sutil
— ordem de colunas, parâmetros do scaler — produziria predições erradas sem
nenhum erro aparente (ADR-005).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)


def separar_tipos(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa colunas numéricas de categóricas."""
    numericas = x.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in x.columns if c not in numericas]
    return numericas, categoricas


def construir_preprocessador(x: pd.DataFrame, config: Config) -> ColumnTransformer:
    """Monta o ``ColumnTransformer`` com um ramo por tipo de variável.

    O dataset principal é inteiramente numérico, mas o ramo categórico existe
    por duas razões: cumpre o requisito de converter variáveis categóricas e
    numéricas em formatos adequados, e permite que o mesmo pipeline receba
    outro dataset sem alteração de código.

    **Imputação mesmo sem ausentes.** O Breast Cancer Wisconsin não tem
    valores faltantes, mas o imputador permanece no pipeline porque a API
    receberá payloads incompletos em produção, e porque o Pima Diabetes tem
    ausentes disfarçados de zero. Um pipeline que só funciona com dados
    perfeitos não serve para servir.
    """
    numericas, categoricas = separar_tipos(x)

    ramo_numerico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy=config.limpeza.estrategia_imputacao_numerica)),
            # Obrigatório para KNN (distância euclidiana) e para a
            # convergência da regressão logística: as features deste dataset
            # diferem em mais de seis ordens de magnitude.
            ("escalonador", StandardScaler()),
        ]
    )

    ramo_categorico = Pipeline(
        steps=[
            (
                "imputador",
                SimpleImputer(strategy=config.limpeza.estrategia_imputacao_categorica),
            ),
            # handle_unknown='ignore': uma categoria nunca vista em produção
            # vira um vetor de zeros em vez de derrubar a API.
            ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformadores = []
    if numericas:
        transformadores.append(("numericas", ramo_numerico, numericas))
    if categoricas:
        transformadores.append(("categoricas", ramo_categorico, categoricas))

    if not transformadores:
        raise ValueError("Nenhuma coluna para transformar")

    logger.info(
        "Pré-processador construído",
        extra={"n_numericas": len(numericas), "n_categoricas": len(categoricas)},
    )
    return ColumnTransformer(
        transformers=transformadores,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construir_pipeline(estimador: BaseEstimator, x: pd.DataFrame, config: Config) -> Pipeline:
    """Encadeia pré-processamento e estimador em um objeto único e serializável."""
    return Pipeline(
        steps=[
            ("preprocessamento", construir_preprocessador(x, config)),
            ("modelo", estimador),
        ]
    )


def nomes_features_saida(preprocessador: ColumnTransformer) -> list[str]:
    """Nomes das features após a transformação, na ordem em que o modelo as vê.

    Essencial para a interpretabilidade: o SHAP devolve uma matriz de
    contribuições cuja ordem de colunas precisa casar com estes nomes, ou a
    explicação atribui o efeito à feature errada.
    """
    return list(preprocessador.get_feature_names_out())
