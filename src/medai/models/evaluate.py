"""Métricas de classificação e escolha do limiar de decisão.

A seleção do limiar é uma decisão clínica disfarçada de parâmetro técnico. O
corte padrão de 0,5 presume que errar para mais e errar para menos custam o
mesmo — premissa falsa em oncologia, onde um falso negativo é um câncer não
sinalizado e um falso positivo é um exame confirmatório a mais.

O limiar é escolhido no **conjunto de validação**, nunca no teste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)

# Rótulos das três faixas de decisão expostas ao usuário clínico.
FAIXA_BENIGNO = "benigno"
FAIXA_REVISAO = "zona_cinzenta_revisao_medica"
FAIXA_MALIGNO = "maligno"


def metricas_classificacao(
    y_verdadeiro: np.ndarray | pd.Series,
    y_predito: np.ndarray | pd.Series,
    y_probabilidade: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Calcula o conjunto de métricas usado em todo o projeto.

    A classe positiva é sempre 1 (maligno), de modo que ``recall`` sem
    qualificação significa sensibilidade para câncer.
    """
    resultado: dict[str, float] = {
        "accuracy": float(accuracy_score(y_verdadeiro, y_predito)),
        "precision": float(precision_score(y_verdadeiro, y_predito, zero_division=0)),
        "recall": float(recall_score(y_verdadeiro, y_predito, zero_division=0)),
        "f1": float(f1_score(y_verdadeiro, y_predito, zero_division=0)),
        "f1_macro": float(f1_score(y_verdadeiro, y_predito, average="macro", zero_division=0)),
    }

    matriz = confusion_matrix(y_verdadeiro, y_predito, labels=[0, 1])
    vn, fp, fn, vp = matriz.ravel()
    resultado.update(
        {
            "verdadeiros_negativos": int(vn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdadeiros_positivos": int(vp),
            # Especificidade: proporção de benignos corretamente identificados.
            "especificidade": float(vn / (vn + fp)) if (vn + fp) else 0.0,
        }
    )

    if y_probabilidade is not None:
        resultado["roc_auc"] = float(roc_auc_score(y_verdadeiro, y_probabilidade))
        # PR-AUC é mais informativa que ROC-AUC sob desbalanceamento, porque
        # não se beneficia da abundância de verdadeiros negativos.
        resultado["average_precision"] = float(
            average_precision_score(y_verdadeiro, y_probabilidade)
        )
        # Brier mede calibração: quão próximas as probabilidades estão da
        # frequência real. Menor é melhor.
        resultado["brier"] = float(brier_score_loss(y_verdadeiro, y_probabilidade))

    return resultado


@dataclass(frozen=True)
class LimiarEscolhido:
    """Resultado da busca pelo limiar de decisão."""

    limiar: float
    recall: float
    precision: float
    f1: float
    especificidade: float
    falsos_negativos: int
    falsos_positivos: int
    recall_alvo: float
    atingiu_alvo: bool

    def __str__(self) -> str:
        status = "atingido" if self.atingiu_alvo else "NÃO atingido"
        return (
            f"Limiar: {self.limiar:.3f} (alvo de recall {self.recall_alvo:.0%} {status})\n"
            f"  recall:         {self.recall:.3f}  ({self.falsos_negativos} falsos negativos)\n"
            f"  precision:      {self.precision:.3f}  ({self.falsos_positivos} falsos positivos)\n"
            f"  especificidade: {self.especificidade:.3f}\n"
            f"  F1:             {self.f1:.3f}"
        )


def escolher_limiar(
    y_verdadeiro: np.ndarray | pd.Series,
    y_probabilidade: np.ndarray | pd.Series,
    recall_minimo: float = 0.98,
    n_candidatos: int = 1001,
) -> LimiarEscolhido:
    """Menor custo em precisão que ainda atinge o recall mínimo exigido.

    Percorre os limiares candidatos, filtra os que atingem ``recall_minimo`` e
    escolhe, entre eles, o de maior precisão — isto é, o limiar **mais alto**
    que ainda não deixa passar casos malignos. Limiar mais alto significa menos
    falsos positivos, logo menos revisões desnecessárias.

    Se nenhum limiar atinge o alvo, devolve o de maior recall possível e sinaliza
    ``atingiu_alvo=False``. Falhar silenciosamente aqui produziria um sistema que
    promete sensibilidade que não entrega.
    """
    y_verdadeiro = np.asarray(y_verdadeiro)
    y_probabilidade = np.asarray(y_probabilidade)
    candidatos = np.linspace(0.0, 1.0, n_candidatos)

    linhas = []
    for limiar in candidatos:
        predito = (y_probabilidade >= limiar).astype(int)
        recall = recall_score(y_verdadeiro, predito, zero_division=0)
        precisao = precision_score(y_verdadeiro, predito, zero_division=0)
        linhas.append((limiar, recall, precisao))

    tabela = pd.DataFrame(linhas, columns=["limiar", "recall", "precision"])
    viaveis = tabela[tabela["recall"] >= recall_minimo]

    if viaveis.empty:
        melhor = tabela.loc[tabela["recall"].idxmax()]
        atingiu = False
        logger.warning(
            "Nenhum limiar atinge o recall mínimo",
            extra={"recall_minimo": recall_minimo, "recall_maximo": float(melhor["recall"])},
        )
    else:
        # Entre os viáveis, o de maior precisão é também o de maior limiar.
        melhor = viaveis.loc[viaveis["precision"].idxmax()]
        atingiu = True

    limiar = float(melhor["limiar"])
    predito = (y_probabilidade >= limiar).astype(int)
    metricas = metricas_classificacao(y_verdadeiro, predito)

    return LimiarEscolhido(
        limiar=limiar,
        recall=metricas["recall"],
        precision=metricas["precision"],
        f1=metricas["f1"],
        especificidade=metricas["especificidade"],
        falsos_negativos=int(metricas["falsos_negativos"]),
        falsos_positivos=int(metricas["falsos_positivos"]),
        recall_alvo=recall_minimo,
        atingiu_alvo=atingiu,
    )


def classificar_com_faixa(
    y_probabilidade: np.ndarray | pd.Series, limiar: float, config: Config
) -> np.ndarray:
    """Converte probabilidades em três faixas, não em rótulo binário.

    A faixa intermediária é a materialização em produto da exigência de que o
    diagnóstico final seja humano: onde o modelo não tem base para decidir, ele
    encaminha em vez de arriscar um rótulo.

    A zona cinzenta é definida pela configuração e **pode conter o limiar**.
    Quando isso acontece, casos acima do limiar mas dentro da zona são
    encaminhados para revisão prioritária em vez de rotulados como malignos —
    o que é conservador na direção certa.
    """
    inferior, superior = config.modelagem.zona_cinzenta
    p = np.asarray(y_probabilidade)

    faixas = np.where(p >= limiar, FAIXA_MALIGNO, FAIXA_BENIGNO).astype(object)
    faixas[(p >= inferior) & (p <= superior)] = FAIXA_REVISAO
    return faixas


def resumo_faixas(faixas: np.ndarray) -> dict[str, int]:
    """Contagem por faixa, para dimensionar a carga de revisão."""
    valores, contagens = np.unique(faixas, return_counts=True)
    return dict(zip(valores.tolist(), contagens.tolist(), strict=True))
