"""Avaliação e correção da calibração das probabilidades.

A UI exibe "87% de probabilidade de malignidade". Essa afirmação só é honesta
se, entre todos os casos a que o modelo atribui 87%, aproximadamente 87%
forem de fato malignos. Um modelo pode ter ROC-AUC excelente — ordenar os
casos corretamente — e ainda assim produzir probabilidades sistematicamente
exageradas: ele acerta *quem* é mais suspeito, mas erra *quanto*.

Isso não é detalhe estético. A zona cinzenta é definida por faixas de
probabilidade; se as probabilidades estiverem deslocadas, a faixa encaminha os
casos errados. E um clínico que lê "95%" de um modelo descalibrado recebe uma
falsa sensação de certeza — exatamente o mecanismo do automation bias.

Boosting e SVM costumam precisar de correção; a regressão logística
normalmente já sai calibrada, por otimizar diretamente a verossimilhança.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)

# Acima deste desvio médio absoluto entre probabilidade prevista e frequência
# observada, consideramos que vale corrigir. 5 pontos percentuais é o ponto a
# partir do qual a diferença passa a ser clinicamente perceptível na leitura.
TOLERANCIA_CALIBRACAO = 0.05


@dataclass(frozen=True)
class DiagnosticoCalibracao:
    """Quão próximas as probabilidades estão da frequência real."""

    brier: float
    erro_calibracao_medio: float
    erro_calibracao_maximo: float
    probabilidades_previstas: np.ndarray
    frequencias_observadas: np.ndarray
    precisa_calibrar: bool

    def __str__(self) -> str:
        veredito = "recomendada" if self.precisa_calibrar else "desnecessária"
        return (
            f"Calibração {veredito}\n"
            f"  Brier score:            {self.brier:.4f}  (menor é melhor)\n"
            f"  erro médio absoluto:    {self.erro_calibracao_medio:.4f}\n"
            f"  erro máximo por faixa:  {self.erro_calibracao_maximo:.4f}"
        )

    def para_tabela(self) -> pd.DataFrame:
        """Faixas de probabilidade versus frequência observada."""
        return pd.DataFrame(
            {
                "probabilidade_prevista": self.probabilidades_previstas.round(4),
                "frequencia_observada": self.frequencias_observadas.round(4),
                "desvio": (self.frequencias_observadas - self.probabilidades_previstas).round(4),
            }
        )


def diagnosticar(
    y_verdadeiro: np.ndarray | pd.Series,
    y_probabilidade: np.ndarray | pd.Series,
    n_faixas: int = 10,
) -> DiagnosticoCalibracao:
    """Mede o desvio entre probabilidade prevista e frequência observada.

    Usa faixas de quantil (``strategy='quantile'``) em vez de faixas
    uniformes: com 114 amostras de validação, faixas uniformes deixariam
    várias praticamente vazias, e a estimativa de frequência nelas seria
    puro ruído.
    """
    y_verdadeiro = np.asarray(y_verdadeiro)
    y_probabilidade = np.asarray(y_probabilidade)

    frequencias, previstas = calibration_curve(
        y_verdadeiro, y_probabilidade, n_bins=n_faixas, strategy="quantile"
    )
    desvios = np.abs(frequencias - previstas)

    erro_medio = float(np.mean(desvios))
    diagnostico = DiagnosticoCalibracao(
        brier=float(brier_score_loss(y_verdadeiro, y_probabilidade)),
        erro_calibracao_medio=erro_medio,
        erro_calibracao_maximo=float(np.max(desvios)),
        probabilidades_previstas=previstas,
        frequencias_observadas=frequencias,
        precisa_calibrar=erro_medio > TOLERANCIA_CALIBRACAO,
    )
    logger.info(
        "Calibração diagnosticada",
        extra={
            "brier": round(diagnostico.brier, 4),
            "erro_medio": round(erro_medio, 4),
            "precisa_calibrar": diagnostico.precisa_calibrar,
        },
    )
    return diagnostico


def calibrar(
    estimador: BaseEstimator,
    x_treino: pd.DataFrame,
    y_treino: pd.Series,
    config: Config,
    metodo: str = "isotonic",
) -> BaseEstimator:
    """Devolve uma versão calibrada do estimador, reajustada no treino.

    **Sobre a escolha de ``cv`` em vez de ``prefit``.** O scikit-learn permite
    calibrar um modelo já treinado usando um conjunto reservado — seria
    natural usar a validação aqui. Não fazemos isso: a validação também
    escolhe o limiar de decisão, e usá-la para as duas coisas tornaria o
    limiar otimista, ajustado a um ruído que ela mesma ajudou a produzir.

    Com ``cv=5``, a calibração acontece inteiramente dentro do treino por
    validação cruzada interna, e a validação permanece limpa para medir o
    efeito e escolher o limiar.

    Args:
        metodo: ``isotonic`` é não paramétrico e mais flexível, mas precisa de
            dados; ``sigmoid`` (Platt) é paramétrico e mais estável em amostras
            pequenas. Com 341 amostras de treino, ``isotonic`` ainda é viável,
            mas a escolha deve ser verificada empiricamente.
    """
    calibrado = CalibratedClassifierCV(
        estimator=clone(estimador),
        method=metodo,
        cv=config.split.n_folds_cv,
        n_jobs=-1,
    )
    calibrado.fit(x_treino, y_treino)
    logger.info("Estimador calibrado", extra={"metodo": metodo, "cv": config.split.n_folds_cv})
    return calibrado


def comparar_metodos(
    estimador: BaseEstimator,
    x_treino: pd.DataFrame,
    y_treino: pd.Series,
    x_validacao: pd.DataFrame,
    y_validacao: pd.Series,
    config: Config,
) -> tuple[pd.DataFrame, BaseEstimator | None]:
    """Compara o modelo original com as duas estratégias de calibração.

    Returns:
        A tabela comparativa e o melhor estimador calibrado — ou ``None`` se
        a calibração não melhorar o Brier score, caso em que o original é
        preferível por ser mais simples e mais rápido.
    """
    probabilidade_original = estimador.predict_proba(x_validacao)[:, 1]
    diagnostico_original = diagnosticar(y_validacao, probabilidade_original)

    linhas = [
        {
            "metodo": "nenhum (original)",
            "brier": diagnostico_original.brier,
            "erro_medio": diagnostico_original.erro_calibracao_medio,
            "erro_maximo": diagnostico_original.erro_calibracao_maximo,
        }
    ]

    candidatos: dict[str, BaseEstimator] = {}
    for metodo in ("sigmoid", "isotonic"):
        modelo = calibrar(estimador, x_treino, y_treino, config, metodo=metodo)
        probabilidade = modelo.predict_proba(x_validacao)[:, 1]
        diagnostico = diagnosticar(y_validacao, probabilidade)
        candidatos[metodo] = modelo
        linhas.append(
            {
                "metodo": metodo,
                "brier": diagnostico.brier,
                "erro_medio": diagnostico.erro_calibracao_medio,
                "erro_maximo": diagnostico.erro_calibracao_maximo,
            }
        )

    tabela = pd.DataFrame(linhas).sort_values("brier").reset_index(drop=True)
    melhor_metodo = tabela.iloc[0]["metodo"]

    if melhor_metodo == "nenhum (original)":
        logger.info("Calibração não melhorou o Brier; mantendo o modelo original")
        return tabela, None

    return tabela, candidatos[melhor_metodo]
