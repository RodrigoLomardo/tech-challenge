"""Análise de correlação e multicolinearidade.

Neste dataset a multicolinearidade não é um achado empírico, é uma
**consequência geométrica**: ``perimeter ≈ 2πr`` e ``area ≈ πr²``, de modo que
``radius``, ``perimeter`` e ``area`` medem a mesma grandeza em unidades
diferentes. O trio se repete nos três sufixos.

O efeito sobre a acurácia é pequeno — árvores escolhem uma feature e ignoram
as redundantes. O efeito sobre a **interpretação** é grande: coeficientes
instáveis na regressão logística e crédito SHAP diluído entre features
correlacionadas, de modo que cada uma parece menos importante do que o grupo
é. Ver ADR-009.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)


def matriz_correlacao(x: pd.DataFrame, metodo: str = "pearson") -> pd.DataFrame:
    """Matriz de correlação entre features.

    Pearson mede associação linear; Spearman mede associação monotônica e é
    robusto a assimetria — relevante aqui, onde ``area_se`` tem skew 5,45.
    Divergência grande entre as duas indica relação não linear.
    """
    return x.corr(method=metodo)


def pares_correlacionados(
    x: pd.DataFrame, limiar: float = 0.95, metodo: str = "pearson"
) -> pd.DataFrame:
    """Pares de features com correlação absoluta acima do limiar."""
    corr = matriz_correlacao(x, metodo).abs()
    triangulo_superior = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pares = corr.where(triangulo_superior).stack()
    acima = pares[pares >= limiar].sort_values(ascending=False)

    return pd.DataFrame(
        {
            "feature_a": [i[0] for i in acima.index],
            "feature_b": [i[1] for i in acima.index],
            "correlacao": acima.to_numpy().round(4),
        }
    )


def correlacao_com_alvo(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Correlação ponto-bisserial entre cada feature e o alvo binário.

    É o Pearson entre uma contínua e uma dicotômica. Usamos como critério de
    desempate na seleção de features: entre duas redundantes, preservamos a
    que carrega mais informação sobre o desfecho.
    """
    registros = []
    for coluna in x.select_dtypes(include="number").columns:
        serie = x[coluna]
        valido = serie.notna()
        if valido.sum() < 3 or serie[valido].nunique() < 2:
            registros.append({"feature": coluna, "r": 0.0, "p_valor": 1.0})
            continue
        r, p = stats.pointbiserialr(y[valido], serie[valido])
        registros.append({"feature": coluna, "r": round(float(r), 4), "p_valor": float(p)})

    resultado = pd.DataFrame(registros).set_index("feature")
    resultado["abs_r"] = resultado["r"].abs()
    return resultado.sort_values("abs_r", ascending=False)


def calcular_vif(x: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor por feature.

    VIF mede quanto a variância do coeficiente de uma feature é inflada pela
    colinearidade com as demais. Convenção usual: acima de 5 é preocupante,
    acima de 10 é severo. Diferente da correlação par a par, o VIF capta
    colinearidade **múltipla** — uma feature pode ter correlação moderada com
    cada uma das outras e ainda assim ser quase uma combinação linear delas.

    Os dados são padronizados antes do cálculo, para que a constante não
    domine o resultado.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    numericas = x.select_dtypes(include="number")
    # Mediana em vez de média: coerente com a estratégia de imputação do
    # pipeline e menos sensível às caudas longas das features `_se`.
    preenchido = numericas.fillna(numericas.median())
    padronizado = (preenchido - preenchido.mean()) / preenchido.std().replace(0, 1)
    com_constante = add_constant(padronizado, has_constant="add")

    valores = []
    for i, coluna in enumerate(com_constante.columns):
        if coluna == "const":
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            vif = variance_inflation_factor(com_constante.to_numpy(), i)
        valores.append({"feature": coluna, "vif": float(vif)})

    resultado = pd.DataFrame(valores).set_index("feature").sort_values("vif", ascending=False)
    resultado["severidade"] = pd.cut(
        resultado["vif"],
        bins=[-np.inf, 5, 10, np.inf],
        labels=["aceitável", "preocupante", "severo"],
    )
    return resultado


@dataclass
class SelecaoFeatures:
    """Resultado da redução por redundância."""

    mantidas: list[str] = field(default_factory=list)
    removidas: list[str] = field(default_factory=list)
    # Para cada removida, com qual feature ela era redundante e o r do par.
    motivo: dict[str, tuple[str, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        linhas = [
            f"Feature set reduzido: {len(self.mantidas)} mantidas, {len(self.removidas)} removidas"
        ]
        for feature, (par, r) in self.motivo.items():
            linhas.append(f"  − {feature}  (|r| = {r:.3f} com {par})")
        return "\n".join(linhas)


def selecionar_por_redundancia(
    x: pd.DataFrame, y: pd.Series, config: Config, metodo: str = "pearson"
) -> SelecaoFeatures:
    """Remove features redundantes, preservando a mais informativa de cada par.

    Estratégia gulosa: percorre os pares acima do limiar em ordem decrescente
    de correlação e, em cada um, descarta a feature com menor correlação
    absoluta com o alvo. Preservar a mais informativa, em vez de escolher pela
    ordem alfabética ou pela posição da coluna, é o que torna o conjunto
    reduzido comparável ao completo em desempenho.

    O resultado alimenta a comparação entre feature sets prevista no ADR-009.
    """
    limiar = config.correlacao.limiar_redundancia
    pares = pares_correlacionados(x, limiar=limiar, metodo=metodo)
    informatividade = correlacao_com_alvo(x, y)["abs_r"]

    removidas: set[str] = set()
    selecao = SelecaoFeatures()

    for _, linha in pares.iterrows():
        a, b = linha["feature_a"], linha["feature_b"]
        if a in removidas or b in removidas:
            continue  # o par já foi desfeito por uma remoção anterior
        descartar = a if informatividade.get(a, 0) < informatividade.get(b, 0) else b
        manter = b if descartar == a else a
        removidas.add(descartar)
        selecao.motivo[descartar] = (manter, float(linha["correlacao"]))

    selecao.removidas = sorted(removidas)
    selecao.mantidas = [c for c in x.columns if c not in removidas]

    logger.info(
        "Seleção por redundância",
        extra={
            "limiar": limiar,
            "mantidas": len(selecao.mantidas),
            "removidas": len(selecao.removidas),
        },
    )
    return selecao
