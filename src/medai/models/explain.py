"""Interpretabilidade: importância de features e valores SHAP.

Três camadas, que respondem a perguntas diferentes:

1. **Coeficientes** — o que o modelo linear aprendeu, em log-odds. Direto,
   mas instável sob multicolinearidade (o VIF deste dataset passa de 3.900).
2. **Permutation importance** — quanto a métrica piora ao embaralhar cada
   feature. Mede o que o modelo *usa*, não o que ele *declara*, e é medida
   no conjunto de teste.
3. **SHAP** — contribuição de cada feature para cada predição individual.
   É o que a UI mostra ao clínico.

**A armadilha presente em todas as três.** Com `radius`, `perimeter` e `area`
correlacionadas acima de 0,98, o crédito se divide entre elas. Cada uma parece
menos importante do que o grupo é. Interpretar a importância individual sem
levar isso em conta levaria à conclusão errada de que o tamanho do núcleo é
pouco relevante — quando ele é, somado, um dos sinais mais fortes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from medai.logging_conf import get_logger

logger = get_logger(__name__)


def separar_pipeline(pipeline: Pipeline) -> tuple[BaseEstimator, BaseEstimator]:
    """Devolve (pré-processador, estimador) do pipeline serializado."""
    return pipeline.named_steps["preprocessamento"], pipeline.named_steps["modelo"]


def nomes_features(pipeline: Pipeline) -> list[str]:
    """Nomes na ordem em que o estimador os vê, após a transformação."""
    preprocessador, _ = separar_pipeline(pipeline)
    return list(preprocessador.get_feature_names_out())


# -----------------------------------------------------------------------------
# 1. Coeficientes
# -----------------------------------------------------------------------------


def coeficientes(pipeline: Pipeline) -> pd.DataFrame:
    """Coeficientes da regressão logística, com razão de chances.

    Como as features foram padronizadas, cada coeficiente é o efeito de **um
    desvio padrão** daquela medida sobre o log-odds de malignidade — o que
    torna as magnitudes comparáveis entre features de unidades diferentes.

    A razão de chances (``exp(coef)``) é a leitura clinicamente legível:
    "cada desvio padrão a mais nesta medida multiplica a chance de
    malignidade por X".
    """
    _, estimador = separar_pipeline(pipeline)
    if not hasattr(estimador, "coef_"):
        raise TypeError(f"{type(estimador).__name__} não expõe coeficientes")

    valores = np.ravel(estimador.coef_)
    tabela = pd.DataFrame(
        {
            "feature": nomes_features(pipeline),
            "coeficiente": valores,
            "razao_de_chances": np.exp(valores),
            "abs_coeficiente": np.abs(valores),
        }
    ).set_index("feature")

    tabela["direcao"] = np.where(tabela["coeficiente"] > 0, "aumenta suspeição", "reduz suspeição")
    return tabela.sort_values("abs_coeficiente", ascending=False)


# -----------------------------------------------------------------------------
# 2. Permutation importance
# -----------------------------------------------------------------------------


def importancia_por_permutacao(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "recall",
    n_repeticoes: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Queda da métrica ao embaralhar cada feature, medida no conjunto dado.

    Preferível à importância nativa de árvores, que é enviesada para features
    de alta cardinalidade e é calculada no treino. Aqui medimos no teste, com
    a métrica que de fato interessa clinicamente.

    **Leitura sob multicolinearidade:** embaralhar `radius_mean` quase não
    piora a métrica, porque `perimeter_mean` carrega a mesma informação e
    permanece intacta. Importância baixa aqui significa "redundante", não
    "irrelevante" — distinção que muda completamente a conclusão.
    """
    resultado = permutation_importance(
        pipeline,
        x,
        y,
        scoring=scoring,
        n_repeats=n_repeticoes,
        random_state=random_state,
        n_jobs=-1,
    )
    tabela = pd.DataFrame(
        {
            "feature": x.columns,
            "queda_media": resultado.importances_mean,
            "desvio": resultado.importances_std,
        }
    ).set_index("feature")

    # Uma feature cuja queda média é menor que seu próprio desvio não tem
    # efeito distinguível de zero.
    tabela["significativa"] = tabela["queda_media"] > tabela["desvio"]
    logger.info(
        "Permutation importance calculada",
        extra={"scoring": scoring, "n_significativas": int(tabela["significativa"].sum())},
    )
    return tabela.sort_values("queda_media", ascending=False)


# -----------------------------------------------------------------------------
# 3. SHAP
# -----------------------------------------------------------------------------


@dataclass
class ExplicacaoSHAP:
    """Valores SHAP prontos para plotagem e para a API."""

    valores: np.ndarray  # (n_amostras, n_features), em log-odds
    valor_base: float
    dados: pd.DataFrame  # features transformadas, na escala padronizada
    dados_originais: pd.DataFrame  # features na escala clínica original
    nomes: list[str]

    @property
    def importancia_global(self) -> pd.Series:
        """Média do valor absoluto por feature — a leitura global padrão."""
        return (
            pd.Series(np.abs(self.valores).mean(axis=0), index=self.nomes)
            .sort_values(ascending=False)
            .rename("shap_medio_absoluto")
        )

    def contribuicoes_do_caso(self, indice: int, top: int = 10) -> pd.DataFrame:
        """As features que mais pesaram em uma predição individual."""
        serie = pd.Series(self.valores[indice], index=self.nomes)
        tabela = pd.DataFrame(
            {
                "contribuicao": serie,
                "valor_medido": self.dados_originais.iloc[indice].reindex(self.nomes),
            }
        )
        tabela["efeito"] = np.where(tabela["contribuicao"] > 0, "↑ suspeição", "↓ suspeição")
        return tabela.reindex(serie.abs().sort_values(ascending=False).index).head(top)


def explicar(
    pipeline: Pipeline, x: pd.DataFrame, x_fundo: pd.DataFrame | None = None
) -> ExplicacaoSHAP:
    """Calcula os valores SHAP das amostras fornecidas.

    O explicador é escolhido pela família do estimador: ``LinearExplainer``
    para modelos lineares (exato e barato), ``TreeExplainer`` para árvores e
    boosting. ``KernelExplainer`` ficaria para os demais, mas é caro o
    bastante para exigir amostragem — e nenhum campeão atual precisa dele.

    **Os valores saem em log-odds**, não em probabilidade. Uma contribuição
    de +2,0 significa que aquela feature multiplicou a chance de malignidade
    por ``e²`` ≈ 7,4. Apresentar isso como "pontos percentuais" seria
    incorreto, porque o efeito sobre a probabilidade depende de onde o caso
    está na curva logística.

    Args:
        x_fundo: Amostra de referência contra a qual as contribuições são
            medidas. Sem ela, usa ``x``. Em produção deve ser o conjunto de
            treino, não o lote que chega — do contrário a explicação de um
            paciente mudaria conforme quem mais está na fila.
    """
    import shap

    preprocessador, estimador = separar_pipeline(pipeline)
    nomes = nomes_features(pipeline)

    transformado = pd.DataFrame(preprocessador.transform(x), columns=nomes, index=x.index)
    fundo = (
        transformado
        if x_fundo is None
        else pd.DataFrame(preprocessador.transform(x_fundo), columns=nomes, index=x_fundo.index)
    )

    tipo = type(estimador).__name__
    if hasattr(estimador, "coef_"):
        explicador = shap.LinearExplainer(estimador, fundo)
    elif tipo in {"XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier"}:
        explicador = shap.TreeExplainer(estimador, fundo)
    else:  # pragma: no cover - nenhum campeão atual cai aqui
        explicador = shap.KernelExplainer(estimador.predict_proba, shap.sample(fundo, 100))

    explicacao = explicador(transformado)
    valores = np.asarray(explicacao.values)
    base = np.asarray(explicacao.base_values)

    # TreeExplainer de classificador binário pode devolver (n, f, 2):
    # mantemos apenas a classe positiva.
    if valores.ndim == 3:
        valores = valores[:, :, 1]
        base = base[:, 1] if base.ndim > 1 else base

    logger.info(
        "SHAP calculado",
        extra={"explicador": type(explicador).__name__, "n_amostras": len(x)},
    )
    return ExplicacaoSHAP(
        valores=valores,
        valor_base=float(np.mean(base)),
        dados=transformado,
        dados_originais=x,
        nomes=nomes,
    )


def agrupar_por_medida(importancia: pd.Series) -> pd.Series:
    """Soma a importância das features que medem a mesma grandeza.

    Antídoto direto contra a diluição por multicolinearidade: `radius_mean`,
    `perimeter_mean` e `area_mean` medem tamanho do núcleo, e somar suas
    contribuições revela o peso real do grupo, que a leitura individual
    esconde.
    """
    grupos = {
        "tamanho (radius/perimeter/area)": ("radius", "perimeter", "area"),
        "irregularidade (concavity/compactness/concave points)": (
            "concavity",
            "compactness",
            "concave points",
        ),
        "textura": ("texture",),
        "suavidade": ("smoothness",),
        "simetria": ("symmetry",),
        "dimensão fractal": ("fractal_dimension",),
    }
    somas = {}
    for rotulo, prefixos in grupos.items():
        selecao = [f for f in importancia.index if any(f.startswith(p) for p in prefixos)]
        somas[rotulo] = float(importancia[selecao].sum())
    return pd.Series(somas).sort_values(ascending=False).rename("importancia_agrupada")
