"""Figuras de avaliação e interpretabilidade.

Separadas de :mod:`medai.viz`, que cuida da exploração de dados. Todas as
funções devolvem um ``Figure`` e não chamam ``plt.show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

from medai.viz import COR_BENIGNO, COR_MALIGNO

# Consequência clínica de cada quadrante — o que realmente importa comunicar.
ROTULOS_MATRIZ = np.array(
    [
        ["Verdadeiro negativo\n(benigno identificado)", "Falso positivo\n(revisão desnecessária)"],
        ["Falso negativo\n(CÂNCER NÃO SINALIZADO)", "Verdadeiro positivo\n(caso priorizado)"],
    ]
)


def plot_matriz_confusao(
    y_verdadeiro: np.ndarray, y_predito: np.ndarray, *, titulo: str = "Matriz de confusão"
) -> plt.Figure:
    """Matriz de confusão com os quadrantes traduzidos para consequência clínica.

    Números sozinhos escondem a assimetria: 2 falsos negativos e 2 falsos
    positivos parecem equivalentes na matriz, e não são.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, normalizar, subtitulo in [
        (eixos[0], None, "Contagens absolutas"),
        (eixos[1], "true", "Normalizada por classe real"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y_verdadeiro,
            y_predito,
            display_labels=["Benigno", "Maligno"],
            cmap="Blues",
            normalize=normalizar,
            colorbar=False,
            values_format=".2f" if normalizar else "d",
            ax=ax,
        )
        ax.set_title(subtitulo, fontsize=11)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    # Anota a consequência clínica sobre a matriz absoluta.
    for i in range(2):
        for j in range(2):
            eixos[0].text(
                j,
                i + 0.30,
                ROTULOS_MATRIZ[i, j],
                ha="center",
                va="center",
                fontsize=7.5,
                color="#444444",
            )

    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_curvas_desempenho(
    y_verdadeiro: np.ndarray,
    y_probabilidade: np.ndarray,
    *,
    limiar: float,
    taxa_positivos: float,
) -> plt.Figure:
    """ROC, Precision-Recall e calibração lado a lado.

    A curva PR é a mais informativa sob desbalanceamento: a ROC se beneficia
    da abundância de verdadeiros negativos e parece ótima mesmo quando a
    precisão é ruim.
    """
    fig, eixos = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- ROC ---------------------------------------------------------------
    fpr, tpr, limiares_roc = roc_curve(y_verdadeiro, y_probabilidade)
    area_roc = auc(fpr, tpr)
    eixos[0].plot(fpr, tpr, color=COR_MALIGNO, linewidth=2, label=f"ROC (AUC = {area_roc:.3f})")
    eixos[0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="aleatório")

    indice = int(np.argmin(np.abs(limiares_roc - limiar)))
    eixos[0].scatter(
        fpr[indice],
        tpr[indice],
        s=90,
        color="#333333",
        zorder=5,
        label=f"ponto de operação ({limiar:.3f})",
    )
    eixos[0].set_xlabel("Taxa de falsos positivos")
    eixos[0].set_ylabel("Taxa de verdadeiros positivos (recall)")
    eixos[0].set_title("Curva ROC")
    eixos[0].legend(loc="lower right", fontsize=8)

    # --- Precision-Recall --------------------------------------------------
    precisao, recall, limiares_pr = precision_recall_curve(y_verdadeiro, y_probabilidade)
    area_pr = auc(recall, precisao)
    eixos[1].plot(
        recall, precisao, color=COR_MALIGNO, linewidth=2, label=f"PR (AUC = {area_pr:.3f})"
    )
    eixos[1].axhline(
        taxa_positivos,
        linestyle="--",
        color="#999999",
        linewidth=1,
        label=f"aleatório ({taxa_positivos:.3f})",
    )
    if len(limiares_pr):
        indice_pr = int(np.argmin(np.abs(limiares_pr - limiar)))
        eixos[1].scatter(recall[indice_pr], precisao[indice_pr], s=90, color="#333333", zorder=5)
    eixos[1].set_xlabel("Recall")
    eixos[1].set_ylabel("Precisão")
    eixos[1].set_title("Curva Precision-Recall")
    eixos[1].legend(loc="lower left", fontsize=8)

    # --- Calibração --------------------------------------------------------
    observado, previsto = calibration_curve(
        y_verdadeiro, y_probabilidade, n_bins=8, strategy="quantile"
    )
    eixos[2].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="calibração perfeita")
    eixos[2].plot(previsto, observado, "o-", color=COR_MALIGNO, linewidth=2, label="modelo")
    eixos[2].set_xlabel("Probabilidade prevista")
    eixos[2].set_ylabel("Frequência observada")
    eixos[2].set_title("Curva de calibração")
    eixos[2].legend(loc="upper left", fontsize=8)

    for ax in eixos:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    return fig


def plot_comparacao_modelos(tabela: pd.DataFrame, *, metrica: str = "recall") -> plt.Figure:
    """Barras comparando os modelos no conjunto de teste."""
    dados = tabela.sort_values(metrica, ascending=True)
    cores = [COR_BENIGNO if m == "dummy" else COR_MALIGNO for m in dados.index]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(dados) + 2))
    barras = ax.barh(dados.index, dados[metrica], color=cores, edgecolor="white")
    for barra, valor in zip(barras, dados[metrica], strict=True):
        ax.text(
            valor + 0.012,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metrica)
    ax.set_title(f"Comparação no conjunto de teste — {metrica}")
    return fig


def plot_coeficientes(tabela: pd.DataFrame, *, top: int = 15) -> plt.Figure:
    """Coeficientes da regressão logística, com sinal."""
    dados = tabela.head(top).iloc[::-1]
    cores = [COR_MALIGNO if v > 0 else COR_BENIGNO for v in dados["coeficiente"]]

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(dados) + 1.8))
    ax.barh(dados.index, dados["coeficiente"], color=cores, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Coeficiente (log-odds por desvio padrão da feature)")
    ax.set_title("Coeficientes da regressão logística")
    ax.tick_params(labelsize=9)
    return fig


def plot_permutation_importance(
    tabela: pd.DataFrame, *, top: int = 15, metrica: str = "recall"
) -> plt.Figure:
    """Queda da métrica ao embaralhar cada feature, com barra de erro."""
    dados = tabela.head(top).iloc[::-1]
    cores = [COR_MALIGNO if s else "#BBBBBB" for s in dados["significativa"]]

    fig, ax = plt.subplots(figsize=(9, 0.35 * len(dados) + 1.8))
    ax.barh(
        dados.index,
        dados["queda_media"],
        xerr=dados["desvio"],
        color=cores,
        edgecolor="white",
        error_kw={"ecolor": "#666666", "elinewidth": 1},
    )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel(f"Queda em {metrica} ao embaralhar a feature")
    ax.set_title("Permutation importance (conjunto de teste)")
    ax.tick_params(labelsize=9)
    return fig


def plot_shap_global(importancia: pd.Series, *, top: int = 15) -> plt.Figure:
    """Importância SHAP média absoluta."""
    dados = importancia.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(dados) + 1.8))
    ax.barh(dados.index, dados.to_numpy(), color=COR_MALIGNO, edgecolor="white")
    ax.set_xlabel("|SHAP| médio (log-odds)")
    ax.set_title("Importância SHAP global")
    ax.tick_params(labelsize=9)
    return fig


def plot_shap_agrupado(agrupado: pd.Series) -> plt.Figure:
    """Importância somada por grandeza medida.

    Antídoto visual contra a diluição: features redundantes dividem crédito,
    e só a soma revela o peso real do grupo.
    """
    dados = agrupado.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(dados) + 1.6))
    barras = ax.barh(dados.index, dados.to_numpy(), color=COR_MALIGNO, edgecolor="white")
    for barra, valor in zip(barras, dados.to_numpy(), strict=True):
        ax.text(
            valor + 0.05,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.2f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Soma do |SHAP| médio das features do grupo")
    ax.set_title("Importância agrupada por grandeza medida")
    ax.tick_params(labelsize=9)
    return fig


def plot_shap_beeswarm(explicacao, *, top: int = 15) -> plt.Figure:
    """Beeswarm: distribuição das contribuições, colorida pelo valor da feature."""
    import shap

    objeto = shap.Explanation(
        values=explicacao.valores,
        base_values=np.full(len(explicacao.valores), explicacao.valor_base),
        data=explicacao.dados.to_numpy(),
        feature_names=explicacao.nomes,
    )
    fig = plt.figure(figsize=(10, 0.42 * top + 2))
    shap.plots.beeswarm(objeto, max_display=top, show=False)
    plt.title("Contribuições SHAP por amostra (conjunto de teste)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_shap_caso(explicacao, indice: int, *, titulo: str, top: int = 10) -> plt.Figure:
    """Contribuições de um caso individual — o que a UI mostra ao clínico."""
    tabela = explicacao.contribuicoes_do_caso(indice, top=top).iloc[::-1]
    cores = [COR_MALIGNO if v > 0 else COR_BENIGNO for v in tabela["contribuicao"]]

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(tabela) + 2))
    ax.barh(tabela.index, tabela["contribuicao"], color=cores, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.9)

    for y, (contribuicao, medido) in enumerate(
        zip(tabela["contribuicao"], tabela["valor_medido"], strict=True)
    ):
        deslocamento = 0.02 if contribuicao > 0 else -0.02
        ax.text(
            contribuicao + deslocamento,
            y,
            f"{medido:.3g}",
            va="center",
            ha="left" if contribuicao > 0 else "right",
            fontsize=8,
            color="#555555",
        )

    ax.set_xlabel("Contribuição SHAP (log-odds)")
    ax.set_title(titulo)
    ax.tick_params(labelsize=9)
    margem = np.abs(tabela["contribuicao"]).max() * 0.35
    ax.set_xlim(tabela["contribuicao"].min() - margem, tabela["contribuicao"].max() + margem)
    return fig


def plot_dependencia(explicacao, feature: str) -> plt.Figure:
    """Contribuição SHAP versus valor medido de uma feature."""
    indice = explicacao.nomes.index(feature)
    valores_x = explicacao.dados_originais[feature].to_numpy()
    valores_y = explicacao.valores[:, indice]

    fig, ax = plt.subplots(figsize=(7, 5))
    dispersao = ax.scatter(
        valores_x,
        valores_y,
        c=valores_y,
        cmap="RdBu_r",
        s=42,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axhline(0, color="#999999", linewidth=0.9, linestyle="--")
    ax.set_xlabel(f"{feature} (escala clínica original)")
    ax.set_ylabel("Contribuição SHAP (log-odds)")
    ax.set_title(f"Dependência — {feature}")
    fig.colorbar(dispersao, ax=ax, label="contribuição")
    return fig


def plot_robustez(
    tabela: pd.DataFrame, *, metricas: tuple[str, ...] = ("recall", "f1", "roc_auc")
) -> plt.Figure:
    """Dispersão das métricas sob diferentes sementes de particionamento.

    A largura destas caixas é a resposta honesta a "quão confiável é o
    número reportado": diferenças entre modelos menores que esta dispersão
    não são distinguíveis de ruído de amostragem.
    """
    longo = tabela[list(metricas)].melt(var_name="metrica", value_name="valor")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=longo, x="metrica", y="valor", color=COR_MALIGNO, width=0.5, ax=ax)
    sns.stripplot(data=longo, x="metrica", y="valor", color="#333333", size=7, alpha=0.8, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Valor no conjunto de teste")
    ax.set_title(f"Robustez a {len(tabela)} sementes de particionamento")
    return fig
