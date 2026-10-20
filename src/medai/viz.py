"""Geração das figuras do relatório.

Todas as funções devolvem um ``matplotlib.figure.Figure`` e **não** chamam
``plt.show()``: quem decide exibir é o notebook, quem decide salvar é o script
de assets. Isso mantém as mesmas figuras idênticas nos dois contextos.

Paleta: azul para benigno, laranja para maligno. O par é distinguível em
deuteranopia e protanopia, e sobrevive à impressão em escala de cinza — a
diferença de luminância entre os dois é suficiente. Vermelho/verde, a escolha
reflexa para "ruim/bom", falharia nos dois critérios.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)

COR_BENIGNO = "#4C72B0"
COR_MALIGNO = "#DD8452"
DPI = 150


def paleta_classes(config: Config) -> dict[str, str]:
    """Mapeia cada rótulo do alvo à sua cor, com a positiva em laranja."""
    positivo = str(config.dataset.valor_positivo)
    negativo = "B" if positivo == "M" else "0"
    return {positivo: COR_MALIGNO, negativo: COR_BENIGNO}


def configurar_estilo(interativo: bool = False) -> None:
    """Aplica o estilo visual do projeto.

    Args:
        interativo: ``False`` força o backend ``Agg`` (scripts e CI, sem
            display). Em notebook, passe ``True`` para preservar o backend
            inline.
    """
    if not interativo:
        matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


def salvar_figura(fig: plt.Figure, nome: str, config: Config, *, fechar: bool = True) -> Path:
    """Grava a figura em ``reports/figures/`` e devolve o caminho."""
    destino = config.caminhos.absoluto("figuras")
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.png"
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight")
    if fechar:
        plt.close(fig)
    logger.info("Figura salva", extra={"arquivo": caminho.name})
    return caminho


def _rotulo_legivel(valor: str, config: Config) -> str:
    """Traduz o rótulo cru para texto clínico."""
    if config.dataset.nome == "breast_cancer_wisconsin":
        return {"M": "Maligno", "B": "Benigno"}.get(str(valor), str(valor))
    positivo = str(config.dataset.valor_positivo)
    return "Positivo" if str(valor) == positivo else "Negativo"


# -----------------------------------------------------------------------------
# Distribuição do alvo
# -----------------------------------------------------------------------------


def plot_distribuicao_classes(df: pd.DataFrame, config: Config) -> plt.Figure:
    """Barras da distribuição do alvo, com a linha do classificador trivial.

    A linha tracejada marca a accuracy que se obtém respondendo sempre a
    classe majoritária. É o argumento visual mais direto contra usar accuracy
    como métrica de decisão neste problema.
    """
    alvo = config.dataset.coluna_alvo
    contagem = df[alvo].value_counts()
    cores = [paleta_classes(config).get(str(i), "#999999") for i in contagem.index]
    rotulos = [_rotulo_legivel(i, config) for i in contagem.index]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(rotulos, contagem.to_numpy(), color=cores, edgecolor="white", linewidth=1.5)

    for barra, valor in zip(barras, contagem.to_numpy(), strict=True):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + len(df) * 0.015,
            f"{valor}\n({valor / len(df):.1%})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    majoritaria = contagem.max()
    ax.axhline(
        majoritaria,
        linestyle="--",
        color="#555555",
        linewidth=1.2,
        label=f"Classificador trivial: {majoritaria / len(df):.1%} de accuracy",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylabel("Número de amostras")
    ax.set_ylim(0, majoritaria * 1.22)
    ax.set_title(f"Distribuição do diagnóstico (n = {len(df)})")
    return fig


# -----------------------------------------------------------------------------
# Distribuições das features
# -----------------------------------------------------------------------------


def plot_grade_distribuicoes(
    df: pd.DataFrame, config: Config, features: list[str], *, titulo: str, n_colunas: int = 5
) -> plt.Figure:
    """Grade de KDEs por classe, uma feature por painel."""
    alvo = config.dataset.coluna_alvo
    paleta = paleta_classes(config)
    n_linhas = int(np.ceil(len(features) / n_colunas))

    fig, eixos = plt.subplots(
        n_linhas, n_colunas, figsize=(3.2 * n_colunas, 2.6 * n_linhas), constrained_layout=True
    )
    eixos = np.atleast_1d(eixos).ravel()

    for ax, feature in zip(eixos, features, strict=False):
        for classe, cor in paleta.items():
            subconjunto = df.loc[df[alvo] == classe, feature]
            if subconjunto.empty:
                continue
            sns.kdeplot(
                subconjunto,
                ax=ax,
                color=cor,
                fill=True,
                alpha=0.4,
                linewidth=1.4,
                label=_rotulo_legivel(classe, config),
            )
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)

    for ax in eixos[len(features) :]:
        ax.set_visible(False)

    manipuladores, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(manipuladores, rotulos, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(titulo)
    return fig


def plot_boxplots(
    df: pd.DataFrame, config: Config, features: list[str], *, titulo: str
) -> plt.Figure:
    """Boxplots por classe, com os pontos atípicos destacados.

    Mostrar os outliers é intencional: a discussão sobre mantê-los depende de
    ver onde eles se concentram.
    """
    alvo = config.dataset.coluna_alvo
    n_colunas = min(4, len(features))
    n_linhas = int(np.ceil(len(features) / n_colunas))

    fig, eixos = plt.subplots(
        n_linhas, n_colunas, figsize=(3.4 * n_colunas, 3.2 * n_linhas), constrained_layout=True
    )
    eixos = np.atleast_1d(eixos).ravel()
    ordem = sorted(df[alvo].unique(), key=str)

    for ax, feature in zip(eixos, features, strict=False):
        sns.boxplot(
            data=df,
            x=alvo,
            y=feature,
            hue=alvo,
            order=ordem,
            palette=paleta_classes(config),
            legend=False,
            ax=ax,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(ordem)))
        ax.set_xticklabels([_rotulo_legivel(c, config) for c in ordem], fontsize=9)

    for ax in eixos[len(features) :]:
        ax.set_visible(False)

    fig.suptitle(titulo)
    return fig


def plot_violinos(
    df: pd.DataFrame, config: Config, features: list[str], *, titulo: str
) -> plt.Figure:
    """Violinos divididos por classe — densidade e sobreposição em um só gráfico."""
    alvo = config.dataset.coluna_alvo
    n_colunas = min(5, len(features))
    n_linhas = int(np.ceil(len(features) / n_colunas))

    fig, eixos = plt.subplots(
        n_linhas, n_colunas, figsize=(3.0 * n_colunas, 3.0 * n_linhas), constrained_layout=True
    )
    eixos = np.atleast_1d(eixos).ravel()
    ordem = sorted(df[alvo].unique(), key=str)

    for ax, feature in zip(eixos, features, strict=False):
        sns.violinplot(
            data=df,
            x=alvo,
            y=feature,
            hue=alvo,
            order=ordem,
            palette=paleta_classes(config),
            legend=False,
            inner="quartile",
            ax=ax,
        )
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(ordem)))
        ax.set_xticklabels([_rotulo_legivel(c, config) for c in ordem], fontsize=9)

    for ax in eixos[len(features) :]:
        ax.set_visible(False)

    fig.suptitle(titulo)
    return fig


def plot_pairplot(df: pd.DataFrame, config: Config, features: list[str]) -> plt.Figure:
    """Dispersão par a par de um subconjunto pequeno de features.

    Deliberadamente limitado a poucas features: um pairplot com as 30 geraria
    900 painéis ilegíveis. A escolha de quais incluir é analítica, não estética.
    """
    if len(features) > 8:
        raise ValueError(f"Pairplot com {len(features)} features é ilegível; use no máximo 8")

    alvo = config.dataset.coluna_alvo
    grade = sns.pairplot(
        df[[*features, alvo]],
        hue=alvo,
        palette=paleta_classes(config),
        diag_kind="kde",
        plot_kws={"alpha": 0.55, "s": 18, "edgecolor": "none"},
        height=1.9,
    )
    grade.figure.suptitle("Dispersão par a par — features de maior separabilidade", y=1.01)
    return grade.figure


# -----------------------------------------------------------------------------
# Estrutura entre features
# -----------------------------------------------------------------------------


def plot_heatmap_correlacao(
    df: pd.DataFrame, features: list[str], *, metodo: str = "pearson", titulo: str | None = None
) -> plt.Figure:
    """Heatmap triangular da matriz de correlação."""
    corr = df[features].corr(method=metodo)
    mascara = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr,
        mask=mascara,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.6, "label": f"Correlação de {metodo.capitalize()}"},
        ax=ax,
    )
    ax.set_title(titulo or f"Matriz de correlação ({metodo}) — {len(features)} features")
    ax.tick_params(labelsize=8)
    return fig


def plot_pca_2d(df: pd.DataFrame, config: Config, features: list[str]) -> plt.Figure:
    """Projeção PCA em duas dimensões, colorida por classe.

    Apenas visualização da separabilidade. O PCA é ajustado sobre o dataset
    inteiro de propósito: não há modelo sendo treinado aqui, e o objetivo é
    enxergar a estrutura dos dados. Usar componentes principais como features
    exigiria ajustar só no treino — e não é o que faremos.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    alvo = config.dataset.coluna_alvo
    x = StandardScaler().fit_transform(df[features])
    pca = PCA(n_components=2, random_state=config.split.random_state)
    componentes = pca.fit_transform(x)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for classe, cor in paleta_classes(config).items():
        selecao = (df[alvo] == classe).to_numpy()
        if not selecao.any():
            continue
        ax.scatter(
            componentes[selecao, 0],
            componentes[selecao, 1],
            c=cor,
            label=_rotulo_legivel(classe, config),
            alpha=0.65,
            s=28,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xlabel(f"PC1 — {var[0]:.1%} da variância")
    ax.set_ylabel(f"PC2 — {var[1]:.1%} da variância")
    ax.set_title(f"Projeção PCA ({var.sum():.1%} da variância em 2 componentes)")
    ax.legend(title="Diagnóstico")
    return fig


def plot_separabilidade(cohen_d: pd.Series, *, top: int = 30) -> plt.Figure:
    """Barras horizontais do ``d`` de Cohen por feature."""
    dados = cohen_d.head(top).iloc[::-1]
    cores = [COR_MALIGNO if v > 0.8 else "#AAAAAA" for v in dados]

    fig, ax = plt.subplots(figsize=(8, 0.28 * len(dados) + 1.5))
    ax.barh(dados.index, dados.to_numpy(), color=cores, edgecolor="white")
    ax.axvline(0.8, linestyle="--", color="#555555", linewidth=1, label="efeito grande (d = 0,8)")
    ax.axvline(0, color="#333333", linewidth=0.8)
    # O sinal é um menos tipográfico (U+2212), correto em rótulo de eixo.
    ax.set_xlabel("d de Cohen (maligno − benigno)")  # noqa: RUF001
    ax.set_title("Poder discriminativo individual de cada feature")
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=8)
    return fig


def plot_outliers_por_classe(tabela: pd.DataFrame, config: Config) -> plt.Figure:
    """Proporção de amostras atípicas em cada classe.

    É a evidência empírica a favor de manter os outliers: se eles se
    concentram na classe maligna, removê-los apaga sinal diagnóstico.
    """
    rotulos = [_rotulo_legivel(i, config) for i in tabela.index]
    cores = [paleta_classes(config).get(str(i), "#999999") for i in tabela.index]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    barras = ax.bar(rotulos, tabela["pct_atipica"], color=cores, edgecolor="white", linewidth=1.5)
    for barra, valor in zip(barras, tabela["pct_atipica"], strict=True):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 1.5,
            f"{valor:.1f}%",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("Amostras com ao menos uma feature fora da cerca IQR (%)")
    ax.set_ylim(0, max(tabela["pct_atipica"]) * 1.25)
    ax.set_title("Onde se concentram os valores atípicos")
    return fig
