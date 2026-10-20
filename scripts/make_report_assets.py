#!/usr/bin/env python
"""Gera as figuras e tabelas da exploração de dados para o relatório.

Executável de forma independente do notebook: o notebook narra, este script
produz os arquivos versionados em ``reports/``. Rodar os dois garante que o
que está no relatório é exatamente o que o código gera.

Uso:
    python scripts/make_report_assets.py --config configs/breast_cancer.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from medai import viz
from medai.config import Config, carregar_config
from medai.data import explore
from medai.data.loader import carregar_dados_brutos
from medai.data.validate import validar_dataset
from medai.logging_conf import configurar_logging, get_logger

logger = get_logger(__name__)

# Quantas features entram nos recortes visuais. Grades com as 30 cabem; o
# pairplot não — ver docstring de viz.plot_pairplot.
N_TOP_BOXPLOT = 8
N_TOP_PAIRPLOT = 6


def features_do_dataset(df: pd.DataFrame, config: Config) -> list[str]:
    """Colunas utilizáveis: tudo menos o alvo e os descartes configurados."""
    excluir = {config.dataset.coluna_alvo, *config.dataset.colunas_descartar}
    return [c for c in df.columns if c not in excluir]


def gerar(config: Config) -> dict[str, object]:
    """Produz todas as figuras e tabelas da Fase 2. Devolve o resumo numérico."""
    viz.configurar_estilo()

    df = carregar_dados_brutos(config)
    validar_dataset(df, config).levantar_se_invalido()
    features = features_do_dataset(df, config)

    # --- Tabelas ------------------------------------------------------------
    destino_metricas = config.caminhos.absoluto("metricas")
    destino_metricas.mkdir(parents=True, exist_ok=True)

    descritivas = explore.estatisticas_descritivas(df, features)
    descritivas.to_csv(destino_metricas / "eda_estatisticas_descritivas.csv")

    cohen_d = explore.separabilidade_cohen_d(df, config, features)
    cohen_d.to_csv(destino_metricas / "eda_separabilidade.csv")

    outliers = explore.detectar_outliers(df, features)
    outliers.to_csv(destino_metricas / "eda_outliers.csv")

    por_classe = explore.proporcao_outliers_por_classe(df, config, features)
    estrutura = explore.resumo_estrutural(df)
    classes = explore.distribuicao_classes(df, config)

    # --- Figuras ------------------------------------------------------------
    top_features = cohen_d.head(N_TOP_BOXPLOT).index.tolist()
    features_worst = [f for f in features if f.endswith("_worst")]

    figuras = [
        ("01_distribuicao_classes", viz.plot_distribuicao_classes(df, config)),
        (
            "02_distribuicoes_todas_features",
            viz.plot_grade_distribuicoes(
                df, config, features, titulo="Distribuição de cada feature por diagnóstico"
            ),
        ),
        (
            "03_boxplots_top_features",
            viz.plot_boxplots(
                df,
                config,
                top_features,
                titulo=f"Boxplots — {N_TOP_BOXPLOT} features de maior separabilidade",
            ),
        ),
        (
            "04_violinos_features_worst",
            viz.plot_violinos(
                df,
                config,
                features_worst[:10],
                titulo="Features '_worst' — o achado mais atípico de cada lâmina",
            ),
        ),
        ("05_pairplot", viz.plot_pairplot(df, config, cohen_d.head(N_TOP_PAIRPLOT).index.tolist())),
        ("06_correlacao_pearson", viz.plot_heatmap_correlacao(df, features, metodo="pearson")),
        ("07_correlacao_spearman", viz.plot_heatmap_correlacao(df, features, metodo="spearman")),
        ("08_pca_2d", viz.plot_pca_2d(df, config, features)),
        ("09_separabilidade_cohen_d", viz.plot_separabilidade(cohen_d)),
        ("10_outliers_por_classe", viz.plot_outliers_por_classe(por_classe, config)),
    ]

    caminhos = [viz.salvar_figura(fig, nome, config) for nome, fig in figuras]

    # --- Resumo consolidado -------------------------------------------------
    resumo: dict[str, object] = {
        "dataset": config.dataset.nome,
        "hash_config": config.hash_config,
        "estrutura": {k: str(v) for k, v in estrutura.items()},
        "classes": classes.to_dict(orient="index"),
        "top_5_separabilidade": cohen_d.head(5).round(3).to_dict(),
        "features_sem_separabilidade": cohen_d[cohen_d.abs() < 0.2].round(3).to_dict(),
        "top_5_assimetria": descritivas["skew"].nlargest(5).round(2).to_dict(),
        "outliers_pct_por_classe": por_classe["pct_atipica"].to_dict(),
        "n_figuras": len(caminhos),
    }
    (destino_metricas / "eda_resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return resumo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/breast_cancer.yaml")
    args = parser.parse_args(argv)

    configurar_logging()
    resumo = gerar(carregar_config(args.config))

    print(f"\n{resumo['n_figuras']} figuras geradas em reports/figures/")
    print("Tabelas em reports/metrics/eda_*.csv")
    print(f"\nTop separabilidade: {list(resumo['top_5_separabilidade'])[:3]}")
    print(f"Sem separabilidade: {list(resumo['features_sem_separabilidade'])}")
    print(f"Atípicos por classe: {resumo['outliers_pct_por_classe']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
