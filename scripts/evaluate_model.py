#!/usr/bin/env python
"""Avaliação final no conjunto de teste e geração dos artefatos da Fase 6.

**Este script abre o conjunto de teste.** Ele foi mantido intocado desde a
Fase 3: nenhuma decisão de modelagem, hiperparâmetro ou limiar foi tomada
com base nele. É a única estimativa honesta de desempenho que o projeto tem.

Uso:
    python scripts/evaluate_model.py --config configs/breast_cancer.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from medai import viz
from medai.config import Config, carregar_config
from medai.logging_conf import configurar_logging, get_logger
from medai.models import explain, persist, plots, robustez
from medai.models.evaluate import (
    classificar_com_faixa,
    metricas_classificacao,
    resumo_faixas,
)
from medai.models.train import construir_feature_sets, executar_experimentos, preparar_dados

logger = get_logger(__name__)


def avaliar_todos_os_modelos(dados, config: Config) -> pd.DataFrame:
    """Treina cada modelo e mede no teste, para a tabela comparativa.

    A seleção do campeão já aconteceu na Fase 4, usando apenas a validação
    cruzada do treino. Esta tabela existe para o relatório — mostrar como os
    demais se saíram —, não para revisar a escolha. Trocar de campeão aqui
    seria selecionar pelo teste e invalidaria a estimativa.
    """
    _, resultados = executar_experimentos(dados, config)
    conjuntos = construir_feature_sets(dados, config)

    linhas = []
    for (nome, feature_set), resultado in resultados.items():
        colunas = conjuntos[feature_set]
        proba = resultado.estimador.predict_proba(dados.x_teste[colunas])[:, 1]
        metricas = metricas_classificacao(dados.y_teste, (proba >= 0.5).astype(int), proba)
        linhas.append({"modelo": nome, "feature_set": feature_set, **metricas})

    tabela = pd.DataFrame(linhas)
    return (
        tabela[tabela["feature_set"] == "completo"]
        .set_index("modelo")
        .drop(columns=["feature_set"])
    )


def analisar_erros(
    x_teste: pd.DataFrame, y_teste: pd.Series, proba: np.ndarray, limiar: float
) -> pd.DataFrame:
    """Casos classificados incorretamente, com sua probabilidade."""
    predito = (proba >= limiar).astype(int)
    errados = predito != y_teste.to_numpy()
    if not errados.any():
        return pd.DataFrame()

    tabela = pd.DataFrame(
        {
            "indice": x_teste.index[errados],
            "real": y_teste.to_numpy()[errados],
            "probabilidade": proba[errados].round(4),
            "tipo": [
                (
                    "falso negativo (câncer não sinalizado)"
                    if real == 1
                    else "falso positivo (revisão extra)"
                )
                for real in y_teste.to_numpy()[errados]
            ],
            "distancia_do_limiar": (proba[errados] - limiar).round(4),
        }
    ).set_index("indice")
    return tabela.sort_values("probabilidade")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/breast_cancer.yaml")
    parser.add_argument("--pular-robustez", action="store_true")
    args = parser.parse_args(argv)

    configurar_logging()
    viz.configurar_estilo()
    config = carregar_config(args.config)

    pipeline, metadados = persist.carregar(config)
    dados = preparar_dados(config)
    colunas = metadados.features_esperadas
    x_teste, y_teste = dados.x_teste[colunas], dados.y_teste

    print("=" * 72)
    print(f"AVALIAÇÃO NO CONJUNTO DE TESTE — {len(x_teste)} amostras, intocadas até agora")
    print(f"Modelo: {metadados.algoritmo} · limiar {metadados.limiar_decisao:.3f}")
    print("=" * 72)

    # --- 1. Métricas --------------------------------------------------------
    proba = pipeline.predict_proba(x_teste)[:, 1]
    predito = (proba >= metadados.limiar_decisao).astype(int)
    metricas = metricas_classificacao(y_teste, predito, proba)

    metricas_padrao = metricas_classificacao(y_teste, (proba >= 0.5).astype(int), proba)

    print("\n--- Métricas no limiar operacional ---")
    for chave in (
        "recall",
        "precision",
        "f1",
        "especificidade",
        "accuracy",
        "roc_auc",
        "average_precision",
        "brier",
    ):
        print(f"  {chave:20} {metricas[chave]:.4f}")
    print(f"  {'falsos negativos':20} {int(metricas['falsos_negativos'])}")
    print(f"  {'falsos positivos':20} {int(metricas['falsos_positivos'])}")

    # --- 2. Figuras ---------------------------------------------------------
    figuras = [
        (
            "11_matriz_confusao",
            plots.plot_matriz_confusao(
                y_teste.to_numpy(),
                predito,
                titulo=f"Conjunto de teste — limiar {metadados.limiar_decisao:.3f}",
            ),
        ),
        (
            "12_curvas_desempenho",
            plots.plot_curvas_desempenho(
                y_teste.to_numpy(),
                proba,
                limiar=metadados.limiar_decisao,
                taxa_positivos=float(y_teste.mean()),
            ),
        ),
    ]

    comparacao = avaliar_todos_os_modelos(dados, config)
    figuras.append(
        ("13_comparacao_modelos", plots.plot_comparacao_modelos(comparacao, metrica="recall"))
    )

    # --- 3. Interpretabilidade ---------------------------------------------
    x_fundo, _ = dados.treino_mais_validacao()
    x_fundo = x_fundo[colunas]

    coeficientes = explain.coeficientes(pipeline)
    permutacao = explain.importancia_por_permutacao(
        pipeline, x_teste, y_teste, scoring="recall", random_state=config.split.random_state
    )
    explicacao = explain.explicar(pipeline, x_teste, x_fundo=x_fundo)
    shap_global = explicacao.importancia_global
    shap_agrupado = explain.agrupar_por_medida(shap_global)

    figuras += [
        ("14_coeficientes", plots.plot_coeficientes(coeficientes)),
        ("15_permutation_importance", plots.plot_permutation_importance(permutacao)),
        ("16_shap_global", plots.plot_shap_global(shap_global)),
        ("17_shap_agrupado", plots.plot_shap_agrupado(shap_agrupado)),
        ("18_shap_beeswarm", plots.plot_shap_beeswarm(explicacao)),
        ("19_dependencia_top", plots.plot_dependencia(explicacao, shap_global.index[0])),
    ]

    # --- 4. Casos individuais ----------------------------------------------
    verdadeiro = y_teste.to_numpy()
    casos = {
        "verdadeiro positivo": np.where((verdadeiro == 1) & (predito == 1))[0],
        "verdadeiro negativo": np.where((verdadeiro == 0) & (predito == 0))[0],
        "falso positivo": np.where((verdadeiro == 0) & (predito == 1))[0],
        "falso negativo": np.where((verdadeiro == 1) & (predito == 0))[0],
    }
    for n, (rotulo, indices) in enumerate(casos.items(), start=20):
        if not len(indices):
            print(f"\n(nenhum caso de '{rotulo}' no teste)")
            continue
        # Para acertos, o caso mais confiante; para erros, o mais extremo.
        escolhido = int(indices[np.argmax(np.abs(proba[indices] - metadados.limiar_decisao))])
        figuras.append(
            (
                f"{n}_shap_{rotulo.replace(' ', '_')}",
                plots.plot_shap_caso(
                    explicacao,
                    escolhido,
                    titulo=f"{rotulo.capitalize()} — probabilidade {proba[escolhido]:.3f}",
                ),
            )
        )

    # --- 5. Robustez --------------------------------------------------------
    tabela_robustez = pd.DataFrame()
    if not args.pular_robustez:
        print("\n--- Robustez sob múltiplas sementes ---")
        hiperparametros = {k: _converter(v) for k, v in metadados.hiperparametros.items()}
        tabela_robustez = robustez.avaliar_sob_sementes(
            config, metadados.algoritmo, hiperparametros, colunas
        )
        resumo = robustez.resumir(tabela_robustez)
        print(resumo.to_string())
        figuras.append(("24_robustez", plots.plot_robustez(tabela_robustez)))

    # --- 6. Faixas e erros --------------------------------------------------
    faixas = classificar_com_faixa(proba, metadados.limiar_decisao, config)
    distribuicao = resumo_faixas(faixas)
    cruzamento = pd.crosstab(faixas, verdadeiro, rownames=["faixa"], colnames=["maligno"])

    print("\n--- Faixas de decisão ---")
    print(cruzamento.to_string())

    erros = analisar_erros(x_teste, y_teste, proba, metadados.limiar_decisao)
    if not erros.empty:
        print("\n--- Casos classificados incorretamente ---")
        print(erros.to_string())

    # --- 7. Persistência ----------------------------------------------------
    caminhos = [viz.salvar_figura(fig, nome, config) for nome, fig in figuras]
    destino = config.caminhos.absoluto("metricas")
    destino.mkdir(parents=True, exist_ok=True)

    comparacao.to_csv(destino / "teste_comparacao_modelos.csv")
    coeficientes.to_csv(destino / "teste_coeficientes.csv")
    permutacao.to_csv(destino / "teste_permutation_importance.csv")
    shap_global.to_csv(destino / "teste_shap_global.csv")
    shap_agrupado.to_csv(destino / "teste_shap_agrupado.csv")
    if not erros.empty:
        erros.to_csv(destino / "teste_erros.csv")
    if not tabela_robustez.empty:
        tabela_robustez.to_csv(destino / "teste_robustez.csv")

    (destino / "test_metrics.json").write_text(
        json.dumps(
            {
                "modelo": metadados.versao_modelo,
                "algoritmo": metadados.algoritmo,
                "limiar": metadados.limiar_decisao,
                "n_teste": len(x_teste),
                "metricas_limiar_operacional": metricas,
                "metricas_limiar_padrao_0_5": metricas_padrao,
                "faixas": distribuicao,
                "robustez": (
                    robustez.resumir(tabela_robustez).to_dict() if not tabela_robustez.empty else {}
                ),
                "shap_top5": shap_global.head(5).round(4).to_dict(),
                "shap_agrupado": shap_agrupado.round(4).to_dict(),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\n{len(caminhos)} figuras geradas · métricas em reports/metrics/test_metrics.json")
    return 0


def _converter(valor: str):
    """Reconstrói o tipo original de um hiperparâmetro serializado como texto."""
    if valor in {"None", "none"}:
        return None
    try:
        return int(valor)
    except ValueError:
        pass
    try:
        return float(valor)
    except ValueError:
        return valor


if __name__ == "__main__":
    sys.exit(main())
