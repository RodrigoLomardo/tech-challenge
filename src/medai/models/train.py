"""Protocolo experimental: validação cruzada, busca de hiperparâmetros e
seleção do modelo campeão.

**A armadilha que este módulo evita.** Otimizar recall puro seleciona o modelo
degenerado: classificar todos os casos como malignos dá recall 1,0 com
precisão de 37%. O recall é a prioridade clínica por razão legítima, mas
usá-lo isoladamente como critério de ``refit`` entregaria um sistema inútil que
encaminha a fila inteira para revisão.

Uma tentativa anterior usava a regra de um erro padrão — aceitar candidatos
dentro de um desvio do melhor recall e desempatar por F1. **Ela falha
exatamente no caso que deveria cobrir:** o modelo degenerado acerta todos os
positivos em todos os folds, seu desvio é zero, a janela de equivalência
colapsa e ele vence sozinho.

A solução correta é **F-beta com beta=2**, que pondera recall quatro vezes
mais que precisão. O degenerado obtém F2 = 0,746 contra 0,963 de um bom
modelo: a prioridade clínica é preservada sem que a solução trivial seja
aceita. A garantia de recall ≥ 98% vem depois, do ajuste de limiar — que é
onde ela pertence, por ser uma decisão de operação e não de escolha de
família de modelo.

O critério de seleção do campeão é fixado **antes** de qualquer contato com o
conjunto de teste, para que a escolha não seja contaminada por conhecimento do
resultado final.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline

from medai.config import RAIZ_PROJETO, Config, carregar_config
from medai.data.loader import carregar_dados_brutos
from medai.data.split import DadosParticionados, construir_cv, particionar
from medai.features import correlation as corr
from medai.features.build_pipeline import construir_pipeline
from medai.features.cleaning import limpar, separar_alvo
from medai.logging_conf import configurar_logging, get_logger
from medai.models import persist, registry
from medai.models.evaluate import escolher_limiar, metricas_classificacao
from medai.models.persist import Metadados

logger = get_logger(__name__)

# Métricas acompanhadas em cada fold. A primária entra em `refit`.
# `zero_division=0` é necessário por causa do baseline: ele nunca prediz a
# classe positiva, e sem isso precision ficaria indefinida e o scikit-learn
# emitiria um aviso por fold. Definir precisão como 0 quando não há predição
# positiva é a convenção correta aqui — o modelo não acertou nenhum caso.
SCORING = {
    # Critério de seleção: recall ponderado 4× sobre precisão.
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
}

# O scikit-learn reajusta o melhor candidato por esta métrica.
METRICA_REFIT = "f2"


@dataclass
class ResultadoExperimento:
    """Uma execução: um modelo, um feature set."""

    modelo: str
    feature_set: str
    n_features: int
    melhores_parametros: dict[str, Any]
    metricas_cv: dict[str, float]
    tempo_segundos: float
    estimador: BaseEstimator = field(repr=False)

    def para_linha(self, config: Config) -> dict[str, Any]:
        """Formato tabular para ``reports/experiments.csv``."""
        linha: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": config.dataset.nome,
            "modelo": self.modelo,
            "feature_set": self.feature_set,
            "n_features": self.n_features,
            "tempo_s": round(self.tempo_segundos, 2),
            "hash_config": config.hash_config,
            "random_state": config.split.random_state,
        }
        linha.update({k: round(v, 4) for k, v in self.metricas_cv.items()})
        linha["parametros"] = str(self.melhores_parametros)
        return linha


def treinar_com_busca(
    especificacao: registry.EspecificacaoModelo,
    x: pd.DataFrame,
    y: pd.Series,
    config: Config,
    feature_set: str = "completo",
) -> ResultadoExperimento:
    """Treina um modelo com validação cruzada e busca de hiperparâmetros.

    A CV reajusta o **pipeline inteiro** a cada fold — imputador e escalonador
    incluídos. É isso que impede que estatísticas do fold de validação vazem
    para o de treino dentro da própria validação cruzada.
    """
    semente = config.split.random_state
    pipeline = construir_pipeline(especificacao.construtor(semente), x, config)
    cv = construir_cv(config)
    inicio = time.perf_counter()

    if especificacao.busca == "nenhuma":
        # O baseline não tem o que ajustar; ainda assim passa pela mesma CV,
        # para que suas métricas sejam comparáveis às dos demais.
        from sklearn.model_selection import cross_validate

        pontuacoes = cross_validate(pipeline, x, y, cv=cv, scoring=SCORING, n_jobs=-1)
        metricas = {f"cv_{nome}": float(np.mean(pontuacoes[f"test_{nome}"])) for nome in SCORING}
        metricas.update(
            {f"cv_{nome}_std": float(np.std(pontuacoes[f"test_{nome}"])) for nome in SCORING}
        )
        pipeline.fit(x, y)
        melhor = pipeline
        parametros: dict[str, Any] = {}
    else:
        comum = {
            "estimator": pipeline,
            "scoring": SCORING,
            "refit": METRICA_REFIT,
            "cv": cv,
            "n_jobs": -1,
            "return_train_score": False,
        }
        if especificacao.busca == "grid":
            busca = GridSearchCV(param_grid=especificacao.grade, **comum)
        else:
            busca = RandomizedSearchCV(
                param_distributions=especificacao.grade,
                n_iter=especificacao.n_iteracoes,
                random_state=semente,
                **comum,
            )

        busca.fit(x, y)
        indice = busca.best_index_
        metricas = {
            f"cv_{nome}": float(busca.cv_results_[f"mean_test_{nome}"][indice]) for nome in SCORING
        }
        metricas.update(
            {
                f"cv_{nome}_std": float(busca.cv_results_[f"std_test_{nome}"][indice])
                for nome in SCORING
            }
        )
        melhor = busca.best_estimator_
        parametros = busca.best_params_

    duracao = time.perf_counter() - inicio
    logger.info(
        "Modelo treinado",
        extra={
            "modelo": especificacao.nome,
            "feature_set": feature_set,
            "cv_f2": round(metricas["cv_f2"], 4),
            "cv_recall": round(metricas["cv_recall"], 4),
            "tempo_s": round(duracao, 1),
        },
    )
    return ResultadoExperimento(
        modelo=especificacao.nome,
        feature_set=feature_set,
        n_features=x.shape[1],
        melhores_parametros=parametros,
        metricas_cv=metricas,
        tempo_segundos=duracao,
        estimador=melhor,
    )


def construir_feature_sets(dados: DadosParticionados, config: Config) -> dict[str, list[str]]:
    """Os dois conjuntos comparados pelo ADR-009.

    A seleção por redundância é calculada **apenas no treino**: é uma decisão
    que aprende dos dados e, feita sobre o conjunto inteiro, vazaria.
    """
    completo = dados.x_treino.columns.tolist()
    selecao = corr.selecionar_por_redundancia(dados.x_treino, dados.y_treino, config)
    return {"completo": completo, "reduzido": selecao.mantidas}


def executar_experimentos(
    dados: DadosParticionados, config: Config
) -> tuple[pd.DataFrame, dict[tuple[str, str], ResultadoExperimento]]:
    """Roda todos os modelos sobre todos os feature sets.

    Returns:
        A tabela de experimentos e o dicionário de resultados indexado por
        ``(modelo, feature_set)``.
    """
    feature_sets = construir_feature_sets(dados, config)
    especificacoes = registry.listar(config.modelagem.modelos)

    linhas: list[dict[str, Any]] = []
    resultados: dict[tuple[str, str], ResultadoExperimento] = {}

    for nome_set, colunas in feature_sets.items():
        x = dados.x_treino[colunas]
        for especificacao in especificacoes:
            resultado = treinar_com_busca(especificacao, x, dados.y_treino, config, nome_set)
            resultados[(especificacao.nome, nome_set)] = resultado
            linhas.append(resultado.para_linha(config))

    tabela = pd.DataFrame(linhas)
    _registrar_experimentos(tabela, config)
    return tabela, resultados


def _registrar_experimentos(tabela: pd.DataFrame, config: Config) -> Path:
    """Acrescenta as execuções ao histórico em ``reports/experiments.csv``.

    Append em vez de sobrescrita: o histórico de experimentos é parte do
    registro do projeto, e uma execução posterior não deve apagar a anterior.
    """
    caminho = config.caminhos.absoluto("experimentos")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    cabecalho = not caminho.exists()
    tabela.to_csv(caminho, mode="a", header=cabecalho, index=False)
    logger.info("Experimentos registrados", extra={"arquivo": str(caminho), "n": len(tabela)})
    return caminho


# O critério é declarado aqui, no código, e não escolhido depois de ver os
# resultados do teste. Documentá-lo antes é o que impede a racionalização
# retrospectiva — "este modelo é melhor porque foi o que ganhou".
CRITERIO_SELECAO = (
    "1) maior F2 médio na CV do treino (recall ponderado 4× sobre precisão); "
    "2) empate dentro de um desvio padrão → maior recall; "
    "3) empate persistente → modelo mais simples, depois menos features."
)

# Ordem de simplicidade, para o terceiro critério de desempate.
_ORDEM_SIMPLICIDADE = [
    "dummy",
    "logistic_regression",
    "decision_tree",
    "knn",
    "svm_rbf",
    "random_forest",
    "xgboost",
]


def selecionar_campeao(tabela: pd.DataFrame) -> pd.Series:
    """Aplica :data:`CRITERIO_SELECAO` à tabela de experimentos.

    O baseline é excluído: ele existe como piso de comparação, não como
    candidato. Com recall zero, aliás, nunca venceria.
    """
    candidatos = tabela[tabela["modelo"] != "dummy"].copy()
    if candidatos.empty:
        raise ValueError("Nenhum candidato além do baseline")

    melhor_f2 = candidatos["cv_f2"].max()
    tolerancia = float(candidatos.loc[candidatos["cv_f2"].idxmax(), "cv_f2_std"])
    equivalentes = candidatos[candidatos["cv_f2"] >= melhor_f2 - tolerancia]

    melhor_recall = equivalentes["cv_recall"].max()
    finalistas = equivalentes[equivalentes["cv_recall"] >= melhor_recall - 1e-9]

    if len(finalistas) > 1:
        finalistas = finalistas.assign(
            _simplicidade=finalistas["modelo"].map(
                lambda m: _ORDEM_SIMPLICIDADE.index(m) if m in _ORDEM_SIMPLICIDADE else 99
            )
        ).sort_values(["_simplicidade", "n_features"])

    campeao = finalistas.iloc[0]
    logger.info(
        "Campeão selecionado",
        extra={
            "modelo": campeao["modelo"],
            "feature_set": campeao["feature_set"],
            "cv_f2": campeao["cv_f2"],
            "cv_recall": campeao["cv_recall"],
        },
    )
    return campeao


# -----------------------------------------------------------------------------
# Treino final e serialização (Fase 5)
# -----------------------------------------------------------------------------


def escolher_limiar_fora_de_fold(
    estimador: BaseEstimator, x: pd.DataFrame, y: pd.Series, config: Config
) -> tuple[float, Any]:
    """Escolhe o limiar do modelo final por predições fora de fold.

    **Por que não reaproveitar o limiar da validação.** Aquele limiar foi
    calculado para um modelo treinado apenas no conjunto de treino. O modelo
    final é treinado em treino + validação — é um modelo diferente, com
    probabilidades ligeiramente diferentes, e o limiar ótimo para um não é
    necessariamente ótimo para o outro.

    Também não se pode medir o limiar nas predições do próprio modelo final
    sobre os dados em que ele foi treinado: ele os ajusta bem demais, e o
    limiar sairia otimista.

    A saída é ``cross_val_predict``: cada amostra recebe uma probabilidade
    predita por um modelo que **não a viu** durante o treino. São 455
    predições honestas, contra as 114 do conjunto de validação — estimativa
    mais estável, e sem tocar no teste.
    """
    probabilidades = cross_val_predict(
        clone(estimador), x, y, cv=construir_cv(config), method="predict_proba", n_jobs=-1
    )[:, 1]
    limiar = escolher_limiar(y, probabilidades, config.modelagem.recall_minimo_alvo)
    return float(limiar.limiar), limiar


def finalizar_modelo(
    campeao: pd.Series,
    resultados: dict[tuple[str, str], ResultadoExperimento],
    dados: DadosParticionados,
    config: Config,
) -> tuple[Pipeline, Metadados]:
    """Retreina o campeão em treino + validação e serializa o artefato.

    O conjunto de teste permanece intocado: o modelo que será avaliado na
    Fase 6 nunca viu aquelas 114 amostras.
    """
    nome_modelo, nome_set = campeao["modelo"], campeao["feature_set"]
    colunas = construir_feature_sets(dados, config)[nome_set]
    resultado = resultados[(nome_modelo, nome_set)]

    # Desempenho na validação, com o modelo treinado só no treino — é a
    # estimativa honesta disponível nesta fase.
    proba_validacao = resultado.estimador.predict_proba(dados.x_validacao[colunas])[:, 1]
    limiar_validacao = escolher_limiar(
        dados.y_validacao, proba_validacao, config.modelagem.recall_minimo_alvo
    )
    metricas_validacao = metricas_classificacao(
        dados.y_validacao,
        (proba_validacao >= limiar_validacao.limiar).astype(int),
        proba_validacao,
    )

    # Retreino no conjunto ampliado.
    x_final, y_final = dados.treino_mais_validacao()
    x_final = x_final[colunas]
    especificacao = registry.obter(nome_modelo)
    pipeline = construir_pipeline(
        especificacao.construtor(config.split.random_state), x_final, config
    )
    pipeline.set_params(**resultado.melhores_parametros)

    limiar_final, detalhe = escolher_limiar_fora_de_fold(pipeline, x_final, y_final, config)
    pipeline.fit(x_final, y_final)

    logger.info(
        "Modelo final treinado",
        extra={
            "n_amostras": len(x_final),
            "limiar_validacao": round(limiar_validacao.limiar, 3),
            "limiar_fora_de_fold": round(limiar_final, 3),
        },
    )

    caminho_dados = config.dataset.caminho_csv
    if caminho_dados is not None and not caminho_dados.is_absolute():
        caminho_dados = RAIZ_PROJETO / caminho_dados

    estatisticas = {
        coluna: {
            "minimo": float(x_final[coluna].min()),
            "p01": float(x_final[coluna].quantile(0.01)),
            "mediana": float(x_final[coluna].median()),
            "p99": float(x_final[coluna].quantile(0.99)),
            "maximo": float(x_final[coluna].max()),
        }
        for coluna in colunas
    }

    metadados = Metadados(
        versao_modelo=f"{config.dataset.nome}_v1",
        nome_arquivo=f"medai_{config.dataset.nome}_v1.joblib",
        treinado_em=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset=config.dataset.nome,
        hash_config=config.hash_config,
        hash_dados=persist.hash_arquivo(caminho_dados) if caminho_dados else "indisponivel",
        features_esperadas=list(colunas),
        feature_set=nome_set,
        limiar_decisao=limiar_final,
        limiar_origem=(
            "predicoes_fora_de_fold_em_treino_mais_validacao "
            f"(alvo de recall {config.modelagem.recall_minimo_alvo:.0%}; "
            f"limiar medido na validação isolada: {limiar_validacao.limiar:.3f})"
        ),
        zona_cinzenta=config.modelagem.zona_cinzenta,
        classe_positiva=str(config.dataset.valor_positivo),
        rotulo_positivo=(
            "maligno" if config.dataset.nome == "breast_cancer_wisconsin" else "positivo"
        ),
        metricas_cv={k: float(campeao[k]) for k in campeao.index if str(k).startswith("cv_")},
        metricas_validacao={k: float(v) for k, v in metricas_validacao.items()},
        estatisticas_features=estatisticas,
        algoritmo=nome_modelo,
        hiperparametros={k: str(v) for k, v in resultado.melhores_parametros.items()},
        random_state=config.split.random_state,
        n_amostras_treino=len(x_final),
    )

    persist.salvar(pipeline, metadados, config)
    return pipeline, metadados


def preparar_dados(config: Config) -> DadosParticionados:
    """Carrega, limpa e particiona — o início comum a todos os pipelines."""
    bruto = carregar_dados_brutos(config)
    limpo, _ = limpar(bruto, config)
    x, y = separar_alvo(limpo, config)
    return particionar(x, y, config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/breast_cancer.yaml")
    args = parser.parse_args(argv)

    configurar_logging()
    config = carregar_config(args.config)
    dados = preparar_dados(config)

    print(f"\nTreinando {len(config.modelagem.modelos)} modelos × 2 feature sets…\n")
    tabela, resultados = executar_experimentos(dados, config)

    colunas = [
        "modelo",
        "feature_set",
        "n_features",
        "cv_f2",
        "cv_recall",
        "cv_precision",
        "cv_f1",
        "cv_roc_auc",
        "tempo_s",
    ]
    print(tabela[colunas].sort_values("cv_f2", ascending=False).to_string(index=False))

    campeao = selecionar_campeao(tabela)
    print(f"\nCritério: {CRITERIO_SELECAO}")
    print(
        f"\nCampeão: {campeao['modelo']} ({campeao['feature_set']}, {campeao['n_features']} features)"
    )
    print(f"  CV F2:     {campeao['cv_f2']:.4f} ± {campeao['cv_f2_std']:.4f}")
    print(f"  CV recall: {campeao['cv_recall']:.4f}")
    print(f"  CV F1:     {campeao['cv_f1']:.4f}")

    # Limiar escolhido na validação — nunca no teste.
    resultado = resultados[(campeao["modelo"], campeao["feature_set"])]
    colunas_set = construir_feature_sets(dados, config)[campeao["feature_set"]]
    proba_validacao = resultado.estimador.predict_proba(dados.x_validacao[colunas_set])[:, 1]

    limiar = escolher_limiar(
        dados.y_validacao, proba_validacao, config.modelagem.recall_minimo_alvo
    )
    print(f"\n{limiar}")

    padrao = metricas_classificacao(
        dados.y_validacao, (proba_validacao >= 0.5).astype(int), proba_validacao
    )
    print("\nPara comparação, no limiar padrão de 0,5:")
    print(f"  recall: {padrao['recall']:.3f} ({padrao['falsos_negativos']} falsos negativos)")
    print(f"  precision: {padrao['precision']:.3f}")

    # --- Fase 5: treino final e serialização --------------------------------
    print("\n" + "─" * 70)
    print("Retreinando o campeão em treino + validação (o teste segue intocado)…")
    _, metadados = finalizar_modelo(campeao, resultados, dados, config)

    print(f"\nArtefato: models/{metadados.nome_arquivo}")
    print(f"  algoritmo:        {metadados.algoritmo}")
    print(f"  amostras:         {metadados.n_amostras_treino}")
    print(f"  features:         {len(metadados.features_esperadas)} ({metadados.feature_set})")
    print(f"  limiar:           {metadados.limiar_decisao:.3f}")
    print(f"  zona cinzenta:    {metadados.zona_cinzenta}")
    print(f"  hash dos dados:   {metadados.hash_dados}")
    print(f"  scikit-learn:     {metadados.versoes['scikit-learn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
