"""Análise de robustez: o mesmo protocolo sob diferentes sementes.

Com um conjunto de teste de 114 amostras, cada erro vale 0,9 ponto
percentual e cada caso maligno vale 2,4 pontos de recall. Um número único
de desempenho esconde essa granularidade e convida à superinterpretação.

Repetir o protocolo inteiro — novo split, novo treino, novo limiar, nova
avaliação — sob sementes diferentes responde à pergunta que importa: **quanto
do resultado é o modelo e quanto é a sorte da divisão?**
"""

from __future__ import annotations

import pandas as pd

from medai.config import Config
from medai.data.loader import carregar_dados_brutos
from medai.data.split import particionar
from medai.features.build_pipeline import construir_pipeline
from medai.features.cleaning import limpar, separar_alvo
from medai.logging_conf import get_logger
from medai.models import registry
from medai.models.evaluate import metricas_classificacao
from medai.models.train import escolher_limiar_fora_de_fold

logger = get_logger(__name__)


def avaliar_sob_sementes(
    config: Config,
    algoritmo: str,
    hiperparametros: dict,
    features: list[str],
    sementes: list[int] | None = None,
) -> pd.DataFrame:
    """Repete particionamento, treino, escolha de limiar e avaliação por semente.

    Os hiperparâmetros são mantidos fixos, nos valores escolhidos pela busca
    original. Refazer a busca a cada semente mediria também a variabilidade
    da seleção, o que é uma pergunta diferente e exigiria validação cruzada
    aninhada.

    Args:
        algoritmo: Nome no catálogo de modelos.
        hiperparametros: Os parâmetros vencedores, no formato ``modelo__*``.
        features: Colunas do feature set escolhido.
        sementes: Sementes a testar. Sem valor, usa as da configuração.
    """
    sementes = sementes or config.modelagem.sementes_robustez
    bruto = carregar_dados_brutos(config)
    limpo, _ = limpar(bruto, config)
    x, y = separar_alvo(limpo, config)

    linhas = []
    for semente in sementes:
        cfg = config.model_copy(deep=True)
        cfg.split.random_state = semente

        dados = particionar(x, y, cfg)
        x_ajuste, y_ajuste = dados.treino_mais_validacao()
        x_ajuste = x_ajuste[features]

        pipeline = construir_pipeline(registry.obter(algoritmo).construtor(semente), x_ajuste, cfg)
        pipeline.set_params(**hiperparametros)

        limiar, _ = escolher_limiar_fora_de_fold(pipeline, x_ajuste, y_ajuste, cfg)
        pipeline.fit(x_ajuste, y_ajuste)

        proba = pipeline.predict_proba(dados.x_teste[features])[:, 1]
        metricas = metricas_classificacao(dados.y_teste, (proba >= limiar).astype(int), proba)
        linhas.append({"semente": semente, "limiar": round(limiar, 4), **metricas})

    tabela = pd.DataFrame(linhas).set_index("semente")
    logger.info(
        "Robustez avaliada",
        extra={
            "n_sementes": len(sementes),
            "recall_medio": round(float(tabela["recall"].mean()), 4),
            "recall_desvio": round(float(tabela["recall"].std()), 4),
        },
    )
    return tabela


def resumir(tabela: pd.DataFrame, metricas: list[str] | None = None) -> pd.DataFrame:
    """Média, desvio, mínimo e máximo por métrica."""
    metricas = metricas or ["recall", "precision", "f1", "roc_auc", "average_precision", "accuracy"]
    disponiveis = [m for m in metricas if m in tabela.columns]
    resumo = tabela[disponiveis].agg(["mean", "std", "min", "max"]).T
    resumo.columns = ["media", "desvio", "minimo", "maximo"]
    resumo["amplitude"] = resumo["maximo"] - resumo["minimo"]
    return resumo.round(4)
