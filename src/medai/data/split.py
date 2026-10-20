"""Particionamento treino / validação / teste.

A separação em **três** conjuntos, e não dois, é o que permite escolher
hiperparâmetros e limiar de decisão sem tocar no teste. Cada conjunto tem um
papel que não se mistura:

===========  ======================================================
Conjunto     Para que serve
===========  ======================================================
**Treino**   Ajustar o modelo. A validação cruzada roda aqui dentro.
**Validação**Escolher o limiar de decisão e conferir a seleção da CV.
**Teste**    Estimar o desempenho. Tocado **uma única vez**, na Fase 6.
===========  ======================================================

Com 569 amostras, o teste tem ~114 linhas e cada erro vale ~0,9 ponto
percentual. Usar o teste para qualquer escolha — modelo, hiperparâmetro ou
limiar — contamina a estimativa final de forma irrecuperável, porque não há
dado reservado para corrigi-la depois.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DadosParticionados:
    """Os três conjuntos, com as proporções registradas para auditoria."""

    x_treino: pd.DataFrame
    y_treino: pd.Series
    x_validacao: pd.DataFrame
    y_validacao: pd.Series
    x_teste: pd.DataFrame
    y_teste: pd.Series
    random_state: int

    @property
    def tamanhos(self) -> dict[str, int]:
        return {
            "treino": len(self.x_treino),
            "validacao": len(self.x_validacao),
            "teste": len(self.x_teste),
        }

    @property
    def taxas_positivas(self) -> dict[str, float]:
        return {
            "treino": round(float(self.y_treino.mean()), 4),
            "validacao": round(float(self.y_validacao.mean()), 4),
            "teste": round(float(self.y_teste.mean()), 4),
        }

    def treino_mais_validacao(self) -> tuple[pd.DataFrame, pd.Series]:
        """Conjunto usado no retreino final, depois de escolhido o campeão.

        O teste continua de fora: o modelo que será avaliado nunca viu
        aquelas amostras.
        """
        x = pd.concat([self.x_treino, self.x_validacao], axis=0)
        y = pd.concat([self.y_treino, self.y_validacao], axis=0)
        return x, y

    def __str__(self) -> str:
        linhas = ["Particionamento (estratificado):"]
        total = sum(self.tamanhos.values())
        for nome, n in self.tamanhos.items():
            taxa = self.taxas_positivas[nome]
            linhas.append(f"  {nome:>10}: {n:>4} ({n / total:.0%})  positivos: {taxa:.1%}")
        linhas.append(f"  random_state: {self.random_state}")
        return "\n".join(linhas)


def particionar(x: pd.DataFrame, y: pd.Series, config: Config) -> DadosParticionados:
    """Divide em treino / validação / teste de forma estratificada.

    Feito em duas etapas: primeiro separa o teste, depois divide o restante
    entre treino e validação. As proporções da configuração são relativas ao
    **total**, então a segunda divisão é reescalada para que 20% de validação
    signifique 20% do dataset inteiro, não 20% do que sobrou.
    """
    estratificar_por = y if config.split.estratificar else None

    x_resto, x_teste, y_resto, y_teste = train_test_split(
        x,
        y,
        test_size=config.split.proporcao_teste,
        stratify=estratificar_por,
        random_state=config.split.random_state,
        shuffle=True,
    )

    # 0.20 do total, mas tirado de um conjunto que já é só 0.80 do total.
    proporcao_validacao_relativa = config.split.proporcao_validacao / (
        1 - config.split.proporcao_teste
    )

    x_treino, x_validacao, y_treino, y_validacao = train_test_split(
        x_resto,
        y_resto,
        test_size=proporcao_validacao_relativa,
        stratify=y_resto if config.split.estratificar else None,
        random_state=config.split.random_state,
        shuffle=True,
    )

    dados = DadosParticionados(
        x_treino=x_treino,
        y_treino=y_treino,
        x_validacao=x_validacao,
        y_validacao=y_validacao,
        x_teste=x_teste,
        y_teste=y_teste,
        random_state=config.split.random_state,
    )

    logger.info(
        "Dados particionados",
        extra={"tamanhos": dados.tamanhos, "taxas_positivas": dados.taxas_positivas},
    )
    return dados


def construir_cv(config: Config) -> StratifiedKFold:
    """Validação cruzada estratificada, usada **dentro do treino**.

    Com base pequena, a CV extrai mais sinal do treino do que um holdout
    único de validação conseguiria: cada amostra serve de validação em
    exatamente um fold, em vez de 20% delas servirem sempre e 80% nunca.
    """
    return StratifiedKFold(
        n_splits=config.split.n_folds_cv,
        shuffle=True,
        random_state=config.split.random_state,
    )


def salvar_particoes(dados: DadosParticionados, config: Config) -> dict[str, str]:
    """Grava os conjuntos em ``data/processed/`` para inspeção e reprodução."""
    destino = config.caminhos.absoluto("dados_processados")
    destino.mkdir(parents=True, exist_ok=True)

    arquivos: dict[str, str] = {}
    conjuntos = {
        "treino": (dados.x_treino, dados.y_treino),
        "validacao": (dados.x_validacao, dados.y_validacao),
        "teste": (dados.x_teste, dados.y_teste),
    }
    for nome, (x, y) in conjuntos.items():
        caminho = destino / f"{config.dataset.nome}_{nome}.csv"
        pd.concat([x, y.rename(config.dataset.coluna_alvo)], axis=1).to_csv(caminho, index=False)
        arquivos[nome] = str(caminho)

    logger.info("Partições salvas", extra={"destino": str(destino)})
    return arquivos
