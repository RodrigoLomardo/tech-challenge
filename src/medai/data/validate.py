"""Validação de sanidade do dataset bruto.

Falhar ruidosamente aqui é barato; descobrir um schema alterado depois de
treinar e publicar o modelo é caro. A validação roda em todo carregamento, no
notebook e no pipeline de treino.

A separação entre **erro** e **aviso** é deliberada: erro significa que o
pipeline não pode prosseguir sem produzir resultado inválido (alvo ausente,
rótulo inesperado); aviso significa algo que exige decisão humana documentada
mas não impede a execução (nulos, duplicatas, valores fora de faixa).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from medai.config import Config
from medai.logging_conf import get_logger

logger = get_logger(__name__)


class ErroDeValidacao(Exception):
    """O dataset viola uma premissa sem a qual o pipeline não pode seguir."""


@dataclass
class ResultadoValidacao:
    """Resumo da validação, usado tanto em log quanto na narrativa do notebook."""

    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    resumo: dict[str, object] = field(default_factory=dict)

    @property
    def valido(self) -> bool:
        return not self.erros

    def levantar_se_invalido(self) -> None:
        if self.erros:
            raise ErroDeValidacao(
                "Dataset reprovado na validação:\n  - " + "\n  - ".join(self.erros)
            )

    def __str__(self) -> str:
        linhas = [f"Validação: {'OK' if self.valido else 'FALHOU'}"]
        for chave, valor in self.resumo.items():
            linhas.append(f"  {chave}: {valor}")
        for erro in self.erros:
            linhas.append(f"  [ERRO]  {erro}")
        for aviso in self.avisos:
            linhas.append(f"  [AVISO] {aviso}")
        return "\n".join(linhas)


def validar_dataset(df: pd.DataFrame, config: Config) -> ResultadoValidacao:
    """Confere estrutura, alvo e qualidade do dataset bruto.

    Args:
        df: Dataset recém-carregado, antes de qualquer limpeza.
        config: Configuração que declara alvo, classe positiva e colunas a
            descartar.

    Returns:
        O resultado com erros, avisos e um resumo quantitativo. Chame
        ``levantar_se_invalido()`` para transformar erros em exceção.
    """
    resultado = ResultadoValidacao()
    alvo = config.dataset.coluna_alvo

    # --- Estrutura mínima ---------------------------------------------------
    if df.empty:
        resultado.erros.append("Dataset vazio")
        return resultado

    if alvo not in df.columns:
        resultado.erros.append(
            f"Coluna alvo '{alvo}' ausente. Colunas disponíveis: {list(df.columns)[:10]}…"
        )
        return resultado

    # --- Alvo ---------------------------------------------------------------
    valores_alvo = df[alvo].dropna()
    classes = sorted(valores_alvo.unique().tolist(), key=str)

    if df[alvo].isna().any():
        resultado.erros.append(
            f"Coluna alvo contém {int(df[alvo].isna().sum())} valor(es) ausente(s); "
            "não é possível imputar rótulo clínico"
        )

    if len(classes) != 2:
        resultado.erros.append(
            f"Alvo deveria ser binário, mas tem {len(classes)} classes distintas: {classes}"
        )
    elif config.dataset.valor_positivo not in classes:
        resultado.erros.append(
            f"Classe positiva '{config.dataset.valor_positivo}' não encontrada no alvo. "
            f"Classes presentes: {classes}. Inverter a classe positiva inverte "
            "precision e recall — ver regra 3 do CLAUDE.md"
        )
    else:
        n_positivos = int((df[alvo] == config.dataset.valor_positivo).sum())
        proporcao = n_positivos / len(df)
        resultado.resumo["classe_positiva"] = config.dataset.valor_positivo
        resultado.resumo["positivos"] = f"{n_positivos} ({proporcao:.1%})"

        # Abaixo de ~10% o desbalanceamento deixa de ser moderado e passa a
        # exigir tratamento explícito; acima disso, estratificar basta.
        if proporcao < 0.10 or proporcao > 0.90:
            resultado.avisos.append(
                f"Desbalanceamento severo ({proporcao:.1%} positivos): reavaliar a "
                "decisão de não aplicar reamostragem"
            )

    # --- Features -----------------------------------------------------------
    colunas_features = [
        c for c in df.columns if c != alvo and c not in config.dataset.colunas_descartar
    ]
    if not colunas_features:
        resultado.erros.append("Nenhuma coluna de feature após descartar as configuradas")
        return resultado

    nao_numericas = [c for c in colunas_features if not pd.api.types.is_numeric_dtype(df[c])]
    if nao_numericas:
        resultado.avisos.append(
            f"{len(nao_numericas)} feature(s) não numérica(s) — serão tratadas pelo "
            f"ramo categórico do pipeline: {nao_numericas[:5]}"
        )

    # --- Qualidade ----------------------------------------------------------
    nulos = df[colunas_features].isna().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        resultado.avisos.append(
            f"{len(nulos)} coluna(s) com valores ausentes (imputação na Fase 3): "
            f"{nulos.head(5).to_dict()}"
        )

    totalmente_nulas = [c for c in colunas_features if df[c].isna().all()]
    if totalmente_nulas:
        resultado.avisos.append(
            f"Coluna(s) 100% nula(s), sem valor informativo: {totalmente_nulas}"
        )

    n_duplicatas = int(df.duplicated().sum())
    if n_duplicatas:
        resultado.avisos.append(f"{n_duplicatas} linha(s) duplicada(s) exata(s)")

    # Medidas morfológicas são grandezas físicas: negativo indica corrupção,
    # não outlier clínico.
    numericas = [c for c in colunas_features if pd.api.types.is_numeric_dtype(df[c])]
    negativas = [c for c in numericas if (df[c].dropna() < 0).any()]
    if negativas:
        resultado.avisos.append(f"Valores negativos em coluna(s) de medida física: {negativas[:5]}")

    constantes = [c for c in numericas if df[c].nunique(dropna=True) <= 1]
    if constantes:
        resultado.avisos.append(f"Coluna(s) constante(s), sem poder discriminativo: {constantes}")

    # --- Resumo -------------------------------------------------------------
    resultado.resumo["linhas"] = len(df)
    resultado.resumo["colunas"] = df.shape[1]
    resultado.resumo["features_uteis"] = len(colunas_features)
    resultado.resumo["memoria"] = f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"

    logger.info(
        "Validação concluída",
        extra={
            "valido": resultado.valido,
            "n_erros": len(resultado.erros),
            "n_avisos": len(resultado.avisos),
        },
    )
    return resultado
