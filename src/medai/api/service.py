"""Lógica de inferência da API.

O modelo e o explicador SHAP são carregados **uma vez**, na inicialização.
Instanciar o explicador por requisição multiplicaria a latência sem nenhum
benefício — ele depende apenas do modelo e do conjunto de referência, não da
amostra que chega.

O conjunto de referência do SHAP é o **conjunto de treino**, não o lote que
chega. Usar o lote faria a explicação de um paciente depender de quem mais
está na fila naquele momento — o mesmo exame produziria justificativas
diferentes conforme o horário.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from medai.config import Config
from medai.logging_conf import get_logger
from medai.models import explain, persist
from medai.models.evaluate import (
    FAIXA_BENIGNO,
    FAIXA_MALIGNO,
    FAIXA_REVISAO,
    classificar_com_faixa,
)
from medai.models.persist import Metadados

logger = get_logger(__name__)

RECOMENDACOES = {
    FAIXA_BENIGNO: "Prioridade padrão na fila de revisão.",
    FAIXA_REVISAO: "Revisão médica prioritária — o modelo não tem base suficiente para classificar.",
    FAIXA_MALIGNO: "Priorizar na fila de revisão.",
}

LIMITACOES = [
    "Treinado com 569 amostras de uma única instituição (Universidade de Wisconsin, anos 1990).",
    "Recebe medidas morfológicas já extraídas por profissional; não analisa a imagem bruta.",
    "O dataset não contém variáveis demográficas, o que torna impossível auditar viés por subgrupo.",
    "Aproximadamente 1 em cada 42 casos malignos não é sinalizado (recall ~98%).",
    "Cerca de 1 em cada 8 casos sinalizados é benigno (precisão ~87%).",
    "Sem validação prospectiva nem aprovação regulatória. Uso educacional.",
]


@dataclass
class ResultadoInferencia:
    """Saída interna da inferência, antes da serialização."""

    predicao: int
    probabilidade: float
    faixa: str
    avisos: list[str]


class ServicoPredicao:
    """Encapsula o artefato carregado e as operações de inferência."""

    def __init__(self, config: Config, carregar_explicador: bool = True) -> None:
        self.config = config
        self.pipeline: Pipeline
        self.metadados: Metadados
        self.pipeline, self.metadados = persist.carregar(config)
        self._referencia: pd.DataFrame | None = None
        self._explicador_pronto = False

        if carregar_explicador:
            self._preparar_referencia()

        logger.info(
            "Serviço de predição inicializado",
            extra={
                "modelo": self.metadados.versao_modelo,
                "algoritmo": self.metadados.algoritmo,
                "limiar": self.metadados.limiar_decisao,
            },
        )

    # --- Referência do SHAP -------------------------------------------------

    def _preparar_referencia(self) -> None:
        """Reconstrói o conjunto de treino usado como referência do SHAP.

        Preferimos reconstruí-lo a partir dos dados e da mesma semente a
        carregá-lo de um CSV: o resultado é idêntico e elimina a dependência
        de um arquivo que pode ficar dessincronizado do artefato.
        """
        try:
            from medai.models.train import preparar_dados

            dados = preparar_dados(self.config)
            x, _ = dados.treino_mais_validacao()
            self._referencia = x[self.metadados.features_esperadas]
            self._explicador_pronto = True
        except Exception as erro:  # pragma: no cover - ambiente sem dados
            logger.warning(
                "Referência do SHAP indisponível; /explain ficará desabilitado",
                extra={"erro": str(erro)},
            )
            self._referencia = None
            self._explicador_pronto = False

    @property
    def explicacao_disponivel(self) -> bool:
        return self._explicador_pronto and self._referencia is not None

    # --- Validação ----------------------------------------------------------

    def verificar_faixas(self, dados: pd.DataFrame) -> list[str]:
        """Sinaliza valores fora da faixa observada no treino.

        **Sinaliza, não recusa.** Um tumor excepcionalmente grande é
        precisamente o caso que o sistema precisa processar. Mas o modelo
        extrapola fora da faixa em que foi ajustado, e o clínico merece saber
        que aquela predição é menos confiável que as demais.
        """
        avisos: list[str] = []
        for coluna, limites in self.metadados.estatisticas_features.items():
            if coluna not in dados.columns:
                continue
            valores = dados[coluna]
            abaixo = valores < limites["minimo"]
            acima = valores > limites["maximo"]
            if abaixo.any() or acima.any():
                avisos.append(
                    f"'{coluna}' fora da faixa observada no treino "
                    f"({limites['minimo']:.4g} a {limites['maximo']:.4g}); "
                    "a predição extrapola e é menos confiável"
                )
        return avisos

    # --- Inferência ---------------------------------------------------------

    def _probabilidades(self, dados: pd.DataFrame) -> np.ndarray:
        entrada = persist.validar_entrada(dados, self.metadados)
        return self.pipeline.predict_proba(entrada)[:, 1]

    def prever(self, dados: pd.DataFrame) -> list[ResultadoInferencia]:
        """Prediz um ou mais casos."""
        probabilidades = self._probabilidades(dados)
        limiar = self.metadados.limiar_decisao
        faixas = classificar_com_faixa(probabilidades, limiar, self.config)
        avisos = self.verificar_faixas(dados)

        return [
            ResultadoInferencia(
                predicao=int(p >= limiar),
                probabilidade=float(p),
                faixa=str(f),
                avisos=avisos,
            )
            for p, f in zip(probabilidades, faixas, strict=True)
        ]

    def explicar(self, dados: pd.DataFrame, top: int = 10) -> tuple[ResultadoInferencia, dict]:
        """Prediz um caso e devolve sua explicação SHAP.

        Returns:
            O resultado da inferência e um dicionário com contribuições
            individuais, agrupadas e o valor base.
        """
        if not self.explicacao_disponivel:
            raise RuntimeError(
                "Explicação indisponível: o conjunto de referência do SHAP não pôde "
                "ser carregado. Verifique se os dados estão em data/raw/."
            )

        resultado = self.prever(dados)[0]
        entrada = persist.validar_entrada(dados, self.metadados)
        explicacao = explain.explicar(self.pipeline, entrada, x_fundo=self._referencia)

        contribuicoes = explicacao.contribuicoes_do_caso(0, top=top)
        serie = pd.Series(np.abs(explicacao.valores[0]), index=explicacao.nomes)
        agrupadas = explain.agrupar_por_medida(serie)

        return resultado, {
            "contribuicoes": [
                {
                    "feature": str(indice),
                    "contribuicao": float(linha["contribuicao"]),
                    "valor_medido": float(linha["valor_medido"]),
                    "efeito": (
                        "aumenta suspeição" if linha["contribuicao"] > 0 else "reduz suspeição"
                    ),
                }
                for indice, linha in contribuicoes.iterrows()
            ],
            "contribuicoes_agrupadas": [
                {"grupo": str(grupo), "contribuicao_absoluta": float(valor)}
                for grupo, valor in agrupadas.items()
            ],
            "valor_base": float(explicacao.valor_base),
        }

    # --- Metadados expostos -------------------------------------------------

    def info(self) -> dict:
        """Informações do modelo servido, para o endpoint ``/model/info``."""
        m = self.metadados
        return {
            "versao_modelo": m.versao_modelo,
            "algoritmo": m.algoritmo,
            "treinado_em": m.treinado_em,
            "dataset": m.dataset,
            "n_amostras_treino": m.n_amostras_treino,
            "features_esperadas": m.features_esperadas,
            "estatisticas_features": m.estatisticas_features,
            "limiar_decisao": m.limiar_decisao,
            "limiar_origem": m.limiar_origem,
            "zona_cinzenta": m.zona_cinzenta,
            "metricas_validacao": m.metricas_validacao,
            "versoes": m.versoes,
            "limitacoes": LIMITACOES,
        }

    def rotulo(self, predicao: int) -> str:
        return self.metadados.rotulo_positivo if predicao else "benigno"
