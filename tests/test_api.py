"""Testes da API.

O teste mais importante é `test_api_concorda_com_o_modelo_direto`: se a API
divergir do pipeline chamado diretamente, o que é servido não é o que foi
avaliado na Fase 6.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from medai import AVISO_CLINICO
from medai.api.main import app
from medai.config import Config
from medai.models import persist
from medai.models.train import preparar_dados


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def dados(config_breast_cancer: Config):
    return preparar_dados(config_breast_cancer)


@pytest.fixture(scope="module")
def caso_maligno(dados) -> dict:
    return dados.x_teste[dados.y_teste == 1].iloc[0].to_dict()


@pytest.fixture(scope="module")
def caso_benigno(dados) -> dict:
    return dados.x_teste[dados.y_teste == 0].iloc[0].to_dict()


# --- Serviço -----------------------------------------------------------------


def test_health(cliente) -> None:
    corpo = cliente.get("/health").json()
    assert corpo["status"] == "ok"
    assert corpo["modelo_carregado"]
    assert corpo["versao_modelo"]


def test_info_do_modelo(cliente) -> None:
    corpo = cliente.get("/model/info").json()
    assert corpo["algoritmo"] == "logistic_regression"
    assert len(corpo["features_esperadas"]) == 30
    assert corpo["limiar_decisao"] < 0.5
    assert len(corpo["estatisticas_features"]) == 30


def test_info_expoe_as_limitacoes(cliente) -> None:
    """Quem integra o sistema precisa das limitações tanto quanto das métricas."""
    limitacoes = cliente.get("/model/info").json()["limitacoes"]
    assert len(limitacoes) >= 5
    texto = " ".join(limitacoes).lower()
    assert "viés" in texto or "vies" in texto
    assert "regulatória" in texto or "regulatoria" in texto


def test_documentacao_disponivel(cliente) -> None:
    assert cliente.get("/docs").status_code == 200
    esquema = cliente.get("/openapi.json").json()
    assert "/predict" in esquema["paths"]
    assert "/explain" in esquema["paths"]


# --- Predição ----------------------------------------------------------------


def test_caso_maligno_conhecido(cliente, caso_maligno: dict) -> None:
    corpo = cliente.post("/predict", json=caso_maligno).json()
    assert corpo["predicao"] == 1
    assert corpo["rotulo"] == "maligno"
    assert corpo["probabilidade_malignidade"] > 0.5
    assert corpo["faixa_decisao"] == "maligno"


def test_caso_benigno_conhecido(cliente, caso_benigno: dict) -> None:
    corpo = cliente.post("/predict", json=caso_benigno).json()
    assert corpo["predicao"] == 0
    assert corpo["probabilidade_malignidade"] < 0.5


def test_aviso_clinico_em_toda_resposta(cliente, caso_maligno: dict) -> None:
    """Exigência de projeto: nenhuma predição sai sem o aviso."""
    for rota in ("/predict", "/explain"):
        assert cliente.post(rota, json=caso_maligno).json()["aviso_clinico"] == AVISO_CLINICO
    lote = cliente.post("/predict/batch", json=[caso_maligno]).json()
    assert lote["aviso_clinico"] == AVISO_CLINICO


def test_api_concorda_com_o_modelo_direto(
    cliente, caso_maligno: dict, config_breast_cancer: Config
) -> None:
    """A API precisa reproduzir exatamente o pipeline avaliado na Fase 6."""
    pipeline, metadados = persist.carregar(config_breast_cancer)
    quadro = pd.DataFrame([caso_maligno])[metadados.features_esperadas]
    esperado = float(pipeline.predict_proba(quadro)[:, 1][0])

    obtido = cliente.post("/predict", json=caso_maligno).json()["probabilidade_malignidade"]
    assert obtido == pytest.approx(esperado, abs=1e-4)


def test_nomes_saneados_funcionam(cliente, caso_maligno: dict) -> None:
    """'concave points_mean' e 'concave_points_mean' devem ser equivalentes."""
    saneado = {k.replace(" ", "_"): v for k, v in caso_maligno.items()}
    a = cliente.post("/predict", json=caso_maligno).json()
    b = cliente.post("/predict", json=saneado).json()
    assert a["probabilidade_malignidade"] == b["probabilidade_malignidade"]


def test_ordem_das_colunas_nao_importa(cliente, caso_maligno: dict) -> None:
    invertido = dict(reversed(list(caso_maligno.items())))
    a = cliente.post("/predict", json=caso_maligno).json()
    b = cliente.post("/predict", json=invertido).json()
    assert a["probabilidade_malignidade"] == b["probabilidade_malignidade"]


# --- Validação de entrada ----------------------------------------------------


def test_feature_faltante_gera_422(cliente, caso_maligno: dict) -> None:
    incompleto = {k: v for k, v in caso_maligno.items() if k != "radius_mean"}
    resposta = cliente.post("/predict", json=incompleto)
    assert resposta.status_code == 422
    assert "radius_mean" in str(resposta.json())


def test_valor_negativo_gera_422(cliente, caso_maligno: dict) -> None:
    """Medida física negativa indica corrupção, não caso atípico."""
    invalido = dict(caso_maligno)
    invalido["area_mean"] = -1.0
    assert cliente.post("/predict", json=invalido).status_code == 422


def test_valor_nao_numerico_gera_422(cliente, caso_maligno: dict) -> None:
    invalido = dict(caso_maligno)
    invalido["radius_mean"] = "não é número"
    assert cliente.post("/predict", json=invalido).status_code == 422


def test_valor_fora_da_faixa_e_aceito_com_aviso(cliente, caso_maligno: dict) -> None:
    """Um tumor excepcionalmente grande é o caso a processar, não a rejeitar.

    O modelo extrapola fora da faixa de treino, e o clínico precisa saber —
    mas recusar a predição seria pior.
    """
    extremo = dict(caso_maligno)
    extremo["area_worst"] = 99_999.0
    corpo = cliente.post("/predict", json=extremo).json()
    assert corpo["avisos"]
    assert "area_worst" in corpo["avisos"][0]
    assert "fora da faixa" in corpo["avisos"][0]


def test_entrada_valida_nao_gera_aviso(cliente, caso_maligno: dict) -> None:
    assert cliente.post("/predict", json=caso_maligno).json()["avisos"] == []


# --- Explicação --------------------------------------------------------------


def test_explain_devolve_as_duas_leituras(cliente, caso_maligno: dict) -> None:
    corpo = cliente.post("/explain", json=caso_maligno).json()
    assert len(corpo["contribuicoes"]) == 10
    assert len(corpo["contribuicoes_agrupadas"]) == 6
    assert {"feature", "contribuicao", "valor_medido", "efeito"} <= set(corpo["contribuicoes"][0])


def test_explain_ordena_por_magnitude(cliente, caso_maligno: dict) -> None:
    contribuicoes = cliente.post("/explain", json=caso_maligno).json()["contribuicoes"]
    magnitudes = [abs(c["contribuicao"]) for c in contribuicoes]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_mantem_a_mesma_probabilidade(cliente, caso_maligno: dict) -> None:
    a = cliente.post("/predict", json=caso_maligno).json()["probabilidade_malignidade"]
    b = cliente.post("/explain", json=caso_maligno).json()["probabilidade_malignidade"]
    assert a == b


def test_agrupamento_lidera_com_tamanho(cliente, caso_maligno: dict) -> None:
    """A leitura agrupada é a clinicamente correta sob multicolinearidade."""
    grupos = cliente.post("/explain", json=caso_maligno).json()["contribuicoes_agrupadas"]
    assert grupos[0]["grupo"].startswith("tamanho")


# --- Lote --------------------------------------------------------------------


def test_lote_ordena_por_suspeicao(cliente, dados) -> None:
    exames = dados.x_teste.head(20).to_dict(orient="records")
    corpo = cliente.post("/predict/batch", json=exames).json()

    assert corpo["total"] == 20
    probabilidades = [r["probabilidade_malignidade"] for r in corpo["resultados"]]
    assert probabilidades == sorted(probabilidades, reverse=True)
    assert sum(corpo["resumo_faixas"].values()) == 20


def test_lote_vazio_gera_422(cliente) -> None:
    assert cliente.post("/predict/batch", json=[]).status_code == 422


def test_csv_valido(cliente, dados, config_breast_cancer: Config) -> None:
    _, metadados = persist.carregar(config_breast_cancer)
    buffer = io.StringIO()
    dados.x_teste.head(10)[metadados.features_esperadas].to_csv(buffer, index=False)

    resposta = cliente.post(
        "/predict/csv",
        files={"arquivo": ("exames.csv", buffer.getvalue(), "text/csv")},
    )
    assert resposta.status_code == 200
    assert resposta.json()["total"] == 10


def test_csv_com_coluna_faltando_gera_422(cliente, dados) -> None:
    buffer = io.StringIO()
    dados.x_teste.head(3).drop(columns=["radius_mean"]).to_csv(buffer, index=False)

    resposta = cliente.post(
        "/predict/csv", files={"arquivo": ("ruim.csv", buffer.getvalue(), "text/csv")}
    )
    assert resposta.status_code == 422
    assert "radius_mean" in str(resposta.json())


def test_arquivo_ilegivel_gera_422(cliente) -> None:
    resposta = cliente.post(
        "/predict/csv", files={"arquivo": ("x.csv", b"\x00\x01\x02 nao e csv", "text/csv")}
    )
    assert resposta.status_code == 422


# --- Faixas ------------------------------------------------------------------


def test_zona_cinzenta_aparece_no_teste(cliente, dados) -> None:
    """A faixa de revisão precisa existir na prática, não só no código."""
    exames = dados.x_teste.to_dict(orient="records")
    corpo = cliente.post("/predict/batch", json=exames).json()
    assert corpo["resumo_faixas"].get("zona_cinzenta_revisao_medica", 0) > 0


def test_toda_resposta_traz_recomendacao(cliente, dados) -> None:
    exames = dados.x_teste.head(15).to_dict(orient="records")
    for resultado in cliente.post("/predict/batch", json=exames).json()["resultados"]:
        assert resultado["recomendacao"]
        assert resultado["faixa_decisao"] in {
            "benigno",
            "zona_cinzenta_revisao_medica",
            "maligno",
        }
