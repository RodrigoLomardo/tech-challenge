"""Testes de carregamento e validação do dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from medai.config import Config, carregar_config
from medai.data.loader import (
    FEATURES_CANONICAS,
    _traduzir_nome_sklearn,
    carregar_breast_cancer_sklearn,
    carregar_dados_brutos,
)
from medai.data.validate import ErroDeValidacao, validar_dataset


@pytest.fixture(scope="module")
def df_bruto() -> pd.DataFrame:
    return carregar_breast_cancer_sklearn()


# --- Tradução de nomes -------------------------------------------------------


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("mean radius", "radius_mean"),
        ("radius error", "radius_se"),
        ("worst radius", "radius_worst"),
        # O Kaggle usa underscore em fractal_dimension mas mantém o espaço em
        # "concave points" — inconsistência da fonte que reproduzimos.
        ("mean fractal dimension", "fractal_dimension_mean"),
        ("worst concave points", "concave points_worst"),
        ("concave points error", "concave points_se"),
    ],
)
def test_traducao_para_schema_canonico(entrada: str, esperado: str) -> None:
    assert _traduzir_nome_sklearn(entrada) == esperado


def test_nome_desconhecido_falha() -> None:
    with pytest.raises(ValueError):
        _traduzir_nome_sklearn("algo inesperado aqui")


# --- Carregamento ------------------------------------------------------------


def test_dimensoes_conhecidas(df_bruto: pd.DataFrame) -> None:
    assert df_bruto.shape == (569, 31)  # 30 features + diagnosis


def test_schema_canonico_completo(df_bruto: pd.DataFrame) -> None:
    assert set(df_bruto.columns) == {"diagnosis", *FEATURES_CANONICAS}
    assert len(FEATURES_CANONICAS) == 30


def test_mapeamento_de_classe_nao_esta_invertido(df_bruto: pd.DataFrame) -> None:
    """O scikit-learn usa target=0 para maligno; confundir isso inverte recall.

    Ancoramos no fato conhecido do dataset: 212 malignos e 357 benignos.
    Se o mapeamento inverter, as contagens trocam e o teste falha.
    """
    contagem = df_bruto["diagnosis"].value_counts()
    assert contagem["M"] == 212
    assert contagem["B"] == 357


def test_malignos_tem_tumor_maior_que_benignos(df_bruto: pd.DataFrame) -> None:
    """Checagem clínica independente da contagem.

    Tumores malignos são sistematicamente maiores. Se o rótulo estiver
    invertido, esta relação se inverte — é uma segunda trava contra o erro
    mais caro possível neste projeto.
    """
    medias = df_bruto.groupby("diagnosis")["area_mean"].mean()
    assert medias["M"] > medias["B"]


def test_sem_valores_ausentes_na_rota_sklearn(df_bruto: pd.DataFrame) -> None:
    assert not df_bruto.isna().any().any()


def test_features_sao_numericas(df_bruto: pd.DataFrame) -> None:
    numericas = df_bruto.drop(columns=["diagnosis"]).select_dtypes("number")
    assert numericas.shape[1] == 30


def test_fallback_para_sklearn_quando_csv_ausente(tmp_path, config_breast_cancer: Config) -> None:
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.dataset.caminho_csv = tmp_path / "inexistente.csv"
    df = carregar_dados_brutos(cfg)
    assert len(df) == 569


def test_sem_fallback_falha_explicitamente(tmp_path, config_breast_cancer: Config) -> None:
    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.dataset.caminho_csv = tmp_path / "inexistente.csv"
    with pytest.raises(FileNotFoundError):
        carregar_dados_brutos(cfg, permitir_fallback=False)


def test_dataset_sem_equivalente_embutido_nao_usa_fallback(tmp_path) -> None:
    """Fallback silencioso para a base errada seria pior que falhar."""
    cfg = carregar_config("configs/diabetes.yaml")
    cfg.dataset.caminho_csv = tmp_path / "inexistente.csv"
    with pytest.raises(FileNotFoundError, match="não há fallback"):
        carregar_dados_brutos(cfg)


# --- Validação ---------------------------------------------------------------


def test_dataset_real_passa_na_validacao(
    df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    resultado = validar_dataset(df_bruto, config_breast_cancer)
    assert resultado.valido, resultado.erros
    assert resultado.resumo["positivos"].startswith("212")


def test_alvo_ausente_e_erro(df_bruto: pd.DataFrame, config_breast_cancer: Config) -> None:
    resultado = validar_dataset(df_bruto.drop(columns=["diagnosis"]), config_breast_cancer)
    assert not resultado.valido
    with pytest.raises(ErroDeValidacao):
        resultado.levantar_se_invalido()


def test_classe_positiva_inexistente_e_erro(
    df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    df = df_bruto.copy()
    df["diagnosis"] = df["diagnosis"].replace({"M": "maligno", "B": "benigno"})
    resultado = validar_dataset(df, config_breast_cancer)
    assert not resultado.valido
    assert any("positiva" in e for e in resultado.erros)


def test_alvo_nao_binario_e_erro(df_bruto: pd.DataFrame, config_breast_cancer: Config) -> None:
    df = df_bruto.copy()
    df.loc[df.index[:5], "diagnosis"] = "X"
    resultado = validar_dataset(df, config_breast_cancer)
    assert not resultado.valido
    assert any("binário" in e for e in resultado.erros)


def test_dataset_vazio_e_erro(config_breast_cancer: Config) -> None:
    resultado = validar_dataset(pd.DataFrame(), config_breast_cancer)
    assert not resultado.valido


def test_coluna_totalmente_nula_gera_aviso(
    df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    """Reproduz a 'Unnamed: 32' do CSV do Kaggle."""
    df = df_bruto.copy()
    df["coluna_fantasma"] = pd.NA
    resultado = validar_dataset(df, config_breast_cancer)
    assert resultado.valido  # é aviso, não erro
    assert any("100% nula" in a for a in resultado.avisos)


def test_duplicatas_geram_aviso(df_bruto: pd.DataFrame, config_breast_cancer: Config) -> None:
    df = pd.concat([df_bruto, df_bruto.head(3)], ignore_index=True)
    resultado = validar_dataset(df, config_breast_cancer)
    assert any("duplicada" in a for a in resultado.avisos)


def test_valor_negativo_em_medida_fisica_gera_aviso(
    df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    df = df_bruto.copy()
    df.loc[df.index[0], "area_mean"] = -1.0
    resultado = validar_dataset(df, config_breast_cancer)
    assert any("negativos" in a for a in resultado.avisos)


# --- Rota CSV ----------------------------------------------------------------


def test_carrega_csv_com_peculiaridades_do_kaggle(
    tmp_path, df_bruto: pd.DataFrame, config_breast_cancer: Config
) -> None:
    """Simula o CSV do Kaggle: coluna 'id' e a coluna fantasma 'Unnamed: 32'."""
    df = df_bruto.copy()
    df.insert(0, "id", range(1, len(df) + 1))
    df["Unnamed: 32"] = pd.NA

    caminho = tmp_path / "kaggle.csv"
    df.to_csv(caminho, index=False)

    cfg = config_breast_cancer.model_copy(deep=True)
    cfg.dataset.caminho_csv = caminho
    carregado = carregar_dados_brutos(cfg)

    assert carregado.shape == (569, 33)  # 31 + id + Unnamed: 32
    assert "id" in carregado.columns
    # Ambas estão declaradas para descarte na configuração.
    assert set(cfg.dataset.colunas_descartar) <= set(carregado.columns)


def test_csv_inexistente_na_rota_direta() -> None:
    from medai.data.loader import carregar_csv

    with pytest.raises(FileNotFoundError):
        carregar_csv("data/raw/nao_existe_jamais.csv")
