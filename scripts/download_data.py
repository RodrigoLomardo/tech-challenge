#!/usr/bin/env python
"""Materializa o dataset em ``data/raw/``.

Tenta, nesta ordem:

1. **CSV já presente** — nada a fazer, apenas valida.
2. **Kaggle CLI** — se a credencial ``~/.kaggle/kaggle.json`` existir.
3. **scikit-learn** — fallback sem credencial, mesma origem UCI (ADR-002).

Uso:
    python scripts/download_data.py --config configs/breast_cancer.yaml
    python scripts/download_data.py --config configs/breast_cancer.yaml --forcar-sklearn
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from medai.config import RAIZ_PROJETO, Config, carregar_config
from medai.data.loader import carregar_breast_cancer_sklearn
from medai.data.validate import validar_dataset
from medai.logging_conf import configurar_logging, get_logger

logger = get_logger(__name__)


def _credencial_kaggle_disponivel() -> bool:
    """Verifica se há credencial do Kaggle configurada."""
    por_arquivo = Path.home() / ".kaggle" / "kaggle.json"
    por_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    return por_arquivo.exists() or por_env


def _baixar_do_kaggle(kaggle_id: str, destino: Path) -> bool:
    """Baixa e extrai o dataset do Kaggle. Devolve ``False`` se não for possível.

    O conteúdo baixado é extraído em diretório temporário isolado e só o CSV
    esperado é copiado para ``data/raw/`` — um arquivo de terceiros não deve
    ser descompactado direto na árvore do projeto.
    """
    if shutil.which("kaggle") is None:
        logger.info("CLI do Kaggle não instalada (pip install kaggle)")
        return False
    if not _credencial_kaggle_disponivel():
        logger.info("Credencial do Kaggle ausente (~/.kaggle/kaggle.json)")
        return False

    with tempfile.TemporaryDirectory(prefix="medai_kaggle_") as tmp:
        comando = ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", tmp, "--unzip"]
        logger.info("Baixando do Kaggle", extra={"dataset": kaggle_id})
        resultado = subprocess.run(comando, capture_output=True, text=True, check=False)

        if resultado.returncode != 0:
            logger.warning(
                "Download do Kaggle falhou", extra={"stderr": resultado.stderr.strip()[:300]}
            )
            return False

        csvs = sorted(Path(tmp).glob("*.csv"))
        if not csvs:
            logger.warning("Nenhum CSV no pacote baixado")
            return False

        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csvs[0], destino)
        logger.info("CSV gravado", extra={"origem": csvs[0].name, "destino": str(destino)})
        return True


def _gerar_do_sklearn(destino: Path) -> None:
    """Grava o dataset do scikit-learn no schema canônico."""
    df = carregar_breast_cancer_sklearn()
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    logger.info(
        "CSV gerado a partir do scikit-learn",
        extra={"destino": str(destino), "linhas": len(df)},
    )


def obter_dataset(config: Config, *, forcar_sklearn: bool = False) -> Path:
    """Garante que o CSV exista em ``data/raw/`` e devolve seu caminho."""
    caminho = config.dataset.caminho_csv
    if caminho is None:
        raise ValueError(
            f"A configuração '{config.dataset.nome}' não declara 'caminho_csv'; "
            "nada a materializar."
        )
    destino = caminho if caminho.is_absolute() else RAIZ_PROJETO / caminho

    if destino.exists() and not forcar_sklearn:
        logger.info("CSV já presente, nada a baixar", extra={"caminho": str(destino)})
        return destino

    if (
        not forcar_sklearn
        and config.dataset.kaggle_id
        and _baixar_do_kaggle(config.dataset.kaggle_id, destino)
    ):
        return destino

    if config.dataset.nome != "breast_cancer_wisconsin":
        raise RuntimeError(
            f"Não foi possível obter '{config.dataset.nome}' e não há fallback para ele.\n"
            f"Baixe manualmente de {config.dataset.url_referencia}\n"
            f"e salve em {destino}"
        )

    _gerar_do_sklearn(destino)
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/breast_cancer.yaml")
    parser.add_argument(
        "--forcar-sklearn",
        action="store_true",
        help="Ignora CSV existente e Kaggle; regenera a partir do scikit-learn",
    )
    args = parser.parse_args(argv)

    configurar_logging()
    config = carregar_config(args.config)
    caminho = obter_dataset(config, forcar_sklearn=args.forcar_sklearn)

    # Validar aqui, e não só no treino, dá o retorno imediato de que o
    # arquivo materializado está utilizável.
    import pandas as pd

    resultado = validar_dataset(pd.read_csv(caminho), config)
    print(f"\nDataset: {caminho.relative_to(RAIZ_PROJETO)}")
    print(resultado)

    if not resultado.valido:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
