# MedAI Triage — imagem única (CPU) para API, UI e treino.
#
# Uma imagem só, com o comando escolhido pelo docker-compose: API e UI usam o
# mesmo pacote `medai` e as mesmas dependências, então duas imagens apenas
# duplicariam camadas sem ganho. Sem CUDA (ADR-011).
#
#   docker build -t medai-triage .
#   docker run --rm -p 8000:8000 medai-triage                  # API (padrão)
#   docker run --rm -p 8501:8501 -e MEDAI_API_URL=http://host.docker.internal:8000 \
#       medai-triage streamlit run src/medai/ui/app.py --server.address=0.0.0.0

# ---------------------------------------------------------------------------
# Estágio 1 — builder: resolve e compila as dependências em um venv isolado.
# Compiladores ficam aqui e não chegam à imagem final.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Dependências antes do código-fonte: alterar um .py não invalida esta camada.
# `requirements.txt` entra como *constraints* (-c): fixa a versão exata de tudo
# que for resolvido (scikit-learn 1.5.2 é crítico para carregar o .joblib), mas
# sem instalar JupyterLab, pytest, ruff etc., que não servem em runtime.
COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -c requirements.txt ".[viz,models,explain,serve]"

# ---------------------------------------------------------------------------
# Estágio 2 — runtime: só o necessário, usuário não-root.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1: runtime OpenMP exigido pelo xgboost. curl: usado no HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 medai

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    OMP_NUM_THREADS=4 \
    MPLBACKEND=Agg \
    MEDAI_LOG_FORMAT=json \
    MEDAI_CONFIG=configs/breast_cancer.yaml \
    MEDAI_API_URL=http://localhost:8000

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=medai:medai configs/ ./configs/
COPY --chown=medai:medai scripts/ ./scripts/
COPY --chown=medai:medai src/ ./src/
COPY --chown=medai:medai models/ ./models/
COPY --chown=medai:medai pyproject.toml ./

# Diretórios graváveis pelo usuário não-root (o .dockerignore exclui data/).
RUN mkdir -p data/raw data/interim data/processed reports/metrics reports/figures \
    && chown -R medai:medai /app

USER medai

# O pacote já está instalado no venv (cópia não editável); src/ no PYTHONPATH
# garante que scripts/ e `python -m medai...` enxerguem o mesmo código.
ENV PYTHONPATH=/app/src

# Dataset materializado na build: a API reconstrói o conjunto de treino para a
# referência do SHAP e precisa do CSV. Sem credencial Kaggle o script cai no
# loader do scikit-learn (mesma origem UCI, ADR-002) — não exige rede.
RUN python scripts/download_data.py --config configs/breast_cancer.yaml

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "medai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
