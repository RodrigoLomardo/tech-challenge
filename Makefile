# MedAI Triage — atalhos de desenvolvimento.
# Uso: make <alvo>. Rode `make help` para a lista completa.

PYTHON      := python3.11
VENV        := .venv
BIN         := $(VENV)/bin
CONFIG      ?= configs/breast_cancer.yaml
API_PORT    ?= 8000
UI_PORT     ?= 8501

.DEFAULT_GOAL := help
.PHONY: help setup data eda train evaluate api ui test lint format notebook clean docker-build docker-up docker-down freeze

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente ---------------------------------------------------------------

setup: $(VENV)/.installed  ## Cria o venv e instala as dependências

$(VENV)/.installed: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip setuptools wheel
	$(BIN)/pip install -e ".[all]"
	$(BIN)/nbstripout --install --attributes .gitattributes || true
	@touch $@
	@echo "Ambiente pronto. Ative com: source activate.sh"

freeze:  ## Congela as versões instaladas em requirements.lock.txt
	$(BIN)/pip freeze --exclude-editable > requirements.lock.txt
	@echo "Lock gravado em requirements.lock.txt"

# --- Pipeline de dados e modelo ---------------------------------------------

data:  ## Baixa/prepara o dataset (Kaggle se disponível, senão sklearn)
	$(BIN)/python scripts/download_data.py --config $(CONFIG)

eda:  ## Gera as figuras de EDA em reports/figures/
	$(BIN)/python scripts/make_report_assets.py --config $(CONFIG)

train:  ## Treina os modelos, seleciona o campeão e serializa o artefato
	$(BIN)/python -m medai.models.train --config $(CONFIG)

evaluate:  ## Avalia no conjunto de teste e gera figuras de interpretabilidade
	$(BIN)/python scripts/evaluate_model.py --config $(CONFIG)

# --- Serviço ----------------------------------------------------------------

api:  ## Sobe a API FastAPI em modo recarregamento
	$(BIN)/uvicorn medai.api.main:app --reload --host 0.0.0.0 --port $(API_PORT)

ui:  ## Sobe o painel Streamlit
	$(BIN)/streamlit run src/medai/ui/app.py --server.port $(UI_PORT)

notebook:  ## Abre o JupyterLab
	$(BIN)/jupyter lab --notebook-dir=notebooks

# --- Qualidade --------------------------------------------------------------

test:  ## Roda a suíte de testes com cobertura
	$(BIN)/pytest

lint:  ## Verifica estilo e erros estáticos
	$(BIN)/ruff check src tests scripts
	$(BIN)/black --check src tests scripts

format:  ## Formata e corrige automaticamente o que for possível
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/black src tests scripts

# --- Docker -----------------------------------------------------------------

docker-build:  ## Constrói as imagens
	docker compose build

docker-up:  ## Sobe API + UI
	docker compose up -d
	@echo "API: http://localhost:$(API_PORT)/docs  |  UI: http://localhost:$(UI_PORT)"

docker-down:  ## Derruba os serviços
	docker compose down

# --- Limpeza ----------------------------------------------------------------

clean:  ## Remove caches e artefatos temporários (preserva dados e modelos)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
