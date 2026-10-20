# Checklist de Requisitos — MedAI Triage

Rastreabilidade entre cada exigência do enunciado do desafio e onde ela é atendida.
Marque os itens conforme a execução avança. Nenhum item pode ficar sem artefato correspondente.

Legenda de status: `[ ]` pendente · `[~]` em andamento · `[x]` concluído

---

## 1. Processamento de dados médicos

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 1.1 | Classificação de exames com Machine Learning sobre base tabular | 4–5 | `src/medai/models/`, `notebooks/03` | [ ] |
| 1.2 | Diagnóstico binário (tem / não tem a doença) | 1 | alvo `diagnosis` M=1 / B=0 | [x] |
| 1.3 | **EXTRA** — diagnóstico com imagem via CNN | 8 (opcional) | `src/medai/vision/`, `notebooks/05` | [ ] |

## 2. Dados e modelos

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 2.1 | Escolha de dataset médico público | 1 | Breast Cancer Wisconsin (Diagnostic) | [x] |
| 2.2 | Definição explícita do problema a resolver | 1 | `RELATORIO_TECNICO.md` §2 | [x] |
| 2.3 | Link de download documentado | 10 | `README.md` seção Dataset | [ ] |

## 3. Exploração de dados

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 3.1 | Carregar a base e expor suas características | 2 | `notebooks/01`, `src/medai/data/loader.py` | [x] |
| 3.2 | Estatísticas descritivas | 2 | `notebooks/01`, `src/medai/data/explore.py` | [x] |
| 3.3 | Visualização de distribuições relevantes | 2 | `reports/figures/01–10` (10 PNGs) | [x] |
| 3.4 | **Discussão dos resultados da EDA** | 2 | `notebooks/01` + `RELATORIO_TECNICO.md` §4 | [x] |

## 4. Pré-processamento

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 4.1 | Limpeza: valores ausentes e inconsistentes | 3 | `src/medai/features/cleaning.py` | [ ] |
| 4.2 | Pipeline de pré-processamento em Python | 3 | `src/medai/features/build_pipeline.py` | [ ] |
| 4.3 | Conversão de variáveis categóricas e numéricas | 3 | `ColumnTransformer` (ramos num + cat) | [ ] |
| 4.4 | **Análise de correlação** | 3 | `src/medai/features/correlation.py`, `notebooks/02` | [ ] |

## 5. Modelagem

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 5.1 | Duas ou mais técnicas de classificação | 4 | 5 modelos + baseline em `registry.py` | [ ] |
| 5.2 | Separação clara treino / validação / teste | 3 | `src/medai/data/split.py` (60/20/20 estratificado) | [ ] |

## 6. Treinamento e avaliação

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 6.1 | Treinar com o conjunto de treinamento | 4–5 | `src/medai/models/train.py` | [ ] |
| 6.2 | Avaliar no teste com accuracy, recall e F1-score | 6 | `reports/metrics/test_metrics.json` | [ ] |
| 6.3 | **Discussão da escolha da métrica face ao problema** | 6 | `RELATORIO_TECNICO.md` §10 | [ ] |
| 6.4 | Interpretação via feature importance | 6 | `src/medai/models/explain.py` | [ ] |
| 6.5 | Interpretação via **SHAP** (global e local) | 6 | `reports/figures/shap_*.png` | [ ] |
| 6.6 | **Discussão crítica: o modelo serve na prática? como?** | 6 | `RELATORIO_TECNICO.md` §12 | [ ] |
| 6.7 | Diagnóstico final sempre do médico (flexibilidade) | 4, 7 | zona cinzenta + aviso clínico na API/UI | [ ] |

## 7. Entregáveis finais

| # | Requisito | Fase | Artefato | Status |
|---|-----------|------|----------|--------|
| 7.1 | Dockerfile | 9 | `docker/Dockerfile` (variante GPU cancelada — sem CUDA) | [ ] |
| 7.2 | README.md com instruções de execução | 10 | `README.md` | [ ] |
| 7.3 | Dataset ou link para download | 10 | `scripts/download_data.py` + README | [ ] |
| 7.4 | Resultados: prints, gráficos e análises | 6, 10 | `reports/figures/`, `reports/metrics/` | [ ] |
| 7.5 | Relatório técnico: pré-processamento, modelos e por quê, resultados interpretados | 10 | `RELATORIO_TECNICO.md` | [ ] |
| 7.6 | Código documentado por completo | todas | docstrings + type hints + `ruff`/`black` limpos | [ ] |

## 8. Além do exigido (diferenciais desta implementação)

| # | Item | Fase | Status |
|---|------|------|--------|
| 8.1 | API FastAPI com validação pydantic e docs automáticas | 7 | [ ] |
| 8.2 | UI Streamlit (caso individual, lote, transparência do modelo) | 7 | [ ] |
| 8.3 | Calibração de probabilidade + Brier score | 4, 6 | [ ] |
| 8.4 | Ajuste de limiar orientado a recall + zona cinzenta | 4 | [ ] |
| 8.5 | Rastreamento de experimentos (`reports/experiments.csv`) | 4 | [ ] |
| 8.6 | Testes automatizados (dados, pipeline, modelo, API) | todas | [ ] |
| 8.7 | Análise de robustez com múltiplas sementes | 6 | [ ] |
| 8.8 | Pipeline parametrizado por YAML (permite plugar o Pima Diabetes) | 3 | [ ] |
| 8.9 | Grad-CAM para interpretabilidade visual da CNN | 8 (opcional) | [ ] |

---

## Conferência final antes de declarar o desafio concluído

- [ ] Todos os itens das seções 1–7 marcados como concluídos.
- [ ] `make test` verde.
- [ ] Os 4 notebooks executam do zero, em ordem, sem erro.
- [ ] `docker compose up` sobe API e UI em máquina limpa.
- [ ] Nenhum dado grande, credencial ou `kaggle.json` versionado no Git.
- [ ] Aviso clínico presente no README, no relatório, na resposta da API e na UI.
