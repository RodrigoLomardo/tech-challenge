# Prints dos resultados

Evidências visuais da execução do pipeline MedAI Triage (Breast Cancer Wisconsin, 569 amostras).

> **Aviso clínico:** resultado de apoio à triagem. Não constitui diagnóstico. A decisão clínica é do médico responsável. Projeto educacional.

## Como foram gerados

Todo o pipeline foi executado dentro da imagem Docker do projeto (`docker run medai-triage ...`), com `reports/` e `models/` apontando para pastas temporárias — o repositório não foi alterado. A UI e a API foram capturadas com `docker compose up` e Chrome headless.

| Pasta | Conteúdo | Origem |
|---|---|---|
| `1_dados_e_eda/` | 10 figuras da análise exploratória (classes, distribuições, correlação, PCA, outliers) | `scripts/make_report_assets.py` |
| `2_modelagem_e_avaliacao/` | 14 figuras: matriz de confusão, curvas, comparação de modelos, coeficientes, SHAP, robustez | `scripts/evaluate_model.py` |
| `3_interface_streamlit/` | Painel: formulário, resultado benigno e maligno, triagem de fila, "Sobre o modelo" | `docker compose up` → `:8501` |
| `4_api_fastapi/` | Swagger (`/docs`) e ReDoc (`/redoc`) | `:8000` |
| `5_terminal/` | Saída de cada etapa: dados, EDA, treino, avaliação, Docker/health e testes | logs da execução |

## Resultados principais (conjunto de teste, n = 114, limiar 0,158)

| Métrica | Valor |
|---|---|
| Recall (maligno) | 0,976 (41 de 42; 1 falso negativo) |
| Precisão | 0,872 (6 falsos positivos) |
| ROC-AUC | 0,997 |
| Testes automatizados | 168 aprovados, cobertura 72% |

Com 114 amostras de teste, 1 caso a mais ou a menos muda o recall em ~2 pontos percentuais: leia estes números com essa margem.
