# Plano — MedAI Triage: Sistema de Suporte ao Diagnóstico (Tech Challenge Pessoal)

> Documento mestre de planejamento. Fonte da verdade para a execução das fases.
> Complementos: [CHECKLIST_REQUISITOS.md](CHECKLIST_REQUISITOS.md) · [DECISOES_TECNICAS.md](DECISOES_TECNICAS.md)

## Contexto

Você se impôs um desafio técnico pessoal: construir, do zero e com desenvolvimento assistido, a **base de um sistema de IA de suporte ao diagnóstico** para um hospital universitário fictício. O problema de negócio simulado é o gargalo de triagem: volume crescente de exames e prontuários, equipe clínica limitada, risco de erro por fadiga. A solução inicial deve acelerar a triagem destacando casos de alta probabilidade de malignidade/doença — **nunca substituindo o diagnóstico médico**, que permanece como decisão final.

O escopo obrigatório cobre: processamento de dados médicos, EDA, pré-processamento, modelagem com ≥2 algoritmos de classificação, split treino/validação/teste, avaliação com métricas adequadas ao domínio clínico, interpretabilidade (feature importance + SHAP), discussão crítica de aplicabilidade real, Docker, README e relatório técnico. Visão computacional com CNN entra como **fase extra opcional**, decidida por você ao final.

**Decisões já tomadas (respostas do questionário):**
- Dataset principal: **Breast Cancer Wisconsin (Diagnostic)** — 569 amostras, 30 features numéricas, alvo binário M/B.
- Entrega: **API FastAPI + UI Streamlit**, tudo em Docker Compose, além dos notebooks.
- CNN (pneumonia em raio-X): **planejada em detalhe como Fase 8 opcional**, executada só se você quiser.
- Hardware: **sem GPU NVIDIA** (verificado na Fase 0 — a máquina tem apenas iGPU AMD Phoenix1, sem CUDA). A Fase 8 roda em **CPU**: 12 núcleos, 30 GB de RAM. Ver [AMBIENTE.md](AMBIENTE.md).

**Resultado esperado ao final:** repositório autocontido, reprodutível com um comando (`docker compose up`), com notebooks narrados, pipeline de ML versionado, API servindo predições explicáveis, painel clínico demonstrável, relatório técnico e README.

---

## Nome e identidade do projeto

**MedAI Triage** — "Sistema Inteligente de Suporte à Triagem Diagnóstica do Hospital Universitário Santa Clara (fictício)".
Esse enquadramento deve aparecer no README, no relatório e na UI, porque sustenta as decisões técnicas (por que recall importa mais que accuracy, por que a UI mostra probabilidade e não só rótulo, por que existe uma faixa de "encaminhar para revisão médica").

---

## Arquitetura alvo

```
tech-challenge/
├── .gitignore
├── .dockerignore
├── .env.example
├── README.md                      # visão geral + como executar
├── RELATORIO_TECNICO.md           # relatório exigido pelo desafio
├── pyproject.toml                 # deps + config de lint/format/pytest
├── Makefile                       # atalhos: setup, data, eda, train, api, ui, test, lint, docker
├── docker/
│   ├── Dockerfile                 # imagem base CPU (API + UI + treino)
│   ├── Dockerfile.gpu             # variante CUDA para a Fase 8 (CNN)
│   └── entrypoint.sh
├── docker-compose.yml             # serviços: api, ui, (opcional) jupyter
├── configs/
│   ├── breast_cancer.yaml         # config do dataset principal
│   ├── diabetes.yaml              # (bônus) mesmo pipeline, outro dataset
│   └── cnn_pneumonia.yaml         # (Fase 8)
├── data/
│   ├── raw/                       # .gitkeep — nunca versionar dados
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_preprocessamento_correlacao.ipynb
│   ├── 03_modelagem_experimentos.ipynb
│   ├── 04_avaliacao_interpretabilidade.ipynb
│   └── 05_cnn_pneumonia.ipynb     # Fase 8 opcional
├── src/medai/
│   ├── __init__.py
│   ├── config.py                  # carregamento/validação de YAML (pydantic)
│   ├── logging_conf.py
│   ├── data/
│   │   ├── loader.py              # download/leitura + schema check
│   │   ├── validate.py            # regras de sanidade clínica
│   │   └── split.py               # treino/validação/teste estratificado
│   ├── features/
│   │   ├── cleaning.py            # ausentes, zeros impossíveis, duplicatas, outliers
│   │   ├── build_pipeline.py      # ColumnTransformer sklearn
│   │   └── correlation.py         # matriz, VIF, seleção
│   ├── models/
│   │   ├── registry.py            # catálogo de estimadores + grids
│   │   ├── train.py               # CV, tuning, logging de experimentos
│   │   ├── evaluate.py            # métricas, curvas, limiar ótimo
│   │   ├── calibrate.py           # calibração de probabilidade
│   │   └── explain.py             # feature importance + SHAP
│   ├── vision/                    # Fase 8 opcional
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train_cnn.py
│   │   └── gradcam.py
│   ├── api/
│   │   ├── main.py                # FastAPI
│   │   ├── schemas.py             # pydantic in/out
│   │   ├── service.py             # carga do artefato + inferência
│   │   └── deps.py
│   └── ui/
│       └── app.py                 # Streamlit
├── models/                        # artefatos .joblib + metadata.json (versionado só o final)
├── reports/
│   ├── figures/                   # PNGs de EDA, curvas, SHAP
│   ├── metrics/                   # JSON/CSV de resultados
│   └── experiments.csv            # tabela comparativa de modelos
├── scripts/
│   ├── download_data.py
│   └── make_report_assets.py
└── tests/
    ├── test_data.py
    ├── test_pipeline.py
    ├── test_models.py
    └── test_api.py
```

**Princípio fundamental:** os notebooks **não contêm lógica**. Eles importam de `src/medai`, executam e narram. Isso garante que o que você explorou no notebook é exatamente o que roda na API — eliminando o descompasso treino/produção que é o erro mais comum nesse tipo de projeto.

---

## Fase 0 — Ambiente de desenvolvimento

**Objetivo:** base reprodutível antes de qualquer linha de ciência de dados.

1. `git init` no diretório `/home/rodrigo/Documents/WorkSpace/tech-challenge` (hoje não é repositório). Branch `main`, commit inicial com esqueleto.
2. Verificar ambiente real da máquina: versão do Python, Docker/Docker Compose, `nvidia-smi` (modelo e VRAM da GPU), núcleos de CPU e RAM. Esses números entram no README na seção de reprodutibilidade e definem os limites da Fase 8.
3. Criar `pyproject.toml` com Python 3.11 e dependências em grupos:
   - **core:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`, `pydantic`, `joblib`
   - **viz:** `matplotlib`, `seaborn`
   - **models:** `xgboost`, `lightgbm` (ou só um deles, ver Fase 4)
   - **explain:** `shap`
   - **serve:** `fastapi`, `uvicorn[standard]`, `streamlit`, `requests`
   - **dev:** `pytest`, `pytest-cov`, `ruff`, `black`, `jupyterlab`, `nbstripout`
   - **vision (extra):** `torch`, `torchvision`, `pillow`, `grad-cam` ou Grad-CAM manual
4. Ambiente virtual local (`.venv`) para desenvolvimento rápido + Docker como fonte de verdade da entrega.
5. **Fixar versões** (lockfile ou pins explícitos). Motivo concreto: SHAP e scikit-learn quebram compatibilidade entre versões menores, e um artefato `.joblib` treinado em sklearn X não carrega confiavelmente em sklearn Y. O README deve declarar as versões exatas.
6. `.gitignore`: `data/raw/`, `data/interim/`, `data/processed/`, `.venv/`, `__pycache__/`, `*.ipynb_checkpoints`, `models/*.joblib` exceto o final, `.env`.
7. Configurar `nbstripout` para limpar saídas pesadas dos notebooks no commit — mas **manter** as figuras exportadas em `reports/figures/` para o relatório.
8. `Makefile` com os alvos já listados na árvore acima.
9. Esqueleto de `src/medai` com todos os `__init__.py` e `logging_conf.py` funcionando (logging estruturado, nível via env var).

**Entregável:** repositório inicializado, `make setup` funcionando, `python -c "import medai"` OK.

---

## Fase 1 — Definição do problema e obtenção dos dados

**Objetivo:** amarrar o problema clínico ao problema estatístico antes de modelar.

1. **Documento de framing** (vai para o início do `RELATORIO_TECNICO.md`):
   - Problema: classificação binária supervisionada — tumor **maligno (1)** vs **benigno (0)** a partir de 30 medidas morfológicas extraídas de imagem digitalizada de punção aspirativa por agulha fina (FNA).
   - Usuário: equipe de triagem radiológica/patológica do hospital fictício.
   - Decisão apoiada: priorização da fila de revisão humana, não diagnóstico autônomo.
   - **Custo assimétrico de erro** — a decisão mais importante do projeto: um **falso negativo** (maligno classificado como benigno) pode significar atraso no tratamento de um câncer; um **falso positivo** gera exame confirmatório adicional e ansiedade. Portanto **recall da classe maligna é a métrica primária**, com F1 e ROC-AUC como suporte, e accuracy explicitamente rejeitada como métrica de decisão.
2. **Aquisição do dataset.** Duas rotas, implementadas em `scripts/download_data.py`:
   - **Rota A (padrão, sem credencial):** `sklearn.datasets.load_breast_cancer()` — é exatamente o Breast Cancer Wisconsin (Diagnostic) da UCI, mesma origem do dataset Kaggle indicado. Zero fricção, 100% reprodutível em Docker.
   - **Rota B (Kaggle):** `kaggle datasets download -d uciml/breast-cancer-wisconsin-data`, exige `~/.kaggle/kaggle.json`. O CSV do Kaggle tem duas peculiaridades que a Rota A não tem e que são **boas de exercitar**: a coluna `id` (identificador que deve ser descartado, nunca usado como feature) e a coluna fantasma `Unnamed: 32` inteiramente nula (artefato de vírgula final no CSV).
   - **Recomendação:** suportar as duas, com a Rota B como padrão quando o CSV existir em `data/raw/`, caindo para a Rota A caso contrário. Assim o pipeline de limpeza tem trabalho real a fazer e o projeto roda em qualquer máquina.
   - O README documenta o link do Kaggle e o comando de download, atendendo ao requisito "dataset ou link para download".
3. **Dicionário de dados** em `docs/dicionario_dados.md`: as 10 medidas base (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) × 3 estatísticas (mean, standard error, worst) = 30 features, com significado clínico de cada uma. Esse entendimento é o que permite interpretar o SHAP depois de forma não-ingênua.
4. **Schema check** em `src/medai/data/validate.py`: tipos esperados, intervalos plausíveis, cardinalidade do alvo. Falha ruidosamente se o CSV mudar.

**Entregável:** dados em `data/raw/`, loader testado, framing escrito, dicionário de dados.

---

## Fase 2 — Exploração de dados (EDA)

**Notebook:** `01_exploracao_dados.ipynb`. **Código:** funções em `src/medai/data/` e geração de figuras em `scripts/make_report_assets.py`.

1. **Caracterização da base:** shape, dtypes, uso de memória, amostra das primeiras linhas, `df.info()`, contagem de duplicatas, contagem de nulos por coluna.
2. **Estatísticas descritivas:** `describe()` completo (count, mean, std, min, quartis, max) + assimetria (skewness) e curtose por feature. Várias features de `area` e `concavity` são fortemente assimétricas à direita — isso justifica a discussão sobre escalonamento e sobre modelos sensíveis a escala.
3. **Distribuição do alvo:** 357 benignos (62,7%) vs 212 malignos (37,3%). Desbalanceamento **moderado** — importante concluir explicitamente no relatório que isso *não* exige SMOTE, mas *exige* estratificação nos splits e `class_weight='balanced'` como variante testada. Evitar o reflexo de aplicar oversampling sem necessidade é um ponto de maturidade técnica.
4. **Visualizações obrigatórias** (todas exportadas em PNG 150 dpi para `reports/figures/`):
   - Barras da distribuição de classes.
   - Grade de histogramas/KDE das 30 features separados por classe (`hue=diagnosis`) — revela visualmente quais features separam bem as classes.
   - Boxplots por classe para as features de maior separação, com atenção a outliers.
   - Violin plots das features `worst` (geralmente as mais discriminantes).
   - Pairplot de um subconjunto selecionado (6–8 features) — pairplot com 30 features é ilegível e deve ser evitado conscientemente.
   - Heatmap de correlação completo 30×30.
   - PCA 2D colorido por classe, só para visualização da separabilidade (não como feature engineering nesta fase).
5. **Análise de outliers:** IQR e z-score por feature. Discussão crítica no relatório: em dados clínicos, um outlier frequentemente **é** o caso patológico de interesse. Remover outliers aqui destruiria sinal. Decisão esperada: **manter**, documentando a escolha e a razão.
6. **Discussão escrita** ao final do notebook — o desafio pede explicitamente "discutiremos os resultados". Cada bloco de gráficos termina com um parágrafo de leitura, não só o gráfico solto.

**Entregável:** notebook 01 executado ponta a ponta, ~12–15 figuras em `reports/figures/`, seção de EDA do relatório redigida.

---

## Fase 3 — Pré-processamento e análise de correlação

**Notebook:** `02_preprocessamento_correlacao.ipynb`. **Código:** `src/medai/features/`.

1. **Limpeza** (`cleaning.py`):
   - Drop de `id` e `Unnamed: 32` quando vindos do CSV Kaggle.
   - Mapeamento do alvo: `M → 1`, `B → 0`. Documentar que a classe positiva é a maligna — crucial para a leitura correta de precision/recall.
   - Remoção de duplicatas exatas (se houver).
   - Tratamento de ausentes: o dataset principal não tem ausentes reais, mas o pipeline **deve** implementar imputação mediana mesmo assim, porque (a) o requisito pede tratamento de ausentes, (b) a API em produção receberá payloads incompletos, e (c) permite plugar o dataset de diabetes, onde `Glucose=0`, `BloodPressure=0`, `SkinThickness=0`, `Insulin=0` e `BMI=0` são ausentes disfarçados de zero. Implementar a regra de "zeros impossíveis → NaN" como transformação configurável por YAML.
2. **Split antes de qualquer ajuste** (`split.py`): separação **treino 60% / validação 20% / teste 20%**, estratificada pelo alvo, `random_state` fixo e registrado. A separação em três conjuntos é requisito explícito do desafio.
   - **Ponto crítico de integridade:** scaler, imputador e seleção de features são ajustados **apenas no treino**. Qualquer `fit` antes do split é vazamento de dados e invalida as métricas. Com 569 amostras isso é fácil de errar e difícil de perceber.
   - Com base pequena, usar **validação cruzada estratificada 5-fold dentro do treino** para tuning, e o conjunto de validação separado como checagem adicional/seleção de limiar. O teste fica intocado até a Fase 6.
3. **Pipeline sklearn** (`build_pipeline.py`): `ColumnTransformer` com
   - ramo numérico: `SimpleImputer(strategy='median')` → `StandardScaler()`
   - ramo categórico: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`
   O dataset principal é todo numérico, mas o ramo categórico existe para generalidade e cumpre o requisito "converter variáveis categóricas e numéricas em formatos adequados". Pipeline único e serializável — o mesmo objeto vai para a API.
4. **Análise de correlação** (`correlation.py`) — requisito explícito:
   - Matriz de Pearson e de Spearman (Spearman importa aqui pela assimetria das distribuições).
   - Heatmap com máscara triangular e clustermap hierárquico.
   - Correlação de cada feature com o alvo (point-biserial), ranqueada.
   - Identificação de pares com |r| > 0,90. Neste dataset há multicolinearidade severa e previsível: `radius`, `perimeter` e `area` são geometricamente redundantes (área ≈ π·raio²), e os trios mean/se/worst se correlacionam internamente.
   - Cálculo de **VIF** para quantificar a multicolinearidade.
   - **Decisão a documentar:** montar duas variantes de feature set — (A) todas as 30 features, (B) conjunto reduzido removendo redundantes com |r|>0,95 — e comparar empiricamente na Fase 5. A multicolinearidade prejudica a *interpretação* dos coeficientes da regressão logística (e dilui a atribuição SHAP entre features correlacionadas) mesmo quando não prejudica a *acurácia* das árvores. Essa distinção é exatamente o tipo de nuance que o relatório deve capturar.
5. **Persistência:** salvar os splits em `data/processed/` e o pipeline não-treinado como função construtora (não como objeto, para evitar artefatos obsoletos).

**Entregável:** notebook 02, pipeline testado por `tests/test_pipeline.py`, figuras de correlação, seção de pré-processamento do relatório.

---

## Fase 4 — Modelagem

**Notebook:** `03_modelagem_experimentos.ipynb`. **Código:** `src/medai/models/registry.py` e `train.py`.

1. **Baseline obrigatório:** `DummyClassifier(strategy='most_frequent')` — 62,7% de accuracy sem aprender nada. É o argumento mais eficaz do relatório contra o uso de accuracy como métrica, e dá o piso contra o qual todo modelo é comparado.
2. **Modelos a treinar** (o desafio pede ≥2; o plano usa 5 para permitir comparação real):
   - **Regressão Logística** — linear, interpretável por coeficientes, referência clínica clássica; com `class_weight='balanced'` como variante.
   - **Árvore de Decisão** — interpretável por regras, serve de contraste de viés/variância.
   - **KNN** — sensível a escala, demonstra o valor do `StandardScaler`.
   - **Random Forest** — ensemble robusto, fornece feature importance nativa.
   - **Gradient Boosting (XGBoost ou LightGBM)** — provável melhor desempenho bruto, com `scale_pos_weight` para o desbalanceamento.
   - *(opcional)* **SVM com kernel RBF** — costuma ir muito bem neste dataset.
3. **Tuning:** `GridSearchCV` para os modelos de espaço pequeno (LogReg, KNN, Árvore) e `RandomizedSearchCV` para RF e boosting. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='recall'` como primário com `refit` nele, registrando também `f1` e `roc_auc` em cada fold. Grids concretos definidos em `registry.py` por modelo.
4. **Rastreamento de experimentos:** cada execução grava uma linha em `reports/experiments.csv` — timestamp, modelo, hiperparâmetros, feature set (A ou B), média e desvio das métricas na CV, tempo de treino, hash do config. Suficiente e mais simples que MLflow para o escopo; se você quiser, MLflow local é um upgrade natural.
5. **Comparação de feature sets:** rodar a grade completa sobre o conjunto de 30 features e sobre o reduzido, e tabelar a diferença.
6. **Seleção do campeão:** escolher pelo recall médio na CV, com desempate por F1 e por simplicidade do modelo. Documentar o critério **antes** de ver os resultados do teste, para não cair em seleção enviesada.
7. **Calibração de probabilidade** (`calibrate.py`): o painel clínico vai exibir "87% de probabilidade de malignidade", e essa afirmação só é honesta se as probabilidades forem calibradas. Avaliar com curva de calibração e Brier score; aplicar `CalibratedClassifierCV` (sigmoid/isotonic) se houver desvio relevante. Modelos de boosting e SVM costumam precisar; regressão logística normalmente não.
8. **Ajuste de limiar de decisão:** o corte 0,5 é arbitrário. Usando o **conjunto de validação** (não o teste), escolher o limiar que atinge recall ≥ 0,98 na classe maligna com a melhor precisão possível, e definir uma **zona cinzenta** (ex.: probabilidade entre 0,30 e 0,70) rotulada como "revisão médica prioritária". Isso materializa no produto a exigência de que o diagnóstico final seja sempre do médico.

**Entregável:** notebook 03, `experiments.csv` com todas as execuções, modelo campeão selecionado e justificado.

---

## Fase 5 — Treinamento final e serialização

1. Retreinar o pipeline campeão em **treino + validação** com os hiperparâmetros vencedores (o teste permanece intocado).
2. Serializar com `joblib` em `models/medai_breast_cancer_v1.joblib`: **o pipeline completo**, incluindo imputador e scaler — nunca só o estimador.
3. Gravar `models/metadata.json` ao lado: versão do modelo, data de treino, dataset e hash do arquivo de dados, versões de Python/sklearn/numpy, lista ordenada de features esperadas, limiar de decisão escolhido, limites da zona cinzenta, métricas de validação, `random_state`. A API valida essa lista de features na entrada — é o que impede o erro silencioso de colunas fora de ordem.
4. Teste de round-trip: salvar, recarregar, prever sobre uma amostra fixa e conferir igualdade exata das probabilidades (`tests/test_models.py`).

**Entregável:** artefato versionado, metadata, teste de round-trip verde.

---

## Fase 6 — Avaliação e interpretabilidade

**Notebook:** `04_avaliacao_interpretabilidade.ipynb`. **Código:** `evaluate.py` e `explain.py`.

1. **Avaliação no conjunto de teste — uma única vez.** Reportar:
   - Accuracy, Precision, Recall, F1 (macro e por classe), ROC-AUC, PR-AUC, especificidade, Brier score.
   - Matriz de confusão absoluta e normalizada, com os quatro quadrantes **traduzidos para consequência clínica**: FN = "câncer não sinalizado", FP = "exame confirmatório desnecessário".
   - Curva ROC e curva Precision-Recall (esta última é a mais informativa sob desbalanceamento).
   - Curva de calibração.
   - Comparação lado a lado de todos os modelos no teste, em tabela.
2. **Discussão da escolha de métrica** — item explicitamente exigido. O argumento a desenvolver: accuracy mascara o custo assimétrico (o baseline burro já faz 62,7%); recall da classe maligna é primário porque o custo de um FN é incomparavelmente maior; F1 equilibra e evita o grau absurdo de recall=1,0 obtido classificando tudo como maligno; ROC-AUC mede separabilidade independente de limiar; PR-AUC é preferível à ROC-AUC sob desbalanceamento.
3. **Feature importance:**
   - Coeficientes padronizados da regressão logística (com sinal e intervalo de confiança).
   - Importância por impureza da Random Forest — e a ressalva de que ela é enviesada para features de alta cardinalidade.
   - **Permutation importance** no conjunto de teste — mais confiável que a nativa, e a que deve guiar as conclusões.
4. **SHAP** — item explicitamente exigido:
   - `TreeExplainer` para o modelo de árvore/boosting; `LinearExplainer` para a logística; `KernelExplainer` apenas se o campeão for SVM/KNN (e aí com amostragem, pelo custo).
   - **Global:** summary plot (beeswarm), bar plot de importância média |SHAP|, dependence plots das 4–5 features de maior impacto.
   - **Local:** waterfall/force plot para casos individuais escolhidos de propósito — um verdadeiro positivo confiante, um verdadeiro negativo confiante, um **falso negativo** e um **falso positivo**. A análise dos erros é o conteúdo mais valioso desta fase e normalmente é o que falta em projetos desse tipo.
   - **Armadilha a discutir:** com multicolinearidade severa (radius/perimeter/area), o SHAP distribui crédito entre features redundantes e subestima cada uma individualmente. Comparar as explicações nos feature sets A e B evidencia isso concretamente.
5. **Análise de erros:** inspecionar todos os casos mal classificados do teste, procurar padrão (estão perto do limiar? têm features extremas?), e documentar.
6. **Robustez:** repetir treino+avaliação com 5 sementes diferentes e reportar média ± desvio das métricas. Com teste de ~114 amostras, uma diferença de 2 pontos percentuais entre modelos pode ser puro ruído — e declarar isso explicitamente é um sinal de rigor, não de fraqueza.
7. **Discussão crítica de uso real** — item exigido. Tópicos a cobrir:
   - O modelo consome **features já extraídas por um patologista** a partir da imagem de FNA; ele não parte da imagem bruta. Isso limita bastante a automação prometida e precisa estar dito com todas as letras.
   - 569 amostras, de uma única instituição (Universidade de Wisconsin), coletadas nos anos 1990. Validade externa para a população e os equipamentos de hoje é desconhecida.
   - Ausência de validação prospectiva, de aprovação regulatória (ANVISA/FDA) e de auditoria de viés por subgrupo demográfico (o dataset nem traz essas variáveis).
   - Risco de automation bias: o médico tende a concordar com a máquina. Mitigações de produto: exibir probabilidade calibrada e não apenas rótulo, exibir a explicação SHAP junto, usar a zona cinzenta, registrar log de auditoria de toda predição, e rotular a saída como **sugestão de triagem**.
   - Como *poderia* ser usado na prática: priorização da fila de laudos, segunda opinião assistiva, sinalização de divergência entre laudo humano e modelo para revisão. Nunca como decisor.
   - Monitoramento necessário em operação: data drift, desempenho por subgrupo, taxa de concordância com o laudo final.

**Entregável:** notebook 04, figuras de avaliação e SHAP em `reports/figures/`, métricas em `reports/metrics/*.json`, seções de avaliação e discussão crítica do relatório.

---

## Fase 7 — Serviço: API FastAPI + UI Streamlit

### 7.1 API (`src/medai/api/`)

Endpoints:
- `GET /health` — liveness.
- `GET /model/info` — devolve o `metadata.json` (versão, features esperadas, limiar, métricas de validação).
- `POST /predict` — recebe um objeto com as features nomeadas; devolve `{ predicao, probabilidade_malignidade, confianca, faixa_decisao, limiar_usado, versao_modelo, aviso }`, onde `faixa_decisao ∈ {benigno, zona_cinzenta_revisao_medica, maligno}`.
- `POST /predict/batch` — lista de casos, ou upload de CSV, para triagem de fila.
- `POST /explain` — predição acompanhada dos top-N valores SHAP daquele caso, para a UI renderizar.

Detalhes de implementação:
- `schemas.py` com pydantic validando **nome, tipo e faixa plausível** de cada feature; erro 422 explícito e legível quando falta coluna ou valor é implausível.
- Carga do modelo uma única vez no startup (lifespan), não por requisição.
- Explainer SHAP também instanciado no startup — `TreeExplainer` é barato, `KernelExplainer` não seria viável por requisição.
- `aviso` fixo em toda resposta: *"Resultado de apoio à triagem. Não constitui diagnóstico. A decisão clínica é do médico responsável."*
- Logging estruturado de cada predição (sem dados identificáveis) para auditoria.
- Docs automáticas em `/docs` com exemplos preenchidos de caso benigno e maligno.

### 7.2 UI Streamlit (`src/medai/ui/app.py`)

Painel "MedAI Triage" com:
- **Aba 1 — Caso individual:** formulário com as 30 features (sliders com faixa derivada dos quantis do treino e valores default na mediana), botão "Analisar", e resultado em gauge de probabilidade + tarja colorida da faixa de decisão + gráfico de barras das contribuições SHAP do caso + aviso clínico persistente no rodapé.
- **Aba 2 — Lote:** upload de CSV, tabela de resultados ordenada por probabilidade decrescente (a "fila de triagem"), download do resultado.
- **Aba 3 — Sobre o modelo:** métricas de teste, matriz de confusão, curva ROC/PR, SHAP global, versões e limitações. Transparência como parte do produto.
- A UI fala com a API por HTTP (nunca carrega o modelo direto) — mantém a separação de responsabilidades e prova que a API funciona.

### 7.3 Testes

`tests/test_api.py` com `TestClient`: health, payload válido, payload faltando campo, payload com valor fora da faixa, batch, e consistência entre a predição da API e a do modelo chamado diretamente.

**Entregável:** `make api` e `make ui` funcionando localmente, testes verdes.

---

## Fase 8 — EXTRA opcional: Visão Computacional com CNN

Executar **somente** se, ao final da Fase 7, você julgar que o sistema precisa desse componente. Planejada aqui em detalhe para poder ser iniciada sem replanejamento.

**Dataset:** Chest X-Ray Pneumonia (Paul Mooney) — ~5.863 imagens JPEG, classes NORMAL/PNEUMONIA.

**Considerações de hardware — CPU apenas (revisado na Fase 0):** a máquina não tem GPU NVIDIA; ROCm não dá suporte oficial à iGPU AMD `gfx1103` presente. O treino roda em CPU (12 núcleos, 30 GB RAM), o que é viável com as seguintes restrições: imagens a 224×224, `batch_size` 32, `num_workers=8`, `torch.set_num_threads(12)`, e **transfer learning com backbone congelado** como caminho principal (só a cabeça treina — ordens de magnitude mais barato que fine-tuning completo). Estimativa: ~8–15 min por época sobre as ~5.200 imagens de treino. Se o tempo inviabilizar, subamostrar para 2.000 imagens balanceadas.

`torch` é instalado na variante CPU (`--index-url https://download.pytorch.org/whl/cpu`), que é muito menor que a imagem CUDA. O `Dockerfile.gpu` previsto originalmente **não será construído** — fica registrado como caminho futuro caso o projeto migre para máquina com CUDA. O código ainda chama `torch.cuda.is_available()` para portabilidade, mas aqui sempre resolve para CPU.

1. **Preparação:** download via Kaggle, inspeção da estrutura `train/val/test`. **Problema conhecido e importante:** o split `val` oficial tem apenas 16 imagens — inutilizável. Refazer o split a partir do `train` (80/20 estratificado), mantendo o `test` original para comparabilidade com a literatura. Documentar isso.
2. **EDA de imagens:** contagem por classe e split (o treino é desbalanceado, ~3:1 pneumonia), grade de amostras de cada classe, distribuição de dimensões e de intensidade média.
3. **Pipeline de dados:** `Dataset`/`DataLoader` PyTorch; augmentação **clinicamente plausível** apenas — rotação ≤10°, translação leve, zoom leve, ajuste de brilho/contraste. **Nunca flip horizontal**: inverte a lateralidade anatômica (coração à direita), ensinando o modelo uma anatomia falsa. Normalização com as estatísticas do ImageNet quando usar pesos pré-treinados.
4. **Modelos:**
   - Baseline: CNN pequena do zero (3 blocos conv) — para ter com o que comparar.
   - Principal: transfer learning com **MobileNetV3-Small** pré-treinado (preferido ao ResNet18 por ser ~5× mais barato em CPU), em duas etapas — (a) backbone congelado, treinar só a cabeça, ~5 épocas; (b) fine-tuning apenas do último bloco com LR baixo, ~3–5 épocas, executado só se a etapa (a) não bastar.
   - `WeightedRandomSampler` ou `pos_weight` na loss para o desbalanceamento.
   - Early stopping por recall de validação, `ReduceLROnPlateau`.
5. **Avaliação:** mesmas métricas da Fase 6 (recall primário, PR-AUC, matriz de confusão), avaliadas no `test` original.
6. **Interpretabilidade visual: Grad-CAM** — mapa de calor sobre o raio-X mostrando onde a rede olhou. Analisar casos corretos e incorretos. Discussão crítica obrigatória: é comum CNNs nesse dataset aprenderem artefatos (marcadores de posicionamento, texto queimado na imagem, diferenças de equipamento entre as coortes) em vez de patologia — e o Grad-CAM é como se detecta isso. Se os mapas apontarem para fora do campo pulmonar, isso **é** um resultado e deve ser relatado.
7. **Integração:** endpoint `POST /predict/image` na API (upload de imagem → classe, probabilidade, PNG do Grad-CAM em base64) e uma quarta aba na UI. Modelo de imagem carregado sob demanda para não pesar o startup.
8. **Notebook 05** narrando tudo.

**Entregável:** modelo CNN, Grad-CAMs, seção extra do relatório, endpoint e aba integrados.

---

## Fase 9 — Dockerização

1. **`docker/Dockerfile`** — multi-stage, base `python:3.11-slim`:
   - stage `builder`: compila dependências em wheels.
   - stage `runtime`: só o necessário, usuário não-root, `HEALTHCHECK` batendo em `/health`.
   - Camadas ordenadas para cache: dependências antes do código-fonte.
2. ~~**`docker/Dockerfile.gpu`**~~ — cancelado na Fase 0: a máquina não tem CUDA. A Fase 8, se executada, usa a mesma imagem CPU com o extra `vision` instalado.
3. **`docker-compose.yml`:**
   - `api` — porta 8000, volume de `models/` em modo leitura, healthcheck.
   - `ui` — porta 8501, `depends_on: api` com condição de saúde, `API_URL` por env var.
   - `jupyter` (perfil opcional) — porta 8888, para rodar os notebooks dentro do container.
   - `train` (perfil opcional) — serviço one-shot que executa o pipeline de treino completo e grava em `models/`.
   - Rede interna nomeada, volumes para `data/` e `models/`.
4. **`.dockerignore`** cobrindo `data/raw`, `.venv`, `.git`, notebooks com saída, `__pycache__`.
5. **Validação de reprodutibilidade:** apagar `models/`, rodar `docker compose --profile train up train`, depois `docker compose up`, e confirmar que a aplicação sobe e prediz corretamente a partir do zero. Esse é o teste real do requisito de execução local.

**Entregável:** `docker compose up` sobe API + UI; `curl localhost:8000/health` responde; `localhost:8501` abre o painel.

---

## Fase 10 — Documentação, relatório e fechamento

1. **`README.md`:**
   - Título, contexto do hospital fictício e problema.
   - **Aviso clínico destacado no topo:** projeto educacional, não é dispositivo médico, não usar em decisão clínica real.
   - Arquitetura (diagrama em Mermaid) e estrutura de diretórios comentada.
   - **Como executar:** caminho Docker (recomendado, um comando) e caminho local (venv), passo a passo, com os comandos em blocos copiáveis.
   - Como obter o dataset (link Kaggle + alternativa sklearn).
   - Como retreinar o modelo e como rodar os notebooks.
   - Exemplos de chamada à API (`curl` + resposta JSON).
   - Tabela-resumo de resultados.
   - Capturas de tela da UI e as figuras principais.
   - Stack, versões, requisitos de hardware, licença, limitações.
2. **`RELATORIO_TECNICO.md`** — o documento que o desafio exige, consolidando o que foi produzido nas fases:
   1. Sumário executivo
   2. Problema e contexto clínico
   3. Dataset: origem, dicionário, limitações
   4. EDA: achados e figuras
   5. **Estratégias de pré-processamento e por que cada uma** (imputação, escalonamento, encoding, tratamento de outliers, estratificação, prevenção de vazamento)
   6. Análise de correlação e multicolinearidade, com a decisão sobre feature sets
   7. **Modelos usados e por que cada um** — e por que o campeão venceu
   8. Protocolo experimental (splits, CV, tuning, sementes)
   9. Resultados: tabela comparativa + curvas + matriz de confusão
   10. **Escolha da métrica e sua justificativa clínica**
   11. Interpretabilidade: feature importance, SHAP global e local, análise de erros
   12. **Discussão crítica:** o modelo serve na prática? como? com quais salvaguardas?
   13. Limitações e riscos (viés, validade externa, regulatório, automation bias)
   14. Trabalhos futuros
   15. (Se a Fase 8 rodar) Anexo de visão computacional
   16. Referências
3. **Conferência final contra o enunciado:** checklist item a item do que você escreveu no desafio, marcando onde cada requisito foi atendido e em qual arquivo.
4. **Qualidade de código:** `ruff` + `black` limpos, docstrings em todas as funções públicas, type hints, cobertura de testes reportada.
5. **Higiene do repositório:** histórico de commits legível por fase, tag `v1.0`, verificação de que nenhum dado grande ou credencial foi versionado.

---

## Verificação end-to-end

Executar na ordem, do repositório limpo:

```bash
make setup && make test && make lint
```

```bash
python scripts/download_data.py --config configs/breast_cancer.yaml
```

```bash
python -m medai.models.train --config configs/breast_cancer.yaml
```

```bash
docker compose up --build -d && curl -s localhost:8000/health
```

Checklist de aceitação:
- [ ] `make test` verde, incluindo round-trip do modelo e testes da API.
- [ ] Os 4 notebooks executam de ponta a ponta sem erro, em ordem, a partir de kernel limpo.
- [ ] `reports/figures/` contém todas as figuras citadas no relatório.
- [ ] `reports/metrics/test_metrics.json` existe e bate com os números do relatório.
- [ ] `POST /predict` com um caso maligno conhecido do conjunto de teste retorna a classe correta; com um benigno, idem.
- [ ] `POST /predict` sem uma feature retorna 422 com mensagem clara.
- [ ] UI em `localhost:8501` renderiza as três abas, prediz e mostra SHAP.
- [ ] Apagar `models/`, retreinar via Docker e subir de novo funciona.
- [ ] README permite a alguém sem contexto rodar o projeto do zero.
- [ ] Todo requisito do enunciado tem um ponto correspondente no relatório.

---

## Ordem de execução sugerida

Fases 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 → 10, com a Fase 8 inserida entre 7 e 9 **somente se você optar por ela**. As Fases 9 e 10 podem começar em paralelo à 6, já que Dockerfile e esqueleto do README não dependem dos resultados finais.

## Decisões em aberto para revisitar durante a execução

- **Dataset de diabetes como bônus comparativo:** o pipeline já nasce parametrizado por YAML, então rodá-lo sobre o Pima custa pouco e rende uma seção extra forte no relatório (um dataset "fácil" vs um "difícil", com teto de desempenho e qualidade de dados muito diferentes). Decidir ao final da Fase 6.
- **XGBoost vs LightGBM:** escolher um só, para não inflar a imagem Docker sem ganho analítico.
- **MLflow:** só se o `experiments.csv` se mostrar insuficiente.
