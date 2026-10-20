# Registro de Decisões Técnicas (ADR) — MedAI Triage

Decisões tomadas no planejamento, com a razão por trás de cada uma. Serve de insumo direto para o `RELATORIO_TECNICO.md` — o desafio exige justificar *por que* cada escolha, não apenas listá-las.

Novas decisões tomadas durante a execução devem ser acrescentadas aqui, na mesma estrutura.

---

## ADR-001 — Dataset principal: Breast Cancer Wisconsin (Diagnostic)

**Status:** aceito (planejamento)

**Contexto.** O enunciado oferece duas bases tabulares: Breast Cancer Wisconsin e Pima Indians Diabetes.

**Decisão.** Breast Cancer Wisconsin como base oficial; Pima Diabetes fica como execução bônus do mesmo pipeline parametrizado.

**Razão.** As 30 features morfológicas com multicolinearidade severa e previsível (radius/perimeter/area são geometricamente dependentes) tornam a análise de correlação e a discussão de SHAP substantivas em vez de protocolares — a correlação deixa de ser um heatmap decorativo e passa a ter consequência de modelagem. O desbalanceamento moderado (37,3% maligno) é realista sem exigir técnicas de reamostragem. E o custo assimétrico de erro em oncologia justifica de forma natural a escolha de recall como métrica primária.

**Custo.** O dataset é pequeno (569 amostras) e "fácil" — modelos atingem ~97% de accuracy rapidamente, o que torna a comparação entre eles estatisticamente ruidosa. Mitigação: análise de robustez com múltiplas sementes (Fase 6) e relato de intervalos, não de números pontuais.

---

## ADR-002 — Dupla rota de aquisição de dados (sklearn + Kaggle)

**Status:** aceito

**Contexto.** O download via Kaggle exige credencial `kaggle.json`, o que quebra a reprodutibilidade em Docker e em máquina de terceiros.

**Decisão.** `scripts/download_data.py` usa o CSV do Kaggle quando presente em `data/raw/`; caso contrário, cai para `sklearn.datasets.load_breast_cancer()`. Mesma origem UCI, mesmos dados.

**Razão.** Reprodutibilidade garantida sem credencial, mantendo o link do Kaggle documentado conforme o enunciado pede. Bônus: o CSV do Kaggle carrega a coluna `id` e a coluna fantasma `Unnamed: 32` (toda nula), que dão trabalho real e legítimo à etapa de limpeza.

---

## ADR-003 — Recall da classe maligna como métrica primária

**Status:** aceito

**Contexto.** O enunciado pede accuracy, recall e F1, e exige discussão sobre a escolha da métrica.

**Decisão.** Recall (sensibilidade) da classe maligna é a métrica de seleção de modelo e de ajuste de limiar. F1 e PR-AUC são secundárias. Accuracy é reportada, mas explicitamente rejeitada como critério de decisão.

**Razão.** Os erros têm custos incomparáveis. Um falso negativo é um câncer não sinalizado, com atraso de tratamento. Um falso positivo é um exame confirmatório a mais. Accuracy trata os dois como equivalentes, e o `DummyClassifier` já alcança 62,7% sem aprender nada — a prova mais direta de que a métrica é inadequada aqui. F1 entra como contrapeso porque recall isolado é trivialmente maximizado classificando tudo como maligno.

---

## ADR-004 — Split 60/20/20 com CV aninhada no treino

**Status:** aceito

**Contexto.** O enunciado exige separação clara entre treino, validação e teste. A base tem apenas 569 amostras.

**Decisão.** Split estratificado 60/20/20. O tuning usa `StratifiedKFold` 5-fold **dentro do treino**; o conjunto de validação serve para escolher o limiar de decisão e para conferência independente; o teste é tocado uma única vez, na Fase 6.

**Razão.** Com teste de ~114 amostras, cada erro vale ~0,9 ponto percentual. Usar o teste para qualquer escolha (modelo, hiperparâmetro ou limiar) contamina a estimativa final. A CV interna extrai mais sinal do treino pequeno do que um holdout único de validação conseguiria.

---

## ADR-005 — Pipeline sklearn serializado inteiro, nunca só o estimador

**Status:** aceito

**Decisão.** O artefato `.joblib` contém `ColumnTransformer` (imputação + escalonamento + encoding) **e** o estimador, em um único objeto.

**Razão.** É a causa raiz mais comum de descompasso treino/produção: salvar só o modelo e reimplementar o pré-processamento na API, com diferença sutil de ordem de colunas ou de parâmetros do scaler. Um pipeline único torna esse erro estruturalmente impossível. O `metadata.json` ao lado registra a lista ordenada de features esperadas, e a API valida contra ela.

---

## ADR-006 — Notebooks sem lógica de negócio

**Status:** aceito

**Decisão.** Toda a lógica vive em `src/medai/`. Os notebooks importam, executam e narram.

**Razão.** Código em célula de notebook não é testável, não é versionável de forma legível e não é importável pela API. Essa separação é o que permite que `tests/` cubra o mesmo código que roda em produção — e que o que foi analisado seja exatamente o que é servido.

---

## ADR-007 — Manter outliers

**Status:** aceito

**Decisão.** Nenhuma remoção automática de outliers por IQR ou z-score. Eles são identificados, quantificados e discutidos.

**Razão.** Em dados clínicos o outlier é frequentemente o caso patológico de interesse. Um tumor com `area_worst` extrema é justamente o sinal que o modelo precisa aprender. Remoção por regra estatística cega descartaria sinal diagnóstico e inflaria artificialmente as métricas.

---

## ADR-008 — Calibração de probabilidade e zona cinzenta

**Status:** aceito

**Decisão.** Avaliar calibração (curva + Brier score) e aplicar `CalibratedClassifierCV` se houver desvio relevante. O limiar é escolhido na validação visando recall ≥ 0,98, e probabilidades em faixa intermediária são rotuladas como "revisão médica prioritária" em vez de receberem um rótulo binário.

**Razão.** A UI exibe "87% de probabilidade de malignidade"; essa afirmação só é honesta se as probabilidades forem calibradas — boosting e SVM tipicamente não são. A zona cinzenta é a materialização em produto da exigência de que o diagnóstico final seja sempre do médico: em vez de forçar uma decisão onde o modelo é incerto, o sistema encaminha.

---

## ADR-009 — Dois conjuntos de features (A: 30 completo, B: reduzido)

**Status:** aceito

**Decisão.** Treinar e comparar sobre as 30 features e sobre um subconjunto com os redundantes de |r| > 0,95 removidos.

**Razão.** A multicolinearidade severa deste dataset afeta pouco a *acurácia* de modelos baseados em árvore, mas distorce a *interpretação*: coeficientes instáveis na regressão logística, e atribuição SHAP diluída entre features correlacionadas (cada uma parece menos importante do que é). Comparar as duas variantes transforma essa armadilha teórica em evidência empírica no relatório.

---

## ADR-010 — Entrega como API + UI, não apenas notebooks

**Status:** aceito

**Decisão.** FastAPI servindo predição e explicação, Streamlit consumindo a API por HTTP, ambos em Docker Compose.

**Razão.** O enunciado pede "um sistema", não um estudo. A separação API/UI força a serialização correta do modelo, o contrato de entrada explícito e o tratamento de erro — questões de engenharia que um notebook esconde. A UI falar com a API por HTTP (em vez de carregar o modelo direto) é o que prova que a API realmente funciona.

---

## ADR-011 — CNN como fase opcional, com transfer learning leve

**Status:** aceito

**Contexto.** Verificado na Fase 0: **não há GPU NVIDIA**. O único dispositivo gráfico é uma iGPU AMD Phoenix1 (`gfx1103`), sem CUDA e sem suporte oficial de ROCm. Disponíveis: 12 núcleos de CPU e 30 GB de RAM. (Revisa a premissa original de planejamento, que supunha GPU NVIDIA modesta.)

**Decisão.** Fase 8 planejada em detalhe, executada por escolha. **MobileNetV3-Small** pré-treinado em CPU, 224×224, batch 32, backbone congelado como caminho principal. `torch` na variante CPU. O `Dockerfile.gpu` foi cancelado.

**Razão.** Sem CUDA, treinar do zero é inviável: seriam muitas horas por um resultado pior que transfer learning. Com o backbone congelado, o custo por época cai drasticamente porque não há backward pass através da rede inteira — o que torna a fase executável em CPU. MobileNetV3-Small é preferido ao ResNet18 por ter ~5× menos FLOPs, diferença que importa pouco em GPU e muito em CPU. A ordem (cabeça primeiro, fine-tuning depois) também evita destruir os pesos pré-treinados na primeira época.

**Custo.** Treino na casa de ~8–15 min por época. Se inviabilizar, subamostrar o conjunto de treino para ~2.000 imagens balanceadas e documentar o impacto nas métricas.

---

## ADR-012 — Sem flip horizontal na augmentação de raio-X

**Status:** aceito

**Decisão.** A augmentação da Fase 8 usa rotação ≤10°, translação e zoom leves e ajuste de brilho/contraste. Flip horizontal é proibido.

**Razão.** Espelhar um raio-X de tórax inverte a lateralidade anatômica — coração à direita, situs inversus artificial. Ensina ao modelo uma anatomia que não existe e descarta a assimetria esquerda/direita, que é informação diagnóstica legítima. É o default copiado de pipelines de visão geral que mais prejudica imagens médicas.

---

## ADR-013 — Rastreamento de experimentos em CSV, não MLflow

**Status:** aceito

**Decisão.** Cada execução de treino grava uma linha em `reports/experiments.csv`.

**Razão.** Para dezenas (não milhares) de execuções, um CSV é inspecionável, versionável em Git e não adiciona serviço nem dependência à imagem Docker. MLflow permanece como upgrade caso o volume justifique.

---

## Decisões ainda em aberto

| Item | Quando decidir | Critério |
|------|----------------|----------|
| XGBoost **ou** LightGBM (escolher um) | Fase 4 | Desempenho na CV; empate → o de imagem Docker menor |
| Rodar o Pima Diabetes como bônus | fim da Fase 6 | Se houver fôlego; rende seção comparativa forte |
| SVM-RBF no catálogo de modelos | Fase 4 | Incluir se o custo de tuning for baixo |
| Executar a Fase 8 (CNN) | fim da Fase 7 | Sua avaliação sobre a completude do sistema |
