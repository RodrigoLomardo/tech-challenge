# Relatório Técnico — MedAI Triage

**Sistema Inteligente de Suporte à Triagem Diagnóstica**
Hospital Universitário Santa Clara (instituição fictícia)

Autor: Rodrigo Lomardo · Documento em construção — seções preenchidas conforme as fases avançam.

> **Aviso.** Projeto educacional. Não é dispositivo médico, não possui validação clínica nem aprovação regulatória, e não deve ser usado em decisão assistencial real.

---

## Índice

1. [Sumário executivo](#1-sumário-executivo) — *pendente (Fase 10)*
2. [Problema e contexto clínico](#2-problema-e-contexto-clínico) ✅
3. [Dataset: origem, dicionário e limitações](#3-dataset-origem-dicionário-e-limitações) ✅
4. [Exploração de dados](#4-exploração-de-dados) ✅
5. Estratégias de pré-processamento — *pendente (Fase 3)*
6. Correlação e multicolinearidade — *pendente (Fase 3)*
7. Modelos utilizados e justificativa — *pendente (Fase 4)*
8. Protocolo experimental — *pendente (Fase 4)*
9. Resultados — *pendente (Fase 6)*
10. Escolha da métrica e justificativa clínica — *pendente (Fase 6)*
11. Interpretabilidade — *pendente (Fase 6)*
12. Discussão crítica: o modelo serve na prática? — *pendente (Fase 6)*
13. Limitações e riscos — *pendente (Fase 6)*
14. Trabalhos futuros — *pendente (Fase 10)*
15. Anexo: visão computacional — *opcional (Fase 8)*
16. Referências

---

## 1. Sumário executivo

*A ser escrito na Fase 10, quando houver resultados a sumarizar.*

---

## 2. Problema e contexto clínico

### 2.1 O cenário

O Hospital Universitário Santa Clara enfrenta um gargalo de triagem. O volume de exames cresce mais rápido que a equipe capaz de laudá-los: radiografias, tomografias, ressonâncias, prontuários digitalizados e lâminas de citopatologia se acumulam em fila. A consequência não é apenas demora — é que a demora se distribui de forma **indiferente à gravidade**. Um caso de alta suspeição aguarda o mesmo tempo que um caso trivial, porque a fila é cronológica e não há como saber o que é urgente antes de examinar.

Há ainda o fator fadiga. Revisão em volume alto, sob pressão de tempo, degrada a acurácia humana — e o erro que mais custa nesse contexto é o de deixar passar.

### 2.2 O que este sistema se propõe a fazer

Produzir, para cada exame na fila, uma **estimativa de probabilidade de malignidade**, acompanhada da explicação de quais medidas sustentaram aquela estimativa. Com isso, a fila deixa de ser cronológica e passa a ser ordenada por suspeição.

O que o sistema **não** faz, e precisa estar dito sem ambiguidade: não emite diagnóstico, não substitui o laudo e não dispensa a avaliação humana de nenhum caso. Todo exame continua sendo revisto por um profissional; o que muda é a ordem e a informação disponível no momento da revisão.

### 2.3 Formulação técnica

| Aspecto | Definição |
|---|---|
| Tipo de problema | Classificação binária supervisionada |
| Entrada | 30 medidas morfológicas de núcleos celulares (punção aspirativa por agulha fina) |
| Saída | Probabilidade calibrada de malignidade + faixa de decisão + atribuição SHAP por feature |
| Classe positiva | **Maligno (`M` → 1)** |
| Classe negativa | Benigno (`B` → 0) |
| Usuário | Equipe de citopatologia e triagem |
| Decisão apoiada | Priorização da fila de revisão |
| Decisão **não** apoiada | Diagnóstico, conduta terapêutica, alta |

### 2.4 A assimetria de custo — a decisão que organiza o projeto

Os dois erros possíveis não são equivalentes, e essa é a premissa da qual decorrem quase todas as escolhas técnicas seguintes.

| Erro | O que acontece | Custo |
|---|---|---|
| **Falso negativo** — maligno classificado como benigno | O caso é despriorizado na fila. O diagnóstico de câncer atrasa. | **Alto e potencialmente irreversível.** Em oncologia, tempo é prognóstico. |
| **Falso positivo** — benigno classificado como maligno | O caso sobe na fila e é revisto antes. Possível exame confirmatório. | **Baixo.** Tempo profissional e ansiedade do paciente. |

Três consequências diretas:

1. **A métrica primária é o recall da classe maligna**, não a accuracy. Accuracy trata os dois erros como equivalentes, e o classificador trivial que responde sempre "benigno" já acerta 62,7% dos casos.
2. **O limiar de decisão não é 0,5.** Ele é escolhido na validação para atingir recall ≥ 0,98, aceitando a perda de precisão que isso impõe.
3. **Casos incertos não recebem rótulo binário.** Probabilidades em faixa intermediária são sinalizadas como "revisão médica prioritária" — o sistema admite não saber em vez de forçar uma resposta.

### 2.5 Onde o médico permanece no circuito

A exigência de que o diagnóstico final seja sempre humano não é uma ressalva no rodapé do README; ela está implementada:

- a saída é **probabilidade calibrada**, não rótulo — a diferença entre "87%" e "maligno" é a diferença entre informar e decidir;
- a explicação SHAP acompanha cada predição, para que o profissional possa discordar com base em algo;
- a **zona cinzenta** encaminha os casos incertos em vez de classificá-los;
- toda predição é registrada em log de auditoria;
- o aviso clínico é centralizado em [`medai.AVISO_CLINICO`](src/medai/__init__.py) e aparece em toda resposta da API e na UI.

O risco que essas medidas endereçam é o **automation bias**: a tendência documentada de profissionais concordarem com a sugestão da máquina mesmo quando ela está errada. Um sistema que exibe apenas um rótulo o agrava; um que exibe incerteza e justificativa o atenua.

---

## 3. Dataset: origem, dicionário e limitações

### 3.1 Escolha

**Breast Cancer Wisconsin (Diagnostic)** — 569 amostras, 30 features numéricas, alvo binário. Dicionário completo em [`docs/dicionario_dados.md`](docs/dicionario_dados.md).

A escolha, entre as bases oferecidas, é justificada no [ADR-001](docs/DECISOES_TECNICAS.md): a multicolinearidade severa e previsível entre `radius`, `perimeter` e `area` torna a análise de correlação uma etapa com consequência real de modelagem, em vez de um heatmap decorativo, e o custo assimétrico de erro em oncologia dá fundamento clínico à escolha de recall como métrica primária.

### 3.2 Aquisição

Duas rotas, implementadas em [`scripts/download_data.py`](scripts/download_data.py) e descritas no [ADR-002](docs/DECISOES_TECNICAS.md):

- **CSV do Kaggle**, quando há credencial. Traz a coluna `id` e a coluna fantasma `Unnamed: 32` (100% nula), que exercitam a limpeza.
- **`sklearn.datasets.load_breast_cancer()`** como fallback. Mesma origem UCI, sem credencial — garante que o projeto roda em qualquer máquina e no Docker.

As duas produzem o mesmo schema canônico, de modo que o restante do pipeline não depende de qual foi usada.

### 3.3 Uma armadilha tratada na origem

No scikit-learn, `target == 0` é **maligno** e `target == 1` é benigno — o inverso da convenção deste projeto. Usar o alvo diretamente inverteria precision e recall silenciosamente: nenhuma exceção, nenhum aviso, e um modelo que pareceria excelente enquanto otimizava exatamente a métrica errada.

O tratamento está no carregador e é protegido por duas travas independentes nos testes: a contagem de classes (212/357) e uma verificação clínica — tumores malignos têm `area_mean` maior que benignos. Se o rótulo inverter, as duas falham.

### 3.4 Limitações que acompanham estes dados

Registradas aqui porque condicionam o que o modelo pode legitimamente prometer, e retomadas na discussão crítica da Fase 6:

| Limitação | Consequência |
|---|---|
| **Features já extraídas por profissional.** O modelo recebe medidas prontas de uma lâmina já preparada e segmentada. | A automação é do *julgamento*, não da *extração*. O ganho de tempo é menor do que "IA analisa exames" sugere. |
| **569 amostras, instituição única** (Universidade de Wisconsin). | Validade externa desconhecida para outra população, outro equipamento, outro protocolo de preparo. |
| **Coletado nos anos 1990.** | Técnica de imagem e critérios citológicos mudaram em três décadas. |
| **Sem variáveis demográficas.** | Impossível auditar viés por idade, etnia ou condição socioeconômica — e um modelo não auditável quanto a viés não deveria operar em saúde. |
| **Conjunto de teste de ~114 amostras.** | Cada erro vale ~0,9 ponto percentual. Diferenças pequenas entre modelos são ruído; por isso a Fase 6 reporta intervalos sobre múltiplas sementes, não números pontuais. |

---

## 4. Exploração de dados

Notebook: [`notebooks/01_exploracao_dados.ipynb`](notebooks/01_exploracao_dados.ipynb) · Figuras: `reports/figures/01–10` · Tabelas: `reports/metrics/eda_*.csv`

### 4.1 Estrutura da base

569 linhas × 31 colunas: 30 features `float64` e um alvo categórico. **Nenhum valor ausente, nenhuma duplicata.** 165,7 KB em memória.

A ausência de ausentes simplifica a Fase 3, mas o tamanho é a restrição que condiciona toda a interpretação dos resultados: com ~114 amostras no teste, **cada erro vale 0,9 ponto percentual**. Qualquer diferença de desempenho abaixo de ~2 pontos entre dois modelos está dentro do ruído amostral, e tratá-la como real seria superinterpretação. É por isso que a Fase 6 reporta média ± desvio sobre cinco sementes em vez de um número por modelo.

### 4.2 Distribuição do alvo

| Classe | Contagem | Proporção |
|---|---:|---:|
| Benigno (`B`) | 357 | 62,7% |
| **Maligno (`M`)** | **212** | **37,3%** |

![Distribuição de classes](reports/figures/01_distribuicao_classes.png)

O desbalanceamento é **moderado**. Duas conclusões:

**Sobre a métrica.** O classificador que responde sempre "benigno" obtém **62,7% de accuracy** sem olhar para nenhuma feature — e **recall zero na classe maligna**, deixando passar os 212 cânceres. Accuracy alta, utilidade clínica negativa. Nenhuma métrica que premie esse comportamento pode orientar a seleção de modelo neste problema.

**Sobre reamostragem.** 37,3% não justifica SMOTE nem undersampling. Gerar amostras sintéticas por interpolação de casos oncológicos reais traz risco concreto — o modelo pode aprender artefatos do interpolador em vez de padrões clínicos — sem benefício correspondente nesta faixa de desbalanceamento. O que se exige é **estratificação nos splits**, e `class_weight='balanced'` entra como variante a testar, não como correção obrigatória.

### 4.3 Estatísticas descritivas

As features ocupam escalas que diferem em mais de seis ordens de magnitude: `area_worst` chega a 4.254, enquanto `fractal_dimension_se` fica na casa de 0,003. Para o KNN, que calcula distância euclidiana, isso significa que sem padronização **`area_worst` sozinha determinaria toda a vizinhança**. A padronização é obrigatória, não uma boa prática opcional.

Features mais assimétricas:

| Feature | Skew |
|---|---:|
| `area_se` | 5,45 |
| `concavity_se` | 5,11 |
| `fractal_dimension_se` | 3,92 |
| `perimeter_se` | 3,44 |
| `radius_se` | 3,09 |

Cabe uma distinção que costuma se perder: **`StandardScaler` não corrige assimetria.** Ele centra e redimensiona; uma distribuição com skew 5,45 continua com skew 5,45 depois dele. Transformação logarítmica corrigiria, mas beneficiaria apenas os modelos lineares — árvores e boosting são indiferentes a transformações monotônicas. Fica registrada como **hipótese a testar na Fase 4**, não como etapa a aplicar preventivamente.

### 4.4 Distribuições por classe

![Distribuições por feature](reports/figures/02_distribuicoes_todas_features.png)

Três padrões:

1. **Separação clara** em `concave points`, `perimeter`, `radius`, `area` e `concavity`.
2. **Sobreposição quase total** em `smoothness_se`, `symmetry_se`, `texture_se` e `fractal_dimension_mean`.
3. **Caudas longas à direita** nas features `_se`.

Os boxplots e violinos (figuras 03 e 04) mostram que, mesmo nas features mais discriminantes, **as caudas das duas classes se sobrepõem**. Existem tumores benignos com medidas na faixa dos malignos. Esses casos serão os erros do modelo — e são exatamente os que mais merecem revisão humana, o que fundamenta a zona cinzenta.

Nas features `_worst`, a distribuição dos malignos é mais larga e mais deslocada. Isso é clinicamente coerente: `_worst` é a média dos três núcleos mais extremos da lâmina, e é neles que o citopatologista se apoia. Uma amostra majoritariamente normal com poucos núcleos muito alterados é suspeita — e a média simples esconderia esse achado.

### 4.5 Separabilidade individual (d de Cohen)

![Separabilidade](reports/figures/09_separabilidade_cohen_d.png)

| Maior separabilidade | d | Sem separabilidade | d |
|---|---:|---|---:|
| `concave points_worst` | **2,60** | `fractal_dimension_se` | 0,17 |
| `perimeter_worst` | **2,37** | `symmetry_se` | −0,01 |
| `concave points_mean` | **2,33** | `texture_se` | −0,02 |
| `radius_worst` | **2,32** | `fractal_dimension_mean` | −0,03 |
| `perimeter_mean` | **2,12** | `smoothness_se` | −0,14 |

A convenção trata d = 0,8 como efeito grande; `concave points_worst` chega a 2,60. As seis primeiras posições são de `concave points`, `perimeter` ou `radius`, nas variantes `_worst` e `_mean`.

**As cinco features sem separabilidade constituem uma expectativa falsificável para a Fase 6.** Se o feature importance ou o SHAP atribuírem peso alto a `smoothness_se`, há duas leituras possíveis: ou a feature tem valor em interação (que o d de Cohen, sendo univariado, não capta), ou o modelo está aprendendo ruído. Registrar a expectativa antes de ver o resultado é o que permite distinguir os dois casos — sem isso, qualquer desfecho pareceria plausível em retrospecto.

### 4.6 Valores atípicos — e por que serão mantidos

![Outliers por classe](reports/figures/10_outliers_por_classe.png)

Pelo critério IQR, `area_se` tem 65 valores fora da cerca (11,4% das amostras), `radius_se` e `perimeter_se` 38 cada. Há bastante coisa que um pipeline automatizado "limparia".

A distribuição desses atípicos entre as classes é o que decide a questão:

| Classe | Amostras com ao menos uma feature fora da cerca IQR |
|---|---:|
| Benigno | **16,0%** |
| Maligno | **53,8%** |

Mais da metade dos casos malignos é atípica, contra um sexto dos benignos. **Os valores extremos não são ruído de medição: são a manifestação da própria patologia.** Um tumor maligno é, por definição, tecido com crescimento desregulado — núcleos maiores, contornos irregulares, maior heterogeneidade. "Fora do padrão" descreve a doença.

Remover outliers por regra estatística descartaria **preferencialmente os casos que o modelo mais precisa identificar**, e o efeito seria perverso: as métricas de teste melhorariam, porque o problema ficaria artificialmente mais fácil, enquanto a utilidade clínica cairia. É uma armadilha em que um pipeline automatizado cai silenciosamente.

**Decisão (ADR-007): manter todos os valores atípicos.** A única checagem de corrupção pertinente é por valores fisicamente impossíveis — medidas geométricas negativas —, e a validação não encontrou nenhum.

### 4.7 Correlação entre features

![Correlação](reports/figures/06_correlacao_pearson.png)

Os blocos saturados não são achado empírico, são **consequência geométrica**: `perimeter ≈ 2πr` e `area ≈ πr²`, de modo que `radius`, `perimeter` e `area` medem a mesma grandeza em unidades diferentes, com |r| > 0,99. O trio se repete nos três sufixos — **nove features carregando essencialmente três informações**.

O dano não é de acurácia. Árvores lidam bem com redundância: escolhem uma feature, ignoram as demais. O dano é de **interpretação** — coeficientes instáveis na regressão logística, e crédito SHAP dividido entre features correlacionadas, fazendo cada uma parecer menos importante do que o grupo é. Daí a comparação entre feature sets prevista no ADR-009. A análise formal, com Spearman e VIF, é a Fase 3.

### 4.8 Separabilidade em conjunto (PCA)

![PCA](reports/figures/08_pca_2d.png)

Duas componentes capturam **63,2% da variância** (PC1 sozinha, 44,3%), com separação nítida ao longo de PC1.

1. **O problema é aprendível.** Separação visível em 2D sem supervisão nenhuma indica que modelos lineares já devem ir bem. Isso calibra a expectativa para a Fase 6: accuracy alta aqui é característica do dataset, não mérito do algoritmo — motivo a mais para não confundir desempenho neste benchmark com prontidão clínica.

2. **Há uma faixa de sobreposição** em torno de PC1 ≈ 0 a 2. **É ali que estarão os erros**, e é a base empírica da zona cinzenta: nessa região, a informação disponível não basta para decidir, e forçar um rótulo binário seria simular uma certeza que os dados não sustentam.

3. **A dispersão dos malignos é muito maior.** Benignos formam aglomerado compacto; malignos se espalham. Coerente com a análise de outliers: "normal" é um estado estreito, "patológico" tem muitas formas.

> **Nota metodológica.** Este PCA foi ajustado sobre o dataset inteiro, o que é aceitável porque **nenhum modelo está sendo treinado** — trata-se de visualizar estrutura. Se as componentes fossem usadas como features, o ajuste teria de se restringir ao treino, sob pena de vazamento.

### 4.9 Síntese da EDA

| Achado | Consequência para as próximas fases |
|---|---|
| 569 × 31, sem nulos nem duplicatas | Teste de ~114: reportar intervalos, não números pontuais |
| 37,3% malignos | Estratificar; **não** reamostrar |
| Classificador trivial acerta 62,7% | Accuracy rejeitada como métrica de decisão |
| Escalas variando 6 ordens de magnitude | `StandardScaler` obrigatório |
| `area_se` com skew 5,45 | Transformação log: hipótese a testar |
| `concave points_worst` com d = 2,60 | Referência para validar o SHAP |
| 5 features com \|d\| < 0,2 | Se o modelo valorizá-las, investigar |
| 53,8% dos malignos atípicos vs 16,0% dos benignos | **Manter outliers** |
| `radius`/`perimeter`/`area` com r > 0,99 | Comparar feature set completo vs reduzido |
| PCA separa, com faixa de sobreposição | Zona cinzenta tem base empírica |

---

## 16. Referências

- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE International Symposium on Electronic Imaging.
- Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository* — Breast Cancer Wisconsin (Diagnostic).
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE.
- Goddard, K., Roudsari, A., & Wyatt, J. C. (2012). *Automation bias: a systematic review of frequency, effect mediators, and mitigators.* JAMIA.
