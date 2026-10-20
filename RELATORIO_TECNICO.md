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
4. Exploração de dados — *pendente (Fase 2)*
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

## 16. Referências

- Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE International Symposium on Electronic Imaging.
- Dua, D. & Graff, C. (2019). *UCI Machine Learning Repository* — Breast Cancer Wisconsin (Diagnostic).
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE.
- Goddard, K., Roudsari, A., & Wyatt, J. C. (2012). *Automation bias: a systematic review of frequency, effect mediators, and mitigators.* JAMIA.
