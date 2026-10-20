# Dicionário de Dados — Breast Cancer Wisconsin (Diagnostic)

Origem: Universidade de Wisconsin, anos 1990 · [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) · [espelho no Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

**569 amostras × 30 features numéricas + 1 alvo binário.** Sem valores ausentes.

## Como os dados foram produzidos

Cada linha corresponde a uma **punção aspirativa por agulha fina (FNA)** de massa mamária. A amostra é depositada em lâmina, fotografada, e um software (`Xcyt`) delimita o contorno dos núcleos celulares visíveis. As features são medidas geométricas e de textura desses contornos.

**Implicação central, que precisa estar no relatório:** as features **não são extraídas da imagem bruta por este sistema**. Elas pressupõem que um profissional já realizou a punção, preparou a lâmina e que o software já segmentou os núcleos. O modelo entra depois disso — ele automatiza o *julgamento* sobre medidas prontas, não a *extração* das medidas. Isso limita bastante a automação que o sistema pode prometer.

## Alvo

| Coluna | Tipo | Valores | Observação |
|---|---|---|---|
| `diagnosis` | categórico | `M` (maligno), `B` (benigno) | **`M` é a classe positiva (1)** |

Distribuição: **212 malignos (37,3%)** e **357 benignos (62,7%)**. Desbalanceamento moderado — exige estratificação nos splits, não exige reamostragem.

> **Armadilha do scikit-learn.** Em `load_breast_cancer()`, `target == 0` é *maligno* e `target == 1` é *benigno* — o inverso da convenção deste projeto. Usar `target` direto inverteria precision e recall sem nenhum erro aparente: o modelo pareceria ótimo enquanto otimizaria exatamente a métrica errada. O tratamento está em [`loader.py`](../src/medai/data/loader.py) e é coberto por dois testes independentes (contagem de classes e tamanho médio do tumor por classe).

## Colunas descartadas

| Coluna | Por quê |
|---|---|
| `id` | Identificador do paciente. Jamais usar como feature: não tem significado clínico e, se correlacionar com o alvo por ordem de coleta, o modelo aprende o vazamento em vez da patologia. |
| `Unnamed: 32` | Coluna 100% nula. Artefato de vírgula final em cada linha do CSV do Kaggle. Não existe na rota `sklearn`. |

## As 10 medidas de base

Cada medida aparece **três vezes**, com sufixos diferentes — total de 30 features.

| Sufixo | Significado | Interpretação |
|---|---|---|
| `_mean` | média entre os núcleos da amostra | tendência central da lâmina |
| `_se` | erro padrão da média | **heterogeneidade** entre núcleos |
| `_worst` | média dos 3 maiores valores | o núcleo mais extremo — o achado mais alarmante |

| Medida | O que mede | Relevância clínica |
|---|---|---|
| `radius` | distância média do centro ao perímetro | Núcleos aumentados indicam proliferação descontrolada. |
| `texture` | desvio padrão dos tons de cinza | Cromatina irregular sugere alteração do material genético. |
| `perimeter` | comprimento do contorno | Redundante com `radius` por construção geométrica. |
| `area` | área delimitada pelo contorno | Redundante: área ≈ π·raio². |
| `smoothness` | variação local no comprimento do raio | Contorno irregular é marcador de malignidade. |
| `compactness` | perímetro² / área − 1 | Quantifica o quanto o núcleo se afasta do círculo. |
| `concavity` | severidade das reentrâncias do contorno | Núcleos malignos têm contorno recortado, não liso. |
| `concave points` | **número** de reentrâncias | A medida mais discriminante do dataset. |
| `symmetry` | simetria entre as metades do núcleo | Perda de simetria acompanha a desorganização celular. |
| `fractal_dimension` | aproximação da dimensão fractal da borda | Complexidade do contorno em múltiplas escalas. |

## Poder discriminativo observado

`d` de Cohen entre as classes (diferença padronizada das médias). Acima de 0,8 é efeito grande; perto de 0, a feature não separa nada.

| Feature | `d` | Feature | `d` |
|---|---:|---|---:|
| `concave points_worst` | **2,60** | `texture_mean` | 0,95 |
| `perimeter_worst` | **2,37** | `smoothness_worst` | 0,95 |
| `concave points_mean` | **2,33** | `concave points_se` | 0,93 |
| `radius_worst` | **2,32** | `symmetry_worst` | 0,88 |
| `perimeter_mean` | **2,12** | `smoothness_mean` | 0,80 |
| `radius_mean` | **2,05** | `symmetry_mean` | 0,71 |
| `area_worst` | **1,97** | `fractal_dimension_worst` | 0,67 |
| `concavity_mean` | **1,87** | `compactness_se` | 0,62 |
| `area_mean` | **1,86** | `concavity_se` | 0,57 |
| `concavity_worst` | **1,75** | `fractal_dimension_se` | 0,17 |
| `compactness_mean` | 1,45 | `symmetry_se` | −0,01 |
| `compactness_worst` | 1,40 | `texture_se` | −0,02 |
| `radius_se` | 1,27 | `fractal_dimension_mean` | −0,03 |
| `perimeter_se` | 1,23 | `smoothness_se` | −0,14 |
| `area_se` | 1,18 | `texture_worst` | 1,06 |

Três leituras que orientam as fases seguintes:

1. **As features `_worst` dominam o topo.** Faz sentido clínico: o diagnóstico de malignidade se apoia no achado mais atípico da lâmina, não na média. Uma amostra majoritariamente normal com três núcleos muito alterados é suspeita.
2. **`concave points` lidera nas duas variantes.** A contagem de reentrâncias do contorno nuclear é o sinal mais forte disponível.
3. **Quatro features não separam nada** (`smoothness_se`, `fractal_dimension_mean`, `texture_se`, `symmetry_se`, todas com |d| < 0,2). Isso não significa que devam ser removidas de imediato — podem ter valor em interação —, mas é a expectativa contra a qual compararemos o feature importance e o SHAP na Fase 6. Se um modelo atribuir alta importância a `smoothness_se`, há o que investigar.

## Assimetria das distribuições

Features de `area` e dos erros padrão são fortemente assimétricas à direita: `area_se` (skew 5,45), `concavity_se` (5,11), `fractal_dimension_se` (3,92), `perimeter_se` (3,44), `radius_se` (3,09).

Consequência prática: KNN e regressão logística exigem escalonamento, e o `StandardScaler` por si só não corrige assimetria. Fica registrado como possibilidade a avaliar na Fase 3 se transformação logarítmica melhora os modelos lineares — mas sem aplicá-la por reflexo, já que os modelos de árvore são indiferentes a ela.

## Multicolinearidade esperada

`radius`, `perimeter` e `area` medem a mesma grandeza física em unidades diferentes (perímetro ≈ 2πr, área ≈ πr²). A correlação entre eles é próxima de 1 **por construção, não por acaso**. O mesmo trio se repete nos três sufixos, gerando nove features que carregam essencialmente três informações.

Isso é quantificado na Fase 3 (correlação + VIF) e tem consequência direta na interpretabilidade: o SHAP distribui crédito entre features redundantes, fazendo cada uma parecer menos importante do que é. Ver ADR-009.

## Schema canônico

As duas rotas de aquisição (`sklearn` e CSV do Kaggle) produzem a mesma nomenclatura. Note a inconsistência preservada da fonte original: `fractal_dimension_mean` usa underscore, mas `concave points_mean` mantém o espaço.

```
diagnosis
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
compactness_mean, concavity_mean, concave points_mean, symmetry_mean,
fractal_dimension_mean
… idem com sufixo _se
… idem com sufixo _worst
```
