# Ambiente de Desenvolvimento — verificado em 2026-10-20 (Fase 0)

Registro factual da máquina onde o projeto foi desenvolvido. Alimenta a seção de
reprodutibilidade do README e define os limites da Fase 8 (visão computacional).

## Sistema

| Item | Valor |
|---|---|
| SO | Ubuntu 26.04.1 LTS (kernel 7.0.0-38-generic) |
| CPU | 12 núcleos lógicos |
| RAM | 30 GiB |
| GPU | **AMD Phoenix1 (iGPU integrada, `gfx1103`)** |
| CUDA | **indisponível** — sem GPU NVIDIA, sem `nvidia-smi`, sem `/proc/driver/nvidia` |
| Git | 2.53.0 |
| Docker | 29.9.1 |
| Docker Compose | v5.3.1 |

### Consequência da ausência de CUDA

A premissa original de planejamento supunha GPU NVIDIA. A verificação mostrou que
não há. ROCm não dá suporte oficial a `gfx1103`, então a rota AMD também não é
confiável.

**Decisão:** a Fase 8 roda em **CPU**, com MobileNetV3-Small de backbone
congelado. `torch` na variante CPU. `Dockerfile.gpu` cancelado.
Ver ADR-011 em [DECISOES_TECNICAS.md](DECISOES_TECNICAS.md).

## Python

| Item | Valor |
|---|---|
| Interpretador do projeto | **3.11.16** (`/usr/bin/python3.11`) |
| Default do sistema | 3.14.4 — **não usado** |
| Ambiente virtual | `.venv/` na raiz do projeto |

O Python 3.14 do sistema é novo demais para o stack de ML (rodas pré-compiladas
de scikit-learn, xgboost e shap ainda não cobrem essa versão de forma estável).
O `Makefile` fixa `PYTHON := python3.11`.

## Versões instaladas (principais)

Lock completo em [`requirements.lock.txt`](../requirements.lock.txt).

| Pacote | Versão |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.14.1 |
| scikit-learn | **1.5.2** |
| joblib | 1.6.0 |
| xgboost | 2.1.4 |
| statsmodels | 0.14.6 |
| shap | 0.46.0 |
| matplotlib | 3.9.4 |
| seaborn | 0.13.2 |
| fastapi | 0.115.14 |
| streamlit | 1.41.1 |
| pydantic | 2.14.1 |
| pytest | 8.4.2 |
| ruff | 0.8.6 |
| black | 24.10.0 |

**scikit-learn 1.5.2 e joblib 1.6.0 são as versões críticas:** o artefato
`.joblib` produzido pelo treino não carrega de forma confiável sob outra versão
menor de scikit-learn. Ambas são registradas em `models/metadata.json` a cada
treino, e a API compara na inicialização.

## Reprodução em outra máquina

```bash
make setup
```

```bash
source activate.sh
```

O `activate.sh` ativa o venv, carrega o `.env` quando existir (sem sobrescrever
variáveis já definidas no shell), fixa `PYTHONHASHSEED` na semente do projeto e
limita `OMP_NUM_THREADS` para que BLAS não dispute os núcleos com o `n_jobs` do
scikit-learn. Precisa ser carregado com `source` — executado, rodaria em
subshell e não afetaria o seu terminal.

Exige apenas Python 3.11 e, para a rota Docker, Docker com Compose v2+.
O dataset é obtido sem credencial pela rota `sklearn` (ADR-002), então nenhum
segredo é necessário para rodar o projeto do zero.
