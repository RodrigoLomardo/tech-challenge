# CLAUDE.md — MedAI Triage

Instruções para desenvolvimento assistido neste repositório. Leia antes de qualquer alteração.

## O que é este projeto

**MedAI Triage** — sistema de apoio à triagem diagnóstica do Hospital Universitário Santa Clara (fictício). Desafio técnico pessoal: construir a base de um sistema de IA que classifica exames e apoia a decisão clínica, sem jamais substituí-la.

- **Problema:** classificação binária — tumor maligno (1) vs benigno (0), a partir de 30 medidas morfológicas de punção aspirativa (FNA).
- **Dataset principal:** Breast Cancer Wisconsin (Diagnostic), 569 amostras, 37,3% maligno.
- **Entrega:** notebooks narrados + pipeline em `src/medai/` + API FastAPI + UI Streamlit + Docker Compose + relatório técnico.
- **Extra opcional (Fase 8):** CNN para pneumonia em raio-X, com Grad-CAM.
- **Idioma:** código e identificadores em inglês; documentação, comentários explicativos, notebooks e relatório em **português**.

### Documentos de referência
| Arquivo | Conteúdo |
|---|---|
| [docs/PLANEJAMENTO.md](docs/PLANEJAMENTO.md) | Plano mestre, 11 fases, arquitetura alvo |
| [docs/CHECKLIST_REQUISITOS.md](docs/CHECKLIST_REQUISITOS.md) | Rastreabilidade requisito → artefato |
| [docs/DECISOES_TECNICAS.md](docs/DECISOES_TECNICAS.md) | ADRs com o *porquê* de cada escolha |
| [docs/AMBIENTE.md](docs/AMBIENTE.md) | Hardware e versões verificadas (sem CUDA — Fase 8 é CPU) |

Antes de propor uma mudança de abordagem, confira se já existe um ADR sobre o assunto. Se houver e você discordar, argumente contra o ADR explicitamente — não o contorne em silêncio. Decisão nova tomada durante a execução vira ADR novo.

---

## Regras invioláveis do domínio

1. **O diagnóstico final é sempre do médico.** Toda saída do sistema é sugestão de triagem. O aviso clínico aparece no README, no relatório, em toda resposta da API e na UI.
2. **Recall da classe maligna é a métrica primária.** Accuracy é reportada mas nunca usada para selecionar modelo ou limiar — o `DummyClassifier` já faz 62,7% aqui. Falso negativo = câncer não sinalizado; falso positivo = exame confirmatório a mais. Os custos não são comparáveis.
3. **Classe positiva = maligno = 1.** Qualquer leitura de precision/recall depende disso. Não inverter.
4. **Nunca `fit` antes do split.** Imputador, scaler e seleção de features são ajustados só no treino. Com 569 amostras, vazamento é fácil de cometer e difícil de notar.
5. **O conjunto de teste é tocado uma única vez**, na Fase 6. Escolha de modelo, hiperparâmetro e limiar acontecem em CV no treino e no conjunto de validação.
6. **Outliers não são removidos.** Em dados clínicos o outlier costuma ser o caso patológico de interesse (ADR-007).
7. **Na Fase 8, nada de flip horizontal em raio-X.** Inverte a lateralidade anatômica (ADR-012).
8. **Não há CUDA nesta máquina.** Nada de código que pressuponha GPU NVIDIA; a Fase 8 é CPU-only (ADR-011).

---

## Convenções de código

- **Python 3.11**, versões de dependências fixadas (`.joblib` de sklearn X não carrega confiavelmente em sklearn Y).
- **Toda lógica vive em `src/medai/`.** Notebooks importam, executam e narram — não contêm lógica. Se você escreveria uma função numa célula, ela pertence a um módulo.
- **O artefato serializado é o pipeline inteiro** (`ColumnTransformer` + estimador), nunca só o estimador (ADR-005). `models/metadata.json` registra a lista ordenada de features; a API valida contra ela.
- Type hints em todas as funções públicas; docstrings explicando *por que*, não *o que*.
- `ruff` + `black` limpos antes de qualquer commit.
- Caminhos vêm do config YAML (`configs/*.yaml`), nunca hardcoded. O pipeline é parametrizado para aceitar outro dataset (ex.: Pima Diabetes) sem mudança de código.
- `random_state` fixo e registrado em todo lugar que sorteia.
- Figuras exportadas em PNG 150 dpi para `reports/figures/`, com nome descritivo.

## Fluxo de trabalho

- **Execute as fases na ordem** do PLANEJAMENTO.md. Cada fase tem entregável próprio; não começar a seguinte com a anterior pela metade.
- **Um commit por unidade coerente de trabalho**, mensagem em português descrevendo o efeito. Commit só quando eu pedir.
- **Commits saem no meu nome, sem co-autoria.** Nada de `Co-Authored-By`, `Generated with` ou qualquer menção a assistente nas mensagens de commit ou em PRs. Este é um desafio pessoal — a autoria é minha.
- **Teste junto, não depois.** Toda função em `src/medai/` nasce com teste em `tests/`.
- **Nunca versionar:** `data/raw/`, `data/interim/`, `data/processed/`, `.venv/`, `kaggle.json`, `.env`, artefatos intermediários de modelo.
- Arquivos temporários vão para o scratchpad da sessão, não para o repositório.

## Comandos

```bash
source activate.sh  # ativa o venv e carrega as variáveis do projeto
```

```bash
make setup    # ambiente + dependências
make test     # pytest com cobertura
make lint     # ruff + black --check
make train    # pipeline de treino completo
make api      # FastAPI em :8000
make ui       # Streamlit em :8501
```

```bash
docker compose up --build
```

---

## Como quero ser auxiliado

- **Honestidade técnica acima de encorajamento.** Se um resultado é ruído estatístico, diga. Com teste de ~114 amostras, diferença de 2 pontos percentuais entre modelos frequentemente não significa nada — e declarar isso é rigor, não fraqueza.
- **Explique o raciocínio das escolhas de ciência de dados**, não só o código. O objetivo deste desafio é eu aprender; código correto sem justificativa falha no propósito.
- **Aponte as armadilhas antes que eu caia nelas** — vazamento de dados, métrica enganosa, SHAP diluído por multicolinearidade, CNN aprendendo artefato em vez de patologia.
- **Não aplique técnica por reflexo.** SMOTE aqui não é necessário (desbalanceamento é moderado); aplicar sem precisar é sinal de imaturidade técnica, não de competência. Vale para qualquer técnica: só entra com justificativa.
- **Prefira reusar o que já existe** em `src/medai/` a criar módulo novo. Verifique antes.
- **Relate falhas com a saída real.** Teste que quebrou, número que não bateu, etapa que foi pulada — tudo explícito.
- Ao concluir uma fase, atualize o status correspondente em `docs/CHECKLIST_REQUISITOS.md`.

## Aviso

Projeto educacional. Não é dispositivo médico, não possui validação clínica nem aprovação regulatória, e não deve ser usado em decisão assistencial real.

## Regras de desenvolvimento
- Documentacao com comentarios em codigos em PT-BR
- Seguir nome de arquivos, classes e etc em PT-BR
- Evitar ultrapassar 400 linhas em arquivos que podem ser modularizados, services nao sao inclusos.
