"""Catálogo de modelos e seus espaços de busca.

Cada entrada declara o construtor do estimador, a grade de hiperparâmetros e
a estratégia de busca. Centralizar isso aqui — em vez de espalhar pelo script
de treino — é o que permite rodar o mesmo protocolo experimental sobre todos
os modelos e sobre os dois feature sets sem duplicar código.

Os nomes das chaves da grade usam o prefixo ``modelo__`` porque o estimador é
o passo ``modelo`` do pipeline construído em :mod:`medai.features.build_pipeline`.

**Sobre paralelismo:** os estimadores são criados com ``n_jobs=1`` e o
paralelismo fica a cargo da busca (``GridSearchCV(n_jobs=-1)``). Paralelizar
nos dois níveis causa superalocação — mais threads que núcleos, disputando
entre si — e deixa o treino mais lento, não mais rápido.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class EspecificacaoModelo:
    """Como treinar e ajustar um modelo."""

    nome: str
    descricao: str
    # Por que este modelo está no catálogo — vai direto para o relatório.
    justificativa: str
    construtor: Callable[[int], BaseEstimator]
    grade: dict[str, list[Any]] = field(default_factory=dict)
    busca: Literal["grid", "random", "nenhuma"] = "grid"
    n_iteracoes: int = 30
    # Modelos cuja saída de probabilidade costuma ser mal calibrada.
    requer_calibracao: bool = False

    @property
    def n_combinacoes(self) -> int:
        """Tamanho do espaço de busca, para estimar o custo."""
        if not self.grade:
            return 1
        total = 1
        for valores in self.grade.values():
            total *= len(valores)
        return total


CATALOGO: dict[str, EspecificacaoModelo] = {
    "dummy": EspecificacaoModelo(
        nome="dummy",
        descricao="Classificador trivial (sempre a classe majoritária)",
        justificativa=(
            "Piso obrigatório de comparação. Atinge 62,7% de accuracy sem aprender "
            "nada e recall zero na classe maligna — é o argumento mais direto contra "
            "usar accuracy como métrica de decisão neste problema."
        ),
        construtor=lambda semente: DummyClassifier(strategy="most_frequent"),  # noqa: ARG005
        busca="nenhuma",
    ),
    "logistic_regression": EspecificacaoModelo(
        nome="logistic_regression",
        descricao="Regressão logística regularizada",
        justificativa=(
            "Referência clássica em modelagem de risco clínico. Interpretável por "
            "coeficientes com sinal e magnitude, produz probabilidades naturalmente "
            "calibradas e, como a Fase 3 mostrou que as relações são predominantemente "
            "lineares, tem chance real de capturar a estrutura dos dados."
        ),
        construtor=lambda semente: LogisticRegression(
            max_iter=5000, random_state=semente, n_jobs=1
        ),
        grade={
            "modelo__C": [0.01, 0.1, 0.5, 1.0, 10.0, 100.0],
            "modelo__penalty": ["l2"],
            "modelo__solver": ["lbfgs"],
            # Variante que pesa a classe minoritária — hipótese aberta da Fase 3.
            "modelo__class_weight": [None, "balanced"],
        },
        busca="grid",
    ),
    "decision_tree": EspecificacaoModelo(
        nome="decision_tree",
        descricao="Árvore de decisão",
        justificativa=(
            "Interpretável por regras explícitas, legíveis por um clínico sem "
            "formação em estatística. Serve de contraste de viés e variância: "
            "árvore única tende a sobreajustar, o que evidencia o ganho dos ensembles."
        ),
        construtor=lambda semente: DecisionTreeClassifier(random_state=semente),
        grade={
            "modelo__max_depth": [3, 4, 5, 7, None],
            "modelo__min_samples_leaf": [1, 3, 5, 10],
            "modelo__criterion": ["gini", "entropy"],
            "modelo__class_weight": [None, "balanced"],
        },
        busca="grid",
    ),
    "knn": EspecificacaoModelo(
        nome="knn",
        descricao="K vizinhos mais próximos",
        justificativa=(
            "Não paramétrico, sem suposição sobre a forma da fronteira. Depende "
            "criticamente de escalonamento — demonstra na prática por que o "
            "StandardScaler é obrigatório quando as features diferem em seis "
            "ordens de magnitude."
        ),
        construtor=lambda semente: KNeighborsClassifier(n_jobs=1),  # noqa: ARG005
        grade={
            "modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "modelo__weights": ["uniform", "distance"],
            # p=1 Manhattan, p=2 Euclidiana.
            "modelo__p": [1, 2],
        },
        busca="grid",
        requer_calibracao=True,
    ),
    "random_forest": EspecificacaoModelo(
        nome="random_forest",
        descricao="Floresta aleatória",
        justificativa=(
            "Ensemble robusto a ruído e a features redundantes — relevante aqui, "
            "onde o VIF máximo passa de 3.900. Fornece importância nativa e, por "
            "agregar árvores descorrelacionadas, sofre menos com a variância que "
            "afeta a árvore única."
        ),
        construtor=lambda semente: RandomForestClassifier(random_state=semente, n_jobs=1),
        grade={
            "modelo__n_estimators": [200, 400, 600],
            "modelo__max_depth": [None, 5, 10, 20],
            "modelo__min_samples_leaf": [1, 2, 4],
            "modelo__max_features": ["sqrt", "log2", 0.5],
            "modelo__class_weight": [None, "balanced", "balanced_subsample"],
        },
        busca="random",
        n_iteracoes=40,
    ),
    "xgboost": EspecificacaoModelo(
        nome="xgboost",
        descricao="Gradient boosting (XGBoost)",
        justificativa=(
            "Costuma oferecer o melhor desempenho bruto em dados tabulares. "
            "Permite ajustar o desbalanceamento via scale_pos_weight e tem suporte "
            "nativo a TreeSHAP, o que torna a interpretabilidade barata."
        ),
        construtor=lambda semente: _construir_xgboost(semente),
        grade={
            "modelo__n_estimators": [100, 200, 400],
            "modelo__max_depth": [2, 3, 4, 6],
            "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "modelo__subsample": [0.7, 0.85, 1.0],
            "modelo__colsample_bytree": [0.7, 0.85, 1.0],
            "modelo__min_child_weight": [1, 3, 5],
            # 1.0 = sem reponderação; ~1,68 = razão benigno/maligno no treino.
            "modelo__scale_pos_weight": [1.0, 1.68],
        },
        busca="random",
        n_iteracoes=50,
        requer_calibracao=True,
    ),
    "svm_rbf": EspecificacaoModelo(
        nome="svm_rbf",
        descricao="SVM com kernel RBF",
        justificativa=(
            "Historicamente forte neste dataset. Captura fronteiras não lineares "
            "e, com poucas amostras e muitas features, a margem máxima é um "
            "princípio de regularização adequado."
        ),
        construtor=lambda semente: SVC(probability=True, random_state=semente),
        grade={
            "modelo__C": [0.1, 1.0, 10.0, 100.0],
            "modelo__gamma": ["scale", "auto", 0.01, 0.1],
            "modelo__class_weight": [None, "balanced"],
        },
        busca="grid",
        requer_calibracao=True,
    ),
}


def _construir_xgboost(semente: int) -> BaseEstimator:
    """XGBoost com os parâmetros fixos que não entram na busca."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=semente,
        n_jobs=1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )


def obter(nome: str) -> EspecificacaoModelo:
    """Recupera a especificação de um modelo pelo nome."""
    if nome not in CATALOGO:
        raise KeyError(f"Modelo '{nome}' não está no catálogo. Disponíveis: {sorted(CATALOGO)}")
    return CATALOGO[nome]


def listar(nomes: list[str] | None = None) -> list[EspecificacaoModelo]:
    """Especificações dos modelos pedidos, ou do catálogo inteiro."""
    if nomes is None:
        return list(CATALOGO.values())
    return [obter(n) for n in nomes]
