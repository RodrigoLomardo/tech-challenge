"""Painel clínico do MedAI Triage.

A UI **não carrega o modelo**: conversa com a API por HTTP. Isso mantém a
separação de responsabilidades e, na prática, prova que a API funciona — se
o painel responde, o contrato REST está correto.

Três decisões de desenho que não são estéticas:

1. **A probabilidade é exibida, não só o rótulo.** "87%" convida à avaliação;
   "maligno" convida à concordância. É a principal mitigação de automation
   bias que a interface pode oferecer.
2. **A explicação aparece junto da predição**, agrupada por grandeza medida.
   A leitura individual do SHAP dilui crédito entre features redundantes e
   diria "textura foi decisiva" quando o fator é tamanho nuclear.
3. **A aba de transparência expõe as limitações do modelo**, inclusive a taxa
   de falsos negativos. Esconder isso tornaria a ferramenta mais confiável do
   que ela merece ser.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("MEDAI_API_URL", "http://localhost:8000")
TEMPO_LIMITE = 30

CORES_FAIXA = {
    "benigno": ("#D4EDDA", "#155724", "Prioridade padrão"),
    "zona_cinzenta_revisao_medica": ("#FFF3CD", "#856404", "Revisão médica prioritária"),
    "maligno": ("#F8D7DA", "#721C24", "Priorizar na fila"),
}

st.set_page_config(page_title="MedAI Triage", page_icon="🔬", layout="wide")


# -----------------------------------------------------------------------------
# Comunicação com a API
# -----------------------------------------------------------------------------


@st.cache_data(ttl=60)
def obter_info() -> dict | None:
    try:
        resposta = requests.get(f"{API}/model/info", timeout=TEMPO_LIMITE)
        resposta.raise_for_status()
        return resposta.json()
    except requests.RequestException:
        return None


def chamar(rota: str, payload) -> tuple[dict | None, str | None]:
    """Devolve (resposta, erro). Erros viram mensagem legível, não exceção."""
    try:
        resposta = requests.post(f"{API}{rota}", json=payload, timeout=TEMPO_LIMITE)
        if resposta.status_code == 422:
            detalhe = resposta.json().get("detail", "entrada inválida")
            return None, f"Entrada rejeitada: {detalhe}"
        resposta.raise_for_status()
        return resposta.json(), None
    except requests.RequestException as erro:
        return None, f"Falha ao comunicar com a API em {API}: {erro}"


# -----------------------------------------------------------------------------
# Componentes
# -----------------------------------------------------------------------------


def cartao_resultado(resultado: dict) -> None:
    """Resultado em destaque: probabilidade, faixa e recomendação."""
    fundo, texto, titulo = CORES_FAIXA.get(
        resultado["faixa_decisao"], ("#E2E3E5", "#383D41", "Indefinido")
    )
    probabilidade = resultado["probabilidade_malignidade"]

    st.markdown(
        f"""
        <div style="background:{fundo};color:{texto};padding:22px;border-radius:10px;
                    border-left:7px solid {texto};margin-bottom:14px">
          <div style="font-size:0.95rem;text-transform:uppercase;letter-spacing:0.5px;
                      opacity:0.85">{titulo}</div>
          <div style="font-size:2.6rem;font-weight:700;line-height:1.15">
            {probabilidade:.1%}</div>
          <div style="font-size:1rem">probabilidade de malignidade</div>
          <div style="margin-top:10px;font-size:0.95rem">{resultado["recomendacao"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(probabilidade, 1.0))

    colunas = st.columns(3)
    colunas[0].metric("Limiar operacional", f"{resultado['limiar_usado']:.3f}")
    colunas[1].metric("Faixa", resultado["faixa_decisao"].replace("_", " "))
    colunas[2].metric("Modelo", resultado["versao_modelo"])

    for aviso in resultado.get("avisos", []):
        st.warning(aviso, icon="⚠️")


def grafico_contribuicoes(explicacao: dict) -> None:
    """Contribuições SHAP, com a leitura agrupada em primeiro plano."""
    st.subheader("Por que o modelo chegou a esse resultado")

    agrupadas = pd.DataFrame(explicacao["contribuicoes_agrupadas"]).set_index("grupo")
    st.caption(
        "**Por grandeza medida.** Features que medem a mesma coisa dividem crédito "
        "entre si; somá-las é o que revela o peso real de cada grandeza."
    )
    st.bar_chart(agrupadas["contribuicao_absoluta"], horizontal=True, color="#DD8452")

    with st.expander("Contribuição individual por feature (log-odds)"):
        individuais = pd.DataFrame(explicacao["contribuicoes"])
        st.dataframe(
            individuais.style.format({"contribuicao": "{:+.4f}", "valor_medido": "{:.4g}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Valores em log-odds: +2,0 multiplica a chance de malignidade por e² ≈ 7,4. "
            "Sob multicolinearidade, a leitura individual subestima grupos de features "
            "redundantes — prefira a visão agrupada acima."
        )


# -----------------------------------------------------------------------------
# Abas
# -----------------------------------------------------------------------------


def aba_caso_individual(info: dict) -> None:
    st.header("Análise de exame individual")
    st.caption(
        "Informe as medidas morfológicas extraídas da punção aspirativa. "
        "Os valores padrão são a mediana do conjunto de treino."
    )

    estatisticas = info.get("estatisticas_features", {})
    features = info["features_esperadas"]

    with st.form("exame"):
        valores: dict[str, float] = {}
        for bloco, sufixo in [("Média", "_mean"), ("Erro padrão", "_se"), ("Pior valor", "_worst")]:
            do_bloco = [f for f in features if f.endswith(sufixo)]
            if not do_bloco:
                continue
            st.markdown(f"**{bloco}**")
            colunas = st.columns(5)
            for i, feature in enumerate(do_bloco):
                est = estatisticas.get(feature, {})
                valores[feature] = colunas[i % 5].number_input(
                    feature.replace(sufixo, ""),
                    min_value=0.0,
                    value=float(est.get("mediana", 0.0)),
                    step=float(max(est.get("mediana", 1.0) / 100, 1e-6)),
                    format="%.5g",
                    key=feature,
                    help=f"Treino: {est.get('minimo', 0):.4g} a {est.get('maximo', 0):.4g}",
                )
        enviado = st.form_submit_button("Analisar", type="primary", use_container_width=True)

    if not enviado:
        return

    with st.spinner("Analisando…"):
        resultado, erro = chamar("/explain", valores)
    if erro:
        st.error(erro)
        return

    cartao_resultado(resultado)
    grafico_contribuicoes(resultado)
    st.info(resultado["aviso_clinico"], icon="ℹ️")


def aba_lote(info: dict) -> None:
    st.header("Triagem de fila")
    st.caption("Envie um CSV com uma linha por exame. A fila sai ordenada por suspeição.")

    modelo_csv = pd.DataFrame(
        [
            {
                f: info.get("estatisticas_features", {}).get(f, {}).get("mediana", 0.0)
                for f in info["features_esperadas"]
            }
        ]
    )
    st.download_button(
        "Baixar CSV de exemplo",
        modelo_csv.to_csv(index=False).encode("utf-8"),
        "exemplo_exames.csv",
        "text/csv",
    )

    arquivo = st.file_uploader("CSV de exames", type="csv")
    if arquivo is None:
        return

    quadro = pd.read_csv(arquivo)
    faltantes = [c for c in info["features_esperadas"] if c not in quadro.columns]
    if faltantes:
        st.error(f"Colunas ausentes no CSV: {faltantes[:8]}")
        return

    st.success(f"{len(quadro)} exames carregados.")
    with st.spinner("Processando fila…"):
        resposta, erro = chamar(
            "/predict/batch", quadro[info["features_esperadas"]].to_dict(orient="records")
        )
    if erro:
        st.error(erro)
        return

    resumo = resposta["resumo_faixas"]
    colunas = st.columns(3)
    colunas[0].metric("Priorizar", resumo.get("maligno", 0))
    colunas[1].metric("Revisão prioritária", resumo.get("zona_cinzenta_revisao_medica", 0))
    colunas[2].metric("Prioridade padrão", resumo.get("benigno", 0))

    fila = pd.DataFrame(resposta["resultados"])[
        ["probabilidade_malignidade", "faixa_decisao", "recomendacao"]
    ]
    fila.index = range(1, len(fila) + 1)
    fila.index.name = "ordem"

    st.subheader("Fila ordenada por suspeição")
    st.dataframe(
        fila.style.format({"probabilidade_malignidade": "{:.1%}"}),
        use_container_width=True,
    )

    buffer = io.StringIO()
    fila.to_csv(buffer)
    st.download_button(
        "Baixar resultado", buffer.getvalue().encode("utf-8"), "fila_triagem.csv", "text/csv"
    )
    st.info(resposta["aviso_clinico"], icon="ℹ️")


def aba_transparencia(info: dict) -> None:
    st.header("Sobre o modelo")

    colunas = st.columns(4)
    colunas[0].metric("Algoritmo", info["algoritmo"])
    colunas[1].metric("Amostras de treino", info["n_amostras_treino"])
    colunas[2].metric("Features", len(info["features_esperadas"]))
    colunas[3].metric("Limiar", f"{info['limiar_decisao']:.3f}")

    st.subheader("Limitações conhecidas")
    st.caption(
        "Expostas deliberadamente. Uma ferramenta de apoio à decisão que esconde "
        "suas falhas é mais perigosa do que uma que não existe."
    )
    for limitacao in info["limitacoes"]:
        st.markdown(f"- {limitacao}")

    st.subheader("Desempenho na validação")
    metricas = info.get("metricas_validacao", {})
    relevantes = {
        k: v
        for k, v in metricas.items()
        if k in {"recall", "precision", "f1", "especificidade", "accuracy", "roc_auc", "brier"}
    }
    if relevantes:
        st.dataframe(
            pd.Series(relevantes, name="valor").to_frame().style.format("{:.4f}"),
            use_container_width=False,
        )

    with st.expander("Ponto de operação"):
        st.markdown(
            f"""
O limiar de **{info["limiar_decisao"]:.3f}** não é o padrão de 0,5. Ele foi escolhido
porque os dois erros possíveis têm custos e mecanismos de correção diferentes:

- um **falso negativo** despriorizia um câncer e sai do sistema — ninguém olha de novo;
- um **falso positivo** sobe um caso benigno na fila, um profissional revisa e corrige
  no mesmo dia.

Origem do limiar: `{info["limiar_origem"]}`

A **zona cinzenta** ({info["zona_cinzenta"][0]:.2f} a {info["zona_cinzenta"][1]:.2f}) encaminha
para revisão prioritária os casos em que o modelo não tem base para decidir, em vez de
forçar um rótulo.
"""
        )

    with st.expander("Versões e reprodutibilidade"):
        st.json(
            {
                "modelo": info["versao_modelo"],
                "treinado_em": info["treinado_em"],
                "dataset": info["dataset"],
                "versoes": info["versoes"],
            }
        )


# -----------------------------------------------------------------------------
# Aplicação
# -----------------------------------------------------------------------------


def main() -> None:
    st.title("🔬 MedAI Triage")
    st.caption("Apoio à triagem diagnóstica — Hospital Universitário Santa Clara (fictício)")

    info = obter_info()
    if info is None:
        st.error(
            f"Não foi possível alcançar a API em **{API}**.\n\n"
            "Suba o serviço com `make api` ou `docker compose up`.",
            icon="🔌",
        )
        st.stop()

    # Complementa a info com as estatísticas das features, que a API não
    # expõe em /model/info por serem volumosas.
    from medai.api import deps

    servico = deps.servico_opcional()
    if servico is not None:
        info["estatisticas_features"] = servico.metadados.estatisticas_features
    else:
        info.setdefault("estatisticas_features", {})

    aba1, aba2, aba3 = st.tabs(["Caso individual", "Triagem de fila", "Sobre o modelo"])
    with aba1:
        aba_caso_individual(info)
    with aba2:
        aba_lote(info)
    with aba3:
        aba_transparencia(info)

    st.divider()
    st.caption(
        "⚠️ Projeto educacional. Não é dispositivo médico, não possui validação clínica "
        "nem aprovação regulatória. **A decisão clínica é do médico responsável.**"
    )


if __name__ == "__main__":
    main()
