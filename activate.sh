#!/usr/bin/env bash
# MedAI Triage — ativação do ambiente de desenvolvimento.
#
# Uso:
#   source activate.sh      (ou: . activate.sh)
#
# Precisa ser carregado com `source`: um script executado roda em subshell,
# e a ativação morreria junto com ela sem afetar o seu terminal.

# --- Guarda: recusa execução direta -----------------------------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Erro: este script precisa ser carregado com 'source', não executado."
    echo "  Use:  source activate.sh"
    exit 1
fi

_medai_raiz="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_medai_venv="${_medai_raiz}/.venv"

# --- Ambiente virtual --------------------------------------------------------
if [ ! -x "${_medai_venv}/bin/python" ]; then
    echo "Ambiente virtual não encontrado em ${_medai_venv}"
    echo "Crie-o com:  make setup"
    unset _medai_raiz _medai_venv
    return 1
fi

# shellcheck disable=SC1091
source "${_medai_venv}/bin/activate"

# --- Variáveis de ambiente ---------------------------------------------------
# Carrega .env quando existir, ignorando comentários e linhas vazias.
# As variáveis já definidas no shell têm precedência: exportamos apenas o
# que ainda não está definido, para não sobrescrever ajustes temporários.
if [ -f "${_medai_raiz}/.env" ]; then
    while IFS='=' read -r _chave _valor; do
        case "${_chave}" in
            ''|\#*) continue ;;
        esac
        _chave="${_chave//[[:space:]]/}"
        if [ -z "${!_chave:-}" ]; then
            export "${_chave}=${_valor}"
        fi
    done < "${_medai_raiz}/.env"
    unset _chave _valor
    _medai_env_origem=".env"
else
    _medai_env_origem="padrões (.env ausente — copie de .env.example se quiser ajustar)"
fi

: "${MEDAI_LOG_LEVEL:=INFO}"
: "${MEDAI_LOG_FORMAT:=text}"
: "${MEDAI_RANDOM_STATE:=42}"
export MEDAI_LOG_LEVEL MEDAI_LOG_FORMAT MEDAI_RANDOM_STATE

# Reprodutibilidade: hash de string do Python é aleatório por processo e pode
# afetar a ordem de iteração em estruturas que dependem dela.
export PYTHONHASHSEED="${MEDAI_RANDOM_STATE}"

# Evita que BLAS/OpenMP disputem os 12 núcleos com o paralelismo do
# scikit-learn (n_jobs), o que torna o treino mais lento, não mais rápido.
: "${OMP_NUM_THREADS:=4}"
export OMP_NUM_THREADS

# --- Diagnóstico -------------------------------------------------------------
_medai_py="$(python --version 2>&1)"
_medai_sklearn="$(python -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || echo 'ausente')"

if [ -f "${_medai_raiz}/models/metadata.json" ]; then
    _medai_modelo="treinado"
else
    _medai_modelo="ainda não treinado (make train)"
fi

cat <<BANNER

  MedAI Triage — ambiente ativo
  ─────────────────────────────────────────────────────────
  ${_medai_py}  ·  scikit-learn ${_medai_sklearn}
  Config:  ${_medai_env_origem}
  Log:     ${MEDAI_LOG_LEVEL} / ${MEDAI_LOG_FORMAT}
  Semente: ${MEDAI_RANDOM_STATE}
  Modelo:  ${_medai_modelo}

  make help     lista os comandos disponíveis
  deactivate    sai do ambiente
  ─────────────────────────────────────────────────────────

BANNER

unset _medai_raiz _medai_venv _medai_py _medai_sklearn _medai_modelo _medai_env_origem
