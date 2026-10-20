"""MedAI Triage — sistema de apoio à triagem diagnóstica.

Projeto educacional. Não é dispositivo médico: nenhuma saída deste pacote
constitui diagnóstico, e a decisão clínica é sempre do médico responsável.
"""

__version__ = "0.1.0"

# Texto exibido em toda resposta da API e no rodapé da UI. Centralizado aqui
# para que exista uma única fonte da verdade do aviso clínico.
AVISO_CLINICO = (
    "Resultado de apoio à triagem. Não constitui diagnóstico. "
    "A decisão clínica é do médico responsável."
)

__all__ = ["AVISO_CLINICO", "__version__"]
