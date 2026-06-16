"""A tese vira invariante: memória adaptativa acerta o follow-up com pouco custo.

- sem memória: pergunta de acompanhamento é ambígua (não respondível)
- adaptativa: resolve a anáfora para a entidade ATIVA, inclusive após topic shift
- histórico cheio: custa muito mais e não acerta mais
- topic shift poda a janela; decay pesa o recente mais que o antigo
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from demo import acuracia, custo_medio, executar  # noqa: E402
from memory import (  # noqa: E402
    SessionMemory,
    load_programs,
    query_adaptive,
    query_full,
    query_none,
)

PROGRAMS = load_programs(ROOT / "data" / "graph.json")


def test_sem_memoria_nao_responde_followup():
    """Os turnos de acompanhamento (anáfora) ficam ambíguos sem memória."""
    regs = executar(query_none)
    anaforicos = [r for r in regs if "ele" in r["texto"].lower() or "dele" in r["texto"].lower()]
    assert anaforicos and all(r["ambigua"] for r in anaforicos)
    assert acuracia(regs) < 0.5


def test_adaptativa_acerta_tudo():
    """Memória adaptativa resolve todos os follow-ups corretamente."""
    assert acuracia(executar(query_adaptive)) == 1.0


def test_adaptativa_resolve_apos_topic_shift():
    """O caso decisivo: depois de trocar de Marte para Júpiter, 'ele' vira Júpiter."""
    mem = SessionMemory(programs=PROGRAMS)
    mem.observe("Qual o período orbital de Marte?")
    assert mem.active == "entidade__marte"
    _, shift = mem.observe("E o período orbital de Júpiter?")
    assert shift is True and mem.active == "entidade__jupiter"
    resolvido = mem.resolve("E qual o diâmetro dele?")
    assert "Jupiter" in resolvido and "Marte" not in resolvido


def test_topic_shift_poda_janela():
    """Ao trocar de assunto, a janela do assunto anterior é descartada."""
    mem = SessionMemory(programs=PROGRAMS)
    mem.observe("Qual o período orbital de Marte?")
    mem.observe("E o diâmetro dele?")
    assert len(mem.turns) == 2
    mem.observe("E o período orbital de Júpiter?")  # topic shift
    assert len(mem.turns) == 1  # janela reiniciada no novo assunto


def test_historico_cheio_custa_mais_sem_acertar_mais():
    """Guardar tudo paga muito mais contexto e não melhora a acurácia."""
    adapt = executar(query_adaptive)
    full = executar(query_full)
    assert custo_medio(full) > 2 * custo_medio(adapt)
    assert acuracia(adapt) > acuracia(full)


def test_decay_pesa_recente():
    """O decay dá peso maior ao turno mais recente."""
    mem = SessionMemory(programs=PROGRAMS)
    for t in ["Qual o período de Marte?", "E o diâmetro dele?", "E a posição dele?"]:
        mem.observe(t)
    pesos = mem.window_weights()
    assert pesos[-1] == 1.0 and pesos[0] < pesos[-1]
    assert abs(pesos[0] - 0.8 ** 2) < 1e-9
