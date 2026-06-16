"""Compara três estratégias de memória numa conversa multi-turno (~1s).

    python demo.py

A conversa tem anáfora ("o prazo dele?") e uma troca de assunto no meio
(Beta -> Gama). O turno decisivo é o último: "com que frequência ele é
atualizado?" — o "ele" precisa resolver para o assunto NOVO (Gama), não
para o antigo (Beta).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from memory import (
    SessionMemory,
    load_programs,
    mentions_program,
    query_adaptive,
    query_full,
    query_none,
)
from retriever import Chunk, TfidfRetriever, load_chunks

ROOT = Path(__file__).parent
CORPUS = ROOT / "data" / "planetas"
TOP_K = 5

# Conversa roteirizada sobre o Sistema Solar (conhecimento público, sem relação
# com nenhum dado interno). Os fatos são PARALELOS entre planetas (mesmo formato,
# valor diferente), então a pergunta de follow-up só resolve com a entidade certa.
# Há uma troca de assunto no meio (Marte -> Júpiter). gold = (arquivo, trecho).
CONVERSA = [
    {"texto": "Qual o período orbital de Marte?",
     "gold_source": "marte.md", "gold_span": "687 dias"},
    {"texto": "E qual o diâmetro dele?",
     "gold_source": "marte.md", "gold_span": "6.779 km"},
    {"texto": "E qual a posição dele a partir do Sol?",
     "gold_source": "marte.md", "gold_span": "Marte é o quarto planeta"},
    {"texto": "E o período orbital de Júpiter?",
     "gold_source": "jupiter.md", "gold_span": "4.333 dias"},
    {"texto": "E qual o diâmetro dele?",
     "gold_source": "jupiter.md", "gold_span": "139.820 km"},
]


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def acerta(results: list[tuple[Chunk, float]], item: dict) -> bool:
    """Verdadeiro se o chunk certo é o top-1 recuperado."""
    if not results:
        return False
    chunk = results[0][0]
    return chunk.source == item["gold_source"] and _norm(item["gold_span"]) in _norm(chunk.text)


def executar(build_query) -> list[dict]:
    """Roda a conversa sob uma estratégia. Retorna um registro por turno."""
    retriever = TfidfRetriever(load_chunks(CORPUS))
    mem = SessionMemory(programs=load_programs(ROOT / "data" / "graph.json"))
    history: list[str] = []
    registros = []
    for item in CONVERSA:
        texto = item["texto"]
        mem.observe(texto)  # a memória sempre observa; as estratégias usam ou não
        efetiva = build_query(mem, history, texto)
        # Sem um programa nomeado, a consulta é ambígua: não há base para responder.
        ambigua = not mentions_program(mem.programs, efetiva)
        ok = False if ambigua else acerta(retriever.search(efetiva, TOP_K), item)
        registros.append({
            "texto": texto, "efetiva": efetiva, "ok": ok,
            "ambigua": ambigua, "custo": len(efetiva.split()),
            "active": mem.active, "janela": len(mem.turns),
        })
        history.append(texto)
    return registros


def acuracia(registros) -> float:
    return sum(r["ok"] for r in registros) / len(registros)


def custo_medio(registros) -> float:
    return sum(r["custo"] for r in registros) / len(registros)


def roda_estrategia(nome: str, build_query) -> tuple[float, float]:
    print(f"\n=== {nome} ===")
    registros = executar(build_query)
    for r in registros:
        nota = "  (ambígua: não sabe a qual programa se refere)" if r["ambigua"] else ""
        print(f"  {'✓' if r['ok'] else '✗'} [{r['custo']:2d} palavras] {r['texto']}{nota}")
        if _norm(r["efetiva"]) != _norm(r["texto"]):
            print(f"       -> consulta efetiva: {r['efetiva']}")
    acc, custo = acuracia(registros), custo_medio(registros)
    print(f"  acurácia {acc:.0%} | custo médio {custo:.1f} palavras/consulta")
    return acc, custo


def main() -> None:
    print("=" * 64)
    print("MEMÓRIA EM RAG: guardar tudo vs lembrar o que importa")
    print("=" * 64)
    resultados = {
        "sem memória": roda_estrategia("SEM MEMÓRIA", query_none),
        "histórico cheio": roda_estrategia("HISTÓRICO CHEIO", query_full),
        "memória adaptativa": roda_estrategia("MEMÓRIA ADAPTATIVA", query_adaptive),
    }
    print("\n" + "=" * 64)
    print(f"{'estratégia':<22}{'acurácia':>10}{'custo médio':>14}")
    print("-" * 64)
    for nome, (acc, custo) in resultados.items():
        print(f"{nome:<22}{acc:>9.0%}{custo:>11.1f} p")


if __name__ == "__main__":
    main()
