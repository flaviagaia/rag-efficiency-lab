"""EXP 04 — Montagem de contexto: ordem importa ("lost in the middle").

Pesquisas mostram que LLMs dão mais atenção ao INÍCIO e ao FIM do
contexto, e "perdem" informação no meio (Liu et al., 2023).

Este experimento compara estratégias de ordenação dos chunks
recuperados e mede em que posição do contexto final a informação
relevante termina, além do tamanho total enviado.

Estratégias:
- score_desc : melhor chunk primeiro (padrão da maioria dos RAGs)
- score_asc  : melhor chunk por último (perto da pergunta)
- sandwich   : melhores nas pontas, piores no meio

Rode: python -m lab.exp04_context_assembly
"""

from lab.common import Chunk, Retriever, is_relevant, load_chunks, load_golden, print_header


def assemble(results: list[tuple[Chunk, float]], strategy: str) -> list[Chunk]:
    ordered = [c for c, _ in results]  # já vem por score desc
    if strategy == "score_desc":
        return ordered
    if strategy == "score_asc":
        return ordered[::-1]
    if strategy == "sandwich":
        # melhores chunks alternam entre as pontas; piores ficam no meio
        front, back = [], []
        for i, chunk in enumerate(ordered):
            (front if i % 2 == 0 else back).append(chunk)
        return front + back[::-1]
    raise ValueError(strategy)


def relevant_zone(context_chunks: list[Chunk], item: dict) -> str:
    """Em que terço do contexto a resposta caiu? (início/meio/fim)"""
    n = len(context_chunks)
    for i, chunk in enumerate(context_chunks):
        if is_relevant(chunk, item):
            terço = i / max(n - 1, 1)
            return "início" if terço < 0.34 else ("meio" if terço < 0.67 else "fim")
    return "ausente"


def run(top_k: int = 5) -> dict:
    retriever = Retriever(load_chunks())
    golden = load_golden()

    zonas: dict[str, dict[str, int]] = {}
    for strategy in ("score_desc", "score_asc", "sandwich"):
        contagem = {"início": 0, "meio": 0, "fim": 0, "ausente": 0}
        for item in golden:
            ctx = assemble(retriever.search(item["question"], top_k), strategy)
            contagem[relevant_zone(ctx, item)] += 1
        zonas[strategy] = contagem

    chars_medio = sum(
        len(c.text) for item in golden for c, _ in retriever.search(item["question"], top_k)
    ) // len(golden)

    print_header(f"EXP 04 — Montagem de contexto (top_k={top_k}, ~{chars_medio} chars/pergunta)")
    print(f"  {'estratégia':<12} {'início':>7} {'meio':>6} {'fim':>5} {'ausente':>8}")
    for strategy, c in zonas.items():
        print(f"  {strategy:<12} {c['início']:>7} {c['meio']:>6} {c['fim']:>5} {c['ausente']:>8}")
    print("\n  Lição: com score_desc a resposta tende ao INÍCIO (zona boa).")
    print("  Evite ordenações que empurram o melhor chunk para o MEIO do contexto.")
    print("  E lembre: contexto maior não é contexto melhor — é só mais caro.")
    return zonas


if __name__ == "__main__":
    run()
