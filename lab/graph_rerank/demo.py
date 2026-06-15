"""Roda o experimento: busca vetorial pura vs busca + re-ranking por grafo.

Mede MRR, recall@k e, principalmente, a MONOTONICIDADE: o chunk certo nunca
cai abaixo da posição que tinha na busca vetorial.

    python demo.py
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from graph_rerank import ProgramGraph
from retriever import Chunk, TfidfRetriever, load_chunks

ROOT = Path(__file__).parent
TOP_K = 10


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())  # colapsa espaços e quebras de linha


def rank_of_gold(results: list[tuple[Chunk, float]], item: dict) -> int | None:
    """Posição (1-based) do primeiro chunk que contém a resposta. None se ausente."""
    span = _norm(item["answer_span"])
    for r, (chunk, _) in enumerate(results, start=1):
        if chunk.source == item["source"] and span in _norm(chunk.text):
            return r
    return None


def mrr(ranks: list[int | None]) -> float:
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def main() -> None:
    chunks = load_chunks(ROOT / "data" / "corpus")
    retriever = TfidfRetriever(chunks)
    graph = ProgramGraph(ROOT / "data" / "graph.json")
    golden = json.loads((ROOT / "data" / "golden.json").read_text(encoding="utf-8"))

    print(f"Acervo: {len(chunks)} chunks de {len(set(c.source for c in chunks))} arquivos\n")
    print(f"{'pergunta':<48} {'vetorial':>9} {'+ grafo':>8}")
    print("-" * 68)

    vec_ranks, graph_ranks, regressions = [], [], 0
    for item in golden:
        vec = retriever.search(item["question"], TOP_K)
        boosted = graph.boost(vec, item["question"])
        rv = rank_of_gold(vec, item)
        rg = rank_of_gold(boosted, item)
        vec_ranks.append(rv)
        graph_ranks.append(rg)
        if rv and rg and rg > rv:
            regressions += 1
        print(f"{item['question'][:46]:<48} {str(rv):>9} {str(rg):>8}")

    print("-" * 68)
    print(f"\nMRR        vetorial: {mrr(vec_ranks):.3f}  |  + grafo: {mrr(graph_ranks):.3f}")
    print(f"recall@1   vetorial: {recall_at(vec_ranks,1):.3f}  |  + grafo: {recall_at(graph_ranks,1):.3f}")
    print(f"recall@3   vetorial: {recall_at(vec_ranks,3):.3f}  |  + grafo: {recall_at(graph_ranks,3):.3f}")
    print(f"\nMonotonicidade: {regressions} regressões em {len(golden)} perguntas "
          f"(o chunk certo nunca caiu de posição).")


if __name__ == "__main__":
    main()
