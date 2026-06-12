"""EXP 07 — Latência: p50 não conta a história, p95 sim.

Mede a latência de retrieval em diferentes configurações e reporta
percentis — porque o usuário que reclama é o do p95, não o da média.

Aqui medimos apenas o estágio de retrieval (offline). Em produção,
o mesmo método se aplica ao pipeline completo (retrieval + LLM),
onde top_k e chunk_size também inflam o tempo de geração.

Rode: python -m lab.exp07_latency
"""

import time

import numpy as np

from lab.common import Retriever, load_chunks, load_golden, print_header

REPETITIONS = 30  # repetições por pergunta para estabilizar a medição


def measure(search_fn, golden, top_k: int) -> np.ndarray:
    samples = []
    for _ in range(REPETITIONS):
        for item in golden:
            start = time.perf_counter()
            search_fn(item["question"], top_k)
            samples.append((time.perf_counter() - start) * 1000)
    return np.array(samples)


def run() -> dict:
    chunks = load_chunks()
    golden = load_golden()

    configs = [
        ("unigram, top_k=1", Retriever(chunks), 1),
        ("unigram, top_k=5", Retriever(chunks), 5),
        ("bigram,  top_k=5", Retriever(chunks, ngram_range=(1, 2)), 5),
        ("char 3-5, top_k=5 (reranker full)", Retriever(chunks, analyzer="char_wb", ngram_range=(3, 5)), 5),
    ]

    print_header(f"EXP 07 — Latência de retrieval ({REPETITIONS}x{len(golden)} medições por config)")
    print(f"  {'config':<34} {'p50 (ms)':>9} {'p95 (ms)':>9} {'p99 (ms)':>9}")
    results = {}
    for name, retriever, top_k in configs:
        lat = measure(retriever.search, golden, top_k)
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        results[name] = {"p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3)}
        print(f"  {name:<34} {p50:>9.3f} {p95:>9.3f} {p99:>9.3f}")

    print("\n  Lição: vetorizadores mais ricos custam latência em TODA pergunta.")
    print("  Por isso reranking em 2 estágios: pague o caro só nos candidatos.")
    print("  E monitore p95/p99 — a média esconde exatamente quem você precisa ver.")
    return results


if __name__ == "__main__":
    run()
