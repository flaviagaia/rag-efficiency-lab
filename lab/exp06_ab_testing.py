"""EXP 06 — A/B testing de configurações de RAG com teste de permutação.

"A config B parece melhor" não é estatística. Com 15 perguntas,
qualquer diferença pode ser sorte. Este experimento compara duas
configs de retrieval e responde: a diferença é significativa?

Método: teste de permutação pareado (10.000 permutações) sobre o
reciprocal rank por pergunta — sem suposição de normalidade,
adequado para amostras pequenas.

Rode: python -m lab.exp06_ab_testing
"""

import numpy as np

from lab.common import Retriever, load_chunks, load_golden, print_header, rank_of_first_relevant

N_PERMUTATIONS = 10_000


def per_question_rr(search_fn, golden, top_k: int) -> np.ndarray:
    rrs = []
    for item in golden:
        rank = rank_of_first_relevant(search_fn(item["question"], top_k), item)
        rrs.append(1.0 / rank if rank else 0.0)
    return np.array(rrs)


def paired_permutation_test(a: np.ndarray, b: np.ndarray, seed: int = 42) -> float:
    """p-valor bilateral: prob. de uma diferença tão extrema sob H0."""
    rng = np.random.default_rng(seed)
    observed = abs((a - b).mean())
    diffs = a - b
    count = 0
    for _ in range(N_PERMUTATIONS):
        signs = rng.choice([1, -1], size=len(diffs))
        if abs((diffs * signs).mean()) >= observed:
            count += 1
    return count / N_PERMUTATIONS


def run() -> dict:
    chunks = load_chunks()
    golden = load_golden()

    # Config A: busca de palavras, top_k=1 | Config B: bigrams, top_k=3
    config_a = ("unigram, top_k=1", Retriever(chunks), 1)
    config_b = ("bigram,  top_k=3", Retriever(chunks, ngram_range=(1, 2)), 3)

    rr_a = per_question_rr(config_a[1].search, golden, config_a[2])
    rr_b = per_question_rr(config_b[1].search, golden, config_b[2])
    p_value = paired_permutation_test(rr_a, rr_b)

    print_header("EXP 06 — A/B test de configs de RAG (teste de permutação pareado)")
    print(f"  config A ({config_a[0]}): MRR = {rr_a.mean():.3f}")
    print(f"  config B ({config_b[0]}): MRR = {rr_b.mean():.3f}")
    print(f"  diferença observada: {abs(rr_a.mean() - rr_b.mean()):.3f}")
    print(f"  p-valor ({N_PERMUTATIONS:,} permutações): {p_value:.3f}")
    veredito = "significativa (p < 0.05)" if p_value < 0.05 else "NÃO significativa — pode ser ruído"
    print(f"  veredito: diferença {veredito}")
    print("\n  Lição: antes de migrar a config 'vencedora' para produção,")
    print("  pergunte ao p-valor se ela venceu ou se deu sorte no seu golden dataset.")
    return {"mrr_a": float(rr_a.mean()), "mrr_b": float(rr_b.mean()), "p_value": p_value}


if __name__ == "__main__":
    run()
