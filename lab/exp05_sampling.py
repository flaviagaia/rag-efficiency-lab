"""EXP 05 — Sampling: o que temperature e top_p realmente fazem.

Implementação do zero (numpy) da matemática de amostragem sobre uma
distribuição de próximo token de exemplo. Sem mística:

- temperature reescala os logits ANTES do softmax:
    p_i ∝ exp(logit_i / T)
  T→0 vira argmax (determinístico); T alto achata a distribuição.

- top_p (nucleus) corta a cauda: mantém o menor conjunto de tokens
  cuja probabilidade acumulada ≥ p, e renormaliza.

Rode: python -m lab.exp05_sampling
"""

import numpy as np

from lab.common import print_header

# Distribuição de exemplo: próximo token após "O prazo de cancelamento é de 7 dias ..."
TOKENS = ["úteis", "corridos", "após", "a", "no", "para", "completos", "bancários"]
LOGITS = np.array([4.0, 2.4, 1.1, 0.3, 0.0, -0.4, -1.2, -2.0])


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        probs = np.zeros_like(logits)
        probs[int(np.argmax(logits))] = 1.0
        return probs
    z = logits / temperature
    z = z - z.max()  # estabilidade numérica
    e = np.exp(z)
    return e / e.sum()


def apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep_n = int(np.searchsorted(cum, top_p) + 1)
    mask = np.zeros_like(probs)
    mask[order[:keep_n]] = probs[order[:keep_n]]
    return mask / mask.sum()


def entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum())


def run(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    results = {}

    print_header("EXP 05 — Temperature e top_p sobre a mesma distribuição")
    print(f"  contexto: 'O prazo de cancelamento é de 7 dias ___'\n")
    print(f"  {'config':<22} {'p(úteis)':>9} {'entropia':>9}  amostras (seed fixa)")

    for temp, top_p in [(0.0, 1.0), (0.2, 1.0), (0.7, 1.0), (1.0, 1.0), (1.5, 1.0), (0.7, 0.9), (1.5, 0.5)]:
        probs = apply_top_p(softmax_with_temperature(LOGITS, temp), top_p)
        amostras = [TOKENS[i] for i in rng.choice(len(TOKENS), size=5, p=probs)]
        label = f"T={temp:<4} top_p={top_p}"
        results[label] = {"p_top1": round(float(probs[0]), 3), "entropy_bits": round(entropy(probs), 2)}
        print(f"  {label:<22} {probs[0]:>9.3f} {entropy(probs):>9.2f}  {amostras}")

    print("\n  Lição: para extração factual e RAG corporativo, T baixa.")
    print("  T alta não deixa o modelo 'mais inteligente' — só mais imprevisível.")
    print("  top_p corta a cauda de tokens raros que causam as alucinações mais bizarras.")
    return results


if __name__ == "__main__":
    run()
