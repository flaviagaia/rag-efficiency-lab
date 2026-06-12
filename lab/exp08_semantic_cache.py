"""EXP 08 — Semantic cache: pergunta repetida não deveria pagar token.

Ideia: antes de chamar o pipeline RAG+LLM, comparar a pergunta com as
já respondidas. Se a similaridade ≥ threshold, devolver a resposta
cacheada (custo ~zero).

O trade-off central:
- threshold BAIXO  → mais cache hits → mais economia → mais risco de
  devolver resposta de OUTRA pergunta (wrong hit)
- threshold ALTO   → cache quase não dispara → economia menor, risco menor

Este experimento varre thresholds e mede hit rate e wrong-hit rate
num tráfego simulado com repetições e paráfrases.

Rode: python -m lab.exp08_semantic_cache
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lab.common import load_golden, print_header

# Paráfrases: mesma intenção, texto diferente (devem acertar o cache)
PARAPHRASES = {
    "Em quanto tempo posso cancelar um pedido?": "qual o prazo para cancelamento de um pedido?",
    "Qual o valor do auxílio home office?": "quanto é o auxílio de home office?",
    "Qual o tamanho mínimo de senha exigido?": "qual o mínimo de caracteres da senha?",
    "Quantos dias presenciais o regime híbrido exige?": "quantos dias por semana preciso ir ao escritório?",
}


def build_traffic(golden: list[dict]) -> list[tuple[str, str]]:
    """Tráfego: (pergunta, id_da_intenção). Originais + repetições + paráfrases."""
    traffic = [(item["question"], item["question"]) for item in golden]
    traffic += [(q, q) for q, _ in traffic[:5]]                      # repetições exatas
    traffic += [(p, original) for original, p in PARAPHRASES.items()]  # paráfrases
    return traffic


def run() -> dict:
    golden = load_golden()
    traffic = build_traffic(golden)
    all_texts = [q for q, _ in traffic]
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode").fit(all_texts)

    print_header("EXP 08 — Semantic cache: hit rate × wrong-hit rate")
    print(f"  tráfego: {len(traffic)} perguntas "
          f"({len(golden)} únicas, 5 repetições, {len(PARAPHRASES)} paráfrases)\n")
    print(f"  {'threshold':>9} {'hits':>5} {'hit rate':>9} {'wrong hits':>11}")

    results = {}
    for threshold in (0.40, 0.50, 0.60, 0.80, 0.95):
        cache: list[tuple[str, str]] = []  # (pergunta_original, intenção)
        hits = wrong = 0
        for question, intent in traffic:
            hit_intent = None
            if cache:
                sims = cosine_similarity(
                    vec.transform([question]), vec.transform([q for q, _ in cache])
                ).ravel()
                best = int(sims.argmax())
                if sims[best] >= threshold:
                    hit_intent = cache[best][1]
            if hit_intent is not None:
                hits += 1
                if hit_intent != intent:
                    wrong += 1  # devolveu resposta de outra intenção!
            else:
                cache.append((question, intent))
        rate = hits / len(traffic)
        results[threshold] = {"hits": hits, "hit_rate": round(rate, 2), "wrong_hits": wrong}
        print(f"  {threshold:>9.2f} {hits:>5} {rate:>9.0%} {wrong:>11}")

    print("\n  Lição: o threshold é uma decisão de NEGÓCIO, não só técnica.")
    print("  Cada wrong hit é um usuário recebendo a resposta da pergunta errada —")
    print("  em política interna é constrangedor; em saúde ou jurídico, é incidente.")
    return results


if __name__ == "__main__":
    run()
