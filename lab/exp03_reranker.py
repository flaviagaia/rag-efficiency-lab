"""EXP 03 — Reranking em dois estágios.

Estágio 1 (rápido, barato): TF-IDF de palavras recupera 20 candidatos.
Estágio 2 (mais caro, melhor): re-pontua os 20 com n-grams de
caracteres (3-5), robustos a erros de digitação e variação morfológica.

Em produção, o estágio 2 seria um cross-encoder; a estrutura do
pipeline é exatamente a mesma — pontuar de novo um conjunto pequeno.

Cenário de teste: perguntas com erros de digitação (realidade de
qualquer caixa de busca corporativa).

Rode: python -m lab.exp03_reranker
"""

from lab.common import Retriever, evaluate, load_chunks, load_golden, print_header

# Perguntas com typos pesados (golden inalterado)
TYPO_QUERIES = {
    "Em quanto tempo posso cancelar um pedido?": "em qnto temp posso cancelr um pedio?",
    "Qual o limite diário de alimentação em viagens?": "limte diaro de alimntacao em viajens",
    "Qual o tamanho mínimo de senha exigido?": "tamanh minmo de senga exigdo",
    "Quantos dias presenciais o regime híbrido exige?": "qntos dias presencias o regme hibrdo exje",
    "Qual o valor do auxílio home office?": "valr do auxlio rome ofice",
}


def run(top_k: int = 3) -> dict:
    chunks = load_chunks()
    stage1 = Retriever(chunks)  # palavras (rápido, frágil a typos)

    golden_typo = [
        {**item, "question": TYPO_QUERIES[item["question"]]}
        for item in load_golden()
        if item["question"] in TYPO_QUERIES
    ]

    def com_rerank(n_candidates: int):
        def busca(q: str, k: int):
            candidatos = [c for c, _ in stage1.search(q, n_candidates)]
            local = Retriever(candidatos, analyzer="char_wb", ngram_range=(3, 5))
            return local.search(q, k)

        return busca

    base = evaluate(stage1.search, golden_typo, top_k)
    resultados = {"stage1": base}

    print_header(f"EXP 03 — Reranking 2 estágios (typos pesados, top_k={top_k}, corpus={len(chunks)} chunks)")
    print(f"  estágio 1 apenas (palavras)     : recall={base['recall_at_k']:.2f} mrr={base['mrr']:.2f}")
    for n in (10, 20, len(chunks)):
        m = evaluate(com_rerank(n), golden_typo, top_k)
        resultados[f"rerank_{n}"] = m
        print(f"  rerank chars sobre {n:>2} candidatos: recall={m['recall_at_k']:.2f} mrr={m['mrr']:.2f}")

    print("\n  Duas lições em uma:")
    print("  1. Reranker NÃO conserta o que o estágio 1 não trouxe (teto de recall).")
    print("     Dimensione o nº de candidatos olhando o recall do estágio 1, não o custo.")
    print("  2. Com candidatos suficientes, o pipeline barato+caro atinge a qualidade")
    print("     do método caro — pagando o caro só numa fração do corpus.")
    return resultados


if __name__ == "__main__":
    run()
