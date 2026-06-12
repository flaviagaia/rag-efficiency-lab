"""EXP 01 — Transformação de query: multi-query + fusão RRF.

Problema: o usuário pergunta com vocabulário coloquial; o documento
usa vocabulário formal. Retrieval lexical (e mesmo vetorial) sofre.

Técnica: gerar variantes da pergunta (em produção, um LLM gera;
aqui usamos um dicionário de reformulação determinístico para manter
o experimento offline e reprodutível) e fundir os rankings com
Reciprocal Rank Fusion (RRF).

Rode: python -m lab.exp01_query_transforms
"""

from lab.common import Chunk, Retriever, evaluate, load_chunks, load_golden, print_header

# Perguntas "difíceis": mesmo golden, vocabulário coloquial distante do corpus
HARD_QUERIES = {
    "Em quanto tempo posso cancelar um pedido?": "até quando dá pra desistir da compra?",
    "Qual o limite diário de alimentação em viagens?": "quanto posso gastar com comida quando viajo a trabalho?",
    "Qual o tamanho mínimo de senha exigido?": "quantos dígitos a senha precisa ter?",
    "Qual o valor do auxílio home office?": "quanto a empresa paga pra quem trabalha de casa?",
    "Preciso de VPN para trabalhar de casa?": "dá pra acessar os sistemas direto do wifi de casa?",
}

# Reformulações determinísticas (simulam o que um LLM geraria)
REWRITE_RULES = [
    ("desistir da compra", "cancelar pedido cancelamento"),
    ("gastar com comida", "alimentação limite diário"),
    ("quando viajo a trabalho", "viagem corporativa reembolso"),
    ("dígitos a senha", "caracteres senha mínimo"),
    ("paga pra quem trabalha de casa", "auxílio home office valor mensal"),
    ("acessar os sistemas direto do wifi de casa", "acesso remoto VPN trabalho remoto"),
]


def generate_variants(query: str) -> list[str]:
    """Variantes: original + reescritas por regra (proxy offline de multi-query)."""
    variants = [query]
    for pattern, rewrite in REWRITE_RULES:
        if pattern in query.lower():
            variants.append(rewrite)
    return variants


def rrf_search(retriever: Retriever, query: str, top_k: int, k_rrf: int = 60):
    """Busca cada variante e funde rankings com Reciprocal Rank Fusion."""
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for variant in generate_variants(query):
        for rank, (chunk, _) in enumerate(retriever.search(variant, top_k * 3), start=1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (k_rrf + rank)
            chunk_by_id[chunk.chunk_id] = chunk
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(chunk_by_id[cid], score) for cid, score in ranked]


def run(top_k: int = 3) -> dict:
    chunks = load_chunks()
    retriever = Retriever(chunks)

    # Golden "difícil": troca as perguntas pelas versões coloquiais
    golden_hard = []
    for item in load_golden():
        if item["question"] in HARD_QUERIES:
            golden_hard.append({**item, "question": HARD_QUERIES[item["question"]]})

    baseline = evaluate(retriever.search, golden_hard, top_k)
    multi = evaluate(lambda q, k: rrf_search(retriever, q, k), golden_hard, top_k)

    print_header("EXP 01 — Query transforms (perguntas coloquiais, top_k=%d)" % top_k)
    print(f"  baseline (pergunta crua) : recall={baseline['recall_at_k']:.2f} mrr={baseline['mrr']:.2f}")
    print(f"  multi-query + RRF        : recall={multi['recall_at_k']:.2f} mrr={multi['mrr']:.2f}")
    print("\n  Lição: o retrieval não falhou por falta de dados —")
    print("  falhou porque a pergunta e o documento não falam a mesma língua.")
    return {"baseline": baseline, "multi_query_rrf": multi}


if __name__ == "__main__":
    run()
