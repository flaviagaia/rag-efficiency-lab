"""EXP 02 — Filtros de metadados: o retrieval começa antes do vetor.

Se a aplicação SABE o domínio da pergunta (área do usuário, tipo de
documento, data), filtrar os chunks ANTES da busca vetorial:
1. elimina falsos positivos de outros domínios;
2. permite top_k menor (menos contexto, menos custo);
3. reduz o espaço de busca (menos latência em índices grandes).

Rode: python -m lab.exp02_metadata_filtering
"""

from lab.common import Retriever, evaluate, load_chunks, load_golden, print_header

# Em produção esse mapeamento vem do contexto da aplicação
# (ex.: o usuário está na página de viagens do portal de RH)
QUESTION_DOMAIN = {
    "politica_cancelamento.md": "pedidos",
    "politica_reembolso_viagens.md": "viagens",
    "politica_seguranca_informacao.md": "seguranca",
    "politica_trabalho_remoto.md": "remoto",
}


def run(top_k: int = 2) -> dict:
    chunks = load_chunks()
    golden = load_golden()

    # Sem filtro: busca no corpus inteiro
    full = Retriever(chunks)
    sem_filtro = evaluate(full.search, golden, top_k)

    # Com filtro: cada pergunta busca apenas nos chunks do seu domínio
    retriever_por_fonte = {
        source: Retriever([c for c in chunks if c.source == source])
        for source in {c.source for c in chunks}
    }

    def busca_filtrada(question: str, k: int):
        item = next(i for i in golden if i["question"] == question)
        return retriever_por_fonte[item["source"]].search(question, k)

    com_filtro = evaluate(busca_filtrada, golden, top_k)

    print_header(f"EXP 02 — Filtro de metadados (top_k={top_k})")
    print(f"  corpus inteiro       : recall={sem_filtro['recall_at_k']:.2f} mrr={sem_filtro['mrr']:.2f}")
    print(f"  filtrado por domínio : recall={com_filtro['recall_at_k']:.2f} mrr={com_filtro['mrr']:.2f}")
    print("\n  Lição: metadado bom vale mais que embedding caro.")
    print("  Filtrar antes = buscar melhor num espaço menor.")
    return {"sem_filtro": sem_filtro, "com_filtro": com_filtro}


if __name__ == "__main__":
    run()
