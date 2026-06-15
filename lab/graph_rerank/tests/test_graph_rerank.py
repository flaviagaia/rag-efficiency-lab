"""A tese do post vira invariante: o grafo nunca piora, só melhora.

- monotonicidade: o chunk certo nunca cai de posição após o re-ranking
- ganho: recall@3 melhora com o grafo
- no-op: sem seed identificado, a ordem vetorial é preservada
- isolamento: o boost só recai sobre arquivos do programa-seed
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from demo import rank_of_gold, recall_at  # noqa: E402
from graph_rerank import BOOST, ProgramGraph  # noqa: E402
from retriever import TfidfRetriever, load_chunks  # noqa: E402

TOP_K = 10
chunks = load_chunks(ROOT / "data" / "corpus")
retriever = TfidfRetriever(chunks)
graph = ProgramGraph(ROOT / "data" / "graph.json")
golden = json.loads((ROOT / "data" / "golden.json").read_text(encoding="utf-8"))


def _pair(item):
    vec = retriever.search(item["question"], TOP_K)
    return vec, graph.boost(vec, item["question"])


def test_monotonicidade_nunca_piora():
    """Para toda pergunta, a posição do chunk certo não aumenta com o grafo."""
    for item in golden:
        vec, boosted = _pair(item)
        rv, rg = rank_of_gold(vec, item), rank_of_gold(boosted, item)
        if rv is not None:
            assert rg is not None and rg <= rv, f"regressão em: {item['question']}"


def test_grafo_melhora_recall():
    """recall@3 do conjunto melhora com o re-ranking por grafo."""
    vr = [rank_of_gold(retriever.search(i["question"], TOP_K), i) for i in golden]
    gr = [rank_of_gold(graph.boost(retriever.search(i["question"], TOP_K), i["question"]), i)
          for i in golden]
    assert recall_at(gr, 3) > recall_at(vr, 3)


def test_no_op_sem_seed():
    """Pergunta sem nome de programa: a ordem vetorial é preservada (no-op)."""
    q = "Qual o prazo de prestação de contas dos recursos federais?"
    assert graph.find_nodes(q) == []
    vec = retriever.search(q, TOP_K)
    boosted = graph.boost(vec, q)
    assert [c.chunk_id for c, _ in boosted] == [c.chunk_id for c, _ in vec]


def test_boost_so_no_programa_seed():
    """O boost recai apenas em chunks de arquivos do programa identificado."""
    q = "Quem recebe o repasse do PDDE?"
    seeds = graph.find_nodes(q)
    assert seeds == ["programa__pdde"]
    seed_files = graph.traverse(seeds)
    vec = {c.chunk_id: s for c, s in retriever.search(q, TOP_K)}
    for chunk, score in graph.boost(retriever.search(q, TOP_K), q):
        esperado = vec[chunk.chunk_id] + (BOOST if chunk.source in seed_files else 0.0)
        assert abs(score - esperado) < 1e-9


def test_referencia_cruzada_nao_engana_o_grafo():
    """O PNLD cita 'PDDE' no texto, mas o grafo usa o ARQUIVO, não a menção:
    o chunk do PNLD não é tratado como sendo do PDDE."""
    seed_files = graph.traverse(["programa__pdde"])
    assert "pnld_faq.md" not in seed_files
    assert "pdde_wiki.md" in seed_files
