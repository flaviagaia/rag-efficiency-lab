"""Smoke tests de todos os experimentos. Rode com: pytest tests/ -v

Cada experimento é executado de verdade (são rápidos e determinísticos)
e suas conclusões centrais são verificadas como invariantes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lab import (  # noqa: E402
    exp01_query_transforms,
    exp02_metadata_filtering,
    exp03_reranker,
    exp04_context_assembly,
    exp05_sampling,
    exp06_ab_testing,
    exp07_latency,
    exp08_semantic_cache,
    exp09_cost_calculator,
)
from lab.common import load_chunks, load_golden, normalize  # noqa: E402


def test_integridade_do_golden_dataset():
    chunks = load_chunks()
    for item in load_golden():
        assert any(
            normalize(item["answer_span"]) in normalize(c.text)
            for c in chunks
            if c.source == item["source"]
        ), f"Span ausente: {item['answer_span']!r}"


def test_exp01_multi_query_melhora_perguntas_coloquiais():
    r = exp01_query_transforms.run()
    assert r["multi_query_rrf"]["recall_at_k"] >= r["baseline"]["recall_at_k"]


def test_exp02_filtro_nao_piora_e_reduz_espaco():
    r = exp02_metadata_filtering.run()
    assert r["com_filtro"]["recall_at_k"] >= r["sem_filtro"]["recall_at_k"]


def test_exp03_rerank_melhora_typos_e_mostra_teto_de_recall():
    r = exp03_reranker.run()
    chave_full = [k for k in r if k.startswith("rerank_")][-1]
    # com todos os candidatos, rerank supera o estágio 1 puro
    assert r[chave_full]["mrr"] > r["stage1"]["mrr"]
    # teto de recall: rerank com poucos candidatos não supera com muitos
    assert r["rerank_10"]["recall_at_k"] <= r[chave_full]["recall_at_k"]


def test_exp04_score_desc_concentra_no_inicio():
    zonas = exp04_context_assembly.run()
    z = zonas["score_desc"]
    assert z["início"] >= z["meio"] and z["início"] >= z["fim"]


def test_exp05_temperatura_aumenta_entropia():
    r = exp05_sampling.run()
    assert r["T=0.0  top_p=1.0"]["entropy_bits"] == 0.0
    assert r["T=1.5  top_p=1.0"]["entropy_bits"] > r["T=0.2  top_p=1.0"]["entropy_bits"]
    # top_p corta a cauda → entropia menor que sem corte na mesma T
    assert r["T=1.5  top_p=0.5"]["entropy_bits"] < r["T=1.5  top_p=1.0"]["entropy_bits"]


def test_exp06_p_valor_em_faixa_valida():
    r = exp06_ab_testing.run()
    assert 0.0 <= r["p_value"] <= 1.0


def test_exp07_reranker_full_mais_lento_que_unigram():
    r = exp07_latency.run()
    assert (
        r["char 3-5, top_k=5 (reranker full)"]["p50_ms"]
        > r["unigram, top_k=1"]["p50_ms"]
    )


def test_exp08_threshold_alto_reduz_wrong_hits():
    r = exp08_semantic_cache.run()
    assert r[0.95]["wrong_hits"] <= r[0.40]["wrong_hits"]
    assert r[0.40]["hit_rate"] >= r[0.95]["hit_rate"]


def test_exp09_otimizada_mais_barata_que_ingenua():
    r = exp09_cost_calculator.run()
    ingenua = r["ingênua (k=10, 1000 chars)"]
    otimizada = r["otimizada (k=1, 850 chars)"]
    for modelo in ingenua:
        assert otimizada[modelo] < ingenua[modelo]
