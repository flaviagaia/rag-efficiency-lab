# RAG Efficiency Lab — 9 Offline Experiments on RAG Tuning

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · numpy · 100% offline, sem API key · MIT License

---

## 🇧🇷 Português

### O que é

Laboratório de experimentos sobre **eficiência de RAG** — cada experimento isola um "botão" do pipeline, mede seu efeito com métricas e termina com uma lição prática. Todos rodam em segundos, offline, com seed fixa: qualquer pessoa reproduz exatamente os mesmos números.

Companheiro do [optuna-rag-tuning](https://github.com/flaviagaia/optuna-rag-tuning) (mesmo corpus e golden dataset), que cobre a otimização conjunta de chunk_size/top_k.

### Os 9 experimentos

| # | Experimento | Pergunta que responde | Resultado no corpus de exemplo |
|---|---|---|---|
| 01 | Query transforms (multi-query + RRF) | E quando o usuário pergunta "coloquial"? | recall 0.60 → **0.80** em perguntas coloquiais |
| 02 | Filtros de metadados | Por que filtrar antes de buscar? | mesma qualidade com espaço de busca 4× menor |
| 03 | Reranking em 2 estágios | Reranker resolve typos? | recall 0.60 → **1.00**, mas só com candidatos suficientes (teto de recall) |
| 04 | Montagem de contexto | Onde a resposta cai no contexto? ("lost in the middle") | score_desc concentra a resposta no INÍCIO (zona de atenção) |
| 05 | Sampling (temperature, top_p) | O que T e top_p fazem de verdade? | matemática implementada do zero, com entropia medida |
| 06 | A/B testing | A config B venceu ou deu sorte? | teste de permutação pareado, p-valor em 10k permutações |
| 07 | Latência | Por que monitorar p95 e não a média? | vetorizadores ricos custam latência em TODA pergunta |
| 08 | Semantic cache | Qual threshold usar? | 0.40 → 38% hits com 3 respostas ERRADAS; 0.50 → 29% com zero |
| 09 | Calculadora de custos | Quanto custa por mês? | config otimizada economiza ~60% em qualquer modelo |

### Execução

```bash
pip install -r requirements.txt
pytest tests/ -v                        # 10 testes: cada lição é um invariante testado
python -m lab.exp01_query_transforms    # rode qualquer experimento individualmente
python -m lab.exp08_semantic_cache
```

### Metodologia

- **Corpus**: 4 políticas internas em Markdown, chunking estrutural por seção (com metadados de origem e seção).
- **Golden dataset**: 15 perguntas com `answer_span` literal; um chunk é relevante se contém o span completo. Teste de integridade garante que todo span existe no corpus.
- **Retriever TF-IDF por decisão**: a metodologia (recall@k, MRR, RRF, dois estágios, permutação) é idêntica com embeddings neurais — só muda o vetorizador. TF-IDF permite reprodução instantânea e gratuita.
- **Cada lição é um teste**: as conclusões dos experimentos não são opinião — estão asseguradas em `tests/test_experiments.py` como invariantes (ex.: "threshold alto reduz wrong hits", "rerank com poucos candidatos não fura o teto de recall").

### Limitações honestas

Corpus pequeno e lexical. Os NÚMEROS não generalizam para o seu domínio — o MÉTODO sim. Use a estrutura dos experimentos com seu corpus, seus embeddings e seu golden dataset. Os preços do exp09 são exemplo: edite a tabela antes de qualquer decisão.

---

## 🇺🇸 English

### What it is

A lab of **RAG efficiency** experiments — each one isolates a single pipeline "knob", measures its effect with metrics and ends with a practical lesson. Everything runs in seconds, offline, with fixed seeds: anyone reproduces the exact same numbers.

Companion to [optuna-rag-tuning](https://github.com/flaviagaia/optuna-rag-tuning) (same corpus and golden dataset), which covers joint chunk_size/top_k optimization.

### The 9 experiments

| # | Experiment | Question it answers | Result on the sample corpus |
|---|---|---|---|
| 01 | Query transforms (multi-query + RRF) | What about colloquial user phrasing? | recall 0.60 → **0.80** on colloquial questions |
| 02 | Metadata filtering | Why filter before searching? | same quality on a 4× smaller search space |
| 03 | Two-stage reranking | Does a reranker fix typos? | recall 0.60 → **1.00**, but only with enough candidates (recall ceiling) |
| 04 | Context assembly | Where does the answer land? ("lost in the middle") | score_desc keeps the answer at the START (high-attention zone) |
| 05 | Sampling (temperature, top_p) | What do T and top_p actually do? | the math implemented from scratch, entropy measured |
| 06 | A/B testing | Did config B win or get lucky? | paired permutation test, p-value over 10k permutations |
| 07 | Latency | Why monitor p95, not the mean? | richer vectorizers cost latency on EVERY query |
| 08 | Semantic cache | Which threshold? | 0.40 → 38% hits with 3 WRONG answers; 0.50 → 29% with zero |
| 09 | Cost calculator | What's the monthly bill? | optimized config saves ~60% on any model |

### Running

```bash
pip install -r requirements.txt
pytest tests/ -v
python -m lab.exp01_query_transforms
```

### Methodology

Structural chunking with metadata; literal `answer_span` relevance; TF-IDF retriever by design (the methodology transfers 1:1 to neural embeddings); and **every lesson is enforced as a tested invariant** in `tests/test_experiments.py`.

### Honest limitations

Small lexical corpus. The NUMBERS won't generalize to your domain — the METHOD will. Re-run the structure with your corpus, your embeddings, your golden set. Prices in exp09 are placeholders: edit before deciding anything.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
