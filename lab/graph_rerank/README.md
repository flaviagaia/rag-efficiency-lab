# Graph Re-ranking — o grafo como sinal estrutural, não como filtro

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline, sem API key · experimento da Série A (RAG Eficiente)

> **Em uma frase:** a busca vetorial varre o índice inteiro e às vezes ranqueia o
> chunk do programa errado; um grafo programa→documentos re-pontua os resultados
> depois, e o chunk certo sobe. Não filtra, re-rankeia. Propriedade central:
> **nunca piora, só melhora.** No experimento: recall@3 88% → 100%, zero regressões.
>
> Programas fictícios (Alfa, Beta, Gama, Delta); nenhum dado real.

---

## 🇧🇷 Português

### O problema, em concreto

Num acervo com vários programas, a similaridade semântica mistura tudo. Pior: um
documento de um programa que cita outro contamina a busca. Exemplo: o FAQ do Gama
diz "diferente do **Beta**...". Uma pergunta sobre o Beta pode trazer esse chunk do
Gama para o topo, porque ele contém a palavra "Beta". O chunk certo cai para a 4ª
ou 5ª posição.

### Como funciona (o técnico)

O fluxo é o de produção, em quatro passos:

```
1. busca vetorial (TF-IDF)   -> top-k chunks por similaridade, varre TUDO
2. find_nodes(pergunta)      -> programas citados (casamento por alias)   # seeds
3. traverse(seeds)           -> arquivos do programa (aresta documentado_em)
4. boost                     -> score += 3.0 se chunk.arquivo ∈ arquivos_do_seed
                                reordena (estável: empates mantêm a ordem vetorial)
```

A re-pontuação é **aditiva** e recai **só** nos chunks dos arquivos do programa-seed:

```
score'(chunk) = sim(chunk, query) + (BOOST se chunk.arquivo ∈ seed_files senão 0)
```

Disso sai a **monotonicidade**, a propriedade que dá segurança para usar em
produção:

- Se o seed é encontrado, os chunks do programa certo sobem; os outros não se mexem,
  então o chunk certo **nunca cai** de posição.
- Se nenhum seed é encontrado, é um **no-op**: a ordem vetorial é preservada.

O grafo usa o **arquivo de origem** do chunk (a aresta `documentado_em`), não a
menção no texto. Por isso o chunk do Gama que cita "Beta" não é tratado como do
Beta. Custo: O(k) para re-pontuar os k candidatos; o índice não é tocado.

### Resultado real deste repositório

Acervo de 18 chunks, 4 programas, com referências cruzadas de propósito:

| Métrica | Só busca vetorial | + grafo |
| ------- | ----------------- | ------- |
| MRR | 0.713 | **0.812** |
| recall@1 | 0.500 | **0.625** |
| recall@3 | 0.875 | **1.000** |
| regressões (chunk certo que caiu) | — | **0 / 8** |

Uma pergunta cujo chunk certo estava em 5º lugar subiu para 2º. Nenhuma piorou.

### Como explicar em 30 segundos

A busca vetorial é o Google: traz por semelhança de texto. O grafo é o especialista
que olha o resultado e diz "esses três são do programa certo, sobe eles". Ele só
empurra para cima o que tem certeza; nunca empurra para baixo. Por isso só ajuda.

### Por que re-ranking e não filtragem hierárquica?

- **Recall:** filtrar antes (só buscar nos docs do programa) perderia legislações
  transversais e programas relacionados.
- **Custo:** varrer o índice é barato; pré-filtrar só compensa em acervos ordens de
  magnitude maiores.
- **Robustez:** se o grafo errar o programa, a busca vetorial ainda segura o
  resultado. O grafo melhora, mas não é ponto único de falha.

### Execução

```
pip install -r requirements.txt
python demo.py            # a tabela acima, com números reais
pytest tests/ -v          # 5 testes (monotonicidade, ganho, no-op, isolamento)
```

### Estrutura

```
data/corpus/       # 4 programas fictícios + legislação transversal (com refs cruzadas)
data/graph.json    # programa -> arquivos (aresta documentado_em) + aliases
retriever.py       # TF-IDF (o "Google" do pipeline)
graph_rerank.py    # find_nodes, traverse, boost (a re-pontuação estrutural)
demo.py            # busca pura vs + grafo
tests/             # um invariante por lição (inclui a monotonicidade)
```

### Quando mudar para filtragem hierárquica

Só se o acervo crescer para dezenas de milhares de chunks e o ruído dominar. Aí o
grafo viraria um pré-filtro (`WHERE file_name IN (...)`) antes da busca. Em acervos
pequenos e médios, re-ranking é a escolha certa.

### Limitações honestas

A monotonicidade vale quando o seed identificado está correto (ou ausente). Se o
casamento de entidade errar numa pergunta ambígua, o boost pode favorecer o programa
errado; por isso o casamento deve ser conservador (na dúvida, sem seed). Recuperação
lexical (TF-IDF); com embeddings o efeito é o mesmo.

---

## 🇺🇸 English

**In one line:** vector search scans the whole index and sometimes ranks a chunk
from the wrong program first; a program→documents graph re-scores the results
afterward, lifting the right chunk. It does not filter, it re-ranks. Core property:
**never hurts, only helps.** Result: recall@3 88% → 100%, zero regressions.
Fictional programs; no real data.

### How it works (technical)

Four steps: vector search (TF-IDF, scans everything) → `find_nodes` (programs named
in the query, by alias) → `traverse` (the program's files via the `documentado_em`
edge) → `boost` (`score += 3.0` for chunks whose file belongs to the seed program,
then stable re-sort). The boost is additive and only touches the seed program's
chunks, which yields **monotonicity**: the correct chunk never drops; with no seed
it is a no-op. The graph uses the chunk's source **file**, not textual mentions, so
a Gama chunk that says "unlike Beta" is not mistaken for Beta. Cost O(k); the index
is untouched.

### Real result

18 chunks, 4 programs, deliberate cross-references: MRR 0.713 → 0.812, recall@1
0.500 → 0.625, recall@3 0.875 → 1.000, 0/8 regressions.

### Explain it in 30 seconds

Vector search is Google (text similarity). The graph is the specialist who looks at
the results and says "these three are from the right program, move them up." It only
pushes up what it is sure of, never down, so it can only help.

### Running

```
pip install -r requirements.txt
python demo.py
pytest tests/ -v          # 5 tests
```

---

## Referências científicas (crédito aos autores)

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Cormack, Clarke & Buettcher (2009). Reciprocal Rank Fusion. SIGIR.
- Edge et al. (2024). From Local to Global: A Graph RAG Approach. Microsoft Research.
- Nogueira & Cho (2019). Passage Re-ranking with BERT.

Este repositório é uma reimplementação didática e offline dessas ideias.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
