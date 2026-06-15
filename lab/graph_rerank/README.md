# Graph Re-ranking — o grafo como sinal estrutural, não como filtro

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline, sem API key · experimento da Série A (RAG Eficiente)

> Os programas (Alfa, Beta, Gama, Delta) e o conteúdo são fictícios, criados só
> para a demonstração. Nenhum dado real ou interno é usado.

---

## 🇧🇷 Português

### A tese

Quando se fala em "grafo no RAG", a intuição é usar o grafo para **filtrar** o
acervo (só buscar nos documentos do programa X). Este experimento defende o
oposto: o grafo como **sinal de relevância estrutural no re-ranking**, depois da
busca vetorial, que varre o índice inteiro.

A propriedade central é a **monotonicidade**: o re-ranking por grafo **nunca
piora, só melhora**. Quando o programa citado na pergunta é identificado, ele
sobe os chunks certos; quando nenhum programa é identificado, é um no-op e a
ordem vetorial é preservada.

### O fluxo (igual ao de produção)

```
1. Busca vetorial (TF-IDF)   -> top-k chunks por similaridade, varre TUDO
2. find_nodes(pergunta)      -> identifica o programa citado (ex.: "Beta")
3. traverse(seeds)           -> arquivos do programa (aresta documentado_em)
4. boost                     -> +3.0 nos chunks desses arquivos, reordena
5. LLM                       -> recebe os chunks na ordem re-rankeada
```

O grafo **não reduz o espaço de busca**: ele age depois, como um sinal que
complementa a similaridade semântica.

### Resultado real deste repositório

Acervo de 18 chunks, 9 arquivos, 4 programas fictícios, com **referências
cruzadas** (um documento do Gama cita o Beta, etc.) para simular a contaminação
que confunde a busca puramente textual.

| Métrica | Só busca vetorial | + grafo |
| ------- | ----------------- | ------- |
| MRR | 0.713 | **0.812** |
| recall@1 | 0.500 | **0.625** |
| recall@3 | 0.875 | **1.000** |
| regressões (chunk certo que caiu) | — | **0 / 8** |

O grafo levou **todos** os chunks certos para o top 3, sem nunca rebaixar uma
resposta correta. O caso mais claro: uma pergunta cujo chunk certo estava em 5º
lugar na busca vetorial subiu para 2º com o grafo.

### Por que re-ranking e não filtragem hierárquica?

- **Recall:** filtrar antes (só buscar nos docs do programa) perderia chunks de
  legislações transversais ou de programas relacionados.
- **Custo:** varrer 18 chunks (ou alguns milhares) é barato; pré-filtrar só
  compensa em acervos ordens de magnitude maiores.
- **Robustez:** se o grafo errar o programa, a busca vetorial ainda segura o
  resultado. O grafo melhora, mas não vira ponto único de falha.

### O detalhe que o grafo acerta e o texto não

O documento do Gama diz "diferente do **Beta**...". Uma busca textual pode trazer
esse chunk do Gama para uma pergunta sobre o Beta. O grafo usa o **arquivo** de
origem (a aresta `documentado_em`), não a menção no texto, então não se confunde.
O teste `test_referencia_cruzada_nao_engana_o_grafo` garante isso.

### Execução

```
pip install -r requirements.txt
python demo.py            # a tabela acima, com números reais
pytest tests/ -v          # 5 testes (monotonicidade, ganho, no-op, isolamento)
```

### Quando mudar para filtragem hierárquica

Só se o acervo crescer para dezenas de milhares de chunks e o ruído começar a
dominar. Aí o grafo viraria um pré-filtro (`WHERE file_name IN (...)`) antes da
busca vetorial. Em acervos pequenos e médios, re-ranking é a escolha certa.

### Limitações honestas

A monotonicidade vale quando o programa-seed identificado está **correto** (ou
ausente). Se `find_nodes` identificar o programa errado numa pergunta ambígua, o
boost pode favorecer o programa errado. Por isso o casamento de entidades deve ser
conservador: na dúvida, não dar seed (e cair no no-op) é melhor que dar o seed
errado.

---

## 🇺🇸 English

### The thesis

The intuition for "graphs in RAG" is to **filter** the corpus (only search program
X's documents). This experiment argues the opposite: the graph as a **structural
relevance signal in re-ranking**, after a vector search that scans the whole index.

The key property is **monotonicity**: graph re-ranking **never hurts, only helps**.
When the program named in the question is identified, it lifts the right chunks;
when no program is found, it is a no-op and the vector ordering is preserved.

> The programs (Alfa, Beta, Gama, Delta) and their content are fictional, made only
> for this demo. No real or internal data is used.

### Real result

18 chunks, 9 files, 4 fictional programs, with **cross-references** (a Gama doc
mentions Beta, etc.) to simulate the contamination that fools purely textual search.

| Metric | Vector only | + graph |
| ------ | ----------- | ------- |
| MRR | 0.713 | **0.812** |
| recall@1 | 0.500 | **0.625** |
| recall@3 | 0.875 | **1.000** |
| regressions (correct chunk demoted) | — | **0 / 8** |

The graph brought every correct chunk into the top 3 without ever demoting a
correct answer.

### Why re-ranking instead of hierarchical filtering?

Recall (pre-filtering drops transversal/related docs), cost (scanning is cheap),
and robustness (a wrong graph guess can't break results, since vector search still
holds). The graph improves but is never a single point of failure.

### The detail the graph gets right

A Gama doc says "unlike Beta...". Textual search may surface that Gama chunk for a
Beta question. The graph uses the source **file** (the `documentado_em` edge), not
the textual mention, so it is not fooled.

### Running

```
pip install -r requirements.txt
python demo.py
pytest tests/ -v          # 5 tests
```

### Honest limitations

Monotonicity holds when the identified seed program is **correct** (or absent). A
wrong seed on an ambiguous question can favor the wrong program, so entity matching
should be conservative: when in doubt, no seed (no-op) beats the wrong seed.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
