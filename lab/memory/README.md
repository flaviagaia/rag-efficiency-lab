# Memória Conversacional — lembrar o que importa, não guardar tudo

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline, sem API key · MIT License

> **Em uma frase:** num chat com RAG, guardar todo o histórico resolve o "e o
> diâmetro **dele**?" mas envenena a busca depois que o assunto muda. Memória
> adaptativa rastreia a entidade ativa, resolve o pronome para ela e descarta o
> assunto anterior. No experimento: **100% de acerto com 1/3 do custo de contexto.**
>
> Demonstração sobre dados públicos do Sistema Solar. É só um veículo neutro.

---

## 🇧🇷 Português

### O problema, em concreto

Conversa real é cheia de anáfora (pronomes que se referem a algo dito antes):

```
T1  Qual o período orbital de Marte?        -> precisa do nome (tem)
T2  E qual o diâmetro dele?                  -> "dele" = ?
T4  E o período orbital de Júpiter?          -> troca de assunto
T5  E qual o diâmetro dele?                  -> "dele" = Júpiter, NÃO Marte
```

Sem memória, T2 e T5 são insolúveis: o sistema não sabe quem é "dele". A reação
comum é jogar todo o histórico no contexto. Isso resolve a anáfora, mas cria dois
problemas: (1) depois da troca de assunto em T4, o histórico ainda está cheio de
Marte, e a busca de T5 é puxada para Marte; (2) o contexto cresce a cada turno,
e você paga por tokens que só atrapalham.

### Como funciona (o técnico)

Três estratégias de construir a *consulta efetiva* que vai ao retriever:

| Estratégia | Consulta efetiva | Custo |
| ---------- | ---------------- | ----- |
| `none` | só o texto do turno atual | O(1) |
| `full` | concatenação de todos os turnos | O(n) por turno, O(n²) na conversa |
| `adaptive` | turno atual + entidade ativa resolvida | O(1) |

O núcleo é uma `SessionMemory` com quatro mecanismos:

```
observe(turno):
    p := programa nomeado no texto (casamento por alias)   # detecção de entidade
    se p e p != ativo:  ativo := p;  janela := []           # topic shift -> poda
    janela.append(turno)

resolve(texto):                                             # resolução de anáfora
    se texto tem pronome ("dele","ele","esse") e ativo:
        retorna texto + nome(ativo)
    retorna texto

window_weights():                                           # decay temporal
    peso do turno i = 0.8 ^ (distância até o turno mais recente)
```

Pontos técnicos que importam:

- **Detecção de entidade** por alias (aqui, nomes de planetas); em produção,
  embeddings ou NER.
- **Topic shift** = ver uma entidade nova. A janela do assunto anterior é
  **descartada**, não apenas despriorizada. É isso que evita o ruído de T5.
- **Decay** `0.8^k`: turnos antigos perdem peso exponencialmente, proxy do
  "esfriar" do contexto. Mantém a entidade recente como a saliente.
- **Falha segura:** sem entidade nomeada e sem entidade ativa, a pergunta é
  marcada como ambígua em vez de chutar. Não inventa resposta.

### Resultado real (mesma conversa, três estratégias)

| Estratégia | Acurácia (follow-up) | Custo médio (palavras/consulta) |
| ---------- | -------------------- | ------------------------------- |
| Sem memória | 40% | 6.2 |
| Histórico cheio | 20% | **18.8** |
| **Memória adaptativa** | **100%** | 6.8 |

O histórico cheio é o pior dos dois mundos: **3x o custo** e acurácia **menor** que
não ter memória, porque o assunto antigo domina a busca depois da troca. A
adaptativa acerta tudo praticamente de graça.

### Como explicar em 30 segundos

Memória de chat não é um diário que cresce para sempre. É um post-it: você anota de
quem está falando agora, troca o post-it quando o assunto muda, e joga o antigo
fora. Guardar tudo não é lembrar melhor; é se confundir mais caro.

### Execução

```
pip install -r requirements.txt
python demo.py            # a conversa sob as 3 estratégias, com números reais
pytest tests/ -v          # 6 testes (resolução, topic shift, decay, custo)
```

### Estrutura

```
data/planetas/   # 4 planetas com fatos paralelos (dados públicos)
data/graph.json  # aliases das entidades (detecção de entidade)
memory.py        # SessionMemory: entidade ativa, anáfora, topic shift, decay
retriever.py     # TF-IDF com stopwords PT (o "Google" do pipeline)
demo.py          # roda e compara as 3 estratégias
tests/           # um invariante por lição
```

### Limitações honestas

Resolução de anáfora por regra, sempre apontando para a última entidade ativa.
Produção precisa de desambiguação mais fina (recência + tema + confiança) para
turnos ambíguos, múltiplas entidades e correções ("não, o outro"). O retriever é
lexical (TF-IDF); com embeddings o efeito é o mesmo, muda o recuperador. O objetivo
é isolar **por que** janela adaptativa vence histórico cheio.

---

## 🇺🇸 English

**In one line:** in a RAG chat, keeping the whole history resolves "and **its**
diameter?" but poisons retrieval after the topic changes. Adaptive memory tracks the
active entity, resolves the pronoun to it, and drops the old topic. Result: **100%
accuracy at 1/3 of the context cost.** Demo over public Solar System data.

### The problem

Conversations are full of anaphora. Without memory, follow-ups like "and its
diameter?" are unsolvable. The common fix, dumping all history into context, both
(1) lets the stale topic dominate retrieval after a topic switch and (2) grows cost
every turn.

### How it works (technical)

Three ways to build the *effective query*: `none` (current turn only), `full`
(concatenate all turns, O(n²) over the conversation), `adaptive` (current turn +
resolved active entity, O(1)). `SessionMemory` does entity detection (alias match),
topic-shift detection (a new entity **drops** the previous window), rule-based
anaphora resolution, and exponential `0.8^k` decay. Fail-safe: with no entity, the
question is flagged ambiguous instead of guessing.

### Real result

| Strategy | Follow-up accuracy | Avg cost (words/query) |
| -------- | ------------------ | ---------------------- |
| No memory | 40% | 6.2 |
| Full history | 20% | **18.8** |
| **Adaptive** | **100%** | 6.8 |

Full history is the worst of both worlds: 3x the cost, lower accuracy than no
memory, because the old topic dominates retrieval after the switch.

### Explain it in 30 seconds

Chat memory isn't a diary that grows forever. It's a sticky note: write down who
you're talking about now, swap it when the subject changes, throw the old one away.

### Running

```
pip install -r requirements.txt
python demo.py
pytest tests/ -v          # 6 tests
```

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
