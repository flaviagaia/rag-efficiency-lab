# Memória Conversacional — lembrar o que importa, não guardar tudo

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline, sem API key · experimento da Série A (RAG Eficiente)

> Demonstração sobre **dados públicos do Sistema Solar** (período orbital, diâmetro,
> posição a partir do Sol). É só um veículo neutro para mostrar as técnicas.

---

## 🇧🇷 Português

### A tese

A memória de um RAG conversacional não é guardar todo o histórico. Despejar a
conversa inteira no contexto resolve a anáfora ("e o diâmetro **dele**?"), mas tem
dois custos escondidos: o ruído dos turnos antigos puxa a busca para a entidade
errada depois de uma troca de assunto, e o contexto cresce a cada pergunta.
Memória adaptativa faz melhor com menos: identifica a **entidade ativa**, resolve
o pronome para ela, e **descarta** os turnos do assunto anterior.

### A conversa de teste

Cinco turnos sobre planetas, com anáfora e uma troca de assunto no meio (Marte →
Júpiter). Os fatos são **paralelos** (todo planeta tem período orbital, diâmetro e
posição), então a pergunta de acompanhamento só resolve com a entidade certa. O
turno decisivo é o último: "e qual o diâmetro **dele**?" logo após mudar para
Júpiter. O "dele" precisa virar Júpiter, não Marte.

### Três estratégias, resultado real

| Estratégia | Acurácia | Custo médio (palavras/consulta) |
| ---------- | -------- | ------------------------------- |
| Sem memória | 40% | 6.2 |
| Histórico cheio | 20% | **18.8** |
| **Memória adaptativa** | **100%** | 6.8 |

Leitura:

- **Sem memória** só responde os turnos que nomeiam o planeta. Os follow-ups com
  "ele/dele" ficam ambíguos: o sistema não sabe a quem você se refere.
- **Histórico cheio** paga **3x mais contexto** e acerta menos que não ter memória:
  o ruído do assunto anterior derruba o follow-up e, no turno pós-troca, o Marte
  antigo domina e a resposta sai errada.
- **Memória adaptativa** acerta **tudo** com basicamente o mesmo custo de não ter
  memória, porque resolve o pronome para a entidade ativa e poda o assunto antigo.

### O que a memória faz aqui

- **Rastreamento de entidade:** guarda a entidade ativa da conversa.
- **Resolução de anáfora:** "ele/dele/esse" vira o nome da entidade ativa.
- **Janela adaptativa por topic shift:** ao citar uma entidade nova, descarta os
  turnos da anterior (eles viraram ruído).
- **Decay:** turnos mais antigos perdem peso (0.8 por passo), proxy do "esfriar"
  do contexto com o tempo.

### Execução

```
pip install -r requirements.txt
python demo.py            # a conversa sob as 3 estratégias, com números reais
pytest tests/ -v          # 6 testes
```

### Estrutura

```
data/planetas/     # 4 planetas com fatos paralelos (dados públicos)
data/graph.json    # aliases das entidades (para o rastreamento)
memory.py          # SessionMemory: entidade ativa, anáfora, topic shift, decay
retriever.py       # TF-IDF (com stopwords PT) — o "Google" do pipeline
demo.py            # roda as 3 estratégias e compara
tests/             # um invariante por lição
```

### Limitações honestas

A resolução de anáfora aqui é por regra (lista de pronomes) e sempre aponta para a
última entidade ativa. Em produção, perguntas ambíguas, múltiplas entidades no
mesmo turno e correções ("não, eu disse o outro") exigem desambiguação mais fina
(recência, tema, confiança). O ponto deste experimento é isolar **por que** janela
adaptativa vence histórico cheio, não cobrir todos os casos.

---

## 🇺🇸 English

### The thesis

Conversational RAG memory is not about storing the whole history. Dumping the full
conversation into context resolves anaphora ("and **its** diameter?") but has two
hidden costs: noise from old turns drags retrieval toward the wrong entity after a
topic switch, and context grows every turn. Adaptive memory does better with less:
it tracks the **active entity**, resolves the pronoun to it, and **drops** the
previous topic's turns.

### Real result

A 5-turn conversation about planets, with anaphora and a mid-conversation topic
switch (Mars → Jupiter). Facts are **parallel** (every planet has an orbital
period, diameter and position), so a follow-up only resolves with the right entity.
The decisive turn is the last: "and its diameter?" right after switching to Jupiter.
"Its" must become Jupiter, not Mars.

| Strategy | Accuracy | Avg cost (words/query) |
| -------- | -------- | ---------------------- |
| No memory | 40% | 6.2 |
| Full history | 20% | **18.8** |
| **Adaptive memory** | **100%** | 6.8 |

No memory only answers turns that name the planet. Full history pays **3x the
context** and is even less accurate than no memory: stale-topic noise breaks the
follow-up, and on the post-switch turn old Mars dominates. Adaptive memory gets
everything right at essentially the cost of having no memory.

### What memory does here

Entity tracking, anaphora resolution, an **adaptive window** that drops the old
topic on a switch, and **decay** (older turns weighted 0.8 per step).

### Running

```
pip install -r requirements.txt
python demo.py
pytest tests/ -v          # 6 tests
```

### Honest limitations

Anaphora resolution is rule-based and always points to the last active entity. Real
systems need finer disambiguation (recency, topic, confidence) for ambiguous turns,
multiple entities, and corrections. The goal here is to isolate **why** an adaptive
window beats full history, not to cover every case.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
