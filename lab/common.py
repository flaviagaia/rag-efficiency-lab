"""Núcleo compartilhado do laboratório: corpus, chunks, retriever e métricas.

Todos os experimentos usam o mesmo corpus (4 políticas internas) e o
mesmo golden dataset (15 perguntas com answer_span literal), garantindo
comparabilidade entre experimentos.

Retriever TF-IDF por decisão de projeto: roda offline, sem API key,
e a metodologia é idêntica à de embeddings neurais — só muda o vetorizador.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str   # metadado: arquivo de origem
    section: str  # metadado: seção (header ##)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def load_chunks() -> list[Chunk]:
    """Chunking estrutural: 1 chunk por seção (##), com metadados."""
    chunks: list[Chunk] = []
    for path in sorted((DATA / "corpus").glob("*.md")):
        content = path.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", content, flags=re.MULTILINE)
        doc_title = title.group(1).strip() if title else path.stem
        for i, part in enumerate(re.split(r"^## ", content, flags=re.MULTILINE)[1:], 1):
            lines = part.strip().splitlines()
            section = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            if body:
                chunks.append(
                    Chunk(f"{path.stem}::{i}", f"{doc_title} — {section}: {body}", path.name, section)
                )
    return chunks


def load_golden() -> list[dict]:
    return json.loads((DATA / "golden_dataset.json").read_text(encoding="utf-8"))


def is_relevant(chunk: Chunk, item: dict) -> bool:
    return normalize(item["answer_span"]) in normalize(chunk.text)


class Retriever:
    """TF-IDF + cosseno. analyzer='word' (padrão) ou 'char_wb' (reranking)."""

    def __init__(self, chunks: list[Chunk], analyzer: str = "word", ngram_range=(1, 1)) -> None:
        self.chunks = chunks
        self._vec = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", analyzer=analyzer, ngram_range=ngram_range
        )
        self._matrix = self._vec.fit_transform(c.text for c in chunks)

    def scores(self, query: str) -> list[float]:
        return cosine_similarity(self._vec.transform([query]), self._matrix).ravel().tolist()

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:top_k]
        return [(self.chunks[i], s[i]) for i in order]


def rank_of_first_relevant(results: list[tuple[Chunk, float]], item: dict) -> int | None:
    for rank, (chunk, _) in enumerate(results, start=1):
        if is_relevant(chunk, item):
            return rank
    return None


def evaluate(search_fn, golden: list[dict], top_k: int) -> dict[str, float]:
    """search_fn(question, top_k) -> list[(Chunk, score)]. Retorna recall@k e MRR."""
    hits, rrs = [], []
    for item in golden:
        rank = rank_of_first_relevant(search_fn(item["question"], top_k), item)
        hits.append(1.0 if rank else 0.0)
        rrs.append(1.0 / rank if rank else 0.0)
    n = len(golden)
    return {"recall_at_k": round(sum(hits) / n, 3), "mrr": round(sum(rrs) / n, 3)}


def print_header(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")
