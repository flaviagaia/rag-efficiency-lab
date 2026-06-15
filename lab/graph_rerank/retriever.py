"""Busca vetorial TF-IDF sobre o acervo, com chunking por parágrafo.

É o "Google" do pipeline: traz chunks por similaridade textual, varrendo o
índice inteiro. Não sabe nada de programas; é justamente essa cegueira
estrutural que o grafo corrige depois, no re-ranking.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str  # nome do arquivo de origem (a aresta para o grafo)


def load_chunks(corpus_dir: Path) -> list[Chunk]:
    """Um chunk por parágrafo (blocos separados por linha em branco)."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n")]
        for i, block in enumerate(b for b in blocks if b and not b.startswith("#")):
            chunks.append(Chunk(f"{path.name}::{i}", block, path.name))
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self._matrix = self._vec.fit_transform(c.text for c in chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self._vec.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]
