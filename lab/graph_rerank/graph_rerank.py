"""O grafo como sinal de relevância estrutural — re-ranking, não filtragem.

Fluxo (igual ao de produção):
  1. find_nodes(pergunta)  -> identifica entidades/programas citados
  2. traverse(seeds)       -> arquivos vinculados aos programas (documentado_em)
  3. boost(resultados)     -> +BOOST nos chunks desses arquivos, reordena

Propriedade central: o boost é ADITIVO e só recai sobre chunks do programa-seed.
Quando o seed é encontrado, ele sobe os chunks certos; quando NÃO há seed, é um
no-op (a ordem vetorial é preservada). Por isso o grafo "nunca piora, só melhora"
no caso em que o seed identificado está correto.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from retriever import Chunk

BOOST = 3.0  # peso do sinal estrutural somado ao score de similaridade


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class ProgramGraph:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.programs = data["programs"]
        self.transversal = set(data.get("transversal_files", []))

    # 1) encontra os programas citados na pergunta (casamento por alias)
    def find_nodes(self, question: str) -> list[str]:
        q = _norm(question)
        seeds = []
        for pid, info in self.programs.items():
            if any(_norm(a) in q for a in info["aliases"]):
                seeds.append(pid)
        return seeds

    # 2) arquivos alcançáveis a partir dos seeds (aresta documentado_em)
    def traverse(self, seeds: list[str]) -> set[str]:
        files: set[str] = set()
        for pid in seeds:
            files.update(self.programs.get(pid, {}).get("files", []))
        return files

    # 3) re-pontuação: +BOOST nos chunks de arquivos do(s) programa(s) seed
    def boost(
        self,
        results: list[tuple[Chunk, float]],
        question: str,
        boost: float = BOOST,
    ) -> list[tuple[Chunk, float]]:
        seeds = self.find_nodes(question)
        seed_files = self.traverse(seeds)
        if not seed_files:
            return list(results)  # no-op: sem seed, preserva a ordem vetorial
        rescored = [
            (chunk, score + (boost if chunk.source in seed_files else 0.0))
            for chunk, score in results
        ]
        # ordenação estável: empates mantêm a ordem vetorial original
        rescored.sort(key=lambda cs: cs[1], reverse=True)
        return rescored
