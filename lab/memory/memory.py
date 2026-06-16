"""Memória conversacional: rastrear entidade, resolver anáfora, podar por topic shift.

A tese: memória em RAG não é guardar tudo. Despejar todo o histórico no contexto
resolve a anáfora ("e o prazo dele?"), mas carrega ruído: depois de uma troca de
assunto, os turnos antigos puxam a busca para o programa errado, e o custo cresce
a cada pergunta. Memória adaptativa faz melhor com menos: identifica a entidade
ativa, resolve o pronome para ela, e descarta os turnos do assunto anterior.

Três estratégias comparadas:
- none:     cada turno é independente (sem histórico). Falha em follow-up.
- full:     concatena TODO o histórico na consulta. Resolve, mas acumula ruído e custo.
- adaptive: resolve a anáfora para a entidade ativa e poda no topic shift. Enxuto e certo.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DECAY = 0.8  # peso de um turno cai 0.8 por passo de distância (proxy de "minutos")
ANAPHORA_MARKERS = ("dele", "dela", "nele", "nela", "esse programa", "esse recurso",
                    "nesse caso", "disso", " ele ", " ela ")


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def _display(program_id: str) -> str:
    return program_id.split("__")[-1].capitalize()  # programa__beta -> Beta


@dataclass
class Turn:
    text: str
    program: str | None = None     # entidade resolvida deste turno
    topic: str | None = None       # assunto ativo no momento do turno


@dataclass
class SessionMemory:
    """Memória de sessão: entidade ativa, janela adaptativa e decay."""
    programs: dict
    active: str | None = None
    turns: list[Turn] = field(default_factory=list)

    def _explicit_program(self, text: str) -> str | None:
        """Programa citado pelo nome no texto (não por pronome). None se ausente."""
        q = _norm(text)
        for pid, info in self.programs.items():
            if any(_norm(a) in q for a in info["aliases"]):
                return pid
        return None

    def observe(self, text: str) -> tuple[Turn, bool]:
        """Registra um turno. Retorna (turno, houve_topic_shift)."""
        explicit = self._explicit_program(text)
        shift = False
        if explicit:
            shift = self.active is not None and explicit != self.active
            if shift:
                self.turns = []          # janela adaptativa: descarta assunto anterior
            self.active = explicit
        turn = Turn(text=text, program=self.active, topic=self.active)
        self.turns.append(turn)
        return turn, shift

    @staticmethod
    def is_anaphoric(text: str) -> bool:
        q = f" {_norm(text)} "
        return any(m in q for m in ANAPHORA_MARKERS)

    def resolve(self, text: str) -> str:
        """Substitui a anáfora pela entidade ativa (resolução de coreferência)."""
        if self.is_anaphoric(text) and self.active:
            return f"{text} {_display(self.active)}"
        return text

    def window_weights(self) -> list[float]:
        """Peso por decay: o turno mais recente vale 1, os anteriores caem 0.8^k."""
        n = len(self.turns)
        return [DECAY ** (n - 1 - i) for i in range(n)]


def load_programs(graph_path: Path) -> dict:
    return json.loads(graph_path.read_text(encoding="utf-8"))["programs"]


def mentions_program(programs: dict, text: str) -> bool:
    """Há algum programa nomeado no texto? Se não, a consulta é ambígua: o
    sistema não sabe a qual programa um 'ele/dele' se refere."""
    q = _norm(text)
    return any(_norm(a) in q for info in programs.values() for a in info["aliases"])


# --------------------------------------------------- construção da consulta efetiva
def query_none(_mem, history: list[str], text: str) -> str:
    """Sem memória: só a pergunta atual, crua."""
    return text


def query_full(_mem, history: list[str], text: str) -> str:
    """Histórico cheio: concatena tudo o que já foi perguntado."""
    return " ".join(history + [text])


def query_adaptive(mem: SessionMemory, _history, text: str) -> str:
    """Adaptativa: resolve a anáfora para a entidade ativa, sem arrastar o passado."""
    return mem.resolve(text)
