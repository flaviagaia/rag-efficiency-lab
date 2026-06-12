"""EXP 09 — Calculadora de custo: R$ por resposta, R$ por mês.

Custo de RAG = f(modelo, top_k, chunk_size, volume). Esta calculadora
torna a função explícita e compara cenários.

⚠️ ATENÇÃO: a tabela de preços abaixo é EXEMPLO (jun/2026) em USD por
1M de tokens. Preços de API mudam o tempo todo — atualize antes de
qualquer decisão. A estrutura da conta é o que importa.

Rode: python -m lab.exp09_cost_calculator
"""

from dataclasses import dataclass

from lab.common import print_header

# USD por 1M tokens (entrada, saída) — VALORES DE EXEMPLO, edite!
PRICE_TABLE = {
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15, 0.60),
    "deepseek-chat": (0.14, 0.28),
}

USD_BRL = 5.40           # câmbio de exemplo — edite
CHARS_PER_TOKEN = 4.0    # aproximação razoável para PT/EN
OUTPUT_TOKENS = 250      # resposta típica
SYSTEM_PROMPT_CHARS = 800


@dataclass
class RagConfig:
    name: str
    top_k: int
    chunk_chars: int


def cost_per_request_usd(model: str, cfg: RagConfig, question_chars: int = 120) -> float:
    in_price, out_price = PRICE_TABLE[model]
    input_chars = SYSTEM_PROMPT_CHARS + question_chars + cfg.top_k * cfg.chunk_chars
    input_tokens = input_chars / CHARS_PER_TOKEN
    return (input_tokens * in_price + OUTPUT_TOKENS * out_price) / 1_000_000


def run(requests_per_day: int = 2000) -> dict:
    configs = [
        RagConfig("ingênua (k=10, 1000 chars)", top_k=10, chunk_chars=1000),
        RagConfig("padrão  (k=5, 800 chars)", top_k=5, chunk_chars=800),
        RagConfig("otimizada (k=1, 850 chars)", top_k=1, chunk_chars=850),  # resultado do optuna-rag-tuning
    ]

    print_header(f"EXP 09 — Custo mensal estimado ({requests_per_day:,} req/dia, câmbio {USD_BRL})")
    print(f"  {'config':<28} " + " ".join(f"{m:>14}" for m in PRICE_TABLE))

    results: dict = {}
    for cfg in configs:
        row = {}
        line = f"  {cfg.name:<28} "
        for model in PRICE_TABLE:
            monthly_brl = cost_per_request_usd(model, cfg) * requests_per_day * 30 * USD_BRL
            row[model] = round(monthly_brl, 2)
            line += f"{'R$ ' + format(monthly_brl, ',.0f'):>14} "
        results[cfg.name] = row
        print(line)

    naive, opt = results[configs[0].name], results[configs[2].name]
    print("\n  Lição: a config otimizada economiza "
          f"{(1 - opt['gpt-4o-mini'] / naive['gpt-4o-mini']):.0%} em qualquer modelo —")
    print("  otimização de retrieval e escolha de modelo são alavancas INDEPENDENTES")
    print("  e multiplicativas. Otimize as duas.")
    return results


if __name__ == "__main__":
    run()
