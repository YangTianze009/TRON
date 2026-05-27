"""
Maximum profit from buy/sell sequence. Given an integer price sequence drawn
as a bar chart, compute the maximum total profit obtainable from any number
of (buy then sell) transactions, with each pair non-overlapping. Equivalent
to summing all positive consecutive differences.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the maximum profit from the daily price series below.\n\n"
    "### Game Rules:\n"
    "1. You may complete as many buy/sell transactions as you like, but you cannot hold more than one share at a time.\n"
    "2. Each transaction: buy on day i, sell on day j > i.\n"
    "3. The maximum profit equals the sum of all positive consecutive-day deltas.\n\n"
    "### Coordinate System:\n"
    "- Prices indexed by day, 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- prices = {prices}\n\n"
    "### Output Format:\n"
    "Output the maximum profit as an integer inside <answer>...</answer>.\n"
    "Example: <answer>27</answer>",

    "Compute the max profit from the price array below.\n\n"
    "### Game Rules:\n"
    "- Unlimited buy/sell transactions, no overlapping holdings.\n"
    "- Output the integer max total profit.\n\n"
    "### Coordinate System:\n"
    "- Prices are listed in day-order.\n\n"
    "### Current Puzzle State:\n"
    "prices = {prices}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",

    "Your task is to find max profit on the prices below.\n\n"
    "### Game Rules:\n"
    "Best Time to Buy and Sell Stock II — sum of positive consecutive deltas.\n\n"
    "### Coordinate System:\n"
    "- Prices are 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "prices = {prices}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]

# 2026-05-04 R4: full-gradient redesign per algorithmic-reasoning bench.
# Compound mode (L6+): At-most-K-transactions variant. Forces DP rather
# than greedy sum of positive deltas.
_TEMPLATES_K = [
    "Your task is to compute the maximum profit from the daily price series below, using AT MOST {K} buy/sell transactions.\n\n"
    "### Game Rules:\n"
    "1. You may complete AT MOST {K} buy/sell transactions (a buy followed by a sell counts as 1 transaction).\n"
    "2. You cannot hold more than one share at a time.\n"
    "3. Each transaction: buy on day i, sell on day j > i.\n\n"
    "### Coordinate System:\n"
    "- Prices indexed by day, 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- prices = {prices}\n\n"
    "### Output Format:\n"
    "Output the maximum profit as an integer inside <answer>...</answer>.",

    "Find the max profit on the prices below using at most K = {K} transactions.\n\n"
    "### Game Rules:\n"
    "- At most {K} buy/sell pairs; one share held max.\n"
    "- This is the K-transaction variant (LeetCode 188).\n\n"
    "### Current Puzzle State:\n"
    "prices = {prices}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _max_profit_unlimited(prices: List[int]) -> int:
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit


def _max_profit_k_transactions(prices: List[int], k: int) -> int:
    """Best Time to Buy and Sell Stock IV — at most k transactions."""
    n = len(prices)
    if n == 0 or k == 0:
        return 0
    if k >= n // 2:
        # Equivalent to unlimited
        return _max_profit_unlimited(prices)
    # DP: buy[i], sell[i] = max profit having done i buys / sells so far
    buy = [-float("inf")] * (k + 1)
    sell = [0] * (k + 1)
    for p in prices:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    return int(sell[k])


class StockBuySellMaxQA(StandaloneVisualEnv):
    ENV_NAME = "stock_buy_sell_max"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per algorithmic-reasoning.
        # L0-L1: 3-4 monotone increasing → trivial single-trade
        # L2-L3: 5-7 mixed prices, unlimited transactions, small range
        # L4-L5: 9-12 prices, unlimited, mid range
        # L6-L7: at-most-K transactions (K=2), 10-12 prices (DP needed)
        # L8-L9: at-most-K transactions (K=3), 15-18 prices, wider range
        level = max(0, min(level, 9))
        if level <= 1:
            n = 3 + level
            return {"level": level, "n": n, "max_price": 8,
                    "qtype": "monotone", "K": None}
        if level <= 3:
            return {"level": level, "n": 5 + (level - 2) * 2,
                    "max_price": 12,
                    "qtype": "unlimited", "K": None}
        if level <= 5:
            return {"level": level, "n": 9 + (level - 4) * 3,
                    "max_price": 20,
                    "qtype": "unlimited", "K": None}
        if level <= 7:
            return {"level": level, "n": 10 + (level - 6) * 2,
                    "max_price": 30,
                    "qtype": "k_trans", "K": 2}
        # L8-L9: K=3, longer
        return {"level": level, "n": 15 + (level - 8) * 3,
                "max_price": 50,
                "qtype": "k_trans", "K": 3}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1289 + level * 53 + 11)

        n = cfg["n"]
        qtype = cfg["qtype"]
        if qtype == "monotone":
            # Monotone increasing prices, trivial single-trade
            base = rng.randint(1, 5)
            steps = [rng.randint(1, 4) for _ in range(n - 1)]
            prices = [base]
            for s in steps:
                prices.append(prices[-1] + s)
        else:
            prices = [rng.randint(1, cfg["max_price"]) for _ in range(n)]
            if len(set(prices)) == 1:
                prices[-1] += 1

        if qtype == "k_trans":
            K = cfg["K"]
            ans = _max_profit_k_transactions(prices, K)
            sidx = (self.seed or 0) % len(_TEMPLATES_K)
            question = _TEMPLATES_K[sidx].format(prices=prices, K=K)
        else:
            ans = _max_profit_unlimited(prices)
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(prices=prices)
        img = self._render(prices)
        return question, str(ans), img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, prices: List[int]) -> Image.Image:
        n = len(prices)
        fig, ax = plt.subplots(figsize=(max(5, n * 0.55), 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bars = ax.bar(range(n), prices, color="#3a8fd6", edgecolor="#1c3f5e",
                      linewidth=1.0, width=0.78)
        ymax = max(prices)
        for i, v in enumerate(prices):
            ax.text(i, v + ymax * 0.01 + 0.2, str(v),
                    ha="center", va="bottom", fontsize=10, color="#222")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"D{i+1}" for i in range(n)])
        ax.set_xlabel("day")
        ax.set_ylabel("price")
        ax.set_ylim(0, ymax * 1.18 + 2)
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
