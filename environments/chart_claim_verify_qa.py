"""
Chart claim verification: render a bar/line chart, then make a declarative
claim that may be true or false. Model answers `true` or `false` (lowercase,
single token).

Claim families:
  - argmax claim: "X has the highest value." (true/false)
  - threshold claim: "X exceeds value T." (true/false)
  - sum-aggregate claim: "Sum of X+Y > Z." (true/false)
  - dominance claim: "Line A is always above Line B in period P." (true/false)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATE_PROMPTS = [
    "Look at the chart and decide whether the following claim is true or false. Answer with the single lowercase word `true` or `false`.\n\nClaim: {claim}",
    "Verify the claim against the chart. Reply with `true` or `false` (lowercase).\n\n{claim}",
    "Is the following claim correct based on the chart shown? Reply with one word: `true` or `false`. Place answer.\n\nClaim: {claim}",
    "Read the chart and judge: `true` or `false`? Reply with the lowercase word.\n\nStatement: {claim}",
    "Decide if the claim is supported by the chart. Single-word answer in lowercase: `true` or `false`. Place.\n\nClaim: {claim}",
    "Fact-check the claim against the chart. Reply with lowercase `true` or `false`.\n\n{claim}",
    "Based on the chart, is this true or false? Lowercase answer.\n\nClaim: {claim}",
    "Evaluate the truth value of the following statement using the chart. Reply with `true` or `false`.\n\nStatement: {claim}",
    "Verify by inspecting the chart. Lowercase reply (`true` or `false`).\n\nClaim: {claim}",
    "True or false based on the chart? Lowercase one-word reply.\n\n{claim}",
    "Determine whether the following is true or false from the chart shown. Single lowercase word.\n\nClaim: {claim}",
    "Use the chart to assess the claim. Reply `true` or `false` (lowercase). Place.\n\nClaim: {claim}",
    "Examine the chart carefully and state whether the claim holds. Reply `true` or `false` (lowercase).\n\nClaim: {claim}",
    "From the chart, is the following statement true or false? Single lowercase word.\n\nClaim: {claim}",
    "Verify the chart-based claim. Reply with `true` or `false` (lowercase). Answer.\n\nClaim: {claim}",
    "Judge: true or false? Use the chart. Lowercase one-word answer.\n\nClaim: {claim}",
]


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches", "Pears", "Cherries"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil", "France", "UK"],
    ["Sales", "Engineering", "Marketing", "Finance", "HR", "Legal", "Support", "R&D"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Philly", "Dallas", "Austin"],
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Hours"]


class ChartClaimVerifyQA(StandaloneVisualEnv):
    ENV_NAME = "chart_claim_verify"
    # Strict bare-text exact-match scoring on raw
    # prediction (no wrapper-stripping). Override the default 3-of-3 wrapper
    # instructions to teach the model to put the bare answer on the final line.
    # Verifier was extended to extract the last non-empty line as a candidate.
    _WRAPPER_INSTRUCTIONS = [
        "Think step by step. End your response with the bare answer (single letter / true|false / single word) on its own final line, with no wrapper around it.",
        "Reason through the problem, then on the very last line of your response output ONLY the bare answer (a single letter, true|false, or a single word) - no <answer>, \\boxed{}, or 'Final answer:' prefix.",
        "Work through the problem step by step. Your final line must be the bare answer alone (single letter / true|false / single word), nothing else on that line.",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_cat = 5 + (level // 2)  # 5 → 9
        if level < 3:
            claim_kind_pool = ["argmax", "threshold"]
        elif level < 6:
            claim_kind_pool = ["argmax", "threshold", "sum_aggregate"]
        else:
            claim_kind_pool = ["argmax", "threshold", "sum_aggregate", "dominance"]
        # 2026-05-04: L8/L9 cluster values to make argmax / threshold near-tie
        # so model can't shortcut by reading axis range only.
        tight = level >= 8
        return {"level": level, "n_cat": n_cat,
                "claim_kind_pool": claim_kind_pool, "tight": tight}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1051 + level * 83 + 11)

        kind = rng.choice(cfg["claim_kind_pool"])
        cats_pool = rng.choice(_CATEGORY_POOLS)
        n = min(cfg["n_cat"], len(cats_pool))
        cats = rng.sample(cats_pool, n)
        if cfg.get("tight", False):
            base = rng.randint(60, 120)
            values = [base + rng.randint(-10, 10) for _ in cats]
        else:
            values = [rng.randint(20, 200) for _ in cats]
        # ensure some spread
        max_idx = values.index(max(values))
        y_label = rng.choice(_Y_LABELS)

        if kind == "argmax":
            true_claim = f"{cats[max_idx]} has the highest {y_label.lower()}."
            wrong_idx = rng.choice([i for i in range(n) if i != max_idx])
            false_claim = f"{cats[wrong_idx]} has the highest {y_label.lower()}."
            is_true = rng.random() < 0.5
            claim = true_claim if is_true else false_claim
        elif kind == "threshold":
            # pick a random category and a threshold near its value
            i = rng.randrange(n)
            v = values[i]
            T = v + rng.choice([-30, -20, -10, 10, 20, 30])
            T = max(0, T)
            actually_exceeds = v > T
            if rng.random() < 0.5:  # true claim
                if actually_exceeds:
                    claim = f"{cats[i]} exceeds {T} {y_label.lower()}."
                    is_true = True
                else:
                    claim = f"{cats[i]} does not exceed {T} {y_label.lower()}."
                    is_true = True
            else:  # false claim
                if actually_exceeds:
                    claim = f"{cats[i]} does not exceed {T} {y_label.lower()}."
                    is_true = False
                else:
                    claim = f"{cats[i]} exceeds {T} {y_label.lower()}."
                    is_true = False
        elif kind == "sum_aggregate":
            i, j = rng.sample(range(n), 2)
            k = rng.choice([x for x in range(n) if x != i and x != j])
            sum_ij = values[i] + values[j]
            actually_gt = sum_ij > values[k]
            if rng.random() < 0.5:
                claim = f"The sum of {cats[i]} and {cats[j]} is greater than {cats[k]}."
                is_true = actually_gt
            else:
                claim = f"The sum of {cats[i]} and {cats[j]} is not greater than {cats[k]}."
                is_true = not actually_gt
        elif kind == "dominance":
            # generate two series instead
            return self._generate_dominance(seed, parameter, cfg, rng,
                                            cats_pool, y_label)
        else:
            return None

        sidx = (self.seed or 0) % len(_TEMPLATE_PROMPTS)
        question = _TEMPLATE_PROMPTS[sidx].format(claim=claim)
        answer = "true" if is_true else "false"
        img = self._render_bars(cats, values, y_label)
        return question, answer, img

    def _generate_dominance(self, seed, parameter, cfg, rng, cats_pool, y_label):
        # 2 lines over time
        months = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"][:max(4, cfg["n_cat"] - 1)]
        a = [rng.randint(40, 100) for _ in months]
        b = [rng.randint(20, 80) for _ in months]
        # Maybe make A always above B
        if rng.random() < 0.5:
            for i in range(len(months)):
                if a[i] <= b[i]:
                    a[i] = b[i] + rng.randint(5, 15)
        a_dominates = all(av > bv for av, bv in zip(a, b))
        labels_a, labels_b = "Series A", "Series B"
        if rng.random() < 0.5:
            claim = f"{labels_a} is greater than {labels_b} in every period shown."
            is_true = a_dominates
        else:
            claim = f"{labels_a} is not greater than {labels_b} in every period shown."
            is_true = not a_dominates
        sidx = (self.seed or 0) % len(_TEMPLATE_PROMPTS)
        question = _TEMPLATE_PROMPTS[sidx].format(claim=claim)
        answer = "true" if is_true else "false"
        img = self._render_lines(months, a, b, labels_a, labels_b, y_label)
        return question, answer, img

    def _render_bars(self, cats, values, y_label):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        bars = ax.bar(range(len(cats)), values,
                      color=["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                             "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
                             "#CCB974", "#64B5CD"][:len(cats)])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                    ha="center", va="bottom", fontsize=9, color="#222")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel(y_label)
        ax.set_ylim(0, max(values) * 1.2 + 5)
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render_lines(self, x_labels, a, b, lab_a, lab_b, y_label):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        x = list(range(len(x_labels)))
        ax.plot(x, a, "-o", color="#1f77b4", label=lab_a, linewidth=2)
        ax.plot(x, b, "-s", color="#d62728", label=lab_b, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Strict lowercase 'true'/'false' single-word match (with mild lenience).
        p = predicted.strip().lower().rstrip(".").strip()
        # Strip wrappers like "the answer is true" -> capture first token
        import re
        m = re.search(r"\b(true|false)\b", p)
        if not m:
            return False
        return m.group(1) == ground_truth.strip().lower()
