"""
Multi-line dominance claim verification.

Render a 2-line chart over time periods (or 3 lines at higher levels) and
present a claim about line dominance, e.g.:
  - "Line A is greater than Line B in every period."
  - "Line A is greater than Line B in at least one period."

Model output: lowercase single word `yes` or `no` (chart-style binary).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Look at the line chart and decide if the claim holds. Reply with the lowercase single word `yes` or `no`.\n\nClaim: {claim}",
    "Verify the following dominance claim using the line chart. Reply with `yes` or `no` (lowercase).\n\n{claim}",
    "Read the line chart and judge whether the claim holds across all periods shown. Single lowercase word.\n\nStatement: {claim}",
    "Inspect each period on the line chart and decide: yes or no? Single lowercase word.\n\nClaim: {claim}",
    "Use the line chart to fact-check the claim. Lowercase reply (`yes`/`no`).\n\nClaim: {claim}",
    "From the line chart, does this hold? Lowercase reply (`yes`/`no`).\n\nClaim: {claim}",
    "Evaluate the claim against the multi-line chart. Reply with `yes` or `no`.\n\nStatement: {claim}",
    "Compare the lines period by period. Reply `yes` or `no` (lowercase).\n\nClaim: {claim}",
    "Yes or no based on the chart? Lowercase one-word reply.\n\n{claim}",
    "Determine whether the dominance statement holds from the chart. Single lowercase word (`yes`/`no`).\n\nClaim: {claim}",
    "Use the line chart to assess the claim. Reply `yes` or `no` (lowercase).\n\nClaim: {claim}",
    "Examine the lines carefully and state whether the claim holds. Reply `yes` or `no` (lowercase).\n\nClaim: {claim}",
    "From the chart, does the following statement hold? Single lowercase word (`yes`/`no`).\n\nClaim: {claim}",
    "Verify the line dominance claim against the chart. Reply with `yes` or `no` (lowercase).\n\nClaim: {claim}",
    "Judge: yes or no? Use the line chart. Lowercase one-word answer.\n\nClaim: {claim}",
    "Check each period and decide if the claim holds. Reply with `yes` or `no` (lowercase).\n\nClaim: {claim}",
    "Read the lines period by period and verify the claim. Lowercase `yes`/`no`.\n\nClaim: {claim}",
    "Cross-check the lines on the chart against the claim. Reply with the lowercase word `yes` or `no`.\n\nClaim: {claim}",
]


_PERIOD_POOLS = [
    ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
    ["Wk1", "Wk2", "Wk3", "Wk4", "Wk5", "Wk6", "Wk7", "Wk8"],
    ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8"],
]

_LINE_LABEL_POOLS = [
    ("Line A", "Line B", "Line C"),
    ("Series A", "Series B", "Series C"),
    ("Group 1", "Group 2", "Group 3"),
    ("North", "South", "East"),
    ("Plan A", "Plan B", "Plan C"),
    ("Set 1", "Set 2", "Set 3"),
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Hours", "Sales", "Count"]


class MultiLineDominanceClaimQA(StandaloneVisualEnv):
    ENV_NAME = "multi_line_dominance_claim"
    # Strict bare-text exact-match scoring on raw
    # prediction (no wrapper-stripping). Override the default 3-of-3 wrapper
    # instructions to teach the model to put the bare answer on the final line.
    # Verifier was extended to extract the last non-empty line as a candidate.
    _WRAPPER_INSTRUCTIONS = [
        "Think step by step. End your response with the bare answer (single word `yes` or `no`) on its own final line, with no wrapper around it.",
        "Reason through the problem, then on the very last line of your response output ONLY the bare answer (`yes` or `no`) - no <answer>, \\boxed{}, or 'Final answer:' prefix.",
        "Work through the problem step by step. Your final line must be the bare answer alone (`yes` or `no`), nothing else on that line.",
    ]

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100.
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_periods": 4, "n_lines": 2, "trivial_l0": True}  # was 3
        if level <= 2:
            return {"n_periods": 5, "n_lines": 2, "trivial_l0": False}  # was 4
        if level <= 4:
            return {"n_periods": 6, "n_lines": 3, "trivial_l0": False}  # was 5,2
        if level <= 6:
            return {"n_periods": 7, "n_lines": 3, "trivial_l0": False}  # was 6,2
        if level <= 8:
            return {"n_periods": 8, "n_lines": 3, "trivial_l0": False}  # was 7,3
        return {"n_periods": 8, "n_lines": 3, "trivial_l0": False, "tight_values": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1093 + level * 91 + 31)

        n_periods = cfg["n_periods"]
        n_lines = cfg["n_lines"]
        period_pool = rng.choice(_PERIOD_POOLS)
        periods = period_pool[:n_periods]
        labels_full = rng.choice(_LINE_LABEL_POOLS)
        labels = list(labels_full[:n_lines])
        y_label = rng.choice(_Y_LABELS)

        if cfg["trivial_l0"]:
            # L0: Line A clearly above Line B in EVERY period.
            # values: 4 periods. A always > B.
            series = [[40, 50, 60, 70][:n_periods], [10, 20, 30, 40][:n_periods]]
            # Always claim "A is greater than B in every period" → true.
            claim = (f"{labels[0]} is greater than {labels[1]} "
                     f"in every period shown.")
            is_true = True
        else:
            # Higher level: random monotone-ish series; sometimes A dominates,
            # sometimes not. Random claim polarity.
            # 2026-05-04: bumped L9 difficulty — when tight_values, lines are
            # closer in value (smaller margin to read), forcing precise reads.
            tight = cfg.get("tight_values", False)
            series = []
            base = 50
            for li in range(n_lines):
                start = rng.randint(35, 65) if tight else rng.randint(20, 80)
                trend = rng.choice([-1, 0, 1]) * rng.randint(0, 4 if tight else 8)
                vals = []
                v = start
                for _ in range(n_periods):
                    jitter = rng.randint(-3, 3) if tight else rng.randint(-6, 6)
                    v = max(5, v + trend + jitter)
                    vals.append(v)
                series.append(vals)

            # 50/50: force A always above B for the "true" case
            forced_dominant = rng.random() < 0.5
            if forced_dominant:
                margin_lo, margin_hi = (2, 5) if tight else (5, 15)
                for i in range(n_periods):
                    if series[0][i] <= series[1][i]:
                        series[0][i] = series[1][i] + rng.randint(margin_lo, margin_hi)

            a = series[0]
            b = series[1]
            a_dominates_all = all(av > bv for av, bv in zip(a, b))
            a_dominates_any = any(av > bv for av, bv in zip(a, b))

            # Pick claim form
            kind = rng.choice(["always", "never", "any"])
            polarity = rng.random() < 0.5  # positive vs negative phrasing
            if kind == "always":
                if polarity:
                    claim = (f"{labels[0]} is greater than {labels[1]} "
                             f"in every period shown.")
                    is_true = a_dominates_all
                else:
                    claim = (f"{labels[0]} is not greater than {labels[1]} "
                             f"in every period shown.")
                    is_true = not a_dominates_all
            elif kind == "never":
                # equivalent to "B >= A always"
                b_ge_all = all(bv >= av for av, bv in zip(a, b))
                if polarity:
                    claim = (f"{labels[0]} is never greater than "
                             f"{labels[1]} in any period shown.")
                    is_true = b_ge_all
                else:
                    claim = (f"{labels[0]} is greater than {labels[1]} "
                             f"in at least one period shown.")
                    is_true = a_dominates_any
            else:  # any
                if polarity:
                    claim = (f"{labels[0]} is greater than {labels[1]} "
                             f"in at least one period shown.")
                    is_true = a_dominates_any
                else:
                    claim = (f"{labels[0]} is never greater than "
                             f"{labels[1]} in any period shown.")
                    is_true = not a_dominates_any

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(claim=claim)
        answer = "yes" if is_true else "no"
        img = self._render(periods, series, labels, y_label, rng)
        return question, answer, img

    def _render(self, periods: List[str], series: List[List[float]],
                labels: List[str], y_label: str, rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        markers = ["o", "s", "D"]
        x = list(range(len(periods)))
        for li, (vals, lab) in enumerate(zip(series, labels)):
            ax.plot(x, vals, marker=markers[li % len(markers)], linewidth=2.0,
                    color=palette[li % len(palette)], label=lab,
                    markersize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlabel("Period", fontsize=11)
        all_vals = [v for s in series for v in s]
        ax.set_ylim(0, max(all_vals) * 1.18 + 5)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Strict lowercase 'yes'/'no' single-word match (with mild lenience).
        import re as _re
        p = predicted.strip().lower().rstrip(".").strip()
        m = _re.search(r"\b(yes|no)\b", p)
        if not m:
            return False
        return m.group(1) == ground_truth.strip().lower()
