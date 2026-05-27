"""
Percent-change reading: render a bar or line chart that shows two values for
the same category (Before/After, two periods, or two consecutive years).
The model must compute (after - before) / before * 100.

Returns a signed float percentage (e.g. "100.0", "-25.0", "12.5").
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
from ._mcq_letter_helper import maybe_mcq_letter_wrap


_TEMPLATES_BAR = [
    "What is the percentage change from {before_label} to {after_label} for {cat}? Reply with a signed numeric percentage (e.g. 25 or -10) in <answer>...</answer>.",
    "Compute the percent change going from {before_label} to {after_label} in the {cat} category. Numeric percent in <answer>...</answer>.",
    "By what percent did {cat} change from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "What percentage increase or decrease in {cat} occurred from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Calculate the percent change for {cat} from {before_label} to {after_label}. Signed value in <answer>...</answer>.",
    "From {before_label} to {after_label}, what is the percent change in {cat}? Signed numeric in <answer>...</answer>.",
    "Find the percent change for {cat} between {before_label} and {after_label}. Signed numeric in <answer>...</answer>.",
    "How much, in percent, did {cat} change from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Determine the percentage change in {cat} from {before_label} to {after_label}. Signed value in <answer>...</answer>.",
    "What is the % change of {cat} from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Compute (After - Before) / Before * 100 for {cat} (Before = {before_label}, After = {after_label}). Signed numeric in <answer>...</answer>.",
    "Percent change in {cat} from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Express the change in {cat} from {before_label} to {after_label} as a signed percentage. Numeric answer in <answer>...</answer>.",
    "What signed percentage change does {cat} show from {before_label} to {after_label}? Numeric in <answer>...</answer>.",
    "Use the chart to find the percent change of {cat} from {before_label} to {after_label}. Signed numeric in <answer>...</answer>.",
    "Read the chart and report the percentage change of {cat} from {before_label} to {after_label}. Signed numeric in <answer>...</answer>.",
]

_TEMPLATES_LINE = [
    "What is the percent change in the value from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Compute the percent change from {before_label} to {after_label} on the line chart. Signed numeric in <answer>...</answer>.",
    "By what percentage did the value change from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Calculate the percent change in the line chart between {before_label} and {after_label}. Signed numeric in <answer>...</answer>.",
    "From {before_label} to {after_label}, what is the signed percentage change? Numeric in <answer>...</answer>.",
    "Find the percent change of the line value from {before_label} to {after_label}. Numeric in <answer>...</answer>.",
    "How much, in percent, did the line chart value change from {before_label} to {after_label}? Numeric in <answer>...</answer>.",
    "Determine the percent change in the line chart value from {before_label} to {after_label}. Numeric in <answer>...</answer>.",
    "What is the signed percent change going from {before_label} to {after_label}? Numeric in <answer>...</answer>.",
    "Compute (After - Before) / Before * 100 with Before at {before_label} and After at {after_label}. Numeric in <answer>...</answer>.",
    "Percentage change from {before_label} to {after_label}? Signed numeric in <answer>...</answer>.",
    "Express the line change from {before_label} to {after_label} as a signed percentage. Numeric in <answer>...</answer>.",
    "Read the line chart and report the percent change from {before_label} to {after_label}. Signed numeric in <answer>...</answer>.",
    "What signed percentage change is shown from {before_label} to {after_label}? Numeric in <answer>...</answer>.",
    "Use the line chart to compute the percent change from {before_label} to {after_label}. Signed numeric in <answer>...</answer>.",
    "From the chart, find the percent change between {before_label} and {after_label}. Signed numeric in <answer>...</answer>.",
]


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes"],
    ["USA", "China", "Germany", "Japan", "India"],
    ["Sales", "Eng", "Mktg", "Finance", "HR"],
    ["NY", "LA", "CHI", "HOU", "PHX"],
    ["A", "B", "C", "D", "E"],
]

_PERIOD_PAIRS = [
    ("Before", "After"),
    ("2020", "2021"),
    ("Q1", "Q2"),
    ("Old", "New"),
    ("Year 1", "Year 2"),
    ("Pre", "Post"),
    ("2022", "2023"),
    ("Phase 1", "Phase 2"),
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Sales", "Count"]


def _format_pct(x: float) -> str:
    """Format a percent-change value as a signed string. Round to 1 decimal
    if non-integer, integer otherwise. No '%' suffix in answer string —
    base verifier strips it from gt anyway."""
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class ChartPercentChangeQA(StandaloneVisualEnv):
    ENV_NAME = "chart_percent_change"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # n_cat: number of side-by-side bar pairs (or line points)
        if level == 0:
            n_cat = 1   # only one category, super easy
        elif level <= 2:
            n_cat = 3
        elif level <= 4:
            n_cat = 4
        elif level <= 6:
            n_cat = 5
        else:
            n_cat = 6
        # chart kind
        if level <= 1:
            kinds = ["bar"]
        elif level <= 5:
            kinds = ["bar", "line"]
        else:
            kinds = ["bar", "line"]
        return {"level": level, "n_cat": n_cat, "kinds": kinds}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1049 + level * 67 + 23)

        kind = rng.choice(cfg["kinds"])
        cats_pool = rng.choice(_CATEGORY_POOLS)
        n = min(cfg["n_cat"], len(cats_pool))
        cats = rng.sample(cats_pool, n)
        before_label, after_label = rng.choice(_PERIOD_PAIRS)
        y_label = rng.choice(_Y_LABELS)

        # Choose target category and clean before/after values for it.
        target_idx = rng.randrange(n)

        # L0: trivial 10 -> 20 -> +100%
        if level == 0:
            target_before = 10
            target_after = 20
        elif level <= 3:
            # Generate clean integer percent changes — pick a "nice" pct
            target_before = rng.choice([10, 20, 25, 40, 50])
            pct = rng.choice([10, 20, 25, 50, 100, -10, -20, -25, -50])
            target_after = round(target_before * (1 + pct / 100))
        else:
            # More realistic — values that yield round-ish percentages
            target_before = rng.randint(20, 100)
            # Random multiplicative factor in [0.4, 2.0]
            factor = rng.uniform(0.4, 2.0)
            target_after = max(1, round(target_before * factor))

        if target_after == target_before:
            target_after = target_before + 1
        if target_before == 0:
            target_before = 1

        # Build full series — distractor categories for n>1
        before_vals = [target_before] + [rng.randint(20, 200) for _ in range(n - 1)]
        after_vals = [target_after] + [rng.randint(20, 200) for _ in range(n - 1)]
        # Now place target at target_idx
        if target_idx != 0:
            before_vals[0], before_vals[target_idx] = before_vals[target_idx], before_vals[0]
            after_vals[0], after_vals[target_idx] = after_vals[target_idx], after_vals[0]

        cat = cats[target_idx]

        pct = (target_after - target_before) / target_before * 100.0
        answer = _format_pct(pct)

        if kind == "bar":
            sidx = (self.seed or 0) % len(_TEMPLATES_BAR)
            question = _TEMPLATES_BAR[sidx].format(
                cat=cat, before_label=before_label, after_label=after_label)
            img = self._render_grouped_bars(cats, before_vals, after_vals,
                                            before_label, after_label, y_label, rng)
        else:  # line — single category over two periods (or N points if n>1)
            # For line: x-axis is two periods, y is the values for target cat
            # If n>1 we collapse to a 2-point line over (before_label, after_label)
            sidx = (self.seed or 0) % len(_TEMPLATES_LINE)
            question = _TEMPLATES_LINE[sidx].format(
                before_label=before_label, after_label=after_label)
            img = self._render_two_point_line(target_before, target_after,
                                              before_label, after_label, y_label,
                                              cat, rng)
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, suffix="%", rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], img
        return question, answer, img

    def _render_grouped_bars(self, cats, before_vals, after_vals,
                             before_label, after_label, y_label, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        c1, c2 = palette[0], palette[1]
        n = len(cats)
        fig_w = max(6.5, 1.0 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = list(range(n))
        bw = 0.36
        b1 = ax.bar([i - bw / 2 for i in x], before_vals, width=bw,
                    color=c1, label=before_label, edgecolor="#1a1a1a", linewidth=0.5)
        b2 = ax.bar([i + bw / 2 for i in x], after_vals, width=bw,
                    color=c2, label=after_label, edgecolor="#1a1a1a", linewidth=0.5)
        max_v = max(max(before_vals), max(after_vals))
        for bar, v in zip(b1, before_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max_v * 0.015,
                    str(v), ha="center", va="bottom", fontsize=10, color="#111",
                    fontweight="bold")
        for bar, v in zip(b2, after_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max_v * 0.015,
                    str(v), ha="center", va="bottom", fontsize=10, color="#111",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 2)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _render_two_point_line(self, before_val, after_val, before_label,
                               after_label, y_label, cat, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        c1 = palette[0]
        fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = [0, 1]
        y = [before_val, after_val]
        ax.plot(x, y, "-o", color=c1, linewidth=2.5, markersize=10,
                markerfacecolor=c1, markeredgecolor="#1a1a1a", label=cat)
        # Annotate points clearly
        ax.text(0, before_val, f" {before_val}", ha="left", va="bottom",
                fontsize=12, color="#111", fontweight="bold")
        ax.text(1, after_val, f" {after_val}", ha="right", va="bottom",
                fontsize=12, color="#111", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([before_label, after_label], fontsize=12)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlim(-0.4, 1.4)
        ymax = max(before_val, after_val)
        ymin = min(before_val, after_val)
        ax.set_ylim(max(0, ymin - (ymax - ymin) * 0.3 - 5),
                    ymax + (ymax - ymin) * 0.3 + 5)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
