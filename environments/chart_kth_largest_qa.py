"""
K-th largest identification: render a bar chart (or scatter) with N items
labeled by short categorical names; ask "which item has the K-th largest
value?" with K varying 2..5.

Targets the Text-in-Chart subtask "identify named entity by rank" (IDX 869
"What is the name of the company that has the fourth largest y value?",
IDX 786 "Which setup has the second lowest median accuracy?", IDX 26
"What is the category with the least percentage in SDBN..."). Also touches
Number-in-Chart variants when the categories are letters.

Output: the category label (case-insensitive). Labels are short alphanumeric
tokens (single letters A-H by default — these route through the verifier's
MCQ extraction for free) — so substring traps don't apply.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter


# Phrasing mirrors IDX 869, 786, 26 verbatim.
_TEMPLATES_LARGEST = [
    "Which category has the {ord} largest value? Reply with the category label in <answer>...</answer>.",
    "What is the name of the category that has the {ord} largest value in the chart? Label in <answer>...</answer>.",
    "Identify the category with the {ord} largest value. Label only in <answer>...</answer>.",
    "Among the bars shown, which one has the {ord} largest value? Label in <answer>...</answer>.",
    "Which item attains the {ord} largest value in the chart? Label answer in <answer>...</answer>.",
    "From the chart, identify the category whose value is the {ord} largest. Label in <answer>...</answer>.",
    "Which label corresponds to the bar with the {ord} largest value? In <answer>...</answer>.",
    "Find the bar with the {ord} largest value and report its label. In <answer>...</answer>.",
]

_TEMPLATES_SMALLEST = [
    "Which category has the {ord} smallest value? Reply with the category label in <answer>...</answer>.",
    "What is the name of the category that has the {ord} smallest value in the chart? Label in <answer>...</answer>.",
    "Identify the category with the {ord} smallest value. Label only in <answer>...</answer>.",
    "Among the bars shown, which one has the {ord} smallest value? Label in <answer>...</answer>.",
    "Which item attains the {ord} smallest value in the chart? Label answer in <answer>...</answer>.",
    "From the chart, identify the category whose value is the {ord} smallest. Label in <answer>...</answer>.",
    "Which label corresponds to the bar with the {ord} smallest value? In <answer>...</answer>.",
    "Find the bar with the {ord} smallest value and report its label. In <answer>...</answer>.",
]


_ORDINAL_WORDS = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}


class ChartKthLargestQA(StandaloneVisualEnv):
    ENV_NAME = "chart_kth_largest"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100 across
        # all levels because every level's difficulty was inside 4B's reach.
        # Now: L0 is trainable but not trivial; mid/high are progressively
        # harder so each level provides distinct training signal.
        level = max(0, min(level, 9))
        if level == 0:
            n_bars = 5  # was 4
            ks = [2, 3]  # was [2]
            chart_type = "bar"
            value_gap = 6.0  # was 10.0 (closer values)
            modes = ["largest", "smallest"]  # was ["largest"]
        elif level <= 2:
            n_bars = 6  # was 4
            ks = [2, 3, 4]
            chart_type = "bar"
            value_gap = 4.0  # was 6.0
            modes = ["largest", "smallest"]
        elif level <= 4:
            n_bars = 7  # was 5
            ks = [3, 4]
            chart_type = "bar"
            value_gap = 3.0  # was 4.0
            modes = ["largest", "smallest"]
        elif level <= 6:
            n_bars = 8  # was 6
            ks = [3, 4, 5]
            chart_type = "bar_or_scatter"  # was bar
            value_gap = 2.0  # was 3.0
            modes = ["largest", "smallest"]
        elif level <= 8:
            n_bars = 9  # was 7
            ks = [4, 5, 6]
            chart_type = "bar_or_scatter"
            value_gap = 1.5  # was 2.5
            modes = ["largest", "smallest"]
        else:
            n_bars = 10
            ks = [4, 5, 6, 7]
            chart_type = "bar_or_scatter"
            value_gap = 1.0
            modes = ["largest", "smallest"]
        return {"level": level, "n_bars": n_bars, "ks": ks,
                "chart_type": chart_type, "value_gap": value_gap,
                "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        import hashlib
        sd = int(hashlib.md5(f"ckl|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        n = cfg["n_bars"]
        k = rng.choice(cfg["ks"])
        mode = rng.choice(cfg["modes"])
        gap = cfg["value_gap"]

        # Generate N well-separated values (gap between adjacent ranks ≥ `gap`)
        # Start at 10, step by gap with some jitter that keeps order strict.
        # Values list (sorted ascending) with strict separation.
        base = 10.0
        sorted_vals = []
        cur = base
        for _ in range(n):
            sorted_vals.append(cur)
            cur += gap + rng.uniform(0.5, 1.5)
        # Now sorted_vals is ascending; permute the assignment to labels
        labels = [chr(ord("A") + i) for i in range(n)]   # A..H — uses MCQ tolerance
        # Permute labels so the k-th rank doesn't always go to a fixed label
        perm = labels.copy()
        rng.shuffle(perm)
        # Map: perm[i] gets sorted_vals[i] (so perm[0] is smallest, perm[-1] is largest)
        label_to_value = {perm[i]: sorted_vals[i] for i in range(n)}

        # Pick chart type
        if cfg["chart_type"] == "bar":
            chart_kind = "bar"
        elif cfg["chart_type"] == "bar_or_scatter":
            chart_kind = rng.choice(["bar", "scatter"])
        else:
            chart_kind = "bar"

        # Determine answer
        if mode == "largest":
            # k-th largest: sorted descending, index k-1
            target_label = perm[-k]   # perm[-1] is the largest, perm[-2] is 2nd largest, ...
        else:
            # k-th smallest: sorted ascending, index k-1
            target_label = perm[k - 1]

        ord_word = _ORDINAL_WORDS.get(k, f"{k}th")
        if mode == "largest":
            sidx = (self.seed or 0) % len(_TEMPLATES_LARGEST)
            question = _TEMPLATES_LARGEST[sidx].format(ord=ord_word)
        else:
            sidx = (self.seed or 0) % len(_TEMPLATES_SMALLEST)
            question = _TEMPLATES_SMALLEST[sidx].format(ord=ord_word)
        answer = target_label

        img = self._render(labels, label_to_value, chart_kind, rng)
        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        n_opts = rng.choice([4, 5])
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts,
            candidate_pool=list(labels))
        return question, answer, img

    def _render(self, labels, label_to_value, chart_kind, rng):
        n = len(labels)
        values = [label_to_value[lbl] for lbl in labels]
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        palette = ["#1f4e79", "#a6192e", "#2e7d32", "#6a1b9a", "#ef6c00",
                    "#00838f", "#5d4037", "#283593"]
        if chart_kind == "bar":
            colors = [palette[i % len(palette)] for i in range(n)]
            ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)
            ax.set_xlabel("Category", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
        else:   # scatter
            xs = list(range(n))
            colors = [palette[i % len(palette)] for i in range(n)]
            ax.scatter(xs, values, c=colors, s=140, edgecolors="black", linewidths=0.8)
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (xs[i], values[i]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=11)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Category", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
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
