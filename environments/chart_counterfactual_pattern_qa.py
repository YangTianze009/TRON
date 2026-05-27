"""
Chart counterfactual pattern-transfer.

Render a bar chart with a clear pattern (e.g. linear trend) where most
categories follow it, but one category (X) is an outlier. Ask the model:
  "If X had followed the pattern of Y and Z, would it be higher / lower /
   the same compared to W?"

Mirrors Hypothetical reference samples:
  - IDX 1375 ("if sales were following the same patterns as the two other categories in 99,
    would it have been higher or lower than the value in 98?" -> Lower)
  - IDX 1015 ("if nuclear energy increased by 25%, how would it compare to coal?" -> Less)

Output: single direction word (higher / lower / same).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Look at the bar chart. If {x_label} had followed the pattern set by {y_label} and {z_label}, would its value be higher, lower, or the same as the value of {w_label}? Reply with one of these words in <answer>...</answer>.",
    "Suppose {x_label} matched the trend established by {y_label} and {z_label}. Would the inferred value of {x_label} be higher, lower, or the same as {w_label}? Single word answer in <answer>...</answer>.",
    "If {x_label} were on the same trend line as {y_label} and {z_label}, how would its inferred value compare to {w_label}? Reply with `higher`, `lower`, or `same` in <answer>...</answer>.",
    "Looking at the chart, the categories {y_label} and {z_label} establish a pattern. If {x_label} had followed that pattern, would its value be higher, lower, or the same as {w_label}? One word in <answer>...</answer>.",
    "From the bar chart, if {x_label} had instead followed the linear pattern shown by {y_label} and {z_label}, would the inferred value of {x_label} be higher, lower, or the same as {w_label}? Reply with one word in <answer>...</answer>.",
    "If {x_label} were aligned with the pattern of {y_label} and {z_label}, how would its value compare to {w_label}? Answer with `higher`, `lower`, or `same` in <answer>...</answer>.",
    "Suppose {x_label}'s value followed the trend defined by {y_label} and {z_label}. Comparing this hypothetical value to {w_label}, is it higher, lower, or the same? Single word in <answer>...</answer>.",
    "From the chart, if {x_label} matched the pattern set by {y_label} and {z_label}, would its inferred value be higher, lower, or the same as {w_label}? Reply with one word in <answer>...</answer>.",
    "Examine the chart. {y_label} and {z_label} establish a trend. If {x_label} had followed that trend, how would its value compare to {w_label}? Reply with `higher`, `lower`, or `same` in <answer>...</answer>.",
    "From the chart, take the pattern shown by {y_label} and {z_label}. If {x_label} had matched that pattern, would the resulting value of {x_label} be higher, lower, or the same as {w_label}? One word in <answer>...</answer>.",
    "Look at the bar chart and infer the trend from {y_label} and {z_label}. Apply the trend to {x_label}: would the implied value be higher, lower, or the same as {w_label}? Reply with one word in <answer>...</answer>.",
    "Suppose {x_label} sits on the trend defined by {y_label} and {z_label}. Compared to {w_label}, would {x_label}'s implied value be higher, lower, or the same? Single word in <answer>...</answer>.",
    "From the chart, the categories {y_label} and {z_label} indicate a linear pattern. If {x_label} had obeyed that pattern, would its value be higher, lower, or the same as {w_label}? Reply with one word in <answer>...</answer>.",
    "Inspect the chart. Using {y_label} and {z_label} to set the pattern, what would {x_label}'s value be compared to {w_label} — higher, lower, or the same? One word in <answer>...</answer>.",
    "If {x_label} were to follow the same trend as {y_label} and {z_label}, would the resulting value be higher, lower, or the same as {w_label}? Single word in <answer>...</answer>.",
    "From the bar chart, project {x_label} onto the trend implied by {y_label} and {z_label}. Compare the projected value to {w_label}: higher, lower, or same? Reply with one word in <answer>...</answer>.",
    "Examine the chart. If {x_label} had behaved like {y_label} and {z_label}, how would its value compare with {w_label}? Use one of `higher`, `lower`, or `same` in <answer>...</answer>.",
    "Suppose the pattern set by {y_label} and {z_label} also applied to {x_label}. Comparing the inferred value to {w_label}, is it higher, lower, or the same? One word in <answer>...</answer>.",
]


_CATEGORY_POOLS = [
    ["A", "B", "C", "D", "E", "F"],
    ["Region 1", "Region 2", "Region 3", "Region 4", "Region 5"],
    ["Plan A", "Plan B", "Plan C", "Plan D", "Plan E"],
    ["Group I", "Group II", "Group III", "Group IV", "Group V"],
    ["P1", "P2", "P3", "P4", "P5", "P6"],
]

_Y_LABELS = ["Revenue", "Sales", "Visitors", "Score", "Customers", "Output"]


class ChartCounterfactualPatternQA(StandaloneVisualEnv):
    ENV_NAME = "chart_counterfactual_pattern"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_cat": 3, "step": 10, "deviation": 5, "trivial_l0": True}
        if level <= 2:
            return {"n_cat": 4, "step": 10, "deviation": 5, "trivial_l0": False}
        if level <= 4:
            return {"n_cat": 4, "step": 8, "deviation": 4, "trivial_l0": False}
        if level <= 6:
            return {"n_cat": 5, "step": 7, "deviation": 4, "trivial_l0": False}
        if level <= 8:
            return {"n_cat": 5, "step": 6, "deviation": 3, "trivial_l0": False}
        return {"n_cat": 6, "step": 5, "deviation": 3, "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1601 + level * 163 + 71)

        cat_pool = rng.choice(_CATEGORY_POOLS)
        n = min(cfg["n_cat"], len(cat_pool))
        cats = cat_pool[:n]
        y_label = rng.choice(_Y_LABELS)

        if cfg["trivial_l0"]:
            # 3 categories [10, 20, 30] linear pattern.
            # X = 15 (outlier), trend says X(at idx=1) = 20, compare with W (cat[2]=30) -> lower.
            base = 10
            step = 10
            true_values = [base + step * i for i in range(n)]
            # Pick X = idx 1, set to base + step*1 - deviation (15)
            x_idx = 1
            actual_values = list(true_values)
            actual_values[x_idx] = true_values[x_idx] - 5  # 15
            # y, z = the OTHER two: 0 and 2
            y_idx, z_idx = 0, 2
            # W = cat[2] (the true-30 one) — same as one of the trend cats. Pick something else.
            # With n=3 there's no "other" beyond y/z. Use w_idx = z_idx (compare X-pattern=20 vs cat[2]=30).
            # The hypothetical X value (20) is lower than W's actual (30).
            w_idx = z_idx
            x_pattern_val = true_values[x_idx]  # 20
            w_actual = actual_values[w_idx]  # 30
            if x_pattern_val > w_actual:
                answer = "higher"
            elif x_pattern_val < w_actual:
                answer = "lower"
            else:
                answer = "same"
            tlist = _TEMPLATES
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(
                x_label=cats[x_idx], y_label=cats[y_idx],
                z_label=cats[z_idx], w_label=cats[w_idx],
            )
            img = self._render_bars(cats, actual_values, x_idx, y_label, rng)
            return question, answer, img

        # General case: build linear trend, perturb one cat as outlier.
        base = rng.randint(20, 80)
        step = cfg["step"] * rng.choice([-1, 1])
        true_values = [base + step * i for i in range(n)]
        # Make sure all >= 5
        if min(true_values) < 5:
            shift = 5 - min(true_values)
            true_values = [v + shift for v in true_values]

        # Pick X (outlier) NOT at endpoint to keep y/z legitimate trend cats.
        x_idx = rng.randint(1, n - 2)
        deviation = cfg["deviation"]
        # Force outlier far enough that pattern-value differs unambiguously
        outlier_dir = rng.choice([-1, 1])
        actual_values = list(true_values)
        actual_values[x_idx] = max(5, true_values[x_idx] + outlier_dir * (deviation + 4))

        # y, z: pick two non-X cats that anchor the trend (prefer endpoints)
        # Endpoints best capture pattern.
        candidates = [i for i in range(n) if i != x_idx]
        y_idx = candidates[0]
        z_idx = candidates[-1]
        # W: pick a non-X, non-y, non-z category
        w_pool = [i for i in candidates if i != y_idx and i != z_idx]
        if not w_pool:
            # n too small — fallback to z
            w_idx = z_idx
        else:
            w_idx = rng.choice(w_pool)

        x_pattern_val = true_values[x_idx]  # value if pattern held
        w_actual = actual_values[w_idx]

        # Force unambiguity: if too close, nudge w
        if abs(x_pattern_val - w_actual) < 3:
            if x_pattern_val >= w_actual:
                actual_values[w_idx] = max(5, w_actual - 5)
            else:
                actual_values[w_idx] = w_actual + 5
            w_actual = actual_values[w_idx]

        if x_pattern_val > w_actual:
            answer = "higher"
        elif x_pattern_val < w_actual:
            answer = "lower"
        else:
            answer = "same"

        tlist = _TEMPLATES
        sidx = (self.seed or 0) % len(tlist)
        question = tlist[sidx].format(
            x_label=cats[x_idx], y_label=cats[y_idx],
            z_label=cats[z_idx], w_label=cats[w_idx],
        )
        img = self._render_bars(cats, actual_values, x_idx, y_label, rng)
        return question, answer, img

    def _render_bars(self, cats: List[str], values: List[float],
                     outlier_idx: int, y_label: str,
                     rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        n = len(cats)
        fig_w = max(6.5, 0.8 * n + 2.2)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Highlight the outlier in a contrasting color (signal "this category
        # doesn't follow the trend"). Other bars use single neutral palette color.
        base_color = palette[0]
        outlier_color = palette[2 % len(palette)]
        bar_colors = [outlier_color if i == outlier_idx else base_color
                      for i in range(n)]
        bars = ax.bar(range(n), values, color=bar_colors,
                       edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.02,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 4)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Custom: match higher / lower / same as a standalone word.
        import re
        p = predicted.strip().lower()
        g = ground_truth.strip().lower()
        # Map common synonyms
        synonym_map = {
            "more": "higher", "greater": "higher", "above": "higher",
            "less": "lower", "below": "lower", "smaller": "lower",
            "equal": "same", "equivalent": "same",
        }
        # First standalone direction word
        m = re.search(r"\b(higher|lower|same|more|less|greater|smaller|above|below|equal|equivalent)\b", p)
        if m:
            w = m.group(1)
            w = synonym_map.get(w, w)
            return w == g
        return False
