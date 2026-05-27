"""Diverging Bar Chart Visual QA Environment.

Horizontal bars extending left (negative) and right (positive) from center.
"""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_ITEM_POOLS = [
    ["Product A", "Product B", "Product C", "Product D", "Product E", "Product F"],
    ["USA", "UK", "Germany", "France", "Japan", "Canada", "Australia"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Dept Sales", "Dept Eng", "Dept HR", "Dept Finance", "Dept Marketing"],
    ["Feature X", "Feature Y", "Feature Z", "Feature W", "Feature V"],
]
_VALUE_LABELS = ["Net Profit ($K)", "Sentiment Score", "Change (%)",
                 "Net Revenue ($M)", "Surplus/Deficit", "Growth (%)"]

class DivergingBarQA(StandaloneVisualEnv):
    ENV_NAME = "diverging_bar"

    def _level_config(self, level: int) -> Dict:
        # Redesigned: L0 starts with net_sum + total_positive on 5 items (old L5).
        # L9 has 8 items, hidden labels, close-margin values.
        if level <= 1:
            return {"n_items": 5, "val_range": (-60, 60), "qtypes": ["total_positive", "net_sum"]}
        if level <= 3:
            return {"n_items": 6, "val_range": (-70, 70), "qtypes": ["net_sum", "largest_magnitude"]}
        if level <= 5:
            return {"n_items": 6, "val_range": (-80, 80), "qtypes": ["net_sum", "total_positive"],
                    "hide_value_labels": True}
        if level <= 7:
            return {"n_items": 7, "val_range": (-80, 80), "qtypes": ["net_sum"],
                    "hide_value_labels": True}
        return {"n_items": 8, "val_range": (-90, 90),
                "qtypes": ["net_sum", "positive_negative_ratio", "abs_total"],
                "hide_value_labels": True}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 706)
        np_rng = np.random.RandomState(seed * 100 + level)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        pool = list(sub_rng.choice(_ITEM_POOLS))
        items = pool[:cfg["n_items"]]
        n = len(items)
        vlo, vhi = cfg["val_range"]
        # Mix of positive and negative values
        values = [int(np_rng.randint(vlo, vhi)) for _ in range(n)]
        # Ensure at least one positive and one negative
        if all(v >= 0 for v in values):
            values[rng.randint(0, n - 1)] = -rng.randint(5, 40)
        if all(v <= 0 for v in values):
            values[rng.randint(0, n - 1)] = rng.randint(5, 40)

        question, answer = self._make_qa(rng, qtype, items, values)
        if question is None:
            return None

        # abs_total requires exact integer sum; force labels visible even at
        # L8-L9 where hide_value_labels is otherwise on.
        render_cfg = dict(cfg)
        if qtype == "abs_total":
            render_cfg["hide_value_labels"] = False

        style = self._random_style()
        image = self._render(style, items, values, render_cfg)
        return question, answer, image

    def _render(self, style, items, values, cfg=None):
        if cfg is None:
            cfg = {}
        n = len(items)
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(7 * style["figsize_scale"],
                                        max(3.5, n * 0.65) * style["figsize_scale"]))
        y_pos = np.arange(n)
        colors = [palette[0] if v >= 0 else palette[2 % len(palette)] for v in values]

        bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
        if not cfg.get("hide_value_labels", False):
            for i, (bar, v) in enumerate(zip(bars, values)):
                ha = "left" if v >= 0 else "right"
                offset = 1.5 if v >= 0 else -1.5
                ax.text(v + offset, i, str(v), ha=ha, va="center",
                        fontsize=style["font_size_base"] - 1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(items, fontsize=style["font_size_base"])
        ax.axvline(0, color="black", linewidth=0.8)
        v_label = self._rng.choice(_VALUE_LABELS)
        ax.set_xlabel(v_label, fontsize=style["font_size_base"])
        ax.set_title("Diverging Bar Chart", fontsize=style["font_size_base"] + 2)
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _make_qa(self, rng, qtype, items, values):
        n = len(items)
        positives = [(items[i], values[i]) for i in range(n) if values[i] > 0]
        negatives = [(items[i], values[i]) for i in range(n) if values[i] < 0]

        if qtype == "total_positive":
            total = sum(v for _, v in positives)
            return "What is the sum of all positive values?", str(total)

        elif qtype == "largest_magnitude":
            abs_vals = [(abs(values[i]), i) for i in range(n)]
            idx = max(abs_vals)[1]
            return "Which item has the largest magnitude (absolute value)?", items[idx]

        elif qtype == "net_sum":
            total = sum(values)
            return "What is the net sum of all values (positive + negative)?", str(total)

        elif qtype == "count_negative":
            cnt = len(negatives)
            return "How many items have negative values?", str(cnt)

        elif qtype == "most_positive":
            best = max(positives, key=lambda x: x[1])
            return "Which item has the most positive value?", best[0]

        elif qtype == "most_negative":
            worst = min(negatives, key=lambda x: x[1])
            return "Which item has the most negative value?", worst[0]

        elif qtype == "positive_negative_ratio":
            pos_sum = sum(v for v in values if v > 0)
            neg_sum = abs(sum(v for v in values if v < 0))
            if neg_sum == 0:
                return None, None
            ratio = round(pos_sum / neg_sum, 2)
            return ("What is the ratio of the sum of positive values to "
                    "the absolute sum of negative values? Round to 2 decimals."), str(ratio)

        elif qtype == "abs_total":
            total = sum(abs(v) for v in values)
            return ("What is the sum of the absolute values of ALL bars?"), str(total)

        else:
            total = sum(values)
            return "What is the total sum of all values?", str(total)
