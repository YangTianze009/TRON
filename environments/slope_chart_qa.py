"""Slope Chart Visual QA Environment.

Two time points connected by lines showing change per item.
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
    ["Company A", "Company B", "Company C", "Company D", "Company E", "Company F"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil"],
    ["Physics", "Chemistry", "Biology", "Math", "English", "History"],
    ["Product X", "Product Y", "Product Z", "Product W", "Product V"],
]
_TIME_PAIRS = [("2020", "2024"), ("Before", "After"), ("Q1", "Q4"),
               ("Year 1", "Year 5"), ("Jan", "Dec")]
_Y_LABELS = ["Revenue ($M)", "Score", "Market Share (%)", "Rating", "Index"]

class SlopeChartQA(StandaloneVisualEnv):
    ENV_NAME = "slope_chart"

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["most_increased", "highest_at_t2"],
                    "qweights": [5, 5], "n_items": (3, 4)}
        if level <= 2:
            return {"qtypes": ["most_increased", "most_decreased", "count_increased"],
                    "qweights": [3, 3, 4], "n_items": (4, 5)}
        if level <= 4:
            return {"qtypes": ["most_increased", "most_decreased", "net_change", "count_increased"],
                    "qweights": [3, 3, 2, 2], "n_items": (4, 6)}
        if level <= 6:
            return {"qtypes": ["net_change", "how_many_crossed", "count_increased", "rank_change_count"],
                    "qweights": [3, 3, 2, 2], "n_items": (5, 6)}
        if level <= 8:
            return {"qtypes": ["rank_change_count", "how_many_crossed", "max_percentage_change"],
                    "qweights": [3, 3, 4], "n_items": (5, 6)}
        return {"qtypes": ["max_percentage_change", "rank_change_count"],
                "qweights": [6, 4], "n_items": (5, 6)}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 9901)
        np_rng = np.random.RandomState(seed * 100 + level)
        qtype = parameter.get("question_type")
        if qtype not in ["most_increased", "most_decreased", "how_many_crossed",
                          "count_increased", "net_change", "highest_at_t2",
                          "rank_change_count", "max_percentage_change"]:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        pool = sub_rng.choice(_ITEM_POOLS)
        n = sub_rng.randint(*cfg["n_items"])
        n = min(n, len(pool))
        items = pool[:n]
        t1_name, t2_name = sub_rng.choice(_TIME_PAIRS)
        vals_t1 = [int(np_rng.randint(10, 90)) for _ in range(n)]
        vals_t2 = [int(np_rng.randint(10, 90)) for _ in range(n)]

        question, answer = self._make_qa(rng, qtype, items, vals_t1, vals_t2, t1_name, t2_name)
        if question is None:
            return None

        style = self._random_style()
        image = self._render(style, items, vals_t1, vals_t2, t1_name, t2_name)
        return question, answer, image

    def _render(self, style, items, vals_t1, vals_t2, t1_name, t2_name):
        n = len(items)
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(9 * style["figsize_scale"], max(5, n * 0.9) * style["figsize_scale"]))

        def _offset_labels(values):
            """Return y-offsets so that labels with close values don't overlap."""
            min_gap = 2.5  # minimum gap in data units between label centers
            indexed = sorted(enumerate(values), key=lambda x: x[1])
            offsets = [0.0] * len(values)
            for k in range(1, len(indexed)):
                prev_idx = indexed[k - 1][0]
                cur_idx = indexed[k][0]
                prev_y = values[prev_idx] + offsets[prev_idx]
                cur_y = values[cur_idx]
                if cur_y - prev_y < min_gap:
                    offsets[cur_idx] = prev_y + min_gap - cur_y
            return offsets

        offsets_t1 = _offset_labels(vals_t1)
        offsets_t2 = _offset_labels(vals_t2)

        for i in range(n):
            color = palette[i % len(palette)]
            ax.plot([0, 1], [vals_t1[i], vals_t2[i]], color=color,
                    linewidth=style["line_width"] + 0.5, marker="o", markersize=7)
            ax.text(-0.05, vals_t1[i] + offsets_t1[i], f"{items[i]} ({vals_t1[i]})",
                    ha="right", va="center", fontsize=style["font_size_base"] - 1,
                    color=color, fontfamily=style["font_family"])
            ax.text(1.05, vals_t2[i] + offsets_t2[i], f"{items[i]} ({vals_t2[i]})",
                    ha="left", va="center", fontsize=style["font_size_base"] - 1,
                    color=color, fontfamily=style["font_family"])

        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([t1_name, t2_name], fontsize=style["font_size_base"] + 1,
                           fontweight="bold")
        ax.set_ylabel(self._rng.choice(_Y_LABELS),
                      fontsize=style["font_size_base"])
        ax.set_title("Slope Chart", fontsize=style["font_size_base"] + 3)
        self._apply_style(fig, ax, style)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _make_qa(self, rng, qtype, items, v1, v2, t1n, t2n):
        n = len(items)
        changes = [v2[i] - v1[i] for i in range(n)]

        if qtype == "most_increased":
            idx = changes.index(max(changes))
            return (f"Which item increased the most from {t1n} to {t2n}?"), items[idx]

        elif qtype == "most_decreased":
            idx = changes.index(min(changes))
            return (f"Which item decreased the most from {t1n} to {t2n}?"), items[idx]

        elif qtype == "how_many_crossed":
            # Count how many lines cross each other (rank changes)
            ranks_t1 = sorted(range(n), key=lambda i: v1[i], reverse=True)
            ranks_t2 = sorted(range(n), key=lambda i: v2[i], reverse=True)
            r1 = {items[ranks_t1[k]]: k for k in range(n)}
            r2 = {items[ranks_t2[k]]: k for k in range(n)}
            rank_changes = sum(1 for item in items if r1[item] != r2[item])
            return (f"How many items changed their rank position from "
                    f"{t1n} to {t2n}?"), str(rank_changes)

        elif qtype == "count_increased":
            cnt = sum(1 for c in changes if c > 0)
            return f"How many items increased from {t1n} to {t2n}?", str(cnt)

        elif qtype == "net_change":
            idx = rng.randint(0, n - 1)
            return (f"What is the net change for '{items[idx]}' from "
                    f"{t1n} to {t2n}?"), str(changes[idx])

        elif qtype == "highest_at_t2":
            idx = v2.index(max(v2))
            return f"Which item has the highest value at {t2n}?", items[idx]

        elif qtype == "rank_change_count":
            # How many items changed their rank position (different from how_many_crossed)
            # This counts items whose ordinal rank changed
            ranks_t1 = sorted(range(n), key=lambda i: v1[i], reverse=True)
            ranks_t2 = sorted(range(n), key=lambda i: v2[i], reverse=True)
            r1 = {i: k for k, i in enumerate(ranks_t1)}
            r2 = {i: k for k, i in enumerate(ranks_t2)}
            count = sum(1 for i in range(n) if r1[i] != r2[i])
            return (f"How many items have a different rank position at "
                    f"{t2n} compared to {t1n}?"), str(count)

        elif qtype == "max_percentage_change":
            # Which item had the largest percentage change
            pct_changes = []
            for i in range(n):
                if v1[i] == 0:
                    pct_changes.append(0)
                else:
                    pct_changes.append(abs((v2[i] - v1[i]) / v1[i] * 100))
            idx = pct_changes.index(max(pct_changes))
            pct = round(pct_changes[idx], 1)
            return (f"Which item had the largest absolute percentage change "
                    f"from {t1n} to {t2n}, and what was it? "
                    f"Answer as 'ItemName, X.X%'."), f"{items[idx]}, {pct}%"

        else:
            abs_changes = [abs(c) for c in changes]
            idx = abs_changes.index(max(abs_changes))
            return "Which item shows the largest absolute change?", items[idx]
