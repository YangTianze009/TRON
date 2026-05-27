"""
Chart No Labels QA environment.

Bar/line charts with NO value annotations, NO gridlines. Only axis ticks
and scale visible. Model must ESTIMATE values from axis position.
Targets: chart-reading Human (72.2%) — "reading from axis" skill.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Data pools
# ------------------------------------------------------------------ #

_CATEGORY_POOLS = [
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
     "Iota", "Kappa", "Lambda", "Mu"],
    ["Region A", "Region B", "Region C", "Region D", "Region E",
     "Region F", "Region G", "Region H", "Region I", "Region J",
     "Region K", "Region L"],
    ["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021",
     "2022", "2023", "2024", "2025"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
     "Sep", "Oct", "Nov", "Dec"],
    ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5",
     "Team 6", "Team 7", "Team 8", "Team 9", "Team 10",
     "Team 11", "Team 12"],
    ["Store A", "Store B", "Store C", "Store D", "Store E", "Store F",
     "Store G", "Store H", "Store I", "Store J", "Store K", "Store L"],
]

_Y_LABELS = [
    "Revenue ($M)", "Temperature (°C)", "Population (K)",
    "Score", "Growth (%)", "Quantity", "Distance (km)",
    "Weight (kg)", "Volume (L)", "Time (min)",
]

_COLOR_SCHEMES = [
    ("#2C3E50", "#ECF0F1"),  # dark bars, light bg
    ("#1A5276", "#EBF5FB"),
    ("#7B241C", "#FDEDEC"),
    ("#1E8449", "#EAFAF1"),
    ("#6C3483", "#F4ECF7"),
]

class ChartNoLabelsQA(StandaloneVisualEnv):
    ENV_NAME = "chart_no_labels"

    QUESTION_TYPES = [
        "estimate_value", "estimate_difference", "estimate_ratio",
        "estimate_total", "which_exceeds_threshold",
        "estimate_pct_of_total", "estimate_rank_distance",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L3 previously spiked to 0.95 because "which_exceeds_threshold" is
        # a nearly-free counting task with sparse correct answers. Move it
        # to L0-1 (where it belongs — it's the easiest qtype) and make L3
        # keep ratio/total on slightly larger charts.
        # Iter 3 (2026-04-17): previous schedule 0.70/0.60/0.75/0.65 was
        # bumpy. L3's estimate_ratio on int values (e.g. 36/74 ≈ 0.49) was
        # harder than L6's estimate_total. Push ratio out of L3 into L5+.
        # Use estimate_value (single bar read-out) at L2-3 to smooth the
        # early curve.
        if level <= 1:
            return {"qtypes": ["which_exceeds_threshold", "estimate_difference"],
                    "n_cats": (5, 6), "sparse_y_ticks": False}
        if level <= 3:
            return {"qtypes": ["estimate_value", "estimate_difference"],
                    "n_cats": (6, 7), "sparse_y_ticks": False}
        if level <= 5:
            return {"qtypes": ["estimate_ratio", "estimate_total"],
                    "n_cats": (7, 8), "sparse_y_ticks": True}
        if level <= 7:
            return {"qtypes": ["estimate_total", "estimate_pct_of_total",
                               "estimate_rank_distance"],
                    "n_cats": (8, 10), "sparse_y_ticks": True,
                    "rotate_labels": True}
        return {"qtypes": ["estimate_pct_of_total", "estimate_rank_distance"],
                "n_cats": (9, 12), "rotate_labels": True,
                "sparse_y_ticks": True, "hide_y_axis": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        np_rng = np.random.RandomState(seed)
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = rng.choice(cfg["qtypes"])

        chart_type = parameter.get("chart_type", rng.choice(["bar", "line"]))

        pool = rng.choice(_CATEGORY_POOLS)
        cats = list(pool)
        rng.shuffle(cats)
        lo_c, hi_c = cfg["n_cats"]
        num_categories = rng.randint(lo_c, min(hi_c, len(cats)))
        categories = cats[:num_categories]

        # Use NON-round numbers to make estimation harder
        base = rng.choice([13, 27, 38, 42, 56, 63, 74, 87])
        spread = rng.randint(20, 80)
        values = [base + int(np_rng.randint(0, spread + 1)) for _ in range(num_categories)]
        # Add some decimal-ish values by adjusting
        values = [v + rng.choice([-3, -1, 0, 2, 4]) for v in values]

        question, answer = self._make_qa(rng, qtype, categories, values)
        if question is None:
            return None

        image = self._render(rng, categories, values, chart_type, cfg)
        return question, answer, image

    def _make_qa(self, rng, qtype, categories, values):
        n = len(categories)

        if qtype == "estimate_value":
            idx = rng.randint(0, n - 1)
            q = (f"Estimate the value of '{categories[idx]}' by reading "
                 f"from the y-axis. No value labels are shown. "
                 f"Give your best integer estimate.")
            return q, str(values[idx])

        elif qtype == "estimate_difference":
            i, j = rng.sample(range(n), 2)
            diff = abs(values[i] - values[j])
            q = (f"Estimate the absolute difference between "
                 f"'{categories[i]}' and '{categories[j]}' by reading "
                 f"from the y-axis. Give your best integer estimate.")
            return q, str(diff)

        elif qtype == "estimate_ratio":
            i, j = rng.sample(range(n), 2)
            if values[j] == 0:
                return None, None
            ratio = round(values[i] / values[j], 2)
            q = (f"Estimate the ratio of '{categories[i]}' to "
                 f"'{categories[j]}' from the chart. Round to 2 decimal places.")
            return q, str(ratio)

        elif qtype == "estimate_total":
            total = sum(values)
            q = ("Estimate the total sum of ALL values shown in the chart "
                 "by reading each from the y-axis. Give your best integer estimate.")
            return q, str(total)

        elif qtype == "which_exceeds_threshold":
            # Pick a threshold that some but not all exceed
            sorted_v = sorted(values)
            mid = sorted_v[n // 2]
            threshold = mid + rng.randint(-5, 5)
            exceeding = [categories[i] for i in range(n) if values[i] > threshold]
            count = len(exceeding)
            if count == 0 or count == n:
                # Adjust threshold
                threshold = sorted_v[n // 3]
                exceeding = [categories[i] for i in range(n) if values[i] > threshold]
                count = len(exceeding)
            q = (f"How many categories have values that exceed {threshold}? "
                 f"Estimate each value from the y-axis and count.")
            return q, str(count)

        elif qtype == "estimate_pct_of_total":
            idx = rng.randint(0, n - 1)
            total = sum(values)
            if total == 0:
                return None, None
            pct = round(values[idx] / total * 100, 1)
            q = (f"Estimate what percentage of the total sum the category "
                 f"'{categories[idx]}' represents. No value labels are shown. "
                 f"Round to 1 decimal place.")
            return q, str(pct)

        elif qtype == "estimate_rank_distance":
            # Ask for the difference between the 2nd and 3rd largest
            if n < 3:
                return None, None
            sorted_vals = sorted(values, reverse=True)
            diff = sorted_vals[1] - sorted_vals[2]
            q = ("Estimate all values from the y-axis. What is the absolute "
                 "difference between the 2nd largest and 3rd largest values?")
            return q, str(diff)

        return None, None

    # ------------------------------------------------------------------ #
    # Rendering — minimal chart, NO labels, NO gridlines
    # ------------------------------------------------------------------ #

    def _render(self, rng, categories, values, chart_type, cfg=None) -> Image.Image:
        if cfg is None:
            cfg = {}
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = style["palette"]
        fs = style["font_size_base"]
        lw = style["line_width"]
        bar_color = palette[0]
        y_label = rng.choice(_Y_LABELS)

        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 0.9 + 1) * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        x = np.arange(len(categories))

        if chart_type == "bar":
            ax.bar(x, values, width=0.55, color=bar_color,
                   edgecolor="white", linewidth=0.8, alpha=0.85)
        else:
            ax.plot(x, values, "o-", color=bar_color, linewidth=lw,
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5)

        ax.set_xticks(x)
        rotate = len(categories) > 6 or max(len(c) for c in categories) > 7
        ax.set_xticklabels(categories, rotation=45 if rotate else 0,
                           ha="right" if rotate else "center", fontsize=fs - 1)
        ax.set_ylabel(y_label, fontsize=fs + 1)

        # Y-axis: show ticks but NO gridlines; fewer ticks at high levels
        if cfg.get("sparse_y_ticks"):
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis="y", labelsize=fs - 1)

        # Remove grid entirely
        ax.grid(False)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(lw)
        ax.spines["left"].set_linewidth(lw)

        # Title
        title = rng.choice([
            f"{y_label} Overview",
            f"{y_label} by Category",
            f"Data Summary: {y_label}",
        ])
        ax.set_title(title, fontsize=fs + 3, fontweight="bold", pad=10)

        # NO value annotations, NO gridlines — purely axis-based reading
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
