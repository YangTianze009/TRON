"""
Filter-then-aggregate on a labeled bar chart with two attributes per item:
  - a numeric value V (the bar height)
  - a binary condition C (encoded by bar color OR by being above a visible
    horizontal red reference line)

The model must filter the items by C, then compute mean / sum / max / min of
V on the filtered subset.

Mirrors reference Factoid IDX 981 ("sum of predicted number of customers
churn ... categories that have a higher churn rate than the red line"),
IDX 1923 ("maximum value among the light blue bars"), and similar
filter+aggregate patterns.

Output: a non-negative number (integer if whole, else `.1f`).
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
from ._mcq_letter_lib import maybe_to_mcq_letter


_TEMPLATES_LINE_MEAN = [
    "What is the mean (average) value across the bars that are above the red dashed line? Numeric answer in <answer>...</answer>.",
    "Compute the average of the bars whose value exceeds the red reference line. Numeric in <answer>...</answer>.",
    "Among bars taller than the red line, what is the mean? Numeric answer in <answer>...</answer>.",
    "Average of the bars that lie above the red dashed reference? Numeric in <answer>...</answer>.",
    "Find the mean value of the bars exceeding the red horizontal line. Numeric answer in <answer>...</answer>.",
    "Filter to the bars above the red dashed line and report their mean. Numeric in <answer>...</answer>.",
    "What is the average of the bars whose height is greater than the red reference line? Numeric in <answer>...</answer>.",
    "Mean of bars above the red line in the chart? Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_LINE_SUM = [
    "What is the sum of the bar values that are above the red dashed line? Numeric answer in <answer>...</answer>.",
    "Compute the total of the bars whose value exceeds the red reference line. Numeric in <answer>...</answer>.",
    "Among bars taller than the red line, what is the sum? Numeric answer in <answer>...</answer>.",
    "Sum the values of the bars that lie above the red dashed reference. Numeric in <answer>...</answer>.",
    "Add up the values of every bar whose height is greater than the red horizontal line. Numeric in <answer>...</answer>.",
    "Filter to the bars above the red dashed line and report their sum. Numeric in <answer>...</answer>.",
    "What total do you get when adding every bar above the red reference line? Numeric in <answer>...</answer>.",
    "Sum of the bars exceeding the red line in the chart? Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_COLOR_MEAN = [
    "What is the mean value among the {color} bars? Numeric answer in <answer>...</answer>.",
    "Compute the average of only the {color} bars in the chart. Numeric in <answer>...</answer>.",
    "Mean of the {color} bars? Numeric answer in <answer>...</answer>.",
    "Average value across the bars colored {color}? Numeric in <answer>...</answer>.",
    "Find the mean of the values for the {color} bars only. Numeric answer in <answer>...</answer>.",
    "Among the {color} bars, what is the average? Numeric answer in <answer>...</answer>.",
    "Filter to the {color} bars and report their mean. Numeric answer in <answer>...</answer>.",
    "What is the arithmetic mean of the {color} bars in the chart? Numeric in <answer>...</answer>.",
]

_TEMPLATES_COLOR_SUM = [
    "What is the sum of the {color} bar values? Numeric answer in <answer>...</answer>.",
    "Compute the total of only the {color} bars in the chart. Numeric in <answer>...</answer>.",
    "Sum of the {color} bars? Numeric answer in <answer>...</answer>.",
    "Add up the values of every {color} bar in the chart. Numeric in <answer>...</answer>.",
    "Find the sum of the values for the {color} bars only. Numeric answer in <answer>...</answer>.",
    "Among the {color} bars, what is the total? Numeric answer in <answer>...</answer>.",
    "Filter to the {color} bars and report their sum. Numeric answer in <answer>...</answer>.",
    "What is the sum of the {color} bars in the chart? Numeric in <answer>...</answer>.",
]

_TEMPLATES_COLOR_MAX = [
    "What is the maximum value among the {color} bars? Numeric answer in <answer>...</answer>.",
    "Find the largest {color} bar value. Numeric answer in <answer>...</answer>.",
    "Highest value across the {color} bars? Numeric in <answer>...</answer>.",
    "Among the {color} bars, what is the max? Numeric answer in <answer>...</answer>.",
    "What is the tallest {color} bar's value? Numeric answer in <answer>...</answer>.",
    "Maximum value of the {color} bars in the chart? Numeric in <answer>...</answer>.",
    "Report the maximum bar value among the {color} bars. Numeric in <answer>...</answer>.",
    "Find the highest value among the {color} bars. Numeric answer in <answer>...</answer>.",
]


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches", "Pears", "Cherries", "Plums", "Kiwis", "Berries", "Limes"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil", "France", "UK", "Italy", "Spain", "Canada", "Mexico"],
    ["Sales", "Eng", "Mktg", "Finance", "HR", "Legal", "Support", "R&D", "Ops", "Design", "QA", "PR"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers",
             "Hours", "Sales", "Count"]

# Color name → matplotlib hex used at render time. The model sees the named
# color in the question; we render with this exact hex so the visual answer
# is unambiguous.
_NAMED_COLORS = [
    ("blue", "#3498db"),
    ("orange", "#e67e22"),
    ("green", "#27ae60"),
    ("purple", "#8e44ad"),
]


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class ChartFilterAggregateQA(StandaloneVisualEnv):
    ENV_NAME = "chart_filter_aggregate"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_cat = 4
        elif level <= 2:
            n_cat = 5
        elif level <= 4:
            n_cat = 6 + (level - 3)  # 7, 8
        elif level <= 6:
            n_cat = 8
        elif level <= 8:
            n_cat = 10
        else:
            n_cat = 12
        # Filter kind + aggregate
        if level == 0:
            kinds = [("color", "mean")]
        elif level <= 2:
            kinds = [("color", "mean"), ("color", "sum")]
        elif level <= 5:
            kinds = [("color", "mean"), ("color", "sum"), ("color", "max"),
                     ("line", "mean")]
        else:
            kinds = [("color", "mean"), ("color", "sum"), ("color", "max"),
                     ("line", "mean"), ("line", "sum")]
        return {"level": level, "n_cat": n_cat, "kinds": kinds}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1097 + level * 71 + 23)

        n = cfg["n_cat"]
        filter_kind, agg = rng.choice(cfg["kinds"])
        cats_pool = rng.choice(_CATEGORY_POOLS)
        cats = rng.sample(cats_pool, n)
        y_label = rng.choice(_Y_LABELS)

        # L0 simple: 4 bars with random integer values 10-50 (multiples of 5),
        # randomized 2 blue + 2 orange assignment, randomized target color.
        # Avoids the prior memorize-able fixed answer (always 30).
        if level == 0:
            values = [rng.choice([10, 15, 20, 25, 30, 35, 40, 45, 50]) for _ in range(4)]
            color_pool = ["orange", "blue"] * 2
            rng.shuffle(color_pool)
            colors_assigned = color_pool
            target_color = rng.choice(["orange", "blue"])
            subset = [v for v, c in zip(values, colors_assigned) if c == target_color]
            if not subset:  # all 4 same color (shouldn't happen with 2+2 split, defensive)
                target_color = "orange" if target_color == "blue" else "blue"
                subset = [v for v, c in zip(values, colors_assigned) if c == target_color]
            ans = sum(subset) / len(subset)
            sidx = (self.seed or 0) % len(_TEMPLATES_COLOR_MEAN)
            question = _TEMPLATES_COLOR_MEAN[sidx].format(color=target_color)
            answer = _format_num(ans)
            img = self._render_color_bars(cats, values, colors_assigned, y_label, rng)
            n_opts = rng.choice([4, 5])
            question, answer = maybe_to_mcq_letter(
                question, answer, rng, prob=0.5, n_options=n_opts)
            return question, answer, img

        # General path
        if level <= 3:
            v_min, v_max, step = 10, 100, 5
        elif level <= 6:
            v_min, v_max, step = 10, 200, 5
        else:
            v_min, v_max, step = 5, 250, 1

        values = [rng.choice(list(range(v_min, v_max + 1, step))) for _ in range(n)]

        if filter_kind == "color":
            # Pick 2 named colors (one is the target); each bar gets one of them.
            two = rng.sample(_NAMED_COLORS, 2)
            target_name, target_hex = two[0]
            other_name, other_hex = two[1]
            # Assign at least 2 to target color and at least 1 to other
            for _ in range(40):
                colors_assigned = [rng.choice([target_name, other_name]) for _ in range(n)]
                tgt_count = sum(1 for c in colors_assigned if c == target_name)
                if 2 <= tgt_count <= n - 1:
                    break
            else:
                return None
            subset = [v for v, c in zip(values, colors_assigned) if c == target_name]
            if not subset:
                return None
            if agg == "mean":
                ans = sum(subset) / len(subset)
                tpl = _TEMPLATES_COLOR_MEAN
            elif agg == "sum":
                ans = float(sum(subset))
                tpl = _TEMPLATES_COLOR_SUM
            elif agg == "max":
                ans = float(max(subset))
                tpl = _TEMPLATES_COLOR_MAX
            else:
                return None
            sidx = (self.seed or 0) % len(tpl)
            question = tpl[sidx].format(color=target_name)
            answer = _format_num(ans)
            img = self._render_color_bars(cats, values, colors_assigned, y_label, rng,
                                          target_color_name=target_name,
                                          target_color_hex=target_hex,
                                          other_color_hex=other_hex)
            n_opts = rng.choice([4, 5])
            question, answer = maybe_to_mcq_letter(
                question, answer, rng, prob=0.5, n_options=n_opts)
            return question, answer, img

        # filter_kind == "line": render an axhline at threshold T; aggregate over
        # the bars strictly above T.
        sorted_v = sorted(set(values))
        if len(sorted_v) < 2:
            return None
        # Pick T strictly between two distinct values so above/below is unambiguous
        i = rng.randrange(len(sorted_v) - 1)
        lo, hi = sorted_v[i], sorted_v[i + 1]
        T = int(round((lo + hi) / 2.0))
        # Avoid T equaling any bar value
        if T in values:
            for delta in (1, -1, 2, -2):
                cand = T + delta
                if cand not in values and lo < cand < hi:
                    T = cand
                    break
        if T in values:
            return None
        subset = [v for v in values if v > T]
        if not subset or len(subset) == n:
            return None
        if agg == "mean":
            ans = sum(subset) / len(subset)
            tpl = _TEMPLATES_LINE_MEAN
        else:  # sum
            ans = float(sum(subset))
            tpl = _TEMPLATES_LINE_SUM
        sidx = (self.seed or 0) % len(tpl)
        question = tpl[sidx]
        answer = _format_num(ans)
        img = self._render_threshold_bars(cats, values, y_label, T, rng)
        n_opts = rng.choice([4, 5])
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts)
        return question, answer, img

    def _render_color_bars(self, cats, values, colors_assigned, y_label, rng,
                           target_color_name="blue",
                           target_color_hex=None,
                           other_color_hex=None):
        # Default L0 hex map
        if target_color_hex is None:
            cmap = {"blue": "#3498db", "orange": "#e67e22", "green": "#27ae60",
                    "purple": "#8e44ad"}
            bar_colors = [cmap[c] for c in colors_assigned]
        else:
            cmap = {target_color_name: target_color_hex}
            other_name = next(c for c in colors_assigned if c != target_color_name)
            cmap[other_name] = other_color_hex
            bar_colors = [cmap[c] for c in colors_assigned]

        n = len(cats)
        bg = "#ffffff"
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        bars = ax.bar(range(n), values, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.015,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.18 + 2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _render_threshold_bars(self, cats, values, y_label, T, rng):
        # Bars all share a neutral palette; the red dashed line is the filter.
        palette = rng.choice(self._COLOR_PALETTES)
        bar_colors = [palette[i % len(palette)] for i in range(len(cats))]
        n = len(cats)
        bg = "#ffffff"
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        bars = ax.bar(range(n), values, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(values + [T])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.015,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        # Prominent red dashed reference line
        ax.axhline(T, color="#c0392b", linewidth=2.0, linestyle="--",
                   label=f"y = {T}")
        ax.legend(fontsize=10, loc="best")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
