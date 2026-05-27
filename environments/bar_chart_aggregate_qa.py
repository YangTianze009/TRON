"""
Bar chart aggregate reading: render a labeled bar chart with N=4-12 bars,
then ask the model to compute the mean / median / sum across all bars or a
filtered subset (e.g. "categories whose value exceeds T").

Tests visual extraction of a small numeric table + numeric aggregation.
Float output (one decimal place when fractional, integer otherwise).
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


_TEMPLATES_ALL_MEAN = [
    "What is the mean (average) of all bar values shown in the chart? Place the numeric answer in <answer>...</answer>.",
    "Compute the mean across every category in the bar chart. Numeric answer in <answer>...</answer>.",
    "Find the average value of all the bars shown. Numeric answer in <answer>...</answer>.",
    "Read every bar in the chart and report their mean. Numeric value in <answer>...</answer>.",
    "Calculate the average of the bar heights in the chart. Place the value in <answer>...</answer>.",
    "What is the arithmetic mean of all the bar values? Numeric answer in <answer>...</answer>.",
    "Average the values of every category in the chart. Numeric answer in <answer>...</answer>.",
    "Determine the mean of all bar heights. Numeric answer in <answer>...</answer>.",
    "Compute the average across all the categories shown in the chart. Numeric answer in <answer>...</answer>.",
    "Find the mean value of the bars displayed. Numeric answer in <answer>...</answer>.",
    "From the chart, what is the average across every bar? Numeric answer in <answer>...</answer>.",
    "Aggregate by mean: what is the average of all the bar values? Numeric answer in <answer>...</answer>.",
    "Tell me the mean of all category values in the bar chart. Numeric value in <answer>...</answer>.",
    "Average all bar heights in the chart and report the result. Numeric answer in <answer>...</answer>.",
    "What value do you get when averaging every bar in the chart? Numeric answer in <answer>...</answer>.",
    "Compute the simple average of all bars. Place numeric value in <answer>...</answer>.",
]

_TEMPLATES_ALL_MEDIAN = [
    "What is the median value across all bars in the chart? Place the numeric answer in <answer>...</answer>.",
    "Find the median of every bar value shown. Numeric answer in <answer>...</answer>.",
    "Report the median across all categories in the bar chart. Numeric answer in <answer>...</answer>.",
    "What is the median bar height in the chart? Numeric answer in <answer>...</answer>.",
    "Compute the median of all bar values. Numeric answer in <answer>...</answer>.",
    "Determine the median across every category. Numeric answer in <answer>...</answer>.",
    "From the bar chart, what is the median value? Numeric answer in <answer>...</answer>.",
    "Find the median of the bar heights shown. Numeric answer in <answer>...</answer>.",
    "Median of all the bars? Numeric answer in <answer>...</answer>.",
    "What is the middle value when every bar height is sorted? Numeric answer in <answer>...</answer>.",
    "Calculate the median across every bar. Numeric answer in <answer>...</answer>.",
    "Tell me the median of all bar values in the chart. Numeric answer in <answer>...</answer>.",
    "Aggregate by median: what is the median across all the bars? Numeric answer in <answer>...</answer>.",
    "Report the median bar height. Numeric answer in <answer>...</answer>.",
    "What is the median of the values of every category? Numeric answer in <answer>...</answer>.",
    "Find the central (median) value across every bar in the chart. Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_ALL_SUM = [
    "What is the sum of all bar values in the chart? Place the numeric answer in <answer>...</answer>.",
    "Compute the total across every bar shown. Numeric answer in <answer>...</answer>.",
    "Add up the values of all the bars in the chart. Numeric answer in <answer>...</answer>.",
    "What is the total of the bar heights? Numeric answer in <answer>...</answer>.",
    "Sum every bar value displayed in the chart. Numeric answer in <answer>...</answer>.",
    "Find the sum of all categories' values. Numeric answer in <answer>...</answer>.",
    "Total: add together every bar value. Numeric answer in <answer>...</answer>.",
    "What do you get when you add every bar height? Numeric answer in <answer>...</answer>.",
    "Compute the cumulative sum across all bars. Numeric answer in <answer>...</answer>.",
    "Add the values of all categories in the chart. Numeric answer in <answer>...</answer>.",
    "Sum every category's value. Numeric answer in <answer>...</answer>.",
    "Aggregate by sum: total of every bar height? Numeric answer in <answer>...</answer>.",
    "What is the grand total of the bar values? Numeric answer in <answer>...</answer>.",
    "Add up every value shown in the bar chart. Numeric answer in <answer>...</answer>.",
    "From the chart, compute the sum of all bar heights. Numeric answer in <answer>...</answer>.",
    "Tell me the total when adding every bar value. Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_FILTER_MEAN = [
    "Among the bars whose value is greater than {T}, what is the mean? Numeric answer in <answer>...</answer>.",
    "Compute the average of the bar values that exceed {T}. Numeric answer in <answer>...</answer>.",
    "Mean of bars with value > {T}? Numeric answer in <answer>...</answer>.",
    "Filter to bars exceeding {T} and report their mean. Numeric answer in <answer>...</answer>.",
    "Average value across only the bars that are above {T}? Numeric answer in <answer>...</answer>.",
    "Find the mean of all bars whose height is greater than {T}. Numeric answer in <answer>...</answer>.",
    "Among bars > {T}, compute the average. Numeric answer in <answer>...</answer>.",
    "What is the mean of the subset of bars exceeding {T}? Numeric answer in <answer>...</answer>.",
    "Compute the average of the bars whose value is above {T}. Numeric answer in <answer>...</answer>.",
    "Mean of all bars exceeding the threshold {T}? Numeric answer in <answer>...</answer>.",
    "Of the bars greater than {T}, what's their mean? Numeric answer in <answer>...</answer>.",
    "Calculate the average across every bar with value > {T}. Numeric answer in <answer>...</answer>.",
    "Among the categories whose value exceeds {T}, what is the mean value? Numeric answer in <answer>...</answer>.",
    "Filter bars by value > {T}, then compute their mean. Numeric answer in <answer>...</answer>.",
    "Mean of every bar whose height is greater than {T}? Numeric answer in <answer>...</answer>.",
    "Average of bars with value strictly greater than {T}? Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_FILTER_SUM = [
    "What is the sum of the bar values that are greater than {T}? Numeric answer in <answer>...</answer>.",
    "Compute the total of bars whose value exceeds {T}. Numeric answer in <answer>...</answer>.",
    "Sum of bars with value > {T}? Numeric answer in <answer>...</answer>.",
    "Add the values of every bar above {T}. Numeric answer in <answer>...</answer>.",
    "Sum the heights of bars exceeding the threshold {T}. Numeric answer in <answer>...</answer>.",
    "Total of the bars greater than {T}? Numeric answer in <answer>...</answer>.",
    "Add together every bar value that is above {T}. Numeric answer in <answer>...</answer>.",
    "Filter to bars > {T} and report their sum. Numeric answer in <answer>...</answer>.",
    "What is the cumulative total of bars exceeding {T}? Numeric answer in <answer>...</answer>.",
    "Sum every bar whose value is greater than {T}. Numeric answer in <answer>...</answer>.",
    "Aggregate (sum) across only the bars above {T}. Numeric answer in <answer>...</answer>.",
    "What total do you get adding every bar > {T}? Numeric answer in <answer>...</answer>.",
    "Among bars exceeding {T}, what is their sum? Numeric answer in <answer>...</answer>.",
    "Add up only the bars that are above {T}. Numeric answer in <answer>...</answer>.",
    "Sum the values of all categories whose bar exceeds {T}. Numeric answer in <answer>...</answer>.",
    "Compute the sum of all bars with height > {T}. Numeric answer in <answer>...</answer>.",
]


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches", "Pears", "Cherries", "Plums", "Kiwis", "Berries", "Limes"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil", "France", "UK", "Italy", "Spain", "Canada", "Mexico"],
    ["Sales", "Eng", "Mktg", "Finance", "HR", "Legal", "Support", "R&D", "Ops", "Design", "QA", "PR"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["NY", "LA", "CHI", "HOU", "PHX", "PHL", "DAL", "AUS", "SD", "SF", "BOS", "SEA"],
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Hours", "Sales", "Count"]


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class BarChartAggregateQA(StandaloneVisualEnv):
    ENV_NAME = "bar_chart_aggregate"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # n_cat: 4 → 12
        if level == 0:
            n_cat = 4
        elif level <= 2:
            n_cat = 5
        elif level <= 4:
            n_cat = 6 + (level - 3)  # 6, 7
        elif level <= 6:
            n_cat = 8
        elif level <= 8:
            n_cat = 10
        else:
            n_cat = 12
        # mode pool
        if level == 0:
            modes = ["all_mean"]   # forced trivial
        elif level <= 2:
            modes = ["all_mean", "all_sum"]
        elif level <= 5:
            modes = ["all_mean", "all_median", "all_sum"]
        else:
            modes = ["all_mean", "all_median", "all_sum",
                     "filter_mean", "filter_sum"]
        return {"level": level, "n_cat": n_cat, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1031 + level * 71 + 19)

        n = cfg["n_cat"]
        mode = rng.choice(cfg["modes"])
        cats_pool = rng.choice(_CATEGORY_POOLS)
        cats = rng.sample(cats_pool, n)
        y_label = rng.choice(_Y_LABELS)

        # Generate values. L0: round 10s for trivial mean. Higher: more spread.
        if level == 0:
            # 4 bars: 10/20/30/40 (mean = 25)
            base_vals = [10, 20, 30, 40]
            rng.shuffle(base_vals)
            values = base_vals
        elif level <= 3:
            # round multiples of 10
            values = [rng.randint(1, 10) * 10 for _ in range(n)]
        else:
            values = [rng.randint(10, 200) for _ in range(n)]

        if mode == "all_mean":
            ans = sum(values) / len(values)
            sidx = (self.seed or 0) % len(_TEMPLATES_ALL_MEAN)
            question = _TEMPLATES_ALL_MEAN[sidx]
        elif mode == "all_median":
            sorted_v = sorted(values)
            m = len(sorted_v)
            if m % 2 == 1:
                ans = float(sorted_v[m // 2])
            else:
                ans = (sorted_v[m // 2 - 1] + sorted_v[m // 2]) / 2.0
            sidx = (self.seed or 0) % len(_TEMPLATES_ALL_MEDIAN)
            question = _TEMPLATES_ALL_MEDIAN[sidx]
        elif mode == "all_sum":
            ans = float(sum(values))
            sidx = (self.seed or 0) % len(_TEMPLATES_ALL_SUM)
            question = _TEMPLATES_ALL_SUM[sidx]
        else:  # filter_mean / filter_sum
            # Pick a threshold T strictly between min and max so subset is non-trivial
            v_sorted = sorted(values)
            # Use median as threshold to roughly split
            T_candidates = [v for v in v_sorted if v_sorted[0] < v < v_sorted[-1]]
            if not T_candidates:
                # Fallback to min+1
                T = v_sorted[0] + 1
            else:
                T = rng.choice(T_candidates)
            subset = [v for v in values if v > T]
            if not subset:
                # Should not happen given T choice but be defensive
                T = v_sorted[0]
                subset = [v for v in values if v > T]
            if not subset:
                return None
            if mode == "filter_mean":
                ans = sum(subset) / len(subset)
                sidx = (self.seed or 0) % len(_TEMPLATES_FILTER_MEAN)
                question = _TEMPLATES_FILTER_MEAN[sidx].format(T=T)
            else:
                ans = float(sum(subset))
                sidx = (self.seed or 0) % len(_TEMPLATES_FILTER_SUM)
                question = _TEMPLATES_FILTER_SUM[sidx].format(T=T)

        answer = _format_num(ans)
        img = self._render_bars(cats, values, y_label, rng)

        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        n_opts = rng.choice([4, 5])
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts)
        return question, answer, img

    def _render_bars(self, cats: List[str], values: List[float], y_label: str,
                     rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        bg_color = "#ffffff"
        n = len(cats)
        # Ensure good readability: scale figure with N
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        bar_colors = [palette[i % len(palette)] for i in range(n)]
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
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
