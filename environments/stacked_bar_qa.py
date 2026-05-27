"""
Stacked bar chart QA — decomposition reasoning on stacked/grouped bars.
Targets: chart-reading (stacked bar subset), statistical reasoning.

Capabilities: V3 (chart extraction), R1 (arithmetic), R4 (statistical reasoning)
8-level difficulty: from reading segment values to computing cross-category ratios.
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
from ._mcq_letter_helper import maybe_mcq_letter_wrap

class StackedBarQA(StandaloneVisualEnv):
    ENV_NAME = "stacked_bar"

    _CATEGORY_POOLS = [
        ["Q1", "Q2", "Q3", "Q4"],
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        ["2019", "2020", "2021", "2022", "2023"],
        ["North", "South", "East", "West"],
        ["Team A", "Team B", "Team C", "Team D", "Team E"],
        ["Mon", "Tue", "Wed", "Thu", "Fri"],
    ]
    _SERIES_POOLS = [
        ["Product A", "Product B", "Product C"],
        ["Revenue", "Cost", "Profit"],
        ["Online", "In-Store", "Wholesale"],
        ["Domestic", "International"],
        ["Fixed", "Variable", "Other"],
    ]
    _Y_LABELS = ["Sales ($K)", "Revenue ($M)", "Units", "Count", "Value", "Score"]

    def _level_config(self, level: int) -> Dict:
        # Difficulty redesign 2026-04-14. L0 = total_height + difference (was L3-L4).
        if level <= 0:
            return {"qtypes": ["total_height", "segment_difference"],
                    "qweights": [5, 5], "n_cat": (4, 5), "n_ser": (2, 3), "show_vals": None}
        if level <= 2:
            return {"qtypes": ["segment_difference", "compare_totals", "segment_percentage"],
                    "qweights": [3, 4, 3], "n_cat": (4, 6), "n_ser": (2, 3), "show_vals": False}
        if level <= 4:
            return {"qtypes": ["segment_percentage", "compare_totals", "rank_by_segment"],
                    "qweights": [3, 3, 4], "n_cat": (4, 6), "n_ser": (2, 4), "show_vals": False}
        if level <= 6:
            return {"qtypes": ["segment_percentage", "segment_difference", "compare_totals", "rank_by_segment"],
                    "qweights": [3, 3, 2, 2], "n_cat": (4, 6), "n_ser": (2, 4), "show_vals": False}
        if level <= 8:
            return {"qtypes": ["segment_ratio", "segment_percentage",
                                "percentage_of_total_stack",
                                "cross_category_segment_ratio",
                                "series_total_across_categories"],
                    "qweights": [2, 2, 3, 3, 3], "n_cat": (5, 7), "n_ser": (3, 4), "show_vals": False}
        # L9: hardest — always multi-series/multi-category reasoning.
        return {"qtypes": ["cross_category_segment_ratio",
                           "series_total_across_categories",
                           "percentage_of_total_stack"],
                "qweights": [4, 4, 2], "n_cat": (5, 7), "n_ser": (3, 4), "show_vals": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 6701)

        question_type = parameter.get("question_type")
        if question_type not in ["segment_value", "total_height", "segment_difference",
                                   "segment_percentage", "largest_segment", "compare_totals",
                                   "segment_ratio", "rank_by_segment", "percentage_of_total_stack",
                                   "cross_category_segment_ratio",
                                   "series_total_across_categories"]:
            question_type = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        n_categories = sub_rng.randint(*cfg["n_cat"])
        n_series = sub_rng.randint(*cfg["n_ser"])
        show_values = cfg["show_vals"]
        if show_values is None:
            show_values = sub_rng.choice([True, False])

        # Generate data
        categories = sub_rng.choice(self._CATEGORY_POOLS)[:n_categories]
        series_names = sub_rng.choice(self._SERIES_POOLS)[:n_series]
        # Values: each series has values for each category
        values = {}
        for s in series_names:
            values[s] = [rng.randint(5, 60) for _ in categories]

        # Generate question + answer
        question, answer = self._make_qa(
            rng, question_type, categories, series_names, values
        )
        if question is None:
            return None

        # Render
        style = self._random_style()
        image = self._render_chart(
            rng, categories, series_names, values, style, show_values
        )
        y_label = rng.choice(self._Y_LABELS)
        title = f"{y_label} by Category"

        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], image
        return question, str(answer), image

    def _make_qa(self, rng, question_type, categories, series_names, values):
        if question_type == "segment_value":
            cat = rng.choice(categories)
            ser = rng.choice(series_names)
            ci = categories.index(cat)
            val = values[ser][ci]
            return (
                f"In the stacked bar chart, what is the value of '{ser}' for '{cat}'?",
                round(val, 1)
            )
        elif question_type == "total_height":
            cat = rng.choice(categories)
            ci = categories.index(cat)
            total = sum(values[s][ci] for s in series_names)
            return (
                f"What is the total height (sum of all segments) for '{cat}'?",
                round(total, 1)
            )
        elif question_type == "segment_difference":
            s1, s2 = rng.sample(series_names, 2)
            cat = rng.choice(categories)
            ci = categories.index(cat)
            diff = abs(values[s1][ci] - values[s2][ci])
            return (
                f"For '{cat}', what is the absolute difference between '{s1}' and '{s2}'?",
                round(diff, 1)
            )
        elif question_type == "segment_percentage":
            cat = rng.choice(categories)
            ser = rng.choice(series_names)
            ci = categories.index(cat)
            total = sum(values[s][ci] for s in series_names)
            if total == 0:
                return None, None
            pct = values[ser][ci] / total * 100
            return (
                f"For '{cat}', what percentage of the total does '{ser}' represent? "
                f"Round to 1 decimal place.",
                round(pct, 1)
            )
        elif question_type == "largest_segment":
            cat = rng.choice(categories)
            ci = categories.index(cat)
            max_s = max(series_names, key=lambda s: values[s][ci])
            return (
                f"For '{cat}', which segment (series) has the largest value?",
                max_s
            )
        elif question_type == "compare_totals":
            c1, c2 = rng.sample(categories, 2)
            ci1, ci2 = categories.index(c1), categories.index(c2)
            t1 = sum(values[s][ci1] for s in series_names)
            t2 = sum(values[s][ci2] for s in series_names)
            larger = c1 if t1 > t2 else c2
            return (
                f"Which category has a larger total: '{c1}' or '{c2}'?",
                larger
            )
        elif question_type == "segment_ratio":
            s1, s2 = rng.sample(series_names, 2)
            cat = rng.choice(categories)
            ci = categories.index(cat)
            if values[s2][ci] == 0:
                return None, None
            ratio = values[s1][ci] / values[s2][ci]
            return (
                f"For '{cat}', what is the ratio of '{s1}' to '{s2}'? Round to 2 decimal places.",
                round(ratio, 2)
            )
        elif question_type == "rank_by_segment":
            ser = rng.choice(series_names)
            vals = [(categories[i], values[ser][i]) for i in range(len(categories))]
            vals.sort(key=lambda x: x[1], reverse=True)
            top = vals[0][0]
            return (
                f"Rank the categories by '{ser}' value. Which category has the highest '{ser}'?",
                top
            )
        elif question_type == "percentage_of_total_stack":
            # What % of ALL stacks combined does segment X in bar Y represent?
            cat = rng.choice(categories)
            ser = rng.choice(series_names)
            ci = categories.index(cat)
            seg_val = values[ser][ci]
            # Total across ALL categories and ALL series
            grand_total = sum(values[s][i] for s in series_names for i in range(len(categories)))
            if grand_total == 0:
                return None, None
            pct = round(seg_val / grand_total * 100, 2)
            return (
                f"What percentage of the GRAND TOTAL (sum of all segments across all bars) "
                f"does '{ser}' in '{cat}' represent? Round to 2 decimal places.",
                round(pct, 2)
            )
        elif question_type == "cross_category_segment_ratio":
            # Ratio of series1 in cat1 to series2 in cat2 (two-axis lookup).
            s1, s2 = rng.sample(series_names, 2)
            c1, c2 = rng.sample(categories, 2)
            ci1, ci2 = categories.index(c1), categories.index(c2)
            v1 = values[s1][ci1]
            v2 = values[s2][ci2]
            if v2 == 0:
                return None, None
            ratio = v1 / v2
            return (
                f"What is the ratio of '{s1}' in '{c1}' to '{s2}' in '{c2}'? "
                f"Round to 2 decimal places.",
                round(ratio, 2)
            )
        elif question_type == "series_total_across_categories":
            # Sum of one series across a subset of two categories minus the
            # same series in a third category. Forces multi-read + arithmetic.
            if len(categories) < 3 or len(series_names) < 2:
                return None, None
            ser = rng.choice(series_names)
            c1, c2, c3 = rng.sample(categories, 3)
            ci1, ci2, ci3 = (categories.index(c1), categories.index(c2),
                             categories.index(c3))
            total_pair = values[ser][ci1] + values[ser][ci2]
            subtract = values[ser][ci3]
            result = total_pair - subtract
            return (
                f"Compute the value of (sum of '{ser}' across '{c1}' and "
                f"'{c2}') minus ('{ser}' in '{c3}'). Answer with a single "
                f"integer.",
                result
            )
        return None, None

    def _render_chart(self, rng, categories, series_names, values, style, show_values):
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(rng.choice([(7, 5), (8, 5), (9, 6)])))
        self._apply_style(fig, ax, style)

        x = np.arange(len(categories))
        width = 0.6
        bottom = np.zeros(len(categories))

        for i, s in enumerate(series_names):
            color = palette[i % len(palette)]
            vals = values[s]
            bars = ax.bar(x, vals, width, bottom=bottom, label=s, color=color,
                         edgecolor="white", linewidth=0.5)
            if show_values:
                for j, (b, v) in enumerate(zip(bars, vals)):
                    if v > 3:
                        ax.text(b.get_x() + b.get_width() / 2, bottom[j] + v / 2,
                               str(int(v)), ha="center", va="center",
                               fontsize=8, color="white", fontweight="bold")
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=style["font_size_base"])
        ax.legend(fontsize=style["font_size_base"] - 1)
        ax.set_ylabel("Value", fontsize=style["font_size_base"])

        return self.fig_to_pil(fig, dpi=style["dpi"])
