"""
Chart Multi-Step QA environment.

Generates bar/line charts with realistic data. Questions require 3-5 step
reasoning chains. NO value labels on bars — model must estimate from axis.
Targets: chart-reading Human (72.2%).
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
# Data generation helpers
# ------------------------------------------------------------------ #

_CATEGORY_POOLS = {
    "companies": ["TechCorp", "DataSoft", "CloudInc", "NetPro", "InfoSys",
                   "DigiTech", "WebFlow", "AppDev", "CyberNet", "SmartAI"],
    "countries": ["USA", "China", "Germany", "Japan", "India", "Brazil",
                  "France", "UK", "Canada", "Australia"],
    "products": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon",
                 "Zeta", "Eta", "Theta", "Iota", "Kappa"],
    "departments": ["Sales", "Engineering", "Marketing", "Finance", "HR",
                    "R&D", "Operations", "Support", "Legal", "Design"],
}

_Y_LABELS = [
    "Revenue ($M)", "Units (thousands)", "Employees", "Score",
    "Budget ($K)", "Production (tons)", "Visitors (K)", "Rating",
]

class ChartMultistepQA(StandaloneVisualEnv):
    ENV_NAME = "chart_multistep"

    QUESTION_TYPES = [
        "filter_then_average", "rank_then_compute",
        "conditional_growth", "ratio_chain",
        # Ultra-hard types
        "weighted_rank_score", "outlier_adjusted_mean",
        "percentile_gap",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"qtypes": ["filter_then_average", "rank_then_compute"],
                    "n_cats": (5, 7)}
        if level <= 5:
            return {"qtypes": ["filter_then_average", "rank_then_compute",
                               "conditional_growth", "ratio_chain"],
                    "n_cats": (6, 8)}
        if level <= 7:
            return {"qtypes": ["ratio_chain", "weighted_rank_score",
                               "outlier_adjusted_mean"],
                    "n_cats": (7, 9)}
        return {"qtypes": ["weighted_rank_score", "outlier_adjusted_mean",
                           "percentile_gap"],
                "n_cats": (8, 10)}

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
        lo_c, hi_c = cfg["n_cats"]
        num_categories = rng.randint(lo_c, hi_c)

        pool_name = rng.choice(list(_CATEGORY_POOLS.keys()))
        pool = list(_CATEGORY_POOLS[pool_name])
        rng.shuffle(pool)
        categories = pool[:num_categories]

        # For growth-type questions, generate 2-year data
        if qtype == "conditional_growth":
            values_2022 = [int(np_rng.randint(30, 200)) for _ in range(num_categories)]
            # Growth factors between 0.8 and 1.5
            growth = [round(0.8 + np_rng.random() * 0.7, 2) for _ in range(num_categories)]
            values_2023 = [int(round(v * g)) for v, g in zip(values_2022, growth)]
            return self._qa_conditional_growth(
                rng, categories, values_2022, values_2023, growth,
                chart_type, pool_name
            )

        # Single series data
        low = rng.choice([10, 20, 50, 100])
        high = low * rng.randint(3, 8)
        values = [int(np_rng.randint(low, high + 1)) for _ in range(num_categories)]

        if qtype == "filter_then_average":
            return self._qa_filter_then_average(
                rng, categories, values, chart_type, pool_name
            )
        elif qtype == "rank_then_compute":
            return self._qa_rank_then_compute(
                rng, categories, values, chart_type, pool_name
            )
        elif qtype == "ratio_chain":
            return self._qa_ratio_chain(
                rng, categories, values, chart_type, pool_name
            )
        elif qtype == "weighted_rank_score":
            return self._qa_weighted_rank_score(
                rng, categories, values, chart_type, pool_name
            )
        elif qtype == "outlier_adjusted_mean":
            return self._qa_outlier_adjusted_mean(
                rng, categories, values, chart_type, pool_name
            )
        elif qtype == "percentile_gap":
            return self._qa_percentile_gap(
                rng, categories, values, chart_type, pool_name
            )

        return None

    # ------------------------------------------------------------------ #
    # Question types
    # ------------------------------------------------------------------ #

    def _qa_filter_then_average(self, rng, categories, values, chart_type, pool):
        # Steps: 1) read all values, 2) compute median, 3) filter > median,
        #        4) compute average of filtered, 5) round
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        else:
            median = sorted_vals[n // 2]
        above = [v for v in values if v > median]
        if len(above) == 0:
            return None
        avg = round(sum(above) / len(above), 1)

        q = ("Read all the values from the chart (no labels are shown on bars). "
             "Compute the median of all values. Then find the average of only "
             "the values that exceed the median. Round to 1 decimal place.")

        image = self._render_single(rng, categories, values, chart_type, pool)
        return q, str(avg), image

    def _qa_rank_then_compute(self, rng, categories, values, chart_type, pool):
        # Steps: 1) read all, 2) sort descending, 3) top-3 sum,
        #        4) bottom-3 sum, 5) difference
        sorted_vals = sorted(values, reverse=True)
        top3 = sum(sorted_vals[:3])
        bottom3 = sum(sorted_vals[-3:])
        diff = top3 - bottom3

        q = ("Read all values from the chart. Sort them from highest to lowest. "
             "Compute the sum of the top 3 values, then subtract the sum of "
             "the bottom 3 values. What is the result?")

        image = self._render_single(rng, categories, values, chart_type, pool)
        return q, str(diff), image

    def _qa_conditional_growth(self, rng, categories, vals_2022, vals_2023,
                               growth, chart_type, pool):
        # Steps: 1) read 2023 values, 2) filter those > 100, 3) read their 2022 values,
        #        4) compute growth rates, 5) find the maximum growth rate
        eligible = [(i, g) for i, (v23, g) in enumerate(zip(vals_2023, growth))
                    if v23 > 100]
        if not eligible:
            return None
        best_idx, best_growth = max(eligible, key=lambda x: x[1])
        growth_pct = round((best_growth - 1) * 100, 1)
        best_cat = categories[best_idx]

        q = ("Two years of data are shown (2022 and 2023). "
             "First, identify all categories whose 2023 value exceeds 100. "
             "For each, compute the growth rate as (2023 - 2022) / 2022 * 100%. "
             "Which category has the highest growth rate among those? "
             "Give the category name.")

        image = self._render_double(rng, categories, vals_2022, vals_2023,
                                     chart_type, pool)
        return q, best_cat, image

    def _qa_ratio_chain(self, rng, categories, values, chart_type, pool):
        # Steps: 1) find max, 2) find min, 3) compute ratio, 4) round to int,
        #        5) find category closest to that rounded value
        max_val = max(values)
        min_val = min(values)
        if min_val == 0:
            return None
        ratio = max_val / min_val
        rounded = round(ratio)
        # Find category whose value is closest to 'rounded'
        diffs = [(abs(v - rounded), i) for i, v in enumerate(values)]
        diffs.sort()
        closest_idx = diffs[0][1]
        closest_cat = categories[closest_idx]

        q = ("Read all values from the chart. Find the ratio of the maximum "
             "value to the minimum value. Round this ratio to the nearest "
             "integer. Then find the category whose value is closest to "
             "that integer. What is the category name?")

        image = self._render_single(rng, categories, values, chart_type, pool)
        return q, closest_cat, image

    def _qa_weighted_rank_score(self, rng, categories, values, chart_type, pool):
        # Steps: 1) sort descending, 2) assign weights (1st=N, 2nd=N-1, ...),
        #        3) compute weighted sum, 4) divide by sum of weights
        n = len(values)
        sorted_vals = sorted(values, reverse=True)
        weights = list(range(n, 0, -1))
        weighted_sum = sum(v * w for v, w in zip(sorted_vals, weights))
        weight_total = sum(weights)
        result = round(weighted_sum / weight_total, 1)

        q = ("Read all values from the chart. Sort them from highest to lowest. "
             "Assign weight N to the highest (where N is the number of categories), "
             "N-1 to the second highest, and so on down to 1 for the lowest. "
             "Compute the weighted average (weighted sum / sum of weights). "
             "Round to 1 decimal place.")

        image = self._render_single(rng, categories, values, chart_type, pool)
        return q, str(result), image

    def _qa_outlier_adjusted_mean(self, rng, categories, values, chart_type, pool):
        # Steps: 1) compute mean, 2) compute std, 3) remove values > mean+1.5*std
        #        or < mean-1.5*std, 4) compute adjusted mean
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = var ** 0.5
        if std < 1:
            return None
        lower = mean - 1.5 * std
        upper = mean + 1.5 * std
        filtered = [v for v in values if lower <= v <= upper]
        if len(filtered) == 0:
            return None
        if len(filtered) == len(values):
            # No outliers — report the mean itself as the adjusted mean
            adj_mean = round(mean, 1)
        else:
            adj_mean = round(sum(filtered) / len(filtered), 1)

        q = ("Read all values from the chart. Compute the mean and standard deviation. "
             "Remove any values more than 1.5 standard deviations from the mean "
             "(outliers). Compute the mean of the remaining values. "
             "Round to 1 decimal place.")

        image = self._render_single(rng, categories, values, chart_type, pool)
        return q, str(adj_mean), image

    def _qa_percentile_gap(self, rng, categories, values, chart_type, pool):
        # Steps: 1) sort ascending, 2) find 25th percentile (Q1),
        #        3) find 75th percentile (Q3), 4) compute IQR = Q3 - Q1,
        #        5) find how many values are outside [Q1-1.5*IQR, Q3+1.5*IQR]
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_vals[q1_idx]
        q3 = sorted_vals[q3_idx]
        iqr = q3 - q1
        if iqr == 0:
            return None
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = sum(1 for v in values if v < lower or v > upper)

        q = ("Read all values from the chart. Compute Q1 (25th percentile) and "
             "Q3 (75th percentile) by sorting values and taking the value at "
             "position N/4 and 3N/4 respectively. Compute IQR = Q3 - Q1. "
             "How many values fall outside the range [Q1 - 1.5×IQR, Q3 + 1.5×IQR]?")

        image = self._render_single(rng, categories, values, chart_type, pool)
        return q, str(outliers), image

    # ------------------------------------------------------------------ #
    # Rendering — single series, NO value labels
    # ------------------------------------------------------------------ #

    def _render_single(self, rng, categories, values, chart_type, pool) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 0.9) * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        y_label = rng.choice(_Y_LABELS)
        palette = style["palette"]
        fs = style["font_size_base"]
        lw = style["line_width"]

        x = np.arange(len(categories))
        if chart_type == "bar":
            bars = ax.bar(x, values, width=0.6,
                          color=[palette[i % len(palette)] for i in range(len(values))],
                          edgecolor="white", linewidth=0.5)
            # NO value labels — this is the key challenge
        else:
            ax.plot(x, values, "o-", color=palette[0], linewidth=lw, markersize=6)
            # NO value annotations

        ax.set_xticks(x)
        rotate = len(categories) > 6 or max(len(c) for c in categories) > 7
        ax.set_xticklabels(categories, rotation=45 if rotate else 0,
                           ha="right" if rotate else "center", fontsize=fs - 1)
        ax.set_ylabel(y_label, fontsize=fs + 1)
        ax.set_title(f"{y_label} by Category", fontsize=fs + 3, fontweight="bold")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

        # NO gridlines — must read from axis
        self._apply_style(fig, ax, style)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_double(self, rng, categories, vals_2022, vals_2023,
                       chart_type, pool) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.0) * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        y_label = rng.choice(_Y_LABELS)
        palette = style["palette"]
        fs = style["font_size_base"]
        lw = style["line_width"]

        x = np.arange(len(categories))
        w = 0.35
        if chart_type == "bar":
            ax.bar(x - w / 2, vals_2022, width=w, color=palette[0],
                   label="2022", edgecolor="white")
            ax.bar(x + w / 2, vals_2023, width=w, color=palette[1 % len(palette)],
                   label="2023", edgecolor="white")
        else:
            ax.plot(x, vals_2022, "o-", color=palette[0], linewidth=lw,
                    markersize=5, label="2022")
            ax.plot(x, vals_2023, "s--", color=palette[1 % len(palette)], linewidth=lw,
                    markersize=5, label="2023")

        ax.set_xticks(x)
        rotate = len(categories) > 6
        ax.set_xticklabels(categories, rotation=45 if rotate else 0,
                           ha="right" if rotate else "center", fontsize=fs - 1)
        ax.set_ylabel(y_label, fontsize=fs + 1)
        ax.set_title(f"{y_label}: 2022 vs 2023", fontsize=fs + 3, fontweight="bold")
        ax.legend(fontsize=fs)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
        self._apply_style(fig, ax, style)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
