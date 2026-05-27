"""
Chart Comparison QA environment.

Two charts side by side with different data but same categories.
Questions require reading BOTH charts and combining information.
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
# Data pools
# ------------------------------------------------------------------ #

_CATEGORY_POOLS = {
    "regions": ["North", "South", "East", "West", "Central",
                "Northeast", "Southwest", "Pacific"],
    "quarters": ["Q1 2022", "Q2 2022", "Q3 2022", "Q4 2022",
                 "Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"],
    "products": ["Widget A", "Widget B", "Widget C", "Widget D",
                 "Widget E", "Widget F", "Widget G"],
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
}

_METRIC_PAIRS = [
    ("Revenue ($M)", "Profit ($M)"),
    ("Units Sold (K)", "Returns (K)"),
    ("Employees", "Revenue per Employee ($K)"),
    ("Budget ($K)", "Actual Spend ($K)"),
    ("Online Sales ($M)", "In-Store Sales ($M)"),
    ("Production (tons)", "Exports (tons)"),
]

class ChartComparisonQA(StandaloneVisualEnv):
    ENV_NAME = "chart_comparison"

    QUESTION_TYPES = [
        "which_higher_at", "largest_gap", "correlation",
        "combined_total", "divergence_point",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"qtypes": ["which_higher_at"],
                    "n_cats": (4, 5)}
        if level <= 5:
            return {"qtypes": ["which_higher_at", "largest_gap",
                               "combined_total"],
                    "n_cats": (5, 6)}
        if level <= 7:
            return {"qtypes": ["largest_gap", "correlation",
                               "combined_total", "divergence_point"],
                    "n_cats": (6, 7)}
        return {"qtypes": ["correlation", "divergence_point",
                           "combined_total"],
                "n_cats": (7, 8)}

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

        pool_name = rng.choice(list(_CATEGORY_POOLS.keys()))
        pool = list(_CATEGORY_POOLS[pool_name])
        rng.shuffle(pool)
        lo_c, hi_c = cfg["n_cats"]
        num_categories = rng.randint(lo_c, min(hi_c, len(pool)))
        categories = pool[:num_categories]

        metric1, metric2 = rng.choice(_METRIC_PAIRS)

        # Generate two data series
        low1, high1 = rng.choice([(10, 80), (20, 150), (50, 300), (100, 500)])
        low2, high2 = rng.choice([(5, 60), (10, 100), (30, 200), (50, 400)])
        values1 = [int(np_rng.randint(low1, high1 + 1)) for _ in range(num_categories)]
        values2 = [int(np_rng.randint(low2, high2 + 1)) for _ in range(num_categories)]

        chart_types = rng.choice([("bar", "bar"), ("bar", "line"), ("line", "line")])

        question, answer = self._make_qa(
            rng, qtype, categories, values1, values2, metric1, metric2
        )
        if question is None:
            return None

        image = self._render(rng, categories, values1, values2,
                              metric1, metric2, chart_types)
        return question, answer, image

    def _make_qa(self, rng, qtype, categories, v1, v2, m1, m2):
        n = len(categories)

        if qtype == "which_higher_at":
            idx = rng.randint(0, n - 1)
            cat = categories[idx]
            if v1[idx] > v2[idx]:
                ans = m1
            elif v2[idx] > v1[idx]:
                ans = m2
            else:
                ans = "equal"
            q = (f"At '{cat}', which metric has a higher value: "
                 f"'{m1}' (left chart) or '{m2}' (right chart)? "
                 f"Give the metric name, or 'equal' if they are the same.")
            return q, ans

        elif qtype == "largest_gap":
            gaps = [(abs(v1[i] - v2[i]), i) for i in range(n)]
            gaps.sort(reverse=True)
            best_idx = gaps[0][1]
            q = (f"For which category is the absolute difference between "
                 f"'{m1}' and '{m2}' the largest? Give the category name.")
            return q, categories[best_idx]

        elif qtype == "correlation":
            # Determine if the two series move in the same direction
            # Simple: count how many adjacent pairs move in same direction
            if n < 3:
                return None, None
            same_dir = 0
            for i in range(n - 1):
                d1 = v1[i + 1] - v1[i]
                d2 = v2[i + 1] - v2[i]
                if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
                    same_dir += 1
            total_pairs = n - 1
            if same_dir > total_pairs * 0.6:
                ans = "positive"
            elif same_dir < total_pairs * 0.4:
                ans = "negative"
            else:
                ans = "none"
            q = (f"Do the two metrics generally move in the same direction "
                 f"across categories? Answer 'positive' if they tend to "
                 f"increase/decrease together, 'negative' if they move "
                 f"in opposite directions, or 'none' if there is no "
                 f"clear pattern.")
            return q, ans

        elif qtype == "combined_total":
            total = sum(v1) + sum(v2)
            q = (f"What is the combined total of ALL values from BOTH charts? "
                 f"Add up every value shown in the left chart ({m1}) "
                 f"and every value in the right chart ({m2}).")
            return q, str(total)

        elif qtype == "divergence_point":
            # Find the category where one first exceeds the other
            # (assuming v1 starts lower than v2 or vice versa)
            if v1[0] >= v2[0]:
                # Look for where v2 first exceeds v1
                for i in range(1, n):
                    if v2[i] > v1[i] and v2[i - 1] <= v1[i - 1]:
                        q = (f"Starting from the left, '{m1}' begins higher than "
                             f"'{m2}'. At which category does '{m2}' first "
                             f"exceed '{m1}'? Give the category name. "
                             f"If it never does, answer 'never'.")
                        return q, categories[i]
                q = (f"Starting from the left, '{m1}' begins higher than "
                     f"'{m2}'. At which category does '{m2}' first "
                     f"exceed '{m1}'? Give the category name. "
                     f"If it never does, answer 'never'.")
                return q, "never"
            else:
                for i in range(1, n):
                    if v1[i] > v2[i] and v1[i - 1] <= v2[i - 1]:
                        q = (f"Starting from the left, '{m2}' begins higher than "
                             f"'{m1}'. At which category does '{m1}' first "
                             f"exceed '{m2}'? Give the category name. "
                             f"If it never does, answer 'never'.")
                        return q, categories[i]
                q = (f"Starting from the left, '{m2}' begins higher than "
                     f"'{m1}'. At which category does '{m1}' first "
                     f"exceed '{m2}'? Give the category name. "
                     f"If it never does, answer 'never'.")
                return q, "never"

        return None, None

    # ------------------------------------------------------------------ #
    # Rendering — two side-by-side subplots
    # ------------------------------------------------------------------ #

    def _render(self, rng, categories, v1, v2, m1, m2, chart_types) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        palette = style["palette"]
        fs = style["font_size_base"]
        lw = style["line_width"]

        x = np.arange(len(categories))

        # Left chart
        ct1 = chart_types[0]
        if ct1 == "bar":
            ax1.bar(x, v1, width=0.6,
                    color=[palette[i % len(palette)] for i in range(len(v1))],
                    edgecolor="white")
        else:
            ax1.plot(x, v1, "o-", color=palette[0], linewidth=lw, markersize=6)
        ax1.set_xticks(x)
        rotate = len(categories) > 5
        ax1.set_xticklabels(categories, rotation=45 if rotate else 0,
                            ha="right" if rotate else "center", fontsize=fs - 2)
        ax1.set_ylabel(m1, fontsize=fs)
        ax1.set_title(m1, fontsize=fs + 2, fontweight="bold")
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        self._apply_style(fig, ax1, style)

        # Right chart
        ct2 = chart_types[1]
        palette2 = palette[len(palette)//2:] + palette[:len(palette)//2]
        if ct2 == "bar":
            ax2.bar(x, v2, width=0.6,
                    color=[palette2[i % len(palette2)] for i in range(len(v2))],
                    edgecolor="white")
        else:
            ax2.plot(x, v2, "s--", color=palette2[0], linewidth=lw, markersize=6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45 if rotate else 0,
                            ha="right" if rotate else "center", fontsize=fs - 2)
        ax2.set_ylabel(m2, fontsize=fs)
        ax2.set_title(m2, fontsize=fs + 2, fontweight="bold")
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        self._apply_style(fig, ax2, style)

        # NO value labels, NO gridlines
        fig.suptitle("Comparison Dashboard", fontsize=fs + 4, fontweight="bold", y=1.02)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
