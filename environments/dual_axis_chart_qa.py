"""
Dual-axis chart QA — two y-axes with different scales, cross-axis reasoning.
Targets: chart-reading (complex chart types), statistical reasoning.

Capabilities: V3 (chart extraction), R1 (arithmetic), R4 (statistical), R5 (multi-step)
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class DualAxisChartQA(StandaloneVisualEnv):
    ENV_NAME = "dual_axis_chart"

    _CATEGORY_POOLS = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
        ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"],
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ]
    _SERIES_PAIRS = [
        ("Revenue ($K)", "Growth Rate (%)"),
        ("Temperature (°C)", "Rainfall (mm)"),
        ("Sales (units)", "Profit Margin (%)"),
        ("Production (tons)", "Efficiency (%)"),
    ]

    def _level_config(self, level: int) -> Dict:
        # Reordered: "max_*" (find tallest bar) is visually much easier than
        # "read_*_axis" (exact numeric read), so max tasks go first.
        # Previous schedule had L0=read_left (0.55) vs L3=max (1.00) — inverted.
        # Iter 3 (2026-04-17): L3 kept dropping to 0.60 (vs L0=1.00)
        # because read_*_axis needs ~exact numerical readout from a shared
        # axis without gridlines. Push read_* tasks to L4-5 and let L3
        # share max + compare_axes — categorical / qualitative comparisons.
        if level <= 0:
            return {"n_cats": 4, "qtypes": ["max_left"]}
        if level == 1:
            return {"n_cats": 4, "qtypes": ["max_left", "max_right"]}
        if level == 2:
            return {"n_cats": 5, "qtypes": ["max_left", "max_right", "compare_axes"]}
        if level == 3:
            return {"n_cats": 5, "qtypes": ["max_left", "max_right", "compare_axes"]}
        if level == 4:
            return {"n_cats": 5, "qtypes": ["read_left_axis", "read_right_axis", "compare_axes"]}
        if level == 5:
            return {"n_cats": 6, "qtypes": ["read_left_axis", "read_right_axis", "crossover_point"]}
        if level == 6:
            return {"n_cats": 6, "qtypes": ["crossover_point", "correlation_direction"]}
        if level == 7:
            return {"n_cats": 6, "qtypes": ["correlation_direction", "ratio_at_time"]}
        if level == 8:
            return {"n_cats": 7, "qtypes": ["ratio_at_time", "max_product"]}
        # L9 iter-4 (2026-04-17): saturated — pivot to a 3-step compound
        # question that requires reading both axes at multiple categories
        # AND performing arithmetic. "weighted_sum_top3": multiply each
        # category's left value by its right value, sort, sum top 3.
        return {"n_cats": 8, "qtypes": ["weighted_sum_top3", "max_product"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 708)
        np_rng = np.random.RandomState(seed)

        question_type = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        n_cats = cfg["n_cats"]

        categories = sub_rng.choice(self._CATEGORY_POOLS)[:n_cats]
        left_label, right_label = sub_rng.choice(self._SERIES_PAIRS)

        # Generate data
        left_vals = [sub_rng.randint(10, 90) for _ in categories]
        right_vals = [sub_rng.randint(5, 50) for _ in categories]

        question, answer = self._make_qa(
            rng, question_type, categories, left_vals, right_vals, left_label, right_label
        )
        if question is None:
            return None

        style = self._random_style()
        image = self._render(rng, categories, left_vals, right_vals, left_label, right_label, style)
        return question, str(answer), image

    def _make_qa(self, rng, qtype, cats, lvals, rvals, llabel, rlabel):
        # Aliases from controller names to internal names
        _aliases = {
            "left_value_at": "read_left_axis",
            "right_value_at": "read_right_axis",
            "max_left": "max_left",
            "max_right": "max_right",
            "ratio_at_time": "ratio_at_time",
            "crossover_point": "crossover_point",
        }
        qtype = _aliases.get(qtype, qtype)

        if qtype == "read_left_axis":
            cat = rng.choice(cats)
            ci = cats.index(cat)
            return f"What is the {llabel.split('(')[0].strip()} for '{cat}'?", lvals[ci]
        elif qtype == "read_right_axis":
            cat = rng.choice(cats)
            ci = cats.index(cat)
            return f"What is the {rlabel.split('(')[0].strip()} for '{cat}'?", rvals[ci]
        elif qtype == "compare_axes":
            cat = rng.choice(cats)
            ci = cats.index(cat)
            higher = llabel.split("(")[0].strip() if lvals[ci] > rvals[ci] else rlabel.split("(")[0].strip()
            return (
                f"For '{cat}', which metric has a higher numerical value: "
                f"{llabel.split('(')[0].strip()} or {rlabel.split('(')[0].strip()}? "
                f"(Compare the raw numbers, ignoring units)",
                higher
            )
        elif qtype == "correlation_direction":
            # Simple correlation check
            n = len(cats)
            lmean = sum(lvals) / n
            rmean = sum(rvals) / n
            cov = sum((lvals[i] - lmean) * (rvals[i] - rmean) for i in range(n))
            direction = "positive" if cov > 0 else "negative"
            return (
                f"Do {llabel.split('(')[0].strip()} and {rlabel.split('(')[0].strip()} "
                f"appear to have a positive or negative correlation?",
                direction
            )
        elif qtype == "max_left":
            max_idx = lvals.index(max(lvals))
            return (
                f"Which category has the highest {llabel.split('(')[0].strip()}?",
                cats[max_idx]
            )
        elif qtype == "max_right":
            max_idx = rvals.index(max(rvals))
            return (
                f"Which category has the highest {rlabel.split('(')[0].strip()}?",
                cats[max_idx]
            )
        elif qtype == "ratio_at_time":
            cat = rng.choice(cats)
            ci = cats.index(cat)
            if rvals[ci] == 0:
                return None, None
            ratio = round(lvals[ci] / rvals[ci], 2)
            return (
                f"For '{cat}', what is the ratio of {llabel.split('(')[0].strip()} to "
                f"{rlabel.split('(')[0].strip()}? (Round to 2 decimals)",
                ratio
            )
        elif qtype == "crossover_point":
            # Find a category where left and right values are closest
            diffs = [abs(lvals[i] - rvals[i]) for i in range(len(cats))]
            min_idx = diffs.index(min(diffs))
            return (
                f"At which category are {llabel.split('(')[0].strip()} and "
                f"{rlabel.split('(')[0].strip()} closest in value?",
                cats[min_idx]
            )
        elif qtype == "max_product":
            products = [lvals[i] * rvals[i] for i in range(len(cats))]
            max_idx = products.index(max(products))
            return (
                f"Which category has the highest product of "
                f"{llabel.split('(')[0].strip()} × {rlabel.split('(')[0].strip()}?",
                cats[max_idx]
            )
        elif qtype == "weighted_sum_top3":
            products = [lvals[i] * rvals[i] for i in range(len(cats))]
            top3 = sorted(products, reverse=True)[:3]
            return (
                f"For each category, compute "
                f"{llabel.split('(')[0].strip()} × "
                f"{rlabel.split('(')[0].strip()}. Rank the categories by "
                f"this product, then sum the TOP 3 products. Answer as a "
                f"single integer.",
                sum(top3)
            )
        return None, None

    def _render(self, rng, categories, left_vals, right_vals, left_label, right_label, style):
        palette = style["palette"]
        fig, ax1 = plt.subplots(figsize=rng.choice([(8, 5), (9, 5)]))
        self._apply_style(fig, ax1, style)

        x = np.arange(len(categories))
        width = 0.4

        # Left axis - bars
        bars = ax1.bar(x, left_vals, width, color=palette[0], alpha=0.8, label=left_label)
        ax1.set_ylabel(left_label, color=palette[0], fontsize=style["font_size_base"])
        ax1.tick_params(axis='y', labelcolor=palette[0])

        # Right axis - line
        ax2 = ax1.twinx()
        ax2.plot(x, right_vals, color=palette[1], marker='o', linewidth=2.5,
                markersize=7, label=right_label)
        ax2.set_ylabel(right_label, color=palette[1], fontsize=style["font_size_base"])
        ax2.tick_params(axis='y', labelcolor=palette[1])

        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=style["font_size_base"])

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=style["font_size_base"] - 1)

        ax1.set_title("Dual-Axis Chart", fontsize=style["font_size_base"] + 2, fontweight="bold")

        return self.fig_to_pil(fig, dpi=style["dpi"])
