"""Read from chart and table together with multiple question types."""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class CrossReferenceChartQA(StandaloneVisualEnv):
    ENV_NAME = "cross_reference_chart"

    _CATEGORY_POOLS = [
        ["Alpha", "Beta", "Gamma", "Delta"],
        ["Product A", "Product B", "Product C", "Product D"],
        ["East", "West", "North", "South"],
        ["Q1", "Q2", "Q3", "Q4"],
        ["Team X", "Team Y", "Team Z", "Team W"],
        ["Segment 1", "Segment 2", "Segment 3", "Segment 4"],
        ["Region A", "Region B", "Region C", "Region D"],
        ["Brand P", "Brand Q", "Brand R", "Brand S"],
    ]

    _TITLE_VARIANTS = [
        "Values", "Performance", "Metrics", "Results", "Measurements",
    ]
    _TABLE_TITLES = [
        "Multipliers", "Weights", "Factors", "Coefficients", "Scaling",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        # Redesigned: L0 starts with multiply + compare_products (old L3-4).
        # L9 requires weighted_average with hidden bar labels.
        if level <= 1:
            return {"n_cats": 4, "val_lo": 10, "val_hi": 70, "mul_lo": 2, "mul_hi": 5,
                    "qtypes": ["multiply", "compare_products"]}
        if level <= 3:
            return {"n_cats": 4, "val_lo": 10, "val_hi": 80, "mul_lo": 2, "mul_hi": 7,
                    "qtypes": ["compare_products", "sum_all_products"]}
        if level <= 5:
            return {"n_cats": 4, "val_lo": 10, "val_hi": 90, "mul_lo": 2, "mul_hi": 8,
                    "qtypes": ["sum_all_products", "difference"]}
        if level <= 7:
            return {"n_cats": 4, "val_lo": 5, "val_hi": 95, "mul_lo": 2, "mul_hi": 10,
                    "qtypes": ["difference", "weighted_average"]}
        # L8-L9: keep labels hidden but only use integer-answer qtypes
        # (sum_all_products, difference). Free-response 1-decimal answers
        # combined with hidden labels + sparse y-ticks made L9
        # underdetermined. Also keep y-ticks at standard density for
        # readability.
        return {"n_cats": 4, "val_lo": 5, "val_hi": 99, "mul_lo": 3, "mul_hi": 12,
                "qtypes": ["sum_all_products", "difference"],
                "hide_bar_labels": True, "sparse_y_ticks": False}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 701)
        style = self._random_style()
        palette = style["palette"]

        n_cats = cfg["n_cats"]
        q_type = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        pool = list(sub_rng.choice(self._CATEGORY_POOLS))
        cats = pool[:n_cats]
        vals = [sub_rng.randint(cfg["val_lo"], cfg["val_hi"]) for _ in cats]
        multipliers = {c: sub_rng.randint(cfg["mul_lo"], cfg["mul_hi"]) for c in cats}

        # Question phrasing variants
        _read_chart_phrasings = [
            "According to the bar chart, what is the value for {t}?",
            "From the chart on the left, what is {t}'s value?",
            "Read the bar chart. What value does {t} have?",
        ]
        _read_table_phrasings = [
            "According to the table, what is the multiplier for {t}?",
            "From the table on the right, what multiplier is listed for {t}?",
            "What factor does the table show for {t}?",
        ]
        _multiply_phrasings = [
            "Using both the chart and table, what is {t}'s value multiplied by its multiplier?",
            "Compute {t}'s bar-chart value times its table multiplier.",
            "What is the product of {t}'s chart value and its table factor?",
        ]

        # Generate question
        if q_type == "read_chart":
            target = sub_rng.choice(cats)
            question = sub_rng.choice(_read_chart_phrasings).format(t=target)
            answer = str(vals[cats.index(target)])
        elif q_type == "read_table":
            target = sub_rng.choice(cats)
            question = sub_rng.choice(_read_table_phrasings).format(t=target)
            answer = str(multipliers[target])
        elif q_type == "multiply":
            target = sub_rng.choice(cats)
            result = vals[cats.index(target)] * multipliers[target]
            question = sub_rng.choice(_multiply_phrasings).format(t=target)
            answer = str(result)
        elif q_type == "compare_products":
            products = {c: vals[cats.index(c)] * multipliers[c] for c in cats}
            max_cat = max(products, key=products.get)
            question = sub_rng.choice([
                "Which item has the highest value times multiplier product?",
                "Which category yields the largest product of chart value and table factor?",
            ])
            answer = max_cat
        elif q_type == "sum_all_products":
            total = sum(vals[cats.index(c)] * multipliers[c] for c in cats)
            question = sub_rng.choice([
                "What is the sum of all (value x multiplier) products across all items?",
                "Add up each category's chart value times its multiplier. What is the total?",
            ])
            answer = str(total)
        elif q_type == "difference":
            products = {c: vals[cats.index(c)] * multipliers[c] for c in cats}
            hi = max(products, key=products.get)
            lo = min(products, key=products.get)
            diff = products[hi] - products[lo]
            question = sub_rng.choice([
                f"What is the difference between the largest and smallest (value x multiplier) products?",
                f"Subtract the lowest product from the highest product. What is the result?",
            ])
            answer = str(diff)
        elif q_type == "weighted_average":
            products = [vals[i] * multipliers[cats[i]] for i in range(n_cats)]
            total_mul = sum(multipliers[c] for c in cats)
            w_avg = round(sum(vals[i] * multipliers[cats[i]] for i in range(n_cats)) / total_mul, 1)
            question = sub_rng.choice([
                "What is the weighted average of the values, using the multipliers as weights? Round to 1 decimal.",
                "Compute sum(value_i * multiplier_i) / sum(multiplier_i). Round to 1 decimal place.",
            ])
            answer = str(w_avg)
        else:
            target = sub_rng.choice(cats)
            result = vals[cats.index(target)] * multipliers[target]
            question = f"What is {target}'s value times its multiplier?"
            answer = str(result)

        # Render
        chart_title = sub_rng.choice(self._TITLE_VARIANTS)
        table_title = sub_rng.choice(self._TABLE_TITLES)
        # Shuffle palette per-seed for visual diversity
        vis_palette = list(palette)
        sub_rng.shuffle(vis_palette)

        fig, (ax1, ax2) = plt.subplots(1, 2,
            figsize=(10 * style["figsize_scale"], 4.5 * style["figsize_scale"]),
            gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor(style["bg_color"])

        # Bar chart
        bar_colors = [vis_palette[i % len(vis_palette)] for i in range(n_cats)]
        bars = ax1.bar(cats, vals, color=bar_colors, edgecolor=style["geo_line_color"],
                       linewidth=style["line_width"] * 0.5)
        self._apply_style(fig, ax1, style)
        ax1.set_title(chart_title, fontsize=style["font_size_base"] + 1,
                      fontweight="bold", fontfamily=style["font_family"])
        ax1.set_ylabel("Amount", fontsize=style["font_size_base"],
                       fontfamily=style["font_family"])
        ax1.tick_params(labelsize=style["font_size_base"] - 1)

        # Add value labels on bars (only if not hidden at high levels)
        if not cfg.get("hide_bar_labels"):
            for bar, val in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(val), ha='center', va='bottom',
                        fontsize=style["font_size_base"] - 1,
                        fontweight='bold', fontfamily=style["font_family"])
        if cfg.get("sparse_y_ticks"):
            import matplotlib.ticker as mticker
            ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))

        # Table
        ax2.set_facecolor(style["bg_color"])
        ax2.axis("off")
        cell_text = [[c, str(multipliers[c])] for c in cats]
        t = ax2.table(cellText=cell_text,
                     colLabels=["Item", "Multiplier"],
                     cellLoc='center', loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(style["font_size_base"])
        t.scale(1, 1.5)
        for (r, c), cell in t.get_celld().items():
            if r == 0:
                cell.set_facecolor(vis_palette[0])
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor(style["bg_color"])
            cell.set_edgecolor(style["geo_line_color"])
        ax2.set_title(table_title, fontsize=style["font_size_base"] + 1,
                      fontweight="bold", pad=15, fontfamily=style["font_family"])

        plt.tight_layout()
        return question, answer, self.fig_to_pil(fig, dpi=style["dpi"])
