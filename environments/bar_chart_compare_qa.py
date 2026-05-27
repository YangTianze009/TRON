"""
Bar Chart Compare QA (batch 3, 2026-04-14).

Target: figure QA / chart comparison. Two bar subsets in a bar
chart — the model compares or computes a ratio. Format is constant short
numeric (integer ratio like "3" or decimal ratio like "1.5").

Difficulty axes:
  A) Pattern A: n_bars (4..10) grows with level.
  B) Pattern D: value-margin (gap between the two bars being compared)
     shrinks so the model must read more carefully.
  C) Pattern G: value labels on bars are visible at L<=2 only.
  D) Pattern H: denominator can be rational at high levels.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CATEGORIES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
               "Gamma", "Hotel", "India", "Juliet", "Kilo", "Lima"]

class BarChartCompareQA(StandaloneVisualEnv):
    ENV_NAME = "bar_chart_compare"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesigned L0-L9: L0 targets ~40-60% pass, L9 targets ~5-15%.
        # Previously L0 was trivial (show labels, 4 bars, simple difference).
        # Now L0 already requires ratio/difference with NO value labels.
        return {
            # Bar count: even L0 has 5 bars, L9 has 10
            "n_bars": 5 + level * 5 // 9,             # 5,5,6,6,7,7,8,8,9,10
            # Value range: larger = harder to read from axis
            "value_range": 30 + level * 7,             # 30..93
            # Never show value labels -- forces axis reading at all levels
            "show_value_labels": False,
            # L0-L2: absolute difference (but no labels!)
            # L3-L5: ratio (non-integer)
            # L6-L9: multi-step
            "ratio_mode": level >= 3,
            "non_integer_ratio": level >= 3,           # always non-integer when ratio
            # Keep Y-axis numeric at ALL levels — hiding it makes the task
            # unsolvable (L6-L9 went to 10% pass because model had no numeric
            # reference). Difficulty comes from multi-step + close margin.
            "hide_y_ticks": False,
            "multi_step": level >= 6,                  # chain two computations
            "close_margin": level >= 4,                # bars differ by < 3
            # L7+: use non-round y-axis scale (offset) to make reading harder
            "y_axis_offset": level >= 7,
            # L8+: bars differ by 2-3 (still readable given y-axis)
            "adversarial_close": False,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_bars"] * 10 + cfg["value_range"]

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_bars"]
        labels = rng.sample(_CATEGORIES, n)
        values = [rng.randint(2, cfg["value_range"]) for _ in range(n)]

        # At high levels, force bars to have close margins
        if cfg.get("adversarial_close"):
            base = rng.randint(25, cfg["value_range"] - 5)
            values = [base + rng.randint(-1, 1) for _ in range(n)]
            values = [max(2, v) for v in values]
        elif cfg.get("close_margin"):
            base = rng.randint(15, cfg["value_range"] - 5)
            values = [base + rng.randint(-2, 2) for _ in range(n)]
            values = [max(2, v) for v in values]

        # Pick two distinct bars to compare
        i, j = rng.sample(range(n), 2)
        vi, vj = values[i], values[j]
        if vi == vj:
            # tweak
            values[i] = max(2, vi - 1)
            vi = values[i]
        if vi == vj:
            return None

        if cfg.get("multi_step"):
            # L8-L9: diverse multi-step operations
            op = rng.choice(["sum_minus_third", "product_mod", "range_ratio", "abs_diff_sum"])
            if op == "sum_minus_third":
                k = rng.choice([idx for idx in range(n) if idx != i and idx != j])
                vk = values[k]
                result = vi + vj - vk
                answer = str(result)
                q = (f"In the bar chart, compute {labels[i]} + {labels[j]} - "
                     f"{labels[k]}. Give your answer as an integer.")
            elif op == "product_mod":
                # Product of two smallest values
                sorted_v = sorted(enumerate(values), key=lambda x: x[1])
                idx1, v1 = sorted_v[0]
                idx2, v2 = sorted_v[1]
                result = v1 * v2
                answer = str(result)
                q = (f"In the bar chart, find the two bars with the smallest values "
                     f"and compute their product. Give an integer.")
            elif op == "range_ratio":
                max_v = max(values)
                min_v = min(values)
                if min_v == 0:
                    return None
                result = round((max_v - min_v) / min_v, 1)
                answer = f"{result:.1f}" if result != int(result) else str(int(result))
                q = ("In the bar chart, compute (max - min) / min across all bars. "
                     "Round to 1 decimal place.")
            else:  # abs_diff_sum
                # Sum of absolute differences between consecutive bars
                total_diff = sum(abs(values[idx_d] - values[idx_d + 1]) for idx_d in range(n - 1))
                result = total_diff
                answer = str(result)
                q = ("Sum the absolute differences between each pair of adjacent bars "
                     "(left to right). Give an integer.")
        elif cfg["ratio_mode"]:
            # Ratio larger / smaller
            big, small = max(vi, vj), min(vi, vj)
            if cfg["non_integer_ratio"]:
                ratio = big / small
                answer = f"{ratio:.1f}"
            else:
                # Try to snap small so big is an integer multiple
                k = big // small
                if k >= 2:
                    big = small * k
                    values[i if vi == max(vi, vj) else j] = big
                ratio = big / small
                if abs(ratio - round(ratio)) > 0.05:
                    answer = f"{ratio:.1f}"
                else:
                    answer = str(int(round(ratio)))
            q = (f"In the bar chart, what is the ratio of the value of "
                 f"{labels[i if vi >= vj else j]} to the value of "
                 f"{labels[j if vi >= vj else i]}? Give your answer as an "
                 "integer or a decimal like 1.5.")
        else:
            # Difference (absolute)
            diff = abs(vi - vj)
            answer = str(diff)
            q = (f"In the bar chart, what is the absolute difference between "
                 f"the values of {labels[i]} and {labels[j]}? Give an integer.")

        image = self._render(labels, values, [i, j], cfg)
        return q, answer, image

    def _render(self, labels, values, highlighted, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        fig, ax = plt.subplots(figsize=(7 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        n = len(values)
        colors = [palette[i % len(palette)] for i in range(n)]
        # Highlight the two compared bars with stronger border
        edgecolors = ["#000" if i in highlighted else "#555" for i in range(n)]
        linewidths = [2.2 if i in highlighted else 0.9 for i in range(n)]

        bars = ax.bar(labels, values, color=colors,
                      edgecolor=edgecolors, linewidth=linewidths)

        if cfg["show_value_labels"]:
            for b, v in zip(bars, values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                        str(v), ha="center", va="bottom",
                        fontsize=fs - 1)

        ax.set_ylim(0, max(values) * 1.2 + 2)
        ax.set_ylabel("Value", fontsize=fs + 1)
        ax.set_title("Bar chart comparison", fontsize=fs + 1)
        ax.tick_params(axis="x", labelsize=fs - 1, rotation=25)
        ax.tick_params(axis="y", labelsize=fs - 1)
        if cfg.get("hide_y_ticks"):
            ax.set_yticklabels([])
            ax.set_ylabel("")
        if style.get("show_grid"):
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = BarChartCompareQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[bar_chart_compare L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"bar_chart_compare_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[bar_chart_compare L{level} s{s}] A={env._answer}")
