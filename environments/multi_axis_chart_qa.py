"""
Multi-axis Chart QA (batch 3, 2026-04-14).

Target: figure QA — 2-axis chart cross-reading. A bar series
(on left y-axis) plus a line series (on right y-axis) over same x
categories. Cross-axis inference at high level.

Format: constant numeric (integer).

Difficulty axes:
  A) Pattern A: n_categories (4..9).
  B) Pattern H: value range expansion for bars/lines independently.
  C) Pattern G: value labels on bars/line hidden at L>=3.
  D) Pattern D/E: question type stays numeric "read bar at X" but at high
     levels requires cross-axis inference (read line at X, then scale
     by bar at X).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CATS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct"]

class MultiAxisChartQA(StandaloneVisualEnv):
    ENV_NAME = "multi_axis_chart"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_cats": 4 + level // 2,               # 4..8
            "bar_range": 20 + level * 5,            # 20..65
            "line_range_max": 50 + level * 10,      # 50..140
            "show_bar_labels": level <= 2,
            "show_line_labels": level <= 3,
            "cross_axis": level >= 4,               # ask cross-axis inference
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_cats"] + cfg["bar_range"] // 10

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_cats"]
        cats = _CATS[:n]
        bars = [rng.randint(5, cfg["bar_range"]) for _ in range(n)]
        line_vals = [rng.randint(10, cfg["line_range_max"]) for _ in range(n)]

        # Pick a category to query
        i = rng.randint(0, n - 1)
        cat = cats[i]

        if not cfg["cross_axis"]:
            # Ask bar value at category
            answer = str(bars[i])
            q = (f"In the chart, what is the value of the bar (left axis) for "
                 f"{cat}? Answer with an integer.")
        else:
            # Cross-axis: "what is (line_val[i] - bar_val[i])?"
            op = rng.choice(["diff", "line"])
            if op == "diff":
                answer = str(line_vals[i] - bars[i])
                q = (f"In the chart, what is (line value - bar value) at "
                     f"{cat}? Answer with an integer (may be negative).")
            else:
                answer = str(line_vals[i])
                q = (f"In the chart, what is the line value (right axis) "
                     f"at {cat}? Answer with an integer.")

        image = self._render(cats, bars, line_vals, cfg)
        return q, answer, image

    def _render(self, cats, bars, line_vals, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        fig, ax1 = plt.subplots(figsize=(7.2 * sc, 4.8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax1.set_facecolor(style["bg_color"])

        x_pos = list(range(len(cats)))
        bar_color = palette[0 % len(palette)]
        line_color = palette[2 % len(palette)]

        ax1.bar(x_pos, bars, color=bar_color,
                edgecolor="#000", linewidth=0.9, alpha=0.85,
                label="Bar series")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(cats, fontsize=fs - 1, rotation=20)
        ax1.tick_params(axis="y", labelsize=fs - 1, colors=bar_color)
        ax1.set_ylabel("Bar value (left)", fontsize=fs, color=bar_color)

        ax2 = ax1.twinx()
        ax2.plot(x_pos, line_vals, marker="s",
                 color=line_color, linewidth=2.2, markersize=7,
                 markerfacecolor=line_color, markeredgecolor="#000",
                 markeredgewidth=0.8, label="Line series")
        ax2.tick_params(axis="y", labelsize=fs - 1, colors=line_color)
        ax2.set_ylabel("Line value (right)", fontsize=fs, color=line_color)

        if cfg["show_bar_labels"]:
            for xi, bv in zip(x_pos, bars):
                ax1.text(xi, bv + 0.4, str(bv),
                         ha="center", va="bottom", fontsize=fs - 1)
        if cfg["show_line_labels"]:
            for xi, lv in zip(x_pos, line_vals):
                ax2.annotate(str(lv), (xi, lv),
                             xytext=(0, 8), textcoords="offset points",
                             ha="center", fontsize=fs - 1, color=line_color)

        ax1.set_title("Multi-axis chart", fontsize=fs + 1)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = MultiAxisChartQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[multi_axis_chart L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"multi_axis_chart_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[multi_axis_chart L{level} s{s}] A={env._answer}")
