"""Sparkline Table Visual QA Environment.

Renders a table with embedded mini line charts (sparklines) per row.
"""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_ROW_NAMES = [
    ["Product A", "Product B", "Product C", "Product D", "Product E"],
    ["Region North", "Region South", "Region East", "Region West"],
    ["Dept Sales", "Dept Eng", "Dept Marketing", "Dept Finance", "Dept HR"],
    ["Stock AAPL", "Stock GOOG", "Stock MSFT", "Stock AMZN"],
]
_TIME_LABELS = ["Week 1-8", "Jan-Jun", "Q1-Q4 (monthly)", "Day 1-10"]

class SparklineTableQA(StandaloneVisualEnv):
    ENV_NAME = "sparkline_table"

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["upward_trend", "highest_peak"],
                    "qweights": [5, 5], "n_points": (5, 7)}
        if level <= 2:
            return {"qtypes": ["upward_trend", "downward_trend", "highest_peak"],
                    "qweights": [3, 3, 4], "n_points": (6, 8)}
        if level <= 4:
            return {"qtypes": ["upward_trend", "downward_trend", "most_volatile", "end_vs_start"],
                    "qweights": [2, 2, 3, 3], "n_points": (6, 10)}
        if level <= 6:
            return {"qtypes": ["most_volatile", "end_vs_start", "correlation_direction"],
                    "qweights": [3, 3, 4], "n_points": (7, 10)}
        if level <= 8:
            return {"qtypes": ["correlation_direction", "most_volatile"],
                    "qweights": [6, 4], "n_points": (8, 12)}
        return {"qtypes": ["correlation_direction", "most_volatile", "end_vs_start"],
                "qweights": [5, 3, 2], "n_points": (8, 12)}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 2301)
        np_rng = np.random.RandomState(seed * 100 + level)

        qtype = parameter.get("question_type")
        if qtype not in ["upward_trend", "downward_trend", "highest_peak",
                          "most_volatile", "end_vs_start", "correlation_direction"]:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        row_names = list(sub_rng.choice(_ROW_NAMES))
        n_rows = len(row_names)
        n_points = sub_rng.randint(*cfg["n_points"])

        # Generate sparkline data per row
        data = []
        for _ in range(n_rows):
            trend = np_rng.choice([-1, 0, 1])  # -1=down, 0=flat, 1=up
            base = np_rng.uniform(20, 80)
            noise = np_rng.normal(0, 5, n_points)
            slope = trend * np_rng.uniform(1, 4)
            vals = [round(base + slope * t + noise[t], 1) for t in range(n_points)]
            data.append(vals)

        question, answer = self._make_qa(rng, qtype, row_names, data)
        if question is None:
            return None

        style = self._random_style()
        image = self._render(style, row_names, data)
        return question, answer, image

    def _render(self, style, row_names, data):
        n_rows = len(row_names)
        n_points = len(data[0])
        row_height = 1.8
        fig_h = (n_rows * row_height + 1.5) * style["figsize_scale"]
        fig, axes = plt.subplots(n_rows, 2, figsize=(8 * style["figsize_scale"], fig_h),
                                  gridspec_kw={"width_ratios": [1, 3]},
                                  constrained_layout=True)
        if n_rows == 1:
            axes = [axes]
        palette = style["palette"]

        # Compute global y-range for correlation / volatility comparability.
        global_min = min(min(row) for row in data)
        global_max = max(max(row) for row in data)
        global_pad = max((global_max - global_min) * 0.1, 1.0)
        # Use unified y-scale at L4+ (where most_volatile/correlation questions
        # appear) to avoid per-row y-scale underdetermining the visual answer.
        unified_scale = True

        for i in range(n_rows):
            # Label cell
            ax_label = axes[i][0]
            ax_label.text(0.5, 0.5, row_names[i], ha="center", va="center",
                         fontsize=style["font_size_base"], fontfamily=style["font_family"],
                         fontweight="bold")
            ax_label.set_xlim(0, 1)
            ax_label.set_ylim(0, 1)
            ax_label.axis("off")

            # Sparkline cell
            ax_spark = axes[i][1]
            color = palette[i % len(palette)]
            ax_spark.plot(range(n_points), data[i], color=color,
                         linewidth=style["line_width"], marker="o", markersize=2)
            y_min = min(data[i])
            y_max = max(data[i])
            if unified_scale:
                baseline = global_min - global_pad
                top = global_max + global_pad
            else:
                y_pad = max((y_max - y_min) * 0.15, 1.0)
                baseline = y_min - y_pad
                top = y_max + y_pad
            ax_spark.fill_between(range(n_points), data[i], baseline,
                                  alpha=0.15, color=color)
            ax_spark.set_ylim(baseline, top)
            # Mark start/end values
            ax_spark.text(0, data[i][0], f"{data[i][0]:.0f}", fontsize=7,
                         ha="right", va="center")
            ax_spark.text(n_points - 1, data[i][-1], f"{data[i][-1]:.0f}",
                         fontsize=7, ha="left", va="center")
            # Label min and max with small markers so the reader knows the
            # per-row range (critical for volatility judgments).
            try:
                idx_min = data[i].index(y_min)
                idx_max = data[i].index(y_max)
                ax_spark.plot(idx_min, y_min, "v", color="#c0392b",
                              markersize=5, zorder=5)
                ax_spark.plot(idx_max, y_max, "^", color="#27ae60",
                              markersize=5, zorder=5)
                ax_spark.text(idx_min, y_min, f" min={y_min:.0f}",
                              fontsize=6, ha="left", va="top", color="#c0392b")
                ax_spark.text(idx_max, y_max, f" max={y_max:.0f}",
                              fontsize=6, ha="left", va="bottom", color="#27ae60")
            except ValueError:
                pass
            ax_spark.set_xlim(-0.8, n_points - 0.2)
            ax_spark.axis("off")

        fig.patch.set_facecolor(style["bg_color"])
        fig.suptitle("Sparkline Table", fontsize=style["font_size_base"] + 3)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _make_qa(self, rng, qtype, row_names, data):
        n_rows = len(row_names)

        def _trend_slope(vals):
            n = len(vals)
            xs = list(range(n))
            x_mean = sum(xs) / n
            y_mean = sum(vals) / n
            num = sum((xs[i] - x_mean) * (vals[i] - y_mean) for i in range(n))
            den = sum((xs[i] - x_mean) ** 2 for i in range(n))
            return num / den if den != 0 else 0

        slopes = [_trend_slope(data[i]) for i in range(n_rows)]

        if qtype == "upward_trend":
            idx = slopes.index(max(slopes))
            return "Which row shows the strongest upward trend?", row_names[idx]

        elif qtype == "downward_trend":
            idx = slopes.index(min(slopes))
            return "Which row shows the strongest downward trend?", row_names[idx]

        elif qtype == "highest_peak":
            peaks = [max(data[i]) for i in range(n_rows)]
            idx = peaks.index(max(peaks))
            return "Which row has the highest peak value?", row_names[idx]

        elif qtype == "most_volatile":
            stds = [float(np.std(data[i])) for i in range(n_rows)]
            idx = stds.index(max(stds))
            return ("Which row shows the most volatile (highest variation) "
                    "trend?"), row_names[idx]

        elif qtype == "end_vs_start":
            diffs = [data[i][-1] - data[i][0] for i in range(n_rows)]
            idx = diffs.index(max(diffs))
            return ("Which row has the largest increase from its first "
                    "to its last data point?"), row_names[idx]

        elif qtype == "correlation_direction":
            # Find the pair of rows with the highest Pearson correlation
            best_corr = -2
            best_pair = (0, 1)
            for i in range(n_rows):
                for j in range(i + 1, n_rows):
                    n_pts = len(data[i])
                    mean_i = sum(data[i]) / n_pts
                    mean_j = sum(data[j]) / n_pts
                    cov = sum((data[i][k] - mean_i) * (data[j][k] - mean_j) for k in range(n_pts))
                    var_i = sum((data[i][k] - mean_i) ** 2 for k in range(n_pts))
                    var_j = sum((data[j][k] - mean_j) ** 2 for k in range(n_pts))
                    denom = (var_i * var_j) ** 0.5
                    if denom > 0:
                        corr = cov / denom
                    else:
                        corr = 0
                    if corr > best_corr:
                        best_corr = corr
                        best_pair = (i, j)
            return (f"Which two rows have the most similar trends "
                    f"(highest positive correlation)? "
                    f"Answer as 'RowA, RowB' (alphabetical order)."),\
                    f"{row_names[best_pair[0]]}, {row_names[best_pair[1]]}"

        else:
            lows = [min(data[i]) for i in range(n_rows)]
            idx = lows.index(min(lows))
            return "Which row has the lowest trough value?", row_names[idx]
