"""
Box Plot Comparison QA environment.

Capabilities: D1 (chart value extraction) + D2 (multi-chart comparison)
Target regression: statistical, dynamic-math statistics.

2-4 side-by-side box-and-whisker plots on the same axis. Each is labeled
with a group name. Median, Q1, Q3, whiskers, and outliers are drawn.

4-option MCQ over groups: "Which group has the largest IQR?" etc.

Difficulty schedule (0..9):
  Axis 1: n_groups = 2 + level // 3  -> 2..5
  Axis 2: similarity  L0 clearly different, L9 overlapping
  Axis 2b: question_complexity  L<=2: median  L3..L6: IQR
           L>=7: outliers / subtle

Output: (question_str, answer_letter, PIL_Image)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_GROUP_POOLS = [
    ["Group A", "Group B", "Group C", "Group D", "Group E"],
    ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"],
    ["Spring", "Summer", "Fall", "Winter", "Annual"],
    ["Control", "Treatment A", "Treatment B", "Treatment C", "Placebo"],
    ["Region X", "Region Y", "Region Z", "Region W", "Region V"],
    ["Site 1", "Site 2", "Site 3", "Site 4", "Site 5"],
]

_Y_LABELS = [
    "Score", "Concentration (mg/L)", "Response Time (ms)", "Yield (%)",
    "Temperature (°C)", "Measurement", "Accuracy (%)", "Value",
]

class BoxPlotComparisonQA(StandaloneVisualEnv):
    ENV_NAME = "box_plot_comparison"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        n_groups = min(5, 2 + level // 3)
        # Similarity: 0 = very different medians/IQRs, 1 = almost identical
        similarity = min(0.95, level / 9.0 * 0.9 + 0.05)
        if level <= 2:
            qtype = "largest_median"
        elif level <= 6:
            qtype = "largest_iqr"
        else:
            qtype = "most_outliers"
        return {
            "n_groups": n_groups,
            "similarity": similarity,
            "qtype": qtype,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2243)
        np_rng = np.random.RandomState(sub_rng.randint(0, 1_000_000))

        for _ in range(10):
            try:
                result = self._try_generate(sub_rng, np_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _try_generate(self, rng, np_rng, cfg, level):
        pool = list(rng.choice(_GROUP_POOLS))
        rng.shuffle(pool)
        n_groups = cfg["n_groups"]
        groups = pool[:n_groups]

        # Generate data
        similarity = cfg["similarity"]
        # When similar: narrow range of centers and spreads
        base_center = rng.uniform(30, 70)
        base_spread = rng.uniform(8, 20)

        data = {}
        stats = {}
        for g in groups:
            # center offset decreases with similarity
            c_off = rng.uniform(-20, 20) * (1 - similarity)
            s_mul = rng.uniform(0.6, 1.6) * (1 - similarity) + similarity
            center = base_center + c_off
            spread = max(3.0, base_spread * s_mul)
            n = np_rng.randint(40, 90)
            d = np_rng.normal(center, spread, n)
            # Add outliers occasionally (higher chance for qtype most_outliers)
            n_out = 0
            if cfg["qtype"] == "most_outliers":
                n_out = rng.randint(0, 3)
            elif np_rng.rand() > 0.6:
                n_out = rng.randint(1, 2)
            if n_out > 0:
                out_vals = center + np_rng.choice([-1, 1], n_out) * spread * np_rng.uniform(
                    2.8, 4.2, n_out)
                d = np.concatenate([d, out_vals])
            data[g] = d
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            iqr = q3 - q1
            # Count outliers using 1.5*IQR rule
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(np.sum((d < lower) | (d > upper)))
            stats[g] = {
                "median": float(med),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "n_outliers": n_outliers,
            }

        qtype = cfg["qtype"]

        if qtype == "largest_median":
            ranked = sorted(groups, key=lambda g: stats[g]["median"],
                            reverse=True)
            # reject if top two medians are too close when level low
            if level <= 2:
                if stats[ranked[0]]["median"] - stats[ranked[1]]["median"] < 5:
                    return None
            correct_name = ranked[0]
            q_stem = ("Which group has the highest median value in the "
                      "side-by-side box plot?")

        elif qtype == "largest_iqr":
            ranked = sorted(groups, key=lambda g: stats[g]["iqr"],
                            reverse=True)
            if stats[ranked[0]]["iqr"] - stats[ranked[1]]["iqr"] < 1.5:
                return None
            correct_name = ranked[0]
            q_stem = ("Which group has the largest interquartile range "
                      "(IQR = Q3 - Q1) in the box plots shown?")

        elif qtype == "most_outliers":
            ranked = sorted(groups, key=lambda g: stats[g]["n_outliers"],
                            reverse=True)
            if stats[ranked[0]]["n_outliers"] == stats[ranked[1]]["n_outliers"]:
                return None
            if stats[ranked[0]]["n_outliers"] == 0:
                return None
            correct_name = ranked[0]
            q_stem = ("Which group has the most outliers (points drawn beyond "
                      "the whiskers) in the box plot comparison?")

        else:
            return None

        # reference expects bare-text answer (group label), not MCQ letter.
        # Provide the group list in the prompt so the answer space is closed
        # but force the model to emit the full group label.
        question = (
            f"{q_stem} Groups shown: " + ", ".join(groups) +
            ". Answer with the group's full name."
        )

        image = self._render(rng, groups, data, stats)
        return question, correct_name, image

    def _render(self, rng, groups, data, stats):
        style = self._random_style()
        palette = style["palette"]
        n = len(groups)
        fig, ax = plt.subplots(figsize=(max(6, n * 1.4) * style["figsize_scale"],
                                        5 * style["figsize_scale"]))

        bp_data = [data[g] for g in groups]
        colors = [palette[i % len(palette)] for i in range(n)]
        bp = ax.boxplot(bp_data, labels=groups, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker='o', markersize=5,
                                         markerfacecolor='red',
                                         markeredgecolor='darkred'))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(2)

        y_label = rng.choice(_Y_LABELS)
        ax.set_ylabel(y_label, fontsize=style["font_size_base"])
        ax.set_title("Box Plot Comparison",
                     fontsize=style["font_size_base"] + 2, pad=10)
        ax.tick_params(labelsize=style["font_size_base"])
        self._apply_style(fig, ax, style)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
