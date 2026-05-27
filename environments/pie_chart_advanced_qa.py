"""Pie chart (donut) QA — percentages, angles, and arc questions over sectors.

Fixes:
  - Previously used autopct to print exact percentages on every slice,
    which is the answer for most qtypes. That leakage is removed.
  - Instead, the learner reads rough sector sizes from visual proportions,
    OR is shown a LEGEND with percentages mapped to labels (so some
    qtypes depend on reading the legend but questions never contain the
    raw percentage values).
  - Diverse sector count (4-8), label pools, donut width, start angle.
  - Expanded question template pool, MCQ version for compare questions.
  - L0 = easy 4-sector difference (largest/smallest); L9 = arc_length
    over many sectors.
"""
import random
import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class PieChartAdvancedQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "pie_chart_advanced"

    _LABEL_POOLS = [
        ["Product A", "Product B", "Product C", "Product D",
         "Product E", "Product F", "Product G", "Product H"],
        ["North", "South", "East", "West", "Central",
         "Northeast", "Northwest", "Southeast"],
        ["Marketing", "R&D", "Sales", "Admin", "Support", "Finance", "HR", "IT"],
        ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
        ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown"],
        ["Team-1", "Team-2", "Team-3", "Team-4",
         "Team-5", "Team-6", "Team-7", "Team-8"],
    ]

    _Q_TEMPLATES = {
        "largest_sector": [
            "Which sector has the largest share?",
            "Which category corresponds to the biggest slice in the pie chart?",
        ],
        "smallest_sector": [
            "Which sector has the smallest share?",
            "Which category corresponds to the smallest slice?",
        ],
        "sum_of_two": [
            "What is the combined percentage of '{a}' and '{b}'?",
            "Add the percentages of '{a}' and '{b}'. What is the sum?",
        ],
        "difference_pct": [
            "What is the absolute difference in percentage between '{a}' and '{b}'?",
            "Compute |pct({a}) - pct({b})|.",
        ],
        "sector_angle": [
            "What is the central angle (in degrees) of '{a}'?",
            "For the slice labeled '{a}', compute its central angle in degrees.",
        ],
        "combined_percentage_smallest": [
            "What is the combined percentage of the {n} smallest sectors ({names})? "
            "Round to 1 decimal.",
        ],
        "arc_length": [
            "If the pie chart has radius 1, what is the arc length of '{a}'? Round to 2 decimals.",
            "For a unit-radius pie, compute the arc length of the '{a}' sector. Round to 2 decimals.",
        ],
        "count_above_threshold": [
            "How many sectors have a share strictly greater than {thr}%?",
        ],
    }

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 0:
            return {"qtypes": ["largest_sector", "smallest_sector"],
                    "qweights": [5, 5], "n_sectors": (4, 4),
                    "hide_pcts_in_legend": False, "similar_colors": False}
        if level <= 1:
            return {"qtypes": ["largest_sector", "smallest_sector", "sum_of_two"],
                    "qweights": [4, 4, 2], "n_sectors": (4, 5),
                    "hide_pcts_in_legend": False, "similar_colors": False}
        if level <= 2:
            return {"qtypes": ["sum_of_two", "difference_pct"],
                    "qweights": [5, 5], "n_sectors": (4, 5),
                    "hide_pcts_in_legend": False, "similar_colors": False}
        if level <= 4:
            return {"qtypes": ["difference_pct", "sector_angle", "count_above_threshold"],
                    "qweights": [4, 4, 2], "n_sectors": (5, 6),
                    "hide_pcts_in_legend": False, "similar_colors": False}
        if level <= 6:
            # L5-L6: hide percentages from legend (model estimates from
            # angles). Keep distinct colors so legend-to-wedge mapping
            # remains possible.
            return {"qtypes": ["sector_angle", "combined_percentage_smallest",
                               "arc_length", "largest_sector", "smallest_sector"],
                    "qweights": [3, 4, 3, 2, 2], "n_sectors": (5, 7),
                    "hide_pcts_in_legend": True, "similar_colors": False}
        if level <= 8:
            return {"qtypes": ["arc_length", "combined_percentage_smallest",
                               "largest_sector", "smallest_sector"],
                    "qweights": [4, 4, 2, 2], "n_sectors": (6, 8),
                    "hide_pcts_in_legend": True, "similar_colors": False}
        return {"qtypes": ["arc_length", "combined_percentage_smallest",
                           "largest_sector", "smallest_sector"],
                "qweights": [3, 4, 2, 2], "n_sectors": (6, 8),
                "hide_pcts_in_legend": True, "similar_colors": False}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub = random.Random((self.seed or 0) * 1000 + level * 37 + 7177)

        qtype = parameter.get("question_type")
        if qtype not in [
            "largest_sector", "smallest_sector", "sum_of_two",
            "difference_pct", "sector_angle", "combined_percentage_smallest",
            "arc_length", "count_above_threshold",
        ]:
            qtype = sub.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        pool = sub.choice(self._LABEL_POOLS)
        pool = list(pool); sub.shuffle(pool)
        n_sectors = sub.randint(*cfg["n_sectors"])
        n_sectors = min(n_sectors, len(pool))
        labels = pool[:n_sectors]

        raw = [sub.randint(5, 40) for _ in labels]
        total = sum(raw)
        pcts = [round(v / total * 100, 1) for v in raw]
        pcts[-1] = round(100 - sum(pcts[:-1]), 1)

        # Build question / answer
        if qtype == "largest_sector":
            idx = pcts.index(max(pcts))
            # MCQ
            options = list(labels); sub.shuffle(options)
            correct = labels[idx]
            idx_opt = options.index(correct)
            letter = chr(ord("A") + idx_opt)
            q = (sub.choice(self._Q_TEMPLATES["largest_sector"])
                 + "  " + "  ".join(f"({chr(ord('A')+i)}) {o}"
                                    for i, o in enumerate(options)))
            answer = letter
        elif qtype == "smallest_sector":
            idx = pcts.index(min(pcts))
            options = list(labels); sub.shuffle(options)
            correct = labels[idx]
            idx_opt = options.index(correct)
            letter = chr(ord("A") + idx_opt)
            q = (sub.choice(self._Q_TEMPLATES["smallest_sector"])
                 + "  " + "  ".join(f"({chr(ord('A')+i)}) {o}"
                                    for i, o in enumerate(options)))
            answer = letter
        elif qtype == "sum_of_two":
            i, j = sub.sample(range(n_sectors), 2)
            s = round(pcts[i] + pcts[j], 1)
            q = sub.choice(self._Q_TEMPLATES["sum_of_two"]).format(
                a=labels[i], b=labels[j])
            answer = str(s)
        elif qtype == "difference_pct":
            i, j = sub.sample(range(n_sectors), 2)
            d = round(abs(pcts[i] - pcts[j]), 1)
            q = sub.choice(self._Q_TEMPLATES["difference_pct"]).format(
                a=labels[i], b=labels[j])
            answer = str(d)
        elif qtype == "sector_angle":
            i = sub.randint(0, n_sectors - 1)
            angle = round(pcts[i] / 100 * 360, 1)
            q = sub.choice(self._Q_TEMPLATES["sector_angle"]).format(a=labels[i])
            answer = str(angle)
        elif qtype == "combined_percentage_smallest":
            n_pick = min(sub.randint(2, 3), n_sectors - 1)
            sorted_pcts = sorted(zip(pcts, labels))
            picked = sorted_pcts[:n_pick]
            total_pct = round(sum(p for p, _ in picked), 1)
            names = ", ".join(l for _, l in picked)
            q = self._Q_TEMPLATES["combined_percentage_smallest"][0].format(
                n=n_pick, names=names)
            answer = str(total_pct)
        elif qtype == "arc_length":
            i = sub.randint(0, n_sectors - 1)
            angle = pcts[i] / 100 * 360
            arc = round(2 * math.pi * (angle / 360), 2)
            q = sub.choice(self._Q_TEMPLATES["arc_length"]).format(a=labels[i])
            answer = str(arc)
        elif qtype == "count_above_threshold":
            thr = sub.choice([10, 15, 20, 25])
            cnt = sum(1 for p in pcts if p > thr)
            q = sub.choice(self._Q_TEMPLATES["count_above_threshold"]).format(
                thr=thr)
            answer = str(cnt)
        else:
            return None

        img = self._render(labels, pcts, sub, cfg)
        return q, answer, img

    # ------------------------------------------------------------------
    # Rendering (donut + legend). We put the percentages in the LEGEND so
    # learners can read off values needed for compute questions, but we
    # never ask "what is X%" in the question text (no text leakage).
    # ------------------------------------------------------------------
    def _render(self, labels, pcts, sub, cfg=None) -> Image.Image:
        cfg = cfg or {}
        style = self._random_style()
        sc = style["figsize_scale"]
        fig_w = sub.uniform(7.0, 8.5) * sc
        fig_h = sub.uniform(6.0, 7.0) * sc
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        palette = list(style["palette"])
        # Ensure enough colors
        while len(palette) < len(labels):
            palette.extend(style["palette"])
        sub.shuffle(palette)
        colors = [palette[i % len(palette)] for i in range(len(labels))]

        # At higher levels, use near-monochromatic palette so sectors are
        # visually confusable.
        if cfg.get("similar_colors"):
            base_r = sub.uniform(0.20, 0.55)
            base_g = sub.uniform(0.20, 0.55)
            base_b = sub.uniform(0.35, 0.70)
            colors = []
            for i in range(len(labels)):
                jitter = sub.uniform(-0.12, 0.12)
                r = max(0.05, min(0.95, base_r + jitter))
                g = max(0.05, min(0.95, base_g + jitter * 0.7))
                b = max(0.05, min(0.95, base_b - jitter * 0.4))
                colors.append((r, g, b))

        start_angle = sub.randint(0, 359)
        donut_width = sub.uniform(0.33, 0.6)
        edge_color = sub.choice(["white", "#f8f9fa", "#222"])
        edge_lw = sub.uniform(1.5, 2.6)

        # Draw donut WITHOUT autopct (no percentage labels on slices)
        wedges, texts = ax.pie(
            pcts,
            labels=None,                  # no sector labels (goes in legend)
            colors=colors,
            startangle=start_angle,
            counterclock=sub.choice([True, False]),
            wedgeprops=dict(width=donut_width,
                            edgecolor=edge_color,
                            linewidth=edge_lw),
        )

        # Legend with labels AND percentages (learner uses legend to compute)
        # At higher levels hide percentages — model must estimate from angles.
        legend_style = sub.choice(["full", "values_only"])
        if cfg.get("hide_pcts_in_legend"):
            legend_labels = list(labels)
        elif legend_style == "full":
            legend_labels = [f"{l}: {p:.1f}%" for l, p in zip(labels, pcts)]
        else:
            legend_labels = [f"{l} ({p:.1f}%)" for l, p in zip(labels, pcts)]
        ax.legend(wedges, legend_labels,
                  title="Sectors",
                  loc=sub.choice(["center left", "upper right",
                                  "lower right", "lower left"]),
                  bbox_to_anchor=sub.choice([(1.02, 0.5), (1.02, 0.0),
                                             (1.02, 1.0)]),
                  fontsize=style["font_size_base"] - 1)

        titles = ["Distribution", "Sector Shares",
                  "Pie Chart", "Donut Chart",
                  "Share Breakdown", "Proportions"]
        ax.set_title(sub.choice(titles),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold",
                     fontfamily=style["font_family"])
        ax.set_aspect("equal")

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
