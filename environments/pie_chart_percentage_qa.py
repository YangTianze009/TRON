"""
Pie Chart Percentage QA (batch 3, 2026-04-14).

Target: figure QA / chart percentage reading. The model reads
the percentage of a specific slice or converts to a count given total.

Format: constant short numeric (integer percentage).

Difficulty axes:
  A) Pattern A: n_slices (3..8).
  B) Pattern D: at high levels, slice sizes are close (small differences).
  C) Pattern G: percentage labels on slices hidden at L>=3.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

class PieChartPercentageQA(StandaloneVisualEnv):
    ENV_NAME = "pie_chart_percentage"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_slices": 3 + level // 2,             # 3..7
            "close_values": level >= 7,             # slices differ only slightly
            "show_labels": level <= 4,              # show % labels through L4
            "start_angle_jitter": level >= 5,
            "snap_increments": max(1, 5 - level // 2),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_slices"] + (2 if cfg["close_values"] else 0)

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_slices"]

        # Sample values that sum to 100
        if cfg["close_values"]:
            base = 100 // n
            remainder = 100 - base * n
            vals = [base + (1 if i < remainder else 0) for i in range(n)]
            for _ in range(n):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                delta = rng.randint(1, 4)
                if vals[i] - delta >= 3:
                    vals[i] -= delta
                    vals[j] += delta
        else:
            # Random splits of 100 in snap increments
            snap = cfg["snap_increments"]
            # Use stars-and-bars via cut points
            steps = 100 // snap
            cuts = sorted(rng.sample(range(1, steps), n - 1))
            chunks = []
            prev = 0
            for c in cuts:
                chunks.append((c - prev) * snap)
                prev = c
            chunks.append((steps - prev) * snap)
            vals = chunks
            vals = [max(snap, v) for v in vals]
            # Renormalise to 100
            s = sum(vals)
            if s != 100:
                vals[0] += 100 - s

        vals = [int(v) for v in vals]
        if sum(vals) != 100 or min(vals) <= 0:
            return None

        labels = _LABELS[:n]

        # Pick a slice to ask about
        ask_idx = rng.randint(0, n - 1)
        asked_label = labels[ask_idx]
        answer = str(vals[ask_idx])

        templates = [
            f"In the pie chart, what percentage does slice {asked_label} represent? Answer with an integer (no % sign needed).",
            f"Looking at the pie chart, what is the percentage value of slice {asked_label}? Provide an integer (no percent sign).",
            f"Read the pie chart and report the percentage assigned to slice {asked_label}. Integer only.",
            f"What percentage of the pie chart does the slice labelled {asked_label} occupy? Answer as an integer (no % sign).",
            f"Estimate the share of slice {asked_label} as a whole-number percentage from the pie chart. (No % sign in answer.)",
        ]
        q = rng.choice(templates)

        image = self._render(labels, vals, cfg, rng)
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, q, answer, suffix="%", rate=0.30)
        if wrapped is not None:
            q, answer = wrapped
        return q, answer, image

    def _render(self, labels, vals, cfg, rng=None) -> Image.Image:
        if rng is None:
            rng = random.Random(0)
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]
        # Shuffle palette to vary slice colors per seed
        pal = list(palette)
        rng.shuffle(pal)
        colors = [pal[i % len(pal)] for i in range(len(vals))]

        fig, ax = plt.subplots(figsize=(6.5 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        if cfg["start_angle_jitter"]:
            startangle = rng.choice([0, 30, 45, 60, 120, 180, 270, 315])
        else:
            startangle = 90
        counter = rng.choice([True, False])

        if cfg["show_labels"]:
            autopct = "%1.0f%%"
        else:
            autopct = None

        # Optional donut style
        wedge_props = {"edgecolor": "#000", "linewidth": 1.3}
        if rng.random() < 0.3:
            # Donut style — narrower wedges
            wedge_props["width"] = rng.uniform(0.4, 0.65)

        ax.pie(vals, labels=labels, colors=colors, autopct=autopct,
               startangle=startangle, counterclock=counter,
               textprops={"fontsize": fs + 1, "fontweight": "bold"},
               wedgeprops=wedge_props)
        title_pool = [
            "Pie chart — slice percentages",
            "Slice Percentages",
            "Distribution Pie Chart",
            "Percentage Breakdown",
            "Pie Chart",
        ]
        ax.set_title(rng.choice(title_pool), fontsize=fs + 1)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = PieChartPercentageQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[pie_chart_percentage L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"pie_chart_percentage_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[pie_chart_percentage L{level} s{s}] A={env._answer}")
