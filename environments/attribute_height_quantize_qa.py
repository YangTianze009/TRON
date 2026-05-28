"""
Attribute Height Quantize QA environment.

Targets a puzzle benchmark `rectangle_height_number` (Δ=−12.0, SINGLE BIGGEST regression).

Shows a horizontal row of 3-8 rectangles of varying heights.  Heights are
drawn from 3 discrete clusters (short / medium / tall) and mapped to the
integers 1 / 2 / 3.  The y-centre of each rectangle is randomly jittered
so the top pixel cannot be used as a shortcut — the model must compute
(bottom − top) per rectangle.

Two question modes:
- **SUM**  "...what is the sum of the represented numbers?"
- **COUNT** "how many tall / medium / short rectangles are there?"

The answer is a single non-negative integer.

Level 0: 3-5 rectangles, huge height gaps, tiny jitter. Class labels are
  sampled uniformly so the GT sum varies across seeds (range 3-12
  approximately — NOT a constant).
Level 5+: 6-8 rectangles, closer height gaps. Jitter is capped at
  height_gap / 3 to keep classes visually distinguishable even at L9.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# Pastel / "reference-style" rectangle fills
_FILL_PALETTE = [
    "#a8e6cf",  # pastel green (the benchmark default)
    "#ffd3b6",  # pastel orange
    "#b5ead7",  # pastel teal
    "#c3f0ca",  # lighter pastel green
    "#ffaaa5",  # soft coral
    "#ffdfba",  # peach
    "#dcedc1",  # lime cream
    "#bee1e6",  # soft aqua
]

class AttributeHeightQuantizeQA(StandaloneVisualEnv):
    """Height quantization with baseline jitter (P3 + A4)."""

    ENV_NAME = "attribute_height_quantize"

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Return difficulty config for a given level (0-9)."""
        if level <= 0:
            return {"n_rects_choices": [3, 4, 5],
                    "heights": (1.0, 3.0, 5.0), "max_jitter": 0.0,
                    "q_modes": ["sum"], "force_all_3": True}
        if level == 1:
            return {"n_rects_choices": [3, 4, 5],
                    "heights": (1.0, 2.4, 3.8), "max_jitter": 0.17,
                    "q_modes": ["sum"], "force_all_3": True}
        if level == 2:
            return {"n_rects_choices": [4, 5],
                    "heights": (1.0, 2.1, 3.3), "max_jitter": 0.24,
                    "q_modes": ["sum", "count"], "force_all_3": True}
        if level == 3:
            return {"n_rects_choices": [4, 5],
                    "heights": (1.0, 2.1, 3.3), "max_jitter": 0.31,
                    "q_modes": ["sum", "count"], "force_all_3": True}
        if level == 4:
            return {"n_rects_choices": [5, 6],
                    "heights": (1.0, 1.9, 2.8), "max_jitter": 0.28,
                    "q_modes": ["sum", "count"], "force_all_3": True}
        if level == 5:
            return {"n_rects_choices": [5, 6],
                    "heights": (1.0, 1.9, 2.8), "max_jitter": 0.28,
                    "q_modes": ["sum", "count"], "force_all_3": False}
        if level == 6:
            return {"n_rects_choices": [6, 7],
                    "heights": (1.0, 1.7, 2.5), "max_jitter": 0.22,
                    "q_modes": ["sum", "count"], "force_all_3": False}
        if level == 7:
            return {"n_rects_choices": [6, 7],
                    "heights": (1.0, 1.7, 2.5), "max_jitter": 0.22,
                    "q_modes": ["sum", "count"], "force_all_3": False}
        if level == 8:
            return {"n_rects_choices": [7, 8],
                    "heights": (1.0, 1.6, 2.3), "max_jitter": 0.19,
                    "q_modes": ["sum", "count"], "force_all_3": False}
        # level >= 9
        return {"n_rects_choices": [7, 8],
                "heights": (1.0, 1.6, 2.3), "max_jitter": 0.19,
                "q_modes": ["sum", "count"], "force_all_3": False}

    # ------------------------------------------------------------------ #
    # Question phrasing and title pools
    # ------------------------------------------------------------------ #

    _SUM_TEMPLATES = [
        "The figure shows a row of rectangles with varying heights. Short rectangles represent the number 1, medium rectangles represent 2, and tall rectangles represent 3. What is the sum of the numbers represented by all of the rectangles? Answer with a single integer.",
        "Each rectangle in the image encodes a number by its height: short=1, medium=2, tall=3. Compute the total sum of the encoded numbers. Answer with a single integer.",
        "A row of rectangles is shown. Their heights map to numbers: short -> 1, medium -> 2, tall -> 3. What is the sum of all these numbers? Answer with a single integer.",
        "Look at the rectangles. Short ones equal 1, medium ones equal 2, and tall ones equal 3. Add up all the numbers. Answer with a single integer.",
    ]

    _COUNT_TEMPLATES = [
        "The figure shows a row of rectangles with varying heights. Heights fall into three discrete classes: short, medium, and tall. How many {target} rectangles are there in the figure? Answer with a single integer.",
        "Count the number of {target} rectangles in the row shown in the image. Heights are categorized as short, medium, or tall. Answer with a single integer.",
        "In the image, rectangles are grouped by height: short, medium, tall. How many are {target}? Answer with a single integer.",
    ]

    _TITLE_VARIANTS = [
        "Row of rectangles",
        "Rectangle heights",
        "Height comparison",
        "Rectangles",
        "Bar heights",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        n_rects = rng.choice(cfg["n_rects_choices"])
        short_h, med_h, tall_h = cfg["heights"]
        max_jitter = cfg["max_jitter"]

        # Clamp jitter to safety bound
        min_gap = min(med_h - short_h, tall_h - med_h)
        max_jitter = min(max_jitter, min_gap / 3.0)

        q_mode = parameter.get("mode", rng.choice(cfg["q_modes"]))

        for _ in range(30):
            result = self._try_generate(
                rng, sub_rng, level, cfg, n_rects, short_h, med_h, tall_h,
                max_jitter, q_mode)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, sub_rng, level, cfg, n_rects,
                      short_h, med_h, tall_h, max_jitter, q_mode):
        heights_map = {1: short_h, 2: med_h, 3: tall_h}

        # Build class labels
        labels = []
        if cfg["force_all_3"] and n_rects >= 3:
            labels = [1, 2, 3]
            while len(labels) < n_rects:
                labels.append(rng.randint(1, 3))
            rng.shuffle(labels)
        elif n_rects >= 3 and rng.random() < 0.85:
            labels = [1, 2, 3]
            while len(labels) < n_rects:
                labels.append(rng.randint(1, 3))
            rng.shuffle(labels)
        else:
            for _ in range(n_rects):
                labels.append(rng.randint(1, 3))

        if len(set(labels)) < 2:
            return None

        heights = [heights_map[l] for l in labels]
        y_centres = [rng.uniform(-max_jitter, max_jitter) for _ in range(n_rects)]

        # Question + answer
        if q_mode == "sum":
            answer_int = sum(labels)
            question = sub_rng.choice(self._SUM_TEMPLATES)
        else:
            target_class = rng.choice([1, 2, 3])
            target_name = {1: "short", 2: "medium", 3: "tall"}[target_class]
            answer_int = labels.count(target_class)
            if answer_int == 0:
                return None
            question = sub_rng.choice(self._COUNT_TEMPLATES).format(target=target_name)

        title = sub_rng.choice(self._TITLE_VARIANTS)
        image = self._render(heights, y_centres, labels, level, title=title, sub_rng=sub_rng)
        self._primary_complexity_feature = n_rects + level * 2
        return question, str(answer_int), image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, heights, y_centres, labels, level,
                title: str = "Rectangles", sub_rng=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        rng = self._rng

        n = len(heights)
        fig_w = max(5.0, 0.9 * n + 1.6) * sc
        fig_h = 5.0 * sc

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        spacing = 1.2
        width = 0.75

        # Fill color variation via sub_rng
        if level <= 3:
            fill = (sub_rng or rng).choice(_FILL_PALETTE)
            fills = [fill] * n
        else:
            fills = [(sub_rng or rng).choice(_FILL_PALETTE) for _ in range(n)]

        _edge_colors = ["#2c3e50", "#1a5276", "#4a235a", "#1b4f72", "#7b241c"]
        edge = (sub_rng or rng).choice(_edge_colors)
        lw = 1.5 + (sub_rng or rng).random() * 1.0

        for i, (h, yc, lbl) in enumerate(zip(heights, y_centres, labels)):
            x0 = i * spacing
            y0 = yc - h / 2
            rect = mpatches.FancyBboxPatch(
                (x0, y0), width, h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=fills[i], edgecolor=edge, linewidth=lw, zorder=3)
            ax.add_patch(rect)

        max_h = max(heights)
        jitter_range = max(abs(y) for y in y_centres) if y_centres else 0
        ax.set_xlim(-0.6, n * spacing)
        ax.set_ylim(-max_h / 2 - jitter_range - 0.6,
                     max_h / 2 + jitter_range + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.set_title(title,
                     fontsize=fs + 2, fontweight="bold", pad=8,
                     fontfamily=ff)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ------------------------------------------------------------------ #
# Smoke-test entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = AttributeHeightQuantizeQA()
    for level in [0, 3, 6]:
        for seed in range(3):
            ok = env.generate(seed=seed * 10 + level,
                              parameter={"level": level})
            if not ok:
                print(f"FAILED seed={seed} level={level}")
                continue
            img = env.render()
            img.save(f"/tmp/env_check/attribute_height_quantize_seed{seed}_L{level}.png")
            print(f"seed={seed} level={level}")
            print("Q:", env.get_instruction())
            print("A:", env._answer)
            print()
