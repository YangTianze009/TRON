"""
Arrow Count Flow QA environment.

Template: figure_counting_qa.py + counting_objects_qa.py.

Goal: count arrows pointing in a specific direction (or of a specific
color) in a grid. Targets counting  and visual-perception
Counting (teach two-attribute filtering).

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): grid_size = 5 + level // 2      -> 5..9
  Axis 2           : n_directions_used = 4 + level // 3 -> 4..7
  Axis 3 (optional): conjunction_required = level >= 5

Output format is constant: 4-option integer MCQ, single letter.
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

_DIRECTION_VECTORS = {
    "right": (1, 0),
    "left": (-1, 0),
    "up": (0, 1),
    "down": (0, -1),
    "upper-right": (1, 1),
    "upper-left": (-1, 1),
    "lower-right": (1, -1),
    "lower-left": (-1, -1),
}

_DIR_NAMES = list(_DIRECTION_VECTORS.keys())

def _arrow_shape(cx, cy, direction, length=0.7):
    """Return list of 2D points for an arrow."""
    dx, dy = direction
    norm = math.hypot(dx, dy)
    ux, uy = dx / norm, dy / norm
    # perpendicular
    px, py = -uy, ux
    half = length * 0.5
    tail = (cx - ux * half, cy - uy * half)
    head = (cx + ux * half, cy + uy * half)
    h_left = (cx + ux * half * 0.4 + px * half * 0.35,
              cy + uy * half * 0.4 + py * half * 0.35)
    h_right = (cx + ux * half * 0.4 - px * half * 0.35,
               cy + uy * half * 0.4 - py * half * 0.35)
    return tail, head, h_left, h_right

class ArrowCountFlowQA(StandaloneVisualEnv):
    ENV_NAME = "arrow_count_flow"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "grid_size": 5 + level // 2,              # 5..9
            "n_directions": 4 + level // 3,           # 4..7
            "conjunction": level >= 5,
            "option_gap": max(1, 3 - level // 3),     # 3..1
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["grid_size"] ** 2

        for _ in range(20):
            r = self._try_once(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_once(self, rng, cfg):
        g = cfg["grid_size"]
        dirs = _DIR_NAMES[: cfg["n_directions"]]
        palette = ["#e74c3c", "#3498db", "#27ae60", "#f1c40f"]

        arrows = []
        for i in range(g):
            for j in range(g):
                d_name = rng.choice(dirs)
                color_idx = rng.randint(0, 3)
                arrows.append((i, j, d_name, color_idx))

        # Decide question
        if cfg["conjunction"]:
            target_color_idx = rng.randint(0, 2)
            target_dir = rng.choice(dirs)
            count = sum(1 for a in arrows
                        if a[2] == target_dir and a[3] == target_color_idx)
            color_names = ["red", "blue", "green", "yellow"]
            question_body = f"{color_names[target_color_idx]} arrows pointing {target_dir}"
        else:
            target_dir = rng.choice(dirs)
            count = sum(1 for a in arrows if a[2] == target_dir)
            target_color_idx = None
            question_body = f"arrows pointing {target_dir}"

        if count < 1:
            return None

        gap = cfg["option_gap"]
        options_pool = [count + k for k in (-2 * gap, -gap, gap, 2 * gap)
                        if count + k >= 0]
        if count not in options_pool:
            options_pool.append(count)
        options_pool = list(set(options_pool))
        rng.shuffle(options_pool)
        options = [count]
        for v in options_pool:
            if len(options) >= 4:
                break
            if v != count and v not in options and v >= 0:
                options.append(v)
        while len(options) < 4:
            v = count + rng.randint(1, 3)
            if v not in options:
                options.append(v)
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(count))

        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        q_stems = [
            f"How many {question_body} are in the image?",
            f"Count the total number of {question_body} shown in the grid.",
            f"Looking at the arrow grid, how many {question_body} can you find?",
            f"Determine the number of {question_body} in the figure.",
        ]
        question = (
            f"{rng.choice(q_stems)} "
            f"Options: {opts_text}. Answer with a single letter."
        )
        image = self._render(arrows, g, palette)
        return question, correct_letter, image

    def _render(self, arrows, g, palette):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(-0.5, g - 0.5)
        ax.set_ylim(-0.5, g - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        for (i, j, d_name, color_idx) in arrows:
            color = palette[color_idx]
            vec = _DIRECTION_VECTORS[d_name]
            tail, head, hl, hr = _arrow_shape(i, j, vec, length=0.7)
            ax.plot([tail[0], head[0]], [tail[1], head[1]], color=color,
                    linewidth=2.0)
            head_poly = plt.Polygon([head, hl, hr], closed=True,
                                     facecolor=color, edgecolor=color)
            ax.add_patch(head_poly)

        title_pool = ["Count the Arrows", "Arrow Grid",
                      "Directional Arrows", "Arrow Flow", "Grid of Arrows"]
        ax.set_title(self._rng.choice(title_pool),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = ArrowCountFlowQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[arrow_count_flow] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/arrow_count_flow_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | cells={env._primary_complexity_feature}")
