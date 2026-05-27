"""
Shape Instance Count QA environment.

Template: counting_objects_qa.py + figure_counting_qa.py.

Goal: count instances of a particular (color, shape) combination in a
scatter of mixed shapes. Targets MMBench CP counting, visual-perception Counting,
counting.

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): n_total_shapes = 5 + level * 2   -> 5..23
  Axis 2           : n_shape_kinds_used = 3 + level // 2 -> 3..7
                     + n_colors_used = 2 + level // 3 -> 2..5
  Axis 3 (optional): clutter_density                 shapes may touch at L>=5

Output format is constant: 4-option integer MCQ, single letter.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_COLOR_HEX = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#27ae60",
    "orange": "#e67e22",
    "purple": "#8e44ad",
    "yellow": "#f1c40f",
    "pink": "#e91e63",
}
_COLOR_NAMES = list(_COLOR_HEX.keys())
_SHAPE_NAMES = ["circle", "square", "triangle", "star", "pentagon",
                "hexagon", "diamond"]

def _polygon_path(cx, cy, r, n, rotation=0):
    verts = []
    for i in range(n):
        a = math.pi / 2 + 2 * math.pi * i / n + rotation
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _star_path(cx, cy, r_out):
    n = 5
    r_in = r_out * 0.45
    verts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = math.pi / 2 + math.pi * i / n
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (2 * n - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _diamond_path(cx, cy, size):
    verts = [(cx, cy + size), (cx + size, cy), (cx, cy - size),
             (cx - size, cy), (cx, cy + size)]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO,
             mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

class ShapeInstanceCountQA(StandaloneVisualEnv):
    ENV_NAME = "shape_instance_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_total": 5 + level * 2,                  # 5..23
            "n_kinds": min(len(_SHAPE_NAMES), 3 + level // 2),
            "n_colors": min(len(_COLOR_NAMES), 2 + level // 3),
            "min_sep": max(0.05, 0.30 - 0.035 * level),
            "option_gap": max(1, 3 - level // 3),      # 3..1
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_total"]

        for _ in range(20):
            r = self._try_once(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_once(self, rng, cfg, level):
        n_total = cfg["n_total"]
        kinds = rng.sample(_SHAPE_NAMES, cfg["n_kinds"])
        colors = rng.sample(_COLOR_NAMES, cfg["n_colors"])

        # Pick target
        target_shape = rng.choice(kinds)
        target_color = rng.choice(colors)

        # Choose how many target instances
        n_target = rng.randint(max(1, n_total // 6), max(2, n_total // 3 + 1))
        if n_target < 1:
            n_target = 1
        if n_target >= n_total:
            n_target = n_total - 1
        if n_target < 1:
            return None

        objs = []
        positions = []
        for _ in range(n_total):
            for _ in range(80):
                x = rng.uniform(5, 95)
                y = rng.uniform(5, 95)
                ok = True
                for (px, py) in positions:
                    if math.hypot(px - x, py - y) < 100 * cfg["min_sep"]:
                        ok = False
                        break
                if ok:
                    positions.append((x, y))
                    break
            else:
                positions.append((rng.uniform(5, 95), rng.uniform(5, 95)))

        # Assign
        for i, (x, y) in enumerate(positions):
            if i < n_target:
                objs.append((x, y, target_color, target_shape))
            else:
                c = rng.choice([cc for cc in colors if cc != target_color] or colors)
                s = rng.choice([ss for ss in kinds if ss != target_shape] or kinds)
                if rng.random() < 0.3:
                    c = target_color  # same color different shape
                if rng.random() < 0.3:
                    s = target_shape  # same shape different color
                # Avoid accidental target duplicate
                if c == target_color and s == target_shape:
                    s = rng.choice([ss for ss in kinds if ss != target_shape] or kinds)
                objs.append((x, y, c, s))

        # Recompute target count
        actual_count = sum(1 for (_x, _y, c, s) in objs
                           if c == target_color and s == target_shape)

        gap = cfg["option_gap"]
        options_pool = [actual_count + k for k in (-2 * gap, -gap, gap, 2 * gap)
                        if actual_count + k >= 0]
        if actual_count not in options_pool:
            options_pool.append(actual_count)
        options_pool = list(set(options_pool))
        rng.shuffle(options_pool)
        options = [actual_count]
        for v in options_pool:
            if len(options) >= 4:
                break
            if v != actual_count and v not in options and v >= 0:
                options.append(v)
        while len(options) < 4:
            v = actual_count + rng.randint(1, 3)
            if v not in options:
                options.append(v)
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(actual_count))

        plural = target_shape + ("s" if not target_shape.endswith("s") else "es")
        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        q_stems = [
            f"How many {target_color} {plural} are in the image?",
            f"Count the total number of {target_color} {plural} shown.",
            f"How many {plural} of {target_color} color appear in the figure?",
            f"Determine the number of {target_color} {plural} in the scatter of shapes.",
        ]
        question = (
            f"{rng.choice(q_stems)} "
            f"Options: {opts_text}. Answer with a single letter."
        )
        rng.shuffle(objs)
        image = self._render(objs)
        return question, correct_letter, image

    def _render(self, objs):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.axis("off")

        size = 3.0
        for (x, y, c, s) in objs:
            hex_c = _COLOR_HEX[c]
            if s == "circle":
                patch = mpatches.Circle((x, y), size, facecolor=hex_c,
                                         edgecolor="black", linewidth=0.6)
            elif s == "square":
                patch = mpatches.Rectangle((x - size, y - size), size * 2, size * 2,
                                            facecolor=hex_c, edgecolor="black",
                                            linewidth=0.6)
            elif s == "triangle":
                patch = mpatches.PathPatch(_polygon_path(x, y, size * 1.1, 3),
                                            facecolor=hex_c, edgecolor="black",
                                            linewidth=0.6)
            elif s == "pentagon":
                patch = mpatches.PathPatch(_polygon_path(x, y, size * 1.1, 5),
                                            facecolor=hex_c, edgecolor="black",
                                            linewidth=0.6)
            elif s == "hexagon":
                patch = mpatches.PathPatch(_polygon_path(x, y, size * 1.1, 6),
                                            facecolor=hex_c, edgecolor="black",
                                            linewidth=0.6)
            elif s == "star":
                patch = mpatches.PathPatch(_star_path(x, y, size * 1.3),
                                            facecolor=hex_c, edgecolor="black",
                                            linewidth=0.6)
            else:  # diamond
                patch = mpatches.PathPatch(_diamond_path(x, y, size),
                                            facecolor=hex_c, edgecolor="black",
                                            linewidth=0.6)
            ax.add_patch(patch)

        title_pool = ["Shape Instance Counting", "Count the Shapes",
                      "Shape Scatter", "Mixed Shapes", "Object Counting"]
        ax.set_title(self._rng.choice(title_pool),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = ShapeInstanceCountQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[shape_instance_count] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/shape_instance_count_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | n={env._primary_complexity_feature}")
