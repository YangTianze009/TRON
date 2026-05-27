"""
Dense 2D Count Warmup QA environment.

Template: counting_objects_qa.py + precise_counting_qa.py.

Goal: direct fix for counting . A bounded
canvas is filled with N shapes. The target class (e.g. "red stars") is
counted; distractor grey shapes are scattered around. At L0 items are
well-separated, at L9 they may touch/overlap.

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): n_total_items = 5 + level * 3    -> 5..32
  Axis 2           : clutter_density = 0..0.3         (items overlap)
  Axis 3           : distractor_option_gap = max(1, 4 - level // 2) -> 4..1

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

_COLORS = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#27ae60",
    "orange": "#e67e22",
    "purple": "#8e44ad",
}
_SHAPES = ["circle", "square", "triangle", "star", "diamond"]

def _star_path(cx, cy, r_out, r_in=None, n_points=5):
    r_in = r_in if r_in is not None else r_out * 0.45
    verts = []
    for i in range(2 * n_points):
        r = r_out if i % 2 == 0 else r_in
        a = math.pi / 2 + math.pi * i / n_points
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (2 * n_points - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _diamond_path(cx, cy, size):
    verts = [(cx, cy + size), (cx + size, cy), (cx, cy - size), (cx - size, cy), (cx, cy + size)]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
             mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _triangle_path(cx, cy, size):
    h = size * math.sqrt(3) / 2
    verts = [(cx, cy + h * 0.67), (cx - size / 2, cy - h * 0.33),
             (cx + size / 2, cy - h * 0.33), (cx, cy + h * 0.67)]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO,
             mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

class Dense2DCountWarmupQA(StandaloneVisualEnv):
    ENV_NAME = "dense_2d_count_warmup"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        # Fixed: wider distractor gaps at L6-L7 (gap=2 instead of 1) so
        # MCQ options are more distinguishable. Also slower item count ramp.
        # Fixes L6=0.00 vs L9=0.20 inversion.
        table = {
            0: (5,  0.40, 4, 0.20),
            1: (7,  0.36, 4, 0.22),
            2: (10, 0.32, 3, 0.25),
            3: (13, 0.28, 3, 0.28),
            4: (16, 0.24, 2, 0.32),
            5: (19, 0.20, 2, 0.35),
            6: (22, 0.16, 2, 0.40),
            7: (25, 0.13, 2, 0.45),
            8: (28, 0.10, 1, 0.50),
            9: (32, 0.08, 1, 0.55),
        }
        n, sep, gap, frac = table[level]
        return dict(n_total_items=n, min_sep=sep, option_gap=gap,
                    distractor_frac=frac)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_total_items"]

        n_total = cfg["n_total_items"]
        target_color = sub_rng.choice(list(_COLORS.keys()))
        target_shape = sub_rng.choice(_SHAPES)

        n_distractors = max(1, int(n_total * cfg["distractor_frac"]))
        n_target = max(2, n_total - n_distractors)
        if n_target < 2 or n_target > n_total - 1:
            n_target = max(2, min(n_total - 1, n_target))

        positions = []
        for _ in range(n_total):
            for _ in range(80):
                x = sub_rng.uniform(5, 95)
                y = sub_rng.uniform(5, 95)
                ok = True
                for (px, py) in positions:
                    if math.hypot(px - x, py - y) < 100 * cfg["min_sep"]:
                        ok = False
                        break
                if ok:
                    positions.append((x, y))
                    break
            else:
                positions.append((sub_rng.uniform(5, 95), sub_rng.uniform(5, 95)))

        objs = []
        for i, (x, y) in enumerate(positions):
            if i < n_target:
                objs.append((x, y, target_color, target_shape))
            else:
                dc = sub_rng.choice([c for c in _COLORS if c != target_color])
                ds = sub_rng.choice([s for s in _SHAPES if s != target_shape])
                objs.append((x, y, dc, ds))
        sub_rng.shuffle(objs)

        gap = cfg["option_gap"]
        options_pool = [n_target + k for k in (-2 * gap, -gap, gap, 2 * gap)
                        if n_target + k >= 0]
        options_pool = list(set(options_pool))
        if n_target not in options_pool:
            options_pool.append(n_target)
        sub_rng.shuffle(options_pool)
        options = [n_target]
        for v in options_pool:
            if len(options) >= 4:
                break
            if v != n_target and v not in options and v >= 0:
                options.append(v)
        while len(options) < 4:
            v = n_target + sub_rng.randint(1, 5)
            if v not in options:
                options.append(v)
        sub_rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(n_target))

        plural = target_shape + "s"
        if target_shape == "diamond":
            plural = "diamonds"
        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        stem = self._rng.choice([
            f"How many {target_color} {plural} are in the image?",
            f"Count the {target_color} {plural} shown in the image.",
            f"The image shows several shapes. How many of them are {target_color} {plural}?",
        ])
        question = (
            f"{stem} Options: {opts_text}. Answer with a single letter."
        )
        image = self._render(objs, target_color, target_shape, sub_rng)
        return question, correct_letter, image

    def _render(self, objs, target_color, target_shape, sub_rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.axis("off")

        size = sub_rng.uniform(2.5, 3.2)
        edge_col = sub_rng.choice(["black", "#111", "#1a1a2e", "#222"])
        edge_w = sub_rng.uniform(0.5, 0.8)
        for (x, y, c, s) in objs:
            hex_c = _COLORS[c]
            if s == "circle":
                patch = mpatches.Circle((x, y), size, facecolor=hex_c,
                                         edgecolor=edge_col, linewidth=edge_w)
            elif s == "square":
                patch = mpatches.Rectangle((x - size, y - size), size * 2, size * 2,
                                            facecolor=hex_c, edgecolor=edge_col,
                                            linewidth=edge_w)
            elif s == "triangle":
                patch = mpatches.PathPatch(
                    _triangle_path(x, y, size * 2),
                    facecolor=hex_c, edgecolor=edge_col, linewidth=edge_w)
            elif s == "star":
                patch = mpatches.PathPatch(
                    _star_path(x, y, size * 1.25),
                    facecolor=hex_c, edgecolor=edge_col, linewidth=edge_w)
            else:
                patch = mpatches.PathPatch(
                    _diamond_path(x, y, size),
                    facecolor=hex_c, edgecolor=edge_col, linewidth=edge_w)
            ax.add_patch(patch)

        title = sub_rng.choice([
            f"Count the {target_color} {target_shape}s",
            f"{target_color.capitalize()} {target_shape}s — how many?",
            f"Find and count {target_color} {target_shape}s",
            f"Counting {target_color} {target_shape}s",
        ])
        ax.set_title(title,
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = Dense2DCountWarmupQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[dense_2d_count_warmup] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/dense_2d_count_warmup_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | n={env._primary_complexity_feature}")
