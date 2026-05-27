"""
Partial Occluded Cube Count QA environment.

Template: isometric_counting_qa.py (reuses _iso_project).

Goal: cube counting . The student sees an
isometric cube stack where some cubes are partially or fully occluded
(for example, hidden behind a taller front column). They must count
all cubes, including those needed to support the visible ones.

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): n_cubes_total = 4 + level       -> 4..13
  Axis 2           : structure_complexity            simple block ->
                                                     non-convex
  Axis 3 (optional): option_gap = max(1, 3 - level // 3) -> 3..1

Output format is constant: 4-option integer MCQ, single letter.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _cube_face_palette(style, rng):
    """BUGFIX: safe 3-color face palette for isometric cubes (top > right > left
    luminance). Avoids #000000 (black faces blending into shadow) and clamps
    base lightness so faces always differ clearly regardless of palette."""
    import colorsys
    pal = [c for c in style['palette']
           if c.lower() not in ('#000000', '#010101', '#0a0a0a', '#ffffff',
                                '#fefefe', '#f1faee')]
    if not pal:
        pal = ['#5dade2', '#48c9b0', '#ec7063', '#f4d03f']
    base = rng.choice(pal)
    h = base.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    base_l = 0.55
    ss = min(1.0, max(0.45, ss))
    top_rgb = colorsys.hls_to_rgb(hh, min(0.85, base_l + 0.22), ss)
    right_rgb = colorsys.hls_to_rgb(hh, base_l, ss)
    left_rgb = colorsys.hls_to_rgb(hh, max(0.18, base_l - 0.25), ss)
    def _hx(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(max(0, min(1, rgb[0])) * 255),
            int(max(0, min(1, rgb[1])) * 255),
            int(max(0, min(1, rgb[2])) * 255))
    return [_hx(top_rgb), _hx(left_rgb), _hx(right_rgb)]

def _support_required(cubes):
    """Ensure every cube has support below it. Add supporting cubes as needed."""
    cube_set = set(cubes)
    stack = list(cubes)
    while stack:
        x, y, z = stack.pop()
        if z > 0 and (x, y, z - 1) not in cube_set:
            cube_set.add((x, y, z - 1))
            stack.append((x, y, z - 1))
    return list(cube_set)

class PartialOccludedCubeCountQA(StandaloneVisualEnv):
    ENV_NAME = "partial_occluded_cube_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        # Distinct per-level (target_cubes, complexity, option_gap).
        # Smoother ramp at low levels so L1/L2 also reach >=50% pass rate
        # — without that the model gets no training reward in mid-easy
        # cases, even though the task is conceptually simple.
        if level == 0:
            return dict(target_cubes=3, complexity="block", option_gap=3, force_flat=False)
        if level == 1:
            return dict(target_cubes=4, complexity="block", option_gap=3, force_flat=True)
        if level == 2:
            return dict(target_cubes=5, complexity="block", option_gap=3, force_flat=True)
        if level == 3:
            return dict(target_cubes=6, complexity="stair", option_gap=2, force_flat=False)
        if level == 4:
            return dict(target_cubes=7, complexity="stair", option_gap=2, force_flat=False)
        if level == 5:
            return dict(target_cubes=8, complexity="l_shape", option_gap=2, force_flat=False)
        if level == 6:
            return dict(target_cubes=10, complexity="l_shape", option_gap=2, force_flat=False)
        if level == 7:
            return dict(target_cubes=11, complexity="non_convex", option_gap=1, force_flat=False)
        if level == 8:
            return dict(target_cubes=12, complexity="non_convex", option_gap=1, force_flat=False)
        return dict(target_cubes=14, complexity="non_convex", option_gap=1, force_flat=False)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["target_cubes"]

        for _ in range(25):
            try:
                result = self._try_generate(sub_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        # L0 special path: flat 2D grid — no 3D isometric rendering
        if level == 0:
            return self._generate_l0_flat(rng, cfg)

        target = cfg["target_cubes"]
        comp = cfg["complexity"]
        cubes = []
        if comp == "block":
            # Random 1xN / 2xN / row+tower / small 2x2 patterns
            if cfg.get("force_flat"):
                # Flat-only at low levels so the iso silhouette tells the
                # full story (no occluded cubes at all → just count).
                sub_comp = rng.choice(["line", "twin_line", "L_flat"])
            else:
                sub_comp = rng.choice(["line", "twin_line", "small_block", "row_tower"])
            if sub_comp == "line":
                for i in range(target):
                    cubes.append((i, 0, 0))
            elif sub_comp == "twin_line":
                rows = 2
                cols = max(2, target // rows + (target % rows))
                for i in range(cols):
                    for j in range(rows):
                        if len(cubes) < target:
                            cubes.append((i, j, 0))
            elif sub_comp == "L_flat":
                arm = max(2, target - 1)
                for i in range(arm):
                    cubes.append((i, 0, 0))
                while len(cubes) < target:
                    cubes.append((0, len(cubes) - arm + 1, 0))
            elif sub_comp == "small_block":
                for i in range(2):
                    for j in range(2):
                        if len(cubes) < target:
                            cubes.append((i, j, 0))
                for k in range(1, target - len(cubes) + 1):
                    if len(cubes) < target:
                        cubes.append((0, 0, k))
            else:  # row_tower
                row_n = max(2, target - 2)
                for i in range(row_n):
                    cubes.append((i, 0, 0))
                while len(cubes) < target:
                    cubes.append((0, 0, len(cubes) - row_n + 1))
        elif comp == "stair":
            # Pick among 3 stair variants
            sub_comp = rng.choice(["up", "zigzag", "double"])
            if sub_comp == "up":
                x = 0
                z = 0
                while len(cubes) < target:
                    cubes.append((x, 0, z))
                    x += 1
                    if rng.random() < 0.4:
                        z += 1
                        if len(cubes) < target:
                            cubes.append((x - 1, 0, z))
            elif sub_comp == "zigzag":
                x = 0
                y = 0
                z = 0
                while len(cubes) < target:
                    cubes.append((x, y, z))
                    delta = rng.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
                    x += delta[0]
                    y += delta[1]
                    z += delta[2]
            else:  # double stair
                for i in range(target // 2 + 1):
                    cubes.append((i, 0, i))
                    if len(cubes) < target:
                        cubes.append((i, 1, i))
                cubes = cubes[:target]
        elif comp == "l_shape":
            # L in 2D floor + partial second layer
            gx = max(2, target // 3)
            for i in range(gx):
                cubes.append((i, 0, 0))
            for j in range(1, min(3, target - gx)):
                cubes.append((0, j, 0))
            # add second layer
            for k in range(1, max(1, (target - len(cubes)) // 2 + 1)):
                if len(cubes) < target:
                    cubes.append((0, 0, k))
            while len(cubes) < target:
                cubes.append((rng.randint(0, gx - 1), rng.randint(0, 1), 1))
        else:
            # non-convex: random walk
            cubes = [(0, 0, 0)]
            while len(cubes) < target:
                s = rng.choice(cubes)
                dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0),
                                          (0, 1, 0), (0, -1, 0),
                                          (0, 0, 1)])
                nc = (s[0] + dx, s[1] + dy, s[2] + dz)
                if nc not in cubes and nc[0] >= 0 and nc[1] >= 0 and nc[2] >= 0:
                    cubes.append(nc)

        # Deduplicate
        cubes = list(set(cubes))
        # Add supporting cubes (if a cube sits at z>0, a cube must be below)
        cubes = _support_required(cubes)
        total = len(cubes)
        if total < 2:
            return None

        # Build MCQ
        gap = cfg["option_gap"]
        options_pool = [total + k for k in (-2 * gap, -gap, gap, 2 * gap)
                        if total + k >= 0]
        if total not in options_pool:
            options_pool.append(total)
        options_pool = list(set(options_pool))
        rng.shuffle(options_pool)
        options = [total]
        for v in options_pool:
            if len(options) >= 4:
                break
            if v != total and v not in options and v >= 1:
                options.append(v)
        while len(options) < 4:
            v = total + rng.randint(1, 3)
            if v not in options:
                options.append(v)
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(total))

        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        if cfg.get("force_flat"):
            # No occluded cubes at this level — phrase it as a simple count.
            stem = self._rng.choice([
                "How many unit cubes are in the structure shown in the image? Just count what you see.",
                "Count the total number of unit cubes in this 3D structure (every cube is fully visible from this iso view).",
                "How many unit cubes make up the figure? Each cube is visible from this isometric angle.",
            ])
        else:
            stem = self._rng.choice([
                "How many unit cubes are in the stack shown in the image? Include any hidden cubes required to support the visible ones.",
                "Count the total number of unit cubes in the 3D structure, including any occluded cubes that must exist to support the visible cubes.",
                "The image shows a cube stack with some cubes possibly hidden from view. How many unit cubes are there in total?",
            ])
        question = f"{stem} Options: {opts_text}. Answer with a single letter."

        image = self._render(cubes, rng)
        return question, correct_letter, image

    def _generate_l0_flat(self, rng, cfg):
        """L0: flat 2D grid with labeled cube count — no 3D isometric.
        Shows a simple 2D arrangement of colored squares and asks 'how many?'"""
        target = cfg["target_cubes"]
        # Generate a simple flat 2D layout
        layouts = [
            # 4 cubes: 2x2 block
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            # 4 cubes: line
            [(0, 0), (0, 1), (0, 2), (0, 3)],
            # 4 cubes: L-shape
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            # 4 cubes: T-shape
            [(0, 0), (0, 1), (0, 2), (1, 1)],
        ]
        cells = rng.choice(layouts)
        total = len(cells)

        # MCQ
        gap = cfg["option_gap"]
        options = [total]
        for d in [gap, -gap, 2 * gap, -2 * gap]:
            v = total + d
            if v >= 1 and v not in options:
                options.append(v)
            if len(options) >= 4:
                break
        while len(options) < 4:
            v = total + rng.randint(1, 5)
            if v not in options:
                options.append(v)
        options = options[:4]
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(total))
        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        stem = self._rng.choice([
            "How many colored squares are shown in the image?",
            "Count the total number of unit squares in the 2D shape below.",
            "How many unit cubes are in this flat arrangement?",
        ])
        question = f"{stem} Options: {opts_text}. Answer with a single letter."

        # Render flat 2D
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        rng.shuffle(palette)
        fig, ax = plt.subplots(figsize=(5 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        max_r, max_c = max(rows) + 1, max(cols) + 1
        from matplotlib.patches import Rectangle
        for r, c in cells:
            rect = Rectangle((c, max_r - 1 - r), 1, 1,
                              facecolor=palette[0], edgecolor="#222",
                              linewidth=2)
            ax.add_patch(rect)
        ax.set_xlim(-0.3, max_c + 0.3)
        ax.set_ylim(-0.3, max_r + 0.3)
        ax.set_title(rng.choice(["Cube Count", "Count the Cubes",
                                  "How Many Squares?"]),
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        image = self.fig_to_pil(fig, dpi=style["dpi"])
        return question, correct_letter, image

    def _render(self, cubes, sub_rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect('equal')
        ax.axis("off")

        # Safe shaded palette (top/left/right always distinguishable).
        top_c, left_c, right_c = _cube_face_palette(style, sub_rng)
        edge_col = sub_rng.choice(["black", "#111", "#0a0a0a", "#1a1a2e"])
        lw = sub_rng.uniform(1.0, 1.4)

        cubes_sorted = sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))
        for (x, y, z) in cubes_sorted:
            pts = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=top_c,
                                 edgecolor=edge_col, lw=lw))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=left_c,
                                 edgecolor=edge_col, lw=lw))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=right_c,
                                 edgecolor=edge_col, lw=lw))

        pts = []
        for (x, y, z) in cubes:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        pts.append(_iso_project(x + dx, y + dy, z + dz))
        arr = np.array(pts)
        mg = 0.6
        ax.set_xlim(arr[:, 0].min() - mg, arr[:, 0].max() + mg)
        ax.set_ylim(arr[:, 1].min() - mg, arr[:, 1].max() + mg)
        title = sub_rng.choice([
            "Cube count",
            "3D cube stack",
            "Unit-cube structure",
            "Cube assembly",
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
    env = PartialOccludedCubeCountQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[partial_occluded_cube_count] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/partial_occluded_cube_count_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | ncubes={env._primary_complexity_feature}")
