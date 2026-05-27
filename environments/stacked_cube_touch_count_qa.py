"""
Stacked Cube Touch Count QA — pure single-letter MCQ.

2026-05-05 R5 Batch 4 (TOO-HARD ease + restructure):
Old design (435 LOC) was free-form numeric over arbitrary cube counts. R3 = 5%.
The model could not learn anything useful at L3+ where stacks have 6+ cubes.

REWRITE strategy:
  - L0/L1: 2-3 cubes in a row, BIG render, every cube labeled A/B/C, 4-MCQ
           {0, 1, 2, 3} touches around the marked cube. Should be near-trivial.
  - L2-L4: 4-5 cubes, simple shapes (row, L, T), 4-MCQ.
  - L5-L7: 6-8 cubes, irregular stacks, 5-MCQ with 15-25% trap.
  - L8/L9: 10+ cubes + 35-40% trap "E. No correct answer".

"Touch" = face-adjacent only (Manhattan distance 1), to remove ambiguity vs the
old face+edge convention. The labelled marked cube is highlighted with a thick
red border; question asks how many other cubes share a face with the marked one.
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


# ====================================================================== #
# Geometry helpers
# ====================================================================== #

def _iso_project(x, y, z):
    """Standard isometric projection."""
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy


def _cube_face_palette(style, rng):
    """Safe shaded face palette (top/left/right) for isometric cubes."""
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


def _count_face_touches(cubes: List[Tuple[int, int, int]],
                        marked: Tuple[int, int, int]) -> int:
    """How many other cubes share a face with `marked`. Face-adjacent only."""
    cs = set(cubes)
    cnt = 0
    mx, my, mz = marked
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        if (mx + dx, my + dy, mz + dz) in cs and \
           (mx + dx, my + dy, mz + dz) != marked:
            cnt += 1
    return cnt


# ====================================================================== #
# Stack generators per level
# ====================================================================== #

def _gen_row(rng, n) -> List[Tuple[int, int, int]]:
    """A simple row of n cubes along the x axis."""
    return [(i, 0, 0) for i in range(n)]


def _gen_L(rng, n) -> List[Tuple[int, int, int]]:
    """L shape with arm + foot in y direction."""
    arm = max(2, n - 1)
    cubes = [(i, 0, 0) for i in range(arm)]
    cubes.append((arm - 1, 1, 0))
    while len(cubes) < n:
        cubes.append((arm - 1, 1, len(cubes) - arm))
    return cubes[:n]


def _gen_T(rng, n) -> List[Tuple[int, int, int]]:
    """T shape: row of 3 + stem."""
    cubes = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    while len(cubes) < n:
        cubes.append((1, 1, len(cubes) - 3))
    return cubes[:n]


def _gen_random(rng, n) -> List[Tuple[int, int, int]]:
    """Random connected polycube of size n."""
    cubes = {(0, 0, 0)}
    attempts = 0
    while len(cubes) < n and attempts < n * 20:
        attempts += 1
        seed_cube = rng.choice(list(cubes))
        dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                  (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        cubes.add((seed_cube[0] + dx, seed_cube[1] + dy, seed_cube[2] + dz))
    return list(cubes)


def _gen_tower(rng, n) -> List[Tuple[int, int, int]]:
    """A small base of 3-4 cubes with a tower on top."""
    cubes = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    while len(cubes) < n:
        cubes.append((rng.choice([0, 1]), rng.choice([0, 1]),
                      len(cubes) - 4 + 1))
    return cubes[:n]


# ====================================================================== #
# Env
# ====================================================================== #

class StackedCubeTouchCountQA(StandaloneVisualEnv):
    """How many cubes share a face with the marked cube? Single-letter MCQ."""

    ALLOW_ROTATION = False  # orientation-sensitive (iso view)
    ENV_NAME = "stacked_cube_touch_count"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {"n_cubes_range": (2, 3), "shape_pool": [_gen_row],
                    "use_5_options": False, "trap_rate": 0.0,
                    "max_value": 3}
        if level <= 2:
            return {"n_cubes_range": (3, 4), "shape_pool": [_gen_row, _gen_L],
                    "use_5_options": False, "trap_rate": 0.0,
                    "max_value": 3}
        if level <= 4:
            return {"n_cubes_range": (4, 5),
                    "shape_pool": [_gen_row, _gen_L, _gen_T],
                    "use_5_options": False, "trap_rate": 0.0,
                    "max_value": 3}
        if level <= 6:
            return {"n_cubes_range": (5, 7),
                    "shape_pool": [_gen_L, _gen_T, _gen_random],
                    "use_5_options": True, "trap_rate": 0.15,
                    "max_value": 4}
        if level == 7:
            return {"n_cubes_range": (7, 9),
                    "shape_pool": [_gen_random, _gen_tower],
                    "use_5_options": True, "trap_rate": 0.25,
                    "max_value": 5}
        if level == 8:
            return {"n_cubes_range": (9, 12),
                    "shape_pool": [_gen_random, _gen_tower],
                    "use_5_options": True, "trap_rate": 0.35,
                    "max_value": 5}
        # L9
        return {"n_cubes_range": (12, 16),
                "shape_pool": [_gen_random, _gen_tower],
                "use_5_options": True, "trap_rate": 0.40,
                "max_value": 6}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        for _ in range(30):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        n_lo, n_hi = cfg["n_cubes_range"]
        n = rng.randint(n_lo, n_hi)
        shape_fn = rng.choice(cfg["shape_pool"])
        cubes = shape_fn(rng, n)
        if len(cubes) < 2:
            return None
        cubes = list(set(cubes))  # dedupe
        if len(cubes) < 2:
            return None

        # Pick marked cube — at low levels prefer cubes with a clear count
        marked = rng.choice(cubes)
        true_count = _count_face_touches(cubes, marked)

        # Cap reported count to max_value to keep MCQ tractable
        if true_count > cfg["max_value"]:
            return None

        # Build option pool of integer labels
        max_v = cfg["max_value"]
        all_vals = list(range(0, max_v + 1))
        wrong_vals = [v for v in all_vals if v != true_count]

        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4
        # Need n_value-1 wrong values (and we want to include true value when no trap)
        if len(wrong_vals) < (n_value - 1):
            # not enough options; fallback to no-trap mode
            return None

        rng.shuffle(wrong_vals)
        opt_letters = ["A", "B", "C", "D", "E"][:n_value]
        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            chosen = wrong_vals[:n_value - 1]
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = "E"
            opt_lines_vals = options + ["No correct answer"]
        else:
            chosen = wrong_vals[:n_value - 1]
            options = [true_count] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_count)]
            opt_lines_vals = list(options)

        # Render image. Label every cube with a unique letter; mark the
        # query cube with red border.
        labels = self._label_cubes(cubes, rng)
        marked_label = labels[cubes.index(marked)]
        img = self._render_stack(cubes, labels, marked, rng)

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {opt_lines_vals[i]}"
            for i in range(n_value)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        if level <= 1:
            stem = (
                f"The image shows {len(cubes)} unit cubes labeled "
                f"{', '.join(labels)}. Cube {marked_label} (highlighted in red) "
                f"is the marked cube. How many of the OTHER cubes share a "
                f"FACE with cube {marked_label}? Hint: only count cubes that "
                f"touch on a flat side. Be concise."
            )
        elif level <= 4:
            stem = (
                f"The image shows {len(cubes)} unit cubes. The marked cube "
                f"(red border, label {marked_label}) is the cube of interest. "
                f"How many of the other cubes share a face with cube "
                f"{marked_label}?"
            )
        else:
            stem = (
                f"In the diagram, {len(cubes)} unit cubes form a 3D shape. "
                f"Cube {marked_label} is highlighted with a red border. "
                f"Count the number of other cubes that share a flat face "
                f"(not just an edge or vertex) with cube {marked_label}."
            )

        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _label_cubes(self, cubes, rng) -> List[str]:
        n = len(cubes)
        # Use uppercase letters for first 26 cubes
        return [chr(ord("A") + i) for i in range(n)]

    def _render_stack(self, cubes, labels, marked, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = _cube_face_palette(style, rng)

        # Increase figure size — readable for L0/L1
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc), dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        # Painter's algorithm: draw farthest first
        sorted_cubes = sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))

        top_color = palette[0]
        left_color = palette[1]
        right_color = palette[2]

        for (x, y, z) in sorted_cubes:
            is_marked = (x, y, z) == marked
            edge_color = "#c0392b" if is_marked else "#2b2b2b"
            edge_lw = 4.0 if is_marked else 1.4
            # face of cube — draw 3 visible faces
            face_top = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            face_left = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            face_right = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(face_top, closed=True, facecolor=top_color,
                                 edgecolor=edge_color, lw=edge_lw))
            ax.add_patch(Polygon(face_left, closed=True, facecolor=left_color,
                                 edgecolor=edge_color, lw=edge_lw))
            ax.add_patch(Polygon(face_right, closed=True, facecolor=right_color,
                                 edgecolor=edge_color, lw=edge_lw))

            # Label on top face center
            cx_top = (face_top[0][0] + face_top[2][0]) / 2
            cy_top = (face_top[0][1] + face_top[2][1]) / 2
            label_idx = cubes.index((x, y, z))
            label_text = labels[label_idx]
            label_color = "#c0392b" if is_marked else "#111111"
            ax.text(cx_top, cy_top, label_text, fontsize=18,
                    fontweight="bold", ha="center", va="center",
                    color=label_color)

        # Auto-scale
        pts = []
        for (x, y, z) in cubes:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        pts.append(_iso_project(x + dx, y + dy, z + dz))
        if pts:
            arr = np.array(pts)
            margin = 0.6
            ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
            ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice([
            "Stack of unit cubes",
            "3D arrangement of unit cubes",
            "Cube stack",
        ]), fontsize=style["font_size_base"] + 4, fontweight="bold")

        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = StackedCubeTouchCountQA()
    for level in [0, 3, 6, 9]:
        for seed in [1, 7, 42]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"L{level} seed{seed} FAILED")
                continue
            img = env.render()
            out = f"/tmp/env_check/scc_seed{seed}_L{level}.png"
            img.save(out)
            print(f"saved {out}, ans={env._answer}")
