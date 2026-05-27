"""
Analogy With Negation QA environment.

A:B :: C:? analogy where the A->B transformation includes a NOT / complement
operation (fill toggle, color inversion, set complement, combined
complement+rotation).

Target: VisualPuzzles analogical + X5 (deductive reasoning).

Difficulty axes:
  1. negation_type: L0-2 fill toggle, L3-5 color inversion, L6-7 set
     complement on a small grid, L8-9 combined (complement + rotation).
  2. n_shapes / distractor_similarity: low-level distractors unrelated, at
     L9 distractors are off-by-one elements of the complement.
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

_SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon", "diamond"]
_BASE_COLORS = [
    ("red", "#e74c3c", "#2ecc71"),      # red <-> green
    ("blue", "#3498db", "#e67e22"),     # blue <-> orange
    ("purple", "#9b59b6", "#f1c40f"),   # purple <-> yellow
    ("teal", "#1abc9c", "#c0392b"),     # teal <-> dark red
    ("navy", "#34495e", "#f39c12"),     # navy <-> amber
]

class AnalogyWithNegationQA(StandaloneVisualEnv):
    ENV_NAME = "analogy_with_negation"

    # ------------------------------------------------------------------ #
    # Level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Return per-level difficulty settings.

        negation_type: "fill" (L0-2), "color" (L3-5), "complement" (L6-7),
        "combined" (L8-9).
        grid_n: size of the grid when using set complement (3..5).
        tight_distractors: whether distractors differ by a single element.
        """
        level = max(0, min(level, 9))
        if level <= 2:
            return {"negation_type": "fill",
                    "grid_n": 0,
                    "tight_distractors": False}
        if level <= 5:
            return {"negation_type": "color",
                    "grid_n": 0,
                    "tight_distractors": level >= 5}
        if level <= 7:
            return {"negation_type": "complement",
                    "grid_n": 3 + (level - 6),  # 3 at L6, 4 at L7
                    "tight_distractors": True}
        # L8-9 combined complement + rotation on a larger grid
        return {"negation_type": "combined",
                "grid_n": 4 + (level - 8),  # 4 at L8, 5 at L9
                "tight_distractors": True}

    _QUESTION_TEMPLATES = [
        "Study the analogy in the image: A is to B as C is to ___. Infer the transformation from A to B, apply the same transformation to C, and choose the matching option. Answer with a single letter.",
        "In the image, A becomes B through some transformation (which may include a NOT operation). Apply the same rule to C. Which option (A-D) shows the correct result? Answer with a single letter.",
        "A : B :: C : ? -- Identify the transformation (involving negation/complement) from A to B, then apply it to C. Pick the correct answer from options A-D. Answer with a single letter.",
        "Look at the top row: A transforms into B. Apply the exact same transformation to C and select the matching option. Answer with a single letter A, B, C, or D.",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1301)
        self._primary_complexity_feature = level

        for _ in range(30):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Per-problem generation
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        neg_type = cfg["negation_type"]
        if neg_type == "fill":
            return self._gen_fill(cfg, rng)
        if neg_type == "color":
            return self._gen_color_invert(cfg, rng)
        if neg_type == "complement":
            return self._gen_complement(cfg, rng, rotate=False)
        return self._gen_complement(cfg, rng, rotate=True)

    # --- Fill-toggle problems (L0-2) ---

    def _gen_fill(self, cfg: Dict, rng: random.Random):
        """A = filled shape, B = outlined version; C = filled other shape,
        ? = outlined version."""
        shape_a = rng.choice(_SHAPES)
        color_a = rng.choice(_BASE_COLORS)
        shape_c = rng.choice([s for s in _SHAPES if s != shape_a])
        color_c = rng.choice(_BASE_COLORS)

        a = {"kind": "single", "shape": shape_a, "color": color_a[1],
             "filled": True}
        b = {"kind": "single", "shape": shape_a, "color": color_a[1],
             "filled": False}
        c = {"kind": "single", "shape": shape_c, "color": color_c[1],
             "filled": True}
        correct = {"kind": "single", "shape": shape_c, "color": color_c[1],
                   "filled": False}

        # Distractors
        d1 = {"kind": "single", "shape": shape_c, "color": color_c[1],
              "filled": True}    # forgot to toggle
        # d2: color-swap distractor must keep filled=True so it visually
        # differs from the unfilled correct answer (unfilled shapes render
        # without color, making a "color-swap + unfilled" option identical
        # to the correct outlined shape — that would be a duplicate option).
        d2 = {"kind": "single", "shape": shape_c, "color": color_c[2],
              "filled": True}    # color swap only (no fill toggle)
        d3_shape = rng.choice([s for s in _SHAPES
                                if s != shape_c and s != shape_a])
        d3 = {"kind": "single", "shape": d3_shape, "color": color_c[1],
              "filled": False}   # wrong shape
        distractors = [d1, d2, d3]
        return self._finalize(a, b, c, correct, distractors, rng)

    # --- Color inversion problems (L3-5) ---

    def _gen_color_invert(self, cfg: Dict, rng: random.Random):
        """A = colored shape1, B = complementary-colored shape1,
        C = colored shape2, ? = complementary-colored shape2."""
        shape_a = rng.choice(_SHAPES)
        color_a = rng.choice(_BASE_COLORS)
        shape_c = rng.choice([s for s in _SHAPES if s != shape_a])
        color_c = rng.choice([cc for cc in _BASE_COLORS if cc[0] != color_a[0]])

        a = {"kind": "single", "shape": shape_a, "color": color_a[1],
             "filled": True}
        b = {"kind": "single", "shape": shape_a, "color": color_a[2],
             "filled": True}
        c = {"kind": "single", "shape": shape_c, "color": color_c[1],
             "filled": True}
        correct = {"kind": "single", "shape": shape_c, "color": color_c[2],
                   "filled": True}

        d1 = {"kind": "single", "shape": shape_c, "color": color_c[1],
              "filled": True}    # no change at all
        d2 = {"kind": "single", "shape": shape_c, "color": color_c[1],
              "filled": False}   # fill toggle instead of color invert
        # Pick wrong complement color
        wrong_color = rng.choice([cc for cc in _BASE_COLORS
                                   if cc[0] != color_c[0]])[2]
        d3 = {"kind": "single", "shape": shape_c, "color": wrong_color,
              "filled": True}
        distractors = [d1, d2, d3]
        return self._finalize(a, b, c, correct, distractors, rng)

    # --- Set complement problems (L6-9) ---

    def _gen_complement(self, cfg: Dict, rng: random.Random,
                        rotate: bool):
        """A = n x n grid with k cells filled, B = grid with the OTHER cells
        filled; C = different grid, ? = complement. If rotate=True, the
        complement is also rotated 90 deg CW."""
        n = cfg["grid_n"]
        total = n * n
        # Pick k so that the problem is non-trivial (0 < k < total).
        k_a = rng.randint(max(1, total // 4), max(2, total - max(1, total // 4)))
        cells_a = set(rng.sample(range(total), k_a))
        a = {"kind": "grid", "n": n, "cells": cells_a, "color": rng.choice(_BASE_COLORS)[1]}
        comp_a = set(range(total)) - cells_a
        if rotate:
            comp_a = _rotate_cells(comp_a, n)
        b = {"kind": "grid", "n": n, "cells": comp_a,
             "color": a["color"]}

        # Ensure C differs from A
        for _ in range(20):
            k_c = rng.randint(max(1, total // 4),
                              max(2, total - max(1, total // 4)))
            cells_c = set(rng.sample(range(total), k_c))
            if cells_c != cells_a and cells_c != comp_a:
                break
        c_color = rng.choice(_BASE_COLORS)[1]
        c = {"kind": "grid", "n": n, "cells": cells_c, "color": c_color}

        comp_c = set(range(total)) - cells_c
        if rotate:
            comp_c = _rotate_cells(comp_c, n)
        correct = {"kind": "grid", "n": n, "cells": comp_c, "color": c_color}

        # Distractors: "same as C" (no operation), "off-by-one element",
        # "rotation only without complement".
        d1 = {"kind": "grid", "n": n, "cells": set(cells_c),
              "color": c_color}
        # off-by-one from correct
        corr_list = list(correct["cells"])
        if corr_list:
            swap = rng.choice(corr_list)
        else:
            swap = 0
        not_in = [i for i in range(total) if i not in correct["cells"]]
        add = rng.choice(not_in) if not_in else 0
        d2_cells = set(correct["cells"])
        if swap in d2_cells:
            d2_cells.remove(swap)
        d2_cells.add(add)
        d2 = {"kind": "grid", "n": n, "cells": d2_cells, "color": c_color}

        if rotate:
            # Distractor: rotation without complement (rotate C itself)
            d3_cells = _rotate_cells(set(cells_c), n)
        else:
            # Distractor: complement but using a wrong cell count
            other_cells = list(range(total))
            rng.shuffle(other_cells)
            d3_cells = set(other_cells[:len(correct["cells"])])
        # Avoid duplicates
        if d3_cells == correct["cells"]:
            # Perturb
            if d3_cells:
                first = next(iter(d3_cells))
                d3_cells = set(d3_cells)
                d3_cells.discard(first)
                for i in range(total):
                    if i not in d3_cells and i != first:
                        d3_cells.add(i)
                        break
        d3 = {"kind": "grid", "n": n, "cells": d3_cells, "color": c_color}

        distractors = [d1, d2, d3]
        return self._finalize(a, b, c, correct, distractors, rng)

    # ------------------------------------------------------------------ #
    # Option assembly + rendering
    # ------------------------------------------------------------------ #

    def _finalize(self, a, b, c, correct, distractors, rng):
        # Ensure options are unique
        def key(o):
            if o["kind"] == "single":
                return ("single", o["shape"], o["color"], o["filled"])
            return ("grid", o["n"], tuple(sorted(o["cells"])), o["color"])

        seen = {key(correct)}
        uniq_d = []
        for d in distractors:
            k = key(d)
            if k not in seen:
                seen.add(k)
                uniq_d.append(d)
        # Pad if fewer than 3 distractors
        attempts = 0
        while len(uniq_d) < 3 and attempts < 20:
            attempts += 1
            if correct["kind"] == "single":
                trial = {"kind": "single",
                         "shape": rng.choice(_SHAPES),
                         "color": rng.choice(_BASE_COLORS)[1],
                         "filled": rng.choice([True, False])}
            else:
                total = correct["n"] * correct["n"]
                k2 = rng.randint(1, total - 1)
                trial = {"kind": "grid", "n": correct["n"],
                         "cells": set(rng.sample(range(total), k2)),
                         "color": correct["color"]}
            tk = key(trial)
            if tk not in seen:
                seen.add(tk)
                uniq_d.append(trial)
        if len(uniq_d) < 3:
            return None
        options = list(uniq_d[:3])
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)
        question = rng.choice(self._QUESTION_TEMPLATES)
        image = self._render(a, b, c, options, rng)
        return question, answer_letter, image

    def _render(self, a, b, c, options, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(10.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3], hspace=0.35)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        ax_top.set_xlim(0, 10)
        ax_top.set_ylim(0, 2.6)
        ax_top.set_aspect("equal")
        ax_top.axis("off")
        title_pool = [
            "Analogy with Negation: A : B :: C : ?",
            "A is to B as C is to ?",
            "Negation Analogy",
            "Complement / Negation Puzzle",
        ]
        ax_top.set_title(rng.choice(title_pool), fontsize=14,
                         fontweight="bold", pad=6)

        cell_size = 1.8

        def _box(ax, cx, cy, dash=False):
            rect = mpatches.FancyBboxPatch(
                (cx - cell_size / 2, cy - cell_size / 2),
                cell_size, cell_size, boxstyle="round,pad=0.05",
                facecolor="#fdfefe",
                edgecolor=("#e74c3c" if dash else "#2c3e50"),
                linewidth=(2.5 if dash else 2.0),
                linestyle=("--" if dash else "-"), zorder=1)
            ax.add_patch(rect)

        _box(ax_top, 1.0, 1.3)
        _draw_item(ax_top, 1.0, 1.3, cell_size, a)
        ax_top.text(1.0, 0.25, "A", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.annotate("", xy=(2.95, 1.3), xytext=(2.0, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color="#e67e22"))

        _box(ax_top, 3.8, 1.3)
        _draw_item(ax_top, 3.8, 1.3, cell_size, b)
        ax_top.text(3.8, 0.25, "B", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.text(5.2, 1.3, "::", fontsize=24, fontweight="bold",
                    ha="center", va="center", color="#7f8c8d")

        _box(ax_top, 6.6, 1.3)
        _draw_item(ax_top, 6.6, 1.3, cell_size, c)
        ax_top.text(6.6, 0.25, "C", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.annotate("", xy=(8.55, 1.3), xytext=(7.6, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color="#e67e22"))

        _box(ax_top, 9.0, 1.3, dash=True)
        ax_top.text(9.0, 1.3, "?", fontsize=28, fontweight="bold",
                    ha="center", va="center", color="#e74c3c")
        ax_top.text(9.0, 0.25, "?", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#e74c3c")

        # Options
        n_opts = len(options)
        ax_bot.set_xlim(0, n_opts * 2.2)
        ax_bot.set_ylim(0, 2.2)
        ax_bot.set_aspect("equal")
        ax_bot.axis("off")
        ax_bot.set_title("Choose the option that matches:",
                         fontsize=11, pad=3)

        opt_cell = 1.6
        for i, opt in enumerate(options):
            cx = i * 2.2 + 1.1
            cy = 1.05
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2),
                opt_cell, opt_cell, boxstyle="round,pad=0.04",
                facecolor="#eaf2f8", edgecolor="#2c3e50", linewidth=1.5,
                zorder=1)
            ax_bot.add_patch(rect)
            _draw_item(ax_bot, cx, cy, opt_cell, opt)
            ax_bot.text(cx, cy - opt_cell / 2 - 0.15, chr(ord("A") + i),
                        fontsize=12, fontweight="bold", ha="center",
                        va="top", color="#2c3e50")

        fig.subplots_adjust(left=0.04, right=0.96, top=0.9,
                            bottom=0.05, hspace=0.35)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _rotate_cells(cells: set, n: int) -> set:
    """Rotate filled-cell indices 90 deg CW on an n x n grid."""
    out = set()
    for idx in cells:
        r = idx // n
        c = idx % n
        new_r = c
        new_c = n - 1 - r
        out.add(new_r * n + new_c)
    return out

def _draw_item(ax, cx, cy, cell_size, item):
    if item["kind"] == "single":
        _draw_shape(ax, cx, cy, cell_size * 0.55, item["shape"],
                    item["color"], item["filled"])
    else:
        _draw_grid(ax, cx, cy, cell_size * 0.85, item["n"],
                   item["cells"], item["color"])

def _draw_shape(ax, cx, cy, size, shape, color, filled):
    fc = color if filled else "none"
    edge = "#2c3e50"
    lw = 2.0 if not filled else 1.5
    if shape == "circle":
        p = mpatches.Circle((cx, cy), size, facecolor=fc, edgecolor=edge,
                             linewidth=lw)
        ax.add_patch(p)
    elif shape == "square":
        p = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                facecolor=fc, edgecolor=edge, linewidth=lw)
        ax.add_patch(p)
    elif shape == "diamond":
        pts = [(cx, cy + size), (cx + size, cy),
               (cx, cy - size), (cx - size, cy)]
        p = mpatches.Polygon(pts, closed=True, facecolor=fc,
                              edgecolor=edge, linewidth=lw)
        ax.add_patch(p)
    elif shape == "triangle":
        p = mpatches.RegularPolygon((cx, cy), 3, radius=size,
                                     orientation=0, facecolor=fc,
                                     edgecolor=edge, linewidth=lw)
        ax.add_patch(p)
    elif shape == "pentagon":
        p = mpatches.RegularPolygon((cx, cy), 5, radius=size,
                                     orientation=0, facecolor=fc,
                                     edgecolor=edge, linewidth=lw)
        ax.add_patch(p)
    elif shape == "hexagon":
        p = mpatches.RegularPolygon((cx, cy), 6, radius=size,
                                     orientation=math.radians(30),
                                     facecolor=fc, edgecolor=edge,
                                     linewidth=lw)
        ax.add_patch(p)
    else:
        p = mpatches.Circle((cx, cy), size, facecolor=fc,
                             edgecolor=edge, linewidth=lw)
        ax.add_patch(p)

def _draw_grid(ax, cx, cy, total_size, n, cells, color):
    """Draw an n x n grid where filled cells are colored."""
    cell_sz = total_size / n
    x0 = cx - total_size / 2
    y0 = cy - total_size / 2
    for idx in range(n * n):
        r = idx // n
        c = idx % n
        gx = x0 + c * cell_sz
        gy = y0 + (n - 1 - r) * cell_sz
        filled = idx in cells
        fc = color if filled else "#ffffff"
        rect = mpatches.Rectangle((gx, gy), cell_sz, cell_sz,
                                   facecolor=fc, edgecolor="#2c3e50",
                                   linewidth=1.0)
        ax.add_patch(rect)

if __name__ == "__main__":
    env = AnalogyWithNegationQA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
