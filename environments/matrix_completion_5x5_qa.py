"""
Matrix Completion 5x5 QA environment.

A 5x5 matrix of shapes where the bottom-right cell is missing. Each row
and column follows a rule (attribute changes systematically). Task: pick
the missing cell from 4 options.

Target: reference inductive, visual-perception IQ_Test.

Difficulty axes:
  1. n_attributes: 2 at L0-4, 3 at L5-9.
  2. rule_type: L0-4 enumeration (each value appears once per row/col),
     L5-6 arithmetic progression, L7-9 XOR / modular combinations.
  3. tight_distractors at L>=4.
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
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c"]
_FILLS = ["solid", "outline", "striped"]
_SIZES = [0.28, 0.36, 0.44, 0.52, 0.60]

class MatrixCompletion5x5QA(StandaloneVisualEnv):
    ENV_NAME = "matrix_completion_5x5"

    # ------------------------------------------------------------------ #
    # Level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 4:
            n_attributes = 2
        else:
            n_attributes = 3
        if level <= 4:
            rule_type = "enumeration"
        elif level <= 6:
            rule_type = "arithmetic"
        else:
            rule_type = "xor"
        return {
            "n_attributes": n_attributes,
            "rule_type": rule_type,
            "tight_distractors": level >= 4,
        }

    _QUESTION_TEMPLATES = [
        "The 5x5 matrix follows a hidden pattern. Each row and each column varies by systematic rules. Which option (A-D) fills the bottom-right cell? Answer with a single letter.",
        "Look at the 5x5 grid. The bottom-right cell is missing. Based on the row/column patterns, pick the correct option from A-D. Answer with a single letter.",
        "A 5x5 matrix has one missing cell. Determine the row and column rules, then pick the option (A-D) that completes the pattern. Answer with a single letter.",
        "Examine the 5x5 matrix. Which of the four options correctly fills the missing bottom-right cell? Answer with a single letter A, B, C, or D.",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        # 2026-05-04: added easier L0 mode (was 12.5% — VLM pattern-induction limit, attempt fix)
        # L0/L1: 3x3 matrix with 1 varying attribute (color or shape).
        if level <= 1:
            return self._generate_easy_l0l1(level)
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1319)
        self._primary_complexity_feature = cfg["n_attributes"] * 5 + level

        for _ in range(40):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    def _generate_easy_l0l1(self, level: int):
        """L0/L1: simple 3x3 matrix with ONE varying attribute (Latin square)."""
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 9931)
        self._primary_complexity_feature = 1
        # Pick which attribute varies — keep all others constant
        attr = rng.choice(["shape", "color"])
        # Use 3 values
        n = 3
        offset = rng.randint(0, n - 1)
        perm = list(range(n))
        rng.shuffle(perm)
        # Build 3x3 matrix
        const_shape = rng.randint(0, 4)
        const_color = rng.randint(0, 4)
        const_fill = rng.randint(0, len(_FILLS) - 1)
        const_size = 2
        matrix = [[{"shape": const_shape, "color": const_color,
                    "fill": const_fill, "size": const_size}
                   for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for c in range(n):
                v = perm[(r + c + offset) % n]
                if attr == "shape":
                    matrix[r][c]["shape"] = v
                else:
                    matrix[r][c]["color"] = v
        correct = dict(matrix[n - 1][n - 1])

        # Distractors: vary the chosen attr by +1 / +2 / random other
        seen = [correct]
        distractors = []

        def _eq(a, b):
            return all(a[k] == b[k] for k in ("shape", "color", "fill", "size"))

        def _add(cand):
            for s in seen:
                if _eq(s, cand):
                    return False
            seen.append(cand)
            distractors.append(cand)
            return True

        for delta in [1, 2, 3]:
            cand = dict(correct)
            if attr == "shape":
                cand["shape"] = (correct["shape"] + delta) % 5
            else:
                cand["color"] = (correct["color"] + delta) % 5
            _add(cand)
            if len(distractors) >= 3:
                break
        # Pad
        attempts = 0
        while len(distractors) < 3 and attempts < 30:
            attempts += 1
            cand = dict(correct)
            cand["shape"] = rng.randint(0, 4)
            cand["color"] = rng.randint(0, 4)
            _add(cand)
        if len(distractors) < 3:
            return None
        options = distractors[:3]
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)

        question = (
            f"A 3x3 matrix follows a simple Latin-square pattern: each row "
            f"and each column contains the same set of {n} {attr}s, just in "
            f"different orders. The bottom-right cell is missing. Which "
            f"option (A-D) fills it? Answer with a single letter."
        )
        image = self._render_3x3(matrix, options, rng, n)
        return question, answer_letter, image

    def _render_3x3(self, matrix, options, rng, n) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(8.5 * sc, 8.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.3)
        ax_m = fig.add_subplot(gs[0])
        ax_o = fig.add_subplot(gs[1])
        ax_m.set_aspect("equal")
        ax_o.set_aspect("equal")
        ax_m.axis("off")
        ax_o.axis("off")
        ax_m.set_title("3x3 Pattern Matrix", fontsize=14,
                       fontweight="bold", pad=8)
        cell_w = 1.3
        cell_h = 1.3
        for r in range(n):
            for c in range(n):
                cx = c * cell_w + cell_w / 2
                cy = (n - 1 - r) * cell_h + cell_h / 2
                is_missing = (r == n - 1 and c == n - 1)
                if is_missing:
                    rect = mpatches.FancyBboxPatch(
                        (cx - cell_w / 2 + 0.03, cy - cell_h / 2 + 0.03),
                        cell_w - 0.06, cell_h - 0.06,
                        boxstyle="round,pad=0.01",
                        facecolor="#fef3c7",
                        edgecolor="#e74c3c", linewidth=2.5,
                        linestyle="--", zorder=1)
                    ax_m.add_patch(rect)
                    ax_m.text(cx, cy, "?", fontsize=22, fontweight="bold",
                               ha="center", va="center", color="#e74c3c",
                               zorder=5)
                else:
                    rect = mpatches.FancyBboxPatch(
                        (cx - cell_w / 2 + 0.03, cy - cell_h / 2 + 0.03),
                        cell_w - 0.06, cell_h - 0.06,
                        boxstyle="round,pad=0.01",
                        facecolor="#ffffff", edgecolor="#34495e",
                        linewidth=1.2, zorder=1)
                    ax_m.add_patch(rect)
                    _draw_cell(ax_m, cx, cy, cell_w, matrix[r][c])
        ax_m.set_xlim(0, n * cell_w)
        ax_m.set_ylim(0, n * cell_h)
        ax_o.set_title("Options", fontsize=12, fontweight="bold", pad=4)
        opt_cell = 1.4
        for i, opt in enumerate(options):
            cx = i * (opt_cell + 0.5) + opt_cell / 2 + 0.3
            cy = opt_cell / 2 + 0.2
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2),
                opt_cell, opt_cell, boxstyle="round,pad=0.02",
                facecolor="#eaf2f8", edgecolor="#2c3e50",
                linewidth=1.4, zorder=1)
            ax_o.add_patch(rect)
            _draw_cell(ax_o, cx, cy, opt_cell, opt)
            ax_o.text(cx, cy - opt_cell / 2 - 0.15, chr(ord("A") + i),
                       fontsize=12, fontweight="bold", ha="center",
                       va="top", color="#2c3e50")
        ax_o.set_xlim(0, len(options) * (opt_cell + 0.5) + 0.3)
        ax_o.set_ylim(-0.3, opt_cell + 0.5)
        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.04,
                             hspace=0.3)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Matrix construction
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n_attrs = cfg["n_attributes"]
        rule = cfg["rule_type"]

        # Pick which attributes vary
        all_attrs = ["shape", "color", "fill", "size"]
        chosen = rng.sample(all_attrs, n_attrs)

        matrix = [[{"shape": 0, "color": 0, "fill": 0, "size": 2}
                    for _ in range(5)] for _ in range(5)]

        if rule == "enumeration":
            # Latin-square style: row r + col c -> value (r+c) % 5 per
            # varying attribute.
            for attr in chosen:
                offset = rng.randint(0, 4)
                perm = list(range(5))
                rng.shuffle(perm)
                for r in range(5):
                    for c in range(5):
                        v = perm[(r + c + offset) % 5]
                        matrix[r][c][attr] = v
        elif rule == "arithmetic":
            # Row adds r; col adds c. (r + c) mod 5.
            for attr in chosen:
                base = rng.randint(0, 4)
                row_step = rng.choice([1, 2, 3])
                col_step = rng.choice([1, 2, 3])
                for r in range(5):
                    for c in range(5):
                        matrix[r][c][attr] = (base + r * row_step
                                                + c * col_step) % 5
        else:  # xor
            for attr in chosen:
                base = rng.randint(0, 4)
                a1 = rng.choice([1, 2, 3])
                a2 = rng.choice([1, 2, 3])
                for r in range(5):
                    for c in range(5):
                        matrix[r][c][attr] = (base ^ (r * a1 + c * a2)) % 5

        # Fix non-varying attributes to a constant of your choice.
        unchanged = [a for a in all_attrs if a not in chosen]
        for attr in unchanged:
            const = rng.randint(0, 4)
            if attr == "fill":
                const = rng.randint(0, len(_FILLS) - 1)
            for r in range(5):
                for c in range(5):
                    matrix[r][c][attr] = const

        correct = dict(matrix[4][4])

        # Distractors
        distractors = self._make_distractors(cfg, rng, chosen, correct)
        if len(distractors) < 3:
            return None

        options = list(distractors[:3])
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)

        question = rng.choice(self._QUESTION_TEMPLATES)
        image = self._render(matrix, options, rng)
        return question, answer_letter, image

    def _make_distractors(self, cfg, rng, varying_attrs, correct):
        distractors = []
        seen = [correct]

        def _eq(a, b):
            return all(a[k] == b[k] for k in ("shape", "color", "fill", "size"))

        def _add(cand):
            for s in seen:
                if _eq(s, cand):
                    return False
            seen.append(cand)
            distractors.append(cand)
            return True

        tight = cfg["tight_distractors"]
        if tight:
            # For each varying attribute, violate its value only (off-by-one).
            for attr in varying_attrs:
                cand = dict(correct)
                if attr == "shape":
                    cand["shape"] = (correct["shape"] + 1) % 5
                elif attr == "color":
                    cand["color"] = (correct["color"] + 1) % 5
                elif attr == "fill":
                    cand["fill"] = (correct["fill"] + 1) % len(_FILLS)
                elif attr == "size":
                    cand["size"] = (correct["size"] + 1) % 5
                _add(cand)
                if len(distractors) >= 3:
                    break
            # Another distractor: flip two attributes
            if len(distractors) < 3:
                cand = dict(correct)
                for attr in varying_attrs[:2]:
                    if attr == "shape":
                        cand["shape"] = (correct["shape"] + 2) % 5
                    elif attr == "color":
                        cand["color"] = (correct["color"] + 2) % 5
                    elif attr == "fill":
                        cand["fill"] = (correct["fill"] + 1) % len(_FILLS)
                    elif attr == "size":
                        cand["size"] = (correct["size"] + 2) % 5
                _add(cand)
        else:
            attempts = 0
            while len(distractors) < 3 and attempts < 50:
                attempts += 1
                cand = dict(correct)
                attrs = rng.sample(varying_attrs,
                                    rng.randint(1, len(varying_attrs)))
                for at in attrs:
                    if at == "shape":
                        cand["shape"] = rng.choice([x for x in range(5)
                                                     if x != correct["shape"]])
                    elif at == "color":
                        cand["color"] = rng.choice([x for x in range(5)
                                                     if x != correct["color"]])
                    elif at == "fill":
                        cand["fill"] = rng.choice([x for x in range(len(_FILLS))
                                                    if x != correct["fill"]])
                    elif at == "size":
                        cand["size"] = rng.choice([x for x in range(5)
                                                    if x != correct["size"]])
                _add(cand)
        # Pad
        attempts = 0
        while len(distractors) < 3 and attempts < 30:
            attempts += 1
            cand = {"shape": rng.randint(0, 4),
                    "color": rng.randint(0, 4),
                    "fill": rng.randint(0, len(_FILLS) - 1),
                    "size": rng.randint(0, 4)}
            _add(cand)
        return distractors

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, matrix, options, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(11.0 * sc, 9.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.3)
        ax_m = fig.add_subplot(gs[0])
        ax_o = fig.add_subplot(gs[1])
        ax_m.set_aspect("equal")
        ax_o.set_aspect("equal")
        ax_m.axis("off")
        ax_o.axis("off")

        mat_title_pool = ["5x5 Pattern Matrix", "5x5 Matrix",
                           "Pattern Grid", "Raven-style 5x5 Matrix"]
        ax_m.set_title(rng.choice(mat_title_pool), fontsize=14,
                        fontweight="bold", pad=8)

        cell_w = 1.3
        cell_h = 1.3
        for r in range(5):
            for c in range(5):
                cx = c * cell_w + cell_w / 2
                cy = (4 - r) * cell_h + cell_h / 2
                is_missing = (r == 4 and c == 4)
                if is_missing:
                    rect = mpatches.FancyBboxPatch(
                        (cx - cell_w / 2 + 0.03, cy - cell_h / 2 + 0.03),
                        cell_w - 0.06, cell_h - 0.06,
                        boxstyle="round,pad=0.01",
                        facecolor="#fef3c7",
                        edgecolor="#e74c3c", linewidth=2.5,
                        linestyle="--", zorder=1)
                    ax_m.add_patch(rect)
                    ax_m.text(cx, cy, "?", fontsize=22, fontweight="bold",
                               ha="center", va="center", color="#e74c3c",
                               zorder=5)
                else:
                    rect = mpatches.FancyBboxPatch(
                        (cx - cell_w / 2 + 0.03, cy - cell_h / 2 + 0.03),
                        cell_w - 0.06, cell_h - 0.06,
                        boxstyle="round,pad=0.01",
                        facecolor="#ffffff", edgecolor="#34495e",
                        linewidth=1.2, zorder=1)
                    ax_m.add_patch(rect)
                    _draw_cell(ax_m, cx, cy, cell_w, matrix[r][c])
        ax_m.set_xlim(0, 5 * cell_w)
        ax_m.set_ylim(0, 5 * cell_h)

        # Options
        ax_o.set_title("Options", fontsize=12, fontweight="bold", pad=4)
        opt_cell = 1.4
        for i, opt in enumerate(options):
            cx = i * (opt_cell + 0.5) + opt_cell / 2 + 0.3
            cy = opt_cell / 2 + 0.2
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2),
                opt_cell, opt_cell, boxstyle="round,pad=0.02",
                facecolor="#eaf2f8", edgecolor="#2c3e50",
                linewidth=1.4, zorder=1)
            ax_o.add_patch(rect)
            _draw_cell(ax_o, cx, cy, opt_cell, opt)
            ax_o.text(cx, cy - opt_cell / 2 - 0.15, chr(ord("A") + i),
                       fontsize=12, fontweight="bold", ha="center",
                       va="top", color="#2c3e50")
        ax_o.set_xlim(0, len(options) * (opt_cell + 0.5) + 0.3)
        ax_o.set_ylim(-0.3, opt_cell + 0.5)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.04,
                             hspace=0.3)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Cell drawing
# ---------------------------------------------------------------------- #

def _draw_cell(ax, cx, cy, cell_w, cell):
    shape_name = _SHAPES[cell["shape"] % len(_SHAPES)]
    color = _COLORS[cell["color"] % len(_COLORS)]
    fill = _FILLS[cell["fill"] % len(_FILLS)]
    size_frac = _SIZES[cell["size"] % len(_SIZES)]
    size = cell_w * size_frac * 0.45

    fc = color if fill == "solid" else "none"
    edge = color if fill == "outline" else "#2c3e50"
    lw = 2.0 if fill == "outline" else 1.5

    if shape_name == "circle":
        p = mpatches.Circle((cx, cy), size, facecolor=fc, edgecolor=edge,
                             linewidth=lw, zorder=3)
        ax.add_patch(p)
    elif shape_name == "square":
        p = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                facecolor=fc, edgecolor=edge,
                                linewidth=lw, zorder=3)
        ax.add_patch(p)
    elif shape_name == "diamond":
        pts = [(cx, cy + size), (cx + size, cy),
               (cx, cy - size), (cx - size, cy)]
        p = mpatches.Polygon(pts, closed=True, facecolor=fc, edgecolor=edge,
                              linewidth=lw, zorder=3)
        ax.add_patch(p)
    elif shape_name == "triangle":
        p = mpatches.RegularPolygon((cx, cy), 3, radius=size,
                                     orientation=0, facecolor=fc,
                                     edgecolor=edge, linewidth=lw,
                                     zorder=3)
        ax.add_patch(p)
    elif shape_name == "pentagon":
        p = mpatches.RegularPolygon((cx, cy), 5, radius=size,
                                     orientation=0, facecolor=fc,
                                     edgecolor=edge, linewidth=lw,
                                     zorder=3)
        ax.add_patch(p)
    elif shape_name == "hexagon":
        p = mpatches.RegularPolygon((cx, cy), 6, radius=size,
                                     orientation=math.radians(30),
                                     facecolor=fc, edgecolor=edge,
                                     linewidth=lw, zorder=3)
        ax.add_patch(p)

    if fill == "striped":
        # Draw horizontal stripes across the shape
        for yoff in (-size * 0.5, 0, size * 0.5):
            ax.plot([cx - size * 0.9, cx + size * 0.9],
                    [cy + yoff, cy + yoff],
                    color=color, linewidth=1.4, zorder=4)

if __name__ == "__main__":
    env = MatrixCompletion5x5QA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
