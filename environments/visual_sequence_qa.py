"""
Visual Sequence QA environment.

Shows 4-5 shapes in a sequence with progressive transformation.
The last position is "?". Multiple-choice answer.
Sequences: growing size, rotating angle, changing color, adding sides,
fractal-like nesting, alternating patterns.
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

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_COLORS = {
    "red": "#e74c3c", "blue": "#3498db", "green": "#27ae60",
    "orange": "#e67e22", "purple": "#8e44ad", "yellow": "#f1c40f",
    "teal": "#1abc9c", "pink": "#e91e8f",
}
_COLOR_CYCLE = list(_COLORS.keys())

_SHAPE_SIDES = {
    "triangle": 3, "square": 4, "pentagon": 5,
    "hexagon": 6, "heptagon": 7, "octagon": 8,
}

_SIZES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

def _polygon_path(cx, cy, n, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(n):
        a = off + 2 * math.pi * i / n
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _star_path(cx, cy, size, n_points=5, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(n_points * 2):
        a = off + 2 * math.pi * i / (n_points * 2)
        r = size if i % 2 == 0 else size * 0.45
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n_points * 2 - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _draw_shape_on_ax(ax, n_sides, cx, cy, size, color_hex, rotation=0,
                      alpha=0.85, linewidth=1.5):
    """Draw a regular polygon with n_sides on the axes."""
    if n_sides <= 2:
        # Circle
        p = mpatches.Circle((cx, cy), size, facecolor=color_hex,
                            edgecolor="#2c3e50", linewidth=linewidth,
                            alpha=alpha, zorder=3)
        ax.add_patch(p)
    else:
        path = _polygon_path(cx, cy, n_sides, size, rotation)
        p = mpatches.PathPatch(path, facecolor=color_hex, edgecolor="#2c3e50",
                               linewidth=linewidth, alpha=alpha, zorder=3)
        ax.add_patch(p)

class _SeqCell:
    """State of one cell in the sequence."""
    __slots__ = ("n_sides", "color_idx", "size_idx", "rotation", "count",
                 "nested_sides")

    def __init__(self, n_sides=4, color_idx=0, size_idx=2, rotation=0,
                 count=1, nested_sides=0):
        self.n_sides = n_sides
        self.color_idx = color_idx
        self.size_idx = size_idx
        self.rotation = rotation
        self.count = count
        self.nested_sides = nested_sides  # 0 means no nesting

    def copy(self):
        return _SeqCell(self.n_sides, self.color_idx, self.size_idx,
                        self.rotation, self.count, self.nested_sides)

    def __eq__(self, other):
        if not isinstance(other, _SeqCell):
            return False
        return (self.n_sides == other.n_sides and
                self.color_idx == other.color_idx and
                self.size_idx == other.size_idx and
                self.rotation == other.rotation and
                self.count == other.count and
                self.nested_sides == other.nested_sides)

    def __hash__(self):
        return hash((self.n_sides, self.color_idx, self.size_idx,
                     self.rotation, self.count, self.nested_sides))

    def draw(self, ax, cx, cy, cell_size):
        sz = _SIZES[min(self.size_idx, len(_SIZES) - 1)] * cell_size
        c_idx = self.color_idx % len(_COLOR_CYCLE)
        color_hex = _COLORS[_COLOR_CYCLE[c_idx]]

        if self.count == 1:
            positions = [(cx, cy)]
        elif self.count == 2:
            positions = [(cx - sz * 0.7, cy), (cx + sz * 0.7, cy)]
        elif self.count == 3:
            positions = [(cx, cy + sz * 0.6),
                         (cx - sz * 0.7, cy - sz * 0.4),
                         (cx + sz * 0.7, cy - sz * 0.4)]
        else:
            positions = [(cx - sz * 0.6, cy + sz * 0.5),
                         (cx + sz * 0.6, cy + sz * 0.5),
                         (cx - sz * 0.6, cy - sz * 0.4),
                         (cx + sz * 0.6, cy - sz * 0.4)]

        sub_sz = sz * (0.55 if self.count > 1 else 1.0)
        for px, py in positions[:self.count]:
            _draw_shape_on_ax(ax, self.n_sides, px, py, sub_sz,
                              color_hex, self.rotation)
            # Nested inner shape
            if self.nested_sides > 0:
                inner_color = _COLORS[_COLOR_CYCLE[(c_idx + 2) % len(_COLOR_CYCLE)]]
                _draw_shape_on_ax(ax, self.nested_sides, px, py,
                                  sub_sz * 0.45, inner_color, self.rotation,
                                  alpha=0.7)

# ------------------------------------------------------------------ #
# Sequence generators
# ------------------------------------------------------------------ #

def _seq_growing_size(rng, length):
    """Each step increases size."""
    start_idx = rng.randint(0, 2)
    color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    sides = rng.choice([3, 4, 5, 6])
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=sides, color_idx=color,
                              size_idx=start_idx + i, rotation=0))
    return cells

def _seq_rotating(rng, length):
    """Each step rotates by a fixed angle."""
    step = rng.choice([30, 45, 60, 90])
    sides = rng.choice([3, 4, 5])
    color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    size = rng.randint(2, 4)
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=sides, color_idx=color,
                              size_idx=size, rotation=step * i))
    return cells

def _seq_color_cycle(rng, length):
    """Each step changes color in a cycle."""
    start = rng.randint(0, len(_COLOR_CYCLE) - 1)
    sides = rng.choice([3, 4, 5, 6])
    size = rng.randint(2, 4)
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=sides, color_idx=(start + i) % len(_COLOR_CYCLE),
                              size_idx=size, rotation=0))
    return cells

def _seq_adding_sides(rng, length):
    """Each step adds a side: triangle -> square -> pentagon -> ..."""
    start = rng.choice([3, 4])
    color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    size = rng.randint(2, 4)
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=start + i, color_idx=color,
                              size_idx=size, rotation=0))
    return cells

def _seq_count_increment(rng, length):
    """Each step increases the number of shapes."""
    sides = rng.choice([3, 4, 5])
    color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    size = rng.randint(2, 3)
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=sides, color_idx=color,
                              size_idx=size, rotation=0,
                              count=min(i + 1, 4)))
    return cells

def _seq_nesting(rng, length):
    """Each step adds a nested inner shape."""
    outer_sides = rng.choice([4, 5, 6])
    color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    size = rng.randint(3, 5)
    cells = []
    inner_options = [0, 3, 4, 5, 6]
    for i in range(length):
        cells.append(_SeqCell(n_sides=outer_sides, color_idx=color,
                              size_idx=size, rotation=0,
                              nested_sides=inner_options[min(i, len(inner_options) - 1)]))
    return cells

def _seq_combined_size_rotation(rng, length):
    """Both size increases and shape rotates each step."""
    start_size = rng.randint(0, 2)
    rot_step = rng.choice([30, 45])
    sides = rng.choice([3, 4, 5])
    color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=sides, color_idx=color,
                              size_idx=start_size + i,
                              rotation=rot_step * i))
    return cells

def _seq_combined_color_sides(rng, length):
    """Color cycles AND sides increase."""
    start_sides = rng.choice([3, 4])
    start_color = rng.randint(0, len(_COLOR_CYCLE) - 1)
    size = rng.randint(2, 4)
    cells = []
    for i in range(length):
        cells.append(_SeqCell(n_sides=start_sides + i,
                              color_idx=(start_color + i) % len(_COLOR_CYCLE),
                              size_idx=size, rotation=0))
    return cells

_SEQUENCE_GENERATORS = [
    _seq_growing_size,
    _seq_rotating,
    _seq_color_cycle,
    _seq_adding_sides,
    _seq_count_increment,
    _seq_nesting,
    _seq_combined_size_rotation,
    _seq_combined_color_sides,
]

# Per-level generator pools and config
def _level_config(level):
    # Reordered: visually distinctive single-attribute patterns first,
    # then combined-attribute and ambiguous patterns later.
    # Fixes L3=0.40 (mixed) vs L6=0.80 (nesting=easy) inversion.
    if level == 0:
        return {"gens": [_seq_growing_size], "length": 3, "n_opts": 3}
    if level == 1:
        return {"gens": [_seq_adding_sides], "length": 3, "n_opts": 3}
    if level == 2:
        return {"gens": [_seq_nesting], "length": 3, "n_opts": 4}
    if level == 3:
        return {"gens": [_seq_color_cycle], "length": 3, "n_opts": 4}
    if level == 4:
        return {"gens": [_seq_count_increment], "length": 4, "n_opts": 4}
    if level == 5:
        return {"gens": [_seq_rotating], "length": 4, "n_opts": 4}
    if level == 6:
        return {"gens": [_seq_growing_size, _seq_color_cycle, _seq_adding_sides],
                "length": 4, "n_opts": 4}
    if level == 7:
        return {"gens": [_seq_combined_size_rotation], "length": 4, "n_opts": 5}
    if level == 8:
        return {"gens": [_seq_combined_color_sides], "length": 4, "n_opts": 5}
    return {"gens": _SEQUENCE_GENERATORS, "length": 5, "n_opts": 5}

class VisualSequenceQA(StandaloneVisualEnv):
    ENV_NAME = "visual_sequence"

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        cfg = _level_config(level)
        for _ in range(30):
            result = self._try_generate(sub_rng, cfg["length"], cfg["n_opts"],
                                         cfg["gens"])
            if result is not None:
                self._primary_complexity_feature = level * 5 + len(result[1])
                return result
        return None

    def _try_generate(self, rng, length, num_options, gens):
        gen_fn = rng.choice(gens)
        # Generate one extra cell to be the answer
        full_seq = gen_fn(rng, length + 1)
        if full_seq is None or len(full_seq) < length + 1:
            return None

        shown = full_seq[:length]
        correct = full_seq[length]

        # Generate distractors
        distractors = self._make_distractors(rng, correct, shown, num_options - 1)
        if len(distractors) < num_options - 1:
            return None

        options = list(distractors)
        correct_idx = rng.randint(0, len(options))
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)

        image = self._render_sequence(shown, options)

        opt_labels = ", ".join(chr(ord("A") + i) for i in range(len(options)))
        question = (
            f"Look at the sequence of shapes from left to right. "
            f"What comes next in the pattern? "
            f"Choose from options {opt_labels}. Answer with a single letter."
        )
        return question, answer_letter, image

    def _make_distractors(self, rng, correct, shown, num):
        distractors = []
        attempts = 0
        while len(distractors) < num and attempts < 200:
            attempts += 1
            d = correct.copy()
            changes = rng.randint(1, 2)
            for _ in range(changes):
                attr = rng.choice(["n_sides", "color_idx", "size_idx",
                                   "rotation", "count"])
                if attr == "n_sides":
                    d.n_sides = rng.choice([n for n in range(3, 9)
                                            if n != correct.n_sides])
                elif attr == "color_idx":
                    d.color_idx = rng.choice([i for i in range(len(_COLOR_CYCLE))
                                              if i != correct.color_idx])
                elif attr == "size_idx":
                    d.size_idx = rng.choice([i for i in range(len(_SIZES))
                                             if i != correct.size_idx])
                elif attr == "rotation":
                    d.rotation = (correct.rotation + rng.choice([30, 60, 90, 120])) % 360
                elif attr == "count":
                    d.count = rng.choice([c for c in [1, 2, 3, 4]
                                          if c != correct.count])
            if d != correct and d not in distractors:
                distractors.append(d)
        return distractors[:num]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_sequence(self, shown, options):
        style = self._random_style()
        sc = style["figsize_scale"]
        n_shown = len(shown)
        n_opts = len(options)
        total_top = n_shown + 1  # +1 for the "?" cell

        fig_w = max(total_top * 2.0 + 0.5, n_opts * 2.0 + 0.5) * sc
        fig_h = 5.5 * sc
        fig, (ax_seq, ax_opts) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1.5, 1.2]})
        fig.patch.set_facecolor(style["bg_color"])

        # -- Sequence row --
        ax_seq.set_xlim(0, total_top * 2.0)
        ax_seq.set_ylim(0, 2)
        ax_seq.set_aspect("equal")
        ax_seq.axis("off")
        ax_seq.set_title("What comes next?", fontsize=14, fontweight="bold", pad=8)

        cell_size = 1.7
        for i, cell in enumerate(shown):
            cx = i * 2.0 + 1.0
            cy = 1.0
            rect = mpatches.FancyBboxPatch(
                (cx - cell_size / 2, cy - cell_size / 2), cell_size, cell_size,
                boxstyle="round,pad=0.04", facecolor="#fdfefe",
                edgecolor="#bdc3c7", linewidth=1.5, zorder=1)
            ax_seq.add_patch(rect)
            cell.draw(ax_seq, cx, cy, cell_size)
            # Arrow between cells
            if i < n_shown - 1:
                ax_seq.annotate("", xy=(cx + 0.95, cy),
                                xytext=(cx + 0.65, cy),
                                arrowprops=dict(arrowstyle="->", lw=1.5,
                                                color="#95a5a6"))

        # "?" cell
        qx = n_shown * 2.0 + 1.0
        qy = 1.0
        # Arrow to "?"
        ax_seq.annotate("", xy=(qx - 0.35, qy),
                        xytext=(qx - 0.65, qy),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#95a5a6"))
        rect_q = mpatches.FancyBboxPatch(
            (qx - cell_size / 2, qy - cell_size / 2), cell_size, cell_size,
            boxstyle="round,pad=0.04", facecolor="#f9e79f",
            edgecolor="#e74c3c", linewidth=2.5, linestyle="--", zorder=1)
        ax_seq.add_patch(rect_q)
        ax_seq.text(qx, qy, "?", fontsize=28, fontweight="bold",
                    ha="center", va="center", color="#e74c3c", zorder=5)

        # -- Options row --
        ax_opts.set_xlim(0, n_opts * 2.0)
        ax_opts.set_ylim(0, 2)
        ax_opts.set_aspect("equal")
        ax_opts.axis("off")
        ax_opts.set_title("Options", fontsize=12, pad=4)

        opt_cell = 1.5
        for i, opt in enumerate(options):
            cx = i * 2.0 + 1.0
            cy = 1.0
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2), opt_cell, opt_cell,
                boxstyle="round,pad=0.04", facecolor="#eaf2f8",
                edgecolor="#2c3e50", linewidth=1.5, zorder=1)
            ax_opts.add_patch(rect)
            opt.draw(ax_opts, cx, cy, opt_cell)
            label = chr(ord("A") + i)
            ax_opts.text(cx, cy - opt_cell / 2 - 0.1, label,
                         fontsize=13, fontweight="bold", ha="center",
                         va="top", color="#2c3e50")

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
