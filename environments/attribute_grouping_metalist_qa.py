"""
Attribute Grouping Metalist QA environment.

Targets VisuLogic Attribute Reasoning.

Shows 6 simple line-art figures arranged in a 2×3 grid, numbered ①–⑥.
The model must find the single attribute (drawn from a meta-list such as
curvature, symmetry type, stroke count, enclosed region count, endpoint
count, or vertex count) that cleanly partitions the figures into two
groups of three.

Four MCQ options are provided: one correct partition + three plausible
distractor partitions (that mis-group the figures along a different
attribute axis or swap a single element).

**Critical design property**: the distinguishing attribute is *non-spatial*
— the model cannot cheat on positions, size, or row/column arrangement.
Rendering is line-art (no fill) to mimic IQ-test style.
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
# Figure prototypes, each tagged with multiple attribute values.
#
# Every prototype is a callable (ax, cx, cy, size, color, lw) -> None
# that draws the figure centred at (cx, cy) into the given axis.
# The outer key is the prototype name; the inner dict lists attribute
# values that the drawing satisfies.
# ------------------------------------------------------------------ #

def _draw_circle(ax, cx, cy, s, color, lw):
    c = mpatches.Circle((cx, cy), s * 0.9, facecolor="none",
                        edgecolor=color, linewidth=lw, zorder=3)
    ax.add_patch(c)

def _draw_ellipse(ax, cx, cy, s, color, lw):
    e = mpatches.Ellipse((cx, cy), s * 1.7, s * 1.05,
                         facecolor="none", edgecolor=color,
                         linewidth=lw, zorder=3)
    ax.add_patch(e)

def _draw_square(ax, cx, cy, s, color, lw):
    r = mpatches.Rectangle((cx - s * 0.85, cy - s * 0.85),
                           s * 1.7, s * 1.7,
                           facecolor="none", edgecolor=color,
                           linewidth=lw, zorder=3)
    ax.add_patch(r)

def _draw_triangle(ax, cx, cy, s, color, lw):
    verts = [(cx, cy + s * 0.95), (cx - s * 0.9, cy - s * 0.8),
             (cx + s * 0.9, cy - s * 0.8), (cx, cy + s * 0.95)]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
             mpath.Path.CLOSEPOLY]
    p = mpath.Path(verts, codes)
    ax.add_patch(mpatches.PathPatch(p, facecolor="none", edgecolor=color,
                                    linewidth=lw, zorder=3))

def _draw_pentagon(ax, cx, cy, s, color, lw):
    verts = []
    for i in range(5):
        a = math.pi / 2 + 2 * math.pi * i / 5
        verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 4 + [mpath.Path.CLOSEPOLY]
    p = mpath.Path(verts, codes)
    ax.add_patch(mpatches.PathPatch(p, facecolor="none", edgecolor=color,
                                    linewidth=lw, zorder=3))

def _draw_hexagon(ax, cx, cy, s, color, lw):
    verts = []
    for i in range(6):
        a = math.pi / 2 + 2 * math.pi * i / 6
        verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 5 + [mpath.Path.CLOSEPOLY]
    p = mpath.Path(verts, codes)
    ax.add_patch(mpatches.PathPatch(p, facecolor="none", edgecolor=color,
                                    linewidth=lw, zorder=3))

def _draw_crescent(ax, cx, cy, s, color, lw):
    # Crescent is outer arc minus inner arc offset; we draw two overlapping
    # circle arcs without filling → looks like a moon outline.
    theta = np.linspace(math.pi * 0.2, math.pi * 1.8, 80)
    ox = cx + s * 0.9 * np.cos(theta)
    oy = cy + s * 0.9 * np.sin(theta)
    ax.plot(ox, oy, color=color, linewidth=lw, zorder=3)
    theta2 = np.linspace(math.pi * 0.25, math.pi * 1.75, 80)
    ix = cx + 0.25 * s + s * 0.8 * np.cos(theta2)
    iy = cy + s * 0.8 * np.sin(theta2)
    ax.plot(ix, iy, color=color, linewidth=lw, zorder=3)

def _draw_spiral(ax, cx, cy, s, color, lw):
    # Open spiral — smoothly curved, has 2 endpoints, no closed region.
    theta = np.linspace(0, 3.2 * math.pi, 160)
    r = np.linspace(0.08 * s, 0.95 * s, 160)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_s_curve(ax, cx, cy, s, color, lw):
    # An S-shaped open curve (2 endpoints, curvy, no straight edges).
    t = np.linspace(-1, 1, 80)
    x = cx + t * s * 0.9
    y = cy + s * 0.75 * np.sin(t * math.pi)
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_wave(ax, cx, cy, s, color, lw):
    t = np.linspace(-1, 1, 80)
    x = cx + t * s * 1.05
    y = cy + s * 0.55 * np.sin(2 * math.pi * t)
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_l_shape(ax, cx, cy, s, color, lw):
    # L polyline — 3 straight segments, 2 endpoints, 0 closed regions.
    x = [cx - s * 0.9, cx - s * 0.9, cx + s * 0.9]
    y = [cy + s * 0.9, cy - s * 0.9, cy - s * 0.9]
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_z_shape(ax, cx, cy, s, color, lw):
    # Z polyline — 3 straight segments, 2 endpoints, 0 closed regions.
    x = [cx - s * 0.9, cx + s * 0.9, cx - s * 0.9, cx + s * 0.9]
    y = [cy + s * 0.9, cy + s * 0.9, cy - s * 0.9, cy - s * 0.9]
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_plus(ax, cx, cy, s, color, lw):
    # Plus sign — 4 endpoints, 2 straight segments, 0 closed regions.
    ax.plot([cx - s * 0.9, cx + s * 0.9], [cy, cy],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx], [cy - s * 0.9, cy + s * 0.9],
            color=color, linewidth=lw, zorder=3)

def _draw_t_shape(ax, cx, cy, s, color, lw):
    # T polyline — 3 endpoints, 0 closed regions, all straight.
    ax.plot([cx - s * 0.9, cx + s * 0.9], [cy + s * 0.8, cy + s * 0.8],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx], [cy + s * 0.8, cy - s * 0.9],
            color=color, linewidth=lw, zorder=3)

def _draw_y_shape(ax, cx, cy, s, color, lw):
    # Y polyline — 3 endpoints, straight, 0 closed regions.
    ax.plot([cx, cx], [cy - s * 0.9, cy],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx - s * 0.8], [cy, cy + s * 0.85],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx + s * 0.8], [cy, cy + s * 0.85],
            color=color, linewidth=lw, zorder=3)

def _draw_heart(ax, cx, cy, s, color, lw):
    t = np.linspace(0, 2 * math.pi, 120)
    x = cx + s * 0.05 * (16 * np.sin(t) ** 3)
    y = cy + s * 0.05 * (13 * np.cos(t) - 5 * np.cos(2 * t)
                         - 2 * np.cos(3 * t) - np.cos(4 * t))
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_arc(ax, cx, cy, s, color, lw):
    # Single arc (half-circle) — 2 endpoints, curvy, 0 closed regions.
    theta = np.linspace(0, math.pi, 80)
    x = cx + s * 0.9 * np.cos(theta)
    y = cy - s * 0.2 + s * 0.9 * np.sin(theta)
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_v_shape(ax, cx, cy, s, color, lw):
    # V polyline — 2 endpoints, 2 straight segments, 0 closed regions.
    ax.plot([cx - s * 0.85, cx, cx + s * 0.85],
            [cy + s * 0.9, cy - s * 0.9, cy + s * 0.9],
            color=color, linewidth=lw, zorder=3)

def _draw_diamond(ax, cx, cy, s, color, lw):
    verts = [(cx, cy + s), (cx + s, cy), (cx, cy - s), (cx - s, cy), (cx, cy + s)]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
             mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    p = mpath.Path(verts, codes)
    ax.add_patch(mpatches.PathPatch(p, facecolor="none", edgecolor=color,
                                    linewidth=lw, zorder=3))

# ------------------------------------------------------------------ #
# Figure catalogue.
#
# Each figure has 6 attribute axes. The "splitter" attributes we train
# on are non-spatial features:
#   curvy:  bool   - drawn with curves (not all straight lines)
#   closed_regions: int  - number of enclosed interior regions
#   endpoints:      int  - free endpoints (0 == closed path)
#   corners:        int  - sharp corners/vertices
#   axes_symmetry:  int  - number of reflection symmetry axes
#   strokes:        int  - minimum connected strokes to draw w/o lifting pen
# ------------------------------------------------------------------ #

_FIGURE_CATALOGUE = {
    "circle":   {"fn": _draw_circle,   "curvy": True,  "closed_regions": 1,
                 "endpoints": 0, "corners": 0, "axes_symmetry": 4, "strokes": 1},
    "ellipse":  {"fn": _draw_ellipse,  "curvy": True,  "closed_regions": 1,
                 "endpoints": 0, "corners": 0, "axes_symmetry": 2, "strokes": 1},
    "square":   {"fn": _draw_square,   "curvy": False, "closed_regions": 1,
                 "endpoints": 0, "corners": 4, "axes_symmetry": 4, "strokes": 1},
    "triangle": {"fn": _draw_triangle, "curvy": False, "closed_regions": 1,
                 "endpoints": 0, "corners": 3, "axes_symmetry": 1, "strokes": 1},
    "pentagon": {"fn": _draw_pentagon, "curvy": False, "closed_regions": 1,
                 "endpoints": 0, "corners": 5, "axes_symmetry": 1, "strokes": 1},
    "hexagon":  {"fn": _draw_hexagon,  "curvy": False, "closed_regions": 1,
                 "endpoints": 0, "corners": 6, "axes_symmetry": 4, "strokes": 1},
    "diamond":  {"fn": _draw_diamond,  "curvy": False, "closed_regions": 1,
                 "endpoints": 0, "corners": 4, "axes_symmetry": 2, "strokes": 1},
    "crescent": {"fn": _draw_crescent, "curvy": True,  "closed_regions": 0,
                 "endpoints": 4, "corners": 0, "axes_symmetry": 1, "strokes": 2},
    "spiral":   {"fn": _draw_spiral,   "curvy": True,  "closed_regions": 0,
                 "endpoints": 2, "corners": 0, "axes_symmetry": 0, "strokes": 1},
    "s_curve":  {"fn": _draw_s_curve,  "curvy": True,  "closed_regions": 0,
                 "endpoints": 2, "corners": 0, "axes_symmetry": 0, "strokes": 1},
    "wave":     {"fn": _draw_wave,     "curvy": True,  "closed_regions": 0,
                 "endpoints": 2, "corners": 0, "axes_symmetry": 0, "strokes": 1},
    "arc":      {"fn": _draw_arc,      "curvy": True,  "closed_regions": 0,
                 "endpoints": 2, "corners": 0, "axes_symmetry": 1, "strokes": 1},
    "l_shape":  {"fn": _draw_l_shape,  "curvy": False, "closed_regions": 0,
                 "endpoints": 2, "corners": 1, "axes_symmetry": 0, "strokes": 1},
    "z_shape":  {"fn": _draw_z_shape,  "curvy": False, "closed_regions": 0,
                 "endpoints": 2, "corners": 2, "axes_symmetry": 0, "strokes": 1},
    "v_shape":  {"fn": _draw_v_shape,  "curvy": False, "closed_regions": 0,
                 "endpoints": 2, "corners": 2, "axes_symmetry": 1, "strokes": 1},
    "t_shape":  {"fn": _draw_t_shape,  "curvy": False, "closed_regions": 0,
                 "endpoints": 3, "corners": 1, "axes_symmetry": 1, "strokes": 2},
    "y_shape":  {"fn": _draw_y_shape,  "curvy": False, "closed_regions": 0,
                 "endpoints": 3, "corners": 0, "axes_symmetry": 1, "strokes": 2},
    "plus":     {"fn": _draw_plus,     "curvy": False, "closed_regions": 0,
                 "endpoints": 4, "corners": 0, "axes_symmetry": 4, "strokes": 2},
    "heart":    {"fn": _draw_heart,    "curvy": True,  "closed_regions": 1,
                 "endpoints": 0, "corners": 1, "axes_symmetry": 1, "strokes": 1},
}

# Splitter attribute → how to bucket a value into a binary label.
# Returns an int (0 or 1) to mark which group the figure goes into.
def _bucket_curvy(fig):
    return 1 if fig["curvy"] else 0

def _bucket_closed(fig):
    return 1 if fig["closed_regions"] >= 1 else 0

def _bucket_endpoints_zero(fig):
    return 1 if fig["endpoints"] == 0 else 0

def _bucket_even_corners(fig):
    return 1 if fig["corners"] % 2 == 0 and fig["corners"] > 0 else 0

def _bucket_sym_high(fig):
    return 1 if fig["axes_symmetry"] >= 2 else 0

def _bucket_strokes_one(fig):
    return 1 if fig["strokes"] == 1 else 0

_ATTRIBUTES = [
    ("curvy_vs_straight",   "curves vs straight lines",
        _bucket_curvy,              {"easy": True}),
    ("closed_vs_open",      "closed shape vs open figure",
        _bucket_closed,             {"easy": True}),
    ("has_endpoints",       "no free endpoints vs has free endpoints",
        _bucket_endpoints_zero,     {"easy": True}),
    ("even_corners",        "even number of corners vs otherwise",
        _bucket_even_corners,       {"easy": False}),
    ("high_symmetry",       "two or more reflection axes vs fewer",
        _bucket_sym_high,           {"easy": False}),
    ("strokes_one",         "drawn in one continuous stroke vs more",
        _bucket_strokes_one,        {"easy": False}),
]

# Figure layout on canvas (2 rows × 3 cols).
_GRID_CELLS = [
    (0, 0), (1, 0), (2, 0),  # row 0 (top)
    (0, 1), (1, 1), (2, 1),  # row 1 (bottom)
]

class AttributeGroupingMetalistQA(StandaloneVisualEnv):
    """6-figure attribute partition task (A4 attribute grouping / metalist)."""

    ENV_NAME = "attribute_grouping_metalist"

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Return difficulty config for a given level (0-9)."""
        if level <= 0:
            return {
                "attr_keys": ["closed_vs_open"],
                "n_total": 6,
                "min_unique_shapes": 4,
            }
        if level == 1:
            return {
                "attr_keys": ["closed_vs_open", "curvy_vs_straight"],
                "n_total": 6,
                "min_unique_shapes": 5,
            }
        if level == 2:
            return {
                "attr_keys": ["curvy_vs_straight"],
                "n_total": 6,
                "min_unique_shapes": 5,
            }
        if level == 3:
            return {
                "attr_keys": ["curvy_vs_straight", "closed_vs_open"],
                "n_total": 6,
                "min_unique_shapes": 5,
            }
        if level == 4:
            return {
                "attr_keys": ["has_endpoints", "curvy_vs_straight"],
                "n_total": 8,
                "min_unique_shapes": 7,
            }
        if level == 5:
            return {
                "attr_keys": ["has_endpoints"],
                "n_total": 8,
                "min_unique_shapes": 7,
            }
        if level == 6:
            return {
                "attr_keys": ["high_symmetry", "even_corners"],
                "n_total": 8,
                "min_unique_shapes": 7,
            }
        if level == 7:
            return {
                "attr_keys": ["even_corners", "high_symmetry"],
                "n_total": 8,
                "min_unique_shapes": 7,
            }
        if level == 8:
            return {
                "attr_keys": ["strokes_one", "high_symmetry"],
                "n_total": 10,
                "min_unique_shapes": 9,
            }
        # level >= 9
        return {
            "attr_keys": ["strokes_one", "even_corners", "high_symmetry"],
            "n_total": 10,
            "min_unique_shapes": 9,
        }

    # ------------------------------------------------------------------ #
    # Question phrasing and title pools
    # ------------------------------------------------------------------ #

    _QUESTION_TEMPLATES = [
        "Divide the {n} figures (numbered 1-{n}) into two groups of {g} so that each group shares one common feature. Which grouping is correct?\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "The image shows {n} line-art figures labelled 1 through {n}. Group them into two sets of {g} based on a shared visual property. Which option is correct?\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "Look at the {n} figures numbered 1 to {n}. One attribute cleanly splits them into two groups of {g}. Which partition below is correct?\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "Which of the following correctly splits figures 1-{n} into two equal groups of {g} that each share a common geometric property?\n{opts}\nAnswer with a single letter A, B, C, or D.",
    ]

    _TITLE_VARIANTS = [
        "{n} figures:",
        "Figure set ({n}):",
        "Shapes 1 \u2013 {n}",
        "Figures",
        "Line-art set",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)

        # Sub-RNG for visual variation independent of answer RNG
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        attr_pool = [a for a in _ATTRIBUTES if a[0] in cfg["attr_keys"]]
        if not attr_pool:
            attr_pool = _ATTRIBUTES

        n_total = cfg["n_total"]
        n_per_group = n_total // 2

        for _ in range(50):
            result = self._try_generate(rng, sub_rng, level, cfg, attr_pool, n_per_group)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, sub_rng, level, cfg, attr_pool, n_per_group=3):
        attr_key, attr_name, bucket_fn, _ = rng.choice(attr_pool)
        n_total = n_per_group * 2

        # Partition all figures into bucket 0 / bucket 1
        group0_names, group1_names = [], []
        for name, fig in _FIGURE_CATALOGUE.items():
            (group1_names if bucket_fn(fig) == 1 else group0_names).append(name)

        if len(group0_names) < n_per_group or len(group1_names) < n_per_group:
            return None

        picked0 = rng.sample(group0_names, n_per_group)
        picked1 = rng.sample(group1_names, n_per_group)

        # Quality filter -- enforce diversity
        all_picked = picked0 + picked1
        if len(set(all_picked)) < min(cfg["min_unique_shapes"], n_total):
            return None

        # Assign figures to positions randomly
        positions = list(range(n_total))
        rng.shuffle(positions)
        figure_at_pos = [None] * n_total
        for p, name in zip(positions[:n_per_group], picked0):
            figure_at_pos[p] = (name, 0)
        for p, name in zip(positions[n_per_group:], picked1):
            figure_at_pos[p] = (name, 1)

        # Correct partition
        group0_positions = sorted(p for p, (_, g) in enumerate(figure_at_pos) if g == 0)
        group1_positions = sorted(p for p, (_, g) in enumerate(figure_at_pos) if g == 1)
        correct_partition = frozenset([frozenset(group0_positions),
                                        frozenset(group1_positions)])

        distractors = self._build_distractors(rng, figure_at_pos,
                                              correct_partition, attr_pool,
                                              attr_key, n_per_group)
        if len(distractors) < 3:
            return None

        # Assemble options
        all_partitions = [correct_partition] + distractors[:3]
        rng.shuffle(all_partitions)
        correct_idx = all_partitions.index(correct_partition)
        answer_letter = chr(ord("A") + correct_idx)

        # Render with sub_rng for visual diversity
        title = sub_rng.choice(self._TITLE_VARIANTS).format(n=n_total)
        image = self._render_problem(figure_at_pos, all_partitions, attr_name,
                                     title=title, sub_rng=sub_rng)

        opt_lines = []
        for i, part in enumerate(all_partitions):
            parts = sorted(part, key=lambda s: min(s))
            left = ",".join(self._to_circled(p + 1) for p in sorted(parts[0]))
            right = ",".join(self._to_circled(p + 1) for p in sorted(parts[1]))
            opt_lines.append(f"  {chr(ord('A') + i)}) {{{left}}} | {{{right}}}")
        opts_text = "\n".join(opt_lines)

        # Pick question template from sub_rng for seed-level variation
        q_template = sub_rng.choice(self._QUESTION_TEMPLATES)
        question = q_template.format(n=n_total, g=n_per_group, opts=opts_text)

        self._primary_complexity_feature = n_total + level * 2
        return question, answer_letter, image

    def _build_distractors(self, rng, figure_at_pos, correct, attr_pool,
                            used_attr_key, n_per_group=3):
        """Build 3 plausible-but-wrong balanced partitions of N positions."""
        from itertools import combinations
        seen = {correct}
        distractors = []
        n_total = n_per_group * 2

        # Distractor source 1: split by some OTHER attribute whose bucket
        # does NOT yield a balanced split — force a balanced split by moving
        # figures across groups (produces a plausible-looking split).
        for attr_key, _, bucket_fn, _ in _ATTRIBUTES:
            if attr_key == used_attr_key:
                continue
            g0, g1 = [], []
            for p, (name, _) in enumerate(figure_at_pos):
                if bucket_fn(_FIGURE_CATALOGUE[name]) == 1:
                    g1.append(p)
                else:
                    g0.append(p)
            # Balance to n/n if not already
            while len(g0) > n_per_group:
                g1.append(g0.pop(rng.randint(0, len(g0) - 1)))
            while len(g1) > n_per_group:
                g0.append(g1.pop(rng.randint(0, len(g1) - 1)))
            if len(g0) != n_per_group or len(g1) != n_per_group:
                continue
            part = frozenset([frozenset(g0), frozenset(g1)])
            if part in seen:
                continue
            seen.add(part)
            distractors.append(part)
            if len(distractors) >= 3:
                return distractors

        # Fall back: single-swap distractors from the correct partition.
        correct_groups = list(correct)
        gA = sorted(list(correct_groups[0]))
        gB = sorted(list(correct_groups[1]))

        attempts = 0
        while len(distractors) < 3 and attempts < 100:
            attempts += 1
            a_i = rng.randint(0, n_per_group - 1)
            b_i = rng.randint(0, n_per_group - 1)
            new_A = list(gA)
            new_B = list(gB)
            new_A[a_i], new_B[b_i] = new_B[b_i], new_A[a_i]
            part = frozenset([frozenset(new_A), frozenset(new_B)])
            if part in seen:
                continue
            seen.add(part)
            distractors.append(part)
        return distractors

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_problem(self, figure_at_pos, options, attr_name,
                        title: str = "Figures", sub_rng=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        line_color = style["geo_line_color"]
        lw = max(1.8, style["line_width"])

        # Per-seed line color variation via sub_rng
        _line_colors = ["#2c3e50", "#1a5276", "#7b241c", "#0d6efd",
                        "#198754", "#6c3483", "#1b4f72", "#922b21"]
        if sub_rng:
            line_color = sub_rng.choice(_line_colors)
            lw = 1.8 + sub_rng.random() * 1.2

        n_total = len(figure_at_pos)
        ncols = n_total // 2
        nrows = 2

        fig_w = max(7.5, 2.0 + 1.25 * ncols) * sc
        fig = plt.figure(figsize=(fig_w, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.4], hspace=0.25)
        ax_fig = fig.add_subplot(gs[0])
        ax_opt = fig.add_subplot(gs[1])

        ax_fig.set_xlim(0, ncols)
        ax_fig.set_ylim(0, nrows)
        ax_fig.set_aspect("equal")
        ax_fig.axis("off")
        ax_fig.set_title(title,
                         fontsize=fs + 2, fontweight="bold", pad=6,
                         fontfamily=ff)

        for pos, (name, _) in enumerate(figure_at_pos):
            col = pos % ncols
            row = pos // ncols
            cx = col + 0.5
            cy = (nrows - 1 - row) + 0.5
            rect = mpatches.FancyBboxPatch(
                (cx - 0.42, cy - 0.42), 0.84, 0.84,
                boxstyle="round,pad=0.02",
                facecolor=style["bg_color"], edgecolor="#bdc3c7",
                linewidth=1.2, zorder=1)
            ax_fig.add_patch(rect)
            _FIGURE_CATALOGUE[name]["fn"](
                ax_fig, cx, cy, 0.32, line_color, lw)
            ax_fig.text(cx, cy - 0.48, str(pos + 1),
                         fontsize=fs + 1, fontweight="bold",
                         ha="center", va="top", color="#2c3e50",
                         fontfamily=ff)

        # --- options panel ---
        ax_opt.set_xlim(0, 1)
        ax_opt.set_ylim(0, 1)
        ax_opt.axis("off")

        n_opts = len(options)
        for i, part in enumerate(options):
            parts = sorted(part, key=lambda s: min(s))
            left = ", ".join(str(p + 1) for p in sorted(parts[0]))
            right = ", ".join(str(p + 1) for p in sorted(parts[1]))
            y = 0.85 - i * (0.85 / n_opts)
            ax_opt.text(0.02, y, f"{chr(ord('A') + i)}. ",
                         fontsize=fs + 2, fontweight="bold",
                         color="#2c3e50", va="center", fontfamily=ff)
            ax_opt.text(0.08, y,
                         f"{{{left}}}  |  {{{right}}}",
                         fontsize=fs + 1, color="#2c3e50",
                         va="center", fontfamily=ff)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_circled(n: int) -> str:
        """Return the plain digit (kept as number for font compatibility)."""
        return str(n)

# ------------------------------------------------------------------ #
# Smoke-test entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = AttributeGroupingMetalistQA()
    for level in [0, 3, 6]:
        for seed in range(3):
            ok = env.generate(seed=seed * 100 + level, parameter={"level": level})
            if not ok:
                print(f"FAILED seed={seed} level={level}")
                continue
            img = env.render()
            img.save(f"/tmp/env_check/attribute_grouping_metalist_seed{seed}_L{level}.png")
            print(f"seed={seed} level={level}")
            print("Q:", env.get_instruction().splitlines()[0])
            print("A:", env._answer)
            print()
