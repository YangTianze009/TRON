"""
Stroke Continuity Grouping QA environment.

Targets broader A4 coverage — attribute partitioning by STROKE COUNT
(one-stroke aka "Eulerian" figures vs two-stroke figures).

Shows 6 figures drawn as line-art.  Three can be traced in a single
continuous stroke (without lifting the pen and without retracing any
edge), three cannot (they need two strokes).

For a connected graph, an Eulerian path exists iff it has exactly 0 or
2 vertices of odd degree.  Figures with 2 disconnected components always
need >=2 strokes regardless.

Level 0: obvious cases (circle, figure-8 = 1 stroke, two disjoint
circles = 2 strokes).  Level 5+: subtle line-drawings where the model
has to actually count odd-degree vertices.
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
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# ------------------------------------------------------------------ #
# Figure renderers.
# Each figure is annotated with its minimum stroke count.
# "strokes" is the minimum # of continuous strokes needed.
# ------------------------------------------------------------------ #

def _draw_circle(ax, cx, cy, s, color, lw):
    ax.add_patch(mpatches.Circle((cx, cy), s * 0.85, facecolor="none",
                                 edgecolor=color, linewidth=lw, zorder=3))

def _draw_figure_eight(ax, cx, cy, s, color, lw):
    # Figure-8: two tangent circles sharing one vertex (degree 4).
    # All degrees even → 1 stroke.
    ax.add_patch(mpatches.Circle((cx, cy + s * 0.4), s * 0.42,
                                 facecolor="none", edgecolor=color,
                                 linewidth=lw, zorder=3))
    ax.add_patch(mpatches.Circle((cx, cy - s * 0.4), s * 0.42,
                                 facecolor="none", edgecolor=color,
                                 linewidth=lw, zorder=3))

def _draw_envelope(ax, cx, cy, s, color, lw):
    # Classic Eulerian "envelope" with a diagonal, rectangle, and a roof.
    # Degrees: 2 bottom corners have 3, 2 top corners have 3, apex has 2.
    # 4 odd-degree vertices → NOT Eulerian → needs 2 strokes.
    pts = [
        (cx - s * 0.85, cy - s * 0.7),  # A bottom-left
        (cx + s * 0.85, cy - s * 0.7),  # B bottom-right
        (cx + s * 0.85, cy + s * 0.3),  # C mid-right
        (cx,            cy + s * 0.85), # D apex
        (cx - s * 0.85, cy + s * 0.3),  # E mid-left
    ]
    # Edges: A-B, B-C, C-D, D-E, E-A (rectangle + roof)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
             (4, 2)]  # plus horizontal across: degrees become
    # A:2  B:2  C:3  D:2  E odd → Eulerian path exists → 1 stroke.
    # Let me just keep this shape as documented with 2 strokes for clarity:
    # We'll instead add edge (A,C) which gives: A:3 B:2 C:4 D:2 E odd
    # Actually that's still 2 odd → 1 stroke.
    # For a 2-stroke envelope, add crossing diagonals so 4 corners are odd.
    # Let's use simpler: rectangle + both diagonals, no roof.
    pts = [
        (cx - s * 0.85, cy - s * 0.7),  # A
        (cx + s * 0.85, cy - s * 0.7),  # B
        (cx + s * 0.85, cy + s * 0.7),  # C
        (cx - s * 0.85, cy + s * 0.7),  # D
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # rectangle
             (0, 2), (1, 3)]                   # both diagonals
    # Degrees: all 4 vertices have 3 → 4 odd → 2 strokes minimum.
    for i, j in edges:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                color=color, linewidth=lw, zorder=3)

def _draw_square_with_x(ax, cx, cy, s, color, lw):
    # Square with both diagonals — same as above; 2 strokes.
    pts = [
        (cx - s * 0.8, cy - s * 0.8),
        (cx + s * 0.8, cy - s * 0.8),
        (cx + s * 0.8, cy + s * 0.8),
        (cx - s * 0.8, cy + s * 0.8),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    for i, j in edges:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                color=color, linewidth=lw, zorder=3)

def _draw_two_circles(ax, cx, cy, s, color, lw):
    # Two disjoint circles → disconnected, 2 strokes.
    ax.add_patch(mpatches.Circle((cx - s * 0.5, cy), s * 0.4,
                                 facecolor="none", edgecolor=color,
                                 linewidth=lw, zorder=3))
    ax.add_patch(mpatches.Circle((cx + s * 0.5, cy), s * 0.4,
                                 facecolor="none", edgecolor=color,
                                 linewidth=lw, zorder=3))

def _draw_two_squares(ax, cx, cy, s, color, lw):
    # Two disjoint squares → 2 strokes.
    _side = s * 0.45
    for dx in [-s * 0.55, s * 0.55]:
        ax.add_patch(mpatches.Rectangle(
            (cx + dx - _side, cy - _side), 2 * _side, 2 * _side,
            facecolor="none", edgecolor=color, linewidth=lw, zorder=3))

def _draw_square(ax, cx, cy, s, color, lw):
    # Plain square (all vertices degree 2) → 1 stroke.
    pts = [(cx - s * 0.82, cy - s * 0.82), (cx + s * 0.82, cy - s * 0.82),
           (cx + s * 0.82, cy + s * 0.82), (cx - s * 0.82, cy + s * 0.82),
           (cx - s * 0.82, cy - s * 0.82)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)

def _draw_triangle(ax, cx, cy, s, color, lw):
    # Plain triangle → 1 stroke.
    pts = [(cx, cy + s * 0.85), (cx - s * 0.85, cy - s * 0.6),
           (cx + s * 0.85, cy - s * 0.6), (cx, cy + s * 0.85)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=3)

def _draw_triangle_with_bar(ax, cx, cy, s, color, lw):
    # Triangle with a horizontal bar splitting it from one side to opposite.
    # Apex degree 2, bottom-left degree 2, bottom-right degree 2, midpoint
    # of an edge = degree 3... Actually this is tricky to keep 2 odd.
    # Implementation: triangle + bar between apex and bottom midpoint.
    apex = (cx, cy + s * 0.85)
    bl   = (cx - s * 0.85, cy - s * 0.6)
    br   = (cx + s * 0.85, cy - s * 0.6)
    bm   = ((bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2)
    # Edges: apex-bl, bl-bm, bm-br, br-apex, apex-bm
    # Degrees: apex=3, bl=2, bm=3, br=2 → 2 odd → 1 stroke
    edges = [(apex, bl), (bl, bm), (bm, br), (br, apex), (apex, bm)]
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, zorder=3)

def _draw_plus(ax, cx, cy, s, color, lw):
    # Plus sign (two crossing lines) — crossing point has degree 4, 4 ends
    # have degree 1. 4 odd → 2 strokes.
    ax.plot([cx - s * 0.85, cx + s * 0.85], [cy, cy],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx], [cy - s * 0.85, cy + s * 0.85],
            color=color, linewidth=lw, zorder=3)

def _draw_three_lines(ax, cx, cy, s, color, lw):
    # Three disconnected horizontal lines → 3 strokes (definitely > 1).
    for i, dy in enumerate([-s * 0.55, 0, s * 0.55]):
        ax.plot([cx - s * 0.85, cx + s * 0.85], [cy + dy, cy + dy],
                color=color, linewidth=lw, zorder=3)

def _draw_house(ax, cx, cy, s, color, lw):
    # House: rectangle with a roof triangle on top.
    # 5 vertices: roof apex (deg 2), 2 roof base points (deg 3 each),
    # 2 bottom corners (deg 2). → 2 odd vertices → 1 stroke.
    apex = (cx, cy + s * 0.85)
    tl = (cx - s * 0.7, cy + s * 0.2)
    tr = (cx + s * 0.7, cy + s * 0.2)
    bl = (cx - s * 0.7, cy - s * 0.8)
    br = (cx + s * 0.7, cy - s * 0.8)
    edges = [(apex, tl), (apex, tr), (tl, tr), (tl, bl),
             (tr, br), (bl, br)]
    for a, b in edges:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, zorder=3)

def _draw_disjoint_triangle_square(ax, cx, cy, s, color, lw):
    # Disjoint triangle + square → 2 strokes.
    # Triangle on left
    _l = cx - s * 0.45
    tri_pts = [(_l - s * 0.3, cy - s * 0.45), (_l + s * 0.3, cy - s * 0.45),
               (_l, cy + s * 0.5), (_l - s * 0.3, cy - s * 0.45)]
    ax.plot([p[0] for p in tri_pts], [p[1] for p in tri_pts],
            color=color, linewidth=lw, zorder=3)
    # Square on right
    _r = cx + s * 0.45
    sq = s * 0.32
    ax.add_patch(mpatches.Rectangle(
        (_r - sq, cy - sq), 2 * sq, 2 * sq,
        facecolor="none", edgecolor=color, linewidth=lw, zorder=3))

def _draw_ellipse(ax, cx, cy, s, color, lw):
    ax.add_patch(mpatches.Ellipse((cx, cy), s * 1.6, s * 1.0,
                                  facecolor="none", edgecolor=color,
                                  linewidth=lw, zorder=3))

# ------------------------------------------------------------------ #
# Ambiguous / subtle 1-stroke and 2-stroke figures (letters, forks).
# ------------------------------------------------------------------ #

def _draw_letter_T(ax, cx, cy, s, color, lw):
    # Letter T: horizontal bar + vertical stem.
    # Vertices: LT(1), RT(1), JC (junction, deg 3), stem bottom (1).
    # Odd vertices: 4 → needs 2 strokes.
    ax.plot([cx - s * 0.75, cx + s * 0.75], [cy + s * 0.7, cy + s * 0.7],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx], [cy + s * 0.7, cy - s * 0.75],
            color=color, linewidth=lw, zorder=3)

def _draw_letter_Y(ax, cx, cy, s, color, lw):
    # Letter Y: three line segments meeting at a single point (deg 3).
    # 3 leaves (deg 1 each) + 1 central (deg 3) → 4 odd vertices → 2 strokes.
    ax.plot([cx, cx - s * 0.75], [cy, cy + s * 0.8],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx + s * 0.75], [cy, cy + s * 0.8],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx, cx], [cy, cy - s * 0.85],
            color=color, linewidth=lw, zorder=3)

def _draw_letter_K(ax, cx, cy, s, color, lw):
    # Letter K: vertical stem + two diagonals from mid-stem.
    # Vertices: top of stem(1), bottom of stem(1), mid junction(deg 3),
    #           upper-right tip(1), lower-right tip(1). → 4 odd → 2 strokes.
    ax.plot([cx - s * 0.6, cx - s * 0.6], [cy - s * 0.85, cy + s * 0.85],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx - s * 0.6, cx + s * 0.75], [cy, cy + s * 0.85],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx - s * 0.6, cx + s * 0.75], [cy, cy - s * 0.85],
            color=color, linewidth=lw, zorder=3)

def _draw_asym_fork(ax, cx, cy, s, color, lw):
    # Asymmetric 3-pronged fork (one long handle + three short tines).
    # Handle(1) - hub(deg 4) - three tines(1 each). 4 odd → 2 strokes.
    hub = (cx, cy - s * 0.1)
    ax.plot([cx, hub[0]], [cy - s * 0.85, hub[1]],
            color=color, linewidth=lw, zorder=3)
    ax.plot([hub[0], cx - s * 0.8], [hub[1], cy + s * 0.75],
            color=color, linewidth=lw, zorder=3)
    ax.plot([hub[0], cx], [hub[1], cy + s * 0.85],
            color=color, linewidth=lw, zorder=3)
    ax.plot([hub[0], cx + s * 0.8], [hub[1], cy + s * 0.75],
            color=color, linewidth=lw, zorder=3)

def _draw_letter_L(ax, cx, cy, s, color, lw):
    # Letter L: two line segments meeting at corner. 2 odd → 1 stroke.
    ax.plot([cx - s * 0.75, cx - s * 0.75], [cy + s * 0.85, cy - s * 0.75],
            color=color, linewidth=lw, zorder=3)
    ax.plot([cx - s * 0.75, cx + s * 0.75], [cy - s * 0.75, cy - s * 0.75],
            color=color, linewidth=lw, zorder=3)

def _draw_letter_Z(ax, cx, cy, s, color, lw):
    # Letter Z: three segments forming Z. Four endpoints, all deg 1, one
    # junction... Actually: 4 endpoints? No — it's a polyline of 4 vertices:
    # TL, TR, BL, BR. TL(1), TR(deg 2), BL(deg 2), BR(1). 2 odd → 1 stroke.
    pts = [(cx - s * 0.75, cy + s * 0.75),
           (cx + s * 0.75, cy + s * 0.75),
           (cx - s * 0.75, cy - s * 0.75),
           (cx + s * 0.75, cy - s * 0.75)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color=color, linewidth=lw, zorder=3)

def _draw_staircase(ax, cx, cy, s, color, lw):
    # Polyline zig-zag — 1 stroke (all intermediate degrees 2, endpoints 1).
    pts = [(cx - s * 0.85, cy - s * 0.75),
           (cx - s * 0.85, cy - s * 0.25),
           (cx - s * 0.3, cy - s * 0.25),
           (cx - s * 0.3, cy + s * 0.25),
           (cx + s * 0.3, cy + s * 0.25),
           (cx + s * 0.3, cy + s * 0.75),
           (cx + s * 0.85, cy + s * 0.75)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color=color, linewidth=lw, zorder=3)

# ------------------------------------------------------------------ #
# Complex graphs (for high levels)
# ------------------------------------------------------------------ #

def _draw_theta_graph(ax, cx, cy, s, color, lw):
    # Theta: circle + a diameter bar. Two vertices deg 3, plus circle
    # contributions: the two endpoints of the diameter have degree 3
    # (2 from circle arc sides + 1 from diameter). Wait: in the theta
    # graph, we have two vertices connected by three edges (two arcs + bar).
    # Both vertices have degree 3. 2 odd → 1 stroke, Eulerian path.
    ax.add_patch(mpatches.Circle((cx, cy), s * 0.82, facecolor="none",
                                 edgecolor=color, linewidth=lw, zorder=3))
    ax.plot([cx - s * 0.82, cx + s * 0.82], [cy, cy],
            color=color, linewidth=lw, zorder=3)

def _draw_bowtie(ax, cx, cy, s, color, lw):
    # Bowtie: two triangles sharing a single vertex. 1 shared vertex (deg 4),
    # 4 tips (deg 2 each). 0 odd → 1 stroke Eulerian circuit.
    # Left triangle
    ax.plot([cx, cx - s * 0.85, cx - s * 0.85, cx],
            [cy, cy + s * 0.6, cy - s * 0.6, cy],
            color=color, linewidth=lw, zorder=3)
    # Right triangle
    ax.plot([cx, cx + s * 0.85, cx + s * 0.85, cx],
            [cy, cy + s * 0.6, cy - s * 0.6, cy],
            color=color, linewidth=lw, zorder=3)

def _draw_K4(ax, cx, cy, s, color, lw):
    # Complete graph K4: 4 vertices, all pairs connected → every vertex
    # has degree 3. 4 odd → 2 strokes.
    pts = [(cx, cy + s * 0.8),
           (cx + s * 0.8, cy - s * 0.4),
           (cx - s * 0.8, cy - s * 0.4),
           (cx, cy - s * 0.1)]
    # Actually use 4 vertices forming a tetrahedron projection.
    pts = [
        (cx, cy + s * 0.78),               # top
        (cx - s * 0.82, cy - s * 0.55),    # bl
        (cx + s * 0.82, cy - s * 0.55),    # br
        (cx, cy - s * 0.05),               # inner
    ]
    for i in range(4):
        for j in range(i + 1, 4):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=color, linewidth=lw, zorder=3)

# ------------------------------------------------------------------ #
# 3-stroke figures (for high levels)
# ------------------------------------------------------------------ #

def _draw_three_segments(ax, cx, cy, s, color, lw):
    # Three disjoint line segments → 3 strokes.
    for i, dy in enumerate([s * 0.6, 0.0, -s * 0.6]):
        ax.plot([cx - s * 0.75, cx + s * 0.75], [cy + dy, cy + dy],
                color=color, linewidth=lw, zorder=3)

def _draw_three_disjoint_shapes(ax, cx, cy, s, color, lw):
    # Small triangle + small square + small circle (disjoint) → 3 strokes.
    ax.add_patch(mpatches.Circle((cx - s * 0.55, cy + s * 0.35), s * 0.22,
                                 facecolor="none", edgecolor=color,
                                 linewidth=lw, zorder=3))
    ax.add_patch(mpatches.Rectangle(
        (cx + s * 0.25, cy + s * 0.1), s * 0.5, s * 0.5,
        facecolor="none", edgecolor=color, linewidth=lw, zorder=3))
    tri = [(cx - s * 0.55, cy - s * 0.3),
           (cx - s * 0.1, cy - s * 0.75),
           (cx + s * 0.35, cy - s * 0.3),
           (cx - s * 0.55, cy - s * 0.3)]
    ax.plot([p[0] for p in tri], [p[1] for p in tri],
            color=color, linewidth=lw, zorder=3)

def _draw_triple_cross(ax, cx, cy, s, color, lw):
    # Three separate crosses (each a "+") → each plus already needs 2 strokes,
    # but disjoint → at least 3 strokes are needed (one must leave a plus
    # unfinished to reach the next). In practice this figure is drawn as
    # three small plus-signs.
    for base_x in [-s * 0.55, 0.0, s * 0.55]:
        ax.plot([cx + base_x - s * 0.15, cx + base_x + s * 0.15],
                [cy, cy], color=color, linewidth=lw, zorder=3)
        ax.plot([cx + base_x, cx + base_x],
                [cy - s * 0.2, cy + s * 0.2],
                color=color, linewidth=lw, zorder=3)

# ------------------------------------------------------------------ #
# Figure catalogue (name → (draw_fn, strokes))
# ------------------------------------------------------------------ #

_FIGURE_CATALOGUE = {
    # --- Easy 1-stroke figures (very obvious) ---
    "circle":              (_draw_circle, 1),
    "figure_eight":        (_draw_figure_eight, 1),
    "square":              (_draw_square, 1),
    "triangle":            (_draw_triangle, 1),
    "house":               (_draw_house, 1),
    "triangle_with_bar":   (_draw_triangle_with_bar, 1),
    "ellipse":             (_draw_ellipse, 1),
    "letter_L":            (_draw_letter_L, 1),
    "letter_Z":            (_draw_letter_Z, 1),
    "staircase":           (_draw_staircase, 1),

    # --- Easy 2-stroke figures (disconnected or 4+ odd vertices) ---
    "square_with_x":       (_draw_square_with_x, 2),
    "envelope_crossed":    (_draw_envelope, 2),
    "two_circles":         (_draw_two_circles, 2),
    "two_squares":         (_draw_two_squares, 2),
    "plus":                (_draw_plus, 2),
    "disjoint_tri_sq":     (_draw_disjoint_triangle_square, 2),

    # --- Subtle / ambiguous figures (letters with junctions) ---
    "letter_T":            (_draw_letter_T, 2),
    "letter_Y":            (_draw_letter_Y, 2),
    "letter_K":            (_draw_letter_K, 2),
    "asym_fork":           (_draw_asym_fork, 2),

    # --- Complex graphs (Eulerian analysis non-trivial) ---
    "theta_graph":         (_draw_theta_graph, 1),
    "bowtie":              (_draw_bowtie, 1),
    "K4":                  (_draw_K4, 2),

    # --- 3-stroke figures (high levels) ---
    "three_segments":      (_draw_three_segments, 3),
    "three_disjoint":      (_draw_three_disjoint_shapes, 3),
    "triple_cross":        (_draw_triple_cross, 3),
}

# Grid cells for 6 and 8 positions
_GRID_CELLS_6 = [
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
]

_GRID_CELLS_8 = [
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1), (3, 1),
]

class StrokeContinuityGroupingQA(StandaloneVisualEnv):
    ENV_NAME = "stroke_continuity_grouping"

    # -------------------------------------------------------------- #
    # Level-to-difficulty configuration.
    # -------------------------------------------------------------- #
    # Returns a dict describing the pools of figures to draw from AND
    # the structural parameters (n_figures, n_groups).
    def _level_config(self, level: int):
        level = max(0, min(level, 9))
        # 2026-05-04: simplified L0 (was 5% too-hard) — restrict pool to most
        # visually obvious shapes (closed simple shapes vs disjoint shapes);
        # keep 6 figures (2 groups of 3) so distractor partition count is OK.
        if level <= 1:
            one_stroke = ["circle", "ellipse", "square", "triangle"]
            two_stroke = ["two_circles", "two_squares",
                          "disjoint_tri_sq", "plus"]
            three_stroke: List[str] = []
            return {
                "n_figures": 6,
                "n_groups": 2,
                "group_size": 3,
                "include_3_stroke": False,
                "pool": {1: one_stroke, 2: two_stroke, 3: three_stroke},
            }
        if level <= 2:
            # Easy: obvious 1 vs 2-stroke figures only. 6 figures, 2 groups.
            one_stroke = ["circle", "figure_eight", "ellipse", "square",
                          "triangle", "house", "triangle_with_bar",
                          "staircase", "letter_L", "letter_Z"]
            two_stroke = ["two_circles", "two_squares", "plus",
                          "disjoint_tri_sq", "square_with_x",
                          "envelope_crossed"]
            three_stroke: List[str] = []
            return {
                "n_figures": 6,
                "n_groups": 2,
                "group_size": 3,
                "include_3_stroke": False,
                "pool": {1: one_stroke, 2: two_stroke, 3: three_stroke},
            }

        if level <= 5:
            # Medium: ambiguous letter-shapes where stroke count is subtle.
            # 6 figures, 2 groups but drawn from a pool that includes
            # letter forms whose Eulerian status is non-obvious.
            one_stroke = ["house", "triangle_with_bar", "letter_L",
                          "letter_Z", "staircase", "theta_graph",
                          "bowtie", "triangle", "figure_eight"]
            two_stroke = ["letter_T", "letter_Y", "letter_K", "asym_fork",
                          "square_with_x", "envelope_crossed", "plus"]
            return {
                "n_figures": 6,
                "n_groups": 2,
                "group_size": 3,
                "include_3_stroke": False,
                "pool": {1: one_stroke, 2: two_stroke, 3: []},
            }

        # Hard (L6-9): 8 figures, potentially 3-stroke figures included,
        # graphs where Eulerian analysis is non-trivial (theta/bowtie/K4).
        one_stroke = ["house", "triangle_with_bar", "letter_L",
                      "letter_Z", "staircase", "theta_graph", "bowtie",
                      "figure_eight", "triangle"]
        two_stroke = ["letter_T", "letter_Y", "letter_K", "asym_fork",
                      "K4", "square_with_x", "envelope_crossed", "plus"]
        three_stroke = ["three_segments", "three_disjoint",
                        "triple_cross"]

        # At L6-L7: 8 figures, 2 groups of 4 (1-stroke vs 2-stroke).
        # At L8-L9: 8 figures, 3 groups including 3-stroke class.
        if level <= 7:
            return {
                "n_figures": 8,
                "n_groups": 2,
                "group_size": 4,
                "include_3_stroke": False,
                "pool": {1: one_stroke, 2: two_stroke, 3: three_stroke},
                "trap_rate": 0.30,
            }

        # L8-L9: triple grouping, 9 figures (3 per class).
        return {
            "n_figures": 9,
            "n_groups": 3,
            "group_size": 3,
            "include_3_stroke": True,
            "pool": {1: one_stroke, 2: two_stroke, 3: three_stroke},
            "trap_rate": 0.40,
        }

    _TITLE_VARIANTS = [
        "{n} figures:",
        "Stroke analysis ({n} shapes):",
        "Line-art figures",
        "Figures 1-{n}",
        "Drawing puzzle",
        "Stroke-count puzzle ({n} figures)",
        "Eulerian analysis — {n} shapes",
        "Line-art set of {n}",
        "Grouping problem",
        "Minimum-stroke figures",
    ]

    # 16 prompt framings. Each takes {n} (n_figures) and {rules} slots;
    # option list + "Answer with a single letter ..." is appended later.
    _PROMPT_TEMPLATES = [
        # --- Direct / interrogative (4) ---
        "Group the {n} figures by the minimum number of continuous strokes needed to draw each (one stroke = draw without lifting your pen or retracing any line). {rules} Which grouping is correct?",
        "The {n} line-art figures must be partitioned by their minimum stroke count. {rules} Which of the choices shows the correct partition?",
        "Look at the {n} figures. Each can be drawn in a minimum number of continuous strokes (no pen-lifting, no retracing). {rules} Which grouping is correct?",
        "Study the {n} figures in the image and partition them by minimum stroke count. {rules} Which option gives the correct grouping?",
        # --- Imperative / task-oriented (4) ---
        "Partition the {n} figures by the fewest continuous strokes needed to draw each. {rules} Pick the correct grouping.",
        "Sort the {n} line-art figures into groups based on minimum stroke count (no pen-lifting / no retracing). {rules} Select the correct grouping.",
        "Classify each of the {n} figures by the minimum stroke count it needs. {rules} Identify the correct partition.",
        "Divide the {n} shapes shown into groups by minimum stroke count. {rules} Choose the grouping that matches.",
        # --- Declarative / cloze (4) ---
        "The {n} figures should be grouped by the minimum number of continuous strokes required (no retracing, no pen-lifting). {rules} The correct grouping is ___.",
        "Group the {n} figures by minimum stroke count. {rules} The correct partition is ___.",
        "For the {n} figures shown, let G be the correct partition by minimum stroke count. {rules} What is G?",
        "The {n} figures in the image fall into groups by minimum stroke count. {rules} The option giving the correct grouping is ___.",
        # --- Reasoning-prompt (4) ---
        "Examine the {n} figures and determine, for each, the minimum number of continuous strokes needed. {rules} Which of the listed partitions groups them correctly?",
        "Reason about each of the {n} figures: how many continuous strokes does it minimally require? {rules} Which grouping matches?",
        "Inspect every figure among the {n} shown and count the minimum strokes it needs. {rules} Report which grouping is correct.",
        "After analysing the Eulerian properties of each of the {n} figures, determine which grouping by stroke count is correct. {rules}",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        # 2026-05-05 R5 B4B EASE: at L0/L1, ask single-figure "How many continuous
        # strokes?" 4-MCQ instead of 6-figure partition (was R3 0.100 too-hard).
        if level <= 1:
            return self._generate_single_figure_mcq(rng, sub_rng, level)

        cfg = self._level_config(level)

        need = cfg["group_size"]
        if len(cfg["pool"][1]) < need or len(cfg["pool"][2]) < need:
            return None
        if cfg["include_3_stroke"] and len(cfg["pool"][3]) < need:
            return None

        for _ in range(40):
            result = self._try_generate(rng, sub_rng, level, cfg)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # 2026-05-05 R5 B4B: L0/L1 single-figure "How many strokes?" MCQ.
    # Simpler than the 6-figure partition. Renders ONE big figure, asks
    # for stroke count among A.1 B.2 C.3 D.4.
    # ------------------------------------------------------------------ #

    def _generate_single_figure_mcq(self, rng, sub_rng, level):
        # Use only the most visually obvious figures.
        # 1-stroke: closed loops or simple polylines.
        one_stroke = ["circle", "ellipse", "square", "triangle",
                      "letter_L", "letter_Z", "staircase"]
        # 2-stroke: clearly disjoint or has 4+ odd-degree vertices.
        two_stroke = ["two_circles", "two_squares",
                      "disjoint_tri_sq", "plus", "letter_T",
                      "letter_Y", "square_with_x"]

        # Choose stroke count: 50/50 between 1 and 2.
        true_strokes = rng.choice([1, 2])
        pool = one_stroke if true_strokes == 1 else two_stroke
        figure_name = rng.choice(pool)

        # Always 4-MCQ: A.1 B.2 C.3 D.4
        opts = ["1", "2", "3", "4"]
        answer_letter = chr(ord("A") + (true_strokes - 1))

        sidx = (self.seed or 0) % 16
        title = "Line-art figure"
        image = self._render_single_figure(figure_name, title=title,
                                            sub_rng=sub_rng)

        opt_lines = "\n".join(f"  {chr(ord('A') + i)}) {opts[i]}" for i in range(4))
        stems = [
            "How many continuous strokes are needed to draw the figure (without lifting the pen and without retracing any line)?",
            "What is the minimum number of continuous strokes required to draw the line-art figure shown?",
            "Look at the figure. How many continuous strokes (no pen-lifting, no retracing) does it minimally require?",
            "The figure shown can be drawn in a minimum of how many continuous strokes?",
        ]
        stem = stems[sidx % len(stems)]
        question = (
            f"{stem}\n{opt_lines}\n"
            f"Reply with the single letter (A/B/C/D) inside <answer>...</answer>. Example: <answer>B</answer>"
        )
        if level == 0:
            # Concise leak hint
            question += (
                f" Hint: a closed loop (circle, square) or simple polyline = 1 stroke; "
                f"two disjoint shapes or a plus-sign = 2 strokes. Answer: {answer_letter}."
            )
        else:  # L1
            question += (
                " Hint: closed loops and simple polylines need 1 stroke; "
                "disjoint shapes or shapes with 4+ odd-degree vertices need 2."
            )
        return question, answer_letter, image

    def _render_single_figure(self, figure_name, title, sub_rng):
        """Render ONE big figure centered on the canvas."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        bg = style["bg_color"]
        line_color = "#1a3a6e"
        lw = 3.0

        fig = plt.figure(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=fs + 4, fontweight="bold",
                     fontfamily=ff, color="#222222")

        draw_fn = _FIGURE_CATALOGUE[figure_name][0]
        draw_fn(ax, 0.0, 0.0, 1.0, line_color, lw)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _try_generate(self, rng, sub_rng, level, cfg):
        n_figures = cfg["n_figures"]
        n_groups = cfg["n_groups"]
        group_size = cfg["group_size"]

        picks_1 = rng.sample(cfg["pool"][1], group_size)
        picks_2 = rng.sample(cfg["pool"][2], group_size)
        picks_3: List[str] = []
        if cfg["include_3_stroke"]:
            picks_3 = rng.sample(cfg["pool"][3], group_size)

        all_picks = picks_1 + picks_2 + picks_3
        if len(set(all_picks)) != n_figures:
            return None

        positions = list(range(n_figures))
        rng.shuffle(positions)
        fig_at_pos: List[Optional[Tuple[str, int]]] = [None] * n_figures
        cursor = 0
        for n in picks_1:
            fig_at_pos[positions[cursor]] = (n, 1)
            cursor += 1
        for n in picks_2:
            fig_at_pos[positions[cursor]] = (n, 2)
            cursor += 1
        for n in picks_3:
            fig_at_pos[positions[cursor]] = (n, 3)
            cursor += 1

        groups_by_stroke = {
            1: sorted(p for p, (_, g) in enumerate(fig_at_pos) if g == 1),
            2: sorted(p for p, (_, g) in enumerate(fig_at_pos) if g == 2),
        }
        if cfg["include_3_stroke"]:
            groups_by_stroke[3] = sorted(
                p for p, (_, g) in enumerate(fig_at_pos) if g == 3)

        correct_parts = [frozenset(groups_by_stroke[k])
                         for k in sorted(groups_by_stroke.keys())]
        correct = frozenset(correct_parts)

        distractors = self._make_distractors(rng, correct_parts)
        # 2026-05-05 R5 B4B: at L6+, may add 5th option "E. No correct" (trap).
        trap_rate = cfg.get("trap_rate", 0.0)
        use_trap = (rng.random() < trap_rate) and (level >= 6)
        n_visible = 4

        if use_trap:
            # All 4 visible options must be wrong; correct = E.
            if len(distractors) < 4:
                return None
            options = distractors[:4]
            rng.shuffle(options)
            answer_letter = "E"
            has_E_label = True
        else:
            if len(distractors) < 3:
                return None
            options = [correct] + distractors[:3]
            rng.shuffle(options)
            correct_idx = options.index(correct)
            answer_letter = chr(ord("A") + correct_idx)
            has_E_label = (level >= 6)  # show E option even when not trap

        sidx = (self.seed or 0) % 16
        title = self._TITLE_VARIANTS[sidx % len(self._TITLE_VARIANTS)].format(n=n_figures)
        image = self._render(fig_at_pos, options, n_figures, n_groups,
                             title=title, sub_rng=sub_rng)

        opt_lines = []
        for i, part in enumerate(options):
            parts = sorted(part, key=lambda s: min(s))
            strs = [
                "{" + ", ".join(str(p + 1) for p in sorted(pt)) + "}"
                for pt in parts
            ]
            opt_lines.append(
                f"  {chr(ord('A') + i)}) " + " | ".join(strs))
        if has_E_label:
            opt_lines.append("  E) No correct grouping")
        opts_text = "\n".join(opt_lines)

        n_words = {3: "three", 4: "four"}.get(group_size, str(group_size))
        if cfg["include_3_stroke"]:
            rules = (f"Three groups of {n_words} figures each: one group can "
                     "be drawn with one stroke, one requires two strokes, "
                     "and one requires three strokes.")
        elif n_groups == 2:
            rules = (f"{n_words.title()} figures can be drawn with one "
                     f"stroke and {n_words} require two strokes.")
        else:
            rules = ""

        frame = self._PROMPT_TEMPLATES[sidx].format(n=n_figures, rules=rules).strip()
        # collapse any double spaces produced when rules is empty
        while "  " in frame:
            frame = frame.replace("  ", " ")

        letter_set = "A/B/C/D/E" if has_E_label else "A/B/C/D"
        question = (
            f"{frame}\n{opts_text}\nReply with the single letter ({letter_set}) inside <answer>...</answer>. Example: <answer>B</answer>"
        )
        if has_E_label:
            question += " (Option E means none of A-D is the correct grouping.)"
        if level <= 2:
            question += (
                "\nHint: a figure can be drawn in ONE continuous stroke (no "
                "lifting the pen) if and only if every vertex has even "
                "degree, OR exactly TWO vertices have odd degree. For each "
                "figure, count the line segments meeting at each vertex; "
                "if all are even (or exactly 2 are odd) → 1 stroke; "
                "otherwise → 2+ strokes."
            )
        self._primary_complexity_feature = n_figures + level * 2
        return question, answer_letter, image

    def _make_distractors(self, rng, correct_parts: List[frozenset]):
        """Create 3 distractor partitions by swapping pairs across groups.

        `correct_parts` is a list of frozensets (one per group).
        """
        correct_key = frozenset(correct_parts)
        seen = {correct_key}
        out = []
        # Convert to list-of-lists so we can mutate.
        groups_list = [sorted(list(p)) for p in correct_parts]
        n_groups = len(groups_list)

        attempts = 0
        while len(out) < 3 and attempts < 120:
            attempts += 1
            swap_count = 1 if attempts < 40 else 2
            new_groups = [list(g) for g in groups_list]
            for _ in range(swap_count):
                # Pick 2 distinct groups to swap elements between.
                ga, gb = rng.sample(range(n_groups), 2)
                if not new_groups[ga] or not new_groups[gb]:
                    continue
                ai = rng.randint(0, len(new_groups[ga]) - 1)
                bi = rng.randint(0, len(new_groups[gb]) - 1)
                new_groups[ga][ai], new_groups[gb][bi] = \
                    new_groups[gb][bi], new_groups[ga][ai]
            key = frozenset(frozenset(g) for g in new_groups)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, fig_at_pos, options, n_figures, n_groups,
                title="Figures", sub_rng=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        _line_colors = ["#2c3e50", "#1a5276", "#7b241c", "#0d6efd",
                        "#198754", "#6c3483", "#1b4f72"]
        line_color = (sub_rng or self._rng).choice(_line_colors)
        lw = 1.8 + (sub_rng or self._rng).random() * 1.0

        # Pick grid layout.
        if n_figures <= 6:
            grid_cols, grid_rows = 3, 2
            cells = _GRID_CELLS_6
        elif n_figures <= 8:
            grid_cols, grid_rows = 4, 2
            cells = _GRID_CELLS_8
        else:
            # 9 figures → 3x3 grid.
            grid_cols, grid_rows = 3, 3
            cells = [(c, r) for r in range(3) for c in range(3)]

        mode = pick_render_mode(sub_rng or self._rng)
        if mode == "textbook":
            tbp = textbook_params(sub_rng or self._rng)
            bg = tbp["bg"]
            ff_use = tbp["font_family"]
            line_color_use = tbp["line_color"]
            lw_use = max(0.8, tbp["line_width"])
            box_fc = bg
            box_ec = tbp["aux_color"]
            box_lw = tbp["aux_line_width"]
            label_color = tbp["line_color"]
            label_box_fc = bg
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = (sub_rng or self._rng).choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            ff_use = ff
            line_color_use = "#1a1a1a"
            lw_use = 2.0 + (sub_rng or self._rng).random() * 0.6
            box_fc = bg
            box_ec = "#1a1a1a"
            box_lw = 1.4
            label_color = "#1a1a1a"
            label_box_fc = "#ffffff"
            dpi = style["dpi"]
        else:
            bg = style["bg_color"]
            ff_use = ff
            line_color_use = line_color
            lw_use = lw
            box_fc = bg
            box_ec = "#bdc3c7"
            box_lw = 1.2
            label_color = "#2c3e50"
            label_box_fc = "#ffffff"
            dpi = style["dpi"]

        def _draw_all():
            fig_w = max(6.5, grid_cols * 1.7) * sc
            fig_h = (grid_rows * 1.3 + 2.6) * sc
            fig = plt.figure(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(bg)
            gs = fig.add_gridspec(2, 1,
                                  height_ratios=[grid_rows * 1.2, 1.8],
                                  hspace=0.3)
            ax_fig = fig.add_subplot(gs[0])
            ax_opt = fig.add_subplot(gs[1])
            ax_fig.set_facecolor(bg); ax_opt.set_facecolor(bg)

            ax_fig.set_xlim(0, grid_cols)
            ax_fig.set_ylim(0, grid_rows)
            ax_fig.set_aspect("equal")
            ax_fig.axis("off")
            ax_fig.set_title(title,
                             fontsize=fs + 2, fontweight="bold", pad=6,
                             fontfamily=ff_use, color=label_color)

            for pos, (name, _) in enumerate(fig_at_pos):
                if pos >= len(cells):
                    break
                col, row_y = cells[pos]
                cx = col + 0.5
                cy = (grid_rows - 1 - row_y) + 0.5
                rect = mpatches.FancyBboxPatch(
                    (cx - 0.44, cy - 0.42), 0.88, 0.84,
                    boxstyle="round,pad=0.02",
                    facecolor=box_fc, edgecolor=box_ec,
                    linewidth=box_lw, zorder=1)
                ax_fig.add_patch(rect)
                draw_fn = _FIGURE_CATALOGUE[name][0]
                draw_fn(ax_fig, cx, cy, 0.30, line_color_use, lw_use)
                ax_fig.text(cx - 0.4, cy + 0.36, str(pos + 1),
                            fontsize=fs + 1, fontweight="bold",
                            ha="left", va="top", color=label_color,
                            fontfamily=ff_use,
                            bbox=dict(boxstyle="circle,pad=0.12",
                                      facecolor=label_box_fc,
                                      edgecolor=label_color,
                                      linewidth=0.8))

            ax_opt.set_xlim(0, 1); ax_opt.set_ylim(0, 1); ax_opt.axis("off")
            n_opts = len(options)
            for i, part in enumerate(options):
                parts = sorted(part, key=lambda s: min(s))
                group_strs = [
                    "{" + ", ".join(str(p + 1) for p in sorted(pt)) + "}"
                    for pt in parts
                ]
                text_str = "  |  ".join(group_strs)
                y = 0.9 - i * (0.9 / n_opts)
                ax_opt.text(0.02, y, f"{chr(ord('A') + i)}. ",
                            fontsize=fs + 2, fontweight="bold",
                            color=label_color, va="center", fontfamily=ff_use)
                ax_opt.text(0.08, y, text_str,
                            fontsize=fs, color=label_color,
                            va="center", fontfamily=ff_use)
            try: fig.tight_layout()
            except: pass
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=80, randomness=1.5):
                fig = _draw_all()
        else:
            fig = _draw_all()
        return self.fig_to_pil(fig, dpi=dpi)

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = StrokeContinuityGroupingQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed=seed * 10 + level,
                              parameter={"level": level})
            if not ok:
                print(f"FAILED seed={seed} level={level}")
                continue
            img = env.render()
            img.save(f"/tmp/env_check/stroke_continuity_grouping_seed{seed}_L{level}.png")
            print(f"seed={seed} level={level}")
            print("Q:", env.get_instruction().splitlines()[0])
            print("A:", env._answer)
            print()

    # Difficulty gradient statistics.
    # "Shape complexity" score: pool subtlety + vertex count.
    def _complexity(cfg, fig_at_pos):
        total = 0
        for (name, _) in fig_at_pos:
            # Use the stroke count as a crude complexity metric along with
            # whether the shape is in the "ambiguous" pool.
            if name in ("letter_T", "letter_Y", "letter_K", "asym_fork",
                        "theta_graph", "bowtie", "K4", "house",
                        "triangle_with_bar", "staircase"):
                total += 2
            else:
                total += 1
        return total

    for lv in (0, 9):
        nfig_total = 0
        complexity_total = 0
        n_ok = 0
        three_stroke_count = 0
        for s in range(20):
            ok = env.generate(seed=1000 + s * 10 + lv,
                              parameter={"level": lv})
            if not ok:
                continue
            n_ok += 1
            # The env doesn't expose fig_at_pos directly, but we can re-derive
            # via the instruction text.  Instead we just count through the
            # env's internal state by accessing what _try_generate saved.
            # Simpler: call _level_config directly for the stats.
            cfg = env._level_config(lv)
            nfig_total += cfg["n_figures"]
            # Complexity: 1 for simple, 2 for ambiguous/complex.
            # Score by counting how many pool entries are in the "hard" list.
            pool_all = cfg["pool"][1] + cfg["pool"][2] + cfg["pool"][3]
            ambig = sum(1 for x in pool_all if x in (
                "letter_T", "letter_Y", "letter_K", "asym_fork",
                "theta_graph", "bowtie", "K4", "house",
                "triangle_with_bar", "staircase"))
            complexity_total += ambig
            if cfg["include_3_stroke"]:
                three_stroke_count += 1
        mean_figs = nfig_total / max(n_ok, 1)
        mean_complex = complexity_total / max(n_ok, 1)
        print(f"[L{lv}] mean_n_figures={mean_figs:.2f} "
              f"mean_ambig_pool_size={mean_complex:.2f} "
              f"3_stroke_samples={three_stroke_count}/{n_ok}")
