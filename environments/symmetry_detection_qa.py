"""
Symmetry Detection QA environment.

Draws a shape on a coordinate plane and asks about lines of symmetry.
Question types: count lines of symmetry, is it symmetric about x/y-axis,
identify axes of symmetry.
Shapes: regular polygons, irregular shapes, letters, composite shapes.
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
# Shape definitions with known symmetry properties
# ------------------------------------------------------------------ #

def _regular_polygon(n, radius=3.0, cx=0, cy=0):
    """Regular n-gon centred at (cx, cy). n lines of symmetry."""
    verts = []
    for i in range(n):
        a = math.pi / 2 + 2 * math.pi * i / n
        verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return verts

def _isoceles_triangle(base=4, height=5, cx=0, cy=0):
    """Isoceles triangle. 1 line of symmetry (vertical through apex)."""
    return [
        (cx, cy + height * 2 / 3),
        (cx - base / 2, cy - height / 3),
        (cx + base / 2, cy - height / 3),
    ]

def _scalene_triangle(cx=0, cy=0):
    """Scalene triangle. 0 lines of symmetry."""
    return [
        (cx - 1, cy - 2),
        (cx + 3, cy - 1),
        (cx + 0.5, cy + 2.5),
    ]

def _rectangle(w=5, h=3, cx=0, cy=0):
    """Rectangle (non-square). 2 lines of symmetry."""
    return [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]

def _parallelogram(cx=0, cy=0):
    """Parallelogram (non-rectangle). 0 lines of symmetry."""
    return [
        (cx - 2, cy - 1.5),
        (cx + 2, cy - 1.5),
        (cx + 3, cy + 1.5),
        (cx - 1, cy + 1.5),
    ]

def _kite(cx=0, cy=0):
    """Kite shape. 1 line of symmetry (vertical)."""
    return [
        (cx, cy + 3),
        (cx + 2, cy + 0.5),
        (cx, cy - 3),
        (cx - 2, cy + 0.5),
    ]

def _arrow_shape(cx=0, cy=0):
    """Arrow pointing right. 1 line of symmetry (horizontal)."""
    return [
        (cx - 3, cy + 1),
        (cx + 1, cy + 1),
        (cx + 1, cy + 2),
        (cx + 3, cy),
        (cx + 1, cy - 2),
        (cx + 1, cy - 1),
        (cx - 3, cy - 1),
    ]

def _plus_shape(cx=0, cy=0, arm=1, length=3):
    """Plus / cross shape. 4 lines of symmetry."""
    return [
        (cx - arm, cy + length),
        (cx + arm, cy + length),
        (cx + arm, cy + arm),
        (cx + length, cy + arm),
        (cx + length, cy - arm),
        (cx + arm, cy - arm),
        (cx + arm, cy - length),
        (cx - arm, cy - length),
        (cx - arm, cy - arm),
        (cx - length, cy - arm),
        (cx - length, cy + arm),
        (cx - arm, cy + arm),
    ]

def _t_shape(cx=0, cy=0):
    """T-shape. 1 line of symmetry (vertical)."""
    return [
        (cx - 3, cy + 2),
        (cx + 3, cy + 2),
        (cx + 3, cy + 1),
        (cx + 1, cy + 1),
        (cx + 1, cy - 2),
        (cx - 1, cy - 2),
        (cx - 1, cy + 1),
        (cx - 3, cy + 1),
    ]

def _l_shape(cx=0, cy=0):
    """L-shape. 0 lines of symmetry."""
    return [
        (cx - 2, cy + 3),
        (cx, cy + 3),
        (cx, cy - 1),
        (cx + 2, cy - 1),
        (cx + 2, cy - 3),
        (cx - 2, cy - 3),
    ]

def _rhombus(cx=0, cy=0, w=3, h=4):
    """Rhombus. 2 lines of symmetry."""
    return [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]

def _ellipse_like(cx=0, cy=0):
    """Approximate ellipse with polygon. 2 lines of symmetry."""
    verts = []
    a, b = 3.5, 2.0
    for i in range(24):
        angle = 2 * math.pi * i / 24
        verts.append((cx + a * math.cos(angle), cy + b * math.sin(angle)))
    return verts

# Complete catalog: (generator_fn, num_symmetry_lines, has_x_symmetry, has_y_symmetry, name, rotational_order, has_point_symmetry)
# rotational_order: smallest rotation angle dividing 360 that maps shape to itself. Order = 360/angle.
# has_point_symmetry: True if shape maps to itself under 180-degree rotation.
_SHAPE_CATALOG = [
    (lambda r: _regular_polygon(3, 3), 3, False, True, "equilateral triangle", 3, False),
    (lambda r: _regular_polygon(4, 3), 4, True, True, "square", 4, True),
    (lambda r: _regular_polygon(5, 3), 5, False, True, "regular pentagon", 5, False),
    (lambda r: _regular_polygon(6, 3), 6, True, True, "regular hexagon", 6, True),
    (lambda r: _regular_polygon(8, 3), 8, True, True, "regular octagon", 8, True),
    (lambda r: _isoceles_triangle(), 1, False, True, "isosceles triangle", 1, False),
    (lambda r: _scalene_triangle(), 0, False, False, "scalene triangle", 1, False),
    (lambda r: _rectangle(5, 3), 2, True, True, "rectangle", 2, True),
    (lambda r: _parallelogram(), 0, False, False, "parallelogram", 2, True),
    (lambda r: _kite(), 1, False, True, "kite", 1, False),
    (lambda r: _arrow_shape(), 1, True, False, "arrow", 1, False),
    (lambda r: _plus_shape(), 4, True, True, "plus sign", 4, True),
    (lambda r: _t_shape(), 1, False, True, "T-shape", 1, False),
    (lambda r: _l_shape(), 0, False, False, "L-shape", 1, False),
    (lambda r: _rhombus(), 2, True, True, "rhombus", 2, True),
    (lambda r: _ellipse_like(), 2, True, True, "ellipse", 2, True),
]

_TITLE_VARIANTS = [
    "Shape in the coordinate plane",
    "Examine this shape",
    "Analyze the symmetry",
    "Shape for analysis",
]

class SymmetryDetectionQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "symmetry_detection"

    QUESTION_TYPES = [
        "count_lines",
        "is_x_symmetric",
        "is_y_symmetric",
        "has_any_symmetry",
        "count_lines_mc",
        # Harder types
        "rotational_symmetry_order",
        "point_symmetry",
    ]

    # Split catalogs by "easy" / "medium" / "hard"
    # Easy: shapes with 0-2 sym lines, simple (triangle, rect, kite)
    # Hard: high-symmetry regular polygons + plus shape

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _easy_pool(self):
        # Subset with n_sym in 0..2 (simpler to count)
        return [e for e in _SHAPE_CATALOG if e[1] <= 2]

    def _hard_pool(self):
        # Hard pool for L8-L9: mix high-symmetry and low-symmetry shapes so
        # the answer to point_symmetry / rotational_order is not determined
        # solely by "is it a regular polygon?". Include parallelogram (order 2,
        # has point symmetry but no reflection symmetry) and scalene/kite/L-shape
        # (no point symmetry) to force careful analysis.
        hard_extras = [e for e in _SHAPE_CATALOG
                       if e[4] in ("parallelogram", "scalene triangle",
                                   "kite", "L-shape", "arrow",
                                   "isosceles triangle", "T-shape")]
        high = [e for e in _SHAPE_CATALOG if e[1] >= 3]
        return high + hard_extras

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # Redesign 2026-04-17: remove MCQ variants from mid-levels (they
        # raised L3 to 1.0 artificially). Keep MCQ only at L8-L9 with hard
        # pool. Push rotational_symmetry_order and point_symmetry to high
        # levels where they're genuinely harder.
        if level <= 1:
            return {"qtypes": ["is_x_symmetric", "is_y_symmetric",
                               "has_any_symmetry"],
                    "shape_pool": self._easy_pool()}
        if level <= 3:
            return {"qtypes": ["is_x_symmetric", "is_y_symmetric",
                               "has_any_symmetry", "count_lines"],
                    "shape_pool": _SHAPE_CATALOG}
        if level <= 5:
            return {"qtypes": ["count_lines", "is_x_symmetric", "is_y_symmetric"],
                    "shape_pool": _SHAPE_CATALOG}
        if level <= 7:
            return {"qtypes": ["count_lines", "rotational_symmetry_order"],
                    "shape_pool": _SHAPE_CATALOG}
        return {"qtypes": ["rotational_symmetry_order", "point_symmetry"],
                "shape_pool": self._hard_pool()}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Sub-RNG includes level so L0 ≠ L9 (previous bug: _rng was used
        # for shape selection and both levels got the same choice).
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(25):
            result = self._try_generate(sub_rng, qtype, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, qtype, cfg):
        pool = cfg.get("shape_pool", _SHAPE_CATALOG)
        entry = rng.choice(pool)
        gen_fn, n_sym, x_sym, y_sym, name = entry[0], entry[1], entry[2], entry[3], entry[4]
        rot_order = entry[5] if len(entry) > 5 else 1
        point_sym = entry[6] if len(entry) > 6 else False
        verts = gen_fn(rng)

        # NOTE: Do NOT include shape name in question — forces model to analyze the image
        if qtype == "count_lines":
            question = (
                "A shape is shown on the coordinate plane. "
                "How many lines of symmetry does it have? "
                "Answer with a single number."
            )
            answer = str(n_sym)

        elif qtype == "count_lines_mc":
            options = sorted(set([n_sym] + [rng.randint(0, 8) for _ in range(3)]))
            if n_sym not in options:
                options[rng.randint(0, len(options) - 1)] = n_sym
                options = sorted(set(options))
            while len(options) < 4:
                candidate = rng.randint(0, 8)
                if candidate not in options:
                    options.append(candidate)
                    options.sort()
            options = options[:4]
            correct_letter = chr(ord("A") + options.index(n_sym))
            opt_str = ", ".join(f"{chr(ord('A') + i)}) {v}" for i, v in enumerate(options))
            question = (
                "A shape is drawn on the coordinate plane. "
                "How many lines of symmetry does it have? "
                f"{opt_str}. Answer with a single letter."
            )
            answer = correct_letter

        elif qtype == "is_x_symmetric":
            question = (
                "A shape is shown on the coordinate plane centered at the origin. "
                "Is it symmetric about the x-axis (horizontal axis)? "
                "Answer Yes or No."
            )
            answer = "Yes" if x_sym else "No"

        elif qtype == "is_y_symmetric":
            question = (
                "A shape is shown on the coordinate plane centered at the origin. "
                "Is it symmetric about the y-axis (vertical axis)? "
                "Answer Yes or No."
            )
            answer = "Yes" if y_sym else "No"

        elif qtype == "has_any_symmetry":
            question = (
                "A shape is shown on the coordinate plane. "
                "Does this shape have any line of symmetry? "
                "Answer Yes or No."
            )
            answer = "Yes" if n_sym > 0 else "No"

        elif qtype == "rotational_symmetry_order":
            question = (
                "A shape is shown on the coordinate plane. "
                "What is its order of rotational symmetry? "
                "(The number of times the shape maps onto itself during a full 360-degree rotation. "
                "A shape with no rotational symmetry has order 1.) "
                "Answer with a single number."
            )
            answer = str(rot_order)

        elif qtype == "point_symmetry":
            question = (
                "A shape is shown on the coordinate plane centered at the origin. "
                "Does it have point symmetry? A shape has point symmetry if "
                "every point P on the shape has a corresponding point P' also "
                "on the shape such that the center is the midpoint of PP'. "
                "Answer Yes or No."
            )
            answer = "Yes" if point_sym else "No"

        else:
            return None

        image = self._render_shape(verts, name, n_sym, x_sym, y_sym, rng)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_shape(self, verts, name, n_sym, x_sym, y_sym, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        # Rotate the shape by a small random angle so image diversity
        # grows cross-seed. Rotation does not change n_sym, x_sym, y_sym
        # *only if* the shape is rotationally symmetric; for shapes that
        # aren't, rotation would change x/y-axis symmetry answers.
        # For safety, we only rotate shapes that have rotational order >= 2.
        import math as _m
        rot_ok = False  # conservative: keep axis-aligned so x_sym/y_sym stay correct

        # Coordinate plane
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2
        x_lo, x_hi = min(xs) - margin, max(xs) + margin
        y_lo, y_hi = min(ys) - margin, max(ys) + margin
        x_lo, y_lo = min(x_lo, -1), min(y_lo, -1)
        x_hi, y_hi = max(x_hi, 1), max(y_hi, 1)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.axhline(0, color="#2c3e50", linewidth=1.2, zorder=1)
        ax.axvline(0, color="#2c3e50", linewidth=1.2, zorder=1)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

        palette = style["palette"]
        lw = style["line_width"]
        # Randomize fill color per seed
        face_col = palette[rng.randrange(len(palette))]
        edge_col = style["geo_line_color"]
        poly = plt.Polygon(verts, closed=True, facecolor=face_col,
                           alpha=0.35, edgecolor=edge_col, linewidth=lw,
                           zorder=3)
        ax.add_patch(poly)

        # Vertices (small markers, no letter labels so vertex count doesn't leak)
        for (x, y) in verts:
            ax.plot(x, y, "o", color="#2c3e50", markersize=4, zorder=4)

        # DO NOT draw symmetry axes as dashed lines — this was a massive
        # answer leak (model could count the orange dashed lines).

        # Title: generic only, does NOT expose shape name
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 4,
                     fontweight="bold", pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
