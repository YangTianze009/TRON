"""
Figure Counting QA environment.

Draws overlapping / nested geometric figures and asks counting questions.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2-4 well-separated simple geometric figures (circle/square/triangle).
    `count_simple_shapes` — "how many shapes?".
L1: 3-5 well-separated simple shapes, possibly 2 types.
L2: 4-6 shapes, 2-3 types, still separated.
L3: subdivided triangle n=2, or 2x2 grid rectangles.
L4: subdivided triangle n=3, or 2x2 grid squares / 2x3 rectangles.
L5: subdivided triangle n=4, 3x3 grid.
L6: 3-4 intersecting lines regions, or 3x3 squares.
L7: subdivided n=5, 4x4 grids, pentagram.
L8: subdivided n=6, 5-6 intersecting lines, hexagram.
L9: max subdivisions / full intersecting lines.

parameter = {"level": int in [0,9]}
"""
import math
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = [
    "Figure Count",
    "Shapes",
    "Counting Figures",
    "Figure Scene",
    "Shape Grid",
]

def _count_triangles_in_subdivided(n):
    known = {1: 1, 2: 5, 3: 13, 4: 27, 5: 48, 6: 78, 7: 118, 8: 170}
    return known.get(n, 0)

def _count_rectangles_in_grid(rows, cols):
    return (rows * (rows + 1) // 2) * (cols * (cols + 1) // 2)

def _count_squares_in_grid(rows, cols):
    total = 0
    for k in range(1, min(rows, cols) + 1):
        total += (rows - k + 1) * (cols - k + 1)
    return total

def _line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

class FigureCountingQA(StandaloneVisualEnv):
    ENV_NAME = "figure_counting"

    QUESTION_TYPES = [
        "count_simple_shapes",    # trivial — separate shapes
        "count_triangles",        # subdivided triangle
        "count_rectangles",       # grid rectangles
        "count_squares",          # grid squares
        "count_regions",          # intersecting lines
        "count_triangles_star",   # pentagram / hexagram
        "count_parallelograms",   # slanted parallelogram grid (merged from config_counting)
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"], weights=cfg["qtype_weights"])[0]
        for _ in range(20):
            result = self._dispatch(qtype, level, cfg, parameter)
            if result is not None:
                self._primary_complexity_feature = level * 5 + len(result[1])
                return result
        return None

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            return {"qtypes": ["count_simple_shapes"],
                    "qtype_weights": [1],
                    "simple_min": 2, "simple_max": 4,
                    "simple_n_types": 1}
        if level == 1:
            return {"qtypes": ["count_simple_shapes"],
                    "qtype_weights": [1],
                    "simple_min": 3, "simple_max": 5,
                    "simple_n_types": 2}
        if level == 2:
            return {"qtypes": ["count_simple_shapes"],
                    "qtype_weights": [1],
                    "simple_min": 4, "simple_max": 6,
                    "simple_n_types": 3}
        if level == 3:
            return {"qtypes": ["count_triangles", "count_rectangles"],
                    "qtype_weights": [5, 5],
                    "subdivide": 2, "grid_range": (2, 2)}
        if level == 4:
            return {"qtypes": ["count_triangles", "count_rectangles",
                               "count_squares"],
                    "qtype_weights": [4, 3, 3],
                    "subdivide": 3, "grid_range": (2, 3)}
        if level == 5:
            return {"qtypes": ["count_triangles", "count_rectangles",
                               "count_squares"],
                    "qtype_weights": [4, 3, 3],
                    "subdivide": 4, "grid_range": (3, 3)}
        if level == 6:
            return {"qtypes": ["count_triangles", "count_rectangles",
                               "count_squares", "count_regions",
                               "count_parallelograms"],
                    "qtype_weights": [3, 3, 2, 2, 2],
                    "subdivide": 4, "grid_range": (3, 4),
                    "n_lines_range": (3, 4), "para_range": (2, 3)}
        if level == 7:
            return {"qtypes": ["count_triangles", "count_rectangles",
                               "count_squares", "count_regions",
                               "count_triangles_star", "count_parallelograms"],
                    "qtype_weights": [3, 2, 2, 2, 1, 2],
                    "subdivide": 5, "grid_range": (4, 4),
                    "n_lines_range": (4, 5), "star_points": 5,
                    "para_range": (3, 4)}
        if level == 8:
            return {"qtypes": ["count_triangles", "count_rectangles",
                               "count_squares", "count_regions",
                               "count_triangles_star", "count_parallelograms"],
                    "qtype_weights": [3, 2, 2, 2, 1, 2],
                    "subdivide": 6, "grid_range": (4, 5),
                    "n_lines_range": (5, 6), "star_points": 6,
                    "para_range": (3, 5)}
        return {"qtypes": ["count_triangles", "count_rectangles",
                           "count_squares", "count_regions",
                           "count_triangles_star", "count_parallelograms"],
                "qtype_weights": [3, 2, 2, 2, 1, 2],
                "subdivide": 7, "grid_range": (5, 5),
                "n_lines_range": (5, 6), "star_points": 5,
                "para_range": (4, 5)}

    def _dispatch(self, qtype, level, cfg, parameter):
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )
        if qtype == "count_simple_shapes":
            return self._count_simple_shapes_problem(sub_rng, cfg)
        if qtype == "count_triangles":
            n = cfg.get("subdivide", 3)
            # add small jitter in n at high levels for variety
            if level >= 4:
                n = max(2, min(8, n + sub_rng.choice([-1, 0, 0])))
            return self._count_triangles_problem(sub_rng, n)
        if qtype == "count_rectangles":
            gr = cfg.get("grid_range", (2, 3))
            rows = sub_rng.randint(gr[0], gr[1])
            cols = sub_rng.randint(gr[0], gr[1])
            return self._count_rectangles_problem(sub_rng, rows, cols)
        if qtype == "count_squares":
            gr = cfg.get("grid_range", (2, 3))
            size = sub_rng.randint(gr[0], gr[1])
            return self._count_squares_problem(sub_rng, size)
        if qtype == "count_regions":
            nr = cfg.get("n_lines_range", (3, 4))
            n = sub_rng.randint(nr[0], nr[1])
            return self._count_regions_problem(sub_rng, n)
        if qtype == "count_triangles_star":
            return self._count_triangles_star_problem(sub_rng, cfg.get("star_points", 5))
        if qtype == "count_parallelograms":
            pr = cfg.get("para_range", (2, 4))
            rows = sub_rng.randint(pr[0], pr[1])
            cols = sub_rng.randint(pr[0], pr[1])
            return self._count_parallelograms_problem(sub_rng, rows, cols)
        return None

    # ------------------------------------------------------------------ #
    # Simple separate shapes (L0-L2)
    # ------------------------------------------------------------------ #

    def _count_simple_shapes_problem(self, rng, cfg):
        n = rng.randint(cfg["simple_min"], cfg["simple_max"])
        types_all = ["circle", "square", "triangle"]
        n_types = min(cfg["simple_n_types"], 3)
        type_pool = rng.sample(types_all, n_types)

        shapes: List[Tuple[float, float, float, str]] = []
        canvas = 100.0
        min_sep = 25
        obj_size = 8
        tries = 0
        while len(shapes) < n and tries < 500:
            x = rng.uniform(obj_size + 5, canvas - obj_size - 5)
            y = rng.uniform(obj_size + 5, canvas - obj_size - 5)
            ok = True
            for sx, sy, _, _ in shapes:
                if math.hypot(x - sx, y - sy) < min_sep:
                    ok = False
                    break
            if ok:
                shapes.append((x, y, obj_size, rng.choice(type_pool)))
            tries += 1
        if len(shapes) < max(1, cfg["simple_min"]):
            return None

        image = self._draw_simple_shapes(rng, shapes, canvas)
        stem = rng.choice([
            "How many shapes are drawn in the image?",
            "Count the total number of geometric figures shown.",
            "How many figures are there in total?",
        ])
        return stem + " Answer with a single integer.", str(len(shapes)), image

    def _draw_simple_shapes(self, rng, shapes, canvas):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, canvas)
        ax.set_ylim(0, canvas)
        ax.set_aspect("equal")
        ax.axis("off")

        palette = list(style["palette"])
        rng.shuffle(palette)
        for i, (x, y, s, kind) in enumerate(shapes):
            col = palette[i % len(palette)]
            if kind == "circle":
                ax.add_patch(plt.Circle((x, y), s, facecolor=col,
                                         edgecolor='black', linewidth=1.2, alpha=0.9))
            elif kind == "square":
                ax.add_patch(plt.Rectangle((x - s, y - s), 2 * s, 2 * s,
                                            facecolor=col, edgecolor='black',
                                            linewidth=1.2, alpha=0.9))
            elif kind == "triangle":
                h = s * math.sqrt(3)
                verts = [(x, y + h * 0.67),
                         (x - s, y - h * 0.33),
                         (x + s, y - h * 0.33)]
                ax.add_patch(plt.Polygon(verts, facecolor=col,
                                          edgecolor='black', linewidth=1.2, alpha=0.9))

        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Nested triangles
    # ------------------------------------------------------------------ #

    def _count_triangles_problem(self, rng, n):
        total = _count_triangles_in_subdivided(n)
        if total == 0:
            return None
        image = self._draw_subdivided_triangle(n, rng)
        stem = rng.choice([
            f"The figure shows an equilateral triangle divided into {n} rows of smaller triangles. "
            f"How many triangles of ALL sizes can you count in total? "
            f"(Include all small, medium, and large triangles, both upward- and downward-pointing.)",
            f"Count every triangle (of any size, any orientation) visible in this subdivided triangle figure.",
        ])
        return stem + " Answer with a single integer.", str(total), image

    def _draw_subdivided_triangle(self, n, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        h = n * math.sqrt(3) / 2
        A = (0, h)
        B = (-n / 2, 0)
        C = (n / 2, 0)
        line_color = style["geo_line_color"]
        lw = style["line_width"]

        points = {}
        for row in range(n + 1):
            for col in range(row + 1):
                x = A[0] + (B[0] - A[0]) * row / n + (C[0] - B[0]) * col / n
                y = A[1] + (B[1] - A[1]) * row / n + (C[1] - B[1]) * col / n
                points[(row, col)] = (x, y)

        drawn = set()
        for row in range(n + 1):
            for col in range(row + 1):
                p = points[(row, col)]
                if col + 1 <= row:
                    q = points[(row, col + 1)]
                    key = (min(p, q), max(p, q))
                    if key not in drawn:
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=line_color,
                                linewidth=lw, zorder=2)
                        drawn.add(key)
                if row + 1 <= n:
                    q = points[(row + 1, col)]
                    key = (min(p, q), max(p, q))
                    if key not in drawn:
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=line_color,
                                linewidth=lw, zorder=2)
                        drawn.add(key)
                if row + 1 <= n and col + 1 <= row + 1:
                    q = points[(row + 1, col + 1)]
                    key = (min(p, q), max(p, q))
                    if key not in drawn:
                        ax.plot([p[0], q[0]], [p[1], q[1]], color=line_color,
                                linewidth=lw, zorder=2)
                        drawn.add(key)

        for p in points.values():
            ax.plot(p[0], p[1], "o", color=line_color, markersize=4, zorder=3)

        for row in range(n):
            for col in range(row + 1):
                p1 = points[(row, col)]
                p2 = points[(row + 1, col)]
                p3 = points[(row + 1, col + 1)]
                tri = plt.Polygon([p1, p2, p3], facecolor="#eaf2f8",
                                  edgecolor="none", alpha=0.5, zorder=1)
                ax.add_patch(tri)
                if col > 0:
                    p1 = points[(row, col - 1)]
                    p2 = points[(row, col)]
                    p3 = points[(row + 1, col)]
                    tri = plt.Polygon([p1, p2, p3], facecolor="#fef9e7",
                                      edgecolor="none", alpha=0.5, zorder=1)
                    ax.add_patch(tri)

        margin = 0.5
        ax.set_xlim(-n / 2 - margin, n / 2 + margin)
        ax.set_ylim(-margin, h + margin)
        title = rng.choice([f"Subdivided Triangle ({n} rows)",
                            f"Triangle Subdivision (n={n})",
                            f"{n}-row Triangle"])
        ax.set_title(title, fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Grids
    # ------------------------------------------------------------------ #

    def _count_rectangles_problem(self, rng, rows, cols):
        total = _count_rectangles_in_grid(rows, cols)
        image = self._draw_grid(rows, cols, rng, "Rectangle")
        stem = (f"The figure shows a {rows} x {cols} grid. "
                f"How many rectangles (including squares) of ALL sizes can be formed "
                f"using the grid lines?")
        return stem + " Answer with a single integer.", str(total), image

    def _count_squares_problem(self, rng, size):
        total = _count_squares_in_grid(size, size)
        image = self._draw_grid(size, size, rng, "Square")
        stem = (f"The figure shows a {size} x {size} grid. "
                f"How many squares of ALL sizes can be formed using the grid lines?")
        return stem + " Answer with a single integer.", str(total), image

    def _draw_grid(self, rows, cols, rng, title_prefix):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        line_color = style["geo_line_color"]
        lw = style["line_width"]

        for r in range(rows + 1):
            ax.plot([0, cols], [r, r], color=line_color, linewidth=lw, zorder=2)
        for c in range(cols + 1):
            ax.plot([c, c], [0, rows], color=line_color, linewidth=lw, zorder=2)

        for r in range(rows):
            for c in range(cols):
                color = "#eaf2f8" if (r + c) % 2 == 0 else "#fef9e7"
                ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color,
                                            edgecolor="none", alpha=0.5, zorder=1))
        for r in range(rows + 1):
            for c in range(cols + 1):
                ax.plot(c, r, "o", color=line_color, markersize=5, zorder=3)

        margin = 0.5
        ax.set_xlim(-margin, cols + margin)
        ax.set_ylim(-margin, rows + margin)
        ax.axis("off")
        title = rng.choice([f"{title_prefix} Grid {rows}x{cols}",
                            f"{rows}x{cols} Grid",
                            f"Grid ({rows}x{cols})"])
        ax.set_title(title, fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Intersecting lines
    # ------------------------------------------------------------------ #

    def _count_regions_problem(self, rng, n_lines):
        lines = []
        boundary = 5.0
        attempts = 0
        while len(lines) < n_lines and attempts < 200:
            attempts += 1
            angle = rng.uniform(0, math.pi)
            if any(abs(angle - existing_a) < 0.18 for _, _, existing_a in lines):
                continue
            offset = rng.uniform(-2, 2)
            dx = math.cos(angle)
            dy = math.sin(angle)
            px = offset * math.sin(angle)
            py = -offset * math.cos(angle)
            t_range = boundary * 2
            p1 = (px - t_range * dx, py - t_range * dy)
            p2 = (px + t_range * dx, py + t_range * dy)
            lines.append((p1, p2, angle))
        if len(lines) < n_lines:
            return None
        n_int = 0
        int_points = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = _line_intersection(lines[i][0], lines[i][1],
                                         lines[j][0], lines[j][1])
                if pt is not None:
                    x, y = pt
                    if abs(x) < boundary and abs(y) < boundary:
                        n_int += 1
                        int_points.append(pt)

        total_regions = 1 + n_lines + n_int
        image = self._draw_intersecting_lines(lines, int_points, boundary, rng)
        stem = (f"The figure shows {n_lines} straight lines drawn across a plane. "
                f"How many regions do these lines divide the plane into?")
        return stem + " Answer with a single integer.", str(total_regions), image

    def _draw_intersecting_lines(self, lines, int_points, boundary, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        colors = list(style["palette"])
        rng.shuffle(colors)
        lw = style["line_width"]
        for i, (p1, p2, _) in enumerate(lines):
            color = colors[i % len(colors)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color,
                    linewidth=lw, zorder=2, label=f"Line {i + 1}")
        for pt in int_points:
            ax.plot(pt[0], pt[1], "o", color="#2c3e50",
                    markersize=6, zorder=4)
        ax.set_xlim(-boundary, boundary)
        ax.set_ylim(-boundary, boundary)
        ax.legend(fontsize=style["font_size_base"] - 1, loc="upper right")
        ax.set_title(f"{len(lines)} Intersecting Lines",
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Stars
    # ------------------------------------------------------------------ #

    def _count_triangles_star_problem(self, rng, n_points):
        if n_points == 5:
            total = 35
            star_name = "five-pointed star (pentagram)"
        else:
            total = 20
            star_name = "six-pointed star (hexagram)"
        image = self._draw_star(n_points, rng)
        stem = (f"The figure shows a {star_name} formed by connecting vertices. "
                f"How many triangles of ALL sizes can you count in the figure?")
        return stem + " Answer with a single integer.", str(total), image

    def _draw_star(self, n_points, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        radius = 4.0
        outer = []
        for i in range(n_points):
            a = math.pi / 2 + 2 * math.pi * i / n_points
            outer.append((radius * math.cos(a), radius * math.sin(a)))

        palette = list(style["palette"])
        rng.shuffle(palette)
        lw = style["line_width"]
        if n_points == 5:
            for i in range(5):
                j = (i + 2) % 5
                ax.plot([outer[i][0], outer[j][0]],
                        [outer[i][1], outer[j][1]],
                        color=style["geo_line_color"], linewidth=lw, zorder=2)
        else:
            tri1 = [outer[0], outer[2], outer[4]]
            tri2 = [outer[1], outer[3], outer[5]]
            for tri, color in [(tri1, palette[0]), (tri2, palette[1 % len(palette)])]:
                ax.add_patch(plt.Polygon(tri, closed=True, facecolor=color,
                                          alpha=0.15, edgecolor=color,
                                          linewidth=2.5, zorder=2))
        for i, (x, y) in enumerate(outer):
            ax.plot(x, y, "o", color=style["geo_line_color"],
                    markersize=6, zorder=3)

        margin = 1.0
        ax.set_xlim(-radius - margin, radius + margin)
        ax.set_ylim(-radius - margin, radius + margin)
        name = "Pentagram" if n_points == 5 else "Hexagram"
        ax.set_title(f"{name} - Count all triangles",
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Parallelograms in slanted grid (merged from config_counting)
    # ------------------------------------------------------------------ #
    def _count_parallelograms_problem(self, rng, rows, cols):
        rows = max(2, min(6, rows))
        cols = max(2, min(6, cols))
        answer = math.comb(rows + 1, 2) * math.comb(cols + 1, 2)
        image = self._draw_parallelogram_grid(rows, cols, rng)
        _POOL = [
            "The figure shows a grid of parallelograms. How many parallelograms of all sizes can be found?",
            "Count the parallelograms of every size in the slanted grid.",
            "From the slanted grid shown, determine the total number of parallelograms (all sizes).",
            "The image shows a parallelogram lattice. How many parallelograms (any size) are present?",
            "In the figure, a grid of parallelograms is drawn. Count all parallelograms of every size.",
            "Given the slanted (parallelogram) grid, how many parallelograms of all sizes can you count?",
            "The figure is a parallelogram grid. Find the total number of parallelograms of every size.",
            "How many parallelograms (of every possible size) appear in the slanted grid?",
            "Counting parallelograms of all sizes in the figure — what is the total?",
            "Find the number of parallelograms of all sizes in the slanted grid shown.",
            "The figure displays a parallelogram lattice. What is the total parallelogram count (all sizes)?",
            "Using the slanted grid in the image, count every parallelogram of any size.",
            "How many parallelograms of all sizes are contained in the figure's slanted grid?",
            "Count every parallelogram — large and small — visible in the figure.",
            "In the parallelogram grid shown, what is the count of parallelograms across all sizes?",
            "Total parallelograms of all sizes in the slanted grid: how many?",
        ]
        sidx = (self.seed or 0) % 16
        stem = _POOL[sidx]
        return stem + " Answer with a single integer.", str(answer), image

    def _draw_parallelogram_grid(self, rows, cols, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        line_color = style["geo_line_color"]
        lw = style["line_width"]
        palette = list(style["palette"])
        rng.shuffle(palette)
        skew = rng.choice([0.3, 0.35, 0.4, 0.45, 0.5])

        for r in range(rows + 1):
            ax.plot([r * skew, cols + r * skew], [r, r],
                    color=line_color, linewidth=lw, zorder=2)
        for c in range(cols + 1):
            ax.plot([c, c + rows * skew], [0, rows],
                    color=line_color, linewidth=lw, zorder=2)
        for r in range(rows + 1):
            for c in range(cols + 1):
                ax.plot(c + r * skew, r, "o", color=line_color,
                        markersize=5, zorder=3)
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    x0 = c + r * skew
                    x1 = c + 1 + r * skew
                    x2 = c + 1 + (r + 1) * skew
                    x3 = c + (r + 1) * skew
                    ax.add_patch(plt.Polygon(
                        [(x0, r), (x1, r), (x2, r + 1), (x3, r + 1)],
                        facecolor=palette[0], alpha=0.08,
                        edgecolor="none", zorder=1))
        total_w = cols + rows * skew
        ax.set_xlim(-0.5, total_w + 0.5)
        ax.set_ylim(-0.5, rows + 0.8)
        title_variants = [
            f"Parallelogram Grid ({rows}x{cols})",
            f"Slanted Grid {rows}x{cols}",
            "Count Parallelograms",
            f"{rows}x{cols} Parallelogram Lattice",
        ]
        ax.set_title(rng.choice(title_variants),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = FigureCountingQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
