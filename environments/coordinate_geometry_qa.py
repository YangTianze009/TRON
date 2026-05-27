"""
Coordinate Geometry QA environment.

Generates points, lines, and shapes on a coordinate plane and asks
for distance, midpoint, slope, area, or line equation.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .base import StandaloneVisualEnv

class CoordinateGeometryQA(StandaloneVisualEnv):
    ENV_NAME = "coordinate_geometry"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    ASK_TYPES = ["distance", "midpoint", "slope", "area", "line_equation"]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"ask_types": ["distance", "midpoint"],
                    "num_points": 2}
        if level <= 5:
            return {"ask_types": ["distance", "midpoint", "slope",
                                  "line_equation"],
                    "num_points": 2}
        if level <= 7:
            return {"ask_types": ["slope", "area", "line_equation",
                                  "perpendicular_line"],
                    "num_points": 3}
        return {"ask_types": ["area", "perpendicular_line",
                              "line_intersection",
                              "point_to_line_distance"],
                "num_points": 3}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        # Support both 'ask_type' (legacy) and 'problem_type' (controller)
        ask_type = parameter.get("problem_type",
                                 parameter.get("ask_type",
                                               self._rng.choice(cfg["ask_types"])))
        num_points = parameter.get("num_points", cfg["num_points"])

        for _ in range(30):
            result = self._dispatch(ask_type, num_points)
            if result is not None:
                return result
        return None

    def _dispatch(self, ask_type: str, num_points: int):
        if ask_type == "distance":
            return self._distance()
        elif ask_type == "midpoint":
            return self._midpoint()
        elif ask_type == "slope":
            return self._slope()
        elif ask_type == "area":
            return self._area(num_points)
        elif ask_type == "line_equation":
            return self._line_equation()
        elif ask_type == "perpendicular_line":
            return self._perpendicular_line()
        elif ask_type == "triangle_area_3points":
            return self._area(3)
        elif ask_type == "line_intersection":
            return self._line_intersection()
        elif ask_type == "point_to_line_distance":
            return self._point_to_line_distance()
        elif ask_type == "circle_equation":
            return self._circle_equation()
        elif ask_type == "parallelogram_vertex":
            return self._parallelogram_vertex()
        elif ask_type == "line_circle_intersection":
            return self._line_circle_intersection()
        elif ask_type == "reflection":
            return self._reflection()
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _rand_int_point(self, lo=-8, hi=8):
        """Generate a random integer coordinate point."""
        x = self._rng.randint(lo, hi)
        y = self._rng.randint(lo, hi)
        return (x, y)

    def _distinct_points(self, n, lo=-8, hi=8):
        """Generate n distinct integer coordinate points."""
        pts = set()
        attempts = 0
        while len(pts) < n and attempts < 200:
            pts.add((self._rng.randint(lo, hi), self._rng.randint(lo, hi)))
            attempts += 1
        if len(pts) < n:
            return None
        return list(pts)

    def _setup_axes(self, ax, points, extra_margin=2, style=None):
        """Configure coordinate axes with grid."""
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]
        x_min = min(all_x) - extra_margin
        x_max = max(all_x) + extra_margin
        y_min = min(all_y) - extra_margin
        y_max = max(all_y) + extra_margin

        # Ensure origin is visible if close
        if x_min > -1:
            x_min = -1
        if x_max < 1:
            x_max = 1
        if y_min > -1:
            y_min = -1
        if y_max < 1:
            y_max = 1

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")

        grid_style = style.get("grid_style", "--") if style else "--"
        grid_alpha = style.get("grid_alpha", 0.4) if style else 0.4
        geo_line = style.get("geo_line_color", "#2c3e50") if style else "#2c3e50"
        lw = style.get("line_width", 1.2) if style else 1.2
        fs_base = style.get("font_size_base", 12) if style else 12
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"

        ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color="#bdc3c7")

        # Axes through origin
        ax.axhline(y=0, color=geo_line, linewidth=lw)
        ax.axvline(x=0, color=geo_line, linewidth=lw)

        # Integer ticks
        ax.set_xticks(range(int(x_min), int(x_max) + 1))
        ax.set_yticks(range(int(y_min), int(y_max) + 1))
        ax.tick_params(labelsize=fs_base - 2)

        ax.set_xlabel("x", fontsize=fs_base, fontfamily=ff)
        ax.set_ylabel("y", fontsize=fs_base, fontfamily=ff)

    def _plot_point(self, ax, x, y, label, color="#e74c3c", style=None):
        """Plot a labeled point."""
        fs = style.get("font_size_base", 12) if style else 12
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"
        bg = style.get("bg_color", "white") if style else "white"
        ax.plot(x, y, "o", color=color, markersize=8, zorder=5)
        ax.text(
            x + 0.35, y + 0.45, f"{label}({x},{y})",
            fontsize=fs, fontweight="bold", color=color,
            fontfamily=ff,
            bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec=color, alpha=0.85),
            zorder=6,
        )

    # ------------------------------------------------------------------ #
    # Distance between two points
    # ------------------------------------------------------------------ #

    def _distance(self):
        rng = self._rng
        pts = self._distinct_points(2)
        if pts is None:
            return None
        (x1, y1), (x2, y2) = pts

        # Avoid trivially zero distance (already ensured by distinct, but also
        # avoid same row/col for more interesting problems)
        if x1 == x2 and y1 == y2:
            return None

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dist = round(dist, 2)

        question_text = (
            "Find the distance between points P1 and P2 shown in the figure. "
            "Round to 2 decimal places."
        )
        answer_str = str(dist)

        # --- Draw ---
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, style=style)

        colors = [style["palette"][0], style["palette"][1]]
        self._plot_point(ax, x1, y1, "P1", colors[0], style=style)
        self._plot_point(ax, x2, y2, "P2", colors[1], style=style)

        # Dashed line between points
        ax.plot([x1, x2], [y1, y2], color=style["geo_line_color"], lw=style["line_width"], ls="--", zorder=3)

        # "?" label on the line
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.3, my - 0.5, "d = ?", fontsize=style["font_size_base"] + 1,
                color=style["palette"][2], fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#ffeaa7",
                          ec=style["palette"][2], alpha=0.9),
                zorder=6)

        ax.set_title("Distance Between Two Points", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Midpoint
    # ------------------------------------------------------------------ #

    def _midpoint(self):
        rng = self._rng
        pts = self._distinct_points(2)
        if pts is None:
            return None
        (x1, y1), (x2, y2) = pts

        mx_val = (x1 + x2) / 2
        my_val = (y1 + y2) / 2

        # Format answer: if halves are integers, use int
        if mx_val == int(mx_val):
            mx_val = int(mx_val)
        else:
            mx_val = round(mx_val, 2)
        if my_val == int(my_val):
            my_val = int(my_val)
        else:
            my_val = round(my_val, 2)

        question_text = (
            "Find the midpoint of the segment from P1 to P2 shown in the "
            "figure. Answer as (x,y)."
        )
        answer_str = f"({mx_val},{my_val})"

        # --- Draw ---
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, style=style)

        colors = [style["palette"][0], style["palette"][1]]
        self._plot_point(ax, x1, y1, "P1", colors[0], style=style)
        self._plot_point(ax, x2, y2, "P2", colors[1], style=style)

        # Segment
        ax.plot([x1, x2], [y1, y2], color=style["geo_line_color"], lw=style["line_width"], zorder=3)

        # Midpoint marker
        mpx, mpy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.plot(mpx, mpy, "D", color=style["palette"][2], markersize=10, zorder=5)
        ax.text(mpx + 0.4, mpy - 0.5, "M = ?", fontsize=style["font_size_base"] + 1,
                color=style["palette"][2], fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#ffeaa7",
                          ec=style["palette"][2], alpha=0.9),
                zorder=6)

        ax.set_title("Midpoint of a Segment", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Slope
    # ------------------------------------------------------------------ #

    def _slope(self):
        rng = self._rng
        pts = self._distinct_points(2)
        if pts is None:
            return None
        (x1, y1), (x2, y2) = pts

        if x1 == x2:
            return None  # undefined slope, retry

        slope_val = (y2 - y1) / (x2 - x1)
        slope_val = round(slope_val, 2)
        if slope_val == int(slope_val):
            slope_val = int(slope_val)

        question_text = (
            "Find the slope of the line through points P1 and P2 shown in "
            "the figure."
        )
        answer_str = str(slope_val)

        # --- Draw ---
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, extra_margin=3, style=style)

        colors = [style["palette"][0], style["palette"][1]]
        self._plot_point(ax, x1, y1, "P1", colors[0], style=style)
        self._plot_point(ax, x2, y2, "P2", colors[1], style=style)

        # Extend line beyond points
        x_lo, x_hi = ax.get_xlim()
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        lx = [x_lo, x_hi]
        ly = [m * x_lo + b, m * x_hi + b]
        ax.plot(lx, ly, color=style["palette"][2], lw=style["line_width"], zorder=2, alpha=0.7)

        # Label slope = ?
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.5, my - 0.6, "slope = ?", fontsize=style["font_size_base"] + 1,
                color=style["palette"][3], fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#ffeaa7",
                          ec=style["palette"][3], alpha=0.9),
                zorder=6)

        ax.set_title("Slope of a Line", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Area (polygon via shoelace)
    # ------------------------------------------------------------------ #

    def _area(self, num_points: int):
        rng = self._rng
        n = max(3, min(num_points, 6))

        pts = self._distinct_points(n, lo=-7, hi=7)
        if pts is None:
            return None

        # Order points by angle from centroid for convex-ish polygon
        cx_c = sum(p[0] for p in pts) / n
        cy_c = sum(p[1] for p in pts) / n
        pts.sort(key=lambda p: math.atan2(p[1] - cy_c, p[0] - cx_c))

        # Shoelace formula
        area = 0
        for i in range(n):
            x_i, y_i = pts[i]
            x_j, y_j = pts[(i + 1) % n]
            area += x_i * y_j - x_j * y_i
        area = abs(area) / 2.0
        area = round(area, 2)

        if area < 1:
            return None  # too small, degenerate

        labels = [f"P{i+1}" for i in range(n)]
        pts_str = ", ".join(labels)

        question_text = (
            f"Find the area of the polygon with vertices {pts_str} shown in "
            f"the figure. Round to 2 decimal places if needed."
        )
        # If area is integer, format as int
        if area == int(area):
            answer_str = str(int(area))
        else:
            answer_str = str(area)

        # --- Draw ---
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, style=style)

        # Draw filled polygon
        poly = plt.Polygon(pts, fill=True, facecolor=style["palette"][0],
                           edgecolor=style["geo_line_color"],
                           linewidth=style["line_width"] + 0.5,
                           alpha=style["geo_fill_alpha"], zorder=3)
        ax.add_patch(poly)

        # Plot and label each vertex
        for i, (x, y) in enumerate(pts):
            col = style["palette"][i % len(style["palette"])]
            self._plot_point(ax, x, y, labels[i], col, style=style)

        # "Area = ?" at centroid
        ax.text(cx_c, cy_c, "Area = ?", fontsize=style["font_size_base"] + 2,
                color=style["palette"][1], fontweight="bold",
                fontfamily=style["font_family"],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffeaa7",
                          ec=style["palette"][1], alpha=0.9),
                zorder=6)

        ax.set_title("Area of a Polygon", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Line equation: y = mx + b
    # ------------------------------------------------------------------ #

    def _line_equation(self):
        rng = self._rng
        pts = self._distinct_points(2)
        if pts is None:
            return None
        (x1, y1), (x2, y2) = pts

        if x1 == x2:
            return None  # vertical line, no y=mx+b form

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        m = round(m, 2)
        b = round(b, 2)
        if m == int(m):
            m = int(m)
        if b == int(b):
            b = int(b)

        # Format equation string
        if b == 0:
            eq = f"y = {m}x"
        elif b > 0:
            eq = f"y = {m}x + {b}"
        else:
            eq = f"y = {m}x - {abs(b)}"

        question_text = (
            "Find the equation of the line through points P1 and P2 shown in "
            "the figure, in the form y = mx + b."
        )
        answer_str = eq

        # --- Draw ---
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, extra_margin=3, style=style)

        colors = [style["palette"][0], style["palette"][1]]
        self._plot_point(ax, x1, y1, "P1", colors[0], style=style)
        self._plot_point(ax, x2, y2, "P2", colors[1], style=style)

        # Draw extended line
        x_lo, x_hi = ax.get_xlim()
        m_float = (y2 - y1) / (x2 - x1)
        b_float = y1 - m_float * x1
        lx = [x_lo, x_hi]
        ly = [m_float * x_lo + b_float, m_float * x_hi + b_float]
        ax.plot(lx, ly, color=style["palette"][2], lw=style["line_width"] + 0.5, zorder=2, alpha=0.8)

        # Label
        mx_p, my_p = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx_p + 0.5, my_p - 0.7, "y = ?", fontsize=style["font_size_base"] + 2,
                color=style["palette"][3], fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#ffeaa7",
                          ec=style["palette"][3], alpha=0.9),
                zorder=6)

        ax.set_title("Equation of a Line", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Perpendicular line through a point
    # ------------------------------------------------------------------ #

    def _perpendicular_line(self):
        rng = self._rng
        pts = self._distinct_points(3)
        if pts is None:
            return None
        (x1, y1), (x2, y2), (x3, y3) = pts
        if x1 == x2:
            return None
        m = (y2 - y1) / (x2 - x1)
        if m == 0:
            return None
        perp_m = -1 / m
        perp_b = y3 - perp_m * x3
        perp_m = round(perp_m, 2)
        perp_b = round(perp_b, 2)
        if perp_m == int(perp_m): perp_m = int(perp_m)
        if perp_b == int(perp_b): perp_b = int(perp_b)

        if perp_b == 0:
            eq = f"y = {perp_m}x"
        elif perp_b > 0:
            eq = f"y = {perp_m}x + {perp_b}"
        else:
            eq = f"y = {perp_m}x - {abs(perp_b)}"

        question_text = (
            "Find the equation of the line perpendicular to the line "
            "through P1 and P2 shown in the figure, passing through P3. "
            "Answer as y = mx + b.")
        answer_str = eq

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, extra_margin=3, style=style)
        self._plot_point(ax, x1, y1, "P1", style["palette"][0], style=style)
        self._plot_point(ax, x2, y2, "P2", style["palette"][1], style=style)
        self._plot_point(ax, x3, y3, "P3", style["palette"][2], style=style)
        x_lo, x_hi = ax.get_xlim()
        m_f = (y2 - y1) / (x2 - x1)
        b_f = y1 - m_f * x1
        ax.plot([x_lo, x_hi], [m_f * x_lo + b_f, m_f * x_hi + b_f],
                color=style["palette"][3], lw=style["line_width"], zorder=2, alpha=0.6, label="given line")
        pm_f = -1 / m_f
        pb_f = y3 - pm_f * x3
        ax.plot([x_lo, x_hi], [pm_f * x_lo + pb_f, pm_f * x_hi + pb_f],
                color=style["palette"][4], lw=style["line_width"], ls="--", zorder=2, alpha=0.6, label="perp line = ?")
        ax.legend(fontsize=style["font_size_base"] - 2)
        ax.set_title("Perpendicular Line", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Line intersection
    # ------------------------------------------------------------------ #

    def _line_intersection(self):
        rng = self._rng
        pts = self._distinct_points(4)
        if pts is None:
            return None
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts
        if x1 == x2 or x3 == x4:
            return None
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)
        if abs(m1 - m2) < 1e-9:
            return None  # parallel
        b1 = y1 - m1 * x1
        b2 = y3 - m2 * x3
        ix = (b2 - b1) / (m1 - m2)
        iy = m1 * ix + b1
        ix = round(ix, 2)
        iy = round(iy, 2)
        if ix == int(ix): ix = int(ix)
        if iy == int(iy): iy = int(iy)

        question_text = (
            "Line L1 passes through points P1 and P2 shown in the figure. "
            "Line L2 passes through points P3 and P4. "
            "Find the intersection point. Answer as (x,y).")
        answer_str = f"({ix},{iy})"

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, extra_margin=3, style=style)
        line_cols = [style["palette"][0], style["palette"][1]]
        for i, (x, y, label, col_idx) in enumerate([
            (x1, y1, "P1", 0), (x2, y2, "P2", 0),
            (x3, y3, "P3", 1), (x4, y4, "P4", 1)]):
            self._plot_point(ax, x, y, label, line_cols[col_idx], style=style)
        x_lo, x_hi = ax.get_xlim()
        for m, b, col in [(m1, b1, line_cols[0]), (m2, b2, line_cols[1])]:
            ax.plot([x_lo, x_hi], [m * x_lo + b, m * x_hi + b],
                    color=col, lw=style["line_width"], alpha=0.5, zorder=2)
        ax.set_title("Line Intersection", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Point to line distance
    # ------------------------------------------------------------------ #

    def _point_to_line_distance(self):
        rng = self._rng
        pts = self._distinct_points(3)
        if pts is None:
            return None
        (x1, y1), (x2, y2), (x3, y3) = pts
        if x1 == x2 and y1 == y2:
            return None
        # Line through P1P2: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        dist = abs(a * x3 + b * y3 + c) / math.sqrt(a ** 2 + b ** 2)
        dist = round(dist, 2)

        question_text = (
            "Find the distance from point P3 to the line through points "
            "P1 and P2 shown in the figure. Round to 2 decimal places.")
        answer_str = str(dist)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, extra_margin=3, style=style)
        self._plot_point(ax, x1, y1, "P1", style["palette"][0], style=style)
        self._plot_point(ax, x2, y2, "P2", style["palette"][0], style=style)
        self._plot_point(ax, x3, y3, "P3", style["palette"][2], style=style)
        if x1 != x2:
            m_f = (y2 - y1) / (x2 - x1)
            b_f = y1 - m_f * x1
            x_lo, x_hi = ax.get_xlim()
            ax.plot([x_lo, x_hi], [m_f * x_lo + b_f, m_f * x_hi + b_f],
                    color=style["palette"][1], lw=style["line_width"], alpha=0.5, zorder=2)
        ax.set_title("Point-to-Line Distance", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Circle equation from center and point
    # ------------------------------------------------------------------ #

    def _circle_equation(self):
        rng = self._rng
        cx = rng.randint(-5, 5)
        cy = rng.randint(-5, 5)
        # Point on the circle
        px = cx + rng.randint(1, 4)
        py = cy + rng.randint(-3, 3)
        r_sq = (px - cx) ** 2 + (py - cy) ** 2
        r = math.sqrt(r_sq)

        question_text = (
            "Find the radius of the circle centered at C that passes "
            "through P, as shown in the figure. Round to 2 decimal places.")
        answer_str = str(round(r, 2))

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        circle = plt.Circle((cx, cy), r, fill=False, edgecolor=style["palette"][0],
                             linewidth=style["line_width"] + 0.5, zorder=3)
        ax.add_patch(circle)
        pts = [(cx, cy), (px, py)]
        self._setup_axes(ax, [(cx - r - 1, cy - r - 1), (cx + r + 1, cy + r + 1)], style=style)
        self._plot_point(ax, cx, cy, "C", style["palette"][1], style=style)
        self._plot_point(ax, px, py, "P", style["palette"][2], style=style)
        ax.set_title("Circle Radius", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Parallelogram fourth vertex
    # ------------------------------------------------------------------ #

    def _parallelogram_vertex(self):
        rng = self._rng
        pts = self._distinct_points(3, lo=-6, hi=6)
        if pts is None:
            return None
        (x1, y1), (x2, y2), (x3, y3) = pts
        # Fourth vertex: P4 = P1 + P3 - P2 (parallelogram P1P2P3P4)
        x4 = x1 + x3 - x2
        y4 = y1 + y3 - y2

        question_text = (
            "Three vertices A, B, C of a parallelogram are shown in the "
            "figure. B is opposite to D. Find vertex D. Answer as (x,y).")
        answer_str = f"({x4},{y4})"

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        all_pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        self._setup_axes(ax, all_pts, style=style)
        for (x, y), label, col_idx in [((x1, y1), "A", 0), ((x2, y2), "B", 1),
                                         ((x3, y3), "C", 2)]:
            self._plot_point(ax, x, y, label, style["palette"][col_idx], style=style)
        ax.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1],
                color=style["geo_line_color"], lw=style["line_width"], ls="--", zorder=2)
        ax.plot(x4, y4, "D", color=style["palette"][3], markersize=12, zorder=5)
        ax.text(x4 + 0.4, y4 + 0.4, "D = ?", fontsize=style["font_size_base"],
                color=style["palette"][3], fontweight="bold",
                fontfamily=style["font_family"], zorder=6)
        ax.set_title("Find the Fourth Vertex", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Line-circle intersection count
    # ------------------------------------------------------------------ #

    def _line_circle_intersection(self):
        rng = self._rng
        cx, cy = rng.randint(-4, 4), rng.randint(-4, 4)
        r = rng.randint(2, 5)
        # Random line through two points
        pts = self._distinct_points(2, lo=-7, hi=7)
        if pts is None:
            return None
        (x1, y1), (x2, y2) = pts
        if x1 == x2 and y1 == y2:
            return None
        # Distance from center to line
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        dist = abs(a * cx + b * cy + c) / math.sqrt(a ** 2 + b ** 2)
        if dist > r + 0.01:
            n_intersections = 0
        elif abs(dist - r) < 0.5:
            n_intersections = 1
        else:
            n_intersections = 2

        question_text = (
            f"A circle has center C and radius {r}, and a line passes through "
            "points P1 and P2 (all shown in the figure). "
            "How many intersection points do the line and circle have?")
        answer_str = str(n_intersections)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        all_pts = [(cx - r - 1, cy - r - 1), (cx + r + 1, cy + r + 1),
                    (x1, y1), (x2, y2)]
        self._setup_axes(ax, all_pts, style=style)
        circle = plt.Circle((cx, cy), r, fill=False, edgecolor=style["palette"][0],
                             linewidth=style["line_width"] + 0.5, zorder=3)
        ax.add_patch(circle)
        self._plot_point(ax, cx, cy, "C", style["palette"][1], style=style)
        self._plot_point(ax, x1, y1, "P1", style["palette"][2], style=style)
        self._plot_point(ax, x2, y2, "P2", style["palette"][2], style=style)
        if x1 != x2:
            m_f = (y2 - y1) / (x2 - x1)
            b_f = y1 - m_f * x1
            x_lo, x_hi = ax.get_xlim()
            ax.plot([x_lo, x_hi], [m_f * x_lo + b_f, m_f * x_hi + b_f],
                    color=style["palette"][2], lw=style["line_width"], alpha=0.6, zorder=2)
        else:
            ax.axvline(x1, color=style["palette"][2], lw=style["line_width"], alpha=0.6, zorder=2)
        ax.set_title("Line-Circle Intersections", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img

    # ------------------------------------------------------------------ #
    # Reflection of a point across a line
    # ------------------------------------------------------------------ #

    def _reflection(self):
        rng = self._rng
        # Reflect point P across a line y=x, y=-x, x-axis, or y-axis
        px, py = rng.randint(-6, 6), rng.randint(-6, 6)
        line_type = rng.choice(["y=x", "y=-x", "x_axis", "y_axis"])
        if line_type == "y=x":
            rx, ry = py, px
            line_desc = "y = x"
        elif line_type == "y=-x":
            rx, ry = -py, -px
            line_desc = "y = -x"
        elif line_type == "x_axis":
            rx, ry = px, -py
            line_desc = "the x-axis"
        else:
            rx, ry = -px, py
            line_desc = "the y-axis"

        question_text = (
            f"Find the reflection of point P shown in the figure across "
            f"{line_desc}. Answer as (x,y).")
        answer_str = f"({rx},{ry})"

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        all_pts = [(px, py), (rx, ry)]
        self._setup_axes(ax, all_pts, extra_margin=3, style=style)
        self._plot_point(ax, px, py, "P", style["palette"][0], style=style)
        ax.plot(rx, ry, "D", color=style["palette"][2], markersize=10, zorder=5)
        ax.text(rx + 0.4, ry + 0.4, "P' = ?", fontsize=style["font_size_base"],
                color=style["palette"][2], fontweight="bold",
                fontfamily=style["font_family"], zorder=6)
        # Draw the reflection line
        x_lo, x_hi = ax.get_xlim()
        ref_color = style["palette"][3]
        if line_type == "y=x":
            ax.plot([x_lo, x_hi], [x_lo, x_hi], "--", color=ref_color, lw=style["line_width"], alpha=0.5, label="y=x")
        elif line_type == "y=-x":
            ax.plot([x_lo, x_hi], [-x_lo, -x_hi], "--", color=ref_color, lw=style["line_width"], alpha=0.5, label="y=-x")
        if line_type in ("y=x", "y=-x"):
            ax.legend(fontsize=style["font_size_base"] - 2)
        ax.set_title("Reflection", fontsize=style["font_size_base"] + 2,
                      fontfamily=style["font_family"], pad=10)
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question_text, answer_str, img
