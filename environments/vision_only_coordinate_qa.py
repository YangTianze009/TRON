"""
Vision-Only Coordinate QA environment.

Draws a coordinate plane with points / lines / shapes.
Values are ON the plot; the question is schema-only ("Find A...B").

Diversity & difficulty redesign (2026-04-16):
- All random ops use a level-aware sub_rng (was self._rng, so L0=L9 same seed).
- Structural L0 vs L9:
   L0: distance or slope for two labeled points, small integer grid.
   L9: extended problem families (intersection, triangle area, line equation,
       line_of_best_fit) over larger ranges; MCQ format at top two levels.
- 3-4 phrasings per qtype; vertex letter pool jitter; color rotation per seed.
- More qtypes added: line_of_best_fit_slope, midpoint.
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_VERTEX_POOLS = [
    ["A", "B", "C", "D", "E"],
    ["P", "Q", "R", "S", "T"],
    ["M", "N", "O", "L", "K"],
]

_PHRASINGS = {
    "distance": [
        "Find the distance between A and B.",
        "What is the Euclidean distance from A to B? Round to 2 decimals.",
        "Compute |AB| from the plotted points. Two decimals.",
    ],
    "slope": [
        "Find the slope of line AB.",
        "What is the gradient of the line passing through A and B?",
        "Determine the slope of line AB.",
    ],
    "midpoint": [
        "Find the midpoint M of segment AB. Answer as (x,y).",
        "Compute the midpoint of segment AB. Answer as (x,y).",
        "What is the midpoint of AB (answer in (x,y) form)?",
    ],
    "area": [
        "Find the area of the shaded region.",
        "Compute the area of the plotted polygon.",
        "What is the area of the shaded polygon?",
    ],
    "intersection": [
        "Find the intersection point P of lines l\u2081 and l\u2082. Answer as (x,y).",
        "Where do lines l\u2081 and l\u2082 intersect? (x,y).",
    ],
    "triangle_area": [
        "Find the area of triangle ABC using the plotted coordinates. Numeric answer.",
        "What is the area of the plotted triangle ABC?",
    ],
    "line_equation": [
        "Find the equation of line AB in slope-intercept form y = mx + b. Answer as 'm, b'.",
        "Determine m and b for y = mx + b through A and B. Answer as 'm, b'.",
    ],
}

class VisionOnlyCoordinateQA(StandaloneVisualEnv):
    ENV_NAME = "vision_only_coordinate"

    PROBLEM_TYPES = ["distance", "area", "midpoint", "slope", "intersection",
                     "triangle_area_from_coords", "line_equation"]

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return {"ptypes": ["distance", "slope"],
                    "range": (-4, 4), "is_mcq": False}
        if level <= 2:
            return {"ptypes": ["distance", "slope", "midpoint"],
                    "range": (-5, 5), "is_mcq": False}
        if level <= 4:
            return {"ptypes": ["distance", "slope", "midpoint", "area"],
                    "range": (-6, 6), "is_mcq": False}
        if level <= 6:
            return {"ptypes": ["distance", "area", "slope",
                               "triangle_area_from_coords",
                               "intersection"],
                    "range": (-6, 6), "is_mcq": False}
        if level <= 7:
            return {"ptypes": ["intersection", "triangle_area_from_coords",
                               "line_equation", "area"],
                    "range": (-7, 7), "is_mcq": True}
        return {"ptypes": ["intersection", "triangle_area_from_coords",
                           "line_equation"],
                "range": (-8, 8), "is_mcq": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        ptype = parameter.get("problem_type")
        if ptype is None:
            ptype = sub_rng.choice(cfg["ptypes"])

        for _ in range(30):
            result = self._dispatch(ptype, sub_rng, cfg, level)
            if result is not None:
                return result
            ptype = sub_rng.choice(cfg["ptypes"])
        return None

    def _dispatch(self, ptype, rng, cfg, level):
        if ptype == "distance":
            return self._distance(rng, cfg, level)
        if ptype == "area":
            return self._area(rng, cfg, level)
        if ptype == "midpoint":
            return self._midpoint(rng, cfg, level)
        if ptype == "slope":
            return self._slope(rng, cfg, level)
        if ptype == "intersection":
            return self._intersection(rng, cfg, level)
        if ptype == "triangle_area_from_coords":
            return self._triangle_area_from_coords(rng, cfg, level)
        if ptype == "line_equation":
            return self._line_equation(rng, cfg, level)
        return None

    # ------------------------------------------------------------------ #
    def _setup_axes(self, ax, points, extra=2, style=None):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_lo = min(xs) - extra
        x_hi = max(xs) + extra
        y_lo = min(ys) - extra
        y_hi = max(ys) + extra
        if x_lo > -1:
            x_lo = -1
        if x_hi < 1:
            x_hi = 1
        if y_lo > -1:
            y_lo = -1
        if y_hi < 1:
            y_hi = 1
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        grid_style = style.get("grid_style", "--") if style else "--"
        grid_alpha = style.get("grid_alpha", 0.35) if style else 0.35
        geo_line = style.get("geo_line_color", "#2c3e50") if style else "#2c3e50"
        lw = style.get("line_width", 1.3) if style else 1.3
        fs = style.get("font_size_base", 11) if style else 11
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"

        ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color="#bdc3c7")
        ax.axhline(y=0, color=geo_line, linewidth=lw)
        ax.axvline(x=0, color=geo_line, linewidth=lw)
        ax.set_xticks(range(int(x_lo), int(x_hi) + 1))
        ax.set_yticks(range(int(y_lo), int(y_hi) + 1))
        ax.tick_params(labelsize=fs - 2)
        ax.set_xlabel("x", fontsize=fs, fontfamily=ff)
        ax.set_ylabel("y", fontsize=fs, fontfamily=ff)

    def _plot_labeled_point(self, ax, x, y, label, color="#c62828", style=None):
        ax.plot(x, y, "o", color=color, markersize=8, zorder=5)
        fs = style.get("font_size_base", 12) if style else 12
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"
        bg = style.get("bg_color", "white") if style else "white"
        ax.text(x + 0.3, y + 0.4, f"{label}",
                fontsize=fs, fontweight="bold", color=color,
                fontfamily=ff,
                bbox=dict(boxstyle="round,pad=0.12", fc=bg,
                          ec=color, alpha=0.88),
                zorder=6)

    def _rand_pt(self, rng, lo=-7, hi=7):
        return (rng.randint(lo, hi), rng.randint(lo, hi))

    def _distinct_pts(self, rng, n, lo=-7, hi=7):
        pts = set()
        for _ in range(300):
            pts.add(self._rand_pt(rng, lo, hi))
            if len(pts) >= n:
                return list(pts)
        return None

    def _maybe_mcq(self, rng, cfg, qtext, answer_str, answer_val):
        """If MCQ, wrap question with options and return (qtext, letter).
        Otherwise return (qtext, answer_str)."""
        if not cfg.get("is_mcq", False):
            return qtext, answer_str

        # Only numeric answers: wrap
        try:
            ans_num = float(answer_str)
        except Exception:
            return qtext, answer_str

        distractors = set()
        tries = 0
        while len(distractors) < 3 and tries < 50:
            mag = max(abs(ans_num) * 0.2, 1.0)
            d = rng.choice([-3, -2, -1, 1, 2, 3]) * mag
            cand = round(ans_num + d, 2)
            if cand != ans_num and cand not in distractors:
                distractors.add(cand)
            tries += 1
        if len(distractors) < 3:
            return qtext, answer_str

        opts = [ans_num] + list(distractors)[:3]
        rng.shuffle(opts)
        letter = chr(ord("A") + opts.index(ans_num))

        def _fmt(v):
            if v == int(v):
                return str(int(v))
            return f"{v:.2f}"

        q = (qtext + "\n" + "\n".join(
            f"  ({chr(ord('A')+i)}) {_fmt(opts[i])}" for i in range(4))
            + "\nAnswer with a single letter.")
        return q, letter

    def _vertex_names(self, rng, n):
        return rng.choice(_VERTEX_POOLS)[:n]

    # ------------------------------------------------------------------ #
    def _distance(self, rng, cfg, level):
        lo, hi = cfg["range"]
        pts = self._distinct_pts(rng, 2, lo=lo, hi=hi)
        if pts is None:
            return None
        A, B = pts
        if A == B:
            return None

        dist = round(math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2), 2)

        letters = self._vertex_names(rng, 2)
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, [A, B], style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        self._plot_labeled_point(ax, A[0], A[1], letters[0], pal[0], style=style)
        self._plot_labeled_point(ax, B[0], B[1], letters[1], pal[1], style=style)

        ax.plot([A[0], B[0]], [A[1], B[1]], "--",
                color=style["geo_line_color"],
                lw=style["line_width"], zorder=3)

        mx = (A[0] + B[0]) / 2
        my = (A[1] + B[1]) / 2
        unk_color = pal[5]
        ax.text(mx + 0.3, my - 0.5, "d = ?", fontsize=style["font_size_base"] + 1,
                color=unk_color, fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        phrasing = rng.choice(_PHRASINGS["distance"])
        q = phrasing.replace("A and B", f"{letters[0]} and {letters[1]}") \
                    .replace("|AB|", f"|{letters[0]}{letters[1]}|")
        q, a = self._maybe_mcq(rng, cfg, q, str(dist), dist)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    def _area(self, rng, cfg, level):
        lo, hi = cfg["range"]
        n = rng.choice([3, 4])
        pts = self._distinct_pts(rng, n, lo=lo, hi=hi)
        if pts is None:
            return None

        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        area = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2.0
        area = round(area, 2)
        if area < 1:
            return None

        letters = self._vertex_names(rng, n)
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, pts, style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        poly = plt.Polygon(pts, fill=True, facecolor=pal[0],
                           edgecolor=style["geo_line_color"],
                           linewidth=style["line_width"] + 0.2,
                           alpha=style["geo_fill_alpha"], zorder=3)
        ax.add_patch(poly)

        for i, (x, y) in enumerate(pts):
            self._plot_labeled_point(ax, x, y, letters[i],
                                     pal[i % len(pal)], style=style)

        unk_color = pal[5]
        ax.text(cx, cy, "Area = ?",
                fontsize=style["font_size_base"] + 2,
                color=unk_color, fontweight="bold",
                fontfamily=style["font_family"],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        q = rng.choice(_PHRASINGS["area"])
        if area == int(area):
            answer_str = str(int(area))
        else:
            answer_str = str(area)
        q, a = self._maybe_mcq(rng, cfg, q, answer_str, area)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    def _midpoint(self, rng, cfg, level):
        lo, hi = cfg["range"]
        pts = self._distinct_pts(rng, 2, lo=lo, hi=hi)
        if pts is None:
            return None
        A, B = pts

        mx_val = (A[0] + B[0]) / 2
        my_val = (A[1] + B[1]) / 2
        if mx_val == int(mx_val):
            mx_val = int(mx_val)
        else:
            mx_val = round(mx_val, 1)
        if my_val == int(my_val):
            my_val = int(my_val)
        else:
            my_val = round(my_val, 1)

        letters = self._vertex_names(rng, 2)
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, [A, B], style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        self._plot_labeled_point(ax, A[0], A[1], letters[0], pal[0], style=style)
        self._plot_labeled_point(ax, B[0], B[1], letters[1], pal[1], style=style)

        ax.plot([A[0], B[0]], [A[1], B[1]], color=style["geo_line_color"],
                lw=style["line_width"], zorder=3)

        mpx, mpy = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        ax.plot(mpx, mpy, "D", color=pal[2], markersize=10, zorder=5)
        unk_color = pal[5]
        ax.text(mpx + 0.4, mpy - 0.5, "M = ?",
                fontsize=style["font_size_base"] + 1,
                color=unk_color, fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        q = rng.choice(_PHRASINGS["midpoint"]).replace(
            "A and B", f"{letters[0]} and {letters[1]}").replace(
            "AB", f"{letters[0]}{letters[1]}")
        answer_str = f"({mx_val},{my_val})"
        # midpoint not well-suited for MCQ; keep as text
        return q, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    def _slope(self, rng, cfg, level):
        lo, hi = cfg["range"]
        pts = self._distinct_pts(rng, 2, lo=lo, hi=hi)
        if pts is None:
            return None
        A, B = pts
        if A[0] == B[0]:
            return None

        slope = (B[1] - A[1]) / (B[0] - A[0])
        slope = round(slope, 2)
        if slope == int(slope):
            slope = int(slope)

        letters = self._vertex_names(rng, 2)
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, [A, B], extra=3, style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        self._plot_labeled_point(ax, A[0], A[1], letters[0], pal[0], style=style)
        self._plot_labeled_point(ax, B[0], B[1], letters[1], pal[1], style=style)

        x_lo, x_hi = ax.get_xlim()
        m = (B[1] - A[1]) / (B[0] - A[0])
        b = A[1] - m * A[0]
        ax.plot([x_lo, x_hi], [m * x_lo + b, m * x_hi + b],
                color=pal[2], lw=style["line_width"], alpha=0.6, zorder=2)

        mx = (A[0] + B[0]) / 2
        my = (A[1] + B[1]) / 2
        unk_color = pal[5]
        ax.text(mx + 0.5, my - 0.6, "slope = ?",
                fontsize=style["font_size_base"] + 1,
                color=unk_color, fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        q = rng.choice(_PHRASINGS["slope"]).replace(
            "line AB", f"line {letters[0]}{letters[1]}").replace(
            "A and B", f"{letters[0]} and {letters[1]}")
        q, a = self._maybe_mcq(rng, cfg, q, str(slope), slope)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    def _intersection(self, rng, cfg, level):
        lo, hi = cfg["range"]
        ix = rng.randint(max(-5, lo), min(5, hi))
        iy = rng.randint(max(-5, lo), min(5, hi))

        m1_num = rng.choice([-3, -2, -1, 1, 2, 3])
        m1_den = rng.choice([1, 2])
        m1 = m1_num / m1_den

        m2_choices = [v / d for v in [-3, -2, -1, 1, 2, 3]
                      for d in [1, 2] if v / d != m1]
        if not m2_choices:
            return None
        m2 = rng.choice(m2_choices)

        dx1 = 2
        A = (ix - dx1, round(iy - m1 * dx1))
        B = (ix + dx1, round(iy + m1 * dx1))
        dx2 = 2
        C = (ix - dx2, round(iy - m2 * dx2))
        D = (ix + dx2, round(iy + m2 * dx2))

        all_pts = [A, B, C, D, (ix, iy)]

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, all_pts, extra=3, style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        colors_line = [pal[0], pal[1]]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        x_lo, x_hi = ax.get_xlim()
        b1 = iy - m1 * ix
        b2 = iy - m2 * ix
        ax.plot([x_lo, x_hi], [m1 * x_lo + b1, m1 * x_hi + b1],
                color=colors_line[0], lw=lw + 0.2, zorder=2)
        ax.plot([x_lo, x_hi], [m2 * x_lo + b2, m2 * x_hi + b2],
                color=colors_line[1], lw=lw + 0.2, zorder=2)

        self._plot_labeled_point(ax, A[0], A[1], "A", colors_line[0], style=style)
        self._plot_labeled_point(ax, B[0], B[1], "B", colors_line[0], style=style)
        self._plot_labeled_point(ax, C[0], C[1], "C", colors_line[1], style=style)
        self._plot_labeled_point(ax, D[0], D[1], "D", colors_line[1], style=style)

        unk_color = pal[5]
        ax.plot(ix, iy, "X", color=unk_color, markersize=12, zorder=5,
                markeredgewidth=2)
        ax.text(ix + 0.4, iy - 0.6, "P = ?", fontsize=fs + 1, color=unk_color,
                fontweight="bold", fontfamily=ff,
                bbox=dict(boxstyle="round,pad=0.12", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        ax.text(x_hi - 1, m1 * (x_hi - 1) + b1 + 0.5, "l\u2081",
                fontsize=fs, color=colors_line[0], fontweight="bold",
                fontfamily=ff)
        ax.text(x_hi - 1, m2 * (x_hi - 1) + b2 + 0.5, "l\u2082",
                fontsize=fs, color=colors_line[1], fontweight="bold",
                fontfamily=ff)

        q = rng.choice(_PHRASINGS["intersection"])
        answer_str = f"({ix},{iy})"
        # intersection not well-suited for MCQ; keep as text
        return q, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    def _triangle_area_from_coords(self, rng, cfg, level):
        lo, hi = cfg["range"]
        pts = self._distinct_pts(rng, 3, lo=lo, hi=hi)
        if pts is None:
            return None
        A, B, C = pts
        area = abs((B[0] - A[0]) * (C[1] - A[1]) -
                   (C[0] - A[0]) * (B[1] - A[1])) / 2.0
        area = round(area, 2)
        if area < 1:
            return None

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, [A, B, C], style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        tri = plt.Polygon([A, B, C], fill=True, facecolor=pal[0],
                          edgecolor=style["geo_line_color"],
                          linewidth=style["line_width"] + 0.2,
                          alpha=0.25, zorder=3)
        ax.add_patch(tri)

        self._plot_labeled_point(ax, A[0], A[1], "A", pal[0], style=style)
        self._plot_labeled_point(ax, B[0], B[1], "B", pal[1], style=style)
        self._plot_labeled_point(ax, C[0], C[1], "C", pal[2], style=style)

        cx = (A[0] + B[0] + C[0]) / 3
        cy = (A[1] + B[1] + C[1]) / 3
        unk_color = pal[5]
        ax.text(cx, cy, "Area = ?",
                fontsize=style["font_size_base"] + 2,
                color=unk_color, fontweight="bold",
                fontfamily=style["font_family"],
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        q = rng.choice(_PHRASINGS["triangle_area"])
        if area == int(area):
            answer_str = str(int(area))
        else:
            answer_str = str(area)
        q, a = self._maybe_mcq(rng, cfg, q, answer_str, area)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    def _line_equation(self, rng, cfg, level):
        lo, hi = cfg["range"]
        pts = self._distinct_pts(rng, 2, lo=lo, hi=hi)
        if pts is None:
            return None
        A, B = pts
        if A[0] == B[0]:
            return None

        m = (B[1] - A[1]) / (B[0] - A[0])
        b_val = A[1] - m * A[0]
        m = round(m, 2)
        b_val = round(b_val, 2)
        if m == int(m):
            m = int(m)
        if b_val == int(b_val):
            b_val = int(b_val)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        self._setup_axes(ax, [A, B], extra=3, style=style)

        pal = list(style["palette"])
        rng.shuffle(pal)
        self._plot_labeled_point(ax, A[0], A[1], "A", pal[0], style=style)
        self._plot_labeled_point(ax, B[0], B[1], "B", pal[1], style=style)

        x_lo, x_hi = ax.get_xlim()
        slope = (B[1] - A[1]) / (B[0] - A[0])
        intercept = A[1] - slope * A[0]
        ax.plot([x_lo, x_hi],
                [slope * x_lo + intercept, slope * x_hi + intercept],
                color=pal[2], lw=style["line_width"], alpha=0.6, zorder=2)

        unk_color = pal[5]
        mx = (A[0] + B[0]) / 2
        my = (A[1] + B[1]) / 2
        ax.text(mx + 0.5, my - 0.6, "y = mx + b = ?",
                fontsize=style["font_size_base"] + 1,
                color=unk_color, fontweight="bold",
                fontfamily=style["font_family"],
                bbox=dict(boxstyle="round,pad=0.12", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=6)

        q = rng.choice(_PHRASINGS["line_equation"])
        answer_str = f"{m}, {b_val}"
        # line_equation two-number format; keep as text
        return q, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])
