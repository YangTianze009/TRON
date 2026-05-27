"""
Vision-Only Length QA environment.

Draws geometric figures with some lengths labeled and one unknown "x".
Uses Pythagorean theorem, similar triangles, perpendicular bisectors,
or medians. Question is only "Find x."

All information is in the diagram -- the model must look at the image.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_QUESTION_TEMPLATES_VOL = [
    "Find the unknown length x shown in the figure. All necessary measurements are labeled in the diagram.",
    "Compute the value of x as marked in the figure. Use only the measurements provided in the image.",
    "What is the value of x in the diagram? Refer to the labeled side lengths.",
    "Solve for x using the labeled diagram. Output a numeric value.",
    "Determine x from the diagram. The needed lengths are labeled in the image.",
]

class VisionOnlyLengthQA(StandaloneVisualEnv):
    ENV_NAME = "vision_only_length"

    PROBLEM_TYPES = ["pythagorean", "similar_ratio", "perpendicular_bisector",
                     "median"]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 2:
            return {"ptypes": ["pythagorean"]}
        if level <= 5:
            return {"ptypes": ["pythagorean", "perpendicular_bisector"]}
        if level <= 7:
            return {"ptypes": ["pythagorean", "similar_ratio", "perpendicular_bisector"]}
        return {"ptypes": self.PROBLEM_TYPES}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._rng = sub_rng  # level-aware downstream
        ptype = parameter.get("problem_type")
        if ptype is None:
            ptype = sub_rng.choice(cfg["ptypes"])

        for _ in range(30):
            result = self._dispatch(ptype)
            if result is not None:
                return result
        return None

    def _dispatch(self, ptype: str):
        if ptype == "pythagorean":
            return self._pythagorean()
        elif ptype == "similar_ratio":
            return self._similar_ratio()
        elif ptype == "perpendicular_bisector":
            return self._perp_bisector()
        elif ptype == "median":
            return self._median()
        return None

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def _label_side(self, ax, x1, y1, x2, y2, text, offset=0.5,
                    color="#2e7d32", fontsize=13, style=None):
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        nx = -dy / length * offset
        ny = dx / length * offset
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"
        bg = style.get("bg_color", "white") if style else "white"
        fs = style.get("font_size_base", fontsize) if style else fontsize
        ax.text(mx + nx, my + ny, str(text), fontsize=fs + 1,
                ha="center", va="center", color=color, fontweight="bold",
                fontfamily=ff,
                bbox=dict(boxstyle="round,pad=0.12", fc=bg,
                          ec=color, alpha=0.9))

    def _label_unknown(self, ax, x1, y1, x2, y2, offset=0.5, style=None):
        unk_color = style["palette"][5] if style else "#d32f2f"
        self._label_side(ax, x1, y1, x2, y2, "x", offset=offset,
                         color=unk_color, style=style)

    def _right_angle_marker(self, ax, vertex, dir1, dir2, size=0.4, style=None):
        """Draw a right-angle square at vertex."""
        vx, vy = vertex
        d1x, d1y = dir1
        d2x, d2y = dir2
        len1 = math.hypot(d1x, d1y)
        len2 = math.hypot(d2x, d2y)
        if len1 < 1e-9 or len2 < 1e-9:
            return
        d1x, d1y = d1x / len1 * size, d1y / len1 * size
        d2x, d2y = d2x / len2 * size, d2y / len2 * size

        lw = style.get("line_width", 1.2) if style else 1.2
        marker_color = style["palette"][3] if style else "#546e7a"
        sq = plt.Polygon([
            (vx, vy),
            (vx + d1x, vy + d1y),
            (vx + d1x + d2x, vy + d1y + d2y),
            (vx + d2x, vy + d2y),
        ], fill=False, edgecolor=marker_color, linewidth=lw * 0.6, zorder=4)
        ax.add_patch(sq)

    def _vertex_label(self, ax, vx, vy, label, centroid, offset=0.65, style=None):
        dx = vx - centroid[0]
        dy = vy - centroid[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            ox, oy = 0, offset
        else:
            ox = dx / dist * offset
            oy = dy / dist * offset
        fs = style.get("font_size_base", 15) if style else 15
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"
        geo_line = style.get("geo_line_color", "#1a237e") if style else "#1a237e"
        ax.text(vx + ox, vy + oy, label, fontsize=fs + 3, fontweight="bold",
                ha="center", va="center", fontfamily=ff, color=geo_line)

    # ------------------------------------------------------------------ #
    # Pythagorean theorem: right triangle
    # ------------------------------------------------------------------ #

    def _pythagorean(self):
        rng = self._rng

        # Generate integer or near-integer Pythagorean triple
        triples = [
            (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
            (6, 8, 10), (9, 12, 15), (9, 40, 41), (12, 16, 20),
            (15, 20, 25), (20, 21, 29),
        ]
        # Also generate scaled versions
        scale = rng.choice([1, 1, 2, 0.5, 1.5])
        base_triple = rng.choice(triples)
        a, b, c = [round(x * scale, 1) for x in base_triple]

        # a, b are legs; c is hypotenuse
        # Ask for one of them
        ask = rng.choice(["hyp", "leg1", "leg2"])

        if ask == "hyp":
            shown = {0: a, 1: b}  # show legs
            answer_val = c
            unknown_idx = 2
        elif ask == "leg1":
            shown = {1: b, 2: c}
            answer_val = a
            unknown_idx = 0
        else:
            shown = {0: a, 2: c}
            answer_val = b
            unknown_idx = 1

        # Place right angle at B (origin), A up, C right
        B = (0.0, 0.0)
        C = (float(a), 0.0)
        A = (0.0, float(b))

        verts = [A, B, C]
        labels = ["A", "B", "C"]
        edges = [(1, 2, a), (0, 1, b), (0, 2, c)]  # BC=a, AB=b, AC=c

        centroid = (sum(v[0] for v in verts) / 3,
                    sum(v[1] for v in verts) / 3)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]

        # Draw triangle
        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=style["geo_fill_alpha"] + 0.3, zorder=2)
        ax.add_patch(tri)

        # Right angle at B
        self._right_angle_marker(ax, B,
                                 (C[0] - B[0], C[1] - B[1]),
                                 (A[0] - B[0], A[1] - B[1]), style=style)

        # Vertex labels
        for i, (vx, vy) in enumerate(verts):
            self._vertex_label(ax, vx, vy, labels[i], centroid, style=style)

        # Side labels
        known_color = pal[2]
        for idx, (i1, i2, val) in enumerate(edges):
            x1, y1 = verts[i1]
            x2, y2 = verts[i2]
            if idx == unknown_idx:
                self._label_unknown(ax, x1, y1, x2, y2, style=style)
            elif idx in shown:
                self._label_side(ax, x1, y1, x2, y2, shown[idx],
                                 color=known_color, style=style)

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        question = self._rng.choice(_QUESTION_TEMPLATES_VOL)
        answer_str = str(round(answer_val, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Similar triangles: find unknown via ratio
    # ------------------------------------------------------------------ #

    def _similar_ratio(self):
        rng = self._rng

        # Two nested similar triangles sharing a vertex
        # Small triangle ABC, big triangle ADE where D on AB, E on AC extended
        # (or a parallel line creates similar triangles)

        # Ratio
        k = round(rng.uniform(1.5, 3.0), 1)

        # Small triangle sides
        s1 = round(rng.uniform(3.0, 6.0), 1)
        s2 = round(rng.uniform(3.0, 6.0), 1)

        # Big triangle corresponding sides
        b1 = round(s1 * k, 1)
        b2 = round(s2 * k, 1)

        # Ask for one big side given small sides and one big side
        ask = rng.choice(["big1", "big2"])
        if ask == "big1":
            answer_val = b1
            shown_big = b2
            shown_big_idx = 1
        else:
            answer_val = b2
            shown_big = b1
            shown_big_idx = 0

        # Geometry: A at top, B bottom-left, C bottom-right (small triangle)
        # D and E are further out on AB and AC extended
        A = (4.0, 8.0)
        angle_left = math.radians(rng.uniform(30, 60))
        angle_right = math.radians(rng.uniform(30, 60))

        # Small triangle
        B = (A[0] - s1 * math.sin(angle_left), A[1] - s1 * math.cos(angle_left))
        C = (A[0] + s2 * math.sin(angle_right), A[1] - s2 * math.cos(angle_right))

        # Big triangle (same angles from A)
        D = (A[0] - b1 * math.sin(angle_left), A[1] - b1 * math.cos(angle_left))
        E = (A[0] + b2 * math.sin(angle_right), A[1] - b2 * math.cos(angle_right))

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]
        fill_alpha = style["geo_fill_alpha"]

        # Draw big triangle (lighter)
        big_tri = plt.Polygon([A, D, E], fill=True, facecolor=pal[4],
                              edgecolor=pal[3], linewidth=lw * 0.75, zorder=1,
                              linestyle="--", alpha=fill_alpha)
        ax.add_patch(big_tri)

        # Draw small triangle (darker)
        small_tri = plt.Polygon([A, B, C], fill=True, facecolor=pal[0],
                                edgecolor=geo_line, linewidth=lw + 0.2,
                                alpha=fill_alpha + 0.3, zorder=2)
        ax.add_patch(small_tri)

        # Parallel line from B to C extended to D-E line
        ax.plot([B[0], C[0]], [B[1], C[1]], color=geo_line, lw=lw, zorder=3)
        ax.plot([D[0], E[0]], [D[1], E[1]], color=pal[3], lw=lw * 0.75,
                linestyle="--", zorder=2)

        mx_bc = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)
        mx_de = ((D[0] + E[0]) / 2, (D[1] + E[1]) / 2)

        all_pts = [A, B, C, D, E]
        centroid = (sum(p[0] for p in all_pts) / 5,
                    sum(p[1] for p in all_pts) / 5)

        # Labels
        for px, py, lbl in [(A[0], A[1], "A"), (B[0], B[1], "B"),
                             (C[0], C[1], "C"), (D[0], D[1], "D"),
                             (E[0], E[1], "E")]:
            self._vertex_label(ax, px, py, lbl, centroid, style=style)

        known_color = pal[2]
        # Side labels: AB, AC (small); AD, AE (big)
        self._label_side(ax, A[0], A[1], B[0], B[1], s1, offset=0.5,
                         color=known_color, style=style)
        self._label_side(ax, A[0], A[1], C[0], C[1], s2, offset=-0.5,
                         color=known_color, style=style)

        if shown_big_idx == 0:
            self._label_side(ax, A[0], A[1], D[0], D[1], shown_big, offset=0.8,
                             color=known_color, style=style)
            self._label_unknown(ax, A[0], A[1], E[0], E[1], offset=-0.8, style=style)
        else:
            self._label_side(ax, A[0], A[1], E[0], E[1], shown_big, offset=-0.8,
                             color=known_color, style=style)
            self._label_unknown(ax, A[0], A[1], D[0], D[1], offset=0.8, style=style)

        # "BC || DE" label
        ax.text(mx_de[0], mx_de[1] - 0.8, "BC \u2225 DE",
                fontsize=style["font_size_base"], color=pal[3], fontweight="bold",
                fontfamily=style["font_family"], ha="center",
                bbox=dict(boxstyle="round,pad=0.12", fc=style["bg_color"],
                          ec=pal[3], alpha=0.9))

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        question = self._rng.choice(_QUESTION_TEMPLATES_VOL)
        answer_str = str(round(answer_val, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Perpendicular bisector: point equidistant from endpoints
    # ------------------------------------------------------------------ #

    def _perp_bisector(self):
        rng = self._rng

        # Triangle with perpendicular bisector from a vertex to the opposite side
        # Creates two right triangles with known sides
        base = round(rng.uniform(6, 12), 1)
        height = round(rng.uniform(4, 9), 1)

        # Foot of perpendicular at distance d from left vertex
        d = round(rng.uniform(base * 0.2, base * 0.8), 1)

        # The two segments of the base
        seg1 = d
        seg2 = round(base - d, 1)

        # Hypotenuses of the two right triangles
        hyp1 = round(math.sqrt(d * d + height * height), 2)
        hyp2 = round(math.sqrt(seg2 * seg2 + height * height), 2)

        # Ask for one of: height, hyp1, hyp2, seg1, seg2
        ask = rng.choice(["height", "hyp1", "hyp2"])

        B = (0.0, 0.0)
        C = (float(base), 0.0)
        A = (float(d), float(height))
        F = (float(d), 0.0)  # foot of perpendicular

        verts = [A, B, C]
        centroid = (sum(v[0] for v in verts) / 3,
                    sum(v[1] for v in verts) / 3)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]

        # Draw triangle
        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=style["geo_fill_alpha"] + 0.3, zorder=2)
        ax.add_patch(tri)

        # Draw height
        height_color = pal[1]
        ax.plot([A[0], F[0]], [A[1], F[1]], "--", color=height_color,
                lw=lw, zorder=3)

        # Right angle at F
        self._right_angle_marker(ax, F,
                                 (C[0] - F[0], 0), (0, A[1] - F[1]), style=style)

        # Labels
        self._vertex_label(ax, A[0], A[1], "A", centroid, style=style)
        self._vertex_label(ax, B[0], B[1], "B", centroid, style=style)
        self._vertex_label(ax, C[0], C[1], "C", centroid, style=style)
        ax.text(F[0] + 0.3, F[1] - 0.5, "F", fontsize=style["font_size_base"] + 1,
                fontweight="bold", fontfamily=style["font_family"],
                color=pal[3], ha="center")

        known_color = pal[2]
        if ask == "height":
            self._label_side(ax, B[0], B[1], F[0], F[1], seg1, offset=-0.4,
                             color=known_color, style=style)
            self._label_side(ax, F[0], F[1], C[0], C[1], seg2, offset=-0.4,
                             color=known_color, style=style)
            self._label_side(ax, A[0], A[1], B[0], B[1], hyp1, offset=0.5,
                             color=known_color, style=style)
            self._label_unknown(ax, A[0], A[1], F[0], F[1], offset=0.5, style=style)
            answer_val = height
        elif ask == "hyp1":
            self._label_side(ax, B[0], B[1], F[0], F[1], seg1, offset=-0.4,
                             color=known_color, style=style)
            self._label_side(ax, A[0], A[1], F[0], F[1], height, offset=0.5,
                             color=height_color, style=style)
            self._label_side(ax, A[0], A[1], C[0], C[1], hyp2, offset=-0.5,
                             color=known_color, style=style)
            self._label_unknown(ax, A[0], A[1], B[0], B[1], offset=0.5, style=style)
            answer_val = hyp1
        else:
            self._label_side(ax, F[0], F[1], C[0], C[1], seg2, offset=-0.4,
                             color=known_color, style=style)
            self._label_side(ax, A[0], A[1], F[0], F[1], height, offset=0.5,
                             color=height_color, style=style)
            self._label_side(ax, A[0], A[1], B[0], B[1], hyp1, offset=0.5,
                             color=known_color, style=style)
            self._label_unknown(ax, A[0], A[1], C[0], C[1], offset=-0.5, style=style)
            answer_val = hyp2

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin - 1, max(ys) + margin)

        question = self._rng.choice(_QUESTION_TEMPLATES_VOL)
        answer_str = str(round(answer_val, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Median: midpoint theorem
    # ------------------------------------------------------------------ #

    def _median(self):
        rng = self._rng

        # Triangle with a median drawn. Use median length formula:
        # m_a = 0.5 * sqrt(2b^2 + 2c^2 - a^2)
        a = round(rng.uniform(5, 12), 1)
        b = round(rng.uniform(5, 12), 1)
        lo = abs(a - b) + 0.5
        hi = a + b - 0.5
        if lo >= hi:
            return None
        c = round(rng.uniform(lo, hi), 1)

        # Median from A to midpoint M of BC
        # a = BC, b = CA, c = AB
        m_a_sq = (2 * b * b + 2 * c * c - a * a) / 4
        if m_a_sq <= 0:
            return None
        m_a = round(math.sqrt(m_a_sq), 2)

        # Place B at origin, C on x-axis
        B = (0.0, 0.0)
        C = (float(a), 0.0)

        # Compute A from sides
        cos_B = (a * a + c * c - b * b) / (2 * a * c)
        if abs(cos_B) > 1:
            return None
        sin_B = math.sqrt(1 - cos_B * cos_B)
        A = (float(c * cos_B), float(c * sin_B))
        if A[1] < 1:
            return None

        M = ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2)

        verts = [A, B, C]
        centroid = (sum(v[0] for v in verts) / 3,
                    sum(v[1] for v in verts) / 3)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]

        # Draw triangle
        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=style["geo_fill_alpha"] + 0.3, zorder=2)
        ax.add_patch(tri)

        # Draw median
        median_color = pal[1]
        ax.plot([A[0], M[0]], [A[1], M[1]], "--", color=median_color,
                lw=lw, zorder=3)

        # Tick marks on BM and MC (equal segments)
        for seg_start, seg_end in [(B, M), (M, C)]:
            mx = (seg_start[0] + seg_end[0]) / 2
            my = (seg_start[1] + seg_end[1]) / 2
            dx = seg_end[0] - seg_start[0]
            dy = seg_end[1] - seg_start[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            nx = -dy / length * 0.2
            ny = dx / length * 0.2
            ax.plot([mx - nx, mx + nx], [my - ny, my + ny],
                    color=geo_line, lw=lw, zorder=4)

        # Labels
        self._vertex_label(ax, A[0], A[1], "A", centroid, style=style)
        self._vertex_label(ax, B[0], B[1], "B", centroid, style=style)
        self._vertex_label(ax, C[0], C[1], "C", centroid, style=style)
        ax.text(M[0] + 0.2, M[1] - 0.5, "M", fontsize=style["font_size_base"] + 1,
                fontweight="bold", fontfamily=style["font_family"],
                color=median_color, ha="center")

        known_color = pal[2]
        # Choose what to show/ask
        ask = rng.choice(["median", "side_a"])

        if ask == "median":
            self._label_side(ax, B[0], B[1], C[0], C[1], a, offset=-0.4,
                             color=known_color, style=style)
            self._label_side(ax, A[0], A[1], C[0], C[1], b, offset=-0.5,
                             color=known_color, style=style)
            self._label_side(ax, A[0], A[1], B[0], B[1], c, offset=0.5,
                             color=known_color, style=style)
            self._label_unknown(ax, A[0], A[1], M[0], M[1], offset=0.6, style=style)
            answer_val = m_a
        else:
            self._label_side(ax, A[0], A[1], M[0], M[1], m_a, offset=0.6,
                             color=median_color, style=style)
            self._label_side(ax, A[0], A[1], C[0], C[1], b, offset=-0.5,
                             color=known_color, style=style)
            self._label_side(ax, A[0], A[1], B[0], B[1], c, offset=0.5,
                             color=known_color, style=style)
            self._label_unknown(ax, B[0], B[1], C[0], C[1], offset=-0.4, style=style)
            answer_val = a

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        question = self._rng.choice(_QUESTION_TEMPLATES_VOL)
        answer_str = str(round(answer_val, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])
