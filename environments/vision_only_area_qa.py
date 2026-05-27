"""
Vision-Only Area QA environment.

Draws composite shapes with dimensions labeled ON the figure.
A shaded region indicates what to compute. Question only says
"Find the area of the shaded region."

Types: rectangle_minus_circle, triangle_plus_semicircle,
       overlapping_shapes, irregular_polygon.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class VisionOnlyAreaQA(StandaloneVisualEnv):
    ENV_NAME = "vision_only_area"

    PROBLEM_TYPES = [
        "rectangle_minus_circle", "triangle_plus_semicircle",
        "overlapping_shapes", "irregular_polygon",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 2:
            return {"ptypes": ["rectangle_minus_circle", "irregular_polygon"]}
        if level <= 5:
            return {"ptypes": ["rectangle_minus_circle", "triangle_plus_semicircle",
                               "irregular_polygon"]}
        return {"ptypes": self.PROBLEM_TYPES}

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
            result = self._dispatch(ptype)
            if result is not None:
                return result
        return None

    def _dispatch(self, ptype: str):
        if ptype == "rectangle_minus_circle":
            return self._rect_minus_circle()
        elif ptype == "triangle_plus_semicircle":
            return self._tri_plus_semi()
        elif ptype == "overlapping_shapes":
            return self._overlapping()
        elif ptype == "irregular_polygon":
            return self._irregular_polygon()
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _label_dim(self, ax, x1, y1, x2, y2, text, offset=0.4,
                   color="#2e7d32", style=None):
        """Label a dimension along a line segment."""
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        nx = -dy / length * offset
        ny = dx / length * offset
        fs = style.get("font_size_base", 13) if style else 13
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"
        bg = style.get("bg_color", "white") if style else "white"
        ax.text(mx + nx, my + ny, str(text), fontsize=fs + 1,
                ha="center", va="center", color=color, fontweight="bold",
                fontfamily=ff,
                bbox=dict(boxstyle="round,pad=0.12", fc=bg,
                          ec=color, alpha=0.9))

    def _dim_arrows(self, ax, x1, y1, x2, y2, offset=0.3, color="#546e7a",
                    style=None):
        """Draw dimension arrows along a segment (offset from it)."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return
        nx = -dy / length * offset
        ny = dx / length * offset
        lw = style.get("line_width", 1.2) if style else 1.2
        ax.annotate("", xy=(x2 + nx, y2 + ny), xytext=(x1 + nx, y1 + ny),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=lw * 0.6))

    # ------------------------------------------------------------------ #
    # Rectangle minus inscribed circle
    # ------------------------------------------------------------------ #

    def _rect_minus_circle(self):
        rng = self._rng

        # Option: rectangle with inscribed circle or quarter circles at corners
        variant = rng.choice(["inscribed", "quarter_corners"])

        style = self._random_style()
        sc = style["figsize_scale"]
        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fill_alpha = style["geo_fill_alpha"]

        if variant == "inscribed":
            if rng.random() < 0.5:
                s = rng.randint(6, 12)
                w, h = s, s
                r = s / 2.0
            else:
                w = rng.randint(6, 12)
                h = rng.randint(6, 12)
                r = min(w, h) / 2.0

            rect_area = w * h
            circle_area = math.pi * r * r
            shaded = round(rect_area - circle_area, 2)

            fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
            fig.patch.set_facecolor(style["bg_color"])
            ax.set_facecolor(style["bg_color"])
            ax.set_aspect("equal")
            ax.axis("off")

            rect = patches.Rectangle((0, 0), w, h, linewidth=lw + 0.2,
                                     edgecolor=geo_line, facecolor=pal[0],
                                     alpha=fill_alpha + 0.3, zorder=2)
            ax.add_patch(rect)

            circ = plt.Circle((w / 2, h / 2), r, facecolor=style["bg_color"],
                              edgecolor=pal[1], linewidth=lw, zorder=3)
            ax.add_patch(circ)

            dim_color = pal[3]
            self._dim_arrows(ax, 0, -0.8, w, -0.8, offset=0, color=dim_color, style=style)
            self._label_dim(ax, 0, -0.8, w, -0.8, w, offset=0.5, color=pal[2], style=style)
            self._dim_arrows(ax, -0.8, 0, -0.8, h, offset=0, color=dim_color, style=style)
            self._label_dim(ax, -0.8, 0, -0.8, h, h, offset=0.5, color=pal[2], style=style)

            ax.plot([w / 2, w / 2 + r], [h / 2, h / 2], "--",
                    color=pal[1], lw=lw * 0.6, zorder=4)
            ax.text(w / 2 + r / 2, h / 2 + 0.4, f"r = {round(r, 1)}",
                    fontsize=fs, color=pal[1], fontweight="bold",
                    fontfamily=ff, ha="center")

            unk_color = pal[5]
            ax.text(w * 0.15, h * 0.15, "?", fontsize=fs + 6, color=unk_color,
                    fontweight="bold", fontfamily=ff, ha="center", va="center")

            margin = 2
            ax.set_xlim(-margin, w + margin)
            ax.set_ylim(-margin, h + margin)

        else:
            w = rng.randint(8, 14)
            h = rng.randint(8, 14)
            r = round(rng.uniform(1.5, min(w, h) / 2 - 0.5), 1)

            rect_area = w * h
            cut_area = math.pi * r * r
            shaded = round(rect_area - cut_area, 2)

            fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
            fig.patch.set_facecolor(style["bg_color"])
            ax.set_facecolor(style["bg_color"])
            ax.set_aspect("equal")
            ax.axis("off")

            rect = patches.Rectangle((0, 0), w, h, linewidth=lw + 0.2,
                                     edgecolor=geo_line, facecolor=pal[0],
                                     alpha=fill_alpha + 0.3, zorder=2)
            ax.add_patch(rect)

            corners = [(0, 0, 0, 90), (w, 0, 90, 180),
                       (w, h, 180, 270), (0, h, 270, 360)]
            for cx_c, cy_c, t1, t2 in corners:
                wedge = patches.Wedge((cx_c, cy_c), r, t1, t2,
                                      facecolor=style["bg_color"], edgecolor=pal[1],
                                      linewidth=lw * 0.75, zorder=3)
                ax.add_patch(wedge)

            dim_color = pal[3]
            self._dim_arrows(ax, 0, -0.8, w, -0.8, offset=0, style=style)
            self._label_dim(ax, 0, -0.8, w, -0.8, w, offset=0.5, color=pal[2], style=style)
            self._dim_arrows(ax, -0.8, 0, -0.8, h, offset=0, style=style)
            self._label_dim(ax, -0.8, 0, -0.8, h, h, offset=0.5, color=pal[2], style=style)

            ax.plot([0, r], [0, 0], "--", color=pal[1], lw=lw * 0.6, zorder=4)
            ax.text(r / 2, -0.4, f"r = {r}", fontsize=fs - 1, color=pal[1],
                    fontweight="bold", fontfamily=ff, ha="center")

            unk_color = pal[5]
            ax.text(w / 2, h / 2, "?", fontsize=fs + 6, color=unk_color,
                    fontweight="bold", fontfamily=ff, ha="center", va="center")

            margin = 2
            ax.set_xlim(-margin, w + margin)
            ax.set_ylim(-margin, h + margin)

        question = "Find the area of the shaded region. Round to 2 decimal places."
        answer_str = str(round(shaded, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Triangle + semicircle on one side
    # ------------------------------------------------------------------ #

    def _tri_plus_semi(self):
        rng = self._rng

        base = rng.randint(6, 12)
        height = rng.randint(4, 10)
        r = base / 2.0

        tri_area = 0.5 * base * height
        semi_area = 0.5 * math.pi * r * r
        total = round(tri_area + semi_area, 2)

        style = self._random_style()
        sc = style["figsize_scale"]
        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fill_alpha = style["geo_fill_alpha"]

        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        # Triangle with base on x-axis
        verts = [(0, 0), (base, 0), (base / 2, height)]
        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=fill_alpha + 0.3, zorder=2)
        ax.add_patch(tri)

        # Semicircle below the base
        theta = np.linspace(math.pi, 2 * math.pi, 100)
        semi_x = base / 2 + r * np.cos(theta)
        semi_y = r * np.sin(theta)
        semi_verts = list(zip(semi_x, semi_y))
        semi_verts = [(0, 0)] + semi_verts + [(base, 0)]
        semi = plt.Polygon(semi_verts, fill=True, facecolor=pal[0],
                           edgecolor=geo_line, linewidth=lw + 0.2,
                           alpha=fill_alpha + 0.3, zorder=2)
        ax.add_patch(semi)

        known_color = pal[2]
        # Base dimension
        self._dim_arrows(ax, 0, -r - 0.8, base, -r - 0.8, offset=0, style=style)
        self._label_dim(ax, 0, -r - 0.8, base, -r - 0.8, base, offset=0.5,
                        color=known_color, style=style)

        # Height dimension (dashed line)
        ax.plot([base / 2, base / 2], [0, height], "--", color=pal[3],
                lw=lw * 0.5, zorder=1)
        self._label_dim(ax, base / 2, 0, base / 2, height, height,
                        offset=0.6, color=known_color, style=style)

        # Radius on semicircle
        ax.plot([base / 2, base], [0, 0], "--", color=pal[1], lw=lw * 0.6,
                zorder=4)
        ax.text(base * 0.75, 0.35, f"r = {round(r, 1)}", fontsize=fs - 1,
                color=pal[1], fontweight="bold", fontfamily=ff, ha="center")

        # "Area = ?" labels
        unk_color = pal[5]
        ax.text(base / 2, height / 3, "?", fontsize=fs + 4, color=unk_color,
                fontweight="bold", fontfamily=ff, ha="center", va="center")
        ax.text(base / 2, -r / 2, "?", fontsize=fs + 2, color=unk_color,
                fontweight="bold", fontfamily=ff, ha="center", va="center")

        margin = 2
        ax.set_xlim(-margin, base + margin)
        ax.set_ylim(-r - margin, height + margin)

        question = "Find the total area of the shaded region. Round to 2 decimal places."
        answer_str = str(round(total, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Two overlapping rectangles
    # ------------------------------------------------------------------ #

    def _overlapping(self):
        rng = self._rng

        # Two rectangles overlapping; shaded = union or just overlap
        w1 = rng.randint(6, 10)
        h1 = rng.randint(4, 8)
        w2 = rng.randint(5, 9)
        h2 = rng.randint(4, 8)

        # Second rectangle offset
        ox = rng.randint(2, w1 - 1)
        oy = rng.randint(2, h1 - 1)

        # Overlap region
        ol_x1 = max(0, ox)
        ol_y1 = max(0, oy)
        ol_x2 = min(w1, ox + w2)
        ol_y2 = min(h1, oy + h2)

        if ol_x2 <= ol_x1 or ol_y2 <= ol_y1:
            return None

        overlap_w = ol_x2 - ol_x1
        overlap_h = ol_y2 - ol_y1
        overlap_area = overlap_w * overlap_h

        ask = rng.choice(["overlap", "total"])
        if ask == "overlap":
            answer_val = overlap_area
            shade_what = "overlap"
        else:
            answer_val = w1 * h1 + w2 * h2 - overlap_area
            shade_what = "total"

        style = self._random_style()
        sc = style["figsize_scale"]
        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fill_alpha = style["geo_fill_alpha"]

        fig, ax = plt.subplots(1, 1, figsize=(8 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        unk_color = pal[5]
        if shade_what == "overlap":
            r1 = patches.Rectangle((0, 0), w1, h1, linewidth=lw + 0.2,
                                   edgecolor=geo_line, facecolor=pal[0],
                                   alpha=fill_alpha, zorder=2)
            r2 = patches.Rectangle((ox, oy), w2, h2, linewidth=lw + 0.2,
                                   edgecolor=pal[1], facecolor=pal[4],
                                   alpha=fill_alpha, zorder=2)
            overlap_rect = patches.Rectangle(
                (ol_x1, ol_y1), overlap_w, overlap_h,
                facecolor=pal[0], edgecolor=geo_line, linewidth=lw,
                alpha=fill_alpha + 0.4, zorder=3)
            ax.add_patch(r1)
            ax.add_patch(r2)
            ax.add_patch(overlap_rect)

            ax.text((ol_x1 + ol_x2) / 2, (ol_y1 + ol_y2) / 2, "?",
                    fontsize=fs + 6, color=unk_color, fontweight="bold",
                    fontfamily=ff, ha="center", va="center", zorder=4)
        else:
            r1 = patches.Rectangle((0, 0), w1, h1, linewidth=lw + 0.2,
                                   edgecolor=geo_line, facecolor=pal[0],
                                   alpha=fill_alpha + 0.2, zorder=2)
            r2 = patches.Rectangle((ox, oy), w2, h2, linewidth=lw + 0.2,
                                   edgecolor=pal[1], facecolor=pal[0],
                                   alpha=fill_alpha + 0.2, zorder=2)
            ax.add_patch(r1)
            ax.add_patch(r2)

            ax.text(w1 / 2, h1 / 4, "?", fontsize=fs + 4, color=unk_color,
                    fontweight="bold", fontfamily=ff, ha="center", va="center", zorder=4)

        known_color = pal[2]
        # Dimension labels for rect 1
        self._dim_arrows(ax, 0, -0.8, w1, -0.8, offset=0, style=style)
        self._label_dim(ax, 0, -0.8, w1, -0.8, w1, offset=0.5, color=known_color, style=style)
        self._dim_arrows(ax, -0.8, 0, -0.8, h1, offset=0, style=style)
        self._label_dim(ax, -0.8, 0, -0.8, h1, h1, offset=0.5, color=known_color, style=style)

        # Dimension labels for rect 2
        self._dim_arrows(ax, ox, oy + h2 + 0.5, ox + w2, oy + h2 + 0.5,
                         offset=0, color=pal[1], style=style)
        self._label_dim(ax, ox, oy + h2 + 0.5, ox + w2, oy + h2 + 0.5,
                        w2, offset=0.5, color=pal[1], style=style)
        self._dim_arrows(ax, ox + w2 + 0.5, oy, ox + w2 + 0.5, oy + h2,
                         offset=0, color=pal[1], style=style)
        self._label_dim(ax, ox + w2 + 0.5, oy, ox + w2 + 0.5, oy + h2,
                        h2, offset=0.5, color=pal[1], style=style)

        margin = 2.5
        ax.set_xlim(-margin, max(w1, ox + w2) + margin)
        ax.set_ylim(-margin, max(h1, oy + h2) + margin)

        question = "Find the area of the shaded region."
        if answer_val == int(answer_val):
            answer_str = str(int(answer_val))
        else:
            answer_str = str(round(answer_val, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Irregular polygon (L-shape, T-shape, etc.)
    # ------------------------------------------------------------------ #

    def _irregular_polygon(self):
        rng = self._rng
        shape = rng.choice(["L", "T", "plus"])

        if shape == "L":
            # L-shape: big rectangle minus top-right rectangle
            W = rng.randint(8, 14)
            H = rng.randint(8, 14)
            w_cut = rng.randint(3, W - 3)
            h_cut = rng.randint(3, H - 3)

            area = W * H - w_cut * h_cut

            verts = [
                (0, 0), (W, 0), (W, H - h_cut),
                (W - w_cut, H - h_cut), (W - w_cut, H), (0, H),
            ]

            dims = [
                ((0, 0), (W, 0), W),
                ((W, 0), (W, H - h_cut), H - h_cut),
                ((W, H - h_cut), (W - w_cut, H - h_cut), w_cut),
                ((W - w_cut, H - h_cut), (W - w_cut, H), h_cut),
                ((W - w_cut, H), (0, H), W - w_cut),
                ((0, H), (0, 0), H),
            ]

        elif shape == "T":
            # T-shape
            stem_w = rng.randint(3, 6)
            stem_h = rng.randint(4, 8)
            top_w = rng.randint(stem_w + 4, stem_w + 10)
            top_h = rng.randint(3, 5)

            area = stem_w * stem_h + top_w * top_h

            x_off = (top_w - stem_w) / 2
            verts = [
                (x_off, 0), (x_off + stem_w, 0),
                (x_off + stem_w, stem_h), (top_w, stem_h),
                (top_w, stem_h + top_h), (0, stem_h + top_h),
                (0, stem_h), (x_off, stem_h),
            ]

            dims = [
                ((x_off, 0), (x_off + stem_w, 0), stem_w),
                ((x_off + stem_w, 0), (x_off + stem_w, stem_h), stem_h),
                ((0, stem_h), (top_w, stem_h), top_w),
                ((top_w, stem_h), (top_w, stem_h + top_h), top_h),
            ]

        else:  # plus
            arm = rng.randint(2, 4)
            center_w = rng.randint(3, 6)
            center_h = center_w  # square center

            area = center_w * center_h + 4 * (arm * center_w)
            # Actually plus shape: center square + 4 arms
            # Each arm is center_w wide and arm tall/wide
            cw = center_w
            verts = [
                (arm, 0), (arm + cw, 0),
                (arm + cw, arm), (2 * arm + cw, arm),
                (2 * arm + cw, arm + cw), (arm + cw, arm + cw),
                (arm + cw, 2 * arm + cw), (arm, 2 * arm + cw),
                (arm, arm + cw), (0, arm + cw),
                (0, arm), (arm, arm),
            ]
            total_size = 2 * arm + cw
            area = cw * total_size + 2 * arm * cw  # cross

            dims = [
                ((arm, 0), (arm + cw, 0), cw),
                ((0, arm), (2 * arm + cw, arm), 2 * arm + cw),
                ((2 * arm + cw, arm), (2 * arm + cw, arm + cw), cw),
                ((arm, 0), (arm, arm), arm),
            ]

        style = self._random_style()
        sc = style["figsize_scale"]
        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fill_alpha = style["geo_fill_alpha"]

        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        poly = plt.Polygon(verts, fill=True, facecolor=pal[0],
                           edgecolor=geo_line, linewidth=lw + 0.2,
                           alpha=fill_alpha + 0.3, zorder=2)
        ax.add_patch(poly)

        known_color = pal[2]
        # Dimension labels
        for (x1, y1), (x2, y2), val in dims:
            self._dim_arrows(ax, x1, y1, x2, y2, offset=0.5, style=style)
            self._label_dim(ax, x1, y1, x2, y2, val, offset=0.8, color=known_color, style=style)

        # "Area = ?"
        unk_color = pal[5]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        ax.text(cx, cy, "Area = ?", fontsize=fs + 2, color=unk_color,
                fontweight="bold", fontfamily=ff, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#fff9c4",
                          ec=unk_color, alpha=0.9), zorder=4)

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        question = "Find the area of the shaded region."
        if area == int(area):
            answer_str = str(int(area))
        else:
            answer_str = str(round(area, 2))
        return question, answer_str, self.fig_to_pil(fig, dpi=style["dpi"])
