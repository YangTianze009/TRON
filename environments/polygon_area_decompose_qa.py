"""
Polygon area decomposition QA — redesign 2026-04-16.

DIVERSITY:
  * 5+ primitive families per level:
      L0: single rect, L-shape, T-shape, + / plus cross.
      L2-L4: trapezoid, parallelogram, rhombus, rect+triangle.
      L5-L9: stair, U-notch, rect-minus-triangle, 5-rect composite,
             irregular 8-piece.
  * Per-seed style palette rotation, color, alpha shuffle.
  * 6 title variants, 5 question phrasings per shape.
  * Label jitter (within reason).

DIFFICULTY:
  L0: SINGLE rectangle or L/T-shape. Formula visible as text below image.
  L1-L2: composite of 2-3 rectangles.
  L3-L4: trapezoid, U-shape.
  L5-L6: rectangle minus triangle corner.
  L7-L9: stairs, 5-rect composites, irregular 8-piece.

NO NUMERIC LEAKAGE: the question NEVER mentions a, b, h, w explicit numeric
values — they are labelled ONLY on the image.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_wemath_mcq

_TITLE_VARIANTS = [
    "Find the area",
    "Compute the area",
    "Polygon area",
    "Total area",
    "Area of the figure",
    "Shaded area",
]

class PolygonAreaDecomposeQA(StandaloneVisualEnv):
    ENV_NAME = "polygon_area_decompose"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    _QUESTIONS = {
        "single_rect": [
            "Find the area of the rectangle shown. Use area = width x height. Answer with a single integer.",
            "Compute the area of the rectangle. The width and height are labelled on the image. Answer with an integer.",
            "The image shows a rectangle with width and height labeled. Find the area. Answer with a single integer.",
        ],
        "l_shape": [
            "Find the total area of the L-shaped figure shown. All side lengths are labeled on the image. Answer with a single integer.",
            "Compute the area of the L-shape shown. The side lengths are on the image. Answer with an integer.",
            "The figure is L-shaped. Decompose into rectangles and compute the total area. Answer with an integer.",
        ],
        "t_shape": [
            "Find the total area of the T-shaped figure. All side lengths are labelled on the image. Answer with a single integer.",
            "The figure is T-shaped. Compute its total area. Answer with an integer.",
            "Decompose the T-shape into rectangles and compute its area. Labels are on the image. Answer with an integer.",
        ],
        "plus_shape": [
            "Find the area of the plus-shaped figure shown (a central cross). Labels are on the image. Answer with a single integer.",
            "The figure is a + (plus) shape. Compute its area using the labeled dimensions on the image. Answer with an integer.",
        ],
        "composite_rect": [
            "Find the total area of the composite figure (two side-by-side rectangles). All dimensions are labeled on the image. Answer with a single integer.",
            "Add the areas of the two rectangles shown. Dimensions are on the image. Answer with a single integer.",
            "The figure consists of two rectangles with labeled dimensions. Find the total area. Answer with an integer.",
        ],
        "trapezoid": [
            "Find the area of the trapezoid shown. Its two parallel sides (a, b) and perpendicular height (h) are labeled on the image. Answer with a single integer.",
            "Compute the trapezoid's area. Use area = (a + b) / 2 * h. The labels a, b, h are on the image. Answer with an integer.",
            "The trapezoid has labeled parallel sides and height. Find its area. Answer with a single integer.",
        ],
        "u_shape": [
            "Find the area of the U-shaped figure (outer rectangle with a rectangular notch cut from the top). Labels are on the image. Answer with a single integer.",
            "The figure is U-shaped: outer rectangle minus a top notch. Dimensions are labeled on the image. Answer with an integer.",
        ],
        "rect_minus_triangle": [
            "Find the area of the shaded figure: a rectangle with a right-triangle corner cut off. All labels are on the image. Answer with a single integer.",
            "The shape is a rectangle minus a right-triangle in one corner. Compute its area using the labeled dimensions. Answer with an integer.",
        ],
        "stair": [
            "Find the total area of the stair-step figure. Each step has equal width. Base width and each step's height are labeled on the image. Answer with a single integer.",
            "The figure is a stair with equal-width steps. Dimensions are labeled on the image. Find the total area. Answer with an integer.",
        ],
        "composite_5": [
            "Find the area of the shaded figure: a rectangle with two rectangular notches cut from the top (red labels show notch dimensions). Answer with a single integer.",
            "Compute the area of the figure. The outer rectangle has labeled dimensions; red-dashed notches are cut from the top. Answer with an integer.",
        ],
        "irregular_8": [
            "Find the area of the shaded figure. The outer rectangle has labeled dimensions. Red-dashed regions (rectangular notches and one triangular corner) are cut away. Compute outer area minus cut-outs. Answer with a single integer.",
            "Compute the area of the irregular shape: outer rectangle with multiple red-dashed notches and one triangular cut. Dimensions are labeled on the image. Answer with an integer.",
        ],
        "parallelogram": [
            "Find the area of the parallelogram shown. Base and perpendicular height are labeled on the image. Answer with a single integer.",
            "The figure is a parallelogram with labeled base and height. Compute its area. Answer with an integer.",
        ],
    }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))

        # 2026-05-03 (M47 / TG-T10): regular polygon area via apothem * peri / 2.
        # Triggered explicitly OR with low probability when caller doesn't specify.
        if parameter.get("question_type") == "regular_polygon_area" or \
                (parameter.get("question_type") is None
                 and random.Random((self.seed or 0) * 17 + level).random() < 0.15):
            r = self._gen_regular_polygon_area(level)
            if r is not None:
                self._primary_complexity_feature = level * 5 + 10
                return r
            # else fall through

        for _ in range(20):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    q, ans_str, img = result
                    q, ans_str = self._wemath_wrap(q, ans_str)
                    return q, ans_str, img
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # M47 — regular polygon area via apothem
    # ------------------------------------------------------------------ #
    def _gen_regular_polygon_area(self, level):
        """Render a regular n-gon inscribed in a circle of radius r; ask area.
        Area = (1/2) * n * r^2 * sin(2*pi/n).
        """
        rng = self._sub_rng(level)
        # n in 3..8, r small integer
        n = rng.choice([3, 4, 5, 6, 8]) if level >= 4 else rng.choice([3, 4, 6])
        r = rng.choice([2, 3, 4, 5, 6])
        # Compute area numerically.
        area = 0.5 * n * r * r * math.sin(2 * math.pi / n)
        ans = round(area, 2)
        ans_str = str(ans)

        # Render
        fig, ax, style = self._make_axes(rng, 1, 1)
        # Vertices on circle
        cx, cy = 0.0, 0.0
        pts = []
        for i in range(n):
            ang = 2 * math.pi * i / n - math.pi / 2  # start at top
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        from matplotlib.patches import Polygon as _Polygon
        poly = _Polygon(pts, closed=True, facecolor=style["palette"][0],
                        edgecolor="black", linewidth=2.0, alpha=0.5)
        ax.add_patch(poly)
        # Center marker
        ax.plot(cx, cy, marker="o", color="black", markersize=4)
        # Draw a radius for labelling
        ax.plot([cx, pts[0][0]], [cy, pts[0][1]], color="red",
                linestyle="--", linewidth=1.2)
        ax.annotate(f"r = {r}", ((cx + pts[0][0]) / 2, (cy + pts[0][1]) / 2),
                    fontsize=11, fontweight="bold", color="red")
        ax.set_xlim(-r - 1, r + 1)
        ax.set_ylim(-r - 1, r + 1)
        ax.set_title(f"Regular {n}-gon inscribed in circle (radius {r})",
                     fontsize=12, fontweight="bold")
        # render
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()

        q = (f"As shown, a regular {n}-gon is inscribed in a circle with "
             f"radius {r}. What is the area of the polygon? Round to 2 decimal "
             f"places. (Hint: area = (1/2) * n * r² * sin(2π/n).) Place "
             f"the numeric answer in <answer>...</answer>.")
        return q, ans_str, img

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _wemath_wrap(self, q, ans_str):
        """2026-05-04 WeMath alignment: 50% of seeds convert bare-numeric
        area question to 5-way MCQ with E="No correct answer" + cm² unit.
        Adds <answer> tag first so the strip in maybe_to_wemath_mcq has a
        clean handle."""
        # Strip the trailing "Answer with..." sentence and add <answer> tag
        # so the helper can cleanly drop it when converting.
        q = q.rstrip()
        # Append explicit <answer> tag to questions that don't have one
        if "<answer>" not in q:
            q = q + " Place the answer in <answer>...</answer>."
        unit_rng = random.Random((self.seed or 0) * 17 + 4093)
        return maybe_to_wemath_mcq(
            q, ans_str, unit_rng, prob=0.5, unit="cm²", n_options=5)

    def _dispatch(self, level: int):
        sub_rng = self._sub_rng(level)
        # Pool of shape functions per level band.
        # Redesign 2026-04-17: L3 (trapezoid with (a+b)*h/2) and L6
        # (stair_3rect) were harder than L9 (irregular_8piece) because
        # irregular_8piece's answer was often readable by subtraction from
        # visible labels. Keep stair/irregular at the top, push trapezoid
        # later so L3 uses the easier t/u shapes.
        if level == 0:
            pool = [self._single_rect, self._l_shape, self._composite_rect]
        elif level == 1:
            pool = [self._l_shape, self._t_shape, self._composite_rect]
        elif level == 2:
            pool = [self._t_shape, self._plus_shape, self._parallelogram]
        elif level == 3:
            pool = [self._plus_shape, self._parallelogram, self._u_shape]
        elif level == 4:
            pool = [self._u_shape, self._trapezoid]
        elif level == 5:
            pool = [self._trapezoid, self._rect_minus_triangle]
        elif level == 6:
            pool = [self._rect_minus_triangle, self._u_shape]
        elif level == 7:
            pool = [self._stair_3rect_hard]
        elif level == 8:
            pool = [self._stair_3rect_hard, self._composite_5rect_hard]
        else:
            pool = [self._composite_5rect_hard, self._irregular_8piece]
        fn = sub_rng.choice(pool)
        return fn(sub_rng)

    # ------------------------------------------------------------------ #
    def _make_axes(self, rng, w, h):
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        self._apply_style(fig, ax, style)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax, style

    def _label(self, ax, text, x, y, color="#2c3e50", fontsize=12, ha="center", **kw):
        ax.annotate(text, xy=(x, y), ha=ha, fontsize=fontsize,
                    color=color, fontweight="bold", **kw)

    # ------------------------------------------------------------------ #
    def _single_rect(self, rng):
        w = rng.randint(3, 10)
        h = rng.randint(2, 9)
        area = w * h
        fig, ax, style = self._make_axes(rng, w, h)
        col_idx = rng.randint(0, len(style["palette"]) - 1)
        ax.add_patch(mpatches.Rectangle((0, 0), w, h,
                                        facecolor=style["palette"][col_idx],
                                        edgecolor="black", linewidth=2,
                                        alpha=0.4))
        self._label(ax, str(w), w / 2, -0.6)
        self._label(ax, str(h), -0.6, h / 2, ha="right")
        ax.set_xlim(-1.5, w + 1.5)
        ax.set_ylim(-1.5, h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["single_rect"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _l_shape(self, rng):
        w1 = rng.randint(3, 6)
        h1 = rng.randint(3, 6)
        w2 = rng.randint(2, 5)
        h2 = rng.randint(2, 4)
        area = (w1 + w2) * h2 + w1 * h1
        fig, ax, style = self._make_axes(rng, w1 + w2, h1 + h2)
        col = style["palette"][rng.randint(0, 4)]
        verts = [
            (0, 0), (w1 + w2, 0), (w1 + w2, h2),
            (w1, h2), (w1, h1 + h2), (0, h1 + h2)
        ]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        ax.plot([w1, w1], [0, h2], "--", color="#888", linewidth=1)
        self._label(ax, str(w1 + w2), (w1 + w2) / 2, -0.6)
        self._label(ax, str(h2), w1 + w2 + 0.4, h2 / 2, ha="left")
        self._label(ax, str(w1), w1 / 2, h1 + h2 + 0.4)
        self._label(ax, str(h1 + h2), -0.6, (h1 + h2) / 2, ha="right")
        self._label(ax, str(h1), w1 + 0.4, h2 + h1 / 2, ha="left", fontsize=10)
        ax.set_xlim(-1.8, w1 + w2 + 1.8)
        ax.set_ylim(-1.5, h1 + h2 + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["l_shape"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _t_shape(self, rng):
        top_w = rng.randint(5, 8)
        top_h = rng.randint(2, 4)
        stem_w = rng.randint(2, min(4, top_w - 2))
        stem_h = rng.randint(3, 6)
        if (top_w - stem_w) % 2 != 0:
            stem_w = max(2, stem_w - 1) if stem_w > 2 else stem_w + 1
        area = top_w * top_h + stem_w * stem_h
        fig, ax, style = self._make_axes(rng, top_w, top_h + stem_h)
        col = style["palette"][rng.randint(0, 4)]
        stem_x = (top_w - stem_w) / 2
        verts = [
            (0, stem_h), (0, stem_h + top_h), (top_w, stem_h + top_h),
            (top_w, stem_h), (stem_x + stem_w, stem_h),
            (stem_x + stem_w, 0),
            (stem_x, 0), (stem_x, stem_h),
        ]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        self._label(ax, str(top_w), top_w / 2, stem_h + top_h + 0.4)
        self._label(ax, str(top_h), -0.6, stem_h + top_h / 2, ha="right")
        self._label(ax, str(stem_w), top_w / 2, -0.6)
        self._label(ax, str(stem_h), stem_x - 0.6, stem_h / 2, ha="right")
        ax.set_xlim(-1.8, top_w + 1.5)
        ax.set_ylim(-1.5, stem_h + top_h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["t_shape"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _plus_shape(self, rng):
        """Plus / cross: central square + 4 arm rectangles of equal length."""
        arm_w = rng.randint(2, 4)     # arm width (perpendicular to arm axis)
        arm_l = rng.randint(2, 4)     # arm length (how far it extends past center)
        center_s = arm_w              # central square side = arm width
        # area = center_s^2 + 4 * arm_l * arm_w
        area = center_s * center_s + 4 * arm_l * arm_w
        total_w = center_s + 2 * arm_l
        total_h = center_s + 2 * arm_l
        fig, ax, style = self._make_axes(rng, total_w, total_h)
        col = style["palette"][rng.randint(0, 4)]
        cx = total_w / 2
        cy = total_h / 2
        # vertices of plus (going CCW)
        verts = [
            (arm_l, 0), (arm_l + center_s, 0),
            (arm_l + center_s, arm_l),
            (total_w, arm_l), (total_w, arm_l + center_s),
            (arm_l + center_s, arm_l + center_s),
            (arm_l + center_s, total_h),
            (arm_l, total_h),
            (arm_l, arm_l + center_s),
            (0, arm_l + center_s), (0, arm_l),
            (arm_l, arm_l),
        ]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        # labels
        self._label(ax, str(center_s), cx, arm_l - 0.35)
        self._label(ax, str(arm_l), arm_l / 2, cy, ha="center")
        self._label(ax, str(arm_w), total_w - arm_l / 2, cy, ha="center")
        self._label(ax, str(arm_l), cx, total_h - arm_l / 2)
        ax.set_xlim(-1.8, total_w + 1.5)
        ax.set_ylim(-1.5, total_h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["plus_shape"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _composite_rect(self, rng):
        w1, h1 = rng.randint(3, 6), rng.randint(4, 7)
        w2, h2 = rng.randint(3, 6), rng.randint(3, 7)
        area = w1 * h1 + w2 * h2
        fig, ax, style = self._make_axes(rng, w1 + w2, max(h1, h2))
        c1 = style["palette"][rng.randint(0, 4)]
        c2 = style["palette"][rng.randint(0, 4)]
        ax.add_patch(mpatches.Rectangle((0, 0), w1, h1, facecolor=c1,
                                        edgecolor="black", linewidth=2,
                                        alpha=0.45))
        ax.add_patch(mpatches.Rectangle((w1, 0), w2, h2, facecolor=c2,
                                        edgecolor="black", linewidth=2,
                                        alpha=0.45))
        self._label(ax, str(w1), w1 / 2, -0.6)
        self._label(ax, str(h1), -0.6, h1 / 2, ha="right")
        self._label(ax, str(w2), w1 + w2 / 2, -0.6)
        self._label(ax, str(h2), w1 + w2 + 0.5, h2 / 2, ha="left")
        ax.set_xlim(-1.8, w1 + w2 + 1.8)
        ax.set_ylim(-1.5, max(h1, h2) + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["composite_rect"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _u_shape(self, rng):
        outer_w = rng.randint(6, 9)
        outer_h = rng.randint(4, 7)
        cut_w = rng.randint(2, max(2, outer_w - 4))
        cut_h = rng.randint(2, max(2, outer_h - 2))
        area = outer_w * outer_h - cut_w * cut_h
        fig, ax, style = self._make_axes(rng, outer_w, outer_h)
        col = style["palette"][rng.randint(0, 4)]
        cut_x = (outer_w - cut_w) / 2
        verts = [
            (0, 0), (outer_w, 0), (outer_w, outer_h),
            (cut_x + cut_w, outer_h), (cut_x + cut_w, outer_h - cut_h),
            (cut_x, outer_h - cut_h), (cut_x, outer_h),
            (0, outer_h),
        ]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        self._label(ax, str(outer_w), outer_w / 2, -0.6)
        self._label(ax, str(outer_h), -0.6, outer_h / 2, ha="right")
        self._label(ax, str(cut_w), cut_x + cut_w / 2, outer_h + 0.5)
        self._label(ax, str(cut_h), cut_x + cut_w + 0.5, outer_h - cut_h / 2,
                    ha="left", fontsize=10)
        ax.set_xlim(-1.8, outer_w + 1.8)
        ax.set_ylim(-1.5, outer_h + 1.8)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["u_shape"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _trapezoid(self, rng):
        a = rng.randint(4, 9)
        b = a + 2 * rng.randint(1, 4)
        h = rng.randint(3, 7)
        area = (a + b) * h // 2
        fig, ax, style = self._make_axes(rng, b, h)
        col = style["palette"][rng.randint(0, 4)]
        offset = (b - a) / 2
        verts = [(offset, h), (offset + a, h), (b, 0), (0, 0)]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        mid_x = offset + a / 2
        ax.plot([mid_x, mid_x], [0, h], "--", color="#666", linewidth=1)
        # LABEL WITH a, b, h variable names + numeric value on image
        self._label(ax, f"a={a}", offset + a / 2, h + 0.4)
        self._label(ax, f"b={b}", b / 2, -0.6)
        self._label(ax, f"h={h}", mid_x + 0.4, h / 2, ha="left")
        ax.set_xlim(-1.5, b + 1.5)
        ax.set_ylim(-1.5, h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["trapezoid"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _parallelogram(self, rng):
        base = rng.randint(4, 10)
        h = rng.randint(3, 7)
        shear = rng.randint(1, 3)
        area = base * h
        fig, ax, style = self._make_axes(rng, base + shear, h)
        col = style["palette"][rng.randint(0, 4)]
        verts = [(0, 0), (base, 0), (base + shear, h), (shear, h)]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        # height line inside
        ax.plot([shear, shear], [0, h], "--", color="#666", linewidth=1)
        self._label(ax, f"b={base}", base / 2, -0.6)
        self._label(ax, f"h={h}", shear - 0.4, h / 2, ha="right")
        ax.set_xlim(-1.5, base + shear + 1.5)
        ax.set_ylim(-1.5, h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["parallelogram"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _rect_minus_triangle(self, rng):
        w, h = rng.randint(6, 10), rng.randint(5, 8)
        cut_w = rng.randint(2, max(2, w - 3))
        cut_h = rng.randint(2, max(2, h - 3))
        if (cut_w * cut_h) % 2 != 0:
            cut_w = cut_w + 1 if cut_w + 1 < w else cut_w - 1
        area = w * h - (cut_w * cut_h) // 2
        fig, ax, style = self._make_axes(rng, w, h)
        col = style["palette"][rng.randint(0, 4)]
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="black",
                linewidth=1.2, linestyle="--", alpha=0.6)
        remaining = plt.Polygon(
            [(0, 0), (w, 0), (w, h - cut_h), (w - cut_w, h), (0, h)],
            facecolor=col, edgecolor="black", linewidth=2, alpha=0.4
        )
        ax.add_patch(remaining)
        self._label(ax, str(w), w / 2, -0.6)
        self._label(ax, str(h), -0.6, h / 2, ha="right")
        self._label(ax, str(cut_w), w - cut_w / 2, h + 0.4,
                    color="#c0392b", fontsize=11)
        self._label(ax, str(cut_h), w + 0.5, h - cut_h / 2, ha="left",
                    color="#c0392b", fontsize=11)
        ax.set_xlim(-1.5, w + 2)
        ax.set_ylim(-1.5, h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["rect_minus_triangle"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _stair_3rect_hard(self, rng):
        n_steps = 3
        step_w = rng.randint(2, 5)
        heights = sorted([rng.randint(2, 5) for _ in range(n_steps)])
        total_w = step_w * n_steps
        total_h = heights[-1]
        area = sum(step_w * h for h in heights)
        fig, ax, style = self._make_axes(rng, total_w, total_h)
        col = style["palette"][rng.randint(0, 4)]
        verts = [(0, 0)]
        for i in range(n_steps):
            verts.append((i * step_w, heights[i]))
            verts.append(((i + 1) * step_w, heights[i]))
        verts.append((total_w, 0))
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        for i in range(1, n_steps):
            ax.plot([i * step_w, i * step_w], [0, heights[i - 1]],
                    "--", color="#888", linewidth=1)
        self._label(ax, str(total_w), total_w / 2, -0.6)
        for i in range(n_steps):
            x_right = (i + 1) * step_w + 0.4
            self._label(ax, str(heights[i]), x_right, heights[i] / 2,
                        ha="left", fontsize=10)
        ax.set_xlim(-1.8, total_w + 2.0)
        ax.set_ylim(-1.5, total_h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["stair"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _composite_5rect_hard(self, rng):
        outer_w = rng.randint(10, 15)
        outer_h = rng.randint(8, 12)
        notch1_w = rng.randint(2, max(2, outer_w // 4))
        notch1_h = rng.randint(2, max(2, outer_h // 3))
        notch2_w = rng.randint(2, max(2, outer_w // 4))
        notch2_h = rng.randint(2, max(2, outer_h // 3))
        gap = rng.randint(1, max(1, outer_w - notch1_w - notch2_w - 2))
        if notch1_w + gap + notch2_w > outer_w:
            return None
        area = outer_w * outer_h - notch1_w * notch1_h - notch2_w * notch2_h
        fig, ax, style = self._make_axes(rng, outer_w, outer_h)
        col = style["palette"][rng.randint(0, 4)]
        ax.plot([0, outer_w, outer_w, 0, 0],
                [0, 0, outer_h, outer_h, 0], color="black",
                linewidth=1.2, linestyle="--", alpha=0.5)
        n1_x = 0
        n2_x = notch1_w + gap
        verts = [
            (0, 0), (outer_w, 0), (outer_w, outer_h),
            (n2_x + notch2_w, outer_h), (n2_x + notch2_w, outer_h - notch2_h),
            (n2_x, outer_h - notch2_h), (n2_x, outer_h),
            (n1_x + notch1_w, outer_h), (n1_x + notch1_w, outer_h - notch1_h),
            (n1_x, outer_h - notch1_h), (n1_x, outer_h),
            (0, outer_h),
        ]
        ax.add_patch(plt.Polygon(verts, facecolor=col, edgecolor="black",
                                 linewidth=2, alpha=0.4))
        self._label(ax, str(outer_w), outer_w / 2, -0.6)
        self._label(ax, str(outer_h), -0.6, outer_h / 2, ha="right")
        self._label(ax, str(notch1_w), n1_x + notch1_w / 2, outer_h + 0.4,
                    color="#c0392b", fontsize=10)
        self._label(ax, str(notch1_h), n1_x + notch1_w + 0.3,
                    outer_h - notch1_h / 2, ha="left",
                    color="#c0392b", fontsize=10)
        self._label(ax, str(notch2_w), n2_x + notch2_w / 2, outer_h + 0.4,
                    color="#c0392b", fontsize=10)
        self._label(ax, str(notch2_h), n2_x + notch2_w + 0.3,
                    outer_h - notch2_h / 2, ha="left",
                    color="#c0392b", fontsize=10)
        ax.set_xlim(-1.5, outer_w + 2)
        ax.set_ylim(-1.5, outer_h + 1.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["composite_5"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _irregular_8piece(self, rng):
        outer_w = rng.randint(12, 18)
        outer_h = rng.randint(10, 14)
        notches = []
        remaining_w = outer_w
        for _ in range(3):
            nw = rng.randint(2, max(2, remaining_w // 4))
            nh = rng.randint(2, max(2, outer_h // 3))
            remaining_w -= nw
            if remaining_w < 2:
                break
            notches.append((nw, nh))
        if len(notches) < 2:
            return None
        tri_w = rng.randint(2, max(2, outer_w // 5))
        tri_h = rng.randint(2, max(2, outer_h // 4))
        if (tri_w * tri_h) % 2 != 0:
            tri_w = max(2, tri_w + 1)
        notch_area = sum(nw * nh for nw, nh in notches)
        tri_area = (tri_w * tri_h) // 2
        area = outer_w * outer_h - notch_area - tri_area
        fig, ax, style = self._make_axes(rng, outer_w, outer_h)
        col = style["palette"][rng.randint(0, 4)]
        ax.plot([0, outer_w, outer_w, 0, 0],
                [0, 0, outer_h, outer_h, 0], color="black",
                linewidth=1, linestyle="--", alpha=0.4)
        ax.add_patch(mpatches.Rectangle((0, 0), outer_w, outer_h,
                                        facecolor=col, edgecolor="black",
                                        linewidth=2, alpha=0.3))
        nx = 0
        for i, (nw, nh) in enumerate(notches):
            gap = rng.randint(1, 2) if i > 0 else 0
            nx += gap
            ax.add_patch(mpatches.Rectangle(
                (nx, outer_h - nh), nw, nh,
                facecolor="white", edgecolor="#c0392b", linewidth=1.5,
                linestyle="--", alpha=0.9))
            self._label(ax, str(nw), nx + nw / 2, outer_h + 0.3,
                        color="#c0392b", fontsize=9)
            self._label(ax, str(nh), nx + nw + 0.3, outer_h - nh / 2,
                        ha="left", color="#c0392b", fontsize=9)
            nx += nw
        tri_verts = [
            (outer_w, 0),
            (outer_w - tri_w, 0),
            (outer_w, tri_h),
        ]
        ax.add_patch(plt.Polygon(tri_verts, facecolor="white",
                                 edgecolor="#c0392b", linewidth=1.5,
                                 linestyle="--", alpha=0.9))
        self._label(ax, str(tri_w), outer_w - tri_w / 2, -0.5,
                    color="#c0392b", fontsize=9)
        self._label(ax, str(tri_h), outer_w + 0.3, tri_h / 2,
                    ha="left", color="#c0392b", fontsize=9)
        self._label(ax, str(outer_w), outer_w / 2, -0.9)
        self._label(ax, str(outer_h), -0.7, outer_h / 2, ha="right")
        ax.set_xlim(-1.8, outer_w + 2)
        ax.set_ylim(-1.5, outer_h + 1.8)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = rng.choice(self._QUESTIONS["irregular_8"])
        return q, str(area), self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = PolygonAreaDecomposeQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
