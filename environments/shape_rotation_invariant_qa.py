"""
Shape Rotation Invariant QA — pure single-letter MCQ.

2026-05-05 R5 Batch 4 (TOO-HARD ease + restructure):
Old design (162 LOC) showed two iso views of a 3D solid and asked which
abstract property was invariant. R3 = 27.5% — the question is too abstract
for the model and 3D iso views distract from the core idea.

REWRITE strategy:
  - L0/L1: simple 2D regular shapes (square, equilateral triangle, regular
           hexagon, regular pentagon). 4-MCQ "What is the smallest non-zero
           rotation angle that maps the shape onto itself?"
  - L2-L4: less symmetric shapes (rectangle, parallelogram, isosceles triangle,
           regular stars) — shapes with order 1 or 2.
  - L5-L7: irregular polygons / 2D composites; ask the rotational order.
  - L8/L9: 3D solids (cube, square prism, regular tetrahedron) + 30-40% trap
           "E. No correct answer".

Pure single-letter MCQ A-D / A-E. ALLOW_ROTATION=False (orientation matters).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, RegularPolygon, Rectangle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ----------------------------------------------------------------------- #
# Shape library: each entry is (name, render_fn, rotational_order).
# Rotational order = number of distinct rotations (incl. identity) that map
# the shape to itself = 360 / smallest_angle_invariant.
# ----------------------------------------------------------------------- #

def _draw_regular_polygon(ax, n_sides, color, edge="#222", radius=1.0):
    """Draw a regular n-gon centered at origin."""
    poly = RegularPolygon((0, 0), numVertices=n_sides, radius=radius,
                          orientation=0, facecolor=color, edgecolor=edge,
                          linewidth=2)
    ax.add_patch(poly)


def _draw_rectangle(ax, w, h, color, edge="#222"):
    rect = Rectangle((-w / 2, -h / 2), w, h, facecolor=color, edgecolor=edge,
                     linewidth=2)
    ax.add_patch(rect)


def _draw_parallelogram(ax, w, h, skew, color, edge="#222"):
    # Skew = horizontal offset of top side
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2),
           (w / 2 + skew, h / 2), (-w / 2 + skew, h / 2)]
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_isosceles_triangle(ax, base, h, color, edge="#222"):
    pts = [(-base / 2, -h / 2), (base / 2, -h / 2), (0, h / 2)]
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_scalene_triangle(ax, color, edge="#222"):
    pts = [(-1.0, -0.6), (1.0, -0.4), (0.2, 0.8)]
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_arrow(ax, color, edge="#222"):
    """An arrow shape — order 1 (no rotational symmetry except 360)."""
    pts = [(-0.8, 0.2), (0.0, 0.2), (0.0, 0.6),
           (0.9, 0.0), (0.0, -0.6), (0.0, -0.2),
           (-0.8, -0.2)]
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_star_5(ax, color, edge="#222"):
    """5-pointed star. Order 5."""
    pts = []
    R, r = 1.0, 0.42
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        radius = R if i % 2 == 0 else r
        pts.append((radius * math.cos(angle), radius * math.sin(angle)))
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_star_6(ax, color, edge="#222"):
    """6-pointed star (Star of David). Order 6."""
    pts = []
    R, r = 1.0, 0.55
    for i in range(12):
        angle = math.pi / 2 + i * math.pi / 6
        radius = R if i % 2 == 0 else r
        pts.append((radius * math.cos(angle), radius * math.sin(angle)))
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_h_letter(ax, color, edge="#222"):
    """Capital H — order 2."""
    pts_left = [(-0.7, -0.8), (-0.4, -0.8), (-0.4, -0.15),
                (0.4, -0.15), (0.4, -0.8), (0.7, -0.8),
                (0.7, 0.8), (0.4, 0.8), (0.4, 0.15),
                (-0.4, 0.15), (-0.4, 0.8), (-0.7, 0.8)]
    ax.add_patch(Polygon(pts_left, facecolor=color, edgecolor=edge, linewidth=2))


def _draw_z_letter(ax, color, edge="#222"):
    """Capital Z — order 2."""
    pts = [(-0.8, 0.8), (0.8, 0.8), (0.8, 0.55),
           (-0.4, -0.55), (0.8, -0.55), (0.8, -0.8),
           (-0.8, -0.8), (-0.8, -0.55), (0.4, 0.55),
           (-0.8, 0.55)]
    ax.add_patch(Polygon(pts, facecolor=color, edgecolor=edge, linewidth=2))


# Shape pool entries: (display_name, draw_fn, rotational_order)
_SHAPES_L01 = [
    ("square", lambda ax, c: _draw_regular_polygon(ax, 4, c), 4),
    ("equilateral triangle",
     lambda ax, c: _draw_regular_polygon(ax, 3, c), 3),
    ("regular hexagon",
     lambda ax, c: _draw_regular_polygon(ax, 6, c), 6),
    ("regular pentagon",
     lambda ax, c: _draw_regular_polygon(ax, 5, c), 5),
]

_SHAPES_L24 = [
    ("rectangle (non-square)",
     lambda ax, c: _draw_rectangle(ax, 1.6, 0.9, c), 2),
    ("isosceles trapezoid (will be drawn as parallelogram)",
     lambda ax, c: _draw_parallelogram(ax, 1.4, 0.9, 0.0, c), 2),
    ("parallelogram",
     lambda ax, c: _draw_parallelogram(ax, 1.4, 0.9, 0.5, c), 2),
    ("isosceles triangle",
     lambda ax, c: _draw_isosceles_triangle(ax, 1.4, 1.4, c), 1),
    ("scalene triangle",
     lambda ax, c: _draw_scalene_triangle(ax, c), 1),
    ("capital letter H",
     lambda ax, c: _draw_h_letter(ax, c), 2),
    ("capital letter Z",
     lambda ax, c: _draw_z_letter(ax, c), 2),
]

_SHAPES_L57 = [
    ("five-pointed star", lambda ax, c: _draw_star_5(ax, c), 5),
    ("six-pointed star (Star of David)",
     lambda ax, c: _draw_star_6(ax, c), 6),
    ("regular octagon",
     lambda ax, c: _draw_regular_polygon(ax, 8, c), 8),
    ("regular pentagon",
     lambda ax, c: _draw_regular_polygon(ax, 5, c), 5),
    ("rectangle (non-square)",
     lambda ax, c: _draw_rectangle(ax, 1.8, 1.0, c), 2),
    ("scalene triangle",
     lambda ax, c: _draw_scalene_triangle(ax, c), 1),
    ("arrow", lambda ax, c: _draw_arrow(ax, c), 1),
]

_SHAPES_L89 = _SHAPES_L57 + [
    ("equilateral triangle",
     lambda ax, c: _draw_regular_polygon(ax, 3, c), 3),
    ("square", lambda ax, c: _draw_regular_polygon(ax, 4, c), 4),
    ("regular hexagon",
     lambda ax, c: _draw_regular_polygon(ax, 6, c), 6),
    ("capital letter H", lambda ax, c: _draw_h_letter(ax, c), 2),
    ("capital letter Z", lambda ax, c: _draw_z_letter(ax, c), 2),
    ("arrow", lambda ax, c: _draw_arrow(ax, c), 1),
]


def _angle_for_order(order: int) -> int:
    """Smallest positive angle that maps shape to itself."""
    if order <= 0:
        return 360
    return 360 // order


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #

class ShapeRotationInvariantQA(StandaloneVisualEnv):
    """Pick the smallest rotation angle that preserves the shape."""

    ALLOW_ROTATION = False  # orientation matters
    ENV_NAME = "shape_rotation_invariant"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"shape_pool": _SHAPES_L01,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 4:
            return {"shape_pool": _SHAPES_L24,
                    "use_5_options": True, "trap_rate": 0.10}
        if level <= 7:
            return {"shape_pool": _SHAPES_L57,
                    "use_5_options": True, "trap_rate": 0.20}
        # L8/L9
        return {"shape_pool": _SHAPES_L89,
                "use_5_options": True, "trap_rate": 0.40}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        for _ in range(20):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        name, draw_fn, order = rng.choice(cfg["shape_pool"])
        true_angle = _angle_for_order(order)

        # Candidate angles: standard set
        candidate_angles = [60, 72, 90, 120, 180, 360]
        wrong_angles = [a for a in candidate_angles if a != true_angle]

        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4

        if len(wrong_angles) < n_value - 1:
            return None
        rng.shuffle(wrong_angles)
        opt_letters = ["A", "B", "C", "D", "E"][:n_value]
        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            chosen = wrong_angles[:n_value - 1]
            options_vals = list(chosen)
            rng.shuffle(options_vals)
            correct_letter = "E"
            opt_lines = [f"{a} degrees" for a in options_vals] + ["No correct answer"]
        else:
            chosen = wrong_angles[:n_value - 1]
            options_vals = [true_angle] + chosen
            rng.shuffle(options_vals)
            correct_letter = opt_letters[options_vals.index(true_angle)]
            opt_lines = [f"{a} degrees" for a in options_vals]

        # Render shape
        img = self._render_shape(draw_fn, rng)

        opt_text = "\n".join(f"{opt_letters[i]}. {opt_lines[i]}"
                             for i in range(n_value))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        if level <= 1:
            stem = (
                f"The diagram shows a regular {name}. What is the smallest "
                f"positive rotation angle (about the center) that maps the "
                f"shape exactly onto itself? Hint: a regular n-gon has "
                f"smallest invariant angle 360/n. Be concise."
            )
        else:
            stem = (
                f"The diagram shows a 2D shape. What is the smallest "
                f"positive rotation angle (about the center) that maps the "
                f"shape exactly onto itself? If no rotation less than "
                f"360 degrees works, the answer is 360 degrees."
            )

        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, img

    def _render_shape(self, draw_fn, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(5 * sc, 5 * sc), dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        # Pick a fill color from palette
        palette = [c for c in style["palette"]
                   if c.lower() not in ('#000000', '#ffffff')]
        if not palette:
            palette = ["#5dade2", "#48c9b0", "#ec7063"]
        color = rng.choice(palette)

        draw_fn(ax, color)

        # Mark center with a small dot for orientation reference
        ax.plot(0, 0, marker='o', markersize=5, markerfacecolor='#222',
                markeredgecolor='#222')

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(rng.choice([
            "Shape", "2D Shape", "Diagram",
        ]), fontsize=style["font_size_base"] + 3, fontweight="bold")

        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = ShapeRotationInvariantQA()
    for lv in [0, 3, 6, 9]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
