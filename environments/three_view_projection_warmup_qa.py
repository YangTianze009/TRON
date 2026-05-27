"""
Three View Projection Warmup QA environment.

2026-05-05 R5 P3: REWRITTEN to align with the Understanding of Solid Figures
benchmark style. Image shows a single 3D solid (with optional dimension
labels). Question asks "the front/top/side view of this solid is which of
the following 2D shapes?" Pure single-letter MCQ A-D / A-E with
"No correct answer" trap.

Per-level design:
  L0/L1 — 4-MCQ. Trivial: cube/cuboid/cylinder/cone with one obvious view
    (e.g., "front view of a cube is a ?" -> square).
  L2/L3 — 5-MCQ with "E. No correct answer" (10% trap). Adds prism /
    sphere / pyramid.
  L4-L6 — 5-MCQ. 15% trap. Asks tough views (e.g., side view of cone).
  L7 — 5-MCQ tighter distractors (square vs rectangle when cuboid edges
    differ subtly), 25% trap.
  L8/L9 — 5-MCQ + 40% trap. Adds composite stacked solids (cube on top of
    cuboid, etc.) where the requested view is a non-trivial silhouette.

Question phrasing matches standard textbook style:
  Q17: "As shown in the figure, the height and the base radius of the cone
        are given. Then the front view of the cone is a ( )"
        Options: A. Square; B. Rectangle; C. Equilateral triangle;
                 D. Isosceles triangle; E. No correct answer"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Map (solid, view_kind, side_axis_choice) -> 2D shape produced.
# For symmetric solids, the answer is the same regardless of side choice.
def _view_shape(solid: str, view: str, dims: dict) -> str:
    """Return the 2D shape name of `view` (front/top/side) of `solid`."""
    if solid == "cube":
        return "square"
    if solid == "cuboid":
        # length=l (x), width=w (y), height=h (z)
        l, w, h = dims["l"], dims["w"], dims["h"]
        if view == "top":
            return "square" if abs(l - w) < 1e-6 else "rectangle"
        if view == "front":
            return "square" if abs(l - h) < 1e-6 else "rectangle"
        # side
        return "square" if abs(w - h) < 1e-6 else "rectangle"
    if solid == "cylinder":
        if view == "top":
            return "circle"
        return "rectangle"  # front/side
    if solid == "cone":
        if view == "top":
            return "circle"
        # front/side: isosceles or equilateral triangle
        # Equilateral iff slant = 2r AND height = sqrt(3)*r ⇒ slant^2 = h^2 + r^2 = 4r^2 ⇒ h = sqrt(3)*r
        h = dims["h"]; r = dims["r"]
        if abs(h - math.sqrt(3) * r) < 0.05:
            return "equilateral triangle"
        return "isosceles triangle"
    if solid == "sphere":
        return "circle"
    if solid == "pyramid":  # square pyramid
        if view == "top":
            return "square"  # outline silhouette
        # front / side -> isosceles or equilateral triangle
        b = dims["b"]; h = dims["h"]
        # For square pyramid, slant of view triangle: base = b, height = h
        if abs(h - math.sqrt(3) / 2 * b) < 0.05:
            return "equilateral triangle"
        return "isosceles triangle"
    if solid == "triangular_prism":
        if view == "top":
            return "rectangle"
        if view == "front":
            return "triangle"
        return "rectangle"
    if solid == "cube_on_cuboid":  # composite L8/L9
        if view == "top":
            return "rectangle"
        # front silhouette of cube on top of cuboid is "T-shape" or
        # "L-shape" depending on placement, but we approximate as
        # "pentagon" if cube is centered on top of cuboid.
        return "pentagon"
    return "unknown"


def _all_2d_shape_names() -> List[str]:
    return ["circle", "square", "rectangle", "triangle",
            "isosceles triangle", "equilateral triangle",
            "ellipse", "trapezoid", "pentagon", "hexagon"]


# Compose level configs
def _solid_pool_for_level(level):
    if level <= 1:
        return ["cube", "cuboid", "cylinder"]
    if level <= 3:
        return ["cube", "cuboid", "cylinder", "cone", "sphere"]
    if level <= 6:
        return ["cube", "cuboid", "cylinder", "cone", "sphere",
                "pyramid", "triangular_prism"]
    if level == 7:
        return ["cuboid", "cylinder", "cone", "pyramid",
                "triangular_prism"]
    return ["cuboid", "cylinder", "cone", "pyramid",
            "triangular_prism", "cube_on_cuboid"]


class ThreeViewProjectionWarmupQA(StandaloneVisualEnv):
    """Pick which 2D shape is the front/top/side view of a given 3D solid."""

    ALLOW_ROTATION = False  # orientation-sensitive
    ENV_NAME = "three_view_projection_warmup"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"use_5_options": False, "trap_rate": 0.0,
                    "tight_distractors": False}
        if level <= 3:
            return {"use_5_options": True, "trap_rate": 0.10,
                    "tight_distractors": False}
        if level <= 6:
            return {"use_5_options": True, "trap_rate": 0.15,
                    "tight_distractors": False}
        if level == 7:
            return {"use_5_options": True, "trap_rate": 0.25,
                    "tight_distractors": True}
        return {"use_5_options": True, "trap_rate": 0.40,
                "tight_distractors": True}

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
        solids = _solid_pool_for_level(level)
        solid = rng.choice(solids)
        # Generate dimensions
        if solid == "cube":
            dims = {"s": rng.randint(2, 5)}
        elif solid == "cuboid":
            l = rng.randint(2, 6)
            w = rng.randint(2, 6)
            h = rng.randint(2, 6)
            # Avoid all-equal -> cube
            if l == w == h:
                w = w + 1
            dims = {"l": l, "w": w, "h": h}
        elif solid == "cylinder":
            dims = {"r": rng.randint(2, 4), "h": rng.randint(3, 6)}
        elif solid == "cone":
            r = rng.randint(2, 4)
            # 30% chance of equilateral (h ≈ sqrt(3)*r)
            if rng.random() < 0.3:
                h = round(math.sqrt(3) * r, 1)
            else:
                h = rng.randint(3, 6)
            dims = {"r": r, "h": h}
        elif solid == "sphere":
            dims = {"r": rng.randint(2, 4)}
        elif solid == "pyramid":
            b = rng.randint(2, 4)
            if rng.random() < 0.3:
                h = round(math.sqrt(3) / 2 * b, 1)
            else:
                h = rng.randint(3, 5)
            dims = {"b": b, "h": h}
        elif solid == "triangular_prism":
            dims = {"b": rng.randint(2, 4), "h_tri": rng.randint(2, 4),
                    "length": rng.randint(3, 6)}
        elif solid == "cube_on_cuboid":
            dims = {"l": rng.randint(3, 5), "w": rng.randint(3, 5),
                    "h_base": rng.randint(2, 3),
                    "s_top": rng.randint(1, 2)}
        else:
            return None

        view = rng.choice(["front", "top", "side"])
        true_shape = _view_shape(solid, view, dims)
        if true_shape == "unknown":
            return None

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        # Distractor pool
        all_shapes = _all_2d_shape_names()
        distractor_pool = [s for s in all_shapes if s != true_shape]
        # Drop "isosceles triangle" if true is "equilateral triangle" (and vice versa)
        # actually we want to keep them — they are good tight distractors.
        rng.shuffle(distractor_pool)

        if cfg["tight_distractors"]:
            similar = {
                "circle": ["ellipse"],
                "ellipse": ["circle"],
                "square": ["rectangle"],
                "rectangle": ["square", "trapezoid"],
                "triangle": ["isosceles triangle", "equilateral triangle"],
                "isosceles triangle": ["equilateral triangle", "triangle"],
                "equilateral triangle": ["isosceles triangle", "triangle"],
                "trapezoid": ["pentagon", "rectangle"],
                "pentagon": ["hexagon", "trapezoid"],
            }
            tight = [s for s in distractor_pool if s in similar.get(true_shape, [])]
            other = [s for s in distractor_pool if s not in tight]
            distractor_pool = tight + other

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            chosen = distractor_pool[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distractor_pool[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [true_shape] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_shape)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i].capitalize()}"
            for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Standard textbook phrasing — Q17 style
        solid_label = {
            "cube": "cube", "cuboid": "rectangular cuboid",
            "cylinder": "cylinder", "cone": "cone",
            "sphere": "sphere", "pyramid": "square pyramid",
            "triangular_prism": "triangular prism",
            "cube_on_cuboid": "stacked solid (a small cube placed on top of a rectangular cuboid)",
        }[solid]
        view_phrase = {
            "front": "front view (the view seen from the front)",
            "top": "top view (the view seen from above)",
            "side": "side view (the view seen from the right)",
        }[view]

        stems = [
            (
                f"As shown in the figure, the dimensions of the {solid_label} "
                f"are given. Then the {view_phrase} of this {solid_label} is "
                f"which of the following 2D shapes ( )?"
            ),
            (
                f"As shown in the diagram, observe the {solid_label}. The "
                f"{view_phrase} of this solid is a ( )."
            ),
            (
                f"The image shows a {solid_label}. Looking at it carefully, "
                f"identify the shape of its {view_phrase} ( )."
            ),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._render_solid(solid, dims, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # 3D rendering
    # ------------------------------------------------------------------ #
    def _new_3d_axes(self, sc, bg):
        fig = plt.figure(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(bg)
        return fig, ax

    def _render_solid(self, solid: str, dims: dict,
                      rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        bg = style["bg_color"]

        fig, ax = self._new_3d_axes(sc, bg)
        if solid == "cube":
            s = dims["s"]
            self._draw_cuboid(ax, s, s, s)
            title = f"Cube (side = {s})"
        elif solid == "cuboid":
            l, w, h = dims["l"], dims["w"], dims["h"]
            self._draw_cuboid(ax, l, w, h)
            title = f"Cuboid (l={l}, w={w}, h={h})"
        elif solid == "cylinder":
            r, h = dims["r"], dims["h"]
            self._draw_cylinder(ax, r, h)
            title = f"Cylinder (r={r}, h={h})"
        elif solid == "cone":
            r, h = dims["r"], dims["h"]
            self._draw_cone(ax, r, h)
            title = f"Cone (r={r}, h={h})"
        elif solid == "sphere":
            r = dims["r"]
            self._draw_sphere(ax, r)
            title = f"Sphere (r={r})"
        elif solid == "pyramid":
            b, h = dims["b"], dims["h"]
            self._draw_pyramid(ax, b, h)
            title = f"Square pyramid (base={b}, h={h})"
        elif solid == "triangular_prism":
            b, h_tri, length = dims["b"], dims["h_tri"], dims["length"]
            self._draw_triangular_prism(ax, b, h_tri, length)
            title = f"Triangular prism (b={b}, h={h_tri}, len={length})"
        elif solid == "cube_on_cuboid":
            l, w, h_base, s_top = dims["l"], dims["w"], dims["h_base"], dims["s_top"]
            self._draw_cuboid(ax, l, w, h_base)
            ox = (l - s_top) / 2
            oy = (w - s_top) / 2
            self._draw_cuboid(ax, s_top, s_top, s_top, ox=ox, oy=oy, oz=h_base,
                              face_color="#f4a261")
            title = f"Stacked solid (cuboid l={l},w={w},h={h_base} + cube s={s_top})"
        else:
            return None

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title, fontsize=style["font_size_base"] + 2,
                     fontweight='bold', pad=12)
        # Front-direction arrow guide (which axis is "front")
        ax.text2D(0.04, 0.02, "Front view: from -Y\n"
                              "Top view: from +Z\n"
                              "Side view: from +X",
                  transform=ax.transAxes,
                  fontsize=style["font_size_base"] - 1,
                  family=style["font_family"])

        ax.view_init(elev=22, azim=42)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_cuboid(self, ax, l, w, h, ox=0, oy=0, oz=0,
                      face_color="#3498db"):
        verts = np.array([
            [ox, oy, oz], [ox + l, oy, oz], [ox + l, oy + w, oz], [ox, oy + w, oz],
            [ox, oy, oz + h], [ox + l, oy, oz + h], [ox + l, oy + w, oz + h], [ox, oy + w, oz + h],
        ], dtype=float)
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],  # bottom
            [verts[4], verts[5], verts[6], verts[7]],  # top
            [verts[0], verts[1], verts[5], verts[4]],  # front (-y)
            [verts[3], verts[2], verts[6], verts[7]],  # back  (+y)
            [verts[0], verts[3], verts[7], verts[4]],  # left  (-x)
            [verts[1], verts[2], verts[6], verts[5]],  # right (+x)
        ]
        poly = Poly3DCollection(faces, alpha=0.2, facecolors=[face_color] * 6,
                                edgecolors='black', linewidths=1.4)
        ax.add_collection3d(poly)
        m = max(l, w, h, ox + l, oy + w, oz + h, 1)
        ax.set_xlim(-0.5, m + 0.5)
        ax.set_ylim(-0.5, m + 0.5)
        ax.set_zlim(-0.5, m + 0.5)
        ax.set_box_aspect((1, 1, 1))

    def _draw_cylinder(self, ax, r, h):
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(0, h, 2)
        theta_g, z_g = np.meshgrid(theta, z)
        ax.plot_surface(r * np.cos(theta_g), r * np.sin(theta_g), z_g,
                        alpha=0.25, color='#3498db')
        for zv in [0, h]:
            ax.plot(r * np.cos(theta), r * np.sin(theta), zv,
                    'k-', lw=1.4)
        m = max(r, h)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-0.5, h + 0.5)

    def _draw_cone(self, ax, r, h):
        theta = np.linspace(0, 2 * np.pi, 50)
        zl = np.linspace(0, h, 30)
        theta_g, z_g = np.meshgrid(theta, zl)
        r_g = r * (1 - z_g / h)
        ax.plot_surface(r_g * np.cos(theta_g), r_g * np.sin(theta_g), z_g,
                        alpha=0.25, color='#e74c3c')
        ax.plot(r * np.cos(theta), r * np.sin(theta), 0, 'k-', lw=1.4)
        ax.scatter([0], [0], [h], color='black', s=30)
        m = max(r, h)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-0.5, h + 0.5)

    def _draw_sphere(self, ax, r):
        u = np.linspace(0, 2 * np.pi, 32)
        v = np.linspace(0, np.pi, 16)
        xs = r * np.outer(np.cos(u), np.sin(v))
        ys = r * np.outer(np.sin(u), np.sin(v))
        zs = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.18, color='#9b59b6')
        ax.set_xlim(-r * 1.3, r * 1.3); ax.set_ylim(-r * 1.3, r * 1.3)
        ax.set_zlim(-r * 1.3, r * 1.3)

    def _draw_pyramid(self, ax, b, h):
        bv = np.array([[0, 0, 0], [b, 0, 0], [b, b, 0], [0, b, 0]])
        apex = np.array([b / 2, b / 2, h])
        faces = [list(bv)]
        for i in range(4):
            faces.append([bv[i], bv[(i + 1) % 4], apex])
        poly = Poly3DCollection(faces, alpha=0.22,
                                facecolors=['#2ecc71'] * 5,
                                edgecolors='black', linewidths=1.4)
        ax.add_collection3d(poly)
        m = max(b, h)
        ax.set_xlim(-0.5, m + 0.5); ax.set_ylim(-0.5, m + 0.5)
        ax.set_zlim(-0.5, m + 0.5)

    def _draw_triangular_prism(self, ax, b, h_tri, length):
        tri_f = np.array([[0, 0, 0], [b, 0, 0], [b / 2, 0, h_tri]])
        tri_b = np.array([[0, length, 0], [b, length, 0], [b / 2, length, h_tri]])
        faces = [
            [tri_f[0], tri_f[1], tri_b[1], tri_b[0]],
            [tri_f[0], tri_f[2], tri_b[2], tri_b[0]],
            [tri_f[1], tri_f[2], tri_b[2], tri_b[1]],
            list(tri_f), list(tri_b),
        ]
        poly = Poly3DCollection(faces, alpha=0.22,
                                facecolors=['#f39c12'] * 5,
                                edgecolors='black', linewidths=1.4)
        ax.add_collection3d(poly)
        m = max(b, h_tri, length)
        ax.set_xlim(-0.5, m + 0.5); ax.set_ylim(-0.5, m + 0.5)
        ax.set_zlim(-0.5, m + 0.5)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        return super()._check_answer(predicted, ground_truth)


if __name__ == "__main__":
    env = ThreeViewProjectionWarmupQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
