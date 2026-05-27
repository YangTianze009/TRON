"""
Cross-Section QA environment.

2026-05-05 R5 P3: REWRITTEN to align with the Understanding of Solid Figures
benchmark style — given a 3D solid being cut by a plane, identify the
2D cross-section shape via single-letter MCQ.

Pure single-letter MCQ (A through D or A through E). Verifier delegates to
the base class which handles letter extraction (3 wrappers, strict-format).

Per-level design:
  L0/L1 — 4-MCQ. Trivial cuts: cube horizontal -> square, cylinder
    horizontal -> circle, sphere -> circle, pyramid parallel-base -> square.
  L2/L3 — 5-MCQ with "E. No correct answer" (10% trap).
  L4-L6 — 5-MCQ. Adds vertical-through-apex (triangle), oblique cylinder
    (ellipse). 15% trap.
  L7 — 5-MCQ tighter distractors. 25% trap.
  L8/L9 — 5-MCQ tight distractors + 40% trap (off-axis cubes producing
    rectangle / hexagon / trapezoid).

The image shows the 3D solid with a translucent red cutting plane drawn
through it. The MCQ options are 2D cross-section shape names.

Question phrasing matches standard 5th-grade textbook style:
  - "As shown in the diagram, ... is cut by a plane. What is the cross-section?"
  - "What 2D shape is the cross-section ABCD?"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Map (solid, cut_variant) -> cross-section shape name
_CUT_RULES = {
    ("cube", "horizontal"): "square",
    ("cube", "vertical_face"): "square",
    ("cube", "vertical_diagonal"): "rectangle",
    ("cube", "space_diagonal_3pt"): "triangle",
    ("cube", "corner_hexagon"): "hexagon",
    ("cylinder", "horizontal"): "circle",
    ("cylinder", "vertical_through_axis"): "rectangle",
    ("cylinder", "oblique"): "ellipse",
    ("sphere", "horizontal"): "circle",
    ("cone", "horizontal"): "circle",
    ("cone", "vertical_through_apex"): "triangle",
    ("cone", "oblique_off_apex"): "ellipse",
    ("pyramid", "parallel_to_base"): "square",
    ("pyramid", "vertical_through_apex"): "triangle",
    ("pyramid", "off_apex_vertical"): "trapezoid",
    ("triangular_prism", "perpendicular"): "triangle",
    ("triangular_prism", "parallel_to_face"): "rectangle",
    ("triangular_prism", "oblique"): "trapezoid",
}

# All shape options that could ever appear
_ALL_SHAPES = ["circle", "square", "rectangle", "triangle",
               "ellipse", "trapezoid", "hexagon", "pentagon"]


class CrossSectionQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a 3D solid + cutting plane, name the cross-section."""

    ALLOW_ROTATION = False  # orientation-sensitive
    ENV_NAME = "cross_section"

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        # Cut catalogues per level — easier first.
        if level == 0:
            return {
                "cuts": [("cube", "horizontal"),
                         ("cylinder", "horizontal"),
                         ("sphere", "horizontal"),
                         ("pyramid", "parallel_to_base")],
                "use_5_options": False, "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level == 1:
            return {
                "cuts": [("cube", "horizontal"), ("cube", "vertical_face"),
                         ("cylinder", "horizontal"),
                         ("sphere", "horizontal"),
                         ("pyramid", "parallel_to_base")],
                "use_5_options": False, "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level == 2:
            return {
                "cuts": [("cube", "horizontal"), ("cube", "vertical_face"),
                         ("cylinder", "horizontal"),
                         ("cone", "horizontal"),
                         ("pyramid", "parallel_to_base")],
                "use_5_options": True, "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level == 3:
            return {
                "cuts": [("cone", "vertical_through_apex"),
                         ("cube", "horizontal"),
                         ("cylinder", "horizontal"),
                         ("pyramid", "vertical_through_apex"),
                         ("triangular_prism", "perpendicular")],
                "use_5_options": True, "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level == 4:
            return {
                "cuts": [("cylinder", "vertical_through_axis"),
                         ("cone", "vertical_through_apex"),
                         ("cylinder", "oblique"),
                         ("triangular_prism", "perpendicular"),
                         ("pyramid", "vertical_through_apex")],
                "use_5_options": True, "trap_rate": 0.15,
                "tight_distractors": False,
            }
        if level == 5:
            return {
                "cuts": [("cylinder", "oblique"),
                         ("cone", "oblique_off_apex"),
                         ("triangular_prism", "oblique"),
                         ("pyramid", "off_apex_vertical"),
                         ("cube", "vertical_diagonal")],
                "use_5_options": True, "trap_rate": 0.15,
                "tight_distractors": False,
            }
        if level == 6:
            return {
                "cuts": [("cube", "vertical_diagonal"),
                         ("triangular_prism", "oblique"),
                         ("pyramid", "off_apex_vertical"),
                         ("cylinder", "oblique")],
                "use_5_options": True, "trap_rate": 0.20,
                "tight_distractors": True,
            }
        if level == 7:
            return {
                "cuts": [("cube", "vertical_diagonal"),
                         ("cube", "space_diagonal_3pt"),
                         ("triangular_prism", "oblique"),
                         ("pyramid", "off_apex_vertical"),
                         ("cone", "oblique_off_apex")],
                "use_5_options": True, "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9 — hardest cuts + 40% trap
        return {
            "cuts": [("cube", "corner_hexagon"),
                     ("cube", "space_diagonal_3pt"),
                     ("cube", "vertical_diagonal"),
                     ("triangular_prism", "oblique"),
                     ("pyramid", "off_apex_vertical")],
            "use_5_options": True, "trap_rate": 0.40,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
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
        true_cut = rng.choice(cfg["cuts"])
        true_shape = _CUT_RULES[true_cut]

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]

        # Build distractor shape pool — all shape names that aren't true_shape
        distractor_pool = [s for s in _ALL_SHAPES if s != true_shape]
        rng.shuffle(distractor_pool)

        if cfg["tight_distractors"]:
            # Bias toward similar shapes
            similar = {
                "circle": ["ellipse"],
                "ellipse": ["circle"],
                "square": ["rectangle"],
                "rectangle": ["square", "trapezoid"],
                "triangle": ["trapezoid"],
                "trapezoid": ["triangle", "rectangle"],
                "hexagon": ["pentagon", "trapezoid"],
            }
            tight = [s for s in distractor_pool if s in similar.get(true_shape, [])]
            other = [s for s in distractor_pool if s not in tight]
            distractor_pool = tight + other

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False
        n_value = len(opt_letters) - (1 if use_5 else 0)

        if use_trap:
            # All value options are wrong shapes; correct = "No correct answer"
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
            f"{opt_letters[i]}. {options[i].capitalize()}" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Standard textbook stem variants
        solid, cut_kind = true_cut
        solid_names = {
            "cube": "cube", "cylinder": "cylinder", "sphere": "sphere",
            "cone": "cone", "pyramid": "square pyramid",
            "triangular_prism": "triangular prism",
        }
        solid_name = solid_names[solid]

        stems = [
            (
                f"As shown in the diagram, a {solid_name} is being cut by "
                f"the plane (shown in red). What 2D shape is the resulting "
                f"cross-section?"
            ),
            (
                f"As shown in the figure, a {solid_name} is sliced by a "
                f"plane (red). What is the shape of the cross-section?"
            ),
            (
                f"The diagram shows a {solid_name} being cut by the red "
                f"plane. The cross-section produced is which of the "
                f"following shapes?"
            ),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._render_solid_with_cut(solid, cut_kind, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Drawing helpers (3D matplotlib)
    # ------------------------------------------------------------------ #
    def _new_3d_axes(self, sc, bg):
        fig = plt.figure(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(bg)
        return fig, ax

    def _render_solid_with_cut(self, solid: str, cut: str,
                                rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        bg = style["bg_color"]

        if solid == "cube":
            return self._draw_cube(cut, sc, bg, rng, style)
        if solid == "cylinder":
            return self._draw_cylinder(cut, sc, bg, rng, style)
        if solid == "sphere":
            return self._draw_sphere(cut, sc, bg, rng, style)
        if solid == "cone":
            return self._draw_cone(cut, sc, bg, rng, style)
        if solid == "pyramid":
            return self._draw_pyramid(cut, sc, bg, rng, style)
        if solid == "triangular_prism":
            return self._draw_prism(cut, sc, bg, rng, style)
        # Fallback
        fig, ax = self._new_3d_axes(sc, bg)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _common_3d_finish(self, ax, fig, title, style):
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title, fontsize=style["font_size_base"] + 2,
                     fontweight='bold', pad=15)
        ax.view_init(elev=22, azim=42)
        try:
            fig.tight_layout()
        except Exception:
            pass

    def _draw_cube(self, cut, sc, bg, rng, style):
        s = rng.uniform(2.5, 3.5)
        fig, ax = self._new_3d_axes(sc, bg)
        # Edges
        verts = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
        ], dtype=float)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            ax.plot3D(*zip(verts[i], verts[j]),
                      color='black', linewidth=1.4, alpha=0.7)

        h = s / 2
        if cut == "horizontal":
            plane_pts = np.array([[0, 0, h], [s, 0, h], [s, s, h], [0, s, h]])
        elif cut == "vertical_face":
            x0 = s * rng.uniform(0.4, 0.6)
            plane_pts = np.array([[x0, 0, 0], [x0, s, 0], [x0, s, s], [x0, 0, s]])
        elif cut == "vertical_diagonal":
            # Plane along face diagonal -> rectangle
            plane_pts = np.array([[0, 0, 0], [s, s, 0], [s, s, s], [0, 0, s]])
        elif cut == "space_diagonal_3pt":
            plane_pts = np.array([[0, 0, 0], [s, s, 0], [s, 0, s]])
        elif cut == "corner_hexagon":
            plane_pts = np.array([
                [h, 0, 0], [s, h, 0], [s, s, h],
                [h, s, s], [0, h, s], [0, 0, h],
            ])
        else:
            plane_pts = np.array([[0, 0, h], [s, 0, h], [s, s, h], [0, s, h]])

        plane_poly = Poly3DCollection([plane_pts], alpha=0.45,
                                      facecolors=['#e74c3c'],
                                      edgecolors=['darkred'], linewidths=2.0)
        ax.add_collection3d(plane_poly)

        ax.set_xlim([-0.5, s + 0.5])
        ax.set_ylim([-0.5, s + 0.5])
        ax.set_zlim([-0.5, s + 0.5])
        title = rng.choice([f"Cube (side = {s:.1f})", "Cube with cutting plane",
                            "Cube"])
        self._common_3d_finish(ax, fig, title, style)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_cylinder(self, cut, sc, bg, rng, style):
        r = rng.uniform(1.5, 2.0)
        h = rng.uniform(3.0, 4.5)
        fig, ax = self._new_3d_axes(sc, bg)

        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(0, h, 2)
        theta_g, z_g = np.meshgrid(theta, z)
        ax.plot_surface(r * np.cos(theta_g), r * np.sin(theta_g), z_g,
                        alpha=0.18, color='#3498db')
        for zv in [0, h]:
            ax.plot(r * np.cos(theta), r * np.sin(theta), zv,
                    'k-', lw=1.4)

        if cut == "horizontal":
            cut_h = h * 0.5
            xs = np.linspace(-r * 1.3, r * 1.3, 10)
            ys = np.linspace(-r * 1.3, r * 1.3, 10)
            Xs, Ys = np.meshgrid(xs, ys)
            Zs = np.full_like(Xs, cut_h)
        elif cut == "vertical_through_axis":
            ys = np.linspace(-r * 1.3, r * 1.3, 10)
            zs = np.linspace(-0.3, h + 0.3, 10)
            Ys, Zs = np.meshgrid(ys, zs)
            Xs = np.zeros_like(Ys)
        else:  # oblique
            xs = np.linspace(-r * 1.3, r * 1.3, 10)
            ys = np.linspace(-r * 1.3, r * 1.3, 10)
            Xs, Ys = np.meshgrid(xs, ys)
            Zs = h / 2 + Xs * (h / (4 * r))

        ax.plot_surface(Xs, Ys, Zs, alpha=0.4, color='red')
        m = max(r, h)
        ax.set_xlim([-m, m])
        ax.set_ylim([-m, m])
        ax.set_zlim([-0.5, h + 0.5])
        title = rng.choice([f"Cylinder (r = {r:.1f}, h = {h:.1f})",
                            "Cylinder with cutting plane", "Cylinder"])
        self._common_3d_finish(ax, fig, title, style)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_sphere(self, cut, sc, bg, rng, style):
        r = rng.uniform(1.5, 2.0)
        fig, ax = self._new_3d_axes(sc, bg)

        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        xs = r * np.outer(np.cos(u), np.sin(v))
        ys = r * np.outer(np.sin(u), np.sin(v))
        zs = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.18, color='#9b59b6')

        # Cutting plane (any orientation produces a circle for sphere)
        xp = np.linspace(-r * 1.3, r * 1.3, 10)
        yp = np.linspace(-r * 1.3, r * 1.3, 10)
        Xp, Yp = np.meshgrid(xp, yp)
        Zp = np.zeros_like(Xp)
        ax.plot_surface(Xp, Yp, Zp, alpha=0.4, color='red')

        ax.set_xlim([-r * 1.3, r * 1.3])
        ax.set_ylim([-r * 1.3, r * 1.3])
        ax.set_zlim([-r * 1.3, r * 1.3])
        title = rng.choice([f"Sphere (r = {r:.1f})", "Sphere with cutting plane",
                            "Sphere"])
        self._common_3d_finish(ax, fig, title, style)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_cone(self, cut, sc, bg, rng, style):
        r = rng.uniform(1.5, 2.0)
        h = rng.uniform(3.0, 4.5)
        fig, ax = self._new_3d_axes(sc, bg)

        theta = np.linspace(0, 2 * np.pi, 50)
        zl = np.linspace(0, h, 30)
        theta_g, z_g = np.meshgrid(theta, zl)
        r_g = r * (1 - z_g / h)
        ax.plot_surface(r_g * np.cos(theta_g), r_g * np.sin(theta_g), z_g,
                        alpha=0.18, color='#e74c3c')
        ax.plot(r * np.cos(theta), r * np.sin(theta), 0, 'k-', lw=1.4)
        ax.scatter([0], [0], [h], color='black', s=30, zorder=5)

        if cut == "horizontal":
            cut_h = h * 0.5
            xs = np.linspace(-r * 1.3, r * 1.3, 10)
            ys = np.linspace(-r * 1.3, r * 1.3, 10)
            Xs, Ys = np.meshgrid(xs, ys)
            Zs = np.full_like(Xs, cut_h)
        elif cut == "vertical_through_apex":
            ys = np.linspace(-r * 1.3, r * 1.3, 10)
            zs = np.linspace(-0.3, h + 0.3, 10)
            Ys, Zs = np.meshgrid(ys, zs)
            Xs = np.zeros_like(Ys)
        else:  # oblique_off_apex
            xs = np.linspace(-r * 1.3, r * 1.3, 10)
            ys = np.linspace(-r * 1.3, r * 1.3, 10)
            Xs, Ys = np.meshgrid(xs, ys)
            Zs = h * 0.4 + Xs * (h / (5 * r))

        ax.plot_surface(Xs, Ys, Zs, alpha=0.4, color='red')
        m = max(r, h)
        ax.set_xlim([-m, m])
        ax.set_ylim([-m, m])
        ax.set_zlim([-0.5, h + 0.5])
        title = rng.choice([f"Cone (r = {r:.1f}, h = {h:.1f})",
                            "Cone with cutting plane", "Cone"])
        self._common_3d_finish(ax, fig, title, style)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_pyramid(self, cut, sc, bg, rng, style):
        b = rng.uniform(2.5, 3.5)
        h = rng.uniform(3.0, 4.0)
        fig, ax = self._new_3d_axes(sc, bg)
        bv = np.array([[0, 0, 0], [b, 0, 0], [b, b, 0], [0, b, 0]])
        apex = np.array([b / 2, b / 2, h])

        faces = [list(bv)]
        for i in range(4):
            faces.append([bv[i], bv[(i + 1) % 4], apex])
        colors = ['#2ecc71'] + ['#3498db'] * 4
        poly = Poly3DCollection(faces, alpha=0.18, facecolors=colors,
                                edgecolors='black', linewidths=1.4)
        ax.add_collection3d(poly)

        if cut == "parallel_to_base":
            cut_h = h * 0.5
            margin = 0.3
            pts = np.array([
                [-margin, -margin, cut_h], [b + margin, -margin, cut_h],
                [b + margin, b + margin, cut_h], [-margin, b + margin, cut_h]
            ])
        elif cut == "vertical_through_apex":
            pts = np.array([
                [b / 2, -0.3, -0.3], [b / 2, b + 0.3, -0.3],
                [b / 2, b + 0.3, h + 0.3], [b / 2, -0.3, h + 0.3]
            ])
        else:  # off_apex_vertical
            offset = b * 0.3
            pts = np.array([
                [offset, -0.3, -0.3], [offset, b + 0.3, -0.3],
                [offset, b + 0.3, h + 0.3], [offset, -0.3, h + 0.3]
            ])
        plane = Poly3DCollection([pts], alpha=0.4, facecolors=['red'],
                                 edgecolors=['darkred'], linewidths=2)
        ax.add_collection3d(plane)

        m = max(b, h)
        ax.set_xlim([-0.5, m + 0.5])
        ax.set_ylim([-0.5, m + 0.5])
        ax.set_zlim([-0.5, m + 0.5])
        title = rng.choice([f"Square pyramid (base = {b:.1f}, h = {h:.1f})",
                            "Square pyramid with cutting plane", "Square pyramid"])
        self._common_3d_finish(ax, fig, title, style)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_prism(self, cut, sc, bg, rng, style):
        b = rng.uniform(2.0, 3.0)
        h_tri = rng.uniform(2.0, 3.0)
        length = rng.uniform(3.0, 5.0)
        fig, ax = self._new_3d_axes(sc, bg)

        tri_f = np.array([[0, 0, 0], [b, 0, 0], [b / 2, 0, h_tri]])
        tri_b = np.array([[0, length, 0], [b, length, 0], [b / 2, length, h_tri]])
        faces = [
            [tri_f[0], tri_f[1], tri_b[1], tri_b[0]],
            [tri_f[0], tri_f[2], tri_b[2], tri_b[0]],
            [tri_f[1], tri_f[2], tri_b[2], tri_b[1]],
            list(tri_f), list(tri_b),
        ]
        poly = Poly3DCollection(faces, alpha=0.18,
                                facecolors=['#3498db'] * 5,
                                edgecolors='black', linewidths=1.4)
        ax.add_collection3d(poly)

        if cut == "perpendicular":
            mid_y = length / 2
            pts = np.array([
                [-0.4, mid_y, -0.4], [b + 0.4, mid_y, -0.4],
                [b + 0.4, mid_y, h_tri + 0.4], [-0.4, mid_y, h_tri + 0.4]
            ])
        elif cut == "parallel_to_face":
            x0 = b * rng.uniform(0.35, 0.65)
            pts = np.array([
                [x0, -0.3, -0.3], [x0, length + 0.3, -0.3],
                [x0, length + 0.3, h_tri + 0.3], [x0, -0.3, h_tri + 0.3]
            ])
        else:  # oblique
            pts = np.array([
                [0, 0, 0], [b, 0, 0],
                [b, length, h_tri * 0.6], [0, length, h_tri * 0.6]
            ])
        plane = Poly3DCollection([pts], alpha=0.4, facecolors=['red'],
                                 edgecolors=['darkred'], linewidths=2)
        ax.add_collection3d(plane)

        m = max(b, h_tri, length)
        ax.set_xlim([-0.5, m + 0.5])
        ax.set_ylim([-0.5, m + 0.5])
        ax.set_zlim([-0.5, m + 0.5])
        title = rng.choice([f"Triangular prism (base = {b:.1f}, h = {h_tri:.1f}, len = {length:.1f})",
                            "Triangular prism with cutting plane",
                            "Triangular prism"])
        self._common_3d_finish(ax, fig, title, style)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        return super()._check_answer(predicted, ground_truth)


if __name__ == "__main__":
    env = CrossSectionQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
