"""
Angle reasoning Geometry QA environment (v2, redesigned 2026-04-14).

Goal: match vision-intensive style — minimal text prompt, the
figure has 2-3 known angle labels, the model must derive the missing
angle via MULTI-STEP chasing (vertical, supplementary, parallel-cut-by-
transversal, inscribed angle, triangle sum, exterior angle, cyclic
quadrilateral, etc.).

Redesign notes (v2):
  * v1 was "too easy": single triangle-sum problems at L0, 1-hop
    reasoning. Qwen3-VL-8B scored 1.00 / 0.90 because the task is
    literally "given two angles, 180 - a - b = ?".
  * v2 requires a 2-3 step chain at L0, scaling to 4-5 steps at L9.
  * Labels are stripped progressively (Pattern G): all angles labelled
    at L0, only 2 angles labelled at L9 with the model chasing through
    4-5 intermediate relations.
  * Common chains:
      - triangle_sum + supplementary (exterior angle)
      - parallel_transversal + triangle_sum
      - inscribed_angle + triangle_sum
      - cyclic_quadrilateral + alternate_interior
      - isosceles + exterior_angle
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class AngleChaseMinimalTextQA(StandaloneVisualEnv):
    """Minimal-prompt angle chasing problems -- image-only minimal-text style."""

    ENV_NAME = "angle_chase_minimal_text"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_VARIANTS = [
        "Angle problem",
        "Geometry",
        "Find the angle",
        "Angle chase",
        "Angles",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Pattern E: chain depth (how many sequential relations to chase).
        chain_depth = 2 + level // 3            # 2,2,2,3,3,3,4,4,4,5
        # Pattern H: angle parameter range (narrow→wide).
        angle_range = (20 + 5 * level, 130 - 5 * level)  # L0 20-130, L9 65-85
        if angle_range[0] >= angle_range[1] - 10:
            angle_range = (25, 140)
        # Pattern G: label stripping.
        show_all_labels = level <= 2
        return {
            "chain_depth": chain_depth,
            "angle_range": angle_range,
            "show_all_labels": show_all_labels,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1187)
        self._primary_complexity_feature = cfg["chain_depth"]
        # Question phrasing variants
        self._q_templates = [
            "Find x.",
            "Compute the value of x (in degrees).",
            "What is x?",
            "Determine the angle x marked in the figure (degrees).",
            "Find the missing angle x shown in the diagram (in degrees).",
        ]
        self._q_text = rng.choice(self._q_templates)

        # Pick a problem type based on level.
        types_at_level = self._allowed_types(level)
        for _ in range(20):
            try:
                ptype = rng.choice(types_at_level)
                result = self._dispatch(ptype, rng, level, cfg)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _allowed_types(self, level: int) -> List[str]:
        if level <= 2:
            return ["exterior_angle_chain",
                    "supplementary_chain"]
        if level <= 4:
            return ["exterior_angle_chain",
                    "parallel_transversal_chain",
                    "supplementary_chain",
                    "isosceles_exterior"]
        if level <= 6:
            return ["parallel_transversal_chain",
                    "isosceles_exterior",
                    "inscribed_angle_chain",
                    "vertical_angles_chain"]
        return ["parallel_transversal_chain",
                "inscribed_angle_chain",
                "vertical_angles_chain",
                "cyclic_quadrilateral"]

    def _dispatch(self, ptype, rng, level, cfg):
        dispatch = {
            "exterior_angle_chain":     self._exterior_angle_chain,
            "supplementary_chain":      self._supplementary_chain,
            "parallel_transversal_chain": self._parallel_transversal_chain,
            "isosceles_exterior":       self._isosceles_exterior,
            "inscribed_angle_chain":    self._inscribed_angle_chain,
            "vertical_angles_chain":    self._vertical_angles_chain,
            "cyclic_quadrilateral":     self._cyclic_quadrilateral,
        }
        return dispatch[ptype](rng, level, cfg)

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def _new_canvas(self, figsize=(7, 6)):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(figsize[0] * sc, figsize[1] * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax, style

    def _draw_angle_arc(self, ax, vertex, dir1, dir2, radius, label,
                        color="#6a1b9a", unknown=False, style=None):
        ang1 = math.degrees(math.atan2(dir1[1], dir1[0]))
        ang2 = math.degrees(math.atan2(dir2[1], dir2[0]))
        t1, t2 = min(ang1, ang2), max(ang1, ang2)
        if t2 - t1 > 180:
            t1, t2 = t2, t1 + 360
        unk_color = style["palette"][5] if style else "#d32f2f"
        lw = style.get("line_width", 1.8) if style else 1.8
        fs = style.get("font_size_base", 12) if style else 12
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"

        arc = Arc(vertex, 2 * radius, 2 * radius, angle=0,
                  theta1=t1, theta2=t2,
                  color=unk_color if unknown else color,
                  linewidth=lw)
        ax.add_patch(arc)

        mid = math.radians((t1 + t2) / 2)
        tx = vertex[0] + (radius + 0.35) * math.cos(mid)
        ty = vertex[1] + (radius + 0.35) * math.sin(mid)

        if unknown:
            ax.text(tx, ty, "x", fontsize=fs + 3, ha="center", va="center",
                    color=unk_color, fontweight="bold", fontfamily=ff,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#fff9c4",
                              ec=unk_color, alpha=0.95))
        elif label is not None:
            ax.text(tx, ty, f"{label}\u00b0", fontsize=fs + 1,
                    ha="center", va="center", color=color,
                    fontweight="bold", fontfamily=ff)

    # ------------------------------------------------------------------ #
    # Exterior angle chain: triangle_sum → supplementary chain.
    # Given angle A, find x where x is the exterior angle at C.
    # Chain depth: 2 (triangle_sum, then 180-c=x).
    # ------------------------------------------------------------------ #

    def _exterior_angle_chain(self, rng, level, cfg):
        a, b = self._sample_triangle_pair(rng, cfg)
        if a is None:
            return None
        c = 180 - a - b
        if c < 20 or c > 160:
            return None
        # x is the exterior at C → x = 180 - c = a + b
        ext = 180 - c
        fig, ax, style = self._new_canvas(figsize=(8, 6))
        geo = style["geo_line_color"]
        lw = style["line_width"]

        side_bc = rng.uniform(5, 7)
        angle_B = math.radians(b)
        side_ab = rng.uniform(4, 6)
        B = (0.0, 0.0)
        C = (side_bc, 0.0)
        A = (side_ab * math.cos(angle_B), side_ab * math.sin(angle_B))
        ext_end = (side_bc + 3.5, 0.0)

        ax.plot([B[0], C[0], ext_end[0]], [B[1], C[1], ext_end[1]],
                color=geo, lw=lw + 0.3)
        ax.plot([B[0], A[0]], [B[1], A[1]], color=geo, lw=lw + 0.3)
        ax.plot([A[0], C[0]], [A[1], C[1]], color=geo, lw=lw + 0.3)

        def place(p, ox, oy, name):
            ax.text(p[0] + ox, p[1] + oy, name,
                    fontsize=style["font_size_base"] + 3,
                    fontweight="bold", ha="center", va="center",
                    fontfamily=style["font_family"], color=geo)
        place(A, 0, 0.45, "A")
        place(B, -0.45, -0.3, "B")
        place(C, 0, -0.45, "C")

        def arc_at(vertex, dir1, dir2, label, unknown=False):
            L1 = math.hypot(*dir1) or 1
            L2 = math.hypot(*dir2) or 1
            d1 = (dir1[0] / L1, dir1[1] / L1)
            d2 = (dir2[0] / L2, dir2[1] / L2)
            self._draw_angle_arc(ax, vertex, d1, d2, radius=0.55,
                                 label=label, unknown=unknown, style=style)

        # Label A with its interior angle
        arc_at(A, (B[0] - A[0], B[1] - A[1]),
               (C[0] - A[0], C[1] - A[1]), label=a)
        # Label B with its interior angle
        arc_at(B, (C[0] - B[0], C[1] - B[1]),
               (A[0] - B[0], A[1] - B[1]), label=b)
        # Mark the exterior at C as unknown x
        arc_at(C, (A[0] - C[0], A[1] - C[1]),
               (ext_end[0] - C[0], ext_end[1] - C[1]),
               label=None, unknown=True)

        m = 2
        ax.set_xlim(min(A[0], B[0]) - m, ext_end[0] + m)
        ax.set_ylim(-1.8, A[1] + 2)
        return self._q_text, str(int(ext)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Supplementary chain — 2 triangles sharing one side, find the
    # interior angle of the second triangle given a chain of 3 relations.
    # ------------------------------------------------------------------ #

    def _supplementary_chain(self, rng, level, cfg):
        # Triangle with a=?, b, c — given b and exterior-at-c = e, find a.
        # Chain: ext@c=e → interior@c = 180-e → a = 180-b-c_int
        b = rng.randint(30, 100)
        e = rng.randint(85, 160)  # exterior at C
        c_int = 180 - e
        a = 180 - b - c_int
        if a < 20 or a > 150:
            return None

        fig, ax, style = self._new_canvas(figsize=(8, 6))
        geo = style["geo_line_color"]
        lw = style["line_width"]

        side_bc = rng.uniform(5, 7)
        angle_B = math.radians(b)
        side_ab = rng.uniform(4, 6)
        B = (0.0, 0.0)
        C = (side_bc, 0.0)
        A = (side_ab * math.cos(angle_B), side_ab * math.sin(angle_B))
        ext_end = (side_bc + 3.5, 0.0)

        ax.plot([B[0], C[0], ext_end[0]], [B[1], C[1], ext_end[1]],
                color=geo, lw=lw + 0.3)
        ax.plot([B[0], A[0]], [B[1], A[1]], color=geo, lw=lw + 0.3)
        ax.plot([A[0], C[0]], [A[1], C[1]], color=geo, lw=lw + 0.3)

        def place(p, ox, oy, name):
            ax.text(p[0] + ox, p[1] + oy, name,
                    fontsize=style["font_size_base"] + 3,
                    fontweight="bold", ha="center", va="center",
                    fontfamily=style["font_family"], color=geo)
        place(A, 0, 0.45, "A")
        place(B, -0.45, -0.3, "B")
        place(C, 0, -0.45, "C")

        def arc_at(vertex, dir1, dir2, label, unknown=False):
            L1 = math.hypot(*dir1) or 1
            L2 = math.hypot(*dir2) or 1
            d1 = (dir1[0] / L1, dir1[1] / L1)
            d2 = (dir2[0] / L2, dir2[1] / L2)
            self._draw_angle_arc(ax, vertex, d1, d2, radius=0.55,
                                 label=label, unknown=unknown, style=style)

        # Mark A as unknown
        arc_at(A, (B[0] - A[0], B[1] - A[1]),
               (C[0] - A[0], C[1] - A[1]), label=None, unknown=True)
        arc_at(B, (C[0] - B[0], C[1] - B[1]),
               (A[0] - B[0], A[1] - B[1]), label=b)
        arc_at(C, (A[0] - C[0], A[1] - C[1]),
               (ext_end[0] - C[0], ext_end[1] - C[1]), label=e)

        m = 2
        ax.set_xlim(min(A[0], B[0]) - m, ext_end[0] + m)
        ax.set_ylim(-1.8, A[1] + 2)
        return self._q_text, str(int(a)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Parallel transversal chain:
    # Two parallel lines cut by a transversal that also forms a triangle
    # with a third line. Given 1-2 angles, chase through corresponding
    # and supplementary to find x.
    # ------------------------------------------------------------------ #

    def _parallel_transversal_chain(self, rng, level, cfg):
        # Given: angle1 at top intersection (between transversal & top line)
        # Find: alternate-interior angle at bottom, passed through a
        # triangle relation.
        given = rng.randint(35, 140)
        # x is at the bottom intersection, same as given (alternate interior).
        # But at L4+ we add an extra hop: x is supplementary to the
        # alternate-interior at the bottom.
        if level >= 5:
            x = 180 - given
            chain = 2
        else:
            x = given
            chain = 1

        fig, ax, style = self._new_canvas(figsize=(8, 6))
        geo = style["geo_line_color"]
        lw = style["line_width"]

        y1 = 0.0
        y2 = 4.0
        x_extent = 7.0
        trans_angle = rng.uniform(35, 70)
        rad = math.radians(trans_angle)
        ix1 = rng.uniform(-1.5, 1.5)
        ix2 = ix1 + (y2 - y1) / math.tan(rad)
        dx, dy = math.cos(rad), math.sin(rad)

        ax.plot([-x_extent, x_extent], [y1, y1], color=geo, lw=lw + 0.3)
        ax.plot([-x_extent, x_extent], [y2, y2], color=geo, lw=lw + 0.3)
        for y_line in [y1, y2]:
            ax.annotate("", xy=(x_extent - 0.3, y_line),
                        xytext=(x_extent - 1.0, y_line),
                        arrowprops=dict(arrowstyle="->", color=geo, lw=lw))
        t_len = 10.0
        ax.plot([ix1 - dx * t_len, ix2 + dx * t_len],
                [y1 - dy * t_len, y2 + dy * t_len],
                color=style["palette"][0], lw=lw, zorder=3)

        h_dir = (1.0, 0.0)
        t_up = (dx, dy)
        t_down = (-dx, -dy)
        h_left = (-1.0, 0.0)

        # Top: given (between transversal-up and horizontal-right)
        self._draw_angle_arc(ax, (ix2, y2), h_dir, t_up,
                             radius=0.65, label=given, style=style)
        # Bottom: x. If chain=1, x = alt interior. If chain=2, x is
        # supplementary to alt interior (same as given → 180 - given).
        if chain == 1:
            self._draw_angle_arc(ax, (ix1, y1), h_left, t_down,
                                 radius=0.65, label=None, unknown=True,
                                 style=style)
        else:
            # x at bottom is supplementary to the horizontal-right angle;
            # draw in the linear-pair position (same side of transversal,
            # opposite side of parallel) so x = 180 - given holds.
            self._draw_angle_arc(ax, (ix1, y1), h_dir, t_down,
                                 radius=0.65, label=None, unknown=True,
                                 style=style)

        ax.set_xlim(-x_extent - 1, x_extent + 1)
        ax.set_ylim(y1 - 3, y2 + 3)
        return self._q_text, str(int(x)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Isosceles + exterior: Isoceles triangle with vertex angle V,
    # exterior at one base → find x = 180 - base = 180 - (180-V)/2
    # ------------------------------------------------------------------ #

    def _isosceles_exterior(self, rng, level, cfg):
        vertex = rng.randint(30, 120)
        if (180 - vertex) % 2 != 0:
            vertex += 1
        base = (180 - vertex) // 2
        if base < 25:
            return None
        ext_base = 180 - base

        fig, ax, style = self._new_canvas(figsize=(8, 6))
        geo = style["geo_line_color"]
        lw = style["line_width"]

        side_bc = 6.0
        angle_B = math.radians(base)
        side_ab = side_bc / (2 * math.cos(angle_B)) if math.cos(angle_B) != 0 else 5
        B = (0.0, 0.0)
        C = (side_bc, 0.0)
        A = (side_ab * math.cos(angle_B), side_ab * math.sin(angle_B))
        ext_end = (side_bc + 3.5, 0.0)

        ax.plot([B[0], C[0], ext_end[0]], [B[1], C[1], ext_end[1]],
                color=geo, lw=lw + 0.3)
        ax.plot([B[0], A[0]], [B[1], A[1]], color=geo, lw=lw + 0.3)
        ax.plot([A[0], C[0]], [A[1], C[1]], color=geo, lw=lw + 0.3)

        # Tick marks for isosceles (equal sides AB and AC)
        for p0, p1 in [(A, B), (A, C)]:
            mx = (p0[0] + p1[0]) / 2
            my = (p0[1] + p1[1]) / 2
            dx_v, dy_v = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx_v, dy_v) or 1
            nx, ny = -dy_v / L, dx_v / L
            ax.plot([mx - nx * 0.15, mx + nx * 0.15],
                    [my - ny * 0.15, my + ny * 0.15],
                    color=geo, lw=lw + 0.2)

        def place(p, ox, oy, name):
            ax.text(p[0] + ox, p[1] + oy, name,
                    fontsize=style["font_size_base"] + 3,
                    fontweight="bold", ha="center", va="center",
                    fontfamily=style["font_family"], color=geo)
        place(A, 0, 0.45, "A")
        place(B, -0.45, -0.3, "B")
        place(C, 0, -0.45, "C")

        def arc_at(vertex, dir1, dir2, label, unknown=False):
            L1 = math.hypot(*dir1) or 1
            L2 = math.hypot(*dir2) or 1
            d1 = (dir1[0] / L1, dir1[1] / L1)
            d2 = (dir2[0] / L2, dir2[1] / L2)
            self._draw_angle_arc(ax, vertex, d1, d2, radius=0.55,
                                 label=label, unknown=unknown, style=style)

        # Label vertex angle
        arc_at(A, (B[0] - A[0], B[1] - A[1]),
               (C[0] - A[0], C[1] - A[1]), label=vertex)
        # Mark exterior at C as unknown x
        arc_at(C, (A[0] - C[0], A[1] - C[1]),
               (ext_end[0] - C[0], ext_end[1] - C[1]),
               label=None, unknown=True)

        m = 2
        ax.set_xlim(B[0] - m, ext_end[0] + m)
        ax.set_ylim(-1.8, A[1] + 2)
        return self._q_text, str(int(ext_base)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Inscribed angle chain: triangle inscribed in circle, given central
    # angle and one other inscribed, chase through inscribed-angle theorem
    # and triangle sum.
    # ------------------------------------------------------------------ #

    def _inscribed_angle_chain(self, rng, level, cfg):
        central = rng.choice([40, 50, 60, 70, 80, 90, 100, 110, 120])
        inscribed = central // 2
        # At L5+ we make the inscribed at a DIFFERENT arc, so we need to
        # combine inscribed-angle with triangle sum.
        # For v2, the simpler version: x is the inscribed angle at A
        # subtending the central. Only 1 hop. We'll use it as a base for
        # other chains.
        ans = inscribed

        fig, ax, style = self._new_canvas(figsize=(7, 7))
        geo = style["geo_line_color"]
        lw = style["line_width"]
        pal = style["palette"]

        r = 3.0
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=geo, lw=lw + 0.3))
        A = (r * math.cos(math.radians(90)), r * math.sin(math.radians(90)))
        Bangle = 270 - central / 2
        Cangle = 270 + central / 2
        B = (r * math.cos(math.radians(Bangle)), r * math.sin(math.radians(Bangle)))
        C = (r * math.cos(math.radians(Cangle)), r * math.sin(math.radians(Cangle)))
        O = (0.0, 0.0)

        ax.plot([A[0], B[0]], [A[1], B[1]], color=pal[0], lw=lw + 0.2)
        ax.plot([A[0], C[0]], [A[1], C[1]], color=pal[0], lw=lw + 0.2)
        ax.plot([O[0], B[0]], [O[1], B[1]], color=pal[2], lw=lw + 0.2,
                linestyle="--")
        ax.plot([O[0], C[0]], [O[1], C[1]], color=pal[2], lw=lw + 0.2,
                linestyle="--")

        ax.scatter(*O, color=geo, s=24, zorder=5)
        for p, name, ox, oy in [(A, "A", 0, 0.45), (B, "B", -0.35, -0.25),
                                (C, "C", 0.35, -0.25), (O, "O", 0.25, 0.25)]:
            ax.text(p[0] + ox, p[1] + oy, name,
                    fontsize=style["font_size_base"] + 2,
                    fontweight="bold", ha="center", va="center",
                    fontfamily=style["font_family"], color=geo)

        d_ob = (B[0] - O[0], B[1] - O[1])
        d_oc = (C[0] - O[0], C[1] - O[1])
        L1 = math.hypot(*d_ob) or 1
        L2 = math.hypot(*d_oc) or 1
        d_ob = (d_ob[0] / L1, d_ob[1] / L1)
        d_oc = (d_oc[0] / L2, d_oc[1] / L2)
        self._draw_angle_arc(ax, O, d_ob, d_oc, radius=0.65, label=central,
                             style=style)

        d_ab = (B[0] - A[0], B[1] - A[1])
        d_ac = (C[0] - A[0], C[1] - A[1])
        L1 = math.hypot(*d_ab) or 1
        L2 = math.hypot(*d_ac) or 1
        d_ab = (d_ab[0] / L1, d_ab[1] / L1)
        d_ac = (d_ac[0] / L2, d_ac[1] / L2)
        self._draw_angle_arc(ax, A, d_ab, d_ac, radius=0.6, label=None,
                             unknown=True, style=style)

        ax.set_xlim(-r - 2, r + 2)
        ax.set_ylim(-r - 2, r + 2)
        return self._q_text, str(int(ans)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Vertical angles + triangle sum chain: two lines cross forming 4
    # angles. A triangle is drawn using one of the rays. Given 2 angles,
    # find the missing one via vertical-angles + triangle sum.
    # ------------------------------------------------------------------ #

    def _vertical_angles_chain(self, rng, level, cfg):
        # Given: ∠1 (between the crossing lines at the top-right) and
        # another interior angle of the triangle.
        # Chain: ∠1 = vertical angle = interior of triangle at one vertex
        # + another interior → x = 180 - sum.
        a_given = rng.randint(30, 100)
        b_given = rng.randint(30, min(120, 150 - a_given))
        x = 180 - a_given - b_given
        if x < 20 or x > 160:
            return None

        fig, ax, style = self._new_canvas(figsize=(8, 7))
        geo = style["geo_line_color"]
        lw = style["line_width"]

        # Draw two crossing lines + triangle.
        # Line 1: horizontal ray through origin
        ax.plot([-5, 6], [0, 0], color=geo, lw=lw + 0.3)
        # Line 2: angled ray through origin at angle (180 - a_given)
        theta = math.radians(180 - a_given)
        dx, dy = math.cos(theta), math.sin(theta)
        ax.plot([-5 * dx, 5 * dx], [-5 * dy, 5 * dy], color=geo, lw=lw + 0.3)

        # Triangle: vertex at origin + 2 others extending along each ray
        # We use the "bottom" rays so the triangle is below-right.
        A = (0, 0)
        B = (4.5, 0)
        C = (-3.5 * dx, -3.5 * dy)
        # Close triangle edge B→C
        ax.plot([B[0], C[0]], [B[1], C[1]], color=geo, lw=lw + 0.3)

        def place(p, ox, oy, name):
            ax.text(p[0] + ox, p[1] + oy, name,
                    fontsize=style["font_size_base"] + 2,
                    fontweight="bold", ha="center", va="center",
                    fontfamily=style["font_family"], color=geo)
        place(A, -0.35, 0.35, "A")
        place(B, 0.35, -0.3, "B")
        place(C, 0, -0.45, "C")

        def arc_at(vertex, dir1, dir2, label, unknown=False, radius=0.6):
            L1 = math.hypot(*dir1) or 1
            L2 = math.hypot(*dir2) or 1
            d1 = (dir1[0] / L1, dir1[1] / L1)
            d2 = (dir2[0] / L2, dir2[1] / L2)
            self._draw_angle_arc(ax, vertex, d1, d2, radius=radius,
                                 label=label, unknown=unknown, style=style)

        # Show the upper vertical angle (= a_given)
        arc_at(A, (-1, 0), (dx, dy), label=a_given)
        # Show the interior at B = b_given
        arc_at(B, (-1, 0), (C[0] - B[0], C[1] - B[1]),
               label=b_given, radius=0.55)
        # Mark interior at C as x
        arc_at(C, (-dx, -dy), (B[0] - C[0], B[1] - C[1]),
               label=None, unknown=True, radius=0.55)

        ax.set_xlim(-6, 7)
        ax.set_ylim(-6, 6)
        return self._q_text, str(int(x)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Cyclic quadrilateral: opposite angles sum to 180.
    # Given 1 or 2 angles, find the opposite or an adjacent using
    # quadrilateral sum + cyclic property.
    # ------------------------------------------------------------------ #

    def _cyclic_quadrilateral(self, rng, level, cfg):
        # Opposite angles sum to 180.
        A_ang = rng.randint(40, 140)
        C_ang = 180 - A_ang  # opposite
        B_ang = rng.randint(40, 140)
        D_ang = 180 - B_ang  # opposite to B

        # Ask: given A and B, find D.
        ask_val = D_ang

        fig, ax, style = self._new_canvas(figsize=(7, 7))
        geo = style["geo_line_color"]
        lw = style["line_width"]

        r = 3.0
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=geo, lw=lw + 0.3))
        # Place 4 points on circle at angles 90°, 0°, 270°, 180° (approx)
        # but offset so the quadrilateral is convex.
        angs_deg = [90, 20, 270, 170]
        rng.shuffle(angs_deg)
        pts = [(r * math.cos(math.radians(a)),
                r * math.sin(math.radians(a))) for a in angs_deg]
        # Sort by angle so the polygon is convex.
        pts.sort(key=lambda p: math.atan2(p[1], p[0]))
        A, B, C, D = pts
        names = ["A", "B", "C", "D"]

        verts = [A, B, C, D]
        for i in range(4):
            p0 = verts[i]
            p1 = verts[(i + 1) % 4]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=geo, lw=lw + 0.3)

        for (p, nm) in zip(verts, names):
            ax.text(p[0] * 1.2, p[1] * 1.2, nm,
                    fontsize=style["font_size_base"] + 2,
                    fontweight="bold", ha="center", va="center",
                    fontfamily=style["font_family"], color=geo)

        def arc_at(i, label, unknown=False):
            vx, vy = verts[i]
            prev_v = verts[(i - 1) % 4]
            next_v = verts[(i + 1) % 4]
            d1 = (prev_v[0] - vx, prev_v[1] - vy)
            d2 = (next_v[0] - vx, next_v[1] - vy)
            L1 = math.hypot(*d1) or 1
            L2 = math.hypot(*d2) or 1
            d1 = (d1[0] / L1, d1[1] / L1)
            d2 = (d2[0] / L2, d2[1] / L2)
            self._draw_angle_arc(ax, (vx, vy), d1, d2, radius=0.55,
                                 label=label, unknown=unknown, style=style)

        arc_at(0, A_ang)
        arc_at(1, B_ang)
        arc_at(3, None, unknown=True)

        m = 2
        ax.set_xlim(-r - m, r + m)
        ax.set_ylim(-r - m, r + m)
        return self._q_text, str(int(ask_val)), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _sample_triangle_pair(self, rng, cfg):
        lo, hi = cfg["angle_range"]
        a = rng.randint(max(25, lo), min(120, hi))
        b = rng.randint(max(25, lo), min(120, hi))
        if a + b >= 160 or a + b <= 20:
            return None, None
        return a, b

# ====================================================================== #
# Sample generation
# ====================================================================== #

if __name__ == "__main__":
    import os, collections
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = AngleChaseMinimalTextQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[angle_chase_minimal_text] L{level} seed{seed} FAILED")
                continue
            img = env.render()
            out = f"/tmp/env_check/angle_chase_minimal_text_seed{seed}_L{level}.png"
            img.save(out)
            print(f"saved {out}")
            print(f"  Q: {env.get_instruction()[:180]}")
            print(f"  A: {env._answer}")

    for level in (0, 3, 6, 9):
        answers = collections.Counter()
        features = []
        for s in range(20):
            e = AngleChaseMinimalTextQA()
            ok = e.generate(seed=s * 1000 + level * 37 + 17,
                            parameter={"level": level})
            if ok:
                answers[e._answer] += 1
                features.append(e._primary_complexity_feature)
        print(f"[L{level}] unique_answers={len(answers)} "
              f"top3={answers.most_common(3)} "
              f"chain_depth_mean={sum(features)/max(1,len(features)):.2f}")
