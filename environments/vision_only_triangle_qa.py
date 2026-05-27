"""
Vision-Only Triangle QA environment.

ALL geometric information is in the diagram (side lengths, angles, unknowns).
The question text gives NO description of the setup -- only says what to find.
The model must LOOK at the image to understand the problem.

Targets vision-only mode.
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

class VisionOnlyTriangleQA(StandaloneVisualEnv):
    ENV_NAME = "vision_only_triangle"

    PROBLEM_TYPES = ["find_side", "find_angle", "find_area", "find_height"]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 2:
            return {"ptypes": ["find_side", "find_angle"], "difficulty": 1}
        if level <= 4:
            return {"ptypes": ["find_side", "find_angle", "find_area"], "difficulty": 1}
        if level <= 6:
            return {"ptypes": self.PROBLEM_TYPES, "difficulty": 2}
        return {"ptypes": self.PROBLEM_TYPES, "difficulty": 3}

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
        difficulty = parameter.get("difficulty", cfg["difficulty"])

        for _ in range(30):
            result = self._try_generate(ptype, difficulty)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Main generation
    # ------------------------------------------------------------------ #

    def _try_generate(self, ptype: str, difficulty: int):
        rng = self._rng

        # Generate a valid triangle with 3 sides
        a = round(rng.uniform(4.0, 14.0), 1)
        b = round(rng.uniform(4.0, 14.0), 1)
        lo = abs(a - b) + 0.5
        hi = a + b - 0.5
        if lo >= hi:
            return None
        c = round(rng.uniform(lo, hi), 1)

        angles_rad = self._angles_from_sides(a, b, c)
        if angles_rad is None:
            return None
        angles_deg = [round(math.degrees(x), 1) for x in angles_rad]

        if any(ang < 10 for ang in angles_deg):
            return None

        sides = [a, b, c]  # a opp A, b opp B, c opp C
        vertex_labels = ["A", "B", "C"]

        # Compute vertex positions: B at origin, C on x-axis
        Bx, By = 0.0, 0.0
        Cx, Cy = float(a), 0.0
        angle_B = angles_rad[1]
        Ax = float(c) * math.cos(angle_B)
        Ay = float(c) * math.sin(angle_B)
        verts = [(Ax, Ay), (Bx, By), (Cx, Cy)]

        # edge_pairs[i] = vertices of side opposite vertex i
        edge_pairs = [(1, 2), (0, 2), (0, 1)]  # a=BC, b=AC, c=AB

        area = round(0.5 * a * b * math.sin(angles_rad[2]), 2)

        if ptype == "find_side":
            return self._gen_find_side(rng, sides, angles_deg, angles_rad,
                                       verts, vertex_labels, edge_pairs, difficulty)
        elif ptype == "find_angle":
            return self._gen_find_angle(rng, sides, angles_deg, angles_rad,
                                        verts, vertex_labels, edge_pairs, difficulty)
        elif ptype == "find_area":
            return self._gen_find_area(rng, sides, angles_deg, angles_rad,
                                       verts, vertex_labels, edge_pairs, area, difficulty)
        elif ptype == "find_height":
            return self._gen_find_height(rng, sides, angles_deg, angles_rad,
                                         verts, vertex_labels, edge_pairs, area, difficulty)
        return None

    # ------------------------------------------------------------------ #
    # find_side: cosine rule
    # ------------------------------------------------------------------ #

    def _gen_find_side(self, rng, sides, angles_deg, angles_rad,
                       verts, vlabels, edge_pairs, difficulty):
        ask_idx = rng.randint(0, 2)
        answer_val = sides[ask_idx]

        if difficulty <= 1:
            # Show 2 sides + included angle -> direct cosine rule
            other_sides = [i for i in range(3) if i != ask_idx]
            shown_sides = {i: sides[i] for i in other_sides}
            shown_angles = {ask_idx: angles_deg[ask_idx]}
            unknown_side = ask_idx
        elif difficulty == 2:
            # Show 1 side + 2 angles -> use sine rule
            show_side = (ask_idx + 1) % 3
            shown_sides = {show_side: sides[show_side]}
            ang1 = ask_idx
            ang2 = show_side
            shown_angles = {ang1: angles_deg[ang1], ang2: angles_deg[ang2]}
            unknown_side = ask_idx
        else:
            # Show 2 sides + non-included angle -> ambiguous, need cosine rule
            s1 = (ask_idx + 1) % 3
            s2 = (ask_idx + 2) % 3
            shown_sides = {s1: sides[s1], s2: sides[s2]}
            shown_angles = {s1: angles_deg[s1]}
            unknown_side = ask_idx

        question = "Find the value of x."
        answer_str = str(round(answer_val, 2))

        img = self._draw_triangle(
            verts, vlabels, edge_pairs, sides, angles_deg,
            shown_sides=shown_sides, shown_angles=shown_angles,
            unknown_side=unknown_side, unknown_angle=None,
        )
        return question, answer_str, img

    # ------------------------------------------------------------------ #
    # find_angle: sine rule
    # ------------------------------------------------------------------ #

    def _gen_find_angle(self, rng, sides, angles_deg, angles_rad,
                        verts, vlabels, edge_pairs, difficulty):
        ask_idx = rng.randint(0, 2)
        answer_val = angles_deg[ask_idx]

        if difficulty <= 1:
            # Show all 3 sides -> cosine rule for angle
            shown_sides = {i: sides[i] for i in range(3)}
            shown_angles = {}
        elif difficulty == 2:
            # Show 2 sides + 1 known angle -> sine rule
            opp_side = ask_idx
            other = (ask_idx + 1) % 3
            shown_sides = {opp_side: sides[opp_side], other: sides[other]}
            shown_angles = {other: angles_deg[other]}
        else:
            # Show 1 side + 1 angle + need to derive third angle
            other1 = (ask_idx + 1) % 3
            other2 = (ask_idx + 2) % 3
            shown_sides = {other1: sides[other1]}
            shown_angles = {other1: angles_deg[other1], other2: angles_deg[other2]}

        question = "Find angle x (in degrees)."
        answer_str = str(round(answer_val, 1))

        img = self._draw_triangle(
            verts, vlabels, edge_pairs, sides, angles_deg,
            shown_sides=shown_sides, shown_angles=shown_angles,
            unknown_side=None, unknown_angle=ask_idx,
        )
        return question, answer_str, img

    # ------------------------------------------------------------------ #
    # find_area: 1/2 ab sin C
    # ------------------------------------------------------------------ #

    def _gen_find_area(self, rng, sides, angles_deg, angles_rad,
                       verts, vlabels, edge_pairs, area, difficulty):
        if difficulty <= 1:
            # Show 2 sides + included angle
            pick = rng.randint(0, 2)
            s1 = (pick + 1) % 3
            s2 = (pick + 2) % 3
            shown_sides = {s1: sides[s1], s2: sides[s2]}
            shown_angles = {pick: angles_deg[pick]}
        elif difficulty == 2:
            # Show base + height approach: show one side + angle + another side
            shown_sides = {0: sides[0], 1: sides[1]}
            shown_angles = {2: angles_deg[2]}
        else:
            # Show all 3 sides (Heron's formula)
            shown_sides = {i: sides[i] for i in range(3)}
            shown_angles = {}

        question = "Find the area of the triangle."
        answer_str = str(round(area, 2))

        img = self._draw_triangle(
            verts, vlabels, edge_pairs, sides, angles_deg,
            shown_sides=shown_sides, shown_angles=shown_angles,
            unknown_side=None, unknown_angle=None,
            show_area_question=True,
        )
        return question, answer_str, img

    # ------------------------------------------------------------------ #
    # find_height
    # ------------------------------------------------------------------ #

    def _gen_find_height(self, rng, sides, angles_deg, angles_rad,
                         verts, vlabels, edge_pairs, area, difficulty):
        # Height from vertex to opposite side
        base_idx = rng.randint(0, 2)
        base_len = sides[base_idx]
        height = round(2 * area / base_len, 2)

        if difficulty <= 1:
            # Show base + area (trivial: h = 2A/b)
            shown_sides = {base_idx: sides[base_idx]}
            shown_angles = {}
            # We'll note area on the diagram
        else:
            # Show base + one adjacent side + included angle
            adj = (base_idx + 1) % 3
            shown_sides = {base_idx: sides[base_idx], adj: sides[adj]}
            ang_between = (3 - base_idx - adj)  # the third vertex
            # Actually, compute which angle is between the two shown sides
            # Side base_idx is opposite vertex base_idx, side adj is opposite vertex adj
            # The common vertex is the one NOT indexed by either opposite vertex
            common = 3 - base_idx - adj
            shown_angles = {common: angles_deg[common]}

        question = "Find the value of x."
        answer_str = str(round(height, 2))

        img = self._draw_triangle_with_height(
            verts, vlabels, edge_pairs, sides, angles_deg,
            shown_sides=shown_sides, shown_angles=shown_angles,
            base_idx=base_idx, height=height,
        )
        return question, answer_str, img

    # ------------------------------------------------------------------ #
    # Drawing: standard triangle
    # ------------------------------------------------------------------ #

    def _draw_triangle(self, verts, vlabels, edge_pairs, sides, angles_deg,
                       shown_sides, shown_angles, unknown_side, unknown_angle,
                       show_area_question=False):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        pal = style["palette"]
        fill_alpha = style["geo_fill_alpha"]

        # Draw filled triangle
        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=fill_alpha + 0.3)
        ax.add_patch(tri)

        centroid = tuple(sum(v[i] for v in verts) / 3 for i in range(2))

        # Vertex labels
        for i, (vx, vy) in enumerate(verts):
            dx = vx - centroid[0]
            dy = vy - centroid[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                ox, oy = 0, 0.5
            else:
                ox = dx / dist * 0.7
                oy = dy / dist * 0.7
            ax.text(vx + ox, vy + oy, vlabels[i],
                    fontsize=fs + 4, fontweight="bold", ha="center", va="center",
                    fontfamily=ff, color=geo_line)

        # Side labels
        known_color = pal[2]
        unknown_color = pal[5]
        for si in range(3):
            i1, i2 = edge_pairs[si]
            mx = (verts[i1][0] + verts[i2][0]) / 2
            my = (verts[i1][1] + verts[i2][1]) / 2
            ex = verts[i2][0] - verts[i1][0]
            ey = verts[i2][1] - verts[i1][1]
            elen = math.hypot(ex, ey)
            if elen < 1e-9:
                continue
            nx, ny = -ey / elen, ex / elen
            to_cent = (centroid[0] - mx, centroid[1] - my)
            if nx * to_cent[0] + ny * to_cent[1] > 0:
                nx, ny = -nx, -ny
            offset = 0.5

            if si in shown_sides:
                ax.text(mx + nx * offset, my + ny * offset,
                        f"{shown_sides[si]}",
                        fontsize=fs + 2, ha="center", va="center",
                        fontfamily=ff,
                        color=known_color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc=style["bg_color"],
                                  ec=known_color, alpha=0.9))
            elif si == unknown_side:
                ax.text(mx + nx * offset, my + ny * offset, "x",
                        fontsize=fs + 4, ha="center", va="center",
                        fontfamily=ff,
                        color=unknown_color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#fff9c4",
                                  ec=unknown_color, alpha=0.95))

        # Angle labels
        arc_color = pal[3]
        for ai in range(3):
            if ai in shown_angles or ai == unknown_angle:
                vx, vy = verts[ai]
                i_prev = (ai - 1) % 3
                i_next = (ai + 1) % 3
                dx1 = verts[i_prev][0] - vx
                dy1 = verts[i_prev][1] - vy
                dx2 = verts[i_next][0] - vx
                dy2 = verts[i_next][1] - vy
                ang1 = math.degrees(math.atan2(dy1, dx1))
                ang2 = math.degrees(math.atan2(dy2, dx2))
                arc_r = 0.8
                theta1 = min(ang1, ang2)
                theta2 = max(ang1, ang2)
                if theta2 - theta1 > 180:
                    theta1, theta2 = theta2, theta1 + 360
                arc = patches.Arc(
                    (vx, vy), 2 * arc_r, 2 * arc_r,
                    angle=0, theta1=theta1, theta2=theta2,
                    color=arc_color, linewidth=lw * 0.75)
                ax.add_patch(arc)
                mid_ang = math.radians((theta1 + theta2) / 2)
                tx = vx + (arc_r + 0.55) * math.cos(mid_ang)
                ty = vy + (arc_r + 0.55) * math.sin(mid_ang)

                if ai in shown_angles:
                    label = f"{shown_angles[ai]}\u00b0"
                    color = arc_color
                else:
                    label = "x"
                    color = unknown_color
                ax.text(tx, ty, label,
                        fontsize=fs + 1, ha="center", va="center",
                        fontfamily=ff, color=color, fontweight="bold")

        # Area question marker
        if show_area_question:
            ax.text(centroid[0], centroid[1], "Area = ?",
                    fontsize=fs + 2, ha="center", va="center",
                    fontfamily=ff,
                    color=unknown_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff9c4",
                              ec=unknown_color, alpha=0.9))

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Drawing: triangle with height
    # ------------------------------------------------------------------ #

    def _draw_triangle_with_height(self, verts, vlabels, edge_pairs, sides,
                                    angles_deg, shown_sides, shown_angles,
                                    base_idx, height):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        pal = style["palette"]
        fill_alpha = style["geo_fill_alpha"]

        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=fill_alpha + 0.3)
        ax.add_patch(tri)

        centroid = tuple(sum(v[i] for v in verts) / 3 for i in range(2))

        # Vertex labels
        for i, (vx, vy) in enumerate(verts):
            dx = vx - centroid[0]
            dy = vy - centroid[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                ox, oy = 0, 0.5
            else:
                ox = dx / dist * 0.7
                oy = dy / dist * 0.7
            ax.text(vx + ox, vy + oy, vlabels[i],
                    fontsize=fs + 4, fontweight="bold", ha="center", va="center",
                    fontfamily=ff, color=geo_line)

        # Draw height line (dashed)
        i1, i2 = edge_pairs[base_idx]
        bx1, by1 = verts[i1]
        bx2, by2 = verts[i2]
        apex = verts[base_idx]

        # Foot of perpendicular from apex to base line
        dx = bx2 - bx1
        dy = by2 - by1
        base_len_sq = dx * dx + dy * dy
        if base_len_sq < 1e-9:
            return self.fig_to_pil(fig, dpi=style["dpi"])
        t = ((apex[0] - bx1) * dx + (apex[1] - by1) * dy) / base_len_sq
        foot_x = bx1 + t * dx
        foot_y = by1 + t * dy

        unknown_color = pal[5]
        ax.plot([apex[0], foot_x], [apex[1], foot_y],
                color=unknown_color, linewidth=lw, linestyle="--", zorder=4)

        # Right angle marker
        sz = 0.4
        perp_dx = -(by2 - by1) / math.sqrt(base_len_sq) * sz
        perp_dy = (bx2 - bx1) / math.sqrt(base_len_sq) * sz
        base_dx = dx / math.sqrt(base_len_sq) * sz
        base_dy = dy / math.sqrt(base_len_sq) * sz
        if (apex[1] - foot_y) > 0:
            pass
        else:
            perp_dx, perp_dy = -perp_dx, -perp_dy
        sq_pts = [
            (foot_x, foot_y),
            (foot_x + base_dx, foot_y + base_dy),
            (foot_x + base_dx + perp_dx, foot_y + base_dy + perp_dy),
            (foot_x + perp_dx, foot_y + perp_dy),
        ]
        sq = plt.Polygon(sq_pts, fill=False, edgecolor=unknown_color, linewidth=lw * 0.6)
        ax.add_patch(sq)

        # Label height as "x"
        hmx = (apex[0] + foot_x) / 2
        hmy = (apex[1] + foot_y) / 2
        ax.text(hmx - 0.6, hmy, "x",
                fontsize=fs + 4, ha="center", va="center",
                fontfamily=ff,
                color=unknown_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="#fff9c4",
                          ec=unknown_color, alpha=0.95))

        # Side labels
        known_color = pal[2]
        for si in range(3):
            if si in shown_sides:
                ei1, ei2 = edge_pairs[si]
                mx = (verts[ei1][0] + verts[ei2][0]) / 2
                my = (verts[ei1][1] + verts[ei2][1]) / 2
                ex = verts[ei2][0] - verts[ei1][0]
                ey = verts[ei2][1] - verts[ei1][1]
                elen = math.hypot(ex, ey)
                if elen < 1e-9:
                    continue
                nx, ny = -ey / elen, ex / elen
                to_cent = (centroid[0] - mx, centroid[1] - my)
                if nx * to_cent[0] + ny * to_cent[1] > 0:
                    nx, ny = -nx, -ny
                ax.text(mx + nx * 0.5, my + ny * 0.5,
                        f"{shown_sides[si]}",
                        fontsize=fs + 2, ha="center", va="center",
                        fontfamily=ff,
                        color=known_color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc=style["bg_color"],
                                  ec=known_color, alpha=0.9))

        # Angle labels
        arc_color = pal[3]
        for ai in shown_angles:
            vx, vy = verts[ai]
            i_prev = (ai - 1) % 3
            i_next = (ai + 1) % 3
            dx1 = verts[i_prev][0] - vx
            dy1 = verts[i_prev][1] - vy
            dx2 = verts[i_next][0] - vx
            dy2 = verts[i_next][1] - vy
            ang1 = math.degrees(math.atan2(dy1, dx1))
            ang2 = math.degrees(math.atan2(dy2, dx2))
            arc_r = 0.8
            theta1 = min(ang1, ang2)
            theta2 = max(ang1, ang2)
            if theta2 - theta1 > 180:
                theta1, theta2 = theta2, theta1 + 360
            arc = patches.Arc(
                (vx, vy), 2 * arc_r, 2 * arc_r,
                angle=0, theta1=theta1, theta2=theta2,
                color=arc_color, linewidth=lw * 0.75)
            ax.add_patch(arc)
            mid_ang = math.radians((theta1 + theta2) / 2)
            tx = vx + (arc_r + 0.55) * math.cos(mid_ang)
            ty = vy + (arc_r + 0.55) * math.sin(mid_ang)
            ax.text(tx, ty, f"{shown_angles[ai]}\u00b0",
                    fontsize=fs + 1, ha="center", va="center",
                    fontfamily=ff, color=arc_color, fontweight="bold")

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _angles_from_sides(a, b, c):
        try:
            cos_A = (b * b + c * c - a * a) / (2 * b * c)
            cos_B = (a * a + c * c - b * b) / (2 * a * c)
            cos_C = (a * a + b * b - c * c) / (2 * a * b)
            for cv in (cos_A, cos_B, cos_C):
                if cv < -1 or cv > 1:
                    return None
            return (math.acos(cos_A), math.acos(cos_B), math.acos(cos_C))
        except (ValueError, ZeroDivisionError):
            return None
