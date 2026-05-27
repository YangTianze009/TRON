"""
Vision-Only Angle QA environment.

Draws parallel lines with transversals, or polygons with some angles marked.
All measurements are ON the diagram. Question only says "Find angle x."

Diversity & difficulty redesign (2026-04-16):
- All geometry RNG is level-aware (uses sub_rng, not shared _rng).
- L0 is structurally simpler: only 2-angle supplementary and vertical problems
  on a single line/crossing. Rounded answers.
- L9 uses 5-7 sided polygons, 3-parallel transversals, and multi-step chaining
  ("x is vertical to y, y is supplementary to z=..."), plus MCQ-style at top
  levels.
- 4+ phrasings; per-seed jitter for arc radius, vertex letter pool, line offset.
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

_QUESTION_PHRASINGS = [
    "Find angle x (in degrees).",
    "What is the measure of angle x in degrees?",
    "Determine the value of the unknown angle x.",
    "Compute angle x (answer in degrees).",
    "Read the diagram. What is x (degrees)?",
]

_TITLE_POOL = [
    "Angle Problem", "Find x", "Geometry Angles",
    "Angle Relationship", "What is x?",
]

_VERTEX_LETTERS = [
    # Intentionally excluding "X" (ambiguous with unknown-angle label "x")
    ["A", "B", "C", "D", "E", "F", "G"],
    ["P", "Q", "R", "S", "T", "U", "V"],
    ["Y", "Z", "W", "V", "U", "T", "S"],
    ["K", "L", "M", "N", "O", "P", "Q"],
]

class VisionOnlyAngleQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "vision_only_angle"

    PROBLEM_TYPES = [
        "parallel_transversal", "triangle_angles", "polygon_angles",
        "supplementary", "vertical",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration — STRUCTURALLY different per level
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            # Only simplest: 2-angle supplementary, vertical (no chaining)
            return {"ptypes": ["supplementary2", "vertical_simple"],
                    "polygon_sides": (3, 3), "n_supp": 2, "is_mcq": False,
                    "trans_chain": 1, "round_step": 5}
        if level <= 2:
            return {"ptypes": ["supplementary2", "vertical_simple",
                               "triangle_easy"],
                    "polygon_sides": (3, 3), "n_supp": 2, "is_mcq": False,
                    "trans_chain": 1, "round_step": 5}
        if level <= 4:
            return {"ptypes": ["triangle_angles", "supplementary",
                               "vertical", "parallel_transversal"],
                    "polygon_sides": (3, 4), "n_supp": 2, "is_mcq": False,
                    "trans_chain": 2, "round_step": 1}
        if level <= 6:
            return {"ptypes": ["triangle_angles", "parallel_transversal",
                               "polygon_angles", "supplementary"],
                    "polygon_sides": (4, 5), "n_supp": 3, "is_mcq": False,
                    "trans_chain": 2, "round_step": 1}
        if level <= 7:
            return {"ptypes": ["triangle_angles", "parallel_transversal",
                               "polygon_angles", "multi_parallel_angles"],
                    "polygon_sides": (5, 6), "n_supp": 3, "is_mcq": True,
                    "trans_chain": 3, "round_step": 1}
        return {"ptypes": ["polygon_angles", "multi_parallel_angles",
                           "parallel_transversal", "triangle_angles"],
                "polygon_sides": (5, 7), "n_supp": 3, "is_mcq": True,
                "trans_chain": 3, "round_step": 1}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Level-aware sub-RNG
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        ptype = parameter.get("problem_type")
        if ptype is None:
            ptype = sub_rng.choice(cfg["ptypes"])

        for _ in range(40):
            try:
                result = self._dispatch(ptype, sub_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                pass
            # Try alternative ptype on failure
            ptype = sub_rng.choice(cfg["ptypes"])
        return None

    def _dispatch(self, ptype, rng, cfg, level):
        if ptype == "supplementary2":
            return self._supplementary(rng, cfg, level, n_forced=2)
        if ptype == "vertical_simple":
            return self._vertical(rng, cfg, level, simple=True)
        if ptype == "triangle_easy":
            return self._triangle_angles(rng, cfg, level, easy=True)
        if ptype == "parallel_transversal":
            return self._parallel_transversal(rng, cfg, level)
        if ptype == "triangle_angles":
            return self._triangle_angles(rng, cfg, level, easy=False)
        if ptype == "polygon_angles":
            return self._polygon_angles(rng, cfg, level)
        if ptype == "supplementary":
            return self._supplementary(rng, cfg, level)
        if ptype == "vertical":
            return self._vertical(rng, cfg, level, simple=False)
        if ptype == "multi_parallel_angles":
            return self._multi_parallel_angles(rng, cfg, level)
        return None

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_angle_arc(self, ax, vertex, dir1, dir2, radius=0.6,
                        color="#6a1b9a", label=None, is_unknown=False,
                        style=None):
        ang1 = math.degrees(math.atan2(dir1[1], dir1[0]))
        ang2 = math.degrees(math.atan2(dir2[1], dir2[0]))
        t1, t2 = min(ang1, ang2), max(ang1, ang2)
        if t2 - t1 > 180:
            t1, t2 = t2, t1 + 360

        lw = style.get("line_width", 1.8) if style else 1.8
        fs = style.get("font_size_base", 12) if style else 12
        ff = style.get("font_family", "sans-serif") if style else "sans-serif"
        unk_color = style["palette"][5] if style else "#d32f2f"

        arc = patches.Arc(vertex, 2 * radius, 2 * radius, angle=0,
                          theta1=t1, theta2=t2,
                          color=color if not is_unknown else unk_color,
                          linewidth=lw + 0.2)
        ax.add_patch(arc)

        mid = math.radians((t1 + t2) / 2)
        tx = vertex[0] + (radius + 0.35) * math.cos(mid)
        ty = vertex[1] + (radius + 0.35) * math.sin(mid)

        if label is not None:
            if is_unknown:
                ax.text(tx, ty, "x", fontsize=fs + 2, ha="center", va="center",
                        color=unk_color, fontweight="bold", fontfamily=ff,
                        bbox=dict(boxstyle="round,pad=0.1", fc="#fff9c4",
                                  ec=unk_color, alpha=0.95))
            else:
                ax.text(tx, ty, f"{label}\u00b0", fontsize=fs, ha="center",
                        va="center", color=color, fontweight="bold",
                        fontfamily=ff)

    # ------------------------------------------------------------------ #
    # Phrase / MCQ helper
    # ------------------------------------------------------------------ #
    def _format_question(self, rng, cfg, answer_val, is_int=True):
        """Return (question_text, answer_str) possibly MCQ-formatted."""
        base = rng.choice(_QUESTION_PHRASINGS)
        if is_int:
            gt_str = str(int(answer_val))
        else:
            gt_str = f"{answer_val:.1f}"
            if float(gt_str) == int(float(gt_str)):
                gt_str = str(int(float(gt_str)))

        if cfg.get("is_mcq", False):
            # Build 4 MCQ options including the correct one
            correct = int(round(answer_val))
            distractors = set()
            attempts = 0
            while len(distractors) < 3 and attempts < 50:
                delta = rng.choice([-40, -30, -20, -15, -10, 10, 15, 20, 30, 40])
                cand = max(5, min(175, correct + delta))
                if cand != correct:
                    distractors.add(cand)
                attempts += 1
            if len(distractors) < 3:
                return base, gt_str
            opts = [correct] + list(distractors)[:3]
            rng.shuffle(opts)
            correct_idx = opts.index(correct)
            letter = chr(ord("A") + correct_idx)
            q = (base + "\n" + "\n".join(
                f"  ({chr(ord('A')+i)}) {opts[i]}" for i in range(4))
                + "\nAnswer with a single letter.")
            return q, letter

        return base, gt_str

    def _vertex_names(self, rng, n):
        pool = rng.choice(_VERTEX_LETTERS)
        return pool[:n]

    # ------------------------------------------------------------------ #
    # Parallel lines with transversal
    # ------------------------------------------------------------------ #
    def _parallel_transversal(self, rng, cfg, level):
        y1 = 0.0
        y2 = rng.uniform(3.0, 5.0)
        x_extent = 8.0

        trans_angle = rng.uniform(30, 75)
        trans_rad = math.radians(trans_angle)

        ix1 = rng.uniform(-2, 2)
        ix2 = ix1 + (y2 - y1) / math.tan(trans_rad)

        given_angle = rng.randint(35, 145)

        rel = rng.choice(["corresponding", "alternate_interior",
                          "co_interior", "alternate_exterior"])
        if rel in ("corresponding", "alternate_interior",
                   "alternate_exterior"):
            answer_val = given_angle
        else:
            answer_val = 180 - given_angle

        dx = math.cos(trans_rad)
        dy = math.sin(trans_rad)
        t_len = 10.0

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(8 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        pal = style["palette"]

        ax.plot([-x_extent, x_extent], [y1, y1], color=geo_line, lw=lw + 0.2)
        ax.plot([-x_extent, x_extent], [y2, y2], color=geo_line, lw=lw + 0.2)

        for y_line in [y1, y2]:
            ax.annotate("", xy=(x_extent - 0.3, y_line),
                        xytext=(x_extent - 1.0, y_line),
                        arrowprops=dict(arrowstyle="->", color=geo_line, lw=lw))

        ax.plot([ix1 - dx * t_len, ix2 + dx * t_len],
                [y1 - dy * t_len, y2 + dy * t_len],
                color=pal[0], lw=lw, zorder=3)

        ax.text(-x_extent + 0.3, y1 + 0.3, "l\u2081", fontsize=fs + 2,
                color=geo_line, fontweight="bold", fontfamily=ff)
        ax.text(-x_extent + 0.3, y2 + 0.3, "l\u2082", fontsize=fs + 2,
                color=geo_line, fontweight="bold", fontfamily=ff)
        ax.plot(ix1, y1, "o", color=pal[1], markersize=6, zorder=5)
        ax.plot(ix2, y2, "o", color=pal[1], markersize=6, zorder=5)

        arc_color = pal[2]
        h_dir = (1, 0)
        t_dir_up = (dx, dy)
        t_dir_down = (-dx, -dy)

        arc_r = rng.uniform(0.55, 0.85)
        if rel == "corresponding":
            self._draw_angle_arc(ax, (ix1, y1), h_dir, t_dir_up,
                                 radius=arc_r, color=arc_color,
                                 label=given_angle, style=style)
            self._draw_angle_arc(ax, (ix2, y2), h_dir, t_dir_up,
                                 radius=arc_r, label="x",
                                 is_unknown=True, style=style)
        elif rel == "alternate_interior":
            self._draw_angle_arc(ax, (ix1, y1), h_dir, t_dir_up,
                                 radius=arc_r, color=arc_color,
                                 label=given_angle, style=style)
            h_dir_left = (-1, 0)
            self._draw_angle_arc(ax, (ix2, y2), h_dir_left, t_dir_down,
                                 radius=arc_r, label="x",
                                 is_unknown=True, style=style)
        elif rel == "co_interior":
            self._draw_angle_arc(ax, (ix1, y1), h_dir, t_dir_up,
                                 radius=arc_r, color=arc_color,
                                 label=given_angle, style=style)
            self._draw_angle_arc(ax, (ix2, y2), h_dir, t_dir_down,
                                 radius=arc_r, label="x",
                                 is_unknown=True, style=style)
        else:  # alternate_exterior
            h_dir_left = (-1, 0)
            self._draw_angle_arc(ax, (ix1, y1), h_dir_left, t_dir_down,
                                 radius=arc_r, color=arc_color,
                                 label=given_angle, style=style)
            self._draw_angle_arc(ax, (ix2, y2), h_dir, t_dir_up,
                                 radius=arc_r, label="x",
                                 is_unknown=True, style=style)

        ax.text(x_extent - 1.5, (y1 + y2) / 2, "l\u2081 \u2225 l\u2082",
                fontsize=fs + 1, color=geo_line, fontweight="bold",
                fontfamily=ff,
                bbox=dict(boxstyle="round,pad=0.15", fc=style["bg_color"],
                          ec=geo_line, alpha=0.9))

        margin = 2.0
        ax.set_xlim(-x_extent - margin, x_extent + margin)
        ax.set_ylim(y1 - 3, y2 + 3)

        q, a = self._format_question(rng, cfg, answer_val)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Triangle angles (sum = 180)
    # ------------------------------------------------------------------ #
    def _triangle_angles(self, rng, cfg, level, easy=False):
        if easy:
            # "Easy" triangle — one angle is 60 or 90 (nicer numbers)
            a1 = rng.choice([30, 45, 50, 60, 70, 90])
            a2 = rng.randint(30, min(90, 175 - a1 - 20))
            a3 = 180 - a1 - a2
            if a3 < 20:
                return None
        else:
            a1 = rng.randint(25, 100)
            a2 = rng.randint(25, min(100, 175 - a1 - 10))
            a3 = 180 - a1 - a2
            if a3 < 15:
                return None

        angles = [a1, a2, a3]
        ask_idx = rng.randint(0, 2)
        answer_val = angles[ask_idx]

        side_bc = rng.uniform(5, 8)
        angle_B_rad = math.radians(angles[1])
        side_ab = rng.uniform(4, 7)

        Bx, By = 0.0, 0.0
        Cx, Cy = side_bc, 0.0
        Ax = side_ab * math.cos(angle_B_rad)
        Ay = side_ab * math.sin(angle_B_rad)
        verts = [(Ax, Ay), (Bx, By), (Cx, Cy)]
        letters = self._vertex_names(rng, 3)
        labels = letters

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

        tri = plt.Polygon(verts, fill=True, facecolor=pal[0],
                          edgecolor=geo_line, linewidth=lw + 0.2,
                          alpha=style["geo_fill_alpha"] + 0.3)
        ax.add_patch(tri)

        centroid = tuple(sum(v[i] for v in verts) / 3 for i in range(2))

        for i, (vx, vy) in enumerate(verts):
            dx = vx - centroid[0]
            dy = vy - centroid[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                ox, oy = 0, 0.5
            else:
                ox = dx / dist * 0.7
                oy = dy / dist * 0.7
            ax.text(vx + ox, vy + oy, labels[i],
                    fontsize=fs + 4, fontweight="bold",
                    ha="center", va="center",
                    fontfamily=ff, color=geo_line)

        arc_color = pal[2]
        arc_r = rng.uniform(0.6, 0.85)
        for i in range(3):
            vx, vy = verts[i]
            i_prev = (i - 1) % 3
            i_next = (i + 1) % 3
            d1 = (verts[i_prev][0] - vx, verts[i_prev][1] - vy)
            d2 = (verts[i_next][0] - vx, verts[i_next][1] - vy)
            len1 = math.hypot(d1[0], d1[1])
            len2 = math.hypot(d2[0], d2[1])
            if len1 < 1e-9 or len2 < 1e-9:
                continue
            d1 = (d1[0] / len1, d1[1] / len1)
            d2 = (d2[0] / len2, d2[1] / len2)

            if i == ask_idx:
                self._draw_angle_arc(ax, (vx, vy), d1, d2, radius=arc_r,
                                     label="x", is_unknown=True, style=style)
            else:
                self._draw_angle_arc(ax, (vx, vy), d1, d2, radius=arc_r,
                                     color=arc_color, label=angles[i],
                                     style=style)

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 2.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        q, a = self._format_question(rng, cfg, answer_val)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Polygon angles
    # ------------------------------------------------------------------ #
    def _polygon_angles(self, rng, cfg, level):
        n_lo, n_hi = cfg["polygon_sides"]
        n = rng.randint(n_lo, n_hi)
        interior_sum = (n - 2) * 180

        angles = []
        remaining = interior_sum
        for i in range(n - 1):
            lo = max(60, remaining - (n - 1 - i) * 160)
            hi = min(160, remaining - (n - 1 - i) * 60)
            if lo > hi:
                return None
            a = rng.randint(int(lo), int(hi))
            angles.append(a)
            remaining -= a
        if remaining < 40 or remaining > 170:
            return None
        angles.append(remaining)

        ask_idx = rng.randint(0, n - 1)
        answer_val = angles[ask_idx]

        r = 4.0
        cx, cy = 0.0, 0.0
        base_angles = [2 * math.pi * i / n + math.pi / 2 + rng.uniform(-0.05, 0.05)
                       for i in range(n)]
        verts = [(cx + r * math.cos(a), cy + r * math.sin(a))
                 for a in base_angles]

        letters = self._vertex_names(rng, n)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        pal = style["palette"]

        poly = plt.Polygon(verts, fill=True, facecolor=pal[0],
                           edgecolor=geo_line, linewidth=lw + 0.2,
                           alpha=style["geo_fill_alpha"] + 0.3)
        ax.add_patch(poly)

        centroid = (sum(v[0] for v in verts) / n,
                    sum(v[1] for v in verts) / n)

        for i, (vx, vy) in enumerate(verts):
            dx = vx - centroid[0]
            dy = vy - centroid[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                ox, oy = 0, 0.5
            else:
                ox = dx / dist * 0.7
                oy = dy / dist * 0.7
            ax.text(vx + ox, vy + oy, letters[i],
                    fontsize=fs + 3, fontweight="bold", ha="center", va="center",
                    fontfamily=ff, color=geo_line)

        arc_color = pal[2]
        arc_r = rng.uniform(0.45, 0.65)
        for i in range(n):
            vx, vy = verts[i]
            i_prev = (i - 1) % n
            i_next = (i + 1) % n
            d1 = (verts[i_prev][0] - vx, verts[i_prev][1] - vy)
            d2 = (verts[i_next][0] - vx, verts[i_next][1] - vy)
            len1 = math.hypot(d1[0], d1[1])
            len2 = math.hypot(d2[0], d2[1])
            if len1 < 1e-9 or len2 < 1e-9:
                continue
            d1 = (d1[0] / len1, d1[1] / len1)
            d2 = (d2[0] / len2, d2[1] / len2)

            if i == ask_idx:
                self._draw_angle_arc(ax, (vx, vy), d1, d2, radius=arc_r,
                                     label="x", is_unknown=True, style=style)
            else:
                self._draw_angle_arc(ax, (vx, vy), d1, d2, radius=arc_r,
                                     color=arc_color, label=angles[i],
                                     style=style)

        margin = 2.5
        ax.set_xlim(cx - r - margin, cx + r + margin)
        ax.set_ylim(cy - r - margin, cy + r + margin)

        q, a = self._format_question(rng, cfg, answer_val)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Supplementary
    # ------------------------------------------------------------------ #
    def _supplementary(self, rng, cfg, level, n_forced=None):
        n_angles = n_forced if n_forced else rng.choice([2, cfg["n_supp"]])

        round_step = cfg.get("round_step", 1)

        if n_angles == 2:
            a1 = rng.randint(25, 155)
            if round_step > 1:
                a1 = max(25, (a1 // round_step) * round_step)
            a2 = 180 - a1
            angles = [a1, a2]
        else:
            a1 = rng.randint(20, 100)
            a2 = rng.randint(20, min(100, 160 - a1))
            a3 = 180 - a1 - a2
            if a3 < 15:
                return None
            angles = [a1, a2, a3]

        ask_idx = rng.randint(0, n_angles - 1)
        answer_val = angles[ask_idx]

        O = (0.0, 0.0)
        line_len = 7.0
        ray_len = 5.0

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(8 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        pal = style["palette"]

        ax.plot([-line_len, line_len], [0, 0], color=geo_line, lw=lw + 0.2)
        ax.plot(O[0], O[1], "o", color=pal[1], markersize=6, zorder=5)

        cum_angle = 0
        for i in range(n_angles - 1):
            cum_angle += angles[i]
            rad = math.radians(cum_angle)
            end = (O[0] + ray_len * math.cos(rad),
                   O[1] + ray_len * math.sin(rad))
            ax.plot([O[0], end[0]], [O[1], end[1]], color=pal[0],
                    lw=lw, zorder=3)

        ax.text(O[0] - 0.3, O[1] - 0.5, "O",
                fontsize=fs + 2, fontweight="bold",
                fontfamily=ff, color=pal[1], ha="center")

        arc_color = pal[2]
        unk_color = pal[5]
        cum = 0
        for i in range(n_angles):
            start_rad = math.radians(cum)
            end_rad = math.radians(cum + angles[i])
            mid_rad = (start_rad + end_rad) / 2

            arc = patches.Arc(O, 1.8, 1.8, angle=0,
                              theta1=cum, theta2=cum + angles[i],
                              color=arc_color if i != ask_idx else unk_color,
                              linewidth=lw)
            ax.add_patch(arc)

            tx = O[0] + 1.4 * math.cos(mid_rad)
            ty = O[1] + 1.4 * math.sin(mid_rad)

            if i == ask_idx:
                ax.text(tx, ty, "x", fontsize=fs + 3,
                        ha="center", va="center",
                        color=unk_color, fontweight="bold", fontfamily=ff,
                        bbox=dict(boxstyle="round,pad=0.1", fc="#fff9c4",
                                  ec=unk_color, alpha=0.95))
            else:
                ax.text(tx, ty, f"{angles[i]}\u00b0",
                        fontsize=fs + 1, ha="center", va="center",
                        color=arc_color, fontweight="bold", fontfamily=ff)

            cum += angles[i]

        ax.set_xlim(-line_len - 1, line_len + 1)
        ax.set_ylim(-2, ray_len + 1)

        q, a = self._format_question(rng, cfg, answer_val)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Vertical angles
    # ------------------------------------------------------------------ #
    def _vertical(self, rng, cfg, level, simple=False):
        angle1 = rng.randint(25, 80)
        if cfg.get("round_step", 1) > 1:
            angle1 = max(25, (angle1 // cfg["round_step"]) * cfg["round_step"])
        angle2 = 180 - angle1

        all_angles = [angle1, angle2, angle1, angle2]

        if simple:
            show_idx = 0
            ask_idx = 2  # Always vertical pair
        else:
            show_idx = rng.choice([0, 1])
            if rng.random() < 0.4:
                ask_idx = (show_idx + 1) % 4
            else:
                ask_idx = show_idx + 2 if show_idx < 2 else show_idx - 2
        answer_val = all_angles[ask_idx]

        O = (0.0, 0.0)
        line_len = 6.0

        rad1 = math.radians(angle1)
        l2_ends = [(-line_len * math.cos(rad1), -line_len * math.sin(rad1)),
                   (line_len * math.cos(rad1), line_len * math.sin(rad1))]

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

        ax.plot([-line_len, line_len], [0, 0], color=geo_line, lw=lw + 0.2)
        ax.plot([l2_ends[0][0], l2_ends[1][0]],
                [l2_ends[0][1], l2_ends[1][1]], color=geo_line, lw=lw + 0.2)

        ax.plot(O[0], O[1], "o", color=pal[1], markersize=6, zorder=5)

        sector_starts = [0, angle1, 180, 180 + angle1]
        sector_sizes = [angle1, angle2, angle1, angle2]
        arc_color = pal[2]
        unk_color = pal[5]

        for i in range(4):
            start = sector_starts[i]
            size = sector_sizes[i]
            mid_rad = math.radians(start + size / 2)

            a_color = arc_color
            if i == ask_idx:
                a_color = unk_color

            arc = patches.Arc(O, 2.0, 2.0, angle=0,
                              theta1=start, theta2=start + size,
                              color=a_color, linewidth=lw)
            ax.add_patch(arc)

            tx = O[0] + 1.5 * math.cos(mid_rad)
            ty = O[1] + 1.5 * math.sin(mid_rad)

            if i == show_idx:
                ax.text(tx, ty, f"{all_angles[i]}\u00b0",
                        fontsize=fs + 1, ha="center", va="center",
                        color=arc_color, fontweight="bold", fontfamily=ff)
            elif i == ask_idx:
                ax.text(tx, ty, "x", fontsize=fs + 3,
                        ha="center", va="center",
                        color=unk_color, fontweight="bold", fontfamily=ff,
                        bbox=dict(boxstyle="round,pad=0.1", fc="#fff9c4",
                                  ec=unk_color, alpha=0.95))

        margin = 2
        ax.set_xlim(-line_len - margin, line_len + margin)
        ax.set_ylim(-line_len - margin, line_len + margin)

        q, a = self._format_question(rng, cfg, answer_val)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Multi-parallel angles (3 parallel lines)
    # ------------------------------------------------------------------ #
    def _multi_parallel_angles(self, rng, cfg, level):
        n_lines = 3 if cfg.get("trans_chain", 1) >= 3 else 2
        if n_lines == 2:
            return self._parallel_transversal(rng, cfg, level)

        y1 = 0.0
        y2 = rng.uniform(2.5, 3.5)
        y3 = rng.uniform(5.5, 7.0)
        x_extent = 8.0

        trans_angle = rng.uniform(35, 70)
        given_angle = rng.randint(30, 150)

        # At L8/L9 alternate between corresponding/alternate for the unknown
        chain_type = rng.choice(["corresponding_l3", "alternate_l2_l3"])
        if chain_type == "corresponding_l3":
            answer_val = given_angle
        else:  # alternate_interior between l2 and l3
            answer_val = given_angle

        rad1 = math.radians(trans_angle)
        ix1_1 = rng.uniform(-1, 1)
        ix1_3 = ix1_1 + (y3 - y1) / math.tan(rad1)

        dx1 = math.cos(rad1)
        dy1 = math.sin(rad1)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(8 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        pal = style["palette"]

        for y_val, label in [(y1, "l\u2081"), (y2, "l\u2082"), (y3, "l\u2083")]:
            ax.plot([-x_extent, x_extent], [y_val, y_val],
                    color=geo_line, lw=lw + 0.2)
            ax.text(-x_extent + 0.3, y_val + 0.3, label,
                    fontsize=fs + 1, color=geo_line, fontweight="bold",
                    fontfamily=ff)

        t_len = 10.0
        ax.plot([ix1_1 - dx1 * t_len, ix1_3 + dx1 * t_len],
                [y1 - dy1 * t_len, y3 + dy1 * t_len],
                color=pal[0], lw=lw, zorder=3)

        ix1_2 = ix1_1 + (y2 - y1) / math.tan(rad1)
        for ix, iy in [(ix1_1, y1), (ix1_2, y2), (ix1_3, y3)]:
            ax.plot(ix, iy, "o", color=pal[1], markersize=5, zorder=5)

        arc_color = pal[2]
        h_dir = (1, 0)
        t_dir_up = (dx1, dy1)

        arc_r = rng.uniform(0.6, 0.8)
        self._draw_angle_arc(ax, (ix1_1, y1), h_dir, t_dir_up,
                             radius=arc_r, color=arc_color,
                             label=given_angle, style=style)
        self._draw_angle_arc(ax, (ix1_3, y3), h_dir, t_dir_up,
                             radius=arc_r, label="x",
                             is_unknown=True, style=style)

        ax.text(x_extent - 2, (y1 + y3) / 2,
                "l\u2081 \u2225 l\u2082 \u2225 l\u2083",
                fontsize=fs, color=geo_line, fontweight="bold",
                fontfamily=ff,
                bbox=dict(boxstyle="round,pad=0.15", fc=style["bg_color"],
                          ec=geo_line, alpha=0.9))

        margin = 2.0
        ax.set_xlim(-x_extent - margin, x_extent + margin)
        ax.set_ylim(y1 - 3, y3 + 3)

        q, a = self._format_question(rng, cfg, answer_val)
        return q, a, self.fig_to_pil(fig, dpi=style["dpi"])
