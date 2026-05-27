"""
Angle Chase Two-Step QA environment (v3 diversity redesign, 2026-04-16).

Template: multi-primitive angle chasing on parallel + transversal +
triangle + circle primitives. Every problem is a 1-5 hop chain.

v3 diversity redesign:
  * 8 diagram families (instead of 3): scalene triangle, obtuse triangle,
    isoceles triangle, right-triangle, parallel + transversal, parallel
    + triangle, two-triangle chain, inscribed circle, kite, quadrilateral
    with diagonal.
  * Per-seed: geometry randomly stretched (vertex offsets), rotation of
    whole figure, flipped orientations.
  * 6 question templates, 8 title variants.
  * L0 = simple triangle sum, L9 = 5-hop chain with red herrings and
    multi-primitive diagrams.
  * Per-seed palette shuffle.

Output format: constant 4-option MCQ (single letter).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_POOL = [
    "Angle chase", "Find angle x", "Geometry problem", "Angles",
    "Angle puzzle", "Figure", "Angle diagram", "Compute x",
]

_QUESTION_TEMPLATES = [
    "Using the angles labeled on the diagram, find the unknown angle x. {opts}. Answer with a single letter.",
    "From the figure, compute angle x using the given angles. {opts}. Answer with a single letter.",
    "The figure labels several angles. Find angle x as shown in the image. {opts}. Answer with a single letter.",
    "Use the labeled angles on the diagram to determine x. {opts}. Answer with a single letter A, B, C, or D.",
    "Given the angles labeled in the figure, what is angle x? {opts}. Answer with a single letter.",
    "Apply angle chasing on the figure to find x. {opts}. Answer with a single letter.",
]

def _rot_point(cx, cy, x, y, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    dx, dy = x - cx, y - cy
    return (cx + c * dx - s * dy, cy + s * dx + c * dy)

class AngleChaseTwoStepQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "angle_chase_two_step"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "chain_depth": 1 + level // 2,                 # 1..5
            "theorem_pool": 2 + level,                      # 2..11
            "n_labeled": 99,
            "use_frac": level >= 6,
            "n_red_herring_lines": level // 3,              # 0..3
            "tight_distractors": level >= 4,
            "allow_flip": level >= 2,
            "allow_rotation": level >= 3,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["chain_depth"]

        depth = cfg["chain_depth"]
        if depth == 1:
            problem_types = ["triangle_sum", "right_triangle", "isoceles"]
        elif depth == 2:
            problem_types = ["parallel_tri", "exterior_angle", "isoceles",
                             "kite"]
        elif depth == 3:
            problem_types = ["parallel_tri", "exterior_angle", "inscribed",
                             "two_triangles", "quad_diag"]
        elif depth == 4:
            problem_types = ["two_triangles", "inscribed", "parallel_double",
                             "quad_diag", "obtuse_chain"]
        else:
            problem_types = ["two_triangles", "parallel_double", "inscribed",
                             "obtuse_chain", "quad_diag"]

        for _ in range(30):
            try:
                ptype = rng.choice(problem_types)
                result = self._dispatch(ptype, rng, level, cfg)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _dispatch(self, ptype, rng, level, cfg):
        if ptype == "triangle_sum":
            return self._triangle_sum(rng, cfg)
        if ptype == "right_triangle":
            return self._right_triangle(rng, cfg)
        if ptype == "exterior_angle":
            return self._exterior_angle(rng, cfg)
        if ptype == "isoceles":
            return self._isoceles_chain(rng, cfg)
        if ptype == "parallel_tri":
            return self._parallel_tri(rng, cfg)
        if ptype == "two_triangles":
            return self._two_triangles(rng, cfg)
        if ptype == "inscribed":
            return self._inscribed_chain(rng, cfg)
        if ptype == "parallel_double":
            return self._parallel_double(rng, cfg)
        if ptype == "kite":
            return self._kite_problem(rng, cfg)
        if ptype == "quad_diag":
            return self._quad_diag(rng, cfg)
        if ptype == "obtuse_chain":
            return self._obtuse_chain(rng, cfg)
        return None

    # ---------------------- problem builders ---------------------- #
    def _pick_angle(self, rng, cfg, lo=25, hi=135) -> float:
        if cfg["use_frac"] and rng.random() < 0.5:
            return round(rng.uniform(lo, hi) * 2) / 2.0
        return float(rng.randint(lo, hi))

    def _triangle_sum(self, rng, cfg):
        a = self._pick_angle(rng, cfg, 30, 90)
        b = self._pick_angle(rng, cfg, 30, 90)
        if a + b >= 170:
            return None
        c = 180 - a - b
        return self._finalize(rng, cfg, "triangle", {
            "labels_at": {"A": a, "B": b, "C": "x"},
            "actual": {"A": a, "B": b, "C": c},
            "shape": "triangle",
            "x_val": c,
        })

    def _right_triangle(self, rng, cfg):
        a = self._pick_angle(rng, cfg, 20, 70)
        c = 90 - a
        return self._finalize(rng, cfg, "right_triangle", {
            "labels_at": {"A": 90, "B": a, "C": "x"},
            "actual": {"A": 90, "B": a, "C": c},
            "shape": "right_triangle",
            "x_val": c,
        })

    def _exterior_angle(self, rng, cfg):
        a = self._pick_angle(rng, cfg, 30, 70)
        b = self._pick_angle(rng, cfg, 30, 70)
        if a + b >= 160:
            return None
        ext = a + b
        return self._finalize(rng, cfg, "triangle", {
            "labels_at": {"A": a, "B": b, "C_exterior": "x"},
            "actual": {"A": a, "B": b, "C_exterior": ext},
            "shape": "triangle",
            "x_val": ext,
            "has_exterior": True,
        })

    def _isoceles_chain(self, rng, cfg):
        apex = self._pick_angle(rng, cfg, 20, 140)
        base = (180 - apex) / 2
        if not cfg["use_frac"]:
            base = round(base)
        if rng.random() < 0.5:
            given = ("apex", apex)
            unknown_val = base
            labels = {"apex": apex, "base": "x", "base2": base}
        else:
            given = ("base", base)
            unknown_val = 180 - 2 * base
            labels = {"apex": "x", "base": base, "base2": base}
        return self._finalize(rng, cfg, "isoceles", {
            "labels_at": labels,
            "actual": {"apex": apex, "base": base, "base2": base},
            "shape": "isoceles",
            "x_val": unknown_val,
        })

    def _parallel_tri(self, rng, cfg):
        a = self._pick_angle(rng, cfg, 30, 80)
        b = self._pick_angle(rng, cfg, 30, 80)
        if a + b >= 170:
            return None
        c = 180 - a - b
        return self._finalize(rng, cfg, "parallel_tri", {
            "labels_at": {"angle1_on_L1": a, "triangle_B": b,
                           "angle_at_C": "x"},
            "actual": {"angle1_on_L1": a, "triangle_B": b,
                        "angle_at_C": c},
            "shape": "parallel_tri",
            "x_val": c,
        })

    def _two_triangles(self, rng, cfg):
        a = self._pick_angle(rng, cfg, 30, 70)
        b = self._pick_angle(rng, cfg, 30, 70)
        if a + b >= 160:
            return None
        shared = 180 - a - b
        d = self._pick_angle(rng, cfg, 30, 80)
        if shared + d >= 170:
            return None
        x = 180 - (180 - shared) - d
        if x <= 5 or x >= 175:
            return None
        return self._finalize(rng, cfg, "two_triangles", {
            "labels_at": {"A": a, "B": b, "D": d, "x": "x"},
            "actual": {"A": a, "B": b, "D": d, "x": x},
            "shape": "two_triangles",
            "x_val": x,
        })

    def _inscribed_chain(self, rng, cfg):
        inscribed = self._pick_angle(rng, cfg, 20, 80)
        central = 2 * inscribed
        if central >= 170:
            return None
        other = self._pick_angle(rng, cfg, 20, 70)
        if central + other >= 170:
            return None
        x = 180 - central - other
        if x <= 5:
            return None
        return self._finalize(rng, cfg, "inscribed", {
            "labels_at": {"inscribed": inscribed, "other": other, "x": "x"},
            "actual": {"inscribed": inscribed, "other": other, "x": x},
            "shape": "inscribed",
            "x_val": x,
        })

    def _parallel_double(self, rng, cfg):
        a = self._pick_angle(rng, cfg, 30, 70)
        b = self._pick_angle(rng, cfg, 30, 60)
        if a + b >= 155:
            return None
        exterior = a + b
        other = self._pick_angle(rng, cfg, 20, 60)
        if exterior + other >= 170:
            return None
        x = 180 - exterior - other
        if x <= 5:
            return None
        return self._finalize(rng, cfg, "parallel_double", {
            "labels_at": {"A": a, "B": b, "other": other, "x": "x"},
            "actual": {"A": a, "B": b, "other": other, "x": x},
            "shape": "parallel_double",
            "x_val": x,
        })

    def _kite_problem(self, rng, cfg):
        # Kite: symmetric quadrilateral. Angles satisfy A + B = 180 style
        # relations (here we use: top=apex, bottom=beta, sides=gamma each;
        # total = 360).
        apex = self._pick_angle(rng, cfg, 40, 130)
        bottom = self._pick_angle(rng, cfg, 40, 130)
        if apex + bottom >= 320:
            return None
        side = (360 - apex - bottom) / 2
        if not cfg["use_frac"]:
            side = round(side)
        if side <= 5 or side >= 175:
            return None
        return self._finalize(rng, cfg, "kite", {
            "labels_at": {"apex": apex, "bottom": bottom, "side_l": side,
                           "side_r": "x"},
            "actual": {"apex": apex, "bottom": bottom, "side_l": side,
                        "side_r": side},
            "shape": "kite",
            "x_val": side,
        })

    def _quad_diag(self, rng, cfg):
        # Quadrilateral with a diagonal: 4 triangle sums chain.
        a = self._pick_angle(rng, cfg, 30, 80)
        b = self._pick_angle(rng, cfg, 30, 80)
        if a + b >= 170:
            return None
        t1_third = 180 - a - b
        c = self._pick_angle(rng, cfg, 30, 80)
        if c >= 170 - t1_third:
            return None
        x = 180 - c - t1_third
        if x <= 5 or x >= 175:
            return None
        return self._finalize(rng, cfg, "quad_diag", {
            "labels_at": {"A": a, "B": b, "C": c, "D": "x"},
            "actual": {"A": a, "B": b, "C": c, "D": x},
            "shape": "quad_diag",
            "x_val": x,
        })

    def _obtuse_chain(self, rng, cfg):
        """Obtuse triangle with an external angle cascade."""
        obtuse = self._pick_angle(rng, cfg, 95, 140)
        other = self._pick_angle(rng, cfg, 20, 60)
        if obtuse + other >= 175:
            return None
        x = 180 - obtuse - other
        if x <= 5:
            return None
        return self._finalize(rng, cfg, "obtuse_chain", {
            "labels_at": {"obtuse": obtuse, "other": other, "x": "x"},
            "actual": {"obtuse": obtuse, "other": other, "x": x},
            "shape": "obtuse_triangle",
            "x_val": x,
        })

    # ---------------------- finalization ---------------------- #
    def _finalize(self, rng, cfg, shape_type, data):
        correct_val = data["x_val"]
        if cfg["use_frac"]:
            correct_val_disp = round(correct_val, 1)
        else:
            correct_val_disp = int(round(correct_val))

        if cfg["tight_distractors"]:
            offsets = [-5, -2, 3, 7, 10, -10]
        else:
            offsets = [-20, -10, 10, 20, 30, -30]
        rng.shuffle(offsets)
        distractors = []
        for off in offsets:
            cand = correct_val_disp + off
            if cand <= 0 or cand >= 180:
                continue
            if abs(cand - correct_val_disp) < 1:
                continue
            if cand in distractors:
                continue
            distractors.append(cand)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            return None
        options = [correct_val_disp] + distractors[:3]
        rng.shuffle(options)
        correct_idx = options.index(correct_val_disp)
        correct_letter = chr(ord("A") + correct_idx)

        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}°" for i, v in enumerate(options)
        )
        instruct = rng.choice(_QUESTION_TEMPLATES).format(opts=opts_text)

        title = rng.choice(_TITLE_POOL)
        image = self._render(shape_type, data, cfg, title, rng)
        return instruct, correct_letter, image

    # ------------------------------ rendering ------------------------------ #

    def _render(self, shape_type, data, cfg, title, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        rng.shuffle(palette)

        fig, ax = plt.subplots(figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=style["font_size_base"] + 2,
                     fontweight="bold", pad=8)

        line_color = style["geo_line_color"]
        lw = style["line_width"] * 1.2

        # Global transform: rotate & flip.
        rot = 0.0
        if cfg["allow_rotation"]:
            rot = rng.uniform(-25, 25)
        flip_x = (cfg["allow_flip"] and rng.random() < 0.5)
        # Offset
        off_x = rng.uniform(-0.6, 0.6)
        off_y = rng.uniform(-0.4, 0.4)

        def transform(x, y):
            if flip_x:
                x = 4 - x
            rx, ry = _rot_point(2, 1.5, x + off_x, y + off_y, rot)
            return rx, ry

        # Dispatch drawing.
        if shape_type == "triangle":
            self._draw_triangle(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "right_triangle":
            self._draw_right_triangle(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "isoceles":
            self._draw_isoceles(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "parallel_tri":
            self._draw_parallel_tri(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "two_triangles":
            self._draw_two_triangles(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "inscribed":
            self._draw_inscribed(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "parallel_double":
            self._draw_parallel_double(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "kite":
            self._draw_kite(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "quad_diag":
            self._draw_quad_diag(ax, data, line_color, lw, transform, cfg, palette, rng)
        elif shape_type == "obtuse_chain":
            self._draw_obtuse(ax, data, line_color, lw, transform, cfg, palette, rng)
        else:
            self._draw_triangle(ax, data, line_color, lw, transform, cfg, palette, rng)

        # Red-herring lines.
        n_red = cfg.get("n_red_herring_lines", 0)
        for i in range(n_red):
            x0 = rng.uniform(-3.5, 0)
            x1 = rng.uniform(5, 7.5)
            y0 = rng.uniform(-3, 4)
            y1 = rng.uniform(-3, 4)
            rx0, ry0 = transform(x0, y0)
            rx1, ry1 = transform(x1, y1)
            ax.plot([rx0, rx1], [ry0, ry1],
                    color="#aaaaaa", alpha=0.22, linewidth=0.8, zorder=0)

        ax.set_xlim(-3.5, 7.5)
        ax.set_ylim(-3.0, 5.2)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _label_angle(self, ax, cx, cy, text, transform, color="#c0392b"):
        tx, ty = transform(cx, cy)
        ax.text(tx, ty, text, fontsize=10, color=color, fontweight="bold",
                zorder=6, ha="center", va="center")

    def _draw_polyline(self, ax, pts, transform, color, lw):
        xs = [transform(p[0], p[1])[0] for p in pts]
        ys = [transform(p[0], p[1])[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=lw, zorder=2)

    def _draw_triangle(self, ax, data, color, lw, transform, cfg, palette, rng):
        # Random triangle vertex offsets for per-seed shape variance.
        dx1 = rng.uniform(-0.5, 0.5)
        dy1 = rng.uniform(-0.3, 0.3)
        dx2 = rng.uniform(-0.6, 0.3)
        p_a = (0.1 + dx1, 0.1 + dy1)
        p_b = (4.0 + dx2, 0.0)
        p_c = (rng.uniform(1.0, 2.5), 3.0 + rng.uniform(-0.3, 0.4))

        pts = [p_a, p_b, p_c, p_a]
        self._draw_polyline(ax, pts, transform, color, lw)
        labels = data["labels_at"]
        actual = data["actual"]

        verts = {"A": p_a, "B": p_b, "C": p_c}
        # Place labels toward centroid
        cx = sum(v[0] for v in verts.values()) / 3
        cy = sum(v[1] for v in verts.values()) / 3
        # Exterior is drawn as an extension of side AB beyond C.
        for name, val in labels.items():
            if name in verts:
                vx, vy = verts[name]
                dx, dy = cx - vx, cy - vy
                norm = math.hypot(dx, dy) + 1e-6
                tx, ty = vx + dx / norm * 0.65, vy + dy / norm * 0.65
                text = (f"{val:.1f}°" if isinstance(val, float)
                        and cfg["use_frac"] and val != "x" else
                        (f"{int(val)}°" if isinstance(val, (int, float))
                         else f"{val}°"))
                self._label_angle(ax, tx, ty, text, transform)
            elif name == "C_exterior":
                # draw extension beyond C along BC direction
                bx, by = verts["B"]
                cxv, cyv = verts["C"]
                dx2 = cxv - bx
                dy2 = cyv - by
                nm = math.hypot(dx2, dy2) + 1e-6
                tip = (cxv + dx2 / nm * 1.5, cyv + dy2 / nm * 1.5)
                self._draw_polyline(ax, [verts["C"], tip], transform,
                                    color, lw * 0.8)
                text = (f"{val}°" if val == "x"
                        else f"{int(val)}°")
                self._label_angle(ax, (cxv + tip[0]) / 2,
                                  (cyv + tip[1]) / 2 + 0.3,
                                  text, transform)
        # Label vertices
        for name, (x, y) in verts.items():
            tx, ty = transform(x + 0.15, y + 0.15)
            ax.text(tx, ty, name, fontsize=10, fontweight="bold", zorder=4)

    def _draw_right_triangle(self, ax, data, color, lw, transform, cfg, palette, rng):
        p_a = (0.2, 0.2)
        p_b = (3.0 + rng.uniform(-0.3, 0.5), 0.2)
        p_c = (0.2, 2.5 + rng.uniform(-0.2, 0.5))
        pts = [p_a, p_b, p_c, p_a]
        self._draw_polyline(ax, pts, transform, color, lw)
        # right-angle marker
        tx, ty = transform(p_a[0] + 0.25, p_a[1] + 0.25)
        ax.plot([transform(p_a[0] + 0.25, p_a[1])[0],
                 transform(p_a[0] + 0.25, p_a[1] + 0.25)[0],
                 transform(p_a[0], p_a[1] + 0.25)[0]],
                [transform(p_a[0] + 0.25, p_a[1])[1],
                 transform(p_a[0] + 0.25, p_a[1] + 0.25)[1],
                 transform(p_a[0], p_a[1] + 0.25)[1]],
                color=color, linewidth=1.0)

        labels = data["labels_at"]
        verts = {"A": p_a, "B": p_b, "C": p_c}
        cx = sum(v[0] for v in verts.values()) / 3
        cy = sum(v[1] for v in verts.values()) / 3
        for name, val in labels.items():
            if name in verts:
                vx, vy = verts[name]
                dx, dy = cx - vx, cy - vy
                norm = math.hypot(dx, dy) + 1e-6
                tx2, ty2 = vx + dx / norm * 0.55, vy + dy / norm * 0.55
                text = (f"{val}°" if val == "x"
                        else (f"{int(val)}°" if val == 90
                              else (f"{val:.1f}°"
                                    if cfg["use_frac"] and isinstance(val, float)
                                    else f"{int(val)}°")))
                self._label_angle(ax, tx2, ty2, text, transform)
        for name, (x, y) in verts.items():
            tx, ty = transform(x + 0.15, y + 0.15)
            ax.text(tx, ty, name, fontsize=10, fontweight="bold", zorder=4)

    def _draw_isoceles(self, ax, data, color, lw, transform, cfg, palette, rng):
        width = rng.uniform(2.0, 3.0)
        height = rng.uniform(2.2, 3.2)
        p_a = (-width / 2 + 2.0, 0)
        p_b = (width / 2 + 2.0, 0)
        p_c = (2.0, height)
        pts = [p_a, p_b, p_c, p_a]
        self._draw_polyline(ax, pts, transform, color, lw)
        labels = data["labels_at"]
        # apex at p_c, base at p_a, p_b
        # Mark equal-tick on sides
        def mid(p, q):
            return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        for a, b in [(p_a, p_c), (p_b, p_c)]:
            mx, my = mid(a, b)
            tx, ty = transform(mx + 0.1, my)
            ax.plot(tx, ty, marker='|', color="#7f8c8d", markersize=8)

        apex_val = labels.get("apex")
        base_val = labels.get("base")
        base2_val = labels.get("base2")
        self._label_angle(ax, p_c[0], p_c[1] - 0.45,
                          f"{apex_val if apex_val == 'x' else int(apex_val)}°"
                          if apex_val is not None else "",
                          transform)
        self._label_angle(ax, p_a[0] + 0.55, p_a[1] + 0.25,
                          f"{base_val if base_val == 'x' else int(base_val)}°"
                          if base_val is not None else "",
                          transform)
        self._label_angle(ax, p_b[0] - 0.55, p_b[1] + 0.25,
                          f"{base2_val if base2_val == 'x' else int(base2_val)}°"
                          if base2_val is not None else "",
                          transform)

    def _draw_parallel_tri(self, ax, data, color, lw, transform, cfg, palette, rng):
        y1 = 2.8 + rng.uniform(-0.2, 0.3)
        y2 = 0 + rng.uniform(-0.2, 0.2)
        self._draw_polyline(ax, [(-2, y1), (6, y1)], transform, color, lw)
        self._draw_polyline(ax, [(-2, y2), (6, y2)], transform, color, lw)
        col1 = palette[1 % len(palette)]
        col2 = palette[2 % len(palette)]
        self._draw_polyline(ax, [(0, y2), (4, y1)], transform, col1, lw)
        self._draw_polyline(ax, [(4, y1), (5.5, y2)], transform, col2, lw)
        self._draw_polyline(ax, [(0, y2), (5.5, y2)], transform, color, lw)
        labels = data["labels_at"]
        if "angle1_on_L1" in labels:
            v = labels["angle1_on_L1"]
            self._label_angle(ax, 4.1, y1 + 0.2,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "triangle_B" in labels:
            v = labels["triangle_B"]
            self._label_angle(ax, 5.0, y2 + 0.3,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "angle_at_C" in labels:
            v = labels["angle_at_C"]
            self._label_angle(ax, 0.4, y2 + 0.25,
                              f"{v if v == 'x' else int(v)}°", transform)
        tx, ty = transform(-1.9, y1 + 0.25)
        ax.text(tx, ty, "l1", fontsize=10)
        tx, ty = transform(-1.9, y2 + 0.25)
        ax.text(tx, ty, "l2", fontsize=10)

    def _draw_two_triangles(self, ax, data, color, lw, transform, cfg, palette, rng):
        p1 = (0, 0 + rng.uniform(-0.2, 0.2))
        p2 = (4 + rng.uniform(-0.3, 0.3), 0)
        p3 = (1.8 + rng.uniform(-0.3, 0.3), 2.8 + rng.uniform(-0.3, 0.3))
        p4 = (5.5 + rng.uniform(-0.3, 0.3), 2.0 + rng.uniform(-0.3, 0.3))
        self._draw_polyline(ax, [p1, p2, p3, p1], transform, color, lw)
        self._draw_polyline(ax, [p2, p4, p3], transform, color, lw)
        tx, ty = transform(p1[0] - 0.2, p1[1] - 0.2)
        ax.text(tx, ty, "A", fontsize=10, fontweight="bold")
        tx, ty = transform(p2[0], p2[1] - 0.2)
        ax.text(tx, ty, "B", fontsize=10, fontweight="bold")
        tx, ty = transform(p3[0] - 0.2, p3[1] + 0.15)
        ax.text(tx, ty, "C", fontsize=10, fontweight="bold")
        tx, ty = transform(p4[0] + 0.1, p4[1])
        ax.text(tx, ty, "D", fontsize=10, fontweight="bold")
        labels = data["labels_at"]
        if "A" in labels:
            v = labels["A"]
            self._label_angle(ax, 0.4, 0.25,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "B" in labels:
            v = labels["B"]
            self._label_angle(ax, 3.3, 0.25,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "D" in labels:
            v = labels["D"]
            self._label_angle(ax, 4.7, 1.7,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "x" in labels:
            v = labels["x"]
            self._label_angle(ax, 2.9, 1.8,
                              f"{v if v == 'x' else int(v)}°", transform)

    def _draw_inscribed(self, ax, data, color, lw, transform, cfg, palette, rng):
        R = rng.uniform(1.6, 1.9)
        cx, cy = 2, 1.2
        # Circle approximated as a polygon for transform-friendliness.
        pts = []
        for i in range(48):
            a = 2 * math.pi * i / 48
            pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
        pts.append(pts[0])
        self._draw_polyline(ax, pts, transform, color, lw)
        cxt, cyt = transform(cx, cy)
        ax.plot(cxt, cyt, "o", color=palette[0], markersize=5)
        txt, tyt = transform(cx + 0.1, cy + 0.1)
        ax.text(txt, tyt, "O", fontsize=10, fontweight="bold")
        a_pts = [math.pi / 2, math.pi + 0.6 + rng.uniform(-0.1, 0.1),
                 -0.3 + rng.uniform(-0.1, 0.1)]
        names = ["A", "B", "C"]
        circle_pts = [(cx + R * math.cos(a), cy + R * math.sin(a))
                      for a in a_pts]
        for i, (x, y) in enumerate(circle_pts):
            tx, ty = transform(x, y)
            ax.plot(tx, ty, "o",
                    color=palette[i % len(palette)], markersize=5)
            ax.text(*transform(x + 0.12, y + 0.1), names[i],
                    fontsize=10, fontweight="bold")
        self._draw_polyline(ax, [circle_pts[0], circle_pts[1]],
                            transform, color, lw)
        self._draw_polyline(ax, [circle_pts[0], circle_pts[2]],
                            transform, color, lw)
        # radii (dashed)
        xs1, ys1 = transform(cx, cy)
        xs2, ys2 = transform(*circle_pts[1])
        ax.plot([xs1, xs2], [ys1, ys2], color=color, linewidth=lw,
                linestyle="--")
        xs2, ys2 = transform(*circle_pts[2])
        ax.plot([xs1, xs2], [ys1, ys2], color=color, linewidth=lw,
                linestyle="--")
        labels = data["labels_at"]
        if "inscribed" in labels:
            v = labels["inscribed"]
            self._label_angle(ax, circle_pts[0][0] - 0.3,
                              circle_pts[0][1] - 0.35,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "other" in labels:
            v = labels["other"]
            self._label_angle(ax, cx - 0.8, cy - 0.15,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "x" in labels:
            v = labels["x"]
            self._label_angle(ax, cx + 0.1, cy - 0.35,
                              f"{v if v == 'x' else int(v)}°", transform)

    def _draw_parallel_double(self, ax, data, color, lw, transform, cfg, palette, rng):
        y1 = 2.8 + rng.uniform(-0.2, 0.2)
        y2 = 0 + rng.uniform(-0.2, 0.2)
        self._draw_polyline(ax, [(-2, y1), (6, y1)], transform, color, lw)
        self._draw_polyline(ax, [(-2, y2), (6, y2)], transform, color, lw)
        self._draw_polyline(ax, [(-1, y1), (2, y2)], transform,
                            palette[1 % len(palette)], lw)
        self._draw_polyline(ax, [(4, y1), (5, y2)], transform,
                            palette[2 % len(palette)], lw)
        self._draw_polyline(ax, [(-1, y1), (2, y2), (0, y2)], transform,
                            color, lw)
        self._draw_polyline(ax, [(4, y1), (5, y2), (3, y2)], transform,
                            color, lw)
        labels = data["labels_at"]
        if "A" in labels:
            v = labels["A"]
            self._label_angle(ax, -0.5, y2 + 0.25,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "B" in labels:
            v = labels["B"]
            self._label_angle(ax, 0.9, y1 - 0.4,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "other" in labels:
            v = labels["other"]
            self._label_angle(ax, 3.1, y2 + 0.25,
                              f"{v if v == 'x' else int(v)}°", transform)
        if "x" in labels:
            v = labels["x"]
            self._label_angle(ax, 4.3, y1 - 0.4,
                              f"{v if v == 'x' else int(v)}°", transform)

    def _draw_kite(self, ax, data, color, lw, transform, cfg, palette, rng):
        # Kite with vertical symmetry axis.
        w = rng.uniform(1.2, 2.0)
        h1 = rng.uniform(1.5, 2.2)
        h2 = rng.uniform(0.8, 1.4)
        cx, cy = 2.0, 1.0
        top = (cx, cy + h1)
        bot = (cx, cy - h2)
        left = (cx - w, cy)
        right = (cx + w, cy)
        self._draw_polyline(ax, [top, right, bot, left, top], transform,
                            color, lw)
        labels = data["labels_at"]
        apex = labels.get("apex")
        bottom = labels.get("bottom")
        side_l = labels.get("side_l")
        side_r = labels.get("side_r")
        if apex is not None:
            self._label_angle(ax, top[0], top[1] - 0.3,
                              f"{apex if apex == 'x' else int(apex)}°",
                              transform)
        if bottom is not None:
            self._label_angle(ax, bot[0], bot[1] + 0.3,
                              f"{bottom if bottom == 'x' else int(bottom)}°",
                              transform)
        if side_l is not None:
            self._label_angle(ax, left[0] + 0.35, left[1],
                              f"{side_l if side_l == 'x' else int(side_l)}°",
                              transform)
        if side_r is not None:
            self._label_angle(ax, right[0] - 0.35, right[1],
                              f"{side_r if side_r == 'x' else int(side_r)}°",
                              transform)

    def _draw_quad_diag(self, ax, data, color, lw, transform, cfg, palette, rng):
        p_a = (0 + rng.uniform(-0.2, 0.2), 0)
        p_b = (3.5 + rng.uniform(-0.3, 0.3), 0)
        p_c = (4.0 + rng.uniform(-0.3, 0.3), 2.5 + rng.uniform(-0.2, 0.3))
        p_d = (0.5 + rng.uniform(-0.3, 0.3), 2.8 + rng.uniform(-0.2, 0.3))
        self._draw_polyline(ax, [p_a, p_b, p_c, p_d, p_a], transform,
                            color, lw)
        # diagonal A-C
        self._draw_polyline(ax, [p_a, p_c], transform, color, lw * 0.85)

        for name, pt in [("A", p_a), ("B", p_b), ("C", p_c), ("D", p_d)]:
            tx, ty = transform(pt[0] + 0.12, pt[1] + 0.12)
            ax.text(tx, ty, name, fontsize=10, fontweight="bold")
        labels = data["labels_at"]
        centroid = ((p_a[0] + p_b[0] + p_c[0] + p_d[0]) / 4,
                    (p_a[1] + p_b[1] + p_c[1] + p_d[1]) / 4)
        for name, pt in [("A", p_a), ("B", p_b), ("C", p_c), ("D", p_d)]:
            if name in labels:
                v = labels[name]
                dx, dy = centroid[0] - pt[0], centroid[1] - pt[1]
                nm = math.hypot(dx, dy) + 1e-6
                tx, ty = pt[0] + dx / nm * 0.65, pt[1] + dy / nm * 0.65
                self._label_angle(ax, tx, ty,
                                  f"{v if v == 'x' else int(v)}°",
                                  transform)

    def _draw_obtuse(self, ax, data, color, lw, transform, cfg, palette, rng):
        p_a = (0, 0 + rng.uniform(-0.2, 0.2))
        p_b = (4 + rng.uniform(-0.3, 0.5), 0)
        p_c = (3.8 + rng.uniform(-0.3, 0.2), 1.8 + rng.uniform(-0.2, 0.3))
        self._draw_polyline(ax, [p_a, p_b, p_c, p_a], transform, color, lw)
        centroid = ((p_a[0] + p_b[0] + p_c[0]) / 3,
                    (p_a[1] + p_b[1] + p_c[1]) / 3)
        verts = {"obtuse": p_b, "other": p_a, "x": p_c}
        # Use distinct single-letter vertex labels (A, B, C) so two
        # vertices don't both render as "O".
        vert_letters = {"other": "A", "obtuse": "B", "x": "C"}
        for name, val in data["labels_at"].items():
            if name in verts:
                v = verts[name]
                dx, dy = centroid[0] - v[0], centroid[1] - v[1]
                nm = math.hypot(dx, dy) + 1e-6
                tx, ty = v[0] + dx / nm * 0.6, v[1] + dy / nm * 0.6
                self._label_angle(ax, tx, ty,
                                  f"{val if val == 'x' else int(val)}°",
                                  transform)
        for name, pt in verts.items():
            tx, ty = transform(pt[0] + 0.15, pt[1] + 0.15)
            ax.text(tx, ty, vert_letters[name], fontsize=10,
                    fontweight="bold")

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_act", exist_ok=True)
    env = AngleChaseTwoStepQA()
    for level in [0, 3, 6, 9]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[angle_chase_two_step] L{level} s{seed} FAILED")
                continue
            img = env.render()
            out = f"/tmp/env_check_act/angle_chase_two_step_s{seed}_L{level}.png"
            img.save(out)
            print(f"saved {out} | A={env._answer}")
