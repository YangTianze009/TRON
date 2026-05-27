"""
Transformation Chain Prediction QA.

2026-05-05 R5 P1: REWRITE — match WeMath Basic Transformations verbatim
phrasing at L2+, KEEP existing single_vertex_coord algebra mode at L0/L1
(this works for 4B per task instructions).

WeMath patterns matched:
  - L0/L1 (UNCHANGED): single_vertex_coord algebra
    "The original point (a, b) is shown in the figure (red dot). Apply the
     transformation: <op>. What is the new coordinate?"
  - L2-L4 (NEW): Q4 style — single rotation around named endpoint O,
    asking which transformation maps shape A → shape B
    "As shown in the diagram, by rotating around the endpoint O, how can
     shape A be transformed into shape B?
     A. Rotate 90° clockwise; B. Rotate 90° counterclockwise; C. Rotate
     45° counterclockwise; D. Rotate 45° clockwise; E. No correct answer"
  - L5-L7 (NEW): Q12 style — chained transforms with named pivots P/Q
    "As shown in the diagram, Figure ① (    ) to obtain Figure ②."
  - L8-L9 (NEW): Q12 style with tighter distractors + higher trap rate
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _rotate_pts(pts, angle_deg, cx=0, cy=0):
    """Rotate points by angle_deg counterclockwise about (cx, cy).
    Pass negative angle for clockwise rotation."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        out.append((round(rx, 3), round(ry, 3)))
    return out


def _translate_pts(pts, tx, ty):
    return [(x + tx, y + ty) for x, y in pts]


def _centroid(pts):
    n = len(pts)
    return (sum(x for x, y in pts) / n, sum(y for x, y in pts) / n)


class TransformationChainPredictionQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "transformation_chain_prediction"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0/L1 — UNCHANGED: single_vertex_coord algebra (works for 4B)
        if level == 0:
            return {
                "mode": "single_vertex_coord",
                "force_ops": ["rotate180"],  # restrict to ONE op type at L0
            }
        if level == 1:
            return {
                "mode": "single_vertex_coord",
                "op_pool_size": 4,
            }
        # L2-L4 — Q4 style single rotation around named point O
        if level <= 4:
            return {
                "mode": "wemath_q4_single_rotation",
                "shape_pool": ["triangle", "rectangle"],
                "use_5_options": (level == 4),
                "trap_rate": 0.0 if level <= 3 else 0.15,
            }
        # L5-L7 — Q12 style compound
        if level <= 7:
            return {
                "mode": "wemath_q12_compound",
                "shape_pool": ["triangle", "rectangle"],
                "use_5_options": False,
                "trap_rate": 0.20,
                "tight_distractors": False,
            }
        # L8-L9 — Q12 style with harder distractors + higher trap rate
        return {
            "mode": "wemath_q12_compound",
            "shape_pool": ["triangle", "rectangle", "l_shape"],
            "use_5_options": (level == 9),
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        mode = cfg["mode"]
        for _ in range(20):
            if mode == "single_vertex_coord":
                r = self._gen_single_vertex_coord(rng, cfg)
            elif mode == "wemath_q4_single_rotation":
                r = self._gen_q4_single_rotation(rng, cfg)
            else:
                r = self._gen_q12_compound(rng, cfg)
            if r is not None:
                return r
        return None

    # ------------------------------------------------------------------ #
    # L0/L1 — single_vertex_coord algebra (UNCHANGED behavior)
    # ------------------------------------------------------------------ #
    def _gen_single_vertex_coord(self, rng, cfg):
        all_ops = ["rotate90", "rotate180", "rotate270",
                   "reflect_h", "reflect_v"]
        if "force_ops" in cfg:
            ops_pool = cfg["force_ops"]
        else:
            ops_pool = all_ops[: cfg.get("op_pool_size", 4)]
        op = rng.choice(ops_pool)
        x0 = rng.randint(1, 4)
        y0 = rng.randint(1, 4)
        if op == "rotate90":
            new_pt = (y0, -x0)
            op_text = "rotate 90° clockwise about origin"
        elif op == "rotate180":
            new_pt = (-x0, -y0)
            op_text = "rotate 180° about origin"
        elif op == "rotate270":
            new_pt = (-y0, x0)
            op_text = "rotate 270° clockwise about origin"
        elif op == "reflect_h":
            new_pt = (-x0, y0)
            op_text = "reflect across the y-axis"
        else:  # reflect_v
            new_pt = (x0, -y0)
            op_text = "reflect across the x-axis"

        all_results = {
            "rotate90": (y0, -x0),
            "rotate180": (-x0, -y0),
            "rotate270": (-y0, x0),
            "reflect_h": (-x0, y0),
            "reflect_v": (x0, -y0),
        }
        distractors = [v for k, v in all_results.items() if v != new_pt]
        rng.shuffle(distractors)
        options = [new_pt] + distractors[:3]
        rng.shuffle(options)
        idx = options.index(new_pt)
        answer = chr(ord("A") + idx)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot([x0], [y0], 'o', color="#c0392b", markersize=10)
        ax.text(x0 + 0.2, y0 + 0.2, f"({x0}, {y0})", fontsize=12,
                fontweight="bold", color="#c0392b")
        ax.axhline(0, color="#2c3e50", lw=1.0)
        ax.axvline(0, color="#2c3e50", lw=1.0)
        ax.grid(True, alpha=0.3)
        max_v = max(x0, y0, abs(x0), abs(y0)) + 2
        ax.set_xlim(-max_v, max_v)
        ax.set_ylim(-max_v, max_v)
        ax.set_aspect("equal")
        ax.set_title("Original point (red dot)", fontsize=11)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()

        opt_text = "\n".join(
            f"{chr(ord('A')+i)}. ({options[i][0]}, {options[i][1]})"
            for i in range(4)
        )
        q = (
            f"The original point ({x0}, {y0}) is shown in the figure "
            f"(red dot). Apply the transformation: {op_text}. What is "
            f"the new coordinate?\n"
            f"{opt_text}\n"
            "Answer with a single letter A, B, C, or D."
        )
        return q, answer, img

    # ------------------------------------------------------------------ #
    # L2-L4 — WeMath Q4 style single rotation around named point O
    # Verbatim Q4: "As shown in the diagram, by rotating around the
    #               endpoint O, how can shape A be transformed into shape B?
    #               A. Rotate 90° clockwise; B. Rotate 90° counterclockwise;
    #               C. Rotate 45° counterclockwise; D. Rotate 45° clockwise;
    #               E. No correct answer"
    # ------------------------------------------------------------------ #
    def _gen_q4_single_rotation(self, rng, cfg):
        shape_type = rng.choice(cfg["shape_pool"])
        if shape_type == "triangle":
            pts = [(1, 1), (4, 1), (1, 3)]
        else:  # rectangle
            pts = [(2, 1), (4, 1), (4, 2), (2, 2)]

        # True rotation angle ∈ {90, 180, 270} (multiples of 90 — clean lattice)
        true_angle = rng.choice([90, 180, 270])
        true_dir = rng.choice(["CW", "CCW"])
        signed = (-true_angle) if true_dir == "CW" else true_angle
        rotated = _rotate_pts(pts, signed, 0, 0)

        truth = (true_angle, true_dir)
        # Distractor pool — other valid rotations (incl. 45° as Q4 does)
        all_options_pool = []
        for ang in (45, 90, 180, 270):
            for d in ("CW", "CCW"):
                if (ang, d) != truth:
                    all_options_pool.append((ang, d))
        rng.shuffle(all_options_pool)

        use_5 = cfg["use_5_options"]
        n_distractor = 3 if use_5 else 2
        distractors = all_options_pool[:n_distractor]

        use_trap = (rng.random() < cfg["trap_rate"])
        if use_trap:
            extra = next(
                (o for o in all_options_pool if o not in distractors), None,
            )
            if extra is None:
                use_trap = False

        if use_trap:
            options_data = [extra] + distractors
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
            correct_letter = opt_letters[-1]
        else:
            options_data = [truth] + distractors
            rng.shuffle(options_data)
            opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
            correct_letter = opt_letters[options_data.index(truth)]

        def _fmt(angle, direction):
            d_word = "clockwise" if direction == "CW" else "counterclockwise"
            return f"Rotate {angle}° {d_word}"

        opt_texts = [_fmt(a, d) for a, d in options_data]
        opt_texts.append("No correct answer")

        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            "As shown in the diagram, by rotating around the endpoint O, "
            "how can shape A be transformed into shape B?\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._draw_two_shapes_with_O(
            shape_a=pts, shape_b=rotated, label_a="A", label_b="B",
            pivot_label="O", pivot_pos=(0, 0), rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L5-L9 — WeMath Q12 style compound (rotate around named pivot + translate)
    # ------------------------------------------------------------------ #
    def _gen_q12_compound(self, rng, cfg):
        shape_type = rng.choice(cfg["shape_pool"])
        if shape_type == "triangle":
            pts = [(1, 1), (3, 1), (1, 3)]
        elif shape_type == "rectangle":
            pts = [(1, 1), (3, 1), (3, 3), (1, 3)]
        else:  # l_shape
            pts = [(1, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3)]

        true_rot = rng.choice([90, 180, 270])
        true_pivot = rng.choice(["P", "Q"])
        pivot_coords = {"P": (0, 0), "Q": (5, 0)}
        true_dir = rng.choice(["left", "right", "up", "down"])
        true_steps = rng.randint(1, 3)
        truth = (true_rot, true_pivot, true_dir, true_steps)

        def _opt_text(rot, pivot, direction, steps):
            unit = "square" if steps == 1 else "squares"
            return (f"Rotate {rot}° counterclockwise around point {pivot}, "
                    f"then move {steps} {unit} to the {direction}")

        distractor_pool = []
        for r in (90, 180, 270):
            if r != true_rot:
                distractor_pool.append((r, true_pivot, true_dir, true_steps))
        for p in ("P", "Q"):
            if p != true_pivot:
                distractor_pool.append((true_rot, p, true_dir, true_steps))
        for d in ("left", "right", "up", "down"):
            if d != true_dir:
                distractor_pool.append((true_rot, true_pivot, d, true_steps))
        for s in (1, 2, 3):
            if s != true_steps:
                distractor_pool.append((true_rot, true_pivot, true_dir, s))

        if cfg["tight_distractors"]:
            tight = [d for d in distractor_pool if d[1] != true_pivot]
            other = [d for d in distractor_pool if d[1] == true_pivot]
            rng.shuffle(tight)
            rng.shuffle(other)
            distractor_pool = tight + other
        else:
            rng.shuffle(distractor_pool)

        use_5 = cfg["use_5_options"]
        n_distractor = 3 if use_5 else 2
        chosen_distractors = []
        seen_texts = {_opt_text(*truth)}
        for d in distractor_pool:
            t = _opt_text(*d)
            if t not in seen_texts:
                chosen_distractors.append(d)
                seen_texts.add(t)
            if len(chosen_distractors) >= n_distractor + 1:
                break
        if len(chosen_distractors) < n_distractor:
            return None

        use_trap = (rng.random() < cfg["trap_rate"])
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]

        if use_trap and len(chosen_distractors) >= n_distractor + 1:
            options_data = chosen_distractors[: n_distractor + 1]
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            options_data = [truth] + chosen_distractors[:n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(truth)]

        opt_texts = [_opt_text(*d) for d in options_data]
        opt_texts.append("No correct answer")

        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            "As shown in the diagram, Figure ① (    ) to obtain Figure ②.\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        cx, cy = pivot_coords[true_pivot]
        rotated = _rotate_pts(pts, true_rot, cx, cy)
        dx, dy = 0, 0
        if true_dir == "left":
            dx = -true_steps
        elif true_dir == "right":
            dx = true_steps
        elif true_dir == "up":
            dy = true_steps
        elif true_dir == "down":
            dy = -true_steps
        result_pts = _translate_pts(rotated, dx, dy)

        image = self._draw_compound_figures(
            orig_pts=pts, result_pts=result_pts,
            pivot_coords=pivot_coords, rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Drawing helpers (shared with composition env style)
    # ------------------------------------------------------------------ #
    def _draw_two_shapes_with_O(self, shape_a, shape_b, label_a, label_b,
                                pivot_label, pivot_pos, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        palette = style["palette"]
        polyA = Polygon(shape_a, closed=True,
                        facecolor=palette[0], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyA)
        cA = _centroid(shape_a)
        ax.text(cA[0], cA[1], label_a,
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='black')
        polyB = Polygon(shape_b, closed=True,
                        facecolor=palette[1 % len(palette)], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyB)
        cB = _centroid(shape_b)
        ax.text(cB[0], cB[1], label_b,
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='black')

        px, py = pivot_pos
        ax.plot(px, py, marker='o', color='black', markersize=10, zorder=5)
        ax.annotate(pivot_label, xy=(px, py), xytext=(px + 0.3, py + 0.3),
                    fontsize=15, fontweight='bold', color='red')

        all_pts = list(shape_a) + list(shape_b) + [pivot_pos]
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        margin = 2
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.axvline(0, color='gray', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticks(range(int(xmin), int(xmax) + 1))
        ax.set_yticks(range(int(ymin), int(ymax) + 1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=8)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_compound_figures(self, orig_pts, result_pts, pivot_coords,
                               rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        all_pts = list(orig_pts) + list(result_pts) + list(pivot_coords.values())
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        margin = 3
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin

        ax.axhline(0, color='gray', linewidth=0.6)
        ax.axvline(0, color='gray', linewidth=0.6)
        ax.grid(True, linestyle='--', alpha=0.35)

        palette = style["palette"]
        poly1 = Polygon(orig_pts, closed=True,
                        facecolor=palette[0], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(poly1)
        c1 = _centroid(orig_pts)
        ax.annotate("Figure ①", xy=c1,
                    xytext=(c1[0] + 0.4, c1[1] + 0.5),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.9, edgecolor=palette[0]))

        poly2 = Polygon(result_pts, closed=True,
                        facecolor=palette[1 % len(palette)], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(poly2)
        c2 = _centroid(result_pts)
        ax.annotate("Figure ②", xy=c2,
                    xytext=(c2[0] + 0.4, c2[1] + 0.5),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.9, edgecolor=palette[1 % len(palette)]))

        for name, (px, py) in pivot_coords.items():
            ax.plot(px, py, marker='o', color='black', markersize=8, zorder=5)
            ax.annotate(name, xy=(px, py),
                        xytext=(px + 0.25, py + 0.25),
                        fontsize=14, fontweight='bold', color='red')

        ax.set_xticks(range(int(xmin), int(xmax) + 1))
        ax.set_yticks(range(int(ymin), int(ymax) + 1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=8)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        return super()._check_answer(predicted, ground_truth)
