"""
Mirror / Reflection QA environment.

2026-05-05 R5 P1: REWRITTEN to align question phrasing and option style with
the Basic Transformations of Figures topic family (single-rotation,
yes/no axis, axis-pick, compound reflect-then-translate).

Pure single-letter MCQ across all levels (A through E). Verifier delegates to
the base class which already handles letter extraction (3 wrappers).

Per-level design:
  L0/L1 — "Yes/No"-style 4-MCQ with vertical/horizontal trial:
    "Can the shaded shape A be reflected across axis L to coincide with
    shape B?
     A. Yes, across the vertical axis L
     B. Yes, across the horizontal axis L
     C. No
     D. No correct answer"
  L2/L3 — 4-MCQ "which labeled axis is the mirror" (4 axes drawn in image
    labeled L1, L2, L3, L4; only one is the true mirror):
    "As shown in the diagram, which axis line acts as the mirror that
    maps shape A onto shape B?
     A. L1; B. L2; C. L3; D. L4"
  L4 — 5-MCQ same as L2/L3 with "E. No correct answer" added (15% trap)
  L5/L6 — 4-MCQ compound: reflect across labeled axis L, then translate;
    "Shape A is reflected across axis L1 then translated K squares to the
    right to obtain shape B. Which option matches?"
     options vary (axis label, direction, magnitude) + "No correct answer"
  L7 — same compound but 5-MCQ with E option (raised trap rate)
  L8/L9 — 5-MCQ tight-distractor compound + 40% "No correct answer" trap

Naming conventions:
  - Shapes labelled "A" / "B" / "C"
  - Pivot/origin labelled "O" if needed
  - Axis lines labelled "L1" .. "L4" on the figure
  - Coordinate grid drawn in every figure for spatial grounding
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ====================================================================== #
# Geometry helpers
# ====================================================================== #

def _reflect_across_vertical(pts, x0):
    """Reflect points across the vertical line x = x0."""
    return [(2 * x0 - x, y) for (x, y) in pts]


def _reflect_across_horizontal(pts, y0):
    """Reflect points across the horizontal line y = y0."""
    return [(x, 2 * y0 - y) for (x, y) in pts]


def _reflect_across_diag_slope_pos(pts, b):
    """Reflect across line y = x + b."""
    out = []
    for (x, y) in pts:
        # General reflection across y = x + b
        nx = y - b
        ny = x + b
        out.append((nx, ny))
    return out


def _reflect_across_diag_slope_neg(pts, b):
    """Reflect across line y = -x + b."""
    out = []
    for (x, y) in pts:
        # General reflection across y = -x + b
        nx = b - y
        ny = b - x
        out.append((nx, ny))
    return out


def _translate(pts, dx, dy):
    return [(x + dx, y + dy) for (x, y) in pts]


def _centroid(pts):
    n = len(pts)
    return (sum(x for (x, _) in pts) / n, sum(y for (_, y) in pts) / n)


def _shapes_equal(a, b, tol=1e-6):
    """Two point lists are equal as polygons (in the same vertex order)."""
    if len(a) != len(b):
        return False
    return all(abs(ax - bx) < tol and abs(ay - by) < tol
               for (ax, ay), (bx, by) in zip(a, b))


# Axis-line representation: a tuple identifying which mirror line it is.
#   ("vert", x0)        -> vertical line x = x0
#   ("horz", y0)        -> horizontal line y = y0
#   ("diag_pos", b)     -> y =  x + b
#   ("diag_neg", b)     -> y = -x + b


def _apply_axis(pts, axis):
    kind = axis[0]
    if kind == "vert":
        return _reflect_across_vertical(pts, axis[1])
    if kind == "horz":
        return _reflect_across_horizontal(pts, axis[1])
    if kind == "diag_pos":
        return _reflect_across_diag_slope_pos(pts, axis[1])
    return _reflect_across_diag_slope_neg(pts, axis[1])


def _axis_human_label(axis):
    """Human-readable description of an axis (used internally only)."""
    kind = axis[0]
    if kind == "vert":
        return f"x = {axis[1]}"
    if kind == "horz":
        return f"y = {axis[1]}"
    if kind == "diag_pos":
        if axis[1] == 0:
            return "y = x"
        sign = "+" if axis[1] >= 0 else "-"
        return f"y = x {sign} {abs(axis[1])}"
    if axis[1] == 0:
        return "y = -x"
    sign = "+" if axis[1] >= 0 else "-"
    return f"y = -x {sign} {abs(axis[1])}"


def _round_pts(pts, ndigits=3):
    return [(round(x, ndigits), round(y, ndigits)) for (x, y) in pts]


# ====================================================================== #
# Main environment class
# ====================================================================== #

class MirrorReflectionQA(StandaloneVisualEnv):
    """Pure single-letter MCQ environment for axial-reflection reasoning."""

    ALLOW_ROTATION = False  # orientation-sensitive
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "mirror_reflection"

    QUESTION_TYPES = [
        "yesno_axis_match",        # L0/L1 — Q5-style yes/no 4-MCQ
        "axis_label_pick",         # L2-L4 — pick which labeled axis
        "compound_reflect_translate",  # L5-L9 — reflect-then-translate
    ]

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "qtype": "yesno_axis_match",
                "trap_rate": 0.10 if level == 1 else 0.0,
            }
        if level <= 3:
            return {
                "qtype": "axis_label_pick",
                "use_5_options": False,
                "include_diagonals": (level == 3),
                "trap_rate": 0.0,
            }
        if level == 4:
            return {
                "qtype": "axis_label_pick",
                "use_5_options": True,
                "include_diagonals": True,
                "trap_rate": 0.15,
            }
        if level <= 6:
            return {
                "qtype": "compound_reflect_translate",
                "use_5_options": False,
                "tight_distractors": False,
                "trap_rate": 0.20,
            }
        if level == 7:
            return {
                "qtype": "compound_reflect_translate",
                "use_5_options": True,
                "tight_distractors": False,
                "trap_rate": 0.25,
            }
        # L8 / L9 — tight distractors + 40% trap rate
        return {
            "qtype": "compound_reflect_translate",
            "use_5_options": True,
            "tight_distractors": True,
            "trap_rate": 0.40,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 613)
        qtype = cfg["qtype"]

        for _ in range(20):
            if qtype == "yesno_axis_match":
                r = self._try_yesno_axis_match(cfg, sub_rng)
            elif qtype == "axis_label_pick":
                r = self._try_axis_label_pick(cfg, sub_rng)
            else:
                r = self._try_compound_reflect_translate(cfg, sub_rng)
            if r is not None:
                return r
        return None

    # ------------------------------------------------------------------ #
    # Shape pool — small lattice polygons that look unambiguously chiral
    # under reflection.
    # ------------------------------------------------------------------ #
    def _pick_shape(self, rng) -> List[Tuple[int, int]]:
        choice = rng.choice(["right_triangle", "right_triangle_b",
                              "l_shape", "l_shape_b", "z_shape"])
        if choice == "right_triangle":
            return [(1, 1), (4, 1), (1, 3)]
        if choice == "right_triangle_b":
            return [(1, 1), (3, 1), (1, 4)]
        if choice == "l_shape":
            return [(1, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3)]
        if choice == "l_shape_b":
            return [(1, 1), (4, 1), (4, 2), (2, 2), (2, 3), (1, 3)]
        return [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (1, 3)]

    # ================================================================== #
    # L0/L1 — Q5-style Yes/No 4-MCQ
    #   "Can the shaded shape A be reflected to coincide with shape B?
    #    A. Yes, across the vertical axis L
    #    B. Yes, across the horizontal axis L
    #    C. No
    #    D. No correct answer"
    # ================================================================== #
    def _try_yesno_axis_match(self, cfg, rng):
        shape_a = self._pick_shape(rng)
        # Choose true axis: vertical or horizontal at integer position.
        true_kind = rng.choice(["vert", "horz", "none"])
        # `none` => B is just a translate (no reflection works) ⇒ correct = C
        # Trap (rare) => correct = D
        use_trap = (rng.random() < cfg.get("trap_rate", 0.0))

        cx, cy = _centroid(shape_a)
        # Pick axis position roughly outside the shape so reflection lands clearly apart
        if true_kind == "vert":
            x0 = int(round(max(p[0] for p in shape_a))) + rng.randint(1, 2)
            true_axis = ("vert", x0)
            shape_b = _round_pts(_apply_axis(shape_a, true_axis))
            answer_letter = "A"
        elif true_kind == "horz":
            y0 = int(round(max(p[1] for p in shape_a))) + rng.randint(1, 2)
            true_axis = ("horz", y0)
            shape_b = _round_pts(_apply_axis(shape_a, true_axis))
            answer_letter = "B"
        else:
            # Just translate so that NO axis-aligned reflection matches.
            dx = rng.choice([-3, -2, 2, 3])
            dy = rng.choice([-2, -1, 1, 2])
            shape_b = _round_pts(_translate(shape_a, dx, dy))
            answer_letter = "C"

        if use_trap:
            # Override: present a B that is rotated 90 deg from a reflection
            # so option C "No" is wrong but no listed axis matches => "D".
            angle = rng.choice([90, 180, 270])
            shape_b = self._rotate_pts(shape_a, angle, _centroid(shape_a))
            shape_b = _round_pts(shape_b)
            answer_letter = "D"

        # Render
        image = self._render_two_shapes(
            shape_a, shape_b, label_a="A", label_b="B",
            show_axes=[],  # Q5-style does not draw axes; just shapes
            rng=rng,
        )

        # Q5-verbatim style stem
        stems = [
            "Can the shaded shape A be reflected across a single straight "
            "axis to coincide exactly with shape B?",
            "As shown in the diagram, can shape A be reflected across one "
            "straight line so that the result coincides with shape B?",
            "Can the shaded figure A be turned into shape B by a single "
            "axial reflection (mirror flip)?",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            "A. Yes, across a vertical axis\n"
            "B. Yes, across a horizontal axis\n"
            "C. No\n"
            "D. No correct answer\n"
            "Answer with a single letter A, B, C, or D."
        )
        return question, answer_letter, image

    # ================================================================== #
    # L2-L4 — Pick which labeled axis is the true mirror
    #   "As shown in the diagram, which labeled axis line maps shape A
    #    onto shape B?
    #    A. L1; B. L2; C. L3; D. L4   (E. No correct answer)"
    # ================================================================== #
    def _try_axis_label_pick(self, cfg, rng):
        shape_a = self._pick_shape(rng)
        cx, cy = _centroid(shape_a)
        # Build candidate axis pool. Always 4 candidates labeled L1..L4.
        candidates = []
        # vertical / horizontal positions outside the shape footprint
        x_ext_right = int(round(max(p[0] for p in shape_a))) + 1
        y_ext_top = int(round(max(p[1] for p in shape_a))) + 1
        x_ext_left = int(round(min(p[0] for p in shape_a))) - 1
        y_ext_bot = int(round(min(p[1] for p in shape_a))) - 1

        candidates.append(("vert", x_ext_right))
        candidates.append(("horz", y_ext_top))
        candidates.append(("vert", x_ext_left))
        candidates.append(("horz", y_ext_bot))
        if cfg.get("include_diagonals"):
            # add some diagonals
            b1 = int(round(cy - cx))
            b2 = int(round(cy + cx))
            candidates.append(("diag_pos", b1))
            candidates.append(("diag_neg", b2))

        rng.shuffle(candidates)
        axis_pool = candidates[:4]  # exactly 4 axes drawn

        use_5 = cfg.get("use_5_options", False)
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        use_trap = (rng.random() < cfg.get("trap_rate", 0.0))

        # Decide true axis: at L4-trap mode we may pick none.
        if use_5 and use_trap:
            # No listed axis maps A to B. Build B as a rotation/translation.
            angle = rng.choice([90, 180, 270])
            shape_b = self._rotate_pts(shape_a, angle, (cx, cy))
            shape_b = _round_pts(shape_b)
            true_idx = -1
            answer_letter = opt_letters[-1]  # "E. No correct answer"
        else:
            # Pick one axis from the drawn axes as the true mirror.
            true_idx = rng.randrange(len(axis_pool))
            true_axis = axis_pool[true_idx]
            shape_b = _round_pts(_apply_axis(shape_a, true_axis))
            answer_letter = opt_letters[true_idx]

        # Render with all 4 axes labeled L1..L4
        image = self._render_axes_pick(
            shape_a, shape_b, axis_pool, axis_labels=["L1", "L2", "L3", "L4"],
            rng=rng,
        )

        opts = [f"L{i + 1}" for i in range(len(axis_pool))]
        opt_lines = "\n".join(f"{opt_letters[i]}. {opts[i]}"
                              for i in range(len(opts)))
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stems = [
            "As shown in the diagram, which labeled axis line acts as the "
            "mirror that reflects shape A onto shape B?",
            "Which of the labeled axes in the figure is the line of "
            "reflection that maps shape A to shape B?",
            "By reflecting shape A across one of the labeled axis lines, "
            "shape B is obtained. Which axis is the mirror?",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, answer_letter, image

    # ================================================================== #
    # L5-L9 — Compound: reflect across axis L then translate K squares
    #   "Shape A is reflected across axis L then translated by some
    #    amount to obtain shape B. Which option correctly describes the
    #    transformation?
    #    A. Reflect across L1, then move 2 squares to the right
    #    B. Reflect across L2, then move 2 squares to the right
    #    C. Reflect across L1, then move 1 square to the right
    #    D. No correct answer"
    # ================================================================== #
    def _try_compound_reflect_translate(self, cfg, rng):
        shape_a = self._pick_shape(rng)
        # Axes drawn = L1, L2 (vertical and horizontal), L3 optional diag
        cx, cy = _centroid(shape_a)
        x_ext_right = int(round(max(p[0] for p in shape_a))) + 1
        y_ext_top = int(round(max(p[1] for p in shape_a))) + 1
        # Two named axes always present
        axis_pool = [
            ("vert", x_ext_right),
            ("horz", y_ext_top),
        ]
        axis_labels = ["L1", "L2"]
        # Pick true axis index, true direction, true steps.
        true_axis_idx = rng.randint(0, 1)
        true_axis = axis_pool[true_axis_idx]
        true_dir = rng.choice(["left", "right", "up", "down"])
        true_steps = rng.randint(1, 3)

        truth = (true_axis_idx, true_dir, true_steps)

        def _opt_text(axis_idx, direction, steps):
            unit = "square" if steps == 1 else "squares"
            return (
                f"Reflect across axis {axis_labels[axis_idx]}, "
                f"then move {steps} {unit} to the {direction}"
            )

        # Build distractor pool — each swaps exactly ONE field
        distractors_pool = []
        for ai in (0, 1):
            if ai != true_axis_idx:
                distractors_pool.append((ai, true_dir, true_steps))
        for d in ("left", "right", "up", "down"):
            if d != true_dir:
                distractors_pool.append((true_axis_idx, d, true_steps))
        for s in (1, 2, 3):
            if s != true_steps:
                distractors_pool.append((true_axis_idx, true_dir, s))

        if cfg.get("tight_distractors"):
            # Bias toward step-difference distractors at L8/L9.
            tight = [d for d in distractors_pool if d[2] != true_steps]
            other = [d for d in distractors_pool if d[2] == true_steps]
            rng.shuffle(tight); rng.shuffle(other)
            distractors_pool = tight + other
        else:
            rng.shuffle(distractors_pool)

        use_5 = cfg.get("use_5_options", False)
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        # Number of MCQ "value" options (excluding final "No correct answer")
        n_val_options = len(opt_letters) - 1
        n_distractor = n_val_options - 1

        # Dedup distractor texts
        chosen = []
        seen_texts = {_opt_text(*truth)}
        for d in distractors_pool:
            t = _opt_text(*d)
            if t not in seen_texts:
                chosen.append(d)
                seen_texts.add(t)
            if len(chosen) >= n_distractor + 1:
                break  # +1 spare for trap mode
        if len(chosen) < n_distractor:
            return None

        use_trap = (rng.random() < cfg.get("trap_rate", 0.0))
        if use_trap and len(chosen) >= n_distractor + 1:
            # All non-final options are distractors; correct = last letter.
            options_data = chosen[: n_val_options]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            options_data = [truth] + chosen[: n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(truth)]

        # Compute actual shape_b = reflect(shape_a, true_axis) then translate
        reflected_a = _apply_axis(shape_a, true_axis)
        if true_dir == "left":
            dx, dy = -true_steps, 0
        elif true_dir == "right":
            dx, dy = true_steps, 0
        elif true_dir == "up":
            dx, dy = 0, true_steps
        else:
            dx, dy = 0, -true_steps
        shape_b = _round_pts(_translate(reflected_a, dx, dy))

        # If trap mode, override shape_b to a different transform so NO listed
        # option is correct. We rotate shape_a + translate by an arbitrary
        # off-list amount.
        if use_trap and correct_letter == opt_letters[-1]:
            # Rotate 90 + arbitrary translate to land somewhere disjoint.
            rotated = self._rotate_pts(shape_a, rng.choice([90, 180, 270]),
                                        (cx, cy))
            shape_b = _round_pts(_translate(rotated, rng.choice([-2, 2]),
                                            rng.choice([-2, 2])))

        opt_texts = [_opt_text(*d) for d in options_data]
        opt_texts.append("No correct answer")

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {opt_texts[i]}"
            for i in range(len(opt_letters))
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stems = [
            "As shown in the diagram, shape A is transformed into shape B "
            "by a reflection across one of the labeled axes followed by a "
            "translation. Which option describes the correct transformation?",
            "By first reflecting shape A across one of the labeled axes "
            "and then translating the result, shape B is obtained. Which "
            "of the following describes the transformation?",
            "Shape A is mapped to shape B in two steps: reflect across an "
            "axis (L1 or L2), then translate by some squares. Which "
            "option matches the actual transformation shown?",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._render_axes_pick(
            shape_a, shape_b, axis_pool, axis_labels=axis_labels, rng=rng,
        )
        return question, correct_letter, image

    # ================================================================== #
    # Drawing helpers
    # ================================================================== #

    def _render_two_shapes(self, shape_a, shape_b, label_a, label_b,
                            show_axes, rng) -> Image.Image:
        """Render two shapes A and B on a coordinate grid (no labeled mirror
        lines drawn). Used for L0/L1 yes/no MCQ."""
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
        ax.text(cA[0], cA[1], label_a, ha='center', va='center',
                fontsize=16, fontweight='bold', color='black')

        polyB = Polygon(shape_b, closed=True,
                        facecolor=palette[1 % len(palette)], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyB)
        cB = _centroid(shape_b)
        ax.text(cB[0], cB[1], label_b, ha='center', va='center',
                fontsize=16, fontweight='bold', color='black')

        all_pts = list(shape_a) + list(shape_b)
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        margin = 2
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin
        ax.axhline(0, color='gray', linewidth=0.6)
        ax.axvline(0, color='gray', linewidth=0.6)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticks(range(int(math.floor(xmin)), int(math.ceil(xmax)) + 1))
        ax.set_yticks(range(int(math.floor(ymin)), int(math.ceil(ymax)) + 1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=8)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_axes_pick(self, shape_a, shape_b, axes, axis_labels,
                          rng) -> Image.Image:
        """Render two shapes plus a list of candidate axes labeled L1..LN."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        palette = style["palette"]
        polyA = Polygon(shape_a, closed=True,
                        facecolor=palette[0], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyA)
        cA = _centroid(shape_a)
        ax.text(cA[0], cA[1], "A", ha='center', va='center',
                fontsize=15, fontweight='bold', color='black')

        polyB = Polygon(shape_b, closed=True,
                        facecolor=palette[1 % len(palette)], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyB)
        cB = _centroid(shape_b)
        ax.text(cB[0], cB[1], "B", ha='center', va='center',
                fontsize=15, fontweight='bold', color='black')

        all_pts = list(shape_a) + list(shape_b)
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        # Include the axis lines in extent estimation
        for kind, val in axes:
            if kind == "vert":
                all_x.extend([val])
            elif kind == "horz":
                all_y.extend([val])
        margin = 3
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin

        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.35)

        # Draw labeled axes
        line_colors = [palette[2 % len(palette)],
                       palette[3 % len(palette)],
                       palette[4 % len(palette)],
                       palette[5 % len(palette)]]
        for i, (axis, lab) in enumerate(zip(axes, axis_labels)):
            col = line_colors[i % len(line_colors)]
            kind = axis[0]
            if kind == "vert":
                xv = axis[1]
                ax.axvline(xv, color=col, linewidth=2.0, linestyle='--',
                           alpha=0.85)
                ax.text(xv + 0.12, ymax - 0.6, lab, fontsize=14,
                        fontweight='bold', color=col)
            elif kind == "horz":
                yv = axis[1]
                ax.axhline(yv, color=col, linewidth=2.0, linestyle='--',
                           alpha=0.85)
                ax.text(xmax - 0.9, yv + 0.12, lab, fontsize=14,
                        fontweight='bold', color=col)
            elif kind == "diag_pos":
                b = axis[1]
                xs = np.linspace(xmin, xmax, 50)
                ax.plot(xs, xs + b, color=col, linewidth=2.0,
                        linestyle='--', alpha=0.85)
                ax.text(xmax - 1.5, xmax - 1.5 + b + 0.3, lab, fontsize=14,
                        fontweight='bold', color=col)
            else:
                b = axis[1]
                xs = np.linspace(xmin, xmax, 50)
                ax.plot(xs, -xs + b, color=col, linewidth=2.0,
                        linestyle='--', alpha=0.85)
                ax.text(xmin + 0.3, -(xmin) + b - 0.6, lab, fontsize=14,
                        fontweight='bold', color=col)

        ax.set_xticks(range(int(math.floor(xmin)), int(math.ceil(xmax)) + 1))
        ax.set_yticks(range(int(math.floor(ymin)), int(math.ceil(ymax)) + 1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=8)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _rotate_pts(pts, angle_deg, pivot):
        """Rotate pts counterclockwise by angle_deg about pivot."""
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        cx, cy = pivot
        out = []
        for (x, y) in pts:
            dx, dy = x - cx, y - cy
            nx = dx * cos_a - dy * sin_a + cx
            ny = dx * sin_a + dy * cos_a + cy
            out.append((round(nx, 3), round(ny, 3)))
        return out

    # ================================================================== #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """All answers in this rewrite are single-letter MCQ. Delegate to
        base class which already handles letter extraction."""
        return super()._check_answer(predicted, ground_truth)
