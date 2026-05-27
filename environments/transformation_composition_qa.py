"""
Transformation Composition QA environment.

2026-05-05 R5 P1: REWRITTEN to match WeMath Basic Transformations of Figures
benchmark verbatim style (Q1-Q20 from plan/research/wemath.md).

WeMath phrasing patterns matched:
  - L0/L1 → Q1-style 2-option direction MCQ:
    "As shown in the diagram, shape B is shape A rotated 90° around point O ( ).
     A. Clockwise; B. Counterclockwise; C. Cannot be determined; D. No correct answer"
    Ground truth: B
  - L2-L4 → Q4/Q9-style 4 or 5 option single-rotation MCQ around named point O:
    "As shown in the diagram, shape A is transformed into shape C by ______.
     A. Rotating 90° counterclockwise around point O; B. Rotating 90° clockwise
     around point O; C. Rotating 45° clockwise around point O; D. No correct answer"
    Ground truth: B
  - L5-L7 → Q12-style 2-step compound (rotate around named pivot + translate):
    "As shown in the diagram, Figure ① (    ) to obtain Figure ②.
     A. Rotate 90° counterclockwise around point P, then move 1 square to the left;
     B. Rotate 90° counterclockwise around point Q, then move 1 square to the left;
     C. Rotate 90° counterclockwise around point Q, then move 2 squares to the left;
     D. No correct answer"
    Ground truth: C
  - L8-L9 → Q12-style 2-step compound with tighter distractors + raised "No
    correct answer" trap rate (40%) — WeMath samples don't have explicit
    3-transform composition, so we keep 2-step but harder.

Naming conventions follow WeMath:
  - Single rotation: shape A / shape B / shape C, pivot called point O
  - Compound: Figure ① / Figure ②, pivots P / Q
  - Always grid-based with axes shown
  - All answers are single-letter MCQ (A through D or A through E)
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


def _rotate_points(pts, angle_deg, cx=0, cy=0):
    """Rotate points by angle_deg counterclockwise about (cx, cy).
    Pass negative angle for clockwise."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        rx = dx * cos_a - dy * sin_a + cx
        ry = dx * sin_a + dy * cos_a + cy
        out.append((round(rx, 3), round(ry, 3)))
    return out


def _translate_points(pts, tx, ty):
    return [(x + tx, y + ty) for x, y in pts]


def _centroid(pts):
    n = len(pts)
    cx = sum(x for x, y in pts) / n
    cy = sum(y for x, y in pts) / n
    return (cx, cy)


class TransformationCompositionQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "transformation_composition"

    QUESTION_TYPES = [
        "wemath_q1_direction",          # L0/L1 — Q1 style 2-option direction
        "wemath_q4q9_single_rotation",  # L2-L4 — Q4/Q9 style 4-5 option single rotation around point O
        "wemath_q12_compound",          # L5-L9 — Q12 style 2-step compound rotate+translate
    ]

    def _level_config(self, level: int) -> dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "qtype": "wemath_q1_direction",
                # L0: simple shape (right triangle); L1: same with mild variety
                "shape_pool": ["triangle"],
            }
        if level <= 4:
            return {
                "qtype": "wemath_q4q9_single_rotation",
                # L2/L3: 4-option (Q9 style); L4: 5-option (Q4 style)
                "use_5_options": (level == 4),
                "shape_pool": ["triangle", "rectangle"],
                "trap_rate": 0.0 if level <= 3 else 0.15,
            }
        if level <= 7:
            return {
                "qtype": "wemath_q12_compound",
                "shape_pool": ["triangle", "rectangle"],
                "use_5_options": False,  # Q12 itself is 4-option
                "trap_rate": 0.20,
                "tight_distractors": False,
            }
        # L8/L9 — same Q12 style but harder distractors + higher trap rate
        return {
            "qtype": "wemath_q12_compound",
            "shape_pool": ["triangle", "rectangle", "l_shape"],
            "use_5_options": (level == 9),  # mix in 5-option at L9
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        qtype = cfg["qtype"]
        for _ in range(20):
            if qtype == "wemath_q1_direction":
                result = self._try_generate_q1_direction(cfg, sub_rng)
            elif qtype == "wemath_q4q9_single_rotation":
                result = self._try_generate_q4q9_single_rotation(cfg, sub_rng)
            else:
                result = self._try_generate_q12_compound(cfg, sub_rng)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # L0/L1 — Q1 style verbatim
    #   "As shown in the diagram, shape B is shape A rotated 90° around
    #    point O ( ).  A. Clockwise; B. Counterclockwise;
    #    C. Cannot be determined; D. No correct answer"
    # ------------------------------------------------------------------ #
    def _try_generate_q1_direction(self, cfg, rng):
        # Asymmetric right triangle for unambiguous direction read
        shape_pts = [(1, 1), (4, 1), (1, 3)]
        # Pick rotation angle from {90, 180, 270} — Q1 fixes 90° but Q1-style
        # questions in the corpus use 90° most often. Keep 90° dominant; add
        # 180/270 occasionally so model does not memorize.
        angle = rng.choices([90, 180, 270], weights=[3, 1, 1], k=1)[0]
        # Pick true direction: CW or CCW
        true_dir = rng.choice(["CW", "CCW"])
        # Apply rotation around the origin O = (0, 0).
        # CCW = positive angle; CW = negative.
        signed_angle = (-angle) if true_dir == "CW" else angle
        rotated = _rotate_points(shape_pts, signed_angle, 0, 0)

        # Build option set — Q1 verbatim has 4 options:
        #   A. Clockwise; B. Counterclockwise;
        #   C. Cannot be determined; D. No correct answer
        # Order is fixed in WeMath Q1; we keep the canonical order to match
        # verbatim. Correct answer letter depends on true_dir.
        if true_dir == "CW":
            correct_letter = "A"
        else:
            correct_letter = "B"

        question = (
            f"As shown in the diagram, shape B is shape A rotated {angle}° "
            f"around point O (    ).\n"
            f"A. Clockwise\n"
            f"B. Counterclockwise\n"
            f"C. Cannot be determined\n"
            f"D. No correct answer\n"
            "Answer with a single letter A, B, C, or D."
        )
        image = self._draw_two_shapes_with_O(
            shape_a=shape_pts, shape_b=rotated, label_a="A", label_b="B",
            pivot_label="O", pivot_pos=(0, 0), rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L2-L4 — Q4/Q9 style verbatim
    #   Q9: "As shown in the diagram, shape A is transformed into shape C
    #        by ______.  A. Rotating 90° counterclockwise around point O;
    #        B. Rotating 90° clockwise around point O; C. Rotating 45°
    #        clockwise around point O; D. No correct answer"
    #   Q4: "As shown in the diagram, by rotating around the endpoint O,
    #        how can shape A be transformed into shape B?
    #        A. Rotate 90° clockwise; B. Rotate 90° counterclockwise;
    #        C. Rotate 45° counterclockwise; D. Rotate 45° clockwise;
    #        E. No correct answer"
    # ------------------------------------------------------------------ #
    def _try_generate_q4q9_single_rotation(self, cfg, rng):
        shape_type = rng.choice(cfg["shape_pool"])
        if shape_type == "triangle":
            pts = [(1, 1), (4, 1), (1, 3)]
        else:  # rectangle
            pts = [(2, 1), (4, 1), (4, 2), (2, 2)]

        # Choose true rotation: angle ∈ {90, 180, 270} (Q9-style multiples of 90)
        # plus a small chance of "trap" (true is none-of-the-above).
        # Direction: CW or CCW.
        true_angle = rng.choice([90, 180, 270])
        true_dir = rng.choice(["CW", "CCW"])
        signed = (-true_angle) if true_dir == "CW" else true_angle
        rotated = _rotate_points(pts, signed, 0, 0)

        # Decide phrasing variant
        # Variant Q9 ("shape A is transformed into shape C by ______"):
        #   options use participial form "Rotating 90° counterclockwise around point O"
        # Variant Q4 ("by rotating around the endpoint O, how can shape A
        #             be transformed into shape B?"):
        #   options use imperative form "Rotate 90° clockwise" (no "around O" repeat)
        use_q9_phrasing = rng.random() < 0.5
        use_5 = cfg["use_5_options"]

        # Build truth + distractors as (angle, direction) tuples.
        truth = (true_angle, true_dir)
        # All possible distractor candidates (other valid rotations)
        all_options_pool = []
        for ang in (45, 90, 180, 270):
            for d in ("CW", "CCW"):
                if (ang, d) != truth:
                    all_options_pool.append((ang, d))
        rng.shuffle(all_options_pool)

        # 4-option layout: 1 correct + 2 distractors + "No correct answer"
        # 5-option layout: 1 correct + 3 distractors + "No correct answer"
        n_distractor = 3 if use_5 else 2
        distractors = all_options_pool[:n_distractor]

        # Trap variant: replace correct with another distractor → answer = last letter
        use_trap = (rng.random() < cfg["trap_rate"])
        if use_trap:
            extra_distractor = next(
                (o for o in all_options_pool if o not in distractors), None,
            )
            if extra_distractor is None:
                use_trap = False  # not enough distractors; abort trap

        if use_trap:
            # All A-(D or D first letters) are wrong; correct = last letter
            options_data = [extra_distractor] + distractors
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
            correct_letter = opt_letters[-1]
        else:
            options_data = [truth] + distractors
            rng.shuffle(options_data)
            opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
            correct_letter = opt_letters[options_data.index(truth)]

        # Format options into text
        def _fmt(angle, direction, q9_style):
            d_word = "clockwise" if direction == "CW" else "counterclockwise"
            if q9_style:
                return f"Rotating {angle}° {d_word} around point O"
            return f"Rotate {angle}° {d_word}"

        opt_texts = [_fmt(a, d, use_q9_phrasing) for a, d in options_data]
        opt_texts.append("No correct answer")

        # Build question stem
        if use_q9_phrasing:
            stem = (
                "As shown in the diagram, shape A is transformed into "
                "shape B by ______."
            )
        else:
            stem = (
                "As shown in the diagram, by rotating around the endpoint "
                "O, how can shape A be transformed into shape B?"
            )

        opt_lines = "\n".join(
            f"{opt_letters[i] if i < len(opt_letters) else 'X'}. {t}"
            for i, t in enumerate(opt_texts)
        )
        # Letter list for instruction line
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._draw_two_shapes_with_O(
            shape_a=pts, shape_b=rotated, label_a="A", label_b="B",
            pivot_label="O", pivot_pos=(0, 0), rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L5-L9 — Q12 style verbatim
    #   "As shown in the diagram, Figure ① (    ) to obtain Figure ②.
    #    A. Rotate 90° counterclockwise around point P, then move 1 square
    #       to the left;
    #    B. Rotate 90° counterclockwise around point Q, then move 1 square
    #       to the left;
    #    C. Rotate 90° counterclockwise around point Q, then move 2 squares
    #       to the left;
    #    D. No correct answer"
    # ------------------------------------------------------------------ #
    def _try_generate_q12_compound(self, cfg, rng):
        shape_type = rng.choice(cfg["shape_pool"])
        if shape_type == "triangle":
            pts = [(1, 1), (3, 1), (1, 3)]
        elif shape_type == "rectangle":
            pts = [(1, 1), (3, 1), (3, 3), (1, 3)]
        else:  # l_shape
            pts = [(1, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3)]

        # True transform: rotation about pivot P or Q + translation
        # Q12 uses 90° CCW; we keep 90° CCW as anchor but mix in 90/180/270
        # with a CCW preference (Q12 is CCW). Most options stay CCW for
        # phrasing fidelity — angle is the main distractor axis.
        true_rot = rng.choice([90, 180, 270])
        true_pivot = rng.choice(["P", "Q"])
        pivot_coords = {"P": (0, 0), "Q": (5, 0)}
        true_dir = rng.choice(["left", "right", "up", "down"])
        true_steps = rng.randint(1, 3)

        truth = (true_rot, true_pivot, true_dir, true_steps)

        def _opt_text(rot, pivot, direction, steps):
            unit = "square" if steps == 1 else "squares"
            return (
                f"Rotate {rot}° counterclockwise around point {pivot}, "
                f"then move {steps} {unit} to the {direction}"
            )

        # Build distractor pool — each swaps exactly ONE field
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

        # Tight distractors at L8/L9: bias toward swap-by-pivot (subtler)
        if cfg["tight_distractors"]:
            tight = [d for d in distractor_pool if d[1] != true_pivot]
            other = [d for d in distractor_pool if d[1] == true_pivot]
            rng.shuffle(tight)
            rng.shuffle(other)
            distractor_pool = tight + other
        else:
            rng.shuffle(distractor_pool)

        # Q12 layout: 4 options total = 1 correct + 2 distractors + "No correct answer"
        # 5-option (mix at L9): 1 correct + 3 distractors + "No correct answer"
        use_5 = cfg["use_5_options"]
        n_distractor = 3 if use_5 else 2
        # Dedup textually
        chosen_distractors = []
        seen_texts = {_opt_text(*truth)}
        for d in distractor_pool:
            t = _opt_text(*d)
            if t not in seen_texts:
                chosen_distractors.append(d)
                seen_texts.add(t)
            if len(chosen_distractors) >= n_distractor + 1:
                break  # +1 spare for trap
        if len(chosen_distractors) < n_distractor:
            return None

        use_trap = (rng.random() < cfg["trap_rate"])
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]

        if use_trap and len(chosen_distractors) >= n_distractor + 1:
            # All non-final options are distractors (none equal truth)
            options_data = chosen_distractors[: n_distractor + 1]
            # Last is "No correct answer" placeholder
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]  # last letter = "No correct answer"
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

        # Render: apply true transform, draw both figures with pivots P and Q
        cx, cy = pivot_coords[true_pivot]
        rotated = _rotate_points(pts, true_rot, cx, cy)
        dx, dy = 0, 0
        if true_dir == "left":
            dx = -true_steps
        elif true_dir == "right":
            dx = true_steps
        elif true_dir == "up":
            dy = true_steps
        elif true_dir == "down":
            dy = -true_steps
        result_pts = _translate_points(rotated, dx, dy)

        image = self._draw_compound_figures(
            orig_pts=pts, result_pts=result_pts,
            pivot_coords=pivot_coords, rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_two_shapes_with_O(self, shape_a, shape_b, label_a, label_b,
                                pivot_label, pivot_pos, rng) -> Image.Image:
        """Draw shape_a and shape_b on a shared coordinate grid with pivot
        labeled (e.g. 'O'). Used for L0-L4 single-rotation MCQs."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        palette = style["palette"]
        # Shape A
        polyA = Polygon(shape_a, closed=True,
                        facecolor=palette[0], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyA)
        cA = _centroid(shape_a)
        ax.text(cA[0], cA[1], label_a,
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='black')
        # Shape B
        polyB = Polygon(shape_b, closed=True,
                        facecolor=palette[1 % len(palette)], edgecolor='black',
                        alpha=0.55, linewidth=2)
        ax.add_patch(polyB)
        cB = _centroid(shape_b)
        ax.text(cB[0], cB[1], label_b,
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='black')

        # Pivot point O
        px, py = pivot_pos
        ax.plot(px, py, marker='o', color='black', markersize=10, zorder=5)
        ax.annotate(pivot_label, xy=(px, py), xytext=(px + 0.3, py + 0.3),
                    fontsize=15, fontweight='bold', color='red')

        # Axes / grid
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
        """Draw Figure ① (orig) and Figure ② (after compound transform)
        with named pivots P and Q on a coordinate grid."""
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

    # ------------------------------------------------------------------ #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """All answers in this rewrite are single-letter MCQ. Delegate to
        base class which already handles letter extraction."""
        return super()._check_answer(predicted, ground_truth)
