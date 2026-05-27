"""
Rotation / Transformation Identification QA environment.

Shows "Before" and "After" images of a shape that has been transformed.
Question types: identify rotation angle, identify reflection axis,
identify translation vector, identify composed transformations.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 90° rotation MCQ of arrow shape. 4 options: 90°, 180°, 270°, + 1 distractor.
L1: 90/180/270° rotation MCQ of any of 4 asymmetric shapes. 4 options.
L2: 90/180/270° rotation free-form (no MCQ).
L3: reflection MCQ over x/y axis.
L4: rotation or reflection MCQ.
L5: reflection free-form, all 4 axes.
L6: translation, small vector.
L7: translation + rotation/reflection mix.
L8: composite transformations MCQ.
L9: composite with larger option set.

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = {
    "rot": ["Rotation about origin", "Rotated shape", "Rotation"],
    "ref": ["Reflection", "Reflected shape", "Mirror reflection"],
    "tra": ["Translation", "Translated shape", "Shape translation"],
    "comp": ["Composite Transformation", "Two-step transformation",
             "Sequence of transformations"],
}

def _make_l_shape(cx=0, cy=0, scale=1.0, rotation_deg=0):
    raw = [
        (-2, -2), (1, -2), (1, -1), (-1, -1), (-1, 2), (-2, 2),
    ]
    rad = math.radians(rotation_deg)
    verts = []
    for x, y in raw:
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        verts.append((cx + rx * scale, cy + ry * scale))
    return verts

def _make_arrow(cx=0, cy=0, scale=1.0, rotation_deg=0):
    raw = [
        (-2, 0.7), (0.5, 0.7), (0.5, 1.5), (2.5, 0),
        (0.5, -1.5), (0.5, -0.7), (-2, -0.7),
    ]
    rad = math.radians(rotation_deg)
    verts = []
    for x, y in raw:
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        verts.append((cx + rx * scale, cy + ry * scale))
    return verts

def _make_flag(cx=0, cy=0, scale=1.0, rotation_deg=0):
    raw = [
        (-1, -2.5), (-1, 2.5), (2, 1.5), (0, 0.5), (2, -0.5), (-1, -1.5),
    ]
    rad = math.radians(rotation_deg)
    verts = []
    for x, y in raw:
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        verts.append((cx + rx * scale, cy + ry * scale))
    return verts

def _make_triangle_with_notch(cx=0, cy=0, scale=1.0, rotation_deg=0):
    raw = [
        (0, 2.5), (2, -1.5), (0.8, -1.5), (0.8, -0.5),
        (0.3, -0.5), (0.3, -1.5), (-2, -1.5),
    ]
    rad = math.radians(rotation_deg)
    verts = []
    for x, y in raw:
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        verts.append((cx + rx * scale, cy + ry * scale))
    return verts

_SHAPE_FNS = [
    ("L-shape", _make_l_shape),
    ("arrow", _make_arrow),
    ("flag", _make_flag),
    ("triangle with notch", _make_triangle_with_notch),
]

_L0_SHAPES = [("arrow", _make_arrow), ("L-shape", _make_l_shape)]

def _rotate_points(verts, angle_deg, pivot=(0, 0)):
    rad = math.radians(angle_deg)
    px, py = pivot
    out = []
    for x, y in verts:
        dx, dy = x - px, y - py
        rx = dx * math.cos(rad) - dy * math.sin(rad) + px
        ry = dx * math.sin(rad) + dy * math.cos(rad) + py
        out.append((rx, ry))
    return out

def _reflect_points(verts, axis):
    if axis == "x":
        return [(x, -y) for x, y in verts]
    elif axis == "y":
        return [(-x, y) for x, y in verts]
    elif axis == "y=x":
        return [(y, x) for x, y in verts]
    elif axis == "y=-x":
        return [(-y, -x) for x, y in verts]
    return verts

def _translate_points(verts, dx, dy):
    return [(x + dx, y + dy) for x, y in verts]

class RotationIdentificationQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "rotation_identification"

    QUESTION_TYPES = [
        "identify_rotation",
        "identify_reflection",
        "identify_translation",
        "identify_rotation_mc",
        "identify_reflection_mc",
        "composite_transform",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"], weights=cfg["qtype_weights"])[0]
        for _ in range(15):
            result = self._dispatch(qtype, level, cfg)
            if result is not None:
                self._primary_complexity_feature = level * 5 + len(result[1])
                return result
        return None

    def _level_config(self, level: int) -> Dict:
        # Reordered by empirical model difficulty:
        # Reflection MCQ (easiest) -> Rotation MCQ -> mixed MCQ -> free-form -> composite
        if level == 0:
            # Reflection MCQ (model finds this easiest: 0.80 before)
            return {"qtypes": ["identify_reflection_mc"], "qtype_weights": [1],
                    "shapes": _L0_SHAPES, "axes": ["x", "y"]}
        if level == 1:
            return {"qtypes": ["identify_reflection_mc"], "qtype_weights": [1],
                    "shapes": _SHAPE_FNS, "axes": ["x", "y"]}
        if level == 2:
            return {"qtypes": ["identify_reflection_mc", "identify_rotation_mc"],
                    "qtype_weights": [5, 5],
                    "shapes": _SHAPE_FNS,
                    "angles": [180], "axes": ["x", "y"]}
        if level == 3:
            return {"qtypes": ["identify_rotation_mc"], "qtype_weights": [1],
                    "shapes": _SHAPE_FNS, "angles": [90, 180, 270]}
        if level == 4:
            return {"qtypes": ["identify_rotation_mc", "identify_reflection_mc"],
                    "qtype_weights": [5, 5],
                    "shapes": _SHAPE_FNS,
                    "angles": [90, 180, 270],
                    "axes": ["x", "y", "y=x", "y=-x"]}
        if level == 5:
            # Free-form rotation angle
            return {"qtypes": ["identify_rotation"], "qtype_weights": [1],
                    "shapes": _SHAPE_FNS, "angles": [90, 180, 270]}
        if level == 6:
            return {"qtypes": ["identify_rotation", "identify_reflection"],
                    "qtype_weights": [5, 5],
                    "shapes": _SHAPE_FNS,
                    "angles": [90, 180, 270],
                    "axes": ["x", "y", "y=x", "y=-x"]}
        if level == 7:
            return {"qtypes": ["identify_rotation", "identify_reflection",
                               "identify_translation"],
                    "qtype_weights": [3, 3, 4],
                    "shapes": _SHAPE_FNS,
                    "angles": [90, 180, 270],
                    "axes": ["x", "y", "y=x", "y=-x"],
                    "trans_range": [-5, 5]}
        if level == 8:
            return {"qtypes": ["composite_transform"], "qtype_weights": [1],
                    "shapes": _SHAPE_FNS, "n_options": 4}
        return {"qtypes": ["composite_transform"], "qtype_weights": [1],
                "shapes": _SHAPE_FNS, "n_options": 6}

    def _dispatch(self, qtype, level, cfg):
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )
        if qtype == "identify_rotation":
            return self._rotation_problem(sub_rng, cfg, mc=False)
        if qtype == "identify_rotation_mc":
            return self._rotation_problem(sub_rng, cfg, mc=True)
        if qtype == "identify_reflection":
            return self._reflection_problem(sub_rng, cfg, mc=False)
        if qtype == "identify_reflection_mc":
            return self._reflection_problem(sub_rng, cfg, mc=True)
        if qtype == "identify_translation":
            return self._translation_problem(sub_rng, cfg)
        if qtype == "composite_transform":
            return self._composite_problem(sub_rng, cfg)
        return None

    def _rotation_problem(self, rng, cfg, mc=False):
        name, shape_fn = rng.choice(cfg["shapes"])
        angle = rng.choice(cfg["angles"])
        before = shape_fn()
        after = _rotate_points(before, angle)

        if mc:
            # Build 4 options: canonical angles first
            options = [90, 180, 270]
            # Add one distractor
            candidates = [45, 60, 120, 135, 150, 210, 240, 300, 315, 360]
            rng.shuffle(candidates)
            for c in candidates:
                if c not in options:
                    options.append(c)
                    break
            if angle not in options:
                options.append(angle)
            options = sorted(set(options))[:4]
            if angle not in options:
                # Should not happen but guard anyway
                options[0] = angle
            rng.shuffle(options)
            correct_idx = options.index(angle)
            correct_letter = chr(ord("A") + correct_idx)
            opt_str = ", ".join(
                f"{chr(ord('A') + i)}) {v}\u00b0" for i, v in enumerate(options))
            stems = [
                f"A {name} has been rotated counter-clockwise about the origin. "
                f"The 'Before' shape (blue) and 'After' shape (red) are shown. "
                f"What is the rotation angle? {opt_str}. Answer with a single letter.",
                f"The blue shape ({name}) is rotated CCW about the origin to give the red shape. "
                f"Which angle was used? {opt_str}. Answer with a single letter.",
            ]
            question = rng.choice(stems)
            answer = correct_letter
        else:
            stems = [
                f"A {name} has been rotated counter-clockwise about the origin. "
                f"The 'Before' shape (blue) and 'After' shape (red) are shown. "
                f"What is the rotation angle in degrees? Answer with a number (e.g., 90, 180, 270).",
                f"By how many degrees CCW has the blue shape ({name}) been rotated to obtain the red shape? "
                f"Answer with a number only.",
            ]
            question = rng.choice(stems)
            answer = str(angle)

        title = rng.choice(_TITLE_VARIANTS["rot"])
        image = self._render_before_after(rng, before, after, title)
        return question, answer, image

    def _reflection_problem(self, rng, cfg, mc=False):
        name, shape_fn = rng.choice(cfg["shapes"])
        axis = rng.choice(cfg["axes"])
        before = shape_fn()
        after = _reflect_points(before, axis)

        axis_display = {
            "x": "x-axis", "y": "y-axis",
            "y=x": "line y = x", "y=-x": "line y = -x",
        }

        if mc:
            axes = ["x-axis", "y-axis", "line y = x", "line y = -x"]
            correct = axis_display[axis]
            rng.shuffle(axes)
            correct_idx = axes.index(correct)
            correct_letter = chr(ord("A") + correct_idx)
            opt_str = ", ".join(f"{chr(ord('A') + i)}) {v}"
                                for i, v in enumerate(axes))
            stems = [
                f"A {name} has been reflected. "
                f"The 'Before' shape (blue) and 'After' shape (red) are shown. "
                f"About which axis was it reflected? {opt_str}. Answer with a single letter.",
                f"The blue shape was reflected to give the red shape. "
                f"Identify the axis of reflection. {opt_str}. Answer with a single letter.",
            ]
            question = rng.choice(stems)
            answer = correct_letter
        else:
            stems = [
                f"A {name} has been reflected. The 'Before' shape (blue) and "
                f"'After' shape (red) are shown. About which axis was it reflected? "
                f"Answer: x-axis, y-axis, y=x, or y=-x.",
            ]
            question = rng.choice(stems)
            answer = {"x": "x-axis", "y": "y-axis",
                      "y=x": "y=x", "y=-x": "y=-x"}[axis]

        title = rng.choice(_TITLE_VARIANTS["ref"])
        # DO NOT draw the reflection axis when asking "which axis?" — that
        # would be direct answer leakage. Only draw it when the question is
        # NOT about identifying the axis (e.g., scaffolding / warmup).
        ask_for_axis = ("which axis" in question.lower()
                        or "axis of reflection" in question.lower())
        image = self._render_before_after(rng, before, after, title,
                                          show_axis=None if ask_for_axis else axis)
        return question, answer, image

    def _translation_problem(self, rng, cfg):
        name, shape_fn = rng.choice(cfg["shapes"])
        tr = cfg.get("trans_range", [-3, 3])
        choices = [v for v in range(tr[0], tr[1] + 1) if v != 0]
        dx = rng.choice(choices)
        dy = rng.choice(choices)
        before = shape_fn()
        after = _translate_points(before, dx, dy)

        stems = [
            f"A {name} has been translated. The 'Before' shape (blue) and 'After' shape (red) are shown. "
            f"What is the translation vector? Answer as (dx,dy), e.g. (3,-2).",
            f"The blue {name} was translated to give the red shape. What translation vector was applied? "
            f"Answer as (dx,dy).",
        ]
        question = rng.choice(stems)
        answer = f"({dx},{dy})"

        title = rng.choice(_TITLE_VARIANTS["tra"])
        image = self._render_before_after(rng, before, after, title,
                                          show_translation=(dx, dy))
        return question, answer, image

    def _composite_problem(self, rng, cfg):
        name, shape_fn = rng.choice(cfg["shapes"])
        before = shape_fn()

        t1_type = rng.choice(["rotate", "reflect"])
        t2_type = rng.choice(["rotate", "reflect"])
        steps = []
        current = list(before)

        if t1_type == "rotate":
            a1 = rng.choice([90, 180, 270])
            current = _rotate_points(current, a1)
            steps.append(f"rotation {a1}\u00b0 CCW")
        else:
            ax1 = rng.choice(["x", "y"])
            current = _reflect_points(current, ax1)
            steps.append(f"reflection over {'x-axis' if ax1 == 'x' else 'y-axis'}")

        if t2_type == "rotate":
            a2 = rng.choice([90, 180, 270])
            current = _rotate_points(current, a2)
            steps.append(f"rotation {a2}\u00b0 CCW")
        else:
            ax2 = rng.choice(["x", "y"])
            current = _reflect_points(current, ax2)
            steps.append(f"reflection over {'x-axis' if ax2 == 'x' else 'y-axis'}")

        after = current
        correct_desc = f"{steps[0]}, then {steps[1]}"

        n_options = cfg.get("n_options", 4)
        options = [correct_desc]
        guard = 0
        while len(options) < n_options and guard < 200:
            guard += 1
            fake_steps = []
            for _ in range(2):
                if rng.random() < 0.5:
                    fake_steps.append(f"rotation {rng.choice([90, 180, 270])}\u00b0 CCW")
                else:
                    fake_steps.append(f"reflection over {rng.choice(['x-axis', 'y-axis'])}")
            candidate = f"{fake_steps[0]}, then {fake_steps[1]}"
            if candidate not in options:
                options.append(candidate)

        rng.shuffle(options)
        correct_idx = options.index(correct_desc)
        answer_letter = chr(ord("A") + correct_idx)
        opt_str = "\n".join(f"  {chr(ord('A') + i)}) {v}"
                            for i, v in enumerate(options))
        question = (
            f"A {name} has undergone TWO transformations in sequence. "
            f"The 'Before' shape (blue) and final 'After' shape (red) are shown. "
            f"Which sequence of transformations was applied?\n{opt_str}\n"
            f"Answer with a single letter."
        )
        title = rng.choice(_TITLE_VARIANTS["comp"])
        image = self._render_before_after(rng, before, after, title)
        return question, answer_letter, image

    def _render_before_after(self, rng, before, after, title,
                             show_axis=None, show_translation=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        all_pts = list(before) + list(after)
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = 2.5
        x_lo, x_hi = min(xs) - margin, max(xs) + margin
        y_lo, y_hi = min(ys) - margin, max(ys) + margin
        x_lo, y_lo = min(x_lo, -1), min(y_lo, -1)
        x_hi, y_hi = max(x_hi, 1), max(y_hi, 1)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.axhline(0, color="#2c3e50", linewidth=1.2, zorder=1)
        ax.axvline(0, color="#2c3e50", linewidth=1.2, zorder=1)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

        poly_b = plt.Polygon(before, closed=True, facecolor="#3498db",
                             alpha=0.4, edgecolor="#2980b9", linewidth=2.5,
                             zorder=3, label="Before")
        ax.add_patch(poly_b)
        for i, (x, y) in enumerate(before):
            ax.plot(x, y, "o", color="#2980b9", markersize=5, zorder=4)
            if i == 0:
                ax.text(x + 0.2, y + 0.2, "P", fontsize=10, color="#2980b9",
                        fontweight="bold", zorder=5)

        poly_a = plt.Polygon(after, closed=True, facecolor="#e74c3c",
                             alpha=0.4, edgecolor="#c0392b", linewidth=2.5,
                             zorder=3, label="After")
        ax.add_patch(poly_a)
        for i, (x, y) in enumerate(after):
            ax.plot(x, y, "o", color="#c0392b", markersize=5, zorder=4)
            if i == 0:
                ax.text(x + 0.2, y + 0.2, "P'", fontsize=10, color="#c0392b",
                        fontweight="bold", zorder=5)

        if show_axis:
            lo = min(x_lo, y_lo) - 1
            hi = max(x_hi, y_hi) + 1
            if show_axis == "x":
                ax.axhline(0, color="#e67e22", linewidth=2, linestyle="--",
                           alpha=0.6, zorder=2)
            elif show_axis == "y":
                ax.axvline(0, color="#e67e22", linewidth=2, linestyle="--",
                           alpha=0.6, zorder=2)
            elif show_axis == "y=x":
                ax.plot([lo, hi], [lo, hi], color="#e67e22", linewidth=2,
                        linestyle="--", alpha=0.6, zorder=2)
            elif show_axis == "y=-x":
                ax.plot([lo, hi], [-lo, -hi], color="#e67e22", linewidth=2,
                        linestyle="--", alpha=0.6, zorder=2)

        if show_translation:
            dx, dy = show_translation
            bcx = sum(p[0] for p in before) / len(before)
            bcy = sum(p[1] for p in before) / len(before)
            ax.annotate("", xy=(bcx + dx, bcy + dy), xytext=(bcx, bcy),
                        arrowprops=dict(arrowstyle="->", color="#e67e22",
                                        lw=2.5), zorder=6)
            ax.text(bcx + dx / 2 + 0.3, bcy + dy / 2 + 0.3,
                    f"({dx},{dy})", fontsize=11, color="#e67e22",
                    fontweight="bold")

        fs = style["font_size_base"]
        ax.legend(fontsize=fs - 1, loc="upper left")
        ax.set_title(title, fontsize=fs + 3, fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        predicted = predicted.strip().lower().rstrip(".")
        ground_truth = ground_truth.strip().lower().rstrip(".")
        if predicted == ground_truth:
            return True
        pred_clean = predicted.replace(" ", "").replace("(", "").replace(")", "")
        gt_clean = ground_truth.replace(" ", "").replace("(", "").replace(")", "")
        if pred_clean == gt_clean:
            return True
        try:
            p_val = float(predicted.replace("\u00b0", "").replace("degrees", "").strip())
            g_val = float(ground_truth.replace("\u00b0", "").replace("degrees", "").strip())
            return abs(p_val - g_val) < 1.0
        except (ValueError, AttributeError):
            pass
        if len(ground_truth) == 1 and ground_truth.upper() in "ABCDEF":
            import re
            mcq_match = re.search(r'\b([a-fA-F])\b', predicted)
            if mcq_match:
                return mcq_match.group(1).upper() == ground_truth.upper()
        return False

if __name__ == "__main__":
    env = RotationIdentificationQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
