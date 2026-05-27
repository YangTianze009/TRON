"""
Asymmetric Shape Rotation QA environment.

Template: rotation_identification_qa.py.

Goal: direct fix for spatial-vision 2DRotation . An
asymmetric 2D shape is rotated by a target angle; 4 candidate
rotations are shown and the model picks the correct one.

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): rotation_angle_granularity -> L0 {90,180,270},
                    L5 {45,90,135,...}, L9 {30,45,60,...,315}
  Axis 2           : distractor_generation_mode -> random -> near-rotation
  Axis 3 (optional): shape_complexity -> 4-vertex L at L0, 8+ vertices at L9

Output format is constant: 4-option MCQ, single letter.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_ANGLE_SETS = [
    [90, 180, 270],                                     # L0
    [90, 180, 270],                                     # L1
    [90, 180, 270, 45],                                 # L2
    [45, 90, 135, 180, 225, 270],                       # L3
    [45, 90, 135, 180, 225, 270, 315],                  # L4
    [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],  # L5
    [30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 270, 315],  # L6
    [30, 45, 60, 90, 120, 135, 150, 180, 225, 270, 315],   # L7
    [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 210, 225, 240, 270, 300, 315],  # L8
    [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345],  # L9
]

def _l_simple():
    return [(-2, -2), (1, -2), (1, -1), (-1, -1), (-1, 2), (-2, 2)]

def _flag():
    return [(-1, -2.5), (-1, 2.5), (2, 1.5), (0, 0.5), (2, -0.5), (-1, -1.5)]

def _arrow_big():
    return [(-2, 0.7), (0.5, 0.7), (0.5, 1.5), (2.5, 0),
            (0.5, -1.5), (0.5, -0.7), (-2, -0.7)]

def _notch_poly():
    """Triangle with notch — 7 vertices."""
    return [(0, 2.5), (2, -1.5), (0.8, -1.5), (0.8, -0.5),
            (0.3, -0.5), (0.3, -1.5), (-2, -1.5)]

def _poly_8():
    """8-vertex irregular polygon."""
    return [(0, 2.5), (1.5, 2.0), (2.0, 0.5), (1.2, -1.0),
            (0, -2.0), (-1.5, -1.2), (-2.0, 0.5), (-1.0, 2.0)]

def _poly_9_irreg():
    return [(0, 2.8), (1.2, 2.2), (2.0, 1.0), (1.5, -0.5),
            (0.8, -1.4), (-0.5, -1.8), (-1.6, -0.8), (-2.0, 0.8), (-1.0, 2.0)]

_SHAPE_BANK = {
    "L-shape": _l_simple,
    "flag": _flag,
    "arrow": _arrow_big,
    "notched triangle": _notch_poly,
    "irregular octagon": _poly_8,
    "irregular nonagon": _poly_9_irreg,
}

def _rotate(verts, deg):
    rad = math.radians(deg)
    return [(x * math.cos(rad) - y * math.sin(rad),
             x * math.sin(rad) + y * math.cos(rad)) for (x, y) in verts]

def _mirror_y(verts):
    return [(-x, y) for (x, y) in verts]

class AsymmetricShapeRotationQA(StandaloneVisualEnv):
    ENV_NAME = "asymmetric_shape_rotation"

    _QUESTION_TEMPLATES = [
        "The figure on the left is a {shape}. It has been rotated by exactly {angle} degrees counter-clockwise about its centroid (no flipping allowed). Which of the four options on the right shows the rotated shape? Answer with a single letter (A, B, C, or D).",
        "A {shape} has been rotated {angle} degrees CCW. Which option (A-D) shows the correct result? No reflection is applied. Answer with a single letter.",
        "Identify the {shape} after a {angle}-degree counter-clockwise rotation. Pick the matching option A, B, C, or D. Answer with a single letter.",
        "The {shape} shown on the left is rotated {angle} degrees counter-clockwise. Which option matches? Answer A, B, C, or D.",
    ]

    _TITLE_VARIANTS = [
        "Shape rotation",
        "Rotation match",
        "Rotate & identify",
        "Orientation",
        "CCW rotation",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "angle_set": _ANGLE_SETS[level],
            "distractor_mode": "random" if level <= 2 else (
                "mirror_near" if level <= 5 else "perturbation"),
            "shape_pool": (["L-shape", "flag", "arrow"] if level <= 3 else
                           (["flag", "arrow", "notched triangle"] if level <= 6 else
                            ["notched triangle", "irregular octagon",
                             "irregular nonagon"])),
            "tight_eps": max(5, 20 - level * 2),   # distractor angle offset window
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = len(cfg["angle_set"])

        shape_name = rng.choice(cfg["shape_pool"])
        base = _SHAPE_BANK[shape_name]()
        angle = rng.choice(cfg["angle_set"])
        target = _rotate(base, angle)

        # Build 4 candidate rotations
        candidates = [("correct", target, angle)]
        angle_options_rest = [a for a in cfg["angle_set"] if a != angle]
        rng.shuffle(angle_options_rest)

        if cfg["distractor_mode"] == "random" and level <= 1:
            # 2026-04-25: at L0/L1 use DIFFERENT shapes as distractors (not
            # rotated variants of the same shape) — model otherwise can't
            # distinguish similar polygon orientations.
            other_shapes = [s for s in cfg["shape_pool"] if s != shape_name]
            rng.shuffle(other_shapes)
            for sname in other_shapes[:3]:
                d_base = _SHAPE_BANK[sname]()
                d_angle = rng.choice(cfg["angle_set"])
                candidates.append(("other_shape", _rotate(d_base, d_angle), d_angle))
            # Fill remainder with rotated other angles if we don't have enough
            for a in angle_options_rest:
                if len(candidates) >= 4: break
                candidates.append(("rot", _rotate(base, a), a))
        elif cfg["distractor_mode"] == "random":
            for a in angle_options_rest[:3]:
                candidates.append(("rot", _rotate(base, a), a))
        elif cfg["distractor_mode"] == "mirror_near":
            # mirror
            candidates.append(("mirror", _mirror_y(base), 0))
            for a in angle_options_rest[:2]:
                candidates.append(("rot", _rotate(base, a), a))
        else:
            # perturbation: angle ± small eps
            eps = cfg["tight_eps"]
            near1 = (angle + eps) % 360
            near2 = (angle - eps) % 360
            candidates.append(("rot", _rotate(base, near1), near1))
            candidates.append(("rot", _rotate(base, near2), near2))
            candidates.append(("mirror", _rotate(_mirror_y(base), angle), angle))

        if len(candidates) < 4:
            for a in angle_options_rest:
                if len(candidates) >= 4:
                    break
                candidates.append(("rot", _rotate(base, a), a))
        candidates = candidates[:4]

        order = list(range(4))
        rng.shuffle(order)
        shuffled = [candidates[i] for i in order]
        correct_letter = chr(ord("A") + order.index(0))

        question = rng.choice(self._QUESTION_TEMPLATES).format(
            shape=shape_name, angle=angle)
        if level <= 2:
            question += (
                f" Hint: track an asymmetric feature of the {shape_name} "
                f"(e.g., the protruding tip or notch). Apply the {angle}° "
                "rotation algebraically: 90° CCW maps (x,y)→(-y,x); 180°: "
                "(x,y)→(-x,-y); 270° CCW: (x,y)→(y,-x). Find which option "
                "has the feature in the predicted position."
            )

        title = rng.choice(self._TITLE_VARIANTS)
        image = self._render(base, shape_name, shuffled, title=title)
        return question, correct_letter, image

    def _render(self, base, shape_name, candidates, title="Shape rotation"):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(10 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        palette = style["palette"]
        ax_left = fig.add_subplot(1, 5, 1)
        ax_left.set_facecolor(style["bg_color"])
        self._draw(ax_left, base, palette[0], f"Original")

        labels = ["A", "B", "C", "D"]
        for i, (kind, verts, _a) in enumerate(candidates):
            ax = fig.add_subplot(1, 5, 2 + i)
            ax.set_facecolor(style["bg_color"])
            self._draw(ax, verts, palette[(i + 1) % len(palette)], f"({labels[i]})")

        fig.suptitle(title,
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw(self, ax, verts, color, title):
        poly = plt.Polygon(verts, closed=True, facecolor=color,
                           alpha=0.55, edgecolor="#222", linewidth=1.6)
        ax.add_patch(poly)
        # Red dot at first vertex as reference
        ax.plot(verts[0][0], verts[0][1], "o", color="#e74c3c",
                markersize=6)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        mg = 1.0
        ax.set_xlim(min(xs) - mg, max(xs) + mg)
        ax.set_ylim(min(ys) - mg, max(ys) + mg)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = AsymmetricShapeRotationQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[asymmetric_shape_rotation] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/asymmetric_shape_rotation_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | n_angles={env._primary_complexity_feature}")
