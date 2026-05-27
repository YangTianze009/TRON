"""
Composite 3-Step Transformation QA.

A point P = (x, y) is shown on a coordinate grid. Three transformations are
applied in sequence (the sequence is given in the figure caption):
  1. Translation by (dx, dy):    (x, y) -> (x + dx, y + dy)
  2. Rotation 90 degrees clockwise about the origin:
                                 (x, y) -> (y, -x)
  3. Reflection about the x-axis: (x, y) -> (x, -y)

Levels vary the magnitudes of the start point and the translation. The base
sequence is fixed (translate -> rotate90CW -> reflect-x) for L0..L4 so the
problem reduces to a clean composition of three tiny rules; at higher levels
we vary the rotation direction and the reflection axis.

Output: final coordinates as `(x', y')` after all three steps.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "As shown in the figure, point P is marked on the coordinate grid. Apply the three sequential transformations listed in the title in order. What are the final coordinates of P?",
    "As shown in the diagram, point P is on the grid. Apply the three transformations shown in the title (in order). What are the resulting coordinates of P?",
    "As shown in the figure, the point P is plotted on the grid. After applying the three transformations listed in the title in order, what are the final coordinates of P?",
    "As shown in the diagram, P is on the coordinate grid. Compose the three transformations in the title and report P's final coordinates.",
    "As shown in the figure, point P is shown on the grid. After the three sequential transformations described in the title, what are the coordinates of P?",
    "As shown in the diagram, point P is at the marked location. After the listed three transformations are applied in order, what are the coordinates of the image of P?",
    "As shown in the figure, the marked point P is on the grid. After applying the three transformations listed in the title (in order), what are P's final coordinates?",
    "As shown in the diagram, P is shown on the coordinate grid. Compose the three transformations from the title; what are the final coordinates of P?",
]


def _parse_point(s):
    """Parse a coordinate pair like '(3, -4)', '(3,-4)', '3, -4', '[3, -4]'.
    Returns (x, y) tuple of floats or None."""
    import re
    if s is None:
        return None
    txt = s.strip()
    # Strip braces
    for ch in '{}[]()':
        txt = txt.replace(ch, ' ')
    # Find two signed numeric tokens
    nums = re.findall(r'-?\d+\.?\d*', txt)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except ValueError:
            return None
    return None


def _format_point(x, y):
    """Compact integer-or-float coord pair."""
    def _fmt(v):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"
    return f"({_fmt(x)}, {_fmt(y)})"


class Composite3StepTransformationQA(StandaloneVisualEnv):
    ENV_NAME = "composite_3step_transformation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 9:
            # L9: larger coordinate range so each transformation step has
            # bigger numbers to track. Same 3-step composition (translate →
            # rotate → reflect) but with the full operator pool and wider
            # value ranges than L7/L8.
            # 2026-05-04: bumped L9 difficulty (was 97.5% saturated → much
            # wider point + translation ranges so coordinate magnitudes hit
            # 50+, breaking small-number-pattern matching).
            # 2026-05-04 R3: benchmark-sample-driven harden — enable
            # "E. No correct answer" trap (~25% chance E is GT).
            return {
                "p_range": 50,
                "t_range": 40,
                "rotations": ["cw90", "ccw90", "180"],
                "reflections": ["x", "y", "y=x"],
                "e_trap": True,
            }
        if level <= 2:
            return {
                "p_range": 2 + level,
                "t_range": 1 + level,
                "rotations": ["cw90"],
                "reflections": ["x"],
                "e_trap": False,
            }
        if level <= 5:
            return {
                "p_range": 3 + level,
                "t_range": 2 + level,
                "rotations": ["cw90", "ccw90"],
                "reflections": ["x", "y"],
                "e_trap": False,
            }
        # 2026-05-04 R3: benchmark-sample-driven harden — at L8 enable
        # the "E. No correct answer" trap. Multi-step style: every Q has
        # option E and ~10-40% have E as correct.
        return {
            "p_range": 4 + level,
            "t_range": 3 + level,
            "rotations": ["cw90", "ccw90", "180"],
            "reflections": ["x", "y", "y=x"],
            "e_trap": level >= 8,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1069 + level * 103 + 7)

        # Sample starting point and translation
        for _ in range(20):
            px = rng.randint(-cfg["p_range"], cfg["p_range"])
            py = rng.randint(-cfg["p_range"], cfg["p_range"])
            dx = rng.randint(-cfg["t_range"], cfg["t_range"])
            dy = rng.randint(-cfg["t_range"], cfg["t_range"])
            if (dx, dy) == (0, 0):
                continue
            break
        else:
            return None

        # Pick rotation and reflection variant for this level
        rot = rng.choice(cfg["rotations"])
        ref = rng.choice(cfg["reflections"])

        # Apply transformations: translate -> rotate -> reflect
        x1, y1 = px + dx, py + dy
        if rot == "cw90":
            x2, y2 = y1, -x1
            rot_desc = "rotate 90 degrees clockwise about the origin"
        elif rot == "ccw90":
            x2, y2 = -y1, x1
            rot_desc = "rotate 90 degrees counter-clockwise about the origin"
        else:  # 180
            x2, y2 = -x1, -y1
            rot_desc = "rotate 180 degrees about the origin"

        if ref == "x":
            x3, y3 = x2, -y2
            ref_desc = "reflect about the x-axis"
        elif ref == "y":
            x3, y3 = -x2, y2
            ref_desc = "reflect about the y-axis"
        else:  # y = x
            x3, y3 = y2, x2
            ref_desc = "reflect about the line y = x"

        # 2026-05-05 R5 B3B HARDEN: stronger distractors. Compute "wrong-
        # order" coords (apply transforms in reverse order: reflect→rotate→
        # translate). Drop weak off-by-one. Add "wrong rotation direction"
        # and "wrong reflection axis" coords — these all share the model's
        # 3-step plan but flip exactly one mistake.
        correct = (x3, y3)
        # Wrong order: reflect first, then rotate, then translate
        if ref == "x":
            rx0, ry0 = px, -py
        elif ref == "y":
            rx0, ry0 = -px, py
        else:
            rx0, ry0 = py, px
        if rot == "cw90":
            rx1, ry1 = ry0, -rx0
        elif rot == "ccw90":
            rx1, ry1 = -ry0, rx0
        else:
            rx1, ry1 = -rx0, -ry0
        d_wrong_order = (rx1 + dx, ry1 + dy)

        # Mistake variants:
        d_swap = (y3, x3)                         # swapped x/y
        d_neg_x = (-x3, y3)                       # forgot reflection
        d_neg_y = (x3, -y3)                       # forgot one sign flip
        d_no_trans = (x2, y2)                    # forgot reflection step
        d_no_rot = (x1, y1)                      # only translated
        candidate_pool = [d_wrong_order, d_swap, d_neg_x, d_neg_y,
                          d_no_trans, d_no_rot]
        seen = {correct}
        distractors = []
        for d in candidate_pool:
            if d not in seen:
                distractors.append(d)
                seen.add(d)
            if len(distractors) >= 4:
                break
        if len(distractors) < 3:
            return None
        # 2026-05-05 R5 B3B HARDEN: bumped trap rate to 40% at L8/L9
        # (was 25% in R3). Also tightened distractors above.
        # 2026-05-05 Phase A: bump to 50% (still saturated 97.5% post-R5).
        if cfg.get("e_trap") and len(distractors) >= 4 and rng.random() < 0.50:
            opts_pts = distractors[:4]
            rng.shuffle(opts_pts)
            answer = "E"
        else:
            opts_pts = [correct] + distractors[:3]
            rng.shuffle(opts_pts)
            correct_idx = opts_pts.index(correct)
            answer = "ABCD"[correct_idx]
        opts_lines = [f"{chr(ord('A')+i)}. {_format_point(*opts_pts[i])}"
                      for i in range(4)]
        opts_lines.append("E. No correct answer")

        sidx = (self.seed or 0) % len(_TEMPLATES)
        stem = _TEMPLATES[sidx]
        question = (
            f"{stem}\n\n"
            + "\n".join(opts_lines)
            + "\n\nChoose the correct option (A, B, C, D, or E)."
        )
        title = (f"Steps: 1) translate by ({dx:+d}, {dy:+d})  "
                 f"2) {rot_desc}  3) {ref_desc}")
        img = self._render(px, py, title)
        return question, answer, img

    # ---------------- answer check ---------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        gp = _parse_point(ground_truth)
        pp = _parse_point(predicted)
        if gp is not None and pp is not None:
            return abs(gp[0] - pp[0]) < 1e-3 and abs(gp[1] - pp[1]) < 1e-3
        return super()._check_answer(predicted, ground_truth)

    # ---------------- render ---------------- #
    def _render(self, px, py, title) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        lim = max(8, abs(px) + 3, abs(py) + 3)

        # Grid
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.axhline(0, color="#000000", linewidth=1.0, zorder=2)
        ax.axvline(0, color="#000000", linewidth=1.0, zorder=2)

        # Mark P
        ax.plot([px], [py], "o", color="#c0392b", markersize=10, zorder=5)
        ax.annotate(f"P = ({px:g}, {py:g})", (px, py),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=12, color="#c0392b", fontweight="bold")
        # Origin marker
        ax.plot([0], [0], "o", color="#000000", markersize=4, zorder=4)
        ax.annotate("O", (0, 0), xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=11, color="#000000", fontweight="bold")

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(range(-lim, lim + 1))
        ax.set_yticks(range(-lim, lim + 1))
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # The transformation sequence is in the title — this is the spec
        ax.set_title(title, fontsize=10)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
