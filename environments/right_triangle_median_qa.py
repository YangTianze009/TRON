"""
Right Triangle Median to Hypotenuse QA (D62).

Reference task:
  qid 38 (HS float): "As shown in the figure, in the right triangle ABC,
   ∠ACB = 90°, D is the midpoint of AB, and AB = 5. What is the length of
   CD?" Ans: 2.5.

Renders a right triangle with right angle at C, midpoint D on the
hypotenuse AB, and side AB labeled with length N. The model computes
CD = AB/2 (a corollary of the right-triangle median theorem).

Verifier: float (often half-integer).
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "In right triangle ABC, ∠ACB = 90°. D is the midpoint of the hypotenuse AB and AB = {AB}. Find the length of CD.",
    "Triangle ABC has a right angle at C. D is the midpoint of AB; AB = {AB}. Compute CD.",
    "ABC is a right triangle (∠C = 90°). D is the midpoint of AB, with AB = {AB}. What is CD?",
    "In the right triangle ABC shown in the figure, ∠ACB = 90° and D is the midpoint of hypotenuse AB. Given AB = {AB}, find CD.",
    "Right triangle ABC: ∠C = 90°. D is the midpoint of AB, AB = {AB}. What is the length of segment CD?",
    "In the displayed right triangle ABC (∠C = 90°), D is the midpoint of AB. Given AB = {AB}, compute CD.",
    "Find the length of CD in the right triangle ABC, where ∠ACB = 90°, D is the midpoint of AB, and AB = {AB}.",
    "In right triangle ABC with right angle at C, D is the midpoint of AB and AB = {AB}. Determine CD.",
]

# 2026-05-04 R4: full-gradient redesign per mmmath. Compound mode (L6+):
# given the two LEGS, model must (1) compute hypotenuse via Pythagoras
# then (2) apply CD = AB/2.
_TEMPLATES_LEGS = [
    "In right triangle ABC, ∠ACB = 90°, leg AC = {a} and leg BC = {b}. D is the midpoint of the hypotenuse AB. Find CD. (Hint: first compute AB by Pythagoras.)",
    "Right triangle ABC has ∠C = 90°, AC = {a}, BC = {b}. D is the midpoint of AB. Compute CD.",
    "Triangle ABC: right angle at C with legs AC = {a}, BC = {b}. D is the midpoint of AB. Find CD using AB = sqrt(AC^2 + BC^2) and CD = AB/2.",
    "In ABC, ∠C = 90°, the legs measure AC = {a} and BC = {b}. D is the midpoint of hypotenuse AB. Determine CD.",
]


class RightTriangleMedianQA(StandaloneVisualEnv):
    ENV_NAME = "right_triangle_median"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per mmmath.
        # L0-L1: trivial AB integer multiples of 4 (CD = whole int)
        # L2-L3: AB integer odd (CD = .5)
        # L4-L5: AB any int 3..20 (CD whole or .5)
        # L6-L7: COMPOUND — give legs, model computes AB then CD (Pythag chain)
        # L8-L9: COMPOUND with non-Pythag-triple legs → AB irrational
        level = max(0, min(9, level))
        if level <= 1:
            return {"mode": "ab", "AB_choices": [4, 8, 12, 16, 20]}
        if level <= 3:
            return {"mode": "ab", "AB_choices": [3, 5, 7, 9, 11]}
        if level <= 5:
            return {"mode": "ab", "AB_choices": list(range(3, 21))}
        if level <= 7:
            # Compound: Pythagorean triples → integer hyp.
            # (3,4)->5, (5,12)->13, (6,8)->10, (8,15)->17, (9,12)->15,
            # (7,24)->25, (12,16)->20.
            return {
                "mode": "legs",
                "leg_choices": [(3, 4), (5, 12), (6, 8), (8, 15),
                                (9, 12), (7, 24), (12, 16),
                                (20, 21), (15, 20), (10, 24)],
            }
        # L8-L9: non-triple legs → irrational AB → CD = sqrt(a^2+b^2)/2
        return {
            "mode": "legs",
            "leg_choices": [(5, 7), (4, 9), (6, 11), (7, 11), (5, 13),
                            (8, 13), (3, 11), (9, 13), (4, 15), (6, 13),
                            (8, 17), (5, 17), (11, 13), (9, 17)],
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4093 + level * 71 + 173)

        if cfg["mode"] == "legs":
            # 2026-05-04 R4: compound (legs given, hypotenuse derived)
            a, b = rng.choice(cfg["leg_choices"])
            # Model computes AB = sqrt(a^2+b^2), then CD = AB/2
            AB = math.sqrt(a * a + b * b)
            CD = AB / 2.0
            if abs(CD - round(CD)) < 1e-9:
                ans_str = str(int(round(CD)))
            else:
                # 3-decimal precision matches BENCHMARK_NUM_TOLERANCE_ABS=0.001
                ans_str = f"{CD:.3f}".rstrip("0").rstrip(".")
            sidx = (self.seed or 0) % len(_TEMPLATES_LEGS)
            question = _TEMPLATES_LEGS[sidx].format(a=a, b=b)
            img = self._render_legs(a, b, AB)
            return question, ans_str, img

        AB = rng.choice(cfg["AB_choices"])
        # CD = AB / 2
        CD = AB / 2.0
        # Format
        if abs(CD - round(CD)) < 1e-9:
            ans_str = str(int(round(CD)))
        else:
            ans_str = f"{CD:.4f}".rstrip("0").rstrip(".")

        # Pick legs (just for visual; not given to model in question text)
        # Choose acute angle uniformly
        theta = rng.uniform(math.radians(25), math.radians(65))
        a = AB * math.cos(theta)  # leg adjacent
        b = AB * math.sin(theta)  # leg opposite

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(AB=AB)
        img = self._render(a, b, AB)
        return question, ans_str, img

    def _render_legs(self, a, b, AB) -> Image.Image:
        """Like _render but labels both legs (AC=a, BC=b) instead of AB."""
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        Cp = (0, 0)
        Ap = (a, 0)
        Bp = (0, b)
        ax.plot([Ap[0], Bp[0]], [Ap[1], Bp[1]], color="#1a1a1a", linewidth=2)
        ax.plot([Cp[0], Ap[0]], [Cp[1], Ap[1]], color="#1a1a1a", linewidth=2)
        ax.plot([Cp[0], Bp[0]], [Cp[1], Bp[1]], color="#1a1a1a", linewidth=2)
        sq_size = min(a, b) * 0.10
        ax.plot([sq_size, sq_size], [0, sq_size], color="#1a1a1a", linewidth=1.2)
        ax.plot([0, sq_size], [sq_size, sq_size], color="#1a1a1a", linewidth=1.2)
        Dp = ((Ap[0] + Bp[0]) / 2, (Ap[1] + Bp[1]) / 2)
        ax.plot([Cp[0], Dp[0]], [Cp[1], Dp[1]],
                color="#b00020", linewidth=1.6, linestyle="--")
        for (x, y), lbl, off in [
            (Ap, "A", (0.1, -0.05)),
            (Bp, "B", (-0.1, 0.1)),
            (Cp, "C", (-0.15, -0.05)),
            (Dp, "D", (0.05, 0.1)),
        ]:
            ax.plot(x, y, "o", color="#003566", markersize=6)
            ax.text(x + off[0], y + off[1], lbl, fontsize=14,
                    fontweight="bold", color="#003566")
        # Label leg AC = a (along x-axis) and BC = b (along y-axis)
        ax.text(a / 2, -0.25, f"AC = {a}", fontsize=12,
                color="#003566", fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="#fff5e6", ec="#a06010", alpha=0.85))
        ax.text(-0.25, b / 2, f"BC = {b}", fontsize=12,
                color="#003566", fontweight="bold", ha="right",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="#fff5e6", ec="#a06010", alpha=0.85))
        cdx = Dp[0] / 2
        cdy = Dp[1] / 2
        ax.text(cdx, cdy + 0.05, "CD = ?", fontsize=11,
                color="#b00020", fontweight="bold")
        margin = max(a, b) * 0.25
        ax.set_xlim(-margin, a + margin)
        ax.set_ylim(-margin, b + margin)
        ax.set_title("Right triangle ABC — find CD (median to hypotenuse)",
                     fontsize=12, pad=8)
        return self.fig_to_pil(fig, dpi=120)

    def _render(self, a, b, AB) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        # Place C at origin, A at (a, 0), B at (0, b)
        Cp = (0, 0)
        Ap = (a, 0)
        Bp = (0, b)
        # Triangle
        ax.plot([Ap[0], Bp[0]], [Ap[1], Bp[1]], color="#1a1a1a", linewidth=2)
        ax.plot([Cp[0], Ap[0]], [Cp[1], Ap[1]], color="#1a1a1a", linewidth=2)
        ax.plot([Cp[0], Bp[0]], [Cp[1], Bp[1]], color="#1a1a1a", linewidth=2)
        # Right angle marker at C
        sq_size = min(a, b) * 0.12
        ax.plot([sq_size, sq_size], [0, sq_size], color="#1a1a1a", linewidth=1.2)
        ax.plot([0, sq_size], [sq_size, sq_size], color="#1a1a1a", linewidth=1.2)
        # D at midpoint of AB
        Dp = ((Ap[0] + Bp[0]) / 2, (Ap[1] + Bp[1]) / 2)
        ax.plot([Cp[0], Dp[0]], [Cp[1], Dp[1]],
                color="#b00020", linewidth=1.6, linestyle="--")
        # Vertices
        for (x, y), lbl, off in [
            (Ap, "A", (0.1, -0.05)),
            (Bp, "B", (-0.1, 0.1)),
            (Cp, "C", (-0.15, -0.05)),
            (Dp, "D", (0.05, 0.1)),
        ]:
            ax.plot(x, y, "o", color="#003566", markersize=6)
            ax.text(x + off[0], y + off[1], lbl, fontsize=14,
                    fontweight="bold", color="#003566")
        # Label AB length on hypotenuse midpoint side
        midx = (Ap[0] + Bp[0]) / 2
        midy = (Ap[1] + Bp[1]) / 2
        ax.text(midx + 0.15, midy + 0.15, f"AB = {AB}", fontsize=12,
                color="#003566", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="#fff5e6", ec="#a06010", alpha=0.85))
        # Label CD = ?
        cdx = Dp[0] / 2
        cdy = Dp[1] / 2
        ax.text(cdx, cdy + 0.05, "CD = ?", fontsize=11,
                color="#b00020", fontweight="bold")
        margin = max(a, b) * 0.2
        ax.set_xlim(-margin, a + margin)
        ax.set_ylim(-margin, b + margin)
        ax.set_title("Right triangle ABC with median CD", fontsize=13, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = RightTriangleMedianQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        from collections import Counter
        print(f"L{level}: {Counter(ans).most_common(5)}")
