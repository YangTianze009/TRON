"""
Parallel Rhombus Property QA (v4 G4, for quadrilateral-property).

Targets:

Failure mode: CoT collapse on text-heavy property problems.

Task: parallelogram / rhombus / rectangle diagrams with 1-2 given
properties (one angle, one side length, or a diagonal). Ask for a
specific derived quantity (another angle, a diagonal length, or an
area).

Reward: numeric within 0.5 absolute tolerance for angles; 1% relative
for lengths/areas.

Level axes:
  A) Figure type: parallelogram (L0-2) -> rhombus (L3-5) -> rectangle/square (L6-7) -> mixed (L8-9)
  B) Given quantities: 1 at L0-2, 2 at L3-6, 3 at L7+
  C) Target: angle at L0-5 -> diagonal/area at L6+
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The figure shows a {shape}. Given {given_txt}, find {target_txt}. Enumerate the property you use (opposite angles equal, consecutive supplementary, diagonals bisect each other, etc.). Put the numeric answer in <answer>...</answer>.",
    "In the {shape}, {given_txt}. Determine {target_txt}. State the property, then put the answer in <answer>...</answer>.",
    "Given the {shape} with {given_txt}, compute {target_txt}. Show which property of the {shape} you apply; put the answer in <answer>...</answer>.",
    "The {shape} has {given_txt}. Find {target_txt}. Enumerate the property used and put the answer in <answer>...</answer>.",
    "{shape}: {given_txt}. What is {target_txt}? List the property and put the numeric value in <answer>...</answer>.",
    "For the {shape} with {given_txt}, what is {target_txt}? State the property, then answer in <answer>...</answer>.",
    "The figure is a {shape}. {given_txt}. Solve for {target_txt}. Explain the property, then put the answer in <answer>...</answer>.",
    "Given {given_txt} in the {shape}, compute {target_txt}. Apply which property? State it, then answer in <answer>...</answer>.",
    "In the {shape}, {given_txt}. Find {target_txt}. Property used: ... Answer: <answer>...</answer>.",
    "Given the {shape}, {given_txt}. Find {target_txt}. Reason via which property; put answer in <answer>...</answer>.",
    "{shape} with {given_txt}. Compute {target_txt}. Name the property used, then put answer in <answer>...</answer>.",
    "For this {shape}: {given_txt}. What is {target_txt}? Property first, then numeric answer in <answer>...</answer>.",
    "In a {shape} with {given_txt}, determine {target_txt}. State property; put answer in <answer>...</answer>.",
    "The {shape} has {given_txt}. Compute {target_txt}. Name the property applied, then answer in <answer>...</answer>.",
    "Using the {shape}'s properties, compute {target_txt}. Given: {given_txt}. Put the answer in <answer>...</answer>.",
    "{shape} properties problem: given {given_txt}, find {target_txt}. Put answer in <answer>...</answer>.",
]

class ParallelRhombusPropertyQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "parallel_rhombus_property"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            shapes = ["parallelogram"]
        elif level <= 5:
            shapes = ["parallelogram", "rhombus"]
        else:
            shapes = ["parallelogram", "rhombus", "rectangle", "square"]
        if level <= 3:
            targets = ["angle"]
        elif level <= 5:
            targets = ["angle", "diagonal"]
        elif level <= 7:
            # Composite problems start: area from side+angle (rhombus needs
            # side² sin(angle), parallelogram needs base*height with one of
            # them derived).
            targets = ["composite_area", "composite_diagonal"]
        else:
            targets = ["composite_area", "composite_diagonal", "composite_inverse"]
        return {"shapes": shapes, "targets": targets, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 629)
        self._primary_complexity_feature = level

        # parallelogram doesn't have a closed-form diagonal-from-given problem
        # in this design (would need an extra angle/altitude); pick a valid
        # (shape, target_type) combo instead of returning None.
        shape = rng.choice(cfg["shapes"])
        target_type = rng.choice(cfg["targets"])
        for _ in range(10):
            if not (shape == "parallelogram" and target_type == "diagonal"):
                break
            shape = rng.choice(cfg["shapes"])
            target_type = rng.choice(cfg["targets"])
        if shape == "parallelogram" and target_type == "diagonal":
            target_type = "area"

        # ------- L6-L9 composite multi-step problems -------
        if target_type.startswith("composite"):
            return self._composite_problem(shape, target_type, rng)

        if target_type == "angle":
            if shape == "parallelogram":
                # given angle A, find angle B (consecutive supplementary)
                angle_A = rng.randint(30, 150)
                angle_B = 180 - angle_A
                given_txt = f"angle A = {angle_A}°"
                which_target = rng.choice(["B", "C", "D"])
                if which_target == "B":
                    answer_val = angle_B
                elif which_target == "C":
                    answer_val = angle_A  # opposite equal
                else:  # D
                    answer_val = angle_B
                target_txt = f"angle {which_target}"
                answer = str(answer_val)
            elif shape == "rhombus":
                # rhombus: diagonals bisect angles; all sides equal
                angle_A = rng.randint(40, 140)
                given_txt = f"angle A = {angle_A}°"
                which_target = rng.choice(["B", "half_A"])
                if which_target == "B":
                    answer_val = 180 - angle_A
                    target_txt = "angle B"
                else:
                    answer_val = angle_A / 2
                    target_txt = "the angle formed by diagonal AC with side AB"
                answer = str(int(answer_val)) if answer_val == int(answer_val) else f"{answer_val:.1f}"
            elif shape == "rectangle":
                given_txt = "one angle = 90°"
                target_txt = "the angle formed by the two diagonals meeting at center (acute)"
                # depends on aspect ratio; say W=2L so diagonals form ~ 53.13° and 126.87°
                # Use specific W, L given
                W = rng.randint(3, 6)
                L = W + rng.randint(1, 4)
                theta = 2 * math.degrees(math.atan(W / L))
                given_txt = f"rectangle with length = {L} and width = {W}"
                answer_val = round(theta, 1)
                answer = str(answer_val)
            else:  # square
                angle_A = 90
                given_txt = "side length = 5"
                target_txt = "the angle between a diagonal and a side"
                answer = "45"
        elif target_type == "diagonal":
            # rhombus / rectangle / square diagonals
            if shape == "rhombus":
                side = rng.randint(3, 10)
                angle_A = rng.randint(40, 140)
                d1 = 2 * side * math.sin(math.radians(angle_A / 2))
                given_txt = f"side = {side}, angle A = {angle_A}°"
                target_txt = f"the diagonal opposite angle A (round to 2 decimal places)"
                answer = f"{round(d1, 2)}"
            elif shape == "rectangle":
                L = rng.randint(3, 10)
                W = rng.randint(3, 10)
                d = math.sqrt(L * L + W * W)
                given_txt = f"length = {L}, width = {W}"
                target_txt = "the diagonal length (round to 2 decimal places)"
                answer = f"{round(d, 2)}"
            elif shape == "square":
                s = rng.randint(3, 10)
                d = s * math.sqrt(2)
                given_txt = f"side = {s}"
                target_txt = "the diagonal length (round to 2 decimal places)"
                answer = f"{round(d, 2)}"
            else:
                return None
        else:  # area
            if shape == "parallelogram":
                base = rng.randint(3, 10)
                h = rng.randint(2, 8)
                area = base * h
                given_txt = f"base = {base}, height = {h}"
                target_txt = "the area"
                answer = str(area)
            elif shape == "rhombus":
                d1 = rng.randint(4, 14)
                d2 = rng.randint(4, 14)
                area = d1 * d2 / 2
                given_txt = f"diagonals = {d1} and {d2}"
                target_txt = "the area"
                answer = str(int(area)) if area == int(area) else f"{area:.1f}"
            elif shape == "rectangle":
                L = rng.randint(3, 12)
                W = rng.randint(2, 8)
                area = L * W
                given_txt = f"length = {L}, width = {W}"
                target_txt = "the area"
                answer = str(area)
            else:  # square
                s = rng.randint(2, 10)
                area = s * s
                given_txt = f"side = {s}"
                target_txt = "the area"
                answer = str(area)

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(shape=shape, given_txt=given_txt, target_txt=target_txt)

        img = self._render(shape, given_txt, rng)
        return q, answer, img

    def _composite_problem(self, shape, target_type, rng):
        """Multi-step composite problems for L6-L9 — model has to compose
        2-3 standard formulas. Designed so memorising one isolated formula
        is no longer enough.
        """
        if target_type == "composite_area":
            if shape == "rhombus":
                # area = side^2 * sin(A)
                side = rng.randint(4, 12)
                ang = rng.choice([30, 45, 60, 75, 105, 120, 135, 150])
                area = side * side * math.sin(math.radians(ang))
                given_txt = f"side = {side}, angle A = {ang}°"
                target_txt = "the area of the rhombus (round to 2 decimal places)"
                answer = f"{round(area, 2)}"
            elif shape == "parallelogram":
                # area = base * side * sin(A)
                base = rng.randint(4, 10)
                side = rng.randint(3, 9)
                ang = rng.choice([30, 45, 60, 75, 105, 120])
                area = base * side * math.sin(math.radians(ang))
                given_txt = (f"base AB = {base}, adjacent side AD = {side}, "
                             f"angle A = {ang}°")
                target_txt = "the area of the parallelogram (round to 2 dp)"
                answer = f"{round(area, 2)}"
            elif shape == "rectangle":
                # area from a diagonal and one side: w² + l² = d², area = w*l
                w = rng.randint(3, 9)
                d = w + rng.randint(2, 7)
                if d * d - w * w <= 0:
                    return None
                l = math.sqrt(d * d - w * w)
                area = l * w
                given_txt = f"width = {w}, diagonal = {d}"
                target_txt = "the area (round to 2 decimal places)"
                answer = f"{round(area, 2)}"
            else:  # square
                # area from diagonal: area = d²/2
                d = rng.randint(4, 16)
                area = d * d / 2
                given_txt = f"diagonal = {d}"
                target_txt = "the area"
                answer = f"{round(area, 2) if area != int(area) else int(area)}"
        elif target_type == "composite_diagonal":
            if shape == "rhombus":
                # given side and angle, find both diagonals (ask longer)
                side = rng.randint(4, 12)
                ang = rng.choice([30, 45, 60, 75, 105, 120])
                # short diag = 2*side*sin(A/2), long diag = 2*side*cos(A/2)
                short_d = 2 * side * math.sin(math.radians(ang / 2))
                long_d = 2 * side * math.cos(math.radians(ang / 2))
                target_d = max(short_d, long_d)
                given_txt = f"side = {side}, angle A = {ang}°"
                target_txt = "the LONGER diagonal (round to 2 decimal places)"
                answer = f"{round(target_d, 2)}"
            elif shape == "rectangle":
                # diagonal from area + one side
                w = rng.randint(3, 8)
                area = w * rng.randint(3, 9)
                l = area / w
                d = math.sqrt(w * w + l * l)
                given_txt = f"width = {w}, area = {int(area)}"
                target_txt = "the diagonal (round to 2 decimal places)"
                answer = f"{round(d, 2)}"
            elif shape == "square":
                # diagonal from area
                area = rng.choice([16, 25, 36, 49, 64, 81, 100])
                s = math.sqrt(area)
                d = s * math.sqrt(2)
                given_txt = f"area = {area}"
                target_txt = "the diagonal (round to 2 decimal places)"
                answer = f"{round(d, 2)}"
            else:  # parallelogram — use law of cosines
                a = rng.randint(4, 10)
                b = rng.randint(3, 9)
                ang = rng.choice([45, 60, 75, 90, 105, 120, 135])
                # diagonal²(opposite to angle) = a² + b² - 2ab cos(angle)
                d2 = a * a + b * b - 2 * a * b * math.cos(math.radians(ang))
                if d2 <= 0:
                    return None
                d = math.sqrt(d2)
                given_txt = (f"adjacent sides a = {a}, b = {b}, angle "
                             f"between them = {ang}°")
                target_txt = ("the diagonal opposite the given angle "
                              "(round to 2 decimal places)")
                answer = f"{round(d, 2)}"
        else:  # composite_inverse: given the answer-like quantity, find an input
            if shape == "rhombus":
                # given area and one diagonal, find the other diagonal
                d1 = rng.randint(6, 16)
                d2 = rng.randint(4, 14)
                area = d1 * d2 / 2
                given_txt = f"area = {int(area) if area == int(area) else round(area,1)}, diagonal d1 = {d1}"
                target_txt = "the other diagonal d2"
                answer = f"{d2}"
            elif shape == "parallelogram":
                # given area and one side + angle, find the other side
                s1 = rng.randint(4, 10)
                ang = rng.choice([30, 45, 60, 75, 90, 120])
                s2 = rng.randint(3, 9)
                area = s1 * s2 * math.sin(math.radians(ang))
                given_txt = (f"area = {round(area, 2)}, side {s1}, angle = {ang}°")
                target_txt = "the other side (round to 2 decimal places)"
                answer = f"{round(s2, 2)}"
            elif shape == "rectangle":
                # given area, perimeter, find shorter side
                w = rng.randint(3, 8)
                l = w + rng.randint(1, 6)
                area = w * l
                perim = 2 * (w + l)
                given_txt = f"area = {area}, perimeter = {perim}"
                target_txt = "the shorter side"
                answer = f"{w}"
            else:  # square
                # given perimeter, find diagonal
                s = rng.randint(3, 12)
                perim = 4 * s
                d = s * math.sqrt(2)
                given_txt = f"perimeter = {perim}"
                target_txt = "the diagonal (round to 2 decimal places)"
                answer = f"{round(d, 2)}"

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(shape=shape, given_txt=given_txt,
                                    target_txt=target_txt)
        img = self._render(shape, given_txt, rng)
        return q, answer, img

    def _render(self, shape, given_txt, rng):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 4)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw generic parallelogram (or rectangle/square by parameter)
        if shape == "parallelogram":
            pts = [(0, 0), (4, 0), (5, 2.5), (1, 2.5)]
        elif shape == "rhombus":
            pts = [(0, 1.5), (2, 0), (5, 1.5), (3, 3)]
        elif shape == "rectangle":
            pts = [(0, 0), (4.5, 0), (4.5, 2), (0, 2)]
        else:  # square
            pts = [(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]

        polygon = mpatches.Polygon(pts, fc="none", ec="black", lw=2.0)
        ax.add_patch(polygon)
        labels = ["A", "B", "C", "D"]
        offsets = [(-0.25, -0.2), (0.15, -0.2), (0.2, 0.1), (-0.25, 0.1)]
        for i, (x, y) in enumerate(pts):
            dx, dy = offsets[i]
            ax.text(x + dx, y + dy, labels[i], fontsize=14, fontweight="bold")
        # Show given text
        ax.text(2.5, -0.6, given_txt, fontsize=11, ha="center",
                bbox=dict(facecolor="lightyellow", edgecolor="black", pad=2))

        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        for sym in ["°", "\\circ", "degrees", "degree", "cm", "mm", "m"]:
            pred = pred.replace(sym, "").strip()
            gt = gt.replace(sym, "").strip()
        try:
            p = float(pred)
            g = float(gt)
            return abs(p - g) < 0.5 or abs(p - g) / max(abs(g), 1e-9) < 0.01
        except ValueError:
            return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_prp"
    os.makedirs(out_dir, exist_ok=True)
    env = ParallelRhombusPropertyQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 11
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[prp L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/prp_s{s}_L{level}.png")
            print(f"[prp L{level} s{s}] A={env._answer}")
