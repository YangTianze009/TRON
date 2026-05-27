"""
Blank Annotation Trap QA (v4 G12).

Targets:

Failure mode: v3 assumes missing information (e.g., assumes a right angle
without a right-angle marker, or fabricates numbers from a blank sign).
Need env where the GT is "cannot determine" / "insufficient info" when
a required annotation is intentionally missing.

Task: render a geometric figure where some required annotation (right angle
mark, isosceles tick marks, labeled length, equal-angle mark) is
intentionally missing. The MCQ includes one literal wrong-assumption answer
(what you get if you ASSUME the missing annotation), one numerically off
option, a random option, and "E. Cannot be determined from the given
information". The correct answer is **always E** in trap cases.

To not make it too easy, 30% of problems are NON-trap (all info provided,
GT is the numeric answer and E is wrong).

Reward: exact MCQ letter match.

Level axes:
  A) Trap rate: 50% at L0-2, 60% at L3-5, 70% at L6+
  B) Figure complexity: triangle at L0-2, circle at L3-5, compound at L6+
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
    "Examine the figure carefully. Compute the requested quantity **only if all necessary information is present**. If information is missing, pick E 'Cannot be determined'. Do NOT assume unmarked angles or sides. Put the letter in <answer>...</answer>.\n{problem_txt}\n{options_txt}",
    "Given the figure, determine the requested value. Options include E = 'Cannot be determined'. Use E if required information is absent from the figure; do NOT assume. Put letter in <answer>...</answer>.\n{problem_txt}\n{options_txt}",
    "{problem_txt} Pick from A-E (E = Cannot be determined). Only answer A-D if all needed info is explicitly shown. Put letter in <answer>...</answer>.\n{options_txt}",
    "Evaluate the figure. If key annotations are missing (right-angle marker, tick marks, labeled value), pick E. Otherwise compute the answer. Letter in <answer>...</answer>.\n{problem_txt}\n{options_txt}",
    "Problem: {problem_txt} MCQ options include E = 'Cannot be determined from the given information'. Be strict about what's actually shown — do not assume. Put letter in <answer>...</answer>.\n{options_txt}",
    "Figure check: {problem_txt} Options: {options_txt} If anything needed is missing, pick E. Put letter in <answer>...</answer>.",
    "Determine the requested value from the figure. Option E = 'Cannot be determined'. Do NOT assume missing information. {problem_txt}\n{options_txt} Letter in <answer>...</answer>.",
    "Look at the figure strictly. {problem_txt} If info is missing, pick E. {options_txt} Letter in <answer>...</answer>.",
    "Answer based only on what the figure shows. {problem_txt} Options: {options_txt} Letter in <answer>...</answer>.",
    "{problem_txt} You must NOT assume additional information (e.g., right angles or equal lengths) unless the figure explicitly marks them. {options_txt} Letter in <answer>...</answer>.",
    "Evaluate rigorously. Only answer with a computed value if all required annotations are shown. Else pick E. {problem_txt}\n{options_txt}",
    "Rigorous answer — do not assume. {problem_txt} {options_txt} Letter in <answer>...</answer>.",
    "From the figure alone (no assumptions), answer the following: {problem_txt} Options: {options_txt} Letter in <answer>...</answer>.",
    "Figure-based answer only. {problem_txt} Options: {options_txt} Letter in <answer>...</answer>.",
    "Strict interpretation: {problem_txt} Options include E = Cannot be determined. {options_txt} Letter in <answer>...</answer>.",
    "{problem_txt} {options_txt} Only compute if all info is shown; else pick E. Letter in <answer>...</answer>.",
]

class BlankAnnotationTrapQA(StandaloneVisualEnv):
    ENV_NAME = "blank_annotation_trap"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04 R3 retry: was 35% → 60% (v5=0.53). Drop to 25% → 55%
        # so L0 even easier (more genuine-numeric problems) and L9 still
        # has solid trap fraction. Keeps monotonic increase.
        trap_rate = 0.25 + 0.033 * level  # 25% → 55%
        return {"trap_rate": trap_rate, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 197)
        self._primary_complexity_feature = level

        # Problem: triangle with sides a, b, and asked for hypotenuse c
        # - If right-angle marker present: c = sqrt(a^2 + b^2)
        # - If marker absent: cannot determine
        a = rng.randint(3, 12)
        b = rng.randint(3, 12)
        right_angle_c = math.sqrt(a * a + b * b)

        is_trap = rng.random() < cfg["trap_rate"]
        has_right_angle_mark = not is_trap

        problem_txt = (f"The triangle has sides a = {a} and b = {b}. "
                        f"Find the length of the third side c.")

        # Options:
        # Trap version: missing right-angle marker, so answer is E
        # Non-trap: right-angle marker shown, answer is the Pythagorean value
        if is_trap:
            # Put 4 plausible-but-unjustified numerics, then E
            opt_values = [round(right_angle_c, 2),  # what you'd get assuming right angle
                          round(a + b, 2),
                          round(abs(a - b), 2),
                          round((a + b) / 2, 2)]
            rng.shuffle(opt_values)
            answer = "E"
        else:
            # genuine answer present
            opt_values = [round(right_angle_c, 2)]
            while len(opt_values) < 4:
                fake = round(right_angle_c + rng.uniform(-3, 3), 2)
                if fake != opt_values[0] and fake not in opt_values and fake > 0:
                    opt_values.append(fake)
            rng.shuffle(opt_values)
            answer = "ABCD"[opt_values.index(round(right_angle_c, 2))]

        options_txt = "A. {}\nB. {}\nC. {}\nD. {}\nE. Cannot be determined from the given information".format(*opt_values)

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(problem_txt=problem_txt, options_txt=options_txt)

        img = self._render(a, b, has_right_angle_mark, rng)
        return q, answer, img

    def _render(self, a, b, has_right_angle_mark, rng):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Make the triangle "right-looking" (but we won't mark the right angle unless has_right_angle_mark)
        # Use corners (0,0), (a, 0), (0, b)
        scale = 0.5
        A = (0, 0)
        B = (a * scale, 0)
        C = (0, b * scale)
        lim = max(a, b) * scale + 1
        ax.set_xlim(-0.5, lim)
        ax.set_ylim(-0.5, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(mpatches.Polygon([A, B, C], fc="none", ec="black", lw=2.0))
        # Side labels
        ax.text(a * scale / 2, -0.25, f"a = {a}", fontsize=12, ha="center", fontweight="bold")
        ax.text(-0.3, b * scale / 2, f"b = {b}", fontsize=12, va="center", fontweight="bold")
        # Third side label 'c'
        ax.text(a * scale / 2 + 0.15, b * scale / 2, "c = ?", fontsize=12,
                color="darkgreen", fontweight="bold")
        # Right angle marker at A (only if has_right_angle_mark)
        if has_right_angle_mark:
            ax.add_patch(mpatches.Rectangle((0, 0), 0.2, 0.2,
                                             fc="none", ec="red", lw=1.5))

        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().upper().rstrip(".")
        gt = ground_truth.strip().upper().rstrip(".")
        return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_bat"
    os.makedirs(out_dir, exist_ok=True)
    env = BlankAnnotationTrapQA()
    for level in (0, 3, 6, 9):
        a_count = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        for seed in range(20):
            s = seed * 100 + level * 37 + 271
            ok = env.generate(seed=s, parameter={"level": level})
            if ok:
                a_count[env._answer] += 1
        print(f"[bat L{level}] {a_count}")
        # Sample one for viz
        env.generate(seed=level * 100 + 271, parameter={"level": level})
        env.render().save(f"{out_dir}/bat_L{level}.png")
