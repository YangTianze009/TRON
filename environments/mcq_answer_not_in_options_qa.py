"""
MCQ Answer Not in Options QA (v4 G11, for bail-pattern (none-of-the-above)).

Targets: minimal-text behavior (idx=12-style cases where model
computes 75° and bails to closest option instead of answering "none of the
above" or re-verifying).

Task: a simple geometry problem (angle / length) with 5 options labeled
A/B/C/D/E. **40% of the time, the correct answer is NOT among A-D but
is "E. none of the above" OR the correct answer IS among A-D** — the
model can't bail to "closest" because E is always a valid option.

Reward: exact MCQ letter match.

Level axes:
  A) Problem complexity: single-hop at L0 -> 2-3 hop at L3+
  B) Trap rate: 30% at L0-3, 40% at L4-6, 50% at L7+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "Given the figure with angle measurements, compute the requested angle. MCQ options include 'E. none of the above'. If your computed answer doesn't match A-D, pick E. Put the letter in <answer>...</answer>. Problem: {problem_txt}",
    "Solve the angle problem below. Option E is 'none of the above' — use it if your computed value isn't in A-D. Do NOT bail to the closest option. Put the letter in <answer>...</answer>. {problem_txt}",
    "Compute the target angle. If no option matches exactly, pick E 'none of the above'. Put letter in <answer>...</answer>. {problem_txt}",
    "Work out the angle. Option E = none of the above. Put letter in <answer>...</answer>. {problem_txt}",
    "Determine the requested angle. Option E exists for when your computation isn't in A-D. Letter in <answer>...</answer>. {problem_txt}",
    "Solve the problem and pick the correct letter from A-E (E = none of the above). Letter in <answer>...</answer>. {problem_txt}",
    "Find the target angle. Option E handles 'none match'. Do NOT pick closest — pick E instead. Letter in <answer>...</answer>. {problem_txt}",
    "Answer the geometry problem. If no option is exactly right, pick E. Letter in <answer>...</answer>. {problem_txt}",
    "Problem: {problem_txt} Pick from A-E (E = none of the above). Put letter in <answer>...</answer>.",
    "The angle problem is {problem_txt} If correct value not in A-D, pick E. Letter in <answer>...</answer>.",
    "Compute the angle. E = none of the above. Put letter in <answer>...</answer>. {problem_txt}",
    "Solve: {problem_txt} Pick letter A-E; E = none match. Put in <answer>...</answer>.",
    "{problem_txt} If your answer isn't in A-D, pick E. Letter in <answer>...</answer>.",
    "Angle problem: {problem_txt} Options include E = none of the above. Pick correct letter in <answer>...</answer>.",
    "Find the angle; if not in A-D pick E. {problem_txt} Put letter in <answer>...</answer>.",
    "{problem_txt} MCQ A-E (E = none of the above). Pick letter and put in <answer>...</answer>.",
]

class MCQAnswerNotInOptionsQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "mcq_answer_not_in_options"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100. Now
        # tighter distractors at higher levels (already L7+) + more aggressive
        # trap rate ramp.
        # 2026-05-04: bumped L9 difficulty (was 100% saturated → push trap_rate
        # to 75% so 3/4 of L9 questions have E="none" as the answer; also
        # use ultra-tight ±1° distractors at L9).
        # 2026-05-04 R3: softened bump — trap_rate=0.75 + ±1° distractors at
        # L9 was too aggressive (L9 dropped to 0.70). Reduce trap_rate to 0.65
        # (still > L8's 0.60, preserving monotonic gradient) and widen
        # distractors to ±2/±3 so closest-option still doesn't trap but the
        # model has visible distance to compute.
        level = max(0, min(level, 9))
        if level == 9:
            trap_rate = 0.65
        else:
            trap_rate = 0.2 + 0.05 * level  # 20% → 65%, was 30%→48%
        return {"trap_rate": trap_rate, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 13)
        self._primary_complexity_feature = level

        # Simple angle problem: "∠A + ∠B = 180°, ∠A = x°, find ∠B"
        angle_A = rng.randint(30, 150)
        angle_B = 180 - angle_A
        problem_txt = (f"Two supplementary angles ∠A and ∠B satisfy "
                        f"∠A + ∠B = 180°. Given ∠A = {angle_A}°, find ∠B.")

        correct = angle_B
        # Decide if this is a trap
        trap = rng.random() < cfg["trap_rate"]

        # Build options
        # 2026-05-04: at L7+ use closer distractors (±1, ±2, ±3) to make
        # trap detection non-obvious. L9 was 100% saturated.
        # 2026-05-04: at L9 use ultra-tight ±1 distractors so model must
        # compute the exact answer rather than "answer closest to one option".
        # 2026-05-04 R3: softened bump — L9 ±1 distractors was too tight
        # (L9 dropped to 0.70). Use the L7+ profile (±2/±3/±5) at L9 too.
        if level >= 7:
            distractor_offsets = [-3, -2, -1, 1, 2, 3, -5, 5]
        else:
            distractor_offsets = [-30, -20, -10, 10, 20, 30, 40, -40]

        if trap:
            # All 4 options are wrong; answer is E
            options_values = set()
            while len(options_values) < 4:
                fake = correct + rng.choice(distractor_offsets)
                if fake == correct or fake <= 0 or fake >= 180:
                    continue
                options_values.add(fake)
            options_values = list(options_values)[:4]
            rng.shuffle(options_values)
            answer = "E"
        else:
            # Include correct answer at a random slot
            options_values = set()
            options_values.add(correct)
            while len(options_values) < 4:
                fake = correct + rng.choice(distractor_offsets)
                if fake == correct or fake <= 0 or fake >= 180:
                    continue
                options_values.add(fake)
            options_values = list(options_values)[:4]
            rng.shuffle(options_values)
            answer = "ABCD"[options_values.index(correct)]

        options_text = "\n".join([f"{'ABCD'[i]}. {v}°"
                                   for i, v in enumerate(options_values)])
        options_text += "\nE. none of the above"

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(problem_txt=problem_txt) + "\nOptions:\n" + options_text

        img = self._render(angle_A, rng)
        return q, answer, img

    def _render(self, angle_A, rng):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw a straight line from (-2, 0) to (2, 0)
        ax.plot([-2, 2], [0, 0], color="black", lw=2.0)
        # Vertex at origin, ray at angle angle_A from the positive x direction
        import math
        xend = 1.5 * math.cos(math.radians(angle_A))
        yend = 1.5 * math.sin(math.radians(angle_A))
        ax.plot([0, xend], [0, yend], color="black", lw=2.0)
        # Mark angle_A
        arc = mpatches.Arc((0, 0), 1, 1, angle=0, theta1=0, theta2=angle_A)
        ax.add_patch(arc)
        ax.text(0.55 * math.cos(math.radians(angle_A / 2)),
                0.55 * math.sin(math.radians(angle_A / 2)) + 0.1,
                f"∠A = {angle_A}°", fontsize=12, fontweight="bold")
        # Mark angle_B (supplementary)
        arc_b = mpatches.Arc((0, 0), 0.8, 0.8, angle=0, theta1=angle_A,
                              theta2=180)
        ax.add_patch(arc_b)
        mid_b = (angle_A + 180) / 2
        ax.text(0.55 * math.cos(math.radians(mid_b)),
                0.55 * math.sin(math.radians(mid_b)) + 0.1,
                "∠B", fontsize=12, fontweight="bold", color="darkgreen")
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().upper().rstrip(".")
        gt = ground_truth.strip().upper().rstrip(".")
        return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_mcq"
    os.makedirs(out_dir, exist_ok=True)
    env = MCQAnswerNotInOptionsQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 1
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[mcq L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/mcq_s{s}_L{level}.png")
            print(f"[mcq L{level} s{s}] A={env._answer}")
