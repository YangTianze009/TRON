"""
Function Injective / Surjective Judgment QA (D7).

Reference task:
  an external reference (UG MCQ): "Is the function injective? choice: (A) Yes (B) No." Ans: B.
  an external reference (UG MCQ): "Is the function surjective? choice: (A) Yes (B) No." Ans: A.

Renders an arrow diagram of f: X -> Y (small finite sets, mapped via arrows).
Asks Yes/No whether the function is injective (1-1) or surjective (onto).

Verifier: single MCQ letter A or B (or "Yes"/"No").
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Aligned with a math benchmark D89/D90: MCQ format "(A) Yes (B) No", answer is letter A or B.
_TEMPLATES_INJ = [
    "Is the function f shown in the diagram injective (one-to-one)? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Examine the arrow diagram. Is f: X -> Y injective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Determine whether f shown above is one-to-one (injective). Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Is the displayed mapping f: X -> Y injective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Look at the function diagram. Is f injective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Decide whether the function f in the image is injective. Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Is the function f drawn as arrows from X to Y a one-to-one (injective) function? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Is f: X -> Y, as shown in the image, injective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
]

_TEMPLATES_SURJ = [
    "Is the function f shown in the diagram surjective (onto)? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Examine the arrow diagram. Is f: X -> Y surjective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Determine whether f shown above is onto (surjective). Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Is the displayed mapping f: X -> Y surjective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Look at the function diagram. Is f surjective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Decide whether the function f in the image is surjective. Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Is the function f drawn as arrows from X to Y an onto (surjective) function? Choices: (A) Yes (B) No. Place the letter in your final answer.",
    "Is f: X -> Y, as shown in the image, surjective? Choices: (A) Yes (B) No. Place the letter in your final answer.",
]


class FunctionInjSurjJudgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "function_inj_surj_judge"

    def _level_config(self, level: int) -> Dict:
        # NOTE: bumped L0-L2 from (3,3) to (4,4) — at (3,3) only 27 distinct
        # mappings exist and many collapse to visually-identical renders, so
        # different seeds produced duplicate images. (4,4) gives 256 mappings.
        level = max(0, min(9, level))
        if level <= 2:
            return {"x_size": 4, "y_size": 4}
        if level <= 5:
            return {"x_size": 4, "y_size": 5}
        if level <= 7:
            return {"x_size": 5, "y_size": 5}
        return {"x_size": 5, "y_size": 6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 9377 + level * 71 + 31)

        # Pick question type: inj or surj
        qtype = rng.choice(["inj", "surj"])
        x_size, y_size = cfg["x_size"], cfg["y_size"]
        # Generate mapping
        # 50% chance to make it have or not have the property
        want_yes = rng.random() < 0.5

        for _ in range(50):
            mapping = [rng.randint(0, y_size - 1) for _ in range(x_size)]
            is_inj = len(set(mapping)) == len(mapping)
            is_surj = len(set(mapping)) == y_size
            if qtype == "inj":
                if (is_inj if want_yes else not is_inj):
                    break
            else:
                if (is_surj if want_yes else not is_surj):
                    break
        else:
            # Construct directly
            if qtype == "inj":
                if want_yes and x_size <= y_size:
                    indices = list(range(y_size))
                    rng.shuffle(indices)
                    mapping = indices[:x_size]
                else:
                    # force a collision
                    mapping = [rng.randint(0, y_size - 1) for _ in range(x_size)]
                    if len(set(mapping)) == x_size:
                        mapping[1] = mapping[0]
            else:
                if want_yes and x_size >= y_size:
                    mapping = []
                    targets = list(range(y_size))
                    rng.shuffle(targets)
                    for t in targets:
                        mapping.append(t)
                    while len(mapping) < x_size:
                        mapping.append(rng.randint(0, y_size - 1))
                    rng.shuffle(mapping)
                else:
                    # missing some y values
                    mapping = [0] * x_size

        is_inj = len(set(mapping)) == len(mapping)
        is_surj = len(set(mapping)) == y_size
        # MCQ letter: (A) Yes (B) No — matches a math benchmark D89/D90 convention.
        if qtype == "inj":
            answer = "A" if is_inj else "B"
            templates = _TEMPLATES_INJ
        else:
            answer = "A" if is_surj else "B"
            templates = _TEMPLATES_SURJ

        sidx = (self.seed or 0) % len(templates)
        question = templates[sidx]
        x_labels = [chr(ord("a") + i) for i in range(x_size)]
        y_labels = [str(i + 1) for i in range(y_size)]
        img = self._render(x_labels, y_labels, mapping)
        return question, answer, img

    def _render(self, x_labels, y_labels, mapping) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Two oval columns
        n_x = len(x_labels)
        n_y = len(y_labels)
        x_xs = 2.0
        y_xs = 7.0
        # Vertical positions
        x_positions = [(x_xs, 9 - i * (8.0 / max(1, n_x - 1)) if n_x > 1 else 5)
                       for i in range(n_x)]
        y_positions = [(y_xs, 9 - i * (8.0 / max(1, n_y - 1)) if n_y > 1 else 5)
                       for i in range(n_y)]

        # Outline ovals around X and Y
        x_oval = mpatches.Ellipse((x_xs, 5), 1.6, 9.5, fill=False,
                                  edgecolor="#1a1a1a", linewidth=2)
        y_oval = mpatches.Ellipse((y_xs, 5), 1.6, 9.5, fill=False,
                                  edgecolor="#1a1a1a", linewidth=2)
        ax.add_patch(x_oval)
        ax.add_patch(y_oval)
        ax.text(x_xs, 9.7, "X", fontsize=20, fontweight="bold",
                ha="center", va="center", color="#003566")
        ax.text(y_xs, 9.7, "Y", fontsize=20, fontweight="bold",
                ha="center", va="center", color="#7f1d1d")

        # Draw points
        for i, (lbl, (x, y)) in enumerate(zip(x_labels, x_positions)):
            ax.plot(x, y, "o", markersize=12, color="#003566")
            ax.text(x - 0.6, y, lbl, fontsize=15, ha="right",
                    va="center", color="#003566")
        for j, (lbl, (x, y)) in enumerate(zip(y_labels, y_positions)):
            ax.plot(x, y, "o", markersize=12, color="#7f1d1d")
            ax.text(x + 0.6, y, lbl, fontsize=15, ha="left",
                    va="center", color="#7f1d1d")

        # Draw arrows
        for i, target in enumerate(mapping):
            x1, y1 = x_positions[i]
            x2, y2 = y_positions[target]
            ax.annotate(
                "",
                xy=(x2 - 0.2, y2),
                xytext=(x1 + 0.2, y1),
                arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=2),
            )

        ax.set_title("Function f: X -> Y", fontsize=15, pad=10)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = FunctionInjSurjJudgeQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
