"""
Digit Permutation Max QA (D38, P2).

Renders a set of digit-cards (image labels), asks for the maximum sum (or
maximum number) you can form by selecting/arranging them.

Verifier: integer (`\\boxed{N}`).
"""
import math
import random
from itertools import permutations
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class DigitPermutationMaxQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "digit_permutation_max"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_digits": 4, "modes": ["max_number"]}
        if level <= 5:
            return {"n_digits": 5, "modes": ["max_number", "max_sum"]}
        return {"n_digits": 6, "modes": ["max_number", "max_sum"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4099 + level * 47 + 13)

        n = cfg["n_digits"]
        digits = rng.sample(range(0, 10), n)

        mode = rng.choice(cfg["modes"])
        if mode == "max_number":
            # Pick how many digits to use
            k = rng.randint(2, max(2, n - 1))
            # Greedy: sort digits descending, pick top k
            top = sorted(digits, reverse=True)[:k]
            # Don't allow leading zero
            # Concatenate digits
            num_str = "".join(str(d) for d in top)
            if num_str.startswith("0"):
                # swap with first non-zero
                for i, c in enumerate(num_str[1:], 1):
                    if c != "0":
                        num_str = num_str[i] + num_str[1:i] + "0" + num_str[i+1:]
                        break
            answer = num_str
            question = (
                f"The figure shows a set of {n} digit cards. Using exactly "
                f"{k} of these cards (each card used at most once), form the "
                f"largest possible {k}-digit number. (Do not use a leading "
                f"zero.)"
            )
        else:  # max_sum: form two k-digit numbers with the cards, maximize sum
            # k = n // 2, two numbers to add
            k = n // 2
            top = sorted(digits, reverse=True)
            # Greedy pairing: alternate placement
            num1_digits = []
            num2_digits = []
            for i, d in enumerate(top[:2 * k]):
                if i % 2 == 0:
                    num1_digits.append(d)
                else:
                    num2_digits.append(d)
            n1 = int("".join(str(d) for d in num1_digits))
            n2 = int("".join(str(d) for d in num2_digits))
            answer = str(n1 + n2)
            question = (
                f"The figure shows a set of {n} digit cards. Use these cards "
                f"to form two {k}-digit numbers (using each card exactly "
                f"once for the first 2{k} positions). What is the maximum "
                f"possible SUM of the two numbers? (No leading zeros.)"
            )

        img = self._render(digits)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, digits) -> Image.Image:
        n = len(digits)
        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 2.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        for i, d in enumerate(digits):
            ax.add_patch(patches.Rectangle((i * 1.5, 0), 1.2, 1.5,
                                           edgecolor="#2c3e50",
                                           facecolor="#ecf0f1",
                                           linewidth=2))
            ax.text(i * 1.5 + 0.6, 0.75, str(d),
                    ha="center", va="center",
                    fontsize=28, fontweight="bold", color="#2c3e50")

        ax.set_xlim(-0.5, n * 1.5)
        ax.set_ylim(-0.5, 2)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = DigitPermutationMaxQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
