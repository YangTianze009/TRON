"""
Parabola Vieta QA (M10 / QF-T10, P1).

Given a parabola y = ax² + bx + c labelled with its equation, find the sum
or product of the two roots using Vieta's formulas:
    x1 + x2 = -b/a
    x1 * x2 =  c/a

Verbatim anchor (research §3.1 Q3 idx 3357): substitution method giving
b=2, c=3 demonstrates Vieta-style computation directly.
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


class ParabolaVietaQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "parabola_vieta"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"a_pool": [1], "coef_range": (-5, 5)}
        if level <= 5:
            return {"a_pool": [1, -1, 2], "coef_range": (-6, 6)}
        # 2026-05-04: bumped L9 difficulty (re-bump) — still saturated.
        # Even wider a_pool + bigger coefs.
        return {"a_pool": [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6],
                "coef_range": (-30, 30)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5443 + level * 37 + 13)

        for _ in range(20):
            a = rng.choice(cfg["a_pool"])
            b = rng.randint(*cfg["coef_range"])
            c = rng.randint(*cfg["coef_range"])
            # Need real roots (discriminant > 0)
            disc = b * b - 4 * a * c
            if disc <= 0:
                continue
            # Choose ask
            ask = rng.choice(["sum", "product"])
            if ask == "sum":
                val = -b / a
                ask_word = "the sum (x₁ + x₂) of the two roots"
            else:
                val = c / a
                ask_word = "the product (x₁ · x₂) of the two roots"
            # Want clean integer or simple fraction
            if abs(val - round(val)) > 0.001:
                # half-integer ok
                if abs(2 * val - round(2 * val)) > 0.001:
                    continue
            # Format
            if abs(val - round(val)) < 1e-9:
                ans_str = str(int(round(val)))
            else:
                ans_str = f"{val:g}"

            a_str = "" if a == 1 else ("-" if a == -1 else f"{a}")
            b_term = ""
            if b > 0:
                b_term = f" + {b}x" if b != 1 else " + x"
            elif b < 0:
                b_term = f" - {abs(b)}x" if b != -1 else " - x"
            c_term = ""
            if c > 0:
                c_term = f" + {c}"
            elif c < 0:
                c_term = f" - {abs(c)}"

            equation = f"y = {a_str}x²{b_term}{c_term}"

            question = (
                f"The figure shows the parabola {equation} which intersects "
                f"the x-axis at two points (with x-coordinates x₁ and x₂). "
                f"Find {ask_word}."
            )
            img = self._render(a, b, c)
            return question, ans_str, img
        return None

    def _render(self, a, b, c) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        max_extent = 8
        xs = np.linspace(-max_extent, max_extent, 300)
        ys = a * xs * xs + b * xs + c
        ax.plot(xs, ys, color="#1f77b4", linewidth=2)

        # Mark roots
        disc = b * b - 4 * a * c
        if disc > 0:
            r1 = (-b - math.sqrt(disc)) / (2 * a)
            r2 = (-b + math.sqrt(disc)) / (2 * a)
            ax.scatter([r1, r2], [0, 0], color="#d62728", s=70, zorder=5,
                       edgecolor="black", linewidth=1.0)

        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.set_xlim(-max_extent, max_extent)
        ymax = max(20, abs(c) * 2)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = ParabolaVietaQA()
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
