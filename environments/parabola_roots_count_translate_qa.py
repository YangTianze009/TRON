"""
Number of real roots after translating a parabola y = x^2 + bx + c (or with
small leading coefficient). Mirrors reference QF-T4: figure shows original
parabola, problem asks for new # roots after the parabola is shifted up/down
by k units. This is a discriminant-counting task.

Mapping: M4 (reference QF-T4). Studied IDX: 1009, 169, 3357, 1008, 5721, 3428,
2958, 5266, 5722, 3378, 1011, 5184. Sample-derived design choice: reference Q11
(idx 1011) explicitly says "shifting the parabola y=x²-4x+1 to the left so that
its vertex falls on the y-axis" — so my templates ALWAYS specify a translation
direction (up/down/left/right) by an integer k, then ask for # intersections
with the x-axis (= # real roots).

2026-05-03 extension (M16 quadratic discriminant + M9 modulus / abs value):
added two new question modes alongside the count-of-roots mode:
  - `discriminant_value`: compute b²-4ac of the translated parabola
  - `abs_root_diff`: compute |r_1 - r_2| of the translated parabola
Both share the same render so visual style is preserved. The answer
remains a numeric integer (or simple fraction for abs_root_diff).
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
    "As shown in the figure, the parabola y = ax² + bx + c is plotted. After translating the parabola {shift_desc}, how many points does the new graph share with the x-axis?",
    "As shown in the figure, the parabola y = ax² + bx + c is shown. After shifting the parabola {shift_desc}, how many real roots does the new equation have?",
    "The figure shows the parabola y = ax² + bx + c. If the parabola is translated {shift_desc}, how many x-intercepts does the new parabola have?",
    "As shown in the figure, the parabola y = ax² + bx + c is drawn. Translate it {shift_desc}; how many times does the new curve cross the x-axis?",
    "Shown is the graph of y = ax² + bx + c. After moving the parabola {shift_desc}, how many distinct real solutions does y = 0 have?",
    "Consider the parabola y = ax² + bx + c shown in the figure. Translating it {shift_desc} produces a new parabola — how many intersections with the x-axis does it have?",
    "The parabola y = ax² + bx + c is depicted in the figure. After shifting {shift_desc}, what is the number of distinct real roots?",
    "As illustrated, the parabola y = ax² + bx + c is shown. After translation {shift_desc}, how many real roots does the resulting quadratic have?",
    "Look at the parabola y = ax² + bx + c. After it is moved {shift_desc}, count its x-intercepts.",
    "The figure shows y = ax² + bx + c. If shifted {shift_desc}, how many real x values satisfy the new equation = 0?",
    "Given the parabola y = ax² + bx + c shown, translate {shift_desc}; how many points does the resulting parabola share with the x-axis?",
    "As shown, the parabola y = ax² + bx + c is graphed. Translating {shift_desc} yields a new parabola — how many real roots does it now have?",
    "The parabola y = ax² + bx + c is shown. After shifting {shift_desc}, how many x-intercepts result?",
    "Refer to the figure of y = ax² + bx + c. After translation {shift_desc}, how many distinct real roots are there?",
    "From the graph of y = ax² + bx + c shown, translate it {shift_desc}. Determine the number of real roots of the new quadratic.",
    "Given the parabola y = ax² + bx + c shown above, after translation {shift_desc}, how many real roots does the resulting equation have?",
]

# M16 — discriminant value templates
_DISC_TEMPLATES = [
    "As shown in the figure, the parabola y = ax² + bx + c is plotted. After translating it {shift_desc}, what is the value of the discriminant b² − 4ac of the resulting quadratic?",
    "The figure shows y = ax² + bx + c. If the parabola is translated {shift_desc}, compute the discriminant of the new quadratic equation.",
    "Refer to the parabola y = ax² + bx + c in the figure. After shifting {shift_desc}, what is b² − 4ac of the resulting parabola?",
    "Given the parabola y = ax² + bx + c shown, translate {shift_desc}; what is the discriminant of the new quadratic?",
]

# M9 — modulus / absolute value (|root1 - root2|) templates
_ABS_ROOT_DIFF_TEMPLATES = [
    "As shown in the figure, the parabola y = ax² + bx + c is plotted. After translating it {shift_desc}, what is the absolute value of the difference between its two real roots, |x_1 − x_2|?",
    "The figure shows y = ax² + bx + c. Translate it {shift_desc}; for the resulting parabola compute |x_1 − x_2| where x_1, x_2 are the real roots.",
    "Refer to the parabola y = ax² + bx + c in the figure. After shifting {shift_desc}, the new parabola has two real roots x_1, x_2. Find |x_1 − x_2|.",
    "Given the parabola y = ax² + bx + c shown, translate {shift_desc}; what is the absolute distance between the two x-intercepts of the new parabola?",
]


def _shift_phrase(direction: str, k: int) -> str:
    """direction in {'up','down','left','right'}; k positive integer."""
    unit = "unit" if k == 1 else "units"
    return f"{k} {unit} {direction}ward" if direction in ("up", "down") else f"{k} {unit} to the {direction}"


def _root_count(a: float, b: float, c: float) -> int:
    disc = b * b - 4 * a * c
    if disc > 1e-9:
        return 2
    if disc < -1e-9:
        return 0
    return 1


class ParabolaRootsCountTranslateQA(StandaloneVisualEnv):
    ENV_NAME = "parabola_roots_count_translate"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: small ints, simple "up/down by k" only
        # L9: arbitrary direction including left/right
        directions = ["up", "down"] if level < 4 else ["up", "down", "left", "right"]
        max_k = 1 + level // 2  # 1 → 5
        max_a = 1 if level < 5 else 2
        return {"level": level, "directions": directions,
                "max_k": max_k, "max_a": max_a}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1019 + level * 71 + 17)

        for _ in range(80):
            a = rng.choice([-cfg["max_a"], -1, 1, cfg["max_a"]])
            if a == 0:
                continue
            r1 = rng.randint(-3, 3)
            r2 = rng.randint(-3, 3)
            if r1 == r2:
                # Same root → discriminant 0; allow only sometimes
                if rng.random() > 0.3:
                    continue
            # Construct parabola y = a*(x-r1)*(x-r2) = a x^2 - a(r1+r2) x + a r1 r2
            b = -a * (r1 + r2)
            c = a * r1 * r2
            # ensure base parabola has 2 distinct real roots (or 1 with prob 0.3)
            base_count = _root_count(a, b, c)
            if base_count == 0:
                continue

            direction = rng.choice(cfg["directions"])
            k = rng.randint(1, cfg["max_k"])
            # Apply shift: vertical: c -> c + k (up) or c - k (down).
            # Horizontal: replace x with x - k (right) or x + k (left).
            # New parabola y = a*(x - h)^2 + new_const_form...
            if direction == "up":
                a_n, b_n, c_n = a, b, c + k
            elif direction == "down":
                a_n, b_n, c_n = a, b, c - k
            elif direction == "left":
                # y = a(x+k)^2 + b(x+k) + c
                a_n = a
                b_n = b + 2 * a * k
                c_n = a * k * k + b * k + c
            else:  # right
                a_n = a
                b_n = b - 2 * a * k
                c_n = a * k * k - b * k + c
            new_count = _root_count(a_n, b_n, c_n)
            shift_desc = _shift_phrase(direction, k)
            # 2026-05-03: pick mode (count / discriminant / abs_root_diff).
            # Use parameter override if caller specified, else random.
            mode_override = parameter.get("question_mode")
            if mode_override in ("count", "discriminant_value", "abs_root_diff"):
                mode = mode_override
            else:
                mode = rng.choice(["count", "count", "discriminant_value",
                                   "abs_root_diff"])

            if mode == "count":
                sidx = (self.seed or 0) % len(_TEMPLATES)
                question = _TEMPLATES[sidx].format(shift_desc=shift_desc)
                answer = str(new_count)
            elif mode == "discriminant_value":
                disc = b_n * b_n - 4 * a_n * c_n
                # Only ask if disc is integer-valued (always true for our int a,b,c).
                if abs(disc - round(disc)) > 1e-6:
                    continue
                sidx = (self.seed or 0) % len(_DISC_TEMPLATES)
                question = _DISC_TEMPLATES[sidx].format(shift_desc=shift_desc)
                answer = str(int(round(disc)))
            else:  # abs_root_diff
                disc = b_n * b_n - 4 * a_n * c_n
                if disc <= 0:
                    continue  # need two real roots
                # |x_1 - x_2| = sqrt(disc) / |a|
                val = math.sqrt(disc) / abs(a_n)
                # Only ask if integer (so judge has unambiguous answer).
                if abs(val - round(val)) > 1e-6:
                    continue
                sidx = (self.seed or 0) % len(_ABS_ROOT_DIFF_TEMPLATES)
                question = _ABS_ROOT_DIFF_TEMPLATES[sidx].format(shift_desc=shift_desc)
                answer = str(int(round(val)))
            img = self._render(a, b, c, direction, k)
            return question, answer, img
        return None

    def _render(self, a, b, c, direction, k) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = np.linspace(-6, 6, 400)
        y = a * x * x + b * x + c
        ax.plot(x, y, color="#2c3e80", linewidth=2.2, label="y = ax² + bx + c")
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-12, 12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=9)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
