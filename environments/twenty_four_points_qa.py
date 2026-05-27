"""
Twenty-four points: given 4 numbers shown in the image, build a math expression
using +, -, *, /, parentheses, and each number exactly once, that evaluates to 24.

Difficulty axes:
  - magnitude of digits (1-9 at L0; up to 1-13 at L9)
  - require multiplication / division at L >=4 (not just additive)
"""
import math
import random
import re
from itertools import permutations, product
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the 24-game with the four numbers below.\n\n"
    "### Game Rules:\n"
    "1. Use each of the four numbers exactly once.\n"
    "2. Allowed operators: +, -, *, / and parentheses.\n"
    "3. The expression must evaluate to 24.\n\n"
    "### Coordinate System:\n"
    "- Numbers are given as a list.\n\n"
    "### Current Puzzle State:\n"
    "- numbers = {numbers}\n\n"
    "### Output Format:\n"
    "Output the arithmetic expression inside <answer>...</answer>.\n"
    "Example: <answer>(3*4)+(5+7)</answer>",

    "Solve the 24-game for the numbers below.\n\n"
    "### Game Rules:\n"
    "- Use each number exactly once.\n"
    "- Operators: +, -, *, /, parentheses.\n"
    "- Result must equal 24.\n\n"
    "### Coordinate System:\n"
    "- Numbers list.\n\n"
    "### Current Puzzle State:\n"
    "numbers = {numbers}\n\n"
    "### Output Format:\n"
    "Output the expression inside <answer>...</answer>.",

    "Your task is to find a 24-game expression for the numbers below.\n\n"
    "### Game Rules:\n"
    "Standard 24-points puzzle.\n\n"
    "### Coordinate System:\n"
    "- 4 numbers.\n\n"
    "### Current Puzzle State:\n"
    "numbers = {numbers}\n\n"
    "### Output Format:\n"
    "Output the expression inside <answer>...</answer>.",
]


def _eval_expr(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression."""
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception:
        return None


def _solve_24(nums):
    """Return a single solution expression for the 4 numbers, or None."""
    target = 24.0
    op_choices = ["+", "-", "*", "/"]
    forms = [
        "(({a}{op1}{b}){op2}{c}){op3}{d}",
        "({a}{op1}({b}{op2}{c})){op3}{d}",
        "({a}{op1}{b}){op2}({c}{op3}{d})",
        "{a}{op1}(({b}{op2}{c}){op3}{d})",
        "{a}{op1}({b}{op2}({c}{op3}{d}))",
    ]
    for perm in permutations(nums):
        a, b, c, d = perm
        for op1, op2, op3 in product(op_choices, repeat=3):
            for f in forms:
                expr = f.format(a=a, b=b, c=c, d=d, op1=op1, op2=op2, op3=op3)
                v = _eval_expr(expr)
                if v is not None and abs(v - target) < 1e-6:
                    return expr
    return None


class TwentyFourPointsQA(StandaloneVisualEnv):
    ENV_NAME = "twenty_four_points"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        max_digit = 9 + (level // 2)  # 9 → 13
        min_digit = 1
        require_mul_div = level >= 4
        return {
            "level": level,
            "max_digit": max_digit,
            "min_digit": min_digit,
            "require_mul_div": require_mul_div,
        }

    def _sample_solvable(self, rng: random.Random, cfg: Dict, max_attempts: int = 200):
        """Sample 4 numbers that have a 24-game solution under cfg."""
        for _ in range(max_attempts):
            nums = sorted(
                rng.randint(cfg["min_digit"], cfg["max_digit"]) for _ in range(4)
            )
            sol = _solve_24(nums)
            if sol is None:
                continue
            if cfg["require_mul_div"] and not any(op in sol for op in "*/"):
                continue
            return nums, sol
        return None, None

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1009 + level * 41 + 19)

        nums, sol = self._sample_solvable(rng, cfg)
        if nums is None:
            return None

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(numbers=list(nums))
        img = self._render(nums, rng)
        return question, sol, img

    def _render(self, nums, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5.5, 2.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Render each number in its own card
        for i, n in enumerate(nums):
            x = 0.5 + i
            ax.add_patch(plt.Rectangle((x - 0.4, 0.1), 0.8, 1.2,
                                       facecolor="#f0f4ff",
                                       edgecolor="#2c3e50", linewidth=1.6))
            ax.text(x, 0.7, str(n), ha="center", va="center",
                    fontsize=28, fontweight="bold", color="#1a2840")
        ax.set_xlim(0, 4.5)
        ax.set_ylim(0, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Make 24", fontsize=14, color="#2c3e50", pad=4)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Override base: parse predicted expression, check it uses exactly the
        # same multiset of digits and evaluates to 24.
        # Strip whitespace, common framing, latex
        p = predicted.strip()
        p = re.sub(r"\\times", "*", p)
        p = re.sub(r"\\div", "/", p)
        p = re.sub(r"×", "*", p)
        p = re.sub(r"÷", "/", p)
        p = p.replace("=24", "").replace("= 24", "").rstrip("=").strip()
        # Extract pure expression
        m = re.search(r"[\d\s\+\-\*\/\(\)\.]+", p)
        if not m:
            return False
        expr = m.group().strip()
        val = _eval_expr(expr)
        if val is None or abs(val - 24.0) > 1e-6:
            return False
        # Verify it uses same multiset of digits as ground truth
        gt_digits = sorted(int(x) for x in re.findall(r"\d+", ground_truth))
        pred_digits = sorted(int(x) for x in re.findall(r"\d+", expr))
        return gt_digits == pred_digits
