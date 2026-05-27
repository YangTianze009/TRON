"""
Omitted Operator QA (D4).

Reference task:
  qid 500/501 (ES MCQ): "Which operation is omitted in the equation as shown
   in the image? Choices: (A) + (B) - (C) * (D) /." Ans: C.
  qid 25 (ES MCQ): "What is the missing computed symbol? Choices: (A) + (B)
   - (C) * (D) /." Ans: A.

Renders a 3-term equation like  a [op] b = c  with the operator hidden by a
box. The model picks the operator from {+, -, *, /} given as MCQ.

Verifier: single MCQ letter A-D (`\\boxed{A}` or bare A).
"""
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Which operation is omitted in the equation shown in the image? Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "An operator is hidden by a box in the displayed equation. Which operation goes in the box? Choices: (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "The equation in the image is missing one operator. Pick the correct one. Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "Identify the omitted operator in the equation. Choices: (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "What is the missing computed symbol in the equation displayed in the image? Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "The image shows an equation a __ b = c with one operator hidden. Choose the operator. (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "Which operator should replace the box in the equation? Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "Pick the operation that completes the equation. Choices: (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "An operator symbol has been omitted from the equation. Which one is it? Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "Find the hidden operator in the equation shown. Choices: (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "Determine the missing operator in the equation. Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "The equation is missing one operator (covered by a box). Which is it? Choices: (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "Identify the operator behind the box. Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "Which symbol from {+, -, *, /} fills the empty operator slot? Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
    "Choose the operator that makes the equation true. Choices: (A) + (B) - (C) * (D) /. Letter in <answer>...</answer>.",
    "What operator is hidden in the displayed equation? Choices: (A) + (B) - (C) * (D) /. Place the letter in <answer>...</answer>.",
]


_OP_TO_LETTER = {"+": "A", "-": "B", "*": "C", "/": "D"}
_LETTER_TO_OP = {v: k for k, v in _OP_TO_LETTER.items()}

# 2026-05-04 R4: full-gradient redesign per mme_reasoning.
# Two-hidden-op MCQ choices: 16 ordered pairs of operators map to A/B/C/D/E
# in a randomly-chosen 4-of-16 pool. The image displays "a [□] b [□] c = d";
# the model must pick the correct (op1, op2) pair from the listed options.
_OP_PAIRS_ALL = [(o1, o2) for o1 in "+-*/" for o2 in "+-*/"]


class OmittedOperatorQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "omitted_operator"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per mme_reasoning.
        # L0-L1: trivial single-op +/- only, max_val=5
        # L2-L3: standard +/- with multiplication, max_val=12
        # L4-L5: full 4-op single-hidden, max_val=20
        # L6-L7: TWO hidden ops, +/- only (4*4=16 pair choice → MCQ)
        # L8-L9: TWO hidden ops, full 4-op pool with bigger numbers
        level = max(0, min(level, 9))
        if level <= 1:
            return {"ops": ["+", "-"], "max_val": 5, "n_holes": 1}
        if level <= 3:
            return {"ops": ["+", "-", "*"], "max_val": 12, "n_holes": 1}
        if level <= 5:
            return {"ops": ["+", "-", "*", "/"], "max_val": 20, "n_holes": 1}
        if level <= 7:
            return {"ops": ["+", "-", "*"], "max_val": 15, "n_holes": 2}
        return {"ops": ["+", "-", "*", "/"], "max_val": 30, "n_holes": 2}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 6113 + level * 71 + 97)

        # 2026-05-04 R4: 2-hole branch (L6-L9) returns 4-letter MCQ over op-pairs.
        if cfg.get("n_holes", 1) == 2:
            return self._generate_two_holes(rng, cfg)

        # Pick one operator and generate values
        for _ in range(60):
            op = rng.choice(cfg["ops"])
            if op == "+":
                a = rng.randint(1, cfg["max_val"])
                b = rng.randint(1, cfg["max_val"])
                c = a + b
            elif op == "-":
                a = rng.randint(2, cfg["max_val"])
                b = rng.randint(1, a - 1)
                c = a - b
            elif op == "*":
                a = rng.randint(2, min(12, cfg["max_val"]))
                b = rng.randint(2, min(12, cfg["max_val"]))
                c = a * b
            else:  # /
                # ensure clean integer division
                b = rng.randint(2, min(9, cfg["max_val"]))
                quotient = rng.randint(2, min(12, cfg["max_val"]))
                a = b * quotient
                c = quotient
            # Verify uniqueness: only one operator from ops yields the same c
            n_ok = 0
            for cand in cfg["ops"]:
                try:
                    if cand == "+":
                        v = a + b
                    elif cand == "-":
                        v = a - b
                    elif cand == "*":
                        v = a * b
                    elif cand == "/":
                        if b == 0:
                            continue
                        v = a / b
                        if abs(v - round(v)) > 1e-9:
                            continue
                        v = int(round(v))
                    if v == c:
                        n_ok += 1
                except Exception:
                    continue
            if n_ok == 1:
                break
        else:
            return None

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer_letter = _OP_TO_LETTER[op]
        img = self._render(a, b, c, rng)
        return question, answer_letter, img

    def _generate_two_holes(self, rng, cfg):
        """L6-L9: equation a [op1] b [op2] c = d.
        Standard left-to-right evaluation. The model picks an (op1,op2) pair
        from a 4-pair MCQ. Use only pair-eval where evaluation yields the
        SAME unique pair (so the answer is unambiguous given the equation).
        """
        ops_pool = cfg["ops"]
        max_val = cfg["max_val"]
        for _ in range(80):
            # Pick the GT pair
            o1 = rng.choice(ops_pool)
            o2 = rng.choice(ops_pool)
            # Pick small integer operands; bias to ensure result is integer
            a = rng.randint(2, max_val)
            b = rng.randint(2, max_val)
            c = rng.randint(2, max_val)
            d = self._eval_lr(a, o1, b, o2, c)
            if d is None:
                continue
            if d != int(d):
                continue
            d = int(d)
            if abs(d) > max_val * max_val * 4:
                continue
            # Build 4-letter MCQ: 1 correct + 3 distractor pairs that DO NOT
            # eval to the same d on (a,b,c).
            distractor_pool = []
            for cand1 in ops_pool:
                for cand2 in ops_pool:
                    if cand1 == o1 and cand2 == o2:
                        continue
                    cd = self._eval_lr(a, cand1, b, cand2, c)
                    if cd is None or cd == d:
                        # If a distractor pair coincidentally yields d, skip
                        # this whole (a,b,c,o1,o2) — answer would be ambiguous
                        if cd == d:
                            distractor_pool = None
                            break
                        continue
                    distractor_pool.append((cand1, cand2))
                if distractor_pool is None:
                    break
            if distractor_pool is None or len(distractor_pool) < 3:
                continue
            rng.shuffle(distractor_pool)
            choices = [(o1, o2)] + distractor_pool[:3]
            rng.shuffle(choices)
            ans_idx = choices.index((o1, o2))
            ans_letter = "ABCD"[ans_idx]
            # Build question text listing 4 op-pair choices
            opt_lines = [
                f"  ({chr(ord('A') + i)}) {p[0]} and {p[1]}"
                for i, p in enumerate(choices)
            ]
            q = (
                "The image shows an equation a [box1] b [box2] c = d with "
                "TWO operators hidden by boxes. The expression is evaluated "
                "STRICTLY LEFT TO RIGHT (no operator precedence — apply "
                "[box1] first, then [box2]). Choose the pair (op1, op2) "
                "that completes the equation.\n"
                + "\n".join(opt_lines)
                + "\nAnswer with the single letter (A/B/C/D)."
            )
            img = self._render_two(a, b, c, d, rng)
            return q, ans_letter, img
        return None

    @staticmethod
    def _eval_lr(a, o1, b, o2, c):
        """Strict left-to-right: ((a o1 b) o2 c). Return float or None on
        invalid (e.g., divide-by-zero or non-int division)."""
        try:
            if o1 == "+":
                m = a + b
            elif o1 == "-":
                m = a - b
            elif o1 == "*":
                m = a * b
            else:
                if b == 0:
                    return None
                if a % b != 0:
                    return None
                m = a // b
            if o2 == "+":
                return m + c
            if o2 == "-":
                return m - c
            if o2 == "*":
                return m * c
            if c == 0:
                return None
            if m % c != 0:
                return None
            return m // c
        except Exception:
            return None

    def _render_two(self, a, b, c, d, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(8.5, 3.0), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.axis("off")
        font_size = 28
        positions = [(1.2, str(a)), (3.4, "[1]"), (4.8, str(b)),
                     (7.0, "[2]"), (8.4, str(c)), (10.0, "="),
                     (11.2, str(d))]
        for x, txt in positions:
            if txt in ("[1]", "[2]"):
                rect = mpatches.Rectangle((x - 0.6, 1.2), 1.4, 1.6,
                                          linewidth=2.0,
                                          edgecolor="#b00020",
                                          facecolor="#ffe6e6",
                                          linestyle="--")
                ax.add_patch(rect)
                ax.text(x + 0.1, 2, "?", ha="center", va="center",
                        fontsize=font_size, color="#b00020",
                        fontweight="bold")
                ax.text(x + 0.1, 0.7, txt, ha="center", va="center",
                        fontsize=12, color="#b00020")
            else:
                ax.text(x, 2, txt, ha="center", va="center",
                        fontsize=font_size, fontweight="bold",
                        color="#1a1a1a")
        ax.set_title("Find the two missing operators (left-to-right eval).",
                     fontsize=12, pad=10)
        return self.fig_to_pil(fig, dpi=120)

    def _render(self, a, b, c, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 3), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        # Render a [box] b = c
        font_size = 32
        # text positions
        ax.text(1.5, 2, str(a), ha="center", va="center", fontsize=font_size,
                fontweight="bold", color="#1a1a1a")
        # Box for missing operator
        rect = mpatches.Rectangle((3.0, 1.0), 1.6, 2.0,
                                  linewidth=2.5, edgecolor="#b00020",
                                  facecolor="#ffe6e6", linestyle="--")
        ax.add_patch(rect)
        ax.text(3.8, 2, "?", ha="center", va="center",
                fontsize=font_size + 4, color="#b00020", fontweight="bold")
        ax.text(5.5, 2, str(b), ha="center", va="center",
                fontsize=font_size, fontweight="bold", color="#1a1a1a")
        ax.text(7.0, 2, "=", ha="center", va="center",
                fontsize=font_size, fontweight="bold", color="#1a1a1a")
        ax.text(8.5, 2, str(c), ha="center", va="center",
                fontsize=font_size, fontweight="bold", color="#1a1a1a")
        ax.set_title("Find the missing operator (in the box).",
                     fontsize=14, pad=10)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = OmittedOperatorQA()
    for level in [0, 3, 6, 9]:
        answers = set()
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                answers.add(env._answer)
        print(f"L{level}: distinct_answers={len(answers)} sample={list(answers)[:5]}")
