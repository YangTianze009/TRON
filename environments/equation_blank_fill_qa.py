"""
Equation-blank-fill MCQ.

A small system of arithmetic equations is shown with several numeric blanks
(rendered as `__`). The model must select which MCQ option (a tuple of
integers) fills the blanks so that all equations hold.

Reference an external reference:
  "Fill in the white spaces to make the equations work. choice: (A) 28, 1,
   0, and 37 (B) 1, 0, 27, and 37 (C) 38, −1, 1, 28." Ans: A.

L0: ONE equation `__ + __ = N` (2 blanks). MCQ has 3-4 options; one fills
correctly. Trivially testable by addition.
L9: 3-4 equations and 4 blanks coupling them; MCQ options differ subtly.

Output: single MCQ letter A/B/C/D.
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Fill in the blanks (shown as __) so that all the equations work. The blanks should be filled in left-to-right, top-to-bottom order. Choose the correct option. {choices_str} Place the letter in <answer>...</answer>.",
    "Determine which set of numbers fills the blanks (__) of the displayed equations so that every equation holds. {choices_str} Letter answer in <answer>...</answer>.",
    "Pick the option that fills the blanks (__) so that all displayed equations are true. {choices_str} Place the letter in <answer>...</answer>.",
    "Equation-fill puzzle: choose the integers (in left-to-right, top-to-bottom order) that fill the blanks. {choices_str} Letter in <answer>...</answer>.",
    "Which option correctly fills the blanks of the equations to make them all valid? {choices_str} Letter answer in <answer>...</answer>.",
    "The equations have blanks shown as __. Choose the option whose values fill the blanks correctly. {choices_str} Place letter in <answer>...</answer>.",
    "Select the tuple of integers that, placed in the blanks (in order), makes every equation hold. {choices_str} Letter in <answer>...</answer>.",
    "Fill in the white spaces to make the equations work. {choices_str} Place the letter in <answer>...</answer>.",
    "Find the option that fills the blanks of the displayed equations so each equation balances. {choices_str} Letter answer in <answer>...</answer>.",
    "Which choice correctly fills the blanks in left-to-right, top-to-bottom order? {choices_str} Place the letter in <answer>...</answer>.",
    "Equation completion: pick the option whose numbers complete the equations. {choices_str} Letter in <answer>...</answer>.",
    "Choose the set of integers that turns every equation into a true statement when placed in the blanks. {choices_str} Letter answer in <answer>...</answer>.",
    "Given the equations with blanks (__), select the option that fills them. {choices_str} Place the letter in <answer>...</answer>.",
    "Which list of numbers fills the blanks correctly, in order? {choices_str} Letter in <answer>...</answer>.",
    "Solve the equation-blank puzzle: pick the option that satisfies all equations. {choices_str} Letter answer in <answer>...</answer>.",
    "Determine the correct option for filling the blanks. {choices_str} Place the letter in <answer>...</answer>.",
]


class EquationBlankFillQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "equation_blank_fill"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_eqs = 1
            n_blanks = 2
            n_choices = 3
        elif level <= 2:
            n_eqs = 2
            n_blanks = 2
            n_choices = 3
        elif level <= 4:
            n_eqs = 2
            n_blanks = 3
            n_choices = 4
        elif level <= 6:
            n_eqs = 3
            n_blanks = 3
            n_choices = 4
        elif level <= 8:
            n_eqs = 3
            n_blanks = 4
            n_choices = 4
        else:
            # 2026-05-04: bumped L9 difficulty — 4 eqs, 5 blanks, 5 choices.
            n_eqs = 4
            n_blanks = 5
            n_choices = 5
        return {"level": level, "n_eqs": n_eqs, "n_blanks": n_blanks,
                "n_choices": n_choices}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 2027 + level * 79 + 37)

        # Generate true integer values for the blanks.
        n_blanks = cfg["n_blanks"]
        n_eqs = cfg["n_eqs"]
        # Choose blank values
        blank_vals = [rng.randint(1, 12) for _ in range(n_blanks)]

        # Generate equations: each equation uses a subset of blanks + maybe a
        # constant. Operators: +, -, *.
        # Equation form: `<lhs> <op> <rhs> = <result>` where each operand is
        # either a blank index (referenced as `__`) or a fixed integer.
        # We'll randomly construct equations and verify they all hold.
        equations: List[Tuple[str, int]] = []  # (display_str, lhs_value)
        # blank rendering placeholder per appearance — we just write `__` with
        # ordering = left-to-right, top-to-bottom (so each appearance counts).
        # For simplicity we assign each blank index to AT LEAST one position.
        # Strategy: build equations covering blanks 0..n_blanks-1 in order,
        # then optionally add additional equations using random pairs.
        used_count = [0] * n_blanks
        # Each equation: pick 2 operands (blank index or integer constant) +
        # operator → compute result. Track display string.
        eq_lines: List[str] = []
        eq_blanks_in_order: List[int] = []  # blank-index per appearance, ordered

        # Force every blank to appear at least once
        blanks_to_place = list(range(n_blanks))
        # Distribute blanks across equations
        for ei in range(n_eqs):
            # Decide how many blanks in this equation
            remaining_blanks = blanks_to_place[:]
            if ei < n_eqs - 1:
                k = max(1, len(remaining_blanks) // (n_eqs - ei))
                this_blanks = remaining_blanks[:k]
                blanks_to_place = remaining_blanks[k:]
            else:
                # last equation: take all remaining
                this_blanks = remaining_blanks
                blanks_to_place = []

            if len(this_blanks) == 0:
                # use a random already-placed blank
                this_blanks = [rng.choice(range(n_blanks))]

            # Build equation: op1 OP op2 = res
            op = rng.choice(["+", "-", "*"])
            # op1 = first blank, op2 = either second blank or integer
            if len(this_blanks) >= 2:
                b1 = this_blanks[0]
                b2 = this_blanks[1]
                op1_str = "__"
                op2_str = "__"
                op1_val = blank_vals[b1]
                op2_val = blank_vals[b2]
                eq_blanks_in_order.append(b1)
                eq_blanks_in_order.append(b2)
            else:
                b1 = this_blanks[0]
                op1_str = "__"
                op1_val = blank_vals[b1]
                eq_blanks_in_order.append(b1)
                # Second operand is a small integer constant
                op2_val = rng.randint(1, 9)
                op2_str = str(op2_val)

            if op == "+":
                res = op1_val + op2_val
            elif op == "-":
                res = op1_val - op2_val
            else:
                res = op1_val * op2_val
            eq_str = f"{op1_str} {op} {op2_str} = {res}"
            eq_lines.append(eq_str)

        # Build choices: include the correct tuple + (n_choices - 1) distractors.
        correct_tuple = tuple(blank_vals)
        distractors = []
        attempts = 0
        while len(distractors) < cfg["n_choices"] - 1 and attempts < 100:
            attempts += 1
            # Modify 1-2 entries
            cand = list(correct_tuple)
            n_modify = rng.randint(1, max(1, len(cand)))
            indices = list(range(len(cand)))
            rng.shuffle(indices)
            for idx in indices[:n_modify]:
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
                cand[idx] = cand[idx] + delta
                if cand[idx] < -5:
                    cand[idx] = abs(cand[idx])
            cand_t = tuple(cand)
            if cand_t == correct_tuple:
                continue
            if cand_t in [tuple(d) for d in distractors]:
                continue
            # Ensure distractor truly fails at least one equation
            ok = self._check_tuple(cand_t, eq_lines, eq_blanks_in_order)
            if ok:
                continue  # accidentally correct → skip
            distractors.append(cand_t)
        if len(distractors) < cfg["n_choices"] - 1:
            return None

        all_choices = [correct_tuple] + distractors
        rng.shuffle(all_choices)
        correct_letter = "ABCDEF"[all_choices.index(correct_tuple)]

        # Format choices
        letters = ["A", "B", "C", "D", "E", "F"]
        choice_strs = []
        for li, ch in enumerate(all_choices):
            choice_strs.append(f"({letters[li]}) " +
                               ", ".join(str(x) for x in ch))
        choices_str = "Choices: " + " ".join(choice_strs) + "."

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(choices_str=choices_str)
        answer = correct_letter
        img = self._render(eq_lines)
        return question, answer, img

    def _check_tuple(self, tup: Tuple[int, ...], eq_lines: List[str],
                     eq_blanks_in_order: List[int]) -> bool:
        """Substitute tuple values into equation lines and check truth."""
        # eq_blanks_in_order lists which blank-index each "__" maps to,
        # in the order they appear left-to-right top-to-bottom.
        appearance_idx = 0
        for eq in eq_lines:
            # Substitute __ with values
            parts = eq.split("__")
            substituted = parts[0]
            for k in range(1, len(parts)):
                if appearance_idx >= len(eq_blanks_in_order):
                    return False
                bi = eq_blanks_in_order[appearance_idx]
                appearance_idx += 1
                if bi >= len(tup):
                    return False
                substituted += str(tup[bi]) + parts[k]
            # Evaluate: split on '='
            if "=" not in substituted:
                return False
            lhs, rhs = substituted.split("=", 1)
            lhs = lhs.strip().replace("×", "*")
            rhs = rhs.strip()
            try:
                lhs_val = eval(lhs, {"__builtins__": {}})
                rhs_val = int(rhs)
            except Exception:
                return False
            if lhs_val != rhs_val:
                return False
        return True

    def _render(self, eq_lines: List[str]) -> Image.Image:
        bg = "#ffffff"
        line_color = "#1a1a1a"
        n = len(eq_lines)

        fig_w = 6.5
        fig_h = max(2.5, n * 0.9 + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        for i, eq in enumerate(eq_lines):
            y = n - 1 - i
            ax.text(0.5, y + 0.5, eq, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=line_color,
                    family="monospace")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.3, n + 0.3)
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg,
                    dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
