"""
Cryptarithm (alphametic / cryptomath) puzzle — matches a puzzle benchmark CryptoMath
benchmark prompt format.

Benchmark prompt verbatim (idx=271, idx=272):
  Solve this cryptarithmetic puzzle, where each letter represents a unique digit (0-9).
  Equation: CDA+FBA=GEAF Different letters must correspond to different values, and no leading letter can be zero.
  Please provide your answer as a list of comma-separated "letter"=number pairs.
  Example answer format: ["A"=5,"B"=3,...,"Z"=9].

Benchmark answer format (string repr of dict):
  {'A': 8, 'B': 2, 'C': 4, 'D': 5, 'E': 0, 'F': 6, 'G': 1}

Rule:
  - Each letter represents a unique digit 0..9.
  - Different letters map to different digits.
  - The leading letter of any multi-digit number cannot be zero.
  - The arithmetic equation must hold.

Self-contained: random equation generator + DFS solver. NO `from RLVE`,
`from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9): tracks benchmark D=1..D=5.
  L0..L1: 2-digit + 2-digit = 2/3-digit (~4-5 letters)
  L2..L3: 3-digit + 3-digit = 3/4-digit (~6-7 letters)
  L4..L7: 3-digit + 3-digit = 4-digit (7-8 letters)
  L8..L9: 4-digit + 3/4-digit = 5-digit (8 letters, SEND+MORE=MONEY-style)
"""
import ast
import random
import re
from io import BytesIO
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Match benchmark wording verbatim: a single short paragraph prompt.
_TEMPLATES = [
    "Solve this cryptarithmetic puzzle, where each letter represents a unique digit (0-9).\n"
    "Equation: {equation} Different letters must correspond to different values, and no leading letter can be zero.\n"
    "Please provide your answer as a list of comma-separated \"letter\"=number pairs.\n"
    "Example answer format: [\"A\"=5,\"B\"=3,...,\"Z\"=9].",
]


def _solve_cryptarithm(equation: str, max_solutions: int = 2):
    """Brute-force letter→digit assignment search. Returns up to
    `max_solutions` valid mappings."""
    # Find letters in equation
    letters = sorted(set(re.findall(r"[A-Z]", equation.upper())))
    if len(letters) > 10:
        return []
    # Find leading letters (first char of each multi-digit word)
    words = re.findall(r"[A-Z]+", equation.upper())
    leading = set(w[0] for w in words if len(w) > 1)

    # Parse equation into LHS and RHS
    parts = equation.upper().split("=")
    if len(parts) != 2:
        return []

    found = []
    for perm in permutations(range(10), len(letters)):
        mapping = dict(zip(letters, perm))
        # Check leading != 0
        if any(mapping[L] == 0 for L in leading):
            continue
        # Substitute and evaluate
        lhs_text = parts[0]
        rhs_text = parts[1]
        for L, d in mapping.items():
            lhs_text = lhs_text.replace(L, str(d))
            rhs_text = rhs_text.replace(L, str(d))
        try:
            # Allow only digits, +, -, *, /, parens
            if not re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", lhs_text):
                continue
            if not re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", rhs_text):
                continue
            lv = eval(lhs_text, {"__builtins__": {}})
            rv = eval(rhs_text, {"__builtins__": {}})
            if lv == rv:
                found.append(mapping)
                if len(found) >= max_solutions:
                    return found
        except Exception:
            continue
    return found


def _gen_cryptarithm(level: int, rng: random.Random,
                     max_attempts: int = 100):
    """Generate a random cryptarithm equation with desired # distinct letters
    and difficulty. Returns (equation, distinct_letters_sorted, mapping) or None.

    Strategy: pick a digit assignment first, build a numeric equation
    a + b = c where lengths fit the level, then map digits → letters
    (keeping uniqueness, avoiding leading-zero situations).

    Equation format matches benchmark: no spaces between operators, e.g.
    `CDA+FBA=GEAF` (idx=271 verbatim).
    """
    # Difficulty: number of digits in operands. Always use '+' (matches all
    # 8 sampled CryptoMath benchmark questions per design notes §CryptoMath).
    # 2026-05-04 R3 retry: extend 2+2 digit through L3 (was only L0/L1);
    # 3+3 only kicks in at L4. Distinct-letter caps tightened too.
    if level <= 3:
        n_digits_a = 2
        n_digits_b = 2
    elif level <= 5:
        n_digits_a = 2
        n_digits_b = 3
    elif level <= 7:
        n_digits_a = 3
        n_digits_b = 3
    else:
        n_digits_a = 4
        n_digits_b = 3
    # Letter cap by tier: L0-L3 ≤4, L4-L5 ≤5, L6-L7 ≤6, L8+ ≤8.
    if level <= 3:
        max_distinct = 4
    elif level <= 5:
        max_distinct = 5
    elif level <= 7:
        max_distinct = 6
    else:
        max_distinct = 8
    op = "+"

    for _ in range(max_attempts):
        # Sample numeric values
        a_min = 10 ** (n_digits_a - 1)
        a_max = 10 ** n_digits_a - 1
        b_min = 10 ** (n_digits_b - 1)
        b_max = 10 ** n_digits_b - 1
        a = rng.randint(a_min, a_max)
        b = rng.randint(b_min, b_max)
        c = a + b
        c_str = str(c)
        a_str = str(a)
        b_str = str(b)
        # Map distinct digits → distinct letters
        digits_used = set(a_str + b_str + c_str)
        n_distinct = len(digits_used)
        # 2026-05-04 R3: softened — apply per-level cap (L0/L1 ≤4 letters,
        # L2/L3 ≤6, else ≤8). Keeps solver tractable AND L0/L1 truly easier.
        if n_distinct > max_distinct:
            continue
        # Pick letters
        all_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        rng.shuffle(all_letters)
        chosen_letters = all_letters[:n_distinct]
        digit_to_letter = dict(zip(sorted(digits_used), chosen_letters))
        # Build letter strings
        a_letters = "".join(digit_to_letter[d] for d in a_str)
        b_letters = "".join(digit_to_letter[d] for d in b_str)
        c_letters = "".join(digit_to_letter[d] for d in c_str)
        # Build equation string — NO SPACES (matches benchmark idx=271 format).
        equation = f"{a_letters}{op}{b_letters}={c_letters}"
        # Build mapping: letter → digit
        mapping = {letter: int(digit) for digit, letter in digit_to_letter.items()}
        # Sanity: solver should find this assignment
        sols = _solve_cryptarithm(equation, max_solutions=1)
        if not sols:
            continue
        distinct_letters_sorted = sorted(set(re.findall(r"[A-Z]", equation)))
        return equation, distinct_letters_sorted, mapping
    return None


class CryptarithmQA(StandaloneVisualEnv):
    ENV_NAME = "cryptarithm"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the digit assignment directly inside "
        "`<answer>...</answer>` as a comma-separated list of \"letter\"=digit pairs."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {"level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((seed or 0) * 7919 + level * 31 + 17)

        result = _gen_cryptarithm(level, rng)
        if result is None:
            return None
        equation, letters, mapping = result

        self._equation = equation
        self._letters = letters
        # Reference solution (just one valid mapping)
        self._reference_mapping = mapping

        sidx = (seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(equation=equation)
        # Answer format = stringified Python dict (matches benchmark answer column).
        ans_str = "{" + ", ".join(f"'{k}': {v}" for k, v in sorted(mapping.items())) + "}"

        img = self._render(equation)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, equation: str) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5.5, 2.0), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.text(0.5, 0.55, equation, ha="center", va="center",
                fontsize=24, fontfamily="monospace",
                fontweight="bold", color="#1a3a6e",
                transform=ax.transAxes)
        ax.text(0.5, 0.15, "Each letter = unique digit 0-9; no leading zeros.",
                ha="center", va="center", fontsize=10,
                color="#777", transform=ax.transAxes)
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Validate any letter→digit mapping that solves the equation.

        Multiple valid assignments may exist; we re-verify against equation.
        Accepts dict literal `{'A': 8, ...}`, list-of-pairs `["A"=5,"B"=3,...]`,
        or any sequence of `LETTER[:=]DIGIT` pairs.
        """
        s = predicted.strip()
        # Strip code fences
        s = re.sub(r"```[^\n]*\n", "", s).replace("```", "").strip()

        # Try Python dict literal
        mapping = None
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                mapping = {str(k).upper(): int(v) for k, v in obj.items()}
        except (ValueError, SyntaxError, TypeError):
            pass
        # Try parsing "A:1, B:2, ..." or "A=1, B=2, ..." or "\"A\"=1,..."
        if mapping is None:
            try:
                mapping = {}
                # Accept optional surrounding quotes around the letter, single
                # uppercase letter, then `:` or `=`, then digit.
                for m in re.finditer(r"['\"]?([A-Za-z])['\"]?\s*[:=]\s*(\d+)", s):
                    mapping[m.group(1).upper()] = int(m.group(2))
                if not mapping:
                    return False
            except (ValueError, AttributeError):
                return False
        if not mapping:
            return False
        # All letters must be present
        for L in self._letters:
            if L not in mapping:
                return False
        # Each digit 0-9
        for v in mapping.values():
            if not (0 <= v <= 9):
                return False
        # Distinct digits
        if len(set(mapping[L] for L in self._letters)) != len(self._letters):
            return False
        # Substitute and evaluate
        eq = self._equation.upper()
        # Find leading letters
        words = re.findall(r"[A-Z]+", eq)
        leading = set(w[0] for w in words if len(w) > 1)
        for L in leading:
            if mapping.get(L, -1) == 0:
                return False
        eval_eq = eq
        for L, d in mapping.items():
            eval_eq = eval_eq.replace(L, str(d))
        try:
            lhs, rhs = eval_eq.split("=")
            if not re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", lhs):
                return False
            if not re.fullmatch(r"[\d\s\+\-\*\/\(\)]+", rhs):
                return False
            return eval(lhs, {"__builtins__": {}}) == eval(rhs, {"__builtins__": {}})
        except Exception:
            return False
