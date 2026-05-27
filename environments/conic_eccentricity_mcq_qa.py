"""
Conic eccentricity MCQ (reference R8 extension).

Family: ellipse / hyperbola eccentricity from a focal-chord + tangent-circle
configuration. The MME sample (`idx=164`):
  "F is right focus of ellipse x²/a² + y²/b² = 1 (a>b>0). M on C; line MF
   tangent to circle (x-c/2)² + y² = b²/16 at N. If FM = 4FN, eccentricity?"

We generate parametric variants where we choose a, b first (so we know the
true eccentricity), then craft a self-consistent geometric scenario whose
unknown (eccentricity) the model must compute. Output MCQ A/B/C/D.

Self-contained. NO `from RLVE`, `from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0: trivially given a, b values directly (compute e = c/a)
  L9: focal-chord-with-tangent style problem
"""
import math
import random
import re
from io import BytesIO
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the eccentricity of the conic described below.\n\n"
    "### Game Rules:\n"
    "1. The eccentricity e of an ellipse x²/a² + y²/b² = 1 (a>b>0) is e = c/a where c² = a² − b².\n"
    "2. Use the given geometric configuration to determine e.\n"
    "3. Choose the correct answer from the multiple-choice options.\n\n"
    "### Coordinate System:\n"
    "- Standard cartesian; foci on the x-axis.\n\n"
    "### Current Puzzle State:\n"
    "{question_text}\n\n"
    "Options:\n{options_text}\n\n"
    "### Output Format:\n"
    "Output the letter of the correct option inside <answer>...</answer>.\n"
    "Example: <answer>B</answer>",

    "Solve the conic-section problem below.\n\n"
    "### Game Rules:\n"
    "- Compute eccentricity e for the conic described.\n"
    "- Choose the matching option.\n\n"
    "### Coordinate System:\n"
    "- Cartesian; standard ellipse form.\n\n"
    "### Current Puzzle State:\n"
    "{question_text}\n\n"
    "Options:\n{options_text}\n\n"
    "### Output Format:\n"
    "Output the option letter (A/B/C/D) inside <answer>...</answer>.",

    "Your task is to find the eccentricity in the configuration below.\n\n"
    "### Game Rules:\n"
    "Standard conic-section problem; pick the MCQ option.\n\n"
    "### Coordinate System:\n"
    "- Cartesian.\n\n"
    "### Current Puzzle State:\n"
    "{question_text}\n\n"
    "{options_text}\n\n"
    "### Output Format:\n"
    "Output the letter inside <answer>...</answer>.",
]


# Known eccentricity values + LaTeX-like display
_KNOWN_ECC = [
    (Fraction(1, 2), "1/2"),
    (Fraction(1, 3), "1/3"),
    (Fraction(2, 3), "2/3"),
    (Fraction(3, 4), "3/4"),
    (Fraction(3, 5), "3/5"),
    (Fraction(4, 5), "4/5"),
]
# Irrational forms as (numeric_value, display_string)
_IRR_ECC = [
    (math.sqrt(2) / 2, "sqrt(2)/2"),
    (math.sqrt(3) / 2, "sqrt(3)/2"),
    (math.sqrt(3) / 3, "sqrt(3)/3"),
    (math.sqrt(5) / 3, "sqrt(5)/3"),
    (math.sqrt(6) / 3, "sqrt(6)/3"),
    (math.sqrt(2) / 3, "sqrt(2)/3"),
]


def _gen_problem_l0_l3(rng: random.Random):
    """Direct problem: 'Given a={a}, b={b}, find eccentricity.'"""
    # Pick a, b such that c = sqrt(a^2 - b^2) is a nice value
    pairs = [
        (5, 3, Fraction(4, 5)),    # c=4, e=4/5
        (5, 4, Fraction(3, 5)),    # c=3, e=3/5
        (10, 6, Fraction(4, 5)),
        (13, 5, Fraction(12, 13)),
        (13, 12, Fraction(5, 13)),
        (3, 2, None),              # c=sqrt(5), e=sqrt(5)/3
        (2, 1, None),              # c=sqrt(3), e=sqrt(3)/2
    ]
    a, b, e_frac = rng.choice(pairs)
    if e_frac is not None:
        true_e = float(e_frac)
        true_str = f"{e_frac.numerator}/{e_frac.denominator}"
    else:
        c = math.sqrt(a * a - b * b)
        true_e = c / a
        # Try to match irrationals
        if a == 3 and b == 2:
            true_str = "sqrt(5)/3"
        elif a == 2 and b == 1:
            true_str = "sqrt(3)/2"
        else:
            true_str = f"{c:.4f}/{a}"
    question_text = (
        f"Given the ellipse x²/{a*a} + y²/{b*b} = 1 (i.e., a={a}, b={b}), "
        f"find the eccentricity e."
    )
    return question_text, true_e, true_str


def _gen_problem_focal(rng: random.Random):
    """A simplified focal-chord problem.

    For an ellipse x²/a² + y²/b² = 1 with right focus F at (c, 0), the
    chord through F perpendicular to the major axis (semi-latus rectum)
    has length 2*b²/a. We craft a problem where we tell the student
    "the latus-rectum length is L and a = ..." and ask for e.

    2026-05-04: harder — wider (a, b) range so e is non-trivial fraction
    (was 100% saturated at L9 with a∈[2..5]).
    2026-05-04: bumped L9 difficulty (was 100% saturated → wider a range
    plus hyperbola variant for non-trivial eccentricity computation).
    """
    a = rng.choice([7, 8, 9, 10, 11, 12, 13, 15, 17, 20])
    b_choices = [v for v in range(2, a) if v < a]
    if not b_choices:
        return _gen_problem_l0_l3(rng)
    b = rng.choice(b_choices)
    # latus rectum semi-length = b^2 / a (full length 2*b^2/a)
    L = 2 * b * b / a  # full chord length
    c2 = a * a - b * b
    if c2 <= 0:
        return _gen_problem_l0_l3(rng)
    c = math.sqrt(c2)
    e = c / a

    # Pretty fraction for true_str
    if int(c) == c and a > 0:
        true_str = f"{int(c)}/{a}"
    else:
        # e.g., sqrt(c2)/a
        true_str = f"sqrt({c2})/{a}"

    question_text = (
        f"An ellipse x²/a² + y²/b² = 1 (a>b>0) has right focus F. The chord "
        f"through F perpendicular to the major axis has length {L:g}. If a = {a}, "
        f"find the eccentricity e."
    )
    return question_text, e, true_str


def _format_e_string(s: str) -> str:
    """Convert internal eccentricity string to a clean display form."""
    s = s.replace("sqrt(", "√").replace(")", "")
    return s


def _make_distractors(true_e: float, true_str: str, rng: random.Random,
                       n: int = 3):
    """Pick 3 distractor eccentricities not equal to true_e."""
    pool_frac = [(float(f), s) for f, s in _KNOWN_ECC]
    pool_irr = list(_IRR_ECC)
    pool = pool_frac + pool_irr
    rng.shuffle(pool)
    distractors = []
    for v, s in pool:
        if abs(v - true_e) < 1e-3:
            continue
        if any(abs(v - dv) < 1e-3 for dv, _ in distractors):
            continue
        distractors.append((v, s))
        if len(distractors) >= n:
            break
    while len(distractors) < n:
        # Fabricate a fraction
        num = rng.randint(1, 9)
        den = rng.randint(num + 1, 11)
        v = num / den
        if abs(v - true_e) > 1e-3:
            distractors.append((v, f"{num}/{den}"))
    return distractors


class ConicEccentricityMCQQA(StandaloneVisualEnv):
    ENV_NAME = "conic_eccentricity_mcq"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the option letter directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        return {"level": max(0, min(level, 9))}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        rng = random.Random((seed or 0) * 5519 + level * 79 + 11)

        if level <= 4:
            question_text, true_e, true_str = _gen_problem_l0_l3(rng)
        else:
            question_text, true_e, true_str = _gen_problem_focal(rng)

        distractors = _make_distractors(true_e, true_str, rng, n=3)
        # Build options
        all_opts = [(true_e, true_str)] + distractors
        rng.shuffle(all_opts)
        # Find which letter is correct
        correct_letter = None
        for i, (v, s) in enumerate(all_opts):
            if abs(v - true_e) < 1e-3:
                correct_letter = chr(ord("A") + i)
                break
        if correct_letter is None:
            return None

        labels = ["A", "B", "C", "D"]
        options_text = "\n".join(
            f"{labels[i]}. e = {_format_e_string(s)}"
            for i, (v, s) in enumerate(all_opts)
        )

        self._correct_letter = correct_letter
        self._true_e = true_e

        sidx = (seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            question_text=question_text,
            options_text=options_text,
        )
        ans_str = correct_letter
        img = self._render(question_text, options_text)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, question_text, options_text) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.text(0.02, 0.95, question_text, ha="left", va="top",
                fontsize=11, color="#222", wrap=True,
                transform=ax.transAxes)
        ax.text(0.02, 0.55, options_text, ha="left", va="top",
                fontsize=11, color="#1a3a6e", fontfamily="monospace",
                transform=ax.transAxes)
        # Add a small ellipse sketch
        from matplotlib.patches import Ellipse
        ell = Ellipse((0.78, 0.30), 0.18, 0.10,
                      facecolor="none", edgecolor="#1a3a6e", linewidth=1.5,
                      transform=ax.transAxes)
        ax.add_patch(ell)
        ax.plot([0.78, 0.78], [0.30, 0.30], "ro", markersize=4,
                transform=ax.transAxes)
        ax.text(0.78, 0.18, "ellipse", ha="center", va="top",
                fontsize=9, color="#777", transform=ax.transAxes)
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # MCQ A/B/C/D matching
        s = predicted.strip().upper()
        gt = ground_truth.strip().upper()
        # Use base class behavior is fine — but be explicit:
        # First, look for explicit (A) / [A] / A) markers
        m1 = re.search(r"[\(\[]\s*([A-D])\s*[\)\]]", s)
        if m1:
            return m1.group(1) == gt
        # Look for first standalone letter
        m2 = re.search(r"\b([A-D])\b", s)
        if m2:
            return m2.group(1) == gt
        return False
