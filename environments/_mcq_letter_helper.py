"""
Shared helper for C20 (reference Multi Choice mode) — converts a numeric or
short-text answer into a 4-option lettered MCQ block whose only valid answer
is a bare lowercase letter ``a|b|c|d``.

Sample target style (from design notes Multi Choice samples):

    Q-IDX 1869 verbatim:
        "what is the difference in percentage points between chinese and
        australian respondents who believe music helps create a better
        shopping environment?\n\na) 10 percentage points\nb) 8 percentage
        points\nc) 12 percentage points\nd) 6 percentage points"
        GT = ``b`` (bare lowercase letter)

The motivation for this helper is the +9.35 delta on Multi Choice subtask
in v3-step175 vs base — the gain was largely format-discipline (model learns
to emit ``c`` instead of ``c) 9.23%``). To stress-train this we let any
chart env optionally wrap its numeric/short-text answer into this 4-option
form when called via ``maybe_mcq_letter_wrap()``.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple


def _format_value(v) -> str:
    """Render a numeric value cleanly so distractor formatting matches."""
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        # Trim trailing zeros after rounding to 2dp.
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def build_numeric_mcq(
    rng: random.Random,
    correct_value: float,
    *,
    suffix: str = "",
    n_options: int = 4,
) -> Tuple[List[str], str]:
    """Build a list of MCQ option *strings* and return the lowercase letter
    of the correct one.

    ``suffix`` is appended to every option (e.g. ``"%"`` or ``" billion"``).
    Distractors are placed within +/- 5-15 % of the absolute value (or +/- 1-3
    of the integer if the value is small).
    """
    correct_value = float(correct_value)
    abs_v = abs(correct_value) if correct_value != 0 else 1.0
    # ±5-15 % offsets, plus a couple absolute-step alternatives to handle
    # very small magnitudes ("3" gets distractors 2/4/5).
    offsets_pct = [0.05, -0.07, 0.11, -0.13, 0.15, -0.05, 0.08, -0.10]
    offsets_abs = [1, -1, 2, -2, 3, -3]
    rng.shuffle(offsets_pct)
    rng.shuffle(offsets_abs)

    # If correct_value is clearly an integer, prefer integer distractors so
    # the option list looks consistent (e.g., 23 / 21 / 25 / 19 not 23.4).
    is_int = float(correct_value).is_integer() and abs_v >= 2

    distractors: List[float] = []
    candidates = []
    for off in offsets_pct:
        candidates.append(correct_value + abs_v * off)
    for off in offsets_abs:
        candidates.append(correct_value + off)
    rng.shuffle(candidates)

    for cand in candidates:
        if is_int:
            cand = float(round(cand))
        else:
            cand = round(cand, 2)
        # Reject if too close to correct or to existing distractor (5%
        # cushion so model isn't forced to pick between near-duplicates).
        sep = max(abs_v * 0.04, 0.5 if is_int else 0.05)
        if abs(cand - correct_value) < sep:
            continue
        if any(abs(cand - d) < sep for d in distractors):
            continue
        distractors.append(cand)
        if len(distractors) >= n_options - 1:
            break
    if len(distractors) < n_options - 1:
        # Last-resort fallback: pad with shifted copies; tests will catch
        # the rare degenerate seed.
        i = 1
        while len(distractors) < n_options - 1:
            distractors.append(round(correct_value + i * (abs_v * 0.2 + 1), 2))
            i += 1

    options_vals = [correct_value] + distractors[: n_options - 1]
    rng.shuffle(options_vals)
    correct_idx = options_vals.index(correct_value)
    correct_letter = chr(ord("a") + correct_idx)
    options_str = [f"{_format_value(v)}{suffix}" for v in options_vals]
    return options_str, correct_letter


def build_text_mcq(
    rng: random.Random,
    correct_text: str,
    distractor_pool: List[str],
    *,
    n_options: int = 4,
) -> Tuple[List[str], str]:
    """Build a 4-option text MCQ. ``distractor_pool`` may include the correct
    answer; it will be excluded from the distractor draw."""
    pool = [d for d in distractor_pool if d.lower().strip()
            != correct_text.lower().strip()]
    rng.shuffle(pool)
    distractors = pool[: n_options - 1]
    if len(distractors) < n_options - 1:
        # Pad with simple variant strings so we always have 4 options.
        i = 1
        while len(distractors) < n_options - 1:
            distractors.append(f"Option {i}")
            i += 1
    options_vals = [correct_text] + distractors
    rng.shuffle(options_vals)
    correct_idx = options_vals.index(correct_text)
    return options_vals, chr(ord("a") + correct_idx)


def format_mcq_question(question_stem: str, options: List[str]) -> str:
    """Return the final question string with `\\na) X\\nb) Y...` block and
    a strict letter-only output instruction.

    Output instruction matches the reference Multi Choice prompt verbatim
    (see design notes prompt_context):
        "Your answer should be one of the options letters only: a, b, c or d
         (just the letter itself without any additional text)."
    """
    letters = "abcdefgh"
    opts_block = "\n".join(f"{letters[i]}) {opt}"
                           for i, opt in enumerate(options))
    return (
        f"{question_stem}\n\n{opts_block}\n\n"
        "Your answer should be one of the options letters only: "
        "a, b, c or d (just the letter itself without any additional text)."
    )


def maybe_mcq_letter_wrap(
    rng: random.Random,
    question_stem: str,
    correct_value,
    *,
    suffix: str = "",
    rate: float = 0.30,
) -> Optional[Tuple[str, str]]:
    """Convenience wrapper: with probability ``rate``, returns a
    ``(wrapped_question, lowercase_letter_answer)`` tuple; else ``None`` so
    the caller falls back to its native answer format.

    Returns ``None`` if the correct value can't be parsed numerically.
    """
    if rng.random() >= rate:
        return None
    try:
        v = float(str(correct_value).replace("%", "").replace(",", ""))
    except (ValueError, TypeError):
        return None
    options, letter = build_numeric_mcq(rng, v, suffix=suffix)
    return format_mcq_question(question_stem, options), letter
