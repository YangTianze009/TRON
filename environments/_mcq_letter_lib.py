"""Shared helper for converting numeric/free-text answers into MCQ-letter style
MCQ-letter answers.

reference expects single-letter MCQ answers (A/B/C/D/E). The standalone_base
verifier already does first-standalone-letter regex fallback, so a ground-truth
of a single uppercase letter A-H matches outputs like:
   <answer>A</answer>
   "(A)"
   "The answer is A"

This module provides helpers to:
  1. Generate plausible numeric / ratio / label distractors near a correct value.
  2. Format option lists in reference style:  (A)val_a (B)val_b (C)val_c (D)val_d
  3. Build a complete MCQ-letter prompt suffix with format instructions.
  4. Wrap a numeric/free-text answer into an MCQ-letter problem.

Usage in env._generate_problem:

    from ._mcq_letter_lib import maybe_to_mcq_letter

    # ... compute (question_text, numeric_answer, image) as before ...
    question, ans, img = self._build_problem(...)

    # Probabilistically convert to letter mode:
    question, ans = maybe_to_mcq_letter(question, ans, rng, prob=0.5)
    return question, ans, img
"""
import random
import re
from typing import List, Optional, Tuple, Union


_MCQ_INSTRUCTION_TEMPLATES = [
    "Choose the correct answer letter (A-{last}). Place your answer in <answer>X</answer>.",
    "Select the correct option (A-{last}) and place the single letter in <answer>X</answer>.",
    "Pick the correct letter (A-{last}). Reply with the letter inside <answer>X</answer>.",
    "Which option is correct? Place the single letter (A-{last}) in <answer>X</answer>.",
]


def _format_num(x: float, decimals: int = 0) -> str:
    """Format a numeric value cleanly.

    - If x is an integer (within 1e-6), render as int.
    - Otherwise render with `decimals` digits, stripping trailing zeros.
    """
    try:
        xf = float(x)
    except (ValueError, TypeError):
        return str(x)
    if abs(xf - round(xf)) < 1e-6:
        return str(int(round(xf)))
    if decimals <= 0:
        decimals = 2
    return f"{xf:.{decimals}f}".rstrip("0").rstrip(".")


def _is_ratio(s: str) -> bool:
    """Match strings like '2:3', '10:15', '1.5:2.5'."""
    return bool(re.match(r"^\s*-?\d+(\.\d+)?\s*:\s*-?\d+(\.\d+)?\s*$", str(s)))


def _is_numeric(s) -> bool:
    """Whether s parses as a single numeric value (not a ratio, not a label)."""
    try:
        float(str(s).strip().replace(",", "").rstrip("%"))
        return True
    except (ValueError, TypeError):
        return False


def _gen_numeric_distractors(
    correct: float, rng: random.Random, k: int = 3, decimals: int = 0,
) -> List[float]:
    """Generate k distractor numeric values near `correct`.

    Strategy (MCQ-letter style):
      - one off-by-step (e.g. ±10% rounded)
      - one off-by-fraction (e.g. ½× or 2× the correct value)
      - one off-by-small-noise (e.g. ±5% rounded)
      - additional ones (if k>3): noise drawn from ±15%

    Distractors are guaranteed unique and != correct.
    """
    distractors = set()
    correct_f = float(correct)
    abs_c = max(abs(correct_f), 1.0)

    candidates = []
    # Off-by-fraction
    for mult in (0.5, 2.0, 0.25, 4.0, 1.5, 0.75):
        candidates.append(correct_f * mult)
    # Off-by-percent
    for pct in (0.05, 0.10, 0.15, -0.05, -0.10, -0.15, 0.20, -0.20):
        candidates.append(correct_f * (1 + pct))
    # Off-by-fixed step (small)
    if abs_c > 5:
        for delta in (1, -1, 5, -5, 2, -2):
            candidates.append(correct_f + delta)
    # Off-by-additive (10% of magnitude)
    for delta in (abs_c * 0.1, -abs_c * 0.1, abs_c * 0.07, -abs_c * 0.07):
        candidates.append(correct_f + delta)

    # Round each candidate consistently with the correct value's decimal style
    def _round_like(v):
        if abs(correct_f - round(correct_f)) < 1e-6:
            r = int(round(v))
            # avoid trivially equal
            if r == int(round(correct_f)):
                r += 1
            return float(r)
        # decimal: keep `decimals` precision
        d = decimals if decimals > 0 else 2
        r = round(v, d)
        if abs(r - correct_f) < 10 ** (-d):
            r += 10 ** (-d) * (1 if v > correct_f else -1)
        return r

    rng.shuffle(candidates)
    for c in candidates:
        c_round = _round_like(c)
        if abs(c_round - correct_f) < 1e-9:
            continue
        # Avoid duplicates within distractors
        is_dup = any(abs(c_round - d) < 1e-9 for d in distractors)
        if is_dup:
            continue
        distractors.add(c_round)
        if len(distractors) >= k:
            break

    # Fallback: if not enough unique distractors found, pad with random noise
    attempts = 0
    while len(distractors) < k and attempts < 50:
        noise = rng.uniform(-0.30, 0.30)
        c = _round_like(correct_f * (1 + noise))
        if abs(c - correct_f) > 1e-9 and not any(abs(c - d) < 1e-9 for d in distractors):
            distractors.add(c)
        attempts += 1

    return list(distractors)[:k]


def _gen_ratio_distractors(
    correct_ratio: str, rng: random.Random, k: int = 3,
) -> List[str]:
    """Generate k distractor ratio strings near `correct_ratio` (e.g. '2:3')."""
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*$",
                 str(correct_ratio))
    if not m:
        return []
    a = float(m.group(1))
    b = float(m.group(2))

    candidates = []
    # Inverted
    candidates.append(f"{_format_num(b)}:{_format_num(a)}")
    # Off-by-one in numerator/denominator
    if a > 1:
        candidates.append(f"{_format_num(a + 1)}:{_format_num(b)}")
        candidates.append(f"{_format_num(a - 1)}:{_format_num(b)}")
    if b > 1:
        candidates.append(f"{_format_num(a)}:{_format_num(b + 1)}")
        candidates.append(f"{_format_num(a)}:{_format_num(b - 1)}")
    # Doubled / halved
    candidates.append(f"{_format_num(a * 2)}:{_format_num(b)}")
    candidates.append(f"{_format_num(a)}:{_format_num(b * 2)}")
    # Alternate close ratios
    candidates.append(f"{_format_num(a + 1)}:{_format_num(b + 1)}")

    distractors = []
    correct_norm = f"{_format_num(a)}:{_format_num(b)}"
    rng.shuffle(candidates)
    for c in candidates:
        if c == correct_norm or c in distractors:
            continue
        distractors.append(c)
        if len(distractors) >= k:
            break
    return distractors[:k]


def _gen_label_distractors(
    correct_label: str, candidate_pool: List[str], rng: random.Random, k: int = 3,
) -> List[str]:
    """Pick k distractor labels from a candidate pool excluding the correct one."""
    pool = [x for x in candidate_pool if x != correct_label]
    if len(pool) < k:
        return pool[:]
    return rng.sample(pool, k)


def make_mcq_letter(
    correct_answer: Union[str, float, int],
    rng: random.Random,
    n_options: int = 4,
    candidate_pool: Optional[List[str]] = None,
    prefix_each: str = " ",
) -> Tuple[str, str]:
    """Convert a `correct_answer` into MCQ-letter format.

    Returns (options_block, correct_letter) where:
      - options_block is a string like "(A)val1 (B)val2 (C)val3 (D)val4"
      - correct_letter is a single uppercase letter A..E

    Args:
        correct_answer: numeric (int/float/str) or string label or ratio "a:b"
        rng: random.Random for reproducibility
        n_options: 4 or 5
        candidate_pool: for label-style answers, pool of plausible alternatives
        prefix_each: separator between (X)val tokens (default single space)

    Returns:
        (options_block_text, correct_letter)

    If distractors cannot be generated (too few plausible alternatives), returns
    (None, None) to signal the caller to keep the original numeric/free-text mode.
    """
    n_options = max(2, min(n_options, 5))
    correct_str = str(correct_answer).strip()

    # Branch on answer type
    if _is_ratio(correct_str):
        distractors = _gen_ratio_distractors(correct_str, rng, k=n_options - 1)
    elif _is_numeric(correct_str):
        try:
            cval = float(correct_str.replace(",", "").rstrip("%"))
        except (ValueError, TypeError):
            return None, None
        # Detect decimal precision in the correct value
        decimals = 0
        if "." in correct_str:
            decimals = len(correct_str.split(".")[1].rstrip("0"))
            if decimals == 0:
                decimals = 2
        distractor_nums = _gen_numeric_distractors(
            cval, rng, k=n_options - 1, decimals=decimals)
        # Format consistently with correct value
        if decimals == 0 and abs(cval - round(cval)) < 1e-6:
            distractors = [str(int(round(d))) for d in distractor_nums]
            correct_str = str(int(round(cval)))
        else:
            d = decimals if decimals > 0 else 2
            distractors = [_format_num(x, decimals=d) for x in distractor_nums]
    else:
        # Label / string answer — use candidate_pool
        if not candidate_pool:
            return None, None
        distractors = _gen_label_distractors(
            correct_str, candidate_pool, rng, k=n_options - 1)

    if len(distractors) < n_options - 1:
        return None, None

    # Shuffle correct + distractors to determine letter assignment
    options = [correct_str] + list(distractors)
    rng.shuffle(options)
    correct_idx = options.index(correct_str)
    correct_letter = "ABCDE"[correct_idx]

    # Format option block
    parts = []
    for i, opt in enumerate(options):
        parts.append(f"({chr(ord('A') + i)}){opt}")
    options_block = prefix_each.join(parts)
    return options_block, correct_letter


def build_mcq_prompt_suffix(n_options: int, rng: random.Random) -> str:
    """Generate a 'choose A-D / A-E / etc.' instruction string."""
    last = "ABCDE"[max(2, min(n_options, 5)) - 1]
    template = rng.choice(_MCQ_INSTRUCTION_TEMPLATES)
    return template.format(last=last)


# Common phrases used to introduce option lists in MCQ-letter mode.
_OPTION_INTROS = [
    "Options:",
    "Choices:",
    "",  # bare
]


def maybe_to_mcq_letter(
    question: str,
    answer: Union[str, float, int],
    rng: random.Random,
    prob: float = 0.5,
    n_options: int = 4,
    candidate_pool: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Probabilistically convert (question, numeric_answer) to MCQ-letter mode.

    With probability `prob`, returns (new_question, letter).
    Otherwise returns (question, answer) unchanged.

    The new_question is the original question with:
      - the `Place ... <answer>...</answer>` instruction stripped
      - an `(A)opt1 (B)opt2 (C)opt3 (D)opt4` block appended
      - an MCQ instruction "Choose the correct answer letter (A-D)..." appended

    If MCQ conversion fails (e.g. cannot generate enough distractors), returns
    the original (question, answer) unchanged.
    """
    if rng.random() >= prob:
        return question, answer

    options_block, letter = make_mcq_letter(
        answer, rng, n_options=n_options, candidate_pool=candidate_pool)
    if options_block is None or letter is None:
        return question, answer

    # Strip the existing "<answer>...</answer>" instruction tail. We
    # backtrack from `<answer>` to find the prior sentence-start (looking
    # for "Place|Report|Reply|Give|Numeric|Answer|Integer|Total" or a sentence
    # boundary), then drop everything from there through the end of the
    # current sentence. Use a permissive multi-step strip:
    new_q = question
    # Step 1: strip the answer-format instruction sentence. Find the last
    # sentence (ending with "." or "?") that mentions <answer>...</answer>;
    # we strip from a preceding sentence boundary, OR from a known
    # instruction verb if no boundary precedes <answer>.
    # First, find the last sentence with <answer>:
    ans_match = re.search(r"<answer>[^<]*</answer>", new_q)
    if ans_match:
        # Walk back from match start to find the last sentence boundary
        # (".", "?", "!", or start of string). The substring we want to drop
        # spans [boundary+1 .. end of trailing "."].
        start = ans_match.start()
        # Look for last sentence boundary (".", "?", "!") before start, but
        # NOT inside parenthetical "(...)" parts.
        boundary = -1
        depth = 0
        for i in range(start - 1, -1, -1):
            ch = new_q[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth = max(0, depth - 1)
            if depth == 0 and ch in ".?!":
                # Skip if this is inside a number like "2.5" or abbreviation.
                # Conservative: only treat as boundary if followed by space/uppercase.
                if i + 1 < len(new_q) and (new_q[i + 1].isspace() or new_q[i + 1] == "\n"):
                    boundary = i
                    break
        # Drop from boundary+1 (skip space) through the next "." after
        # </answer> (= end-of-instruction sentence).
        end_match = re.search(r"</answer>[^.]*\.?", new_q[ans_match.end() - 1:])
        end = ans_match.end()
        if end_match:
            end = ans_match.end() - 1 + end_match.end()
        if boundary >= 0:
            cut_start = boundary + 1
            new_q = new_q[:boundary + 1] + new_q[end:]
        else:
            # No prior sentence boundary — strip from a leading instruction verb
            sub = new_q
            verb_match = re.search(
                r"(?:Place|Report|Reply\s+with|Give|Provide|State)\b",
                sub,
                flags=re.IGNORECASE,
            )
            if verb_match:
                new_q = sub[:verb_match.start()].rstrip() + sub[end:]
            else:
                new_q = sub[:ans_match.start()].rstrip() + sub[end:]
    # Step 3: residual <answer>...</answer> anywhere (catch-all)
    new_q = re.sub(r"<answer>[^<]*</answer>", "", new_q)
    # Normalize trailing punctuation: collapse "?." or "?.." to "?", and
    # drop double ".." down to ".".
    new_q = re.sub(r"\?\s*\.+\s*$", "?", new_q.rstrip())
    new_q = re.sub(r"\.{2,}\s*$", ".", new_q.rstrip())
    new_q = new_q.rstrip()
    if not (new_q.endswith("?") or new_q.endswith(".") or new_q.endswith(":")):
        new_q += "."

    intro = rng.choice(_OPTION_INTROS)
    instr = build_mcq_prompt_suffix(n_options, rng)
    if intro:
        new_q = f"{new_q}\n{intro} {options_block}\n{instr}"
    else:
        new_q = f"{new_q}\n{options_block}\n{instr}"
    return new_q, letter


def maybe_to_unit_mcq(
    question: str,
    answer: Union[str, float, int],
    rng: random.Random,
    prob: float = 0.5,
    unit: Optional[str] = None,
    n_options: int = 4,
) -> Tuple[str, str]:
    """exam-style MCQ wrapper. Probabilistically converts a numeric answer to
    a 4-5 option letter MCQ that mirrors the dominant reference surface format:

        Options: A. <val> <unit>; B. <val> <unit>; ...; E. No correct answer

    The 5th option ("E. No correct answer") is included when n_options==5;
    correct answer always lies in A-D so the correct letter never becomes E.

    Args:
        question: original numeric-mode prompt (with `<answer>...</answer>`
                  instruction tail to be stripped).
        answer:   correct numeric answer (int/float/str).
        rng:      random.Random.
        prob:     probability of converting (else returns originals unchanged).
        unit:     unit string appended to each numeric option (e.g. "cm²", "m").
                  None / "" disables units.
        n_options: 4 or 5. n_options==5 adds "E. No correct answer".

    Returns (new_question, correct_letter) on conversion, else
    (question, str(answer)) unchanged.
    """
    if rng.random() >= prob:
        return question, str(answer)
    n_options = 5 if n_options >= 5 else 4
    # Build the 4-letter (A-D) value MCQ first.
    options_block, letter = make_mcq_letter(answer, rng, n_options=4)
    if options_block is None or letter is None:
        return question, str(answer)
    # Append unit to each numeric value if requested. options_block format:
    # "(A)val1 (B)val2 (C)val3 (D)val4". Use a regex pass.
    if unit:
        options_block = re.sub(
            r"(\([A-D]\))(-?\d+(?:\.\d+)?)",
            lambda m: f"{m.group(1)}{m.group(2)} {unit}",
            options_block,
        )
    # Build option lines in the exam-style surface format ("A. val unit; B. ...")
    parts = re.findall(r"\(([A-D])\)([^\(]+)", options_block)
    opt_lines = [f"{p[0]}. {p[1].strip()}" for p in parts]
    if n_options == 5:
        opt_lines.append("E. No correct answer")
    options_text = "; ".join(opt_lines)

    # Strip the existing "<answer>...</answer>" instruction tail (reuse logic).
    new_q = question
    ans_match = re.search(r"<answer>[^<]*</answer>", new_q)
    if ans_match:
        start = ans_match.start()
        boundary = -1
        depth = 0
        for i in range(start - 1, -1, -1):
            ch = new_q[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth = max(0, depth - 1)
            if depth == 0 and ch in ".?!":
                if i + 1 < len(new_q) and (new_q[i + 1].isspace() or new_q[i + 1] == "\n"):
                    boundary = i
                    break
        end_match = re.search(r"</answer>[^.]*\.?", new_q[ans_match.end() - 1:])
        end = ans_match.end()
        if end_match:
            end = ans_match.end() - 1 + end_match.end()
        if boundary >= 0:
            new_q = new_q[:boundary + 1] + new_q[end:]
        else:
            verb_match = re.search(
                r"(?:Place|Report|Reply\s+with|Give|Provide|State|Answer)\b",
                new_q,
                flags=re.IGNORECASE,
            )
            if verb_match:
                new_q = new_q[:verb_match.start()].rstrip() + new_q[end:]
            else:
                new_q = new_q[:ans_match.start()].rstrip() + new_q[end:]
    new_q = re.sub(r"<answer>[^<]*</answer>", "", new_q)
    new_q = re.sub(r"\?\s*\.+\s*$", "?", new_q.rstrip())
    new_q = re.sub(r"\.{2,}\s*$", ".", new_q.rstrip())
    new_q = new_q.rstrip()
    if not (new_q.endswith("?") or new_q.endswith(".") or new_q.endswith(":")):
        new_q += "."

    last_letter = "ABCDE"[n_options - 1]
    instr = build_mcq_prompt_suffix(n_options, rng)
    new_q = f"{new_q}\nOptions: {options_text}\n{instr}"
    return new_q, letter


__all__ = [
    "make_mcq_letter",
    "maybe_to_mcq_letter",
    "maybe_to_unit_mcq",
    "build_mcq_prompt_suffix",
]
