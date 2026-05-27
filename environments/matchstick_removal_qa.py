"""
Matchstick equation puzzle: an equation made from matchsticks (operands and
operator + sign segments) is mathematically wrong. Remove exactly ONE matchstick
to make it true. Output the resulting equation (e.g., `5-3=2`).

We model digits with 7-segment representation:
   _
  |_|
  |_|

Segments labelled a..g per standard:
  a (top)
  b (upper-right vertical)
  c (lower-right vertical)
  d (bottom)
  e (lower-left vertical)
  f (upper-left vertical)
  g (middle)

Plus operator/equals matchsticks: '+' has 2 sticks (vertical + horizontal),
'-' has 1 stick (horizontal), '=' has 2 sticks (top + bottom).

Multiple valid solutions accepted — _check_answer parses model's equation,
evaluates LHS vs RHS, and verifies that the digits are 1-stick-removed away
from the original digits.

L0 = `5+3=2` style trivial: only one obvious move (`5-3=2`).
L9 = bigger digits + harder equations (multi-digit operands).
"""
import math
import random
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# 7-segment encoding for digits 0-9 — segment names a,b,c,d,e,f,g.
DIGIT_SEGMENTS = {
    "0": frozenset("abcdef"),
    "1": frozenset("bc"),
    "2": frozenset("abged"),
    "3": frozenset("abgcd"),
    "4": frozenset("fgbc"),
    "5": frozenset("afgcd"),
    "6": frozenset("afgcde"),
    "7": frozenset("abc"),
    "8": frozenset("abcdefg"),
    "9": frozenset("abcdfg"),
}

# Reverse: which digits can be reached from a given digit by ADDING N sticks
# OR REMOVING N sticks. We compute on the fly.
def _digits_reachable(d: str, delta: int) -> List[str]:
    """delta=-1 → digits reachable by removing 1 stick; +1 → by adding 1."""
    if d not in DIGIT_SEGMENTS:
        return []
    src = DIGIT_SEGMENTS[d]
    out = []
    for cand, segs in DIGIT_SEGMENTS.items():
        if delta == -1:
            # cand is src minus exactly 1 stick
            if segs.issubset(src) and len(src - segs) == 1:
                out.append(cand)
        elif delta == 1:
            if src.issubset(segs) and len(segs - src) == 1:
                out.append(cand)
    return out


# Operator transforms (matchstick view)
# '+' is made of 2 sticks (one horizontal, one vertical):
#   removing one stick → '-' (single horizontal)
#   removing both is not allowed (we remove one)
# '-' is 1 stick: removing leaves nothing (invalid digit-equation), but adding
#   1 stick gives '+'.
# '=' is 2 sticks (two parallel horizontal):
#   removing 1 stick → '-' (single horizontal). But that breaks equation
#   structure — we won't allow operator=becoming-minus.
# We implement TWO families:
#  Family A: Remove one stick from the LHS digit, transforming it into
#            another digit so the equation becomes true. (e.g. 5→6, 9→3, etc.)
#  Family B: Remove one stick from a '+' operator, turning it into '-' (or
#            same). e.g., 5+3=2 → 5-3=2 (true if 5-3=2).
#
# Generation: pick a TARGET TRUE equation, then INTRODUCE one extra stick
# somewhere (so the displayed equation is the wrong one). The "remove one
# stick" puzzle is the inverse.


_TEMPLATES = [
    "Fix the matchstick equation by removing exactly one matchstick.\n\n"
    "### Game Rules:\n"
    "1. The equation shown is currently mathematically wrong (LHS != RHS).\n"
    "2. Remove EXACTLY ONE matchstick from a digit or operator to make the equation true.\n"
    "3. Each digit is rendered as a 7-segment display; the `+` operator is two sticks (horizontal + vertical), the `-` operator is one stick, the `=` operator is two parallel sticks.\n"
    "4. Removing a stick from `+` produces `-`. Other valid removals turn one digit into another (e.g., 9 -> 3 by removing top-right stick).\n"
    "5. Multiple valid resulting equations may exist; output any one.\n\n"
    "### Coordinate System:\n"
    "- Equation is a single line: digits and operators left-to-right, no spaces.\n"
    "- Allowed operators: `+`, `-`, `=`. Allowed digits: 0-9.\n\n"
    "### Current Puzzle State:\n"
    "- Equation shown: {equation}\n\n"
    "### Output Format:\n"
    "Output the corrected equation in `LHS=RHS` form (no spaces) inside <answer>...</answer>.\n"
    "Example: <answer>5-3=2</answer>",

    "Remove exactly one matchstick from the equation below to make it true.\n\n"
    "### Game Rules:\n"
    "- The equation as shown is mathematically false.\n"
    "- Remove ONE matchstick (from a digit or operator) so the resulting equation is true.\n"
    "- Digits are 7-segment; `+` is 2 sticks (horizontal + vertical); `-` is 1 stick; `=` is 2 sticks.\n"
    "- Multiple valid solutions may exist; provide any one.\n\n"
    "### Coordinate System:\n"
    "- Single-line `LHS=RHS` form.\n"
    "- Operators: `+`, `-`, `=`. Digits: 0-9.\n\n"
    "### Current Puzzle State:\n"
    "- Equation: {equation}\n\n"
    "### Output Format:\n"
    "Output the corrected equation as `LHS=RHS` (no spaces) inside <answer>...</answer>.",

    "Solve the matchstick puzzle: remove ONE stick so the equation becomes true.\n\n"
    "### Game Rules:\n"
    "Each digit uses a 7-segment representation; `+` has 2 sticks, `-` has 1 stick, `=` has 2 sticks. Remove exactly one stick so LHS = RHS.\n\n"
    "### Coordinate System:\n"
    "- Equation is a single line `LHS=RHS`, no spaces.\n\n"
    "### Current Puzzle State:\n"
    "- Equation: {equation}\n\n"
    "### Output Format:\n"
    "Output the corrected equation `LHS=RHS` inside <answer>...</answer>.",
]


# L0-L2 probe: YES/NO probe — is the equation in the image correct?
# Half the time we show a TRUE equation (answer YES); half the time WRONG (answer NO).
_PROBE_TEMPLATES = [
    "Decide whether the matchstick equation below is mathematically correct as shown.\n\n"
    "### Game Rules:\n"
    "1. The equation is shown using matchstick digits and operators.\n"
    "2. Determine whether LHS equals RHS as currently displayed.\n\n"
    "### Coordinate System:\n"
    "- Equation is a single line: digits and operators left-to-right.\n"
    "- Operators: `+`, `-`, `=`.\n\n"
    "### Current Puzzle State:\n"
    "- Equation shown: {equation}\n\n"
    "### Output Format:\n"
    "Output `YES` or `NO` (uppercase) inside <answer>...</answer>.\n"
    "Example: <answer>YES</answer>",

    "Is the matchstick equation correct?\n\n"
    "### Game Rules:\n"
    "- Determine whether the displayed equation evaluates to a true arithmetic statement.\n\n"
    "### Coordinate System:\n"
    "- Single-line `LHS OP RHS` form, where OP is `=`.\n\n"
    "### Current Puzzle State:\n"
    "- Equation: {equation}\n\n"
    "### Output Format:\n"
    "Output `YES` or `NO` inside <answer>...</answer>.",

    "Examine the matchstick equation; decide if it is true as displayed.\n\n"
    "### Game Rules:\n"
    "Standard arithmetic: check if LHS evaluates to RHS.\n\n"
    "### Coordinate System:\n"
    "- Single-line equation; operators `+`, `-`, `=`.\n\n"
    "### Current Puzzle State:\n"
    "- Equation: {equation}\n\n"
    "### Output Format:\n"
    "Output `YES` or `NO` inside <answer>...</answer>.",
]


def _eval_eq(eq: str) -> Optional[bool]:
    """Parse 'LHS=RHS', evaluate both sides as integers, return True if equal."""
    eq = eq.replace(" ", "")
    if "=" not in eq:
        return None
    parts = eq.split("=")
    if len(parts) != 2:
        return None
    lhs, rhs = parts
    if not lhs or not rhs:
        return None
    try:
        # Only allow digits and + - operators
        for ch in lhs + rhs:
            if not (ch.isdigit() or ch in "+-"):
                return None
        # Reject leading + or -; require digit at start of each operand
        # Actually let's just evaluate — Python handles - signs
        # Disallow Python's eval shenanigans by only-digit-and-op check above.
        l_val = eval(lhs, {"__builtins__": {}})
        r_val = eval(rhs, {"__builtins__": {}})
    except Exception:
        return None
    return l_val == r_val


def _parse_equation_to_tokens(eq: str) -> Optional[List[str]]:
    """Tokenize equation into ['1', '+', '2', '=', '3'] etc."""
    eq = eq.replace(" ", "")
    out: List[str] = []
    cur = ""
    for ch in eq:
        if ch.isdigit():
            cur += ch
        elif ch in "+-=":
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
        else:
            return None
    if cur:
        out.append(cur)
    return out


def _stick_equiv(a: str, b: str, op_change: bool = True) -> Tuple[bool, int]:
    """Return (is_one_stick_diff, sticks_to_remove). sticks_to_remove > 0 means
    a was modified by removing that many sticks to obtain b. We allow:
      - digit → digit by removing N sticks (N=1 means one removal)
      - '+' → '-' (removing 1 stick from +) when op_change
      - '=' → '-' (NOT allowed in our scheme — would break equation)
    Returns (False, 0) if not transformable."""
    if a == b:
        return True, 0  # no removal
    # digit → digit
    if a.isdigit() and b.isdigit() and len(a) == 1 and len(b) == 1:
        sa = DIGIT_SEGMENTS[a]
        sb = DIGIT_SEGMENTS[b]
        if sb.issubset(sa):
            return True, len(sa - sb)
        return False, 0
    # operator change
    if op_change and a == "+" and b == "-":
        return True, 1
    return False, 0


def _diff_sticks_token_seq(orig_tokens: List[str], new_tokens: List[str]) -> Optional[int]:
    """Compute total number of sticks removed (and ensure no sticks ADDED).
    Returns None if any token can't be reached from orig by removing sticks
    only, OR if the token structure differs (number of operands etc.)."""
    if len(orig_tokens) != len(new_tokens):
        return None
    total = 0
    for o, n in zip(orig_tokens, new_tokens):
        if o.isdigit() and n.isdigit():
            if len(o) != len(n):
                return None  # multi-digit length mismatch
            for od, nd in zip(o, n):
                ok, k = _stick_equiv(od, nd, op_change=False)
                if not ok:
                    return None
                total += k
        elif o == n:
            continue
        else:
            ok, k = _stick_equiv(o, n, op_change=True)
            if not ok:
                return None
            total += k
    return total


class MatchstickRemovalQA(StandaloneVisualEnv):
    ENV_NAME = "matchstick_removal_qa"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Identify the wrong digit/operator and pick the single "
        "match to remove. Final answer in any of: <answer>X</answer>, "
        "\\boxed{{X}}, or `Final answer: X`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode — YES/NO "is equation correct?".
        if level <= 2:
            modes = ["op_only"]
            return {"level": level, "modes": modes, "mx": 9, "probe": True}
        if level <= 5:
            modes = ["op_only", "digit_swap"]
            mx = 9
        elif level <= 7:
            modes = ["op_only", "digit_swap"]
            mx = 9
        else:
            modes = ["digit_swap"]
            mx = 9
        return {"level": level, "modes": modes, "mx": mx, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1811 + level * 67 + 31)
        is_probe = cfg.get("probe", False)

        # PROBE mode: show a simple equation, ask YES/NO if correct.
        if is_probe:
            # 50/50 split between true and false equations.
            mx = cfg["mx"]
            for _ in range(20):
                a = rng.randint(1, mx)
                b = rng.randint(1, mx)
                op = rng.choice(["+", "-"])
                if op == "-" and b > a:
                    a, b = b, a
                c_correct = a + b if op == "+" else a - b
                if c_correct < 0:
                    continue
                show_correct = rng.random() < 0.5
                if show_correct:
                    shown = f"{a}{op}{b}={c_correct}"
                    expected = "YES"
                else:
                    # Show wrong c
                    delta = rng.choice([-2, -1, 1, 2, 3])
                    c_shown = c_correct + delta
                    if c_shown < 0 or c_shown == c_correct:
                        continue
                    shown = f"{a}{op}{b}={c_shown}"
                    expected = "NO"
                # Verify
                if _eval_eq(shown) is None:
                    continue
                if (_eval_eq(shown) is True) != (expected == "YES"):
                    continue
                self._matchstick_shown = shown
                self._probe_mode = True
                self._probe_answer = expected
                self._matchstick_solutions = set()  # not used in probe
                sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
                question = _PROBE_TEMPLATES[sidx].format(equation=shown)
                img = self._render_equation(shown)
                return question, expected, img
            return None

        self._probe_mode = False
        for _ in range(40):
            mode = rng.choice(cfg["modes"])
            mx = cfg["mx"]
            # Pick a target TRUE equation
            if mode == "op_only":
                # Build: a OP b = c, with OP that we'll then turn into the WRONG operator
                # by ADDING a stick. The model removes a stick to flip back.
                # We want '+' shown but '-' is correct, OR vice-versa.
                # Force display equation to use '+' but truth to use '-'.
                # Pick a, b, c such that a - b = c and a, b, c are 1-9.
                a = rng.randint(1, mx)
                b = rng.randint(1, a)  # ensure non-negative
                c = a - b
                if c < 0:
                    continue
                truth = f"{a}-{b}={c}"
                shown = f"{a}+{b}={c}"  # wrong: shown
                if _eval_eq(shown):
                    # If a + b = c == a - b, only when b=0 — skip
                    continue
            else:
                # digit_swap: pick TRUE equation, then SWAP one digit by adding 1 stick
                # First pick a true equation a OP b = c
                op = rng.choice(["+", "-"])
                a = rng.randint(1, mx)
                b = rng.randint(1, mx)
                if op == "-" and b > a:
                    a, b = b, a
                c = a + b if op == "+" else a - b
                if c < 0 or c > mx * 2:
                    continue
                truth = f"{a}{op}{b}={c}"
                # Pick one digit position (a / b / c) and one digit value to mutate
                # by adding 1 stick (so the SHOWN equation has more sticks).
                truth_tokens = _parse_equation_to_tokens(truth)
                if truth_tokens is None:
                    continue
                # find positions where digit can be transformed by ADDING 1 stick
                positions: List[Tuple[int, int, str, str]] = []
                # (token_idx, char_idx, old_digit, new_digit)
                for ti, tok in enumerate(truth_tokens):
                    if tok.isdigit():
                        for ci, d in enumerate(tok):
                            for nd in _digits_reachable(d, +1):
                                if nd != d:
                                    positions.append((ti, ci, d, nd))
                if not positions:
                    continue
                rng.shuffle(positions)
                # Try a position; build shown equation; ensure shown != truth and shown is wrong
                ok = False
                for ti, ci, old, new in positions:
                    new_tok = list(truth_tokens[ti])
                    new_tok[ci] = new
                    new_tok = "".join(new_tok)
                    shown_tokens = truth_tokens[:]
                    shown_tokens[ti] = new_tok
                    shown = "".join(shown_tokens)
                    if _eval_eq(shown) is True:
                        # mutation accidentally still true; skip
                        continue
                    truth = "".join(truth_tokens)
                    ok = True
                    break
                if not ok:
                    continue

            # Verify shown is wrong, truth is right
            if _eval_eq(shown) is True:
                continue
            if _eval_eq(truth) is not True:
                continue

            # Enumerate ALL valid one-stick removals from shown.
            valid_solutions = self._enumerate_solutions(shown)
            if truth not in valid_solutions:
                # something went wrong; skip
                continue
            if not valid_solutions:
                continue

            # Save for _check_answer
            self._matchstick_shown = shown
            self._matchstick_solutions = set(valid_solutions)

            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(equation=shown)
            ans_str = truth  # representative
            img = self._render_equation(shown)
            return question, ans_str, img
        return None

    @staticmethod
    def _format_state(equation: str) -> str:
        """Render the equation as text (mirrors the Current Puzzle State)."""
        return f"Equation: {equation}"

    def _enumerate_solutions(self, shown: str) -> List[str]:
        """Enumerate all equations reachable from `shown` by removing exactly
        one matchstick that are mathematically true."""
        tokens = _parse_equation_to_tokens(shown)
        if tokens is None:
            return []
        sols = set()
        # Try removing one stick from each digit (digit → digit reachable by -1)
        for ti, tok in enumerate(tokens):
            if tok.isdigit():
                for ci, d in enumerate(tok):
                    for nd in _digits_reachable(d, -1):
                        if nd != d:
                            new_tok = list(tok)
                            new_tok[ci] = nd
                            new_tok_s = "".join(new_tok)
                            new_tokens = tokens[:]
                            new_tokens[ti] = new_tok_s
                            cand = "".join(new_tokens)
                            if _eval_eq(cand) is True:
                                sols.add(cand)
            elif tok == "+":
                # Remove one stick from + → -
                new_tokens = tokens[:]
                new_tokens[ti] = "-"
                cand = "".join(new_tokens)
                if _eval_eq(cand) is True:
                    sols.add(cand)
        return list(sols)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # PROBE MODE: yes/no answer
        if getattr(self, "_probe_mode", False):
            text = predicted.strip().lower()
            # Look for yes/no/y/n
            m = re.search(r"\b(yes|no|y|n)\b", text)
            if not m:
                # Strict equality
                if text in ("yes", "no"):
                    return text.upper() == self._probe_answer
                return False
            w = m.group(1)
            pred = "YES" if w.startswith("y") else "NO"
            return pred == self._probe_answer

        if not hasattr(self, "_matchstick_solutions"):
            return super()._check_answer(predicted, ground_truth)
        # Normalize predicted: strip spaces, remove latex \times etc.
        s = predicted.strip()
        s = s.replace(" ", "")
        s = s.replace("$", "").replace("`", "").replace("\\,", "")
        # Remove LaTeX
        s = s.replace("\\times", "*").replace("\\cdot", "*")
        # Allow first-line variants
        # If multiple lines, take the first that contains '='
        for line in s.replace("\\n", "\n").splitlines():
            line = line.strip()
            if "=" in line and any(c.isdigit() for c in line):
                s = line
                break
        # If predicted contains a stripping like "5-3=2." remove trailing dot
        s = s.rstrip(".")
        # Direct match against a known solution
        if s in self._matchstick_solutions:
            return True
        # Some equivalence: '2=5-3' vs '5-3=2'
        if "=" in s:
            l, r = s.split("=", 1)
            swapped = f"{r}={l}"
            if swapped in self._matchstick_solutions:
                return True
        # Fallback: parse as equation and check (a) it evaluates true,
        # (b) it is reachable from `_matchstick_shown` by removing exactly 1 stick.
        if _eval_eq(s) is True:
            shown_tokens = _parse_equation_to_tokens(self._matchstick_shown)
            cand_tokens = _parse_equation_to_tokens(s)
            if shown_tokens is None or cand_tokens is None:
                return False
            d = _diff_sticks_token_seq(shown_tokens, cand_tokens)
            if d == 1:
                return True
            # Try swapped sides
            if "=" in s:
                l, r = s.split("=", 1)
                swapped_tokens = _parse_equation_to_tokens(f"{r}={l}")
                if swapped_tokens is not None:
                    d2 = _diff_sticks_token_seq(shown_tokens, swapped_tokens)
                    if d2 == 1:
                        return True
        return False

    # ------------------------------------------------------------ render
    def _render_equation(self, eq: str) -> Image.Image:
        rng = self._rng
        bg = "#fdfdfd"
        # Layout: each character is drawn in a fixed-width slot
        slot_w = 1.4
        slot_h = 2.2
        n = len(eq)
        fig_w = max(6.0, slot_w * n + 1.0)
        fig_h = max(3.0, slot_h + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        match_color = "#a4541b"  # wood
        head_color = "#c0392b"   # match-head red

        for i, ch in enumerate(eq):
            cx = i * slot_w + slot_w / 2
            cy = slot_h / 2
            if ch.isdigit():
                self._draw_digit(ax, ch, cx, cy, match_color, head_color)
            elif ch == "+":
                self._draw_plus(ax, cx, cy, match_color, head_color)
            elif ch == "-":
                self._draw_minus(ax, cx, cy, match_color, head_color)
            elif ch == "=":
                self._draw_equals(ax, cx, cy, match_color, head_color)

        ax.set_xlim(-0.3, n * slot_w + 0.3)
        ax.set_ylim(-0.4, slot_h + 0.4)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg, dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    # 7-segment helpers — segment endpoints in a normalized (cx, cy) frame.
    def _seg_endpoints(self, seg, cx, cy):
        """Return ((x0,y0),(x1,y1)) for segment in {a,b,c,d,e,f,g}.
        Digit box is 0.8 wide and 1.6 tall, centered at (cx, cy)."""
        w = 0.8
        h = 1.6
        x0 = cx - w / 2
        x1 = cx + w / 2
        y_top = cy + h / 2
        y_mid = cy
        y_bot = cy - h / 2
        if seg == "a":
            return (x0, y_top), (x1, y_top)
        if seg == "b":
            return (x1, y_top), (x1, y_mid)
        if seg == "c":
            return (x1, y_mid), (x1, y_bot)
        if seg == "d":
            return (x0, y_bot), (x1, y_bot)
        if seg == "e":
            return (x0, y_mid), (x0, y_bot)
        if seg == "f":
            return (x0, y_top), (x0, y_mid)
        if seg == "g":
            return (x0, y_mid), (x1, y_mid)
        raise ValueError(seg)

    def _draw_match(self, ax, p0, p1, match_color, head_color, lw=4.5):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=match_color,
                linewidth=lw, solid_capstyle="round", zorder=2)
        # Match head at one end (random end for variety)
        # Use p1 by convention
        ax.plot([p1[0]], [p1[1]], marker="o", color=head_color,
                markersize=6.5, zorder=3, markeredgecolor=head_color)

    def _draw_digit(self, ax, ch, cx, cy, match_color, head_color):
        for seg in DIGIT_SEGMENTS[ch]:
            p0, p1 = self._seg_endpoints(seg, cx, cy)
            self._draw_match(ax, p0, p1, match_color, head_color)

    def _draw_plus(self, ax, cx, cy, match_color, head_color):
        # Horizontal stick + Vertical stick, both shorter than digit width
        L = 0.7
        # horizontal
        self._draw_match(ax, (cx - L / 2, cy), (cx + L / 2, cy),
                         match_color, head_color)
        # vertical
        self._draw_match(ax, (cx, cy - L / 2), (cx, cy + L / 2),
                         match_color, head_color)

    def _draw_minus(self, ax, cx, cy, match_color, head_color):
        L = 0.7
        self._draw_match(ax, (cx - L / 2, cy), (cx + L / 2, cy),
                         match_color, head_color)

    def _draw_equals(self, ax, cx, cy, match_color, head_color):
        L = 0.7
        self._draw_match(ax, (cx - L / 2, cy + 0.18), (cx + L / 2, cy + 0.18),
                         match_color, head_color)
        self._draw_match(ax, (cx - L / 2, cy - 0.18), (cx + L / 2, cy - 0.18),
                         match_color, head_color)
