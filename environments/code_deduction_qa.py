"""
Code Deduction QA (D71, P1).

Reference qid 313:
  "The statements on the right give clues to the identity of a three-digit
   code. What is the code?"  Ans: 275

Generates a 3-digit code (digits 0-9, optional uniqueness constraint), then
synthesizes 3-6 short natural-language clues that together pin it (e.g.,
"The hundreds digit is even", "The sum of all three digits is 14",
"The tens digit is the largest"). Render the clues as a side-text image.

Verifier: integer answer (`\\boxed{275}` or bare `275`). Substring match also
works because the standalone verifier handles single-number numeric matching.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _digits_satisfy(d, clues_check_fns):
    """Check if digit triple (h,t,u) satisfies all clue check functions."""
    return all(fn(d) for fn in clues_check_fns)


class CodeDeductionQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "code_deduction"

    # 2026-05-04 R4: full-gradient redesign per dynamath qid 313 sample.
    # Benchmark hardness: deduce N-digit code from natural-language clues.
    # Trivial clues (specific value, even/odd, sum) saturate easily — model
    # just enumerates. Real harder modes need:
    #   - more digits (4 vs 3)
    #   - modular-arithmetic clues (digit ≡ k mod m)
    #   - negative clues (digit is NOT k)
    #   - composite clues (sum of two adjacent + product of others)
    # Each level adds NEW clue type to the pool, not just more clues.
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_digits": 3, "n_clues": 5, "code_unique": True,
                    "clue_modes": ["basic_specific"]}
        if level == 1:
            return {"n_digits": 3, "n_clues": 4, "code_unique": True,
                    "clue_modes": ["basic"]}
        if level == 2:
            return {"n_digits": 3, "n_clues": 4, "code_unique": False,
                    "clue_modes": ["basic"]}
        if level == 3:
            return {"n_digits": 3, "n_clues": 5, "code_unique": False,
                    "clue_modes": ["basic"]}
        if level == 4:
            # introduce modular clue type
            return {"n_digits": 3, "n_clues": 5, "code_unique": False,
                    "clue_modes": ["basic", "modular"]}
        if level == 5:
            return {"n_digits": 3, "n_clues": 6, "code_unique": False,
                    "clue_modes": ["basic", "modular", "negative"]}
        if level == 6:
            # bump to 4 digits
            return {"n_digits": 4, "n_clues": 6, "code_unique": False,
                    "clue_modes": ["basic", "modular", "negative"]}
        if level == 7:
            return {"n_digits": 4, "n_clues": 7, "code_unique": False,
                    "clue_modes": ["basic", "modular", "negative",
                                    "composite"]}
        if level == 8:
            return {"n_digits": 4, "n_clues": 8, "code_unique": True,
                    "clue_modes": ["basic", "modular", "negative",
                                    "composite"]}
        return {"n_digits": 5, "n_clues": 9, "code_unique": False,
                "clue_modes": ["basic", "modular", "negative", "composite"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 9091 + level * 71 + 11)

        for _ in range(40):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n_digits = cfg.get("n_digits", 3)
        # First digit in 1..9 (no leading zero); others in 0..9
        code = tuple([rng.randint(1, 9)] +
                     [rng.randint(0, 9) for _ in range(n_digits - 1)])
        if cfg["code_unique"]:
            if len(set(code)) != n_digits:
                return None

        clue_pool = self._generate_clue_pool(code, cfg.get("clue_modes",
                                                            ["basic"]))
        rng.shuffle(clue_pool)

        # Build candidate space (skip if too large for safety: 5-digit = 1e5
        # which is fine; 4-digit unique = ~5040 fine).
        candidates = self._all_candidates(n_digits, cfg["code_unique"])
        chosen_clues = []
        chosen_check = []
        for clue_text, check_fn in clue_pool:
            new_cands = [d for d in candidates if check_fn(d)]
            if len(new_cands) < len(candidates):
                chosen_clues.append(clue_text)
                chosen_check.append(check_fn)
                candidates = new_cands
            if len(candidates) == 1 and len(chosen_clues) >= cfg["n_clues"]:
                break
            if len(candidates) == 1:
                # Pad with additional verified-true clues
                for c2_text, c2_fn in clue_pool:
                    if c2_text in chosen_clues:
                        continue
                    if c2_fn(code):
                        chosen_clues.append(c2_text)
                        chosen_check.append(c2_fn)
                    if len(chosen_clues) >= cfg["n_clues"]:
                        break
                break

        if len(candidates) != 1 or candidates[0] != code:
            return None
        if len(chosen_clues) < cfg["n_clues"]:
            return None

        digit_word = {2: "two", 3: "three", 4: "four", 5: "five"}.get(
            n_digits, str(n_digits))
        question = (
            f"The statements shown in the figure give clues to the identity "
            f"of a {digit_word}-digit code. What is the code?"
        )
        answer_str = "".join(str(d) for d in code)
        img = self._render(chosen_clues)
        return question, answer_str, img

    def _all_candidates(self, n_digits, unique):
        # Cap product to avoid huge spaces
        if n_digits == 3:
            cands = [(a, b, c) for a in range(1, 10)
                     for b in range(0, 10) for c in range(0, 10)]
        elif n_digits == 4:
            cands = [(a, b, c, d) for a in range(1, 10)
                     for b in range(0, 10) for c in range(0, 10)
                     for d in range(0, 10)]
        else:  # 5
            cands = [(a, b, c, d, e) for a in range(1, 10)
                     for b in range(0, 10) for c in range(0, 10)
                     for d in range(0, 10) for e in range(0, 10)]
        if unique:
            cands = [t for t in cands if len(set(t)) == n_digits]
        return cands

    # ------------------------------------------------------------------ #
    def _generate_clue_pool(self, code,
                             clue_modes=None) -> List[Tuple[str, callable]]:
        if clue_modes is None:
            clue_modes = ["basic"]
        n = len(code)
        # Position name lookup based on n_digits.
        # For 3-digit: hundreds, tens, units. For 4: thousands, hundreds,
        # tens, units. For 5: ten-thousands, thousands, hundreds, tens,
        # units. We'll just use ordinal names "first", "second" etc., to
        # stay general.
        names_3 = ["hundreds", "tens", "units"]
        names_4 = ["thousands", "hundreds", "tens", "units"]
        names_5 = ["ten-thousands", "thousands", "hundreds", "tens",
                   "units"]
        names = {3: names_3, 4: names_4, 5: names_5}.get(n,
                    [f"position-{i+1}" for i in range(n)])

        pool = []

        # ---- BASIC family ----
        # Sum
        s = sum(code)
        pool.append((f"The sum of the {n} digits is {s}.",
                     lambda d, S=s: sum(d) == S))
        # Each digit even/odd
        for i in range(n):
            di = code[i]
            ni = names[i]
            if di % 2 == 0:
                pool.append((f"The {ni} digit is even.",
                             lambda d, ix=i: d[ix] % 2 == 0))
            else:
                pool.append((f"The {ni} digit is odd.",
                             lambda d, ix=i: d[ix] % 2 == 1))
        # Pairwise comparisons
        for i in range(n - 1):
            ai, bi = code[i], code[i + 1]
            na, nb = names[i], names[i + 1]
            if ai > bi:
                pool.append((f"The {na} digit is greater than the {nb} digit.",
                             lambda d, x=i, y=i + 1: d[x] > d[y]))
            elif ai < bi:
                pool.append((f"The {na} digit is less than the {nb} digit.",
                             lambda d, x=i, y=i + 1: d[x] < d[y]))
        # Pairwise differences
        i, j = 0, n - 1
        if code[i] != code[j]:
            diff = abs(code[i] - code[j])
            pool.append((f"The absolute difference between the {names[i]} "
                         f"digit and the {names[j]} digit is {diff}.",
                         lambda d, x=i, y=j, D=diff: abs(d[x] - d[y]) == D))
        # Largest digit
        max_d = max(code)
        max_idxs = [i for i, x in enumerate(code) if x == max_d]
        if len(max_idxs) == 1:
            mi = max_idxs[0]
            pool.append((f"The {names[mi]} digit is the largest digit "
                         f"(strictly larger than all others).",
                         lambda d, ix=mi: all(d[ix] > d[k] for k in range(n)
                                                if k != ix)))
        # Specific value (very revealing — only "basic_specific" mode)
        if "basic_specific" in clue_modes:
            for i in range(n):
                pool.append((f"The {names[i]} digit is {code[i]}.",
                             lambda d, ix=i, V=code[i]: d[ix] == V))

        # ---- MODULAR family ----
        if "modular" in clue_modes:
            for i in range(n):
                di = code[i]
                ni = names[i]
                # Mod 3
                m3 = di % 3
                pool.append((
                    f"The {ni} digit is congruent to {m3} modulo 3 "
                    f"(i.e. d % 3 = {m3}).",
                    lambda d, ix=i, R=m3: d[ix] % 3 == R))
            # Sum mod 5
            sm = sum(code) % 5
            pool.append((
                f"The sum of all {n} digits is congruent to {sm} modulo 5.",
                lambda d, S=sm: sum(d) % 5 == S))
            # Whether digit is divisible by 4
            for i in range(n):
                if code[i] != 0 and code[i] % 4 == 0:
                    pool.append((
                        f"The {names[i]} digit is divisible by 4.",
                        lambda d, ix=i: d[ix] != 0 and d[ix] % 4 == 0))

        # ---- NEGATIVE family ----
        if "negative" in clue_modes:
            for i in range(n):
                # Pick a few NOT-values that the digit isn't
                not_vals = [v for v in range(10) if v != code[i]]
                # take 2 randomly to ensure clues are useful
                for nv in not_vals[:2]:
                    pool.append((
                        f"The {names[i]} digit is NOT {nv}.",
                        lambda d, ix=i, V=nv: d[ix] != V))
            # No two adjacent digits equal (if true)
            adj_eq = any(code[k] == code[k + 1] for k in range(n - 1))
            if not adj_eq:
                pool.append((
                    "No two adjacent digits are equal.",
                    lambda d: all(d[k] != d[k + 1] for k in range(n - 1))))

        # ---- COMPOSITE family ----
        if "composite" in clue_modes and n >= 3:
            # Sum of first two equals last
            if code[0] + code[1] == code[n - 1]:
                pool.append((
                    f"The sum of the first two digits equals the {names[-1]} "
                    "digit.",
                    lambda d: d[0] + d[1] == d[-1]))
            # Sum of first half == sum of second half
            half = n // 2
            if sum(code[:half]) == sum(code[-half:]):
                pool.append((
                    f"The sum of the first {half} digits equals the sum of "
                    f"the last {half} digits.",
                    lambda d, h=half: sum(d[:h]) == sum(d[-h:])))
            # Product of two specific digits
            i = 0
            j = n - 1
            prod = code[i] * code[j]
            pool.append((
                f"The product of the {names[i]} digit and the {names[j]} "
                f"digit is {prod}.",
                lambda d, x=i, y=j, P=prod: d[x] * d[y] == P))
            # Concatenated 2-digit number is multiple of 7
            if n >= 2:
                two_digit = code[0] * 10 + code[1]
                if two_digit > 0 and two_digit % 7 == 0:
                    pool.append((
                        "The number formed by the first two digits "
                        "(read in order) is a multiple of 7.",
                        lambda d: (d[0] * 10 + d[1]) % 7 == 0
                                   and (d[0] * 10 + d[1]) > 0))
        return pool

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match for the 3-digit code. The base class allows
        a 5% tolerance which falsely accepts 793 for 786, so override it.
        """
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, clues) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
        fig.patch.set_facecolor("#fffaf0")
        ax.set_facecolor("#fffaf0")

        # Draw a code-card placeholder
        ax.text(0.18, 0.85, "_  _  _", ha="center", va="center",
                fontsize=44, color="#2c3e50", family="monospace",
                fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.18, 0.65, "(H  T  U)", ha="center", va="center",
                fontsize=14, color="#7f8c8d",
                transform=ax.transAxes)

        # Draw clues on the right
        ax.text(0.55, 0.95, "Clues:", ha="left", va="top",
                fontsize=15, color="#2c3e50", fontweight="bold",
                transform=ax.transAxes)
        for i, clue in enumerate(clues):
            ax.text(0.55, 0.85 - 0.13 * i,
                    f"{i+1}. {clue}", ha="left", va="top",
                    fontsize=11, color="#34495e",
                    transform=ax.transAxes,
                    wrap=True)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = CodeDeductionQA()
    pass_count = 0
    total = 0
    wrappers_for = lambda a: [
        f"<answer>{a}</answer>",        # idx 0
        f"\\boxed{{{a}}}",              # idx 1
        f"Final answer: {a}",           # idx 2
    ]
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrapped = wrappers_for(env._answer)[s % 3]
                v_correct = env.verify(wrapped)
                v_wrong = env.verify(wrappers_for("000")[s % 3])
                print(f"   correct={v_correct['accuracy']} wrong={v_wrong['accuracy']}")
                if v_correct['accuracy'] == 1 and v_wrong['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
