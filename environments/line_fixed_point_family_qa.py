"""
Line Fixed-Point Family QA — pure single-letter MCQ A-D / A-E with
"No correct answer".

2026-05-05 R5 Batch 3: HARDENED rewrite. Was 97.5% saturated because the env
accepted free-form `(x, y)` outputs and the canonical form y = k(x - x0) + y0
literally spelled the answer in the question text. Now a pure MCQ with
expanded forms so the fixed point is NOT visible by inspection — model must
factor / solve.

Per-level design:
  L0/L1 — 4-MCQ. Identify fixed point given two specific values of m (find
          intersection of two concrete lines).
  L2/L3 — 5-MCQ + 10% trap. Family in expanded form y = mx + (c - mx0).
  L4/L5 — 5-MCQ + 15% trap. Same family-form; bigger ranges.
  L6/L7 — 5-MCQ + 25% trap. Family form y = m·f(x) + g(x) (with f, g LINEAR
          in x and integer coefficients).
  L8/L9 — 5-MCQ + 40% trap. Hardest disguised forms; tightest distractors
          (swap x/y, sign flip on each coord, off-by-1 on each coord).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _fmt_int_term(coef, var):
    """Format `c·var` cleanly: 1·x → 'x', -1·x → '-x', 0·x → ''."""
    if coef == 0:
        return ""
    if coef == 1:
        return var
    if coef == -1:
        return f"-{var}"
    return f"{coef}{var}"


def _join_terms(*terms):
    """Join non-empty signed terms with cleaned + / - signs."""
    out = ""
    for t in terms:
        if not t:
            continue
        if not out:
            out = t
        else:
            if t.startswith("-"):
                out += " - " + t[1:]
            else:
                out += " + " + t
    return out or "0"


def _pair_str(x, y):
    return f"({x}, {y})"


class LineFixedPointFamilyQA(StandaloneVisualEnv):
    ENV_NAME = "line_fixed_point_family"
    ALLOW_ROTATION = False

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "form": "two_concrete",
                "x0_range": (-3, 3),
                "y0_range": (-3, 3),
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "form": "expanded",
                "x0_range": (-4, 4),
                "y0_range": (-4, 4),
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "form": "expanded",
                "x0_range": (-5, 5),
                "y0_range": (-5, 5),
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": True,
            }
        if level <= 7:
            return {
                "form": "linear_combo",
                "x0_range": (-6, 6),
                "y0_range": (-6, 6),
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (env still saturated 95%).
        return {
            "form": "linear_combo",
            "x0_range": (-8, 8),
            "y0_range": (-8, 8),
            "use_5_options": True,
            "trap_rate": 0.50,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5879 + level * 41 + 11)

        for _ in range(20):
            try:
                res = self._try_generate(rng, cfg, level)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        x0 = rng.randint(*cfg["x0_range"])
        y0 = rng.randint(*cfg["y0_range"])
        # Avoid trivial (0,0) fixed point at low levels
        if x0 == 0 and y0 == 0:
            x0 = rng.choice([-2, -1, 1, 2])

        if cfg["form"] == "two_concrete":
            return self._gen_two_concrete(rng, cfg, x0, y0)
        if cfg["form"] == "expanded":
            return self._gen_expanded(rng, cfg, x0, y0)
        return self._gen_linear_combo(rng, cfg, x0, y0)

    # ------------------------------------------------------------------ #
    # form two_concrete: present TWO specific lines (e.g., m=1 and m=-1) of
    # the family and ask for their intersection (= fixed point).
    # ------------------------------------------------------------------ #
    def _gen_two_concrete(self, rng, cfg, x0, y0):
        # Pick two distinct slopes
        slopes_pool = [-3, -2, -1, 1, 2, 3]
        m1, m2 = rng.sample(slopes_pool, 2)
        # Each line: y = m*(x - x0) + y0  →  y = m*x + (y0 - m*x0)
        b1 = y0 - m1 * x0
        b2 = y0 - m2 * x0
        line1 = f"y = {_fmt_int_term(m1, 'x')} {'+' if b1 >= 0 else '-'} {abs(b1)}".replace("+ -", "- ")
        line2 = f"y = {_fmt_int_term(m2, 'x')} {'+' if b2 >= 0 else '-'} {abs(b2)}".replace("+ -", "- ")
        # Cleaner: build manually
        line1 = "y = " + _join_terms(_fmt_int_term(m1, "x"), str(b1) if b1 != 0 else "")
        line2 = "y = " + _join_terms(_fmt_int_term(m2, "x"), str(b2) if b2 != 0 else "")
        # When b is negative, _join_terms prepends '-' correctly; when b=0 omit.
        # Edge: _join_terms expects signed strings.
        if b1 < 0:
            line1 = "y = " + _join_terms(_fmt_int_term(m1, "x"), str(b1))
        elif b1 > 0:
            line1 = "y = " + _join_terms(_fmt_int_term(m1, "x"), f"+{b1}".lstrip("+"))
            # safer rewrite:
            line1 = f"y = {_fmt_int_term(m1, 'x')} + {b1}"
        if b1 == 0:
            line1 = f"y = {_fmt_int_term(m1, 'x')}"
        # Same for line2
        if b2 < 0:
            line2 = f"y = {_fmt_int_term(m2, 'x')} - {abs(b2)}"
        elif b2 > 0:
            line2 = f"y = {_fmt_int_term(m2, 'x')} + {b2}"
        else:
            line2 = f"y = {_fmt_int_term(m2, 'x')}"
        if b1 < 0:
            line1 = f"y = {_fmt_int_term(m1, 'x')} - {abs(b1)}"

        true_pair_str = _pair_str(x0, y0)
        distractor_strs = self._make_pair_distractors_str(rng, x0, y0,
                                                          cfg["tight_distractors"])
        if len(distractor_strs) < 4:
            return None

        stem = (
            f"Two lines from the same family of lines are shown:\n"
            f"  {line1}\n"
            f"  {line2}\n"
            f"All lines in this family pass through a common fixed point. "
            f"What are the coordinates of that fixed point?"
        )
        return self._finalize_mcq(rng, cfg, true_pair_str, distractor_strs,
                                  stem, self._render(x0, y0,
                                                     [(m1, b1), (m2, b2)]))

    # ------------------------------------------------------------------ #
    # form expanded: family written as y = m·x + (c - m·x0), where the
    # m-coefficient (-x0) and the constant y0 = c need to be PARSED and
    # FACTORED to find (x0, y0) = the fixed point.
    # ------------------------------------------------------------------ #
    def _gen_expanded(self, rng, cfg, x0, y0):
        # Family: y = mx + (y0 - m*x0) = mx + y0 - m*x0
        # Render: y = mx + y0 + (-x0)*m
        # Build the textual form: "y = mx - x0·m + y0"
        # Use named parameter "m" for the slope variable.
        coef_m = -x0  # coefficient of m in the constant term
        # Build "y = mx + (-x0)·m + y0" — write the (-x0) as a clean integer
        # and keep the "·m" suffix.
        kterm = ""
        if coef_m != 0:
            if coef_m == 1:
                kterm = " + m"
            elif coef_m == -1:
                kterm = " - m"
            elif coef_m > 0:
                kterm = f" + {coef_m}m"
            else:
                kterm = f" - {abs(coef_m)}m"
        const = ""
        if y0 != 0:
            const = f" + {y0}" if y0 > 0 else f" - {abs(y0)}"
        family_str = f"y = mx{kterm}{const}".replace("+ -", "- ")

        true_pair_str = _pair_str(x0, y0)
        distractor_strs = self._make_pair_distractors_str(
            rng, x0, y0, cfg["tight_distractors"])
        if len(distractor_strs) < 4:
            return None

        stem = (
            f"Consider the family of lines\n  {family_str}\n"
            f"where m can be any real number. All lines in this family pass "
            f"through a common fixed point regardless of m. What are the "
            f"coordinates of that fixed point?"
        )
        return self._finalize_mcq(rng, cfg, true_pair_str, distractor_strs,
                                  stem, self._render(x0, y0,
                                                     [(-1, y0 - (-1) * x0),
                                                      (1, y0 - 1 * x0),
                                                      (2, y0 - 2 * x0)]))

    # ------------------------------------------------------------------ #
    # form linear_combo: y = m * f(x) + g(x) where f, g are LINEAR.
    # Setting f(x*) = 0 fixes x* = root of f, then y* = g(x*).
    # ------------------------------------------------------------------ #
    def _gen_linear_combo(self, rng, cfg, x0, y0):
        # Choose a coefficient `p` for x in f and an intercept `q` so that
        # f(x) = p*x + q with root x* = -q/p = x0  →  q = -p*x0.
        for _ in range(8):
            p = rng.choice([1, -1, 2, -2, 3, -3])
            q = -p * x0
            # Choose g(x) = r*x + s such that g(x0) = y0  →  s = y0 - r*x0
            r = rng.choice([0, 1, -1, 2, -2])
            s = y0 - r * x0
            if abs(s) > 30:
                continue
            # Build f(x) and g(x) strings
            f_str = self._poly_str(p, q)
            g_str = self._poly_str(r, s)
            family_str = f"y = ({f_str}) m + ({g_str})"

            true_pair_str = _pair_str(x0, y0)
            distractor_strs = self._make_pair_distractors_str(
                rng, x0, y0, cfg["tight_distractors"])
            if len(distractor_strs) < 4:
                continue

            stem = (
                f"Consider the family of lines\n  {family_str}\n"
                f"where m can be any real number. All lines in this family "
                f"pass through a common fixed point regardless of m. What "
                f"are the coordinates of that fixed point?"
            )
            # Render: pick three sample slopes
            sample_lines = []
            for mm in [-1, 1, 2]:
                # y = (p*x + q)*mm + (r*x + s) = (p*mm + r)*x + (q*mm + s)
                slope = p * mm + r
                intercept = q * mm + s
                sample_lines.append((slope, intercept))
            return self._finalize_mcq(rng, cfg, true_pair_str, distractor_strs,
                                      stem, self._render(x0, y0, sample_lines))
        return None

    @staticmethod
    def _poly_str(a, b):
        if a == 0 and b == 0:
            return "0"
        if a == 0:
            return str(b)
        ax = "x" if a == 1 else ("-x" if a == -1 else f"{a}x")
        if b == 0:
            return ax
        if b > 0:
            return f"{ax} + {b}"
        return f"{ax} - {abs(b)}"

    # ------------------------------------------------------------------ #
    def _make_pair_distractors_str(self, rng, x0, y0, tight):
        cand = []
        # Swap x↔y
        cand.append((y0, x0))
        # Sign flip
        cand.append((-x0, y0))
        cand.append((x0, -y0))
        cand.append((-x0, -y0))
        # Off-by-one
        cand.append((x0 + 1, y0))
        cand.append((x0 - 1, y0))
        cand.append((x0, y0 + 1))
        cand.append((x0, y0 - 1))
        # Origin (for those who guess)
        cand.append((0, 0))
        seen = {(x0, y0)}
        out = []
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(_pair_str(*c))
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------------ #
    def _finalize_mcq(self, rng, cfg, true_str, distractor_strs, stem, image):
        distractor_strs = [d for d in distractor_strs if d != true_str]
        seen = {true_str}; clean = []
        for d in distractor_strs:
            if d not in seen:
                seen.add(d); clean.append(d)
        distractor_strs = clean
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:5 if use_5 else 4]
        n_value = len(opt_letters) - (1 if use_5 else 0)
        if len(distractor_strs) < n_value:
            return None
        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False
        if use_trap:
            chosen = distractor_strs[:n_value]
            options = list(chosen); rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            chosen = distractor_strs[:n_value - 1]
            options = [true_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_str)]
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{stem}\n\n{opt_lines}\n\n"
            f"Answer with a single letter ({letter_str})."
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _render(self, x0, y0, lines):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(x0), abs(y0), 8) + 2
        for slope, intercept in lines:
            xs = np.linspace(-ext, ext, 100)
            ax.plot(xs, slope * xs + intercept, alpha=0.7, linewidth=1.4)
        ax.scatter([x0], [y0], color="#d62728", s=120, zorder=5,
                   edgecolor="black", linewidth=1.5)
        # Render with letter P only — coords are the answer; do NOT label them
        ax.annotate("P", (x0, y0),
                    textcoords="offset points",
                    xytext=(8, 8), fontsize=12, color="#d62728",
                    fontweight="bold")
        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("A family of lines through a common point P")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = LineFixedPointFamilyQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
