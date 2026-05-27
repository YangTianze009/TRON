"""
Line × Hyperbola Intersection QA — pure single-letter MCQ A-D / A-E with
"No correct answer".

2026-05-05 R5 Batch 3: HARDENED rewrite. Was 95% saturated because the
verifier accepted both `y = m x + b` and `(x_A, y_A), (x_B, y_B)` formats with
loose tolerance and the model could read intersection coordinates straight
from the figure. Now a pure MCQ asking either:

  - sum_x   : sum of x-coordinates of the two intersections (Vieta tells us
              this equals -b/m for the quadratic m x^2 + b x - k = 0)
  - one_pt  : coordinates of one specific intersection point as `(x, y)`

with tight numerical/sign distractors so the model must actually compute, not
read.

Per-level design:
  L0/L1 — 4-MCQ. sum_x mode only, small positive integers.
  L2/L3 — 5-MCQ + 10% trap. sum_x; small ints with neg k allowed.
  L4/L5 — 5-MCQ + 15% trap. Adds one_pt mode (returns ordered pair).
  L6/L7 — 5-MCQ + 25% trap. Both modes; bigger ranges.
  L8/L9 — 5-MCQ + 40% trap. Both modes; tightest distractors (sign flips,
          coord swaps, off-by-one).
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


def _format_linear(m, b):
    """Format y = m*x + b — clean signs."""
    parts = []
    if m == 1:
        parts.append("x")
    elif m == -1:
        parts.append("-x")
    else:
        parts.append(f"{m}x")
    if b == 0:
        return parts[0]
    if b > 0:
        parts.append(f"+ {b}")
    else:
        parts.append(f"- {abs(b)}")
    return " ".join(parts)


def _pair_str(x, y):
    return f"({x}, {y})"


class LineHyperbolaIntersectionQA(StandaloneVisualEnv):
    ENV_NAME = "line_hyperbola_intersection"
    ALLOW_ROTATION = False

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "modes": ["sum_x"],
                "k_range": (2, 6),
                "allow_neg_k": False,
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "modes": ["sum_x"],
                "k_range": (3, 8),
                "allow_neg_k": True,
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "modes": ["sum_x", "one_pt"],
                "k_range": (3, 10),
                "allow_neg_k": True,
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": True,
            }
        if level <= 7:
            return {
                "modes": ["sum_x", "one_pt"],
                "k_range": (4, 14),
                "allow_neg_k": True,
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        return {
            "modes": ["one_pt", "sum_x"],
            "k_range": (5, 18),
            "allow_neg_k": True,
            "use_5_options": True,
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 6151 + level * 89 + 31)

        for _ in range(40):
            try:
                res = self._try_generate(rng, cfg, level)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        lo, hi = cfg["k_range"]
        for _ in range(20):
            k = rng.randint(lo, hi)
            if cfg["allow_neg_k"] and rng.random() < 0.4:
                k = -k
            divisors = [d for d in range(1, abs(k) + 1) if abs(k) % d == 0]
            divisors = divisors + [-d for d in divisors]
            if len(divisors) < 2:
                continue
            x1, x2 = rng.sample(divisors, 2)
            if x1 == x2:
                continue
            y1 = k / x1
            y2 = k / x2
            if abs(y1) > 18 or abs(y2) > 18:
                continue
            if abs(y1 - round(y1)) > 1e-9 or abs(y2 - round(y2)) > 1e-9:
                continue
            y1 = int(round(y1)); y2 = int(round(y2))
            m_num = y2 - y1
            m_den = x2 - x1
            if m_den == 0:
                continue
            if m_num % m_den != 0:
                continue
            m = m_num // m_den
            if m == 0:
                continue
            b = y1 - m * x1
            if abs(b) > 18 or abs(m) > 8:
                continue
            if x1 < x2:
                xa, ya, xb, yb = x1, y1, x2, y2
            else:
                xa, ya, xb, yb = x2, y2, x1, y1

            mode = rng.choice(cfg["modes"])
            if mode == "sum_x":
                target = xa + xb
                distractors = self._make_sum_distractors(
                    rng, target, xa, xb, ya, yb, m, b, cfg["tight_distractors"])
                if len(distractors) < 4:
                    continue
                stem = (
                    f"As shown in the figure, the line y = {_format_linear(m, b)} "
                    f"intersects the graph of the inverse proportion function "
                    f"y = {k}/x at two points A and B. What is the value of "
                    f"x_A + x_B (the sum of the x-coordinates of the two "
                    f"intersection points)?"
                )
                return self._finalize_mcq(rng, cfg, str(target),
                                          [str(d) for d in distractors],
                                          stem,
                                          self._render(k, m, b, xa, ya, xb, yb))
            else:  # one_pt — ask for coords of A (smaller x)
                # Choose which intersection to ask
                ask_a = rng.choice([True, False])
                if ask_a:
                    target_pair = (xa, ya); other_pair = (xb, yb); name = "A"
                else:
                    target_pair = (xb, yb); other_pair = (xa, ya); name = "B"
                target_str = _pair_str(*target_pair)
                distractors_pairs = self._make_pair_distractors(
                    rng, target_pair, other_pair, cfg["tight_distractors"])
                distractor_strs = [_pair_str(*p) for p in distractors_pairs
                                   if _pair_str(*p) != target_str]
                if len(distractor_strs) < 4:
                    continue
                stem = (
                    f"As shown in the figure, the line y = {_format_linear(m, b)} "
                    f"intersects the graph of y = {k}/x at points A and B "
                    f"(with x_A < x_B). What are the coordinates of point "
                    f"{name}?"
                )
                return self._finalize_mcq(rng, cfg, target_str,
                                          distractor_strs,
                                          stem,
                                          self._render(k, m, b, xa, ya, xb, yb))
        return None

    # ------------------------------------------------------------------ #
    def _make_sum_distractors(self, rng, target, xa, xb, ya, yb, m, b, tight):
        cand = []
        # Sign flip
        if -target != target:
            cand.append(-target)
        # Off-by-one / two
        cand.append(target + 1); cand.append(target - 1)
        cand.append(target + 2); cand.append(target - 2)
        # Confused with sum of y's
        cand.append(ya + yb)
        # Confused with product (Vieta's other formula)
        cand.append(xa * xb)
        # Just one of the x's (forgot to add)
        cand.append(xa); cand.append(xb)
        # Confused with -b
        cand.append(-b); cand.append(b)
        seen = {target}
        out = []
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(c)
        rng.shuffle(out)
        return out

    def _make_pair_distractors(self, rng, target, other, tight):
        tx, ty = target
        ox, oy = other
        cand = []
        # Swap x↔y of target
        cand.append((ty, tx))
        # Sign flip on x or y
        cand.append((-tx, ty))
        cand.append((tx, -ty))
        cand.append((-tx, -ty))
        # The OTHER intersection (most natural confusion)
        cand.append(other)
        # Off-by-one on x
        cand.append((tx + 1, ty))
        cand.append((tx - 1, ty))
        # Off-by-one on y
        cand.append((tx, ty + 1))
        cand.append((tx, ty - 1))
        seen = {target}
        out = []
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(c)
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
            n_distractor = n_value - 1
            chosen = distractor_strs[:n_distractor]
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
    def _render(self, k, m, b, xa, ya, xb, yb):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(xa), abs(xb), abs(ya), abs(yb), 6) + 3
        ext = min(15, ext)
        eps = 0.05
        xs1 = np.linspace(eps, ext, 200)
        xs2 = np.linspace(-ext, -eps, 200)
        ax.plot(xs1, k / xs1, color="#1f77b4", linewidth=2,
                label=f"y = {k}/x")
        ax.plot(xs2, k / xs2, color="#1f77b4", linewidth=2)
        xs = np.linspace(-ext, ext, 200)
        ax.plot(xs, m * xs + b, color="#d62728", linewidth=2,
                label=f"y = {_format_linear(m, b)}")
        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")
        for x, y, lab in [(xa, ya, "A"), (xb, yb, "B")]:
            ax.scatter([x], [y], color="#2ca02c", s=80, zorder=5,
                       edgecolor="black", linewidth=1.0)
            ax.annotate(lab, (x, y),
                        textcoords="offset points",
                        xytext=(8, 8), fontsize=12, color="#2ca02c",
                        fontweight="bold")
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=9)
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = LineHyperbolaIntersectionQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
