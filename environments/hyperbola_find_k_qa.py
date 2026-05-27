"""
Hyperbola Find K QA — pure single-letter MCQ A-D / A-E with "No correct answer".

2026-05-05 R5 Batch 3: HARDENED rewrite. Was 100% saturated because the
verifier accepted any of `y=k/x`, `\\frac{k}{x}`, bare integer — and at L0 the
question was a single-step k = a*b read. Now a pure MCQ with tight numerical
distractors so the model can no longer "look at the picture and read the
formula" — it has to actually compute and pick the right letter.

Per-level design:
  L0/L1 — 4-MCQ. Single point on hyperbola, integer coords (k = x*y).
  L2/L3 — 5-MCQ + 10% trap. Adds quadrant-sign reasoning (point in Q1..Q4).
  L4/L5 — 5-MCQ + 15% trap. Two-points mode (find k from labelled A; check B).
  L6/L7 — 5-MCQ + 25% trap. Adds "two-branch" mode: hyperbola y=k/x with TWO
          marked points, only one actually on the curve, identify which.
  L8/L9 — 5-MCQ + 40% trap. Two hyperbolas y=k1/x and y=k2/x both drawn;
          given one point on each branch, compute |k1 - k2| or k1 + k2.

Question phrasing matches inverse-proportional textbook style:
  "As shown in the figure, the graph of the inverse proportion function
   y = k/x passes through point A(a, b). What is the value of k?"
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


_QUAD_WORDS = {
    1: "the first quadrant",
    2: "the second quadrant",
    3: "the third quadrant",
    4: "the fourth quadrant",
}


class HyperbolaFindKQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a labeled hyperbola figure, pick k."""

    ENV_NAME = "hyperbola_find_k"
    ALLOW_ROTATION = False  # axis-orientation matters

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "modes": ["single_point"],
                "ab_range": (2, 6),
                "allow_neg": False,
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "modes": ["single_point"],
                "ab_range": (2, 8),
                "allow_neg": True,
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "modes": ["single_point", "two_points"],
                "ab_range": (3, 10),
                "allow_neg": True,
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": True,
            }
        if level <= 7:
            return {
                "modes": ["single_point", "two_points", "which_on_curve"],
                "ab_range": (3, 12),
                "allow_neg": True,
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (env still saturated 95%).
        return {
            "modes": ["which_on_curve", "two_branches", "two_points"],
            "ab_range": (4, 14),
            "allow_neg": True,
            "use_5_options": True,
            "trap_rate": 0.50,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7919 + level * 101 + 23)

        for _ in range(20):
            try:
                res = self._try_generate(rng, cfg, level)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        mode = rng.choice(cfg["modes"])
        if mode == "single_point":
            return self._gen_single(rng, cfg)
        if mode == "two_points":
            return self._gen_two_points(rng, cfg)
        if mode == "which_on_curve":
            return self._gen_which(rng, cfg)
        if mode == "two_branches":
            return self._gen_two_branches(rng, cfg)
        return None

    # ------------------------------------------------------------------ #
    # Mode 1: single_point — A(a,b) on y=k/x, find k
    # ------------------------------------------------------------------ #
    def _gen_single(self, rng, cfg):
        lo, hi = cfg["ab_range"]
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if cfg["allow_neg"]:
            a *= rng.choice([-1, 1])
            b *= rng.choice([-1, 1])
        if a == 0 or b == 0:
            return None
        k = a * b

        distractors = self._make_k_distractors(rng, k, a, b, cfg["tight_distractors"])
        if len(distractors) < 4:
            return None

        question_stem = self._stem_single(a, b, rng)
        return self._finalize_mcq(rng, cfg, str(k),
                                  [str(d) for d in distractors],
                                  question_stem,
                                  self._render_single(k, a, b))

    def _stem_single(self, a, b, rng):
        templates = [
            f"As shown in the figure, the graph of the inverse proportion function "
            f"y = k/x passes through the labelled point A({a}, {b}). What is the "
            f"value of k?",
            f"In the diagram, the hyperbola y = k/x passes through point "
            f"A({a}, {b}). Find the value of k.",
            f"The graph of the inverse proportion function y = k/x is shown, with "
            f"point A({a}, {b}) on the curve. Determine k.",
        ]
        return rng.choice(templates)

    # ------------------------------------------------------------------ #
    # Mode 2: two_points — A(a,b) given, B(?,d) given (only y), find k
    # ------------------------------------------------------------------ #
    def _gen_two_points(self, rng, cfg):
        lo, hi = cfg["ab_range"]
        for _ in range(10):
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            if cfg["allow_neg"]:
                a *= rng.choice([-1, 1])
                b *= rng.choice([-1, 1])
            if a == 0 or b == 0:
                continue
            k = a * b
            possible_d = [d for d in range(-hi, hi + 1)
                          if d != 0 and d != b and k % d == 0]
            if not possible_d:
                continue
            d = rng.choice(possible_d)
            x_B = k // d
            if x_B == a:
                continue

            distractors = self._make_k_distractors(rng, k, a, b, cfg["tight_distractors"])
            if len(distractors) < 4:
                continue

            stem = (
                f"As shown in the figure, the graph of the inverse proportion "
                f"function y = k/x passes through points A({a}, {b}) and "
                f"B(x_B, {d}) (the x-coordinate of B is not labelled). What is "
                f"the value of k?"
            )
            return self._finalize_mcq(rng, cfg, str(k),
                                      [str(x) for x in distractors],
                                      stem,
                                      self._render_two_points(k, a, b, x_B, d))
        return None

    # ------------------------------------------------------------------ #
    # Mode 3: which_on_curve — A on y=k/x, B not. Find k.
    # ------------------------------------------------------------------ #
    def _gen_which(self, rng, cfg):
        lo, hi = cfg["ab_range"]
        for _ in range(10):
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            if cfg["allow_neg"]:
                a *= rng.choice([-1, 1])
                b *= rng.choice([-1, 1])
            if a == 0 or b == 0:
                continue
            k = a * b
            # Decoy point B on x-axis (or off-curve elsewhere)
            bx = rng.randint(lo, hi) * rng.choice([-1, 1])
            by = 0  # decoy is on the x-axis (a common visual confusable)
            if bx == a:
                continue

            distractors = self._make_k_distractors(rng, k, a, b, cfg["tight_distractors"])
            if len(distractors) < 4:
                continue

            stem = (
                f"As shown in the figure, two points are marked on the diagram: "
                f"A({a}, {b}) and B({bx}, {by}). Only one of these lies on the "
                f"graph of the inverse proportion function y = k/x. Identify the "
                f"point that is on the curve and compute the value of k."
            )
            return self._finalize_mcq(rng, cfg, str(k),
                                      [str(x) for x in distractors],
                                      stem,
                                      self._render_which(k, a, b, bx, by))
        return None

    # ------------------------------------------------------------------ #
    # Mode 4: two_branches — y=k1/x AND y=k2/x both drawn; given points
    # on each, compute (k1 + k2) or |k1 - k2|.
    # ------------------------------------------------------------------ #
    def _gen_two_branches(self, rng, cfg):
        lo, hi = cfg["ab_range"]
        for _ in range(10):
            a1 = rng.randint(lo, hi)
            b1 = rng.randint(lo, hi)
            a2 = rng.randint(lo, hi)
            b2 = rng.randint(lo, hi)
            if cfg["allow_neg"]:
                a1 *= rng.choice([-1, 1]); b1 *= rng.choice([-1, 1])
                a2 *= rng.choice([-1, 1]); b2 *= rng.choice([-1, 1])
            if 0 in (a1, b1, a2, b2):
                continue
            k1 = a1 * b1
            k2 = a2 * b2
            if k1 == k2:
                continue
            op = rng.choice(["sum", "absdiff"])
            if op == "sum":
                target = k1 + k2
                op_word = "k1 + k2"
            else:
                target = abs(k1 - k2)
                op_word = "|k1 - k2|"

            # Distractors mix sign / sub-step errors
            distractors = self._make_branches_distractors(rng, k1, k2, op,
                                                          cfg["tight_distractors"])
            if len(distractors) < 4:
                continue

            stem = (
                f"As shown in the figure, two inverse proportion functions are "
                f"drawn in the same plane: y = k1/x passes through "
                f"P1({a1}, {b1}), and y = k2/x passes through P2({a2}, {b2}). "
                f"What is the value of {op_word}?"
            )
            return self._finalize_mcq(rng, cfg, str(target),
                                      [str(x) for x in distractors],
                                      stem,
                                      self._render_two_branches(k1, k2,
                                                                a1, b1,
                                                                a2, b2))
        return None

    # ------------------------------------------------------------------ #
    # Distractor builders
    # ------------------------------------------------------------------ #
    def _make_k_distractors(self, rng, k, a, b, tight):
        """Off-by-sign, off-by-quadrant, ±2, /2 etc."""
        cand = []
        # Sign flip (most common quadrant-confusion error)
        if -k != k:
            cand.append(-k)
        # Off-by-2
        cand.append(k + 2)
        cand.append(k - 2)
        # Off-by-1
        cand.append(k + 1)
        cand.append(k - 1)
        # ±a, ±b, a+b (read-off-of-image errors)
        cand.append(a + b)
        cand.append(a - b)
        cand.append(b - a)
        if abs(k) >= 4 and k % 2 == 0:
            cand.append(k // 2)
            cand.append(2 * k)
        # Drop duplicates and filter out the true answer
        seen = {k}
        out = []
        for c in cand:
            if c not in seen:
                seen.add(c)
                out.append(c)
        rng.shuffle(out)
        return out

    def _make_branches_distractors(self, rng, k1, k2, op, tight):
        """Distractors for two_branches: include the OTHER op as a tight error."""
        s = k1 + k2
        d = abs(k1 - k2)
        cand = []
        # Always include the rival operation result
        if op == "sum":
            cand.append(d)
        else:
            cand.append(s)
        # Single-k distractors
        cand.append(k1)
        cand.append(k2)
        cand.append(-k1)
        cand.append(-k2)
        # Off-by-2 from target
        target = s if op == "sum" else d
        cand.append(target + 2)
        cand.append(target - 2)
        # Misuse: forgot abs
        cand.append(k1 - k2)
        seen = {target}
        out = []
        for c in cand:
            if c not in seen:
                seen.add(c)
                out.append(c)
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------------ #
    # Final MCQ assembly
    # ------------------------------------------------------------------ #
    def _finalize_mcq(self, rng, cfg, true_str, distractor_strs,
                      stem, image):
        # Filter: true_str must not appear in distractors
        distractor_strs = [d for d in distractor_strs if d != true_str]
        # Dedup
        seen = {true_str}
        clean = []
        for d in distractor_strs:
            if d not in seen:
                seen.add(d)
                clean.append(d)
        distractor_strs = clean

        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4
        n_value -= 1 if use_5 else 0  # leave room for E
        # n_value = number of value options (4 for 5-MCQ since E is separate)
        opt_letters = ["A", "B", "C", "D", "E"][:5 if use_5 else 4]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        if len(distractor_strs) < n_value:
            return None

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            chosen = distractor_strs[:n_value]
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]  # 'E'
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
    # Rendering
    # ------------------------------------------------------------------ #
    def _draw_axes_and_hyperbola(self, ax, k, x_extent):
        eps = 0.05
        xs1 = np.linspace(eps, x_extent, 200)
        xs2 = np.linspace(-x_extent, -eps, 200)
        ax.plot(xs1, k / xs1, color="#1f77b4", linewidth=2)
        ax.plot(xs2, k / xs2, color="#1f77b4", linewidth=2)
        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")

    def _set_extent(self, ax, x_extent):
        ax.set_xlim(-x_extent, x_extent)
        ax.set_ylim(-x_extent, x_extent)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def _render_single(self, k, a, b):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(a), abs(b), abs(k) // 2 + 2, 6) + 2
        ext = min(15, ext)
        self._draw_axes_and_hyperbola(ax, k, ext)
        ax.scatter([a], [b], color="#2ca02c", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(f"A({a}, {b})", (a, b),
                    textcoords="offset points",
                    xytext=(8, 8), fontsize=12, color="#2ca02c",
                    fontweight="bold")
        self._set_extent(ax, ext)
        ax.set_title("y = k/x")
        return self.fig_to_pil(fig, dpi=110)

    def _render_two_points(self, k, a, b, x_B, d):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(a), abs(b), abs(x_B), abs(d), 6) + 2
        ext = min(15, ext)
        self._draw_axes_and_hyperbola(ax, k, ext)
        for x, y, lab, label_full in [(a, b, "A", f"A({a}, {b})"),
                                       (x_B, d, "B", f"B(?, {d})")]:
            ax.scatter([x], [y], color="#2ca02c", s=80, zorder=5,
                       edgecolor="black", linewidth=1.0)
            ax.annotate(label_full, (x, y),
                        textcoords="offset points",
                        xytext=(8, 8), fontsize=11, color="#2ca02c",
                        fontweight="bold")
        self._set_extent(ax, ext)
        ax.set_title("y = k/x")
        return self.fig_to_pil(fig, dpi=110)

    def _render_which(self, k, a, b, bx, by):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(a), abs(b), abs(bx), abs(by), 6) + 2
        ext = min(15, ext)
        self._draw_axes_and_hyperbola(ax, k, ext)
        ax.scatter([a], [b], color="#2ca02c", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(f"A({a}, {b})", (a, b),
                    textcoords="offset points",
                    xytext=(8, 8), fontsize=11, color="#2ca02c",
                    fontweight="bold")
        ax.scatter([bx], [by], color="#c0392b", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0, marker="s")
        ax.annotate(f"B({bx}, {by})", (bx, by),
                    textcoords="offset points",
                    xytext=(8, -16), fontsize=11, color="#c0392b",
                    fontweight="bold")
        self._set_extent(ax, ext)
        ax.set_title("y = k/x with two candidate points")
        return self.fig_to_pil(fig, dpi=110)

    def _render_two_branches(self, k1, k2, a1, b1, a2, b2):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(a1), abs(b1), abs(a2), abs(b2), 6) + 2
        ext = min(15, ext)
        eps = 0.05
        for k, color, label in [(k1, "#1f77b4", f"y = k1/x"),
                                 (k2, "#d62728", f"y = k2/x")]:
            xs1 = np.linspace(eps, ext, 200)
            xs2 = np.linspace(-ext, -eps, 200)
            ax.plot(xs1, k / xs1, color=color, linewidth=2, label=label)
            ax.plot(xs2, k / xs2, color=color, linewidth=2)
        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.scatter([a1], [b1], color="#1f77b4", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(f"P1({a1}, {b1})", (a1, b1),
                    textcoords="offset points",
                    xytext=(8, 8), fontsize=11, color="#1f77b4",
                    fontweight="bold")
        ax.scatter([a2], [b2], color="#d62728", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(f"P2({a2}, {b2})", (a2, b2),
                    textcoords="offset points",
                    xytext=(8, -16), fontsize=11, color="#d62728",
                    fontweight="bold")
        self._set_extent(ax, ext)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Two hyperbolas in the same plane")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = HyperbolaFindKQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
