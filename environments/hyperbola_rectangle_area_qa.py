"""
Hyperbola Rectangle Area QA — pure single-letter MCQ A-D / A-E with
"No correct answer".

2026-05-05 R5 Batch 3: HARDENED rewrite. Was 100% saturated because the env
gave away k directly (e.g. k=6 → area = |k| = 6 was a one-step literal read).
Now a pure MCQ with two modes:

  - rect (single point):  given area = |k|, asked for k (with quadrant sign).
  - quad (two points):    given two points P1 in Q1, P2 in Q3 on y=k/x,
                          asked for area of quadrilateral P1OP2X (where X is
                          the foot or fourth vertex).

Per-level design:
  L0/L1 — 4-MCQ. rect mode, Q1/Q3 only (positive k), small areas.
  L2/L3 — 5-MCQ + 10% trap. rect mode, all 4 quads.
  L4/L5 — 5-MCQ + 15% trap. rect; expression-form OUT (we return integer k).
  L6/L7 — 5-MCQ + 25% trap. Adds quad mode (trapezoid area).
  L8/L9 — 5-MCQ + 40% trap. quad mode dominant; tightest distractors
          (off-by-2, half/double, swap with rectangle area).
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


def _quad_signs(quad: int) -> Tuple[int, int]:
    return {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}[quad]


def _signed_k(area: int, quad: int) -> int:
    sx, sy = _quad_signs(quad)
    return area * (sx * sy)


class HyperbolaRectangleAreaQA(StandaloneVisualEnv):
    ENV_NAME = "hyperbola_rectangle_area"
    ALLOW_ROTATION = False

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "modes": ["rect"],
                "areas": [4, 6, 8, 10, 12],
                "quads": [1, 3],
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "modes": ["rect"],
                "areas": [4, 6, 8, 10, 12, 15, 18],
                "quads": [1, 2, 3, 4],
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "modes": ["rect"],
                "areas": [6, 8, 10, 12, 15, 18, 20, 24],
                "quads": [1, 2, 3, 4],
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": True,
            }
        if level <= 7:
            return {
                "modes": ["rect", "quad"],
                "areas": [8, 10, 12, 15, 18, 20, 24, 30],
                "quads": [1, 2, 3, 4],
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        return {
            "modes": ["quad", "rect"],
            "areas": [12, 18, 24, 30, 36, 48],
            "quads": [1, 2, 3, 4],
            "use_5_options": True,
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1289 + level * 41 + 13)
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
        if mode == "rect":
            return self._gen_rect(rng, cfg)
        return self._gen_quad(rng, cfg)

    # ------------------------------------------------------------------ #
    # Mode rect: single point on hyperbola, area = |k|, find signed k.
    # ------------------------------------------------------------------ #
    def _gen_rect(self, rng, cfg):
        area = rng.choice(cfg["areas"])
        quad = rng.choice(cfg["quads"])
        k = _signed_k(area, quad)

        # Pick clean integer point in chosen quadrant
        divisors = [d for d in range(1, area + 1) if area % d == 0]
        ax = rng.choice(divisors)
        ay = area // ax
        sx, sy = _quad_signs(quad)
        a = sx * ax; b = sy * ay
        if a == 0 or b == 0:
            return None

        target = str(k)
        # Distractors: sign flip is the natural quadrant-confusion error
        cand = []
        cand.append(-k)         # forgot quadrant sign
        cand.append(area)       # always-positive answer (forgot sign entirely)
        cand.append(-area)
        cand.append(k + 2); cand.append(k - 2)
        cand.append(k * 2)      # confused with 2*area triangle ↔ rectangle
        cand.append(k // 2 if k % 2 == 0 else k + 1)
        cand.append(area * 2)
        cand.append(area // 2 if area % 2 == 0 else area + 1)
        # Off-by-1
        cand.append(k + 1); cand.append(k - 1)
        seen = {k}; clean = []
        for c in cand:
            if c not in seen:
                seen.add(c); clean.append(c)
        rng.shuffle(clean)
        distractors = [str(x) for x in clean]
        if len(distractors) < 4:
            return None

        stem = (
            f"As shown in the figure, point P lies on the graph of the "
            f"inverse proportion function y = k/x in {_QUAD_WORDS[quad]}. PM "
            f"is perpendicular to the x-axis at M and PN is perpendicular to "
            f"the y-axis at N. The area of rectangle PMON (with O at the "
            f"origin) is {area}. What is the value of k?"
        )
        return self._finalize_mcq(rng, cfg, target, distractors, stem,
                                  self._render_rect(k, a, b, area, quad))

    # ------------------------------------------------------------------ #
    # Mode quad: two points P1 in Q1, P2 in Q3 — both on y=k/x — and origin
    # forms a triangle / quadrilateral. We ask for area of triangle OP1P2,
    # which by the |k|=area lemma applied twice plus reflection is exactly
    # |k| * (something). Simpler: area of rectangle through P1 horizontals and
    # vertical through P2 — equivalently the trapezoid P1MNP2 where M, N are
    # feet on the x-axis. Area = ½ * |x1 - x2| * (|y1| + |y2|).
    # ------------------------------------------------------------------ #
    def _gen_quad(self, rng, cfg):
        for _ in range(15):
            k_pos = rng.choice(cfg["areas"])  # |k|
            divisors = [d for d in range(1, k_pos + 1) if k_pos % d == 0]
            if len(divisors) < 2:
                continue
            x1 = rng.choice(divisors)
            y1 = k_pos // x1
            # Second point in Q3 with same |k|: pick a different divisor
            other = [d for d in divisors if d != x1]
            if not other:
                continue
            x2_abs = rng.choice(other)
            y2_abs = k_pos // x2_abs
            P1 = (x1, y1)
            P2 = (-x2_abs, -y2_abs)
            # Trapezoid area: feet F1=(x1, 0), F2=(-x2_abs, 0).
            # Trapezoid F1 P1 P2 F2: parallel sides P1F1=|y1| and P2F2=|y2|;
            # but they point in OPPOSITE directions, so it's not a normal
            # trapezoid. Use direct polygon-area formula instead via shoelace.
            poly = [F := (x1, 0), P1, P2, (-x2_abs, 0)]
            area_quad = self._shoelace(poly)
            if area_quad <= 0 or abs(area_quad - round(area_quad)) > 1e-9:
                continue
            target_val = int(round(area_quad))

            # Distractors
            cand = []
            cand.append(k_pos)              # area of single rectangle
            cand.append(2 * k_pos)          # forgot to halve
            cand.append(k_pos // 2 if k_pos % 2 == 0 else k_pos + 1)
            cand.append(target_val + 2); cand.append(target_val - 2)
            cand.append(target_val * 2); cand.append(target_val // 2 if target_val % 2 == 0 else target_val + 1)
            cand.append(target_val + 1); cand.append(target_val - 1)
            cand.append(abs(x1 * y2_abs))   # cross-product confusion
            cand.append(abs(x2_abs * y1))
            seen = {target_val}; clean = []
            for c in cand:
                if c not in seen and c > 0:
                    seen.add(c); clean.append(c)
            rng.shuffle(clean)
            distractors = [str(x) for x in clean]
            if len(distractors) < 4:
                continue

            stem = (
                f"As shown in the figure, the graph of the inverse proportion "
                f"function y = k/x passes through P1({P1[0]}, {P1[1]}) in the "
                f"first quadrant and P2({P2[0]}, {P2[1]}) in the third "
                f"quadrant. Drop perpendiculars from P1 and P2 to the x-axis "
                f"meeting at points F1 and F2 respectively. What is the area "
                f"of the quadrilateral F1 P1 P2 F2?"
            )
            return self._finalize_mcq(rng, cfg, str(target_val),
                                      distractors, stem,
                                      self._render_quad(k_pos, P1, P2))
        return None

    @staticmethod
    def _shoelace(pts):
        n = len(pts)
        s = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

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
    def _render_rect(self, k, a, b, area, quad):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(a), abs(b), 6) + 2
        eps = 0.05
        xs1 = np.linspace(eps, ext, 200)
        xs2 = np.linspace(-ext, -eps, 200)
        ax.plot(xs1, k / xs1, color="#1f77b4", linewidth=2)
        ax.plot(xs2, k / xs2, color="#1f77b4", linewidth=2)
        ax.scatter([a], [b], color="#c0392b", s=80, zorder=5)
        # Rectangle
        ax.plot([a, a], [0, b], linestyle="--", color="#7f8c8d", linewidth=1.5)
        ax.plot([0, a], [b, b], linestyle="--", color="#7f8c8d", linewidth=1.5)
        ax.fill([0, a, a, 0], [0, 0, b, b], color="#ffe2a8", alpha=0.4)
        ax.annotate("P", (a, b),
                    textcoords="offset points",
                    xytext=(8 if a > 0 else -16, 8 if b > 0 else -18),
                    fontsize=12, color="#c0392b", fontweight="bold")
        ax.annotate("M", (a, 0),
                    textcoords="offset points",
                    xytext=(6 if a > 0 else -15, -18),
                    fontsize=11, color="#222", fontweight="bold")
        ax.annotate("N", (0, b),
                    textcoords="offset points",
                    xytext=(-15 if a > 0 else 6, 6 if b > 0 else -18),
                    fontsize=11, color="#222", fontweight="bold")
        ax.annotate("O", (0, 0),
                    textcoords="offset points",
                    xytext=(-14, -16),
                    fontsize=11, color="#222", fontweight="bold")
        ax.axhline(0, color="#222", linewidth=0.9)
        ax.axvline(0, color="#222", linewidth=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"y = k/x; rectangle area = {area}")
        return self.fig_to_pil(fig, dpi=110)

    def _render_quad(self, k_pos, P1, P2):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ext = max(abs(P1[0]), abs(P1[1]), abs(P2[0]), abs(P2[1]), 6) + 2
        eps = 0.05
        xs1 = np.linspace(eps, ext, 200)
        xs2 = np.linspace(-ext, -eps, 200)
        ax.plot(xs1, k_pos / xs1, color="#1f77b4", linewidth=2)
        ax.plot(xs2, k_pos / xs2, color="#1f77b4", linewidth=2)
        # Points and polygon F1 P1 P2 F2
        F1 = (P1[0], 0); F2 = (P2[0], 0)
        poly_pts = [F1, P1, P2, F2]
        xs_p = [p[0] for p in poly_pts] + [poly_pts[0][0]]
        ys_p = [p[1] for p in poly_pts] + [poly_pts[0][1]]
        ax.fill(xs_p, ys_p, color="#ffe2a8", alpha=0.4)
        ax.plot(xs_p, ys_p, color="#7f8c8d", linewidth=1.5)
        for pt, lab, col in [(P1, "P1", "#c0392b"), (P2, "P2", "#27ae60"),
                              (F1, "F1", "#222"), (F2, "F2", "#222")]:
            ax.scatter([pt[0]], [pt[1]], color=col, s=70, zorder=5)
            ax.annotate(lab, pt,
                        textcoords="offset points",
                        xytext=(8, 8), fontsize=11, color=col, fontweight="bold")
        ax.axhline(0, color="#222", linewidth=0.9)
        ax.axvline(0, color="#222", linewidth=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(-ext, ext)
        ax.set_ylim(-ext, ext)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("y = k/x; quadrilateral F1 P1 P2 F2")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = HyperbolaRectangleAreaQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
