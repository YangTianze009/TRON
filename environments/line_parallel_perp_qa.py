"""
Line Parallel/Perpendicular QA (M15 / LF-T5, P0).

A reference line y = m₀x + b₀ is drawn together with a labelled point P.
The model is asked for the equation of the line through P parallel to (or
perpendicular to) the reference line.

Two question modes:
  - parallel: same slope, b' = y_P - m₀ * x_P
  - perpendicular: slope m' = -1/m₀, b' = y_P - m' * x_P

To keep the answer integer-valued, we restrict to integer slopes m₀ in
{-2, -1, 1, 2} for parallel mode, and to perpendicular pairs that yield
integer slopes (e.g. m₀=1 → m'=-1; m₀=-1 → m'=1; pairs with m₀=2 → m'=-1/2,
shown only at high level).

Answer format: simplified `y = mx + b` string with integer coefficients
when possible, or `y = (-1/2)x + 3` otherwise.
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _format_line(m_num, m_den, b_num, b_den):
    """Format y = (m_num/m_den)x + (b_num/b_den) as a clean string."""
    # Reduce
    def reduce(p, q):
        if q < 0:
            p, q = -p, -q
        g = math.gcd(abs(p), abs(q)) if p else q
        if g == 0:
            return p, q
        return p // g, q // g
    m_num, m_den = reduce(m_num, m_den)
    b_num, b_den = reduce(b_num, b_den)
    parts = ["y", "="]
    # slope term
    if m_num == 0:
        slope_str = "0"
    elif m_den == 1:
        if m_num == 1:
            slope_str = "x"
        elif m_num == -1:
            slope_str = "-x"
        else:
            slope_str = f"{m_num}x"
    else:
        if m_num < 0:
            slope_str = f"-({abs(m_num)}/{m_den})x"
        else:
            slope_str = f"({m_num}/{m_den})x"
    parts.append(slope_str)
    # intercept term
    if b_num == 0:
        return " ".join(parts)
    if b_den == 1:
        if b_num > 0:
            parts.append(f"+ {b_num}")
        else:
            parts.append(f"- {abs(b_num)}")
    else:
        if b_num > 0:
            parts.append(f"+ {b_num}/{b_den}")
        else:
            parts.append(f"- {abs(b_num)}/{b_den}")
    return " ".join(parts)


class LineParallelPerpQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "line_parallel_perp"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"modes": ["parallel"], "slopes": [-2, -1, 1, 2]}
        if level <= 5:
            return {"modes": ["parallel", "perpendicular"], "slopes": [-2, -1, 1, 2]}
        if level <= 7:
            return {"modes": ["parallel", "perpendicular"], "slopes": [-3, -2, -1, 1, 2, 3]}
        return {"modes": ["parallel", "perpendicular"], "slopes": [-3, -2, -1, 1, 2, 3]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 6133 + level * 47 + 13)

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        mode = rng.choice(cfg["modes"])
        m0 = rng.choice(cfg["slopes"])
        b0 = rng.randint(-5, 5)

        # Point P
        xP = rng.randint(-5, 5)
        yP = rng.randint(-6, 6)

        if mode == "parallel":
            # m' = m0
            new_b = yP - m0 * xP
            if abs(new_b) > 12:
                return None
            ans_str = _format_line(m0, 1, new_b, 1)
            mode_word = "parallel to"
            new_m_num, new_m_den = m0, 1
        else:  # perpendicular
            # m' = -1/m0; only integer if m0 in {1, -1}
            # Pre-filter so we get clean answers
            new_m_num = -1
            new_m_den = m0
            if m0 < 0:
                new_m_num = 1
                new_m_den = -m0
            # b' = yP - m' * xP = (yP * m_den - new_m_num * xP) / m_den
            new_b_num = yP * new_m_den - new_m_num * xP
            new_b_den = new_m_den
            # Reduce
            g = math.gcd(abs(new_b_num), abs(new_b_den)) if new_b_num else new_b_den
            if g == 0:
                g = 1
            nbn, nbd = new_b_num // g, new_b_den // g
            if abs(nbn) > 30:
                return None
            ans_str = _format_line(new_m_num, new_m_den, nbn, nbd)
            mode_word = "perpendicular to"

        question = (
            f"The figure shows a line with equation y = {self._fmt_int_line(m0, b0)} "
            f"and a labelled point P({xP}, {yP}). Find the equation of the line "
            f"that passes through P and is {mode_word} the given line. Express "
            f"the equation in slope-intercept form 'y = mx + b'."
        )

        img = self._render(m0, b0, xP, yP, mode, new_m_num, new_m_den)
        return question, ans_str, img

    @staticmethod
    def _fmt_int_line(m, b):
        parts = []
        if m == 0:
            return f"{b}"
        if m == 1:
            parts.append("x")
        elif m == -1:
            parts.append("-x")
        else:
            parts.append(f"{m}x")
        if b == 0:
            return parts[0]
        parts.append(f"+ {b}" if b > 0 else f"- {abs(b)}")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    def _render(self, m0, b0, xP, yP, mode, new_m_num, new_m_den) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        max_extent = max(abs(xP) + 2, abs(yP) + 2, abs(b0) + 2, 8)
        x_lo, x_hi = -max_extent, max_extent
        y_lo, y_hi = -max_extent, max_extent

        xs = np.linspace(x_lo, x_hi, 200)
        ys = m0 * xs + b0
        ax.plot(xs, ys, color="#1f77b4", linewidth=2,
                label=f"y = {self._fmt_int_line(m0, b0)}")

        # Plot point P
        ax.scatter([xP], [yP], color="#d62728", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(f"P({xP}, {yP})", (xP, yP),
                    textcoords="offset points",
                    xytext=(10, 10), fontsize=12, color="#d62728",
                    fontweight="bold")

        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=10)

        return self.fig_to_pil(fig, dpi=110)

    # ------------------------------------------------------------------ #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Parse both as line equations (slope, intercept) and compare.
        gt_m, gt_b = self._parse_line_eq(ground_truth)
        pred_m, pred_b = self._parse_line_eq(predicted)
        if gt_m is None:
            return super()._check_answer(predicted, ground_truth)
        if pred_m is None:
            return False
        return (abs(gt_m - pred_m) < 0.05 and abs(gt_b - pred_b) < 0.5)

    @staticmethod
    def _parse_line_eq(s):
        """Parse 'y = mx + b' (or variants like '(1/2)x') into (m, b)."""
        if not s:
            return None, None
        t = s.strip().lower()
        # remove any 'y' or 'y =' prefix
        m = re.match(r"^\s*y\s*[=]\s*(.*?)$", t)
        if m:
            t = m.group(1).strip()
        # If the line is just a constant 'y = N'
        try:
            return 0.0, float(t)
        except ValueError:
            pass

        # Try to extract m and b from 'mx + b' or 'mx - b'
        # Replace common LaTeX
        t = t.replace("\\cdot", "*").replace("\\frac", "")
        # Handle (1/2) or 1/2 fractions in slope: e.g. (1/2)x or 1/2x
        # Pattern: slope_str x ± b
        # Find x position
        idx = t.find("x")
        if idx < 0:
            return None, None
        slope_str = t[:idx].strip()
        rest = t[idx + 1:].strip()

        slope_str = slope_str.replace("(", "").replace(")", "").strip()
        if slope_str in ("", "+"):
            slope = 1.0
        elif slope_str == "-":
            slope = -1.0
        else:
            try:
                # support "1/2", "-1/2", "0.5", etc
                if "/" in slope_str:
                    p, q = slope_str.split("/")
                    slope = float(p) / float(q)
                else:
                    slope = float(slope_str.replace("*", ""))
            except ValueError:
                return None, None
        # rest is "+ b" or "- b" or empty
        if not rest:
            intercept = 0.0
        else:
            rest = rest.replace(" ", "")
            sign = 1.0
            if rest[0] in ("+", "-"):
                if rest[0] == "-":
                    sign = -1.0
                rest = rest[1:]
            try:
                if "/" in rest:
                    p, q = rest.split("/")
                    intercept = sign * float(p) / float(q)
                else:
                    intercept = sign * float(rest)
            except ValueError:
                return None, None

        return slope, intercept


import re

if __name__ == "__main__":
    env = LineParallelPerpQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
