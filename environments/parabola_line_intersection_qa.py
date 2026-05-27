"""
Find the x-coordinates where a parabola y = a x² + b x + c intersects a
line y = m x + n. Both curves are drawn on a coordinate plane with their
equations labeled.

Solving a x² + b x + c = m x + n
   ⇒  a x² + (b - m) x + (c - n) = 0
The number of intersections depends on the discriminant.

Output policy:
  - Two real intersections → comma-separated x-values, sorted ascending
    (e.g. "-1,1" or "0,3"). Spaces around the comma are accepted.
  - One intersection (tangent) → single x-value (e.g. "2").
  (No-intersection cases are not generated for this env to keep the answer
   numeric and avoid free-text matching brittleness.)

L0: parabola y = x², line y = 1 → intersect at x = -1 and x = 1.
"""
import math
import random
import re
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_TWO = [
    "The figure shows the parabola y = {par} and the line y = {ln}. Find the x-coordinates of their intersection points. List them as a comma-separated pair sorted ascending (e.g. -1,2). Place the answer in <answer>...</answer>.",
    "On the coordinate plane the parabola y = {par} meets the line y = {ln} at two points. Compute the two x-values, sorted from smallest to largest, separated by a comma. Place the answer in <answer>...</answer>.",
    "Determine where y = {par} and y = {ln} intersect. Output the two x-coordinates, comma-separated and ascending. Place in <answer>...</answer>.",
    "The parabola y = {par} and the line y = {ln} cross at two points. Give the two x-values (sorted ascending, comma-separated). Place the answer in <answer>...</answer>.",
    "Find the intersection x-coordinates of y = {par} and y = {ln}. Comma-separated, sorted small to large. Place the answer in <answer>...</answer>.",
    "The image shows y = {par} (parabola) and y = {ln} (line). Compute their intersection x-values; format as a comma-separated, ascending pair. Place in <answer>...</answer>.",
    "Solve {par} = {ln} for x to find where the parabola meets the line. Two real roots — list them comma-separated, sorted small to large. Place answer in <answer>...</answer>.",
    "Identify the x-coordinates at which the parabola y = {par} crosses the line y = {ln}. Output as `x1,x2` ascending. Place answer in <answer>...</answer>.",
    "Find the two intersection x-values of y = {par} and y = {ln}. Comma-separated, ascending order. Place value in <answer>...</answer>.",
    "Where does y = {par} intersect y = {ln}? Provide the two x-coordinates, comma-separated, sorted ascending. Place in <answer>...</answer>.",
    "Solve the system y = {par} and y = {ln}; report the x-values of the two intersections, comma-separated and sorted. Place answer in <answer>...</answer>.",
    "From the figure, find the two x-values where the parabola y = {par} meets the line y = {ln}. Format `x1,x2` ascending. Place in <answer>...</answer>.",
    "The parabola y = {par} cuts the line y = {ln} at two points. Compute the two x-coordinates and report them as a comma-separated, ascending pair. Place in <answer>...</answer>.",
    "Use the labeled equations y = {par} and y = {ln} to find both intersection x-values. Sort smallest to largest, comma-separated. Place answer in <answer>...</answer>.",
    "Compute the x-values at which y = {par} and y = {ln} are equal. Two real solutions; list them ascending and comma-separated. Place in <answer>...</answer>.",
    "From the figure, the line y = {ln} crosses the parabola y = {par} at two distinct points. Provide both x-coordinates as `x1,x2` (smallest first). Place answer in <answer>...</answer>.",
    "The parabola y = {par} intersects the line y = {ln}. Find the two x-coordinates of the intersections; output them as a comma-separated ascending pair. Place in <answer>...</answer>.",
]

_TEMPLATES_TANGENT = [
    "The line y = {ln} is tangent to the parabola y = {par}. Find the x-coordinate of the tangent point. Place the value in <answer>...</answer>.",
    "y = {ln} touches the parabola y = {par} at exactly one point. Compute the x-coordinate of this point. Place the answer in <answer>...</answer>.",
    "The parabola y = {par} and the line y = {ln} meet at exactly one point (tangent). Find that point's x-coordinate. Place value in <answer>...</answer>.",
    "Find the x-coordinate where the line y = {ln} is tangent to the parabola y = {par}. Place answer in <answer>...</answer>.",
    "y = {par} meets y = {ln} at a single tangent point. Determine its x-coordinate. Place the value in <answer>...</answer>.",
]


def _format_quadratic(a: int, b: int, c: int) -> str:
    """Format a x^2 + b x + c with neat signs."""
    parts = []
    # ax^2
    if a == 1:
        parts.append("x²")
    elif a == -1:
        parts.append("-x²")
    elif a != 0:
        parts.append(f"{a}x²")
    # bx
    if b == 0:
        pass
    elif b == 1:
        parts.append("+ x" if parts else "x")
    elif b == -1:
        parts.append("- x")
    elif b > 0:
        parts.append(f"+ {b}x" if parts else f"{b}x")
    else:
        parts.append(f"- {abs(b)}x")
    # c
    if c == 0:
        pass
    elif c > 0:
        parts.append(f"+ {c}" if parts else f"{c}")
    else:
        parts.append(f"- {abs(c)}")
    if not parts:
        return "0"
    return " ".join(parts)


def _format_linear(m: int, n: int) -> str:
    parts = []
    if m == 0:
        parts.append("")
    elif m == 1:
        parts.append("x")
    elif m == -1:
        parts.append("-x")
    else:
        parts.append(f"{m}x")
    if m == 0:
        pass
    elif n == 0:
        pass
    elif n > 0:
        parts.append(f"+ {n}")
    else:
        parts.append(f"- {abs(n)}")
    if m == 0:
        return f"{n}"
    return " ".join(parts).strip()


class ParabolaLineIntersectionQA(StandaloneVisualEnv):
    ENV_NAME = "parabola_line_intersection"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {"level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        rng = random.Random((self.seed or 0) * 1481 + level * 73 + 5)

        # Decide whether tangent (high-level only) or two-point (always)
        if level >= 6:
            tangent_prob = 0.25
        else:
            tangent_prob = 0.0

        if level == 0:
            # Canonical L0: parabola y = x², line y = 1 → x = ±1
            a, b, c = 1, 0, 0
            m, n = 0, 1
            r1, r2 = -1, 1
        else:
            # Choose two integer roots r1 < r2 of (a x^2 + (b-m) x + (c - n)) = 0
            # Strategy: pick r1, r2; then pick a; then pick line slope m and
            # solve b, c, n consistent.
            for _ in range(100):
                if rng.random() < tangent_prob:
                    # tangent: r1 == r2
                    if level <= 6:
                        r1 = r2 = rng.randint(-3, 3)
                    else:
                        r1 = r2 = rng.randint(-5, 5)
                else:
                    if level <= 2:
                        r1 = rng.randint(-3, 3)
                        r2 = rng.randint(-3, 3)
                    elif level <= 4:
                        r1 = rng.randint(-5, 5)
                        r2 = rng.randint(-5, 5)
                    elif level <= 6:
                        r1 = rng.randint(-6, 6)
                        r2 = rng.randint(-6, 6)
                    elif level <= 8:
                        # 2026-05-04 R3 retry: shrink L7-L8 from (-8..8)
                        # to (-7..7) for slightly simpler intersection ints.
                        r1 = rng.randint(-7, 7)
                        r2 = rng.randint(-7, 7)
                    else:
                        # L9 keeps softened range from prior R3.
                        r1 = rng.randint(-8, 8)
                        r2 = rng.randint(-8, 8)
                    if r1 == r2:
                        continue
                    if r1 > r2:
                        r1, r2 = r2, r1

                # Choose a
                if level <= 2:
                    a = rng.choice([1, 1, -1])
                elif level <= 5:
                    a = rng.choice([1, -1, 1, 2, -2])
                else:
                    a = rng.choice([1, -1, 2, -2, 1, -1])
                if a == 0:
                    continue

                # Now: a x^2 + (b - m) x + (c - n) = a (x - r1)(x - r2)
                # Pick b, c independently, then derive m, n.
                if level <= 3:
                    b = rng.randint(-3, 3)
                    c = rng.randint(-4, 4)
                elif level <= 6:
                    b = rng.randint(-5, 5)
                    c = rng.randint(-6, 6)
                elif level <= 8:
                    # 2026-05-04 R3 retry: shrink L7-L8 b,c too (matches r1,r2).
                    b = rng.randint(-7, 7)
                    c = rng.randint(-7, 7)
                else:
                    # L9 keeps softened range.
                    b = rng.randint(-8, 8)
                    c = rng.randint(-8, 8)
                # From a (x - r1)(x - r2) = a x^2 - a(r1+r2) x + a r1 r2
                # ⇒ b - m = -a(r1 + r2)  ⇒  m = b + a(r1 + r2)
                # ⇒ c - n =  a r1 r2     ⇒  n = c - a*r1*r2
                m = b + a * (r1 + r2)
                n = c - a * r1 * r2
                # Bound m, n to keep the figure readable. L9 allows larger
                # bounds because roots and coefficients are wider.
                if level == 9:
                    # 2026-05-04 R3: softened — bounds (24/80) → (16/50)
                    # match the narrower r1,r2 / b,c ranges above.
                    if abs(m) > 16 or abs(n) > 50:
                        continue
                elif abs(m) > 12 or abs(n) > 30:
                    continue
                # Avoid degenerate parabola (a == 0)
                if a == 0:
                    continue
                # Sanity: re-solve and check our roots match
                # discriminant test
                A = a
                B = b - m
                C = c - n
                disc = B * B - 4 * A * C
                # Tangent ⇒ disc should be 0
                if r1 == r2:
                    if disc != 0:
                        continue
                else:
                    if disc <= 0:
                        continue
                    rt = math.sqrt(disc)
                    cr1 = (-B - rt) / (2 * A)
                    cr2 = (-B + rt) / (2 * A)
                    if abs(cr1 - r1) > 1e-6 or abs(cr2 - r2) > 1e-6:
                        # could swap if a < 0
                        if abs(cr1 - r2) > 1e-6 or abs(cr2 - r1) > 1e-6:
                            continue
                break
            else:
                return None

        # Build answer
        if r1 == r2:
            sidx = (self.seed or 0) % len(_TEMPLATES_TANGENT)
            question = _TEMPLATES_TANGENT[sidx].format(
                par=_format_quadratic(a, b, c),
                ln=_format_linear(m, n))
            answer = f"{r1}"
        else:
            sidx = (self.seed or 0) % len(_TEMPLATES_TWO)
            question = _TEMPLATES_TWO[sidx].format(
                par=_format_quadratic(a, b, c),
                ln=_format_linear(m, n))
            answer = f"{r1},{r2}"

        img = self._render(a, b, c, m, n, r1, r2)
        return question, answer, img

    def _render(self, a, b, c, m, n, r1, r2) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # x-range based on roots and useful zoom
        x_lo = min(r1, r2) - 2.5
        x_hi = max(r1, r2) + 2.5
        x_lo = min(x_lo, -3)
        x_hi = max(x_hi, 3)
        xs = np.linspace(x_lo, x_hi, 300)
        ys_par = a * xs * xs + b * xs + c
        ys_lin = m * xs + n
        ax.plot(xs, ys_par, color="#1f77b4", linewidth=2.0,
                label=f"y = {_format_quadratic(a, b, c)}")
        ax.plot(xs, ys_lin, color="#d62728", linewidth=2.0,
                label=f"y = {_format_linear(m, n)}")
        # Intersection points
        if r1 == r2:
            pts_x = [r1]
        else:
            pts_x = [r1, r2]
        pts_y = [m * px + n for px in pts_x]
        ax.scatter(pts_x, pts_y, color="#2ca02c", s=70, zorder=5,
                   edgecolor="black", linewidth=1.0)
        # Axes & grid
        ax.axhline(0, color="#444", linewidth=0.8)
        ax.axvline(0, color="#444", linewidth=0.8)
        ax.grid(True, alpha=0.3, linestyle="--")

        # y-range: clip to visible
        all_y = list(ys_par) + list(ys_lin) + pts_y
        y_lo = max(min(all_y) - 2, -25)
        y_hi = min(max(all_y) + 2, 25)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=10)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ---------- custom answer checking for comma-separated x-list ---------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Parse a list of x-values from prediction & ground truth and compare
        them as a sorted set within tolerance. Falls back to base behaviour
        if either side fails to parse as a numeric list/value."""
        gt_vals = self._parse_xlist(ground_truth)
        pred_vals = self._parse_xlist(predicted)
        if gt_vals is None:
            return super()._check_answer(predicted, ground_truth)
        if pred_vals is None:
            return False
        if len(pred_vals) != len(gt_vals):
            return False
        for p, g in zip(sorted(pred_vals), sorted(gt_vals)):
            if abs(p - g) > 0.05 * max(1.0, abs(g)) + 0.05:
                return False
        return True

    @staticmethod
    def _parse_xlist(s: str):
        """Parse '-1,1', '-1, 1', '{-1, 1}', 'x = -1, 1', etc.  to [-1, 1].
        Single value '2' or 'x = 2' → [2]. Returns None if unparseable."""
        if s is None:
            return None
        t = s.strip().lower()
        # Remove common prefixes / wrappings
        t = re.sub(r"^x\s*=\s*", "", t)
        t = t.strip("{}[]() ").strip()
        # Split on commas / "and" / semicolons / "or"
        parts = re.split(r"[,;]|\band\b|\bor\b", t)
        vals = []
        for p in parts:
            p = p.strip().strip("{}[]() ")
            p = re.sub(r"^x\s*=\s*", "", p).strip()
            if not p:
                continue
            # Try direct float parse
            try:
                vals.append(float(p))
                continue
            except ValueError:
                pass
            # Allow simple expressions (e.g. "-3/2")
            m = re.match(r"^(-?\d+)\s*/\s*(\d+)$", p)
            if m:
                vals.append(int(m.group(1)) / int(m.group(2)))
                continue
            return None
        if not vals:
            return None
        return vals
