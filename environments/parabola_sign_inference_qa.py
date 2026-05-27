"""
Parabola Sign Inference QA.

A parabola y = ax^2 + bx + c is plotted with NO equation labels and minimal
axis annotations: only the x-axis and y-axis lines (with the origin O marked)
are visible. From the curve's shape — opening direction, vertex position
relative to the y-axis, and y-intercept sign — the student infers the signs
of a, b, and c (each as `>0`, `<0`, or `=0` if exactly zero — we avoid the
zero case here).

Sign rules:
  - sign(a) = sign of opening direction: opens up -> a>0; opens down -> a<0.
  - sign(b) -> from axis of symmetry x = -b/(2a). Vertex on the right of
    y-axis (x_v > 0) means -b/(2a) > 0, i.e. sign(b) is opposite to sign(a).
    Vertex on the left (x_v < 0) means sign(b) matches sign(a).
  - sign(c) = sign of y-intercept (value at x=0).

Difficulty axes:
  - L0..L2: vertex always far from y-axis; y-intercept clearly nonzero.
  - L3..L5: smaller magnitudes; closer to axis.
  - L6..L9: vertex closer to y-axis (still nonzero); harder to read.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows a parabola y = ax^2 + bx + c plotted on a coordinate plane (no equation given). From the shape of the curve, determine the SIGNS of a, b, and c. Reply with three signs separated by commas, e.g. `a>0, b<0, c>0`. Place answer in <answer>...</answer>.",
    "Look at the parabola in the figure. Without solving anything, infer the signs of the coefficients a, b, c in y = ax^2 + bx + c from its shape and intercepts. Format `a?0, b?0, c?0` with `?` replaced by `>` or `<`. Place answer in <answer>...</answer>.",
    "From the displayed parabola, determine whether each of a, b, c (in y = ax^2 + bx + c) is positive or negative. Reply as `a>0, b<0, c>0` style. Place answer in <answer>...</answer>.",
    "A parabola is drawn on the coordinate plane. Infer the signs of a, b, and c in y = ax^2 + bx + c. Reply with three comparisons separated by commas, e.g. `a<0, b>0, c<0`. Place answer in <answer>...</answer>.",
    "Determine the signs of a, b, c for the parabola y = ax^2 + bx + c shown in the image. Format your answer like `a>0, b<0, c>0`. Place in <answer>...</answer>.",
    "Examine the parabola in the figure (no equation given). Identify the signs of a, b, c. Format the response as `a>0, b<0, c>0`. Put it in <answer>...</answer>.",
    "Given the parabola plotted, infer sign(a), sign(b), sign(c) in y = ax^2 + bx + c. Output three signs in order separated by commas. Place in <answer>...</answer>.",
    "From the curve drawn (a parabola y = ax^2 + bx + c), state the signs of the three coefficients. Format `a>0, b<0, c>0`. Place answer in <answer>...</answer>.",
    "The figure shows a parabola y = ax^2 + bx + c without labels. Determine the signs of a, b, and c. Place in <answer>...</answer> as `a?0, b?0, c?0`.",
    "Use only the visual shape of the parabola y = ax^2 + bx + c shown to infer the signs of a, b, c. Reply like `a>0, b<0, c>0`. Place in <answer>...</answer>.",
    "Identify whether each of a, b, c in y = ax^2 + bx + c is positive or negative for the parabola in the image. Place the three signed comparisons in <answer>...</answer>.",
    "Read the signs of a, b, c off the parabola y = ax^2 + bx + c shown. Comma-separated like `a>0, b<0, c>0`. Place in <answer>...</answer>.",
    "From the parabola in the figure, deduce the signs of the coefficients a, b, c (in y = ax^2 + bx + c). Reply with three sign expressions. Place in <answer>...</answer>.",
    "Look at the curve in the figure. Determine sign(a), sign(b), sign(c) for y = ax^2 + bx + c. Format `a>0, b<0, c>0`. Place in <answer>...</answer>.",
    "The plot shows a parabola y = ax^2 + bx + c. Infer the signs of a, b, c from its appearance. Output three comparisons separated by commas. Place in <answer>...</answer>.",
    "From the shape of the displayed parabola (y = ax^2 + bx + c, equation hidden), determine the signs of a, b, c. Place the answer in <answer>...</answer> like `a>0, b<0, c>0`.",
]


def _normalize_sign_str(s):
    """Normalize a string like 'a>0, b<0, c>0' or 'a > 0; b < 0; c > 0' or
    '+, -, +' into a tuple ('+', '-', '+'). Returns None if cannot parse."""
    import re
    if s is None:
        return None
    txt = s.strip().replace(';', ',').replace('|', ',').replace('\n', ',')
    # Try the explicit-name form first
    pat = re.compile(r'a\s*([><=])\s*0[^a-zA-Z<>=]*?b\s*([><=])\s*0[^a-zA-Z<>=]*?c\s*([><=])\s*0',
                     flags=re.IGNORECASE)
    m = pat.search(txt)
    if m:
        return tuple('+' if g == '>' else ('-' if g == '<' else '0') for g in m.groups())
    # Try just three sign tokens (>0, <0, ...)
    sign_tokens = re.findall(r'([><=])\s*0', txt)
    if len(sign_tokens) >= 3:
        return tuple('+' if g == '>' else ('-' if g == '<' else '0')
                     for g in sign_tokens[:3])
    # Try plain '+/-' separated by commas
    plain = [t.strip() for t in txt.split(',') if t.strip() in ('+', '-', '0',
                                                                 'positive', 'negative', 'zero',
                                                                 'pos', 'neg')]
    if len(plain) >= 3:
        def _norm(x):
            x = x.lower()
            if x in ('+', 'positive', 'pos'):
                return '+'
            if x in ('-', 'negative', 'neg'):
                return '-'
            return '0'
        return tuple(_norm(plain[i]) for i in range(3))
    return None


class ParabolaSignInferenceQA(StandaloneVisualEnv):
    ENV_NAME = "parabola_sign_inference"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            # Vertex clearly off the y-axis, |a| = 1, large c-magnitude.
            return {"vx_min": 1.5, "vx_max": 3.0, "vy_range": 3,
                    "a_pool": [-1, 1], "c_min_abs": 1}
        if level <= 5:
            return {"vx_min": 1.0, "vx_max": 3.0, "vy_range": 4,
                    "a_pool": [-2, -1, 1, 2], "c_min_abs": 1}
        return {"vx_min": 0.5, "vx_max": 2.5, "vy_range": 5,
                "a_pool": [-3, -2, -1, 1, 2, 3], "c_min_abs": 1}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1051 + level * 89 + 23)

        for _ in range(40):
            a = rng.choice(cfg["a_pool"])
            # Vertex x-coord: use half-step floats to make the visual position
            # look natural but keep b and c integer-friendly when possible.
            # We allow integer vertex_x since 2*a*vx then gives integer b.
            vx_sign = rng.choice([-1, 1])
            vx_mag = rng.choice([n / 2 for n in range(int(cfg["vx_min"] * 2),
                                                       int(cfg["vx_max"] * 2) + 1)])
            vx = vx_sign * vx_mag
            vy = rng.randint(-cfg["vy_range"], cfg["vy_range"])

            b = -2 * a * vx
            c = a * vx * vx + vy

            # Require b nonzero (else sign undefined) and c nonzero.
            if abs(b) < 1e-9:
                continue
            if abs(c) < cfg["c_min_abs"]:
                continue
            break
        else:
            return None

        # Determine signs
        def _sign(x):
            if x > 1e-9:
                return '+'
            if x < -1e-9:
                return '-'
            return '0'
        sa, sb, sc = _sign(a), _sign(b), _sign(c)

        # Format the canonical answer string as 'a>0, b<0, c>0'
        sym = {'+': '>', '-': '<', '0': '='}
        answer = f"a{sym[sa]}0, b{sym[sb]}0, c{sym[sc]}0"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(a, b, c, vx, vy)
        return question, answer, img

    # ---------------- answer check ---------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        gp = _normalize_sign_str(ground_truth)
        pp = _normalize_sign_str(predicted)
        if gp is not None and pp is not None:
            return gp == pp
        return super()._check_answer(predicted, ground_truth)

    # ---------------- render ---------------- #
    def _render(self, a, b, c, vx, vy) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Axis window: ensure vertex and y-intercept are both in view; show
        # enough of the curve to read opening direction.
        x_lo = min(-4, vx - 4)
        x_hi = max(4, vx + 4)
        xs = np.linspace(x_lo, x_hi, 200)
        ys = a * xs * xs + b * xs + c
        # Pick y window from data
        y_lo = min(min(ys), vy, c, -2) - 1
        y_hi = max(max(ys), vy, c, 2) + 1

        ax.plot(xs, ys, color="#1a5276", linewidth=2.4)

        # Axes lines and origin only — NO tick labels, NO grid (per spec).
        ax.axhline(0, color="#000000", linewidth=1.1, zorder=2)
        ax.axvline(0, color="#000000", linewidth=1.1, zorder=2)
        # Mark the origin O
        ax.plot([0], [0], "o", color="#000000", markersize=4, zorder=4)
        ax.annotate("O", (0, 0), xytext=(5, -12), textcoords="offset points",
                    fontsize=12, color="#000000", fontweight="bold")
        # Label x and y arrows
        ax.annotate("x", xy=(x_hi, 0), xytext=(-12, 5),
                    textcoords="offset points", fontsize=12, fontweight="bold")
        ax.annotate("y", xy=(0, y_hi), xytext=(5, -12),
                    textcoords="offset points", fontsize=12, fontweight="bold")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([])
        ax.set_yticks([])
        # Hide spines, only the axhline/axvline serve as the axes
        for sp in ax.spines.values():
            sp.set_visible(False)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
