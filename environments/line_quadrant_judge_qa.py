"""
Line Quadrant Judge QA.

A line y = kx + b is drawn on a coordinate plane. The quadrants of the plane
are numbered I, II, III, IV in the standard way:
   II  | I        (x<0,y>0) (x>0,y>0)
   --------
   III | IV       (x<0,y<0) (x>0,y<0)

The student must list the quadrants the line passes through, comma-separated,
in ascending order: e.g. `I, III` or `II, IV` for axis-symmetric simple lines,
`I, II, III` etc. for offset lines.

Difficulty axes:
  - L0..L2: integer slope and integer intercept; clear quadrant set.
  - L3..L5: rational slope, larger intercepts.
  - L6..L9: tighter axis range so the model must reason from y = kx + b
    rather than just visually following the line.
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
    "The image shows a straight line y = kx + b on the coordinate plane. List the quadrants (I, II, III, IV) the line passes through, comma-separated in ascending order. Place the answer in <answer>...</answer>.",
    "Examine the line drawn in the figure. Which quadrants of the coordinate plane does it pass through? Reply with quadrant labels (I, II, III, IV) separated by commas. Place in <answer>...</answer>.",
    "Identify all quadrants the displayed line y = kx + b passes through. Use Roman numerals I, II, III, IV separated by commas. Place answer in <answer>...</answer>.",
    "From the line shown, list every quadrant it crosses. Use Roman numerals (I-IV), comma-separated, ascending. Place the answer in <answer>...</answer>.",
    "List the quadrants traversed by the line y = kx + b shown in the figure. Format `I, III` style. Place in <answer>...</answer>.",
    "The displayed line passes through some of the four quadrants. Identify them. Comma-separated Roman numerals in <answer>...</answer>.",
    "Determine which quadrants the line y = kx + b drawn on the coordinate plane passes through. Place the comma-separated list in <answer>...</answer>.",
    "From the figure, name all quadrants the line crosses. Use I, II, III, IV separated by commas. Place in <answer>...</answer>.",
    "Look at the line in the image. Which of quadrants I-IV does it pass through? Reply with Roman numerals separated by commas. Place in <answer>...</answer>.",
    "List the quadrants the displayed line y = kx + b enters. Format `II, IV` style. Place answer in <answer>...</answer>.",
    "Identify the quadrants crossed by the line drawn. Return the labels (I, II, III, IV) ascending, comma-separated. Place in <answer>...</answer>.",
    "Read the figure: which quadrants does the line y = kx + b pass through? Comma-separated Roman numerals in <answer>...</answer>.",
    "Determine the set of quadrants the displayed line passes through. Output them in ascending order, comma-separated. Place in <answer>...</answer>.",
    "List all quadrants visited by the line shown in the image. Format like `I, II, III`. Place in <answer>...</answer>.",
    "From the diagram, which quadrants does the line y = kx + b cross? Reply with comma-separated Roman numerals. Place in <answer>...</answer>.",
    "The figure shows a straight line. Determine which quadrants of the coordinate plane the line passes through, and report them ascending with commas. Place in <answer>...</answer>.",
    "Examine the displayed line and report all quadrants it passes through. Use I, II, III, IV. Place answer in <answer>...</answer>.",
]


def _quadrants_of_line(k: float, b: float, samples: int = 4001):
    """Return sorted tuple of quadrant numbers (1..4) the line y = kx + b
    passes through. Uses a wide x-sweep with a fine sample so we don't miss
    near-axis transitions. Points exactly on an axis are treated as belonging
    to neither quadrant (axes are quadrant boundaries)."""
    xs = np.linspace(-1000, 1000, samples)
    ys = k * xs + b
    qs = set()
    for x, y in zip(xs, ys):
        if x > 1e-9 and y > 1e-9:
            qs.add(1)
        elif x < -1e-9 and y > 1e-9:
            qs.add(2)
        elif x < -1e-9 and y < -1e-9:
            qs.add(3)
        elif x > 1e-9 and y < -1e-9:
            qs.add(4)
        if len(qs) == 4:
            break
    return tuple(sorted(qs))


def _normalize_quadrant_str(s):
    """Parse a quadrant list. Accepts I/II/III/IV, 1/2/3/4, mixed casing,
    optional braces, semi/comma/space separators. Returns sorted tuple of
    ints, or None on parse failure."""
    import re
    if s is None:
        return None
    txt = s.strip()
    # Replace common separators with comma, drop braces / parens
    for ch in '{}[]()':
        txt = txt.replace(ch, '')
    txt = txt.replace(';', ',').replace('|', ',').replace('and', ',')
    # Split tokens (commas or whitespace)
    raw_tokens = re.split(r'[,\s]+', txt)
    tokens = [t.strip().upper() for t in raw_tokens if t.strip()]
    roman_to_int = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    out = set()
    for t in tokens:
        if t in roman_to_int:
            out.add(roman_to_int[t])
        elif t in ('1', '2', '3', '4'):
            out.add(int(t))
        else:
            # Unknown token — bail
            return None
    if not out:
        return None
    return tuple(sorted(out))


class LineQuadrantJudgeQA(StandaloneVisualEnv):
    ENV_NAME = "line_quadrant_judge"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"slope_int_only": True, "slope_max": 2,
                    "intercept_max": 2 + level, "show_grid": True,
                    "axis_pad": 5}
        if level <= 5:
            return {"slope_int_only": False, "slope_max": 3,
                    "intercept_max": 4, "show_grid": True,
                    "axis_pad": 5}
        return {"slope_int_only": False, "slope_max": 4,
                "intercept_max": 6, "show_grid": False,
                "axis_pad": 5}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1063 + level * 97 + 11)

        for _ in range(40):
            if cfg["slope_int_only"]:
                k_num = rng.choice([-cfg["slope_max"], -1, 1, cfg["slope_max"]])
                k_den = 1
            else:
                k_num = rng.choice(list(range(-cfg["slope_max"], 0)) +
                                   list(range(1, cfg["slope_max"] + 1)))
                k_den = rng.choice([1, 1, 2, 3])
            k = k_num / k_den
            # Special L0 trivial: y = x (k=1, b=0)
            if level == 0 and rng.random() < 0.4:
                k_num, k_den = 1, 1
                k = 1.0
                b = 0
            else:
                b = rng.randint(-cfg["intercept_max"], cfg["intercept_max"])
            # Avoid k = 0 (degenerate horizontal line — not "y = kx + b" emphasis)
            if abs(k) < 1e-9:
                continue
            # Compute quadrants the line passes through
            qs = _quadrants_of_line(k, b)
            # Should always be 2 or 3 quadrants for a non-degenerate line.
            if len(qs) < 2:
                continue
            break
        else:
            return None

        # Format answer as Roman numerals comma-separated ascending
        roman = {1: "I", 2: "II", 3: "III", 4: "IV"}
        answer = ", ".join(roman[q] for q in qs)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(k, b, cfg)
        return question, answer, img

    # ---------------- answer check ---------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        gp = _normalize_quadrant_str(ground_truth)
        pp = _normalize_quadrant_str(predicted)
        if gp is not None and pp is not None:
            return gp == pp
        return super()._check_answer(predicted, ground_truth)

    # ---------------- render ---------------- #
    def _render(self, k, b, cfg) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        x_lim = cfg["axis_pad"]
        y_lim = cfg["axis_pad"]
        xs = np.linspace(-x_lim - 1, x_lim + 1, 200)
        ys = k * xs + b
        ax.plot(xs, ys, color="#1a5276", linewidth=2.4, zorder=3)

        # Axes
        ax.axhline(0, color="#000000", linewidth=1.0, zorder=2)
        ax.axvline(0, color="#000000", linewidth=1.0, zorder=2)

        if cfg["show_grid"]:
            ax.grid(True, alpha=0.3, linestyle="--")

        # Label quadrant markers visually for reference (helps reading the
        # figure; not the answer itself).
        offset = x_lim * 0.7
        ax.text(offset, offset, "I", fontsize=14, color="#888888",
                ha="center", va="center", alpha=0.8)
        ax.text(-offset, offset, "II", fontsize=14, color="#888888",
                ha="center", va="center", alpha=0.8)
        ax.text(-offset, -offset, "III", fontsize=14, color="#888888",
                ha="center", va="center", alpha=0.8)
        ax.text(offset, -offset, "IV", fontsize=14, color="#888888",
                ha="center", va="center", alpha=0.8)

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xticks(range(-x_lim, x_lim + 1))
        ax.set_yticks(range(-y_lim, y_lim + 1))
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # Title gives k and b in textbook form so the model can reason
        # algebraically too; the figure itself shows the line.
        def _fmt_slope(num, den):
            if den == 1:
                return str(num)
            return f"{num}/{den}"
        # We don't know k_num/k_den here cleanly; print k as float
        if abs(k - round(k)) < 1e-9:
            k_disp = str(int(round(k)))
        else:
            # try to detect rational form
            for d in (2, 3, 4, 5):
                if abs(k * d - round(k * d)) < 1e-9:
                    n = int(round(k * d))
                    g = math.gcd(abs(n), d)
                    n //= g
                    d2 = d // g
                    k_disp = f"{n}/{d2}"
                    break
            else:
                k_disp = f"{k:g}"
        if b == 0:
            eq = f"y = {k_disp}x"
        elif b > 0:
            eq = f"y = {k_disp}x + {b}"
        else:
            eq = f"y = {k_disp}x - {abs(b)}"
        ax.set_title(eq, fontsize=12)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
