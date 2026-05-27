"""
Line Slope Intercept QA (batch 3, 2026-04-14).

Target: Coordinate sub-category (-6.25 across splits). Renders a
straight line on a labelled coordinate grid and asks for either the slope
or the y-intercept. Format is constant short-numeric (fraction or integer).

Difficulty axes:
  A) Pattern H — parameter range expansion: slope space grows
     L0: integer slope in {-2,-1,1,2}, intercept integer in [-4,4]
     L6: rational slope p/q with q in {2,3,4}, intercept in [-6,6]
     L9: rational slope, negative both signs, tighter ticks
  B) Pattern G — hint hiding: grid/tick labels thinned at higher levels
  C) Pattern D — ambiguity: axis limits grow so line is subtle

2026-05-03 extension (M11 / LF-T1 + M15 / LF-T5 + M18 / LF-T8):
added 3 new ask modes alongside slope/intercept:
  - `full_equation`: "Give the equation of the line in y=mx+b form"
    (mirrors reference LF-T1 — line through two lattice points → equation)
  - `parallel_perp_eq`: "Find the equation of the line through P parallel
    [or perpendicular] to the line shown" (mirrors LF-T5)
  - `translate_intercept`: "If the line is shifted up by k units, what is
    the new y-intercept?" (mirrors LF-T8)
The visual render stays the same single-line plot; only question + answer
change. Answers stay numeric (intercept value) or short string (`y=mx+b`).
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

def _format_slope(num: int, den: int) -> str:
    if den == 0:
        return "undefined"
    # Normalise sign
    if den < 0:
        num, den = -num, -den
    g = math.gcd(abs(num), abs(den))
    if g:
        num //= g
        den //= g
    if den == 1:
        return str(num)
    return f"{num}/{den}"

class LineSlopeInterceptQA(StandaloneVisualEnv):
    ENV_NAME = "line_slope_intercept"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            # Pattern H: parameter range expansion
            "allow_fraction": level >= 3,
            "max_den": 2 + level // 2 if level >= 3 else 1,   # 1..6
            "intercept_range": 3 + level,                      # 3..12
            "slope_num_range": 2 + level // 2,                 # 2..6
            # Pattern G: hide more labels
            "show_minor_ticks": level <= 2,
            "show_grid": level <= 5,
            # Pattern D: ambiguity — axis window expansion
            "axis_pad": 2 + level // 3,                        # 2..5
            # MCQ-like? No, keep numeric throughout
            "ask_slope_bias": 0.5,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["max_den"] * 10 + cfg["intercept_range"]

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        # Generate slope (numerator / denominator)
        if cfg["allow_fraction"] and rng.random() < 0.7:
            den = rng.randint(2, cfg["max_den"])
            num_abs = rng.randint(1, cfg["slope_num_range"])
            num = num_abs * rng.choice([-1, 1])
        else:
            den = 1
            num_abs = rng.randint(1, max(1, cfg["slope_num_range"]))
            num = num_abs * rng.choice([-1, 1])
        slope = num / den
        if abs(slope) < 1e-9:
            return None

        # Intercept — integer
        b = rng.randint(-cfg["intercept_range"], cfg["intercept_range"])
        if b == 0 and rng.random() < 0.5:
            # avoid all-b=0; allow a few though
            b = rng.choice([-1, 1, -2, 2])

        # Axis window
        pad = cfg["axis_pad"]
        x_min = -6 - pad
        x_max = 6 + pad
        y_min = -8 - pad
        y_max = 8 + pad

        # Ensure the line is visible: at x=0 we have y=b; at a midpoint x we
        # have slope * x + b. Clip the window if needed.
        y_at_x_min = slope * x_min + b
        y_at_x_max = slope * x_max + b
        if min(y_at_x_min, y_at_x_max) > y_max or max(y_at_x_min, y_at_x_max) < y_min:
            return None

        # Choose ask type: slope, intercept, or one of the 3 new extensions
        # (full_equation / parallel_perp_eq / translate_intercept). Bias
        # toward old asks to keep training distribution stable.
        ask_pool = ["slope", "slope", "intercept", "intercept",
                    "full_equation", "parallel_perp_eq", "translate_intercept"]
        ask = rng.choice(ask_pool)

        if ask == "slope":
            answer = _format_slope(num, den)
            q = ("What is the slope of the line shown in the graph? Give your "
                 "answer as an integer or a simplified fraction like 'p/q'.")
        elif ask == "intercept":
            answer = str(b)
            q = ("What is the y-intercept of the line shown in the graph? Give "
                 "your answer as an integer.")
        elif ask == "full_equation":
            # Build equation y=mx+b string.
            slope_str = _format_slope(num, den)
            if slope_str == "1":
                m_part = "x"
            elif slope_str == "-1":
                m_part = "-x"
            else:
                m_part = f"{slope_str}x"
            if b > 0:
                eq = f"y={m_part}+{b}"
            elif b < 0:
                eq = f"y={m_part}-{abs(b)}"
            else:
                eq = f"y={m_part}"
            answer = eq
            q = ("Give the equation of the line shown in the graph in the form "
                 "y=mx+b (using a simplified fraction p/q for non-integer slope). "
                 "Use exact format like 'y=2x-3' or 'y=1/2x+1' (no spaces).")
        elif ask == "parallel_perp_eq":
            # Pick a point P=(p_x, p_y) on integer coords NOT on the line.
            # Compute the b' for the parallel line: y = slope * x + (p_y - slope*p_x).
            # For integer-friendly answer, force slope rational with den=1.
            if den != 1:
                # Skip when slope isn't integer — keeps answer integer.
                return None
            p_x = rng.randint(-3, 3)
            p_y = rng.randint(-3, 3)
            if (p_y - slope * p_x) == b:
                p_y += 1  # force not-on-line
            mode_pp = rng.choice(["parallel", "perpendicular"])
            if mode_pp == "parallel":
                m_new = slope
                b_new = p_y - m_new * p_x
                m_str = _format_slope(int(m_new), 1)
            else:
                # Perpendicular: m_new = -1/slope. Only well-defined if slope ≠ 0.
                if slope == 0:
                    return None
                # We need int slope. Perpendicular = -1/slope. Skip unless |slope|==1
                # so the answer stays a clean integer.
                if abs(slope) != 1:
                    return None
                m_new = -1 / slope
                b_new = p_y - m_new * p_x
                m_str = _format_slope(int(m_new), 1)
            if abs(b_new - round(b_new)) > 1e-6:
                return None
            b_new_int = int(round(b_new))
            if m_str == "1":
                m_part = "x"
            elif m_str == "-1":
                m_part = "-x"
            else:
                m_part = f"{m_str}x"
            if b_new_int > 0:
                eq = f"y={m_part}+{b_new_int}"
            elif b_new_int < 0:
                eq = f"y={m_part}-{abs(b_new_int)}"
            else:
                eq = f"y={m_part}"
            answer = eq
            q = (f"A line is shown in the graph. Find the equation of the line "
                 f"that passes through the point ({p_x}, {p_y}) and is "
                 f"{mode_pp} to the line shown. Give the equation in the form "
                 f"y=mx+b (no spaces).")
        else:  # translate_intercept
            # Shift line up/down by k; new y-intercept = b + k (up) or b - k (down).
            direction = rng.choice(["up", "down"])
            k = rng.randint(1, 4)
            if direction == "up":
                new_b = b + k
            else:
                new_b = b - k
            answer = str(new_b)
            q = (f"The graph shows a line. If the line is shifted {k} unit"
                 f"{'s' if k > 1 else ''} {direction}, what is the new "
                 f"y-intercept of the translated line? Give an integer.")

        image = self._render(slope, b, cfg, x_min, x_max, y_min, y_max)
        return q, answer, image

    def _render(self, slope, b, cfg, x_min, x_max, y_min, y_max) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        fig, ax = plt.subplots(figsize=(6.5 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        # Grid
        if cfg["show_grid"]:
            ax.grid(True, which="major", alpha=0.3, linestyle="--")
            if cfg["show_minor_ticks"]:
                ax.grid(True, which="minor", alpha=0.12, linestyle=":")
                ax.minorticks_on()

        # Axes lines
        ax.axhline(0, color="#333", linewidth=1.2, zorder=2)
        ax.axvline(0, color="#333", linewidth=1.2, zorder=2)

        # Plot line
        xs = np.linspace(x_min, x_max, 200)
        ys = slope * xs + b
        ax.plot(xs, ys, color=palette[0 % len(palette)],
                linewidth=2.6, zorder=3)

        # Mark two points on the line for readability
        x_mark = [-2, 2]
        for x0 in x_mark:
            y0 = slope * x0 + b
            if y_min <= y0 <= y_max:
                ax.plot([x0], [y0], "o", color=palette[1 % len(palette)],
                        markersize=6, zorder=4,
                        markeredgecolor="#000", markeredgewidth=0.8)

        # Tick marks (integer)
        ax.set_xticks(range(int(x_min) + 1, int(x_max) + 1))
        ax.set_yticks(range(int(y_min) + 1, int(y_max) + 1))
        ax.tick_params(labelsize=fs - 1)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")

        ax.set_xlabel("x", fontsize=fs + 1)
        ax.set_ylabel("y", fontsize=fs + 1)
        ax.set_title("Line y = mx + b", fontsize=fs + 2)
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = LineSlopeInterceptQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[line_slope_intercept L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"line_slope_intercept_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[line_slope_intercept L{level} s{s}] saved | A={env._answer}")
