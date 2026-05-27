"""
Area Under Curve Estimation QA environment.

Shows a function curve on gridded axes with a shaded region between the
curve and x-axis, bounded by x = a and x = b (vertical dashed lines).
Grid squares are clearly visible for counting. The model estimates the
area in grid-square units.

Difficulty axes:
  - curve_type: constant/linear -> quadratic -> trig/piecewise
  - grid_size (1x1 -> 0.5x0.5) and sign_changes (curve dips below axis)
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

def _fmt_num(v):
    """Compact number formatter used in curve labels."""
    if isinstance(v, int):
        return str(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"

class AreaUnderCurveEstimationQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "area_under_curve_estimation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # Expanded curve-family pools for seed diversity: each level
        # offers 4+ visually distinct function families so 20 seeds
        # produce genuinely different graphs.
        if level <= 1:
            curves = ["constant", "constant_neg", "linear_flat",
                      "step"]
        elif level <= 3:
            curves = ["constant", "linear", "linear_neg", "abs_v",
                      "step"]
        elif level <= 5:
            curves = ["linear", "quadratic_up", "quadratic_down",
                      "abs_v", "cubic_mild", "sqrt_curve"]
        elif level <= 7:
            curves = ["quadratic_up", "quadratic_down", "sin_half",
                      "cos_half", "exp_decay", "cubic_bump",
                      "sqrt_curve"]
        else:
            curves = ["sin_full", "cos_shift", "piecewise_lin",
                      "piecewise_tri", "cubic_sign", "damped_sin",
                      "exp_grow"]
        # grid_size: L0 = 1.0, L9 = 0.5
        grid_size = 1.0 if level <= 4 else (0.5 if level >= 7 else 0.75)
        sign_changes = level >= 5
        return {
            "curves": curves,
            "grid_size": grid_size,
            "sign_changes": sign_changes,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1637)

        for _ in range(30):
            result = self._try_generate(sub_rng, level, cfg)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    def _build_func(self, curve_type: str, sub_rng: random.Random,
                    sign_changes: bool):
        """Return (f, label, a, b). Many distinct function families so
        20 seeds at a given level produce visually different graphs."""
        if curve_type == "constant":
            c = sub_rng.choice([1, 2, 3, 4])
            f = lambda x, _c=c: _c
            label = f"f(x) = {c}"
            a = sub_rng.choice([0, 1])
            b = a + sub_rng.choice([2, 3, 4])
        elif curve_type == "constant_neg":
            c = sub_rng.choice([-1, -2, -3])
            f = lambda x, _c=c: _c
            label = f"f(x) = {c}"
            a = sub_rng.choice([0, 1])
            b = a + sub_rng.choice([2, 3, 4])
        elif curve_type == "linear":
            m = sub_rng.choice([1, 2])
            bint = sub_rng.choice([0, 1, 2, 3])
            f = lambda x, _m=m, _b=bint: _m * x + _b
            sign = "+" if bint >= 0 else "-"
            label = f"f(x) = {m}x {sign} {abs(bint)}"
            a = sub_rng.choice([0, 1])
            b = a + sub_rng.choice([2, 3, 4])
        elif curve_type == "linear_neg":
            m = sub_rng.choice([-1, -2])
            bint = sub_rng.choice([3, 4, 5, 6])
            f = lambda x, _m=m, _b=bint: _m * x + _b
            label = f"f(x) = {m}x + {bint}"
            a = sub_rng.choice([0, 1])
            b = a + sub_rng.choice([2, 3])
        elif curve_type == "linear_flat":
            m = sub_rng.choice([0.5, -0.5])
            bint = sub_rng.choice([1, 2, 3])
            f = lambda x, _m=m, _b=bint: _m * x + _b
            label = f"f(x) = {m:g}x + {bint}"
            a = 0
            b = sub_rng.choice([3, 4])
        elif curve_type == "abs_v":
            # V-shape: f(x) = |x - k| + c
            k = sub_rng.choice([1.0, 2.0, 2.5, 3.0])
            c = sub_rng.choice([0, 1])
            f = lambda x, _k=k, _c=c: abs(x - _k) + _c
            label = f"f(x) = |x - {_fmt_num(k)}| + {c}"
            a = 0
            b = sub_rng.choice([4, 5])
        elif curve_type == "step":
            # stepped: 2 for first half, a different value for second
            cut = sub_rng.choice([2, 3])
            low = sub_rng.choice([1, 2])
            high = low + sub_rng.choice([1, 2])
            def f_step(x, _cut=cut, _lo=low, _hi=high):
                return _lo if x < _cut else _hi
            f = f_step
            label = f"step: {low} for x<{cut}, else {high}"
            a = 0
            b = cut + sub_rng.choice([1, 2])
        elif curve_type == "quadratic_up":
            # upward parabola f(x) = c*(x-h)^2 + k  (k >= 0)
            h = sub_rng.choice([1.5, 2.0, 2.5])
            cc = sub_rng.choice([0.25, 0.4, 0.5])
            k = sub_rng.choice([0, 1])
            f = lambda x, _h=h, _c=cc, _k=k: _c * (x - _h) ** 2 + _k
            label = f"f(x) = {cc:g}(x-{_fmt_num(h)})² + {k}"
            a = 0
            b = sub_rng.choice([3, 4, 5])
        elif curve_type == "quadratic_down":
            # downward parabola f(x) = k - c*(x-h)^2
            h = sub_rng.choice([1.5, 2.0, 2.5])
            cc = sub_rng.choice([0.25, 0.4, 0.5])
            k = sub_rng.choice([3, 4])
            f = lambda x, _h=h, _c=cc, _k=k: _k - _c * (x - _h) ** 2
            label = f"f(x) = {k} - {cc:g}(x-{_fmt_num(h)})²"
            a = 0
            b = sub_rng.choice([3, 4])
        # legacy alias
        elif curve_type == "quadratic":
            sign = 1 if not sign_changes else sub_rng.choice([-1, 1])
            if sign < 0:
                f = lambda x: -(3 - 0.3 * (x - 2) ** 2)
                label = "f(x) = -(3 - 0.3(x-2)²)"
            else:
                f = lambda x: 3 - 0.3 * (x - 2) ** 2
                label = "f(x) = 3 - 0.3(x-2)²"
            a = 0
            b = sub_rng.choice([3, 4])
        elif curve_type == "cubic_mild":
            # mild cubic rising: f(x) = 0.1 x^3 + c
            cc = sub_rng.choice([0, 1])
            f = lambda x, _c=cc: 0.1 * x ** 3 + _c
            label = f"f(x) = 0.1 x³ + {cc}"
            a = 0
            b = sub_rng.choice([3, 4])
        elif curve_type == "cubic_bump":
            # f(x) = -0.05 (x-2)^3 + 2
            h = sub_rng.choice([2.0, 2.5])
            f = lambda x, _h=h: -0.05 * (x - _h) ** 3 + 2
            label = f"f(x) = -0.05(x-{_fmt_num(h)})³ + 2"
            a = 0
            b = sub_rng.choice([4, 5])
        elif curve_type == "cubic_sign":
            # f(x) = 0.3 * (x - 2) * (x - 1) * (x - 3); crosses axis
            f = lambda x: 0.3 * (x - 2) * (x - 1) * (x - 3) + 1.5
            label = "f(x) = 0.3(x-1)(x-2)(x-3) + 1.5"
            a = 0
            b = 4
        elif curve_type == "sqrt_curve":
            # f(x) = sqrt(x) + c
            cc = sub_rng.choice([0.5, 1.0, 1.5])
            f = lambda x, _c=cc: math.sqrt(max(0.0, x)) + _c
            label = f"f(x) = √x + {cc:g}"
            a = 0
            b = sub_rng.choice([4, 5, 6])
        elif curve_type == "sin_half":
            # f(x) = A sin(x) on [0, pi]
            A = sub_rng.choice([1.5, 2.0, 2.5])
            f = lambda x, _A=A: _A * math.sin(x)
            label = f"f(x) = {A:g} sin(x)"
            a = 0
            b = math.pi
        elif curve_type == "cos_half":
            # f(x) = A cos(x) + B on [0, pi]
            A = sub_rng.choice([1.0, 1.5])
            B = sub_rng.choice([1.5, 2.0])
            f = lambda x, _A=A, _B=B: _A * math.cos(x) + _B
            label = f"f(x) = {A:g} cos(x) + {B:g}"
            a = 0
            b = math.pi
        elif curve_type == "sin_full":
            # full sine wave with sign changes: [0, 2pi]
            A = sub_rng.choice([1.5, 2.0])
            f = lambda x, _A=A: _A * math.sin(x)
            label = f"f(x) = {A:g} sin(x)"
            a = 0
            b = sub_rng.choice([math.pi * 1.5, 2 * math.pi])
        elif curve_type == "cos_shift":
            # shifted cosine with amplitude
            A = sub_rng.choice([1.5, 2.0])
            phi = sub_rng.choice([0.0, math.pi / 4, math.pi / 2])
            f = lambda x, _A=A, _p=phi: _A * math.cos(x - _p)
            label = f"f(x) = {A:g} cos(x - {_fmt_num(phi)})"
            a = 0
            b = sub_rng.choice([math.pi, math.pi * 1.5])
        elif curve_type == "damped_sin":
            # e^(-0.3 x) * sin(2x)
            A = sub_rng.choice([2.0, 2.5])
            f = lambda x, _A=A: _A * math.exp(-0.3 * x) * math.sin(2 * x)
            label = f"f(x) = {A:g} e^(-0.3 x) sin(2x)"
            a = 0
            b = sub_rng.choice([math.pi, math.pi * 1.2])
        # legacy alias
        elif curve_type == "trig":
            A = sub_rng.choice([1.5, 2.0])
            f = lambda x, _A=A: _A * math.sin(x)
            label = f"f(x) = {A:g} sin(x)"
            a = 0
            b = sub_rng.choice([math.pi, math.pi * 1.5]) if sign_changes else math.pi
        elif curve_type == "exp_decay":
            # A e^(-kx) + c
            A = sub_rng.choice([2.0, 3.0])
            k = sub_rng.choice([0.3, 0.5])
            cc = sub_rng.choice([0, 0.5])
            f = lambda x, _A=A, _k=k, _c=cc: _A * math.exp(-_k * x) + _c
            label = f"f(x) = {A:g} e^(-{k:g}x) + {cc:g}"
            a = 0
            b = sub_rng.choice([3, 4, 5])
        elif curve_type == "exp_grow":
            # A (1 - e^(-kx))
            A = sub_rng.choice([3.0, 4.0])
            k = sub_rng.choice([0.4, 0.6])
            f = lambda x, _A=A, _k=k: _A * (1 - math.exp(-_k * x))
            label = f"f(x) = {A:g}(1 - e^(-{k:g}x))"
            a = 0
            b = sub_rng.choice([3, 4, 5])
        elif curve_type == "piecewise_lin":
            # two linear pieces joined at x=2, 3
            cut = sub_rng.choice([2.0, 2.5, 3.0])
            slope_l = sub_rng.choice([1, 1.5, 2])
            slope_r = sub_rng.choice([-1, -1.5, -2])
            off = sub_rng.choice([0, 1])
            def f_pw(x, _cut=cut, _sl=slope_l, _sr=slope_r, _o=off):
                if x <= _cut:
                    return _sl * x + _o
                else:
                    k = _sl * _cut + _o - _sr * _cut
                    return _sr * x + k
            f = f_pw
            label = f"piecewise linear (peak at x={_fmt_num(cut)})"
            a = 0
            b = sub_rng.choice([4, 5])
        elif curve_type == "piecewise_tri":
            # triangular pulse: 0 for x<a1, rises to peak at a2, back to 0 at a3
            a1 = sub_rng.choice([0.5, 1.0])
            a2 = a1 + sub_rng.choice([1.0, 1.5])
            a3 = a2 + sub_rng.choice([1.0, 1.5])
            peak = sub_rng.choice([2.0, 2.5, 3.0])
            def f_tri(x, _a1=a1, _a2=a2, _a3=a3, _p=peak):
                if x <= _a1 or x >= _a3:
                    return 0.0
                if x <= _a2:
                    return _p * (x - _a1) / (_a2 - _a1)
                return _p * (_a3 - x) / (_a3 - _a2)
            f = f_tri
            label = "triangular pulse"
            a = 0
            b = a3 + sub_rng.choice([0.5, 1.0])
        else:  # fallback
            slope_l = sub_rng.choice([1, 2])
            slope_r = sub_rng.choice([-1, -2])
            off = sub_rng.choice([0, 1])
            def f_pw(x, _sl=slope_l, _sr=slope_r, _o=off):
                if x <= 2:
                    return _sl * x + _o
                else:
                    k = _sl * 2 + _o - _sr * 2
                    return _sr * x + k
            f = f_pw
            label = "piecewise linear"
            a = 0
            b = sub_rng.choice([4, 5])
        return f, label, a, b

    # ------------------------------------------------------------------ #
    def _try_generate(self, sub_rng: random.Random, level: int, cfg: Dict):
        curve_type = sub_rng.choice(cfg["curves"])
        f, label, a, b = self._build_func(curve_type, sub_rng, cfg["sign_changes"])

        # Compute numerical area (sum of |f(x)|)
        xs = np.linspace(a, b, 2000)
        ys = np.array([f(x) for x in xs])
        abs_area = float(np.trapz(np.abs(ys), xs))

        grid_size = cfg["grid_size"]
        # Area in grid-square units
        # one grid square has area grid_size * grid_size
        area_in_squares = abs_area / (grid_size * grid_size)
        answer_int = int(round(area_in_squares))

        if answer_int < 1 or answer_int > 60:
            return None

        # Build MCQ options
        correct = answer_int
        # distractors: near-correct integers
        distractors = set()
        tries = 0
        while len(distractors) < 3 and tries < 40:
            tries += 1
            offs = sub_rng.choice([-3, -2, -1, 1, 2, 3])
            cand = correct + offs
            if cand < 0 or cand == correct:
                continue
            distractors.add(cand)
        if len(distractors) < 3:
            return None

        options = sorted(list(distractors) + [correct])
        sub_rng.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = chr(ord("A") + correct_idx)
        opt_str = ", ".join(f"{chr(ord('A') + i)}) {v}" for i, v in enumerate(options))

        # Render
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 5.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        plot_xs = np.linspace(min(a, 0) - 0.5, max(b, 4) + 0.5, 400)
        plot_ys = np.array([f(x) for x in plot_xs])
        color = style["palette"][0]
        ax.plot(plot_xs, plot_ys, color=color, linewidth=2.2, label=label)

        # Shade region between curve and x-axis from a to b
        shade_xs = np.linspace(a, b, 300)
        shade_ys = np.array([f(x) for x in shade_xs])
        ax.fill_between(shade_xs, 0, shade_ys,
                        where=shade_ys >= 0, interpolate=True,
                        color=color, alpha=0.35, label="shaded region")
        if cfg["sign_changes"]:
            ax.fill_between(shade_xs, 0, shade_ys,
                            where=shade_ys < 0, interpolate=True,
                            color="#e67e22", alpha=0.35)

        # Vertical dashed lines at x=a and x=b
        ax.axvline(a, color="#333333", linewidth=1.2, linestyle="--")
        ax.axvline(b, color="#333333", linewidth=1.2, linestyle="--")

        # gridlines at grid_size
        xmin, xmax = min(a, 0) - 0.5, max(b, 4) + 0.5
        y_min = min(plot_ys.min(), -0.5) - 0.5
        y_max = max(plot_ys.max(), 0.5) + 0.5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(y_min, y_max)

        xticks = np.arange(math.floor(xmin), math.ceil(xmax) + grid_size, grid_size)
        yticks = np.arange(math.floor(y_min), math.ceil(y_max) + grid_size, grid_size)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(True, which="major", color="#999999", linewidth=0.6, alpha=0.8)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("f(x)", fontsize=11)
        ax.set_title(f"Shaded area from x = {round(a, 2)} to x = {round(b, 2)}",
                      fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)

        gs_str = f"{grid_size:g}×{grid_size:g}"
        a_r, b_r = round(a, 2), round(b, 2)
        _TEMPLATES = [
            f"The graph shows a function f(x) with a shaded region between the curve and the x-axis, bounded by the dashed lines x = {a_r} and x = {b_r}. Each grid square is {gs_str}. Approximately how many grid squares does the shaded area cover (counting areas below the x-axis as positive)? {opt_str}. Answer with a single letter.",
            f"The figure depicts f(x) with a shaded area between the curve and the x-axis between x = {a_r} and x = {b_r}. Grid cells measure {gs_str}. Estimate the number of grid squares covered by the shaded region (below-axis counts as positive). {opt_str}. Answer: a single letter.",
            f"A function f(x) is plotted; the shaded region lies between the curve and the x-axis from x = {a_r} to x = {b_r}. Grid squares are {gs_str}. Roughly how many grid squares does the shaded area occupy (count below-axis positively)? {opt_str}. Answer with one letter.",
            f"Estimate the grid-square count of the shaded area between f(x) and the x-axis on [{a_r}, {b_r}] shown in the plot. Each cell is {gs_str}. (Sub-axis area counted positive.) {opt_str}. Respond with a single letter.",
            f"The plot shows f(x); a region between the curve and the x-axis is shaded from x = {a_r} to x = {b_r}. Each grid box is {gs_str}. How many grid squares does the shaded region approximately cover? (Areas below x-axis are positive.) {opt_str}. Single letter answer.",
            f"Using the gridded plot of f(x), estimate the number of grid squares in the shaded region on [{a_r}, {b_r}]. Grid squares are {gs_str}. Below-axis areas count positively. {opt_str}. Reply with a single letter.",
            f"Look at the graph of f(x) with its shaded region from x = {a_r} to x = {b_r}. Each small square is {gs_str}. Approximately how many squares are within the shaded area (sub-axis regions counted as positive)? {opt_str}. Answer: one letter.",
            f"From the graph, the shaded region between f(x) and the x-axis spans x ∈ [{a_r}, {b_r}]. Grid cells are {gs_str}. Estimate the grid-square count of the shaded area (below-axis positive). {opt_str}. Provide a single letter.",
            f"The curve f(x) is plotted with a shaded region delimited by x = {a_r} and x = {b_r}. Each grid square measures {gs_str}. Roughly how many squares cover the shaded area? (below-axis counted as positive.) {opt_str}. One letter.",
            f"Based on the graph, the shaded area under/above the curve f(x) between x = {a_r} and x = {b_r} is visible. With {gs_str} grid squares, how many squares does the shaded region approximately span (sub-axis positive)? {opt_str}. Answer with a letter.",
            f"Shaded region between curve f(x) and x-axis is shown from x = {a_r} to x = {b_r}. Grid = {gs_str}. Estimate the number of grid squares in the shaded area; count below-axis parts as positive. {opt_str}. Answer with one letter.",
            f"In the plotted f(x), the shaded area (bounded by x = {a_r} and x = {b_r}) is marked. Grid cell = {gs_str}. Approximately how many grid cells are in the shaded region? Below-axis counts positively. {opt_str}. Single letter.",
            f"The curve f(x) is graphed. Between x = {a_r} and x = {b_r} there is a shaded area between the curve and the x-axis. With {gs_str} grid squares, estimate the covered square count (sub-axis as positive). {opt_str}. One-letter answer.",
            f"A shaded region lies between f(x) and the x-axis on [{a_r}, {b_r}] in the plot. Each grid square = {gs_str}. Give a rough count of grid squares in the shaded region (below-axis positive). {opt_str}. Answer with one letter.",
            f"Given the graph of f(x), the shaded area from x = {a_r} to x = {b_r} is highlighted. Grid squares are {gs_str}. Estimate roughly how many grid squares lie within the shaded area (sub-axis as positive). {opt_str}. Reply with a single letter.",
            f"The figure highlights a shaded region between f(x) and the x-axis on [{a_r}, {b_r}]. Grid square size: {gs_str}. Approximately how many grid squares does the shaded region cover (below-axis regions counted as positive)? {opt_str}. Provide one letter.",
        ]
        sidx = (self.seed or 0) % 16
        question = _TEMPLATES[sidx]
        answer = correct_letter

        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question, answer, img
