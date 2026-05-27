"""Inverse Function QA — diversity + difficulty redesign 2026-04-16.

Round-2 fix: Grade D difficulty (too easy) and limited diversity. Expanded
function families (linear, sqrt, cubic, cubic_root, log, piecewise_2piece,
piecewise_3piece) and operation types (inverse_value, inverse_slope, intersect,
compose_inverse, double_compose, inverse_at_zero, inverse_derivative_like).

  L0-L1  simple: given f(x) linear, find f^{-1}(y). Axis labels shown.
  L2     find slope of inverse (linear only).
  L3     intersection of f(x) with y=x.
  L4-L5  inverse of sqrt / cubic / cubic_root; read-off.
  L6     piecewise (2 pieces) – find f^{-1}(y).
  L7     compose_inverse: f^{-1}(f^{-1}(y)).
  L8     double_compose: f(f^{-1}(a) + f^{-1}(b)).
  L9     piecewise (3 pieces) – compose or harder operation.

2026-05-03 extension (M21 / IP-T1): added `hyperbola` family (y=k/x) and
qtype `find_k_from_point` mirroring reference samples
(Q7 idx 3952 "y=m/x passes (2,4) → m=8", Q13 idx 4017 "y=k/x passes
(1,-3) → k=-3", Q19 idx 3969 "y=m/x passes (1,2) → m=2"). When this
qtype runs, a hyperbola is rendered with one labeled point P; model
must compute k = x_P * y_P. Verifier accepts `\\boxed{X}`.
"""
import random, math
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_COLOURS_CURVE = ["#2d5ba9", "#b91d1d", "#1f7a3a", "#c07214",
                   "#702563", "#0b6e8f", "#a0522d", "#264653"]
# Titles MUST NOT reveal the function form — only the curve shape in the image
# may be used to answer. (Text-leakage fix 2026-04-17.)
_TITLE_POOL = [
    "Graph of f(x)",
    "Plot of f(x)",
    "Function Graph",
    "Function f(x)",
    "f(x) as shown",
]

class InverseFunctionQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "inverse_function"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        # Monotonic schedule: inverse_value on progressively harder families,
        # with compose / intersection / piecewise introduced ONLY after
        # inverse_value has been mastered on non-linear families.
        # Previous: L3 jumped to intersection (0.35 drop), L9 used
        # piecewise3 + compose_inverse (0.05 — effectively impossible).
        if level <= 1:
            return {"families": ["linear"], "qtype": "inverse_value",
                    "show_ticks": True, "show_grid": True, "show_yx": True,
                    "allow_frac": False}
        if level == 2:
            return {"families": ["linear"], "qtype": "inverse_slope",
                    "show_ticks": True, "show_grid": True, "show_yx": True,
                    "allow_frac": False}
        if level == 3:
            # Stay on inverse_value with linear+sqrt so L3 tracks the curve.
            return {"families": ["linear", "sqrt"], "qtype": "inverse_value",
                    "show_ticks": True, "show_grid": True, "show_yx": True,
                    "allow_frac": False}
        if level == 4:
            return {"families": ["sqrt", "cubic_root"], "qtype": "inverse_value",
                    "show_ticks": True, "show_grid": True, "show_yx": True,
                    "allow_frac": True}
        if level == 5:
            return {"families": ["sqrt", "cubic_root", "cubic"],
                    "qtype": "inverse_value",
                    "show_ticks": True, "show_grid": True, "show_yx": False,
                    "allow_frac": True}
        if level == 6:
            # intersection now lands at L6 (was L3 — too hard there)
            return {"families": ["linear", "sqrt"], "qtype": "intersection",
                    "show_ticks": True, "show_grid": True, "show_yx": True,
                    "allow_frac": True}
        if level == 7:
            return {"families": ["piecewise2"], "qtype": "inverse_value",
                    "show_ticks": True, "show_grid": False, "show_yx": False,
                    "allow_frac": True}
        if level == 8:
            return {"families": ["linear", "sqrt", "cubic_root"],
                    "qtype": "compose_inverse",
                    "show_ticks": True, "show_grid": False, "show_yx": False,
                    "allow_frac": True}
        # L9: keep compose_inverse but on simpler piecewise2, NOT piecewise3.
        # Drop double_compose entirely — answer rate was 0.05 before.
        return {"families": ["piecewise2", "sqrt"], "qtype": "compose_inverse",
                "show_ticks": True, "show_grid": False, "show_yx": False,
                "allow_frac": True}

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 571)

        # M21 / IP-T1 hyperbola mode injection (reference IPF). Roughly 25%
        # of calls (or whenever caller passes question_type='find_k_from_point')
        # we route to the hyperbola find-k generator. This keeps existing
        # inverse-of-function behaviour while ADDING the missing
        # y=k/x variety reference actually tests.
        force_hyperbola = parameter.get("question_type") == "find_k_from_point"
        if force_hyperbola or rng.random() < 0.25:
            r = self._try_gen_hyperbola_find_k(rng, level)
            if r is not None:
                return r
            if force_hyperbola:
                return None  # caller asked specifically; don't silently fallback

        for _ in range(25):
            r = self._try_gen(rng, cfg, level)
            if r is not None:
                return r
        return None

    # ------------------------------------------------------------------ #
    # M21 / IP-T1 hyperbola find-k generator
    # ------------------------------------------------------------------ #
    def _try_gen_hyperbola_find_k(self, rng, level):
        """Render hyperbola y=k/x with point P=(a,b) labeled; ask for k.

        Anchors (design notes IPF):
          Q7 idx 3952: y=m/x passes (2,4) → m=8
          Q13 idx 4017: y=k/x passes (1,-3) → k=-3
          Q19 idx 3969: y=m/x passes (1,2) → m=2
        Higher levels use larger |k| or two-point asks.
        """
        # Pick k. At low levels small integer; at high levels include negatives.
        if level <= 2:
            k_choices = [2, 3, 4, 6, 8]
        elif level <= 5:
            k_choices = [-12, -8, -6, -4, -3, 2, 3, 4, 6, 8, 12]
        else:
            k_choices = [-24, -18, -15, -12, -10, -8, -6, 6, 8, 10, 12, 15, 18, 24]
        k = rng.choice(k_choices)

        # Pick a point P=(a, b) with b = k/a, a integer, b a nice rational.
        # Prefer a where k/a is integer (factor of k).
        a_candidates = [d for d in [1, 2, 3, 4, 6] if k % d == 0 and d != 0]
        if not a_candidates:
            a_candidates = [1, 2, 3]
        a = rng.choice(a_candidates)
        if k > 0 and rng.random() < 0.3:
            a = -a  # second-quadrant point if k<0 OR third-quadrant if k>0
        if k < 0 and rng.random() < 0.5:
            a = -a
        b = k / a
        # Clean up b to int when possible
        if abs(b - round(b)) < 1e-6:
            b = int(round(b))

        # Render hyperbola
        image = self._render_hyperbola(k, a, b, rng)

        ans_str = str(int(k)) if k == int(k) else self._fmt_num(k)
        # Three phrasings (verbatim style from reference).
        phrasings = [
            f"As shown in the figure, the graph of the inverse proportion function y=k/x passes through the point P({a}, {b}). What is the value of k?",
            f"The hyperbola y=k/x is shown. Point P({a}, {b}) lies on the curve. Find k.",
            f"Looking at the figure, the inverse proportion function y=k/x passes through point ({a}, {b}). Determine the value of k.",
        ]
        q = rng.choice(phrasings)
        return q, ans_str, image

    def _render_hyperbola(self, k, a, b, rng):
        """Render a hyperbola y=k/x with point P=(a,b) labeled."""
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        self._apply_style(fig, ax, style)
        palette = style["palette"]

        # Plot two branches.
        eps = 0.1
        x_max = max(8, abs(a) + 3)
        x_pos = np.linspace(eps, x_max, 200)
        x_neg = np.linspace(-x_max, -eps, 200)
        ax.plot(x_pos, k / x_pos, color=palette[0], linewidth=2,
                label=f"y = k/x")
        ax.plot(x_neg, k / x_neg, color=palette[0], linewidth=2)

        # Mark the point P
        ax.plot(a, b, marker='o', color=palette[3 % len(palette)],
                markersize=10, zorder=5)
        ax.annotate(f"P({a}, {b})", xy=(a, b),
                    xytext=(a + 0.4, b + 0.4),
                    fontsize=12, fontweight='bold')

        ax.axhline(y=0, color='gray', linewidth=0.6)
        ax.axvline(x=0, color='gray', linewidth=0.6)
        # Symmetric range so both branches visible.
        y_max = max(8, abs(b) + 3)
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title("Inverse proportion function", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc=style["legend_loc"])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _try_gen(self, rng, cfg, level):
        func_type = rng.choice(cfg["families"])
        qtype = cfg["qtype"]

        family = self._make_family(func_type, cfg, rng)
        if family is None:
            return None

        f, f_inv, x_range, f_label = family[0], family[1], family[2], family[3]

        # Build Q&A
        try:
            if qtype == "inverse_value":
                y_val = self._choose_y_in_range(f, x_range, rng)
                if y_val is None:
                    return None
                result = float(f_inv(y_val))
                if not np.isfinite(result):
                    return None
                ans_str = self._fmt_num(result)
                q = self._pick_inverse_value_phrasing(y_val, rng)
            elif qtype == "inverse_slope":
                # Only for linear
                # recover by querying f(1) - f(0)
                slope_val = float(f(1)) - float(f(0))
                if abs(slope_val) < 1e-9:
                    return None
                inv_slope = 1.0 / slope_val
                ans_str = self._fmt_num(inv_slope)
                q = ("The graph shows a linear function f(x). "
                     "What is the SLOPE of its inverse function f^{-1}(x)? "
                     "Round to 2 decimal places.")
            elif qtype == "intersection":
                # Solve f(x) = x
                xs = np.linspace(x_range[0], x_range[1], 4000)
                ys = f(xs)
                diffs = np.asarray(ys) - xs
                sign_changes = np.where(np.sign(diffs[:-1]) * np.sign(diffs[1:]) <= 0)[0]
                if len(sign_changes) == 0:
                    return None
                # pick a sensible non-trivial root (not the boundary)
                best = None
                for ix in sign_changes:
                    x_int = float(xs[ix])
                    # Avoid f(x)=x=0 trivial — allow if in-range
                    if abs(x_int) > 0.01:
                        best = x_int
                        break
                if best is None:
                    best = float(xs[sign_changes[0]])
                ans_str = self._fmt_num(best)
                q = rng.choice([
                    "At what value of x does f(x) = x? Round to 2 decimals.",
                    "Find the non-trivial x for which the graphed f satisfies f(x) = x. Round to 2 decimals.",
                    "The graph shows f(x). Determine an x such that f(x) = x (not necessarily zero). Round to 2 decimals.",
                ])
            elif qtype == "compose_inverse":
                y_val = self._choose_y_in_range(f, x_range, rng)
                if y_val is None:
                    return None
                step1 = float(f_inv(y_val))
                if not np.isfinite(step1):
                    return None
                step2 = float(f_inv(step1))
                if not np.isfinite(step2):
                    return None
                ans_str = self._fmt_num(step2)
                q = (f"From the graph of f(x) shown, compute "
                     f"f^{{-1}}(f^{{-1}}({self._fmt_num(y_val)})). Round to 2 decimals.")
            elif qtype == "double_compose":
                a_val = rng.choice([1, 2, 3, 4])
                b_val = rng.choice([v for v in range(-3, 5) if v != a_val])
                inv_a = float(f_inv(a_val))
                inv_b = float(f_inv(b_val))
                if not (np.isfinite(inv_a) and np.isfinite(inv_b)):
                    return None
                # f expects scalar
                try:
                    result = float(f(inv_a + inv_b))
                except Exception:
                    arr = f(np.array([inv_a + inv_b]))
                    result = float(np.asarray(arr).reshape[0])
                if not np.isfinite(result):
                    return None
                ans_str = self._fmt_num(result)
                q = (f"Using the graph of f(x) shown, compute "
                     f"f(f^{{-1}}({a_val}) + f^{{-1}}({b_val})). Round to 2 decimals.")
            else:
                return None
        except Exception:
            return None

        image = self._render(f, x_range, f_label, cfg, rng)
        return q, ans_str, image

    # ------------------------------------------------------------------ #
    # Family factory
    # ------------------------------------------------------------------ #

    def _make_family(self, func_type, cfg, rng):
        """Return (f, f_inv, x_range, label). For linear we append slope last."""
        if func_type == "linear":
            if cfg["allow_frac"]:
                a = rng.choice([0.5, 1.5, 2, 2.5, 3, -0.5, -1.5, -2, -2.5])
            else:
                a = rng.choice([2, 3, -1, -2, 4])
            b = rng.randint(-5, 5)
            f = lambda x, _a=a, _b=b: _a * x + _b
            f_inv = lambda y, _a=a, _b=b: (y - _b) / _a
            x_range = (-6, 6)
            label = f"f(x) = {a}x + {b}" if b >= 0 else f"f(x) = {a}x - {abs(b)}"
            return (f, f_inv, x_range, label, a)
        if func_type == "sqrt":
            a = rng.choice([1, 2, 3])
            f = lambda x, _a=a: _a * np.sqrt(np.maximum(x, 0))
            f_inv = lambda y, _a=a: (y / _a) ** 2
            x_range = (0, 10)
            label = f"f(x) = {a}\u221ax" if a > 1 else "f(x) = \u221ax"
            return (f, f_inv, x_range, label)
        if func_type == "cubic":
            # f(x) = a*x^3
            a = rng.choice([1, 0.5, 2])
            f = lambda x, _a=a: _a * (x ** 3)
            f_inv = lambda y, _a=a: np.sign(y / _a) * np.abs(y / _a) ** (1/3)
            x_range = (-3, 3)
            label = f"f(x) = {a}x\u00b3"
            return (f, f_inv, x_range, label)
        if func_type == "cubic_root":
            f = lambda x: np.cbrt(x)
            f_inv = lambda y: y ** 3
            x_range = (-8, 8)
            label = "f(x) = \u221bx"
            return (f, f_inv, x_range, label)
        if func_type == "piecewise2":
            slopes = rng.sample([1, 2, 3], 2)
            s1, s2 = slopes
            b_val = rng.randint(1, 4)
            f = lambda x, _s1=s1, _s2=s2, _b=b_val: np.where(
                x < 0, _s1 * x + _b, _s2 * x + _b)
            def f_inv(y, _s1=s1, _s2=s2, _b=b_val):
                if y < _b:
                    return (y - _b) / _s1
                return (y - _b) / _s2
            x_range = (-5, 5)
            label = f"f(x) = {s1}x+{b_val} (x<0), {s2}x+{b_val} (x>=0)"
            return (f, f_inv, x_range, label)
        if func_type == "piecewise3":
            # 3-piece monotone: left slope, middle slope, right slope, all >0
            slopes = rng.sample([1, 2, 3], 3)
            s1, s2, s3 = slopes
            b_val = rng.randint(0, 3)
            # breakpoints at -2 and 2; make continuous
            def f(x, _s1=s1, _s2=s2, _s3=s3, _b=b_val):
                arr = np.asarray(x, dtype=float)
                y = np.zeros_like(arr)
                mid_val_at_neg2 = _b + _s2 * (-2)
                # Continuity at x = -2: s1 * -2 + c1 = s2 * -2 + b => c1 = -2(s2-s1) + b
                c1 = -2 * (_s2 - _s1) + _b
                # Continuity at x = 2: s2*2 + b = s3*2 + c3 => c3 = 2(s2 - s3) + b
                c3 = 2 * (_s2 - _s3) + _b
                y = np.where(arr < -2, _s1 * arr + c1, y)
                y = np.where((arr >= -2) & (arr <= 2), _s2 * arr + _b, y)
                y = np.where(arr > 2, _s3 * arr + c3, y)
                return y
            # Build inverse by segments
            def f_inv(y_in, _s1=s1, _s2=s2, _s3=s3, _b=b_val):
                c1 = -2 * (_s2 - _s1) + _b
                c3 = 2 * (_s2 - _s3) + _b
                y = float(y_in)
                # boundary f-values at -2 and 2: y_left = s2*-2+b, y_right = s2*2+b
                y_left = _s2 *  + _b
                y_right = _s2 * 2 + _b
                if y < y_left:
                    return (y - c1) / _s1
                elif y <= y_right:
                    return (y - _b) / _s2
                else:
                    return (y - c3) / _s3
            x_range = (-5, 5)
            label = f"piecewise: {s1}|{s2}|{s3} slopes, b={b_val}"
            return (f, f_inv, x_range, label)
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _choose_y_in_range(self, f, x_range, rng):
        """Pick a y-value that has a valid preimage within range."""
        xs = np.linspace(x_range[0], x_range[1], 400)
        try:
            ys = f(xs)
        except Exception:
            return None
        ys = np.asarray(ys)
        y_min = float(np.nanmin(ys))
        y_max = float(np.nanmax(ys))
        # Pick integer in range
        lo = math.ceil(y_min + 0.5)
        hi = math.floor(y_max - 0.5)
        if hi <= lo:
            return None
        for _ in range(10):
            y_try = rng.randint(lo, hi)
            if y_try != 0:
                return y_try
        return lo

    @staticmethod
    def _fmt_num(x):
        val = round(float(x), 2)
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        return f"{val:.2f}"

    def _pick_inverse_value_phrasing(self, y_val, rng):
        return rng.choice([
            f"Given the graph of f(x), compute f^{{-1}}({self._fmt_num(y_val)}). Round to 2 decimals.",
            f"From the graph of f(x), find the value of f^{{-1}}({self._fmt_num(y_val)}). Round to 2 decimals.",
            f"What x satisfies f(x) = {self._fmt_num(y_val)} in the graph? Round to 2 decimals.",
            f"The graph shows f(x). Determine f^{{-1}}({self._fmt_num(y_val)}). Round to 2 decimals.",
        ])

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, f, x_range, f_label, cfg, rng) -> Image.Image:
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        self._apply_style(fig, ax, style)
        palette = style["palette"]
        cc = rng.choice(_COLOURS_CURVE)

        x = np.linspace(x_range[0], x_range[1], 500)
        y = f(x)
        lw = 2.0 + rng.uniform(-0.4, 1.0)
        linestyle = rng.choice(["-", "-", "--"])
        # Legend label intentionally generic — never reveal analytic form of f.
        ax.plot(x, y, color=cc, linewidth=lw, linestyle=linestyle, label="f(x)")

        if cfg["show_yx"]:
            ax.plot([-10, 10], [-10, 10], '--', color='#bdc3c7', linewidth=1,
                    alpha=0.5, label="y = x")

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
        y_vals = y[np.isfinite(y)] if hasattr(y, '__iter__') else None
        if y_vals is not None and len(y_vals) > 0:
            ax.set_ylim(float(np.nanmin(y_vals)) - 1, float(np.nanmax(y_vals)) + 1)

        if cfg["show_grid"]:
            ax.grid(True, alpha=rng.uniform(0.18, 0.35),
                    linestyle=rng.choice(["--", "-.", ":"]))

        if not cfg["show_ticks"]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)

        ax.legend(fontsize=11, loc=style["legend_loc"])
        ax.set_title(rng.choice(_TITLE_POOL), fontsize=13, fontweight="bold")

        return self.fig_to_pil(fig, dpi=style["dpi"])
