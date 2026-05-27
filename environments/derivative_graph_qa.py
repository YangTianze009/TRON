"""
Derivative Graph QA environment.

Shows the graph of f(x) and asks about properties of f'(x), or shows
f'(x) and asks about properties of f(x).  Covers increasing/decreasing
intervals, local extrema, inflection points, and sign of derivative.
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

class DerivativeGraphQA(StandaloneVisualEnv):
    ENV_NAME = "derivative_graph"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    QUESTION_TYPES = [
        "increasing_interval",
        "decreasing_interval",
        "local_max",
        "local_min",
        "inflection_point",
        "sign_of_derivative",
    ]

    # ------------------------------------------------------------------ #
    # Entry
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"degree_range": (2, 2), "qtypes": ["sign_of_derivative", "local_max", "local_min"]}
        if level == 1:
            return {"degree_range": (2, 2), "qtypes": ["sign_of_derivative", "local_max", "local_min"]}
        if level == 2:
            return {"degree_range": (2, 3), "qtypes": ["local_max", "local_min", "increasing_interval"]}
        if level == 3:
            return {"degree_range": (2, 3), "qtypes": ["increasing_interval", "decreasing_interval"]}
        if level == 4:
            return {"degree_range": (3, 3), "qtypes": ["increasing_interval", "decreasing_interval", "local_max"]}
        if level == 5:
            return {"degree_range": (3, 4), "qtypes": ["local_max", "local_min", "inflection_point"]}
        if level == 6:
            return {"degree_range": (3, 4), "qtypes": ["inflection_point", "increasing_interval"]}
        if level == 7:
            return {"degree_range": (3, 4), "qtypes": ["inflection_point", "decreasing_interval"]}
        if level == 8:
            return {"degree_range": (4, 4), "qtypes": ["inflection_point", "increasing_interval"]}
        return {"degree_range": (4, 4), "qtypes": ["inflection_point", "increasing_interval", "decreasing_interval"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        self._level = level
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 704)
        question_type = parameter.get("question_type")
        if question_type is None:
            question_type = sub_rng.choice(cfg["qtypes"])

        for _ in range(30):
            result = self._try_generate(question_type, cfg)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Function generation (polynomial up to degree 4)
    # ------------------------------------------------------------------ #

    def _gen_poly(self, cfg=None) -> dict:
        """Generate polynomial coefficients. Degree 2-4."""
        lo, hi = (cfg or {}).get("degree_range", (2, 4))
        degree = self._rng.randint(lo, hi)
        if degree == 2:
            a = self._rng.choice([-2, -1, 1, 2])
            b = self._rng.randint(-4, 4)
            c = self._rng.randint(-5, 5)
            return {"coeffs": [a, b, c], "degree": 2}
        elif degree == 3:
            a = self._rng.choice([-1, 1])
            b = self._rng.randint(-3, 3)
            c = self._rng.randint(-5, 5)
            d = self._rng.randint(-5, 5)
            return {"coeffs": [a, b, c, d], "degree": 3}
        else:  # degree 4
            a = self._rng.choice([-1, 1]) * self._rng.choice([0.25, 0.5, 1])
            b = 0
            c = self._rng.choice([-3, -2, -1, 1, 2, 3])
            d = 0
            e = self._rng.randint(-3, 3)
            return {"coeffs": [a, b, c, d, e], "degree": 4}

    @staticmethod
    def _eval_poly(coeffs: list, x):
        """Evaluate polynomial with coefficients [highest, ..., constant]."""
        result = np.zeros_like(x, dtype=float) if hasattr(x, '__len__') else 0.0
        for c in coeffs:
            result = result * x + c
        return result

    @staticmethod
    def _deriv_coeffs(coeffs: list) -> list:
        """Compute derivative coefficients."""
        n = len(coeffs) - 1
        return [coeffs[i] * (n - i) for i in range(n)]

    @staticmethod
    def _second_deriv_coeffs(coeffs: list) -> list:
        d1 = DerivativeGraphQA._deriv_coeffs(coeffs)
        return DerivativeGraphQA._deriv_coeffs(d1)

    # ------------------------------------------------------------------ #
    # Critical points (real roots of derivative)
    # ------------------------------------------------------------------ #

    def _find_real_roots(self, coeffs: list, x_range: Tuple[float, float]) -> List[float]:
        """Find real roots of polynomial in the given range using numpy."""
        if len(coeffs) <= 1:
            return []
        roots = np.roots(coeffs)
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8:
                rv = float(r.real)
                if x_range[0] - 0.01 <= rv <= x_range[1] + 0.01:
                    real_roots.append(round(rv, 2))
        real_roots.sort()
        return real_roots

    # ------------------------------------------------------------------ #
    # Answer computation
    # ------------------------------------------------------------------ #

    def _compute_answer(
        self, poly: dict, question_type: str, x_range: Tuple[float, float],
        show_derivative: bool
    ) -> Optional[Tuple[str, str]]:
        coeffs = poly["coeffs"]
        d_coeffs = self._deriv_coeffs(coeffs)
        d2_coeffs = self._second_deriv_coeffs(coeffs)

        # Critical points of f (roots of f')
        crits = self._find_real_roots(d_coeffs, x_range)
        # Inflection points (roots of f'')
        inflections = self._find_real_roots(d2_coeffs, x_range)

        if show_derivative:
            prefix = "The graph shows f'(x). "
        else:
            prefix = "The graph shows f(x). "

        if question_type == "increasing_interval":
            # f increasing where f' > 0
            if not crits:
                # Check sign of f'
                test_val = self._eval_poly(d_coeffs, np.array([0.0]))[0] if hasattr(self._eval_poly(d_coeffs, np.array([0.0])), '__len__') else self._eval_poly(d_coeffs, 0.0)
                if test_val > 0:
                    ans = f"({x_range[0]}, {x_range[1]})"
                else:
                    ans = "none"
            else:
                intervals = []
                test_points = [x_range[0] - 0.1] + crits + [x_range[1] + 0.1]
                for i in range(len(test_points) - 1):
                    mid = (test_points[i] + test_points[i + 1]) / 2
                    val = float(self._eval_poly(d_coeffs, mid))
                    if val > 0:
                        lo = max(test_points[i], x_range[0])
                        hi = min(test_points[i + 1], x_range[1])
                        lo_r = round(lo, 2)
                        hi_r = round(hi, 2)
                        intervals.append(f"({lo_r}, {hi_r})")
                ans = " and ".join(intervals) if intervals else "none"
            question = prefix + "On which interval(s) is f(x) increasing? Use interval notation."
            return question, ans

        if question_type == "decreasing_interval":
            if not crits:
                test_val = float(self._eval_poly(d_coeffs, 0.0))
                if test_val < 0:
                    ans = f"({x_range[0]}, {x_range[1]})"
                else:
                    ans = "none"
            else:
                intervals = []
                test_points = [x_range[0] - 0.1] + crits + [x_range[1] + 0.1]
                for i in range(len(test_points) - 1):
                    mid = (test_points[i] + test_points[i + 1]) / 2
                    val = float(self._eval_poly(d_coeffs, mid))
                    if val < 0:
                        lo = max(test_points[i], x_range[0])
                        hi = min(test_points[i + 1], x_range[1])
                        lo_r = round(lo, 2)
                        hi_r = round(hi, 2)
                        intervals.append(f"({lo_r}, {hi_r})")
                ans = " and ".join(intervals) if intervals else "none"
            question = prefix + "On which interval(s) is f(x) decreasing? Use interval notation."
            return question, ans

        if question_type == "local_max":
            # f has local max where f' changes from + to -
            maxima = []
            if len(crits) >= 1:
                test_points = [x_range[0] - 0.5] + crits + [x_range[1] + 0.5]
                for i in range(1, len(test_points) - 1):
                    left_mid = (test_points[i - 1] + test_points[i]) / 2
                    right_mid = (test_points[i] + test_points[i + 1]) / 2
                    left_val = float(self._eval_poly(d_coeffs, left_mid))
                    right_val = float(self._eval_poly(d_coeffs, right_mid))
                    if left_val > 0 and right_val < 0:
                        xc = test_points[i]
                        yc = round(float(self._eval_poly(coeffs, xc)), 2)
                        maxima.append((round(xc, 2), yc))
            if not maxima:
                question = prefix + "At what x-value does f(x) have a local maximum?"
                return question, "none"
            if len(maxima) == 1:
                question = prefix + "At what x-value does f(x) have a local maximum?"
                return question, str(maxima[0][0])
            question = prefix + "How many local maxima does f(x) have?"
            return question, str(len(maxima))

        if question_type == "local_min":
            minima = []
            if len(crits) >= 1:
                test_points = [x_range[0] - 0.5] + crits + [x_range[1] + 0.5]
                for i in range(1, len(test_points) - 1):
                    left_mid = (test_points[i - 1] + test_points[i]) / 2
                    right_mid = (test_points[i] + test_points[i + 1]) / 2
                    left_val = float(self._eval_poly(d_coeffs, left_mid))
                    right_val = float(self._eval_poly(d_coeffs, right_mid))
                    if left_val < 0 and right_val > 0:
                        xc = test_points[i]
                        yc = round(float(self._eval_poly(coeffs, xc)), 2)
                        minima.append((round(xc, 2), yc))
            if not minima:
                question = prefix + "At what x-value does f(x) have a local minimum?"
                return question, "none"
            if len(minima) == 1:
                question = prefix + "At what x-value does f(x) have a local minimum?"
                return question, str(minima[0][0])
            question = prefix + "How many local minima does f(x) have?"
            return question, str(len(minima))

        if question_type == "inflection_point":
            if not inflections:
                question = prefix + "How many inflection points does f(x) have?"
                return question, "0"
            # Verify actual sign change in f''
            true_inflections = []
            for ip in inflections:
                left = float(self._eval_poly(d2_coeffs, ip - 0.1))
                right = float(self._eval_poly(d2_coeffs, ip + 0.1))
                if left * right < 0:
                    true_inflections.append(round(ip, 2))
            if len(true_inflections) == 0:
                question = prefix + "How many inflection points does f(x) have?"
                return question, "0"
            if len(true_inflections) == 1:
                question = prefix + "At what x-value does f(x) have an inflection point?"
                return question, str(true_inflections[0])
            question = prefix + "How many inflection points does f(x) have?"
            return question, str(len(true_inflections))

        if question_type == "sign_of_derivative":
            xv = self._rng.choice([-3, -2, -1, 0, 1, 2, 3])
            # Make sure xv is in range
            xv = max(x_range[0] + 0.5, min(x_range[1] - 0.5, xv))
            xv = round(xv, 1)
            dv = float(self._eval_poly(d_coeffs, xv))
            if abs(dv) < 0.05:
                return None
            sign = "positive" if dv > 0 else "negative"
            self._query_x = xv
            question = prefix + f"Is f'({xv}) positive or negative?"
            return question, sign

        return None

    # ------------------------------------------------------------------ #
    # Expression string
    # ------------------------------------------------------------------ #

    @staticmethod
    def _poly_expr(coeffs: list, var="x") -> str:
        n = len(coeffs) - 1
        parts = []
        for i, c in enumerate(coeffs):
            power = n - i
            if abs(c) < 1e-12:
                continue
            if power == 0:
                term = str(round(c, 2))
            elif power == 1:
                if abs(c) == 1:
                    term = f"{var}" if c > 0 else f"-{var}"
                else:
                    term = f"{round(c, 2)}{var}"
            else:
                if abs(c) == 1:
                    term = f"{var}^{power}" if c > 0 else f"-{var}^{power}"
                else:
                    term = f"{round(c, 2)}{var}^{power}"
            if parts and c > 0:
                parts.append(f"+ {term}")
            else:
                parts.append(term)
        return " ".join(parts) if parts else "0"

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def _try_generate(self, question_type: str, cfg=None) -> Optional[Tuple[str, str, Image.Image]]:
        self._query_x = None
        poly = self._gen_poly(cfg)
        coeffs = poly["coeffs"]
        d_coeffs = self._deriv_coeffs(coeffs)

        x_range = (-5.0, 5.0)

        # Randomly decide whether to show f or f'
        show_derivative = self._rng.random() < 0.4

        result = self._compute_answer(poly, question_type, x_range, show_derivative)
        if result is None:
            return None
        question, answer = result

        # ---- Plot ----
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(1, 1, figsize=(8 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        xs = np.linspace(x_range[0], x_range[1], 1000)

        if show_derivative:
            ys = self._eval_poly(d_coeffs, xs)
            curve_label = "f'(x)"
            # Title must not leak the polynomial expression (which would let a
            # model solve algebraically instead of reading the graph).
            title_expr = "Derivative: f'(x)"
        else:
            ys = self._eval_poly(coeffs, xs)
            curve_label = "f(x)"
            title_expr = "Function: f(x)"

        # Clip for display
        y_finite = ys[np.isfinite(ys)]
        if len(y_finite) == 0:
            plt.close(fig)
            return None
        y_lo_p = float(np.percentile(y_finite, 2))
        y_hi_p = float(np.percentile(y_finite, 98))
        y_pad = max((y_hi_p - y_lo_p) * 0.15, 2.0)

        ax.plot(xs, ys, color=style["geo_line_color"], linewidth=style["line_width"], label=curve_label)
        ax.set_xlim(x_range)
        ax.set_ylim(y_lo_p - y_pad, y_hi_p + y_pad)

        # Do NOT annotate critical or inflection points with their coordinates:
        # that would leak the answer for local_max/local_min/inflection_point
        # and count questions. The model must read them from the curve itself.
        # Also do not draw special markers that reveal critical points directly
        # (other than the query marker for sign_of_derivative, which only
        # identifies the x-location being asked about, not the answer).

        # Mark query point for sign_of_derivative — we need to tell the model
        # which x to evaluate f' at. This does NOT leak the sign answer.
        if self._query_x is not None and not show_derivative:
            xv = self._query_x
            yv = float(self._eval_poly(coeffs, xv))
            ax.plot(xv, yv, "m*", markersize=12, zorder=6)
            ax.annotate(f"x={xv}", (xv, yv), textcoords="offset points",
                        xytext=(10, -12), fontsize=11, color="#8e44ad",
                        fontweight="bold")
        elif self._query_x is not None and show_derivative:
            xv = self._query_x
            yv = float(self._eval_poly(d_coeffs, xv))
            ax.plot(xv, yv, "m*", markersize=12, zorder=6)
            ax.annotate(f"x={xv}", (xv, yv), textcoords="offset points",
                        xytext=(10, -12), fontsize=11, color="#8e44ad",
                        fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        self._apply_style(fig, ax, style)
        fs = style["font_size_base"]
        ax.set_xlabel("x", fontsize=fs + 1)
        ax.set_ylabel("y", fontsize=fs + 1)
        ax.set_title(title_expr, fontsize=fs + 1, pad=10)
        ax.legend(fontsize=fs - 1, loc="best")
        ax.tick_params(labelsize=fs - 1)

        fig.tight_layout()
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        if getattr(self, "_level", 0) <= 2:
            question += (
                " Hint: read f(x) values from the graph at the specified x. "
                "For increasing/decreasing intervals: where the curve goes "
                "up = increasing, down = decreasing. Local max: peak; "
                "local min: trough. f'(x) = slope of the tangent at x; "
                "estimate visually as rise/run."
            )
        return question, answer, img
