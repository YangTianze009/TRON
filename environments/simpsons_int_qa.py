"""
Simpson's / Trapezoidal Rule Numerical Integration QA (D45, P3 — reference
arithmetic).

Reference an external reference:
  "what is the Simpson's approximation of the integral of f(x) from 1 to 4?"
  Ans: 154.0

reference an external reference verbatim (sibling, trapezoidal):
  "what is the trapezoidal approximation of the integral of f(x) from 1 to 5?"
  Ans: 264.0

This env shows the curve y = f(x) on a coordinate grid with vertical
dashed lines at the integration limits and asks the model to compute
either the trapezoidal rule sum or the composite Simpson's rule sum.

Verifier: float (`\\boxed{154}` or `\\boxed{154.0}` accepted; tolerant).

Difficulty:
  L0..L2 — Trapezoidal, simple polynomial f(x), small interval.
  L3..L5 — Trapezoidal or Simpson's, medium polynomial.
  L6..L9 — Simpson's rule, larger interval, more partition points.
"""
import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _trapezoidal(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Composite trapezoidal rule with n+1 sample points (n intervals)."""
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for k in range(1, n):
        s += f(a + k * h)
    return h * s


def _simpsons(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Composite Simpson's 1/3 rule. n must be even; uses n+1 points."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for k in range(1, n):
        c = f(a + k * h)
        s += 4 * c if k % 2 == 1 else 2 * c
    return (h / 3.0) * s


class SimpsonsIntQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "simpsons_int"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"rule": "trapezoidal", "n_intervals": 4,
                    "curve_pool": ["lin", "quad_pos", "abs_v"]}
        if level <= 4:
            return {"rule": "trapezoidal", "n_intervals": 6,
                    "curve_pool": ["quad_pos", "quad_neg", "cubic"]}
        if level <= 5:
            return {"rule": "simpsons", "n_intervals": 4,
                    "curve_pool": ["quad_pos", "quad_neg", "cubic"]}
        if level <= 7:
            return {"rule": "simpsons", "n_intervals": 6,
                    "curve_pool": ["quad_pos", "cubic", "exp_decay"]}
        return {"rule": "simpsons", "n_intervals": 8,
                "curve_pool": ["cubic", "exp_decay", "polyq"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7919 + level * 197 + 31)

        for _ in range(20):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg):
        curve = rng.choice(cfg["curve_pool"])
        f, label = self._build_curve(curve, rng)

        # Pick an integration interval whose left/right endpoints are integers
        a = rng.choice([0, 1, 2])
        # length 3, 4, or 5 (use even length so Simpson's `n` divides nicely)
        length = rng.choice([3, 4]) if cfg["rule"] == "simpsons" else rng.choice([3, 4, 5])
        b = a + length

        n_int = cfg["n_intervals"]
        if cfg["rule"] == "simpsons" and n_int % 2 != 0:
            n_int += 1

        # Compute the chosen approximation
        if cfg["rule"] == "trapezoidal":
            ans_raw = _trapezoidal(f, a, b, n_int)
        else:
            ans_raw = _simpsons(f, a, b, n_int)

        # Reject if too small or weird
        if not math.isfinite(ans_raw) or abs(ans_raw) < 0.1 or abs(ans_raw) > 5000:
            return None

        # Round to 3 decimals (reference float tolerance)
        ans = round(ans_raw, 3)

        # Format string: prefer integer form when ≈ integer
        if abs(ans - round(ans)) < 0.005:
            ans_str = f"{int(round(ans))}"
        elif abs(ans - round(ans, 1)) < 0.005:
            ans_str = f"{round(ans, 1):.1f}"
        else:
            ans_str = f"{ans:.3f}"

        # Build question
        rule_name = "trapezoidal" if cfg["rule"] == "trapezoidal" else "Simpson's"
        question = (
            f"The image shows the curve y = {label}. Using the {rule_name} "
            f"rule with {n_int} subintervals, approximate the integral of "
            f"f(x) from x = {a} to x = {b}. Provide your answer as a number."
        )

        # Render
        img = self._render(f, label, a, b, n_int, rule_name)
        return question, ans_str, img

    def _build_curve(self, curve: str, rng: random.Random):
        """Return (f, label) tuple."""
        if curve == "lin":
            m = rng.choice([1, 2, 3])
            c = rng.choice([1, 2, 3])
            return (lambda x, _m=m, _c=c: _m * x + _c, f"{m}x + {c}")
        if curve == "quad_pos":
            a = rng.choice([1, 2])
            c = rng.choice([0, 1, 2])
            return (lambda x, _a=a, _c=c: _a * x * x + _c, f"{a}x^2 + {c}")
        if curve == "quad_neg":
            a = rng.choice([1, 2])
            shift = rng.choice([3, 4, 5])
            return (lambda x, _a=a, _s=shift: _a * x * x + _s, f"{a}x^2 + {shift}")
        if curve == "cubic":
            a = rng.choice([1, 2])
            offset = rng.choice([1, 2, 3])
            return (lambda x, _a=a, _o=offset: _a * x ** 3 + _o,
                    f"{a}x^3 + {offset}")
        if curve == "polyq":
            return (lambda x: x ** 2 - x + 4, "x^2 - x + 4")
        if curve == "exp_decay":
            A = rng.choice([5, 10, 15])
            k = rng.choice([0.3, 0.4])
            return (lambda x, _A=A, _k=k: _A * math.exp(-_k * x) + 2,
                    f"{A}e^(-{k}x) + 2")
        if curve == "abs_v":
            k = rng.choice([1, 2, 3])
            return (lambda x, _k=k: abs(x - _k) + 1, f"|x - {k}| + 1")
        # fallback
        return (lambda x: x ** 2, "x^2")

    # ------------------------------------------------------------------ #
    def _render(self, f, label, a, b, n_int, rule_name) -> Image.Image:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Plot range slightly wider than [a, b]
        pad = max(1.0, (b - a) * 0.25)
        plot_xs = np.linspace(a - pad, b + pad, 400)
        try:
            plot_ys = np.array([f(x) for x in plot_xs])
        except Exception:
            return None
        ax.plot(plot_xs, plot_ys, color="#1565c0", linewidth=2.0,
                label=f"f(x) = {label}")

        # Sample points for the rule
        h = (b - a) / n_int
        xs_sample = [a + k * h for k in range(n_int + 1)]
        ys_sample = [f(x) for x in xs_sample]

        # Draw chord segments (trapezoid tops) or parabolic arcs
        for k in range(n_int):
            x0, x1 = xs_sample[k], xs_sample[k + 1]
            y0, y1 = ys_sample[k], ys_sample[k + 1]
            ax.plot([x0, x1], [y0, y1], color="#e67e22", linewidth=1.4,
                    linestyle="--", alpha=0.9)
            # Vertical drop at sample points
            ax.plot([x0, x0], [0, y0], color="#7f8c8d",
                    linewidth=0.7, alpha=0.5)
        # Last vertical
        ax.plot([xs_sample[-1], xs_sample[-1]], [0, ys_sample[-1]],
                color="#7f8c8d", linewidth=0.7, alpha=0.5)
        # Sample dots
        ax.plot(xs_sample, ys_sample, "o", color="#c0392b", markersize=5)

        # Bounding dashed verticals at a, b
        ax.axvline(a, color="#333333", linewidth=1.4, linestyle=":")
        ax.axvline(b, color="#333333", linewidth=1.4, linestyle=":")

        # Axes
        ymin = float(min(min(ys_sample), 0)) - 1
        ymax = float(max(max(ys_sample), 0)) + 1
        ax.set_xlim(a - pad, b + pad)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(list(range(int(a - pad), int(b + pad) + 1)))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("f(x)", fontsize=11)

        # Annotate sample x-values for clarity (model needs to know which n)
        for x, y in zip(xs_sample, ys_sample):
            xstr = f"{int(x)}" if float(x).is_integer() else f"{x:.1f}"
            ax.text(x, ymin + 0.3, xstr, fontsize=8, ha="center", color="#555")

        ax.set_title(f"y = {label} with {n_int} subintervals on [{a}, {b}]",
                     fontsize=11, fontweight="bold")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = SimpsonsIntQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok}; A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
