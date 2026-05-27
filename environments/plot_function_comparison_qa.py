"""
Plot Function Comparison QA (batch 3, 2026-04-14).

Target: Coordinate / figure QA. Two functions are
plotted on the same axes, each labelled (f and g). The question asks
which has a greater value at a specific x, or where they intersect, etc.

Format: constant MCQ letter (A/B/C/D).

Difficulty axes:
  A) Function class pool grows with level (linear only → linear+quadratic
     → linear+quadratic+exponential).
  B) Number of intersection points / query x distance from intersection
     shrinks → harder discrimination.
  C) Gridline and tick label density thin at high level.
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

def _eval(fn_spec, x):
    t = fn_spec["type"]
    if t == "linear":
        return fn_spec["a"] * x + fn_spec["b"]
    if t == "quadratic":
        return fn_spec["a"] * (x - fn_spec["h"]) ** 2 + fn_spec["k"]
    if t == "exp":
        return fn_spec["a"] * math.exp(fn_spec["k"] * x) + fn_spec["b"]
    raise ValueError(t)

class PlotFunctionComparisonQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "plot_function_comparison"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        classes = ["linear"]
        if level >= 2:
            classes.append("quadratic")
        if level >= 5:
            classes.append("exp")
        return {
            "fn_classes": classes,
            "query_margin": max(0.5, 4.0 - 0.4 * level),   # how far from intersection
            "x_mag": 3 + level // 2,                       # query x magnitude range
            "show_grid": level <= 5,
            # In-plot inline f/g annotations are only at easy levels to
            # avoid clutter. Legend must ALWAYS be shown so the viewer can
            # identify which curve is f vs g — otherwise the task is
            # underdetermined.
            "show_inline_labels": level <= 3,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = len(cfg["fn_classes"]) * 10 + cfg["x_mag"]

        for _ in range(25):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _sample_fn(self, rng, cfg):
        t = rng.choice(cfg["fn_classes"])
        if t == "linear":
            a = rng.choice([-3, -2, -1, 1, 2, 3])
            b = rng.randint(-4, 4)
            return {"type": "linear", "a": a, "b": b}
        if t == "quadratic":
            a = rng.choice([-1, 1, -1, 1])  # mostly unit
            h = rng.randint(-2, 2)
            k = rng.randint(-3, 3)
            return {"type": "quadratic", "a": a, "h": h, "k": k}
        if t == "exp":
            a = rng.choice([1, 1, -1])
            k = rng.choice([0.3, 0.5, -0.3, -0.5])
            b = rng.randint(-2, 2)
            return {"type": "exp", "a": a, "k": k, "b": b}

    def _try_generate(self, rng, cfg, level):
        f = self._sample_fn(rng, cfg)
        g = self._sample_fn(rng, cfg)

        # Pick a query x in [-x_mag, x_mag] that yields a clear difference
        for _ in range(20):
            x_q = rng.randint(-cfg["x_mag"], cfg["x_mag"])
            try:
                fv = _eval(f, x_q)
                gv = _eval(g, x_q)
            except OverflowError:
                continue
            if not (math.isfinite(fv) and math.isfinite(gv)):
                continue
            diff = abs(fv - gv)
            # need difference big enough (at least 0.8) to avoid ambiguity
            if diff >= 0.8 and abs(fv) < 50 and abs(gv) < 50:
                break
        else:
            return None

        # reference expects bare-text answer ("f" or "g"), not MCQ letter.
        # The two functions are clearly distinguishable in the plot (one
        # solid one dashed) and named via legend.
        greater_is_f = fv > gv
        answer = "f" if greater_is_f else "g"
        q = (f"Two functions f and g are plotted. At x = {x_q}, which has "
             f"the greater value? Answer with the single letter `f` or `g`.")

        image = self._render(f, g, x_q, cfg)
        return q, answer, image

    def _render(self, f, g, x_q, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        x_min, x_max = -8, 8
        y_min, y_max = -12, 12
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        if cfg["show_grid"]:
            ax.grid(True, which="major", alpha=0.3, linestyle="--")

        ax.axhline(0, color="#333", linewidth=1.1, zorder=1)
        ax.axvline(0, color="#333", linewidth=1.1, zorder=1)

        xs = np.linspace(x_min, x_max, 400)
        with np.errstate(over="ignore"):
            ys_f = np.array([_eval(f, float(x)) for x in xs])
            ys_g = np.array([_eval(g, float(x)) for x in xs])
        mask_f = (ys_f >= y_min) & (ys_f <= y_max) & np.isfinite(ys_f)
        mask_g = (ys_g >= y_min) & (ys_g <= y_max) & np.isfinite(ys_g)
        # Always label both curves so legend can identify f vs g.
        ax.plot(xs[mask_f], ys_f[mask_f], color=palette[0 % len(palette)],
                linewidth=2.4, zorder=3, label="f(x)")
        ax.plot(xs[mask_g], ys_g[mask_g], color=palette[2 % len(palette)],
                linewidth=2.4, zorder=3, linestyle="--", label="g(x)")

        # Mark the query x with a vertical line
        ax.axvline(x_q, color="#888", linewidth=1.2, linestyle=":", zorder=2)
        ax.text(x_q, y_max - 0.5, f"x = {x_q}",
                fontsize=fs, ha="center", va="top",
                bbox=dict(facecolor="#fff", alpha=0.8, edgecolor="#888"))

        if cfg["show_inline_labels"]:
            # Add f/g label near left edge
            xf = -5
            yf = _eval(f, xf)
            if y_min <= yf <= y_max:
                ax.annotate("f", (xf, yf), xytext=(0, 4),
                            textcoords="offset points", color=palette[0 % len(palette)],
                            fontsize=fs + 3, fontweight="bold")
            xg = 5
            yg = _eval(g, xg)
            if y_min <= yg <= y_max:
                ax.annotate("g", (xg, yg), xytext=(0, 4),
                            textcoords="offset points", color=palette[2 % len(palette)],
                            fontsize=fs + 3, fontweight="bold")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(range(x_min, x_max + 1, 2))
        ax.set_yticks(range(y_min, y_max + 1, 2))
        ax.tick_params(labelsize=fs - 2)
        ax.set_xlabel("x", fontsize=fs + 1)
        ax.set_ylabel("y", fontsize=fs + 1)
        # Always show legend — otherwise f/g are indistinguishable.
        ax.legend(loc=style["legend_loc"], fontsize=fs)
        ax.set_title("Compare f(x) and g(x)", fontsize=fs + 1)
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = PlotFunctionComparisonQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[plot_function_comparison L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"plot_function_comparison_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[plot_function_comparison L{level} s{s}] A={env._answer}")
