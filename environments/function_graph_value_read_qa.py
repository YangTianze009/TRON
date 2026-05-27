"""
Function Graph Value Read QA environment.

Target: Functions -3.14 (across the three splits). The v2 suite
has chart envs, but no env that plots an analytic function and asks the
model to read f(x*) off the graph. This environment does exactly that.

Two independent difficulty axes scale with level:
  * function complexity (linear -> quadratic -> piecewise -> intersection
    of two functions)
  * grid / label granularity (dense gridlines with integer labels at L0,
    sparser gridlines at higher levels)

All questions ask for a single integer answer, format constant.
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class FunctionGraphValueReadQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "function_graph_value_read"

    # ------------------------------------------------------------------ #
    # Difficulty schedule
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesigned: L0 starts with quadratic (old L2-3).
        # L9 requires intersection_no_label + quadratic_inverse.
        if level <= 1:
            kind_pool = ["quadratic"]
        elif level <= 3:
            kind_pool = ["quadratic", "piecewise"]
        elif level <= 5:
            kind_pool = ["piecewise", "intersection"]
        elif level <= 7:
            kind_pool = ["intersection"]
        else:
            kind_pool = ["intersection_no_label", "quadratic_inverse"]
        # Coefficient range grows with level
        coef_hi = 4 + level // 2           # 4 -> 8
        x_range = 6 + level                 # 6 -> 15
        # No minor grid at any level (forces axis reading)
        show_minor_grid = False
        return {
            "kind_pool": kind_pool,
            "coef_hi": coef_hi,
            "x_range": x_range,
            "show_minor_grid": show_minor_grid,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(20):
            result = self._try_generate(parameter)
            if result is not None:
                return result
        return None

    def _try_generate(self, parameter: Dict):
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        kind = rng.choice(cfg["kind_pool"])

        if kind == "linear":
            primary = 1
            a = rng.choice([-3, -2, -1, 1, 2, 3])
            b = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x_q = rng.randint(-cfg["x_range"] // 2, cfg["x_range"] // 2)
            if x_q == 0:
                x_q = 1
            y_q = a * x_q + b
            xs = np.linspace(-cfg["x_range"] // 2, cfg["x_range"] // 2, 200)
            ys = a * xs + b
            label = "y = f(x)"
            img = self._plot_function(
                [(xs, ys, label)], marker=(x_q, y_q),
                show_minor=cfg["show_minor_grid"],
                title="Function plot",
            )
            question = (
                f"The graph shows a function y = f(x). What is the value of "
                f"f({x_q})? Answer with a single integer."
            )
            answer = str(int(y_q))
            if abs(y_q) > 40:
                return None

        elif kind == "quadratic":
            primary = 2
            a = rng.choice([-1, 1])
            # vertex form: y = a*(x-h)^2 + k with clean integer roots
            h = rng.randint(-3, 3)
            k = rng.randint(-4, 4)
            xs = np.linspace(h - 5, h + 5, 200)
            ys = a * (xs - h) ** 2 + k
            x_q = rng.randint(int(h - 3), int(h + 3))
            y_q = a * (x_q - h) ** 2 + k
            label = "y = f(x)"
            img = self._plot_function(
                [(xs, ys, label)], marker=(x_q, y_q),
                show_minor=cfg["show_minor_grid"],
                title="Function plot",
            )
            question = (
                f"The graph shows a quadratic function y = f(x). What is the "
                f"value of f({x_q})? Answer with a single integer."
            )
            answer = str(int(y_q))
            if abs(y_q) > 40:
                return None

        elif kind == "piecewise":
            primary = 3
            # Two-branch piecewise, split at x = s
            s = rng.randint(-2, 3)
            # left branch: y = a1 * x + b1
            a1 = rng.choice([-2, -1, 1, 2])
            b1 = rng.randint(-3, 3)
            # right branch: y = a2 * x + b2 - ensure continuity at x=s by
            # only matching sometimes (dis)continuous is fine visually
            a2 = rng.choice([-2, -1, 1, 2])
            b2 = rng.randint(-3, 3)
            # Choose query from whichever side; record y value.
            side = rng.choice(["left", "right"])
            if side == "left":
                x_q = rng.randint(int(s - 4), int(s - 1))
                y_q = a1 * x_q + b1
            else:
                x_q = rng.randint(int(s + 1), int(s + 4))
                y_q = a2 * x_q + b2
            # Plot both branches
            xs_left = np.linspace(s - 5, s, 100)
            ys_left = a1 * xs_left + b1
            xs_right = np.linspace(s, s + 5, 100)
            ys_right = a2 * xs_right + b2
            label_left = f"f(x), x ≤ {s}"
            label_right = f"f(x), x > {s}"
            img = self._plot_function(
                [(xs_left, ys_left, label_left),
                 (xs_right, ys_right, label_right)],
                marker=(x_q, y_q),
                show_minor=cfg["show_minor_grid"],
                title="Piecewise function",
            )
            question = (
                f"The graph shows a piecewise-linear function y = f(x). What "
                f"is the value of f({x_q})? Answer with a single integer."
            )
            answer = str(int(y_q))
            if abs(y_q) > 40:
                return None

        elif kind == "intersection":
            primary = 4
            # Ask "at what x do the two lines intersect?"
            # Line 1: y = m1*x + b1; Line 2: y = m2*x + b2, m1 != m2
            # Require clean integer intersection
            max_tries = 30
            found = None
            for _t in range(max_tries):
                m1 = rng.choice([-2, -1, 1, 2, 3])
                b1 = rng.randint(-5, 5)
                m2 = rng.choice([-2, -1, 1, 2, 3])
                b2 = rng.randint(-5, 5)
                if m1 == m2:
                    continue
                # x* = (b2 - b1)/(m1 - m2)
                denom = m1 - m2
                num = b2 - b1
                if num % denom != 0:
                    continue
                x_star = num // denom
                if abs(x_star) > 6:
                    continue
                y_star = m1 * x_star + b1
                if abs(y_star) > 12:
                    continue
                found = (m1, b1, m2, b2, x_star, y_star)
                break
            if found is None:
                return None
            m1, b1, m2, b2, x_star, y_star = found
            xs = np.linspace(-7, 7, 200)
            ys1 = m1 * xs + b1
            ys2 = m2 * xs + b2
            label1 = "Line 1"
            label2 = "Line 2"
            img = self._plot_function(
                [(xs, ys1, label1), (xs, ys2, label2)],
                marker=(x_star, y_star),
                show_minor=cfg["show_minor_grid"],
                title="Intersection of two lines",
            )
            question = (
                "The graph shows two lines. At what x-coordinate do the two "
                "lines intersect? Answer with a single integer."
            )
            answer = str(int(x_star))

        elif kind == "intersection_no_label":
            # Same as intersection but NO marker annotation on the graph
            primary = 5
            max_tries = 30
            found = None
            for _t in range(max_tries):
                m1 = rng.choice([-2, -1, 1, 2, 3])
                b1 = rng.randint(-5, 5)
                m2 = rng.choice([-2, -1, 1, 2, 3])
                b2 = rng.randint(-5, 5)
                if m1 == m2:
                    continue
                denom = m1 - m2
                num = b2 - b1
                if num % denom != 0:
                    continue
                x_star = num // denom
                if abs(x_star) > 6:
                    continue
                y_star = m1 * x_star + b1
                if abs(y_star) > 12:
                    continue
                found = (m1, b1, m2, b2, x_star, y_star)
                break
            if found is None:
                return None
            m1, b1, m2, b2, x_star, y_star = found
            xs = np.linspace(-7, 7, 200)
            ys1 = m1 * xs + b1
            ys2 = m2 * xs + b2
            label1 = f"Line 1"
            label2 = f"Line 2"
            # Plot WITHOUT marker label to force reading from graph
            img = self._plot_function_no_marker(
                [(xs, ys1, label1), (xs, ys2, label2)],
                show_minor=False,
                title="Two lines",
            )
            # Ask for y-coordinate (harder since no label)
            question = (
                "The graph shows two lines that intersect. What is the "
                "y-coordinate of the intersection point? Answer with a "
                "single integer."
            )
            answer = str(int(y_star))

        elif kind == "quadratic_inverse":
            # Given a quadratic y = a(x-h)^2 + k, ask: for what x (x > h)
            # does f(x) = target? (inverse question)
            primary = 6
            a = rng.choice([-1, 1])
            h = rng.randint(-2, 2)
            k = rng.randint(-3, 3)
            # Pick a target y that has an integer solution
            dx = rng.randint(1, 4)
            x_answer = h + dx
            target_y = a * dx**2 + k
            xs = np.linspace(h - 5, h + 5, 200)
            ys = a * (xs - h) ** 2 + k
            label = f"y = f(x)"
            img = self._plot_function_no_marker(
                [(xs, ys, label)],
                show_minor=False,
                title="Quadratic function",
                h_line=target_y,
            )
            question = (
                f"The graph shows a quadratic f(x). For what x > {h} does "
                f"f(x) = {target_y}? Answer with a single integer."
            )
            answer = str(int(x_answer))
            if abs(target_y) > 30:
                return None

        else:
            return None

        self._primary_complexity_feature = primary
        return question, answer, img

    # ------------------------------------------------------------------ #
    # Plot helper
    # ------------------------------------------------------------------ #
    def _plot_function(self, curves: List[Tuple[np.ndarray, np.ndarray, str]],
                       marker: Tuple[float, float], show_minor: bool,
                       title: str) -> Image.Image:
        style = self._random_style()
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Plot each curve
        colors_used = palette[:len(curves)]
        for (xs, ys, label), c in zip(curves, colors_used):
            ax.plot(xs, ys, color=c, linewidth=2.2, label=label)

        # Marker at the queried point (show only x-coordinate to avoid answer leak)
        mx, my = marker
        ax.scatter([mx], [my], color="#e60000", s=90, zorder=5,
                   edgecolor="black", linewidth=1.2)
        # Dashed vertical guide from x-axis to marker
        ax.axvline(mx, color="#e60000", linestyle=":", linewidth=0.9,
                   alpha=0.55, zorder=2)

        # Determine plot bounds
        all_x = np.concatenate([c[0] for c in curves])
        all_y = np.concatenate([c[1] for c in curves])
        x_min = min(all_x.min(), mx) - 1
        x_max = max(all_x.max(), mx) + 1
        y_min = min(all_y.min(), my) - 1
        y_max = max(all_y.max(), my) + 1
        # Clamp y for sanity
        y_min = max(y_min, -25)
        y_max = min(y_max, 25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Grid + axes
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.grid(True, which="major", color="#cccccc", linewidth=0.8, alpha=0.8)
        if show_minor:
            ax.minorticks_on()
            ax.grid(True, which="minor", color="#eeeeee", linewidth=0.4, alpha=0.7)
        # Integer ticks only
        ax.set_xticks(np.arange(math.ceil(x_min), math.floor(x_max) + 1))
        ax.set_yticks(np.arange(math.ceil(y_min), math.floor(y_max) + 1))

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=10)

        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _plot_function_no_marker(self, curves, show_minor, title,
                                  h_line=None) -> Image.Image:
        """Plot without annotated marker -- forces visual reading."""
        style = self._random_style()
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        colors_used = palette[:len(curves)]
        for (xs, ys, label), c in zip(curves, colors_used):
            ax.plot(xs, ys, color=c, linewidth=2.2, label=label)

        all_x = np.concatenate([c[0] for c in curves])
        all_y = np.concatenate([c[1] for c in curves])
        x_min = all_x.min() - 1
        x_max = all_x.max() + 1
        y_min = max(all_y.min() - 1, -25)
        y_max = min(all_y.max() + 1, 25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.axvline(0, color="#333333", linewidth=1.0)
        ax.grid(True, which="major", color="#cccccc", linewidth=0.8, alpha=0.8)
        if show_minor:
            ax.minorticks_on()
            ax.grid(True, which="minor", color="#eeeeee", linewidth=0.4, alpha=0.7)
        ax.set_xticks(np.arange(math.ceil(x_min), math.floor(x_max) + 1))
        ax.set_yticks(np.arange(math.ceil(y_min), math.floor(y_max) + 1))

        if h_line is not None:
            ax.axhline(h_line, color="#e60000", linewidth=1.5, linestyle="--",
                        alpha=0.7, label=f"y = {h_line}")

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b4"
    os.makedirs(out_dir, exist_ok=True)
    env = FunctionGraphValueReadQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            status = "OK" if ok else "FAIL"
            print(f"[function_graph_value_read] L{level} seed{seed}: {status} "
                  f"ans={env._answer!r}")
            if ok:
                env._image.save(
                    f"{out_dir}/function_graph_value_read_s{seed}_L{level}.png")
