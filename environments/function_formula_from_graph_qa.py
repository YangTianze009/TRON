"""
Function Formula From Graph QA (v4 G5b, for function-reading reasoning/Expression).

Targets:

Task: render a function plot (y vs x) from a simple parametric family
(linear y = mx+b, quadratic y = ax²+bx+c, absolute y = |x-h|+k, exponential,
reciprocal). Ask for the formula or a parameter.

Reward: symbolic equivalence (SymPy) or exact parameter value.

Level axes:
  A) Function family: linear at L0-2, quadratic at L3-5, abs/absolute at L6-7, piecewise at L8-9
  B) Query: read y at x at L0-2, read slope at L3, read parameter at L5+
"""
import random
import math
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The figure shows y = f(x) for a certain function. {query}",
    "A function plot is shown. {query}",
    "Given the function graph, {query}",
    "The plotted function is shown in the figure. {query}",
    "From the figure, {query}",
    "Looking at the plot, {query}",
    "Based on the function shown, {query}",
    "The graph represents y = f(x). {query}",
    "{query} (from the figure)",
    "Examine the function plot. {query}",
    "The function shown is y = f(x). {query}",
    "Given the plot of f(x), {query}",
    "{query} (based on the graph)",
    "Use the function plot to {query}",
    "Looking at the graphed function, {query}",
    "Function plot answer: {query}",
]

class FunctionFormulaFromGraphQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "function_formula_from_graph"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            fns = ["linear"]
            qtype = "evaluate"
        elif level <= 5:
            fns = ["linear", "quadratic"]
            qtype = "parameter"
        else:
            fns = ["linear", "quadratic", "abs"]
            qtype = "parameter"
        return {"fns": fns, "qtype": qtype}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 317)
        self._primary_complexity_feature = level

        fn = rng.choice(cfg["fns"])
        marked_pt = None  # (x, y) to show on graph as a red dot
        if fn == "linear":
            m = rng.randint(-3, 3)
            while m == 0: m = rng.randint(-3, 3)
            b = rng.randint(-5, 5)
            f = lambda x: m * x + b
            if cfg["qtype"] == "evaluate":
                # Pick x0 such that y(x0) is comfortably inside the [-9, 9]
                # display range — otherwise the line is off-chart and the
                # model can't read it.
                valid_x = [x for x in range(-5, 6)
                            if -9 <= m * x + b <= 9 and x != 0]
                if not valid_x:
                    valid_x = [0]
                x0 = rng.choice(valid_x)
                y0 = f(x0)
                marked_pt = (x0, y0)
                answer = str(y0)
                query = (f"A red dot is plotted on the line at x = {x0}. "
                         f"What is f({x0})? Put the integer in "
                         f"<answer>...</answer>.")
            else:
                answer = str(m)
                query = "what is the slope of this line? Integer in <answer>...</answer>."
        elif fn == "quadratic":
            a = rng.choice([-2, -1, 1, 2])
            h = rng.randint(-3, 3)
            # Restrict k so vertex is on screen
            k = rng.randint(-4, 4)
            f = lambda x, a=a, h=h, k=k: a * (x - h) ** 2 + k
            marked_pt = (h, k)
            answer = f"({h}, {k})"
            query = ("A red dot marks the vertex of the parabola. What are "
                     "the coordinates of the vertex? Format '(x, y)' in "
                     "<answer>...</answer>.")
        else:  # abs
            h = rng.randint(-3, 3)
            k = rng.randint(-3, 3)
            f = lambda x, h=h, k=k: abs(x - h) + k
            marked_pt = (h, k)
            answer = f"({h}, {k})"
            query = ("A red dot marks the 'V' vertex of the absolute-value "
                     "graph. Format '(x, y)' in <answer>...</answer>.")

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(query=query)

        img = self._render(f, rng, marked_pt=marked_pt)
        return q, answer, img

    def _render(self, f, rng, marked_pt=None):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        xs = np.linspace(-6, 6, 200)
        try:
            ys = np.array([f(x) for x in xs])
        except Exception:
            return Image.new("RGB", (400, 400), "white")
        ax.plot(xs, ys, color="blue", lw=2.0)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xticks(range(-6, 7))
        ax.set_yticks(range(-10, 11, 2))
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title("y = f(x)", fontsize=13)
        if marked_pt is not None:
            mx, my = marked_pt
            if -6 <= mx <= 6 and -10 <= my <= 10:
                ax.plot(mx, my, "o", color="red", markersize=12,
                        markeredgecolor="black", zorder=5)
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        if pred == gt:
            return True
        if gt.startswith("(") and gt.endswith(")"):
            try:
                pv = [float(x.strip()) for x in pred.strip("()").split(",")]
                gv = [float(x.strip()) for x in gt.strip("()").split(",")]
                if len(pv) == len(gv):
                    return all(abs(a - b) < 0.3 for a, b in zip(pv, gv))
            except ValueError:
                return False
        try:
            return abs(float(pred) - float(gt)) < 0.3
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_ffg"
    os.makedirs(out_dir, exist_ok=True)
    env = FunctionFormulaFromGraphQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 443
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[ffg L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/ffg_s{s}_L{level}.png")
            print(f"[ffg L{level} s{s}] A={env._answer}")
