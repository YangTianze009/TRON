"""
Rate of Change From Graph QA environment.

Shows a smooth function curve on gridded axes with two points A and B
marked and connected by a dashed secant line. The model estimates the
average rate of change (slope) between A and B.

Difficulty axes:
  - curve_type: linear -> quadratic -> cubic -> trig/exponential
  - point_separation (wide -> narrow, approaches derivative) and
    grid_visibility (fine grid -> no gridlines)
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

class RateOfChangeFromGraphQA(StandaloneVisualEnv):
    ENV_NAME = "rate_of_change_from_graph"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            curves = ["linear"]
        elif level <= 3:
            curves = ["linear", "quadratic"]
        elif level <= 5:
            curves = ["quadratic", "cubic"]
        elif level <= 7:
            curves = ["cubic", "trig"]
        else:
            curves = ["trig", "exponential"]
        # Point separation: wider at low levels, narrow at high
        if level <= 1:
            sep_range = (3.0, 4.0)
        elif level <= 3:
            sep_range = (2.0, 3.5)
        elif level <= 5:
            sep_range = (1.5, 2.5)
        elif level <= 7:
            sep_range = (0.8, 1.5)
        else:
            sep_range = (0.3, 0.9)
        # Grid visibility: L0 fine, L9 none
        if level <= 2:
            grid_style = "fine"
        elif level <= 5:
            grid_style = "major"
        elif level <= 7:
            grid_style = "coarse"
        else:
            grid_style = "none"
        # Hide coordinate values from point annotations so model must read
        # the graph, not OCR the label. L0-L2 show just letters (scaffolding
        # via grid/ticks), L3-L6 hide x too, L7+ no labels at all.
        if level <= 2:
            label_mode = "letter_only"   # just "A"
        elif level <= 6:
            label_mode = "letter_only"   # still just "A" — never leak coords
        else:
            label_mode = "letter_only"
        # L8+: tighter distractor spacing
        tight_distractors = level >= 8
        return {
            "curves": curves,
            "sep_range": sep_range,
            "grid_style": grid_style,
            "label_mode": label_mode,
            "tight_distractors": tight_distractors,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1471)

        for _ in range(30):
            result = self._try_generate(sub_rng, level, cfg)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    def _build_func(self, curve_type: str, sub_rng: random.Random):
        if curve_type == "linear":
            m = sub_rng.choice([-3, -2, -1, 1, 2, 3])
            b = sub_rng.randint(-3, 3)
            f = lambda x, _m=m, _b=b: _m * x + _b
            label = "linear"  # don't leak slope in the legend
            x_range = (-5, 5)
        elif curve_type == "quadratic":
            a = sub_rng.choice([-1, 1]) * sub_rng.choice([0.25, 0.5, 1.0])
            h = sub_rng.randint(-2, 2)
            k = sub_rng.randint(-2, 2)
            f = lambda x, _a=a, _h=h, _k=k: _a * (x - _h) ** 2 + _k
            label = f"quadratic"
            x_range = (-5, 5)
        elif curve_type == "cubic":
            a = sub_rng.choice([-0.25, -0.1, 0.1, 0.25])
            b = sub_rng.choice([-1, 0, 1])
            c = sub_rng.randint(-2, 2)
            f = lambda x, _a=a, _b=b, _c=c: _a * x ** 3 + _b * x + _c
            label = "cubic"
            x_range = (-4, 4)
        elif curve_type == "trig":
            fn = sub_rng.choice(["sin", "cos"])
            A = sub_rng.choice([1.0, 1.5, 2.0])
            phi = sub_rng.choice([0, math.pi / 4, math.pi / 3, math.pi / 6])
            if fn == "sin":
                f = lambda x, _A=A, _p=phi: _A * math.sin(x + _p)
            else:
                f = lambda x, _A=A, _p=phi: _A * math.cos(x + _p)
            label = f"{fn}-based"
            x_range = (-4, 4)
        else:  # exponential
            base = sub_rng.choice([0.5, 1.5, 2.0])
            sign = sub_rng.choice([-1, 1])
            f = lambda x, _b=base, _s=sign: _s * (_b ** x) - _s
            label = "exponential"
            x_range = (-3, 3)
        return f, label, x_range

    # ------------------------------------------------------------------ #
    def _try_generate(self, sub_rng: random.Random, level: int, cfg: Dict):
        curve_type = sub_rng.choice(cfg["curves"])
        f, label, x_range = self._build_func(curve_type, sub_rng)

        sep_lo, sep_hi = cfg["sep_range"]
        separation = round(sub_rng.uniform(sep_lo, sep_hi), 1)
        # choose A and B such that both are in range
        x_min, x_max = x_range
        # midpoint
        mid = sub_rng.uniform(x_min + separation / 2 + 0.5, x_max - separation / 2 - 0.5)
        # round to nearest 0.5
        xA = round(mid - separation / 2, 1)
        xB = round(mid + separation / 2, 1)
        if xA < x_min + 0.2 or xB > x_max - 0.2:
            return None

        yA = f(xA)
        yB = f(xB)
        dx = xB - xA
        if abs(dx) < 1e-6:
            return None
        slope = (yB - yA) / dx
        slope_r = round(slope, 2)

        if abs(yA) > 8 or abs(yB) > 8:
            return None

        # Build MCQ options
        correct = slope_r
        distractors = set()
        tries = 0
        if cfg.get("tight_distractors"):
            # Tight: spacing ~0.10-0.30 of correct value
            delta_pool = [-0.6, -0.4, -0.25, -0.15, 0.15, 0.25, 0.4, 0.6]
            min_sep = 0.12
        else:
            delta_pool = [-1.5, -1.0, -0.6, -0.4, 0.4, 0.6, 1.0, 1.5]
            min_sep = 0.25
        while len(distractors) < 3 and tries < 80:
            tries += 1
            dv = sub_rng.choice(delta_pool)
            cand = round(correct + dv, 2)
            if abs(cand - correct) < min_sep:
                continue
            if cand in distractors:
                continue
            # avoid near duplicates
            if any(abs(cand - d) < min_sep for d in distractors):
                continue
            distractors.add(cand)
        if len(distractors) < 3:
            return None

        options = list(distractors) + [correct]
        sub_rng.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = chr(ord("A") + correct_idx)
        opt_str = ", ".join(f"{chr(ord('A') + i)}) {v:.2f}" for i, v in enumerate(options))

        # Render
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 5.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        xs = np.linspace(x_min, x_max, 400)
        ys = np.array([f(x) for x in xs])
        color = style["palette"][0]
        ax.plot(xs, ys, color=color, linewidth=2.2, label=label)

        # Secant line from A to B (dashed)
        # Extend slightly beyond A and B
        span = max(abs(xB - xA), 0.4)
        lx = xA - 0.3 * span
        rx = xB + 0.3 * span
        ax.plot([lx, rx],
                [yA + slope * (lx - xA), yA + slope * (rx - xA)],
                color="#e60000", linewidth=1.8, linestyle="--",
                label="secant line")

        # Mark A and B
        ax.plot(xA, yA, "o", color="#e60000", markersize=9,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.plot(xB, yB, "o", color="#e60000", markersize=9,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        label_mode = cfg.get("label_mode", "full")
        if label_mode == "full":
            labA = f"A({xA}, {round(yA, 1)})"
            labB = f"B({xB}, {round(yB, 1)})"
        elif label_mode == "x_only":
            labA = f"A (x={xA})"
            labB = f"B (x={xB})"
        else:
            labA = "A"
            labB = "B"
        ax.annotate(labA, xy=(xA, yA),
                    xytext=(xA - 0.2, yA - 0.8),
                    fontsize=10, color="#1a1a1a")
        ax.annotate(labB, xy=(xB, yB),
                    xytext=(xB + 0.15, yB + 0.3),
                    fontsize=10, color="#1a1a1a")

        # Bounds
        y_lo = min(min(ys), yA, yB) - 1.0
        y_hi = max(max(ys), yA, yB) + 1.0
        y_lo = max(y_lo, -10)
        y_hi = min(y_hi, 10)
        ax.set_xlim(x_min - 0.2, x_max + 0.2)
        ax.set_ylim(y_lo, y_hi)

        # Grid style
        gstyle = cfg["grid_style"]
        if gstyle == "fine":
            ax.minorticks_on()
            ax.grid(True, which="major", color="#bbbbbb", linewidth=0.7, alpha=0.9)
            ax.grid(True, which="minor", color="#eeeeee", linewidth=0.4, alpha=0.7)
        elif gstyle == "major":
            ax.grid(True, which="major", color="#bbbbbb", linewidth=0.7, alpha=0.8)
        elif gstyle == "coarse":
            ax.grid(True, which="major", color="#dddddd", linewidth=0.4, alpha=0.5)
        else:
            ax.grid(False)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("f(x)", fontsize=11)
        title_pool = [
            "Average rate of change",
            "Slope from A to B",
            "Secant slope",
            "Function with secant line",
            "Average Rate of Change",
        ]
        ax.set_title(sub_rng.choice(title_pool), fontsize=12, fontweight="bold")
        ax.legend(loc=sub_rng.choice(["best", "upper left", "upper right",
                                       "lower right", "lower left"]),
                  fontsize=9)

        templates = [
            f"The graph shows a function f(x) with two points A and B marked on the curve; the dashed line is the secant through A and B. What is the approximate average rate of change (slope) of f between A and B? {opt_str}. Answer with a single letter.",
            f"From the graph, points A and B lie on f(x), connected by a dashed secant. Estimate the average rate of change of f from A to B. {opt_str}. Reply with one letter.",
            f"Looking at the curve and the dashed secant joining the marked points A and B, what is f's average rate of change between A and B? {opt_str}. Provide a single letter.",
            f"The dashed line in the graph is the secant joining points A and B on f(x). Approximate the slope of this secant (i.e., the average rate of change). {opt_str}. Single letter answer.",
            f"Estimate the average rate of change of f(x) between points A and B as shown in the graph. {opt_str}. Answer with one letter.",
        ]
        question = sub_rng.choice(templates)
        answer = correct_letter

        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question, answer, img
