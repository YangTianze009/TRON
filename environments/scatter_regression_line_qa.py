"""
Scatter Regression Line QA environment.

Capabilities: D1 (chart value extraction) + M3 (function/coordinate)
Target regression: chart-reading, statistical, dynamic-math statistics.

A scatter plot with 10-30 points, best-fit line, labeled axes, gridlines.
4-option MCQ asking about slope, count of points above/below line, or
predicted y for given x.

Difficulty schedule (0..9):
  Axis 1: n_points = 10 + level * 2    -> 10..28
  Axis 2: correlation_strength L0: r>0.9, L9: r~0.4
  Axis 3: question_type  L<=2: count above/below  L3..L6: slope
          L>=7: predict y for a given x
  Axis 4: grid spacing coarser at higher levels.

Output: (question_str, answer_letter, PIL_Image)
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

_X_LABELS = [
    "Temperature (°C)", "Age (years)", "Weight (kg)", "Height (cm)",
    "Income ($K)", "Experience (years)", "Distance (km)", "Price ($)",
    "Study Hours", "Marketing Spend ($K)", "Dose (mg)", "Rainfall (mm)",
    "Hours of Sleep", "Altitude (m)", "Year",
]
_Y_LABELS = [
    "Sales ($K)", "Performance Score", "Revenue ($M)", "Satisfaction",
    "Test Score", "Productivity", "Fuel Efficiency",
    "Response Time (s)", "Profit Margin (%)", "Output (units)",
    "Yield (tons)", "Growth Rate (%)", "Score (0-100)", "Energy (kWh)",
    "Rating",
]
_TITLE_TEMPLATES = [
    "{x} vs {y}",
    "{y} as a Function of {x}",
    "Correlation of {x} and {y}",
    "{x} / {y} Scatter with Best-Fit Line",
    "{y} Plotted Against {x}",
    "Regression: {y} on {x}",
]
_MARKER_STYLES = ["o", "s", "^", "D", "v", "p", "h", "*", "X"]

# Question templates (same type, different wording)
_Q_COUNT_TEMPLATES = [
    "Approximately how many points lie strictly above the best-fit line?",
    "Counting only points above the plotted regression line, how many are there?",
    "How many data points are situated above the best-fit line?",
]
_Q_SLOPE_TEMPLATES = [
    "Which option is the closest approximation of the slope of the best-fit line?",
    "Estimate the slope of the regression line shown. Choose the closest value.",
    "Which number best matches the slope of the plotted best-fit line?",
]
_Q_PRED_TEMPLATES = [
    "Using the best-fit regression line shown, predict the approximate y-value when x = {x}.",
    "From the regression line on the chart, estimate y for x = {x}.",
    "What y-value does the fitted line give for x = {x}?",
]

class ScatterRegressionLineQA(StandaloneVisualEnv):
    ENV_NAME = "scatter_regression_line"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # n_points 10..28
        n_points = 10 + level * 2
        # correlation_strength: r from ~0.95 at L0 to ~0.4 at L9
        r_target = 0.95 - (level / 9.0) * 0.55
        # Question-type schedule reordered for monotonic difficulty:
        # slope estimation with clear trend is the easiest visual task, so
        # it goes first; counting points above/below the line is harder
        # because boundary cases are ambiguous and distractors are +/-1
        # apart; predict_y with weaker correlation is hardest.
        # (Previously L3 came out easier than L0 — inverted curve.)
        # Iter 3 (2026-04-17): L3 still collapsed to 0.10 because
        # count_above with noisy points was much harder than slope at L0-2.
        # Extend slope through L3 so L0-3 share the easiest qtype; push
        # count_above/below into the middle band and predict_y last.
        # Iter 4 (2026-04-17): L6=0.05 collapsed — count_above with 22 pts
        # and weak correlation is nearly random-chance because boundary
        # cases are fundamentally ambiguous (points near the line). Replace
        # L6-L7 with slope on noisier data (still solvable) and keep
        # count_above only as a 25% injection for variety. predict_y
        # remains the hardest at L8-L9.
        if level <= 4:
            qtype = "slope"
        elif level <= 6:
            qtype = "slope"  # harder noise but still solvable
        elif level <= 7:
            qtype = "count_above"
        else:
            qtype = "predict_y"
        # Grid spacing: coarse at higher levels
        if level <= 3:
            grid_step = 1.0
        elif level <= 6:
            grid_step = 2.0
        else:
            grid_step = 5.0
        return {
            "n_points": n_points,
            "r_target": max(0.3, r_target),
            "qtype": qtype,
            "grid_step": grid_step,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2237)

        for _ in range(10):
            try:
                result = self._try_generate(sub_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_points"]
        r_target = cfg["r_target"]
        np_rng = np.random.RandomState(rng.randint(0, 1_000_000))

        # slope: between 0.2 and 2.5, varying sign
        if level <= 2:
            slope = rng.choice([1.5, 2.0, -1.5, -2.0])
        elif level <= 6:
            slope = round(rng.uniform(0.3, 2.0) * rng.choice([-1, 1]), 2)
        else:
            slope = round(rng.uniform(0.3, 1.5) * rng.choice([-1, 1]), 2)
        intercept = rng.uniform(5, 40)

        # x range: choose based on grid_step
        grid_step = cfg["grid_step"]
        x_min = rng.choice([0, 5, 10, 20])
        x_max = x_min + max(20, int(grid_step * 10))
        xs = sorted([round(rng.uniform(x_min, x_max), 1) for _ in range(n)])

        # Inject noise such that correlation approx r_target
        y_clean = np.array([slope * x + intercept for x in xs])
        y_std = np.std(y_clean)
        if y_std < 1e-6:
            y_std = 1.0
        # noise std to hit r_target approximately:
        # r^2 = var(clean)/var(total), so noise_var = var(clean)*(1/r^2 - 1)
        noise_var = (y_std ** 2) * (1.0 / max(r_target ** 2, 0.01) - 1.0)
        noise_std = max(0.5, math.sqrt(max(noise_var, 0.25)))
        noise = np_rng.normal(0, noise_std, n)
        ys = [round(float(y_clean[i] + noise[i]), 2) for i in range(n)]

        # Compute best-fit line (OLS)
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        x_mean = x_arr.mean()
        y_mean = y_arr.mean()
        num = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)))
        den = float(np.sum((x_arr - x_mean) ** 2))
        if abs(den) < 1e-9:
            return None
        fit_slope = num / den
        fit_intercept = y_mean - fit_slope * x_mean

        qtype = cfg["qtype"]
        # Diversity: occasional cross-type injection so models can't shortcut
        # from level -> qtype. Keep dominant qtype for monotonic difficulty.
        if level in (3, 4, 5) and rng.random() < 0.2:
            qtype = "count_above"
        question, correct_letter, options = self._make_qa(
            rng, qtype, xs, ys, fit_slope, fit_intercept, level
        )
        if question is None:
            return None

        image = self._render(rng, xs, ys, fit_slope, fit_intercept,
                             x_min, x_max, grid_step)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Question / options
    # ------------------------------------------------------------------ #

    def _make_qa(self, rng, qtype, xs, ys, fit_slope, fit_intercept, level):
        # reference reasoning_val: GPT-4o-mini judge expects bare numeric/text
        # answer (e.g. `0.5`, `negatively`), not MCQ letter. We removed the
        # 4-option letter wrapping below; answer is just the numeric value.
        n = len(xs)
        if qtype in ("count_above", "count_below"):
            if qtype == "count_above":
                correct = sum(1 for x, y in zip(xs, ys)
                              if y > fit_slope * x + fit_intercept)
            else:
                correct = sum(1 for x, y in zip(xs, ys)
                              if y < fit_slope * x + fit_intercept)
            if qtype == "count_above":
                prompt = rng.choice(_Q_COUNT_TEMPLATES)
            else:
                prompt = ("How many data points lie strictly below the "
                          "best-fit line?")
            q = f"{prompt} Provide just the integer count."
            return q, str(int(correct)), None

        if qtype == "slope":
            correct_val = round(float(fit_slope), 2)
            q = (f"{rng.choice(_Q_SLOPE_TEMPLATES)} Provide just the numeric "
                 f"value (rounded to 2 decimal places).")
            return q, str(correct_val), None

        if qtype == "predict_y":
            x_lo = min(xs)
            x_hi = max(xs)
            x_target = round(rng.uniform(x_lo + (x_hi - x_lo) * 0.2,
                                          x_lo + (x_hi - x_lo) * 0.8), 1)
            pred = fit_slope * x_target + fit_intercept
            correct_disp = round(pred, 1)
            prompt = rng.choice(_Q_PRED_TEMPLATES).format(x=x_target)
            q = (f"{prompt} Provide just the numeric value (rounded to 1 "
                 f"decimal place).")
            return q, str(correct_disp), None

        return None, None, None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, rng, xs, ys, fit_slope, fit_intercept,
                x_min, x_max, grid_step):
        vs = self._random_style()
        palette = list(vs["palette"])
        rng.shuffle(palette)
        x_label = rng.choice(_X_LABELS)
        y_label = rng.choice(_Y_LABELS)
        title = rng.choice(_TITLE_TEMPLATES).format(x=x_label, y=y_label)
        color = palette[0]
        line_color = palette[1] if len(palette) > 1 else "#e63946"
        marker = rng.choice(_MARKER_STYLES)

        fig_w = rng.uniform(6, 8) * vs["figsize_scale"]
        fig_h = rng.uniform(5, 6.5) * vs["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        ax.scatter(xs, ys, c=color, marker=marker,
                   s=rng.choice([30, 40, 50, 60]),
                   alpha=rng.uniform(0.65, 0.9),
                   edgecolors="white", linewidths=0.5, zorder=3)

        # Plot best-fit line across the x-range
        xs_line = np.linspace(x_min, x_max, 50)
        ys_line = fit_slope * xs_line + fit_intercept
        ax.plot(xs_line, ys_line, color=line_color,
                linewidth=rng.choice([1.8, 2.0, 2.5]),
                linestyle=rng.choice(["-", "--"]),
                label="Best-fit line", zorder=2)

        ax.set_xlabel(x_label, fontsize=vs["font_size_base"])
        ax.set_ylabel(y_label, fontsize=vs["font_size_base"])
        ax.set_title(title, fontsize=vs["font_size_base"] + 3, pad=10)

        # Set ticks at explicit grid intervals
        try:
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_locator(mticker.MultipleLocator(grid_step))
            y_min = min(min(ys), fit_slope * x_min + fit_intercept) - 5
            y_max = max(max(ys), fit_slope * x_max + fit_intercept) + 5
            y_range = max(1.0, y_max - y_min)
            # y grid spacing: roughly 8-12 ticks
            y_step = max(1, round(y_range / 10, 1))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(y_step))
        except Exception:
            pass

        self._apply_style(fig, ax, vs)
        ax.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc=vs["legend_loc"], fontsize=vs["font_size_base"] - 1,
                   framealpha=0.85)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vs["dpi"])
