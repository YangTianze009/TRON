"""
Line Chart Visual QA Environment.

Generates line charts with one or more series and asks data-reading /
computation questions.  Answers are computed from the raw data.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

from .base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

# ------------------------------------------------------------------ #
# X-axis label pools (time-like)
# ------------------------------------------------------------------ #

def _year_labels(n: int, rng: random.Random) -> List[str]:
    start = rng.randint(2000, 2025 - n)
    return [str(start + i) for i in range(n)]

def _month_labels(n: int, rng: random.Random) -> List[str]:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    start = rng.randint(0, max(0, 12 - n))
    return months[start:start + n]

def _quarter_labels(n: int, rng: random.Random) -> List[str]:
    start_year = rng.randint(2018, 2024)
    labels = []
    q, y = 1, start_year
    for _ in range(n):
        labels.append(f"Q{q} {y}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return labels

_X_GENERATORS = [_year_labels, _month_labels, _quarter_labels]

_SERIES_NAME_POOLS = [
    ["Revenue", "Profit", "Expenses", "Investments"],
    ["Company A", "Company B", "Company C"],
    ["Product X", "Product Y", "Product Z"],
    ["Domestic", "International", "Online"],
    ["Temp (C)", "Humidity (%)", "Wind (km/h)"],
    ["Group 1", "Group 2", "Group 3"],
]

_Y_LABELS = [
    "Value", "Revenue ($M)", "Units", "Temperature (C)",
    "Score", "Index", "Growth (%)", "Count (thousands)",
]

_TITLE_TEMPLATES = [
    "{y_label} Over Time",
    "Trend of {y_label}",
    "{y_label} — Time Series",
    "Historical {y_label}",
]

_COLOR_PALETTES = [
    ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"],
    ["#003f5c", "#bc5090", "#ffa600", "#58508d"],
]

_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]

class LineChartQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.05
    ENV_NAME = "line_chart"

    def _level_config(self, level: int) -> dict:
        """Difficulty redesign 2026-04-14. L0 starts at ~L4 difficulty."""
        configs = {
            0: {'question_type': 'average', 'num_points': 7, 'num_series': 1, 'no_labels': False},
            1: {'question_type': 'max_change', 'num_points': 8, 'num_series': 1, 'no_labels': False},
            2: {'question_type': 'consecutive_increase', 'num_points': 8, 'num_series': 1, 'no_labels': False},
            3: {'question_type': 'volatility', 'num_points': 8, 'num_series': 2, 'no_labels': False},
            4: {'question_type': 'sum_range', 'num_points': 10, 'num_series': 2, 'no_labels': False},
            5: {'question_type': 'ultra_hard_cumulative_sum', 'num_points': 10, 'num_series': 1, 'no_labels': False},
            6: {'question_type': 'ultra_hard_steepest_decline_after_peak', 'num_points': 10, 'num_series': 1, 'no_labels': False},
            # L7-L9 use weighted-avg (exact value required) — keep labels
            # visible so task stays solvable from image.
            7: {'question_type': 'ultra_hard_weighted_avg_by_position', 'num_points': 12, 'num_series': 1, 'no_labels': False},
            8: {'question_type': 'ultra_hard_weighted_avg_by_position', 'num_points': 14, 'num_series': 2, 'no_labels': False},
            9: {'question_type': 'ultra_hard_weighted_avg_by_position', 'num_points': 16, 'num_series': 2, 'no_labels': False},
        }
        return configs.get(max(0, min(9, level)), configs[0])

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        # ── Level routing (auto-generated) ──
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)

        rng = self._rng
        np_rng = np.random.RandomState(seed)

        num_points = parameter.get("num_points", 8)
        num_series = parameter.get("num_series", 1)
        question_type = parameter.get("question_type", "trend")

        # X-axis labels — clamp to what generators can produce
        if num_points > 12:
            # Only year and quarter generators can handle >12 points
            gen = rng.choice([_year_labels, _quarter_labels])
        else:
            gen = rng.choice(_X_GENERATORS)
        x_labels = gen(num_points, rng)

        # Series names
        if num_series == 1:
            spool = rng.choice(_SERIES_NAME_POOLS)
            series_names = [rng.choice(spool)]
        else:
            spool = list(rng.choice(_SERIES_NAME_POOLS))
            rng.shuffle(spool)
            series_names = spool[:num_series]

        # Generate data — random-walk style with rounding to int
        values: List[List[int]] = []
        for _ in range(num_series):
            base = rng.randint(20, 200)
            step_std = rng.uniform(5, 30)
            pts = [base]
            for _ in range(num_points - 1):
                delta = np_rng.normal(0, step_std)
                pts.append(max(0, int(round(pts[-1] + delta))))
            values.append(pts)

        # Determine if no_labels mode (ultra-hard: hide data point values)
        no_labels = parameter.get("no_labels", False)

        # Build QA
        question, answer = self._make_qa(
            rng, question_type, x_labels, series_names, values
        )
        if question is None:
            return None

        image = self._render_chart(rng, x_labels, series_names, values,
                                    no_labels=no_labels)

        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, rate=0.30)
        if wrapped is not None:
            question, answer = wrapped
        return question, answer, image

    # ------------------------------------------------------------------ #
    # QA
    # ------------------------------------------------------------------ #

    def _make_qa(
        self,
        rng: random.Random,
        qtype: str,
        x_labels: List[str],
        series_names: List[str],
        values: List[List[int]],
    ) -> Tuple[Optional[str], Optional[str]]:
        n_pts = len(x_labels)
        n_ser = len(values)
        si = rng.randint(0, n_ser - 1) if n_ser > 1 else 0
        vals = values[si]
        sname = series_names[si]

        def _for_series():
            return f" for {sname}" if n_ser > 1 else ""

        if qtype == "max_point":
            idx = vals.index(max(vals))
            q = f"At which point does the value reach its maximum{_for_series()}?"
            a = x_labels[idx]

        elif qtype == "min_point":
            idx = vals.index(min(vals))
            q = f"At which point is the value lowest{_for_series()}?"
            a = x_labels[idx]

        elif qtype == "value_at":
            ci = rng.randint(0, n_pts - 1)
            q = f"What is the value at {x_labels[ci]}{_for_series()}?"
            a = str(vals[ci])

        elif qtype == "increase_decrease":
            i1, i2 = sorted(rng.sample(range(n_pts), 2))
            v1, v2 = vals[i1], vals[i2]
            direction = "increase" if v2 >= v1 else "decrease"
            q = (f"Between {x_labels[i1]} and {x_labels[i2]}, "
                 f"did the value increase or decrease{_for_series()}?")
            a = direction

        elif qtype == "max_change":
            changes = [abs(vals[i + 1] - vals[i]) for i in range(n_pts - 1)]
            max_c = max(changes)
            q = f"What is the largest single-step change in absolute value{_for_series()}?"
            a = str(max_c)

        elif qtype == "trend_direction":
            # Overall from first to last
            direction = "increase" if vals[-1] >= vals[0] else "decrease"
            q = f"Overall, does the series show an increase or decrease from start to end{_for_series()}?"
            a = direction

        elif qtype == "crossover" or qtype == "crossover_count":
            if n_ser < 2:
                # Fall back
                idx = vals.index(max(vals))
                q = "At which point does the value reach its maximum?"
                a = x_labels[idx]
            else:
                # Find first crossover between two series
                s1, s2 = rng.sample(range(n_ser), 2)
                v1, v2 = values[s1], values[s2]
                cross_idx = None
                for i in range(1, n_pts):
                    if (v1[i - 1] - v2[i - 1]) * (v1[i] - v2[i]) < 0:
                        cross_idx = i
                        break
                if cross_idx is not None:
                    q = (f"At which point do {series_names[s1]} and "
                         f"{series_names[s2]} first cross?")
                    a = x_labels[cross_idx]
                else:
                    # No crossover — fallback to value_at
                    ci = rng.randint(0, n_pts - 1)
                    q = f"What is the value at {x_labels[ci]} for {series_names[s1]}?"
                    a = str(v1[ci])

        elif qtype == "trend" or qtype == "trend_direction":
            # Overall from first to last
            direction = "increase" if vals[-1] >= vals[0] else "decrease"
            q = f"Overall, does the series show an increase or decrease from start to end{_for_series()}?"
            a = direction

        elif qtype == "average":
            avg = round(sum(vals) / len(vals), 1)
            q = f"What is the average (mean) of all values{_for_series()}? Round to 1 decimal place."
            a = str(avg)

        elif qtype == "range":
            r = max(vals) - min(vals)
            q = f"What is the range (max minus min) of the values{_for_series()}?"
            a = str(r)

        elif qtype == "growth_rate":
            if vals[0] == 0:
                return None, None
            rate = round((vals[-1] - vals[0]) / vals[0] * 100, 1)
            q = (f"What is the overall growth rate from the first to the last point{_for_series()}? "
                 f"Compute as (last - first) / first × 100. Round to 1 decimal place.")
            a = str(rate)

        elif qtype == "consecutive_increase":
            # Longest streak of consecutive increases
            max_streak = 0
            cur_streak = 0
            for i in range(1, n_pts):
                if vals[i] > vals[i - 1]:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 0
            q = (f"What is the longest streak of consecutive increases "
                 f"(each point higher than the previous){_for_series()}?")
            a = str(max_streak)

        elif qtype == "volatility":
            # Standard deviation of step changes
            changes = [vals[i + 1] - vals[i] for i in range(n_pts - 1)]
            mean_c = sum(changes) / len(changes)
            var_c = sum((c - mean_c) ** 2 for c in changes) / len(changes)
            vol = round(var_c ** 0.5, 1)
            q = (f"Compute the volatility{_for_series()}: the standard deviation of "
                 f"all step-to-step changes. Round to 1 decimal place.")
            a = str(vol)

        elif qtype == "peak_trough_diff":
            # Difference between the largest local max and smallest local min
            peaks = [vals[i] for i in range(1, n_pts - 1)
                     if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]]
            troughs = [vals[i] for i in range(1, n_pts - 1)
                       if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]]
            if not peaks or not troughs:
                # Fall back to range
                r = max(vals) - min(vals)
                q = f"What is the range (max minus min) of the values{_for_series()}?"
                a = str(r)
            else:
                diff = max(peaks) - min(troughs)
                q = (f"Find the highest local peak and the lowest local trough in the chart"
                     f"{_for_series()}. What is the difference between them?")
                a = str(diff)

        elif qtype == "max_drop":
            # Largest single-step decrease
            drops = [vals[i] - vals[i + 1] for i in range(n_pts - 1)]
            max_d = max(drops) if drops else 0
            if max_d <= 0:
                # No drops — fall back
                max_d = max(abs(d) for d in drops) if drops else 0
                q = f"What is the largest single-step change in absolute value{_for_series()}?"
                a = str(max_d)
            else:
                q = f"What is the largest single-step drop (decrease) in the chart{_for_series()}?"
                a = str(max_d)

        elif qtype == "sum_range":
            # Sum of all values in a sub-range
            if n_pts < 4:
                return None, None
            i1 = rng.randint(0, n_pts // 2 - 1)
            i2 = rng.randint(n_pts // 2, n_pts - 1)
            s = sum(vals[i1:i2 + 1])
            q = (f"What is the sum of all values from {x_labels[i1]} to "
                 f"{x_labels[i2]} (inclusive){_for_series()}?")
            a = str(s)

        # ---- Ultra-hard question types ----

        elif qtype == "ultra_hard_interpolation":
            # Ask what the estimated value is halfway between two labeled points
            if n_pts < 3:
                return None, None
            i1 = rng.randint(0, n_pts - 2)
            i2 = i1 + 1
            v1, v2 = vals[i1], vals[i2]
            midval = round((v1 + v2) / 2, 1)
            q = (f"Estimate the value at the midpoint between {x_labels[i1]} "
                 f"and {x_labels[i2]}{_for_series()} using linear interpolation. "
                 f"Round to 1 decimal place.")
            a = str(midval)

        elif qtype == "ultra_hard_steepest_decline_after_peak":
            # Find the steepest single-step decline that occurs after the first peak
            # Step 1: find first local max. Step 2: find max drop after it.
            if n_pts < 4:
                return None, None
            # Find first peak (local max)
            peak_idx = None
            for i in range(1, n_pts - 1):
                if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1] and vals[i] > vals[i + 1]:
                    peak_idx = i
                    break
            if peak_idx is None or peak_idx >= n_pts - 2:
                return None, None
            # Find steepest decline after peak
            max_drop = 0
            drop_start = peak_idx
            for i in range(peak_idx, n_pts - 1):
                drop = vals[i] - vals[i + 1]
                if drop > max_drop:
                    max_drop = drop
                    drop_start = i
            if max_drop <= 0:
                return None, None
            q = (f"First find the first peak (local maximum) in the chart{_for_series()}. "
                 f"Then, looking only at the data after that peak, what is the "
                 f"largest single-step decline?")
            a = str(max_drop)

        elif qtype == "ultra_hard_cumulative_sum":
            # Compute the cumulative (running) sum and report its value at a specific point
            if n_pts < 4:
                return None, None
            target_idx = rng.randint(2, n_pts - 1)
            cum_sum = sum(vals[:target_idx + 1])
            q = (f"Compute the cumulative sum of all values from the first point "
                 f"through {x_labels[target_idx]}{_for_series()} (inclusive). "
                 f"What is the total?")
            a = str(cum_sum)

        elif qtype == "ultra_hard_weighted_avg_by_position":
            # Weighted average: weight each value by its 1-based position index
            if n_pts < 4:
                return None, None
            weighted_sum = sum((i + 1) * v for i, v in enumerate(vals))
            weight_total = n_pts * (n_pts + 1) // 2
            wavg = round(weighted_sum / weight_total, 2)
            q = (f"Compute the position-weighted average{_for_series()}: "
                 f"weight the 1st point by 1, the 2nd by 2, etc. up to "
                 f"the {n_pts}th point weighted by {n_pts}. "
                 f"Divide the weighted sum by the sum of weights. "
                 f"Round to 2 decimal places.")
            a = str(wavg)

        else:
            idx = vals.index(max(vals))
            q = f"At which point does the value reach its maximum{_for_series()}?"
            a = x_labels[idx]

        return q, a

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_chart(
        self,
        rng: random.Random,
        x_labels: List[str],
        series_names: List[str],
        values: List[List[int]],
        no_labels: bool = False,
    ) -> Image.Image:
        vs = self._random_style()
        palette = list(vs["palette"])
        markers = list(_MARKERS)
        rng.shuffle(markers)

        y_label = rng.choice(_Y_LABELS)
        title = rng.choice(_TITLE_TEMPLATES).format(y_label=y_label)

        n_pts = len(x_labels)
        n_ser = len(values)

        fig_w = max(6, n_pts * 0.6 + 2) * vs["figsize_scale"]
        fig_h = rng.uniform(4.5, 6.0) * vs["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        label_fs = vs["font_size_base"]
        title_fs = label_fs + 3
        font_family = vs["font_family"]
        lw = vs["line_width"]

        x = np.arange(n_pts)

        for s in range(n_ser):
            color = palette[s % len(palette)]
            marker = markers[s % len(markers)]
            ax.plot(x, values[s], color=color, marker=marker,
                    linewidth=lw, markersize=rng.randint(5, 8),
                    label=series_names[s])

        ax.set_xticks(x)
        rotate = n_pts > 8 or max(len(l) for l in x_labels) > 5
        ax.set_xticklabels(x_labels, rotation=45 if rotate else 0,
                           ha="right" if rotate else "center",
                           fontsize=label_fs, fontfamily=font_family)
        ax.set_ylabel(y_label, fontsize=label_fs, fontfamily=font_family)
        ax.set_title(title, fontsize=title_fs, pad=10, fontfamily=font_family)

        if n_ser > 1:
            ax.legend(fontsize=label_fs - 1, framealpha=0.9,
                      loc=vs["legend_loc"])

        # Apply style (bg, grid, spines)
        self._apply_style(fig, ax, vs)
        ax.set_axisbelow(True)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if no_labels:
            # Remove y-axis tick labels to force reading from gridlines only
            ax.set_yticklabels([])
            ax.set_ylabel("")

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vs["dpi"])
