"""
Chart linear extrapolation forecast.

Render a time-series line chart, then ask:
  - "If the trend continues at the same pace, what value at year T?" (numeric)
  - "When will X reach Y?" (year as 4-digit string or period label)

Chart shows N labeled points; model must compute slope from observed points
and extrapolate forward.

Mirrors Hypothetical reference samples:
  - IDX 1416 ("if trend continued, support level in 2028?" -> 10%)
  - IDX 1399 ("if trend continued, food waste by 2020?" -> 6.146)
  - IDX 1404 ("by what season will rpi drop below 50?" -> 2011-12)
  - IDX 1401 ("if etf continues at march pace, surpass within two weeks?" -> Yes)
  - IDX 1782 ("Norway EV CAGR forecast" - linear extrapolation)

Output: numeric value or 4-digit year.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter


_TEMPLATES_VALUE_AT_YEAR = [
    "Look at the line chart. If the trend continues at the same pace, what would the value be in {year}? Place the numeric answer in <answer>...</answer>.",
    "Assuming a linear trend continues, what is the expected value in {year}? Numeric answer in <answer>...</answer>.",
    "If the line in the chart continues at its current pace, what would the value be in {year}? Numeric in <answer>...</answer>.",
    "Extrapolate the chart linearly. What is the predicted value in {year}? Numeric in <answer>...</answer>.",
    "Assuming the same average rate shown in the chart continues, what would the value reach by {year}? Numeric in <answer>...</answer>.",
    "If the pace of change in the chart held steady, what value would be observed in {year}? Numeric in <answer>...</answer>.",
    "Linearly extrapolate the trend in the chart to {year}. What is the value? Numeric in <answer>...</answer>.",
    "Continue the line at the current rate. What is the expected value in {year}? Numeric in <answer>...</answer>.",
]

_TEMPLATES_VALUE_AT_PERIOD = [
    "Look at the line chart. If the trend continues at the same pace, what would the value be at {period}? Place the numeric answer in <answer>...</answer>.",
    "Assuming a linear trend continues, what is the expected value at {period}? Numeric in <answer>...</answer>.",
    "If the line continues at its current pace, what would the value be at {period}? Numeric in <answer>...</answer>.",
    "Extrapolate the chart linearly to {period}. What is the value? Numeric in <answer>...</answer>.",
    "Continue the line at the current rate. What is the expected value at {period}? Numeric in <answer>...</answer>.",
]

_TEMPLATES_WHEN_REACH = [
    "Look at the line chart. If the trend continues at the same pace, in which year will the value first reach {target}? Reply with the 4-digit year in <answer>...</answer>.",
    "Assuming the linear trend continues, in what year will the value reach {target}? Reply with the year in <answer>...</answer>.",
    "If the line in the chart continues at its current pace, in which year will it first hit {target}? 4-digit year in <answer>...</answer>.",
    "Extrapolate the trend linearly. In what year will the value first reach {target}? Year in <answer>...</answer>.",
    "Continuing at the current rate, when (which year) will the value reach {target}? 4-digit year in <answer>...</answer>.",
]


_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Index", "Rate", "Capacity"]


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class ChartLinearForecastQA(StandaloneVisualEnv):
    ENV_NAME = "chart_linear_forecast"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            # 3 points showing slope of 5/year, "value at year+2" -> trivial
            return {"n_points": 3, "modes": ["value_at_year"], "trivial_l0": True,
                    "extrap_dist": 2, "noise": 0}
        if level <= 2:
            return {"n_points": 4, "modes": ["value_at_year"],
                    "trivial_l0": False, "extrap_dist": 2, "noise": 0}
        if level <= 4:
            return {"n_points": 5, "modes": ["value_at_year", "when_reach"],
                    "trivial_l0": False, "extrap_dist": 3, "noise": 1}
        if level <= 6:
            return {"n_points": 5, "modes": ["value_at_year", "when_reach"],
                    "trivial_l0": False, "extrap_dist": 4, "noise": 2}
        if level <= 8:
            return {"n_points": 6, "modes": ["value_at_year", "when_reach"],
                    "trivial_l0": False, "extrap_dist": 5, "noise": 3}
        return {"n_points": 7, "modes": ["value_at_year", "when_reach"],
                "trivial_l0": False, "extrap_dist": 6, "noise": 3}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1487 + level * 149 + 53)

        n = cfg["n_points"]

        if cfg["trivial_l0"]:
            # 3 points: 2020->10, 2021->15, 2022->20 (slope=5).
            # Ask "value at 2024?" -> 30.
            start_year = 2020
            slope = 5
            base = 10
            years = [start_year + i for i in range(n)]
            values = [base + slope * i for i in range(n)]
            target_year = years[-1] + cfg["extrap_dist"]
            answer_val = base + slope * (target_year - start_year)
            answer = _format_num(answer_val)
            tlist = _TEMPLATES_VALUE_AT_YEAR
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(year=target_year)
            y_label = rng.choice(_Y_LABELS)
            img = self._render_line(years, values, y_label, rng)
            # MCQ-letter style MCQ-letter: with prob 0.5, convert to MCQ.
            n_opts = rng.choice([4, 5])
            question, answer = maybe_to_mcq_letter(
                question, answer, rng, prob=0.5, n_options=n_opts)
            return question, answer, img

        # General case
        start_year = rng.choice([2010, 2011, 2012, 2013, 2014, 2015, 2016])
        years = [start_year + i for i in range(n)]
        slope = rng.choice([-12, -10, -8, -6, -5, -4, -3, 3, 4, 5, 6, 8, 10, 12])
        base = rng.randint(40, 200)
        if slope < 0:
            # Make sure it doesn't go negative within the chart
            base = max(base, abs(slope) * (n - 1) + 40)
        # Add small zero-mean noise; keep linear trend dominant
        noise_amp = cfg["noise"]
        values = []
        for i in range(n):
            v = base + slope * i + (rng.uniform(-noise_amp, noise_amp) if noise_amp else 0)
            values.append(round(v, 1))

        mode = rng.choice(cfg["modes"])
        if mode == "value_at_year":
            target_year = years[-1] + cfg["extrap_dist"]
            answer_val = base + slope * (target_year - start_year)
            answer = _format_num(answer_val)
            tlist = _TEMPLATES_VALUE_AT_YEAR
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(year=target_year)
        else:
            # when_reach: pick a future target value that is k steps ahead
            extrap = cfg["extrap_dist"]
            target_idx_offset = extrap
            target_val = base + slope * (n - 1 + target_idx_offset)
            # Round target to multiple of 5 to make question clean
            if slope > 0:
                target_val = math.ceil(target_val / 5.0) * 5
            else:
                target_val = math.floor(target_val / 5.0) * 5
            # Solve: base + slope*(year - start_year) = target_val
            # year = start_year + (target_val - base) / slope (rounded up if slope>0)
            if slope == 0:
                return None
            steps_needed = (target_val - base) / slope
            if slope > 0:
                steps_needed = math.ceil(steps_needed)
            else:
                steps_needed = math.ceil(steps_needed)
            target_year = start_year + steps_needed
            # Sanity: target year must be after the last shown year
            if target_year <= years[-1]:
                # Bump target value
                if slope > 0:
                    target_val = base + slope * (n - 1 + extrap)
                    target_val = math.ceil(target_val / 5.0) * 5
                else:
                    target_val = base + slope * (n - 1 + extrap)
                    target_val = math.floor(target_val / 5.0) * 5
                steps_needed = (target_val - base) / slope
                steps_needed = math.ceil(steps_needed)
                target_year = start_year + steps_needed
            if target_year <= years[-1]:
                return None
            answer = str(int(target_year))
            tlist = _TEMPLATES_WHEN_REACH
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(target=int(target_val))

        y_label = rng.choice(_Y_LABELS)
        img = self._render_line(years, values, y_label, rng)
        # MCQ-letter style MCQ-letter: with prob 0.5, convert to MCQ.
        n_opts = rng.choice([4, 5])
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts)
        return question, answer, img

    def _render_line(self, years: List[int], values: List[float],
                     y_label: str, rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = list(range(len(years)))
        line_color = palette[0]
        ax.plot(x, values, marker="o", linewidth=2.2, color=line_color,
                markersize=8)
        max_v = max(values)
        min_v = min(values)
        rng_v = max(1.0, max_v - min_v)
        for xi, vi in zip(x, values):
            ax.text(xi, vi + rng_v * 0.04, str(int(vi) if abs(vi - int(vi)) < 1e-6 else f"{vi:.1f}"),
                    ha="center", va="bottom", fontsize=10, color="#111",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlabel("Year", fontsize=10)
        # y limits
        pad = max(rng_v * 0.20, 5)
        ax.set_ylim(min_v - pad, max_v + pad * 1.5)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add a faint trend annotation line connecting first and last points
        ax.plot([x[0], x[-1]], [values[0], values[-1]], "--",
                color=line_color, alpha=0.35, linewidth=1.2)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
