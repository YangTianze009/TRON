"""
Year-axis line chart with a linear trend; ask "in what year did the value
first exceed T?" or, in extrapolation variant, "by what year will it drop
below T at this rate?".

Mirrors reference Factoid IDX 516 ("in which year did the carrot index first
exceed 2,000?"; year-flag YES → exact-match scoring) and Hypothetical IDX 1404
("by what season years will their RPI drop below 50?").

Output: a single 4-digit integer year (e.g. "2009"). Strict equality required —
the base verifier's 5% relative tolerance would silently pass "2016" for
ground truth "2009" (|2016-2009|/2009 = 0.35% < 5%), so this env overrides
`_check_answer` to require exact 4-digit-year match.
"""
import math
import random
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_FIRST_EXCEED = [
    "In which year did the value first exceed {T}? Reply with a 4-digit year (e.g. 2015) in <answer>...</answer>.",
    "In what year did the line first rise above {T}? 4-digit year in <answer>...</answer>.",
    "Find the year in which the value first goes above {T}. 4-digit year in <answer>...</answer>.",
    "Identify the first year on the chart where the value exceeds {T}. 4-digit year in <answer>...</answer>.",
    "When did the line first cross above {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "What is the earliest year shown for which the value is greater than {T}? 4-digit year in <answer>...</answer>.",
    "Read the chart: in what year does the value first exceed {T}? 4-digit year in <answer>...</answer>.",
    "Pinpoint the first year in which the line is above {T}. 4-digit year in <answer>...</answer>.",
    "In which year does the value first surpass {T}? 4-digit year in <answer>...</answer>.",
    "Find the first year on the line chart where the value passes {T}. 4-digit year in <answer>...</answer>.",
    "From the chart, what is the first year where the value exceeds {T}? 4-digit year in <answer>...</answer>.",
    "When does the line first go above {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "Look at the chart and identify the first year above {T}. 4-digit year in <answer>...</answer>.",
    "In what calendar year did the value first cross {T}? 4-digit year in <answer>...</answer>.",
    "What year does the chart first show a value greater than {T}? 4-digit year in <answer>...</answer>.",
    "Find the earliest year for which the chart's value is greater than {T}. 4-digit year in <answer>...</answer>.",
]

_TEMPLATES_FIRST_BELOW = [
    "In which year did the value first drop below {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "In what year did the line first fall under {T}? 4-digit year in <answer>...</answer>.",
    "Find the year in which the value first goes below {T}. 4-digit year in <answer>...</answer>.",
    "Identify the first year on the chart where the value is less than {T}. 4-digit year in <answer>...</answer>.",
    "When did the line first fall below {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "What is the earliest year shown for which the value is less than {T}? 4-digit year in <answer>...</answer>.",
    "Read the chart: in what year does the value first dip below {T}? 4-digit year in <answer>...</answer>.",
    "Pinpoint the first year in which the line is under {T}. 4-digit year in <answer>...</answer>.",
    "In which year does the value first sink below {T}? 4-digit year in <answer>...</answer>.",
    "Find the first year on the line chart where the value drops under {T}. 4-digit year in <answer>...</answer>.",
    "From the chart, what is the first year where the value is below {T}? 4-digit year in <answer>...</answer>.",
    "When does the line first move below {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "Look at the chart and identify the first year below {T}. 4-digit year in <answer>...</answer>.",
    "In what calendar year did the value first fall below {T}? 4-digit year in <answer>...</answer>.",
    "What year does the chart first show a value less than {T}? 4-digit year in <answer>...</answer>.",
    "Find the earliest year for which the chart's value is less than {T}. 4-digit year in <answer>...</answer>.",
]

_TEMPLATES_EXTRAP_EXCEED = [
    "If the trend continues at the same average rate, by what year will the value first exceed {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "Given the linear trend in the chart, by what year would the value first exceed {T}? 4-digit year in <answer>...</answer>.",
    "Assuming the same yearly increase observed in the chart continues, in what year will the value first surpass {T}? 4-digit year in <answer>...</answer>.",
    "If the value continues to rise at the same rate as shown, in which year will it first cross {T}? 4-digit year in <answer>...</answer>.",
    "Extrapolating the linear trend, by what year will the chart's value first reach above {T}? 4-digit year in <answer>...</answer>.",
    "If the trend in the chart continues, what is the first future year in which the value will exceed {T}? 4-digit year in <answer>...</answer>.",
    "Continuing the linear trend, find the first year in which the value will be above {T}. 4-digit year in <answer>...</answer>.",
    "If the rate of change observed continues, in what year will the value first exceed {T}? 4-digit year in <answer>...</answer>.",
]

_TEMPLATES_EXTRAP_BELOW = [
    "If the trend continues at the same average rate, by what year will the value drop below {T}? Reply with a 4-digit year in <answer>...</answer>.",
    "Given the linear trend in the chart, by what year would the value first fall below {T}? 4-digit year in <answer>...</answer>.",
    "Assuming the same yearly decrease observed continues, in what year will the value first dip below {T}? 4-digit year in <answer>...</answer>.",
    "If the value continues to decline at the same rate as shown, in which year will it first cross below {T}? 4-digit year in <answer>...</answer>.",
    "Extrapolating the linear trend, by what year will the chart's value first move below {T}? 4-digit year in <answer>...</answer>.",
    "If the trend in the chart continues, what is the first future year in which the value will be under {T}? 4-digit year in <answer>...</answer>.",
    "Continuing the linear trend, find the first year in which the value will be below {T}. 4-digit year in <answer>...</answer>.",
    "If the rate of decline observed continues, in what year will the value first fall below {T}? 4-digit year in <answer>...</answer>.",
]


_Y_LABELS = ["Index", "Sales (k)", "Revenue", "Visitors (k)", "Users",
             "Cases", "Score", "Volume", "Capacity (GW)"]


class LineChartFirstCrossingYearQA(StandaloneVisualEnv):
    ENV_NAME = "line_chart_first_crossing_year"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Number of plotted years
        if level == 0:
            n = 5
        elif level <= 2:
            n = 6
        elif level <= 4:
            n = 7
        elif level <= 6:
            n = 8
        elif level <= 8:
            n = 10
        else:
            n = 12
        # Mode: historical-only (answer is a year already on the chart) vs
        # extrapolation (answer is a year beyond the chart).
        if level == 0:
            modes = ["historical_exceed"]
        elif level <= 2:
            modes = ["historical_exceed", "historical_below"]
        elif level <= 5:
            modes = ["historical_exceed", "historical_below", "extrap_exceed"]
        else:
            modes = ["historical_exceed", "historical_below",
                     "extrap_exceed", "extrap_below"]
        return {"level": level, "n": n, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1069 + level * 97 + 19)

        n = cfg["n"]
        mode = rng.choice(cfg["modes"])
        # Pick a start year — keep recent decades for realism
        start_year = rng.choice([2005, 2006, 2007, 2008, 2009, 2010,
                                  2011, 2012, 2013, 2014, 2015])
        years = [start_year + i for i in range(n)]
        y_label = rng.choice(_Y_LABELS)

        # L0 trivial: monotonic clearly-rising line, threshold crossed at an
        # obvious year. e.g. years 2010..2014, vals [10,15,20,25,30], T=22 →
        # first exceed in 2013 (value=25)
        if level == 0:
            start_year = 2010
            years = list(range(2010, 2015))
            values = [10, 15, 20, 25, 30]
            T = 22
            # First year value > T
            answer_year = next(y for y, v in zip(years, values) if v > T)
            sidx = (self.seed or 0) % len(_TEMPLATES_FIRST_EXCEED)
            question = _TEMPLATES_FIRST_EXCEED[sidx].format(T=T)
            answer = str(answer_year)
            img = self._render_line(years, values, y_label, T, rng,
                                    extrap_to_year=None)
            return question, answer, img

        if mode in ("historical_exceed", "historical_below"):
            # Build a clean monotonic trend (with mild noise at higher levels)
            # so that the threshold is crossed at exactly one year.
            ascending = (mode == "historical_exceed")
            # Pick base value and per-year step
            if level <= 3:
                step = rng.choice([5, 10])
                v0 = rng.choice([10, 20, 30, 40, 50])
            else:
                step = rng.choice([3, 5, 8, 10, 12])
                v0 = rng.choice([10, 20, 30, 40, 50, 60, 80])
            if ascending:
                values = [v0 + i * step for i in range(n)]
            else:
                v0_top = v0 + (n - 1) * step
                values = [v0_top - i * step for i in range(n)]

            # Optional mild noise — but keep monotonic so first-crossing is
            # well-defined.
            if level >= 5:
                noise = [rng.choice([-1, 0, 1]) for _ in range(n)]
                # Apply only if it preserves monotonicity
                cand = [v + noise[i] for i, v in enumerate(values)]
                if ascending and all(cand[i] < cand[i+1] for i in range(n - 1)):
                    values = cand
                elif (not ascending) and all(cand[i] > cand[i+1] for i in range(n - 1)):
                    values = cand

            # Pick threshold strictly between two consecutive values so cross
            # happens at exactly one year. Prefer a year not at the very ends.
            cross_idx = rng.randrange(1, n)  # crossing year index (>=1)
            if ascending:
                lo, hi = values[cross_idx - 1], values[cross_idx]
                if hi - lo < 2:
                    return None
                T = (lo + hi) // 2
                if T == lo or T == hi:
                    T = lo + 1 if hi - lo >= 2 else None
                if T is None or T in values:
                    return None
                # First year > T
                answer_year = next(y for y, v in zip(years, values) if v > T)
                tpl = _TEMPLATES_FIRST_EXCEED
            else:
                lo, hi = values[cross_idx], values[cross_idx - 1]
                if hi - lo < 2:
                    return None
                T = (lo + hi) // 2
                if T == lo or T == hi:
                    T = hi - 1 if hi - lo >= 2 else None
                if T is None or T in values:
                    return None
                answer_year = next(y for y, v in zip(years, values) if v < T)
                tpl = _TEMPLATES_FIRST_BELOW
            sidx = (self.seed or 0) % len(tpl)
            question = tpl[sidx].format(T=T)
            answer = str(answer_year)
            img = self._render_line(years, values, y_label, T, rng,
                                    extrap_to_year=None)
            return question, answer, img

        # Extrapolation modes — no point on the historical chart crosses T;
        # answer is the first year beyond the chart at which the linear trend
        # crosses T.
        ascending = (mode == "extrap_exceed")
        # Build a clean linear trend
        if level <= 6:
            step = rng.choice([5, 10])
        else:
            step = rng.choice([3, 5, 8, 10])
        v0 = rng.choice([10, 20, 30, 40, 50])
        if ascending:
            values = [v0 + i * step for i in range(n)]
            v_last = values[-1]
            # Choose T strictly above v_last so future year crossing well-defined
            # Want answer year = years[-1] + k for an integer k>=1.
            k = rng.choice([1, 2, 3, 4, 5])
            # T such that v_last + (k-1)*step <= T < v_last + k*step
            T_low = v_last + (k - 1) * step + 1
            T_high = v_last + k * step
            if T_low > T_high:
                return None
            T = rng.randint(T_low, T_high)
            answer_year = years[-1] + k
            tpl = _TEMPLATES_EXTRAP_EXCEED
        else:
            # Descending
            v_top = v0 + (n - 1) * step
            values = [v_top - i * step for i in range(n)]
            v_last = values[-1]
            k = rng.choice([1, 2, 3, 4, 5])
            # Want T strictly below v_last; answer year v_last - k*step < T <= v_last - (k-1)*step
            T_high = v_last - (k - 1) * step - 1
            T_low = v_last - k * step
            if T_low > T_high:
                return None
            T = rng.randint(T_low, T_high)
            answer_year = years[-1] + k
            tpl = _TEMPLATES_EXTRAP_BELOW

        sidx = (self.seed or 0) % len(tpl)
        question = tpl[sidx].format(T=T)
        answer = str(answer_year)
        img = self._render_line(years, values, y_label, T, rng,
                                extrap_to_year=answer_year + 1)
        return question, answer, img

    def _render_line(self, years, values, y_label, T, rng, extrap_to_year):
        palette = rng.choice(self._COLOR_PALETTES)
        line_color = palette[0]
        bg = "#ffffff"
        n = len(years)
        fig_w = max(7.0, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.plot(years, values, "-o", color=line_color, linewidth=2.4,
                markersize=8, markerfacecolor=line_color,
                markeredgecolor="#1a1a1a")
        max_v = max(values + [T])
        for y, v in zip(years, values):
            ax.text(y, v + max_v * 0.025, str(v),
                    ha="center", va="bottom", fontsize=10,
                    color="#111", fontweight="bold")
        ax.axhline(T, color="#c0392b", linewidth=1.6, linestyle="--",
                   label=f"y = {T}")
        # Year ticks: only every year visible
        if extrap_to_year is not None and extrap_to_year > years[-1]:
            ax.set_xlim(years[0] - 0.5, extrap_to_year + 0.5)
            ax.set_xticks(list(range(years[0], extrap_to_year + 1)))
            ax.set_xticklabels([str(y) for y in range(years[0], extrap_to_year + 1)],
                               rotation=20 if (extrap_to_year - years[0] + 1) > 6 else 0,
                               ha="right" if (extrap_to_year - years[0] + 1) > 6 else "center",
                               fontsize=10)
        else:
            ax.set_xticks(years)
            ax.set_xticklabels([str(y) for y in years],
                               rotation=20 if n > 6 else 0,
                               ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        y_lo = min(min(values), T) - max_v * 0.10 - 5
        y_hi = max(max(values), T) + max_v * 0.18 + 5
        ax.set_ylim(y_lo, y_hi)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict 4-digit-year match. The base verifier's 5% relative tolerance
        would silently pass nearby wrong years (e.g. 2016 vs GT 2009), which
        is exactly the year-flag failure mode (reference IDX 516). So we
        only accept exact match on the first 4-digit number found in the
        prediction."""
        gt = ground_truth.strip()
        if not (len(gt) == 4 and gt.isdigit()):
            # fallback to base if gt format unexpected
            return super()._check_answer(predicted, ground_truth)
        m = re.search(r"\b(\d{4})\b", predicted)
        if not m:
            return False
        return m.group(1) == gt
