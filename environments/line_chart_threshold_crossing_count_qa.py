"""
Line-chart threshold crossing: render a time-series line over N=4-12 periods
(months / quarters / years), then ask "in how many periods did the value
exceed T?" — or, in some variants, "did the value ever cross T?" (yes/no).

Mirrors reference Fact-Checking IDX 1568 ("federal deficits ... exceeded $400
billion five times between 2003 and 2010"), IDX 1589 ("number of deaths per
week exceeded 10 on only one occasion in 2021"), and Factoid threshold-counting
patterns. The answer is either an integer count (default) or "yes" / "no" for
a binary "ever exceeded" question.
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


_TEMPLATES_COUNT = [
    "In how many {period_unit} did the value exceed {T}? Reply with a non-negative integer in <answer>...</answer>.",
    "Count the number of {period_unit} whose value is greater than {T}. Integer in <answer>...</answer>.",
    "How many {period_unit} on the chart show a value above {T}? Integer count in <answer>...</answer>.",
    "How many times did the value rise above {T}? Integer count in <answer>...</answer>.",
    "Count how many data points on the line are higher than {T}. Integer in <answer>...</answer>.",
    "How many {period_unit} have a value strictly greater than {T}? Integer in <answer>...</answer>.",
    "How many points on the line chart exceed {T}? Integer in <answer>...</answer>.",
    "Tally the {period_unit} during which the value exceeded {T}. Integer count in <answer>...</answer>.",
    "Count the {period_unit} above {T} on the line chart. Integer in <answer>...</answer>.",
    "How many {period_unit} had a reading above {T}? Integer in <answer>...</answer>.",
    "Determine how many {period_unit} the value was greater than {T}. Integer in <answer>...</answer>.",
    "How many of the plotted {period_unit} exceed {T}? Integer in <answer>...</answer>.",
    "Tally the number of plotted points above {T}. Integer in <answer>...</answer>.",
    "Count the points where the line is above {T}. Integer answer in <answer>...</answer>.",
    "In how many of the {period_unit} shown was the value greater than {T}? Integer in <answer>...</answer>.",
    "From the chart, count the {period_unit} above {T}. Integer in <answer>...</answer>.",
]

_TEMPLATES_BELOW_COUNT = [
    "In how many {period_unit} did the value fall below {T}? Reply with a non-negative integer in <answer>...</answer>.",
    "Count the number of {period_unit} whose value is less than {T}. Integer in <answer>...</answer>.",
    "How many {period_unit} on the chart show a value under {T}? Integer count in <answer>...</answer>.",
    "How many data points on the line are below {T}? Integer in <answer>...</answer>.",
    "How many {period_unit} have a value strictly less than {T}? Integer in <answer>...</answer>.",
    "How many points on the line chart are under {T}? Integer in <answer>...</answer>.",
    "Tally the {period_unit} during which the value was less than {T}. Integer count in <answer>...</answer>.",
    "Count the {period_unit} below {T} on the line chart. Integer in <answer>...</answer>.",
    "How many {period_unit} had a reading below {T}? Integer in <answer>...</answer>.",
    "Determine how many {period_unit} the value was less than {T}. Integer in <answer>...</answer>.",
    "How many of the plotted {period_unit} are below {T}? Integer in <answer>...</answer>.",
    "Tally the number of plotted points under {T}. Integer in <answer>...</answer>.",
    "Count the points where the line is below {T}. Integer answer in <answer>...</answer>.",
    "In how many of the {period_unit} shown was the value less than {T}? Integer in <answer>...</answer>.",
    "From the chart, count the {period_unit} below {T}. Integer in <answer>...</answer>.",
    "How many {period_unit} are below the threshold {T}? Integer in <answer>...</answer>.",
]

_TEMPLATES_YESNO_EVER = [
    "Did the value ever exceed {T}? Reply with `yes` or `no` in <answer>...</answer>.",
    "Did the line cross above {T} at any point? Single word `yes` or `no` in <answer>...</answer>.",
    "Has the value gone above {T} at any time on the chart? Reply `yes` or `no` in <answer>...</answer>.",
    "Was there at least one period where the value exceeded {T}? `yes` or `no` in <answer>...</answer>.",
    "Did the line ever rise above {T}? Lowercase `yes` or `no` in <answer>...</answer>.",
    "At any point in the time series, did the value go above {T}? `yes` / `no` in <answer>...</answer>.",
    "Does any point on the line exceed {T}? `yes` or `no` in <answer>...</answer>.",
    "Did the chart show any reading above {T}? `yes` or `no` in <answer>...</answer>.",
    "Does the value cross above {T} at any time? Reply with `yes` or `no` in <answer>...</answer>.",
    "Was {T} ever exceeded by the line? `yes` / `no` in <answer>...</answer>.",
    "Did the time series ever show a value greater than {T}? `yes` or `no` in <answer>...</answer>.",
    "At any point, did the value pass {T}? `yes` or `no` in <answer>...</answer>.",
    "Did any of the plotted points lie above {T}? `yes` or `no` in <answer>...</answer>.",
    "Was the line ever above the {T} mark? Reply `yes` or `no` in <answer>...</answer>.",
    "Did the chart's line exceed {T} at least once? `yes` or `no` in <answer>...</answer>.",
    "Has the value ever been higher than {T}? `yes` or `no` in <answer>...</answer>.",
]


def _month_labels(n):
    base = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return base[:n]

def _quarter_labels(n):
    return [f"Q{i+1}" for i in range(n)]

def _year_labels(n, start=2010):
    return [str(start + i) for i in range(n)]


_Y_LABELS = ["Sales (k)", "Revenue", "Visitors (k)", "Users", "Cases",
             "Index", "Score", "Volume"]


class LineChartThresholdCrossingCountQA(StandaloneVisualEnv):
    ENV_NAME = "line_chart_threshold_crossing_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04 R3 retry: shrink n across mid-high tiers — VLM
        # threshold-counting at n=10/12 was eating accuracy (v5=0.70).
        if level == 0:
            n = 4
        elif level <= 2:
            n = 5
        elif level <= 4:
            n = 6
        elif level <= 6:
            n = 7
        elif level <= 8:
            n = 9
        else:
            n = 10
        if level == 0:
            modes = ["count"]
        elif level <= 2:
            modes = ["count", "yesno"]
        else:
            modes = ["count", "below_count", "yesno"]
        return {"level": level, "n": n, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1063 + level * 89 + 17)

        n = cfg["n"]
        mode = rng.choice(cfg["modes"])
        # Choose period style — for L0 force months for clarity
        if level == 0:
            period_kind = "months"
        else:
            period_kind = rng.choice(["months", "quarters", "years"])
        if period_kind == "months":
            x_labels = _month_labels(n)
            period_unit = "months"
        elif period_kind == "quarters":
            x_labels = _quarter_labels(n)
            period_unit = "quarters"
        else:
            start = rng.choice([2010, 2012, 2014, 2015, 2016, 2017])
            x_labels = _year_labels(n, start)
            period_unit = "years"

        y_label = rng.choice(_Y_LABELS)

        # L0 trivial: 4-month line that clearly crosses T once.
        # values [10, 20, 50, 30], T=40 → only one (50) above T
        if level == 0:
            values = [10, 20, 50, 30]
            T = 40
            count = sum(1 for v in values if v > T)  # = 1
            sidx = (self.seed or 0) % len(_TEMPLATES_COUNT)
            question = _TEMPLATES_COUNT[sidx].format(T=T, period_unit=period_unit)
            answer = str(count)
            img = self._render_line(x_labels, values, y_label, T, rng)
            return question, answer, img

        # Generate values that yield a non-trivial count vs threshold.
        if level <= 3:
            v_min, v_max, step = 10, 100, 5
        elif level <= 6:
            v_min, v_max, step = 10, 200, 5
        else:
            # 2026-05-04 R3: softened — step 1 → 5 (less noise around T,
            # answer remains discrete-readable from chart).
            v_min, v_max, step = 5, 250, 5

        values = None
        for _ in range(60):
            cand = [rng.choice(list(range(v_min, v_max + 1, step))) for _ in range(n)]
            if max(cand) - min(cand) >= 4 * step:
                values = cand
                break
        if values is None:
            return None

        if mode == "yesno":
            # Decide yes or no by adjusting T relative to max
            actually_yes = rng.random() < 0.5
            mx, mn = max(values), min(values)
            if actually_yes:
                # T strictly between min and max so at least one point > T
                T = (mn + mx) // 2
                if T in values:
                    T = T + 1 if T + 1 < mx else T - 1
                if not any(v > T for v in values):
                    return None
                ans = "yes"
            else:
                # T above max so no point exceeds
                T = mx + rng.choice([10, 20, 30])
                if any(v > T for v in values):
                    return None
                ans = "no"
            sidx = (self.seed or 0) % len(_TEMPLATES_YESNO_EVER)
            question = _TEMPLATES_YESNO_EVER[sidx].format(T=T)
            answer = ans
            img = self._render_line(x_labels, values, y_label, T, rng)
            return question, answer, img

        # count or below_count
        sorted_v = sorted(set(values))
        if len(sorted_v) < 2:
            return None
        i = rng.randrange(len(sorted_v) - 1)
        lo, hi = sorted_v[i], sorted_v[i + 1]
        T = int(round((lo + hi) / 2.0))
        # Ensure T not equal to any value (so >/< unambiguous)
        if T in values:
            for delta in (1, -1, 2, -2):
                cand = T + delta
                if cand not in values and lo < cand < hi:
                    T = cand
                    break
        if T in values:
            return None

        if mode == "count":
            count = sum(1 for v in values if v > T)
            tpl = _TEMPLATES_COUNT
        else:
            count = sum(1 for v in values if v < T)
            tpl = _TEMPLATES_BELOW_COUNT
        if count == 0 or count == n:
            return None
        sidx = (self.seed or 0) % len(tpl)
        question = tpl[sidx].format(T=T, period_unit=period_unit)
        answer = str(count)
        img = self._render_line(x_labels, values, y_label, T, rng)
        return question, answer, img

    def _render_line(self, x_labels, values, y_label, T, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        line_color = palette[0]
        bg = "#ffffff"
        n = len(x_labels)
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        x = list(range(n))
        ax.plot(x, values, "-o", color=line_color, linewidth=2.4,
                markersize=8, markerfacecolor=line_color,
                markeredgecolor="#1a1a1a")
        # Annotate each point
        max_v = max(values)
        for xi, v in zip(x, values):
            ax.text(xi, v + max_v * 0.02, str(v),
                    ha="center", va="bottom", fontsize=10,
                    color="#111", fontweight="bold")
        # Threshold line
        ax.axhline(T, color="#c0392b", linewidth=1.6, linestyle="--",
                   label=f"y = {T}")
        ax.legend(loc="best", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        # Make sure threshold line is visible — y_lim covers both
        y_lo = min(min(values), T) - max_v * 0.10 - 5
        y_hi = max(max(values), T) + max_v * 0.18 + 5
        ax.set_ylim(y_lo, y_hi)
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
