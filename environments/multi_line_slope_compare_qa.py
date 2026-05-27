"""
Multi-line slope/steepness comparison.

Render a multi-line chart (3-5 lines at low/mid level, up to 5 at high level)
and ask which line is steepest (or declined fastest) over the full window or
between two specified periods. Answer is a single line label string.

Mirrors Factoid + Conversational style reference questions:
  - IDX 108 ("which greenhouse gas grew the most from 1979 to 2021" -> HFCs)
  - IDX 1213 ("which country's policymakers overguesses the most on average" -> Kenya)
  - IDX 707 ("which category has the worst revenue ..." -> Veggie)
  - IDX 1203 ("which city has the highest agglomeration index over time" -> Singapore)
  - IDX 1192 ("any age group decrease sugar consistently" -> No / list)

Output: single line label exactly as shown in legend.
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


_TEMPLATES_FULL_STEEPEST_UP = [
    "Looking at the multi-line chart, which line increased the fastest over the full period shown? Reply with the line label as it appears in the legend, in <answer>...</answer>.",
    "Which line in the chart shows the steepest upward trend across the entire range? Place the line label in <answer>...</answer>.",
    "From the chart, which series grew the most from the first period to the last? Answer with the line label in <answer>...</answer>.",
    "Identify the line that rose fastest across the periods shown. Reply with its label in <answer>...</answer>.",
    "Which series exhibited the largest overall increase from start to end of the chart? Line label in <answer>...</answer>.",
    "On the chart, which line had the strongest upward slope across the whole window? Place the label in <answer>...</answer>.",
    "Among all the lines, which one increased the most over the full period? Line label as shown in <answer>...</answer>.",
    "Which series shows the steepest positive trend over the entire chart? Reply with its legend label in <answer>...</answer>.",
]

_TEMPLATES_FULL_STEEPEST_DOWN = [
    "Which line in the chart declined the fastest over the full period shown? Reply with the line label as in the legend, in <answer>...</answer>.",
    "From the chart, which series fell the most from the first period to the last? Answer with the line label in <answer>...</answer>.",
    "Identify the line with the steepest downward slope across the entire window. Reply with the label in <answer>...</answer>.",
    "Which line dropped the most across the periods shown? Place the label in <answer>...</answer>.",
    "Among all the lines, which one decreased the most over the full chart range? Line label in <answer>...</answer>.",
    "Which series had the strongest downward trend across the entire chart? Line label in <answer>...</answer>.",
]

_TEMPLATES_BETWEEN_STEEPER_UP = [
    "Between {a} and {b}, which line in the chart increased the most? Reply with the line label as in the legend, in <answer>...</answer>.",
    "From {a} to {b}, which series rose the fastest? Place the line label in <answer>...</answer>.",
    "Looking only at the segment from {a} to {b}, which line shows the steepest upward slope? Line label in <answer>...</answer>.",
    "Which line had the largest gain between {a} and {b}? Reply with the label in <answer>...</answer>.",
    "From {a} to {b}, which series increased more steeply than the others? Line label in <answer>...</answer>.",
]

_TEMPLATES_BETWEEN_STEEPER_DOWN = [
    "Between {a} and {b}, which line in the chart declined the fastest? Reply with the line label as in the legend, in <answer>...</answer>.",
    "From {a} to {b}, which series dropped the most? Place the line label in <answer>...</answer>.",
    "Looking only at the segment from {a} to {b}, which line shows the steepest downward slope? Line label in <answer>...</answer>.",
    "Which line had the largest decline between {a} and {b}? Reply with the label in <answer>...</answer>.",
    "From {a} to {b}, which series decreased more steeply than the others? Line label in <answer>...</answer>.",
]


_PERIOD_POOLS = [
    ["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"],
    ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    ["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5", "Wk 6", "Wk 7", "Wk 8"],
    ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8"],
]

_LABEL_POOLS = [
    ["United States", "China", "Germany", "Japan", "India", "Brazil"],
    ["North", "South", "East", "West", "Central", "Pacific"],
    ["Region A", "Region B", "Region C", "Region D", "Region E"],
    ["Plan A", "Plan B", "Plan C", "Plan D", "Plan E"],
    ["Group 1", "Group 2", "Group 3", "Group 4", "Group 5"],
    ["Apple", "Mango", "Orange", "Peach", "Grape"],
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Index"]


class MultiLineSlopeCompareQA(StandaloneVisualEnv):
    ENV_NAME = "multi_line_slope_compare"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            # L0 trivial: 2 lines, A clearly steepest-up by huge margin.
            return {"n_lines": 2, "n_periods": 4, "between": False,
                    "trivial_l0": True}
        if level <= 2:
            return {"n_lines": 3, "n_periods": 5, "between": False,
                    "trivial_l0": False}
        if level <= 4:
            return {"n_lines": 4, "n_periods": 6, "between": False,
                    "trivial_l0": False}
        if level <= 6:
            return {"n_lines": 4, "n_periods": 7, "between": True,
                    "trivial_l0": False}
        if level <= 8:
            return {"n_lines": 5, "n_periods": 8, "between": True,
                    "trivial_l0": False}
        return {"n_lines": 5, "n_periods": 8, "between": True,
                "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1259 + level * 113 + 7)

        period_pool = rng.choice(_PERIOD_POOLS)
        n_periods = cfg["n_periods"]
        periods = period_pool[:n_periods]
        label_pool = rng.choice(_LABEL_POOLS)
        n_lines = min(cfg["n_lines"], len(label_pool))
        labels = rng.sample(label_pool, n_lines)
        y_label = rng.choice(_Y_LABELS)

        if cfg["trivial_l0"]:
            # 2 lines, A clearly steeper-up than B.
            # A: 10 -> 20 -> 50 -> 90 (delta = +80)
            # B: 30 -> 32 -> 34 -> 36 (delta = +6)
            series = [[10, 20, 50, 90], [30, 32, 34, 36]]
            ask_direction = "up"
            ask_between = False
            target_idx = 0
        else:
            # Generate distinct trends so winner is unambiguous.
            series = []
            slopes = []
            for _ in range(n_lines):
                slope = rng.choice([-12, -8, -5, -2, 2, 5, 8, 12])
                slopes.append(slope)
                start = rng.randint(40, 100)
                vals = []
                v = float(start)
                for _ in range(n_periods):
                    v = max(5.0, v + slope + rng.uniform(-2.0, 2.0))
                    vals.append(round(v, 1))
                series.append(vals)
            # Decide direction by random; pick steepest in that direction.
            ask_direction = rng.choice(["up", "down"])
            ask_between = cfg["between"]

            if ask_between:
                # Choose two periods and compute deltas only over [a, b].
                ai = rng.randint(0, n_periods - 3)
                bi = rng.randint(ai + 2, n_periods - 1)
            else:
                ai, bi = 0, n_periods - 1

            deltas = [s[bi] - s[ai] for s in series]
            if ask_direction == "up":
                # If no positive delta, force one by inverting the largest negative line.
                if max(deltas) <= 0:
                    j = deltas.index(max(deltas))
                    series[j] = [series[j][0] + (i * 10) for i in range(n_periods)]
                    deltas = [s[bi] - s[ai] for s in series]
                target_idx = max(range(n_lines), key=lambda i: deltas[i])
                # Tie-break safety: nudge winner up
                m = max(deltas)
                ties = [i for i in range(n_lines) if abs(deltas[i] - m) < 1.5]
                if len(ties) > 1:
                    win = ties[0]
                    extra = 8 + rng.randint(0, 5)
                    series[win][bi] = series[win][bi] + extra
                    deltas = [s[bi] - s[ai] for s in series]
                    target_idx = max(range(n_lines), key=lambda i: deltas[i])
            else:
                if min(deltas) >= 0:
                    j = deltas.index(min(deltas))
                    series[j] = [series[j][0] - (i * 10) for i in range(n_periods)]
                    series[j] = [max(5.0, v) for v in series[j]]
                    deltas = [s[bi] - s[ai] for s in series]
                target_idx = min(range(n_lines), key=lambda i: deltas[i])
                m = min(deltas)
                ties = [i for i in range(n_lines) if abs(deltas[i] - m) < 1.5]
                if len(ties) > 1:
                    win = ties[0]
                    extra = 8 + rng.randint(0, 5)
                    series[win][bi] = max(5.0, series[win][bi] - extra)
                    deltas = [s[bi] - s[ai] for s in series]
                    target_idx = min(range(n_lines), key=lambda i: deltas[i])

        # Build question
        if cfg["trivial_l0"]:
            tlist = _TEMPLATES_FULL_STEEPEST_UP
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx]
        else:
            if ask_between:
                a_label = periods[ai]
                b_label = periods[bi]
                if ask_direction == "up":
                    tlist = _TEMPLATES_BETWEEN_STEEPER_UP
                else:
                    tlist = _TEMPLATES_BETWEEN_STEEPER_DOWN
                sidx = (self.seed or 0) % len(tlist)
                question = tlist[sidx].format(a=a_label, b=b_label)
            else:
                if ask_direction == "up":
                    tlist = _TEMPLATES_FULL_STEEPEST_UP
                else:
                    tlist = _TEMPLATES_FULL_STEEPEST_DOWN
                sidx = (self.seed or 0) % len(tlist)
                question = tlist[sidx]

        answer = labels[target_idx]
        img = self._render(periods, series, labels, y_label, rng)
        return question, answer, img

    def _render(self, periods: List[str], series: List[List[float]],
                labels: List[str], y_label: str, rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        markers = ["o", "s", "^", "D", "v", "P"]
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = list(range(len(periods)))
        for i, (vals, lab) in enumerate(zip(series, labels)):
            ax.plot(x, vals, marker=markers[i % len(markers)],
                    linewidth=2.0, color=palette[i % len(palette)],
                    label=lab, markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlabel("Period", fontsize=10)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Accept exact label match, case-insensitive, with mild lenience for
        # leading/trailing whitespace, quoting, or "the line for X" phrasing.
        p = predicted.strip().lower().rstrip(".").strip()
        g = ground_truth.strip().lower().rstrip(".").strip()
        if p == g:
            return True
        # Strip enclosing quotes / brackets
        for ch in ['"', "'", "`", "[", "]", "(", ")"]:
            p = p.replace(ch, "")
        p = p.strip()
        if p == g:
            return True
        # Substring match: gt contained in pred and pred isn't massively long
        if g in p and len(p) <= len(g) * 4:
            return True
        return False
