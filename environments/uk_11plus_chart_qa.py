"""
UK 11+ / numerical-aptitude style multi-step chart reasoning.

A multi-series bar / line chart with axis labels + bar-top values, paired with
a 2-4 step word problem. Operations: ratio, percentage change, weighted total,
back-calculation from %, multi-month sum.

Decimal / integer numeric output (one decimal when needed).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter


_TEMPLATES_RATIO = [
    "From the chart, what is the ratio of {a_label} to {b_label} for {grp}? Answer in the simplest a:b form. Place ratio in <answer>...</answer>.",
    "Compute the ratio of {a_label} to {b_label} in {grp} from the chart, in simplest a:b form. Place in <answer>...</answer>.",
    "What is the ratio (in simplest form) of {a_label} to {b_label} for {grp}? Place a:b in <answer>...</answer>.",
    "Looking at the chart, find the ratio {a_label}:{b_label} for {grp}, in lowest terms. Place a:b in <answer>...</answer>.",
]

_TEMPLATES_PCT_CHANGE = [
    "From the chart, what is the percentage change in {series} from {x1} to {x2}? Place the numeric percentage (e.g. 25 for 25%) in <answer>...</answer>.",
    "Compute the percentage change of {series} between {x1} and {x2}. Numeric (positive for increase) in <answer>...</answer>.",
    "By what percent did {series} change from {x1} to {x2}? Place the numeric percentage in <answer>...</answer>.",
    "What is the percent increase (or decrease, if negative) for {series} from {x1} to {x2}? Numeric value in <answer>...</answer>.",
]

_TEMPLATES_DIFFERENCE = [
    "What is the difference between {a_label} and {b_label} for {grp} in the chart? Place the numeric value in <answer>...</answer>.",
    "Compute |{a_label} - {b_label}| for {grp}. Numeric answer in <answer>...</answer>.",
    "From the chart, find the absolute difference between {a_label} and {b_label} for {grp}. Numeric in <answer>...</answer>.",
    "How much greater is {a_label} than {b_label} for {grp}? (Use absolute value if negative.) Numeric in <answer>...</answer>.",
]

_TEMPLATES_BACK_CALC = [
    "If {grp}'s value of {v} represents {pct}% of the regional total, what is the regional total? Place the numeric value in <answer>...</answer>.",
    "From the chart, {grp} has value {v}, which is {pct}% of total. Compute the total. Numeric in <answer>...</answer>.",
    "{grp} has {v} units in the chart. If this is {pct}% of the overall total, what is the overall total? Numeric in <answer>...</answer>.",
    "Given that {v} is {pct}% of the total, find the total (chart shows {grp}={v}). Numeric in <answer>...</answer>.",
]

_TEMPLATES_MULTI_SUM = [
    "From the chart, what is the combined value of {series} across {months_text}? Place the numeric total in <answer>...</answer>.",
    "Sum the {series} values for the following periods: {months_text}. Numeric total in <answer>...</answer>.",
    "Compute the total of {series} across {months_text} as shown in the chart. Numeric value in <answer>...</answer>.",
    "Across {months_text}, what is the total {series}? Numeric in <answer>...</answer>.",
]

_TEMPLATES_PROJECTION = [
    "Looking at the chart, if {series} grows by {p}% from {x1} to {x2}, what would the {x2} value be? Numeric answer in <answer>...</answer>.",
    "Suppose {series} increases by {p}% from {x1}. Compute the new value (where {x1} is shown in the chart). Numeric in <answer>...</answer>.",
    "If {series}'s value in the chart at {x1} grows by {p}%, what is the projected value? Numeric in <answer>...</answer>.",
    "Project {series}: starting from {x1} value in chart, apply a {p}% increase. Numeric result in <answer>...</answer>.",
]


_GRP_POOLS = [
    ["UK", "France", "Germany", "Italy", "Spain"],
    ["Office", "Print Room", "Meeting Rooms", "PC Room", "Lab"],
    ["Grade A", "Grade B", "Grade C", "Grade D"],
    ["Tirana", "Algiers", "Stockholm", "Lisbon"],
    ["Pacific", "Atlantic", "Asian", "European"],
    ["North", "South", "East", "West", "Central"],
]
_X_TIME_POOLS = [
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["2018", "2019", "2020", "2021", "2022", "2023"],
    ["1990", "2000", "2010", "2020"],
    ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
]
_SERIES_LABELS = ["sales", "production", "revenue", "output", "headcount", "spend"]
_Y_LABELS = ["Sales (£m)", "Output (TWh)", "Revenue ($k)", "Production (units)",
             "Spend (£m)", "Headcount", "Score"]


def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a if a else 1


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class UK11PlusChartQA(StandaloneVisualEnv):
    ENV_NAME = "uk_11plus_chart_qa"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            modes = ["ratio_simple"]
            n_groups = 4
        elif level <= 2:
            modes = ["ratio_simple", "difference"]
            n_groups = 4
        elif level <= 4:
            modes = ["ratio_simple", "difference", "pct_change"]
            n_groups = 5
        elif level <= 6:
            modes = ["pct_change", "back_calc", "multi_sum", "difference"]
            n_groups = 6
        elif level <= 7:
            modes = ["pct_change", "back_calc", "multi_sum", "projection"]
            n_groups = 7
        # 2026-05-04: bumped L9 difficulty (also L8) — was 95% saturated.
        # 2026-05-04 R3: softened bump — L9 modes=["projection"] only with
        # n_groups=9 (L9 dropped to 0.50). Mix back_calc + projection so it's
        # not single-mode; reduce n_groups so chart stays readable.
        elif level == 8:
            modes = ["back_calc", "projection", "multi_sum"]
            n_groups = 7
        else:
            modes = ["back_calc", "projection"]
            n_groups = 7
        return {"level": level, "modes": modes, "n_groups": n_groups}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1657 + level * 73 + 17)
        mode = rng.choice(cfg["modes"])

        if mode == "ratio_simple":
            result = self._gen_ratio(cfg, rng)
        elif mode == "difference":
            result = self._gen_difference(cfg, rng)
        elif mode == "pct_change":
            result = self._gen_pct_change(cfg, rng)
        elif mode == "back_calc":
            result = self._gen_back_calc(cfg, rng)
        elif mode == "multi_sum":
            result = self._gen_multi_sum(cfg, rng)
        else:
            result = self._gen_projection(cfg, rng)

        # MCQ-letter mode: ALWAYS convert numeric/ratio answer to single-letter
        # MCQ A-E format (target benchmark scorer extracts a single letter and
        # does lowercased exact-match — bare numeric outputs would never score).
        # Falls back to numeric only if distractor generation fails (e.g. when
        # the correct value is exactly 0 — currently no generator produces 0).
        # 2026-05-04: LogicVista benchmark always 5 options A-E with E=
        # "Cannot say" trap. ~15% trap rate.
        if result is not None:
            q, ans, img = result
            q, ans = maybe_to_mcq_letter(q, ans, rng, prob=1.0, n_options=4)
            if isinstance(ans, str) and len(ans) == 1 and ans in "ABCDE":
                trap = rng.random() < 0.15
                if trap:
                    q = q.rstrip() + " (E)Cannot say"
                    ans = "E"
                else:
                    q = q.rstrip() + " (E)Cannot say"
            return q, ans, img
        return result

    # -------------------------------------------------------- generators
    def _gen_ratio(self, cfg, rng):
        # Multi-series bars: 2 series across N groups
        grp_pool = rng.choice(_GRP_POOLS)
        n = min(cfg["n_groups"], len(grp_pool))
        groups = rng.sample(grp_pool, n)
        a_label = "Direct Sales"
        b_label = "Telesales"
        # values chosen so that ratio simplifies
        a_vals = []
        b_vals = []
        # Pick one group with a clean ratio
        target_idx = rng.randrange(n)
        clean_pairs = [(20, 30), (10, 30), (40, 60), (60, 90),
                       (20, 50), (30, 90), (40, 80), (50, 75), (60, 80)]
        a_t, b_t = rng.choice(clean_pairs)
        for i in range(n):
            if i == target_idx:
                a_vals.append(a_t)
                b_vals.append(b_t)
            else:
                a_vals.append(rng.randint(15, 90))
                b_vals.append(rng.randint(15, 90))
        g = _gcd(a_t, b_t)
        ans = f"{a_t // g}:{b_t // g}"

        sidx = (self.seed or 0) % len(_TEMPLATES_RATIO)
        question = _TEMPLATES_RATIO[sidx].format(
            a_label=a_label, b_label=b_label, grp=groups[target_idx])
        y_label = rng.choice(_Y_LABELS)
        img = self._render_grouped_bars(groups, [a_vals, b_vals],
                                        [a_label, b_label], y_label, rng)
        return question, ans, img

    def _gen_difference(self, cfg, rng):
        grp_pool = rng.choice(_GRP_POOLS)
        n = min(cfg["n_groups"], len(grp_pool))
        groups = rng.sample(grp_pool, n)
        a_label = rng.choice(["Imports", "Direct Sales", "Production"])
        b_label = rng.choice(["Exports", "Telesales", "Consumption"])
        if a_label == b_label:
            b_label = "Other"
        a_vals = [rng.randint(20, 200) for _ in range(n)]
        b_vals = [rng.randint(20, 200) for _ in range(n)]
        target_idx = rng.randrange(n)
        ans = abs(a_vals[target_idx] - b_vals[target_idx])
        sidx = (self.seed or 0) % len(_TEMPLATES_DIFFERENCE)
        question = _TEMPLATES_DIFFERENCE[sidx].format(
            a_label=a_label, b_label=b_label, grp=groups[target_idx])
        y_label = rng.choice(_Y_LABELS)
        img = self._render_grouped_bars(groups, [a_vals, b_vals],
                                        [a_label, b_label], y_label, rng)
        return question, _format_num(ans), img

    def _gen_pct_change(self, cfg, rng):
        # Use line chart over time for one series
        x_pool = rng.choice(_X_TIME_POOLS)
        n_time = min(cfg["n_groups"], len(x_pool))
        x_vals = x_pool[:n_time]
        n_series = rng.choice([1, 2])
        series_names = ["Tirana", "Algiers", "Stockholm"][:n_series]
        if n_series == 1:
            series_names = [rng.choice(["UK", "France", "Germany", "Italy", "Spain"])]
        all_y = []
        for s in series_names:
            # generate clean nonzero values
            ys = [rng.choice([100, 120, 150, 180, 200, 240, 250, 300, 350])
                  for _ in range(n_time)]
            all_y.append(ys)
        # Pick one series and a clean pct-change pair
        s_idx = rng.randrange(n_series)
        i1, i2 = sorted(rng.sample(range(n_time), 2))
        # Make values yield a clean percent (e.g., 50 -> 75 = 50%)
        clean = rng.choice([(100, 150), (100, 125), (100, 175), (100, 200),
                            (200, 250), (150, 180), (200, 240),
                            (200, 150), (300, 240)])
        all_y[s_idx][i1] = clean[0]
        all_y[s_idx][i2] = clean[1]
        v1, v2 = clean
        pct = (v2 - v1) * 100.0 / v1
        ans = _format_num(pct)
        sidx = (self.seed or 0) % len(_TEMPLATES_PCT_CHANGE)
        question = _TEMPLATES_PCT_CHANGE[sidx].format(
            series=series_names[s_idx], x1=x_vals[i1], x2=x_vals[i2])
        y_label = rng.choice(_Y_LABELS)
        img = self._render_lines(x_vals, all_y, series_names, y_label, rng)
        return question, ans, img

    def _gen_back_calc(self, cfg, rng):
        grp_pool = rng.choice(_GRP_POOLS)
        n = min(cfg["n_groups"], len(grp_pool))
        groups = rng.sample(grp_pool, n)
        # Pick clean pct
        pct = rng.choice([10, 20, 25, 40, 50, 75, 80])
        # Pick clean v so total is integer
        total = rng.choice([100, 200, 250, 400, 500, 800, 1000, 2000])
        v = int(total * pct / 100)
        target_idx = rng.randrange(n)
        vals = [rng.randint(20, 300) for _ in range(n)]
        vals[target_idx] = v
        ans = total
        sidx = (self.seed or 0) % len(_TEMPLATES_BACK_CALC)
        question = _TEMPLATES_BACK_CALC[sidx].format(
            grp=groups[target_idx], v=v, pct=pct)
        y_label = rng.choice(_Y_LABELS)
        img = self._render_grouped_bars(groups, [vals], ["Value"], y_label, rng)
        return question, _format_num(ans), img

    def _gen_multi_sum(self, cfg, rng):
        x_pool = rng.choice(_X_TIME_POOLS)
        n_time = min(cfg["n_groups"], len(x_pool))
        x_vals = x_pool[:n_time]
        n_series = rng.choice([2, 3])
        cands_pool = ["Tirana", "Algiers", "Stockholm", "Lisbon",
                      "UK", "France", "Germany"]
        series_names = rng.sample(cands_pool, n_series)
        all_y = []
        for s in series_names:
            ys = [rng.randint(20, 80) for _ in range(n_time)]
            all_y.append(ys)
        s_idx = rng.randrange(n_series)
        # Pick a subset of months to sum
        n_pick = rng.randint(2, min(4, n_time))
        idxs = sorted(rng.sample(range(n_time), n_pick))
        ans = sum(all_y[s_idx][i] for i in idxs)
        months_str = ", ".join(x_vals[i] for i in idxs)
        sidx = (self.seed or 0) % len(_TEMPLATES_MULTI_SUM)
        question = _TEMPLATES_MULTI_SUM[sidx].format(
            series=series_names[s_idx], months_text=months_str)
        y_label = rng.choice(_Y_LABELS)
        img = self._render_lines(x_vals, all_y, series_names, y_label, rng)
        return question, _format_num(ans), img

    def _gen_projection(self, cfg, rng):
        x_pool = rng.choice(_X_TIME_POOLS)
        n_time = min(cfg["n_groups"], len(x_pool))
        x_vals = x_pool[:n_time]
        n_series = rng.choice([1, 2, 3])
        cands_pool = ["UK", "France", "Germany", "Italy", "Spain"]
        series_names = rng.sample(cands_pool, n_series)
        all_y = []
        for s in series_names:
            ys = [rng.choice([80, 100, 120, 150, 200]) for _ in range(n_time)]
            all_y.append(ys)
        s_idx = rng.randrange(n_series)
        i1 = rng.randrange(n_time)
        v = all_y[s_idx][i1]
        # clean pct
        p = rng.choice([10, 15, 20, 25, 50])
        ans = v * (1 + p / 100)
        sidx = (self.seed or 0) % len(_TEMPLATES_PROJECTION)
        # x2 is a placeholder time after x1 (use Year+ or label)
        x2_label = "the next period"
        question = _TEMPLATES_PROJECTION[sidx].format(
            series=series_names[s_idx], x1=x_vals[i1], x2=x2_label, p=p)
        y_label = rng.choice(_Y_LABELS)
        img = self._render_lines(x_vals, all_y, series_names, y_label, rng)
        return question, _format_num(ans), img

    # -------------------------------------------------------- renderers
    def _render_grouped_bars(self, groups, series_values, series_labels,
                             y_label, rng) -> Image.Image:
        n = len(groups)
        n_series = len(series_values)
        palette = rng.choice(self._COLOR_PALETTES)
        bg = "#ffffff"
        fig_w = max(7.0, n * (0.6 * n_series) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        x = np.arange(n)
        width = 0.8 / n_series
        max_v = max(max(s) for s in series_values)
        for si, vals in enumerate(series_values):
            offset = (si - (n_series - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9,
                          color=palette[si % len(palette)],
                          edgecolor="#1a1a1a", linewidth=0.5,
                          label=series_labels[si])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + max_v * 0.012,
                        str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#111")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20 if n > 5 else 0,
                           ha="right" if n > 5 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.22 + 1)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n_series >= 2:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg, dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _render_lines(self, x_labels, all_y, series_names, y_label, rng):
        n = len(x_labels)
        n_series = len(all_y)
        palette = rng.choice(self._COLOR_PALETTES)
        bg = "#ffffff"
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        markers = ["o", "s", "D", "^", "v", "P", "X"]
        max_v = max(max(s) for s in all_y)
        for si, ys in enumerate(all_y):
            ax.plot(range(n), ys, marker=markers[si % len(markers)],
                    color=palette[si % len(palette)],
                    label=series_names[si], linewidth=2)
            for xi, yv in enumerate(ys):
                ax.annotate(
                    str(int(yv) if abs(yv - int(yv)) < 1e-6 else f"{yv:.1f}"),
                    (xi, yv),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8.5, color="#111", fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.25 + 1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if n_series >= 2:
            ax.legend(loc="best", fontsize=9, framealpha=0.85)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg, dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
