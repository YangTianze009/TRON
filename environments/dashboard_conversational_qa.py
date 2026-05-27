"""
Dashboard reading: a single image with 2-3 sub-charts (e.g. donut + line +
bar) on a shared figure. Question identifies one specific value to read or
a small composite computation across two sub-charts.

L0: simply read one labeled value off the bar sub-chart.
L>0: composite reads (e.g. donut share + bar value sum).
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


_TEMPLATES_BAR_VALUE = [
    "Read the bar sub-chart in the dashboard. What is the value of the bar labeled '{label}'? Numeric answer in <answer>...</answer>.",
    "From the bar chart on the dashboard, what value does '{label}' have? Numeric in <answer>...</answer>.",
    "Look at the bar chart panel of the dashboard. Report the value of '{label}'. Numeric in <answer>...</answer>.",
    "On the bar sub-chart in the dashboard image, what is the height of the '{label}' bar? Numeric in <answer>...</answer>.",
    "Read the dashboard's bar panel. Report '{label}' in <answer>...</answer>.",
    "From the dashboard, what numeric value does the bar labeled '{label}' show? Numeric answer in <answer>...</answer>.",
    "Inspect the bar panel of the dashboard. What is '{label}'? Numeric in <answer>...</answer>.",
    "The dashboard contains a bar chart. What is the value of '{label}' in that bar chart? Numeric in <answer>...</answer>.",
    "From the dashboard's bar sub-chart, read the value of '{label}'. Numeric answer in <answer>...</answer>.",
    "What numeric value is shown for '{label}' in the dashboard's bar sub-chart? Place it in <answer>...</answer>.",
    "Find '{label}' in the dashboard's bar chart and report its value. Numeric in <answer>...</answer>.",
    "Look at the bar chart in the dashboard image. What is the value of '{label}'? Numeric in <answer>...</answer>.",
    "From the bar panel of the dashboard, find '{label}' and read its value. Numeric in <answer>...</answer>.",
    "Read the dashboard. The bar sub-chart shows '{label}' with what value? Numeric in <answer>...</answer>.",
    "In the bar chart panel of the dashboard, what value does '{label}' display? Numeric in <answer>...</answer>.",
    "Identify the bar '{label}' in the dashboard's bar sub-chart and read its value. Numeric in <answer>...</answer>.",
    "The dashboard's bar chart panel shows multiple bars. Report the value of '{label}'. Numeric in <answer>...</answer>.",
    "Read off the bar height of '{label}' from the dashboard's bar sub-chart. Numeric in <answer>...</answer>.",
]

_TEMPLATES_LINE_DELTA = [
    "From the dashboard, look at the line sub-chart. Compute the value at the last period minus the value at the first period. Numeric (signed) answer in <answer>...</answer>.",
    "On the dashboard's line panel, find the change from the first period to the last period (last minus first). Numeric in <answer>...</answer>.",
    "Read the line chart on the dashboard. Report (last value) - (first value). Numeric in <answer>...</answer>.",
    "Inspect the dashboard's line sub-chart and compute the difference between the last and first values shown. Numeric in <answer>...</answer>.",
    "Compute the change over the full range of the line sub-chart on the dashboard (last minus first). Numeric in <answer>...</answer>.",
    "From the dashboard's line panel, what is the net change from first to last period? Numeric in <answer>...</answer>.",
]

_TEMPLATES_COMPOSITE = [
    "From the dashboard, find the maximum value in the bar sub-chart and the maximum value in the line sub-chart. Report their sum. Numeric in <answer>...</answer>.",
    "Compute the sum of (max bar value in the bar sub-chart) + (max value in the line sub-chart). Place numeric in <answer>...</answer>.",
    "From the bar sub-chart, take the largest bar's value. From the line sub-chart, take the highest point's value. Add them. Numeric in <answer>...</answer>.",
    "Add the maximum bar value to the maximum line value across the dashboard panels. Numeric in <answer>...</answer>.",
    "On the dashboard, find the largest bar and the highest line point. Sum their values. Numeric in <answer>...</answer>.",
]

_TEMPLATES_COMPOSITE_TOP2_PLUS_DELTA = [
    "From the dashboard, take the two largest bar values from the bar sub-chart and add them; then add (last value - first value) of the line sub-chart. Numeric in <answer>...</answer>.",
    "Compute (sum of the top 2 bars in the bar sub-chart) + (last - first of the line sub-chart). Numeric in <answer>...</answer>.",
    "Add the two highest bar values together, then add the line sub-chart's net change (last minus first). Numeric in <answer>...</answer>.",
    "From the bar panel: pick the top two values and sum them. From the line panel: compute (last - first). Add the two results. Numeric in <answer>...</answer>.",
]


_BAR_LABEL_POOLS = [
    ["Alpha", "Bravo", "Charlie", "Delta"],
    ["North", "South", "East", "West"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Apples", "Pears", "Mangos", "Plums"],
    ["Plan A", "Plan B", "Plan C", "Plan D"],
    ["Site 1", "Site 2", "Site 3", "Site 4"],
]

_LINE_PERIOD_POOLS = [
    ["2018", "2019", "2020", "2021", "2022"],
    ["Jan", "Feb", "Mar", "Apr", "May"],
    ["Wk1", "Wk2", "Wk3", "Wk4", "Wk5"],
    ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
]


class DashboardConversationalQA(StandaloneVisualEnv):
    ENV_NAME = "dashboard_conversational"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100. Each
        # level now jumps mode to make every level distinct.
        level = max(0, min(level, 9))
        if level == 0:
            return {"mode": "bar_value", "n_panels": 2, "trivial_l0": True}
        if level <= 2:
            return {"mode": "line_delta", "n_panels": 2, "trivial_l0": False}
        if level <= 4:
            return {"mode": "composite", "n_panels": 3, "trivial_l0": False}
        if level <= 6:
            return {"mode": "composite", "n_panels": 3, "trivial_l0": False}
        # 2026-05-04 R3 retry: L7-L8 also drop down to 'composite' (was
        # composite_top2_plus_delta) — only L9 keeps the hardest 4-step
        # arithmetic mode. Smoother gradient so v5 mean improves.
        if level <= 8:
            return {"mode": "composite", "n_panels": 3,
                    "trivial_l0": False}
        return {"mode": "composite_top2_plus_delta", "n_panels": 3,
                "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1163 + level * 109 + 67)

        bar_pool = rng.choice(_BAR_LABEL_POOLS)
        bar_labels = bar_pool[:4]
        line_periods_pool = rng.choice(_LINE_PERIOD_POOLS)
        line_periods = line_periods_pool[:5]

        if cfg["trivial_l0"]:
            # Fixed values so the answer is deterministic across seeds.
            bar_values = [10, 20, 30, 40]
            line_values = [50, 55, 60, 65, 70]
            target_label = bar_labels[2]  # value 30
            target_value = bar_values[2]
            mode = "bar_value"
        else:
            # Distinct bar values within range
            # 2026-05-04: bumped L9 difficulty via wide_values flag.
            if cfg.get("wide_values"):
                bar_values = rng.sample(range(10, 200), 4)
                line_values = [rng.randint(20, 200) for _ in range(len(line_periods))]
            else:
                bar_values = rng.sample(range(10, 95), 4)
                line_values = [rng.randint(20, 95) for _ in range(len(line_periods))]
            mode = cfg["mode"]
            if mode == "bar_value":
                target_idx = rng.randrange(4)
                target_label = bar_labels[target_idx]
                target_value = bar_values[target_idx]
            elif mode == "line_delta":
                target_value = line_values[-1] - line_values[0]
                target_label = None
            elif mode == "composite_top2_plus_delta":
                # L9: top-2 bar sum + line delta. 4 sub-steps: rank bars,
                # sum top 2, read line endpoints, compute delta, add.
                top2 = sorted(bar_values, reverse=True)[:2]
                target_value = sum(top2) + (line_values[-1] - line_values[0])
                target_label = None
            else:  # composite
                target_value = max(bar_values) + max(line_values)
                target_label = None

        # Build question
        sidx_bar = (self.seed or 0) % len(_TEMPLATES_BAR_VALUE)
        sidx_line = (self.seed or 0) % len(_TEMPLATES_LINE_DELTA)
        sidx_comp = (self.seed or 0) % len(_TEMPLATES_COMPOSITE)
        sidx_t2d = (self.seed or 0) % len(_TEMPLATES_COMPOSITE_TOP2_PLUS_DELTA)
        if mode == "bar_value":
            question = _TEMPLATES_BAR_VALUE[sidx_bar].format(label=target_label)
        elif mode == "line_delta":
            question = _TEMPLATES_LINE_DELTA[sidx_line]
        elif mode == "composite_top2_plus_delta":
            question = _TEMPLATES_COMPOSITE_TOP2_PLUS_DELTA[sidx_t2d]
        else:
            question = _TEMPLATES_COMPOSITE[sidx_comp]

        answer = str(int(target_value))
        img = self._render(bar_labels, bar_values, line_periods, line_values,
                            cfg["n_panels"], rng)
        return question, answer, img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, bar_labels: List[str], bar_values: List[int],
                line_periods: List[str], line_values: List[int],
                n_panels: int, rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        if n_panels == 2:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        # Panel 0: bar chart
        ax = axes[0]
        ax.set_facecolor("#ffffff")
        bar_colors = [palette[i % len(palette)] for i in range(len(bar_labels))]
        bars = ax.bar(range(len(bar_labels)), bar_values, color=bar_colors,
                       edgecolor="#1a1a1a", linewidth=0.6)
        max_bv = max(bar_values)
        for b, v in zip(bars, bar_values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_bv * 0.02,
                    str(int(v)), ha="center", va="bottom",
                    fontsize=10, color="#111", fontweight="bold")
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_ylim(0, max_bv * 1.22 + 4)
        ax.set_title("Bar chart", fontsize=11, loc="left")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel 1: line chart
        ax = axes[1]
        ax.set_facecolor("#ffffff")
        x = list(range(len(line_periods)))
        ax.plot(x, line_values, marker="o", linewidth=2.0,
                color=palette[0], markersize=7)
        for xi, vi in zip(x, line_values):
            ax.text(xi, vi + max(line_values) * 0.025, str(int(vi)),
                     ha="center", va="bottom", fontsize=9, color="#111",
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(line_periods, fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_xlabel("Period", fontsize=10)
        ax.set_ylim(0, max(line_values) * 1.25 + 4)
        ax.set_title("Line chart", fontsize=11, loc="left")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Optional Panel 2: donut (decorative; not asked in current modes)
        if n_panels == 3:
            ax = axes[2]
            ax.set_facecolor("#ffffff")
            # Use the first 4 bar values as donut shares for visual variety
            shares = [max(1, v) for v in bar_values]
            colors = [palette[(i + 2) % len(palette)] for i in range(len(shares))]
            ax.pie(shares, labels=bar_labels, colors=colors,
                    startangle=90, wedgeprops=dict(width=0.4),
                    textprops={"fontsize": 9})
            ax.set_title("Donut chart", fontsize=11, loc="left")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
