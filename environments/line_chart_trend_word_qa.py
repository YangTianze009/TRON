"""
Line Chart Trend-Direction Word QA — C11 (reference Factoid trend-word).

Renders a line chart over N=5-12 periods showing a clear monotonic /
monotonic-decreasing / fluctuating shape and asks the model to name the
trend with one word from a small fixed vocabulary.

Verbatim sample style (design notes Factoid):

    Q-IDX 468:  "what was the general trend in the national accounts taxes
                 as a share of gdp from 2010 onwards according to the
                 outturn line?"
                 GT = `Increasing`

    Q-IDX 1318: "what's the main trend shown in this graph for the number
                 of quarter point hikes priced?"
                 GT = `Decreasing`

    Q-IDX 1607: "the chart shows a steady decline in monthly production"
                 (claim style — uses Decreasing language)

Output vocabulary: ``Increasing``, ``Decreasing``, ``Fluctuating``,
``Stable``.  Verifier uses case-insensitive exact match (relaxed_correctness
falls back to ANLS so "upward" → "Increasing" works at threshold ≥0.5).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_X_LABEL_POOLS = [
    [str(2000 + i) for i in range(20)],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    [f"Q{i}" for i in range(1, 13)],
    [f"W{i}" for i in range(1, 16)],
]

_SERIES_NAME_POOLS = [
    "Revenue", "Profit", "Sales", "Visitors", "Members", "Score",
    "Index", "Rate", "Volume", "GDP", "Price",
]

_Y_LABELS = ["Value", "Index", "Count (K)", "Score", "Rate (%)", "Volume"]

# Closed vocabulary (case as appears verbatim in benchmark samples).
_TREND_VOCAB = ["Increasing", "Decreasing", "Fluctuating", "Stable"]


class LineChartTrendWordQA(StandaloneVisualEnv):
    ENV_NAME = "line_chart_trend_word"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # n_points scales with level so distinguishing trend gets harder.
        if level <= 1:
            n = 5
            allowed = ["Increasing", "Decreasing"]
            noise = 0.05
        elif level <= 3:
            n = 7
            allowed = ["Increasing", "Decreasing", "Stable"]
            noise = 0.10
        elif level <= 6:
            n = 9
            allowed = ["Increasing", "Decreasing", "Stable", "Fluctuating"]
            noise = 0.20
        else:
            n = 12
            allowed = ["Increasing", "Decreasing", "Stable", "Fluctuating"]
            noise = 0.30
        return {"n": n, "allowed": allowed, "noise": noise}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 911 + level * 31 + 7)
        np_rng = np.random.RandomState(rng.randint(0, 1_000_000))

        for _ in range(15):
            res = self._try(rng, np_rng, cfg, level)
            if res is not None:
                return res
        return None

    def _try(self, rng, np_rng, cfg, level):
        target_trend = rng.choice(cfg["allowed"])
        n = cfg["n"]
        noise = cfg["noise"]

        x_pool = list(rng.choice(_X_LABEL_POOLS))
        if len(x_pool) < n:
            return None
        start = rng.randrange(len(x_pool) - n + 1)
        x_labels = x_pool[start:start + n]

        # Generate values to match target trend.
        base = rng.uniform(40, 120)
        amplitude = base * 0.6
        if target_trend == "Increasing":
            slope = abs(rng.uniform(amplitude * 0.07, amplitude * 0.15))
            vals = [base + slope * i + np_rng.normal(0, base * noise * 0.5)
                    for i in range(n)]
        elif target_trend == "Decreasing":
            slope = abs(rng.uniform(amplitude * 0.07, amplitude * 0.15))
            vals = [base - slope * i + np_rng.normal(0, base * noise * 0.5)
                    for i in range(n)]
        elif target_trend == "Stable":
            vals = [base + np_rng.normal(0, base * 0.03) for _ in range(n)]
        elif target_trend == "Fluctuating":
            # Periodic with amplitude bigger than the secular drift.
            phase = rng.uniform(0, 2 * np.pi)
            vals = [base + amplitude * np.sin(phase + 1.7 * i)
                    + np_rng.normal(0, base * 0.05) for i in range(n)]
        else:
            return None

        vals = [round(max(1, v), 1) for v in vals]
        # Re-check trend stability so the GT label is unambiguous.
        actual = self._classify(vals)
        if actual != target_trend:
            return None

        series_name = rng.choice(_SERIES_NAME_POOLS)
        templates = [
            f"What is the overall trend of {series_name} shown in the line chart?",
            f"Looking at the line chart, what is the main trend in {series_name}?",
            f"How would you describe the overall trend in {series_name}?",
            f"Based on the chart, what is the general trend of {series_name}?",
        ]
        question = (
            rng.choice(templates) +
            f" Answer with one word from: {', '.join(_TREND_VOCAB)}."
        )

        image = self._render(rng, x_labels, vals, series_name)
        return question, target_trend, image

    @staticmethod
    def _classify(vals: List[float]) -> str:
        n = len(vals)
        if n < 3:
            return "Stable"
        diffs = [vals[i + 1] - vals[i] for i in range(n - 1)]
        # How monotonic? Use sign agreement.
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        net = vals[-1] - vals[0]
        rng_v = max(vals) - min(vals)
        mean_v = sum(vals) / n
        # Monotonic threshold = ≥80 % of step signs agree AND net change > 15 %
        if pos >= int(0.8 * (n - 1)) and net > 0.15 * abs(mean_v):
            return "Increasing"
        if neg >= int(0.8 * (n - 1)) and net < -0.15 * abs(mean_v):
            return "Decreasing"
        if rng_v < 0.10 * abs(mean_v):
            return "Stable"
        return "Fluctuating"

    def _render(self, rng, x_labels, vals, series_name):
        style = self._random_style()
        palette = list(style["palette"])
        n = len(x_labels)
        fig_w = max(6, n * 0.55 + 2) * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, 4.5 * style["figsize_scale"]))
        ax.plot(range(n), vals, marker="o", color=palette[0],
                linewidth=style["line_width"], markersize=6,
                label=series_name)
        ax.set_xticks(range(n))
        ax.set_xticklabels(x_labels, fontsize=style["font_size_base"],
                           rotation=30 if max(len(l) for l in x_labels) > 4 else 0,
                           ha="right" if max(len(l) for l in x_labels) > 4 else "center")
        ax.set_ylabel(rng.choice(_Y_LABELS),
                      fontsize=style["font_size_base"])
        ax.set_title(f"{series_name} over time",
                     fontsize=style["font_size_base"] + 2, pad=10)
        ax.legend(fontsize=style["font_size_base"] - 1)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
