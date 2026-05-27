"""
Area chart QA — read values, compare series, infer trend.
Targets: chart-reading, statistical reasoning.

Capabilities: V3 (chart extraction), R1 (arithmetic), R4 (statistical)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2 series, 4 x-points, integer values, NON-stacked overlay. Ask "which series
    is bigger at x=N" — clear visual gap.
L1: 2 series, 4 x-points, ask "what is the value of series A at x=N" — read off.
L2: 3 series, 5 x-points, non-stacked. Ask "which series is largest at x=N".
L3: 3 series, 5 x-points, ask "what is the value of series A at x=N".
L4: 3 series, 6 x-points, ask "trend direction" (increasing/decreasing).
L5: 3 series, 6 x-points, stacked. Ask "what is the total at x=N".
L6: 3 series, 8 x-points, stacked. Ask "which time has the highest total".
L7: 4 series, 8 x-points, stacked. Ask "what percentage of total is X at time T".
L8: 4 series, 10 x-points, stacked overlapping with decimal values, ask trend.
L9: 4 series, 10 x-points, stacked, ask "when does series X overtake series Y".

parameter = {"level": int in [0, 9]}
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

_TIME_LABEL_POOLS = [
    ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027"],
    ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Day8", "Day9", "Day10"],
    ["Wk1", "Wk2", "Wk3", "Wk4", "Wk5", "Wk6", "Wk7", "Wk8", "Wk9", "Wk10"],
]
_SERIES_NAME_POOLS = [
    ["Apples", "Bananas", "Cherries", "Dates"],
    ["Online", "Offline", "Mobile", "Email"],
    ["North", "South", "East", "West"],
    ["Red", "Blue", "Green", "Yellow"],
    ["Alpha", "Beta", "Gamma", "Delta"],
    ["Cars", "Trucks", "Bikes", "Vans"],
]
_TITLE_VARIANTS = [
    "Area Chart",
    "Trend over time",
    "Volume by category",
    "Series comparison",
    "Time series",
]

class AreaChartQA(StandaloneVisualEnv):
    ENV_NAME = "area_chart"

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(15):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            return {"n_series": 2, "n_time": 4, "stacked": False,
                    "qtype": "which_largest_at"}
        if level == 1:
            return {"n_series": 2, "n_time": 4, "stacked": False,
                    "qtype": "value_at"}
        if level == 2:
            return {"n_series": 3, "n_time": 5, "stacked": False,
                    "qtype": "which_largest_at"}
        if level == 3:
            return {"n_series": 3, "n_time": 5, "stacked": False,
                    "qtype": "value_at"}
        if level == 4:
            return {"n_series": 3, "n_time": 6, "stacked": False,
                    "qtype": "trend"}
        if level == 5:
            return {"n_series": 3, "n_time": 6, "stacked": True,
                    "qtype": "total_at"}
        if level == 6:
            return {"n_series": 3, "n_time": 8, "stacked": True,
                    "qtype": "when_max_total"}
        if level == 7:
            return {"n_series": 4, "n_time": 8, "stacked": True,
                    "qtype": "percent_at"}
        if level == 8:
            return {"n_series": 4, "n_time": 10, "stacked": True,
                    "qtype": "trend"}
        return {"n_series": 4, "n_time": 10, "stacked": True,
                "qtype": "value_at"}

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)

        time_pool = rng.choice(_TIME_LABEL_POOLS)
        time_labels = time_pool[:cfg["n_time"]]
        series_pool = rng.choice(_SERIES_NAME_POOLS)
        series = series_pool[:cfg["n_series"]]

        # Generate data with clear separation at low levels
        data = {}
        if level <= 2:
            # Highly distinct ranges for clarity
            base_values = [10, 30, 50, 70][:cfg["n_series"]]
            rng.shuffle(base_values)
            for i, s in enumerate(series):
                base = base_values[i]
                vals = [base + rng.randint(-2, 2) for _ in time_labels]
                data[s] = vals
        else:
            for i, s in enumerate(series):
                base = rng.randint(5, 50)
                trend = rng.uniform(-2, 3)
                vals = [max(1, int(base + trend * t + rng.randint(-3, 3)))
                        for t in range(cfg["n_time"])]
                data[s] = vals

        qtype = cfg["qtype"]
        question, answer = self._build_qa(rng, qtype, data, series, time_labels)
        if question is None:
            return None
        image = self._render(rng, data, series, time_labels, cfg["stacked"])
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], image
        return question, str(answer), image

    def _build_qa(self, rng, qtype, data, series, time_labels):
        if qtype == "which_largest_at":
            t_idx = rng.randint(0, len(time_labels) - 1)
            best = max(series, key=lambda s: data[s][t_idx])
            stems = [
                f"At '{time_labels[t_idx]}', which series has the largest value? "
                f"Answer with the series name.",
                f"Which category is highest at '{time_labels[t_idx]}'? "
                f"Reply with the series name only.",
            ]
            return rng.choice(stems), best

        if qtype == "value_at":
            s = rng.choice(series)
            t_idx = rng.randint(0, len(time_labels) - 1)
            stems = [
                f"What is the value of '{s}' at '{time_labels[t_idx]}'? "
                f"Answer with a single integer.",
                f"Read the value for '{s}' at time '{time_labels[t_idx]}'.",
            ]
            return rng.choice(stems), data[s][t_idx]

        if qtype == "trend":
            s = rng.choice(series)
            first = data[s][0]
            last = data[s][-1]
            if last > first + 1:
                direction = "increasing"
            elif last < first - 1:
                direction = "decreasing"
            else:
                direction = "flat"
            stems = [
                f"Is '{s}' generally increasing or decreasing over the time period? "
                f"Answer 'increasing', 'decreasing', or 'flat'.",
                f"Describe the overall trend of '{s}' from {time_labels[0]} to "
                f"{time_labels[-1]}.",
            ]
            return rng.choice(stems), direction

        if qtype == "total_at":
            t_idx = rng.randint(0, len(time_labels) - 1)
            total = sum(data[s][t_idx] for s in series)
            stems = [
                f"What is the total (sum of all series) at '{time_labels[t_idx]}'? "
                f"Answer with a single integer.",
                f"Compute the sum across all categories at time '{time_labels[t_idx]}'.",
            ]
            return rng.choice(stems), total

        if qtype == "when_max_total":
            totals = [sum(data[s][t] for s in series)
                      for t in range(len(time_labels))]
            max_idx = totals.index(max(totals))
            stems = [
                "At which time point is the total (sum of all series) highest? "
                "Answer with the time label.",
                "Which time has the largest combined total across all series?",
            ]
            return rng.choice(stems), time_labels[max_idx]

        if qtype == "percent_at":
            t_idx = rng.randint(0, len(time_labels) - 1)
            s = rng.choice(series)
            total = sum(data[s2][t_idx] for s2 in series)
            if total == 0:
                return None, None
            pct = round(data[s][t_idx] / total * 100)
            stems = [
                f"What percentage (rounded to nearest integer) of the total is '{s}' "
                f"at '{time_labels[t_idx]}'?",
                f"Compute the share of '{s}' at '{time_labels[t_idx]}' as an "
                f"integer percent of the total.",
            ]
            return rng.choice(stems), pct

        return None, None

    def _render(self, rng, data, series, time_labels, stacked):
        style = self._random_style()
        fig, ax = plt.subplots(figsize=rng.choice([(8, 5), (9, 5), (8.5, 5.5)]))
        self._apply_style(fig, ax, style)
        palette = list(style["palette"])
        rng.shuffle(palette)
        # If series are named after colors, pin each series to its matching
        # visual color so the legend is not misleading (e.g. a "Red" entry
        # drawn in blue).
        _COLOR_NAME_TO_HEX = {
            "Red": "#e74c3c", "Blue": "#3498db",
            "Green": "#2ecc71", "Yellow": "#f1c40f",
        }
        if all(s in _COLOR_NAME_TO_HEX for s in series):
            palette = [_COLOR_NAME_TO_HEX[s] for s in series]

        x = np.arange(len(time_labels))
        if stacked:
            cum = np.zeros(len(time_labels))
            for i, s in enumerate(series):
                vals = np.array(data[s], dtype=float)
                ax.fill_between(x, cum, cum + vals, alpha=0.65,
                                color=palette[i % len(palette)], label=s,
                                edgecolor="black", linewidth=0.5)
                cum += vals
        else:
            for i, s in enumerate(series):
                vals = np.array(data[s], dtype=float)
                ax.fill_between(x, 0, vals, alpha=0.45,
                                color=palette[i % len(palette)], label=s,
                                edgecolor="black", linewidth=0.5)
                ax.plot(x, vals, color=palette[i % len(palette)],
                        linewidth=2)

        ax.set_xticks(x)
        ax.set_xticklabels(time_labels, fontsize=style["font_size_base"])
        ax.legend(fontsize=style["font_size_base"] - 1, loc="upper left")
        ax.set_ylabel("Value", fontsize=style["font_size_base"])
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        ax.grid(True, alpha=0.3)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = AreaChartQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
