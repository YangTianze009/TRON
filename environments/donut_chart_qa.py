"""Donut (Ring) Chart Visual QA Environment."""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_OUTER_LABELS = [
    ["Product A", "Product B", "Product C", "Product D", "Product E"],
    ["North", "South", "East", "West"],
    ["Sales", "Marketing", "R&D", "Support", "Admin"],
    ["Q1", "Q2", "Q3", "Q4"],
]
_INNER_LABELS = [
    ["Online", "Offline"],
    ["Domestic", "International"],
    ["Direct", "Indirect"],
    ["New", "Returning"],
]

class DonutChartQA(StandaloneVisualEnv):
    ENV_NAME = "donut_chart"

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["largest_outer"]}
        if level == 1:
            return {"qtypes": ["largest_outer", "smallest_outer"]}
        if level == 2:
            return {"qtypes": ["largest_outer", "largest_inner"]}
        if level == 3:
            return {"qtypes": ["largest_inner", "smallest_outer"]}
        if level == 4:
            return {"qtypes": ["outer_percentage", "inner_vs_outer_total"]}
        if level == 5:
            return {"qtypes": ["outer_percentage", "count_above_pct"]}
        if level == 6:
            return {"qtypes": ["outer_percentage", "count_above_pct", "inner_vs_outer_total"]}
        if level == 7:
            return {"qtypes": ["outer_percentage", "count_above_pct"]}
        if level == 8:
            return {"qtypes": ["count_above_pct"]}
        return {"qtypes": ["outer_percentage", "count_above_pct"]}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 707)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        outer_labels = list(sub_rng.choice(_OUTER_LABELS))
        inner_labels = list(sub_rng.choice(_INNER_LABELS))
        outer_vals = [sub_rng.randint(10, 80) for _ in outer_labels]
        inner_vals = [sub_rng.randint(20, 90) for _ in inner_labels]

        question, answer = self._make_qa(rng, qtype, outer_labels, outer_vals,
                                          inner_labels, inner_vals)
        if question is None:
            return None

        style = self._random_style()
        image = self._render(style, outer_labels, outer_vals, inner_labels, inner_vals)
        return question, answer, image

    def _render(self, style, outer_labels, outer_vals, inner_labels, inner_vals):
        fig, ax = plt.subplots(figsize=(6.5 * style["figsize_scale"], 5.5 * style["figsize_scale"]))
        palette = style["palette"]
        outer_colors = [palette[i % len(palette)] for i in range(len(outer_labels))]
        inner_colors = [palette[(i + 3) % len(palette)] for i in range(len(inner_labels))]

        # Outer ring
        outer_total = sum(outer_vals)
        wedges1, texts1, autotexts1 = ax.pie(
            outer_vals, labels=outer_labels, colors=outer_colors,
            autopct=lambda p: f"{p:.1f}%", pctdistance=0.82,
            radius=1.0, wedgeprops=dict(width=0.35, edgecolor="white"),
            textprops={"fontsize": style["font_size_base"] - 1})

        # Inner ring
        inner_total = sum(inner_vals)
        wedges2, texts2, autotexts2 = ax.pie(
            inner_vals, labels=inner_labels, colors=inner_colors,
            autopct=lambda p: f"{p:.1f}%", pctdistance=0.75,
            radius=0.6, wedgeprops=dict(width=0.3, edgecolor="white"),
            textprops={"fontsize": style["font_size_base"] - 2})

        ax.set_title("Donut Chart", fontsize=style["font_size_base"] + 2, pad=15)
        fig.patch.set_facecolor(style["bg_color"])
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _make_qa(self, rng, qtype, ol, ov, il, iv):
        outer_total = sum(ov)
        inner_total = sum(iv)
        sidx = (self.seed or 0) % 16

        if qtype == "largest_outer":
            idx = ov.index(max(ov))
            _P = ["Which segment in the outer ring is the largest?",
                  "Identify the outer-ring segment with the maximum value.",
                  "Which outer-ring segment has the highest value in the donut chart?",
                  "Find the largest segment in the outer ring.",
                  "What outer-ring category dominates (largest value)?",
                  "Which category in the outer ring is the biggest by value?",
                  "In the outer ring of the donut, which segment is largest?",
                  "Which outer-ring slice is the largest?",
                  "Pick the outer-ring segment with the greatest value.",
                  "Which outer ring category has the max value?",
                  "Find the outer-ring segment with the top value.",
                  "Which outer category occupies the largest portion of its ring?",
                  "Report the largest segment from the outer ring.",
                  "Largest outer-ring segment?",
                  "Which slice of the outer ring has the highest value?",
                  "Among the outer ring segments, which is the largest?"]
            return _P[sidx], ol[idx]

        elif qtype == "largest_inner":
            idx = iv.index(max(iv))
            _P = ["Which segment in the inner ring is the largest?",
                  "Identify the inner-ring segment with the maximum value.",
                  "Which inner-ring segment has the highest value?",
                  "Find the largest segment in the inner ring.",
                  "What inner-ring category is the biggest?",
                  "Which category in the inner ring is the largest by value?",
                  "In the inner ring of the donut chart, which segment is largest?",
                  "Which inner-ring slice is the biggest?",
                  "Pick the inner-ring segment with the greatest value.",
                  "Which inner ring category has the max value?",
                  "Find the inner-ring segment with the top value.",
                  "Which inner-ring segment dominates the inner ring?",
                  "Report the largest segment from the inner ring.",
                  "Largest inner-ring segment?",
                  "Which slice of the inner ring has the highest value?",
                  "Among the inner ring segments, which is the largest?"]
            return _P[sidx], il[idx]

        elif qtype == "outer_percentage":
            idx = rng.randint(0, len(ol) - 1)
            label = ol[idx]
            pct = round(ov[idx] / outer_total * 100, 1)
            _P = [f"What percentage of the outer ring does '{label}' represent? Round to 1 decimal place.",
                  f"Compute the percentage of the outer ring taken by '{label}'. Round to 1 decimal.",
                  f"What share (as %) of the outer-ring total is '{label}'? 1 decimal place.",
                  f"Find the percentage of the outer-ring total contributed by '{label}' (1 d.p.).",
                  f"'{label}' contributes what % of the outer ring? Round to 1 decimal place.",
                  f"Calculate the percentage of the outer ring represented by '{label}' (1 decimal).",
                  f"What portion (in %) of the outer ring does '{label}' cover? 1 d.p.",
                  f"Determine the outer-ring percentage for '{label}' (round to 1 decimal place).",
                  f"What is the % of the outer-ring total for '{label}'? Round to 1 decimal.",
                  f"Compute the percent share of the outer ring held by '{label}'. 1 decimal place.",
                  f"As a percentage of the outer ring, how much does '{label}' account for? (1 d.p.)",
                  f"Report '{label}'s percentage of the outer-ring total. Round to 1 decimal.",
                  f"What % of the outer-ring sum is '{label}'? One-decimal answer.",
                  f"Find the percentage contribution of '{label}' to the outer ring (1 d.p.).",
                  f"How much of the outer ring (in %) is '{label}'? 1-decimal answer.",
                  f"Calculate the fraction (as %, 1 d.p.) of the outer ring taken by '{label}'."]
            return _P[sidx], str(pct)

        elif qtype == "inner_vs_outer_total":
            diff = abs(outer_total - inner_total)
            bigger = "outer" if outer_total >= inner_total else "inner"
            _P = ["Which ring has a larger total raw value, outer or inner?",
                  "Between the outer and inner rings, which has the larger total value?",
                  "Compare the ring totals: is the outer or inner ring's sum larger?",
                  "Which ring in the donut chart totals more in raw value: outer or inner?",
                  "Does the outer or inner ring have a higher sum of values?",
                  "Which ring (outer/inner) has a greater combined total?",
                  "Is the outer ring or inner ring's total value bigger?",
                  "Determine the ring with the larger raw-value total (answer 'outer' or 'inner').",
                  "Compare outer-ring total to inner-ring total; which is larger?",
                  "Which ring's aggregate value is greater, outer or inner?",
                  "Outer total vs inner total: which wins (bigger)?",
                  "Which has a higher sum — the outer ring or the inner ring?",
                  "Report whether the outer or inner ring has the larger total.",
                  "Is the outer ring larger or smaller in total value than the inner ring? Answer 'outer' or 'inner'.",
                  "Which ring is greater by total, outer or inner?",
                  "Identify the ring (outer or inner) with the larger sum."]
            return _P[sidx], bigger

        elif qtype == "smallest_outer":
            idx = ov.index(min(ov))
            _P = ["Which segment in the outer ring is the smallest?",
                  "Identify the outer-ring segment with the minimum value.",
                  "Which outer-ring segment has the lowest value?",
                  "Find the smallest segment in the outer ring.",
                  "What outer-ring category has the minimum value?",
                  "Which category in the outer ring is the smallest?",
                  "In the outer ring, which segment has the least value?",
                  "Which outer-ring slice is the smallest?",
                  "Pick the outer-ring segment with the least value.",
                  "Which outer ring category has the min value?",
                  "Find the outer-ring segment with the lowest value.",
                  "Which outer category takes the smallest portion of its ring?",
                  "Report the smallest segment from the outer ring.",
                  "Smallest outer-ring segment?",
                  "Which slice of the outer ring has the lowest value?",
                  "Among the outer ring segments, which is the smallest?"]
            return _P[sidx], ol[idx]

        elif qtype == "count_above_pct":
            thresh = rng.choice([15, 20, 25])
            cnt = sum(1 for v in ov if v / outer_total * 100 > thresh)
            _P = [f"How many outer ring segments represent more than {thresh}% of the outer ring total?",
                  f"Count the outer-ring segments whose share exceeds {thresh}% of the outer total.",
                  f"How many outer-ring segments have more than {thresh}% of the total outer-ring value?",
                  f"In the outer ring, how many segments are >{thresh}% of the outer-ring total?",
                  f"Count outer-ring slices with share > {thresh}%.",
                  f"How many segments of the outer ring exceed {thresh}% of its total?",
                  f"Determine the number of outer-ring segments above {thresh}% of the outer total.",
                  f"How many categories in the outer ring hold more than {thresh}% of its total?",
                  f"Count outer-ring segments above the {thresh}% threshold.",
                  f"How many outer-ring slices are strictly greater than {thresh}% of the total?",
                  f"Find the count of outer-ring segments exceeding {thresh}% of the outer-ring total.",
                  f"How many outer-ring segments contribute over {thresh}% to the outer-ring total?",
                  f"Count: how many outer-ring slices represent > {thresh}% of outer-ring sum?",
                  f"Number of outer-ring segments with share greater than {thresh}%?",
                  f"How many outer-ring categories make up more than {thresh}% of the outer-ring total?",
                  f"Count the outer ring segments > {thresh}% of the outer-ring total."]
            return _P[sidx], str(cnt)

        else:
            idx = ov.index(max(ov))
            _P = ["Which outer ring segment has the highest value?",
                  "Identify the outer-ring segment with max value.",
                  "Which outer-ring slice is the largest by value?",
                  "Find the outer-ring segment holding the maximum value.",
                  "Which outer-ring category has the highest raw value?",
                  "Outer-ring segment with the max value?",
                  "Largest outer-ring value belongs to which segment?",
                  "Among outer-ring segments, which has the greatest value?",
                  "Pick the outer-ring segment with the top raw value.",
                  "Which outer-ring label has the highest value?",
                  "Which outer-ring slice has the highest raw value?",
                  "Max-value outer-ring segment?",
                  "Identify the outer-ring segment with the greatest value.",
                  "Which of the outer ring segments has the highest value?",
                  "Top-value outer-ring segment?",
                  "Report the outer-ring segment with the largest value."]
            return _P[sidx], ol[idx]
