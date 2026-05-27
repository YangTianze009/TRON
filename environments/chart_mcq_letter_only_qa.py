"""
Chart MCQ Letter-Only QA — C20 (reference Multi Choice subtask).

Format-discipline trainer: render any of bar/line/pie/percent-change chart
families, ask a numeric question, and force the answer into a 4-option
lettered MCQ block whose only valid output is a bare lowercase letter
``a|b|c|d``.

Verbatim sample style from design notes Multi Choice:

    Q-IDX 1869:
      "what is the difference in percentage points between chinese and
       australian respondents who believe music helps create a better
       shopping environment?\\n\\na) 10 percentage points\\nb) 8 percentage
       points\\nc) 12 percentage points\\nd) 6 percentage points"
      GT = ``b`` (lowercase)

    Q-IDX 1738:
      "if saturday's sales are $27,312, what percentage higher are these
       sales compared to sunday's sales?  opts 5.23/7.23/9.23/11.23"
      GT = ``c``

    Q-IDX 1848:
      "what is the approximate ratio of dollars collected per dollar spent
       in oregon compared to south dakota?  opts 0.6/0.7/0.8/0.9"
      GT = ``a``

The +9.35 delta on this subtask in v3-step175 was almost entirely format
discipline (bare letter vs ``c) 9.23%``); this env stress-trains that.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_helper import build_numeric_mcq, format_mcq_question


_CATEGORY_POOLS = [
    ["USA", "China", "Germany", "Japan", "India", "Brazil",
     "France", "UK", "Canada", "Australia"],
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes",
     "Peaches", "Pears", "Cherries"],
    ["Sales", "Engineering", "Marketing", "Finance", "HR",
     "Legal", "Support", "R&D"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Dallas"],
    ["Plan A", "Plan B", "Plan C", "Plan D"],
]

_Y_LABELS = ["Revenue ($M)", "Units Sold", "Visitors (K)", "Score",
             "Customers", "Sales (%)"]

_PERIOD_POOLS = [
    ["2018", "2019", "2020", "2021", "2022"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    ["Q1", "Q2", "Q3", "Q4"],
]


class ChartMcqLetterOnlyQA(StandaloneVisualEnv):
    ENV_NAME = "chart_mcq_letter_only"
    # Strict bare-text exact-match scoring on raw
    # prediction (no wrapper-stripping). Override the default 3-of-3 wrapper
    # instructions to teach the model to put the bare answer on the final line.
    # Verifier was extended to extract the last non-empty line as a candidate.
    _WRAPPER_INSTRUCTIONS = [
        "Think step by step. End your response with the bare answer (single letter / true|false / single word) on its own final line, with no wrapper around it.",
        "Reason through the problem, then on the very last line of your response output ONLY the bare answer (a single letter, true|false, or a single word) - no <answer>, \\boxed{}, or 'Final answer:' prefix.",
        "Work through the problem step by step. Your final line must be the bare answer alone (single letter / true|false / single word), nothing else on that line.",
    ]

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/?/100. Each
        # level now provides distinct training signal.
        level = max(0, min(level, 9))
        if level <= 1:
            # Was just value_of/argmax — now adds diff to make L0 non-trivial.
            return {"chart_kind": "bar",
                    "qtype_pool": ["value_of", "argmax_index", "difference"]}
        if level <= 3:
            return {"chart_kind": "bar",
                    "qtype_pool": ["difference", "ratio", "weighted_avg"]}
        if level <= 5:
            return {"chart_kind": "line",
                    "qtype_pool": ["max_value", "value_at",
                                   "first_to_last_change",
                                   "pct_change_two_periods"]}
        if level <= 7:
            return {"chart_kind": "pie",
                    "qtype_pool": ["pct_share", "ratio_two_slices"],
                    "n_categories_max": 7}
        # 2026-05-04: bumped L9 difficulty — was 95% saturated.
        # More categories + only weighted_avg (hardest).
        return {"chart_kind": "bar",
                "qtype_pool": ["weighted_avg"],
                "n_categories_max": 10,
                "hide_labels": True,
                "tight_distractors": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1117 + level * 53 + 13)
        chart_kind = cfg["chart_kind"]

        # Retry internally to handle degenerate seeds (e.g. divide-by-zero
        # in ratio mode, distractors collapsing onto correct value).
        for _ in range(20):
            qtype = rng.choice(cfg["qtype_pool"])
            try:
                n_cats_max = cfg.get("n_categories_max", 6)
                if chart_kind == "bar":
                    res = self._gen_bar(rng, qtype, n_cats_max=n_cats_max)
                elif chart_kind == "line":
                    res = self._gen_line(rng, qtype)
                elif chart_kind == "pie":
                    res = self._gen_pie(rng, qtype)
                else:
                    res = None
            except Exception:
                res = None
            if res is not None:
                return res
        return None

    # -------------------------------------------------------------- #
    # Bar variants
    # -------------------------------------------------------------- #

    def _gen_bar(self, rng, qtype, n_cats_max=6):
        cats = list(rng.choice(_CATEGORY_POOLS))
        rng.shuffle(cats)
        n = rng.choice(list(range(4, n_cats_max + 1)))
        cats = cats[:n]
        values = [rng.randint(15, 200) for _ in range(n)]
        # Ensure max is unique (so argmax stories are unambiguous)
        while values.count(max(values)) > 1:
            values[values.index(max(values))] += rng.randint(1, 5)

        y_label = rng.choice(_Y_LABELS)

        if qtype == "value_of":
            i = rng.randrange(n)
            stem = f"What is the {y_label.lower()} of {cats[i]}?"
            correct = float(values[i])
            suffix = ""
        elif qtype == "argmax_index":
            # Use bar position MCQ; correct is the cat name -> distractor pool of other cats
            stem = f"Which category has the highest {y_label.lower()}?"
            i = values.index(max(values))
            correct_name = cats[i]
            other = [c for c in cats if c != correct_name]
            rng.shuffle(other)
            options = [correct_name] + other[:3]
            rng.shuffle(options)
            letter = chr(ord("a") + options.index(correct_name))
            return format_mcq_question(stem, options), letter, \
                self._render_bar(rng, cats, values, y_label)
        elif qtype == "difference":
            i, j = rng.sample(range(n), 2)
            stem = (f"What is the absolute difference in {y_label.lower()} "
                    f"between {cats[i]} and {cats[j]}?")
            correct = float(abs(values[i] - values[j]))
            suffix = ""
        elif qtype == "ratio":
            i, j = rng.sample(range(n), 2)
            if values[j] == 0:
                return None
            ratio = round(values[i] / values[j], 2)
            stem = (f"What is the ratio of {cats[i]} to {cats[j]} in "
                    f"{y_label.lower()}? (rounded to 2 decimal places)")
            correct = float(ratio)
            suffix = ""
        elif qtype == "weighted_avg":
            weights = list(range(1, n + 1))
            wavg = round(sum(w * v for w, v in zip(weights, values))
                         / sum(weights), 2)
            stem = (f"Assign weights 1, 2, ..., {n} to the bars from left "
                    f"to right and compute the weighted average. (rounded "
                    f"to 2 decimal places)")
            correct = float(wavg)
            suffix = ""
        elif qtype == "pct_change_two_periods":
            i, j = rng.sample(range(n), 2)
            base = values[j]
            if base == 0:
                return None
            change = round((values[i] - base) / base * 100, 1)
            stem = (f"What is the percentage change in {y_label.lower()} "
                    f"from {cats[j]} to {cats[i]}? (positive = increase, "
                    f"negative = decrease)")
            correct = float(change)
            suffix = "%"
        else:
            return None

        options, letter = build_numeric_mcq(rng, correct, suffix=suffix)
        question = format_mcq_question(stem, options)
        image = self._render_bar(rng, cats, values, y_label)
        return question, letter, image

    # -------------------------------------------------------------- #
    # Line variants
    # -------------------------------------------------------------- #

    def _gen_line(self, rng, qtype):
        periods = list(rng.choice(_PERIOD_POOLS))
        n = min(rng.choice([5, 6, 7]), len(periods))
        periods = periods[:n]
        values = [rng.randint(20, 200) for _ in range(n)]
        while len(set(values)) != len(values):
            values[rng.randrange(n)] += rng.randint(1, 4)
        y_label = rng.choice(_Y_LABELS)

        if qtype == "max_value":
            stem = f"What is the maximum {y_label.lower()} across all periods?"
            correct = float(max(values))
            suffix = ""
        elif qtype == "value_at":
            i = rng.randrange(n)
            stem = f"What is the {y_label.lower()} at {periods[i]}?"
            correct = float(values[i])
            suffix = ""
        elif qtype == "first_to_last_change":
            change = values[-1] - values[0]
            stem = (f"What is the net change from {periods[0]} to "
                    f"{periods[-1]} (last minus first)?")
            correct = float(change)
            suffix = ""
        else:
            return None

        options, letter = build_numeric_mcq(rng, correct, suffix=suffix)
        question = format_mcq_question(stem, options)
        image = self._render_line(rng, periods, values, y_label)
        return question, letter, image

    # -------------------------------------------------------------- #
    # Pie variants
    # -------------------------------------------------------------- #

    def _gen_pie(self, rng, qtype):
        cats = list(rng.choice(_CATEGORY_POOLS))
        rng.shuffle(cats)
        n = rng.choice([4, 5])
        cats = cats[:n]
        # Generate shares that sum to 100
        raw = [rng.randint(10, 40) for _ in range(n)]
        total = sum(raw)
        shares = [round(v / total * 100, 1) for v in raw]
        # Re-balance the last so total == 100
        shares[-1] = round(100 - sum(shares[:-1]), 1)

        if qtype == "pct_share":
            i = rng.randrange(n)
            stem = (f"What percentage of the pie is the {cats[i]} slice?")
            correct = float(shares[i])
            suffix = "%"
        elif qtype == "ratio_two_slices":
            i, j = rng.sample(range(n), 2)
            if shares[j] == 0:
                return None
            ratio = round(shares[i] / shares[j], 2)
            stem = (f"What is the ratio of the {cats[i]} slice to the "
                    f"{cats[j]} slice? (rounded to 2 decimal places)")
            correct = float(ratio)
            suffix = ""
        else:
            return None

        options, letter = build_numeric_mcq(rng, correct, suffix=suffix)
        question = format_mcq_question(stem, options)
        image = self._render_pie(rng, cats, shares)
        return question, letter, image

    # -------------------------------------------------------------- #
    # Renderers
    # -------------------------------------------------------------- #

    def _render_bar(self, rng, cats, values, y_label):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        fig_w = max(6, len(cats) * 0.9 + 2) * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, 4.6 * style["figsize_scale"]))
        bars = ax.bar(range(len(cats)), values,
                      color=[palette[i % len(palette)] for i in range(len(cats))],
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02,
                    str(v), ha="center", va="bottom",
                    fontsize=style["font_size_base"] - 1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, fontsize=style["font_size_base"],
                            rotation=30 if max(len(c) for c in cats) > 6 else 0,
                            ha="right" if max(len(c) for c in cats) > 6 else "center")
        ax.set_ylabel(y_label, fontsize=style["font_size_base"])
        ax.set_title("Multiple-Choice Chart Question",
                     fontsize=style["font_size_base"] + 2, pad=10)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_line(self, rng, periods, values, y_label):
        style = self._random_style()
        palette = list(style["palette"])
        fig, ax = plt.subplots(figsize=(7 * style["figsize_scale"],
                                        4.5 * style["figsize_scale"]))
        ax.plot(range(len(periods)), values, marker="o",
                color=palette[0], linewidth=style["line_width"],
                markersize=7)
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.025, str(v),
                    ha="center", va="bottom",
                    fontsize=style["font_size_base"] - 1)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontsize=style["font_size_base"])
        ax.set_ylabel(y_label, fontsize=style["font_size_base"])
        ax.set_title("Multiple-Choice Chart Question",
                     fontsize=style["font_size_base"] + 2, pad=10)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_pie(self, rng, cats, shares):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        fig, ax = plt.subplots(figsize=(6 * style["figsize_scale"],
                                        5 * style["figsize_scale"]))
        ax.pie(shares, labels=cats,
               colors=[palette[i % len(palette)] for i in range(len(cats))],
               autopct=lambda p: f"{p:.1f}%",
               startangle=90,
               textprops={"fontsize": style["font_size_base"]})
        ax.set_title("Multiple-Choice Chart Question",
                     fontsize=style["font_size_base"] + 2, pad=10)
        fig.patch.set_facecolor(style["bg_color"])
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
