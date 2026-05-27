"""
Chart with deliberately omitted information. The model is asked a question
whose answer is NOT present on the chart. The correct response is the
literal lowercase word `unanswerable`. About half of generated problems are
genuine readable questions (so the model must not blanket-refuse).

Mirrors verbatim Q-IDX 220 ("how much did victoria spend on gas in 2021?"
GT: Unanswerable), Q-IDX 779 ("name of the county with the maximum number
of cases"), Q-IDX 433 ("export destination received the most meat"),
Q-IDX 1473 (causal attribution claim that chart can't support),
Q-IDX 60 ("what were the main issues driving voter preferences"),
Q-IDX 1015 (counterfactual on a dimension not shown), Q-IDX 1019
(median across a column not on chart), Q-IDX 855 ("exact number of
hispanic parents"), Q-IDX 423 ("avg uber/lyft trip cost"), Q-IDX 1845
(MCQ where chart lacks needed range, options 30/50/70/90, GT
"unanswerable"), Q-IDX 1875, Q-IDX 1840, Q-IDX 1823.

Question style: factoid-style "what is X for Y?" but Y is something the
chart doesn't break down by, or a derived calculation that requires data
not labelled. Some are MCQ (4 lettered numeric options) where all 4 are
plausible-sounding distractors; correct answer is `unanswerable`.

L0: trivial — paragraph context says explicitly the chart shows X but the
question asks for Y not in chart.
L>0: subtler — half the time the question IS answerable; half it isn't.
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


# Questions where the chart genuinely lacks the asked info.
_UNANSWERABLE_TEMPLATES = [
    "What is the total amount of {asked_metric} in {asked_year}? If the chart does not show this, reply with the single word `unanswerable` (lowercase).",
    "How much {asked_metric} was reported by {asked_subject}? If the chart cannot answer, reply `unanswerable`.",
    "What is the average {asked_metric} across all {asked_subject}? If unavailable from the chart, reply `unanswerable`.",
    "Which {asked_subject} had the highest {asked_metric}? If the chart does not provide this breakdown, reply `unanswerable`.",
    "What was the {asked_metric} during {asked_year}? If the chart cannot answer, reply `unanswerable`.",
    "Report the median {asked_metric} for the {asked_subject} category. If not derivable from the chart, reply `unanswerable`.",
    "What is the difference in {asked_metric} between two {asked_subject} groups? If the chart lacks this, reply `unanswerable`.",
    "Identify the {asked_subject} that had the lowest {asked_metric}. If unavailable, reply `unanswerable`.",
    "What proportion of total {asked_metric} corresponds to {asked_subject}? If the chart does not show this, reply `unanswerable`.",
]

# Genuinely answerable questions — mirroring reference factoid style.
_ANSWERABLE_TEMPLATES = [
    "What value does the bar labeled '{label}' have in the chart? Numeric answer (no units).",
    "Read the chart. What is the value of '{label}'? Numeric.",
    "From the chart, report the value of '{label}'. Numeric.",
    "What numeric value is shown for the '{label}' bar in the chart?.",
    "Find the '{label}' bar and report its height. Numeric.",
    "Inspect the chart. What value does the '{label}' bar display? Numeric.",
    "What is the value of '{label}' in the chart? Numeric.",
    "Read off the value of '{label}' from the chart. Numeric.",
]


_TOPICS = [
    {
        "title": "Annual Sales by Region (2023)",
        "y_label": "Sales (millions USD)",
        "categories": ["North", "South", "East", "West"],
        # Chart tracks REGIONS in 2023. UNANSWERABLE asks for a year (2021)
        # or a subject (manager) that's not on chart.
        "trap_metric_year": ["2019", "2020", "2021"],
        "trap_subject_breakdown": ["product line", "manager", "store size"],
        "answerable_metric": "Sales",
    },
    {
        "title": "Customer Satisfaction by Service Type",
        "y_label": "Avg Satisfaction Score",
        "categories": ["Online", "Phone", "In-Store", "Mobile App"],
        "trap_metric_year": ["2019", "2018", "2017"],
        "trap_subject_breakdown": ["age group", "income bracket", "gender"],
        "answerable_metric": "Satisfaction",
    },
    {
        "title": "Quarterly Revenue by Department",
        "y_label": "Revenue (k USD)",
        "categories": ["Sales", "Engineering", "Marketing", "Support"],
        "trap_metric_year": ["2019", "2020", "2018"],
        "trap_subject_breakdown": ["region", "city", "country"],
        "answerable_metric": "Revenue",
    },
    {
        "title": "Active Users by Platform",
        "y_label": "Users (thousands)",
        "categories": ["iOS", "Android", "Web", "Desktop"],
        "trap_metric_year": ["2018", "2017", "2019"],
        "trap_subject_breakdown": ["country", "subscription tier", "age band"],
        "answerable_metric": "Users",
    },
    {
        "title": "Annual Energy Use by Source (2024)",
        "y_label": "Use (TWh)",
        "categories": ["Solar", "Wind", "Hydro", "Coal"],
        "trap_metric_year": ["2010", "2015", "2018"],
        "trap_subject_breakdown": ["country", "company", "season"],
        "answerable_metric": "Energy use",
    },
    {
        "title": "Application Counts by School (Spring)",
        "y_label": "Applications",
        "categories": ["Cambridge", "Oxford", "Imperial", "UCL"],
        "trap_metric_year": ["Fall", "Winter", "Summer"],
        "trap_subject_breakdown": ["major", "country of origin", "scholarship type"],
        "answerable_metric": "Applications",
    },
]


class ChartUnanswerableTrapQA(StandaloneVisualEnv):
    ENV_NAME = "chart_unanswerable_trap"
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
        level = max(0, min(level, 9))
        if level == 0:
            return {"unanswerable_prob": 1.0, "n_bars": 4}
        if level <= 2:
            return {"unanswerable_prob": 0.7, "n_bars": 4}
        if level <= 5:
            return {"unanswerable_prob": 0.5, "n_bars": 5}
        if level <= 7:
            return {"unanswerable_prob": 0.5, "n_bars": 6}
        # 2026-05-04: bumped L9 difficulty (also L8) — was 92.5% saturated.
        if level == 8:
            return {"unanswerable_prob": 0.5, "n_bars": 8}
        return {"unanswerable_prob": 0.5, "n_bars": 10}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        import hashlib
        sd = int(hashlib.md5(f"cut|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        topic = rng.choice(_TOPICS)
        n = min(cfg["n_bars"], len(topic["categories"]))
        cats = topic["categories"][:n]
        # bar values
        bar_values = rng.sample(range(15, 200), n)

        is_unanswerable = rng.random() < cfg["unanswerable_prob"]

        if is_unanswerable:
            tmpl = rng.choice(_UNANSWERABLE_TEMPLATES)
            asked_year = rng.choice(topic["trap_metric_year"])
            asked_subject = rng.choice(topic["trap_subject_breakdown"])
            asked_metric = topic["answerable_metric"].lower()
            question = tmpl.format(
                asked_metric=asked_metric,
                asked_year=asked_year,
                asked_subject=asked_subject,
            )
            answer = "unanswerable"
        else:
            tmpl = rng.choice(_ANSWERABLE_TEMPLATES)
            i = rng.randrange(n)
            question = tmpl.format(label=cats[i])
            answer = str(int(bar_values[i]))

        img = self._render_chart(topic, cats, bar_values, rng)
        return question, answer, img

    def _render_chart(self, topic: Dict, cats: List[str], values: List[int],
                       rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bars = ax.bar(range(len(cats)), values,
                       color=[palette[i % len(palette)] for i in range(len(cats))],
                       edgecolor="#222", linewidth=0.6)
        max_v = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.02, str(int(v)),
                    ha="center", va="bottom", fontsize=10, color="#111",
                    fontweight="bold")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel(topic["y_label"], fontsize=10)
        ax.set_title(topic["title"], fontsize=12, loc="left")
        ax.set_ylim(0, max_v * 1.25 + 5)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
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
        import re
        p = predicted.strip().lower().rstrip(".").rstrip(",").strip()
        g = ground_truth.strip().lower().rstrip(".").strip()
        if g == "unanswerable":
            # Accept "unanswerable", "the answer is unanswerable", "n/a",
            # "not answerable" — but be strict against any number leak.
            if "unanswerable" in p:
                return True
            if re.search(r"\b(n/a|na|cannot\s+(?:be\s+)?(?:determined|answered)|not\s+(?:answerable|determinable|available))\b", p):
                return True
            return False
        # Numeric ground truth — fall back to base numeric handling.
        return super()._check_answer(predicted, ground_truth)
