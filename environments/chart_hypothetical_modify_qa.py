"""
Chart hypothetical modify-then-recompute.

Render a bar chart, then ask a question with a counterfactual modifier:
  - "If X were doubled, what would Y be?" (numeric)
  - "If X decreased by N%, what would the new total / new Y be?" (numeric)
  - "If X were doubled, would the sum exceed Z?" (Yes/No)

Critical: model must imagine the modification — chart is NOT redrawn.

Mirrors Hypothetical reference samples:
  - IDX 1456 ("if italy's gdp had declined twice as much... new gdp pct?" -> -18.2)
  - IDX 1457 ("if percentage increased by 50%, what would be the new percentage?" -> 15.45)
  - IDX 1364 ("if avg wealth in asia increases by 50%, new avg?" -> 45.3K)
  - IDX 1430 ("if sign-ups in week 25 had decreased by 100, new total?" -> 180)
  - IDX 1380 ("if births had been 10% lower, what would deaths-births be?" -> 74757)
  - IDX 1427 ("if stock investment had been 50% higher, total = ?" -> 7.5)
  - IDX 1400 ("if china's peak had been 2 lower, would it still be higher than azerbaijan?" -> No)
  - IDX 1432 ("if women caregivers doubled, would females be leading category?" -> No)

Output: numeric (modified value or new total) or Yes/No (would-it-exceed).
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
from ._mcq_letter_lib import maybe_to_mcq_letter


_TEMPLATES_DOUBLE_VALUE = [
    "Look at the bar chart. If the value of {label} were doubled, what would the new value be? Place the numeric answer in <answer>...</answer>.",
    "If {label}'s value in the chart were doubled, what would the new value of {label} be? Numeric answer in <answer>...</answer>.",
    "Suppose {label} had double its current value as shown in the chart. What would the new value be? Numeric in <answer>...</answer>.",
    "From the chart, if {label} were twice as large, what would the new value of {label} be? Numeric in <answer>...</answer>.",
]

_TEMPLATES_HALF_VALUE = [
    "Look at the bar chart. If the value of {label} were halved, what would the new value be? Place the numeric answer in <answer>...</answer>.",
    "If {label}'s value in the chart were halved, what would the new value of {label} be? Numeric in <answer>...</answer>.",
    "Suppose {label} had half its current value as shown in the chart. What would the new value be? Numeric in <answer>...</answer>.",
    "From the chart, if {label} were reduced to half, what would the new value of {label} be? Numeric in <answer>...</answer>.",
]

_TEMPLATES_PCT_INCREASE = [
    "Look at the bar chart. If the value of {label} increased by {pct}%, what would the new value be? Numeric in <answer>...</answer>.",
    "If {label} from the chart were {pct}% higher, what would the new value of {label} be? Numeric in <answer>...</answer>.",
    "Suppose {label}'s value rose by {pct}%. What would the new value be? Numeric in <answer>...</answer>.",
    "From the chart, what would {label} be after a {pct}% increase? Numeric in <answer>...</answer>.",
]

_TEMPLATES_PCT_DECREASE = [
    "Look at the bar chart. If the value of {label} decreased by {pct}%, what would the new value be? Numeric in <answer>...</answer>.",
    "If {label} from the chart were {pct}% lower, what would the new value of {label} be? Numeric in <answer>...</answer>.",
    "Suppose {label}'s value fell by {pct}%. What would the new value be? Numeric in <answer>...</answer>.",
    "From the chart, what would {label} be after a {pct}% decrease? Numeric in <answer>...</answer>.",
]

_TEMPLATES_NEW_TOTAL_DOUBLE = [
    "If {label} in the bar chart were doubled, what would the new total across all bars be? Numeric in <answer>...</answer>.",
    "Suppose {label} doubled. What is the new sum of all the bar values? Numeric in <answer>...</answer>.",
    "After doubling {label}, what would the new total of all the bars be? Numeric in <answer>...</answer>.",
    "If {label}'s value in the chart were twice what is shown, compute the new sum across all categories. Numeric in <answer>...</answer>.",
]

_TEMPLATES_NEW_TOTAL_PCT = [
    "If {label} in the bar chart {direction} by {pct}%, what would the new sum across all bars be? Numeric in <answer>...</answer>.",
    "Suppose {label} {direction} by {pct}%. What is the new total of all the bar values? Numeric in <answer>...</answer>.",
    "After {label} {direction} by {pct}%, what would the new sum across all categories be? Numeric in <answer>...</answer>.",
    "If {label}'s value were {direction} by {pct}%, compute the new total across the chart. Numeric in <answer>...</answer>.",
]

_TEMPLATES_EXCEED_DOUBLE = [
    "If {label} in the bar chart were doubled, would its new value exceed {target_label}? Reply with `Yes` or `No` in <answer>...</answer>.",
    "Suppose {label} doubled. Would the new value be greater than {target_label}? Yes/No in <answer>...</answer>.",
    "After doubling {label}, would it exceed the value of {target_label}? Reply Yes or No in <answer>...</answer>.",
    "If {label} were twice its current value as shown, would it surpass {target_label}? Yes/No in <answer>...</answer>.",
]

_TEMPLATES_EXCEED_PCT = [
    "If {label} {direction} by {pct}%, would its new value exceed {target_label}? Reply with `Yes` or `No` in <answer>...</answer>.",
    "Suppose {label} {direction} by {pct}%. Would the new value be greater than {target_label}? Yes/No in <answer>...</answer>.",
    "After {label} {direction} by {pct}%, would it exceed the value of {target_label}? Reply Yes or No in <answer>...</answer>.",
]


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches", "Pears", "Cherries"],
    ["United States", "China", "Germany", "Japan", "India", "Brazil"],
    ["Sales", "Engineering", "Marketing", "Finance", "Operations", "Support"],
    ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"],
    ["Region A", "Region B", "Region C", "Region D", "Region E"],
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Hours"]


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class ChartHypotheticalModifyQA(StandaloneVisualEnv):
    ENV_NAME = "chart_hypothetical_modify"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_cat": 4, "modes": ["double_value"], "trivial_l0": True}
        if level <= 2:
            return {"n_cat": 5, "modes": ["double_value", "half_value", "pct_inc", "pct_dec"],
                    "trivial_l0": False}
        if level <= 4:
            return {"n_cat": 6,
                    "modes": ["double_value", "half_value", "pct_inc", "pct_dec",
                              "new_total_double"],
                    "trivial_l0": False}
        if level <= 6:
            return {"n_cat": 6,
                    "modes": ["pct_inc", "pct_dec", "new_total_double",
                              "new_total_pct", "exceed_double"],
                    "trivial_l0": False}
        if level <= 8:
            return {"n_cat": 7,
                    "modes": ["pct_inc", "pct_dec", "new_total_pct",
                              "exceed_double", "exceed_pct"],
                    "trivial_l0": False}
        return {"n_cat": 8,
                "modes": ["new_total_pct", "exceed_double", "exceed_pct"],
                "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1373 + level * 137 + 41)

        cat_pool = rng.choice(_CATEGORY_POOLS)
        n = min(cfg["n_cat"], len(cat_pool))
        cats = rng.sample(cat_pool, n)
        y_label = rng.choice(_Y_LABELS)

        if cfg["trivial_l0"]:
            # 4 bars [10, 20, 30, 40], "if A doubled" -> 20.
            values = [10, 20, 30, 40]
            rng.shuffle(values)
            mode = "double_value"
            target_idx = rng.randrange(n)
            answer_val = values[target_idx] * 2
            answer = _format_num(answer_val)
            label = cats[target_idx]
            tlist = _TEMPLATES_DOUBLE_VALUE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label)
            img = self._render_bars(cats, values, y_label, rng)
            n_opts = rng.choice([4, 5])
            question, answer = maybe_to_mcq_letter(
                question, answer, rng, prob=0.5, n_options=n_opts)
            return question, answer, img

        # Non-trivial generation
        if level <= 3:
            values = [rng.randint(2, 12) * 10 for _ in range(n)]
        else:
            values = [rng.randint(15, 200) for _ in range(n)]

        mode = rng.choice(cfg["modes"])
        target_idx = rng.randrange(n)
        label = cats[target_idx]
        v = values[target_idx]

        if mode == "double_value":
            answer_val = v * 2
            tlist = _TEMPLATES_DOUBLE_VALUE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label)
            answer = _format_num(answer_val)
        elif mode == "half_value":
            answer_val = v / 2.0
            tlist = _TEMPLATES_HALF_VALUE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label)
            answer = _format_num(answer_val)
        elif mode == "pct_inc":
            pct = rng.choice([10, 20, 25, 50])
            answer_val = v * (1 + pct / 100.0)
            tlist = _TEMPLATES_PCT_INCREASE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label, pct=pct)
            answer = _format_num(answer_val)
        elif mode == "pct_dec":
            pct = rng.choice([10, 20, 25, 50])
            answer_val = v * (1 - pct / 100.0)
            tlist = _TEMPLATES_PCT_DECREASE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label, pct=pct)
            answer = _format_num(answer_val)
        elif mode == "new_total_double":
            new_v = v * 2
            answer_val = sum(values) - v + new_v
            tlist = _TEMPLATES_NEW_TOTAL_DOUBLE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label)
            answer = _format_num(answer_val)
        elif mode == "new_total_pct":
            pct = rng.choice([10, 20, 25, 50])
            up = rng.choice([True, False])
            direction = "increased" if up else "decreased"
            mult = (1 + pct / 100.0) if up else (1 - pct / 100.0)
            new_v = v * mult
            answer_val = sum(values) - v + new_v
            tlist = _TEMPLATES_NEW_TOTAL_PCT
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label, pct=pct, direction=direction)
            answer = _format_num(answer_val)
        elif mode == "exceed_double":
            # Pick a different category as the comparator
            other_idx = rng.choice([i for i in range(n) if i != target_idx])
            target_label = cats[other_idx]
            new_v = v * 2
            other_v = values[other_idx]
            answer = "Yes" if new_v > other_v else "No"
            tlist = _TEMPLATES_EXCEED_DOUBLE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label, target_label=target_label)
        elif mode == "exceed_pct":
            other_idx = rng.choice([i for i in range(n) if i != target_idx])
            target_label = cats[other_idx]
            pct = rng.choice([10, 20, 25, 50])
            up = rng.choice([True, False])
            direction = "increased" if up else "decreased"
            mult = (1 + pct / 100.0) if up else (1 - pct / 100.0)
            new_v = v * mult
            other_v = values[other_idx]
            answer = "Yes" if new_v > other_v else "No"
            tlist = _TEMPLATES_EXCEED_PCT
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=label, pct=pct,
                                           direction=direction,
                                           target_label=target_label)
        else:
            return None

        img = self._render_bars(cats, values, y_label, rng)

        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        # For exceed_double / exceed_pct (Yes/No), use Yes/No candidate pool.
        n_opts = rng.choice([4, 5])
        if mode in ("exceed_double", "exceed_pct"):
            cand_pool = ["Yes", "No"]
            n_opts = 2  # Only Yes/No options
        else:
            cand_pool = None
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts,
            candidate_pool=cand_pool)
        return question, answer, img

    def _render_bars(self, cats: List[str], values: List[float], y_label: str,
                     rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        n = len(cats)
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bar_colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(range(n), values, color=bar_colors,
                       edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.018,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 4)
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
