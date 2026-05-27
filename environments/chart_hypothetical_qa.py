"""
Chart Hypothetical Q&A — C33 (reference Hypothetical, counterfactual).

Render a chart, then ask "if X had happened, what would Y be?" — model must
apply a single-step counterfactual modification to the chart values and
report the new derived quantity.

Verbatim sample style (design notes Hypothetical):

    Q-IDX 1456: "if italy's gdp had declined twice as much as its actual
                 decline, what would be its new gdp percentage change?"
                 GT = `-18.2`

    Q-IDX 1430: "if sign-ups in week 25 of 2020 had decreased by 100, what
                 would the new total for that week be?"
                 GT = `180`

    Q-IDX 1457: "if the percentage of native american women rated as 'very
                 attractive' increased by 50%, what would be the new
                 percentage?"
                 GT = `15.45`

Format: Factoid-style answer (bare numeric, ±5 % tolerance).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CATEGORY_POOLS = [
    # 2026-05-04: extended pools to >=8 to support L9 n_cat_range bump.
    ["USA", "China", "Germany", "Japan", "India", "Brazil",
     "France", "UK", "Italy"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Dallas",
     "Boston", "Miami", "Seattle"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Plan A", "Plan B", "Plan C", "Plan D"],
    ["Sales", "Engineering", "Marketing", "Finance", "HR",
     "Legal", "Ops", "Support"],
    ["Apples", "Bananas", "Oranges", "Grapes",
     "Pears", "Plums", "Peaches", "Mangoes"],
]

_Y_LABELS = ["Revenue ($M)", "Sign-ups", "Visitors (K)", "Score",
             "Customers", "Sales (units)"]


class ChartHypotheticalQA(StandaloneVisualEnv):
    ENV_NAME = "chart_hypothetical_qa"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"qtypes": ["increase_by_pct", "decrease_by_amount"],
                    "n_cat_range": (3, 4)}
        if level <= 4:
            return {"qtypes": ["increase_by_pct", "decrease_by_amount",
                               "halve", "double", "modify_then_total"],
                    "n_cat_range": (4, 5)}
        if level <= 7:
            return {"qtypes": ["modify_then_total", "modify_then_diff",
                               "modify_then_average", "modify_then_ratio"],
                    "n_cat_range": (4, 6)}
        # 2026-05-04: bumped L9 difficulty — was 95% saturated.
        # More categories (some pools are <7, retry handles it) + harder qtypes only.
        # 2026-05-04 R3: benchmark-sample-driven harden — add two_modify
        # ("if X had increased by 20% AND Y had decreased by 10%, what would
        # the new total / ratio be") — IDX 1430+1457 style compound
        # hypothetical. Forces 2-step modify-then-aggregate.
        return {"qtypes": ["modify_then_ratio", "double_decline",
                           "two_modify_total", "two_modify_diff"],
                "n_cat_range": (6, 8)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1559 + level * 113 + 41)

        for _ in range(20):
            res = self._try(rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, cfg):
        n = rng.randint(*cfg["n_cat_range"])
        cats = list(rng.choice(_CATEGORY_POOLS))
        rng.shuffle(cats)
        if len(cats) < n:
            return None
        cats = cats[:n]
        vals = [rng.randint(50, 500) for _ in range(n)]
        y_label = rng.choice(_Y_LABELS)
        qtype = rng.choice(cfg["qtypes"])

        i = rng.randrange(n)

        if qtype == "increase_by_pct":
            pct = rng.choice([10, 20, 25, 50])
            new_v = round(vals[i] * (1 + pct / 100), 2)
            stem = (f"If {cats[i]} {y_label.lower()} had been {pct}% higher "
                    f"than the recorded value, what would the new value be?")
            answer = self._fmt(new_v)
        elif qtype == "decrease_by_amount":
            amt = rng.choice([20, 50, 100])
            new_v = vals[i] - amt
            if new_v <= 0:
                return None
            stem = (f"If {cats[i]} {y_label.lower()} had decreased by {amt}, "
                    f"what would the new value be?")
            answer = str(new_v)
        elif qtype == "halve":
            new_v = round(vals[i] / 2, 2)
            stem = (f"If {cats[i]} {y_label.lower()} had been halved, "
                    f"what would the new value be?")
            answer = self._fmt(new_v)
        elif qtype == "double":
            new_v = vals[i] * 2
            stem = (f"If {cats[i]} {y_label.lower()} had doubled, "
                    f"what would the new value be?")
            answer = str(new_v)
        elif qtype == "modify_then_total":
            pct = rng.choice([-30, -20, 20, 50])
            new_vi = vals[i] * (1 + pct / 100)
            new_total = round(sum(v if k != i else new_vi
                                  for k, v in enumerate(vals)), 2)
            sign = "increased" if pct > 0 else "decreased"
            stem = (f"If {cats[i]} {y_label.lower()} had {sign} by "
                    f"{abs(pct)}%, what would the new total across all "
                    f"categories be?")
            answer = self._fmt(new_total)
        elif qtype == "modify_then_diff":
            j = (i + 1) % n
            pct = rng.choice([-50, -25, 25, 50])
            new_vi = vals[i] * (1 + pct / 100)
            new_diff = round(abs(new_vi - vals[j]), 2)
            sign = "higher" if pct > 0 else "lower"
            stem = (f"If {cats[i]} {y_label.lower()} had been {abs(pct)}% "
                    f"{sign}, what would the absolute difference between "
                    f"{cats[i]} and {cats[j]} be?")
            answer = self._fmt(new_diff)
        elif qtype == "modify_then_average":
            pct = rng.choice([-25, 25, 50])
            new_vi = vals[i] * (1 + pct / 100)
            new_avg = round(sum(v if k != i else new_vi
                                for k, v in enumerate(vals)) / n, 2)
            sign = "higher" if pct > 0 else "lower"
            stem = (f"If {cats[i]} {y_label.lower()} had been {abs(pct)}% "
                    f"{sign}, what would the new average across all "
                    f"categories be?")
            answer = self._fmt(new_avg)
        elif qtype == "modify_then_ratio":
            j = (i + 2) % n
            pct = rng.choice([-25, 25, 50])
            new_vi = vals[i] * (1 + pct / 100)
            if vals[j] == 0:
                return None
            new_ratio = round(new_vi / vals[j], 2)
            sign = "higher" if pct > 0 else "lower"
            stem = (f"If {cats[i]} {y_label.lower()} had been {abs(pct)}% "
                    f"{sign}, what would the ratio of {cats[i]} to {cats[j]} "
                    f"be? (rounded to 2 decimal places)")
            answer = self._fmt(new_ratio)
        elif qtype == "double_decline":
            # IDX 1456 style: declined twice as much as actual.
            # Treat the original value relative to the chart-mean as the
            # "actual decline".
            mean = sum(vals) / n
            actual_decline = mean - vals[i]
            if actual_decline <= 0:
                return None
            new_v = round(vals[i] - actual_decline, 2)  # Decline twice as much
            stem = (f"If {cats[i]} had declined twice as much from the average "
                    f"({self._fmt(round(mean, 2))}) as it actually did, what "
                    f"would its new {y_label.lower()} be?")
            answer = self._fmt(new_v)
        elif qtype == "two_modify_total":
            # 2026-05-04 R3: benchmark-sample-driven harden — IDX 1430/1457
            # style 2-modify hypothetical. Increase one and decrease another,
            # then ask for the new total. Two simultaneous modifications.
            j = (i + 2) % n
            if i == j:
                return None
            pct_i = rng.choice([20, 25, 50])
            pct_j = rng.choice([-30, -20, -25])
            new_vi = vals[i] * (1 + pct_i / 100)
            new_vj = vals[j] * (1 + pct_j / 100)
            new_total = round(sum(vals[k] if k not in (i, j) else
                                  (new_vi if k == i else new_vj)
                                  for k in range(n)), 2)
            stem = (f"If {cats[i]} {y_label.lower()} had increased by "
                    f"{pct_i}% and at the same time {cats[j]} "
                    f"{y_label.lower()} had decreased by {abs(pct_j)}%, what "
                    f"would the new total across all categories be?")
            answer = self._fmt(new_total)
        elif qtype == "two_modify_diff":
            # 2026-05-04 R3: benchmark-sample-driven harden — apply two
            # modifications then ask absolute difference between the two
            # modified values.
            j = (i + 3) % n
            if i == j:
                return None
            pct_i = rng.choice([25, 50])
            pct_j = rng.choice([-30, -25])
            new_vi = vals[i] * (1 + pct_i / 100)
            new_vj = vals[j] * (1 + pct_j / 100)
            new_diff = round(abs(new_vi - new_vj), 2)
            stem = (f"If {cats[i]} {y_label.lower()} had increased by "
                    f"{pct_i}% and {cats[j]} {y_label.lower()} had decreased "
                    f"by {abs(pct_j)}%, what would the absolute difference "
                    f"between {cats[i]} and {cats[j]} be?")
            answer = self._fmt(new_diff)
        else:
            return None

        question = stem + " Provide just the numeric value."
        image = self._render(rng, cats, vals, y_label)
        return question, answer, image

    @staticmethod
    def _fmt(v):
        if isinstance(v, int) or float(v).is_integer():
            return str(int(v))
        return f"{v:.2f}".rstrip("0").rstrip(".")

    def _render(self, rng, cats, vals, y_label):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        n = len(cats)
        fig_w = max(6, n * 0.95 + 2) * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, 4.6 * style["figsize_scale"]))
        bars = ax.bar(range(n), vals,
                      color=[palette[i % len(palette)] for i in range(n)],
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max(vals) * 0.02, str(v),
                    ha="center", va="bottom",
                    fontsize=style["font_size_base"] - 1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, fontsize=style["font_size_base"],
                           rotation=20 if max(len(c) for c in cats) > 5 else 0,
                           ha="right" if max(len(c) for c in cats) > 5 else "center")
        ax.set_ylabel(y_label, fontsize=style["font_size_base"])
        ax.set_title("Hypothetical Reasoning",
                     fontsize=style["font_size_base"] + 2, pad=10)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
