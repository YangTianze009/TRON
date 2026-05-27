"""
Pie chart proportional rebalance.

Render a pie chart with N slices summing to 100%. Ask the model:
  "If X's share decreases by N (percentage points), what does Y need to
   increase to to keep the total at 100%?"

Mirrors Hypothetical reference samples:
  - IDX 1391 ("proportional adjustment to keep total at 100%" -> 39%)
  - IDX 965 ("if percentage spent on fun decreased by half, new ratio?" -> 2.33)
  - IDX 1444 ("suppose emissions increased to 30% of total. how many tonnes?" -> 15.72)

Output: numeric percentage (no % sign).
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


_TEMPLATES_INCREASE = [
    "Look at the pie chart. If {x_label}'s share decreases by {pct} percentage points, what would {y_label}'s new share need to be (in percent) to keep the total at 100%? Numeric answer, no % sign, in <answer>...</answer>.",
    "From the pie chart, if {x_label} drops by {pct} percentage points, what new percentage must {y_label} take to keep the total summing to 100%? Numeric (no %) in <answer>...</answer>.",
    "Suppose {x_label}'s slice in the pie chart shrinks by {pct} percentage points. Holding all other slices fixed, what does {y_label} need to be to keep the total at 100%? Integer percent (no % sign) in <answer>...</answer>.",
    "If {x_label}'s share in the pie chart decreased by {pct} percentage points and only {y_label} absorbed the change, what would {y_label}'s new share be (in percent)? Numeric (no %) in <answer>...</answer>.",
    "From the pie chart, {x_label} drops by {pct} pp. Compute {y_label}'s new share (in percent) so the chart still totals 100%. Numeric, no % sign, in <answer>...</answer>.",
]

_TEMPLATES_DECREASE = [
    "Look at the pie chart. If {x_label}'s share increases by {pct} percentage points, what would {y_label}'s new share need to be (in percent) to keep the total at 100%? Numeric, no % sign, in <answer>...</answer>.",
    "From the pie chart, if {x_label} rises by {pct} percentage points, what new percentage must {y_label} take to keep the total summing to 100%? Numeric (no %) in <answer>...</answer>.",
    "Suppose {x_label}'s slice in the pie chart grows by {pct} percentage points. Holding all other slices fixed, what does {y_label} need to be to keep the total at 100%? Integer percent (no % sign) in <answer>...</answer>.",
    "If {x_label}'s share increased by {pct} percentage points and only {y_label} compensated, what would {y_label}'s new share be (in percent)? Numeric (no %) in <answer>...</answer>.",
    "From the pie chart, {x_label} rises by {pct} pp. Compute {y_label}'s new share (in percent) so the chart still totals 100%. Numeric, no % sign, in <answer>...</answer>.",
]

_TEMPLATES_NEW_X = [
    "Look at the pie chart. If {x_label}'s share were to {direction} by {pct} percentage points, what would the new percentage of {x_label} be? Numeric, no % sign, in <answer>...</answer>.",
    "From the pie chart, if {x_label} {direction} by {pct} percentage points, what is the new share of {x_label}? Numeric (no %) in <answer>...</answer>.",
    "Suppose {x_label}'s slice {direction} by {pct} percentage points. What is the new value of {x_label} as a percent? Integer percent (no % sign) in <answer>...</answer>.",
    "If {x_label} {direction} by {pct} pp from its value in the pie chart, what is its new percentage? Numeric, no % sign, in <answer>...</answer>.",
    "Take the pie chart's value for {x_label}. If it {direction} by {pct} percentage points, what is the new value of {x_label} (in percent)? Numeric, no % sign, in <answer>...</answer>.",
    "From the pie, after {x_label} {direction} by {pct} percentage points, what would the new percentage of {x_label} be? Numeric (no %) in <answer>...</answer>.",
    "If {x_label}'s slice in the pie chart were to {direction} by {pct} pp, compute the new share of {x_label}. Numeric (no %) in <answer>...</answer>.",
]


_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_NAMED_LABELS = [
    ["Marketing", "Sales", "R&D", "Operations", "Support", "Finance"],
    ["Food", "Housing", "Transport", "Health", "Leisure", "Other"],
    ["Asia", "Europe", "Americas", "Africa", "Oceania", "Middle East"],
]


class PieChartRebalanceQA(StandaloneVisualEnv):
    ENV_NAME = "pie_chart_rebalance"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_slices": 4, "modes": ["rebalance_increase"],
                    "trivial_l0": True}
        if level <= 2:
            return {"n_slices": 4,
                    "modes": ["rebalance_increase", "rebalance_decrease",
                              "new_x"], "trivial_l0": False}
        if level <= 4:
            return {"n_slices": 5,
                    "modes": ["rebalance_increase", "rebalance_decrease",
                              "new_x"], "trivial_l0": False}
        if level <= 6:
            return {"n_slices": 6,
                    "modes": ["rebalance_increase", "rebalance_decrease",
                              "new_x"], "trivial_l0": False}
        if level <= 8:
            return {"n_slices": 7,
                    "modes": ["rebalance_increase", "rebalance_decrease",
                              "new_x"], "trivial_l0": False}
        return {"n_slices": 8,
                "modes": ["rebalance_increase", "rebalance_decrease",
                          "new_x"], "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1733 + level * 173 + 89)
        # Try multiple times — internal generation can fail on tight value
        # constraints (slices too small, no valid mode/y combination).
        for _try in range(15):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_slices"]

        if cfg["trivial_l0"]:
            # 4 equal slices [25, 25, 25, 25]. "If A drops 5 pp, B becomes?" -> 30.
            values = [25, 25, 25, 25]
            labels = _LABELS[:n]
            x_idx, y_idx = 0, 1
            pct = 5
            new_y = values[y_idx] + pct
            answer = str(int(new_y))
            tlist = _TEMPLATES_INCREASE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(x_label=labels[x_idx],
                                            y_label=labels[y_idx], pct=pct)
            img = self._render_pie(labels, values, rng)
            return question, answer, img

        # General: random slices summing to 100, in 5% increments
        snap = 5
        steps = 100 // snap
        for _attempt in range(20):
            cuts = sorted(rng.sample(range(1, steps), n - 1))
            chunks = []
            prev = 0
            for c in cuts:
                chunks.append((c - prev) * snap)
                prev = c
            chunks.append((steps - prev) * snap)
            if all(v >= 5 for v in chunks):
                break
        values = chunks
        if sum(values) != 100:
            values[0] += 100 - sum(values)
        if min(values) < 5:
            return None

        # Choose label naming style
        if rng.random() < 0.5:
            labels = _LABELS[:n]
        else:
            pool = rng.choice(_NAMED_LABELS)
            labels = pool[:n]

        mode = rng.choice(cfg["modes"])
        x_idx = rng.randrange(n)
        # pct points by which X is shifted; choose so resulting numbers stay in [5, 95]
        max_dec = max(0, values[x_idx] - 5)
        max_inc = max(0, 95 - values[x_idx])

        if mode == "rebalance_increase":
            # X decreases by pct; Y absorbs.
            if max_dec < 5:
                return None
            pct = rng.choice([p for p in [5, 10, 15] if p <= max_dec])
            y_pool = [i for i in range(n) if i != x_idx and values[i] + pct <= 95]
            if not y_pool:
                return None
            y_idx = rng.choice(y_pool)
            new_y = values[y_idx] + pct
            answer = str(int(new_y))
            tlist = _TEMPLATES_INCREASE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(x_label=labels[x_idx],
                                            y_label=labels[y_idx], pct=pct)
        elif mode == "rebalance_decrease":
            # X increases by pct; Y donates.
            if max_inc < 5:
                return None
            pct = rng.choice([p for p in [5, 10, 15] if p <= max_inc])
            y_pool = [i for i in range(n) if i != x_idx and values[i] - pct >= 5]
            if not y_pool:
                return None
            y_idx = rng.choice(y_pool)
            new_y = values[y_idx] - pct
            answer = str(int(new_y))
            tlist = _TEMPLATES_DECREASE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(x_label=labels[x_idx],
                                            y_label=labels[y_idx], pct=pct)
        else:  # new_x
            up = rng.choice([True, False])
            direction = "increased" if up else "decreased"
            if up:
                if max_inc < 5:
                    return None
                pct = rng.choice([p for p in [5, 10, 15, 20] if p <= max_inc])
                new_x = values[x_idx] + pct
            else:
                if max_dec < 5:
                    return None
                pct = rng.choice([p for p in [5, 10, 15, 20] if p <= max_dec])
                new_x = values[x_idx] - pct
            answer = str(int(new_x))
            tlist = _TEMPLATES_NEW_X
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(x_label=labels[x_idx],
                                            direction=direction, pct=pct)

        img = self._render_pie(labels, values, rng)
        return question, answer, img

    def _render_pie(self, labels: List[str], values: List[int],
                    rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        pal = list(palette)
        rng.shuffle(pal)
        colors = [pal[i % len(pal)] for i in range(len(values))]
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        startangle = rng.choice([0, 30, 45, 60, 90, 120])
        wedge_props = {"edgecolor": "#000", "linewidth": 1.2}
        if rng.random() < 0.3:
            wedge_props["width"] = rng.uniform(0.45, 0.65)
        ax.pie(values, labels=labels, colors=colors, autopct="%1.0f%%",
                startangle=startangle, counterclock=rng.choice([True, False]),
                textprops={"fontsize": 11, "fontweight": "bold"},
                wedgeprops=wedge_props)
        ax.set_title("Distribution (sums to 100%)", fontsize=11)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
