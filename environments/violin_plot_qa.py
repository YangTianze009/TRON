"""
Violin Plot QA — X30 (reference reasoning_val).

Renders an N=2-5 group violin plot (matplotlib `violinplot`) and asks
median-comparison / trend-as-X-increases / Q1 vs Q2 type questions. Output
is a group label, position word, or trend word.

Verbatim sample style (design notes §X30):

    Q-IDX 230 (Violin, econ):
      "Compare the median firm-product exports of Quartile 1 to Quartile 2
       under TPV Yes condition in Ecuador. Which is higher?"
       GT `Quartile 1`

    Q-IDX 760 (Violin, q-bio):
      "As the SILVR rate increases in subplots b, are the median values
       increasing or decreasing?" GT `Decreasing`

    Q-IDX 722 (Violin, stat):
      "Compared to the Staf-Gate Generated FC, does the Empirical have a
       higher or lower median value?" GT `higher`

reference judge expects bare label / position / comparator word.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_GROUP_POOLS = [
    ["Quartile 1", "Quartile 2", "Quartile 3", "Quartile 4"],
    ["Group A", "Group B", "Group C", "Group D", "Group E"],
    ["Control", "Treatment 1", "Treatment 2", "Treatment 3"],
    ["Region X", "Region Y", "Region Z", "Region W"],
    ["Site 1", "Site 2", "Site 3", "Site 4"],
]

_Y_LABELS = ["exports", "score", "concentration", "rate",
             "response time", "yield"]


class ViolinPlotQA(StandaloneVisualEnv):
    ENV_NAME = "violin_plot"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_groups": 2,
                    "qtypes": ["median_compare"]}
        if level <= 5:
            return {"n_groups": 3,
                    "qtypes": ["median_compare", "median_trend"]}
        return {"n_groups": 4,
                "qtypes": ["median_compare", "median_trend",
                           "highest_median"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1823 + level * 163 + 73)
        np_rng = np.random.RandomState(rng.randint(0, 1_000_000))

        for _ in range(15):
            res = self._try(rng, np_rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, np_rng, cfg):
        groups_pool = list(rng.choice(_GROUP_POOLS))
        rng.shuffle(groups_pool)
        n = cfg["n_groups"]
        if len(groups_pool) < n:
            return None
        groups = groups_pool[:n]

        # Pick a single y-axis quantity now so the chart label and the
        # question phrasing AGREE (CharXiv X30 samples like "Compare the
        # median firm-product exports..." imply the chart actually labels
        # the y-axis "exports").
        y_label = rng.choice(_Y_LABELS)

        qtype = rng.choice(cfg["qtypes"])

        # Generate per-group samples — pick centers spaced enough so the
        # visual median ordering is unambiguous.
        if qtype == "median_trend":
            # Force monotonic trend
            direction = rng.choice(["Increasing", "Decreasing"])
            base = rng.uniform(20, 60)
            step = rng.uniform(8, 18)
            if direction == "Decreasing":
                step = -step
            centers = [base + i * step for i in range(n)]
        else:
            # Random distinct centers
            centers = [rng.uniform(20, 80) for _ in range(n)]
            # Ensure clear separation of medians (>5 apart for top two).
            sorted_c = sorted(centers, reverse=True)
            if sorted_c[0] - sorted_c[1] < 6:
                return None

        # Collect samples for each group.
        data = []
        for c in centers:
            spread = rng.uniform(2.5, 5.5)
            samples = np_rng.normal(c, spread, 60)
            data.append(samples)

        medians = [float(np.median(d)) for d in data]

        if qtype == "median_compare":
            # Compare two random groups
            i, j = rng.sample(range(n), 2)
            higher = groups[i] if medians[i] > medians[j] else groups[j]
            stem = (f"Compare the median {y_label} of "
                    f"{groups[i]} to {groups[j]}. Which is higher?")
            answer = higher
            prompt = (f"{stem} Answer with the group's full name as shown "
                      f"on the chart.")
        elif qtype == "median_trend":
            stem = (f"As you move from {groups[0]} to {groups[-1]} in the "
                    f"violin plot, are the median {y_label} values "
                    f"increasing or decreasing?")
            answer = direction
            prompt = (f"{stem} Answer with one word: Increasing or "
                      f"Decreasing.")
        elif qtype == "highest_median":
            i = medians.index(max(medians))
            stem = f"Which group has the highest median {y_label}?"
            answer = groups[i]
            prompt = (f"{stem} Answer with the group's full name as shown "
                      f"on the chart.")
        else:
            return None

        image = self._render(groups, data, y_label)
        return prompt, answer, image

    def _render(self, groups, data, y_label):
        style = self._random_style()
        n = len(groups)
        fig, ax = plt.subplots(figsize=(max(6, n * 1.5) * style["figsize_scale"],
                                        4.6 * style["figsize_scale"]))
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        palette = style["palette"]
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(palette[i % len(palette)])
            body.set_alpha(0.7)
            body.set_edgecolor("black")
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)
        ax.set_xticks(list(range(1, n + 1)))
        ax.set_xticklabels(groups, fontsize=style["font_size_base"],
                           rotation=15)
        # y-axis label matches the question's noun (CharXiv-style: y axes
        # carry the actual variable name, not a generic "Value"). Capitalize
        # for chart aesthetics.
        ax.set_ylabel(y_label.capitalize(),
                      fontsize=style["font_size_base"])
        ax.set_title(f"Distribution of {y_label} by group",
                     fontsize=style["font_size_base"] + 1, pad=10)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
