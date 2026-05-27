"""
Chart Conversational Argmax-Coreference QA — C14 (reference Conversational).

Multi-turn dialogue: a previous Q1/A1 pair is provided in the prompt, then a
follow-up Q2 with coreference (e.g. "what about for X?") swapping the
concept being argmax'd. Final answer is graded.

Verbatim sample style (design notes Conversational):

    Q-IDX 1213:
      Q[0]: "which country's policymakers overguesses the most on average?"
            → "Kenya"
      Q[1] (graded): "what about for underguessing?" → "Indonesia"

    Q-IDX 1184:
      Q[0]: "what's the largest gap between actual and reasonable wait
             times?" → "9.4 weeks"
      Q[1] (graded): "how about the smallest gap value then?" → "0.7 weeks"

Format: prompt zips prior Q/A pairs as ``Conversation: [Q1: ..., A1: ...]
— Q2: ?``. Final answer is the category label (a country / city / item
name).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_GROUP_LABELS = [
    # Pools of *category* labels — argmax-on-category questions like "which X
    # has the most Y?" must answer with one of these labels. We avoid
    # time-period pools (Q1 2022 etc) here because the natural verb in the
    # benchmark coreference question ("which X has the most Y?" / "what about
    # for Z?") expects a noun-style category, not a quarter.
    ["USA", "China", "Germany", "Japan", "India", "Brazil",
     "Kenya", "Indonesia", "Mexico", "Egypt"],
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes",
     "Peaches", "Pears", "Cherries"],
    ["Sales", "Engineering", "Marketing", "Finance", "HR",
     "Legal", "Support", "R&D"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Dallas",
     "Boston", "Seattle"],
]

_METRIC_PAIRS = [
    # (metric_a, metric_b) — pairs of two truly-parallel measurements that
    # naturally co-exist on the same chart as two separate value series. Q1
    # asks for argmax of metric_a, Q2 (final) asks for argmax of metric_b
    # ("what about for X?" coreference, IDX 1213 style).
    ("overguessing", "underguessing"),
    ("imports", "exports"),
    ("revenue", "expenses"),
    ("males", "females"),
    ("urban residents", "rural residents"),
    ("full-time staff", "part-time staff"),
    ("incoming calls", "outgoing calls"),
    ("morning sales", "evening sales"),
]


class ChartConversationalArgmaxQA(StandaloneVisualEnv):
    ENV_NAME = "chart_conversational_argmax"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"n_groups": 4}
        if level <= 4:
            return {"n_groups": 5}
        if level <= 7:
            return {"n_groups": 6}
        return {"n_groups": 8}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1373 + level * 89 + 17)

        for _ in range(15):
            res = self._try(rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, cfg):
        n = cfg["n_groups"]
        labels_pool = list(rng.choice(_GROUP_LABELS))
        rng.shuffle(labels_pool)
        if len(labels_pool) < n:
            return None
        labels = labels_pool[:n]

        # Two metric series — antonym pairs (so coreference Q makes sense).
        metric_a, metric_b = rng.choice(_METRIC_PAIRS)

        # Generate two value series; require argmax(a) != argmax(b).
        vals_a = [rng.randint(20, 200) for _ in range(n)]
        vals_b = [rng.randint(20, 200) for _ in range(n)]
        # Ensure unique max for both and they are different categories.
        while vals_a.count(max(vals_a)) > 1:
            vals_a[vals_a.index(max(vals_a))] += rng.randint(1, 5)
        while vals_b.count(max(vals_b)) > 1:
            vals_b[vals_b.index(max(vals_b))] += rng.randint(1, 5)
        argmax_a = labels[vals_a.index(max(vals_a))]
        argmax_b = labels[vals_b.index(max(vals_b))]
        if argmax_a == argmax_b:
            return None

        # Build the conversational prompt mirroring Q-IDX 1213 / 1184 style:
        # ``Conversation: [Q1: ... A1: ...]\n\nQ2: ?`` — Q2 uses pure
        # coreference ("what about for X", "how about for X") with no
        # standalone verb, forcing the model to thread Q1's argmax frame.
        q1 = f"Which group has the most {metric_a}?"
        a1 = argmax_a
        coref_templates = [
            f"What about for {metric_b}?",
            f"How about for {metric_b}?",
            f"And the most {metric_b}?",
        ]
        q2 = rng.choice(coref_templates)

        prompt = (
            f"Conversation: [Q1: {q1} A1: {a1}]\n\n"
            f"Q2: {q2}"
        )

        image = self._render(rng, labels, vals_a, vals_b,
                              metric_a, metric_b)
        return prompt, argmax_b, image

    def _render(self, rng, labels, vals_a, vals_b, metric_a, metric_b):
        style = self._random_style()
        palette = list(style["palette"])
        n = len(labels)
        fig_w = max(7, n * 0.95 + 2) * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, 4.6 * style["figsize_scale"]))
        x = list(range(n))
        bw = 0.4
        ax.bar([xi - bw / 2 for xi in x], vals_a, bw,
               color=palette[0], label=metric_a, edgecolor="white")
        ax.bar([xi + bw / 2 for xi in x], vals_b, bw,
               color=palette[2 % len(palette)], label=metric_b,
               edgecolor="white")
        for xi, v in zip(x, vals_a):
            ax.text(xi - bw / 2, v + max(vals_a + vals_b) * 0.02,
                    str(v), ha="center", va="bottom",
                    fontsize=style["font_size_base"] - 2)
        for xi, v in zip(x, vals_b):
            ax.text(xi + bw / 2, v + max(vals_a + vals_b) * 0.02,
                    str(v), ha="center", va="bottom",
                    fontsize=style["font_size_base"] - 2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,
                           rotation=25 if max(len(l) for l in labels) > 5 else 0,
                           ha="right" if max(len(l) for l in labels) > 5 else "center",
                           fontsize=style["font_size_base"])
        ax.set_ylabel("Value", fontsize=style["font_size_base"])
        ax.set_title(f"{metric_a} vs {metric_b}",
                     fontsize=style["font_size_base"] + 2, pad=10)
        ax.legend(fontsize=style["font_size_base"] - 1,
                  loc=style["legend_loc"])
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
