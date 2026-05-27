"""
Cross-subplot reasoning over shared categories.

The image contains 2-3 side-by-side subplots labeled (a), (b), (c) with
the same category set on the x-axis but different value series. The task:
locate a target value Y in subplot B (e.g. the maximum), then find the
category in subplot A whose value is closest to Y. Output: category label.
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


_TEMPLATES = [
    "Subplot (a) and subplot (b) share the same set of categories on the x-axis but show different value series. Find the maximum value in subplot (b), then identify the category in subplot (a) whose value is closest to that. Place the category label in <answer>...</answer>.",
    "Look at the two subplots. In subplot (b), find the largest bar. Then, in subplot (a), report the category label whose bar height is closest to that maximum value. Place the label in <answer>...</answer>.",
    "Across the two subplots, the categories on the x-axis are shared. Take the maximum value from subplot (b) and find which category in subplot (a) has the closest value. Output the category label in <answer>...</answer>.",
    "Read subplot (b) to find its maximum value. Then locate the category in subplot (a) whose value most closely matches that. Place the matching label in <answer>...</answer>.",
    "Subplot (a) and (b) use shared category labels. Identify the maximum bar in subplot (b), then the closest-valued bar in subplot (a). Output that category's label in <answer>...</answer>.",
    "Step 1: in subplot (b), find the maximum value. Step 2: in subplot (a), find the category whose value is closest to it. Report the category label in <answer>...</answer>.",
    "From the two subplots, take the largest value in subplot (b) and find the closest-matching category in subplot (a). Place the category name in <answer>...</answer>.",
    "Cross-reference subplot (b)'s max with subplot (a). Which category in subplot (a) has a value closest to subplot (b)'s maximum? Place its label in <answer>...</answer>.",
    "Shared-category subplots are shown. Pick subplot (b)'s maximum value, then find the closest match in subplot (a). Output the matching label in <answer>...</answer>.",
    "Find the maximum in subplot (b). Then, in subplot (a), report the category whose value is closest to that. Category label in <answer>...</answer>.",
    "Both subplots share the same x-axis categories. Take the largest value in (b); find the category in (a) whose value is nearest. Label in <answer>...</answer>.",
    "Look across the two subplots. The maximum value in (b) most closely matches which category's value in (a)? Place the category label in <answer>...</answer>.",
    "Cross-plot question: which category in (a) has the value closest to the maximum value in (b)? Place the label in <answer>...</answer>.",
    "Identify the highest value in subplot (b), then locate which category in subplot (a) has the closest value. Place its label in <answer>...</answer>.",
    "From subplot (b), grab the max. Then find which category in subplot (a) has the nearest value. Output the label in <answer>...</answer>.",
    "Subplot (a) shows one series and subplot (b) shows another over shared categories. Find subplot (b)'s maximum and match it to the closest-valued category in (a). Place the label in <answer>...</answer>.",
    "Compare the subplots. Which category in (a) is closest in value to the largest entry in (b)? Place its name in <answer>...</answer>.",
    "Read both subplots side by side. Find the max of (b), then the closest in (a). Category label in <answer>...</answer>.",
]


# Use distinct, multi-character single-token labels so the base
# `_check_answer` substring fallback (gt-in-pred for len>=3) cannot
# false-positive across overlapping names.
_CATEGORY_POOLS = [
    ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"],
    ["Walnut", "Pecan", "Cashew", "Almond", "Pistachio", "Hazelnut", "Brazil"],
    ["Aurora", "Borealis", "Comet", "Eclipse", "Galaxy", "Nebula", "Quasar"],
    ["Maple", "Cedar", "Birch", "Spruce", "Willow", "Poplar", "Aspen"],
    ["Raptor", "Falcon", "Hornet", "Viper", "Eagle", "Tiger", "Mantis"],
    ["Lyra", "Vega", "Sirius", "Polaris", "Rigel", "Antares", "Capella"],
]

_Y_LABELS = ["Score", "Revenue", "Units", "Visitors", "Hours", "Sales", "Count"]


class SubplotCrossReasoningQA(StandaloneVisualEnv):
    ENV_NAME = "subplot_cross_reasoning"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100.
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_subplots": 2, "n_cat": 4, "trivial_l0": True}  # was n_cat=3
        if level <= 2:
            return {"n_subplots": 3, "n_cat": 5, "trivial_l0": False}  # bumped
        if level <= 4:
            return {"n_subplots": 3, "n_cat": 6, "trivial_l0": False}  # bumped
        if level <= 6:
            return {"n_subplots": 4, "n_cat": 6, "trivial_l0": False}  # bumped
        if level <= 8:
            return {"n_subplots": 4, "n_cat": 7, "trivial_l0": False,
                    "tight_margin": True}
        # 2026-05-04: bumped L9 difficulty — n_subplots 4→5 (more panels to scan).
        return {"n_subplots": 5, "n_cat": 7, "trivial_l0": False,
                "tight_margin": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1129 + level * 73 + 53)

        n_subplots = cfg["n_subplots"]
        n_cat = cfg["n_cat"]

        cats_pool = rng.choice(_CATEGORY_POOLS)
        cats = rng.sample(cats_pool, n_cat)
        y_label = rng.choice(_Y_LABELS)

        if cfg["trivial_l0"]:
            # 2 simple bar subplots, 4 categories (was 3). subplot (b)
            # max is closest to ONE category in subplot (a).
            # subplot (a): [10, 50, 30, 20]
            # subplot (b): [5, 12, 47, 18] -> max=47, closest in (a) = 50 = idx 1
            series = [[10, 50, 30, 20][:n_cat],
                      [5, 12, 47, 18][:n_cat]]
            target_subplot = 1  # 'b'
            search_subplot = 0  # 'a'
            target_value = max(series[target_subplot])
        else:
            # Random series with controlled separation in the search subplot
            # to keep the closest-match unique.
            for _attempt in range(40):
                series = []
                for s in range(n_subplots):
                    vals = [rng.randint(10, 99) for _ in range(n_cat)]
                    series.append(vals)
                # Ensure distinct values within search subplot (idx 0)
                if len(set(series[0])) < n_cat:
                    continue
                # Ensure target subplot has a unique max to make "the maximum"
                # well-defined.
                if series[1].count(max(series[1])) > 1:
                    continue
                target_value = max(series[1])
                # Distance from each (a) bar to target
                dists = [abs(v - target_value) for v in series[0]]
                # Need a unique closest with margin. L9 uses tighter margin
                # (=1) so the model must read bar heights precisely; lower
                # levels keep margin >=3 for unambiguous matches.
                sd = sorted(dists)
                min_margin = 1 if cfg.get("tight_margin") else 3
                if sd[1] - sd[0] < min_margin:
                    continue
                target_subplot = 1
                search_subplot = 0
                break
            else:
                # Could not find a clean instance; fall back to L0-style
                # values.
                series = [[10, 50, 30] + [20] * (n_cat - 3),
                          [5, 12, 47] + [8] * (n_cat - 3)]
                series[0] = series[0][:n_cat]
                series[1] = series[1][:n_cat]
                target_subplot = 1
                search_subplot = 0
                target_value = max(series[target_subplot])

        # Compute closest category
        dists = [abs(v - target_value) for v in series[search_subplot]]
        closest_idx = dists.index(min(dists))
        answer = cats[closest_idx]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]

        img = self._render(cats, series, y_label, rng)
        return question, answer, img

    def _render(self, cats: List[str], series: List[List[int]],
                y_label: str, rng: random.Random) -> Image.Image:
        n_subplots = len(series)
        palette = rng.choice(self._COLOR_PALETTES)
        fig_w = max(4.5 * n_subplots, 9.0)
        fig, axes = plt.subplots(1, n_subplots, figsize=(fig_w, 4.4),
                                  dpi=120)
        fig.patch.set_facecolor("#ffffff")
        if n_subplots == 1:
            axes = [axes]
        for s, vals in enumerate(series):
            ax = axes[s]
            ax.set_facecolor("#ffffff")
            colors = [palette[(s * 2 + i) % len(palette)] for i in range(len(cats))]
            bars = ax.bar(range(len(cats)), vals, color=colors,
                          edgecolor="#1a1a1a", linewidth=0.6)
            max_v = max(vals)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + max_v * 0.02, str(int(v)),
                        ha="center", va="bottom",
                        fontsize=10, color="#111", fontweight="bold")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=20 if len(cats) > 4 else 0,
                                ha="right" if len(cats) > 4 else "center",
                                fontsize=9)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_ylim(0, max_v * 1.22 + 4)
            ax.set_title(f"({chr(ord('a') + s)})", fontsize=12, loc="left")
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
        # Case-insensitive label match. Handle "the answer is X" forms by
        # tokenizing and looking for the gt label as a whole word.
        import re as _re
        p = predicted.strip().lower().rstrip(".").strip()
        g = ground_truth.strip().lower()
        if p == g:
            return True
        # Whole-word match in the prediction.
        pattern = r"\b" + _re.escape(g) + r"\b"
        if _re.search(pattern, p):
            return True
        return False
