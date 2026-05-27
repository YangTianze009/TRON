"""Two bar charts with one value changed."""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class BeforeAfterQA(StandaloneVisualEnv):
    ENV_NAME = "before_after"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # More categories + smaller changes = harder to spot
        return {
            "n_cats": min(3 + level // 2, 7),     # 3..7
            "change_range": max(2, 12 - level),     # 12..2
            "n_changes": 1 + level // 5,            # 1 or 2 changes at L5+
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        style = self._random_style()
        palette = style["palette"]
        # Two bar charts side by side, values changed
        all_cats = ["A", "B", "C", "D", "E", "F", "G"]
        cats = all_cats[:cfg["n_cats"]]
        vals1 = [rng.randint(5, 30) for _ in cats]
        n_ch = min(cfg["n_changes"], len(cats))
        change_idxs = rng.sample(range(len(cats)), n_ch)
        vals2 = vals1[:]
        for ci in change_idxs:
            change_amount = rng.choice([-1, 1]) * rng.randint(1, cfg["change_range"])
            vals2[ci] += change_amount
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor(style["bg_color"])
        ax1.bar(cats, vals1, color=palette[0], alpha=0.8)
        ax1.set_title("Before", fontsize=12, fontweight="bold")
        ax2.bar(cats, vals2, color=palette[1], alpha=0.8)
        ax2.set_title("After", fontsize=12, fontweight="bold")
        for ax in [ax1, ax2]:
            self._apply_style(fig, ax, style)
        fig.suptitle("Spot the Change", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        sidx = (self.seed or 0) % 16
        if n_ch == 1:
            single_templates = [
                "Which category changed between Before and After?",
                "Identify the category whose value differs from Before to After.",
                "Compare the Before and After bar charts: which category's value changed?",
                "Between the two charts (Before / After), which category was modified?",
                "Find the single category that changed between Before and After.",
                "Which bar is different in height between Before and After?",
                "Looking at Before vs After, which category shows a changed value?",
                "The Before and After charts differ in exactly one category. Which one?",
                "Which category's value is not the same in the Before and After charts?",
                "Which category's bar height changed from Before to After?",
                "Spot the category that was altered between the Before and After charts.",
                "From Before to After, one category's value changed. Which category is it?",
                "Which of the categories has a different bar height in After compared to Before?",
                "Pinpoint the category whose value differs between the Before and After bar charts.",
                "Which category did NOT stay the same between Before and After?",
                "Identify the single category whose bar changed height from Before to After.",
            ]
            question = single_templates[sidx]
            answer = cats[change_idxs[0]]
        else:
            multi_templates = [
                "Which categories changed between Before and After? List them separated by commas.",
                "Identify all categories whose values differ between Before and After; separate with commas.",
                "Compare Before and After: list the categories that changed, comma-separated.",
                "Which bars have different heights between Before and After? List them (comma-separated).",
                "Find all modified categories between Before and After. Respond with a comma-separated list.",
                "From Before to After, list the categories whose values changed (comma-separated).",
                "Which categories show different values between the two charts? Give a comma-separated list.",
                "List every category that was altered between Before and After, separated by commas.",
                "Identify the categories whose bar heights changed from Before to After. Comma-separated list.",
                "Which categories differ in value between Before and After? List them, separated by commas.",
                "Spot every category that changed between the Before and After charts. Comma-separated.",
                "Between Before and After, multiple categories changed. Name them (comma-separated).",
                "Which categories in the After chart differ from the Before chart? List them, comma-separated.",
                "Give a comma-separated list of categories whose values changed between Before and After.",
                "Identify all modified categories between Before and After, separating names with commas.",
                "Report the categories that changed value from Before to After, separated by commas.",
            ]
            question = multi_templates[sidx]
            answer = ", ".join(sorted(cats[ci] for ci in change_idxs))
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
