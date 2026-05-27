"""Error Bar Chart Visual QA Environment."""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CATEGORIES = [
    ["Method A", "Method B", "Method C", "Method D", "Method E"],
    ["Drug X", "Drug Y", "Drug Z", "Placebo"],
    ["Config 1", "Config 2", "Config 3", "Config 4", "Config 5"],
    ["Group Alpha", "Group Beta", "Group Gamma", "Group Delta"],
]
_Y_LABELS = ["Accuracy (%)", "Response Time (ms)", "Yield (%)",
             "Score", "Concentration (mg/L)", "Efficiency (%)"]

class ErrorBarChartQA(StandaloneVisualEnv):
    ENV_NAME = "error_bar_chart"

    def _level_config(self, level: int) -> Dict:
        # Redesigned: L0 starts with overlap + highest_upper_bound (old L4-5).
        # L9 requires significantly_higher + combined_uncertainty, no labels.
        if level <= 1:
            return {"qtypes": ["overlap", "highest_upper_bound"]}
        if level <= 3:
            return {"qtypes": ["significantly_higher", "overlap"]}
        if level <= 5:
            return {"qtypes": ["significantly_higher", "highest_upper_bound"]}
        if level <= 7:
            return {"qtypes": ["significantly_higher"]}
        return {"qtypes": ["significantly_higher", "overlap",
                           "confidence_width_ratio", "combined_uncertainty"],
                "hide_labels": True}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 710)
        np_rng = np.random.RandomState(seed * 100 + level)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        categories = list(sub_rng.choice(_CATEGORIES))
        n = len(categories)
        means = [round(float(np_rng.uniform(20, 90)), 1) for _ in range(n)]
        errors = [round(float(np_rng.uniform(2, 15)), 1) for _ in range(n)]

        question, answer = self._make_qa(rng, qtype, categories, means, errors)
        if question is None:
            return None

        style = self._random_style()
        y_label = rng.choice(_Y_LABELS)
        image = self._render(style, categories, means, errors, y_label,
                              hide_labels=cfg.get("hide_labels", False))
        return question, answer, image

    def _render(self, style, categories, means, errors, y_label, hide_labels=False):
        n = len(categories)
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(max(5.5, n * 1.1) * style["figsize_scale"],
                                        5 * style["figsize_scale"]))
        x = np.arange(n)
        colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors,
                      edgecolor="white", linewidth=0.6, error_kw={"linewidth": 1.5})
        if not hide_labels:
            for bar, m, e in zip(bars, means, errors):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 1,
                        f"{m}±{e}", ha="center", va="bottom",
                        fontsize=style["font_size_base"] - 2)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=style["font_size_base"],
                           rotation=25 if n > 4 else 0,
                           ha="right" if n > 4 else "center")
        ax.set_ylabel(y_label, fontsize=style["font_size_base"])
        ax.set_title("Measurements with Error Bars", fontsize=style["font_size_base"] + 2)
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _make_qa(self, rng, qtype, categories, means, errors):
        n = len(categories)

        if qtype == "largest_error":
            idx = errors.index(max(errors))
            return "Which bar has the largest error (uncertainty)?", categories[idx]

        elif qtype == "overlap":
            i, j = rng.sample(range(n), 2)
            lo_i, hi_i = means[i] - errors[i], means[i] + errors[i]
            lo_j, hi_j = means[j] - errors[j], means[j] + errors[j]
            overlap = lo_i <= hi_j and lo_j <= hi_i
            return (f"Do the error ranges of '{categories[i]}' and "
                    f"'{categories[j]}' overlap? Answer Yes or No."), "Yes" if overlap else "No"

        elif qtype == "significantly_higher":
            i, j = rng.sample(range(n), 2)
            # i is "significantly higher" if its lower bound > j's upper bound
            sig = (means[i] - errors[i]) > (means[j] + errors[j])
            return (f"Is '{categories[i]}' significantly higher than "
                    f"'{categories[j]}' (non-overlapping error bars)? "
                    f"Answer Yes or No."), "Yes" if sig else "No"

        elif qtype == "smallest_error":
            idx = errors.index(min(errors))
            return "Which bar has the smallest error?", categories[idx]

        elif qtype == "highest_upper_bound":
            uppers = [means[i] + errors[i] for i in range(n)]
            idx = uppers.index(max(uppers))
            return ("Which bar has the highest upper bound "
                    "(mean + error)?"), categories[idx]

        elif qtype == "mean_value":
            idx = rng.randint(0, n - 1)
            return f"What is the mean value of '{categories[idx]}'?", str(means[idx])

        elif qtype == "confidence_width_ratio":
            # Ratio of the widest error bar to the narrowest
            if min(errors) == 0:
                return None, None
            ratio = round(max(errors) / min(errors), 2)
            return ("What is the ratio of the widest error bar to the narrowest? "
                    "Round to 2 decimal places."), str(ratio)

        elif qtype == "combined_uncertainty":
            # Sum of all (mean * error) products
            total = round(sum(m * e for m, e in zip(means, errors)), 2)
            return ("What is the sum of (mean x error) for all categories? "
                    "Round to 2 decimal places."), str(total)

        else:
            idx = means.index(max(means))
            return "Which bar has the highest mean value?", categories[idx]
