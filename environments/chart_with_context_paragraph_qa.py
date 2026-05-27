"""
Chart with a context paragraph rendered above the chart.

The image consists of a 2-3 sentence framing paragraph at the top and a
bar chart below. The paragraph sets a story context (e.g. who the data
is about) but the asked numeric value comes from the chart. At higher
levels, the question requires combining a small fact stated in the
paragraph (e.g. "the survey was conducted in 2024") with a chart value.

L0: paragraph is pure framing; question is "what is the value of bar X?".
L>0: combine paragraph fact + chart value (e.g. "what value did the bar
labeled with the year mentioned in the paragraph have?").
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


_TEMPLATES_BAR_VALUE = [
    "Read the paragraph above the chart, then answer using the chart. What value does the bar labeled '{label}' show? Numeric answer.",
    "Take into account the framing paragraph and read the bar chart. Report the value of '{label}'. Numeric.",
    "After reading the paragraph, look at the chart. What is '{label}'? Numeric.",
    "The paragraph at the top frames the chart. From the chart, what value does '{label}' have? Numeric.",
    "Use the paragraph as context. Then read the bar chart and report '{label}'. Numeric.",
    "Read the context paragraph and the chart. What numeric value is shown for '{label}'? Numeric.",
    "After reading the paragraph above, find the bar labeled '{label}' in the chart and report its value. Numeric.",
    "Combine the paragraph context with the chart. From the bar chart, what is the value of '{label}'? Numeric.",
    "Paragraph plus chart: what value does '{label}' display in the bar chart? Numeric.",
    "Read the framing paragraph, then read the chart. Report the value of '{label}'. Numeric.",
    "From the chart shown (with the paragraph above), what is '{label}'? Numeric.",
    "Inspect the paragraph and the chart together. What is the value of '{label}' in the chart? Numeric.",
    "Looking at the chart with the paragraph as context, report the value of the '{label}' bar. Numeric.",
    "Following the framing paragraph, what value is shown for '{label}' in the bar chart? Numeric.",
    "Use the paragraph to set context and the chart for the data. Report '{label}'. Numeric.",
    "After reading the paragraph and inspecting the chart, what is the value of '{label}'? Numeric.",
    "Paragraph + chart joint reading. Report '{label}'. Numeric.",
    "From the bar chart (paragraph above for context), what value does '{label}' have? Numeric.",
]

_TEMPLATES_PARAGRAPH_LABEL = [
    "The paragraph above the chart names a specific category. From the bar chart, what value does that named category have? Numeric answer.",
    "Read the paragraph carefully. It mentions one of the chart categories. What is its value in the bar chart? Numeric.",
    "The paragraph highlights one category by name. Look at the bar chart and report that category's value. Numeric.",
    "Combine the paragraph and the chart: report the value of the category mentioned in the paragraph. Numeric.",
    "The paragraph names a single category. Find that category in the bar chart and report its value. Numeric.",
    "What value does the chart show for the category named in the paragraph? Numeric.",
]

_TEMPLATES_PARAGRAPH_RUNNERUP = [
    "The paragraph above names one category to ignore. From the remaining bars in the chart, find the one with the LARGEST value and report that value. Numeric.",
    "Ignore the category mentioned in the paragraph. Among the OTHER bars, what is the maximum value? Numeric.",
    "Reading the paragraph then the chart: ignore the named category and report the highest value among the remaining bars. Numeric.",
    "The paragraph singles out one category — exclude it. From the remaining bars, what is the largest value? Numeric.",
    "Drop the category named in the paragraph. Of the bars that remain, report the maximum value. Numeric.",
]


_BAR_LABEL_POOLS = [
    ["Region North", "Region South", "Region East", "Region West"],
    ["Plan Alpha", "Plan Beta", "Plan Gamma", "Plan Delta"],
    ["Group One", "Group Two", "Group Three", "Group Four"],
    ["Site A", "Site B", "Site C", "Site D"],
    ["Cohort I", "Cohort II", "Cohort III", "Cohort IV"],
    ["School X", "School Y", "School Z", "School W"],
]

_PARAGRAPH_TEMPLATES = [
    ("A nationwide survey conducted last year asked respondents about their "
     "weekly habits. Results were broken down by group, with each value "
     "representing the average count reported. The chart below summarizes "
     "the findings."),
    ("In a recent study of four communities, researchers tracked monthly "
     "activity levels across the regions. Each bar in the chart below "
     "corresponds to one community's measured value over the study window."),
    ("As part of an ongoing project, analysts collected aggregated data "
     "from each of the four locations. The resulting values, summarized "
     "in the chart below, provide a quick comparison across the groups."),
    ("During the most recent reporting period, four cohorts were measured "
     "on a common metric. The chart below visualizes the value for each "
     "cohort under identical measurement conditions."),
    ("A research team compiled the latest figures across four categories. "
     "The chart below presents the average reported value per category, "
     "ordered along the x-axis."),
    ("Across four sites, monthly totals were recorded throughout the "
     "quarter. Each bar in the chart below corresponds to one site's "
     "summary value for the period."),
]


class ChartWithContextParagraphQA(StandaloneVisualEnv):
    ENV_NAME = "chart_with_context_paragraph"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 v2: full-gradient redesign — was 100/100/100/100 across.
        # Bump every level: L0 still trivial mode but with 5 bars (was 4),
        # mid levels jump to harder modes, L9 stays at 4th-rank.
        level = max(0, min(level, 9))
        if level == 0:
            return {"mode": "bar_value", "n_bars": 5, "trivial_l0": True}
        if level <= 2:
            return {"mode": "bar_value", "n_bars": 6, "trivial_l0": False}
        if level <= 4:
            return {"mode": "paragraph_label", "n_bars": 7, "trivial_l0": False}
        if level <= 6:
            return {"mode": "paragraph_runnerup", "n_bars": 8, "trivial_l0": False,
                    "rank_target": 1}  # find max-of-remaining
        if level <= 8:
            return {"mode": "paragraph_runnerup", "n_bars": 9, "trivial_l0": False,
                    "rank_target": 2}  # find 2nd-of-remaining
        return {"mode": "paragraph_runnerup", "n_bars": 10, "trivial_l0": False,
                "rank_target": 4}  # find 4th-of-remaining

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1187 + level * 113 + 71)

        bar_pool = rng.choice(_BAR_LABEL_POOLS)
        # Repeat or extend pool when more than 4 bars are requested.
        if cfg["n_bars"] > len(bar_pool):
            extras = ["Group V", "Group VI", "Group VII", "Group VIII",
                      "Group IX", "Group X", "Group XI", "Group XII"]
            bar_labels = (bar_pool + extras)[:cfg["n_bars"]]
        else:
            bar_labels = bar_pool[:cfg["n_bars"]]
        paragraph = rng.choice(_PARAGRAPH_TEMPLATES)

        if cfg["trivial_l0"]:
            # Deterministic L0: 5 bars [10, 25, 40, 55, 70]; ask for the third
            # bar's label (constant across seeds).
            bar_values = [10, 25, 40, 55, 70][:cfg["n_bars"]]
            target_idx = min(2, cfg["n_bars"] - 1)
            target_value = bar_values[target_idx]
            target_label = bar_labels[target_idx]
            mode = "bar_value"
            paragraph_extra = ""
        else:
            mode = cfg["mode"]
            # Distinct values to keep questions unambiguous.
            bar_values = rng.sample(range(10, 95), cfg["n_bars"])
            if mode == "bar_value":
                target_idx = rng.randrange(cfg["n_bars"])
                target_label = bar_labels[target_idx]
                target_value = bar_values[target_idx]
                paragraph_extra = ""
            elif mode == "paragraph_runnerup":
                # 2026-05-04: support `rank_target` to harden L9 — exclude
                # cued bar then find the (rank_target-1)th-largest remaining.
                max_idx = bar_values.index(max(bar_values))
                target_label = bar_labels[max_idx]
                remaining = sorted(
                    [v for i, v in enumerate(bar_values) if i != max_idx],
                    reverse=True
                )
                rank = cfg.get("rank_target", 1)
                idx = min(rank - 1, len(remaining) - 1)
                target_value = remaining[idx]
                rank_word = {1: "highest", 2: "second-highest", 3: "third-highest",
                             4: "fourth-highest", 5: "fifth-highest"}.get(rank, f"{rank}th-highest")
                paragraph_extra = (
                    f" Note: for the question below, exclude the category "
                    f"'{target_label}' from consideration. Find the "
                    f"{rank_word} value among the remaining categories."
                )
            else:
                # paragraph_label: paragraph appends "Among the categories,
                # X stood out." where X is the answer's label.
                target_idx = rng.randrange(cfg["n_bars"])
                target_label = bar_labels[target_idx]
                target_value = bar_values[target_idx]
                paragraph_extra = (
                    f" Among the categories, {target_label} drew particular "
                    f"attention from the analysts."
                )

        sidx_bv = (self.seed or 0) % len(_TEMPLATES_BAR_VALUE)
        sidx_pl = (self.seed or 0) % len(_TEMPLATES_PARAGRAPH_LABEL)
        sidx_pr = (self.seed or 0) % len(_TEMPLATES_PARAGRAPH_RUNNERUP)
        if mode == "bar_value":
            question = _TEMPLATES_BAR_VALUE[sidx_bv].format(label=target_label)
        elif mode == "paragraph_runnerup":
            question = _TEMPLATES_PARAGRAPH_RUNNERUP[sidx_pr]
        else:
            question = _TEMPLATES_PARAGRAPH_LABEL[sidx_pl]

        answer = str(int(target_value))
        full_paragraph = paragraph + paragraph_extra
        img = self._render(full_paragraph, bar_labels, bar_values, rng)
        return question, answer, img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, paragraph: str, bar_labels: List[str],
                bar_values: List[int], rng: random.Random) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        n = len(bar_labels)
        # Layout: paragraph on top (text axis), chart below (bar axis).
        # Use gridspec with a small text panel and a bigger chart panel.
        fig = plt.figure(figsize=(max(8.0, 0.9 * n + 4.0), 5.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 3.0], hspace=0.35)
        ax_text = fig.add_subplot(gs[0])
        ax_chart = fig.add_subplot(gs[1])

        # Render paragraph as wrapped text. Use ax.text() with a dedicated
        # axis whose own visuals are turned off.
        ax_text.set_facecolor("#fdf9f1")
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.axis("off")
        # Box around the paragraph for visual separation.
        for spine_name in ("left", "right", "top", "bottom"):
            ax_text.spines[spine_name].set_visible(False)
        ax_text.add_patch(plt.Rectangle((0.005, 0.05), 0.99, 0.9,
                                          facecolor="#fdf9f1",
                                          edgecolor="#888888",
                                          linewidth=1.0,
                                          transform=ax_text.transAxes))
        ax_text.text(
            0.02, 0.5, paragraph,
            transform=ax_text.transAxes,
            fontsize=11, color="#111",
            verticalalignment="center",
            horizontalalignment="left",
            wrap=True,
        )

        # Bar chart
        ax_chart.set_facecolor("#ffffff")
        bar_colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax_chart.bar(range(n), bar_values, color=bar_colors,
                              edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(bar_values)
        for b, v in zip(bars, bar_values):
            ax_chart.text(b.get_x() + b.get_width() / 2,
                           v + max_v * 0.02, str(int(v)),
                           ha="center", va="bottom",
                           fontsize=10, color="#111", fontweight="bold")
        ax_chart.set_xticks(range(n))
        ax_chart.set_xticklabels(bar_labels,
                                  rotation=20 if n > 4 else 0,
                                  ha="right" if n > 4 else "center",
                                  fontsize=10)
        ax_chart.set_ylabel("Value", fontsize=10)
        ax_chart.set_ylim(0, max_v * 1.22 + 4)
        ax_chart.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax_chart.spines["top"].set_visible(False)
        ax_chart.spines["right"].set_visible(False)

        # Manual margins because the text axis is incompatible with
        # tight_layout.
        fig.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.10)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
