"""
Chart Conversational Year-Exact QA — C30 (reference Conversational with
year-flag final answer requiring strict 4-digit-year exact match).

Multi-turn dialogue where the final answer is a year value; year-format
answers require strict 4-digit-year exact match (no 5 % numeric tolerance) —
the model must emit the exact year string (e.g. "1993").

Verbatim sample style (design notes Conversational):

    Q-IDX 1137 (year-YES, 2-turn):
      Q[0]: "in which year did the total number of tubewells first exceed
             400,000?" → "1993" (year:YES)
      Q[1] (graded): "which year did it double that value?" → "2003"
                     (year:YES, exact match)

    Q-IDX 1079 (5-turn, year-YES on Q[2]/Q[3]):
      Q[2]: "which year has the highest number of sectors falling?" → "2009"
      Q[3]: "when do things start to look better and the number of sectors
             failing decreases?" → "2010"

We override `_check_answer` to require strict 4-digit-year exact match (no
±0.5 tolerance), matching the year:YES grader behaviour.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SERIES_NAMES = [
    "tubewells", "subscribers", "patients", "production",
    "registrations", "applicants", "sales (units)",
    "exports", "infections", "births",
]


class ChartConversationalYearQA(StandaloneVisualEnv):
    ENV_NAME = "chart_conversational_year"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"n_years": 8}
        if level <= 4:
            return {"n_years": 12}
        if level <= 7:
            return {"n_years": 16}
        return {"n_years": 22}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1499 + level * 97 + 31)

        for _ in range(15):
            res = self._try(rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, cfg):
        n = cfg["n_years"]
        start_year = rng.randint(1980, 2025 - n)
        years = [start_year + i for i in range(n)]
        # Generate roughly monotone-increasing series with random walk.
        series_name = rng.choice(_SERIES_NAMES)
        base = rng.randint(50, 300)
        step = rng.randint(40, 120)
        vals = [base]
        for _ in range(n - 1):
            vals.append(vals[-1] + step + rng.randint(-step // 3, step // 2))
        vals = [max(0, v) for v in vals]

        # Pick a threshold T for Q1 (year-YES): first year that vals exceed T.
        # Choose T so it's crossed somewhere in the middle.
        mid_v = vals[n // 2]
        low_T = int(mid_v * 0.6)
        T1 = low_T + rng.randint(-mid_v // 10, mid_v // 10)
        # Find first crossing
        cross_idx = None
        for i, v in enumerate(vals):
            if v > T1:
                cross_idx = i
                break
        if cross_idx is None or cross_idx == 0:
            return None

        # Q2: year that doubles that value (final, graded).
        target_val = vals[cross_idx] * 2
        double_idx = None
        for i in range(cross_idx + 1, n):
            if vals[i] >= target_val:
                double_idx = i
                break
        if double_idx is None:
            return None

        year_q1 = years[cross_idx]
        year_q2 = years[double_idx]

        # Mirror Q-IDX 1137 verbatim style:
        #   Q[0]: "in which year did the total number of tubewells first
        #          exceed 400,000?" → "1993"
        #   Q[1] (graded): "which year did it double that value?" → "2003"
        # We supply the explicit threshold (`exceed {target_val}`) inline so
        # the question is unambiguous from a static chart read; benchmark
        # relies on context but we cannot. This still preserves the "double
        # that value" coreference to A1.
        prompt = (
            f"Conversation: [Q1: In which year did the total number of "
            f"{series_name} first exceed {T1}? A1: {year_q1}]\n\n"
            f"Q2: Which year did it first double that value "
            f"(i.e. first exceed {target_val})?"
        )

        image = self._render(rng, years, vals, series_name)
        # Year-exact-match is enforced unconditionally for this env via the
        # `_check_answer` override below (any 4-digit-year ground truth ⇒
        # strict regex match, no numeric tolerance).
        return prompt, str(year_q2), image

    # Year-flag YES: strict 4-digit exact match (no ±0.5 numeric tolerance).
    # We override `_check_answer` so e.g. "2003" passes and "2002" fails.
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import re
        predicted = predicted.strip().lower().rstrip(".")
        ground_truth = ground_truth.strip().lower().rstrip(".")
        # If both are 4-digit years, require exact match.
        if re.match(r"^\d{4}$", ground_truth):
            # Pull first 4-digit year from prediction (handles model
            # outputs like "the year 1993" / "1993." / "**1993**").
            m = re.search(r"\b(\d{4})\b", predicted)
            if m:
                return m.group(1) == ground_truth
            return False
        return super()._check_answer(predicted, ground_truth)

    def _render(self, rng, years, vals, series_name):
        style = self._random_style()
        palette = list(style["palette"])
        n = len(years)
        fig_w = max(7, n * 0.45 + 2) * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, 4.6 * style["figsize_scale"]))
        ax.plot(range(n), vals, marker="o", color=palette[0],
                linewidth=style["line_width"], markersize=5,
                label=series_name)
        # Tick every other year if many points
        tick_step = 1 if n <= 12 else 2
        ax.set_xticks(range(0, n, tick_step))
        ax.set_xticklabels([str(years[i]) for i in range(0, n, tick_step)],
                           rotation=45, ha="right",
                           fontsize=style["font_size_base"] - 1)
        ax.set_ylabel("Value", fontsize=style["font_size_base"])
        ax.set_title(f"Total {series_name} over time",
                     fontsize=style["font_size_base"] + 2, pad=10)
        ax.legend(fontsize=style["font_size_base"] - 1)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
