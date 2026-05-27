"""
Universal/existential claim verification on a time-series line chart.

Pose claims with quantifier scope:
  - "X never exceeded T."        — universal negation
  - "X exceeded T at every period."  — universal affirmative
  - "X dropped below T at least once."  — existential
  - "X stayed above T for every {period_unit}."  — universal

Mirrors reference Fact-Checking IDX 1567 ("the chart shows that neither trump
nor biden's polling average ever exceeded 46%."), IDX 1491 ("5g smartphones
will consistently show an increasing trend"), and IDX 1589 ("number of deaths
per week exceeded 10 on only one occasion").

Output: lowercase `true` / `false`.
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


_TEMPLATES_PROMPTS = [
    "Look at the chart and decide whether the following claim is true or false. Answer with the single lowercase word `true` or `false`.\n\nClaim: {claim}",
    "Verify the claim against the chart. Reply with `true` or `false` (lowercase).\n\n{claim}",
    "Is the following claim correct based on the chart shown? Reply with one word: `true` or `false`. Place answer.\n\nClaim: {claim}",
    "Read the chart and judge: `true` or `false`? Reply with the lowercase word.\n\nStatement: {claim}",
    "Decide if the claim is supported by the chart. Single-word answer in lowercase: `true` or `false`. Place.\n\nClaim: {claim}",
    "Fact-check the claim against the chart. Reply with lowercase `true` or `false`.\n\n{claim}",
    "Based on the chart, is this true or false? Lowercase answer.\n\nClaim: {claim}",
    "Evaluate the truth value of the following statement using the chart. Reply with `true` or `false`.\n\nStatement: {claim}",
    "Verify by inspecting the chart. Lowercase reply (`true` or `false`).\n\nClaim: {claim}",
    "True or false based on the chart? Lowercase one-word reply.\n\n{claim}",
    "Determine whether the following is true or false from the chart shown. Single lowercase word.\n\nClaim: {claim}",
    "Use the chart to assess the claim. Reply `true` or `false` (lowercase). Place.\n\nClaim: {claim}",
    "Examine the chart carefully and state whether the claim holds. Reply `true` or `false` (lowercase).\n\nClaim: {claim}",
    "From the chart, is the following statement true or false? Single lowercase word.\n\nClaim: {claim}",
    "Verify the chart-based claim. Reply with `true` or `false` (lowercase). Answer.\n\nClaim: {claim}",
    "Judge: true or false? Use the chart. Lowercase one-word answer.\n\nClaim: {claim}",
]


def _month_labels(n):
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:n]

def _quarter_labels(n):
    return [f"Q{i+1}" for i in range(n)]

def _year_labels(n, start=2010):
    return [str(start + i) for i in range(n)]


_Y_LABELS = ["Sales", "Revenue", "Visitors", "Users", "Cases",
             "Index", "Score", "Volume"]


# Subject phrases used in the claim text
_SUBJECTS = ["the value", "the index", "the count", "the figure",
             "the metric", "the line"]


class ChartQuantifierClaimQA(StandaloneVisualEnv):
    ENV_NAME = "chart_quantifier_claim"
    # Strict bare-text exact-match scoring on raw
    # prediction (no wrapper-stripping). Override the default 3-of-3 wrapper
    # instructions to teach the model to put the bare answer on the final line.
    # Verifier was extended to extract the last non-empty line as a candidate.
    _WRAPPER_INSTRUCTIONS = [
        "Think step by step. End your response with the bare answer (single letter / true|false / single word) on its own final line, with no wrapper around it.",
        "Reason through the problem, then on the very last line of your response output ONLY the bare answer (a single letter, true|false, or a single word) - no <answer>, \\boxed{}, or 'Final answer:' prefix.",
        "Work through the problem step by step. Your final line must be the bare answer alone (single letter / true|false / single word), nothing else on that line.",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n = 4
        elif level <= 2:
            n = 5
        elif level <= 4:
            n = 6 + (level - 3)
        elif level <= 6:
            n = 8
        elif level <= 7:
            n = 10
        elif level == 8:
            n = 12  # 2026-05-04: bumped L9 difficulty (also L8)
        else:
            n = 14
        if level == 0:
            kinds = ["never_above"]
        elif level <= 2:
            kinds = ["never_above", "always_above", "at_least_once_below"]
        else:
            kinds = ["never_above", "always_above", "at_least_once_below",
                     "at_least_once_above", "always_below"]
        return {"level": level, "n": n, "kinds": kinds}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1103 + level * 73 + 29)

        n = cfg["n"]
        # period
        if level == 0:
            period_kind = "months"
        else:
            period_kind = rng.choice(["months", "quarters", "years"])
        if period_kind == "months":
            x_labels = _month_labels(n)
            period_unit = "month"
        elif period_kind == "quarters":
            x_labels = _quarter_labels(n)
            period_unit = "quarter"
        else:
            start = rng.choice([2010, 2012, 2014, 2015, 2016, 2017])
            x_labels = _year_labels(n, start)
            period_unit = "year"

        y_label = rng.choice(_Y_LABELS)
        subj = rng.choice(_SUBJECTS)
        kind = rng.choice(cfg["kinds"])

        # L0 trivial: 4-month line all <= 80, claim "never exceeded 100" → true.
        if level == 0:
            values = [50, 60, 70, 80]
            T = 100
            actually_never_above = max(values) <= T
            claim = f"{subj.capitalize()} never exceeded {T} during the period shown."
            is_true = actually_never_above  # = True
            answer = "true" if is_true else "false"
            sidx = (self.seed or 0) % len(_TEMPLATES_PROMPTS)
            question = _TEMPLATES_PROMPTS[sidx].format(claim=claim)
            img = self._render_line(x_labels, values, y_label, T_ref=None, rng=rng)
            return question, answer, img

        # General — retry the value+threshold engineering up to 40 times so
        # that occasional unlucky draws (e.g. min(values) = 10 → margin pushes
        # T below 1) don't fail the whole problem.
        if level <= 3:
            v_min, v_max, step = 10, 100, 5
        elif level <= 6:
            v_min, v_max, step = 10, 200, 5
        else:
            v_min, v_max, step = 5, 250, 1

        # Helper: only accept positive thresholds for non-negative-data charts
        def _ok_T(Tv):
            return Tv >= 1

        problem = None
        for _ in range(40):
            values_try = [rng.choice(list(range(v_min, v_max + 1, step))) for _ in range(n)]
            if max(values_try) - min(values_try) < 4 * step:
                continue
            mx, mn = max(values_try), min(values_try)
            margin = max(int((mx - mn) * 0.2), step * 2)
            is_true_try = rng.random() < 0.5
            attempt = self._build_claim(kind, values_try, mn, mx, margin,
                                        is_true_try, subj, period_unit, _ok_T)
            if attempt is not None:
                problem = (values_try, attempt, is_true_try)
                break
        if problem is None:
            return None
        values, (claim, T), is_true = problem

        answer = "true" if is_true else "false"
        sidx = (self.seed or 0) % len(_TEMPLATES_PROMPTS)
        question = _TEMPLATES_PROMPTS[sidx].format(claim=claim)
        img = self._render_line(x_labels, values, y_label, T_ref=T, rng=rng)
        return question, answer, img

    def _build_claim(self, kind, values, mn, mx, margin, is_true, subj,
                     period_unit, ok_T_fn):
        """Engineer a threshold T and claim text such that the claim has the
        requested truth value with at least `margin` separation from boundary.

        Returns (claim_str, T_value) or None if infeasible for this draw.
        """
        if kind == "never_above":
            # Claim: subj never exceeded T   ⇔ max(values) <= T
            if is_true:
                T = mx + margin
                if not ok_T_fn(T) or any(v >= T for v in values):
                    return None
            else:
                T = mx - margin
                if not ok_T_fn(T) or T < mn or not any(v > T for v in values):
                    return None
                if max(values) - T < margin:
                    return None
            claim = f"{subj.capitalize()} never exceeded {T} during the period shown."
            return claim, T

        if kind == "always_above":
            # Claim: subj exceeded T in every period  ⇔ min(values) > T
            if is_true:
                T = mn - margin
                if not ok_T_fn(T) or not all(v > T for v in values):
                    return None
                if min(values) - T < margin:
                    return None
            else:
                T = mn + margin
                if not ok_T_fn(T) or not any(v <= T for v in values):
                    return None
                if T - min(values) < margin:
                    return None
            claim = f"{subj.capitalize()} exceeded {T} in every {period_unit} shown."
            return claim, T

        if kind == "at_least_once_below":
            if is_true:
                T = mn + margin
                if not ok_T_fn(T) or not any(v < T for v in values):
                    return None
                if T - min(values) < margin:
                    return None
            else:
                T = mn - margin
                if not ok_T_fn(T) or any(v < T for v in values):
                    return None
                if mn - T < margin:
                    return None
            claim = f"{subj.capitalize()} dropped below {T} at least once during the period shown."
            return claim, T

        if kind == "at_least_once_above":
            if is_true:
                T = mx - margin
                if not ok_T_fn(T) or not any(v > T for v in values):
                    return None
                if mx - T < margin:
                    return None
            else:
                T = mx + margin
                if not ok_T_fn(T) or any(v > T for v in values):
                    return None
                if T - mx < margin:
                    return None
            claim = f"{subj.capitalize()} rose above {T} at least once during the period shown."
            return claim, T

        if kind == "always_below":
            if is_true:
                T = mx + margin
                if not ok_T_fn(T) or not all(v < T for v in values):
                    return None
                if T - mx < margin:
                    return None
            else:
                T = mx - margin
                if not ok_T_fn(T) or not any(v >= T for v in values):
                    return None
                if mx - T < margin:
                    return None
            claim = f"{subj.capitalize()} stayed below {T} in every {period_unit} shown."
            return claim, T

        return None

    def _render_line(self, x_labels, values, y_label, T_ref, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        line_color = palette[0]
        bg = "#ffffff"
        n = len(x_labels)
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        x = list(range(n))
        ax.plot(x, values, "-o", color=line_color, linewidth=2.4,
                markersize=8, markerfacecolor=line_color,
                markeredgecolor="#1a1a1a")
        max_v = max(values)
        for xi, v in zip(x, values):
            ax.text(xi, v + max_v * 0.02, str(v),
                    ha="center", va="bottom", fontsize=10,
                    color="#111", fontweight="bold")
        if T_ref is not None:
            ax.axhline(T_ref, color="#c0392b", linewidth=1.6, linestyle="--",
                       label=f"y = {T_ref}")
            ax.legend(loc="best", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        # y-limits include T_ref so claim/threshold visible
        ref_for_lim = T_ref if T_ref is not None else max_v
        y_lo = min(min(values), ref_for_lim) - max_v * 0.10 - 5
        y_hi = max(max_v, ref_for_lim) + max_v * 0.18 + 5
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Strict lowercase 'true'/'false' single-word match (with mild lenience).
        # Same pattern as chart_claim_verify_qa.
        import re
        p = predicted.strip().lower().rstrip(".").strip()
        m = re.search(r"\b(true|false)\b", p)
        if not m:
            return False
        return m.group(1) == ground_truth.strip().lower()
