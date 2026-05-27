"""
Bar Chart Position Claim QA — fact-checking task.

A bar chart is shown with N bars labelled by category (e.g., years or months).
A textual claim references the *position* of the longest or shortest bar in
the time period ("at the start", "in the middle", "at the end"). The model
outputs `true` or `false`.

Difficulty axes:
  - bar count N (4 -> 12)
  - whether highlight (target color) is mentioned
  - position language: "in the middle of the time period", "at the start",
    "at the end" — for false claims, max actually appears at a different position
  - close runner-up (second-highest is near max) at high levels
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Look at the bar chart. Claim: \"{claim}\" Output `true` or `false` (lowercase, single word).",
    "Examine the bar chart in the image. Decide whether the following claim is true or false: \"{claim}\" Reply with `true` or `false` only,.",
    "Bar chart fact check. Claim: \"{claim}\" Place `true` or `false`.",
    "From the bar chart in the image, evaluate: \"{claim}\" Answer with a single word (true/false).",
    "Inspect the bar chart, then verify the claim: \"{claim}\" Output `true` or `false` (no other text).",
    "Bar chart shown. Verify: \"{claim}\" Answer with `true` or `false`.",
    "Read the bar chart and judge whether the claim is correct: \"{claim}\" Reply with one word — `true` or `false` —.",
    "Look at the chart. Is the following statement true or false? \"{claim}\" Reply `true` or `false`.",
    "Use the bar chart to determine the truth of: \"{claim}\" Output `true` or `false` only.",
    "Bar chart claim verification. \"{claim}\" — Place `true` or `false`.",
    "Examine the chart. Decide: \"{claim}\" Output `true` or `false` (lowercase).",
    "Review the bar chart. Is this true or false? \"{claim}\" Place answer.",
    "From the chart in the image, judge the claim: \"{claim}\" Output `true` or `false`.",
    "Bar chart in image. Claim: \"{claim}\" Reply with one word (`true` or `false`).",
    "Verify the claim about the bar chart: \"{claim}\" Output `true` or `false` only.",
    "Read the chart carefully and evaluate: \"{claim}\" Answer is `true` or `false`, placed.",
    "Bar chart fact check. \"{claim}\" — Reply `true` or `false`.",
]

_CATEGORY_SETS = [
    # Year sequences
    [str(y) for y in range(2010, 2024)],
    [str(y) for y in range(2008, 2022)],
    [str(y) for y in range(2014, 2026)],
    # Months
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"],
    # Quarters
    ["Q1", "Q2", "Q3", "Q4"],
    # Custom labels
    ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Gamma",
     "Hotel", "India", "Juliet", "Kilo", "Lima"],
]

_COLOR_NAMES = ["red", "blue", "green", "orange", "purple", "teal"]
_COLOR_HEX = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#27ae60",
    "orange": "#e67e22",
    "purple": "#8e44ad",
    "teal": "#16a085",
}


def _position_label(idx: int, n: int) -> str:
    """Return 'start' / 'middle' / 'end' for bar position."""
    if idx < n / 3.0:
        return "start"
    elif idx < 2 * n / 3.0:
        return "middle"
    else:
        return "end"


def _position_phrase(pos: str, period_word: str) -> str:
    if pos == "start":
        return f"at the start of the {period_word} shown"
    elif pos == "middle":
        return f"in the middle of the {period_word} shown"
    return f"at the end of the {period_word} shown"


class BarChartPositionClaimQA(StandaloneVisualEnv):
    ENV_NAME = "bar_chart_position_claim"
    # Strict bare-text exact-match scoring on raw
    # prediction (no wrapper-stripping). Override the default 3-of-3 wrapper
    # instructions to teach the model to put the bare answer on the final line.
    # Verifier was extended to extract the last non-empty line as a candidate.
    _WRAPPER_INSTRUCTIONS = [
        "Think step by step. End your response with the bare answer (single letter / true|false / single word) on its own final line, with no wrapper around it.",
        "Reason through the problem, then on the very last line of your response output ONLY the bare answer (a single letter, true|false, or a single word) - no <answer>, \\boxed{}, or 'Final answer:' prefix.",
        "Work through the problem step by step. Your final line must be the bare answer alone (single letter / true|false / single word), nothing else on that line.",
    ]

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output exactly `true` or `false` (lowercase) inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_bars": 4, "use_color_mention": False, "extreme": "longest",
                    "close_runner_up": False}
        if level <= 2:
            return {"n_bars": 5, "use_color_mention": False, "extreme": "any",
                    "close_runner_up": False}
        if level <= 4:
            return {"n_bars": 7, "use_color_mention": True, "extreme": "any",
                    "close_runner_up": False}
        if level <= 6:
            return {"n_bars": 9, "use_color_mention": True, "extreme": "any",
                    "close_runner_up": False}
        if level <= 8:
            return {"n_bars": 11, "use_color_mention": True, "extreme": "any",
                    "close_runner_up": True}
        return {"n_bars": 12, "use_color_mention": True, "extreme": "any",
                "close_runner_up": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4423 + level * 29 + 17)

        n = cfg["n_bars"]
        # Pick category set that has at least n labels
        candidate_sets = [s for s in _CATEGORY_SETS if len(s) >= n]
        cat_set = rng.choice(candidate_sets)
        # Take a contiguous slice of length n if it's a year/month sequence,
        # else random sample
        if cat_set[0].isdigit() or cat_set[0] in ["Jan", "Q1"]:
            start_idx = rng.randint(0, len(cat_set) - n)
            labels = cat_set[start_idx:start_idx + n]
            period_word = rng.choice(["period", "time period", "range"])
        else:
            labels = rng.sample(cat_set, n)
            period_word = rng.choice(["set of categories", "list", "set"])

        # Generate bar values
        if cfg["close_runner_up"]:
            base = rng.randint(20, 50)
            values = [base + rng.randint(-5, 5) for _ in range(n)]
            # Set the actual extreme bar to be just slightly above runner-up
            extreme_idx = rng.randint(0, n - 1)
            values[extreme_idx] = max(values) + rng.choice([1, 2, 3])
        else:
            values = [rng.randint(5, 60) for _ in range(n)]
            # Make extreme unique
            extreme_idx = max(range(n), key=lambda i: values[i])
            # Bump it slightly to ensure uniqueness
            second = sorted(values)[-2] if n >= 2 else 0
            if values[extreme_idx] == second:
                values[extreme_idx] += rng.choice([2, 3, 4])

        # Decide which extreme to claim about: longest or shortest
        if cfg["extreme"] == "longest":
            claim_about = "longest"
        else:
            claim_about = rng.choice(["longest", "shortest"])
        if claim_about == "longest":
            actual_extreme_idx = max(range(n), key=lambda i: values[i])
            extreme_word = rng.choice(["longest", "tallest", "highest"])
        else:
            actual_extreme_idx = min(range(n), key=lambda i: values[i])
            extreme_word = rng.choice(["shortest", "smallest", "lowest"])

        # Pick the color (used for highlight/mention)
        if cfg["use_color_mention"]:
            color_name = rng.choice(_COLOR_NAMES)
            # Assign extreme bar the chosen color; others get neutral grey
            bar_colors = ["#bdc3c7"] * n
            bar_colors[actual_extreme_idx] = _COLOR_HEX[color_name]
            color_phrase = f"{color_name} bar"
        else:
            color_name = None
            # Use a default palette
            palette = ["#3498db", "#9b59b6", "#1abc9c", "#f39c12", "#e74c3c",
                        "#2ecc71", "#34495e", "#e67e22", "#16a085", "#d35400",
                        "#7f8c8d", "#c0392b"]
            bar_colors = [palette[i % len(palette)] for i in range(n)]
            color_phrase = "bar"

        # Build claim — sometimes true (the position matches), sometimes false
        is_true = rng.random() < 0.5
        actual_pos = _position_label(actual_extreme_idx, n)
        if is_true:
            claim_pos = actual_pos
        else:
            others = [p for p in ["start", "middle", "end"] if p != actual_pos]
            claim_pos = rng.choice(others)

        pos_phrase = _position_phrase(claim_pos, period_word)
        claim = (f"the {extreme_word} {color_phrase} is {pos_phrase}")

        gt = "true" if is_true else "false"

        template = rng.choice(_TEMPLATES)
        question = template.format(claim=claim)

        img = self._render(labels, values, bar_colors, n, rng)
        return question, gt, img

    def _render(self, labels, values, bar_colors, n, rng) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0"])
        sc = rng.uniform(0.95, 1.15)
        fig, ax = plt.subplots(figsize=(7.5 * sc, 4.5 * sc), dpi=120)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.bar(range(n), values, color=bar_colors,
                edgecolor="#222", linewidth=0.6)
        ax.set_xticks(range(n))
        rotation = 30 if n >= 8 else 0
        ax.set_xticklabels(labels, rotation=rotation,
                            fontsize=10, ha="right" if rotation else "center")
        ax.set_ylabel(rng.choice(["Value", "Count", "Amount", "Quantity"]),
                       fontsize=10)
        ax.set_title(rng.choice(["Distribution", "Bar Chart", "Values by Category",
                                   "Series", "Data"]),
                       fontsize=12)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylim(0, max(values) * 1.18 + 2)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=120)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # The base _check_answer would handle exact match; but predicted may
        # be 'True', 'TRUE', 'true.', 'True.', '`true`', etc. Normalize.
        p = predicted.strip().lower().rstrip(".").strip("`'\" ")
        g = ground_truth.strip().lower()
        # Map yes/no synonyms
        if p in ("true", "t", "yes"):
            p = "true"
        elif p in ("false", "f", "no"):
            p = "false"
        return p == g


if __name__ == "__main__":
    env = BarChartPositionClaimQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer}")
