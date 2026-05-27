"""
Chart with LaTeX-Label QA — X26 (reference reasoning_val).

Renders a multi-line chart whose curve labels use matplotlib mathtext
(subscripts, Greek letters, units like Å, intervals like p ∈ [0.9, 0.999]).
Asks "which curve has highest X" / "second-highest at X=Y", and the answer
is the verbatim label string. Tests subscript fidelity (e.g. ``b_3`` vs
``b_2`` vs ``b_4``).

Verbatim sample style (design notes Text-in-Chart §X26):

    Q-IDX 433: "Which variable in subplot (b) has the highest exact
                 solution?" GT `b_3` (lost to `μ_s` baseline error)
    Q-IDX 631: "What is the label of the line that has the second highest
                 fidelity between 1 and 2 iterations in subplot (d)?"
                 GT `R=0.2Å` (lost to `R=0.7Å`)

reference judge accepts unicode-equivalent matches (`α↔alpha`); our env
chooses ASCII-canonical answer strings (e.g. ``b_3``) to maximize match
robustness.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Each label-family is a list of (display_mathtext, ascii_canonical) pairs.
# ``display`` is what appears in the legend; ``ascii`` is what the model
# should emit and what we use as ground truth.
_LABEL_FAMILIES = [
    {
        "name": "b_subscript",
        "labels": [(f"$b_{i}$", f"b_{i}") for i in range(1, 6)],
    },
    {
        "name": "R_angstrom",
        "labels": [(f"$R={v}\\,\\AA$", f"R={v}A")
                   for v in [0.2, 0.4, 0.5, 0.7, 1.0]],
    },
    {
        "name": "mu_subscript",
        "labels": [(f"$\\mu_{c}$", f"mu_{c}")
                   for c in ["s", "k", "d", "a"]],
    },
    {
        "name": "lambda_subscript",
        "labels": [(f"$\\lambda_{{{n}}}$", f"lambda_{n}")
                   for n in ["AIC", "BIC", "GIC", "DIC"]],
    },
    {
        "name": "p_interval",
        "labels": [(f"$p\\in[0.{a},\\,0.{b}]$", f"p in [0.{a}, 0.{b}]")
                   for a, b in [("9", "999"), ("8", "99"), ("7", "9"),
                                ("5", "8")]],
    },
    {
        "name": "gamma_subscript",
        "labels": [(f"$\\gamma_{i}$", f"gamma_{i}") for i in range(1, 6)],
    },
]


class ChartWithLatexLabelQA(StandaloneVisualEnv):
    ENV_NAME = "chart_with_latex_label"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_curves": 3}
        if level <= 5:
            return {"n_curves": 4}
        return {"n_curves": 5}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1789 + level * 149 + 71)
        np_rng = np.random.RandomState(rng.randint(0, 1_000_000))

        for _ in range(15):
            res = self._try(rng, np_rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, np_rng, cfg):
        n = cfg["n_curves"]
        family = rng.choice(_LABEL_FAMILIES)
        labels_pool = list(family["labels"])
        rng.shuffle(labels_pool)
        if len(labels_pool) < n:
            return None
        labels = labels_pool[:n]

        # Generate diverse curves: each curve is y = base + slope * x + noise
        x = np.linspace(0, 10, 30)
        curves = []
        for i in range(n):
            base = rng.uniform(0, 10)
            slope = rng.uniform(-1.0, 1.0)
            curve_amp = rng.uniform(2, 5)
            phase = rng.uniform(0, 2 * np.pi)
            y = base + slope * x + curve_amp * np.sin(phase + 0.4 * x)
            curves.append(y)

        # Pick question: highest peak / second-highest peak
        peaks = [float(c.max()) for c in curves]
        # Ensure unique peaks (separation >= 1).
        srt = sorted(peaks, reverse=True)
        if srt[0] - srt[1] < 1.0:
            return None
        if n >= 3 and srt[1] - srt[2] < 1.0:
            return None

        kind = rng.choice(["highest_peak", "second_highest", "lowest_peak"])
        if kind == "highest_peak":
            idx = peaks.index(max(peaks))
            stem = "Which curve has the highest peak?"
        elif kind == "second_highest":
            sorted_peaks = sorted(peaks, reverse=True)
            second_val = sorted_peaks[1]
            idx = peaks.index(second_val)
            stem = "Which curve has the second-highest peak?"
        else:
            idx = peaks.index(min(peaks))
            stem = "Which curve has the lowest peak?"

        ascii_labels = [a for (_, a) in labels]
        answer = ascii_labels[idx]
        prompt = (
            f"{stem} Provide just the curve's label as shown in the "
            f"legend (e.g. b_3, R=0.7A, mu_s — preserve subscripts and "
            f"units exactly)."
        )
        image = self._render(x, curves, labels)
        return prompt, answer, image

    # Unicode -> ASCII normalization for label matching. Models reading
    # the rendered legend produce Unicode (e.g. λ_AIC, μ_s, Å, ∈) instead
    # of the ASCII canonical form we use as ground truth.
    _UNICODE_TO_ASCII = [
        ("λ", "lambda"), ("Λ", "lambda"),
        ("μ", "mu"), ("Μ", "mu"),
        ("γ", "gamma"), ("Γ", "gamma"),
        ("α", "alpha"), ("Α", "alpha"),
        ("β", "beta"), ("Β", "beta"),
        ("Å", "A"),
        ("∈", " in "),
        (" ", " "), (" ", " "), (" ", " "),
        ("−", "-"),
        # Unicode subscripts → "_<digit>" (so model output `b₃` ≡ gt `b_3`)
        ("₀", "_0"), ("₁", "_1"), ("₂", "_2"), ("₃", "_3"), ("₄", "_4"),
        ("₅", "_5"), ("₆", "_6"), ("₇", "_7"), ("₈", "_8"), ("₉", "_9"),
        # Unicode superscripts → "^<digit>"
        ("⁰", "^0"), ("¹", "^1"), ("²", "^2"), ("³", "^3"), ("⁴", "^4"),
        ("⁵", "^5"), ("⁶", "^6"), ("⁷", "^7"), ("⁸", "^8"), ("⁹", "^9"),
        ("$", ""), ("\\", ""),
        ("{", ""), ("}", ""),
    ]

    def _normalize_label(self, s: str) -> str:
        s = s.strip()
        for u, a in self._UNICODE_TO_ASCII:
            s = s.replace(u, a)
        # Collapse multiple spaces
        s = " ".join(s.split())
        return s.lower()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Try direct standard check first
        if super()._check_answer(predicted, ground_truth):
            return True
        # Normalize both sides (Unicode↔ASCII) and compare
        p = self._normalize_label(predicted)
        g = self._normalize_label(ground_truth)
        if p == g:
            return True
        # Also accept stripping spaces entirely (e.g. "p in [0.9,0.999]"
        # vs "p in [0.9, 0.999]")
        if p.replace(" ", "") == g.replace(" ", ""):
            return True
        return False

    def _render(self, x, curves, labels):
        style = self._random_style()
        palette = list(style["palette"])
        n = len(curves)
        fig, ax = plt.subplots(figsize=(7 * style["figsize_scale"],
                                        4.6 * style["figsize_scale"]))
        for i, y in enumerate(curves):
            ax.plot(x, y, label=labels[i][0],
                    color=palette[i % len(palette)],
                    linewidth=style["line_width"], marker="o",
                    markersize=4)
        ax.set_xlabel("x", fontsize=style["font_size_base"])
        ax.set_ylabel("y", fontsize=style["font_size_base"])
        ax.legend(fontsize=style["font_size_base"], loc=style["legend_loc"])
        ax.set_title("Curves with LaTeX-formatted labels",
                     fontsize=style["font_size_base"] + 1)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
