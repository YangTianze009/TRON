"""
Density-plot peak reading: render a density plot (1-3 smooth bell-shaped
curves), then ask either:
  (a) the y-coordinate of the peak of a specific named curve, or
  (b) which curve has the highest peak (label answer).

Curves are scaled so peak heights are in the range [5, 50] — avoids the
base verifier's loose ±0.5 absolute tolerance trapping otherwise-wrong
guesses (a peak of 0.5 would let any answer in [0, 1.0] pass).

Output: float (peak y) or single uppercase letter A-C (curve label).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_VALUE_SINGLE = [
    "What is the y-value of the peak (highest point) of the density curve shown? Numeric answer in <answer>...</answer>.",
    "Read the peak y-value of the density plot. Numeric in <answer>...</answer>.",
    "At the maximum of the density curve, what is the y-coordinate? Numeric in <answer>...</answer>.",
    "What is the maximum height of the density curve? Numeric in <answer>...</answer>.",
    "Report the peak y-value of the density curve in the image. Numeric answer in <answer>...</answer>.",
    "Find the y-value at the curve's maximum. Numeric in <answer>...</answer>.",
    "What is the height of the density peak? Numeric value in <answer>...</answer>.",
    "Read off the y-coordinate at the highest point of the density curve. Numeric in <answer>...</answer>.",
    "Numeric value of the density curve's peak (y at max)? In <answer>...</answer>.",
    "What is the peak height (y at maximum) of the displayed density curve? Numeric in <answer>...</answer>.",
    "Report the maximum y-value attained by the density curve shown. Numeric in <answer>...</answer>.",
    "Find the highest y-coordinate of the density curve. Numeric in <answer>...</answer>.",
    "What's the peak value of the density curve in the plot? Numeric in <answer>...</answer>.",
    "From the density plot, what is the y at the curve's peak? Numeric in <answer>...</answer>.",
    "Maximum y on the density curve? Numeric answer in <answer>...</answer>.",
    "At its peak, what y does the density curve reach? Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_VALUE_NAMED = [
    "What is the peak y-value of the {name} density curve? Numeric answer in <answer>...</answer>.",
    "Read the peak height of the curve labeled {name}. Numeric in <answer>...</answer>.",
    "At its maximum, what y-value does the {name} curve reach? Numeric in <answer>...</answer>.",
    "What is the maximum y of the {name} density curve? Numeric in <answer>...</answer>.",
    "Report the peak height of the {name} curve. Numeric in <answer>...</answer>.",
    "Find the y-value at the peak of the {name} density. Numeric in <answer>...</answer>.",
    "What is the maximum height of the {name} curve in the density plot? Numeric in <answer>...</answer>.",
    "Read off the peak y-coordinate of the {name} curve. Numeric in <answer>...</answer>.",
    "Numeric peak value (y at max) of the {name} curve? In <answer>...</answer>.",
    "What peak y does the {name} density curve attain? Numeric in <answer>...</answer>.",
    "Report the maximum y-value of the {name} curve. Numeric in <answer>...</answer>.",
    "Find the highest y-coordinate of the {name} density curve. Numeric in <answer>...</answer>.",
    "What is the peak value of the {name} density curve? Numeric in <answer>...</answer>.",
    "From the density plot, what y does the {name} curve hit at its peak? Numeric in <answer>...</answer>.",
    "Maximum y on the {name} density curve? Numeric answer in <answer>...</answer>.",
    "At its peak, what y does the {name} density curve reach? Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_HIGHEST = [
    "Which density curve has the highest peak? Reply with the single letter (A, B, or C) labeling the curve in <answer>...</answer>.",
    "Among the curves shown, which one has the largest peak? Single letter answer in <answer>...</answer>.",
    "Which curve in the density plot reaches the highest y-value? Single letter in <answer>...</answer>.",
    "Identify by letter the density curve with the tallest peak. Letter in <answer>...</answer>.",
    "Which lettered curve attains the highest maximum? Letter answer in <answer>...</answer>.",
    "Of the density curves, which one has the highest peak? Single letter in <answer>...</answer>.",
    "Which curve label corresponds to the highest peak in the density plot? Letter in <answer>...</answer>.",
    "Tell me the letter of the density curve with the largest peak. Letter in <answer>...</answer>.",
    "Across the density curves, which one has the highest peak? Letter in <answer>...</answer>.",
    "Letter of the curve attaining the largest maximum? Letter in <answer>...</answer>.",
    "Which density curve in the plot reaches the greatest height? Letter in <answer>...</answer>.",
    "From the density curves, identify the one with the tallest peak. Letter in <answer>...</answer>.",
    "Which lettered curve has the largest peak height? Letter in <answer>...</answer>.",
    "Inspect the density curves and report which has the highest peak. Letter in <answer>...</answer>.",
    "Which curve's peak is highest in the density plot? Letter in <answer>...</answer>.",
    "Identify the density curve with the greatest maximum y. Letter in <answer>...</answer>.",
]


def _gaussian_density(x, mu, sigma, amp):
    """Gaussian-shaped curve with amplitude `amp` (peak height = amp)."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class DensityPlotPeakQA(StandaloneVisualEnv):
    ENV_NAME = "density_plot_peak"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_curves = 1
            modes = ["value_single"]
        elif level <= 2:
            n_curves = 1
            modes = ["value_single"]
        elif level <= 4:
            n_curves = 2
            modes = ["value_named", "highest"]
        elif level <= 6:
            n_curves = 3
            modes = ["value_named", "highest"]
        elif level <= 8:
            n_curves = 3
            modes = ["value_named", "highest"]
        else:
            n_curves = 3
            modes = ["value_named", "highest"]
        return {"level": level, "n_curves": n_curves, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1103 + level * 41 + 67)

        n_curves = cfg["n_curves"]
        mode = rng.choice(cfg["modes"])

        # Generate curve params: each (mu, sigma, amp). Amps are well-separated
        # to avoid ambiguity for "highest". Amps are integers in 5..50 range
        # for clean reading and to comfortably exceed verifier 0.5 absolute slack.
        x = np.linspace(0, 100, 500)

        labels = ["A", "B", "C"][:n_curves]
        # Mu spread out so curves are visually distinguishable (peaks at different x)
        if n_curves == 1:
            mus = [rng.uniform(40, 60)]
        elif n_curves == 2:
            mus = [rng.uniform(20, 40), rng.uniform(60, 80)]
        else:
            mus = [rng.uniform(10, 25), rng.uniform(40, 55), rng.uniform(70, 90)]

        sigmas = [rng.uniform(7, 12) for _ in range(n_curves)]

        # L0: 1 curve, peak amp = 10 (clear integer)
        if level == 0:
            amps = [10]
        else:
            # Generate amplitudes well-separated. Pick distinct integers from
            # 10..40 to make peak readouts unambiguous and verification crisp.
            base_amps = list(range(10, 50, 5))
            rng.shuffle(base_amps)
            amps = sorted(base_amps[:n_curves])
            # Space them by at least 5 (already by construction) and shuffle order
            # so target curve isn't always tallest.
            rng.shuffle(amps)

        curves = []
        for mu, sigma, amp in zip(mus, sigmas, amps):
            y = _gaussian_density(x, mu, sigma, amp)
            curves.append(y)

        # Build answer
        if mode == "value_single":
            # Single curve plot — peak height
            target_amp = amps[0]
            answer = _format_num(target_amp)
            sidx = (self.seed or 0) % len(_TEMPLATES_VALUE_SINGLE)
            question = _TEMPLATES_VALUE_SINGLE[sidx]
        elif mode == "value_named":
            # Pick a target curve by label
            target_idx = rng.randrange(n_curves)
            name = labels[target_idx]
            target_amp = amps[target_idx]
            answer = _format_num(target_amp)
            sidx = (self.seed or 0) % len(_TEMPLATES_VALUE_NAMED)
            question = _TEMPLATES_VALUE_NAMED[sidx].format(name=name)
        else:  # highest
            target_idx = max(range(n_curves), key=lambda i: amps[i])
            answer = labels[target_idx]
            sidx = (self.seed or 0) % len(_TEMPLATES_HIGHEST)
            question = _TEMPLATES_HIGHEST[sidx]

        img = self._render_density(x, curves, labels, rng)
        return question, answer, img

    def _render_density(self, x, curves, labels, rng):
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for i, (y, name) in enumerate(zip(curves, labels)):
            color = palette[i % len(palette)]
            ax.plot(x, y, color=color, linewidth=2.2, label=name)
            ax.fill_between(x, 0, y, color=color, alpha=0.18)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        # y-axis: from 0 to max-of-all + headroom
        max_y = max(c.max() for c in curves)
        ax.set_ylim(0, max_y * 1.15 + 1)
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
