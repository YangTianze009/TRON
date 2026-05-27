"""
Count peaks of an oscillating curve in a stated x-range. Targets the
"Number-in-General" gap on chart benchmarks where counting envs are
underrepresented.

Difficulty axes:
  - frequency / # peaks (1 → 6)
  - subplot grid (1 → 4)
  - condition complexity (count peaks vs count peaks above threshold)
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


_TEMPLATES = [
    "How many local maxima are visible on the curve in the image (within the displayed x-range)? Reply with the integer count in <answer>...</answer>.",
    "Count the number of peaks (local maxima) of the curve shown. Place the integer in <answer>...</answer>.",
    "Examine the curve and count the prominent local maxima within the visible x-range. Integer answer in <answer>...</answer>.",
    "How many peaks does the displayed curve have in the shown x-range? Integer in <answer>...</answer>.",
    "Identify the local maxima of the curve shown and report how many there are. Integer in <answer>...</answer>.",
    "Inspect the plot and count the peaks (local maxima). Place integer in <answer>...</answer>.",
    "Number of local maxima on the visible curve? Integer answer in <answer>...</answer>.",
    "Tally the peaks on the curve shown. Place integer in <answer>...</answer>.",
    "Count the maxima (peaks) of the curve in the image. Integer in <answer>...</answer>.",
    "How many distinct peaks are present on the curve over the x-range shown? Integer in <answer>...</answer>.",
    "Count the local maxima visible on the plotted curve. Integer answer in <answer>...</answer>.",
    "Report the number of peaks of the curve shown. Integer in <answer>...</answer>.",
    "Examine the curve in the image. How many local maxima does it have within the displayed window? Integer in <answer>...</answer>.",
    "Count the peaks of the plotted function in the visible x-range. Integer in <answer>...</answer>.",
    "How many local maxima can you count on the curve shown? Place integer in <answer>...</answer>.",
    "Identify and count the peaks of the curve in the plot. Integer answer in <answer>...</answer>.",
]

_TEMPLATES_THRESH = [
    "How many local maxima of the curve shown lie above y = {T}? Integer count in <answer>...</answer>.",
    "Count the peaks of the curve whose y-value is greater than {T}. Integer in <answer>...</answer>.",
    "How many of the curve's local maxima exceed y = {T}? Integer in <answer>...</answer>.",
    "Tally the peaks on the curve whose height is above {T}. Integer in <answer>...</answer>.",
    "Among the local maxima of the curve in the image, how many lie above y = {T}? Integer in <answer>...</answer>.",
]

_TEMPLATES_SUBPLOT = [
    "How many of the {grid} subplots have a curve with at least one peak above y = {T}? Integer in <answer>...</answer>.",
    "Count the subplots in which the curve has any local maximum exceeding y = {T}. Integer in <answer>...</answer>.",
    "In how many of the subplots does the curve reach a peak above y = {T}? Integer in <answer>...</answer>.",
]


def _count_peaks(y: np.ndarray) -> int:
    # local max = y[i-1] < y[i] > y[i+1] (interior)
    n = len(y)
    cnt = 0
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            cnt += 1
    return cnt


def _count_peaks_above(y: np.ndarray, T: float) -> int:
    n = len(y)
    cnt = 0
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > T:
            cnt += 1
    return cnt


def _make_curve(rng: random.Random, n_peaks_target: int, x_range=(0, 10),
                noise=0.02):
    """Generate a curve with approximately n_peaks_target local maxima."""
    x = np.linspace(x_range[0], x_range[1], 800)
    # Sum of sinusoids with frequencies tuned to give target peak count
    freq = max(0.5, n_peaks_target * math.pi / (x_range[1] - x_range[0]))
    phase = rng.uniform(0, 2 * math.pi)
    amp1 = 1.0
    y = amp1 * np.sin(freq * x + phase)
    # Add a slow drift
    drift_freq = 0.1 + rng.random() * 0.2
    drift_phase = rng.uniform(0, 2 * math.pi)
    y += 0.3 * np.sin(drift_freq * x + drift_phase)
    # Add noise — small enough not to create spurious peaks
    if noise > 0:
        y += rng.uniform(-noise, noise) * np.random.RandomState(
            rng.randint(0, 1 << 20)).randn(len(x)) * 0.05
    return x, y


class FunctionPeakCountQA(StandaloneVisualEnv):
    ENV_NAME = "function_peak_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            mode = "single_count"
            target_peaks = (1, 3)
        elif level <= 5:
            mode = "single_threshold"
            target_peaks = (3, 5)
        elif level <= 7:
            mode = "subplot_threshold"
            target_peaks = (2, 5)
        else:
            mode = "subplot_threshold"
            target_peaks = (3, 6)
        return {"level": level, "mode": mode, "target_peaks": target_peaks}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1063 + level * 89 + 17)

        if cfg["mode"] == "single_count":
            n_target = rng.randint(*cfg["target_peaks"])
            x, y = _make_curve(rng, n_target)
            cnt = _count_peaks(y)
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx]
            answer = str(cnt)
            img = self._render_single(x, y)
        elif cfg["mode"] == "single_threshold":
            n_target = rng.randint(*cfg["target_peaks"])
            x, y = _make_curve(rng, n_target)
            T = round(rng.uniform(0.0, 0.6), 1)
            cnt = _count_peaks_above(y, T)
            sidx = (self.seed or 0) % len(_TEMPLATES_THRESH)
            question = _TEMPLATES_THRESH[sidx].format(T=T)
            answer = str(cnt)
            img = self._render_single(x, y, threshold=T)
        else:  # subplot_threshold
            grid_n = rng.choice([4, 6])
            T = round(rng.uniform(0.0, 0.5), 1)
            curves = []
            cnt_meeting = 0
            for _ in range(grid_n):
                n_target = rng.randint(*cfg["target_peaks"])
                x, y = _make_curve(rng, n_target)
                curves.append((x, y))
                if _count_peaks_above(y, T) > 0:
                    cnt_meeting += 1
            sidx = (self.seed or 0) % len(_TEMPLATES_SUBPLOT)
            question = _TEMPLATES_SUBPLOT[sidx].format(grid=grid_n, T=T)
            answer = str(cnt_meeting)
            img = self._render_subplots(curves, threshold=T)
        return question, answer, img

    def _render_single(self, x, y, threshold=None):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.plot(x, y, color="#2c3e80", linewidth=2)
        if threshold is not None:
            ax.axhline(threshold, color="#c0392b", linewidth=1.5,
                       linestyle="--", label=f"y = {threshold}")
            ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render_subplots(self, curves, threshold=None):
        n = len(curves)
        cols = 3 if n in (3, 6) else 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows),
                                 dpi=120)
        fig.patch.set_facecolor("#ffffff")
        if rows * cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        else:
            axes = list(axes)
        for k, (x, y) in enumerate(curves):
            r, c = divmod(k, cols)
            ax = axes[r][c] if rows > 1 else axes[0][c]
            ax.plot(x, y, color="#2c3e80", linewidth=1.5)
            if threshold is not None:
                ax.axhline(threshold, color="#c0392b", linewidth=1.0,
                           linestyle="--")
            ax.set_title(f"({chr(ord('a') + k)})", fontsize=11, loc="left")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
        # Hide unused
        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            ax = axes[r][c] if rows > 1 else axes[0][c]
            ax.axis("off")
        plt.tight_layout()

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
