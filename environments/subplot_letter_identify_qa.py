"""
Subplot identification: render an arXiv-style figure with multiple subplots
labeled (a), (b), (c), ..., each containing a small chart (line, bars, or
scatter). Pose a global condition (e.g. "in which subplot is the highest
peak") and ask for the single subplot letter answer.

Output: single uppercase letter A-H (the base verifier handles A-H mapping
case-insensitively; we cap subplot count at 8 to stay within that range).
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


_TEMPLATES_HIGHEST_PEAK = [
    "In which subplot does the curve reach the highest peak (largest maximum y-value)? Reply with the single subplot letter (a, b, c, ...) in <answer>...</answer>.",
    "Which subplot contains the curve with the greatest maximum y-value? Single letter answer in <answer>...</answer>.",
    "Examine the subplots. Which one shows the highest peak? Single letter in <answer>...</answer>.",
    "Identify the subplot whose curve attains the largest y-value. Letter only in <answer>...</answer>.",
    "Which subplot's curve has the highest peak overall? Reply with one lowercase letter (a-h) in <answer>...</answer>.",
    "Compare the subplots. Which one has the highest maximum on its curve? Letter in <answer>...</answer>.",
    "In which subplot does the curve reach its largest peak? Letter answer in <answer>...</answer>.",
    "Look across the subplots. Which one shows the curve with the greatest maximum value? Letter in <answer>...</answer>.",
    "Which subplot has the curve with the largest peak? Single letter in <answer>...</answer>.",
    "From the subplots shown, identify the one with the highest peak. Letter in <answer>...</answer>.",
    "Which lettered subplot contains the curve with the highest maximum? Letter in <answer>...</answer>.",
    "Tell me the letter of the subplot that has the curve reaching the highest y-value. Letter in <answer>...</answer>.",
    "Which subplot's curve goes the highest? Single letter answer in <answer>...</answer>.",
    "Identify by letter the subplot whose curve has the largest peak. Letter in <answer>...</answer>.",
    "Across the subplots, which one shows the highest curve maximum? Letter in <answer>...</answer>.",
    "Letter of the subplot with the curve reaching the largest y? Letter in <answer>...</answer>.",
]

_TEMPLATES_LOWEST = [
    "In which subplot does the curve reach the lowest minimum y-value? Reply with the single subplot letter (a, b, c, ...) in <answer>...</answer>.",
    "Which subplot contains the curve with the smallest minimum y-value? Single letter in <answer>...</answer>.",
    "Examine the subplots. Which one shows the lowest valley (smallest min)? Single letter in <answer>...</answer>.",
    "Identify the subplot whose curve attains the smallest y-value. Letter only in <answer>...</answer>.",
    "Which subplot's curve drops to the lowest y? Reply with one lowercase letter in <answer>...</answer>.",
    "Compare the subplots. Which one has the smallest minimum on its curve? Letter in <answer>...</answer>.",
    "In which subplot does the curve reach its smallest value? Letter answer in <answer>...</answer>.",
    "Look across the subplots. Which one shows the curve with the lowest minimum? Letter in <answer>...</answer>.",
    "Which subplot has the curve dipping to the lowest value? Single letter in <answer>...</answer>.",
    "From the subplots, identify the one with the lowest curve minimum. Letter in <answer>...</answer>.",
    "Which lettered subplot contains the curve with the lowest minimum? Letter in <answer>...</answer>.",
    "Tell me the letter of the subplot whose curve reaches the smallest y-value. Letter in <answer>...</answer>.",
    "Which subplot's curve goes the lowest? Single letter answer in <answer>...</answer>.",
    "Identify by letter the subplot whose curve attains the smallest minimum. Letter in <answer>...</answer>.",
    "Across the subplots, which one shows the lowest curve minimum? Letter in <answer>...</answer>.",
    "Letter of the subplot whose curve drops to the smallest y? Letter in <answer>...</answer>.",
]

_TEMPLATES_EXCEED = [
    "In which subplot does the curve exceed y = {T} at some point? Single letter (a, b, c, ...) in <answer>...</answer>.",
    "Which subplot's curve crosses above y = {T}? Letter in <answer>...</answer>.",
    "Identify the subplot where the curve goes above y = {T}. Letter only in <answer>...</answer>.",
    "Which subplot contains a curve that exceeds y = {T}? Single letter in <answer>...</answer>.",
    "In which lettered subplot does the curve surpass y = {T}? Letter in <answer>...</answer>.",
    "Find the subplot where the curve passes above y = {T}. Letter in <answer>...</answer>.",
    "Which subplot's curve goes above y = {T}? Letter in <answer>...</answer>.",
    "Letter of the subplot whose curve exceeds y = {T}? Letter in <answer>...</answer>.",
    "Across the subplots, which one shows the curve crossing above y = {T}? Letter in <answer>...</answer>.",
    "In which subplot does the curve reach y > {T} at some point? Letter in <answer>...</answer>.",
    "Identify by letter the subplot where the curve exceeds y = {T}. Letter in <answer>...</answer>.",
    "Which subplot has its curve passing above y = {T}? Single letter in <answer>...</answer>.",
    "Tell me the letter of the subplot whose curve goes above y = {T}. Letter in <answer>...</answer>.",
    "Which subplot is the only one where the curve exceeds y = {T}? Letter in <answer>...</answer>.",
    "Letter of the subplot in which the curve crosses y = {T}? Letter in <answer>...</answer>.",
    "From the lettered subplots, which one has its curve above y = {T}? Letter in <answer>...</answer>.",
]


def _make_curve(rng: random.Random, peak_y: float, n_peaks: int = 2):
    """Make a smooth curve scaled so its max is approx peak_y and min ~ -peak_y/3."""
    x = np.linspace(0, 10, 200)
    freq = max(0.5, n_peaks * math.pi / 10.0)
    phase = rng.uniform(0, 2 * math.pi)
    y = np.sin(freq * x + phase)
    # Add slow drift
    drift = 0.2 * np.sin(0.2 * x + rng.uniform(0, 2 * math.pi))
    y = y + drift
    # Normalize: max → peak_y
    cur_max = y.max()
    cur_min = y.min()
    if cur_max != 0:
        y = y / cur_max * peak_y
    return x, y


class SubplotLetterIdentifyQA(StandaloneVisualEnv):
    ENV_NAME = "subplot_letter_identify"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_sub = 4   # 2x2
            modes = ["highest_peak"]
        elif level <= 2:
            n_sub = 4
            modes = ["highest_peak", "lowest"]
        elif level <= 4:
            n_sub = 6
            modes = ["highest_peak", "lowest", "exceed"]
        elif level <= 6:
            n_sub = 6
            modes = ["highest_peak", "lowest", "exceed"]
        elif level <= 8:
            n_sub = 8   # 2x4
            modes = ["highest_peak", "lowest", "exceed"]
        else:
            n_sub = 8   # cap at 8 for letter-A-H verifier compatibility
            modes = ["highest_peak", "lowest", "exceed"]
        return {"level": level, "n_sub": n_sub, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1091 + level * 53 + 41)

        n_sub = cfg["n_sub"]
        mode = rng.choice(cfg["modes"])

        # Generate per-subplot curves with controlled peak heights
        # so the answer is unambiguous.
        if level == 0:
            # 4 subplots: 1 with obvious tall peak, 3 with low / flat curves
            target_idx = rng.randrange(4)
            curves = []
            for i in range(4):
                if i == target_idx:
                    x, y = _make_curve(rng, peak_y=10.0, n_peaks=2)
                else:
                    x, y = _make_curve(rng, peak_y=2.0, n_peaks=2)
                curves.append((x, y))
            answer_idx = target_idx
            sidx = (self.seed or 0) % len(_TEMPLATES_HIGHEST_PEAK)
            question = _TEMPLATES_HIGHEST_PEAK[sidx]
        elif mode == "highest_peak":
            # one subplot has clearly highest peak
            target_idx = rng.randrange(n_sub)
            curves = []
            tall = rng.uniform(7.0, 10.0)
            for i in range(n_sub):
                if i == target_idx:
                    h = tall
                else:
                    h = rng.uniform(2.0, 5.5)   # gap > 1.5 from tall
                x, y = _make_curve(rng, peak_y=h, n_peaks=rng.choice([2, 3, 4]))
                curves.append((x, y))
            answer_idx = target_idx
            sidx = (self.seed or 0) % len(_TEMPLATES_HIGHEST_PEAK)
            question = _TEMPLATES_HIGHEST_PEAK[sidx]
        elif mode == "lowest":
            target_idx = rng.randrange(n_sub)
            curves = []
            low = rng.uniform(7.0, 10.0)   # peak height — but we'll flip to use min
            # We want the "lowest minimum". Easiest: ensure target subplot
            # has biggest amplitude (extends both up and down farther)
            for i in range(n_sub):
                if i == target_idx:
                    h = low   # large amplitude → also lowest min
                else:
                    h = rng.uniform(2.0, 5.5)
                x, y = _make_curve(rng, peak_y=h, n_peaks=rng.choice([2, 3, 4]))
                curves.append((x, y))
            # Verify by min
            mins = [c[1].min() for c in curves]
            answer_idx = mins.index(min(mins))
            sidx = (self.seed or 0) % len(_TEMPLATES_LOWEST)
            question = _TEMPLATES_LOWEST[sidx]
        else:  # exceed
            T = round(rng.uniform(4.0, 6.0), 1)
            # Ensure exactly one subplot exceeds T
            target_idx = rng.randrange(n_sub)
            curves = []
            for i in range(n_sub):
                if i == target_idx:
                    h = rng.uniform(T + 2.0, T + 4.0)   # well above T
                else:
                    h = rng.uniform(0.5, T - 1.0)   # well below T
                x, y = _make_curve(rng, peak_y=h, n_peaks=rng.choice([2, 3, 4]))
                curves.append((x, y))
            # Verify by max
            maxes = [c[1].max() for c in curves]
            exceeders = [i for i, m in enumerate(maxes) if m > T]
            if len(exceeders) != 1:
                # fallback to highest_peak mode
                answer_idx = maxes.index(max(maxes))
            else:
                answer_idx = exceeders[0]
            sidx = (self.seed or 0) % len(_TEMPLATES_EXCEED)
            question = _TEMPLATES_EXCEED[sidx].format(T=T)

        # Answer: uppercase letter
        answer = chr(ord("A") + answer_idx)
        # Determine if exceed line should be shown
        T_show = None
        if mode == "exceed":
            T_show = T
        img = self._render_subplots(curves, threshold=T_show, rng=rng)
        return question, answer, img

    def _render_subplots(self, curves, threshold=None, rng=None):
        n = len(curves)
        if n == 4:
            rows, cols = 2, 2
        elif n == 6:
            rows, cols = 2, 3
        elif n == 8:
            rows, cols = 2, 4
        else:
            cols = 3
            rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(2.6 * cols, 2.2 * rows), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[a] for a in axes]
        else:
            axes = [list(row) for row in axes]
        palette = ["#1f4e79", "#a6192e", "#2e7d32", "#6a1b9a",
                    "#ef6c00", "#00838f", "#5d4037", "#283593"]
        for k, (x, y) in enumerate(curves):
            r, c = divmod(k, cols)
            ax = axes[r][c]
            ax.plot(x, y, color=palette[k % len(palette)], linewidth=1.7)
            if threshold is not None:
                ax.axhline(threshold, color="#c0392b", linewidth=1.0,
                           linestyle="--")
            letter = chr(ord('a') + k)
            ax.set_title(f"({letter})", fontsize=12, loc="left",
                         fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            # Set consistent y-range across subplots so they're visually comparable
            ax.set_ylim(-12, 12)
        # Hide unused
        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            ax = axes[r][c]
            ax.axis("off")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
