"""
Subplot position-word identification: render an academic-paper-style row of
subplots, then ask "from left to right, at what position is the subplot
most/least X?" — answer is an ordinal word ("first"/"second"/"third"/...)
or a side word ("left"/"middle"/"right").

Targets the Text-in-General gap on chart benchmarks where position-word answers
appear (IDX 442 "First subplot", IDX 562 "Third", IDX 117 "first", IDX 977
"left"). Style mirrors academic chart-paper figure layouts.

Output: single word — accepts ordinal (first/second/...), numeric ordinal
(1st/2nd/...), and side words (left/middle/right). Override _check_answer
to map these synonyms.
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


# Templates for ordinal-form questions ("first/second/third/fourth")
# Wording mirrors IDX 442 ("...answer by the order from left to right"),
# IDX 562 ("From top (first) to bottom (fourth), at what index..."),
# IDX 117 ("...answer by their order, i.e. the first, second or third one").
_TEMPLATES_ORDINAL_HIGH = [
    "From left to right, at what position is the subplot whose curve reaches the highest peak? Answer with an ordinal word (first, second, third, ...) in <answer>...</answer>.",
    "Counting from the left, at what index is the subplot showing the largest maximum? Reply with the ordinal (first, second, third, ...) in <answer>...</answer>.",
    "Answer by the order from left to right: which subplot has the highest peak? Ordinal (first/second/third/...) in <answer>...</answer>.",
    "Across the row of subplots, the one with the largest peak is at which position from left to right? Ordinal word in <answer>...</answer>.",
    "Identify the position (use the ordinal word: first, second, third, ...) of the subplot with the highest curve, counting from the left. Ordinal word in <answer>...</answer>.",
    "Which subplot, counting from the left, has the highest maximum? Reply with first, second, third, ... in <answer>...</answer>.",
    "Going left to right, at what index is the subplot the most peaked? Ordinal answer in <answer>...</answer>.",
    "Going left to right, which subplot reaches the largest y-value? Ordinal in <answer>...</answer>.",
]

_TEMPLATES_ORDINAL_LOW = [
    "From left to right, at what position is the subplot whose curve drops to the lowest value? Ordinal word (first, second, third, ...) in <answer>...</answer>.",
    "Counting from the left, at what index is the subplot showing the lowest minimum? Reply with the ordinal in <answer>...</answer>.",
    "Answer by the order from left to right: which subplot has the lowest valley? Ordinal in <answer>...</answer>.",
    "Across the row of subplots, the one with the smallest minimum is at which position from left to right? Ordinal word in <answer>...</answer>.",
    "Identify the position (use the ordinal word: first, second, third, ...) of the subplot reaching the smallest y-value, counting from the left. Ordinal in <answer>...</answer>.",
    "Which subplot, counting from the left, dips to the lowest value? Ordinal answer in <answer>...</answer>.",
    "Going left to right, at what index is the subplot the most negative? Ordinal in <answer>...</answer>.",
    "Going left to right, which subplot reaches the smallest y? Ordinal in <answer>...</answer>.",
]

# Templates for side-word questions ("left/middle/right") — IDX 977 phrasing
_TEMPLATES_SIDE_HIGH = [
    "Which subplot has the highest peak — left, middle, or right? Reply with one of left/middle/right in <answer>...</answer>.",
    "Of the three subplots, which one shows the highest maximum? Answer left, middle, or right in <answer>...</answer>.",
    "Which subplot reaches the largest y-value (left, middle, or right)? Single word in <answer>...</answer>.",
    "Identify which subplot has the tallest peak: left, middle, or right? Place the word in <answer>...</answer>.",
    "Among left, middle, and right subplots, which one shows the highest curve? In <answer>...</answer>.",
    "Compare the three subplots. Which has the largest peak — left, middle, or right? In <answer>...</answer>.",
    "Of left, middle, right, which subplot's curve goes highest? Word answer in <answer>...</answer>.",
    "Which side (left, middle, or right) has the subplot with the largest maximum? In <answer>...</answer>.",
]

_TEMPLATES_SIDE_LOW = [
    "Which subplot has the lowest valley — left, middle, or right? Reply with one of left/middle/right in <answer>...</answer>.",
    "Of the three subplots, which one shows the smallest minimum? Answer left, middle, or right in <answer>...</answer>.",
    "Which subplot drops to the smallest y-value (left, middle, or right)? Single word in <answer>...</answer>.",
    "Identify which subplot has the deepest dip: left, middle, or right? Place the word in <answer>...</answer>.",
    "Among left, middle, and right subplots, which one reaches the lowest curve? In <answer>...</answer>.",
    "Compare the three subplots. Which has the smallest minimum — left, middle, or right? In <answer>...</answer>.",
    "Of left, middle, right, which subplot's curve goes lowest? Word answer in <answer>...</answer>.",
    "Which side (left, middle, or right) has the subplot with the smallest minimum? In <answer>...</answer>.",
]


_ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth"]
_ORDINAL_NUM = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
_SIDE_WORDS = ["left", "middle", "right"]


def _make_curve(rng: random.Random, peak_y: float, n_peaks: int = 2):
    x = np.linspace(0, 10, 200)
    freq = max(0.5, n_peaks * math.pi / 10.0)
    phase = rng.uniform(0, 2 * math.pi)
    y = np.sin(freq * x + phase)
    drift = 0.15 * np.sin(0.2 * x + rng.uniform(0, 2 * math.pi))
    y = y + drift
    cur_max = y.max()
    if cur_max != 0:
        y = y / cur_max * peak_y
    return x, y


class SubplotPositionWordQA(StandaloneVisualEnv):
    ENV_NAME = "subplot_position_word"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_sub = 3
            modes = ["side_high"]
            gap = 6.0   # huge gap → trivial
        elif level <= 2:
            n_sub = 3
            modes = ["side_high", "side_low"]
            gap = 4.0
        elif level <= 4:
            n_sub = 4
            modes = ["ordinal_high", "ordinal_low"]
            gap = 3.0
        elif level <= 6:
            n_sub = 4
            modes = ["ordinal_high", "ordinal_low", "side_high", "side_low"]
            gap = 2.0
        elif level <= 8:
            n_sub = 5
            modes = ["ordinal_high", "ordinal_low"]
            gap = 1.5
        else:
            n_sub = 6
            modes = ["ordinal_high", "ordinal_low"]
            gap = 1.2
        return {"level": level, "n_sub": n_sub, "modes": modes, "gap": gap}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        # mix seed via hash so adjacent seeds produce well-spread RNG states
        # (raw multiply leaves the low bits of small seeds correlated, so e.g.
        # seeds 1/7/42 all picked the same target index at L=0 — bad train signal)
        import hashlib
        sd = int(hashlib.md5(f"spw|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        n_sub = cfg["n_sub"]
        gap = cfg["gap"]
        # If a side mode is requested, force n_sub=3
        mode = rng.choice(cfg["modes"])
        if mode.startswith("side"):
            n_sub = 3

        target_idx = rng.randrange(n_sub)
        # Generate per-subplot peak heights so target is unambiguous.
        # For HIGH modes: target gets the highest peak; others stay below by `gap`.
        # For LOW modes: target gets the largest amplitude (extends most negative);
        #   we control by giving target a strongly larger amplitude than the rest.
        is_high = "high" in mode
        curves = []
        if is_high:
            target_h = rng.uniform(7.5, 9.5)
            other_max = target_h - gap
            for i in range(n_sub):
                if i == target_idx:
                    h = target_h
                else:
                    h = rng.uniform(max(1.0, other_max - 2.0), other_max)
                x, y = _make_curve(rng, peak_y=h, n_peaks=rng.choice([2, 3, 4]))
                curves.append((x, y))
            # verify
            maxes = [c[1].max() for c in curves]
            answer_idx = maxes.index(max(maxes))
        else:
            # LOW mode: pick a target amplitude largest by `gap`
            target_h = rng.uniform(7.5, 9.5)
            other_max = target_h - gap
            for i in range(n_sub):
                if i == target_idx:
                    h = target_h
                else:
                    h = rng.uniform(max(1.0, other_max - 2.0), other_max)
                x, y = _make_curve(rng, peak_y=h, n_peaks=rng.choice([2, 3, 4]))
                curves.append((x, y))
            mins = [c[1].min() for c in curves]
            answer_idx = mins.index(min(mins))

        # Build answer + question
        if mode == "side_high":
            answer = _SIDE_WORDS[answer_idx]   # n_sub == 3
            sidx = (self.seed or 0) % len(_TEMPLATES_SIDE_HIGH)
            question = _TEMPLATES_SIDE_HIGH[sidx]
        elif mode == "side_low":
            answer = _SIDE_WORDS[answer_idx]
            sidx = (self.seed or 0) % len(_TEMPLATES_SIDE_LOW)
            question = _TEMPLATES_SIDE_LOW[sidx]
        elif mode == "ordinal_high":
            answer = _ORDINAL_WORDS[answer_idx]
            sidx = (self.seed or 0) % len(_TEMPLATES_ORDINAL_HIGH)
            question = _TEMPLATES_ORDINAL_HIGH[sidx]
        else:   # ordinal_low
            answer = _ORDINAL_WORDS[answer_idx]
            sidx = (self.seed or 0) % len(_TEMPLATES_ORDINAL_LOW)
            question = _TEMPLATES_ORDINAL_LOW[sidx]

        img = self._render(curves, n_sub, rng)
        return question, answer, img

    def _render(self, curves, n_sub, rng):
        # Always single-row layout so left-to-right ordinal / left-middle-right
        # mapping is unambiguous.
        rows, cols = 1, n_sub
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(2.5 * cols, 2.4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        if cols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        palette = ["#1f4e79", "#a6192e", "#2e7d32", "#6a1b9a",
                    "#ef6c00", "#00838f"]
        for k, (x, y) in enumerate(curves):
            ax = axes[k]
            ax.plot(x, y, color=palette[k % len(palette)], linewidth=1.7)
            letter = chr(ord('a') + k)
            ax.set_title(f"({letter})", fontsize=11, loc="left", fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-12, 12)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    # ---- override _check_answer to accept ordinal numeric / synonyms ----
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        p = predicted.strip().lower().rstrip(".")
        g = ground_truth.strip().lower().rstrip(".")
        # exact
        if p == g:
            return True
        # ordinal mapping: "1st" <-> "first", etc.
        if g in _ORDINAL_WORDS:
            idx = _ORDINAL_WORDS.index(g)
            # accept "1st", "2nd", ...
            if _ORDINAL_NUM[idx] in p:
                return True
            # accept bare digit like "1"
            try:
                pn = int(p.strip().rstrip(",.;)").lstrip("("))
                if pn == idx + 1:
                    return True
            except (ValueError, TypeError):
                pass
            # accept "first" wrapped in extra words ("the first one", "first subplot")
            # but require the word to appear as a whole token (not e.g. "fourth" matching "fourth" only).
            import re
            if re.search(rf"\b{g}\b", p):
                return True
            return False
        if g in _SIDE_WORDS:
            import re
            if re.search(rf"\b{g}\b", p):
                return True
            return False
        # fall back to default behaviour
        return super()._check_answer(predicted, ground_truth)
