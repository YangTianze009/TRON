"""Find missing term(s) in a sequence.

Diversity: supports arithmetic, geometric, fibonacci-like, square/cube, alternating
rules, power-n, and first-difference pattern sequences. Layout randomized
(boxes, circles, triangle cards), colors shuffled per seed, multiple question
templates. No numeric answer in the text — only "shown in the image".

Difficulty:
  L0-1:  arithmetic or geometric, 5-6 terms, single middle hidden, integer step.
  L2-4:  arithmetic + geometric + fibonacci-like, 6-7 terms, any position hidden.
  L5-7:  add square/cube sequences, power-n, 7 terms, any position.
  L8-9:  alternating operations, recurrence, negative step, 7+ terms, any
         position, tighter distractors.
"""
import random
import math
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_Q_TEMPLATES = [
    "What number replaces the '?' in the sequence shown?",
    "One entry in the sequence is hidden. What value belongs in its place?",
    "Identify the missing term represented by '?'.",
    "Fill in the missing number marked '?' in the sequence above.",
]

class SequenceInterpolationQA(StandaloneVisualEnv):
    ENV_NAME = "sequence_interpolation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "seq_types": ["arithmetic", "geometric"],
                "length": 5 + (level > 0),
                "hidden": "middle",
                "allow_neg_step": False,
                "layout_pool": ["boxes"],
                "style_pool": ["boxes"],
            }
        if level <= 4:
            return {
                "seq_types": ["arithmetic", "geometric", "fibonacci", "square"],
                "length": 6 + (level > 2),
                "hidden": "any",
                "allow_neg_step": False,
                "layout_pool": ["boxes", "circles"],
                "style_pool": ["boxes", "circles"],
            }
        if level <= 7:
            return {
                "seq_types": ["arithmetic", "geometric", "fibonacci",
                              "square", "cube", "power"],
                "length": 7,
                "hidden": "any",
                "allow_neg_step": True,
                "layout_pool": ["boxes", "circles", "cards"],
                "style_pool": ["boxes", "circles", "cards"],
            }
        # L8-9
        return {
            "seq_types": ["arithmetic", "geometric", "fibonacci",
                          "square", "cube", "power",
                          "alternating", "linear_recurrence"],
            "length": 7 + (level == 9),
            "hidden": "any",
            "allow_neg_step": True,
            "layout_pool": ["boxes", "circles", "cards"],
            "style_pool": ["boxes", "circles", "cards"],
        }

    # ------------------------------------------------------------------ #
    # Sequence builders
    # ------------------------------------------------------------------ #

    def _build_sequence(self, rng: random.Random, seq_type: str,
                        length: int, allow_neg: bool) -> List[int]:
        if seq_type == "arithmetic":
            start = rng.randint(1, 20)
            step = rng.randint(2, 9)
            if allow_neg and rng.random() < 0.35:
                step = -step
                start = rng.randint(40, 80)
            return [start + i * step for i in range(length)]
        if seq_type == "geometric":
            start = rng.randint(1, 4)
            ratio = rng.choice([2, 3])
            # keep magnitudes reasonable
            seq = [start * (ratio ** i) for i in range(length)]
            return seq
        if seq_type == "fibonacci":
            a = rng.randint(1, 6)
            b = rng.randint(a, a + 5)
            seq = [a, b]
            for _ in range(length - 2):
                seq.append(seq[-1] + seq[-2])
            return seq
        if seq_type == "square":
            start = rng.randint(1, 5)
            return [(start + i) ** 2 for i in range(length)]
        if seq_type == "cube":
            start = rng.randint(1, 4)
            return [(start + i) ** 3 for i in range(length)]
        if seq_type == "power":
            base = rng.choice([2, 3])
            start_exp = rng.randint(0, 2)
            return [base ** (start_exp + i) for i in range(length)]
        if seq_type == "alternating":
            # a, a*k, a*k + d, a*k*k + d, ... doubled alternating: +d then *k
            start = rng.randint(2, 6)
            d = rng.randint(2, 5)
            k = rng.choice([2, 3])
            seq = [start]
            for i in range(length - 1):
                if i % 2 == 0:
                    seq.append(seq[-1] + d)
                else:
                    seq.append(seq[-1] * k)
            return seq
        if seq_type == "linear_recurrence":
            # a_n = 2*a_{n-1} - a_{n-2} + c  (arithmetic-like but shaped)
            a = rng.randint(1, 6)
            b = rng.randint(a + 1, a + 8)
            c = rng.randint(-2, 2)
            seq = [a, b]
            for _ in range(length - 2):
                seq.append(2 * seq[-1] - seq[-2] + c)
            return seq
        # fallback
        return [i + 1 for i in range(length)]

    # ------------------------------------------------------------------ #

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 8801)
        style = self._random_style()

        seq_type = sub_rng.choice(cfg["seq_types"])
        length = cfg["length"]
        # A few retries to avoid crazy-large values
        for _ in range(8):
            seq = self._build_sequence(sub_rng, seq_type, length,
                                       cfg["allow_neg_step"])
            # avoid overflow in display
            if max(abs(v) for v in seq) <= 5000:
                break
        else:
            seq_type = "arithmetic"
            seq = self._build_sequence(sub_rng, seq_type, length, False)

        if cfg["hidden"] == "middle":
            hidden_idx = length // 2
        else:
            hidden_idx = sub_rng.randint(1, length - 2)

        answer = seq[hidden_idx]

        layout = sub_rng.choice(cfg["layout_pool"])
        image = self._render(sub_rng, seq, hidden_idx, style, layout, seq_type)
        question = sub_rng.choice(_Q_TEMPLATES)
        return question, str(answer), image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, rng: random.Random, seq: List[int], hidden_idx: int,
                style: Dict, layout: str, seq_type: str) -> Image.Image:
        palette = list(style["palette"])
        rng.shuffle(palette)
        length = len(seq)
        bg = style["bg_color"]
        fs_val = max(14, style["font_size_base"] + 2)

        if layout == "boxes":
            return self._render_boxes(rng, seq, hidden_idx, style, palette,
                                      fs_val, seq_type)
        if layout == "circles":
            return self._render_circles(rng, seq, hidden_idx, style, palette,
                                        fs_val, seq_type)
        # cards
        return self._render_cards(rng, seq, hidden_idx, style, palette,
                                  fs_val, seq_type)

    def _render_boxes(self, rng, seq, hidden_idx, style, palette, fs, seq_type):
        length = len(seq)
        cell_w = 1.4 + rng.uniform(-0.15, 0.2)
        gap = 0.3
        fig_w = max(8, length * (cell_w + gap) + 1)
        fig_h = 3.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        hide_color = rng.choice(["#fff3cd", "#ffe0e0", "#ffe8d6", "#e0ffe0"])
        normal_color = "#ffffff"
        edge_color = rng.choice(["#2c3e50", "#1a5276", "#4a4e69", "#5e548e"])
        arrow_color = rng.choice(["#7f8c8d", "#636e72", "#95a5a6"])
        for i, val in enumerate(seq):
            x = 0.5 + i * (cell_w + gap)
            y = 0.7
            face = hide_color if i == hidden_idx else normal_color
            rect = mpatches.FancyBboxPatch((x, y), cell_w, 1.25,
                                            facecolor=face,
                                            edgecolor=edge_color,
                                            linewidth=2.0,
                                            boxstyle="round,pad=0.08")
            ax.add_patch(rect)
            label = "?" if i == hidden_idx else str(val)
            txt_color = "red" if i == hidden_idx else "#1a1a1a"
            ax.text(x + cell_w / 2, y + 0.65, label,
                    ha="center", va="center",
                    fontsize=fs + (4 if i == hidden_idx else 0),
                    color=txt_color,
                    fontweight="bold")
            if i < length - 1:
                ax.annotate("", xy=(x + cell_w + gap - 0.02, y + 0.62),
                            xytext=(x + cell_w + 0.02, y + 0.62),
                            arrowprops=dict(arrowstyle="->",
                                            color=arrow_color, lw=1.8))
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, 2.4)
        ax.set_aspect("equal")
        ax.axis("off")
        titles = ["Find the Missing Number",
                  "Sequence Puzzle",
                  "What comes here?",
                  "Complete the Pattern"]
        ax.set_title(rng.choice(titles), fontsize=fs + 1, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_circles(self, rng, seq, hidden_idx, style, palette, fs, seq_type):
        length = len(seq)
        r = 0.7
        gap = 0.4
        fig_w = max(8, length * (2 * r + gap) + 1)
        fig, ax = plt.subplots(figsize=(fig_w, 3.2))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        edge = rng.choice(["#2c3e50", "#1a5276", "#6a040f", "#184e77"])
        for i, val in enumerate(seq):
            cx = 0.8 + i * (2 * r + gap) + r
            cy = 1.4
            face = palette[i % len(palette)] if i != hidden_idx else "#ffd6d6"
            circ = mpatches.Circle((cx, cy), r, facecolor=face,
                                   edgecolor=edge, linewidth=2.2,
                                   alpha=0.85)
            ax.add_patch(circ)
            label = "?" if i == hidden_idx else str(val)
            txt_color = "#b00020" if i == hidden_idx else "#111"
            ax.text(cx, cy, label,
                    ha="center", va="center",
                    fontsize=fs + (4 if i == hidden_idx else 0),
                    color=txt_color, fontweight="bold")
            if i < length - 1:
                ax.annotate("", xy=(cx + r + gap - 0.05, cy),
                            xytext=(cx + r + 0.05, cy),
                            arrowprops=dict(arrowstyle="->",
                                            color="#444", lw=1.8))
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, 3.0)
        ax.set_aspect("equal")
        ax.axis("off")
        titles = ["Circle Sequence", "Find the Missing Term",
                  "Ring of Numbers", "Pattern Rings"]
        ax.set_title(rng.choice(titles), fontsize=fs + 1, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_cards(self, rng, seq, hidden_idx, style, palette, fs, seq_type):
        length = len(seq)
        cell_w = 1.6
        cell_h = 1.8
        gap = 0.35
        fig_w = max(8, length * (cell_w + gap) + 1.2)
        fig, ax = plt.subplots(figsize=(fig_w, 3.3))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        edge = rng.choice(["#333", "#111", "#3a0ca3"])
        stripe_color = rng.choice(palette)
        for i, val in enumerate(seq):
            x = 0.5 + i * (cell_w + gap)
            y = 0.4
            # Card body
            face = "#fef7ff" if i != hidden_idx else "#ffe1e1"
            body = mpatches.FancyBboxPatch((x, y), cell_w, cell_h,
                                           facecolor=face,
                                           edgecolor=edge,
                                           linewidth=2.0,
                                           boxstyle="round,pad=0.06")
            ax.add_patch(body)
            # Top stripe
            stripe = mpatches.Rectangle((x + 0.08, y + cell_h - 0.35),
                                         cell_w - 0.16, 0.25,
                                         facecolor=stripe_color,
                                         edgecolor="none", alpha=0.85)
            ax.add_patch(stripe)
            # Index label
            ax.text(x + 0.2, y + cell_h - 0.22, f"n{i+1}",
                    fontsize=fs - 4, color="white", fontweight="bold")
            label = "?" if i == hidden_idx else str(val)
            txt_color = "#b00020" if i == hidden_idx else "#1a1a1a"
            ax.text(x + cell_w / 2, y + 0.75, label,
                    ha="center", va="center",
                    fontsize=fs + (4 if i == hidden_idx else 0),
                    color=txt_color, fontweight="bold")
            if i < length - 1:
                ax.annotate("", xy=(x + cell_w + gap - 0.02, y + 0.8),
                            xytext=(x + cell_w + 0.02, y + 0.8),
                            arrowprops=dict(arrowstyle="->",
                                            color="#555", lw=1.6))
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, 3.0)
        ax.set_aspect("equal")
        ax.axis("off")
        titles = ["Sequence of Cards", "Missing Card",
                  "Pattern Cards", "Which value?"]
        ax.set_title(rng.choice(titles), fontsize=fs + 1, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = SequenceInterpolationQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer if ok else 'X'}")
