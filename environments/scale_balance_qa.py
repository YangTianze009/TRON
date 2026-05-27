"""
Scale Balance QA environment.

Renders a balance scale with objects of different weights.
Questions: which side heavier, what weight balances, effect of adding weight.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS_SB = [
    "Balance Scale", "Weighing Scale", "Lever Balance",
    "Two-Pan Scale", "Scale Reading",
]
_QUESTION_PHRASES_SB = {
    "which_heavier": [
        "Which side of the scale is heavier? Answer left, right, or balanced.",
        "Looking at the scale shown, which pan holds more total weight? Answer left, right, or balanced.",
        "Compare the two pans of the scale. Which is heavier? Answer with left, right, or balanced.",
    ],
    "total_weight": [
        "What is the total weight on both sides combined?",
        "Sum the weights on both pans of the scale. What is the total in kg?",
        "Add up every weight shown across both pans. Output the total (kg).",
    ],
    "difference": [
        "What is the weight difference between the two sides?",
        "How many kg heavier is one pan compared to the other?",
        "Compute the absolute weight difference between left and right pans (kg).",
    ],
}

class ScaleBalanceQA(StandaloneVisualEnv):
    ENV_NAME = "scale_balance"

    QUESTION_TYPES = [
        "which_heavier", "weight_to_balance", "add_effect",
        "total_weight", "difference", "remove_to_balance",
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["which_heavier", "total_weight"],
                    "qweights": [5, 5], "n_items": (1, 2), "w_range": (1, 10)}
        if level <= 2:
            return {"qtypes": ["which_heavier", "total_weight", "difference"],
                    "qweights": [3, 3, 4], "n_items": (1, 2), "w_range": (1, 12)}
        if level <= 4:
            return {"qtypes": ["weight_to_balance", "difference", "total_weight"],
                    "qweights": [4, 3, 3], "n_items": (1, 3), "w_range": (1, 15)}
        if level <= 6:
            return {"qtypes": ["weight_to_balance", "add_effect", "remove_to_balance"],
                    "qweights": [3, 4, 3], "n_items": (2, 3), "w_range": (1, 15)}
        if level <= 8:
            return {"qtypes": ["add_effect", "remove_to_balance", "weight_to_balance"],
                    "qweights": [4, 3, 3], "n_items": (2, 4), "w_range": (2, 20)}
        return {"qtypes": ["remove_to_balance", "add_effect"],
                "qweights": [5, 5], "n_items": (2, 4), "w_range": (3, 25)}

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 5501)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        # Generate items with weights
        n_left = sub_rng.randint(*cfg["n_items"])
        n_right = sub_rng.randint(*cfg["n_items"])
        left_weights = [sub_rng.randint(*cfg["w_range"]) for _ in range(n_left)]
        right_weights = [sub_rng.randint(*cfg["w_range"]) for _ in range(n_right)]
        left_total = sum(left_weights)
        right_total = sum(right_weights)

        left_labels = [f"{w}kg" for w in left_weights]
        right_labels = [f"{w}kg" for w in right_weights]

        # Pick a random style/title/shape per seed for diversity
        item_shape = sub_rng.choice(["round", "square", "hex", "triangle"])
        title_str = sub_rng.choice(_TITLE_VARIANTS_SB)
        # Random ground/beam color theme
        beam_color = sub_rng.choice(["#5C3317", "#374151", "#1f2937", "#4a235a", "#0d4f6c"])
        img = self._render(left_labels, right_labels, left_total, right_total,
                           item_shape=item_shape, title_str=title_str,
                           beam_color=beam_color, sub_rng=sub_rng)

        if qtype == "which_heavier":
            if left_total > right_total:
                a = "left"
            elif right_total > left_total:
                a = "right"
            else:
                a = "balanced"
            q = sub_rng.choice(_QUESTION_PHRASES_SB["which_heavier"])

        elif qtype == "weight_to_balance":
            diff = abs(left_total - right_total)
            lighter = "right" if left_total > right_total else "left"
            if left_total == right_total:
                q = "The scale is balanced. How much weight must be added to either side to tip it?"
                a = "0"
            else:
                q = (f"How many kg must be added to the {lighter} side "
                     f"to balance the scale?")
                a = str(diff)

        elif qtype == "add_effect":
            # Use seed-deterministic sub_rng (not self._rng) so that the
            # question is reproducible per (seed, level).
            add_kg = sub_rng.randint(1, 10)
            side = sub_rng.choice(["left", "right"])
            new_left = left_total + (add_kg if side == "left" else 0)
            new_right = right_total + (add_kg if side == "right" else 0)
            if new_left > new_right:
                result = "left"
            elif new_right > new_left:
                result = "right"
            else:
                result = "balanced"
            q = (f"If you add {add_kg}kg to the {side} side, "
                 f"which side will be heavier? Answer left, right, or balanced.")
            a = result

        elif qtype == "total_weight":
            q = sub_rng.choice(_QUESTION_PHRASES_SB["total_weight"])
            a = str(left_total + right_total)

        elif qtype == "difference":
            q = sub_rng.choice(_QUESTION_PHRASES_SB["difference"])
            a = str(abs(left_total - right_total))

        elif qtype == "remove_to_balance":
            if left_total == right_total:
                q = "The scale is balanced. What weight must be removed from either side to unbalance it?"
                a = "any"
                # re-generate to get unbalanced
                return None
            heavier = "left" if left_total > right_total else "right"
            diff = abs(left_total - right_total)
            q = (f"How many kg must be removed from the {heavier} side "
                 f"to balance the scale?")
            a = str(diff)
        else:
            return None

        if level <= 2:
            q += (
                " Hint: read the weight labels on each item. Sum the "
                "weights on each side of the scale, then apply the formula "
                "for the question type (compare totals, or take the "
                "difference)."
            )
        return q, a, img

    def _render(self, left_labels, right_labels, left_total, right_total,
                item_shape="round", title_str="Balance Scale",
                beam_color="#5C3317", sub_rng=None):
        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * s, 6 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        palette = list(style["palette"])
        if sub_rng is not None:
            sub_rng.shuffle(palette)
        fs = style["font_size_base"]
        ff = style["font_family"]
        lw = style["line_width"]

        # Base / fulcrum
        base_x, base_y = 5.0, 1.0
        tri = Polygon([(base_x - 0.5, base_y), (base_x + 0.5, base_y),
                        (base_x, base_y + 0.8)], facecolor=beam_color,
                       edgecolor="#222", linewidth=lw + 1)
        ax.add_patch(tri)
        # Ground
        ax.plot([2, 8], [base_y, base_y], color=beam_color, linewidth=lw + 2)

        # Tilt angle based on weight difference
        max_tilt = 0.15
        diff = left_total - right_total
        max_weight = max(left_total, right_total, 1)
        tilt = -(diff / max_weight) * max_tilt  # negative = left heavy tilts left down

        # Beam
        beam_cx, beam_cy = base_x, base_y + 0.8
        beam_len = 3.5
        lx = beam_cx - beam_len * math.cos(tilt)
        ly = beam_cy - beam_len * math.sin(tilt)
        rx = beam_cx + beam_len * math.cos(tilt)
        ry = beam_cy + beam_len * math.sin(tilt)
        ax.plot([lx, rx], [ly, ry], color=beam_color, linewidth=lw + 3, zorder=3)

        # Pans (hanging from beam ends)
        pan_drop = 0.8
        for side, px, py, labels in [("left", lx, ly, left_labels),
                                      ("right", rx, ry, right_labels)]:
            # Strings
            ax.plot([px, px - 0.5], [py, py - pan_drop], color="#666", linewidth=1)
            ax.plot([px, px + 0.5], [py, py - pan_drop], color="#666", linewidth=1)
            # Pan
            pan_y = py - pan_drop
            pan_rect = mpatches.FancyBboxPatch(
                (px - 0.7, pan_y - 0.15), 1.4, 0.15,
                boxstyle="round,pad=0.02", facecolor="#d4a574",
                edgecolor="#8B4513", linewidth=lw)
            ax.add_patch(pan_rect)
            # Items
            for i, label in enumerate(labels):
                bx = px - 0.3 + i * 0.4
                by = pan_y + 0.1 + i * 0.35
                color = palette[i % len(palette)]
                if item_shape == "round":
                    item_patch = mpatches.Circle(
                        (bx, by + 0.15), 0.18,
                        facecolor=color, edgecolor="#222",
                        linewidth=1, zorder=5)
                elif item_shape == "hex":
                    item_patch = mpatches.RegularPolygon(
                        (bx, by + 0.15), 6, radius=0.2,
                        facecolor=color, edgecolor="#222",
                        linewidth=1, zorder=5)
                elif item_shape == "triangle":
                    item_patch = mpatches.RegularPolygon(
                        (bx, by + 0.15), 3, radius=0.2,
                        facecolor=color, edgecolor="#222",
                        linewidth=1, zorder=5)
                else:
                    item_patch = mpatches.FancyBboxPatch(
                        (bx - 0.18, by), 0.36, 0.3,
                        boxstyle="round,pad=0.02", facecolor=color,
                        edgecolor="#222", linewidth=1, zorder=5)
                ax.add_patch(item_patch)
                ax.text(bx, by + 0.15, label, ha="center", va="center",
                        fontsize=fs - 1, fontweight="bold", color="white",
                        fontfamily=ff, zorder=6)

        # Side labels
        ax.text(lx, ly + 0.5, "Left", ha="center", fontsize=fs, fontfamily=ff,
                fontweight="bold")
        ax.text(rx, ry + 0.5, "Right", ha="center", fontsize=fs, fontfamily=ff,
                fontweight="bold")

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5.5)
        ax.set_title(title_str, fontsize=fs + 4, fontweight="bold",
                      fontfamily=ff, pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
