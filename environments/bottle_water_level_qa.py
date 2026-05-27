"""
Bottle Water Level QA.

A cylindrical bottle of total height H_bottle (cm) holds water up to height
H_water (cm) when standing upright. The figure shows the upright bottle on
the left and the same bottle inverted (flipped upside down) on the right.
After flipping, the SAME volume of water sits at the bottom of the inverted
bottle while an air gap occupies the top. Since the bottle is uniform-radius
(cylindrical), the air gap height in either orientation equals
`H_bottle - H_water`.

The question asks for one of:
  - the air-gap height in cm (always = H_bottle - H_water);
  - the water height in cm in the flipped state (this equals H_water for a
    cylinder — the SAME water occupies the same vertical extent because
    cross-section is constant).
  - or the upright air-gap (warm-up at L0).

Difficulty axes:
  - L0..L2: integer measurements, simple "what's left over" question.
  - L3..L5: ask about the flipped configuration.
  - L6..L9: ask about float-cm levels and require careful reading of the
    labeled measurements.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_AIR_GAP = [
    "The figure shows a uniform cylindrical bottle. Compute the AIR GAP height (in cm), i.e. the empty space above the water when the bottle is standing upright. Place the numeric answer in <answer>...</answer>.",
    "From the labeled measurements, find the height (in cm) of the empty air gap above the water in the upright bottle. Place the value in <answer>...</answer>.",
    "Determine the air-gap height (cm) in the upright cylindrical bottle. Place the numeric answer in <answer>...</answer>.",
    "How many cm of empty space sit above the water in the upright bottle shown? Place the numeric answer in <answer>...</answer>.",
]

_TEMPLATES_FLIPPED_AIR = [
    "The figure shows the bottle upright on the left and inverted on the right. After flipping, what is the height (in cm) of the air gap (now BELOW the water)? Place the numeric answer in <answer>...</answer>.",
    "After flipping the cylindrical bottle upside down, the water sits at the top and an air gap forms below it. Compute that air-gap height in cm. Place the value in <answer>...</answer>.",
    "When the bottle is flipped, find the air-gap height (cm) below the water. Place the numeric answer in <answer>...</answer>.",
    "Compute the height (cm) of the empty space below the water after the bottle is inverted. Place the answer in <answer>...</answer>.",
    "The bottle is inverted in the right panel. What is the height (cm) of the air gap underneath the water? Place the value in <answer>...</answer>.",
]

_TEMPLATES_FLIPPED_WATER = [
    "The figure shows the bottle upright on the left and inverted on the right. After flipping, what is the vertical extent (height in cm) of the water column? Place the numeric answer in <answer>...</answer>.",
    "When the cylindrical bottle is inverted, the water sits at the top. Compute the height (cm) of the water column in the flipped bottle. Place the value in <answer>...</answer>.",
    "After flipping the bottle upside down, find the height (cm) of the water column. Place the numeric answer in <answer>...</answer>.",
    "Compute the height (cm) the water occupies in the inverted bottle. Place the answer in <answer>...</answer>.",
]

_TEMPLATES_UPRIGHT_WATER = [
    "Read the upright cylindrical bottle: what is the height (cm) of the water column? Place the numeric answer in <answer>...</answer>.",
    "From the labels, give the water-column height (cm) in the upright bottle shown. Place the value in <answer>...</answer>.",
    "Determine the water height (cm) in the upright bottle. Place the answer in <answer>...</answer>.",
    "How tall (cm) is the water column in the upright bottle in the figure? Place the answer in <answer>...</answer>.",
]


class BottleWaterLevelQA(StandaloneVisualEnv):
    ENV_NAME = "bottle_water_level"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "h_bottle_pool": [10, 12, 15, 20],
                "frac_pool": [0.3, 0.4, 0.5, 0.6, 0.7],
                "ask_modes": ["upright_air", "upright_water"],
                "allow_float": False,
            }
        if level <= 5:
            return {
                "h_bottle_pool": [12, 15, 16, 18, 20, 24, 25],
                "frac_pool": [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.75],
                "ask_modes": ["flipped_air", "flipped_water", "upright_air"],
                "allow_float": False,
            }
        return {
            "h_bottle_pool": [12.5, 14, 15, 16.5, 18, 20, 22.5, 25, 28],
            "frac_pool": [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8],
            "ask_modes": ["flipped_air", "flipped_water"],
            "allow_float": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1097 + level * 109 + 19)

        for _ in range(20):
            H_bottle = rng.choice(cfg["h_bottle_pool"])
            frac = rng.choice(cfg["frac_pool"])
            H_water = H_bottle * frac
            if cfg["allow_float"]:
                H_water = round(H_water, 1)
            else:
                H_water = round(H_water)
            H_air = H_bottle - H_water
            if H_water <= 0 or H_air <= 0:
                continue
            break
        else:
            return None

        ask_mode = rng.choice(cfg["ask_modes"])
        if ask_mode == "upright_air":
            answer_val = H_air
            templates = _TEMPLATES_AIR_GAP
            show_flipped = False
        elif ask_mode == "upright_water":
            answer_val = H_water
            templates = _TEMPLATES_UPRIGHT_WATER
            show_flipped = False
        elif ask_mode == "flipped_air":
            answer_val = H_air
            templates = _TEMPLATES_FLIPPED_AIR
            show_flipped = True
        else:  # flipped_water
            answer_val = H_water
            templates = _TEMPLATES_FLIPPED_WATER
            show_flipped = True

        # Format answer compactly
        if abs(answer_val - round(answer_val)) < 1e-9:
            answer = str(int(round(answer_val)))
        else:
            answer = f"{answer_val:g}"

        sidx = (self.seed or 0) % len(templates)
        question = templates[sidx]
        img = self._render(H_bottle, H_water, H_air, show_flipped)
        return question, answer, img

    # ---------------- render ---------------- #
    def _render(self, H_bottle, H_water, H_air, show_flipped) -> Image.Image:
        if show_flipped:
            fig, axes = plt.subplots(1, 2, figsize=(8, 6.5), dpi=120)
        else:
            fig, ax_only = plt.subplots(1, 1, figsize=(4.5, 6), dpi=120)
            axes = [ax_only]
        fig.patch.set_facecolor("#ffffff")

        # Draw upright bottle on first axis
        self._draw_one(axes[0], H_bottle, H_water, H_air, inverted=False,
                       title="Upright")
        if show_flipped:
            self._draw_one(axes[1], H_bottle, H_water, H_air, inverted=True,
                           title="Inverted (flipped)")

        try:
            fig.tight_layout()
        except Exception:
            pass

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _draw_one(self, ax, H_bottle, H_water, H_air, inverted, title):
        ax.set_facecolor("#ffffff")
        # Bottle drawn as a rectangle from y=0 to y=H_bottle, width=2.
        bottle_w = 2.0
        body_x = -bottle_w / 2
        # Outline
        bottle_rect = patches.Rectangle((body_x, 0), bottle_w, H_bottle,
                                         linewidth=2.0, edgecolor="#2c3e50",
                                         facecolor="none", zorder=3)
        ax.add_patch(bottle_rect)

        # Water region: in upright orientation, sits from y=0 to y=H_water;
        # in flipped (inverted) orientation, the bottle is conceptually
        # flipped so the water occupies the TOP of the visual rectangle:
        # y in [H_bottle - H_water, H_bottle].
        if inverted:
            water_y_lo = H_bottle - H_water
            water_y_hi = H_bottle
        else:
            water_y_lo = 0
            water_y_hi = H_water
        water_rect = patches.Rectangle((body_x, water_y_lo), bottle_w,
                                       water_y_hi - water_y_lo,
                                       linewidth=0, facecolor="#5dade2",
                                       alpha=0.55, zorder=2)
        ax.add_patch(water_rect)

        # Add measurement annotations to the right side
        # Total bottle height arrow on far right
        arrow_x = bottle_w / 2 + 1.2
        ax.annotate("", xy=(arrow_x, H_bottle), xytext=(arrow_x, 0),
                    arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
        ax.text(arrow_x + 0.25, H_bottle / 2, f"H = {H_bottle:g} cm",
                ha="left", va="center", fontsize=11, color="#2c3e50",
                fontweight="bold")

        # Water height arrow on the immediate right of bottle
        if inverted:
            wlo, whi = water_y_lo, water_y_hi
        else:
            wlo, whi = 0, H_water
        ax.annotate("", xy=(bottle_w / 2 + 0.3, whi),
                    xytext=(bottle_w / 2 + 0.3, wlo),
                    arrowprops=dict(arrowstyle="<->", color="#1b4f72", lw=1.4))
        wlabel = f"water = {H_water:g} cm"
        ax.text(bottle_w / 2 + 0.45, (wlo + whi) / 2, wlabel,
                ha="left", va="center", fontsize=10, color="#1b4f72",
                fontweight="bold")

        # If inverted, draw the upside-down indicator (small arrow / label).
        if inverted:
            ax.annotate("", xy=(0, -0.6), xytext=(0, 0.4),
                        arrowprops=dict(arrowstyle="->", color="#7b241c",
                                        lw=2.0))
            ax.text(0, -1.2, "(bottle inverted: cap at top originally,"
                              " now at bottom)",
                    ha="center", va="top", fontsize=8, color="#7b241c",
                    fontstyle="italic")

        ax.set_xlim(-3.5, 5.5)
        ax.set_ylim(-2.5, H_bottle + 1.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
