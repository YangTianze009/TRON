"""
Hexagon Triple Relation QA (v4 G21b, for a puzzle benchmark color_number_hexagon).

Targets:
  - a puzzle benchmark color_number_hexagon -7
  - a puzzle benchmark polygon_sides_color -10

Task: 6 hexagons arranged in a ring; each hexagon has (color, number)
combination following a rule. One hexagon is missing one attribute.
Ask what it should be.

Reward: exact string match.

Level axes:
  A) Rule type: color from number at L0-3, color from position at L4-6, triple rule at L7+
"""
import random
import math
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_HEX_COLORS = ["red", "orange", "yellow", "green", "blue", "purple"]
_COLOR_HEX = {
    "red": "#e74c3c", "orange": "#e67e22", "yellow": "#f1c40f",
    "green": "#2ecc71", "blue": "#3498db", "purple": "#9b59b6"
}

_TEMPLATES = [
    "The figure shows 6 hexagons arranged in a ring. Each has a (color, number) pair following a hidden rule based on the position. State the rule, then give the {attr} of the hexagon at position {pos} (where '?' is). Put the {attr} in <answer>...</answer>.",
    "6 hexagons with (color, number) pairs follow a position-based rule. Position {pos} has unknown {attr}. State rule then give the {attr}. Put in <answer>...</answer>.",
    "Ring of 6 hexagons; each's (color, number) is fixed by position. State the rule, then give the missing {attr} at position {pos}. Put in <answer>...</answer>.",
    "6 hexagons in a ring follow a (color, number)-by-position rule. Position {pos}'s {attr} is missing. Derive the rule, then the {attr}. Put in <answer>...</answer>.",
    "The hexagon ring obeys a position→(color, number) rule. Position {pos} missing {attr}. State rule then give {attr}. Put in <answer>...</answer>.",
    "Hexagon ring (color, number) rule based on position. At position {pos}, {attr} is unknown. Rule first, then answer. Put in <answer>...</answer>.",
    "6 hexagons form a ring with position-determined (color, number). Missing {attr} at position {pos}? State rule, then give {attr}. Put in <answer>...</answer>.",
    "From the ring of 6 hexagons, derive the rule linking position to (color, number). Position {pos}'s {attr} missing. Rule + answer. Put in <answer>...</answer>.",
    "Position-based rule for (color, number) in the hexagon ring. Missing {attr} at pos {pos}. State rule, give answer. Put in <answer>...</answer>.",
    "6 hexagons around a center. (color, number) follows position rule. Missing {attr} at position {pos}. Rule + answer. Put in <answer>...</answer>.",
    "The figure: 6 hexagons in a ring with (color, number) per position. Position {pos}'s {attr} unknown. Rule and answer. Put in <answer>...</answer>.",
    "Given 6 hexagons in a ring with position-determined attributes, find the missing {attr} at position {pos}. State rule first. Put in <answer>...</answer>.",
    "Ring of hexagons with (color, number) per position. Position {pos} missing {attr}. Derive rule, give {attr}. Put in <answer>...</answer>.",
    "State the rule linking hexagon position to (color, number), then give the {attr} at position {pos}. Put in <answer>...</answer>.",
    "The {attr} at position {pos} of the hexagon ring is missing. State the position-based rule, then give the answer. Put in <answer>...</answer>.",
    "Hexagon ring puzzle: derive rule, give missing {attr} at position {pos}. Put in <answer>...</answer>.",
]

class HexagonTripleRelationQA(StandaloneVisualEnv):
    ENV_NAME = "hexagon_triple_relation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            rule = "color_by_position"
        elif level <= 6:
            rule = "color_cycle"
        else:
            rule = "color_and_number"
        return {"rule": rule}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 683)
        self._primary_complexity_feature = level

        # Each of 6 positions gets (color, number)
        if cfg["rule"] == "color_by_position":
            # color[i] = _HEX_COLORS[i]
            # number[i] = i+1
            colors = list(_HEX_COLORS)
            numbers = [i + 1 for i in range(6)]
        elif cfg["rule"] == "color_cycle":
            # Shift colors by a random offset
            offset = rng.randint(0, 5)
            colors = [_HEX_COLORS[(i + offset) % 6] for i in range(6)]
            numbers = [i + 1 for i in range(6)]
        else:  # color_and_number
            # color = alternating 2 colors, number = i mod 3
            colors = [_HEX_COLORS[i % 3] for i in range(6)]
            numbers = [(i % 3) * 2 + 1 for i in range(6)]

        # Pick missing position and attribute
        miss_pos = rng.randint(0, 5)
        miss_attr = rng.choice(["color", "number"])

        answer = colors[miss_pos] if miss_attr == "color" else str(numbers[miss_pos])

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(attr=miss_attr, pos=miss_pos + 1)

        img = self._render(colors, numbers, miss_pos, miss_attr, rng)
        return q, answer, img

    def _render(self, colors, numbers, miss_pos, miss_attr, rng):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.axis("off")

        R = 2.0
        for i in range(6):
            theta = math.pi / 2 - 2 * math.pi * i / 6
            cx = R * math.cos(theta)
            cy = R * math.sin(theta)
            # Draw hexagon
            hex_points = []
            for k in range(6):
                angle = k * math.pi / 3
                hex_points.append((cx + 0.5 * math.cos(angle),
                                    cy + 0.5 * math.sin(angle)))
            if miss_pos == i and miss_attr == "color":
                fc = "#cccccc"
            else:
                fc = _COLOR_HEX[colors[i]]
            ax.add_patch(mpatches.Polygon(hex_points, fc=fc, ec="black", lw=1.3))
            # Number inside
            if miss_pos == i and miss_attr == "number":
                ax.text(cx, cy, "?", fontsize=16, ha="center", va="center",
                        fontweight="bold", color="red")
            else:
                ax.text(cx, cy, str(numbers[i]), fontsize=14, ha="center",
                        va="center", fontweight="bold", color="white")
            # Position label outside
            ax.text(cx * 1.4, cy * 1.4, f"pos{i + 1}", fontsize=9, ha="center",
                    va="center")

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_htr"
    os.makedirs(out_dir, exist_ok=True)
    env = HexagonTripleRelationQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 89
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[htr L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/htr_s{s}_L{level}.png")
            print(f"[htr L{level} s{s}] A={env._answer}")
