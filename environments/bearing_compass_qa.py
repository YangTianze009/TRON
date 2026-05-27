"""
Bearing / Compass Direction QA.

A compass overlay (N at top, E right, S bottom, W left) is drawn with an
arrow from origin O pointing at one of:
  - a cardinal direction (N / S / E / W)
  - an intercardinal direction (NE / SE / SW / NW)
  - a "north-by-east X°" or "south-by-west X°" bearing (the arrow points
    at angle X° from the named cardinal axis, toward the named secondary
    cardinal axis).

Question modes (each problem has exactly ONE answer type, not both):
  - "deg_from_north": ask for the angle measured CLOCKWISE from north
    (standard navigational bearing). Integer 0..360. Answer = integer.
  - "direction_word": the diagram shows the arrow at one of the 8 cardinal
    directions; ask for the direction name. Answer = single word like
    "north", "northeast", "south", etc.
  - "bearing_decompose": diagram shows an arrow with bearing form
    "N X° E" labeled; ask for the angle from north (clockwise). Answer =
    integer.

Difficulty axes:
  - L0..L2: pure cardinal/intercardinal directions, "direction_word" mode
    only. Answer is a word from the 8-direction set.
  - L3..L5: bearings with X in {15, 30, 45, 60, 75}; "deg_from_north" or
    "bearing_decompose" modes.
  - L6..L9: arbitrary X angles 5..85 degrees; "deg_from_north".
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Direction-word mode templates (answer is a single word).
_TEMPLATES_WORD = [
    "The compass diagram shows an arrow from O. Identify the direction the arrow points (one of: north, south, east, west, northeast, northwest, southeast, southwest). Place the single direction word in <answer>...</answer>.",
    "From the compass diagram, what direction does the arrow from origin O point? Use one word: north, south, east, west, northeast, northwest, southeast, or southwest. Place it in <answer>...</answer>.",
    "Read the compass overlay. The arrow indicates which compass direction? Answer with one word (e.g. north, northeast). Place it in <answer>...</answer>.",
    "Which compass direction does the arrow point in? Pick from: north, south, east, west, northeast, northwest, southeast, southwest. Place the word in <answer>...</answer>.",
    "Identify the direction of the arrow shown on the compass. Place the single direction word in <answer>...</answer>.",
    "The figure shows a compass with an arrow. State the direction the arrow points (one word: cardinal or intercardinal). Place the answer in <answer>...</answer>.",
    "Looking at the compass diagram, in which direction does the arrow go? Use a single direction word. Place it in <answer>...</answer>.",
    "Compass arrow direction = ? Choose from: north, south, east, west, northeast, northwest, southeast, southwest. Place the word in <answer>...</answer>.",
    "From the compass, what is the direction of the arrow? Single word answer. Place it in <answer>...</answer>.",
]

# Bearing-decomposition mode (label "N X° E", answer is integer X).
_TEMPLATES_DECOMPOSE = [
    "The figure labels the arrow's direction in bearing form '{prefix}'. What angle (in degrees) does the arrow make with the north axis (measured clockwise)? Place the integer in <answer>...</answer>.",
    "An arrow's direction is labeled as '{prefix}' on the compass. Compute the bearing measured clockwise from the north axis (in degrees). Place the integer in <answer>...</answer>.",
    "From the bearing label '{prefix}', determine the bearing (in degrees) measured clockwise from north. Place the integer answer in <answer>...</answer>.",
    "Bearing labeled '{prefix}'. Convert to a single bearing measured clockwise from north (in degrees). Place the integer in <answer>...</answer>.",
    "The figure labels the arrow as '{prefix}'. What is the equivalent bearing from north (clockwise, degrees)? Place the integer in <answer>...</answer>.",
    "Given the bearing notation '{prefix}', find the angle in degrees from north measured clockwise. Place the integer answer in <answer>...</answer>.",
]

# Bearing-from-figure mode: arrow drawn at angle, no label. Answer = integer (degrees from north, clockwise).
_TEMPLATES_DEG = [
    "The compass diagram shows an arrow from O. Compute the bearing (in degrees, measured clockwise from north). Place the integer in <answer>...</answer>.",
    "From the compass figure, find the angle (degrees) the arrow makes with the north axis, measured clockwise. Place the integer answer in <answer>...</answer>.",
    "Using the compass overlay, compute the bearing of the arrow. The bearing is measured clockwise from north, in degrees. Place the integer in <answer>...</answer>.",
    "Determine the arrow's bearing (clockwise from north, in degrees). Place the integer answer in <answer>...</answer>.",
    "The arrow on the compass points at an angle. Express it as a bearing (clockwise from north, degrees). Place the integer in <answer>...</answer>.",
    "Compute the angle (degrees, clockwise from north) the arrow makes. Place the integer answer in <answer>...</answer>.",
]


_DIRECTION_NAMES = {
    0: "north",
    45: "northeast",
    90: "east",
    135: "southeast",
    180: "south",
    225: "southwest",
    270: "west",
    315: "northwest",
}


class BearingCompassQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "bearing_compass"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "mode": "word",
                "bearings": list(_DIRECTION_NAMES.keys()),
            }
        if level <= 5:
            return {
                "mode": "decompose_or_deg",
                "off_axis_angles": [15, 30, 45, 60, 75],
            }
        return {
            "mode": "deg",
            "off_axis_angles": [5, 10, 18, 22, 27, 33, 37, 43, 47, 52,
                                57, 62, 67, 72, 78, 82],
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1487 + level * 103 + 31)

        if cfg["mode"] == "word":
            bearing = rng.choice(cfg["bearings"])
            direction_name = _DIRECTION_NAMES[bearing]
            answer = direction_name
            templates = _TEMPLATES_WORD
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx]
            label_text = None
            img = self._render(bearing, label_text=None,
                               highlight_bearing=False)
            return question, answer, img

        if cfg["mode"] == "decompose_or_deg":
            sub_mode = rng.choice(["decompose", "deg"])
            off = rng.choice(cfg["off_axis_angles"])
            quadrant = rng.choice(["NE", "NW", "SE", "SW"])
        else:
            sub_mode = "deg"
            off = rng.choice(cfg["off_axis_angles"])
            quadrant = rng.choice(["NE", "NW", "SE", "SW"])

        # Convert quadrant + off-angle to bearing in [0, 360).
        if quadrant == "NE":
            bearing = off
            prefix_word = f"N {off}° E"
        elif quadrant == "SE":
            bearing = 180 - off
            prefix_word = f"S {off}° E"
        elif quadrant == "SW":
            bearing = 180 + off
            prefix_word = f"S {off}° W"
        else:  # NW
            bearing = 360 - off
            prefix_word = f"N {off}° W"

        if sub_mode == "decompose":
            templates = _TEMPLATES_DECOMPOSE
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx].format(prefix=prefix_word)
            label_text = prefix_word
        else:
            templates = _TEMPLATES_DEG
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx]
            label_text = None

        answer = str(int(bearing))
        img = self._render(bearing, label_text=label_text,
                           highlight_bearing=(sub_mode == "deg"))
        return question, answer, img

    def _render(self, bearing_deg, label_text, highlight_bearing) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Compass: N up, E right, S down, W left.
        # bearing_deg is measured clockwise from north. Convert to math
        # angle (CCW from +x): math = 90 - bearing.
        compass_color = "#34495e"
        rose_radius = 1.0
        # Outer circle
        circ = patches.Circle((0, 0), rose_radius, fill=False,
                              edgecolor=compass_color, lw=1.6)
        ax.add_patch(circ)
        # Cardinal cross
        ax.plot([-rose_radius, rose_radius], [0, 0], color=compass_color,
                lw=0.8, alpha=0.6)
        ax.plot([0, 0], [-rose_radius, rose_radius], color=compass_color,
                lw=0.8, alpha=0.6)
        # Direction labels
        ax.text(0, rose_radius + 0.12, "N", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#c0392b")
        ax.text(0, -rose_radius - 0.12, "S", ha="center", va="top",
                fontsize=14, fontweight="bold", color=compass_color)
        ax.text(rose_radius + 0.12, 0, "E", ha="left", va="center",
                fontsize=14, fontweight="bold", color=compass_color)
        ax.text(-rose_radius - 0.12, 0, "W", ha="right", va="center",
                fontsize=14, fontweight="bold", color=compass_color)

        # Origin
        ax.plot([0], [0], 'ko', markersize=6)
        ax.text(0.07, -0.12, "O", ha="left", va="top", fontsize=12,
                fontweight="bold", color=compass_color)

        # Arrow from origin in the bearing direction.
        math_ang = math.radians(90 - bearing_deg)
        arrow_len = 1.5
        ax_x = arrow_len * math.cos(math_ang)
        ax_y = arrow_len * math.sin(math_ang)
        ax.annotate("", xy=(ax_x, ax_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=2.5))
        # Endpoint label
        ax.text(ax_x * 1.08, ax_y * 1.08, "P", ha="center", va="center",
                fontsize=12, fontweight="bold", color="#1b4f72")

        # If label_text set (decompose mode), put it next to the arrow.
        if label_text:
            label_x = ax_x * 1.05 + (0.25 if ax_x >= 0 else -0.25)
            label_y = ax_y * 1.05 + 0.18
            ax.text(label_x, label_y, label_text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#7b241c",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#fff8f0", edgecolor="#7b241c",
                              lw=0.9))

        # If highlight_bearing, draw the clockwise-from-north arc and a
        # small "?" label.
        if highlight_bearing:
            # Arc from north (90 in math angles = top) sweeping CLOCKWISE
            # by `bearing_deg` to the arrow.
            # In matplotlib Arc, theta1/theta2 are CCW. From +x axis, north
            # is at 90 deg, the arrow is at (90 - bearing_deg) deg.
            # CCW sweep from arrow up to north covers (bearing_deg) degrees,
            # so theta1 = 90 - bearing_deg, theta2 = 90.
            arc = patches.Arc((0, 0), 0.8, 0.8, angle=0,
                              theta1=90 - bearing_deg, theta2=90,
                              color="#7b241c", lw=2.0)
            ax.add_patch(arc)
            mid_ang_math = math.radians(90 - bearing_deg / 2)
            ax.text(0.55 * math.cos(mid_ang_math),
                    0.55 * math.sin(mid_ang_math),
                    "?",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="#7b241c",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="#fff", edgecolor="#7b241c", lw=0.8))

        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
