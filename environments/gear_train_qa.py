"""
Gear Train QA environment.

Renders connected gears of different sizes (tooth counts).
Questions: rotation direction, speed ratio, output rotations.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class GearTrainQA(StandaloneVisualEnv):
    ENV_NAME = "gear_train"

    QUESTION_TYPES = [
        "last_gear_direction", "speed_ratio", "output_rotations",
        "gear_count", "largest_gear", "teeth_ratio",
        "angular_velocity_ratio", "torque_amplification",
    ]

    def _level_config(self, level: int):
        if level <= 0:
            return {"n_gears": (3, 4), "qtypes": ["gear_count", "largest_gear", "last_gear_direction"]}
        if level == 1:
            return {"n_gears": (3, 3), "qtypes": ["last_gear_direction", "largest_gear"]}
        if level == 2:
            return {"n_gears": (3, 4), "qtypes": ["last_gear_direction", "teeth_ratio"]}
        if level == 3:
            return {"n_gears": (3, 4), "qtypes": ["teeth_ratio", "speed_ratio"]}
        if level == 4:
            return {"n_gears": (3, 5), "qtypes": ["speed_ratio", "output_rotations"]}
        if level == 5:
            return {"n_gears": (4, 5), "qtypes": ["output_rotations", "angular_velocity_ratio"]}
        if level == 6:
            return {"n_gears": (4, 5), "qtypes": ["angular_velocity_ratio", "torque_amplification"]}
        if level == 7:
            return {"n_gears": (4, 6), "qtypes": ["torque_amplification", "output_rotations"]}
        if level == 8:
            return {"n_gears": (5, 6), "qtypes": ["torque_amplification"]}
        return {"n_gears": (5, 6), "qtypes": ["torque_amplification", "angular_velocity_ratio"]}

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 716)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        lo_g, hi_g = cfg["n_gears"]
        n_gears = sub_rng.randint(lo_g, hi_g)
        teeth = [sub_rng.choice([10, 12, 15, 18, 20, 24, 30, 36]) for _ in range(n_gears)]

        # Direction: first gear CW, alternating
        directions = []
        for i in range(n_gears):
            directions.append("clockwise" if i % 2 == 0 else "counterclockwise")

        # Speed: product of gear ratios
        # gear i drives gear i+1: speed_ratio = teeth[i] / teeth[i+1]
        input_rpm = rng.choice([10, 12, 15, 20, 30, 60])

        img = self._render(teeth, directions)

        if qtype == "last_gear_direction":
            q = ("The first gear (leftmost) rotates clockwise. "
                 "In which direction does the last gear rotate? "
                 "Answer clockwise or counterclockwise.")
            a = directions[-1]

        elif qtype == "speed_ratio":
            ratio = teeth[0] / teeth[-1]
            # Express as simplified fraction or decimal
            from math import gcd
            g = gcd(teeth[0], teeth[-1])
            num, den = teeth[0] // g, teeth[-1] // g
            q = (f"The first gear has {teeth[0]} teeth and the last gear has "
                 f"{teeth[-1]} teeth. What is the speed ratio (first:last) "
                 f"as a simplified fraction? Answer as num/den.")
            a = f"{num}/{den}" if den != 1 else str(num)

        elif qtype == "output_rotations":
            n_input = rng.choice([2, 3, 4, 5, 6])
            # Output rotations = input_rotations * teeth[0] / teeth[-1]
            out = n_input * teeth[0] / teeth[-1]
            if out == int(out):
                out = int(out)
                q = (f"If the first gear ({teeth[0]} teeth) makes {n_input} full rotations, "
                     f"how many full rotations does the last gear ({teeth[-1]} teeth) make?")
                a = str(out)
            else:
                q = (f"If the first gear ({teeth[0]} teeth) makes {n_input} rotations, "
                     f"how many rotations does the last gear ({teeth[-1]} teeth) make? "
                     f"Give your answer as a decimal or fraction.")
                # Simplify fraction
                from math import gcd
                num = n_input * teeth[0]
                den = teeth[-1]
                g = gcd(num, den)
                num, den = num // g, den // g
                a = f"{num}/{den}" if den != 1 else str(num)

        elif qtype == "gear_count":
            q = "How many gears are shown in the gear train?"
            a = str(n_gears)

        elif qtype == "largest_gear":
            max_t = max(teeth)
            idx = teeth.index(max_t) + 1
            q = "Which gear (by position from left, starting at 1) has the most teeth?"
            a = str(idx)

        elif qtype == "teeth_ratio":
            i = rng.randint(0, n_gears - 2)
            from math import gcd
            g = gcd(teeth[i], teeth[i + 1])
            num, den = teeth[i] // g, teeth[i + 1] // g
            q = (f"What is the teeth ratio between gear {i+1} ({teeth[i]} teeth) "
                 f"and gear {i+2} ({teeth[i+1]} teeth)? Answer as a simplified ratio a:b.")
            a = f"{num}:{den}"

        elif qtype == "angular_velocity_ratio":
            # omega_out / omega_in = teeth_first / teeth_last
            input_rpm = rng.choice([60, 120, 180, 240, 300])
            output_rpm = round(input_rpm * teeth[0] / teeth[-1], 1)
            if output_rpm == int(output_rpm):
                output_rpm = int(output_rpm)
            q = (f"The first gear ({teeth[0]} teeth) rotates at {input_rpm} RPM. "
                 f"What is the RPM of the last gear ({teeth[-1]} teeth)?")
            a = str(output_rpm)

        elif qtype == "torque_amplification":
            # Torque ratio = teeth_last / teeth_first (inverse of speed ratio)
            from math import gcd
            g = gcd(teeth[-1], teeth[0])
            num, den = teeth[-1] // g, teeth[0] // g
            q = (f"In this gear train, the input gear has {teeth[0]} teeth and "
                 f"the output gear has {teeth[-1]} teeth. "
                 f"What is the torque amplification ratio (output torque / input torque)? "
                 f"Answer as a simplified fraction or integer.")
            a = f"{num}/{den}" if den != 1 else str(num)

        else:
            return None

        return q, a, img

    def _render(self, teeth, directions):
        style = self._random_style()
        n = len(teeth)
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(6, n * 2.2) * s, 5 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        palette = style["palette"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        # Scale radii
        max_teeth = max(teeth)
        radii = [t / max_teeth * 1.5 + 0.5 for t in teeth]

        # Position gears touching
        cx = [0.0]
        for i in range(1, n):
            cx.append(cx[-1] + radii[i - 1] + radii[i] + 0.05)
        cy = 2.5

        for i in range(n):
            color = palette[i % len(palette)]
            # Draw gear body
            circle = plt.Circle((cx[i], cy), radii[i], facecolor=color,
                                edgecolor="#2c3e50", linewidth=lw + 1,
                                alpha=0.7, zorder=3)
            ax.add_patch(circle)
            # Draw teeth marks
            n_teeth_draw = min(teeth[i], 24)
            for t in range(n_teeth_draw):
                angle = 2 * math.pi * t / n_teeth_draw
                x1 = cx[i] + radii[i] * math.cos(angle)
                y1 = cy + radii[i] * math.sin(angle)
                x2 = cx[i] + (radii[i] + 0.12) * math.cos(angle)
                y2 = cy + (radii[i] + 0.12) * math.sin(angle)
                ax.plot([x1, x2], [y1, y2], color="#2c3e50", linewidth=lw, zorder=4)
            # Center dot
            ax.plot(cx[i], cy, "ko", markersize=3, zorder=5)
            # Label
            ax.text(cx[i], cy - radii[i] - 0.35, f"{teeth[i]}T",
                    ha="center", va="top", fontsize=fs, fontweight="bold",
                    fontfamily=ff)
            # Direction arrow (larger, above center, on contrasting badge)
            arrow = "CW" if directions[i] == "clockwise" else "CCW"
            ax.text(cx[i], cy, arrow, ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#2c3e50", lw=1.2, alpha=0.95),
                    zorder=7)

        pad = 1
        ax.set_xlim(cx[0] - radii[0] - pad, cx[-1] + radii[-1] + pad)
        ax.set_ylim(0, 5)
        ax.set_title("Gear Train", fontsize=fs + 4, fontweight="bold",
                      fontfamily=ff, pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
