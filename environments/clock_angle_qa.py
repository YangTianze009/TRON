"""
Clock angle QA -- analog clock showing a time.
Questions: angle between hands, time shown, angle after N minutes, overlap count.
"""
import math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class ClockAngleQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "clock_angle"

    def _hand_angle(self, h, m):
        """Return (hour_deg, minute_deg) measured clockwise from 12."""
        min_deg = 6 * m
        hour_deg = 30 * (h % 12) + 0.5 * m
        return hour_deg, min_deg

    def _angle_between(self, h, m):
        hd, md = self._hand_angle(h, m)
        diff = abs(hd - md)
        return min(diff, 360 - diff)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"qtypes": ["read_time", "minute_hand_angle"]}
        if level <= 5:
            return {"qtypes": ["read_time", "angle_between",
                               "minute_hand_angle"]}
        if level <= 7:
            return {"qtypes": ["angle_between", "angle_after_n"]}
        return {"qtypes": ["angle_after_n", "overlap_count"]}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        style = self._random_style()
        qtype = parameter.get("question_type", rng.choice(cfg["qtypes"]))

        h = rng.randint(1, 12)
        m = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

        hd, md = self._hand_angle(h, m)

        # Draw clock
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(5 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        clock_face = plt.Circle((0, 0), 1.05, fc="white", ec=style["geo_line_color"],
                                linewidth=style["line_width"] + 1)
        ax.add_patch(clock_face)

        # Hour markers
        for i in range(1, 13):
            ang = math.radians(90 - 30 * i)
            ax.text(0.85 * math.cos(ang), 0.85 * math.sin(ang), str(i),
                    ha="center", va="center", fontsize=style["font_size_base"],
                    fontweight="bold", fontfamily=style["font_family"])
            # tick
            ax.plot([0.95 * math.cos(ang), 1.0 * math.cos(ang)],
                    [0.95 * math.sin(ang), 1.0 * math.sin(ang)],
                    color=style["geo_line_color"], linewidth=1.5)

        # Hour hand
        h_ang = math.radians(90 - hd)
        ax.plot([0, 0.5 * math.cos(h_ang)], [0, 0.5 * math.sin(h_ang)],
                color=style["palette"][0], linewidth=style["line_width"] + 2, solid_capstyle="round")

        # Minute hand
        m_ang = math.radians(90 - md)
        ax.plot([0, 0.75 * math.cos(m_ang)], [0, 0.75 * math.sin(m_ang)],
                color=style["palette"][1], linewidth=style["line_width"] + 0.5, solid_capstyle="round")

        ax.plot(0, 0, "o", color=style["geo_line_color"], markersize=5, zorder=5)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Analog Clock", fontsize=style["font_size_base"] + 2, fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        time_str = f"{h}:{m:02d}"
        angle = round(self._angle_between(h, m), 1)

        if qtype == "angle_between":
            q = ("For the time shown on the clock, what is the angle (in "
                 "degrees) between the hour and minute hands?")
            return q, str(angle), img
        elif qtype == "read_time":
            q = "What time is shown on the clock? Answer in H:MM format."
            return q, time_str, img
        elif qtype == "angle_after_n":
            dm = rng.choice([15, 30, 45, 60])
            new_m = (m + dm) % 60
            new_h = h + (m + dm) // 60
            new_angle = round(self._angle_between(new_h, new_m), 1)
            q = (f"Starting from the time shown on the clock, what will the "
                 f"angle between the hands be after {dm} minutes? Round to "
                 f"1 decimal.")
            return q, str(new_angle), img
        elif qtype == "overlap_count":
            n_hours = rng.choice([6, 12, 24])
            # Hands overlap ~11 times per 12 hours
            overlaps = round(n_hours * 11 / 12)
            if n_hours == 12:
                overlaps = 11
            elif n_hours == 24:
                overlaps = 22
            elif n_hours == 6:
                overlaps = 5
            q = f"How many times do the clock hands overlap in {n_hours} hours?"
            return q, str(overlaps), img
        elif qtype == "minute_hand_angle":
            q = ("For the time shown on the clock, how many degrees has the "
                 "minute hand moved from 12? Answer as a number.")
            return q, str(round(md, 1)), img
        return None
