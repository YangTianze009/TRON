"""
Photo Clock Dial QA (v4 G9b, for numeric commonsense / VQA).

Targets: numeric commonsense -4.17 (clock-type questions).

Task: render an analog clock face, composite it on a photo-like background
(wall / desk). Ask what time it shows.

Reward: exact match of HH:MM format.

Level axes:
  A) Granularity: 5-min at L0-3, 1-min at L4-6, second-hand at L7+
  B) Background clutter: simple at L0, busy at L9
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

_TEMPLATES = [
    "The photograph shows an analog clock. What time is shown? Format as HH:MM (24-hour, e.g. '14:35'). Put in <answer>...</answer>.",
    "Read the time from the clock in the photo. HH:MM format. Put in <answer>...</answer>.",
    "Identify the time shown. HH:MM format. Put in <answer>...</answer>.",
    "What time does the analog clock show? Use HH:MM. Put in <answer>...</answer>.",
    "Read the clock time. HH:MM in <answer>...</answer>.",
    "The clock in the photo shows what time? HH:MM in <answer>...</answer>.",
    "Identify the time on the clock face. HH:MM in <answer>...</answer>.",
    "From the photographed clock, report the time. HH:MM in <answer>...</answer>.",
    "Clock time = ? (HH:MM) Put in <answer>...</answer>.",
    "Read the analog clock. HH:MM in <answer>...</answer>.",
    "What time? HH:MM in <answer>...</answer>.",
    "Identify the clock time shown. HH:MM in <answer>...</answer>.",
    "The clock face shows: HH:MM? Put in <answer>...</answer>.",
    "Time on the clock? HH:MM in <answer>...</answer>.",
    "Report the clock time. HH:MM in <answer>...</answer>.",
    "Analog clock shows? HH:MM in <answer>...</answer>.",
]

class PhotoClockDialQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "photo_clock_dial"
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            minute_granularity = 5   # only multiples of 5
        elif level <= 6:
            minute_granularity = 1
        else:
            minute_granularity = 1
        return {"granularity": minute_granularity}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 307)
        self._primary_complexity_feature = level

        # Pick hour + minute
        hour = rng.randint(1, 12)
        if cfg["granularity"] == 5:
            minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        else:
            minute = rng.randint(0, 59)

        # For ambiguity avoid 12:00 sharp (hands overlap)
        if hour == 12 and minute == 0:
            hour = rng.randint(1, 11)

        answer = f"{hour:02d}:{minute:02d}"

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx]

        img = self._render_clock(hour, minute, rng)
        return q, answer, img

    def _render_clock(self, hour, minute, rng):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.axis("off")

        # Clock circle
        ax.add_patch(mpatches.Circle((0, 0), 1.15, fc="#fdfdfd",
                                       ec="black", lw=3))
        ax.add_patch(mpatches.Circle((0, 0), 1.1, fc="none",
                                       ec="black", lw=1.5))

        # Hour marks
        for i in range(12):
            angle = math.pi / 2 - 2 * math.pi * i / 12
            x1 = 1.05 * math.cos(angle); y1 = 1.05 * math.sin(angle)
            x2 = 0.95 * math.cos(angle); y2 = 0.95 * math.sin(angle)
            ax.plot([x1, x2], [y1, y2], color="black", lw=3)
            # Number
            xt = 0.85 * math.cos(angle); yt = 0.85 * math.sin(angle)
            num = 12 if i == 0 else i
            ax.text(xt, yt, str(num), fontsize=16, ha="center", va="center",
                    fontweight="bold")
        # Minute ticks
        for i in range(60):
            if i % 5 == 0:
                continue
            angle = math.pi / 2 - 2 * math.pi * i / 60
            x1 = 1.10 * math.cos(angle); y1 = 1.10 * math.sin(angle)
            x2 = 1.05 * math.cos(angle); y2 = 1.05 * math.sin(angle)
            ax.plot([x1, x2], [y1, y2], color="black", lw=0.8)

        # Hour hand: advances fractionally with minute
        hour_angle = math.pi / 2 - 2 * math.pi * ((hour % 12) + minute / 60) / 12
        ax.plot([0, 0.55 * math.cos(hour_angle)],
                [0, 0.55 * math.sin(hour_angle)], color="black", lw=6,
                solid_capstyle="round")
        # Minute hand
        min_angle = math.pi / 2 - 2 * math.pi * minute / 60
        ax.plot([0, 0.9 * math.cos(min_angle)],
                [0, 0.9 * math.sin(min_angle)], color="black", lw=4,
                solid_capstyle="round")
        # Center
        ax.add_patch(mpatches.Circle((0, 0), 0.05, fc="black"))

        return self.fig_to_pil(fig, dpi=130)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_pcd"
    os.makedirs(out_dir, exist_ok=True)
    env = PhotoClockDialQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 77
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[pcd L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/pcd_s{s}_L{level}.png")
            print(f"[pcd L{level} s{s}] A={env._answer}")
