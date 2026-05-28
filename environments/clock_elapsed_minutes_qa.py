"""
Clock Elapsed Minutes QA (D36, P1).

Reference an external reference — minutes between two clocks.

Renders two analog clocks side by side showing different times. Asks for
the elapsed minutes between them.

Verifier: integer minutes (`\\boxed{N}`).
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class ClockElapsedMinutesQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "clock_elapsed_minutes"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"minute_step": 15, "max_elapsed": 60}
        if level <= 5:
            return {"minute_step": 5, "max_elapsed": 180}
        return {"minute_step": 1, "max_elapsed": 300}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7411 + level * 67 + 11)

        step = cfg["minute_step"]

        h1 = rng.randint(1, 12)
        m1 = rng.randrange(0, 60, step)

        elapsed = rng.randrange(step, cfg["max_elapsed"] + step, step)
        total = h1 * 60 + m1 + elapsed
        # Wrap to 0-720 (12-hour clock)
        total %= 720
        h2 = total // 60
        m2 = total % 60
        if h2 == 0:
            h2 = 12

        question = (
            f"The figure shows two analog clocks. The first clock shows the "
            f"start time and the second clock shows the end time. How many "
            f"minutes elapsed from the start time to the end time?"
        )
        answer = str(elapsed)
        img = self._render(h1, m1, h2, m2)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, h1, m1, h2, m2) -> Image.Image:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=110)
        fig.patch.set_facecolor("#ffffff")

        for ax, h, m, title in [(axes[0], h1, m1, "Start"),
                                  (axes[1], h2, m2, "End")]:
            ax.set_facecolor("#ffffff")
            self._draw_clock(ax, h, m, title)

        return self.fig_to_pil(fig, dpi=110)

    def _draw_clock(self, ax, hour, minute, title):
        # Clock face
        circle = plt.Circle((0, 0), 1, fill=False, edgecolor="#2c3e50",
                            linewidth=2.5)
        ax.add_patch(circle)

        # Hour ticks + numerals
        for i in range(1, 13):
            theta = math.radians(90 - i * 30)
            x_o, y_o = 0.92 * math.cos(theta), 0.92 * math.sin(theta)
            x_i, y_i = 1.0 * math.cos(theta), 1.0 * math.sin(theta)
            ax.plot([x_o, x_i], [y_o, y_i], color="#2c3e50", linewidth=2)
            x_n, y_n = 0.78 * math.cos(theta), 0.78 * math.sin(theta)
            ax.text(x_n, y_n, str(i), ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#2c3e50")

        # Minute ticks
        for i in range(60):
            if i % 5 == 0:
                continue
            theta = math.radians(90 - i * 6)
            x_o, y_o = 0.96 * math.cos(theta), 0.96 * math.sin(theta)
            x_i, y_i = 1.0 * math.cos(theta), 1.0 * math.sin(theta)
            ax.plot([x_o, x_i], [y_o, y_i], color="#7f8c8d", linewidth=0.8)

        # Hour hand
        hour_angle_deg = 90 - (hour % 12) * 30 - minute * 0.5
        hh = math.radians(hour_angle_deg)
        ax.plot([0, 0.5 * math.cos(hh)], [0, 0.5 * math.sin(hh)],
                color="#2c3e50", linewidth=4, solid_capstyle="round")

        # Minute hand
        min_angle_deg = 90 - minute * 6
        mh = math.radians(min_angle_deg)
        ax.plot([0, 0.8 * math.cos(mh)], [0, 0.8 * math.sin(mh)],
                color="#d62728", linewidth=2.5, solid_capstyle="round")

        # Center dot
        ax.scatter([0], [0], color="#2c3e50", s=20, zorder=5)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50")


if __name__ == "__main__":
    env = ClockElapsedMinutesQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
