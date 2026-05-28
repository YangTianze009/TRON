"""
Instrument Panel QA (D77, P3).

Reference an external reference:
  "A ticket will be issued when the vehicle speed exceeds 70 mph. According
   to the instrument panel below, will this vehicle get the ticket?
   choice: (A) Yes (B) No."  Ans: A

Renders an analog speedometer with a needle pointing at some speed value.
Asks (a) read the needle value or (b) compare against a threshold (Yes/No).
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


class InstrumentPanelQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "instrument_panel"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"max_speed": 100, "modes": ["read", "threshold"],
                    "step": 10}
        if level <= 5:
            return {"max_speed": 140, "modes": ["read", "threshold"],
                    "step": 5}
        return {"max_speed": 200, "modes": ["read", "threshold"],
                "step": 1}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4861 + level * 53 + 19)

        max_speed = cfg["max_speed"]
        step = cfg["step"]
        speed = rng.randrange(0, max_speed + step, step)
        if speed > max_speed:
            speed = max_speed

        mode = rng.choice(cfg["modes"])
        if mode == "read":
            answer = str(speed)
            question = (
                f"The figure shows a speedometer instrument panel (max "
                f"speed {max_speed} mph). According to the needle position, "
                f"what is the current vehicle speed in mph? Answer with an "
                f"integer."
            )
        else:
            threshold = rng.choice([55, 65, 70, 75, 80])
            if threshold > max_speed - step:
                threshold = max_speed // 2
            if speed > threshold:
                answer = "A"
                ans_word = "Yes"
            else:
                answer = "B"
                ans_word = "No"
            question = (
                f"The figure shows a speedometer instrument panel (max speed "
                f"{max_speed} mph). A ticket will be issued when the vehicle "
                f"speed exceeds {threshold} mph. According to the panel, "
                f"will this vehicle get the ticket?\n\n"
                f"A. Yes\nB. No\n\nChoose A or B."
            )

        img = self._render(speed, max_speed)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, speed, max_speed) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        # Outer dial (semi-circle from 180° down to 0°)
        # We use angle range from 200° (left side) to -20° (right side)
        # Convention: angle 0 = pointing right; we sweep CCW from 200° to -20°
        # (this means the speedometer is the lower half rotated)
        # Use a more standard layout: 180° at left zero, 0° at right max
        # We use full bottom half: speed=0 at 220°, speed=max at -40°.
        start_deg = 220
        end_deg = -40
        sweep = end_deg - start_deg  # negative

        # Major ticks every step
        n_ticks = 11
        for i in range(n_ticks):
            t_speed = i / (n_ticks - 1) * max_speed
            t_angle = start_deg + sweep * (i / (n_ticks - 1))
            tr = math.radians(t_angle)
            xo, yo = 0.92 * math.cos(tr), 0.92 * math.sin(tr)
            xi, yi = 1.0 * math.cos(tr), 1.0 * math.sin(tr)
            ax.plot([xo, xi], [yo, yi], color="#ecf0f1", linewidth=2)
            xn, yn = 0.78 * math.cos(tr), 0.78 * math.sin(tr)
            ax.text(xn, yn, str(int(round(t_speed))),
                    ha="center", va="center", fontsize=11, color="#ecf0f1")

        # Outer arc
        from matplotlib.patches import Arc
        arc = Arc((0, 0), 2, 2, angle=0, theta1=end_deg, theta2=start_deg,
                  color="#ecf0f1", linewidth=2)
        ax.add_patch(arc)

        # Needle
        frac = speed / max_speed
        needle_angle = start_deg + sweep * frac
        nr = math.radians(needle_angle)
        ax.plot([0, 0.85 * math.cos(nr)], [0, 0.85 * math.sin(nr)],
                color="#e74c3c", linewidth=4, solid_capstyle="round")
        # Pivot
        ax.scatter([0], [0], s=80, color="#ecf0f1", zorder=5)

        # Title
        ax.text(0, -0.5, "MPH", ha="center", fontsize=14,
                fontweight="bold", color="#ecf0f1")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.0, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = InstrumentPanelQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
