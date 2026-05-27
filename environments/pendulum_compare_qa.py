"""
Pendulum length-comparison MCQ. Two (or three) pendulums of different
string lengths hang from a fixed support; identify which swings faster
or which completes more swings in a fixed time.

Targets the "swing-period intuition" gap on mechanical-aptitude visual
benchmarks (Bennett-style template family, swing-faster question).

Visual style: a horizontal support bar + 2-3 strings hanging straight
down, each ending in a circular bob. Strings labelled A / B / (C).
Plain monochrome line diagram, white background.

Rule: shorter string ⇒ shorter period ⇒ faster (more swings). Mass
of bob is irrelevant by design (small physics intuition test).

Difficulty axes:
  - L0: 2 pendulums, length ratio ≥ 1.8 (very obvious)
  - L9: 3 pendulums, length ratio ≥ 1.15 (subtle)
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_FASTER = [
    "Which pendulum will swing faster? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Of the pendulums shown, which one will swing faster? Answer with a single letter (A/B/C/D) in <answer>...</answer>.",
    "Look at the pendulums hanging from the support. Which one swings faster? Answer with the letter (A/B/C/D) in <answer>...</answer>.",
    "Which of the pendulums in the image will swing the fastest? Place the letter (A/B/C/D) in <answer>...</answer>.",
    "Identify the pendulum that swings the fastest. Reply with one letter (A/B/C/D) in <answer>...</answer>.",
    "Compare the pendulums in the diagram. Which one has the shortest swing period (i.e., swings fastest)? Letter in <answer>...</answer>.",
    "Each labelled pendulum hangs from the same support. Which one swings the fastest? Reply with a single letter (A/B/C/D) in <answer>...</answer>.",
    "Which pendulum among those shown will complete a back-and-forth swing in the shortest time? Letter (A/B/C/D) in <answer>...</answer>.",
]

_TEMPLATES_SLOWER = [
    "Which pendulum will swing the slowest? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Of the pendulums shown, which one will swing slower than the others? Answer with a single letter (A/B/C/D) in <answer>...</answer>.",
    "Identify the pendulum that swings the slowest. Letter (A/B/C/D) in <answer>...</answer>.",
    "Which of the pendulums in the image takes the longest time to swing back and forth? Reply with one letter (A/B/C/D) in <answer>...</answer>.",
    "Compare the pendulums shown. Which one has the longest swing period? Letter (A/B/C/D) in <answer>...</answer>.",
    "Which pendulum among those shown will complete a back-and-forth swing in the longest time? Letter (A/B/C/D) in <answer>...</answer>.",
]

_TEMPLATES_MORE_SWINGS = [
    "If all the pendulums are released together, which one will complete the most swings in the same amount of time? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Released at the same instant, which pendulum will finish the most full swings within a fixed time interval? Letter (A/B/C/D) in <answer>...</answer>.",
    "Which pendulum will undergo the largest number of complete oscillations in a given time? Reply with a single letter (A/B/C/D) in <answer>...</answer>.",
    "If you start all the pendulums swinging at once, which will complete the greatest number of swings within one minute? Letter (A/B/C/D) in <answer>...</answer>.",
]

_TEMPLATES_FEWER_SWINGS = [
    "If all the pendulums are released together, which one will complete the fewest swings in the same amount of time? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Released at the same instant, which pendulum will finish the fewest full swings within a fixed time interval? Letter (A/B/C/D) in <answer>...</answer>.",
    "Which pendulum will undergo the smallest number of complete oscillations in a given time? Reply with a single letter (A/B/C/D) in <answer>...</answer>.",
]

_HINT_BASE = (
    " Hint: a pendulum's period depends on the length of its string — shorter "
    "string means shorter period (faster swing). The mass of the bob does not "
    "affect the period."
)

_HINT_CONCISE = " Hint: shorter string = faster swing. Bob mass does not matter."


class PendulumCompareQA(StandaloneVisualEnv):
    ENV_NAME = "pendulum_compare"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"n_pend": 2, "min_ratio": 1.8, "vary_mass": False, "templates": "faster_only"}
        if level <= 3:
            return {"n_pend": 2, "min_ratio": 1.5, "vary_mass": True, "templates": "faster_or_slower"}
        if level <= 5:
            return {"n_pend": 3, "min_ratio": 1.3, "vary_mass": True, "templates": "faster_or_slower"}
        if level <= 7:
            return {"n_pend": 3, "min_ratio": 1.2, "vary_mass": True, "templates": "all"}
        return {"n_pend": 3, "min_ratio": 1.15, "vary_mass": True, "templates": "all"}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7919 + level * 41 + 113)

        n = cfg["n_pend"]
        # Pick lengths on a 0.4..1.0 normalized scale
        for _ in range(50):
            lengths = sorted([round(rng.uniform(0.30, 1.00), 2) for _ in range(n)])
            # Check minimum ratio between any consecutive pair
            ok = True
            for i in range(n - 1):
                if lengths[i + 1] / lengths[i] < cfg["min_ratio"]:
                    ok = False
                    break
            if ok:
                break
        else:
            # fallback: force-spread
            lengths = [0.30 + i * (0.7 / max(1, n - 1)) for i in range(n)]
            lengths = [round(x, 2) for x in lengths]

        # Choose mode: which question template family
        modes = ["faster"]
        if cfg["templates"] in ("faster_or_slower", "all"):
            modes.append("slower")
        if cfg["templates"] == "all":
            modes.extend(["more_swings", "fewer_swings"])
        mode = rng.choice(modes)

        # Decide labels and answer
        # Shuffle which length goes to which letter
        order = list(range(n))
        rng.shuffle(order)
        # order[i] = pendulum index in `lengths` for letter i
        # length_for_letter[i] = lengths[order[i]]
        length_for_letter = [lengths[i] for i in order]

        if mode == "faster" or mode == "more_swings":
            min_idx = length_for_letter.index(min(length_for_letter))
            answer = chr(ord("A") + min_idx)
        else:
            max_idx = length_for_letter.index(max(length_for_letter))
            answer = chr(ord("A") + max_idx)

        # Bob mass varies if cfg permits, but it's a distractor
        if cfg["vary_mass"]:
            masses = [round(rng.uniform(0.5, 1.0), 2) for _ in range(n)]
        else:
            masses = [0.7] * n

        # Question text
        if mode == "faster":
            q = rng.choice(_TEMPLATES_FASTER)
        elif mode == "slower":
            q = rng.choice(_TEMPLATES_SLOWER)
        elif mode == "more_swings":
            q = rng.choice(_TEMPLATES_MORE_SWINGS)
        else:
            q = rng.choice(_TEMPLATES_FEWER_SWINGS)

        if level <= 1:
            q = q + _HINT_BASE
        elif level <= 4:
            q = q + _HINT_CONCISE

        img = self._render(length_for_letter, masses, rng)
        return q, answer, img

    # -------------------- rendering -------------------- #
    def _render(self, lengths, masses, rng) -> Image.Image:
        n = len(lengths)
        fig, ax = plt.subplots(figsize=(3.2 + 1.6 * n, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Support bar y position at top; pendulums hang downward
        support_y = 1.0
        # X spacing
        spacing = 1.6
        x0 = -spacing * (n - 1) / 2.0
        # Choose colors
        bob_color = rng.choice(["#1a5276", "#7b241c", "#196f3d", "#1f618d", "#5d4037"])
        line_color = "#212121"

        # Support bar (slightly wider than the pendulum spread)
        bar_x_lo = x0 - spacing * 0.7
        bar_x_hi = x0 + spacing * (n - 1) + spacing * 0.7
        ax.plot([bar_x_lo, bar_x_hi], [support_y, support_y],
                color=line_color, linewidth=4)
        # Hatching above the support to indicate fixed surface
        for k in range(int((bar_x_hi - bar_x_lo) / 0.12)):
            xa = bar_x_lo + 0.06 + k * 0.12
            ax.plot([xa, xa - 0.10], [support_y + 0.03, support_y + 0.18],
                    color=line_color, linewidth=1.0)
        ax.plot([bar_x_lo, bar_x_hi], [support_y + 0.18, support_y + 0.18],
                color=line_color, linewidth=1.5)

        # Each pendulum: vertical string + bob at the bottom
        for i, (L, m) in enumerate(zip(lengths, masses)):
            cx = x0 + i * spacing
            # String from (cx, support_y) to (cx, support_y - L * 3.5)
            tip_y = support_y - L * 3.5
            ax.plot([cx, cx], [support_y, tip_y],
                    color=line_color, linewidth=1.4)
            # Pivot dot
            ax.plot([cx], [support_y], "o", color=line_color,
                    markersize=4, zorder=5)
            # Bob (radius scales mildly with mass — but mass should not matter)
            radius = 0.10 + m * 0.10
            bob = mpatches.Circle((cx, tip_y - radius), radius,
                                  facecolor=bob_color,
                                  edgecolor="#212121", linewidth=1.2,
                                  zorder=4)
            ax.add_patch(bob)
            # Label below the bob
            ax.text(cx, tip_y - 2 * radius - 0.18, chr(ord("A") + i),
                    fontsize=18, fontweight="bold", ha="center", va="top",
                    family="DejaVu Sans")

        ax.set_xlim(bar_x_lo - 0.3, bar_x_hi + 0.3)
        # Keep enough vertical space for all
        max_drop = max(L for L in lengths) * 3.5
        ax.set_ylim(support_y - max_drop - 0.9, support_y + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_pendulum"
    os.makedirs(out_dir, exist_ok=True)
    env = PendulumCompareQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']}")
