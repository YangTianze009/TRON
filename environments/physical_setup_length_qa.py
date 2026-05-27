"""
Physical Setup Length QA (v4 G10/G2d, for length).

Targets: metric geometry - length -0.89 (pulleys, inclined planes,
ladder-against-wall cases).

Failure mode: model applies arithmetic without identifying the physical setup
first (e.g., 2:1 pulley mechanical advantage, angle-of-walk formula).

Task: render a physical setup (pulley / ladder / slope) with labeled
dimensions. Ask for a specific length given the setup-specific relation.

Reward: numeric within 1% relative tolerance.

Level axes:
  A) Setup: ladder at L0-3, simple pulley at L4-6, compound pulley at L7+
  B) Number of givens: 2 at L0, 3 at L3+
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

_TEMPLATES_LADDER = [
    "A ladder of length {L} leans against a vertical wall. The base of the ladder is {d} from the wall. Compute the height the ladder reaches on the wall. Round to 2 decimal places; put in <answer>...</answer>.",
    "Ladder length {L}, base distance {d} from wall. Wall height reached? Round 2dp; put in <answer>...</answer>.",
    "A ladder of length {L} is placed against a wall, base {d} from it. Find wall-reach height. 2dp; put in <answer>...</answer>.",
    "Given a {L}-long ladder with base {d} from a wall, compute wall-contact height. 2dp in <answer>...</answer>.",
    "A vertical wall meets a horizontal floor. A ladder of length {L} leans against the wall; base {d} from wall. Height on wall? 2dp in <answer>...</answer>.",
    "Ladder: length {L}, base distance {d}. Find wall-reach. 2dp in <answer>...</answer>.",
    "Compute the height h of the ladder's top, given ladder length {L} and base {d}. 2dp in <answer>...</answer>.",
    "The ladder (length {L}) touches a wall; its base is {d} out. Wall-touch height? 2dp in <answer>...</answer>.",
    "A {L}-unit ladder rests with base {d} from a vertical wall. Wall-height reached? 2dp in <answer>...</answer>.",
    "Compute ladder reach: length {L}, base-from-wall {d}. 2dp in <answer>...</answer>.",
    "Given ladder length {L} and base distance {d}, find wall height reached (Pythagoras). 2dp in <answer>...</answer>.",
    "Length {L}, base {d}, wall height? 2dp in <answer>...</answer>.",
    "Ladder of length {L} with base {d} away — find the wall reach. 2dp in <answer>...</answer>.",
    "Ladder reach problem: length {L}, base distance {d}. 2dp in <answer>...</answer>.",
    "Use the Pythagorean theorem: ladder length {L}, base {d}, wall height? 2dp in <answer>...</answer>.",
    "Compute wall-reach h of a {L}-unit ladder, base {d}. 2dp in <answer>...</answer>.",
]

_TEMPLATES_PULLEY = [
    "A simple pulley system with {N} supporting ropes lifts a load. The free end is pulled down by {d} units. How far does the load rise? Mechanical advantage = N. Round to 2 decimal places; put in <answer>...</answer>.",
    "Pulley with {N} supporting ropes. Free end pulled {d} down. Load rise? 2dp in <answer>...</answer>.",
    "Given a pulley MA of {N} and rope pulled {d}, how much does the load lift? 2dp in <answer>...</answer>.",
    "Compound pulley: {N} ropes, {d}-unit pull. Load lift? 2dp in <answer>...</answer>.",
    "The pulley has {N} lifting ropes. Pulling free end {d} units lifts the load by? 2dp in <answer>...</answer>.",
    "Pulley advantage {N}; rope pull {d}. Load displacement? 2dp in <answer>...</answer>.",
    "A {N}-rope pulley with free end pulled {d}. Load rise? 2dp in <answer>...</answer>.",
    "With N={N} supporting ropes, pulling {d} gives what load lift? 2dp in <answer>...</answer>.",
    "Compute load lift for N={N} ropes, pull {d}. 2dp in <answer>...</answer>.",
    "Pulley: {N} ropes, pull {d}. Load rise? 2dp in <answer>...</answer>.",
    "Load lift = pull / N. Given N={N}, pull={d}. Result? 2dp in <answer>...</answer>.",
    "Given mechanical advantage {N} and free-end pull {d}, lift = ? 2dp in <answer>...</answer>.",
    "Pulley problem: N={N}, pull={d}. Lift? 2dp in <answer>...</answer>.",
    "Compute load displacement: N={N}, pull={d}. 2dp in <answer>...</answer>.",
    "N ropes = {N}, free-end pulled {d}. Load rise? 2dp in <answer>...</answer>.",
    "Pulley with {N} supports, pull {d}. Load displacement? 2dp in <answer>...</answer>.",
]

class PhysicalSetupLengthQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "physical_setup_length"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            setups = ["ladder"]
        elif level <= 6:
            setups = ["ladder", "pulley"]
        else:
            setups = ["ladder", "pulley"]
        return {"setups": setups, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 293)
        self._primary_complexity_feature = level

        setup = rng.choice(cfg["setups"])
        if setup == "ladder":
            L = rng.randint(5, 20)
            d = rng.randint(2, L - 2)
            h = math.sqrt(L * L - d * d)
            answer = f"{round(h, 2)}"
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_LADDER[sidx].format(L=L, d=d)
            img = self._render_ladder(L, d, rng)
        else:
            N = rng.choice([2, 3, 4])
            d = rng.randint(4, 20)
            lift = d / N
            answer = f"{round(lift, 2)}"
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_PULLEY[sidx].format(N=N, d=d)
            img = self._render_pulley(N, d, rng)

        return q, answer, img

    def _render_ladder(self, L, d, rng):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        h = math.sqrt(L * L - d * d)
        lim = max(L, h, d) + 2
        ax.set_xlim(-0.5, lim)
        ax.set_ylim(-0.5, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        # wall (left)
        ax.plot([0, 0], [0, lim], color="black", lw=3)
        # floor
        ax.plot([0, lim], [0, 0], color="black", lw=3)
        # ladder (from base (d, 0) to top (0, h))
        ax.plot([d, 0], [0, h], color="brown", lw=4)
        # labels
        ax.text(d / 2, -0.3, f"base = {d}", fontsize=12, ha="center",
                fontweight="bold")
        ax.text((d + 0) / 2, h / 2, f"L = {L}", fontsize=12,
                rotation=math.degrees(math.atan2(h, d)) + 90, ha="center",
                fontweight="bold")
        ax.text(-0.4, h / 2, "h = ?", fontsize=12, ha="center",
                color="darkgreen", fontweight="bold")
        return self.fig_to_pil(fig)

    def _render_pulley(self, N, d, rng):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")
        # ceiling
        ax.plot([0, 6], [7, 7], color="black", lw=3)
        # top pulley
        ax.add_patch(mpatches.Circle((3, 6.5), 0.35, fc="#cccccc",
                                       ec="black", lw=1.5))
        # Rope strands (N pieces)
        for i in range(N):
            xoff = 2.5 + i * (1.0 / max(N - 1, 1)) if N > 1 else 3
            ax.plot([xoff, xoff], [2, 6.5], color="black", lw=1.3)
        # Load
        ax.add_patch(mpatches.Rectangle((2.3, 1), 1.4, 1,
                                         fc="#95a5a6", ec="black", lw=1.5))
        ax.text(3, 1.5, "LOAD", fontsize=10, ha="center", fontweight="bold")
        # Free end label
        ax.annotate(f"pull = {d}", xy=(3 + (N - 1) * 0.5, 6), xytext=(5, 4),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=12, fontweight="bold", color="red")
        ax.text(0.3, 6, f"N = {N} supporting ropes",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="lightyellow", edgecolor="gray"))
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        try:
            p = float(pred)
            g = float(gt)
            return abs(p - g) < 0.1 or abs(p - g) / max(abs(g), 1e-9) < 0.01
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_psl"
    os.makedirs(out_dir, exist_ok=True)
    env = PhysicalSetupLengthQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 139
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[psl L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/psl_s{s}_L{level}.png")
            print(f"[psl L{level} s{s}] A={env._answer}")
