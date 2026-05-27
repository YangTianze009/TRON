"""
Indoor Region Compass QA (v4 G13, for Pos-Reg-Reg (spatial)).

Targets: spatial-reasoning Pos-Reg-Reg -9.88 (single biggest spatial-reasoning drop).

Task (top-down, no THOR required): render a top-down floor plan of a
rectangular room with functional-region overlays (e.g., "sleeping area",
"bathing area", "cooking area", "lounge area"). Given a premise ("assume the
X area is on the south wall"), ask where another area Y is located.

This complements spatial-reasoning -style questions without needing AI2-THOR
rendering. The room is drawn with named colored zones, with a compass
indicator so the premise can be applied consistently.

Reward: MCQ letter match.

Level axes:
  A) Number of regions: 3 at L0, 4 at L3, 5 at L6+
  B) Options: 4 corners at L0-5, 4 corners + 4 walls (8 way) at L6+
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_REGION_TYPES = {
    "sleeping area": ("#ffe6ee", ["bed", "nightstand", "wardrobe"]),
    "bathing area": ("#e3f2fd", ["sink", "toilet", "mirror"]),
    "cooking area": ("#fff3e0", ["stove", "fridge", "microwave"]),
    "lounge area": ("#f3e5f5", ["sofa", "tv", "rug"]),
    "dining area": ("#e8f5e8", ["dining_table", "chairs"]),
    "reading area": ("#fff9c4", ["bookshelf", "desk", "lamp"]),
    "storage area": ("#efebe9", ["cabinet", "shelf", "box"]),
    "work area": ("#d1c4e9", ["desk", "chair", "computer"]),
}

_TEMPLATES_CORNERS = [
    "The top-down floor plan shows a rectangular room divided into functional regions. **Premise**: the {anchor} is on the {anchor_wall} wall. Given this, in which corner of the room is the {target} located? Options: A. northwest  B. northeast  C. southwest  D. southeast. Put the letter in <answer>...</answer>.",
    "Floor plan with functional regions. Given that the {anchor} is on the {anchor_wall} wall, where is the {target}? A-D (NW/NE/SW/SE). Put letter in <answer>...</answer>.",
    "Given the top-down plan and the premise that the {anchor} is on the {anchor_wall} wall, determine the corner location of the {target}. A-D. Put letter in <answer>...</answer>.",
    "If the {anchor} is on the {anchor_wall} wall, in which corner is the {target}? A-D. Put letter in <answer>...</answer>.",
    "Floor plan: {anchor} on {anchor_wall} wall (premise). {target} is at which corner? A-D. Put letter in <answer>...</answer>.",
    "Room layout with premise: {anchor} = {anchor_wall} wall. Corner of {target}? A-D. Put letter in <answer>...</answer>.",
    "Determine the corner of {target} given: {anchor} = {anchor_wall} wall. A-D. Put letter in <answer>...</answer>.",
    "Assuming the {anchor} is on the {anchor_wall} wall, where is the {target}? A-D corners. Put letter in <answer>...</answer>.",
    "Given premise ({anchor} at {anchor_wall} wall), {target} corner? A-D. Put letter in <answer>...</answer>.",
    "Compass premise {anchor} = {anchor_wall}, find {target}. A-D. Put letter in <answer>...</answer>.",
    "If {anchor} is the {anchor_wall}-wall region, {target} is at which corner? A-D. Put letter in <answer>...</answer>.",
    "Premise: {anchor} = {anchor_wall}. Answer: {target} corner? A-D. Put letter in <answer>...</answer>.",
    "Floor-plan compass question: {anchor} on {anchor_wall} wall. {target} at which corner? A-D. Put letter in <answer>...</answer>.",
    "Given {anchor} at {anchor_wall} wall, {target} is in: A. NW  B. NE  C. SW  D. SE? Put letter in <answer>...</answer>.",
    "With {anchor} = {anchor_wall}, identify {target}'s corner (A-D). Put letter in <answer>...</answer>.",
    "Floor plan problem: {anchor} = {anchor_wall} wall, find {target}'s corner. A-D. Put letter in <answer>...</answer>.",
]

class IndoorRegionCompassQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "indoor_region_compass"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_regions = min(5, 3 + level // 3)
        return {"n_regions": n_regions}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 457)
        self._primary_complexity_feature = level

        # Pick regions (cap at 4 to match corner count — keep it simple)
        n_regions = min(4, cfg["n_regions"])
        region_names = rng.sample(list(_REGION_TYPES.keys()), n_regions)

        # Layout: room is 10x8. Split into 4 quadrants (NW, NE, SW, SE).
        corners = [("NW", (0, 4, 5, 8)), ("NE", (5, 4, 10, 8)),
                   ("SW", (0, 0, 5, 4)), ("SE", (5, 0, 10, 4))]
        rng.shuffle(corners)

        # Assign each region to a quadrant
        region_to_corner = {}
        for i, region in enumerate(region_names):
            corner_label, _ = corners[i]
            region_to_corner[region] = corner_label
        extras = []

        # Anchor is the first region, specify which wall it's on.
        anchor = region_names[0]
        anchor_corner = region_to_corner[anchor]
        # Pick a wall that's consistent with the corner (anchor_corner is one
        # of NW/NE/SW/SE; so 'N' or 'W' for NW, etc.)
        wall_choices = {
            "NW": ["north", "west"], "NE": ["north", "east"],
            "SW": ["south", "west"], "SE": ["south", "east"],
        }
        anchor_wall = rng.choice(wall_choices[anchor_corner])

        # Target is a different region
        target = rng.choice([r for r in region_names if r != anchor])
        target_corner = region_to_corner[target]

        # Answer: target_corner letter (A=NW, B=NE, C=SW, D=SE)
        letter = {"NW": "A", "NE": "B", "SW": "C", "SE": "D"}[target_corner]

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES_CORNERS[sidx].format(
            anchor=anchor, anchor_wall=anchor_wall, target=target)
        if level <= 2:
            q += (
                f" Hint: the {anchor} is on the {anchor_wall} wall, so "
                f"that establishes which direction is {anchor_wall}. "
                "From there, infer the other compass directions (N/S/E/W). "
                f"Locate the {target} in the layout and read off its "
                "corner relative to the established compass."
            )

        img = self._render(region_to_corner, extras, rng)
        return q, letter, img

    def _render(self, region_to_corner, extras, rng):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-0.5, 9.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Room outline
        ax.add_patch(mpatches.Rectangle((0, 0), 10, 8, fc="none",
                                         ec="black", lw=2.5))
        # Compass rose
        ax.annotate("N", xy=(-0.3, 8.3), fontsize=14, ha="center",
                    fontweight="bold", color="red")
        ax.annotate("S", xy=(-0.3, -0.3), fontsize=14, ha="center",
                    fontweight="bold", color="red")
        ax.annotate("E", xy=(10.3, 4), fontsize=14, ha="center",
                    fontweight="bold", color="red")
        ax.annotate("W", xy=(-0.3, 4), fontsize=14, ha="center",
                    fontweight="bold", color="red")

        corners = {"NW": (0, 4, 5, 8), "NE": (5, 4, 10, 8),
                    "SW": (0, 0, 5, 4), "SE": (5, 0, 10, 4)}

        for region, corner in region_to_corner.items():
            x1, y1, x2, y2 = corners[corner]
            color, _ = _REGION_TYPES[region]
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             fc=color, ec="gray", lw=1.0,
                                             alpha=0.7))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(cx, cy, region, fontsize=11, ha="center", va="center",
                    fontweight="bold")

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_irc"
    os.makedirs(out_dir, exist_ok=True)
    env = IndoorRegionCompassQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 31
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[irc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/irc_s{s}_L{level}.png")
            print(f"[irc L{level} s{s}] A={env._answer}")
