"""
Object Vertical Relation QA (v4 G15b, for Pos-Obj-Obj (spatial) height comparison).

Targets: spatial-reasoning Pos-Obj-Obj -4.26 (idx=423 "which is higher: green bottle
or hand sanitizer" style).

Task: side-view schematic with multiple objects at different heights on
shelves. Ask which object is higher (or at the same height).

Reward: MCQ letter match.

Level axes:
  A) Number of objects: 2 at L0, 4 at L3, 6 at L6+
  B) Differences in height smaller at higher levels
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_OBJECT_NAMES = ["red bottle", "green bottle", "blue jar", "yellow can",
                  "white mug", "black vase", "orange pot", "purple box",
                  "gray tin", "brown crate"]

_TEMPLATES = [
    "Looking at the side-view shelf diagram, which is higher: the {obj1} or the {obj2}? Options: A. {obj1}  B. {obj2}  C. same height  D. cannot be determined. Put letter in <answer>...</answer>.",
    "Which object is placed higher, {obj1} or {obj2}? A-D (A={obj1}, B={obj2}, C=same, D=cannot determine). Put letter in <answer>...</answer>.",
    "Compare heights of {obj1} and {obj2}. A={obj1}, B={obj2}, C=same, D=cannot determine. Put letter in <answer>...</answer>.",
    "Side view shows shelves. Which is higher: {obj1} or {obj2}? A-D. Put letter in <answer>...</answer>.",
    "{obj1} vs {obj2}: which is higher? A/B/C/D. Put letter in <answer>...</answer>.",
    "From the shelf diagram, which of {obj1} or {obj2} is at a higher position? A-D. Put letter in <answer>...</answer>.",
    "Determine the higher-placed object: {obj1} or {obj2}? A-D. Put letter in <answer>...</answer>.",
    "Which is higher placed, {obj1} or {obj2}? A-D. Put letter in <answer>...</answer>.",
    "From the side view, which of A={obj1} vs B={obj2} is higher? C=same, D=cannot. Put letter in <answer>...</answer>.",
    "Height comparison: {obj1} vs {obj2}. A-D. Letter in <answer>...</answer>.",
    "Pick the higher-placed object: {obj1} or {obj2}. A-D. Put letter in <answer>...</answer>.",
    "A = {obj1}, B = {obj2}, C = same, D = cannot determine. Which is higher? Put letter in <answer>...</answer>.",
    "Compare positions: {obj1}, {obj2}. Which is higher? A-D. Put letter in <answer>...</answer>.",
    "From the figure, the higher object between {obj1} and {obj2} is: A-D. Put letter in <answer>...</answer>.",
    "{obj1} or {obj2}: which sits higher? A-D. Put letter in <answer>...</answer>.",
    "Side-view height question: {obj1} or {obj2}, which is higher? A-D. Put letter in <answer>...</answer>.",
]

class ObjectVerticalRelationQA(StandaloneVisualEnv):
    ENV_NAME = "object_vertical_relation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_obj = min(6, 2 + level // 2)
        return {"n_obj": n_obj}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 311)
        self._primary_complexity_feature = level

        n = cfg["n_obj"]
        names = rng.sample(_OBJECT_NAMES, n)
        # Each object gets a random height on 3 shelves (bottom / mid / top)
        shelf_levels = [1.0, 2.5, 4.0]
        heights = [rng.choice(shelf_levels) for _ in range(n)]
        # Arrange on x axis evenly
        positions = [(i * 2 + 1, heights[i]) for i in range(n)]

        # Pick two objects to compare
        i, j = rng.sample(range(n), 2)
        obj1, obj2 = names[i], names[j]
        h1, h2 = heights[i], heights[j]
        if h1 > h2:
            letter = "A"
        elif h2 > h1:
            letter = "B"
        else:
            letter = "C"

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(obj1=obj1, obj2=obj2)

        img = self._render(names, positions, rng)
        return q, letter, img

    def _render(self, names, positions, rng):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        xlim = max(x for x, _ in positions) + 2
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, 5.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw 3 shelves
        for y in [1.0, 2.5, 4.0]:
            ax.plot([0, xlim], [y - 0.1, y - 0.1], color="black", lw=1.5)

        # Draw objects
        color_map = {}
        for name in names:
            if "red" in name: color_map[name] = "#e74c3c"
            elif "green" in name: color_map[name] = "#2ecc71"
            elif "blue" in name: color_map[name] = "#3498db"
            elif "yellow" in name: color_map[name] = "#f1c40f"
            elif "white" in name: color_map[name] = "#ecf0f1"
            elif "black" in name: color_map[name] = "#2c3e50"
            elif "orange" in name: color_map[name] = "#e67e22"
            elif "purple" in name: color_map[name] = "#9b59b6"
            elif "gray" in name: color_map[name] = "#95a5a6"
            else: color_map[name] = "#8d6e63"

        for name, (x, y) in zip(names, positions):
            ax.add_patch(mpatches.Rectangle((x - 0.3, y), 0.6, 0.6,
                                             fc=color_map[name],
                                             ec="black", lw=1.2))
            ax.text(x, y - 0.25, name, fontsize=8, ha="center", va="top",
                    rotation=0)

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_ovr"
    os.makedirs(out_dir, exist_ok=True)
    env = ObjectVerticalRelationQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 193
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[ovr L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/ovr_s{s}_L{level}.png")
            print(f"[ovr L{level} s{s}] A={env._answer}")
