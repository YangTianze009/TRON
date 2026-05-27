"""
Numeric Commonsense Visual QA (Task C, numeric commonsense gap, ).

Visual scene + common-sense numeric question. E.g., given an image of a room
with furniture, "About how many people can fit?" or "How many chairs are there?"

Difficulty axes:
  A) Scene complexity (few objects -> many objects)
  B) Question type (count -> estimate -> ratio)
  C) Visual noise / clutter
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_FURNITURE = ["chair", "table", "desk", "shelf", "lamp", "plant", "sofa", "rug"]
_FURNITURE_COLORS = {
    "chair": "#8B4513", "table": "#D2691E", "desk": "#A0522D",
    "shelf": "#6B8E23", "lamp": "#FFD700", "plant": "#228B22",
    "sofa": "#4682B4", "rug": "#BC8F8F",
}

class NumericCommonsenseVisualQA(StandaloneVisualEnv):
    ENV_NAME = "numeric_commonsense_visual"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_items": 3 + level,                  # 3..12
            "n_types": min(8, 2 + level // 2),     # 2..6
            "qtypes": ["count_type"] if level <= 2 else
                     ["count_type", "total_count"] if level <= 5 else
                     ["total_count", "ratio", "estimate_area"],
            "visual_noise": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_items"]

        n_items = cfg["n_items"]
        types_pool = _FURNITURE[:cfg["n_types"]]
        qtype = rng.choice(cfg["qtypes"])

        # Place items
        items = []
        for _ in range(n_items):
            item_type = rng.choice(types_pool)
            items.append({
                "type": item_type,
                "x": rng.uniform(1, 9),
                "y": rng.uniform(1, 9),
                "size": rng.uniform(0.3, 0.6),
            })

        # Generate question
        if qtype == "count_type":
            target = rng.choice(types_pool)
            count = sum(1 for i in items if i["type"] == target)
            if count == 0:
                # Ensure at least 1
                items[0]["type"] = target
                count = sum(1 for i in items if i["type"] == target)
            question = (f"The image shows a room layout with various items. "
                       f"How many {target}s are in the room? "
                       f"Answer with a single integer.")
            answer = str(count)

        elif qtype == "total_count":
            question = (f"The image shows a room layout with various items. "
                       f"How many total items (furniture/objects) are in the room? "
                       f"Answer with a single integer.")
            answer = str(n_items)

        elif qtype == "ratio":
            t1 = rng.choice(types_pool)
            t2 = rng.choice([t for t in types_pool if t != t1])
            c1 = sum(1 for i in items if i["type"] == t1)
            c2 = sum(1 for i in items if i["type"] == t2)
            if c1 == 0:
                items[0]["type"] = t1
                c1 = 1
            if c2 == 0:
                items[-1]["type"] = t2
                c2 = 1
            c1 = sum(1 for i in items if i["type"] == t1)
            c2 = sum(1 for i in items if i["type"] == t2)
            question = (f"In the room layout, what is the ratio of {t1}s to {t2}s? "
                       f"Express as a simplified fraction like '2/3' or an integer.")
            from math import gcd
            g = gcd(c1, c2)
            if c2 == 0:
                answer = str(c1)
            elif c1 % c2 == 0:
                answer = str(c1 // c2)
            else:
                answer = f"{c1 // g}/{c2 // g}"

        else:  # estimate_area
            question = (f"The room is 10x10 units. Approximately what fraction "
                       f"of the floor area is occupied by items? "
                       f"Answer as a percentage (integer, e.g., '15').")
            total_area = sum(math.pi * i["size"] ** 2 for i in items)
            pct = int(round(total_area / 100 * 100))
            answer = str(max(1, pct))

        img = self._render_room(items, cfg, rng)
        return question, answer, img

    def _render_room(self, items, cfg, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#FFF8DC")  # cream floor
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")

        # Room walls
        ax.plot([0, 10, 10, 0, 0], [0, 0, 10, 10, 0], "k-", linewidth=2.5)
        ax.set_title("Room Layout (top-down view)", fontsize=12, fontweight="bold")

        for item in items:
            x, y, s = item["x"], item["y"], item["size"]
            color = _FURNITURE_COLORS.get(item["type"], "#999999")
            itype = item["type"]

            if itype in ("chair", "lamp", "plant"):
                patch = plt.Circle((x, y), s, fc=color, ec="black",
                                  lw=1.0, alpha=0.85)
                ax.add_patch(patch)
            elif itype in ("table", "desk", "rug"):
                patch = mpatches.Rectangle((x - s, y - s * 0.6), 2 * s, 1.2 * s,
                                           fc=color, ec="black", lw=1.0, alpha=0.85)
                ax.add_patch(patch)
            elif itype == "sofa":
                patch = mpatches.FancyBboxPatch((x - s, y - s * 0.4), 2 * s, 0.8 * s,
                                                 boxstyle="round,pad=0.05",
                                                 fc=color, ec="black", lw=1.0, alpha=0.85)
                ax.add_patch(patch)
            else:
                patch = mpatches.Rectangle((x - s * 0.3, y - s), 0.6 * s, 2 * s,
                                           fc=color, ec="black", lw=1.0, alpha=0.85)
                ax.add_patch(patch)

            # Label
            ax.text(x, y, itype[0].upper(), fontsize=7, ha="center", va="center",
                   fontweight="bold", color="white")

        # Legend
        used_types = sorted(set(i["type"] for i in items))
        legend_y = 0.5
        for t in used_types:
            color = _FURNITURE_COLORS.get(t, "#999999")
            ax.plot([], [], "s", color=color, markersize=8, label=t)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        ax.set_xticks([])
        ax.set_yticks([])

        # Visual noise
        if cfg.get("visual_noise"):
            for _ in range(rng.randint(3, 8)):
                nx = rng.uniform(0, 10)
                ny = rng.uniform(0, 10)
                ax.plot(nx, ny, ".", color="#cccccc", markersize=2)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskc"
    os.makedirs(out_dir, exist_ok=True)
    env = NumericCommonsenseVisualQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[numeric_commonsense_visual L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"numeric_commonsense_visual_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[numeric_commonsense_visual L{level} s{s}] A={env._answer}")
