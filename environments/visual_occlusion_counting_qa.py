"""
Visual Occlusion Counting QA (v4 G22, for counting).

Targets: counting -2.99 (dog-paths, metal-strip counts, kangaroo
circles — visual counting with overlap/occlusion).

Task: render N overlapping shapes (circles, rectangles, triangles) where
some are partially hidden behind others. Ask how many distinct shapes of
a specific type are in the image.

Reward: exact integer.

Level axes:
  A) N shapes: 3 at L0 -> 12 at L9
  B) Occlusion: none at L0 -> heavy at L6+
  C) Multi-type at L4+ (mixed circles + rectangles)
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

_TEMPLATES = [
    "Count the total number of {target_type} in the image. Some may overlap or be partially hidden. Put the integer in <answer>...</answer>.",
    "How many {target_type} are shown? Some are partially behind others. Integer in <answer>...</answer>.",
    "Count every {target_type} in the figure, including partially-occluded ones. Integer in <answer>...</answer>.",
    "Number of {target_type}? Put integer in <answer>...</answer>.",
    "How many distinct {target_type}? Integer in <answer>...</answer>.",
    "Count the {target_type} (including hidden parts). Integer in <answer>...</answer>.",
    "Total {target_type} in the image? Integer in <answer>...</answer>.",
    "How many {target_type} can be identified? Integer in <answer>...</answer>.",
    "Count all {target_type} visible or partially visible. Integer in <answer>...</answer>.",
    "Distinct {target_type} count? Integer in <answer>...</answer>.",
    "Identify every {target_type} and count them. Integer in <answer>...</answer>.",
    "Number of {target_type} shown? Integer in <answer>...</answer>.",
    "Count the {target_type} in the scene. Integer in <answer>...</answer>.",
    "{target_type} count (including occluded)? Integer in <answer>...</answer>.",
    "Total distinct {target_type}? Integer in <answer>...</answer>.",
    "Count every {target_type} in the image. Integer in <answer>...</answer>.",
]

class VisualOcclusionCountingQA(StandaloneVisualEnv):
    ENV_NAME = "visual_occlusion_counting"
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R3: softened — was n_shapes=3+level (L9 had 12 shapes)
        # with overlap=0.4 (heavy occlusion) at L7+ (L9 dropped to 0.73).
        # Cap shapes at 9 and overlap at 0.30 so heavily-occluded targets
        # remain countable.
        level = max(0, min(level, 9))
        n_shapes = min(3 + level, 9)
        if level <= 3:
            shape_types = ["circle"]
            overlap = 0.1
        elif level <= 6:
            shape_types = ["circle", "rectangle"]
            overlap = 0.25
        else:
            shape_types = ["circle", "rectangle", "triangle"]
            overlap = 0.30
        return {"n_shapes": n_shapes, "shape_types": shape_types,
                 "overlap": overlap}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 277)
        self._primary_complexity_feature = level

        # Generate shape positions with controlled overlap
        shapes = []  # (type, x, y, size, color)
        W, H = 10, 8
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
                    "#9b59b6", "#1abc9c", "#34495e"]
        for _ in range(cfg["n_shapes"]):
            st = rng.choice(cfg["shape_types"])
            for _ in range(20):  # retry placement
                x = rng.uniform(1, W - 1)
                y = rng.uniform(1, H - 1)
                size = rng.uniform(0.5, 1.1)
                # Check overlap
                if cfg["overlap"] < 1.0:
                    min_dist = size + 0.4
                    conflict = any(
                        math.hypot(x - s[1], y - s[2]) < min_dist * (1 - cfg["overlap"])
                        for s in shapes)
                    if conflict:
                        continue
                shapes.append((st, x, y, size, rng.choice(colors)))
                break

        if not shapes:
            return None

        # Target: pick a specific type (or "all shapes" if only one type)
        if len(cfg["shape_types"]) == 1:
            target_type = cfg["shape_types"][0] + "s"
            count = len(shapes)
        else:
            target_plural = {"circle": "circles", "rectangle": "rectangles",
                              "triangle": "triangles"}
            target = rng.choice(cfg["shape_types"])
            count = sum(1 for s in shapes if s[0] == target)
            target_type = target_plural[target]

        answer = str(count)
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(target_type=target_type)

        img = self._render(shapes, W, H, rng)
        return q, answer, img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, shapes, W, H, rng):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw shapes in zorder such that later ones occlude earlier
        for i, (st, x, y, size, color) in enumerate(shapes):
            if st == "circle":
                patch = mpatches.Circle((x, y), size,
                                          fc=color, ec="black", lw=1.5,
                                          alpha=0.95, zorder=i+1)
            elif st == "rectangle":
                patch = mpatches.Rectangle((x - size, y - size), 2 * size, 1.5 * size,
                                            fc=color, ec="black", lw=1.5,
                                            alpha=0.95, zorder=i+1)
            else:  # triangle
                pts = [(x, y + size),
                       (x - size, y - size * 0.5),
                       (x + size, y - size * 0.5)]
                patch = mpatches.Polygon(pts, fc=color, ec="black", lw=1.5,
                                          alpha=0.95, zorder=i+1)
            ax.add_patch(patch)
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_voc"
    os.makedirs(out_dir, exist_ok=True)
    env = VisualOcclusionCountingQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 131
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[voc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/voc_s{s}_L{level}.png")
            print(f"[voc L{level} s{s}] A={env._answer}")
