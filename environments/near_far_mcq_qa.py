"""
Near-Far MCQ QA (batch 3, 2026-04-14).

Target: visual-perception Relative_Depth, core-vision-bench-3D Depth. A pseudo-3D alley / room
scene with labelled marker spheres at different depths. The model picks
the nearest OR the farthest (randomised so letter balance is preserved).

Format: constant MCQ letter A/B/C/D.

Difficulty axes:
  A) Pattern A: n_objects (3..6).
  B) Pattern D: depth separation shrinks.
  C) Pattern G: size-hint suppression at higher levels.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class NearFarMcqQA(StandaloneVisualEnv):
    ENV_NAME = "near_far_mcq"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Empirically, more objects = more landmarks = EASIER (L3=0.80 with
        # 4 objects > L0=0.50 with 3 objects). Fix: more objects at low
        # levels, fewer at high. Size cue strength clamped to 0.35 minimum
        # so objects remain visible at all levels.
        n_objects = max(3, 5 - level // 4)          # 5,5,5,5,4,4,3,3,3,3
        # min_gap must allow n_objects to fit in [0.6, 7.4] range (span=6.8)
        max_gap = 6.8 / max(1, n_objects - 1) - 0.05
        base_gap = max(0.15, 1.2 - 0.12 * level)
        return {
            "n_objects": n_objects,
            "min_gap": min(base_gap, max_gap),
            "size_cue_strength": max(0.35, 1.0 - 0.08 * level),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_objects"] * 10 - int(cfg["min_gap"] * 10)

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_objects"]
        depths = []
        for _ in range(n):
            for _ in range(50):
                d = rng.uniform(0.6, 7.4)
                if all(abs(d - e) >= cfg["min_gap"] for e in depths):
                    depths.append(d)
                    break
        if len(depths) != n:
            return None

        labels = [chr(ord("A") + i) for i in range(n)]
        xs = [rng.uniform(-3.5, 3.5) for _ in range(n)]

        # Pick Near or Far question (so we keep 4-way letter balanced)
        ask = rng.choice(["nearest", "farthest"])
        if ask == "nearest":
            target = min(range(n), key=lambda i: depths[i])
        else:
            target = max(range(n), key=lambda i: depths[i])
        answer = labels[target]

        descriptor = "closest to" if ask == "nearest" else "farthest from"
        templates = [
            f"Four or more labelled objects sit in a 3D scene at different depths. Which of the labelled objects is {descriptor} the camera? Answer with the single letter label.",
            f"In the 3D room scene, several labelled spheres are placed at different distances. Which sphere is {descriptor} the viewer? Answer with the letter label.",
            f"Several labelled balls are positioned at varying depths in the scene. Identify which ball is {descriptor} the camera. Reply with one letter.",
            f"Look at the labelled objects placed in the perspective scene. Pick the label of the object {descriptor} the viewing camera.",
            f"The room contains labelled markers at different depths. Which marker label corresponds to the object {descriptor} the camera?",
        ]
        q = rng.choice(templates)

        image = self._render(labels, depths, xs, cfg, rng)
        return q, answer, image

    def _render(self, labels, depths, xs, cfg, rng=None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]
        if rng is None:
            rng = random.Random(0)

        fig, ax = plt.subplots(figsize=(8 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        # Vary floor and wall colors per render
        floor_color = rng.choice(["#eaeaea", "#e3e8ed", "#f0e6d2", "#d4e7e0", "#e7d8d3"])
        wall_color = rng.choice(["#cfd8dc", "#b0bec5", "#bcaaa4", "#a5d6a7", "#90caf9"])
        # Perspective floor (two walls + floor)
        floor = mpatches.Polygon(
            [(-7, 0), (7, 0), (3.0, 5.0), (-3.0, 5.0)],
            facecolor=floor_color, edgecolor="#555", linewidth=1.2)
        ax.add_patch(floor)
        # Side walls
        ax.add_patch(mpatches.Polygon(
            [(-7, 0), (-3.0, 5.0), (-3.0, 7.0), (-7, 5.0)],
            facecolor=wall_color, edgecolor="#555", linewidth=1.0))
        ax.add_patch(mpatches.Polygon(
            [(7, 0), (3.0, 5.0), (3.0, 7.0), (7, 5.0)],
            facecolor=wall_color, edgecolor="#555", linewidth=1.0))

        n = len(labels)
        order_rev = sorted(range(n), key=lambda i: -depths[i])
        for i in order_rev:
            d = depths[i]
            y = 0.55 * d
            # Size proportional to (1 - depth/9) but muted at high level
            raw = 1.0 - d / 9.0
            size = 0.2 + 0.5 * (raw if raw > 0.1 else 0.1)
            size *= cfg["size_cue_strength"]
            # Floor on rendered size to keep labels readable even at L9.
            size = max(size, 0.25)
            cx = xs[i] * (1.0 - 0.09 * d)
            color = palette[i % len(palette)]
            circ = mpatches.Circle((cx, y + size / 2), size,
                                   facecolor=color, edgecolor="#222",
                                   linewidth=1.2)
            ax.add_patch(circ)
            # Pick contrasting label color based on ball luminance.
            try:
                h = color.lstrip('#')
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = "#000" if lum > 150 else "#fff"
            except Exception:
                txt_color = "#fff"
            ax.text(cx, y + size / 2, labels[i],
                    fontsize=fs + 2, fontweight="bold",
                    color=txt_color, ha="center", va="center")

        ax.set_xlim(-7, 7)
        ax.set_ylim(-0.5, 7.2)
        ax.set_aspect("equal")
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = NearFarMcqQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[near_far_mcq L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"near_far_mcq_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[near_far_mcq L{level} s{s}] A={env._answer}")
