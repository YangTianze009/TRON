"""
Depth Order QA (batch 3, 2026-04-14).

Target: visual-perception Relative_Depth / core-vision-bench-3D Depth. A pseudo-3D scene with
several colored ground-placed boxes is drawn using parallel projection
(isometric). The model must list them in order from nearest to farthest
(or pick the farthest).

Format: constant short-answer (comma-separated labels, e.g. "B,C,A,D") at
all levels — we normalise comparison.

Difficulty axes:
  A) Pattern A: n_objects (3..6).
  B) Pattern D: depth spacing shrinks so some objects at similar depths.
  C) Pattern G: size-cue consistency randomised at higher levels.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class DepthOrderQA(StandaloneVisualEnv):
    ENV_NAME = "depth_order"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Empirically, more objects = MORE landmarks = EASIER for the model
        # (L6 with 6 objects scored 0.80 vs L3 with 4 objects at 0.60).
        # Fix: use FEWER objects at higher levels + tighter depth gaps.
        # Also: inconsistent size cues make it harder (objects that are far
        # can appear large, breaking the size-depth heuristic).
        n_objects = max(3, 6 - level // 2)         # 6,6,5,5,4,4,3,3,3,3
        return {
            "n_objects": n_objects,
            "depth_min_gap": max(0.15, 1.4 - 0.14 * level),
            "consistent_size": level <= 3,           # inconsistent from L4+
            "n_distractors": level // 3,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_objects"]

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_objects"]
        # Depths in [0, 8]
        depths = []
        for _ in range(n):
            d = rng.uniform(0.5, 7.5)
            # enforce min gap
            if any(abs(d - e) < cfg["depth_min_gap"] for e in depths):
                for _ in range(50):
                    d = rng.uniform(0.5, 7.5)
                    if all(abs(d - e) >= cfg["depth_min_gap"] for e in depths):
                        break
            depths.append(d)

        labels = [chr(ord("A") + i) for i in range(n)]
        # x positions to spread the objects laterally
        xs = [rng.uniform(-3.5, 3.5) for _ in range(n)]

        # Order labels by depth, nearest (smallest depth) first
        order_idx = sorted(range(n), key=lambda i: depths[i])
        nearest_label = labels[order_idx[0]]
        farthest_label = labels[order_idx[-1]]

        # Multiple question types: nearest / farthest / pair_compare
        # higher level => more variants
        if level >= 5:
            qtype = rng.choice(["nearest", "farthest", "pair_compare"])
        else:
            qtype = rng.choice(["nearest", "farthest"])

        if qtype == "nearest":
            answer = nearest_label
            templates = [
                "Several labelled boxes are placed at different depths in the scene (nearer objects are in the foreground). Which labelled box is closest to the camera? Answer with a single letter.",
                "Multiple labelled cubes appear at different distances from the viewer. Which one is closest (in the foreground)? Answer with a single letter.",
                "The figure shows boxes at varying depths. Identify the box that is nearest to the camera. Answer with a single letter.",
            ]
            q = rng.choice(templates)
        elif qtype == "farthest":
            answer = farthest_label
            templates = [
                "Several labelled boxes are placed at different depths in the scene. Which labelled box is FARTHEST from the camera (deepest in the scene)? Answer with a single letter.",
                "Identify the box deepest in the scene (farthest from the viewer). Answer with a single letter.",
                "Which labelled cube appears farthest away (closest to the horizon line)? Answer with a single letter.",
            ]
            q = rng.choice(templates)
        else:  # pair_compare
            i, j = rng.sample(range(n), 2)
            la, lb = labels[i], labels[j]
            ans = la if depths[i] < depths[j] else lb
            templates = [
                f"Compare the two boxes labelled {la} and {lb}. Which one is CLOSER to the camera? Answer with a single letter ({la} or {lb}).",
                f"Between boxes {la} and {lb}, which is in the foreground (nearer)? Answer with the single letter.",
            ]
            q = rng.choice(templates)
            answer = ans

        image = self._render(labels, depths, xs, cfg, rng)
        return q, answer, image

    def _render(self, labels, depths, xs, cfg, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        fig, ax = plt.subplots(figsize=(8 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        # Draw ground plane (trapezoid)
        ground = mpatches.Polygon(
            [(-7, 0), (7, 0), (4.5, 5), (-4.5, 5)],
            facecolor="#e9ebee", edgecolor="#555", linewidth=1.2)
        ax.add_patch(ground)
        # Horizon line
        ax.axhline(5.0, color="#777", linewidth=1.0, linestyle="--")

        # Draw objects: depth determines vertical y (nearer = lower y;
        # farther = higher y toward horizon) and size (farther → smaller).
        n = len(labels)
        order_rev = sorted(range(n), key=lambda i: -depths[i])
        for i in order_rev:
            d = depths[i]
            # project: y = 0.6*d, x scale = (1 - 0.1*d)
            y = 0.55 * d
            scale = max(0.35, 1.0 - 0.1 * d)
            if not cfg["consistent_size"]:
                scale *= rng.uniform(0.75, 1.25)
            half = 0.45 * scale
            cx = xs[i] * (1.0 - 0.09 * d)
            # Box: draw as a quadrilateral with a top face for 3D feel
            color = palette[i % len(palette)]
            rect = mpatches.Rectangle(
                (cx - half, y), 2 * half, 1.2 * scale,
                facecolor=color, edgecolor="#222", linewidth=1.2)
            ax.add_patch(rect)
            # top face: parallelogram
            top = mpatches.Polygon(
                [(cx - half, y + 1.2 * scale),
                 (cx - half + 0.25 * scale, y + 1.5 * scale),
                 (cx + half + 0.25 * scale, y + 1.5 * scale),
                 (cx + half, y + 1.2 * scale)],
                facecolor=tuple(min(1.0, c + 0.15) for c in self._hex_to_rgb(color)),
                edgecolor="#222", linewidth=1.0)
            ax.add_patch(top)

            # White bold text with black outline so labels stay legible on
            # every box color (including pale yellows) and through occlusion.
            ax.text(cx, y + 0.6 * scale, labels[i],
                    fontsize=fs + 4, fontweight="bold", ha="center",
                    va="center", color="#fff",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="#222")],
                    zorder=20)

        ax.set_xlim(-7, 7)
        ax.set_ylim(-0.5, 6)
        ax.set_aspect("equal")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _hex_to_rgb(h: str):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = DepthOrderQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[depth_order L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"depth_order_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[depth_order L{level} s{s}] A={env._answer}")
