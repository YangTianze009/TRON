"""
Visual Counting Ultra Easy QA environment.

Goal: warmup rescue for counting. N large, well-separated, high-contrast
objects on white background. "How many red circles are in the image?"
Targets baseline counting — scaffolds counting,
visual-perception Counting, MMBench CP.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2-3 target objects, 0 distractors, 4 MCQ options separated by gap=2.
L1: 3 targets, 0 distractors, gap=2.
L2: 3-4 targets, 1 distractor, gap=2.
L3: 4 targets, 1 distractor, gap=2.
L4: 4-5 targets, 2 distractors, gap=1.
L5: 5 targets, 2 distractors, gap=1.
L6: 5-6 targets, 3 distractors, gap=1.
L7: 6-7 targets, 3 distractors, gap=1, tight options.
L8: 7-8 targets, 4 distractors, gap=1, tight options.
L9: 8-10 targets, 5 distractors, gap=1, tightest options.

======================================================================
Diversity axes
======================================================================
  1. Target shape: 4 shapes (circle, square, triangle, star)
  2. Target color: 6 named colors
  3. Distractor shape and color: different from target
  4. Object positions: jittered grid
  5. MCQ option set generation: gap-based
  6. Question phrasing: 3 variants
  7. Title variants: 4
  8. Background color from _random_style
  9. Object outline color: 4 dark options
 10. Distractor placement density
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

_OBJ_SHAPES = ["circle", "square", "triangle", "star", "diamond"]
_OBJ_COLORS = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "purple": "#9b59b6",
    "orange": "#e67e22",
}

class VisualCountingUltraEasyQA(StandaloneVisualEnv):
    ENV_NAME = "visual_counting_ultra_easy"

    # ------------------------------------------------------------------ #
    # Per-level config
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return dict(n_target_range=(2, 3), n_distractor=0, gap=2, tight=False)
        if level == 1:
            return dict(n_target_range=(3, 3), n_distractor=0, gap=2, tight=False)
        if level == 2:
            return dict(n_target_range=(3, 4), n_distractor=1, gap=2, tight=False)
        if level == 3:
            return dict(n_target_range=(4, 4), n_distractor=1, gap=2, tight=False)
        if level == 4:
            return dict(n_target_range=(4, 5), n_distractor=2, gap=1, tight=False)
        if level == 5:
            return dict(n_target_range=(5, 5), n_distractor=2, gap=1, tight=False)
        if level == 6:
            return dict(n_target_range=(5, 6), n_distractor=3, gap=1, tight=False)
        if level == 7:
            return dict(n_target_range=(6, 7), n_distractor=3, gap=1, tight=True)
        if level == 8:
            return dict(n_target_range=(7, 8), n_distractor=4, gap=1, tight=True)
        return dict(n_target_range=(8, 10), n_distractor=5, gap=1, tight=True)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_target_range"][0]

        for _ in range(30):
            r = self._try_generate(sub_rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, sub_rng: random.Random,
                      cfg: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        target_shape = sub_rng.choice(_OBJ_SHAPES)
        target_color_name = sub_rng.choice(list(_OBJ_COLORS.keys()))
        target_color = _OBJ_COLORS[target_color_name]

        n_t = sub_rng.randint(*cfg["n_target_range"])
        n_d = cfg["n_distractor"]

        total = n_t + n_d
        coords = self._place_objects(sub_rng, total)
        if coords is None:
            return None

        objects = []
        for i in range(n_t):
            objects.append({
                "shape": target_shape,
                "color": target_color,
                "x": coords[i][0],
                "y": coords[i][1],
            })
        for i in range(n_d):
            alt_shape = sub_rng.choice([s for s in _OBJ_SHAPES if s != target_shape])
            alt_color_name = sub_rng.choice(
                [c for c in _OBJ_COLORS if c != target_color_name])
            objects.append({
                "shape": alt_shape,
                "color": _OBJ_COLORS[alt_color_name],
                "x": coords[n_t + i][0],
                "y": coords[n_t + i][1],
            })
        sub_rng.shuffle(objects)

        gt = n_t
        gap = cfg["gap"]
        if cfg.get("tight"):
            pool = [gt - 1, gt + 1, gt - 2, gt + 2]
        else:
            pool = [gt - 2 * gap, gt - gap, gt + gap, gt + 2 * gap]
        pool = [p for p in pool if p >= 0 and p != gt]
        sub_rng.shuffle(pool)
        distractors = []
        for p in pool:
            if p not in distractors and p != gt:
                distractors.append(p)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            return None

        options_vals = [gt] + distractors[:3]
        sub_rng.shuffle(options_vals)
        if options_vals.count(gt) > 1:
            return None
        answer_letter = chr(ord("A") + options_vals.index(gt))
        options_str = [str(v) for v in options_vals]

        stem = self._rng.choice([
            f"How many {target_color_name} {target_shape}s are in the image?",
            f"Count the number of {target_color_name} {target_shape}s shown in the image.",
            f"The image contains several shapes. How many of them are {target_color_name} {target_shape}s?",
        ])
        q = (
            f"{stem}\n"
            + "\n".join(f"  ({chr(ord('A') + i)}) {options_str[i]}" for i in range(4))
            + "\nAnswer with the single letter of the correct option."
        )
        image = self._render(objects, target_color_name, target_shape,
                             options_str, sub_rng)
        return q, answer_letter, image

    def _place_objects(self, rng: random.Random, n: int):
        side = int(math.ceil(math.sqrt(n * 1.4)))
        side = max(side, 2)
        cells = []
        for r in range(side):
            for c in range(side):
                cells.append((c + 0.5, r + 0.5))
        if len(cells) < n:
            return None
        rng.shuffle(cells)
        pts = []
        for (cx, cy) in cells[:n]:
            jx = rng.uniform(-0.2, 0.2)
            jy = rng.uniform(-0.2, 0.2)
            pts.append((cx + jx, cy + jy))
        return pts

    def _draw(self, ax, cx, cy, obj, size, edge_col):
        shape = obj["shape"]
        color = obj["color"]
        if shape == "circle":
            p = mpatches.Circle((cx, cy), size, facecolor=color,
                                 edgecolor=edge_col, linewidth=1.1)
        elif shape == "square":
            p = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                    facecolor=color, edgecolor=edge_col,
                                    linewidth=1.1)
        elif shape == "triangle":
            p = mpatches.Polygon([(cx, cy + size),
                                   (cx - size, cy - size),
                                   (cx + size, cy - size)],
                                  closed=True, facecolor=color,
                                  edgecolor=edge_col, linewidth=1.1)
        elif shape == "diamond":
            p = mpatches.Polygon([(cx, cy + size), (cx + size, cy),
                                   (cx, cy - size), (cx - size, cy)],
                                  closed=True, facecolor=color,
                                  edgecolor=edge_col, linewidth=1.1)
        elif shape == "star":
            pts = []
            for i in range(10):
                a = math.radians(36 * i - 90)
                r = size if i % 2 == 0 else size * 0.45
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                  edgecolor=edge_col, linewidth=1.1)
        else:
            p = mpatches.Circle((cx, cy), size, facecolor=color,
                                 edgecolor=edge_col, linewidth=1.1)
        ax.add_patch(p)

    def _render(self, objects, color_name, shape_name, options, sub_rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(9.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        edge_col = sub_rng.choice(["#111", "#1a1a1a", "#222", "#0a0a0a"])
        obj_size = sub_rng.uniform(0.30, 0.42)

        xs = [o["x"] for o in objects]
        ys = [o["y"] for o in objects]
        if xs and ys:
            pad = 1.0
            ax_f.set_xlim(min(xs) - pad, max(xs) + pad)
            ax_f.set_ylim(min(ys) - pad, max(ys) + pad)
        for o in objects:
            self._draw(ax_f, o["x"], o["y"], o, obj_size, edge_col)

        title_l = sub_rng.choice(["Counting", "Count the shapes", "Shapes", "Find the targets"])
        ax_f.set_title(title_l, fontsize=fs + 1, family=ff)

        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Query:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        ax_t.text(0.3, y,
                  f"How many {color_name} {shape_name}s are in the image?",
                  fontsize=fs, family=ff, ha="left", va="top",
                  color="#1a1a1a")
        y -= 0.85
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs + 1, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.7

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    env = VisualCountingUltraEasyQA()
    for level in (0, 3, 6, 9):
        gts = {}
        for seed in range(10):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                continue
            gts[env._answer] = gts.get(env._answer, 0) + 1
        print(f"L{level} GT: {gts}")
