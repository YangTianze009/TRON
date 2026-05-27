"""
Composite Area Decomposition QA (v4 G24, for metric geometry - area).

Targets: metric geometry - area -2.40.

Task: an L-shape / cross-shape / stepped polygon composed of rectangles,
with dimension labels. Ask for the total area.

Reward: numeric within 2% relative tolerance.

Level axes:
  A) Rect count: 2 at L0-3, 3 at L4-6, 4 at L7+
  B) Shape type: L-shape, T-shape, cross, stepped
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_wemath_mcq

_TEMPLATES = [
    "The compound shape is formed by rectangles as shown. Given the labeled dimensions, compute the total area. Put the numeric answer in <answer>...</answer>.",
    "Find the total area of the composite shape (in unit²). Put in <answer>...</answer>.",
    "Compute the area of the shown compound shape. Integer/decimal in <answer>...</answer>.",
    "Total area of the composite figure? Put in <answer>...</answer>.",
    "Decompose and sum: total area of the shape? Put in <answer>...</answer>.",
    "The figure is a union of rectangles. Find total area. Put in <answer>...</answer>.",
    "Compute composite area. Put in <answer>...</answer>.",
    "Area of the compound rectilinear figure? Put in <answer>...</answer>.",
    "Sum the rectangle areas to find total area. Put in <answer>...</answer>.",
    "Find area of composite shape. Put in <answer>...</answer>.",
    "What is the total area? Put in <answer>...</answer>.",
    "Decompose to rectangles; total area? Put in <answer>...</answer>.",
    "Composite-shape area? Put in <answer>...</answer>.",
    "Compute total area of the figure. Put in <answer>...</answer>.",
    "Total enclosed area? Put in <answer>...</answer>.",
    "Sum rectangle areas for the compound shape. Put in <answer>...</answer>.",
]

class CompositeAreaDecompositionQA(StandaloneVisualEnv):
    ENV_NAME = "composite_area_decomposition"
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            shape = "L"
        elif level <= 6:
            shape = rng_choice = "T"
        else:
            shape = "cross"
        return {"shape": shape}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 479)
        self._primary_complexity_feature = level

        if level <= 3:
            shape = "L"
        elif level <= 6:
            shape = "T"
        else:
            shape = "cross"

        # Build shape, compute total area
        if shape == "L":
            w1 = rng.randint(4, 10)
            h1 = rng.randint(3, 6)
            w2 = rng.randint(2, w1 - 2)
            h2 = rng.randint(2, 5)
            total = w1 * h1 + w2 * h2
            img = self._render_L(w1, h1, w2, h2, rng)
        elif shape == "T":
            w1 = rng.randint(6, 12)
            h1 = rng.randint(2, 4)
            w2 = rng.randint(2, 4)
            h2 = rng.randint(3, 6)
            total = w1 * h1 + w2 * h2
            img = self._render_T(w1, h1, w2, h2, rng)
        else:  # cross
            arm_long = rng.randint(6, 10)
            arm_short = rng.randint(2, 4)
            arm_thick = rng.randint(2, 3)
            total = arm_long * arm_thick + (arm_short * 2) * arm_thick
            img = self._render_cross(arm_long, arm_short, arm_thick, rng)

        answer = str(total)
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx]
        # 2026-05-04 WeMath alignment: 50% → 5-way MCQ with E="No correct
        # answer" + "cm²" unit (area).
        unit_rng = random.Random((self.seed or 0) * 17 + 7331)
        q, answer = maybe_to_wemath_mcq(
            q, answer, unit_rng, prob=0.5, unit="cm²", n_options=5)
        return q, answer, img

    def _render_L(self, w1, h1, w2, h2, rng):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        total_w = w1
        total_h = h1 + h2
        ax.set_xlim(-1, total_w + 2)
        ax.set_ylim(-1, total_h + 2)
        ax.set_aspect("equal")
        ax.axis("off")
        # Bottom rectangle (w1 x h1)
        ax.add_patch(mpatches.Rectangle((0, 0), w1, h1, fc="#d6eaf8",
                                         ec="black", lw=2))
        # Upper rectangle (w2 x h2) stacked on left
        ax.add_patch(mpatches.Rectangle((0, h1), w2, h2, fc="#fadbd8",
                                         ec="black", lw=2))
        # Labels
        ax.annotate(f"{w1}", xy=(w1 / 2, -0.3), fontsize=12, ha="center",
                    fontweight="bold")
        ax.annotate(f"{h1}", xy=(-0.3, h1 / 2), fontsize=12, ha="right",
                    fontweight="bold")
        ax.annotate(f"{w2}", xy=(w2 / 2, h1 + h2 + 0.3), fontsize=12,
                    ha="center", fontweight="bold")
        ax.annotate(f"{h2}", xy=(w2 + 0.3, h1 + h2 / 2), fontsize=12,
                    fontweight="bold")
        return self.fig_to_pil(fig)

    def _render_T(self, w1, h1, w2, h2, rng):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, w1 + 2); ax.set_ylim(-1, h1 + h2 + 2)
        ax.set_aspect("equal")
        ax.axis("off")
        # Top bar
        ax.add_patch(mpatches.Rectangle((0, h2), w1, h1, fc="#d6eaf8",
                                         ec="black", lw=2))
        # Stem
        stem_x = (w1 - w2) / 2
        ax.add_patch(mpatches.Rectangle((stem_x, 0), w2, h2, fc="#fadbd8",
                                         ec="black", lw=2))
        # Labels
        ax.annotate(f"{w1}", xy=(w1 / 2, h2 + h1 + 0.3), fontsize=12, ha="center",
                    fontweight="bold")
        ax.annotate(f"{h1}", xy=(-0.3, h2 + h1 / 2), fontsize=12,
                    fontweight="bold")
        ax.annotate(f"{w2}", xy=(stem_x + w2 / 2, -0.4), fontsize=12,
                    ha="center", fontweight="bold")
        ax.annotate(f"{h2}", xy=(stem_x - 0.3, h2 / 2), fontsize=12,
                    fontweight="bold", ha="right")
        return self.fig_to_pil(fig)

    def _render_cross(self, long, short, thick, rng):
        # BUGFIX 2026-04-24: vertical arm height must be short*2 + thick
        # (matching area formula), and arm_short must be labeled on the image.
        # 2026-05-04: add dimension arrows to disambiguate "long" label
        # (was floating below horizontal arm; could be confused with vertical
        # arm bottom width).
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        vert_h = short * 2 + thick
        total_x = long + 2
        total_y = max(vert_h, long) + 2
        ax.set_xlim(-2, total_x); ax.set_ylim(-2, total_y)
        ax.set_aspect("equal"); ax.axis("off")
        # Center horizontal arm vertically relative to vertical arm
        horiz_y = vert_h / 2 - thick / 2
        # Horizontal arm
        ax.add_patch(mpatches.Rectangle((0, horiz_y), long, thick,
                                         fc="#d6eaf8", ec="black", lw=2))
        # Vertical arm (height = short*2 + thick, spans the cross properly)
        ax.add_patch(mpatches.Rectangle((long / 2 - thick / 2, 0), thick, vert_h,
                                         fc="#fadbd8", ec="black", lw=2,
                                         alpha=0.7))
        # 2026-05-04: dimension arrows for "long" so it's unambiguous which
        # segment "10" labels (the horizontal arm length, not vertical arm).
        dim_y = -1.2  # well below figure
        ax.annotate("", xy=(long, dim_y), xytext=(0, dim_y),
                    arrowprops=dict(arrowstyle="<->", color="#1a1a1a", lw=1.5))
        # Tick marks at endpoints
        ax.plot([0, 0], [dim_y - 0.15, dim_y + 0.15], color="#1a1a1a", lw=1.5)
        ax.plot([long, long], [dim_y - 0.15, dim_y + 0.15], color="#1a1a1a", lw=1.5)
        ax.annotate(f"{long}", xy=(long / 2, dim_y - 0.5),
                    fontsize=12, ha="center", fontweight="bold")
        ax.annotate(f"{thick}", xy=(-0.3, horiz_y + thick / 2), fontsize=12,
                    ha="right", fontweight="bold")
        # Label arm_short on the vertical arm's upper half (above the horizontal arm)
        ax.annotate(f"{short}", xy=(long / 2 + thick / 2 + 0.3,
                                      horiz_y + thick + short / 2),
                    fontsize=12, fontweight="bold")
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip()
        for sym in ["unit²", "unit2", "units", "unit", "square units"]:
            pred = pred.replace(sym, "").strip()
            gt = gt.replace(sym, "").strip()
        try:
            p = float(pred); g = float(gt)
            return abs(p - g) / max(abs(g), 1e-9) < 0.02 or abs(p - g) < 0.5
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_cad"
    os.makedirs(out_dir, exist_ok=True)
    env = CompositeAreaDecompositionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 203
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[cad L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/cad_s{s}_L{level}.png")
            print(f"[cad L{level} s{s}] A={env._answer}")
