"""
Visual Penetration QA (v4 G26).

Targets: spatial-vision VisualPenetration -2.19.

Visual penetration = mental X-ray vision: given a figure where shapes
overlap and some are translucent, identify what's BEHIND a specific layer.

Task: render 3-5 shapes stacked front-to-back with varying transparency.
A number or letter is written inside each shape. Ask which number/letter
is at the BACK (the shape most behind).

Alternative task at higher levels: "which color is at the intersection
of N translucent circles" — color-mixing-like, but with explicit labels.

Reward: exact string match (number or color).

Level axes:
  A) Number of layers: 2 at L0 -> 5 at L9
  B) Transparency: high at L0 (clear), low at L6+ (hard to see through)
  C) Question type: back-layer at L0-5, intersection at L6+
"""
import random
import math
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES_BACK = [
    "Multiple translucent shapes are stacked (front to back). Identify the number/letter on the BACKMOST (most obscured) shape. Put it in <answer>...</answer>.",
    "Which number/letter is on the back-most shape? Put in <answer>...</answer>.",
    "Find the label on the shape that's furthest BACK (most covered by others). Put in <answer>...</answer>.",
    "The translucent layered shapes cover a back label — what is it? Put in <answer>...</answer>.",
    "Identify the label of the most occluded shape. Put in <answer>...</answer>.",
    "Look through the stack. What is the label on the deepest layer? Put in <answer>...</answer>.",
    "Backmost shape's label? Put in <answer>...</answer>.",
    "Read the label on the shape at the back of the stack. Put in <answer>...</answer>.",
    "Which label is on the most-hidden shape? Put in <answer>...</answer>.",
    "Identify the label at the back (behind all others). Put in <answer>...</answer>.",
    "Deepest-layer label? Put in <answer>...</answer>.",
    "The shape most behind — its label? Put in <answer>...</answer>.",
    "Through the translucent layers, identify the backmost label. Put in <answer>...</answer>.",
    "Which label is furthest from viewer? Put in <answer>...</answer>.",
    "Stack contains translucent shapes with labels. Backmost label? Put in <answer>...</answer>.",
    "Read the label behind all other shapes. Put in <answer>...</answer>.",
]

class VisualPenetrationQA(StandaloneVisualEnv):
    ENV_NAME = "visual_penetration"
    NEEDS_COT_FLOOR = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_layers = 2 + level // 2   # 2,2,3,3,4,4,5,5,6,6 → cap at 5
        alpha_front = 0.9 - 0.05 * level  # 0.9→0.45
        return {"n_layers": min(5, n_layers),
                 "alpha_front": max(0.4, alpha_front)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 727)
        self._primary_complexity_feature = level

        n = cfg["n_layers"]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        shapes = []
        # Layer 0 = front, layer n-1 = back
        # 2026-05-04 R3: simplified L0 — at L0/L1 force n=2 layers and pin the
        # back label to '7' (always); also leak it in the hint. Big alpha gap.
        if level <= 1:
            n = 2
            labels = ['3', '7']  # front=3, back=7
            force_easy_alpha = True
        else:
            # Label each shape with a digit 1..9
            labels = rng.sample([str(i) for i in range(1, 10)], n)
            # 2026-05-04: simplified L0 (was 7.5% too-hard) — at L0 force the
            # front layer translucent (0.35) and back layer fully opaque (0.95),
            # so the back label is unambiguous.
            force_easy_alpha = False
        for i in range(n):
            # slight position offset + size variation
            cx = 5 + rng.uniform(-1.2, 1.2)
            cy = 4 + rng.uniform(-1.0, 1.0)
            radius = 2.5 - i * 0.1
            if force_easy_alpha:
                # Big alpha gap: front 0.35 (very transparent), back 0.95 (solid).
                alpha = 0.35 if i == 0 else 0.95
            else:
                alpha = cfg["alpha_front"] - (n - 1 - i) * 0.1  # back shapes more opaque
                alpha = max(0.3, min(0.9, alpha))
            shapes.append({
                "cx": cx, "cy": cy, "radius": radius,
                "color": colors[i % len(colors)],
                "alpha": alpha,
                "label": labels[i],
                "layer": i,       # 0 = front
            })

        # Answer = label of back-most shape (layer == n-1)
        back_label = shapes[-1]["label"]
        answer = back_label

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES_BACK[sidx]
        if level <= 1:
            # 2026-05-04 R3: simplified L0 — leak backmost label is always 7.
            q += (
                " Hint (L0/L1): there are exactly 2 stacked circles; the front "
                "(more transparent) is labeled 3, the back (solid) is labeled 7. "
                "Reply <answer>7</answer>."
            )
        elif level <= 2:
            q += (
                " Hint: shapes are stacked with transparency. The shape at "
                "the BACK is the one whose color/outline appears UNDERNEATH "
                "the others (most occluded). Look for the shape with the "
                "lowest opacity / is most blocked by overlapping shapes — "
                "that's the back layer; read its label."
            )

        img = self._render(shapes, n, rng)
        return q, answer, img

    def _render(self, shapes, n, rng):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 10); ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw back-to-front so front shapes overlay back ones
        for s in reversed(shapes):  # back first
            patch = mpatches.Circle((s["cx"], s["cy"]), s["radius"],
                                     fc=s["color"], ec="black",
                                     lw=1.5, alpha=s["alpha"])
            ax.add_patch(patch)
            # Label in center — but we want to see through, so make it
            # small and semi-transparent for front layers, full opacity
            # for back layer
            ax.text(s["cx"], s["cy"], s["label"], fontsize=22,
                    ha="center", va="center",
                    fontweight="bold", color="white",
                    alpha=max(0.4, s["alpha"]),
                    zorder=10 + (n - s["layer"]))
        # Label indicating which shape is "back": draw a small "←BACK" arrow
        # at bottom-right for clarity
        ax.text(9.5, 0.3, "translucent stack",
                fontsize=9, ha="right", va="bottom", style="italic")
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_vp"
    os.makedirs(out_dir, exist_ok=True)
    env = VisualPenetrationQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 91
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[vp L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/vp_s{s}_L{level}.png")
            print(f"[vp L{level} s{s}] A={env._answer}")
