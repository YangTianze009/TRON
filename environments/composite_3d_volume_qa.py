"""
Composite 3D Volume QA (v4 G6, for image-only Volume + dynamic-math solid).

Targets:

Failure mode: model collapses to bare letter or wrong setup. Need to force
setup identification → decomposition → arithmetic chain.

Task: render a compound solid (e.g., cylinder on top of a prism, or a box
with a hemisphere cavity) with labeled dimensions. Ask for total volume.

Reward: numeric within 1% relative tolerance.

Level axes:
  A) Composition: 1 shape (L0), 2 stacked (L1-3), 2 subtracted (L4-6), 3+ (L7+)
  B) Shape mix: box+box at L0-2, add cylinder at L3-5, add cone/pyramid at L6+
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
    "The figure shows a compound solid built from {desc}. All dimensions are in cm. Compute the TOTAL volume (cm³). Show each component's volume, then sum. Put the numeric answer in <answer>...</answer>.",
    "A composite 3D solid is made by combining {desc}. What is its total volume in cm³? List component volumes, then sum. Put answer in <answer>...</answer>.",
    "The figure is a compound solid: {desc}. Compute total volume (cm³). Enumerate each piece; put sum in <answer>...</answer>.",
    "Find the total volume of the compound solid shown, composed of {desc}. Show each sub-volume. Put the sum in <answer>...</answer>.",
    "The compound solid consists of {desc}. Compute its total volume in cm³. List components, then put sum in <answer>...</answer>.",
    "A solid is formed from {desc}. What is the total volume? Show each piece's volume; put the sum in <answer>...</answer>.",
    "Compute the total volume of the compound solid ({desc}). Put the sum in <answer>...</answer>.",
    "The figure shows {desc} combined. Find the total volume in cm³. Show each sub-volume; final in <answer>...</answer>.",
    "Volume problem: solid = {desc}. Compute total volume. Enumerate and sum; put answer in <answer>...</answer>.",
    "Decompose the compound solid ({desc}) and find its total volume. Show all sub-volumes; put total in <answer>...</answer>.",
    "Total volume of the solid built from {desc}? Show component volumes, sum them, put in <answer>...</answer>.",
    "Find the compound solid's volume. Components: {desc}. List each volume; put the total in <answer>...</answer>.",
    "The solid is composed of {desc}. Compute total volume in cm³. Put in <answer>...</answer>.",
    "Given the compound solid ({desc}), compute its volume. Put in <answer>...</answer>.",
    "Sum the volumes of the components ({desc}) to get the total. Put in <answer>...</answer>.",
    "Compound solid from {desc}. Compute total volume. Put sum in <answer>...</answer>.",
]

class Composite3DVolumeQA(StandaloneVisualEnv):
    ENV_NAME = "composite_3d_volume"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            compositions = ["box_box"]
        elif level <= 5:
            compositions = ["box_box", "box_cylinder"]
        else:
            compositions = ["box_box", "box_cylinder", "box_sub_hemisphere",
                             "cylinder_cone"]
        return {"compositions": compositions, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 389)
        self._primary_complexity_feature = level

        comp = rng.choice(cfg["compositions"])
        # dim_lines: short labels printed on the figure so the figure is
        # informative about which sub-shape has which size.
        dim_lines = []
        if comp == "box_box":
            l1, w1, h1 = rng.randint(3, 8), rng.randint(3, 8), rng.randint(2, 6)
            l2, w2, h2 = rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 5)
            v1 = l1 * w1 * h1
            v2 = l2 * w2 * h2
            total = v1 + v2
            desc = (f"a rectangular prism with dimensions {l1}×{w1}×{h1} and a "
                    f"second rectangular prism with dimensions {l2}×{w2}×{h2} "
                    "stacked on top")
            dim_lines = [f"box1: {l1}×{w1}×{h1}", f"box2: {l2}×{w2}×{h2}"]
        elif comp == "box_cylinder":
            l1, w1, h1 = rng.randint(3, 8), rng.randint(3, 8), rng.randint(2, 4)
            r = rng.randint(2, 4)
            h2 = rng.randint(2, 5)
            v1 = l1 * w1 * h1
            v2 = math.pi * r * r * h2
            total = v1 + v2
            desc = (f"a rectangular prism {l1}×{w1}×{h1} with a cylinder of "
                    f"radius {r} and height {h2} on top")
            dim_lines = [f"box: {l1}×{w1}×{h1}", f"cyl: r={r}, h={h2}"]
        elif comp == "box_sub_hemisphere":
            l1, w1, h1 = rng.randint(4, 10), rng.randint(4, 10), rng.randint(3, 6)
            r = rng.randint(1, min(3, w1 // 2))
            v1 = l1 * w1 * h1
            v2 = (2.0 / 3.0) * math.pi * r ** 3
            total = v1 - v2
            desc = (f"a rectangular prism {l1}×{w1}×{h1} with a hemispherical "
                    f"cavity of radius {r} carved out of the top")
            dim_lines = [f"box: {l1}×{w1}×{h1}", f"hemi cavity: r={r}"]
        elif comp == "cylinder_cone":
            r = rng.randint(2, 5)
            h1 = rng.randint(3, 7)
            h2 = rng.randint(2, 5)
            v1 = math.pi * r * r * h1
            v2 = (1.0 / 3.0) * math.pi * r * r * h2
            total = v1 + v2
            desc = (f"a cylinder of radius {r} and height {h1} with a cone of "
                    f"radius {r} and height {h2} mounted on top")
            dim_lines = [f"cyl: r={r}, h={h1}", f"cone: r={r}, h={h2}"]
        else:
            return None

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(desc=desc)
        answer = f"{round(total, 2)}"

        img = self._render(comp, desc, dim_lines, rng)
        return q, answer, img

    def _render(self, comp, desc, dim_lines, rng):
        """Simple schematic view (doesn't need to be perfect 3D)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        # draw a simple schematic of two shapes stacked
        if comp == "box_box":
            ax.add_patch(mpatches.Rectangle((2, 1), 6, 3, fc="#cccccc",
                                             ec="black", lw=2.0))
            ax.add_patch(mpatches.Rectangle((3.5, 4), 3, 2, fc="#aaaaaa",
                                             ec="black", lw=2.0))
        elif comp == "box_cylinder":
            ax.add_patch(mpatches.Rectangle((2, 1), 6, 3, fc="#cccccc",
                                             ec="black", lw=2.0))
            # cylinder schematic
            ax.add_patch(mpatches.Ellipse((5, 4.2), 2.5, 0.6, fc="#aaaaaa",
                                            ec="black", lw=2.0))
            ax.plot([3.75, 3.75], [4.2, 6.5], color="black", lw=2.0)
            ax.plot([6.25, 6.25], [4.2, 6.5], color="black", lw=2.0)
            ax.add_patch(mpatches.Ellipse((5, 6.5), 2.5, 0.6, fc="#aaaaaa",
                                            ec="black", lw=2.0))
        elif comp == "box_sub_hemisphere":
            ax.add_patch(mpatches.Rectangle((1.5, 1), 7, 4, fc="#cccccc",
                                             ec="black", lw=2.0))
            # hemisphere cavity on top
            ax.add_patch(mpatches.Wedge((5, 5), 1.5, 0, 180, fc="white",
                                         ec="black", lw=2.0))
        elif comp == "cylinder_cone":
            # cylinder body
            ax.plot([3, 3], [1, 5], color="black", lw=2.0)
            ax.plot([7, 7], [1, 5], color="black", lw=2.0)
            ax.add_patch(mpatches.Ellipse((5, 1), 4, 0.8, fc="#cccccc",
                                            ec="black", lw=2.0))
            ax.add_patch(mpatches.Ellipse((5, 5), 4, 0.8, fc="#cccccc",
                                            ec="black", lw=2.0))
            # cone on top
            ax.plot([3, 5], [5, 8], color="black", lw=2.0)
            ax.plot([7, 5], [5, 8], color="black", lw=2.0)
            ax.add_patch(mpatches.Ellipse((5, 5), 4, 0.8, fc="#aaaaaa",
                                            ec="black", lw=2.0))

        # Inline dimension labels on the figure (concise, visible)
        for i, line in enumerate(dim_lines):
            ax.text(0.3, 9.4 - i * 0.7, line, fontsize=11, ha="left",
                    fontweight="bold", color="#1a5cb0",
                    bbox=dict(facecolor="lightyellow", edgecolor="gray", pad=2))
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import re as _re
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        for sym in ["cm³", "cm^3", "cm3", "cubic cm", "cubic centimeters",
                    "cubic units", "units³", "units^3", "units"]:
            pred = pred.replace(sym, "").strip()
            gt = gt.replace(sym, "").strip()
        # Direct numeric compare
        try:
            p = float(pred)
            g = float(gt)
            return abs(p - g) / max(abs(g), 1e-9) < 0.015 or abs(p - g) < 0.5
        except ValueError:
            pass

        # Try symbolic compare with sympy: convert predicted to a numeric value
        # using pi, sqrt, etc. The grader treats expressions like "54 + 16π"
        # or "16*pi + 54" as equivalent to their numeric ground truth.
        def _normalize(s):
            s = s.replace("π", "pi").replace("\\pi", "pi")
            s = s.replace("^", "**")
            s = _re.sub(r"(\d)\s*\(", r"\1*(", s)
            s = _re.sub(r"(\d)pi", r"\1*pi", s)
            s = _re.sub(r"pi(\d)", r"pi*\1", s)
            s = s.replace("\\frac", "frac")
            s = s.replace("{", "(").replace("}", ")")
            s = s.replace("\\sqrt", "sqrt")
            s = s.replace("\\cdot", "*").replace("\\times", "*")
            s = _re.sub(r"\bcm\b|\bunits?\b|\bcubic\b", "", s)
            return s

        try:
            from sympy import sympify, pi, sqrt, N
            p_val = float(N(sympify(_normalize(pred), {"pi": pi, "sqrt": sqrt})))
            g_val = float(N(sympify(_normalize(gt), {"pi": pi, "sqrt": sqrt})))
            return abs(p_val - g_val) / max(abs(g_val), 1e-9) < 0.015 or abs(p_val - g_val) < 0.5
        except Exception:
            return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_c3v"
    os.makedirs(out_dir, exist_ok=True)
    env = Composite3DVolumeQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 181
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[c3v L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/c3v_s{s}_L{level}.png")
            print(f"[c3v L{level} s{s}] A={env._answer}")
