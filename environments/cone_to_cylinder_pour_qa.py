"""
Cone-To-Cylinder Pour QA.

A cone and a cylinder are drawn side by side. They share the same base
radius and the same height. The student is asked: how many full cone-fulls
of water are needed to fill the cylinder?

Mathematics:
  V_cone     = (1/3) * pi * r^2 * h
  V_cylinder = pi * r^2 * h
  ratio      = V_cylinder / V_cone = 3

So the answer is always 3 when r and h match — the question tests whether
the student knows the volume formulas, not whether they can divide.

At higher levels we extend the question with explicit-but-mismatched
dimensions (cone radius r1 with cylinder radius r2, and so on) to force
real computation. The answer is then the integer (or one-decimal) ratio.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_MATCHING = [
    "The figure shows a cone and a cylinder. They have the SAME base radius and the SAME height (both labeled). How many cone-fulls of water are needed to completely fill the cylinder? Place the integer answer in <answer>...</answer>.",
    "A cone and a cylinder share the same base radius r and the same height h. How many full cone-fulls fill up the cylinder? Place the answer in <answer>...</answer>.",
    "Given the cone and cylinder shown (same r and h), determine the number of cone-fulls needed to fill the cylinder. Place the integer in <answer>...</answer>.",
    "The cone and cylinder in the figure have identical radius r and height h. How many cone-fulls of water fill the cylinder exactly? Place the answer in <answer>...</answer>.",
    "Both the cone and the cylinder shown have the same r and same h. Compute how many full cone-fulls are needed to fill the cylinder. Place the integer in <answer>...</answer>.",
    "Cone and cylinder have matching dimensions. How many cone-fulls of water make one cylinder-full? Place the integer in <answer>...</answer>.",
    "Same r and same h for both the cone and the cylinder. How many cone-fulls equal one cylinder-full? Integer in <answer>...</answer>.",
    "How many cone-shaped containers of water fit into the cylinder if both share the same base radius and the same height? Place the integer in <answer>...</answer>.",
]

_TEMPLATES_GENERAL = [
    "The figure shows a cone and a cylinder with their dimensions labeled. How many full cone-fulls of water are needed to fill the cylinder? (Use V_cone = (1/3) pi r^2 h and V_cyl = pi r^2 h.) Place the numeric answer in <answer>...</answer>.",
    "From the labeled measurements of the cone and the cylinder, compute the number of cone-fulls needed to fill the cylinder (round to 1 decimal). Place the answer in <answer>...</answer>.",
    "Given the dimensions in the figure, determine the ratio V_cylinder / V_cone (i.e. how many cone-fulls fill the cylinder). Place the numeric value in <answer>...</answer>.",
    "How many cone-fulls of water are needed to fill the cylinder? Use the labeled dimensions. Place the numeric answer in <answer>...</answer>.",
    "Compute V_cylinder / V_cone using the labeled radii and heights. Place the numeric answer in <answer>...</answer>.",
    "Find the number of cone-fulls of water that make one cylinder-full, using the dimensions shown. Round to 1 decimal place. Place answer in <answer>...</answer>.",
    "Using the labeled dimensions of the cone and cylinder, compute how many cone-fulls are needed to fill the cylinder. Place the numeric value in <answer>...</answer>.",
    "Read r and h for both shapes from the figure. Compute V_cylinder / V_cone. Place the numeric ratio in <answer>...</answer>.",
    "From the dimensions in the figure, determine how many full cone-fulls fill the cylinder. Round to 1 decimal place. Place in <answer>...</answer>.",
    "Compute the ratio of the cylinder's volume to the cone's volume from the labeled measurements. Place the numeric value in <answer>...</answer>.",
]


class ConeToCylinderPourQA(StandaloneVisualEnv):
    ENV_NAME = "cone_to_cylinder_pour"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "matching": True,
                "r_pool": [1, 2, 3, 4, 5],
                "h_pool": [2, 3, 4, 5, 6, 8, 10],
            }
        if level <= 5:
            # Mismatched but ratio still integer (radius ratio 1:1, height
            # mismatch by integer factor; or radius mismatch 1:k).
            return {
                "matching": False,
                "r_pool": [2, 3, 4, 5],
                "h_pool": [3, 4, 6, 8, 10, 12],
                "ratio_class": "integer",
            }
        # 2026-05-04: bumped L9 difficulty — wider r/h pools widen the
        # arithmetic and the candidate ratio space (still one_decimal so
        # tolerance behavior is unchanged).
        # 2026-05-04 R3: benchmark-sample-driven harden — at L9 require
        # 2-decimal ratio (e.g. 5.33 instead of 5.3) so off-by-one in the
        # final round step gives wrong answer. WeMath samples like Q5
        # ("113.04 cm³") use 2-decimal precision routinely.
        if level == 9:
            return {
                "matching": False,
                "r_pool": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                "h_pool": [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20],
                "ratio_class": "two_decimal",
            }
        return {
            "matching": False,
            "r_pool": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "h_pool": [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20],
            "ratio_class": "one_decimal",
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1103 + level * 113 + 31)

        if cfg["matching"]:
            r_cone = r_cyl = rng.choice(cfg["r_pool"])
            h_cone = h_cyl = rng.choice(cfg["h_pool"])
            templates = _TEMPLATES_MATCHING
            answer_val = 3
            answer = "3"
        else:
            for _ in range(40):
                r_cone = rng.choice(cfg["r_pool"])
                r_cyl = rng.choice(cfg["r_pool"])
                h_cone = rng.choice(cfg["h_pool"])
                h_cyl = rng.choice(cfg["h_pool"])
                # Skip the trivially matching case so problems are non-trivial.
                if (r_cone, h_cone) == (r_cyl, h_cyl):
                    continue
                # ratio = 3 * (r_cyl^2 * h_cyl) / (r_cone^2 * h_cone)
                num = 3 * (r_cyl ** 2) * h_cyl
                den = (r_cone ** 2) * h_cone
                if den == 0:
                    continue
                ratio = num / den
                if cfg["ratio_class"] == "integer":
                    if num % den == 0 and 1 < (num // den) <= 30:
                        answer_val = num // den
                        answer = str(int(answer_val))
                        break
                elif cfg["ratio_class"] == "two_decimal":
                    # 2026-05-04 R3: benchmark-sample-driven harden — accept
                    # 2-decimal ratios (e.g. 5.33). Force ratio NOT to be
                    # 1-decimal-clean so it actually requires 2-dp.
                    val_round = round(ratio, 2)
                    val_round_1dp = round(ratio, 1)
                    if (0.5 <= val_round <= 30
                            and abs(ratio - val_round) < 1e-6
                            and abs(val_round - val_round_1dp) > 1e-6):
                        answer_val = val_round
                        if abs(answer_val - round(answer_val)) < 1e-9:
                            answer = str(int(round(answer_val)))
                        else:
                            answer = f"{answer_val:.2f}"
                        break
                else:
                    # one-decimal ratio in a sensible range
                    val_round = round(ratio, 1)
                    if 0.5 <= val_round <= 30 and abs(ratio - val_round) < 1e-6:
                        answer_val = val_round
                        if abs(answer_val - round(answer_val)) < 1e-9:
                            answer = str(int(round(answer_val)))
                        else:
                            answer = f"{answer_val:g}"
                        break
            else:
                return None
            templates = _TEMPLATES_GENERAL

        sidx = (self.seed or 0) % len(templates)
        question = templates[sidx]
        img = self._render(r_cone, h_cone, r_cyl, h_cyl, cfg["matching"])
        return question, answer, img

    # ---------------- render ---------------- #
    def _render(self, r_cone, h_cone, r_cyl, h_cyl, matching) -> Image.Image:
        fig, axes = plt.subplots(1, 2, figsize=(8, 6.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")

        self._draw_cone(axes[0], r_cone, h_cone)
        self._draw_cylinder(axes[1], r_cyl, h_cyl)

        if matching:
            fig.suptitle(f"Cone and Cylinder (same r = {r_cone}, "
                          f"same h = {h_cone})",
                         fontsize=12, fontweight="bold")
        else:
            fig.suptitle("Cone and Cylinder (dimensions labeled)",
                         fontsize=12, fontweight="bold")

        try:
            fig.tight_layout()
        except Exception:
            pass

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _draw_cone(self, ax, r, h):
        ax.set_facecolor("#ffffff")
        # Approximate 2D projection of a cone: triangle silhouette + an
        # ellipse base.
        # Triangle apex at (0, h); base from (-r, 0) to (r, 0).
        # Base ellipse to give 3D illusion.
        from matplotlib.patches import Polygon, Ellipse
        # Lateral fill (triangle)
        tri = Polygon([(-r, 0), (r, 0), (0, h)], closed=True,
                      facecolor="#dbe5f5", edgecolor="#2c3e50",
                      linewidth=1.5)
        ax.add_patch(tri)
        # Front ellipse arc (base)
        ell_front = Ellipse((0, 0), 2 * r, r * 0.5,
                            facecolor="#bcd2ec", edgecolor="#2c3e50",
                            linewidth=1.5)
        ax.add_patch(ell_front)
        # Apex dot
        ax.plot([0], [h], "o", color="#2c3e50", markersize=4)

        # r label (horizontal arrow on base)
        ax.annotate("", xy=(r, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(r / 2, 0.45, f"r = {r}",
                ha="center", va="bottom", fontsize=12, color="#c0392b",
                fontweight="bold")

        # h label (vertical arrow on right)
        x_h = r + 1.0
        ax.annotate("", xy=(x_h, 0), xytext=(x_h, h),
                    arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
        ax.text(x_h + 0.25, h / 2, f"h = {h}",
                ha="left", va="center", fontsize=12, color="#2c3e50",
                fontweight="bold")

        ax.set_xlim(-r - 1, r + 3.5)
        ax.set_ylim(-r * 0.6, h + 1.5)
        ax.set_aspect("equal")
        ax.set_title("Cone", fontsize=12, fontweight="bold")
        ax.axis("off")

    def _draw_cylinder(self, ax, r, h):
        ax.set_facecolor("#ffffff")
        from matplotlib.patches import Ellipse
        # Side rectangle
        ax.add_patch(plt.Rectangle((-r, 0), 2 * r, h,
                                    facecolor="#e7eef9",
                                    edgecolor="none", zorder=2))
        # Bottom ellipse
        e_bot = Ellipse((0, 0), 2 * r, r * 0.5,
                        facecolor="#bcd2ec", edgecolor="#2c3e50",
                        linewidth=1.5, zorder=3)
        ax.add_patch(e_bot)
        # Top ellipse
        e_top = Ellipse((0, h), 2 * r, r * 0.5,
                        facecolor="#dbe5f5", edgecolor="#2c3e50",
                        linewidth=1.5, zorder=3)
        ax.add_patch(e_top)
        # Side edges
        ax.plot([-r, -r], [0, h], color="#2c3e50", linewidth=1.5, zorder=4)
        ax.plot([r, r], [0, h], color="#2c3e50", linewidth=1.5, zorder=4)

        # r label (top ellipse arrow)
        ax.annotate("", xy=(r, h), xytext=(0, h),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(r / 2, h + 0.4, f"r = {r}",
                ha="center", va="bottom", fontsize=12, color="#c0392b",
                fontweight="bold")

        # h label
        x_h = r + 1.0
        ax.annotate("", xy=(x_h, 0), xytext=(x_h, h),
                    arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
        ax.text(x_h + 0.25, h / 2, f"h = {h}",
                ha="left", va="center", fontsize=12, color="#2c3e50",
                fontweight="bold")

        ax.set_xlim(-r - 1, r + 3.5)
        ax.set_ylim(-r * 0.6, h + 1.5)
        ax.set_aspect("equal")
        ax.set_title("Cylinder", fontsize=12, fontweight="bold")
        ax.axis("off")
