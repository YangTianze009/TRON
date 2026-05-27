"""
Open-Top Container Surface Area QA.

A rectangular cuboid container with NO top face (open top): 5 faces total.
Given length l, width w, and height h, the model must compute the surface
area of the 5 faces:
    SA = l*w (bottom) + 2*l*h (front + back) + 2*w*h (left + right)
       = l*w + 2*l*h + 2*w*h

The figure shows the cuboid in 3D with the top face removed (visible interior),
and the dimensions l, w, h labeled.

Difficulty axes:
  - L0..L2: small integer dims (2..5), all integer answer.
  - L3..L5: integer dims up to 10, integer answer.
  - L6..L9: mixed integer/decimal dims; one-decimal answer.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The figure shows an open-top rectangular container (no top face). The length, width, and height are labeled. Compute the total surface area of the 5 faces (bottom + 4 sides). Place the numeric answer in <answer>...</answer>.",
    "An open-top cuboid container is shown. Find the surface area of all the visible faces (the bottom and the 4 lateral sides). Place the numeric value in <answer>...</answer>.",
    "Shown is a rectangular box without a top. Using the labeled length l, width w, and height h, compute the total surface area (5 faces). Place the answer in <answer>...</answer>.",
    "The container in the figure has no top face. Determine the surface area of the remaining 5 faces. Place the numeric answer in <answer>...</answer>.",
    "Compute the surface area of the open-top rectangular container shown. Use the labeled l, w, h. Place the value in <answer>...</answer>.",
    "An open box (no top) has the labeled dimensions. Find the sum of the bottom plus the 4 side rectangle areas. Place the answer in <answer>...</answer>.",
    "From the figure, identify l, w, h of the open-top container, then compute SA = l*w + 2*l*h + 2*w*h. Place the numeric answer in <answer>...</answer>.",
    "The cuboid container is missing its top. Compute its 5-face surface area using the labeled dimensions. Place the value in <answer>...</answer>.",
    "Determine the total area of the 5 faces of the open-top container shown. Place the numeric answer in <answer>...</answer>.",
    "Read l, w, h from the figure. The container has no top — compute the area of the bottom plus the 4 side faces. Place the answer in <answer>...</answer>.",
    "The open-top cuboid container has labeled l, w, h. Find its surface area (bottom + 4 sides). Place the numeric value in <answer>...</answer>.",
    "Compute the surface area of the open-top rectangular box. Place the numeric answer in <answer>...</answer>.",
    "The figure shows a box without a top face. From the labeled dimensions, calculate the area of the 5 remaining faces. Place the value in <answer>...</answer>.",
    "An open container (no top) has length l, width w, and height h. Compute its surface area (the 5 faces shown). Place the answer in <answer>...</answer>.",
    "Shown: an open-top cuboid. Using l, w, h, compute SA = lw + 2lh + 2wh. Place the numeric answer in <answer>...</answer>.",
    "The cuboid container in the figure is open at the top. Find the total surface area of the 5 faces. Place the answer in <answer>...</answer>.",
    "Compute the total surface area of the open-top rectangular container, where the top face is missing. Place the numeric value in <answer>...</answer>.",
]


class OpenTopContainerSaQA(StandaloneVisualEnv):
    ENV_NAME = "open_top_container_sa"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "dim_pool": [2, 3, 4, 5],
                "allow_float": False,
            }
        if level <= 5:
            return {
                "dim_pool": [3, 4, 5, 6, 7, 8, 10],
                "allow_float": False,
            }
        if level <= 7:
            return {
                "dim_pool": [4, 5, 6, 8, 10, 12, 15],
                "allow_float": False,
            }
        return {
            "dim_pool": [2.5, 3, 3.5, 4, 5, 5.5, 6, 7.5, 8, 10],
            "allow_float": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1187 + level * 71 + 13)

        for _ in range(40):
            l = rng.choice(cfg["dim_pool"])
            w = rng.choice(cfg["dim_pool"])
            h = rng.choice(cfg["dim_pool"])
            sa = l * w + 2 * l * h + 2 * w * h
            if sa <= 0:
                continue
            # Ensure clean answer: integer or simple decimal.
            if cfg["allow_float"]:
                sa_round = round(sa, 1)
                if abs(sa - sa_round) > 1e-6:
                    continue
            else:
                if abs(sa - round(sa)) > 1e-6:
                    continue
                sa_round = int(round(sa))
            break
        else:
            return None

        if isinstance(sa_round, int):
            answer = str(sa_round)
        elif abs(sa_round - round(sa_round)) < 1e-9:
            answer = str(int(round(sa_round)))
        else:
            answer = f"{sa_round:g}"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(l, w, h)
        return question, answer, img

    def _render(self, l, w, h) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Project a 3D cuboid using simple axonometric (x: right, y: depth-back-up-and-right, z: up).
        # Choose visual scale so dims map to plot units.
        scale = 1.0
        # Front-bottom-left corner at origin in screen coordinates.
        # Axes:
        #   length (l) -> screen +x
        #   width (w)  -> screen (+0.55x, +0.55y)  (depth, going back-and-up)
        #   height (h) -> screen +y
        dx = 0.55  # depth x-component
        dy = 0.55  # depth y-component
        L = l * scale
        W = w * scale
        H = h * scale

        # Bottom face corners (front-bottom-left, front-bottom-right, back-bottom-right, back-bottom-left)
        FBL = (0.0, 0.0)
        FBR = (L, 0.0)
        BBR = (L + W * dx, W * dy)
        BBL = (W * dx, W * dy)
        # Top face corners (where the open top would be)
        FTL = (0.0, H)
        FTR = (L, H)
        BTR = (L + W * dx, H + W * dy)
        BTL = (W * dx, H + W * dy)

        edge_color = "#2c3e50"
        face_fill = "#e7eef9"
        side_fill = "#dbe5f5"
        bottom_fill = "#cdd9eb"
        interior_fill = "#f4f6fa"

        # Visible interior (bottom face seen through the open top): draw the bottom face fill
        # with a slightly darker shade so it's visible through the missing top.
        ax.add_patch(patches.Polygon(
            [FBL, FBR, BBR, BBL], closed=True,
            facecolor=bottom_fill, edgecolor=edge_color, linewidth=1.6))

        # Back-left side (visible interior face): the inside of the back wall +
        # the inside of the left wall. To suggest open-top, draw the inside
        # surfaces as light fills.
        # Inside of back wall (rectangle BBL-BBR-BTR-BTL)
        ax.add_patch(patches.Polygon(
            [BBL, BBR, BTR, BTL], closed=True,
            facecolor=interior_fill, edgecolor=edge_color, linewidth=1.4,
            alpha=0.85, linestyle="--"))
        # Inside of left wall (rectangle FBL-BBL-BTL-FTL)
        ax.add_patch(patches.Polygon(
            [FBL, BBL, BTL, FTL], closed=True,
            facecolor=interior_fill, edgecolor=edge_color, linewidth=1.4,
            alpha=0.85, linestyle="--"))

        # Front face (outside)
        ax.add_patch(patches.Polygon(
            [FBL, FBR, FTR, FTL], closed=True,
            facecolor=face_fill, edgecolor=edge_color, linewidth=2.0))
        # Right face (outside)
        ax.add_patch(patches.Polygon(
            [FBR, BBR, BTR, FTR], closed=True,
            facecolor=side_fill, edgecolor=edge_color, linewidth=2.0))

        # Top opening: draw only the rim (4 edges) — no top fill.
        ax.plot([FTL[0], FTR[0]], [FTL[1], FTR[1]],
                color=edge_color, lw=2.0)
        ax.plot([FTR[0], BTR[0]], [FTR[1], BTR[1]],
                color=edge_color, lw=2.0)
        ax.plot([BTR[0], BTL[0]], [BTR[1], BTL[1]],
                color=edge_color, lw=2.0)
        ax.plot([BTL[0], FTL[0]], [BTL[1], FTL[1]],
                color=edge_color, lw=2.0)

        # Length label (along front bottom edge)
        ax.annotate("", xy=(L, -0.4), xytext=(0, -0.4),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(L / 2, -0.65, f"l = {self._fmt(l)}",
                ha="center", va="top", fontsize=12, color="#c0392b",
                fontweight="bold")
        # Width label (along bottom-right depth)
        wmid = ((L + (L + W * dx)) / 2 + 0.4, (0 + W * dy) / 2 - 0.2)
        ax.annotate("", xy=BBR, xytext=FBR,
                    arrowprops=dict(arrowstyle="<->", color="#1e8449", lw=1.4))
        ax.text(L + W * dx / 2 + 0.5, W * dy / 2 - 0.1,
                f"w = {self._fmt(w)}",
                ha="left", va="center", fontsize=12, color="#1e8449",
                fontweight="bold")
        # Height label (along front-left vertical edge — keeps it away from the
        # depth/width arrow on the right side).
        ax.annotate("", xy=(-0.4, H), xytext=(-0.4, 0),
                    arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
        ax.text(-0.55, H / 2, f"h = {self._fmt(h)}",
                ha="right", va="center", fontsize=12, color="#2c3e50",
                fontweight="bold")

        # Legend: label "open top"
        ax.text((L + W * dx) / 2, H + W * dy + 0.4,
                "(open top: no top face)",
                ha="center", va="bottom", fontsize=10, color="#7b241c",
                fontstyle="italic")

        ax.set_xlim(-1.7, L + W * dx + 1.5)
        ax.set_ylim(-1.4, H + W * dy + 1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    @staticmethod
    def _fmt(v):
        if isinstance(v, int):
            return str(v)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"
