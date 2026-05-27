"""
Coordinate Geometry — Named Lattice Point QA.

A grid (Cartesian or rectangular) with several labelled lattice points (A, B,
C, D, ...). The model is asked for the (x, y) coordinates of one named point.

Difficulty axes:
  - grid size (5x5 -> 12x12)
  - number of labelled points (2 -> 8)
  - origin location: bottom-left corner at L0-L3, anywhere on grid at L4-L9
    (shifted-origin variant — model must read where (0,0) actually is)
  - axis ticks: shown at L<=5, hidden at L>=6 (must count grid lines)
  - reference-point shifted-frame variant at L>=7 (one named point's coords are
    given as a reference; model must compute another's coords relative to that)
"""
import math
import random
import re
import string
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows a coordinate grid with labelled points. What are the coordinates of point {label}? Give your answer as (x, y) inside <answer>...</answer>.",
    "Look at the labelled lattice points on the grid. What is the position of point {label}? Output as (x, y) in <answer>...</answer>.",
    "On the grid below, several points are marked with letters. Determine the (x, y) coordinates of point {label}. Place the answer in <answer>...</answer>.",
    "Read the diagram. What ordered pair (x, y) describes the position of point {label}? Answer in <answer>(x, y)</answer>.",
    "From the grid in the figure, identify the coordinates of point {label}. Output the ordered pair in <answer>(x, y)</answer>.",
    "The image displays a coordinate plane with named points. State the coordinates of point {label} in the form (x, y). Place the result in <answer>...</answer>.",
    "Find the (x, y) location of point {label} on the grid shown. Provide the ordered pair in <answer>...</answer>.",
    "Use the grid in the image to determine where point {label} sits. Output the coordinates as (x, y) inside <answer>...</answer>.",
    "Several lattice points are labelled in the figure. What are the coordinates of point {label}? Format: (x, y), in <answer>...</answer>.",
    "Examine the marked points on the coordinate grid. What ordered pair (x, y) gives point {label}? Answer in <answer>...</answer>.",
    "The grid shows labelled positions. Output the coordinates of point {label} as an ordered pair (x, y) in <answer>...</answer>.",
    "From the diagram, read off the position of point {label}. Express it as (x, y) in <answer>...</answer>.",
    "Where on the coordinate plane is point {label} located? Output the (x, y) pair inside <answer>...</answer>.",
    "Locate point {label} on the labelled grid in the image. Provide its (x, y) coordinates in <answer>...</answer>.",
    "What ordered pair represents point {label} in the figure? Place the (x, y) answer in <answer>...</answer>.",
    "Identify the (x, y) position of point {label} on the grid. Answer inside <answer>...</answer>.",
    "Inspect the labelled grid. Output the coordinates of point {label} as (x, y) in <answer>...</answer>.",
]

_TEMPLATES_REL = [
    "The image shows a grid with labelled points. Given that point {ref_label} is at coordinates {ref_xy}, what are the coordinates of point {label}? Output as (x, y) in <answer>...</answer>.",
    "Use point {ref_label} at coordinates {ref_xy} as a reference for the grid. What are the coordinates of point {label}? Place (x, y) in <answer>...</answer>.",
    "On the grid shown, point {ref_label} is located at {ref_xy}. Determine the (x, y) position of point {label}. Answer in <answer>...</answer>.",
    "Given that {ref_label} = {ref_xy} on the lattice grid in the image, what ordered pair represents {label}? Place the answer in <answer>...</answer>.",
    "If point {ref_label} on the grid corresponds to {ref_xy}, find the coordinates of point {label}. Output as (x, y) in <answer>...</answer>.",
    "The diagram labels lattice points. Take {ref_label} as {ref_xy} and read off the coordinates of {label}. Provide the ordered pair in <answer>...</answer>.",
    "Knowing {ref_label} sits at {ref_xy}, deduce the (x, y) position of {label} on the grid in the image. Place result in <answer>...</answer>.",
    "Given {ref_label} at {ref_xy} in the figure, what are the coordinates of {label}? Output (x, y) in <answer>...</answer>.",
]


class CoordinateGeometryNamedPointQA(StandaloneVisualEnv):
    ENV_NAME = "coordinate_geometry_named_point"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the ordered pair (x, y) directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"size": 5, "n_points": 2, "origin_shift": False,
                    "show_ticks": True, "use_reference": False}
        if level <= 2:
            return {"size": 6, "n_points": 3, "origin_shift": False,
                    "show_ticks": True, "use_reference": False}
        if level <= 4:
            return {"size": 7, "n_points": 4, "origin_shift": True,
                    "show_ticks": True, "use_reference": False}
        if level <= 5:
            return {"size": 8, "n_points": 5, "origin_shift": True,
                    "show_ticks": True, "use_reference": False}
        if level <= 6:
            return {"size": 9, "n_points": 5, "origin_shift": True,
                    "show_ticks": False, "use_reference": False}
        if level <= 7:
            return {"size": 9, "n_points": 6, "origin_shift": True,
                    "show_ticks": False, "use_reference": True}
        if level <= 8:
            return {"size": 10, "n_points": 7, "origin_shift": True,
                    "show_ticks": False, "use_reference": True}
        return {"size": 12, "n_points": 8, "origin_shift": True,
                "show_ticks": False, "use_reference": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1409 + level * 19 + 7)

        size = cfg["size"]
        n_pts = cfg["n_points"]

        # Origin shift: where is (0,0) drawn on the grid? For non-shifted,
        # the grid is [0..size] x [0..size]. For shifted, the origin is at
        # (ox, oy) and the grid spans [-ox..size-ox] x [-oy..size-oy].
        if cfg["origin_shift"]:
            ox = rng.randint(1, size - 2)
            oy = rng.randint(1, size - 2)
        else:
            ox = 0
            oy = 0

        # Sample n_pts unique cells (in display coords 0..size)
        cells = [(r, c) for r in range(size + 1) for c in range(size + 1)]
        rng.shuffle(cells)
        chosen = cells[:n_pts]
        labels = list(string.ascii_uppercase[:n_pts])
        rng.shuffle(labels)

        # Convert chosen display coords (col, row) into world coords
        # world x = col - ox, world y = row - oy
        point_world = {}
        for (col, row), lab in zip(chosen, labels):
            point_world[lab] = (col - ox, row - oy)

        # Pick the target label (the one we ask about)
        target_lab = rng.choice(labels)
        tx, ty = point_world[target_lab]

        # Optional reference label (for L>=7): pick a different point and
        # state its coords; model uses that to deduce target
        ref_lab = None
        ref_xy = None
        if cfg["use_reference"]:
            others = [l for l in labels if l != target_lab]
            if others:
                ref_lab = rng.choice(others)
                ref_xy = point_world[ref_lab]

        # Question text
        if ref_lab is not None:
            template = rng.choice(_TEMPLATES_REL)
            question = template.format(label=target_lab, ref_label=ref_lab,
                                        ref_xy=f"({ref_xy[0]}, {ref_xy[1]})")
        else:
            template = rng.choice(_TEMPLATES)
            question = template.format(label=target_lab)

        answer = f"({tx}, {ty})"
        # Stash for verify
        self._target_xy = (tx, ty)

        img = self._render(size, ox, oy, point_world, target_lab, cfg, rng,
                            ref_lab=ref_lab, hide_origin_label=cfg["use_reference"])
        return question, answer, img

    def _render(self, size, ox, oy, points, target_lab, cfg, rng,
                 ref_lab=None, hide_origin_label=False) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0", "#f5f9ff"])
        fig, ax = plt.subplots(figsize=(0.6 * (size + 2), 0.6 * (size + 2)),
                                dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Draw grid lines [0..size]
        for k in range(size + 1):
            ax.plot([0, size], [k, k], color="#cfd8dc", linewidth=0.7,
                     zorder=1)
            ax.plot([k, k], [0, size], color="#cfd8dc", linewidth=0.7,
                     zorder=1)
        # Axes through origin
        axis_color = rng.choice(["#37474f", "#1d3557", "#2c3e50"])
        ax.plot([0, size], [oy, oy], color=axis_color, linewidth=1.5,
                 zorder=2)
        ax.plot([ox, ox], [0, size], color=axis_color, linewidth=1.5,
                 zorder=2)
        # Arrowheads (drawn as text triangles)
        ax.annotate("", xy=(size + 0.1, oy), xytext=(size - 0.05, oy),
                     arrowprops=dict(arrowstyle="->", color=axis_color,
                                       lw=1.5))
        ax.annotate("", xy=(ox, size + 0.1), xytext=(ox, size - 0.05),
                     arrowprops=dict(arrowstyle="->", color=axis_color,
                                       lw=1.5))
        # Axis labels (x/y)
        ax.text(size + 0.25, oy - 0.15, "x", fontsize=11,
                 color=axis_color, ha="left", va="top")
        ax.text(ox - 0.15, size + 0.25, "y", fontsize=11,
                 color=axis_color, ha="right", va="bottom")

        # Tick labels along x axis (only when show_ticks)
        if cfg["show_ticks"] and not hide_origin_label:
            for col in range(0, size + 1):
                xv = col - ox
                if xv == 0 and not hide_origin_label:
                    ax.text(col, oy - 0.35, "0", ha="center", va="top",
                             fontsize=8, color="#555")
                elif col % max(1, size // 6) == 0 and xv != 0:
                    ax.text(col, oy - 0.35, str(xv), ha="center", va="top",
                             fontsize=8, color="#555")
            for row in range(0, size + 1):
                yv = row - oy
                if yv == 0:
                    pass  # already labelled by x-axis 0
                elif row % max(1, size // 6) == 0 and yv != 0:
                    ax.text(ox - 0.25, row, str(yv), ha="right", va="center",
                             fontsize=8, color="#555")
        elif not hide_origin_label:
            # Show only the origin tick "O"
            ax.text(ox - 0.18, oy - 0.18, "O", ha="right", va="top",
                     fontsize=10, color=axis_color)

        # Draw points
        for lab, (wx, wy) in points.items():
            display_x = wx + ox
            display_y = wy + oy
            color = "#e74c3c" if lab == target_lab else \
                    ("#27ae60" if lab == ref_lab else "#1565c0")
            ax.scatter([display_x], [display_y], s=70, color=color, zorder=4,
                        edgecolor="black", linewidth=0.8)
            ax.text(display_x + 0.18, display_y + 0.18, lab,
                     fontsize=13, fontweight="bold", color=color, zorder=5,
                     ha="left", va="bottom")

        ax.set_xlim(-0.5, size + 0.7)
        ax.set_ylim(-0.5, size + 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                     facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ----------------------------------------------------------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Accept "(x, y)", "x, y", "[x, y]", "x,y", with optional whitespace
        # and optional "+" / "-" signs. Compare ints (we use integer lattice
        # points).
        gt_x, gt_y = self._target_xy
        # Pull all signed numbers in order
        nums = re.findall(r"-?\d+", predicted)
        if len(nums) < 2:
            return False
        try:
            px = int(nums[0])
            py = int(nums[1])
        except ValueError:
            return False
        return px == gt_x and py == gt_y


if __name__ == "__main__":
    env = CoordinateGeometryNamedPointQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer}")
