"""
Cube Decomposition Count QA.

Task: count the unit cubes in a stack rendered from an isometric viewpoint.
Two sub-panels are shown:
  - Left: isometric 3D view of the stack
  - Right: top-down view with the height of each visible column annotated
    as a small number (this removes any ambiguity about hidden cubes — the
    model just sums the labelled column heights)

Reward: exact integer match.

Level axes (10 levels, L0-L9):
  A) footprint size (n_x, n_y) — 2x2 at L0/L1, 2x3 at L2, 3x3 at L3..L5,
     3x4 at L6/L7, 4x4 at L8, 4x5 at L9
  B) max column height — 1 at L0, 2 at L1/L2, 3 at L3..L5, 4 at L6/L7,
     5 at L8/L9
  C) shape family — at low levels we restrict to staircases / single
     plateau layouts so that EVERY column is visible from the iso view; at
     higher levels we allow full random column heights (still correct,
     since the top-view panel shows the heights explicitly)
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PROMPT_VARIANTS = [
    "How many unit cubes are in the stack? The right panel shows a top-down view; the small number in each cell is the height (number of cubes) of that column. Sum the heights to count every unit cube. Put the integer in <answer>...</answer>.",
    "Count every unit cube in the stack. The 3D view is on the left; the top-down view on the right labels each column with its height. The total cube count equals the sum of the labelled heights. Integer in <answer>...</answer>.",
    "Two views of a unit-cube stack are shown — an isometric 3D view (left) and a labelled top-down view (right) where each cell shows that column's height. How many unit cubes total? Integer in <answer>...</answer>.",
    "Determine the total number of unit cubes that make up the stack. The top-down panel on the right gives the height of each column as a number; sum them. Put the integer in <answer>...</answer>.",
    "The figure shows a stack of unit cubes from two angles. The right panel labels each footprint cell with its column height (e.g. '2' means a stack of 2 cubes there). What is the total cube count? Integer in <answer>...</answer>.",
    "How many unit cubes does the structure contain? Use the labelled top-down view (right) to read off each column's height and sum. Put the integer in <answer>...</answer>.",
    "Count the unit cubes. Left: 3D isometric view. Right: top-down view with the height of each column written in its cell. Total cubes? Integer in <answer>...</answer>.",
    "Each cell of the right (top-down) panel is labelled with the number of cubes stacked in that column. Sum the labels. Total unit cubes in <answer>...</answer>.",
    "Compute the total number of unit cubes in the stack shown. The right-hand top-down view gives the column heights — add them up. Integer in <answer>...</answer>.",
    "The right panel labels each column of the stack with its height (1, 2, 3, ... or empty if no cube there). Sum the labels to get the total cube count. Put the integer in <answer>...</answer>.",
    "Total cubes? Heights are written on the top-down view (right). Add them. Put the integer in <answer>...</answer>.",
    "Two views of the same stack: 3D iso (left) and labelled top-down (right). Sum the labels on the top-down panel. Integer in <answer>...</answer>.",
]


def _draw_iso_cubes(ax, cubes, color):
    for (x, y, z) in cubes:
        for face in [
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
            [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
            [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        ]:
            poly = Poly3DCollection(
                [[(p[0] + x, p[1] + y, p[2] + z) for p in face]],
                facecolor=color, edgecolor="black", linewidths=1.0, alpha=1.0,
            )
            ax.add_collection3d(poly)


class CubeDecompositionCountQA(StandaloneVisualEnv):
    ENV_NAME = "cube_decomposition_count"

    # -------------------------------------------------------------- #
    # Level config
    # -------------------------------------------------------------- #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Footprint, max column height, shape family, top-view label fraction.
        # `label_frac` is what fraction of non-zero column heights are
        # written into the top-view panel; the rest are shown as "?". The
        # iso 3D view always shows the actual structure, so at low levels
        # the labels make the answer trivial (sum), and at high levels the
        # model has to look at the iso view and infer the missing heights.
        if level == 0:
            return {"nx": 2, "ny": 2, "max_h": 1, "family": "flat_subset", "label_frac": 1.0}
        if level == 1:
            return {"nx": 2, "ny": 2, "max_h": 2, "family": "stair", "label_frac": 1.0}
        if level == 2:
            return {"nx": 2, "ny": 3, "max_h": 2, "family": "stair", "label_frac": 1.0}
        if level == 3:
            return {"nx": 3, "ny": 3, "max_h": 2, "family": "stair", "label_frac": 0.8}
        if level == 4:
            return {"nx": 3, "ny": 3, "max_h": 3, "family": "stair", "label_frac": 0.7}
        if level == 5:
            return {"nx": 3, "ny": 3, "max_h": 3, "family": "random", "label_frac": 0.6}
        if level == 6:
            return {"nx": 3, "ny": 4, "max_h": 3, "family": "random", "label_frac": 0.5}
        if level == 7:
            return {"nx": 3, "ny": 4, "max_h": 4, "family": "random", "label_frac": 0.4}
        if level == 8:
            return {"nx": 4, "ny": 4, "max_h": 4, "family": "random", "label_frac": 0.3}
        return {"nx": 4, "ny": 5, "max_h": 5, "family": "random", "label_frac": 0.2}

    # -------------------------------------------------------------- #
    # Generation
    # -------------------------------------------------------------- #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 107)
        self._primary_complexity_feature = level

        nx, ny = cfg["nx"], cfg["ny"]
        max_h = cfg["max_h"]

        for _ in range(40):
            heights = self._sample_heights(rng, nx, ny, max_h, cfg["family"])
            if heights is None:
                continue
            cube_count = sum(sum(row) for row in heights)
            if cube_count >= 1:
                break
        else:
            heights = [[1 for _ in range(ny)] for _ in range(nx)]
            cube_count = nx * ny

        cubes = []
        for x in range(nx):
            for y in range(ny):
                for z in range(heights[x][y]):
                    cubes.append((x, y, z))

        # Decide which non-zero columns get a numeric label on the top view.
        positive = [(x, y) for x in range(nx) for y in range(ny)
                    if heights[x][y] > 0]
        rng.shuffle(positive)
        n_label = max(0, int(round(len(positive) * cfg["label_frac"])))
        labelled = set(positive[:n_label])
        # Always label all zero cells as "0" — that part is unambiguous and
        # tells the model what's *not* there.

        if cfg["label_frac"] >= 0.95:
            prompt = _PROMPT_VARIANTS[(self.seed or 0) % len(_PROMPT_VARIANTS)]
        else:
            prompt = (
                "Two views of a unit-cube stack are shown. Left: 3D iso view. "
                "Right: top-down view; some columns show their height as a "
                "number, others are marked '?' — for the '?' cells you must "
                "infer the height from the 3D view. Sum every column's height. "
                "Put the total number of unit cubes in <answer>...</answer>."
            )

        img = self._render(cubes, heights, nx, ny, max_h, rng, labelled)
        return prompt, str(cube_count), img

    def _sample_heights(self, rng, nx, ny, max_h, family):
        if family == "flat_subset":
            # L0: a connected polyomino of 1-cube towers (heights 0/1)
            cells = self._random_polyomino(rng, nx, ny,
                                           rng.randint(2, nx * ny))
            heights = [[1 if (x, y) in cells else 0
                        for y in range(ny)] for x in range(nx)]
            return heights
        if family == "stair":
            # Monotone staircase: heights non-increasing along x and y from
            # the back-left corner. Every column is visible from the iso view.
            base = rng.randint(1, max_h)
            heights = [[0] * ny for _ in range(nx)]
            heights[0][0] = base
            for x in range(nx):
                for y in range(ny):
                    if (x, y) == (0, 0):
                        continue
                    upper = base
                    if x > 0:
                        upper = min(upper, heights[x - 1][y])
                    if y > 0:
                        upper = min(upper, heights[x][y - 1])
                    heights[x][y] = rng.randint(0, upper)
            return heights
        # family == "random"
        heights = [[rng.randint(0, max_h) for _ in range(ny)]
                   for _ in range(nx)]
        return heights

    def _random_polyomino(self, rng, nx, ny, n_cells):
        cells = {(0, 0)}
        n_cells = min(n_cells, nx * ny)
        while len(cells) < n_cells:
            cx, cy = rng.choice(list(cells))
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            nxt = (cx + dx, cy + dy)
            if 0 <= nxt[0] < nx and 0 <= nxt[1] < ny:
                cells.add(nxt)
        return cells

    # -------------------------------------------------------------- #
    # Rendering
    # -------------------------------------------------------------- #
    def _render(self, cubes, heights, nx, ny, max_h, rng, labelled=None):
        labelled = labelled if labelled is not None else {
            (x, y) for x in range(nx) for y in range(ny) if heights[x][y] > 0
        }
        color = rng.choice(["#3498db", "#2ecc71", "#e74c3c",
                            "#f39c12", "#9b59b6", "#1abc9c"])

        fig = plt.figure(figsize=(11, 5.2))

        # Left: 3D isometric view
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        _draw_iso_cubes(ax3d, cubes, color=color)
        # Use the actual occupied bounding box so the camera is tight on the
        # structure regardless of whether columns are zero-height.
        if cubes:
            x_max = max(c[0] for c in cubes) + 1
            y_max = max(c[1] for c in cubes) + 1
            z_max = max(c[2] for c in cubes) + 1
        else:
            x_max, y_max, z_max = nx, ny, 1
        lim = max(x_max, y_max, z_max, 1)
        ax3d.set_xlim(0, lim)
        ax3d.set_ylim(0, lim)
        ax3d.set_zlim(0, lim)
        ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
        ax3d.view_init(elev=25, azim=35)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1, 1, 1, 0))
        try:
            ax3d.set_axis_off()
        except Exception:
            pass
        ax3d.set_title("3D View", fontsize=13, fontweight="bold")

        # Right: top-down view with each column's height labelled
        ax2d = fig.add_subplot(1, 2, 2)
        ax2d.set_xlim(-0.4, ny + 0.4)
        ax2d.set_ylim(-0.4, nx + 0.4)
        ax2d.set_aspect("equal")
        for x in range(nx):
            for y in range(ny):
                h = heights[x][y]
                fc = color if h > 0 else "#f4f4f4"
                rect = plt.Rectangle((y, nx - 1 - x), 1, 1,
                                     facecolor=fc, edgecolor="black", lw=1.4)
                ax2d.add_patch(rect)
                if h > 0:
                    if (x, y) in labelled:
                        ax2d.text(y + 0.5, nx - 1 - x + 0.5, str(h),
                                  ha="center", va="center",
                                  fontsize=18, fontweight="bold",
                                  color="black")
                    else:
                        ax2d.text(y + 0.5, nx - 1 - x + 0.5, "?",
                                  ha="center", va="center",
                                  fontsize=18, fontweight="bold",
                                  color="#c0392b")
                else:
                    ax2d.text(y + 0.5, nx - 1 - x + 0.5, "0",
                              ha="center", va="center",
                              fontsize=14, color="#777777")
        # axis labels
        for y in range(ny):
            ax2d.text(y + 0.5, -0.25, str(y + 1),
                      ha="center", fontsize=10, color="#444")
        for x in range(nx):
            ax2d.text(-0.25, nx - 0.5 - x, str(x + 1),
                      va="center", fontsize=10, color="#444")
        ax2d.axis("off")
        ax2d.set_title("Top View (column heights)",
                       fontsize=13, fontweight="bold")

        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig)
        plt.close(fig)
        return img


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_cdc"
    os.makedirs(out_dir, exist_ok=True)
    env = CubeDecompositionCountQA()
    for level in (0, 1, 3, 5, 7, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(f"{out_dir}/cdc_L{level}_s{seed}.png")
            print(f"L{level} s{seed}: gt={env._answer}")
