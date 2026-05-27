"""
Multi-View Cube Count QA.

Task progression:
  L0-L2  — single front view only. The image shows a stacked-bar "front view"
           of a 1D cube stack (column = number). Model just sums the column
           heights. This is the easiest training signal for cube counting.
  L3-L5  — front view + top view, but the top view colour-codes how tall each
           column is (the height is also written as a number). Model still
           just sums.
  L6-L9  — front view + top view, with only the TOP-VIEW shape (filled cells)
           given. Model must compute the MINIMUM cube count consistent with
           both views: for each top-view column c, contribute
             front_height[c] + (filled_in_col_c - 1)
           cubes. This is the original "two-orthographic-views minimum cubes"
           task from spatial reasoning benches.

Reward: exact integer.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PROMPT_FRONT_ONLY = [
    "The image shows the FRONT VIEW of a cube stack — column 1, column 2, ... drawn as stacked unit squares from left to right. How many unit cubes does the stack contain in total? Sum the column heights. Put the integer in <answer>...</answer>.",
    "Front view of a 1D stack: column 1, column 2, ... each made of unit cubes. Total unit cubes? Sum the heights. Integer in <answer>...</answer>.",
    "Count the unit cubes in the stack. Each column's height is shown in the front view. Sum heights → total cubes. Integer in <answer>...</answer>.",
    "How many unit cubes? The image shows the front view of a single-row stack — count by summing the column heights. Integer in <answer>...</answer>.",
    "Sum the column heights in the front view to find the total cube count. Put the integer in <answer>...</answer>.",
]

_PROMPT_TWO_VIEWS_LABELLED = [
    "Two views of a cube stack: front view (left) and top view (right). The top-view cells are labelled with each column's height. Sum the labels to get the total number of unit cubes. Integer in <answer>...</answer>.",
    "Front + labelled top view of a stack of unit cubes. Each top-view cell shows the height of that column. Sum them. Integer in <answer>...</answer>.",
    "The right (top-view) panel labels every cell with the column's height. Add up the labels to count the cubes. Integer in <answer>...</answer>.",
    "Count the unit cubes by summing the column heights labelled in the top-view panel. Integer in <answer>...</answer>.",
]

_PROMPT_MIN_CUBES = [
    "Two orthographic views: front (left) and top (right). The front view gives the maximum height in each column; the top view gives the footprint (which cells are filled, with a '?' since heights aren't shown directly). What is the MINIMUM number of unit cubes consistent with both views? Hint: for each column c, contribute front_height[c] + (filled_cells_in_col_c − 1). Integer in <answer>...</answer>.",
    "Front + top views of a unit-cube stack. The top view shows the footprint only. Compute the minimum cube count consistent with both views (each '?' cell has at least 1 cube; one cell per column reaches the front-view height). Integer in <answer>...</answer>.",
    "Two views (front, top). The top view shows the footprint; heights aren't given there. Find the smallest number of unit cubes consistent with both views. For each column: max-height + (filled-cells-in-that-column − 1). Integer in <answer>...</answer>.",
    "Given the front (column heights) and top (footprint) views, what is the minimum total cube count? Use: for each column, front-view-height + (number of filled cells in that column − 1). Integer in <answer>...</answer>.",
]


class MultiViewCubeCountQA(StandaloneVisualEnv):
    ENV_NAME = "multi_view_cube_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # variant: which of the three task forms to use
        # n_cols / n_rows: front view length / top view depth
        # max_h: max column height
        if level == 0:
            return {"variant": "front_only", "n_cols": 3, "max_h": 3}
        if level == 1:
            return {"variant": "front_only", "n_cols": 4, "max_h": 4}
        if level == 2:
            return {"variant": "front_only", "n_cols": 5, "max_h": 5}
        if level == 3:
            return {"variant": "labelled", "n_cols": 2, "n_rows": 2, "max_h": 2}
        if level == 4:
            return {"variant": "labelled", "n_cols": 3, "n_rows": 2, "max_h": 3}
        if level == 5:
            return {"variant": "labelled", "n_cols": 3, "n_rows": 3, "max_h": 3}
        if level == 6:
            return {"variant": "min_cubes", "n_cols": 2, "n_rows": 2, "max_h": 2}
        if level == 7:
            return {"variant": "min_cubes", "n_cols": 3, "n_rows": 2, "max_h": 3}
        if level == 8:
            return {"variant": "min_cubes", "n_cols": 3, "n_rows": 3, "max_h": 3}
        return {"variant": "min_cubes", "n_cols": 4, "n_rows": 3, "max_h": 4}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 531)
        self._primary_complexity_feature = level

        if cfg["variant"] == "front_only":
            return self._front_only_problem(cfg, rng)
        if cfg["variant"] == "labelled":
            return self._labelled_problem(cfg, rng)
        return self._min_cubes_problem(cfg, rng)

    # -------------------------------------------------------------- #
    def _front_only_problem(self, cfg, rng):
        for _ in range(20):
            heights = [rng.randint(1, cfg["max_h"]) for _ in range(cfg["n_cols"])]
            if sum(heights) >= 2:
                break
        ans = sum(heights)
        prompt = _PROMPT_FRONT_ONLY[(self.seed or 0) % len(_PROMPT_FRONT_ONLY)]
        img = self._render_front(heights)
        return prompt, str(ans), img

    def _labelled_problem(self, cfg, rng):
        nx, ny = cfg["n_rows"], cfg["n_cols"]
        for _ in range(20):
            heights = [[rng.randint(0, cfg["max_h"]) for _ in range(ny)]
                       for _ in range(nx)]
            if sum(sum(r) for r in heights) >= 2:
                break
        ans = sum(sum(r) for r in heights)
        front_view = [max(heights[r][c] for r in range(nx)) for c in range(ny)]
        prompt = _PROMPT_TWO_VIEWS_LABELLED[(self.seed or 0) % len(_PROMPT_TWO_VIEWS_LABELLED)]
        img = self._render_two_views(front_view, heights, label_heights=True)
        return prompt, str(ans), img

    def _min_cubes_problem(self, cfg, rng):
        nx, ny = cfg["n_rows"], cfg["n_cols"]
        for _ in range(40):
            heights = [[rng.randint(0, cfg["max_h"]) for _ in range(ny)]
                       for _ in range(nx)]
            top_view = [[1 if heights[r][c] >= 1 else 0
                         for c in range(ny)] for r in range(nx)]
            front_view = [max(heights[r][c] for r in range(nx))
                          for c in range(ny)]
            # require non-trivial
            if sum(front_view) >= 2 and sum(sum(r) for r in top_view) >= 2:
                break

        # Compute MCQ ground truth = minimum cubes
        # For each column c: top_cells_in_col + max(0, front_view[c] - 1)
        min_cubes = 0
        for c in range(ny):
            top_cells_in_col = sum(top_view[r][c] for r in range(nx))
            if top_cells_in_col == 0:
                continue
            min_cubes += top_cells_in_col
            if front_view[c] > 1:
                min_cubes += front_view[c] - 1

        prompt = _PROMPT_MIN_CUBES[(self.seed or 0) % len(_PROMPT_MIN_CUBES)]
        img = self._render_two_views(front_view, top_view, label_heights=False)
        return prompt, str(min_cubes), img

    # -------------------------------------------------------------- #
    def _render_front(self, heights: List[int]):
        n = len(heights)
        max_h = max(heights)
        fig, ax = plt.subplots(figsize=(max(4, n * 1.0), max_h * 0.8 + 1.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-0.5, max_h + 1.5)
        ax.set_aspect("equal")
        for c, h in enumerate(heights):
            for k in range(h):
                rect = mpatches.Rectangle((c, k), 1, 1,
                                          fc="#5dade2", ec="black", lw=1.5)
                ax.add_patch(rect)
            ax.text(c + 0.5, -0.25, f"col {c+1}",
                    ha="center", fontsize=10, color="#444")
        ax.set_title("Front view", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        img = self.fig_to_pil(fig)
        plt.close(fig)
        return img

    def _render_two_views(self, front_view, top_or_heights, label_heights):
        nx = len(top_or_heights)
        ny = len(top_or_heights[0])
        max_h = max(front_view) if front_view else 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
        fig.patch.set_facecolor("#ffffff")
        for ax in axes:
            ax.set_facecolor("#ffffff")

        # Front view bars with explicit unit-cell separators + col labels
        ax0 = axes[0]
        ax0.set_xlim(-0.5, ny + 0.5)
        ax0.set_ylim(-0.5, max_h + 1.5)
        ax0.set_aspect("equal")
        for c in range(ny):
            for k in range(front_view[c]):
                rect = mpatches.Rectangle((c, k), 1, 1,
                                          fc="#5dade2", ec="black", lw=1.4)
                ax0.add_patch(rect)
            ax0.text(c + 0.5, -0.25, f"col {c+1}",
                     ha="center", fontsize=10, color="#444")
            # Annotate the height number above the bar
            if front_view[c] > 0:
                ax0.text(c + 0.5, front_view[c] + 0.2, f"h={front_view[c]}",
                         ha="center", fontsize=10, color="#1b4f72",
                         fontweight="bold")
        ax0.set_title("Front view", fontsize=14, fontweight="bold")
        ax0.axis("off")

        # Top view
        ax1 = axes[1]
        ax1.set_xlim(-0.5, ny + 0.5)
        ax1.set_ylim(-0.5, nx + 0.5)
        ax1.set_aspect("equal")
        for r in range(nx):
            for c in range(ny):
                val = top_or_heights[r][c]
                if label_heights:
                    h = val
                    fc = "#abebc6" if h > 0 else "#fdfefe"
                    rect = mpatches.Rectangle((c, nx - 1 - r), 1, 1,
                                              fc=fc, ec="black", lw=1.4)
                    ax1.add_patch(rect)
                    if h > 0:
                        ax1.text(c + 0.5, nx - 1 - r + 0.5, str(h),
                                 ha="center", va="center",
                                 fontsize=18, fontweight="bold",
                                 color="black")
                    else:
                        ax1.text(c + 0.5, nx - 1 - r + 0.5, "0",
                                 ha="center", va="center",
                                 fontsize=14, color="#999")
                else:
                    fc = "#abebc6" if val == 1 else "#fdfefe"
                    rect = mpatches.Rectangle((c, nx - 1 - r), 1, 1,
                                              fc=fc, ec="black", lw=1.4)
                    ax1.add_patch(rect)
                    if val == 1:
                        ax1.text(c + 0.5, nx - 1 - r + 0.5, "?",
                                 ha="center", va="center",
                                 fontsize=18, fontweight="bold",
                                 color="#c0392b")
        for c in range(ny):
            ax1.text(c + 0.5, -0.25, f"col {c+1}",
                     ha="center", fontsize=10, color="#444")
        for r in range(nx):
            ax1.text(-0.25, nx - 0.5 - r, f"row {r+1}",
                     va="center", ha="right", fontsize=10, color="#444")
        ax1.set_title("Top view", fontsize=14, fontweight="bold")
        ax1.axis("off")
        plt.tight_layout()
        img = self.fig_to_pil(fig)
        plt.close(fig)
        return img


if __name__ == "__main__":
    import os
    out = "/tmp/env_check_mvc"
    os.makedirs(out, exist_ok=True)
    env = MultiViewCubeCountQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if ok:
                env.render().save(f"{out}/mvc_L{level}_s{seed}.png")
                print(f"L{level} s{seed} ans={env._answer}")
