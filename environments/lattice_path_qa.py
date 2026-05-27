"""
Lattice Path QA environment.

Draws a grid with start (top-left) and end (bottom-right), possibly with
blocked cells. Questions about counting paths (right/down moves only)
and shortest path length.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class LatticePathQA(StandaloneVisualEnv):
    ENV_NAME = "lattice_path"

    QUESTION_TYPES = [
        "count_paths", "count_paths_with_obstacles", "shortest_path_length",
    ]

    def _level_config(self, level: int) -> dict:
        """Return difficulty config for a given level (0-9)."""
        lv = max(0, min(9, level))
        # Level-scaled grid size so L0 != L9 structurally.
        # L0-1: tiny 2x2/2x3, L9: up to 6x6.
        size_ranges = [
            (2, 3),   # L0
            (2, 3),   # L1
            (2, 4),   # L2
            (3, 4),   # L3
            (3, 5),   # L4
            (3, 5),   # L5
            (4, 5),   # L6
            (4, 6),   # L7
            (5, 6),   # L8
            (5, 6),   # L9
        ]
        qtypes = [
            'count_paths', 'count_paths',
            'shortest_path_length', 'count_paths',
            'count_paths_with_obstacles', 'count_paths_with_obstacles',
            'count_paths_with_obstacles', 'count_paths_with_obstacles',
            'count_paths_with_obstacles', 'count_paths_with_obstacles',
        ]
        lo, hi = size_ranges[lv]
        return {
            'question_type': qtypes[lv],
            'grid_min': lo,
            'grid_max': hi,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        # ── Level routing (auto-generated) ──
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)

        qtype = parameter.get("question_type", self._rng.choice(self.QUESTION_TYPES))
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choice(self.QUESTION_TYPES)

        # Re-seed self._rng to differentiate levels — otherwise L0/L3 with
        # identical qtype produce identical grids for same seed, and
        # L6/L9 produce identical obstacle layouts.
        self._rng.seed((self.seed or 0) * 1000 + level * 37 + 991)

        grid_min = parameter.get("grid_min", 3)
        grid_max = parameter.get("grid_max", 6)

        for _ in range(20):
            result = self._try_generate(qtype, grid_min, grid_max)
            if result is not None:
                return result
        return None

    # -------------------------------------------------------------- #
    # Path counting algorithms
    # -------------------------------------------------------------- #

    def _count_paths_no_obstacles(self, rows, cols):
        """C(rows+cols, rows) paths on a rows x cols grid (moves: right or down)."""
        return math.comb(rows + cols, rows)

    def _count_paths_dp(self, rows, cols, blocked):
        """Count paths using DP with blocked cells.
        Grid is (rows+1) x (cols+1) intersection points.
        Move from (0,0) to (rows, cols) going right (+col) or down (+row).
        """
        nrows = rows + 1
        ncols = cols + 1
        dp = [[0] * ncols for _ in range(nrows)]

        if (0, 0) in blocked:
            return 0
        dp[0][0] = 1

        for r in range(nrows):
            for c in range(ncols):
                if r == 0 and c == 0:
                    continue
                if (r, c) in blocked:
                    dp[r][c] = 0
                    continue
                val = 0
                if r > 0:
                    val += dp[r - 1][c]
                if c > 0:
                    val += dp[r][c - 1]
                dp[r][c] = val
        return dp[rows][cols]

    def _shortest_with_obstacles(self, rows, cols, blocked):
        """BFS shortest path from (0,0) to (rows,cols) moving right or down."""
        from collections import deque
        start = (0, 0)
        end = (rows, cols)
        if start in blocked or end in blocked:
            return -1
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            (r, c), d = queue.popleft()
            for nr, nc in [(r + 1, c), (r, c + 1)]:
                if nr <= rows and nc <= cols and (nr, nc) not in visited and (nr, nc) not in blocked:
                    if (nr, nc) == end:
                        return d + 1
                    visited.add((nr, nc))
                    queue.append(((nr, nc), d + 1))
        return -1

    # -------------------------------------------------------------- #
    # Problem generation
    # -------------------------------------------------------------- #

    def _try_generate(self, qtype: str, grid_min: int = 3, grid_max: int = 6
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        rows = rng.randint(grid_min, grid_max)
        cols = rng.randint(grid_min, grid_max)

        if qtype == "count_paths":
            answer = self._count_paths_no_obstacles(rows, cols)
            blocked = set()
            img = self._render(rows, cols, blocked)
            q = (f"On this {rows}x{cols} grid, how many distinct paths are there "
                 f"from S (top-left) to E (bottom-right), moving only right or "
                 f"down along the grid lines?")
            return q, str(answer), img

        elif qtype == "count_paths_with_obstacles":
            # Place 1-3 obstacles (not at start or end)
            num_obstacles = rng.randint(1, min(3, rows * cols // 3))
            blocked = set()
            all_interior = [(r, c) for r in range(rows + 1) for c in range(cols + 1)
                            if (r, c) != (0, 0) and (r, c) != (rows, cols)]
            if len(all_interior) < num_obstacles:
                return None
            obstacle_cells = rng.sample(all_interior, num_obstacles)
            blocked = set(obstacle_cells)

            answer = self._count_paths_dp(rows, cols, blocked)
            if answer <= 0:
                return None  # No valid paths; retry

            img = self._render(rows, cols, blocked)
            q = (f"On this {rows}x{cols} grid with {num_obstacles} blocked "
                 f"intersection(s) (marked X), how many distinct paths are there "
                 f"from S to E, moving only right or down?")
            return q, str(answer), img

        elif qtype == "shortest_path_length":
            num_obstacles = rng.randint(1, min(3, rows * cols // 3))
            all_interior = [(r, c) for r in range(rows + 1) for c in range(cols + 1)
                            if (r, c) != (0, 0) and (r, c) != (rows, cols)]
            if len(all_interior) < num_obstacles:
                return None
            blocked = set(rng.sample(all_interior, num_obstacles))

            sp = self._shortest_with_obstacles(rows, cols, blocked)
            if sp < 0:
                return None
            # Without obstacles shortest is rows+cols; with obstacles same or impossible
            img = self._render(rows, cols, blocked)
            q = (f"What is the shortest path length (number of steps) from S to E "
                 f"on this grid, moving only right or down and avoiding blocked "
                 f"intersections (marked X)?")
            return q, str(sp), img

        return None

    # -------------------------------------------------------------- #
    # Rendering
    # -------------------------------------------------------------- #

    def _render(self, rows, cols, blocked):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(7, cols + 2) * sc, max(7, rows + 2) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        lw = style["line_width"]
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw grid lines
        for r in range(rows + 1):
            ax.plot([0, cols], [rows - r, rows - r], color=palette[7],
                    linewidth=lw * 0.7, zorder=1)
        for c in range(cols + 1):
            ax.plot([c, c], [0, rows], color=palette[7],
                    linewidth=lw * 0.7, zorder=1)

        # Draw intersection points
        for r in range(rows + 1):
            for c in range(cols + 1):
                y = rows - r
                if (r, c) in blocked:
                    ax.plot(c, y, 'rx', markersize=16, markeredgewidth=3, zorder=4)
                    ax.add_patch(patches.Rectangle(
                        (c - 0.3, y - 0.3), 0.6, 0.6,
                        facecolor=palette[1], alpha=0.2, zorder=2))
                else:
                    ax.plot(c, y, 'ko', markersize=6, zorder=3)

        # Row/column labels
        for r in range(rows + 1):
            ax.text(-0.4, rows - r, str(r), ha='center', va='center',
                    fontsize=fs - 3, color='#7f8c8d', fontfamily=ff)
        for c in range(cols + 1):
            ax.text(c, rows + 0.4, str(c), ha='center', va='center',
                    fontsize=fs - 3, color='#7f8c8d', fontfamily=ff)

        # Start marker
        ax.plot(0, rows, 'o', color=palette[2], markersize=20, zorder=5)
        ax.text(0, rows, 'S', ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white', zorder=6,
                fontfamily=ff)

        # End marker
        ax.plot(cols, 0, 's', color=palette[1], markersize=20, zorder=5)
        ax.text(cols, 0, 'E', ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white', zorder=6,
                fontfamily=ff)

        # Direction arrows
        ax.annotate('', xy=(1.5, rows + 0.8), xytext=(0.5, rows + 0.8),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=lw * 0.7))
        ax.text(1.0, rows + 1.1, 'right', ha='center', fontsize=fs - 3,
                color='#2c3e50', fontfamily=ff)

        ax.annotate('', xy=(-0.8, rows - 1.5), xytext=(-0.8, rows - 0.5),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=lw * 0.7))
        ax.text(-0.8, rows - 1.0, 'down', ha='center', fontsize=fs - 3,
                color='#2c3e50', rotation=90, fontfamily=ff)

        ax.set_xlim(-1.2, cols + 0.8)
        ax.set_ylim(-0.8, rows + 1.5)
        title = 'Lattice Grid'
        if blocked:
            title += f' ({len(blocked)} blocked)'
        ax.set_title(title, fontsize=fs + 2, fontweight='bold', pad=12,
                     fontfamily=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
