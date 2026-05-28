"""
Maze Solution Length QA (batch 3, 2026-04-14).

Target: reference games / a spatial benchmark mazenav + a puzzle benchmark. Renders a small
grid maze with walls, start (S), and goal (G). Asks for the minimum path
length (in cells stepped on excluding start, or BFS distance).

Format: constant integer.

Difficulty axes:
  A) Pattern A: grid size (3..7).
  B) Pattern D: wall density (0.15..0.40).
  C) Pattern E: chain depth via longer shortest path.
"""
import collections
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class MazeSolutionLengthQA(StandaloneVisualEnv):
    ENV_NAME = "maze_solution_length"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "grid_size": 3 + level // 2,            # 3..7
            "wall_density": 0.1 + 0.03 * level,     # 0.1..0.37
            "min_path_len": 2 + level // 2,         # 2..6
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["grid_size"] * 10 + int(cfg["wall_density"] * 100)

        for _ in range(40):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _bfs(self, grid, start, goal):
        n = len(grid)
        visited = {start: 0}
        q = collections.deque([start])
        while q:
            (x, y) = q.popleft()
            if (x, y) == goal:
                return visited[(x, y)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and grid[ny][nx] == 0:
                    if (nx, ny) not in visited:
                        visited[(nx, ny)] = visited[(x, y)] + 1
                        q.append((nx, ny))
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["grid_size"]
        grid = [[0] * n for _ in range(n)]

        # Place walls
        n_walls = int(cfg["wall_density"] * n * n)
        for _ in range(n_walls):
            x, y = rng.randint(0, n - 1), rng.randint(0, n - 1)
            grid[y][x] = 1

        # Choose start at top-left free cell, goal at bottom-right area
        candidates = [(x, y) for y in range(n) for x in range(n) if grid[y][x] == 0]
        if len(candidates) < 2:
            return None
        start = rng.choice(candidates[: max(1, len(candidates) // 3)])
        goal = rng.choice(candidates[max(1, 2 * len(candidates) // 3):])
        if start == goal:
            return None
        grid[start[1]][start[0]] = 0
        grid[goal[1]][goal[0]] = 0

        dist = self._bfs(grid, start, goal)
        if dist is None or dist < cfg["min_path_len"]:
            return None
        # Keep answer reasonable
        if dist > 3 * n:
            return None

        answer = str(dist)
        q = ("The image shows a small maze (gray = wall, white = open). "
             "'S' marks the start cell and 'G' marks the goal cell. Moves "
             "are horizontal or vertical only. What is the minimum number "
             "of moves required to go from S to G? Answer with an integer.")
        image = self._render(grid, start, goal, cfg)
        return q, answer, image

    def _render(self, grid, start, goal, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]

        n = len(grid)
        fig, ax = plt.subplots(figsize=(5.5 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")

        for y in range(n):
            for x in range(n):
                if grid[y][x] == 1:
                    fc = "#2d3436"
                else:
                    fc = "#fdfdfd"
                rect = mpatches.Rectangle(
                    (x, n - 1 - y), 1, 1,
                    facecolor=fc, edgecolor="#555", linewidth=1.0)
                ax.add_patch(rect)

        sx, sy = start
        gx, gy = goal
        ax.text(sx + 0.5, (n - 1 - sy) + 0.5, "S",
                fontsize=fs + 8, fontweight="bold",
                ha="center", va="center", color="#16a085")
        ax.text(gx + 0.5, (n - 1 - gy) + 0.5, "G",
                fontsize=fs + 8, fontweight="bold",
                ha="center", va="center", color="#c0392b")

        ax.set_xlim(-0.2, n + 0.2)
        ax.set_ylim(-0.2, n + 0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = MazeSolutionLengthQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[maze_solution_length L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"maze_solution_length_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[maze_solution_length L{level} s{s}] A={env._answer}")
