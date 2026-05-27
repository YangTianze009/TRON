"""
Path Length Grid Count QA.

NxN grid with start/end cells and walls. Asks for the shortest path length.

Difficulty axes:
  A) grid_size: 4..8
  B) n_walls: 2..20
"""
import collections, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class PathLengthGridCountQA(StandaloneVisualEnv):
    ENV_NAME = "path_length_grid_count"

    def _level_config(self, level):
        # 2026-05-04: simplified L0 (was 7.5% too-hard) — 3x3 with 1 wall.
        if level == 0:
            return {'grid_size': 3, 'n_walls': 1}
        if level == 1:
            return {'grid_size': 4, 'n_walls': 2}
        return {
            'grid_size': 4 + level // 2,
            'n_walls': 2 + level * 2,
        }

    def _bfs(self, grid, start, goal, n):
        visited = {start: 0}
        q = collections.deque([start])
        while q:
            (x, y) = q.popleft()
            if (x, y) == goal:
                return visited[(x, y)]
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < n and 0 <= ny < n and grid[ny][nx] == 0 and (nx, ny) not in visited:
                    visited[(nx, ny)] = visited[(x, y)] + 1
                    q.append((nx, ny))
        return None

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1013)
        style = self._random_style()

        n = cfg['grid_size']
        for _attempt in range(40):
            grid = [[0]*n for _ in range(n)]
            # Place walls
            wall_count = 0
            for _ in range(cfg['n_walls']):
                x, y = rng.randint(0, n-1), rng.randint(0, n-1)
                grid[y][x] = 1
                wall_count += 1

            # Start/goal
            free = [(x,y) for y in range(n) for x in range(n) if grid[y][x] == 0]
            if len(free) < 2: continue
            start = free[0]
            goal = free[-1]
            grid[start[1]][start[0]] = 0
            grid[goal[1]][goal[0]] = 0

            dist = self._bfs(grid, start, goal, n)
            if dist is not None and dist >= 3:
                break
        else:
            return None

        answer = str(dist)

        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(max(5, n+1)*sc, max(5, n+1)*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_xlim(-0.1, n+0.1); ax.set_ylim(-0.1, n+0.1)
        ax.set_aspect('equal'); ax.axis('off')

        for y in range(n):
            for x in range(n):
                if (x, y) == start:
                    color = '#2ecc71'
                elif (x, y) == goal:
                    color = '#e74c3c'
                elif grid[y][x] == 1:
                    color = '#2c3e50'
                else:
                    color = '#ecf0f1'
                rect = mpatches.FancyBboxPatch((x, n-1-y), 1, 1,
                    facecolor=color, edgecolor='#555', linewidth=1.2)
                ax.add_patch(rect)

        # Labels
        ax.text(start[0]+0.5, n-0.5-start[1], 'S', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(goal[0]+0.5, n-0.5-goal[1], 'G', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

        ax.set_title("Shortest Path", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q = ("The grid shows a start cell (green, S) and a goal cell (red, G). "
             "Dark cells are walls. Moving up/down/left/right only, "
             "what is the shortest path length (number of steps) from S to G? "
             "Answer with a single integer.")
        if level <= 1:
            q += (" Be concise. Trace the shortest sequence of "
                  "up/down/left/right moves from S to G avoiding dark walls; "
                  "output only the integer step count.")
        return q, answer, img

if __name__ == "__main__":
    env = PathLengthGridCountQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
