"""
Grid Navigation With Obstacles QA.

NxN grid with start, end, walls, and optional one-way arrows.
Asks minimum number of moves. MCQ with 4 integer options.

Difficulty axes:
  A) grid_size: 4..7
  B) n_one_way_cells + n_walls
"""
import collections, random
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv


# Structured 4-section templates per question mode (probe = pick-direction MCQ;
# main = min-moves MCQ).
_TEMPLATES_PROBE = [
    "Decide which direction is the first optimal move on a shortest path from S to G in the {n}x{n} grid below.\n\n"
    "### Game Rules:\n"
    "1. Start at S (green cell) and reach G (red cell) in the minimum number of moves.\n"
    "2. Each move steps one cell up/down/left/right (no diagonal moves).\n"
    "3. Dark cells are walls and cannot be entered.\n"
    "4. The first move on any shortest path is what we are asking for.\n\n"
    "### Coordinate System:\n"
    "- The grid is {n}x{n}, indexed (row, col) with row 0 at the top.\n"
    "- S is at (row 0, col 0); G is at (row {n_minus_one}, col {n_minus_one}).\n"
    "- Cell symbols: `S` = start, `G` = goal, `#` = wall, `.` = open.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Choose ONE of the four options below and place its letter (A/B/C/D) inside <answer>...</answer>.\n"
    "{options}\n"
    "Example: <answer>A</answer>",

    "From S to G, identify the FIRST optimal move along any shortest path in the {n}x{n} grid.\n\n"
    "### Game Rules:\n"
    "- Each step moves one cell N/S/E/W (no diagonal).\n"
    "- Walls (dark cells) block entry.\n"
    "- We want the first move on a shortest path.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n} grid indexed (row, col), row 0 at top, col 0 at left.\n"
    "- S = start, G = goal, `#` = wall, `.` = open.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Choose ONE option letter (A/B/C/D) and place inside <answer>...</answer>.\n"
    "{options}",

    "Pick the first optimal step from S to G in the {n}x{n} grid below.\n\n"
    "### Game Rules:\n"
    "Move one cell at a time horizontally or vertically; walls cannot be entered.\n\n"
    "### Coordinate System:\n"
    "- Rows 0..{n_minus_one} top-to-bottom; columns 0..{n_minus_one} left-to-right.\n"
    "- `S`/`G`/`#`/`.` cell symbols.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output one letter (A/B/C/D) inside <answer>...</answer>.\n"
    "{options}",
]

_TEMPLATES_MAIN = [
    "Compute the minimum number of moves to reach G from S in the {n}x{n} grid below.\n\n"
    "### Game Rules:\n"
    "1. Start at S (green) and reach G (red) using up/down/left/right moves only (no diagonal).\n"
    "2. Dark cells are walls and cannot be entered.\n"
    "3. Cells with an arrow are one-way: from such a cell you may only step in the indicated direction.{arrow_clause}\n"
    "4. Find the shortest sequence of moves and report its length.\n\n"
    "### Coordinate System:\n"
    "- The grid is {n}x{n}, indexed (row, col) with row 0 at the top.\n"
    "- S is at (row 0, col 0); G is at (row {n_minus_one}, col {n_minus_one}).\n"
    "- Cell symbols: `S`, `G`, `#` = wall, `.` = open. One-way cells annotated with `^v<>` for N/S/W/E.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Choose ONE of the four options below and place its letter (A/B/C/D) inside <answer>...</answer>.\n"
    "{options}\n"
    "Example: <answer>A</answer>",

    "Find the minimum number of moves from S to G in the {n}x{n} grid below.\n\n"
    "### Game Rules:\n"
    "- Step one cell up/down/left/right at a time.\n"
    "- Walls (dark cells) block entry.\n"
    "- One-way cells (cells with an arrow) restrict the next step to the arrow's direction.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n} grid; rows 0..{n_minus_one} top-to-bottom, cols 0..{n_minus_one} left-to-right.\n"
    "- `S`, `G`, `#`, `.`. One-way symbols: `^` (N), `v` (S), `<` (W), `>` (E).\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the letter (A/B/C/D) inside <answer>...</answer>.\n"
    "{options}",

    "Determine the minimum-move path length from S to G in the {n}x{n} grid below.\n\n"
    "### Game Rules:\n"
    "Move one cell N/S/E/W per step; walls are impassable; one-way cells force the indicated direction.\n\n"
    "### Coordinate System:\n"
    "- Rows 0..{n_minus_one}, columns 0..{n_minus_one}.\n"
    "- `S` = start, `G` = goal, `#` = wall, `.` = open, `^v<>` = one-way direction.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the letter (A/B/C/D) inside <answer>...</answer>.\n"
    "{options}",
]


class GridNavigationWithObstaclesQA(StandaloneVisualEnv):
    ENV_NAME = "grid_navigation_with_obstacles"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Trace the shortest path mentally; report only the "
        "minimum-move count. Final answer in any of: <answer>X</answer>, "
        "\\boxed{{X}}, or `Final answer: X`."
    )

    @staticmethod
    def _format_state(grid, one_ways, start, goal, n) -> str:
        """Render the grid as ASCII rows (newline-separated). Cell symbols:
        `S`, `G`, `#` (wall), `.` (open). One-way arrows: `^v<>` for N/S/W/E."""
        arrow_map = {(0, -1): "^", (0, 1): "v", (-1, 0): "<", (1, 0): ">"}
        rows = []
        for y in range(n):
            row_chars = []
            for x in range(n):
                if (x, y) == start:
                    ch = "S"
                elif (x, y) == goal:
                    ch = "G"
                elif grid[y][x] == 1:
                    ch = "#"
                elif (x, y) in one_ways:
                    ch = arrow_map.get(one_ways[(x, y)], "?")
                else:
                    ch = "."
                row_chars.append(ch)
            rows.append("".join(row_chars))
        return "\n".join(rows)

    def _level_config(self, level):
        # L0..L2: probe mode — ask for FIRST optimal direction (MCQ A-D over
        # N/S/E/W). Easier because no full path-length compute needed.
        if level <= 2:
            return {
                'grid_size': 4 + level // 3,
                'n_walls': max(1, int(2 + level)),
                'n_one_way': 0,
                'probe': True,
            }
        return {
            'grid_size': 4 + level // 3,
            'n_walls': int(3 + level * 1.5),
            'n_one_way': level // 2,
            'probe': False,
        }

    def _bfs(self, grid, one_ways, start, goal, n):
        visited = {start: 0}
        q = collections.deque([start])
        while q:
            (x, y) = q.popleft()
            if (x, y) == goal:
                return visited[(x, y)]
            moves = [(1,0),(-1,0),(0,1),(0,-1)]
            # Check one-way constraints
            if (x, y) in one_ways:
                # Can only move in the one-way direction
                moves = [one_ways[(x, y)]]
            for dx, dy in moves:
                nx, ny = x+dx, y+dy
                if 0 <= nx < n and 0 <= ny < n and grid[ny][nx] == 0 and (nx,ny) not in visited:
                    visited[(nx, ny)] = visited[(x, y)] + 1
                    q.append((nx, ny))
        return None

    def _bfs_first_optimal_dirs(self, grid, one_ways, start, goal, n):
        """Return set of first-move (dx,dy) directions that lie on a shortest
        path from start to goal."""
        # BFS from goal backward to compute distances.
        dist = {goal: 0}
        q = collections.deque([goal])
        while q:
            (x, y) = q.popleft()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                if grid[ny][nx] != 0:
                    continue
                # Check that going FROM (nx,ny) TO (x,y) is allowed: respect
                # one-way at (nx,ny) — must be direction (-dx,-dy).
                if (nx, ny) in one_ways:
                    ow = one_ways[(nx, ny)]
                    if ow != (-dx, -dy):
                        continue
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
        if start not in dist:
            return set()
        opt = dist[start]
        good = set()
        # Determine which directions from start lead to dist=opt-1.
        # Respect start's one_way constraint.
        candidate_moves = [(1,0),(-1,0),(0,1),(0,-1)]
        if start in one_ways:
            candidate_moves = [one_ways[start]]
        for dx, dy in candidate_moves:
            nx, ny = start[0] + dx, start[1] + dy
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if grid[ny][nx] != 0:
                continue
            if (nx, ny) in dist and dist[(nx, ny)] == opt - 1:
                good.add((dx, dy))
        return good

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1024)
        style = self._random_style()

        n = cfg['grid_size']
        is_probe = cfg.get('probe', False)
        # In probe mode use shorter dist requirement and no one-ways for clarity.
        min_dist = 2 if is_probe else 3
        for _attempt in range(50):
            grid = [[0]*n for _ in range(n)]
            for _ in range(cfg['n_walls']):
                x, y = rng.randint(0, n-1), rng.randint(0, n-1)
                grid[y][x] = 1

            # One-way cells
            one_ways = {}
            free = [(x,y) for y in range(n) for x in range(n) if grid[y][x] == 0]
            if len(free) < 2: continue
            for _ in range(cfg['n_one_way']):
                if not free: break
                cell = rng.choice(free)
                d = rng.choice([(1,0),(-1,0),(0,1),(0,-1)])
                one_ways[cell] = d

            start = (0, 0)
            goal = (n-1, n-1)
            grid[start[1]][start[0]] = 0
            grid[goal[1]][goal[0]] = 0

            dist = self._bfs(grid, one_ways, start, goal, n)
            if dist is not None and dist >= min_dist:
                if is_probe:
                    # Require unique optimal first direction so PROBE has a
                    # single correct answer (verifier accepts only one letter).
                    fd = self._bfs_first_optimal_dirs(grid, one_ways, start, goal, n)
                    if len(fd) != 1:
                        continue
                break
        else:
            return None

        self._probe_mode = is_probe

        if is_probe:
            # PROBE: which direction is the first move on a shortest path?
            # MCQ over 4 options: N (up, dy=-1), S (down, dy=+1), E (right, dx=+1), W (left, dx=-1).
            first_dirs = self._bfs_first_optimal_dirs(grid, one_ways, start, goal, n)
            if not first_dirs:
                return None
            # Build all 4 options regardless of which are valid.
            dir_letter = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W"}
            dir_word = {(0, -1): "north (up)", (0, 1): "south (down)",
                         (1, 0): "east (right)", (-1, 0): "west (left)"}
            # Pick a valid direction as correct (deterministic by seed).
            valid_list = sorted(first_dirs, key=lambda d: dir_letter[d])
            chosen = valid_list[rng.randint(0, len(valid_list)-1)]
            options_keys = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            rng.shuffle(options_keys)
            correct_idx = options_keys.index(chosen)
            correct = "ABCD"[correct_idx]
            options_text = [f"({chr(65+i)}) Move {dir_word[options_keys[i]]}"
                            for i in range(4)]
            opt_str = "  ".join(options_text)
            self._probe_valid_dirs = first_dirs
            self._probe_options_keys = options_keys
        else:
            # Build MCQ options for path-length question.
            options = sorted(set([dist, dist+1, dist+2, max(1, dist-1)]))
            while len(options) < 4:
                options.append(options[-1] + 1)
            options = options[:4]
            rng.shuffle(options)
            correct_idx = options.index(dist)
            correct = "ABCD"[correct_idx]

        # Draw
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
                # One-way arrow
                if (x, y) in one_ways:
                    dx, dy = one_ways[(x, y)]
                    cx, cy = x + 0.5, n - 0.5 - y
                    ax.annotate('', xy=(cx + dx*0.35, cy - dy*0.35),
                                xytext=(cx - dx*0.15, cy + dy*0.15),
                                arrowprops=dict(arrowstyle='->', color='orange', lw=2))

        ax.text(start[0]+0.5, n-0.5-start[1], 'S', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(goal[0]+0.5, n-0.5-goal[1], 'G', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

        ax.set_title("Grid Navigation", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        state_str = self._format_state(grid, one_ways, start, goal, n)
        if is_probe:
            sidx = (self.seed or 0) % len(_TEMPLATES_PROBE)
            q = _TEMPLATES_PROBE[sidx].format(
                n=n,
                n_minus_one=n - 1,
                state=state_str,
                options=opt_str,
            )
        else:
            opt_str = "  ".join(f"({chr(65+i)}) {options[i]}" for i in range(4))
            arrow_clause = (" Cells with one-way arrows are present in this puzzle."
                            if one_ways else "")
            sidx = (self.seed or 0) % len(_TEMPLATES_MAIN)
            q = _TEMPLATES_MAIN[sidx].format(
                n=n,
                n_minus_one=n - 1,
                state=state_str,
                options=opt_str,
                arrow_clause=arrow_clause,
            )
        return q, correct, img

if __name__ == "__main__":
    env = GridNavigationWithObstaclesQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
