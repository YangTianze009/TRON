"""
Maze pathfinding (structured-puzzle style).

reference qids studied (maze, total 30): idx=721, 722, 727, 728, 733, 734, 745, 746
(verbatim per `design notes` §maze, lines 606-642). Plus verified
TSV row idx=721 D=1 answer = "down left down right" (single space-separated
direction words). Sample TSV format confirmed via reference dataset:
  - question_text: "Your task is to solve the maze game...up/down/left/right..."
  - answer: lowercase direction words separated by spaces, e.g. "down left down right"
  - start = green circle (top-left), end = red marker (bottom-right)

Self-contained: inlines the maze generator (random walls + BFS solvability check)
from /scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/maze/environment.py;
no `from RLVE`, `from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 3x3, low wall density (almost always direct path)
  L9 within solvable range: 8x8, denser walls
"""
import random
from collections import deque
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the {N}x{N} maze according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Move from S (top-left) to E (bottom-right) on the grid.\n"
    "2. Each move is `up`, `down`, `left`, or `right` and goes one cell.\n"
    "3. `#` cells are walls (blocked); `.` cells are open.\n\n"
    "### Coordinate System:\n"
    "- Cells indexed (row, col), 0-indexed; row 0 at top.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Output the move sequence as space-separated direction words inside <answer>...</answer>.\n"
    "Example: <answer>down left down right</answer>",

    "Solve the {N}x{N} maze below.\n\n"
    "### Game Rules:\n"
    "- Walk from S to E avoiding `#` walls.\n"
    "- Each move is up/down/left/right.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Output the move sequence inside <answer>...</answer>.",

    "Your task is to navigate the {N}x{N} maze described below.\n\n"
    "### Game Rules:\n"
    "Standard maze: S to E via 4-direction moves; `#` blocks.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Output the move sequence inside <answer>...</answer>.",
]


# L0-L2 probe templates: ask for ONLY the FIRST optimal move (one direction).
_PROBE_TEMPLATES = [
    "Your task is to identify the first optimal move in the {N}x{N} maze below:\n\n"
    "### Game Rules:\n"
    "1. Move from S (top-left) to E (bottom-right) on the grid.\n"
    "2. Each move is `up`, `down`, `left`, or `right` and goes one cell.\n"
    "3. `#` cells are walls (blocked); `.` cells are open.\n\n"
    "### Coordinate System:\n"
    "- Cells indexed (row, col), 0-indexed; row 0 at top.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Output ONLY the first move from S along a shortest path. One direction word: `up`, `down`, `left`, or `right`. Place inside <answer>...</answer>.\n"
    "Example: <answer>down</answer>",

    "Look at the {N}x{N} maze below.\n\n"
    "### Game Rules:\n"
    "- Walk from S to E avoiding `#` walls.\n"
    "- Each move is up/down/left/right.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "What is the first move from S on a shortest path to E? Output a single direction word (up/down/left/right) inside <answer>...</answer>.",

    "Maze ({N}x{N}).\n\n"
    "### Game Rules:\n"
    "Standard maze: S to E via 4-direction moves; `#` blocks.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Provide just the first direction (up/down/left/right) of an optimal path inside <answer>...</answer>.",
]


def _maze_state_text(maze, N):
    rows = []
    for r in range(N):
        row = []
        for c in range(N):
            if r == 0 and c == 0:
                row.append("S")
            elif r == N - 1 and c == N - 1:
                row.append("E")
            else:
                row.append(maze[r][c])
        rows.append("".join(row))
    return "\n".join(rows)


_DIR_DELTA = {
    "up":    (-1, 0),
    "down":  (+1, 0),
    "left":  (0, -1),
    "right": (0, +1),
}


def _generate_maze(N: int, density: float, rng: random.Random,
                   max_tries: int = 50) -> Optional[Tuple[List[List[str]], List[str]]]:
    """Generate an NxN maze (`#` walls, `.` open) and a shortest path from
    (0,0) to (N-1,N-1) as a list of direction words.

    Inlined from /scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/maze/environment.py
    """
    for _ in range(max_tries):
        maze = [["#" if rng.random() < density else "." for _ in range(N)]
                for _ in range(N)]
        maze[0][0] = maze[N - 1][N - 1] = "."
        # BFS for reachability + parent tracking
        prev = [[None] * N for _ in range(N)]
        prev[0][0] = (0, 0)
        q = deque([(0, 0)])
        while q:
            x, y = q.popleft()
            for d, (dx, dy) in _DIR_DELTA.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N and maze[nx][ny] == "." \
                        and prev[nx][ny] is None:
                    prev[nx][ny] = (x, y)
                    q.append((nx, ny))
        if prev[N - 1][N - 1] is not None:
            # Reconstruct path
            path = []
            x, y = N - 1, N - 1
            while (x, y) != (0, 0):
                px, py = prev[x][y]
                for d, (dx, dy) in _DIR_DELTA.items():
                    if (x, y) == (px + dx, py + dy):
                        path.append(d)
                        break
                x, y = px, py
            path.reverse()
            return maze, path
    return None


class MazePathfindQA(StandaloneVisualEnv):
    ENV_NAME = "maze_pathfind"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the move sequence directly inside "
        "`<answer>...</answer>` tags as space-separated direction words."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode — small maze, ask for first move only.
        # L3..L9: full path (existing).
        if level == 0:
            N, density, probe = 3, 0.05, True
        elif level == 1:
            N, density, probe = 3, 0.18, True
        elif level == 2:
            N, density, probe = 4, 0.18, True
        elif level == 3:
            N, density, probe = 4, 0.25, False
        elif level == 4:
            N, density, probe = 5, 0.25, False
        elif level == 5:
            N, density, probe = 5, 0.30, False
        elif level == 6:
            N, density, probe = 6, 0.28, False
        elif level == 7:
            N, density, probe = 6, 0.32, False
        elif level == 8:
            N, density, probe = 7, 0.30, False
        else:
            N, density, probe = 8, 0.30, False
        return {"N": N, "density": density, "level": level, "probe": probe}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, density = cfg["N"], cfg["density"]
        rng = random.Random((seed or 0) * 991 + level * 31 + 11)

        result = _generate_maze(N, density, rng)
        if result is None:
            return None
        maze, path = result

        self._maze = maze
        self._N = N
        self._probe_mode = cfg["probe"]

        # In probe mode, store full path for verifier; answer is first move.
        # For probe verification, we accept any direction that's the first
        # step of *some* shortest path (BFS gives one shortest, but multiple
        # shortest moves at the start may exist).
        if self._probe_mode:
            # Compute set of first-move directions on any shortest path from
            # (0,0) to (N-1,N-1).
            self._probe_first_dirs = self._compute_first_optimal_dirs(maze, N)
            # Pick a random valid direction as the canonical answer (so the
            # dataset shows variety; verifier accepts any of the valid set).
            valid_dirs = sorted(self._probe_first_dirs)
            if not valid_dirs:
                return None
            ans_str = rng.choice(valid_dirs)
            sidx = (seed or 0) % len(_PROBE_TEMPLATES)
            state_text = _maze_state_text(maze, N)
            question = _PROBE_TEMPLATES[sidx].format(N=N, state=state_text)
        else:
            sidx = (seed or 0) % len(_TEMPLATES)
            state_text = _maze_state_text(maze, N)
            question = _TEMPLATES[sidx].format(N=N, state=state_text)
            ans_str = " ".join(path)

        img = self._render(maze, N)
        return question, ans_str, img

    def _compute_first_optimal_dirs(self, maze, N):
        """Compute the set of directions D such that moving D from (0,0)
        leads to a cell whose shortest-path distance to E is exactly
        (total_optimal_dist - 1). I.e., D is the first move on at least one
        optimal path.
        """
        # BFS from end (N-1,N-1) to build distance map.
        dist = [[None] * N for _ in range(N)]
        dist[N-1][N-1] = 0
        q = deque([(N-1, N-1)])
        while q:
            x, y = q.popleft()
            for d, (dx, dy) in _DIR_DELTA.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N and maze[nx][ny] == "." \
                        and dist[nx][ny] is None:
                    dist[nx][ny] = dist[x][y] + 1
                    q.append((nx, ny))
        if dist[0][0] is None:
            return set()
        opt_total = dist[0][0]
        # First-move directions that lead to a cell with dist = opt_total - 1.
        good = set()
        for d, (dx, dy) in _DIR_DELTA.items():
            nx, ny = dx, dy
            if 0 <= nx < N and 0 <= ny < N and maze[nx][ny] == "." \
                    and dist[nx][ny] is not None \
                    and dist[nx][ny] == opt_total - 1:
                good.add(d)
        return good

    # ---------------------------------------------------------------- render
    def _render(self, maze: List[List[str]], N: int) -> Image.Image:
        cell_size = max(40, 400 // N)
        margin = 0.6
        fig_w = (N * cell_size / 100) + 1 + margin * 0.4
        fig_h = (N * cell_size / 100) + 1 + margin * 0.4
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)

        for i in range(N):
            for j in range(N):
                color = "#2c3e50" if maze[i][j] == "#" else "#ecf0f1"
                rect = patches.Rectangle(
                    (j, N - 1 - i), 1, 1,
                    linewidth=0.5, edgecolor="#bdc3c7", facecolor=color
                )
                ax.add_patch(rect)

        label_font = max(7, 14 - N // 3)
        for j in range(N):
            ax.text(j + 0.5, N + 0.25, str(j), ha="center", va="bottom",
                    fontsize=label_font, fontweight="bold", color="#555555")
        for i in range(N):
            ax.text(-0.25, N - 1 - i + 0.5, str(i), ha="right", va="center",
                    fontsize=label_font, fontweight="bold", color="#555555")

        # Start marker (green circle)
        ax.add_patch(patches.Circle(
            (0.5, N - 0.5), 0.3, facecolor="#27ae60",
            edgecolor="white", linewidth=1.5
        ))
        ax.text(0.5, N - 0.5, "S", ha="center", va="center",
                fontsize=max(8, 16 - N // 3), fontweight="bold", color="white")

        # End marker (red circle)
        ax.add_patch(patches.Circle(
            (N - 0.5, 0.5), 0.3, facecolor="#e74c3c",
            edgecolor="white", linewidth=1.5
        ))
        ax.text(N - 0.5, 0.5, "E", ha="center", va="center",
                fontsize=max(8, 16 - N // 3), fontweight="bold", color="white")

        ax.set_xlim(-margin, N + 0.1)
        ax.set_ylim(-0.1, N + margin)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Pull direction words (lowercase, robust to punctuation/extra text).
        # Also accept single-letter forms u/d/l/r.
        import re
        words = re.findall(r"\b(up|down|left|right|u|d|l|r)\b",
                           predicted.lower())
        if not words:
            return False
        moves = []
        letter_map = {"u": "up", "d": "down", "l": "left", "r": "right"}
        for w in words:
            moves.append(letter_map.get(w, w))

        # PROBE MODE: only check first direction matches an optimal first move.
        if getattr(self, "_probe_mode", False):
            return moves[0] in getattr(self, "_probe_first_dirs", set())

        # Simulate
        N = self._N
        maze = self._maze
        x, y = 0, 0
        for m in moves:
            dx, dy = _DIR_DELTA[m]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < N and 0 <= ny < N):
                return False
            if maze[nx][ny] == "#":
                return False
            x, y = nx, ny
        return (x, y) == (N - 1, N - 1)
