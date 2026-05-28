"""
Sliding (15-)Puzzle — structured-puzzle style game.

Rules (matching a puzzle benchmark slidingpuzzle benchmark):
  - 4x4 grid with tiles 1..15 plus one empty cell (0).
  - On each move, slide a tile orthogonally adjacent to the empty cell
    INTO the empty cell.
  - Goal: tiles in row-major order with empty in the bottom-right cell.

IMPORTANT (direction semantics — matches benchmark prompt verbatim):
  - "up"    means the tile BELOW the empty space moves up into the empty space.
            (Equivalently, empty moves DOWN.)
  - "down"  means the tile ABOVE the empty space moves down into the empty space.
            (Equivalently, empty moves UP.)
  - "left"  means the tile to the RIGHT of the empty space moves left.
            (Equivalently, empty moves RIGHT.)
  - "right" means the tile to the LEFT of the empty space moves right.
            (Equivalently, empty moves LEFT.)

Verified by tracing benchmark sample idx=961:
  start `[[1,2,3,4],[5,6,7,8],[9,0,10,12],[13,14,11,15]]`, answer `left up left`.
  step "left": tile 10 at (2,2) moves left → empty moves (2,1)→(2,2).
  step "up":   tile 11 at (3,2) moves up   → empty moves (2,2)→(3,2).
  step "left": tile 15 at (3,3) moves left → empty moves (3,2)→(3,3) = goal.

Answer format: sequence of words separated by spaces, each word in
{up, down, left, right}, describing the direction the TILE moves.

Difficulty axis: distance from goal (number of random scrambles).
"""
import random
import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "\nYour task is to solve the 15-puzzle game according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. The puzzle is played on a 4x4 grid with 15 numbered tiles and one empty space\n"
    "2. You can only move tiles horizontally or vertically into the empty space\n"
    "3. The goal is to arrange the tiles in numerical order with:\n"
    "   - First row: 1, 2, 3, 4\n"
    "   - Second row: 5, 6, 7, 8\n"
    "   - Third row: 9, 10, 11, 12\n"
    "   - Fourth row: 13, 14, 15, empty space\n\n"
    "### Coordinate System:\n"
    "- The grid positions are numbered from left to right and top to bottom\n"
    "- Columns (horizontal): numbered 1, 2, 3, 4 from left to right\n"
    "- Rows (vertical): numbered 1, 2, 3, 4 from top to bottom\n"
    "- Each position can be identified by its row and column (row, column)\n\n"
    "### Current Puzzle State:\n"
    "The initial_state {state} represents a 4x4 grid reading from left to right, top to bottom, where 0 represents the empty space and numbers 1-15 represent the tiles.\n\n"
    "### Output Format Requirements:\n"
    "\"up\" means the tile below the empty space moves up into the empty space\n"
    "\"down\" means the tile above the empty space moves down into the empty space\n"
    "\"left\" means the tile to the right of the empty space moves left into the empty space\n"
    "\"right\" means the tile to the left of the empty space moves right into the empty space\n\n"
    "Your final answer format should be given like: up down up left right\n",
]


# Direction word → direction the EMPTY cell moves.
# Benchmark semantics: word names the tile's direction; empty moves opposite.
_DIRS = {
    "up": (1, 0),     # tile below moves up  → empty moves DOWN (+row)
    "down": (-1, 0),  # tile above moves down → empty moves UP (-row)
    "left": (0, 1),   # tile right moves left → empty moves RIGHT (+col)
    "right": (0, -1), # tile left moves right → empty moves LEFT (-col)
}
_INV_DIR = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _find_zero(state: Tuple[Tuple[int, ...], ...]) -> Tuple[int, int]:
    for r, row in enumerate(state):
        for c, v in enumerate(row):
            if v == 0:
                return (r, c)
    raise ValueError("zero not found")


def _apply_move(state: Tuple[Tuple[int, ...], ...], dr: int, dc: int,
                n: int) -> Optional[Tuple[Tuple[int, ...], ...]]:
    r0, c0 = _find_zero(state)
    nr, nc = r0 + dr, c0 + dc
    if not (0 <= nr < n and 0 <= nc < n):
        return None
    new = [list(row) for row in state]
    new[r0][c0], new[nr][nc] = new[nr][nc], new[r0][c0]
    return tuple(tuple(row) for row in new)


def _goal_state(n: int) -> Tuple[Tuple[int, ...], ...]:
    flat = list(range(1, n * n)) + [0]
    return tuple(tuple(flat[r * n:(r + 1) * n]) for r in range(n))


def _bfs_solve(start: Tuple[Tuple[int, ...], ...], n: int,
               max_states: int = 200000) -> Optional[List[str]]:
    """BFS over states; return a list of direction words (direction the
    TILE moves; matches benchmark)."""
    goal = _goal_state(n)
    if start == goal:
        return []
    visited = {start}
    parent: Dict[Tuple[Tuple[int, ...], ...],
                  Tuple[Tuple[Tuple[int, ...], ...], str]] = {}
    q = deque([start])
    states_examined = 0
    while q:
        cur = q.popleft()
        states_examined += 1
        if states_examined > max_states:
            return None
        for name, (dr, dc) in _DIRS.items():
            nxt = _apply_move(cur, dr, dc, n)
            if nxt is None or nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = (cur, name)
            if nxt == goal:
                # reconstruct
                path: List[str] = []
                x = nxt
                while x in parent:
                    p, mv = parent[x]
                    path.append(mv)
                    x = p
                return list(reversed(path))
            q.append(nxt)
    return None


def _scramble_from_goal(n: int, scramble_steps: int,
                        rng: random.Random) -> Tuple[Tuple[Tuple[int, ...], ...],
                                                     int]:
    """Apply random valid moves to the goal state to get a guaranteed
    solvable start. Returns (state, actual_unique_moves)."""
    state = _goal_state(n)
    last_dir = None
    actual = 0
    for _ in range(scramble_steps):
        dirs = list(_DIRS.keys())
        rng.shuffle(dirs)
        for name in dirs:
            if last_dir is not None and name == _INV_DIR[last_dir]:
                continue  # avoid undo
            dr, dc = _DIRS[name]
            nxt = _apply_move(state, dr, dc, n)
            if nxt is not None:
                state = nxt
                last_dir = name
                actual += 1
                break
    return state, actual


class SlidingPuzzleQA(StandaloneVisualEnv):
    ENV_NAME = "sliding_puzzle"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the sequence of direction words "
        "(`up`/`down`/`left`/`right`) directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Benchmark is always 4x4 (15-puzzle). Difficulty = scramble depth.
        n = 4
        if level == 0:
            scramble = 1
        elif level == 1:
            scramble = 2
        elif level == 2:
            scramble = 4
        elif level == 3:
            scramble = 6
        elif level == 4:
            scramble = 8
        elif level == 5:
            scramble = 11
        elif level == 6:
            scramble = 14
        elif level == 7:
            scramble = 17
        elif level == 8:
            scramble = 21
        else:
            scramble = 25
        return {"level": level, "n": n, "scramble": scramble}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        scramble = cfg["scramble"]
        rng = random.Random((self.seed or 0) * 4231 + level * 71 + 23)

        for _attempt in range(8):
            start, _actual = _scramble_from_goal(n, scramble, rng)
            if start == _goal_state(n):
                continue
            sol = _bfs_solve(start, n)
            if sol is None or len(sol) == 0:
                continue
            # Cache for verifier
            self._n = n
            self._start = start
            ans_str = " ".join(sol)
            sidx = (self.seed or 0) % len(_TEMPLATES)
            # Convert start (tuple of tuples) to nested-list str matching
            # benchmark sample format `[[1,2,3,4],[5,6,7,8],[9,0,10,12],[13,14,11,15]]`.
            state_list = [list(row) for row in start]
            state_text = str(state_list)
            question = _TEMPLATES[sidx].format(state=state_text)
            # 2026-05-04: simplified L0 (was 10% too-hard) — at L0 the puzzle
            # is just 1 scramble step (so 1 move solves it); add concise hint.
            if level <= 1:
                question += (
                    "\nBe concise. The puzzle is only one or two moves from "
                    "the goal — find the single tile that's misplaced and "
                    "the move that puts it back. Output the direction word(s) "
                    "directly inside <answer>...</answer>."
                )
            img = self._render(start, n)
            return question, ans_str, img
        return None

    def _render(self, state, n) -> Image.Image:
        cell_in = 0.65
        fig_w = (n + 0.6) * cell_in + 0.3
        fig_h = (n + 0.6) * cell_in + 0.3
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for r in range(n):
            for c in range(n):
                v = state[r][c]
                if v == 0:
                    bg = "#ffffff"
                    edge = "#bbb"
                    txt = ""
                else:
                    bg = "#3498db"
                    edge = "#1a3a6e"
                    txt = str(v)
                ax.add_patch(patches.FancyBboxPatch(
                    (c + 0.05, n - 1 - r + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.04",
                    linewidth=1.5, edgecolor=edge, facecolor=bg,
                ))
                if txt:
                    ax.text(c + 0.5, n - 1 - r + 0.5, txt,
                            ha="center", va="center",
                            fontsize=max(12, 26 - n * 3),
                            fontweight="bold", color="white")
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        n = self._n
        s = predicted.strip().lower()
        s = re.sub(r"```[^\n]*\n", "", s)
        s = s.replace("```", "").strip()
        # Extract direction words
        words = re.findall(r"\b(up|down|left|right)\b", s)
        if not words:
            return False
        # Apply moves to start state (each word = direction the TILE moves;
        # _DIRS maps word → empty-cell delta consistently).
        state = self._start
        for w in words:
            dr, dc = _DIRS[w]
            nxt = _apply_move(state, dr, dc, n)
            if nxt is None:
                return False
            state = nxt
        return state == _goal_state(n)
