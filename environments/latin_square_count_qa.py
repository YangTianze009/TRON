"""
Latin-square count puzzle.

A partly filled NxN grid of digits 1..N (each row and column must contain
each digit exactly once). One empty cell is marked with `x`. Question:
"How many distinct digit values can replace `x` (across all valid
completions of the rest of the grid)?"

Reference qid 315:
  "Write the numbers 1, 2, 3, 4, 5 in the square 5x5 in such a way that
   every row and every column has each number. How many kinds of number
   can be put to replace 'x'?" Ans: 3.

L0: 3x3 with 8/9 cells filled (only `x` empty). Trivially answer = 1.
L9: 5x5 with several empty cells; counting requires real backtracking.
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "In the {N}x{N} grid below, each row and each column must contain every digit from 1 to {N} exactly once. How many distinct values can replace the cell marked x? Place the integer in <answer>...</answer>.",
    "Latin square: every row and column of the {N}x{N} grid contains each digit 1..{N} once. How many different digits could go in the cell labeled x? Integer answer in <answer>...</answer>.",
    "Consider the partially filled {N}x{N} Latin square. Each row and each column must use the digits 1 through {N} exactly once. How many digits can validly fill the cell marked x? Integer in <answer>...</answer>.",
    "The {N}x{N} grid is a Latin square (each row and column contains 1..{N} once). For the empty cell marked x, count the number of digit values that could legitimately go there. Integer in <answer>...</answer>.",
    "How many different digits from 1..{N} can be placed in the cell marked x of this {N}x{N} Latin square (each row and column must have every digit once)? Integer answer in <answer>...</answer>.",
    "Latin square completion: the {N}x{N} grid below has rows and columns containing 1..{N} once each. For the cell labeled x, count the distinct possible digit values. Integer in <answer>...</answer>.",
    "Examine the partly filled {N}x{N} Latin square. Each row and column needs the digits 1..{N} (no repeats). How many digits could replace x? Place integer in <answer>...</answer>.",
    "In a Latin square, every row and every column contains each digit from 1 to {N} exactly once. For this {N}x{N} grid, how many digit values are valid for the cell marked x? Integer answer in <answer>...</answer>.",
    "Count the number of valid digits that can be placed at the cell labeled x in this {N}x{N} Latin square. Integer in <answer>...</answer>.",
    "Latin-square puzzle: each row and column of the {N}x{N} grid must contain digits 1..{N} once. How many distinct values could fill cell x? Place integer in <answer>...</answer>.",
    "The grid is a {N}x{N} Latin square (rows and columns are permutations of 1..{N}). Determine how many digits are possible at the cell marked x. Integer in <answer>...</answer>.",
    "Find the number of different digit values that can validly occupy the cell labeled x in the displayed {N}x{N} Latin square. Integer answer in <answer>...</answer>.",
    "A {N}x{N} grid is partially filled to be a Latin square (every row and column lists 1..{N} once). For the empty cell x, count distinct digits that could go there. Integer in <answer>...</answer>.",
    "How many digits in 1..{N} can fill cell x so that the {N}x{N} grid completes to a Latin square (each row and column contains 1..{N} once)? Integer in <answer>...</answer>.",
    "Partial Latin square ({N}x{N}): each row and column contains 1..{N} exactly once. Count valid digit choices for cell x. Integer answer in <answer>...</answer>.",
    "Determine the count of digit values from 1 to {N} that can be placed in the cell marked x of the {N}x{N} Latin square shown. Integer in <answer>...</answer>.",
]


def _all_valid_completions(grid: List[List[Optional[int]]], N: int,
                           cap: int = 5000) -> List[List[List[int]]]:
    """Backtracking enumerator. Returns up to `cap` valid completions."""
    grid = [row[:] for row in grid]
    # Track row/col used sets
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    empties = []
    for i in range(N):
        for j in range(N):
            v = grid[i][j]
            if v is not None:
                row_used[i].add(v)
                col_used[j].add(v)
            else:
                empties.append((i, j))
    out = []

    def bt(k):
        if len(out) >= cap:
            return
        if k == len(empties):
            out.append([row[:] for row in grid])
            return
        i, j = empties[k]
        for v in range(1, N + 1):
            if v in row_used[i] or v in col_used[j]:
                continue
            grid[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            bt(k + 1)
            row_used[i].remove(v)
            col_used[j].remove(v)
            grid[i][j] = None
    bt(0)
    return out


def _random_latin_square(N: int, rng: random.Random) -> Optional[List[List[int]]]:
    """Generate a random Latin square via row-by-row backtracking."""
    grid: List[List[Optional[int]]] = [[None] * N for _ in range(N)]
    col_used = [set() for _ in range(N)]

    def bt(i):
        if i == N:
            return True
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        # Try permutations of this row that satisfy column constraints
        attempts = []
        # Use a randomized column-by-column DFS
        row = [None] * N
        row_used = set()

        def fill_row(j):
            if j == N:
                return True
            shuffled = perm[:]
            for v in shuffled:
                if v in row_used or v in col_used[j]:
                    continue
                row[j] = v
                row_used.add(v)
                col_used[j].add(v)
                if fill_row(j + 1):
                    return True
                col_used[j].remove(v)
                row_used.remove(v)
                row[j] = None
            return False
        if not fill_row(0):
            return False
        for j in range(N):
            grid[i][j] = row[j]
        return bt(i + 1)
    if bt(0):
        return [[grid[i][j] for j in range(N)] for i in range(N)]
    return None


class LatinSquareCountQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "latin_square_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            N = 3
            n_remove = 1     # only x, answer must be 1
        elif level <= 2:
            N = 3
            n_remove = 2
        elif level <= 4:
            N = 4
            n_remove = 3
        elif level <= 6:
            N = 4
            n_remove = 5
        elif level <= 8:
            N = 5
            n_remove = 6
        else:
            N = 5
            n_remove = 9
        return {"level": level, "N": N, "n_remove": n_remove}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N = cfg["N"]
        n_remove = cfg["n_remove"]
        rng = random.Random((self.seed or 0) * 1657 + level * 41 + 11)

        # Try several random Latin squares + removal patterns until we find
        # one where the answer is well-defined (1..N) and the marked cell is
        # actually in the empties.
        for _attempt in range(40):
            base = _random_latin_square(N, rng)
            if base is None:
                continue
            # Random removal
            cells = [(i, j) for i in range(N) for j in range(N)]
            rng.shuffle(cells)
            removed = cells[:n_remove]
            grid: List[List[Optional[int]]] = [
                [None if (i, j) in removed else base[i][j]
                 for j in range(N)]
                for i in range(N)
            ]
            # Pick marked cell = one of the removed ones
            x_cell = rng.choice(removed)
            # Enumerate valid completions; collect distinct values at x_cell.
            completions = _all_valid_completions(grid, N, cap=5000)
            if not completions:
                continue
            distinct_x = {comp[x_cell[0]][x_cell[1]] for comp in completions}
            n_kinds = len(distinct_x)
            # Sanity guard: the answer should be 1..N
            if n_kinds < 1 or n_kinds > N:
                continue
            # At L0 specifically we want the trivial 1-answer
            if level == 0 and n_kinds != 1:
                continue
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(N=N)
            answer = str(n_kinds)
            img = self._render(grid, x_cell, N)
            return question, answer, img
        return None

    def _render(self, grid: List[List[Optional[int]]],
                x_cell: Tuple[int, int], N: int) -> Image.Image:
        bg = "#ffffff"
        cell_color = "#ffffff"
        unknown_color = "#f4f4f4"
        marked_color = "#fde2e4"
        edge_color = "#222222"

        cell_size = 1.0
        fig_w = max(4.0, N * cell_size + 1.0)
        fig_h = max(4.0, N * cell_size + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        for i in range(N):
            for j in range(N):
                x = j * cell_size
                y = (N - 1 - i) * cell_size  # row 0 at top
                v = grid[i][j]
                if (i, j) == x_cell:
                    fc = marked_color
                elif v is None:
                    fc = unknown_color
                else:
                    fc = cell_color
                ax.add_patch(plt.Rectangle((x, y), cell_size, cell_size,
                                           facecolor=fc, edgecolor=edge_color,
                                           linewidth=1.5))
                cx = x + cell_size / 2
                cy = y + cell_size / 2
                if (i, j) == x_cell:
                    ax.text(cx, cy, "x", ha="center", va="center",
                            fontsize=18, fontweight="bold", color="#b00020")
                elif v is not None:
                    ax.text(cx, cy, str(v), ha="center", va="center",
                            fontsize=16, fontweight="bold", color="#1a1a1a")
                # Empty (non-x) cells stay blank.

        ax.set_xlim(-0.2, N * cell_size + 0.2)
        ax.set_ylim(-0.2, N * cell_size + 0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{N}x{N} Latin square (each row & column = 1..{N})",
                     fontsize=11, pad=6)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg,
                    dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
