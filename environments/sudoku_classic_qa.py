"""
Classic NxN Sudoku puzzle (structured-puzzle style).

reference qids studied (sudoku, total 30): idx=1051, 1052, 1057, 1058, 1063, 1064,
1069, 1070, 1075, 1076 (verbatim per `design notes` §sudoku, lines
876-895). Sample TSV format confirmed via reference dataset:
  - question_text starts with "Your task is to solve the 9x9 Sudoku puzzle..."
  - answer is a Python list-of-lists, e.g. [[3,8,2,...], ...]
  - empty cells in input grid encoded as 0
  - subgrids are 3x3 (rules 1-3 per row/col/box)

Self-contained: this file inlines a small NxN Sudoku generator + backtracking
solver for verification. NO `from RLVE`, `from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 4x4 with ~3 empties (smallest grid, max givens)
  L9 within solvable range: 9x9 with ~50 empties

Renderer adapted from /scratch/ty45972/finetuning/RL_env/RLVE_Visual/environments/sudoku_visual.py
(only the matplotlib drawing logic — generator inlined here).
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# structured 4-section templates.
# Sections: ### Game Rules / ### Coordinate System / ### Current Puzzle State /
# ### Output Format. The image still visualizes the same state.
_TEMPLATES = [
    "Your task is to solve the {NM}x{NM} Sudoku puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each row contains every digit from 1 to {NM} exactly once.\n"
    "2. Each column contains every digit from 1 to {NM} exactly once.\n"
    "3. Each {N}x{M} subgrid contains every digit from 1 to {NM} exactly once.\n"
    "4. Given digits cannot be changed; only fill empty cells.\n\n"
    "### Coordinate System:\n"
    "- The puzzle is a {NM}x{NM} grid indexed by (row, column), 0-indexed.\n"
    "- Rows are listed top-to-bottom and columns left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "- The puzzle is given as a {NM}x{NM} list of lists (0 = empty cell):\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Provide the completed grid as a Python list of lists wrapped in <answer>...</answer>.\n"
    "Example: <answer>{example}</answer>",

    "Your task is to solve the {NM}x{NM} Sudoku puzzle described below:\n\n"
    "### Game Rules:\n"
    "- Fill the grid so that every row, every column, and every {N}x{M} subgrid contains digits 1..{NM} exactly once.\n"
    "- Preserve all given (non-zero) digits.\n\n"
    "### Coordinate System:\n"
    "- The grid is {NM}x{NM}, 0-indexed by (row, col).\n"
    "- Subgrid blocks have height {N} and width {M}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "(0 means an empty cell.)\n\n"
    "### Output Format:\n"
    "Output the solved grid as a Python list of lists inside <answer>...</answer>.",

    "Solve the {NM}x{NM} Sudoku puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Sudoku: each of the {NM} rows, {NM} columns, and {NM} subgrids of size {N}x{M} must contain digits 1..{NM} exactly once. Givens are fixed.\n\n"
    "### Coordinate System:\n"
    "- Rows numbered 0..{NM_minus_one} top-to-bottom.\n"
    "- Columns numbered 0..{NM_minus_one} left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "Initial board (0 = empty):\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Return the fully filled grid as a Python list-of-lists inside <answer>...</answer>.",
]


def _generate_solved_sudoku(N: int, M: int, rng: random.Random) -> List[List[int]]:
    """Generate a fully-solved NxM-subgrid Sudoku of size NM x NM.

    Uses the standard latin-shift + group-permutation technique (same recipe as
    the original RLVE Sudoku_Environment, copied inline so we don't import it).
    """
    NM = N * M
    base = [
        [(M * (row % N) + row // N + column) % NM + 1 for column in range(NM)]
        for row in range(NM)
    ]
    perm = list(range(1, NM + 1))
    rng.shuffle(perm)
    grid = [[perm[base[row][column] - 1] for column in range(NM)] for row in range(NM)]

    def shuffle_groups(data, group_size):
        G = len(data) // group_size
        for g in range(G):
            start = g * group_size
            slice_ = data[start:start + group_size]
            rng.shuffle(slice_)
            data[start:start + group_size] = slice_
        groups = [data[g * group_size:(g + 1) * group_size] for g in range(G)]
        rng.shuffle(groups)
        data[:] = [row for group in groups for row in group]

    shuffle_groups(grid, N)
    grid_t = list(map(list, zip(*grid)))
    shuffle_groups(grid_t, M)
    grid = list(map(list, zip(*grid_t)))
    return grid


def _check_sudoku_solution(predicted: List[List[int]], puzzle: List[List[int]],
                           N: int, M: int) -> bool:
    """Verify a Sudoku solution against the puzzle's clues + standard rules."""
    NM = N * M
    if len(predicted) != NM:
        return False
    for row in predicted:
        if len(row) != NM:
            return False
    # Cells must be 1..NM and preserve givens
    for i in range(NM):
        for j in range(NM):
            v = predicted[i][j]
            if not (1 <= v <= NM):
                return False
            if puzzle[i][j] != 0 and predicted[i][j] != puzzle[i][j]:
                return False
    expected = set(range(1, NM + 1))
    # Rows
    for i in range(NM):
        if set(predicted[i]) != expected:
            return False
    # Columns
    for j in range(NM):
        col = [predicted[i][j] for i in range(NM)]
        if set(col) != expected:
            return False
    # Subgrids: M x N grid of N x M boxes
    for bi in range(M):
        for bj in range(N):
            cells = [
                predicted[bi * N + r][bj * M + c]
                for r in range(N) for c in range(M)
            ]
            if set(cells) != expected:
                return False
    return True


class SudokuClassicQA(StandaloneVisualEnv):
    ENV_NAME = "sudoku_classic"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the solved grid directly inside "
        "`<answer>...</answer>` tags as a Python list of lists."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L1: 4x4 (N=2, M=2), few empties
        # L2..L3: 6x6 (N=2, M=3)
        # L4..L5: 6x6 (N=3, M=2) sparser
        # L6..L9: 9x9 (N=3, M=3) increasing sparsity
        # 2026-05-04 R3 retry: softened — v5=0.30 (worst). L5→L6 cliff
        # (6x6 → 9x9) was untouched; push 9x9 to L8+ only and extend 6x6
        # through L7. 9x9 sparsity capped at 0.30→0.40 (was 0.30→0.45).
        if level <= 1:
            N, M = 2, 2
            sparsity = 0.15 + 0.10 * level  # ~2-4 empties out of 16
        elif level <= 3:
            N, M = 2, 3
            sparsity = 0.20 + 0.07 * (level - 2)
        elif level <= 5:
            N, M = 3, 2
            sparsity = 0.25 + 0.06 * (level - 4)  # softer 6x6
        elif level <= 7:
            N, M = 3, 2
            sparsity = 0.32 + 0.05 * (level - 6)  # extend 6x6 to L7
        else:
            N, M = 3, 3
            sparsity = 0.30 + 0.05 * (level - 8)  # only L8/L9 are 9x9
        return {"N": N, "M": M, "sparsity": sparsity, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, M, sparsity = cfg["N"], cfg["M"], cfg["sparsity"]
        NM = N * M
        rng = random.Random((seed or 0) * 991 + level * 31 + 7)

        try:
            solved = _generate_solved_sudoku(N, M, rng)
        except Exception:
            return None

        # Carve out empties
        puzzle = [row[:] for row in solved]
        n_empty = max(1, int(NM * NM * sparsity))
        empty_cells = rng.sample(range(NM * NM), n_empty)
        for cell in empty_cells:
            r, c = divmod(cell, NM)
            puzzle[r][c] = 0

        # Save state for verification
        self._puzzle = puzzle
        self._N, self._M = N, M

        sidx = (seed or 0) % len(_TEMPLATES)
        # Build a tiny example matching grid size for the template
        example_row = list(range(1, NM + 1))
        example = "[" + ",".join(f"[{','.join(str(x) for x in example_row)}]"
                                 for _ in range(NM)) + "]"
        # For very large grids the example bloats; cap it.
        if NM > 4:
            example = "[[1,2,...],...]"
        # Format puzzle state as Python list-of-lists with row-per-line indent
        state_str = "[\n" + ",\n".join("  " + str(row) for row in puzzle) + "\n]"
        question = _TEMPLATES[sidx].format(
            N=N, M=M, NM=NM, NM_minus_one=NM - 1,
            example=example, state=state_str,
        )

        # Answer = the solved grid as a list of lists string
        ans_str = str(solved)
        img = self._render(puzzle, N, M)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, grid: List[List[int]], N: int, M: int) -> Image.Image:
        NM = N * M
        cell_size = max(30, 360 // NM)
        fig_size = (NM * cell_size / 80 + 1.5, NM * cell_size / 80 + 1.5)
        fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=100)

        for i in range(NM):
            for j in range(NM):
                val = grid[i][j]
                bg = "#f7f9fc" if val == 0 else "#e8edf2"
                rect = patches.Rectangle(
                    (j, NM - 1 - i), 1, 1,
                    linewidth=0.5, edgecolor="#bdc3c7", facecolor=bg
                )
                ax.add_patch(rect)
                if val != 0:
                    ax.text(j + 0.5, NM - 0.5 - i, str(val),
                            ha="center", va="center",
                            fontsize=max(7, 18 - NM),
                            fontweight="bold", color="#2c3e50")

        # Thick lines for subgrid borders.
        for i in range(0, NM + 1):
            lw = 2.5 if i % N == 0 else 0.5
            ax.plot([0, NM], [i, i], color="#2c3e50", linewidth=lw)
        for j in range(0, NM + 1):
            lw = 2.5 if j % M == 0 else 0.5
            ax.plot([j, j], [0, NM], color="#2c3e50", linewidth=lw)

        ax.set_xlim(-0.1, NM + 0.1)
        ax.set_ylim(-0.1, NM + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Parse predicted as a Python literal list-of-lists, OR as N lines of
        # space-separated digits (RLVE legacy form). Re-validate against the
        # puzzle (multiple valid solutions possible — gt is just one).
        import ast, re
        N, M = self._N, self._M
        NM = N * M
        pred_grid = None
        # Try Python literal first
        try:
            obj = ast.literal_eval(predicted.strip())
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                pred_grid = [[int(x) for x in row] for row in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        # Fallback: NM lines of NM space-separated ints
        if pred_grid is None:
            try:
                lines = [l.strip() for l in predicted.splitlines() if l.strip()]
                if len(lines) == NM:
                    g = [list(map(int, l.split())) for l in lines]
                    if all(len(r) == NM for r in g):
                        pred_grid = g
            except (ValueError, AttributeError):
                pass
        # Fallback: extract any NM*NM ints in order
        if pred_grid is None:
            try:
                nums = [int(x) for x in re.findall(r"-?\d+", predicted)]
                if len(nums) == NM * NM:
                    pred_grid = [nums[i * NM:(i + 1) * NM] for i in range(NM)]
            except (ValueError, TypeError):
                pass
        if pred_grid is None:
            return False
        return _check_sudoku_solution(pred_grid, self._puzzle, N, M)
