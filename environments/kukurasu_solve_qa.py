"""
Kukurasu puzzle — structured-puzzle style.

Rules:
  - N x N (or N x M) binary grid (0/1).
  - For each row i (1-indexed), the sum of the COLUMN INDICES (1-indexed)
    of the cells where the value is 1 equals row_sum[i].
  - For each column j (1-indexed), the sum of the ROW INDICES (1-indexed)
    of the cells where the value is 1 equals col_sum[j].

Studied reference qids (design notes lines 240-251):
  - idx=601 (D=1): 3x3, row_sums=[6,3,5], col_sums=[3,6,4] ->
    `[[1,1,1],[1,1,0],[0,1,1]]`
  - idx=620 (D=3): 5x5
  - idx=630 (D=5): 6x6

Answer format: 2D list of 0/1 like `[[1,1,1],[1,1,0],[0,1,1]]`.

Difficulty axis: N (3 -> 7).
"""
import random
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

from .standalone_base import StandaloneVisualEnv


# Match a puzzle benchmark Kukurasu prompt verbatim (idx=601, idx=602):
# - Single-paragraph game rules (no 4-section headers).
# - Row clues listed as `Row 1: x\nRow 2: y\n...`.
# - Column clues listed as `Column 1: a\nColumn 2: b\n...`.
# - Answer format example uses 3x3 grid `[[1, 1, 0], [1, 0, 1], [0, 0, 1]]`.
_TEMPLATES = [
    "This is a {n}x{n} Kukurasu puzzle. You need to fill the grid with black cells according to the following rules:\n"
    "1. The sum of column positions (1 to {n}) of black cells in each row must equal the number on the right.\n"
    "2. The sum of row positions (1 to {n}) of black cells in each column must equal the number at the bottom.\n\n"
    "{row_block}\n\n"
    "Column constraints (sum of row positions of black cells in each column):\n"
    "{col_block}\n\n"
    "Please solve the puzzle and provide the solution as a two-dimensional array, using 0 for white cells and 1 for blackcells.\n"
    "Example answer format: [[1, 1, 0], [1, 0, 1], [0, 0, 1]].",
]


def _is_kukurasu_valid(grid: List[List[int]], n: int,
                       row_sums: List[int], col_sums: List[int]) -> bool:
    if len(grid) != n:
        return False
    if any(len(row) != n for row in grid):
        return False
    for row in grid:
        if any(v not in (0, 1) for v in row):
            return False
    # Row sums (column-index sum of 1s, 1-indexed)
    for i in range(n):
        s = sum((j + 1) for j in range(n) if grid[i][j] == 1)
        if s != row_sums[i]:
            return False
    for j in range(n):
        s = sum((i + 1) for i in range(n) if grid[i][j] == 1)
        if s != col_sums[j]:
            return False
    return True


def _solve_kukurasu(n: int, row_sums: List[int], col_sums: List[int],
                    limit: int = 2,
                    max_steps: int = 200000) -> List[List[List[int]]]:
    """Backtracking solver. Returns up to `limit` solutions."""
    grid = [[0] * n for _ in range(n)]
    found = []
    steps = [0]

    def backtrack(idx):
        if len(found) >= limit:
            return
        steps[0] += 1
        if steps[0] > max_steps:
            return
        if idx == n * n:
            if _is_kukurasu_valid(grid, n, row_sums, col_sums):
                found.append([row[:] for row in grid])
            return
        r, c = divmod(idx, n)
        # Pruning per row when we end the row
        for v in [0, 1]:
            grid[r][c] = v
            ok = True
            # End-of-row check
            if c == n - 1:
                s = sum((j + 1) for j in range(n) if grid[r][j] == 1)
                if s != row_sums[r]:
                    ok = False
            # End-of-grid check is implicit
            # Column partial check (only when we have a full column = at last row r=n-1 for col c)
            if ok and r == n - 1:
                s = sum((i + 1) for i in range(n) if grid[i][c] == 1)
                if s != col_sums[c]:
                    ok = False
            if ok:
                backtrack(idx + 1)
            grid[r][c] = 0
            if len(found) >= limit:
                return

    backtrack(0)
    return found


def _gen_kukurasu(n: int, rng: random.Random,
                  max_attempts: int = 30) -> Optional[Tuple[List[List[int]],
                                                             List[int],
                                                             List[int]]]:
    """Sample a random binary grid; compute clues; verify uniqueness."""
    for _ in range(max_attempts):
        # Random density 0.3-0.7
        density = rng.uniform(0.3, 0.7)
        grid = [[1 if rng.random() < density else 0 for _ in range(n)]
                for _ in range(n)]
        # Avoid degenerate (all 0 or all 1)
        total = sum(sum(row) for row in grid)
        if total == 0 or total == n * n:
            continue
        row_sums = [sum((j + 1) for j in range(n) if grid[i][j] == 1)
                    for i in range(n)]
        col_sums = [sum((i + 1) for i in range(n) if grid[i][j] == 1)
                    for j in range(n)]
        sols = _solve_kukurasu(n, row_sums, col_sums, limit=2,
                               max_steps=80000)
        if len(sols) == 1:
            return grid, row_sums, col_sums
    # Fall back: even non-unique (still valid)
    return grid, row_sums, col_sums if 'row_sums' in locals() else None


class KukurasuSolveQA(StandaloneVisualEnv):
    ENV_NAME = "kukurasu_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the completed grid directly as a 2D list of 0/1 "
        "inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n = 3
        elif level <= 2:
            n = 4
        elif level <= 4:
            n = 5
        elif level <= 6:
            n = 6
        else:
            n = 7
        return {"level": level, "n": n}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 6151 + level * 59 + 17)

        result = _gen_kukurasu(n, rng)
        if result is None:
            return None
        grid, row_sums, col_sums = result

        self._n = n
        self._row_sums = row_sums
        self._col_sums = col_sums

        gt = "[" + ",".join(
            "[" + ",".join(str(v) for v in row) + "]" for row in grid
        ) + "]"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        # Build per-row / per-column clue blocks matching benchmark format.
        row_block = "\n".join(f"Row {i + 1}: {row_sums[i]}" for i in range(n))
        col_block = "\n".join(f"Column {j + 1}: {col_sums[j]}" for j in range(n))
        question = _TEMPLATES[sidx].format(
            n=n,
            row_block=row_block,
            col_block=col_block,
        )
        img = self._render(n, row_sums, col_sums)
        return question, gt, img

    def _render(self, n, row_sums, col_sums) -> Image.Image:
        cell_in = 0.55
        fig_w = (n + 2.0) * cell_in + 0.4
        fig_h = (n + 2.0) * cell_in + 0.4
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Empty grid cells with 1-indexed coordinate labels (column # weights)
        for r in range(n):
            for c in range(n):
                ax.add_patch(patches.Rectangle((c, n - 1 - r), 1, 1,
                                                linewidth=1.0,
                                                edgecolor="#445",
                                                facecolor="#f7f9fc"))
        ax.add_patch(patches.Rectangle((0, 0), n, n, linewidth=2.5,
                                        edgecolor="#222", facecolor="none"))
        # Top: column index labels (small, gray) above grid
        for c in range(n):
            ax.text(c + 0.5, n + 0.25, f"{c + 1}",
                    ha="center", va="center", fontsize=9, color="#555")
        # Left: row index labels
        for r in range(n):
            ax.text(-0.3, n - 1 - r + 0.5, f"{r + 1}",
                    ha="center", va="center", fontsize=9, color="#555")
        # Right: row sums (clues)
        for r in range(n):
            ax.text(n + 0.45, n - 1 - r + 0.5, str(row_sums[r]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#c0392b")
        # Bottom: col sums (clues)
        for c in range(n):
            ax.text(c + 0.5, -0.45, str(col_sums[c]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#c0392b")
        # Annotate clue meaning
        ax.text(n + 0.45, n + 0.25, "Σ", ha="center", va="center",
                fontsize=10, color="#7f8c8d")
        ax.text(-0.3, -0.45, "Σ", ha="center", va="center",
                fontsize=10, color="#7f8c8d")
        ax.set_xlim(-0.7, n + 0.95)
        ax.set_ylim(-0.85, n + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        n = self._n
        s = predicted.strip()
        s = re.sub(r"```[^\n]*\n", "", s)
        s = s.replace("```", "").strip()
        # Try Python-style 2D list parsing first via regex
        # Find each `[a,b,c,...]` row
        rows_raw = re.findall(r"\[([^\[\]]+)\]", s)
        if len(rows_raw) >= n:
            grid = []
            for row_raw in rows_raw[:n]:
                tokens = re.findall(r"[01]", row_raw)
                if len(tokens) != n:
                    grid = None
                    break
                grid.append([int(t) for t in tokens])
            if grid is not None and len(grid) == n:
                return _is_kukurasu_valid(grid, n, self._row_sums,
                                          self._col_sums)
        # Fallback: try newline-separated rows of digits
        rows = [r for r in s.splitlines() if r.strip()]
        if len(rows) == n:
            grid = []
            for row in rows:
                tokens = re.findall(r"[01]", row)
                if len(tokens) != n:
                    return False
                grid.append([int(t) for t in tokens])
            return _is_kukurasu_valid(grid, n, self._row_sums, self._col_sums)
        return False
