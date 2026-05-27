"""
Minesweeper puzzle (structured-puzzle style).

reference qids studied (minesweeper, total 30): idx=751, 752, 753, 754, 755, 756,
757, 758, 759, 760, 761, 762 (verbatim per reference dataset;
research file `design notes` §minesweeper only listed 4 samples (lines
645-657), so 8 additional were sourced directly from the TSV across D=1..D=2).
Format confirmed:
  - question_text: "Your task is to solve the Minesweeper puzzle...\\n
                    1. Numbers represent how many mines are adjacent (including diagonally)..."
  - answer: comma-separated `(row, col)` 0-indexed coordinates of mines, e.g.
    "(1, 2),(2, 1)" — listing the location of every mine.
  - The image shows revealed numbers (and unrevealed cells); the model must
    deduce mine positions.

Self-contained: inlines a simple mine generator + neighbor-count revealer
(adapted from /scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/minesweeping/environment.py).
NO `from RLVE`, `from Gym`, `sys.path.insert`.

NOTE on uniqueness: the original RLVE generator does NOT guarantee uniquely
solvable boards (the same set of revealed numbers may admit multiple mine
configurations). For verification, we therefore re-validate that the
predicted mine set is *consistent* with all revealed clues (every revealed
cell's number = count of predicted mines in its 8-neighborhood, and no
revealed cell is itself predicted as a mine). This is the same logic used
by the RLVE scorer for `satisfied/all`. Strict equality with the reference
mine set is too brittle for ambiguous boards.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 3x3 with 1 mine, almost all cells revealed
  L9 within solvable range: 6x6, several mines, sparse reveals
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


_NUMBER_COLORS = {
    0: "#b0b0b0",
    1: "#1565c0",
    2: "#2e7d32",
    3: "#c62828",
    4: "#6a1b9a",
    5: "#e65100",
    6: "#00838f",
    7: "#424242",
    8: "#78909c",
}


_TEMPLATES = [
    "Your task is to solve the {N}x{M} Minesweeper puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Some cells are revealed and contain a digit 0-8 = the count of mines in the 8 surrounding cells (including diagonals).\n"
    "2. Unrevealed cells (`?`) may or may not contain a mine.\n"
    "3. Your task is to identify every cell that contains a mine.\n\n"
    "### Coordinate System:\n"
    "- Cells are indexed (row, col), 0-indexed; row 0 at top, col 0 at left.\n"
    "- A revealed cell shows its mine-count digit; an unrevealed cell is `?`.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the mine coordinates as `(row, col)` (0-indexed) separated by commas, inside <answer>...</answer>.\n"
    "Example: <answer>(1, 2),(2, 1)</answer>",

    "Solve the Minesweeper puzzle below.\n\n"
    "### Game Rules:\n"
    "- Numbers on revealed cells count the mines among their 8 neighbors.\n"
    "- Identify every mine in the grid.\n\n"
    "### Coordinate System:\n"
    "- {N}x{M} grid; positions are (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the comma-separated list of mine coordinates inside <answer>...</answer>.",

    "Your task is to find every mine in the Minesweeper puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard Minesweeper rules: each revealed digit equals the count of adjacent mines (8-neighborhood).\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output mine `(r, c)` coordinates separated by commas inside <answer>...</answer>.",
]


def _ms_state_text(display, N, M):
    rows = []
    for r in range(N):
        row = []
        for c in range(M):
            v = display[r][c]
            row.append("?" if v == -1 else str(v))
        rows.append("|" + "|".join(row) + "|")
    return "\n".join(rows)


def _generate_minesweeper(N: int, M: int, mine_density: float, reveal_density: float,
                          rng: random.Random,
                          retries: int = 10) -> Optional[Tuple[List[List[object]],
                                                                List[Tuple[int, int]]]]:
    """Place mines randomly, then reveal a random subset of non-mine cells with
    their neighbor-mine counts. Returns (display_grid, mine_positions).
      display_grid[i][j] = -1 (unrevealed/possibly mine) or int 0-8 (revealed
      count). NB: revealed cells are guaranteed safe (no mine).
    """
    for _ in range(retries):
        # Place mines
        n_total = N * M
        n_mines = max(1, min(int(n_total * mine_density), n_total - 1))
        mine_cells = rng.sample(range(n_total), n_mines)
        mines = set()
        for cell in mine_cells:
            r, c = divmod(cell, M)
            mines.add((r, c))

        # Compute reveal count per non-mine cell
        non_mines = [(i, j) for i in range(N) for j in range(M)
                     if (i, j) not in mines]
        if not non_mines:
            continue
        n_reveal = max(1, int(len(non_mines) * reveal_density))
        revealed = rng.sample(non_mines, min(n_reveal, len(non_mines)))

        display = [[-1] * M for _ in range(N)]
        for (i, j) in revealed:
            count = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < N and 0 <= nj < M and (ni, nj) in mines:
                        count += 1
            display[i][j] = count
        return display, sorted(mines)
    return None


class MinesweeperSolveQA(StandaloneVisualEnv):
    ENV_NAME = "minesweeper_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the mine coordinate list directly "
        "inside `<answer>...</answer>` tags."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L1: 3x3 — 1 mine, ≥75% revealed
        # L2..L4: 4x4
        # L5..L7: 5x5
        # L8..L9: 6x6
        if level <= 1:
            N = M = 3
            mine_density = 0.20
            reveal = 0.85 - 0.05 * level
        elif level <= 4:
            N = M = 4
            mine_density = 0.20 + 0.03 * (level - 2)
            reveal = 0.75 - 0.05 * (level - 2)
        elif level <= 7:
            N = M = 5
            mine_density = 0.25 + 0.02 * (level - 5)
            reveal = 0.70 - 0.05 * (level - 5)
        else:
            N = M = 6
            mine_density = 0.28 + 0.03 * (level - 8)
            reveal = 0.65 - 0.05 * (level - 8)
        return {"N": N, "M": M, "mine_density": mine_density,
                "reveal_density": reveal, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, M = cfg["N"], cfg["M"]
        rng = random.Random((seed or 0) * 991 + level * 31 + 31)

        result = _generate_minesweeper(N, M, cfg["mine_density"],
                                       cfg["reveal_density"], rng)
        if result is None:
            return None
        display, mines = result

        self._display = display
        self._mines = set(mines)
        self._N, self._M = N, M

        sidx = (seed or 0) % len(_TEMPLATES)
        state_text = _ms_state_text(display, N, M)
        question = _TEMPLATES[sidx].format(N=N, M=M, state=state_text)
        # structured-puzzle format: "(1, 2),(2, 1)" — comma-separated tuples
        ans_str = ",".join(f"({r}, {c})" for r, c in mines)
        img = self._render(display, N, M)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, grid, N, M) -> Image.Image:
        cell_size = max(40, 400 // max(N, M))
        fig_w = M * cell_size / 80 + 1.2
        fig_h = N * cell_size / 80 + 1.2
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)

        font_size = max(10, 22 - max(N, M) * 2)

        for i in range(N):
            for j in range(M):
                cell = grid[i][j]
                bg = "#b0bec5" if cell == -1 else "#ffffff"
                ax.add_patch(patches.Rectangle(
                    (j, N - 1 - i), 1, 1,
                    linewidth=1.0, edgecolor="#546e7a", facecolor=bg,
                ))
                if cell != -1:
                    color = _NUMBER_COLORS.get(cell, "#000000")
                    ax.text(j + 0.5, N - 0.5 - i, str(cell),
                            ha="center", va="center",
                            fontsize=font_size, fontweight="bold",
                            color=color)

        label_fs = max(8, font_size - 2)
        for i in range(N):
            ax.text(-0.15, N - 0.5 - i, str(i),
                    ha="right", va="center", fontsize=label_fs, color="#37474f")
        for j in range(M):
            ax.text(j + 0.5, N + 0.15, str(j),
                    ha="center", va="bottom", fontsize=label_fs, color="#37474f")

        ax.add_patch(patches.Rectangle(
            (0, 0), M, N, linewidth=2.5,
            edgecolor="#263238", facecolor="none",
        ))

        ax.set_xlim(-0.5, M + 0.2)
        ax.set_ylim(-0.2, N + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import re
        N, M = self._N, self._M
        # Parse coordinates (r,c) from predicted text
        coords = set()
        for r, c in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", predicted):
            coords.add((int(r), int(c)))

        # All mines must be within bounds
        for r, c in coords:
            if not (0 <= r < N and 0 <= c < M):
                return False

        # No mine on a revealed cell
        for (r, c) in coords:
            if self._display[r][c] != -1:
                return False

        # Every revealed number must equal the count of predicted mines among
        # its 8 neighbors
        for i in range(N):
            for j in range(M):
                cell = self._display[i][j]
                if cell == -1:
                    continue
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if (ni, nj) in coords:
                            count += 1
                if count != cell:
                    return False
        return True
