"""
Numbrix puzzle (structured-puzzle style).

reference qids studied (numbrix, total 30): idx=841, 842, 847, 848, 853, 854,
859, 860, 865, 866 (verbatim per `design notes` §numbrix, lines
706-751). Format confirmed via reference dataset:
  - question_text: "Your task is to solve the Numbrix puzzle...\\n
                    1. Numbrix is played on a square grid...\\n
                    2. ...starting from 1 up to the maximum number..."
  - answer: pipe-delimited grid like "\\n|5|6|7|8|\\n|4|11|10|9|\\n..."
  - Numbers are 1..NM (NOT 0-indexed)
  - empty cells in input shown as ` ` (single space) between pipes

Self-contained: inlines a random Hamiltonian-path generator on an NxM grid
(adapted from /scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/numbrix/environment.py
but RE-RANGED to 1..NM (reference) instead of 0..NM-1). NO `from RLVE`,
`from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 3x3 with most cells given (Hamiltonian path is short)
  L9: 6x6 sparse
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


# Match MM-HELIX numbrix prompt verbatim (idx=841, idx=842):
# - 6-item Game Rules (NOT 4) including the "not every empty cell needs to be
#   filled" rule.
# - Important Notes section.
# - Current Numbrix State framing line + post-grid "In this representation:"
#   bullet list.
# - Output Format Requirements with the canonical 5x5 example grid.
_TEMPLATES = [
    "\nYour task is to solve the Numbrix puzzle based on the following rules and the current state:\n\n"
    "### Game Rules:\n\n"
    "1. Numbrix is played on a square grid, where some cells are already filled with numbers.\n"
    "2. You must fill in the empty cells with numbers to create a continuous path starting from 1 up to the **maximum number in the sequence**, which is **not necessarily equal to the total number of cells (n²)**.\n"
    "3. The numbers must be adjacent either horizontally or vertically (not diagonally).\n"
    "4. Each number can only be used once.\n"
    "5. The path must form a single continuous sequence where consecutive numbers are adjacent.\n"
    "6. **Not every empty cell needs to be filled.** Depending on the puzzle configuration, some cells may remain empty.\n\n"
    "### Important Notes:\n"
    "* The highest number in the puzzle might be equal or less than the total number of grid cells (e.g., $n^2 - 1$, or even smaller).\n"
    "* It is your job to determine what the highest number is, based on the filled numbers and the constraints of the puzzle.\n\n"
    "### Current Numbrix State:\n"
    "The current state of the Numbrix puzzle is shown below:\n\n"
    "{state}\n\n"
    "In this representation:\n\n"
    "* Filled cells contain the given numbers.\n"
    "* Empty cells are blank spaces.\n"
    "* Your goal is to fill in the empty cells to complete a valid number sequence from 1 to the correct maximum number, following the rules above.\n\n"
    "### Output Format Requirements:\n\n"
    "1. The final answer should be the completed grid with all numbers from 1 to the correct highest number, aligned clearly in rows and columns.\n\n"
    "#### Example answer format for a 5x5 grid:\n\n"
    "|11|10|9|2|3|\n"
    "|12|13|8|1|4|\n"
    "|15|14|7|6|5|\n"
    "|16|19|20|23|24|\n"
    "|17|18|21|22|25|\n",
]


def _puzzle_to_pipes(puzzle, N, M):
    rows = []
    for r in range(N):
        cells = [str(v) if v != 0 else " " for v in puzzle[r]]
        rows.append("|" + "|".join(cells) + "|")
    return "\n".join(rows)


def _gen_hamiltonian_path(N: int, M: int, rng: random.Random,
                          retries: int = 6) -> Optional[List[List[int]]]:
    """Generate an NxM matrix where each cell holds a number in 1..N*M and
    the numbers form a Hamiltonian path (orthogonal neighbors are consecutive).

    Adapted from /scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/numbrix/environment.py.
    Reranged to 1..NM (puzzle convention).
    """
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def is_inside(x, y):
        return 0 <= x < N and 0 <= y < M

    for _ in range(retries):
        sx = rng.randint(0, N - 1)
        sy = rng.randint(0, M - 1)
        visited = [[False] * M for _ in range(N)]
        order = [[-1] * M for _ in range(N)]
        visited[sx][sy] = True
        order[sx][sy] = 0  # 0-indexed step (we'll shift to 1-based at the end)

        def count_unvisited_degree(x, y):
            cnt = 0
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cnt += 1
            return cnt

        def check_connectivity(remain):
            start = None
            for i in range(N):
                for j in range(M):
                    if not visited[i][j]:
                        start = (i, j)
                        break
                if start:
                    break
            if not start:
                return True
            stack = [start]
            seen = {start}
            count = 1
            while stack:
                x, y = stack.pop()
                for dx, dy in dirs:
                    xx, yy = x + dx, y + dy
                    if is_inside(xx, yy) and not visited[xx][yy] \
                            and (xx, yy) not in seen:
                        seen.add((xx, yy))
                        stack.append((xx, yy))
                        count += 1
            return count == remain

        def dfs(step, x, y):
            if step == N * M:
                return True
            cand = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if is_inside(nx, ny) and not visited[nx][ny]:
                    cand.append((nx, ny))
            if not cand:
                return False
            rng.shuffle(cand)
            cand_scores = [(count_unvisited_degree(nx, ny), nx, ny)
                           for nx, ny in cand]
            cand_scores.sort(key=lambda t: t[0])
            for _, nx, ny in cand_scores:
                visited[nx][ny] = True
                order[nx][ny] = step
                remain = N * M - (step + 1)
                if check_connectivity(remain):
                    if dfs(step + 1, nx, ny):
                        return True
                visited[nx][ny] = False
                order[nx][ny] = -1
            return False

        if dfs(1, sx, sy):
            # Shift to 1..NM
            return [[order[i][j] + 1 for j in range(M)] for i in range(N)]
    return None


class NumbrixSolveQA(StandaloneVisualEnv):
    ENV_NAME = "numbrix_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the completed grid directly inside "
        "`<answer>...</answer>` tags as pipe-delimited rows."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L1: 3x3, sparsity 0.20 (most cells given)
        # L2..L4: 4x4
        # L5..L7: 5x5
        # L8..L9: 6x6
        if level <= 1:
            N = M = 3
            sparsity = 0.20 + 0.10 * level
        elif level <= 4:
            N = M = 4
            sparsity = 0.30 + 0.10 * (level - 2)
        elif level <= 7:
            N = M = 5
            sparsity = 0.40 + 0.05 * (level - 5)
        else:
            N = M = 6
            sparsity = 0.45 + 0.05 * (level - 8)
        return {"N": N, "M": M, "sparsity": sparsity, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, M, sparsity = cfg["N"], cfg["M"], cfg["sparsity"]
        rng = random.Random((seed or 0) * 991 + level * 31 + 29)

        solved = _gen_hamiltonian_path(N, M, rng)
        if solved is None:
            return None

        # Carve empties — store as 0 (we'll display blank)
        puzzle = [row[:] for row in solved]
        n_total = N * M
        n_empty = max(1, int(n_total * sparsity))
        empties = rng.sample(range(n_total), n_empty)
        for cell in empties:
            r, c = divmod(cell, M)
            puzzle[r][c] = 0

        self._puzzle = puzzle
        self._N, self._M = N, M

        sidx = (seed or 0) % len(_TEMPLATES)
        state_text = _puzzle_to_pipes(puzzle, N, M)
        question = _TEMPLATES[sidx].format(state=state_text)
        # Format answer like reference: leading newline, then |n|n|...| rows
        ans_str = "\n" + "\n".join("|" + "|".join(str(v) for v in row) + "|"
                                   for row in solved)
        img = self._render(puzzle, N, M)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, matrix, N, M) -> Image.Image:
        cell_size = max(40, 400 // max(N, M))
        fig_w = M * cell_size / 80 + 1.2
        fig_h = N * cell_size / 80 + 1.2
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)

        font_size = max(9, 20 - max(N, M) * 2)

        for i in range(N):
            for j in range(M):
                v = matrix[i][j]
                if v == 0:
                    bg = "#e8eaf6"
                    text = ""
                    color = "#000000"
                else:
                    bg = "#ffffff"
                    text = str(v)
                    color = "#1a237e"
                ax.add_patch(patches.Rectangle(
                    (j, N - 1 - i), 1, 1,
                    linewidth=1.0, edgecolor="#546e7a", facecolor=bg,
                ))
                if text:
                    ax.text(j + 0.5, N - 0.5 - i, text,
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
        # Robust parsing: pull all integers per non-empty line. Accept formats:
        # "|1|2|3|", "1 2 3", "[[1,2,3], ...]" (we'll treat any line with N
        # ints as a row; need exactly N rows of M ints).
        text = predicted.strip()
        # First try: split into lines, ignoring lines without digits
        lines = [l.strip() for l in text.splitlines() if l.strip() and re.search(r"\d", l)]
        rows = []
        for line in lines:
            # Strip pipes/brackets/commas
            line = re.sub(r"[\[\]\|,]", " ", line)
            ints = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(ints) == M:
                rows.append(ints)
        # Fallback: pull ALL ints, reshape if exactly N*M
        if len(rows) != N:
            try:
                all_ints = [int(x) for x in re.findall(r"-?\d+", text)]
                if len(all_ints) == N * M:
                    rows = [all_ints[i * M:(i + 1) * M] for i in range(N)]
            except (ValueError, TypeError):
                pass
        if len(rows) != N:
            return False
        if any(len(r) != M for r in rows):
            return False

        # Each value in 1..NM, all distinct, givens preserved
        loc = [None] * (N * M + 1)  # 1-indexed location array
        for i in range(N):
            for j in range(M):
                v = rows[i][j]
                g = self._puzzle[i][j]
                if g != 0 and g != v:
                    return False
                if not (1 <= v <= N * M):
                    return False
                if loc[v] is not None:
                    return False
                loc[v] = (i, j)
        # Path: every consecutive pair must be adjacent
        for v in range(1, N * M):
            x1, y1 = loc[v]
            x2, y2 = loc[v + 1]
            if abs(x1 - x2) + abs(y1 - y2) != 1:
                return False
        return True
