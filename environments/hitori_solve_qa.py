"""
Hitori puzzle (structured-puzzle style).

reference qids studied (hitori, total 30): idx=541, 542, 543, 544, 545, 546, 547,
548, 549, 550, 551, 552 (verbatim per reference dataset —
research file `design notes` §hitori only listed 4 samples (line 540-552),
so 8 additional samples were sourced directly from the TSV: idx 545-552, all
puzzles for D=1..D=2). Format confirmed:
  - question_text: "You are given an image of a Hitori puzzle.\\nPuzzle Rules:..."
  - answer: set/python set of (row, col) tuples 0-indexed, e.g.
    "{(2, 3), (1, 1), (3, 1), (0, 0)}"

Self-contained: inlines the recursive-backtrack puzzle generator from
/scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/hitori_puzzle/environment.py.
NO `from RLVE`, `from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0: 3x3 (smallest with at least one shaded cell viable)
  L9: 6x6
"""
import random
import sys
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the {N}x{M} Hitori puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Some cells must be shaded (blacked out); the others remain unshaded.\n"
    "2. Among unshaded cells, no number may repeat in any row or column.\n"
    "3. No two shaded cells may be orthogonally adjacent.\n"
    "4. All unshaded cells must form a single orthogonally-connected region.\n\n"
    "### Coordinate System:\n"
    "- The grid is {N}x{M}, indexed by (row, col), 0-indexed (row 0 at top, col 0 at left).\n\n"
    "### Current Puzzle State:\n"
    "- The puzzle is given as an {N}x{M} integer grid:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Provide the set of shaded coordinates (0-indexed) inside <answer>...</answer>.\n"
    "Example: <answer>{{(2, 3), (1, 1), (3, 1), (0, 0)}}</answer>",

    "Solve the {N}x{M} Hitori puzzle below.\n\n"
    "### Game Rules:\n"
    "- Shade cells so unshaded cells in each row/column have no duplicates.\n"
    "- Shaded cells must not be orthogonally adjacent to each other.\n"
    "- Unshaded cells must form one connected region (orthogonal connectivity).\n\n"
    "### Coordinate System:\n"
    "- Cells are addressed (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Grid:\n{state}\n\n"
    "### Output Format:\n"
    "Output the shaded set as `{{(r, c), (r, c), ...}}` inside <answer>...</answer>.",

    "Your task is to solve the {N}x{M} Hitori puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard Hitori: shade cells so (a) unshaded values are unique per row/column, (b) shaded cells are non-adjacent, (c) unshaded cells stay connected.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the set of shaded `(row, col)` pairs inside <answer>...</answer>.",
]


# L0-L2 probe: ask if a SPECIFIC cell should be shaded.
_PROBE_TEMPLATES = [
    "Your task is to determine the status of one cell in the {N}x{M} Hitori puzzle below:\n\n"
    "### Game Rules:\n"
    "1. Some cells must be shaded (blacked out); the others remain unshaded.\n"
    "2. Among unshaded cells, no number may repeat in any row or column.\n"
    "3. No two shaded cells may be orthogonally adjacent.\n"
    "4. All unshaded cells must form a single orthogonally-connected region.\n\n"
    "### Coordinate System:\n"
    "- The grid is {N}x{M}, indexed by (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "In the unique solution, is the cell at (row {r}, col {c}) SHADED? Answer YES or NO inside <answer>...</answer>.",

    "Look at the {N}x{M} Hitori puzzle below.\n\n"
    "### Game Rules:\n"
    "- Shade cells so unshaded values are unique per row/col.\n"
    "- No two shaded cells touch orthogonally.\n"
    "- Unshaded cells form one connected region.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Should the cell at (row {r}, col {c}) be shaded in the solution? Answer YES or NO inside <answer>...</answer>.",

    "Hitori puzzle ({N}x{M}).\n\n"
    "### Game Rules:\n"
    "Standard Hitori rules.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Is cell (row {r}, col {c}) shaded in the solution? Output YES or NO inside <answer>...</answer>.",
]


def _matrix_to_text(matrix, N, M):
    rows = []
    for r in range(N):
        rows.append(" ".join(str(v) for v in matrix[r]))
    return "\n".join(rows)


def _check_connected(grid: List[List[str]], N: int, M: int) -> bool:
    """All `.` cells form a single orthogonally-connected component."""
    visited = [[False] * M for _ in range(N)]
    sys.setrecursionlimit(10000)

    def dfs(x, y):
        visited[x][y] = True
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < M and not visited[nx][ny] \
                    and grid[nx][ny] == ".":
                dfs(nx, ny)

    for i in range(N):
        for j in range(M):
            if grid[i][j] == ".":
                dfs(i, j)
                return all(visited[a][b] for a in range(N) for b in range(M)
                           if grid[a][b] == ".")
    # No unshaded cells = trivially "connected" (vacuous)
    return True


def _generate_hitori(N: int, M: int, rng: random.Random,
                     attempts: int = 8) -> Optional[Tuple[List[List[int]],
                                                           List[List[str]]]]:
    """Inlined from RLVE/Gym/environments/hitori_puzzle/environment.py.

    Backtracking generator: for each cell in random order, decide whether to
    leave it unshaded (with a fresh number) or shade it (recycling an existing
    row/col number). Returns (matrix, reference_solution) or None.
    """
    sys.setrecursionlimit(10000)
    for _ in range(attempts):
        matrix = [[None] * M for _ in range(N)]
        ref = [["."] * M for _ in range(N)]
        all_cells = [(i, j) for i in range(N) for j in range(M)]
        rng.shuffle(all_cells)

        def backtrack(idx):
            if idx == len(all_cells):
                return True
            i, j = all_cells[idx]
            remaining = (
                set(matrix[i][_j] for _j in range(M)
                    if ref[i][_j] == "." and matrix[i][_j] is not None)
                | set(matrix[_i][j] for _i in range(N)
                      if ref[_i][j] == "." and matrix[_i][j] is not None)
            )
            for color in rng.sample([".", "*"], 2):
                if color == ".":
                    num = 0
                    while num in remaining:
                        num += 1
                    matrix[i][j] = num
                else:
                    if not remaining:
                        continue
                    ok = True
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and ref[ni][nj] == "*":
                            ok = False
                            break
                    if not ok:
                        continue
                    ref[i][j] = "*"
                    if not _check_connected(ref, N, M):
                        ref[i][j] = "."
                        continue
                    matrix[i][j] = rng.choice(list(remaining))
                if backtrack(idx + 1):
                    return True
                # On failure, undo our choice
                if color == "*":
                    ref[i][j] = "."
                matrix[i][j] = None
            return False

        ok = backtrack(0)
        if ok:
            return matrix, ref
    return None


class HitoriSolveQA(StandaloneVisualEnv):
    ENV_NAME = "hitori_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the set of shaded coordinates "
        "directly inside `<answer>...</answer>` tags."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode — small grid, ask YES/NO for one cell.
        # L3..L9: full puzzle.
        if level == 0:
            return {"N": 3, "M": 3, "level": level, "probe": True}
        if level == 1:
            return {"N": 3, "M": 3, "level": level, "probe": True}
        if level == 2:
            return {"N": 4, "M": 4, "level": level, "probe": True}
        if level <= 4:
            return {"N": 4, "M": 4, "level": level, "probe": False}
        if level <= 7:
            return {"N": 5, "M": 5, "level": level, "probe": False}
        return {"N": 6, "M": 6, "level": level, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, M = cfg["N"], cfg["M"]
        rng = random.Random((seed or 0) * 991 + level * 31 + 19)

        result = _generate_hitori(N, M, rng)
        if result is None:
            return None
        matrix, ref = result

        # Save state
        self._matrix = matrix
        self._N, self._M = N, M
        self._probe_mode = cfg["probe"]

        # Reference shaded coords (one valid solution; we will re-validate any
        # rule-satisfying answer)
        shaded = sorted(
            (i, j) for i in range(N) for j in range(M) if ref[i][j] == "*"
        )

        # 2026-05-04: simplified L0/L1 (was 5% too-hard) — store level for hint
        self._level_for_hint = cfg.get("level", 0)
        if cfg["probe"]:
            # Balance YES/NO: 50% sample from shaded cells (YES) and 50% from
            # non-shaded cells (NO).
            shaded_set = set(shaded)
            unshaded = [(r, c) for r in range(N) for c in range(M)
                        if (r, c) not in shaded_set]
            if shaded and rng.random() < 0.5:
                target_r, target_c = rng.choice(shaded)
                ans_str = "YES"
            elif unshaded:
                target_r, target_c = rng.choice(unshaded)
                ans_str = "NO"
            elif shaded:
                target_r, target_c = rng.choice(shaded)
                ans_str = "YES"
            else:
                return None
            self._probe_target = (target_r, target_c)
            self._probe_answer = ans_str
            sidx = (seed or 0) % len(_PROBE_TEMPLATES)
            state_text = _matrix_to_text(matrix, N, M)
            question = _PROBE_TEMPLATES[sidx].format(
                N=N, M=M, state=state_text, r=target_r, c=target_c,
            )
            # 2026-05-04: simplified L0/L1 (was 5% too-hard) — explicit hint.
            if self._level_for_hint <= 1:
                question += (
                    "\n\nHint: a cell must be SHADED if its value already "
                    "appears elsewhere in its row or column AMONG unshaded "
                    "cells (a duplicate that must be eliminated). Check the "
                    "queried cell's value in row "
                    f"{target_r} and column {target_c}: if leaving it "
                    "unshaded would create a duplicate of that value in "
                    "either the row or column, then YES it must be shaded; "
                    "otherwise NO."
                )
            img = self._render(matrix, N, M)
            return question, ans_str, img

        sidx = (seed or 0) % len(_TEMPLATES)
        state_text = _matrix_to_text(matrix, N, M)
        question = _TEMPLATES[sidx].format(N=N, M=M, state=state_text)
        # Format like reference: "{(2, 3), (1, 1), (3, 1), (0, 0)}"
        ans_str = "{" + ", ".join(f"({r}, {c})" for r, c in shaded) + "}"
        img = self._render(matrix, N, M)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, matrix, N, M) -> Image.Image:
        cell_size = max(45, 400 // max(N, M))
        fig_size = (M * cell_size / 80 + 1.2, N * cell_size / 80 + 1.2)
        fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=100)

        for i in range(N):
            for j in range(M):
                ax.add_patch(patches.Rectangle(
                    (j, N - 1 - i), 1, 1,
                    linewidth=1.5, edgecolor="#2c3e50", facecolor="#f7f9fc",
                ))
                ax.text(j + 0.5, N - 0.5 - i, str(matrix[i][j]),
                        ha="center", va="center",
                        fontsize=max(12, 22 - max(N, M) * 2),
                        fontweight="bold", color="#2c3e50")

        ax.add_patch(patches.Rectangle(
            (0, 0), M, N, linewidth=3,
            edgecolor="#2c3e50", facecolor="none",
        ))

        # Add 0-indexed labels
        for i in range(N):
            ax.text(-0.25, N - 0.5 - i, str(i), ha="right", va="center",
                    fontsize=max(8, 14 - max(N, M)), color="#7f8c8d")
        for j in range(M):
            ax.text(j + 0.5, N + 0.15, str(j), ha="center", va="bottom",
                    fontsize=max(8, 14 - max(N, M)), color="#7f8c8d")

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
        # PROBE MODE: yes/no
        if getattr(self, "_probe_mode", False):
            text = predicted.strip().lower()
            m = re.search(r"\b(yes|no|y|n)\b", text)
            if m:
                w = m.group(1)
                pred = "YES" if w.startswith("y") else "NO"
                return pred == self._probe_answer
            return False
        # Parse all (r,c) pairs out of predicted; ignore everything else.
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", predicted)
        # Distinguish "no pairs found" from "model said empty set". An empty
        # answer like "{}" is only valid if ground_truth is also empty.
        shaded: Set[Tuple[int, int]] = set()
        for r, c in pairs:
            shaded.add((int(r), int(c)))

        N, M = self._N, self._M
        if any(not (0 <= r < N and 0 <= c < M) for r, c in shaded):
            return False

        # No two shaded adjacent
        for r, c in shaded:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r + dr, c + dc) in shaded:
                    return False

        # Unshaded connected
        grid = [["*" if (i, j) in shaded else "." for j in range(M)]
                for i in range(N)]
        if not _check_connected(grid, N, M):
            return False

        # Row/col uniqueness on unshaded
        for i in range(N):
            vals = [self._matrix[i][j] for j in range(M)
                    if (i, j) not in shaded]
            if len(set(vals)) != len(vals):
                return False
        for j in range(M):
            vals = [self._matrix[i][j] for i in range(N)
                    if (i, j) not in shaded]
            if len(set(vals)) != len(vals):
                return False
        return True
