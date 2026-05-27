"""
Snake-style path puzzle on a grid.

Rules:
  - NxN grid with a marked START cell (S) and END cell (E).
  - The grid carries row counts and column counts indicating how many cells
    in that row / column are part of the snake.
  - Find a snake = a sequence of orthogonally adjacent cells starting at S
    and ending at E, with no repeated cells, satisfying:
      * row k contains exactly the labelled count of snake cells.
      * col k contains exactly the labelled count of snake cells.
      * The snake may turn at right angles (L-bends are allowed). However,
        the snake cannot fold back on itself: any two snake cells that are
        more than 2 path-steps apart must NOT be diagonally adjacent.
        (Cells exactly 2 path-steps apart, formed by a single L-bend, may
         be diagonally adjacent — that's how a normal turn looks.)

The task: output the snake as a sequence of `(r,c)` coordinates from S to E.

Difficulty axes:
  - grid size N (3 → 6)
  - snake length / clue density
"""
import random
import re
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the {n}x{n} snake puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Place a snake (a sequence of orthogonally adjacent cells, no repeats) starting at the marked S cell and ending at the marked E cell.\n"
    "2. The number of snake cells in each row must equal the row count; same for each column.\n"
    "3. The snake may make L-bend turns, but cells more than 2 path-steps apart must not be diagonally adjacent (the snake does not fold back to touch itself).\n\n"
    "### Coordinate System:\n"
    "- Cells are addressed (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- Start S = {S}\n"
    "- End E = {E}\n"
    "- Row counts: {row_counts}\n"
    "- Column counts: {col_counts}\n\n"
    "### Output Format:\n"
    "Output the snake path as space-separated `(r,c)` pairs from S to E inside <answer>...</answer>.\n"
    "Example: <answer>(4,4) (4,3) (4,2) (3,2) (2,2)</answer>",

    "Solve the snake puzzle below.\n\n"
    "### Game Rules:\n"
    "- Snake from S to E, orthogonal steps, no repeats.\n"
    "- Row/col counts must match clues.\n"
    "- No fold-back diagonal touches.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n} grid, 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "S = {S}, E = {E}\n"
    "Row counts: {row_counts}\n"
    "Col counts: {col_counts}\n\n"
    "### Output Format:\n"
    "Output the snake `(r,c)` coordinate sequence inside <answer>...</answer>.",

    "Your task is to solve the snake puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard snake-path puzzle.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "S = {S}, E = {E}\n"
    "Row counts: {row_counts}; Col counts: {col_counts}\n\n"
    "### Output Format:\n"
    "Output the snake path inside <answer>...</answer>.",
]


def _path_valid(path: List[Tuple[int, int]], n: int,
                row_counts: List[int], col_counts: List[int],
                S: Tuple[int, int], E: Tuple[int, int]) -> bool:
    if not path:
        return False
    if path[0] != S or path[-1] != E:
        return False
    if len(set(path)) != len(path):
        return False
    # All in bounds
    for r, c in path:
        if not (0 <= r < n and 0 <= c < n):
            return False
    # Adjacency
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    # Row & col counts
    rc = [0] * n
    cc = [0] * n
    for r, c in path:
        rc[r] += 1
        cc[c] += 1
    if rc != row_counts or cc != col_counts:
        return False
    # No diagonal self-touch for cells far apart in path. We allow L-bends
    # (cells 2 apart in path can be diagonal — that's a normal path turn),
    # but forbid the snake from looping back and brushing itself.
    cell_to_idx = {cell: i for i, cell in enumerate(path)}
    for i, (r, c) in enumerate(path):
        for dr in (-1, 1):
            for dc in (-1, 1):
                nb = (r + dr, c + dc)
                if nb in cell_to_idx:
                    j = cell_to_idx[nb]
                    if abs(i - j) > 2:
                        return False
    return True


def _gen_snake(n: int, length: int, rng: random.Random,
               max_attempts: int = 400) -> Optional[List[Tuple[int, int]]]:
    """Random walk attempt to build a snake of given length on NxN grid
    that doesn't self-touch diagonally (between non-consecutive cells)."""
    for attempt in range(max_attempts):
        start = (rng.randint(0, n - 1), rng.randint(0, n - 1))
        path = [start]
        used = {start}
        success = True
        for step in range(length - 1):
            r, c = path[-1]
            options = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if (nr, nc) in used:
                    continue
                # Check diagonal-touching to non-consecutive path cells.
                # New cell (nr, nc) must not be diagonal to any cell in path
                # except path[-1] (which is (r, c) — orthogonal, not diagonal).
                # Also it must not be orthogonal to a non-consecutive path
                # cell (otherwise the path "bends back" and effectively
                # touches — but actually orthogonal to non-consecutive cells
                # would create a duplicate-touching issue too; the puzzle
                # spec only forbids diagonal touching though).
                bad = False
                # Check diagonal touch with all path cells except the last
                # two (last is current orthogonal neighbor; second-to-last
                # may be diagonal during L-bend, which is allowed).
                for idx, (pr, pc) in enumerate(path[:-2]):
                    if abs(pr - nr) == 1 and abs(pc - nc) == 1:
                        bad = True
                        break
                if bad:
                    continue
                options.append((nr, nc))
            if not options:
                success = False
                break
            nxt = rng.choice(options)
            path.append(nxt)
            used.add(nxt)
        if success and len(path) == length:
            return path
    return None


def _solve_snake(n: int, row_counts: List[int], col_counts: List[int],
                 S: Tuple[int, int], E: Tuple[int, int],
                 limit: int = 1) -> Optional[List[Tuple[int, int]]]:
    """Backtracking solver — find one valid snake (or up to `limit` of them).
    Returns the FIRST valid path found (or None)."""
    target_len = sum(row_counts)
    if target_len != sum(col_counts):
        return None
    found = []

    def backtrack(path, used, rc, cc):
        if len(found) >= limit:
            return
        if len(path) == target_len:
            if path[-1] == E and rc == row_counts and cc == col_counts:
                found.append(path[:])
            return
        r, c = path[-1]
        # last must be E
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr, nc) in used:
                continue
            if rc[nr] + 1 > row_counts[nr]:
                continue
            if cc[nc] + 1 > col_counts[nc]:
                continue
            # Diagonal-non-touch check (allow L-bends; only forbid
            # diagonal with cells >= 2 steps before current).
            bad = False
            for idx, (pr, pc) in enumerate(path[:-2]):
                if abs(pr - nr) == 1 and abs(pc - nc) == 1:
                    bad = True
                    break
            if bad:
                continue
            # If this would be the last cell, must equal E
            if len(path) + 1 == target_len and (nr, nc) != E:
                continue
            path.append((nr, nc))
            used.add((nr, nc))
            rc[nr] += 1
            cc[nc] += 1
            backtrack(path, used, rc, cc)
            path.pop()
            used.discard((nr, nc))
            rc[nr] -= 1
            cc[nc] -= 1
            if len(found) >= limit:
                return

    rc = [0] * n
    cc = [0] * n
    rc[S[0]] += 1
    cc[S[1]] += 1
    backtrack([S], {S}, rc, cc)
    if found:
        return found[0]
    return None


class SnakePuzzleQA(StandaloneVisualEnv):
    ENV_NAME = "snake_puzzle"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the snake coord sequence directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n = 3
            length_frac = 0.4   # snake covers ~40% of cells
        elif level <= 3:
            n = 4
            length_frac = 0.45
        elif level <= 6:
            n = 5
            length_frac = 0.5
        else:
            n = 6
            length_frac = 0.55
        return {"level": level, "n": n, "length_frac": length_frac}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 5527 + level * 41 + 11)

        # L0 — hardcoded straight 3-cell snakes for triviality.
        if level == 0:
            patterns = [
                # (S, E, path)
                ((0, 0), (0, 2), [(0, 0), (0, 1), (0, 2)]),
                ((1, 0), (1, 2), [(1, 0), (1, 1), (1, 2)]),
                ((2, 0), (2, 2), [(2, 0), (2, 1), (2, 2)]),
                ((0, 0), (2, 0), [(0, 0), (1, 0), (2, 0)]),
                ((0, 1), (2, 1), [(0, 1), (1, 1), (2, 1)]),
            ]
            S, E, path = patterns[(self.seed or 0) % len(patterns)]
            row_counts = [0] * n
            col_counts = [0] * n
            for r, c in path:
                row_counts[r] += 1
                col_counts[c] += 1
            self._n = n
            self._row_counts = row_counts
            self._col_counts = col_counts
            self._S = S
            self._E = E
            self._solution = path
            ans_str = " ".join(f"({r},{c})" for r, c in path)
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(
                n=n, S=str(S), E=str(E),
                row_counts=row_counts, col_counts=col_counts,
            )
            img = self._render(n, row_counts, col_counts, S, E)
            return question, ans_str, img

        target_length = max(3, int(n * n * cfg["length_frac"]))
        for attempt in range(60):
            path = _gen_snake(n, target_length, rng, max_attempts=80)
            if path is None:
                continue
            S = path[0]
            E = path[-1]
            row_counts = [0] * n
            col_counts = [0] * n
            for r, c in path:
                row_counts[r] += 1
                col_counts[c] += 1
            self._n = n
            self._row_counts = row_counts
            self._col_counts = col_counts
            self._S = S
            self._E = E
            self._solution = path
            ans_str = " ".join(f"({r},{c})" for r, c in path)
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(
                n=n, S=str(S), E=str(E),
                row_counts=row_counts, col_counts=col_counts,
            )
            img = self._render(n, row_counts, col_counts, S, E)
            return question, ans_str, img
        return None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, n, row_counts, col_counts, S, E) -> Image.Image:
        # Cells inside [0..n) x [0..n); row counts on right side, col counts on bottom
        fig, ax = plt.subplots(figsize=(0.85 * (n + 2), 0.85 * (n + 2)), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for r in range(n):
            for c in range(n):
                fc = "#fafafa"
                if (r, c) == S:
                    fc = "#ffd6a5"
                elif (r, c) == E:
                    fc = "#a8dadc"
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                           facecolor=fc, edgecolor="#777",
                                           linewidth=0.8))
                if (r, c) == S:
                    ax.text(c + 0.5, n - 0.5 - r, "S", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="#222")
                elif (r, c) == E:
                    ax.text(c + 0.5, n - 0.5 - r, "E", ha="center", va="center",
                            fontsize=14, fontweight="bold", color="#222")
                else:
                    # tiny coord marker
                    ax.text(c + 0.06, n - 0.06 - r, f"{r},{c}",
                            ha="left", va="top", fontsize=6, color="#bbb")
        # Column counts (bottom)
        for c in range(n):
            ax.text(c + 0.5, -0.25, str(col_counts[c]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1d3557")
        # Row counts (right)
        for r in range(n):
            ax.text(n + 0.25, n - 0.5 - r, str(row_counts[r]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1d3557")
        ax.set_xlim(-0.05, n + 0.65)
        ax.set_ylim(-0.55, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ------------------------------------------------------------------ #
    # Answer checking
    # ------------------------------------------------------------------ #

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Parse predicted as a sequence of (r,c) coords
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", predicted)
        if not pairs:
            return False
        try:
            path = [(int(r), int(c)) for r, c in pairs]
        except ValueError:
            return False
        return _path_valid(
            path, self._n, self._row_counts, self._col_counts,
            self._S, self._E,
        )
