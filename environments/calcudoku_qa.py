"""
Calcudoku-style arithmetic Latin-square puzzle.

Rules:
  - NxN grid divided into "cages" (groups of orthogonally connected cells).
  - Each cage has an operator (+, -, *, /) and a target number.
  - Fill each cell with a digit from 1..N such that:
      (a) every row is a permutation of 1..N,
      (b) every column is a permutation of 1..N,
      (c) every cage's cells produce the cage target via the cage operator.
        - For + / *, apply the operation across all cells in the cage.
        - For - / /, the cage has exactly 2 cells and we use |a-b| / max(a,b)//min(a,b).

The task: output a 2D list giving the digit in each cell, e.g.
    [[1,2,3],[2,3,1],[3,1,2]]

Difficulty axes:
  - grid size N (3 -> 5)
  - cage density / hint count
"""
import json
import random
import re
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the {n}x{n} Calcudoku puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each row of the {n}x{n} grid must contain digits 1..{n} exactly once.\n"
    "2. Each column must contain digits 1..{n} exactly once.\n"
    "3. The grid is divided into cages. Each cage has an operator (+, -, *, /) and a target value.\n"
    "4. The cells in a cage must combine via the operator to produce the target. For - and / cages (always 2 cells), |a-b| or max(a,b)/min(a,b) is used.\n\n"
    "### Coordinate System:\n"
    "- Cells are addressed (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- Cages (each entry is `target<op>: [(r,c), ...]`):\n"
    "{cages_text}\n\n"
    "### Output Format:\n"
    "Provide the completed grid as a Python 2D list inside <answer>...</answer>.\n"
    "Example: <answer>[[1,2,3],[2,3,1],[3,1,2]]</answer>",

    "Solve the {n}x{n} Calcudoku puzzle below.\n\n"
    "### Game Rules:\n"
    "- Rows and columns are permutations of 1..{n}.\n"
    "- Each cage has an operator and a target; the cage's cells combine via the operator to the target.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Cages: {cages_text}\n\n"
    "### Output Format:\n"
    "Output the filled grid as a Python list of lists inside <answer>...</answer>.",

    "Your task is to solve the {n}x{n} Calcudoku puzzle described below.\n\n"
    "### Game Rules:\n"
    "Latin square rules + cage arithmetic. Each cage's cells combine via its operator to give its target.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "Cages: {cages_text}\n\n"
    "### Output Format:\n"
    "Output a Python 2D list inside <answer>...</answer>.",
]


def _cages_to_text(cages_with_clues):
    parts = []
    for cells, op, tgt in cages_with_clues:
        cell_str = ",".join(f"({r},{c})" for r, c in cells)
        parts.append(f"{tgt}{op}: [{cell_str}]")
    return "; ".join(parts)


def _is_latin(grid: List[List[int]], n: int) -> bool:
    for i in range(n):
        if sorted(grid[i]) != list(range(1, n + 1)):
            return False
        if sorted(grid[r][i] for r in range(n)) != list(range(1, n + 1)):
            return False
    return True


def _cage_satisfied(cells: List[Tuple[int, int]], op: str, target: int,
                    grid: List[List[int]]) -> bool:
    vals = [grid[r][c] for r, c in cells]
    if op == "+":
        return sum(vals) == target
    if op == "*":
        prod = 1
        for v in vals:
            prod *= v
        return prod == target
    if op == "-":
        if len(vals) != 2:
            return False
        return abs(vals[0] - vals[1]) == target
    if op == "/":
        if len(vals) != 2:
            return False
        a, b = vals
        if a == 0 or b == 0:
            return False
        hi, lo = max(a, b), min(a, b)
        return hi % lo == 0 and hi // lo == target
    return False


def _random_latin(n: int, rng: random.Random) -> List[List[int]]:
    """Random NxN Latin square via cyclic-shift rows + row/col permutations."""
    base = [[((i + j) % n) + 1 for j in range(n)] for i in range(n)]
    rows = list(range(n))
    cols = list(range(n))
    rng.shuffle(rows)
    rng.shuffle(cols)
    g = [[base[rows[i]][cols[j]] for j in range(n)] for i in range(n)]
    # Optional digit relabel
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    return [[perm[v - 1] for v in row] for row in g]


def _build_cages(n: int, rng: random.Random,
                 max_cage: int = 3) -> List[List[Tuple[int, int]]]:
    """Partition the grid into orthogonally connected cages of size 1..max_cage."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    assigned = {}
    cages = []
    for cell in cells:
        if cell in assigned:
            continue
        size = rng.choices([1, 2, 3], weights=[2, 4, 2], k=1)[0]
        size = min(size, max_cage)
        cage = [cell]
        assigned[cell] = len(cages)
        # Grow by random BFS
        frontier = [cell]
        while len(cage) < size and frontier:
            cur = rng.choice(frontier)
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in assigned:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.remove(cur)
                continue
            nxt = rng.choice(neighbors)
            assigned[nxt] = len(cages)
            cage.append(nxt)
            frontier.append(nxt)
        cages.append(cage)
    return cages


def _label_cage(cells: List[Tuple[int, int]], grid: List[List[int]],
                rng: random.Random) -> Tuple[str, int]:
    """Assign an operator + target to a cage of cells given the underlying solution."""
    vals = [grid[r][c] for r, c in cells]
    if len(cells) == 1:
        return ("+", vals[0])
    if len(cells) == 2:
        a, b = vals
        choices = ["+", "*", "-"]
        hi, lo = max(a, b), min(a, b)
        if lo > 0 and hi % lo == 0:
            choices.append("/")
        op = rng.choice(choices)
        if op == "+":
            return ("+", a + b)
        if op == "*":
            return ("*", a * b)
        if op == "-":
            return ("-", abs(a - b))
        if op == "/":
            return ("/", hi // lo)
    # 3+ cells: just + or *
    op = rng.choice(["+", "*"])
    if op == "+":
        return ("+", sum(vals))
    prod = 1
    for v in vals:
        prod *= v
    return ("*", prod)


def _count_solutions(n: int, cages_with_clues, limit: int = 2) -> int:
    """Backtracking solution counter (stops at limit)."""
    grid = [[0] * n for _ in range(n)]
    cell_to_cage = {}
    for ci, (cells, op, tgt) in enumerate(cages_with_clues):
        for cell in cells:
            cell_to_cage[cell] = ci
    found = [0]

    def cage_partial_ok(ci):
        cells, op, tgt = cages_with_clues[ci]
        vals = [grid[r][c] for r, c in cells]
        unfilled = [v for v in vals if v == 0]
        filled = [v for v in vals if v != 0]
        if op == "+":
            cur = sum(filled)
            if not unfilled:
                return cur == tgt
            # Each unfilled in [1,n], so range is [cur+len(unfilled), cur+n*len(unfilled)]
            lo, hi = cur + len(unfilled) * 1, cur + len(unfilled) * n
            return lo <= tgt <= hi
        if op == "*":
            cur = 1
            for v in filled:
                cur *= v
            if not unfilled:
                return cur == tgt
            return tgt % cur == 0 if cur > 0 else False
        if op == "-":
            if not unfilled:
                return abs(filled[0] - filled[1]) == tgt
            return True  # too cheap to constrain partially
        if op == "/":
            if not unfilled:
                a, b = filled
                hi_, lo_ = max(a, b), min(a, b)
                return lo_ > 0 and hi_ % lo_ == 0 and hi_ // lo_ == tgt
            return True
        return False

    def backtrack(idx):
        if found[0] >= limit:
            return
        if idx == n * n:
            found[0] += 1
            return
        r, c = divmod(idx, n)
        # Cells already in the same row/col
        used_row = set(grid[r][cc] for cc in range(n) if grid[r][cc] != 0)
        used_col = set(grid[rr][c] for rr in range(n) if grid[rr][c] != 0)
        for v in range(1, n + 1):
            if v in used_row or v in used_col:
                continue
            grid[r][c] = v
            ci = cell_to_cage[(r, c)]
            if cage_partial_ok(ci):
                backtrack(idx + 1)
            grid[r][c] = 0
            if found[0] >= limit:
                return

    backtrack(0)
    return found[0]


class CalcudokuQA(StandaloneVisualEnv):
    ENV_NAME = "calcudoku"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the filled 2D grid directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 0:
            n = 3
            max_cage = 2
        elif level <= 3:
            n = 3 if level <= 1 else 4
            max_cage = 2
        elif level <= 6:
            n = 4 if level == 4 else 5
            max_cage = 3
        else:
            n = 5
            max_cage = 3
        return {"level": level, "n": n, "max_cage": max_cage}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        max_cage = cfg["max_cage"]
        rng = random.Random((self.seed or 0) * 9301 + level * 19 + 31)

        # Up to 30 attempts to land on a unique-solution puzzle
        for _ in range(30):
            grid = _random_latin(n, rng)
            cages = _build_cages(n, rng, max_cage=max_cage)
            cages_with_clues = []
            for cells in cages:
                op, tgt = _label_cage(cells, grid, rng)
                cages_with_clues.append((cells, op, tgt))
            # For L0 we want maximal givens — turn ALL cages into single-cell
            # cages (op='+', target=value). This is just "show me the grid".
            if level == 0:
                # Re-do as singletons
                cages_with_clues = []
                for r in range(n):
                    for c in range(n):
                        cages_with_clues.append(([(r, c)], "+", grid[r][c]))
            n_sols = _count_solutions(n, cages_with_clues, limit=2)
            if n_sols == 1:
                self._n = n
                self._cages = cages_with_clues
                self._solution = grid
                ans_str = json.dumps(grid)
                sidx = (self.seed or 0) % len(_TEMPLATES)
                cages_text = _cages_to_text(cages_with_clues)
                question = _TEMPLATES[sidx].format(n=n, cages_text=cages_text)
                img = self._render(n, cages_with_clues)
                return question, ans_str, img
        return None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, n: int, cages_with_clues) -> Image.Image:
        cell_size = 1.0
        fig, ax = plt.subplots(figsize=(0.9 * n + 1.2, 0.9 * n + 1.2), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Distinct soft pastel for each cage
        palette = [
            "#FFE5E5", "#E5F3FF", "#E5FFE5", "#FFF8E5", "#F0E5FF", "#E5FFFA",
            "#FFE5F8", "#FFEFD5", "#E0F7FA", "#F1F8E9", "#FBE9E7", "#EDE7F6",
            "#FFF3E0", "#E8F5E9", "#FCE4EC", "#E3F2FD",
        ]

        for ci, (cells, op, tgt) in enumerate(cages_with_clues):
            color = palette[ci % len(palette)]
            for r, c in cells:
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                           facecolor=color, edgecolor="#888",
                                           linewidth=0.6))
        # Cage borders: thicker line between cells of different cages
        cell_to_cage = {}
        for ci, (cells, _op, _tgt) in enumerate(cages_with_clues):
            for cell in cells:
                cell_to_cage[cell] = ci
        for r in range(n):
            for c in range(n):
                # Top
                if r == 0 or cell_to_cage.get((r - 1, c)) != cell_to_cage[(r, c)]:
                    ax.plot([c, c + 1], [n - r, n - r], color="black", linewidth=2.4)
                # Bottom
                if r == n - 1 or cell_to_cage.get((r + 1, c)) != cell_to_cage[(r, c)]:
                    ax.plot([c, c + 1], [n - 1 - r, n - 1 - r], color="black", linewidth=2.4)
                # Left
                if c == 0 or cell_to_cage.get((r, c - 1)) != cell_to_cage[(r, c)]:
                    ax.plot([c, c], [n - 1 - r, n - r], color="black", linewidth=2.4)
                # Right
                if c == n - 1 or cell_to_cage.get((r, c + 1)) != cell_to_cage[(r, c)]:
                    ax.plot([c + 1, c + 1], [n - 1 - r, n - r], color="black", linewidth=2.4)
        # Place clue label in top-left corner of cage's lexicographically smallest cell
        for cells, op, tgt in cages_with_clues:
            cells_sorted = sorted(cells)
            r, c = cells_sorted[0]
            label = f"{tgt}{op}" if len(cells) > 1 else f"{tgt}"
            ax.text(c + 0.05, n - 0.1 - r, label, ha="left", va="top",
                    fontsize=11, fontweight="bold", color="#222")
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
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
        # Parse predicted as 2D list. Tolerate Python-list / JSON / lines-of-digits.
        n = self._n
        grid = self._parse_grid(predicted, n)
        if grid is None:
            return False
        # Latin?
        if not _is_latin(grid, n):
            return False
        # Cages?
        for cells, op, tgt in self._cages:
            if not _cage_satisfied(cells, op, tgt, grid):
                return False
        return True

    def _parse_grid(self, text: str, n: int) -> Optional[List[List[int]]]:
        s = text.strip()
        # Try JSON / Python literal
        try:
            obj = json.loads(s)
            if (isinstance(obj, list) and len(obj) == n
                    and all(isinstance(r, list) and len(r) == n for r in obj)
                    and all(isinstance(v, int) for r in obj for v in r)):
                return obj
        except Exception:
            pass
        # Try parsing as nested-list-with-Python-syntax
        try:
            import ast
            obj = ast.literal_eval(s)
            if (isinstance(obj, list) and len(obj) == n
                    and all(isinstance(r, (list, tuple)) and len(r) == n for r in obj)
                    and all(isinstance(v, int) for r in obj for v in r)):
                return [list(r) for r in obj]
        except Exception:
            pass
        # Try parsing line-by-line: each line is digits separated by spaces or commas
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if len(lines) == n:
            grid = []
            for ln in lines:
                # Strip brackets / quotes
                ln_clean = ln.strip("[]() ,")
                tokens = re.findall(r"\d+", ln_clean)
                if len(tokens) != n:
                    return None
                try:
                    grid.append([int(t) for t in tokens])
                except ValueError:
                    return None
            return grid
        # Try "all digits in a row" — only safe for small N where digits are 1..N (single digit)
        if n <= 9:
            digits = re.findall(r"\d", s)
            if len(digits) == n * n:
                vals = [int(d) for d in digits]
                return [vals[i * n:(i + 1) * n] for i in range(n)]
        return None
