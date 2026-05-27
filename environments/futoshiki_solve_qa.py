"""
Futoshiki puzzle (structured-puzzle style).

reference qids studied (futoshiki, total 30): idx=391, 392, 393, 394, 395, 396,
397, 398, 399, 400, 401, 402 (verbatim per reference dataset;
research file `design notes` §futoshiki only listed 4 samples (lines
451-466), so 8 additional samples were sourced directly from the TSV across
D=1..D=2). Format confirmed:
  - question_text: "You are given an image of a Futoshiki puzzle...\\n
                    1. The puzzle is a NxN grid (e.g., 5x5).\\n
                    2. Fill each cell with a number from 1 to N..."
  - answer: list-of-lists with 1..N values, e.g. "[[4, 2, 1, 3], [3, 4, 2, 1], ...]"
  - 0 in the input grid = empty cell

Self-contained: inlines a Latin-square + inequality-pair generator (adapted from
/scratch/ty45972/finetuning/RL_env/RLVE/Gym/environments/futoshiki_puzzle/environment.py
but RE-RANGED to 1..N (puzzle convention) instead of 0..N-1). NO `from RLVE`,
`from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 3x3, 1 inequality, 80% pre-filled
  L9 within solvable range: 6x6, several inequalities, sparse givens
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


_TEMPLATES = [
    "Your task is to solve the {N}x{N} Futoshiki puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each row contains digits 1..{N} exactly once.\n"
    "2. Each column contains digits 1..{N} exactly once.\n"
    "3. Inequality constraints between cell pairs (`<` or `>`) must be satisfied.\n"
    "4. Pre-filled (given) cells cannot be changed.\n\n"
    "### Coordinate System:\n"
    "- Cells are 0-indexed (row, col).\n"
    "- Empty cells are shown as 0.\n\n"
    "### Current Puzzle State:\n"
    "- Grid:\n{state}\n"
    "- Inequalities (each entry is `(r1,c1) <op> (r2,c2)`): {ineqs}\n\n"
    "### Output Format:\n"
    "Output the solved grid as a Python list of lists inside <answer>...</answer>.\n"
    "Example: <answer>{example}</answer>",

    "Solve the {N}x{N} Futoshiki puzzle below.\n\n"
    "### Game Rules:\n"
    "- Fill the grid so each row/column has 1..{N} exactly once.\n"
    "- Honor every inequality constraint between cell pairs.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col); empty cells are 0.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Inequalities: {ineqs}\n\n"
    "### Output Format:\n"
    "Output a Python list of lists inside <answer>...</answer>.",

    "Your task is to solve the {N}x{N} Futoshiki puzzle described below.\n\n"
    "### Game Rules:\n"
    "Latin square 1..{N} per row/column + inequality constraints.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Inequalities: {ineqs}\n\n"
    "### Output Format:\n"
    "Output the filled grid inside <answer>...</answer>.",
]


def _futoshiki_state_text(puzzle):
    return "[\n" + ",\n".join("  " + str(row) for row in puzzle) + "\n]"


def _futoshiki_ineqs_text(ineqs):
    if not ineqs:
        return "(none)"
    return "; ".join(f"({a[0]},{a[1]}) {sym} ({b[0]},{b[1]})"
                     for (a, b, sym) in ineqs)


def _gen_latin_square(N: int, rng: random.Random) -> List[List[int]]:
    """Random Latin square via row & column shuffling.

    Values are 1..N (puzzle convention), NOT 0..N-1.
    """
    # Cyclic Latin square base (1-indexed)
    base = [[((i + j) % N) + 1 for j in range(N)] for i in range(N)]
    # Shuffle rows
    rng.shuffle(base)
    # Shuffle columns
    cols = list(range(N))
    rng.shuffle(cols)
    grid = [[base[i][cols[j]] for j in range(N)] for i in range(N)]
    # Permute the value-set (1..N → random permutation)
    perm = list(range(1, N + 1))
    rng.shuffle(perm)
    grid = [[perm[grid[i][j] - 1] for j in range(N)] for i in range(N)]
    return grid


class FutoshikiSolveQA(StandaloneVisualEnv):
    ENV_NAME = "futoshiki_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the completed grid directly inside "
        "`<answer>...</answer>` tags as a Python list of lists."
    )

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R3: softened — 6x6 with sparsity 0.5+ (>= 18 empty cells)
        # at L8/L9 was unsolvable for the model (L9 dropped to 0.10).
        # Stay at 5x5 for L8/L9 and use higher sparsity / more inequalities
        # to keep some difficulty gradient.
        level = max(0, min(level, 9))
        if level <= 1:
            N = 3
            sparsity = 0.20 + 0.10 * level
            n_ineq = 1 + level
        elif level <= 4:
            N = 4
            sparsity = 0.30 + 0.10 * (level - 2)
            n_ineq = 2 + (level - 2)
        elif level <= 7:
            N = 5
            sparsity = 0.40 + 0.05 * (level - 5)
            n_ineq = 3 + (level - 5)
        else:
            N = 5
            sparsity = 0.55 + 0.05 * (level - 8)
            n_ineq = 5 + (level - 8)
        return {"N": N, "sparsity": sparsity, "n_ineq": n_ineq, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, sparsity, n_ineq = cfg["N"], cfg["sparsity"], cfg["n_ineq"]
        rng = random.Random((seed or 0) * 991 + level * 31 + 23)

        try:
            solved = _gen_latin_square(N, rng)
        except Exception:
            return None

        # Generate inequality constraints: pick orthogonally adjacent cell
        # pairs (image visualization is clearest for adjacent pairs) with
        # different values; emit `<` if first < second, else `>`.
        adjacent_pairs = []
        for i in range(N):
            for j in range(N):
                if j + 1 < N:
                    adjacent_pairs.append(((i, j), (i, j + 1)))
                if i + 1 < N:
                    adjacent_pairs.append(((i, j), (i + 1, j)))
        rng.shuffle(adjacent_pairs)
        chosen_ineqs: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
        for (a, b) in adjacent_pairs:
            if len(chosen_ineqs) >= n_ineq:
                break
            va = solved[a[0]][a[1]]
            vb = solved[b[0]][b[1]]
            if va == vb:
                continue
            sym = "<" if va < vb else ">"
            chosen_ineqs.append((a, b, sym))

        # Build puzzle = solved grid with some cells set to 0 (empty)
        puzzle = [row[:] for row in solved]
        n_total = N * N
        n_empty = max(1, int(n_total * sparsity))
        empties = rng.sample(range(n_total), n_empty)
        for cell in empties:
            r, c = divmod(cell, N)
            puzzle[r][c] = 0

        self._puzzle = puzzle
        self._N = N
        self._inequalities = chosen_ineqs

        sidx = (seed or 0) % len(_TEMPLATES)
        state_text = _futoshiki_state_text(puzzle)
        ineqs_text = _futoshiki_ineqs_text(chosen_ineqs)
        # Tiny example for the template (3x3 placeholder if N>3)
        if N <= 3:
            example = "[[" + ",".join(str(x) for x in range(1, N + 1)) + "],...]"
        else:
            example = "[[1,2,3,...],...]"
        question = _TEMPLATES[sidx].format(
            N=N, example=example, state=state_text, ineqs=ineqs_text,
        )
        ans_str = str(solved)
        img = self._render(puzzle, N, chosen_ineqs)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, puzzle, N, ineqs) -> Image.Image:
        spacing = 1.4
        cell_size = max(40, 400 // N)
        fig_w = N * spacing * cell_size / 80 + 1.5
        fig_h = N * spacing * cell_size / 80 + 1.5
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)

        for i in range(N):
            for j in range(N):
                v = puzzle[i][j]
                cx = j * spacing
                cy = (N - 1 - i) * spacing
                bg = "#e8edf2" if v != 0 else "#f7f9fc"
                ax.add_patch(patches.FancyBboxPatch(
                    (cx, cy), 1, 1,
                    boxstyle="round,pad=0.05",
                    linewidth=1.5, edgecolor="#2c3e50", facecolor=bg,
                ))
                if v != 0:
                    ax.text(cx + 0.5, cy + 0.5, str(v),
                            ha="center", va="center",
                            fontsize=max(12, 22 - N * 2),
                            fontweight="bold", color="#2c3e50")

        # Draw inequality symbols between adjacent cells
        for (a, b, sym) in ineqs:
            (r1, c1), (r2, c2) = a, b
            x1, y1 = c1 * spacing + 0.5, (N - 1 - r1) * spacing + 0.5
            x2, y2 = c2 * spacing + 0.5, (N - 1 - r2) * spacing + 0.5
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            # Horizontal pair
            if r1 == r2:
                # Symbol points from a to b: '<' means a<b
                disp = sym
            else:
                # Vertical pair: symbol mapped to ^/v
                # '<' (a<b, b is below) → '∧' (top of inequality, smaller side up)
                # We just print < or > rotated: use the chars 'v' / '^'
                disp = "v" if sym == "<" else "^"
            ax.text(mx, my, disp, ha="center", va="center",
                    fontsize=max(14, 22 - N), fontweight="bold",
                    color="#c0392b")

        # 0-indexed labels
        for i in range(N):
            ax.text(-0.20, (N - 1 - i) * spacing + 0.5, str(i),
                    ha="right", va="center",
                    fontsize=max(8, 14 - N), color="#7f8c8d")
        for j in range(N):
            ax.text(j * spacing + 0.5, N * spacing - 0.05 + 0.20, str(j),
                    ha="center", va="bottom",
                    fontsize=max(8, 14 - N), color="#7f8c8d")

        total = N * spacing
        ax.set_xlim(-0.5, total + 0.3)
        ax.set_ylim(-0.3, total + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import ast, re
        N = self._N
        pred_grid = None
        # Try Python literal
        try:
            obj = ast.literal_eval(predicted.strip())
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                pred_grid = [[int(x) for x in row] for row in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        # Fallback: N lines of N space-separated ints
        if pred_grid is None:
            try:
                lines = [l.strip() for l in predicted.splitlines() if l.strip()]
                if len(lines) == N:
                    g = []
                    for line in lines:
                        # Strip brackets/commas
                        line = re.sub(r"[\[\],]", " ", line)
                        nums = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
                        g.append(nums)
                    if all(len(r) == N for r in g):
                        pred_grid = g
            except (ValueError, AttributeError):
                pass
        # Fallback: extract any N*N ints in order
        if pred_grid is None:
            try:
                nums = [int(x) for x in re.findall(r"\d+", predicted)]
                if len(nums) == N * N:
                    pred_grid = [nums[i * N:(i + 1) * N] for i in range(N)]
            except (ValueError, TypeError):
                pass
        if pred_grid is None:
            return False
        if len(pred_grid) != N or any(len(r) != N for r in pred_grid):
            return False
        # Range check: 1..N
        for row in pred_grid:
            for v in row:
                if not (1 <= v <= N):
                    return False
        # Givens preserved
        for i in range(N):
            for j in range(N):
                g = self._puzzle[i][j]
                if g != 0 and pred_grid[i][j] != g:
                    return False
        # Row/col uniqueness
        expected = set(range(1, N + 1))
        for row in pred_grid:
            if set(row) != expected:
                return False
        for j in range(N):
            if set(pred_grid[i][j] for i in range(N)) != expected:
                return False
        # Inequalities
        for (a, b, sym) in self._inequalities:
            va = pred_grid[a[0]][a[1]]
            vb = pred_grid[b[0]][b[1]]
            if sym == "<" and not (va < vb):
                return False
            if sym == ">" and not (va > vb):
                return False
        return True
