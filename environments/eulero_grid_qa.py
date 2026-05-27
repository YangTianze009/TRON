"""
Eulero / Greco-Latin square grid puzzle.

An NxN grid where each cell holds a (letter, digit) pair. Both the letters
A..N and the digits 1..N must appear exactly once in each row and each column,
and every (letter, digit) pair must be globally unique.

Some cells are pre-filled (givens) and some are blank. The model must output
the completed grid using `|` separators per row and `\n` between rows.

Difficulty axes:
  - grid size N (3 -> 5)
  - number of givens (more givens = easier)
"""
import random
import re
from itertools import product
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the {n}x{n} Greco-Latin square (Eulero) puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each cell holds a (letter, digit) pair, e.g., A1.\n"
    "2. Each letter A..{last_letter} appears exactly once per row and per column.\n"
    "3. Each digit 1..{n} appears exactly once per row and per column.\n"
    "4. Every (letter, digit) pair appears at most once in the entire grid.\n"
    "5. Givens cannot be changed.\n\n"
    "### Coordinate System:\n"
    "- Cells are indexed (row, col), 0-indexed.\n"
    "- Cell separator: `|`; row separator: newline. Empty cell shown as a space.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Provide the completed grid as rows of `|`-separated letter+digit pairs (newline-separated rows) inside <answer>...</answer>.\n"
    "Example: <answer>B3|A2|C1\\nA1|C3|B2\\nC2|B1|A3</answer>",

    "Solve the {n}x{n} Eulero (Greco-Latin square) puzzle below.\n\n"
    "### Game Rules:\n"
    "- Each row/column has each letter once and each digit once.\n"
    "- Every (letter, digit) pair is globally unique.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output `|`-separated cells, newline-separated rows, inside <answer>...</answer>.",

    "Your task is to solve the {n}x{n} Greco-Latin square described below.\n\n"
    "### Game Rules:\n"
    "Standard Eulero / Greco-Latin square.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the completed grid inside <answer>...</answer>.",
]


def _eulero_state_text(grid, blanks, n):
    rows = []
    for r in range(n):
        cells = []
        for c in range(n):
            if (r, c) in blanks:
                cells.append("  ")
            else:
                p = grid[r][c]
                cells.append(f"{p[0]}{p[1]}")
        rows.append("|".join(cells))
    return "\n".join(rows)


def _build_eulero(n: int, rng: random.Random,
                    max_attempts: int = 80) -> Optional[List[List[Tuple[str, int]]]]:
    """Construct a valid N×N Greco-Latin square using two orthogonal Latin
    squares. For prime / odd N (3, 5, 7) we use cyclic constructions:
        L1[r][c] = (r + c) mod n  (letter index)
        L2[r][c] = (r + 2*c) mod n  (digit index)
    Then permute rows / cols / letter-symbols / digit-symbols randomly.

    For N=4 we construct using GF(4) directly.

    Returns a 2D list of (letter, digit_int) tuples, or None if construction
    fails (shouldn't for n in {3, 4, 5}).
    """
    if n == 4:
        # Use a hand-coded 4x4 Greco-Latin square (well-known)
        base_letters = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]
        base_digits = [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
        ]
    else:
        base_letters = [[(r + c) % n for c in range(n)] for r in range(n)]
        base_digits = [[(r + 2 * c) % n for c in range(n)] for r in range(n)]
        if n == 6:
            # No 6x6 Greco-Latin square exists (Euler's conjecture proven)
            # We don't generate n=6 anyway, just guard.
            return None

    # Verify orthogonality (every pair appears once)
    seen = set()
    ok = True
    for r in range(n):
        for c in range(n):
            p = (base_letters[r][c], base_digits[r][c])
            if p in seen:
                ok = False
                break
            seen.add(p)
        if not ok:
            break
    if not ok:
        return None

    # Now apply random row, col, letter-symbol, digit-symbol permutations
    row_perm = list(range(n))
    col_perm = list(range(n))
    letter_perm = list(range(n))
    digit_perm = list(range(n))
    rng.shuffle(row_perm)
    rng.shuffle(col_perm)
    rng.shuffle(letter_perm)
    rng.shuffle(digit_perm)

    final = [[None] * n for _ in range(n)]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:n]
    for r in range(n):
        for c in range(n):
            sr = row_perm[r]
            sc = col_perm[c]
            l_idx = letter_perm[base_letters[sr][sc]]
            d_idx = digit_perm[base_digits[sr][sc]]
            final[r][c] = (letters[l_idx], d_idx + 1)
    # Sanity check
    for r in range(n):
        if len(set(p[0] for p in final[r])) != n:
            return None
        if len(set(p[1] for p in final[r])) != n:
            return None
    for c in range(n):
        if len(set(final[r][c][0] for r in range(n))) != n:
            return None
        if len(set(final[r][c][1] for r in range(n))) != n:
            return None
    pairs = set()
    for r in range(n):
        for c in range(n):
            if final[r][c] in pairs:
                return None
            pairs.add(final[r][c])
    return final


def _is_eulero_valid(grid: List[List[Tuple[str, int]]], n: int) -> bool:
    """Check Latin-square + orthogonality constraints."""
    letters = set("ABCDEFGHIJ"[:n])
    digits = set(range(1, n + 1))
    if len(grid) != n:
        return False
    for row in grid:
        if len(row) != n:
            return False
    # Rows
    for r in range(n):
        if set(p[0] for p in grid[r]) != letters:
            return False
        if set(p[1] for p in grid[r]) != digits:
            return False
    # Cols
    for c in range(n):
        if set(grid[r][c][0] for r in range(n)) != letters:
            return False
        if set(grid[r][c][1] for r in range(n)) != digits:
            return False
    # Pairs unique
    pairs = set()
    for r in range(n):
        for c in range(n):
            p = grid[r][c]
            if p in pairs:
                return False
            pairs.add(p)
    return True


class EuleroGridQA(StandaloneVisualEnv):
    ENV_NAME = "eulero_grid"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the completed grid (rows of letter+digit pairs "
        "separated by `|`, rows on separate lines) directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n": 3, "blank_frac": 0.2}
        if level <= 2:
            return {"n": 3, "blank_frac": 0.4}
        if level <= 4:
            return {"n": 4, "blank_frac": 0.35}
        if level <= 6:
            return {"n": 4, "blank_frac": 0.55}
        if level <= 8:
            return {"n": 5, "blank_frac": 0.45}
        return {"n": 5, "blank_frac": 0.6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        blank_frac = cfg["blank_frac"]
        rng = random.Random((self.seed or 0) * 5527 + level * 41 + 23)

        for attempt in range(20):
            grid = _build_eulero(n, rng)
            if grid is None:
                continue
            # Pick a set of cells to blank out
            n_total = n * n
            n_blanks = max(1, int(n_total * blank_frac))
            cells = [(r, c) for r in range(n) for c in range(n)]
            rng.shuffle(cells)
            blanks = set(cells[:n_blanks])
            self._n = n
            self._solution = grid
            self._blanks = blanks

            # Format answer string
            ans_str = "\n".join(
                "|".join(f"{p[0]}{p[1]}" for p in grid[r]) for r in range(n)
            )
            sidx = (self.seed or 0) % len(_TEMPLATES)
            state_text = _eulero_state_text(grid, blanks, n)
            last_letter = chr(ord("A") + n - 1)
            question = _TEMPLATES[sidx].format(
                n=n, last_letter=last_letter, state=state_text,
            )
            img = self._render(grid, blanks, n, rng)
            return question, ans_str, img
        return None

    # ----------------------------------------------------------------- #
    def _render(self, grid, blanks, n, rng) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0"])
        cell = 1.0
        size_in = 0.85 * (n + 1)
        fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=150)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        for r in range(n):
            for c in range(n):
                fc = "#fff8e1" if (r, c) in blanks else "#e3f2fd"
                ax.add_patch(plt.Rectangle((c, n - 1 - r), cell, cell,
                                              facecolor=fc, edgecolor="#444",
                                              linewidth=1.4))
                if (r, c) not in blanks:
                    p = grid[r][c]
                    ax.text(c + 0.5, n - 0.5 - r, f"{p[0]}{p[1]}",
                             ha="center", va="center",
                             fontsize=22, fontweight="bold",
                             color="#1a237e")
                else:
                    # leave blank — could draw a faint placeholder
                    pass

        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Eulero (Greco-Latin) puzzle",
                     fontsize=12, color="#1a237e", pad=4)
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                     facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ----------------------------------------------------------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Parse predicted as N rows separated by newlines, cells separated by
        # `|`. Each cell should be `<letter><digit>` (e.g., A1, B12 not allowed
        # at our sizes).
        n = self._n
        # Strip backtick wrappers and surrounding whitespace
        text = predicted.strip().strip("`").strip()
        # Split on newlines
        rows = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(rows) != n:
            return False
        grid = []
        for row in rows:
            # cells are separated by `|`. Trim trailing/leading `|`.
            cells = [c.strip() for c in row.strip("|").split("|") if c.strip()]
            if len(cells) != n:
                return False
            row_pairs = []
            for cell in cells:
                # cell should be one upper-case letter followed by digits
                m = re.match(r"^([A-Ja-j])\s*(\d+)$", cell)
                if not m:
                    return False
                letter = m.group(1).upper()
                try:
                    digit = int(m.group(2))
                except ValueError:
                    return False
                row_pairs.append((letter, digit))
            grid.append(row_pairs)
        if not _is_eulero_valid(grid, n):
            return False
        # Also: must respect the givens (cells not in blanks must match)
        for r in range(n):
            for c in range(n):
                if (r, c) not in self._blanks:
                    if grid[r][c] != self._solution[r][c]:
                        return False
        return True


if __name__ == "__main__":
    env = EuleroGridQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer!r}")
