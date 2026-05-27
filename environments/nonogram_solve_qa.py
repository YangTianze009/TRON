"""
Nonogram (Picross) puzzle (structured-puzzle style).

reference qids studied (nonogram, total 30): idx=811, 812, 817, 818, 823, 824,
829, 830, 835, 836 (verbatim per `design notes` §nonogram, lines
677-700). Sample TSV format confirmed via reference dataset:
  - question_text starts with: "You will be given an image of a Nonogram puzzle..."
  - Symbols: 'X' = filled cell, '.' = empty cell
  - answer: N lines, each N chars long, e.g. "XXXX\\n....\\nXXXX\\nXXXX"

Self-contained: random binary grid + clue computation. NO `from RLVE`,
`from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 3x3 with 30% density
  L9 within solvable range: 7x7 with ~50% density
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
    "Your task is to solve the {N}x{N} Nonogram puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each row and column has a sequence of clue numbers giving the lengths of consecutive filled-cell runs (in order).\n"
    "2. Runs in the same row/column are separated by at least one empty cell.\n"
    "3. Use `X` for a filled cell and `.` for an empty cell.\n\n"
    "### Coordinate System:\n"
    "- Grid is {N}x{N}, indexed (row, col) with row 0 at top and column 0 at left.\n\n"
    "### Current Puzzle State:\n"
    "- Row clues (top-to-bottom, each is the list of run lengths): {row_clues}\n"
    "- Column clues (left-to-right, each is the list of run lengths): {col_clues}\n\n"
    "### Output Format:\n"
    "Output {N} rows of {N} characters each (`X`/`.`) separated by newlines, inside <answer>...</answer>.\n"
    "Example: <answer>XXXX\\n....\\nXXXX\\nXXXX</answer>",

    "Solve the Nonogram puzzle below.\n\n"
    "### Game Rules:\n"
    "- Each clue lists run lengths of filled cells in that row/column, in order.\n"
    "- Distinct runs are separated by at least one empty cell.\n"
    "- Use `X` (filled) and `.` (empty).\n\n"
    "### Coordinate System:\n"
    "- Grid is {N}x{N}, 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Row clues: {row_clues}\n"
    "Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "Output {N} lines of {N} characters from `X`/`.` inside <answer>...</answer>.",

    "Your task is to solve the {N}x{N} Nonogram described below.\n\n"
    "### Game Rules:\n"
    "Standard Nonogram: row and column clues encode consecutive filled-cell run lengths. Output the X/. grid that matches every clue.\n\n"
    "### Coordinate System:\n"
    "- {N}x{N} grid, rows top-to-bottom, columns left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "Rows: {row_clues}\n"
    "Cols: {col_clues}\n\n"
    "### Output Format:\n"
    "Output {N} rows of `X`/`.` characters separated by newlines inside <answer>...</answer>.",
]


# L0-L2 probe: ask if a SPECIFIC cell is FILLED in the solution.
_PROBE_TEMPLATES = [
    "Your task is to determine the status of one cell in the {N}x{N} Nonogram puzzle below:\n\n"
    "### Game Rules:\n"
    "1. Each row and column has clue numbers giving consecutive filled-cell run lengths.\n"
    "2. `X` is a filled cell; `.` is empty.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed; row 0 at top.\n\n"
    "### Current Puzzle State:\n"
    "- Row clues (top-to-bottom): {row_clues}\n"
    "- Column clues (left-to-right): {col_clues}\n\n"
    "### Output Format:\n"
    "In the solution, is the cell at (row {r}, col {c}) FILLED (`X`)? Answer YES or NO inside <answer>...</answer>.",

    "Look at the Nonogram puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Nonogram: clue numbers encode run lengths.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Row clues: {row_clues}\n"
    "Col clues: {col_clues}\n\n"
    "### Output Format:\n"
    "Should cell (row {r}, col {c}) be filled (`X`) in the solution? Answer YES or NO inside <answer>...</answer>.",

    "Nonogram puzzle ({N}x{N}).\n\n"
    "### Game Rules:\n"
    "Standard Nonogram rules.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Rows: {row_clues}\n"
    "Cols: {col_clues}\n\n"
    "### Output Format:\n"
    "Is cell (row {r}, col {c}) filled in the solution? Output YES or NO inside <answer>...</answer>.",
]


def _compute_clue(line: List[int]) -> List[int]:
    """Compute the run-length clue for a binary line."""
    clue = []
    count = 0
    for v in line:
        if v == 1:
            count += 1
        else:
            if count > 0:
                clue.append(count)
            count = 0
    if count > 0:
        clue.append(count)
    if not clue:
        clue = [0]
    return clue


class NonogramSolveQA(StandaloneVisualEnv):
    ENV_NAME = "nonogram_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the final grid directly inside "
        "`<answer>...</answer>` tags."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode — small grid, ask YES/NO about one cell.
        if level == 0:
            return {"N": 3, "density": 0.3, "level": level, "probe": True}
        if level == 1:
            return {"N": 3, "density": 0.4, "level": level, "probe": True}
        if level == 2:
            return {"N": 4, "density": 0.4, "level": level, "probe": True}
        if level <= 4:
            N, density = 4, 0.40 + 0.05 * (level - 2)
        elif level <= 7:
            N, density = 5, 0.40 + 0.05 * (level - 5)
        elif level == 8:
            N, density = 6, 0.45
        else:
            N, density = 7, 0.45
        return {"N": N, "density": density, "level": level, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        N, density = cfg["N"], cfg["density"]
        rng = random.Random((seed or 0) * 991 + level * 31 + 17)

        # Random binary solution
        solution = [[1 if rng.random() < density else 0 for _ in range(N)]
                    for _ in range(N)]
        # Ensure at least one filled cell exists
        if all(solution[i][j] == 0 for i in range(N) for j in range(N)):
            solution[rng.randint(0, N - 1)][rng.randint(0, N - 1)] = 1

        row_clues = [_compute_clue(solution[i]) for i in range(N)]
        col_clues = [_compute_clue([solution[i][j] for i in range(N)])
                     for j in range(N)]

        self._N = N
        self._row_clues = row_clues
        self._col_clues = col_clues
        self._probe_mode = cfg["probe"]

        if cfg["probe"]:
            target_r = rng.randint(0, N - 1)
            target_c = rng.randint(0, N - 1)
            is_filled = (solution[target_r][target_c] == 1)
            ans_str = "YES" if is_filled else "NO"
            self._probe_target = (target_r, target_c)
            self._probe_answer = ans_str
            sidx = (seed or 0) % len(_PROBE_TEMPLATES)
            question = _PROBE_TEMPLATES[sidx].format(
                N=N, row_clues=str(row_clues), col_clues=str(col_clues),
                r=target_r, c=target_c,
            )
            img = self._render(N, row_clues, col_clues)
            return question, ans_str, img

        sidx = (seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            N=N, row_clues=str(row_clues), col_clues=str(col_clues),
        )
        ans_str = "\n".join("".join("X" if c == 1 else "." for c in row)
                            for row in solution)
        img = self._render(N, row_clues, col_clues)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, N, row_clues, col_clues) -> Image.Image:
        max_row = max(len(c) for c in row_clues)
        max_col = max(len(c) for c in col_clues)
        cell = 0.55
        clue_left = max_row * cell
        clue_top = max_col * cell
        fig_w = clue_left + N * cell + 0.8
        fig_h = clue_top + N * cell + 0.8
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=120)

        BORDER = "#2c3e50"
        CLUE_BG = "#ecf0f1"
        CELL_BG = "#fafafa"
        GRID_LINE = "#bdc3c7"

        font_size = max(8, min(14, int(cell * 20)))
        clue_fs = max(7, min(13, int(cell * 18)))

        grid_x0 = clue_left
        grid_y0 = 0.0

        for i in range(N):
            for j in range(N):
                x = grid_x0 + j * cell
                y = grid_y0 + (N - 1 - i) * cell
                ax.add_patch(patches.Rectangle(
                    (x, y), cell, cell, linewidth=0.8,
                    edgecolor=GRID_LINE, facecolor=CELL_BG,
                ))
        ax.add_patch(patches.Rectangle(
            (grid_x0, grid_y0), N * cell, N * cell,
            linewidth=2.5, edgecolor=BORDER, facecolor="none",
        ))

        # Row clues (left)
        for i in range(N):
            clue = row_clues[i]
            y_center = grid_y0 + (N - 1 - i) * cell + cell / 2
            for k, val in enumerate(reversed(clue)):
                x_pos = grid_x0 - (k + 1) * cell + cell / 2
                ax.add_patch(patches.Rectangle(
                    (grid_x0 - (k + 1) * cell, y_center - cell / 2),
                    cell, cell, linewidth=0.5,
                    edgecolor=GRID_LINE, facecolor=CLUE_BG,
                ))
                ax.text(x_pos, y_center, str(val),
                        ha="center", va="center",
                        fontsize=clue_fs, fontweight="bold", color=BORDER)
        # Col clues (top)
        for j in range(N):
            clue = col_clues[j]
            x_center = grid_x0 + j * cell + cell / 2
            for k, val in enumerate(reversed(clue)):
                y_pos = grid_y0 + N * cell + k * cell + cell / 2
                ax.add_patch(patches.Rectangle(
                    (x_center - cell / 2, grid_y0 + N * cell + k * cell),
                    cell, cell, linewidth=0.5,
                    edgecolor=GRID_LINE, facecolor=CLUE_BG,
                ))
                ax.text(x_center, y_pos, str(val),
                        ha="center", va="center",
                        fontsize=clue_fs, fontweight="bold", color=BORDER)

        # Labels
        label_fs = max(6, clue_fs - 2)
        for i in range(N):
            ax.text(grid_x0 + N * cell + 0.12,
                    grid_y0 + (N - 1 - i) * cell + cell / 2,
                    str(i + 1), ha="left", va="center",
                    fontsize=label_fs, color="#999")
        for j in range(N):
            ax.text(grid_x0 + j * cell + cell / 2, grid_y0 - 0.12,
                    str(j + 1), ha="center", va="top",
                    fontsize=label_fs, color="#999")

        ax.set_title(f"Nonogram  ({N} x {N})",
                     fontsize=font_size + 2, fontweight="bold",
                     color=BORDER, pad=8)

        margin = 0.3
        ax.set_xlim(grid_x0 - clue_left - margin, grid_x0 + N * cell + margin + 0.3)
        ax.set_ylim(grid_y0 - margin - 0.2, grid_y0 + N * cell + clue_top + margin)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # PROBE MODE: yes/no
        if getattr(self, "_probe_mode", False):
            import re
            text = predicted.strip().lower()
            m = re.search(r"\b(yes|no|y|n)\b", text)
            if m:
                w = m.group(1)
                pred = "YES" if w.startswith("y") else "NO"
                return pred == self._probe_answer
            return False
        N = self._N
        # Decode literal "\n" the model sometimes emits inside <answer>.
        if "\\n" in predicted:
            predicted = predicted.replace("\\n", "\n").replace("\\t", "\t")
        # Strip optional brackets/quotes; consider any line that is non-empty
        # after stripping spaces and that contains only `X`/`.`/`x` chars.
        # reference uses uppercase `X` and `.` per format.
        text = predicted.strip()
        # Replace `x` (lowercase) with `X` and remove inline spaces
        lines_raw = [l.strip() for l in text.splitlines()]
        # Filter out lines that are purely separators (only commas/spaces) but
        # not ones with X/.
        candidate = []
        for ln in lines_raw:
            cleaned = ln.replace(" ", "").replace(",", "")
            if not cleaned:
                continue
            # Accept rows containing only X/x/./# (some models use #)
            allowed = set("Xx.#")
            if all(c in allowed for c in cleaned):
                # Map # to X (alternative filled symbol); x to X
                cleaned = cleaned.replace("#", "X").replace("x", "X")
                candidate.append(cleaned)
        if len(candidate) != N:
            return False
        for ln in candidate:
            if len(ln) != N:
                return False

        grid = []
        for ln in candidate:
            row = [1 if ch == "X" else 0 for ch in ln]
            grid.append(row)

        # Re-validate clues
        for i in range(N):
            if _compute_clue(grid[i]) != self._row_clues[i]:
                return False
        for j in range(N):
            col = [grid[i][j] for i in range(N)]
            if _compute_clue(col) != self._col_clues[j]:
                return False
        return True
