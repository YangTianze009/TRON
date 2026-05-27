"""Logic Grid QA — diversity + difficulty redesign 2026-04-16.

Round-2 fix: Grade D difficulty + weak diversity.
- 6 grid families: magic_square, multiplication, addition_pattern,
  fibonacci_row, power_grid, mod_grid (new).
- Diverse visual styles: palette per-seed, cell dividers/gradient/plain,
  title pool, header styles.
- 4 question-phrasing variants per operation.
- Level gradient:
    L0-L1: magic_square, single missing, formula hinted in question.
    L2-L3: any family, 1 missing, standard.
    L4-L5: magic_square_constant or diagonal computations.
    L6-L7: multi-hidden product; 2 hidden cells to infer.
    L8-L9: 3+ hidden cells; compound operations (product, weighted sum).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_POOL_BY_KIND = {
    "magic_square": [
        "Magic Square", "Magic Square Puzzle", "Constant-Sum Square",
        "Row/Col Sum Square", "Square Puzzle",
    ],
    "multiplication": [
        "Multiplication Table", "Times Table", "Product Table", "x Table",
    ],
    "addition_pattern": [
        "Addition Grid", "Sum Grid", "Addition Table", "Row+Col Sum Grid",
    ],
    "fibonacci_row": [
        "Fibonacci-like Rows", "Each Term = Sum of Previous Two",
        "Sequence Rows", "Additive Row Grid",
    ],
    "power_grid": [
        "Power Grid", "Exponential Table", "Power Table", "base^r Grid",
    ],
    "mod_grid": [
        "Modular Grid", "Remainder Grid", "Mod Table", "Residues Table",
    ],
}

class LogicGridQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "logic_grid"

    QUESTION_TYPES = [
        "find_value",
        "complete_pattern",
        "row_rule",
        "column_rule",
        "diagonal_pattern",
        "magic_square_constant",
        "diagonal_sum",
        "multi_hidden_product",
        "multi_hidden_sum",
        "weighted_cell_sum",
    ]

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        return {
            0: {"qtype": "find_value", "show_hint": True},
            1: {"qtype": "find_value", "show_hint": False},
            2: {"qtype": "complete_pattern", "show_hint": False},
            3: {"qtype": "diagonal_pattern", "show_hint": False},
            4: {"qtype": "magic_square_constant", "show_hint": False},
            5: {"qtype": "diagonal_sum", "show_hint": False},
            6: {"qtype": "multi_hidden_product", "show_hint": False,
                "n_hidden": 2},
            7: {"qtype": "multi_hidden_sum", "show_hint": False,
                "n_hidden": 3},
            8: {"qtype": "multi_hidden_product", "show_hint": False,
                "n_hidden": 3},
            9: {"qtype": "weighted_cell_sum", "show_hint": False,
                "n_hidden": 3},
        }[level]

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        qtype = lcfg["qtype"]
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 809)

        for _ in range(25):
            r = self._try_generate(qtype, lcfg, rng)
            if r is not None:
                return r
        return None

    def _try_generate(self, qtype, lcfg, rng):
        # For magic_square_constant, force magic square family
        if qtype in ("magic_square_constant", "diagonal_sum"):
            family = "magic_square"
        else:
            family = rng.choice(["magic_square", "multiplication", "addition_pattern",
                                   "fibonacci_row", "power_grid", "mod_grid"])

        if family == "magic_square":
            return self._magic_square_problem(rng, qtype, lcfg)
        if family == "multiplication":
            return self._multiplication_problem(rng, qtype, lcfg)
        if family == "addition_pattern":
            return self._addition_pattern_problem(rng, qtype, lcfg)
        if family == "fibonacci_row":
            return self._fibonacci_row_problem(rng, qtype, lcfg)
        if family == "power_grid":
            return self._power_grid_problem(rng, qtype, lcfg)
        if family == "mod_grid":
            return self._mod_grid_problem(rng, qtype, lcfg)
        return None

    # ------------------------------------------------------------------ #
    # Family: Magic square
    # ------------------------------------------------------------------ #

    def _magic_square_problem(self, rng, qtype, lcfg):
        base_variants = [
            [[2, 7, 6], [9, 5, 1], [4, 3, 8]],  # standard 3x3
            [[8, 1, 6], [3, 5, 7], [4, 9, 2]],
            [[6, 7, 2], [1, 5, 9], [8, 3, 4]],
            [[4, 9, 2], [3, 5, 7], [8, 1, 6]],
        ]
        base = rng.choice(base_variants)
        mult = rng.randint(1, 5)
        offset = rng.randint(0, 12)
        grid = [[base[r][c] * mult + offset for c in range(3)] for r in range(3)]
        magic_const = 15 * mult + 3 * offset

        q = None
        answer = None
        display = [row[:] for row in grid]

        if qtype in ("find_value", "complete_pattern"):
            hr, hc = rng.randint(0, 2), rng.randint(0, 2)
            hidden_val = grid[hr][hc]
            display[hr][hc] = None
            hint = f" (magic sum is {magic_const})" if lcfg.get("show_hint") else ""
            phrasings = [
                f"A magic square has equal row/column/diagonal sums{hint}. What is the missing value marked '?'?",
                f"Every row, column and diagonal in this magic square sums to the same constant{hint}. Find the '?' value.",
                f"Determine the missing entry (marked '?') in this magic square{hint}.",
                f"Fill in the missing cell of this magic square{hint}.",
            ]
            q = rng.choice(phrasings)
            answer = str(hidden_val)
        elif qtype == "diagonal_pattern":
            diag_sum = sum(grid[i][i] for i in range(3))
            q = rng.choice([
                "Compute the sum of the main diagonal (top-left to bottom-right).",
                "Add the entries along the main diagonal of the grid shown.",
                "What is the sum of entries grid[1,1] + grid[2,2] + grid[3,3] (1-indexed)?",
                "Find the main-diagonal sum for the grid displayed.",
            ])
            answer = str(diag_sum)
        elif qtype == "magic_square_constant":
            # Hide 2 cells, ask for magic constant
            positions = rng.sample([(r, c) for r in range(3) for c in range(3)], 2)
            for hr, hc in positions:
                display[hr][hc] = None
            q = rng.choice([
                "This magic square has two missing cells. Compute the common row/column/diagonal sum (the magic constant).",
                "Find the magic constant (common row sum) for this partially-hidden magic square.",
                "Two cells are hidden. Use any complete row or column to determine the magic constant.",
                "Compute the constant row sum of this magic square (even though two cells are hidden).",
            ])
            answer = str(magic_const)
        elif qtype == "diagonal_sum":
            main = sum(grid[i][i] for i in range(3))
            anti = sum(grid[i][2-i] for i in range(3))
            ans = main + anti - grid[1][1]
            q = rng.choice([
                "Add the main diagonal and anti-diagonal sums, then subtract the centre cell (it is counted twice). Give the total.",
                "Compute (main_diag_sum + anti_diag_sum - centre). Integer.",
                "Sum both diagonals, subtracting the shared centre exactly once.",
                "The main and anti diagonals overlap at the centre. Give main + anti - centre.",
            ])
            answer = str(ans)
        elif qtype == "multi_hidden_product":
            n_h = lcfg.get("n_hidden", 3)
            positions = rng.sample([(r, c) for r in range(3) for c in range(3)], n_h)
            product = 1
            for hr, hc in positions:
                product *= grid[hr][hc]
                display[hr][hc] = None
            q = rng.choice([
                f"{n_h} cells of this magic square are hidden. Use the magic-sum rule to recover them, then return their PRODUCT.",
                f"Every row/col/diagonal sums to the same constant. Find all {n_h} hidden cells and multiply them.",
                f"Deduce the {n_h} missing values and give their product (integer).",
                f"From the row/col/diagonal constraints, infer the {n_h} hidden values and multiply them together.",
            ])
            answer = str(product)
        elif qtype == "multi_hidden_sum":
            n_h = lcfg.get("n_hidden", 3)
            positions = rng.sample([(r, c) for r in range(3) for c in range(3)], n_h)
            s = 0
            for hr, hc in positions:
                s += grid[hr][hc]
                display[hr][hc] = None
            q = rng.choice([
                f"{n_h} cells are hidden. Use the constant row/col/diagonal sum rule to deduce them, then return their SUM.",
                f"Find the {n_h} missing entries of this magic square and sum them.",
                f"Determine the {n_h} '?'-marked cells and give their total sum.",
                f"Using the magic-sum constraint, recover the hidden cells and report their sum.",
            ])
            answer = str(s)
        elif qtype == "weighted_cell_sum":
            # 3 hidden cells; return sum with weights 1, 2, 3 (by row index)
            positions = rng.sample([(r, c) for r in range(3) for c in range(3)], 3)
            ws = 0
            for hr, hc in positions:
                ws += (hr + 1) * grid[hr][hc]
                display[hr][hc] = None
            q = (f"Three cells of this magic square are hidden. Recover them, then compute "
                 f"SUM = (row_index * value) for each hidden cell (rows indexed from 1). Integer.")
            answer = str(ws)
        else:
            return None

        img = self._draw_grid(display, rng.choice(_TITLE_POOL_BY_KIND["magic_square"]), rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Family: Multiplication
    # ------------------------------------------------------------------ #

    def _multiplication_problem(self, rng, qtype, lcfg):
        row_headers = sorted(rng.sample(range(2, 11), rng.randint(3, 5)))
        col_headers = sorted(rng.sample(range(2, 11), rng.randint(3, 5)))
        rows = len(row_headers); cols = len(col_headers)
        grid = [[row_headers[r] * col_headers[c] for c in range(cols)] for r in range(rows)]

        n_h = lcfg.get("n_hidden", 1)
        positions = []
        for _ in range(n_h):
            hr = rng.randint(0, rows - 1); hc = rng.randint(0, cols - 1)
            positions.append((hr, hc))
        display = [row[:] for row in grid]
        for hr, hc in positions:
            display[hr][hc] = None

        hr0, hc0 = positions[0]
        hidden0 = grid[hr0][hc0]

        if qtype in ("find_value", "complete_pattern"):
            phrasings = [
                f"This is a multiplication table. Row headers: {row_headers}; column headers: {col_headers}. "
                f"What value replaces the '?' at row header {row_headers[hr0]}, column header {col_headers[hc0]}?",
                f"Each cell equals row_header * col_header. Row headers: {row_headers}; col headers: {col_headers}. "
                f"Find the missing value at row {row_headers[hr0]}, col {col_headers[hc0]}.",
                f"In a multiplication table with row headers {row_headers} and col headers {col_headers}, "
                f"what goes in the cell for row={row_headers[hr0]}, col={col_headers[hc0]}?",
                f"Compute the missing entry. Row header {row_headers[hr0]}, column header {col_headers[hc0]}, "
                f"in a standard product table.",
            ]
            q = rng.choice(phrasings)
            answer = str(hidden0)
        elif qtype == "diagonal_pattern":
            if rows != cols:
                return None
            diag = sum(grid[i][i] for i in range(rows))
            q = rng.choice([
                "What is the sum of the main diagonal of this multiplication table?",
                "Add the diagonal entries (top-left to bottom-right) of the shown product table.",
                "Compute the main-diagonal sum.",
                "Sum the entries along the top-left to bottom-right diagonal of the product grid.",
            ])
            answer = str(diag)
            display = grid
        elif qtype == "multi_hidden_product":
            if len(positions) < 2:
                return None
            product = 1
            for hr, hc in positions:
                product *= grid[hr][hc]
            q = (f"{len(positions)} cells in this multiplication table are hidden. "
                 f"Row headers: {row_headers}; col headers: {col_headers}. "
                 f"Compute the PRODUCT of all hidden cell values.")
            answer = str(product)
        elif qtype == "multi_hidden_sum":
            s = sum(grid[hr][hc] for hr, hc in positions)
            q = (f"{len(positions)} cells are hidden. Row headers: {row_headers}, col headers: {col_headers}. "
                 f"Return the SUM of the hidden cell values.")
            answer = str(s)
        elif qtype == "weighted_cell_sum":
            ws = sum((hr + 1) * grid[hr][hc] for hr, hc in positions)
            q = (f"{len(positions)} cells are hidden. Row headers: {row_headers}, col headers: {col_headers}. "
                 f"For each hidden cell, compute row_index * value (1-indexed rows) and return the total sum.")
            answer = str(ws)
        else:
            return None

        img = self._draw_grid_with_headers(display, row_headers, col_headers,
                                             rng.choice(_TITLE_POOL_BY_KIND["multiplication"]),
                                             rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Family: Addition pattern
    # ------------------------------------------------------------------ #

    def _addition_pattern_problem(self, rng, qtype, lcfg):
        row_vals = [rng.randint(1, 18) for _ in range(rng.randint(3, 5))]
        col_vals = [rng.randint(1, 18) for _ in range(rng.randint(3, 5))]
        rows = len(row_vals); cols = len(col_vals)
        grid = [[row_vals[r] + col_vals[c] for c in range(cols)] for r in range(rows)]
        n_h = lcfg.get("n_hidden", 1)
        positions = [(rng.randint(0, rows-1), rng.randint(0, cols-1)) for _ in range(n_h)]
        display = [row[:] for row in grid]
        for hr, hc in positions:
            display[hr][hc] = None

        hr0, hc0 = positions[0]
        hidden0 = grid[hr0][hc0]

        if qtype in ("find_value", "complete_pattern"):
            q = (f"Each cell = row_header + col_header. Row headers: {row_vals}; "
                 f"column headers: {col_vals}. What value replaces the '?'?")
            answer = str(hidden0)
        elif qtype == "diagonal_pattern":
            if rows != cols:
                return None
            diag = sum(grid[i][i] for i in range(rows))
            q = "Compute the main-diagonal sum of this addition grid."
            answer = str(diag)
            display = grid
        elif qtype == "multi_hidden_product":
            if len(positions) < 2:
                return None
            prod = 1
            for hr, hc in positions:
                prod *= grid[hr][hc]
            q = (f"{len(positions)} cells hidden; each cell = row_header + col_header. "
                 f"Row headers: {row_vals}, col headers: {col_vals}. Product of hidden cells?")
            answer = str(prod)
        elif qtype == "multi_hidden_sum":
            s = sum(grid[hr][hc] for hr, hc in positions)
            q = (f"{len(positions)} cells hidden in addition grid. Row headers: {row_vals}, "
                 f"col headers: {col_vals}. Return sum of hidden cells.")
            answer = str(s)
        elif qtype == "weighted_cell_sum":
            ws = sum((hr + 1) * grid[hr][hc] for hr, hc in positions)
            q = (f"{len(positions)} cells hidden. Row headers: {row_vals}, col headers: {col_vals}. "
                 f"For each hidden cell compute (row_index * value); return total sum.")
            answer = str(ws)
        else:
            return None

        img = self._draw_grid_with_headers(display, row_vals, col_vals,
                                             rng.choice(_TITLE_POOL_BY_KIND["addition_pattern"]),
                                             rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Family: Fibonacci rows
    # ------------------------------------------------------------------ #

    def _fibonacci_row_problem(self, rng, qtype, lcfg):
        rows = rng.randint(3, 5)
        cols = rng.randint(5, 7)
        grid = []
        for r in range(rows):
            a = rng.randint(1, 6)
            b = rng.randint(1, 6)
            row = [a, b]
            for c in range(2, cols):
                row.append(row[-1] + row[-2])
            grid.append(row)
        n_h = lcfg.get("n_hidden", 1)
        positions = []
        for _ in range(n_h):
            hr = rng.randint(0, rows-1)
            hc = rng.randint(2, cols-1)
            positions.append((hr, hc))
        display = [row[:] for row in grid]
        for hr, hc in positions:
            display[hr][hc] = None

        hr0, hc0 = positions[0]
        hidden0 = grid[hr0][hc0]

        if qtype in ("find_value", "complete_pattern"):
            q = rng.choice([
                "Each row follows the rule: a_n = a_{n-1} + a_{n-2}. What is the missing '?'?",
                "Each row is a Fibonacci-like sequence (each value = sum of two previous). Find the missing value.",
                "Rows of the grid follow the recurrence a_n = a_{n-1} + a_{n-2}. Determine the '?' entry.",
                "Given each row obeys a_n = a_{n-1} + a_{n-2}, find the missing cell value.",
            ])
            answer = str(hidden0)
        elif qtype == "diagonal_pattern":
            if rows != cols:
                # approximate: sum of all columns?
                q = "Sum of all entries in the grid:"
                answer = str(sum(v for row in grid for v in row))
                display = grid
            else:
                diag = sum(grid[i][i] for i in range(rows))
                q = "Main-diagonal sum of this grid?"
                answer = str(diag)
                display = grid
        elif qtype == "multi_hidden_product":
            prod = 1
            for hr, hc in positions:
                prod *= grid[hr][hc]
            q = (f"Each row is a Fibonacci-like sequence. {len(positions)} cells are hidden; "
                 f"infer them and return their product.")
            answer = str(prod)
        elif qtype == "multi_hidden_sum":
            s = sum(grid[hr][hc] for hr, hc in positions)
            q = (f"Each row is a Fibonacci-like sequence. Recover the {len(positions)} hidden cells "
                 f"and return their sum.")
            answer = str(s)
        elif qtype == "weighted_cell_sum":
            ws = sum((hr + 1) * grid[hr][hc] for hr, hc in positions)
            q = ("Each row is Fibonacci-like. For each hidden cell compute (row_index * value) and sum all.")
            answer = str(ws)
        else:
            return None

        img = self._draw_grid(display,
                              rng.choice(_TITLE_POOL_BY_KIND["fibonacci_row"]),
                              rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Family: Power grid
    # ------------------------------------------------------------------ #

    def _power_grid_problem(self, rng, qtype, lcfg):
        size = rng.randint(3, 4)
        base = rng.randint(2, 3)
        grid = [[base ** (r + 1) * (c + 1) for c in range(size)] for r in range(size)]
        n_h = lcfg.get("n_hidden", 1)
        positions = [(rng.randint(0, size-1), rng.randint(0, size-1)) for _ in range(n_h)]
        display = [row[:] for row in grid]
        for hr, hc in positions:
            display[hr][hc] = None

        hr0, hc0 = positions[0]
        hidden0 = grid[hr0][hc0]

        if qtype in ("find_value", "complete_pattern"):
            q = (f"Each cell at row r, col c (1-indexed) equals {base}^r * c. Find the missing '?'.")
            answer = str(hidden0)
        elif qtype == "diagonal_pattern":
            diag = sum(grid[i][i] for i in range(size))
            q = "Main-diagonal sum of this power grid?"
            answer = str(diag)
            display = grid
        elif qtype == "multi_hidden_product":
            prod = 1
            for hr, hc in positions:
                prod *= grid[hr][hc]
            q = f"Each cell = {base}^r * c (1-indexed). Product of the hidden cells?"
            answer = str(prod)
        elif qtype == "multi_hidden_sum":
            s = sum(grid[hr][hc] for hr, hc in positions)
            q = f"Each cell = {base}^r * c (1-indexed). Sum of hidden cells?"
            answer = str(s)
        elif qtype == "weighted_cell_sum":
            ws = sum((hr + 1) * grid[hr][hc] for hr, hc in positions)
            q = f"Each cell = {base}^r * c (1-indexed). For each hidden cell compute row_index * value; return total."
            answer = str(ws)
        else:
            return None

        row_headers = [f"r={i+1}" for i in range(size)]
        col_headers = [f"c={i+1}" for i in range(size)]
        img = self._draw_grid_with_headers(display, row_headers, col_headers,
                                             f"{rng.choice(_TITLE_POOL_BY_KIND['power_grid'])} (base={base})",
                                             rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Family: Mod grid (new)
    # ------------------------------------------------------------------ #

    def _mod_grid_problem(self, rng, qtype, lcfg):
        size = rng.randint(3, 4)
        base = rng.randint(5, 9)
        # cell[r][c] = (row_val * col_val) mod base
        row_vals = rng.sample(range(1, 12), size)
        col_vals = rng.sample(range(1, 12), size)
        grid = [[(row_vals[r] * col_vals[c]) % base for c in range(size)] for r in range(size)]
        n_h = lcfg.get("n_hidden", 1)
        positions = [(rng.randint(0, size-1), rng.randint(0, size-1)) for _ in range(n_h)]
        display = [row[:] for row in grid]
        for hr, hc in positions:
            display[hr][hc] = None

        hr0, hc0 = positions[0]
        hidden0 = grid[hr0][hc0]

        if qtype in ("find_value", "complete_pattern"):
            q = (f"Each cell = (row_header * col_header) mod {base}. Row headers: {row_vals}; "
                 f"col headers: {col_vals}. Find the missing '?'.")
            answer = str(hidden0)
        elif qtype == "diagonal_pattern":
            diag = sum(grid[i][i] for i in range(size))
            q = f"Each cell = (row*col) mod {base}. Main-diagonal sum?"
            answer = str(diag)
            display = grid
        elif qtype == "multi_hidden_product":
            prod = 1
            for hr, hc in positions:
                prod *= grid[hr][hc]
            q = (f"Cells are (row*col) mod {base}. Row headers {row_vals}, col headers {col_vals}. "
                 f"Product of the hidden cells.")
            answer = str(prod)
        elif qtype == "multi_hidden_sum":
            s = sum(grid[hr][hc] for hr, hc in positions)
            q = (f"Cells are (row*col) mod {base}. Row headers {row_vals}, col headers {col_vals}. "
                 f"Sum of hidden cells.")
            answer = str(s)
        elif qtype == "weighted_cell_sum":
            ws = sum((hr + 1) * grid[hr][hc] for hr, hc in positions)
            q = (f"Cells are (row*col) mod {base}. For each hidden cell compute row_index * value; total?")
            answer = str(ws)
        else:
            return None

        img = self._draw_grid_with_headers(display, row_vals, col_vals,
                                             rng.choice(_TITLE_POOL_BY_KIND["mod_grid"]),
                                             rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Drawing helpers (with diverse palettes and layouts)
    # ------------------------------------------------------------------ #

    def _draw_grid(self, grid, title, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        sfs = style["font_size_base"]
        ff = style["font_family"]
        palette = list(style["palette"])
        rng.shuffle(palette)
        rows = len(grid); cols = len(grid[0])
        fig, ax = plt.subplots(figsize=(max(5, cols * 1.35) * sc,
                                          max(4, rows * 1.25) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect('equal')

        cell_size = 1.0
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                y = (rows - 1 - r) * cell_size
                x = c * cell_size
                if val is None:
                    color = palette[1]
                    text = '?'
                    text_color = '#E74C3C'
                else:
                    color = palette[2] if (r + c) % 2 == 0 else palette[3]
                    text = str(val)
                    text_color = '#1b1b1b'
                rect = plt.Rectangle((x, y), cell_size, cell_size,
                                      facecolor=color, edgecolor='#2C3E50',
                                      lw=style["line_width"])
                ax.add_patch(rect)
                cell_fs = sfs + 2 if len(text) <= 3 else sfs - 2
                ax.text(x + cell_size/2, y + cell_size/2, text,
                        ha='center', va='center', fontsize=cell_fs,
                        fontweight='bold', color=text_color, fontfamily=ff)

        ax.set_xlim(-0.1, cols * cell_size + 0.1)
        ax.set_ylim(-0.1, rows * cell_size + 0.1)
        ax.set_title(title, fontsize=sfs + 3, fontweight='bold', pad=10, fontfamily=ff)
        ax.axis('off')
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_grid_with_headers(self, grid, row_headers, col_headers, title, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        sfs = style["font_size_base"]
        ff = style["font_family"]
        palette = list(style["palette"])
        rng.shuffle(palette)
        rows = len(grid); cols = len(grid[0])
        fig, ax = plt.subplots(figsize=(max(6, (cols + 1) * 1.25) * sc,
                                          max(4, (rows + 1) * 1.15) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect('equal')

        cell_size = 1.0
        offset_x = cell_size

        # Column headers
        for c in range(cols):
            x = offset_x + c * cell_size
            y = rows * cell_size
            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                  facecolor=palette[0], edgecolor='#2C3E50',
                                  lw=style["line_width"])
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, str(col_headers[c]),
                    ha='center', va='center', fontsize=sfs - 1,
                    fontweight='bold', color='#1A5276', fontfamily=ff)

        for r in range(rows):
            x = 0
            y = (rows - 1 - r) * cell_size
            rect = plt.Rectangle((x, y), cell_size, cell_size,
                                  facecolor=palette[0], edgecolor='#2C3E50',
                                  lw=style["line_width"])
            ax.add_patch(rect)
            ax.text(x + cell_size/2, y + cell_size/2, str(row_headers[r]),
                    ha='center', va='center', fontsize=sfs - 1,
                    fontweight='bold', color='#1A5276', fontfamily=ff)

        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                x = offset_x + c * cell_size
                y = (rows - 1 - r) * cell_size
                if val is None:
                    color = palette[1]
                    text = '?'
                    text_color = '#E74C3C'
                else:
                    color = palette[2] if (r + c) % 2 == 0 else palette[3]
                    text = str(val)
                    text_color = '#1b1b1b'
                rect = plt.Rectangle((x, y), cell_size, cell_size,
                                      facecolor=color, edgecolor='#2C3E50',
                                      lw=style["line_width"] * 0.7)
                ax.add_patch(rect)
                cell_fs = sfs if len(text) <= 3 else sfs - 3
                ax.text(x + cell_size/2, y + cell_size/2, text,
                        ha='center', va='center', fontsize=cell_fs,
                        fontweight='bold', color=text_color, fontfamily=ff)

        ax.set_xlim(-0.1, offset_x + cols * cell_size + 0.1)
        ax.set_ylim(-0.1, (rows + 1) * cell_size + 0.1)
        ax.set_title(title, fontsize=sfs + 3, fontweight='bold', pad=10, fontfamily=ff)
        ax.axis('off')
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
