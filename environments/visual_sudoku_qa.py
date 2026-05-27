"""Visual sudoku QA — shapes/colors in grid with row/col constraints."""
import random
from typing import Dict, List, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv


# Structured 4-section templates per question type. The Output Format and
# Game Rules differ across types so we keep separate template families.
_TEMPLATES_IDENTIFY = [
    "Identify which symbol belongs in the highlighted cell of the {n}x{n} grid.\n\n"
    "### Game Rules:\n"
    "1. The grid is a {n}x{n} Latin square: each shape from the legend appears exactly once per row.\n"
    "2. Each shape from the legend appears exactly once per column.\n"
    "3. The cell marked `?` is currently hidden; deduce it from the visible cells.\n\n"
    "### Coordinate System:\n"
    "- Rows numbered 1..{n} top-to-bottom; columns 1..{n} left-to-right.\n"
    "- Visible cells in the image show their assigned shape; greyed-out cells are hidden.\n\n"
    "### Current Puzzle State:\n"
    "- Visible-shape grid (`?` = hidden, `_` = also hidden, no value):\n"
    "{state}\n"
    "- Legend (allowed shapes): {legend}\n"
    "- Target cell: Row {target_row}, Column {target_col}\n\n"
    "### Output Format:\n"
    "Output the shape name (one of: {legend}) for the target cell inside <answer>...</answer>.\n"
    "Example: <answer>{example}</answer>",

    "Determine the shape that belongs in the cell marked `?` of the {n}x{n} Latin-square grid.\n\n"
    "### Game Rules:\n"
    "- Every shape in the legend appears exactly once in each row and once in each column.\n"
    "- The hidden cell is uniquely determined by the visible cells.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n} top-to-bottom; columns 1..{n} left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target: (Row {target_row}, Column {target_col})\n\n"
    "### Output Format:\n"
    "Provide the shape name inside <answer>...</answer>.",

    "Solve the {n}x{n} visual Latin-square puzzle below and report the shape at the target cell.\n\n"
    "### Game Rules:\n"
    "Each shape in the legend appears once per row and once per column. Use the visible cells to deduce the hidden one marked `?`.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Allowed shapes: {legend}\n"
    "- Target: Row {target_row}, Column {target_col}\n\n"
    "### Output Format:\n"
    "Output the shape name inside <answer>...</answer>.",
]

_TEMPLATES_COUNT = [
    "Count the visible occurrences of one symbol in the {n}x{n} grid.\n\n"
    "### Game Rules:\n"
    "1. Some cells are hidden (greyed out, no symbol shown); only count cells whose symbol is currently visible.\n"
    "2. Hidden cells are not counted, even if the symbol would appear there in the full solution.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n} top-to-bottom; columns 1..{n} left-to-right.\n"
    "- Hidden cells render as a blank grey tile.\n\n"
    "### Current Puzzle State:\n"
    "- Visible-shape grid (`_` = hidden cell):\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape to count: {target_shape}\n\n"
    "### Output Format:\n"
    "Output a single non-negative integer (the visible count) inside <answer>...</answer>.\n"
    "Example: <answer>2</answer>",

    "Count how many times one shape currently appears (visibly) in the {n}x{n} grid.\n\n"
    "### Game Rules:\n"
    "- Only visible (non-hidden) cells contribute to the count.\n"
    "- Greyed-out cells are hidden and not counted.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Count visible occurrences of: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the integer count inside <answer>...</answer>.",

    "How many cells currently show a particular symbol in the {n}x{n} grid?\n\n"
    "### Game Rules:\n"
    "Greyed-out cells are hidden; do not count them.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Symbol of interest: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the integer count inside <answer>...</answer>.",
]

_TEMPLATES_ROW_MISSING = [
    "Identify which row of the {n}x{n} grid currently has no visible occurrence of one shape.\n\n"
    "### Game Rules:\n"
    "1. The grid is a {n}x{n} Latin square; in the complete solution every shape appears once per row.\n"
    "2. Some cells are hidden — only one row is missing the target shape among visible cells.\n\n"
    "### Coordinate System:\n"
    "- Rows numbered 1..{n} top-to-bottom; columns 1..{n} left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "- Visible-shape grid (`_` = hidden cell):\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the row number (1..{n}) as an integer inside <answer>...</answer>.\n"
    "Example: <answer>2</answer>",

    "Find the unique row of the {n}x{n} grid that currently shows no visible instance of one shape.\n\n"
    "### Game Rules:\n"
    "- Each shape appears once per row in the complete solution.\n"
    "- Exactly one row is missing the target shape among its visible cells.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the row number inside <answer>...</answer>.",

    "Which row of the {n}x{n} grid is currently missing the target shape?\n\n"
    "### Game Rules:\n"
    "Each shape appears once per row in the full solution; exactly one row currently has no visible copy of the target shape.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the row number (1..{n}) inside <answer>...</answer>.",
]

_TEMPLATES_COL_MISSING = [
    "Identify which column of the {n}x{n} grid currently has no visible occurrence of one shape.\n\n"
    "### Game Rules:\n"
    "1. The grid is a {n}x{n} Latin square; every shape appears once per column in the full solution.\n"
    "2. Exactly one column is currently missing the target shape among its visible cells.\n\n"
    "### Coordinate System:\n"
    "- Rows numbered 1..{n} top-to-bottom; columns 1..{n} left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "- Visible-shape grid (`_` = hidden cell):\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the column number (1..{n}) as an integer inside <answer>...</answer>.\n"
    "Example: <answer>3</answer>",

    "Find the unique column of the {n}x{n} grid that has no visible copy of one shape.\n\n"
    "### Game Rules:\n"
    "- Each shape appears once per column in the complete solution.\n"
    "- Exactly one column is missing the target shape among its visible cells.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the column number inside <answer>...</answer>.",

    "Which column of the {n}x{n} grid currently lacks the target shape?\n\n"
    "### Game Rules:\n"
    "Each shape appears once per column in the full solution; exactly one column has no visible copy of the target shape.\n\n"
    "### Coordinate System:\n"
    "- Rows 1..{n}; columns 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "- Legend: {legend}\n"
    "- Target shape: {target_shape}\n\n"
    "### Output Format:\n"
    "Output the column number (1..{n}) inside <answer>...</answer>.",
]


class VisualSudokuQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "visual_sudoku"

    _SHAPE_SETS = [
        # Set 0: geometric shapes
        {"names": ["circle", "square", "triangle", "diamond", "star"],
         "colors": ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]},
        # Set 1: colors only (colored squares)
        {"names": ["red", "blue", "green", "yellow", "purple"],
         "colors": ["#e74c3c", "#3498db", "#27ae60", "#f1c40f", "#9b59b6"]},
        # Set 2: symbols (rendered as text)
        {"names": ["star", "heart", "moon", "sun", "diamond"],
         "colors": ["#9b59b6", "#e74c3c", "#2c3e50", "#f39c12", "#3498db"]},
    ]

    _QUESTION_TYPES = [
        "identify_shape",     # What shape goes in the ? cell
        "count_shape",        # How many circles are visible in the grid
        "which_row_missing",  # Which row is missing a triangle
        "which_col_missing",  # Which column is missing a given shape
    ]

    @staticmethod
    def _format_state(grid, removed, target_cell, n, shapes) -> str:
        """Render the visible-shape grid as ASCII rows: each cell is either
        the shape name (visible), `?` (target — only for identify_shape), or
        `_` (otherwise hidden). Aligned column widths."""
        max_w = max(len(s) for s in shapes + ["_", "?"])
        rows = []
        for r in range(n):
            cells = []
            for c in range(n):
                if target_cell == (r, c):
                    cells.append("?".rjust(max_w))
                elif (r, c) in removed:
                    cells.append("_".rjust(max_w))
                else:
                    cells.append(shapes[grid[r][c]].rjust(max_w))
            rows.append("  " + " | ".join(cells))
        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 1:
            return {"grid_size": 3, "qtypes": ["identify_shape", "count_shape"],
                    "remove_frac": 0.3}
        if level <= 3:
            return {"grid_size": 4, "qtypes": ["identify_shape", "count_shape"],
                    "remove_frac": 0.4}
        if level <= 5:
            return {"grid_size": 4, "qtypes": ["identify_shape", "count_shape",
                                                "which_row_missing"],
                    "remove_frac": 0.5}
        if level <= 7:
            return {"grid_size": 5, "qtypes": self._QUESTION_TYPES,
                    "remove_frac": 0.5}
        return {"grid_size": 5, "qtypes": self._QUESTION_TYPES,
                "remove_frac": 0.6}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        n = parameter.get("grid_size", cfg["grid_size"])
        q_type = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        n_remove = parameter.get("n_remove", max(1, int(n * n * cfg["remove_frac"])))

        shape_set = rng.choice(self._SHAPE_SETS)
        shapes = shape_set["names"][:n]
        colors = shape_set["colors"][:n]

        # Generate a valid Latin square (each symbol once per row/col)
        grid = [[None]*n for _ in range(n)]
        # Shift pattern construction
        for r in range(n):
            for c in range(n):
                grid[r][c] = (r + c) % n

        # Shuffle to randomize
        perm_r = list(range(n))
        perm_c = list(range(n))
        rng.shuffle(perm_r)
        rng.shuffle(perm_c)
        grid = [[grid[perm_r[r]][perm_c[c]] for c in range(n)] for r in range(n)]

        # Also shuffle symbol mapping
        sym_perm = list(range(n))
        rng.shuffle(sym_perm)
        grid = [[sym_perm[grid[r][c]] for c in range(n)] for r in range(n)]

        # Remove cells
        cells = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(cells)
        removed = set(cells[:min(n_remove, n * n - n)])  # Keep at least n cells visible

        legend_str = ", ".join(shapes)

        def _identify_shape_question(target_r, target_c, answer):
            state_str = self._format_state(grid, removed, (target_r, target_c),
                                            n, shapes)
            sidx = (self.seed or 0) % len(_TEMPLATES_IDENTIFY)
            return _TEMPLATES_IDENTIFY[sidx].format(
                n=n,
                state=state_str,
                legend=legend_str,
                target_row=target_r + 1,
                target_col=target_c + 1,
                example=shapes[0],
            )

        def _count_question(target_shape_idx):
            state_str = self._format_state(grid, removed, None, n, shapes)
            sidx = (self.seed or 0) % len(_TEMPLATES_COUNT)
            return _TEMPLATES_COUNT[sidx].format(
                n=n,
                state=state_str,
                legend=legend_str,
                target_shape=shapes[target_shape_idx],
            )

        def _row_missing_question(target_shape_idx):
            state_str = self._format_state(grid, removed, None, n, shapes)
            sidx = (self.seed or 0) % len(_TEMPLATES_ROW_MISSING)
            return _TEMPLATES_ROW_MISSING[sidx].format(
                n=n,
                state=state_str,
                legend=legend_str,
                target_shape=shapes[target_shape_idx],
            )

        def _col_missing_question(target_shape_idx):
            state_str = self._format_state(grid, removed, None, n, shapes)
            sidx = (self.seed or 0) % len(_TEMPLATES_COL_MISSING)
            return _TEMPLATES_COL_MISSING[sidx].format(
                n=n,
                state=state_str,
                legend=legend_str,
                target_shape=shapes[target_shape_idx],
            )

        if q_type == "identify_shape":
            # Pick one removed cell as question target
            if not removed:
                return None
            target_r, target_c = rng.choice(list(removed))
            answer_idx = grid[target_r][target_c]
            answer = shapes[answer_idx]
            question = _identify_shape_question(target_r, target_c, answer)

        elif q_type == "count_shape":
            # Count visible instances of a shape
            target_shape_idx = rng.randint(0, n - 1)
            count = sum(1 for r in range(n) for c in range(n)
                       if (r, c) not in removed and grid[r][c] == target_shape_idx)
            answer = str(count)
            # No target cell for this question type
            target_r, target_c = -1, -1
            question = _count_question(target_shape_idx)

        elif q_type == "which_row_missing":
            # Which row has no visible instance of a given shape.
            # Pick a shape where EXACTLY ONE row is missing it (else ambiguous).
            uniq_target = None
            uniq_row = None
            shuffled = list(range(n))
            rng.shuffle(shuffled)
            for cand_idx in shuffled:
                missing_rows = []
                for r in range(n):
                    visible_in_row = any(grid[r][c] == cand_idx and (r, c) not in removed
                                        for c in range(n))
                    if not visible_in_row:
                        missing_rows.append(r)
                if len(missing_rows) == 1:
                    uniq_target = cand_idx
                    uniq_row = missing_rows[0]
                    break
            if uniq_target is None:
                # Fall back to identify_shape
                target_r, target_c = rng.choice(list(removed)) if removed else (0, 0)
                answer = shapes[grid[target_r][target_c]]
                question = _identify_shape_question(target_r, target_c, answer)
                q_type = "identify_shape"
            else:
                target_shape_idx = uniq_target
                answer = str(uniq_row + 1)
                target_r, target_c = -1, -1
                question = _row_missing_question(target_shape_idx)

        elif q_type == "which_col_missing":
            # Which column has no visible instance of a given shape.
            # Pick a shape where EXACTLY ONE column is missing it (else ambiguous).
            uniq_target = None
            uniq_col = None
            shuffled = list(range(n))
            rng.shuffle(shuffled)
            for cand_idx in shuffled:
                missing_cols = []
                for c in range(n):
                    visible_in_col = any(grid[r][c] == cand_idx and (r, c) not in removed
                                        for r in range(n))
                    if not visible_in_col:
                        missing_cols.append(c)
                if len(missing_cols) == 1:
                    uniq_target = cand_idx
                    uniq_col = missing_cols[0]
                    break
            if uniq_target is None:
                # Fall back to identify_shape
                target_r, target_c = rng.choice(list(removed)) if removed else (0, 0)
                answer = shapes[grid[target_r][target_c]]
                question = _identify_shape_question(target_r, target_c, answer)
                q_type = "identify_shape"
            else:
                target_shape_idx = uniq_target
                answer = str(uniq_col + 1)
                target_r, target_c = -1, -1
                question = _col_missing_question(target_shape_idx)

        else:
            return None

        # Render
        style = self._random_style()
        fig_size = n + 2
        fig, ax = plt.subplots(figsize=(fig_size * style["figsize_scale"],
                                        fig_size * style["figsize_scale"]))
        fig.patch.set_facecolor(style["bg_color"])

        for r in range(n):
            for c in range(n):
                y_pos = n - 1 - r  # flip so row 1 is top

                if q_type == "identify_shape" and (r, c) == (target_r, target_c):
                    bg = "#fff3cd"
                    rect = mpatches.Rectangle((c, y_pos), 1, 1, facecolor=bg,
                                             edgecolor="black", linewidth=1.5)
                    ax.add_patch(rect)
                    ax.text(c + 0.5, y_pos + 0.5, "?", ha="center", va="center",
                           fontsize=style["font_size_base"] + 12, color="red",
                           fontweight="bold")
                elif (r, c) in removed:
                    bg = "#f0f0f0"
                    rect = mpatches.Rectangle((c, y_pos), 1, 1, facecolor=bg,
                                             edgecolor="#bbb", linewidth=1)
                    ax.add_patch(rect)
                else:
                    bg = "#ffffff"
                    rect = mpatches.Rectangle((c, y_pos), 1, 1, facecolor=bg,
                                             edgecolor="black", linewidth=1.5)
                    ax.add_patch(rect)
                    sym_idx = grid[r][c]
                    self._draw_shape(ax, c + 0.5, y_pos + 0.5,
                                    shapes[sym_idx], colors[sym_idx], 0.3, style)

        # Row/col labels
        for i in range(n):
            ax.text(-0.4, n - 0.5 - i, f"R{i+1}", ha="center", va="center",
                   fontsize=style["font_size_base"], fontfamily=style["font_family"])
            ax.text(i + 0.5, n + 0.35, f"C{i+1}", ha="center", va="center",
                   fontsize=style["font_size_base"], fontfamily=style["font_family"])

        # Legend
        legend_x = n + 0.5
        ax.text(legend_x + 0.3, n - 0.1, "Shapes:", fontsize=style["font_size_base"],
               fontweight="bold", fontfamily=style["font_family"])
        for i in range(min(n, len(shapes))):
            self._draw_shape(ax, legend_x + 0.3, n - 0.7 - i * 0.7,
                           shapes[i], colors[i], 0.2, style)
            ax.text(legend_x + 0.7, n - 0.7 - i * 0.7, shapes[i],
                   fontsize=style["font_size_base"] - 1, va="center",
                   fontfamily=style["font_family"])

        ax.set_xlim(-0.6, n + 2)
        ax.set_ylim(-0.5, n + 0.7)
        ax.set_aspect("equal")
        ax.axis("off")

        return question, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_shape(self, ax, x, y, shape, color, size, style=None):
        lw = style.get("line_width", 1.5) if style else 1.5

        if shape in ("circle", "red", "blue", "green", "yellow", "purple"):
            if shape in ("red", "blue", "green", "yellow", "purple"):
                # Color squares
                rect = mpatches.Rectangle((x - size, y - size), size * 2, size * 2,
                                         facecolor=color, edgecolor="black", linewidth=lw)
                ax.add_patch(rect)
            else:
                circle = plt.Circle((x, y), size, facecolor=color,
                                   edgecolor="black", linewidth=lw)
                ax.add_patch(circle)
        elif shape == "square":
            rect = mpatches.Rectangle((x - size, y - size), size * 2, size * 2,
                                     facecolor=color, edgecolor="black", linewidth=lw)
            ax.add_patch(rect)
        elif shape == "triangle":
            tri = plt.Polygon([(x, y + size), (x - size, y - size), (x + size, y - size)],
                             facecolor=color, edgecolor="black", linewidth=lw)
            ax.add_patch(tri)
        elif shape == "diamond":
            diamond = plt.Polygon([(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
                                 facecolor=color, edgecolor="black", linewidth=lw)
            ax.add_patch(diamond)
        elif shape == "star":
            ax.text(x, y, "\u2605", ha="center", va="center",
                   fontsize=size * 60, color=color)
        elif shape == "heart":
            ax.text(x, y, "\u2665", ha="center", va="center",
                   fontsize=size * 60, color=color)
        elif shape == "moon":
            ax.text(x, y, "\u263D", ha="center", va="center",
                   fontsize=size * 60, color=color)
        elif shape == "sun":
            ax.text(x, y, "\u2600", ha="center", va="center",
                   fontsize=size * 60, color=color)
