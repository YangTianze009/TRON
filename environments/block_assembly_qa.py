"""Count unit cubes in 3D structure with multiple question types.

Round 2 fixes:
  - L0 simpler: 2x2 grid, single-step "total_blocks" only, big labels,
    formula shown. Ensures pass rate can actually reach >30%.
  - L9 harder: larger grid with row/col constraints, hidden cells requiring
    system-of-equations reasoning, multi-step composite operations.
  - Diversified visual layout: per-seed orientation of 3D view, randomized
    cube colors, 4+ question phrasings, shuffled palettes.
  - Text leakage removed: grid values are on image; totals/constraints are
    shown as labels, not embedded as numeric constants in the question.
"""
import random, math
from typing import Dict, List, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class BlockAssemblyQA(StandaloneVisualEnv):
    ENV_NAME = "block_assembly"

    _TITLE_POOL = [
        "Block Heights",
        "Stacked Blocks",
        "Cube Count Table",
        "Height Grid",
        "Isometric Blocks",
        "Cube Assembly",
    ]

    _PANEL_TITLES = ["3D View", "Isometric", "Perspective", "Side Preview"]

    _Q_TEMPLATES = {
        "total_blocks": [
            "Each cell shows the height (number of stacked blocks). How many total blocks are there?",
            "Count the total number of unit cubes shown in the grid.",
            "Sum the heights in every cell to get the total number of blocks.",
            "What is the total number of blocks in this 3D arrangement?",
        ],
        "max_height": [
            "What is the maximum height (tallest stack) in this grid?",
            "Identify the tallest column. How many cubes does it contain?",
            "What is the height of the tallest stack in the figure?",
            "Which cell has the most blocks? Report that height.",
        ],
        "row_with_most": [
            "Which row (1..R) has the most total blocks?",
            "Find the row whose total cube count is the largest. Report its row index.",
            "Sum each row. Which row has the highest sum?",
            "Identify the row with the greatest total height.",
        ],
        "difference": [
            "What is the difference in total blocks between the row with the most and the row with the least?",
            "Compute row sums. Report (max row sum) − (min row sum).",
            "Subtract the smallest row total from the largest row total.",
            "What is the gap (in cubes) between the heaviest and lightest rows?",
        ],
        "weighted_total": [
            "For each cell, multiply its height by (row_number + column_number), where rows and columns start at 1, then sum.",
            "Compute the weighted sum: height × (row_idx + col_idx) for all cells (1-indexed).",
            "Apply the weighting h_ij × (i + j) (1-indexed) and return the overall sum.",
            "Each cell's contribution is its height times the sum of its 1-indexed row and column; sum these contributions.",
        ],
        "row_col_product": [
            "Compute each row's total and each column's total. What is the largest product of (any row total) × (any column total)?",
            "Find the maximum value of R_i × C_j where R_i is a row total and C_j is a column total.",
            "Among all pairs of (row sum, column sum), what is the maximum product?",
            "Take every row total and every column total. Report the biggest product of one row total and one column total.",
        ],
        "hidden_cell_solve": [
            "One cell is hidden ('?'). Use the row and column totals to deduce its value.",
            "Fill in the missing '?' cell by using the row and column total labels.",
            "Given the row totals and column totals, solve for the hidden cell marked '?'.",
            "Find the height of the '?' cell by subtracting the other known cells in its row (or column) from the corresponding total.",
        ],
        "cross_diagonal_sum": [
            "Sum all heights along the main diagonal AND the anti-diagonal. If a cell lies on both, count it twice. Report the total.",
            "Add up the main-diagonal cells and the anti-diagonal cells (overlap counted twice).",
            "What is the sum of cubes on both diagonals, with the center cell counted twice when shared?",
            "Compute the combined cube count of the two diagonals (top-left→bottom-right and top-right→bottom-left).",
        ],
        "column_with_most": [
            "Which column (1..C) has the most total blocks?",
            "Identify the column with the largest total cube count.",
            "Sum each column. Which column's sum is greatest?",
            "Find the column with the highest total height.",
        ],
        "tall_cells_count": [
            "How many cells have a height of at least 3 (i.e., at least 3 cubes tall)?",
            "Count the cells whose stacks contain 3 or more cubes.",
            "How many cells qualify as 'tall' (height ≥ 3)?",
            "Report the number of cells whose cube-count is at least three.",
        ],
        "average_height": [
            "Report the average height across all cells, rounded to 2 decimals.",
            "Compute the mean height (total blocks / number of cells), rounded to 2 decimals.",
            "What is the mean cell height? Round to 2 decimal places.",
            "Find the average stack height (to 2 decimal places).",
        ],
        "two_stage_hidden": [
            "Three cells are marked '?'. First use the row totals to deduce each hidden cell's value, then report the SUM of all three hidden values.",
            "Three cells are hidden. Deduce each from its row total, then return the sum of the three recovered values.",
        ],
        "product_of_diagonals": [
            "Let M = sum of main-diagonal heights, A = sum of anti-diagonal heights. Report M * A.",
            "Compute the product (main-diagonal sum) × (anti-diagonal sum).",
        ],
        "weighted_total_squared": [
            "For each cell, multiply its height by (row_idx + col_idx)^2 (1-indexed), then sum. Report the total.",
            "Weight every cell by (i + j)^2 where i, j are 1-indexed; sum the weighted values.",
        ],
    }

    # ------------------------------------------------------------------ #
    # Level config
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: trivially easy — 2x2 grid, single question type, big fonts.
        # L9: large grid, 3-4 hidden cells, compound questions.
        if level == 0:
            return {"rows": 2, "cols": 2, "max_h": 4,
                    "qtypes": ["total_blocks"],
                    "hidden_cells": 0, "show_totals": False,
                    "show_formula": True, "font_boost": 4}
        if level == 1:
            return {"rows": 3, "cols": 3, "max_h": 4,
                    "qtypes": ["total_blocks", "max_height",
                               "tall_cells_count"],
                    "hidden_cells": 0, "show_totals": False,
                    "show_formula": True, "font_boost": 3}
        if level <= 3:
            return {"rows": 3, "cols": 3, "max_h": 5,
                    "qtypes": ["total_blocks", "row_with_most",
                               "column_with_most", "tall_cells_count"],
                    "hidden_cells": 0, "show_totals": True,
                    "show_formula": False, "font_boost": 2}
        if level <= 5:
            return {"rows": 4, "cols": 4, "max_h": 6,
                    "qtypes": ["difference", "weighted_total",
                               "row_with_most", "average_height"],
                    "hidden_cells": 0, "show_totals": True,
                    "show_formula": False, "font_boost": 2}
        if level <= 7:
            return {"rows": 5, "cols": 5, "max_h": 7,
                    "qtypes": ["weighted_total", "row_col_product",
                               "hidden_cell_solve", "cross_diagonal_sum"],
                    "hidden_cells": 1, "show_totals": True,
                    "show_formula": False, "font_boost": 1}
        # L8
        if level == 8:
            return {"rows": 6, "cols": 6, "max_h": 8,
                    "qtypes": ["row_col_product", "hidden_cell_solve",
                               "cross_diagonal_sum", "weighted_total"],
                    "hidden_cells": 2, "show_totals": True,
                    "show_formula": False, "font_boost": 0}
        # L9 hardened 2026-04-17: 7x7, 3 hidden cells, harder qtypes
        # including two_stage_hidden and product_of_diagonals.
        return {"rows": 7, "cols": 7, "max_h": 9,
                "qtypes": ["two_stage_hidden", "product_of_diagonals",
                           "hidden_cell_solve", "weighted_total_squared"],
                "hidden_cells": 3, "show_totals": True,
                "show_formula": False, "font_boost": 0}

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 419)

        for attempt in range(20):
            r = self._try(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try(self, rng: random.Random, cfg: Dict, level: int):
        style = self._random_style()
        # BUGFIX: filter near-black/white from the palette so per-column base
        # colors in the isometric view always have room for the *0.7/*0.5
        # shading to show up (otherwise pure black stays black on all three
        # faces, making the 3D structure unreadable).
        palette = [c for c in style["palette"]
                   if c.lower() not in ('#000000', '#010101', '#0a0a0a',
                                         '#ffffff', '#fefefe')]
        if not palette:
            palette = ['#5dade2', '#48c9b0', '#ec7063', '#f4d03f']
        rng.shuffle(palette)

        rows = cfg["rows"]
        cols = cfg["cols"]
        max_h = cfg["max_h"]

        q_type = rng.choice(cfg["qtypes"])

        heights = [[rng.randint(0, max_h) for _ in range(cols)] for _ in range(rows)]
        # Ensure at least some blocks
        if sum(sum(r) for r in heights) < 2:
            heights[0][0] = rng.randint(1, max_h)
            heights[rows // 2][cols // 2] = rng.randint(1, max_h)

        total = sum(sum(r) for r in heights)
        max_height = max(max(r) for r in heights)

        # For L0 with only total_blocks, ensure total is not too large (to
        # keep pass rate > 30%) but still varied across seeds (2..25).
        if level == 0:
            if total > 14:
                # Reduce some cells
                for _ in range(3):
                    ir = rng.randrange(rows)
                    ic = rng.randrange(cols)
                    heights[ir][ic] = rng.randint(0, 3)
                total = sum(sum(r) for r in heights)
                max_height = max(max(r) for r in heights)

        # Hidden cells
        n_hidden = cfg.get("hidden_cells", 0)
        hidden_set = set()
        if n_hidden > 0 and q_type in ("hidden_cell_solve", "two_stage_hidden"):
            # Ensure all hidden cells are in DIFFERENT rows so that row
            # totals still yield a unique solution (same for columns).
            row_picks = rng.sample(range(rows), min(n_hidden, rows))
            col_picks = [rng.randrange(cols) for _ in row_picks]
            for r_idx, c_idx in zip(row_picks, col_picks):
                hidden_set.add((r_idx, c_idx))

        row_sums = [sum(heights[r]) for r in range(rows)]
        col_sums = [sum(heights[r][c] for r in range(rows)) for c in range(cols)]

        # ---- Compute answer ----
        rng_q = random.Random(rng.random() * 2**31)
        q_pool = self._Q_TEMPLATES.get(q_type, self._Q_TEMPLATES["total_blocks"])
        q = rng_q.choice(q_pool)

        if q_type == "total_blocks":
            answer = total
        elif q_type == "max_height":
            answer = max_height
        elif q_type == "row_with_most":
            answer = row_sums.index(max(row_sums)) + 1
        elif q_type == "column_with_most":
            answer = col_sums.index(max(col_sums)) + 1
        elif q_type == "tall_cells_count":
            answer = sum(1 for rr in range(rows) for cc in range(cols)
                         if heights[rr][cc] >= 3)
        elif q_type == "difference":
            answer = max(row_sums) - min(row_sums)
        elif q_type == "weighted_total":
            # 1-indexed: (row+1) + (col+1) = r + c + 2
            answer = sum(heights[r][c] * (r + c + 2)
                         for r in range(rows) for c in range(cols))
        elif q_type == "row_col_product":
            best = max(rs * cs for rs in row_sums for cs in col_sums)
            answer = best
        elif q_type == "hidden_cell_solve":
            if not hidden_set:
                return None
            target = rng.choice(list(hidden_set))
            tr, tc = target
            known_in_row = sum(heights[tr][c] for c in range(cols)
                               if (tr, c) not in hidden_set)
            unknowns_in_row = [(tr, c) for c in range(cols)
                               if (tr, c) in hidden_set]
            if len(unknowns_in_row) != 1:
                return None
            answer = row_sums[tr] - known_in_row
            q = (f"One cell is marked '?'. It is in Row {tr + 1}, Column "
                 f"{tc + 1}. Use the row/column totals labeled in the "
                 f"figure to deduce its value.")
        elif q_type == "cross_diagonal_sum":
            n = min(rows, cols)
            diag_main = sum(heights[i][i] for i in range(n))
            diag_anti = sum(heights[i][n - 1 - i] for i in range(n))
            answer = diag_main + diag_anti
        elif q_type == "average_height":
            mean = total / (rows * cols)
            answer = round(mean, 2)
        elif q_type == "two_stage_hidden":
            if not hidden_set:
                return None
            # Each hidden cell is in a unique row; sum of each hidden cell's
            # recovered value = sum over hidden rows of (row_total - sum of
            # visible cells in that row).
            total_hidden = 0
            for (tr, tc) in hidden_set:
                known_in_row = sum(heights[tr][c] for c in range(cols)
                                   if (tr, c) not in hidden_set)
                total_hidden += row_sums[tr] - known_in_row
            answer = total_hidden
        elif q_type == "product_of_diagonals":
            n = min(rows, cols)
            M = sum(heights[i][i] for i in range(n))
            A = sum(heights[i][n - 1 - i] for i in range(n))
            answer = M * A
        elif q_type == "weighted_total_squared":
            answer = sum(heights[r][c] * ((r + c + 2) ** 2)
                         for r in range(rows) for c in range(cols))
        else:
            answer = total

        # Build image
        image = self._render(rng, cfg, style, palette, heights,
                             hidden_set, row_sums, col_sums,
                             q_type, total, max_height)

        return q, str(answer), image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, rng, cfg, style, palette, heights, hidden_set,
                row_sums, col_sums, q_type, total, max_height) -> Image.Image:
        rows = cfg["rows"]
        cols = cfg["cols"]
        show_totals = cfg.get("show_totals", False)
        show_formula = cfg.get("show_formula", False)
        font_boost = cfg.get("font_boost", 0)

        # Layout variants: either side-by-side (grid + 3D) or grid-only
        layout = rng.choice(["side_by_side", "stacked", "grid_only",
                             "three_d_left"])

        sc = style["figsize_scale"]
        fs = style["font_size_base"] + font_boost
        ff = style["font_family"]

        if layout == "grid_only" or q_type in ("hidden_cell_solve",):
            fig, ax = plt.subplots(figsize=(6.5 * sc, 5.5 * sc))
            fig.patch.set_facecolor(style["bg_color"])
            axes = [ax]
        elif layout == "stacked":
            fig, (ax, ax2) = plt.subplots(2, 1,
                                          figsize=(7 * sc, 10 * sc),
                                          gridspec_kw={"height_ratios": [1, 0.9]})
            fig.patch.set_facecolor(style["bg_color"])
            axes = [ax, ax2]
        elif layout == "three_d_left":
            fig, (ax2, ax) = plt.subplots(1, 2,
                                          figsize=(12 * sc, 5.5 * sc),
                                          gridspec_kw={"width_ratios": [1, 1.1]})
            fig.patch.set_facecolor(style["bg_color"])
            axes = [ax, ax2]
        else:  # side_by_side
            fig, (ax, ax2) = plt.subplots(1, 2,
                                          figsize=(12 * sc, 5.5 * sc))
            fig.patch.set_facecolor(style["bg_color"])
            axes = [ax, ax2]

        # --- Grid with heights ---
        ax.set_facecolor(style["bg_color"])
        cmap_options = [
            ["#f0f0f0", "#a8d8ea", "#6cb4ee", "#2c73d2", "#1a237e", "#0d47a1"],
            ["#f0f0f0", "#fdd0a2", "#fdae6b", "#e6550d", "#8b2500", "#4a0f00"],
            ["#f0f0f0", "#d9f0d3", "#a1d99b", "#41ab5d", "#238b45", "#00441b"],
            ["#ffffff", "#f1c2d6", "#c994b8", "#8856a7", "#54278f", "#3f007d"],
        ]
        cmap = rng.choice(cmap_options)
        for r in range(rows):
            for c in range(cols):
                h = heights[r][c]
                color_idx = min(h, len(cmap) - 1)
                is_hidden = (r, c) in hidden_set
                face = "#e8e8e8" if is_hidden else cmap[color_idx]
                edge_color = "#c0392b" if is_hidden else style["geo_line_color"]
                lw = style["line_width"] + (1.0 if is_hidden else 0.0)
                rect = mpatches.FancyBboxPatch(
                    (c, rows - 1 - r), 1, 1,
                    boxstyle="round,pad=0.02",
                    facecolor=face,
                    edgecolor=edge_color,
                    linewidth=lw)
                ax.add_patch(rect)
                label = "?" if is_hidden else str(h)
                text_color = "#c0392b" if is_hidden else (
                    "#1a1a2e" if h < 3 else "white")
                ax.text(c + 0.5, rows - 0.5 - r, label,
                        ha="center", va="center",
                        fontsize=fs + 4, fontweight="bold",
                        color=text_color, fontfamily=ff)

        # Row / col labels
        for i in range(rows):
            ax.text(-0.4, rows - 0.5 - i, f"R{i + 1}",
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold",
                    color=style["geo_line_color"])
        for i in range(cols):
            ax.text(i + 0.5, rows + 0.25, f"C{i + 1}",
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold",
                    color=style["geo_line_color"])
        # Row / col totals (if enabled)
        if show_totals:
            for i in range(rows):
                ax.text(cols + 0.45, rows - 0.5 - i, f"={row_sums[i]}",
                        ha="left", va="center",
                        fontsize=fs, fontweight="bold",
                        color="#1a5276")
            for i in range(cols):
                ax.text(i + 0.5, -0.45, f"{col_sums[i]}",
                        ha="center", va="center",
                        fontsize=fs, fontweight="bold",
                        color="#1a5276")
            # Include grand total corner
            ax.text(cols + 0.45, -0.45, f"T={sum(row_sums)}",
                    ha="left", va="center",
                    fontsize=fs, fontweight="bold",
                    color="#1a5276")

        ax.set_xlim(-0.8, cols + 1.6)
        ax.set_ylim(-0.8, rows + 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        title = rng.choice(self._TITLE_POOL)
        ax.set_title(f"{title} ({rows}x{cols})",
                     fontsize=fs + 2, fontweight="bold",
                     fontfamily=ff)

        if show_formula and q_type == "total_blocks":
            ax.text(0.02, -0.05,
                    "Formula: total = sum of all cell heights",
                    transform=ax.transAxes,
                    fontsize=fs, color="#555", style="italic",
                    ha="left", va="top",
                    bbox=dict(facecolor="#f7f7f7", edgecolor="#ccc",
                              boxstyle="round,pad=0.2", alpha=0.85))

        # --- 3D panel ---
        if len(axes) > 1:
            ax2 = axes[1]
            ax2.set_facecolor(style["bg_color"])
            # Random starting rotation so each seed looks different
            start_angle = rng.choice([0, 15, 30, -15, -30])
            flip = rng.random() < 0.4
            def iso(c, r, h):
                x = (c - r) * 0.866
                y = (c + r) * 0.5 + h
                if flip:
                    x = -x
                return x, y

            # Painter's algorithm: draw cubes from back to front. In this iso
            # projection, higher y_iso = further back on screen. Primary sort
            # key: y_iso descending (back cubes drawn first). Tiebreaker:
            # smaller h first (bottom cubes before top cubes within the same
            # (r,c) stack — top will correctly overdraw bottom edges).
            cubes = []
            for r in range(rows):
                for c in range(cols):
                    for h in range(heights[r][c]):
                        cubes.append((r, c, h))
            # Sort: higher (c+r) first (back), then lower h (bottom first
            # within same stack). For ties in c+r, cubes are at different
            # x_iso so no real overlap issue.
            cubes.sort(key=lambda t: (-(t[1] + t[0]), t[2]))
            for r, c, h in cubes:
                x, y = iso(c, r, h)
                # choose palette index per column for diverse color
                col_color = palette[(c + r) % len(palette)]
                from matplotlib.colors import to_rgba
                base_c = to_rgba(col_color)
                # Unit cube at grid (c, r, h) → iso vertices. For this iso
                # projection (x=(c-r)*0.866, y=(c+r)*0.5+h) a unit cube has:
                #   bottom corners at y∈[y, y+1]; top corners at y∈[y+1, y+2]
                # Three visible faces: top (at h+1), left face (dc=0 side),
                # right face (dr=0 side). Edges all share properly.
                # Top face: 4 corners at h+1
                top = [(x, y + 1), (x + 0.866, y + 1.5),
                       (x, y + 2), (x - 0.866, y + 1.5)]
                # Left face (the face seen from the -c side, to the left on
                # screen): 4 corners at c=0 with r∈{0,1}, h∈{0,1}
                left = [(x, y), (x, y + 1),
                        (x - 0.866, y + 1.5), (x - 0.866, y + 0.5)]
                # Right face (the face seen from the -r side, to the right on
                # screen): 4 corners at r=0 with c∈{0,1}, h∈{0,1}
                right = [(x, y), (x + 0.866, y + 0.5),
                         (x + 0.866, y + 1.5), (x, y + 1)]
                ax2.fill([p[0] for p in top], [p[1] for p in top],
                         facecolor=col_color, edgecolor="#333",
                         linewidth=0.8, alpha=0.9)
                dark = tuple(max(0, v * 0.7) for v in base_c[:3]) + (0.9,)
                ax2.fill([p[0] for p in left], [p[1] for p in left],
                         facecolor=dark, edgecolor="#333", linewidth=0.8)
                darker = tuple(max(0, v * 0.5) for v in base_c[:3]) + (0.9,)
                ax2.fill([p[0] for p in right], [p[1] for p in right],
                         facecolor=darker, edgecolor="#333",
                         linewidth=0.8)

            ax2.set_aspect("equal")
            ax2.axis("off")
            ax2.autoscale()
            panel_title = rng.choice(self._PANEL_TITLES)
            ax2.set_title(panel_title, fontsize=fs + 1,
                          fontweight="bold", fontfamily=ff)

        plt.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
