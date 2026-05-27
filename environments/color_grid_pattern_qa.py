"""
Color Grid Pattern QA (Task C, PuzzleVQA color_grid gap, ).

Colored grid cells with a pattern rule. Ask which cell breaks the pattern
or what the missing cell's color should be.

Difficulty axes:
  A) Grid size (3x3 -> 6x6)
  B) Pattern complexity (row-repeat -> diagonal -> modular arithmetic)
  C) Number of colors (3 -> 8)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

_MISSING_TEMPLATES = [
    "A {gs}x{gs} color grid follows a pattern. One cell (row {r}, column {c}) is marked '?'. What color should replace '?'? Answer with the color name (e.g., 'red').",
    "The {gs}x{gs} grid below shows a color pattern with a missing cell at row {r}, column {c}. What color belongs there? Give the color name.",
    "Examine the {gs}x{gs} color grid; its cell at (row {r}, col {c}) is blank. Which color completes the pattern?",
    "In the {gs}x{gs} colored pattern, cell (r={r}, c={c}) is unknown. Report the color name.",
    "A color rule governs the {gs}x{gs} grid. What color fills the missing cell at row {r}, column {c}?",
    "Given the {gs}x{gs} pattern shown, determine the color of the cell at ({r}, {c}).",
    "The missing cell at row {r}, column {c} of this {gs}x{gs} grid should be what color?",
    "Identify the color that completes the {gs}x{gs} pattern at row {r}, column {c}.",
    "The {gs}x{gs} grid has an unknown color at cell (row {r}, col {c}). Name it.",
    "Looking at the {gs}x{gs} color pattern, state the color at row {r}, column {c}.",
    "What color belongs at (row {r}, column {c}) of the {gs}x{gs} grid? Give the color name.",
    "Deduce the color at row {r}, column {c} of the {gs}x{gs} pattern shown.",
    "The {gs}x{gs} pattern grid is missing one cell at row {r}, column {c}. Fill in the color.",
    "Based on the {gs}x{gs} color pattern, what color should the cell at ({r}, {c}) be?",
    "Examine the color pattern of the {gs}x{gs} grid and name the missing cell's color (row {r}, col {c}).",
    "Report the color at row {r}, column {c} of this {gs}x{gs} color pattern.",
]

_WRONG_TEMPLATES = [
    "A {gs}x{gs} color grid follows a pattern, but one cell has the wrong color. Which cell has the wrong color? Answer as 'row,column' (e.g., '2,3').",
    "The {gs}x{gs} pattern has one incorrect cell. Identify it as 'row,column'.",
    "Find the wrongly-colored cell in the {gs}x{gs} grid. Report it as row,column.",
    "One cell in the {gs}x{gs} color grid violates the pattern. Give its location as 'row,column'.",
    "Which cell in this {gs}x{gs} color pattern breaks the rule? Answer row,column.",
    "Spot the misplaced-color cell in the {gs}x{gs} grid. Report 'row,column'.",
    "Locate the cell with the wrong color in the {gs}x{gs} pattern. Answer row,column.",
    "The {gs}x{gs} grid has exactly one anomaly. Report its row,column.",
    "Identify the cell whose color breaks the {gs}x{gs} pattern. Format: row,column.",
    "Determine the incorrect cell in the {gs}x{gs} color pattern. Give row,column.",
    "Point out the wrong cell in the {gs}x{gs} color grid using 'row,column'.",
    "State the location (row,column) of the inconsistent cell in this {gs}x{gs} pattern.",
    "The {gs}x{gs} grid has an odd-one-out cell. Report it as row,column.",
    "Find the row,column of the cell that does not fit the {gs}x{gs} pattern.",
    "Which coordinates (row,column) is the miscoloured cell in the {gs}x{gs} grid?",
    "Pinpoint the cell that breaks the {gs}x{gs} color pattern; answer row,column.",
]

_COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink"]
_COLOR_HEX = {
    "red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71",
    "yellow": "#f1c40f", "purple": "#9b59b6", "orange": "#e67e22",
    "cyan": "#1abc9c", "pink": "#e91e63",
}

class ColorGridPatternQA(StandaloneVisualEnv):
    ENV_NAME = "color_grid_pattern"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "grid_size": 3 + level // 3,             # 3,3,3,4,4,4,5,5,5,6
            "n_colors": min(8, 3 + level // 2),      # 3,3,4,4,5,5,6,6,7,7
            "pattern": "row_repeat" if level <= 2 else
                      "col_repeat" if level <= 4 else
                      "diagonal" if level <= 6 else "modular",
            "qtype": "find_missing" if level <= 5 else "find_wrong",
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["grid_size"]

        gs = cfg["grid_size"]
        nc = cfg["n_colors"]
        colors = _COLOR_NAMES[:nc]
        pattern = cfg["pattern"]

        # Generate grid based on pattern
        grid = [[None] * gs for _ in range(gs)]

        if pattern == "row_repeat":
            for r in range(gs):
                for c in range(gs):
                    grid[r][c] = colors[c % nc]
        elif pattern == "col_repeat":
            for r in range(gs):
                for c in range(gs):
                    grid[r][c] = colors[r % nc]
        elif pattern == "diagonal":
            for r in range(gs):
                for c in range(gs):
                    grid[r][c] = colors[(r + c) % nc]
        elif pattern == "modular":
            for r in range(gs):
                for c in range(gs):
                    grid[r][c] = colors[(r * 2 + c) % nc]

        sidx = (self.seed or 0) % 16
        if cfg["qtype"] == "find_missing":
            # Replace one cell with "?" and ask its color
            mr, mc = rng.randint(0, gs - 1), rng.randint(0, gs - 1)
            correct_color = grid[mr][mc]
            grid[mr][mc] = "?"

            question = _MISSING_TEMPLATES[sidx].format(gs=gs, r=mr+1, c=mc+1)
            answer = correct_color
        else:
            # Replace one cell with wrong color and ask which cell is wrong
            wr, wc = rng.randint(0, gs - 1), rng.randint(0, gs - 1)
            correct_color = grid[wr][wc]
            wrong_color = rng.choice([c for c in colors if c != correct_color])
            grid[wr][wc] = wrong_color

            question = _WRONG_TEMPLATES[sidx].format(gs=gs)
            answer = f"{wr+1},{wc+1}"

        img = self._render_grid(grid, gs, cfg, rng)
        return question, answer, img

    def _render_grid(self, grid, gs, cfg, rng):
        style = self._random_style()
        sc = style["figsize_scale"]

        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            edge_col = tbp["line_color"]
            edge_lw = tbp["line_width"]
            missing_fc = tbp["fill_color"]
            q_color = tbp["line_color"]
            title_kw = {"fontfamily": tbp["font_family"], "color": tbp["line_color"]}
            dpi = tbp["dpi"]
            cell_alpha = 0.75
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            edge_col = "#1a1a1a"
            edge_lw = 1.6
            missing_fc = "#dddddd"
            q_color = "red"
            title_kw = {}
            dpi = style["dpi"]
            cell_alpha = 0.85
        else:
            bg = "#ffffff"
            edge_col = "black"
            edge_lw = 1.0
            missing_fc = "#dddddd"
            q_color = "red"
            title_kw = {}
            dpi = style["dpi"]
            cell_alpha = 0.85

        def _draw():
            fig, ax = plt.subplots(figsize=(gs * 1.2 * sc, gs * 1.2 * sc))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.set_xlim(0, gs)
            ax.set_ylim(0, gs)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("Color Grid Pattern", fontsize=13, fontweight="bold",
                         **title_kw)

            for r in range(gs):
                for c in range(gs):
                    val = grid[r][c]
                    if val == "?":
                        rect = mpatches.Rectangle((c, gs - 1 - r), 1, 1,
                                                   fc=missing_fc, ec=edge_col,
                                                   lw=max(edge_lw, 1.3))
                        ax.add_patch(rect)
                        ax.text(c + 0.5, gs - 0.5 - r, "?",
                               fontsize=16, ha="center", va="center",
                               fontweight="bold", color=q_color)
                    else:
                        color_hex = _COLOR_HEX.get(val, "#999999")
                        rect = mpatches.Rectangle((c, gs - 1 - r), 1, 1,
                                                   fc=color_hex, ec=edge_col,
                                                   lw=edge_lw, alpha=cell_alpha)
                        ax.add_patch(rect)

            for i in range(gs):
                ax.text(-0.3, gs - 0.5 - i, str(i + 1), fontsize=9, ha="center",
                        va="center", **title_kw)
                ax.text(i + 0.5, gs + 0.2, str(i + 1), fontsize=9, ha="center",
                        va="center", **title_kw)

            fig.tight_layout()
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()
        return self.fig_to_pil(fig, dpi=dpi)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskc"
    os.makedirs(out_dir, exist_ok=True)
    env = ColorGridPatternQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[color_grid_pattern L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"color_grid_pattern_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[color_grid_pattern L{level} s{s}] A={env._answer}")
