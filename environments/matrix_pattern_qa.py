"""
Matrix number pattern QA — find missing value in NxN grid following arithmetic rules.
Targets: VisualPuzzles algorithmic, visual-perception IQ_Test, MMStar logical reasoning.
Capabilities: V10 (counting), R1 (arithmetic), R5 (pattern reasoning)
"""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class MatrixPatternQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "matrix_pattern"

    def _level_config(self, level: int) -> dict:
        """Difficulty redesign 2026-04-14. L0 starts at multiply rule, L9 multi-hidden + large grid."""
        configs = {
            0: {'question_type': 'multiply_rule', 'grid_size': 3, 'n_hidden': 1},
            1: {'question_type': 'multiply_rule', 'grid_size': 3, 'n_hidden': 1},
            2: {'question_type': 'row_product', 'grid_size': 3, 'n_hidden': 1},
            3: {'question_type': 'row_product', 'grid_size': 4, 'n_hidden': 1},
            4: {'question_type': 'col_arithmetic', 'grid_size': 4, 'n_hidden': 2},
            5: {'question_type': 'diagonal_pattern', 'grid_size': 4, 'n_hidden': 2},
            6: {'question_type': 'multiply_rule', 'grid_size': 5, 'n_hidden': 2},
            7: {'question_type': 'row_product', 'grid_size': 5, 'n_hidden': 2},
            8: {'question_type': 'row_product', 'grid_size': 5, 'n_hidden': 3},
            9: {'question_type': 'row_product', 'grid_size': 6, 'n_hidden': 3},
        }
        return configs.get(max(0, min(9, level)), configs[0])

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:

        # ── Level routing (auto-generated) ──
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)
        rng = self._rng
        n = parameter.get("grid_size", rng.choice([3, 4, 5]))
        # Bugfix 2026-04-17: level_config uses key 'question_type' but
        # generator was reading 'rule' — random rule was picked. Honor
        # question_type first so L9=row_product actually uses row_product.
        rule = parameter.get("rule",
                parameter.get("question_type", rng.choice([
                    "row_sum_constant", "col_arithmetic", "diagonal_pattern",
                    "multiply_rule", "row_product",
                ])))

        grid = [[0]*n for _ in range(n)]

        if rule == "row_sum_constant":
            target_sum = rng.randint(15, 30)
            for r in range(n):
                vals = [rng.randint(1, target_sum // n + 3) for _ in range(n-1)]
                vals.append(target_sum - sum(vals))
                if vals[-1] < 1:
                    vals[-1] = rng.randint(1, 5)
                    vals[0] = target_sum - sum(vals[1:])
                grid[r] = vals

        elif rule == "col_arithmetic":
            # Each column is arithmetic sequence
            for c in range(n):
                start = rng.randint(1, 10)
                step = rng.randint(1, 5)
                for r in range(n):
                    grid[r][c] = start + r * step

        elif rule == "diagonal_pattern":
            # Main diagonal has a pattern
            base = rng.randint(1, 5)
            step = rng.randint(2, 4)
            for r in range(n):
                for c in range(n):
                    grid[r][c] = base + (r + c) * step

        elif rule == "multiply_rule":
            # Each cell = row_header * col_header
            row_h = [rng.randint(1, 9) for _ in range(n)]
            col_h = [rng.randint(1, 9) for _ in range(n)]
            for r in range(n):
                for c in range(n):
                    grid[r][c] = row_h[r] * col_h[c]

        elif rule == "row_product":
            for r in range(n):
                ratio = rng.choice([2, 3])
                start = rng.randint(1, 5)
                for c in range(n):
                    grid[r][c] = start * (ratio ** c)

        # Pick cell(s) to hide
        n_hidden = parameter.get("n_hidden", 1)
        all_cells = [(r, c) for r in range(n) for c in range(n)]
        hidden_cells = rng.sample(all_cells, min(n_hidden, len(all_cells)))
        hidden_vals = [grid[r][c] for r, c in hidden_cells]
        hidden_set = set(hidden_cells)
        if n_hidden == 1:
            answer = hidden_vals[0]
        else:
            answer = sum(hidden_vals)  # sum of all hidden

        # Render
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(n + 1.5, n + 1.5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        for r in range(n):
            for c in range(n):
                is_hidden = (r, c) in hidden_set
                bg = "#fff3cd" if is_hidden else "#ffffff"
                rect = mpatches.Rectangle((c, n - r - 1), 1, 1, facecolor=bg,
                                         edgecolor="#2c3e50", linewidth=2)
                ax.add_patch(rect)

                if is_hidden:
                    ax.text(c + 0.5, n - r - 0.5, "?", ha="center", va="center",
                           fontsize=20, color="red", fontweight="bold")
                else:
                    ax.text(c + 0.5, n - r - 0.5, str(grid[r][c]), ha="center", va="center",
                           fontsize=14, fontweight="bold", color="#2c3e50")

        ax.set_xlim(-0.1, n + 0.1)
        ax.set_ylim(-0.1, n + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Find the Missing Number", fontsize=14, fontweight="bold")

        if n_hidden == 1:
            question = "Study the number pattern in the grid. What number should replace the '?'?"
        else:
            question = (f"Study the number pattern in the grid. There are {n_hidden} hidden cells "
                       f"marked '?'. Find each hidden value and give their SUM.")
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
