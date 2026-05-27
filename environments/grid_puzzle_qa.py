"""Grid puzzle QA — row/column sum constraints, find missing value.

Difficulty redesign 2026-04-14.
L0: 4x4 grid, one missing, values 1-9 (was L3-L4).
L9: 6x6 grid, 3 missing (product of all three), no row/col labels on
    some axes, multiplicative constraints mixed in — target 5-15%.
"""
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class GridPuzzleQA(StandaloneVisualEnv):
    ENV_NAME = "grid_puzzle"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "grid_size": [4, 4, 4, 5, 5, 5, 5, 6, 6, 6][level],
            "n_hidden": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3][level],
            "val_range": (1 + level // 3, 9 + level),  # (1,9)..(4,18)
            # L5+: ask product instead of sum of hidden cells
            "ask_product": level >= 7,
            # L6+: hide some row/col sum labels
            "hide_some_sums": level >= 6,
            # L8+: use multiplication constraint for one axis
            "mixed_constraints": level >= 8,
        }

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 720)
        n = cfg["grid_size"]
        vlo, vhi = cfg["val_range"]

        grid = [[sub_rng.randint(vlo, vhi) for _ in range(n)] for _ in range(n)]
        row_sums = [sum(row) for row in grid]
        col_sums = [sum(grid[r][c] for r in range(n)) for c in range(n)]

        # For mixed_constraints: col constraints are products
        if cfg["mixed_constraints"]:
            col_products = [1 for _ in range(n)]
            for c in range(n):
                p = 1
                for r in range(n):
                    p *= grid[r][c]
                col_products[c] = p
        else:
            col_products = None

        # Select hidden cells
        nh = cfg["n_hidden"]
        all_cells = [(r, c) for r in range(n) for c in range(n)]
        if nh == 1:
            hr, hc = sub_rng.randint(0, n-1), sub_rng.randint(0, n-1)
            hidden = [(hr, hc)]
        elif nh == 2:
            # Same row — solvable via col sums
            hr = sub_rng.randint(0, n-1)
            cols = sub_rng.sample(range(n), 2)
            hidden = [(hr, cols[0]), (hr, cols[1])]
        else:
            # 3 hidden: pick from different rows AND different columns
            rows = sub_rng.sample(range(n), min(3, n))
            cols_avail = list(range(n))
            hidden = []
            for r in rows:
                c = sub_rng.choice(cols_avail)
                cols_avail.remove(c)
                hidden.append((r, c))

        hidden_set = set(hidden)
        hidden_vals = [grid[r][c] for r, c in hidden]

        # Build question
        if nh == 1:
            answer = str(hidden_vals[0])
            question = "Each row and column sum is shown. What number replaces '?'?"
        elif cfg.get("ask_product"):
            answer = str(hidden_vals[0] * hidden_vals[1] * hidden_vals[2] if nh == 3
                         else hidden_vals[0] * hidden_vals[1])
            question = f"There are {nh} hidden cells marked with '?'. What is the PRODUCT of all hidden values?"
        else:
            answer = str(sum(hidden_vals))
            question = f"There are {nh} hidden cells marked with '?'. What is the SUM of all hidden values?"

        # Draw
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(n + 2.5, n + 2.5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        for r in range(n):
            for c in range(n):
                is_h = (r, c) in hidden_set
                bg = "#fff3cd" if is_h else "#ffffff"
                rect = mpatches.Rectangle((c, n-r-1), 1, 1, facecolor=bg,
                                          edgecolor="#2c3e50", linewidth=2)
                ax.add_patch(rect)
                if is_h:
                    ax.text(c+0.5, n-r-0.5, "?", ha="center", va="center",
                            fontsize=20, color="red", fontweight="bold")
                else:
                    ax.text(c+0.5, n-r-0.5, str(grid[r][c]), ha="center",
                            va="center", fontsize=14, fontweight="bold")

        # Decide which sums to hide
        hide_rows = set()
        hide_cols = set()
        if cfg["hide_some_sums"]:
            # Hide about 1/3 of row sums and 1/3 of col sums
            nr_hide = max(1, n // 3)
            hide_rows = set(sub_rng.sample(range(n), nr_hide))
            hide_cols = set(sub_rng.sample(range(n), nr_hide))
            # Make sure hidden cells can still be solved
            # (at least show sum for the row/col containing a hidden cell)
            for hr, hc in hidden:
                hide_rows.discard(hr)
                hide_cols.discard(hc)

        # Row sums
        for r in range(n):
            if r in hide_rows:
                continue
            ax.text(n+0.5, n-r-0.5, f"\u03a3={row_sums[r]}", ha="center",
                    va="center", fontsize=11, color=palette[0], fontweight="bold")

        # Col sums / products
        for c in range(n):
            if c in hide_cols:
                continue
            if col_products is not None:
                # Rotate 45deg + smaller font to avoid overlap for long
                # product values (can be 6-7 digits with large grids).
                ax.text(c+0.5, -0.5, f"\u03a0={col_products[c]}", ha="center",
                        va="center", fontsize=8, color=palette[1],
                        fontweight="bold", rotation=30)
            else:
                ax.text(c+0.5, -0.5, f"\u03a3={col_sums[c]}", ha="center",
                        va="center", fontsize=11, color=palette[1], fontweight="bold")

        ax.set_xlim(-0.3, n+1.5); ax.set_ylim(-1.6, n+0.3)
        ax.set_aspect("equal"); ax.axis("off")
        title = "Find the hidden value(s)" if nh > 1 else "Find '?' using row/column sums"
        ax.set_title(title, fontsize=13, fontweight="bold")

        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
