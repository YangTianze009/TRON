"""
Path Counting QA environment.

Renders a grid with blocked cells. Count paths from top-left to bottom-right
using only right/down moves.
Questions: total paths, paths through cell, paths avoiding cell.
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _count_paths(rows, cols, blocked):
    """Count paths from (0,0) to (rows-1, cols-1) with blocked cells."""
    dp = [[0] * cols for _ in range(rows)]
    blocked_set = set(blocked)
    if (0, 0) in blocked_set or (rows - 1, cols - 1) in blocked_set:
        return 0
    dp[0][0] = 1
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked_set:
                dp[r][c] = 0
                continue
            if r == 0 and c == 0:
                continue
            from_top = dp[r - 1][c] if r > 0 else 0
            from_left = dp[r][c - 1] if c > 0 else 0
            dp[r][c] = from_top + from_left
    return dp[rows - 1][cols - 1]

def _count_paths_through(rows, cols, blocked, tr, tc):
    """Count paths that pass through cell (tr, tc)."""
    if (tr, tc) in set(blocked):
        return 0
    # Paths from (0,0) to (tr,tc) * paths from (tr,tc) to (rows-1,cols-1)
    # Sub-grid 1: (0,0) to (tr,tc)
    p1 = _count_paths(tr + 1, tc + 1,
                       [(r, c) for r, c in blocked if r <= tr and c <= tc])
    # Sub-grid 2: (tr,tc) to (rows-1,cols-1)
    sub_rows = rows - tr
    sub_cols = cols - tc
    shifted_blocked = [(r - tr, c - tc) for r, c in blocked
                       if r >= tr and c >= tc]
    p2 = _count_paths(sub_rows, sub_cols, shifted_blocked)
    return p1 * p2

class PathCountingQA(StandaloneVisualEnv):
    ENV_NAME = "path_counting"

    QUESTION_TYPES = [
        "total_paths", "paths_through", "paths_avoiding",
        "is_path_possible", "blocked_count",
        # Harder types
        "paths_through_exactly_k",
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"rows": (2, 3), "cols": (2, 3), "blocked": (0, 1),
                    "qtypes": ["blocked_count", "is_path_possible", "total_paths"],
                    "qweights": [4, 3, 3]}
        if level == 1:
            return {"rows": (3, 3), "cols": (3, 3), "blocked": (0, 1),
                    "qtypes": ["total_paths", "is_path_possible", "blocked_count"],
                    "qweights": [5, 3, 2]}
        if level == 2:
            return {"rows": (3, 4), "cols": (3, 4), "blocked": (1, 2),
                    "qtypes": ["total_paths", "paths_through", "blocked_count"],
                    "qweights": [4, 4, 2]}
        if level == 3:
            return {"rows": (3, 4), "cols": (3, 4), "blocked": (1, 2),
                    "qtypes": ["total_paths", "paths_through", "paths_avoiding"],
                    "qweights": [3, 4, 3]}
        if level == 4:
            return {"rows": (3, 5), "cols": (3, 5), "blocked": (1, 3),
                    "qtypes": ["total_paths", "paths_through", "paths_avoiding"],
                    "qweights": [3, 3, 4]}
        if level == 5:
            return {"rows": (4, 5), "cols": (4, 5), "blocked": (2, 3),
                    "qtypes": ["total_paths", "paths_through", "paths_avoiding"],
                    "qweights": [3, 3, 4]}
        if level == 6:
            return {"rows": (4, 5), "cols": (4, 5), "blocked": (2, 4),
                    "qtypes": ["total_paths", "paths_through", "paths_avoiding", "paths_through_exactly_k"],
                    "qweights": [2, 3, 3, 2]}
        if level == 7:
            return {"rows": (4, 6), "cols": (4, 6), "blocked": (2, 4),
                    "qtypes": ["paths_through", "paths_avoiding", "paths_through_exactly_k"],
                    "qweights": [3, 3, 4]}
        if level == 8:
            return {"rows": (5, 6), "cols": (5, 6), "blocked": (3, 5),
                    "qtypes": ["paths_through", "paths_avoiding", "paths_through_exactly_k"],
                    "qweights": [3, 3, 4]}
        return {"rows": (5, 7), "cols": (5, 7), "blocked": (3, 6),
                "qtypes": ["paths_avoiding", "paths_through_exactly_k"],
                "qweights": [4, 6]}

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 4201)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        rows = sub_rng.randint(*cfg["rows"])
        cols = sub_rng.randint(*cfg["cols"])
        n_blocked = sub_rng.randint(*cfg["blocked"])

        # Generate blocked cells (not start or end)
        all_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if (r, c) != (0, 0) and (r, c) != (rows - 1, cols - 1)]
        blocked = rng.sample(all_cells, min(n_blocked, len(all_cells)))

        total = _count_paths(rows, cols, blocked)

        img = self._render(rows, cols, blocked)

        if qtype == "total_paths":
            q = (f"In this {rows}x{cols} grid, how many distinct paths exist from "
                 f"the top-left corner to the bottom-right corner? "
                 f"You can only move right or down. Blocked cells (marked X) cannot be entered.")
            a = str(total)

        elif qtype == "paths_through":
            free = [(r, c) for r in range(rows) for c in range(cols)
                    if (r, c) not in set(blocked) and (r, c) != (0, 0)
                    and (r, c) != (rows - 1, cols - 1)]
            if not free:
                return None
            tr, tc = rng.choice(free)
            ct = _count_paths_through(rows, cols, blocked, tr, tc)
            q = (f"How many paths from top-left to bottom-right pass through "
                 f"cell ({tr+1},{tc+1})? Only right/down moves allowed.")
            a = str(ct)

        elif qtype == "paths_avoiding":
            free = [(r, c) for r in range(rows) for c in range(cols)
                    if (r, c) not in set(blocked) and (r, c) != (0, 0)
                    and (r, c) != (rows - 1, cols - 1)]
            if not free:
                return None
            ar, ac = rng.choice(free)
            new_blocked = blocked + [(ar, ac)]
            ct = _count_paths(rows, cols, new_blocked)
            q = (f"How many paths from top-left to bottom-right avoid "
                 f"cell ({ar+1},{ac+1})? Only right/down moves allowed, "
                 f"and blocked cells (X) also cannot be entered.")
            a = str(ct)

        elif qtype == "is_path_possible":
            ans = "Yes" if total > 0 else "No"
            q = ("Is there any path from the top-left to the bottom-right "
                 "using only right/down moves? Answer Yes or No.")
            a = ans

        elif qtype == "blocked_count":
            q = "How many blocked cells (marked X) are in the grid?"
            a = str(len(blocked))

        elif qtype == "paths_through_exactly_k":
            # Find cells adjacent to blocked cells (that are free)
            blocked_set = set(blocked)
            adj_to_blocked = set()
            for br, bc in blocked:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr2, nc2 = br+dr, bc+dc
                    if 0 <= nr2 < rows and 0 <= nc2 < cols:
                        if (nr2, nc2) not in blocked_set and (nr2, nc2) != (0,0) and (nr2, nc2) != (rows-1, cols-1):
                            adj_to_blocked.add((nr2, nc2))
            adj_list = sorted(adj_to_blocked)
            if len(adj_list) < 2:
                return None
            # Pick k = 1 and count paths through exactly 1 of these adjacent cells
            # P(through exactly 1) = sum of P(through cell_i) - 2*sum of P(through cell_i AND cell_j) + ...
            # Simplified: use inclusion-exclusion for k=1
            # Actually: count paths through at least one, then subtract paths through at least two
            # For simplicity: pick 2 specific cells, ask: how many paths go through cell A but NOT cell B?
            if len(adj_list) >= 2:
                ca, cb = rng.sample(adj_list, 2)
                # paths through A = total through A
                paths_a = _count_paths_through(rows, cols, blocked, ca[0], ca[1])
                # paths through both A and B
                # For A then B (A must be reachable before B in right-down path)
                paths_ab = 0
                if ca[0] <= cb[0] and ca[1] <= cb[1]:
                    # A before B
                    p1 = _count_paths_through(rows, cols, blocked, ca[0], ca[1])
                    # Of those through A, how many also go through B?
                    # paths(S->A) * paths(A->B) * paths(B->E)
                    p_sa = _count_paths(ca[0]+1, ca[1]+1,
                                        [(r,c) for r,c in blocked if r<=ca[0] and c<=ca[1]])
                    sub_r = cb[0]-ca[0]+1
                    sub_c = cb[1]-ca[1]+1
                    p_ab = _count_paths(sub_r, sub_c,
                                        [(r-ca[0],c-ca[1]) for r,c in blocked if ca[0]<=r<=cb[0] and ca[1]<=c<=cb[1]])
                    sub_r2 = rows-cb[0]
                    sub_c2 = cols-cb[1]
                    p_be = _count_paths(sub_r2, sub_c2,
                                        [(r-cb[0],c-cb[1]) for r,c in blocked if r>=cb[0] and c>=cb[1]])
                    paths_ab = p_sa * p_ab * p_be
                elif cb[0] <= ca[0] and cb[1] <= ca[1]:
                    # B before A
                    p_sb = _count_paths(cb[0]+1, cb[1]+1,
                                        [(r,c) for r,c in blocked if r<=cb[0] and c<=cb[1]])
                    sub_r = ca[0]-cb[0]+1
                    sub_c = ca[1]-cb[1]+1
                    p_ba = _count_paths(sub_r, sub_c,
                                        [(r-cb[0],c-cb[1]) for r,c in blocked if cb[0]<=r<=ca[0] and cb[1]<=c<=ca[1]])
                    sub_r2 = rows-ca[0]
                    sub_c2 = cols-ca[1]
                    p_ae = _count_paths(sub_r2, sub_c2,
                                        [(r-ca[0],c-ca[1]) for r,c in blocked if r>=ca[0] and c>=ca[1]])
                    paths_ab = p_sb * p_ba * p_ae

                result = paths_a - paths_ab
                q = (f"Consider cells adjacent to blocked cells: "
                     f"({ca[0]+1},{ca[1]+1}) and ({cb[0]+1},{cb[1]+1}). "
                     f"How many paths from S to E pass through ({ca[0]+1},{ca[1]+1}) "
                     f"but NOT through ({cb[0]+1},{cb[1]+1})? "
                     f"Only right/down moves allowed.")
                a = str(result)
            else:
                return None
        else:
            return None

        return q, a, img

    def _render(self, rows, cols, blocked):
        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(4, cols * 1.2) * s, max(4, rows * 1.2) * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")

        palette = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        blocked_set = set(blocked)

        for r in range(rows):
            for c in range(cols):
                y = rows - 1 - r
                if (r, c) in blocked_set:
                    color = "#444444"
                elif (r, c) == (0, 0):
                    color = palette[0]
                elif (r, c) == (rows - 1, cols - 1):
                    color = palette[1]
                else:
                    color = "#f0f0f0"
                rect = mpatches.FancyBboxPatch(
                    (c + 0.05, y + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02", facecolor=color,
                    edgecolor="#999", linewidth=1)
                ax.add_patch(rect)

                if (r, c) in blocked_set:
                    ax.text(c + 0.5, y + 0.5, "X", ha="center", va="center",
                            fontsize=fs + 4, fontweight="bold", color="white",
                            fontfamily=ff)
                elif (r, c) == (0, 0):
                    ax.text(c + 0.5, y + 0.5, "S", ha="center", va="center",
                            fontsize=fs + 2, fontweight="bold", color="white",
                            fontfamily=ff)
                elif (r, c) == (rows - 1, cols - 1):
                    ax.text(c + 0.5, y + 0.5, "E", ha="center", va="center",
                            fontsize=fs + 2, fontweight="bold", color="white",
                            fontfamily=ff)

        # Row/col labels
        for r in range(rows):
            ax.text(-0.3, rows - 1 - r + 0.5, str(r + 1), ha="center", va="center",
                    fontsize=fs - 1, fontfamily=ff, color="#666")
        for c in range(cols):
            ax.text(c + 0.5, rows + 0.3, str(c + 1), ha="center", va="center",
                    fontsize=fs - 1, fontfamily=ff, color="#666")

        ax.set_xlim(-0.6, cols + 0.3)
        ax.set_ylim(-0.3, rows + 0.7)
        ax.axis("off")
        ax.set_title("Grid Path Counting", fontsize=fs + 3, fontweight="bold",
                      fontfamily=ff, pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
