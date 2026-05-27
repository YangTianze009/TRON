"""
Tessellation QA — which shape tessellates, count tiles, identify pattern unit,
count-colors, missing-tile, row/column slicing.

Capabilities: V1 (shape recognition), R5 (pattern reasoning), R2 (geometry)

Difficulty gradient:
  L0-L1: count_tiles (3x3), does_it_tessellate (trivial regular polygon)
  L2-L3: identify_shape + does_it_tessellate + count_colors
  L4-L5: count_colors (3-4 cols), which_row (which row has N tiles),
         missing_tile (binary pattern)
  L6-L7: count_colors (up to 5), which_row, count_shape_edges
  L8-L9: pattern cycle length, longest_runs, combined

2026-05-03 extension (W36 / reference cutting & combining): added a new
question mode `cuts_largest_square_remainder` mirroring reference Q2:
"A rectangular piece of paper is 8 cm long and 5 cm wide. After cutting
out the largest possible square from it, the dimensions of the remaining
figure are (    ) cm." Answer is "5, 3" (or formatted dim pair). The
mode is enabled at L1+ and the env still keeps all existing tessellation
modes active.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TILE_SHAPE_POOL = ["square", "rectangle", "triangle", "hexagon"]
_TITLE_VARIANTS_COUNT = [
    "Tile grid — count tiles",
    "How many tiles in this pattern?",
    "Tessellation layout",
    "Tile count problem",
]

class TessellationQA(StandaloneVisualEnv):
    ENV_NAME = "tessellation"

    QUESTION_TYPES = [
        "count_tiles", "identify_shape", "does_it_tessellate",
        "count_colors", "missing_tile_color",
        "count_shape_edges", "which_row_has_more",
        "most_common_color",
        # W36 extension
        "cuts_largest_square_remainder",
        "cuts_perimeter_increase",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        level = max(0, min(level, 9))
        # Redesign 2026-04-17: monotonic difficulty.
        # Previous L6 (0.7) used a mixed pool that included missing_tile_color
        # and which_row_has_more that are harder than L9's pool. Now: L6
        # stays on mid-difficulty pool, L8-L9 use ONLY the hardest qtypes
        # with maximum grid size + color count.
        if level <= 1:
            return {"qtypes": ["count_tiles", "does_it_tessellate",
                               "cuts_largest_square_remainder"],
                    "grid_range": (3, 4), "n_colors_range": (2, 2)}
        if level <= 3:
            return {"qtypes": ["count_tiles", "identify_shape",
                               "does_it_tessellate", "count_colors",
                               "cuts_largest_square_remainder",
                               "cuts_perimeter_increase"],
                    "grid_range": (3, 5), "n_colors_range": (2, 3)}
        if level <= 5:
            return {"qtypes": ["count_tiles", "identify_shape",
                               "count_colors", "count_shape_edges",
                               "cuts_largest_square_remainder",
                               "cuts_perimeter_increase"],
                    "grid_range": (4, 6), "n_colors_range": (2, 4)}
        if level <= 7:
            return {"qtypes": ["count_colors", "count_shape_edges",
                               "which_row_has_more", "most_common_color",
                               "cuts_perimeter_increase"],
                    "grid_range": (4, 6), "n_colors_range": (3, 4)}
        return {"qtypes": ["missing_tile_color", "which_row_has_more",
                           "most_common_color",
                           "cuts_perimeter_increase"],
                "grid_range": (5, 7), "n_colors_range": (4, 5)}

    def _generate_problem(self, seed: int, parameter: Dict):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        try:
            dispatch = {
                "count_tiles": self._count_tiles,
                "identify_shape": self._identify_shape,
                "does_it_tessellate": self._does_tessellate,
                "count_colors": self._count_colors,
                "missing_tile_color": self._missing_tile,
                "count_shape_edges": self._count_shape_edges,
                "which_row_has_more": self._which_row_has_more,
                "most_common_color": self._most_common_color,
                "cuts_largest_square_remainder": self._cuts_largest_square_remainder,
                "cuts_perimeter_increase": self._cuts_perimeter_increase,
            }
            fn = dispatch.get(qtype)
            if fn is None:
                return None
            return fn(sub_rng, cfg)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def _count_tiles(self, rng, cfg):
        shape = rng.choice(["square", "rectangle"])
        lo, hi = cfg["grid_range"]
        rows = rng.randint(lo, hi)
        cols = rng.randint(lo, hi)
        total = rows * cols

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(max(5, cols), max(4, rows)))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        n_colors = rng.randint(*cfg["n_colors_range"])

        for r in range(rows):
            for c in range(cols):
                color = palette[(r + c) % n_colors]
                if shape == "square":
                    rect = mpatches.Rectangle(
                        (c, r), 1, 1, facecolor=color,
                        edgecolor="black", linewidth=1.5, alpha=0.75)
                else:
                    rect = mpatches.Rectangle(
                        (c * 1.5, r), 1.5, 1, facecolor=color,
                        edgecolor="black", linewidth=1.5, alpha=0.75)
                ax.add_patch(rect)

        if shape == "square":
            ax.set_xlim(-0.2, cols + 0.2)
        else:
            ax.set_xlim(-0.2, cols * 1.5 + 0.2)
        ax.set_ylim(-0.2, rows + 0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS_COUNT),
                     fontsize=13, fontweight="bold")

        q = "How many tiles are in this tessellation? Answer with an integer."
        return q, str(total), self.fig_to_pil(fig, dpi=style["dpi"])

    def _identify_shape(self, rng, cfg):
        shape = rng.choice(["triangle", "square", "hexagon"])
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        if shape == "triangle":
            for r in range(4):
                for c in range(6):
                    x0 = c * 0.5; y0 = r * 0.866
                    if (r + c) % 2 == 0:
                        tri = plt.Polygon(
                            [(x0, y0), (x0 + 0.5, y0), (x0 + 0.25, y0 + 0.433)],
                            facecolor=palette[c % len(palette)],
                            edgecolor="black", alpha=0.7)
                    else:
                        tri = plt.Polygon(
                            [(x0, y0 + 0.433), (x0 + 0.5, y0 + 0.433),
                             (x0 + 0.25, y0)],
                            facecolor=palette[(c + 1) % len(palette)],
                            edgecolor="black", alpha=0.7)
                    ax.add_patch(tri)
        elif shape == "square":
            for r in range(4):
                for c in range(5):
                    rect = mpatches.Rectangle(
                        (c, r), 1, 1,
                        facecolor=palette[(r + c) % len(palette)],
                        edgecolor="black", linewidth=1.5, alpha=0.7)
                    ax.add_patch(rect)
        elif shape == "hexagon":
            for r in range(3):
                for c in range(4):
                    cx = c * 1.5
                    cy = r * 1.732 + (0.866 if c % 2 == 1 else 0)
                    hex_pts = [(cx + math.cos(math.radians(60 * i)),
                                cy + math.sin(math.radians(60 * i)))
                               for i in range(6)]
                    ax.add_patch(plt.Polygon(
                        hex_pts, facecolor=palette[c % len(palette)],
                        edgecolor="black", linewidth=1.5, alpha=0.7))

        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
        ax.set_title("Tessellation pattern", fontsize=13, fontweight="bold")

        q = ("What shape is used as the tile in this tessellation: "
             "triangle, square, or hexagon?")
        return q, shape, self.fig_to_pil(fig, dpi=style["dpi"])

    def _does_tessellate(self, rng, cfg):
        n_sides = rng.choice([3, 4, 5, 6, 7, 8])
        can_tess = n_sides in [3, 4, 6]
        shape_names = {3: "triangle", 4: "square", 5: "pentagon",
                       6: "hexagon", 7: "heptagon", 8: "octagon"}
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(style["bg_color"])

        angles = [2 * math.pi * i / n_sides for i in range(n_sides)]
        pts = [(math.cos(a), math.sin(a)) for a in angles]
        poly = plt.Polygon(pts, facecolor=style["palette"][0],
                           edgecolor="black", linewidth=2, alpha=0.6)
        ax.add_patch(poly)
        ax.text(0, 0, f"{n_sides} sides", ha="center", va="center",
                fontsize=14, fontweight="bold")

        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Regular {shape_names[n_sides].title()}",
                     fontsize=14, fontweight="bold")

        q = (f"Can a regular {shape_names[n_sides]} tessellate the plane "
             "by itself (fill the plane with no gaps and no overlaps)? "
             "Answer Yes or No.")
        return q, "Yes" if can_tess else "No", self.fig_to_pil(fig, dpi=style["dpi"])

    def _count_colors(self, rng, cfg):
        n_colors = rng.randint(*cfg["n_colors_range"])
        lo, hi = cfg["grid_range"]
        rows = rng.randint(lo, hi)
        cols = rng.randint(lo, hi)
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"][:n_colors]
        # Build grid: iterate so each color appears ≥ once
        grid = []
        for r in range(rows):
            grid.append([])
            for c in range(cols):
                grid[-1].append((r * 3 + c * 5) % n_colors)
        # Sanity: if not all colors present, force patch
        used = {v for row in grid for v in row}
        for ci in range(n_colors):
            if ci not in used:
                grid[0][ci % cols] = ci

        for r in range(rows):
            for c in range(cols):
                color = palette[grid[r][c]]
                rect = mpatches.Rectangle(
                    (c, r), 1, 1, facecolor=color,
                    edgecolor="black", linewidth=1.5, alpha=0.75)
                ax.add_patch(rect)

        ax.set_xlim(-0.2, cols + 0.2); ax.set_ylim(-0.2, rows + 0.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Colored Tessellation", fontsize=13, fontweight="bold")
        q = ("How many distinct colors (excluding the background) are "
             "used in this tessellation? Answer with an integer.")
        return q, str(n_colors), self.fig_to_pil(fig, dpi=style["dpi"])

    def _missing_tile(self, rng, cfg):
        # 2-color pattern with one missing tile. Randomly choose
        # pattern_type so that the answer is balanced Yes/No:
        #   "checker": (r+c) % 2 -> diagonals SAME color (Yes)
        #   "row_stripe": r % 2 -> diagonals DIFFERENT color (No)
        #   "col_stripe": c % 2 -> diagonals DIFFERENT color (No)
        rows, cols = 4, 4
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"][:2]
        pattern_type = rng.choice(["checker", "row_stripe", "col_stripe"])

        # Pick a missing cell that has at least one diagonal neighbour in-grid
        # (corner cells have only one diagonal; that is still fine).
        missing_r = rng.randint(0, rows - 1)
        missing_c = rng.randint(0, cols - 1)

        def color_idx(r, c):
            if pattern_type == "checker":
                return (r + c) % 2
            if pattern_type == "row_stripe":
                return r % 2
            return c % 2

        for r in range(rows):
            for c in range(cols):
                if r == missing_r and c == missing_c:
                    rect = mpatches.Rectangle(
                        (c, r), 1, 1, facecolor="#f0f0f0",
                        edgecolor="red", linewidth=2, linestyle="--")
                    ax.add_patch(rect)
                    ax.text(c + 0.5, r + 0.5, "?", ha="center", va="center",
                            fontsize=20, color="red", fontweight="bold")
                else:
                    color = palette[color_idx(r, c)]
                    rect = mpatches.Rectangle(
                        (c, r), 1, 1, facecolor=color, edgecolor="black",
                        linewidth=1.5, alpha=0.75)
                    ax.add_patch(rect)

        ax.set_xlim(-0.2, cols + 0.2); ax.set_ylim(-0.2, rows + 0.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Fill the missing tile", fontsize=13, fontweight="bold")
        q = ("The tessellation uses a 2-color pattern with one tile marked "
             "'?'. Looking at the visible pattern, should the missing tile "
             "be the SAME COLOR as its diagonal neighbors (the tiles that "
             "share a CORNER with it)? Answer Yes or No.")
        # Diagonals have indices (r+/-1, c+/-1). For checker: (r+c)%2 ==
        # (r+c+/-2)%2 -> same. For row/col stripes: differs by 1 -> different.
        answer = "Yes" if pattern_type == "checker" else "No"
        return q, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    def _count_shape_edges(self, rng, cfg):
        # Draw a single regular polygon tile (pentagon, hex, octagon)
        n_sides = rng.choice([3, 4, 5, 6, 7, 8])
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        rot = rng.uniform(0, math.pi)
        pts = [(math.cos(2 * math.pi * i / n_sides + rot),
                math.sin(2 * math.pi * i / n_sides + rot))
               for i in range(n_sides)]
        ax.add_patch(plt.Polygon(pts, facecolor=palette[0],
                                 edgecolor="black", linewidth=2, alpha=0.6))
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Single tile", fontsize=13, fontweight="bold")
        q = ("The image shows a single tile of a tessellation. "
             "How many edges (straight sides) does this tile have? "
             "Answer with an integer.")
        return q, str(n_sides), self.fig_to_pil(fig, dpi=style["dpi"])

    def _which_row_has_more(self, rng, cfg):
        # Two rows have different tile counts. Which has more?
        # Render as 2 stacked strips of squares of different lengths.
        a = rng.randint(3, 6)
        b = rng.randint(3, 6)
        while b == a:
            b = rng.randint(3, 6)
        r1, r2 = a, b
        top_label = rng.choice(["A", "Top"])
        bot_label = "B" if top_label == "A" else "Bottom"
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(max(5, max(r1, r2)), 3))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        # Row 1 at y=1
        for c in range(r1):
            ax.add_patch(mpatches.Rectangle(
                (c, 1.2), 1, 1, facecolor=palette[c % len(palette)],
                edgecolor="black", linewidth=1.3, alpha=0.7))
        for c in range(r2):
            ax.add_patch(mpatches.Rectangle(
                (c, 0), 1, 1, facecolor=palette[(c + 2) % len(palette)],
                edgecolor="black", linewidth=1.3, alpha=0.7))
        ax.text(-0.4, 1.7, top_label, fontsize=13, fontweight="bold", ha="right")
        ax.text(-0.4, 0.5, bot_label, fontsize=13, fontweight="bold", ha="right")
        ax.set_xlim(-1.5, max(r1, r2) + 0.5)
        ax.set_ylim(-0.3, 2.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Two tile rows", fontsize=13, fontweight="bold")
        answer = top_label if r1 > r2 else bot_label
        q = (f"The image shows two labeled rows of tiles ({top_label} and "
             f"{bot_label}). Which row contains MORE tiles? "
             f"Answer with the row label.")
        return q, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    def _most_common_color(self, rng, cfg):
        # Grid with weighted color distribution; one color dominates.
        rows = rng.randint(4, 6); cols = rng.randint(4, 6)
        n_colors = rng.randint(3, min(5, cfg["n_colors_range"][1] or 4))
        style = self._random_style()
        palette = style["palette"][:n_colors]
        # Color names (we'll label colors as Color A/B/C/D/E and render legend)
        color_names = ["A", "B", "C", "D", "E"]
        # Assign dominant color
        dom = rng.randrange(n_colors)
        grid = []
        counts = [0] * n_colors
        for r in range(rows):
            grid.append([])
            for c in range(cols):
                if rng.random() < 0.5:
                    idx = dom
                else:
                    idx = rng.randrange(n_colors)
                grid[-1].append(idx)
                counts[idx] += 1
        # Ensure dominant has strict majority
        if not (counts[dom] - max(v for i, v in enumerate(counts) if i != dom) >= 2):
            # Force extra dominant entries
            for r in range(rows):
                for c in range(cols):
                    if counts[dom] - max(
                            v for i, v in enumerate(counts) if i != dom) < 2:
                        old = grid[r][c]
                        grid[r][c] = dom
                        counts[old] -= 1
                        counts[dom] += 1
                    else:
                        break

        fig, ax = plt.subplots(figsize=(cols + 1.5, rows + 0.5))
        fig.patch.set_facecolor(style["bg_color"])
        for r in range(rows):
            for c in range(cols):
                ax.add_patch(mpatches.Rectangle(
                    (c, r), 1, 1, facecolor=palette[grid[r][c]],
                    edgecolor="black", linewidth=1.2, alpha=0.78))

        # Legend: color label box
        for i in range(n_colors):
            ax.add_patch(mpatches.Rectangle(
                (cols + 0.3, rows - 1 - i), 0.4, 0.4,
                facecolor=palette[i], edgecolor="black"))
            ax.text(cols + 0.9, rows - 1 - i + 0.2, color_names[i],
                    fontsize=12, va="center", fontweight="bold")

        ax.set_xlim(-0.3, cols + 2)
        ax.set_ylim(-0.3, rows + 0.3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Find the most common color", fontsize=13, fontweight="bold")

        answer = color_names[dom]
        q = ("The tessellation on the left uses several colors, labeled "
             "A-E in the legend. Which color appears in the MOST tiles? "
             "Answer with just the color letter.")
        return q, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # W36 (cutting & combining) extension modes
    # ------------------------------------------------------------------ #

    def _cuts_largest_square_remainder(self, rng, cfg):
        """reference W36 — rectangle, cut largest square, give remaining dims.

        Sample (design notes Q2): "A rectangular piece of paper is 8
        cm long and 5 cm wide. After cutting out the largest possible square
        from it, the dimensions of the remaining figure are (    ) cm. → 5, 3"
        """
        # Choose rectangle with L > W, both integer. Larger L variants at
        # higher levels; keep small for easy levels.
        if cfg.get("grid_range") and cfg["grid_range"][1] >= 5:
            L = rng.randint(6, 12)
            W = rng.randint(3, L - 2)
        else:
            L = rng.randint(4, 8)
            W = rng.randint(2, L - 1)
        # Largest square inscribed has side W; remaining strip = (L - W) by W.
        rem_long = max(L - W, W)
        rem_short = min(L - W, W)
        if L - W <= 0:
            return None
        answer = f"{rem_long}, {rem_short}"

        # Render rectangle with dashed cut line indicating the largest square.
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(5, L * 0.5) * sc, max(4, W * 0.5) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        # Original rectangle outline
        rect = mpatches.Rectangle((0, 0), L, W, facecolor=style["palette"][0],
                                  edgecolor="black", linewidth=2.0, alpha=0.45)
        ax.add_patch(rect)
        # Dashed cut line at x=W (the largest-square boundary)
        ax.plot([W, W], [0, W], color="red", linestyle="--", linewidth=2.0)
        # Annotate side lengths
        ax.text(L / 2, -0.4, f"{L} cm", ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(-0.4, W / 2, f"{W} cm", ha="right", va="center", fontsize=12, fontweight="bold")
        # Mark the inscribed square label
        ax.text(W / 2, W / 2, "Largest\nsquare", ha="center", va="center",
                fontsize=10, fontweight="bold", color="black")
        if L - W > 0:
            ax.text((W + L) / 2, W / 2, "Remainder",
                    ha="center", va="center", fontsize=9, fontstyle="italic")
        ax.set_xlim(-1, L + 1)
        ax.set_ylim(-1, W + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Cut the largest square from the rectangle",
                     fontsize=13, fontweight="bold")

        q = (f"A rectangular piece of paper is {L} cm long and {W} cm wide. "
             f"After cutting out the largest possible square from it, what "
             f"are the dimensions of the remaining (rectangular) figure? "
             f"Answer as two integers separated by a comma (long side, short "
             f"side), e.g. '5, 3'.")
        return q, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    def _cuts_perimeter_increase(self, rng, cfg):
        """reference W36 — sector / rectangle cut: total NEW edge length added.

        Sample (design notes Q1): "5×3 cake cut along EF and GH —
        additional edge length compared to original perimeter equals (    )"
        Answer: 2HG + 2EF (each cut adds 2× the cut length to perimeter).

        Keep this concrete: render an L×W rectangle with one or two horizontal/
        vertical cut lines; ask for total NEW edge length added (numeric, in cm).
        Each interior cut of length c adds 2c (both sides become new edge).
        """
        L = rng.randint(4, 10)
        W = rng.randint(3, L - 1)
        # Number of cuts: 1 or 2. Each cut is either horizontal (length L) or
        # vertical (length W).
        n_cuts = rng.choice([1, 2])
        cuts = []  # list of ("h" or "v", length)
        for _ in range(n_cuts):
            direction = rng.choice(["h", "v"])
            if direction == "h":
                cuts.append(("h", L))
            else:
                cuts.append(("v", W))
        # Perimeter increase: sum of 2 * cut.length for each cut
        increase = sum(2 * c[1] for c in cuts)
        answer = str(increase)

        # Render rectangle with cut lines
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(5, L * 0.5) * sc, max(4, W * 0.5) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        rect = mpatches.Rectangle((0, 0), L, W, facecolor=style["palette"][0],
                                  edgecolor="black", linewidth=2.0, alpha=0.45)
        ax.add_patch(rect)
        # Choose distinct positions for cuts
        h_positions = []
        v_positions = []
        for (direction, _) in cuts:
            if direction == "h":
                # interior y position
                tries = 0
                y = None
                while tries < 8:
                    cand = rng.randint(1, W - 1)
                    if cand not in h_positions:
                        y = cand; h_positions.append(cand); break
                    tries += 1
                if y is None:
                    y = h_positions[0] if h_positions else 1
                ax.plot([0, L], [y, y], color="red", linestyle="--", linewidth=2.0)
            else:
                tries = 0
                x = None
                while tries < 8:
                    cand = rng.randint(1, L - 1)
                    if cand not in v_positions:
                        x = cand; v_positions.append(cand); break
                    tries += 1
                if x is None:
                    x = v_positions[0] if v_positions else 1
                ax.plot([x, x], [0, W], color="red", linestyle="--", linewidth=2.0)

        # Side labels
        ax.text(L / 2, -0.5, f"{L} cm", ha="center", va="top",
                fontsize=12, fontweight="bold")
        ax.text(-0.4, W / 2, f"{W} cm", ha="right", va="center",
                fontsize=12, fontweight="bold")
        ax.set_xlim(-1, L + 1)
        ax.set_ylim(-1, W + 1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Rectangle with cut lines (dashed)",
                     fontsize=13, fontweight="bold")

        q = (f"A rectangle with length {L} cm and width {W} cm is shown, "
             f"with {n_cuts} dashed cut line"
             f"{'s' if n_cuts > 1 else ''} that fully traverse the rectangle. "
             f"After the cut(s), what is the TOTAL increase in edge length "
             f"compared to the original perimeter? "
             f"(Each interior cut of length c adds 2c to the total edge length, "
             f"because both sides of the cut become new edges.) "
             f"Give an integer in cm.")
        return q, answer, self.fig_to_pil(fig, dpi=style["dpi"])
