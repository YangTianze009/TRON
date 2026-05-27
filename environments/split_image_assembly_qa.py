"""
Split Image Assembly QA (multi-image P5 + S2.4 tiling / jigsaw).

Shown: 4 (or 6) image fragments labeled A-D as pieces of a split geometric
pattern + 4 candidate assembled images (A-D). Ask which assembled
candidate correctly combines all pieces.

Difficulty axes:
  A) n_pieces (4 for level <= 6, 6 for level >= 7).
  B) cut_type: L0 = straight horizontal/vertical; L5 = diagonal; L9 =
     irregular (jagged). Also piece_rotation_needed = level >= 4.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#f1c40f", "#e91e63"]

class SplitImageAssemblyQA(StandaloneVisualEnv):
    ENV_NAME = "split_image_assembly"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0/L1: 4 pieces, each piece is a SINGLE solid color (no noisy
        # mosaic), straight cuts, no rotation. The model just has to match
        # the 2x2 quadrant colors.
        if level <= 1:
            return {
                "n_pieces": 4,
                "cut_type": "straight",
                "piece_rotation": False,
                "solid_color_pieces": True,
            }
        # Piece rotation was previously enabled at L4+, but pieces were
        # rendered with no orientation indicator (no arrow / original-up
        # marker), making the un-rotation task visually impossible and
        # driving passrate to 0%. Keeping rotation off makes the task
        # solvable: the model matches piece pixels to quadrants of each
        # candidate. Difficulty at higher levels is driven by n_pieces,
        # cut_type, and pattern complexity.
        return {
            "n_pieces": 4 if level < 7 else 6,
            "cut_type": ("straight" if level <= 3 else
                         "diagonal" if level <= 6 else
                         "irregular"),
            "piece_rotation": False,
            "solid_color_pieces": False,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1483)
        self._primary_complexity_feature = cfg["n_pieces"] * 2 + level

        n_pieces = cfg["n_pieces"]
        # Build a "full" pattern as a grid of colored cells
        # Size N x M so we have n_pieces blocks
        if n_pieces == 4:
            nr, nc = 2, 2
        else:
            nr, nc = 2, 3

        # The total grid has higher resolution to create visible patterns
        res = 4  # each piece is res x res cells
        grid_h, grid_w = nr * res, nc * res
        if cfg.get("solid_color_pieces"):
            # L0/L1: each piece is a SINGLE distinct solid color so that
            # piece-to-quadrant matching is a direct color lookup.
            piece_colors = sub_rng.sample(_COLORS, k=nr * nc)
            grid = [[piece_colors[(r // res) * nc + (c // res)]
                     for c in range(grid_w)] for r in range(grid_h)]
        else:
            grid = [[sub_rng.choice(_COLORS) for _ in range(grid_w)]
                    for _ in range(grid_h)]
            # Add simple pattern motifs
            for _ in range(sub_rng.randint(1, 3)):
                cr = sub_rng.randint(0, grid_h - 1)
                cc = sub_rng.randint(0, grid_w - 1)
                motif_col = sub_rng.choice(_COLORS)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r, c = cr + dr, cc + dc
                        if 0 <= r < grid_h and 0 <= c < grid_w:
                            grid[r][c] = motif_col

        # Divide into pieces. piece[i] = sub-grid.
        pieces = []
        for pr in range(nr):
            for pc in range(nc):
                piece_cells = [
                    [grid[pr * res + r][pc * res + c] for c in range(res)]
                    for r in range(res)
                ]
                pieces.append({
                    "cells": piece_cells,
                    "pos": (pr, pc),
                    "rotation": 0,
                    "cut_type": cfg["cut_type"],
                })

        # If rotation needed, rotate each displayed piece by a random multiple
        # of 90 (but the reconstructed answer must correspond to un-rotated)
        if cfg["piece_rotation"]:
            for p in pieces:
                p["rotation"] = 90 * sub_rng.randint(0, 3)

        # Label pieces A-D(-F)
        piece_labels = [chr(ord("A") + i) for i in range(len(pieces))]

        # Candidate assembled images
        # Correct: assemble using correct piece positions + correct rotations
        correct_grid = [row[:] for row in grid]

        candidates = [{"grid": correct_grid, "desc": "correct"}]

        if cfg.get("solid_color_pieces"):
            # L0/L1: pieces are solid colors. Distractors are 3 different
            # SWAPS (swapping different pairs / triples) producing visibly
            # different assembled grids.
            swaps = [
                [(0, 1)],                # swap pieces 0 and 1
                [(0, 2)],                # swap pieces 0 and 2
                [(0, 3)],                # swap pieces 0 and 3
                [(1, 2)],                # swap pieces 1 and 2
                [(1, 3)],                # swap pieces 1 and 3
                [(2, 3)],                # swap pieces 2 and 3
            ]
            sub_rng.shuffle(swaps)
            chosen = swaps[:3]
            for swap_pairs in chosen:
                d_grid = [row[:] for row in correct_grid]
                for (i, j) in swap_pairs:
                    if i >= len(pieces) or j >= len(pieces):
                        continue
                    pi_pos = pieces[i]["pos"]
                    pj_pos = pieces[j]["pos"]
                    for r in range(res):
                        for c in range(res):
                            d_grid[pi_pos[0] * res + r][pi_pos[1] * res + c] = \
                                pieces[j]["cells"][r][c]
                            d_grid[pj_pos[0] * res + r][pj_pos[1] * res + c] = \
                                pieces[i]["cells"][r][c]
                candidates.append({"grid": d_grid, "desc": "swapped"})
        else:
            # Distractor 1: swap two pieces
            swap_grid = [row[:] for row in correct_grid]
            if len(pieces) >= 2:
                i, j = sub_rng.sample(range(len(pieces)), 2)
                pi_pos = pieces[i]["pos"]
                pj_pos = pieces[j]["pos"]
                for r in range(res):
                    for c in range(res):
                        swap_grid[pi_pos[0] * res + r][pi_pos[1] * res + c] = pieces[j]["cells"][r][c]
                        swap_grid[pj_pos[0] * res + r][pj_pos[1] * res + c] = pieces[i]["cells"][r][c]
            candidates.append({"grid": swap_grid, "desc": "swapped"})

            # Distractor 2: rotate one piece by 180
            rot_grid = [row[:] for row in correct_grid]
            if pieces:
                idx = sub_rng.randint(0, len(pieces) - 1)
                pr_pos = pieces[idx]["pos"]
                rotated_cells = [[pieces[idx]["cells"][res - 1 - r][res - 1 - c]
                                  for c in range(res)]
                                 for r in range(res)]
                for r in range(res):
                    for c in range(res):
                        rot_grid[pr_pos[0] * res + r][pr_pos[1] * res + c] = rotated_cells[r][c]
            candidates.append({"grid": rot_grid, "desc": "rotated"})

            # Distractor 3: flip a piece horizontally
            flip_grid = [row[:] for row in correct_grid]
            if pieces:
                idx = sub_rng.randint(0, len(pieces) - 1)
                pf_pos = pieces[idx]["pos"]
                flipped_cells = [list(reversed(row)) for row in pieces[idx]["cells"]]
                for r in range(res):
                    for c in range(res):
                        flip_grid[pf_pos[0] * res + r][pf_pos[1] * res + c] = flipped_cells[r][c]
            candidates.append({"grid": flip_grid, "desc": "flipped"})

        # Ensure all candidates are distinct grids
        unique = []
        seen = set()
        for c in candidates:
            key = tuple(tuple(row) for row in c["grid"])
            if key not in seen:
                unique.append(c)
                seen.add(key)
        while len(unique) < 4:
            # Create an extra random-perturb distractor
            extra = {"grid": [row[:] for row in correct_grid], "desc": "noise"}
            r0 = sub_rng.randint(0, grid_h - 1)
            c0 = sub_rng.randint(0, grid_w - 1)
            extra["grid"][r0][c0] = sub_rng.choice(_COLORS)
            key = tuple(tuple(row) for row in extra["grid"])
            if key not in seen:
                unique.append(extra)
                seen.add(key)

        sub_rng.shuffle(unique)
        correct_idx = next(i for i, c in enumerate(unique)
                           if c["desc"] == "correct")
        answer_letter = chr(ord("A") + correct_idx)

        image = self._render(pieces, piece_labels, unique, nr, nc, res, sub_rng)

        # Build an explicit piece->position map for the question text so the
        # model knows the reassembly convention (previously implicit).
        pos_map_parts = []
        for i, p in enumerate(pieces):
            pr, pc = p["pos"]
            row_word = "row " + str(pr + 1)
            col_word = "column " + str(pc + 1)
            pos_map_parts.append(f"{piece_labels[i]}={row_word} {col_word}")
        pos_map = "; ".join(pos_map_parts)

        question = (
            f"{n_pieces} image pieces are shown in the top row, labeled "
            f"{', '.join(piece_labels)}. Each piece belongs at a fixed "
            f"position in a {nr}-row by {nc}-column grid: {pos_map}. "
            f"Four candidate assembled images are shown below, labeled "
            f"A, B, C, D. Which candidate correctly combines all pieces "
            f"at their proper positions (row 1 is the top row, column 1 "
            f"is the leftmost)? Answer with a single letter."
        )
        return question, answer_letter, image

    def _render(self, pieces, piece_labels, candidates, nr, nc, res,
                rng: random.Random) -> Image.Image:
        style = self._random_style()
        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_facecolor(style["bg_color"])
        n_pieces = len(pieces)
        gs = fig.add_gridspec(3, max(4, n_pieces),
                              height_ratios=[1.0, 0.08, 1.2],
                              hspace=0.18, wspace=0.15)

        # Top row: piece fragments
        for i in range(n_pieces):
            ax = fig.add_subplot(gs[0, i])
            self._draw_piece(ax, pieces[i], piece_labels[i],
                             cut_type=pieces[i]["cut_type"])

        # Separator row
        ax_sep = fig.add_subplot(gs[1, :])
        ax_sep.axis("off")
        ax_sep.text(0.5, 0.5, "--- Assembled candidates ---",
                    ha="center", va="center", fontsize=11, color="#7f8c8d",
                    style="italic", transform=ax_sep.transAxes)

        # Bottom row: 4 assembled candidates
        letters = ["A", "B", "C", "D"]
        for i, cand in enumerate(candidates[:4]):
            ax = fig.add_subplot(gs[2, i])
            self._draw_assembled(ax, cand["grid"], letters[i], nr, nc, res)

        fig.suptitle("Split Image Assembly",
                     fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.03)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_piece(self, ax, piece, label, cut_type="straight"):
        cells = piece["cells"]
        rot = piece["rotation"]
        # Apply rotation to cells
        for _ in range(rot // 90):
            cells = [list(r) for r in zip(*cells[::-1])]
        res = len(cells)
        ax.set_xlim(-0.3, res + 0.3)
        ax.set_ylim(-0.3, res + 0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=13, fontweight="bold", pad=4)
        for r in range(res):
            for c in range(res):
                rect = mpatches.Rectangle((c, res - 1 - r), 1, 1,
                                           fc=cells[r][c], ec="black", lw=0.5)
                ax.add_patch(rect)
        # Frame depending on cut_type
        if cut_type == "diagonal":
            ax.plot([-0.05, res + 0.05], [-0.05, -0.05], color="#2c3e50", lw=1.2)
            ax.plot([-0.05, res + 0.05], [res + 0.05, res + 0.05], color="#2c3e50", lw=1.2)
        elif cut_type == "irregular":
            # Jagged edges
            jag_x = [x / 2.0 for x in range(0, 2 * res + 1)]
            jag_y_top = [res + 0.1 + (0.15 if i % 2 == 0 else -0.05)
                         for i in range(len(jag_x))]
            ax.plot(jag_x, jag_y_top, color="#2c3e50", lw=1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_assembled(self, ax, grid, label, nr, nc, res):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        ax.set_xlim(-0.3, cols + 0.3)
        ax.set_ylim(-0.3, rows + 0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"({label})", fontsize=13, fontweight="bold", pad=4)
        for r in range(rows):
            for c in range(cols):
                rect = mpatches.Rectangle((c, rows - 1 - r), 1, 1,
                                           fc=grid[r][c], ec="black", lw=0.3)
                ax.add_patch(rect)
        # Show piece boundaries
        for pr in range(1, nr):
            ax.plot([0, cols], [rows - pr * res, rows - pr * res],
                    color="#2c3e50", lw=1.6)
        for pc in range(1, nc):
            ax.plot([pc * res, pc * res], [0, rows],
                    color="#2c3e50", lw=1.6)
        for spine in ax.spines.values():
            spine.set_visible(False)

if __name__ == "__main__":
    env = SplitImageAssemblyQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
