"""
Jigsaw Piece Match QA (Task B, multi-image P5 capability).

Show a grid with a missing piece and 4 candidate pieces as MCQ.
Ask which piece fills the gap. L0: only 1 fits (shape match).
L9: all similar shape, need pattern/color matching.

Difficulty axes:
  A) Grid size (3x3 -> 5x5)
  B) Pattern complexity (solid color -> gradient -> texture)
  C) Distractor similarity (random -> close match)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import BytesIO
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class JigsawPieceMatchQA(StandaloneVisualEnv):
    ENV_NAME = "jigsaw_piece_match"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "grid_size": 3 + level // 3,           # 3,3,3,4,4,4,5,5,5,5
            "pattern_mode": "solid" if level <= 2 else
                           "gradient" if level <= 5 else "texture",
            "distractor_similarity": min(0.9, 0.3 + level * 0.07),  # 0.3 -> 0.9
            "show_grid_lines": level <= 4,
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
        palette = self._random_style()["palette"]

        # Generate a grid of colored cells. 2026-04-25 redesign:
        # - L0/L1 (rule_mode="row_constant"): each row is a single color so
        #   the missing cell's color is the same as its row neighbors. Trivial.
        # - L2 (rule_mode="col_constant"): each column is a single color.
        # - L3-L5 (rule_mode="solid"): linear index cycle (current).
        # - L6+ (rule_mode="gradient"/"texture"): harder.
        if level <= 1:
            rule_mode = "row_constant"
        elif level == 2:
            rule_mode = "col_constant"
        elif cfg["pattern_mode"] == "solid":
            rule_mode = "solid"
        else:
            rule_mode = cfg["pattern_mode"]

        grid_colors = [[None] * gs for _ in range(gs)]
        if rule_mode == "row_constant":
            row_colors = [palette[i % len(palette)] for i in range(gs)]
            rng.shuffle(row_colors)
            for r in range(gs):
                for c in range(gs):
                    grid_colors[r][c] = row_colors[r]
        elif rule_mode == "col_constant":
            col_colors = [palette[i % len(palette)] for i in range(gs)]
            rng.shuffle(col_colors)
            for r in range(gs):
                for c in range(gs):
                    grid_colors[r][c] = col_colors[c]
        elif rule_mode == "solid":
            for r in range(gs):
                for c in range(gs):
                    grid_colors[r][c] = palette[(r * gs + c) % len(palette)]
        elif rule_mode == "gradient":
            for r in range(gs):
                for c in range(gs):
                    t = (r * gs + c) / (gs * gs)
                    grid_colors[r][c] = palette[int(t * (len(palette) - 1))]
        else:  # texture
            for r in range(gs):
                for c in range(gs):
                    grid_colors[r][c] = palette[(r + c) % len(palette)]

        # Pick a cell to remove
        mr, mc = rng.randint(0, gs - 1), rng.randint(0, gs - 1)
        correct_color = grid_colors[mr][mc]

        # Generate 4 candidate pieces (one correct, three distinct distractors).
        # CRITICAL: distractors must NOT equal correct_color or each other, else
        # the MCQ becomes ambiguous (multiple "correct" letters).
        candidates = [correct_color]
        attempts = 0
        while len(candidates) < 4 and attempts < 50:
            attempts += 1
            if cfg["distractor_similarity"] > 0.6:
                # Pick a neighboring cell's color
                nr, nc = mr + rng.choice([-1, 0, 1]), mc + rng.choice([-1, 0, 1])
                nr = max(0, min(gs - 1, nr))
                nc = max(0, min(gs - 1, nc))
                if (nr, nc) != (mr, mc):
                    cand = grid_colors[nr][nc]
                else:
                    cand = rng.choice(palette)
            else:
                cand = rng.choice(palette)
            if cand not in candidates:
                candidates.append(cand)
        # Fallback: fill remaining slots with any unused palette color
        for c in palette:
            if len(candidates) >= 4:
                break
            if c not in candidates:
                candidates.append(c)
        if len(candidates) < 4:
            return None

        # Shuffle and find answer (candidates are guaranteed distinct above)
        rng.shuffle(candidates)
        answer_idx = candidates.index(correct_color)
        # Safety: if correct color appears more than once somehow, abort
        if candidates.count(correct_color) != 1:
            return None
        answer_letter = chr(ord("A") + answer_idx)

        # Render
        img = self._render_puzzle(grid_colors, mr, mc, candidates, gs, cfg, rng)

        templates = [
            f"A {gs}x{gs} grid is shown with one piece missing (marked '?'). Four candidate pieces (A, B, C, D) are shown below. Which piece correctly fills the gap? Answer with a single letter (A, B, C, or D).",
            f"In the image above, a {gs}x{gs} jigsaw grid has one missing cell labelled '?'. Below it are four candidate pieces. Which option (A-D) correctly completes the grid? Reply with one letter only.",
            f"The puzzle shows a {gs}x{gs} colored grid with one cell marked '?'. From the four candidate pieces (A, B, C, D) shown, identify which one belongs in the missing slot. Answer with a single letter.",
            f"Look at the {gs}x{gs} grid: the cell marked '?' must be filled. Pick the correct candidate piece from A, B, C, D. Single letter answer.",
            f"The {gs}x{gs} grid has a missing piece (the '?' cell). Choose the candidate (A, B, C, or D) that should occupy that position. Provide only the letter.",
        ]
        question = rng.choice(templates)
        # 2026-04-25: hint at low levels — point to the row/column rule.
        if level <= 1:
            question += (
                " Hint: notice that each row of the grid is a single color. "
                "The missing cell's color is the same as the other cells "
                "in its row."
            )
        elif level == 2:
            question += (
                " Hint: notice that each column of the grid is a single "
                "color. The missing cell's color is the same as the other "
                "cells in its column."
            )

        return question, answer_letter, img

    def _render_puzzle(self, grid_colors, mr, mc, candidates, gs, cfg, rng):
        style = self._random_style()
        fig = plt.figure(figsize=(8, 10))
        fig.patch.set_facecolor(style["bg_color"])

        # Grid axes
        ax_grid = fig.add_axes([0.1, 0.35, 0.8, 0.6])
        ax_grid.set_xlim(0, gs)
        ax_grid.set_ylim(0, gs)
        ax_grid.set_aspect("equal")
        ax_grid.set_facecolor(style["bg_color"])
        ax_grid.axis("off")
        title_pool = [
            "Jigsaw Grid (find the missing piece)",
            "Color Puzzle Grid",
            "Pattern Grid - Missing Cell",
            "Find the Missing Piece",
            "Tile Puzzle",
        ]
        ax_grid.set_title(rng.choice(title_pool), fontsize=13, fontweight="bold")

        # Use random missing-cell question mark color
        q_color = rng.choice(["red", "#c0392b", "#e67e22", "#16a085", "#8e44ad"])
        for r in range(gs):
            for c in range(gs):
                if r == mr and c == mc:
                    # Missing piece
                    rect = mpatches.Rectangle((c, gs - 1 - r), 1, 1,
                                               fc="#dddddd", ec="black",
                                               lw=1.5, linestyle="--")
                    ax_grid.add_patch(rect)
                    ax_grid.text(c + 0.5, gs - 0.5 - r, "?",
                                fontsize=18, ha="center", va="center",
                                fontweight="bold", color=q_color)
                else:
                    rect = mpatches.Rectangle((c, gs - 1 - r), 1, 1,
                                               fc=grid_colors[r][c], ec="black",
                                               lw=0.8 if cfg["show_grid_lines"] else 0.3)
                    ax_grid.add_patch(rect)

        # Candidate pieces — randomize candidate shape per render
        ax_cands = fig.add_axes([0.1, 0.05, 0.8, 0.22])
        ax_cands.set_xlim(0, 8)
        ax_cands.set_ylim(0, 2)
        ax_cands.set_aspect("equal")
        ax_cands.set_facecolor(style["bg_color"])
        ax_cands.axis("off")
        cand_title_pool = [
            "Candidate Pieces", "Options", "Pick One", "Choices A-D",
            "Candidate Tiles",
        ]
        ax_cands.set_title(rng.choice(cand_title_pool),
                           fontsize=11, fontweight="bold")

        cand_shape = rng.choice(["square", "rounded", "circle"])
        for i, color in enumerate(candidates):
            x = 0.5 + i * 2
            if cand_shape == "circle":
                p = mpatches.Circle((x + 0.6, 0.9), 0.55, fc=color,
                                     ec="black", lw=1.5)
            elif cand_shape == "rounded":
                p = mpatches.FancyBboxPatch((x, 0.3), 1.2, 1.2,
                                             boxstyle="round,pad=0.02",
                                             fc=color, ec="black", lw=1.5)
            else:
                p = mpatches.Rectangle((x, 0.3), 1.2, 1.2,
                                        fc=color, ec="black", lw=1.5)
            ax_cands.add_patch(p)
            label = chr(ord("A") + i)
            ax_cands.text(x + 0.6, 0.05, label, fontsize=12,
                         ha="center", va="center", fontweight="bold")

        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskb"
    os.makedirs(out_dir, exist_ok=True)
    env = JigsawPieceMatchQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[jigsaw_piece_match L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"jigsaw_piece_match_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[jigsaw_piece_match L{level} s{s}] A={env._answer}")
