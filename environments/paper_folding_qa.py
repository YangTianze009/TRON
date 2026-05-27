"""
Paper folding QA environment.

2026-05-05 R5 P1: REWRITTEN to align question phrasing and option style
with the Basic Transformations / Cutting and Combining of Figures family
(named-cell overlap, count-overlap, and silhouette figure-MCQ).

Pure single-letter MCQ across all levels (A through D / A through E).
Verifier delegates to the base class which already handles letter
extraction (3 wrappers).

Per-level design:
  L0/L1 — 4-MCQ "after one fold, which named cell coincides with the
    shaded cell?" Cells labeled A-G on a small grid; answer is the named
    cell letter that overlaps the shaded one after the fold.
    L0 uses no trap, L1 ≤10% trap.
  L2/L4 — 4-MCQ "how many small unit cells overlap after one fold?"
    options are integer counts + "No correct answer"; uses ruler-like
    grid with labeled crease.
  L5/L7 — figure-MCQ: "what is the silhouette of the unfolded paper after
    these 2 folds + 1 cut/punch?" — render 4 candidate silhouettes in a
    2x2 panel labeled A/B/C/D, model picks the matching one.
  L8/L9 — same figure-MCQ style but with 3 folds, tighter distractors,
    and 40% "E. No correct answer" trap rate.

Naming conventions:
  - Crease lines drawn dashed, labeled (e.g. "crease").
  - Cells labeled A, B, C, D, E, F.
  - Always grid-based with axis ticks.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ====================================================================== #
# Cell-grid helpers — used by L0-L4 modes
# ====================================================================== #

def _cell_center(col, row, w=1.0, h=1.0):
    return (col * w + w / 2, row * h + h / 2)


def _reflect_cell_v(col, row, axis_col, n_cols):
    """Reflect cell (col, row) across vertical axis at column boundary axis_col."""
    new_col = 2 * axis_col - 1 - col  # boundary at axis_col, mirror by axis
    return (new_col, row)


def _reflect_cell_h(col, row, axis_row, n_rows):
    new_row = 2 * axis_row - 1 - row
    return (col, new_row)


# ====================================================================== #
# Silhouette helpers — used by L5-L9 figure-MCQ
# ====================================================================== #

# A silhouette is a set of (col, row) integer cells; we render it as a grid
# of filled squares.

def _silhouette_after_fold_and_punch(cells_per_side, fold_seq, punch_cells):
    """Simulate folding a `cells_per_side x cells_per_side` square grid
    along axes given by fold_seq, then punching `punch_cells` (a set of
    (col, row) inside the FOLDED region), then unfolding to find the union
    of all hole positions on the original sheet.

    fold_seq: list of ('v', axis_col) | ('h', axis_row) representing folds.
              After a fold, the region's column or row range halves.

    Returns: a frozenset of (col, row) cells punched on the unfolded sheet.
    """
    # We track the active region with bounds. After each fold, the region
    # shrinks. When unfolding, each punch produces 2^n images (one per fold)
    # where each is reflected across the fold's axis.
    # Here we store, for each punch cell, the list of images after unfolding.
    n = cells_per_side
    # At fold time, the punch is in the folded region. We unfold step by step,
    # reflecting each image across the fold axis to produce its mirror image.
    images = list(punch_cells)
    for fold in reversed(fold_seq):
        kind, axis = fold
        new_images = []
        for (c, r) in images:
            new_images.append((c, r))
            if kind == "v":
                new_images.append((2 * axis - 1 - c, r))
            else:
                new_images.append((c, 2 * axis - 1 - r))
        images = list({pt for pt in new_images})
    return frozenset(images)


# ====================================================================== #
# Main environment class
# ====================================================================== #

class PaperFoldingQA(StandaloneVisualEnv):
    """Pure single-letter MCQ environment for paper-folding reasoning."""

    ALLOW_ROTATION = False  # orientation-sensitive
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "paper_folding"

    QUESTION_TYPES = [
        "fold_overlap_cell",       # L0/L1 — Q4-style 4-MCQ over named cells
        "fold_overlap_count",      # L2/L4 — 4-MCQ count of overlapping cells
        "unfold_silhouette",       # L5-L9 — 2-3 fold figure-MCQ silhouette
    ]

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "qtype": "fold_overlap_cell",
                "trap_rate": 0.10 if level == 1 else 0.0,
            }
        if level <= 4:
            return {
                "qtype": "fold_overlap_count",
                "use_5_options": (level == 4),
                "trap_rate": 0.0 if level <= 3 else 0.15,
                "n_cells": 4,
            }
        if level <= 6:
            return {
                "qtype": "unfold_silhouette",
                "n_folds": 2,
                "use_5_options": False,
                "tight_distractors": False,
                "trap_rate": 0.20,
                "cells_per_side": 4,
            }
        if level == 7:
            return {
                "qtype": "unfold_silhouette",
                "n_folds": 2,
                "use_5_options": True,
                "tight_distractors": False,
                "trap_rate": 0.25,
                "cells_per_side": 4,
            }
        # L8 / L9 — 3 folds + tight distractors + raised "No correct" trap
        return {
            "qtype": "unfold_silhouette",
            "n_folds": 3,
            "use_5_options": True,
            "tight_distractors": True,
            "trap_rate": 0.40,
            "cells_per_side": 4,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 7717)
        qtype = cfg["qtype"]

        for _ in range(20):
            if qtype == "fold_overlap_cell":
                r = self._try_fold_overlap_cell(cfg, sub_rng)
            elif qtype == "fold_overlap_count":
                r = self._try_fold_overlap_count(cfg, sub_rng)
            else:
                r = self._try_unfold_silhouette(cfg, sub_rng)
            if r is not None:
                return r
        return None

    # ================================================================== #
    # L0/L1 — Q4-style 4-MCQ overlap question
    #   Layout: 1xN row of named cells A..F (one shaded). After folding
    #   along the labeled vertical crease, which named cell ends up
    #   ON TOP of (overlapping with) the shaded cell?
    # ================================================================== #
    def _try_fold_overlap_cell(self, cfg, rng):
        # Use a 1x6 row of named cells A..F. Fold along a vertical
        # crease at column boundary 3 (mid-line) so left half (cols 0..2)
        # folds onto right half (cols 3..5).
        n_cols = 6
        labels = ["A", "B", "C", "D", "E", "F"]
        crease = 3  # boundary between col 2 and col 3
        # Choose a "shaded" cell on the LEFT side (cols 0..2). After the
        # fold, that cell flips to col (2*crease - 1 - col) = 5-col.
        shaded_col = rng.choice([0, 1, 2])
        shaded_label = labels[shaded_col]
        target_col = 5 - shaded_col
        target_label = labels[target_col]

        use_trap = (rng.random() < cfg.get("trap_rate", 0.0))

        # Build options — 4 named cells total (A-D). Always present 4 cells
        # from the 6 labels: include `target_label` (correct) plus 3 others.
        opt_letters = ["A", "B", "C", "D"]
        # Candidate pool excludes the shaded cell (since you can't say a
        # cell folds onto itself)
        pool = [l for l in labels if l != shaded_label]
        rng.shuffle(pool)

        if use_trap:
            # Remove target from pool so correct = "D. No correct answer"
            distractors = [l for l in pool if l != target_label][:3]
            opt_values = distractors[:]  # 3 distractors only
            # Build 4-letter MCQ where last is "No correct answer"
            rng.shuffle(opt_values)
            correct_letter = "D"
            opt_lines_data = list(zip(opt_letters[:3], opt_values))
            opt_lines = "\n".join(f"{lt}. {v}" for lt, v in opt_lines_data)
            opt_lines += "\nD. No correct answer"
        else:
            distractors = [l for l in pool if l != target_label][:3]
            options = [target_label] + distractors
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(target_label)]
            opt_lines = "\n".join(f"{lt}. {v}"
                                  for lt, v in zip(opt_letters, options))

        # Render: 1x6 row of cells, shaded one filled, crease drawn vertical
        image = self._render_cell_row_fold(
            n_cols=n_cols, labels=labels, shaded_col=shaded_col,
            crease_col=crease, rng=rng,
        )

        stems = [
            "As shown in the diagram, the strip of paper has labeled cells "
            f"{', '.join(labels[:-1])}, and {labels[-1]}. The shaded cell "
            f"is {shaded_label}. After folding the strip along the labeled "
            "vertical crease, which cell ends up directly ON TOP of (i.e. "
            f"coincides with) cell {shaded_label}?",
            "The strip below shows cells labeled A through F, with cell "
            f"{shaded_label} shaded. The dashed vertical line is the "
            f"crease. After folding the left half of the strip onto the "
            "right half along this crease, which other cell will land "
            f"directly on top of cell {shaded_label}?",
            "As shown in the figure, the rectangle is divided into 6 "
            "labeled cells (A-F). The crease (dashed vertical line) is "
            f"the fold line; the shaded cell is {shaded_label}. When the "
            "strip is folded along the crease, which cell coincides "
            f"exactly with cell {shaded_label}?",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            "Answer with a single letter A, B, C, or D."
        )
        return question, correct_letter, image

    # ================================================================== #
    # L2-L4 — 4-MCQ count of overlapping cells
    #   Layout: 2xN grid of small cells; some cells shaded. After folding
    #   along labeled crease, count the number of fully-overlapping cell
    #   pairs (i.e. shaded-on-top-of-shaded).
    # ================================================================== #
    def _try_fold_overlap_count(self, cfg, rng):
        n_cells = cfg.get("n_cells", 4)
        # Use a 2 x (2 * n_cells) grid; horizontal mid-line is the crease.
        n_cols = 2 * n_cells
        n_rows = 2
        crease_row = 1  # boundary between row 0 and row 1
        # Each cell is shaded with prob ~0.4
        shaded = set()
        # Force at least 2 shaded cells for visual clarity
        for _ in range(8):
            shaded = set()
            for c in range(n_cols):
                for r in range(n_rows):
                    if rng.random() < 0.45:
                        shaded.add((c, r))
            if 2 <= len(shaded) <= n_cols * n_rows - 1:
                break
        if len(shaded) < 2:
            return None

        # Count overlap pairs: cell (c, 0) overlaps with (c, 1) after fold
        overlap_count = 0
        for c in range(n_cols):
            if (c, 0) in shaded and (c, 1) in shaded:
                overlap_count += 1
        # Sometimes overlap is 0; that's a valid answer.

        use_5 = cfg.get("use_5_options", False)
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_val_options = len(opt_letters) - 1  # last reserved for "No correct"
        n_distractor = n_val_options - 1

        # Build distractors as integers near `overlap_count`
        max_possible = n_cols
        cand = set()
        for delta in (1, -1, 2, -2, 3, -3):
            v = overlap_count + delta
            if 0 <= v <= max_possible and v != overlap_count:
                cand.add(v)
        # Add boundary distractors
        cand.add(0)
        cand.add(max_possible)
        cand.discard(overlap_count)
        # Use sorted-then-shuffle for determinism
        cand = list(cand)
        rng.shuffle(cand)
        distractors = cand[:n_distractor]
        if len(distractors) < n_distractor:
            return None

        use_trap = (rng.random() < cfg.get("trap_rate", 0.0))
        if use_trap and len(cand) > n_distractor:
            # Drop the correct value; final option = "No correct answer"
            options_data = cand[:n_val_options]  # all wrong values
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            options_data = [overlap_count] + distractors
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(overlap_count)]

        opt_lines_parts = [f"{opt_letters[i]}. {options_data[i]}"
                           for i in range(len(options_data))]
        opt_lines_parts.append(f"{opt_letters[-1]}. No correct answer")
        opt_lines = "\n".join(opt_lines_parts)
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Render
        image = self._render_grid_fold_count(
            n_cols=n_cols, n_rows=n_rows, shaded=shaded,
            crease_row=crease_row, rng=rng,
        )
        stems = [
            "As shown in the diagram, the rectangular paper is divided "
            f"into a {n_rows}-row grid of small unit cells. The shaded "
            "cells are filled. The dashed horizontal line is the crease. "
            "After folding the bottom half of the paper onto the top half "
            "along this crease, how many small cells will have a shaded "
            "cell on TOP of another shaded cell (i.e. how many cell "
            "positions show shaded-over-shaded overlap)?",
            "The figure shows a paper grid with shaded cells. The dashed "
            "horizontal line is a fold crease. When the lower half is "
            "folded up along this crease, count the number of small cell "
            "positions where shaded overlaps shaded.",
            "Below is a paper divided into a small-cell grid. Some cells "
            "are shaded (filled). The dashed line is the crease. After "
            "folding the bottom strip onto the top strip along the "
            "crease, how many cell positions show shaded-on-shaded "
            "overlap?",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, image

    # ================================================================== #
    # L5-L9 — figure-MCQ: predict unfolded silhouette after folds + punch
    #   Render 4 candidate silhouettes labeled A/B/C/D in a 2x2 panel grid
    #   below the question; correct one is the actual unfolded result.
    #   At L8/L9 with 5-MCQ trap, all 4 candidates can be wrong (E = NCA).
    # ================================================================== #
    def _try_unfold_silhouette(self, cfg, rng):
        n = cfg.get("cells_per_side", 4)
        n_folds = cfg.get("n_folds", 2)
        use_5 = cfg.get("use_5_options", False)
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]

        # Build a fold sequence. Each fold halves the active region. We
        # pick fold axes at the middle of the current region so the fold
        # produces an even halving (which keeps cell counts integer).
        # Active region is tracked as (cmin, cmax, rmin, rmax) inclusive.
        cmin, cmax, rmin, rmax = 0, n - 1, 0, n - 1
        fold_seq = []
        # Track the "active folded region" which is the smaller post-fold side.
        for _ in range(n_folds):
            cur_w = cmax - cmin + 1
            cur_h = rmax - rmin + 1
            # Try horizontal vs vertical fold; pick the longer dim
            if cur_w >= cur_h and cur_w >= 2:
                # Vertical fold: axis at column boundary cmin + cur_w/2
                axis = cmin + cur_w // 2
                fold_seq.append(("v", axis))
                # After fold, active region is the top/right half; pick
                # cells from cols [axis .. cmax]. Old cells in cols
                # [cmin .. axis-1] reflect onto cols [axis .. axis + (axis-cmin) - 1].
                cmin = axis
            elif cur_h >= 2:
                axis = rmin + cur_h // 2
                fold_seq.append(("h", axis))
                rmin = axis
            else:
                break  # can't fold further

        if len(fold_seq) < n_folds:
            return None

        # Pick a punch cell within the final folded region. Use 1 punch.
        # The punch represents a single hole made through all layers.
        for _ in range(10):
            punch_col = rng.randint(cmin, cmax)
            punch_row = rng.randint(rmin, rmax)
            # Ensure the unfolded silhouette has more than 1 hole (interesting).
            unfolded = _silhouette_after_fold_and_punch(
                n, fold_seq, [(punch_col, punch_row)],
            )
            if len(unfolded) >= 2:
                break
        else:
            return None

        true_silhouette = unfolded
        # Correct silhouette = `true_silhouette`. Build distractor silhouettes:
        #   - swap one punch cell mirror direction
        #   - drop a fold from the unfold step (so silhouette has fewer holes)
        #   - rotate 90° (orientation distractor)
        #   - random reflection
        distractors = self._gen_silhouette_distractors(
            true_silhouette, n, fold_seq, (punch_col, punch_row), rng,
            n_needed=4,  # generate up to 4 distractors; we only need n_distr
            tight=cfg.get("tight_distractors", False),
        )

        n_val_options = len(opt_letters) - 1  # excluding final "No correct"
        n_distractor = n_val_options - 1

        if len(distractors) < n_distractor:
            return None

        use_trap = (rng.random() < cfg.get("trap_rate", 0.0))
        if use_trap and len(distractors) >= n_val_options:
            # All listed candidates are wrong; correct = last letter "No correct"
            options_data = distractors[: n_val_options]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            options_data = [true_silhouette] + distractors[: n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(true_silhouette)]

        # Render: 2-panel sequence (original + folded with punch) + 2x2 grid
        # of candidate silhouettes labeled A/B/C/D (and E text option).
        image = self._render_silhouette_mcq(
            n=n, fold_seq=fold_seq, punch_cell=(punch_col, punch_row),
            options=options_data, opt_letters=opt_letters[:-1],
            include_e_text=use_5, rng=rng,
        )

        # Build option labels — the candidate IMAGES carry the visual.
        opt_text_lines = "\n".join(
            f"{l}. (see figure labeled {l})"
            for l in opt_letters[:-1]
        )
        if use_5:
            opt_text_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stems = [
            f"As shown in the diagram, a square paper divided into {n}x{n} "
            f"unit cells is folded {n_folds} time(s) along the labeled "
            f"creases (top panels). After folding, a single hole is "
            "punched through all layers at the marked cell. When the "
            "paper is fully unfolded, which of the four candidate "
            "silhouettes (labeled A, B, C, D) shows the resulting hole "
            "pattern?",
            f"The figure shows a {n}x{n} square paper that has been "
            f"folded {n_folds} time(s) (top), then a single hole is "
            "punched through all layers at the marked position. The "
            "bottom panel shows four candidate unfolded patterns "
            "labeled A through D. Which one matches the actual unfolded "
            "hole pattern?",
            f"A {n}x{n} unit-cell paper is folded {n_folds} time(s) along "
            "the indicated creases and then a single hole is punched "
            "through all layers at the marked location. Among the four "
            "candidate unfolded patterns shown (A, B, C, D), which one "
            "is the actual unfolded silhouette of holes?",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_text_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, image

    # ================================================================== #
    # Distractor generation for silhouette MCQ
    # ================================================================== #
    def _gen_silhouette_distractors(self, true_silhouette, n, fold_seq,
                                     punch_cell, rng, n_needed=4,
                                     tight=False):
        """Build up to `n_needed` plausible-but-wrong silhouette frozensets.

        Strategies (in order of preference):
          (a) Drop one fold: imagine only n-1 folds were applied — fewer
              mirror images.
          (b) Apply only mirror reflection across one fold axis (not all).
          (c) 90-deg rotation of true silhouette.
          (d) Off-by-one punch shift (different cell punched).
          (e) Reflect entire true silhouette across center.
        """
        candidates = []
        # (a) drop one fold from each end
        if len(fold_seq) >= 2:
            for skip_idx in range(len(fold_seq)):
                shorter = [f for i, f in enumerate(fold_seq) if i != skip_idx]
                cand = _silhouette_after_fold_and_punch(n, shorter, [punch_cell])
                if cand != true_silhouette and cand not in candidates:
                    candidates.append(cand)

        # (b) only first fold's mirror
        if len(fold_seq) >= 2:
            first_only = fold_seq[:1]
            cand = _silhouette_after_fold_and_punch(n, first_only, [punch_cell])
            if cand != true_silhouette and cand not in candidates:
                candidates.append(cand)

        # (c) 90° rotation
        c, r = punch_cell
        rot90 = []
        for (cc, rr) in true_silhouette:
            # rotate 90° clockwise about center: (cc,rr) -> (rr, n-1-cc)
            rot90.append((rr, n - 1 - cc))
        cand = frozenset(rot90)
        if cand != true_silhouette and cand not in candidates:
            candidates.append(cand)

        # (d) off-by-one punch shift
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_punch = (c + dc, r + dr)
            if 0 <= new_punch[0] < n and 0 <= new_punch[1] < n:
                cand = _silhouette_after_fold_and_punch(n, fold_seq, [new_punch])
                if cand != true_silhouette and cand not in candidates:
                    candidates.append(cand)

        # (e) reflect across center
        refl = [(n - 1 - cc, n - 1 - rr) for (cc, rr) in true_silhouette]
        cand = frozenset(refl)
        if cand != true_silhouette and cand not in candidates:
            candidates.append(cand)

        # Tight distractors: prefer those with same cell count as true
        if tight:
            true_n = len(true_silhouette)
            tight_pool = [c for c in candidates if len(c) == true_n]
            other_pool = [c for c in candidates if len(c) != true_n]
            rng.shuffle(tight_pool); rng.shuffle(other_pool)
            candidates = tight_pool + other_pool
        else:
            rng.shuffle(candidates)

        return candidates[:n_needed]

    # ================================================================== #
    # Drawing helpers
    # ================================================================== #
    def _render_cell_row_fold(self, n_cols, labels, shaded_col, crease_col,
                              rng) -> Image.Image:
        """L0/L1: render 1xN strip with labeled cells, shaded one filled,
        crease as dashed vertical line."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 2.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        palette = style["palette"]
        for c in range(n_cols):
            face = palette[0] if c == shaded_col else "white"
            alpha = 0.65 if c == shaded_col else 1.0
            rect = Rectangle((c, 0), 1, 1, facecolor=face, edgecolor='black',
                             linewidth=2, alpha=alpha)
            ax.add_patch(rect)
            ax.text(c + 0.5, 0.5, labels[c], ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')

        # Draw crease as dashed vertical line at boundary
        ax.axvline(crease_col, color='red', linestyle='--', linewidth=2.0,
                   alpha=0.85)
        # crease label
        ax.text(crease_col + 0.05, 1.2, "crease",
                fontsize=11, fontweight='bold', color='red')

        ax.set_xlim(-0.4, n_cols + 0.4)
        ax.set_ylim(-0.4, 1.6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xticks(range(n_cols + 1))
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_grid_fold_count(self, n_cols, n_rows, shaded, crease_row,
                                 rng) -> Image.Image:
        """L2-L4: render 2xN grid with shaded cells filled, crease line."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 3 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        palette = style["palette"]
        shade_color = palette[0]
        for c in range(n_cols):
            for r in range(n_rows):
                face = shade_color if (c, r) in shaded else "white"
                alpha = 0.7 if (c, r) in shaded else 1.0
                rect = Rectangle((c, r), 1, 1, facecolor=face,
                                 edgecolor='black', linewidth=1.6,
                                 alpha=alpha)
                ax.add_patch(rect)
        # crease horizontal line
        ax.axhline(crease_row, color='red', linestyle='--', linewidth=2.0,
                   alpha=0.85)
        ax.text(n_cols * 0.02, crease_row + 0.08, "crease",
                fontsize=11, fontweight='bold', color='red')

        ax.set_xlim(-0.4, n_cols + 0.4)
        ax.set_ylim(-0.4, n_rows + 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_silhouette_mcq(self, n, fold_seq, punch_cell, options,
                                opt_letters, include_e_text, rng
                                ) -> Image.Image:
        """L5-L9: render 4-panel silhouette MCQ.

        Top row: original square (left) + folded square with punch marked (right)
        Bottom rows: 2x2 grid of candidate silhouettes labeled A/B/C/D
        """
        style = self._random_style()
        sc = style["figsize_scale"]
        # 3 rows: top (instruction panels) + 2 rows of candidates (2 each)
        fig = plt.figure(figsize=(8 * sc, 9 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0],
                              hspace=0.35, wspace=0.25)

        palette = style["palette"]
        shade_color = palette[0]

        # Top-left: original n x n grid with crease lines for every fold
        ax_orig = fig.add_subplot(gs[0, 0])
        self._draw_grid(ax_orig, n, shaded_cells=set(), title="1. Original",
                        style=style)
        # Draw all fold lines
        for kind, axis in fold_seq:
            if kind == "v":
                ax_orig.axvline(axis, color='red', linestyle='--',
                                linewidth=1.6, alpha=0.85)
            else:
                ax_orig.axhline(axis, color='red', linestyle='--',
                                linewidth=1.6, alpha=0.85)

        # Top-right: folded paper bounds + punch marked
        ax_fold = fig.add_subplot(gs[0, 1])
        # Compute folded region bounds
        cmin, cmax, rmin, rmax = 0, n - 1, 0, n - 1
        for kind, axis in fold_seq:
            if kind == "v":
                cmin = axis
            else:
                rmin = axis
        # Draw the folded region as a smaller grid; mark punch cell
        self._draw_folded_region(ax_fold, cmin, cmax, rmin, rmax,
                                  punch_cell=punch_cell,
                                  shade_color=shade_color, style=style,
                                  title=f"2. After {len(fold_seq)} fold(s) + punch")

        # Bottom 2x2: candidate silhouettes A/B/C/D
        # options is up to 4 frozensets
        # opt_letters is e.g. ["A","B","C","D"]
        for i in range(min(4, len(options))):
            row = 1 + (i // 2)
            col = i % 2
            ax_opt = fig.add_subplot(gs[row, col])
            label = opt_letters[i] if i < len(opt_letters) else "?"
            self._draw_grid(ax_opt, n, shaded_cells=options[i],
                            title=f"({label})", style=style,
                            shade_color=shade_color)

        # If fewer than 4 options (shouldn't happen here), fill blanks
        for i in range(len(options), 4):
            row = 1 + (i // 2)
            col = i % 2
            ax_blank = fig.add_subplot(gs[row, col])
            ax_blank.axis('off')

        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _draw_grid(ax, n, shaded_cells, title, style, shade_color=None):
        """Draw an n x n grid with `shaded_cells` filled."""
        if shade_color is None:
            shade_color = style["palette"][0]
        for c in range(n):
            for r in range(n):
                face = shade_color if (c, r) in shaded_cells else "white"
                alpha = 0.75 if (c, r) in shaded_cells else 1.0
                rect = Rectangle((c, r), 1, 1, facecolor=face,
                                 edgecolor='black', linewidth=1.0,
                                 alpha=alpha)
                ax.add_patch(rect)
        ax.set_xlim(-0.2, n + 0.2)
        ax.set_ylim(-0.2, n + 0.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=11, fontweight='bold')

    @staticmethod
    def _draw_folded_region(ax, cmin, cmax, rmin, rmax, punch_cell,
                              shade_color, style, title):
        """Draw the folded paper as a small grid covering the folded region.
        Mark the punch_cell with a hole symbol."""
        # Show the folded region as cells from (cmin..cmax, rmin..rmax)
        for c in range(cmin, cmax + 1):
            for r in range(rmin, rmax + 1):
                rect = Rectangle((c, r), 1, 1, facecolor='#fef9e7',
                                 edgecolor='black', linewidth=1.2)
                ax.add_patch(rect)
        # Mark punch cell with a black filled circle
        pc, pr = punch_cell
        ax.plot(pc + 0.5, pr + 0.5, 'o', color='black', markersize=12)
        # crosshair "punch" symbol
        ax.text(pc + 0.5, pr + 0.5, '✕', ha='center', va='center',
                fontsize=14, color='white', fontweight='bold')
        ax.set_xlim(cmin - 0.5, cmax + 1.5)
        ax.set_ylim(rmin - 0.5, rmax + 1.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10, fontweight='bold')

    # ================================================================== #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """All answers in this rewrite are single-letter MCQ. Delegate to
        base class which already handles letter extraction."""
        return super()._check_answer(predicted, ground_truth)
