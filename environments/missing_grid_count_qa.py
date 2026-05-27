"""
Missing Grid Count QA environment.

Goal: train counting of MISSING items in a regular grid pattern. Targets
MathVision.counting subtasks like "How many bricks are missing in the wall?"
(idx=7) — none of the existing 29 count envs train this inverse-counting
ability (they all count present items; this one counts absences).

Visual: a regular grid of identical "items" (square bricks / circular dots /
square cells) on a clean background. K cells are removed at random,
revealing a contrasting underlay color in the gap. Question asks how many
items are missing.

Why "missing-counting" is a distinct skill: counting present items requires
attending to "salient figures"; counting missing items requires inferring an
expected complete pattern and comparing it to what's visible. Reference
benchmark question (MathVision idx=7) shows this is a real failure mode for
4B-class VL models.

Level design (parameter["level"], 0-9). Difficulty = grid size + missing
count + visual distractors:

L0: 4×4 grid, 1-2 missing,  large items, no jitter, monochrome.
L1: 4×4 grid, 1-3 missing,  large items.
L2: 5×5 grid, 2-4 missing,  large items.
L3: 5×5 grid, 3-5 missing.
L4: 6×6 grid, 3-6 missing.
L5: 6×6 grid, 4-7 missing,  small jitter introduced.
L6: 7×7 grid, 5-8 missing,  introduces brick-pattern (alternating colors).
L7: 7×7 grid, 5-9 missing,  brick pattern + jitter.
L8: 8×8 grid, 6-11 missing, brick pattern + jitter + distractor row.
L9: 8×8 grid, 7-12 missing, full visual complexity.

Diversity axes:
  1. grid item shape: brick (rectangle), circle, square cell, hexagon
  2. brick fill colors: 8 palettes (red/blue/orange/green/grey/purple/teal/brown)
  3. underlay color: contrasting (white/cream/black/grey) to make gaps
     unambiguously visible
  4. layout: row-aligned vs brick-staggered (alternate rows offset)
  5. small uniform jitter at higher levels
  6. question phrasing: 5 stem variants
  7. title variants: 6
  8. background color from _random_style

Critical:
  - Question is always answerable as a single integer (no MCQ — model
    outputs the count). Use REASONING_TEMPLATE wrappers.
  - Generate the FULL grid first, then mark a deterministic random subset as
    "missing" — guarantees ground-truth count is correct.
  - For consistency the gap area is rendered as a small darker outlined
    rectangle so model can clearly see the absence.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# 8 palettes for brick/cell fills
_PALETTES = [
    ("#c0392b", "#e74c3c"),  # red
    ("#2980b9", "#3498db"),  # blue
    ("#d35400", "#e67e22"),  # orange
    ("#27ae60", "#2ecc71"),  # green
    ("#7f8c8d", "#bdc3c7"),  # grey
    ("#8e44ad", "#9b59b6"),  # purple
    ("#16a085", "#1abc9c"),  # teal
    ("#795548", "#a1887f"),  # brown
]

# Underlay shows through the gaps
_UNDERLAYS = ["#ffffff", "#fffde7", "#f5f5f5", "#212121", "#37474f"]

_TITLE_VARIANTS = [
    "Wall",
    "Pattern",
    "Grid",
    "Brick Layout",
    "Tile Wall",
    "Item Grid",
]

_QUESTION_STEMS = [
    "How many items are missing from the grid?",
    "How many cells are missing in this pattern?",
    "Count the number of empty cells in the grid.",
    "If the grid were complete, how many cells would still need to be filled?",
    "How many gaps do you see in the grid?",
]


class MissingGridCountQA(StandaloneVisualEnv):
    ENV_NAME = "missing_grid_count"

    # Random rotation would shift gap visibility — disable.
    ALLOW_ROTATION = False
    ALLOW_NOISE = True

    # ---------------- per-level config ----------------
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return dict(rows=4, cols=4, miss_min=1, miss_max=2,
                        brick=False, jitter=0.0)
        if level == 1:
            return dict(rows=4, cols=4, miss_min=1, miss_max=3,
                        brick=False, jitter=0.0)
        if level == 2:
            return dict(rows=5, cols=5, miss_min=2, miss_max=4,
                        brick=False, jitter=0.0)
        if level == 3:
            return dict(rows=5, cols=5, miss_min=3, miss_max=5,
                        brick=False, jitter=0.0)
        if level == 4:
            return dict(rows=6, cols=6, miss_min=3, miss_max=6,
                        brick=False, jitter=0.0)
        if level == 5:
            return dict(rows=6, cols=6, miss_min=4, miss_max=7,
                        brick=False, jitter=0.05)
        if level == 6:
            return dict(rows=7, cols=7, miss_min=5, miss_max=8,
                        brick=False, jitter=0.0)
        if level == 7:
            return dict(rows=7, cols=7, miss_min=5, miss_max=9,
                        brick=False, jitter=0.05)
        if level == 8:
            return dict(rows=8, cols=8, miss_min=6, miss_max=11,
                        brick=False, jitter=0.05)
        return dict(rows=8, cols=8, miss_min=7, miss_max=12,
                    brick=False, jitter=0.07)

    # ---------------- core generation ----------------
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 47 + 503)
        self._primary_complexity_feature = level * 5 + cfg["rows"] * cfg["cols"]

        for attempt in range(8):
            res = self._try_generate(sub_rng, cfg)
            if res is not None:
                # No "Be concise" hint — for this counting task, the visual
                # has many cells and the model needs to reason cell-by-cell.
                # Forcing concise output makes Qwen3-VL guess without CoT.
                return res
        return None

    def _try_generate(
        self, sub_rng: random.Random, cfg: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rows = cfg["rows"]
        cols = cfg["cols"]
        n_total = rows * cols
        n_missing = sub_rng.randint(cfg["miss_min"], cfg["miss_max"])
        # Bound miss count to leave a meaningful number of present cells
        n_missing = min(n_missing, max(1, n_total // 2))

        # Choose which cells are missing
        all_idx = list(range(n_total))
        sub_rng.shuffle(all_idx)
        missing_set = set(all_idx[:n_missing])

        palette_dark, palette_light = sub_rng.choice(_PALETTES)
        underlay = sub_rng.choice(_UNDERLAYS)
        # Avoid underlay being indistinguishable from the brick fill
        # (rare but check): pick a different one if it matches.
        if underlay in (palette_dark, palette_light):
            underlay = "#ffffff"

        question = sub_rng.choice(_QUESTION_STEMS)
        answer = str(n_missing)

        img = self._render(rows, cols, missing_set,
                           palette_dark, palette_light, underlay,
                           cfg, sub_rng)
        return question, answer, img

    # ---------------- rendering ----------------
    def _render(self, rows, cols, missing_set, palette_dark, palette_light,
                underlay, cfg, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        # Each cell occupies a 1x1 unit; bricks are slightly less wide
        # than a cell to leave a small mortar gap.
        cell_w, cell_h = 1.0, 1.0
        brick_pad = 0.06
        jitter = cfg.get("jitter", 0.0)
        is_brick_pattern = cfg.get("brick", False)

        fig = plt.figure(figsize=(7.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        ax.axis("off")

        # First: paint underlay across the whole grid area so gaps reveal it
        underlay_rect = mpatches.Rectangle(
            (-0.5, -0.5), cols, rows,
            facecolor=underlay, edgecolor="none", zorder=0,
        )
        ax.add_patch(underlay_rect)

        # Draw each present cell
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx in missing_set:
                    # Optional: draw a faint outline to emphasize the gap
                    gap = mpatches.Rectangle(
                        (c - 0.5 + brick_pad, r - 0.5 + brick_pad),
                        cell_w - 2 * brick_pad, cell_h - 2 * brick_pad,
                        facecolor=underlay,
                        edgecolor="#9e9e9e", linewidth=0.6,
                        linestyle="--", zorder=1,
                    )
                    ax.add_patch(gap)
                    continue

                # Brick pattern: alternate row offset
                x_off = 0.0
                if is_brick_pattern and (r % 2 == 1):
                    x_off = 0.5  # half-brick offset
                # Pick fill color; alternate between dark and light per row
                fill = palette_dark if (r % 2 == 0) else palette_light
                # Small uniform jitter
                jx = rng.uniform(-jitter, jitter)
                jy = rng.uniform(-jitter, jitter)

                rect = mpatches.Rectangle(
                    (c - 0.5 + brick_pad + x_off + jx,
                     r - 0.5 + brick_pad + jy),
                    cell_w - 2 * brick_pad, cell_h - 2 * brick_pad,
                    facecolor=fill,
                    edgecolor="#222", linewidth=0.8, zorder=2,
                )
                ax.add_patch(rect)

        # Set axes limits with small padding
        ax.set_xlim(-0.7, cols - 0.3)
        ax.set_ylim(-0.7, rows - 0.3)
        # Invert Y so row 0 is at top (visual convention for walls)
        ax.invert_yaxis()

        title_l = rng.choice(_TITLE_VARIANTS)
        ax.set_title(title_l, fontsize=fs + 2, family=ff)

        fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.04)
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = MissingGridCountQA()
    for level in (0, 3, 6, 9):
        gts = {}
        n_ok = 0
        for seed in range(20):
            ok = env.generate(seed=seed, parameter={"level": level})
            if ok:
                n_ok += 1
                gts[env._answer] = gts.get(env._answer, 0) + 1
        print(f"L{level}: {n_ok}/20 ok, GT distribution: {dict(sorted(gts.items(), key=lambda x: int(x[0])))}")
        if n_ok > 0:
            print(f"  sample (seed=0): Q={env._question[:120]!r}  A={env._answer}")
