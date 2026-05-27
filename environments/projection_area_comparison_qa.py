"""
Projection Area Comparison QA.

A 3D block structure with three orthographic projections. Asks which has
the largest area or the exact area count.

Difficulty axes:
  A) n_cubes: 4..8
  B) question_type: MCQ comparison -> exact integer
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._template_lib import title_pool
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# 16 MCQ phrasings. Options list is appended by caller.
_MCQ_TEMPLATES = [
    "Which projection has the largest area?\n{opts}",
    "Which of the three projections shown has the greatest area?\n{opts}",
    "In the figure, which projection covers the largest area?\n{opts}",
    "Looking at the three views, which has the greatest projected area?\n{opts}",
    "Identify the projection with the largest area.\n{opts}",
    "Select the projection that has the greatest area.\n{opts}",
    "Determine which of the three projections is the largest.\n{opts}",
    "Pick the projection that covers the most grid squares.\n{opts}",
    "The projection with the largest area is ___.\n{opts}",
    "Among the three views, the one with the greatest area is ___.\n{opts}",
    "Let X be the projection with the largest area. X = ?\n{opts}",
    "The largest projection is labelled ___.\n{opts}",
    "Examine the three projections and decide which has the largest area.\n{opts}",
    "Compare the areas of the three projections; which is biggest?\n{opts}",
    "Inspect each view and report which projection has the largest area.\n{opts}",
    "After comparing the three projections, state which has the greatest area.\n{opts}",
]

# 16 integer-count phrasings for "area of the {view} projection in grid squares".
_INT_TEMPLATES = [
    "What is the area of the {view} projection in grid squares? Answer with a single integer.",
    "In the figure, what is the area of the {view} projection (grid squares)?",
    "How many grid squares does the {view} projection cover?",
    "The {view} projection covers how many grid squares? Single integer answer.",
    "Count the filled grid squares in the {view} projection. Give one integer.",
    "Find the area of the {view} projection, measured in grid squares.",
    "Compute the number of grid squares in the {view} projection.",
    "Determine the area (in grid squares) of the {view} projection.",
    "The area of the {view} projection is ___ grid squares. Fill in the integer.",
    "Let A be the area of the {view} projection in grid squares. Find A.",
    "The {view} projection contains N grid squares. What is N?",
    "In the image, the {view} projection has area equal to ___ grid squares.",
    "Examine the {view} projection and report its area in grid squares.",
    "Reason about the {view} projection shown and state its area (integer, grid squares).",
    "Inspect the {view} projection and give the number of grid squares it covers.",
    "After looking at the {view} projection, state its area in grid squares.",
]

_MCQ_OPTS = "(A) Front  (B) Top  (C) Right  (D) All equal"
_TITLE_POOL = title_pool("3d") + ["Orthographic Projections", "Three-View Projection", "Projection Diagram"]

def _get_projection_area(grid, view):
    gx, gy, gz = grid.shape
    count = 0
    if view == 'front':
        for i in range(gx):
            for k in range(gz):
                if any(grid[i, j, k] for j in range(gy)):
                    count += 1
    elif view == 'top':
        for i in range(gx):
            for j in range(gy):
                if any(grid[i, j, k] for k in range(gz)):
                    count += 1
    else:  # right
        for j in range(gy):
            for k in range(gz):
                if any(grid[i, j, k] for i in range(gx)):
                    count += 1
    return count

def _get_projection_grid(grid, view):
    gx, gy, gz = grid.shape
    if view == 'front':
        proj = np.zeros((gx, gz), dtype=int)
        for i in range(gx):
            for k in range(gz):
                if any(grid[i, j, k] for j in range(gy)):
                    proj[i, k] = 1
        return proj
    elif view == 'top':
        proj = np.zeros((gx, gy), dtype=int)
        for i in range(gx):
            for j in range(gy):
                if any(grid[i, j, k] for k in range(gz)):
                    proj[i, j] = 1
        return proj
    else:
        proj = np.zeros((gy, gz), dtype=int)
        for j in range(gy):
            for k in range(gz):
                if any(grid[i, j, k] for i in range(gx)):
                    proj[j, k] = 1
        return proj

class ProjectionAreaComparisonQA(StandaloneVisualEnv):
    ENV_NAME = "projection_area_comparison"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Count cells in each projection briefly, then compare. "
        "Final answer in any of: <answer>X</answer>, \\boxed{{X}}, or "
        "`Final answer: X`."
    )

    def _level_config(self, level):
        # Start with bbox=3 so that different projections already have
        # different areas at L0 (2x2x2 was nearly always "All equal").
        # Grow n_cubes so some cubes can hide behind others (giving
        # distinct projection areas). MCQ only for L<=2; from L3 ask for
        # exact integer counts. The view asked also grows in difficulty
        # (L3-L5: a single known view; L6+: any of 3 views, any answer).
        level = max(0, min(9, int(level)))
        return {
            'n_cubes': 3 + level,            # 3..12
            'bbox':     3 + level // 3,      # 3..6
            'mcq':      level <= 2,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1020)
        style = self._random_style()

        bb = cfg['bbox']
        grid = np.zeros((bb, bb, bb), dtype=int)

        # 2026-05-04 R3: simplified L0 — at L0/L1 force a 2x2 flat slab on
        # ground (Top view 4, Front 2, Right 2 → answer always 'Top' = B).
        if level <= 1:
            cubes = [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
        else:
            # Start from a random cell so that different seeds diverge early.
            # With n_cubes=3 at L0 the greedy growth alone lands on <10 unique
            # shapes across 16 seeds; a random start breaks that degeneracy.
            start = (rng.randrange(bb), rng.randrange(bb), rng.randrange(bb))
            cubes = [start]
            dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
            for _ in range(cfg['n_cubes'] - 1):
                for _a in range(50):
                    base = rng.choice(cubes)
                    d = rng.choice(dirs)
                    nc = tuple(base[i]+d[i] for i in range(3))
                    if all(0 <= nc[i] < bb for i in range(3)) and nc not in cubes:
                        cubes.append(nc); break
        for c in cubes:
            grid[c] = 1

        areas = {
            'Front': _get_projection_area(grid, 'front'),
            'Top': _get_projection_area(grid, 'top'),
            'Right': _get_projection_area(grid, 'right'),
        }
        projs = {
            'Front': _get_projection_grid(grid, 'front'),
            'Top': _get_projection_grid(grid, 'top'),
            'Right': _get_projection_grid(grid, 'right'),
        }

        sidx = (self.seed or 0) % 16
        if cfg['mcq']:
            largest = max(areas, key=areas.get)
            options = ['Front', 'Top', 'Right', 'All equal']
            if areas['Front'] == areas['Top'] == areas['Right']:
                correct = 'D'
            else:
                correct = "ABCD"[options.index(largest)]
            q_text = _MCQ_TEMPLATES[sidx].format(opts=_MCQ_OPTS)
        else:
            view = rng.choice(['Front', 'Top', 'Right'])
            correct = str(areas[view])
            q_text = _INT_TEMPLATES[sidx].format(view=view)

        # Draw projections — clean / textbook / sketch mode mix
        sc = style['figsize_scale']
        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            fill_col = tbp["fill_color"]
            fill_alpha = tbp["fill_alpha"]
            empty_col = bg
            edge_col = tbp["line_color"]
            edge_lw = tbp["line_width"]
            title_kw = {"fontfamily": tbp["font_family"], "color": tbp["line_color"]}
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            fill_col = style['palette'][0]
            fill_alpha = 0.85
            empty_col = bg
            edge_col = "#1a1a1a"
            edge_lw = 1.6
            title_kw = {}
            dpi = style['dpi']
        else:
            bg = style['bg_color']
            fill_col = style['palette'][0]
            fill_alpha = 1.0
            empty_col = '#ecf0f1'
            edge_col = '#555'
            edge_lw = 1.0
            title_kw = {}
            dpi = style['dpi']

        def _draw_axes():
            fig, axes = plt.subplots(1, 3, figsize=(12*sc, 4*sc))
            fig.patch.set_facecolor(bg)
            for idx, (name, proj) in enumerate(projs.items()):
                ax = axes[idx]
                ax.set_facecolor(bg)
                rows, cols = proj.shape
                for i in range(rows):
                    for j in range(cols):
                        c = fill_col if proj[i, j] else empty_col
                        a = fill_alpha if proj[i, j] else 1.0
                        rect = mpatches.FancyBboxPatch((i, j), 1, 1,
                            facecolor=c, edgecolor=edge_col, linewidth=edge_lw,
                            alpha=a)
                        ax.add_patch(rect)
                ax.set_xlim(-0.1, rows+0.1); ax.set_ylim(-0.1, cols+0.1)
                ax.set_aspect('equal'); ax.axis('off')
                ax.set_title(f"{name}", fontsize=10, fontweight='bold', **title_kw)
            fig.suptitle(_TITLE_POOL[sidx % len(_TITLE_POOL)],
                         fontsize=style['font_size_base']+3, fontweight='bold',
                         **title_kw)
            try: fig.tight_layout()
            except: pass
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=80, randomness=1.5):
                fig = _draw_axes()
        else:
            fig = _draw_axes()
        img = self.fig_to_pil(fig, dpi=dpi)

        if level <= 1:
            # 2026-05-04 R3: simplified L0 — leak that answer is always B (Top).
            q_text += (
                " Hint (L0/L1): the shape is a flat 2x2 slab on the ground. "
                "Top view sees a 2x2 = 4 squares, Front and Right each see 1x2 = 2 "
                "squares. So the largest projection is Top; answer is B."
            )
        elif level <= 2:
            q_text += (
                " Hint: each projection (front/top/side) is the silhouette "
                "of the 3D shape from that direction. The projection AREA "
                "is the number of unit squares in the silhouette. For each "
                "view, count the filled grid cells in that direction's "
                "shadow. Compare totals to answer."
            )
        return q_text, correct, img

if __name__ == "__main__":
    env = ProjectionAreaComparisonQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
