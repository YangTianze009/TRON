"""
Multi-View Consistency Check QA — pure single-letter MCQ.

2026-05-05 R5 Batch 4 (TOO-HARD ease + restructure):
Old design (260 LOC) asked which-of-4 view is wrong (very hard, requires the
model to mentally cross-check 3 projections). R3 = 17.5% (mostly random).

REWRITE strategy:
  - L0/L1: "Are the 3 views consistent with a single 3D solid?" 4-MCQ
            {A. Yes  B. No  C. Insufficient info  D. No correct}.
            Use trivially simple solids (cube → all 3 views are squares,
            cuboid → at least one rectangle, etc.). 50/50 split: half consistent,
            half inconsistent (deliberately swap one view's dim).
  - L2-L4: composite stacked solids; same Yes/No structure. 4-MCQ.
  - L5-L7: 5-MCQ with E="No correct answer" (15-25% trap), small voxel
            structures using single-cell flips.
  - L8/L9: subtle inconsistencies, 5-MCQ + 35-40% trap.

Pure single-letter MCQ A-D / A-E. ALLOW_ROTATION=False.
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


def _get_projection(grid: np.ndarray, view: str) -> np.ndarray:
    """Get 2D projection. view: 'front'(y=0), 'top'(z=top), 'right'(x=max)."""
    gx, gy, gz = grid.shape
    if view == 'front':
        proj = np.zeros((gx, gz), dtype=int)
        for i in range(gx):
            for k in range(gz):
                if any(grid[i, j, k] for j in range(gy)):
                    proj[i, k] = 1
        return proj
    if view == 'top':
        proj = np.zeros((gx, gy), dtype=int)
        for i in range(gx):
            for j in range(gy):
                if any(grid[i, j, k] for k in range(gz)):
                    proj[i, j] = 1
        return proj
    # right
    proj = np.zeros((gy, gz), dtype=int)
    for j in range(gy):
        for k in range(gz):
            if any(grid[i, j, k] for i in range(gx)):
                proj[j, k] = 1
    return proj


def _are_projections_realizable(front: np.ndarray, top: np.ndarray,
                                right: np.ndarray) -> bool:
    """Check whether (front, top, right) shadows are realizable by SOME 3D
    binary grid. Uses the maximal-grid theorem.
    """
    gx, gz = front.shape
    gx2, gy = top.shape
    gy2, gz2 = right.shape
    if gx != gx2 or gy != gy2 or gz != gz2:
        return False
    G = np.zeros((gx, gy, gz), dtype=int)
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                if front[i, k] and top[i, j] and right[j, k]:
                    G[i, j, k] = 1
    f2 = _get_projection(G, 'front')
    t2 = _get_projection(G, 'top')
    r2 = _get_projection(G, 'right')
    return (np.array_equal(f2, front)
            and np.array_equal(t2, top)
            and np.array_equal(r2, right))


# ----------------------------------------------------------------------- #
# Simple (1x1x1, etc.) solid generators for L0/L1
# ----------------------------------------------------------------------- #

def _gen_cube(rng) -> np.ndarray:
    """1x1x1 cube."""
    g = np.zeros((1, 1, 1), dtype=int)
    g[0, 0, 0] = 1
    return g


def _gen_2x1x1(rng) -> np.ndarray:
    g = np.zeros((2, 1, 1), dtype=int)
    g[0, 0, 0] = 1
    g[1, 0, 0] = 1
    return g


def _gen_1x2x1(rng) -> np.ndarray:
    g = np.zeros((1, 2, 1), dtype=int)
    g[0, 0, 0] = 1
    g[0, 1, 0] = 1
    return g


def _gen_1x1x2(rng) -> np.ndarray:
    g = np.zeros((1, 1, 2), dtype=int)
    g[0, 0, 0] = 1
    g[0, 0, 1] = 1
    return g


def _gen_2x2x1(rng) -> np.ndarray:
    g = np.zeros((2, 2, 1), dtype=int)
    g[:, :, :] = 1
    return g


def _gen_random_grid(rng, bb: int, n_cubes: int) -> np.ndarray:
    """Random connected voxel structure inside a bb x bb x bb bbox."""
    grid = np.zeros((bb, bb, bb), dtype=int)
    cubes = [(0, 0, 0)]
    grid[0, 0, 0] = 1
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    attempts = 0
    while len(cubes) < n_cubes and attempts < n_cubes * 50:
        attempts += 1
        base = rng.choice(cubes)
        d = rng.choice(dirs)
        nc = (base[0] + d[0], base[1] + d[1], base[2] + d[2])
        if all(0 <= nc[i] < bb for i in range(3)) and grid[nc] == 0:
            cubes.append(nc)
            grid[nc] = 1
    return grid


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #

class MultiViewConsistencyCheckQA(StandaloneVisualEnv):
    """Yes/No-style MCQ: are the 3 views consistent with a single 3D solid?"""

    ALLOW_ROTATION = False
    ENV_NAME = "multi_view_consistency_check"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            # Simple solids; binary Yes/No 4-MCQ
            return {"mode": "simple", "use_5_options": False,
                    "trap_rate": 0.0}
        if level <= 4:
            return {"mode": "voxel", "bbox": 2, "n_cubes": 3 + level,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 6:
            return {"mode": "voxel", "bbox": 3, "n_cubes": 4 + level // 2,
                    "use_5_options": True, "trap_rate": 0.15}
        if level == 7:
            return {"mode": "voxel", "bbox": 3, "n_cubes": 6,
                    "use_5_options": True, "trap_rate": 0.25}
        if level == 8:
            return {"mode": "voxel", "bbox": 3, "n_cubes": 7,
                    "use_5_options": True, "trap_rate": 0.35}
        # L9
        return {"mode": "voxel", "bbox": 3, "n_cubes": 8,
                "use_5_options": True, "trap_rate": 0.40}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1015)
        for _ in range(30):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        mode = cfg["mode"]
        if mode == "simple":
            grid_fn = rng.choice([_gen_cube, _gen_2x1x1, _gen_1x2x1,
                                  _gen_1x1x2, _gen_2x2x1])
            grid = grid_fn(rng)
        else:
            grid = _gen_random_grid(rng, cfg["bbox"], cfg["n_cubes"])
            if grid.sum() < 2:
                return None

        front = _get_projection(grid, 'front')
        top = _get_projection(grid, 'top')
        right = _get_projection(grid, 'right')

        # 50/50 corrupt one view to make it inconsistent
        is_consistent = rng.random() < 0.5
        projs = [front.copy(), top.copy(), right.copy()]
        if not is_consistent:
            # Try corrupting each view in turn until we find one that becomes
            # unrealizable.
            indices = [0, 1, 2]
            rng.shuffle(indices)
            corrupted = False
            for idx in indices:
                p = projs[idx].copy()
                rows, cols = p.shape
                cells = [(i, j) for i in range(rows) for j in range(cols)]
                rng.shuffle(cells)
                # Prefer flipping a 1->0 first
                cells_priority = sorted(cells,
                                        key=lambda c: 0 if p[c[0], c[1]] == 1 else 1)
                for (i, j) in cells_priority:
                    cand = p.copy()
                    cand[i, j] = 1 - cand[i, j]
                    trial = list(projs)
                    trial[idx] = cand
                    if not _are_projections_realizable(trial[0], trial[1],
                                                       trial[2]):
                        projs[idx] = cand
                        corrupted = True
                        break
                if corrupted:
                    break
            if not corrupted:
                # Could not corrupt — fall back to consistent
                is_consistent = True

        # Build MCQ
        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4
        opt_letters = ["A", "B", "C", "D", "E"][:n_value]

        # Options:
        #   A. Yes, the views are consistent with one 3D solid
        #   B. No, no single 3D solid produces these views
        #   C. Insufficient information
        #   D. None of the above
        #   (E. No correct answer — only when use_5)
        if not use_5:
            options = [
                "Yes, the views are consistent with one 3D solid",
                "No, no single 3D solid produces these views",
                "Insufficient information",
                "None of the above",
            ]
            true_text = options[0] if is_consistent else options[1]
            # Shuffle so that "Yes" / "No" don't always sit at A / B.
            opt_lines = list(options)
            rng.shuffle(opt_lines)
            correct_letter = opt_letters[opt_lines.index(true_text)]
        else:
            base_options = [
                "Yes, the views are consistent with one 3D solid",
                "No, no single 3D solid produces these views",
                "Insufficient information",
                "None of the above",
            ]
            use_trap = (rng.random() < cfg["trap_rate"])
            if use_trap:
                # Need an option pool that does NOT include the right answer.
                if is_consistent:
                    pool = [o for o in base_options if "Yes" not in o]
                else:
                    pool = [o for o in base_options if "No, no single" not in o]
                if len(pool) < n_value - 1:
                    return None
                rng.shuffle(pool)
                chosen = pool[:n_value - 1]
                rng.shuffle(chosen)
                opt_lines = chosen + ["No correct answer"]
                correct_letter = opt_letters[-1]
            else:
                true_text = base_options[0] if is_consistent else base_options[1]
                # Use all 4 base shuffled + "No correct answer" as 5th option
                shuffled = list(base_options)
                rng.shuffle(shuffled)
                opt_lines = shuffled + ["No correct answer"]
                correct_letter = opt_letters[opt_lines.index(true_text)]

        # Render image
        img = self._render_views(projs, rng, mode=mode)

        opt_text = "\n".join(f"{opt_letters[i]}. {opt_lines[i]}"
                             for i in range(n_value))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        if level <= 1:
            stem = (
                "The image shows three orthographic projections (front, top, "
                "right) of a 3D solid composed of unit cubes. Could a single "
                "3D solid produce ALL three of these views simultaneously? "
                "Hint: front+top share width, front+right share height, "
                "top+right share depth. Be concise."
            )
        else:
            stem = (
                "The image shows three orthographic projections (front, top, "
                "right) of a 3D structure made of unit cubes. Determine "
                "whether the three views are mutually consistent with a "
                "single 3D voxel solid."
            )

        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    def _render_views(self, projs, rng, mode="voxel") -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(11 * sc, 4 * sc), dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])

        view_titles = ["Front view", "Top view", "Right view"]
        empty_color = rng.choice(['#ecf0f1', '#f5f5f5', '#fafafa', '#eeeeee'])
        palette = [c for c in style["palette"]
                   if c.lower() not in ('#000000', '#ffffff')]
        if not palette:
            palette = ["#5dade2"]
        fill_color = rng.choice(palette)

        for idx, proj in enumerate(projs):
            ax = fig.add_subplot(1, 3, idx + 1)
            ax.set_facecolor(style["bg_color"])
            rows, cols = proj.shape
            for i in range(rows):
                for j in range(cols):
                    color = fill_color if proj[i, j] else empty_color
                    rect = mpatches.Rectangle((i, j), 1, 1,
                                              facecolor=color,
                                              edgecolor='#444',
                                              linewidth=1.5)
                    ax.add_patch(rect)
            ax.set_xlim(-0.2, rows + 0.2)
            ax.set_ylim(-0.2, cols + 0.2)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(view_titles[idx],
                         fontsize=style["font_size_base"] + 2,
                         fontweight="bold")

        fig.suptitle("Three Orthographic Views",
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = MultiViewConsistencyCheckQA()
    for lv in [0, 3, 6, 9]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
