"""
Region Count Grid QA environment.

Template: region_counting_qa.py.

Goal: count the number of disjoint regions a set of straight lines
divides a unit square into. Targets counting 
and VisuLogic Attribute Reasoning (region-count as attribute).

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): n_lines = 2 + level // 2        -> 2..6
  Axis 2           : line_generality                 axis-parallel -> diagonals
  Axis 3 (optional): option_gap = max(1, 4 - level // 2) -> 4..1

Output format is constant: 4-option integer MCQ, single letter.

Region counting is done by flood-filling a rasterized version of the
square — robust and easy (and avoids the shapely dependency).
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

def _rasterize_lines(segments, res=120):
    """Rasterize lines into a res x res grid. Returns 0 for white cells,
    1 for line cells."""
    grid = np.zeros((res, res), dtype=np.uint8)
    for seg in segments:
        (x0, y0), (x1, y1) = seg
        n_steps = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) * res * 1.4))
        xs = np.linspace(x0, x1, n_steps)
        ys = np.linspace(y0, y1, n_steps)
        ix = np.clip((xs * res).astype(int), 0, res - 1)
        iy = np.clip((ys * res).astype(int), 0, res - 1)
        grid[iy, ix] = 1
        grid[np.clip(iy + 1, 0, res - 1), ix] = 1
        grid[iy, np.clip(ix + 1, 0, res - 1)] = 1
        grid[np.clip(iy - 1, 0, res - 1), ix] = 1
        grid[iy, np.clip(ix - 1, 0, res - 1)] = 1
    return grid

def _connected_components(grid):
    """Count connected components of zero pixels via iterative flood-fill."""
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0 and not visited[i, j]:
                count += 1
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if grid[ny, nx] == 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
    return count

def _count_regions(segments, res=120):
    """Draw segments inside the [0,1]^2 square and count disjoint regions."""
    grid = _rasterize_lines(segments, res=res)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return _connected_components(grid)

class RegionCountGridQA(StandaloneVisualEnv):
    ENV_NAME = "region_count_grid"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_lines": 2 + level // 2,               # 2..6
            "allow_diagonal": level >= 3,
            "allow_partial": level >= 6,
            "option_gap": max(1, 4 - level // 2),    # 4..1
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_lines"]

        for _ in range(20):
            try:
                result = self._try_generate(rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _sample_line(self, rng, cfg):
        """Sample a (p0, p1) line segment inside [0,1]^2."""
        if not cfg["allow_diagonal"] or rng.random() < 0.5:
            if rng.random() < 0.5:
                y = round(rng.uniform(0.15, 0.85), 3)
                return ((0, y), (1, y))
            x = round(rng.uniform(0.15, 0.85), 3)
            return ((x, 0), (x, 1))
        # Diagonal
        a = rng.uniform(-2.0, 2.0)
        if abs(a) < 0.3:
            a = 0.8 if a >= 0 else -0.8
        b = rng.uniform(-0.3, 0.3)
        pts = []
        for x in (0, 1):
            y = a * x + b
            if 0 <= y <= 1:
                pts.append((x, y))
        for y in (0, 1):
            if abs(a) > 1e-6:
                x = (y - b) / a
                if 0 <= x <= 1:
                    pts.append((x, y))
        seen = set()
        clean = []
        for p in pts:
            key = (round(p[0], 3), round(p[1], 3))
            if key not in seen:
                clean.append(p)
                seen.add(key)
        if len(clean) < 2:
            return None
        p0, p1 = clean[0], clean[1]
        if cfg["allow_partial"] and rng.random() < 0.4:
            t0 = rng.uniform(0, 0.3)
            t1 = rng.uniform(0.7, 1.0)
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            p0 = (p0[0] + t0 * dx, p0[1] + t0 * dy)
            p1 = (p0[0] + (t1 - t0) * dx, p0[1] + (t1 - t0) * dy)
        return (p0, p1)

    def _try_generate(self, rng, cfg, level):
        segments = []
        for _ in range(80):
            if len(segments) >= cfg["n_lines"]:
                break
            seg = self._sample_line(rng, cfg)
            if seg is None:
                continue
            segments.append(seg)
        if len(segments) < 2:
            return None

        regions = _count_regions(segments)
        if regions < 2 or regions > 40:
            return None

        gap = cfg["option_gap"]
        options_pool = [regions + k for k in (-2 * gap, -gap, gap, 2 * gap)
                        if regions + k >= 1]
        if regions not in options_pool:
            options_pool.append(regions)
        options_pool = list(set(options_pool))
        rng.shuffle(options_pool)
        options = [regions]
        for v in options_pool:
            if len(options) >= 4:
                break
            if v not in options and v >= 1:
                options.append(v)
        while len(options) < 4:
            v = regions + rng.randint(1, 5)
            if v not in options:
                options.append(v)
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(regions))

        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        q_stems = [
            "The square is divided by the lines drawn inside it. How many separate regions does the square contain in total?",
            "Count all the disjoint regions that the lines create inside the square.",
            "How many distinct enclosed regions are formed by the lines within the square boundary?",
            "Looking at the square and the lines inside it, determine the total number of separate regions.",
        ]
        question = (
            f"{rng.choice(q_stems)} "
            f"Options: {opts_text}. Answer with a single letter."
        )

        image = self._render(segments, cfg)
        return question, correct_letter, image

    def _render(self, segments, cfg):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]

        rect = mpatches.Rectangle((0, 0), 1, 1, facecolor="none",
                                   edgecolor="#2c3e50",
                                   linewidth=style["line_width"] * 1.4,
                                   zorder=1)
        ax.add_patch(rect)
        lw = style["line_width"] * 1.1
        for i, ((x0, y0), (x1, y1)) in enumerate(segments):
            c = palette[i % len(palette)]
            ax.plot([x0, x1], [y0, y1], color=c, linewidth=lw, zorder=2)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        title_pool = ["Count the Regions", "Region Counting",
                      "Line Partition", "Divided Square", "Regions"]
        ax.set_title(self._rng.choice(title_pool),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = RegionCountGridQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[region_count_grid] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/region_count_grid_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | nlines={env._primary_complexity_feature}")
