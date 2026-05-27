"""
Lattice Path Count QA (v4 G8d, for combinatorics).

Targets: combinatorics -1.79 (and broader graph/enumeration
regressions).

Task: given a rectangular grid with blocked cells, count the number of
monotonic paths (only right/up moves) from (0,0) to (m, n).

Reward: exact integer.

Level axes:
  A) Grid size: 2x2 at L0 -> 5x5 at L9
  B) Number of blocked cells: 0 at L0-2, 1 at L3-5, 2-3 at L6+
"""
import random
import math
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "Count the number of monotonic paths (moving only RIGHT or UP) from the lower-left corner to the upper-right corner of a {m}×{n} grid. Blocked cells (marked ✗): {blocked_desc}. Paths cannot pass through blocked cells. Put the integer in <answer>...</answer>.",
    "How many right-and-up-only paths go from lower-left to upper-right on a {m}×{n} grid where the following cells are blocked: {blocked_desc}? Integer in <answer>...</answer>.",
    "A {m}×{n} grid has blocked cells {blocked_desc}. Count lattice paths (R/U only) from bottom-left to top-right. Integer in <answer>...</answer>.",
    "Grid {m}×{n}, blocked cells {blocked_desc}. Paths LL → UR using R/U only? Integer in <answer>...</answer>.",
    "Count monotonic paths on a {m}×{n} grid with blocked cells {blocked_desc}. Integer in <answer>...</answer>.",
    "Lattice paths from (0,0) to ({m},{n}), only R/U moves, blocked cells {blocked_desc}. Count? Integer in <answer>...</answer>.",
    "Paths count on {m}×{n} grid (R/U only), with {blocked_desc} blocked. Put integer in <answer>...</answer>.",
    "Enumerate paths: {m}×{n} grid, blocked {blocked_desc}, R/U moves only. Put integer in <answer>...</answer>.",
    "Count right/up lattice paths on {m}×{n} grid avoiding blocked cells {blocked_desc}. Integer in <answer>...</answer>.",
    "On a {m}×{n} grid with blocked {blocked_desc}, how many R/U-only paths from LL to UR? Integer in <answer>...</answer>.",
    "Grid paths problem: {m}×{n}, blocked {blocked_desc}. Put count in <answer>...</answer>.",
    "Count monotonic paths, {m}×{n}, blocked cells {blocked_desc}. Integer in <answer>...</answer>.",
    "Monotonic lattice paths on {m}×{n} with {blocked_desc} blocked. Put integer in <answer>...</answer>.",
    "Right-or-up paths LL → UR, grid {m}×{n}, blocks {blocked_desc}. Count? Integer in <answer>...</answer>.",
    "Path count: {m}×{n} grid, R/U only, avoid {blocked_desc}. Integer in <answer>...</answer>.",
    "Number of paths (R or U) on {m}×{n} avoiding {blocked_desc}? Integer in <answer>...</answer>.",
]

def _count_paths(m: int, n: int, blocked: Set[Tuple[int, int]]) -> int:
    """Count R/U paths from (0,0) to (m, n) avoiding blocked cells."""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    if (0, 0) in blocked:
        return 0
    dp[0][0] = 1
    for i in range(m + 1):
        for j in range(n + 1):
            if (i, j) in blocked:
                dp[i][j] = 0
                continue
            if i == 0 and j == 0:
                continue
            dp[i][j] = (dp[i - 1][j] if i > 0 else 0) + \
                      (dp[i][j - 1] if j > 0 else 0)
    return dp[m][n]

class LatticePathCountQA(StandaloneVisualEnv):
    ENV_NAME = "lattice_path_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        m = 2 + level // 2
        n = 2 + (level + 1) // 2
        if level <= 2:
            n_blocked = 0
        elif level <= 5:
            n_blocked = 1
        else:
            n_blocked = 2
        return {"m": m, "n": n, "n_blocked": n_blocked}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 347)
        self._primary_complexity_feature = level

        m, n = cfg["m"], cfg["n"]
        blocked = set()
        for _ in range(cfg["n_blocked"]):
            attempts = 0
            while attempts < 20:
                attempts += 1
                bx = rng.randint(1, m - 1) if m > 1 else 0
                by = rng.randint(1, n - 1) if n > 1 else 0
                if (bx, by) not in blocked and (bx, by) != (0, 0) and \
                   (bx, by) != (m, n):
                    blocked.add((bx, by))
                    break

        count = _count_paths(m, n, blocked)
        if count == 0:
            return None
        answer = str(count)

        blocked_desc = ", ".join(f"({x},{y})" for x, y in sorted(blocked)) if blocked else "none"
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(m=m, n=n, blocked_desc=blocked_desc)

        img = self._render(m, n, blocked, rng)
        return q, answer, img

    def _render(self, m, n, blocked, rng):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, m + 1.0)
        ax.set_ylim(-0.5, n + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")

        for i in range(m + 1):
            for j in range(n + 1):
                if (i, j) in blocked:
                    ax.scatter(i, j, s=200, color="#e74c3c", marker="x",
                               linewidths=3)
                else:
                    color = "#27ae60" if (i, j) in [(0, 0), (m, n)] else "#3498db"
                    ax.scatter(i, j, s=120, color=color, zorder=3)
        # Draw grid lines
        for i in range(m + 1):
            ax.plot([i, i], [0, n], color="#cccccc", lw=0.8, zorder=1)
        for j in range(n + 1):
            ax.plot([0, m], [j, j], color="#cccccc", lw=0.8, zorder=1)

        # Labels
        ax.text(0, -0.3, "START\n(0,0)", fontsize=9, ha="center",
                va="top", color="darkgreen", fontweight="bold")
        ax.text(m, n + 0.35, f"END\n({m},{n})", fontsize=9, ha="center",
                va="bottom", color="darkgreen", fontweight="bold")
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_lpc"
    os.makedirs(out_dir, exist_ok=True)
    env = LatticePathCountQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 173
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[lpc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/lpc_s{s}_L{level}.png")
            print(f"[lpc L{level} s{s}] A={env._answer}")
