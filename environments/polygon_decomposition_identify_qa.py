"""
Polygon Decomposition Identify QA (v4 G3b, for metric-angle).

Targets: metric geometry - angle -0.58 (idx=1284 "sum of 10
marked angles" — failure mode: mis-counts figure as a pentagon pattern).

Task: compound figure (2+ overlapping polygons) labeled with vertex letters.
Ask how many of a specific shape (triangles, quads, etc.) are present —
the model must decompose the compound correctly.

Reward: exact integer.

Level axes:
  A) Number of overlapping polygons: 2 at L0-3, 3 at L4-6, 4+ at L7+
  B) Target shape: triangles at L0-5, quadrilaterals at L6+
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The figure shows overlapping polygons. Count the number of distinct {target} in the figure (not just the outer boundary — include inner triangles formed by overlap lines). Put the integer in <answer>...</answer>.",
    "How many {target} do you see in the figure? Count all (small and large, including those formed by overlap). Integer in <answer>...</answer>.",
    "The compound figure is made of overlapping shapes. Count the distinct {target} visible. Integer in <answer>...</answer>.",
    "Identify and count every {target} in the figure. Integer in <answer>...</answer>.",
    "How many {target} appear in the diagram? Integer in <answer>...</answer>.",
    "Count all {target} (small and large) in the figure. Integer in <answer>...</answer>.",
    "How many distinct {target} can you identify? Integer in <answer>...</answer>.",
    "Enumerate {target} in the figure. Integer in <answer>...</answer>.",
    "Figure: count all {target}. Integer in <answer>...</answer>.",
    "Total distinct {target} in the figure? Integer in <answer>...</answer>.",
    "Count {target} — small and large. Integer in <answer>...</answer>.",
    "How many {target} are visible? Integer in <answer>...</answer>.",
    "Integer count of {target} in the figure: Put in <answer>...</answer>.",
    "Count every {target} (including nested ones). Integer in <answer>...</answer>.",
    "Identify total {target} count. Integer in <answer>...</answer>.",
    "{target} count in the figure? Integer in <answer>...</answer>.",
]

class PolygonDecompositionIdentifyQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "polygon_decomposition_identify"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04 R3: simplified L0 — at L0/L1 use n_lines=1
        # (yields only 3 triangles, count is trivial).
        if level <= 1:
            n_lines = 1
        elif level <= 3:
            n_lines = 3
        elif level <= 6:
            n_lines = 4
        else:
            n_lines = 5
        return {"n_lines": n_lines}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 977)
        self._primary_complexity_feature = level

        # Draw a regular triangle with `n_lines` additional lines from each vertex
        # to opposite sides (medians or random). Count all smaller triangles formed.
        n = cfg["n_lines"]
        # Use a known triangle configuration:
        #   - base triangle has 1 large triangle
        #   - each additional line adds ~2 small triangles and splits some others
        # For simplicity we use a precomputed table based on n_lines from single vertex:
        #   lines from one vertex = 1 large + (n + 1) small
        # But n_lines is total added edges — we split to be more interesting.

        # BUGFIX 2026-04-24: question says "count all including those formed by
        # overlap". For n cevians from one vertex, the total number of triangles
        # is C(n+2, 2) = (n+2)(n+1)/2. Previous dict {3:5, 4:8, 5:13} was the
        # undercount ignoring overlap-combinations.
        # 2026-05-04 R3: at n_lines=1 (L0/L1) the count is C(3,2)=3.
        triangle_counts = {1: 3, 3: 10, 4: 15, 5: 21}
        count = triangle_counts.get(n, (n + 2) * (n + 1) // 2)

        answer = str(count)
        target = "triangles"
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(target=target)
        if level <= 1:
            # 2026-05-04 R3: simplified L0 — leak the formula and answer.
            q += (
                f"\nHint (L0/L1): there is exactly ONE inner line from vertex A "
                f"to side BC, splitting the big triangle into 2 small triangles. "
                f"Count is 1 (big) + 2 (small) = 3. Reply <answer>3</answer>."
            )

        img = self._render(n, rng)
        return q, answer, img

    def _render(self, n_lines, rng):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.5, 5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Main triangle with vertices A, B, C
        A = (0, 0); B = (5, 0); C = (2.5, 4.3)
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]],
                color="black", lw=2.0)

        # Add n_lines cevians from vertex A to base BC
        for i in range(n_lines):
            t = (i + 1) / (n_lines + 1)
            # Point on BC: B + t * (C - B)
            px = B[0] + t * (C[0] - B[0])
            py = B[1] + t * (C[1] - B[1])
            ax.plot([A[0], px], [A[1], py], color="black", lw=1.3)

        # Labels
        ax.text(A[0] - 0.3, A[1] - 0.2, "A", fontsize=16, fontweight="bold")
        ax.text(B[0] + 0.1, B[1] - 0.2, "B", fontsize=16, fontweight="bold")
        ax.text(C[0], C[1] + 0.2, "C", fontsize=16, fontweight="bold",
                ha="center")

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_pdi"
    os.makedirs(out_dir, exist_ok=True)
    env = PolygonDecompositionIdentifyQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 227
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[pdi L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/pdi_s{s}_L{level}.png")
            print(f"[pdi L{level} s{s}] A={env._answer}")
