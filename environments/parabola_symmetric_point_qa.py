"""
Parabola Symmetric Point QA (M8 / QF-T8, P1).

Given a parabola with axis of symmetry x = h shown, and a labelled point
P = (p, q) on the parabola, find the x-coordinate (or full coordinates) of
the point P' on the parabola symmetric to P about the axis.

Verbatim anchor: research §3.1 Q3 idx 3357 (find equation given two roots —
by symmetry both go to vertex).

P' has x' = 2*h - p and y' = q (same y).
"""
import math
import random
from typing import Dict, Optional, Tuple
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class ParabolaSymmetricPointQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "parabola_symmetric_point"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"a_pool": [1], "h_range": (-2, 2), "k_range": (-2, 2),
                    "p_offset": (1, 3)}
        if level <= 5:
            return {"a_pool": [1, -1], "h_range": (-3, 3), "k_range": (-3, 3),
                    "p_offset": (1, 4)}
        # 2026-05-04: L9 was 100% saturated. Wider ranges + bigger a_pool.
        return {"a_pool": [1, -1, 2, -2, 3, -3], "h_range": (-7, 7), "k_range": (-7, 7),
                "p_offset": (2, 8)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4271 + level * 31 + 7)

        a = rng.choice(cfg["a_pool"])
        h = rng.randint(*cfg["h_range"])
        k = rng.randint(*cfg["k_range"])
        # P = (h + offset, a*offset^2 + k)
        offset = rng.randint(*cfg["p_offset"]) * rng.choice([-1, 1])
        p = h + offset
        q = a * offset * offset + k
        # P' has x' = h - offset = 2h - p
        p_prime = h - offset
        q_prime = q

        question = (
            f"The figure shows a parabola with axis of symmetry x = {h}. "
            f"The point P = ({p}, {q}) is labelled on the parabola. Find "
            f"the x-coordinate of the point P' on the parabola which is "
            f"symmetric to P about the axis of symmetry."
        )
        answer = str(p_prime)

        img = self._render(a, h, k, p, q)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, a, h, k, p, q) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        max_extent = max(abs(h), abs(k), abs(p), abs(q), 6) + 3
        x_lo, x_hi = -max_extent, max_extent
        y_lo, y_hi = -max_extent, max_extent

        xs = np.linspace(x_lo, x_hi, 300)
        ys = a * (xs - h) ** 2 + k
        ax.plot(xs, ys, color="#1f77b4", linewidth=2)

        # Mark vertex
        ax.scatter([h], [k], color="#27ae60", s=70, zorder=5,
                   edgecolor="black", linewidth=1.0)
        # Axis of symmetry
        ax.axvline(h, color="#7f8c8d", linewidth=1, linestyle="--",
                   alpha=0.7)
        ax.text(h, y_hi - 0.5, f"x = {h}", color="#7f8c8d",
                fontsize=10, ha="center")

        # Mark P
        ax.scatter([p], [q], color="#d62728", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)
        ax.annotate(f"P({p}, {q})", (p, q),
                    textcoords="offset points",
                    xytext=(10, 10), fontsize=11, color="#d62728",
                    fontweight="bold")

        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = ParabolaSymmetricPointQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
