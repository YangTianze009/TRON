"""
Triangle Vertex Angle Trace QA (v4 G3c, for metric angle / Angle reasoning).

Targets: metric geometry - angle -0.58 (shared-vertex angle
decomposition cases like idx=1034 "two equilateral triangles share vertex").

Task: 2-3 triangles sharing a common vertex. Given 2-3 known angles at
the shared vertex, compute a remaining angle formed by the arrangement.

Reward: numeric within 0.5°.

Level axes:
  A) Number of triangles: 2 at L0-3, 3 at L4+
  B) Share pattern: full vertex at L0-3, partial overlap at L4+
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "Two triangles share a common vertex P. At P, {given_txt}. Compute the total angle remaining at P (360° minus occupied). Round to 2 decimals.",
    "Two triangles meet at P. Given {given_txt}, compute the 'gap' angle at P in degrees. Round to 2 decimals.",
    "At shared vertex P of two triangles: {given_txt}. Find the uncovered angle at P (degrees, 2 decimals).",
    "Two triangles share vertex P. {given_txt}. The unoccupied angle at P in degrees (2 decimals)?",
    "Given {given_txt} at the shared vertex P of two triangles, compute the remaining angle (degrees, 2 decimals).",
    "Shared-vertex triangles: {given_txt}. Gap angle in degrees (2 decimals)?",
    "At P, two triangles meet with angles: {given_txt}. Remaining angle in degrees (2 decimals)?",
    "Compute the uncovered angle at P: {given_txt} of two sharing triangles. Degrees, 2 decimals.",
    "Given angle info {given_txt} at shared vertex P, compute leftover angle in degrees (2 decimals).",
    "Leftover at P after {given_txt}? Degrees, 2 decimals.",
    "Two triangles at P: {given_txt}. Gap (360° - sum) in degrees, 2 decimals?",
    "At shared vertex: {given_txt}. Find remaining angle in degrees (2 decimals).",
    "Compute the unfilled angle at P. {given_txt}. Degrees, 2 decimals.",
    "Given the shared-vertex setup and {given_txt}, what's the remaining P angle in degrees (2 decimals)?",
    "Two triangles share vertex P; angles there: {given_txt}. Leftover angle in degrees (2 decimals)?",
    "Triangles at P with {given_txt}. Gap in degrees (2 decimals)?",
]

class TriangleVertexAngleTraceQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "triangle_vertex_angle_trace"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Progressive difficulty: more triangles, more hidden vertex
        # angles that have to be derived from in-triangle interior sums.
        if level <= 1:
            n_tri = 2; n_hidden = 0
        elif level <= 3:
            n_tri = 3; n_hidden = 0
        elif level <= 5:
            n_tri = 3; n_hidden = 1   # one vertex angle must be derived
        elif level <= 7:
            n_tri = 4; n_hidden = 1
        else:
            n_tri = 4; n_hidden = 2
        return {"n_tri": n_tri, "n_hidden": n_hidden}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 739)
        self._primary_complexity_feature = level

        # Pick angles at shared vertex P (each strictly < 180 and total < 360
        # with at least 30° gap left over for the answer).
        angles = []
        remaining = 360
        for i in range(cfg["n_tri"]):
            high = min(120, remaining - 30 * (cfg["n_tri"] - i))
            if high < 30:
                return None
            a = rng.randint(30, high)
            angles.append(a)
            remaining -= a

        gap = 360 - sum(angles)
        if gap < 30:
            return None

        # Hide some vertex angles by replacing the displayed label with the
        # OTHER two interior angles of that triangle (so the model has to use
        # triangle sum = 180 to recover the vertex angle, then 360-sum to
        # answer). The hidden angles still appear as a number elsewhere in
        # the figure but as the two interior counterparts.
        hidden_idx = []
        if cfg["n_hidden"] > 0:
            choices = list(range(cfg["n_tri"]))
            rng.shuffle(choices)
            hidden_idx = choices[: cfg["n_hidden"]]

        # For each hidden triangle pick two other interior angles that sum
        # to 180 - vertex_angle.
        hidden_pairs = {}
        for i in hidden_idx:
            v = angles[i]
            other_total = 180 - v
            if other_total < 20:
                return None
            a1 = rng.randint(10, max(11, other_total - 10))
            a2 = other_total - a1
            hidden_pairs[i] = (a1, a2)

        given_parts = []
        for i, a in enumerate(angles):
            if i in hidden_pairs:
                a1, a2 = hidden_pairs[i]
                given_parts.append(
                    f"triangle {i+1}'s other interior angles are {a1}° and {a2}°"
                )
            else:
                given_parts.append(f"angle {i+1} = {a}°")
        given_txt = ", ".join(given_parts)
        answer = str(gap)

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(given_txt=given_txt)

        img = self._render(angles, rng, hidden_pairs)
        return q, answer, img

    def _render(self, angles, rng, hidden_pairs=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw triangles radiating from center
        P = (0, 0)
        ax.scatter(*P, s=150, color="black", zorder=5)
        ax.text(P[0] - 0.3, P[1] - 0.3, "P", fontsize=16, fontweight="bold")
        current_angle = 90  # start pointing up
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
        hidden_pairs = hidden_pairs or {}
        for i, a in enumerate(angles):
            # Triangle: vertex P + 2 other vertices spaced a° apart
            v1_angle = math.radians(current_angle)
            v2_angle = math.radians(current_angle - a)
            v1 = (2.2 * math.cos(v1_angle), 2.2 * math.sin(v1_angle))
            v2 = (2.2 * math.cos(v2_angle), 2.2 * math.sin(v2_angle))
            ax.add_patch(mpatches.Polygon([P, v1, v2],
                                          fc=colors[i % len(colors)],
                                          ec="black", lw=1.5, alpha=0.5))
            mid_angle = (current_angle + current_angle - a) / 2
            lx = 0.8 * math.cos(math.radians(mid_angle))
            ly = 0.8 * math.sin(math.radians(mid_angle))
            if i in hidden_pairs:
                # Hide the vertex angle; show the other two interior angles of
                # this triangle near v1 and v2 instead.
                a1, a2 = hidden_pairs[i]
                ax.text(lx, ly, "?°", fontsize=12, ha="center",
                        fontweight="bold", color="#c0392b")
                # near v1
                ax.text(v1[0] * 0.65 + lx * 0.35,
                        v1[1] * 0.65 + ly * 0.35,
                        f"{a1}°", fontsize=10, ha="center", color="#1e8449")
                # near v2
                ax.text(v2[0] * 0.65 + lx * 0.35,
                        v2[1] * 0.65 + ly * 0.35,
                        f"{a2}°", fontsize=10, ha="center", color="#1e8449")
            else:
                ax.text(lx, ly, f"{a}°", fontsize=12, ha="center",
                        fontweight="bold")
            current_angle -= a
        # The gap
        ax.text(0, -2.7, f"Find the remaining (gap) angle at P",
                fontsize=11, ha="center", fontweight="bold",
                bbox=dict(facecolor="lightyellow", edgecolor="gray"))
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip()
        for sym in ["°", "\\circ", "degrees", "degree"]:
            pred = pred.replace(sym, "").strip()
            gt = gt.replace(sym, "").strip()
        try:
            return abs(float(pred) - float(gt)) < 0.5
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_tva"
    os.makedirs(out_dir, exist_ok=True)
    env = TriangleVertexAngleTraceQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 43
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[tva L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/tva_s{s}_L{level}.png")
            print(f"[tva L{level} s{s}] A={env._answer}")
