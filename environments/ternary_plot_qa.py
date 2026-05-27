"""
Ternary Plot QA (D123, P3).

Reference task:
  qid 295 (UG MCQ): "The ternary plot shows the three-sector model of an
   unknown country. Which sector contributes most to the economy of this
   country? choice: (A) Primary Sector (B) Secondary Sector (C) Tertiary
   Sector." Ans: B.

Renders a ternary plot (equilateral triangle with three axes labeled
Primary, Secondary, Tertiary) with one labeled point. The model picks the
sector that contributes most to that point's barycentric coordinates.

Verifier: single MCQ letter A/B/C.
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
    "The ternary plot shows the three-sector model of an unknown country. Which sector contributes most to the economy of this country (the marked point)? Choices: (A) Primary Sector (B) Secondary Sector (C) Tertiary Sector. Place letter in <answer>...</answer>.",
    "Looking at the ternary plot with three sectors, which contributes the most? Choices: (A) Primary (B) Secondary (C) Tertiary. Place letter in <answer>...</answer>.",
    "Examine the labeled point in the ternary diagram. Which sector dominates? Choices: (A) Primary (B) Secondary (C) Tertiary. Letter in <answer>...</answer>.",
    "The ternary diagram displays a country's three-sector breakdown. Which sector is largest? Choices: (A) Primary (B) Secondary (C) Tertiary. Place letter in <answer>...</answer>.",
    "From the ternary plot, identify the dominant sector at the marked point. Choices: (A) Primary (B) Secondary (C) Tertiary. Letter in <answer>...</answer>.",
    "Pick the largest contributing sector from the ternary diagram. Choices: (A) Primary (B) Secondary (C) Tertiary. Place letter in <answer>...</answer>.",
    "Which sector has the largest share at the labeled point in the ternary plot? Choices: (A) Primary (B) Secondary (C) Tertiary. Letter in <answer>...</answer>.",
    "Determine the dominant sector from the ternary plot. Choices: (A) Primary (B) Secondary (C) Tertiary. Place letter in <answer>...</answer>.",
]


class TernaryPlotQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "ternary_plot"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 3:
            return {"min_dominant": 0.55}
        if level <= 6:
            return {"min_dominant": 0.45}
        return {"min_dominant": 0.40}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7541 + level * 71 + 137)

        # Pick which sector dominates
        dominant_idx = rng.randint(0, 2)
        # Generate barycentric coords (p, q, r) summing to 1, with the
        # dominant fraction >= min_dominant
        for _ in range(60):
            dom = rng.uniform(cfg["min_dominant"],
                              cfg["min_dominant"] + 0.25)
            other = (1 - dom)
            split = rng.uniform(0.2, 0.8) * other
            other2 = other - split
            shares = [0.0, 0.0, 0.0]
            shares[dominant_idx] = dom
            others = [(dominant_idx + 1) % 3, (dominant_idx + 2) % 3]
            shares[others[0]] = split
            shares[others[1]] = other2
            # Sanity check
            if min(shares) >= 0 and abs(sum(shares) - 1.0) < 0.01:
                if shares[dominant_idx] > max(
                    shares[others[0]], shares[others[1]]
                ) + 0.05:
                    break
        else:
            return None

        # Point coords in 2D from barycentric
        # Vertices: A=(0,0) primary; B=(1,0) secondary; C=(0.5, sqrt(3)/2) tertiary
        Av = (0, 0)
        Bv = (1, 0)
        Cv = (0.5, math.sqrt(3) / 2)
        x = shares[0] * Av[0] + shares[1] * Bv[0] + shares[2] * Cv[0]
        y = shares[0] * Av[1] + shares[1] * Bv[1] + shares[2] * Cv[1]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer = "ABC"[dominant_idx]
        img = self._render(x, y, shares)
        return question, answer, img

    def _render(self, x, y, shares) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        # Triangle vertices
        Av = (0, 0)
        Bv = (1, 0)
        Cv = (0.5, math.sqrt(3) / 2)
        # Edges
        ax.plot([Av[0], Bv[0], Cv[0], Av[0]],
                [Av[1], Bv[1], Cv[1], Av[1]], color="#1a1a1a", linewidth=2)
        # Vertex labels
        ax.text(Av[0] - 0.05, Av[1] - 0.05, "Primary", fontsize=12,
                fontweight="bold", color="#003566", ha="right", va="top")
        ax.text(Bv[0] + 0.05, Bv[1] - 0.05, "Secondary", fontsize=12,
                fontweight="bold", color="#7f1d1d", ha="left", va="top")
        ax.text(Cv[0], Cv[1] + 0.04, "Tertiary", fontsize=12,
                fontweight="bold", color="#15803d",
                ha="center", va="bottom")

        # Grid lines (parallel to sides) at 0.25 intervals
        for k in [0.25, 0.5, 0.75]:
            # parallel to BC (constant primary share = k)
            p1 = (k * Cv[0], k * Cv[1])
            p2 = (k * Bv[0] + (1 - k) * 0, k * Bv[1])
            # actually points along edges
            # parallel to BC: from (0,0)+(k)*B side to (0,0)+(k)*C
            # Easier: lerp along sides
            p1 = (k * Cv[0] + (1 - k) * Av[0],
                  k * Cv[1] + (1 - k) * Av[1])
            p2 = (k * Cv[0] + (1 - k) * Bv[0],
                  k * Cv[1] + (1 - k) * Bv[1])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color="#bbb", linewidth=0.6, linestyle=":")

        # Plot the data point
        ax.plot(x, y, "o", color="#b00020", markersize=16,
                markeredgecolor="#333", markeredgewidth=1.5)
        ax.text(x + 0.03, y + 0.03, "Country", fontsize=11,
                fontweight="bold", color="#b00020")

        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.15, 1.0)
        ax.set_title("Ternary plot (three-sector model)",
                     fontsize=13, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = TernaryPlotQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
