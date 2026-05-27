"""
Parallel Lines Angle Chain QA (v4 G3, for angle-chain reasoning/Property).

Targets (from s300 regression data):

Failure mode (from pattern extraction):
  text-lite: 49% very_short_response, 35% bare_letter_no_cot.
  Model skips the 2-4 angle-hop reasoning and picks a letter directly.

Task: 2 parallel lines cut by a transversal (and optionally an extra line),
with 1-2 given angle values. Ask for a target angle requiring 2-4 hops
(corresponding / alternate-interior / supplementary / triangle-sum /
vertical angles).

Reward: numeric angle within 0.5° tolerance.

Level axes:
  A) Number of hops: 1 at L0, 2 at L3, 3 at L5, 4 at L7+
  B) Extra transversal introduced at L4+
  C) Triangle closure required at L6+
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
    "Lines l1 and l2 are parallel, cut by transversal t. Given {given_txt}. Find {target_txt}. Enumerate each angle relation (corresponding / alternate / supplementary / vertical / triangle-sum) you use. Give the numeric angle in degrees.",
    "In the figure, l1 ∥ l2. Transversal t creates angles as shown. Given {given_txt}, find {target_txt}. Show each hop in your reasoning. Give the degree value.",
    "Parallel lines l1 and l2 with transversal t. Given {given_txt}, determine {target_txt}. List each angle-relation you use. Give the numeric answer in degrees.",
    "Given l1 ∥ l2 and transversal t with {given_txt}, find {target_txt}. Enumerate the angle relationships applied. Give the degree answer.",
    "Two parallel lines l1 and l2 are cut by transversal t. {given_txt} is given. What is {target_txt}? Show each hop. Give the answer in degrees.",
    "From the figure of parallel lines l1, l2 with transversal t: {given_txt} is given. Compute {target_txt}. Give the numeric angle in degrees.",
    "Assume l1 ∥ l2 and t is a transversal. {given_txt}. Solve for {target_txt}. Reason step by step. Give the answer in degrees.",
    "Parallel lines intersected by a transversal. Given {given_txt}, find {target_txt} in degrees. Enumerate hops; give the angle.",
    "l1 ∥ l2 cut by transversal t. {given_txt} — find {target_txt}. Show reasoning; give the degree value.",
    "The figure shows parallel lines l1 and l2 with transversal t. Given {given_txt}, compute {target_txt}. Explain each step; give the answer in degrees.",
    "With l1 ∥ l2 and transversal t: {given_txt}. Find {target_txt}. Describe each angle relationship; give the degree answer.",
    "Parallel lines l1, l2 and transversal t. Given {given_txt}, what is {target_txt} in degrees? List angle relations used; give the result.",
    "In the figure, l1 is parallel to l2, cut by a transversal. {given_txt}. Compute {target_txt}. Enumerate angle-hop relationships; give the degree answer.",
    "Consider l1 ∥ l2, transversal t. {given_txt}. Find the measure of {target_txt} (degrees). Show each step; give the answer.",
    "Given the figure with l1 ∥ l2 and transversal t: {given_txt}. Determine {target_txt} in degrees. Explain relations; give the answer.",
    "Parallel lines l1 and l2 cut by transversal t. {given_txt}. Find {target_txt}. Provide each angle-chaining step; give the degree value.",
]

class ParallelLinesAngleChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "parallel_lines_angle_chain"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            n_hops = 1
        elif level <= 3:
            n_hops = 2
        elif level <= 5:
            n_hops = 3
        else:
            n_hops = 4
        return {"n_hops": n_hops, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 421)
        self._primary_complexity_feature = level

        # Build the angle puzzle:
        # alpha is the transversal-l1 acute angle
        alpha = rng.randint(25, 75)
        # labeled angles (8 positions around two crossings)
        # We'll pick one as "given" and another as "target"
        # Relations: let's label positions 1..8 around the two intersections:
        # Intersection A (l1 meets t): 4 angles a1 a2 a3 a4 going CCW
        # Intersection B (l2 meets t): b1 b2 b3 b4
        # Under parallel: a1 = b1 (corresponding), a1 = a3 (vertical),
        #                 a1 + a2 = 180 (supplementary), a1 = b3 (alternate-int)
        def ang_val(pos, alpha):
            """Return the angle at pos (1-8) given alpha on a1."""
            # pos 1-4: at A; 5-8: at B
            base = (pos - 1) % 4
            if base == 0: return alpha
            if base == 1: return 180 - alpha
            if base == 2: return alpha
            if base == 3: return 180 - alpha
        # pick given pos + target pos ≥ n_hops apart
        given_pos = rng.choice([1, 5, 2, 6])
        # Start picking target to require n_hops hops
        if cfg["n_hops"] == 1:
            target_pos = rng.choice([p for p in range(1, 9) if p != given_pos])
        else:
            # require crossing at least n_hops different relations
            candidate_targets = [p for p in range(1, 9) if p != given_pos]
            target_pos = rng.choice(candidate_targets)

        given_val = ang_val(given_pos, alpha)
        target_val = ang_val(target_pos, alpha)

        given_txt = f"angle at position {given_pos} = {given_val}°"
        target_txt = f"the angle at position {target_pos}"

        # Optional: add extra triangle closure at L6+
        if cfg["level"] >= 6:
            # introduce a third angle in a triangle
            beta = rng.randint(25, 70)
            gamma = 180 - alpha - beta
            # phrase as "an additional line closes a triangle with l1 at angle beta"
            given_txt = (f"angle at position {given_pos} = {given_val}°, "
                         f"and an additional transversal forms a triangle with "
                         f"interior angle {beta}° adjacent to the transversal")
            target_txt = (f"the remaining triangle interior angle (which equals "
                           f"180° − angle at position {given_pos} − {beta}°)")
            target_val = gamma

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(given_txt=given_txt, target_txt=target_txt)
        answer = str(target_val)

        img = self._render(alpha, given_pos, target_pos, given_val, cfg, rng)
        return q, answer, img

    def _render(self, alpha, given_pos, target_pos, given_val, cfg, rng):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Horizontal parallel lines l1 (top y=4) and l2 (bottom y=1)
        ax.plot([-0.5, 7.5], [4, 4], color="black", lw=2.0)
        ax.plot([-0.5, 7.5], [1, 1], color="black", lw=2.0)
        ax.text(7.6, 4, "l1", fontsize=13, fontweight="bold")
        ax.text(7.6, 1, "l2", fontsize=13, fontweight="bold")

        # Transversal t, slope chosen so crossing angle = alpha at top
        # t goes from (0.5, 0.4) to (5.5, 4.8) crossing both lines
        slope = math.tan(math.radians(alpha))
        # Need pts where t crosses y=4 and y=1
        # t: x = x0 + (y - y0)/slope (careful with vertical...)
        # Easiest: pick (x=3, y=1) and (x=3 + 3/tan(alpha), y=4)
        xA = 3 + 3 / math.tan(math.radians(alpha))
        ax.plot([3, xA + 1], [1, 4 + 1 * math.tan(math.radians(alpha))],
                color="black", lw=2.0)
        ax.plot([3 - 0.5, 3], [1 - 0.5 * math.tan(math.radians(alpha)), 1],
                color="black", lw=2.0)
        ax.text(xA + 1.2, 4 + 1 * math.tan(math.radians(alpha)) + 0.1,
                "t", fontsize=13, fontweight="bold")

        # Mark the given angle with its value
        # Annotate position labels at both crossings
        def annotate_pos(pos, x, y):
            ax.text(x, y, str(pos), fontsize=11, color="red", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="red", pad=1))
        # Positions around intersection A at (xA, 4):
        annotate_pos(1, xA - 0.4, 4.3)    # upper-left
        annotate_pos(2, xA + 0.4, 4.3)    # upper-right
        annotate_pos(3, xA + 0.4, 3.7)    # lower-right
        annotate_pos(4, xA - 0.4, 3.7)    # lower-left
        # Positions around intersection B at (3, 1):
        annotate_pos(5, 3 - 0.4, 1.3)
        annotate_pos(6, 3 + 0.4, 1.3)
        annotate_pos(7, 3 + 0.4, 0.7)
        annotate_pos(8, 3 - 0.4, 0.7)

        # Show the given value near the given position
        pos_to_xy = {
            1: (xA - 0.9, 4.3), 2: (xA + 0.9, 4.3),
            3: (xA + 0.9, 3.7), 4: (xA - 0.9, 3.7),
            5: (3 - 0.9, 1.3), 6: (3 + 0.9, 1.3),
            7: (3 + 0.9, 0.7), 8: (3 - 0.9, 0.7),
        }
        gx, gy = pos_to_xy[given_pos]
        ax.text(gx, gy, f"{given_val}°", fontsize=14, fontweight="bold",
                color="blue", ha="center", va="center")
        # Mark target with '?'
        tx, ty = pos_to_xy[target_pos]
        ax.text(tx, ty, "?", fontsize=16, fontweight="bold",
                color="darkgreen", ha="center", va="center")

        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        # remove degree symbol
        for sym in ["°", "\\circ", "degrees", "degree"]:
            pred = pred.replace(sym, "").strip()
            gt = gt.replace(sym, "").strip()
        try:
            return abs(float(pred) - float(gt)) < 0.5
        except ValueError:
            return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_plac"
    os.makedirs(out_dir, exist_ok=True)
    env = ParallelLinesAngleChainQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 29
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[plac L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/plac_s{s}_L{level}.png")
            print(f"[plac L{level} s{s}] A={env._answer}")
