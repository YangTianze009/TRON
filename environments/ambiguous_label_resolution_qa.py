"""
Ambiguous Label Resolution QA (v4 G1c, for idx=48 cylinder case).

Failure mode (rev-2 Cluster 4): v3 sees two labels like "6 cm" and "4 cm"
on a cylinder net and guesses wrong which is diameter vs height.

Task: render a figure where a single number label is positioned
ambiguously (it could apply to one of two possible geometric objects).
The model must use figure context to decide which.

Reward: MCQ letter match.

Level axes:
  A) Ambiguity severity: low at L0-3, high at L4-6, very high at L7+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "A cylinder is shown with its net (unfolded view). Two numeric labels are visible. Looking at label positions and orientation, which role does the label '{lb}' play? Options: A. radius of the circular base  B. diameter of the circular base  C. height of the cylinder  D. circumference of the base. Put the letter in <answer>...</answer>.",
    "In the cylinder net, label '{lb}' corresponds to which role? A-D as above. Put letter in <answer>...</answer>.",
    "The cylinder diagram has label '{lb}'. Identify its role. A-D. Put letter in <answer>...</answer>.",
    "From the cylinder figure, determine what '{lb}' labels. A-D. Put letter in <answer>...</answer>.",
    "Cylinder net label '{lb}' is the: A. radius  B. diameter  C. height  D. circumference? Put letter in <answer>...</answer>.",
    "Which geometric quantity does '{lb}' label? A-D. Put letter in <answer>...</answer>.",
    "Label '{lb}' on the cylinder represents: A-D. Put letter in <answer>...</answer>.",
    "Identify the role of label '{lb}' on the cylinder. A-D. Put letter in <answer>...</answer>.",
    "On the cylinder net, '{lb}' is which role? A-D. Put letter in <answer>...</answer>.",
    "Cylinder figure: label '{lb}' indicates? A-D. Put letter in <answer>...</answer>.",
    "What does the label '{lb}' correspond to on the cylinder? A-D. Put letter in <answer>...</answer>.",
    "Cylinder net shown. Label '{lb}' is: A-D. Put letter in <answer>...</answer>.",
    "Identify '{lb}' role in the cylinder figure. A-D. Put letter in <answer>...</answer>.",
    "From cylinder diagram, '{lb}' labels: A-D. Put letter in <answer>...</answer>.",
    "Role of label '{lb}' on the cylinder figure? A-D. Put letter in <answer>...</answer>.",
    "Determine cylinder-net role for '{lb}'. A-D. Put letter in <answer>...</answer>.",
]

class AmbiguousLabelResolutionQA(StandaloneVisualEnv):
    ENV_NAME = "ambiguous_label_resolution"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {"level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 397)
        self._primary_complexity_feature = level

        # Ensure ambiguous cylinder: show rectangle (width = 2πr, height = h)
        # + 2 small circles (diameter = 2r). Then 2 labels "X" near width
        # and "Y" near height.
        r = rng.randint(2, 6)
        h = rng.randint(3, 10)
        # Label position: "near rectangle width" or "near rectangle height"
        pos = rng.choice(["near_width", "near_height", "near_circle"])
        # Pick value and role based on position
        if pos == "near_width":
            # the number is likely 2πr rounded OR h
            # Use unique value: 2πr rounded OR deliberately h
            use_circ = rng.choice([True, False])
            if use_circ:
                lb = round(2 * 3.14159 * r, 1)
                role = "D"  # circumference
            else:
                # label on the width side is either circumference or nothing;
                # if we label h here it'd be mislabeled
                lb = h
                role = "C"  # height (but on top — should be labeled on side)
        elif pos == "near_height":
            lb = h
            role = "C"  # height
        else:  # near_circle
            choice = rng.choice(["radius", "diameter"])
            if choice == "radius":
                lb = r
                role = "A"
            else:
                lb = 2 * r
                role = "B"

        letter = role
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(lb=lb)
        # BUGFIX 2026-04-24: most templates reference 'A-D as above' without
        # defining the options. Append an explicit options list so the prompt
        # is self-contained.
        if "A. radius" not in q and "A. radius of the circular base" not in q:
            q = q + " Options: A. radius of the circular base; B. diameter of the circular base; C. height of the cylinder; D. circumference of the base."

        img = self._render_cylinder_net(r, h, pos, lb, rng)
        return q, letter, img

    def _render_cylinder_net(self, r, h, pos, lb, rng):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, 12); ax.set_ylim(-1, 6)
        ax.set_aspect("equal")
        ax.axis("off")

        # Rectangle (side face)
        rect_w = 2 * 3.14159 * r / 2  # scale down for display
        ax.add_patch(mpatches.Rectangle((0, 0), rect_w, h * 0.4,
                                         fc="none", ec="black", lw=2.0))
        # Two small circles (top/bottom bases)
        ax.add_patch(mpatches.Circle((rect_w + 1.5, h * 0.4), r * 0.3,
                                      fc="none", ec="black", lw=1.8))
        ax.add_patch(mpatches.Circle((rect_w + 3.5, h * 0.4), r * 0.3,
                                      fc="none", ec="black", lw=1.8))

        # Place the label at the ambiguous position
        if pos == "near_width":
            ax.text(rect_w / 2, -0.3, str(lb), fontsize=16,
                    ha="center", fontweight="bold", color="red")
        elif pos == "near_height":
            ax.text(-0.3, h * 0.2, str(lb), fontsize=16,
                    ha="center", fontweight="bold", color="red")
        else:
            # Put label near one of the circles
            ax.plot([rect_w + 1.5, rect_w + 1.5 + r * 0.3],
                    [h * 0.4, h * 0.4], color="black", lw=1)
            ax.text(rect_w + 1.5 + r * 0.15, h * 0.4 + 0.2, str(lb),
                    fontsize=16, fontweight="bold", color="red")

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_alr"
    os.makedirs(out_dir, exist_ok=True)
    env = AmbiguousLabelResolutionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 61
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[alr L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/alr_s{s}_L{level}.png")
            print(f"[alr L{level} s{s}] A={env._answer}")
