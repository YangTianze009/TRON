"""
Sparse Image Commonsense QA (v4 G12b, for VQA/numeric_commonsense).

Targets:

Failure mode: v3 fabricates specific numbers from sparse images, or
over-counts (5/5 vs correct 4/5).

Task: a minimal / sparse rendering with a count-related or
commonsense question. GT may be 0 (empty), a specific small count, or
'none' / 'cannot determine'.

Reward: MCQ or numeric.

Level axes:
  A) Stimulus sparsity: blank at L0, sparse at L3, ambiguous at L6+
  B) Question type: count at L0-3, fraction at L4-6, commonsense at L7+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES_BLANK = [
    "Count the blue circles inside the rectangular frame. If there are no circles, answer 0. The frame itself does not count. Put the integer in <answer>...</answer>.",
    "How many blue circles are inside the frame? The rectangular border itself does NOT count. Integer in <answer>...</answer>.",
    "Count the blue circles shown inside the rectangle (do not count the rectangle itself). Integer (0 if none) in <answer>...</answer>.",
    "Number of blue circles visible inside the frame (frame itself excluded)? Integer in <answer>...</answer>.",
    "How many blue circles are drawn inside the rectangular frame? Frame does not count. Put integer in <answer>...</answer>.",
    "Count only the blue circles inside the rectangle. The rectangle itself is not counted. Integer in <answer>...</answer>.",
    "Integer: count of blue circles inside the rectangular frame. Frame is excluded. Put in <answer>...</answer>.",
    "How many blue circles do you see inside the rectangle? Don't count the rectangle. Integer in <answer>...</answer>.",
    "Count just the blue circles inside the frame (not the frame). Integer in <answer>...</answer>.",
    "How many blue circles are inside the rectangular boundary? Boundary itself excluded. Integer in <answer>...</answer>.",
    "Tell me the number of blue circles inside the frame. The frame is not counted. Integer in <answer>...</answer>.",
    "Count blue circles inside the rectangle (0 if none, frame doesn't count). Integer in <answer>...</answer>.",
    "Blue circles count (inside the frame only, not the frame)? Integer in <answer>...</answer>.",
    "Integer count of blue circles inside the rectangular frame. Frame doesn't count. Put in <answer>...</answer>.",
    "How many blue circles do you see inside the rectangle? (Do not count the rectangle itself.) Integer in <answer>...</answer>.",
    "Count blue circles inside the frame. Frame is not an item. Put integer in <answer>...</answer>.",
]

_TEMPLATES_FRACTION = [
    "What fraction of the {total_kind} are {target_adj}? Express as a decimal (e.g., 0.6). Put in <answer>...</answer>.",
    "Fraction of {total_kind} that are {target_adj}? Decimal in <answer>...</answer>.",
    "Decimal fraction of {target_adj} {total_kind} in the figure? Put in <answer>...</answer>.",
    "What proportion of the {total_kind} are {target_adj}? Decimal in <answer>...</answer>.",
    "Count {target_adj} / total {total_kind}, as decimal. Put in <answer>...</answer>.",
    "Decimal: {target_adj} {total_kind} / total. Put in <answer>...</answer>.",
    "What is the fraction of {target_adj} {total_kind}? Decimal in <answer>...</answer>.",
    "As a decimal, what fraction are {target_adj}? Put in <answer>...</answer>.",
    "Fraction {target_adj}/{total_kind}? Decimal in <answer>...</answer>.",
    "Decimal proportion of {target_adj}? Put in <answer>...</answer>.",
    "What decimal fraction of the {total_kind} are {target_adj}? Put in <answer>...</answer>.",
    "Compute fraction of {target_adj} as decimal. Put in <answer>...</answer>.",
    "Decimal fraction: {target_adj} of total {total_kind}? Put in <answer>...</answer>.",
    "What is the decimal proportion of {target_adj}? Put in <answer>...</answer>.",
    "Express {target_adj} count / total {total_kind} as decimal. Put in <answer>...</answer>.",
    "Decimal value of {target_adj} fraction? Put in <answer>...</answer>.",
]

class SparseImageCommonsenseQA(StandaloneVisualEnv):
    ENV_NAME = "sparse_image_commonsense"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            qtype = "blank_count"
        elif level <= 6:
            qtype = "fraction"
        else:
            qtype = "fraction"
        return {"qtype": qtype, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 199)
        self._primary_complexity_feature = level

        if cfg["qtype"] == "blank_count":
            n = rng.choice([0, 0, 0, 1, 2])  # mostly empty, forcing 0
            answer = str(n)
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_BLANK[sidx]
            img = self._render_blank(n, rng)
        else:  # fraction
            total = rng.randint(3, 8)
            red_count = rng.randint(0, total)
            fraction = red_count / total
            answer = f"{round(fraction, 2)}"
            total_kind = "circles"
            target_adj = "red"
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_FRACTION[sidx].format(
                total_kind=total_kind, target_adj=target_adj)
            img = self._render_fraction(total, red_count, rng)

        return q, answer, img

    def _render_blank(self, n, rng):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 8); ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        # Blank sign or canvas
        ax.add_patch(mpatches.Rectangle((1, 1), 6, 4, fc="#f5f5f5",
                                         ec="black", lw=2.0))
        # Add n items (if any)
        for i in range(n):
            x = rng.uniform(2, 6)
            y = rng.uniform(2, 4)
            ax.add_patch(mpatches.Circle((x, y), 0.3, fc="#3498db",
                                          ec="black", lw=1.0))
        return self.fig_to_pil(fig)

    def _render_fraction(self, total, red_count, rng):
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, total + 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        for i in range(total):
            color = "#e74c3c" if i < red_count else "#3498db"
            ax.add_patch(mpatches.Circle((i + 0.5, 0.5), 0.35, fc=color,
                                          ec="black", lw=1.2))
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        if pred == gt:
            return True
        try:
            return abs(float(pred) - float(gt)) < 0.02
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_sic"
    os.makedirs(out_dir, exist_ok=True)
    env = SparseImageCommonsenseQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 59
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[sic L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/sic_s{s}_L{level}.png")
            print(f"[sic L{level} s{s}] A={env._answer}")
