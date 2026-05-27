"""
Coordinate Transform Algebraic QA (v4 G2e, for trans-geo / analytic).

Task: pure symbolic 2D transform: "apply f(x,y) = (-y, x) to P(3, 5)".
Chains 1-3 transformations. Text-only (no image).

Reward: exact tuple match after symbolic equivalence.

Level axes:
  A) Single transform at L0-2, 2 at L3-5, 3 at L6+
  B) Transform types: rotate/reflect/translate at L0-5, dilate + glide at L6+
"""
import random
from typing import Dict, List, Optional, Tuple

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .standalone_base import StandaloneVisualEnv

_TRANSFORMS = [
    ("rotate 90° CCW about origin", lambda p: (-p[1], p[0])),
    ("rotate 90° CW about origin",  lambda p: (p[1], -p[0])),
    ("rotate 180° about origin",    lambda p: (-p[0], -p[1])),
    ("reflect across x-axis",       lambda p: (p[0], -p[1])),
    ("reflect across y-axis",       lambda p: (-p[0], p[1])),
    ("reflect across y = x",        lambda p: (p[1], p[0])),
    ("reflect across y = -x",       lambda p: (-p[1], -p[0])),
    ("translate by (3, 0)",         lambda p: (p[0] + 3, p[1])),
    ("translate by (0, -2)",        lambda p: (p[0], p[1] - 2)),
    ("translate by (-1, 4)",        lambda p: (p[0] - 1, p[1] + 4)),
    ("translate by (2, -3)",        lambda p: (p[0] + 2, p[1] - 3)),
    ("dilate by factor 2 about origin", lambda p: (p[0] * 2, p[1] * 2)),
    ("dilate by factor 1/2 about origin", lambda p: (p[0] / 2, p[1] / 2)),
]

_TEMPLATES = [
    "Starting from the point P({p0}, {p1}), apply the following transforms in sequence: {trans_list}. What are the final coordinates? Output as '(x, y)'. Put in <answer>...</answer>.",
    "Apply the transforms {trans_list} in order to P({p0}, {p1}). Final coordinates? '(x, y)' in <answer>...</answer>.",
    "P = ({p0}, {p1}). Compose {trans_list}. Final position? '(x, y)' in <answer>...</answer>.",
    "Transform the point ({p0}, {p1}) by {trans_list} (in order). Final coordinates? '(x, y)' in <answer>...</answer>.",
    "Given P({p0}, {p1}), apply {trans_list}. Report final '(x, y)' in <answer>...</answer>.",
    "Compose {trans_list} on P({p0}, {p1}). Final? '(x, y)' in <answer>...</answer>.",
    "Sequence: {trans_list}. Apply to P({p0}, {p1}). Final '(x, y)' in <answer>...</answer>.",
    "Transform the point ({p0}, {p1}) step by step using {trans_list}. Output '(x, y)' in <answer>...</answer>.",
    "Point ({p0}, {p1}) under {trans_list}. Final? '(x, y)' in <answer>...</answer>.",
    "Apply {trans_list} to P({p0}, {p1}). Final coordinates in <answer>...</answer>.",
    "Transform sequence {trans_list} on ({p0}, {p1}). Final '(x, y)' in <answer>...</answer>.",
    "Compute the image of ({p0}, {p1}) under {trans_list}. Put '(x, y)' in <answer>...</answer>.",
    "({p0}, {p1}) goes through {trans_list}. Where? '(x, y)' in <answer>...</answer>.",
    "Final coordinates after applying {trans_list} to ({p0}, {p1})? '(x, y)' in <answer>...</answer>.",
    "Apply {trans_list} in sequence to ({p0}, {p1}). Output final coordinates. Put in <answer>...</answer>.",
    "Transforms: {trans_list}. Apply to ({p0}, {p1}). Final? '(x, y)' in <answer>...</answer>.",
]

class CoordTransformAlgebraicQA(StandaloneVisualEnv):
    ENV_NAME = "coord_transform_algebraic"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 deep-redesign: was 90/90/80/90 near-flat. Bump per-level.
        level = max(0, min(level, 9))
        if level <= 1:
            n_transforms = 1
            include_dilate = False
        elif level <= 3:
            n_transforms = 2
            include_dilate = False
        elif level <= 5:
            n_transforms = 3
            include_dilate = False
        elif level <= 7:
            n_transforms = 3
            include_dilate = True
        else:
            n_transforms = 4
            include_dilate = True
        return {"n_transforms": n_transforms, "include_dilate": include_dilate}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 263)
        self._primary_complexity_feature = level

        pool = [t for t in _TRANSFORMS if cfg["include_dilate"] or "dilate" not in t[0]]
        chosen = [rng.choice(pool) for _ in range(cfg["n_transforms"])]

        p = (rng.randint(-6, 6), rng.randint(-6, 6))
        if p == (0, 0):
            p = (rng.randint(1, 6), rng.randint(1, 6))

        # Apply
        current = p
        for _, fn in chosen:
            current = fn(current)
        # Round for integer-ish answers
        if all(x == int(x) for x in current):
            current = (int(current[0]), int(current[1]))
        else:
            current = (round(current[0], 2), round(current[1], 2))
        answer = f"({current[0]}, {current[1]})"

        trans_list = ", then ".join(name for name, _ in chosen)
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(p0=p[0], p1=p[1], trans_list=trans_list)

        img = self._render_blank(q, rng)
        return q, answer, img

    def _render_blank(self, q, rng):
        # Text-only env — render a minimal placeholder image so base class is happy
        fig, ax = plt.subplots(figsize=(4, 2))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2)
        ax.axis("off")
        ax.text(2, 1, "(Text-only problem — see question)",
                fontsize=10, ha="center", va="center", style="italic")
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().replace(" ", "")
        gt = ground_truth.strip().lower().replace(" ", "")
        if pred == gt:
            return True
        # Try parsing tuples
        try:
            pred_vals = [float(x) for x in pred.strip("()").split(",")]
            gt_vals = [float(x) for x in gt.strip("()").split(",")]
            if len(pred_vals) != len(gt_vals):
                return False
            return all(abs(p - g) < 0.02 for p, g in zip(pred_vals, gt_vals))
        except ValueError:
            return False

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_cta"
    os.makedirs(out_dir, exist_ok=True)
    env = CoordTransformAlgebraicQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 211
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[cta L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/cta_s{s}_L{level}.png")
            print(f"[cta L{level} s{s}] A={env._answer}")
