"""
Shape Transform Compose QA (v4 G2, for transformation geometry).

Targets: transformation geometry 46.4 -> 35.1 (Δ -11.31, single biggest
math subtask drop at s300).

Failure mode (from 26 base-right/v3-wrong cases pulled 2026-04-23):
  - idx=65 ferris wheel rotation: v3 skips angle accumulation, picks by appearance
  - idx=66 building block 10 turns: v3 looks at 3rd-frame appearance match only
  - idx=45 kangaroo-card 180° rotate: v3 picks a superficial match
  - idx=134 cogs rotation: v3 short-circuits reasoning
  - idx=198 billiard 45° reflection: v3 guesses without mirror-image trick

Fix: environment generates a sequence of 1-3 rigid transforms on a simple
marked shape; the image shows the initial shape, and the model must compose
the transforms mentally to identify which of 4 candidate final shapes matches.

**Reward requires the model to enumerate intermediate states** (Step 1 after
rotate 90° … Step 2 after reflect across y … Step 3 after translate …) before
choosing the MCQ letter. The grader checks for (a) the MCQ answer AND (b)
at least one step description per transform in the composition.

Level axes:
  A) Number of transforms: 1 at L0, 2 at L3-4, 3 at L5+
  B) Transform types: rotate-only at L0-2, rotate+reflect at L3-5, all at L6+
  C) Rotation granularity: 90° only at L0-3, 45° at L4-6, 30° at L7+
  D) Distractors similarity: random at L0-3, "1-step-off" at L4-6, "chirality twin" at L7+
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

def _pick_render_mode_no_sketch(rng) -> str:
    """Avoid 'sketch' mode here — the xkcd font context hangs on this
    cluster (fonts missing) and produces noisy transforms that hurt
    shape-match answer accuracy anyway. Textbook mode also defers to
    matplotlib font loading so we just keep clean rendering."""
    return "clean"

# Base shape: an L-shape (asymmetric, so orientation matters)
# represented as a list of (x, y) unit squares in [0, 2] x [0, 3]
_L_SHAPE = [(0, 0), (0, 1), (0, 2), (1, 0)]
# A different asymmetric shape for variety
_T_SHAPE = [(0, 2), (1, 2), (2, 2), (1, 1), (1, 0)]
# An F-shape
_F_SHAPE = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1)]
# A P-shape
_P_SHAPE = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1)]  # same as F but different mark
# Kangaroo-card inspired shape (dot+rect)
_ARROW = [(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)]
# Bigger zigzag
_ZIGZAG = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]

_SHAPES = {
    "L": _L_SHAPE,
    "T": _T_SHAPE,
    "F": _F_SHAPE,
    "arrow": _ARROW,
    "zigzag": _ZIGZAG,
}

def _apply_rotate(cells: List[Tuple[int, int]], angle_deg: int) -> List[Tuple[int, int]]:
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    new = []
    for (x, y) in cells:
        nx = x * cos_a - y * sin_a
        ny = x * sin_a + y * cos_a
        new.append((round(nx, 3), round(ny, 3)))
    return new

def _apply_reflect(cells: List[Tuple[float, float]], axis: str) -> List[Tuple[float, float]]:
    if axis == "x":  # flip y → -y
        return [(x, -y) for (x, y) in cells]
    elif axis == "y":  # flip x → -x
        return [(-x, y) for (x, y) in cells]
    elif axis == "y=x":  # swap x, y
        return [(y, x) for (x, y) in cells]
    elif axis == "y=-x":
        return [(-y, -x) for (x, y) in cells]
    return cells

def _normalize(cells: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not cells:
        return cells
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    # Deduplicate: a polyomino has unique unit cells.
    return sorted({(round(x - min_x, 3), round(y - min_y, 3)) for (x, y) in cells})

def _describe_transform(t: Dict) -> str:
    if t["type"] == "rotate":
        return f"rotate {t['angle']}° counterclockwise about the origin"
    elif t["type"] == "reflect":
        return f"reflect across the {t['axis']} axis" if t["axis"] in ("x", "y") else f"reflect across the line {t['axis']}"
    elif t["type"] == "translate":
        return f"translate by ({t['dx']}, {t['dy']})"
    return "?"

_TEMPLATES = [
    "The shape in the image starts in position S0. Apply the following transforms in order: {trans_list}. Which of the four options A, B, C, D shows the final position? Enumerate intermediate positions (Step 1 after {t0_desc}, Step 2 after {t1_desc}...) then put the final MCQ letter in <answer>...</answer>.",
    "Given the initial shape (shown as 'Start') and the transform sequence {trans_list}, determine the final position among options A-D. List each intermediate state, then give the letter in <answer>...</answer>.",
    "Starting from the figure labeled 'Start', apply {trans_list} in sequence. Which option (A-D) matches the result? Enumerate each intermediate state, then wrap the MCQ letter in <answer>...</answer>.",
    "The shape is labeled 'Start'. Apply the composition {trans_list} in order. Identify the matching final option A-D. List the intermediate positions, then put the letter in <answer>...</answer>.",
    "Apply the transform sequence {trans_list} to the 'Start' shape. Which final option (A/B/C/D) results? Show each intermediate position, then put the letter in <answer>...</answer>.",
    "From the initial 'Start' shape, perform {trans_list} one transform at a time. Which of the four candidates is the final? Give intermediate states, then answer with the letter in <answer>...</answer>.",
    "Compose the transforms {trans_list} on the 'Start' shape. Match to one of A, B, C, D. List each intermediate state, then put the final letter in <answer>...</answer>.",
    "Perform {trans_list} in order on the 'Start' shape. Which option (A-D) is the final result? Enumerate intermediates then answer in <answer>...</answer>.",
    "The initial shape is labeled 'Start'. Apply {trans_list}. Which final option (A-D) matches? List each intermediate position, then put the letter in <answer>...</answer>.",
    "Given 'Start' and sequence {trans_list}, find the matching final option (A-D). Intermediate states must be enumerated; put the letter in <answer>...</answer>.",
    "Compose the transforms {trans_list} on the starting shape. Match to A, B, C, or D. Show intermediate states, then answer with the letter in <answer>...</answer>.",
    "Given the 'Start' shape, apply the composition {trans_list}. Among A-D, which is the final position? List intermediates then put the letter in <answer>...</answer>.",
    "Apply {trans_list} in the listed order to 'Start'. Which of A/B/C/D is the result? Enumerate each intermediate, then answer in <answer>...</answer>.",
    "The 'Start' shape undergoes transforms {trans_list}. Identify the final option A-D. Intermediate states required; put the letter in <answer>...</answer>.",
    "Trace the 'Start' shape through {trans_list}. Which of A, B, C, D is the outcome? List intermediate states, then answer in <answer>...</answer>.",
    "Transform 'Start' by {trans_list}. Which option A-D matches? Intermediate positions required; put letter in <answer>...</answer>.",
]

class ShapeTransformComposeQA(StandaloneVisualEnv):
    ENV_NAME = "shape_transform_compose"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # v3 design (verified L0=1.00 against Qwen3-VL-8B):
        # - label_cells=True at L0/L1: cells display their (x,y) coords so
        #   the model can read positions + apply the rotation algebra.
        # - show_grid=True through L5: coordinate axes visible.
        # - Past L5, grid hidden — pure visual transformation.
        if level == 0:
            n_transforms = 1
            t_types = ["rotate"]
            rot_angles = [180]
            shape_pool = ["L"]
            show_grid = True
            label_cells = True
        elif level == 1:
            n_transforms = 1
            t_types = ["rotate"]
            rot_angles = [90, 180, 270]
            shape_pool = ["L", "T"]
            show_grid = True
            label_cells = True
        elif level == 2:
            n_transforms = 1
            t_types = ["rotate", "reflect"]
            rot_angles = [90, 180, 270]
            shape_pool = ["L", "T", "F"]
            show_grid = True
            label_cells = True   # keep coord crutch at L2 for signal
        elif level == 3:
            # 2026-04-26: L3 also keeps cell labels for signal
            n_transforms = 1
            t_types = ["rotate", "reflect"]
            rot_angles = [90, 180, 270]
            shape_pool = ["L", "T", "F"]
            show_grid = True
            label_cells = True
        elif level <= 5:
            n_transforms = 2
            t_types = ["rotate", "reflect"]
            rot_angles = [90, 180, 270]
            shape_pool = list(_SHAPES.keys())
            show_grid = True
            label_cells = True   # keep through L5 for difficulty rampup
        elif level <= 7:
            n_transforms = 2
            t_types = ["rotate", "reflect"]
            rot_angles = [90, 180, 270]
            shape_pool = list(_SHAPES.keys())
            show_grid = False
            label_cells = False
        else:
            n_transforms = 3
            t_types = ["rotate", "reflect"]
            rot_angles = [90, 180, 270]
            shape_pool = list(_SHAPES.keys())
            show_grid = False
            label_cells = False
        return {
            "n_transforms": n_transforms,
            "t_types": t_types,
            "rot_angles": rot_angles,
            "shape_pool": shape_pool,
            "show_grid": show_grid,
            "label_cells": label_cells,
            "level": level,
        }

    def _random_transform(self, rng, cfg: Dict) -> Dict:
        tt = rng.choice(cfg["t_types"])
        if tt == "rotate":
            return {"type": "rotate", "angle": rng.choice(cfg["rot_angles"])}
        elif tt == "reflect":
            return {"type": "reflect", "axis": rng.choice(["x", "y", "y=x", "y=-x"])}
        return {"type": "rotate", "angle": 90}

    def _apply_transforms(self, cells, transforms):
        for t in transforms:
            if t["type"] == "rotate":
                cells = _apply_rotate(cells, t["angle"])
            elif t["type"] == "reflect":
                cells = _apply_reflect(cells, t["axis"])
        return cells

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 911)
        self._primary_complexity_feature = level

        shape_name = rng.choice(cfg.get("shape_pool", list(_SHAPES.keys())))
        start = _SHAPES[shape_name]

        # Generate transform sequence
        transforms = [self._random_transform(rng, cfg) for _ in range(cfg["n_transforms"])]

        # Apply to get the correct final
        final = self._apply_transforms(list(start), transforms)
        final_norm = _normalize(final)

        # Generate 3 distractors. At L0, use COMPLETELY DIFFERENT
        # polyominoes (other shapes from the pool) so it's a pure visual
        # shape-match task. At higher levels, use rotated/reflected
        # variants of the same shape (harder to distinguish).
        distractor_pool = ["rotate", "reflect"]
        distractor_angles = [90, 180, 270]
        distractor_axes = ["x", "y", "y=x", "y=-x"]

        distractors = []
        attempts = 0
        if level == 0:
            # Use other base shapes as distractors — visually unambiguous
            other_names = [n for n in _SHAPES.keys() if n != shape_name]
            rng.shuffle(other_names)
            for name in other_names[:3]:
                # Apply a small random rotation so they're not all axis-aligned
                fake = _apply_rotate(_SHAPES[name], rng.choice([0, 90, 180, 270]))
                if _normalize(fake) == final_norm:
                    continue
                distractors.append(fake)
        while len(distractors) < 3 and attempts < 60:
            attempts += 1
            # Slightly-wrong version: flip one transform's parameters
            fake_transforms = [dict(t) for t in transforms]
            pick = rng.randint(0, len(fake_transforms) - 1)
            # Replace with a random transform from the wider pool
            new_type = rng.choice(distractor_pool)
            if new_type == "rotate":
                fake_transforms[pick] = {"type": "rotate",
                                          "angle": rng.choice(distractor_angles)}
            else:
                fake_transforms[pick] = {"type": "reflect",
                                          "axis": rng.choice(distractor_axes)}
            # Skip if identical to original transform
            orig_t = transforms[pick]
            if (fake_transforms[pick].get("type") == orig_t.get("type") and
                fake_transforms[pick].get("angle") == orig_t.get("angle") and
                fake_transforms[pick].get("axis") == orig_t.get("axis")):
                continue
            fake_final = self._apply_transforms(list(start), fake_transforms)
            fake_norm = _normalize(fake_final)
            if fake_norm == final_norm:
                continue
            if any(fake_norm == _normalize(d) for d in distractors):
                continue
            distractors.append(fake_final)

        if len(distractors) < 3:
            # fallback: rotate original by random angle for each missing distractor.
            # Capped at 20 attempts to avoid any infinite loop.
            for _ in range(20):
                if len(distractors) >= 3:
                    break
                fake = _apply_rotate(list(start), rng.choice([90, 180, 270]))
                if _normalize(fake) == final_norm:
                    continue
                if any(_normalize(fake) == _normalize(d) for d in distractors):
                    continue
                distractors.append(fake)
            # If still short of distractors, abandon this seed (caller will retry)
            if len(distractors) < 3:
                return None

        options = [final] + distractors
        rng.shuffle(options)
        correct_idx = None
        for i, opt in enumerate(options):
            if _normalize(opt) == final_norm:
                correct_idx = i
                break
        if correct_idx is None:
            return None
        letter = "ABCD"[correct_idx]

        # Build question text
        trans_list_str = ", then ".join(_describe_transform(t) for t in transforms)
        sidx = (self.seed or 0) % 16
        t0_desc = _describe_transform(transforms[0])
        t1_desc = _describe_transform(transforms[1]) if len(transforms) > 1 else "(no step 2)"
        level = max(0, min(9, int(parameter.get("level", 0))))
        if level <= 2:
            # At low levels, drop the "enumerate intermediates" requirement
            # to give the model freedom to think step by step without being
            # forced into a specific output structure.
            q = (
                f"Apply transforms in order: {trans_list_str}. Which option "
                f"A/B/C/D shows the final shape? Use the (x,y) labels and "
                f"rotation/reflection algebra. Put the final letter in "
                f"<answer>...</answer>."
            )
        else:
            q = _TEMPLATES[sidx].format(
                trans_list=trans_list_str, t0_desc=t0_desc, t1_desc=t1_desc)

        img = self._render(start, options, letter, transforms, cfg, rng)
        return q, letter, img

    def _render(self, start, options, correct_letter, transforms, cfg, rng):
        # 5-panel layout: Start + A/B/C/D, each in its own coordinate plot.
        # Cells drawn at original post-transform coords (NOT normalized) so
        # the model can do coordinate algebra: read (x,y) labels at L0/L1,
        # apply rotation/reflection algebraically, find matching option.
        bg = "#ffffff"
        edge_col = "black"
        edge_lw = 1.4
        fill = rng.choice(["#3498db", "#2ecc71", "#e74c3c", "#f39c12"])

        show_grid = cfg.get("show_grid", False)
        label_cells = cfg.get("label_cells", False)

        # Common symmetric axis range that covers Start + all options
        all_pts = [(x, y) for cs in [start] + list(options) for (x, y) in cs]
        if all_pts:
            min_x = min(x for x, _ in all_pts)
            max_x = max(x for x, _ in all_pts)
            min_y = min(y for _, y in all_pts)
            max_y = max(y for _, y in all_pts)
        else:
            min_x = max_x = min_y = max_y = 0
        rng_max = max(abs(min_x), abs(max_x + 1), abs(min_y), abs(max_y + 1), 3)
        ax_lim = (-rng_max, rng_max)

        def _draw_panel(ax, cells, label):
            ax.set_xlim(*ax_lim)
            ax.set_ylim(*ax_lim)
            ax.set_aspect("equal")
            if show_grid:
                ax.set_xticks(range(int(ax_lim[0]), int(ax_lim[1]) + 1))
                ax.set_yticks(range(int(ax_lim[0]), int(ax_lim[1]) + 1))
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax.axhline(0, color="gray", linewidth=0.8)
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.tick_params(axis="both", labelsize=7)
            else:
                ax.set_xticks([]); ax.set_yticks([])
            for (x, y) in cells:
                rect = mpatches.Rectangle((x, y), 1, 1, fc=fill,
                                           ec=edge_col, lw=edge_lw, alpha=0.9)
                ax.add_patch(rect)
                if label_cells:
                    ax.text(x + 0.5, y + 0.5, f"({int(x)},{int(y)})",
                            fontsize=10, ha="center", va="center",
                            color="white", fontweight="bold")
            ax.set_title(label, fontsize=12, fontweight="bold", pad=4)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
        fig.patch.set_facecolor(bg)
        for a in axes:
            a.set_facecolor(bg)
        _draw_panel(axes[0], start, "Start")
        for i, (opt, letter) in enumerate(zip(options, "ABCD")):
            _draw_panel(axes[i + 1], opt, letter)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=110)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_stc"
    os.makedirs(out_dir, exist_ok=True)
    env = ShapeTransformComposeQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 51
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[stc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/stc_s{s}_L{level}.png")
            print(f"[stc L{level} s{s}] A={env._answer}")
