"""
Rotate Marked Sheet QA (D59, P3 — reference plane geometry).

Reference qid 127:
  "The ♠ is drawn on the sheet. We turn the sheet clockwise through 90°
   and then turn counter-clockwise 180°. Which figure can we now see?
   choice: (A) (B) (C) (D)."  Ans: D

This env shows a rectangular sheet with a marked shape (heart, spade,
star, arrow, etc.) drawn on it, then asks which of four candidate
figures matches the result of applying a sequence of rotations.

Verifier: MCQ letter A/B/C/D (`\\boxed{D}`).

Difficulty:
  L0..L2 — single rotation (90, 180, 270).
  L3..L5 — two-rotation composition.
  L6..L9 — three-rotation composition + harder distractors.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Each shape has a draw method that emits points relative to (0, 0)
# centred orientation = 0 deg. Shapes are intentionally chiral / not
# rotationally symmetric, so different orientations look different.
def _spade_pts():
    return [(0, 1.0), (0.6, 0.4), (0.6, -0.2), (0.3, -0.4),
            (0.15, -0.2), (0.15, -0.7), (0.4, -0.85), (-0.4, -0.85),
            (-0.15, -0.7), (-0.15, -0.2), (-0.3, -0.4), (-0.6, -0.2),
            (-0.6, 0.4), (0, 1.0)]


def _arrow_pts():
    return [(-0.7, 0.0), (0.4, 0.0), (0.4, 0.3), (0.85, 0.0),
            (0.4, -0.3), (0.4, 0.0)]


def _flag_pts():
    return [(-0.6, -0.7), (-0.6, 0.7), (0.6, 0.5), (-0.4, 0.4),
            (0.55, 0.0), (-0.4, -0.05), (0.5, -0.55)]


def _hook_pts():
    return [(-0.3, 0.7), (0.3, 0.7), (0.3, -0.4),
            (-0.4, -0.4), (-0.4, -0.7), (0.4, -0.7), (0.4, -0.85)]


def _L_pts():
    return [(-0.4, 0.7), (-0.4, -0.7), (0.5, -0.7), (0.5, -0.4),
            (-0.1, -0.4), (-0.1, 0.7), (-0.4, 0.7)]


def _F_pts():
    # Asymmetric letter F
    return [(-0.4, -0.7), (-0.4, 0.7), (0.4, 0.7), (0.4, 0.4),
            (-0.1, 0.4), (-0.1, 0.1), (0.3, 0.1), (0.3, -0.2),
            (-0.1, -0.2), (-0.1, -0.7), (-0.4, -0.7)]


SHAPE_POOL = {
    "spade": _spade_pts,
    "arrow": _arrow_pts,
    "flag": _flag_pts,
    "hook": _hook_pts,
    "L": _L_pts,
    "F": _F_pts,
}

SHAPE_DESCR = {
    "spade": "spade ♠ symbol",
    "arrow": "right-pointing arrow",
    "flag": "flag pennant",
    "hook": "letter J / hook",
    "L": "letter L",
    "F": "letter F",
}


def _rotate_pts(pts, deg):
    rad = math.radians(deg)
    cs, sn = math.cos(rad), math.sin(rad)
    return [(x * cs - y * sn, x * sn + y * cs) for (x, y) in pts]


class RotateMarkedSheetQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "rotate_marked_sheet"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_rotations": 1, "rot_pool": [90, 180, 270]}
        if level <= 5:
            return {"n_rotations": 2, "rot_pool": [90, 180, 270, -90]}
        if level <= 7:
            return {"n_rotations": 3, "rot_pool": [90, 180, 270, -90]}
        return {"n_rotations": 3, "rot_pool": [60, 90, 120, 180, 270, -90]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7561 + level * 233 + 17)

        for _ in range(20):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg):
        shape_name = rng.choice(list(SHAPE_POOL.keys()))
        shape_pts = SHAPE_POOL[shape_name]()

        # Compose rotations
        n_rot = cfg["n_rotations"]
        rotations = []
        rot_descr = []
        for _ in range(n_rot):
            angle = rng.choice(cfg["rot_pool"])
            direction = rng.choice(["cw", "ccw"])
            # CW rotation by θ° = anticlockwise rotation by -θ°
            actual = -angle if direction == "cw" else angle
            rotations.append(actual)
            d_text = "clockwise" if direction == "cw" else "counter-clockwise"
            rot_descr.append(f"{abs(angle)}° {d_text}")

        total_rot = sum(rotations) % 360

        # Build the 4 MCQ options
        options_angles = [total_rot]
        # Distractors: other plausible orientations
        distractors_pool = [(total_rot + 90) % 360,
                            (total_rot - 90) % 360,
                            (total_rot + 180) % 360,
                            (total_rot + 45) % 360,
                            (total_rot + 270) % 360]
        rng.shuffle(distractors_pool)
        for ang in distractors_pool:
            if ang not in options_angles:
                options_angles.append(ang)
            if len(options_angles) == 4:
                break
        if len(options_angles) < 4:
            return None
        rng.shuffle(options_angles)
        correct_idx = options_angles.index(total_rot)
        correct_letter = chr(ord("A") + correct_idx)

        rot_steps_text = " then ".join(rot_descr)

        question = (
            f"A {SHAPE_DESCR[shape_name]} is drawn on the sheet (top-left "
            f"of the figure). We turn the sheet {rot_steps_text}. Which of "
            f"the figures (A) (B) (C) (D) shows what we now see? Answer "
            f"with a single letter A, B, C, or D."
        )

        img = self._render(shape_pts, options_angles, shape_name)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    def _render(self, shape_pts, options_angles, shape_name) -> Image.Image:
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 6.0), dpi=110)
        fig.patch.set_facecolor("#ffffff")

        # Top-left: original sheet
        ax_orig = axes[0, 0]
        self._draw_sheet(ax_orig, shape_pts, "original")

        # Top-middle: blank / instructions
        axes[0, 1].axis("off")
        axes[0, 1].text(0.5, 0.5,
                        f"Original\nshape:\n{SHAPE_DESCR[shape_name]}",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold",
                        transform=axes[0, 1].transAxes)
        # Top-right: blank
        axes[0, 2].axis("off")

        # Bottom row: 4 options (we use bottom-left, bottom-mid, bottom-
        # right and reuse top-right as A/B/C/D). Layout: A,B,C,D in
        # bottom-left, bottom-mid, bottom-right, top-right.
        # Arrangement: A=top-right, B=bottom-left, C=bottom-mid, D=bottom-right
        positions = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
        labels = ["(A)", "(B)", "(C)", "(D)"]
        for ax, ang, lbl in zip(positions, options_angles, labels):
            rot_pts = _rotate_pts(shape_pts, ang)
            self._draw_sheet(ax, rot_pts, lbl)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=110)

    @staticmethod
    def _draw_sheet(ax, pts, title):
        """Draw a rectangular sheet with the given shape points."""
        ax.add_patch(plt.Rectangle((-1.2, -1.2), 2.4, 2.4,
                                   facecolor="#fffefa",
                                   edgecolor="#7f8c8d", linewidth=1.5))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.fill(xs, ys, color="#1a5276", alpha=0.85)
        ax.plot(xs, ys, color="#1a5276", linewidth=1.0)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)


if __name__ == "__main__":
    env = RotateMarkedSheetQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok}; A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("(Z)")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
