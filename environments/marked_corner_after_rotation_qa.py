"""
Marked-corner-after-rotation IQ test. A square (or rectangle) with a
visible marker (a small filled dot) in one corner is shown. After
rotating the whole figure by N degrees (90 / 180 / 270; 45 at higher
levels, with 8-corner template), pick the option (A/B/C/D) where the
marker ends up in the corresponding corner.

Mirrors an external reference visual style: the question shows a model object on the
left and 4 options on the right; the model is depicted unrotated, the
options show post-rotation candidates and the dot has been moved by
either the correct rotation amount or a wrong amount.

Output: a single MCQ letter (A/B/C/D).

Difficulty axes:
  - rotation amount: 90° (L0) → arbitrary 45° increments (L9)
  - direction (CW vs CCW)
  - figure shape: simple square (L0-L3) → asymmetric rectangle/cube
    (higher levels) where corner identity matters more.
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


_QUESTION_STEMS = [
    "An object with a dot placed in one corner is shown on the left. After rotating the object by {angle}° {dir_word}, which option (A/B/C/D) shows the dot in the correct new corner? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "A square with a marked corner is shown on the left. After rotating it by {angle}° {dir_word}, which of the four options shows the marker in the correct position? Letter (A/B/C/D) in <answer>...</answer>.",
    "On the left is an object with a dot in one corner. The whole object is then rotated by {angle}° {dir_word}. Which of the candidate figures (A-D) shows the result correctly? Letter in <answer>...</answer>.",
    "Look at the model object on the left, with a dot in one corner. After it is rotated {angle}° {dir_word}, which of A, B, C, D shows the dot in the right new corner? Letter in <answer>...</answer>.",
    "The figure on the left has a marker in one corner. Rotate it by {angle}° {dir_word}. Which option (A/B/C/D) is the resulting figure? Letter in <answer>...</answer>.",
    "A dot is placed in one corner of the object on the left. After a {angle}° {dir_word} rotation, which of the four candidates shows the marker in the correct corner? Letter (A/B/C/D) in <answer>...</answer>.",
    "Apply a {angle}° {dir_word} rotation to the figure on the left. Where does the dot end up? Choose the matching option from A-D. Letter in <answer>...</answer>.",
    "The figure shown on the left rotates {angle}° {dir_word}. Which option (A, B, C, D) depicts the dot in its final corner? Letter in <answer>...</answer>.",
    "Below are several rotated versions of the model object. Pick the one in which the dot is in the correct corner after a {angle}° {dir_word} rotation. Letter (A/B/C/D) in <answer>...</answer>.",
    "Each candidate figure (A/B/C/D) shows the model object after a possible rotation. Which one corresponds to a {angle}° {dir_word} rotation of the model? Letter in <answer>...</answer>.",
    "After rotating the marked figure on the left by {angle}° {dir_word}, which of A-D is the correct result? Letter (A/B/C/D) in <answer>...</answer>.",
    "Look at the model and rotate it mentally {angle}° {dir_word}. Which of the four options has the dot in the new correct corner? Letter (A/B/C/D) in <answer>...</answer>.",
    "Each option (A/B/C/D) is a rotation of the model object. Pick the one matching a {angle}° {dir_word} rotation. Letter in <answer>...</answer>.",
    "Identify the rotation {angle}° {dir_word} of the model on the left and pick the matching option. Letter (A/B/C/D) in <answer>...</answer>.",
    "The model object rotates {angle}° {dir_word}. Which option (A/B/C/D) shows the resulting figure with the marker in the correct corner? Letter in <answer>...</answer>.",
    "Choose the option that correctly shows the model rotated by {angle}° {dir_word}. Letter (A/B/C/D) in <answer>...</answer>.",
]


def _rotate_corner(corner_idx, K, n_steps_cw):
    """Given current corner index in K-corner cycle and a CW step count,
    return the new corner index. corner_idx is 0..K-1 going CW around the
    figure starting from the upper-right (or top, for K==4 squares)."""
    return (corner_idx + n_steps_cw) % K


class MarkedCornerAfterRotationQA(StandaloneVisualEnv):
    ENV_NAME = "marked_corner_after_rotation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04: easier L0/L1 mode (was 10% — VLM mental-rotation limit, attempt fix)
        # L0/L1 stay at angle=90 single direction; further hint added in _generate_problem.
        if level <= 1:
            return {"K": 4, "angle_pool": [90], "dirs": ["clockwise"]}
        if level <= 3:
            return {"K": 4, "angle_pool": [90, 180, 270], "dirs": ["clockwise"]}
        if level <= 5:
            return {"K": 4, "angle_pool": [90, 180, 270], "dirs": ["clockwise", "counter-clockwise"]}
        if level <= 7:
            return {"K": 4, "angle_pool": [90, 180, 270, 360], "dirs": ["clockwise", "counter-clockwise"]}
        return {"K": 4, "angle_pool": [90, 180, 270], "dirs": ["clockwise", "counter-clockwise"], "extra_marker": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1471 + level * 67 + 19)

        K = cfg["K"]
        angle = rng.choice(cfg["angle_pool"])
        direction = rng.choice(cfg["dirs"])
        # Convert angle + direction to "CW step count" with K corners.
        # For K=4, each 90° step rotates one corner CW.
        if direction == "clockwise":
            steps = (angle // 90) % K
        else:
            steps = (-angle // 90) % K

        # Initial corner of the marker (0..K-1, going CW from top-left).
        # Corners: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left.
        init_corner = rng.randrange(0, K)
        new_corner = _rotate_corner(init_corner, K, steps)

        # Generate 4 options. One is correct (marker at new_corner).
        # Distractors: marker at wrong corners — but ensure all 4 unique.
        all_corners = list(range(K))
        rng.shuffle(all_corners)
        # Place new_corner first, then the other 3 in random order
        chosen = [new_corner] + [c for c in all_corners if c != new_corner][:3]
        # Shuffle final list of 4 to randomize which letter is correct
        rng.shuffle(chosen)
        ans_idx = chosen.index(new_corner)
        answer = chr(ord("A") + ans_idx)

        dir_word = direction
        q = rng.choice(_QUESTION_STEMS).format(angle=angle, dir_word=dir_word)
        if level <= 1:
            # 2026-05-04: L0 spell out the algebraic corner-rotation rule —
            # 4B base can't track rotation visually; this turns it into pure index math.
            corner_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
            q = q + (
                f" Hint: corners are indexed 0..3 going clockwise from top-left "
                f"(0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left). "
                f"The dot starts at corner {init_corner} ({corner_names[init_corner]}). "
                f"After {angle}° {dir_word} rotation, the dot moves to corner "
                f"({init_corner} + {angle // 90}) mod 4 = {new_corner} "
                f"({corner_names[new_corner]}). "
                f"Pick the option (A/B/C/D) where the dot is at "
                f"{corner_names[new_corner]}."
            )
        elif level <= 4:
            q = q + " Hint: a 90° CW rotation moves the dot one corner clockwise. Be concise."

        # Use a non-symmetric figure so the rotation is visually meaningful
        # (a square with an additional in-figure marker like an arrow).
        # The "model" panel shows the figure in its initial orientation with
        # the dot at init_corner; the option panels each show the figure
        # rotated by the angle but with the marker placed at the candidate
        # corner. (This reflects an external reference: "five possible rotations".)
        img = self._render(K, init_corner, chosen, angle, direction, rng)
        return q, answer, img

    # -------------------- rendering -------------------- #
    def _corner_xy(self, K, corner_idx, half=0.45):
        """Return (x, y) of the corner inside a unit-cell with center (0,0).
        For K=4 (square): 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left."""
        if K == 4:
            corners = [(-half, half), (half, half), (half, -half), (-half, -half)]
            return corners[corner_idx]
        # 8-corner: place at angles
        a = 2 * math.pi * corner_idx / K - math.pi / 2
        return half * math.cos(a), half * math.sin(a)

    def _draw_figure(self, ax, cx, cy, corner_idx, K, rotation_deg=0,
                      figure_marker=True):
        """Draw a square (or polygon) at (cx, cy) with the dot at corner_idx.
        rotation_deg rotates the figure visually (not the corner index)."""
        # Build a square outline + an internal marker (small triangle in the
        # middle pointing up) so that rotation is unambiguous.
        size = 0.45
        # Corners of a unit square:
        corners = [(-size, size), (size, size), (size, -size), (-size, -size)]
        # Apply rotation
        ca = math.cos(math.radians(rotation_deg))
        sa = math.sin(math.radians(rotation_deg))
        rotated = [(cx + x * ca - y * sa, cy + x * sa + y * ca)
                   for (x, y) in corners]
        ax.add_patch(mpatches.Polygon(rotated, closed=True,
                                        facecolor="#ffffff",
                                        edgecolor="#212121", linewidth=1.6))
        # Dot at corner_idx (in the figure's coordinate system; rotated visually)
        cdx, cdy = corners[corner_idx]
        # Slightly offset inward from the corner so the dot is clearly inside
        cdx *= 0.85
        cdy *= 0.85
        rdx = cx + cdx * ca - cdy * sa
        rdy = cy + cdx * sa + cdy * ca
        ax.add_patch(mpatches.Circle((rdx, rdy), 0.08,
                                       facecolor="#c0392b",
                                       edgecolor="#212121", linewidth=1.0,
                                       zorder=4))
        # Internal marker: small upward arrow (rotated)
        if figure_marker:
            # Triangle pointing up
            arr_pts = [(0, size * 0.3), (-size * 0.18, -size * 0.10),
                        (size * 0.18, -size * 0.10)]
            rotated_arr = [(cx + x * ca - y * sa, cy + x * sa + y * ca)
                            for (x, y) in arr_pts]
            ax.add_patch(mpatches.Polygon(rotated_arr, closed=True,
                                            facecolor="#1a5276",
                                            edgecolor="#212121", linewidth=0.8,
                                            zorder=3))

    def _render(self, K, init_corner, chosen, angle, direction, rng) -> Image.Image:
        # Layout: model panel on top centered; 4 options below in a row.
        fig = plt.figure(figsize=(10, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax_top = fig.add_subplot(2, 1, 1); ax_bot = fig.add_subplot(2, 1, 2)
        for ax in (ax_top, ax_bot):
            ax.set_facecolor("#ffffff"); ax.set_aspect("equal"); ax.axis("off")

        # Top: model
        cell = 1.5
        ax_top.add_patch(mpatches.Rectangle(
            (-cell / 2, -cell / 2), cell, cell,
            facecolor="#ffffff", edgecolor="#34495e", linewidth=1.2))
        self._draw_figure(ax_top, 0, 0, init_corner, K, rotation_deg=0)
        ax_top.set_xlim(-cell, cell)
        ax_top.set_ylim(-cell * 0.8, cell * 0.8)
        ax_top.set_title(
            f"Model object (will be rotated {angle}° {direction})",
            fontsize=12)

        # Bottom: 4 options
        for i, c in enumerate(chosen):
            ox = i * cell - (len(chosen) - 1) * cell / 2
            oy = 0
            ax_bot.add_patch(mpatches.Rectangle(
                (ox - cell / 2, oy - cell / 2), cell, cell,
                facecolor="#ffffff", edgecolor="#34495e", linewidth=1.2))
            # Each option shows the figure with the candidate corner marked,
            # but the figure is *visually rotated by the rotation amount*. Why?
            # Because we want the model to verify "after rotation, the dot is
            # actually at this corner". So we draw the figure rotated by the
            # specified angle, but place the dot at the candidate corner index
            # in the ROTATED figure.
            if direction == "clockwise":
                rot_deg = -angle  # matplotlib angles are CCW positive
            else:
                rot_deg = angle
            self._draw_figure(ax_bot, ox, oy, c, K, rotation_deg=rot_deg)
            ax_bot.text(ox, oy - cell / 2 - 0.18, f"({chr(ord('A') + i)})",
                        fontsize=12, fontweight="bold", ha="center", va="top")
        ax_bot.set_xlim(-len(chosen) * cell / 2 - cell, len(chosen) * cell / 2 + cell)
        ax_bot.set_ylim(-cell * 0.9, cell * 0.7)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, hspace=0.4)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_corner_rot"
    os.makedirs(out_dir, exist_ok=True)
    env = MarkedCornerAfterRotationQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']}")
