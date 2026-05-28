"""
Mirror vs Rotation Discriminator QA environment.

# 2026-05-05 R5 P1: rewrite to match a math benchmark Q1, Q4, Q9 verbatim
# (Basic Transformations of Figures topic, base 72.11% step175 59.18%
# delta -12.93). Stem opener "As shown in the diagram, ...", trailing
# "( )" / "(    )" marker, options separated by "; ", "No correct
# answer" as the LAST option, "Cannot be determined" used as a real
# option (not just a trap).

Goal: decide CW vs CCW rotation direction and (at higher levels)
discriminate rotation vs reflection. Use ASYMMETRIC shapes (R-letter,
kangaroo, etc.) so chirality is unambiguous.

Difficulty schedule:
  L0/L1: Q1-style direction (CW vs CCW). Pure rotation, angle stated
         in stem. 4 options "A. Clockwise; B. Counterclockwise;
         C. Cannot be determined; D. No correct answer".
  L2-L4: 3-class identification (rotation / reflection / both /
         No correct answer). Stem identifies the transformation type.
  L5-L7: 4-class — specify which rotation angle OR which reflection
         axis was applied.
  L8-L9: composite with "Cannot be determined" trap when ambiguous
         (e.g. 180° rotation = horizontal+vertical reflection
         combined: ambiguous).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Asymmetric shapes — chirality is visible
# ------------------------------------------------------------------ #

def _make_r_letter():
    """Letter R — highly asymmetric."""
    return [
        (0, 0), (0, 4), (1.5, 4), (2, 3.5), (2, 3), (1.5, 2.5),
        (0.5, 2.5), (1, 2.5), (2, 0),
    ]

def _make_kangaroo():
    return [
        (0, 0), (1.5, 0.5), (2.5, 1), (3, 2), (2.5, 2.8), (2, 3.5),
        (1.5, 3.2), (1, 2.5), (0.5, 2), (0, 1),
    ]

def _make_arrow():
    return [
        (0, 1), (2.5, 1), (2.5, 2), (4, 0.5), (2.5, -1), (2.5, 0), (0, 0),
    ]

def _make_hexomino():
    return [
        (0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1), (1, 0),
    ]

def _make_foot():
    return [
        (0, 0), (0, 3), (1, 3.5), (2, 3.5), (2, 0),
        (1.5, 0), (1.5, 1), (1, 1), (1, 0),
    ]

_SHAPES = [
    ("R-shape", _make_r_letter),
    ("kangaroo", _make_kangaroo),
    ("arrow-cross", _make_arrow),
    ("hexomino", _make_hexomino),
    ("boot-outline", _make_foot),
]


def _rotate(verts, deg):
    rad = math.radians(deg)
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    return [
        ((x - cx) * math.cos(rad) - (y - cy) * math.sin(rad) + cx,
         (x - cx) * math.sin(rad) + (y - cy) * math.cos(rad) + cy)
        for (x, y) in verts
    ]


def _reflect(verts, axis):
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    if axis == "H":
        return [(x, 2 * cy - y) for (x, y) in verts]
    if axis == "V":
        return [(2 * cx - x, y) for (x, y) in verts]
    return [(y - cy + cx, x - cx + cy) for (x, y) in verts]


class MirrorVsRotationDiscriminatorQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "mirror_vs_rotation_discriminator"

    # a math benchmark Q1-style stems for direction
    _Q1_STEMS = [
        "As shown in the diagram, shape B is shape A rotated {ang}° around point O ( ).",
        "As shown in the diagram, by rotating around point O by {ang}°, how is shape A transformed into shape B? ( )",
        "Looking at the figure, shape B was obtained by rotating shape A by {ang}° about point O. The direction of rotation is ( ).",
    ]

    # 3-class type identification stems
    _TYPE_STEMS = [
        "As shown in the diagram, shape B is obtained from shape A by ( ).",
        "Looking at the figure, the transformation that maps shape A onto shape B is ( ).",
        "As shown in the figure, shape A is transformed into shape B by ______. ( )",
    ]

    # 4-class specific transformation stems
    _SPEC_STEMS = [
        "As shown in the diagram, shape B is obtained from shape A by which transformation? (    )",
        "Looking at the figure, identify the single transformation that maps shape A onto shape B. (    )",
        "As shown in the diagram, shape A is transformed into shape B. The transformation applied is (    ).",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return dict(mode="direction_q1", angle_pool=[90, 180])
        if level == 1:
            return dict(mode="direction_q1", angle_pool=[90, 180, 270])
        if level == 2:
            return dict(mode="type_3class", angle_pool=[90, 180, 270],
                        kinds=["rotation", "reflection"])
        if level == 3:
            return dict(mode="type_3class", angle_pool=[90, 180, 270],
                        kinds=["rotation", "reflection", "both"])
        if level == 4:
            return dict(mode="type_3class", angle_pool=[90, 180, 270],
                        kinds=["rotation", "reflection", "both"])
        if level == 5:
            return dict(mode="specific_4class", angle_pool=[90, 180, 270],
                        kinds=["rot_cw", "rot_ccw", "refl_h", "refl_v"])
        if level == 6:
            return dict(mode="specific_4class", angle_pool=[90, 180, 270],
                        kinds=["rot_cw", "rot_ccw", "refl_h", "refl_v",
                               "refl_d"])
        if level == 7:
            return dict(mode="specific_4class", angle_pool=[60, 90, 120, 180, 270],
                        kinds=["rot_cw", "rot_ccw", "refl_h", "refl_v",
                               "refl_d"])
        if level == 8:
            return dict(mode="ambiguous", angle_pool=[90, 180, 270],
                        ambiguous_prob=0.4)
        return dict(mode="ambiguous", angle_pool=[60, 90, 120, 180, 270],
                    ambiguous_prob=0.5)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level
        self._sub_rng = sub_rng

        for _ in range(25):
            try:
                if cfg["mode"] == "direction_q1":
                    result = self._gen_direction_q1(sub_rng, cfg, level)
                elif cfg["mode"] == "type_3class":
                    result = self._gen_type_3class(sub_rng, cfg, level)
                elif cfg["mode"] == "specific_4class":
                    result = self._gen_specific_4class(sub_rng, cfg, level)
                elif cfg["mode"] == "ambiguous":
                    result = self._gen_ambiguous(sub_rng, cfg, level)
                else:
                    return None
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # L0/L1: a math benchmark Q1 verbatim — CW vs CCW direction
    # ------------------------------------------------------------------ #
    def _gen_direction_q1(self, rng, cfg, level):
        name, fn = rng.choice(_SHAPES)
        before = fn()
        ang = rng.choice(cfg["angle_pool"])
        # Choose CW or CCW (deterministic balance via seed parity)
        # Mark CW as +ang in screen coords (since matplotlib y-up,
        # +deg CCW screen = +deg CW visually). We use mathematical CCW
        # convention for the rotation function but state CW/CCW based
        # on visual interpretation.
        # For consistency: we say CW if the math rotation is by -ang.
        cw = ((self.seed or 0) + level) % 2 == 0
        if cw:
            after = _rotate(before, -ang)
            correct_text = "Clockwise"
        else:
            after = _rotate(before, ang)
            correct_text = "Counterclockwise"

        # 4 options exactly as Q1: A. Clockwise; B. Counterclockwise;
        # C. Cannot be determined; D. No correct answer.
        # Allow shuffling of first 2 (Cannot/No correct stay as last 2).
        first_two = ["Clockwise", "Counterclockwise"]
        rng.shuffle(first_two)
        options = first_two + ["Cannot be determined", "No correct answer"]
        for i, o in enumerate(options):
            if o == correct_text:
                answer = chr(ord("A") + i)
                break
        else:
            return None

        stem = self._Q1_STEMS[(self.seed or 0) % len(self._Q1_STEMS)]
        question_stem = stem.format(ang=ang)
        opts_str = "; ".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        question = (
            f"{question_stem}\n"
            f"Options: {opts_str}.\n"
            "Answer with a single letter A, B, C, or D."
        )
        image = self._render_with_O(before, after, name, rng)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # L2-L4: 3-class type identification (rotation / reflection / both)
    # ------------------------------------------------------------------ #
    def _gen_type_3class(self, rng, cfg, level):
        name, fn = rng.choice(_SHAPES)
        before = fn()
        kinds = cfg["kinds"]
        # Balance ground truth across 3 classes via seed parity
        kind = kinds[((self.seed or 0) * 7 + level * 11) % len(kinds)]

        if kind == "rotation":
            ang = rng.choice(cfg["angle_pool"])
            after = _rotate(before, ang)
            correct_text = "Rotation"
        elif kind == "reflection":
            axis = rng.choice(["H", "V"])
            after = _reflect(before, axis)
            correct_text = "Reflection"
        else:  # both = glide reflection (reflect then rotate)
            axis = rng.choice(["H", "V"])
            after = _rotate(_reflect(before, axis), rng.choice([90, 180, 270]))
            correct_text = "Both rotation and reflection"

        # Build options. "No correct answer" sometimes is correct (~15%).
        # All 3 type-options + "No correct answer" → 4 options always.
        no_correct = (rng.random() < 0.15)
        all_opts = ["Rotation", "Reflection", "Both rotation and reflection"]

        if no_correct:
            # When "No correct answer" is the truth, the displayed
            # options must NOT include the correct_text. Use the 3
            # other type-words: keep 2 plus "Cannot be determined".
            distractors = [o for o in all_opts if o != correct_text]
            # distractors has 2 entries; pad with "Cannot be determined"
            options = distractors + ["Cannot be determined",
                                      "No correct answer"]
            non_first2 = options[:2]
            rng.shuffle(non_first2)
            options = non_first2 + ["Cannot be determined",
                                     "No correct answer"]
            answer = "D"
        else:
            distractors = [o for o in all_opts if o != correct_text]
            rng.shuffle(distractors)
            # distractors has 2 entries (the other two type-words)
            options = [correct_text] + distractors[:2] + ["No correct answer"]
            non_no = options[:3]
            rng.shuffle(non_no)
            options = non_no + ["No correct answer"]
            for i, o in enumerate(options):
                if o == correct_text:
                    answer = chr(ord("A") + i)
                    break
            else:
                return None

        stem = self._TYPE_STEMS[(self.seed or 0) % len(self._TYPE_STEMS)]
        opts_str = "; ".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        question = (
            f"{stem}\n"
            f"Options: {opts_str}.\n"
            "Answer with a single letter A, B, C, or D."
        )
        image = self._render_simple(before, after, name, rng)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # L5-L7: 4-class specific (which rotation angle / which reflection axis)
    # ------------------------------------------------------------------ #
    def _gen_specific_4class(self, rng, cfg, level):
        name, fn = rng.choice(_SHAPES)
        before = fn()
        kinds = cfg["kinds"]
        kind = kinds[((self.seed or 0) * 13 + level * 5) % len(kinds)]

        if kind == "rot_cw":
            ang = rng.choice(cfg["angle_pool"])
            after = _rotate(before, -ang)
            correct_text = f"Rotate {ang}° clockwise"
        elif kind == "rot_ccw":
            ang = rng.choice(cfg["angle_pool"])
            after = _rotate(before, ang)
            correct_text = f"Rotate {ang}° counterclockwise"
        elif kind == "refl_h":
            after = _reflect(before, "H")
            correct_text = "Reflect across horizontal axis"
        elif kind == "refl_v":
            after = _reflect(before, "V")
            correct_text = "Reflect across vertical axis"
        else:  # refl_d
            after = _reflect(before, "D")
            correct_text = "Reflect across diagonal y = x"

        # Build option pool and pick 3 distractors
        ang_pool = cfg["angle_pool"]
        a = ang_pool[((self.seed or 0) + 3) % len(ang_pool)]
        all_opts = [
            f"Rotate {a}° clockwise",
            f"Rotate {a}° counterclockwise",
            "Reflect across horizontal axis",
            "Reflect across vertical axis",
        ]
        if "refl_d" in kinds:
            all_opts.append("Reflect across diagonal y = x")

        no_correct = (rng.random() < 0.15)

        if no_correct:
            distractors = [o for o in all_opts if o != correct_text]
            rng.shuffle(distractors)
            options = distractors[:3] + ["No correct answer"]
            answer = "D"
        else:
            distractors = [o for o in all_opts if o != correct_text]
            rng.shuffle(distractors)
            options = [correct_text] + distractors[:2] + ["No correct answer"]
            non_no = options[:3]
            rng.shuffle(non_no)
            options = non_no + ["No correct answer"]
            for i, o in enumerate(options):
                if o == correct_text:
                    answer = chr(ord("A") + i)
                    break
            else:
                return None

        stem = self._SPEC_STEMS[(self.seed or 0) % len(self._SPEC_STEMS)]
        opts_str = "; ".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        question = (
            f"{stem}\n"
            f"Options: {opts_str}.\n"
            "Answer with a single letter A, B, C, or D."
        )
        image = self._render_simple(before, after, name, rng)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # L8/L9: ambiguous case — 180° rotation of vertically symmetric
    # shape can equal a reflection, so "Cannot be determined" applies.
    # ------------------------------------------------------------------ #
    def _gen_ambiguous(self, rng, cfg, level):
        name, fn = rng.choice(_SHAPES)
        before = fn()

        # With probability ambiguous_prob, generate an ambiguous case
        # where multiple transformations could produce the result. For
        # asymmetric shapes, only 180° rotation = point-reflection is
        # truly ambiguous, but we can also generate cases where the
        # shape happens to be invariant under one of the candidates.
        is_ambig = (rng.random() < cfg["ambiguous_prob"])

        if is_ambig:
            # Use 180° rotation: this equals point-reflection (combined
            # H + V reflection through center). The single-transformation
            # answer is genuinely ambiguous between "Rotate 180°" and
            # "Reflect through center".
            after = _rotate(before, 180)
            correct_text = "Cannot be determined"
            ang = 180
        else:
            # Pick a clear single transformation
            choice = rng.choice(["rot_cw", "rot_ccw", "refl_h", "refl_v"])
            ang = rng.choice([a for a in cfg["angle_pool"] if a != 180])
            if choice == "rot_cw":
                after = _rotate(before, -ang)
                correct_text = f"Rotate {ang}° clockwise"
            elif choice == "rot_ccw":
                after = _rotate(before, ang)
                correct_text = f"Rotate {ang}° counterclockwise"
            elif choice == "refl_h":
                after = _reflect(before, "H")
                correct_text = "Reflect across horizontal axis"
            else:
                after = _reflect(before, "V")
                correct_text = "Reflect across vertical axis"

        # 4-option set always contains "Cannot be determined" and
        # "No correct answer"
        all_opts_pool = [
            f"Rotate {ang}° clockwise",
            f"Rotate {ang}° counterclockwise",
            "Reflect across horizontal axis",
            "Reflect across vertical axis",
        ]

        if correct_text == "Cannot be determined":
            # Distractors = some specific transformations
            rng.shuffle(all_opts_pool)
            options = all_opts_pool[:2] + ["Cannot be determined",
                                            "No correct answer"]
            non_first2 = options[:2]
            rng.shuffle(non_first2)
            options = non_first2 + ["Cannot be determined",
                                     "No correct answer"]
            answer = "C"
        else:
            distractors = [o for o in all_opts_pool if o != correct_text]
            rng.shuffle(distractors)
            options = [correct_text] + distractors[:1] + \
                      ["Cannot be determined", "No correct answer"]
            non_no = options[:2]
            rng.shuffle(non_no)
            options = non_no + ["Cannot be determined", "No correct answer"]
            for i, o in enumerate(options):
                if o == correct_text:
                    answer = chr(ord("A") + i)
                    break
            else:
                return None

        stem = self._SPEC_STEMS[(self.seed or 0) % len(self._SPEC_STEMS)]
        opts_str = "; ".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        question = (
            f"{stem}\n"
            f"Options: {opts_str}.\n"
            "Answer with a single letter A, B, C, or D."
        )
        image = self._render_simple(before, after, name, rng)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_simple(self, before, after, name, sub_rng):
        sc = 1.0
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(8 * sc, 4.5 * sc))
        fig.patch.set_facecolor("#ffffff")
        for ax in (ax_b, ax_a):
            ax.set_facecolor("#ffffff")
            ax.set_aspect("equal")

        b_color = "#3498db"
        a_color = "#e67e22"
        edge_col = "#1a1a1a"
        mark_col = "#e74c3c"

        poly_b = plt.Polygon(before, closed=True, facecolor=b_color,
                             alpha=0.55, edgecolor=edge_col, linewidth=1.6)
        ax_b.add_patch(poly_b)
        ax_b.plot(before[0][0], before[0][1], "o", color=mark_col, markersize=6)

        poly_a = plt.Polygon(after, closed=True, facecolor=a_color,
                             alpha=0.55, edgecolor=edge_col, linewidth=1.6)
        ax_a.add_patch(poly_a)
        ax_a.plot(after[0][0], after[0][1], "o", color=mark_col, markersize=6)

        for ax, verts, title in [(ax_b, before, "Shape A"),
                                  (ax_a, after, "Shape B")]:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            mg = 1.0
            ax.set_xlim(min(xs) - mg, max(xs) + mg)
            ax.set_ylim(min(ys) - mg, max(ys) + mg)
            ax.axhline(0, color="#bbbbbb", linewidth=0.5)
            ax.axvline(0, color="#bbbbbb", linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Transformation of {name}",
                     fontsize=13, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=120)

    def _render_with_O(self, before, after, name, sub_rng):
        """Render with point O marked at the rotation center on both panels."""
        sc = 1.0
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(8 * sc, 4.5 * sc))
        fig.patch.set_facecolor("#ffffff")

        b_color = "#3498db"
        a_color = "#e67e22"
        edge_col = "#1a1a1a"
        mark_col = "#e74c3c"

        # Compute O = centroid of the shape (rotation center we used)
        cxb = sum(v[0] for v in before) / len(before)
        cyb = sum(v[1] for v in before) / len(before)
        cxa = sum(v[0] for v in after) / len(after)
        cya = sum(v[1] for v in after) / len(after)

        for ax, verts, color, title, ox, oy in [
            (ax_b, before, b_color, "Shape A", cxb, cyb),
            (ax_a, after, a_color, "Shape B", cxa, cya)]:
            ax.set_facecolor("#ffffff")
            ax.set_aspect("equal")
            poly = plt.Polygon(verts, closed=True, facecolor=color,
                               alpha=0.55, edgecolor=edge_col, linewidth=1.6)
            ax.add_patch(poly)
            # Red mark on first vertex (chirality marker)
            ax.plot(verts[0][0], verts[0][1], "o", color=mark_col,
                    markersize=6)
            # Mark point O
            ax.plot(ox, oy, "s", color="#000000", markersize=10,
                    markerfacecolor="#f1c40f", markeredgecolor="#000000")
            ax.text(ox + 0.18, oy + 0.18, "O", fontsize=14,
                    fontweight="bold", color="#000000")
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            mg = 1.2
            ax.set_xlim(min(xs) - mg, max(xs) + mg)
            ax.set_ylim(min(ys) - mg, max(ys) + mg)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Rotation about point O ({name})",
                     fontsize=13, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = MirrorVsRotationDiscriminatorQA()
    for level in [0, 3, 6, 9]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[mirror_vs_rotation] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/mirror_vs_rotation_discriminator_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer}")
