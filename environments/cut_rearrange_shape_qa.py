"""
Cut-Rearrange Shape QA.

2026-05-05 R5 B5A: REWRITTEN — pure single-letter MCQ A-D / A-E.

Renders an original 2D figure with one (or two) labelled cut lines, then
asks which target shape can be made by rearranging the cut pieces (without
overlap or gaps).

Per-level layout
  L0/L1 — 4-MCQ. Single-cut simplest templates (square/rectangle midline,
          square diagonal). 6.5x6 figure, large fonts, contrasting cut color.
          Concise reasoning hint (no answer leak).
  L2/L3 — 4-MCQ. Single-cut, expanded template pool to 4 templates.
  L4/L5 — 5-MCQ with "E. No correct answer" (15% trap). 5 templates incl
          parallelogram altitude / right trapezoid.
  L6/L7 — 5-MCQ. 25% trap. All 6 templates.
  L8/L9 — 5-MCQ. 30/40% trap. All templates plus tighter distractor sets
          ("circle" replaced with closely-confusable shapes).

Strict MCQ design — _answer is single letter A-E. Verifier handled by
standalone_base default (no override needed).
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Each template:
#   shape_id          — render key
#   orig_desc         — short text describing the original shape + cut
#   cut_desc          — short text describing the resulting pieces
#   correct           — name of correct rearranged shape (option text)
#   distractors       — 3 plausible-but-wrong rearrangements
#   tight_distractors — 3 closer-to-correct distractors used at L8/L9
_TEMPLATES = [
    # 0. Rectangle 4x2 cut diagonally → two right triangles → can form an isosceles triangle
    ("rectangle_diag",
     "A rectangle is cut along the diagonal from one corner to the opposite corner, dividing it into two pieces.",
     "two congruent right triangles",
     "isosceles triangle",
     ["square", "regular pentagon", "circle"],
     ["right triangle", "rectangle", "parallelogram"]),
    # 1. Square 3x3 cut horizontally midline → two equal rectangles → form a longer rectangle
    ("square_midhoriz",
     "A square is cut along its horizontal midline, dividing it into two equal rectangles.",
     "two equal (congruent) rectangles",
     "rectangle",
     ["triangle", "trapezoid", "circle"],
     ["square", "parallelogram", "right trapezoid"]),
    # 2. Square cut along diagonal → two right triangles → larger right triangle
    ("square_diag",
     "A square is cut along its diagonal, dividing it into two congruent right triangles.",
     "two congruent right triangles",
     "right triangle",
     ["square", "rhombus", "circle"],
     ["isosceles triangle", "parallelogram", "trapezoid"]),
    # 3. Right trapezoid cut perpendicular from top-left vertex → rectangle + right triangle
    ("trap_perp",
     "A right trapezoid (with two right angles on the bottom) is cut along a vertical line from the top-left vertex to the base, perpendicular to the parallel sides.",
     "a rectangle and a right triangle",
     "right triangle",
     ["square", "rhombus", "circle"],
     ["isosceles triangle", "parallelogram", "trapezoid"]),
    # 4. Parallelogram cut along altitude → can form a rectangle
    ("para_altitude",
     "A parallelogram is cut along its altitude from one top vertex perpendicular to the base.",
     "a right triangle and a quadrilateral that together rearrange",
     "rectangle",
     ["circle", "regular hexagon", "isosceles triangle"],
     ["square", "parallelogram", "right trapezoid"]),
    # 5. L-shape cut by single line → two pieces forming a rectangle (2x2 square)
    ("lshape_cut",
     "An L-shape (composed of three unit squares arranged in an L) is cut by a single straight line into two pieces.",
     "two pieces that together can be rearranged",
     "square",
     ["circle", "regular hexagon", "regular pentagon"],
     ["rectangle", "trapezoid", "parallelogram"]),
]


# Question stem variants
_STEM_TEMPLATES = [
    "As shown in the figure, {orig_desc} The pieces — {cut_desc} — are then rearranged (without overlap or gaps) to form a new shape. Which of the following shapes can be obtained ( )?",
    "Look at the figure. {orig_desc} The two pieces are rearranged into a new shape (no overlap, no gaps). What shape can be formed ( )?",
    "In the diagram, {orig_desc} After rearranging the pieces, which of the following shapes can be obtained ( )?",
    "The figure shows: {orig_desc} The pieces (now {cut_desc}) are rearranged without overlap. Which shape can result ( )?",
]


class CutRearrangeShapeQA(StandaloneVisualEnv):
    ENV_NAME = "cut_rearrange_shape"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Briefly check each option. Final answer in the format "
        "this prompt requested."
    )

    # ------------------------------------------------------------------ #
    # Per-level config
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "templates": [_TEMPLATES[1], _TEMPLATES[2]],   # square_midhoriz + square_diag
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "templates": _TEMPLATES[:4],
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "templates": _TEMPLATES[:5],
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": False,
            }
        if level == 6:
            return {
                "templates": _TEMPLATES,
                "use_5_options": True,
                "trap_rate": 0.20,
                "tight_distractors": False,
            }
        if level == 7:
            return {
                "templates": _TEMPLATES,
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        if level == 8:
            return {
                "templates": _TEMPLATES,
                "use_5_options": True,
                "trap_rate": 0.30,
                "tight_distractors": True,
            }
        # L9
        return {
            "templates": _TEMPLATES,
            "use_5_options": True,
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4231 + level * 89 + 17)

        for _ in range(20):
            res = self._try_generate(cfg, level, rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        tpl = rng.choice(cfg["templates"])
        shape_id, orig_desc, cut_desc, correct, distractors, tight_distractors = tpl

        # Pick distractor pool
        dist_pool = list(tight_distractors if cfg["tight_distractors"] else distractors)
        # Filter accidental dup with correct
        dist_pool = [d for d in dist_pool if d != correct]
        # In trap mode (5-MCQ, all-wrong), need at least n_value=4 distinct
        # wrongs. Each template only has 3 distractors per pool, so merge
        # both pools (and dedupe) when we may need 4.
        all_wrong_pool = []
        seen_w = set()
        for w in (list(tight_distractors) + list(distractors)):
            if w == correct or w in seen_w:
                continue
            seen_w.add(w)
            all_wrong_pool.append(w)

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])

        if use_trap:
            # All visible options are wrong; correct = "E. No correct answer"
            chosen = list(all_wrong_pool)
            rng.shuffle(chosen)
            chosen = chosen[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
            opt_lines = [f"{opt_letters[i]}. {options[i]}" for i in range(n_value)]
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        else:
            n_distractor = n_value - 1
            chosen = list(dist_pool)
            rng.shuffle(chosen)
            chosen = chosen[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [correct] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(correct)]
            opt_lines = [f"{opt_letters[i]}. {options[i]}" for i in range(n_value)]
            if use_5:
                opt_lines.append(f"{opt_letters[-1]}. No correct answer")

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        sidx = (self.seed or 0) % len(_STEM_TEMPLATES)
        stem = _STEM_TEMPLATES[sidx].format(orig_desc=orig_desc, cut_desc=cut_desc)
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )

        # Concise reasoning hint at L0/L1 (no answer leak)
        if level <= 1:
            question += (
                "\nHint: trace each piece in your head. The two pieces tile the new shape "
                "without overlap or gaps. Pick the option matching that tiling."
            )
        elif level <= 3:
            question += (
                "\nHint: rotate or flip each piece mentally; check which option's outline "
                "the two pieces can fully tile."
            )

        img = self._render(shape_id)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering — bigger figsize and contrasting cut color for clarity
    # ------------------------------------------------------------------ #
    def _render(self, shape_id) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        cut_color = "#d62728"
        edge_color = "#2c3e50"
        face_color = "#ecf0f1"

        if shape_id == "rectangle_diag":
            ax.add_patch(patches.Rectangle((0, 0), 4, 2,
                                           edgecolor=edge_color,
                                           facecolor=face_color, linewidth=2.4))
            ax.plot([0, 4], [0, 2], color=cut_color, linewidth=3.2)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 2.5)

        elif shape_id == "square_midhoriz":
            ax.add_patch(patches.Rectangle((0, 0), 3, 3,
                                           edgecolor=edge_color,
                                           facecolor=face_color, linewidth=2.4))
            ax.plot([0, 3], [1.5, 1.5], color=cut_color, linewidth=3.2)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)

        elif shape_id == "square_diag":
            ax.add_patch(patches.Rectangle((0, 0), 3, 3,
                                           edgecolor=edge_color,
                                           facecolor=face_color, linewidth=2.4))
            ax.plot([0, 3], [0, 3], color=cut_color, linewidth=3.2)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)

        elif shape_id == "trap_perp":
            pts = [(0, 0), (4, 0), (4, 2), (1.5, 2)]
            ax.add_patch(patches.Polygon(pts, edgecolor=edge_color,
                                         facecolor=face_color, linewidth=2.4))
            ax.plot([1.5, 1.5], [0, 2], color=cut_color, linewidth=3.2)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 2.5)

        elif shape_id == "para_altitude":
            pts = [(0, 0), (3, 0), (4, 2), (1, 2)]
            ax.add_patch(patches.Polygon(pts, edgecolor=edge_color,
                                         facecolor=face_color, linewidth=2.4))
            ax.plot([1, 1], [0, 2], color=cut_color, linewidth=3.2)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 2.5)

        elif shape_id == "lshape_cut":
            pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
            ax.add_patch(patches.Polygon(pts, edgecolor=edge_color,
                                         facecolor=face_color, linewidth=2.4))
            ax.plot([1, 2], [1, 0], color=cut_color, linewidth=3.2)
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.5, 2.5)

        ax.set_aspect("equal")
        ax.axis("off")
        # Title for clarity
        ax.set_title("Cut along the red line", fontsize=13, fontweight="bold",
                     color=edge_color, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = CutRearrangeShapeQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
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
                print(f"   pos={v_pos['accuracy']} neg={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
