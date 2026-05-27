"""
Solid Cross Section Identify QA environment.

2026-05-05 R5 P3: REWRITTEN to align with the Understanding of Solid Figures
benchmark style — INVERSE of cross_section_qa. Given a 2D cross-section
shape, identify which 3D solid + cutting plane configuration produced it.

Pure single-letter MCQ (A through D or A through E). Verifier delegates to
the base class which handles letter extraction (3 wrappers, strict-format).

Per-level design:
  L0/L1 — 4-MCQ. Trivial: render a cross-section circle / square. Options
    are 4 (solid + cut) candidates one of which can produce that shape.
  L2/L3 — 5-MCQ with "E. No correct answer" (10% trap rate). Cross-section
    can be circle / square / rectangle / triangle.
  L4-L6 — 5-MCQ. More options (ellipse, trapezoid). 10-15% trap rate.
  L7 — 5-MCQ tighter distractors. 25% trap rate.
  L8/L9 — 5-MCQ tight distractors + 40% "No correct answer" trap rate.

The image at the top is the rendered 2D cross-section shape (the result).
Each MCQ option describes a (solid, cut-direction) pair in standard textbook style
phrasing, e.g. "A. A cylinder cut perpendicular to the axis (horizontally)"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle, Ellipse, FancyBboxPatch
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Inverse rules: cross-section shape -> list of (solid, cut) configs that
# produce that shape.
_INVERSE_RULES = {
    "circle": [
        ("cylinder", "horizontal"),
        ("cone", "horizontal"),
        ("sphere", "any"),
    ],
    "square": [
        ("cube", "horizontal"),
        ("cube", "vertical"),
    ],
    "rectangle": [
        ("cuboid", "horizontal"),
        ("cuboid", "vertical"),
        ("cylinder", "vertical-through-axis"),
    ],
    "triangle": [
        ("cone", "vertical-through-apex"),
        ("triangular_prism", "perpendicular"),
        ("square_pyramid", "vertical-through-apex"),
    ],
    "ellipse": [
        ("cylinder", "oblique"),
        ("cone", "oblique"),
    ],
    "trapezoid": [
        ("triangular_prism", "oblique"),
        ("square_pyramid", "off-apex"),
    ],
}

# Verbatim textbook-style descriptions for each (solid, cut) pair.
_OPTION_DESCRIPTIONS = {
    ("cylinder", "horizontal"): "A cylinder cut horizontally (perpendicular to its axis)",
    ("cone", "horizontal"): "A cone cut horizontally (parallel to its base)",
    ("sphere", "any"): "A sphere cut by a flat plane",
    ("cube", "horizontal"): "A cube cut horizontally (parallel to its top face)",
    ("cube", "vertical"): "A cube cut vertically (parallel to one of its side faces)",
    ("cuboid", "horizontal"): "A rectangular cuboid cut horizontally (parallel to its top face)",
    ("cuboid", "vertical"): "A rectangular cuboid cut vertically (parallel to one of its side faces)",
    ("cylinder", "vertical-through-axis"): "A cylinder cut vertically along its diameter (through its axis)",
    ("cone", "vertical-through-apex"): "A cone cut vertically through its apex (along its axis)",
    ("triangular_prism", "perpendicular"): "A triangular prism cut perpendicular to its length",
    ("square_pyramid", "vertical-through-apex"): "A square pyramid cut vertically through its apex",
    ("cylinder", "oblique"): "A cylinder cut by an oblique (slanted) plane",
    ("cone", "oblique"): "A cone cut by an oblique (slanted) plane that does not pass through the apex",
    ("triangular_prism", "oblique"): "A triangular prism cut by an oblique plane along its length",
    ("square_pyramid", "off-apex"): "A square pyramid cut by a vertical plane that misses the apex",
}

# Distractor pool (configs that are plausible but produce a DIFFERENT shape).
# Used when we need additional wrong answers.
_ALL_CONFIGS = list(_OPTION_DESCRIPTIONS.keys())


def _config_produces(config, shape):
    """Return True iff this (solid, cut) config produces the given shape."""
    targets = _INVERSE_RULES.get(shape, [])
    return config in targets


def _shapes_for_level(level):
    """Pool of valid cross-section shapes per level."""
    if level <= 1:
        return ["circle", "square"]
    if level <= 3:
        return ["circle", "square", "rectangle", "triangle"]
    if level <= 6:
        return ["circle", "square", "rectangle", "triangle", "ellipse"]
    return ["circle", "square", "rectangle", "triangle", "ellipse", "trapezoid"]


class SolidCrossSectionIdentifyQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a 2D cross-section, pick the (solid, cut)."""

    ALLOW_ROTATION = False  # orientation-sensitive
    ENV_NAME = "solid_cross_section_identify"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"use_5_options": False, "trap_rate": 0.0,
                    "tight_distractors": False}
        if level <= 3:
            return {"use_5_options": True, "trap_rate": 0.10,
                    "tight_distractors": False}
        if level <= 6:
            return {"use_5_options": True, "trap_rate": 0.15,
                    "tight_distractors": False}
        if level == 7:
            return {"use_5_options": True, "trap_rate": 0.25,
                    "tight_distractors": True}
        # L8 / L9
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (env still saturated 95%).
        return {"use_5_options": True, "trap_rate": 0.50,
                "tight_distractors": True}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 281)

        for _ in range(20):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        shape_pool = _shapes_for_level(level)
        true_shape = rng.choice(shape_pool)

        valid_configs = list(_INVERSE_RULES.get(true_shape, []))
        if not valid_configs:
            return None
        valid_configs = [c for c in valid_configs if c in _OPTION_DESCRIPTIONS]
        if not valid_configs:
            return None

        # Pick a "correct" config for non-trap cases.
        true_config = rng.choice(valid_configs)

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value_options = len(opt_letters) - 1 if use_5 else len(opt_letters)
        # Note: use_5=False means 4 options, no "No correct answer" entry.
        # use_5=True means 5 options with "E. No correct answer" as last.

        # Build distractor pool — configs that DO NOT produce true_shape
        all_distractors = [c for c in _ALL_CONFIGS
                           if not _config_produces(c, true_shape)]
        rng.shuffle(all_distractors)

        # Tight distractor mode: bias toward configs whose answer is geometrically
        # similar to the correct shape (e.g. circle vs ellipse, triangle vs
        # trapezoid). Compute "produces shape" for each distractor.
        if cfg["tight_distractors"]:
            similar_pairs = {
                "circle": {"ellipse"},
                "ellipse": {"circle"},
                "square": {"rectangle"},
                "rectangle": {"square"},
                "triangle": {"trapezoid"},
                "trapezoid": {"triangle"},
            }
            target_similar = similar_pairs.get(true_shape, set())
            tight = []
            other = []
            for dc in all_distractors:
                d_shapes = set()
                for s, configs in _INVERSE_RULES.items():
                    if dc in configs:
                        d_shapes.add(s)
                if d_shapes & target_similar:
                    tight.append(dc)
                else:
                    other.append(dc)
            all_distractors = tight + other

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            # All "value" options are wrong; correct = "No correct answer"
            n_distractors = n_value_options  # fill all value slots with wrong configs
            chosen_distractors = all_distractors[:n_distractors]
            if len(chosen_distractors) < n_distractors:
                return None
            options_data = list(chosen_distractors)
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            n_distractors = n_value_options - 1
            chosen_distractors = all_distractors[:n_distractors]
            if len(chosen_distractors) < n_distractors:
                return None
            options_data = [true_config] + chosen_distractors
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(true_config)]

        # Build option strings
        opt_texts = [_OPTION_DESCRIPTIONS[c] for c in options_data]
        if use_5:
            opt_texts.append("No correct answer")
        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Stem variants — Standard textbook phrasing
        stems = [
            (
                "As shown in the diagram, the figure depicts a cross-section "
                "of a solid figure. Which of the following solid + cutting "
                "configurations could produce this cross-section?"
            ),
            (
                "The figure shows a cross-section obtained by cutting a "
                "three-dimensional solid with a flat plane. Which of the "
                "following describes how the cross-section was produced?"
            ),
            (
                "As shown in the diagram, the cross-section of a 3D solid "
                "is given. Which of the following options correctly describes "
                "the solid and the way the cutting plane was applied?"
            ),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._render_cross_section(true_shape, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # 2D cross-section rendering
    # ------------------------------------------------------------------ #
    def _render_cross_section(self, shape: str, rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        palette = list(style["palette"])
        rng.shuffle(palette)
        fill = palette[0]
        edge = "#222"

        if shape == "circle":
            r = rng.uniform(1.5, 2.0)
            c = Circle((0, 0), r, facecolor=fill, edgecolor=edge,
                       linewidth=2.0, alpha=0.6)
            ax.add_patch(c)
            ax.set_xlim(-r * 1.5, r * 1.5)
            ax.set_ylim(-r * 1.5, r * 1.5)
        elif shape == "square":
            s = rng.uniform(2.5, 3.5)
            sq = mpatches.Rectangle((-s / 2, -s / 2), s, s,
                                    facecolor=fill, edgecolor=edge,
                                    linewidth=2.0, alpha=0.6)
            ax.add_patch(sq)
            ax.set_xlim(-s, s)
            ax.set_ylim(-s, s)
        elif shape == "rectangle":
            w = rng.uniform(3.0, 4.0)
            h = rng.uniform(1.5, 2.5)
            r = mpatches.Rectangle((-w / 2, -h / 2), w, h,
                                   facecolor=fill, edgecolor=edge,
                                   linewidth=2.0, alpha=0.6)
            ax.add_patch(r)
            mx = max(w, h)
            ax.set_xlim(-mx, mx)
            ax.set_ylim(-mx, mx)
        elif shape == "triangle":
            base = rng.uniform(2.5, 3.5)
            height = rng.uniform(2.0, 3.0)
            pts = [(-base / 2, -height / 2),
                   (base / 2, -height / 2),
                   (rng.uniform(-base / 4, base / 4), height / 2)]
            t = Polygon(pts, closed=True, facecolor=fill,
                        edgecolor=edge, linewidth=2.0, alpha=0.6)
            ax.add_patch(t)
            mx = max(base, height) * 0.9
            ax.set_xlim(-mx, mx)
            ax.set_ylim(-mx, mx)
        elif shape == "ellipse":
            a = rng.uniform(2.0, 2.5)
            b = rng.uniform(1.0, 1.4)
            e = Ellipse((0, 0), 2 * a, 2 * b, facecolor=fill,
                        edgecolor=edge, linewidth=2.0, alpha=0.6)
            ax.add_patch(e)
            ax.set_xlim(-a * 1.5, a * 1.5)
            ax.set_ylim(-a * 1.5, a * 1.5)
        elif shape == "trapezoid":
            top = rng.uniform(1.5, 2.0)
            bottom = rng.uniform(2.5, 3.5)
            h = rng.uniform(1.5, 2.5)
            pts = [(-bottom / 2, -h / 2), (bottom / 2, -h / 2),
                   (top / 2, h / 2), (-top / 2, h / 2)]
            tr = Polygon(pts, closed=True, facecolor=fill,
                         edgecolor=edge, linewidth=2.0, alpha=0.6)
            ax.add_patch(tr)
            mx = bottom * 0.9
            ax.set_xlim(-mx, mx)
            ax.set_ylim(-mx, mx)
        else:
            return None

        title_pool = [
            "Cross-section",
            "Cross-section of the solid",
            "Cut cross-section",
            "Section view",
        ]
        ax.set_title(rng.choice(title_pool),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", fontfamily=style["font_family"])
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """All answers are single-letter MCQ. Delegate to base class."""
        return super()._check_answer(predicted, ground_truth)


if __name__ == "__main__":
    env = SolidCrossSectionIdentifyQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
