"""
Visual Rule Exception QA environment (v3 planning env 58).

Targets VisuLogic Attribute Reasoning. A row of 6-10 figures
is drawn that all follow a shared rule EXCEPT one. The model must identify
the exception figure, labeled A-J. MCQ with 4 options: the correct exception
letter and 3 distractor letters.

Rule types by level:
  L0-2: shape type (e.g., all circles except a triangle)
  L3-5: color attribute (e.g., all blue except a red)
  L6-7: count-based (e.g., all have 3 internal dots except one with 4)
  L8-9: multi-attribute relationship (n_dots == n_sides of outer shape
        except for one figure off by 1)
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_COLOR_MAP = {
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "green":  "#27ae60",
    "orange": "#e67e22",
    "purple": "#8e44ad",
    "yellow": "#f1c40f",
}

_SHAPE_OPTIONS = ["circle", "square", "triangle", "diamond",
                   "pentagon", "hexagon"]

def _draw_outer_shape(ax, cx, cy, shape: str, color_hex: str,
                       size: float = 0.45,
                       filled: bool = True):
    fc = color_hex if filled else "none"
    ec = "#2c3e50" if filled else color_hex
    lw = 1.4 if filled else 2.0
    if shape == "circle":
        ax.add_patch(mpatches.Circle((cx, cy), size,
                                     facecolor=fc, edgecolor=ec,
                                     linewidth=lw, zorder=3))
    elif shape == "square":
        ax.add_patch(mpatches.Rectangle((cx - size, cy - size),
                                         2 * size, 2 * size,
                                         facecolor=fc, edgecolor=ec,
                                         linewidth=lw, zorder=3))
    else:
        n_sides = {"triangle": 3, "diamond": 4, "pentagon": 5,
                   "hexagon": 6}[shape]
        orient = math.pi / 2
        ax.add_patch(mpatches.RegularPolygon(
            (cx, cy), n_sides, radius=size, orientation=orient,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))

def _draw_dots_inside(ax, cx, cy, n: int, size: float = 0.32):
    if n <= 0:
        return
    if n == 1:
        positions = [(cx, cy)]
    elif n == 2:
        positions = [(cx - 0.10, cy), (cx + 0.10, cy)]
    else:
        positions = []
        r_dots = 0.14
        for i in range(n):
            a = math.pi / 2 + 2 * math.pi * i / n
            positions.append((cx + r_dots * math.cos(a),
                               cy + r_dots * math.sin(a)))
    dot_r = min(0.04, 0.18 / max(n, 1))
    for x, y in positions:
        ax.add_patch(mpatches.Circle((x, y), dot_r,
                                     facecolor="#2c3e50",
                                     edgecolor="#2c3e50", zorder=4))

def _shape_sides(shape: str) -> int:
    mapping = {"triangle": 3, "square": 4, "diamond": 4, "pentagon": 5,
               "hexagon": 6, "circle": 0}
    return mapping[shape]

class VisualRuleExceptionQA(StandaloneVisualEnv):
    """Find the exception in a row of figures (A3 + A4)."""

    ENV_NAME = "visual_rule_exception"

    _TITLE_VARIANTS = [
        "Find the exception",
        "Which figure breaks the rule?",
        "Visual rule check",
        "Odd one in line",
        "Spot the outlier",
    ]

    _QUESTION_STEMS = [
        "All figures below follow the same visual rule except one. "
        "Which figure is the exception?",
        "Every figure obeys a shared rule except one. "
        "Which is the outlier?",
        "Find the figure that does NOT match the common rule.",
        "One figure breaks the pattern the rest share. Which one?",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Difficulty axes:
          1) rule_type: shape is easiest (many common shapes, odd stands out).
             count is harder than color because counting dots is work.
             multi (n_dots == n_sides) is hardest (compose two rules).
             We gradually interleave rules so L3 is NOT pure color (previously
             0.30 — worst level) but mixes shape|color to smooth transition.
          2) n_figures: 6 + level // 2 (6..10)
        """
        level = max(0, min(9, int(level)))
        # Pure shape at L0-1 (easy). Gradually add color distractors.
        if level <= 1:
            rule_types = ["shape"]
        elif level <= 3:
            rule_types = ["shape", "color"]     # was pure color at L3
        elif level <= 5:
            rule_types = ["color", "count"]
        elif level <= 7:
            rule_types = ["count"]
        else:
            rule_types = ["multi"]
        n_figures = min(10, 6 + level // 2)
        return {
            # Pick single rule_type per sample from pool (drive via rng in gen).
            "rule_type": rule_types[0] if len(rule_types) == 1 else rule_types,
            "rule_type_pool": rule_types,
            "n_figures": n_figures,
        }

    # ------------------------------------------------------------------ #
    # Problem generation
    # ------------------------------------------------------------------ #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1373)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 43 + 1373)
        self._primary_complexity_feature = cfg["n_figures"] + level * 2

        for _ in range(40):
            r = self._try_generate(rng, sub_rng, level, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, sub_rng, level, cfg):
        rule_type = cfg["rule_type"]
        if isinstance(rule_type, list):
            rule_type = sub_rng.choice(rule_type)
        n_fig = cfg["n_figures"]

        # Each figure stored as dict with shape/color/n_dots
        figs: List[Dict] = []

        # Pick a "common rule" baseline
        shapes_pool = list(_SHAPE_OPTIONS)
        colors_pool = list(_COLOR_MAP.keys())
        common_shape = rng.choice(shapes_pool)
        common_color = rng.choice(colors_pool)
        common_count = rng.choice([2, 3, 4])

        exception_idx = rng.randint(0, n_fig - 1)

        for i in range(n_fig):
            if rule_type == "shape":
                shape = common_shape if i != exception_idx else rng.choice(
                    [s for s in shapes_pool if s != common_shape])
                color = rng.choice(colors_pool)
                n_dots = 0
            elif rule_type == "color":
                shape = rng.choice(shapes_pool)
                color = common_color if i != exception_idx else rng.choice(
                    [c for c in colors_pool if c != common_color])
                n_dots = 0
            elif rule_type == "count":
                shape = rng.choice(shapes_pool)
                color = rng.choice(colors_pool)
                if i != exception_idx:
                    n_dots = common_count
                else:
                    alt = [c for c in [1, 2, 3, 4, 5] if c != common_count]
                    n_dots = rng.choice(alt)
            else:
                # Multi rule: n_dots == n_sides of outer polygon (excl. circle).
                # Iter 3 (2026-04-17): L9 was 0.20 because off-by-one is
                # hard to spot at small dot radii. Allow exception to
                # differ by at least 2 (or by direction in {-2, +2}) so
                # the visual gap is unambiguous.
                shape = rng.choice(
                    ["triangle", "square", "pentagon", "hexagon"])
                color = rng.choice(colors_pool)
                sides = _shape_sides(shape)
                if i != exception_idx:
                    n_dots = sides
                else:
                    # Offset by at least 2 in either direction, clamped
                    # to [1, 6]. Ensures inequality.
                    delta = rng.choice([-2, 2, -3, 3])
                    n_dots = max(1, min(6, sides + delta))
                    # Guarantee inequality
                    if n_dots == sides:
                        n_dots = max(1, n_dots - 2) if n_dots > 2 else n_dots + 2

            figs.append({"shape": shape, "color": color, "n_dots": n_dots})

        # Sanity: verify the exception truly stands out.
        if not self._verify_exception(figs, rule_type, exception_idx):
            return None

        # Labels A, B, C, ...
        labels = [chr(ord("A") + i) for i in range(n_fig)]
        correct_label = labels[exception_idx]

        # Build MCQ: 4 options. Correct + 3 distractor letters.
        distractors = [l for l in labels if l != correct_label]
        rng.shuffle(distractors)
        options = [correct_label] + distractors[:3]
        rng.shuffle(options)
        answer_letter = chr(ord("A") + options.index(correct_label))

        title = sub_rng.choice(self._TITLE_VARIANTS)
        image = self._render(figs, labels, options, title=title,
                             sub_rng=sub_rng)

        opts_block = "\n".join(
            f"  ({chr(ord('A') + i)}) Figure {opt}"
            for i, opt in enumerate(options))
        stem = sub_rng.choice(self._QUESTION_STEMS)
        question = (
            f"{stem}\n{opts_block}\n"
            "Answer with a single letter."
        )
        return question, answer_letter, image

    @staticmethod
    def _verify_exception(figs: List[Dict], rule_type: str,
                          exc_idx: int) -> bool:
        """Ensure all non-exception figures share the rule and exception doesn't."""
        if rule_type == "shape":
            shapes = [f["shape"] for f in figs]
            common = shapes[exc_idx - 1 if exc_idx > 0 else exc_idx + 1]
            ok_others = all(shapes[i] == common for i in range(len(figs))
                             if i != exc_idx)
            return ok_others and shapes[exc_idx] != common
        if rule_type == "color":
            colors = [f["color"] for f in figs]
            common = colors[exc_idx - 1 if exc_idx > 0 else exc_idx + 1]
            ok_others = all(colors[i] == common for i in range(len(figs))
                             if i != exc_idx)
            return ok_others and colors[exc_idx] != common
        if rule_type == "count":
            counts = [f["n_dots"] for f in figs]
            common = counts[exc_idx - 1 if exc_idx > 0 else exc_idx + 1]
            ok_others = all(counts[i] == common for i in range(len(figs))
                             if i != exc_idx)
            return ok_others and counts[exc_idx] != common
        if rule_type == "multi":
            ok_others = all(
                f["n_dots"] == _shape_sides(f["shape"])
                for i, f in enumerate(figs) if i != exc_idx)
            exc = figs[exc_idx]
            return ok_others and exc["n_dots"] != _shape_sides(exc["shape"])
        return False

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, figs: List[Dict], labels: List[str],
                options: List[str],
                title: str = "Find the exception",
                sub_rng: Optional[random.Random] = None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        n = len(figs)
        fig_w = max(9.5, 1.3 * n + 2.5) * sc
        fig_h = 5.5 * sc

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.3], hspace=0.18)
        ax_img = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        ax_img.set_aspect("equal")
        ax_img.axis("off")
        spacing = 1.3
        ax_img.set_xlim(-0.2, n * spacing + 0.2)
        ax_img.set_ylim(-0.3, 1.6)
        ax_img.set_title(title, fontsize=fs + 2, fontweight="bold",
                         fontfamily=ff, pad=6)

        for i, f in enumerate(figs):
            cx = (i + 0.5) * spacing
            cy = 0.7
            color_hex = _COLOR_MAP[f["color"]]
            # For count / multi rules: draw an outlined shape so the dots
            # inside are visible. Otherwise draw filled.
            rule_uses_dots = (f["n_dots"] > 0)
            _draw_outer_shape(ax_img, cx, cy, f["shape"], color_hex,
                              size=0.38, filled=not rule_uses_dots)
            if f["n_dots"] > 0:
                _draw_dots_inside(ax_img, cx, cy, f["n_dots"])

            ax_img.text(cx, -0.15, labels[i],
                        fontsize=fs + 2, fontweight="bold",
                        ha="center", va="top", color="#2c3e50",
                        fontfamily=ff)

        # Options panel
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 10)
        ax_txt.set_ylim(0, 6)
        ax_txt.text(0.3, 5.5, "Which figure is the exception?",
                    fontsize=fs + 1, fontweight="bold",
                    fontfamily=ff, color="#2c3e50",
                    ha="left", va="top")
        y = 4.6
        for i, opt in enumerate(options):
            ax_txt.text(0.5, y, f"({chr(ord('A') + i)}) Figure {opt}",
                        fontsize=fs + 1, fontfamily=ff,
                        color="#1a1a1a", ha="left", va="top")
            y -= 0.9

        fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.05)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = VisualRuleExceptionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 7 + level, parameter={"level": level})
            print(f"L{level} seed={seed} ok={ok} A={env._answer}")
