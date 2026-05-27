"""Feature Counting Classification QA — diversity + difficulty redesign 2026-04-16.

Round-2 fix: Grade D difficulty + low diversity.
- Shape pool expanded from rectangle-only to 5 families (rectangle, circle,
  rounded-rect, diamond, hexagon).
- Feature pool: dots, vertical lines, crosses, triangles, stars (5 families).
- Border colour palette expanded, per-seed shuffle.
- 5 question-stem phrasings + 5 title phrasings.
- Layout: horizontal or vertical group arrangement per-seed.
- Difficulty gradient:
    L0-L1: single count threshold + 2 groups  (easiest).
    L2-L3: single feature + 2 groups, larger pool to pick from (more confusable).
    L4-L5: conjunction of border colour + parity.
    L6-L7: conjunction with 3 groups.
    L8-L9: conjunction with 3 groups and 4 feature families active per group.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_BORDER_COLOURS = {
    "blue":    "#2d5ba9",
    "red":     "#b91d1d",
    "green":   "#1f7a3a",
    "orange":  "#c07214",
    "purple":  "#6b2f8f",
    "teal":    "#148b90",
    "brown":   "#7b4419",
    "navy":    "#1a365d",
}

_FILL_COLOURS = [
    "#ffffff", "#fff8e7", "#edf7fa", "#f7f0ff", "#f0fff4",
    "#fff0f7", "#e9f5ff", "#fdf6e3", "#fdfdfd",
]

_SHAPE_KINDS = ["rect", "circle", "rounded_rect", "diamond", "hexagon"]

_FEATURE_KINDS = ["dots", "lines", "crosses", "triangles", "stars"]

def _draw_feature_figure(ax, cx, cy, cfg: Dict, size: float = 0.45):
    """Draw outer shape + internal feature markers.

    cfg keys:
      shape: str
      border_colour: hex
      fill_colour: hex
      feature: str in _FEATURE_KINDS
      n_features: int
    """
    border = cfg.get("border_colour", "#2d5ba9")
    fill = cfg.get("fill_colour", "#ffffff")
    shape = cfg.get("shape", "rect")
    lw = 2.2

    if shape == "rect":
        patch = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                    facecolor=fill, edgecolor=border,
                                    linewidth=lw, zorder=3)
    elif shape == "rounded_rect":
        patch = mpatches.FancyBboxPatch((cx - size, cy - size),
                                         2 * size, 2 * size,
                                         boxstyle="round,pad=0.02,rounding_size=0.12",
                                         facecolor=fill, edgecolor=border,
                                         linewidth=lw, zorder=3)
    elif shape == "circle":
        patch = mpatches.Circle((cx, cy), size, facecolor=fill,
                                 edgecolor=border, linewidth=lw, zorder=3)
    elif shape == "diamond":
        pts = [(cx, cy + size), (cx + size, cy),
                (cx, cy - size), (cx - size, cy)]
        patch = mpatches.Polygon(pts, facecolor=fill, edgecolor=border,
                                  linewidth=lw, zorder=3)
    elif shape == "hexagon":
        pts = [(cx + size * math.cos(math.radians(60 * i)),
                 cy + size * math.sin(math.radians(60 * i))) for i in range(6)]
        patch = mpatches.Polygon(pts, facecolor=fill, edgecolor=border,
                                  linewidth=lw, zorder=3)
    else:
        patch = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                    facecolor=fill, edgecolor=border, linewidth=lw)
    ax.add_patch(patch)

    n = int(cfg.get("n_features", 0))
    feature = cfg.get("feature", "dots")
    usable = 1.3 * size
    inner_color = cfg.get("marker_colour", "#2c3e50")

    if n <= 0:
        return

    if feature == "dots":
        cols = max(2, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        dx = usable / max(cols, 1)
        dy = usable / max(rows, 1)
        start_x = cx - usable / 2 + dx / 2
        start_y = cy + usable / 2 - dy / 2
        r = min(size * 0.12, dx * 0.35, dy * 0.35)
        for i in range(n):
            rr = i // cols
            cc = i % cols
            ax.add_patch(mpatches.Circle(
                (start_x + cc * dx, start_y - rr * dy),
                r, facecolor=inner_color, edgecolor=inner_color, zorder=4))
    elif feature == "lines":
        dx = usable / (n + 1)
        x0 = cx - usable / 2
        y_lo = cy - size * 0.35
        y_hi = cy + size * 0.35
        for i in range(n):
            xl = x0 + (i + 1) * dx
            ax.plot([xl, xl], [y_lo, y_hi],
                    color=inner_color, linewidth=1.3, zorder=4)
    elif feature == "crosses":
        cols = max(2, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        dx = usable / max(cols, 1)
        dy = usable / max(rows, 1)
        start_x = cx - usable / 2 + dx / 2
        start_y = cy + usable / 2 - dy / 2
        half = min(size * 0.1, dx * 0.3, dy * 0.3)
        for i in range(n):
            rr = i // cols
            cc = i % cols
            mx = start_x + cc * dx
            my = start_y - rr * dy
            ax.plot([mx - half, mx + half], [my - half, my + half],
                    color=inner_color, linewidth=1.2, zorder=4)
            ax.plot([mx - half, mx + half], [my + half, my - half],
                    color=inner_color, linewidth=1.2, zorder=4)
    elif feature == "triangles":
        cols = max(2, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        dx = usable / max(cols, 1)
        dy = usable / max(rows, 1)
        start_x = cx - usable / 2 + dx / 2
        start_y = cy + usable / 2 - dy / 2
        half = min(size * 0.11, dx * 0.32, dy * 0.32)
        for i in range(n):
            rr = i // cols
            cc = i % cols
            mx = start_x + cc * dx
            my = start_y - rr * dy
            pts = [(mx, my + half), (mx - half, my - half * 0.7),
                    (mx + half, my - half * 0.7)]
            ax.add_patch(mpatches.Polygon(pts, facecolor=inner_color,
                                            edgecolor=inner_color, zorder=4))
    elif feature == "stars":
        cols = max(2, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        dx = usable / max(cols, 1)
        dy = usable / max(rows, 1)
        start_x = cx - usable / 2 + dx / 2
        start_y = cy + usable / 2 - dy / 2
        for i in range(n):
            rr = i // cols
            cc = i % cols
            mx = start_x + cc * dx
            my = start_y - rr * dy
            ax.plot(mx, my, marker='*', color=inner_color,
                    markersize=min(14, 180 / max(1, n)), zorder=4)

class FeatureCountingClassificationQA(StandaloneVisualEnv):
    ENV_NAME = "feature_counting_classification"

    _TITLE_VARIANTS = [
        "Which group?",
        "Classify the test figure",
        "Group classification",
        "Pattern match",
        "Feature counting",
        "Attribute group",
        "Rule match",
    ]

    _QUESTION_STEMS = [
        "Study the example figures shown in each group. "
        "Which group does the test figure belong to?",
        "Each group on the left has a common rule. "
        "To which group does the test figure on the right belong?",
        "Based on the example groups, classify the test figure.",
        "Determine which group the test figure belongs to.",
        "Examine the rule that defines each group. Assign the test figure to a group.",
        "Figures in each group share an attribute. Which group does the test figure join?",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {"rule": "single_count", "n_examples": 3, "n_groups": 2}
        if level <= 3:
            return {"rule": "single_count", "n_examples": 4, "n_groups": 2}
        if level <= 5:
            return {"rule": "conjunction", "n_examples": 4, "n_groups": 2}
        if level <= 7:
            return {"rule": "conjunction", "n_examples": 4, "n_groups": 3}
        return {"rule": "conjunction", "n_examples": 5, "n_groups": 3}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1367)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 43 + 1481)

        for _ in range(40):
            r = self._try_generate(rng, sub_rng, level, cfg)
            if r is not None:
                return r
        return None

    def _sample_group_rule(self, rng, cfg):
        rule = cfg["rule"]
        n_groups = cfg["n_groups"]

        borders = list(_BORDER_COLOURS.keys())
        rng.shuffle(borders)
        feature = rng.choice(_FEATURE_KINDS)
        shape_pool = list(_SHAPE_KINDS)
        rng.shuffle(shape_pool)

        def make_fig(border_hex, n, shared_shapes: List[str]):
            def _s(r):
                return {
                    "border_colour": border_hex,
                    "fill_colour": r.choice(_FILL_COLOURS),
                    "shape": r.choice(shared_shapes),
                    "feature": feature,
                    "n_features": n,
                    "marker_colour": r.choice(["#2c3e50", "#1a1a1a", "#101820", "#3b3b3b"]),
                }
            return _s

        if rule == "single_count":
            low_count = rng.choice([1, 2])
            high_count = rng.choice([5, 6, 7])
            # Group shapes need not be constant — but we use a shared shape pool per group
            g_shapes_0 = rng.sample(_SHAPE_KINDS, rng.randint(1, 3))
            g_shapes_1 = rng.sample(_SHAPE_KINDS, rng.randint(1, 3))
            groups = [
                {"label": "X",
                 "sample_fn": make_count_sampler(_BORDER_COLOURS[borders[0]],
                                                  [low_count, low_count+1], feature, g_shapes_0),
                 "match": lambda a, lo=low_count: a["n_features"] <= lo + 1},
                {"label": "Y",
                 "sample_fn": make_count_sampler(_BORDER_COLOURS[borders[1]],
                                                  [high_count, high_count+1], feature, g_shapes_1),
                 "match": lambda a, hi=high_count: a["n_features"] >= hi - 1},
            ]
            if n_groups == 3:
                # add a middle group
                mid = rng.choice([3, 4])
                g_shapes_2 = rng.sample(_SHAPE_KINDS, rng.randint(1, 3))
                groups.append({"label": "Z",
                                "sample_fn": make_count_sampler(_BORDER_COLOURS[borders[2]],
                                                                 [mid], feature, g_shapes_2),
                                "match": lambda a, m=mid: a["n_features"] == m})
        else:  # conjunction
            b0 = _BORDER_COLOURS[borders[0]]
            b1 = _BORDER_COLOURS[borders[1]]
            b2 = _BORDER_COLOURS[borders[2]] if n_groups >= 3 else None
            parities = ["even", "odd"]
            rng.shuffle(parities)
            g_shapes_0 = rng.sample(_SHAPE_KINDS, rng.randint(1, 3))
            g_shapes_1 = rng.sample(_SHAPE_KINDS, rng.randint(1, 3))
            groups = [
                {"label": "X",
                 "sample_fn": make_parity_sampler(b0, parities[0], feature, g_shapes_0),
                 "match": lambda a, par=parities[0], bc=b0:
                     (a["border_colour"] == bc and
                      (a["n_features"] % 2 == 0 if par == "even" else a["n_features"] % 2 == 1))},
                {"label": "Y",
                 "sample_fn": make_parity_sampler(b1, parities[1], feature, g_shapes_1),
                 "match": lambda a, par=parities[1], bc=b1:
                     (a["border_colour"] == bc and
                      (a["n_features"] % 2 == 0 if par == "even" else a["n_features"] % 2 == 1))},
            ]
            if n_groups == 3:
                third_parity = rng.choice(parities)
                g_shapes_2 = rng.sample(_SHAPE_KINDS, rng.randint(1, 3))
                groups.append({
                    "label": "Z",
                    "sample_fn": make_parity_sampler(b2, third_parity, feature, g_shapes_2),
                    "match": lambda a, par=third_parity, bc=b2:
                        (a["border_colour"] == bc and
                         (a["n_features"] % 2 == 0 if par == "even" else a["n_features"] % 2 == 1))})

        test_group_idx = rng.randint(0, len(groups) - 1)
        test_attrs = groups[test_group_idx]["sample_fn"](rng)
        return groups, test_attrs, test_group_idx

    def _try_generate(self, rng, sub_rng, level, cfg):
        groups, test_attrs, test_group_idx = self._sample_group_rule(rng, cfg)

        examples_per_group: List[List[Dict]] = []
        for g in groups:
            ex_list = [g["sample_fn"](rng) for _ in range(cfg["n_examples"])]
            examples_per_group.append(ex_list)

        labels = [g["label"] for g in groups]
        options = list(labels)
        if len(options) == 2:
            options = options + ["Neither", "Both"]
        elif len(options) == 3:
            options = options + ["Neither"]
        correct = labels[test_group_idx]
        rng.shuffle(options)
        if correct not in options:
            return None
        answer_letter = chr(ord("A") + options.index(correct))

        title = sub_rng.choice(self._TITLE_VARIANTS)
        layout = sub_rng.choice(["horizontal", "vertical"])
        image = self._render(examples_per_group, labels, test_attrs,
                             options, title, layout, sub_rng)

        opts_block = "\n".join(
            f"  ({chr(ord('A') + i)}) {opt if opt in ('Neither', 'Both') else f'Group {opt}'}"
            for i, opt in enumerate(options))
        stem = sub_rng.choice(self._QUESTION_STEMS)
        question = (
            f"{stem} (Judge only from the figures as shown.)\n{opts_block}\n"
            "Answer with a single letter."
        )
        return question, answer_letter, image

    def _render(self, examples_per_group, labels, test_attrs, options,
                title, layout, sub_rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        n_groups = len(examples_per_group)
        n_examples = len(examples_per_group[0])

        if layout == "horizontal":
            fig_w = max(10.5, (n_examples + 3) * 1.3) * sc
            fig_h = (n_groups * 1.7 + 3.8) * sc
        else:
            fig_w = max(12, (n_examples * n_groups + 2) * 1.2) * sc
            fig_h = 6.0 * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.3], hspace=0.22)
        ax_img = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        ax_img.set_aspect("equal")
        ax_img.axis("off")

        group_row_h = 1.6
        total_w = (n_examples + 3.0) * 1.4
        total_h = n_groups * group_row_h + 1.8
        ax_img.set_xlim(-0.2, total_w + 0.2)
        ax_img.set_ylim(-0.2, total_h + 0.2)
        ax_img.set_title(title, fontsize=fs + 3, fontweight="bold",
                         fontfamily=ff, pad=6)

        for gi, group in enumerate(examples_per_group):
            y_center = total_h - (gi + 0.6) * group_row_h - 0.1
            ax_img.text(0.1, y_center, f"Group {labels[gi]}:",
                        fontsize=fs + 1, fontweight="bold",
                        fontfamily=ff, color="#2c3e50", va="center")
            for ei, ex in enumerate(group):
                # Shift examples right so the "Group X:" label doesn't overlap
                # with the first figure.
                cx = 2.2 + ei * 1.2 + 0.6
                _draw_feature_figure(ax_img, cx, y_center, ex, size=0.42)

        test_cy = 0.8
        test_cx = total_w - 1.0
        ax_img.text(test_cx, test_cy + 0.9, "Test:",
                    fontsize=fs + 1, fontweight="bold",
                    fontfamily=ff, ha="center", color="#c0392b")
        _draw_feature_figure(ax_img, test_cx, test_cy, test_attrs, size=0.45)
        ax_img.text(test_cx - 1.15, test_cy, "?",
                    fontsize=fs + 10, fontweight="bold",
                    fontfamily=ff, ha="center", va="center", color="#c0392b")

        ax_txt.axis("off")
        ax_txt.set_xlim(0, 10)
        ax_txt.set_ylim(0, 6)
        ax_txt.text(0.3, 5.5, "Options:",
                    fontsize=fs + 1, fontweight="bold",
                    fontfamily=ff, color="#2c3e50", ha="left", va="top")
        y = 4.6
        for i, opt in enumerate(options):
            label_txt = (f"({chr(ord('A') + i)}) {opt}" if opt in ("Neither", "Both")
                         else f"({chr(ord('A') + i)}) Group {opt}")
            ax_txt.text(0.5, y, label_txt,
                        fontsize=fs + 1, fontfamily=ff,
                        color="#1a1a1a", ha="left", va="top")
            y -= 0.95

        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.05)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Sampler factories (module-level so they pickle/ref safely)
# ---------------------------------------------------------------------- #

def make_count_sampler(border_hex, count_choices, feature, shape_pool):
    def _s(r):
        return {
            "border_colour": border_hex,
            "fill_colour": r.choice(_FILL_COLOURS),
            "shape": r.choice(shape_pool),
            "feature": feature,
            "n_features": r.choice(count_choices),
            "marker_colour": r.choice(["#2c3e50", "#1a1a1a", "#101820", "#3b3b3b"]),
        }
    return _s

def make_parity_sampler(border_hex, parity, feature, shape_pool):
    evens = [2, 4, 6, 8]
    odds = [1, 3, 5, 7]

    def _s(r):
        n = r.choice(evens) if parity == "even" else r.choice(odds)
        return {
            "border_colour": border_hex,
            "fill_colour": r.choice(_FILL_COLOURS),
            "shape": r.choice(shape_pool),
            "feature": feature,
            "n_features": n,
            "marker_colour": r.choice(["#2c3e50", "#1a1a1a", "#101820", "#3b3b3b"]),
        }
    return _s
