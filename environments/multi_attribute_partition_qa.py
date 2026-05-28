"""
Multi-Attribute Partition QA environment (v3 planning env 55).

Targets a logic benchmark Attribute Reasoning and a puzzle benchmark
rectangle_height_number. A set of 8-17 shapes is displayed,
partitioned by a dashed boundary into two groups. Each shape varies across
2-3 attributes (color, shape, size). The model must decide which set of
attributes defines the partition.

MCQ with 4 options: one correct attribute description and three plausible
distractors. Distractors describe alternate attributes that either do not
produce a clean split or describe a conjunction that mis-groups a couple
of items.
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

_SHAPE_NAMES = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon"]
_SIZE_NAMES = ["small", "large"]
_SIZE_MAP = {"small": 0.22, "large": 0.36}

def _draw_shape(ax, cx, cy, shape: str, color_hex: str, size: float):
    """Render a single shape at (cx, cy) with given color and radius."""
    if shape == "circle":
        ax.add_patch(mpatches.Circle((cx, cy), size,
                                     facecolor=color_hex,
                                     edgecolor="#2c3e50", linewidth=1.2,
                                     zorder=3))
    elif shape == "square":
        ax.add_patch(mpatches.Rectangle((cx - size, cy - size),
                                         2 * size, 2 * size,
                                         facecolor=color_hex,
                                         edgecolor="#2c3e50", linewidth=1.2,
                                         zorder=3))
    else:
        n_sides = {"triangle": 3, "diamond": 4, "pentagon": 5,
                   "hexagon": 6}[shape]
        orient = math.pi / 2 if shape != "diamond" else math.pi / 2
        p = mpatches.RegularPolygon((cx, cy), n_sides, radius=size,
                                    orientation=orient,
                                    facecolor=color_hex,
                                    edgecolor="#2c3e50", linewidth=1.2,
                                    zorder=3)
        ax.add_patch(p)

class MultiAttributePartitionQA(StandaloneVisualEnv):
    """Partition shapes by one or two attributes (A4)."""

    ENV_NAME = "multi_attribute_partition"

    _TITLE_VARIANTS = [
        "Partitioned shapes",
        "Which rule splits the shapes?",
        "Shape partition",
        "Find the partition rule",
        "Classify by attribute",
    ]

    _QUESTION_STEMS = [
        "The shapes on either side of the dashed line form two groups. "
        "Which rule partitions them?",
        "Look at the shapes split by the dashed boundary. "
        "Which attribute(s) define the two groups?",
        "A dashed line divides the shapes into two groups. "
        "Which statement best describes how they are partitioned?",
        "The dashed boundary separates the shapes into two groups. "
        "Which of the following partition rules matches the grouping?",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Difficulty axes:
          1) n_shapes: 8 at L0 -> 17 at L9
          2) n_partition_attributes: 1 at L<=3, 2 at L>=4
             n_attribute_values: 2 + level // 3 (applies to color/shape pool)
        """
        level = max(0, min(9, int(level)))
        n_shapes = 8 + level
        n_partition_attrs = 1 if level <= 3 else 2
        n_values = 2 + level // 3  # 2, 2, 2, 3, 3, 3, 4, 4, 4, 5
        # Which single attributes are eligible at each level (varies by difficulty).
        if level <= 1:
            eligible = ["color"]
        elif level <= 3:
            eligible = ["color", "shape"]
        elif level <= 6:
            eligible = ["color", "shape", "size"]
        else:
            eligible = ["color", "shape", "size"]
        return {
            "n_shapes": n_shapes,
            "n_partition_attrs": n_partition_attrs,
            "n_values": n_values,
            "eligible": eligible,
        }

    # ------------------------------------------------------------------ #
    # Problem generation
    # ------------------------------------------------------------------ #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1327)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 41 + 1327)
        self._primary_complexity_feature = cfg["n_shapes"] + level * 2

        for _ in range(40):
            r = self._try_generate(rng, sub_rng, level, cfg)
            if r is not None:
                return r
        return None

    def _sample_attr_pool(self, rng, cfg):
        """Pick the pools of possible values for color / shape / size."""
        n_vals = cfg["n_values"]
        color_pool = rng.sample(list(_COLOR_MAP.keys()),
                                min(n_vals, len(_COLOR_MAP)))
        shape_pool = rng.sample(_SHAPE_NAMES,
                                min(n_vals, len(_SHAPE_NAMES)))
        size_pool = list(_SIZE_NAMES)  # always 2
        return color_pool, shape_pool, size_pool

    def _try_generate(self, rng, sub_rng, level, cfg):
        color_pool, shape_pool, size_pool = self._sample_attr_pool(rng, cfg)

        n_shapes = cfg["n_shapes"]
        n_half = n_shapes // 2
        # For single-attr: pick the partitioning attribute, then split values
        # into two halves. For two-attr: the intersection of two binary cuts.

        part_attrs = []
        if cfg["n_partition_attrs"] == 1:
            part_attrs = [rng.choice(cfg["eligible"])]
        else:
            part_attrs = rng.sample(cfg["eligible"], 2)

        # Define group predicates: a shape is in group A if (value of attr1
        # is in bucket_A1) AND (value of attr2 is in bucket_A2). We split
        # each attribute's value pool into two halves.
        def split_values(values):
            values = list(values)
            rng.shuffle(values)
            half = max(1, len(values) // 2)
            return set(values[:half]), set(values[half:])

        buckets = {}
        for a in part_attrs:
            if a == "color":
                buckets[a] = split_values(color_pool)
            elif a == "shape":
                buckets[a] = split_values(shape_pool)
            elif a == "size":
                buckets[a] = split_values(size_pool)

        # Generate shapes. Assign each to group A or B roughly balanced.
        shapes = []
        # We want each group to be non-empty and to roughly split the set.
        # Rejection-sample a random assignment with the right group sizes.
        labels = [0] * n_half + [1] * (n_shapes - n_half)
        rng.shuffle(labels)

        for idx in range(n_shapes):
            g = labels[idx]
            # For each partition attr, pick a value from the proper bucket.
            picked = {}
            for a in part_attrs:
                bA, bB = buckets[a]
                target = bA if g == 0 else bB
                picked[a] = rng.choice(list(target))
            # Non-partitioning attrs are random
            for a in ["color", "shape", "size"]:
                if a in picked:
                    continue
                if a == "color":
                    picked[a] = rng.choice(color_pool)
                elif a == "shape":
                    picked[a] = rng.choice(shape_pool)
                elif a == "size":
                    picked[a] = rng.choice(size_pool)
            shapes.append({"group": g, **picked})

        # Verify: among group 0 and group 1, the partition attribute(s) actually
        # separate cleanly.
        for a in part_attrs:
            bA, bB = buckets[a]
            g0_vals = {s[a] for s in shapes if s["group"] == 0}
            g1_vals = {s[a] for s in shapes if s["group"] == 1}
            if not g0_vals.issubset(bA) or not g1_vals.issubset(bB):
                return None

        # --- Build the 4 MCQ options ---
        correct_text = self._describe_partition(part_attrs)
        distractors = self._build_distractors(rng, cfg, part_attrs,
                                              shapes, correct_text)
        if len(distractors) < 3:
            return None
        options = [correct_text] + distractors[:3]
        rng.shuffle(options)
        answer_idx = options.index(correct_text)
        answer_letter = chr(ord("A") + answer_idx)

        # Render
        title = sub_rng.choice(self._TITLE_VARIANTS)
        image = self._render(shapes, options, title=title,
                             sub_rng=sub_rng)

        opts_block = "\n".join(
            f"  ({chr(ord('A') + i)}) {opt}"
            for i, opt in enumerate(options))
        q_stem = sub_rng.choice(self._QUESTION_STEMS)
        question = (
            f"{q_stem}\n{opts_block}\n"
            "Answer with a single letter."
        )
        return question, answer_letter, image

    # ------------------------------------------------------------------ #
    # Partition description helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _describe_partition(attrs: List[str]) -> str:
        if len(attrs) == 1:
            return f"{attrs[0]} only"
        s_attrs = sorted(attrs)
        return f"{s_attrs[0]} AND {s_attrs[1]}"

    def _build_distractors(self, rng, cfg, used_attrs, shapes, correct_text):
        """Build 3 plausible but incorrect partition descriptions."""
        all_attrs = ["color", "shape", "size"]
        used_set = set(used_attrs)
        pool = []
        # Single-attribute distractors (attribute not actually used)
        for a in all_attrs:
            if a not in used_set:
                pool.append(f"{a} only")
        # Paired-attribute distractors
        for i in range(len(all_attrs)):
            for j in range(i + 1, len(all_attrs)):
                a, b = all_attrs[i], all_attrs[j]
                pair_txt = f"{a} AND {b}"
                if pair_txt == correct_text:
                    continue
                pool.append(pair_txt)
        # If correct used 1 attr, some single-attr distractors clash; remove.
        pool = [p for p in pool if p != correct_text]
        rng.shuffle(pool)
        # Select 3 unique distractors.
        out = []
        for c in pool:
            if c not in out and c != correct_text:
                out.append(c)
            if len(out) == 3:
                break
        # Ensure distractors differ from correct even textually.
        out = [o for o in out if o != correct_text]
        return out

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, shapes: List[Dict], options: List[str],
                title: str = "Partitioned shapes",
                sub_rng: Optional[random.Random] = None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        fig = plt.figure(figsize=(10.0 * sc, 6.2 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.3], wspace=0.18)
        ax_img = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        ax_img.set_aspect("equal")
        ax_img.axis("off")
        ax_img.set_xlim(0, 10)
        ax_img.set_ylim(0, 6)

        ax_img.set_title(title, fontsize=fs + 2, fontweight="bold",
                         fontfamily=ff, pad=8)

        # Place group 0 on the left, group 1 on the right. Within each group,
        # positions are randomly jittered in a grid.
        g0 = [s for s in shapes if s["group"] == 0]
        g1 = [s for s in shapes if s["group"] == 1]

        rng_local = sub_rng or self._rng

        def place(group_items, x_lo, x_hi, y_lo, y_hi):
            n = len(group_items)
            if n == 0:
                return
            cols = max(2, math.ceil(math.sqrt(n * (x_hi - x_lo) / (y_hi - y_lo))))
            rows = math.ceil(n / cols)
            cell_w = (x_hi - x_lo) / cols
            cell_h = (y_hi - y_lo) / rows
            order = list(range(n))
            rng_local.shuffle(order)
            for idx, i in enumerate(order):
                r = idx // cols
                c = idx % cols
                cx = x_lo + (c + 0.5) * cell_w + rng_local.uniform(-0.08, 0.08)
                cy = y_hi - (r + 0.5) * cell_h + rng_local.uniform(-0.08, 0.08)
                s = group_items[i]
                size_val = _SIZE_MAP[s["size"]]
                color_hex = _COLOR_MAP[s["color"]]
                _draw_shape(ax_img, cx, cy, s["shape"], color_hex, size_val)

        place(g0, 0.4, 4.6, 0.6, 5.4)
        place(g1, 5.4, 9.6, 0.6, 5.4)

        # Dashed partition line down the middle (slightly wavy for variety).
        line_x = 5.0 + rng_local.uniform(-0.15, 0.15)
        ys = np.linspace(0.2, 5.8, 40)
        xs = line_x + 0.08 * np.sin(np.linspace(0, math.pi, 40)
                                    + rng_local.uniform(0, math.pi))
        ax_img.plot(xs, ys, linestyle="--", color="#34495e",
                    linewidth=2.0, zorder=2)

        # Options panel
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 10)
        ax_txt.set_ylim(0, 12)
        ax_txt.text(0.3, 11.5, "Partition rule options:",
                    fontsize=fs + 1, fontweight="bold",
                    fontfamily=ff, color="#2c3e50",
                    ha="left", va="top")
        y = 10.5
        for i, opt in enumerate(options):
            ax_txt.text(0.5, y, f"({chr(ord('A') + i)}) {opt}",
                        fontsize=fs + 1, fontfamily=ff,
                        color="#1a1a1a", ha="left", va="top")
            y -= 0.9

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92,
                             bottom=0.05)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = MultiAttributePartitionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 7 + level, parameter={"level": level})
            print(f"L{level} seed={seed} ok={ok} A={env._answer}")
