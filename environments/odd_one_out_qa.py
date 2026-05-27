"""
Odd One Out QA environment.

Shows 5-6 shapes/patterns. One does not follow the shared rule.
Rules: same color (except one), same shape (except one), same rotation,
same count, or combinations of two rules.
Question: "Which item does not belong?"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_COLORS = {
    "red": "#e74c3c", "blue": "#3498db", "green": "#27ae60",
    "orange": "#e67e22", "purple": "#8e44ad", "yellow": "#f1c40f",
    "teal": "#1abc9c", "pink": "#e91e8f",
}
_COLOR_LIST = list(_COLORS.keys())

_SHAPES = ["circle", "square", "triangle", "pentagon", "star", "hexagon"]
_ROTATIONS = [0, 45, 90, 135, 180]
_COUNTS = [1, 2, 3]
_SIZES = {"small": 0.25, "medium": 0.35, "large": 0.45}
_SIZE_NAMES = list(_SIZES.keys())

def _polygon_path(cx, cy, n, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(n):
        a = off + 2 * math.pi * i / n
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _star_path(cx, cy, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(10):
        a = off + 2 * math.pi * i / 10
        r = size if i % 2 == 0 else size * 0.45
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 9 + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _draw_item(ax, cx, cy, shape, color_name, size, rotation, count):
    """Draw one item (potentially with multiple sub-shapes)."""
    color_hex = _COLORS[color_name]
    if count == 1:
        positions = [(cx, cy)]
    elif count == 2:
        positions = [(cx - size * 0.6, cy), (cx + size * 0.6, cy)]
    else:
        positions = [(cx, cy + size * 0.5),
                     (cx - size * 0.6, cy - size * 0.35),
                     (cx + size * 0.6, cy - size * 0.35)]

    sub_size = size * (0.6 if count > 1 else 1.0)
    for px, py in positions[:count]:
        if shape == "circle":
            p = mpatches.Circle((px, py), sub_size, facecolor=color_hex,
                                edgecolor="#2c3e50", linewidth=1.5, zorder=3)
            ax.add_patch(p)
        elif shape == "star":
            path = _star_path(px, py, sub_size, rotation)
            p = mpatches.PathPatch(path, facecolor=color_hex, edgecolor="#2c3e50",
                                   linewidth=1.5, zorder=3)
            ax.add_patch(p)
        else:
            n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6}.get(shape, 4)
            path = _polygon_path(px, py, n, sub_size, rotation)
            p = mpatches.PathPatch(path, facecolor=color_hex, edgecolor="#2c3e50",
                                   linewidth=1.5, zorder=3)
            ax.add_patch(p)

class _Item:
    __slots__ = ("shape", "color", "size_name", "rotation", "count")

    def __init__(self, shape, color, size_name, rotation, count):
        self.shape = shape
        self.color = color
        self.size_name = size_name
        self.rotation = rotation
        self.count = count

    def copy(self):
        return _Item(self.shape, self.color, self.size_name, self.rotation, self.count)

class OddOneOutQA(StandaloneVisualEnv):
    ENV_NAME = "odd_one_out"

    RULE_TYPES = [
        "same_color", "same_shape", "same_rotation", "same_count",
        "same_size", "color_and_shape", "color_and_count",
        "shape_and_size",
    ]

    def _level_config(self, level: int) -> dict:
        """Return difficulty config for a given level (0-9)."""
        configs = {
            0: {'rule_type': 'same_color', 'num_items': 5},
            1: {'rule_type': 'same_shape', 'num_items': 5},
            2: {'rule_type': 'same_count', 'num_items': 5},
            3: {'rule_type': 'same_size', 'num_items': 5},
            4: {'rule_type': 'same_rotation', 'num_items': 6},
            5: {'rule_type': 'color_and_shape', 'num_items': 6},
            6: {'rule_type': 'color_and_count', 'num_items': 6},
            7: {'rule_type': 'shape_and_size', 'num_items': 6},
            8: {'rule_type': 'shape_and_size', 'num_items': 6},
            9: {'rule_type': 'shape_and_size', 'num_items': 6},
        }
        return configs.get(max(0, min(9, level)), configs[0])

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        # ── Level routing (auto-generated) ──
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "rule_type" not in parameter or parameter.get("rule_type") is None:
            parameter = dict(parameter, **lcfg)

        rng = self._rng
        num_items = parameter.get("num_items", rng.choice([5, 6]))
        rule_type = parameter.get("rule_type")
        if rule_type not in self.RULE_TYPES:
            rule_type = rng.choice(self.RULE_TYPES)

        for _ in range(30):
            result = self._try_generate(rng, num_items, rule_type, level)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, num_items, rule_type, level=0):
        items, odd_idx = self._build_items(rng, num_items, rule_type)
        if items is None:
            return None

        image = self._render(items, rng) if hasattr(self, '_render') else self._render_items(items)
        labels = ", ".join(str(i + 1) for i in range(len(items)))
        question = (
            f"Look at the {len(items)} items labeled 1 through {len(items)}. "
            f"One item does not belong with the others. "
            f"Which item is the odd one out? Answer with a single number ({labels})."
        )
        if level <= 2:
            question += (
                " Hint: examine common attributes (shape, color, count, "
                "size, orientation, ...). Most items share a property; "
                "find the one that breaks the pattern."
            )
        answer = str(odd_idx + 1)
        return question, answer, image

    def _build_items(self, rng, num_items, rule_type):
        """Build items where all share a property except the odd one."""
        # Pick shared attributes
        shared_shape = rng.choice(_SHAPES)
        shared_color = rng.choice(_COLOR_LIST)
        shared_size = rng.choice(_SIZE_NAMES)
        shared_rotation = rng.choice(_ROTATIONS)
        shared_count = rng.choice(_COUNTS)

        odd_idx = rng.randint(0, num_items - 1)
        items = []

        for i in range(num_items):
            if rule_type == "same_color":
                # All same color, odd has different color
                shape = rng.choice(_SHAPES)
                color = shared_color
                size_n = rng.choice(_SIZE_NAMES)
                rot = rng.choice(_ROTATIONS)
                count = rng.choice(_COUNTS)
                if i == odd_idx:
                    color = rng.choice([c for c in _COLOR_LIST if c != shared_color])

            elif rule_type == "same_shape":
                shape = shared_shape
                color = rng.choice(_COLOR_LIST)
                size_n = rng.choice(_SIZE_NAMES)
                rot = rng.choice(_ROTATIONS)
                count = rng.choice(_COUNTS)
                if i == odd_idx:
                    shape = rng.choice([s for s in _SHAPES if s != shared_shape])

            elif rule_type == "same_rotation":
                shape = rng.choice([s for s in _SHAPES if s != "circle"])
                color = rng.choice(_COLOR_LIST)
                size_n = rng.choice(_SIZE_NAMES)
                rot = shared_rotation
                count = rng.choice(_COUNTS)
                if i == odd_idx:
                    rot = rng.choice([r for r in _ROTATIONS if r != shared_rotation])

            elif rule_type == "same_count":
                shape = rng.choice(_SHAPES)
                color = rng.choice(_COLOR_LIST)
                size_n = rng.choice(_SIZE_NAMES)
                rot = rng.choice(_ROTATIONS)
                count = shared_count
                if i == odd_idx:
                    count = rng.choice([c for c in _COUNTS if c != shared_count])

            elif rule_type == "same_size":
                shape = rng.choice(_SHAPES)
                color = rng.choice(_COLOR_LIST)
                size_n = shared_size
                rot = rng.choice(_ROTATIONS)
                count = rng.choice(_COUNTS)
                if i == odd_idx:
                    size_n = rng.choice([s for s in _SIZE_NAMES if s != shared_size])

            elif rule_type == "color_and_shape":
                shape = shared_shape
                color = shared_color
                size_n = rng.choice(_SIZE_NAMES)
                rot = rng.choice(_ROTATIONS)
                count = rng.choice(_COUNTS)
                if i == odd_idx:
                    # Break one of the two
                    if rng.random() < 0.5:
                        color = rng.choice([c for c in _COLOR_LIST if c != shared_color])
                    else:
                        shape = rng.choice([s for s in _SHAPES if s != shared_shape])

            elif rule_type == "color_and_count":
                shape = rng.choice(_SHAPES)
                color = shared_color
                size_n = rng.choice(_SIZE_NAMES)
                rot = rng.choice(_ROTATIONS)
                count = shared_count
                if i == odd_idx:
                    if rng.random() < 0.5:
                        color = rng.choice([c for c in _COLOR_LIST if c != shared_color])
                    else:
                        count = rng.choice([c for c in _COUNTS if c != shared_count])

            elif rule_type == "shape_and_size":
                shape = shared_shape
                # Lock color too — otherwise a unique color on a non-odd item
                # distracts more than the actual shape/size diff. (Previously:
                # color = rng.choice(_COLOR_LIST) made "odd color" compete
                # with "odd shape/size" for attention.)
                color = shared_color
                size_n = shared_size
                rot = rng.choice(_ROTATIONS)
                count = rng.choice(_COUNTS)
                if i == odd_idx:
                    if rng.random() < 0.5:
                        shape = rng.choice([s for s in _SHAPES if s != shared_shape])
                    else:
                        size_n = rng.choice([s for s in _SIZE_NAMES if s != shared_size])
            else:
                return None, -1

            items.append(_Item(shape, color, size_n, rot, count))

        return items, odd_idx

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_items(self, items: list) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        n = len(items)
        cols = min(n, 6)
        rows = (n + cols - 1) // cols
        fig_w = (cols * 2.2 + 0.5) * sc
        fig_h = (rows * 2.5 + 0.8) * sc

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, cols * 2.2)
        ax.set_ylim(0, rows * 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Which item does not belong?", fontsize=fs + 2,
                      fontweight="bold", pad=10, fontfamily=ff)

        cell_w = 2.0
        cell_h = 2.0
        for idx, item in enumerate(items):
            col = idx % cols
            row = rows - 1 - idx // cols
            cx = col * 2.2 + 1.1
            cy = row * 2.5 + 1.25

            rect = mpatches.FancyBboxPatch(
                (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
                boxstyle="round,pad=0.06", facecolor=style["bg_color"],
                edgecolor="#bdc3c7", linewidth=style["line_width"], zorder=1)
            ax.add_patch(rect)

            size_val = _SIZES[item.size_name]
            _draw_item(ax, cx, cy, item.shape, item.color, size_val,
                       item.rotation, item.count)

            ax.text(cx, cy - cell_h / 2 - 0.12, str(idx + 1),
                    fontsize=fs + 2, fontweight="bold", ha="center", va="top",
                    color="#2c3e50", zorder=5, fontfamily=ff)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
