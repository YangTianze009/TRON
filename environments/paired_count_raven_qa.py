"""
Paired-count Raven-style set-membership QA.

Two reference sets (Set A, Set B) each contain a few cells. Within each cell
there are two attribute groups (shape-attr1 and shape-attr2). The hidden rule
is a paired-count rule: in Set A, the count of attribute1 items in each cell
equals the count of attribute2 items; in Set B, the count of attribute1 items
equals the count of a *different* attribute. Then a Figure cell is shown and
the model classifies it: belongs to Set A, Set B, or Neither.

MCQ: A / B / C (Neither).

Difficulty levels 0-9:
  L0-L1: 2 cells per set, very small counts (1-2), Neither rare; only A or B.
  L2-L3: 2-3 cells per set, counts 1-3, Neither possible.
  L4-L5: 3 cells per set, counts 1-4.
  L6-L7: 3-4 cells, counts 1-5; harder distractor counts.
  L8-L9: 4 cells, counts 1-5, more candidate attributes.
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


_SHAPE_TYPES = ["star", "circle", "square", "triangle", "hexagon", "diamond"]
_SHAPE_NAMES = {
    "star": "stars",
    "circle": "circles",
    "square": "squares",
    "triangle": "triangles",
    "hexagon": "hexagons",
    "diamond": "diamonds",
}

_QUESTION_TEMPLATES = [
    "Examine the two sets and the Figure. Each set follows a hidden paired-count rule. To which set does the Figure belong? Answer with one letter: A (Set A), B (Set B), or C (Neither).",
    "Set A and Set B follow different paired-count rules. Which set does the Figure belong to? Reply (A) Set A, (B) Set B, or (C) Neither.",
    "Identify the rule for each set, then classify the Figure: (A) Set A, (B) Set B, (C) Neither. Answer with a single letter.",
    "Each set's cells share a paired-count relationship between two shape types. To which set does the Figure belong? A=Set A, B=Set B, C=Neither.",
    "Two sets are shown. Determine the rule of each, then place the Figure: (A) Set A, (B) Set B, or (C) Neither. Single letter answer.",
    "Look at how the counts of two shape types relate in each set. The Figure belongs to: (A) Set A, (B) Set B, (C) Neither.",
    "Inspect Set A and Set B. Each follows a paired-count constraint. Where does the Figure fit? A, B, or C (Neither).",
    "Study the two sets. Which set best matches the Figure's attribute counts? (A) Set A (B) Set B (C) Neither.",
    "After working out each set's rule, classify the Figure with one letter: A (Set A), B (Set B), C (Neither).",
    "Pick the correct membership: (A) Set A, (B) Set B, (C) Neither. Use the paired-count pattern in each set.",
    "Both sets share an internal counting rule. Decide whether the Figure belongs to Set A (A), Set B (B), or Neither (C).",
    "Identify the paired-count rule in each set. The Figure belongs to: A (Set A) / B (Set B) / C (Neither).",
    "Examine each cell carefully. Which classification fits the Figure? (A) Set A (B) Set B (C) Neither — single letter.",
    "Each set defines a paired-count relationship between two shape types. Classify the Figure: A, B, or C (Neither).",
    "Decide membership of the Figure based on the two sets' rules: (A) Set A (B) Set B (C) Neither.",
    "After inferring each set's count rule, choose: (A) Set A, (B) Set B, (C) Neither. One letter only.",
]


def _draw_shape(ax, shape: str, cx: float, cy: float, s: float,
                color: str, edge: str = "#222"):
    """Render a single small shape centered at (cx, cy)."""
    if shape == "circle":
        ax.add_patch(mpatches.Circle((cx, cy), s, facecolor=color,
                                     edgecolor=edge, linewidth=1.0, zorder=3))
    elif shape == "square":
        ax.add_patch(mpatches.Rectangle((cx - s, cy - s), 2 * s, 2 * s,
                                        facecolor=color, edgecolor=edge,
                                        linewidth=1.0, zorder=3))
    elif shape == "triangle":
        verts = [(cx, cy + s), (cx - s, cy - s * 0.85),
                 (cx + s, cy - s * 0.85)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=color, edgecolor=edge,
                                      linewidth=1.0, zorder=3))
    elif shape == "diamond":
        verts = [(cx, cy + s), (cx + s, cy),
                 (cx, cy - s), (cx - s, cy)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=color, edgecolor=edge,
                                      linewidth=1.0, zorder=3))
    elif shape == "pentagon":
        verts = [(cx + s * math.cos(math.radians(72 * i + 90)),
                  cy + s * math.sin(math.radians(72 * i + 90)))
                 for i in range(5)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=color, edgecolor=edge,
                                      linewidth=1.0, zorder=3))
    elif shape == "hexagon":
        verts = [(cx + s * math.cos(math.radians(60 * i + 30)),
                  cy + s * math.sin(math.radians(60 * i + 30)))
                 for i in range(6)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=color, edgecolor=edge,
                                      linewidth=1.0, zorder=3))
    elif shape == "star":
        verts = []
        for i in range(10):
            a = math.pi / 2 + 2 * math.pi * i / 10
            r = s if i % 2 == 0 else s * 0.45
            verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        ax.add_patch(mpatches.Polygon(verts, facecolor=color, edgecolor=edge,
                                      linewidth=1.0, zorder=3))


def _grid_positions(n: int, cell_w: float, cell_h: float):
    """Return n positions inside a cell with width/height = cell_w, cell_h."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-cell_w * 0.22, 0.0), (cell_w * 0.22, 0.0)]
    if n == 3:
        return [(0.0, cell_h * 0.18),
                (-cell_w * 0.22, -cell_h * 0.15),
                (cell_w * 0.22, -cell_h * 0.15)]
    if n == 4:
        return [(-cell_w * 0.22, cell_h * 0.18), (cell_w * 0.22, cell_h * 0.18),
                (-cell_w * 0.22, -cell_h * 0.18), (cell_w * 0.22, -cell_h * 0.18)]
    if n == 5:
        return [(-cell_w * 0.27, cell_h * 0.20), (0.0, cell_h * 0.20),
                (cell_w * 0.27, cell_h * 0.20),
                (-cell_w * 0.18, -cell_h * 0.18),
                (cell_w * 0.18, -cell_h * 0.18)]
    # 6+
    out = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        out.append((cell_w * 0.27 * math.cos(ang),
                    cell_h * 0.27 * math.sin(ang)))
    return out


class PairedCountRavenQA(StandaloneVisualEnv):
    ENV_NAME = "paired_count_raven"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"cells_per_set": 2, "max_count": 2, "neither_prob": 0.10}
        if level <= 3:
            return {"cells_per_set": 2, "max_count": 3, "neither_prob": 0.25}
        if level <= 5:
            return {"cells_per_set": 3, "max_count": 4, "neither_prob": 0.30}
        if level <= 7:
            return {"cells_per_set": 3, "max_count": 5, "neither_prob": 0.33}
        return {"cells_per_set": 4, "max_count": 5, "neither_prob": 0.34}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 8801)

        for _ in range(20):
            result = self._try_generate(sub_rng, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        # Pick 3 distinct shape types: pivot + match_A + match_B.
        # Set A rule: count(pivot) == count(match_A)
        # Set B rule: count(pivot) == count(match_B)
        shapes = rng.sample(_SHAPE_TYPES, 3)
        pivot, match_a, match_b = shapes
        max_count = cfg["max_count"]
        cells_per_set = cfg["cells_per_set"]

        # Each Set A cell must satisfy Set A's rule (pivot==match_a) and NOT
        # satisfy Set B's rule (pivot != match_b), so the two sets are
        # cleanly separable.
        def _make_setA_cell():
            n_pivot = rng.randint(1, max_count)
            # match_b must differ from pivot
            choices = [c for c in range(max_count + 1) if c != n_pivot]
            return {"pivot": n_pivot, "match_a": n_pivot,
                    "match_b": rng.choice(choices)}

        def _make_setB_cell():
            n_pivot = rng.randint(1, max_count)
            choices = [c for c in range(max_count + 1) if c != n_pivot]
            return {"pivot": n_pivot,
                    "match_a": rng.choice(choices),
                    "match_b": n_pivot}

        set_a_cells = [_make_setA_cell() for _ in range(cells_per_set)]
        set_b_cells = [_make_setB_cell() for _ in range(cells_per_set)]

        # Pick a target classification (A, B, or C=Neither) by weighted draw.
        u = rng.random()
        if u < cfg["neither_prob"]:
            target = "C"
        elif u < cfg["neither_prob"] + (1 - cfg["neither_prob"]) / 2:
            target = "A"
        else:
            target = "B"

        if target == "A":
            n_pivot = rng.randint(1, max_count)
            # Match Set A rule: pivot == match_a, but pivot != match_b
            n_match_b = rng.randint(0, max_count)
            if n_match_b == n_pivot:
                n_match_b = (n_pivot + 1) % (max_count + 1)
            figure = {"pivot": n_pivot, "match_a": n_pivot, "match_b": n_match_b}
        elif target == "B":
            n_pivot = rng.randint(1, max_count)
            n_match_a = rng.randint(0, max_count)
            if n_match_a == n_pivot:
                n_match_a = (n_pivot + 1) % (max_count + 1)
            figure = {"pivot": n_pivot, "match_a": n_match_a, "match_b": n_pivot}
        else:  # Neither
            n_pivot = rng.randint(1, max_count)
            n_match_a = rng.randint(0, max_count)
            n_match_b = rng.randint(0, max_count)
            attempts = 0
            while (n_match_a == n_pivot or n_match_b == n_pivot) and attempts < 30:
                n_match_a = rng.randint(0, max_count)
                n_match_b = rng.randint(0, max_count)
                attempts += 1
            if n_match_a == n_pivot or n_match_b == n_pivot:
                return None
            figure = {"pivot": n_pivot, "match_a": n_match_a, "match_b": n_match_b}

        # Sanity check: figure must satisfy exactly the target's rule
        sat_a = (figure["pivot"] == figure["match_a"])
        sat_b = (figure["pivot"] == figure["match_b"])
        if target == "A" and not (sat_a and not sat_b):
            return None
        if target == "B" and not (sat_b and not sat_a):
            return None
        if target == "C" and (sat_a or sat_b):
            return None

        img = self._render(set_a_cells, set_b_cells, figure,
                           pivot, match_a, match_b, rng)
        sidx = (self.seed or 0) % len(_QUESTION_TEMPLATES)
        question = _QUESTION_TEMPLATES[sidx]
        return question, target, img

    def _render(self, set_a_cells, set_b_cells, figure_cell,
                pivot: str, match_a: str, match_b: str, rng) -> Image.Image:
        style = self._random_style()
        palette = style["palette"]
        # Map each shape to a stable color from palette
        color_pivot = palette[0]
        color_match_a = palette[2]
        color_match_b = palette[4]

        n_a = len(set_a_cells)
        n_b = len(set_b_cells)
        n_cols = max(n_a, n_b)
        fig_w = max(7, 1.4 * n_cols + 2)
        fig_h = 8.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, n_cols + 1)
        ax.set_ylim(0, 7)
        ax.set_aspect("equal")
        ax.axis("off")

        cell_w = 0.9
        cell_h = 1.4

        def _draw_set(cells, y_top: float, label: str):
            ax.text(0.05, y_top + cell_h * 0.55, label,
                    fontsize=14, fontweight="bold", color="#222",
                    ha="left", va="center")
            for i, cell in enumerate(cells):
                cx = (i + 1) * 1.05
                cy = y_top
                ax.add_patch(mpatches.Rectangle(
                    (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
                    facecolor="#ffffff", edgecolor="#222", linewidth=1.5))
                # Place pivot shapes in top half, match_a + match_b in bottom half
                # Pivot top, then match_a left-bottom and match_b right-bottom mixed
                # We will just place all of them: pivots tinted color_pivot,
                # match_a color_match_a, match_b color_match_b.
                # Layout: split cell into 3 horizontal strips top/mid/bot.
                strip_y = [cy + cell_h * 0.30, cy, cy - cell_h * 0.30]
                # Pivot strip
                pos = _grid_positions(cell["pivot"], cell_w, cell_h * 0.3)
                for px, py in pos:
                    _draw_shape(ax, pivot, cx + px, strip_y[0] + py * 0.3,
                                0.06, color_pivot)
                # Match A strip
                pos = _grid_positions(cell["match_a"], cell_w, cell_h * 0.3)
                for px, py in pos:
                    _draw_shape(ax, match_a, cx + px, strip_y[1] + py * 0.3,
                                0.06, color_match_a)
                # Match B strip
                pos = _grid_positions(cell["match_b"], cell_w, cell_h * 0.3)
                for px, py in pos:
                    _draw_shape(ax, match_b, cx + px, strip_y[2] + py * 0.3,
                                0.06, color_match_b)

        _draw_set(set_a_cells, y_top=5.6, label="Set A")
        _draw_set(set_b_cells, y_top=3.4, label="Set B")

        # Figure
        ax.text(0.05, 1.3, "Figure", fontsize=14, fontweight="bold",
                color="#b00", ha="left", va="center")
        cx = 1.05
        cy = 1.0
        ax.add_patch(mpatches.Rectangle(
            (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
            facecolor="#fff8f0", edgecolor="#b00", linewidth=2.0,
            linestyle="--"))
        strip_y = [cy + cell_h * 0.30, cy, cy - cell_h * 0.30]
        pos = _grid_positions(figure_cell["pivot"], cell_w, cell_h * 0.3)
        for px, py in pos:
            _draw_shape(ax, pivot, cx + px, strip_y[0] + py * 0.3,
                        0.06, color_pivot)
        pos = _grid_positions(figure_cell["match_a"], cell_w, cell_h * 0.3)
        for px, py in pos:
            _draw_shape(ax, match_a, cx + px, strip_y[1] + py * 0.3,
                        0.06, color_match_a)
        pos = _grid_positions(figure_cell["match_b"], cell_w, cell_h * 0.3)
        for px, py in pos:
            _draw_shape(ax, match_b, cx + px, strip_y[2] + py * 0.3,
                        0.06, color_match_b)

        # Legend / hint - identify the three shape types
        leg_x = (n_cols + 1) * 0.5
        ax.text(leg_x, 0.15,
                f"Top row: {_SHAPE_NAMES[pivot]} · "
                f"Middle row: {_SHAPE_NAMES[match_a]} · "
                f"Bottom row: {_SHAPE_NAMES[match_b]}",
                fontsize=9, color="#333", ha="center", va="bottom",
                style="italic")
        # Options reminder bar at bottom
        ax.text(leg_x, -0.05,
                "(A) Set A     (B) Set B     (C) Neither",
                fontsize=11, fontweight="bold", color="#222",
                ha="center", va="top")

        return self.fig_to_pil(fig, dpi=style["dpi"])
