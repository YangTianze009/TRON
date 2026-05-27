"""
Raven-style Progressive Matrix QA environment.

Generates a 3x3 grid of shapes/patterns where the bottom-right cell is missing.
Rules (color, size, rotation, count, shape morph) apply across rows or columns.
Answer is multiple choice (A-F) with 1 correct + distractors.
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
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_COLORS = ["#e74c3c", "#3498db", "#27ae60", "#e67e22", "#8e44ad", "#f1c40f"]
_COLOR_NAMES = ["red", "blue", "green", "orange", "purple", "yellow"]
_SHAPES = ["circle", "square", "triangle", "pentagon", "star"]
_SIZES = [0.15, 0.22, 0.30, 0.38]
_ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]

# 16 question templates — {opts} is filled with option letters.
_Q_TEMPLATES = [
    "Look at the 3x3 pattern grid. The bottom-right cell is missing. Which option ({opts}) completes the pattern? Answer with a single letter.",
    "The 3x3 matrix below has one missing cell. Which choice ({opts}) best fits?",
    "Examine the puzzle matrix. Which option ({opts}) belongs in the empty square?",
    "A 3x3 analogy pattern is shown with the bottom-right cell removed. Pick the correct completion ({opts}).",
    "Study the 3x3 pattern. Choose the option ({opts}) that continues the rule.",
    "Which single letter option ({opts}) fills the missing tile in this 3x3 grid?",
    "In the Raven-style matrix shown, which option among {opts} completes the pattern?",
    "Identify the cell that correctly completes the 3x3 pattern: {opts}.",
    "Given the 3x3 pattern with a missing tile, determine the correct answer ({opts}).",
    "Select the option ({opts}) that matches the rule governing the 3x3 grid.",
    "Which option ({opts}) satisfies the pattern rules and fills the empty cell?",
    "Inspect the 3x3 analogy grid. Which of {opts} is the correct fill?",
    "The missing tile in the 3x3 pattern corresponds to which option? {opts}.",
    "After working out the rule in the 3x3 grid, pick the right option ({opts}).",
    "Decide which of the candidates ({opts}) completes the 3x3 pattern shown.",
    "Reason about the 3x3 pattern and choose the answer from {opts}.",
]

def _shape_path(shape: str, cx: float, cy: float, size: float,
                rotation_deg: float = 0) -> mpath.Path:
    """Generate a matplotlib Path for a shape centred at (cx, cy)."""
    n_pts = {"circle": 32, "square": 4, "triangle": 3,
             "pentagon": 5, "star": 10}.get(shape, 4)
    angle_offset = math.radians(rotation_deg)

    if shape == "circle":
        angles = [2 * math.pi * i / 32 for i in range(33)]
        verts = [(cx + size * math.cos(a), cy + size * math.sin(a))
                 for a in angles]
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 31 + [mpath.Path.CLOSEPOLY]
        return mpath.Path(verts, codes)
    elif shape == "star":
        verts = []
        for i in range(10):
            a = angle_offset + math.pi / 2 + 2 * math.pi * i / 10
            r = size if i % 2 == 0 else size * 0.45
            verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        verts.append(verts[0])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 9 + [mpath.Path.CLOSEPOLY]
        return mpath.Path(verts, codes)
    else:
        n = {"triangle": 3, "square": 4, "pentagon": 5}[shape]
        verts = []
        for i in range(n):
            a = angle_offset + math.pi / 2 + 2 * math.pi * i / n
            verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
        verts.append(verts[0])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1) + [mpath.Path.CLOSEPOLY]
        return mpath.Path(verts, codes)

def _draw_cell(ax, cx, cy, cell_w, cell_h, shape, color, size, rotation,
               count=1, bg="#ffffff"):
    """Draw shape(s) inside a cell area."""
    rect = mpatches.FancyBboxPatch(
        (cx - cell_w / 2 + 0.01, cy - cell_h / 2 + 0.01),
        cell_w - 0.02, cell_h - 0.02,
        boxstyle="round,pad=0.005", facecolor=bg, edgecolor="#bdc3c7",
        linewidth=1.5, zorder=1)
    ax.add_patch(rect)

    if count == 1:
        positions = [(cx, cy)]
    elif count == 2:
        positions = [(cx - cell_w * 0.15, cy), (cx + cell_w * 0.15, cy)]
    elif count == 3:
        positions = [(cx, cy + cell_h * 0.12),
                     (cx - cell_w * 0.15, cy - cell_h * 0.1),
                     (cx + cell_w * 0.15, cy - cell_h * 0.1)]
    else:
        positions = [(cx - cell_w * 0.15, cy + cell_h * 0.12),
                     (cx + cell_w * 0.15, cy + cell_h * 0.12),
                     (cx - cell_w * 0.15, cy - cell_h * 0.1),
                     (cx + cell_w * 0.15, cy - cell_h * 0.1)]

    scaled = size * 0.7 if count > 1 else size
    for px, py in positions[:count]:
        p = _shape_path(shape, px, py, scaled * cell_w, rotation)
        patch = mpatches.PathPatch(p, facecolor=color, edgecolor="#2c3e50",
                                   linewidth=1.2, zorder=3)
        ax.add_patch(patch)

class RavenMatrixQA(StandaloneVisualEnv):
    ENV_NAME = "raven_matrix"

    DIFFICULTY_LEVELS = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
        "expert": 4,
    }

    # ------------------------------------------------------------------ #
    # Rule application
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_rule(rule_name: str, values: list, rng) -> list:
        """Given a rule and row-0 values, produce values for rows 1 and 2."""
        if rule_name == "color_cycle":
            return values  # pre-built
        elif rule_name == "size_increase":
            return values
        elif rule_name == "rotation_step":
            return values
        elif rule_name == "count_increment":
            return values
        elif rule_name == "shape_morph":
            return values
        return values

    def _build_attribute_grid(self, rng, num_rules: int):
        """Build a 3x3 grid of cell attributes following `num_rules` rules.

        Each cell: dict(shape, color, size, rotation, count).
        Returns (grid_3x3, rule_descriptions).
        """
        available_rules = ["color_cycle", "size_increase", "rotation_step",
                           "count_increment", "shape_morph"]
        chosen = rng.sample(available_rules, min(num_rules, len(available_rules)))
        # Per-rule direction so rules are not all collapsed to a single axis.
        # (If every rule uses the same direction the whole matrix flattens to
        # three identical rows/cols, which kills the puzzle.) When we have
        # >=2 rules, force at least one along each axis.
        if len(chosen) >= 2:
            shuffled = list(chosen)
            rng.shuffle(shuffled)
            # Force one row and one col, randomize the rest.
            forced = {shuffled[0]: "row", shuffled[1]: "col"}
            rule_dirs = dict(forced)
            for r in shuffled[2:]:
                rule_dirs[r] = rng.choice(["row", "col"])
        else:
            rule_dirs = {r: rng.choice(["row", "col"]) for r in chosen}
        # Keep a default 'direction' alias for any legacy references.
        direction = rule_dirs[chosen[0]] if chosen else "row"

        # Base attributes (constant unless a rule changes them)
        base_shape = rng.choice(_SHAPES)
        base_color = rng.choice(_COLORS)
        base_size = _SIZES[1]
        base_rot = 0
        base_count = 1

        # Build 3x3 grid
        grid = [[dict(shape=base_shape, color=base_color, size=base_size,
                       rotation=base_rot, count=base_count)
                 for _ in range(3)] for _ in range(3)]

        for rule in chosen:
            d = rule_dirs[rule]
            if rule == "color_cycle":
                c_indices = rng.sample(range(len(_COLORS)), 3)
                for line in range(3):
                    for step in range(3):
                        r, c = (line, step) if d == "row" else (step, line)
                        grid[r][c]["color"] = _COLORS[c_indices[step]]
            elif rule == "size_increase":
                sizes = [_SIZES[0], _SIZES[1], _SIZES[2]]
                for line in range(3):
                    for step in range(3):
                        r, c = (line, step) if d == "row" else (step, line)
                        grid[r][c]["size"] = sizes[step]
            elif rule == "rotation_step":
                rot_step = rng.choice([45, 90, 120])
                for line in range(3):
                    for step in range(3):
                        r, c = (line, step) if d == "row" else (step, line)
                        grid[r][c]["rotation"] = rot_step * step
            elif rule == "count_increment":
                for line in range(3):
                    for step in range(3):
                        r, c = (line, step) if d == "row" else (step, line)
                        grid[r][c]["count"] = step + 1
            elif rule == "shape_morph":
                shapes = rng.sample(_SHAPES, 3)
                for line in range(3):
                    for step in range(3):
                        r, c = (line, step) if d == "row" else (step, line)
                        grid[r][c]["shape"] = shapes[step]

        return grid, chosen, direction

    def _make_distractors(self, rng, correct_cell: dict, num: int) -> list:
        """Generate distractor cells that differ from correct in 1-2 attributes."""
        distractors = []
        attempts = 0
        while len(distractors) < num and attempts < 200:
            attempts += 1
            d = dict(correct_cell)
            n_changes = rng.randint(1, 2)
            attrs = rng.sample(["shape", "color", "size", "rotation", "count"],
                               n_changes)
            for attr in attrs:
                if attr == "shape":
                    d["shape"] = rng.choice([s for s in _SHAPES if s != correct_cell["shape"]])
                elif attr == "color":
                    d["color"] = rng.choice([c for c in _COLORS if c != correct_cell["color"]])
                elif attr == "size":
                    d["size"] = rng.choice([s for s in _SIZES if s != correct_cell["size"]])
                elif attr == "rotation":
                    d["rotation"] = rng.choice([r for r in _ROTATIONS
                                                if r != correct_cell["rotation"]])
                elif attr == "count":
                    d["count"] = rng.choice([c for c in [1, 2, 3, 4]
                                             if c != correct_cell["count"]])
            # Ensure not duplicate of correct
            if d != correct_cell:
                distractors.append(d)
        return distractors[:num]

    # ------------------------------------------------------------------ #
    # Main generation
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"num_rules": 1, "num_options": 4}
        if level <= 2:
            return {"num_rules": 1, "num_options": 5}
        if level <= 4:
            return {"num_rules": 2, "num_options": 5}
        if level <= 6:
            return {"num_rules": 2, "num_options": 6}
        if level <= 8:
            return {"num_rules": 3, "num_options": 6}
        return {"num_rules": 4, "num_options": 6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Use sub_rng that includes level so L0 != L9 for same seed.
        sub_rng = random.Random(seed * 1000 + level * 37 + 4401)
        num_rules = cfg["num_rules"]
        num_options = cfg["num_options"]

        for _ in range(30):
            result = self._try_generate(sub_rng, num_rules, num_options)
            if result is not None:
                # 2026-05-04: simplified L0 (was 10% too-hard) — concise + hint
                # naming the single rule axis (only 1 rule at L0/L1).
                if level <= 1:
                    q, ans, img = result
                    q = q + (
                        " Be concise. Exactly ONE attribute (color, shape, "
                        "size, count, or rotation) varies systematically across "
                        "the rows or columns; pick the option whose attributes "
                        "match. Output only the letter."
                    )
                    result = (q, ans, img)
                return result
        return None

    def _try_generate(self, rng, num_rules, num_options):
        grid, rules, direction = self._build_attribute_grid(rng, num_rules)
        correct_cell = grid[2][2]

        distractors = self._make_distractors(rng, correct_cell, num_options - 1)
        if len(distractors) < num_options - 1:
            return None

        # Build options: insert correct at random position
        options = list(distractors)
        correct_idx = rng.randint(0, len(options))
        options.insert(correct_idx, correct_cell)
        answer_letter = chr(ord("A") + correct_idx)

        # Render
        image = self._render_matrix(grid, options, rng)
        option_str = ", ".join(chr(ord("A") + i) for i in range(len(options)))
        sidx = (self.seed or 0) % len(_Q_TEMPLATES)
        question = _Q_TEMPLATES[sidx].format(opts=option_str)
        return question, answer_letter, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_matrix(self, grid, options, rng=None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        n_opts = len(options)

        # Render mode: clean / textbook / sketch
        mode_rng = rng if rng is not None else random.Random()
        mode = pick_render_mode(mode_rng)
        if mode == "textbook":
            tbp = textbook_params(mode_rng)
            bg = tbp["bg"]
            missing_fill = tbp["fill_color"]
            missing_edge = tbp["line_color"]
            q_color = tbp["line_color"]
            font_kw = {"fontfamily": tbp["font_family"]}
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = mode_rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            missing_fill = "#f5f0e0"
            missing_edge = "#1a1a1a"
            q_color = "#1a1a1a"
            font_kw = {}
            dpi = style["dpi"]
        else:
            bg = style["bg_color"]
            missing_fill = style["palette"][5]
            missing_edge = style["palette"][1]
            q_color = style["palette"][1]
            font_kw = {"fontfamily": ff}
            dpi = style["dpi"]

        def _draw():
            fig_h = 8.5 * sc
            fig_w = max(8, n_opts * 1.6) * sc
            fig, (ax_grid, ax_opts) = plt.subplots(
                2, 1, figsize=(fig_w, fig_h),
                gridspec_kw={"height_ratios": [3, 1.2]})
            fig.patch.set_facecolor(bg)

            ax_grid.set_facecolor(bg)
            ax_grid.set_xlim(0, 3)
            ax_grid.set_ylim(0, 3)
            ax_grid.set_aspect("equal")
            ax_grid.axis("off")
            ax_grid.set_title("Pattern Matrix", fontsize=fs + 3,
                              fontweight="bold", pad=8, **font_kw)

            cell_w = 1.0
            cell_h = 1.0
            for r in range(3):
                for c in range(3):
                    cx = c * cell_w + cell_w / 2
                    cy = (2 - r) * cell_h + cell_h / 2
                    if r == 2 and c == 2:
                        rect = mpatches.FancyBboxPatch(
                            (cx - cell_w / 2 + 0.01, cy - cell_h / 2 + 0.01),
                            cell_w - 0.02, cell_h - 0.02,
                            boxstyle="round,pad=0.005", facecolor=missing_fill,
                            edgecolor=missing_edge, linewidth=style["line_width"],
                            linestyle="--", zorder=1)
                        ax_grid.add_patch(rect)
                        ax_grid.text(cx, cy, "?", fontsize=fs + 16,
                                     fontweight="bold", color=q_color,
                                     ha="center", va="center", zorder=5,
                                     **font_kw)
                    else:
                        cell = grid[r][c]
                        _draw_cell(ax_grid, cx, cy, cell_w, cell_h,
                                   cell["shape"], cell["color"], cell["size"],
                                   cell["rotation"], cell["count"], bg=bg)

            ax_opts.set_facecolor(bg)
            ax_opts.set_xlim(0, n_opts)
            ax_opts.set_ylim(0, 1)
            ax_opts.set_aspect("equal")
            ax_opts.axis("off")
            ax_opts.set_title("Options", fontsize=fs + 1, fontweight="bold",
                              pad=6, **font_kw)

            opt_w = 1.0
            for i, opt in enumerate(options):
                cx = i * opt_w + opt_w / 2
                _draw_cell(ax_opts, cx, 0.5, opt_w * 0.9, 0.9,
                           opt["shape"], opt["color"], opt["size"],
                           opt["rotation"], opt["count"], bg=bg)
                label = chr(ord("A") + i)
                ax_opts.text(cx, 0.02, label, fontsize=fs, fontweight="bold",
                             ha="center", va="bottom", color=q_color,
                             zorder=5, **font_kw)

            fig.tight_layout()
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=80, randomness=1.5):
                fig = _draw()
        else:
            fig = _draw()
        return self.fig_to_pil(fig, dpi=dpi)
