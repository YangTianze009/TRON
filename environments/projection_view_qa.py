"""
Projection View QA environment.

Shows an isometric 3D view of a simple object built from unit cubes,
alongside multiple-choice 2D projection views.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: single cube or 2-cube line. `identify_top_view` or `count_visible_from_front`.
L1: 2-3 cube L or line, identify front/top view MCQ.
L2: simple L/T shapes (3-4 cubes), identify MCQ with wildly distinct distractors.
L3: L, T, plus shapes, identify MCQ.
L4: all 6 shapes, MCQ.
L5: + count_visible_from_front.
L6: all 6 shapes + similar distractors.
L7: harder shapes (larger).
L8: largest shapes, harder distractors.
L9: full complexity.

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _project_2d(grid, axis):
    gx, gy, gz = grid.shape
    if axis == "front":
        proj = np.zeros((gz, gx), dtype=int)
        for i in range(gx):
            for k in range(gz):
                if np.any(grid[i, :, k]):
                    proj[gz - 1 - k, i] = 1
    elif axis == "side":
        proj = np.zeros((gz, gy), dtype=int)
        for j in range(gy):
            for k in range(gz):
                if np.any(grid[:, j, k]):
                    proj[gz - 1 - k, j] = 1
    elif axis == "top":
        proj = np.zeros((gy, gx), dtype=int)
        for i in range(gx):
            for j in range(gy):
                if np.any(grid[i, j, :]):
                    proj[gy - 1 - j, i] = 1
    else:
        proj = np.zeros((1, 1), dtype=int)
    return proj

def _make_distractor(proj, rng, strong=True):
    """Make a distractor by flipping cells in the projection."""
    dist = proj.copy()
    h, w = dist.shape
    if strong:
        num_flips = rng.randint(max(2, h * w // 3), max(3, h * w // 2))
    else:
        num_flips = rng.randint(1, max(2, h * w // 4))
    for _ in range(num_flips):
        r, c = rng.randint(0, h - 1), rng.randint(0, w - 1)
        dist[r, c] = 1 - dist[r, c]
    if np.array_equal(dist, proj):
        r, c = rng.randint(0, h - 1), rng.randint(0, w - 1)
        dist[r, c] = 1 - dist[r, c]
    return dist

def _pad_to_shape(proj, target_shape):
    """Pad a 2D binary array to target_shape (centered)."""
    h, w = proj.shape
    th, tw = target_shape
    if h >= th and w >= tw:
        return proj[:th, :tw]
    padded = np.zeros((max(th, h), max(tw, w)), dtype=int)
    padded[:h, :w] = proj
    return padded[:th, :tw]

class ProjectionViewQA(StandaloneVisualEnv):
    ENV_NAME = "projection_view"

    QUESTION_TYPES = [
        "identify_front_view", "identify_side_view",
        "identify_top_view", "count_visible_from_front",
    ]

    SHAPES_L0 = ["single", "line2"]
    SHAPES_L2 = ["line2", "line3", "l_flat", "l_shape"]
    SHAPES_L4 = ["l_shape", "t_shape", "stairs", "u_shape", "plus", "tower"]
    SHAPES_HARD = ["t_shape", "stairs", "u_shape", "plus", "tower"]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"], weights=cfg["qtype_weights"])[0]
        for _ in range(15):
            result = self._try_generate(qtype, level, cfg)
            if result is not None:
                self._primary_complexity_feature = level * 3 + len(result[1])
                return result
        return None

    def _level_config(self, level):
        if level == 0:
            return {"qtypes": ["identify_top_view"], "qtype_weights": [1],
                    "shapes": self.SHAPES_L0, "strong_distractor": True,
                    "n_options": 4}
        if level == 1:
            return {"qtypes": ["identify_top_view", "identify_front_view"],
                    "qtype_weights": [5, 5],
                    "shapes": self.SHAPES_L0 + ["line3"],
                    "strong_distractor": True, "n_options": 4}
        if level == 2:
            return {"qtypes": ["identify_top_view", "identify_front_view"],
                    "qtype_weights": [5, 5],
                    "shapes": self.SHAPES_L2,
                    "strong_distractor": True, "n_options": 4}
        if level == 3:
            return {"qtypes": ["identify_top_view", "identify_front_view",
                               "identify_side_view"],
                    "qtype_weights": [4, 4, 2],
                    "shapes": self.SHAPES_L2 + ["t_shape"],
                    "strong_distractor": True, "n_options": 4}
        if level == 4:
            return {"qtypes": ["identify_top_view", "identify_front_view",
                               "identify_side_view"],
                    "qtype_weights": [4, 3, 3],
                    "shapes": self.SHAPES_L4,
                    "strong_distractor": True, "n_options": 4}
        if level == 5:
            return {"qtypes": ["identify_top_view", "identify_front_view",
                               "identify_side_view", "count_visible_from_front"],
                    "qtype_weights": [3, 3, 2, 2],
                    "shapes": self.SHAPES_L4,
                    "strong_distractor": False, "n_options": 4}
        if level == 6:
            return {"qtypes": ["identify_top_view", "identify_front_view",
                               "identify_side_view", "count_visible_from_front"],
                    "qtype_weights": [3, 3, 2, 2],
                    "shapes": self.SHAPES_L4,
                    "strong_distractor": False, "n_options": 4}
        if level == 7:
            return {"qtypes": ["identify_top_view", "identify_front_view",
                               "identify_side_view", "count_visible_from_front"],
                    "qtype_weights": [3, 3, 2, 2],
                    "shapes": self.SHAPES_HARD,
                    "strong_distractor": False, "n_options": 4}
        if level == 8:
            return {"qtypes": ["identify_top_view", "identify_front_view",
                               "identify_side_view", "count_visible_from_front"],
                    "qtype_weights": [3, 3, 2, 2],
                    "shapes": self.SHAPES_HARD,
                    "strong_distractor": False, "n_options": 4}
        return {"qtypes": ["identify_top_view", "identify_front_view",
                           "identify_side_view", "count_visible_from_front"],
                "qtype_weights": [3, 3, 2, 2],
                "shapes": self.SHAPES_HARD,
                "strong_distractor": False, "n_options": 4}

    def _try_generate(self, qtype, level, cfg):
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )

        # L0: flat 2D rendering to avoid 3D isometric truncation loops
        if level == 0:
            return self._generate_l0_flat(sub_rng, cfg)

        shape = sub_rng.choice(cfg["shapes"])
        grid = self._make_shape(sub_rng, shape)
        if grid is None or int(np.sum(grid)) == 0:
            return None

        if qtype == "count_visible_from_front":
            front_proj = _project_2d(grid, "front")
            count = int(np.sum(front_proj))
            stems = [
                "Looking at the 3D structure shown, how many unit squares are visible when viewed from the front (along the y-axis)?",
                "Count the total number of unit squares in the front-view silhouette of the structure.",
            ]
            q = sub_rng.choice(stems) + " Reply with a single integer inside <answer>...</answer>. Example: <answer>3</answer>"
            image = self._render_iso_only(sub_rng, grid, shape)
            return q, str(count), image

        if qtype == "identify_front_view":
            axis, view_name = "front", "front"
        elif qtype == "identify_side_view":
            axis, view_name = "side", "right side"
        else:
            axis, view_name = "top", "top"

        correct_proj = _project_2d(grid, axis)
        options = [correct_proj]
        max_tries = 20
        tries = 0
        while len(options) < cfg["n_options"] and tries < max_tries:
            tries += 1
            d = _make_distractor(correct_proj, sub_rng,
                                 strong=cfg["strong_distractor"])
            if not any(np.array_equal(d, o) for o in options):
                options.append(d)
        if len(options) < cfg["n_options"]:
            return None

        indices = list(range(len(options)))
        sub_rng.shuffle(indices)
        shuffled_options = [options[i] for i in indices]
        correct_label = chr(65 + indices.index(0))

        stems = [
            f"The 3D structure is shown on the left. Which of the options (A, B, C, D) shows the correct {view_name} view? Answer with just the letter.",
            f"Identify the correct {view_name}-view projection of the 3D object shown. Answer with one of A, B, C, D.",
        ]
        q = sub_rng.choice(stems)
        image = self._render_with_options(sub_rng, grid, shape,
                                          shuffled_options, view_name)
        return q, correct_label, image

    def _generate_l0_flat(self, sub_rng, cfg):
        """L0: show a simple 2D grid and ask how many cubes. Avoids 3D isometric."""
        shape = sub_rng.choice(cfg["shapes"])
        grid = self._make_shape(sub_rng, shape)
        if grid is None:
            return None
        total = int(np.sum(grid))
        if total < 1:
            return None

        # Get top-view projection
        top_proj = _project_2d(grid, "top")

        # MCQ: how many cubes?
        opts = [total]
        for d in [1, -1, 2, -2]:
            v = total + d
            if v >= 1 and v not in opts:
                opts.append(v)
        while len(opts) < 4:
            opts.append(total + len(opts))
        opts = opts[:4]
        sub_rng.shuffle(opts)
        correct_letter = chr(65 + opts.index(total))
        opt_str = ", ".join(f"{chr(65 + i)}) {v}" for i, v in enumerate(opts))

        stems = [
            "The image shows a top-down view of a structure built from unit cubes. Each dark cell is one cube. How many cubes are there?",
            "Count the dark squares in this bird's-eye view. Each represents one unit cube. How many total?",
        ]
        q = f"{sub_rng.choice(stems)} Options: {opt_str}. Reply with the single letter (A/B/C/D) inside <answer>...</answer>. Example: <answer>B</answer>"

        style = self._random_style()
        sc = style["figsize_scale"]
        h, w = top_proj.shape
        fig, ax = plt.subplots(figsize=(max(4, w + 1) * sc, max(4, h + 1) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        for r in range(h):
            for c in range(w):
                color = "#34495E" if top_proj[r, c] else "#EAECEE"
                rect = mpatches.Rectangle(
                    (c, h - 1 - r), 1, 1,
                    facecolor=color, edgecolor="black", linewidth=2)
                ax.add_patch(rect)
        ax.set_xlim(-0.2, w + 0.2)
        ax.set_ylim(-0.2, h + 0.2)
        ax.set_title("Top View", fontsize=14, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        image = self.fig_to_pil(fig, dpi=style["dpi"])
        self._primary_complexity_feature = total
        return q, correct_letter, image

    def _make_shape(self, rng, shape):
        if shape == "single":
            g = np.zeros((2, 2, 2), dtype=int)
            g[0, 0, 0] = 1
            return g
        if shape == "line2":
            g = np.zeros((3, 2, 2), dtype=int)
            g[0, 0, 0] = 1
            g[1, 0, 0] = 1
            return g
        if shape == "line3":
            g = np.zeros((4, 2, 2), dtype=int)
            for i in range(3):
                g[i, 0, 0] = 1
            return g
        if shape == "l_flat":
            g = np.zeros((3, 3, 2), dtype=int)
            for i in range(3):
                g[i, 0, 0] = 1
            g[0, 1, 0] = 1
            g[0, 2, 0] = 1
            return g
        if shape == "l_shape":
            g = np.zeros((3, 3, 3), dtype=int)
            for k in range(3):
                g[0, 0, k] = 1
            for i in range(3):
                g[i, 0, 0] = 1
            return g
        if shape == "t_shape":
            g = np.zeros((3, 3, 2), dtype=int)
            for i in range(3):
                g[i, 1, 0] = 1
            g[1, 0, 0] = 1
            g[1, 2, 0] = 1
            g[1, 1, 1] = 1
            return g
        if shape == "stairs":
            g = np.zeros((3, 3, 3), dtype=int)
            for i in range(3):
                for j in range(3):
                    for k in range(i + 1):
                        g[i, j, k] = 1
            return g
        if shape == "u_shape":
            g = np.zeros((3, 3, 2), dtype=int)
            for i in range(3):
                g[i, 0, 0] = 1
                g[i, 2, 0] = 1
            g[0, 0, 1] = 1
            g[0, 2, 1] = 1
            g[0, 1, 0] = 1
            return g
        if shape == "plus":
            g = np.zeros((3, 3, 2), dtype=int)
            g[1, :, 0] = 1
            g[:, 1, 0] = 1
            g[1, 1, 1] = 1
            return g
        if shape == "tower":
            g = np.zeros((3, 3, 4), dtype=int)
            for i in range(3):
                for j in range(3):
                    g[i, j, 0] = 1
            g[1, 1, 1] = 1
            g[1, 1, 2] = 1
            g[1, 1, 3] = 1
            return g
        g = np.zeros((2, 2, 2), dtype=int)
        g[0, 0, 0] = 1
        return g

    def _draw_iso(self, ax, grid):
        gx, gy, gz = grid.shape
        cubes = [(i, j, k) for i in range(gx) for j in range(gy)
                 for k in range(gz) if grid[i, j, k]]
        cubes.sort(key=lambda c: -(c[0] + c[1] - c[2]))
        top_c = '#5DADE2'
        left_c = '#2E86C1'
        right_c = '#1B4F72'
        for (x, y, z) in cubes:
            pts = [_iso(x, y, z + 1), _iso(x + 1, y, z + 1),
                   _iso(x + 1, y + 1, z + 1), _iso(x, y + 1, z + 1)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=top_c,
                                 edgecolor='black', lw=1))
            pts = [_iso(x, y, z), _iso(x, y + 1, z),
                   _iso(x, y + 1, z + 1), _iso(x, y, z + 1)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=left_c,
                                 edgecolor='black', lw=1))
            pts = [_iso(x, y, z), _iso(x + 1, y, z),
                   _iso(x + 1, y, z + 1), _iso(x, y, z + 1)]
            ax.add_patch(Polygon(pts, closed=True, facecolor=right_c,
                                 edgecolor='black', lw=1))

    def _draw_2d_grid(self, ax, proj, title=""):
        h, w = proj.shape
        for r in range(h):
            for c in range(w):
                color = "#34495E" if proj[r, c] else "#EAECEE"
                rect = mpatches.Rectangle(
                    (c, h - 1 - r), 1, 1,
                    facecolor=color, edgecolor="black", linewidth=1)
                ax.add_patch(rect)
        ax.set_xlim(-0.1, w + 0.1)
        ax.set_ylim(-0.1, h + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold")

    def _render_iso_only(self, rng, grid, shape):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        self._draw_iso(ax, grid)
        gx, gy, gz = grid.shape
        pts = [_iso(x, y, z) for x in range(gx + 1)
               for y in range(gy + 1) for z in range(gz + 1)]
        pts = np.array(pts)
        m = 0.5
        ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
        ax.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)
        ax.set_title(f"3D Structure ({shape.replace('_', ' ').title()})",
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_with_options(self, rng, grid, shape, options, view_name):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(14 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        ax_iso = fig.add_subplot(1, 5, (1, 2))
        ax_iso.set_aspect("equal")
        ax_iso.axis("off")
        self._draw_iso(ax_iso, grid)
        gx, gy, gz = grid.shape
        pts = [_iso(x, y, z) for x in range(gx + 1) for y in range(gy + 1)
               for z in range(gz + 1)]
        pts = np.array(pts)
        m = 0.5
        ax_iso.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
        ax_iso.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)
        ax_iso.set_title("3D View", fontsize=12, fontweight="bold")

        labels = ["A", "B", "C", "D"]
        for idx, (opt, label) in enumerate(zip(options, labels)):
            ax_opt = fig.add_subplot(2, 5, 3 + idx if idx < 3 else 8 + idx - 3)
            self._draw_2d_grid(ax_opt, opt, title=f"Option {label}")

        fig.suptitle(f"Which is the {view_name} view?",
                     fontsize=style["font_size_base"] + 4,
                     fontweight="bold", y=1.02)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = ProjectionViewQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
