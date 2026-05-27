"""
Visual Difference QA environment.

Generates two charts/figures side by side, nearly identical with 1-3
subtle differences. Tests precise visual comparison ability.

Diversity & difficulty redesign (2026-04-16):
- All random choices use a level-aware sub_rng.
- L0: only 2-bar / 3-bar side-by-side with LARGE differences (easier to spot).
- L9: 5-7 bars with small deltas, composite questions (identify AND count).
- Added category pools, more bar colors, per-seed color shuffle.
"""
import math
import random
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_BAR_COLORS = ['#3498db', '#e74c3c', '#27ae60', '#f39c12',
               '#9b59b6', '#1abc9c', '#e67e22', '#8e44ad',
               '#16a085', '#c0392b', '#2c3e50']

_CATEGORIES = [
    ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
    ['Math', 'Science', 'English', 'History', 'Art', 'Music'],
    ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry', 'Fig'],
    ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
    ['North', 'South', 'East', 'West', 'Central'],
    ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta'],
]

_BAR_TITLES = [
    ('Figure A', 'Figure B'),
    ('Original', 'Modified'),
    ('Before', 'After'),
    ('Left', 'Right'),
    ('Chart 1', 'Chart 2'),
]

class VisualDifferenceQA(StandaloneVisualEnv):
    ENV_NAME = "visual_difference"

    QUESTION_TYPES = [
        "count_differences", "which_bar_changed", "what_value_changed",
        "which_element_added", "which_element_removed",
        "identify_changed_category", "total_value_change",
        "count_shape_differences",
    ]

    _SHAPE_POOL = ["circle", "square", "triangle", "hexagon", "diamond", "star"]
    _SHAPE_COLOR_POOL = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#f1c40f", "#e91e63",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return {"qtypes": ["count_differences", "which_bar_changed"],
                    "n_bars": (3, 4), "n_scatter": (5, 6),
                    "delta_range": (10, 20), "max_changes": 2}
        if level <= 2:
            return {"qtypes": ["count_differences", "which_bar_changed",
                               "what_value_changed"],
                    "n_bars": (3, 5), "n_scatter": (5, 7),
                    "delta_range": (8, 18), "max_changes": 2}
        if level <= 4:
            return {"qtypes": ["which_bar_changed", "which_element_added",
                               "which_element_removed",
                               "count_differences"],
                    "n_bars": (4, 6), "n_scatter": (6, 9),
                    "delta_range": (6, 15), "max_changes": 3}
        if level <= 6:
            return {"qtypes": ["which_element_added", "which_element_removed",
                               "identify_changed_category",
                               "total_value_change",
                               "count_shape_differences"],
                    "n_bars": (5, 6), "n_scatter": (7, 10),
                    "delta_range": (5, 12), "max_changes": 3,
                    "shape_n_objects": 12, "shape_n_diff_base": 4,
                    "shape_change_types": ["remove", "color", "size", "position"]}
        if level <= 7:
            return {"qtypes": ["identify_changed_category",
                               "total_value_change",
                               "which_element_removed",
                               "count_shape_differences"],
                    "n_bars": (5, 7), "n_scatter": (7, 11),
                    "delta_range": (4, 10), "max_changes": 3,
                    "shape_n_objects": 14, "shape_n_diff_base": 5,
                    "shape_change_types": ["remove", "color", "size", "position"]}
        return {"qtypes": self.QUESTION_TYPES,
                "n_bars": (6, 7), "n_scatter": (8, 12),
                "delta_range": (3, 8), "max_changes": 3,
                "shape_n_objects": 17, "shape_n_diff_base": 6,
                "shape_change_types": ["color", "size", "position", "shade"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(25):
            result = self._try_generate(qtype, sub_rng, cfg, level)
            if result is not None:
                return result
            qtype = sub_rng.choice(cfg["qtypes"])
        return None

    # -------------------------------------------------------------- #
    def _gen_bar_data(self, rng, cfg):
        """Generate bar chart data with level-aware n."""
        cats_pool = rng.choice(_CATEGORIES)
        n_lo, n_hi = cfg["n_bars"]
        n = min(rng.randint(n_lo, n_hi), len(cats_pool))
        cats = cats_pool[:n]
        values = [rng.randint(5, 50) for _ in range(n)]
        colors = list(_BAR_COLORS)
        rng.shuffle(colors)
        return cats, values, colors[:n]

    def _gen_scatter_data(self, rng, cfg, n_points=None):
        """Generate scatter plot data."""
        if n_points is None:
            lo, hi = cfg["n_scatter"]
            n_points = rng.randint(lo, hi)
        xs = [rng.uniform(1, 10) for _ in range(n_points)]
        ys = [rng.uniform(1, 10) for _ in range(n_points)]
        labels = [chr(65 + i) for i in range(n_points)]
        return xs, ys, labels

    # -------------------------------------------------------------- #
    def _try_generate(self, qtype, rng, cfg, level):
        if qtype == "count_differences":
            return self._gen_count_differences(rng, cfg)
        elif qtype == "which_bar_changed":
            return self._gen_which_bar_changed(rng, cfg)
        elif qtype == "what_value_changed":
            return self._gen_what_value_changed(rng, cfg)
        elif qtype == "which_element_added":
            return self._gen_element_added(rng, cfg)
        elif qtype == "which_element_removed":
            return self._gen_element_removed(rng, cfg)
        elif qtype == "identify_changed_category":
            return self._gen_identify_changed_category(rng, cfg)
        elif qtype == "total_value_change":
            return self._gen_total_value_change(rng, cfg)
        elif qtype == "count_shape_differences":
            return self._gen_count_shape_differences(rng, cfg)
        return None

    def _gen_count_differences(self, rng, cfg):
        cats, values_orig, colors = self._gen_bar_data(rng, cfg)
        n = len(cats)
        max_changes = cfg.get("max_changes", 3)
        num_changes = rng.randint(1, min(max_changes, n))
        changed_indices = rng.sample(range(n), num_changes)
        d_lo, d_hi = cfg["delta_range"]

        values_mod = list(values_orig)
        for idx in changed_indices:
            delta = rng.choice([-1, 1]) * rng.randint(d_lo, d_hi)
            values_mod[idx] = max(1, values_orig[idx] + delta)

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_bar(
            cats, values_orig, values_mod, colors, ta, tb)
        q = (f"How many bars have different values between {ta} (left) "
             f"and {tb} (right)?")
        return q, str(num_changes), img

    def _gen_which_bar_changed(self, rng, cfg):
        cats, values_orig, colors = self._gen_bar_data(rng, cfg)
        n = len(cats)
        changed_idx = rng.randint(0, n - 1)
        d_lo, d_hi = cfg["delta_range"]

        values_mod = list(values_orig)
        delta = rng.choice([-1, 1]) * rng.randint(d_lo, d_hi)
        values_mod[changed_idx] = max(1, values_orig[changed_idx] + delta)

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_bar(
            cats, values_orig, values_mod, colors, ta, tb)
        q = (f"Which bar changed its value between {ta} (left) and "
             f"{tb} (right)? Answer with the category name.")
        return q, cats[changed_idx], img

    def _gen_what_value_changed(self, rng, cfg):
        cats, values_orig, colors = self._gen_bar_data(rng, cfg)
        n = len(cats)
        changed_idx = rng.randint(0, n - 1)
        d_lo, d_hi = cfg["delta_range"]

        values_mod = list(values_orig)
        delta = rng.choice([-1, 1]) * rng.randint(d_lo, d_hi)
        new_val = max(1, values_orig[changed_idx] + delta)
        values_mod[changed_idx] = new_val

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_bar(
            cats, values_orig, values_mod, colors, ta, tb)
        q = (f"One bar changed between {ta} (left) and {tb} (right). "
             f"Which bar changed? Answer with the category name.")
        return q, cats[changed_idx], img

    def _gen_identify_changed_category(self, rng, cfg):
        cats, values_orig, colors = self._gen_bar_data(rng, cfg)
        n = len(cats)
        max_changes = cfg.get("max_changes", 3)
        num_changes = rng.randint(2, min(max_changes, n))
        changed_indices = rng.sample(range(n), num_changes)
        d_lo, d_hi = cfg["delta_range"]

        values_mod = list(values_orig)
        for idx in changed_indices:
            delta = rng.choice([-1, 1]) * rng.randint(d_lo, d_hi)
            values_mod[idx] = max(1, values_orig[idx] + delta)

        changed_cats = sorted([cats[i] for i in changed_indices])
        answer = ", ".join(changed_cats)

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_bar(
            cats, values_orig, values_mod, colors, ta, tb)
        q = (f"Which categories have different values between {ta} (left) "
             f"and {tb} (right)? List changed categories, comma-separated, "
             f"in alphabetical order.")
        return q, answer, img

    def _gen_total_value_change(self, rng, cfg):
        cats, values_orig, colors = self._gen_bar_data(rng, cfg)
        n = len(cats)
        max_changes = cfg.get("max_changes", 3)
        num_changes = rng.randint(1, min(max_changes, n))
        changed_indices = rng.sample(range(n), num_changes)
        d_lo, d_hi = cfg["delta_range"]

        values_mod = list(values_orig)
        for idx in changed_indices:
            delta = rng.choice([-1, 1]) * rng.randint(d_lo, d_hi)
            values_mod[idx] = max(1, values_orig[idx] + delta)

        total_change = sum(values_mod) - sum(values_orig)

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_bar(
            cats, values_orig, values_mod, colors, ta, tb)
        q = (f"What is the net change in the total sum of all bar values "
             f"from {ta} (left) to {tb} (right)? Positive or negative number.")
        return q, str(total_change), img

    def _gen_element_added(self, rng, cfg):
        xs, ys, labels = self._gen_scatter_data(rng, cfg)
        n = len(xs)
        new_x = rng.uniform(1, 10)
        new_y = rng.uniform(1, 10)
        new_label = chr(65 + n)

        xs_mod = list(xs) + [new_x]
        ys_mod = list(ys) + [new_y]
        labels_mod = list(labels) + [new_label]

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_scatter(
            xs, ys, labels, xs_mod, ys_mod, labels_mod, ta, tb)
        q = (f"A new point was added in {tb} (right) that was not in "
             f"{ta} (left). What is the label of the added point?")
        return q, new_label, img

    def _gen_element_removed(self, rng, cfg):
        xs, ys, labels = self._gen_scatter_data(rng, cfg)
        n = len(xs)
        removed_idx = rng.randint(0, n - 1)
        removed_label = labels[removed_idx]

        xs_mod = [x for i, x in enumerate(xs) if i != removed_idx]
        ys_mod = [y for i, y in enumerate(ys) if i != removed_idx]
        labels_mod = [l for i, l in enumerate(labels) if i != removed_idx]

        ta, tb = rng.choice(_BAR_TITLES)
        img = self._render_side_by_side_scatter(
            xs, ys, labels, xs_mod, ys_mod, labels_mod, ta, tb)
        q = (f"One point was removed in {tb} (right) that was present in "
             f"{ta} (left). What is the label of the removed point?")
        return q, removed_label, img

    # -------------------------------------------------------------- #
    # Shape-scene count differences (merged from visual_difference_count_hard)
    # -------------------------------------------------------------- #

    _SHAPE_Q_TEMPLATES = [
        "Two complex scenes are shown side by side. Image 1 (left) is the original; Image 2 (right) has been modified. Count the number of differences between the two images (each modified object counts as one difference). Answer with a single integer.",
        "Compare the two scenes. The right panel has been altered from the left. How many objects differ between Image 1 and Image 2? Reply with just the integer count.",
        "The left scene was modified to produce the right scene. Count how many individual objects changed (removed, recolored, resized, or repositioned). Answer with a single number.",
        "Look at the pair of scenes side by side. How many objects are different between the original (left) and the modified version (right)? Give an integer answer.",
        "Two similar scenes are displayed. A number of objects in the right panel differ from the left panel. Count the total number of differences and answer with one integer.",
        "Examine Image 1 (left) and Image 2 (right). Determine the total number of objects that are different between the two scenes. Respond with an integer.",
        "Inspect both panels carefully. Each modified object (changed color, size, position, or presence) counts as one difference. How many differences exist? Single integer.",
        "Spot the differences between the left and right scenes. Count every changed object and give the total as a single integer.",
        "The right image is a modified copy of the left one. Count the number of altered objects. Answer with one integer.",
        "How many objects are visibly different between the two panels (left=original, right=modified)? Write the integer count.",
        "Two side-by-side images differ in several objects. Count the differences between the original and the modified scene. Answer is a single integer.",
        "Find and count all object-level differences between the two pictures. Respond with an integer.",
        "Carefully compare the pair of scenes. For each object that changed (removed/recolored/resized/moved), add one to the count. What is the total? Answer with one integer.",
        "A number of objects differ between the left (original) and the right (modified) scene. Count them. Output a single integer.",
        "The two scenes on this page share most objects but a few are different. How many? Answer as an integer.",
        "Count the distinct object differences between the two scenes shown. Reply with a single integer.",
    ]

    def _draw_shape_obj(self, ax, obj: Dict):
        if obj["size"] <= 0:
            return
        cx, cy, size = obj["x"], obj["y"], obj["size"]
        color = obj["color"]
        shape = obj["shape"]
        if shape == "circle":
            ax.add_patch(plt.Circle((cx, cy), size, fc=color, ec="black",
                                    lw=0.8, alpha=0.88))
        elif shape == "square":
            ax.add_patch(mpatches.Rectangle((cx - size, cy - size),
                                            2 * size, 2 * size,
                                            fc=color, ec="black", lw=0.8,
                                            alpha=0.88))
        elif shape == "triangle":
            ax.add_patch(RegularPolygon((cx, cy), 3, radius=size * 1.15,
                                        orientation=math.pi / 2,
                                        fc=color, ec="black", lw=0.8,
                                        alpha=0.88))
        elif shape == "hexagon":
            ax.add_patch(RegularPolygon((cx, cy), 6, radius=size * 1.1,
                                        fc=color, ec="black", lw=0.8,
                                        alpha=0.88))
        elif shape == "diamond":
            pts = [(cx, cy + size), (cx + size * 0.7, cy),
                   (cx, cy - size), (cx - size * 0.7, cy)]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=0.8,
                                     alpha=0.88))
        elif shape == "star":
            pts = []
            for i in range(10):
                a = math.radians(90 + i * 36)
                r = size if i % 2 == 0 else size * 0.45
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=0.8,
                                     alpha=0.88))
        if obj.get("label") is not None:
            ax.text(cx, cy, obj["label"], ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)

    def _render_shape_panel(self, ax, objects, title):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(mpatches.Rectangle((0.1, 0.1), 9.8, 9.8,
                                        fc="#f8f9fa", ec="#34495e",
                                        lw=1.2, zorder=0))
        for t in range(1, 10):
            ax.plot([t, t], [0.2, 9.8], color="#dfe4ea", lw=0.6, zorder=0.5)
            ax.plot([0.2, 9.8], [t, t], color="#dfe4ea", lw=0.6, zorder=0.5)
        for obj in objects:
            self._draw_shape_obj(ax, obj)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=4)

    def _gen_count_shape_differences(self, rng, cfg):
        n_obj = cfg.get("shape_n_objects", 12)
        n_diff_base = cfg.get("shape_n_diff_base", 4)
        change_types = cfg.get("shape_change_types",
                               ["remove", "color", "size", "position"])
        add_labels = cfg.get("shape_n_objects", 12) >= 12
        n_diff = max(1, n_diff_base + rng.randint(-1, 1))

        base_objects = []
        for i in range(n_obj):
            base_objects.append({
                "id": i,
                "shape": rng.choice(self._SHAPE_POOL),
                "color": rng.choice(self._SHAPE_COLOR_POOL),
                "x": rng.uniform(1.0, 9.0),
                "y": rng.uniform(1.0, 9.0),
                "size": rng.uniform(0.3, 0.55),
                "label": chr(ord("A") + i) if add_labels else None,
            })

        modified = [dict(o) for o in base_objects]
        diff_indices = rng.sample(range(n_obj), min(n_diff, n_obj))
        for idx in diff_indices:
            change = rng.choice(change_types)
            if change == "remove":
                modified[idx]["size"] = 0
            elif change == "color":
                other = [c for c in self._SHAPE_COLOR_POOL
                         if c != modified[idx]["color"]]
                modified[idx]["color"] = rng.choice(other)
            elif change == "size":
                modified[idx]["size"] *= rng.choice([0.5, 0.7, 1.4, 1.7])
                modified[idx]["size"] = max(0.15, min(0.9, modified[idx]["size"]))
            elif change == "position":
                mag = 0.8
                modified[idx]["x"] = max(0.5, min(9.5,
                                         modified[idx]["x"]
                                         + rng.uniform(-mag, mag)))
                modified[idx]["y"] = max(0.5, min(9.5,
                                         modified[idx]["y"]
                                         + rng.uniform(-mag, mag)))
            elif change == "shade":
                from matplotlib.colors import to_rgba, rgb2hex
                r, g, b, _ = to_rgba(modified[idx]["color"])
                shade_delta = 0.18
                r = max(0, min(1, r + rng.uniform(-shade_delta, shade_delta)))
                g = max(0, min(1, g + rng.uniform(-shade_delta, shade_delta)))
                b = max(0, min(1, b + rng.uniform(-shade_delta, shade_delta)))
                modified[idx]["color"] = rgb2hex((r, g, b))

        style = self._random_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.patch.set_facecolor(style["bg_color"])
        self._render_shape_panel(ax1, base_objects, "Image 1 (Original)")
        self._render_shape_panel(ax2, modified, "Image 2 (Modified)")
        fig.suptitle("Count the differences",
                     fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.04,
                            wspace=0.12)
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        sidx = (self.seed or 0) % len(self._SHAPE_Q_TEMPLATES)
        q = self._SHAPE_Q_TEMPLATES[sidx]
        return q, str(n_diff), img

    # -------------------------------------------------------------- #
    # Rendering: bar charts
    # -------------------------------------------------------------- #

    def _render_side_by_side_bar(self, cats, vals_a, vals_b, colors,
                                  title_a, title_b):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        fs = style["font_size_base"]
        bar_colors = [palette[i % len(palette)] for i in range(len(cats))]

        max_val = max(max(vals_a), max(vals_b)) + 5

        for ax, vals, title in [(ax1, vals_a, title_a), (ax2, vals_b, title_b)]:
            ax.set_facecolor(style["bg_color"])
            x_pos = range(len(cats))
            bars = ax.bar(x_pos, vals, color=bar_colors,
                          edgecolor='black', linewidth=1.2)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(cats, fontsize=fs, rotation=30, ha='right')
            ax.set_ylim(0, max_val)
            ax.set_title(title, fontsize=fs + 4, fontweight='bold')
            ax.set_ylabel('Value', fontsize=fs + 1)
            ax.grid(axis='y', alpha=0.3)

            # NO value labels — force visual comparison of bar heights

        fig.suptitle('Spot the Difference', fontsize=fs + 6, fontweight='bold', y=1.02)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # -------------------------------------------------------------- #
    # Rendering: scatter plots
    # -------------------------------------------------------------- #

    def _render_side_by_side_scatter(self, xs_a, ys_a, labels_a,
                                      xs_b, ys_b, labels_b,
                                      title_a, title_b):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        all_x = list(xs_a) + list(xs_b)
        all_y = list(ys_a) + list(ys_b)
        x_lim = (min(all_x) - 1, max(all_x) + 1)
        y_lim = (min(all_y) - 1, max(all_y) + 1)

        palette = style["palette"]
        fs = style["font_size_base"]
        for ax, xs, ys, labels, title in [
            (ax1, xs_a, ys_a, labels_a, title_a),
            (ax2, xs_b, ys_b, labels_b, title_b),
        ]:
            ax.set_facecolor(style["bg_color"])
            for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
                color = palette[i % len(palette)]
                ax.plot(x, y, 'o', color=color, markersize=10,
                        markeredgecolor='black', markeredgewidth=1.5, zorder=3)
                ax.text(x + 0.15, y + 0.15, label, fontsize=fs,
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.8,
                                  edgecolor=color),
                        zorder=4)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_title(title, fontsize=fs + 4, fontweight='bold')
            ax.set_xlabel('X', fontsize=fs + 1)
            ax.set_ylabel('Y', fontsize=fs + 1)
            ax.grid(alpha=0.3)

        fig.suptitle('Spot the Difference', fontsize=fs + 6, fontweight='bold', y=1.02)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
