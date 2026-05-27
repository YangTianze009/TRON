"""
Precise Counting QA environment (redesigned 2026-04-16).

Dense scene with small objects (dots, shapes) in various colors.
Tests fine-grained visual counting ability.

Critical fix (vs Grade D baseline):
  * Title NO LONGER leaks the object count. Old version put
    "(45 objects)" in the title, which gave the answer away.
  * Now: title is a neutral label like "Objects" / "Scene" /
    "Counting" — the image alone encodes the count.
  * Expanded to 7 shape types and 12 colors
  * L0 has well-spaced circles only; L9 has 5 shapes and 8 colors
  * Randomized layout strategies: uniform, clustered, gridded,
    corner-heavy, border-heavy
  * 6+ paraphrased question stems per question type

Level design (parameter["level"], 0-9)

L0: 3-5 large, well-spaced objects, one shape, one color. `count_all`.
L1: 4-6 objects, 1 shape, 2 colors. `count_all`.
L2: 5-8 objects, 2 shapes, 2 colors. `count_all` or `count_by_color`.
L3: 7-11 objects, 2 shapes, 3 colors. `count_all` / `color` / `shape`.
L4: 10-15 objects, 3 shapes, 3 colors. Mix, introduces quadrant.
L5: 14-20 objects, 3 shapes, 4 colors. All question types.
L6: 18-28 objects, 4 shapes, 5 colors. All types.
L7: 22-34 objects, 4 shapes, 5 colors. All types + small objects.
L8: 28-42 objects, 5 shapes, 6 colors. All types + smallest objects.
L9: 34-50 objects, 5 shapes, 7 colors. Full complexity.
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

_COLORS = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#27ae60",
    "orange": "#e67e22",
    "purple": "#8e44ad",
    "yellow": "#f1c40f",
    "pink": "#e91e63",
    "cyan": "#00bcd4",
    "brown": "#8d6e63",
    "teal": "#009688",
    "lime": "#a4c639",
    "gray": "#7f8c8d",
}

_SHAPES = ["circle", "square", "triangle", "diamond", "star",
           "pentagon", "hexagon"]

_BORDER_MARGIN = 3.0

# Neutral titles — do NOT mention the object count!
_TITLE_VARIANTS = [
    "Scene",
    "Objects",
    "Shape Layout",
    "Shapes",
    "Visual Scene",
    "Counting",
    "Observation",
    "Image",
    "Figure",
    "Arrangement",
    "Pattern",
    "Collection",
    "Display",
    "View",
]

def _make_triangle_path(cx, cy, size):
    h = size * math.sqrt(3) / 2
    verts = [
        (cx, cy + h * 0.67),
        (cx - size / 2, cy - h * 0.33),
        (cx + size / 2, cy - h * 0.33),
        (cx, cy + h * 0.67),
    ]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO,
             mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _make_diamond_path(cx, cy, size):
    verts = [
        (cx, cy + size * 0.7),
        (cx - size * 0.5, cy),
        (cx, cy - size * 0.7),
        (cx + size * 0.5, cy),
        (cx, cy + size * 0.7),
    ]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO,
             mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _make_star_path(cx, cy, size):
    verts = []
    for i in range(5):
        angle_out = math.pi / 2 + i * 2 * math.pi / 5
        verts.append((cx + size * 0.6 * math.cos(angle_out),
                      cy + size * 0.6 * math.sin(angle_out)))
        angle_in = angle_out + math.pi / 5
        verts.append((cx + size * 0.25 * math.cos(angle_in),
                      cy + size * 0.25 * math.sin(angle_in)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 9 + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _make_pentagon_path(cx, cy, size):
    verts = []
    for i in range(5):
        a = math.pi / 2 + i * 2 * math.pi / 5
        verts.append((cx + size * 0.7 * math.cos(a),
                      cy + size * 0.7 * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 4 + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _make_hexagon_path(cx, cy, size):
    verts = []
    for i in range(6):
        a = i * math.pi / 3
        verts.append((cx + size * 0.75 * math.cos(a),
                      cy + size * 0.75 * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 5 + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

# ---------------------------------------------------------------------
# Layout strategies (for visual diversity across seeds)
# ---------------------------------------------------------------------

def _place_uniform(rng, n, canvas_size, obj_size, min_sep):
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 40:
        x = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
        y = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
        if all(math.hypot(x - ox, y - oy) >= min_sep for ox, oy in pts):
            pts.append((x, y))
        tries += 1
    return pts

def _place_clustered(rng, n, canvas_size, obj_size, min_sep):
    """Two or three cluster centers, points fall around them."""
    pts = []
    n_clusters = rng.randint(2, 4)
    centers = [(rng.uniform(canvas_size * 0.2, canvas_size * 0.8),
                rng.uniform(canvas_size * 0.2, canvas_size * 0.8))
               for _ in range(n_clusters)]
    tries = 0
    while len(pts) < n and tries < n * 60:
        cx, cy = rng.choice(centers)
        sigma = canvas_size * 0.1
        x = cx + rng.gauss(0, sigma)
        y = cy + rng.gauss(0, sigma)
        if not (obj_size + 1 <= x <= canvas_size - obj_size - 1 and
                obj_size + 1 <= y <= canvas_size - obj_size - 1):
            tries += 1
            continue
        if all(math.hypot(x - ox, y - oy) >= min_sep for ox, oy in pts):
            pts.append((x, y))
        tries += 1
    # Fill remainder uniformly if needed
    if len(pts) < n:
        pts.extend(_place_uniform(
            rng, n - len(pts), canvas_size, obj_size, min_sep))
    return pts[:n]

def _place_grid_jitter(rng, n, canvas_size, obj_size, min_sep):
    """Gridded layout with jitter."""
    cols = max(2, int(math.ceil(math.sqrt(n * 1.3))))
    rows = max(2, int(math.ceil(n / cols)))
    step_x = (canvas_size - 2 * (obj_size + 1)) / max(1, cols - 1) \
        if cols > 1 else 0
    step_y = (canvas_size - 2 * (obj_size + 1)) / max(1, rows - 1) \
        if rows > 1 else 0
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    pts = []
    for c, r in cells[:n]:
        cx = obj_size + 1 + c * step_x if cols > 1 else canvas_size / 2
        cy = obj_size + 1 + r * step_y if rows > 1 else canvas_size / 2
        jitter = min(step_x, step_y) * 0.22 if step_x and step_y else 3
        x = cx + rng.uniform(-jitter, jitter)
        y = cy + rng.uniform(-jitter, jitter)
        x = max(obj_size + 1, min(canvas_size - obj_size - 1, x))
        y = max(obj_size + 1, min(canvas_size - obj_size - 1, y))
        if all(math.hypot(x - ox, y - oy) >= min_sep * 0.8 for ox, oy in pts):
            pts.append((x, y))
    if len(pts) < n:
        pts.extend(_place_uniform(
            rng, n - len(pts), canvas_size, obj_size, min_sep))
    return pts[:n]

def _place_border_heavy(rng, n, canvas_size, obj_size, min_sep):
    """Many points near the border — useful for `count_touching_border`."""
    pts = []
    border_frac = 0.55
    tries = 0
    while len(pts) < n and tries < n * 50:
        if rng.random() < border_frac:
            edge = rng.choice(["top", "bottom", "left", "right"])
            if edge == "top":
                x = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
                y = rng.uniform(canvas_size - obj_size - 1 - _BORDER_MARGIN,
                                canvas_size - obj_size - 1)
            elif edge == "bottom":
                x = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
                y = rng.uniform(obj_size + 1,
                                obj_size + 1 + _BORDER_MARGIN)
            elif edge == "left":
                x = rng.uniform(obj_size + 1,
                                obj_size + 1 + _BORDER_MARGIN)
                y = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
            else:
                x = rng.uniform(canvas_size - obj_size - 1 - _BORDER_MARGIN,
                                canvas_size - obj_size - 1)
                y = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
        else:
            x = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
            y = rng.uniform(obj_size + 1, canvas_size - obj_size - 1)
        if all(math.hypot(x - ox, y - oy) >= min_sep for ox, oy in pts):
            pts.append((x, y))
        tries += 1
    if len(pts) < n:
        pts.extend(_place_uniform(
            rng, n - len(pts), canvas_size, obj_size, min_sep))
    return pts[:n]

def _place_quadrant_biased(rng, n, canvas_size, obj_size, min_sep):
    """Bias toward one or two quadrants."""
    quads = [(0, canvas_size / 2, 0, canvas_size / 2),             # bl
             (canvas_size / 2, canvas_size, 0, canvas_size / 2),   # br
             (0, canvas_size / 2, canvas_size / 2, canvas_size),   # tl
             (canvas_size / 2, canvas_size,
              canvas_size / 2, canvas_size)]                       # tr
    primary = rng.choice(quads)
    secondary = rng.choice(quads)
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 50:
        q = primary if rng.random() < 0.65 else secondary
        x = rng.uniform(max(obj_size + 1, q[0]),
                        min(canvas_size - obj_size - 1, q[1]))
        y = rng.uniform(max(obj_size + 1, q[2]),
                        min(canvas_size - obj_size - 1, q[3]))
        if all(math.hypot(x - ox, y - oy) >= min_sep for ox, oy in pts):
            pts.append((x, y))
        tries += 1
    if len(pts) < n:
        pts.extend(_place_uniform(
            rng, n - len(pts), canvas_size, obj_size, min_sep))
    return pts[:n]

_LAYOUT_STRATEGIES = {
    "uniform": _place_uniform,
    "clustered": _place_clustered,
    "grid": _place_grid_jitter,
    "border": _place_border_heavy,
    "quadrant": _place_quadrant_biased,
}

class PreciseCountingQA(StandaloneVisualEnv):
    ENV_NAME = "precise_counting"

    QUESTION_TYPES = [
        "count_all", "count_by_color", "count_by_shape",
        "count_in_quadrant", "count_touching_border",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"],
                                      weights=cfg["qtype_weights"])[0]

        for attempt in range(10):
            result = self._try_generate(qtype, level, cfg, attempt)
            if result is not None:
                self._primary_complexity_feature = level * 5 + len(result[1])
                # 2026-05-04: simplified L0 (was 10% too-hard) — add concise
                # hint so model doesn't run out of token budget on CoT.
                if level <= 1:
                    q, ans, img = result
                    q = q + " Be concise. Output only the integer."
                    result = (q, ans, img)
                return result
        return None

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            return {
                "qtypes": ["count_all"], "qtype_weights": [1],
                "min_obj": 3, "max_obj": 5,
                "n_colors": 1, "n_shapes": 1,
                "obj_size_range": (3.5, 4.5),
                "min_sep": 12,
                "layouts": ["uniform", "grid"],
            }
        if level == 1:
            return {
                "qtypes": ["count_all"], "qtype_weights": [1],
                "min_obj": 4, "max_obj": 6,
                "n_colors": 2, "n_shapes": 1,
                "obj_size_range": (3.0, 4.0),
                "min_sep": 10,
                "layouts": ["uniform", "grid", "clustered"],
            }
        if level == 2:
            return {
                "qtypes": ["count_all", "count_by_color"],
                "qtype_weights": [6, 4],
                "min_obj": 5, "max_obj": 8,
                "n_colors": 2, "n_shapes": 2,
                "obj_size_range": (2.8, 3.8),
                "min_sep": 9,
                "layouts": ["uniform", "grid", "clustered"],
            }
        if level == 3:
            return {
                "qtypes": ["count_all", "count_by_color", "count_by_shape"],
                "qtype_weights": [5, 3, 2],
                "min_obj": 7, "max_obj": 11,
                "n_colors": 3, "n_shapes": 2,
                "obj_size_range": (2.5, 3.5),
                "min_sep": 7,
                "layouts": ["uniform", "grid", "clustered", "quadrant"],
            }
        if level == 4:
            return {
                "qtypes": ["count_all", "count_by_color", "count_by_shape",
                           "count_in_quadrant"],
                "qtype_weights": [3, 3, 2, 2],
                "min_obj": 10, "max_obj": 15,
                "n_colors": 3, "n_shapes": 3,
                "obj_size_range": (2.2, 3.2),
                "min_sep": 6,
                "layouts": ["uniform", "grid", "clustered", "quadrant",
                            "border"],
            }
        if level == 5:
            return {
                "qtypes": ["count_all", "count_by_color", "count_by_shape",
                           "count_in_quadrant", "count_touching_border"],
                "qtype_weights": [3, 3, 2, 1, 1],
                "min_obj": 14, "max_obj": 20,
                "n_colors": 4, "n_shapes": 3,
                "obj_size_range": (2.0, 2.8),
                "min_sep": 5,
                "layouts": ["uniform", "grid", "clustered", "quadrant",
                            "border"],
            }
        if level == 6:
            return {
                "qtypes": ["count_all", "count_by_color", "count_by_shape",
                           "count_in_quadrant", "count_touching_border"],
                "qtype_weights": [3, 3, 2, 1, 1],
                "min_obj": 18, "max_obj": 28,
                "n_colors": 5, "n_shapes": 4,
                "obj_size_range": (1.8, 2.5),
                "min_sep": 4,
                "layouts": ["uniform", "grid", "clustered", "quadrant",
                            "border"],
            }
        if level == 7:
            return {
                "qtypes": ["count_all", "count_by_color", "count_by_shape",
                           "count_in_quadrant", "count_touching_border"],
                "qtype_weights": [2, 3, 3, 1, 1],
                "min_obj": 22, "max_obj": 34,
                "n_colors": 5, "n_shapes": 4,
                "obj_size_range": (1.6, 2.3),
                "min_sep": 3.5,
                "layouts": ["uniform", "grid", "clustered", "quadrant",
                            "border"],
            }
        if level == 8:
            return {
                "qtypes": ["count_all", "count_by_color", "count_by_shape",
                           "count_in_quadrant", "count_touching_border"],
                "qtype_weights": [2, 3, 3, 1, 1],
                "min_obj": 28, "max_obj": 42,
                "n_colors": 6, "n_shapes": 5,
                "obj_size_range": (1.5, 2.1),
                "min_sep": 3,
                "layouts": ["uniform", "grid", "clustered", "quadrant",
                            "border"],
            }
        return {  # level 9
            "qtypes": ["count_all", "count_by_color", "count_by_shape",
                       "count_in_quadrant", "count_touching_border"],
            "qtype_weights": [2, 3, 3, 1, 1],
            "min_obj": 34, "max_obj": 50,
            "n_colors": 7, "n_shapes": 5,
            "obj_size_range": (1.3, 1.9),
            "min_sep": 2.5,
            "layouts": ["uniform", "grid", "clustered", "quadrant",
                        "border"],
        }

    def _try_generate(self, qtype, level, cfg, attempt
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + attempt * 13
        )

        num_objects = sub_rng.randint(cfg["min_obj"], cfg["max_obj"])
        n_colors = min(cfg["n_colors"], len(_COLORS))
        n_shapes = min(cfg["n_shapes"], len(_SHAPES))

        color_names = sub_rng.sample(list(_COLORS.keys()), n_colors)
        shape_names = sub_rng.sample(_SHAPES, n_shapes)

        canvas_size = 100.0
        obj_size = sub_rng.uniform(*cfg["obj_size_range"])
        min_sep = cfg["min_sep"]

        layout_name = sub_rng.choice(cfg["layouts"])
        positions = _LAYOUT_STRATEGIES[layout_name](
            sub_rng, num_objects, canvas_size, obj_size, min_sep)

        if len(positions) < max(1, cfg["min_obj"]):
            return None

        objects: List[Tuple[float, float, str, str]] = []
        for (x, y) in positions:
            c = sub_rng.choice(color_names)
            s = sub_rng.choice(shape_names)
            objects.append((x, y, c, s))

        question, answer = self._make_qa(
            sub_rng, qtype, objects, color_names, shape_names, canvas_size
        )
        if question is None:
            return None

        image = self._render(sub_rng, objects, obj_size, canvas_size,
                             level, layout_name)
        return question, answer, image

    def _make_qa(self, rng, qtype, objects, color_names, shape_names,
                 canvas_size):
        if qtype == "count_all":
            stem = rng.choice([
                "How many objects are there in total in the image?",
                "Count the total number of shapes shown in the image.",
                "What is the total number of objects visible in the scene?",
                "How many objects can you count in the image above?",
                "Count all the shapes shown in the image.",
                "What is the total count of objects in this image?",
            ])
            return (stem + " Answer with a single integer.", str(len(objects)))

        elif qtype == "count_by_color":
            present = sorted({c for _, _, c, _ in objects})
            if not present:
                return None, None
            target = rng.choice(present)
            count = sum(1 for _, _, c, _ in objects if c == target)
            stem = rng.choice([
                f"How many {target} objects are in the image?",
                f"Count the number of {target}-colored shapes shown.",
                f"How many shapes in the image are colored {target}?",
                f"How many {target} shapes are visible in the scene?",
                f"Count the {target} objects.",
                f"Count the number of {target} objects shown in the image.",
            ])
            return (stem + " Answer with a single integer.", str(count))

        elif qtype == "count_by_shape":
            present = sorted({s for _, _, _, s in objects})
            if not present:
                return None, None
            target = rng.choice(present)
            count = sum(1 for _, _, _, s in objects if s == target)
            plural = target + "s"
            stem = rng.choice([
                f"How many {plural} are in the image?",
                f"Count the number of {plural} shown in the scene.",
                f"How many {target} shapes are visible?",
                f"How many {plural} can you count in the image above?",
                f"Count the {plural}.",
                f"What is the total number of {plural} shown?",
            ])
            return (stem + " Answer with a single integer.", str(count))

        elif qtype == "count_in_quadrant":
            quadrant = rng.choice(["top-left", "top-right",
                                   "bottom-left", "bottom-right"])
            half = canvas_size / 2
            if quadrant == "top-left":
                count = sum(1 for x, y, _, _ in objects
                            if x < half and y >= half)
            elif quadrant == "top-right":
                count = sum(1 for x, y, _, _ in objects
                            if x >= half and y >= half)
            elif quadrant == "bottom-left":
                count = sum(1 for x, y, _, _ in objects
                            if x < half and y < half)
            else:
                count = sum(1 for x, y, _, _ in objects
                            if x >= half and y < half)
            stem = rng.choice([
                f"How many objects are in the {quadrant} quadrant of the image?",
                f"Count the number of shapes located in the {quadrant} region.",
                f"How many shapes are in the {quadrant} section of the image?",
                f"Count the objects that appear in the {quadrant} quadrant.",
            ])
            return (stem + " Answer with a single integer.", str(count))

        elif qtype == "count_touching_border":
            margin = _BORDER_MARGIN
            count = sum(1 for x, y, _, _ in objects
                        if x < margin or x > canvas_size - margin
                        or y < margin or y > canvas_size - margin)
            stem = rng.choice([
                "How many objects are touching or very close to the border of the image?",
                "Count the number of shapes that touch or nearly touch the edge of the image.",
                "How many shapes are near the border of the scene?",
                "Count the objects that lie on or close to the image's edge.",
            ])
            return (stem + " Answer with a single integer.", str(count))
        return None, None

    def _render(self, rng, objects, obj_size, canvas_size, level,
                layout_name):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, canvas_size)
        ax.set_ylim(0, canvas_size)
        ax.set_aspect('equal')
        ax.axis('off')

        # Light quadrant lines (only L4+ where quadrant questions appear)
        if level >= 4 and rng.random() < 0.65:
            half = canvas_size / 2
            ax.axhline(y=half, color='#dcdcdc', linewidth=0.8,
                       linestyle='--', zorder=0)
            ax.axvline(x=half, color='#dcdcdc', linewidth=0.8,
                       linestyle='--', zorder=0)

        edge_color = rng.choice(["black", "#1a1a1a", "#2c3e50", "#333"])
        edge_lw = rng.uniform(0.4, 0.7)
        alpha = 0.85 + rng.random() * 0.12

        for x, y, color_name, shape_name in objects:
            hex_color = _COLORS[color_name]
            if shape_name == "circle":
                ax.add_patch(mpatches.Circle(
                    (x, y), obj_size,
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))
            elif shape_name == "square":
                ax.add_patch(mpatches.Rectangle(
                    (x - obj_size, y - obj_size),
                    obj_size * 2, obj_size * 2,
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))
            elif shape_name == "triangle":
                ax.add_patch(mpatches.PathPatch(
                    _make_triangle_path(x, y, obj_size * 2),
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))
            elif shape_name == "diamond":
                ax.add_patch(mpatches.PathPatch(
                    _make_diamond_path(x, y, obj_size * 2),
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))
            elif shape_name == "star":
                ax.add_patch(mpatches.PathPatch(
                    _make_star_path(x, y, obj_size * 2),
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))
            elif shape_name == "pentagon":
                ax.add_patch(mpatches.PathPatch(
                    _make_pentagon_path(x, y, obj_size * 2),
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))
            elif shape_name == "hexagon":
                ax.add_patch(mpatches.PathPatch(
                    _make_hexagon_path(x, y, obj_size * 2),
                    facecolor=hex_color, edgecolor=edge_color,
                    linewidth=edge_lw, alpha=alpha, zorder=2))

        # Legend at L>=3 where colors matter
        if level >= 3 and rng.random() < 0.8:
            color_set = sorted(set(c for _, _, c, _ in objects))
            handles = [mpatches.Patch(color=_COLORS[c], label=c)
                       for c in color_set]
            ax.legend(handles=handles, loc=rng.choice(["upper right",
                                                        "upper left",
                                                        "lower left"]),
                      fontsize=8, framealpha=0.8, edgecolor='#cccccc')

        # IMPORTANT: title does NOT mention the object count!
        title_base = rng.choice(_TITLE_VARIANTS)
        ax.set_title(title_base,
                     fontsize=style["font_size_base"] + 3,
                     fontweight='bold', pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = PreciseCountingQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
