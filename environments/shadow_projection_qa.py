"""
Shadow Projection QA environment.

Renders a 3D object with light source and its shadow outline.
Questions: shadow shape, which object casts shadow, shadow dimensions.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: cube/sphere with overhead light, ask `identify_shadow` ("What shape?").
L1: + cone/cylinder, overhead only.
L2: pyramid added, overhead only.
L3: all 5 objects, overhead; + `shadow_shape_name`.
L4: introduces front-right / front-left light; + `count_shadow_sides`.
L5: + `identify_object`.
L6: + `light_direction`.
L7: all question types.
L8: harder light directions.
L9: full complexity.

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Ellipse
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_OBJECTS = [
    {"name": "sphere", "shadow": "circle", "shadow_desc": "circle"},
    {"name": "cube", "shadow": "square", "shadow_desc": "square"},
    {"name": "cylinder", "shadow": "rectangle", "shadow_desc": "rectangle"},
    {"name": "cone", "shadow": "triangle", "shadow_desc": "triangle"},
    {"name": "pyramid", "shadow": "square", "shadow_desc": "square"},
]

_TITLE_VARIANTS = ["3D Object & Shadow", "Shadow Cast", "Object and Shadow",
                   "Light and Shadow", "Shadow Projection",
                   "Solid & Cast Shadow", "Shadow Test"]

class ShadowProjectionQA(StandaloneVisualEnv):
    ENV_NAME = "shadow_projection"

    QUESTION_TYPES = [
        "identify_shadow", "identify_object", "shadow_shape_name",
        "count_shadow_sides", "light_direction",
    ]

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"], weights=cfg["qtype_weights"])[0]
        for _ in range(10):
            result = self._try_generate(qtype, level, cfg)
            if result is not None:
                self._primary_complexity_feature = level * 3 + len(result[1])
                return result
        return None

    def _level_config(self, level):
        # Keep question type consistent (identify_shadow) for L0-L5 to
        # increase monotonically with object count and light direction.
        # Harder question types (count_shadow_sides, light_direction) only
        # at L6+. Fixes progressive increase from L3->L9.
        if level == 0:
            return {"qtypes": ["identify_shadow"], "qtype_weights": [1],
                    "objects": [_OBJECTS[0], _OBJECTS[1]],  # sphere, cube
                    "light_dirs": ["above"]}
        if level == 1:
            return {"qtypes": ["identify_shadow"], "qtype_weights": [1],
                    "objects": _OBJECTS[:3],  # +cylinder
                    "light_dirs": ["above"]}
        if level == 2:
            return {"qtypes": ["identify_shadow"], "qtype_weights": [1],
                    "objects": _OBJECTS[:4],  # +cone
                    "light_dirs": ["above"]}
        if level == 3:
            return {"qtypes": ["identify_shadow"], "qtype_weights": [1],
                    "objects": _OBJECTS,  # all 5
                    "light_dirs": ["above"]}
        if level == 4:
            return {"qtypes": ["identify_shadow"], "qtype_weights": [1],
                    "objects": _OBJECTS,
                    "light_dirs": ["above", "front-right"]}
        if level == 5:
            return {"qtypes": ["identify_shadow"], "qtype_weights": [1],
                    "objects": _OBJECTS,
                    "light_dirs": ["above", "front-right", "front-left"]}
        if level == 6:
            return {"qtypes": ["identify_shadow", "count_shadow_sides"],
                    "qtype_weights": [5, 5],
                    "objects": _OBJECTS,
                    "light_dirs": ["above", "front-right", "front-left"]}
        if level == 7:
            return {"qtypes": ["identify_shadow", "count_shadow_sides",
                               "identify_object"],
                    "qtype_weights": [4, 3, 3],
                    "objects": _OBJECTS,
                    "light_dirs": ["above", "front-right", "front-left"]}
        if level == 8:
            return {"qtypes": ["identify_shadow", "count_shadow_sides",
                               "identify_object", "light_direction"],
                    "qtype_weights": [3, 3, 2, 2],
                    "objects": _OBJECTS,
                    "light_dirs": ["above", "front-right", "front-left"]}
        return {"qtypes": ["count_shadow_sides", "identify_object",
                           "light_direction"],
                "qtype_weights": [4, 3, 3],
                "objects": _OBJECTS,
                "light_dirs": ["above", "front-right", "front-left"]}

    def _try_generate(self, qtype, level, cfg):
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )

        obj = sub_rng.choice(cfg["objects"])
        obj_size = sub_rng.uniform(0.8, 1.3)
        light_dir = sub_rng.choice(cfg["light_dirs"])

        img = self._render(obj, obj_size, light_dir, sub_rng)

        if qtype == "identify_shadow":
            stems = [
                "A light shines on the 3D solid shown in the image. "
                "What shape is its cast shadow? Answer: circle, square, rectangle, or triangle.",
                "The figure shows a 3D solid and a light source. "
                "Which shape best describes the shadow it casts?",
                "Identify the silhouette cast by the 3D solid in the image. "
                "Answer with: circle, square, rectangle, or triangle.",
                "Given the 3D solid and light in the figure, name the "
                "resulting shadow shape (circle/square/rectangle/triangle).",
            ]
            q = sub_rng.choice(stems)
            a = obj["shadow_desc"]

        elif qtype == "identify_object":
            # Only valid when the shadow shape uniquely identifies the
            # solid — so exclude the cube/pyramid ambiguity (both cast a
            # square shadow from above).
            if obj["shadow"] == "square":
                return None
            stems = [
                f"The cast shadow visible in the image has a {obj['shadow_desc']} outline. "
                f"Which 3D object casts this shadow? Choose from: sphere, cube, cylinder, cone, pyramid.",
                f"A {obj['shadow_desc']} shadow is depicted. Which solid (sphere, cube, cylinder, cone, pyramid) produces this shadow?",
                f"The shadow shown is a {obj['shadow_desc']}. Identify the originating 3D object from: sphere, cube, cylinder, cone, pyramid.",
            ]
            q = sub_rng.choice(stems)
            a = obj["name"]

        elif qtype == "shadow_shape_name":
            stems = [
                "What is the shape of the shadow shown on the ground plane?",
                "Describe the shape of the shadow in the image.",
                "What planar shape best matches the shadow visible in the figure?",
                "Identify the geometric outline of the shadow displayed.",
            ]
            q = sub_rng.choice(stems) + " Answer with the shape name."
            a = obj["shadow_desc"]

        elif qtype == "count_shadow_sides":
            sides = {"circle": 0, "square": 4, "rectangle": 4, "triangle": 3}
            n = sides.get(obj["shadow"], 0)
            stems = [
                "How many straight sides does the shadow have? Answer 0 for a circle.",
                "Count the number of straight edges in the shadow. Answer 0 for curved shadows.",
                "How many polygon edges form the boundary of the shadow? Answer 0 if the boundary is curved.",
            ]
            q = sub_rng.choice(stems)
            a = str(n)

        elif qtype == "light_direction":
            stems = [
                "From which direction is the light source shining on the object? Answer: above, front-right, or front-left.",
                "Where is the light source located relative to the object? Answer: above, front-right, or front-left.",
                "Identify the position of the light source. Choose: above, front-right, or front-left.",
            ]
            q = sub_rng.choice(stems)
            a = light_dir
        else:
            return None

        return q, a, img

    def _render(self, obj, obj_size, light_dir, sub_rng):
        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * s, 6 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")

        palette = list(style["palette"])
        sub_rng.shuffle(palette)
        fs = style["font_size_base"]
        ff = style["font_family"]

        ax.fill_between([0, 10], [0, 0], [1.5, 1.5], color="#e0e0e0", alpha=0.4)
        ax.axhline(y=1.5, color="#999", linewidth=1)

        obj_cx, obj_cy = 3.5, 4.0
        self._draw_3d_object(ax, obj["name"], obj_cx, obj_cy, obj_size, palette, style)

        shadow_cx = 6.0 if "right" in light_dir else 4.5 if "left" in light_dir else 3.5
        shadow_cy = 0.8
        self._draw_shadow(ax, obj["shadow"], shadow_cx, shadow_cy, obj_size * 1.2, style)

        lx = 1.0 if "left" in light_dir else 6.5 if "right" in light_dir else 3.5
        ly = 7.0
        ax.plot(lx, ly, "*", color="#ffd700", markersize=20, zorder=10)
        ax.text(lx, ly + 0.4, "Light", ha="center", fontsize=fs - 1,
                fontfamily=ff, color="#b8860b")
        for dx, dy in [(obj_cx, obj_cy + obj_size * 0.5),
                       (shadow_cx, shadow_cy + 0.3)]:
            ax.annotate("", xy=(dx, dy), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle="-", color="#ffd700",
                                        alpha=0.4, linewidth=1, linestyle="--"))

        ax.set_xlim(-0.5, 8)
        ax.set_ylim(-0.5, 8.5)
        ax.axis("off")
        title = sub_rng.choice(_TITLE_VARIANTS)
        ax.set_title(title, fontsize=fs + 3, fontweight="bold",
                     fontfamily=ff, pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_3d_object(self, ax, name, cx, cy, size, palette, style):
        color = palette[0]
        lw = style["line_width"]
        if name == "sphere":
            ax.add_patch(Circle((cx, cy), size * 0.8, facecolor=color,
                                 edgecolor="#333", linewidth=lw, alpha=0.8, zorder=5))
            ax.add_patch(Circle((cx - size * 0.2, cy + size * 0.2),
                                 size * 0.2, facecolor="white", alpha=0.3,
                                 zorder=6))
        elif name == "cube":
            sq = size * 0.8
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - sq / 2, cy - sq / 2), sq, sq,
                facecolor=color, edgecolor="#333", linewidth=lw, zorder=5))
            off = sq * 0.35
            verts = [(cx - sq / 2, cy + sq / 2),
                     (cx - sq / 2 + off, cy + sq / 2 + off),
                     (cx + sq / 2 + off, cy + sq / 2 + off),
                     (cx + sq / 2, cy + sq / 2)]
            ax.add_patch(Polygon(verts, facecolor=palette[1],
                                 edgecolor="#333", linewidth=lw,
                                 alpha=0.8, zorder=5))
            verts2 = [(cx + sq / 2, cy - sq / 2),
                      (cx + sq / 2 + off, cy - sq / 2 + off),
                      (cx + sq / 2 + off, cy + sq / 2 + off),
                      (cx + sq / 2, cy + sq / 2)]
            ax.add_patch(Polygon(verts2, facecolor=palette[2],
                                 edgecolor="#333", linewidth=lw,
                                 alpha=0.7, zorder=5))
        elif name == "cylinder":
            w, h = size * 0.7, size * 1.2
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                facecolor=color, edgecolor="#333", linewidth=lw, zorder=5))
            ax.add_patch(Ellipse((cx, cy + h / 2), w, size * 0.3,
                                  facecolor=palette[1], edgecolor="#333",
                                  linewidth=lw, zorder=6))
        elif name == "cone":
            h = size * 1.3
            base_w = size * 0.8
            verts = [(cx, cy + h / 2), (cx - base_w / 2, cy - h / 2),
                     (cx + base_w / 2, cy - h / 2)]
            ax.add_patch(Polygon(verts, facecolor=color, edgecolor="#333",
                                 linewidth=lw, zorder=5))
        elif name == "pyramid":
            h = size * 1.2
            base_w = size * 0.9
            verts = [(cx, cy + h / 2), (cx - base_w / 2, cy - h / 2),
                     (cx + base_w / 2, cy - h / 2)]
            ax.add_patch(Polygon(verts, facecolor=color, edgecolor="#333",
                                 linewidth=lw, zorder=5))
            off = base_w * 0.3
            verts2 = [(cx, cy + h / 2), (cx + base_w / 2, cy - h / 2),
                      (cx + base_w / 2 + off, cy - h / 2 + off * 0.3)]
            ax.add_patch(Polygon(verts2, facecolor=palette[1],
                                 edgecolor="#333", linewidth=lw,
                                 alpha=0.7, zorder=5))

        # Intentionally do NOT label the object name on the figure —
        # would leak the answer for identify_shadow / identify_object.

    def _draw_shadow(self, ax, shape, cx, cy, size, style):
        # Shadows are drawn at a reasonable size so their shape is
        # unambiguously recognizable (circle vs square vs triangle).
        color = "#555555"
        alpha = 0.55
        if shape == "circle":
            # Make it look like a proper circle, not a thin ellipse.
            p = Circle((cx, cy), size * 0.55, facecolor=color,
                       alpha=alpha, zorder=2, edgecolor="#333", linewidth=1.0)
            ax.add_patch(p)
        elif shape == "square":
            sq = size * 1.0
            ax.add_patch(mpatches.Rectangle(
                (cx - sq / 2, cy - sq / 2), sq, sq,
                facecolor=color, alpha=alpha, zorder=2,
                edgecolor="#333", linewidth=1.0))
        elif shape == "rectangle":
            w, h = size * 1.2, size * 0.6
            ax.add_patch(mpatches.Rectangle(
                (cx - w / 2, cy - h / 2), w, h,
                facecolor=color, alpha=alpha, zorder=2,
                edgecolor="#333", linewidth=1.0))
        elif shape == "triangle":
            sq = size * 1.0
            verts = [(cx, cy + sq * 0.5), (cx - sq / 2, cy - sq * 0.5),
                     (cx + sq / 2, cy - sq * 0.5)]
            ax.add_patch(Polygon(verts, facecolor=color, alpha=alpha, zorder=2,
                                 edgecolor="#333", linewidth=1.0))

        ax.text(cx, cy - 0.5, "Shadow", ha="center",
                fontsize=style["font_size_base"] - 1,
                fontfamily=style["font_family"], color="#666", style="italic")

if __name__ == "__main__":
    env = ShadowProjectionQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
