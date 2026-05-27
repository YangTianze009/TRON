"""
Spatial Rotation QA environment.

Shows a 3D shape from one angle, then asks what it looks like after
rotation or from a different viewpoint. Multiple choice format.
Tests mental rotation — one of the hardest spatial tasks.
Targets: 3D spatial (29.8%).
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

# ------------------------------------------------------------------ #
# Color definitions for cube faces
# ------------------------------------------------------------------ #

_FACE_COLORS = {
    "red": "#E74C3C",
    "blue": "#3498DB",
    "green": "#27AE60",
    "yellow": "#F1C40F",
    "purple": "#8E44AD",
    "orange": "#E67E22",
    "white": "#ECF0F1",
    "gray": "#95A5A6",
}

_COLOR_NAMES = list(_FACE_COLORS.keys())

# Face indices: front=0, back=1, left=2, right=3, top=4, bottom=5
_FACE_NAMES = ["front", "back", "left", "right", "top", "bottom"]

def _rotate_faces_y90cw(faces):
    """Rotate 90 degrees clockwise around vertical (Y) axis.
    front->right, right->back, back->left, left->front.
    Top and bottom stay."""
    return [faces[2], faces[3], faces[1], faces[0], faces[4], faces[5]]

def _rotate_faces_y90ccw(faces):
    """Rotate 90 degrees counter-clockwise around Y axis."""
    return [faces[3], faces[2], faces[0], faces[1], faces[4], faces[5]]

def _rotate_faces_x90cw(faces):
    """Rotate 90 degrees clockwise around X axis (tilt forward).
    front->bottom, bottom->back, back->top, top->front."""
    return [faces[4], faces[5], faces[2], faces[3], faces[0], faces[1]]

def _rotate_faces_z90cw(faces):
    """Rotate 90 degrees clockwise around Z axis (looking from front).
    top->right, right->bottom, bottom->left, left->top."""
    return [faces[0], faces[1], faces[5], faces[4], faces[2], faces[3]]

_ROTATIONS = {
    "90_clockwise_vertical": (_rotate_faces_y90cw, "90° clockwise around the vertical axis"),
    "90_counterclockwise_vertical": (_rotate_faces_y90ccw, "90° counter-clockwise around the vertical axis"),
    "90_forward_tilt": (_rotate_faces_x90cw, "90° forward tilt (around horizontal axis)"),
    "90_roll_clockwise": (_rotate_faces_z90cw, "90° clockwise roll (around front-back axis)"),
}

_VIEWPOINTS = {
    "from_right": ("right side", lambda f: [f[3], f[2], f[0], f[1], f[4], f[5]]),
    "from_left": ("left side", lambda f: [f[2], f[3], f[1], f[0], f[4], f[5]]),
    "from_top": ("top", lambda f: [f[4], f[5], f[2], f[3], f[1], f[0]]),
    "from_back": ("back", lambda f: [f[1], f[0], f[3], f[2], f[4], f[5]]),
}

class SpatialRotationQA(StandaloneVisualEnv):
    ENV_NAME = "spatial_rotation"

    QUESTION_TYPES = [
        "which_face_visible", "identify_after_rotation",
    ]

    def _level_config(self, level: int) -> Dict:
        # Removed "view_from_direction" (text leakage: hidden-face colors
        # were printed as plain text on the image, turning the task into a
        # text-lookup) and "count_color_visible" (answer was always 3
        # because all 6 faces use distinct colors, so the 3 visible faces
        # are always distinct).
        if level <= 0:
            return {"qtypes": ["which_face_visible"],
                    "qweights": [10]}
        if level <= 2:
            return {"qtypes": ["which_face_visible"],
                    "qweights": [10]}
        if level <= 4:
            return {"qtypes": ["which_face_visible"],
                    "qweights": [10]}
        if level <= 6:
            return {"qtypes": ["which_face_visible", "identify_after_rotation"],
                    "qweights": [4, 6]}
        if level <= 8:
            return {"qtypes": ["identify_after_rotation", "which_face_visible"],
                    "qweights": [6, 4]}
        return {"qtypes": ["identify_after_rotation"],
                "qweights": [10]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 5601)
        # Use level-aware sub_rng for all randomness (was `self._rng` which
        # produced identical L0/L3 and L6/L9 images at same seed).
        rng = sub_rng

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        # Generate a cube with colored faces
        available = list(_COLOR_NAMES)
        rng.shuffle(available)
        faces = available[:6]  # [front, back, left, right, top, bottom]

        if qtype == "which_face_visible":
            return self._qa_which_face(rng, faces, level)
        elif qtype == "identify_after_rotation":
            return self._qa_after_rotation(rng, faces, level)
        elif qtype == "view_from_direction":
            return self._qa_view_from(rng, faces, level)
        elif qtype == "count_color_visible":
            return self._qa_count_color(rng, faces, level)
        return None

    def _qa_which_face(self, rng, faces, level=0):
        """After rotation, which color is on the front face?"""
        # Face text labels appear on the three VISIBLE faces (front, top,
        # right). Rotations that move a visible face onto the new front
        # (e.g., x90cw: top->front; y90ccw: right->front) let the model
        # read the answer directly from the text label. We therefore
        # restrict to rotations whose new front comes from a currently
        # hidden face (back, left, or bottom). The hidden-face swatches
        # in _render_cube are color-only (no text), so the model must
        # reason about the rotation to find which swatch is the new front.
        # Visible face indices: 0=front, 3=right, 4=top.
        # Hidden indices: 1=back, 2=left, 5=bottom.
        rot_keys = []
        for k, (fn, _) in _ROTATIONS.items():
            new_faces_k = fn(faces)
            # new_front must come from a hidden original face.
            if new_faces_k[0] in (faces[1], faces[2], faces[5]) and \
               new_faces_k[0] != faces[0]:
                rot_keys.append(k)
        if not rot_keys:
            # Fallback: any rotation that changes the front face.
            rot_keys = [k for k in _ROTATIONS
                        if _ROTATIONS[k][0](faces)[0] != faces[0]]
        if not rot_keys:
            rot_keys = list(_ROTATIONS.keys())
        rot_key = rng.choice(rot_keys)
        rot_fn, rot_desc = _ROTATIONS[rot_key]
        new_faces = rot_fn(faces)
        front_color = new_faces[0]

        q = (f"A cube has colored faces as shown (front view visible). "
             f"After rotating the cube {rot_desc}, "
             f"what color will be on the front face?")
        if level <= 2:
            q += (
                " Hint: identify the visible faces (front, top, right are "
                "shown in isometric view; back/bottom/left are inferred). "
                "Trace through the rotation: e.g. rotating 90° clockwise "
                "about the vertical (Z) axis brings the LEFT face to the "
                "FRONT; about horizontal (X) brings TOP to FRONT, etc."
            )
        image = self._render_cube(rng, faces, rot_desc, new_faces)
        return q, front_color, image

    def _qa_after_rotation(self, rng, faces, level=0):
        """Multiple choice: which shows the cube after rotation?"""
        rot_key = rng.choice(list(_ROTATIONS.keys()))
        rot_fn, rot_desc = _ROTATIONS[rot_key]
        correct_faces = rot_fn(faces)

        # Generate distractors by applying wrong rotations
        distractor_fns = [v[0] for k, v in _ROTATIONS.items() if k != rot_key]
        options = [correct_faces]
        for fn in distractor_fns[:3]:
            d = fn(faces)
            if not any(d == o for o in options):
                options.append(d)
        while len(options) < 4:
            d = list(faces)
            rng.shuffle(d)
            if not any(d == o for o in options):
                options.append(d)

        indices = list(range(len(options)))
        rng.shuffle(indices)
        shuffled = [options[i] for i in indices]
        correct_label = chr(65 + indices.index(0))

        q = (f"The original cube is shown on the left. After rotating "
             f"{rot_desc}, which option (A, B, C, D) shows the correct "
             f"result? Answer with just the letter.")

        image = self._render_multi(rng, faces, shuffled, rot_desc)
        return q, correct_label, image

    def _qa_view_from(self, rng, faces, level=0):
        """What does the cube look like from a different direction?"""
        vp_key = rng.choice(list(_VIEWPOINTS.keys()))
        direction, transform_fn = _VIEWPOINTS[vp_key]
        viewed_faces = transform_fn(faces)
        front_in_view = viewed_faces[0]

        q = (f"The cube is shown from the front. If you look at it "
             f"from the {direction}, what color do you see on the face "
             f"that is now directly facing you?")
        image = self._render_cube_simple(rng, faces)
        return q, front_in_view, image

    def _qa_count_color(self, rng, faces, level=0):
        """From front-top-right view, how many distinct colors visible?"""
        # From isometric view, front, top, and right are visible
        visible = {faces[0], faces[4], faces[3]}  # front, top, right
        count = len(visible)
        q = ("Looking at the cube from an isometric perspective "
             "(front, top, and right faces visible), how many distinct "
             "colors can you see?")
        image = self._render_cube_simple(rng, faces)
        return q, str(count), image

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    def _draw_cube_face(self, ax, cx, cy, size, front, top, right,
                         label="", show_labels=True):
        """Draw an isometric cube showing front, top, and right faces."""
        s = size
        h = s * 0.5  # isometric height factor

        # Front face (parallelogram)
        front_pts = [
            (cx - s / 2, cy - h / 2),
            (cx + s / 2, cy - h / 2),
            (cx + s / 2, cy + h / 2),
            (cx - s / 2, cy + h / 2),
        ]
        ax.add_patch(Polygon(front_pts, closed=True,
                             facecolor=_FACE_COLORS.get(front, "#CCC"),
                             edgecolor="black", linewidth=2))
        if show_labels:
            ax.text(cx, cy, front, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.7, pad=1,
                              edgecolor="none"))

        # Top face (parallelogram above)
        top_pts = [
            (cx - s / 2, cy + h / 2),
            (cx, cy + h),
            (cx + s / 2, cy + h / 2 + h / 2),
            (cx, cy + h / 2),
        ]
        # Simplified: rectangle above
        top_pts = [
            (cx - s / 2, cy + h / 2),
            (cx - s / 4, cy + h),
            (cx + s / 4 + s / 2, cy + h),
            (cx + s / 2, cy + h / 2),
        ]
        ax.add_patch(Polygon(top_pts, closed=True,
                             facecolor=_FACE_COLORS.get(top, "#CCC"),
                             edgecolor="black", linewidth=2))
        if show_labels:
            top_cx = cx + s / 8
            top_cy = cy + h / 2 + h / 4
            ax.text(top_cx, top_cy, top, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.7, pad=1,
                              edgecolor="none"))

        # Right face (parallelogram to the right)
        right_pts = [
            (cx + s / 2, cy - h / 2),
            (cx + s / 2 + s / 4, cy),
            (cx + s / 4 + s / 2, cy + h),
            (cx + s / 2, cy + h / 2),
        ]
        ax.add_patch(Polygon(right_pts, closed=True,
                             facecolor=_FACE_COLORS.get(right, "#CCC"),
                             edgecolor="black", linewidth=2))
        if show_labels:
            r_cx = cx + s / 2 + s / 8
            r_cy = cy + h / 4
            ax.text(r_cx, r_cy, right, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.7, pad=1,
                              edgecolor="none"))

        if label:
            ax.text(cx, cy - h / 2 - 0.3, label, ha="center",
                    fontsize=11, fontweight="bold")

    def _render_cube_simple(self, rng, faces) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        self._draw_cube_face(ax, 3, 3, 3, faces[0], faces[4], faces[3])

        ax.text(0.5, 5.5, f"Back: {faces[1]}", fontsize=fs - 3, color="gray",
                fontfamily=ff)
        ax.text(0.5, 5.0, f"Left: {faces[2]}", fontsize=fs - 3, color="gray",
                fontfamily=ff)
        ax.text(0.5, 4.5, f"Bottom: {faces[5]}", fontsize=fs - 3, color="gray",
                fontfamily=ff)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6.5)
        ax.set_title("Colored Cube", fontsize=fs + 1, fontweight="bold",
                     fontfamily=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_cube(self, rng, original_faces, rot_desc, new_faces) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        for ax in [ax1, ax2]:
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor(style["bg_color"])

        # Show a small unfolded net for the three HIDDEN faces so the
        # model has all color information without leaking answers via
        # plain text (previous "Back: color" text was trivial OCR).
        self._draw_cube_face(ax1, 3, 2.5, 2.5, original_faces[0],
                             original_faces[4], original_faces[3])
        # Small preview swatches for back/left/bottom, color-only (no name).
        from matplotlib.patches import Rectangle
        swatch_specs = [
            (0.4, 4.6, "Back",   original_faces[1]),
            (0.4, 4.1, "Left",   original_faces[2]),
            (0.4, 3.6, "Bottom", original_faces[5]),
        ]
        for x, y, name, color in swatch_specs:
            ax1.add_patch(Rectangle((x, y), 0.4, 0.35,
                                    facecolor=_FACE_COLORS.get(color, "#CCC"),
                                    edgecolor="black", linewidth=1))
            ax1.text(x + 0.5, y + 0.17, name, fontsize=fs - 4,
                     color="gray", fontfamily=ff, va="center")
        ax1.set_xlim(0, 6)
        ax1.set_ylim(0, 5.5)
        ax1.set_title("Original", fontsize=fs, fontweight="bold", fontfamily=ff)

        self._draw_cube_face(ax2, 3, 2.5, 2.5, "?", "?", "?", show_labels=False)
        ax2.text(3, 2.5, "?", ha="center", va="center", fontsize=fs + 12,
                 fontweight="bold", color="black", fontfamily=ff)
        ax2.set_xlim(0, 6)
        ax2.set_ylim(0, 5.5)
        ax2.set_title(f"After: {rot_desc}", fontsize=fs - 1, fontweight="bold",
                     fontfamily=ff)

        fig.suptitle("Spatial Rotation", fontsize=fs + 2, fontweight="bold",
                     fontfamily=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_multi(self, rng, original, options, rot_desc) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        fig = plt.figure(figsize=(16 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        ax0 = fig.add_subplot(1, 5, 1)
        ax0.set_aspect("equal")
        ax0.axis("off")
        ax0.set_facecolor(style["bg_color"])
        self._draw_cube_face(ax0, 2, 2, 2, original[0], original[4], original[3])
        # Replace text color-names with small colored swatches so the
        # hidden-face colors do not leak via OCR.
        from matplotlib.patches import Rectangle
        swatches = [("B", original[1]), ("L", original[2]), ("Bot", original[5])]
        for i, (name, color) in enumerate(swatches):
            x = 0.2 + i * 1.1
            ax0.add_patch(Rectangle((x, 3.55), 0.3, 0.3,
                                    facecolor=_FACE_COLORS.get(color, "#CCC"),
                                    edgecolor="black", linewidth=1))
            ax0.text(x + 0.35, 3.70, name, fontsize=fs - 6,
                     color="gray", fontfamily=ff, va="center")
        ax0.set_xlim(0, 4.5)
        ax0.set_ylim(0, 4.5)
        ax0.set_title("Original", fontsize=fs - 2, fontweight="bold", fontfamily=ff)

        labels = ["A", "B", "C", "D"]
        for idx in range(min(4, len(options))):
            ax = fig.add_subplot(1, 5, idx + 2)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor(style["bg_color"])
            f = options[idx]
            self._draw_cube_face(ax, 2, 2, 2, f[0], f[4], f[3])
            ax.set_xlim(0, 4.5)
            ax.set_ylim(0, 4.5)
            ax.set_title(f"Option {labels[idx]}", fontsize=fs - 2,
                        fontweight="bold", fontfamily=ff)

        fig.suptitle(f"After {rot_desc}, which is correct?",
                     fontsize=fs, fontweight="bold", y=1.02, fontfamily=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
