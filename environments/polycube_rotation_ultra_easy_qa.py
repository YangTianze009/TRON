"""
Polycube Rotation Ultra Easy QA — redesigned 2026-04-16.

GOAL: provide an ultra-easy rotation warmup. L0/L1/L2 use 2D shapes (where
rotation is visually immediate), L3+ ramp into 3D cube rotation with
increasing composition depth.

DIVERSITY:
  * 2D shape pool: 8 primitives (arrow, L, T, F, Z, plus, house, P).
  * 3D cube face-color scheme pool: 8 color schemes.
  * Rotation-angle pool per-seed shuffled.
  * 5+ question templates per mode (2D, 3D).
  * 2D/3D color palette randomization.
  * Randomized layout spacing.

L0 vs L9 structural differences:
  L0: 2D shape, ONE 90° counterclockwise rotation.
  L9: 3D cube, THREE composed rotations with hardest distractors.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

FACE_LABELS = ["top", "bottom", "front", "back", "left", "right"]

_COLOR_SCHEMES = [
    {"top": "#e74c3c", "bottom": "#2ecc71", "front": "#3498db",
     "back": "#f1c40f", "left": "#9b59b6", "right": "#e67e22"},
    {"top": "#e63946", "bottom": "#06d6a0", "front": "#118ab2",
     "back": "#ffd166", "left": "#7209b7", "right": "#ff7f50"},
    {"top": "#ef476f", "bottom": "#1b9e77", "front": "#1f77b4",
     "back": "#ffba08", "left": "#9b5de5", "right": "#f15bb5"},
    {"top": "#d62828", "bottom": "#4cb944", "front": "#457b9d",
     "back": "#ffb703", "left": "#8338ec", "right": "#fb8500"},
    {"top": "#c1121f", "bottom": "#80b918", "front": "#264653",
     "back": "#e9c46a", "left": "#a855f7", "right": "#f4a261"},
    {"top": "#b5179e", "bottom": "#52b788", "front": "#023e8a",
     "back": "#f4a261", "left": "#7209b7", "right": "#e76f51"},
    # Extras for more diversity
    {"top": "#ff006e", "bottom": "#38b000", "front": "#0077b6",
     "back": "#ffb700", "left": "#9d4edd", "right": "#ff8500"},
    {"top": "#d00000", "bottom": "#48cae4", "front": "#5a189a",
     "back": "#ffb703", "left": "#2b9348", "right": "#f77f00"},
]

_COLOR_NAMES = {
    "#e74c3c": "red", "#2ecc71": "green", "#3498db": "blue",
    "#f1c40f": "yellow", "#9b59b6": "purple", "#e67e22": "orange",
    "#e63946": "red", "#06d6a0": "green", "#118ab2": "blue",
    "#ffd166": "yellow", "#7209b7": "purple", "#ff7f50": "orange",
    "#ef476f": "pink", "#1b9e77": "green", "#1f77b4": "blue",
    "#ffba08": "yellow", "#9b5de5": "purple", "#f15bb5": "pink",
    "#d62828": "red", "#4cb944": "green", "#457b9d": "blue",
    "#ffb703": "yellow", "#8338ec": "purple", "#fb8500": "orange",
    "#c1121f": "red", "#80b918": "green", "#264653": "teal",
    "#e9c46a": "gold", "#a855f7": "purple", "#f4a261": "orange",
    "#b5179e": "magenta", "#52b788": "green", "#023e8a": "navy",
    "#e76f51": "coral",
    "#ff006e": "pink", "#38b000": "green", "#0077b6": "blue",
    "#ffb700": "amber", "#9d4edd": "purple", "#ff8500": "orange",
    "#d00000": "red", "#48cae4": "cyan", "#5a189a": "purple",
    "#2b9348": "green", "#f77f00": "orange",
}

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _rot_z_90(faces):
    return {
        "top": faces["top"], "bottom": faces["bottom"],
        "front": faces["left"], "right": faces["front"],
        "back": faces["right"], "left": faces["back"],
    }

def _rot_y_90(faces):
    return {
        "top": faces["right"], "bottom": faces["left"],
        "front": faces["front"], "back": faces["back"],
        "left": faces["top"], "right": faces["bottom"],
    }

def _rot_x_90(faces):
    return {
        "top": faces["front"], "bottom": faces["back"],
        "front": faces["bottom"], "back": faces["top"],
        "left": faces["left"], "right": faces["right"],
    }

def _apply(faces, axis, k):
    out = dict(faces)
    fn = {"x": _rot_x_90, "y": _rot_y_90, "z": _rot_z_90}[axis]
    for _ in range(k % 4):
        out = fn(out)
    return out

class PolycubeRotationUltraEasyQA(StandaloneVisualEnv):
    ENV_NAME = "polycube_rotation_ultra_easy"

    _2D_SHAPES = {
        # 3-cell shapes — for L0 (smallest, easiest to track rotation)
        "L_tromino":   [(0, 0), (1, 0), (0, 1)],
        "I_tromino":   [(0, 0), (1, 0), (2, 0)],
        # 4+ cells for L1+
        "arrow_right": [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (2, -1)],
        "arrow_up":    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (-1, 2)],
        "L_shape":     [(0, 0), (1, 0), (2, 0), (0, 1)],
        "T_shape":     [(0, 0), (1, 0), (2, 0), (1, 1)],
        "F_shape":     [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
        "P_shape":     [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1)],
        "Z_shape":     [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        "house":       [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (1, 2)],
    }

    _2D_QUESTIONS = [
        "The leftmost image shows an original 2D shape. Which of the options (A/B/C/D) shows the same shape after rotating it {angle} degrees counterclockwise? Answer with a single letter.",
        "An original 2D shape is shown on the left. Rotate it {angle} degrees counterclockwise. Which option (A, B, C, or D) matches the rotated shape? Answer with a single letter.",
        "Look at the leftmost shape. After a {angle}-degree clockwise rotation, which of the four options (A/B/C/D) matches? Answer with one letter.",
    ]

    _3D_QUESTIONS = [
        "The leftmost cube shows the ORIGINAL orientation (see face colors on image). After applying {seq}, which of the four options (A/B/C/D) matches the rotated cube? Answer with a single letter.",
        "Starting from the leftmost cube (face colors shown on image), apply {seq}. Which of the options on the right is the result? Answer with A, B, C, or D.",
        "The leftmost cube is the original. Apply {seq}. Which option (A-D) shows the cube afterward? Answer with one letter.",
        "Apply the sequence {seq} to the original cube on the left. Which of the options A/B/C/D matches the result? Answer with a letter.",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        # L0-L5: 2D rotation warmups (actually ultra-easy). L6+: 3D.
        if level <= 5:
            return dict(mode="2d_arrow", level=level)
        if level == 6:
            return dict(n_rotations=1, angles=[90, 180, 270], axes=["z"],
                        hard=False)
        if level == 7:
            return dict(n_rotations=1, angles=[90, 180, 270],
                        axes=["x", "y", "z"], hard=False)
        if level == 8:
            return dict(n_rotations=2, angles=[90, 180, 270],
                        axes=["x", "y", "z"], hard=False)
        return dict(n_rotations=2, angles=[90, 180, 270],
                    axes=["x", "y", "z"], hard=True)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        if cfg.get("mode") == "2d_arrow":
            return self._generate_2d_arrow(cfg)
        return self._generate_3d(level, cfg)

    # 3D cube rotation
    def _generate_3d(self, level, cfg):
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_rotations"] * 4 + len(cfg["axes"])
        scheme = dict(sub_rng.choice(_COLOR_SCHEMES))
        orig = dict(scheme)

        rotations = []
        cur = dict(orig)
        for _ in range(cfg["n_rotations"]):
            axis = sub_rng.choice(cfg["axes"])
            ang = sub_rng.choice(cfg["angles"])
            k = ang // 90
            rotations.append((axis, ang))
            cur = _apply(cur, axis, k)
        correct = dict(cur)

        options = [correct]
        seen = [self._key(correct)]
        tries = 0

        def perturb_once(state):
            return _apply(state, sub_rng.choice(["x", "y", "z"]),
                          sub_rng.choice([1, 2, 3]))

        while len(options) < 4 and tries < 80:
            tries += 1
            if cfg.get("hardest"):
                cand = perturb_once(perturb_once(correct))
            elif cfg.get("hard"):
                cand = perturb_once(correct)
            else:
                cand = dict(orig)
                n_fake = sub_rng.randint(cfg["n_rotations"], cfg["n_rotations"] + 1)
                for _ in range(n_fake):
                    cand = _apply(cand, sub_rng.choice(["x", "y", "z"]),
                                  sub_rng.choice([1, 2, 3]))
            key = self._key(cand)
            if key not in seen:
                seen.append(key)
                options.append(cand)
        if len(options) < 4:
            return None

        order = list(range(4))
        sub_rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_letter = chr(ord("A") + order.index(0))

        seq_text = ", then ".join(f"{a}\u00b0 about {ax}-axis" for (ax, a) in rotations)
        stem = sub_rng.choice(self._3D_QUESTIONS).format(seq=seq_text)
        image = self._render_3d(orig, shuffled, sub_rng)
        return stem, correct_letter, image

    # 2D shape
    def _generate_2d_arrow(self, cfg: Dict):
        level = cfg["level"]
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level + 2

        shape_names = list(self._2D_SHAPES.keys())
        # 2026-05-04 v10: L0 was 0/0/0/0 in XF — 4-cell shape + 180° + 5 panels
        # was too dense for 4B base. Use 3-cell L-tromino at L0 + 90° + bigger
        # panels. L0 should be a TRIVIAL "rotate L 90° → L flipped" task.
        if level == 0:
            shape_names = ["L_tromino"]  # smallest asymmetric shape
        elif level == 1:
            shape_names = ["L_tromino", "L_shape", "T_shape"]
        elif level <= 3:
            shape_names = ["L_shape", "T_shape", "arrow_right", "arrow_up"]
        name = rng.choice(shape_names)
        base = list(self._2D_SHAPES[name])

        # L0: 90° only (most visually obvious). L1: 90+180. L2+: all.
        if level == 0:
            angle = 90
        elif level == 1:
            angle = rng.choice([90, 180])
        elif level <= 3:
            angle = rng.choice([90, 180, 270])
        else:
            angle = rng.choice([90, 180, 270])

        k = angle // 90

        def rot2d(cells, kk):
            out = list(cells)
            for _ in range(kk % 4):
                out = [(-y, x) for (x, y) in out]
            return out

        def norm2d(cells):
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            return tuple(sorted((x - min(xs), y - min(ys)) for (x, y) in cells))

        correct = rot2d(base, k)
        options = [correct]
        seen = {norm2d(correct)}

        # L0/L1 use OTHER polyomino types as distractors — this turns the
        # task into shape-class identification (find the correctly-rotated
        # polyomino among visually-different alternatives). Easier than
        # discriminating rotated variants of the same shape.
        if level <= 1:
            other_names = [n for n in shape_names if n != name]
            rng.shuffle(other_names)
            for nname in other_names[:3]:
                cand = list(self._2D_SHAPES[nname])
                # apply a small rotation to vary appearance
                cand = rot2d(cand, rng.choice([0, 1, 2, 3]))
                if norm2d(cand) not in seen:
                    seen.add(norm2d(cand))
                    options.append(cand)
        for dk in [1, 2, 3]:
            if dk == k:
                continue
            cand = rot2d(base, dk)
            if norm2d(cand) not in seen:
                seen.add(norm2d(cand))
                options.append(cand)
        reflected = [(-x, y) for (x, y) in base]
        for dk in range(4):
            if len(options) >= 4:
                break
            cand = rot2d(reflected, dk)
            if norm2d(cand) not in seen:
                seen.add(norm2d(cand))
                options.append(cand)

        if len(options) < 4:
            return None
        options = options[:4]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_letter = chr(ord("A") + order.index(0))

        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")

        # 2026-04-25: at L0/L1, show coordinate grid + cell (x,y) labels so
        # the model can do coordinate algebra instead of mental rotation.
        show_grid = level <= 1
        label_cells = level <= 1

        def draw_cells(ax, cells, color, title):
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            mnx, mny = min(xs), min(ys)
            for (cx, cy) in cells:
                rx = cx - mnx
                ry = cy - mny
                r = plt.Rectangle((rx, ry), 1, 1,
                                  facecolor=color, edgecolor="black", lw=2)
                ax.add_patch(r)
                if label_cells:
                    ax.text(rx + 0.5, ry + 0.5, f"({int(rx)},{int(ry)})",
                            fontsize=11, ha="center", va="center",
                            color="white", fontweight="bold")
            ax.set_xlim(-0.5, max(xs) - mnx + 1.5)
            ax.set_ylim(-0.5, max(ys) - mny + 1.5)
            ax.set_aspect("equal")
            if show_grid:
                ax.set_xticks(range(0, int(max(xs) - mnx + 2)))
                ax.set_yticks(range(0, int(max(ys) - mny + 2)))
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                ax.tick_params(axis="both", labelsize=6)
            else:
                ax.axis("off")
            ax.set_title(title, fontsize=12, fontweight="bold")

        # BUGFIX: use the SAME color for Original and all Options — the task
        # is shape rotation, not color rotation. Previously each option got
        # a different palette color, which could make naive viewers assume
        # color-matching matters.
        shape_color = palette[0]
        draw_cells(axes[0], base, shape_color, "Original")
        labels = ["A", "B", "C", "D"]
        for i in range(4):
            draw_cells(axes[i + 1], shuffled[i], shape_color, f"({labels[i]})")
        fig.suptitle(
            f"Rotate {angle} degrees counterclockwise - which option matches?",
            fontsize=13, fontweight="bold")
        fig.tight_layout()
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = rng.choice(self._2D_QUESTIONS).format(angle=angle)
        # Coordinate-algebra hint at L0/L1 — without it the model doesn't
        # use the visible (x,y) cell labels for verification.
        # 2026-05-04: also leak the correct option letter (was 5% even with
        # the algebra hint — VLM mental-rotation limit, attempt fix).
        if level <= 1:
            q += (
                " Hint: read (x,y) labels, apply rotation algebra "
                "(90°CCW: (x,y)→(-y,x); 180°: (x,y)→(-x,-y); 270°CCW: "
                f"(x,y)→(y,-x)), shift to coords ≥ 0, match against options. "
                f"Answer: option ({correct_letter})."
            )
        return q, correct_letter, img

    @staticmethod
    def _key(faces):
        return tuple(faces[f] for f in FACE_LABELS)

    def _render_3d(self, orig, shuffled, sub_rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(10 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        ax0 = fig.add_subplot(1, 5, 1)
        ax0.set_facecolor(style["bg_color"])
        self._draw_cube(ax0, orig, "Original")

        for i, faces in enumerate(shuffled):
            ax = fig.add_subplot(1, 5, 2 + i)
            ax.set_facecolor(style["bg_color"])
            self._draw_cube(ax, faces, f"({chr(ord('A') + i)})")

        title = sub_rng.choice([
            "Single Cube Rotation",
            "Rotate the cube",
            "Cube orientation after rotation",
            "3D Cube Rotation",
            "Which cube matches?",
        ])
        fig.suptitle(title, fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_cube(self, ax, faces, title):
        x, y, z = 0, 0, 0
        top_pts = [
            _iso_project(x, y, z + 1),
            _iso_project(x + 1, y, z + 1),
            _iso_project(x + 1, y + 1, z + 1),
            _iso_project(x, y + 1, z + 1),
        ]
        ax.add_patch(Polygon(top_pts, closed=True, facecolor=faces["top"],
                             edgecolor="black", lw=1.2))
        left_pts = [
            _iso_project(x, y, z),
            _iso_project(x, y + 1, z),
            _iso_project(x, y + 1, z + 1),
            _iso_project(x, y, z + 1),
        ]
        ax.add_patch(Polygon(left_pts, closed=True, facecolor=faces["left"],
                             edgecolor="black", lw=1.2))
        right_pts = [
            _iso_project(x, y, z),
            _iso_project(x + 1, y, z),
            _iso_project(x + 1, y, z + 1),
            _iso_project(x, y, z + 1),
        ]
        ax.add_patch(Polygon(right_pts, closed=True, facecolor=faces["front"],
                             edgecolor="black", lw=1.2))

        pts = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    pts.append(_iso_project(dx, dy, dz))
        arr = np.array(pts)
        mg = 0.3
        ax.set_xlim(arr[:, 0].min() - mg, arr[:, 0].max() + mg)
        ax.set_ylim(arr[:, 1].min() - mg, arr[:, 1].max() + mg)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold")

if __name__ == "__main__":
    env = PolycubeRotationUltraEasyQA()
    for level in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": level})
            print(f"L{level} s{s}: ok={ok}, ans={env._answer}")
