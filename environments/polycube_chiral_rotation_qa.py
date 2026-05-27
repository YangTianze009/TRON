"""
Polycube Chiral Rotation QA — pure single-letter MCQ.

2026-05-05 R5 Batch 4 (TOO-HARD ease + restructure):
Old design (779 LOC) had wrapper-mismatch (template said `<answer>...</answer>`
explicitly) and mixed 2D/3D modes with confusing question types. R3 = 2.5%.

REWRITE strategy:
  - L0/L1: 2D — render TWO 3-cube polyominoes side by side. 4-MCQ
            {A. Yes (rotation)  B. No (mirror)  C. Cannot tell  D. No correct}
            "Are shapes A and B the same shape (one rotation maps A to B,
             without reflection)?"
  - L2-L4: 4-cube 2D polyominoes. Same Yes/No structure (rotation vs mirror).
  - L5-L7: 5-cube 3D polycubes; rotation MCQ with axis-labeled views.
  - L8/L9: 6-cube 3D polycubes; 5-MCQ with 30-40% trap.

Pure single-letter MCQ A-D / A-E. ALLOW_ROTATION=False.
No wrapper text in question — framework wraps.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ----------------------------------------------------------------------- #
# 2D polyomino helpers
# ----------------------------------------------------------------------- #

def _rot2d(cells: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
    """Rotate 90*k degrees in xy plane."""
    out = list(cells)
    for _ in range(k % 4):
        out = [(-y, x) for (x, y) in out]
    return out


def _reflect2d(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(-x, y) for (x, y) in cells]


def _norm2d(cells: List[Tuple[int, int]]) -> Tuple:
    if not cells:
        return tuple()
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    mnx, mny = min(xs), min(ys)
    return tuple(sorted((x - mnx, y - mny) for (x, y) in cells))


def _all_rots2d(cells: List[Tuple[int, int]]) -> List[Tuple]:
    return [_norm2d(_rot2d(cells, k)) for k in range(4)]


def _is_rotation_of(a: List[Tuple[int, int]],
                    b: List[Tuple[int, int]]) -> bool:
    """Is b a pure rotation of a (no reflection)?"""
    bn = _norm2d(b)
    return bn in _all_rots2d(a)


def _is_chiral2d(cells: List[Tuple[int, int]]) -> bool:
    """A polyomino is chiral if its reflection is not equal to any rotation."""
    return _norm2d(_reflect2d(cells)) not in _all_rots2d(cells)


# 2D polyomino library (all chiral examples). Each tuple contains the cells.
_POLY2D_3 = [
    [(0, 0), (1, 0), (0, 1)],   # L (3-cell)
]

_POLY2D_4 = [
    [(0, 0), (1, 0), (2, 0), (0, 1)],  # L tetromino (chiral)
    [(0, 0), (1, 0), (2, 0), (2, 1)],  # J tetromino (chiral mirror of L)
    [(0, 0), (1, 0), (1, 1), (2, 1)],  # S tetromino (chiral)
    [(0, 1), (1, 1), (1, 0), (2, 0)],  # Z tetromino (chiral mirror of S)
]

_POLY2D_5 = [
    [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)],  # L pentomino
    [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],  # T pentomino (NOT chiral)
    [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)],  # U pentomino
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],  # W pentomino
]


def _gen_chiral_poly2d(rng, n: int):
    """Pick a chiral 2D polyomino with n cells."""
    pool = {3: _POLY2D_3, 4: _POLY2D_4, 5: _POLY2D_5}.get(n, _POLY2D_4)
    chiral_pool = [p for p in pool if _is_chiral2d(p)]
    if not chiral_pool:
        chiral_pool = pool
    return list(rng.choice(chiral_pool))


# ----------------------------------------------------------------------- #
# 3D polycube helpers
# ----------------------------------------------------------------------- #

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy


def _rot_x(cubes, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            y, z = -z, y
        out.append((x, y, z))
    return out


def _rot_y(cubes, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            x, z = z, -x
        out.append((x, y, z))
    return out


def _rot_z(cubes, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            x, y = -y, x
        out.append((x, y, z))
    return out


def _mirror3(cubes, axis):
    if axis == "x":
        return [(-x, y, z) for (x, y, z) in cubes]
    if axis == "y":
        return [(x, -y, z) for (x, y, z) in cubes]
    return [(x, y, -z) for (x, y, z) in cubes]


def _normalize3(cubes):
    if not cubes:
        return frozenset()
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    mx, my, mz = min(xs), min(ys), min(zs)
    return frozenset((x - mx, y - my, z - mz) for (x, y, z) in cubes)


def _all_rots3d(cubes):
    seen = set()
    out = []
    for rx in range(4):
        for ry in range(4):
            for rz in range(4):
                r = _rot_x(cubes, rx)
                r = _rot_y(r, ry)
                r = _rot_z(r, rz)
                key = _normalize3(r)
                if key not in seen:
                    seen.add(key)
                    out.append(r)
    return out


def _is_rotation_of_3d(a, b) -> bool:
    bn = _normalize3(b)
    return any(_normalize3(r) == bn for r in _all_rots3d(a))


def _is_chiral3d(cubes) -> bool:
    return not _is_rotation_of_3d(cubes, _mirror3(cubes, "x"))


# 3D polycube library
def _gen_l_polycube(n: int):
    arm = max(2, n - 1)
    cubes = [(i, 0, 0) for i in range(arm)]
    cubes.append((arm - 1, 1, 0))
    while len(cubes) < n:
        cubes.append((arm - 1, 1, len(cubes) - arm))
    return cubes[:n]


def _gen_t_polycube(n: int):
    cubes = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)]
    if n >= 5:
        cubes.append((0, 0, 1))
    if n >= 6:
        cubes.append((2, 1, 0))
    return cubes[:n]


def _gen_s_polycube(n: int):
    cubes = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]
    if n >= 5:
        cubes.append((0, 0, 1))
    if n >= 6:
        cubes.append((2, 1, 1))
    return cubes[:n]


def _gen_step_polycube(n: int):
    cubes = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)]
    if n >= 5:
        cubes.append((2, 1, 1))
    if n >= 6:
        cubes.append((2, 1, 2))
    return cubes[:n]


def _gen_chiral_polycube(rng, n: int):
    pool = [_gen_l_polycube, _gen_t_polycube, _gen_s_polycube,
            _gen_step_polycube]
    rng.shuffle(pool)
    for fn in pool:
        cubes = fn(n)
        if len(cubes) == n and _is_chiral3d(cubes):
            return cubes
    # Fallback
    return _gen_s_polycube(n)


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #

class PolycubeChiralRotationQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — distinguish rotation from mirror reflection."""

    ALLOW_ROTATION = False
    ENV_NAME = "polycube_chiral_rotation"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"mode": "2d", "n": 3,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 4:
            return {"mode": "2d", "n": 4,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 6:
            return {"mode": "3d", "n": 5,
                    "use_5_options": True, "trap_rate": 0.15}
        if level == 7:
            return {"mode": "3d", "n": 5,
                    "use_5_options": True, "trap_rate": 0.25}
        if level == 8:
            return {"mode": "3d", "n": 6,
                    "use_5_options": True, "trap_rate": 0.35}
        return {"mode": "3d", "n": 6,
                "use_5_options": True, "trap_rate": 0.40}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        for _ in range(40):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        if cfg["mode"] == "2d":
            return self._gen_2d(cfg, level, rng)
        return self._gen_3d(cfg, level, rng)

    # -------------------- 2D mode (L0-L4) -------------------- #
    def _gen_2d(self, cfg, level, rng):
        n = cfg["n"]
        if n == 3:
            # 3-cube polyominoes are NOT chiral, so we ask the simpler
            # question: "Is B the same SHAPE as A (allowing rotation)?"
            # with 50/50 split: same-shape (rotation) vs different-shape.
            shapes_3 = [
                [(0, 0), (1, 0), (0, 1)],   # L
                [(0, 0), (1, 0), (2, 0)],   # I (straight)
            ]
            shape_a = list(rng.choice(shapes_3))
            same_shape = rng.random() < 0.5
            if same_shape:
                k = rng.choice([1, 2, 3])
                shape_b = _rot2d(shape_a, k)
                is_yes = True
            else:
                other = [s for s in shapes_3 if _norm2d(s) != _norm2d(shape_a)]
                shape_b = list(rng.choice(other))
                shape_b = _rot2d(shape_b, rng.choice([0, 1, 2, 3]))
                is_yes = False
            base = shape_a
            shown_b = shape_b
        else:
            base = _gen_chiral_poly2d(rng, n)
            if not _is_chiral2d(base):
                return None
            is_rotation = rng.random() < 0.5
            if is_rotation:
                k = rng.choice([1, 2, 3])
                shown_b = _rot2d(base, k)
                is_yes = True
            else:
                shown_b = _reflect2d(base)
                shown_b = _rot2d(shown_b, rng.choice([0, 1, 2, 3]))
                is_yes = False

        # Build options
        base_options = [
            "Yes (one rotation maps A to B)",
            "No (B cannot be obtained by rotation alone)",
            "Cannot determine from the diagram",
            "None of the above",
        ]
        true_text = base_options[0] if is_yes else base_options[1]
        opt_letters = ["A", "B", "C", "D"]
        opt_lines = list(base_options)
        rng.shuffle(opt_lines)
        correct_letter = opt_letters[opt_lines.index(true_text)]

        # Render
        img = self._render_2d_pair(base, shown_b, rng)

        opt_text = "\n".join(f"{opt_letters[i]}. {opt_lines[i]}"
                             for i in range(4))
        if n == 3:
            stem = (
                "The diagram shows two 2D shapes A and B made of unit "
                "squares. Can shape A be rotated in the 2D plane (by 0, "
                "90, 180, or 270 degrees) so that it matches shape B "
                "exactly? (Two shapes are different if they are made up of "
                "different cube arrangements.) Be concise."
            )
        else:
            stem = (
                "The diagram shows two 2D shapes A and B made of unit "
                "squares. Without flipping (no mirror reflection allowed), "
                "can shape A be rotated in the 2D plane (by 0, 90, 180, or "
                "270 degrees) to match shape B exactly?"
            )
        if level <= 1 and n != 3:
            stem += (
                " Hint: if A and B are mirror images of each other, no "
                "in-plane rotation maps A to B (only flipping does)."
            )

        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter A, B, C, or D."
        )
        return question, correct_letter, img

    # -------------------- 3D mode (L5+) -------------------- #
    def _gen_3d(self, cfg, level, rng):
        original = _gen_chiral_polycube(rng, cfg["n"])
        if not _is_chiral3d(original):
            return None

        axis = rng.choice(["x", "y", "z"])
        angle = rng.choice([90, 180, 270])
        rot_fn = {"x": _rot_x, "y": _rot_y, "z": _rot_z}[axis]
        rotated = rot_fn(original, angle // 90)

        # Build options: correct = rotated; distractors = mirror + perturbations
        options = [rotated]

        # Distractor 1: mirror impostor
        mirror_axis = rng.choice(["x", "y", "z"])
        mirrored = _mirror3(original, mirror_axis)
        if not _is_rotation_of_3d(mirrored, original):
            options.append(mirrored)

        # Distractor 2-4: perturbed shapes (different number of cubes or
        # different connectivity)
        for _ in range(20):
            if len(options) >= 4:
                break
            n = cfg["n"]
            cand = _gen_chiral_polycube(rng, n + rng.choice([-1, 1, 1]))
            if any(_is_rotation_of_3d(cand, o) for o in options):
                continue
            options.append(cand)

        if len(options) < 4:
            return None
        options = options[:4]

        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4
        opt_letters = ["A", "B", "C", "D", "E"][:n_value]
        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            # Replace correct option with a distractor
            wrong_pool = options[1:]  # mirror + perturb
            extras = []
            for _ in range(20):
                if len(wrong_pool) + len(extras) >= n_value - 1:
                    break
                cand = _gen_chiral_polycube(rng, cfg["n"])
                if not any(_is_rotation_of_3d(cand, w) for w in wrong_pool + extras + [rotated]):
                    extras.append(cand)
            all_wrong = wrong_pool + extras
            if len(all_wrong) < n_value - 1:
                return None
            chosen = all_wrong[:n_value - 1]
            rng.shuffle(chosen)
            opt_pieces = chosen
            correct_letter = opt_letters[-1]
        else:
            if use_5:
                # Need 4 visual options + "No correct answer" 5th
                pieces = list(options)
                rng.shuffle(pieces)
                correct_idx = next(i for i, p in enumerate(pieces)
                                   if _normalize3(p) == _normalize3(rotated))
                correct_letter = opt_letters[correct_idx]
                opt_pieces = pieces
            else:
                pieces = list(options)
                rng.shuffle(pieces)
                correct_idx = next(i for i, p in enumerate(pieces)
                                   if _normalize3(p) == _normalize3(rotated))
                correct_letter = opt_letters[correct_idx]
                opt_pieces = pieces

        # Render
        img = self._render_3d(original, opt_pieces, rng,
                              opt_letters[:len(opt_pieces)])

        # Build option lines
        opt_lines_text = [f"Option {opt_letters[i]} (see diagram)"
                          for i in range(len(opt_pieces))]
        if use_5:
            opt_lines_text.append("No correct answer")
        opt_text = "\n".join(opt_lines_text[:n_value])
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        axis_label = {"x": "x-axis", "y": "y-axis", "z": "z-axis (vertical)"}[axis]
        stem = (
            f"The top image shows an ORIGINAL 3D polycube. Apply a proper "
            f"3D rotation of {angle} degrees about the {axis_label} (no "
            f"reflection). Which option below shows the resulting polycube?"
        )

        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, img

    # -------------------- Rendering -------------------- #
    def _render_2d_pair(self, a_cells, b_cells, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = [c for c in style["palette"]
                   if c.lower() not in ('#000000', '#ffffff')]
        if not palette:
            palette = ["#5dade2", "#48c9b0"]
        color_a = rng.choice(palette)
        color_b = rng.choice(palette)

        fig, axes = plt.subplots(1, 2, figsize=(8 * sc, 4 * sc),
                                 dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])

        def draw(ax, cells, color, title):
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            mnx, mny = min(xs), min(ys)
            for (cx, cy) in cells:
                r = Rectangle((cx - mnx, cy - mny), 1, 1,
                              facecolor=color, edgecolor="#222",
                              linewidth=2.0)
                ax.add_patch(r)
            ax.set_xlim(-0.5, max(xs) - mnx + 1.5)
            ax.set_ylim(-0.5, max(ys) - mny + 1.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontweight="bold")

        draw(axes[0], a_cells, color_a, "Shape A")
        draw(axes[1], b_cells, color_b, "Shape B")

        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_3d(self, original, options, rng, opt_letters) -> Image.Image:
        import colorsys
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = [c for c in style["palette"]
                   if c.lower() not in ('#000000', '#ffffff')]
        if not palette:
            palette = ["#5dade2"]
        base = rng.choice(palette)
        h = base.lstrip("#")
        r, g, b = (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
                   int(h[4:6], 16) / 255)
        hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
        ss = min(1.0, max(0.45, ss))
        top_rgb = colorsys.hls_to_rgb(hh, 0.77, ss)
        right_rgb = colorsys.hls_to_rgb(hh, 0.55, ss)
        left_rgb = colorsys.hls_to_rgb(hh, 0.30, ss)
        def _hx(rgb):
            return '#{:02x}{:02x}{:02x}'.format(
                int(max(0, min(1, rgb[0])) * 255),
                int(max(0, min(1, rgb[1])) * 255),
                int(max(0, min(1, rgb[2])) * 255))
        face_pal = [_hx(top_rgb), _hx(left_rgb), _hx(right_rgb)]

        n_opt = len(options)
        cols = max(2, n_opt)
        fig = plt.figure(figsize=(cols * 2.7 * sc, 6.5 * sc), dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])

        ax_orig = fig.add_subplot(2, cols, 1)
        ax_orig.set_aspect("equal")
        ax_orig.axis("off")
        self._draw_polycube_iso(ax_orig, original, face_pal)
        ax_orig.set_title("ORIGINAL", fontsize=14, fontweight="bold")

        for i in range(2, cols):
            ax = fig.add_subplot(2, cols, i + 1)
            ax.axis("off")

        for i, opt in enumerate(options):
            ax = fig.add_subplot(2, cols, cols + i + 1)
            ax.set_aspect("equal")
            ax.axis("off")
            self._draw_polycube_iso(ax, opt, face_pal)
            ax.set_title(f"({opt_letters[i]})", fontsize=13, fontweight="bold")

        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_polycube_iso(self, ax, cubes, palette):
        if not cubes:
            return
        sorted_cubes = sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))
        top, left, right = palette
        for (x, y, z) in sorted_cubes:
            face_top = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            face_left = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            face_right = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(face_top, closed=True, facecolor=top,
                                 edgecolor="#222", lw=1.0))
            ax.add_patch(Polygon(face_left, closed=True, facecolor=left,
                                 edgecolor="#222", lw=1.0))
            ax.add_patch(Polygon(face_right, closed=True, facecolor=right,
                                 edgecolor="#222", lw=1.0))
        # Auto-scale
        pts = []
        for (x, y, z) in cubes:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        pts.append(_iso_project(x + dx, y + dy, z + dz))
        if pts:
            arr = np.array(pts)
            margin = 0.4
            ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
            ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)


if __name__ == "__main__":
    env = PolycubeChiralRotationQA()
    for lv in [0, 3, 6, 9]:
        for s in [1, 7, 42]:
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, answer={env._answer}")
