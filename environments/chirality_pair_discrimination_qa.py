"""
Chirality Pair Discrimination QA.

2026-05-05 R5 B5A: REWRITTEN — pure single-letter MCQ A-D / A-E with
proper structural ease at L0/L1 (no answer-leak hints) and trap rate at
L8/L9.

Task: show 4 labeled shapes (A, B, C, D). Two of them are mirror images
of each other (chiral pair); the other two are unrelated. Pick the pair
from MCQ options.

Per-level layout
  L0/L1 — 4-MCQ. 2D mirror-letter task using highly asymmetric letters
          (R / G / J / P) drawn at large size with thick strokes. Easiest
          possible discrimination. No answer-leak hint; concise reasoning
          hint only.
  L2/L3 — 4-MCQ. 2D polyomino chirality (4-5 cell shapes, axis-aligned).
  L4/L5 — 4-MCQ. 2D polyomino + rotation disguise on the mirror partner.
  L6   — 4-MCQ. 3D polycubes (4-5 cubes), simple no-rotation disguise.
  L7   — 5-MCQ + 25% trap. 3D polycubes (5-6 cubes), 1 rotation disguise.
  L8/L9 — 5-MCQ + 30/40% trap. 3D polycubes (6-7 cubes), 2 rotations.

ALLOW_ROTATION = False — orientation-sensitive (mirror chirality is the
only signal; image rotation can transform a chiral pair into a different
visual layout).
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ---------------------------------------------------------------------- #
# Shared geometry helpers (3D and 2D)
# ---------------------------------------------------------------------- #
def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy


def _normalize3(cubes):
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    minx, miny, minz = min(xs), min(ys), min(zs)
    return tuple(sorted((x - minx, y - miny, z - minz) for (x, y, z) in cubes))


def _normalize_set3(cubes):
    return frozenset(_normalize3(cubes))


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


def _all_rotations3(cubes):
    seen = set()
    result = []
    base_rotations = [
        (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
        (0, 1, 0), (0, 3, 0),
    ]
    for rx, ry, rz in base_rotations:
        base = _rot_x(cubes, rx)
        base = _rot_y(base, ry)
        base = _rot_z(base, rz)
        for spin in range(4):
            r = _rot_z(base, spin)
            key = _normalize_set3(r)
            if key not in seen:
                seen.add(key)
                result.append(r)
    return result


def _shapes_equal3(a, b):
    b_norm = _normalize_set3(b)
    for r in _all_rotations3(a):
        if _normalize_set3(r) == b_norm:
            return True
    return False


def _mirror_x3(cubes):
    return [(-x, y, z) for (x, y, z) in cubes]


def _is_chiral3(cubes):
    return not _shapes_equal3(cubes, _mirror_x3(cubes))


def _random_polycube(rng, n):
    cubes = {(0, 0, 0)}
    tries = 0
    while len(cubes) < n and tries < 300:
        tries += 1
        seed_cube = rng.choice(list(cubes))
        dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0),
                                  (0, 1, 0), (0, -1, 0),
                                  (0, 0, 1)])
        nxt = (seed_cube[0] + dx, seed_cube[1] + dy, seed_cube[2] + dz)
        cubes.add(nxt)
    return list(cubes)


# ---------------------------------------------------------------------- #
# 2D polyomino utilities
# ---------------------------------------------------------------------- #
_2D_CHIRAL = [
    [(0, 0), (1, 0), (2, 0), (0, 1)],            # L
    [(0, 0), (1, 0), (1, 1), (2, 1)],            # S
    [(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)],    # U with arm
    [(0, 0), (1, 0), (2, 0), (1, 1), (0, 1)],    # T variant
]


def _norm2(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return tuple(sorted((x - min(xs), y - min(ys)) for (x, y) in cells))


def _reflect2(cells):
    return [(-x, y) for (x, y) in cells]


def _rot2(cells, k):
    out = list(cells)
    for _ in range(k % 4):
        out = [(-y, x) for (x, y) in out]
    return out


# ---------------------------------------------------------------------- #
# Letter shapes for L0/L1 — drawn as filled paths via line bundles
# ---------------------------------------------------------------------- #
# Each "letter" is a list of (x, y) line segments — chiral letters where
# the mirror image is visibly different from the original.
# We use simple block-letter strokes on a 5x7 grid.
_LETTERS_CHIRAL = ["R", "G", "J", "P", "L", "F"]

_LETTER_STROKES = {
    "R": [
        # Vertical bar
        ([(0, 0), (0, 6)], 0.55),
        # Top horizontal
        ([(0, 6), (3, 6)], 0.55),
        # Right side of bowl
        ([(3, 6), (4, 5)], 0.55),
        ([(4, 5), (4, 4)], 0.55),
        ([(4, 4), (3, 3)], 0.55),
        # Middle horizontal
        ([(0, 3), (3, 3)], 0.55),
        # Diagonal leg
        ([(2, 3), (4, 0)], 0.55),
    ],
    "G": [
        ([(4, 5), (3, 6)], 0.55),
        ([(3, 6), (1, 6)], 0.55),
        ([(1, 6), (0, 5)], 0.55),
        ([(0, 5), (0, 1)], 0.55),
        ([(0, 1), (1, 0)], 0.55),
        ([(1, 0), (3, 0)], 0.55),
        ([(3, 0), (4, 1)], 0.55),
        ([(4, 1), (4, 3)], 0.55),
        ([(4, 3), (2, 3)], 0.55),
    ],
    "J": [
        ([(4, 6), (4, 1)], 0.55),
        ([(4, 1), (3, 0)], 0.55),
        ([(3, 0), (1, 0)], 0.55),
        ([(1, 0), (0, 1)], 0.55),
    ],
    "P": [
        ([(0, 0), (0, 6)], 0.55),
        ([(0, 6), (3, 6)], 0.55),
        ([(3, 6), (4, 5)], 0.55),
        ([(4, 5), (4, 4)], 0.55),
        ([(4, 4), (3, 3)], 0.55),
        ([(0, 3), (3, 3)], 0.55),
    ],
    "L": [
        ([(0, 0), (0, 6)], 0.55),
        ([(0, 0), (4, 0)], 0.55),
    ],
    "F": [
        ([(0, 0), (0, 6)], 0.55),
        ([(0, 6), (4, 6)], 0.55),
        ([(0, 3), (3, 3)], 0.55),
    ],
}


def _letter_strokes_mirrored(letter: str):
    """Return strokes mirrored horizontally (negate x)."""
    base = _LETTER_STROKES[letter]
    out = []
    for (pts, lw) in base:
        out.append(([(-x, y) for (x, y) in pts], lw))
    return out


# ---------------------------------------------------------------------- #
# Question stem variants
# ---------------------------------------------------------------------- #
_Q_2D_TEMPLATES = [
    "Four 2D shapes labeled A, B, C, D are shown. Exactly one pair are mirror images (one is the horizontal reflection of the other); the other two are unrelated. Which pair is the mirror pair ( )?",
    "The image shows four labeled 2D shapes. Two are horizontal mirror images; the other two are unrelated. Identify the mirror pair ( ).",
    "Among the four labeled 2D shapes A-D, exactly one pair are reflections of each other. Which pair ( )?",
    "Look at the four shapes A, B, C, D. Two of them are mirror images (flipped horizontally). Which pair forms the mirror pair ( )?",
]

_Q_3D_TEMPLATES = [
    "The image shows four polycubes labeled A, B, C, D. Exactly one pair are mirror images (chiral pair) of each other; the other two are unrelated. Which pair is the mirror pair ( )?",
    "Four 3D polycubes labeled A-D are shown. Exactly one pair are enantiomers (mirror images, not rotations). Which pair ( )?",
    "Inspect the four polycubes A/B/C/D. Two of them are mirror images of each other. Identify the pair ( ).",
    "Among the four polycubes labeled A, B, C, D, identify the pair related by reflection (not rotation) ( ).",
]


class ChiralityPairDiscriminationQA(StandaloneVisualEnv):
    ENV_NAME = "chirality_pair_discrimination"
    ALLOW_ROTATION = False  # mirror chirality is the entire signal

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Pick one asymmetric feature (e.g. an L-arm direction) "
        "and compare. Final answer in the format this prompt requested."
    )

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"mode": "letters", "use_5_options": False, "trap_rate": 0.0}
        if level <= 3:
            return {"mode": "2d_polyomino", "rot_disguise": 0,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 5:
            return {"mode": "2d_polyomino", "rot_disguise": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 6:
            return {"mode": "3d", "n_cubes": 4, "rot_disguise": 0,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 7:
            return {"mode": "3d", "n_cubes": 5, "rot_disguise": 1,
                    "use_5_options": True, "trap_rate": 0.25}
        if level == 8:
            return {"mode": "3d", "n_cubes": 6, "rot_disguise": 2,
                    "use_5_options": True, "trap_rate": 0.30}
        # L9
        return {"mode": "3d", "n_cubes": 7, "rot_disguise": 2,
                "use_5_options": True, "trap_rate": 0.40}

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        for attempt in range(20):
            try:
                if cfg["mode"] == "letters":
                    r = self._try_generate_letters(cfg, level, attempt)
                elif cfg["mode"] == "2d_polyomino":
                    r = self._try_generate_2d(cfg, level, attempt)
                else:
                    r = self._try_generate_3d(cfg, level, attempt)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # Common option building
    # ------------------------------------------------------------------ #
    def _build_options(self, correct_pair_str: str, all_pairs: List[str],
                       cfg: Dict, rng: random.Random) -> Tuple[List[str], str]:
        """Returns (option_pair_strings, correct_letter)."""
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)
        wrong = [p for p in all_pairs if p != correct_pair_str]
        rng.shuffle(wrong)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            chosen = wrong[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = wrong[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [correct_pair_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(correct_pair_str)]
        return options, correct_letter

    def _format_question(self, q_template: str, options: List[str],
                         use_5: bool, level: int) -> str:
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        opt_lines = [f"{opt_letters[i]}. {options[i]}" for i in range(len(options))]
        if use_5:
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{q_template}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )
        if level <= 1:
            question += (
                "\nHint: a mirror image flips left/right. Track an "
                "asymmetric stroke (e.g. the leg of an R, the bowl of a P). "
                "If the stroke direction reverses between two shapes, they "
                "are mirror images."
            )
        elif level <= 3:
            question += (
                "\nHint: rotation preserves handedness; mirror flips it. "
                "Pick an asymmetric protrusion and compare its direction "
                "across the four shapes."
            )
        return question

    # ------------------------------------------------------------------ #
    # L0/L1 — letter mirror task
    # ------------------------------------------------------------------ #
    def _try_generate_letters(self, cfg: Dict, level: int, attempt: int):
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991
                             + attempt * 7919)

        # Pick base letter for the chiral pair
        base_letter = rng.choice(_LETTERS_CHIRAL)
        # Pick 2 other distinct chiral letters as distractors
        others = [l for l in _LETTERS_CHIRAL if l != base_letter]
        rng.shuffle(others)
        distractor_letters = others[:2]

        # Build 4 strokes — base, mirror(base), distractor1, distractor2
        items = [
            ("base", _LETTER_STROKES[base_letter], base_letter, False),
            ("mirror", _letter_strokes_mirrored(base_letter), base_letter, True),
            ("distract", _LETTER_STROKES[distractor_letters[0]],
             distractor_letters[0], False),
            ("distract", _LETTER_STROKES[distractor_letters[1]],
             distractor_letters[1], False),
        ]
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [items[i] for i in indices]
        labels = ["A", "B", "C", "D"]

        pair_label_set = sorted([labels[indices.index(0)],
                                  labels[indices.index(1)]])
        correct_pair_str = f"{pair_label_set[0]} and {pair_label_set[1]}"

        all_pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                all_pairs.append(f"{labels[i]} and {labels[j]}")

        opt_result = self._build_options(correct_pair_str, all_pairs, cfg, rng)
        if opt_result is None:
            return None
        options, correct_letter = opt_result

        # Render
        fig, axes = plt.subplots(1, 4, figsize=(13, 4.0), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        stroke_color = "#1a5276"
        for idx in range(4):
            ax = axes[idx]
            _, strokes, _ltr, _is_mir = shuffled[idx]
            for (pts, lw) in strokes:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=stroke_color, linewidth=8.0,
                        solid_capstyle='round', solid_joinstyle='round')
            # Compute bounds
            all_xs = [p[0] for s, _ in strokes for p in s]
            all_ys = [p[1] for s, _ in strokes for p in s]
            mnx, mxx = min(all_xs), max(all_xs)
            mny, mxy = min(all_ys), max(all_ys)
            pad = 1.2
            ax.set_xlim(mnx - pad, mxx + pad)
            ax.set_ylim(mny - pad, mxy + pad)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"({labels[idx]})", fontsize=18, fontweight="bold",
                         color="#1a1a1a")

        fig.suptitle("Which pair are mirror images?", fontsize=14,
                     fontweight="bold", color="#1a1a1a")
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=120)

        sidx = (self.seed or 0) % len(_Q_2D_TEMPLATES)
        question = self._format_question(_Q_2D_TEMPLATES[sidx], options,
                                         cfg["use_5_options"], level)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # L2-L5 — 2D polyomino chirality
    # ------------------------------------------------------------------ #
    def _try_generate_2d(self, cfg: Dict, level: int, attempt: int):
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991
                             + attempt * 7919)

        base = list(rng.choice(_2D_CHIRAL))
        mirror = _reflect2(base)

        # Verify chiral
        base_rots = {_norm2(_rot2(base, k)) for k in range(4)}
        if _norm2(mirror) in base_rots:
            return None

        if cfg["rot_disguise"]:
            mirror = _rot2(mirror, rng.randint(0, 3))

        # 2 unrelated shapes
        unrelated = []
        for _ in range(20):
            d = list(rng.choice(_2D_CHIRAL))
            d = _rot2(d, rng.randint(0, 3))
            if _norm2(d) in base_rots:
                continue
            if _norm2(d) == _norm2(mirror):
                continue
            if any(_norm2(d) == _norm2(u) for u in unrelated):
                continue
            unrelated.append(d)
            if len(unrelated) >= 2:
                break
        if len(unrelated) < 2:
            return None

        shapes = [base, mirror, unrelated[0], unrelated[1]]
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [shapes[i] for i in indices]
        labels = ["A", "B", "C", "D"]

        pair_label_set = sorted([labels[indices.index(0)],
                                  labels[indices.index(1)]])
        correct_pair_str = f"{pair_label_set[0]} and {pair_label_set[1]}"

        all_pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                all_pairs.append(f"{labels[i]} and {labels[j]}")

        opt_result = self._build_options(correct_pair_str, all_pairs, cfg, rng)
        if opt_result is None:
            return None
        options, correct_letter = opt_result

        # Render
        style = self._random_style()
        palette = style["palette"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        shape_color = palette[0]
        for idx in range(4):
            ax = axes[idx]
            cells = shuffled[idx]
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            mnx, mny = min(xs), min(ys)
            for (cx, cy) in cells:
                rec = plt.Rectangle((cx - mnx, cy - mny), 1, 1,
                                    facecolor=shape_color,
                                    edgecolor="black", lw=2)
                ax.add_patch(rec)
            ax.set_xlim(-0.5, max(xs) - mnx + 1.5)
            ax.set_ylim(-0.5, max(ys) - mny + 1.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"({labels[idx]})", fontsize=15, fontweight="bold")

        fig.suptitle("Which two shapes are mirror images?",
                     fontsize=13, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        sidx = (self.seed or 0) % len(_Q_2D_TEMPLATES)
        question = self._format_question(_Q_2D_TEMPLATES[sidx], options,
                                         cfg["use_5_options"], level)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # L6-L9 — 3D polycube chirality
    # ------------------------------------------------------------------ #
    def _try_generate_3d(self, cfg: Dict, level: int, attempt: int):
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991
                             + attempt * 7919)
        n_cubes = cfg["n_cubes"]

        base = None
        for _ in range(60):
            cand = _random_polycube(rng, n_cubes)
            if len(cand) == n_cubes and _is_chiral3(cand):
                base = cand
                break
        if base is None:
            return None

        mirror = _mirror_x3(base)
        if cfg["rot_disguise"] > 0:
            rots = _all_rotations3(mirror)
            rng.shuffle(rots)
            mirror = rots[0]

        unrelated = []
        for _ in range(80):
            cand = _random_polycube(rng, n_cubes)
            if len(cand) != n_cubes:
                continue
            if _shapes_equal3(cand, base) or _shapes_equal3(cand, mirror):
                continue
            if _shapes_equal3(_mirror_x3(cand), base):
                continue
            if any(_shapes_equal3(cand, u) for u in unrelated):
                continue
            unrelated.append(cand)
            if len(unrelated) >= 2:
                break
        if len(unrelated) < 2:
            return None

        shapes = [("pair", base), ("pair", mirror),
                  ("unrelated", unrelated[0]),
                  ("unrelated", unrelated[1])]
        rng.shuffle(shapes)
        shapes_vis = []
        for role, s in shapes:
            if role == "unrelated" or (role == "pair" and cfg["rot_disguise"] >= 2):
                rots = _all_rotations3(s)
                rng.shuffle(rots)
                shapes_vis.append((role, rots[0]))
            else:
                shapes_vis.append((role, s))

        labels = ["A", "B", "C", "D"]
        pair_indices = [i for i, (r, _) in enumerate(shapes_vis) if r == "pair"]
        if len(pair_indices) != 2:
            return None
        correct_pair = sorted(labels[i] for i in pair_indices)
        correct_pair_str = f"{correct_pair[0]} and {correct_pair[1]}"

        all_pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                all_pairs.append(f"{labels[i]} and {labels[j]}")

        opt_result = self._build_options(correct_pair_str, all_pairs, cfg, rng)
        if opt_result is None:
            return None
        options, correct_letter = opt_result

        img = self._render_3d(shapes_vis, labels)

        sidx = (self.seed or 0) % len(_Q_3D_TEMPLATES)
        question = self._format_question(_Q_3D_TEMPLATES[sidx], options,
                                         cfg["use_5_options"], level)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # 3D polycube rendering
    # ------------------------------------------------------------------ #
    def _draw_polycube(self, ax, cubes, palette, title):
        shifted = sorted(_normalize3(cubes),
                         key=lambda c: -(c[0] + c[1] - c[2]))
        top_color = palette[0]
        left_color = palette[1 % len(palette)]
        right_color = palette[2 % len(palette)]

        for (x, y, z) in shifted:
            pts_top = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            ax.add_patch(Polygon(pts_top, closed=True, facecolor=top_color,
                                 edgecolor="black", lw=1.1))
            pts_left = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts_left, closed=True, facecolor=left_color,
                                 edgecolor="black", lw=1.1))
            pts_right = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts_right, closed=True, facecolor=right_color,
                                 edgecolor="black", lw=1.1))

        pts = []
        for (x, y, z) in shifted:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        pts.append(_iso_project(x + dx, y + dy, z + dz))
        if pts:
            arr = np.array(pts)
            margin = 0.4
            ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
            ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=4)

    def _render_3d(self, shapes_vis, labels) -> Image.Image:
        style = self._random_style()
        palette = style["palette"]
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        pal = list(palette)
        for i, (role, s) in enumerate(shapes_vis):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            ax.set_facecolor("#ffffff")
            self._draw_polycube(ax, s, pal, title=labels[i])
        fig.suptitle("Which two polycubes are mirror images?",
                     fontsize=14, fontweight="bold")
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = ChiralityPairDiscriminationQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   pos={v_pos['accuracy']} neg={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
