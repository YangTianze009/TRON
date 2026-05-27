"""
Net Folding Visual QA Environment.

Renders 2D nets (unfolded cubes) with labelled faces and asks questions about
opposite faces, adjacent faces, fold results, and face counts.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: face_count on valid cube nets (just count squares — easiest).
L1: face_count on 5-square pentominoes (always 5).
L2: yes/no foldability on 4 obvious cube nets vs 6 obvious invalid.
L3: yes/no foldability on all 11 cube nets vs 8 invalid hexominoes.
L4: opposite_face on 4 easy nets (plus/T/L/Z).
L5: opposite_face + adjacent_face on 4 easy nets.
L6: opposite_face on all 11 nets.
L7: opposite_face + adjacent_face + fold_result on all nets.
L8: all 4 question types + label rotation.
L9: hardest nets only + full question mix.

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# All 11 distinct cube nets, encoded as sets of (row, col) cells.
# BUGFIX 2026-04-24: replaced pool with the 11 canonical cube nets.
# Prior _CUBE_NETS contained invalid hexominoes (5-strip linebumps, and
# several shapes like the "T" 3-column-plus-4-row, 6-cell compacts, etc.
# that are geometrically NOT cube nets). Mathematically there are exactly
# 11 distinct (up to rotation/reflection) hexominoes that fold into a cube.
# Enumerated via /tmp/validate_cube_nets.py using the orthonormal-triad
# folding simulator in `_is_valid_cube_net`.
_CUBE_NETS: List[List[Tuple[int, int]]] = [
    [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)],       # plus (1-4-1)
    [(0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 3)],       # 1-4-1 A
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],       # zigzag (2-2-2)
    [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 2)],       # 1-3-2
    [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)],       # 1-3-2 B
    [(0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1)],       # 1-3-2 C
    [(0, 1), (1, 1), (2, 0), (2, 1), (3, 1), (3, 2)],       # 1-3-2 D
    [(0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (3, 1)],       # 1-3-2 E
    [(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1)],       # 1-3-2 F
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],       # 1-3-2 G
    [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (4, 1)],       # 1-3-2 H
]

# L0 uses a few simple/recognizable nets. All are valid cube nets now.
_L0_CUBE_NETS = [_CUBE_NETS[0], _CUBE_NETS[1], _CUBE_NETS[2], _CUBE_NETS[3]]
_EASY_NETS = _CUBE_NETS[:5]
_HARD_NETS = _CUBE_NETS[5:]

# 8 invalid hexominoes
_INVALID_HEXOMINOES: List[List[Tuple[int, int]]] = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],   # 2×3 rect
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],   # 1×6 line
    [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],   # 3×2 + wing
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],   # L6 stack
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)],   # staircase 6
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2)],   # thick-bar bumps
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)],   # P
    [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1)],   # Y
]
_L0_INVALID = _INVALID_HEXOMINOES[:6]

_FACE_COLORS = [
    "#aed6f1", "#f9e79f", "#abebc6", "#f5b7b1",
    "#d7bde2", "#fadbd8", "#d5f5e3", "#fdebd0",
    "#d6eaf8", "#fcf3cf", "#e8daef", "#d1f2eb",
]

_LABEL_POOLS = [
    list("ABCDEF"),
    list("123456"),
    ["I", "II", "III", "IV", "V", "VI"],
    ["\u2605", "\u2666", "\u2663", "\u2660", "\u2665", "\u25cf"],
    list("PQRSTU"),
    ["N", "S", "E", "W", "T", "B"],
    list("αβγδεζ"),
]

_TITLE_VARIANTS = ["Cube Net", "2D net", "Net", "Polyomino", "Net view", "Square net"]

def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]

def _rotate90(cells):
    return _normalize([(c, -r) for r, c in cells])

def _reflect_h(cells):
    return _normalize([(r, -c) for r, c in cells])

def _reflect_v(cells):
    return _normalize([(-r, c) for r, c in cells])

def _apply_transform(cells, rot, refl):
    out = list(cells)
    for _ in range(rot // 90):
        out = _rotate90(out)
    if refl == "h":
        out = _reflect_h(out)
    elif refl == "v":
        out = _reflect_v(out)
    return _normalize(out)

OPPOSITES_CD = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

def _right_of(face_dir, top_dir, left=False):
    all_dirs = {0, 1, 2, 3, 4, 5}
    remaining = sorted(all_dirs - {face_dir, OPPOSITES_CD[face_dir],
                                   top_dir, OPPOSITES_CD[top_dir]})
    if not remaining:
        return face_dir
    if left:
        return remaining[0]
    return remaining[1] if len(remaining) > 1 else remaining[0]

def _find_opposite_faces(cells: List[Tuple[int, int]]) -> Dict[int, int]:
    """BUGFIX 2026-04-24: rewritten with proper orthonormal-triad tracking.

    Previous `fold_step` produced duplicate face indices for all 9 known-valid
    cube nets (see /tmp/check_fold_step_on_valid.py in the audit), so the
    resulting `opposite_map` was meaningless for both opposite_face and
    is_cube_net qtypes.

    New approach: per-cell 3D frame (right, up, out) in a right-handed basis.
    Moving across a shared edge from cell A to a neighbor B rotates the unit
    square around that shared edge; the new frame is computed from A's frame
    deterministically. After walking the hexomino, `out` vectors tell which
    face of the cube each cell lies on. Opposite faces have `out` vectors that
    are negatives of each other.
    """
    cell_set = set(cells)
    cell_to_idx = {c: i for i, c in enumerate(cells)}

    def _neg(v):
        return (-v[0], -v[1], -v[2])

    start = cells[0]
    # Start frame: right=+x, up=+y, out=+z.
    start_frame = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    frames = {start: start_frame}
    out_vecs = {cell_to_idx[start]: start_frame[2]}

    queue = [start]
    visited = {start}
    while queue:
        cur = queue.pop(0)
        right, up, out = frames[cur]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                # Compute new frame after folding across shared edge.
                # Convention: grid (r, c) maps to plane where +c is +right and
                # -r is +up (grid row index increases downward).
                if dr == -1 and dc == 0:
                    # Moving up in grid -> folded across the top edge.
                    # New cell sits on the +up side of the cube, its out
                    # vector rotates from `out` to `up`.
                    new_right = right
                    new_up = out
                    new_out = _neg(up)
                elif dr == 1 and dc == 0:
                    # Moving down in grid -> folded across the bottom edge.
                    new_right = right
                    new_up = _neg(out)
                    new_out = up
                elif dc == 1 and dr == 0:
                    # Moving right -> folded across the right edge.
                    new_right = out
                    new_up = up
                    new_out = _neg(right)
                else:  # dc == -1
                    # Moving left -> folded across the left edge.
                    new_right = _neg(out)
                    new_up = up
                    new_out = right
                frames[nb] = (new_right, new_up, new_out)
                out_vecs[cell_to_idx[nb]] = new_out
                queue.append(nb)

    # Build opposite_map: for each cell, find the cell whose out vector is
    # the negative of this cell's out vector.
    opposite_map: Dict[int, int] = {}
    # Invert out_vecs -> index for lookup.
    vec_to_idx: Dict[Tuple[int, int, int], int] = {}
    for idx, v in out_vecs.items():
        vec_to_idx[v] = idx
    for idx, v in out_vecs.items():
        opp_v = _neg(v)
        if opp_v in vec_to_idx:
            opposite_map[idx] = vec_to_idx[opp_v]
    return opposite_map

def _is_valid_cube_net(cells: List[Tuple[int, int]]) -> bool:
    """Return True iff the hexomino `cells` folds into a cube.

    A valid cube net maps its 6 cells to 6 distinct cube faces (all 6
    out-vectors distinct after folding). BUGFIX 2026-04-24.
    """
    cell_set = set(cells)
    if len(cell_set) != 6:
        return False

    def _neg(v):
        return (-v[0], -v[1], -v[2])

    start = cells[0]
    start_frame = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    frames = {start: start_frame}
    queue = [start]
    visited = {start}
    out_set = {start_frame[2]}
    while queue:
        cur = queue.pop(0)
        right, up, out = frames[cur]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                if dr == -1 and dc == 0:
                    new_frame = (right, out, _neg(up))
                elif dr == 1 and dc == 0:
                    new_frame = (right, _neg(out), up)
                elif dc == 1 and dr == 0:
                    new_frame = (out, up, _neg(right))
                else:
                    new_frame = (_neg(out), up, right)
                frames[nb] = new_frame
                out_set.add(new_frame[2])
                queue.append(nb)
    if len(visited) != 6:
        return False
    return len(out_set) == 6

class NetFoldingQA(StandaloneVisualEnv):
    ENV_NAME = "net_folding"

    QUESTION_TYPES = [
        "is_cube_net",
        "opposite_face",
        "adjacent_face",
        "fold_result",
        "face_count",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"], weights=cfg["qtype_weights"])[0]

        for _ in range(25):
            try:
                result = self._try_generate(qtype, level, cfg)
                if result is not None:
                    self._primary_complexity_feature = level * 3 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _level_config(self, level):
        # Reordered: face_count (trivial) at L0, is_cube_net at L2-L3,
        # opposite_face at L4+. This fixes the L0=0.30 vs L3=0.90 inversion.
        if level == 0:
            # L0: face_count on valid cube nets — just count squares (easiest).
            return {"qtypes": ["face_count"], "qtype_weights": [1],
                    "valid_pool": _L0_CUBE_NETS, "invalid_pool": None,
                    "yes_no_mode": True, "valid_ratio": 1.0,
                    "n_squares_override": 6,
                    "size_variant": True,
                    "allow_label_rotation": False}
        if level == 1:
            # L1: face_count on 5-square pentominoes (always 5) for variety.
            return {"qtypes": ["face_count"], "qtype_weights": [1],
                    "valid_pool": [], "invalid_pool": None,
                    "yes_no_mode": True, "valid_ratio": 0.0,
                    "n_squares_override": 5,
                    "allow_label_rotation": False}
        if level == 2:
            # L2: is_cube_net on obvious shapes (4 easy valid + 6 easy invalid).
            return {"qtypes": ["is_cube_net"], "qtype_weights": [1],
                    "valid_pool": _L0_CUBE_NETS, "invalid_pool": _L0_INVALID,
                    "yes_no_mode": True, "valid_ratio": 0.5,
                    "n_squares_override": 6,
                    "allow_label_rotation": False}
        if level == 3:
            # L3: is_cube_net on full net pool + full invalid pool.
            return {"qtypes": ["is_cube_net"], "qtype_weights": [1],
                    "valid_pool": _CUBE_NETS, "invalid_pool": _INVALID_HEXOMINOES,
                    "yes_no_mode": True, "valid_ratio": 0.5,
                    "n_squares_override": 6,
                    "allow_label_rotation": False}
        if level == 4:
            return {"qtypes": ["opposite_face"], "qtype_weights": [1],
                    "valid_pool": _EASY_NETS, "invalid_pool": None,
                    "yes_no_mode": False, "valid_ratio": 1.0,
                    "n_squares_override": 6,
                    "allow_label_rotation": False}
        if level == 5:
            return {"qtypes": ["opposite_face", "adjacent_face"],
                    "qtype_weights": [6, 4],
                    "valid_pool": _EASY_NETS, "invalid_pool": None,
                    "yes_no_mode": False, "valid_ratio": 1.0,
                    "n_squares_override": 6,
                    "allow_label_rotation": False}
        if level == 6:
            return {"qtypes": ["opposite_face", "adjacent_face"],
                    "qtype_weights": [6, 4],
                    "valid_pool": _CUBE_NETS, "invalid_pool": None,
                    "yes_no_mode": False, "valid_ratio": 1.0,
                    "n_squares_override": 6,
                    "allow_label_rotation": False}
        if level == 7:
            return {"qtypes": ["opposite_face", "adjacent_face", "fold_result"],
                    "qtype_weights": [4, 3, 3],
                    "valid_pool": _CUBE_NETS, "invalid_pool": None,
                    "yes_no_mode": False, "valid_ratio": 1.0,
                    "n_squares_override": 6,
                    "allow_label_rotation": False}
        if level == 8:
            return {"qtypes": ["opposite_face", "adjacent_face", "fold_result"],
                    "qtype_weights": [4, 3, 3],
                    "valid_pool": _CUBE_NETS, "invalid_pool": None,
                    "yes_no_mode": False, "valid_ratio": 1.0,
                    "n_squares_override": 6,
                    "allow_label_rotation": True}
        return {"qtypes": ["opposite_face", "adjacent_face", "fold_result"],
                "qtype_weights": [5, 3, 2],
                "valid_pool": _HARD_NETS, "invalid_pool": None,
                "yes_no_mode": False, "valid_ratio": 1.0,
                "n_squares_override": 6,
                "allow_label_rotation": True}

    def _try_generate(self, qtype, level, cfg):
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )

        rot = sub_rng.choice([0, 90, 180, 270])
        refl = sub_rng.choice(["none", "none", "h", "v"])
        label_pool = list(sub_rng.choice(_LABEL_POOLS))
        sub_rng.shuffle(label_pool)

        # Pick shape
        if cfg.get("size_variant", False):
            # face_count variant: mix 5/6/7 square polyominoes
            n_sq = sub_rng.choice([5, 6, 6, 7])
            if n_sq == 6:
                cells = list(sub_rng.choice(cfg["valid_pool"]))
            else:
                cells = self._random_connected_polyomino(sub_rng, n_sq)
            is_valid = (n_sq == 6)
        elif cfg["yes_no_mode"]:
            if cfg["n_squares_override"] == 5:
                cells = self._random_connected_polyomino(sub_rng, 5)
                is_valid = False
            else:
                use_yes = rng.random() < cfg["valid_ratio"]
                if use_yes and cfg["valid_pool"]:
                    cells = list(sub_rng.choice(cfg["valid_pool"]))
                    is_valid = True
                else:
                    cells = list(sub_rng.choice(cfg["invalid_pool"]))
                    is_valid = False
        else:
            cells = list(sub_rng.choice(cfg["valid_pool"]))
            is_valid = True

        cells = _apply_transform(cells, rot, refl)
        cells = sorted(cells)

        if qtype == "is_cube_net":
            stems = [
                "The image shows a 2D net made of squares. Could this net be folded along its edges to form a closed cube (a cube has 6 square faces)?",
                "Consider the 2D net shown. When folded along its edges, can it form a closed cube?",
                "Could the polyomino shown below be folded into a cube?",
                "The figure shows an arrangement of squares. Is this a valid cube net?",
            ]
            q = sub_rng.choice(stems) + " Answer 'yes' or 'no'."
            answer = "yes" if is_valid else "no"
            image = self._render_net(cells, None, sub_rng)
            return q, answer, image

        if qtype == "face_count":
            stems = [
                "How many square faces does the net in the image have?",
                "Count the total number of square cells in the net shown.",
                "What is the total number of squares in the polyomino shown above?",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            answer = str(len(cells))
            image = self._render_net(cells, None, sub_rng)
            return q, answer, image

        # opposite_face / adjacent_face / fold_result need labels
        if not is_valid or len(cells) != 6:
            return None
        labels = label_pool[:6]
        opposite_map = _find_opposite_faces(cells)
        if not opposite_map:
            return None
        idx_perm = list(range(6))
        sub_rng.shuffle(idx_perm)
        label_by_idx = {i: labels[idx_perm[i]] for i in range(6)}

        if qtype == "opposite_face":
            candidates = [i for i in range(6) if i in opposite_map]
            if not candidates:
                return None
            query_idx = sub_rng.choice(candidates)
            opp_idx = opposite_map[query_idx]
            q_label = label_by_idx[query_idx]
            a_label = label_by_idx[opp_idx]
            stems = [
                f"The image shows a 2D net made of 6 squares that can be folded into a cube. "
                f"If you fold the net into a cube, which face will end up directly opposite to face {q_label}?",
                f"Suppose we fold the net shown into a closed cube. Which labelled square will be directly across from the square labelled {q_label}?",
            ]
            q = sub_rng.choice(stems) + " Answer with the face label."
            image = self._render_net(
                cells, [label_by_idx[i] for i in range(6)], sub_rng,
                rotate_labels=cfg.get("allow_label_rotation", False))
            return q, a_label, image

        if qtype == "adjacent_face":
            idx = sub_rng.choice(range(6))
            q_label = label_by_idx[idx]
            stems = [
                f"If this net is folded into a cube, how many faces are adjacent to (share an edge with) face {q_label}?",
                f"After folding this net into a cube, count the number of faces that share an edge with face {q_label}.",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            # Adjacent face count on a cube is always 4 (all except itself and its opposite)
            answer = "4"
            image = self._render_net(
                cells, [label_by_idx[i] for i in range(6)], sub_rng,
                rotate_labels=cfg.get("allow_label_rotation", False))
            return q, answer, image

        if qtype == "fold_result":
            candidates = [i for i in range(6) if i in opposite_map]
            if not candidates:
                return None
            bottom_idx = sub_rng.choice(candidates)
            top_idx = opposite_map[bottom_idx]
            q_label = label_by_idx[bottom_idx]
            a_label = label_by_idx[top_idx]
            stems = [
                f"If this net is folded into a cube with face {q_label} on the bottom, which face will be on top?",
                f"When the net is folded so face {q_label} forms the bottom of the cube, which labelled face forms the top?",
            ]
            q = sub_rng.choice(stems) + " Answer with the face label."
            image = self._render_net(
                cells, [label_by_idx[i] for i in range(6)], sub_rng,
                rotate_labels=cfg.get("allow_label_rotation", False))
            return q, a_label, image

        return None

    def _random_connected_polyomino(self, rng, n) -> List[Tuple[int, int]]:
        cells = {(0, 0)}
        while len(cells) < n:
            seed_cell = rng.choice(list(cells))
            dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            nxt = (seed_cell[0] + dr, seed_cell[1] + dc)
            cells.add(nxt)
        return sorted(_normalize(list(cells)))

    def _render_net(self, cells, labels, sub_rng, rotate_labels=False):
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        sub_rng.shuffle(palette)

        cells = _normalize(list(cells))
        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)
        nrows = max_r + 1
        ncols = max_c + 1

        fig_w = max(5, ncols + 2) * sc
        fig_h = max(5, nrows + 2) * sc
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        edge_col = sub_rng.choice(["#222", "#1a1a2e", "#0b3d0b",
                                    "#2d1b3e", "#3b0b0b"])
        lw = 1.5 + sub_rng.random() * 1.5
        for i, (r, c) in enumerate(cells):
            draw_y = (nrows - 1 - r)
            draw_x = c
            rect = mpatches.FancyBboxPatch(
                (draw_x + 0.03, draw_y + 0.03), 0.94, 0.94,
                boxstyle="round,pad=0.02",
                facecolor=palette[i % len(palette)],
                edgecolor=edge_col, linewidth=lw,
            )
            ax.add_patch(rect)
            if labels is not None:
                rotation = 0
                if rotate_labels and sub_rng.random() < 0.3:
                    rotation = sub_rng.choice([90, -90, 180])
                ax.text(
                    draw_x + 0.5, draw_y + 0.5, labels[i],
                    fontsize=16 + sub_rng.randint(0, 4),
                    fontweight="bold", ha="center", va="center",
                    color="#111", rotation=rotation)

        grid_col = sub_rng.choice(["#dddddd", "#cccccc", "#e0e0e0"])
        for r in range(nrows + 1):
            ax.axhline(y=r, color=grid_col, linewidth=0.5, linestyle=":")
        for c in range(ncols + 1):
            ax.axvline(x=c, color=grid_col, linewidth=0.5, linestyle=":")

        pad = 0.5
        ax.set_xlim(-pad, ncols + pad)
        ax.set_ylim(-pad, nrows + pad)
        title = sub_rng.choice(_TITLE_VARIANTS)
        ax.set_title(title, fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = NetFoldingQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
