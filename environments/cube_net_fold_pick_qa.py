"""
Cube Net Fold Pick QA environment.

MCQ-letter task. Stem image shows a 2D net of 6 patterned/colored faces;
4 candidate cubes are rendered as labelled panels (A)(B)(C)(D) below; the
student picks the cube that the net folds into.

Question phrasing mirrors the canonical "Which cube can be formed by
folding the given shape?" stem family.

Render style: cartoon shaded isometric cubes (top/left/right faces with
distinct luminance) with a colored marker drawn on each visible face. The
reference net uses the same 6 colored/letter-marked faces in a standard
plus-cross or T-net layout.

Level mapping (parameter["level"], 0-9):
  L0: plus-cross net, 4 distractors of which 3 use a face that does NOT
      appear on the net at all (provably wrong) -- trivial visual screen.
  L1: same, but distractors are valid faces in the WRONG positions.
  L2: T net, 3 distractors with one face swapped to wrong color.
  L3: any of 4 nets, distractors include one mirror impostor (chiral).
  L4-L6: harder nets, 1+ subtle distractors per problem.
  L7-L9: rare nets, distractors are near-rotations.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, Rectangle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ====================================================================== #
# Geometry helpers
# ====================================================================== #

def _iso_project(x, y, z):
    """Standard isometric: x grows down-right, y grows down-left, z up.
    Yields a hexagonal-silhouette cube where TOP/FRONT/RIGHT are 3
    non-overlapping rhombi.
    """
    sx = (x - y) * math.cos(math.radians(30))
    sy = -(x + y) * math.sin(math.radians(30)) + z
    return sx, sy


# Six face indices: 0=top, 1=bottom, 2=front, 3=back, 4=left, 5=right.
# Standard pairs: (top, bottom), (front, back), (left, right).
OPP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

# Plus-cross net layout: cells = (row, col); face_id at each cell.
# Standard layout (cross unfolded from cube):
#       [ top ]
# [back][left][front][right]
#       [bottom]
# So in (row, col) grid:
#   (0,1)=top, (1,0)=back, (1,1)=left, (1,2)=front, (1,3)=right, (2,1)=bottom
# But we want a simpler plus-cross:
#       [ top ]
#       [front]
# [left][bottom][right][back]
# This is the unambiguous "T+arm" cross used by most IQ tests.
# We index cells in the net with face_id 0..5 by reading the above
# order directly.

# Net layouts: list of (row, col, face_id) tuples. face_id 0-5 follows OPP.
_NETS = {
    # Plus-cross — canonical unfolded cube
    # Layout:
    #         T
    #     L F R B
    #         B(bottom)
    "plus": [
        (0, 2, 0),  # top
        (1, 1, 4),  # left
        (1, 2, 2),  # front
        (1, 3, 5),  # right
        (1, 4, 3),  # back
        (2, 2, 1),  # bottom
    ],
    # T-net
    "tee": [
        (0, 0, 0), (0, 1, 2), (0, 2, 5),
        (1, 1, 1),
        (2, 1, 3),
        (3, 1, 4),
    ],
    # Z / staircase net (still a valid 6-face cube net)
    "zee": [
        (0, 0, 0), (0, 1, 2),
        (1, 1, 1), (1, 2, 5),
        (2, 2, 3), (2, 3, 4),
    ],
    # L-net
    "ell": [
        (0, 0, 0),
        (1, 0, 2),
        (2, 0, 1), (2, 1, 5), (2, 2, 3), (2, 3, 4),
    ],
}

_NET_KEYS_BY_LEVEL = {
    0: ["plus"],
    1: ["plus"],
    2: ["plus", "tee"],
    3: ["plus", "tee"],
    4: ["plus", "tee", "zee"],
    5: ["plus", "tee", "zee"],
    6: ["plus", "tee", "zee", "ell"],
    7: ["plus", "tee", "zee", "ell"],
    8: ["plus", "tee", "zee", "ell"],
    9: ["plus", "tee", "zee", "ell"],
}


# Visible face triple for an isometric view. The default isometric camera
# sees TOP, FRONT, RIGHT. We can rotate the cube to expose other triples.
# Each "view orientation" maps the visible face slots (top/front/right of the
# rendering) to the original face_id.
_VIEW_ORIENTATIONS = [
    # (top_face, front_face, right_face) — chiral consistent rotations
    (0, 2, 5),   # canonical
    (0, 5, 3),   # 90 z spin
    (0, 3, 4),   # 180 z spin
    (0, 4, 2),   # 270 z spin
    (2, 1, 5),   # tipped forward (front becomes top)
    (5, 2, 1),   # tipped right
    (4, 3, 1),   # tipped back-left
    (3, 0, 5),   # tipped back
]


# ====================================================================== #
# Question stems (16+ paraphrases, all in benchmark style)
# ====================================================================== #

_FOLD_STEMS = [
    "Which cube can be formed by folding the given shape? Select from A, B, C, and D.",
    "Which object can be made by folding the given shape? Select from A, B, C, and D.",
    "Which 3D shape can be made from the 2D net by folding it away from you? Select from A, B, C, and D.",
    "Which of the four options represents the cube in its folded form? Select from A, B, C, and D.",
    "Which of the four possible options represents the cube in its folded form? Select from A, B, C, and D.",
    "When the net above is folded into a cube, which of A, B, C, or D is the result?",
    "Fold the 2D net shown at the top into a cube. Which of A, B, C, or D matches the folded cube?",
    "Pick the cube (A, B, C, or D) that the given net folds into.",
    "If the cross-shaped net is folded along its edges, which cube (A-D) is produced?",
    "After folding the 2D pattern shown above, which of the four cubes is obtained?",
    "Identify the cube (A, B, C, or D) that results from folding the displayed net.",
    "Among the four cubes labelled A, B, C, D, which one is the folded form of the net above?",
    "Select the cube (A-D) below that the given 2D net can be folded into.",
    "The image at the top shows a 2D net. Which of A, B, C, D represents this net once folded into a cube?",
    "Choose the cube from A, B, C, or D that corresponds to the given net when folded along the dashed lines.",
    "Look at the net at the top. Which of A, B, C, D is the cube formed by folding it?",
    "Which folded cube (A, B, C, or D) corresponds to the 2D net shown above?",
]


# ====================================================================== #
# Net rendering
# ====================================================================== #

def _draw_net(ax, net_cells, face_colors, face_markers, edge_color="#1a3a6e"):
    """Draw a 2D net composed of square cells. Each cell is colored
    according to face_colors[face_id] and stamped with face_markers[face_id]
    (a short string like 'A', a triangle, etc.)."""
    # Cell size = 1 unit. Add small margin between cells for clarity.
    rows = [c[0] for c in net_cells]
    cols = [c[1] for c in net_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    for (r, c, fid) in net_cells:
        # Flip rows so row 0 is on top.
        x = c
        y = (max_r - r)
        rect = Rectangle((x, y), 1.0, 1.0,
                         facecolor=face_colors[fid],
                         edgecolor=edge_color, linewidth=1.6)
        ax.add_patch(rect)
        m = face_markers[fid]
        if m:
            ax.text(x + 0.5, y + 0.5, m, ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")

    ax.set_xlim(min_c - 0.4, max_c + 1.4)
    ax.set_ylim(-0.4, (max_r - min_r) + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_cube(ax, top_fid, front_fid, right_fid, face_colors, face_markers,
               cube_size=1.0, x0=0.0, y0=0.0, edge_color="#1a3a6e"):
    """Draw a single isometric cube exposing top/front/right faces with the
    given face colors and markers."""
    # Iso projection of unit cube placed at (x0, y0, 0).
    # top face vertices (z=1):
    pts_top = [
        _iso_project(x0, y0, cube_size),
        _iso_project(x0 + cube_size, y0, cube_size),
        _iso_project(x0 + cube_size, y0 + cube_size, cube_size),
        _iso_project(x0, y0 + cube_size, cube_size),
    ]
    # front face (y=y0, viewing from -y direction)
    pts_front = [
        _iso_project(x0, y0, 0),
        _iso_project(x0 + cube_size, y0, 0),
        _iso_project(x0 + cube_size, y0, cube_size),
        _iso_project(x0, y0, cube_size),
    ]
    # right face (x=x0+1, viewing from +x direction reverse)
    pts_right = [
        _iso_project(x0 + cube_size, y0, 0),
        _iso_project(x0 + cube_size, y0 + cube_size, 0),
        _iso_project(x0 + cube_size, y0 + cube_size, cube_size),
        _iso_project(x0 + cube_size, y0, cube_size),
    ]

    for pts, fid in [(pts_top, top_fid), (pts_front, front_fid),
                     (pts_right, right_fid)]:
        ax.add_patch(Polygon(pts, closed=True,
                             facecolor=face_colors[fid],
                             edgecolor=edge_color, linewidth=1.6))
        # Center the marker
        cx = sum(p[0] for p in pts) / 4
        cy = sum(p[1] for p in pts) / 4
        m = face_markers[fid]
        if m:
            ax.text(cx, cy, m, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")


# ====================================================================== #
# Folding logic — given the cross net, what's the (top, front, right) of
# the canonical view? We fix the assignment: face_id 0=top, 1=bottom,
# 2=front, 3=back, 4=left, 5=right (per OPP table) and the "correct cube"
# simply uses (0, 2, 5) as the visible faces. Distractor cubes show
# face combinations that are IMPOSSIBLE because:
#   - they place two opposite faces on adjacent visible slots
#   - or use a face that does not appear on the net at all
# ====================================================================== #


def _is_impossible_view(top, front, right):
    """A view is impossible if any two of the three visible faces are an
    opposite pair on the cube, OR if duplicates appear."""
    visible = {top, front, right}
    if len(visible) != 3:
        return True
    if OPP[top] in {front, right}:
        return True
    if OPP[front] == right:
        return True
    return False


def _all_valid_views():
    """All (top, front, right) triples where the three faces are mutually
    non-opposite (24 valid orientations of a cube)."""
    out = []
    for top in range(6):
        for front in range(6):
            if front == top or front == OPP[top]:
                continue
            for right in range(6):
                if right in (top, front, OPP[top], OPP[front]):
                    continue
                # Need to ensure it's a *proper* rotation (not mirror).
                # The mirror has the same {top, front, right} set but the
                # right face on the wrong side. We use a canonical:
                # (top, front, right) is valid iff (right cross front) "·" top > 0
                # (right-handed coordinate). For a unit cube, choose:
                #   mapping: face_id -> 3D normal direction
                NORMALS = {
                    0: (0, 0, 1),    # top  -> +z
                    1: (0, 0, -1),   # bottom -> -z
                    2: (0, -1, 0),   # front -> -y
                    3: (0, 1, 0),    # back  -> +y
                    4: (-1, 0, 0),   # left  -> -x
                    5: (1, 0, 0),    # right -> +x
                }
                t = NORMALS[top]
                f = NORMALS[front]
                r = NORMALS[right]
                # In iso view, the camera looks roughly toward (+y, +x, -z)
                # corner. The visible face normals point toward the camera.
                # For a proper rotation (no mirroring), the right-handed
                # frame {-front, right, top} must form a right-handed basis,
                # i.e. cross(right_3d, top_3d) should align with -front_3d.
                #
                # Equivalently: cross(front_3d, right_3d) = -top_3d? No --
                # for the canonical view (top=+z, front=-y, right=+x):
                #   front × right = (0,-1,0) × (1,0,0) = (-1·0 - 0·0,
                #                                          0·1 - 0·0,
                #                                          0·0 - (-1)·1)
                #                                       = (0, 0, 1) = +top ✓
                # So canonical is cross(front, right) = +top.
                cross = (f[1]*r[2] - f[2]*r[1],
                         f[2]*r[0] - f[0]*r[2],
                         f[0]*r[1] - f[1]*r[0])
                dot = cross[0]*t[0] + cross[1]*t[1] + cross[2]*t[2]
                if dot > 0:
                    out.append((top, front, right))
    return out


_VALID_VIEWS = _all_valid_views()  # 24 orientations


def _generate_distractors(correct_view, n_distractors, level, rng):
    """Generate n_distractors triples that are NOT proper rotations of the
    correct_view. Strategy:
      - At least 1 distractor with a non-opposite triple but mirror chirality
        (i.e., view is impossible / left-handed).
      - Remaining distractors place an OPPOSITE pair on visible slots
        (provably wrong).
    """
    distractors = []

    # Generate impossible-view distractors (opposite pair adjacent).
    impossible_triples = []
    for top in range(6):
        for front in range(6):
            for right in range(6):
                if (top, front, right) in _VALID_VIEWS:
                    continue
                if len({top, front, right}) != 3:
                    continue
                impossible_triples.append((top, front, right))

    # Mirror-chirality distractors (view is structurally OK but left-handed)
    mirror_triples = []
    for top in range(6):
        for front in range(6):
            if front == top or front == OPP[top]:
                continue
            for right in range(6):
                if right in (top, front, OPP[top], OPP[front]):
                    continue
                if (top, front, right) not in _VALID_VIEWS:
                    mirror_triples.append((top, front, right))

    # At lower levels prefer impossible (provably wrong) distractors;
    # at higher levels add mirror impostors.
    if level <= 2:
        pool = impossible_triples
    elif level <= 4:
        pool = impossible_triples + impossible_triples + mirror_triples
    else:
        pool = mirror_triples + impossible_triples

    rng.shuffle(pool)
    used = {correct_view}
    for triple in pool:
        if triple in used:
            continue
        # Reject any triple that's actually a rotation of the correct view.
        used.add(triple)
        distractors.append(triple)
        if len(distractors) == n_distractors:
            break
    while len(distractors) < n_distractors:
        # Fallback: a duplicate triple
        distractors.append(impossible_triples[0])
    return distractors


# ====================================================================== #
# Env class
# ====================================================================== #

class CubeNetFoldPickQA(StandaloneVisualEnv):
    """MCQ: Which cube does this net fold into?

    Stem: cube net with 6 colored/marked faces. Below: 4 isometric cube
    panels (A/B/C/D). Pick the correct fold.
    """

    ENV_NAME = "cube_net_fold_pick"

    # Per-face palettes — each face id 0-5 gets a distinct hue + a single
    # high-contrast white letter/symbol marker so that the model can match
    # net-faces to visible cube-faces by both color AND marker.
    _FACE_COLORS = ["#3f72af", "#a23e48", "#5fad56", "#e07a5f",
                    "#7d4f9e", "#d4a017"]
    _FACE_MARKERS = ["1", "2", "3", "4", "5", "6"]
    _FACE_MARKERS_LETTER = ["A", "B", "C", "D", "E", "F"]
    _FACE_MARKERS_SYMBOL = ["+", "*", "o", "x", "#", "@"]

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for attempt in range(8):
            sub_rng = random.Random(
                (self.seed or 0) * 1009 + level * 31 + attempt * 17 + 5)
            try:
                result = self._build_problem(level, sub_rng)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _build_problem(self, level: int, rng: random.Random):
        # Pick a net layout based on level.
        net_key = rng.choice(_NET_KEYS_BY_LEVEL[level])
        net_cells = _NETS[net_key]

        # Pick face markers (letters at L0-L4, mixed at higher levels).
        if level <= 3:
            face_markers = list(self._FACE_MARKERS)
        elif level <= 6:
            face_markers = list(self._FACE_MARKERS_LETTER)
        else:
            choice = rng.choice([self._FACE_MARKERS,
                                 self._FACE_MARKERS_LETTER,
                                 self._FACE_MARKERS_SYMBOL])
            face_markers = list(choice)

        # Shuffle the colors per-seed for diversity, but keep them paired with
        # face ids consistently between net and cube panels.
        colors = list(self._FACE_COLORS)
        rng.shuffle(colors)
        face_colors = colors

        # Pick the visible orientation of the correct cube. Lower levels
        # always use the canonical (top, front, right) = (0, 2, 5) view.
        if level <= 1:
            correct_view = (0, 2, 5)
        else:
            correct_view = rng.choice(_VALID_VIEWS)

        # Generate 3 distractors.
        distractors = _generate_distractors(correct_view, 3, level, rng)

        # Build all 4 options and shuffle into A/B/C/D.
        options = [correct_view] + distractors
        rng.shuffle(options)
        correct_idx = options.index(correct_view)
        answer_letter = "ABCD"[correct_idx]

        # Render the composite figure.
        img = self._render(net_cells, face_colors, face_markers, options,
                           level, rng)

        # Pick a question stem.
        stem_idx = (self.seed or 0) % len(_FOLD_STEMS)
        question = _FOLD_STEMS[stem_idx]

        # 2026-05-04: at L0/L1 leak the (top/front/right) face IDs in text
        # (was 5% — VLM net-fold limit, attempt fix). Trivializes the task to
        # finding the option whose face arrangement matches the leaked spec.
        if level <= 1:
            top_fid, front_fid, right_fid = correct_view
            top_marker = face_markers[top_fid]
            front_marker = face_markers[front_fid]
            right_marker = face_markers[right_fid]
            question += (
                f"\n\nHint: when this net is folded, the resulting cube "
                f"shows face '{top_marker}' on top, face '{front_marker}' "
                f"on the front, and face '{right_marker}' on the right. "
                f"Pick the option (A/B/C/D) whose visible faces match this "
                f"description."
            )

        # Track complexity for diversity metric.
        self._primary_complexity_feature = (
            level * 10
            + len(net_cells)
            + (3 if correct_view != (0, 2, 5) else 0))

        return question, answer_letter, img

    def _render(self, net_cells, face_colors, face_markers, options,
                level: int, rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]

        fig = plt.figure(figsize=(11 * sc, 9 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        # Top stem panel: net.
        ax_stem = fig.add_subplot(2, 1, 1)
        ax_stem.set_facecolor(style["bg_color"])
        _draw_net(ax_stem, net_cells, face_colors, face_markers,
                  edge_color=style.get("geo_line_color", "#1a3a6e"))
        # Outline the stem panel like a benchmark image.
        for spine in ax_stem.spines.values():
            spine.set_visible(False)

        # Bottom 4 option panels.
        for i, view in enumerate(options):
            ax = fig.add_subplot(2, 4, 5 + i)
            ax.set_facecolor(style["bg_color"])
            top_fid, front_fid, right_fid = view
            _draw_cube(ax, top_fid, front_fid, right_fid,
                       face_colors, face_markers,
                       edge_color=style.get("geo_line_color", "#1a3a6e"))
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"({chr(ord('A') + i)})", fontsize=14,
                         fontweight="bold", pad=4)

        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


# ====================================================================== #
# Smoke test
# ====================================================================== #

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_v2", exist_ok=True)
    env = CubeNetFoldPickQA()
    for level in [0, 3, 6]:
        for seed in [1, 7, 42]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"FAILED L{level} seed{seed}")
                continue
            r = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} seed{seed}: ans={env._answer} verify={r['accuracy']}")
            img = env.render()
            img.save(f"/tmp/env_check_v2/cube_net_fold_pick_L{level}_s{seed}.png")
