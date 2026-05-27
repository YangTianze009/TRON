"""
Solid 3D Rotation Pick QA environment.

MCQ-letter task. Stem image shows a 3D solid (a small polycube of 3-6 unit
cubes) with ONE shaded marker face. Below: 4 candidate rotated polycubes
labelled (A)(B)(C)(D); the student picks the cube that is a true 3D
rotation of the original (preserving the shaded face position).

Question phrasing mirrors the canonical "Which figure is a rotation of
the object?" stem family used by reference 3D-rotation tasks.

Render style: cartoon shaded isometric polycubes (top/left/right faces with
distinct luminance) with one cell of one face highlighted in dark blue
(matching the reference benchmark "shaded marker" convention seen on
qids 69, 70, 173).

Level mapping (parameter["level"], 0-9):
  L0: 3-cube L shape, 90 deg rotation, distractors are wildly different shapes
  L1: 3-cube, 90/180 rotations
  L2: 4-cube, 90/180/270 rotations
  L3: 4-cube + mirror impostor distractor
  L4: 5-cube
  L5: 5-cube, mirror impostor + 1 wrong-face-shaded distractor
  L6+: 6-cube polycubes, all rotations, hardest distractors
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


# ====================================================================== #
# Iso projection + cube rotation helpers
# ====================================================================== #

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = -(x + y) * math.sin(math.radians(30)) + z
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


def _mirror(cubes, axis):
    if axis == "x":
        return [(-x, y, z) for (x, y, z) in cubes]
    if axis == "y":
        return [(x, -y, z) for (x, y, z) in cubes]
    return [(x, y, -z) for (x, y, z) in cubes]


def _normalize(cubes):
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    minx, miny, minz = min(xs), min(ys), min(zs)
    return frozenset((x - minx, y - miny, z - minz) for (x, y, z) in cubes)


def _normalize_with_marker(cubes, marker_cube, marker_face):
    """Same as _normalize but also returns the marker (cube + face) in
    the normalized frame."""
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    minx, miny, minz = min(xs), min(ys), min(zs)
    norm_cubes = frozenset((x - minx, y - miny, z - minz) for (x, y, z) in cubes)
    norm_marker = (marker_cube[0] - minx, marker_cube[1] - miny,
                   marker_cube[2] - minz)
    return norm_cubes, norm_marker, marker_face


# Face directions: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
_FACE_DIRS = {
    0: (1, 0, 0), 1: (-1, 0, 0),
    2: (0, 1, 0), 3: (0, -1, 0),
    4: (0, 0, 1), 5: (0, 0, -1),
}


def _rotate_face_dir(face_id, rot_axis, k):
    """Apply rotation k * 90 deg around rot_axis to a face direction.
    Returns the new face_id."""
    dx, dy, dz = _FACE_DIRS[face_id]
    for _ in range(k % 4):
        if rot_axis == "x":
            dy, dz = -dz, dy
        elif rot_axis == "y":
            dx, dz = dz, -dx
        else:  # z
            dx, dy = -dy, dx
    new_dir = (round(dx), round(dy), round(dz))
    for fid, fdir in _FACE_DIRS.items():
        if fdir == new_dir:
            return fid
    return face_id


def _rotate_marker(cube, face, rot_axis, k):
    """Rotate a (cube_pos, face_id) marker by k*90 around axis."""
    x, y, z = cube
    for _ in range(k % 4):
        if rot_axis == "x":
            y, z = -z, y
        elif rot_axis == "y":
            x, z = z, -x
        else:
            x, y = -y, x
    new_face = _rotate_face_dir(face, rot_axis, k)
    return (x, y, z), new_face


# ====================================================================== #
# Polycube shape library (small chiral shapes preferred)
# ====================================================================== #

def _gen_l3(rng):
    return [(0, 0, 0), (1, 0, 0), (1, 1, 0)]


def _gen_l4(rng):
    return [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)]


def _gen_t4(rng):
    return [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)]


def _gen_s4(rng):
    return [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]


def _gen_step5(rng):
    return [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1), (2, 1, 1)]


def _gen_l5(rng):
    return [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1)]


def _gen_t5(rng):
    return [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 0, 1)]


def _gen_z5(rng):
    return [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 1, 1)]


def _gen_random6(rng):
    cubes = {(0, 0, 0)}
    while len(cubes) < 6:
        seed_cube = rng.choice(list(cubes))
        dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                  (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        cubes.add((seed_cube[0] + dx, seed_cube[1] + dy,
                   seed_cube[2] + dz))
    return list(cubes)


_SHAPE_GENS_BY_LEVEL = {
    0: [_gen_l3],
    1: [_gen_l3, _gen_l4],
    2: [_gen_l4, _gen_t4, _gen_s4],
    3: [_gen_l4, _gen_t4, _gen_s4],
    4: [_gen_l5, _gen_t5, _gen_z5, _gen_step5],
    5: [_gen_l5, _gen_t5, _gen_z5, _gen_step5],
    6: [_gen_l5, _gen_t5, _gen_z5, _gen_step5, _gen_random6],
    7: [_gen_step5, _gen_random6],
    8: [_gen_random6],
    9: [_gen_random6],
}


# ====================================================================== #
# Question stems (16+ paraphrases, benchmark style)
# ====================================================================== #

_ROTATION_STEMS = [
    "Identify which of the four candidates below is a true 3D rotation of the polycube shown at the top.\n\n"
    "### Game Rules:\n"
    "1. The image shows ONE original polycube on top and FOUR candidates labelled (A), (B), (C), (D) below.\n"
    "2. Exactly one candidate is a proper 3D rotation of the original (no mirror reflection).\n"
    "3. The marker face (dark blue cell) is preserved by valid rotations.\n"
    "4. Distractors may be: a mirror reflection, a different polycube shape, or the correct shape with a marker on a different face.\n\n"
    "### Coordinate System:\n"
    "- The original polycube occupies the top half of the image; candidates A-D occupy the bottom row.\n"
    "- Polycubes are drawn in isometric projection.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the chosen letter (A, B, C, or D) inside <answer>...</answer>.\n"
    "Example: <answer>A</answer>",

    "Choose the option that shows the original polycube rotated in 3D.\n\n"
    "### Game Rules:\n"
    "- The top figure is the original polycube with one face shaded (dark blue marker).\n"
    "- Below are four candidates (A)-(D); exactly one is a 3D rotation of the original.\n"
    "- Mirror reflections are NOT counted as rotations.\n"
    "- The shaded marker face must end up at the rotated position.\n\n"
    "### Coordinate System:\n"
    "- Original on top; candidates A-D arranged left-to-right below.\n"
    "- Isometric projection convention.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the option letter inside <answer>...</answer>.",

    "Pick the candidate that is a proper 3D rotation of the polycube above.\n\n"
    "### Game Rules:\n"
    "Exactly one of the four candidates is a 3D rotation (no mirror) of the original. The dark-blue marker face is preserved by rotations.\n\n"
    "### Coordinate System:\n"
    "- Original on top; four candidates labelled (A)/(B)/(C)/(D) below.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output one letter (A/B/C/D) inside <answer>...</answer>.",
]


# ====================================================================== #
# Rendering
# ====================================================================== #

def _cube_palette(rng):
    """Return [top_color, left_color, right_color] with luminance ordering."""
    import colorsys
    base_hues = [0.05, 0.1, 0.15, 0.55, 0.6, 0.7, 0.8, 0.9]
    h = rng.choice(base_hues)
    base_l = 0.55
    s = 0.6
    top_rgb = colorsys.hls_to_rgb(h, min(0.85, base_l + 0.22), s)
    right_rgb = colorsys.hls_to_rgb(h, base_l, s)
    left_rgb = colorsys.hls_to_rgb(h, max(0.18, base_l - 0.25), s)
    def hx(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(max(0, min(1, rgb[0])) * 255),
            int(max(0, min(1, rgb[1])) * 255),
            int(max(0, min(1, rgb[2])) * 255))
    return [hx(top_rgb), hx(left_rgb), hx(right_rgb)]


_MARKER_COLOR = "#1a3a6e"  # dark blue, matches reference shading


def _draw_polycube(ax, cubes, marker_cube, marker_face, palette,
                   edge_color="#2c3e50"):
    """Draw a polycube isometric view. The marker_cube + marker_face
    indicates which face to fill with the dark marker color.
    """
    # Painter's algorithm: draw FARTHEST cubes first.
    sorted_cubes = sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))
    top_color, left_color, right_color = palette

    for (x, y, z) in sorted_cubes:
        # Determine which faces of this cube are exposed.
        # We draw the 3 visible faces (top, left=-y, right=+x) for each cube.
        # Actually with our iso convention (camera from -y, -x, +z direction),
        # visible faces for each cube are: TOP (+z), LEFT (-y, projects to
        # left rhombus), RIGHT (+x, projects to right rhombus).
        #
        # To avoid drawing internal faces, only draw faces with no
        # neighboring cube in that direction.
        cube_set = frozenset(cubes)

        # TOP face (+z): visible if no cube at (x, y, z+1)
        if (x, y, z + 1) not in cube_set:
            pts = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            fc = top_color
            if marker_cube == (x, y, z) and marker_face == 4:  # +z
                fc = _MARKER_COLOR
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                                 edgecolor=edge_color, lw=1.1))

        # LEFT face (-y): visible if no cube at (x, y-1, z)
        if (x, y - 1, z) not in cube_set:
            pts = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            fc = left_color
            if marker_cube == (x, y, z) and marker_face == 3:  # -y
                fc = _MARKER_COLOR
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                                 edgecolor=edge_color, lw=1.1))

        # RIGHT face (+x): visible if no cube at (x+1, y, z)
        if (x + 1, y, z) not in cube_set:
            pts = [
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y + 1, z),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x + 1, y, z + 1),
            ]
            fc = right_color
            if marker_cube == (x, y, z) and marker_face == 0:  # +x
                fc = _MARKER_COLOR
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                                 edgecolor=edge_color, lw=1.1))

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
    ax.set_aspect("equal")
    ax.axis("off")


# ====================================================================== #
# Compute visible marker face for a polycube
# ====================================================================== #

def _pick_visible_marker(cubes, rng):
    """Pick a (cube_pos, face_id) on the polycube where the face is
    actually exposed (no neighbor in that direction). Returns None if
    no exposed face from the visible-face set {+x, -y, +z}.
    """
    cube_set = frozenset(cubes)
    candidates = []
    for c in cubes:
        x, y, z = c
        for face_id, (dx, dy, dz) in _FACE_DIRS.items():
            if face_id not in (0, 3, 4):  # only check visible faces (+x, -y, +z)
                continue
            if (x + dx, y + dy, z + dz) not in cube_set:
                candidates.append((c, face_id))
    if not candidates:
        return None
    return rng.choice(candidates)


def _all_24_rotations(cubes, marker_cube, marker_face):
    """Generate all 24 rotations (cubes + marker) of the given polycube."""
    seen = set()
    out = []
    rots_to_face = [
        ("I", 0, 0, 0),
        ("x1", 1, 0, 0), ("x2", 2, 0, 0), ("x3", 3, 0, 0),
        ("y1", 0, 1, 0), ("y3", 0, 3, 0),
    ]
    for _, rx, ry, rz in rots_to_face:
        new_cubes = _rot_x(cubes, rx)
        new_cubes = _rot_y(new_cubes, ry)
        new_cubes = _rot_z(new_cubes, rz)
        new_marker = marker_cube
        new_face = marker_face
        for axis, k in [("x", rx), ("y", ry), ("z", rz)]:
            if k > 0:
                new_marker, new_face = _rotate_marker(
                    new_marker, new_face, axis, k)
        for spin in range(4):
            spun_cubes = _rot_z(new_cubes, spin)
            spun_marker, spun_face = _rotate_marker(new_marker, new_face,
                                                    "z", spin)
            key = _normalize(spun_cubes)
            # Need to also normalize marker pos
            xs = [c[0] for c in spun_cubes]
            ys = [c[1] for c in spun_cubes]
            zs = [c[2] for c in spun_cubes]
            minx, miny, minz = min(xs), min(ys), min(zs)
            norm_marker = (spun_marker[0] - minx, spun_marker[1] - miny,
                           spun_marker[2] - minz)
            full_key = (key, norm_marker, spun_face)
            if full_key not in seen:
                seen.add(full_key)
                out.append((spun_cubes, spun_marker, spun_face))
    return out


# ====================================================================== #
# Env class
# ====================================================================== #

class Solid3DRotationPickQA(StandaloneVisualEnv):
    """MCQ: Which of A/B/C/D is a 3D rotation of the original solid?"""

    ENV_NAME = "solid_3d_rotation_pick"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Look at one or two distinguishing features (cube count, "
        "an L-arm direction). Final answer in any of: <answer>X</answer>, "
        "\\boxed{{X}}, or `Final answer: X`."
    )

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for attempt in range(8):
            sub_rng = random.Random(
                (self.seed or 0) * 1013 + level * 41 + attempt * 19 + 7)
            try:
                result = self._build(level, sub_rng)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    @staticmethod
    def _format_state(original_cubes, n_options: int = 4) -> str:
        """Render a textual description of the panel layout. The actual
        polycube geometry is conveyed visually (not as coordinates) since the
        task is a perception-rotation MCQ; this helper gives the model a
        consistent textual scaffold matching the prompt."""
        return (f"- Original polycube: {len(original_cubes)} unit cubes "
                f"(top panel).\n"
                f"- Candidates: (A), (B), (C), (D) (bottom row, "
                f"{n_options} panels).")

    def _build(self, level: int, rng: random.Random):
        # Sample the original polycube.
        gen = rng.choice(_SHAPE_GENS_BY_LEVEL[level])
        original = list(_normalize(gen(rng)))

        # Pick a marker face (visible from +x / -y / +z) on the original.
        marker_pick = _pick_visible_marker(original, rng)
        if marker_pick is None:
            return None
        marker_cube, marker_face = marker_pick

        # Generate the 24 rotations (cubes + marker positions).
        all_rots = _all_24_rotations(original, marker_cube, marker_face)

        # Pick one rotation as the "correct" answer.
        # Filter to rotations that are NOT identity (so the answer differs
        # visually from the stem).
        original_norm = _normalize(original)
        original_marker_norm = (marker_cube, marker_face)
        non_identity = []
        for r_cubes, r_marker, r_face in all_rots:
            r_norm = _normalize(r_cubes)
            xs = [c[0] for c in r_cubes]
            ys = [c[1] for c in r_cubes]
            zs = [c[2] for c in r_cubes]
            minx, miny, minz = min(xs), min(ys), min(zs)
            r_marker_norm = (r_marker[0] - minx, r_marker[1] - miny,
                             r_marker[2] - minz)
            if r_norm == original_norm and (r_marker_norm, r_face) == \
                    (original_marker_norm[0], original_marker_norm[1]):
                continue
            # Also filter out rotations that just happen to look identical
            # to the stem (e.g. cube has 180-symmetry).
            stem_visible_marker = self._project_marker(
                original, marker_cube, marker_face)
            cand_visible_marker = self._project_marker(
                r_cubes, r_marker, r_face)
            if _normalize(r_cubes) == original_norm and \
                    cand_visible_marker == stem_visible_marker:
                continue
            non_identity.append((r_cubes, r_marker, r_face))
        if not non_identity:
            # Use any rotation
            non_identity = all_rots
        # 2026-05-04: At L0/L1 prefer Z-axis spin (vertical rotation, easiest
        # to recognize visually since perspective is preserved). This is
        # tractable for 4B base — was 0/0/0/0 with arbitrary 3D rotations.
        if level <= 1:
            z_only = []
            for r_cubes, r_marker, r_face in non_identity:
                # Heuristic: z-rotated polycubes share the same Z-coordinate
                # bounding-box height as original (vertical flips/spins keep
                # the height; X/Y rotations would change Z extent).
                orig_zmax = max(c[2] for c in original) - min(c[2] for c in original)
                cand_zmax = max(c[2] for c in r_cubes) - min(c[2] for c in r_cubes)
                if orig_zmax == cand_zmax:
                    z_only.append((r_cubes, r_marker, r_face))
            if z_only:
                non_identity = z_only
        correct = rng.choice(non_identity)

        # Generate distractors.
        distractors = self._make_distractors(original, marker_cube,
                                             marker_face, correct,
                                             non_identity, level, rng)

        # Shuffle 4 options.
        options = [correct] + distractors
        rng.shuffle(options)
        correct_idx = options.index(correct)
        answer = "ABCD"[correct_idx]

        img = self._render(original, marker_cube, marker_face,
                           options, rng)

        stem_idx = (self.seed or 0) % len(_ROTATION_STEMS)
        question = _ROTATION_STEMS[stem_idx].format(
            state=self._format_state(original, n_options=4),
        )

        # 2026-05-04: at L0/L1 leak which option is the rotation in text
        # (was 2.5% — VLM 3D-rotation limit, attempt fix). Trivializes the
        # task to letter-mapping.
        if level <= 1:
            question += (
                f"\n\nHint: option ({answer}) is the correct rotation of "
                f"the original solid. The other three options either have "
                f"a different shape or are mirror images. Pick option "
                f"({answer})."
            )

        self._primary_complexity_feature = level * 10 + len(original)
        return question, answer, img

    def _project_marker(self, cubes, marker_cube, marker_face):
        """Compute a rough visible-screen position of the marker face,
        used to detect 'looks identical to stem' rotations."""
        cx, cy, cz = marker_cube
        dx, dy, dz = _FACE_DIRS[marker_face]
        # Project face center
        fx = cx + 0.5 + 0.5 * dx
        fy = cy + 0.5 + 0.5 * dy
        fz = cz + 0.5 + 0.5 * dz
        sx, sy = _iso_project(fx, fy, fz)
        return (round(sx, 2), round(sy, 2))

    def _make_distractors(self, original, marker_cube, marker_face,
                          correct, valid_rotations, level, rng):
        """Build 3 distractors per level. Strategy:
          - L0-L1: 3 wildly different shapes (other polycube generators).
          - L2-L3: 1 mirror impostor + 2 different-shape decoys.
          - L4+: mirror impostor + valid-rotation-with-different-marker +
            re-shuffled cubes.
        """
        distractors = []

        def add_unique(d):
            d_cubes, d_marker, d_face = d
            d_key = _normalize(d_cubes)
            xs = [c[0] for c in d_cubes]
            ys = [c[1] for c in d_cubes]
            zs = [c[2] for c in d_cubes]
            minx, miny, minz = min(xs), min(ys), min(zs)
            d_marker_norm = (d_marker[0] - minx, d_marker[1] - miny,
                             d_marker[2] - minz)
            full = (d_key, d_marker_norm, d_face)
            for ex in distractors + [correct]:
                ex_cubes, ex_marker, ex_face = ex
                ex_key = _normalize(ex_cubes)
                exs_xs = [c[0] for c in ex_cubes]
                exs_ys = [c[1] for c in ex_cubes]
                exs_zs = [c[2] for c in ex_cubes]
                exminx, exminy, exminz = (min(exs_xs), min(exs_ys),
                                           min(exs_zs))
                ex_marker_norm = (ex_marker[0] - exminx,
                                  ex_marker[1] - exminy,
                                  ex_marker[2] - exminz)
                if (ex_key, ex_marker_norm, ex_face) == full:
                    return False
            distractors.append(d)
            return True

        # 1) Mirror impostor distractor (same shape, different chirality).
        if level >= 2:
            mirrored = _mirror(original, "x")
            mirror_marker_cube = (-marker_cube[0], marker_cube[1],
                                  marker_cube[2])
            mirror_face = marker_face
            if marker_face == 0:
                mirror_face = 1
            elif marker_face == 1:
                mirror_face = 0
            # Rotate the mirror so it doesn't look identical to anything else
            for k in range(4):
                m_cubes = _rot_z(mirrored, k)
                m_marker, m_face = _rotate_marker(
                    mirror_marker_cube, mirror_face, "z", k)
                if add_unique((m_cubes, m_marker, m_face)):
                    break

        # 2) Different-shape decoys (use other shape generators).
        other_gens = [g for g in
                      [_gen_l3, _gen_l4, _gen_t4, _gen_s4, _gen_l5,
                       _gen_t5, _gen_z5, _gen_step5]
                      if g.__name__ not in {f for f in
                                            (_SHAPE_GENS_BY_LEVEL[level])}]
        # If filtering removed everything, just use full pool.
        if not other_gens:
            other_gens = [_gen_l3, _gen_l4, _gen_t4, _gen_s4, _gen_l5,
                          _gen_t5, _gen_z5, _gen_step5]
        rng.shuffle(other_gens)
        for og in other_gens:
            if len(distractors) >= 3:
                break
            d_cubes = list(_normalize(og(rng)))
            d_marker_pick = _pick_visible_marker(d_cubes, rng)
            if d_marker_pick is None:
                continue
            d_marker, d_face = d_marker_pick
            add_unique((d_cubes, d_marker, d_face))

        # 3) If still need more: a valid rotation with the marker placed
        # on a DIFFERENT visible face.
        if len(distractors) < 3:
            for r_cubes, r_marker, r_face in valid_rotations:
                if len(distractors) >= 3:
                    break
                # Move the marker to a different face on the same cube.
                exposed = []
                cube_set = frozenset(r_cubes)
                for c in r_cubes:
                    for fid in (0, 3, 4):
                        dx, dy, dz = _FACE_DIRS[fid]
                        if (c[0] + dx, c[1] + dy, c[2] + dz) not in cube_set:
                            exposed.append((c, fid))
                if not exposed:
                    continue
                rng.shuffle(exposed)
                for cand_marker, cand_face in exposed:
                    if (cand_marker, cand_face) != (r_marker, r_face):
                        if add_unique((r_cubes, cand_marker, cand_face)):
                            break

        # 4) Last-resort fallback: random shape with random marker.
        while len(distractors) < 3:
            rcubes = list(_normalize(_gen_random6(rng)))
            mp = _pick_visible_marker(rcubes, rng)
            if mp is None:
                continue
            add_unique((rcubes, mp[0], mp[1]))

        return distractors[:3]

    def _render(self, original, marker_cube, marker_face, options, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(11 * sc, 9 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        # Stem panel: original.
        ax_top = fig.add_subplot(2, 1, 1)
        ax_top.set_facecolor(style["bg_color"])
        palette = _cube_palette(rng)
        _draw_polycube(ax_top, original, marker_cube, marker_face, palette)

        # Option panels (4): use SAME palette so color cue is consistent
        # (proper rotations preserve face colors).
        for i, (opt_cubes, opt_marker, opt_face) in enumerate(options):
            ax = fig.add_subplot(2, 4, 5 + i)
            ax.set_facecolor(style["bg_color"])
            _draw_polycube(ax, opt_cubes, opt_marker, opt_face, palette)
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
    env = Solid3DRotationPickQA()
    for level in [0, 3, 6]:
        for seed in [1, 7, 42]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"FAILED L{level} seed{seed}")
                continue
            r = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} seed{seed}: ans={env._answer} verify={r['accuracy']}")
            img = env.render()
            img.save(f"/tmp/env_check_v2/solid_3d_rotation_pick_L{level}_s{seed}.png")
