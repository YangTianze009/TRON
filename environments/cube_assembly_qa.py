"""
Cube Assembly QA environment.

Goal: fix the cube assembly regression by
training "reverse engineering" of polycube compound shapes. Shows 2-3
labelled sub-polycubes and a target compound polycube, asks which subset of
the sub-shapes assembles into the target.

Level mapping (parameter["level"], 0-9):
  L0:  TRIVIAL — 2 obviously-distinct sub-shapes (1-cube + 1-cube, or a
       2-cube domino + a 1-cube). Target is always A+B (the union). MCQ
       options are wildly different: "A+B", "A only", "A+B+C" (with phantom
       C), "neither". Always MCQ, no rotation, no counting.
  L1:  3 sub-shapes, always 1-2 cubes each, no rotation.  MCQ over subsets.
  L2:  3 sub-shapes, 2-3 cubes each, single rotation allowed.
  L3:  3 sub-shapes, 2-5 cubes each. Must allow rotation of sub-shapes.
  L4+: 3-4 sub-shapes, 3-5 cubes each, rotations, with plausible subset
       distractors.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    """Copied from isometric_counting_qa so the file is standalone."""
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _shaded_face_triplet(base_hex):
    """BUGFIX: given a base hex color, return [top, left, right] shaded so
    top is lightest and left darkest — standard isometric lighting cue.
    Used to build per-piece palettes where each piece has a distinct hue
    but the three faces within a piece are always clearly distinguishable."""
    import colorsys
    hx = base_hex.lstrip('#')
    r, g, b = int(hx[0:2], 16) / 255, int(hx[2:4], 16) / 255, int(hx[4:6], 16) / 255
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    base_l = 0.55
    ss = min(1.0, max(0.45, ss))
    top_rgb = colorsys.hls_to_rgb(hh, min(0.85, base_l + 0.22), ss)
    right_rgb = colorsys.hls_to_rgb(hh, base_l, ss)
    left_rgb = colorsys.hls_to_rgb(hh, max(0.18, base_l - 0.25), ss)
    def _hx(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(max(0, min(1, rgb[0])) * 255),
            int(max(0, min(1, rgb[1])) * 255),
            int(max(0, min(1, rgb[2])) * 255))
    return [_hx(top_rgb), _hx(left_rgb), _hx(right_rgb)]

def _piece_bases(style, rng, n):
    """BUGFIX: return `n` visually distinct base hex colors from the style
    palette, filtering out pure black/white. If the palette runs out, rotate
    hues to keep making new distinct bases."""
    import colorsys
    pal = [c for c in style['palette']
           if c.lower() not in ('#000000', '#010101', '#0a0a0a', '#ffffff',
                                '#fefefe', '#f1faee')]
    if len(pal) < max(4, n):
        pal = ['#5dade2', '#48c9b0', '#ec7063', '#f4d03f', '#a569bd', '#e67e22']
    rng.shuffle(pal)
    return pal[:n] if len(pal) >= n else pal + pal[: n - len(pal)]

# ------------------------------------------------------------------ #
# Polycube utilities
# ------------------------------------------------------------------ #

def _translate(cubes, dx, dy, dz):
    return [(x + dx, y + dy, z + dz) for (x, y, z) in cubes]

def _normalize(cubes):
    """Shift to (0,0,0) min corner and return a canonical sorted tuple."""
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    minx, miny, minz = min(xs), min(ys), min(zs)
    shifted = sorted((x - minx, y - miny, z - minz) for (x, y, z) in cubes)
    return tuple(shifted)

def _normalize_set(cubes):
    return frozenset(_normalize(cubes))

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

def _all_rotations(cubes):
    """Enumerate all 24 rotations."""
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
            key = _normalize_set(r)
            if key not in seen:
                seen.add(key)
                result.append(r)
    return result

def _shapes_equal(a, b):
    b_norm = _normalize_set(b)
    for r in _all_rotations(a):
        if _normalize_set(r) == b_norm:
            return True
    return False

def _random_polycube(rng, n):
    """Grow a random connected polycube of n cubes."""
    cubes = {(0, 0, 0)}
    tries = 0
    while len(cubes) < n and tries < 200:
        tries += 1
        seed_cube = rng.choice(list(cubes))
        dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                  (0, -1, 0), (0, 0, 1)])
        nxt = (seed_cube[0] + dx, seed_cube[1] + dy, seed_cube[2] + dz)
        cubes.add(nxt)
    return list(cubes)

def _domino_l_pieces():
    """Small shape library: dominoes, trominoes, L pieces."""
    return [
        [(0, 0, 0), (1, 0, 0)],                                  # 2 cube bar
        [(0, 0, 0), (0, 1, 0)],                                  # 2 cube bar y
        [(0, 0, 0), (0, 0, 1)],                                  # 2 cube column
        [(0, 0, 0), (1, 0, 0), (2, 0, 0)],                       # 3 cube bar
        [(0, 0, 0), (1, 0, 0), (1, 1, 0)],                       # L-tromino
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],                       # L-tromino 2
        [(0, 0, 0), (1, 0, 0), (0, 0, 1)],                       # corner piece
    ]

def _assemble(pieces, placements):
    """Apply rotation + translation to each piece and union them into a
    compound polycube. pieces: list of list of (x,y,z). placements: list
    of (rx, ry, rz, dx, dy, dz) — rotations then translation.

    Returns list of cubes if assembly is valid (no overlap), None otherwise.
    """
    result: List[Tuple[int, int, int]] = []
    result_set: Set[Tuple[int, int, int]] = set()
    for piece, plac in zip(pieces, placements):
        rx, ry, rz, dx, dy, dz = plac
        rot = _rot_x(piece, rx)
        rot = _rot_y(rot, ry)
        rot = _rot_z(rot, rz)
        rot = _translate(rot, dx, dy, dz)
        for c in rot:
            if c in result_set:
                return None  # overlap
            result_set.add(c)
            result.append(c)
    return result

def _is_connected(cubes):
    if not cubes:
        return False
    cube_set = set(cubes)
    visited = {cubes[0]}
    stack = [cubes[0]]
    while stack:
        cur = stack.pop()
        x, y, z = cur
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)]:
            nxt = (x + dx, y + dy, z + dz)
            if nxt in cube_set and nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return len(visited) == len(cube_set)

class CubeAssemblyQA(StandaloneVisualEnv):
    """Given 2-3 labelled sub-polycubes and a target compound shape, ask
    which subset assembles into the target."""

    ENV_NAME = "cube_assembly"

    QUESTION_TYPES = [
        "which_subset_mcq",  # MCQ over possible subsets
        "count_pieces",      # integer: how many sub-pieces used
        "uses_piece_x",      # yes/no: is piece X needed?
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(20):
            try:
                result = self._try_generate(parameter)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return dict(special_l0=True)
        if level == 1:
            # MCQ only, small pieces, no rotation
            return dict(n_pieces=3, piece_size=(1, 2), allow_rotation=False,
                        qweights=[10, 0, 0])
        if level == 2:
            return dict(n_pieces=3, piece_size=(2, 3), allow_rotation=False,
                        qweights=[10, 0, 0])
        if level == 3:
            return dict(n_pieces=3, piece_size=(2, 3), allow_rotation=True,
                        qweights=[10, 0, 0])
        if level == 4:
            return dict(n_pieces=3, piece_size=(2, 4), allow_rotation=True,
                        qweights=[8, 2, 0])
        if level == 5:
            return dict(n_pieces=3, piece_size=(3, 5), allow_rotation=True,
                        qweights=[6, 2, 2])
        if level == 6:
            return dict(n_pieces=3, piece_size=(3, 5), allow_rotation=True,
                        qweights=[5, 2, 3])
        if level == 7:
            return dict(n_pieces=4, piece_size=(3, 5), allow_rotation=True,
                        qweights=[4, 3, 3])
        if level == 8:
            return dict(n_pieces=4, piece_size=(4, 6), allow_rotation=True,
                        qweights=[4, 3, 3])
        return dict(n_pieces=4, piece_size=(4, 7), allow_rotation=True,
                    qweights=[4, 2, 4])

    def _try_generate(self, parameter: Dict):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._sub_rng = sub_rng

        if cfg.get("special_l0"):
            return self._generate_l0(sub_rng)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = rng.choices(self.QUESTION_TYPES, weights=cfg["qweights"])[0]

        n_pieces = cfg["n_pieces"]
        piece_size = cfg["piece_size"]
        allow_rotation = cfg["allow_rotation"]

        # Generate pieces
        pieces = []
        max_attempts = 20
        for _ in range(n_pieces):
            for attempt in range(max_attempts):
                if level <= 2:
                    piece = list(sub_rng.choice(_domino_l_pieces()))
                else:
                    size = sub_rng.randint(*piece_size)
                    piece = _random_polycube(sub_rng, size)
                if 2 <= len(piece) <= piece_size[1]:
                    pieces.append(piece)
                    break
            else:
                return None

        # Pick a subset to use
        subset_indices = self._pick_subset(rng, n_pieces, level)
        if not subset_indices:
            return None

        # Assemble
        target = self._try_assemble_subset(
            rng, pieces, subset_indices, allow_rotation
        )
        if target is None:
            return None
        if not _is_connected(target):
            return None
        if len(target) < 3:
            return None

        self._primary_complexity_feature = n_pieces * 2 + sum(len(p) for p in pieces)
        labels = ["A", "B", "C", "D"][:n_pieces]
        correct_subset_str = "+".join(labels[i] for i in subset_indices)

        sidx = (self.seed or 0) % 16
        if qtype == "count_pieces":
            _COUNT_PIECES = [
                "Top: labelled sub-polycubes. Bottom: target compound. How many sub-polycubes are used to assemble the target? Answer with a single integer.",
                "Look at the sub-polycubes (top) and the target compound (bottom). How many of the pieces are used to build the target?",
                "Count the number of sub-polycubes (shown at top) that are used to assemble the target compound shown at bottom. Answer with an integer.",
                "The top row shows labelled sub-polycubes; the bottom shows the target. How many pieces from the top are used in the target? Integer answer.",
                "Given the sub-polycubes at top and the target below, how many of the sub-polycubes contribute to the target compound? Integer.",
                "How many of the labelled pieces (top) appear in the target compound (bottom)? Give a single integer.",
                "Count how many sub-polycubes from the top row were used to build the target compound at the bottom. Integer answer.",
                "Based on the figure (sub-polycubes on top, target on bottom), how many pieces went into the target? Single integer.",
                "The target compound below is assembled from a subset of the labelled sub-polycubes above. How many pieces are in that subset? Integer.",
                "Determine the number of sub-polycubes (top) that were combined to form the target compound (bottom). Integer answer.",
                "How many of the sub-polycubes shown at top are part of the target assembly shown at bottom? Give an integer.",
                "From the top (sub-polycubes) and bottom (target), report how many sub-polycubes were used in the assembly. Single integer.",
                "Count: how many of the labelled top-row pieces are used to build the bottom target? Integer answer.",
                "Observing the top (pieces) and bottom (target), how many pieces were used in the target's assembly? Integer.",
                "How many of the available sub-polycubes (top) are involved in constructing the target (bottom)? Answer as an integer.",
                "Report the integer count of sub-polycubes (top) that appear in the target compound (bottom).",
            ]
            q = _COUNT_PIECES[sidx]
            answer = str(len(subset_indices))
            image = self._render(pieces, target, labels)
            return q, answer, image

        if qtype == "uses_piece_x":
            pick = rng.randint(0, n_pieces - 1)
            in_subset = pick in subset_indices
            _USES_X = [
                f"Top: labelled sub-polycubes. Bottom: target compound. Is sub-polycube {labels[pick]} used in the assembly? Answer 'yes' or 'no'.",
                f"Does the target compound shown at bottom use sub-polycube {labels[pick]}? Answer 'yes' or 'no'.",
                f"Is piece {labels[pick]} needed to build the target compound below? Answer 'yes' or 'no'.",
                f"Looking at the target (bottom), is sub-polycube {labels[pick]} part of its assembly? Reply 'yes' or 'no'.",
                f"Among the labelled sub-polycubes, does piece {labels[pick]} appear in the target compound? 'yes' or 'no'.",
                f"Is sub-polycube labelled {labels[pick]} included in the assembly of the target shown below? 'yes' / 'no'.",
                f"Check whether piece {labels[pick]} is one of the sub-polycubes used in the target compound. Answer 'yes' or 'no'.",
                f"Is {labels[pick]} used in building the target compound? Answer yes or no.",
                f"Given the target compound (bottom), does it incorporate sub-polycube {labels[pick]}? 'yes' or 'no'.",
                f"Is piece {labels[pick]} one of the components of the target? Respond 'yes' or 'no'.",
                f"Tell me whether sub-polycube {labels[pick]} is part of the target assembly. Answer 'yes' or 'no'.",
                f"Does sub-polycube {labels[pick]} contribute to the target compound shown? Answer 'yes' or 'no'.",
                f"Is {labels[pick]} among the pieces assembled into the target compound? 'yes' / 'no'.",
                f"From the figure, is sub-polycube {labels[pick]} used to form the target? 'yes' or 'no'.",
                f"Is piece labelled {labels[pick]} present in the assembly of the target compound? Reply yes or no.",
                f"Determine whether sub-polycube {labels[pick]} is used in the target compound. 'yes' or 'no'.",
            ]
            q = _USES_X[sidx]
            answer = "yes" if in_subset else "no"
            image = self._render(pieces, target, labels)
            return q, answer, image

        options = self._build_subset_options(rng, n_pieces, subset_indices, labels)
        rng.shuffle(options)
        correct_idx = options.index(correct_subset_str)
        correct_letter = chr(ord("A") + correct_idx)

        opt_str = ", ".join(f"{chr(ord('A') + i)}) {v}" for i, v in enumerate(options))
        _SUBSET = [
            f"Which combination forms the target? Options: {opt_str}.",
            f"Pick the set of pieces that, combined, form the target. Options: {opt_str}.",
            f"Which subset of the sub-polycubes assembles into the target compound? Options: {opt_str}.",
            f"Identify the combination of labelled pieces that build the target. Options: {opt_str}.",
            f"Which of the option sets of sub-polycubes produces the target compound? Options: {opt_str}.",
            f"Choose the correct group of pieces that assembles into the target. Options: {opt_str}.",
            f"Which labeled subset of pieces combines to form the target shown? Options: {opt_str}.",
            f"Find the collection of sub-polycubes whose union equals the target compound. Options: {opt_str}.",
            f"Among the options, which set of pieces unions to form the target? Options: {opt_str}.",
            f"Select the subset of sub-polycubes that produces the target compound. Options: {opt_str}.",
            f"Which option correctly lists the sub-polycubes used in the target? Options: {opt_str}.",
            f"Pick the option whose pieces join to create the target compound. Options: {opt_str}.",
            f"Identify the assembling subset of labelled pieces. Options: {opt_str}.",
            f"Which group of pieces makes up the target compound shown? Options: {opt_str}.",
            f"Determine which subset, when combined, equals the target. Options: {opt_str}.",
            f"Among the choices, which subset of labelled sub-polycubes forms the target compound? Options: {opt_str}.",
        ]
        stem = _SUBSET[sidx]
        q = f"{stem} Answer with a single letter (A/B/C/D)."
        image = self._render(pieces, target, labels)
        return q, correct_letter, image

    def _generate_l0(self, sub_rng):
        rng = sub_rng  # use sub_rng for all L0 randomness
        """Trivial L0 path — 2D flat grid rendering instead of 3D isometric.

        Two tiny flat pieces (1-2 cells) shown as colored squares on a 2D grid.
        The target is their union. Always 4-way MCQ: "How many squares are in
        the target shape?" with obviously wrong distractors.
        This avoids 3D isometric rendering that confuses models into truncation.
        """
        # 2D flat piece templates: list of (row, col) cells
        choices_2d = [
            # 1 + 1 = 2 cells (domino)
            ([(0, 0)], [(0, 1)], [(0, 0), (0, 1)]),
            # 1 + 2 = 3 cells (row)
            ([(0, 0)], [(0, 1), (0, 2)], [(0, 0), (0, 1), (0, 2)]),
            # 2 + 1 = 3 cells (L)
            ([(0, 0), (1, 0)], [(1, 1)], [(0, 0), (1, 0), (1, 1)]),
            # 2 + 2 = 4 cells (2x2)
            ([(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)]),
            # 1 + 1 = 2 cells (vertical)
            ([(0, 0)], [(1, 0)], [(0, 0), (1, 0)]),
        ]
        piece_a_cells, piece_b_cells, target_cells = rng.choice(choices_2d)
        total_count = len(target_cells)

        # MCQ: "How many squares in the combined shape?"
        correct_str = str(total_count)
        opts = [total_count]
        for d in [1, -1, 2, -2, 3]:
            v = total_count + d
            if v >= 1 and v not in opts:
                opts.append(v)
            if len(opts) >= 4:
                break
        while len(opts) < 4:
            opts.append(total_count + len(opts))
        opts = opts[:4]
        rng.shuffle(opts)
        correct_idx = opts.index(total_count)
        correct_letter = chr(ord("A") + correct_idx)

        opt_str = ", ".join(f"{chr(ord('A') + i)}) {v}"
                            for i, v in enumerate(opts))
        _L0_TEMPLATES = [
            f"Piece 1 and Piece 2 are combined to form a target shape (shown below). How many unit squares are in the target? Options: {opt_str}. Answer with a single letter (A/B/C/D).",
            f"Piece 1 and Piece 2 together make the target (below). Count the unit squares in the combined target. Options: {opt_str}. Respond with a single letter.",
            f"Two pieces (1 and 2) are joined into a target shape shown below. How many cells (unit squares) does the target contain? Options: {opt_str}. One letter answer.",
            f"Combine Piece 1 and Piece 2 as shown to get the target. How many unit squares make up the target? Options: {opt_str}. Answer with A/B/C/D.",
            f"Piece 1 + Piece 2 = target (shown). Count the unit cells in the target shape. Options: {opt_str}. Give a single letter.",
            f"Given the two pieces combine into the target below, how many unit squares are in that target? Options: {opt_str}. Single-letter answer.",
            f"The target shape (below) is formed by merging Piece 1 and Piece 2. What is the total count of unit squares? Options: {opt_str}. Answer: one letter.",
            f"Piece 1 and Piece 2 merge to form the target (shown). Count the squares in the target shape. Options: {opt_str}. Respond with a single letter.",
            f"How many unit squares are in the target formed by combining Piece 1 and Piece 2 (shown below)? Options: {opt_str}. Answer with one letter.",
            f"Two labeled pieces combine into the target shown below. Determine the number of unit cells in the target. Options: {opt_str}. Single letter.",
            f"The combined target (Piece 1 ∪ Piece 2) is shown. How many unit squares does it have? Options: {opt_str}. One letter.",
            f"Count the cells in the target shape formed by joining Piece 1 and Piece 2. Options: {opt_str}. Answer with a single letter.",
            f"Piece 1 plus Piece 2 gives the target (shown). Total number of unit squares in the target? Options: {opt_str}. A single letter answer.",
            f"Looking at the combined target below (Piece 1 + Piece 2), how many unit cells does it contain? Options: {opt_str}. One letter.",
            f"The target shape at the bottom is Piece 1 joined with Piece 2. Count the unit squares in the target. Options: {opt_str}. Single letter.",
            f"From the figure, Piece 1 and Piece 2 combine into the target. How many unit squares make up the target shape? Options: {opt_str}. Answer with a letter.",
        ]
        sidx = (self.seed or 0) % 16
        q = _L0_TEMPLATES[sidx]
        image = self._render_l0_2d(piece_a_cells, piece_b_cells,
                                    target_cells, rng)
        return q, correct_letter, image

    def _render_l0_2d(self, piece_a, piece_b, target, rng):
        """Render L0 as flat 2D colored grids (no 3D isometric)."""
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        rng.shuffle(palette)
        color_a = palette[0]
        color_b = palette[1 % len(palette)]
        edge_col = "#222222"

        fig, axes = plt.subplots(1, 3, figsize=(9 * sc, 3.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        panels = [
            ("Piece 1", piece_a, color_a),
            ("Piece 2", piece_b, color_b),
            ("Target", target, None),
        ]
        for ax, (title, cells, fill_color) in zip(axes, panels):
            ax.set_facecolor(style["bg_color"])
            ax.set_aspect("equal")
            ax.axis("off")
            if not cells:
                continue
            rows = [r for r, c in cells]
            cols = [c for r, c in cells]
            max_r, max_c = max(rows) + 1, max(cols) + 1
            for r, c in cells:
                if fill_color is None:
                    # Target: color by origin
                    if (r, c) in piece_a:
                        fc = color_a
                    elif (r, c) in piece_b:
                        fc = color_b
                    else:
                        fc = "#cccccc"
                else:
                    fc = fill_color
                from matplotlib.patches import Rectangle
                rect = Rectangle((c, max_r - 1 - r), 1, 1,
                                  facecolor=fc, edgecolor=edge_col,
                                  linewidth=2)
                ax.add_patch(rect)
            ax.set_xlim(-0.3, max_c + 0.3)
            ax.set_ylim(-0.3, max_r + 0.3)
            ax.set_title(title, fontsize=style["font_size_base"] + 2,
                         fontweight="bold")

        fig.suptitle("Combine the pieces",
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _pick_subset(self, rng, n_pieces, level):
        # At low levels, prefer small subsets (single piece or pair)
        if level == 1:
            size = rng.choice([2, 3])
        elif level == 2:
            size = rng.choice([2, 3])
        else:
            size = rng.randint(2, n_pieces)
        size = min(size, n_pieces)
        return sorted(rng.sample(range(n_pieces), size))

    def _try_assemble_subset(self, rng, pieces, subset_indices, allow_rotation):
        """Place each piece, avoiding overlap, producing a connected compound."""
        # We build incrementally. The first piece is at origin, subsequent
        # pieces are translated to a random valid position adjacent to the
        # assembly.
        placed_cubes: Set[Tuple[int, int, int]] = set()
        out: List[Tuple[int, int, int]] = []

        for i, idx in enumerate(subset_indices):
            piece = pieces[idx]
            if allow_rotation:
                rotations = _all_rotations(piece)
                rng.shuffle(rotations)
            else:
                rotations = [piece]

            placed = False
            for rot in rotations[:12]:
                if i == 0:
                    # First piece at origin
                    for c in rot:
                        placed_cubes.add(c)
                        out.append(c)
                    placed = True
                    break
                # Try translating so that some cube is adjacent to the assembly
                for attempt in range(50):
                    anchor = rng.choice(list(placed_cubes))
                    ax, ay, az = anchor
                    dx, dy, dz = rng.choice(
                        [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                         (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                    )
                    target_x, target_y, target_z = ax + dx, ay + dy, az + dz
                    # Pick a cube from rot to align with target
                    align_cube = rng.choice(rot)
                    tx = target_x - align_cube[0]
                    ty = target_y - align_cube[1]
                    tz = target_z - align_cube[2]
                    translated = _translate(rot, tx, ty, tz)
                    if any(c in placed_cubes for c in translated):
                        continue
                    # Valid
                    for c in translated:
                        placed_cubes.add(c)
                        out.append(c)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                return None
        return out

    def _build_subset_options(self, rng, n_pieces, correct_indices, labels):
        """Build 4 textual subset options, with the correct one in there."""
        correct_str = "+".join(labels[i] for i in correct_indices)
        options = [correct_str]
        all_subsets = []
        for mask in range(1, 1 << n_pieces):
            bits = [i for i in range(n_pieces) if (mask >> i) & 1]
            s = "+".join(labels[i] for i in bits)
            all_subsets.append(s)
        for s in all_subsets:
            if s not in options:
                options.append(s)
            if len(options) == 4:
                break
        # If still short, pad with duplicates just in case
        while len(options) < 4:
            options.append("none")
        return options[:4]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _draw_polycube(self, ax, cubes, palette, title=None):
        shifted = sorted(_normalize(cubes), key=lambda c: -(c[0] + c[1] - c[2]))
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
        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=4)

    def _render(self, pieces, target, labels,
                target_provenance=None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        sub_rng = (self._sub_rng if hasattr(self, "_sub_rng")
                   and self._sub_rng is not None else random.Random())
        n_pieces = len(pieces)
        ncols = max(n_pieces, 2)
        fig = plt.figure(figsize=(3 * ncols * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        # BUGFIX: pick N distinct piece hues and shade each into a safe
        # top/left/right triplet. Previously `palette[i:]+palette[:i]` was
        # used which sometimes put a very dark color as the top face (or
        # pure black from the high-contrast palette) and made face contrast
        # random/unpredictable.
        bases = _piece_bases(style, sub_rng, n_pieces + 1)
        piece_palettes = []
        for i, piece in enumerate(pieces):
            ax = fig.add_subplot(2, ncols, i + 1)
            ax.set_facecolor(style["bg_color"])
            pal = _shaded_face_triplet(bases[i])
            piece_palettes.append(pal)
            self._draw_polycube(ax, piece, pal,
                                title=f"Piece {labels[i]}")

        # Bottom: target (spans full width)
        ax_t = fig.add_subplot(2, 1, 2)
        ax_t.set_facecolor(style["bg_color"])
        if target_provenance is not None:
            # Color-code the target using each piece's palette so that the
            # cubes in the target inherit the color of their source piece.
            self._draw_colored_target(
                ax_t, target, target_provenance, piece_palettes, labels
            )
            ax_t.set_title("Target compound", fontsize=11,
                           fontweight="bold", pad=4)
        else:
            # Use a distinct base for the target (last color in `bases`).
            tgt_palette = _shaded_face_triplet(bases[-1])
            self._draw_polycube(ax_t, target, tgt_palette, title="Target compound")

        suptitle = "Cube Assembly"
        if hasattr(self, "_sub_rng") and self._sub_rng is not None:
            suptitle = self._sub_rng.choice([
                "Cube Assembly", "Polycube Assembly",
                "Which pieces build the target?",
                "Compound polycube puzzle",
            ])
        fig.suptitle(suptitle,
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_colored_target(self, ax, target, provenance,
                              piece_palettes, labels):
        """Draw the target compound with each cube colored by its source
        piece. `provenance` is a dict label->list_of_cubes_in_target.
        """
        # Compute normalization offset for the target as in _draw_polycube.
        all_cubes = list(target)
        xs = [c[0] for c in all_cubes]
        ys = [c[1] for c in all_cubes]
        zs = [c[2] for c in all_cubes]
        minx, miny, minz = min(xs), min(ys), min(zs)

        draw_order = []
        for label, cubes in provenance.items():
            palette_idx = labels.index(label)
            pal = piece_palettes[palette_idx]
            for c in cubes:
                draw_order.append((c, pal))

        # Sort by depth so back cubes draw first
        draw_order.sort(key=lambda item: -(item[0][0] + item[0][1] + item[0][2]))

        for (cube, pal) in draw_order:
            x = cube[0] - minx
            y = cube[1] - miny
            z = cube[2] - minz
            top_color = pal[0]
            left_color = pal[1 % len(pal)]
            right_color = pal[2 % len(pal)]

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
        for (x0, y0, z0) in all_cubes:
            x = x0 - minx
            y = y0 - miny
            z = z0 - minz
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
# Sample generation
# ====================================================================== #

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = CubeAssemblyQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[cube_assembly] L{level} seed{seed} FAILED")
                continue
            img = env.render()
            out = f"/tmp/env_check/cube_assembly_seed{seed}_L{level}.png"
            img.save(out)
            print(f"saved {out}")
            print(f"  Q: {env.get_instruction()[:160]}")
            print(f"  A: {env._answer}")
