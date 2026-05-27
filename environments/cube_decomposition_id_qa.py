"""
Cube Decomposition ID QA environment.

Target regression: spatial-vision CubeReconstruction -3.33,
CubeAssembly -3.33 (inverse direction). This env teaches the *reverse*
of cube_assembly_qa: given a compound polycube ("target"), pick which
MCQ option shows a valid decomposition of that target into sub-pieces.

Two independent difficulty axes scale with level:
  * n_pieces in the decomposition (2 -> 3)
  * number of cubes per piece (+ rotation)

All questions are 4-way MCQ with a single letter answer (A/B/C/D).
"""
import math
import random
from typing import Dict, Optional, Tuple, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    """Isometric projection (same as isometric_counting_qa)."""
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _cube_face_palette(style, rng):
    """BUGFIX: safe 3-color face palette for isometric cubes (top > right > left
    luminance). Avoids #000000 (black faces blending into shadow) and clamps
    base lightness so faces always differ clearly regardless of palette."""
    import colorsys
    pal = [c for c in style['palette']
           if c.lower() not in ('#000000', '#010101', '#0a0a0a', '#ffffff',
                                '#fefefe', '#f1faee')]
    if not pal:
        pal = ['#5dade2', '#48c9b0', '#ec7063', '#f4d03f']
    base = rng.choice(pal)
    h = base.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
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

def _translate(cubes, dx, dy, dz):
    return [(x + dx, y + dy, z + dz) for (x, y, z) in cubes]

def _normalize(cubes):
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

class CubeDecompositionIdQA(StandaloneVisualEnv):
    ENV_NAME = "cube_decomposition_id"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            # L0-L2: 2D flat grid counting (no 3D)
            return {"mode": "2d_count", "level": level}
        # L3+: 3D decomposition (original task)
        n_pieces = 2 if level <= 6 else 3
        if level <= 4:
            piece_size = 3
        elif level <= 7:
            piece_size = 4
        else:
            piece_size = 4
        allow_rotation = level >= 5
        tight_distractors = level >= 7
        return {
            "mode": "3d_decompose",
            "n_pieces": n_pieces,
            "piece_size": piece_size,
            "allow_rotation": allow_rotation,
            "tight_distractors": tight_distractors,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(20):
            level = int(parameter.get("level", 0))
            cfg = self._level_config(level)
            if cfg.get("mode") == "2d_count":
                result = self._try_generate_2d_count(cfg)
            else:
                result = self._try_generate(parameter)
            if result is not None:
                return result
        return None

    # ---- 2D flat-grid counting (L0-L2) ----
    def _try_generate_2d_count(self, cfg: Dict):
        """Show a 2D colored grid shape, ask how many squares it contains. MCQ."""
        level = cfg["level"]
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        # Difficulty: L0 = 3-5 squares, L1 = 4-7, L2 = 5-9
        if level == 0:
            n_cells = rng.randint(3, 5)
        elif level == 1:
            n_cells = rng.randint(4, 7)
        else:
            n_cells = rng.randint(5, 9)

        self._primary_complexity_feature = n_cells

        # Generate a connected polyomino on a 2D grid
        cells = {(0, 0)}
        while len(cells) < n_cells:
            anchor = rng.choice(list(cells))
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            cells.add((anchor[0] + dx, anchor[1] + dy))

        # Build MCQ: correct answer is n_cells, distractors are nearby wrong counts
        labels = ["A", "B", "C", "D"]
        correct = n_cells
        distractors = set()
        for d in [correct - 2, correct - 1, correct + 1, correct + 2]:
            if d >= 1 and d != correct:
                distractors.add(d)
        distractors = sorted(distractors)
        while len(distractors) < 3:
            distractors.append(correct + len(distractors) + 2)
        distractors = rng.sample(distractors, 3)
        options = [correct] + distractors
        rng.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = labels[correct_idx]

        # Render 2D grid
        style = self._random_style()
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(5, 5), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        fill_color = palette[0]
        for (cx, cy) in cells:
            rect = plt.Rectangle((cx, cy), 1, 1, facecolor=fill_color,
                                  edgecolor="black", linewidth=2)
            ax.add_patch(rect)

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        ax.set_xlim(min(xs) - 0.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 0.5, max(ys) + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        opt_str = "  ".join(f"{labels[i]}) {options[i]}" for i in range(4))
        ax.set_title(f"How many squares?\n{opt_str}",
                     fontsize=14, fontweight="bold", pad=10)
        fig.tight_layout()
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        _L0 = [
            f"The image shows a shape made of colored unit squares on a flat grid. How many unit squares make up this shape? Options: {opt_str}. Answer with a single letter (A/B/C/D).",
            f"A flat grid shape made of colored unit squares is shown. How many unit squares form this shape? Options: {opt_str}. One-letter answer.",
            f"Count the unit squares comprising the colored shape on the grid. Options: {opt_str}. Answer with a single letter.",
            f"The figure displays a shape built from colored unit squares on a grid. What is the total count of unit squares? Options: {opt_str}. Single letter.",
            f"How many unit squares does the colored shape in the image contain? Options: {opt_str}. Answer with one letter (A/B/C/D).",
            f"Determine the number of unit squares that make up the colored shape on the flat grid. Options: {opt_str}. Single letter.",
            f"Look at the colored unit-square shape in the image. Count its unit squares. Options: {opt_str}. Respond with a single letter.",
            f"The shape in the image is composed of colored unit squares. Count them. Options: {opt_str}. Reply with a letter.",
            f"In the image, a shape is formed from colored unit squares on a grid. How many squares is it? Options: {opt_str}. One letter.",
            f"Identify the number of unit squares in the colored grid shape shown. Options: {opt_str}. Give a single letter answer.",
            f"How many total unit squares form the shape shown on the colored grid? Options: {opt_str}. Single-letter answer.",
            f"The image contains a flat-grid shape of colored unit squares. Count the unit squares. Options: {opt_str}. A/B/C/D.",
            f"Count how many unit squares the colored grid shape contains. Options: {opt_str}. Answer with one letter.",
            f"From the image, determine the count of unit squares that build the shape. Options: {opt_str}. Single letter.",
            f"The colored shape on the grid is made up of unit squares. How many unit squares total? Options: {opt_str}. Answer: one letter.",
            f"Report the number of unit squares constituting the shape in the image. Options: {opt_str}. Respond with a single letter.",
        ]
        sidx = (self.seed or 0) % 16
        q = _L0[sidx]
        return q, correct_letter, img

    def _try_generate(self, parameter: Dict):
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        n_pieces = cfg["n_pieces"]
        piece_size = cfg["piece_size"]
        self._primary_complexity_feature = n_pieces * 10 + piece_size

        # 1. Generate pieces
        pieces = []
        for _ in range(n_pieces):
            size = piece_size
            for attempt in range(30):
                p = _random_polycube(rng, size)
                if len(p) == size and _is_connected(p):
                    pieces.append(p)
                    break
            else:
                return None

        # 2. Assemble target by placing pieces adjacently
        placed_set: Set[Tuple[int, int, int]] = set()
        piece_placements = []
        for i, piece in enumerate(pieces):
            rotations = _all_rotations(piece) if cfg["allow_rotation"] else [piece]
            rng.shuffle(rotations)
            placed = False
            for rot in rotations[:10]:
                if i == 0:
                    for c in rot:
                        placed_set.add(c)
                    piece_placements.append(list(rot))
                    placed = True
                    break
                for attempt in range(50):
                    anchor = rng.choice(list(placed_set))
                    dxdy = rng.choice([
                        (1, 0, 0), (-1, 0, 0), (0, 1, 0),
                        (0, -1, 0), (0, 0, 1), (0, 0, -1)
                    ])
                    tgt = (anchor[0] + dxdy[0], anchor[1] + dxdy[1],
                           anchor[2] + dxdy[2])
                    align = rng.choice(rot)
                    tx = tgt[0] - align[0]
                    ty = tgt[1] - align[1]
                    tz = tgt[2] - align[2]
                    translated = _translate(rot, tx, ty, tz)
                    if any(c in placed_set for c in translated):
                        continue
                    for c in translated:
                        placed_set.add(c)
                    piece_placements.append(list(translated))
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                return None

        target = []
        for pp in piece_placements:
            target.extend(pp)
        if not _is_connected(target):
            return None

        # 3. Build MCQ options
        # Correct: pieces as a set of sub-polycubes (normalized)
        # Distractors: (a) swap one piece for a different polycube,
        # (b) use fewer / more cubes total, (c) rearrange cube counts.
        labels_letter = ["A", "B", "C", "D"]
        correct_pieces = [list(_normalize(p)) for p in pieces]
        options: List[List[List[Tuple[int, int, int]]]] = [correct_pieces]

        # Distractor strategies
        d_target = 3  # need 3 distractors
        d_attempts = 0
        while len(options) < 4 and d_attempts < 40:
            d_attempts += 1
            d_type = rng.choice(["swap_piece", "change_size", "swap_piece"])
            if d_type == "swap_piece":
                # Pick one piece, replace with a different polycube of same size
                idx = rng.randint(0, len(pieces) - 1)
                orig = pieces[idx]
                for _t in range(15):
                    alt = _random_polycube(rng, len(orig))
                    if not _shapes_equal(alt, orig):
                        new_set = [list(_normalize(p)) for p in pieces]
                        new_set[idx] = list(_normalize(alt))
                        if not any(self._option_eq(new_set, o) for o in options):
                            options.append(new_set)
                            break
            else:  # change_size
                # Add a cube to one piece, remove from another
                idx1 = rng.randint(0, len(pieces) - 1)
                idx2 = (idx1 + 1) % len(pieces)
                alt1 = _random_polycube(rng, len(pieces[idx1]) + 1)
                alt2 = _random_polycube(rng, max(1, len(pieces[idx2]) - 1))
                if not alt2:
                    continue
                new_set = [list(_normalize(p)) for p in pieces]
                new_set[idx1] = list(_normalize(alt1))
                new_set[idx2] = list(_normalize(alt2))
                if not any(self._option_eq(new_set, o) for o in options):
                    options.append(new_set)

        if len(options) < 4:
            return None

        rng.shuffle(options)
        correct_idx = next(
            i for i, o in enumerate(options)
            if self._option_eq(o, correct_pieces)
        )
        correct_letter = labels_letter[correct_idx]

        img = self._render(target, options, labels_letter, rng)
        _DECOMP = [
            "The top image shows a target compound polycube. Each of the four labelled options below shows a possible decomposition of the target into sub-polycubes. Which option correctly decomposes the target (ignoring rotation and position)? Answer with a single letter (A/B/C/D).",
            "A target compound polycube is shown at the top. Four labelled options below give possible decompositions. Pick the one that correctly decomposes the target (rotation/position ignored). Respond with a single letter.",
            "The image (top) shows a target polycube and (below) four labelled decompositions. Which decomposition matches the target (ignoring rotation/position)? Single letter.",
            "Which of the four labelled options below correctly decomposes the target polycube shown at top (disregarding rotation and position)? Answer with a letter (A/B/C/D).",
            "Shown: a target compound (top) and four possible decompositions (labelled A-D). Pick the correct decomposition of the target, ignoring rotation and position. Single letter.",
            "Given the target compound polycube at top and four candidate decompositions below, choose the one that validly decomposes the target (rotation/position do not matter). Answer: one letter.",
            "Four decompositions are shown (labelled A-D); one correctly splits the target compound at top into sub-polycubes. Identify it (rotation and position ignored). Respond with a single letter.",
            "Identify which labelled option below (A/B/C/D) correctly decomposes the target polycube above, disregarding rotation and position. Reply with a single letter.",
            "The target compound is shown at top. Four labelled decomposition candidates follow. Which decomposition is valid (ignoring rotation/position)? One letter.",
            "Which of A, B, C, D correctly splits the target compound polycube (shown at top) into the depicted sub-polycubes? Rotation/position ignored. Answer: a single letter.",
            "Pick the option that properly decomposes the target polycube (top) into sub-polycubes, disregarding rotation and position. Answer with a single letter.",
            "Four labelled decompositions of the target are shown below the target compound. Which is correct (ignoring rotation and translation)? Single letter.",
            "Select the correct decomposition of the target compound from the four labelled options (A-D). Rotation and position don't matter. Answer with one letter.",
            "The top image shows a target compound polycube; below are four possible decompositions. Which labelled option decomposes it correctly? Single letter.",
            "Among the four labelled options below, which one gives the correct decomposition of the target compound at top (up to rotation and position)? One letter.",
            "Given the target polycube and four candidate decompositions (labelled), choose the valid decomposition. Rotation and translation are ignored. Single letter.",
        ]
        sidx = (self.seed or 0) % 16
        q = _DECOMP[sidx]
        return q, correct_letter, img

    def _option_eq(self, opt_a, opt_b) -> bool:
        """Two option entries are equal if they contain the same multiset
        of polycubes (up to rotation)."""
        if len(opt_a) != len(opt_b):
            return False
        remaining = list(opt_b)
        for a in opt_a:
            match = None
            for i, b in enumerate(remaining):
                if _shapes_equal(a, b):
                    match = i
                    break
            if match is None:
                return False
            remaining.pop(match)
        return True

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _draw_polycube(self, ax, cubes, palette, title=None):
        # Painter's algorithm: draw back-to-front so nearer cubes occlude
        # farther ones. In our isometric projection, larger (x+y+z) is
        # farther from the viewer, so sort descending (back first).
        shifted = sorted(_normalize(cubes),
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
                                 edgecolor="black", lw=1.0))
            pts_left = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts_left, closed=True, facecolor=left_color,
                                 edgecolor="black", lw=1.0))
            pts_right = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts_right, closed=True, facecolor=right_color,
                                 edgecolor="black", lw=1.0))

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

    def _draw_option(self, ax, pieces, palette, title):
        """Draw several sub-polycubes side by side inside the given axes."""
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
        # Shift each piece along the x axis so they don't overlap
        offset_x = 0.0
        pad = 1.3
        all_pts = []
        # BUGFIX: per-piece hue shift instead of rotating a luminance triplet
        # (rotating broke the top>right>left luminance cue and sometimes made
        # the top face end up as the left-shade). We hue-rotate the base to
        # give each piece a distinct color while preserving the shaded-face
        # visual cue.
        import colorsys as _cs
        def _rotate_hue(hex_color, delta_h):
            hx = hex_color.lstrip('#')
            rr = int(hx[0:2], 16) / 255
            gg = int(hx[2:4], 16) / 255
            bb = int(hx[4:6], 16) / 255
            hh, ll, sss = _cs.rgb_to_hls(rr, gg, bb)
            hh = (hh + delta_h) % 1.0
            rr2, gg2, bb2 = _cs.hls_to_rgb(hh, ll, sss)
            return '#{:02x}{:02x}{:02x}'.format(
                int(rr2 * 255), int(gg2 * 255), int(bb2 * 255))
        for i, piece in enumerate(pieces):
            norm = list(_normalize(piece))
            # Bounding box
            xs = [c[0] for c in norm]
            ys = [c[1] for c in norm]
            zs = [c[2] for c in norm]
            w = (max(xs) - min(xs) + 1) + 1
            # Shift by offset_x in grid coords
            shifted = [(c[0] + int(offset_x), c[1], c[2]) for c in norm]
            dh = 0.33 * i  # 0, 120deg, 240deg hue rotations between pieces
            pal = [_rotate_hue(palette[0], dh),
                   _rotate_hue(palette[1], dh),
                   _rotate_hue(palette[2], dh)]
            # Draw cubes
            shifted_sorted = sorted(shifted, key=lambda c: -(c[0] + c[1] - c[2]))
            for (x, y, z) in shifted_sorted:
                pts_top = [
                    _iso_project(x, y, z + 1),
                    _iso_project(x + 1, y, z + 1),
                    _iso_project(x + 1, y + 1, z + 1),
                    _iso_project(x, y + 1, z + 1),
                ]
                ax.add_patch(Polygon(pts_top, closed=True, facecolor=pal[0],
                                     edgecolor="black", lw=1.0))
                pts_left = [
                    _iso_project(x, y, z),
                    _iso_project(x, y + 1, z),
                    _iso_project(x, y + 1, z + 1),
                    _iso_project(x, y, z + 1),
                ]
                ax.add_patch(Polygon(pts_left, closed=True, facecolor=pal[1],
                                     edgecolor="black", lw=1.0))
                pts_right = [
                    _iso_project(x, y, z),
                    _iso_project(x + 1, y, z),
                    _iso_project(x + 1, y, z + 1),
                    _iso_project(x, y, z + 1),
                ]
                ax.add_patch(Polygon(pts_right, closed=True, facecolor=pal[2],
                                     edgecolor="black", lw=1.0))
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            all_pts.append(_iso_project(x + dx, y + dy, z + dz))
            offset_x += w + pad

        if all_pts:
            arr = np.array(all_pts)
            margin = 0.6
            ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
            ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)

    def _render(self, target, options, labels_letter, rng=None) -> Image.Image:
        style = self._random_style()
        # BUGFIX: shaded face palette (top > right > left luminance) avoids
        # solid-black top faces from the high-contrast palette and ensures
        # faces are distinguishable across any palette choice.
        if rng is None:
            rng = random.Random()
        palette = _cube_face_palette(style, rng)
        fig = plt.figure(figsize=(10.0, 8.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")

        # Top: target (2 rows tall at top of a 3-row grid)
        ax_t = fig.add_subplot(3, 1, 1)
        ax_t.set_facecolor("#ffffff")
        self._draw_polycube(ax_t, target, palette, title="Target compound")

        # Bottom: 4 options in a 2x2 grid
        for idx in range(4):
            row = 2 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(3, 2, 2 * (row - 1) + col + 1)
            ax.set_facecolor("#ffffff")
            self._draw_option(ax, options[idx], palette,
                              title=f"{labels_letter[idx]}")

        fig.suptitle("Pick the correct decomposition",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b4"
    os.makedirs(out_dir, exist_ok=True)
    env = CubeDecompositionIdQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            status = "OK" if ok else "FAIL"
            print(f"[cube_decomposition_id] L{level} seed{seed}: {status} "
                  f"ans={env._answer!r}")
            if ok:
                env._image.save(
                    f"{out_dir}/cube_decomposition_id_s{seed}_L{level}.png")
