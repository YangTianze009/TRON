"""
Polycube Counting Warmup QA environment.

Goal: give the `isometric_counting` env a trivial starting point at L0, so
that models can actually "get on the scoreboard" before being graded on full
polycube structures. Closes cube counting by
providing a ramp.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2-3 cubes, laid out in a row. Zero occlusion, every cube fully visible
    from the isometric view. Question forced to `count_cubes`.
    Target pass rate: 75-90%.
L1: 3-4 cubes. Simple L-pair / short row + one stacked. No hidden cube.
    Forced `count_cubes`.
    Target pass rate: 65-85%.
L2: 4-5 cubes. Choice of {line, L, stairs, small column}. One cube may be
    behind another but still visible from top.
    Question: count_cubes (70%) / count_visible_faces (30%).
    Target pass rate: 55-75%.
L3: 5-6 cubes. Random shape or wider stairs. No hidden cubes.
    Mix: count_cubes / is_count / count_visible_faces.
    Target pass rate: 45-65%.
L4: 6-7 cubes. Random shape, 0-1 hidden cube. Full mix.
    Target pass rate: 35-55%.
L5: 7-8 cubes. Random shape, 0-1 hidden cube, slight tower bias.
    Target pass rate: 30-50%.
L6: 8-9 cubes. Larger random shape, up to 1 hidden cube.
    Target pass rate: 25-45%.
L7: 9-10 cubes. Up to 2 hidden cubes, stricter distractors on is_count.
    Target pass rate: 20-40%.
L8: 10-12 cubes. Up to 2 hidden cubes, tower/wall biased.
    Target pass rate: 15-30%.
L9: 12-15 cubes. Up to 3 hidden cubes, full random, tight distractors.
    Target pass rate: 10-25%.

======================================================================
Diversity axes (same level, different seed)
======================================================================
  1. Shape family: line / L / stairs / column / base_plus_tower / T /
                   random — 7 families at L2+
  2. Row/col/height extents vary inside per-level bounds
  3. Random rotation: swap axes (6 orientations) via _rotate_axes
  4. Random reflection: mirror x or y
  5. Face color palette: shuffled per seed (top/left/right permuted)
  6. Edge color: 5 dark options
  7. Line width randomized
  8. Question phrasing: 4-6 stems per question type
  9. Title variants: 5 strings
 10. is_count distractor offset distribution (tighter at higher levels)
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

def _iso_project(x, y, z):
    """Convert 3D grid coords to 2D isometric screen coords."""
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _cube_face_palette(style, rng):
    """BUGFIX: safe 3-color face palette for isometric cubes.

    Returns [top, left, right] shaded from a single base color so faces
    are always distinguishable. Filters out #000000 (which otherwise
    rendered solid-black cube faces that merged with shadows) and
    avoids near-white bases that would make all faces look the same.
    """
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

def _rotate_axes(cubes: List[Tuple[int, int, int]], perm: Tuple[int, int, int],
                 signs: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Permute and/or flip axes for additional per-seed diversity."""
    out = []
    for c in cubes:
        a = [c[perm[0]] * signs[0], c[perm[1]] * signs[1], c[perm[2]] * signs[2]]
        out.append(tuple(a))
    return out

class PolycubeCountingWarmupQA(StandaloneVisualEnv):
    """Warmup polycube counting, ramp from 2 trivially-visible cubes to 15
    partially-occluded cubes. All 10 levels supported."""

    ENV_NAME = "polycube_counting_warmup"

    QUESTION_TYPES = [
        "count_cubes",           # integer
        "count_visible_faces",   # integer
        "is_count",              # yes/no
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Return difficulty config per level."""
        level = max(0, min(9, int(level)))
        # L0-L2: 2D flat grid counting (no 3D isometric)
        if level <= 2:
            return dict(mode="2d_flat", level=level)
        # L3+: 3D isometric counting (original task)
        if level == 3:
            return dict(n_min=5, n_max=6,
                        shapes=["line", "l", "stairs", "column",
                                "base_plus_tower", "t_shape"],
                        force_qtype="count_cubes", qtype_weights=None,
                        allow_hidden=False, max_hidden=0,
                        offsets=[-2, -1, 1, 2])
        if level == 4:
            return dict(n_min=6, n_max=7,
                        shapes=["l", "stairs", "base_plus_tower", "t_shape"],
                        force_qtype="count_cubes", qtype_weights=None,
                        allow_hidden=False, max_hidden=0,
                        offsets=[-2, -1, 1, 2])
        if level == 5:
            return dict(n_min=7, n_max=8,
                        shapes=["random", "stairs", "base_plus_tower",
                                "t_shape", "tower_mix"],
                        force_qtype=None, qtype_weights=[7, 0, 3],
                        allow_hidden=True, max_hidden=1,
                        offsets=[-1, 1, -2, 2])
        if level == 6:
            return dict(n_min=8, n_max=9,
                        shapes=["random", "tower_mix", "base_plus_tower"],
                        force_qtype=None, qtype_weights=[6, 0, 4],
                        allow_hidden=True, max_hidden=1,
                        offsets=[-1, 1])
        if level == 7:
            return dict(n_min=9, n_max=10,
                        shapes=["random", "tower_mix"],
                        force_qtype=None, qtype_weights=[4, 3, 3],
                        allow_hidden=True, max_hidden=2,
                        offsets=[-1, 1])
        if level == 8:
            return dict(n_min=10, n_max=12,
                        shapes=["random", "tower_mix"],
                        force_qtype=None, qtype_weights=[4, 2, 4],
                        allow_hidden=True, max_hidden=2,
                        offsets=[-1, 1])
        return dict(n_min=12, n_max=15,
                    shapes=["random"],
                    force_qtype=None, qtype_weights=[5, 2, 3],
                    allow_hidden=True, max_hidden=3,
                    offsets=[-1, 1])

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        if cfg.get("mode") == "2d_flat":
            return self._generate_2d_flat(cfg)
        for _ in range(20):
            try:
                result = self._try_generate(parameter)
                if result is not None:
                    self._primary_complexity_feature = len(result[1]) + int(parameter.get("level", 0)) * 2
                    return result
            except Exception:
                continue
        return None

    def _generate_2d_flat(self, cfg: Dict):
        """2D flat grid counting for L0-L2. Each cell is numbered."""
        level = cfg["level"]
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        if level == 0:
            n_cells = rng.randint(2, 4)
        elif level == 1:
            n_cells = rng.randint(3, 6)
        else:
            n_cells = rng.randint(5, 8)
        self._primary_complexity_feature = n_cells

        # Generate connected polyomino
        cells = {(0, 0)}
        while len(cells) < n_cells:
            anchor = rng.choice(list(cells))
            dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            cells.add((anchor[0] + dx, anchor[1] + dy))
        cells = sorted(cells)

        # Render 2D flat grid with numbered cells
        style = self._random_style()
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(5, 5), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Do NOT number the cells — that is direct answer leakage (model
        # just reads the largest integer). Just draw filled squares.
        fill_color = palette[0]
        for (cx, cy) in cells:
            rect = plt.Rectangle((cx, cy), 1, 1, facecolor=fill_color,
                                  edgecolor="black", linewidth=2.5)
            ax.add_patch(rect)

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        ax.set_xlim(min(xs) - 0.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 0.5, max(ys) + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Count the colored squares", fontsize=14, fontweight="bold")
        fig.tight_layout()
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = rng.choice([
            "How many colored squares are shown in the image? Answer with a single integer.",
            "Count the total number of colored unit squares in the figure. Reply with an integer.",
            "The image shows a shape made of colored unit squares. How many squares are there in total?",
        ])
        return q, str(n_cells), img

    def _try_generate(self, parameter: Dict):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        # --- Question type ---
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            if cfg["force_qtype"] is not None:
                qtype = cfg["force_qtype"]
            else:
                qtype = rng.choices(
                    self.QUESTION_TYPES, weights=cfg["qtype_weights"]
                )[0]

        # --- Pick a shape ---
        shape = sub_rng.choice(cfg["shapes"])
        n_target = sub_rng.randint(cfg["n_min"], cfg["n_max"])
        cubes = self._build_shape(sub_rng, shape, n_target, cfg)
        if cubes is None or len(cubes) < 2:
            return None

        # --- Random axis permutation / sign flip for orientation variety ---
        perm_choices = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0)]
        perm = sub_rng.choice(perm_choices)
        # Only flip xy, keep z upright
        signs = (sub_rng.choice([1, -1]), sub_rng.choice([1, -1]), 1)
        cubes = _rotate_axes(cubes, perm, signs)

        total = len(cubes)
        visible_faces = self._count_visible_faces(cubes)

        # --- Question text & answer ---
        if qtype == "count_cubes":
            stem = rng.choice([
                "How many unit cubes are shown in this structure? Answer with a single integer.",
                "Count the total number of small cubes in the figure above. Reply with an integer.",
                "The structure in the image is built from identical unit cubes. How many unit cubes are there in total?",
                "The image shows an isometric view of a 3D arrangement of unit cubes. How many cubes does the structure contain?",
                "What is the total number of unit cubes making up the shape in the image?",
            ])
            q = stem
            answer = str(total)
        elif qtype == "count_visible_faces":
            stem = rng.choice([
                "Count the total number of unit-square faces that are visible from outside the structure. Only faces not hidden by another cube count. Answer with a single integer.",
                "Look at the isometric view. Counting only the unit-square faces visible from outside (top, left, right of each exposed cube), how many visible faces are there in total?",
                "How many unit-square faces of the cubes are exposed (not covered by another cube)? Answer with an integer.",
            ])
            q = stem
            answer = str(visible_faces)
        else:  # is_count
            offset = sub_rng.choice(cfg["offsets"])
            guess = max(1, total + offset)
            # flip some to match — so roughly 50/50 yes/no
            if sub_rng.random() < 0.5:
                guess = total
            noun = "cube" if guess == 1 else "cubes"
            stem = rng.choice([
                f"Does this structure contain exactly {guess} unit {noun}? Answer yes or no.",
                f"Is the total number of unit cubes in the image equal to {guess}? Answer yes or no.",
                f"Someone claims that the figure is built from {guess} unit {noun}. Is that correct? Answer yes or no.",
            ])
            q = stem
            answer = "yes" if guess == total else "no"

        image = self._draw(cubes, sub_rng, level)
        return q, answer, image

    # ------------------------------------------------------------------ #
    # Shape constructors
    # ------------------------------------------------------------------ #

    def _build_shape(self, rng: random.Random, shape: str, n: int,
                     cfg: Dict) -> Optional[List[Tuple[int, int, int]]]:
        if shape == "line":
            return [(i, 0, 0) for i in range(n)]
        if shape == "l":
            arm = max(2, n - 1)
            cubes = [(i, 0, 0) for i in range(arm)]
            while len(cubes) < n:
                cubes.append((arm - 1, len(cubes) - arm + 1, 0))
            return cubes
        if shape == "stairs":
            cubes = []
            for i in range(n):
                cubes.append((i, 0, i))
            return cubes
        if shape == "column":
            return [(0, 0, i) for i in range(n)]
        if shape == "base_plus_tower":
            base_n = max(2, n // 2)
            cubes = [(i, 0, 0) for i in range(base_n)]
            for k in range(n - base_n):
                cubes.append((0, 0, k + 1))
            return cubes
        if shape == "t_shape":
            arm = max(2, n - 3)
            cubes = [(i, 0, 0) for i in range(arm)]
            mid = arm // 2
            for dy in range(1, min(4, n - arm) + 1):
                cubes.append((mid, dy, 0))
                if len(cubes) >= n:
                    break
            while len(cubes) < n:
                cubes.append((mid, 0, 1 + (len(cubes) - arm)))
            return cubes[:n]
        if shape == "tower_mix":
            # a base of some cubes with 1-3 towers
            base_n = max(3, n * 2 // 3)
            cubes = []
            # small 2xk footprint
            w = max(2, (base_n + 1) // 2)
            for i in range(w):
                for j in range(min(2, base_n - i * 2)):
                    if len(cubes) < base_n:
                        cubes.append((i, j, 0))
            # stack towers on some existing
            while len(cubes) < n:
                base = rng.choice(cubes[:min(len(cubes), base_n)])
                # find the current top of that column
                col = [c for c in cubes if c[0] == base[0] and c[1] == base[1]]
                top_z = max(c[2] for c in col)
                cubes.append((base[0], base[1], top_z + 1))
            return cubes
        if shape == "random":
            return self._random_shape(rng, n, cfg)
        return None

    def _random_shape(self, rng: random.Random, count: int, cfg: Dict):
        cubes = {(0, 0, 0)}
        tries = 0
        max_hidden = cfg.get("max_hidden", 0)
        while len(cubes) < count and tries < 200:
            tries += 1
            seed_cube = rng.choice(list(cubes))
            dx, dy, dz = rng.choice([
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1),
            ])
            nxt = (seed_cube[0] + dx, seed_cube[1] + dy, seed_cube[2] + dz)
            if nxt[2] >= 0 and nxt not in cubes:
                cubes.add(nxt)
        if cfg.get("allow_hidden") and max_hidden > 0:
            n_hid = rng.randint(0, max_hidden)
            for _ in range(n_hid):
                # Try to add a cube that's fully surrounded at +x, +y, or +z
                added = False
                cube_list = list(cubes)
                rng.shuffle(cube_list)
                for c in cube_list:
                    cand = (c[0] + 1, c[1] + 1, c[2])
                    if cand not in cubes and cand[2] >= 0:
                        cubes.add(cand)
                        added = True
                        break
                if not added:
                    break
        return list(cubes)

    def _count_visible_faces(self, cubes):
        cube_set = set(cubes)
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)]
        count = 0
        for (x, y, z) in cubes:
            for dx, dy, dz in dirs:
                if (x + dx, y + dy, z + dz) not in cube_set:
                    count += 1
        return count

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def _draw(self, cubes, sub_rng: random.Random, level: int) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        # Safe shaded face palette (top/left/right always distinguishable).
        top_color, left_color, right_color = _cube_face_palette(style, sub_rng)

        # Shift to positive
        xs = [c[0] for c in cubes]
        ys = [c[1] for c in cubes]
        zs = [c[2] for c in cubes]
        minx, miny, minz = min(xs), min(ys), min(zs)
        shifted = [(x - minx, y - miny, z - minz) for (x, y, z) in cubes]
        # Back-to-front painter's sort: farthest cubes first (low x+y+z),
        # so nearer cubes drawn last and correctly occlude farther ones.
        shifted.sort(key=lambda c: -(c[0] + c[1] - c[2]))

        edge_col = sub_rng.choice(["#222", "#0a0a0a", "#1a1a2e", "#3b0b0b", "#0b3d0b"])
        lw = 1.0 + sub_rng.random() * 1.0

        for (x, y, z) in shifted:
            pts = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=top_color,
                                 edgecolor=edge_col, lw=lw))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=left_color,
                                 edgecolor=edge_col, lw=lw))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=right_color,
                                 edgecolor=edge_col, lw=lw))

        all_pts = []
        for (x, y, z) in shifted:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        all_pts.append(_iso_project(x + dx, y + dy, z + dz))
        arr = np.array(all_pts)
        margin = 0.5
        ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
        ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)

        title = sub_rng.choice([
            "Cube Structure",
            "Polycube",
            "Unit-Cube Assembly",
            "3D Cube Shape",
            "Isometric View",
        ])
        ax.set_title(title, fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ====================================================================== #
# Sample generation
# ====================================================================== #

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = PolycubeCountingWarmupQA()
    for level in [0, 3, 6, 9]:
        gt_counts = {}
        for seed in range(10):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[polycube_counting_warmup] L{level} seed{seed} FAILED")
                continue
            img = env.render()
            out = f"/tmp/env_check/polycube_counting_warmup_s{seed}_L{level}.png"
            img.save(out)
            gt_counts[env._answer] = gt_counts.get(env._answer, 0) + 1
        print(f"L{level} GT distribution: {gt_counts}")
