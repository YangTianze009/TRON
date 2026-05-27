"""
Isometric Counting QA environment.

Draws isometric views of stacked unit cubes (like Minecraft blocks).
Questions about counting cubes, visible faces, layers, etc.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2-3 unit cubes in a SINGLE horizontal line (1D). Only `count_cubes`.
    Target pass rate: 70-90% (trivial count).
L1: 3-4 unit cubes in a 1D line or simple L. Only `count_cubes`.
    Target pass rate: 60-80%.
L2: 4-6 cubes, 2D flat layout (single layer, up to 3x2). `count_cubes`.
    Target pass rate: 50-70%.
L3: 5-7 cubes, up to 2 layers, simple stairs/tower. `count_cubes`, `layer_count`.
    Target pass rate: 40-60%.
L4: 6-9 cubes, 3x3 bounding. `count_cubes`, `layer_count`.
    Target pass rate: 35-55%.
L5: 7-11 cubes, light occlusion. Add `missing_cube` / `count_visible_faces`.
    Target pass rate: 25-45%.
L6: 8-13 cubes, more occlusion.
    Target pass rate: 20-40%.
L7: 9-15 cubes, moderate occlusion, all question types.
    Target pass rate: 15-35%.
L8: 10-18 cubes, heavy occlusion, all question types.
    Target pass rate: 10-25%.
L9: 12-22 cubes, heavy occlusion + `count_hidden_cubes`.
    Target pass rate: 5-20%.

======================================================================
Training interface
======================================================================

parameter = {"level": int in [0, 9]}

Diversity axes per seed (at same level):
  - voxel shape choice (stairs/L/tower/random/pyramid)
  - random rotation of the entire grid (0/90/180/270 around z)
  - random face-color permutation (top/left/right)
  - random edge color
  - 3+ paraphrased question stems per question type
  - 4+ title variants
  - randomized dpi / figsize_scale / background color
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    """Convert 3D grid coords to 2D isometric screen coords."""
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

_TITLE_VARIANTS = [
    "Isometric Cube Structure",
    "Isometric View",
    "Cube Stack",
    "3D Cube Structure",
    "Unit Cube Arrangement",
    "Blocks",
]

class IsometricCountingQA(StandaloneVisualEnv):
    ENV_NAME = "isometric_counting"

    QUESTION_TYPES = [
        "count_cubes",
        "count_visible_faces",
        "count_hidden_cubes",
        "layer_count",
        "missing_cube",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(
                cfg["qtypes"], weights=cfg["qtype_weights"]
            )[0]

        for attempt in range(30):
            result = self._try_generate(qtype, level, cfg)
            if result is not None:
                self._primary_complexity_feature = int(parameter.get("level", 0)) * 3 + len(result[1])
                return result
        return None

    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            return {
                "qtypes": ["count_cubes"],
                "qtype_weights": [1],
                "shape_types": ["line"],
                "min_cubes": 2, "max_cubes": 3,
                "max_layers": 1,
                "bbox": (3, 1, 1),
            }
        if level == 1:
            return {
                "qtypes": ["count_cubes"],
                "qtype_weights": [1],
                "shape_types": ["line", "l_flat"],
                "min_cubes": 3, "max_cubes": 4,
                "max_layers": 1,
                "bbox": (3, 2, 1),
            }
        if level == 2:
            return {
                "qtypes": ["count_cubes"],
                "qtype_weights": [1],
                "shape_types": ["flat_rect", "l_flat", "t_flat"],
                "min_cubes": 4, "max_cubes": 6,
                "max_layers": 1,
                "bbox": (3, 3, 1),
            }
        if level == 3:
            return {
                "qtypes": ["count_cubes", "layer_count"],
                "qtype_weights": [7, 3],
                "shape_types": ["stairs", "tower", "l_shape"],
                "min_cubes": 5, "max_cubes": 7,
                "max_layers": 2,
                "bbox": (3, 3, 2),
            }
        if level == 4:
            return {
                "qtypes": ["count_cubes", "layer_count"],
                "qtype_weights": [6, 4],
                "shape_types": ["stairs", "tower", "l_shape", "pyramid"],
                "min_cubes": 6, "max_cubes": 9,
                "max_layers": 3,
                "bbox": (3, 3, 3),
            }
        if level == 5:
            return {
                "qtypes": ["count_cubes", "layer_count", "missing_cube"],
                "qtype_weights": [6, 2, 2],
                "shape_types": ["stairs", "tower", "l_shape", "pyramid", "random"],
                "min_cubes": 7, "max_cubes": 11,
                "max_layers": 3,
                "bbox": (4, 3, 3),
            }
        if level == 6:
            return {
                "qtypes": ["count_cubes", "layer_count", "missing_cube",
                           "count_visible_faces"],
                "qtype_weights": [5, 2, 2, 1],
                "shape_types": ["stairs", "tower", "l_shape", "pyramid", "random"],
                "min_cubes": 8, "max_cubes": 13,
                "max_layers": 3,
                "bbox": (4, 4, 3),
            }
        if level == 7:
            return {
                "qtypes": ["count_cubes", "layer_count", "missing_cube",
                           "count_visible_faces"],
                "qtype_weights": [4, 2, 2, 2],
                "shape_types": ["stairs", "tower", "l_shape", "pyramid", "random"],
                "min_cubes": 9, "max_cubes": 15,
                "max_layers": 4,
                "bbox": (4, 4, 4),
            }
        if level == 8:
            return {
                "qtypes": ["count_cubes", "layer_count", "missing_cube",
                           "count_visible_faces", "count_hidden_cubes"],
                "qtype_weights": [3, 2, 2, 2, 1],
                "shape_types": ["stairs", "tower", "l_shape", "pyramid", "random"],
                "min_cubes": 10, "max_cubes": 18,
                "max_layers": 4,
                "bbox": (4, 4, 4),
            }
        return {  # level 9
            "qtypes": ["count_cubes", "missing_cube", "count_visible_faces",
                       "count_hidden_cubes"],
            "qtype_weights": [3, 2, 2, 2],
            "shape_types": ["random", "stairs", "tower", "pyramid"],
            "min_cubes": 12, "max_cubes": 22,
            "max_layers": 5,
            "bbox": (5, 5, 5),
        }

    # ------------------------------------------------------------------ #

    def _try_generate(self, qtype: str, level: int, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        # Use seed-derived sub_rng, plus attempt-level perturbation.
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )

        # L0: flat 2D rendering instead of isometric to avoid model confusion
        if level == 0:
            return self._generate_l0_flat(sub_rng, cfg)

        bx, by, bz = cfg["bbox"]

        # For count_hidden_cubes, we MUST construct a shape that actually has
        # at least one fully-enclosed interior cube — otherwise the answer is
        # trivially 0 (models can always guess 0). Force a solid-block-based
        # shape whose interior contains hidden cubes. We relax the cube-count
        # cap for this question type because the minimum feasible hidden
        # shape (3x3x3 = 27 cubes) exceeds the max for L8/L9.
        if qtype == "count_hidden_cubes":
            # Allow up to 48 cubes so we can pick 3x3x3, 3x3x4, 3x4x4, etc.
            # for a mix of hidden answers (1, 2, 3, 4).
            grid = self._make_hidden_shape(sub_rng, bx, by, bz,
                                           max(10, cfg["min_cubes"]),
                                           48)
            if grid is None:
                return None
            shape_type = "hidden"
        else:
            shape_type = sub_rng.choice(cfg["shape_types"])

            target_cubes = sub_rng.randint(cfg["min_cubes"], cfg["max_cubes"])
            grid = self._make_shape(sub_rng, bx, by, bz, shape_type,
                                    target_cubes, cfg["max_layers"])
            if grid is None:
                return None

        # If too many cubes, randomly prune some to land near target.
        # (Skip pruning for the hidden-shape generator — it already enforces
        # min/max cube counts internally, and pruning would destroy the
        # interior hidden cube.)
        if shape_type == "hidden":
            total_cubes = int(np.sum(grid))
            # Relaxed upper bound for hidden-shape samples: cube counts can
            # go up to 48 (3x4x4 solid) to give variety in hidden answers.
            if total_cubes < 10 or total_cubes > 50:
                return None
            layers = self._count_layers(grid)
            hidden = self._count_hidden_cubes(grid)
            if hidden < 1:
                return None
            stems = [
                "How many unit cubes are completely hidden (not visible from any direction)?",
                "Count the number of cubes that are fully enclosed — not visible from above, below, front, back, left, or right.",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            img = self._draw_isometric(grid, sub_rng)
            return q, str(hidden), img

        total_cubes = int(np.sum(grid))
        if total_cubes > cfg["max_cubes"]:
            positions = list(zip(*np.nonzero(grid)))
            sub_rng.shuffle(positions)
            for pos in positions:
                if total_cubes <= target_cubes:
                    break
                # Don't prune if it disconnects the bottom - keep it simple,
                # just prune from top or edges.
                i, j, k = pos
                if k == grid.shape[2] - 1 or np.sum(grid) == 1:
                    grid[i, j, k] = 0
                    total_cubes -= 1
            # Re-prune more aggressively if still over
            positions = list(zip(*np.nonzero(grid)))
            sub_rng.shuffle(positions)
            for pos in positions:
                if total_cubes <= cfg["max_cubes"]:
                    break
                i, j, k = pos
                grid[i, j, k] = 0
                total_cubes -= 1

        total_cubes = int(np.sum(grid))
        if total_cubes < cfg["min_cubes"] or total_cubes > cfg["max_cubes"] + 2:
            return None

        # Random grid rotation around z-axis for diversity (L >= 2 only,
        # so L0/L1 line shape still looks like an obvious line)
        if level >= 2:
            n_rot = sub_rng.choice([0, 1, 2, 3])
            for _ in range(n_rot):
                grid = np.rot90(grid, axes=(0, 1))

        layers = self._count_layers(grid)

        if qtype == "count_cubes":
            stems = [
                "How many unit cubes are in this structure?",
                "Count the total number of unit cubes shown in the image.",
                "How many cubes make up the structure shown above?",
                "Determine the total number of unit cubes in the arrangement.",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            answer = str(total_cubes)

        elif qtype == "layer_count":
            stems = [
                "How many horizontal layers (levels) does this structure have?",
                "Count the number of distinct horizontal levels in the arrangement.",
                "How many stacked layers are visible in the cube structure?",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            answer = str(layers)

        elif qtype == "missing_cube":
            bbox = self._bounding_box(grid)
            bxx, byy, bzz = bbox
            full_bbox = bxx * byy * bzz
            missing = full_bbox - total_cubes
            if missing < 0:
                return None
            stems = [
                (f"The bounding box of this structure is {bxx} x {byy} x {bzz}. "
                 f"How many cubes are missing to fill the bounding box completely?"),
                (f"If this structure were completed into a solid {bxx} x {byy} x {bzz} box, "
                 f"how many additional unit cubes would be needed?"),
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            answer = str(missing)

        elif qtype == "count_visible_faces":
            vis = self._count_visible_faces(grid)
            stems = [
                "How many unit-square faces are visible from the outside of this structure? (Count all exposed faces.)",
                "Count all exposed unit-square faces of the structure — faces shared between two cubes are not counted.",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            answer = str(vis)

        elif qtype == "count_hidden_cubes":
            hidden = self._count_hidden_cubes(grid)
            stems = [
                "How many unit cubes are completely hidden (not visible from any direction)?",
                "Count the number of cubes that are fully enclosed — not visible from above, below, front, back, left, or right.",
            ]
            q = sub_rng.choice(stems) + " Answer with a single integer."
            answer = str(hidden)
        else:
            return None

        img = self._draw_isometric(grid, sub_rng)
        return q, answer, img

    def _generate_l0_flat(self, sub_rng, cfg):
        """L0: 2D flat grid — no isometric. 2-3 cubes in a single row."""
        n = sub_rng.randint(cfg["min_cubes"], cfg["max_cubes"])
        cells = [(0, i) for i in range(n)]

        stems = [
            "How many colored squares are shown in the image? Answer with a single integer.",
            "Count the total number of unit squares. Answer with a single integer.",
            "How many cubes are in this flat arrangement? Answer with a single integer.",
        ]
        q = sub_rng.choice(stems)
        answer = str(n)

        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        sub_rng.shuffle(palette)
        fig, ax = plt.subplots(figsize=(max(4, n + 1) * sc, 3 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        from matplotlib.patches import Rectangle as _Rect
        for r, c in cells:
            ax.add_patch(_Rect((c, 0), 1, 1,
                               facecolor=palette[0], edgecolor="#222",
                               linewidth=2))
        ax.set_xlim(-0.3, n + 0.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_title(sub_rng.choice(["Count the Cubes", "How Many Squares?",
                                      "Unit Cube Count"]),
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return q, answer, img

    def _make_shape(self, rng, gx, gy, gz, shape_type, target=None, max_layers=None):
        """Generate a voxel grid of the requested shape type.
        Returns numpy array or None if generation failed."""
        grid = np.zeros((gx, gy, gz), dtype=int)

        if shape_type == "line":
            # horizontal line of `target` cubes along x axis at (0,0)
            n = target if target else rng.randint(2, 3)
            n = min(n, gx)
            for i in range(n):
                grid[i, 0, 0] = 1

        elif shape_type == "l_flat":
            # flat L on the ground
            arm_a = rng.randint(2, max(2, gx - 1))
            arm_b = rng.randint(1, max(1, gy - 1))
            for i in range(arm_a):
                if i < gx:
                    grid[i, 0, 0] = 1
            for j in range(arm_b):
                if j < gy:
                    grid[0, j, 0] = 1

        elif shape_type == "t_flat":
            # flat T
            bar = min(gx, 3)
            for i in range(bar):
                grid[i, 0, 0] = 1
            # stem
            if gy >= 2:
                grid[bar // 2, 1, 0] = 1

        elif shape_type == "flat_rect":
            # a flat rectangle area, possibly with a notch
            rx = rng.randint(2, max(2, gx))
            ry = rng.randint(2, max(2, gy))
            for i in range(rx):
                for j in range(ry):
                    grid[i, j, 0] = 1
            # Occasionally remove a corner cube for diversity
            if rng.random() < 0.3 and rx * ry > 4:
                grid[rx - 1, ry - 1, 0] = 0

        elif shape_type == "stairs":
            steps = min(gx, gz, 4)
            for i in range(steps):
                for j in range(gy):
                    for k in range(i + 1):
                        grid[i, j, k] = 1

        elif shape_type == "l_shape":
            for i in range(gx):
                for j in range(gy):
                    grid[i, j, 0] = 1
            col_x, col_y = 0, 0
            for k in range(min(gz, max_layers or gz)):
                grid[col_x, col_y, k] = 1

        elif shape_type == "tower":
            cx, cy = gx // 2, gy // 2
            tower_h = min(gz, max_layers or gz)
            for k in range(tower_h):
                grid[cx, cy, k] = 1
            for i in range(gx):
                for j in range(gy):
                    if rng.random() < 0.55:
                        grid[i, j, 0] = 1

        elif shape_type == "pyramid":
            layers = min(gz, max_layers or gz)
            for k in range(layers):
                for i in range(k, max(k + 1, gx - k)):
                    for j in range(k, max(k + 1, gy - k)):
                        if 0 <= i < gx and 0 <= j < gy:
                            grid[i, j, k] = 1

        else:  # random
            for i in range(gx):
                for j in range(gy):
                    max_h = rng.randint(0, min(gz, max_layers or gz))
                    for k in range(max_h):
                        grid[i, j, k] = 1

        if int(np.sum(grid)) == 0:
            return None
        return grid

    def _make_hidden_shape(self, rng, gx, gy, gz,
                           min_cubes=10, max_cubes=22):
        """Build a shape that deliberately contains at least one hidden cube.
        Strategy: pick a solid sub-block of size sx×sy×sz where all of
        sx,sy,sz >= 3 (so there's an interior cell), then optionally add a
        small tail of random surface cubes. The interior count is
        (sx-2)*(sy-2)*(sz-2)."""
        # Candidate solid sub-block shapes that fit in bbox and yield at least
        # one hidden cube while staying under max_cubes.
        candidates = []
        for sx in range(3, gx + 1):
            for sy in range(3, gy + 1):
                for sz in range(3, gz + 1):
                    n = sx * sy * sz
                    if min_cubes <= n <= max_cubes + 6:
                        candidates.append((sx, sy, sz))
        if not candidates:
            # Fallback: 3x3x3 solid (27 cubes, 1 hidden) even if slightly
            # over max — better than returning None.
            candidates = [(3, 3, 3)]
        sx, sy, sz = rng.choice(candidates)

        grid = np.zeros((gx, gy, gz), dtype=int)
        # Place the solid block at origin corner (bottom-front-left).
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    grid[i, j, k] = 1

        # Optionally remove CORNER cubes only (cubes on multiple faces). Do NOT
        # remove face-center surface cubes because that would expose the
        # interior hidden cube along that axis (e.g. removing (1,0,1) on a
        # 3x3x3 block exposes the (1,1,1) hidden cube from the -y direction).
        def _is_interior(i, j, k):
            return (0 < i < sx - 1 and 0 < j < sy - 1 and 0 < k < sz - 1)

        def _is_face_adjacent_to_interior(i, j, k):
            # A surface cell is "face adjacent" to an interior cell if
            # exactly one coordinate is at the boundary (0 or sx-1 etc.).
            bx_count = (i == 0 or i == sx - 1) + (j == 0 or j == sy - 1) + (k == 0 or k == sz - 1)
            return bx_count == 1

        n_remove = rng.randint(0, 3)
        for _ in range(n_remove):
            for _try in range(20):
                i = rng.randint(0, sx - 1)
                j = rng.randint(0, sy - 1)
                k = rng.randint(0, sz - 1)
                if (not _is_interior(i, j, k)
                        and not _is_face_adjacent_to_interior(i, j, k)
                        and grid[i, j, k] == 1):
                    grid[i, j, k] = 0
                    break
        return grid

    def _count_layers(self, grid):
        count = 0
        for k in range(grid.shape[2]):
            if np.any(grid[:, :, k]):
                count += 1
        return count

    def _count_visible_faces(self, grid):
        gx, gy, gz = grid.shape
        count = 0
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)]
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    if grid[i, j, k] == 0:
                        continue
                    for di, dj, dk in dirs:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if (ni < 0 or ni >= gx or nj < 0 or nj >= gy or
                                nk < 0 or nk >= gz or grid[ni, nj, nk] == 0):
                            count += 1
        return count

    def _count_hidden_cubes(self, grid):
        gx, gy, gz = grid.shape
        visible = set()
        for j in range(gy):
            for k in range(gz):
                for i in range(gx - 1, -1, -1):
                    if grid[i, j, k]:
                        visible.add((i, j, k)); break
        for j in range(gy):
            for k in range(gz):
                for i in range(gx):
                    if grid[i, j, k]:
                        visible.add((i, j, k)); break
        for i in range(gx):
            for k in range(gz):
                for j in range(gy - 1, -1, -1):
                    if grid[i, j, k]:
                        visible.add((i, j, k)); break
        for i in range(gx):
            for k in range(gz):
                for j in range(gy):
                    if grid[i, j, k]:
                        visible.add((i, j, k)); break
        for i in range(gx):
            for j in range(gy):
                for k in range(gz - 1, -1, -1):
                    if grid[i, j, k]:
                        visible.add((i, j, k)); break
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    if grid[i, j, k]:
                        visible.add((i, j, k)); break
        total = int(np.sum(grid))
        return total - len(visible)

    def _bounding_box(self, grid):
        positions = np.argwhere(grid > 0)
        if len(positions) == 0:
            return (0, 0, 0)
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        return tuple((maxs - mins + 1).tolist())

    def _draw_isometric(self, grid, sub_rng=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect('equal')
        ax.axis('off')

        gx, gy, gz = grid.shape
        cubes = []
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    if grid[i, j, k]:
                        cubes.append((i, j, k))
        cubes.sort(key=lambda c: -(c[0] + c[1] - c[2]))

        palette = list(style["palette"])
        if sub_rng is not None:
            sub_rng.shuffle(palette)
        top_color = palette[0]
        left_color = palette[1 % len(palette)]
        right_color = palette[2 % len(palette)]

        edge_pool = ["#111", "#1a1a2e", "#0b3d0b", "#2d1b3e", "#3b0b0b"]
        edge_col = (sub_rng or random).choice(edge_pool)
        lw = 1.0 + (sub_rng or random).random() * 1.0

        for (x, y, z) in cubes:
            p0 = _iso_project(x, y, z + 1)
            p1 = _iso_project(x + 1, y, z + 1)
            p2 = _iso_project(x + 1, y + 1, z + 1)
            p3 = _iso_project(x, y + 1, z + 1)
            ax.add_patch(Polygon([p0, p1, p2, p3], closed=True,
                                 facecolor=top_color, edgecolor=edge_col, lw=lw))
            p0 = _iso_project(x, y, z)
            p1 = _iso_project(x, y + 1, z)
            p2 = _iso_project(x, y + 1, z + 1)
            p3 = _iso_project(x, y, z + 1)
            ax.add_patch(Polygon([p0, p1, p2, p3], closed=True,
                                 facecolor=left_color, edgecolor=edge_col, lw=lw))
            p0 = _iso_project(x, y, z)
            p1 = _iso_project(x + 1, y, z)
            p2 = _iso_project(x + 1, y, z + 1)
            p3 = _iso_project(x, y, z + 1)
            ax.add_patch(Polygon([p0, p1, p2, p3], closed=True,
                                 facecolor=right_color, edgecolor=edge_col, lw=lw))

        all_pts = []
        for x in range(gx + 1):
            for y in range(gy + 1):
                for z in range(gz + 1):
                    all_pts.append(_iso_project(x, y, z))
        all_pts = np.array(all_pts)
        margin = 0.5
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

        title = (sub_rng or random).choice(_TITLE_VARIANTS)
        ax.set_title(title, fontsize=style["font_size_base"] + 3,
                     fontweight='bold', pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = IsometricCountingQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
