"""
Slice And Count QA.

A 2D cross-section / planar view of a solid with 1-4 cutting lines. Asks how
many separate pieces result from the cuts.

L0 uses simple straight cuts through a plain rectangle/square (always well
formed). Higher levels introduce composite solids, rings (annular regions),
diagonal and oblique cuts, and multiple intersecting cuts.

Primitives pool (cross-seed diversity):
  - square / rectangle
  - L-shape / T-shape / plus-shape / U-shape
  - triangle / trapezoid
  - disk / annulus (ring)
  - two-disjoint squares
Cut families:
  - horizontal cut(s)
  - vertical cut(s)
  - diagonal / oblique cut(s)
  - radial cuts (for disk/annulus)
  - combination cut (horizontal + vertical etc.)

Piece counts are computed by shapely union/difference when available, else
by a small mesh rasterization flood-fill on a numpy array.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# --------------------------------------------------------------------- #
# Geometry utilities (flood-fill based piece counter)
# --------------------------------------------------------------------- #

def _point_in_poly(x, y, verts):
    """Even-odd rule."""
    inside = False
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def _point_in_region(x, y, region):
    """Region is a dict with kind + params. True if (x,y) is inside the solid."""
    kind = region["kind"]
    if kind == "poly":
        return _point_in_poly(x, y, region["verts"])
    if kind == "union":
        return any(_point_in_region(x, y, r) for r in region["parts"])
    if kind == "disk":
        cx, cy = region["center"]
        r = region["r"]
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if kind == "annulus":
        cx, cy = region["center"]
        r_out = region["r_out"]
        r_in = region["r_in"]
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        return (d2 <= r_out * r_out) and (d2 >= r_in * r_in)
    return False

def _line_sep_sign(x, y, cut):
    """Return +1 / -1 / 0 based on which half-plane (x,y) is on for a linear cut."""
    kind = cut["kind"]
    if kind in ("hline", "vline", "diag"):
        a, b, c = cut["a"], cut["b"], cut["c"]
        v = a * x + b * y - c
        if abs(v) < 1e-9:
            return 0
        return 1 if v > 0 else -1
    if kind == "diameter":
        # A diameter cut is a full line through `center` at angle theta.
        cx, cy = cut["center"]
        theta = cut["theta"]
        # Normal vector to line = (-sin theta, cos theta).
        nx = -math.sin(theta)
        ny = math.cos(theta)
        v = nx * (x - cx) + ny * (y - cy)
        if abs(v) < 1e-9:
            return 0
        return 1 if v > 0 else -1
    return 0

def _count_pieces_by_flood(region, cuts, bbox, resolution=180):
    """Rasterize `region` onto a grid; remove cut boundaries as walls; flood-fill
    count connected components. Robust and simple."""
    x_min, y_min, x_max, y_max = bbox
    W = resolution
    H = int(resolution * (y_max - y_min) / max(1e-6, (x_max - x_min)))
    H = max(60, min(H, 260))
    grid = np.zeros((H, W), dtype=np.int8)
    # Fill region
    for iy in range(H):
        # sample pixel centers
        y = y_min + (iy + 0.5) * (y_max - y_min) / H
        for ix in range(W):
            x = x_min + (ix + 0.5) * (x_max - x_min) / W
            if _point_in_region(x, y, region):
                grid[iy, ix] = 1

    if not cuts:
        # Count components directly.
        return _flood_count(grid)

    # For each cut, determine the sign per pixel; at boundary transitions
    # between neighbors with different signs, block connectivity.
    signs = np.zeros((len(cuts), H, W), dtype=np.int8)
    for ci, cut in enumerate(cuts):
        for iy in range(H):
            y = y_min + (iy + 0.5) * (y_max - y_min) / H
            for ix in range(W):
                x = x_min + (ix + 0.5) * (x_max - x_min) / W
                signs[ci, iy, ix] = _line_sep_sign(x, y, cut)

    # Flood-fill but disallow crossing between pixels that disagree on any cut.
    visited = np.zeros_like(grid, dtype=np.int8)
    components = 0
    for iy in range(H):
        for ix in range(W):
            if grid[iy, ix] != 1 or visited[iy, ix]:
                continue
            components += 1
            stack = [(iy, ix)]
            visited[iy, ix] = 1
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 1 and not visited[ny, nx]:
                        # Check each cut: both must lie on same side (same sign).
                        same = True
                        for ci in range(len(cuts)):
                            a = signs[ci, cy, cx]
                            b = signs[ci, ny, nx]
                            if a != 0 and b != 0 and a != b:
                                same = False
                                break
                        if same:
                            visited[ny, nx] = 1
                            stack.append((ny, nx))
    return components

def _flood_count(grid):
    H, W = grid.shape
    visited = np.zeros_like(grid)
    components = 0
    for iy in range(H):
        for ix in range(W):
            if grid[iy, ix] != 1 or visited[iy, ix]:
                continue
            components += 1
            stack = [(iy, ix)]
            visited[iy, ix] = 1
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 1 and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
    return components

# --------------------------------------------------------------------- #
# Solid builders
# --------------------------------------------------------------------- #

def _solid_square():
    verts = [(0, 0), (4, 0), (4, 4), (0, 4)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 4.3, 4.3), "square"

def _solid_rect(rng):
    w = rng.choice([3, 4, 5, 6])
    h = rng.choice([2, 3, 4])
    verts = [(0, 0), (w, 0), (w, h), (0, h)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, w + 0.3, h + 0.3), "rectangle"

def _solid_L(rng):
    verts = [(0, 0), (3, 0), (3, 2), (1, 2), (1, 4), (0, 4)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 3.3, 4.3), "L-shape"

def _solid_T(rng):
    verts = [(0, 2), (1, 2), (1, 0), (3, 0), (3, 2), (4, 2), (4, 4), (0, 4)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 4.3, 4.3), "T-shape"

def _solid_plus(rng):
    verts = [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3), (1, 2), (0, 2), (0, 1), (1, 1)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 3.3, 3.3), "plus-shape"

def _solid_U(rng):
    verts = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 4.3, 4.3), "U-shape"

def _solid_triangle(rng):
    verts = [(0, 0), (5, 0), (2.5, 4)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 5.3, 4.3), "triangle"

def _solid_trapezoid(rng):
    verts = [(0, 0), (5, 0), (4, 3), (1, 3)]
    return {"kind": "poly", "verts": verts}, (-0.3, -0.3, 5.3, 3.3), "trapezoid"

def _solid_disk(rng):
    return ({"kind": "disk", "center": (0, 0), "r": 2.0},
            (-2.3, -2.3, 2.3, 2.3), "disk")

def _solid_annulus(rng):
    return ({"kind": "annulus", "center": (0, 0), "r_out": 2.0, "r_in": 0.9},
            (-2.3, -2.3, 2.3, 2.3), "ring (annulus)")

def _solid_two_squares(rng):
    r1 = {"kind": "poly", "verts": [(0, 0), (2, 0), (2, 2), (0, 2)]}
    r2 = {"kind": "poly", "verts": [(3, 0), (5, 0), (5, 2), (3, 2)]}
    return ({"kind": "union", "parts": [r1, r2]},
            (-0.3, -0.3, 5.3, 2.3), "two separate squares")

_SOLID_BUILDERS = {
    "square":       _solid_square,
    "rectangle":    _solid_rect,
    "L":            _solid_L,
    "T":            _solid_T,
    "plus":         _solid_plus,
    "U":            _solid_U,
    "triangle":     _solid_triangle,
    "trapezoid":    _solid_trapezoid,
    "disk":         _solid_disk,
    "annulus":      _solid_annulus,
    "two_squares":  _solid_two_squares,
}

# --------------------------------------------------------------------- #
# Cut builders (return list of cut dicts + description text)
# --------------------------------------------------------------------- #

def _make_cuts(rng, solid_name, bbox, n_cuts):
    """Returns (cuts, description)."""
    x_min, y_min, x_max, y_max = bbox
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cuts = []
    descs = []

    # For disks/annuli, prefer diameter cuts.
    radial_mode = solid_name in ("disk", "ring (annulus)")

    for _ in range(n_cuts):
        if radial_mode:
            # Random angle. Ensure different angles between cuts by >10 deg.
            for _try in range(30):
                theta = rng.uniform(0, math.pi)
                if all(abs(((theta - c["theta"]) % math.pi) - 0) > math.radians(15)
                       and abs(((theta - c["theta"]) % math.pi) - math.pi) > math.radians(15)
                       for c in cuts if c["kind"] == "diameter"):
                    break
            cuts.append({"kind": "diameter", "center": (cx, cy), "theta": theta})
            descs.append(f"diameter cut at {math.degrees(theta):.0f}°")
        else:
            choice = rng.choice(["h", "v", "diag", "h", "v"])
            if choice == "h":
                y = rng.uniform(y_min + 0.4 * (y_max - y_min),
                                y_min + 0.7 * (y_max - y_min))
                cuts.append({"kind": "hline", "a": 0, "b": 1, "c": y})
                descs.append(f"horizontal cut")
            elif choice == "v":
                x = rng.uniform(x_min + 0.35 * (x_max - x_min),
                                x_min + 0.75 * (x_max - x_min))
                cuts.append({"kind": "vline", "a": 1, "b": 0, "c": x})
                descs.append(f"vertical cut")
            else:  # diag
                # Line through interior with slope != 0 and != inf.
                m = rng.choice([0.7, -0.7, 1.3, -1.3, 0.5, -0.5])
                # line: y = m*(x - cx) + cy  -> m*x - y = m*cx - cy
                c = m * cx - cy
                cuts.append({"kind": "diag", "a": m, "b": -1, "c": c})
                descs.append(f"diagonal cut (slope {m:+.1f})")

    # Dedup identical description labels.
    if len(descs) > 1 and all(d == descs[0] for d in descs):
        desc_text = f"{len(descs)} parallel " + descs[0].replace("cut", "cuts")
    else:
        desc_text = " + ".join(descs)
    return cuts, desc_text

# --------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------- #

def _draw_region(ax, region, face, edge, lw):
    kind = region["kind"]
    if kind == "poly":
        ax.add_patch(MplPolygon(region["verts"], closed=True,
                                facecolor=face, edgecolor=edge,
                                linewidth=lw, alpha=0.55))
    elif kind == "union":
        for p in region["parts"]:
            _draw_region(ax, p, face, edge, lw)
    elif kind == "disk":
        cx, cy = region["center"]
        ax.add_patch(mpatches.Circle((cx, cy), region["r"], facecolor=face,
                                     edgecolor=edge, linewidth=lw, alpha=0.55))
    elif kind == "annulus":
        cx, cy = region["center"]
        outer = mpatches.Circle((cx, cy), region["r_out"], facecolor=face,
                                edgecolor=edge, linewidth=lw, alpha=0.55)
        inner = mpatches.Circle((cx, cy), region["r_in"], facecolor="white",
                                edgecolor=edge, linewidth=lw, alpha=1.0)
        ax.add_patch(outer)
        ax.add_patch(inner)

def _draw_cuts(ax, cuts, bbox, cut_colors):
    x_min, y_min, x_max, y_max = bbox
    pad = 0.35
    xs = np.linspace(x_min - pad, x_max + pad, 50)
    for i, cut in enumerate(cuts):
        col = cut_colors[i % len(cut_colors)]
        kind = cut["kind"]
        if kind == "hline":
            y = cut["c"] / cut["b"]
            ax.plot([x_min - pad, x_max + pad], [y, y], "--",
                    color=col, linewidth=2.4, alpha=0.9)
        elif kind == "vline":
            x = cut["c"] / cut["a"]
            ax.plot([x, x], [y_min - pad, y_max + pad], "--",
                    color=col, linewidth=2.4, alpha=0.9)
        elif kind == "diag":
            a, b, c = cut["a"], cut["b"], cut["c"]
            # y = (a*x - c) / (-b) == a*x - c  (b = -1)
            ys = [a * x - c for x in xs]
            ax.plot(xs, ys, "--", color=col, linewidth=2.4, alpha=0.9)
        elif kind == "diameter":
            cx, cy = cut["center"]
            # full chord across bbox
            dx = math.cos(cut["theta"])
            dy = math.sin(cut["theta"])
            length = max(x_max - x_min, y_max - y_min) * 1.5
            ax.plot([cx - dx * length, cx + dx * length],
                    [cy - dy * length, cy + dy * length], "--",
                    color=col, linewidth=2.4, alpha=0.9)

# --------------------------------------------------------------------- #
# Main environment
# --------------------------------------------------------------------- #

class SliceAndCountQA(StandaloneVisualEnv):
    ENV_NAME = "slice_and_count"

    def _level_config(self, level):
        level = max(0, min(9, int(level)))
        # Max cuts
        if level <= 1:
            max_cuts = 1
        elif level <= 4:
            max_cuts = 2
        elif level <= 7:
            max_cuts = 3
        else:
            max_cuts = 4
        # Solid pool
        if level <= 1:
            solid_pool = ["square", "rectangle"]
        elif level <= 3:
            solid_pool = ["square", "rectangle", "L", "T", "triangle", "trapezoid"]
        elif level <= 6:
            solid_pool = ["L", "T", "U", "plus", "triangle", "trapezoid",
                          "disk", "annulus"]
        else:
            solid_pool = ["L", "T", "U", "plus", "annulus", "two_squares",
                          "trapezoid", "disk"]
        return {
            "max_cuts": max_cuts,
            "min_cuts": 1 if level <= 3 else 2 if level <= 6 else 2,
            "solid_pool": solid_pool,
        }

    # -------- templates --------
    _Q_TEMPLATES = [
        "How many separate pieces result from all the cuts shown? Answer with a single integer.",
        "Dashed lines show the cuts made through the figure. Count the pieces produced. Answer with an integer.",
        "After applying every dashed cut, how many pieces are there in total? Reply with one integer.",
        "The figure is divided by the dashed lines. Output the number of resulting pieces as a single integer.",
    ]

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1022)
        style = self._random_style()

        for _ in range(20):
            result = self._try_make(rng, cfg, level, style)
            if result is not None:
                return result
        return None

    def _try_make(self, rng, cfg, level, style):
        solid_name = rng.choice(cfg["solid_pool"])
        builder = _SOLID_BUILDERS[solid_name]
        if solid_name in ("square",):
            region, bbox, desc_solid = builder()
        else:
            region, bbox, desc_solid = builder(rng)

        n_cuts = rng.randint(cfg["min_cuts"], cfg["max_cuts"])
        cuts, cut_desc = _make_cuts(rng, desc_solid, bbox, n_cuts)

        # Compute piece count via flood fill
        pieces = _count_pieces_by_flood(region, cuts, bbox, resolution=150)
        if pieces < 1 or pieces > 20:
            return None
        # Sanity: L0 simple square/rect with 1 cut must give 2 pieces unless cut misses.
        if level <= 1 and pieces < 2:
            return None

        # Render
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(6.2 * sc, 6.2 * sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal')
        ax.axis('off')

        face = rng.choice(style['palette'])
        edge = style['geo_line_color']
        lw = max(1.8, style['line_width'])
        _draw_region(ax, region, face, edge, lw)

        cut_colors = ['#e63946', '#1d3557', '#2a9d8f', '#e76f51']
        rng.shuffle(cut_colors)
        _draw_cuts(ax, cuts, bbox, cut_colors)

        x_min, y_min, x_max, y_max = bbox
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        title_templates = [
            f"{desc_solid.title()} with {n_cuts} cut(s)",
            f"Cut the {desc_solid}",
            f"Slice Puzzle: {desc_solid}",
            f"Count the pieces",
        ]
        ax.set_title(rng.choice(title_templates),
                     fontsize=style['font_size_base'] + 1,
                     fontweight='bold')

        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q = rng.choice(self._Q_TEMPLATES)
        return q, str(pieces), img

if __name__ == "__main__":
    env = SliceAndCountQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, answer={env._answer if ok else 'X'}")
