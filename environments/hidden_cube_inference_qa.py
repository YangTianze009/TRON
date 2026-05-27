"""
Hidden Cube Inference QA.

Shows an isometric view of a 3D block structure. Asks how many cubes are
completely hidden (not visible from any direction).

Difficulty axes:
  A) grid_size: 2x2x2 -> 4x3x3
  B) show_transparency + fill_ratio
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _cube_face_palette(style, rng):
    """BUGFIX: produce a SAFE 3-color face palette for isometric cubes.

    Returns [top, left, right] where top is the lightest and left the
    darkest (standard isometric lighting cue). Previously the code used
    three unrelated colors from the random palette: when the palette was
    "high contrast" the top/side could be solid BLACK, and when it was
    "pastel" all three faces blended together. Shading a single base
    color guarantees the three faces are always distinguishable.
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
    # Normalize to a mid-range base lightness so the derived top/left/right
    # shades always differ by clearly visible amounts. Boost saturation a
    # touch so near-grey bases still produce a readable hue difference.
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

def _count_hidden(grid):
    gx, gy, gz = grid.shape
    visible = set()
    # From each of 6 directions, first cube hit is visible
    for j in range(gy):
        for k in range(gz):
            for i in range(gx): # +x direction
                if grid[i,j,k]: visible.add((i,j,k)); break
            for i in range(gx-1,-1,-1):
                if grid[i,j,k]: visible.add((i,j,k)); break
    for i in range(gx):
        for k in range(gz):
            for j in range(gy):
                if grid[i,j,k]: visible.add((i,j,k)); break
            for j in range(gy-1,-1,-1):
                if grid[i,j,k]: visible.add((i,j,k)); break
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                if grid[i,j,k]: visible.add((i,j,k)); break
            for k in range(gz-1,-1,-1):
                if grid[i,j,k]: visible.add((i,j,k)); break
    total = int(np.sum(grid))
    return total - len(visible)

class HiddenCubeInferenceQA(StandaloneVisualEnv):
    ENV_NAME = "hidden_cube_inference"

    def _level_config(self, level):
        # Grid must have interior cubes (>=3x3x3) so "hidden" count can vary.
        sizes = [(3,3,3),(3,3,3),(3,3,3),(4,3,3),(4,3,3),(4,4,3),(4,4,3),(4,4,4),(5,4,4),(5,4,4)]
        s = sizes[min(level, 9)]
        return {
            'grid': s,
            # Fill factor < 1 so interior cube presence varies across seeds.
            'fill': max(0.65, 0.92 - level * 0.03),
            'show_hint': level <= 2,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1009)
        style = self._random_style()

        gx, gy, gz = cfg['grid']
        grid = np.zeros((gx, gy, gz), dtype=int)

        # Fill grid
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    if rng.random() < cfg['fill']:
                        grid[i,j,k] = 1

        # Ensure there's a ground layer
        for i in range(gx):
            for j in range(gy):
                grid[i,j,0] = 1

        hidden = _count_hidden(grid)
        total = int(np.sum(grid))
        answer = str(hidden)

        # Draw isometric with a safe shaded face palette (top > right > left).
        palette = _cube_face_palette(style, rng)
        sc = style['figsize_scale']
        # 2026-04-25: at L0-L2, show layer-by-layer top-views so model can
        # directly count interior (hidden) cubes per layer.
        show_layers = level <= 2
        if show_layers:
            n_panels = 1 + gz
            fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels * sc, 6 * sc))
            ax = axes[0]
        else:
            fig, ax = plt.subplots(figsize=(7*sc, 7*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal'); ax.axis('off')

        cubes = [(i,j,k) for i in range(gx) for j in range(gy) for k in range(gz) if grid[i,j,k]]
        for (x, y, z) in sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2])):
            for face, fc in [
                ([(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)], palette[0]),
                ([(x,y,z),(x,y+1,z),(x,y+1,z+1),(x,y,z+1)], palette[1]),
                ([(x,y,z),(x+1,y,z),(x+1,y,z+1),(x,y,z+1)], palette[2]),
            ]:
                pts = [_iso_project(*p) for p in face]
                ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor='#333', lw=1.0))

        if cfg['show_hint']:
            ax.text(0.02, 0.02, f"Total cubes: {total}", transform=ax.transAxes,
                    fontsize=style['font_size_base'], alpha=0.7, style='italic')

        ax.autoscale_view(); ax.margins(0.15)
        ax.set_title("Hidden Cube Count", fontsize=style['font_size_base']+3, fontweight='bold')

        if show_layers:
            for k in range(gz):
                axk = axes[1 + k]
                axk.set_facecolor(style['bg_color'])
                axk.set_aspect('equal')
                axk.set_xlim(-0.5, gx + 0.5)
                axk.set_ylim(-0.5, gy + 0.5)
                for i in range(gx):
                    for j in range(gy):
                        col = palette[0] if grid[i, j, k] else "#f0f0f0"
                        axk.add_patch(plt.Rectangle((i, j), 1, 1, fc=col,
                                                      ec="#333", lw=1.0))
                axk.set_xticks([]); axk.set_yticks([])
                axk.set_title(f"Layer z={k}",
                              fontsize=style['font_size_base'] + 2,
                              fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        hint = f" The structure contains {total} total cubes." if cfg['show_hint'] else ""
        q = (f"This isometric view shows a structure of unit cubes.{hint} "
             f"How many cubes are completely hidden (not visible from any direction)? "
             f"Answer with a single integer.")
        # 2026-04-25 add reasoning hint at low levels
        if level <= 2:
            q += (
                f" Hint: the structure is a {gx}×{gy}×{gz} bounding box. "
                "A cube is hidden if it has no exposed face — i.e. it is "
                "interior (not on top, not on the front, back, left, or "
                "right surface). For each cube, check whether all 6 "
                "neighboring positions (±x, ±y, +z) contain another cube; "
                "if yes, it is hidden."
            )
        return q, answer, img

if __name__ == "__main__":
    env = HiddenCubeInferenceQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
