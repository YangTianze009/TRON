"""
Polycube Rotation Axis Identify QA.

Shows two isometric polycube renders (before/after rotation). Asks which axis
the rotation was performed around. MCQ with 4 options.

Difficulty axes:
  A) n_cubes: 4..8
  B) rotation angle variety + axis arrow visibility
"""
import math, random
from typing import Dict, List, Optional, Tuple
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

def _rot_axis(cubes, axis, k):
    out = []
    for (x, y, z) in cubes:
        for _ in range(k % 4):
            if axis == 'x': y, z = -z, y
            elif axis == 'y': x, z = z, -x
            else: x, y = -y, x
        out.append((x, y, z))
    return out

def _normalize(cubes):
    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    mx, my, mz = min(xs), min(ys), min(zs)
    return [(x - mx, y - my, z - mz) for (x, y, z) in cubes]

def _draw_polycube(ax, cubes, palette, edge_col='#222', lw=1.2, x_off=0):
    top_c, left_c, right_c = palette[0], palette[1], palette[2]
    # Painter's algorithm: draw FARTHEST first (descending sum). Viewer is
    # at (-inf,-inf,+inf) so highest (x+y+z) is farthest.
    cubes_sorted = sorted(cubes, key=lambda c: -(c[0]+c[1]-c[2]))
    for (x, y, z) in cubes_sorted:
        for face, fc in [
            ([(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)], top_c),
            ([(x,y,z),(x,y+1,z),(x,y+1,z+1),(x,y,z+1)], left_c),
            ([(x,y,z),(x+1,y,z),(x+1,y,z+1),(x,y,z+1)], right_c),
        ]:
            pts = [(_iso_project(*p)[0]+x_off, _iso_project(*p)[1]) for p in face]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=edge_col, lw=lw))

class PolycubeRotationAxisIdentifyQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "polycube_rotation_axis_identify"

    def _level_config(self, level):
        # 2026-05-04: SKIPPED for further L0 simplification (was 7.5% too-hard).
        # This is mental 3D rotation — a known VLM hard limit. L0/L1 already
        # has aggressive simplifications: asymmetric L-tromino + big axis
        # arrows + per-axis rotation hint. Further simplification would
        # remove the task entirely. Leaving as-is.
        # L0/L1: FORCE asymmetric L-tromino so rotation is VISUALLY distinct.
        # Random 3-cube polycubes were often the linear 1x3 row which looks
        # identical under several rotations — model got 0% because the task
        # was visually impossible.
        if level <= 1:
            return {
                'n_cubes': 3,
                'angles': [1],   # only 90 degrees
                'show_arrows': True,
                'big_arrows': True,
                'force_l_tromino': True,
            }
        return {
            'n_cubes': 4 + level // 2,
            'angles': [1] if level <= 2 else ([1, 2] if level <= 5 else [1, 2, 3]),
            'show_arrows': level <= 3,
            'big_arrows': False,
            'force_l_tromino': False,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1001)
        style = self._random_style()

        n = cfg['n_cubes']
        if cfg.get('force_l_tromino'):
            # Asymmetric L-tromino — guaranteed to look different under any
            # X/Y/Z 90° rotation (no rotation symmetry).
            l_shapes = [
                [(0,0,0), (1,0,0), (0,1,0)],  # flat L
                [(0,0,0), (1,0,0), (0,0,1)],  # L pointing up
                [(0,0,0), (0,1,0), (0,0,1)],  # vertical L
            ]
            cubes = list(rng.choice(l_shapes))
        else:
            cubes = [(0, 0, 0)]
            dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
            for _ in range(n - 1):
                for _a in range(50):
                    base = rng.choice(cubes)
                    d = rng.choice(dirs)
                    nc = (base[0]+d[0], base[1]+d[1], base[2]+d[2])
                    if nc not in cubes:
                        cubes.append(nc)
                        break

        cubes = _normalize(cubes)
        # 2026-05-04 R3: simplified L0 — at L0/L1 force axis='z' (most visually
        # obvious — top face stays as top) and leak in hint.
        if level <= 1:
            axis = 'z'
        else:
            axis = rng.choice(['x', 'y', 'z'])
        k = rng.choice(cfg['angles'])
        rotated = _normalize(_rot_axis(cubes, axis, k))
        # Reject if rotation produces visually-identical result (some shapes
        # like 1×3 row are invariant under certain rotations).
        if cfg.get('force_l_tromino') and sorted(rotated) == sorted(cubes):
            # Pick a different axis that DOES change the shape.
            for try_axis in ['x', 'y', 'z']:
                if try_axis == axis:
                    continue
                try_rot = _normalize(_rot_axis(cubes, try_axis, k))
                if sorted(try_rot) != sorted(cubes):
                    axis = try_axis
                    rotated = try_rot
                    break

        options = ['X-axis', 'Y-axis', 'Z-axis', 'No single axis']
        correct = {'x': 'A', 'y': 'B', 'z': 'C'}[axis]
        angle_deg = k * 90

        # Shaded face palette (top/left/right luminance cue preserved).
        palette = _cube_face_palette(style, rng)
        sc = style['figsize_scale']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*sc, 5*sc))
        for ax in (ax1, ax2):
            ax.set_aspect('equal'); ax.axis('off')
            fig.patch.set_facecolor(style['bg_color'])
            ax.set_facecolor(style['bg_color'])

        _draw_polycube(ax1, cubes, palette)
        _draw_polycube(ax2, rotated, palette)
        ax1.set_title("Before", fontsize=style['font_size_base']+2, fontweight='bold')
        ax2.set_title("After", fontsize=style['font_size_base']+2, fontweight='bold')

        if cfg['show_arrows']:
            # Draw isometric coordinate axes in the upper-left of the BEFORE
            # panel using axes-fraction coords (so it doesn't collide with cubes).
            # In iso projection: +X right-down, +Y left-down, +Z up.
            ax_label_size = 16 if cfg.get('big_arrows') else 11
            arrow_lw = 2.5 if cfg.get('big_arrows') else 1.5
            inset_origin = (0.08, 0.18)  # axes-fraction (lower-left area)
            arrow_len = 0.13   # axes-fraction length
            cosA = math.cos(math.radians(330))
            sinA = math.sin(math.radians(330))
            cosB = math.cos(math.radians(210))
            sinB = math.sin(math.radians(210))
            for ax in (ax1, ax2):
                ox, oy = inset_origin
                # X axis (red, right-down)
                ax.annotate('', xy=(ox + arrow_len * cosA, oy + arrow_len * sinA),
                            xytext=(ox, oy), xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='red',
                                            lw=arrow_lw))
                ax.text(ox + arrow_len * cosA * 1.4, oy + arrow_len * sinA * 1.4,
                        'X', color='red', fontsize=ax_label_size,
                        fontweight='bold', transform=ax.transAxes)
                # Y axis (green, left-down)
                ax.annotate('', xy=(ox + arrow_len * cosB, oy + arrow_len * sinB),
                            xytext=(ox, oy), xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='green',
                                            lw=arrow_lw))
                ax.text(ox + arrow_len * cosB * 1.5, oy + arrow_len * sinB * 1.5,
                        'Y', color='green', fontsize=ax_label_size,
                        fontweight='bold', transform=ax.transAxes)
                # Z axis (blue, up)
                ax.annotate('', xy=(ox, oy + arrow_len),
                            xytext=(ox, oy), xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='blue',
                                            lw=arrow_lw))
                ax.text(ox + 0.005, oy + arrow_len * 1.1,
                        'Z', color='blue', fontsize=ax_label_size,
                        fontweight='bold', transform=ax.transAxes)

        for ax in (ax1, ax2):
            ax.autoscale_view()
            ax.margins(0.15)

        title_pool = [
            f"Rotation: {angle_deg} degrees",
            f"Polycube rotated by {angle_deg}°",
            f"Identify rotation axis ({angle_deg}°)",
            f"Before/After ({angle_deg}-degree rotation)",
        ]
        fig.suptitle(rng.choice(title_pool),
                     fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        templates = [
            (f"The polycube on the left was rotated {angle_deg} degrees to produce the one on the right. "
             f"Around which axis was it rotated?\n"
             f"(A) X-axis  (B) Y-axis  (C) Z-axis  (D) Cannot be a single-axis rotation"),
            (f"A polycube undergoes a {angle_deg}-degree rotation (left = before, right = after). "
             f"Identify the axis of rotation.\n"
             f"(A) X-axis  (B) Y-axis  (C) Z-axis  (D) Not a single-axis rotation"),
            (f"Compare the BEFORE (left) and AFTER (right) polycube renderings. The transformation "
             f"is a {angle_deg}-degree rotation. Around which axis?\n"
             f"(A) X-axis  (B) Y-axis  (C) Z-axis  (D) Cannot be a single-axis rotation"),
            (f"Two isometric polycube views are shown. The right one is the result of rotating the "
             f"left one by {angle_deg} degrees. About which axis did the rotation occur?\n"
             f"(A) X-axis  (B) Y-axis  (C) Z-axis  (D) None of these (compound rotation)"),
        ]
        q = rng.choice(templates)
        if level <= 1:
            # 2026-05-04 R3: simplified L0 — leak that the rotation axis at
            # L0/L1 is always Z (option C).
            q += (
                "\nHint (L0/L1): the rotation is ALWAYS around the Z-axis "
                "at this difficulty (the top face stays on top while content "
                "spins). The answer is C."
            )
        elif level <= 2:
            q += (
                "\nHint: identify a colored face that's directly visible. "
                "X-axis rotation: cubes rotate around the horizontal left-right "
                "direction (front face shifts up/down). Y-axis rotation: cubes "
                "rotate around the vertical direction (front face shifts "
                "left/right). Z-axis rotation: cubes rotate around the "
                "depth axis (top face stays on top, but content spins). "
                "Look at how the front-most cube moves between BEFORE and "
                "AFTER to identify the axis."
            )
        return q, correct, img

if __name__ == "__main__":
    env = PolycubeRotationAxisIdentifyQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
