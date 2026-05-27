"""
Mirror Plane Identification QA.

Shows a symmetric 3D object rendered isometrically with candidate mirror planes
drawn as colored semi-transparent rectangles. Asks which planes are valid.

Difficulty axes:
  A) n_candidate_planes: 2..3
  B) solid_complexity: rectangular prism -> irregular shapes
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z):
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

def _draw_block(ax, cubes, palette, lw=1.0):
    for (x, y, z) in sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2])):
        for face, fc in [
            ([(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)], palette[0]),
            ([(x,y,z),(x,y+1,z),(x,y+1,z+1),(x,y,z+1)], palette[1]),
            ([(x,y,z),(x+1,y,z),(x+1,y,z+1),(x,y,z+1)], palette[2]),
        ]:
            pts = [_iso_project(*p) for p in face]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor='#333', lw=lw))

class MirrorPlaneIdentificationQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "mirror_plane_identification"

    def _level_config(self, level):
        # Previous: grid_size=2..5, n_planes jumps from 2 to 3 at L4. With
        # grid_size=5 at L9 the scene was too cluttered (0.10 pass-rate).
        # Cap grid_size at 4 and pace n_planes across the scale.
        # Iter 3 (2026-04-17): L3=0.55 spike vs L0=0.40. gs=2 at L0 produced
        # 2x2x2 blocks where the small footprint + random 0.7 fill often
        # yielded degenerate visuals (1-2 cubes). Standardise on gs=3 at
        # L0-4 so L0 is a fully-formed block with 2 candidate planes, and
        # only bump plane count at L5+.
        level = max(0, min(9, int(level)))
        if level <= 2:
            gs, np_ = 3, 2
        elif level <= 4:
            gs, np_ = 3, 2
        elif level <= 6:
            gs, np_ = 3, 3
        elif level <= 8:
            gs, np_ = 4, 3
        else:
            gs, np_ = 4, 3            # was grid_size=5 before — too cluttered
        return {
            'n_planes': np_,
            'grid_size': gs,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1005)
        style = self._random_style()

        gs = cfg['grid_size']
        # Build a symmetric block structure
        cubes = []
        for x in range(gs):
            for y in range(gs):
                for z in range(min(gs, 3)):
                    if rng.random() < 0.7:
                        cubes.append((x, y, z))
        if not cubes:
            cubes = [(0,0,0), (1,0,0), (0,1,0)]

        # Determine actual symmetry planes
        # Check XZ plane (y-symmetry), YZ plane (x-symmetry), XY plane (z-symmetry)
        cube_set = set(cubes)
        max_x = max(c[0] for c in cubes)
        max_y = max(c[1] for c in cubes)
        max_z = max(c[2] for c in cubes)

        planes = []
        # YZ mirror (reflect x)
        yz_sym = all(((max_x - c[0], c[1], c[2]) in cube_set) for c in cubes)
        planes.append(('YZ (red)', yz_sym, 'red'))
        # XZ mirror (reflect y)
        xz_sym = all(((c[0], max_y - c[1], c[2]) in cube_set) for c in cubes)
        planes.append(('XZ (blue)', xz_sym, 'blue'))
        # XY mirror (reflect z)
        xy_sym = all(((c[0], c[1], max_z - c[2]) in cube_set) for c in cubes)
        planes.append(('XY (green)', xy_sym, 'green'))

        rng.shuffle(planes)
        selected = planes[:cfg['n_planes']]

        valid_names = [p[0] for p in selected if p[1]]
        if len(valid_names) == 0:
            answer_text = "None"
        elif len(valid_names) == 1:
            answer_text = valid_names[0].split(' ')[0] + " only"
        else:
            answer_text = " and ".join(p.split(' ')[0] for p in valid_names)

        # Build MCQ (shuffle so correct letter varies across seeds)
        plane_names_short = [p[0].split(' ')[0] for p in selected]
        if cfg['n_planes'] == 2:
            raw_options = [
                (f"{plane_names_short[0]} only", [plane_names_short[0]]),
                (f"{plane_names_short[1]} only", [plane_names_short[1]]),
                (f"Both {plane_names_short[0]} and {plane_names_short[1]}", plane_names_short[:2]),
                ("Neither", []),
            ]
        else:
            raw_options = [
                (f"{plane_names_short[0]} only", [plane_names_short[0]]),
                (f"{plane_names_short[1]} only", [plane_names_short[1]]),
                (f"{plane_names_short[0]} and {plane_names_short[1]}", plane_names_short[:2]),
                ("All three", plane_names_short[:3]),
            ]

        # Determine the correct option by exact set match of valid mirror planes
        valid_short = [n.split(' ')[0] for n in valid_names]
        correct_payload = sorted(valid_short)
        correct_idx_orig = -1
        for i, (_, payload) in enumerate(raw_options):
            if sorted(payload) == correct_payload:
                correct_idx_orig = i
                break
        if correct_idx_orig < 0:
            correct_idx_orig = len(raw_options) - 1  # fallback: "Neither" / "All three"

        order = list(range(4))
        rng.shuffle(order)
        options = [raw_options[i][0] for i in order]
        correct_idx = order.index(correct_idx_orig)
        correct = "ABCD"[correct_idx]

        # Draw
        palette = list(style['palette'][:3])
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(7*sc, 7*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal'); ax.axis('off')

        _draw_block(ax, cubes, palette)

        # Draw plane indicators as semi-transparent parallelograms — true
        # projected plane slices through the object — rather than single
        # lines. Each plane is the locus of midpoints along the reflection
        # axis; we render it as a quadrilateral spanning the bounding box.
        # A short dashed outline keeps the plane visible against the cube
        # faces; a faint fill communicates it is a 2D slice, not an edge.
        from matplotlib.patches import Polygon as _Poly
        for pname, is_valid, color in selected:
            mid_x = (max_x + 1) / 2.0
            mid_y = (max_y + 1) / 2.0
            mid_z = (max_z + 1) / 2.0
            pad = 0.6
            if 'YZ' in pname:
                # plane x = mid_x, spanning (y in [-pad, max_y+1+pad], z in
                # [-pad, max_z+1+pad])
                corners_3d = [
                    (mid_x, -pad,          -pad),
                    (mid_x, max_y + 1 + pad, -pad),
                    (mid_x, max_y + 1 + pad, max_z + 1 + pad),
                    (mid_x, -pad,          max_z + 1 + pad),
                ]
            elif 'XZ' in pname:
                corners_3d = [
                    (-pad,          mid_y, -pad),
                    (max_x + 1 + pad, mid_y, -pad),
                    (max_x + 1 + pad, mid_y, max_z + 1 + pad),
                    (-pad,          mid_y, max_z + 1 + pad),
                ]
            else:  # XY plane
                corners_3d = [
                    (-pad,          -pad,          mid_z),
                    (max_x + 1 + pad, -pad,          mid_z),
                    (max_x + 1 + pad, max_y + 1 + pad, mid_z),
                    (-pad,          max_y + 1 + pad, mid_z),
                ]
            corners_2d = [_iso_project(*p) for p in corners_3d]
            # Semi-transparent fill + dashed outline = clear plane cue
            ax.add_patch(_Poly(corners_2d, closed=True,
                               facecolor=color, edgecolor=color,
                               linewidth=2.2, linestyle='--',
                               alpha=0.22, label=pname, zorder=5))

        ax.legend(fontsize=style['font_size_base'], loc='upper left')
        ax.autoscale_view(); ax.margins(0.15)
        ax.set_title("Mirror Plane Identification", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        opt_str = "  ".join(f"({chr(65+i)}) {options[i]}" for i in range(len(options)))
        q = (f"The colored translucent rectangles represent candidate mirror "
             f"planes slicing through the block structure. Which plane(s) are "
             f"true mirror symmetry planes of this block structure?\n{opt_str}")
        return q, correct, img

if __name__ == "__main__":
    env = MirrorPlaneIdentificationQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
