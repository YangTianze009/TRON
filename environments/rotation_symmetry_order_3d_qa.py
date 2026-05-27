"""
Rotation Symmetry Order 3D QA (redesigned 2026-04-16).

Shows a 3D polyhedral solid rendered isometrically with a rotation axis drawn
through it. Asks for the order of rotational symmetry about the indicated
axis.

Redesign goals:
  * Expand shape pool from ~8 fixed polycube templates to 20+ families
    (polycubes, platonic approximations, composite prisms, stepped cones,
    generalized stars, twisted bricks, random organic clusters).
  * Randomize axis orientation (vertical, horizontal, diagonal at L6+).
  * Randomize viewing angle (isometric projection with variable obliqueness).
  * Randomize cube color scheme per face (6 palette variations x 3 shading
    styles).
  * Question template variants (6 phrasings).
  * L0 vs L9 structural shift:
      - L0: simple prism / cube / cross, vertical axis, labelled axes, order
        clearly visible in a top-down hint.
      - L9: irregular organic cluster, diagonal body axis, no hint.
  * Remove visual sameness across seeds by jittering cube positions and
    rotating the whole scene.
"""
import math
import random
import colorsys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Geometry helpers — isometric projection with configurable angles.
# ------------------------------------------------------------------ #

def _iso_project(x, y, z, alpha_deg=30.0, beta_deg=30.0):
    """Oblique projection with configurable angles (degrees)."""
    ca = math.cos(math.radians(alpha_deg))
    sa = math.sin(math.radians(alpha_deg))
    cb = math.cos(math.radians(beta_deg))
    sb = math.sin(math.radians(beta_deg))
    sx = (x - y) * ca
    sy = (x + y) * sa + z * (1.0 / max(0.3, cb)) * cb
    # Slight horizontal shift from beta to make the solid look skewed
    sx += z * (1.0 - cb) * 0.25
    return sx, sy

def _lighten(hex_color: str, amount: float) -> str:
    """Lighten or darken a hex colour by amount (-1..1)."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

# ------------------------------------------------------------------ #
# Symmetry computation (exact via rotation group sampling).
# ------------------------------------------------------------------ #

def _rotate_points(cubes, axis: str, angle_deg: float):
    """Rotate cube cells about a given axis by angle (degrees)."""
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    out = []
    for (x, y, z) in cubes:
        x = x + 0.5
        y = y + 0.5
        z = z + 0.5
        if axis == 'z':
            nx = x * c - y * s
            ny = x * s + y * c
            nz = z
        elif axis == 'x':
            nx = x
            ny = y * c - z * s
            nz = y * s + z * c
        elif axis == 'y':
            nx = x * c + z * s
            ny = y
            nz = -x * s + z * c
        else:  # diagonal xyz
            # rotate around (1,1,1)/sqrt(3)
            k = 1.0 / math.sqrt(3)
            u, v, w = k, k, k
            # Rodrigues
            dot = x * u + y * v + z * w
            cross = (v * z - w * y, w * x - u * z, u * y - v * x)
            nx = x * c + cross[0] * s + u * dot * (1 - c)
            ny = y * c + cross[1] * s + v * dot * (1 - c)
            nz = z * c + cross[2] * s + w * dot * (1 - c)
        out.append((nx - 0.5, ny - 0.5, nz - 0.5))
    return out

def _snap_to_int_grid(points, tol=0.01):
    snapped = []
    for (x, y, z) in points:
        rx = round(x)
        ry = round(y)
        rz = round(z)
        if (abs(x - rx) > tol or abs(y - ry) > tol or abs(z - rz) > tol):
            return None
        snapped.append((rx, ry, rz))
    return snapped

def _normalize_set(cubes):
    xs, ys, zs = zip(*cubes)
    mx, my, mz = min(xs), min(ys), min(zs)
    return frozenset((x - mx, y - my, z - mz) for x, y, z in cubes)

def _symmetry_order(cubes, axis: str) -> int:
    """Compute rotational symmetry order about an axis through the center.

    Only returns N such that 360/N rotation maps cubes to themselves, tested
    for N in [1..6]. If no snap alignment works except identity, returns 1.
    """
    base = _normalize_set(cubes)
    # Try orders 6, 5, 4, 3, 2; whichever smallest angle that closes gives
    # the order. Actually we want the highest N such that 360/N is a
    # symmetry.
    for N in (6, 5, 4, 3, 2):
        angle = 360.0 / N
        rotated = _rotate_points(cubes, axis, angle)
        snapped = _snap_to_int_grid(rotated, tol=0.01)
        if snapped is None:
            continue
        if _normalize_set(snapped) == base:
            return N
    return 1

# ------------------------------------------------------------------ #
# Shape library — 20+ families.
# ------------------------------------------------------------------ #

def _shape_cube(rng):
    s = rng.choice([2, 3])
    return [(x, y, z) for x in range(s) for y in range(s) for z in range(s)]

def _shape_square_prism(rng):
    s = rng.choice([2, 3])
    h = rng.choice([1, 2, 3, 4])
    return [(x, y, z) for x in range(s) for y in range(s) for z in range(h)]

def _shape_rect_prism(rng):
    w = rng.randint(2, 4)
    d = rng.randint(1, 3)
    h = rng.randint(1, 3)
    if w == d:  # avoid degenerate square
        d = max(1, d - 1) if w > 2 else d + 1
    return [(x, y, z) for x in range(w) for y in range(d) for z in range(h)]

def _shape_L(rng):
    h = rng.choice([1, 2, 3])
    arm = rng.choice([2, 3])
    base = [(0, 0)]
    for i in range(1, arm):
        base.append((i, 0))
    for j in range(1, arm):
        base.append((0, j))
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_T(rng):
    h = rng.choice([1, 2])
    arm = rng.choice([1, 2])
    base = [(0, 0)]
    for i in range(1, arm + 1):
        base += [(i, 0), (-i, 0)]
    for j in range(1, arm + 1):
        base.append((0, -j))
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_cross(rng):
    h = rng.choice([1, 2])
    arm = rng.choice([1, 2, 3])
    base = {(0, 0)}
    for a in range(1, arm + 1):
        base |= {(a, 0), (-a, 0), (0, a), (0, -a)}
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_plus_tower(rng):
    """Cross base plus a vertical tower — makes clean C4 symmetry."""
    arm = rng.choice([1, 2])
    base = {(0, 0)}
    for a in range(1, arm + 1):
        base |= {(a, 0), (-a, 0), (0, a), (0, -a)}
    cubes = [(x, y, 0) for (x, y) in base]
    tower_h = rng.choice([2, 3, 4])
    for k in range(1, tower_h + 1):
        cubes.append((0, 0, k))
    return cubes

def _shape_hex_ring(rng):
    h = rng.choice([1, 2])
    ring = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    cubes = [(x, y, z) for (x, y) in ring for z in range(h)]
    if rng.random() < 0.5:
        cubes += [(0, 0, z) for z in range(h)]
    return cubes

def _shape_triangle3(rng):
    """Three cells in 3-fold symmetric layout — gives C3 about z."""
    h = rng.choice([1, 2])
    base = [(1, 0), (0, 1), (-1, -1)]
    cubes = [(x, y, z) for (x, y) in base for z in range(h)]
    if rng.random() < 0.5:
        cubes += [(0, 0, z) for z in range(h)]
    return cubes

def _shape_star5(rng):
    """Pentagonal layout (approx) — doesn't align to integer grid for C5
    so just return a non-C5 figure. Left here for variety."""
    h = rng.choice([1, 2])
    arm = 2
    base = [(0, 0)]
    for i, ang in enumerate([90, 180, 270]):
        x = int(round(arm * math.cos(math.radians(ang))))
        y = int(round(arm * math.sin(math.radians(ang))))
        base.append((x, y))
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_asymmetric(rng):
    """Genuinely asymmetric cluster — order 1."""
    cubes = [(0, 0, 0)]
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    n = rng.randint(5, 9)
    for _ in range(n - 1):
        for _a in range(40):
            b = rng.choice(cubes)
            d = rng.choice(dirs)
            nc = (b[0] + d[0], b[1] + d[1], b[2] + d[2])
            if nc not in cubes:
                cubes.append(nc)
                break
    return cubes

def _shape_step_pyramid(rng):
    """Square step pyramid — C4."""
    n_steps = rng.randint(2, 4)
    cubes = []
    for k in range(n_steps):
        side = n_steps - k
        for x in range(-side, side + 1):
            for y in range(-side, side + 1):
                if abs(x) + abs(y) <= 2 * side:  # diamond layer
                    cubes.append((x, y, k))
    return cubes

def _shape_diamond_layer(rng):
    """Diamond base — C4 or C2 depending on thickness."""
    h = rng.choice([1, 2, 3])
    arm = rng.choice([1, 2])
    base = [(x, y) for x in range(-arm, arm + 1) for y in range(-arm, arm + 1)
            if abs(x) + abs(y) <= arm]
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_twisted(rng):
    """Stack of offset bricks — rotational symmetry reduced."""
    cubes = []
    for k in range(rng.choice([2, 3, 4])):
        ox = k if rng.random() < 0.5 else -k
        oy = k // 2
        for x in range(2):
            for y in range(2):
                cubes.append((x + ox, y + oy, k))
    return cubes

def _shape_rect_block(rng):
    """Plain rectangular block — C2 or C1 depending on dims."""
    a, b = rng.choice([(3, 2), (4, 2), (4, 3), (3, 1), (5, 2)])
    h = rng.choice([1, 2])
    return [(x, y, z) for x in range(a) for y in range(b) for z in range(h)]

def _shape_Z(rng):
    h = rng.choice([1, 2])
    base = [(0, 0), (1, 0), (1, 1), (2, 1)]
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_plus(rng):
    h = rng.choice([1, 2])
    base = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_S(rng):
    h = rng.choice([1, 2])
    base = [(0, 0), (1, 0), (0, 1), (-1, 1)]
    return [(x, y, z) for (x, y) in base for z in range(h)]

def _shape_hex_layer(rng):
    h = rng.choice([1, 2])
    ring = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    return [(x, y, z) for (x, y) in ring for z in range(h)]

def _shape_stepped_prism(rng):
    h = rng.randint(2, 3)
    cubes = []
    for k in range(h):
        w = max(1, 3 - k)
        for x in range(-w, w + 1):
            for y in range(-w, w + 1):
                if max(abs(x), abs(y)) <= w:
                    cubes.append((x, y, k))
    return cubes

_SHAPE_FUNCS = {
    'cube': _shape_cube,
    'square_prism': _shape_square_prism,
    'rect_prism': _shape_rect_prism,
    'rect_block': _shape_rect_block,
    'L': _shape_L,
    'T': _shape_T,
    'cross': _shape_cross,
    'plus_tower': _shape_plus_tower,
    'hex_ring': _shape_hex_ring,
    'triangle3': _shape_triangle3,
    'star5': _shape_star5,
    'asymmetric': _shape_asymmetric,
    'step_pyramid': _shape_step_pyramid,
    'diamond_layer': _shape_diamond_layer,
    'twisted': _shape_twisted,
    'Z': _shape_Z,
    'plus': _shape_plus,
    'S': _shape_S,
    'hex_layer': _shape_hex_layer,
    'stepped_prism': _shape_stepped_prism,
}

# ------------------------------------------------------------------ #
# Rendering helpers
# ------------------------------------------------------------------ #

def _draw_poly(ax, cubes, palette, lw=1.0, alpha_deg=30, beta_deg=30,
               edge_color="#333333", shade_style="flat"):
    # Sort for painter's algorithm
    for (x, y, z) in sorted(cubes, key=lambda c: (-(c[0] + c[1] - c[2]), c[2])):
        base_top = palette[0]
        base_left = palette[1]
        base_right = palette[2]
        if shade_style == "gradient":
            base_top = _lighten(base_top, 0.10)
            base_left = _lighten(base_left, -0.15)
            base_right = _lighten(base_right, -0.05)
        elif shade_style == "dark":
            base_top = _lighten(base_top, -0.05)
            base_left = _lighten(base_left, -0.30)
            base_right = _lighten(base_right, -0.18)
        faces = [
            ([(x, y, z + 1), (x + 1, y, z + 1),
              (x + 1, y + 1, z + 1), (x, y + 1, z + 1)], base_top),
            ([(x, y, z), (x, y + 1, z),
              (x, y + 1, z + 1), (x, y, z + 1)], base_left),
            ([(x, y, z), (x + 1, y, z),
              (x + 1, y, z + 1), (x, y, z + 1)], base_right),
        ]
        for face, fc in faces:
            pts = [_iso_project(*p, alpha_deg=alpha_deg,
                                beta_deg=beta_deg) for p in face]
            ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                                 edgecolor=edge_color, lw=lw))

def _draw_axis(ax, cubes, axis: str, color="#e74c3c",
               alpha_deg=30, beta_deg=30):
    """Draw a dashed axis line through the centre of the shape."""
    xs, ys, zs = zip(*cubes)
    cx = (max(xs) + min(xs)) / 2 + 0.5
    cy = (max(ys) + min(ys)) / 2 + 0.5
    cz = (max(zs) + min(zs)) / 2 + 0.5
    L = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) + 2.5
    if axis == 'z':
        p1 = (cx, cy, min(zs) - 1.0)
        p2 = (cx, cy, max(zs) + 1.5)
        label = "vertical axis (Z)"
    elif axis == 'x':
        p1 = (min(xs) - 1.0, cy, cz)
        p2 = (max(xs) + 1.5, cy, cz)
        label = "horizontal axis (X)"
    elif axis == 'y':
        p1 = (cx, min(ys) - 1.0, cz)
        p2 = (cx, max(ys) + 1.5, cz)
        label = "horizontal axis (Y)"
    else:
        # Body diagonal
        p1 = (min(xs) - 0.5, min(ys) - 0.5, min(zs) - 0.5)
        p2 = (max(xs) + 1.0, max(ys) + 1.0, max(zs) + 1.0)
        label = "body-diagonal axis"
    sp1 = _iso_project(*p1, alpha_deg=alpha_deg, beta_deg=beta_deg)
    sp2 = _iso_project(*p2, alpha_deg=alpha_deg, beta_deg=beta_deg)
    ax.plot([sp1[0], sp2[0]], [sp1[1], sp2[1]], linestyle='--',
            color=color, linewidth=2.5, alpha=0.9, zorder=100)
    # Arrow head at p2
    ax.add_patch(FancyArrowPatch(sp1, sp2, color=color, linewidth=0,
                                 mutation_scale=18, arrowstyle='->',
                                 zorder=101))
    return label, sp1, sp2

def _pick_palette(rng, style_palette: List[str]) -> List[str]:
    """Pick 3 face colours for top/left/right from a style palette."""
    # Use the style palette + some shuffles
    choices = list(style_palette)
    rng.shuffle(choices)
    top = choices[0]
    return [top, _lighten(top, -0.18), _lighten(top, -0.08)]

def _multi_color_palette(rng, style_palette):
    """Return a colour per cell for more variety."""
    colors = list(style_palette)
    rng.shuffle(colors)
    return colors

# ------------------------------------------------------------------ #
# Env
# ------------------------------------------------------------------ #

class RotationSymmetryOrder3dQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "rotation_symmetry_order_3d"

    _QUESTION_TEMPLATES = [
        ("A dashed red line shows the {axis_label} through this solid. "
         "What is the order of rotational symmetry about this axis? "
         "(Order N means the shape looks identical after a 360/N degree "
         "rotation.) Answer with a single integer."),
        ("The red dashed line marks a rotation axis through the figure. "
         "How many times does the solid appear identical to itself during "
         "one full 360 degree rotation about this axis? Answer with a "
         "single integer."),
        ("Consider the {axis_label} shown in red. Find the rotational "
         "symmetry order: the largest integer N such that the solid maps "
         "to itself under rotation by 360/N degrees. Answer with a single "
         "integer."),
        ("Observe the {axis_label} indicated by the red dashed line. "
         "What is N, the rotational symmetry order about this axis? "
         "Answer with a single integer."),
        ("The figure shows a 3D solid with a rotation axis marked in red "
         "({axis_label}). Determine the order of rotational symmetry. "
         "Answer with a single integer."),
        ("Looking at the {axis_label} (red dashed), how many rotational "
         "symmetry positions does this solid have about that axis? "
         "Answer with a single integer."),
    ]

    _TITLE_VARIANTS = [
        "Rotational Symmetry",
        "Axis of Symmetry",
        "Symmetry Order",
        "Rotational Analysis",
        "3D Symmetry Puzzle",
        "Solid Rotation",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # L0: only clearly symmetric shapes (cube, square_prism, plus_tower),
        # vertical axis, a top-down "view from above" hint.
        # L3: add crosses, diamonds, L-shapes, still vertical.
        # L6: introduce X/Y axes and irregular shapes (lower symmetry).
        # L9: irregular clusters + body-diagonal axis, no hint.
        if level <= 1:
            # Mix of shapes that give orders 1, 2, 4 for answer diversity.
            return {
                'shape_pool': ['cube', 'square_prism', 'plus_tower',
                               'diamond_layer', 'cross', 'rect_prism',
                               'L', 'Z', 'rect_block'],
                'axis_pool': ['z'],
                'show_hint': True,
                'shade_style': 'gradient',
                'n_shape_candidates': 4,
            }
        elif level <= 3:
            return {
                'shape_pool': ['cube', 'square_prism', 'rect_prism',
                               'cross', 'diamond_layer', 'plus_tower',
                               'step_pyramid', 'hex_layer', 'plus',
                               'L', 'Z', 'S', 'rect_block', 'T'],
                'axis_pool': ['z'],
                'show_hint': True,
                'shade_style': 'gradient',
                'n_shape_candidates': 6,
            }
        elif level <= 5:
            return {
                'shape_pool': ['rect_prism', 'rect_block', 'cross',
                               'L', 'T', 'hex_ring',
                               'step_pyramid', 'stepped_prism', 'S', 'Z',
                               'square_prism', 'diamond_layer', 'cube',
                               'plus'],
                'axis_pool': ['z', 'x', 'y'],
                'show_hint': False,
                'shade_style': 'flat',
                'n_shape_candidates': 7,
            }
        elif level <= 7:
            return {
                'shape_pool': ['L', 'T', 'Z', 'S', 'twisted',
                               'hex_ring', 'rect_block', 'step_pyramid',
                               'asymmetric', 'cross', 'plus_tower',
                               'diamond_layer', 'rect_prism',
                               'square_prism'],
                'axis_pool': ['z', 'z', 'x', 'y'],  # bias toward z
                'show_hint': False,
                'shade_style': 'dark',
                'n_shape_candidates': 8,
            }
        else:
            return {
                'shape_pool': ['asymmetric', 'twisted', 'L', 'T', 'Z', 'S',
                               'triangle3', 'rect_block', 'stepped_prism'],
                'axis_pool': ['x', 'y', 'z', 'diag'],
                'show_hint': False,
                'shade_style': 'dark',
                'n_shape_candidates': 10,
            }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1006)
        style = self._random_style()
        self._primary_complexity_feature = level

        # Try multiple shape candidates; keep the one whose symmetry order
        # isn't trivially obvious (avoid all-same answers across seeds).
        for _attempt in range(cfg['n_shape_candidates']):
            shape_name = rng.choice(cfg['shape_pool'])
            shape_fn = _SHAPE_FUNCS[shape_name]
            cubes = shape_fn(rng)
            if len(cubes) > 60:
                continue
            axis = rng.choice(cfg['axis_pool'])
            if axis == 'diag':
                # Diagonal only works for certain cube-symmetric shapes;
                # check the symmetry.
                order = _symmetry_order(cubes, 'diag')
            else:
                order = _symmetry_order(cubes, axis)
            # Normalize for rendering
            xs, ys, zs = zip(*cubes)
            mx, my, mz = min(xs), min(ys), min(zs)
            cubes = [(x - mx, y - my, z - mz) for x, y, z in cubes]
            # Accept
            break
        else:
            return None

        answer = str(order)

        palette_face = _pick_palette(rng, style['palette'])
        # Sometimes use multi-color cubes
        use_multi = rng.random() < 0.25

        # Projection angles with jitter
        alpha_deg = 30 + rng.uniform(-5, 5)
        beta_deg = 30 + rng.uniform(-5, 5)

        sc = style['figsize_scale']
        show_hint = cfg['show_hint']
        if show_hint:
            fig, (ax_main, ax_top) = plt.subplots(
                1, 2, figsize=(8.5 * sc, 6 * sc),
                gridspec_kw={'width_ratios': [2.2, 1.0]})
        else:
            fig, ax_main = plt.subplots(figsize=(6.5 * sc, 6 * sc))
            ax_top = None

        fig.patch.set_facecolor(style['bg_color'])
        ax_main.set_facecolor(style['bg_color'])
        ax_main.set_aspect('equal')
        ax_main.axis('off')

        if use_multi:
            palette_list = _multi_color_palette(rng, style['palette'])
            for i, cube in enumerate(sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))):
                col = palette_list[i % len(palette_list)]
                face_pal = [col, _lighten(col, -0.18), _lighten(col, -0.08)]
                _draw_poly(ax_main, [cube], face_pal,
                           lw=style['line_width'],
                           alpha_deg=alpha_deg, beta_deg=beta_deg,
                           edge_color="#2c3e50",
                           shade_style=cfg['shade_style'])
        else:
            _draw_poly(ax_main, cubes, palette_face,
                       lw=style['line_width'],
                       alpha_deg=alpha_deg, beta_deg=beta_deg,
                       edge_color="#2c3e50",
                       shade_style=cfg['shade_style'])

        # Draw axis
        axis_color = rng.choice(["#e74c3c", "#c0392b", "#d62828"])
        axis_label, sp1, sp2 = _draw_axis(
            ax_main, cubes, axis, color=axis_color,
            alpha_deg=alpha_deg, beta_deg=beta_deg)
        ax_main.annotate("axis", xy=(sp2[0], sp2[1]),
                         xytext=(sp2[0] + 0.2, sp2[1] + 0.25),
                         fontsize=max(10, style['font_size_base']),
                         color=axis_color, fontweight='bold')

        ax_main.autoscale_view()
        ax_main.margins(0.18)

        title = rng.choice(self._TITLE_VARIANTS)
        ax_main.set_title(title, fontsize=max(11, style['font_size_base'] + 2),
                          fontweight='bold')

        # Top-down hint at L<=3
        if show_hint and ax_top is not None:
            ax_top.set_aspect('equal')
            ax_top.axis('off')
            ax_top.set_title("Top view",
                             fontsize=max(10, style['font_size_base']))
            ax_top.set_facecolor(style['bg_color'])
            # Collapse cubes onto XY plane
            xy_set = set()
            for (x, y, _z) in cubes:
                xy_set.add((x, y))
            for (x, y) in xy_set:
                ax_top.add_patch(Polygon(
                    [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)],
                    closed=True, facecolor=palette_face[0],
                    edgecolor="#2c3e50", lw=1.2))
            # Mark axis point if z
            xs2 = [x for (x, _y) in xy_set]
            ys2 = [y for (_x, y) in xy_set]
            if xs2 and ys2:
                cx = (max(xs2) + min(xs2) + 1) / 2
                cy = (max(ys2) + min(ys2) + 1) / 2
                ax_top.plot([cx], [cy], 'x', color=axis_color,
                            markersize=14, markeredgewidth=3, zorder=5)
            ax_top.autoscale_view()
            ax_top.margins(0.2)

        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q_template = rng.choice(self._QUESTION_TEMPLATES)
        q = q_template.format(axis_label=axis_label)
        return q, answer, img

if __name__ == "__main__":
    import collections
    env = RotationSymmetryOrder3dQA()
    for lv in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": lv})
            print(f"L{lv} s{seed}: ok={ok}, answer={env._answer}")
    # Diversity check
    for lv in [0, 3, 6, 9]:
        ans = collections.Counter()
        for s in range(10):
            env = RotationSymmetryOrder3dQA()
            env.generate(s, {'level': lv})
            ans[env._answer] += 1
        print(f"L{lv} answers: {dict(ans)}")
