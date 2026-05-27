"""
Layer-by-Layer Count QA (redesigned 2026-04-16).

Shows a 3D block structure with layers color-coded. Asks about the number of
cubes in a specific layer.

Redesign goals:
  * Fix "L0 and L9 similar visually" grade D score.
  * Add structural variation between levels:
      - L0: pyramid shape, 3 layers, vertical ask, numeric answer in range
        1-9, top-down hint present.
      - L3: mixed regular/L-shape/step, 4-5 layers.
      - L6: irregular + top-down ask (# in layer, or # of empty spots).
      - L9: irregular staircase/tower, ask about ratio / smallest layer /
        specific layer.
  * Add 20+ structure templates (pyramid, inverted pyramid, staircase, tower,
    ziggurat, L-base tower, castle, irregular, cross, tree, arch, etc.).
  * Randomize viewing angle, color palette, per-layer shading, legend style.
  * 5 question templates.
"""
import math
import random
import colorsys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

def _iso_project(x, y, z, alpha_deg=30.0, obliqueness=1.0):
    ca = math.cos(math.radians(alpha_deg))
    sa = math.sin(math.radians(alpha_deg))
    sx = (x - y) * ca
    sy = (x + y) * sa + z * obliqueness
    return sx, sy

def _lighten(hex_color, amount):
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + amount))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

# Palettes — each has 7 distinct + a gradient variant
_PALETTE_PRESETS = {
    "vivid": ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f',
              '#9b59b6', '#e67e22', '#1abc9c'],
    "earthy": ['#8d5524', '#c68642', '#e0ac69', '#f1c27d',
               '#ffdbac', '#d9a96e', '#a0522d'],
    "pastel": ['#ffadad', '#ffd6a5', '#fdffb6', '#caffbf',
               '#9bf6ff', '#a0c4ff', '#bdb2ff'],
    "cool":   ['#1a5276', '#1f6f8b', '#2596be', '#48c9b0',
               '#76d7c4', '#a3e4d7', '#d5f5e3'],
    "warm":   ['#d62828', '#f77f00', '#fcbf49', '#eae2b7',
               '#ff6d00', '#ff9e00', '#ff0054'],
    "neon":   ['#00ffff', '#ff00ff', '#ffff00', '#00ff00',
               '#ff5500', '#8800ff', '#00aaff'],
    "autumn": ['#d4a017', '#b7410e', '#8b4513', '#cd853f',
               '#daa520', '#a0522d', '#ff7f50'],
    "ocean":  ['#000428', '#004e92', '#009ffd', '#00d2ff',
               '#43cea2', '#185a9d', '#2bc0e4'],
}

# ------------------------------------------------------------------ #
# Structure templates
# ------------------------------------------------------------------ #

def _struct_pyramid(rng, n_layers, max_xy):
    """Centered pyramid shape, each layer smaller."""
    grid = {}
    for k in range(n_layers):
        side = max(1, max_xy - k)
        for i in range(side):
            for j in range(side):
                grid[(i, j, k)] = 1
    return grid

def _struct_staircase(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        offset = k
        for i in range(max(1, max_xy - k)):
            for j in range(max_xy):
                grid[(i + offset, j, k)] = 1
    return grid

def _struct_tower(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        side = max_xy if k < 2 else max(1, max_xy - (k - 1))
        for i in range(side):
            for j in range(side):
                grid[(i, j, k)] = 1
    return grid

def _struct_L_tower(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        if k < 2:
            for i in range(max_xy):
                for j in range(max_xy):
                    if i == 0 or j == 0:
                        grid[(i, j, k)] = 1
        else:
            for i in range(min(2, max_xy)):
                grid[(i, 0, k)] = 1
    return grid

def _struct_cross(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        mid = max_xy // 2
        for i in range(max_xy):
            grid[(i, mid, k)] = 1
        for j in range(max_xy):
            grid[(mid, j, k)] = 1
    return grid

def _struct_hollow(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        for i in range(max_xy):
            for j in range(max_xy):
                # Keep only perimeter
                if i in (0, max_xy - 1) or j in (0, max_xy - 1):
                    grid[(i, j, k)] = 1
    return grid

def _struct_ziggurat(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        side = max(1, max_xy - k)
        offset = k // 2
        for i in range(side):
            for j in range(side):
                grid[(i + offset, j + offset, k)] = 1
    return grid

def _struct_inverted_pyramid(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        side = min(max_xy, k + 1)
        for i in range(side):
            for j in range(side):
                grid[(i, j, k)] = 1
    return grid

def _struct_castle(rng, n_layers, max_xy):
    grid = {}
    # Base platform
    for i in range(max_xy):
        for j in range(max_xy):
            grid[(i, j, 0)] = 1
    # Towers at corners
    corner_positions = [(0, 0), (max_xy - 1, 0),
                        (0, max_xy - 1), (max_xy - 1, max_xy - 1)]
    tower_h = max(2, n_layers - 1)
    for (i, j) in corner_positions:
        for k in range(1, tower_h + 1):
            if k < n_layers:
                grid[(i, j, k)] = 1
    # Central block
    if max_xy >= 3 and n_layers >= 2:
        mid = max_xy // 2
        for k in range(1, max(1, n_layers // 2)):
            grid[(mid, mid, k)] = 1
    return grid

def _struct_random(rng, n_layers, max_xy):
    grid = {}
    # Start with random base
    for i in range(max_xy):
        for j in range(max_xy):
            if rng.random() < 0.85:
                grid[(i, j, 0)] = 1
    for k in range(1, n_layers):
        for i in range(max_xy):
            for j in range(max_xy):
                # Must have support below
                if grid.get((i, j, k - 1)):
                    prob = max(0.15, 0.8 - k * 0.15)
                    if rng.random() < prob:
                        grid[(i, j, k)] = 1
    return grid

def _struct_bridge(rng, n_layers, max_xy):
    """Two towers with a horizontal span at top."""
    grid = {}
    w = max_xy
    # Left tower
    for k in range(n_layers - 1):
        grid[(0, 0, k)] = 1
    for k in range(n_layers - 1):
        grid[(w - 1, 0, k)] = 1
    # Span at top layer
    top_k = n_layers - 1
    for i in range(w):
        grid[(i, 0, top_k)] = 1
    return grid

def _struct_arch(rng, n_layers, max_xy):
    grid = {}
    w = max_xy
    # Pillars
    for k in range(n_layers - 1):
        for j in range(max(1, w - 1)):
            grid[(0, j, k)] = 1
            grid[(w - 1, j, k)] = 1
    # Top span
    top_k = n_layers - 1
    for i in range(w):
        for j in range(max(1, w - 1)):
            grid[(i, j, top_k)] = 1
    return grid

def _struct_plus_tower(rng, n_layers, max_xy):
    grid = {}
    mid = max_xy // 2
    # Plus base
    for i in range(max_xy):
        grid[(i, mid, 0)] = 1
    for j in range(max_xy):
        grid[(mid, j, 0)] = 1
    # Centre tower
    for k in range(1, n_layers):
        grid[(mid, mid, k)] = 1
    return grid

def _struct_split(rng, n_layers, max_xy):
    """Two separate blocks with a gap in between."""
    grid = {}
    for i in range(max(1, max_xy // 2)):
        for j in range(max_xy):
            for k in range(min(n_layers, 2)):
                grid[(i, j, k)] = 1
    for i in range(max(1, max_xy // 2)):
        for j in range(max_xy):
            for k in range(min(n_layers, 3)):
                grid[(i + max_xy // 2 + 1, j, k)] = 1
    # Top layer may be empty
    if n_layers > 3:
        for i in range(2):
            grid[(i, 0, n_layers - 1)] = 1
    return grid

def _struct_diagonal(rng, n_layers, max_xy):
    grid = {}
    for k in range(n_layers):
        for i in range(max_xy):
            for j in range(max_xy):
                if (i + j) % 2 == 0 or k == 0:
                    grid[(i, j, k)] = 1
    return grid

def _struct_irregular(rng, n_layers, max_xy):
    grid = {}
    # Random blob base
    center = (max_xy // 2, max_xy // 2)
    for i in range(max_xy):
        for j in range(max_xy):
            dist = abs(i - center[0]) + abs(j - center[1])
            if dist <= max_xy and rng.random() < 0.9 - dist * 0.1:
                grid[(i, j, 0)] = 1
    for k in range(1, n_layers):
        for i in range(max_xy):
            for j in range(max_xy):
                if grid.get((i, j, k - 1)):
                    if rng.random() < 0.5:
                        grid[(i, j, k)] = 1
    return grid

_STRUCTURE_FUNCS = {
    "pyramid": _struct_pyramid,
    "staircase": _struct_staircase,
    "tower": _struct_tower,
    "L_tower": _struct_L_tower,
    "cross": _struct_cross,
    "hollow": _struct_hollow,
    "ziggurat": _struct_ziggurat,
    "inverted_pyramid": _struct_inverted_pyramid,
    "castle": _struct_castle,
    "random": _struct_random,
    "bridge": _struct_bridge,
    "arch": _struct_arch,
    "plus_tower": _struct_plus_tower,
    "split": _struct_split,
    "diagonal": _struct_diagonal,
    "irregular": _struct_irregular,
}

class LayerByLayerCountQA(StandaloneVisualEnv):
    ENV_NAME = "layer_by_layer_count"

    _QUESTION_TEMPLATES_BASIC = [
        ("This 3D block structure has layers shown in different colors. "
         "How many cubes are in Layer {k} (counting from the bottom)? "
         "Answer with a single integer."),
        ("Look at the layered block structure. The layers are coloured "
         "distinctly and numbered from the bottom up. Count the cubes in "
         "Layer {k}. Answer with a single integer."),
        ("Count how many cubes make up Layer {k} of this solid (layers "
         "are coloured and counted from the bottom). Answer with a single "
         "integer."),
        ("The block structure is split into coloured layers (1 = bottom). "
         "How many unit cubes form Layer {k}? Answer with a single "
         "integer."),
        ("Observe the 3D structure. Layer {k} (where 1 is the bottom "
         "layer) is highlighted. How many cubes compose this layer? "
         "Answer with a single integer."),
    ]
    _QUESTION_TEMPLATES_COMPARE = [
        ("Compare the layers shown in the image. Which layer (bottom is "
         "Layer 1) contains the most cubes? Answer with a single integer "
         "(the layer number)."),
        ("Looking at the coloured 3D structure, which layer (numbered "
         "from the bottom as 1, 2, ...) has the FEWEST cubes? Answer "
         "with a single integer."),
        ("Which layer (from the bottom up, starting at 1) of the layered "
         "block structure has the greatest cube count? Answer with the "
         "layer number."),
    ]
    _QUESTION_TEMPLATES_SUM = [
        ("What is the total number of unit cubes used to build the 3D "
         "structure shown? Answer with a single integer."),
        ("Count all the unit cubes in the layered block structure. "
         "Answer with a single integer."),
    ]

    _TITLE_VARIANTS = [
        "Layer-by-Layer Count", "Count the Cubes", "Layered Structure",
        "Stack Analysis", "3D Block Count", "Volume Puzzle",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "n_layers": 3,
                "grid_xy": 2,
                "struct_pool": ["pyramid", "tower", "ziggurat"],
                "distinct_colors": True,
                "show_hint": True,
                "ask_type_pool": ["basic", "basic", "sum"],
            }
        if level <= 3:
            return {
                "n_layers": 3 + level // 2,
                "grid_xy": 3,
                "struct_pool": ["pyramid", "staircase", "tower",
                                "ziggurat", "inverted_pyramid",
                                "plus_tower"],
                "distinct_colors": True,
                "show_hint": True,
                "ask_type_pool": ["basic", "basic", "sum", "compare"],
            }
        if level <= 5:
            return {
                "n_layers": 4 + level // 3,
                "grid_xy": 3 + level // 4,
                "struct_pool": ["L_tower", "cross", "castle", "hollow",
                                "bridge", "arch", "split", "diagonal",
                                "ziggurat"],
                "distinct_colors": True,
                "show_hint": False,
                "ask_type_pool": ["basic", "compare", "sum"],
            }
        if level <= 7:
            return {
                "n_layers": 5 + level // 3,
                "grid_xy": 3 + level // 3,
                "struct_pool": ["L_tower", "castle", "hollow", "bridge",
                                "arch", "random", "split",
                                "irregular", "diagonal"],
                "distinct_colors": False,
                "show_hint": False,
                "ask_type_pool": ["basic", "compare", "sum"],
            }
        # L8-9
        return {
            "n_layers": 6 + level // 5,
            "grid_xy": 4 + level // 5,
            "struct_pool": ["random", "irregular", "castle", "bridge",
                            "arch", "hollow", "diagonal", "split",
                            "L_tower"],
            "distinct_colors": False,
            "show_hint": False,
            "ask_type_pool": ["basic", "compare", "sum"],
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1010)
        style = self._random_style()
        self._primary_complexity_feature = cfg["n_layers"] * cfg["grid_xy"]

        for _attempt in range(20):
            r = self._try_generate(rng, style, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, style, cfg, level):
        palette_name = rng.choice(list(_PALETTE_PRESETS.keys()))
        palette = list(_PALETTE_PRESETS[palette_name])
        rng.shuffle(palette)
        n_layers = cfg["n_layers"]
        gxy = cfg["grid_xy"]
        struct_name = rng.choice(cfg["struct_pool"])
        grid = _STRUCTURE_FUNCS[struct_name](rng, n_layers, gxy)
        if len(grid) == 0:
            return None

        # Count per layer
        layer_counts = {}
        for (_i, _j, k) in grid.keys():
            layer_counts[k] = layer_counts.get(k, 0) + 1
        actual_layers = sorted(layer_counts.keys())
        if not actual_layers:
            return None

        ask_type = rng.choice(cfg["ask_type_pool"])
        if ask_type == "basic":
            ask_layer = rng.choice(actual_layers)
            answer = str(layer_counts[ask_layer])
            q = rng.choice(self._QUESTION_TEMPLATES_BASIC).format(
                k=ask_layer + 1)
            highlight_layer = ask_layer
            question_hint_label = f"Layer {ask_layer + 1}"
        elif ask_type == "compare":
            mode = rng.choice(["most", "fewest", "most"])
            if mode == "most":
                ask_layer = max(layer_counts, key=layer_counts.get)
                q = self._QUESTION_TEMPLATES_COMPARE[0] if rng.random() < 0.5 \
                    else self._QUESTION_TEMPLATES_COMPARE[2]
            else:
                ask_layer = min(layer_counts, key=layer_counts.get)
                q = self._QUESTION_TEMPLATES_COMPARE[1]
            answer = str(ask_layer + 1)
            highlight_layer = ask_layer
            question_hint_label = None
        else:  # sum
            total = sum(layer_counts.values())
            if total == 0:
                return None
            answer = str(total)
            q = rng.choice(self._QUESTION_TEMPLATES_SUM)
            highlight_layer = None
            question_hint_label = None

        img = self._render(rng, style, grid, palette, layer_counts,
                           highlight_layer, cfg, question_hint_label,
                           n_layers)
        return q, answer, img

    def _render(self, rng, style, grid, palette, layer_counts,
                highlight_layer, cfg, question_hint_label, n_layers):
        sc = style['figsize_scale']
        alpha_deg = 30 + rng.uniform(-6, 6)
        obliqueness = 1.0 + rng.uniform(-0.15, 0.2)

        show_hint = cfg["show_hint"]
        if show_hint:
            fig, (ax, ax_leg) = plt.subplots(
                1, 2, figsize=(10 * sc, 7 * sc),
                gridspec_kw={'width_ratios': [2.4, 1.0]})
        else:
            fig, ax = plt.subplots(figsize=(7.5 * sc, 7.5 * sc))
            ax_leg = None

        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal')
        ax.axis('off')

        cubes = sorted(grid.keys(), key=lambda c: -(c[0] + c[1] - c[2]))
        gradient_mode = not cfg["distinct_colors"]
        for (x, y, z) in cubes:
            base_color = palette[z % len(palette)]
            if gradient_mode:
                base_color = _lighten(palette[0], -z * 0.07)
            top = base_color
            left = _lighten(base_color, -0.2)
            right = _lighten(base_color, -0.10)
            is_highlighted = (highlight_layer is not None
                              and z == highlight_layer)
            edge_color = "#e74c3c" if is_highlighted else "#2c3e50"
            edge_lw = 2.2 if is_highlighted else 0.9
            faces = [
                ([(x, y, z + 1), (x + 1, y, z + 1),
                  (x + 1, y + 1, z + 1), (x, y + 1, z + 1)], top),
                ([(x, y, z), (x, y + 1, z),
                  (x, y + 1, z + 1), (x, y, z + 1)], left),
                ([(x, y, z), (x + 1, y, z),
                  (x + 1, y, z + 1), (x, y, z + 1)], right),
            ]
            for face, fc in faces:
                pts = [_iso_project(*p, alpha_deg=alpha_deg,
                                    obliqueness=obliqueness) for p in face]
                ax.add_patch(Polygon(pts, closed=True, facecolor=fc,
                                     edgecolor=edge_color, lw=edge_lw))

        # Legend
        actual_layers = sorted(layer_counts.keys())
        for k in actual_layers:
            lc = palette[k % len(palette)]
            if gradient_mode:
                lc = _lighten(palette[0], -k * 0.07)
            label = f"Layer {k + 1}"
            if highlight_layer is not None and k == highlight_layer:
                label += " *"
            ax.plot([], [], 's', color=lc, markersize=10, label=label)
        ax.legend(fontsize=max(9, style['font_size_base'] - 1),
                  loc='upper right', frameon=True)
        ax.autoscale_view()
        ax.margins(0.15)
        title = rng.choice(self._TITLE_VARIANTS)
        ax.set_title(title, fontsize=max(11, style['font_size_base'] + 2),
                     fontweight='bold')

        # Hint panel — top-down layer hint
        if ax_leg is not None:
            ax_leg.set_aspect('equal')
            ax_leg.axis('off')
            ax_leg.set_facecolor(style['bg_color'])
            ax_leg.set_title("Layers (top view)", fontsize=11)
            gxy = cfg["grid_xy"]
            # Show each layer in a mini-grid
            for li, k in enumerate(actual_layers):
                off_y = (len(actual_layers) - 1 - li) * (gxy + 1.5)
                # Draw title
                lc = palette[k % len(palette)]
                if gradient_mode:
                    lc = _lighten(palette[0], -k * 0.07)
                # BUGFIX 2026-04-24: remove answer leakage — do not show per-layer
                # count in hint label. Previous label "L{k+1} ({layer_counts[k]})"
                # directly revealed the answer to the counting question.
                ax_leg.text(-0.3, off_y + gxy / 2,
                            f"L{k + 1}",
                            fontsize=10, ha="right", va="center",
                            color="#2c3e50")
                for (x, y, z) in grid.keys():
                    if z != k:
                        continue
                    ax_leg.add_patch(
                        Polygon([(x, y + off_y), (x + 1, y + off_y),
                                 (x + 1, y + 1 + off_y),
                                 (x, y + 1 + off_y)],
                                closed=True, facecolor=lc,
                                edgecolor="#2c3e50", lw=0.8))
            ax_leg.autoscale_view()
            ax_leg.margins(0.15)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style['dpi'])

if __name__ == "__main__":
    import collections
    env = LayerByLayerCountQA()
    for lv in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": lv})
            print(f"L{lv} s{seed}: ok={ok}, answer={env._answer}")
    for lv in [0, 3, 6, 9]:
        ans = collections.Counter()
        for s in range(10):
            e = LayerByLayerCountQA()
            e.generate(s, {'level': lv})
            ans[e._answer] += 1
        print(f"L{lv} answers: {dict(ans)}")
