"""
Layered Stack Count QA environment.

Template: isometric_counting_qa.py.

Goal: count cubes in a specific horizontal layer of an isometric
polycube. Each layer is colored differently so the question can ask
about a specific layer by its color or position. Targets
cube counting .

Difficulty schedule (multi-axis, continuous):
  Axis 1 (primary): n_layers = 2 + level // 3      -> 2..5
  Axis 2           : max_cubes_per_layer = 2 + level  -> 2..11
  Axis 3 (optional): occlusion_in_middle = level >= 3

Output format is constant: 4-option integer MCQ, single letter.
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
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

_LAYER_COLORS = [
    ("blue", "#3498db"),
    ("green", "#27ae60"),
    ("orange", "#e67e22"),
    ("purple", "#8e44ad"),
    ("red", "#e74c3c"),
    ("yellow", "#f1c40f"),
]

class LayeredStackCountQA(StandaloneVisualEnv):
    ENV_NAME = "layered_stack_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_layers": 2 + level // 3,              # 2..5
            "max_per_layer": 2 + level,              # 2..11
            "ambiguous_color": level >= 6,
            "option_gap": max(1, 3 - level // 3),    # 3..1
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_layers"] * 10 + cfg["max_per_layer"]

        for _ in range(20):
            try:
                r = self._try_once(rng, cfg, level)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _try_once(self, rng, cfg, level):
        n_layers = cfg["n_layers"]
        max_cubes = cfg["max_per_layer"]

        # Build layered stack: each layer has (x,y) positions on a grid
        layers = []  # list[list[(x,y)]]
        base_positions = []
        # start with n_layers=2, grow
        grid_extent = max(2, min(4, 1 + max_cubes // 3))
        prev_positions = None
        for k in range(n_layers):
            if k == 0:
                # Bottom layer: dense
                n = rng.randint(max(2, max_cubes - 2), max_cubes)
                positions = []
                candidates = [(i, j) for i in range(grid_extent)
                              for j in range(grid_extent)]
                rng.shuffle(candidates)
                positions = candidates[:n]
                layers.append(positions)
            else:
                # Upper layers: subset of previous layer
                if not layers[-1]:
                    break
                n_prev = len(layers[-1])
                n = rng.randint(max(1, n_prev - 2), max(1, n_prev))
                n = min(n, n_prev)
                positions = random.Random(rng.random()).sample(layers[-1], n) \
                    if n <= n_prev else layers[-1][:]
                # Make it deterministic via local copy
                positions = list(layers[-1])
                rng.shuffle(positions)
                positions = positions[:n]
                layers.append(positions)

        # Build cubes: (x, y, z, layer_idx)
        cubes = []
        for z, layer_positions in enumerate(layers):
            for (x, y) in layer_positions:
                cubes.append((x, y, z))

        # Pick target layer
        target_layer = rng.randint(0, n_layers - 1)
        target_count = len(layers[target_layer])
        if target_count < 1:
            return None

        # Assign colors
        layer_color_names = []
        layer_hex = []
        # Shift palette random offset for diversity
        offset = rng.randint(0, len(_LAYER_COLORS) - 1)
        for k in range(n_layers):
            if cfg["ambiguous_color"] and k > 0 and rng.random() < 0.5:
                # reuse previous color hue with slight variation
                name, hex_c = _LAYER_COLORS[(offset + k - 1) % len(_LAYER_COLORS)]
                layer_color_names.append(name)
                # darker variant
                layer_hex.append(hex_c)
            else:
                name, hex_c = _LAYER_COLORS[(offset + k) % len(_LAYER_COLORS)]
                layer_color_names.append(name)
                layer_hex.append(hex_c)

        target_color_name = layer_color_names[target_layer]
        # Position name RELATIVE to stack height. For a 3-layer stack, only
        # bottom/middle/top are meaningful; don't use "4th-from-bottom" etc.
        # unless the stack actually has that many layers.
        if target_layer == 0:
            layer_pos_name = "bottom"
        elif target_layer == n_layers - 1:
            layer_pos_name = "top"
        elif n_layers % 2 == 1 and target_layer == n_layers // 2:
            layer_pos_name = "middle"
        else:
            # Use from-bottom ordinal with correct suffix
            ordinal_map = {2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th"}
            ordinal = ordinal_map.get(target_layer + 1, f"{target_layer + 1}th")
            layer_pos_name = f"{ordinal}-from-bottom"

        gap = cfg["option_gap"]
        options_pool = [target_count + k for k in (-2 * gap, -gap, gap, 2 * gap)
                        if target_count + k >= 0]
        if target_count not in options_pool:
            options_pool.append(target_count)
        options_pool = list(set(options_pool))
        rng.shuffle(options_pool)
        options = [target_count]
        for v in options_pool:
            if len(options) >= 4:
                break
            if v != target_count and v not in options and v >= 0:
                options.append(v)
        while len(options) < 4:
            v = target_count + rng.randint(1, 3)
            if v not in options:
                options.append(v)
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(target_count))

        opts_text = " ".join(
            f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(options)
        )
        q_stems = [
            f"The isometric structure consists of {n_layers} layers, each colored differently. How many cubes are in the {target_color_name} ({layer_pos_name}) layer?",
            f"Count the cubes in the {target_color_name} layer ({layer_pos_name}) of the {n_layers}-layer structure shown.",
            f"The stacked cube structure has {n_layers} colored layers. How many cubes make up the {target_color_name} ({layer_pos_name}) layer?",
            f"Looking at the isometric view, determine the number of cubes in the {target_color_name} layer (the {layer_pos_name} one).",
        ]
        question = (
            f"{rng.choice(q_stems)} "
            f"Options: {opts_text}. Answer with a single letter."
        )
        if level <= 2:
            question += (
                f" Hint: locate the {target_color_name} layer in the "
                f"isometric view (it's the {layer_pos_name} layer). "
                "Count the cubes visible in that single colored layer. "
                "Other layers (different colors) are not counted."
            )
        image = self._render(cubes, layers, layer_hex)
        return question, correct_letter, image

    def _render(self, cubes, layers, layer_hex):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        cubes_sorted = sorted(cubes, key=lambda c: -(c[0] + c[1] - c[2]))
        for (x, y, z) in cubes_sorted:
            color = layer_hex[z] if z < len(layer_hex) else "#cccccc"
            pts = [
                _iso_project(x, y, z + 1),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x + 1, y + 1, z + 1),
                _iso_project(x, y + 1, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                                 edgecolor="black", lw=1.2))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x, y + 1, z),
                _iso_project(x, y + 1, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True,
                                 facecolor=self._shade(color, 0.75),
                                 edgecolor="black", lw=1.2))
            pts = [
                _iso_project(x, y, z),
                _iso_project(x + 1, y, z),
                _iso_project(x + 1, y, z + 1),
                _iso_project(x, y, z + 1),
            ]
            ax.add_patch(Polygon(pts, closed=True,
                                 facecolor=self._shade(color, 0.5),
                                 edgecolor="black", lw=1.2))

        pts = []
        for (x, y, z) in cubes:
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        pts.append(_iso_project(x + dx, y + dy, z + dz))
        if pts:
            arr = np.array(pts)
            mg = 0.6
            ax.set_xlim(arr[:, 0].min() - mg, arr[:, 0].max() + mg)
            ax.set_ylim(arr[:, 1].min() - mg, arr[:, 1].max() + mg)
        title_pool = ["Layered Cube Structure", "Stacked Cubes",
                      "Isometric Layers", "Cube Stack", "Layer Count"]
        ax.set_title(self._rng.choice(title_pool),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _shade(hex_c, factor):
        """Darken hex color by factor (0..1)."""
        hex_c = hex_c.lstrip("#")
        r = int(hex_c[0:2], 16)
        g = int(hex_c[2:4], 16)
        b = int(hex_c[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = LayeredStackCountQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[layered_stack_count] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/layered_stack_count_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer} | feat={env._primary_complexity_feature}")
