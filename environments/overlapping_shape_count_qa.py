"""
Overlapping Shape Count QA.

Multiple semi-transparent geometric shapes overlapping on a canvas.
Asks to count shapes of a specific type or total shapes.

Difficulty axes:
  A) n_shapes: 3..12
  B) overlap_density + shape_diversity
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class OverlappingShapeCountQA(StandaloneVisualEnv):
    ENV_NAME = "overlapping_shape_count"

    def _level_config(self, level):
        return {
            'n_shapes_base': 3 + level,
            'alpha': max(0.3, 0.6 - level * 0.03),
            'n_types': 1 + level // 3,
            'canvas': 8 + level,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1011)
        style = self._random_style()

        shape_types = ['circle', 'rectangle', 'triangle', 'pentagon'][:cfg['n_types']]
        # Randomize shape count in a window around the base so answers vary.
        n = cfg['n_shapes_base'] + rng.randint(-1, 2)
        n = max(2, n)
        canvas_size = cfg['canvas']
        alpha = cfg['alpha']

        shapes = []
        for _ in range(n):
            st = rng.choice(shape_types)
            cx = rng.uniform(1.5, canvas_size - 1.5)
            cy = rng.uniform(1.5, canvas_size - 1.5)
            size = rng.uniform(0.8, 2.0)
            color = rng.choice(style['palette'])
            shapes.append({'type': st, 'cx': cx, 'cy': cy, 'size': size, 'color': color})

        # Pick question type
        if cfg['n_types'] == 1:
            ask_type = shape_types[0]
            ask_count = n
            q_phrase = f"How many {ask_type}s are in the image?"
        else:
            ask_type = rng.choice(shape_types)
            ask_count = sum(1 for s in shapes if s['type'] == ask_type)
            # Ensure non-zero
            if ask_count == 0:
                shapes[0]['type'] = ask_type
                ask_count = 1
            q_phrase = f"How many {ask_type}s are in the image (including partially hidden ones)?"

        answer = str(ask_count)

        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(7*sc, 7*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_xlim(0, canvas_size); ax.set_ylim(0, canvas_size)
        ax.set_aspect('equal'); ax.axis('off')

        for s in shapes:
            if s['type'] == 'circle':
                p = plt.Circle((s['cx'], s['cy']), s['size'], facecolor=s['color'],
                               edgecolor='#333', alpha=alpha, linewidth=1.5)
            elif s['type'] == 'rectangle':
                p = mpatches.FancyBboxPatch((s['cx']-s['size'], s['cy']-s['size']*0.7),
                                            s['size']*2, s['size']*1.4,
                                            facecolor=s['color'], edgecolor='#333',
                                            alpha=alpha, linewidth=1.5)
            elif s['type'] == 'triangle':
                p = RegularPolygon((s['cx'], s['cy']), 3, radius=s['size'],
                                   facecolor=s['color'], edgecolor='#333',
                                   alpha=alpha, linewidth=1.5)
            else:  # pentagon
                p = RegularPolygon((s['cx'], s['cy']), 5, radius=s['size'],
                                   facecolor=s['color'], edgecolor='#333',
                                   alpha=alpha, linewidth=1.5)
            ax.add_patch(p)

        ax.set_title("Overlapping Shapes", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q = f"{q_phrase} Answer with a single integer."
        return q, answer, img

if __name__ == "__main__":
    env = OverlappingShapeCountQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
