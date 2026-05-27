"""
Object Count With Occlusion QA.

2D scene with multiple objects at varying depths. Objects in front partially
occlude objects behind. Asks to count objects including partially hidden ones.

Difficulty axes:
  A) n_objects: 4..13
  B) occlusion_severity + shape_diversity
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon, Circle
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class ObjectCountWithOcclusionQA(StandaloneVisualEnv):
    ENV_NAME = "object_count_with_occlusion"

    def _level_config(self, level):
        return {
            'n_objects_base': 4 + level,
            'n_types': 1 + level // 3,
            'canvas': 10,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1014)
        style = self._random_style()

        # Randomize n_objects in a window around the base so answer varies across seeds.
        n = cfg['n_objects_base'] + rng.randint(-1, 2)
        n = max(3, n)
        canvas = cfg['canvas']
        shape_types = ['circle', 'square', 'triangle', 'star'][:cfg['n_types']]

        # Size schedule: shrink sizes as levels grow so occlusion stays
        # readable (each object remains partially visible).
        size_hi = max(0.9, 1.8 - 0.09 * level)
        size_lo = max(0.55, size_hi - 0.9)

        objects = []
        # Limit overlap: disallow placement where centers are very close
        # (this enforces each object remains partially visible at level 9).
        min_center_gap = max(0.8, 1.6 - 0.08 * level)
        attempts_per_obj = 60
        for i in range(n):
            st = rng.choice(shape_types)
            placed = False
            size = rng.uniform(size_lo, size_hi)
            for _ in range(attempts_per_obj):
                cx = rng.uniform(1.5, canvas - 1.5)
                cy = rng.uniform(1.5, canvas - 1.5)
                ok_place = True
                for o in objects:
                    if math.hypot(cx - o['cx'], cy - o['cy']) < min_center_gap:
                        ok_place = False
                        break
                if ok_place:
                    placed = True
                    break
            # if we couldn't place without collision, allow closer placement as fallback
            if not placed:
                cx = rng.uniform(1.5, canvas - 1.5)
                cy = rng.uniform(1.5, canvas - 1.5)
            color = rng.choice(style['palette'])
            objects.append({'type': st, 'cx': cx, 'cy': cy, 'size': size,
                            'color': color, 'z': i})  # z = depth/order

        # Pick which type to ask about
        if cfg['n_types'] == 1:
            ask_type = shape_types[0]
        else:
            ask_type = rng.choice(shape_types)
        ask_count = sum(1 for o in objects if o['type'] == ask_type)
        if ask_count == 0:
            objects[0]['type'] = ask_type
            ask_count = 1

        answer = str(ask_count)
        type_plural = ask_type + ('s' if ask_type != 'star' else 's')

        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(7*sc, 7*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_xlim(0, canvas); ax.set_ylim(0, canvas)
        ax.set_aspect('equal'); ax.axis('off')

        # Draw in z-order (back to front)
        for obj in sorted(objects, key=lambda o: o['z']):
            if obj['type'] == 'circle':
                p = Circle((obj['cx'], obj['cy']), obj['size'],
                           facecolor=obj['color'], edgecolor='#333', linewidth=1.5)
            elif obj['type'] == 'square':
                p = mpatches.FancyBboxPatch(
                    (obj['cx']-obj['size'], obj['cy']-obj['size']),
                    obj['size']*2, obj['size']*2,
                    facecolor=obj['color'], edgecolor='#333', linewidth=1.5)
            elif obj['type'] == 'triangle':
                p = RegularPolygon((obj['cx'], obj['cy']), 3, radius=obj['size'],
                                   facecolor=obj['color'], edgecolor='#333', linewidth=1.5)
            else:  # star
                p = RegularPolygon((obj['cx'], obj['cy']), 5, radius=obj['size'],
                                   facecolor=obj['color'], edgecolor='#333', linewidth=1.5)
            ax.add_patch(p)

        ax.set_title("Count Objects", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q = (f"The image shows overlapping objects at different depths. "
             f"How many {type_plural} are in the scene, including partially hidden ones? "
             f"Answer with a single integer.")
        return q, answer, img

if __name__ == "__main__":
    env = ObjectCountWithOcclusionQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
