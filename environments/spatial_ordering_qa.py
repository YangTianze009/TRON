"""
Spatial Ordering QA.

2D scatter of labeled objects with a reference point R. Asks to order
objects by distance from R, or identify the Nth closest. MCQ.

Difficulty axes:
  A) n_objects: 5..8
  B) distance_ambiguity + show_grid
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class SpatialOrderingQA(StandaloneVisualEnv):
    ENV_NAME = "spatial_ordering"

    def _level_config(self, level):
        return {
            'n_objects': 5 + level // 3,
            'show_grid': level <= 4,
            'canvas': 10,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1027)
        style = self._random_style()

        n = cfg['n_objects']
        canvas = cfg['canvas']
        labels = [chr(65 + i) for i in range(n)]

        # Try several placements to avoid near-ties between the asked rank
        # and the next rank (previously, differences as small as 0.2 units
        # were produced, making the answer visually ambiguous).
        best_attempt = None
        for attempt in range(30):
            ref = (canvas / 2 + rng.uniform(-1, 1),
                   canvas / 2 + rng.uniform(-1, 1))
            objects = {}
            for label in labels:
                for _a in range(50):
                    x = rng.uniform(0.5, canvas - 0.5)
                    y = rng.uniform(0.5, canvas - 0.5)
                    dist = math.sqrt((x - ref[0])**2 + (y - ref[1])**2)
                    # Require separation from existing objects too, so
                    # markers do not overlap at high densities.
                    if dist > 0.5 and all(
                            math.hypot(x - ox, y - oy) >= 1.0
                            for (ox, oy, _) in objects.values()):
                        objects[label] = (x, y, dist)
                        break
            if len(objects) < n:
                continue
            sorted_objs = sorted(objects.items(), key=lambda i: i[1][2])
            ask_rank = rng.randint(1, min(3, n))
            # Ambiguity filter: require >= 1.0 unit gap between rank (k) and
            # rank (k+1) so the correct answer is visually unambiguous.
            if ask_rank < len(sorted_objs):
                gap = sorted_objs[ask_rank][1][2] - sorted_objs[ask_rank - 1][1][2]
            else:
                gap = 999
            if gap >= 1.0:
                best_attempt = (ref, objects, sorted_objs, ask_rank)
                break
            # Keep the largest-gap attempt as a fallback.
            if best_attempt is None or gap > best_attempt[4]:
                best_attempt = (ref, objects, sorted_objs, ask_rank, gap)

        if best_attempt is None:
            return None
        ref, objects, sorted_objs, ask_rank = best_attempt[:4]
        target_label = sorted_objs[ask_rank - 1][0]

        # MCQ options
        options = [target_label]
        other_labels = [l for l in labels if l != target_label]
        rng.shuffle(other_labels)
        options.extend(other_labels[:3])
        rng.shuffle(options)
        correct_idx = options.index(target_label)
        correct = "ABCD"[correct_idx]

        rank_word = {1: "closest", 2: "2nd closest", 3: "3rd closest"}.get(ask_rank, f"{ask_rank}th closest")

        # Draw
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(7*sc, 7*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_xlim(0, canvas); ax.set_ylim(0, canvas)
        ax.set_aspect('equal')

        # Reference point
        ax.plot(*ref, '*', color='red', markersize=20, markeredgecolor='#333')
        ax.text(ref[0] + 0.3, ref[1] + 0.3, 'R', fontsize=12, color='red', fontweight='bold')

        # Objects
        for label, (x, y, d) in objects.items():
            color = style['palette'][labels.index(label) % len(style['palette'])]
            ax.plot(x, y, 'o', color=color, markersize=14, markeredgecolor='#333', markeredgewidth=1.5)
            ax.text(x + 0.25, y + 0.25, label, fontsize=style['font_size_base']+1, fontweight='bold')

        if cfg['show_grid']:
            ax.grid(True, alpha=0.2, linestyle='--')
        ax.axis('off')
        ax.set_title("Distance Ordering", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        opt_str = "  ".join(f"({chr(65+i)}) {options[i]}" for i in range(4))
        q = (f"The red star R is the reference point. "
             f"Which labeled object is the {rank_word} to R?\n{opt_str}")
        return q, correct, img

if __name__ == "__main__":
    env = SpatialOrderingQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
