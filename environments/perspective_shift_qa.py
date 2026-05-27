"""
Perspective Shift QA.

Top-down view of a room with labeled objects and a person at P1/P2.
Asks what object is on their left after moving. MCQ.

Difficulty axes:
  A) n_objects: 3..7
  B) rotation_angle/135
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class PerspectiveShiftQA(StandaloneVisualEnv):
    ENV_NAME = "perspective_shift"

    def _level_config(self, level):
        return {
            'n_objects': 3 + level // 2,
            'rotation': 180 if level <= 2 else (90 if level <= 5 else 45),
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1026)
        n_objects = 3 + level // 2
        # Ensure P2 faces clear cardinal direction at easy levels so the
        # arrow marker is unambiguous. Full set of 8 compass directions at
        # higher levels.
        rot_options = [90, 180, 270] if level <= 2 else ([90, 180, 270] if level <= 5 else [45, 90, 135, 180, 225, 270, 315])
        style = self._random_style()

        room_size = 8
        # Place person at P1 facing north (up)
        p1 = (room_size / 2, room_size / 4)
        p1_facing = 90  # degrees, 0=east, 90=north

        # P2 position and facing
        rotation = rng.choice(rot_options)
        p2_facing = (p1_facing + rotation) % 360
        p2 = (room_size / 2 + rng.uniform(-2, 2), room_size / 2 + rng.uniform(-1, 1))

        # Place objects with minimum spacing from P2 and from each other,
        # so "LEFT" has an unambiguous winner (score margin > 0.8).
        object_names = ['Table', 'Chair', 'Lamp', 'Plant', 'Shelf', 'Desk', 'Sofa'][:n_objects]
        left_dir = math.radians(p2_facing + 90)
        for _retry in range(40):
            objects = {}
            for name in object_names:
                placed = False
                for _a in range(80):
                    ox = rng.uniform(0.5, room_size - 0.5)
                    oy = rng.uniform(0.5, room_size - 0.5)
                    if abs(ox - p2[0]) + abs(oy - p2[1]) < 2.0:
                        continue
                    too_close_other = False
                    for _, (qx, qy) in objects.items():
                        if math.hypot(ox - qx, oy - qy) < 1.2:
                            too_close_other = True
                            break
                    if too_close_other:
                        continue
                    objects[name] = (ox, oy)
                    placed = True
                    break
                if not placed:
                    objects[name] = (rng.uniform(0.5, room_size-0.5),
                                     rng.uniform(0.5, room_size-0.5))
            # Score each object for LEFT at P2
            scored = []
            for name, (ox, oy) in objects.items():
                dx = ox - p2[0]; dy = oy - p2[1]
                score = dx * math.cos(left_dir) + dy * math.sin(left_dir)
                scored.append((score, name))
            scored.sort(reverse=True)
            # Require a positive winner with clear margin
            if len(scored) >= 2 and scored[0][0] > 0.8 and (scored[0][0] - scored[1][0]) > 0.9:
                break
        best_name = scored[0][1]

        # MCQ
        options = list(objects.keys())
        rng.shuffle(options)
        options = options[:4]
        # Pad if fewer than 4 options
        fillers = ['Window', 'Door', 'Rug', 'Cabinet']
        while len(options) < 4:
            filler = fillers.pop(0)
            if filler not in options:
                options.append(filler)
        if best_name not in options:
            options[0] = best_name
        rng.shuffle(options)
        correct_idx = options.index(best_name)
        correct = "ABCD"[correct_idx]

        # Draw
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(7*sc, 7*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor('#f5f5f0')
        ax.set_xlim(-0.5, room_size + 0.5)
        ax.set_ylim(-0.5, room_size + 0.5)
        ax.set_aspect('equal')

        # Room border
        ax.add_patch(mpatches.FancyBboxPatch((0, 0), room_size, room_size,
            facecolor='#f5f5f0', edgecolor='#333', linewidth=3, fill=True))

        # Objects
        for name, (ox, oy) in objects.items():
            color = style['palette'][list(objects.keys()).index(name) % len(style['palette'])]
            ax.plot(ox, oy, 's', color=color, markersize=15, markeredgecolor='#333')
            ax.text(ox + 0.2, oy + 0.2, name, fontsize=8, fontweight='bold')

        # Person at P1 (facing north / up) — circle + single bold arrow up.
        ax.plot(*p1, 'o', color='green', markersize=16, markeredgecolor='#333', markeredgewidth=1.5, zorder=5)
        p1_arrow_dx = 0.9 * math.cos(math.radians(p1_facing))
        p1_arrow_dy = 0.9 * math.sin(math.radians(p1_facing))
        ax.annotate('', xy=(p1[0]+p1_arrow_dx, p1[1]+p1_arrow_dy), xytext=p1,
                    arrowprops=dict(arrowstyle='-|>', color='green', lw=3,
                                    mutation_scale=20), zorder=6)
        ax.text(p1[0]+0.35, p1[1]-0.35, 'P1', fontsize=11, color='green', fontweight='bold')

        # Person at P2 — circle (not triangle, which is misleading) + single
        # bold arrow showing actual facing. Also dashed 'L' marker showing
        # the LEFT direction as a visual hint (low levels).
        ax.plot(*p2, 'o', color='blue', markersize=16, markeredgecolor='#333', markeredgewidth=1.5, zorder=5)
        arrow_dx = 0.9 * math.cos(math.radians(p2_facing))
        arrow_dy = 0.9 * math.sin(math.radians(p2_facing))
        ax.annotate('', xy=(p2[0]+arrow_dx, p2[1]+arrow_dy), xytext=p2,
                    arrowprops=dict(arrowstyle='-|>', color='blue', lw=3,
                                    mutation_scale=20), zorder=6)
        ax.text(p2[0]+0.35, p2[1]-0.35, 'P2 (facing arrow)', fontsize=10, color='blue', fontweight='bold')

        ax.axis('off')
        ax.set_title("Perspective Shift", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        opt_str = "  ".join(f"({chr(65+i)}) {options[i]}" for i in range(4))
        q = (f"A person moves from P1 (green, facing up) to P2 (blue, facing the arrow direction). "
             f"From their new position P2, which object is on their LEFT?\n{opt_str}")
        if level <= 2:
            q += (
                "\nHint: at P2, the person faces the arrow. Their LEFT is "
                "perpendicular to that direction, 90° counterclockwise from "
                "facing. Compute the left direction vector, then for each "
                "object, check whether it's in the half-plane to the left "
                "of P2 along that vector."
            )
        return q, correct, img

if __name__ == "__main__":
    env = PerspectiveShiftQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
