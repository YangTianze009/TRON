"""
Shadow Overlap QA.

Two 3D objects on a ground plane with a single light source. Their shadows
overlap. Asks for the area of the overlap region.

Difficulty axes:
  A) object_complexity: 2..4 cubes per object
  B) light_angle: near-vertical -> oblique
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# 16 phrasings asking for overlap area. The purple/region description is
# kept in body so we get 16 distinct normalized templates.
_Q_TEMPLATES = [
    "Two block structures cast shadows on the ground (colored regions). The purple area shows where both shadows overlap. What is the area of the overlap region in grid squares? Answer with a single integer.",
    "Examine the two objects' shadows. The purple region is the overlap. How many grid squares does the overlap cover?",
    "The figure shows two shadows and their purple overlap. Give the overlap area in grid squares as a single integer.",
    "In the image, two solids cast shadows whose overlap is shown in purple. Count the squares in that overlap.",
    "Count the grid squares in the purple overlap region of the two shadows. Answer as an integer.",
    "Two shapes cast shadows; their common area is purple. Report the overlap area (grid squares).",
    "Report the number of grid squares inside the purple intersection of the two shadows.",
    "How large is the overlap of the two shadows? Give the answer in grid squares.",
    "The intersection of the two shadows is shaded purple. What is its area in grid squares?",
    "Find the area (grid squares) of the shadow overlap shown in purple. Integer answer.",
    "Determine the number of grid cells that are part of BOTH shadows. Integer answer.",
    "Compute the area of the purple overlap between the two cast shadows. Grid squares only.",
    "The purple cells lie in both shadow regions. How many such cells are there?",
    "Study the shadow diagram. What is the area (squares) of the region where both shadows overlap?",
    "Looking at the image, state the size of the purple overlap region in grid squares.",
    "After inspecting the shadows, give the overlap area in grid squares (integer).",
]

def _shadow_cells(cubes, light_dx, light_dy):
    """Project cubes to ground shadow cells given horizontal light offset per z-unit."""
    cells = set()
    for (cx, cy, cz) in cubes:
        for dx in range(2):
            for dy in range(2):
                sx = int(math.floor(cx + dx + light_dx * (cz + 1)))
                sy = int(math.floor(cy + dy + light_dy * (cz + 1)))
                cells.add((sx, sy))
    return cells

class ShadowOverlapQA(StandaloneVisualEnv):
    ENV_NAME = "shadow_overlap"

    def _level_config(self, level):
        # Keep object separation close enough that shadows usually touch/overlap.
        # L0/L1: simple — 1 cube each, large light offset so shadows clearly
        # overlap (and overlap area is 1-3 squares, easy to count).
        if level <= 1:
            return {
                'cubes_per_obj': 1,
                'light_dx': 0.8,
                'light_dy': 0.8,
                'separation': 1,
                'force_overlap': True,
            }
        return {
            'cubes_per_obj': 2 + (level - 1) // 2,
            'light_dx': 0.4 + 0.06 * level,
            'light_dy': 0.5 + 0.05 * level,
            'separation': max(0, 2 - level // 4),
            'force_overlap': False,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1021)
        style = self._random_style()

        n = cfg['cubes_per_obj']
        sep = cfg['separation']

        # Build two objects
        def build_obj(start_x, start_y):
            cubes = [(start_x, start_y, 0)]
            dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1)]
            for _ in range(n - 1):
                for _a in range(30):
                    base = rng.choice(cubes)
                    d = rng.choice(dirs)
                    nc = (base[0]+d[0], base[1]+d[1], base[2]+d[2])
                    if nc not in cubes and nc[2] >= 0:
                        cubes.append(nc); break
            return cubes

        obj1 = build_obj(0, 0)
        # Put obj2 close enough that shadows can overlap; exact offset randomized per seed.
        if cfg.get('force_overlap'):
            # L0/L1: place obj2 very near obj1 so shadows are guaranteed to overlap
            # by 1+ grid squares.
            obj2 = build_obj(sep, rng.choice([-1, 0, 1]))
            shadow1 = _shadow_cells(obj1, cfg['light_dx'], cfg['light_dy'])
            shadow2 = _shadow_cells(obj2, cfg['light_dx'], cfg['light_dy'])
            overlap = shadow1 & shadow2
            # If still no overlap, retry with different placements (max 20 tries)
            attempts = 0
            while len(overlap) == 0 and attempts < 20:
                attempts += 1
                obj2 = build_obj(sep + rng.choice([-1, 0]), rng.choice([-1, 0, 1]))
                shadow1 = _shadow_cells(obj1, cfg['light_dx'], cfg['light_dy'])
                shadow2 = _shadow_cells(obj2, cfg['light_dx'], cfg['light_dy'])
                overlap = shadow1 & shadow2
        else:
            obj2 = build_obj(sep + rng.randint(0, 2), rng.randint(-1, 2))
            shadow1 = _shadow_cells(obj1, cfg['light_dx'], cfg['light_dy'])
            shadow2 = _shadow_cells(obj2, cfg['light_dx'], cfg['light_dy'])
            overlap = shadow1 & shadow2
        answer = str(len(overlap))

        # Draw top-down view — clean / textbook / sketch mode mix
        sc = style['figsize_scale']
        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            grid_col = tbp["aux_color"]
            title_kw = {"fontfamily": tbp["font_family"], "color": tbp["line_color"]}
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            grid_col = "#bbbbbb"
            title_kw = {}
            dpi = style['dpi']
        else:
            bg = style['bg_color']
            grid_col = "#dddddd"
            title_kw = {}
            dpi = style['dpi']

        fig, ax = plt.subplots(figsize=(8*sc, 8*sc))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.set_aspect('equal'); ax.axis('off')

        # Draw shadows (use plain Rectangle to avoid rounded-corner blur)
        all_cells = shadow1 | shadow2
        for cell in all_cells:
            if cell in overlap:
                color, alpha = '#9b59b6', 0.7  # purple overlap (high alpha)
            elif cell in shadow1:
                color, alpha = 'red', 0.25
            else:
                color, alpha = 'blue', 0.25
            rect = mpatches.Rectangle((cell[0], cell[1]), 1, 1,
                facecolor=color, alpha=alpha, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)

        # Draw objects (top-down) — outline only with thick border so shadows
        # underneath remain visible. Use hatch to indicate solid object.
        for (cx, cy, cz) in obj1:
            rect = mpatches.Rectangle((cx, cy), 1, 1,
                facecolor='none', edgecolor=style['palette'][0],
                linewidth=2.5, hatch='//')
            ax.add_patch(rect)
        for (cx, cy, cz) in obj2:
            rect = mpatches.Rectangle((cx, cy), 1, 1,
                facecolor='none', edgecolor=style['palette'][1],
                linewidth=2.5, hatch='\\\\')
            ax.add_patch(rect)

        # Grid
        all_pts = list(all_cells) + [(c[0], c[1]) for c in obj1 + obj2]
        if all_pts:
            min_x = min(p[0] for p in all_pts) - 1
            max_x = max(p[0] for p in all_pts) + 2
            min_y = min(p[1] for p in all_pts) - 1
            max_y = max(p[1] for p in all_pts) + 2
            ax.set_xlim(min_x, max_x); ax.set_ylim(min_y, max_y)
            for x in range(int(min_x), int(max_x)+1):
                ax.axvline(x, color=grid_col, linewidth=0.3)
            for y in range(int(min_y), int(max_y)+1):
                ax.axhline(y, color=grid_col, linewidth=0.3)

        # Legend
        ax.plot([], [], 's', color='red', alpha=0.3, markersize=10, label='Shadow 1')
        ax.plot([], [], 's', color='blue', alpha=0.3, markersize=10, label='Shadow 2')
        ax.plot([], [], 's', color='#9b59b6', alpha=0.5, markersize=10, label='Overlap')
        ax.legend(fontsize=style['font_size_base']-1, loc='upper right')

        ax.set_title("Shadow Overlap Area",
                     fontsize=style['font_size_base']+3, fontweight='bold',
                     **title_kw)
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=dpi)

        sidx = (self.seed or 0) % len(_Q_TEMPLATES)
        q = _Q_TEMPLATES[sidx]
        q += "\n\nReply with a single integer inside <answer>...</answer>. Example: <answer>3</answer>"
        if level <= 1:
            q += (
                "\n\nHint: count cells where BOTH shadow regions overlap "
                "(visible as the darker/blended color where the two shadow "
                "polygons cross). Each unit grid square counts as 1."
            )
        return q, answer, img

if __name__ == "__main__":
    env = ShadowOverlapQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
