"""
Net Validity Advanced QA environment.

2026-05-05 R5 P3: REWRITTEN to align with the Understanding of Solid Figures
benchmark style. Image shows a cube with ONE face shaded plus FOUR
candidate net unfoldings (each labeled A/B/C/D); the model must pick the
candidate net whose shaded cell folds onto the correct shaded cube face.

Pure single-letter MCQ A-D / A-E with "No correct answer" trap.

Per-level design:
  L0/L1 — 4-MCQ. Plus-cross net only. Trivial: shade a face whose net cell
    is the visually-obvious one in the unfolding. No trap.
  L2/L3 — 5-MCQ with "E. No correct answer" (10% trap). Adds T-net.
  L4-L6 — 5-MCQ. 15% trap. Adds 1-4-1 row net.
  L7 — 5-MCQ tighter distractors (shaded cell is opposite-face in distractor),
    25% trap.
  L8/L9 — 5-MCQ + 40% trap. All 4 valid nets + tightest distractors.

Question phrasing matches standard textbook style:
  Q14: "As shown in the diagram, color the square above the square labeled
        A on the cube, and then unfold the cube. Which of the following
        expanded views is correct?"
  Q22: "As shown in the figure, color the square to the right above the
        square marked A on the cube. Then, when the cube is unfolded,
        which of the following unfolded diagrams is correct?"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# --------------------------------------------------------------------- #
# Cube net topology (mirrors dice_net_qa)
# --------------------------------------------------------------------- #

# Net layouts — each is list of (col, row) cells.
_NETS = [
    ("plus_cross",
     [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2)]),
    ("t_shape",
     [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (1, 3)]),
    ("row_141",
     [(0, 1), (1, 1), (2, 1), (3, 1), (1, 0), (2, 2)]),
    ("zigzag",
     [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]),
]

# Cube face IDs:
#   0 = top, 1 = bottom, 2 = front, 3 = back, 4 = right, 5 = left
_OPP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
_FACE_NAMES = {0: "top", 1: "bottom", 2: "front",
                3: "back", 4: "right", 5: "left"}


def _fold_face_map(cells: List[Tuple[int, int]]
                   ) -> Dict[int, int]:
    """Map net cell index -> cube face id when folded.

    Same BFS approach as dice_net_qa._fold_opposite_map but returns the
    direct cell-to-face mapping.
    """
    cell_set = set(cells)
    c2i = {c: i for i, c in enumerate(cells)}

    def roll(face, top, dr, dc):
        RIGHT = {
            (0, 2): 4, (0, 3): 5, (0, 4): 3, (0, 5): 2,
            (1, 2): 5, (1, 3): 4, (1, 4): 2, (1, 5): 3,
            (2, 0): 5, (2, 1): 4, (2, 4): 0, (2, 5): 1,
            (3, 0): 4, (3, 1): 5, (3, 4): 1, (3, 5): 0,
            (4, 0): 2, (4, 1): 3, (4, 2): 1, (4, 3): 0,
            (5, 0): 3, (5, 1): 2, (5, 2): 0, (5, 3): 1,
        }
        right = RIGHT[(face, top)]
        if dr == -1 and dc == 0:
            return top, _OPP[face]
        if dr == 1 and dc == 0:
            return _OPP[top], face
        if dr == 0 and dc == 1:
            return right, top
        if dr == 0 and dc == -1:
            return _OPP[right], top
        return face, top

    if not cells:
        return {}

    start = cells[0]
    state = {start: (0, 2)}
    queue = [start]
    visited = {start}
    while queue:
        cur = queue.pop(0)
        cf, ct = state[cur]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (cur[0] + dc, cur[1] + dr)
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                nf, nt = roll(cf, ct, dr, dc)
                state[nb] = (nf, nt)
                queue.append(nb)

    return {c2i[c]: state[c][0] for c in state}


class NetValidityAdvancedQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — pick which net unfolding correctly places a
    shaded square corresponding to a shaded face on the rendered cube."""

    ALLOW_ROTATION = False
    ENV_NAME = "net_validity_advanced"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"use_5_options": False, "trap_rate": 0.0,
                    "tight_distractors": False, "nets": _NETS[:1]}
        if level <= 3:
            return {"use_5_options": True, "trap_rate": 0.10,
                    "tight_distractors": False, "nets": _NETS[:2]}
        if level <= 6:
            return {"use_5_options": True, "trap_rate": 0.15,
                    "tight_distractors": False, "nets": _NETS[:3]}
        if level == 7:
            return {"use_5_options": True, "trap_rate": 0.25,
                    "tight_distractors": True, "nets": _NETS}
        # L8/L9
        return {"use_5_options": True, "trap_rate": 0.40,
                "tight_distractors": True, "nets": _NETS}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1019)

        for _ in range(20):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        net_name, net_cells = rng.choice(cfg["nets"])
        cells = list(net_cells)
        n = len(cells)
        face_map = _fold_face_map(cells)
        if len(face_map) < n:
            return None

        # Pick the cube face to "shade" (target). Pick a face that is non-trivial
        # to identify in the net.
        cube_face_id = rng.choice([0, 1, 2, 3, 4, 5])
        # Find the cell index whose folded face matches cube_face_id
        correct_cell = None
        for ci, fid in face_map.items():
            if fid == cube_face_id:
                correct_cell = ci
                break
        if correct_cell is None:
            return None

        # Build candidate cells for distractors. Each distractor candidate is a
        # cell index OTHER than correct_cell (so the distractor net shades the
        # wrong cell).
        all_cells = list(range(n))
        distractor_pool = [c for c in all_cells if c != correct_cell]
        rng.shuffle(distractor_pool)

        # Tight distractor mode: prefer the OPPOSITE-face cell as a primary
        # distractor (most confusable).
        if cfg["tight_distractors"]:
            opp_face = _OPP[cube_face_id]
            opp_cell = None
            for ci, fid in face_map.items():
                if fid == opp_face:
                    opp_cell = ci
                    break
            if opp_cell is not None and opp_cell in distractor_pool:
                distractor_pool.remove(opp_cell)
                distractor_pool.insert(0, opp_cell)

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            # All value options are wrong cells; correct = "No correct answer"
            chosen = distractor_pool[:n_value]
            if len(chosen) < n_value:
                return None
            options_cells = list(chosen)
            rng.shuffle(options_cells)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distractor_pool[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options_cells = [correct_cell] + chosen
            rng.shuffle(options_cells)
            correct_letter = opt_letters[options_cells.index(correct_cell)]

        # Render the panel: cube on top showing shaded face, plus N candidate nets
        image = self._render_panel(cells, options_cells, opt_letters,
                                    cube_face_id, use_5, rng)

        # Standard textbook phrasing — Q14 / Q22 style
        face_name = _FACE_NAMES[cube_face_id]
        stems = [
            (
                f"As shown in the diagram, the {face_name} face of the cube is "
                f"shaded (highlighted in red). When the cube is unfolded into "
                f"a flat net, which of the following expanded views correctly "
                f"shows the position of the shaded face?"
            ),
            (
                f"As shown in the figure, the cube has its {face_name} face "
                f"shaded. Unfold the cube into a flat net. Which of the "
                f"following unfolded diagrams correctly places the shaded "
                f"square?"
            ),
            (
                f"The diagram shows a cube whose {face_name} face is colored. "
                f"Unfold the cube. Which of the following candidate net "
                f"unfoldings is correct?"
            ),
        ]
        stem = rng.choice(stems)
        # Build option list
        opt_lines = "\n".join(
            f"{opt_letters[i]}. Net option {opt_letters[i]} (in the figure)"
            for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Rendering: cube (3D iso) + N candidate net unfoldings (2D grid each)
    # ------------------------------------------------------------------ #
    def _render_panel(self, net_cells: List[Tuple[int, int]],
                      options_cells: List[int], opt_letters: List[str],
                      cube_face_id: int, use_5: bool,
                      rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        bg = style["bg_color"]

        n_opts = len(options_cells)
        # Layout: 2 rows. Top row is the cube (full width). Bottom row is N
        # candidate nets side by side.
        fig = plt.figure(figsize=(11 * sc, 8 * sc))
        fig.patch.set_facecolor(bg)
        gs = fig.add_gridspec(2, max(1, n_opts), height_ratios=[1.1, 1.0],
                              hspace=0.35, wspace=0.30)
        # --- top: cube --- spans all columns
        ax_cube = fig.add_subplot(gs[0, :], projection='3d')
        ax_cube.set_facecolor(bg)
        self._draw_cube_with_shaded_face(ax_cube, cube_face_id, rng)

        # --- bottom: N candidate nets ---
        # Normalize net layout
        min_c = min(c for c, r in net_cells)
        min_r = min(r for c, r in net_cells)
        norm_cells = [(c - min_c, r - min_r) for c, r in net_cells]
        max_c = max(c for c, r in norm_cells) + 1
        max_r = max(r for c, r in norm_cells) + 1

        for i, shaded_cell_idx in enumerate(options_cells):
            ax = fig.add_subplot(gs[1, i])
            ax.set_facecolor(bg)
            ax.set_aspect("equal")
            ax.axis("off")
            self._draw_net(ax, norm_cells, shaded_cell_idx, max_c, max_r,
                           opt_letters[i], style, rng)

        suptitle = "Cube with shaded face + candidate net unfoldings"
        fig.suptitle(suptitle,
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", y=0.98)
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                fig.subplots_adjust(top=0.92, bottom=0.05, left=0.04, right=0.98)
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_cube_with_shaded_face(self, ax, shaded_face_id: int,
                                    rng: random.Random):
        """Draw a 3D iso cube with one face highlighted in red."""
        s = 1.0
        verts = np.array([
            [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
            [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
        ], dtype=float)

        # Face vertex orderings:
        # 0 = top    -> z = s (4,5,6,7)
        # 1 = bottom -> z = 0 (0,1,2,3)
        # 2 = front  -> y = 0 (0,1,5,4)
        # 3 = back   -> y = s (3,2,6,7)
        # 4 = right  -> x = s (1,2,6,5)
        # 5 = left   -> x = 0 (0,3,7,4)
        face_indices = {
            0: [4, 5, 6, 7],
            1: [0, 1, 2, 3],
            2: [0, 1, 5, 4],
            3: [3, 2, 6, 7],
            4: [1, 2, 6, 5],
            5: [0, 3, 7, 4],
        }
        all_face_polys = []
        all_face_colors = []
        for fid, idxs in face_indices.items():
            face_pts = [tuple(verts[i]) for i in idxs]
            all_face_polys.append(face_pts)
            if fid == shaded_face_id:
                all_face_colors.append("#e74c3c")
            else:
                all_face_colors.append("#dfe6e9")
        poly = Poly3DCollection(all_face_polys, alpha=0.85,
                                facecolors=all_face_colors,
                                edgecolors='black', linewidths=1.6)
        ax.add_collection3d(poly)

        ax.set_xlim(-0.2, s + 0.2)
        ax.set_ylim(-0.2, s + 0.2)
        ax.set_zlim(-0.2, s + 0.2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title("Cube (shaded face = " + _FACE_NAMES[shaded_face_id] + ")",
                     fontsize=11, fontweight="bold", pad=4)
        ax.view_init(elev=22, azim=42)
        # Hide the panes
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        except Exception:
            pass

    def _draw_net(self, ax, norm_cells, shaded_cell_idx, max_c, max_r,
                  label: str, style: dict, rng: random.Random):
        # Solid background colors for cells
        palette = list(style["palette"])
        rng.shuffle(palette)

        for i, (c, r) in enumerate(norm_cells):
            dy = max_r - 1 - r
            if i == shaded_cell_idx:
                color = "#e74c3c"
                alpha = 0.9
            else:
                color = "#bdc3c7"
                alpha = 0.45
            rect = mpatches.Rectangle(
                (c + 0.05, dy + 0.05), 0.90, 0.90,
                facecolor=color, edgecolor="#222",
                linewidth=1.6, alpha=alpha)
            ax.add_patch(rect)

        ax.set_xlim(-0.4, max_c + 0.4)
        ax.set_ylim(-0.6, max_r + 0.4)
        ax.set_title(f"({label})", fontsize=12, fontweight="bold", pad=2)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        return super()._check_answer(predicted, ground_truth)


if __name__ == "__main__":
    env = NetValidityAdvancedQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
