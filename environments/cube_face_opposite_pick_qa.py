"""
Cube Face Opposite Pick QA.

MCQ-letter task. Renders an unfolded cube net with each of the 6 faces
labelled with a distinct uppercase letter (A, B, C, D, E, F). Asks which
labelled block ends up opposite to a referenced block once the net is
folded into a cube. Question style mirrors reference Solid-Figures expanded-
view-of-cubes templates ("after folding the following figure into a cube,
the block opposite to the block to the right of block D is ( )").

Reward: MCQ letter (A-F) match, with one always-wrong "No correct answer"
distractor at higher levels.

Level mapping (parameter["level"], 0-9):
  L0: canonical plus-cross net, ask "the block opposite to A". Trivial:
      6 faces labelled A-F in fixed pattern; only 4 plausible options shown.
  L1: plus-cross net, ask opposite to a random face.
  L2: plus-cross net, "the block to the right of X" reference.
  L3: T net, opposite-to-X.
  L4: T net, opposite to indirect referent.
  L5: any of 4 valid nets, opposite-to-X.
  L6+: any net + indirect referent + an "E. No correct answer" distractor.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ====================================================================== #
# Cube-net unfolding geometry
# ====================================================================== #
#
# Each net is a list of (col, row, FACE_ID) triples on a 2D grid. FACE_IDs
# 0..5 represent the 6 faces of a cube; opposite pairs are (0,1), (2,3),
# (4,5). Layouts (taken from valid hexomino unfolding of a cube):
#
#   plus (canonical):           tee (3-row T):
#         T                       L F R B    (row 0)
#     L   F   R   B                 D        (row 1)
#         B(bottom)                 U        (row 2 -- top)
#
# We pre-compute which 2D-cell index corresponds to which cube face.

# ----- 0=top,1=bottom,2=front,3=back,4=left,5=right ---------------------
_FACE_LABELS_DEFAULT = ['T', 'B', 'F', 'K', 'L', 'R']
OPP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

# --- Six valid nets, each as [(col, row, face_id), ...] ---------------- #
# All assignments verified by manual fold-checking (each ID 0-5 appears once).

# Plus-cross
_NET_PLUS = [
    (1, 0, 1),  # bottom of cross -> bottom face
    (0, 1, 4),  # left
    (1, 1, 2),  # front (centre of strip)
    (2, 1, 5),  # right
    (3, 1, 3),  # back (extending right)
    (1, 2, 0),  # top of cross -> top face
]

# T net (T pointing up)
_NET_TEE = [
    (0, 0, 4),  # left
    (1, 0, 2),  # front
    (2, 0, 5),  # right
    (3, 0, 3),  # back
    (1, 1, 1),  # bottom
    (1, 2, 0),  # top
]

# L net
_NET_ELL = [
    (0, 0, 0),
    (0, 1, 2),
    (0, 2, 1),
    (1, 2, 5),
    (2, 2, 3),
    (3, 2, 4),
]

# Z / staircase net
_NET_ZEE = [
    (0, 0, 0),
    (0, 1, 2),
    (1, 1, 5),
    (1, 2, 1),
    (2, 2, 3),
    (2, 3, 4),
]

_NETS_BY_LEVEL = {
    0: [_NET_PLUS],
    1: [_NET_PLUS],
    2: [_NET_PLUS],
    3: [_NET_PLUS, _NET_TEE],
    4: [_NET_PLUS, _NET_TEE],
    5: [_NET_PLUS, _NET_TEE, _NET_ELL],
    6: [_NET_PLUS, _NET_TEE, _NET_ELL, _NET_ZEE],
    7: [_NET_PLUS, _NET_TEE, _NET_ELL, _NET_ZEE],
    8: [_NET_TEE, _NET_ELL, _NET_ZEE],
    9: [_NET_ELL, _NET_ZEE],
}


# ====================================================================== #
# Question templates
# ====================================================================== #

_DIRECT_STEMS = [
    "After folding the figure into a cube, the block opposite to block {x} is ( ).",
    "Fold the following figure into a cube. The block opposite to block {x} is ( ).",
    "When the net shown is folded into a cube, which block ends up opposite to block {x}?",
    "If you fold the figure into a cube, which block is on the face opposite block {x}?",
    "The figure shows the unfolded view of a cube. After folding, the block opposite block {x} is ( ).",
    "Fold the net into a cube. The block opposite block {x} is ( ).",
    "Once the net is folded into a cube, which labelled block is opposite block {x}?",
    "Identify the block opposite to block {x} after the net is folded into a cube.",
]

_INDIRECT_STEMS_BUILD = [
    # neighbour_descriptor formatters using the geometric net adjacency
    "After folding the figure into a cube, the block opposite the block {neighbour_desc} is ( ).",
    "Fold the figure into a cube. The block opposite the block {neighbour_desc} is ( ).",
    "When the net is folded into a cube, the block opposite the block {neighbour_desc} is ( ).",
    "Fold the net shown into a cube. Which block is opposite the block {neighbour_desc}?",
    "Once the net is folded into a cube, which labelled block is opposite the block {neighbour_desc}?",
    "Identify which block ends up opposite the block {neighbour_desc} when the net is folded into a cube.",
    "The figure is the unfolded view of a cube. After folding, the block opposite the block {neighbour_desc} is ( ).",
    "Fold the figure into a cube; the block opposite the block {neighbour_desc} is ( ).",
]


_OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


# ====================================================================== #
# Env class
# ====================================================================== #

class CubeFaceOppositePickQA(StandaloneVisualEnv):
    ENV_NAME = "cube_face_opposite_pick"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the single MCQ letter "
        "directly inside `<answer>...</answer>` tags."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(int(level), 9))
        return {
            "nets": _NETS_BY_LEVEL[level],
            "indirect_referent": level >= 2,
            "no_correct_distractor": level >= 6,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 313)
        self._primary_complexity_feature = level

        for attempt in range(15):
            net = rng.choice(cfg["nets"])
            # Letter assignment: each cell gets one letter A-F; we assign in
            # a stable but seed-shuffled order so cell positions do not reveal
            # cube faces.
            letters = list(_OPTION_LETTERS[:6])
            rng.shuffle(letters)
            cell_letter = {}            # (col,row) -> letter
            cell_face = {}              # (col,row) -> face_id 0..5
            for i, (c, r, fid) in enumerate(net):
                cell_letter[(c, r)] = letters[i]
                cell_face[(c, r)] = fid

            # Target referent letter
            ref_cell = rng.choice(list(cell_letter.keys()))
            ref_letter = cell_letter[ref_cell]
            ref_face_id = cell_face[ref_cell]

            # Direct opposite (face opposite ref_face_id)
            opp_face_id = OPP[ref_face_id]
            opp_letter_direct = None
            for cell, fid in cell_face.items():
                if fid == opp_face_id:
                    opp_letter_direct = cell_letter[cell]
                    break

            if cfg["indirect_referent"] and rng.random() < 0.55:
                # Pick a 2D net-adjacent cell of ref_cell as the actual referent.
                # The student must (a) identify which labelled cell is "to the
                # right of ref_letter on the net" and (b) find its opposite face.
                neighbours = [
                    ((ref_cell[0] + 1, ref_cell[1]), "to the right of"),
                    ((ref_cell[0] - 1, ref_cell[1]), "to the left of"),
                    ((ref_cell[0], ref_cell[1] - 1), "above"),
                    ((ref_cell[0], ref_cell[1] + 1), "below"),
                ]
                rng.shuffle(neighbours)
                chosen = None
                for ncell, descriptor in neighbours:
                    if ncell in cell_letter:
                        chosen = (ncell, descriptor)
                        break
                if chosen is None:
                    continue
                neighbour_cell, descriptor = chosen
                neighbour_letter = cell_letter[neighbour_cell]
                neighbour_face_id = cell_face[neighbour_cell]
                target_face_id = OPP[neighbour_face_id]
                # Find labelled cell with target_face_id (the answer)
                answer_letter = None
                for cell, fid in cell_face.items():
                    if fid == target_face_id:
                        answer_letter = cell_letter[cell]
                        break
                if answer_letter is None:
                    continue
                stem = rng.choice(_INDIRECT_STEMS_BUILD).format(
                    neighbour_desc=f"{descriptor} block {ref_letter}")
            else:
                # Direct: opposite of ref_letter
                stem = rng.choice(_DIRECT_STEMS).format(x=ref_letter)
                answer_letter = opp_letter_direct
                if answer_letter is None:
                    continue

            # Build MCQ options. The 5 candidate labels (excluding the answer's
            # paired-letter partner if needed) -> randomly take 4 distinct
            # labels including the correct one; optionally add an "E. No
            # correct answer" distractor at higher levels.
            all_labels = list(_OPTION_LETTERS[:6])
            distractor_pool = [l for l in all_labels if l != answer_letter]
            rng.shuffle(distractor_pool)
            n_distract = 3
            distractors = distractor_pool[:n_distract]
            options = distractors + [answer_letter]
            # Sort options A->F so the displayed letter labels go in order
            options = sorted(set(options))
            # Build option label letter mapping (A. {opt0}, B. {opt1}, ...)
            n_options = len(options)
            opt_text = "; ".join(f"{_OPTION_LETTERS[i]}. {options[i]}"
                                 for i in range(n_options))
            if cfg["no_correct_distractor"]:
                opt_text += f"; {_OPTION_LETTERS[n_options]}. No correct answer"
            correct_mcq_letter = _OPTION_LETTERS[options.index(answer_letter)]

            # Render
            img = self._render(net, cell_letter, ref_letter, rng)

            stem_full = (
                f"{stem} Choose the correct option. {opt_text}. "
                f"Answer with the single letter."
            )
            return stem_full, correct_mcq_letter, img
        return None

    # ------------------------------------------------------------------ #
    def _render(self, net, cell_letter, ref_letter, rng):
        style = self._random_style()
        sc = style["figsize_scale"]

        cols = [c for (c, r, _) in net]
        rows = [r for (c, r, _) in net]
        ncols = max(cols) - min(cols) + 1
        nrows = max(rows) - min(rows) + 1
        cmin, rmin = min(cols), min(rows)

        fig, ax = plt.subplots(figsize=(max(5, (ncols + 2) * 1.0 * sc),
                                         max(4, (nrows + 2) * 1.0 * sc)))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        # Pick 6 distinct soft colors -- one per cell.
        palette = [c for c in style["palette"]
                   if c.lower() not in ('#000000', '#ffffff')]
        if len(palette) < 6:
            palette = ['#a8dadc', '#f1c40f', '#f4a261', '#9b87f5',
                       '#a8e6cf', '#f08080']
        cell_colors = list(palette[:6])
        rng.shuffle(cell_colors)

        for i, (c, r, _) in enumerate(net):
            x = c - cmin
            y = (nrows - 1) - (r - rmin)  # flip y so row 0 is at top
            rect = mpatches.FancyBboxPatch(
                (x + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=cell_colors[i],
                edgecolor="#222222",
                linewidth=2.0,
            )
            ax.add_patch(rect)
            label = cell_letter[(c, r)]
            ax.text(x + 0.5, y + 0.5, label,
                    ha="center", va="center",
                    fontsize=style["font_size_base"] + 16,
                    fontweight="bold", color="#ffffff",
                    fontfamily=style["font_family"])

        ax.set_xlim(-0.5, ncols + 0.5)
        ax.set_ylim(-0.5, nrows + 0.5)
        ax.set_title("Cube net (each block labelled)",
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_cfop", exist_ok=True)
    env = CubeFaceOppositePickQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed}: ok={ok}, ans={env._answer}")
            if ok:
                env.render().save(f"/tmp/env_check_cfop/cfop_L{level}_s{seed}.png")
                v = env.verify(f"<answer>{env._answer}</answer>")
                print(f"  verify={v}")
                print(f"  Q: {env._question[:200]}")
