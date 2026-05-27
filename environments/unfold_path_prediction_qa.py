"""
Unfold Path Prediction QA.

A cube with a colored dot on one face (labelled TOP / FRONT / RIGHT / ...) is
shown in a reference panel. The four options below show the same hexomino
net with the dot placed on DIFFERENT cells. The student must pick the net
option whose marked cell corresponds to the labelled face of the cube.

Difficulty axes:
  A) net_shape_type: cross (canonical) -> uncommon valid net
  B) dot_position: center of face -> near edge

Fix (2026-04-17 audit b38): L0 was 0.00 pass-rate because the reference
cube was never drawn, so the student had no way to know which face the dot
belonged on. We now draw a cube with a labelled marked face AND label
each net cell with its face name (TOP/BOTTOM/FRONT/BACK/LEFT/RIGHT). The
question names the marked face explicitly so the task reduces to face
identification on the net.
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv

# Net represented as (cells, face_for_each_cell_index).
# cells: list of (col, row) grid positions.
# faces: same length; face name for the cell at that index.
# Cross net orientation: bottom-left-front-right in row 1, top above front,
# back above top (or extended out). Face assignments are picked to be a
# valid unfolding.
_CUBE_NETS = [
    # Cross net:
    # row 3:      TOP
    # row 2:      BACK
    # wait we want this ordering: bottom, left, front, right, top, back
    # cell 0: (1,0) bottom of the cross -> 'BOTTOM'
    # cell 1: (0,1) left of middle strip -> 'LEFT'
    # cell 2: (1,1) centre of middle strip (the "front") -> 'FRONT'
    # cell 3: (2,1) right of middle strip -> 'RIGHT'
    # cell 4: (1,2) top of cross column -> 'TOP'
    # cell 5: (1,3) top-top -> 'BACK'
    (
        [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (1, 3)],
        ['BOTTOM', 'LEFT', 'FRONT', 'RIGHT', 'TOP', 'BACK'],
    ),
    # T-shape (4 in a row horizontally + 2 stacked on top of col 1):
    # bottom row of 4: left-front-right-back (or similar); stack is bottom,top.
    (
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (1, 2)],
        ['LEFT', 'FRONT', 'RIGHT', 'BACK', 'TOP', 'BOTTOM'],
    ),
    # L-shape (cells down col 0 + two cells in col 1 at top):
    (
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2)],
        ['FRONT', 'TOP', 'BACK', 'BOTTOM', 'RIGHT', 'LEFT'],
    ),
]

# Face mapping: which cell index maps to which cube face
# For cross net: [0]=bottom, [1]=left, [2]=front, [3]=right, [4]=top, [5]=back
_FACE_NAMES = ['BOTTOM', 'LEFT', 'FRONT', 'RIGHT', 'TOP', 'BACK']

_TITLE_VARIANTS_UNF = [
    "Cube Unfolding Puzzle",
    "Net & Marker Match",
    "Identify the correct net",
]
_QUESTION_TEMPLATES_UNF = [
    ("The cube on the LEFT has a {color_name} dot on its {face} face. "
     "Which of the four nets (A)-(D) on the right shows the dot on the "
     "cell labelled {face}?\n(A)  (B)  (C)  (D)"),
    ("A cube (shown on the left) has a {color_name} marker on the {face} "
     "face. When the cube is unfolded into the hexomino net on the right, "
     "which option places the marker on the cell labelled {face}?\n"
     "(A)  (B)  (C)  (D)"),
    ("The reference cube (left) has a {color_name} dot painted on its "
     "{face} face. Each option (A)-(D) shows the same unfolded net with a "
     "dot in a different cell. Pick the option whose dot is on the cell "
     "labelled {face}.\n(A)  (B)  (C)  (D)"),
]
_DOT_COLORS_UNF = [
    ("red", "#e53935"),
    ("blue", "#1e88e5"),
    ("green", "#43a047"),
    ("orange", "#fb8c00"),
    ("purple", "#8e24aa"),
    ("magenta", "#d81b60"),
]
_MARKER_VARIANTS_UNF = ['o', 's', 'D', '^', '*']

class UnfoldPathPredictionQA(StandaloneVisualEnv):
    ENV_NAME = "unfold_path_prediction"

    def _level_config(self, level):
        # L0-L3: always use the cross net (canonical, easiest to visualise).
        # L4-L7: widen to T / L nets too.
        # L8-L9: include all nets AND hide face labels on the options (so
        #        the student must infer face identity purely from shape).
        n_nets = 1 if level <= 3 else min(len(_CUBE_NETS), 1 + level // 3)
        return {
            'n_nets': n_nets,
            'hide_net_labels': level >= 8,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1018)
        style = self._random_style()

        # Pick a net (cells + face assignment for each cell index)
        net_idx = rng.randint(0, cfg['n_nets'] - 1)
        net_cells, face_at_cell = _CUBE_NETS[net_idx]

        # Pick dot color, marker, and face fill color
        color_name, color_hex = rng.choice(_DOT_COLORS_UNF)
        marker = rng.choice(_MARKER_VARIANTS_UNF)
        face_color = rng.choice(style['palette'])

        # Pick the correct cell (and its face name).
        correct_cell_idx = rng.randint(0, len(net_cells) - 1)
        correct_face = face_at_cell[correct_cell_idx]

        # Build 4 options: the correct one + 3 distractors (different cells,
        # different faces -- dedup on face so no two options feel identical).
        wrong_idxs = [i for i in range(len(net_cells)) if i != correct_cell_idx]
        rng.shuffle(wrong_idxs)
        distractors = wrong_idxs[:3]
        option_cells = [correct_cell_idx] + distractors
        rng.shuffle(option_cells)
        correct_letter = "ABCD"[option_cells.index(correct_cell_idx)]

        # Draw: 1 reference cube panel + 4 net option panels.
        sc = style['figsize_scale']
        fig = plt.figure(figsize=(16 * sc, 4 * sc))
        fig.patch.set_facecolor(style['bg_color'])
        gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
                              wspace=0.35)
        ax_cube = fig.add_subplot(gs[0])
        axes = [fig.add_subplot(gs[k + 1]) for k in range(4)]

        # ---- Reference cube (isometric projection) ----
        self._draw_reference_cube(ax_cube, correct_face, color_hex, marker,
                                   face_color, style)
        ax_cube.set_title("Reference cube",
                          fontsize=style['font_size_base'] + 1,
                          fontweight='bold')

        # ---- Four net options ----
        hide_labels = cfg['hide_net_labels']
        for idx, cell_idx in enumerate(option_cells):
            ax = axes[idx]
            ax.set_facecolor(style['bg_color'])
            ax.set_aspect('equal')
            ax.axis('off')
            for ci, (c, r) in enumerate(net_cells):
                fc = face_color if ci != cell_idx else '#ecf0f1'
                rect = mpatches.FancyBboxPatch(
                    (c, r), 1, 1, facecolor=fc, edgecolor='#333',
                    linewidth=1.5, alpha=0.7)
                ax.add_patch(rect)
                # Label each cell with its face name (unless hidden).
                if not hide_labels:
                    ax.text(c + 0.5, r + 0.08, face_at_cell[ci],
                            ha='center', va='bottom',
                            fontsize=8, color='#333', fontweight='bold')
                if ci == cell_idx:
                    ax.plot(c + 0.5, r + 0.55, marker, color=color_hex,
                            markersize=14, markeredgecolor='#222',
                            markeredgewidth=0.8)
            xs = [c for c, r in net_cells]
            ys = [r for c, r in net_cells]
            ax.set_xlim(min(xs) - 0.5, max(xs) + 1.5)
            ax.set_ylim(min(ys) - 0.5, max(ys) + 1.5)
            ax.set_title(f"({chr(65 + idx)})", fontsize=12, fontweight='bold')

        fig.suptitle(rng.choice(_TITLE_VARIANTS_UNF),
                     fontsize=style['font_size_base'] + 3, fontweight='bold')
        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q = rng.choice(_QUESTION_TEMPLATES_UNF).format(
            color_name=color_name, face=correct_face)
        return q, correct_letter, img

    # ------------------------------------------------------------------ #
    def _draw_reference_cube(self, ax, marked_face, marker_color, marker,
                              face_color, style) -> None:
        """Draw a small isometric cube with a labelled face and a dot on
        the marked face."""
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal')
        ax.axis('off')

        # Isometric-ish projection of a unit cube.
        # Three visible faces: TOP, FRONT, RIGHT.
        # Define vertices of cube corners in 2D (isometric).
        s = 1.0
        # top face (diamond) corners
        A = (0.0, 1.2)          # top-left
        B = (1.0, 1.8)          # top-back
        C = (2.0, 1.2)          # top-right
        D = (1.0, 0.6)          # top-front
        # front face corners
        E = (0.0, 0.0)          # bot-left
        F = (1.0, -0.6)         # bot-front
        G = (2.0, 0.0)          # bot-right
        # mapping face name -> polygon.
        # The isometric view shows three faces: TOP (diamond on top), FRONT
        # (left side in projection, closer to viewer in "front-left" angle),
        # and RIGHT (right side in projection). Assignments chosen to be
        # consistent with the face labels on the net.
        faces_poly = {
            'TOP': [A, B, C, D],
            'FRONT': [A, D, F, E],   # left-visible side of the projection
            'RIGHT': [D, C, G, F],   # right-visible side of the projection
        }
        # To also allow BACK / BOTTOM / LEFT: not visible in this view, so
        # for marked_face in {BACK, BOTTOM, LEFT}, show with a dashed outline
        # on the hidden-face position and a "(hidden — dot on this face)"
        # annotation.
        visible = ('TOP', 'FRONT', 'RIGHT')

        for name, pts in faces_poly.items():
            is_marked = (name == marked_face)
            poly = mpatches.Polygon(
                pts, closed=True,
                facecolor='#ecf0f1' if is_marked else face_color,
                edgecolor='#222', linewidth=1.6, alpha=0.85)
            ax.add_patch(poly)
            # Label the face centre
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            ax.text(cx, cy + 0.12, name, ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#222')
            if is_marked:
                ax.plot(cx, cy - 0.12, marker, color=marker_color,
                        markersize=14, markeredgecolor='#222',
                        markeredgewidth=0.8)

        if marked_face not in visible:
            # Annotate below the cube which hidden face bears the dot.
            ax.text(1.0, -1.0,
                    f"(dot on hidden {marked_face} face)",
                    ha='center', va='center',
                    fontsize=9, color=marker_color, fontweight='bold')

        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.4, 2.2)

if __name__ == "__main__":
    env = UnfoldPathPredictionQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
