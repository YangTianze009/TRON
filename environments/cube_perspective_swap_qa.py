"""
Cube perspective-swap QA.

A cube is shown in standard isometric view with 3 visible faces. Each of the
6 faces is labelled with a distinct uppercase letter (A, B, C, D, E, F).
After a specified 90 degree rotation about an axis (vertical or horizontal),
which face becomes hidden (i.e., was previously visible but is no longer
visible)? The model picks the answer letter from 4 candidates.

Standard view: visible faces are TOP (T), FRONT (F), RIGHT (R).
After rotating CW about vertical axis, FRONT becomes hidden (now LEFT side
is visible: T, RIGHT-old becomes FRONT, BACK becomes RIGHT, LEFT-old hidden).
Wait — careful: in CW about vertical, RIGHT becomes BACK (hidden), so RIGHT
becomes hidden. We model this carefully below.

Difficulty levels 0-9:
  L0-L1: rotation about vertical axis only, only CW.
  L2-L3: vertical CW or CCW.
  L4-L5: vertical or horizontal X-axis CW.
  L6-L7: any axis (X or Y or Z) any direction.
  L8-L9: same + answer-options include faces that were already hidden
         (so model must distinguish "hidden after rotation" from "previously
         hidden").
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Face IDs:
#   0: top (T)
#   1: bottom (Bo)
#   2: front (F)
#   3: back (Bk)
#   4: right (R)
#   5: left (L)
#
# Visible-face triple in initial view: (top, front, right) = (0, 2, 4).
# Hidden-face triple in initial view: (bottom, back, left) = (1, 3, 5).

_VISIBLE_INITIAL = [0, 2, 4]   # top, front, right
_HIDDEN_INITIAL = [1, 3, 5]    # bottom, back, left

# Rotation specifications. Each rotation is a permutation of face IDs:
# new_face[id] = old_face_at_that_position.
# We define the inverse map: old_pos -> new_pos.
# After rotating, what was at position X is now at position perm[X].
#
# Vertical axis = Z axis (top-bottom). Looking down from top:
#   CW: front -> right -> back -> left -> front (top, bottom unchanged)
#   So new top=top, new bottom=bottom, new front=left, new right=front,
#      new back=right, new left=back.
# We track: which face IDs are visible (in positions top/front/right) AFTER
# rotation. To compute, we apply the rotation to each face's "current position"
# label, then check which faces end up at top/front/right.

# Each rotation rotates POSITIONS. We track which face ID is at which position.
# Initial: position->face_id is identity (pos i has face i).
# After rotation, position->face_id changes.

def _rot_z_cw(pos_to_face: List[int]) -> List[int]:
    """Rotate cube CW about vertical (z) axis (looking down).
    Faces at front/right/back/left positions cycle: front<-right, right<-back, back<-left, left<-front.
    Top, bottom unchanged."""
    new = list(pos_to_face)
    # position 2=front, 4=right, 3=back, 5=left
    # After CW from top: face at right moves to front; face at back moves to right; face at left moves to back; face at front moves to left.
    new[2] = pos_to_face[4]
    new[4] = pos_to_face[3]
    new[3] = pos_to_face[5]
    new[5] = pos_to_face[2]
    return new


def _rot_z_ccw(pos_to_face: List[int]) -> List[int]:
    new = list(pos_to_face)
    new[2] = pos_to_face[5]
    new[5] = pos_to_face[3]
    new[3] = pos_to_face[4]
    new[4] = pos_to_face[2]
    return new


def _rot_x_cw(pos_to_face: List[int]) -> List[int]:
    """Rotate cube CW about x-axis (left-right). Looking from the right.
    top -> front -> bottom -> back -> top. Right, left unchanged."""
    new = list(pos_to_face)
    new[2] = pos_to_face[0]   # front <- top
    new[1] = pos_to_face[2]   # bottom <- front
    new[3] = pos_to_face[1]   # back <- bottom
    new[0] = pos_to_face[3]   # top <- back
    return new


def _rot_x_ccw(pos_to_face: List[int]) -> List[int]:
    new = list(pos_to_face)
    new[0] = pos_to_face[2]   # top <- front
    new[2] = pos_to_face[1]   # front <- bottom
    new[1] = pos_to_face[3]   # bottom <- back
    new[3] = pos_to_face[0]   # back <- top
    return new


def _rot_y_cw(pos_to_face: List[int]) -> List[int]:
    """Rotate CW about y-axis (front-back). Looking from front.
    top -> right -> bottom -> left -> top. Front, back unchanged."""
    new = list(pos_to_face)
    new[4] = pos_to_face[0]   # right <- top
    new[1] = pos_to_face[4]   # bottom <- right
    new[5] = pos_to_face[1]   # left <- bottom
    new[0] = pos_to_face[5]   # top <- left
    return new


def _rot_y_ccw(pos_to_face: List[int]) -> List[int]:
    new = list(pos_to_face)
    new[0] = pos_to_face[4]   # top <- right
    new[4] = pos_to_face[1]   # right <- bottom
    new[1] = pos_to_face[5]   # bottom <- left
    new[5] = pos_to_face[0]   # left <- top
    return new


_ROTATIONS = {
    "z_cw": (_rot_z_cw, "rotate the cube 90 degrees clockwise about the vertical axis (as viewed from above)"),
    "z_ccw": (_rot_z_ccw, "rotate the cube 90 degrees counterclockwise about the vertical axis (as viewed from above)"),
    "x_cw": (_rot_x_cw, "rotate the cube 90 degrees forward (top tilts toward you)"),
    "x_ccw": (_rot_x_ccw, "rotate the cube 90 degrees backward (top tilts away from you)"),
    "y_cw": (_rot_y_cw, "rotate the cube 90 degrees so its right side tips downward"),
    "y_ccw": (_rot_y_ccw, "rotate the cube 90 degrees so its left side tips downward"),
}


_QUESTION_TEMPLATES = [
    "A cube is shown with letters on its faces. After we {rot_desc}, which face that was visible before is now hidden? Answer with one letter from the options A-D.",
    "Examine the cube and its visible face letters. After we {rot_desc}, which previously-visible face becomes hidden? Pick A, B, C, or D.",
    "The cube has letters on each face. We {rot_desc}. Which formerly-visible face is no longer visible? Choose from A, B, C, D.",
    "Identify the face that was visible before the rotation but is hidden afterwards. We {rot_desc}. Pick the answer (A-D).",
    "Look at the cube's labelled faces. After we {rot_desc}, exactly one of the originally-visible faces ends up hidden. Which? A, B, C, or D.",
    "Mentally {rot_desc}. Which face that was visible at the start is hidden after the rotation? Single letter (A-D).",
    "After applying the rotation to the cube, one of the visible-at-start faces is no longer visible. We {rot_desc}. Which letter is it? A, B, C, D.",
    "Determine which previously-visible face becomes hidden when we {rot_desc}. Choose A, B, C, or D.",
    "We {rot_desc}. Among the originally-visible faces, which one is now hidden? Reply with one letter A-D.",
    "Pick the face letter that was visible before but is hidden after the rotation. We {rot_desc}. Options: A, B, C, D.",
    "After we {rot_desc}, name (A-D) the face that was previously visible but is now out of sight.",
    "The cube has labelled faces. After the rotation ({rot_desc}), which originally-visible face becomes hidden? A, B, C, or D.",
    "Apply the rotation: {rot_desc}. Which face among the previously-visible ones is now hidden? Single letter A-D.",
    "Choose the letter (A-D) of the face that was visible at the start but hidden after we {rot_desc}.",
    "Identify the face that disappears from view after we {rot_desc}. Choose A, B, C, or D.",
    "We {rot_desc}. Which of the originally-visible face letters can no longer be seen? Pick A, B, C, or D.",
]


def _draw_cube_isometric(ax, x: float, y: float, size: float,
                         labels: Dict[int, str], style: Dict):
    """Draw a cube in isometric view at center (x, y) with face labels.

    labels: face_id -> letter to draw on that face (only top, front, right
    are drawn since those are visible).
    """
    s = size
    # Front face (square)
    front = [(x - s, y - s * 0.6), (x + s, y - s * 0.6),
             (x + s, y + s * 0.7), (x - s, y + s * 0.7)]
    # Top face (parallelogram angled up)
    top = [(x - s, y + s * 0.7), (x + s, y + s * 0.7),
           (x + s + s * 0.5, y + s * 1.3), (x - s + s * 0.5, y + s * 1.3)]
    # Right face (parallelogram angled right)
    right = [(x + s, y - s * 0.6), (x + s + s * 0.5, y),
             (x + s + s * 0.5, y + s * 1.3), (x + s, y + s * 0.7)]

    front_color = "#fffaf0"
    top_color = "#fff0d8"
    right_color = "#ffe9c8"
    edge = "#222"

    ax.add_patch(mpatches.Polygon(front, facecolor=front_color,
                                   edgecolor=edge, linewidth=1.6, zorder=2))
    ax.add_patch(mpatches.Polygon(top, facecolor=top_color,
                                   edgecolor=edge, linewidth=1.6, zorder=2))
    ax.add_patch(mpatches.Polygon(right, facecolor=right_color,
                                   edgecolor=edge, linewidth=1.6, zorder=2))

    # Draw face labels (the letters on the visible faces)
    text_kw = dict(fontsize=22, fontweight="bold", color="#222",
                   ha="center", va="center", zorder=4)
    # Front center
    ax.text(x, y + s * 0.05, labels.get(2, ""), **text_kw)
    # Top center (slightly shifted right to fit parallelogram)
    ax.text(x + s * 0.25, y + s * 1.0, labels.get(0, ""), **text_kw)
    # Right center (shifted right)
    ax.text(x + s + s * 0.25, y + s * 0.35, labels.get(4, ""), **text_kw)


def _hidden_faces_legend(ax, x: float, y: float, hidden_labels: Dict[int, str]):
    """Render a small legend listing the hidden-face letters."""
    ax.text(x, y, "Hidden faces (not visible in this view): "
                  f"bottom={hidden_labels.get(1, '?')}  "
                  f"back={hidden_labels.get(3, '?')}  "
                  f"left={hidden_labels.get(5, '?')}",
            fontsize=10, color="#444", ha="center", va="center", style="italic")


class CubePerspectiveSwapQA(StandaloneVisualEnv):
    ENV_NAME = "cube_perspective_swap"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"rot_pool": ["z_cw"], "include_hidden_distractors": False}
        if level <= 3:
            return {"rot_pool": ["z_cw", "z_ccw"], "include_hidden_distractors": False}
        if level <= 5:
            return {"rot_pool": ["z_cw", "z_ccw", "x_cw"],
                    "include_hidden_distractors": False}
        if level <= 7:
            return {"rot_pool": list(_ROTATIONS.keys()),
                    "include_hidden_distractors": False}
        return {"rot_pool": list(_ROTATIONS.keys()),
                "include_hidden_distractors": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2007)

        for _ in range(20):
            result = self._try_generate(sub_rng, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        # Assign letters to faces (A..F). Random assignment.
        letters = ["A", "B", "C", "D", "E", "F"]
        rng.shuffle(letters)
        face_letters = {i: letters[i] for i in range(6)}  # face_id -> letter

        rot_name = rng.choice(cfg["rot_pool"])
        rot_fn, rot_desc = _ROTATIONS[rot_name]

        # Initial pos_to_face: position i has face i.
        pos_to_face = list(range(6))
        new_pos_to_face = rot_fn(pos_to_face)

        # Visible positions are 0 (top), 2 (front), 4 (right).
        # Faces visible BEFORE: face IDs at positions [0, 2, 4] = [0, 2, 4]
        visible_before_face_ids = [pos_to_face[p] for p in _VISIBLE_INITIAL]
        # Faces visible AFTER: face IDs at positions [0, 2, 4] in new_pos_to_face
        visible_after_face_ids = [new_pos_to_face[p] for p in _VISIBLE_INITIAL]

        # Faces previously visible but now hidden:
        previously_visible_now_hidden = [f for f in visible_before_face_ids
                                          if f not in visible_after_face_ids]

        if len(previously_visible_now_hidden) != 1:
            # Some rotations may keep all visible faces visible (e.g., rotation
            # by 90 about an axis that aligns with a visible face, but actually
            # all our rotations should swap exactly one). Skip if not exactly 1.
            return None

        correct_face_id = previously_visible_now_hidden[0]
        correct_letter = face_letters[correct_face_id]

        # Generate distractors among the OTHER previously-visible faces (those
        # still visible after rotation).
        other_visible = [f for f in visible_before_face_ids
                         if f != correct_face_id]
        # If we want hidden-face distractors too (high level), include some.
        if cfg["include_hidden_distractors"]:
            hidden_face_ids = [f for f in range(6) if f not in visible_before_face_ids]
            distractor_pool = other_visible + hidden_face_ids
        else:
            distractor_pool = other_visible + [
                f for f in [1, 3, 5] if f != correct_face_id
            ]

        # Pick 3 distractor letters distinct from correct
        distractor_pool = [f for f in distractor_pool if f != correct_face_id]
        distractor_letters = list({face_letters[f] for f in distractor_pool})
        if len(distractor_letters) < 3:
            return None
        rng.shuffle(distractor_letters)
        distractor_letters = distractor_letters[:3]

        options = [correct_letter] + distractor_letters
        rng.shuffle(options)
        correct_idx = options.index(correct_letter)
        answer_letter = chr(ord("A") + correct_idx)

        # Build a list of (option_label, face_letter) so we can render the MCQ
        option_pairs = [(chr(ord("A") + i), opt) for i, opt in enumerate(options)]

        img = self._render(face_letters, rot_desc, option_pairs, rng)
        sidx = (self.seed or 0) % len(_QUESTION_TEMPLATES)
        question = _QUESTION_TEMPLATES[sidx].format(rot_desc=rot_desc)
        # 2026-05-04: simplified L0 (was 10% too-hard) — concise rule + key
        # insight that the FRONT-facing letter goes to LEFT (hidden) under
        # CW vertical rotation, which is the only rotation at L0. The image
        # also shows initial face letters so the model just needs to identify
        # the FRONT letter and pick that option.
        if int(self.parameter.get("level", 0)) <= 1:
            question += (
                " Be concise. The cube is rotated CW about the vertical axis "
                "(seen from above), so the FRONT face rotates to the LEFT and "
                "becomes hidden. Identify the letter currently on the FRONT "
                "face of the cube (the letter shown on the square front-facing "
                "rectangle in the image), find that letter among the options, "
                "and output its A/B/C/D label."
            )
        return question, answer_letter, img

    def _render(self, face_letters: Dict[int, str], rot_desc: str,
                option_pairs: List[Tuple[str, str]], rng) -> Image.Image:
        style = self._random_style()
        fig, (ax_cube, ax_opts) = plt.subplots(
            2, 1, figsize=(8, 8),
            gridspec_kw={"height_ratios": [1.5, 0.7]})
        fig.patch.set_facecolor(style["bg_color"])

        # Cube view
        ax_cube.set_facecolor(style["bg_color"])
        ax_cube.set_xlim(0, 6)
        ax_cube.set_ylim(0, 5)
        ax_cube.set_aspect("equal")
        ax_cube.axis("off")
        ax_cube.set_title("Cube (initial view)", fontsize=12, fontweight="bold")

        _draw_cube_isometric(ax_cube, 3.0, 2.0, 1.0, face_letters, style)
        # Note hidden faces
        _hidden_faces_legend(ax_cube, 3.0, 0.4, face_letters)

        # Options
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, 1)
        ax_opts.set_ylim(0, len(option_pairs) * 0.45)
        ax_opts.axis("off")
        ax_opts.set_title("Options", fontsize=11, fontweight="bold")
        for i, (lbl, face_letter) in enumerate(option_pairs):
            y = (len(option_pairs) - 1 - i) * 0.45 + 0.1
            ax_opts.text(0.05, y, f"({lbl}) {face_letter}",
                          fontsize=14, ha="left", va="center", color="#222")

        return self.fig_to_pil(fig, dpi=style["dpi"])
