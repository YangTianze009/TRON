"""
Geometric Transform QA.

Translation-MCQ environment in textbook style. Asks the model to compute the
new coordinates of a labeled point or vertex after a (compound) translation
on a 2D coordinate grid, then pick the correct answer letter.

Difficulty axes:
  L0/L1 -- Single-point single-translation, named lattice points A-G on a
           5x5 (or 6x6) grid. 4-option MCQ where the destination point is
           one of the labeled grid points.
  L2/L4 -- Triangle ABC translated by (dx,dy); pick the new coords of
           one labeled vertex (4-option coord-tuple MCQ + "No correct answer").
  L5/L7 -- Compound 2-step translation "(dx1, dy1) then (dx2, dy2)".
           4-option coord-tuple MCQ + 1 "No correct answer".
  L8/L9 -- Tight off-by-one distractors in either coordinate; 40% of
           problems have GT = "No correct answer" (true coords not in
           options).

Question phrasing matches textbook style:
  - "If point C is moved one grid to the left, the resulting point is ( )"
  - "Triangle ABC is translated 2 units down and then 1 unit to the right.
     What are the coordinates of A' after the translation?"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ====================================================================== #
# Helpers
# ====================================================================== #

def _fmt_coord_tuple(x: int, y: int) -> str:
    """Format coords as '(x, y)' with single space after comma (textbook style)."""
    return f"({x}, {y})"


def _coord_str_to_tuple(s: str) -> Optional[Tuple[int, int]]:
    """Parse '(x, y)' or 'x,y' into (x, y) ints, or None on failure."""
    s = s.strip()
    s = s.replace("(", "").replace(")", "").replace(" ", "")
    if "," not in s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        try:
            return int(round(float(parts[0]))), int(round(float(parts[1])))
        except ValueError:
            return None


def _step_phrase(dx: int, dy: int) -> str:
    """Convert (dx, dy) to a textbook-style phrase like
    'left 2 units' or 'right 1 unit and up 3 units'."""
    parts = []
    if dx != 0:
        if dx < 0:
            parts.append(f"left {abs(dx)} unit{'s' if abs(dx) != 1 else ''}")
        else:
            parts.append(f"right {dx} unit{'s' if dx != 1 else ''}")
    if dy != 0:
        if dy < 0:
            parts.append(f"down {abs(dy)} unit{'s' if abs(dy) != 1 else ''}")
        else:
            parts.append(f"up {dy} unit{'s' if dy != 1 else ''}")
    if not parts:
        return "0 units (no movement)"
    if len(parts) == 1:
        return parts[0]
    return parts[0] + " and " + parts[1]


def _step_phrase_compound(dx: int, dy: int) -> str:
    """Step phrasing for compound translations (no 'and' joiner needed)."""
    parts = []
    if dx != 0:
        if dx < 0:
            parts.append(f"left {abs(dx)} unit{'s' if abs(dx) != 1 else ''}")
        else:
            parts.append(f"right {dx} unit{'s' if dx != 1 else ''}")
    if dy != 0:
        if dy < 0:
            parts.append(f"down {abs(dy)} unit{'s' if abs(dy) != 1 else ''}")
        else:
            parts.append(f"up {dy} unit{'s' if dy != 1 else ''}")
    if not parts:
        return "0 units"
    if len(parts) == 1:
        return parts[0]
    return parts[0] + " then " + parts[1]


def _generate_off_by_one_distractors(
    correct: Tuple[int, int], rng: random.Random, k: int,
) -> List[Tuple[int, int]]:
    """Generate up to k unique distractor coord tuples that are 'tight':
    off by 1 in exactly one axis, sign-flip in one axis, or swap of (x, y).
    """
    cx, cy = correct
    candidates = [
        (cx + 1, cy),
        (cx - 1, cy),
        (cx, cy + 1),
        (cx, cy - 1),
        (cx + 1, cy + 1),
        (cx - 1, cy - 1),
        (cx + 1, cy - 1),
        (cx - 1, cy + 1),
        (cx + 2, cy),
        (cx - 2, cy),
        (cx, cy + 2),
        (cx, cy - 2),
        (-cx, cy),
        (cx, -cy),
        (cy, cx),  # swap
    ]
    rng.shuffle(candidates)
    seen = set()
    seen.add((cx, cy))
    out = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _generate_loose_distractors(
    correct: Tuple[int, int], rng: random.Random, k: int,
) -> List[Tuple[int, int]]:
    """Looser distractors for low levels: ±1, ±2 in each axis or sign flips."""
    cx, cy = correct
    candidates = []
    for dx in (-2, -1, 1, 2):
        candidates.append((cx + dx, cy))
        candidates.append((cx, cy + dx))
    for dx in (-1, 1):
        for dy in (-1, 1):
            candidates.append((cx + dx, cy + dy))
    candidates.append((-cx, cy))
    candidates.append((cx, -cy))
    candidates.append((cy, cx))
    rng.shuffle(candidates)
    seen = {(cx, cy)}
    out = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= k:
            break
    return out


# ====================================================================== #
# Main env class
# ====================================================================== #

class GeometricTransformQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "geometric_transform"

    QUESTION_TYPES = [
        "named_point_l01",        # L0/L1 -- single point translation, named lattice
        "triangle_l24",           # L2-L4 -- triangle vertex translation
        "compound_l57",           # L5-L7 -- compound 2-step translation
        "compound_tight_l89",     # L8/L9 -- tight distractors + 40% trap
    ]

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "qtype": "named_point_l01",
                "n_options": 4,
                "trap_rate": 0.0,
                "max_step": 2,  # small moves on small grid
            }
        if level <= 4:
            return {
                "qtype": "triangle_l24",
                "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.20,
                "max_step": 3,
                "tight_distractors": False,
            }
        if level <= 7:
            return {
                "qtype": "compound_l57",
                "n_options": 5,
                "trap_rate": 0.20,
                "max_step": 3,
                "tight_distractors": False,
            }
        # L8/L9
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (env still saturated 97.5%).
        return {
            "qtype": "compound_tight_l89",
            "n_options": 5,
            "trap_rate": 0.50,
            "max_step": 4,
            "tight_distractors": True,
        }

    def level_to_complexity(self, level: int) -> float:
        return float(level)

    # ------------------------------------------------------------------ #
    # Generate dispatcher
    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 717)
        qtype = cfg["qtype"]

        for _ in range(20):
            if qtype == "named_point_l01":
                result = self._try_generate_named_point(cfg, sub_rng)
            elif qtype == "triangle_l24":
                result = self._try_generate_triangle(cfg, sub_rng,
                                                       compound=False)
            elif qtype == "compound_l57":
                result = self._try_generate_triangle(cfg, sub_rng,
                                                       compound=True)
            else:  # compound_tight_l89
                result = self._try_generate_triangle(cfg, sub_rng,
                                                       compound=True)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # L0/L1 -- named lattice point single translation
    # Uses textbook phrasing: "If point C is moved one grid to the left,
    # the resulting point is ( ) -- A. E; B. F; C. G; D. No correct answer"
    # ------------------------------------------------------------------ #
    def _try_generate_named_point(self, cfg, rng):
        # Place 7 named points on a small grid (5x5 or 6x6).
        grid_size = rng.choice([5, 6])
        # Generate distinct lattice points in [-grid_size//2, grid_size//2]
        half = grid_size // 2
        coords_pool = [(x, y) for x in range(-half, half + 1)
                       for y in range(-half, half + 1)]
        rng.shuffle(coords_pool)
        # Need at least 7 distinct points
        if len(coords_pool) < 7:
            return None
        chosen = coords_pool[:7]
        # Assign labels A-G to chosen points
        labels = ["A", "B", "C", "D", "E", "F", "G"]
        pt_by_label = dict(zip(labels, chosen))
        coord_to_label = {pt: lbl for lbl, pt in pt_by_label.items()}

        # Pick the source point and the move direction
        src_label = rng.choice(labels)
        sx, sy = pt_by_label[src_label]
        # Valid moves: 1-2 units in one cardinal direction
        max_step = cfg["max_step"]
        # Build candidate moves
        move_choices = []
        for d in range(1, max_step + 1):
            move_choices.append((-d, 0, "left"))
            move_choices.append((d, 0, "right"))
            move_choices.append((0, d, "up"))
            move_choices.append((0, -d, "down"))
        rng.shuffle(move_choices)

        # We want destination to land on ANOTHER labeled point so that the
        # answer is one of the lettered options.
        chosen_move = None
        dest_pt = None
        for dx, dy, dir_word in move_choices:
            d = (sx + dx, sy + dy)
            if d in coord_to_label and coord_to_label[d] != src_label:
                chosen_move = (dx, dy, dir_word)
                dest_pt = d
                break
        if chosen_move is None:
            return None

        dx, dy, dir_word = chosen_move
        dest_label = coord_to_label[dest_pt]
        # Build the question phrase, mirroring "If point C is moved one grid
        # to the left, the resulting point is ( )". For up/down we say
        # "moved one grid up" (no "to the") since "to the up" is awkward.
        units = "grid" if abs(dx + dy) == 1 else "grids"
        steps_word = abs(dx + dy)
        steps_str = "one" if steps_word == 1 else str(steps_word)
        if dir_word in ("left", "right"):
            phrase = f"{steps_str} {units} to the {dir_word}"
        else:  # up / down
            phrase = f"{steps_str} {units} {dir_word}"
        question_stem = (
            f"As shown in the diagram, if point {src_label} is moved "
            f"{phrase}, the resulting point is (    )."
        )

        # Build options: 4-option (3 distractor labels + "D. No correct answer")
        n_options = cfg["n_options"]  # 4
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]
        # Distractor labels: pick from labels excluding src and dest
        distractor_labels = [l for l in labels
                             if l != src_label and l != dest_label]
        rng.shuffle(distractor_labels)
        # 3 letter distractors + 1 "No correct answer"  → 4 options total
        opt_distractors = distractor_labels[: n_options - 2]
        if len(opt_distractors) < n_options - 2:
            return None
        # Decide trap (always 0% at L0/L1 per cfg)
        use_trap = (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Should never hit at L0/L1; defensive
            options_text = opt_distractors[:]
            options_text.append("No correct answer")
            correct_letter = last_letter
        else:
            options_text = [dest_label] + opt_distractors[: n_options - 2]
            rng.shuffle(options_text)
            options_text.append("No correct answer")
            correct_letter = opt_letters[options_text.index(dest_label)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options_text[i]}"
            for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{question_stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        # Render the labeled grid
        image = self._draw_named_grid(
            pt_by_label, src_label, dx, dy,
            grid_half=half, rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L2-L4 (single translation triangle) and L5-L9 (compound) shared
    # backbone. Triangle ABC is drawn on a grid; question asks for new
    # coords of one labeled vertex after the (compound) translation.
    # ------------------------------------------------------------------ #
    def _try_generate_triangle(self, cfg, rng, compound: bool):
        max_step = cfg["max_step"]
        # Sample triangle vertices on integer grid in a window.
        cr = (-4, 4)
        for _ in range(30):
            pts = [
                (rng.randint(*cr), rng.randint(*cr))
                for _ in range(3)
            ]
            # Non-degenerate (area > 0)
            (x1, y1), (x2, y2), (x3, y3) = pts
            area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area2 >= 4:
                break
        else:
            return None
        labels = ["A", "B", "C"]
        # Pick the target vertex
        target_idx = rng.randint(0, 2)
        target_label = labels[target_idx]

        # Build translation step(s)
        if compound:
            # 2 steps; each step has nonzero (dx, dy) total or just one axis
            step_pool = []
            for ddx in range(-max_step, max_step + 1):
                for ddy in range(-max_step, max_step + 1):
                    if (ddx, ddy) != (0, 0):
                        step_pool.append((ddx, ddy))
            rng.shuffle(step_pool)
            step1 = step_pool[0]
            step2 = step_pool[1]
            steps = [step1, step2]
        else:
            step_pool = []
            for ddx in range(-max_step, max_step + 1):
                for ddy in range(-max_step, max_step + 1):
                    if (ddx, ddy) != (0, 0):
                        step_pool.append((ddx, ddy))
            rng.shuffle(step_pool)
            steps = [step_pool[0]]

        # Apply translation to all vertices, get new coords
        total_dx = sum(s[0] for s in steps)
        total_dy = sum(s[1] for s in steps)
        sx, sy = pts[target_idx]
        true_x = sx + total_dx
        true_y = sy + total_dy
        true_coords = (true_x, true_y)

        n_options = cfg["n_options"]  # 5
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]

        use_trap = (rng.random() < cfg["trap_rate"])
        # Generate distractors
        if cfg["tight_distractors"]:
            distractor_pool = _generate_off_by_one_distractors(
                true_coords, rng, k=n_options + 2,
            )
        else:
            distractor_pool = _generate_loose_distractors(
                true_coords, rng, k=n_options + 2,
            )

        if len(distractor_pool) < n_options:
            return None

        if use_trap:
            # All 4 (or n_options - 1) numeric options are distractors;
            # true_coords NOT included; correct = last letter ("No correct").
            n_numeric = n_options - 1  # one slot reserved for "No correct"
            numeric_options = distractor_pool[:n_numeric]
            # Shuffle for randomness
            rng.shuffle(numeric_options)
            option_texts = [_fmt_coord_tuple(*c) for c in numeric_options]
            option_texts.append("No correct answer")
            correct_letter = last_letter
        else:
            # 1 truth + (n_options - 2) distractors + "No correct"
            n_distractors = n_options - 2
            numeric_options = [true_coords] + distractor_pool[:n_distractors]
            rng.shuffle(numeric_options)
            option_texts = [_fmt_coord_tuple(*c) for c in numeric_options]
            option_texts.append("No correct answer")
            true_idx = numeric_options.index(true_coords)
            correct_letter = opt_letters[true_idx]

        # Build question
        if compound:
            step1, step2 = steps
            step1_phrase = _step_phrase_compound(*step1)
            step2_phrase = _step_phrase_compound(*step2)
            stem = (
                f"As shown in the diagram, triangle ABC is translated "
                f"{step1_phrase}, and then translated {step2_phrase}. "
                f"What are the coordinates of point {target_label}' "
                f"(the image of vertex {target_label}) after the translation?"
            )
        else:
            (sdx, sdy) = steps[0]
            step_phrase = _step_phrase(sdx, sdy)
            stem = (
                f"As shown in the diagram, triangle ABC is translated "
                f"{step_phrase}. What are the coordinates of point "
                f"{target_label}' (the image of vertex {target_label}) "
                f"after the translation?"
            )

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {option_texts[i]}"
            for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        # Render -- show only original triangle + grid (no translated triangle,
        # avoid leaking the answer)
        image = self._draw_triangle_grid(
            pts, labels, target_idx, rng=rng,
        )
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_named_grid(self, pt_by_label: Dict[str, Tuple[int, int]],
                         src_label: str, dx: int, dy: int,
                         grid_half: int, rng: random.Random) -> Image.Image:
        """Draw 7 named lattice points A-G on a coordinate grid. The source
        point gets a highlighted marker."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)

        # Draw the integer grid
        ax.set_xticks(range(-grid_half, grid_half + 1))
        ax.set_yticks(range(-grid_half, grid_half + 1))
        ax.grid(True, alpha=0.45, linestyle="--", linewidth=0.6)
        ax.set_xlim(-grid_half - 0.5, grid_half + 0.5)
        ax.set_ylim(-grid_half - 0.5, grid_half + 0.5)
        ax.set_aspect("equal")
        ax.axhline(0, color="#555", linewidth=0.7, alpha=0.7)
        ax.axvline(0, color="#555", linewidth=0.7, alpha=0.7)

        for label, (px, py) in pt_by_label.items():
            is_src = (label == src_label)
            color = palette[0] if is_src else "#222"
            ax.plot(px, py, "o", color=color, markersize=10 if is_src else 6,
                    markerfacecolor=color, markeredgecolor="black",
                    markeredgewidth=1.0, zorder=5)
            # Place label slightly offset
            ax.annotate(
                label,
                xy=(px, py),
                xytext=(px + 0.20, py + 0.20),
                fontsize=14,
                fontweight="bold",
                color=color,
                zorder=6,
            )

        ax.tick_params(axis="both", labelsize=9)
        ax.set_title("Named lattice points on coordinate grid",
                     fontsize=style["font_size_base"] + 1)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_triangle_grid(self, pts: List[Tuple[int, int]],
                            labels: List[str], target_idx: int,
                            rng: random.Random) -> Image.Image:
        """Draw labeled triangle on a coordinate grid (axes + grid lines)."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)

        # Compute grid bounds based on triangle pts (extra margin for labels)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = 2
        xmin = min(xs) - margin
        xmax = max(xs) + margin
        ymin = min(ys) - margin
        ymax = max(ys) + margin
        ax.set_xticks(range(xmin, xmax + 1))
        ax.set_yticks(range(ymin, ymax + 1))
        ax.grid(True, alpha=0.40, linestyle="--", linewidth=0.5)
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.5, ymax + 0.5)
        ax.set_aspect("equal")
        ax.axhline(0, color="#444", linewidth=0.7, alpha=0.7)
        ax.axvline(0, color="#444", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Draw the triangle filled + edged
        poly = MplPolygon(
            pts, closed=True, facecolor=palette[0], alpha=0.30,
            edgecolor=palette[0], linewidth=2, zorder=3,
        )
        ax.add_patch(poly)

        # Label each vertex
        for i, (x, y) in enumerate(pts):
            is_target = (i == target_idx)
            ax.plot(x, y, "o", color=palette[0],
                    markersize=12 if is_target else 7,
                    markerfacecolor=palette[0],
                    markeredgecolor="black",
                    markeredgewidth=1.2, zorder=5)
            label_text = labels[i]
            if is_target:
                # Mark target vertex more prominently
                label_text = f"{labels[i]} (target)"
            ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(x + 0.30, y + 0.30),
                fontsize=14 if not is_target else 13,
                fontweight="bold",
                color="black",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=palette[0], alpha=0.85)
                if is_target else None,
            )

        ax.tick_params(axis="both", labelsize=9)
        ax.set_title("Triangle ABC on coordinate grid",
                     fontsize=style["font_size_base"] + 1)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
