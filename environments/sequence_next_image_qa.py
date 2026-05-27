"""
Sequence Next Image QA (Task B, multi-image P5 capability).

Show 3-4 images in a progression and 4 MCQ options for the 5th.
L0: obvious size/color progression. L9: compound rule.

Difficulty axes:
  A) Rule complexity (single attribute -> two attributes -> compound)
  B) Number of sequence images (3 -> 4)
  C) Distractor closeness
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import BytesIO
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

_SHAPE_TYPES = ["circle", "square", "triangle", "hexagon", "pentagon", "diamond", "star"]
_COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#d35400", "#16a085",
           "#8e44ad", "#2980b9"]

# Natural-hue-ordered palette for the "color" rule: monotonically shifts through
# the rainbow so the next colour is predictable from the visible trend.
_COLORS_HUE_ORDERED = [
    "#e74c3c",  # red
    "#e67e22",  # orange
    "#f1c40f",  # yellow
    "#2ecc71",  # green
    "#1abc9c",  # teal
    "#3498db",  # blue
    "#9b59b6",  # purple
]

_QUESTION_TEMPLATES_SNI = [
    # --- Direct interrogative (4) ---
    "A sequence of {n} images is shown in the top row. Which of the 4 options (A, B, C, D) in the bottom row should come next? Answer with a single letter.",
    "The top row shows {n} images that follow a hidden rule. Which option (A-D) continues the pattern?",
    "In the figure, {n} images are arranged in order. Which of A, B, C, or D should come next?",
    "Looking at the {n}-image sequence, which candidate (A-D) completes the pattern?",
    # --- Imperative (4) ---
    "Examine the {n}-image progression in the top row. Pick the option (A-D) that completes the pattern. Answer with a single letter.",
    "Below the {n}-image sequence, four candidate images A-D are shown. Identify which one continues the pattern. Answer A, B, C, or D.",
    "Select the option that comes next in the {n}-image sequence. Reply A-D.",
    "Determine which of A-D should come after the {n} images shown. Single letter answer.",
    # --- Declarative / cloze (4) ---
    "The image that continues the {n}-image progression shown is labelled ___ (A-D).",
    "After the {n} images in the top row, the next in the pattern corresponds to option ___.",
    "Let X be the letter of the option that comes next after the {n}-image sequence. X = ?",
    "The correct continuation of the {n}-image progression is option ___ (A-D).",
    # --- Reasoning-prompt (4) ---
    "Examine the {n} images carefully and reason about the progression rule. Which of A-D should come next?",
    "Study the {n}-image sequence, figure out the hidden pattern, and pick the next image from A-D.",
    "Look at the {n} ordered images. What should come next? Reason about the trend and select one of (A)-(D).",
    "After identifying the rule governing the {n}-image sequence, choose the option (A-D) that continues it.",
]

class SequenceNextImageQA(StandaloneVisualEnv):
    ENV_NAME = "sequence_next_image"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Structural: L0..L4 use 3 frames; L5..L7 use 4; L8..L9 use 5.
        if level <= 4:
            n_seq = 3
        elif level <= 7:
            n_seq = 4
        else:
            n_seq = 5
        # 2026-04-26: rule_type schedule prioritizes EASY (color → distinct
        # color cycles) at L0/L1 since size is ambiguous to VL.
        if level <= 1:
            rule_type = "color"
        elif level <= 3:
            rule_type = "size"
        elif level <= 5:
            rule_type = "shape_rotate"
        else:
            rule_type = "compound"
        return {
            "n_seq": n_seq,
            "rule_type": rule_type,
            "distractor_closeness": min(0.9, 0.3 + level * 0.07),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level

        n_seq = cfg["n_seq"]
        rule = cfg["rule_type"]

        # Generate sequence based on rule
        shape = rng.choice(_SHAPE_TYPES)
        base_color_idx = rng.randint(0, len(_COLORS) - 1)
        base_size = 0.5

        sequence = []
        next_attrs = {}

        if rule == "size":
            # Size increases linearly
            increment = rng.choice([0.15, 0.2, 0.25])
            for i in range(n_seq + 1):
                sequence.append({
                    "shape": shape,
                    "color": _COLORS[base_color_idx],
                    "size": base_size + i * increment,
                    "rotation": 0,
                })
            next_attrs = sequence[-1]

        elif rule == "color":
            # Color steps through the hue-ordered palette (red->orange->yellow->
            # green->teal->blue->purple). This makes the "next colour" visually
            # predictable from the trend (vs an arbitrary index cycle).
            pal = _COLORS_HUE_ORDERED
            max_start = max(0, len(pal) - (n_seq + 1))
            base_color_idx = rng.randint(0, max_start)
            for i in range(n_seq + 1):
                sequence.append({
                    "shape": shape,
                    "color": pal[base_color_idx + i],
                    "size": base_size + 0.3,
                    "rotation": 0,
                })
            next_attrs = sequence[-1]

        elif rule == "shape_rotate":
            # Shape rotates by fixed angle
            rot_step = rng.choice([30, 45, 60, 90])
            for i in range(n_seq + 1):
                sequence.append({
                    "shape": shape,
                    "color": _COLORS[base_color_idx],
                    "size": base_size + 0.3,
                    "rotation": i * rot_step,
                })
            next_attrs = sequence[-1]

        else:  # compound
            # Both size AND color change (hue-ordered palette).
            increment = rng.choice([0.12, 0.15])
            pal = _COLORS_HUE_ORDERED
            max_start = max(0, len(pal) - (n_seq + 1))
            base_color_idx = rng.randint(0, max_start)
            for i in range(n_seq + 1):
                sequence.append({
                    "shape": shape,
                    "color": pal[base_color_idx + i],
                    "size": base_size + i * increment,
                    "rotation": 0,
                })
            next_attrs = sequence[-1]

        # Separate shown sequence and correct next
        shown = sequence[:n_seq]
        correct = sequence[n_seq]

        # Generate distractors. Each distractor must be visibly distinct from
        # the correct answer AND from every other distractor. For "size" tweaks
        # we enforce a delta >= 0.2 so the size change is obvious.
        distractors = []
        tweak_pool = ["color", "size", "rotation"]
        rng.shuffle(tweak_pool)
        for tweak in tweak_pool:
            d = dict(correct)
            if tweak == "color":
                candidates = [c for c in _COLORS if c != correct["color"]]
                d["color"] = rng.choice(candidates)
            elif tweak == "size":
                delta = rng.choice([-0.35, -0.25, 0.25, 0.35])
                d["size"] = max(0.2, correct["size"] + delta)
            else:  # rotation
                # For circles rotation is invisible; fall back to size tweak.
                if correct["shape"] == "circle":
                    d["size"] = max(0.2, correct["size"] + rng.choice(
                        [-0.35, 0.35]))
                else:
                    d["rotation"] = correct["rotation"] + rng.choice(
                        [-45, 45, 90])
            distractors.append(d)

        # Build options
        options = [correct] + distractors
        rng.shuffle(options)
        answer_idx = options.index(correct)
        answer_letter = chr(ord("A") + answer_idx)

        # Render sequence + options
        img = self._render_sequence(shown, options, cfg, rng)

        sidx = (self.seed or 0) % 16
        question = _QUESTION_TEMPLATES_SNI[sidx].format(n=n_seq)
        if level <= 2:
            rule = cfg.get("rule_type", "?")
            question += (
                f" Hint: examine the {n_seq} shown images. The rule applies "
                f"a CONSISTENT change between consecutive images "
                f"(here: {rule} rule — could be size scaling, color shift, "
                "shape rotation, etc.). Identify the change pattern, then "
                "apply it once more to predict the next image."
            )

        return question, answer_letter, img

    def _render_sequence(self, shown, options, cfg, rng):
        style = self._random_style()
        n_seq = len(shown)
        n_opt = len(options)
        # Top row must fit n_seq sequence frames AND one "?" frame.
        ncols = max(n_seq + 1, n_opt)

        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            title_kw = {"fontfamily": tbp["font_family"], "color": tbp["line_color"]}
            edge_override = tbp["line_color"]
            edge_lw_override = tbp["line_width"]
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            title_kw = {}
            edge_override = "#1a1a1a"
            edge_lw_override = 1.8
            dpi = style["dpi"]
        else:
            bg = "#ffffff"
            title_kw = {}
            edge_override = None
            edge_lw_override = None
            dpi = style["dpi"]

        def _draw():
            fig = plt.figure(figsize=(max(8, (n_seq + 1) * 2), 6))
            fig.patch.set_facecolor(bg)

            for i, attrs in enumerate(shown):
                ax = fig.add_subplot(2, ncols, i + 1)
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                ax.set_aspect("equal"); ax.axis("off")
                ax.set_facecolor(bg)
                ax.set_title(f"#{i+1}", fontsize=10, fontweight="bold", **title_kw)
                self._draw_shape(ax, attrs, edge=edge_override, lw=edge_lw_override)

            ax_q = fig.add_subplot(2, ncols, n_seq + 1)
            ax_q.set_xlim(-1.5, 1.5); ax_q.set_ylim(-1.5, 1.5)
            ax_q.set_aspect("equal"); ax_q.axis("off")
            ax_q.set_facecolor(bg)
            ax_q.set_title("next", fontsize=10, fontweight="bold", color="red")
            ax_q.text(0, 0, "?", fontsize=30, ha="center", va="center",
                     fontweight="bold", color="red")

            for i, attrs in enumerate(options):
                ax = fig.add_subplot(2, ncols, ncols + i + 1)
                ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
                ax.set_aspect("equal"); ax.axis("off")
                ax.set_facecolor(bg)
                letter = chr(ord("A") + i)
                ax.set_title(f"({letter})", fontsize=10, fontweight="bold", **title_kw)
                self._draw_shape(ax, attrs, edge=edge_override, lw=edge_lw_override)
            try: fig.tight_layout()
            except: pass
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=80, randomness=1.5):
                fig = _draw()
        else:
            fig = _draw()
        return self.fig_to_pil(fig, dpi=dpi)

    def _draw_shape(self, ax, attrs, edge=None, lw=None):
        shape = attrs["shape"]
        color = attrs["color"]
        size = attrs["size"]
        rot = math.radians(attrs.get("rotation", 0))
        ec = edge if edge is not None else "black"
        lwv = lw if lw is not None else 1.5

        if shape == "circle":
            patch = plt.Circle((0, 0), size, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)
        elif shape == "square":
            s = size
            pts = [(-s, -s), (s, -s), (s, s), (-s, s)]
            if rot != 0:
                pts = [(x * math.cos(rot) - y * math.sin(rot),
                        x * math.sin(rot) + y * math.cos(rot)) for x, y in pts]
            patch = plt.Polygon(pts, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)
        elif shape == "triangle":
            pts = [(0, size), (-size, -size * 0.6), (size, -size * 0.6)]
            if rot != 0:
                pts = [(x * math.cos(rot) - y * math.sin(rot),
                        x * math.sin(rot) + y * math.cos(rot)) for x, y in pts]
            patch = plt.Polygon(pts, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)
        elif shape == "hexagon":
            pts = [(size * math.cos(math.radians(60 * i) + rot),
                    size * math.sin(math.radians(60 * i) + rot)) for i in range(6)]
            patch = plt.Polygon(pts, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)
        elif shape == "pentagon":
            pts = [(size * math.cos(math.radians(72 * i + 90) + rot),
                    size * math.sin(math.radians(72 * i + 90) + rot)) for i in range(5)]
            patch = plt.Polygon(pts, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)
        elif shape == "diamond":
            pts = [(0, size), (size * 0.7, 0), (0, -size), (-size * 0.7, 0)]
            if rot != 0:
                pts = [(x * math.cos(rot) - y * math.sin(rot),
                        x * math.sin(rot) + y * math.cos(rot)) for x, y in pts]
            patch = plt.Polygon(pts, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)
        elif shape == "star":
            pts = []
            for i in range(10):
                ang = math.radians(36 * i + 90) + rot
                r_use = size if i % 2 == 0 else size * 0.45
                pts.append((r_use * math.cos(ang), r_use * math.sin(ang)))
            patch = plt.Polygon(pts, fc=color, ec=ec, lw=lwv, alpha=0.85)
            ax.add_patch(patch)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskb"
    os.makedirs(out_dir, exist_ok=True)
    env = SequenceNextImageQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[sequence_next_image L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"sequence_next_image_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[sequence_next_image L{level} s{s}] A={env._answer}")
