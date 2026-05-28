"""
Visual Analogy Raven 3x3 QA environment (batch 2 Part B, 2026-04-14).

Goal: 3x3 Raven-style matrix completion. 8 cells filled, bottom-right is
the unknown. The task is to pick the 9th cell from 4 options. Targets
a puzzle benchmark analogical, visual-perception IQ_Test, a puzzle benchmark.

Difficulty axes:
  A) Pattern C n_rules_composed (1..3).
  B) rule_salience — single attribute at L0, multi-attribute at L≥4.
  C) Pattern B distractor_similarity — random vs N-1 rule satisfied.

Format: 4-way MCQ (letter).
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

_SHAPES = ["circle", "square", "triangle", "hexagon", "diamond", "pentagon"]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c",
           "#e67e22", "#34495e", "#d35400", "#8e44ad"]
_SIZES = [0.25, 0.35, 0.48]

# BUGFIX 2026-04-24: drop "2x2" option layout. In 2x2 layout, the label
# "(A)"/"(B)" for top-row boxes was placed at y=cy-1.35 which falls INSIDE
# the bottom-row boxes (which span y in [0.1, 1.9] for cy=1). This caused
# natural readers to invert the label-to-box mapping. 1x4 is unambiguous.
_OPT_LAYOUTS_RAVEN = ["1x4"]
_FRAME_COLORS_RAVEN = ["#34495e", "#1a5276", "#7d3c98", "#0d4f6c", "#1b4f72", "#5d4037"]

class VisualAnalogyRaven3x3QA(StandaloneVisualEnv):
    ENV_NAME = "visual_analogy_raven_3x3"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Fix inversion: more rules = more visual contrast = easier for model.
        # L0: 3 rules (shape+color+size all change), L9: 1 subtle rule.
        if level <= 2:
            n_rules = 3
        elif level <= 5:
            n_rules = 2
        else:
            n_rules = 1
        return {
            "n_rules":            n_rules,
            "multi_attr":         level >= 3,
            "tight_distractors":  level >= 5,
            "n_options":          4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_rules"]

        for _ in range(30):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random,
                      cfg: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        n_rules = cfg["n_rules"]
        # Rules: shape cycle by column, color cycle by column, size cycle by column.
        # At L0: one rule applied to shape. At L6: 2 rules (shape + color).
        all_rules = ["shape_col", "color_col", "size_col", "shape_row"]
        rules = rng.sample(all_rules, n_rules)

        # Default attribute for each cell.
        base_shape = rng.choice(_SHAPES)
        base_color = rng.choice(_COLORS)
        base_size = rng.choice(_SIZES)

        matrix = [[{"shape": base_shape, "color": base_color,
                    "size": base_size}
                   for _ in range(3)] for _ in range(3)]

        # Pick sub-pools for cycling.
        shape_pool = rng.sample(_SHAPES, 3)
        color_pool = rng.sample(_COLORS, 3)
        size_pool = _SIZES[:]
        rng.shuffle(size_pool)

        for row in range(3):
            for col in range(3):
                cell = matrix[row][col]
                if "shape_col" in rules:
                    cell["shape"] = shape_pool[col]
                if "shape_row" in rules:
                    cell["shape"] = shape_pool[row]
                if "color_col" in rules:
                    cell["color"] = color_pool[col]
                if "size_col" in rules:
                    cell["size"] = size_pool[col]

        # The correct answer: matrix[2][2] after applying rules.
        correct = matrix[2][2]

        # Build distractors: variations that match N-1 rules.
        distractors: List[Dict] = []
        other_shapes = [s for s in _SHAPES if s != correct["shape"]]
        other_colors = [c for c in _COLORS if c != correct["color"]]
        other_sizes = [s for s in _SIZES if s != correct["size"]]

        if cfg["tight_distractors"]:
            d1 = {**correct, "shape": rng.choice(other_shapes)}
            d2 = {**correct, "color": rng.choice(other_colors)}
            d3 = {**correct, "size": rng.choice(other_sizes)}
            distractors = [d1, d2, d3]
        else:
            for _ in range(3):
                d = {
                    "shape": rng.choice(_SHAPES),
                    "color": rng.choice(_COLORS),
                    "size": rng.choice(_SIZES),
                }
                if d == correct:
                    d["shape"] = rng.choice(other_shapes)
                distractors.append(d)

        # Unique options
        options = [correct] + distractors
        # Dedup
        seen = []
        uniq = []
        for o in options:
            key = (o["shape"], o["color"], o["size"])
            if key not in seen:
                seen.append(key)
                uniq.append(o)
        if len(uniq) < 4:
            for s in _SHAPES:
                for c in _COLORS:
                    for sz in _SIZES:
                        key = (s, c, sz)
                        if key not in seen:
                            seen.append(key)
                            uniq.append({"shape": s, "color": c, "size": sz})
                            if len(uniq) >= 4:
                                break
                    if len(uniq) >= 4:
                        break
                if len(uniq) >= 4:
                    break
        options = uniq[:4]
        rng.shuffle(options)
        # Find correct position
        idx = None
        for i, o in enumerate(options):
            if (o["shape"] == correct["shape"] and
                    o["color"] == correct["color"] and
                    o["size"] == correct["size"]):
                idx = i
                break
        if idx is None:
            return None
        answer_letter = chr(ord("A") + idx)

        # Include n_rules in the question so seed-differentiation test
        # catches level changes in matrix content.
        q_stems = [
            f"The 3x3 matrix shows 8 figures following a hidden pattern. Which of the options A, B, C, D should replace the '?' in the bottom-right cell?",
            f"Look at the 3x3 grid of figures. The bottom-right cell is missing. Which option completes the pattern?",
            f"The matrix has 8 filled cells and one '?'. Based on the pattern in each row and column, which option fills the missing cell?",
            f"Examine the 3x3 visual matrix. Identify which option (A-D) correctly completes the bottom-right position.",
        ]
        question = (
            f"{rng.choice(q_stems)} "
            f"Answer with a single letter."
        )
        image = self._render(matrix, options, cfg)
        return question, answer_letter, image

    def _draw_shape(self, ax, cx, cy, cell: Dict):
        shape = cell["shape"]
        color = cell["color"]
        size = cell["size"]
        if shape == "circle":
            p = mpatches.Circle((cx, cy), size, facecolor=color,
                                 edgecolor="#1a1a1a", linewidth=1.2)
        elif shape == "square":
            p = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                    facecolor=color, edgecolor="#1a1a1a",
                                    linewidth=1.2)
        elif shape == "triangle":
            p = mpatches.Polygon([(cx, cy + size),
                                   (cx - size, cy - size),
                                   (cx + size, cy - size)],
                                  closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.2)
        elif shape == "hexagon":
            pts = [(cx + size * math.cos(math.radians(60 * i + 30)),
                    cy + size * math.sin(math.radians(60 * i + 30)))
                   for i in range(6)]
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.2)
        elif shape == "diamond":
            p = mpatches.Polygon([(cx, cy + size), (cx + size, cy),
                                   (cx, cy - size), (cx - size, cy)],
                                  closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.2)
        elif shape == "pentagon":
            pts = [(cx + size * math.cos(math.radians(72 * i + 90)),
                    cy + size * math.sin(math.radians(72 * i + 90)))
                   for i in range(5)]
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                  edgecolor="#1a1a1a", linewidth=1.2)
        else:
            p = mpatches.Circle((cx, cy), size, facecolor=color,
                                 edgecolor="#1a1a1a", linewidth=1.2)
        ax.add_patch(p)

    def _render(self, matrix, options, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        # Per-image randomization
        frame_color = self._rng.choice(_FRAME_COLORS_RAVEN)
        opt_layout = self._rng.choice(_OPT_LAYOUTS_RAVEN)

        fig = plt.figure(figsize=(10.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_m = fig.add_subplot(1, 2, 1)
        ax_o = fig.add_subplot(1, 2, 2)
        ax_m.set_aspect("equal")
        ax_o.set_aspect("equal")
        ax_m.axis("off")
        ax_o.axis("off")

        # Matrix
        for row in range(3):
            for col in range(3):
                cx = col * 2 + 1
                cy = (2 - row) * 2 + 1
                rect = mpatches.Rectangle((cx - 0.9, cy - 0.9), 1.8, 1.8,
                                           facecolor="#ffffff",
                                           edgecolor=frame_color,
                                           linewidth=1.5)
                ax_m.add_patch(rect)
                if row == 2 and col == 2:
                    ax_m.text(cx, cy, "?", fontsize=fs + 6,
                              fontweight="bold", ha="center", va="center",
                              family=ff, color="#c0392b")
                else:
                    self._draw_shape(ax_m, cx, cy, matrix[row][col])
        ax_m.set_xlim(-0.5, 6.5)
        ax_m.set_ylim(-0.5, 6.5)
        mat_title_pool = ["3x3 Matrix", "Pattern Matrix", "Visual Grid",
                          "Raven Matrix", "Figure Grid"]
        ax_m.set_title(self._rng.choice(mat_title_pool),
                       fontsize=fs + 1, family=ff)

        # Options - layout selectable: 2x2 grid or single 1x4 row
        for i, opt in enumerate(options):
            if opt_layout == "1x4":
                col = i
                row = 0
                cx = col * 2 + 1
                cy = 1
            else:
                col = i % 2
                row = i // 2
                cx = col * 2 + 1
                cy = (1 - row) * 2 + 1
            rect = mpatches.Rectangle((cx - 0.9, cy - 0.9), 1.8, 1.8,
                                       facecolor="#ffffff",
                                       edgecolor=frame_color,
                                       linewidth=1.5)
            ax_o.add_patch(rect)
            self._draw_shape(ax_o, cx, cy, opt)
            letter = chr(ord("A") + i)
            ax_o.text(cx - 0.7, cy - 1.35, f"({letter})",
                      fontsize=fs + 1, fontweight="bold",
                      family=ff, color="#1a1a1a")
        if opt_layout == "1x4":
            ax_o.set_xlim(-0.5, 8.5)
            ax_o.set_ylim(-2.0, 3.0)
        else:
            ax_o.set_xlim(-0.5, 4.5)
            ax_o.set_ylim(-2.0, 4.5)
        ax_o.set_title("Options", fontsize=fs + 1, family=ff)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b2b"
    os.makedirs(out_dir, exist_ok=True)
    env = VisualAnalogyRaven3x3QA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"visual_analogy_raven_3x3_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  Q (first 100): {env.get_instruction()[:100]}")
            print(f"  A: {env._answer}")
