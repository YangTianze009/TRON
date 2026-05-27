"""
Shape Symmetry Grouping QA environment.

Targets broader A4 coverage — attribute partitioning by SYMMETRY.

Shows 6 shapes.  Three have one kind of symmetry (e.g., reflection), three
do not (or have a different kind).  The model must identify the split
that organizes the shapes by their symmetry type.

Two symmetry modes:
- `reflection`: 3 shapes with >=1 reflection axis, 3 without
- `rotational`: 3 shapes with >=2-fold rotational symmetry, 3 without

At higher levels we add subtle cases like parallelograms (no reflection
symmetry but has 2-fold rotation) that trip up naive heuristics.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Shape renderers
# ------------------------------------------------------------------ #

def _poly(ax, verts, color, lw, fill=False, alpha=0.25):
    verts = list(verts) + [verts[0]]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    p = mpath.Path(verts, codes)
    fc = color if fill else "none"
    ax.add_patch(mpatches.PathPatch(
        p, facecolor=fc, edgecolor=color,
        linewidth=lw, alpha=(alpha if fill else 1.0), zorder=3))

def _draw_circle(ax, cx, cy, s, color, lw):
    ax.add_patch(mpatches.Circle(
        (cx, cy), s * 0.9, facecolor="none",
        edgecolor=color, linewidth=lw, zorder=3))

def _draw_square(ax, cx, cy, s, color, lw):
    _poly(ax, [(cx - s * 0.85, cy - s * 0.85),
               (cx + s * 0.85, cy - s * 0.85),
               (cx + s * 0.85, cy + s * 0.85),
               (cx - s * 0.85, cy + s * 0.85)],
          color, lw)

def _draw_equilateral_triangle(ax, cx, cy, s, color, lw):
    verts = []
    for i in range(3):
        a = math.pi / 2 + 2 * math.pi * i / 3
        verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    _poly(ax, verts, color, lw)

def _draw_regular_pentagon(ax, cx, cy, s, color, lw):
    verts = []
    for i in range(5):
        a = math.pi / 2 + 2 * math.pi * i / 5
        verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    _poly(ax, verts, color, lw)

def _draw_regular_hexagon(ax, cx, cy, s, color, lw):
    verts = []
    for i in range(6):
        a = 2 * math.pi * i / 6
        verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    _poly(ax, verts, color, lw)

def _draw_isoceles_triangle(ax, cx, cy, s, color, lw):
    verts = [
        (cx, cy + s * 0.9),
        (cx - s * 0.55, cy - s * 0.7),
        (cx + s * 0.55, cy - s * 0.7),
    ]
    _poly(ax, verts, color, lw)

def _draw_rectangle(ax, cx, cy, s, color, lw):
    _poly(ax, [(cx - s * 0.95, cy - s * 0.55),
               (cx + s * 0.95, cy - s * 0.55),
               (cx + s * 0.95, cy + s * 0.55),
               (cx - s * 0.95, cy + s * 0.55)],
          color, lw)

def _draw_ellipse(ax, cx, cy, s, color, lw):
    ax.add_patch(mpatches.Ellipse(
        (cx, cy), s * 1.85, s * 1.1,
        facecolor="none", edgecolor=color,
        linewidth=lw, zorder=3))

def _draw_plus(ax, cx, cy, s, color, lw):
    # Plus sign polygon (has 4 reflection axes + C4 rotational symmetry)
    arm = s * 0.32
    l = s * 0.9
    verts = [
        (cx - arm, cy - l), (cx + arm, cy - l), (cx + arm, cy - arm),
        (cx + l, cy - arm), (cx + l, cy + arm), (cx + arm, cy + arm),
        (cx + arm, cy + l), (cx - arm, cy + l), (cx - arm, cy + arm),
        (cx - l, cy + arm), (cx - l, cy - arm), (cx - arm, cy - arm),
    ]
    _poly(ax, verts, color, lw)

# Asymmetric shapes
def _draw_scalene_triangle(ax, cx, cy, s, color, lw):
    verts = [
        (cx - s * 0.9, cy - s * 0.6),
        (cx + s * 0.95, cy - s * 0.45),
        (cx - s * 0.3, cy + s * 0.9),
    ]
    _poly(ax, verts, color, lw)

def _draw_parallelogram(ax, cx, cy, s, color, lw):
    # Parallelogram: 2-fold rotational symmetry, NO reflection symmetry.
    verts = [
        (cx - s * 0.95, cy - s * 0.55),
        (cx + s * 0.65, cy - s * 0.55),
        (cx + s * 0.95, cy + s * 0.55),
        (cx - s * 0.65, cy + s * 0.55),
    ]
    _poly(ax, verts, color, lw)

def _draw_l_shape_solid(ax, cx, cy, s, color, lw):
    # L polygon (asymmetric; no symmetry at all).
    verts = [
        (cx - s * 0.85, cy - s * 0.85),
        (cx + s * 0.3, cy - s * 0.85),
        (cx + s * 0.3, cy - s * 0.15),
        (cx - s * 0.2, cy - s * 0.15),
        (cx - s * 0.2, cy + s * 0.85),
        (cx - s * 0.85, cy + s * 0.85),
    ]
    _poly(ax, verts, color, lw)

def _draw_spiral(ax, cx, cy, s, color, lw):
    # Spiral polyline (asymmetric).
    t = np.linspace(0, 3 * math.pi, 120)
    r = np.linspace(0.1 * s, 0.9 * s, 120)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t),
            color=color, linewidth=lw, zorder=3)

def _draw_comma(ax, cx, cy, s, color, lw):
    # Comma-like: half-ellipse plus a curved tail — clearly asymmetric.
    theta = np.linspace(math.pi * 0.1, math.pi * 1.9, 120)
    ax.plot(cx + s * 0.7 * np.cos(theta),
            cy + s * 0.6 * np.sin(theta) + s * 0.2,
            color=color, linewidth=lw, zorder=3)
    t2 = np.linspace(0, 1, 60)
    x = cx + s * 0.7 + t2 * s * 0.3
    y = cy + s * 0.2 - t2 * s * 0.8
    ax.plot(x, y, color=color, linewidth=lw, zorder=3)

def _draw_rhombus(ax, cx, cy, s, color, lw):
    # Rhombus: 2 reflection axes (vertical + horizontal) + 2-fold rotation.
    _poly(ax, [(cx, cy + s * 0.9),
               (cx + s * 0.65, cy),
               (cx, cy - s * 0.9),
               (cx - s * 0.65, cy)], color, lw)

def _draw_trapezoid(ax, cx, cy, s, color, lw):
    # Isoceles trapezoid: 1 reflection axis (vertical), no rotational symmetry.
    _poly(ax, [(cx - s * 0.9, cy - s * 0.6),
               (cx + s * 0.9, cy - s * 0.6),
               (cx + s * 0.5, cy + s * 0.6),
               (cx - s * 0.5, cy + s * 0.6)],
          color, lw)

def _draw_right_triangle(ax, cx, cy, s, color, lw):
    # Right triangle — generally asymmetric (unless isoceles).
    _poly(ax, [(cx - s * 0.85, cy - s * 0.85),
               (cx + s * 0.85, cy - s * 0.85),
               (cx - s * 0.85, cy + s * 0.85)],
          color, lw)

# ------------------------------------------------------------------ #
# Shape catalogue tagged by symmetry properties
# ------------------------------------------------------------------ #

# Each entry: (draw_fn, reflection_axes, rotation_order, asymmetric_bool)
#   reflection_axes: number of reflection symmetry axes (0 = none)
#   rotation_order:  smallest n such that n-fold rotation maps shape to self
#                    (1 = only identity, i.e. no rotational symmetry)
_SHAPE_CATALOGUE = {
    "circle":         (_draw_circle,              99, 99),  # infinite
    "square":         (_draw_square,               4, 4),
    "eq_triangle":    (_draw_equilateral_triangle, 3, 3),
    "reg_pentagon":   (_draw_regular_pentagon,     5, 5),
    "reg_hexagon":    (_draw_regular_hexagon,      6, 6),
    "iso_triangle":   (_draw_isoceles_triangle,    1, 1),
    "rectangle":      (_draw_rectangle,            2, 2),
    "ellipse":        (_draw_ellipse,              2, 2),
    "plus":           (_draw_plus,                 4, 4),
    "rhombus":        (_draw_rhombus,              2, 2),
    "trapezoid":      (_draw_trapezoid,            1, 1),
    # Asymmetric
    "scalene_tri":    (_draw_scalene_triangle,     0, 1),
    "parallelogram":  (_draw_parallelogram,        0, 2),
    "l_shape":        (_draw_l_shape_solid,        0, 1),
    "spiral":         (_draw_spiral,               0, 1),
    "comma":          (_draw_comma,                0, 1),
    "right_triangle": (_draw_right_triangle,       0, 1),
}

_GRID_CELLS = [
    (0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1),
]

class ShapeSymmetryGroupingQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "shape_symmetry_grouping"

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Return difficulty config for a given level (0-9)."""
        if level <= 0:
            return {"modes": ["reflection"], "n_per_group": 3,
                    "exclude_traps": True}
        if level == 1:
            return {"modes": ["reflection"], "n_per_group": 3,
                    "exclude_traps": True}
        if level == 2:
            return {"modes": ["reflection"], "n_per_group": 3,
                    "exclude_traps": False}
        if level == 3:
            return {"modes": ["reflection", "rotational"], "n_per_group": 3,
                    "exclude_traps": False}
        if level == 4:
            return {"modes": ["reflection", "rotational"], "n_per_group": 4,
                    "exclude_traps": False}
        if level == 5:
            return {"modes": ["reflection", "rotational"], "n_per_group": 4,
                    "exclude_traps": False}
        if level == 6:
            return {"modes": ["reflection", "rotational"], "n_per_group": 4,
                    "exclude_traps": False}
        if level == 7:
            return {"modes": ["reflection", "rotational"], "n_per_group": 5,
                    "exclude_traps": False}
        if level == 8:
            return {"modes": ["reflection", "rotational"], "n_per_group": 5,
                    "exclude_traps": False}
        # level >= 9
        return {"modes": ["reflection", "rotational"], "n_per_group": 5,
                "exclude_traps": False}

    _QUESTION_TEMPLATES = [
        "Group the {n} figures into two sets of {g} based on their {mode_text} symmetry: one set has {mode_text} symmetry and the other does not. Which of the following groupings is correct?\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "The image shows {n} shapes numbered 1-{n}. Split them into two groups of {g} by whether they possess {mode_text} symmetry. Which partition is correct?\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "Classify the {n} figures into two equal sets of {g}: those WITH {mode_text} symmetry and those WITHOUT. Select the correct grouping.\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "Which option correctly divides figures 1-{n} into two groups of {g} based on {mode_text} symmetry?\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "Partition the {n} figures (numbered 1-{n}) into two equal sets of {g} according to {mode_text} symmetry. Identify the correct partition.\n{opts}\nAnswer with a single letter A, B, C, or D.",
        "Sort the {n} figures by {mode_text} symmetry into two groups of {g}. Which option (A-D) is the correct sorting?\n{opts}\nAnswer with a single letter A, B, C, or D.",
    ]

    _TITLE_VARIANTS = [
        "{n} figures:",
        "Shape set ({n}):",
        "Shapes 1-{n}",
        "Figures",
        "Symmetry test",
        "Numbered shape grid",
        "{n} numbered figures",
    ]

    _DIVIDER_STYLES = ["{a} | {b}", "{a} vs {b}", "[{a}] / [{b}]",
                       "{{{a}}} & {{{b}}}", "({a}) ; ({b})"]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        mode = rng.choice(cfg["modes"])
        n_per_group = cfg["n_per_group"]

        for _ in range(40):
            result = self._try_generate(rng, sub_rng, level, cfg, mode, n_per_group)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, sub_rng, level, cfg, mode, n_per_group=3):
        has = []
        not_has = []
        for name, (_, refl, rot) in _SHAPE_CATALOGUE.items():
            if cfg["exclude_traps"]:
                if name == "parallelogram":
                    continue
            if mode == "reflection":
                hv = refl >= 1
            else:
                hv = rot >= 2
            (has if hv else not_has).append(name)

        if len(has) < n_per_group or len(not_has) < n_per_group:
            return None

        picks_has = rng.sample(has, n_per_group)
        picks_not = rng.sample(not_has, n_per_group)

        # At higher levels, INCLUDE the tricky parallelogram as part of the
        # "no reflection" group when mode is reflection (it has rotational
        # symmetry but no mirror) to teach the distinction.
        if level >= 4 and mode == "reflection" and rng.random() < 0.5:
            if "parallelogram" in not_has and "parallelogram" not in picks_not:
                picks_not[0] = "parallelogram"

        all_picks = picks_has + picks_not
        n_total = n_per_group * 2
        if len(set(all_picks)) != n_total:
            return None

        positions = list(range(n_total))
        rng.shuffle(positions)
        fig_at_pos = [None] * n_total
        for p, n in zip(positions[:n_per_group], picks_has):
            fig_at_pos[p] = (n, 1)
        for p, n in zip(positions[n_per_group:], picks_not):
            fig_at_pos[p] = (n, 0)

        g_has = sorted(p for p, (_, g) in enumerate(fig_at_pos) if g == 1)
        g_not = sorted(p for p, (_, g) in enumerate(fig_at_pos) if g == 0)
        correct = frozenset([frozenset(g_has), frozenset(g_not)])

        distractors = self._make_distractors(rng, correct, n_per_group)
        if len(distractors) < 3:
            return None

        options = [correct] + distractors[:3]
        rng.shuffle(options)
        correct_idx = options.index(correct)
        answer_letter = chr(ord("A") + correct_idx)

        title = sub_rng.choice(self._TITLE_VARIANTS).format(n=n_total)
        image = self._render(fig_at_pos, options, title=title, sub_rng=sub_rng)

        opt_lines = []
        for i, part in enumerate(options):
            parts = sorted(part, key=lambda s: min(s))
            left = ", ".join(str(p + 1) for p in sorted(parts[0]))
            right = ", ".join(str(p + 1) for p in sorted(parts[1]))
            opt_lines.append(
                f"  {chr(ord('A') + i)}) {{{left}}} | {{{right}}}")
        opts_text = "\n".join(opt_lines)
        # apply chosen divider style (cosmetic on prompt, no leakage)
        # (kept as the canonical "{left} | {right}" so MCQ remains parseable)

        mode_text = "reflection (mirror)" if mode == "reflection" else "rotational"
        q_template = sub_rng.choice(self._QUESTION_TEMPLATES)
        question = q_template.format(n=n_total, g=n_per_group,
                                     mode_text=mode_text, opts=opts_text)
        self._primary_complexity_feature = n_total + level * 2
        return question, answer_letter, image

    def _make_distractors(self, rng, correct, n_per_group=3):
        seen = {correct}
        out = []
        correct_groups = list(correct)
        gA = sorted(list(correct_groups[0]))
        gB = sorted(list(correct_groups[1]))

        attempts = 0
        while len(out) < 3 and attempts < 200:
            attempts += 1
            # Swap 1 or 2 elements across
            swap_count = 1 if attempts < 60 else 2
            new_A = list(gA)
            new_B = list(gB)
            for _ in range(swap_count):
                a_i = rng.randint(0, n_per_group - 1)
                b_i = rng.randint(0, n_per_group - 1)
                new_A[a_i], new_B[b_i] = new_B[b_i], new_A[a_i]
            part = frozenset([frozenset(new_A), frozenset(new_B)])
            if part in seen:
                continue
            seen.add(part)
            out.append(part)
        return out

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, fig_at_pos, options, title="Figures", sub_rng=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        _line_colors = ["#2c3e50", "#1a5276", "#7b241c", "#0d6efd",
                        "#198754", "#6c3483", "#1b4f72"]
        line_color = (sub_rng or self._rng).choice(_line_colors)
        lw = 1.8 + (sub_rng or self._rng).random() * 1.0

        n_total = len(fig_at_pos)
        ncols = n_total // 2
        nrows = 2

        fig_w = max(7.5, 2.0 + 1.25 * ncols) * sc
        fig = plt.figure(figsize=(fig_w, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.4], hspace=0.25)
        ax_fig = fig.add_subplot(gs[0])
        ax_opt = fig.add_subplot(gs[1])

        ax_fig.set_xlim(0, ncols)
        ax_fig.set_ylim(0, nrows)
        ax_fig.set_aspect("equal")
        ax_fig.axis("off")
        ax_fig.set_title(title,
                         fontsize=fs + 2, fontweight="bold", pad=6,
                         fontfamily=ff)

        for pos, (name, _) in enumerate(fig_at_pos):
            col = pos % ncols
            row = pos // ncols
            cx = col + 0.5
            cy = (nrows - 1 - row) + 0.5
            rect = mpatches.FancyBboxPatch(
                (cx - 0.44, cy - 0.44), 0.88, 0.88,
                boxstyle="round,pad=0.02",
                facecolor=style["bg_color"], edgecolor="#bdc3c7",
                linewidth=1.2, zorder=1)
            ax_fig.add_patch(rect)
            draw_fn = _SHAPE_CATALOGUE[name][0]
            draw_fn(ax_fig, cx, cy, 0.33, line_color, lw)
            ax_fig.text(cx, cy - 0.5, str(pos + 1),
                         fontsize=fs + 1, fontweight="bold",
                         ha="center", va="top", color="#2c3e50",
                         fontfamily=ff)

        ax_opt.set_xlim(0, 1)
        ax_opt.set_ylim(0, 1)
        ax_opt.axis("off")
        n_opts = len(options)
        for i, part in enumerate(options):
            parts = sorted(part, key=lambda s: min(s))
            left = ", ".join(str(p + 1) for p in sorted(parts[0]))
            right = ", ".join(str(p + 1) for p in sorted(parts[1]))
            y = 0.85 - i * (0.85 / n_opts)
            ax_opt.text(0.02, y, f"{chr(ord('A') + i)}. ",
                         fontsize=fs + 2, fontweight="bold",
                         color="#2c3e50", va="center", fontfamily=ff)
            ax_opt.text(0.08, y,
                         f"{{{left}}}  |  {{{right}}}",
                         fontsize=fs + 1, color="#2c3e50",
                         va="center", fontfamily=ff)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check", exist_ok=True)
    env = ShapeSymmetryGroupingQA()
    for level in [0, 3, 6]:
        for seed in range(3):
            ok = env.generate(seed=seed * 10 + level,
                              parameter={"level": level})
            if not ok:
                print(f"FAILED seed={seed} level={level}")
                continue
            img = env.render()
            img.save(f"/tmp/env_check/shape_symmetry_grouping_seed{seed}_L{level}.png")
            print(f"seed={seed} level={level}")
            print("Q:", env.get_instruction().splitlines()[0])
            print("A:", env._answer)
            print()
