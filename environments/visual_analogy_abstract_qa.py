"""
Visual Analogy Abstract QA environment.

Goal: train abstract analogical reasoning of the form A:B :: C:? where
A, B, C, D are all abstract synthetic shapes and the relation is a formal
transformation (rotate, scale, reflect, add hole, toggle fill, split,
etc.). This targets the A1 (visual analogy) + X5 (symbolic reasoning)
atomic capabilities and specifically addresses the VisualPuzzles /
analogical regression for the abstract-shape sub-type.

Unlike the existing `visual_analogy` env which mixes concrete attributes,
this env focuses on FORMAL transformations on pure abstract shapes:
  * rotate 90° / 180°
  * scale 2x / shrink
  * reflect (horizontal flip encoded as a marker)
  * add inner hole (concentric smaller shape)
  * toggle fill ↔ outline only
  * add radial stripes
  * translate element (e.g., dot moves to a corner)

At higher levels, two transformations are composed (rotate + add hole).

Difficulty (level 0-9):
  L0-2: SIMPLE single transformations (rotate 90, rotate 180, scale up,
        scale down, toggle fill). Distractors are visually distinct —
        apply a completely different simple rule.
  L3-5: MEDIUM single transformations (add hole, add stripes, dot rotate).
        Distractors are plausible: apply the WRONG medium rule.
  L6-9: COMPOSITE two-rule transformations (e.g., rotate + add hole).
        Distractors apply ONLY ONE of the two rules (the hardest kind of
        distractor — you have to verify BOTH transformations were applied).

Fix target: VisualPuzzles / analogical (abstract '==' matrices sub-type).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Shape representation
# ------------------------------------------------------------------ #

_SHAPES = ["circle", "triangle", "square", "pentagon", "hexagon", "star"]
_COLORS = ["#e74c3c", "#3498db", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]
_SIZES = [0.25, 0.35, 0.50]  # small, medium, large (fraction of cell)

def _polygon_path(cx, cy, n, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(n):
        a = off + 2 * math.pi * i / n
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _star_path(cx, cy, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(10):
        a = off + 2 * math.pi * i / 10
        r = size if i % 2 == 0 else size * 0.45
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 9 + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

class _AbstractShape:
    """State: shape + colour + size_idx + rotation + filled + has_hole +
    stripes + dot_pos (None / TL / TR / BL / BR)."""
    __slots__ = ("shape", "color", "size_idx", "rotation",
                 "filled", "has_hole", "stripes", "dot_pos")

    def __init__(self, shape="square", color="#3498db", size_idx=1,
                 rotation=0, filled=True, has_hole=False,
                 stripes=False, dot_pos=None):
        self.shape = shape
        self.color = color
        self.size_idx = size_idx
        self.rotation = rotation % 360
        self.filled = filled
        self.has_hole = has_hole
        self.stripes = stripes
        self.dot_pos = dot_pos  # None or one of "TL", "TR", "BL", "BR"

    def copy(self):
        return _AbstractShape(self.shape, self.color, self.size_idx,
                              self.rotation, self.filled, self.has_hole,
                              self.stripes, self.dot_pos)

    def __eq__(self, other):
        if not isinstance(other, _AbstractShape):
            return False
        return (self.shape == other.shape and self.color == other.color and
                self.size_idx == other.size_idx and
                self.rotation == other.rotation and
                self.filled == other.filled and
                self.has_hole == other.has_hole and
                self.stripes == other.stripes and
                self.dot_pos == other.dot_pos)

    def __hash__(self):
        return hash((self.shape, self.color, self.size_idx, self.rotation,
                     self.filled, self.has_hole, self.stripes, self.dot_pos))

    def draw(self, ax, cx, cy, cell_size):
        size_val = _SIZES[self.size_idx] * cell_size

        # Build path for the main shape (needed for hole/stripe clipping).
        if self.shape == "circle":
            main_path = None
        elif self.shape == "star":
            main_path = _star_path(cx, cy, size_val, self.rotation)
        else:
            n = {"triangle": 3, "square": 4, "pentagon": 5,
                 "hexagon": 6}[self.shape]
            main_path = _polygon_path(cx, cy, n, size_val, self.rotation)

        # Main face
        fc = self.color if self.filled else "none"
        lw = 2.0 if not self.filled else 1.5

        if self.shape == "circle":
            p = mpatches.Circle((cx, cy), size_val,
                                facecolor=fc, edgecolor="#2c3e50",
                                linewidth=lw, zorder=3)
            ax.add_patch(p)
        else:
            p = mpatches.PathPatch(main_path, facecolor=fc,
                                   edgecolor="#2c3e50", linewidth=lw,
                                   zorder=3)
            ax.add_patch(p)

        # Hole: draw a white concentric shape of half size
        if self.has_hole:
            inner = size_val * 0.45
            if self.shape == "circle":
                ph = mpatches.Circle((cx, cy), inner, facecolor="#ffffff",
                                     edgecolor="#2c3e50", linewidth=1.2,
                                     zorder=4)
                ax.add_patch(ph)
            elif self.shape == "star":
                ph_path = _star_path(cx, cy, inner, self.rotation)
                ph = mpatches.PathPatch(ph_path, facecolor="#ffffff",
                                        edgecolor="#2c3e50", linewidth=1.2,
                                        zorder=4)
                ax.add_patch(ph)
            else:
                n = {"triangle": 3, "square": 4, "pentagon": 5,
                     "hexagon": 6}[self.shape]
                ph_path = _polygon_path(cx, cy, n, inner, self.rotation)
                ph = mpatches.PathPatch(ph_path, facecolor="#ffffff",
                                        edgecolor="#2c3e50", linewidth=1.2,
                                        zorder=4)
                ax.add_patch(ph)

        # Stripes: draw 3 horizontal dark lines across the shape's bbox
        if self.stripes:
            for yoff in (-size_val * 0.4, 0, size_val * 0.4):
                ax.plot([cx - size_val * 0.85, cx + size_val * 0.85],
                        [cy + yoff, cy + yoff],
                        color="#2c3e50", linewidth=1.4, zorder=4.5)

        # Dot position (corner marker)
        if self.dot_pos is not None:
            off = cell_size * 0.33
            pos_map = {
                "TL": (cx - off, cy + off),
                "TR": (cx + off, cy + off),
                "BL": (cx - off, cy - off),
                "BR": (cx + off, cy - off),
            }
            px, py = pos_map[self.dot_pos]
            dot = mpatches.Circle((px, py), cell_size * 0.05,
                                  facecolor="#1a1a1a", zorder=5)
            ax.add_patch(dot)

# ------------------------------------------------------------------ #
# Transformations
# ------------------------------------------------------------------ #

def _t_rotate_90(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.rotation = (n.rotation + 90) % 360
    return n

def _t_rotate_180(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.rotation = (n.rotation + 180) % 360
    return n

def _t_scale_up(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.size_idx = min(len(_SIZES) - 1, n.size_idx + 1)
    return n

def _t_scale_down(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.size_idx = max(0, n.size_idx - 1)
    return n

def _t_toggle_fill(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.filled = not n.filled
    return n

def _t_add_hole(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.has_hole = True
    return n

def _t_add_stripes(s: _AbstractShape, rng) -> _AbstractShape:
    n = s.copy()
    n.stripes = True
    return n

def _t_dot_rotate(s: _AbstractShape, rng) -> _AbstractShape:
    """Move the dot one corner clockwise (TL -> TR -> BR -> BL -> TL).
    If no dot, place at TL."""
    n = s.copy()
    if n.dot_pos is None:
        n.dot_pos = "TL"
        return n
    seq = ["TL", "TR", "BR", "BL"]
    idx = seq.index(n.dot_pos)
    n.dot_pos = seq[(idx + 1) % 4]
    return n

_TRANSFORMS_SIMPLE = [
    ("rotate 90°",      _t_rotate_90),
    ("rotate 180°",     _t_rotate_180),
    ("scale up",        _t_scale_up),
    ("scale down",      _t_scale_down),
    ("toggle fill",     _t_toggle_fill),
]

_TRANSFORMS_MEDIUM = [
    ("add hole",        _t_add_hole),
    ("add stripes",     _t_add_stripes),
    ("dot rotates CW",  _t_dot_rotate),
]

class VisualAnalogyAbstractQA(StandaloneVisualEnv):
    ENV_NAME = "visual_analogy_abstract"

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Return difficulty config for a given level (0-9).

        Key insight: larger transform pools produce MORE visually distinct
        distractors, making the task EASIER (model at L6 n_rules=1 scored
        0.90, same as L0). Composite rules (n_rules>=2) produce partial-
        application distractors that look similar to the correct answer,
        making the task genuinely harder.

        Progression:
          L0-2: n_rules=1, small pool (obvious transforms)
          L3-5: n_rules=1, medium+subtle transforms
          L6-7: n_rules=2, composite (partial distractors)
          L8-9: n_rules=3, triple composite (hardest)
        """
        if level <= 0:
            # Single obvious transform, small pool
            return {"pool": [("scale up", _t_scale_up),
                             ("scale down", _t_scale_down),
                             ("toggle fill", _t_toggle_fill),
                             ("rotate 180\u00b0", _t_rotate_180)],
                    "n_rules": 1}
        if level <= 2:
            return {"pool": list(_TRANSFORMS_SIMPLE),
                    "n_rules": 1}
        if level <= 4:
            # Add medium transforms (add hole, add stripes, dot_rotate)
            return {"pool": list(_TRANSFORMS_SIMPLE) + list(_TRANSFORMS_MEDIUM),
                    "n_rules": 1}
        if level == 5:
            # Medium-only transforms (harder to distinguish)
            return {"pool": list(_TRANSFORMS_MEDIUM),
                    "n_rules": 1}
        if level <= 7:
            # Composite: 2 rules. Distractors are partial applications
            # (only 1 of 2 rules applied) which look similar to correct.
            return {"pool": list(_TRANSFORMS_SIMPLE) + list(_TRANSFORMS_MEDIUM),
                    "n_rules": 2}
        # L8-9: Triple composite. Must verify all 3 rules applied.
        return {"pool": list(_TRANSFORMS_SIMPLE) + list(_TRANSFORMS_MEDIUM),
                "n_rules": 3}

    _QUESTION_TEMPLATES = [
        "Study the analogy in the image: A is to B as C is to ___. Infer the transformation from A to B, apply the same transformation to C, and choose the matching option (A, B, C, or D). Answer with a single letter.",
        "In the image, shape A transforms into shape B. Apply the same rule to shape C. Which option (A-D) shows the correct result? Answer with a single letter.",
        "A : B :: C : ? -- Identify the transformation from A to B, then apply it to C. Pick the correct answer from options A-D. Answer with a single letter.",
        "The image shows an analogy puzzle. What transformation converts A into B? Apply it to C and select the matching option. Answer with a single letter A, B, C, or D.",
        "Determine the rule that maps A to B in the image. Apply that rule to C and choose the matching option (A-D). Answer with a single letter.",
        "Find the transformation A->B and use it on C. Pick the option (A-D) that results from this transformation. Answer with a single letter.",
    ]

    _TITLE_VARIANTS = [
        "Abstract Analogy: A : B :: C : ?",
        "Shape Analogy",
        "Visual Analogy Puzzle",
        "A is to B as C is to ?",
        "Transformation Analogy",
        "Analogy Reasoning Test",
        "Pattern Mapping",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        pool = cfg["pool"]
        n_rules = cfg["n_rules"]

        for _ in range(30):
            result = self._try_generate(level, pool, n_rules, sub_rng)
            if result is not None:
                return result
        return None

    def _try_generate(self, level, pool, n_rules, sub_rng=None):
        rng = self._rng

        if n_rules == 1:
            rule_names = [rng.choice(pool)]
        else:
            # For composite, pick n_rules that DON'T trivially cancel
            # (e.g., scale_up + scale_down).
            for _ in range(20):
                if len(pool) < n_rules:
                    return None
                rule_names = rng.sample(pool, n_rules)
                fns = [fn for _, fn in rule_names]
                if (_t_scale_up in fns and _t_scale_down in fns):
                    continue
                if (_t_rotate_90 in fns and _t_rotate_180 in fns):
                    # allowed but redundant — bias towards more diverse pairs
                    if rng.random() < 0.5:
                        continue
                break
            else:
                return None
        rule_fns = [fn for _, fn in rule_names]

        def apply_rule(s):
            x = s
            for fn in rule_fns:
                x = fn(x, rng)
            return x

        # Pick A (random abstract shape). For lower levels keep A especially
        # plain so the distractors are maximally distinguishable.
        a = _AbstractShape(
            shape=rng.choice(_SHAPES),
            color=rng.choice(_COLORS),
            size_idx=1,  # always medium — scale up/down moves it predictably
            rotation=0,
            filled=True,
            has_hole=False,
            stripes=False,
            dot_pos=None,
        )
        b = apply_rule(a)
        if b == a:
            return None

        # Pick C distinct from A.
        tries = 0
        c = None
        while tries < 50:
            tries += 1
            cand = _AbstractShape(
                shape=rng.choice(_SHAPES),
                color=rng.choice(_COLORS),
                size_idx=1,
                rotation=0,
                filled=True,
                has_hole=False,
                stripes=False,
                dot_pos=None,
            )
            if cand == a:
                continue
            d_cand = apply_rule(cand)
            if d_cand == cand:
                continue
            c = cand
            d = d_cand
            break
        if c is None:
            return None

        # Distractor strategy depends on level.
        distractors = self._make_distractors(
            rng, c, d, pool, rule_fns, level, n_rules
        )
        if len(distractors) < 3:
            return None

        options = list(distractors[:3])
        correct_idx = rng.randint(0, len(options))
        options.insert(correct_idx, d)
        answer_letter = chr(ord("A") + correct_idx)

        title = (sub_rng or rng).choice(self._TITLE_VARIANTS)
        image = self._render(a, b, c, options, title=title, sub_rng=sub_rng)
        question = (sub_rng or rng).choice(self._QUESTION_TEMPLATES)
        self._primary_complexity_feature = n_rules * 3 + level
        return question, answer_letter, image

    def _make_distractors(self, rng, c: _AbstractShape,
                          correct: _AbstractShape, pool,
                          used_fns, level: int,
                          n_rules: int) -> List[_AbstractShape]:
        """Build distractors. Strategy depends on level.

        L0-2 (single simple rule): distractors apply a DIFFERENT simple rule.
                                   This keeps each distractor visually
                                   distinct from the correct answer.
        L3-5 (single medium rule): distractors apply a DIFFERENT rule from
                                   the same pool — still visually
                                   distinguishable but more plausible.
        L6-9 (composite 2-rule):   distractors apply ONLY ONE of the two
                                   rules — the model must verify BOTH were
                                   applied.
        """
        distractors: List[_AbstractShape] = []
        used = set(used_fns)

        if n_rules >= 2 and level >= 6:
            # At composite levels, the key distractors are partials:
            # apply only SOME of the n_rules. The model must verify that
            # all n_rules were applied.
            # Generate all proper subsets of used_fns.
            from itertools import combinations
            for subset_size in range(n_rules):  # 0..n_rules-1 (all proper subsets)
                for subset in combinations(used_fns, subset_size):
                    d = c.copy()
                    for fn in subset:
                        d = fn(d, rng)
                    if d != correct and d != c and d not in distractors:
                        distractors.append(d)
                    if len(distractors) >= 6:
                        break
                if len(distractors) >= 6:
                    break

            # If still need more, add wrong-rule distractors
            wrong_pool = [fn for _, fn in pool if fn not in used]
            attempts = 0
            while len(distractors) < 6 and attempts < 50:
                attempts += 1
                if not wrong_pool:
                    break
                fn = rng.choice(wrong_pool)
                d = fn(c.copy(), rng)
                if d != correct and d != c and d not in distractors:
                    distractors.append(d)
            return distractors

        # Single-rule levels (L0-5): distractors are DIFFERENT rules.
        wrong_pool = [fn for _, fn in pool if fn not in used]
        attempts = 0
        while len(distractors) < 6 and attempts < 200:
            attempts += 1
            if not wrong_pool:
                break
            fn = rng.choice(wrong_pool)
            d = fn(c.copy(), rng)
            if d == correct or d == c or d in distractors:
                continue
            distractors.append(d)
        return distractors

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _render(self, a, b, c, options, title="Abstract Analogy: A : B :: C : ?",
                sub_rng=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig_w = 10.0 * sc
        fig_h = 6.5 * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.2], hspace=0.25)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        # Top row layout: A → B :: C → ?
        ax_top.set_xlim(0, 10)
        ax_top.set_ylim(0, 2.5)
        ax_top.set_aspect("equal")
        ax_top.axis("off")
        ax_top.set_title(title,
                         fontsize=15, fontweight="bold", pad=8)

        cell_size = 1.8

        def _cell(ax, cx, cy, color, dash=False):
            rect = mpatches.FancyBboxPatch(
                (cx - cell_size / 2, cy - cell_size / 2),
                cell_size, cell_size,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor=("#e74c3c" if dash else "#2c3e50"),
                linewidth=(2.5 if dash else 2.0),
                linestyle=("--" if dash else "-"),
                zorder=1)
            ax.add_patch(rect)

        _cell(ax_top, 1.0, 1.25, "#fdfefe")
        a.draw(ax_top, 1.0, 1.25, cell_size)
        ax_top.text(1.0, 0.15, "A", fontsize=14, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.annotate("", xy=(3.0, 1.25), xytext=(2.0, 1.25),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color="#e67e22"))

        _cell(ax_top, 3.8, 1.25, "#fdfefe")
        b.draw(ax_top, 3.8, 1.25, cell_size)
        ax_top.text(3.8, 0.15, "B", fontsize=14, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.text(5.2, 1.25, "::", fontsize=26, fontweight="bold",
                    ha="center", va="center", color="#7f8c8d")

        _cell(ax_top, 6.6, 1.25, "#fdfefe")
        c.draw(ax_top, 6.6, 1.25, cell_size)
        ax_top.text(6.6, 0.15, "C", fontsize=14, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.annotate("", xy=(8.6, 1.25), xytext=(7.6, 1.25),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color="#e67e22"))

        _cell(ax_top, 9.0, 1.25, "#fef3c7", dash=True)
        ax_top.text(9.0, 1.25, "?", fontsize=30, fontweight="bold",
                    ha="center", va="center", color="#e74c3c", zorder=5)
        ax_top.text(9.0, 0.15, "?", fontsize=14, fontweight="bold",
                    ha="center", va="top", color="#e74c3c")

        # -- Options --
        n_opts = len(options)
        ax_bot.set_xlim(0, n_opts * 2.2)
        ax_bot.set_ylim(0, 2.1)
        ax_bot.set_aspect("equal")
        ax_bot.axis("off")
        ax_bot.set_title("Choose the option that matches:",
                         fontsize=12, pad=3)

        opt_cell = 1.6
        for i, opt in enumerate(options):
            cx = i * 2.2 + 1.1
            cy = 1.0
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2),
                opt_cell, opt_cell,
                boxstyle="round,pad=0.04",
                facecolor="#eaf2f8", edgecolor="#2c3e50",
                linewidth=1.5, zorder=1)
            ax_bot.add_patch(rect)
            opt.draw(ax_bot, cx, cy, opt_cell)
            label = chr(ord("A") + i)
            ax_bot.text(cx, cy - opt_cell / 2 - 0.1, label,
                        fontsize=13, fontweight="bold", ha="center",
                        va="top", color="#2c3e50")

        fig.subplots_adjust(left=0.05, right=0.95, top=0.92,
                            bottom=0.05, hspace=0.25)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Local smoke test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check"
    os.makedirs(out_dir, exist_ok=True)
    env = VisualAnalogyAbstractQA()
    for level in (0, 3, 6):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED to generate")
                continue
            img = env.render()
            q = env.get_instruction()
            a = env._answer
            path = os.path.join(
                out_dir, f"visual_analogy_abstract_seed{seed}_L{level}.png")
            img.save(path)
            print(f"[seed={seed} L{level}] saved {path}")
            print(f"  Q: {q}")
            print(f"  A: {a}")
