"""
Analogy Multiple Dimensions QA environment.

A:B :: C:? where B differs from A in 2-3 attributes simultaneously
(e.g., color + shape + size). Each attribute change follows a rule.
Distractors at high levels get all but one attribute right.

Target: VisualPuzzles analogical,
VisuLogic Attribute Reasoning.

Difficulty axes:
  1. n_dimensions = 2 (L0-3) or 3 (L4-9).
  2. rule_complexity: L0-2 simple swap/toggle, L3-5 arithmetic rule
     (+2, x2), L6-9 function composition.
  3. distractor_correct_on_k_dims: at high level distractors agree
     with the correct answer on N-1 dimensions.
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

_SHAPES_BY_SIDES = {3: "triangle", 4: "square", 5: "pentagon",
                    6: "hexagon", 7: "heptagon", 8: "octagon"}
_SIZES = [0.25, 0.35, 0.45, 0.55]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c",
           "#e67e22", "#34495e"]

class AnalogyMultipleDimensionsQA(StandaloneVisualEnv):
    ENV_NAME = "analogy_multiple_dimensions"

    # ------------------------------------------------------------------ #
    # Level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            n_dims = 2
        else:
            n_dims = 3
        if level <= 2:
            rule_class = "simple"
        elif level <= 5:
            rule_class = "arithmetic"
        else:
            rule_class = "composition"
        return {
            "n_dimensions": n_dims,
            "rule_class": rule_class,
            "tight_distractors": level >= 6,
        }

    _QUESTION_TEMPLATES = [
        "Study the analogy: A is to B as C is to ___. Multiple attributes change from A to B -- identify all of them and apply the same changes to C. Answer with a single letter (A-D).",
        "In the image, A becomes B by changing several attributes at once (e.g., color + size + shape). Apply the same combined transformation to C and pick the correct option. Answer with a single letter.",
        "A : B :: C : ? -- Find every attribute that changes from A to B, then apply ALL those changes to C. Choose the matching option from A-D. Answer with a single letter.",
        "Multiple attributes change between A and B. Identify them and transform C accordingly. Which option (A-D) matches? Answer with a single letter.",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1303)
        self._primary_complexity_feature = cfg["n_dimensions"] * 4 + level

        for _ in range(40):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Transformations on state
    # ------------------------------------------------------------------ #

    _ATTR_NAMES = ["sides", "color_idx", "size_idx", "count"]

    def _choose_transforms(self, cfg: Dict, rng: random.Random):
        """Return a dict attribute_name -> (rule_name, transform_fn)."""
        dims = rng.sample(self._ATTR_NAMES, cfg["n_dimensions"])
        rule_class = cfg["rule_class"]
        transforms = {}
        for attr in dims:
            if attr == "sides":
                if rule_class == "simple":
                    step = rng.choice([1, -1])
                elif rule_class == "arithmetic":
                    step = rng.choice([2, -2])
                else:
                    step = rng.choice([2, 3, -2])
                transforms[attr] = ("sides+%d" % step, ("add", step))
            elif attr == "color_idx":
                if rule_class == "simple":
                    step = rng.choice([1, -1])
                elif rule_class == "arithmetic":
                    step = rng.choice([2, -2, 3])
                else:
                    step = rng.choice([2, 3, 4])
                transforms[attr] = ("color+%d" % step, ("add_mod", step))
            elif attr == "size_idx":
                if rule_class == "simple":
                    step = rng.choice([1, -1])
                elif rule_class == "arithmetic":
                    step = rng.choice([2, -2])
                else:
                    step = rng.choice([2, 3, -2])
                transforms[attr] = ("size+%d" % step, ("add_clamp", step))
            elif attr == "count":
                if rule_class == "simple":
                    step = rng.choice([1, -1])
                elif rule_class == "arithmetic":
                    step = rng.choice([2, -1])
                else:
                    step = rng.choice([2, 3])
                transforms[attr] = ("count+%d" % step, ("add_clamp_count", step))
        return transforms

    def _apply_transforms(self, state: Dict, transforms: Dict):
        new = dict(state)
        for attr, (_name, (op, param)) in transforms.items():
            if op == "add":
                if attr == "sides":
                    v = new["sides"] + param
                    v = max(3, min(8, v))
                    new["sides"] = v
            elif op == "add_mod":
                new[attr] = (new[attr] + param) % len(_COLORS)
            elif op == "add_clamp":
                if attr == "size_idx":
                    new[attr] = max(0, min(len(_SIZES) - 1,
                                             new[attr] + param))
            elif op == "add_clamp_count":
                new["count"] = max(1, min(4, new["count"] + param))
        return new

    # ------------------------------------------------------------------ #
    # Problem construction
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        transforms = self._choose_transforms(cfg, rng)
        # Pick a start state from the "safe" middle so transforms don't clip.
        a = {
            "sides": rng.randint(4, 6),
            "color_idx": rng.randint(2, len(_COLORS) - 3),
            "size_idx": rng.randint(1, len(_SIZES) - 2),
            "count": rng.randint(1, 2),
        }
        b = self._apply_transforms(a, transforms)
        if b == a:
            return None

        # Pick C that is different from A and produces a valid non-trivial D.
        c = None
        correct = None
        for _ in range(40):
            cand = {
                "sides": rng.randint(4, 6),
                "color_idx": rng.randint(2, len(_COLORS) - 3),
                "size_idx": rng.randint(1, len(_SIZES) - 2),
                "count": rng.randint(1, 2),
            }
            if cand == a:
                continue
            d_cand = self._apply_transforms(cand, transforms)
            if d_cand == cand:
                continue
            c = cand
            correct = d_cand
            break
        if c is None:
            return None

        # Distractors
        distractors = self._make_distractors(cfg, rng, transforms, c, correct)
        if len(distractors) < 3:
            return None

        options = list(distractors[:3])
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)

        question = rng.choice(self._QUESTION_TEMPLATES)
        image = self._render(a, b, c, options, rng)
        return question, answer_letter, image

    def _make_distractors(self, cfg, rng, transforms, c, correct):
        """Build distractors. At high levels, each distractor agrees with
        correct on all but one dimension (tight)."""
        tight = cfg["tight_distractors"]
        dims = list(transforms.keys())
        distractors = []
        seen = [correct]

        def _same(x, y):
            return all(x[k] == y[k] for k in ["sides", "color_idx",
                                                 "size_idx", "count"])

        def _add(cand):
            for s in seen:
                if _same(s, cand):
                    return False
            if _same(cand, c):
                return False
            seen.append(cand)
            distractors.append(cand)
            return True

        if tight:
            # For each dim, break the transformation for that dim only
            for skip_dim in dims:
                partial = {k: v for k, v in transforms.items()
                           if k != skip_dim}
                cand = self._apply_transforms(c, partial)
                _add(cand)
                if len(distractors) >= 3:
                    break
            # If not enough, apply opposite rule on one dim
            if len(distractors) < 3:
                for flip_dim in dims:
                    alt = dict(transforms)
                    name, (op, param) = alt[flip_dim]
                    alt[flip_dim] = (name, (op, -param if isinstance(param, int)
                                            else param))
                    cand = self._apply_transforms(c, alt)
                    _add(cand)
                    if len(distractors) >= 3:
                        break
        else:
            # Loose distractors: random attribute deviations from correct
            attempts = 0
            while len(distractors) < 3 and attempts < 40:
                attempts += 1
                cand = dict(correct)
                n_changes = rng.randint(1, 2)
                changed_attrs = rng.sample(["sides", "color_idx",
                                              "size_idx", "count"],
                                             n_changes)
                for at in changed_attrs:
                    if at == "sides":
                        cand[at] = rng.choice([s for s in range(3, 9)
                                                 if s != correct[at]])
                    elif at == "color_idx":
                        cand[at] = rng.choice([i for i in range(len(_COLORS))
                                                 if i != correct[at]])
                    elif at == "size_idx":
                        cand[at] = rng.choice([i for i in range(len(_SIZES))
                                                 if i != correct[at]])
                    elif at == "count":
                        cand[at] = rng.choice([v for v in [1, 2, 3, 4]
                                                 if v != correct[at]])
                _add(cand)
        # Final padding if still short
        attempts = 0
        while len(distractors) < 3 and attempts < 30:
            attempts += 1
            cand = dict(correct)
            cand["sides"] = rng.choice([s for s in range(3, 9)
                                          if s != correct["sides"]])
            cand["color_idx"] = rng.choice([i for i in range(len(_COLORS))
                                              if i != correct["color_idx"]])
            _add(cand)
        return distractors

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, a, b, c, options, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(10.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.3], hspace=0.3)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])

        ax_top.set_xlim(0, 10)
        ax_top.set_ylim(0, 2.6)
        ax_top.set_aspect("equal")
        ax_top.axis("off")
        title_pool = [
            "Multi-Dimension Analogy: A : B :: C : ?",
            "Multi-Attribute Analogy",
            "A is to B as C is to ?",
            "Attribute Analogy Puzzle",
        ]
        ax_top.set_title(rng.choice(title_pool), fontsize=14,
                         fontweight="bold", pad=6)

        cell_size = 1.8

        def _box(ax, cx, cy, dash=False):
            rect = mpatches.FancyBboxPatch(
                (cx - cell_size / 2, cy - cell_size / 2),
                cell_size, cell_size, boxstyle="round,pad=0.05",
                facecolor="#fdfefe",
                edgecolor=("#e74c3c" if dash else "#2c3e50"),
                linewidth=(2.5 if dash else 2.0),
                linestyle=("--" if dash else "-"), zorder=1)
            ax.add_patch(rect)

        _box(ax_top, 1.0, 1.3)
        _draw_state(ax_top, 1.0, 1.3, cell_size, a)
        ax_top.text(1.0, 0.25, "A", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.annotate("", xy=(2.95, 1.3), xytext=(2.0, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color="#e67e22"))

        _box(ax_top, 3.8, 1.3)
        _draw_state(ax_top, 3.8, 1.3, cell_size, b)
        ax_top.text(3.8, 0.25, "B", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.text(5.2, 1.3, "::", fontsize=24, fontweight="bold",
                    ha="center", va="center", color="#7f8c8d")

        _box(ax_top, 6.6, 1.3)
        _draw_state(ax_top, 6.6, 1.3, cell_size, c)
        ax_top.text(6.6, 0.25, "C", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#2c3e50")

        ax_top.annotate("", xy=(8.55, 1.3), xytext=(7.6, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=2.5,
                                        color="#e67e22"))

        _box(ax_top, 9.0, 1.3, dash=True)
        ax_top.text(9.0, 1.3, "?", fontsize=28, fontweight="bold",
                    ha="center", va="center", color="#e74c3c")
        ax_top.text(9.0, 0.25, "?", fontsize=13, fontweight="bold",
                    ha="center", va="top", color="#e74c3c")

        # Options row
        n_opts = len(options)
        ax_bot.set_xlim(0, n_opts * 2.2)
        ax_bot.set_ylim(0, 2.2)
        ax_bot.set_aspect("equal")
        ax_bot.axis("off")
        ax_bot.set_title("Choose the option that matches:",
                         fontsize=11, pad=3)
        opt_cell = 1.6
        for i, opt in enumerate(options):
            cx = i * 2.2 + 1.1
            cy = 1.05
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2),
                opt_cell, opt_cell, boxstyle="round,pad=0.04",
                facecolor="#eaf2f8", edgecolor="#2c3e50", linewidth=1.5,
                zorder=1)
            ax_bot.add_patch(rect)
            _draw_state(ax_bot, cx, cy, opt_cell, opt)
            ax_bot.text(cx, cy - opt_cell / 2 - 0.15, chr(ord("A") + i),
                        fontsize=12, fontweight="bold", ha="center",
                        va="top", color="#2c3e50")

        fig.subplots_adjust(left=0.04, right=0.96, top=0.9,
                            bottom=0.05, hspace=0.35)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# State drawing
# ---------------------------------------------------------------------- #

def _draw_state(ax, cx, cy, cell_size, state):
    """Draw a shape with attributes: sides, color_idx, size_idx, count."""
    sides = state["sides"]
    color = _COLORS[state["color_idx"] % len(_COLORS)]
    size_frac = _SIZES[state["size_idx"]]
    size = cell_size * size_frac * 0.5
    n = state["count"]

    if n == 1:
        positions = [(cx, cy)]
    elif n == 2:
        positions = [(cx - cell_size * 0.17, cy),
                     (cx + cell_size * 0.17, cy)]
    elif n == 3:
        positions = [(cx, cy + cell_size * 0.17),
                     (cx - cell_size * 0.17, cy - cell_size * 0.12),
                     (cx + cell_size * 0.17, cy - cell_size * 0.12)]
    else:
        positions = [(cx - cell_size * 0.17, cy + cell_size * 0.17),
                     (cx + cell_size * 0.17, cy + cell_size * 0.17),
                     (cx - cell_size * 0.17, cy - cell_size * 0.17),
                     (cx + cell_size * 0.17, cy - cell_size * 0.17)]

    # Scale down when many
    if n > 1:
        size = size * 0.72

    for px, py in positions:
        if sides == 3:
            orient = math.pi / 2
        else:
            orient = 0
        p = mpatches.RegularPolygon((px, py), sides, radius=size,
                                     orientation=orient, facecolor=color,
                                     edgecolor="#2c3e50", linewidth=1.5)
        ax.add_patch(p)

if __name__ == "__main__":
    env = AnalogyMultipleDimensionsQA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
