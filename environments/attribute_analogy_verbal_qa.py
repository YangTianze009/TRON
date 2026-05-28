"""
Attribute Analogy Verbal QA (batch 3, 2026-04-14).

Target: a puzzle benchmark analogical / a logic benchmark Stylistic / a puzzle benchmark shape_morph.
"A : B :: C : ?" — a pair of figures on the left and a single figure on
the right with "?" for the answer, plus 4 candidate options.

Format: constant MCQ letter.

Difficulty axes:
  A) Pattern C: n_attributes that change between A→B (1..3 at L0..L9).
  B) Pattern B: tight distractors via partial-rule distractors.
  C) Pattern A: candidate pool size (4 always, but option similarity grows).
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

_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c"]
_SHAPES = ["circle", "square", "triangle", "diamond"]

class AttributeAnalogyVerbalQA(StandaloneVisualEnv):
    ENV_NAME = "attribute_analogy_verbal"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_changing": 1 + level // 4,            # 1..3 attributes change
            "tight_distractors": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_changing"]

        for _ in range(25):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _sample_state(self, rng):
        return {
            "shape": rng.choice(_SHAPES),
            "color_idx": rng.randint(0, len(_COLORS) - 1),
            "size": rng.choice([0.3, 0.45]),
            "rot": rng.choice([0, 1, 2, 3]),
        }

    def _apply_rule(self, s, rules):
        out = dict(s)
        for r in rules:
            if r == "color_shift":
                out["color_idx"] = (out["color_idx"] + 1) % len(_COLORS)
            elif r == "size_grow":
                out["size"] = min(0.6, out["size"] + 0.15)
            elif r == "rotate":
                out["rot"] = (out["rot"] + 1) % 4
            elif r == "shape_cycle":
                idx = (_SHAPES.index(out["shape"]) + 1) % len(_SHAPES)
                out["shape"] = _SHAPES[idx]
        return out

    def _try_generate(self, rng, cfg, level):
        all_rules = ["color_shift", "size_grow", "rotate", "shape_cycle"]
        rules = rng.sample(all_rules, cfg["n_changing"])

        A = self._sample_state(rng)
        B = self._apply_rule(A, rules)
        C = self._sample_state(rng)
        # Ensure C different from A
        if _state_key(C) == _state_key(A):
            return None
        D_correct = self._apply_rule(C, rules)

        # Distractors: wrong rule applied, or apply rules twice
        distractors = []
        other_rules = [r for r in all_rules if r not in rules]
        for r in other_rules:
            distractors.append(self._apply_rule(C, [r]))
        distractors.append(self._apply_rule(D_correct, rules))   # twice
        distractors.append(dict(C))                               # identity

        seen = {_state_key(D_correct)}
        uniq = [D_correct]
        for d in distractors:
            k = _state_key(d)
            if k not in seen:
                seen.add(k)
                uniq.append(d)
            if len(uniq) >= 4:
                break
        tries = 0
        while len(uniq) < 4 and tries < 40:
            tries += 1
            d = dict(D_correct)
            d["color_idx"] = (d["color_idx"] + rng.randint(1, 5)) % len(_COLORS)
            k = _state_key(d)
            if k not in seen:
                seen.add(k)
                uniq.append(d)
        if len(uniq) < 4:
            return None

        options = uniq[:4]
        rng.shuffle(options)
        idx = options.index(D_correct)
        answer = chr(ord("A") + idx)

        q_pool = [
            "The image shows two pairs of figures: the left pair (A ↦ B) demonstrates a hidden transformation rule. The right side shows C ↦ ?. Which of the options A/B/C/D on the bottom is the correct transformation of C? Answer with a single letter.",
            "The figure displays an analogy A ↦ B on the left; C maps to one of the four options (A/B/C/D) below. Which option best matches the rule?",
            "An analogy A ↦ B is shown; on the right, C maps to an unknown. Pick the option (A/B/C/D) that follows the same rule.",
            "Study A → B, then find C → ? among the four options A/B/C/D. Answer single letter.",
            "Determine the transformation from A to B, then select the option (A/B/C/D) that maps from C in the same way.",
            "Given A ↦ B, which option (A/B/C/D) should C map to? Single-letter answer.",
            "Inspect the pair A ↦ B. Choose the option letter (A/B/C/D) representing C ↦ ?.",
            "The left panel shows the rule A → B; choose from A/B/C/D the right answer to C → ?.",
            "Figure out the rule from A ↦ B, then pick the matching transformation of C among A/B/C/D.",
            "Given the analogy A ↦ B, which letter option (A/B/C/D) completes C ↦ ? correctly?",
            "What transformation maps A to B? Apply it to C and pick the matching option (A/B/C/D).",
            "Apply the rule from A ↦ B to C, and identify the correct target from A/B/C/D.",
            "Reason about the change A → B, then pick A/B/C/D so C → that option follows the rule.",
            "Work out the analogy A ↦ B. Which lettered option (A/B/C/D) fits C ↦ ?",
            "Look at A ↦ B in the figure. Which of A/B/C/D is the correct image of C under the same rule?",
            "Based on the A ↦ B analogy, choose the correct answer for C ↦ ? from options A/B/C/D.",
        ]
        q = q_pool[(self.seed or 0) % len(q_pool)]
        image = self._render(A, B, C, options)
        return q, answer, image

    def _draw(self, ax, cx, cy, cell):
        from matplotlib.transforms import Affine2D
        color = _COLORS[cell["color_idx"]]
        size = cell["size"]
        rot = cell["rot"] * 45
        if cell["shape"] == "circle":
            p = mpatches.Circle((cx, cy), size * 0.5,
                                facecolor=color, edgecolor="#1a1a1a",
                                linewidth=1.2)
        elif cell["shape"] == "square":
            p = mpatches.Rectangle(
                (cx - size * 0.5, cy - size * 0.5),
                size, size, facecolor=color, edgecolor="#1a1a1a",
                linewidth=1.2)
            tr = Affine2D().rotate_deg_around(cx, cy, rot) + ax.transData
            p.set_transform(tr)
        elif cell["shape"] == "triangle":
            pts = [(cx, cy + size * 0.55),
                   (cx - size * 0.55, cy - size * 0.35),
                   (cx + size * 0.55, cy - size * 0.35)]
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                 edgecolor="#1a1a1a", linewidth=1.2)
            tr = Affine2D().rotate_deg_around(cx, cy, rot) + ax.transData
            p.set_transform(tr)
        else:  # diamond
            pts = [(cx, cy + size * 0.55),
                   (cx + size * 0.55, cy),
                   (cx, cy - size * 0.55),
                   (cx - size * 0.55, cy)]
            p = mpatches.Polygon(pts, closed=True, facecolor=color,
                                 edgecolor="#1a1a1a", linewidth=1.2)
            tr = Affine2D().rotate_deg_around(cx, cy, rot) + ax.transData
            p.set_transform(tr)
        ax.add_patch(p)

    def _render(self, A, B, C, options) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(10 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_t = fig.add_subplot(2, 1, 1)
        ax_b = fig.add_subplot(2, 1, 2)
        ax_t.set_aspect("equal")
        ax_b.set_aspect("equal")
        ax_t.axis("off")
        ax_b.axis("off")

        cell_w = 1.8
        self._cell(ax_t, 1 * cell_w, 1.3, A)
        ax_t.annotate("", xy=(1 * cell_w + 0.6, 1.3),
                      xytext=(2 * cell_w - 0.6, 1.3),
                      arrowprops=dict(arrowstyle="<-", lw=1.8))
        self._cell(ax_t, 2 * cell_w, 1.3, B)
        ax_t.text(1.5 * cell_w, 2.2, "A ↦ B",
                  fontsize=fs + 1, fontweight="bold", ha="center")

        # Right pair
        self._cell(ax_t, 4 * cell_w, 1.3, C)
        ax_t.annotate("", xy=(4 * cell_w + 0.6, 1.3),
                      xytext=(5 * cell_w - 0.6, 1.3),
                      arrowprops=dict(arrowstyle="<-", lw=1.8, color="#c0392b"))
        ax_t.add_patch(mpatches.Rectangle(
            (5 * cell_w - cell_w / 2 + 0.1, 1.3 - cell_w / 2 + 0.1),
            cell_w - 0.2, cell_w - 0.2,
            facecolor="#ffeaa7", edgecolor="#c0392b", linewidth=1.8))
        ax_t.text(5 * cell_w, 1.3, "?", fontsize=fs + 8,
                  fontweight="bold", ha="center", va="center", color="#c0392b")
        ax_t.text(4.5 * cell_w, 2.2, "C ↦ ?",
                  fontsize=fs + 1, fontweight="bold", ha="center")

        ax_t.set_xlim(-0.2, 6 * cell_w + 0.8)
        ax_t.set_ylim(0, 3.5)

        for i, opt in enumerate(options):
            self._cell(ax_b, (i + 0.5) * cell_w + 0.6, 1.3, opt)
            ax_b.text((i + 0.5) * cell_w + 0.6, 0.3,
                      f"({chr(ord('A') + i)})",
                      fontsize=fs + 1, fontweight="bold",
                      ha="center", va="center")
        ax_b.set_xlim(0, 5 * cell_w)
        ax_b.set_ylim(0, 2.5)
        ax_b.set_title("Options", fontsize=fs + 1)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95,
                            bottom=0.05, hspace=0.4)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _cell(self, ax, cx, cy, cell):
        ax.add_patch(mpatches.Rectangle(
            (cx - 0.7, cy - 0.7), 1.4, 1.4,
            facecolor="#ffffff", edgecolor="#7f8c8d", linewidth=1))
        self._draw(ax, cx, cy, cell)

def _state_key(c):
    return (c["shape"], c["color_idx"], round(c["size"], 2), c["rot"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = AttributeAnalogyVerbalQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[attribute_analogy_verbal L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"attribute_analogy_verbal_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[attribute_analogy_verbal L{level} s{s}] A={env._answer}")
