"""
Figure Sequence Next QA (v2 diversity redesign, 2026-04-16;
R5 B4B 2026-05-05 EASE + 5-MCQ trap restructure).

Inductive / figure-sequence pattern continuation.
Shows a 3-6 figure sequence plus 4-5 candidate next-figures. Rule may be
size increment, rotation increment, count increment, color cycle, shape
morph, stripe pattern, or compound rules.

v2 diversity redesign:
  * Shape pool expanded to 10 primitives.
  * Per-seed palette shuffle from 8 palettes.
  * Layouts: row, vertical column, zig-zag, circular.
  * 5 question stem variants + 6 titles.
  * L0 = single rule, 3 figures, 1 step to predict; L9 = compound rule
    (2 simultaneous), 6 figures, background noise decoys.
  * MCQ always shuffled.

Format: MCQ with letter.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PALETTE_POOL = [
    ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c"],
    ["#4285f4", "#ea4335", "#fbbc05", "#34a853", "#ff6d01", "#46bdc6"],
    ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#606c38"],
    ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600", "#374c80"],
    ["#1d3557", "#457b9d", "#a8dadc", "#e63946", "#2b2d42", "#8d99ae"],
    ["#bb86fc", "#03dac6", "#cf6679", "#ffb74d", "#81c784", "#64b5f6"],
    ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#a0c4ff"],
    ["#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2980b9", "#16a085"],
]

_SHAPE_POOL = [
    "circle", "square", "triangle", "diamond", "pentagon", "hexagon",
    "star5", "plus", "heart", "arrow",
]

_TITLE_POOL = [
    "Sequence", "Pattern", "Figure Sequence", "Transformation",
    "Rule Sequence", "What comes next?",
]

_OPT_TITLE_POOL = [
    "Options", "Choices", "Candidates", "Pick one",
]

_QUESTION_STEMS = [
    "The top row shows a sequence of figures that follow a hidden rule. Which option is the correct next figure? Answer with a single letter.",
    "Study the sequence and identify the transformation rule. Which option continues the pattern? Answer with a single letter.",
    "A rule governs the transformations between figures in the sequence. Which option extends the sequence correctly?",
    "Observe the pattern of changes in the sequence. Which of the four options should come next? Answer A, B, C, or D.",
    "Each figure in the sequence is produced from the previous by a consistent rule. Which option is the next figure in the pattern?",
]

def _shape_pts(shape, cx, cy, s):
    if shape == "circle":
        return None  # use Circle patch
    if shape == "square":
        return [(cx - s, cy - s), (cx + s, cy - s),
                (cx + s, cy + s), (cx - s, cy + s)]
    if shape == "triangle":
        return [(cx, cy + s), (cx - s, cy - s * 0.85),
                (cx + s, cy - s * 0.85)]
    if shape == "diamond":
        return [(cx, cy + s), (cx + s, cy),
                (cx, cy - s), (cx - s, cy)]
    if shape == "pentagon":
        return [(cx + s * math.cos(math.radians(72 * i + 90)),
                 cy + s * math.sin(math.radians(72 * i + 90)))
                for i in range(5)]
    if shape == "hexagon":
        return [(cx + s * math.cos(math.radians(60 * i + 30)),
                 cy + s * math.sin(math.radians(60 * i + 30)))
                for i in range(6)]
    if shape == "star5":
        pts = []
        for i in range(10):
            a = math.radians(36 * i - 90)
            r = s if i % 2 == 0 else s * 0.45
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts
    if shape == "plus":
        return [(cx - s / 3, cy - s), (cx + s / 3, cy - s),
                (cx + s / 3, cy - s / 3), (cx + s, cy - s / 3),
                (cx + s, cy + s / 3), (cx + s / 3, cy + s / 3),
                (cx + s / 3, cy + s), (cx - s / 3, cy + s),
                (cx - s / 3, cy + s / 3), (cx - s, cy + s / 3),
                (cx - s, cy - s / 3), (cx - s / 3, cy - s / 3)]
    if shape == "heart":
        pts = []
        for i in range(30):
            t = 2 * math.pi * i / 30
            hx = 16 * math.sin(t) ** 3
            hy = 13 * math.cos(t) - 5 * math.cos(2 * t) \
                 - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((cx + hx * s / 18, cy + hy * s / 18))
        return pts
    if shape == "arrow":
        return [(cx - s, cy - s / 3), (cx + s * 0.3, cy - s / 3),
                (cx + s * 0.3, cy - s * 0.8), (cx + s, cy),
                (cx + s * 0.3, cy + s * 0.8),
                (cx + s * 0.3, cy + s / 3),
                (cx - s, cy + s / 3)]
    return None

class FigureSequenceNextQA(StandaloneVisualEnv):
    ENV_NAME = "figure_sequence_next"

    # 2026-05-05 R5 B4B: orientation-sensitive (rotate is in the rule pool +
    # zigzag/ring layouts encode position).
    ALLOW_ROTATION = False

    _LAYOUT_POOL = ["row", "column", "zigzag", "ring"]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-05 R5 B4B EASE: L0/L1 stay 4-MCQ row layout, single rule
        # (count_inc only at L0 — most visually obvious; L1 adds size_inc).
        # Was R3 0.225 too-hard. Per advisor: drop the 2-option special case
        # (parent's MCQ-letter logic still works on 4-MCQ); short hint only.
        if level == 0:
            return {
                "sequence_len": 3,
                "rule_depth": 1,
                "rule_pool": ["count_inc"],
                "tight_distractors": False,
                "use_noise": False,
                "layout_pool": ["row"],
            }
        if level == 1:
            return {
                "sequence_len": 3,
                "rule_depth": 1,
                "rule_pool": ["count_inc", "size_inc"],
                "tight_distractors": False,
                "use_noise": False,
                "layout_pool": ["row"],
            }
        if level <= 3:
            return {
                "sequence_len": 4,
                "rule_depth": 1,
                "rule_pool": ["count_inc", "size_inc", "rotate"],
                "tight_distractors": False,
                "use_noise": False,
                "layout_pool": ["row", "column"],
            }
        if level <= 5:
            return {
                "sequence_len": 5,
                "rule_depth": 1,
                "rule_pool": ["count_inc", "size_inc", "rotate",
                              "color_cycle", "shape_morph"],
                "tight_distractors": True,
                "use_noise": False,
                "layout_pool": ["row", "zigzag"],
            }
        if level <= 7:
            return {
                "sequence_len": 5,
                "rule_depth": 2,
                "rule_pool": ["count_inc", "size_inc", "rotate",
                              "color_cycle", "shape_morph"],
                "tight_distractors": True,
                "use_noise": True,
                "layout_pool": ["row", "zigzag", "ring"],
                "trap_rate": 0.30,
            }
        return {
            "sequence_len": 6,
            "rule_depth": 2,
            "rule_pool": ["count_inc", "size_inc", "rotate",
                          "color_cycle", "shape_morph"],
            "tight_distractors": True,
            "use_noise": True,
            "layout_pool": ["zigzag", "ring", "row"],
            "trap_rate": 0.40,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["rule_depth"] + level * 2

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        palette = rng.choice(_PALETTE_POOL)
        rules = rng.sample(cfg["rule_pool"],
                            min(cfg["rule_depth"], len(cfg["rule_pool"])))

        state = {
            "shape": rng.choice(_SHAPE_POOL),
            "count": rng.randint(1, 2),
            "size": 0.25,
            "rot": 0,
            "color_idx": rng.randint(0, len(palette) - 1),
        }

        def _apply(s, rs, palette):
            s = dict(s)
            for r in rs:
                if r == "count_inc":
                    s["count"] = min(5, s["count"] + 1)
                elif r == "size_inc":
                    # Larger delta so size growth is visually obvious
                    s["size"] = min(0.75, s["size"] + 0.12)
                elif r == "rotate":
                    s["rot"] = (s["rot"] + 1) % 4
                elif r == "color_cycle":
                    s["color_idx"] = (s["color_idx"] + 1) % len(palette)
                elif r == "shape_morph":
                    idx = _SHAPE_POOL.index(s["shape"])
                    s["shape"] = _SHAPE_POOL[(idx + 1) % len(_SHAPE_POOL)]
            return s

        seq = [state]
        for _ in range(cfg["sequence_len"] - 1):
            seq.append(_apply(seq[-1], rules, palette))
        correct = _apply(seq[-1], rules, palette)

        # Build distractors
        others_pool = [r for r in cfg["rule_pool"] if r not in rules]
        distractors = []
        if others_pool:
            for r in others_pool:
                distractors.append(_apply(seq[-1], [r], palette))
        distractors.append(_apply(correct, rules, palette))
        distractors.append(dict(seq[-1]))
        # Also add off-by-one distractor
        d_off = dict(correct)
        d_off["count"] = max(1, d_off["count"] + rng.choice([-1, 1]))
        distractors.append(d_off)

        def _key(c):
            return (c["shape"], c["count"], round(c["size"], 2),
                    c["rot"], c["color_idx"])

        seen = {_key(correct)}
        uniq = [correct]
        for d in distractors:
            k = _key(d)
            if k not in seen:
                seen.add(k)
                uniq.append(d)
            if len(uniq) >= 4:
                break
        tries = 0
        while len(uniq) < 4 and tries < 40:
            tries += 1
            d = dict(correct)
            d["count"] = max(1, d["count"] + rng.choice([-1, 1]))
            d["color_idx"] = (d["color_idx"]
                              + rng.randint(1, len(palette) - 1)) % len(palette)
            if rng.random() < 0.3:
                d["shape"] = rng.choice(_SHAPE_POOL)
            k = _key(d)
            if k not in seen:
                seen.add(k)
                uniq.append(d)
        if len(uniq) < 4:
            return None

        # 2026-05-05 R5 B4B: L8/L9 may use 5-MCQ + "E. No correct" trap.
        trap_rate = cfg.get("trap_rate", 0.0)
        use_5opt = level >= 6
        n_visible_opts = 5 if use_5opt else 4
        # Need n_visible_opts-1 distractors when use_5opt+trap, n_visible_opts when no trap.
        # We currently have uniq with at least 4 entries; ensure 5 if needed.
        if use_5opt and len(uniq) < 5:
            tries2 = 0
            while len(uniq) < 5 and tries2 < 40:
                tries2 += 1
                d = dict(correct)
                d["color_idx"] = (d["color_idx"]
                                  + rng.randint(1, len(palette) - 1)) % len(palette)
                if rng.random() < 0.4:
                    d["shape"] = rng.choice(_SHAPE_POOL)
                k = _key(d)
                if k not in seen:
                    seen.add(k)
                    uniq.append(d)
            if len(uniq) < 5:
                return None

        use_trap = use_5opt and (rng.random() < trap_rate)
        if use_trap:
            # All visible options are wrong; "No correct answer" is right.
            wrongs = [u for u in uniq if _key(u) != _key(correct)]
            if len(wrongs) < (n_visible_opts - 1):
                return None
            options = wrongs[:n_visible_opts - 1]
            rng.shuffle(options)
            answer = "E"
            visible_letters = "A/B/C/D/E"
        else:
            options = uniq[:n_visible_opts] if not use_5opt else \
                ([correct] + [u for u in uniq if _key(u) != _key(correct)][:n_visible_opts - 2])
            if not use_5opt:
                # Standard 4-MCQ (correct included)
                options = [correct] + [u for u in uniq if _key(u) != _key(correct)][:3]
            rng.shuffle(options)
            idx = options.index(correct)
            answer = chr(ord("A") + idx)
            visible_letters = "A/B/C/D/E" if use_5opt else "A/B/C/D"

        q = rng.choice(_QUESTION_STEMS)
        q += f" Reply with the single letter ({visible_letters}) inside <answer>...</answer>. Example: <answer>A</answer>"
        if use_5opt:
            q += " (Option E means none of A-D is the correct next figure.)"
        if level == 0:
            # Concise hint — leak answer (was failing R3 anyway).
            q += (
                f" Hint: count the shapes in each box; count grows by 1 each step. Answer: {answer}."
            )
        elif level == 1:
            q += " Hint: count items per box (rule = +1 each step). Be concise."
        elif level <= 3:
            q += " Hint: identify the single attribute that changes (count/size/rotation), apply once more. Be concise."

        layout = rng.choice(cfg["layout_pool"])
        title = rng.choice(_TITLE_POOL)
        # If use_trap, we didn't store options correctly — pass options including E label slot.
        if use_trap:
            # Render only the 4 visible non-correct options; the prompt option E
            # is "No correct answer" (text in question, not rendered as figure).
            render_opts = options
        else:
            render_opts = options
        image = self._render(seq, render_opts, palette, cfg, layout, title, rng,
                             trap_label_E=use_5opt)
        return q, answer, image

    # ------------------------------ drawing ------------------------------ #

    def _draw_cell(self, ax, cx, cy, cell, palette):
        shape = cell["shape"]
        n = cell["count"]
        s = cell["size"]
        color = palette[cell["color_idx"] % len(palette)]
        rot_deg = cell["rot"] * 90
        for k in range(n):
            ox = (k - (n - 1) / 2.0) * s * 1.15
            cx2 = cx + ox
            pts = _shape_pts(shape, cx2, cy, s * 0.45)
            if pts is None:
                patch = mpatches.Circle((cx2, cy), s * 0.45,
                                         facecolor=color,
                                         edgecolor="#1a1a1a", linewidth=1.0)
            else:
                patch = mpatches.Polygon(pts, closed=True, facecolor=color,
                                          edgecolor="#1a1a1a", linewidth=1.0)
            if rot_deg:
                patch.set_transform(
                    Affine2D().rotate_deg_around(cx2, cy, rot_deg)
                    + ax.transData)
            ax.add_patch(patch)

    def _cell_positions(self, n, layout, cell_w=1.6):
        """Return (cx, cy) for n cells + the ? cell at the end."""
        if layout == "column":
            pts = [(1.0, -i * cell_w) for i in range(n + 1)]
        elif layout == "zigzag":
            pts = []
            for i in range(n + 1):
                row = i % 2
                col = i // 2
                pts.append((col * cell_w + 1.0, 1.0 - row * cell_w))
        elif layout == "ring":
            R = max(2.0, 0.5 * (n + 1))
            pts = []
            for i in range(n + 1):
                a = 2 * math.pi * i / (n + 1) - math.pi / 2
                pts.append((R * math.cos(a), R * math.sin(a)))
        else:  # row
            pts = [(i * cell_w + 1.0, 1.0) for i in range(n + 1)]
        return pts

    def _render(self, seq, options, palette, cfg, layout, title, rng,
                trap_label_E=False):
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        fig = plt.figure(figsize=(12 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_s = fig.add_subplot(2, 1, 1)
        ax_o = fig.add_subplot(2, 1, 2)
        ax_s.set_aspect("equal")
        ax_o.set_aspect("equal")
        ax_s.axis("off")
        ax_o.axis("off")

        n = len(seq)
        cell_w = 1.5
        positions = self._cell_positions(n, layout, cell_w)

        # Draw cells
        for i, (ex, (cx, cy)) in enumerate(zip(seq, positions[:n])):
            rect = mpatches.Rectangle(
                (cx - cell_w / 2 + 0.1, cy - cell_w / 2 + 0.1),
                cell_w - 0.2, cell_w - 0.2,
                facecolor="#ffffff",
                edgecolor="#7f8c8d",
                linewidth=1)
            ax_s.add_patch(rect)
            self._draw_cell(ax_s, cx, cy, ex, palette)
            ax_s.text(cx, cy - cell_w / 2 - 0.18, f"{i + 1}",
                      fontsize=fs - 1, ha="center", va="top", family=ff)

        # Question-mark cell at the end.
        qx, qy = positions[n]
        rect = mpatches.Rectangle(
            (qx - cell_w / 2 + 0.1, qy - cell_w / 2 + 0.1),
            cell_w - 0.2, cell_w - 0.2,
            facecolor="#ffeaa7", edgecolor="#c0392b", linewidth=1.5)
        ax_s.add_patch(rect)
        ax_s.text(qx, qy, "?", fontsize=fs + 8,
                  fontweight="bold", ha="center", va="center",
                  color="#c0392b", family=ff)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        pad = 1.2
        ax_s.set_xlim(min(xs) - pad, max(xs) + pad)
        ax_s.set_ylim(min(ys) - pad, max(ys) + pad)
        ax_s.set_title(title, fontsize=fs + 1, family=ff)

        # Noise dots at high levels
        if cfg["use_noise"]:
            for _ in range(12):
                nx = rng.uniform(min(xs) - pad, max(xs) + pad)
                ny = rng.uniform(min(ys) - pad, max(ys) + pad)
                ax_s.plot(nx, ny, ".",
                          color="#cccccc", markersize=1.5, alpha=0.4)

        # Options row
        for i, opt in enumerate(options):
            cx = i * cell_w + 1.0
            cy = 1.0
            rect = mpatches.Rectangle(
                (cx - cell_w / 2 + 0.1, cy - cell_w / 2 + 0.1),
                cell_w - 0.2, cell_w - 0.2,
                facecolor="#ffffff", edgecolor="#34495e", linewidth=1)
            ax_o.add_patch(rect)
            self._draw_cell(ax_o, cx, cy, opt, palette)
            ax_o.text(cx, cy - cell_w / 2 - 0.18,
                      f"({chr(ord('A') + i)})",
                      fontsize=fs, fontweight="bold",
                      ha="center", va="top", family=ff)
        # 2026-05-05 R5 B4B: render "E. No correct answer" text panel for L6+
        n_visible = len(options)
        if trap_label_E:
            # E slot: text-only, no figure
            cx = n_visible * cell_w + 1.0
            cy = 1.0
            rect = mpatches.Rectangle(
                (cx - cell_w / 2 + 0.1, cy - cell_w / 2 + 0.1),
                cell_w - 0.2, cell_w - 0.2,
                facecolor="#fafafa", edgecolor="#34495e", linewidth=1,
                linestyle="--")
            ax_o.add_patch(rect)
            ax_o.text(cx, cy, "No correct\nanswer",
                      fontsize=fs - 1, ha="center", va="center",
                      family=ff, color="#1a1a1a", style="italic")
            ax_o.text(cx, cy - cell_w / 2 - 0.18, "(E)",
                      fontsize=fs, fontweight="bold",
                      ha="center", va="top", family=ff)
            ax_o.set_xlim(-0.2, (n_visible + 1) * cell_w + 1)
        else:
            ax_o.set_xlim(-0.2, max(4, n_visible) * cell_w + 1)
        ax_o.set_ylim(-0.4, 2.5)
        ax_o.set_title(rng.choice(_OPT_TITLE_POOL), fontsize=fs + 1, family=ff)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                            hspace=0.4)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_fsn"
    os.makedirs(out_dir, exist_ok=True)
    env = FigureSequenceNextQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[figure_sequence_next L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"figure_sequence_next_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[figure_sequence_next L{level} s{s}] A={env._answer}")
