"""
Pictogram-rule application QA.

Layout: a top row shows N example input→output transformations (each pair:
"input shape group" maps to "output shape group" by some rule). Then a
single new input is shown with its output marked '?'. The model picks the
correct output from 4 candidates.

Rules implemented (pictogram-style):
  - shape_swap: input shape X is replaced by shape Y (mapping).
  - count_to_count: number of input items determines output items
    (e.g., k input squares -> k output circles).
  - color_swap: input color X is mapped to output color Y.
  - count_match: output count equals count of input attribute (e.g.,
    "number of grey arrows out = number of black-shaded shapes in").

Difficulty levels 0-9:
  L0-L1: 2 example pairs, shape_swap rule, 3 distractors (clearly wrong).
  L2-L3: 3 example pairs, shape_swap or color_swap.
  L4-L5: 3 example pairs, count_match rule with small counts.
  L6-L7: 3 example pairs, count_to_count rule.
  L8-L9: 3 example pairs, mixed shape+count rule (each input item maps to
         output item by both shape AND count).
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


_SHAPE_TYPES = ["circle", "square", "triangle", "diamond", "hexagon", "pentagon"]

_QUESTION_TEMPLATES = [
    "Each example pair shows an 'input → output' transformation. Apply the same rule to the new input and pick the correct output: A, B, C, or D. Single letter.",
    "The bottom example pairs define a transformation rule. Apply that rule to the input on top, then choose the matching output (A/B/C/D).",
    "Examine the example transformations carefully. Which option (A, B, C, D) shows the correct output for the new input?",
    "Each input maps to an output by a hidden rule. Pick the option (A-D) that follows that rule for the new input. Single letter answer.",
    "Identify the transformation rule from the examples and apply it. Which of A, B, C, D is the correct output?",
    "Look at the example pairs (input → output). Which candidate (A, B, C, D) is the correct output for the new input?",
    "Determine the rule from the examples and select the matching output among A, B, C, D.",
    "After working out the rule, choose the option (A-D) that gives the right output. Single letter only.",
    "The examples illustrate the transformation. Which of the four options (A, B, C, D) applies it correctly?",
    "Work out what the examples show, then pick the correct output for the new input: A, B, C, or D.",
    "Pick the option (A, B, C, D) that follows the example pattern. Reply with one letter.",
    "The examples encode a rule. Apply it to the new input and pick A, B, C, or D as the output.",
    "Choose the output that matches the rule shown in the examples. (A) (B) (C) (D).",
    "Inspect each input → output example. The correct output for the new input is which letter (A-D)?",
    "Which letter (A, B, C, D) shows the correct output for the new input, following the rule in the examples?",
    "Apply the demonstrated transformation. Which option (A, B, C, D) is the right output?",
]


def _draw_shape(ax, shape: str, cx: float, cy: float, size: float,
                color: str, edge: str = "#222", linewidth: float = 1.2,
                fill: bool = True, zorder: int = 3):
    fc = color if fill else "none"
    if shape == "circle":
        ax.add_patch(mpatches.Circle((cx, cy), size, facecolor=fc,
                                     edgecolor=edge, linewidth=linewidth,
                                     zorder=zorder))
    elif shape == "square":
        ax.add_patch(mpatches.Rectangle(
            (cx - size, cy - size), 2 * size, 2 * size, facecolor=fc,
            edgecolor=edge, linewidth=linewidth, zorder=zorder))
    elif shape == "triangle":
        verts = [(cx, cy + size), (cx - size, cy - size * 0.85),
                 (cx + size, cy - size * 0.85)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "diamond":
        verts = [(cx, cy + size), (cx + size, cy),
                 (cx, cy - size), (cx - size, cy)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "pentagon":
        verts = [(cx + size * math.cos(math.radians(72 * i + 90)),
                  cy + size * math.sin(math.radians(72 * i + 90)))
                 for i in range(5)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "hexagon":
        verts = [(cx + size * math.cos(math.radians(60 * i + 30)),
                  cy + size * math.sin(math.radians(60 * i + 30)))
                 for i in range(6)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))


def _grid_positions(n: int, cell_w: float, cell_h: float):
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-cell_w * 0.20, 0.0), (cell_w * 0.20, 0.0)]
    if n == 3:
        return [(-cell_w * 0.25, 0.0), (0.0, 0.0), (cell_w * 0.25, 0.0)]
    if n == 4:
        return [(-cell_w * 0.22, cell_h * 0.18), (cell_w * 0.22, cell_h * 0.18),
                (-cell_w * 0.22, -cell_h * 0.18), (cell_w * 0.22, -cell_h * 0.18)]
    if n == 5:
        return [(-cell_w * 0.27, cell_h * 0.18), (0.0, cell_h * 0.18),
                (cell_w * 0.27, cell_h * 0.18),
                (-cell_w * 0.18, -cell_h * 0.18),
                (cell_w * 0.18, -cell_h * 0.18)]
    out = []
    for i in range(n):
        ang = 2 * math.pi * i / n
        out.append((cell_w * 0.27 * math.cos(ang),
                    cell_h * 0.27 * math.sin(ang)))
    return out


class PictogramRuleQA(StandaloneVisualEnv):
    ENV_NAME = "pictogram_rule"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"n_examples": 2, "rule_pool": ["shape_swap"], "max_count": 2}
        if level <= 3:
            return {"n_examples": 3, "rule_pool": ["shape_swap", "color_swap"], "max_count": 2}
        if level <= 5:
            return {"n_examples": 3, "rule_pool": ["count_match"], "max_count": 4}
        if level <= 7:
            return {"n_examples": 3, "rule_pool": ["count_to_count"], "max_count": 6}
        return {"n_examples": 3,
                "rule_pool": ["shape_swap", "count_match", "count_to_count", "color_swap"],
                "max_count": 6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1011)

        for _ in range(20):
            result = self._try_generate(sub_rng, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        rule = rng.choice(cfg["rule_pool"])
        n_examples = cfg["n_examples"]
        max_count = cfg["max_count"]

        # Pick visual palette
        palette = self._random_style()["palette"]
        # Two distinct colors for input/output unless rule is color_swap
        in_color = palette[2]
        out_color = palette[0]

        if rule == "shape_swap":
            # Input shape -> output shape (fixed mapping). Single shape per cell.
            in_shape, out_shape = rng.sample(_SHAPE_TYPES, 2)
            examples = []
            used_counts = set()
            for _ in range(n_examples):
                k = rng.randint(1, max_count)
                # avoid repeating count to vary visuals
                attempts = 0
                while k in used_counts and attempts < 5:
                    k = rng.randint(1, max_count)
                    attempts += 1
                used_counts.add(k)
                examples.append({
                    "input": {"shape": in_shape, "count": k, "color": in_color},
                    "output": {"shape": out_shape, "count": k, "color": out_color},
                })
            # New input
            new_k = rng.randint(1, max_count)
            new_input = {"shape": in_shape, "count": new_k, "color": in_color}
            correct_output = {"shape": out_shape, "count": new_k, "color": out_color}

            distractors = [
                {"shape": in_shape, "count": new_k, "color": out_color},  # wrong shape
                {"shape": out_shape, "count": max(1, new_k - 1), "color": out_color},  # wrong count
                {"shape": rng.choice([s for s in _SHAPE_TYPES
                                       if s not in (in_shape, out_shape)]),
                 "count": new_k, "color": out_color},  # wrong shape (3rd)
                {"shape": out_shape, "count": new_k + 1, "color": out_color},  # wrong count up
            ]

        elif rule == "color_swap":
            in_color = palette[2]
            out_color = palette[5 if len(palette) > 5 else 1]
            shape = rng.choice(_SHAPE_TYPES)
            examples = []
            for _ in range(n_examples):
                k = rng.randint(1, max_count)
                examples.append({
                    "input": {"shape": shape, "count": k, "color": in_color},
                    "output": {"shape": shape, "count": k, "color": out_color},
                })
            new_k = rng.randint(1, max_count)
            new_input = {"shape": shape, "count": new_k, "color": in_color}
            correct_output = {"shape": shape, "count": new_k, "color": out_color}
            other_color = palette[3]
            distractors = [
                {"shape": shape, "count": new_k, "color": in_color},  # color unchanged
                {"shape": shape, "count": new_k, "color": other_color},  # wrong color
                {"shape": shape, "count": max(1, new_k - 1), "color": out_color},  # wrong count
                {"shape": rng.choice([s for s in _SHAPE_TYPES if s != shape]),
                 "count": new_k, "color": out_color},  # wrong shape
            ]

        elif rule == "count_match":
            # Input: m of shape A. Output: same m of shape B (count comes from
            # input count). Same as shape_swap structurally; treat as same.
            in_shape, out_shape = rng.sample(_SHAPE_TYPES, 2)
            examples = []
            used_counts = set()
            for _ in range(n_examples):
                k = rng.randint(1, max_count)
                attempts = 0
                while k in used_counts and attempts < 5:
                    k = rng.randint(1, max_count)
                    attempts += 1
                used_counts.add(k)
                examples.append({
                    "input": {"shape": in_shape, "count": k, "color": in_color},
                    "output": {"shape": out_shape, "count": k, "color": out_color},
                })
            new_k = rng.randint(1, max_count)
            new_input = {"shape": in_shape, "count": new_k, "color": in_color}
            correct_output = {"shape": out_shape, "count": new_k, "color": out_color}
            distractors = [
                {"shape": out_shape, "count": max(1, new_k - 1), "color": out_color},
                {"shape": out_shape, "count": new_k + 1, "color": out_color},
                {"shape": in_shape, "count": new_k, "color": out_color},
                {"shape": out_shape, "count": 1 if new_k > 1 else max_count,
                 "color": out_color},
            ]

        else:  # count_to_count: count(in) -> 2 * count(in) (or similar mult)
            in_shape, out_shape = rng.sample(_SHAPE_TYPES, 2)
            multiplier = rng.choice([2, 3])
            max_k = max(2, max_count // multiplier)
            # We need >=2 distinct k values to make the rule inferrable.
            if max_k < 2:
                return None
            examples = []
            chosen_ks = rng.sample(range(1, max_k + 1), min(n_examples, max_k))
            # If we need more examples than distinct k values, repeat some.
            while len(chosen_ks) < n_examples:
                chosen_ks.append(rng.randint(1, max_k))
            for k in chosen_ks:
                examples.append({
                    "input": {"shape": in_shape, "count": k, "color": in_color},
                    "output": {"shape": out_shape, "count": k * multiplier,
                               "color": out_color},
                })
            new_k = rng.randint(1, max_k)
            new_input = {"shape": in_shape, "count": new_k, "color": in_color}
            correct_output = {"shape": out_shape, "count": new_k * multiplier,
                              "color": out_color}
            distractors = [
                {"shape": out_shape, "count": new_k, "color": out_color},  # forgot mult
                {"shape": out_shape, "count": new_k * multiplier - 1,
                 "color": out_color},  # off-by-one
                {"shape": out_shape, "count": new_k * multiplier + 1,
                 "color": out_color},  # off-by-one
                {"shape": in_shape, "count": new_k * multiplier,
                 "color": out_color},  # wrong shape
            ]

        # Select 3 unique distractors that are not equal to correct
        rng.shuffle(distractors)
        chosen = []
        for d in distractors:
            if self._cells_equal(d, correct_output):
                continue
            if any(self._cells_equal(d, c) for c in chosen):
                continue
            chosen.append(d)
            if len(chosen) >= 3:
                break
        if len(chosen) < 3:
            return None

        candidates = [correct_output] + chosen
        rng.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates)
                           if self._cells_equal(c, correct_output))
        answer_letter = chr(ord("A") + correct_idx)

        img = self._render(examples, new_input, candidates, rng)
        sidx = (self.seed or 0) % len(_QUESTION_TEMPLATES)
        question = _QUESTION_TEMPLATES[sidx]
        return question, answer_letter, img

    @staticmethod
    def _cells_equal(a: Dict, b: Dict) -> bool:
        return (a["shape"] == b["shape"] and a["count"] == b["count"]
                and a["color"] == b["color"])

    def _render(self, examples: List[Dict], new_input: Dict,
                candidates: List[Dict], rng) -> Image.Image:
        style = self._random_style()
        n_ex = len(examples)
        n_cands = len(candidates)
        # Layout:
        #   Row 1 (top): example pairs, each pair = [input cell] -> [output cell]
        #   Row 2 (mid): new input -> ?
        #   Row 3 (bottom): candidates A-D
        fig_w = max(8.0, 2.6 * n_ex)
        fig_h = 8.0
        fig, (ax_ex, ax_input, ax_opts) = plt.subplots(
            3, 1, figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]})
        fig.patch.set_facecolor(style["bg_color"])

        # ----- Examples row -----
        ax_ex.set_facecolor(style["bg_color"])
        ax_ex.set_xlim(0, n_ex * 2.4)
        ax_ex.set_ylim(0, 1.2)
        ax_ex.set_aspect("equal")
        ax_ex.axis("off")
        ax_ex.set_title("Examples (input -> output)", fontsize=12,
                        fontweight="bold")

        cell_w = 0.9
        cell_h = 0.9

        for i, ex in enumerate(examples):
            x_in = i * 2.4 + 0.4
            x_arrow = i * 2.4 + 1.2
            x_out = i * 2.4 + 1.55
            y_mid = 0.6
            self._render_cell(ax_ex, x_in, y_mid, cell_w, cell_h, ex["input"])
            ax_ex.annotate("", xy=(x_out + cell_w / 2 - 0.2, y_mid),
                            xytext=(x_in + cell_w / 2 + 0.05, y_mid),
                            arrowprops=dict(arrowstyle="->", color="#222",
                                            lw=1.6))
            self._render_cell(ax_ex, x_out + cell_w / 2 + 0.05, y_mid,
                              cell_w, cell_h, ex["output"])

        # ----- New input row -----
        ax_input.set_facecolor(style["bg_color"])
        ax_input.set_xlim(0, 6.0)
        ax_input.set_ylim(0, 1.2)
        ax_input.set_aspect("equal")
        ax_input.axis("off")
        ax_input.set_title("New input -> ?", fontsize=12, fontweight="bold",
                           color="#b00")
        x_in = 1.6
        x_q = 4.0
        y_mid = 0.6
        self._render_cell(ax_input, x_in, y_mid, cell_w, cell_h, new_input)
        ax_input.annotate("", xy=(x_q - 0.4, y_mid),
                           xytext=(x_in + cell_w / 2 + 0.05, y_mid),
                           arrowprops=dict(arrowstyle="->", color="#b00",
                                           lw=2.0))
        # Question mark cell
        ax_input.add_patch(mpatches.Rectangle(
            (x_q - cell_w / 2, y_mid - cell_h / 2), cell_w, cell_h,
            facecolor="#fff8f0", edgecolor="#b00", linewidth=2.0,
            linestyle="--", zorder=1))
        ax_input.text(x_q, y_mid, "?", fontsize=28, fontweight="bold",
                      color="#b00", ha="center", va="center", zorder=5)

        # ----- Options row -----
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, n_cands * 1.3)
        ax_opts.set_ylim(0, 1.4)
        ax_opts.set_aspect("equal")
        ax_opts.axis("off")
        ax_opts.set_title("Options", fontsize=11, fontweight="bold")

        for i, cand in enumerate(candidates):
            cx = i * 1.3 + 0.65
            cy = 0.55
            self._render_cell(ax_opts, cx, cy, cell_w, cell_h, cand)
            label = chr(ord("A") + i)
            ax_opts.text(cx, 1.15, label, fontsize=14, fontweight="bold",
                          ha="center", va="center", color="#222")

        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _render_cell(ax, cx: float, cy: float, cell_w: float, cell_h: float,
                     cell: Dict):
        ax.add_patch(mpatches.Rectangle(
            (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
            facecolor="#ffffff", edgecolor="#222", linewidth=1.2, zorder=1))
        positions = _grid_positions(cell["count"], cell_w * 0.85, cell_h * 0.7)
        item_size = 0.10 if cell["count"] <= 2 else 0.07
        for px, py in positions:
            _draw_shape(ax, cell["shape"], cx + px, cy + py,
                        item_size, cell["color"], edge="#222", zorder=3)
