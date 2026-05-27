"""
Attribute Count Quantize QA.

2026-05-05 R5 B5A: REWRITTEN — pure single-letter MCQ A-D / A-E.

Two task modes (selected by level):

  COUNT mode (L0-L4) — render N simple shapes (circles, squares,
  triangles, stars). Ask "How many <shape> are in the image?" or
  "How many <color> objects?". 4-MCQ with the correct count + 3 close
  distractors (correct ± 1, ± 2).

  GROUP mode (L5-L9) — render 6 figures with countable features (dots,
  polygon sides, line bundles, ...). The figures split into two groups
  of 3 by a count-based rule. Pick the partition. 4-MCQ at L5/L6,
  5-MCQ + trap rate at L7-L9.

Per-level layout
  L0/L1 — COUNT, 4-MCQ. 4-6 simple shapes, 1-2 colors. Concise hint.
  L2/L3 — COUNT, 4-MCQ. 7-10 shapes, multi-attribute (color+shape).
  L4    — COUNT, 4-MCQ. 11-15 shapes.
  L5/L6 — GROUP, 4-MCQ. Even-odd or simple count rule. Single layout.
  L7    — GROUP, 5-MCQ + 25% trap. Mid rules.
  L8/L9 — GROUP, 5-MCQ + 30/40% trap. Hard rules.
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


# ---------------------------------------------------------------------- #
# COUNT mode — render N simple labeled shapes
# ---------------------------------------------------------------------- #
_SHAPE_NAMES = ["circle", "square", "triangle", "star"]
_COLOR_NAMES = ["red", "blue", "green", "purple"]
_COLOR_HEX = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "purple": "#9b59b6",
}

_COUNT_STEM_SHAPE = [
    "How many {target} are shown in the image ( )?",
    "Count the {target} in the image. How many are there ( )?",
    "Look at the image. How many {target} can you see ( )?",
    "The image contains shapes of different kinds. How many {target} are there ( )?",
]
_COUNT_STEM_COLOR = [
    "How many {target} objects are shown in the image ( )?",
    "Count the {target} shapes in the image. How many are there ( )?",
    "Look at the image and count the {target} objects ( ).",
    "How many shapes in the image are colored {target} ( )?",
]


def _draw_simple_shape(ax, cx, cy, s, shape_name, color):
    """Draw one shape centered at (cx, cy) with bounding-radius s."""
    if shape_name == "circle":
        ax.add_patch(mpatches.Circle((cx, cy), s,
                                     facecolor=color, edgecolor="black",
                                     linewidth=1.6, zorder=3))
    elif shape_name == "square":
        ax.add_patch(mpatches.Rectangle((cx - s, cy - s), 2 * s, 2 * s,
                                        facecolor=color, edgecolor="black",
                                        linewidth=1.6, zorder=3))
    elif shape_name == "triangle":
        verts = [
            (cx, cy + s * 1.0),
            (cx - s * 0.95, cy - s * 0.7),
            (cx + s * 0.95, cy - s * 0.7),
        ]
        ax.add_patch(mpatches.Polygon(verts, closed=True,
                                      facecolor=color, edgecolor="black",
                                      linewidth=1.6, zorder=3))
    elif shape_name == "star":
        # 5-point star
        verts = []
        for i in range(10):
            a = math.pi / 2 + i * math.pi / 5
            r = s if i % 2 == 0 else s * 0.45
            verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        ax.add_patch(mpatches.Polygon(verts, closed=True,
                                      facecolor=color, edgecolor="black",
                                      linewidth=1.6, zorder=3))


# ---------------------------------------------------------------------- #
# GROUP mode — count-feature figures (legacy diversity)
# ---------------------------------------------------------------------- #
def _draw_dot_cluster(ax, cx, cy, s, n, color, lw):
    inner = s * 0.55
    if n == 1:
        positions = [(cx, cy)]
    elif n == 2:
        positions = [(cx - inner * 0.5, cy), (cx + inner * 0.5, cy)]
    elif n == 3:
        positions = [
            (cx, cy + inner * 0.4),
            (cx - inner * 0.5, cy - inner * 0.3),
            (cx + inner * 0.5, cy - inner * 0.3)]
    elif n == 4:
        positions = [
            (cx - inner * 0.4, cy + inner * 0.4),
            (cx + inner * 0.4, cy + inner * 0.4),
            (cx - inner * 0.4, cy - inner * 0.4),
            (cx + inner * 0.4, cy - inner * 0.4)]
    elif n == 5:
        positions = [
            (cx - inner * 0.4, cy + inner * 0.4),
            (cx + inner * 0.4, cy + inner * 0.4),
            (cx, cy),
            (cx - inner * 0.4, cy - inner * 0.4),
            (cx + inner * 0.4, cy - inner * 0.4)]
    elif n == 6:
        positions = [
            (cx - inner * 0.5, cy + inner * 0.5),
            (cx + inner * 0.5, cy + inner * 0.5),
            (cx - inner * 0.5, cy),
            (cx + inner * 0.5, cy),
            (cx - inner * 0.5, cy - inner * 0.5),
            (cx + inner * 0.5, cy - inner * 0.5)]
    else:
        cols = max(3, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        usable = s * 1.4
        dx = usable / max(cols, 1)
        dy = usable / max(rows, 1)
        positions = []
        start_x = cx - usable / 2 + dx / 2
        start_y = cy + usable / 2 - dy / 2
        for i in range(n):
            r = i // cols
            c = i % cols
            positions.append((start_x + c * dx, start_y - r * dy))
    dot_r = s * 0.10 if n <= 6 else max(s * 0.045, s * 0.10 * (6 / n))
    for px, py in positions[:n]:
        d = mpatches.Circle((px, py), dot_r, facecolor=color,
                            edgecolor=color, linewidth=1.0, zorder=4)
        ax.add_patch(d)


def _draw_regular_polygon(ax, cx, cy, s, n_sides, color, lw):
    verts = []
    for i in range(n_sides):
        a = math.pi / 2 + 2 * math.pi * i / n_sides
        verts.append((cx + s * 0.88 * math.cos(a),
                      cy + s * 0.88 * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n_sides - 1) + [mpath.Path.CLOSEPOLY]
    p = mpath.Path(verts, codes)
    ax.add_patch(mpatches.PathPatch(p, facecolor="none", edgecolor=color,
                                    linewidth=lw, zorder=3))


def _draw_line_bundle(ax, cx, cy, s, n_lines, color, lw):
    spacing = s * 1.5 / max(n_lines + 1, 1)
    y0 = cy + (n_lines - 1) / 2 * spacing
    eff_lw = lw if n_lines <= 6 else max(0.8, lw * 6 / n_lines)
    for i in range(n_lines):
        y = y0 - i * spacing
        ax.plot([cx - s * 0.85, cx + s * 0.85], [y, y],
                color=color, linewidth=eff_lw, zorder=3)


def _draw_star_arms(ax, cx, cy, s, n_arms, color, lw):
    eff_lw = lw if n_arms <= 8 else max(0.8, lw * 8 / n_arms)
    for i in range(n_arms):
        a = math.pi / 2 + 2 * math.pi * i / n_arms
        x = cx + s * 0.85 * math.cos(a)
        y = cy + s * 0.85 * math.sin(a)
        ax.plot([cx, x], [cy, y], color=color, linewidth=eff_lw, zorder=3)
    ax.add_patch(mpatches.Circle((cx, cy), s * 0.05,
                                 facecolor=color, edgecolor=color, zorder=4))


_GROUP_FIG_TEMPLATES = [
    ("dots", _draw_dot_cluster, 1, 12, "dots inside"),
    ("polygon", _draw_regular_polygon, 3, 10, "sides"),
    ("lines", _draw_line_bundle, 1, 10, "parallel lines"),
    ("star", _draw_star_arms, 3, 10, "arms"),
]


def _rule_even(n): return 1 if n % 2 == 0 else 0
def _rule_prime(n): return 1 if n in (2, 3, 5, 7, 11, 13) else 0
def _rule_multiple_of_3(n): return 1 if n % 3 == 0 else 0
def _rule_perfect_square(n): return 1 if n in (1, 4, 9, 16) else 0


_RULES_EASY = [("even_odd", _rule_even),
                ("multiple_of_3", _rule_multiple_of_3)]
_RULES_MEDIUM = [("even_odd", _rule_even),
                  ("prime_composite", _rule_prime),
                  ("multiple_of_3", _rule_multiple_of_3),
                  ("perfect_square", _rule_perfect_square)]


_GROUP_STEM = [
    "The six figures each contain a different number of some countable feature. Divide them into two groups of three such that each group shares the same count-based property. Which grouping is correct ( )?",
    "Each of the six figures has a countable feature (dots, sides, lines, arms). Group them into two sets of three based on a shared count property. Which partition is correct ( )?",
    "The image shows six figures with varying counts of a visual element. Split them into two groups of three using a count-based rule. Which grouping is correct ( )?",
    "In the figure, each of the six shapes has a different count of some feature. Which option partitions them into two count-based groups of three ( )?",
]


class AttributeCountQuantizeQA(StandaloneVisualEnv):
    ENV_NAME = "attribute_count_quantize"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Count the target carefully. "
        "Final answer in the format this prompt requested."
    )

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "mode": "count",
                "n_objects": 5 if level == 0 else 6,
                "shape_pool": ["circle", "square", "triangle"],
                "color_pool": ["red", "blue", "green"],
                "task": "shape" if level == 0 else "shape",
                "use_5_options": False,
                "trap_rate": 0.0,
            }
        if level <= 3:
            return {
                "mode": "count",
                "n_objects": 8 + (level - 2) * 2,  # 8, 10
                "shape_pool": _SHAPE_NAMES,
                "color_pool": _COLOR_NAMES,
                "task": "shape" if level == 2 else "color",
                "use_5_options": False,
                "trap_rate": 0.0,
            }
        if level == 4:
            return {
                "mode": "count",
                "n_objects": 12,
                "shape_pool": _SHAPE_NAMES,
                "color_pool": _COLOR_NAMES,
                "task": "shape",
                "use_5_options": False,
                "trap_rate": 0.0,
            }
        if level <= 6:
            return {
                "mode": "group",
                "rules": _RULES_EASY,
                "force_big_gap": True,
                "use_5_options": False,
                "trap_rate": 0.0,
                "n_min": 2, "n_max": 9,
            }
        if level == 7:
            return {
                "mode": "group",
                "rules": _RULES_EASY,
                "force_big_gap": False,
                "use_5_options": True,
                "trap_rate": 0.25,
                "n_min": 2, "n_max": 10,
            }
        if level == 8:
            return {
                "mode": "group",
                "rules": _RULES_MEDIUM,
                "force_big_gap": False,
                "use_5_options": True,
                "trap_rate": 0.30,
                "n_min": 2, "n_max": 10,
            }
        # L9
        return {
            "mode": "group",
            "rules": _RULES_MEDIUM,
            "force_big_gap": False,
            "use_5_options": True,
            "trap_rate": 0.40,
            "n_min": 2, "n_max": 11,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 31 + level * 29 + 71)
        for _ in range(50):
            try:
                if cfg["mode"] == "count":
                    res = self._try_generate_count(cfg, level, rng)
                else:
                    res = self._try_generate_group(cfg, level, rng)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # COUNT mode generator
    # ------------------------------------------------------------------ #
    def _try_generate_count(self, cfg: Dict, level: int, rng: random.Random):
        n_objects = cfg["n_objects"]
        shape_pool = list(cfg["shape_pool"])
        color_pool = list(cfg["color_pool"])
        task = cfg["task"]

        # Generate n_objects items, each with a (shape, color)
        objects: List[Tuple[str, str]] = []
        for _ in range(n_objects):
            shape = rng.choice(shape_pool)
            color = rng.choice(color_pool)
            objects.append((shape, color))

        # Determine target attribute for the question
        if task == "shape":
            # Pick a shape that has at least 1 instance and at most n-1 (so it
            # is not the only shape).
            counts = {s: 0 for s in shape_pool}
            for sh, _ in objects:
                counts[sh] += 1
            valid_targets = [s for s, c in counts.items() if 1 <= c < n_objects]
            if not valid_targets:
                return None
            target = rng.choice(valid_targets)
            true_count = counts[target]
            target_text = self._pluralize(target)
            stem_pool = _COUNT_STEM_SHAPE
        else:
            counts = {c: 0 for c in color_pool}
            for _, col in objects:
                counts[col] += 1
            valid_targets = [c for c, n in counts.items() if 1 <= n < n_objects]
            if not valid_targets:
                return None
            target = rng.choice(valid_targets)
            true_count = counts[target]
            target_text = target
            stem_pool = _COUNT_STEM_COLOR

        # Distractors: ±1, ±2, ±3 (clipped to >= 0 and <= n_objects + 2)
        distractors = []
        for delta in (-2, -1, 1, 2, 3):
            cand = true_count + delta
            if cand < 0 or cand > n_objects + 2:
                continue
            if cand == true_count:
                continue
            distractors.append(cand)
        # Dedupe
        distractors = list(dict.fromkeys(distractors))
        rng.shuffle(distractors)

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)
        n_distractor = n_value - 1
        chosen = distractors[:n_distractor]
        if len(chosen) < n_distractor:
            return None
        options_int = [true_count] + chosen
        rng.shuffle(options_int)
        correct_letter = opt_letters[options_int.index(true_count)]
        opt_lines = [f"{opt_letters[i]}. {options_int[i]}" for i in range(n_value)]
        if use_5:
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(stem_pool)
        stem = stem_pool[sidx].format(target=target_text)
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )
        if level <= 1:
            question += (
                f"\nHint: count each {target} carefully one by one, then "
                f"pick the matching number."
            )
        elif level <= 3:
            question += (
                f"\nHint: scan the image left-to-right and tally the "
                f"target attribute."
            )

        img = self._render_count(objects, task, target, level)
        return question, correct_letter, img

    @staticmethod
    def _pluralize(shape_name: str) -> str:
        # Simple pluralization
        if shape_name.endswith("y"):
            return shape_name[:-1] + "ies"
        if shape_name.endswith("s"):
            return shape_name + "es"
        return shape_name + "s"

    def _render_count(self, objects, task, target, level) -> Image.Image:
        n = len(objects)
        # Pick a layout: roughly square grid
        cols = max(3, math.ceil(math.sqrt(n) * 1.1))
        rows = math.ceil(n / cols)
        cell = 1.2
        fig_w = max(6.5, cols * cell + 1.0)
        fig_h = max(4.5, rows * cell + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")

        # Place each object on grid + slight jitter to avoid mechanical look
        rng = self._rng
        for i, (shape, color) in enumerate(objects):
            r = i // cols
            c = i % cols
            cx = c * cell + cell / 2
            cy = (rows - 1 - r) * cell + cell / 2
            # jitter
            jx = rng.uniform(-0.1, 0.1) * cell
            jy = rng.uniform(-0.1, 0.1) * cell
            _draw_simple_shape(ax, cx + jx, cy + jy, cell * 0.36,
                               shape, _COLOR_HEX[color])
        ax.set_xlim(-0.3, cols * cell + 0.3)
        ax.set_ylim(-0.3, rows * cell + 0.3)

        target_text = (target if task == "color"
                       else self._pluralize(target))
        title = f"Count the {target_text}"
        ax.set_title(title, fontsize=15, fontweight="bold", color="#1a1a1a",
                     pad=10)
        return self.fig_to_pil(fig, dpi=120)

    # ------------------------------------------------------------------ #
    # GROUP mode generator (kept similar to original, but pure A-D / A-E)
    # ------------------------------------------------------------------ #
    def _try_generate_group(self, cfg: Dict, level: int, rng: random.Random):
        tmpl_name, draw_fn, tmpl_min, tmpl_max, _ = rng.choice(_GROUP_FIG_TEMPLATES)
        rule_name, rule_fn = rng.choice(cfg["rules"])

        n_lo = max(cfg["n_min"], tmpl_min)
        n_hi = min(cfg["n_max"], tmpl_max)
        if n_hi - n_lo + 1 < 6:
            return None

        available = list(range(n_lo, n_hi + 1))
        group0 = [n for n in available if rule_fn(n) == 0]
        group1 = [n for n in available if rule_fn(n) == 1]
        if len(group0) < 3 or len(group1) < 3:
            return None

        if cfg["force_big_gap"]:
            group0.sort()
            group1.sort()
            picks0 = group0[:3]
            picks1 = group1[-3:]
            if set(picks0) & set(picks1):
                return None
        else:
            picks0 = rng.sample(group0, 3)
            picks1 = rng.sample(group1, 3)
        if len(set(picks0 + picks1)) != 6:
            return None

        positions = list(range(6))
        rng.shuffle(positions)
        fig_at_pos = [None] * 6
        for p, n in zip(positions[:3], picks0):
            fig_at_pos[p] = (n, 0)
        for p, n in zip(positions[3:], picks1):
            fig_at_pos[p] = (n, 1)

        g0_pos = sorted(p for p, (_, g) in enumerate(fig_at_pos) if g == 0)
        g1_pos = sorted(p for p, (_, g) in enumerate(fig_at_pos) if g == 1)
        correct_part = frozenset([frozenset(g0_pos), frozenset(g1_pos)])

        # Distractors from other rules + swap-based
        seen = {correct_part}
        distractor_parts: List[frozenset] = []
        counts_by_pos = {p: n for p, (n, _) in enumerate(fig_at_pos)}
        for rname, rfn in cfg["rules"]:
            g0d, g1d = [], []
            for p in range(6):
                if rfn(counts_by_pos[p]) == 1:
                    g1d.append(p)
                else:
                    g0d.append(p)
            while len(g0d) > 3:
                g1d.append(g0d.pop(rng.randint(0, len(g0d) - 1)))
            while len(g1d) > 3:
                g0d.append(g1d.pop(rng.randint(0, len(g1d) - 1)))
            if len(g0d) != 3 or len(g1d) != 3:
                continue
            part = frozenset([frozenset(g0d), frozenset(g1d)])
            if part not in seen:
                seen.add(part)
                distractor_parts.append(part)

        # Top-up via swaps. Want at least 4 distinct distractors so trap
        # mode (5-MCQ all-wrong, n_value=4) can be filled.
        attempts = 0
        gA = sorted(g0_pos)
        gB = sorted(g1_pos)
        target_distractors = 5  # enough for both trap (4) and non-trap (3)
        while len(distractor_parts) < target_distractors and attempts < 200:
            attempts += 1
            a_i = rng.randint(0, 2)
            b_i = rng.randint(0, 2)
            new_A = list(gA)
            new_B = list(gB)
            new_A[a_i], new_B[b_i] = new_B[b_i], new_A[a_i]
            part = frozenset([frozenset(new_A), frozenset(new_B)])
            if part not in seen:
                seen.add(part)
                distractor_parts.append(part)
            # Try double-swap as alternative
            if len(distractor_parts) < target_distractors and attempts % 5 == 0:
                new_A = list(gA)
                new_B = list(gB)
                a_i2 = rng.randint(0, 2)
                b_i2 = rng.randint(0, 2)
                new_A[a_i], new_B[b_i] = new_B[b_i], new_A[a_i]
                if a_i2 != a_i and b_i2 != b_i:
                    new_A[a_i2], new_B[b_i2] = new_B[b_i2], new_A[a_i2]
                part = frozenset([frozenset(new_A), frozenset(new_B)])
                if part not in seen:
                    seen.add(part)
                    distractor_parts.append(part)

        rng.shuffle(distractor_parts)

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            chosen = distractor_parts[:n_value]
            if len(chosen) < n_value:
                return None
            options_part = list(chosen)
            rng.shuffle(options_part)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distractor_parts[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options_part = [correct_part] + chosen
            rng.shuffle(options_part)
            correct_letter = opt_letters[options_part.index(correct_part)]

        def _part_to_str(part):
            parts = sorted(part, key=lambda s: min(s))
            left = ", ".join(str(p + 1) for p in sorted(parts[0]))
            right = ", ".join(str(p + 1) for p in sorted(parts[1]))
            return f"{{{left}}} | {{{right}}}"

        opt_lines = [f"{opt_letters[i]}. {_part_to_str(options_part[i])}"
                     for i in range(n_value)]
        if use_5:
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(_GROUP_STEM)
        stem = _GROUP_STEM[sidx]
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )

        img = self._render_group(fig_at_pos, draw_fn, options_part, level, rng)
        return question, correct_letter, img

    def _render_group(self, fig_at_pos, draw_fn, options_part, level, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        palette = list(style["palette"])
        rng.shuffle(palette)
        line_color = palette[0]
        lw = max(1.8, style["line_width"])
        bg = "#ffffff"
        ff = style["font_family"]

        fig = plt.figure(figsize=(8.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(bg)
        gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.4], hspace=0.30)
        ax_fig = fig.add_subplot(gs[0])
        ax_opt = fig.add_subplot(gs[1])
        ax_fig.set_facecolor(bg)
        ax_opt.set_facecolor(bg)

        # Single-row layout
        positions = []
        for i in range(6):
            positions.append((i * 1.1 + 0.6, 1.0))
        ax_fig.set_xlim(-0.2, 6.8)
        ax_fig.set_ylim(0.0, 2.0)
        ax_fig.set_aspect("equal")
        ax_fig.axis("off")
        ax_fig.set_title("Six figures", fontsize=style["font_size_base"] + 2,
                         fontweight="bold")

        for pos, ((n, _), (cx, cy)) in enumerate(zip(fig_at_pos, positions)):
            rect = mpatches.FancyBboxPatch(
                (cx - 0.44, cy - 0.44), 0.88, 0.88,
                boxstyle="round,pad=0.02", facecolor=bg,
                edgecolor="#bdc3c7", linewidth=1.2, zorder=1)
            ax_fig.add_patch(rect)
            fig_color = palette[(pos + 1) % len(palette)]
            draw_fn(ax_fig, cx, cy, 0.32, n, fig_color, lw)
            ax_fig.text(cx, cy - 0.52, str(pos + 1),
                        fontsize=style["font_size_base"] + 1,
                        fontweight="bold", ha="center", va="top",
                        color="#2c3e50", fontfamily=ff)

        # Show options as text
        ax_opt.set_xlim(0, 1)
        ax_opt.set_ylim(0, 1)
        ax_opt.axis("off")
        n_opt = len(options_part)
        for i, part in enumerate(options_part):
            parts = sorted(part, key=lambda s: min(s))
            left = ", ".join(str(p + 1) for p in sorted(parts[0]))
            right = ", ".join(str(p + 1) for p in sorted(parts[1]))
            y = 0.85 - i * (0.85 / max(1, n_opt))
            ax_opt.text(0.02, y, f"{chr(ord('A') + i)}. ",
                        fontsize=style["font_size_base"] + 2,
                        fontweight="bold", color="#2c3e50",
                        va="center", fontfamily=ff)
            ax_opt.text(0.08, y, f"{{{left}}}  |  {{{right}}}",
                        fontsize=style["font_size_base"] + 1,
                        color="#2c3e50", va="center", fontfamily=ff)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = AttributeCountQuantizeQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   pos={v_pos['accuracy']} neg={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
