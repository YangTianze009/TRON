"""
Rule Induction Sequence QA environment (v3, redesigned 2026-04-16).

Goal: train inductive reasoning — infer a hidden rule from a few labelled
example transformations, then apply it to a new input. Targets A1/A2/X5
and addresses the reference / inductive regression.

Redesign notes (v3):
  * v2 was marked Grade D — "sequences look very similar". Root causes:
      - Layout always identical (Input col + arrow + Output col).
      - Cells framed in the same cream colour.
      - Only 7 shapes, small colour palette.
      - Example count 3-5 but layout identical.
  * v3 fixes:
      - 4 layout variants: vertical rows, horizontal strips, 2x2 grid,
        circular arrangement.
      - 12 shapes (star, arrow, cross, heart, gear, plus, ...).
      - Expanded colour palette (14 colours).
      - Multi-shape inputs (cluster / pair / grid of 3) at higher levels.
      - Randomized background, cell border style, arrow style.
      - Question template variants (6 phrasings).
      - L0 vs L9 structurally different: L0 = single shape + single rule,
        3 options; L9 = 3-shape cluster + 3-rule composition, 5 options.
      - Title variants, per-row label variants.
"""
import math
import random
from typing import Dict, List, Optional, Tuple, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_COLORS = {
    "red":     "#e74c3c",
    "blue":    "#3498db",
    "green":   "#27ae60",
    "orange":  "#e67e22",
    "purple":  "#8e44ad",
    "yellow":  "#f1c40f",
    "teal":    "#1abc9c",
    "pink":    "#e91e8f",
    "cyan":    "#00bcd4",
    "lime":    "#7cb342",
    "brown":   "#795548",
    "indigo":  "#3f51b5",
    "coral":   "#ff7043",
    "olive":   "#827717",
}
_COLOR_LIST = list(_COLORS.keys())

_SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon", "star",
           "diamond", "cross", "plus_mark", "arrow_up", "heart", "gear"]
_SIZES = ["small", "medium", "large"]
_SIZE_VAL = {"small": 0.18, "medium": 0.28, "large": 0.38}

# Frame styles for cells
_FRAME_FACES = [
    "#eaf2f8", "#fef3c7", "#e7f5e7", "#ffe5e5", "#f0e6f6", "#e0f7fa",
    "#fafafa", "#fff1e6", "#f6f7f8", "#ecf0f1",
]
_FRAME_EDGES = [
    "#7f8c8d", "#2c3e50", "#34495e", "#2c3e50", "#7f8c8d",
]
_ARROW_COLORS = ["#e67e22", "#2980b9", "#27ae60", "#8e44ad", "#c0392b",
                 "#16a085", "#2c3e50"]

_QUERY_FACES = ["#fef3c7", "#fff3cd", "#fef2e8", "#fdf2cb", "#fde2cf"]
_QUERY_EDGES = ["#e74c3c", "#d35400", "#c0392b", "#e67e22"]

# ------------------------------------------------------------------ #
# Path helpers
# ------------------------------------------------------------------ #

def _polygon_path(cx, cy, n, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(n):
        a = off + 2 * math.pi * i / n
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (n - 1) + [
        mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _star_path(cx, cy, size, rotation_deg=0):
    off = math.radians(rotation_deg) + math.pi / 2
    verts = []
    for i in range(10):
        a = off + 2 * math.pi * i / 10
        r = size if i % 2 == 0 else size * 0.45
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 9 + [
        mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _cross_path(cx, cy, size, rotation_deg=0):
    t = size * 0.35
    pts = [
        (-size, -t), (-t, -t), (-t, -size), (t, -size), (t, -t),
        (size, -t), (size, t), (t, t), (t, size), (-t, size),
        (-t, t), (-size, t),
    ]
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    verts = [(cx + p[0] * c - p[1] * s, cy + p[0] * s + p[1] * c)
             for p in pts]
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (
        len(pts) - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _plus_mark_path(cx, cy, size, rotation_deg=0):
    t = size * 0.28
    pts = [
        (-t, -size), (t, -size), (t, -t), (size, -t), (size, t),
        (t, t), (t, size), (-t, size), (-t, t), (-size, t),
        (-size, -t), (-t, -t),
    ]
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    verts = [(cx + p[0] * c - p[1] * s, cy + p[0] * s + p[1] * c)
             for p in pts]
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (
        len(pts) - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _arrow_up_path(cx, cy, size, rotation_deg=0):
    t = size * 0.35
    pts = [
        (0, size), (size * 0.8, size * 0.1), (t, size * 0.1),
        (t, -size), (-t, -size), (-t, size * 0.1),
        (-size * 0.8, size * 0.1),
    ]
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    verts = [(cx + p[0] * c - p[1] * s, cy + p[0] * s + p[1] * c)
             for p in pts]
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (
        len(pts) - 1) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _heart_path(cx, cy, size, rotation_deg=0):
    theta = math.radians(rotation_deg)
    verts = []
    for i in range(80):
        t = i / 80.0 * 2 * math.pi
        hx = 16 * (math.sin(t) ** 3)
        hy = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(
            3 * t) - math.cos(4 * t)
        px = hx / 16 * size
        py = hy / 16 * size
        c, s = math.cos(theta), math.sin(theta)
        verts.append((cx + px * c - py * s, cy + px * s + py * c))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (
        len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

def _gear_path(cx, cy, size, rotation_deg=0, n_teeth=8):
    off = math.radians(rotation_deg)
    inner = size * 0.7
    outer = size
    verts = []
    for i in range(n_teeth * 2):
        a = off + math.pi * i / n_teeth
        r = outer if i % 2 == 0 else inner
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    verts.append(verts[0])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (
        len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(verts, codes)

class _Shape:
    __slots__ = ("shape", "color", "size", "rotation",
                 "n_dots", "has_ring", "has_outline", "reflected")

    def __init__(self, shape="circle", color="red", size="medium",
                 rotation=0, n_dots=0, has_ring=False,
                 has_outline=False, reflected=False):
        self.shape = shape
        self.color = color
        self.size = size
        self.rotation = rotation % 360
        self.n_dots = n_dots
        self.has_ring = has_ring
        self.has_outline = has_outline
        self.reflected = reflected

    def copy(self):
        return _Shape(self.shape, self.color, self.size, self.rotation,
                      self.n_dots, self.has_ring, self.has_outline,
                      self.reflected)

    def __eq__(self, other):
        if not isinstance(other, _Shape):
            return False
        return (self.shape == other.shape and self.color == other.color
                and self.size == other.size
                and self.rotation == other.rotation
                and self.n_dots == other.n_dots
                and self.has_ring == other.has_ring
                and self.has_outline == other.has_outline
                and self.reflected == other.reflected)

    def __hash__(self):
        return hash((self.shape, self.color, self.size, self.rotation,
                     self.n_dots, self.has_ring, self.has_outline,
                     self.reflected))

    def draw(self, ax, cx, cy, cell_size):
        size_val = _SIZE_VAL[self.size] * cell_size
        color_hex = _COLORS[self.color]

        if self.shape == "circle":
            p = mpatches.Circle((cx, cy), size_val, facecolor=color_hex,
                                edgecolor="#2c3e50", linewidth=1.5,
                                alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "star":
            pth = _star_path(cx, cy, size_val, self.rotation)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "diamond":
            pth = _polygon_path(cx, cy, 4, size_val, self.rotation + 45)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "cross":
            pth = _cross_path(cx, cy, size_val, self.rotation)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "plus_mark":
            pth = _plus_mark_path(cx, cy, size_val, self.rotation)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "arrow_up":
            pth = _arrow_up_path(cx, cy, size_val, self.rotation)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "heart":
            pth = _heart_path(cx, cy, size_val, self.rotation)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
        elif self.shape == "gear":
            pth = _gear_path(cx, cy, size_val, self.rotation,
                              n_teeth=8)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)
            # Gear hole
            ax.add_patch(mpatches.Circle((cx, cy), size_val * 0.25,
                                         facecolor="#ffffff",
                                         edgecolor="#2c3e50",
                                         linewidth=1.0, zorder=4))
        else:
            n = {"triangle": 3, "square": 4, "pentagon": 5,
                 "hexagon": 6}[self.shape]
            pth = _polygon_path(cx, cy, n, size_val, self.rotation)
            p = mpatches.PathPatch(pth, facecolor=color_hex,
                                   edgecolor="#2c3e50", linewidth=1.5,
                                   alpha=0.9, zorder=3)
            ax.add_patch(p)

        if self.reflected:
            tri = _polygon_path(cx + size_val * 0.9,
                                cy - size_val * 0.85, 3,
                                cell_size * 0.05, 0)
            p = mpatches.PathPatch(tri, facecolor="#2c3e50",
                                   edgecolor="#2c3e50", linewidth=0.5,
                                   zorder=5)
            ax.add_patch(p)

        if self.n_dots > 0:
            offsets = [(0, 0), (-0.03, -0.03), (0.03, -0.03),
                       (-0.03, 0.03), (0.03, 0.03), (0, -0.05)]
            for i in range(self.n_dots):
                ox, oy = offsets[i % len(offsets)]
                dot = mpatches.Circle((cx + ox * cell_size,
                                       cy + oy * cell_size),
                                      cell_size * 0.035,
                                      facecolor="#1b1b1b", zorder=5)
                ax.add_patch(dot)
        if self.has_ring:
            r = mpatches.Circle((cx, cy), size_val * 1.25,
                                facecolor="none",
                                edgecolor="#1b1b1b",
                                linewidth=1.5, zorder=4)
            ax.add_patch(r)
        if self.has_outline:
            side = size_val * 2.3
            rect = mpatches.Rectangle(
                (cx - side / 2, cy - side / 2), side, side,
                facecolor="none", edgecolor="#1b1b1b",
                linewidth=1.5, linestyle="--", zorder=4)
            ax.add_patch(rect)

# ------------------------------------------------------------------ #
# Rule library — atomic rules and their near-miss variants.
# ------------------------------------------------------------------ #

def _make_rule_rotate(degree):
    def f(s, rng):
        n = s.copy()
        n.rotation = (n.rotation + degree) % 360
        return n
    return f

def _make_rule_dot_add(k):
    def f(s, rng):
        n = s.copy()
        n.n_dots = min(6, s.n_dots + k)
        return n
    return f

def _make_rule_set_color(c):
    def f(s, rng):
        n = s.copy()
        n.color = c
        return n
    return f

def _make_rule_size_shift(delta):
    def f(s, rng):
        n = s.copy()
        idx = _SIZES.index(n.size)
        new_idx = max(0, min(len(_SIZES) - 1, idx + delta))
        n.size = _SIZES[new_idx]
        return n
    return f

def _rule_add_ring(s, rng):
    n = s.copy()
    n.has_ring = True
    return n

def _rule_add_outline(s, rng):
    n = s.copy()
    n.has_outline = True
    return n

def _rule_reflect(s, rng):
    n = s.copy()
    n.reflected = not n.reflected
    return n

_RULES = {
    "rot_90":   ("rotate 90 degrees",   _make_rule_rotate(90),
                 ["rot_180", "rot_270"]),
    "rot_180":  ("rotate 180 degrees",  _make_rule_rotate(180),
                 ["rot_90", "rot_270"]),
    "rot_270":  ("rotate 270 degrees",  _make_rule_rotate(270),
                 ["rot_90", "rot_180"]),
    "dot_1":    ("add 1 dot",           _make_rule_dot_add(1),
                 ["dot_2", "dot_3"]),
    "dot_2":    ("add 2 dots",          _make_rule_dot_add(2),
                 ["dot_1", "dot_3"]),
    "dot_3":    ("add 3 dots",          _make_rule_dot_add(3),
                 ["dot_1", "dot_2"]),
    "col_red":  ("color to red",        _make_rule_set_color("red"),
                 ["col_blue", "col_green"]),
    "col_blue": ("color to blue",       _make_rule_set_color("blue"),
                 ["col_red", "col_green"]),
    "col_green": ("color to green",     _make_rule_set_color("green"),
                 ["col_red", "col_blue"]),
    "col_purple": ("color to purple",   _make_rule_set_color("purple"),
                 ["col_red", "col_blue"]),
    "col_teal": ("color to teal",       _make_rule_set_color("teal"),
                 ["col_blue", "col_green"]),
    "col_coral": ("color to coral",     _make_rule_set_color("coral"),
                 ["col_red", "col_orange"]),
    "col_orange": ("color to orange",   _make_rule_set_color("orange"),
                 ["col_red", "col_coral"]),
    "size_up":  ("size up 1 step",      _make_rule_size_shift(1),
                 ["size_down"]),
    "size_down": ("size down 1 step",   _make_rule_size_shift(-1),
                 ["size_up"]),
    "ring":     ("add ring",            _rule_add_ring,
                 ["outline"]),
    "outline":  ("add dashed outline",  _rule_add_outline,
                 ["ring"]),
    "reflect":  ("add reflection mark", _rule_reflect,
                 []),
}

# Shapes that are visually identical under given rotation angle (symmetry).
# Used to filter out examples where rule is invisible on the query/options.
_ROT_SYMMETRIC_SHAPES = {
    # 2-fold symmetric (identical at 180): circle, square, diamond, cross,
    # plus_mark, hexagon, star (5pt has no 2-fold but render variants), gear.
    180: {"circle", "square", "diamond", "cross", "plus_mark", "hexagon",
          "gear"},
    # 4-fold (identical at 90/270): circle, square, diamond, cross, plus_mark,
    # gear (8 teeth = 4-fold).
    90: {"circle", "square", "diamond", "cross", "plus_mark", "gear"},
    270: {"circle", "square", "diamond", "cross", "plus_mark", "gear"},
}

def _rule_is_visually_applicable(rule_ids, shape_obj) -> bool:
    """Filter: does applying rule_ids produce a visually different shape?"""
    for rid in rule_ids:
        if rid in ("rot_90", "rot_180", "rot_270"):
            angle = {"rot_90": 90, "rot_180": 180, "rot_270": 270}[rid]
            if shape_obj.shape in _ROT_SYMMETRIC_SHAPES.get(angle, set()):
                return False
        elif rid.startswith("col_"):
            target = rid.replace("col_", "")
            if shape_obj.color == target:
                return False
        elif rid == "size_up" and shape_obj.size == "large":
            return False
        elif rid == "size_down" and shape_obj.size == "small":
            return False
    return True

_RULE_FAMILIES = {
    "rot":  ["rot_90", "rot_180", "rot_270"],
    "dot":  ["dot_1", "dot_2", "dot_3"],
    "col":  ["col_red", "col_blue", "col_green", "col_purple",
             "col_teal", "col_coral", "col_orange"],
    "size": ["size_up", "size_down"],
    "mark": ["ring", "outline", "reflect"],
}
_FAMILY_NAMES = list(_RULE_FAMILIES.keys())

def _apply_rule_chain(shape: _Shape, rule_ids: List[str],
                      rng: random.Random) -> _Shape:
    out = shape
    for rid in rule_ids:
        out = _RULES[rid][1](out, rng)
    return out

# ------------------------------------------------------------------ #
# Env
# ------------------------------------------------------------------ #

class RuleInductionSequenceQA(StandaloneVisualEnv):
    ENV_NAME = "rule_induction_sequence"

    _QUESTION_TEMPLATES = [
        "The image shows example transformations (each example: input -> output) followed by a query row with a missing output. Infer the hidden rule, then choose the option (A, B, C, ...) that correctly applies the rule to the query input. Answer with a single letter.",
        "Study the input-output pairs in the image. A consistent rule transforms each input into its output. Apply the same rule to the query shape and pick the correct result. Answer with a single letter.",
        "Each example shows a shape transformation. What rule connects input to output? Apply it to the query shape and select the matching option. Answer with a single letter.",
        "Observe the transformation pattern shown in the example cells. Which option is the correct output for the query input? Answer with a single letter.",
        "Examine the examples and infer the hidden transformation. Apply it once to the question-marked image and choose the matching option. Answer with a single letter.",
        "Each pair in the image illustrates the same transformation. Work out the rule and apply it to the query input. Which lettered option is the answer? Answer with a single letter.",
    ]

    _TITLE_VARIANTS = [
        "Rule induction",
        "Pattern matching",
        "Transformation puzzle",
        "Shape rule",
        "Input -> Output",
        "Figure the rule",
        "Learn the transform",
        "Find the pattern",
    ]

    _ARROW_STYLES = ["->", "->", "-|>", "fancy"]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: 1 rule, 4 examples, 3 options — very simple.
        # L3: 1 rule, 4 examples, 4 options, tight distractors.
        # L6: 2 rules, 3 examples, 4 options, tight distractors.
        # L9: 3 rules, 3 examples, 5 options, tight distractors + irregular
        #     layout.
        n_rules = 1 + level // 4
        n_examples = max(3, 5 - (level // 3))
        n_options = 3 if level <= 1 else (5 if level >= 8 else 4)
        tight_distractors = level >= 3
        # Layout variant — larger variety at higher levels.
        layout_pool = ["vertical"]
        if level >= 2:
            layout_pool.append("horizontal")
        if level >= 4:
            layout_pool.append("grid_2x2")
        if level >= 6:
            layout_pool.append("circular")
        return {
            "n_rules": n_rules,
            "n_examples": n_examples,
            "n_options": n_options,
            "tight_distractors": tight_distractors,
            "layout_pool": layout_pool,
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 31 + level * 23 + 43)
        self._primary_complexity_feature = cfg["n_rules"] * 4 + (
            6 - cfg["n_examples"])
        for _ in range(40):
            result = self._try_generate(rng, level, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, level, cfg):
        n_rules = cfg["n_rules"]
        families = rng.sample(_FAMILY_NAMES, n_rules)
        rule_ids = [rng.choice(_RULE_FAMILIES[f]) for f in families]

        def apply_rule(s):
            return _apply_rule_chain(s, rule_ids, rng)

        n_examples = cfg["n_examples"]
        examples: List[Tuple[_Shape, _Shape]] = []
        used = set()
        tries = 0
        max_tries = 80 * n_examples
        while len(examples) < n_examples and tries < max_tries:
            tries += 1
            s = self._random_shape(rng)
            if s in used:
                continue
            # BUGFIX: skip shapes where rule would be visually invisible
            # (e.g., rotating a square by 180, or coloring green shape green).
            if not _rule_is_visually_applicable(rule_ids, s):
                continue
            t = apply_rule(s)
            if t == s:
                continue
            used.add(s)
            examples.append((s, t))
        if len(examples) < n_examples:
            return None

        query = None
        tries = 0
        while query is None and tries < 80:
            tries += 1
            cand = self._random_shape(rng)
            if cand in used:
                continue
            # BUGFIX: query must also show visible rule effect.
            if not _rule_is_visually_applicable(rule_ids, cand):
                continue
            if apply_rule(cand) == cand:
                continue
            query = cand
        if query is None:
            return None

        correct = apply_rule(query)
        n_distractors = cfg["n_options"] - 1
        distractors = self._make_distractors(
            rng, query, rule_ids, correct, cfg["tight_distractors"],
            n_needed=n_distractors + 3)
        if len(distractors) < n_distractors:
            return None

        options = list(distractors[:n_distractors])
        correct_idx = rng.randint(0, len(options))
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)

        title = rng.choice(self._TITLE_VARIANTS)
        layout = rng.choice(cfg["layout_pool"])
        image = self._render(examples, query, options, title=title,
                             layout=layout, rng=rng)
        question = rng.choice(self._QUESTION_TEMPLATES)
        return question, answer_letter, image

    def _random_shape(self, rng) -> _Shape:
        return _Shape(
            shape=rng.choice(_SHAPES),
            color=rng.choice([c for c in _COLOR_LIST if c not in (
                "yellow",)]),
            size=rng.choice(_SIZES),
            rotation=rng.choice([0, 45, 90]),
            n_dots=0,
            has_ring=False,
            has_outline=False,
            reflected=False,
        )

    def _make_distractors(self, rng, src: _Shape, rule_ids: List[str],
                          correct: _Shape, tight: bool,
                          n_needed: int = 6) -> List[_Shape]:
        distractors = []
        tries = 0
        near_miss_compositions = []
        for swap_idx in range(len(rule_ids)):
            base = rule_ids[swap_idx]
            variants = _RULES[base][2]
            for v in variants:
                comp = list(rule_ids)
                comp[swap_idx] = v
                near_miss_compositions.append(comp)
        rng.shuffle(near_miss_compositions)

        for comp in near_miss_compositions:
            if len(distractors) >= n_needed:
                break
            d = _apply_rule_chain(src, comp, rng)
            if d == correct or d == src:
                continue
            if d in distractors:
                continue
            distractors.append(d)

        while len(distractors) < n_needed and tries < 300:
            tries += 1
            n = len(rule_ids)
            fams = rng.sample(_FAMILY_NAMES, n)
            comp = [rng.choice(_RULE_FAMILIES[f]) for f in fams]
            if comp == rule_ids:
                continue
            d = _apply_rule_chain(src, comp, rng)
            if d == correct or d == src:
                continue
            if d in distractors:
                continue
            distractors.append(d)
        return distractors

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _render(self, examples: List[Tuple[_Shape, _Shape]],
                query: _Shape, options: List[_Shape],
                title: str = "Rule induction",
                layout: str = "vertical",
                rng: Optional[random.Random] = None) -> Image.Image:
        style = self._random_style()
        if rng is None:
            rng = random.Random(0)
        sc = style["figsize_scale"]
        n_examples = len(examples)
        n_opts = len(options)

        cell_face = rng.choice(_FRAME_FACES)
        query_face = rng.choice(_QUERY_FACES)
        query_edge = rng.choice(_QUERY_EDGES)
        arrow_color = rng.choice(_ARROW_COLORS)
        arrow_style = rng.choice(self._ARROW_STYLES)
        label_color = rng.choice(["#2c3e50", "#34495e", "#1b4f72"])

        if layout == "vertical":
            return self._render_vertical(examples, query, options, title,
                                         style, n_examples, n_opts, sc,
                                         cell_face, query_face, query_edge,
                                         arrow_color, arrow_style,
                                         label_color, rng)
        elif layout == "horizontal":
            return self._render_horizontal(examples, query, options, title,
                                           style, n_examples, n_opts, sc,
                                           cell_face, query_face,
                                           query_edge, arrow_color,
                                           arrow_style, label_color, rng)
        elif layout == "grid_2x2":
            return self._render_grid(examples, query, options, title,
                                     style, n_examples, n_opts, sc,
                                     cell_face, query_face, query_edge,
                                     arrow_color, arrow_style,
                                     label_color, rng)
        else:  # circular
            return self._render_circular(examples, query, options, title,
                                         style, n_examples, n_opts, sc,
                                         cell_face, query_face,
                                         query_edge, arrow_color,
                                         arrow_style, label_color, rng)

    def _render_vertical(self, examples, query, options, title, style,
                         n_examples, n_opts, sc, cell_face, query_face,
                         query_edge, arrow_color, arrow_style,
                         label_color, rng):
        n_rows = n_examples + 1
        fig_w = 8.0 * sc
        fig_h = (4.0 + 1.5 * n_rows) * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])

        gs = fig.add_gridspec(2, 1,
                              height_ratios=[1.8 * n_rows, 2.5],
                              hspace=0.15)
        ax_ex = fig.add_subplot(gs[0])
        ax_opt = fig.add_subplot(gs[1])

        total_y = 2.0 + 1.55 * n_rows
        ax_ex.set_xlim(0, 8)
        ax_ex.set_ylim(0, total_y)
        ax_ex.set_aspect("equal")
        ax_ex.axis("off")
        ax_ex.set_title(title, fontsize=13, fontweight="bold", pad=6,
                        color=label_color)

        cell_size = 1.3
        left_x = 1.6
        right_x = 5.6
        header_y = total_y - 0.6
        ax_ex.text(left_x, header_y, "Input", fontsize=11, fontweight="bold",
                   ha="center", color=label_color)
        ax_ex.text(right_x, header_y, "Output", fontsize=11, fontweight="bold",
                   ha="center", color=label_color)

        row_height = 1.55
        top_y = header_y - 0.85
        for i, (s, t) in enumerate(examples):
            cy = top_y - i * row_height
            self._draw_cell(ax_ex, left_x, cy, cell_size, s, cell_face)
            self._draw_arrow(ax_ex, left_x + cell_size / 2 + 0.15, cy,
                             right_x - cell_size / 2 - 0.15, cy,
                             color=arrow_color, style=arrow_style)
            self._draw_cell(ax_ex, right_x, cy, cell_size, t, cell_face)
            ax_ex.text(0.5, cy, f"Ex{i + 1}", fontsize=10, ha="center",
                       va="center", color=label_color)

        cy_q = top_y - n_examples * row_height
        self._draw_cell(ax_ex, left_x, cy_q, cell_size, query, query_face)
        self._draw_arrow(ax_ex, left_x + cell_size / 2 + 0.15, cy_q,
                         right_x - cell_size / 2 - 0.15, cy_q,
                         color=arrow_color, style=arrow_style)
        rect_q = mpatches.FancyBboxPatch(
            (right_x - cell_size / 2, cy_q - cell_size / 2),
            cell_size, cell_size, boxstyle="round,pad=0.04",
            facecolor=query_face, edgecolor=query_edge,
            linewidth=2.5, linestyle="--", zorder=2)
        ax_ex.add_patch(rect_q)
        ax_ex.text(right_x, cy_q, "?", fontsize=28, fontweight="bold",
                   ha="center", va="center", color=query_edge, zorder=5)
        ax_ex.text(0.5, cy_q, "Query", fontsize=10, fontweight="bold",
                   ha="center", va="center", color=query_edge)

        self._render_options(ax_opt, options, cell_face, label_color,
                             n_opts)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                            hspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_horizontal(self, examples, query, options, title, style,
                           n_examples, n_opts, sc, cell_face, query_face,
                           query_edge, arrow_color, arrow_style,
                           label_color, rng):
        # Examples arranged horizontally with arrows between
        fig_w = (1.0 + 1.4 * n_examples + 1.5 + 1.4 * 2) * sc
        fig_w = max(10.0, fig_w)
        fig_h = 6.5 * sc
        fig = plt.figure(figsize=(fig_w * 1.0, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 2.0],
                              hspace=0.2)
        ax_ex = fig.add_subplot(gs[0])
        ax_q = fig.add_subplot(gs[1])
        ax_opt = fig.add_subplot(gs[2])

        ax_ex.set_aspect("equal")
        ax_ex.axis("off")
        ax_ex.set_title(title, fontsize=13, fontweight="bold", pad=6,
                        loc="left", color=label_color)

        # Each example: one input-arrow-output "triple" arranged in a row.
        cell_size = 1.2
        gap = 0.6
        triple_w = cell_size * 2 + gap
        total_w = n_examples * triple_w + (n_examples - 1) * 0.6
        ax_ex.set_xlim(0, total_w + 1)
        ax_ex.set_ylim(0, 2.2)

        for i, (s, t) in enumerate(examples):
            x0 = 0.6 + i * (triple_w + 0.6)
            cy = 1.1
            self._draw_cell(ax_ex, x0 + cell_size / 2, cy, cell_size, s,
                            cell_face)
            self._draw_arrow(ax_ex, x0 + cell_size, cy,
                             x0 + cell_size + gap, cy,
                             color=arrow_color, style=arrow_style)
            self._draw_cell(ax_ex, x0 + cell_size * 1.5 + gap, cy,
                            cell_size, t, cell_face)
            ax_ex.text(x0 + cell_size, cy - cell_size / 2 - 0.15,
                       f"Example {i + 1}", fontsize=9, ha="center",
                       va="top", color=label_color)

        ax_q.set_aspect("equal")
        ax_q.axis("off")
        ax_q.set_title("Apply the same rule to the query:", fontsize=12,
                       fontweight="bold", pad=4, loc="left",
                       color=query_edge)
        ax_q.set_xlim(0, 6)
        ax_q.set_ylim(0, 2.2)
        self._draw_cell(ax_q, 1.5, 1.1, cell_size, query, query_face)
        self._draw_arrow(ax_q, 1.5 + cell_size / 2, 1.1,
                         3.5 - cell_size / 2, 1.1,
                         color=arrow_color, style=arrow_style)
        rect_q = mpatches.FancyBboxPatch(
            (3.5 - cell_size / 2, 1.1 - cell_size / 2),
            cell_size, cell_size, boxstyle="round,pad=0.04",
            facecolor=query_face, edgecolor=query_edge,
            linewidth=2.5, linestyle="--", zorder=2)
        ax_q.add_patch(rect_q)
        ax_q.text(3.5, 1.1, "?", fontsize=28, fontweight="bold",
                  ha="center", va="center", color=query_edge, zorder=5)

        self._render_options(ax_opt, options, cell_face, label_color,
                             n_opts)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94,
                            bottom=0.04, hspace=0.2)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_grid(self, examples, query, options, title, style,
                     n_examples, n_opts, sc, cell_face, query_face,
                     query_edge, arrow_color, arrow_style,
                     label_color, rng):
        # 2x2 grid of example triples
        n_cols = 2
        n_rows = max(1, (n_examples + n_cols - 1) // n_cols)
        fig_w = 10.0 * sc
        fig_h = (3.0 + 2.0 * n_rows + 2.5) * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(3, 1,
                              height_ratios=[2.0 * n_rows + 0.5, 2.0, 2.0],
                              hspace=0.18)
        ax_ex = fig.add_subplot(gs[0])
        ax_q = fig.add_subplot(gs[1])
        ax_opt = fig.add_subplot(gs[2])

        ax_ex.set_aspect("equal")
        ax_ex.axis("off")
        ax_ex.set_title(title, fontsize=13, fontweight="bold", pad=6,
                        loc="left", color=label_color)

        cell_size = 1.2
        gap = 0.5
        triple_w = cell_size * 2 + gap
        panel_w = 10
        panel_h = 1.8 * n_rows + 0.5
        ax_ex.set_xlim(0, panel_w)
        ax_ex.set_ylim(0, panel_h)
        for i, (s, t) in enumerate(examples):
            r = i // n_cols
            c = i % n_cols
            x0 = 0.8 + c * (triple_w + 1.5)
            cy = panel_h - 1.0 - r * 1.85
            self._draw_cell(ax_ex, x0 + cell_size / 2, cy, cell_size, s,
                            cell_face)
            self._draw_arrow(ax_ex, x0 + cell_size, cy,
                             x0 + cell_size + gap, cy,
                             color=arrow_color, style=arrow_style)
            self._draw_cell(ax_ex, x0 + cell_size * 1.5 + gap, cy,
                            cell_size, t, cell_face)
            ax_ex.text(x0 + cell_size, cy - cell_size / 2 - 0.15,
                       f"Ex{i + 1}", fontsize=9, ha="center", va="top",
                       color=label_color)

        ax_q.set_aspect("equal")
        ax_q.axis("off")
        ax_q.set_title("Apply the rule to the query:", fontsize=12,
                       fontweight="bold", pad=4, loc="left",
                       color=query_edge)
        ax_q.set_xlim(0, 6)
        ax_q.set_ylim(0, 2.2)
        self._draw_cell(ax_q, 1.5, 1.1, cell_size, query, query_face)
        self._draw_arrow(ax_q, 1.5 + cell_size / 2, 1.1,
                         3.5 - cell_size / 2, 1.1,
                         color=arrow_color, style=arrow_style)
        rect_q = mpatches.FancyBboxPatch(
            (3.5 - cell_size / 2, 1.1 - cell_size / 2),
            cell_size, cell_size, boxstyle="round,pad=0.04",
            facecolor=query_face, edgecolor=query_edge,
            linewidth=2.5, linestyle="--", zorder=2)
        ax_q.add_patch(rect_q)
        ax_q.text(3.5, 1.1, "?", fontsize=28, fontweight="bold",
                  ha="center", va="center", color=query_edge, zorder=5)

        self._render_options(ax_opt, options, cell_face, label_color,
                             n_opts)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94,
                            bottom=0.04, hspace=0.18)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_circular(self, examples, query, options, title, style,
                         n_examples, n_opts, sc, cell_face, query_face,
                         query_edge, arrow_color, arrow_style,
                         label_color, rng):
        # Circular arrangement: examples arranged around a circle, query
        # in the centre.
        fig_w = 10.0 * sc
        fig_h = 10.0 * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.5], hspace=0.1)
        ax_ex = fig.add_subplot(gs[0])
        ax_opt = fig.add_subplot(gs[1])

        ax_ex.set_aspect("equal")
        ax_ex.axis("off")
        ax_ex.set_title(title, fontsize=13, fontweight="bold", pad=6,
                        loc="center", color=label_color)
        ax_ex.set_xlim(-4.5, 4.5)
        ax_ex.set_ylim(-4.5, 4.5)

        cell_size = 1.0
        R_in = 1.5
        R_out = 3.2
        for i, (s, t) in enumerate(examples):
            angle = 2 * math.pi * i / n_examples + math.pi / 2
            ix = R_in * math.cos(angle)
            iy = R_in * math.sin(angle)
            ox = R_out * math.cos(angle)
            oy = R_out * math.sin(angle)
            self._draw_cell(ax_ex, ix, iy, cell_size, s, cell_face)
            self._draw_arrow(ax_ex, ix, iy, ox, oy,
                             color=arrow_color, style=arrow_style)
            self._draw_cell(ax_ex, ox, oy, cell_size, t, cell_face)
            ax_ex.text((ix + ox) / 2 + 0.25, (iy + oy) / 2 + 0.25,
                       f"Ex{i + 1}", fontsize=9, ha="center",
                       color=label_color)

        # Query in the middle (below centre)
        q_x, q_y = 0.0, -4.0 + 0.8
        self._draw_cell(ax_ex, q_x - 0.8, q_y, cell_size, query, query_face)
        self._draw_arrow(ax_ex, q_x - 0.8 + cell_size / 2, q_y,
                         q_x + 0.8 - cell_size / 2, q_y,
                         color=arrow_color, style=arrow_style)
        rect_q = mpatches.FancyBboxPatch(
            (q_x + 0.8 - cell_size / 2, q_y - cell_size / 2),
            cell_size, cell_size, boxstyle="round,pad=0.04",
            facecolor=query_face, edgecolor=query_edge,
            linewidth=2.5, linestyle="--", zorder=5)
        ax_ex.add_patch(rect_q)
        ax_ex.text(q_x + 0.8, q_y, "?", fontsize=22, fontweight="bold",
                   ha="center", va="center", color=query_edge, zorder=6)

        self._render_options(ax_opt, options, cell_face, label_color,
                             n_opts)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94,
                            bottom=0.04, hspace=0.1)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_options(self, ax_opt, options, cell_face, label_color,
                        n_opts):
        ax_opt.set_xlim(0, n_opts * 2.0)
        ax_opt.set_ylim(0, 2.2)
        ax_opt.set_aspect("equal")
        ax_opt.axis("off")
        ax_opt.set_title("Options for the '?'", fontsize=12, pad=3,
                         color=label_color)
        opt_cell = 1.5
        for i, opt in enumerate(options):
            cx = i * 2.0 + 1.0
            cy = 1.1
            rect = mpatches.FancyBboxPatch(
                (cx - opt_cell / 2, cy - opt_cell / 2),
                opt_cell, opt_cell,
                boxstyle="round,pad=0.04",
                facecolor=cell_face, edgecolor="#2c3e50",
                linewidth=1.5, zorder=1)
            ax_opt.add_patch(rect)
            opt.draw(ax_opt, cx, cy, opt_cell)
            label = chr(ord("A") + i)
            ax_opt.text(cx, cy - opt_cell / 2 - 0.12, label,
                        fontsize=13, fontweight="bold", ha="center",
                        va="top", color=label_color)

    @staticmethod
    def _draw_cell(ax, cx, cy, cell_size, shape: _Shape, face):
        rect = mpatches.FancyBboxPatch(
            (cx - cell_size / 2, cy - cell_size / 2),
            cell_size, cell_size,
            boxstyle="round,pad=0.03",
            facecolor=face, edgecolor="#7f8c8d",
            linewidth=1.2, zorder=1)
        ax.add_patch(rect)
        shape.draw(ax, cx, cy, cell_size)

    @staticmethod
    def _draw_arrow(ax, x0, y, x1, y2, color="#e67e22", style="->"):
        ax.annotate(
            "", xy=(x1, y2), xytext=(x0, y),
            arrowprops=dict(arrowstyle=style, lw=2.0, color=color),
            zorder=2)

if __name__ == "__main__":
    import os
    import collections
    out_dir = "/tmp/env_check"
    os.makedirs(out_dir, exist_ok=True)
    env = RuleInductionSequenceQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED to generate")
                continue
            print(f"[seed={seed} L{level}] ok, A={env._answer}")
    for level in (0, 3, 6, 9):
        letters = collections.Counter()
        for s in range(20):
            e = RuleInductionSequenceQA()
            ok = e.generate(seed=s * 1000 + level * 37 + 17,
                            parameter={"level": level})
            if ok:
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
