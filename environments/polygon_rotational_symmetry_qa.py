"""
Polygon Rotational Symmetry QA.

Single-image and figure-MCQ variants for "what is the order of rotational
symmetry of this shape" questions in textbook-style 4-option / 5-option MCQ
format with "No correct answer" trap.

Difficulty axes:
  L0/L1 -- single regular polygon (square, equilateral triangle, regular
           hexagon, regular pentagon) shown on grid. 4-option/5-option MCQ
           "the order of rotational symmetry of the shown shape is ( )".
  L2/L4 -- composite shape (regular polygon + decoration that preserves or
           breaks symmetry). 5-option MCQ "Shape A has rotational symmetry
           of order ( )".
  L5/L7 -- figure-MCQ: 4 candidate shapes labeled A/B/C/D, ask
           "Which of the following shapes has rotational symmetry of order N?"
           with one near-trap that is reflection-symmetric only (kite /
           isoceles trapezoid / arrow) but rotational order 1.
  L8/L9 -- tight numeric distractors {2,3,4,6} clustering, 40% of problems
           have ground truth "No correct answer" (target order is 5/7/8 or
           the shape is rotationally asymmetric).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ====================================================================== #
# Shape factories — each returns (vertex_list, rotational_order)
# Rotational order = number of distinct rotations in [0, 360) deg that
# map the shape onto itself (including identity).
# ====================================================================== #

def _regular_polygon(n: int, radius: float = 3.0,
                     rotation_rad: float = 0.0) -> List[Tuple[float, float]]:
    return [(radius * math.cos(math.pi / 2 + 2 * math.pi * i / n + rotation_rad),
             radius * math.sin(math.pi / 2 + 2 * math.pi * i / n + rotation_rad))
            for i in range(n)]


def _star(n_points: int, outer: float = 3.0, inner: float = 1.3,
          rotation_rad: float = 0.0) -> List[Tuple[float, float]]:
    pts = []
    for i in range(2 * n_points):
        r = outer if i % 2 == 0 else inner
        a = math.pi / 2 + math.pi * i / n_points + rotation_rad
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _pinwheel(n_blades: int, outer: float = 3.0, inner: float = 1.0,
              skew: float = 0.6) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n_blades):
        a0 = 2 * math.pi * i / n_blades
        a1 = a0 + skew
        pts.append((inner * math.cos(a0), inner * math.sin(a0)))
        pts.append((outer * math.cos(a1), outer * math.sin(a1)))
    return pts


def _gear(n_teeth: int, outer: float = 3.0,
          inner: float = 2.4) -> List[Tuple[float, float]]:
    pts = []
    for i in range(2 * n_teeth):
        r = outer if i % 2 == 0 else inner
        a = 2 * math.pi * i / (2 * n_teeth)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def _rectangle(w: float = 4, h: float = 2) -> List[Tuple[float, float]]:
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def _square(s: float = 3) -> List[Tuple[float, float]]:
    return [(-s / 2, -s / 2), (s / 2, -s / 2), (s / 2, s / 2), (-s / 2, s / 2)]


def _rhombus(a: float = 3, b: float = 1.5) -> List[Tuple[float, float]]:
    return [(0, b), (a, 0), (0, -b), (-a, 0)]


def _scalene_triangle() -> List[Tuple[float, float]]:
    return [(-2, -1), (2.5, -0.5), (-0.3, 2.4)]


def _isoceles_triangle() -> List[Tuple[float, float]]:
    return [(-2, -1), (2, -1), (0, 2.5)]


def _kite(w: float = 2.5, h_top: float = 2.5,
          h_bot: float = 1.5) -> List[Tuple[float, float]]:
    return [(0, h_top), (w, 0), (0, -h_bot), (-w, 0)]


def _arrow() -> List[Tuple[float, float]]:
    return [(-2, -0.6), (1, -0.6), (1, -1.5), (2.7, 0),
            (1, 1.5), (1, 0.6), (-2, 0.6)]


def _isoceles_trapezoid() -> List[Tuple[float, float]]:
    return [(-2.4, -1.2), (2.4, -1.2), (1.4, 1.2), (-1.4, 1.2)]


def _z_shape() -> List[Tuple[float, float]]:
    return [(-2, 1.8), (2, 1.8), (2, 0.6), (-0.6, -0.6),
            (2, -0.6), (2, -1.8), (-2, -1.8), (-2, -0.6),
            (0.6, 0.6), (-2, 0.6)]


def _t_shape() -> List[Tuple[float, float]]:
    return [(-2.4, 1.8), (2.4, 1.8), (2.4, 0.6), (0.8, 0.6),
            (0.8, -1.8), (-0.8, -1.8), (-0.8, 0.6), (-2.4, 0.6)]


def _h_shape() -> List[Tuple[float, float]]:
    return [(-1.8, -1.8), (-0.7, -1.8), (-0.7, -0.3),
            (0.7, -0.3), (0.7, -1.8), (1.8, -1.8), (1.8, 1.8),
            (0.7, 1.8), (0.7, 0.3), (-0.7, 0.3), (-0.7, 1.8),
            (-1.8, 1.8)]


def _l_shape() -> List[Tuple[float, float]]:
    return [(-1.6, -1.6), (1.6, -1.6), (1.6, -0.4),
            (-0.3, -0.4), (-0.3, 1.6), (-1.6, 1.6)]


def _plus_cross(arm: float = 1, length: float = 3) -> List[Tuple[float, float]]:
    return [
        (-arm, length), (arm, length), (arm, arm),
        (length, arm), (length, -arm), (arm, -arm),
        (arm, -length), (-arm, -length), (-arm, -arm),
        (-length, -arm), (-length, arm), (-arm, arm),
    ]


# ====================================================================== #
# Centralized shape catalog: name -> (factory_callable, order)
# All shapes are centered roughly at origin and sized so they fit nicely
# in a [-3.5, 3.5] viewport.
# ====================================================================== #

def _build_catalog(rng: random.Random) -> Dict[str, Tuple[List[Tuple[float, float]], int]]:
    """Return a shape catalog keyed by identifier. Each entry is
    (vertex_list, rotational_order)."""
    s = 2.6 + rng.random() * 0.8
    cat: Dict[str, Tuple[List[Tuple[float, float]], int]] = {}
    # Regular polygons: order = n
    cat["square"] = (_square(s), 4)
    cat["eq_triangle"] = (_regular_polygon(3, radius=s), 3)
    cat["pentagon"] = (_regular_polygon(5, radius=s), 5)
    cat["hexagon"] = (_regular_polygon(6, radius=s), 6)
    cat["heptagon"] = (_regular_polygon(7, radius=s), 7)
    cat["octagon"] = (_regular_polygon(8, radius=s), 8)
    # Rect / rhombus: order 2
    cat["rectangle"] = (_rectangle(s * 1.3, s * 0.7), 2)
    cat["rhombus"] = (_rhombus(s * 1.0, s * 0.55), 2)
    # Plus cross: order 4
    cat["plus_cross"] = (_plus_cross(arm=s * 0.32, length=s * 0.95), 4)
    # Stars: order n_points
    cat["star5"] = (_star(5, outer=s, inner=s * 0.42), 5)
    cat["star6"] = (_star(6, outer=s, inner=s * 0.42), 6)
    cat["star8"] = (_star(8, outer=s, inner=s * 0.45), 8)
    # Pinwheels: order n_blades
    cat["pinwheel3"] = (_pinwheel(3, outer=s, inner=s * 0.35,
                                   skew=0.6), 3)
    cat["pinwheel4"] = (_pinwheel(4, outer=s, inner=s * 0.35,
                                   skew=0.6), 4)
    cat["pinwheel6"] = (_pinwheel(6, outer=s, inner=s * 0.32,
                                   skew=0.55), 6)
    # Gears: order n_teeth
    cat["gear4"] = (_gear(4, outer=s, inner=s * 0.78), 4)
    cat["gear6"] = (_gear(6, outer=s, inner=s * 0.78), 6)
    # Rotation-asymmetric shapes (order 1)
    cat["isoceles_triangle"] = (_isoceles_triangle(), 1)
    cat["scalene_triangle"] = (_scalene_triangle(), 1)
    cat["kite"] = (_kite(s * 0.85, s * 0.95, s * 0.55), 1)
    cat["arrow"] = (_arrow(), 1)
    cat["isoceles_trapezoid"] = (_isoceles_trapezoid(), 1)
    cat["t_shape"] = (_t_shape(), 1)
    cat["l_shape"] = (_l_shape(), 1)
    # Shapes with order 2 (line + half-turn symmetric only)
    cat["z_shape"] = (_z_shape(), 2)
    cat["h_shape"] = (_h_shape(), 2)
    return cat


# ====================================================================== #
# Question-style helpers (textbook-style phrasing, no benchmark name)
# ====================================================================== #

# Stem variants for "Shape A has rotational symmetry of order ( )"
_Q4_STEMS = [
    "As shown in the diagram, the shape has rotational symmetry of order (    ).",
    "As shown in the diagram, shape A has rotational symmetry of order (    ).",
    "As shown in the figure, the order of rotational symmetry of the figure is (    ).",
    "Look at the figure. The order of rotational symmetry of the shape is (    ).",
    "As shown in the diagram, the figure has rotational symmetry of order (    ).",
    "The figure shown has rotational symmetry of order (    ).",
]

# L0/L1 simpler stems for first-introduced learners
_Q4_STEMS_L0 = [
    "As shown in the diagram, the shape has rotational symmetry of order (    ).",
    "Look at the figure. Its order of rotational symmetry is (    ).",
    "The figure has rotational symmetry of order (    ).",
    "What is the order of rotational symmetry of the shape shown in the figure? It is (    ).",
]

# Figure-MCQ stems for "which has rotational symmetry of order N"
_Q9_STEMS = [
    "Which of the following shapes has rotational symmetry of order {n}?",
    "Among the four shapes shown, which has rotational symmetry of order {n}?",
    "As shown in the diagram, which figure has rotational symmetry of order {n}?",
    "Which of the four candidate shapes has rotational symmetry of order {n}?",
]


# ====================================================================== #
# Main env class
# ====================================================================== #

class PolygonRotationalSymmetryQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive
    ENV_NAME = "polygon_rotational_symmetry"

    QUESTION_TYPES = [
        "single_shape_l01",       # L0/L1 -- regular polygon, simple options
        "single_shape_l24",       # L2-L4 -- composite, 5-option MCQ
        "figure_mcq_l57",         # L5-L7 -- 2x2 panel figure MCQ
        "single_shape_l89",       # L8/L9 -- tight distractors + 40% trap
    ]

    # ------------------------------------------------------------------ #
    # Level config
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "qtype": "single_shape_l01",
                # Easiest: square (order 4); pentagon, hexagon, eq triangle
                "shape_pool": ["square", "eq_triangle", "hexagon", "pentagon"],
                "n_options": 4 if level == 0 else 5,
                "trap_rate": 0.0,
            }
        if level <= 4:
            return {
                "qtype": "single_shape_l24",
                # composite-pool: regular polygons + plus cross + stars +
                # gears (still "clean" symmetric shapes)
                "shape_pool": [
                    "square", "eq_triangle", "pentagon", "hexagon",
                    "rectangle", "rhombus", "plus_cross", "star5",
                    "star6", "gear4", "gear6", "pinwheel4", "pinwheel6",
                ],
                "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.20,
            }
        if level <= 7:
            return {
                "qtype": "figure_mcq_l57",
                # Used for the target panel + 3 distractor panels
                "n_options": 4,
                "trap_rate": 0.0,  # No "No correct answer" in figure-MCQ
                # target order from this set
                "target_order_pool": [2, 3, 4, 6],
            }
        # L8/L9 -- tighter distractors + 40% "No correct answer"
        return {
            "qtype": "single_shape_l89",
            # Mix of clean rotational shapes + asymmetric trap shapes
            "shape_pool": [
                "square", "eq_triangle", "pentagon", "hexagon", "heptagon",
                "octagon", "rectangle", "rhombus", "plus_cross", "star5",
                "star6", "star8", "gear4", "gear6", "pinwheel3",
                "pinwheel4", "pinwheel6", "isoceles_triangle",
                "kite", "arrow", "isoceles_trapezoid", "z_shape",
                "h_shape",
            ],
            "n_options": 5,
            "trap_rate": 0.40,
        }

    def level_to_complexity(self, level: int) -> float:
        return float(level)

    # ------------------------------------------------------------------ #
    # Generate dispatcher
    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = cfg["qtype"]
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        for _ in range(20):
            if qtype == "figure_mcq_l57":
                result = self._try_generate_figure_mcq(cfg, sub_rng, level)
            else:
                # All single-shape variants share the same backbone
                result = self._try_generate_single_shape(cfg, sub_rng, level,
                                                         qtype)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # SINGLE-SHAPE backbone (L0/L1, L2-L4, L8/L9)
    # ------------------------------------------------------------------ #
    def _try_generate_single_shape(self, cfg, rng, level, qtype):
        shape_pool = cfg["shape_pool"]
        catalog = _build_catalog(rng)

        n_options = cfg["n_options"]  # 4 or 5
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        # Last letter reserved for "No correct answer"
        last_letter = opt_letters[-1]

        # Decide whether this is a trap problem (GT = last letter)
        use_trap = (rng.random() < cfg["trap_rate"])

        # Pool of canonical numeric distractors, common textbook orders
        canonical_distractors = [2, 3, 4, 5, 6, 7, 8]

        if use_trap:
            # We want the SHAPE's true order to NOT appear in any of the
            # numeric options. Pick a shape with order in {5, 7, 8} (or 1
            # for asymmetric shapes) and offer numeric distractors only
            # from {2, 3, 4, 6} (the canonical textbook order values).
            trap_orders = [1, 5, 7, 8]
            trap_shapes = [k for k in shape_pool
                           if catalog[k][1] in trap_orders]
            if not trap_shapes:
                # Fall back to non-trap if pool can't support a trap.
                use_trap = False
            else:
                shape_key = rng.choice(trap_shapes)
                verts, true_order = catalog[shape_key]
                # Numeric options drawn from {2, 3, 4, 6} excluding true_order
                numeric_pool = [2, 3, 4, 6]
                # Edge case: true_order might happen to be in {2, 3, 4, 6}
                # (it shouldn't given trap_orders, but defensive)
                numeric_pool = [v for v in numeric_pool if v != true_order]
                if len(numeric_pool) < n_options - 1:
                    # Not enough; pad
                    pad_pool = [v for v in canonical_distractors
                                if v != true_order and v not in numeric_pool]
                    rng.shuffle(pad_pool)
                    numeric_pool += pad_pool[: (n_options - 1 - len(numeric_pool))]
                rng.shuffle(numeric_pool)
                numeric_options = numeric_pool[: n_options - 1]
                # Sort options ascending (textbook convention)
                numeric_options_sorted = sorted(numeric_options)
                # Assemble option lines
                option_texts = [str(v) for v in numeric_options_sorted]
                option_texts.append("No correct answer")
                correct_letter = last_letter
                shape_used_pts = verts

        if not use_trap:
            # Non-trap path: pick a shape, true order MUST appear as one
            # of the numeric options (and thus correct_letter is its slot).
            tries = 0
            shape_key = None
            while tries < 30:
                cand = rng.choice(shape_pool)
                cand_order = catalog[cand][1]
                # Constraints differ by qtype:
                if qtype == "single_shape_l01":
                    # L0/L1: only orders {3, 4, 5, 6} appear
                    if cand_order in (3, 4, 5, 6):
                        shape_key = cand
                        break
                elif qtype == "single_shape_l24":
                    # L2-L4: orders {2, 3, 4, 5, 6}
                    if cand_order in (2, 3, 4, 5, 6):
                        shape_key = cand
                        break
                else:  # single_shape_l89
                    # L8/L9 non-trap: orders {2, 3, 4, 6} (canonical)
                    if cand_order in (2, 3, 4, 6):
                        shape_key = cand
                        break
                tries += 1
            if shape_key is None:
                # Unable to pick acceptable shape
                return None
            verts, true_order = catalog[shape_key]
            shape_used_pts = verts

            # Build numeric distractors
            distractor_pool = [v for v in canonical_distractors
                               if v != true_order]
            # At L8/L9 (tight), prefer close orders
            if qtype == "single_shape_l89":
                # Prefer {2, 3, 4, 6} adjacency
                tight = [v for v in distractor_pool if v in (2, 3, 4, 6)]
                other = [v for v in distractor_pool if v not in (2, 3, 4, 6)]
                rng.shuffle(tight)
                rng.shuffle(other)
                distractor_pool = tight + other
            else:
                rng.shuffle(distractor_pool)
            distractors = distractor_pool[: n_options - 2]
            if len(distractors) < n_options - 2:
                return None
            numeric_options = sorted([true_order] + distractors)
            option_texts = [str(v) for v in numeric_options]
            option_texts.append("No correct answer")
            correct_letter = opt_letters[numeric_options.index(true_order)]

        # Apply small visual jitter for shape rendering
        scale = 0.85 + rng.random() * 0.30
        shape_verts = [(x * scale, y * scale) for (x, y) in shape_used_pts]
        # Random rotation -- order is rotation-invariant so this is safe
        rot_angle = rng.uniform(0, 2 * math.pi)
        cos_a, sin_a = math.cos(rot_angle), math.sin(rot_angle)
        shape_verts = [(x * cos_a - y * sin_a, x * sin_a + y * cos_a)
                       for (x, y) in shape_verts]

        # Build question stem
        if qtype == "single_shape_l01":
            stem = rng.choice(_Q4_STEMS_L0)
        else:
            stem = rng.choice(_Q4_STEMS)

        # Format options "A. 2; B. 3; C. 4; D. 6; E. No correct answer"
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {option_texts[i]}"
            for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        # Render shape on coordinate grid
        image = self._draw_single_shape(shape_verts, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # FIGURE-MCQ backbone (L5-L7)
    # ------------------------------------------------------------------ #
    def _try_generate_figure_mcq(self, cfg, rng, level):
        catalog = _build_catalog(rng)
        target_order = rng.choice(cfg["target_order_pool"])

        # Find shapes that match target order
        target_keys = [k for k, (_, o) in catalog.items() if o == target_order]
        if not target_keys:
            return None
        target_key = rng.choice(target_keys)
        target_verts, _ = catalog[target_key]

        # Distractor candidates: shapes with order != target_order. At L7
        # we prefer trap shapes (kite, isoceles_triangle, isoceles_trapezoid,
        # arrow) which look symmetric but are rotational order 1.
        prefer_trap_shapes = ["kite", "isoceles_triangle",
                              "isoceles_trapezoid", "arrow", "t_shape",
                              "l_shape"]
        all_distractor_keys = [k for k, (_, o) in catalog.items()
                               if o != target_order and k != target_key]
        # Don't include shapes that would visually duplicate target choice
        # (e.g. another order-target shape). Filter is already enforced above.
        if level >= 6:
            # Bias toward including at least 1 trap shape
            traps = [k for k in all_distractor_keys
                     if k in prefer_trap_shapes]
            others = [k for k in all_distractor_keys
                      if k not in prefer_trap_shapes]
            rng.shuffle(traps)
            rng.shuffle(others)
            # pick 1 trap + 2 others
            chosen_distractors = []
            if traps:
                chosen_distractors.append(traps[0])
                # pick 2 distinct others (distinct orders)
                seen_orders = {target_order, catalog[traps[0]][1]}
                for k in others:
                    if catalog[k][1] not in seen_orders:
                        chosen_distractors.append(k)
                        seen_orders.add(catalog[k][1])
                    if len(chosen_distractors) >= 3:
                        break
            else:
                # No trap shapes available; pick 3 distractors with distinct orders
                seen_orders = {target_order}
                for k in others:
                    if catalog[k][1] not in seen_orders:
                        chosen_distractors.append(k)
                        seen_orders.add(catalog[k][1])
                    if len(chosen_distractors) >= 3:
                        break
            # If we couldn't find 3 distinct-order distractors, top up
            if len(chosen_distractors) < 3:
                more = [k for k in others if k not in chosen_distractors]
                while len(chosen_distractors) < 3 and more:
                    chosen_distractors.append(more.pop(0))
        else:
            # L5: just pick 3 distinct-order distractors, no trap bias
            rng.shuffle(all_distractor_keys)
            chosen_distractors = []
            seen_orders = {target_order}
            for k in all_distractor_keys:
                if catalog[k][1] not in seen_orders:
                    chosen_distractors.append(k)
                    seen_orders.add(catalog[k][1])
                if len(chosen_distractors) >= 3:
                    break
            if len(chosen_distractors) < 3:
                # Top up without distinct-order constraint
                for k in all_distractor_keys:
                    if k not in chosen_distractors:
                        chosen_distractors.append(k)
                    if len(chosen_distractors) >= 3:
                        break

        if len(chosen_distractors) < 3:
            return None

        # Build the 4 panels: target + 3 distractors
        panel_keys = [target_key] + chosen_distractors
        # Shuffle so target is in random position
        rng.shuffle(panel_keys)
        panel_letters = ["A", "B", "C", "D"]
        correct_letter = panel_letters[panel_keys.index(target_key)]

        # Apply per-panel scale + rotation jitter
        panel_data = []
        for k in panel_keys:
            verts, _ = catalog[k]
            scale = 0.85 + rng.random() * 0.3
            v_jit = [(x * scale, y * scale) for x, y in verts]
            rot = rng.uniform(0, 2 * math.pi)
            ca, sa = math.cos(rot), math.sin(rot)
            v_jit = [(x * ca - y * sa, x * sa + y * ca) for x, y in v_jit]
            panel_data.append((k, v_jit))

        # Build question
        stem_template = rng.choice(_Q9_STEMS)
        stem = stem_template.format(n=target_order)
        # Add explicit "What is the rotational symmetry order" clarification
        question = (
            f"{stem}\n"
            f"Each shape is shown in its own labeled panel (A, B, C, D).\n"
            f"A. Shape in panel A\n"
            f"B. Shape in panel B\n"
            f"C. Shape in panel C\n"
            f"D. Shape in panel D\n"
            f"Answer with a single letter A, B, C, or D."
        )
        image = self._draw_panel_grid(panel_data, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_single_shape(self, verts: List[Tuple[float, float]],
                           rng: random.Random) -> Image.Image:
        """Draw a shape on a coordinate grid (axes + grid lines)."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)
        edge_c = "#222"
        lw = 1.8 + rng.random() * 1.2

        poly = MplPolygon(verts, closed=True, facecolor=palette[0],
                          alpha=0.55, edgecolor=edge_c, linewidth=lw)
        ax.add_patch(poly)

        # Mark vertices for clarity
        if rng.random() < 0.6:
            for (x, y) in verts:
                ax.plot(x, y, "o", color=edge_c, markersize=3,
                        markerfacecolor="white",
                        markeredgecolor=edge_c)

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        margin = 1.0 + rng.random() * 0.6
        xmin, xmax = min(xs) - margin, max(xs) + margin
        ymin, ymax = min(ys) - margin, max(ys) + margin
        # Make grid integer-based for textbook look
        xmin_i = int(math.floor(xmin))
        xmax_i = int(math.ceil(xmax))
        ymin_i = int(math.floor(ymin))
        ymax_i = int(math.ceil(ymax))
        ax.set_xticks(range(xmin_i, xmax_i + 1))
        ax.set_yticks(range(ymin_i, ymax_i + 1))
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.5)
        ax.set_xlim(xmin_i, xmax_i)
        ax.set_ylim(ymin_i, ymax_i)
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=8)
        # Soft axes
        ax.axhline(0, color="#555", linewidth=0.6, alpha=0.7)
        ax.axvline(0, color="#555", linewidth=0.6, alpha=0.7)
        ax.set_title("Figure", fontsize=style["font_size_base"] + 1,
                     fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_panel_grid(self, panel_data: List[Tuple[str, List[Tuple[float, float]]]],
                         rng: random.Random) -> Image.Image:
        """Draw 2x2 panel grid with shapes labeled A/B/C/D."""
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, axes = plt.subplots(2, 2, figsize=(8.5 * sc, 8.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)
        labels = ["A", "B", "C", "D"]
        for idx, (key, verts) in enumerate(panel_data):
            ax = axes[idx // 2][idx % 2]
            ax.set_facecolor(style["bg_color"])
            poly = MplPolygon(verts, closed=True,
                              facecolor=palette[idx % len(palette)],
                              alpha=0.55, edgecolor="#222",
                              linewidth=1.8)
            ax.add_patch(poly)
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            margin = 1.0
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
            ax.set_title(labels[idx], fontsize=15, fontweight="bold")
            # box around panel
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.3)
                spine.set_edgecolor("#444")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
