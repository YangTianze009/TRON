"""
Analogy From Sequence QA environment (redesigned 2026-04-16).

A sequence S1, S2, S3 (and optionally S4, S5) showing a progressive
transformation; then a new image C is shown and the same rule must be
applied once to produce "?".

Target: VisualPuzzles analogical,
reference inductive.

Redesign notes (v3):
  * v2: only 5 rule types, rules opaque at L3+, L0 only "count_grow".
  * v3:
      - 11 rule families (count_grow, count_shrink, stretch_x, stretch_y,
        color_shift, rotation, mirror, size_cycle, shape_cycle,
        grid_evolve, multi_evolve, position_shift).
      - L0 has 3 rule types (count_grow, shape_cycle, size_cycle) for
        visual variety.
      - Distinct layouts (vertical, horizontal, grid 2x2 for 4-step seq).
      - Options box shuffled.
      - 6 question templates.
      - L0 vs L9 structurally different: L0 = 3-step obvious visual,
        3 options; L9 = 5-step multi-attribute evolution, 5 options.
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

# Visually-coherent palette ordered along HSV hue ring. With this order,
# "color_shift step=1" corresponds to a fixed visible hue rotation, so the
# rule is inferrable from any visible transition and generalizes to C.
# 12 colors spanning the full hue wheel: red, orange, yellow, lime, green,
# spring-green, cyan, azure, blue, violet, magenta, rose.
_COLORS = [
    "#e74c3c",  # 0 red
    "#e67e22",  # 1 orange
    "#f1c40f",  # 2 yellow
    "#a3d92f",  # 3 lime
    "#2ecc71",  # 4 green
    "#1abc9c",  # 5 spring-green / teal
    "#00bcd4",  # 6 cyan
    "#3498db",  # 7 azure
    "#2c5fbf",  # 8 blue
    "#8e44ad",  # 9 violet
    "#d63384",  # 10 magenta
    "#e84393",  # 11 rose
]
_BASE_SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon",
                "diamond", "star", "ellipse"]

# Shapes that are rotationally symmetric under certain rotation steps.
# For each shape, list the smallest rotation (in degrees) under which the
# rendered shape is visually indistinguishable from its unrotated form.
# 0 means "invariant under ALL rotations" (e.g. circle).
_ROT_SYMMETRY = {
    "circle": 0,      # any rotation = identity
    "square": 90,
    "diamond": 90,
    "hexagon": 60,
    "ellipse": 180,
    "triangle": 360,  # not symmetric under any sub-360 rotation in our pool
    "pentagon": 360,
    "star": 360,
}

def _rotation_visible(shape: str, rot_step: int) -> bool:
    """Return True iff applying rot_step to this shape produces a visibly
    different rendering (so the rotation rule can be observed)."""
    sym = _ROT_SYMMETRY.get(shape, 360)
    if sym == 0:
        return False  # circle: no rotation is visible
    return (rot_step % sym) != 0

class AnalogyFromSequenceQA(StandaloneVisualEnv):
    ENV_NAME = "analogy_from_sequence"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "sequence_length": 3,
                "rule_pool": ["count_grow", "shape_cycle", "size_cycle",
                              "color_shift"],
                "tight_distractors": False,
                "n_options": 3,
                "layout_pool": ["horizontal"],
            }
        if level <= 3:
            return {
                "sequence_length": 3,
                "rule_pool": ["count_grow", "count_shrink", "shape_cycle",
                              "size_cycle", "color_shift",
                              "stretch_x", "rotation"],
                "tight_distractors": False,
                "n_options": 4,
                "layout_pool": ["horizontal"],
            }
        if level <= 5:
            # "mirror" removed from rule pool: all polygons we draw are
            # y-axis-symmetric as rendered, so mirroring is visually
            # invisible and the rule is unsolvable from the image alone.
            return {
                "sequence_length": 4,
                "rule_pool": ["color_shift", "stretch_x", "stretch_y",
                              "rotation", "size_cycle",
                              "position_shift"],
                "tight_distractors": True,
                "n_options": 4,
                "layout_pool": ["horizontal", "grid_2x2"],
            }
        if level <= 7:
            return {
                "sequence_length": 4,
                "rule_pool": ["grid_evolve", "color_shift",
                              "rotation_plus_color",
                              "stretch_x", "stretch_y",
                              "position_shift"],
                "tight_distractors": True,
                "n_options": 4,
                "layout_pool": ["horizontal", "grid_2x2"],
            }
        return {
            "sequence_length": 5,
            "rule_pool": ["multi_evolve", "grid_evolve",
                          "rotation_plus_color", "rotation_plus_stretch",
                          "position_shift", "color_shift"],
            "tight_distractors": True,
            "n_options": 5,
            "layout_pool": ["horizontal", "grid_2x2"],
        }

    _QUESTION_TEMPLATES = [
        "The top sequence follows a hidden rule. Apply the same rule ONCE to the target image C to produce the answer. Which option matches? Answer with a single letter.",
        "Infer the transformation rule from the sequence S1 -> S2 -> S3. Then apply it once to the target image C. Pick the matching option. Answer with a single letter.",
        "A progressive transformation is shown in the top sequence. Given the new starting image C, what is the result of applying the rule once more? Choose an option. Answer with a single letter.",
        "Study the sequence, determine the rule, and transform C one step forward. Which option matches? Answer with a single letter.",
        "Each step in the sequence applies the same rule. What does C become after one application? Answer with a single letter.",
        "Read the transformation from S1 to S2 to S3 (to S4/S5). Apply it once to C. Which lettered option is correct? Answer with a single letter.",
    ]

    _TITLE_VARIANTS = [
        "Sequence (infer the rule):",
        "Top: observe the transformation rule",
        "Top row follows a hidden rule",
        "Study the sequence, infer the rule",
        "Pattern progression",
        "Rule-based transformation",
    ]

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1307)
        self._primary_complexity_feature = cfg["sequence_length"] + level

        for _ in range(40):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # State + transformations
    # ------------------------------------------------------------------ #

    def _initial_state(self, rule_type: str, rng: random.Random,
                       params: Optional[Dict] = None) -> Dict:
        params = params or {}
        # For rotation-involving rules, restrict shape choice so the
        # rotation is visually observable.
        rot_step = params.get("rot_step")
        if rule_type in ("rotation", "rotation_plus_color",
                         "rotation_plus_stretch") and rot_step is not None:
            eligible_shapes = [s for s in _BASE_SHAPES
                               if _rotation_visible(s, rot_step)]
            if not eligible_shapes:
                eligible_shapes = ["triangle", "pentagon", "star"]
            rot_shape_pool = eligible_shapes
        else:
            rot_shape_pool = _BASE_SHAPES
        if rule_type in ("count_grow", "count_shrink"):
            init_count = 1 if rule_type == "count_grow" else \
                rng.randint(4, 5)
            return {
                "kind": "count",
                "shape": rng.choice(_BASE_SHAPES),
                "count": init_count,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
            }
        if rule_type == "shape_cycle":
            return {
                "kind": "shape",
                "shape_cycle_idx": 0,
                "shape_cycle_len": rng.choice([3, 4, 5]),
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "size_val": 1.0,
                "stretch": 1.0,
                "rotation": 0,
            }
        if rule_type == "size_cycle":
            return {
                "kind": "size",
                "shape": rng.choice(_BASE_SHAPES),
                "size_step": 0,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "rotation": 0,
            }
        if rule_type in ("stretch_x", "stretch_y"):
            # avoid circle for stretch (stretched circle is just an ellipse,
            # but fine); however a stretched ellipse on x-axis works. Keep
            # all shapes; visibility is ensured by lower factor cap.
            return {
                "kind": "shape",
                "shape": rng.choice(_BASE_SHAPES),
                "stretch": 1.0,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "rotation": 0,
                "stretch_axis": rule_type[-1],
            }
        if rule_type == "rotation":
            return {
                "kind": "shape",
                "shape": rng.choice(rot_shape_pool),
                "stretch": 1.0,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "rotation": 0,
                "pos_offset": (0.0, 0.0),
                "mirrored": False,
            }
        if rule_type in ("color_shift", "mirror", "position_shift"):
            # mirror: prefer shapes that are NOT left-right symmetric as drawn,
            # otherwise mirroring is visually invisible. Our renderer draws
            # all polygons aligned with y-axis → most shapes ARE symmetric.
            # Keep the rule set but avoid circles/ellipses/squares when mirror
            # is the rule (they look identical mirrored).
            if rule_type == "mirror":
                # Honestly none of our shapes show a visible mirror effect
                # unless they are rotated. Use triangle/pentagon (which are
                # drawn y-symmetric) → still invisible. To make mirror
                # observable we would need asymmetric shapes. This rule
                # is problematic; mitigate by excluding rule from generation
                # (done via _rule_params / _try_generate path).
                shape_pool = ["triangle", "pentagon", "star"]
            else:
                shape_pool = _BASE_SHAPES
            return {
                "kind": "shape",
                "shape": rng.choice(shape_pool),
                "stretch": 1.0,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "rotation": 0,
                "pos_offset": (0.0, 0.0),
                "mirrored": False,
            }
        if rule_type == "rotation_plus_color":
            return {
                "kind": "shape",
                "shape": rng.choice(rot_shape_pool),
                "stretch": 1.0,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "rotation": 0,
            }
        if rule_type == "rotation_plus_stretch":
            return {
                "kind": "shape",
                "shape": rng.choice(rot_shape_pool),
                "stretch": 1.0,
                "color_idx": rng.randint(0, len(_COLORS) - 1),
                "rotation": 0,
            }
        if rule_type == "grid_evolve":
            n = 4
            live = set()
            k = rng.randint(1, 3)
            while len(live) < k:
                live.add(rng.randint(0, n * n - 1))
            return {
                "kind": "grid",
                "n": n,
                "live": live,
                "color": _COLORS[rng.randint(0, len(_COLORS) - 1)],
            }
        # multi_evolve
        n = 4
        live = set()
        k = rng.randint(2, 4)
        while len(live) < k:
            live.add(rng.randint(0, n * n - 1))
        return {
            "kind": "grid",
            "n": n,
            "live": live,
            "color_idx": rng.randint(0, len(_COLORS) - 1),
        }

    def _apply_rule(self, state: Dict, rule_type: str, params: Dict,
                    rng: random.Random) -> Dict:
        new = dict(state)
        if state["kind"] == "count":
            step = params.get("count_step", 1)
            new["count"] = max(1, state["count"] + step)
            return new
        if state["kind"] == "size":
            new["size_step"] = state["size_step"] + params.get(
                "size_step", 1)
            return new
        if state["kind"] == "shape":
            if rule_type == "shape_cycle":
                new["shape_cycle_idx"] = (
                    state["shape_cycle_idx"] + params.get("cycle_step", 1)
                ) % state["shape_cycle_len"]
                return new
            if rule_type == "stretch_x":
                new["stretch"] = state["stretch"] * params.get(
                    "factor", 1.3)
                new["stretch_axis"] = "x"
                return new
            if rule_type == "stretch_y":
                new["stretch"] = state["stretch"] * params.get(
                    "factor", 1.3)
                new["stretch_axis"] = "y"
                return new
            if rule_type == "color_shift":
                new["color_idx"] = (state["color_idx"]
                                    + params.get("step", 1)) % len(_COLORS)
                return new
            if rule_type == "rotation":
                new["rotation"] = (state["rotation"]
                                   + params.get("rot_step", 45)) % 360
                return new
            if rule_type == "mirror":
                new["mirrored"] = not state.get("mirrored", False)
                # also shift color slightly so mirrored + unmirrored look
                # distinct after round-trip
                return new
            if rule_type == "position_shift":
                ox, oy = state.get("pos_offset", (0.0, 0.0))
                dx = params.get("dx", 0.15)
                dy = params.get("dy", 0.0)
                new["pos_offset"] = (ox + dx, oy + dy)
                return new
            if rule_type == "rotation_plus_color":
                new["rotation"] = (state["rotation"]
                                   + params.get("rot_step", 45)) % 360
                new["color_idx"] = (state["color_idx"]
                                    + params.get("color_step", 1)) % len(
                    _COLORS)
                return new
            if rule_type == "rotation_plus_stretch":
                new["rotation"] = (state["rotation"]
                                   + params.get("rot_step", 45)) % 360
                new["stretch"] = state["stretch"] * params.get(
                    "factor", 1.3)
                return new
            return new
        # grid
        if rule_type == "grid_evolve":
            d = params.get("delta", 1)
            new_live = set()
            for idx in state["live"]:
                new_live.add(idx)
                cand = idx + d
                if 0 <= cand < state["n"] * state["n"]:
                    new_live.add(cand)
            new["live"] = new_live
            new["color"] = state["color"]
            return new
        # multi_evolve
        d = params.get("delta", 1)
        new_live = set()
        for idx in state["live"]:
            new_live.add(idx)
            cand = idx + d
            if 0 <= cand < state["n"] * state["n"]:
                new_live.add(cand)
        new["live"] = new_live
        new["color_idx"] = (state.get("color_idx", 0)
                              + params.get("color_step", 1)) % len(_COLORS)
        return new

    # ------------------------------------------------------------------ #
    # Problem construction
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[
                          Tuple[str, str, Image.Image]]:
        rule_type = rng.choice(cfg["rule_pool"])
        params = self._rule_params(rule_type, rng)

        seq_len = cfg["sequence_length"]
        sequence = [self._initial_state(rule_type, rng, params)]
        for _ in range(seq_len - 1):
            nxt = self._apply_rule(sequence[-1], rule_type, params, rng)
            if _states_equal(nxt, sequence[-1]):
                return None
            sequence.append(nxt)

        # For color-involving rules, ensure the target C starts at a color
        # whose "next" is visibly demonstrable by re-picking its color_idx
        # so it shares the same monotonic cycle direction. With the new
        # hue-ordered palette a step=k rule looks the same everywhere in
        # the palette, so this is primarily defensive — but we additionally
        # pick C's starting color from a range adjacent to (not equal to)
        # the sequence's range, so the analogy is unambiguous.
        c_state = None
        for _ in range(25):
            cand = self._initial_state(rule_type, rng, params)
            if not any(_states_equal(cand, s) for s in sequence):
                c_state = cand
                break
        if c_state is None:
            return None
        correct = self._apply_rule(c_state, rule_type, params, rng)
        if _states_equal(correct, c_state):
            return None

        n_distractors = cfg["n_options"] - 1
        distractors = self._make_distractors(cfg, rng, rule_type, params,
                                             c_state, correct,
                                             n_distractors + 2)
        if len(distractors) < n_distractors:
            return None

        options = list(distractors[:n_distractors])
        correct_idx = rng.randint(0, len(options))
        options.insert(correct_idx, correct)
        answer_letter = chr(ord("A") + correct_idx)

        question = rng.choice(self._QUESTION_TEMPLATES)
        layout = rng.choice(cfg["layout_pool"])
        image = self._render(sequence, c_state, options, rng, layout)
        return question, answer_letter, image

    def _rule_params(self, rule_type: str, rng: random.Random) -> Dict:
        if rule_type == "count_grow":
            return {"count_step": 1}
        if rule_type == "count_shrink":
            return {"count_step": -1}
        if rule_type == "shape_cycle":
            return {"cycle_step": 1}
        if rule_type == "size_cycle":
            return {"size_step": rng.choice([1, -1])}
        if rule_type in ("stretch_x", "stretch_y"):
            # Keep factors low so progression remains visible even at
            # seq_len=5 (1.2**4 = 2.07, well within new cap of 1.96).
            return {"factor": rng.choice([1.15, 1.2, 1.25])}
        if rule_type == "color_shift":
            # With a 12-color hue-ordered palette, steps 1-2 stay visually
            # coherent; larger steps jump across the color wheel and become
            # hard to read.
            return {"step": rng.choice([1, 2])}
        if rule_type == "rotation":
            return {"rot_step": rng.choice([30, 45, 60, 90])}
        if rule_type == "mirror":
            return {}
        if rule_type == "position_shift":
            # dx=0.1 is borderline visible; force minimum ≥ 0.15.
            return {"dx": rng.choice([0.15, 0.2, -0.15, -0.2]),
                    "dy": rng.choice([0.0, 0.15, -0.15])}
        if rule_type == "rotation_plus_color":
            return {"rot_step": rng.choice([30, 45, 60]),
                    "color_step": rng.choice([1, 2])}
        if rule_type == "rotation_plus_stretch":
            return {"rot_step": rng.choice([30, 45, 60]),
                    "factor": rng.choice([1.15, 1.2, 1.25])}
        if rule_type == "grid_evolve":
            return {"delta": rng.choice([1, 4, 5])}
        # multi_evolve
        return {"delta": rng.choice([1, 4, 5]),
                "color_step": rng.choice([1, 2])}

    def _make_distractors(self, cfg, rng, rule_type, params, c_state,
                          correct, n_needed):
        distractors = []
        seen = [correct, c_state]
        seen_sigs = {_visual_signature(correct), _visual_signature(c_state)}

        def _add(cand):
            # Reject if logically equal to an existing option or if the
            # rendered image would be visually indistinguishable from one.
            for s in seen:
                if _states_equal(cand, s):
                    return False
            sig = _visual_signature(cand)
            if sig in seen_sigs:
                return False
            seen.append(cand)
            seen_sigs.add(sig)
            distractors.append(cand)
            return True

        _add(dict(c_state))
        twice = self._apply_rule(correct, rule_type, params, rng)
        _add(twice)

        # wrong parameter
        alt_params = dict(params)
        if rule_type in ("count_grow", "count_shrink"):
            alt_params["count_step"] = -params.get("count_step", 1)
        elif rule_type == "shape_cycle":
            alt_params["cycle_step"] = (params.get("cycle_step", 1) + 1)
        elif rule_type == "size_cycle":
            alt_params["size_step"] = -params.get("size_step", 1)
        elif rule_type in ("stretch_x", "stretch_y"):
            alt_params["factor"] = params["factor"] * 0.7
        elif rule_type == "color_shift":
            alt_params["step"] = -params.get("step", 1)
        elif rule_type == "rotation":
            alt_params["rot_step"] = -params.get("rot_step", 45)
        elif rule_type == "position_shift":
            alt_params["dx"] = -params.get("dx", 0.15)
            alt_params["dy"] = -params.get("dy", 0.0)
        elif rule_type == "rotation_plus_color":
            alt_params["color_step"] = -params.get("color_step", 1)
        elif rule_type == "rotation_plus_stretch":
            alt_params["factor"] = params["factor"] * 0.7
        elif rule_type in ("grid_evolve", "multi_evolve"):
            alts = [1, 4, 5]
            alts = [a for a in alts if a != params.get("delta")]
            if alts:
                alt_params["delta"] = rng.choice(alts)
        alt_result = self._apply_rule(c_state, rule_type, alt_params, rng)
        _add(alt_result)

        attempts = 0
        # Aux: jitter factors / alternate steps to generate more variations
        alt_factors = [1.15, 1.35, 1.5, 0.8, 0.6]
        alt_rots = [15, 30, 60, 75, 90, -30, -45, -60, -90]
        alt_steps = [2, -2, 3, -3, 4, -4]
        alt_deltas = [1, 4, 5, 2, 3, 6]
        alt_count_steps = [2, -2, 3, -1, 4, -3]
        alt_dxs = [0.2, -0.2, 0.25, 0.3, -0.1]
        alt_dys = [0.15, -0.15, 0.2, -0.2, 0.0]
        while len(distractors) < n_needed and attempts < 80:
            attempts += 1
            if rule_type in ("count_grow", "count_shrink"):
                base_count = c_state["count"]
                offsets = [2, 3, -1, 4, -2, 5]
                off = offsets[attempts % len(offsets)]
                cand = dict(c_state)
                cand["count"] = max(1, base_count + off) if off != 1 \
                    else base_count + 2
                if attempts >= 2:
                    alt_shapes = [s for s in _BASE_SHAPES if s != c_state[
                        "shape"]]
                    if alt_shapes:
                        cand["shape"] = alt_shapes[attempts %
                                                   len(alt_shapes)]
            elif rule_type == "shape_cycle":
                cand = dict(c_state)
                cand["shape_cycle_idx"] = (
                    cand["shape_cycle_idx"] + attempts + 2
                ) % cand["shape_cycle_len"]
            elif rule_type == "size_cycle":
                cand = dict(c_state)
                cand["size_step"] = attempts + 2
            else:
                ap = dict(params)
                ap["factor"] = alt_factors[attempts % len(alt_factors)]
                ap["rot_step"] = alt_rots[attempts % len(alt_rots)]
                ap["step"] = alt_steps[attempts % len(alt_steps)]
                ap["color_step"] = alt_steps[attempts % len(alt_steps)]
                ap["delta"] = alt_deltas[attempts % len(alt_deltas)]
                ap["count_step"] = alt_count_steps[attempts % len(
                    alt_count_steps)]
                ap["dx"] = alt_dxs[attempts % len(alt_dxs)]
                ap["dy"] = alt_dys[attempts % len(alt_dys)]
                cand = self._apply_rule(c_state, rule_type, ap, rng)
            _add(cand)
        return distractors

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, sequence, c_state, options, rng, layout) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        seq_len = len(sequence)
        n_opts = len(options)

        if layout == "grid_2x2" and seq_len >= 4:
            return self._render_grid(sequence, c_state, options, rng,
                                     style, seq_len, n_opts, sc)
        return self._render_horizontal(sequence, c_state, options, rng,
                                       style, seq_len, n_opts, sc)

    def _render_horizontal(self, sequence, c_state, options, rng, style,
                           seq_len, n_opts, sc):
        fig = plt.figure(figsize=(max(11.5, 2.0 * seq_len + 2.5) * sc,
                                  7.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.2],
                              hspace=0.45)
        ax_seq = fig.add_subplot(gs[0])
        ax_c = fig.add_subplot(gs[1])
        ax_opt = fig.add_subplot(gs[2])

        ax_seq.set_aspect("equal")
        ax_seq.axis("off")
        ax_seq.set_title(rng.choice(self._TITLE_VARIANTS),
                         fontsize=12, fontweight="bold", pad=6, loc="left")
        cell_size = 1.5
        arrow_color = rng.choice(["#e67e22", "#d35400", "#2980b9",
                                  "#8e44ad", "#c0392b"])
        cell_face = rng.choice(["#fdfefe", "#fef9e7", "#eaf2f8",
                                "#e8f8f5", "#fdf2e9"])
        for i, st in enumerate(sequence):
            cx = i * 2.0 + 1.1
            cy = 1.1
            _draw_cell_bg(ax_seq, cx, cy, cell_size, facecolor=cell_face)
            _draw_state(ax_seq, cx, cy, cell_size, st)
            ax_seq.text(cx, cy - cell_size / 2 - 0.15, f"S{i + 1}",
                        fontsize=11, fontweight="bold", ha="center",
                        va="top")
            if i < seq_len - 1:
                ax_seq.annotate("",
                                xy=(cx + cell_size / 2 + 0.3, cy),
                                xytext=(cx + cell_size / 2 + 0.05, cy),
                                arrowprops=dict(arrowstyle="->", lw=2,
                                                color=arrow_color))
        ax_seq.set_xlim(0, seq_len * 2.0 + 0.5)
        ax_seq.set_ylim(0, 2.2)

        ax_c.set_aspect("equal")
        ax_c.axis("off")
        ax_c.set_title("Apply the same rule to C:", fontsize=12,
                       fontweight="bold", pad=4, loc="left")
        _draw_cell_bg(ax_c, 1.1, 1.1, cell_size, facecolor=cell_face)
        _draw_state(ax_c, 1.1, 1.1, cell_size, c_state)
        ax_c.text(1.1, 1.1 - cell_size / 2 - 0.15, "C",
                  fontsize=11, fontweight="bold", ha="center", va="top")
        ax_c.annotate("", xy=(1.1 + cell_size / 2 + 1.0, 1.1),
                      xytext=(1.1 + cell_size / 2 + 0.1, 1.1),
                      arrowprops=dict(arrowstyle="->", lw=2,
                                      color=arrow_color))
        _draw_cell_bg(ax_c, 3.2, 1.1, cell_size, dash=True)
        ax_c.text(3.2, 1.1, "?", fontsize=26, fontweight="bold",
                  ha="center", va="center", color="#e74c3c")
        ax_c.text(3.2, 1.1 - cell_size / 2 - 0.15, "?",
                  fontsize=11, fontweight="bold", ha="center",
                  va="top", color="#e74c3c")
        ax_c.set_xlim(0, 5)
        ax_c.set_ylim(0, 2.2)

        ax_opt.set_aspect("equal")
        ax_opt.axis("off")
        ax_opt.set_title("Options:", fontsize=12, fontweight="bold",
                         pad=4, loc="left")
        opt_cell = 1.4
        for i, opt in enumerate(options):
            cx = i * 1.8 + 1.1
            cy = 1.05
            _draw_cell_bg(ax_opt, cx, cy, opt_cell,
                          facecolor="#eaf2f8")
            _draw_state(ax_opt, cx, cy, opt_cell, opt)
            ax_opt.text(cx, cy - opt_cell / 2 - 0.15,
                        chr(ord("A") + i), fontsize=12,
                        fontweight="bold", ha="center", va="top")
        ax_opt.set_xlim(0, n_opts * 1.8 + 0.5)
        ax_opt.set_ylim(0, 2.1)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.04,
                            hspace=0.45)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_grid(self, sequence, c_state, options, rng, style,
                     seq_len, n_opts, sc):
        # 2x2 for the sequence
        fig = plt.figure(figsize=(11.0 * sc, 9.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.2, 1.3],
                              hspace=0.35)
        ax_seq = fig.add_subplot(gs[0])
        ax_c = fig.add_subplot(gs[1])
        ax_opt = fig.add_subplot(gs[2])

        ax_seq.set_aspect("equal")
        ax_seq.axis("off")
        ax_seq.set_title(rng.choice(self._TITLE_VARIANTS),
                         fontsize=13, fontweight="bold", pad=6, loc="left")
        arrow_color = rng.choice(["#e67e22", "#d35400", "#2980b9",
                                  "#8e44ad"])
        cell_face = rng.choice(["#fdfefe", "#fef9e7", "#eaf2f8",
                                "#e8f8f5", "#fdf2e9"])
        cell_size = 1.4
        # Arrange in 2x3 (up to 6)
        n_cols = min(3, (seq_len + 1) // 2)
        ax_seq.set_xlim(0, n_cols * 2.0 + 0.5)
        ax_seq.set_ylim(0, 4.2)
        for i, st in enumerate(sequence):
            r = i // n_cols
            c = i % n_cols
            cx = c * 2.0 + 1.1
            cy = 3.0 - r * 1.8
            _draw_cell_bg(ax_seq, cx, cy, cell_size, facecolor=cell_face)
            _draw_state(ax_seq, cx, cy, cell_size, st)
            ax_seq.text(cx, cy - cell_size / 2 - 0.15, f"S{i + 1}",
                        fontsize=11, fontweight="bold", ha="center",
                        va="top")
            if i < seq_len - 1:
                next_r = (i + 1) // n_cols
                next_c = (i + 1) % n_cols
                next_cx = next_c * 2.0 + 1.1
                next_cy = 3.0 - next_r * 1.8
                ax_seq.annotate("",
                                xy=(next_cx - cell_size / 2 - 0.1,
                                    next_cy),
                                xytext=(cx + cell_size / 2 + 0.1, cy),
                                arrowprops=dict(
                                    arrowstyle="->", lw=1.8,
                                    color=arrow_color,
                                    connectionstyle="arc3,rad=-0.1"))

        ax_c.set_aspect("equal")
        ax_c.axis("off")
        ax_c.set_title("Apply the same rule to C:", fontsize=12,
                       fontweight="bold", pad=4, loc="left")
        _draw_cell_bg(ax_c, 1.1, 1.1, cell_size, facecolor=cell_face)
        _draw_state(ax_c, 1.1, 1.1, cell_size, c_state)
        ax_c.text(1.1, 1.1 - cell_size / 2 - 0.15, "C", fontsize=11,
                  fontweight="bold", ha="center", va="top")
        ax_c.annotate("", xy=(1.1 + cell_size / 2 + 1.0, 1.1),
                      xytext=(1.1 + cell_size / 2 + 0.1, 1.1),
                      arrowprops=dict(arrowstyle="->", lw=2,
                                      color=arrow_color))
        _draw_cell_bg(ax_c, 3.2, 1.1, cell_size, dash=True)
        ax_c.text(3.2, 1.1, "?", fontsize=26, fontweight="bold",
                  ha="center", va="center", color="#e74c3c")
        ax_c.text(3.2, 1.1 - cell_size / 2 - 0.15, "?", fontsize=11,
                  fontweight="bold", ha="center", va="top",
                  color="#e74c3c")
        ax_c.set_xlim(0, 5)
        ax_c.set_ylim(0, 2.2)

        ax_opt.set_aspect("equal")
        ax_opt.axis("off")
        ax_opt.set_title("Options:", fontsize=12, fontweight="bold",
                         pad=4, loc="left")
        opt_cell = 1.3
        for i, opt in enumerate(options):
            cx = i * 1.7 + 1.1
            cy = 1.05
            _draw_cell_bg(ax_opt, cx, cy, opt_cell,
                          facecolor="#eaf2f8")
            _draw_state(ax_opt, cx, cy, opt_cell, opt)
            ax_opt.text(cx, cy - opt_cell / 2 - 0.15,
                        chr(ord("A") + i), fontsize=12,
                        fontweight="bold", ha="center", va="top")
        ax_opt.set_xlim(0, n_opts * 1.7 + 0.5)
        ax_opt.set_ylim(0, 2.1)

        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.04,
                            hspace=0.35)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Drawing helpers
# ---------------------------------------------------------------------- #

_SHAPE_CYCLE = ["circle", "square", "triangle", "pentagon", "hexagon",
                "star", "diamond"]

def _visual_signature(state: Dict) -> Tuple:
    """Return a tuple capturing the state's visible rendering.

    Two states producing indistinguishable rendered cells share the same
    signature. Used to filter duplicate distractor options.
    """
    k = state.get("kind")
    if k == "count":
        n = max(1, min(6, state.get("count", 1)))
        return ("count", state.get("shape"), n, state.get("color_idx"))
    if k == "size":
        step = state.get("size_step", 0)
        # Matches the clamping in _draw_size.
        idx = max(0, min(8, 4 + step))
        return ("size", state.get("shape"), idx, state.get("color_idx"))
    if k == "shape":
        if "shape_cycle_idx" in state:
            shape = _SHAPE_CYCLE[state["shape_cycle_idx"]
                                 % len(_SHAPE_CYCLE)]
        else:
            shape = state.get("shape", "circle")
        # Rotation folded by the shape's visible symmetry group.
        sym = _ROT_SYMMETRY.get(shape, 360)
        rot = state.get("rotation", 0) % 360
        if sym == 0:
            eff_rot = 0
        else:
            eff_rot = rot % sym
        # Stretch folded by the renderer's saturation cap. base=0.28,
        # cap=0.55 → effective max multiplier = 0.55 / 0.28 ≈ 1.964.
        stretch = state.get("stretch", 1.0)
        eff_stretch = min(stretch, 0.55 / 0.28)
        # Round to 2 decimals to collapse near-identical rendered widths.
        eff_stretch_r = round(eff_stretch, 2)
        axis = state.get("stretch_axis", "x")
        pos = state.get("pos_offset", (0.0, 0.0))
        pos_r = (round(pos[0], 3), round(pos[1], 3))
        # Mirror flag: on shapes that are drawn y-axis symmetric and
        # unrotated, mirroring is invisible. In this env all polygon
        # renderings are y-symmetric, so we only treat mirror as visible
        # when combined with a non-symmetric rotation.
        mirrored = bool(state.get("mirrored", False))
        y_symmetric_shapes = {"circle", "square", "triangle", "pentagon",
                              "hexagon", "diamond", "star", "ellipse"}
        if mirrored and shape in y_symmetric_shapes and eff_rot == 0:
            mirrored = False
        return ("shape", shape, eff_stretch_r, axis,
                state.get("color_idx"), eff_rot, mirrored, pos_r)
    if k == "grid":
        live = tuple(sorted(state.get("live", set())))
        return ("grid", state.get("n"), live,
                state.get("color_idx", -1), state.get("color"))
    return ("unknown",)

def _states_equal(a: Dict, b: Dict) -> bool:
    if a.get("kind") != b.get("kind"):
        return False
    if a["kind"] == "count":
        return (a["shape"] == b["shape"]
                and a["count"] == b["count"]
                and a["color_idx"] == b["color_idx"])
    if a["kind"] == "size":
        return (a["shape"] == b["shape"]
                and a["size_step"] == b["size_step"]
                and a["color_idx"] == b["color_idx"])
    if a["kind"] == "shape":
        return (a.get("shape") == b.get("shape")
                and a.get("shape_cycle_idx") == b.get("shape_cycle_idx")
                and a.get("shape_cycle_len") == b.get("shape_cycle_len")
                and abs(a.get("stretch", 1.0)
                        - b.get("stretch", 1.0)) < 1e-6
                and a["color_idx"] == b["color_idx"]
                and a.get("rotation", 0) == b.get("rotation", 0)
                and a.get("mirrored", False) == b.get("mirrored", False)
                and a.get("pos_offset", (0.0, 0.0))
                == b.get("pos_offset", (0.0, 0.0))
                and a.get("stretch_axis", "x")
                == b.get("stretch_axis", "x"))
    return (a["n"] == b["n"]
            and a["live"] == b["live"]
            and a.get("color_idx", -1) == b.get("color_idx", -1)
            and a.get("color") == b.get("color"))

def _draw_cell_bg(ax, cx, cy, cell_size, dash=False, facecolor="#fdfefe"):
    rect = mpatches.FancyBboxPatch(
        (cx - cell_size / 2, cy - cell_size / 2),
        cell_size, cell_size, boxstyle="round,pad=0.04",
        facecolor=facecolor,
        edgecolor=("#e74c3c" if dash else "#2c3e50"),
        linewidth=(2.5 if dash else 1.5),
        linestyle=("--" if dash else "-"), zorder=1)
    ax.add_patch(rect)

def _draw_state(ax, cx, cy, cell_size, state: Dict):
    kind = state["kind"]
    if kind == "shape":
        _draw_shape_state(ax, cx, cy, cell_size, state)
    elif kind == "count":
        _draw_count(ax, cx, cy, cell_size, state)
    elif kind == "size":
        _draw_size(ax, cx, cy, cell_size, state)
    else:
        _draw_grid(ax, cx, cy, cell_size * 0.88, state)

def _draw_count(ax, cx, cy, cell_size, state: Dict):
    shape = state["shape"]
    n = state["count"]
    n = max(1, min(6, n))
    inner = cell_size * 0.78
    r = inner / (2 * max(1, n + 0.3))
    total_w = r * 2 * n + r * (n - 1) * 0.2
    start_x = cx - total_w / 2 + r
    shape_state = {"kind": "shape", "shape": shape,
                   "color_idx": state["color_idx"], "stretch": 1.0,
                   "rotation": 0}
    for i in range(n):
        _draw_shape_state(ax, start_x + i * (r * 2.2), cy, r * 3.33,
                          shape_state)

def _draw_size(ax, cx, cy, cell_size, state: Dict):
    """Size cycle: monotonic visible growth / shrink based on step.

    Uses an ordered scale so positive steps always grow and negative
    steps always shrink with no modular zig-zag.
    """
    shape = state["shape"]
    step = state["size_step"]
    # 9-point scale, step=0 centered at idx 4. Saturates at ends for
    # out-of-range distractors but does not zig-zag.
    scale = [0.16, 0.20, 0.25, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60]
    names = ["XXS", "XS", "S", "SM", "M", "ML", "L", "XL", "XXL"]
    idx = max(0, min(len(scale) - 1, 4 + step))
    factor = scale[idx]
    color = _COLORS[state["color_idx"] % len(_COLORS)]
    radius = cell_size * factor
    _draw_base_shape(ax, cx, cy, radius, shape, color,
                     state.get("rotation", 0))
    # BUGFIX 2026-04-24: removed size-name text label (was "M", "SM", etc.)
    # drawn under each shape. The label OCR'd directly encoded the hidden
    # rule, letting VL models bypass visual reasoning and text-match.
    # `names` table retained above as documentation of the scale only.
    _ = names  # unused after label removal; kept for reference.

def _draw_shape_state(ax, cx, cy, cell_size, state):
    color = _COLORS[state["color_idx"] % len(_COLORS)]
    rotation = state.get("rotation", 0)
    mirrored = state.get("mirrored", False)
    stretch = state.get("stretch", 1.0)
    pos_offset = state.get("pos_offset", (0.0, 0.0))
    base_size = cell_size * 0.28
    h = base_size
    w = base_size
    axis = state.get("stretch_axis", "x")
    if axis == "x":
        w = base_size * stretch
    else:
        h = base_size * stretch
    # Raise the stretch cap to allow multi-step progression to remain
    # visually distinguishable (cap must be > base_size * max_stretch).
    w = min(w, cell_size * 0.55)
    h = min(h, cell_size * 0.55)

    # Shape selection
    if "shape_cycle_idx" in state:
        shape = _SHAPE_CYCLE[state["shape_cycle_idx"] % len(_SHAPE_CYCLE)]
    else:
        shape = state.get("shape", "circle")

    cx_draw = cx + pos_offset[0] * cell_size
    cy_draw = cy + pos_offset[1] * cell_size
    _draw_stretched_shape(ax, cx_draw, cy_draw, w, h, shape, color,
                          rotation, mirrored)

def _draw_stretched_shape(ax, cx, cy, w, h, shape, color, rotation,
                          mirrored):
    from matplotlib.transforms import Affine2D
    if mirrored:
        trans = Affine2D().scale(-1, 1).translate(2 * cx, 0) \
            .rotate_deg_around(cx, cy, rotation) + ax.transData
    elif rotation:
        trans = Affine2D().rotate_deg_around(cx, cy, rotation) + ax.transData
    else:
        trans = None

    if shape == "circle":
        p = mpatches.Ellipse((cx, cy), 2 * w, 2 * h, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    elif shape == "ellipse":
        p = mpatches.Ellipse((cx, cy), 2 * w, 2 * h * 0.7, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    elif shape == "square":
        p = mpatches.Rectangle((cx - w, cy - h), 2 * w, 2 * h,
                                facecolor=color, edgecolor="#2c3e50",
                                linewidth=1.5)
    elif shape == "diamond":
        pts = [(cx, cy + h), (cx + w, cy), (cx, cy - h), (cx - w, cy)]
        p = mpatches.Polygon(pts, closed=True, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    elif shape == "triangle":
        pts = [(cx, cy + h), (cx - w, cy - h), (cx + w, cy - h)]
        p = mpatches.Polygon(pts, closed=True, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    elif shape == "pentagon":
        pts = []
        for i in range(5):
            a = math.pi / 2 + 2 * math.pi * i / 5
            pts.append((cx + w * math.cos(a), cy + h * math.sin(a)))
        p = mpatches.Polygon(pts, closed=True, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    elif shape == "hexagon":
        pts = []
        for i in range(6):
            a = math.radians(30) + 2 * math.pi * i / 6
            pts.append((cx + w * math.cos(a), cy + h * math.sin(a)))
        p = mpatches.Polygon(pts, closed=True, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    elif shape == "star":
        pts = []
        for i in range(10):
            a = math.pi / 2 + 2 * math.pi * i / 10
            r = (w + h) / 2 if i % 2 == 0 else (w + h) / 2 * 0.45
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        p = mpatches.Polygon(pts, closed=True, facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5)
    else:
        p = mpatches.Circle((cx, cy), max(w, h), facecolor=color,
                             edgecolor="#2c3e50", linewidth=1.5)

    if trans is not None:
        p.set_transform(trans)
    ax.add_patch(p)

def _draw_base_shape(ax, cx, cy, radius, shape, color, rotation):
    _draw_stretched_shape(ax, cx, cy, radius, radius, shape, color,
                          rotation, False)

def _draw_grid(ax, cx, cy, total_size, state):
    n = state["n"]
    live = state["live"]
    color = state.get("color") or _COLORS[state["color_idx"] % len(_COLORS)]
    cell_sz = total_size / n
    x0 = cx - total_size / 2
    y0 = cy - total_size / 2
    for idx in range(n * n):
        r = idx // n
        c = idx % n
        gx = x0 + c * cell_sz
        gy = y0 + (n - 1 - r) * cell_sz
        fc = color if idx in live else "#ffffff"
        rect = mpatches.Rectangle((gx, gy), cell_sz, cell_sz,
                                   facecolor=fc, edgecolor="#2c3e50",
                                   linewidth=0.8)
        ax.add_patch(rect)

if __name__ == "__main__":
    import collections
    env = AnalogyFromSequenceQA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
    for lv in (0, 3, 6, 9):
        ans = collections.Counter()
        for s in range(20):
            e = AnalogyFromSequenceQA()
            e.generate(s * 1000 + lv * 37 + 17, {'level': lv})
            ans[e._answer] += 1
        print(f"L{lv} letters: {dict(ans)}")
