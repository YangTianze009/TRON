"""
Similar Figure Area Ratio QA environment.

Goal: given two similar figures (squares, rectangles, triangles, or
general polygons) drawn side by side with at least one dimension
labeled on each, determine the area ratio or the area of one figure
given the other's area.

Targets Plane Geometry reasoning Property.

Difficulty axes:
  A) shape_type: square/rectangle -> triangle -> irregular polygon.
  B) ratio_complexity & question_type: simple integer ratio with ratio
     question at L0; L5 asks for computed area; L9 uses mixed
     non-corresponding labeled sides.

Format: 4-way MCQ (single letter).
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

_SHAPE_POOL_PER_LEVEL = {
    # level → list of shape names
}

_Q_TEMPLATES_RATIO = [
    "What is the ratio of their areas (smaller : larger)?",
    "Using the labeled side lengths, compute the area ratio (smaller to larger).",
    "Find the ratio of the smaller figure's area to the larger figure's area.",
    "Determine the smaller-to-larger area ratio from the labeled side lengths.",
]

_Q_TEMPLATES_AREA_LARGE = [
    "Using the known area labeled inside the smaller figure, what is the area of the larger figure?",
    "Given the labeled small-figure area, compute the larger figure's area.",
    "Based on the dimensions shown, what is the area of the larger figure?",
]

_Q_TEMPLATES_AREA_SMALL = [
    "Using the known area labeled inside the larger figure, what is the area of the smaller figure?",
    "Given the labeled large-figure area, compute the smaller figure's area.",
    "Based on the dimensions shown, what is the area of the smaller figure?",
]

# L9 chain: given BOTH areas + smaller side, find the larger side (the
# scale factor must be derived from sqrt(area ratio) — two steps).
_Q_TEMPLATES_SIDE_CHAIN = [
    "Given the two areas labeled inside the figures and the labeled "
    "side of the smaller figure, what is the corresponding side length "
    "of the larger figure? Round to the nearest 0.1.",
    "From the two labeled areas and the smaller figure's labeled side, "
    "determine the matching side length of the larger figure.",
    "Using both area labels and the smaller figure's side, find the "
    "larger figure's corresponding side length.",
]

class SimilarFigureAreaRatioQA(StandaloneVisualEnv):
    ENV_NAME = "similar_figure_area_ratio"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per mmmath similarity-of-shapes.
        # L0-L1: rect/square only, simple ratio (1:2, 1:3), ASK RATIO
        # L2-L3: + triangle, simple ratios, ASK AREA
        # L4-L5: + polygons, mid ratios, ASK AREA
        # L6-L7: + irregular, mid-hard ratios, ASK AREA + non-corresponding
        # L8: ugly ratios + non-corresponding + tight distractors
        # L9: ugly ratios + CHAIN (sqrt of area ratio → find side, 2-step)
        level = max(0, min(level, 9))
        if level <= 1:
            shape_pool = ["rect_or_square"]
        elif level <= 3:
            shape_pool = ["rect_or_square", "triangle"]
        elif level <= 5:
            shape_pool = ["rect_or_square", "triangle", "regular_polygon_5",
                          "regular_polygon_6"]
        elif level <= 7:
            shape_pool = ["triangle", "regular_polygon_5", "regular_polygon_6",
                          "regular_polygon_8", "irregular"]
        else:
            shape_pool = ["regular_polygon_5", "regular_polygon_6",
                          "regular_polygon_8", "irregular", "trapezoid_iso"]
        if level <= 1:
            ratio_pool = [(1, 2), (1, 3)]
        elif level <= 3:
            ratio_pool = [(1, 2), (2, 3), (1, 3)]
        elif level <= 5:
            ratio_pool = [(1, 2), (2, 3), (3, 4), (1, 3)]
        elif level <= 7:
            ratio_pool = [(2, 3), (3, 4), (3, 5), (4, 5), (2, 5), (3, 7)]
        elif level == 8:
            # 2026-05-04 R4: ugly ratios at L8
            ratio_pool = [(2, 5), (3, 7), (4, 7), (5, 8), (3, 8)]
        else:
            # L9: very ugly ratios for chain mode
            ratio_pool = [(5, 11), (4, 9), (7, 13), (6, 11), (3, 11), (5, 13)]
        if level >= 9:
            question_type = "chain"
        elif level <= 1:
            question_type = "ratio"
        else:
            question_type = "area"
        # 2026-05-04 R4: non-corresponding starts at L6 (was L8)
        non_corresponding = level >= 6
        hide_ratio_in_text = True
        tight_distractors = level >= 6
        return {
            "shape_pool": shape_pool,
            "ratio_pool": ratio_pool,
            "question_type": question_type,
            "non_corresponding": non_corresponding,
            "hide_ratio_in_text": hide_ratio_in_text,
            "tight_distractors": tight_distractors,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 827)
        self._primary_complexity_feature = level

        for _ in range(25):
            try:
                r = self._try_generate(sub_rng, cfg, level)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        a, b = rng.choice(cfg["ratio_pool"])  # side ratio a:b
        if a == b:
            return None
        # Pick shape for this seed from the level pool (adds diversity per seed).
        chosen_shape = rng.choice(cfg["shape_pool"])
        cfg = dict(cfg)  # shallow copy to add chosen shape
        cfg["shape"] = chosen_shape
        # Small side length on smaller figure
        s_small = rng.randint(2, 5)
        s_large = s_small * b // a if (s_small * b) % a == 0 else s_small * b / a
        if cfg["question_type"] == "ratio":
            # Ask for area ratio
            gt_ratio = f"{a*a}:{b*b}"
            distractors_set = set()
            # Common misconceptions
            distractors_set.add(f"{a}:{b}")
            distractors_set.add(f"{b}:{a}")
            distractors_set.add(f"{a*a*a}:{b*b*b}")
            # Prepare 4 options
            opts = [gt_ratio]
            for d in list(distractors_set):
                if d != gt_ratio and d not in opts:
                    opts.append(d)
                if len(opts) == 4:
                    break
            if len(opts) < 4:
                opts.append(f"{a+1}:{b+1}")
            rng.shuffle(opts)
            if opts.count(gt_ratio) > 1:
                return None
            answer_letter = chr(ord("A") + opts.index(gt_ratio))

            if cfg.get("hide_ratio_in_text"):
                given_text = (
                    f"Two similar {self._shape_name(cfg['shape'])}s are shown. "
                    f"Each figure has a corresponding side labeled with its "
                    f"length. Use the labeled side lengths to determine the "
                    f"side ratio."
                )
            else:
                given_text = (
                    f"Two similar {self._shape_name(cfg['shape'])}s are shown. A "
                    f"side of the smaller figure is labeled, and the corresponding "
                    f"side of the larger figure is labeled, giving a side ratio "
                    f"of {a}:{b}."
                )
            ask_text = rng.choice(_Q_TEMPLATES_RATIO)
            shape_data = self._build_shape(rng, cfg["shape"], s_small,
                                           a, b, cfg["non_corresponding"])
            image = self._render(shape_data, given_text, ask_text,
                                 [str(o) for o in opts], cfg)
            q = (f"{given_text} {ask_text}\n" + "\n".join(
                f"  ({chr(ord('A')+i)}) {opts[i]}" for i in range(4))
                + "\nAnswer with a single letter.")
            return q, answer_letter, image

        # question_type == "chain" — L9 multi-step: model must derive
        # scale factor from sqrt(area ratio) and apply it to a side.
        if cfg["question_type"] == "chain":
            # Choose integer-friendly small area and a ratio whose
            # larger-area is also integer-friendly.
            small_area = rng.choice([36, 64, 100, 144, 196])
            area_ratio = (b * b) / (a * a)
            large_area = small_area * area_ratio
            # Only keep integer large area for clean chain problems.
            if abs(large_area - round(large_area)) > 1e-6:
                return None
            large_area = int(round(large_area))
            # BUGFIX 2026-04-24: previously used s_small (circumradius for
            # polygons n!=6) to compute GT, but the image labels v_small (the
            # visible side length, = 2*r*sin(π/n) for polygons). Build shape
            # first and use shape_data['v_small'] (the actual labeled side).
            _shape_pre = self._build_shape(rng, cfg["shape"], s_small,
                                            a, b, cfg["non_corresponding"])
            labeled_side = _shape_pre.get("v_small", s_small)
            # Ground truth: corresponding side of LARGER = labeled_side * (b/a).
            gt_val = labeled_side * (b / a)
            gt_r = round(gt_val, 1)
            if abs(gt_r - round(gt_r)) < 1e-6:
                gt_r = int(round(gt_r))
            # Distractors — common mistakes
            miscons = [
                round(labeled_side * area_ratio, 1),       # used area ratio as side ratio
                round(labeled_side * (a / b), 1),          # inverted ratio
                round(labeled_side + (b - a), 1),          # additive mistake
                round(labeled_side * math.sqrt(area_ratio) * 1.5, 1),  # wrong scale
                round(s_small * (b / a), 1),               # old buggy answer (s_small-based)
            ]
            miscons = [int(m) if abs(m - round(m)) < 1e-6 else m
                       for m in miscons]
            distractors = []
            for d in miscons:
                if d != gt_r and d > 0 and d not in distractors:
                    distractors.append(d)
                if len(distractors) == 3:
                    break
            if len(distractors) < 3:
                return None
            opts_vals = [gt_r] + distractors[:3]
            rng.shuffle(opts_vals)
            if opts_vals.count(gt_r) > 1:
                return None
            answer_letter = chr(ord("A") + opts_vals.index(gt_r))
            def _fmt_chain(v):
                if isinstance(v, int):
                    return str(v)
                if abs(v - round(v)) < 1e-6:
                    return str(int(round(v)))
                return f"{v:.1f}"
            opt_strs = [_fmt_chain(v) for v in opts_vals]
            non_corr_note = ""
            if cfg["non_corresponding"]:
                non_corr_note = (" Note: the labeled side on the larger figure "
                                 "may not correspond to the labeled side on the "
                                 "smaller figure.")
            given_text = (
                f"Two similar {self._shape_name(cfg['shape'])}s are shown. "
                f"The area of each figure is labeled inside it, and the "
                f"length of one side of the smaller figure is labeled.{non_corr_note}"
            )
            ask_text = rng.choice(_Q_TEMPLATES_SIDE_CHAIN)
            # BUGFIX 2026-04-24: reuse _shape_pre built above so v_small matches.
            shape_data = _shape_pre
            shape_data["known_on"] = "both"
            shape_data["known_area_val"] = small_area
            shape_data["known_area_large"] = large_area
            shape_data["suppress_large_side_label"] = True
            image = self._render(shape_data, given_text, ask_text,
                                 opt_strs, cfg)
            q = (f"{given_text} {ask_text}\n" + "\n".join(
                f"  ({chr(ord('A')+i)}) {opt_strs[i]}" for i in range(4))
                + "\nAnswer with a single letter.")
            return q, answer_letter, image

        # question_type == "area"
        # Given area of small figure, compute area of large (or vice versa).
        # Choose a round small area.
        small_area_base = rng.choice([12, 18, 20, 24, 27, 32, 36, 45, 50])
        small_area = small_area_base
        # Area ratio = a^2 : b^2
        area_ratio = (b * b) / (a * a)
        ask_for_large = rng.random() < 0.7
        if ask_for_large:
            gt_val = small_area * area_ratio
            ask_text = rng.choice(_Q_TEMPLATES_AREA_LARGE)
            known_on = "small"
            known_area_val = small_area
        else:
            large_area = small_area * area_ratio
            # Make large_area integer-friendly
            if abs(large_area - round(large_area)) > 1e-6:
                return None
            large_area = int(round(large_area))
            gt_val = large_area / area_ratio
            ask_text = rng.choice(_Q_TEMPLATES_AREA_SMALL)
            known_on = "large"
            known_area_val = large_area

        gt_r = round(gt_val, 2)
        if abs(gt_r - round(gt_r)) < 1e-6:
            gt_r = int(round(gt_r))

        # distractor misconceptions
        miscons = []
        # linear ratio applied to area
        miscons.append(small_area * (b / a) if ask_for_large else small_area / (b / a))
        # inverse direction
        miscons.append(small_area / area_ratio if ask_for_large else small_area * area_ratio)
        if cfg.get("tight_distractors"):
            # Near-miss: use (b/a)^2 with off-by-one ratio
            miscons.append(small_area * ((b + 1) / a) ** 2 if ask_for_large
                            else small_area / ((b + 1) / a) ** 2)
            miscons.append(small_area * ((b - 1) / a) ** 2 if ask_for_large
                            else small_area / ((b - 1) / a) ** 2)
            # Small +/- offsets near correct
            miscons.append(gt_r + rng.choice([-2, -1, 1, 2]))
        else:
            # +/- offsets
            miscons.append(gt_r + rng.choice([-5, -3, 3, 5]))
            miscons.append(gt_r + rng.choice([-10, -7, 7, 10]))
        miscons = [round(m, 2) for m in miscons]
        miscons = [int(m) if abs(m - round(m)) < 1e-6 else m for m in miscons]

        distractors = []
        for d in miscons:
            if d != gt_r and d > 0 and d not in distractors:
                distractors.append(d)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            return None

        opts_vals = [gt_r] + distractors[:3]
        rng.shuffle(opts_vals)
        if opts_vals.count(gt_r) > 1:
            return None
        answer_letter = chr(ord("A") + opts_vals.index(gt_r))

        def fmt(v):
            if isinstance(v, int):
                return str(v)
            if abs(v - round(v)) < 1e-6:
                return str(int(round(v)))
            return f"{v:.2f}"

        opt_strs = [fmt(v) for v in opts_vals]

        non_corr_note = ""
        if cfg["non_corresponding"]:
            non_corr_note = (" Note: the labeled sides may not be corresponding — "
                             "you must identify which sides correspond.")
        given_text = (
            f"Two similar {self._shape_name(cfg['shape'])}s are shown. "
            f"Each figure has a corresponding side labeled with its "
            f"length, and the known area of one figure is labeled "
            f"inside it. Use the labeled lengths to find the side "
            f"ratio.{non_corr_note}"
        )
        shape_data = self._build_shape(rng, cfg["shape"], s_small,
                                       a, b, cfg["non_corresponding"])
        # Pass known-area info for rendering
        shape_data["known_on"] = known_on
        shape_data["known_area_val"] = known_area_val
        image = self._render(shape_data, given_text, ask_text,
                             opt_strs, cfg)
        q = (f"{given_text} {ask_text}\n" + "\n".join(
            f"  ({chr(ord('A')+i)}) {opt_strs[i]}" for i in range(4))
            + "\nAnswer with a single letter.")
        return q, answer_letter, image

    @staticmethod
    def _shape_name(shape_type):
        return {
            "rect_or_square": "rectangle",
            "triangle": "triangle",
            "regular_polygon": "regular pentagon",
            "regular_polygon_5": "regular pentagon",
            "regular_polygon_6": "regular hexagon",
            "regular_polygon_8": "regular octagon",
            "trapezoid_iso": "isosceles trapezoid",
            "irregular": "polygon",
        }.get(shape_type, "figure")

    def _build_shape(self, rng, shape_type, s_small, a, b, non_corr):
        """Return dict of vertex lists for the two similar figures."""
        if shape_type == "rect_or_square":
            is_square = rng.random() < 0.5
            w1 = s_small
            h1 = s_small if is_square else s_small + rng.randint(1, 3)
            w2 = w1 * b / a
            h2 = h1 * b / a
            return {
                "type": "rect",
                "small": [(0, 0), (w1, 0), (w1, h1), (0, h1)],
                "large": [(0, 0), (w2, 0), (w2, h2), (0, h2)],
                "label_side": "w",
                "v_small": w1, "v_large": w2,
                "non_corr": non_corr,
            }
        if shape_type == "triangle":
            # right triangle for clarity
            base1 = s_small
            h1 = s_small + rng.randint(1, 3)
            base2 = base1 * b / a
            h2 = h1 * b / a
            return {
                "type": "triangle",
                "small": [(0, 0), (base1, 0), (0, h1)],
                "large": [(0, 0), (base2, 0), (0, h2)],
                "label_side": "base",
                "v_small": base1, "v_large": base2,
                "non_corr": non_corr,
            }
        if shape_type in ("regular_polygon", "regular_polygon_5",
                           "regular_polygon_6", "regular_polygon_8"):
            n_sides = {
                "regular_polygon": 5,
                "regular_polygon_5": 5,
                "regular_polygon_6": 6,
                "regular_polygon_8": 8,
            }[shape_type]
            r1 = s_small
            r2 = r1 * b / a
            verts_small = [(r1 * math.cos(2 * math.pi * i / n_sides),
                            r1 * math.sin(2 * math.pi * i / n_sides))
                           for i in range(n_sides)]
            verts_large = [(r2 * math.cos(2 * math.pi * i / n_sides),
                            r2 * math.sin(2 * math.pi * i / n_sides))
                           for i in range(n_sides)]
            s_len1 = 2 * r1 * math.sin(math.pi / n_sides)
            s_len2 = 2 * r2 * math.sin(math.pi / n_sides)
            return {
                "type": "poly",
                "small": verts_small,
                "large": verts_large,
                "label_side": "side",
                "v_small": round(s_len1, 2),
                "v_large": round(s_len2, 2),
                "non_corr": non_corr,
            }
        if shape_type == "trapezoid_iso":
            # isosceles trapezoid with a base/top ratio randomized
            top = s_small
            base = s_small + rng.randint(1, 3)
            h = s_small + rng.randint(1, 2)
            verts_small = [(0, 0), (base, 0),
                           ((base + top) / 2, h),
                           ((base - top) / 2, h)]
            k = b / a
            verts_large = [(x * k, y * k) for x, y in verts_small]
            return {
                "type": "poly",
                "small": verts_small,
                "large": verts_large,
                "label_side": "base",
                "v_small": base,
                "v_large": round(base * k, 2),
                "non_corr": non_corr,
            }
        # irregular polygon
        base_vert = [(0, 0), (3, 0), (4, 1.5), (3, 3), (1, 3.2), (-0.3, 1.5)]
        verts_small = [(x * s_small / 3, y * s_small / 3) for x, y in base_vert]
        k = b / a
        verts_large = [(x * k, y * k) for x, y in verts_small]
        s_len1 = math.hypot(verts_small[1][0] - verts_small[0][0],
                             verts_small[1][1] - verts_small[0][1])
        s_len2 = s_len1 * k
        return {
            "type": "poly",
            "small": verts_small,
            "large": verts_large,
            "label_side": "side",
            "v_small": round(s_len1, 2),
            "v_large": round(s_len2, 2),
            "non_corr": non_corr,
        }

    # -------------------------------------------------- #
    def _render(self, shape_data, given_text, ask_text,
                opts, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        palette = style["palette"]
        lw = style["line_width"]
        geo_line = style["geo_line_color"]
        rng = self._rng

        rot = rng.uniform(-0.15, 0.15)
        def R(pt):
            c, s = math.cos(rot), math.sin(rot)
            return (pt[0] * c - pt[1] * s, pt[0] * s + pt[1] * c)

        fig = plt.figure(figsize=(10.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        # Draw small figure centered on left, large on right
        verts_small = [R(p) for p in shape_data["small"]]
        verts_large = [R(p) for p in shape_data["large"]]

        # Shift small
        xs_s = [p[0] for p in verts_small]
        ys_s = [p[1] for p in verts_small]
        w_s = max(xs_s) - min(xs_s)
        small_shift = (-min(xs_s), -min(ys_s))
        verts_small = [(p[0] + small_shift[0], p[1] + small_shift[1])
                       for p in verts_small]

        xs_l = [p[0] for p in verts_large]
        ys_l = [p[1] for p in verts_large]
        w_l = max(xs_l) - min(xs_l)
        # Place large shape to the right of small
        x_offset = w_s + 2.0 - min(xs_l)
        y_offset = -min(ys_l)
        verts_large = [(p[0] + x_offset, p[1] + y_offset)
                       for p in verts_large]

        poly1 = mpatches.Polygon(verts_small, closed=True,
                                  facecolor=palette[0],
                                  edgecolor=geo_line,
                                  linewidth=lw + 0.4, alpha=0.32)
        poly2 = mpatches.Polygon(verts_large, closed=True,
                                  facecolor=palette[2],
                                  edgecolor=geo_line,
                                  linewidth=lw + 0.4, alpha=0.32)
        ax_f.add_patch(poly1)
        ax_f.add_patch(poly2)

        # Label first side — use a strong dark color (not palette[4]
        # which can be a light tint invisible on light backgrounds).
        label_color = "#8b0000"  # dark red, always readable
        def label_first_edge(verts, label, color):
            p1 = verts[0]
            p2 = verts[1]
            m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            ax_f.text(m[0], m[1] - 0.3, label,
                      fontsize=fs + 1, family=ff, fontweight="bold",
                      ha="center", color=color,
                      bbox=dict(boxstyle="round,pad=0.18",
                                facecolor="white",
                                edgecolor=color, alpha=0.95))

        # Helper: format number with at most 2 decimals, drop trailing zeros.
        def _fmt_num(v):
            if isinstance(v, int): return str(v)
            try:
                r = round(float(v), 2)
                if abs(r - round(r)) < 1e-9: return str(int(round(r)))
                return f"{r:.2f}"
            except Exception:
                return str(v)

        # If non_corresponding: label one side on small, a different
        # side on large (but the side ratio description in question is still correct).
        suppress_large = shape_data.get("suppress_large_side_label", False)
        if shape_data["non_corr"] and len(verts_large) >= 3 and not suppress_large:
            label_first_edge(verts_small, _fmt_num(shape_data["v_small"]),
                             label_color)
            # label a different edge (edge 1-2 instead of 0-1)
            p1 = verts_large[1]
            p2 = verts_large[2]
            m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            ax_f.text(m[0] + 0.3, m[1], _fmt_num(shape_data["v_large"]),
                      fontsize=fs + 1, family=ff, color=label_color,
                      fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.18",
                                facecolor="white", edgecolor=label_color,
                                alpha=0.95))
        else:
            label_first_edge(verts_small, _fmt_num(shape_data["v_small"]),
                             label_color)
            if not suppress_large:
                label_first_edge(verts_large, str(shape_data["v_large"]),
                                 label_color)

        # Add figure names (and known area if set)
        cs = (sum(p[0] for p in verts_small) / len(verts_small),
              sum(p[1] for p in verts_small) / len(verts_small))
        cl = (sum(p[0] for p in verts_large) / len(verts_large),
              sum(p[1] for p in verts_large) / len(verts_large))
        known_on = shape_data.get("known_on")
        known_area_val = shape_data.get("known_area_val")
        known_area_large = shape_data.get("known_area_large")
        f1_txt = "F1"
        f2_txt = "F2"
        if known_on == "small" and known_area_val is not None:
            f1_txt = f"F1\narea = {known_area_val}"
        elif known_on == "large" and known_area_val is not None:
            f2_txt = f"F2\narea = {known_area_val}"
        elif known_on == "both" and known_area_val is not None:
            f1_txt = f"F1\narea = {known_area_val}"
            if known_area_large is not None:
                f2_txt = f"F2\narea = {known_area_large}"
        ax_f.text(cs[0], cs[1], f1_txt, fontsize=fs + 1,
                  fontweight="bold", family=ff, ha="center", va="center",
                  color=geo_line)
        ax_f.text(cl[0], cl[1], f2_txt, fontsize=fs + 1,
                  fontweight="bold", family=ff, ha="center", va="center",
                  color=geo_line)

        all_xs = [p[0] for p in verts_small + verts_large]
        all_ys = [p[1] for p in verts_small + verts_large]
        pad = max(max(all_xs) - min(all_xs),
                  max(all_ys) - min(all_ys)) * 0.15 + 1.0
        ax_f.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax_f.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

        title_pool = ["Similar Figures", "Area Ratio",
                      "Similar Shapes", "Scaled Figures", "Geometry"]
        ax_f.set_title(rng.choice(title_pool),
                       fontsize=fs + 2, family=ff, pad=8)

        # Right panel: question + options
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Given:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        for ln in self._wrap(given_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, "Ask:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for ln in self._wrap(ask_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(opts):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _wrap(text: str, width: int = 40) -> List[str]:
        out, cur = [], ""
        for word in text.split():
            if len(cur) + len(word) + 1 > width:
                out.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        if cur:
            out.append(cur)
        return out
