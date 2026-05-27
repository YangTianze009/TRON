"""
Trapezoid advanced QA — midsegment, area, diagonal properties.
Targets: VO Area/Length, geometry problem solving.
Capabilities: V1, V2, R1, R2 (geometry formulas)

Round 2 diversity + difficulty fix (2026-04-16):
- Level-gated qtypes (L0 simplest, L9 hardest, structurally distinct)
- Sub-RNG seeded with level to avoid L0==L9 on same seed
- No text leakage: values appear only on image (labels); question says "as shown"
- Trapezoid variant pool: isosceles, right, generic (+ orientation rotations)
- Rich per-seed style jitter, 4+ question templates per qtype
"""
import random, math
from typing import Dict, Optional, Tuple, List
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

# ---------------------------------------------------------------------- #
# Question template pools per qtype (4+ variants each)
# ---------------------------------------------------------------------- #
_TMPL = {
    "midsegment": [
        "What is the length of the midsegment of the trapezoid shown?",
        "Find the midsegment (median) length of the trapezoid in the image.",
        "Compute the length of the line joining the midpoints of the legs.",
        "From the figure, determine the midsegment of the trapezoid.",
    ],
    "area": [
        "Find the area of the trapezoid as labeled in the image.",
        "Compute the area of the trapezoid shown.",
        "From the labeled dimensions in the figure, determine the area.",
        "What is the enclosed area of the trapezoid in the figure?",
    ],
    "height_from_area": [
        "Given the area and parallel sides marked on the figure, find the height h.",
        "The trapezoid's area and bases are labeled. What is its height h?",
        "Using the labels in the image, solve for the missing height h.",
        "From the figure's area and parallel side lengths, compute h.",
    ],
    "perimeter": [
        "Find the perimeter of the isosceles trapezoid as labeled.",
        "From the labeled sides and height, compute the perimeter. Round to 2 decimals.",
        "What is the total perimeter of the trapezoid shown? Round to 2 decimals.",
        "Compute the perimeter of the figure. Round to 2 decimals.",
    ],
    "diagonal_length": [
        "Find the length of a diagonal of the isosceles trapezoid shown. Round to 2 decimals.",
        "From the labels in the figure, compute the diagonal length. Round to 2 decimals.",
        "What is the diagonal length of the trapezoid? Round to 2 decimals.",
        "Using the labeled dimensions, compute the trapezoid's diagonal. Round to 2 decimals.",
    ],
    "angle_between_diagonals": [
        "Find the angle (deg) between the trapezoid's two diagonals. Round to 1 decimal.",
        "From the figure, compute the angle between the diagonals (deg). 1 decimal.",
        "What is the intersection angle of the diagonals in degrees? 1 decimal.",
        "Determine the acute angle between the two diagonals (deg). 1 decimal.",
    ],
    "leg_length": [
        "Find the length of a leg of the isosceles trapezoid shown. Round to 2 decimals.",
        "From the labels, compute the leg (non-parallel side) length. 2 decimals.",
        "What is the length of the slanted side (leg)? Round to 2 decimals.",
        "Using the bases and height, determine the leg length. 2 decimals.",
    ],
    "diag_intersection_ratio": [
        "The diagonals of the trapezoid shown intersect at P. Find the ratio (longer/shorter) of the two segments on one diagonal. Round to 2 decimals.",
        "For the trapezoid, diagonals intersect at P. What is the ratio of segments on a diagonal (longer/shorter)? 2 decimals.",
        "Find the ratio in which the diagonals divide each other (longer/shorter). 2 decimals.",
        "Compute the segment ratio (longer:shorter as decimal) where diagonals cross. 2 decimals.",
        # 2026-05-03 (M55 additional variants): reference style phrasings
        "In trapezoid ABCD shown, AD is parallel to BC. The diagonals intersect at point O. Compute the ratio AO/OC (the segment of diagonal AC on the AD-side, divided by the segment on the BC-side). Round to 2 decimals.",
        "In the trapezoid shown with AD || BC, point O is the intersection of diagonals AC and BD. Find the ratio of OA to OC (the segment closer to AD divided by the segment closer to BC). Round to 2 decimals.",
        "The trapezoid in the figure has parallel sides AD and BC where AD is the shorter base. Diagonal AC is divided by the intersection point O into two segments. Compute (longer segment) / (shorter segment). Round to 2 decimals.",
    ],
}

class TrapezoidAdvancedQA(StandaloneVisualEnv):
    ENV_NAME = "trapezoid_advanced"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    # ------------------------------------------------------------------ #
    # Per-level configuration (L0 structurally simpler than L9)
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        # L0-L1: only 2 simple qtypes
        if level <= 1:
            return {"qtypes": ["midsegment", "area"],
                    "shape_pool": ["iso"], "a_range": (3, 6),
                    "b_extra_range": (2, 5), "h_range": (3, 6)}
        if level <= 3:
            return {"qtypes": ["midsegment", "area", "height_from_area"],
                    "shape_pool": ["iso", "right"], "a_range": (3, 7),
                    "b_extra_range": (2, 7), "h_range": (3, 7)}
        if level <= 5:
            return {"qtypes": ["midsegment", "area", "height_from_area",
                               "perimeter", "leg_length"],
                    "shape_pool": ["iso", "right", "generic"],
                    "a_range": (3, 8), "b_extra_range": (2, 9),
                    "h_range": (3, 8)}
        if level <= 7:
            return {"qtypes": ["area", "height_from_area", "perimeter",
                               "leg_length", "diagonal_length"],
                    "shape_pool": ["iso", "right", "generic"],
                    "a_range": (4, 9), "b_extra_range": (3, 10),
                    "h_range": (4, 9)}
        # L8-L9: hardest — adds 2 new operations not at L0
        return {"qtypes": ["perimeter", "diagonal_length",
                           "angle_between_diagonals",
                           "diag_intersection_ratio", "leg_length"],
                "shape_pool": ["iso", "right", "generic"],
                "a_range": (4, 10), "b_extra_range": (3, 12),
                "h_range": (4, 10)}

    # ------------------------------------------------------------------ #

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Level-dependent sub-RNG (prime 991 unique to this env)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        forced_qtype = parameter.get("problem_type")

        # Per-seed geometric parameters
        a_lo, a_hi = cfg["a_range"]
        be_lo, be_hi = cfg["b_extra_range"]
        h_lo, h_hi = cfg["h_range"]

        # Qtypes that strictly need an isosceles shape
        iso_only_qtypes = {"perimeter", "leg_length", "diagonal_length",
                            "angle_between_diagonals",
                            "diag_intersection_ratio"}

        for _ in range(30):
            qtype = forced_qtype or sub_rng.choice(cfg["qtypes"])
            if qtype in iso_only_qtypes:
                shape_kind = "iso"
            else:
                shape_kind = sub_rng.choice(cfg["shape_pool"])
            result = self._try_generate(sub_rng, rng, cfg, qtype,
                                         shape_kind, a_lo, a_hi, be_lo,
                                         be_hi, h_lo, h_hi, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #

    def _try_generate(self, sub_rng, rng, cfg, question_type, shape_kind,
                      a_lo, a_hi, be_lo, be_hi, h_lo, h_hi, level
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        a = sub_rng.randint(a_lo, a_hi)
        b = sub_rng.randint(a + be_lo, a + be_hi)
        h = sub_rng.randint(h_lo, h_hi)

        # Shape determines vertex layout; in all cases, top side has length a,
        # bottom side length b. Left/right horizontal offsets differ.
        if shape_kind == "iso":
            left_off = (b - a) / 2.0
            right_off = (b - a) / 2.0
        elif shape_kind == "right":
            # Right trapezoid: left leg perpendicular (aligned with y-axis)
            left_off = 0.0
            right_off = b - a
        else:  # generic: asymmetric offsets
            # Ensure both offsets positive and sum to b-a
            total_off = b - a
            if total_off < 2:
                left_off = total_off / 2.0
                right_off = total_off / 2.0
            else:
                left_off = sub_rng.randint(1, max(1, total_off - 1))
                right_off = total_off - left_off

        # Vertices (CCW starting bottom-left)
        # Bottom-left=(0,0), bottom-right=(b,0), top-right=(left_off+a,h), top-left=(left_off,h)
        V_BL = (0.0, 0.0)
        V_BR = (b, 0.0)
        V_TR = (left_off + a, float(h))
        V_TL = (left_off, float(h))
        vertices = [V_BL, V_BR, V_TR, V_TL]

        # Geometry quantities
        left_leg = math.hypot(left_off, h)
        right_leg = math.hypot(right_off, h)
        area_val = (a + b) * h / 2.0
        mid_len = (a + b) / 2.0
        perimeter = a + b + left_leg + right_leg
        # Diagonal from BL (0,0) to TR (left_off+a, h)
        diag_BL_TR = math.hypot(left_off + a, h)
        # Diagonal from BR (b, 0) to TL (left_off, h)
        diag_BR_TL = math.hypot(b - left_off, h)

        # Build Q/A
        if question_type == "midsegment":
            answer_val = mid_len
            label_set = {"a", "b"}
        elif question_type == "area":
            answer_val = area_val
            label_set = {"a", "b", "h"}
        elif question_type == "height_from_area":
            answer_val = float(h)
            label_set = {"a", "b", "area"}
        elif question_type == "perimeter":
            if shape_kind != "iso":
                return None  # only meaningful labels for iso
            answer_val = perimeter
            label_set = {"a", "b", "h"}
        elif question_type == "leg_length":
            if shape_kind != "iso":
                return None
            answer_val = left_leg
            label_set = {"a", "b", "h"}
        elif question_type == "diagonal_length":
            if shape_kind != "iso":
                return None
            answer_val = diag_BL_TR
            label_set = {"a", "b", "h"}
        elif question_type == "angle_between_diagonals":
            if shape_kind != "iso":
                return None
            # Vectors from intersection: use vectors from corners
            # Vector BL->TR and BR->TL
            v1 = (left_off + a, h)
            v2 = (left_off - b, h)
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            m1 = math.hypot(*v1); m2 = math.hypot(*v2)
            if m1 == 0 or m2 == 0:
                return None
            cos_ang = max(-1.0, min(1.0, dot / (m1 * m2)))
            answer_val = math.degrees(math.acos(cos_ang))
            label_set = {"a", "b", "h"}
        elif question_type == "diag_intersection_ratio":
            if shape_kind != "iso":
                return None
            # For any trapezoid, the diagonals divide each other in ratio a:b
            # (or b:a) so ratio longer/shorter = max(a,b)/min(a,b)
            if min(a, b) <= 0:
                return None
            answer_val = max(a, b) / float(min(a, b))
            label_set = {"a", "b", "h"}
        else:
            return None

        # Format answer
        if question_type in ("midsegment", "area"):
            answer_str = str(int(answer_val)) if answer_val == int(answer_val) \
                else str(round(answer_val, 1))
        elif question_type == "height_from_area":
            answer_str = str(int(answer_val))
        elif question_type == "angle_between_diagonals":
            answer_str = str(round(answer_val, 1))
        else:
            answer_str = str(round(answer_val, 2))

        # Question text: no numeric values leak into the question
        question = sub_rng.choice(_TMPL[question_type])

        # Render
        img = self._render(sub_rng, shape_kind, vertices, a, b, h, left_off,
                           right_off, label_set, int(area_val), question_type,
                           level)
        return question, answer_str, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, sub_rng, shape_kind, vertices, a, b, h, left_off,
                right_off, label_set, area_int, question_type, level
                ) -> Image.Image:
        style = self._random_style()
        # Per-seed jitter
        rotate_deg = sub_rng.choice([0, 0, 0, 90, 180, 270])
        fig_sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * fig_sc, 5.3 * fig_sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        ax.set_facecolor(style["bg_color"])

        # Optional rotation (rigid)
        def rot(p):
            cx, cy = b / 2.0, h / 2.0
            x, y = p[0] - cx, p[1] - cy
            th = math.radians(rotate_deg)
            rx = x * math.cos(th) - y * math.sin(th)
            ry = x * math.sin(th) + y * math.cos(th)
            return (rx + cx, ry + cy)

        vrot = [rot(p) for p in vertices]
        xs = [p[0] for p in vrot]; ys = [p[1] for p in vrot]
        V_BL, V_BR, V_TR, V_TL = vrot

        poly = plt.Polygon(vrot, facecolor=palette[0], edgecolor=palette[4],
                           linewidth=style["line_width"] + 0.5,
                           alpha=style["geo_fill_alpha"] + 0.25)
        ax.add_patch(poly)

        fs = style["font_size_base"] + 1
        fs_big = fs + 1

        def _label_edge(p1, p2, txt, color):
            mx = (p1[0] + p2[0]) / 2.0
            my = (p1[1] + p2[1]) / 2.0
            # Perpendicular outward direction
            ex = p2[0] - p1[0]; ey = p2[1] - p1[1]
            L = math.hypot(ex, ey) + 1e-9
            nx = -ey / L; ny = ex / L
            cx = sum(xs) / 4.0; cy = sum(ys) / 4.0
            if nx * (mx - cx) + ny * (my - cy) < 0:
                nx, ny = -nx, -ny
            off = 0.55
            ax.text(mx + nx * off, my + ny * off, txt,
                    ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc=style["bg_color"],
                              ec=color, alpha=0.9))

        # Label sides only if in label_set
        if "a" in label_set:
            _label_edge(V_TL, V_TR, f"a = {a}", palette[1])
        if "b" in label_set:
            _label_edge(V_BL, V_BR, f"b = {b}", palette[2])

        # Height marker (drawn as dashed segment; label only if h in set)
        if "h" in label_set:
            # Draw a vertical (pre-rotation) dashed line inside the shape
            # Use the rotated midpoint of top edge and drop to bottom edge
            pT = ((V_TL[0] + V_TR[0]) / 2.0, (V_TL[1] + V_TR[1]) / 2.0)
            pB_pre = ((vertices[0][0] + vertices[1][0]) / 2.0,
                      (vertices[0][1] + vertices[1][1]) / 2.0)
            # Perp from top to bottom base pre-rotation is always vertical
            # Use the center axis from pre-rotation then rotate
            # We'll just draw segment from pT straight toward center of bottom edge
            pB = ((V_BL[0] + V_BR[0]) / 2.0, (V_BL[1] + V_BR[1]) / 2.0)
            ax.plot([pT[0], pB[0]], [pT[1], pB[1]], color=palette[3],
                    linestyle="--", linewidth=style["line_width"])
            # Label offset: push outward perpendicular to the h-line so it
            # does not overlap the dashed line or nearby edge labels. At
            # rotation=0 (h vertical) this shifts the label sideways rather
            # than on top of the segment.
            ex = pB[0] - pT[0]; ey = pB[1] - pT[1]
            Lh = math.hypot(ex, ey) + 1e-9
            perp_x = -ey / Lh
            perp_y = ex / Lh
            off_mag = 0.9
            lx = (pT[0] + pB[0]) / 2.0 + perp_x * off_mag
            ly = (pT[1] + pB[1]) / 2.0 + perp_y * off_mag
            ax.text(lx, ly, f"h = {h}", fontsize=fs, fontweight="bold",
                    color=palette[3], ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=style["bg_color"],
                              ec=palette[3], alpha=0.85))

        # If the question gives area, show area label
        if "area" in label_set:
            cx = sum(xs) / 4.0; cy = sum(ys) / 4.0
            ax.text(cx, cy, f"Area = {area_int}", ha="center", va="center",
                    fontsize=fs_big, fontweight="bold", color=palette[5],
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff7d6",
                              ec=palette[5], alpha=0.9))

        # For diagonal/angle qtypes draw the diagonals
        if question_type in ("diagonal_length", "angle_between_diagonals",
                              "diag_intersection_ratio"):
            ax.plot([V_BL[0], V_TR[0]], [V_BL[1], V_TR[1]],
                    color=palette[6], linestyle="-",
                    linewidth=style["line_width"], alpha=0.7)
            ax.plot([V_BR[0], V_TL[0]], [V_BR[1], V_TL[1]],
                    color=palette[6], linestyle="-",
                    linewidth=style["line_width"], alpha=0.7)

        # Vertex letters (jittered label)
        vletters = ["B", "C", "D", "A"]  # match conventional labeling
        sub_rng.shuffle(vletters)
        for (vx, vy), letter in zip(vrot, vletters):
            cx = sum(xs) / 4.0; cy = sum(ys) / 4.0
            dx = vx - cx; dy = vy - cy
            n = math.hypot(dx, dy) + 1e-9
            ax.text(vx + 0.4 * dx / n, vy + 0.4 * dy / n, letter,
                    fontsize=fs - 1, color=palette[7], fontweight="bold")

        margin = max(1.5, (max(xs) - min(xs)) * 0.12)
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_aspect("equal")
        ax.axis("off")

        title_pool = ["Trapezoid", "Figure: Trapezoid", "Geometry: Trapezoid",
                      f"Trapezoid (L{level})"]
        ax.set_title(sub_rng.choice(title_pool),
                     fontsize=fs + 2, fontweight="bold")

        return self.fig_to_pil(fig, dpi=style["dpi"])
