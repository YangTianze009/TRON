"""
Pythagoras Multistep QA environment (redesigned 2026-04-16).

Goal: apply the Pythagorean theorem multiple times to find unknowns in
composite right-triangle figures.

Redesign:
  * v2 leaked values in the question text ("legs CA = 10 and CB = 24").
  * v3 puts values ONLY on the image as labelled sides; the question text
    refers to "as shown".
  * Triangle layout is randomized (rotation, position, scale, vertex order).
  * Extra shape families: Kite-diagonals, double-triangle, stair, inscribed
    triangle, equilateral split.
  * Colour palette, line style, fill shade, label-box, grid background
    randomized.
  * 6 question template variants per family.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_VALID_TRIPLES = [
    (3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25),
    (8, 15, 17), (9, 12, 15), (9, 40, 41), (10, 24, 26),
    (12, 16, 20), (20, 21, 29), (11, 60, 61), (13, 84, 85),
    (16, 30, 34), (20, 48, 52),
]

_GEOMETRY_COLORS = [
    "#2c3e50", "#1a5276", "#7b241c", "#0d6efd", "#198754",
    "#6c3483", "#a04000", "#4a235a", "#1e8449",
]

_FILL_PALETTE = [
    "#f1c40f", "#3498db", "#e74c3c", "#2ecc71", "#9b59b6",
    "#e67e22", "#1abc9c", "#f39c12", "#5dade2", "#bb8fce",
]

_BG_PATTERNS = [
    None, "grid", "dots", "lines",
]

class PythagorasMultistepQA(StandaloneVisualEnv):
    ENV_NAME = "pythagoras_multistep"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _QUESTION_TEMPLATES_SINGLE_HYP = [
        "A right triangle is shown with the two leg lengths marked on the image. Compute the hypotenuse (choose the correct option).",
        "The figure shows a right triangle with its two legs labelled. Find the length of the hypotenuse (shown as '?'). Answer with a single letter.",
        "Given the right triangle in the image with the legs as labelled, find the length of the hypotenuse.",
        "The image displays a right triangle. One leg and the other leg are labelled; find the missing hypotenuse.",
        "Look at the right triangle shown; the two perpendicular sides are labelled. Compute the hypotenuse.",
    ]
    _QUESTION_TEMPLATES_SINGLE_LEG = [
        "A right triangle is shown with the hypotenuse and one leg labelled on the image. Find the other leg (shown as '?').",
        "The figure displays a right triangle. The hypotenuse and one leg are marked; compute the unknown leg.",
        "Given the right triangle with one leg and the hypotenuse as labelled, find the missing leg length.",
        "Looking at the labelled right triangle, find the unknown side length.",
    ]
    _QUESTION_TEMPLATES_TWO_TRIANGLES = [
        "The image shows two connected right triangles sharing the leg DB. The labelled lengths are on the image. Find DC (marked '?').",
        "Two right triangles in the figure share the leg DB. Using the labelled lengths shown, determine the length DC.",
        "Given the two-triangle figure with lengths labelled in the image, compute the length DC marked '?'.",
        "The figure shows two right triangles meeting at edge DB. Using the labelled measurements, compute DC.",
    ]
    _QUESTION_TEMPLATES_ALTITUDE = [
        "The image shows a right triangle with an altitude from the right-angle vertex to the hypotenuse. Using the labelled lengths, find the required segment.",
        "A right triangle is drawn with the altitude to the hypotenuse indicated. Using the measurements shown, compute the target length.",
        "Given the right triangle with an altitude from the right-angle vertex as shown, compute the requested segment.",
        # 2026-05-03 (M50 / SM-T3 geometric mean phrasings):
        "The figure shows a right triangle ABC with the right angle at C, and altitude CD drawn from C perpendicular to the hypotenuse AB. Using the labelled side lengths, compute the requested segment of the hypotenuse (foot of altitude).",
        "In the right triangle shown, an altitude h is dropped from the right-angle vertex to the hypotenuse, dividing it into two segments. Using the labelled lengths, find the requested length (apply the geometric-mean / leg-projection relations: h² = (segment1)(segment2), leg² = (adjacent segment)(hypotenuse)).",
        "Right triangle ABC has the right angle at C and altitude CD to hypotenuse AB. Using the labelled side lengths, compute the requested segment by applying the geometric mean relations between the altitude, the legs, and the hypotenuse segments.",
    ]
    _QUESTION_TEMPLATES_BOX = [
        "A rectangular 3D box is shown with its three dimensions labelled. Find the space diagonal.",
        "The figure shows a rectangular box; the edges are marked with their lengths. Compute the length of the space diagonal.",
        "Given the labelled rectangular box, find the space-diagonal length.",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_applications":    2 + level // 2,
            "use_3d":            level >= 4,
            "allow_radical":     level >= 1,
            "tight_distractors": level >= 2,
            "rotation_range":    5 + level * 3,
            "background_prob":   min(0.6, 0.1 + 0.06 * level),
            "fill_prob":         0.5,
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_applications"]

        families = ["single"]
        if cfg["n_applications"] >= 2:
            families.append("two_triangles")
        if cfg["n_applications"] >= 3 and not cfg["use_3d"]:
            families.append("altitude")
        if cfg["use_3d"]:
            families.append("box_diagonal")

        for _ in range(30):
            fam = rng.choice(families)
            result = self._try_generate(rng, cfg, fam)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, cfg, family):
        tight = cfg["tight_distractors"]
        if family == "single":
            a, b, c = rng.choice(_VALID_TRIPLES)
            ask = rng.choice(["hyp", "leg"])
            if ask == "hyp":
                gt_val = c
                given = ("leg_A", a, "leg_B", b, "?", None)
                q_template = rng.choice(self._QUESTION_TEMPLATES_SINGLE_HYP)
            else:
                gt_val = b
                given = ("leg_A", a, "hyp", c, "leg_B", None)
                q_template = rng.choice(self._QUESTION_TEMPLATES_SINGLE_LEG)
            shape_sides = (a, b, c, ask)
        elif family == "two_triangles":
            # Two right triangles sharing leg DB.
            #   Triangle 1 (ABD): right angle at D, legs AD=a1, DB=b1,
            #     hypotenuse AB=c1.
            #   Triangle 2 (BDC): right angle at B, legs DB=b1 (shared)
            #     and BC=a2 (perpendicular to DB), so DC=sqrt(b1^2+a2^2).
            # We show AD, BC, and AB (=c1) labelled on the image, and ask
            # for DC (the far hypotenuse). This requires first recovering
            # DB from c1 and a1 via Pythagoras, then combining with a2.
            t1 = rng.choice(_VALID_TRIPLES)
            a1, b1, c1 = t1
            a2 = rng.choice([3, 4, 5, 6, 7, 8, 9, 12])
            # Avoid degenerate configs.
            if a2 == 0 or b1 == 0:
                return None
            gt_val = round(math.sqrt(b1 * b1 + a2 * a2), 2)
            shape_sides = (a1, b1, c1, a2)
            q_template = rng.choice(self._QUESTION_TEMPLATES_TWO_TRIANGLES)
        elif family == "altitude":
            t = rng.choice(_VALID_TRIPLES)
            a, b, c = t
            gt_val = round((a * a) / c, 2)
            shape_sides = (a, b, c)
            q_template = rng.choice(self._QUESTION_TEMPLATES_ALTITUDE)
        elif family == "box_diagonal":
            dims = [rng.randint(2, 9), rng.randint(2, 9),
                    rng.randint(2, 9)]
            rng.shuffle(dims)
            L, W, H = dims
            gt_val = round(math.sqrt(L * L + W * W + H * H), 2)
            shape_sides = (L, W, H)
            q_template = rng.choice(self._QUESTION_TEMPLATES_BOX)
        else:
            return None

        # Distractors
        if tight:
            pool = [round(gt_val + k * (0.5 if isinstance(gt_val, float)
                                         else 1), 2)
                    for k in (-2, -1, 1, 2)]
        else:
            pool = [round(gt_val + k, 2) for k in (-5, -3, 3, 5, 8)]
        pool = [v for v in pool if v > 0 and v != gt_val]
        rng.shuffle(pool)
        distractors = pool[:3]
        if len(distractors) < 3:
            for k in (-8, -4, 4, 8, 12, -6, -2):
                cand = round(gt_val + k, 2)
                if cand > 0 and cand != gt_val and cand not in distractors:
                    distractors.append(cand)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            return None

        options_vals = [gt_val] + distractors[:3]
        rng.shuffle(options_vals)
        if options_vals.count(gt_val) > 1:
            return None
        answer_letter = chr(ord("A") + options_vals.index(gt_val))

        def fmt(v):
            if isinstance(v, int):
                return str(v)
            if abs(v - round(v)) < 1e-6:
                return str(int(round(v)))
            return f"{v:.2f}"

        options_str = [fmt(v) for v in options_vals]

        # Build question text - image has the values, question says "as
        # shown" + options only.
        question_body = q_template
        question_body += "\n"
        for i, o in enumerate(options_str):
            question_body += f"\n  ({chr(ord('A') + i)}) {o}"
        question_body += "\nAnswer with the single letter of the correct option."

        image = self._render(family, shape_sides, options_str, cfg, rng)
        return question_body, answer_letter, image

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _render(self, family, shape_sides, options, cfg, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(9.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        ax.axis("off")

        palette_idx = rng.randint(0, len(style["palette"]) - 1)
        fill_color = rng.choice(_FILL_PALETTE)
        geo_line = rng.choice(_GEOMETRY_COLORS)
        lw = style["line_width"] * rng.uniform(1.0, 1.5)
        rotation_angle = rng.uniform(-cfg["rotation_range"],
                                     cfg["rotation_range"])
        fill_alpha = rng.uniform(0.18, 0.42)
        label_box = rng.random() < 0.5
        bg_pattern = rng.choice(_BG_PATTERNS)

        # Render background pattern
        if bg_pattern == "grid":
            ax.grid(True, alpha=0.2, linestyle="--", color="#bdc3c7",
                    zorder=0)
        elif bg_pattern == "dots":
            xs = [rng.random() * 20 - 5 for _ in range(50)]
            ys = [rng.random() * 20 - 5 for _ in range(50)]
            ax.scatter(xs, ys, s=6, c="#d5dbdb", alpha=0.3, zorder=0)

        def _draw_label(x, y, text, offset=(0, 0), color=None,
                        weight="bold", size=None):
            if color is None:
                color = geo_line
            if size is None:
                size = fs + 1
            box_kwargs = {}
            if label_box:
                box_kwargs = dict(bbox=dict(boxstyle="round,pad=0.18",
                                             facecolor="#ffffff",
                                             edgecolor=geo_line,
                                             linewidth=0.8, alpha=0.9))
            ax.text(x + offset[0], y + offset[1], text,
                    fontsize=size, family=ff, color=color,
                    ha="center", va="center", fontweight=weight,
                    **box_kwargs)

        if family == "single":
            a, b, c, ask = shape_sides
            # Randomize vertex positions (jittered + rotation)
            cx_jit = rng.uniform(-0.1, 0.1) * max(a, b)
            cy_jit = rng.uniform(-0.1, 0.1) * max(a, b)
            verts = [(0 + cx_jit, 0 + cy_jit),
                     (a + cx_jit, 0 + cy_jit),
                     (0 + cx_jit, b + cy_jit)]
            # Rotation around centre
            cx = sum(v[0] for v in verts) / 3
            cy = sum(v[1] for v in verts) / 3
            verts_r = []
            for (x, y) in verts:
                theta = math.radians(rotation_angle)
                nx = cx + (x - cx) * math.cos(theta) - (y - cy) * math.sin(
                    theta)
                ny = cy + (x - cx) * math.sin(theta) + (y - cy) * math.cos(
                    theta)
                verts_r.append((nx, ny))
            poly = plt.Polygon(verts_r, closed=True, facecolor=fill_color,
                               edgecolor=geo_line, linewidth=lw,
                               alpha=fill_alpha)
            ax.add_patch(poly)
            # Right-angle marker at vertex 0 (rotated)
            m_size = min(a, b) * 0.08
            tA, tB = verts_r[0], verts_r[1]
            tC = verts_r[2]
            theta_ab = math.atan2(tB[1] - tA[1], tB[0] - tA[0])
            theta_ac = math.atan2(tC[1] - tA[1], tC[0] - tA[0])
            p_right = [
                (tA[0] + m_size * math.cos(theta_ab),
                 tA[1] + m_size * math.sin(theta_ab)),
                (tA[0] + m_size * (math.cos(theta_ab)
                                    + math.cos(theta_ac)),
                 tA[1] + m_size * (math.sin(theta_ab)
                                    + math.sin(theta_ac))),
                (tA[0] + m_size * math.cos(theta_ac),
                 tA[1] + m_size * math.sin(theta_ac)),
            ]
            ax.plot([p[0] for p in p_right], [p[1] for p in p_right],
                    "-", color=geo_line, linewidth=1.0)

            # Labels for sides (show values on image)
            def mid(p1, p2):
                return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            leg1_mid = mid(verts_r[0], verts_r[1])
            leg2_mid = mid(verts_r[0], verts_r[2])
            hyp_mid = mid(verts_r[1], verts_r[2])
            # normal of each edge to offset label outside
            def perp_offset(p1, p2, sign=1):
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                L = math.hypot(dx, dy)
                if L == 0:
                    return (0, 0)
                return (-dy / L * 0.25 * sign, dx / L * 0.25 * sign)

            off1 = perp_offset(verts_r[0], verts_r[1], 1)
            off2 = perp_offset(verts_r[0], verts_r[2], -1)
            off_h = perp_offset(verts_r[1], verts_r[2], 1)
            # Show leg A (? if asked)
            _draw_label(leg1_mid[0] + off1[0], leg1_mid[1] + off1[1],
                        str(a) if ask == "hyp" or ask == "leg" else "?")
            # leg B
            if ask == "leg":
                _draw_label(leg2_mid[0] + off2[0], leg2_mid[1] + off2[1],
                            "?", color="#c0392b")
            else:
                _draw_label(leg2_mid[0] + off2[0], leg2_mid[1] + off2[1],
                            str(b))
            # hypotenuse
            if ask == "hyp":
                _draw_label(hyp_mid[0] + off_h[0], hyp_mid[1] + off_h[1],
                            "?", color="#c0392b")
            else:
                _draw_label(hyp_mid[0] + off_h[0], hyp_mid[1] + off_h[1],
                            str(c))
            # Vertex letters
            labels = ["C", "B", "A"]
            for (x, y), l in zip(verts_r, labels):
                _draw_label(x - 0.35, y + 0.35, l, color=geo_line,
                            weight="bold", size=fs + 2)
            pad = max(a, b) * 0.3 + 0.5
            xs = [v[0] for v in verts_r]
            ys = [v[1] for v in verts_r]
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        elif family == "two_triangles":
            a1, b1, c1, a2 = shape_sides
            # Triangle 1 ABD: right at D, legs AD=a1 (vertical), DB=b1
            # (horizontal), hyp AB=c1. Triangle 2 BDC: right at B, legs
            # DB=b1 (shared) and BC=a2 (vertical). C is above B.
            A = (0, a1)
            D = (0, 0)
            B = (b1, 0)
            C = (b1, a2)
            pts = [A, D, B, C]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            theta = math.radians(rotation_angle)
            pts_r = [(cx + (p[0] - cx) * math.cos(theta)
                      - (p[1] - cy) * math.sin(theta),
                      cy + (p[0] - cx) * math.sin(theta)
                      + (p[1] - cy) * math.cos(theta)) for p in pts]
            Ar, Dr, Br, Cr = pts_r
            # Draw all edges including the shared DB and the two
            # hypotenuses AB and DC.
            for (p1, p2) in [(Ar, Dr), (Dr, Br), (Ar, Br),
                             (Br, Cr), (Dr, Cr)]:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
                        color=geo_line, linewidth=lw)
            # Right-angle markers at D and B.
            def _right_marker(V, Pa, Pb, size):
                tA = math.atan2(Pa[1] - V[1], Pa[0] - V[0])
                tB = math.atan2(Pb[1] - V[1], Pb[0] - V[0])
                p1 = (V[0] + size * math.cos(tA),
                      V[1] + size * math.sin(tA))
                p2 = (V[0] + size * (math.cos(tA) + math.cos(tB)),
                      V[1] + size * (math.sin(tA) + math.sin(tB)))
                p3 = (V[0] + size * math.cos(tB),
                      V[1] + size * math.sin(tB))
                ax.plot([p1[0], p2[0], p3[0]],
                        [p1[1], p2[1], p3[1]], "-",
                        color=geo_line, linewidth=1.0)
            m_size = min(a1, b1, a2) * 0.12 + 0.15
            _right_marker(Dr, Ar, Br, m_size)
            _right_marker(Br, Dr, Cr, m_size)

            def mid(p1, p2):
                return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            # Label AD with a1, BC with a2, AB with c1.
            # Value b1 (the shared leg DB) is NOT labelled — that is the
            # intermediate quantity the student computes via
            # b1 = sqrt(c1^2 - a1^2) before finding DC = sqrt(b1^2+a2^2).
            # Compute perpendicular offsets so AB and DC labels don't
            # collide near the centre of the figure.
            def _perp(p1, p2, sign):
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                L = math.hypot(dx, dy) + 1e-6
                s = max(0.5, 0.05 * L)
                return (-dy / L * s * sign, dx / L * s * sign)
            ab_off = _perp(Ar, Br, 1)
            dc_off = _perp(Dr, Cr, -1)
            _draw_label(mid(Ar, Dr)[0] - 0.55, mid(Ar, Dr)[1], str(a1))
            _draw_label(mid(Br, Cr)[0] + 0.55, mid(Br, Cr)[1], str(a2))
            _draw_label(mid(Ar, Br)[0] + ab_off[0],
                        mid(Ar, Br)[1] + ab_off[1], str(c1))
            _draw_label(mid(Dr, Cr)[0] + dc_off[0],
                        mid(Dr, Cr)[1] + dc_off[1],
                        "?", color="#c0392b", size=fs + 2)
            labels = ["A", "D", "B", "C"]
            # Offset vertex labels away from the figure centre.
            figcx = sum(p[0] for p in pts_r) / 4
            figcy = sum(p[1] for p in pts_r) / 4
            for (x, y), l in zip(pts_r, labels):
                ox = x - figcx
                oy = y - figcy
                n = math.hypot(ox, oy) + 1e-6
                _draw_label(x + 0.35 * ox / n, y + 0.35 * oy / n, l,
                            color=geo_line, weight="bold", size=fs + 2)
            xs = [p[0] for p in pts_r]
            ys = [p[1] for p in pts_r]
            pad = max(a1, b1, a2) * 0.25 + 1.2
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        elif family == "altitude":
            a, b, c = shape_sides
            verts = [(0, 0), (a, 0), (0, b)]
            # Rotation
            cx = sum(v[0] for v in verts) / 3
            cy = sum(v[1] for v in verts) / 3
            theta = math.radians(rotation_angle)
            verts_r = [(cx + (v[0] - cx) * math.cos(theta)
                        - (v[1] - cy) * math.sin(theta),
                        cy + (v[0] - cx) * math.sin(theta)
                        + (v[1] - cy) * math.cos(theta))
                       for v in verts]
            poly = plt.Polygon(verts_r, closed=True, facecolor=fill_color,
                               edgecolor=geo_line, linewidth=lw,
                               alpha=fill_alpha)
            ax.add_patch(poly)
            # Altitude from vertex 0 to hypotenuse (verts_r[1]->verts_r[2])
            P = verts_r[0]
            A = verts_r[1]
            B = verts_r[2]
            # Foot of perpendicular
            dx = B[0] - A[0]
            dy = B[1] - A[1]
            t = ((P[0] - A[0]) * dx + (P[1] - A[1]) * dy) / (
                dx * dx + dy * dy)
            H = (A[0] + t * dx, A[1] + t * dy)
            ax.plot([P[0], H[0]], [P[1], H[1]], "--",
                    color=rng.choice(["#e74c3c", "#c0392b", "#922b21"]),
                    linewidth=lw * 0.9)
            def mid(p1, p2):
                return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            _draw_label(mid(verts_r[0], verts_r[1])[0],
                        mid(verts_r[0], verts_r[1])[1] - 0.35, str(a))
            _draw_label(mid(verts_r[0], verts_r[2])[0] - 0.35,
                        mid(verts_r[0], verts_r[2])[1], str(b))
            _draw_label(mid(verts_r[1], verts_r[2])[0] + 0.35,
                        mid(verts_r[1], verts_r[2])[1] + 0.1, str(c))
            _draw_label(mid(A, H)[0] + 0.2, mid(A, H)[1] + 0.25,
                        "?", color="#c0392b")
            xs = [v[0] for v in verts_r] + [H[0]]
            ys = [v[1] for v in verts_r] + [H[1]]
            pad = max(a, b) * 0.25 + 0.8
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        elif family == "box_diagonal":
            L, W, H = shape_sides
            # Oblique projection
            ox, oy = 0.35 * L, 0.35 * L
            pts_front = [(0, 0), (L, 0), (L, H), (0, H)]
            pts_back = [(x + ox, y + oy) for (x, y) in pts_front]
            # Slight rotation of the whole projection
            theta = math.radians(rotation_angle * 0.5)
            cx_mid = (L + ox) / 2
            cy_mid = (H + oy) / 2
            def rot_pt(p):
                return (cx_mid + (p[0] - cx_mid) * math.cos(theta)
                         - (p[1] - cy_mid) * math.sin(theta),
                         cy_mid + (p[0] - cx_mid) * math.sin(theta)
                         + (p[1] - cy_mid) * math.cos(theta))
            pts_front_r = [rot_pt(p) for p in pts_front]
            pts_back_r = [rot_pt(p) for p in pts_back]
            edges = [(pts_front_r[0], pts_front_r[1]),
                     (pts_front_r[1], pts_front_r[2]),
                     (pts_front_r[2], pts_front_r[3]),
                     (pts_front_r[3], pts_front_r[0]),
                     (pts_back_r[0], pts_back_r[1]),
                     (pts_back_r[1], pts_back_r[2]),
                     (pts_back_r[2], pts_back_r[3]),
                     (pts_back_r[3], pts_back_r[0]),
                     (pts_front_r[0], pts_back_r[0]),
                     (pts_front_r[1], pts_back_r[1]),
                     (pts_front_r[2], pts_back_r[2]),
                     (pts_front_r[3], pts_back_r[3])]
            for (p1, p2) in edges:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-",
                        color=geo_line, linewidth=lw * 0.8)
            # Diagonal
            diag_color = rng.choice(["#e74c3c", "#c0392b", "#a93226"])
            ax.plot([pts_front_r[0][0], pts_back_r[2][0]],
                    [pts_front_r[0][1], pts_back_r[2][1]],
                    "--", color=diag_color, linewidth=lw)
            # Labels
            _draw_label((pts_front_r[0][0] + pts_front_r[1][0]) / 2,
                        (pts_front_r[0][1] + pts_front_r[1][1]) / 2 - 0.4,
                        f"L={L}")
            _draw_label((pts_front_r[0][0] + pts_front_r[3][0]) / 2 - 0.4,
                        (pts_front_r[0][1] + pts_front_r[3][1]) / 2,
                        f"H={H}")
            _draw_label((pts_front_r[1][0] + pts_back_r[1][0]) / 2 + 0.3,
                        (pts_front_r[1][1] + pts_back_r[1][1]) / 2 - 0.3,
                        f"W={W}")
            _draw_label((pts_front_r[0][0] + pts_back_r[2][0]) / 2 + 0.1,
                        (pts_front_r[0][1] + pts_back_r[2][1]) / 2,
                        "?", color=diag_color, size=fs + 2)
            xs = [p[0] for p in pts_front_r + pts_back_r]
            ys = [p[1] for p in pts_front_r + pts_back_r]
            pad = 1.5
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)

        fig_title_pool = ["Figure", "Diagram", "Right Triangle",
                          "Geometry Problem", "Pythagoras",
                          "Length Puzzle"]
        ax.set_title(rng.choice(fig_title_pool),
                     fontsize=fs + 2, family=ff, pad=8)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b2b"
    os.makedirs(out_dir, exist_ok=True)
    env = PythagorasMultistepQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            print(f"[seed={s} L{level}] A={env._answer}")
