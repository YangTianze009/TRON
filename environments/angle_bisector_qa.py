"""
Angle bisector QA — triangle with bisector / median / perpendicular bisector /
altitude, plus centroid, incenter, circumcenter at higher levels.

Fixes (round 2): move numeric values off the question text onto the image,
introduce structurally different L0 vs L9 question types, diversify triangle
shapes (scalene / right / isosceles / obtuse / equilateral), randomize colors
and rotations per seed, add new question operations at higher levels.
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

class AngleBisectorQA(StandaloneVisualEnv):
    ENV_NAME = "angle_bisector"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_POOL = [
        "Triangle Construction",
        "Triangle with Cevian",
        "Bisector and Median",
        "Geometry of Triangles",
        "Triangle Segment Problem",
        "Cevians in a Triangle",
    ]

    _TRIANGLE_FAMILIES = ["scalene", "right", "isosceles", "obtuse", "acute_tall"]

    # ------------------------------------------------------------------ #
    # Difficulty schedule
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesign 2026-04-17: L0 pass-rate was only 0.4 because
        # midpoint_coord requires tuple answer matching that's fragile.
        # Move tuple-answer qtypes to mid-levels and keep scalar answers for
        # L0-L1 (equal_area_half, bisector_half_angle).
        # L3=1.0 dipped because median_length on integer triangles produces
        # clean numbers the model reads directly.
        if level <= 1:
            return {
                "qtypes": ["equal_area_half", "bisector_half_angle"],
                "coord_range": (4, 8),
                "show_formula": True,
                "tri_families": ["right", "isosceles"],
            }
        if level <= 3:
            return {
                "qtypes": ["median_length", "bisector_half_angle",
                           "equal_area_half"],
                "coord_range": (4, 10),
                "show_formula": True,
                "tri_families": ["right", "isosceles", "scalene"],
            }
        if level <= 5:
            return {
                "qtypes": ["median_length", "bisector_ratio",
                           "centroid_coord"],
                "coord_range": (5, 12),
                "show_formula": False,
                "tri_families": ["scalene", "right", "acute_tall"],
            }
        if level <= 7:
            return {
                "qtypes": ["bisector_length_difference",
                           "centroid_coord", "incenter_inradius"],
                "coord_range": (6, 13),
                "show_formula": False,
                "tri_families": ["scalene", "obtuse", "acute_tall"],
            }
        return {
            "qtypes": ["sub_triangle_ratio",
                       "circumradius_from_sides",
                       "incenter_inradius"],
            "coord_range": (7, 15),
            "show_formula": False,
            "tri_families": ["scalene", "obtuse", "acute_tall"],
        }

    # ------------------------------------------------------------------ #
    # Triangle generation (varied families)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _triangle_ok(A, B, C) -> bool:
        """Reject narrow / near-degenerate triangles where side labels
        collide. The metric is aspect_ratio = min_altitude / max_edge;
        ratio > ~0.25 keeps triangles wide enough that midpoint-based
        labels don't bunch up on the same line."""
        Ax, Ay = A; Bx, By = B; Cx, Cy = C
        AB = math.hypot(Bx - Ax, By - Ay)
        BC = math.hypot(Cx - Bx, Cy - By)
        CA = math.hypot(Ax - Cx, Ay - Cy)
        if min(AB, BC, CA) < 2.0:
            return False
        area2 = abs((Bx - Ax) * (Cy - Ay) - (Cx - Ax) * (By - Ay))  # 2 * area
        if area2 < 4.0:
            return False
        max_edge = max(AB, BC, CA)
        min_alt = area2 / max_edge  # shortest altitude
        # aspect ratio: min_alt / max_edge
        return (min_alt / max_edge) > 0.25

    def _make_triangle(self, rng: random.Random, cfg: Dict):
        lo, hi = cfg["coord_range"]
        fam = rng.choice(cfg["tri_families"])
        for _ in range(30):
            if fam == "right":
                # A at origin, B on +x, C on +y
                Bx = rng.randint(lo, hi)
                Cy = rng.randint(lo, hi)
                Ax, Ay, By, Cx = 0, 0, 0, 0
                A, B, C = (Ax, Ay), (Bx, By), (Cx, Cy)
                if self._triangle_ok(A, B, C):
                    return A, B, C, fam
            elif fam == "isosceles":
                Bx = rng.randint(lo, hi)
                Cx = Bx / 2.0
                Cy = rng.randint(lo, hi)
                A, B, C = (0, 0), (Bx, 0), (Cx, Cy)
                if self._triangle_ok(A, B, C):
                    return A, B, C, fam
            elif fam == "scalene":
                Bx = rng.randint(lo, hi)
                Cx = rng.randint(1, max(2, Bx - 1))
                Cy = rng.randint(lo, hi)
                A, B, C = (0, 0), (Bx, 0), (Cx, Cy)
                if self._triangle_ok(A, B, C):
                    return A, B, C, fam
            elif fam == "obtuse":
                Bx = rng.randint(lo, hi)
                Cx = -rng.randint(2, max(3, lo))
                Cy = rng.randint(max(2, lo - 2), hi)
                A, B, C = (0, 0), (Bx, 0), (Cx, Cy)
                if self._triangle_ok(A, B, C):
                    return A, B, C, fam
            elif fam == "acute_tall":
                # relaxed: was very narrow (Cy >= hi). Scale back so the
                # triangle isn't pathologically thin.
                Bx = rng.randint(lo, hi)
                Cx = rng.randint(max(1, lo // 2), max(2, Bx - 1))
                Cy = rng.randint(max(lo, 4), hi + 2)
                A, B, C = (0, 0), (Bx, 0), (Cx, Cy)
                if self._triangle_ok(A, B, C):
                    return A, B, C, fam
        # fallback: safe scalene with known-good aspect
        return (0, 0), (8, 0), (3, 5), "scalene"

    # ------------------------------------------------------------------ #
    # Problem generator
    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 601)

        for _ in range(25):
            try:
                r = self._try_problem(rng, cfg, level)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _try_problem(self, rng: random.Random, cfg: Dict, level: int):
        qtype = parameter_choice = rng.choice(cfg["qtypes"])
        A, B, C, fam = self._make_triangle(rng, cfg)
        Ax, Ay = A
        Bx, By = B
        Cx, Cy = C
        AB = math.hypot(Bx - Ax, By - Ay)
        BC = math.hypot(Cx - Bx, Cy - By)
        CA = math.hypot(Ax - Cx, Ay - Cy)
        # Degenerate guard
        if AB < 1.5 or BC < 1.5 or CA < 1.5:
            return None
        area = abs((Bx - Ax) * (Cy - Ay) - (Cx - Ax) * (By - Ay)) / 2.0
        if area < 1.0:
            return None

        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        line_color = rng.choice(["#2c3e50", "#1a5276", "#7b241c", "#0d6efd",
                                 "#198754", "#5d4037", "#6a1b9a"])
        tri_fill = rng.choice(palette[:4])

        # Label permutation: sometimes swap B/C so vertices aren't always in
        # the same orientation across seeds.
        swap = rng.random() < 0.45
        if swap:
            B, C = C, B
            Bx, By = B
            Cx, Cy = C
            AB = math.hypot(Bx - Ax, By - Ay)
            BC = math.hypot(Cx - Bx, Cy - By)
            CA = math.hypot(Ax - Cx, Ay - Cy)

        # ---- Decide answer + formula variants per qtype ----
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.4 * sc, 5.4 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        tri = plt.Polygon([(Ax, Ay), (Bx, By), (Cx, Cy)], fill=True,
                          fc=tri_fill, ec=line_color,
                          linewidth=style["line_width"] + 0.5,
                          alpha=style["geo_fill_alpha"])
        ax.add_patch(tri)

        fs = style["font_size_base"] + 1
        # Vertex labels — dynamic offsets based on rough centroid direction
        centroid = ((Ax + Bx + Cx) / 3.0, (Ay + By + Cy) / 3.0)
        def _out_offset(px, py, dist=0.55):
            dx, dy = px - centroid[0], py - centroid[1]
            n = math.hypot(dx, dy) + 1e-9
            return (dx / n * dist, dy / n * dist)
        # Record the outward offset per vertex so the coordinate-label code
        # (used by midpoint_coord, median_length, centroid_coord) can emit
        # a single combined label "A(0,0)" instead of stacking two labels
        # that collide on narrow triangles.
        vertex_out = {}
        for (px, py), lbl in [((Ax, Ay), "A"), ((Bx, By), "B"), ((Cx, Cy), "C")]:
            ox, oy = _out_offset(px, py, dist=0.7)
            vertex_out[lbl] = (ox, oy)
            ax.plot(px, py, "o", color=line_color, markersize=5)
            ax.text(px + ox, py + oy, lbl, fontsize=fs + 2, fontweight="bold",
                    color=line_color, ha="center", va="center")

        # Labels for side lengths on the image (the text leakage fix)
        def _label_side(p1, p2, text, color, out_scale=0.4):
            mx = (p1[0] + p2[0]) / 2.0
            my = (p1[1] + p2[1]) / 2.0
            # push label away from centroid; use perpendicular-to-edge
            # offset so labels on adjacent sides of a narrow triangle
            # don't land at near-identical y values.
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            en = math.hypot(ex, ey) + 1e-9
            # outward normal (rotate edge 90 degrees, pick the side that
            # points away from centroid).
            nx, ny = -ey / en, ex / en
            cdx, cdy = mx - centroid[0], my - centroid[1]
            if nx * cdx + ny * cdy < 0:
                nx, ny = -nx, -ny
            ox, oy = nx * out_scale, ny * out_scale
            ax.text(mx + ox, my + oy, text, fontsize=fs - 1,
                    color=color, fontweight="bold", ha="center", va="center")

        def _label_vertex_coord(lbl, vx, vy):
            """Place 'A(0,0)'-style coord labels along the outward direction
            so they don't collide with the 'A' vertex letter on narrow
            triangles."""
            ox, oy = vertex_out[lbl]
            # push coordinate text farther out than the vertex letter.
            scale = 2.2
            ax.text(vx + ox * scale, vy + oy * scale,
                    f"{lbl}({int(round(vx))},{int(round(vy))})",
                    fontsize=fs - 1, color=line_color, ha="center",
                    va="center")

        # Helper accents
        formula_txt = None
        q = None
        answer_str = None

        def _round(v, n=2):
            v = round(v, n)
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:.{n}f}"

        title_suffix = ""

        if qtype == "midpoint_coord":
            # L0: read coords A and B from image, compute midpoint
            Mx = (Ax + Bx) / 2.0
            My = (Ay + By) / 2.0
            ax.plot([Ax, Bx], [Ay, By], color=line_color, linewidth=style["line_width"] + 0.5)
            ax.plot(Mx, My, "o", color=palette[3], markersize=7)
            ax.text(Mx, My - 0.5, "M", fontsize=fs + 1, fontweight="bold",
                    color=palette[3], ha="center")
            _label_vertex_coord("A", Ax, Ay)
            _label_vertex_coord("B", Bx, By)
            if cfg["show_formula"]:
                formula_txt = "M = ((Ax+Bx)/2, (Ay+By)/2)"
            q = ("In the figure, M is the midpoint of AB. Read the coordinates "
                 "of A and B from the image and compute the coordinates of M. "
                 "Answer as (x,y) with decimals.")
            answer_str = f"({_round(Mx, 1)},{_round(My, 1)})"
            title_suffix = "Midpoint of AB"

        elif qtype == "equal_area_half":
            # L0-ish: median divides a triangle into two equal halves
            Mx = (Ax + Bx) / 2.0
            My = (Ay + By) / 2.0
            ax.plot([Cx, Mx], [Cy, My], color=palette[3], linewidth=style["line_width"] + 0.8,
                    linestyle="--")
            ax.plot(Mx, My, "o", color=palette[3], markersize=6)
            ax.text(Mx + 0.1, My - 0.5, "M", fontsize=fs, fontweight="bold",
                    color=palette[3])
            ax.text(centroid[0], centroid[1] + 0.1, f"Total area = {_round(area)}",
                    fontsize=fs, fontweight="bold", color="#7b241c",
                    ha="center")
            if cfg["show_formula"]:
                formula_txt = "A median splits the triangle into two equal-area halves."
            half = area / 2.0
            q = ("CM is the median from C to the midpoint M of AB. Use the "
                 "total area shown in the figure to find the area of one of "
                 "the two sub-triangles. Round to 2 decimals.")
            answer_str = _round(half)
            title_suffix = "Median halves the Area"

        elif qtype == "median_length":
            Mx = (Ax + Bx) / 2.0
            My = (Ay + By) / 2.0
            ax.plot([Cx, Mx], [Cy, My], color=palette[3],
                    linewidth=style["line_width"] + 0.5, linestyle="--")
            ax.plot(Mx, My, "o", color=palette[3], markersize=6)
            ax.text(Mx + 0.1, My - 0.5, "M", fontsize=fs, fontweight="bold",
                    color=palette[3])
            _label_vertex_coord("A", Ax, Ay)
            _label_vertex_coord("B", Bx, By)
            _label_vertex_coord("C", Cx, Cy)
            if cfg["show_formula"]:
                formula_txt = "CM = sqrt((Cx - Mx)^2 + (Cy - My)^2)"
            med_len = math.hypot(Cx - Mx, Cy - My)
            q = ("M is the midpoint of AB. Read the vertex coordinates from "
                 "the figure and compute the length of the median CM. Round "
                 "to 2 decimals.")
            answer_str = _round(med_len)
            title_suffix = "Median CM"

        elif qtype == "bisector_half_angle":
            # Full angle at C shown on figure; answer is half.
            cos_C = (CA**2 + BC**2 - AB**2) / (2 * CA * BC)
            cos_C = max(-1, min(1, cos_C))
            angle_C = math.degrees(math.acos(cos_C))
            if angle_C < 20 or angle_C > 170:
                return None
            Dx = (BC * Ax + CA * Bx) / (CA + BC)
            Dy = (BC * Ay + CA * By) / (CA + BC)
            ax.plot([Cx, Dx], [Cy, Dy], color=palette[3],
                    linewidth=style["line_width"] + 0.8, linestyle="--")
            ax.plot(Dx, Dy, "o", color=palette[3], markersize=6)
            ax.text(Dx, Dy - 0.5, "D", fontsize=fs, fontweight="bold",
                    color=palette[3])
            # Show angle C on image
            ox, oy = _out_offset(Cx, Cy, dist=-0.7)
            ax.text(Cx + ox, Cy + oy, f"∠C = {_round(angle_C, 1)}°",
                    fontsize=fs, fontweight="bold", color="#7b241c")
            if cfg["show_formula"]:
                formula_txt = "Bisector splits ∠C in half."
            q = ("CD is the bisector of ∠C shown on the figure. Using the "
                 "value of ∠C labeled in the figure, find the measure of "
                 "∠ACD in degrees. Round to 2 decimals.")
            answer_str = _round(angle_C / 2.0)
            title_suffix = "Bisector of ∠C"

        elif qtype == "bisector_ratio":
            # Angle bisector theorem: AD / DB = CA / CB
            Dx = (BC * Ax + CA * Bx) / (CA + BC)
            Dy = (BC * Ay + CA * By) / (CA + BC)
            AD = math.hypot(Dx - Ax, Dy - Ay)
            ax.plot([Cx, Dx], [Cy, Dy], color=palette[3],
                    linewidth=style["line_width"] + 0.8, linestyle="--")
            ax.plot(Dx, Dy, "o", color=palette[3], markersize=6)
            ax.text(Dx, Dy - 0.5, "D", fontsize=fs, fontweight="bold",
                    color=palette[3])
            _label_side((Ax, Ay), (Cx, Cy), f"CA = {_round(CA)}", "#1a5276")
            _label_side((Bx, By), (Cx, Cy), f"CB = {_round(BC)}", "#1a5276")
            _label_side((Ax, Ay), (Bx, By), f"AB = {_round(AB)}", "#1a5276")
            q = ("CD is the angle bisector of ∠C. Using the side lengths "
                 "labeled in the figure, compute AD (length of the segment "
                 "from A to D on side AB). Round to 2 decimals.")
            answer_str = _round(AD)
            title_suffix = "Angle Bisector Theorem"

        elif qtype == "centroid_coord":
            Gx = (Ax + Bx + Cx) / 3.0
            Gy = (Ay + By + Cy) / 3.0
            for V, W in [((Ax, Ay), ((Bx + Cx) / 2, (By + Cy) / 2)),
                         ((Bx, By), ((Ax + Cx) / 2, (Ay + Cy) / 2)),
                         ((Cx, Cy), ((Ax + Bx) / 2, (Ay + By) / 2))]:
                ax.plot([V[0], W[0]], [V[1], W[1]], color=palette[3],
                        linewidth=style["line_width"] - 0.2, linestyle="--")
            ax.plot(Gx, Gy, "o", color=palette[2], markersize=7)
            ax.text(Gx + 0.25, Gy + 0.25, "G", fontsize=fs + 1, fontweight="bold",
                    color=palette[2])
            _label_vertex_coord("A", Ax, Ay)
            _label_vertex_coord("B", Bx, By)
            _label_vertex_coord("C", Cx, Cy)
            if cfg["show_formula"]:
                formula_txt = "G = ((Ax+Bx+Cx)/3, (Ay+By+Cy)/3)"
            # Ask x or y (random)
            which = rng.choice(["x", "y"])
            q = (f"G is the centroid of triangle ABC (intersection of the "
                 f"three medians). Using the vertex coordinates shown in the "
                 f"figure, compute the {which}-coordinate of G. Round to 2 "
                 f"decimals.")
            answer_str = _round(Gx if which == "x" else Gy)
            title_suffix = "Centroid"

        elif qtype == "bisector_length_difference":
            # AD - DB where D is the bisector foot (theorem: AD/DB = CA/CB)
            Dx = (BC * Ax + CA * Bx) / (CA + BC)
            Dy = (BC * Ay + CA * By) / (CA + BC)
            AD = math.hypot(Dx - Ax, Dy - Ay)
            DB = math.hypot(Dx - Bx, Dy - By)
            ax.plot([Cx, Dx], [Cy, Dy], color=palette[3],
                    linewidth=style["line_width"] + 0.8, linestyle="--")
            ax.plot(Dx, Dy, "o", color=palette[3], markersize=6)
            ax.text(Dx, Dy - 0.5, "D", fontsize=fs, fontweight="bold",
                    color=palette[3])
            _label_side((Ax, Ay), (Cx, Cy), f"CA = {_round(CA)}", "#1a5276")
            _label_side((Bx, By), (Cx, Cy), f"CB = {_round(BC)}", "#1a5276")
            _label_side((Ax, Ay), (Bx, By), f"AB = {_round(AB)}", "#1a5276")
            q = ("CD is the angle bisector of ∠C. Using the side lengths "
                 "labeled in the figure, compute |AD - DB|. Round to 2 "
                 "decimals.")
            answer_str = _round(abs(AD - DB))
            title_suffix = "Bisector Segment Difference"

        elif qtype == "incenter_inradius":
            # r = Area / semi-perimeter; incenter via side-weighted vertices.
            s = (AB + BC + CA) / 2.0
            r_in = area / s
            # incenter = (a*A + b*B + c*C) / (a+b+c), where a = |BC|, etc.
            a_len = BC
            b_len = CA
            c_len = AB
            Ix = (a_len * Ax + b_len * Bx + c_len * Cx) / (a_len + b_len + c_len)
            Iy = (a_len * Ay + b_len * By + c_len * Cy) / (a_len + b_len + c_len)
            circ = mpatches.Circle((Ix, Iy), r_in, fill=False,
                                   ec=palette[2], linewidth=style["line_width"] + 0.3,
                                   linestyle="--")
            ax.add_patch(circ)
            ax.plot(Ix, Iy, "o", color=palette[2], markersize=6)
            ax.text(Ix + 0.25, Iy + 0.25, "I", fontsize=fs, fontweight="bold",
                    color=palette[2])
            _label_side((Ax, Ay), (Bx, By), f"AB = {_round(AB)}", "#1a5276")
            _label_side((Bx, By), (Cx, Cy), f"BC = {_round(BC)}", "#1a5276")
            _label_side((Cx, Cy), (Ax, Ay), f"CA = {_round(CA)}", "#1a5276")
            ax.text(centroid[0], centroid[1] - 0.3,
                    f"Area = {_round(area)}", fontsize=fs, fontweight="bold",
                    color="#7b241c", ha="center")
            q = ("Using the side lengths and area labeled in the figure, "
                 "compute the inradius r of triangle ABC (r = Area / "
                 "semi-perimeter). Round to 2 decimals.")
            answer_str = _round(r_in)
            title_suffix = "Inradius"

        elif qtype == "sub_triangle_ratio":
            # Angle bisector from C divides triangle into sub-triangles
            # ACD and BCD. Their area ratio = AC:BC. Answer: area of ACD.
            Dx = (BC * Ax + CA * Bx) / (CA + BC)
            Dy = (BC * Ay + CA * By) / (CA + BC)
            ax.plot([Cx, Dx], [Cy, Dy], color=palette[3],
                    linewidth=style["line_width"] + 0.8, linestyle="--")
            ax.plot(Dx, Dy, "o", color=palette[3], markersize=6)
            ax.text(Dx, Dy - 0.5, "D", fontsize=fs, fontweight="bold",
                    color=palette[3])
            _label_side((Ax, Ay), (Cx, Cy), f"CA = {_round(CA)}", "#1a5276")
            _label_side((Bx, By), (Cx, Cy), f"CB = {_round(BC)}", "#1a5276")
            _label_side((Ax, Ay), (Bx, By), f"AB = {_round(AB)}", "#1a5276")
            ax.text(centroid[0], centroid[1] - 0.3,
                    f"Area ABC = {_round(area)}", fontsize=fs, fontweight="bold",
                    color="#7b241c", ha="center")
            # Ratio AC:(AC+BC) of total area
            sub_ACD = area * CA / (CA + BC)
            q = ("CD is the angle bisector of ∠C. Using the side lengths "
                 "and total area labeled in the figure, compute the area of "
                 "sub-triangle ACD. Round to 2 decimals.")
            answer_str = _round(sub_ACD)
            title_suffix = "Area via Bisector Ratio"

        elif qtype == "circumradius_from_sides":
            # R = (abc) / (4 * Area)
            R = (AB * BC * CA) / (4 * area)
            # draw circumscribed circle (may go off-frame)
            # circumcenter: perpendicular bisector intersection
            def _perp_bisector_pt(P1, P2):
                mx = (P1[0] + P2[0]) / 2.0
                my = (P1[1] + P2[1]) / 2.0
                dx = P2[0] - P1[0]
                dy = P2[1] - P1[1]
                return (mx, my), (-dy, dx)
            M1, d1 = _perp_bisector_pt((Ax, Ay), (Bx, By))
            M2, d2 = _perp_bisector_pt((Bx, By), (Cx, Cy))
            # Solve M1 + t*d1 = M2 + u*d2
            det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
            if abs(det) > 1e-6:
                t = ((M2[0] - M1[0]) * (-d2[1]) - (M2[1] - M1[1]) * (-d2[0])) / det
                Ox = M1[0] + t * d1[0]
                Oy = M1[1] + t * d1[1]
                circ = mpatches.Circle((Ox, Oy), R, fill=False,
                                       ec=palette[2], linewidth=style["line_width"] + 0.3,
                                       linestyle="--")
                ax.add_patch(circ)
                ax.plot(Ox, Oy, "o", color=palette[2], markersize=6)
                ax.text(Ox + 0.25, Oy + 0.25, "O", fontsize=fs, fontweight="bold",
                        color=palette[2])
            _label_side((Ax, Ay), (Bx, By), f"AB = {_round(AB)}", "#1a5276")
            _label_side((Bx, By), (Cx, Cy), f"BC = {_round(BC)}", "#1a5276")
            _label_side((Cx, Cy), (Ax, Ay), f"CA = {_round(CA)}", "#1a5276")
            ax.text(centroid[0], centroid[1] - 0.3,
                    f"Area = {_round(area)}", fontsize=fs, fontweight="bold",
                    color="#7b241c", ha="center")
            q = ("Using the three side lengths and the area labeled in the "
                 "figure, compute the circumradius R = (a·b·c) / (4·Area). "
                 "Round to 2 decimals.")
            answer_str = _round(R)
            title_suffix = "Circumradius R"

        else:
            return None

        # Finalize plot
        pad = 1.8
        all_x = [Ax, Bx, Cx]
        all_y = [Ay, By, Cy]
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_aspect("equal")
        ax.axis("off")

        title = rng.choice(self._TITLE_POOL)
        if title_suffix:
            title = f"{title} — {title_suffix}"
        ax.set_title(title, fontsize=fs + 2, fontweight="bold",
                     family=style["font_family"])
        if formula_txt:
            ax.text(0.02, 0.04, formula_txt, transform=ax.transAxes,
                    fontsize=fs - 1, color="#555", style="italic",
                    ha="left", va="bottom",
                    bbox=dict(facecolor="#f7f7f7", edgecolor="#ccc",
                              boxstyle="round,pad=0.2", alpha=0.85))

        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return q, answer_str, img
