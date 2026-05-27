"""
Cyclic Quadrilateral Advanced QA environment.

Goal: targeted fix for Plane Geometry Property.
A cyclic quadrilateral inscribed in a circle, possibly with tangents,
secants, or diameters. Problems require combining multiple circle
theorems: opposite angles sum to 180, inscribed angle = arc/2,
tangent-chord angle = arc/2, arc addition.

Difficulty schedule:
  Axis 1: n_extra_lines = level // 2       -> 0..4
  Axis 2: n_theorems_needed = 1 + level // 2 -> 1..5
  Axis 3: non_integer_angles = level >= 6

Output: single integer (degrees).
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

class CyclicQuadrilateralAdvancedQA(StandaloneVisualEnv):
    ENV_NAME = "cyclic_quadrilateral_advanced"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_VARIANTS = [
        "Cyclic quadrilateral",
        "Inscribed quadrilateral",
        "Cyclic quad. + tangent",
        "Circle + cyclic quad",
        "Circle theorem problem",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_extra_lines":       level // 2,              # 0..4
            "n_theorems":          1 + level // 2,          # 1..5
            "non_integer_angles":  level >= 6,
            "tight_distractors":   level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 359)
        self._primary_complexity_feature = cfg["n_theorems"]

        # Problem pool keyed on n_theorems
        nt = cfg["n_theorems"]
        if nt <= 1:
            ptypes = ["opposite_angles"]
        elif nt == 2:
            ptypes = ["opposite_angles", "inscribed_on_quad"]
        elif nt == 3:
            ptypes = ["inscribed_on_quad", "tangent_chord_quad"]
        elif nt == 4:
            ptypes = ["tangent_chord_quad", "diagonal_chain"]
        else:
            ptypes = ["diagonal_chain", "external_secant"]

        for _ in range(25):
            try:
                ptype = sub_rng.choice(ptypes)
                r = self._dispatch(ptype, sub_rng, cfg)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _dispatch(self, ptype, rng, cfg):
        if ptype == "opposite_angles":
            return self._opposite_angles(rng, cfg)
        if ptype == "inscribed_on_quad":
            return self._inscribed_on_quad(rng, cfg)
        if ptype == "tangent_chord_quad":
            return self._tangent_chord_quad(rng, cfg)
        if ptype == "diagonal_chain":
            return self._diagonal_chain(rng, cfg)
        if ptype == "external_secant":
            return self._external_secant(rng, cfg)
        return None

    # ------------------------------------------------------------------ #
    def _opposite_angles(self, rng, cfg):
        """Given 3 angles of cyclic quad, find 4th (opposite sum = 180)."""
        A = rng.randint(60, 120)
        B = rng.randint(60, 130)
        C = 180 - A
        # D = 180 - B
        D = 180 - B
        # Ask for D given A, B, C (three givens)
        return self._finalize(rng, cfg, answer=D, variant="opposite_angles",
                              givens={"A": A, "B": B, "C": C},
                              extras={})

    def _inscribed_on_quad(self, rng, cfg):
        """Cyclic quad + an arc given. Use inscribed angle = arc/2.
        Given arc BC = x and angle DAB = a, find angle ADC or similar."""
        # opposite angles: A + C = 180
        A = rng.randint(70, 120)
        C = 180 - A
        # arc ADB or something: introduce arc BD = given
        arc_BD = rng.randint(60, 150)
        # inscribed angle at C subtending arc BD = arc_BD / 2? depends
        # Simpler: give A, ask for the inscribed angle at point C
        # subtending arc (not containing C): relation inscribed = arc/2.
        # Angle BDC subtends arc BC. Give arc BC = arc1.
        arc1 = rng.randint(60, 160)
        ans = arc1 / 2
        if cfg["non_integer_angles"] and arc1 % 2 == 1:
            ans_val = round(ans, 1)
        else:
            ans_val = int(round(ans))
        return self._finalize(rng, cfg, answer=ans_val, variant="inscribed_on_quad",
                              givens={"A": A, "arc_BC": arc1},
                              extras={})

    def _tangent_chord_quad(self, rng, cfg):
        """Cyclic quad + tangent at one vertex. Tangent-chord angle = arc/2.
        Given tangent-chord angle and opposite vertex angle, find the arc."""
        tc = rng.randint(30, 80)
        arc = 2 * tc
        # opposite angle to A in cyclic quad = 180 - A
        A = rng.randint(70, 120)
        C = 180 - A
        # Ask for arc (given tangent-chord angle tc) → arc = 2*tc.
        ans = arc
        if cfg["non_integer_angles"] and rng.random() < 0.3:
            ans = round(arc + rng.choice([-0.5, 0.5]), 1)
        return self._finalize(rng, cfg, answer=ans, variant="tangent_chord_quad",
                              givens={"A": A, "tc_angle": tc},
                              extras={})

    def _diagonal_chain(self, rng, cfg):
        """Cyclic quad with a diagonal. Given one angle + diagonal angle,
        find another angle using inscribed angle + triangle sum."""
        # cyclic: angles at A,B,C,D with A+C=180, B+D=180.
        # diagonal BD splits ABCD into triangles ABD and BCD.
        # Given angle ABD (inscribed angle on arc AD) and angle BDC
        # (inscribed on arc BC), find angle ABC... or just use
        # triangle sum in one of the sub-triangles.
        A = rng.randint(55, 110)
        B = rng.randint(60, 120)
        C = 180 - A
        D = 180 - B
        # In triangle ABD: A + angle_ABD + angle_ADB = 180
        # Choose angle ABD = given, compute angle ADB
        given_abd = rng.randint(25, 60)
        if A + given_abd >= 170:
            return None
        ang_adb = 180 - A - given_abd
        if ang_adb <= 10:
            return None
        ans = ang_adb
        return self._finalize(rng, cfg, answer=ans, variant="diagonal_chain",
                              givens={"A": A, "ABD": given_abd},
                              extras={"B": B, "C": C, "D": D})

    def _external_secant(self, rng, cfg):
        """Cyclic quad with a tangent at A and a secant from external point
        P passing through B and a point X on the circle. Use
        tangent-secant angle = (far arc - near arc) / 2."""
        far = rng.randint(100, 160)
        near = rng.randint(20, 60)
        if far <= near:
            return None
        ans = (far - near) / 2
        if cfg["non_integer_angles"] and (far - near) % 2 == 1:
            ans = round(ans, 1)
        else:
            ans = int(round(ans))
        return self._finalize(rng, cfg, answer=ans, variant="external_secant",
                              givens={"far_arc": far, "near_arc": near},
                              extras={})

    # ------------------------------------------------------------------ #
    def _finalize(self, rng, cfg, answer, variant, givens, extras):
        if isinstance(answer, float):
            answer_str = f"{answer:.1f}"
        else:
            answer_str = str(int(round(answer)))
        # Range check
        try:
            v = float(answer_str)
            if v <= 0 or v >= 180:
                return None
        except ValueError:
            return None

        sidx = (self.seed or 0) % 16
        # Question text — numeric values are labeled on the image; do NOT
        # restate them here.
        if variant == "opposite_angles":
            _POOL = [
                "ABCD is a cyclic quadrilateral inscribed in a circle, as shown. Using the angles labeled in the figure, find angle D in degrees.",
                "Given the cyclic quadrilateral ABCD (inscribed in a circle, as shown), use the labeled angle values to find angle D in degrees.",
                "Quadrilateral ABCD is inscribed in a circle as shown. With the labeled angles in the figure, compute the measure of angle D in degrees.",
                "The figure shows cyclic quadrilateral ABCD with labeled angles. Determine angle D (in degrees).",
                "ABCD is inscribed in a circle (cyclic quadrilateral) as depicted. From the labeled angles, find m∠D in degrees.",
                "Use the labeled angles in the figure for cyclic quadrilateral ABCD to calculate angle D in degrees.",
                "Cyclic quadrilateral ABCD is shown, angles labeled. What is the measure of angle D? Give in degrees.",
                "ABCD lies inscribed in a circle (as shown). Compute angle D using the given labeled angles. Answer in degrees.",
                "Given the cyclic quadrilateral ABCD with labeled angle values, find the measure of angle D (degrees).",
                "From the inscribed quadrilateral ABCD (labels visible), determine angle D in degrees.",
                "ABCD is cyclic as in the figure. Use the labeled angles to compute m∠D in degrees.",
                "The cyclic quadrilateral ABCD has labeled angles in the figure. Find angle D (degrees).",
                "In the figure, ABCD is inscribed in a circle. Using the labeled angles, what is angle D in degrees?",
                "Find the measure of angle D in the cyclic quadrilateral ABCD shown, using the labeled angles. Answer in degrees.",
                "Cyclic quadrilateral ABCD (inscribed in a circle) has labeled angles. Compute the measure of angle D. Answer: integer degrees.",
                "Using the labeled angles on the cyclic quadrilateral ABCD, determine angle D in degrees.",
            ]
            q = _POOL[sidx]
        elif variant == "inscribed_on_quad":
            _POOL = [
                "ABCD is a cyclic quadrilateral, as shown. Using the angle and arc values labeled in the figure, find the inscribed angle BDC (an inscribed angle subtending arc BC) in degrees.",
                "Given the cyclic quadrilateral ABCD with labeled angles and arcs, find the inscribed angle BDC (subtending arc BC) in degrees.",
                "Using the labeled angle and arc values on cyclic quadrilateral ABCD, determine the measure of inscribed angle BDC (on arc BC). Answer in degrees.",
                "From the figure (cyclic ABCD with labeled angles and arcs), compute inscribed angle BDC subtending arc BC, in degrees.",
                "The cyclic quadrilateral ABCD has labeled angle/arc values. Find inscribed angle BDC (subtends arc BC). Degrees.",
                "Compute the inscribed angle BDC (subtending arc BC) of cyclic quadrilateral ABCD using the labels in the figure. Answer in degrees.",
                "Inscribed quadrilateral ABCD has labeled arcs and angles. What is angle BDC (subtending arc BC), in degrees?",
                "Find the measure of inscribed angle BDC in cyclic quadrilateral ABCD, based on the labeled angle/arc values. Degrees.",
                "Given the figure (ABCD cyclic, with labeled angles and arcs), determine the inscribed angle BDC subtending arc BC. Answer in degrees.",
                "Cyclic quadrilateral ABCD with labels shown. Compute inscribed angle BDC (arc BC). Give answer in degrees.",
                "Using the labels in the figure, find the measure of ∠BDC (inscribed on arc BC) of cyclic quadrilateral ABCD, in degrees.",
                "From cyclic ABCD with labeled arcs/angles, determine the inscribed angle BDC (subtending arc BC). Integer degrees.",
                "Compute inscribed ∠BDC for cyclic quadrilateral ABCD (arc BC), using the labeled values on the figure. Degrees.",
                "ABCD is inscribed in a circle; find inscribed angle BDC (subtending arc BC) using the figure's labels. Answer: degrees.",
                "Given cyclic ABCD with labeled angle and arc measures, calculate inscribed angle BDC (over arc BC) in degrees.",
                "The cyclic quadrilateral ABCD has its relevant arcs/angles labeled. Find inscribed angle BDC. Degrees.",
            ]
            q = _POOL[sidx]
        elif variant == "tangent_chord_quad":
            _POOL = [
                "ABCD is a cyclic quadrilateral with a tangent drawn at vertex A, as shown. Using the tangent-chord angle labeled in the figure, find the measure of arc AB in degrees.",
                "Given cyclic quadrilateral ABCD with a tangent at A (as shown) and a labeled tangent-chord angle, find the measure of arc AB in degrees.",
                "A tangent at vertex A of cyclic quadrilateral ABCD is drawn. With the labeled tangent-chord angle, compute arc AB in degrees.",
                "From the figure, cyclic ABCD has a tangent at A and a labeled tangent-chord angle. Find arc AB (degrees).",
                "Using the tangent-chord angle at A (labeled) for cyclic quadrilateral ABCD, determine the measure of arc AB. Degrees.",
                "In the figure, the tangent at A to cyclic ABCD has a labeled tangent-chord angle. What is arc AB (in degrees)?",
                "Given the labeled tangent-chord angle at A of cyclic ABCD, find arc AB. Answer in degrees.",
                "The figure shows cyclic quadrilateral ABCD with a tangent at vertex A. With the labeled tangent-chord angle, compute arc AB. Degrees.",
                "Cyclic ABCD has a tangent at A; the tangent-chord angle is labeled. Find the measure of arc AB in degrees.",
                "Using the labeled tangent-chord angle from the tangent at A in cyclic ABCD, determine arc AB (degrees).",
                "From the figure (cyclic ABCD, tangent at A, labeled angle), find the arc AB in degrees.",
                "The tangent-chord angle at A of cyclic quadrilateral ABCD is labeled. Determine the measure of arc AB. Degrees.",
                "Compute arc AB using the tangent-chord angle labeled at A for cyclic ABCD. Answer in degrees.",
                "Given cyclic ABCD with a tangent at A and a labeled tangent-chord angle, find m(arc AB) in degrees.",
                "Find the measure of arc AB in the figure: cyclic ABCD with tangent at A and labeled tangent-chord angle. Degrees.",
                "The cyclic quadrilateral ABCD has a tangent drawn at vertex A; use the labeled tangent-chord angle to find arc AB. Degrees.",
            ]
            q = _POOL[sidx]
        elif variant == "diagonal_chain":
            _POOL = [
                "ABCD is a cyclic quadrilateral with diagonal BD drawn, as shown. Using the angles labeled in the figure, find angle ADB in degrees.",
                "Given cyclic quadrilateral ABCD with diagonal BD (as shown) and labeled angles, find angle ADB in degrees.",
                "From the figure (cyclic ABCD, diagonal BD, labeled angles), determine angle ADB. Degrees.",
                "In cyclic ABCD with diagonal BD drawn, use the labeled angles to compute ∠ADB. Degrees.",
                "Compute angle ADB in cyclic quadrilateral ABCD (diagonal BD shown, labeled angles). Answer in degrees.",
                "The figure shows cyclic ABCD with diagonal BD and labeled angles. What is angle ADB, in degrees?",
                "Using the labeled angles for cyclic ABCD with diagonal BD, find m∠ADB in degrees.",
                "Given cyclic ABCD with its BD-diagonal drawn and labeled angles, determine angle ADB. Degrees.",
                "Cyclic quadrilateral ABCD with diagonal BD is shown (angles labeled). Find angle ADB, in degrees.",
                "From the labels on cyclic ABCD (BD drawn), compute ∠ADB in degrees.",
                "In the figure, cyclic ABCD has diagonal BD and labeled angles. Determine angle ADB (degrees).",
                "Given the inscribed quadrilateral ABCD with BD drawn and labeled angle values, find angle ADB. Answer in degrees.",
                "Using the labeled angles on cyclic ABCD (with diagonal BD), determine the measure of angle ADB in degrees.",
                "Find ∠ADB in the cyclic quadrilateral ABCD with diagonal BD (labels in the figure). Degrees.",
                "Cyclic ABCD with BD and labeled angles: compute angle ADB. Give in degrees.",
                "Determine angle ADB in cyclic quadrilateral ABCD (diagonal BD drawn; angles labeled). Degrees.",
            ]
            q = _POOL[sidx]
        else:  # external_secant
            _POOL = [
                "From an external point P, a tangent and a secant are drawn to a circle containing the cyclic quadrilateral ABCD, as shown. Using the arc values labeled in the figure, find the angle at P in degrees.",
                "Given an external point P with a tangent and a secant to the circle of cyclic ABCD (as shown), and labeled arcs, find ∠P in degrees.",
                "External point P sends a tangent and a secant to the circle containing cyclic ABCD. With labeled arcs, compute angle at P. Degrees.",
                "From external P, a tangent and secant cut the circle (containing cyclic ABCD). Using labeled arcs, find angle P. Degrees.",
                "Using the labeled arc measures, find the angle at external point P where tangent and secant meet the circle. Degrees.",
                "The figure shows an external point P with a tangent and secant to a circle of cyclic ABCD; labeled arcs are given. Find ∠P in degrees.",
                "Given external P with a tangent and secant to the circle containing ABCD, use the labeled arcs to determine angle P. Degrees.",
                "From P outside the circle, a tangent and a secant are drawn; arcs labeled. Compute the angle at P in degrees.",
                "External point P, tangent and secant to the circle (cyclic ABCD). With labeled arcs, determine ∠P in degrees.",
                "The labeled arc values help find the angle at external point P (tangent + secant configuration). Degrees.",
                "Using arcs labeled in the figure, compute the angle at external point P with its tangent and secant. Degrees.",
                "With tangent and secant drawn from external P to the circle (containing ABCD), and the labeled arcs, find ∠P in degrees.",
                "From the external point P's tangent-secant to the circle of cyclic ABCD, using labeled arcs, determine angle P. Degrees.",
                "Given the labeled arcs in the external tangent-secant configuration at P, calculate the measure of angle P. Degrees.",
                "The figure shows the external point P with tangent and secant; using the labeled arcs, find ∠P (in degrees).",
                "Find angle at external point P (tangent and secant to circle of ABCD) using labeled arcs. Degrees.",
            ]
            q = _POOL[sidx]
        q += " Answer with a single number (integer or one decimal)."

        img = self._render(variant, givens, extras, cfg)
        return q, answer_str, img

    # ------------------------------------------------------------------ #
    def _point_on_circle(self, cx, cy, r, angle_deg):
        rad = math.radians(angle_deg)
        return (cx + r * math.cos(rad), cy + r * math.sin(rad))

    def _render(self, variant, givens, extras, cfg):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        palette = style["palette"]
        line_color = style["geo_line_color"]
        lw = style["line_width"] + 0.3
        fs = style["font_size_base"]
        ff = style["font_family"]

        cx, cy, R = 0.0, 0.0, 3.0
        rot_off = self._rng.randint(0, 359)
        circle = plt.Circle((cx, cy), R, fill=False,
                            edgecolor=palette[0], linewidth=lw + 0.2)
        ax.add_patch(circle)
        ax.plot(cx, cy, "+", markersize=8, color=line_color)
        ax.text(cx - 0.15, cy - 0.35, "O", fontsize=fs,
                fontweight="bold", family=ff, color=line_color)

        if variant in ("opposite_angles", "inscribed_on_quad",
                       "tangent_chord_quad", "diagonal_chain"):
            # Place 4 vertices of ABCD roughly equispaced but slightly
            # jittered, then connect into quadrilateral.
            base = [rot_off + k * 90 for k in range(4)]
            base = [b + self._rng.randint(-20, 20) for b in base]
            pts = [self._point_on_circle(cx, cy, R, b) for b in base]
            labels = ["A", "B", "C", "D"]
            # draw sides
            for i in range(4):
                j = (i + 1) % 4
                ax.plot([pts[i][0], pts[j][0]],
                        [pts[i][1], pts[j][1]],
                        color=palette[2 % len(palette)], linewidth=lw)
            # Draw extra lines per config
            n_extra = cfg["n_extra_lines"]
            if variant == "tangent_chord_quad" or n_extra >= 1:
                # tangent at A: perpendicular to OA
                Ax, Ay = pts[0]
                tx, ty = -Ay + cy, Ax - cx
                nrm = math.hypot(tx, ty) + 1e-9
                tx, ty = tx / nrm, ty / nrm
                t1 = (Ax + 2.0 * tx, Ay + 2.0 * ty)
                t2 = (Ax - 2.0 * tx, Ay - 2.0 * ty)
                ax.plot([t1[0], t2[0]], [t1[1], t2[1]],
                        color=palette[4 % len(palette)], linewidth=lw,
                        linestyle="--")
                ax.text(t1[0] + 0.05, t1[1] + 0.1, "tangent",
                        fontsize=fs - 1, color=palette[4 % len(palette)])
            if variant == "diagonal_chain" or n_extra >= 2:
                # diagonal BD
                ax.plot([pts[1][0], pts[3][0]],
                        [pts[1][1], pts[3][1]],
                        color=palette[5 % len(palette)], linewidth=lw * 0.9,
                        linestyle=":")
            if n_extra >= 3:
                # diameter through O
                diam1 = self._point_on_circle(cx, cy, R, rot_off + 35)
                diam2 = self._point_on_circle(cx, cy, R, rot_off + 35 + 180)
                ax.plot([diam1[0], diam2[0]],
                        [diam1[1], diam2[1]],
                        color="#888888", linewidth=lw * 0.7, linestyle="-.")
            if n_extra >= 4:
                # extra chord
                p1 = self._point_on_circle(cx, cy, R, rot_off - 40)
                p2 = self._point_on_circle(cx, cy, R, rot_off + 220)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#666666", linewidth=lw * 0.7, linestyle=":")
            # vertex labels
            for p, lbl in zip(pts, labels):
                ax.plot(p[0], p[1], "o", color=palette[0], markersize=5)
                nx = p[0] * 1.12
                ny = p[1] * 1.12
                ax.text(nx, ny, lbl, fontsize=fs + 2,
                        fontweight="bold", family=ff, color=line_color)
            # Draw given labels
            def fmt(v):
                if isinstance(v, float) and cfg["non_integer_angles"]:
                    return f"{v:.1f}"
                return f"{int(round(v))}"

            A_text = f"∠A = {fmt(givens.get('A', 0))}°" if "A" in givens else None
            B_text = f"∠B = {fmt(givens.get('B', 0))}°" if "B" in givens else None
            C_text = f"∠C = {fmt(givens.get('C', 0))}°" if "C" in givens else None
            # Add white bboxes so labels remain legible when strokes cross
            lbl_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.82)
            if A_text:
                ax.text(pts[0][0] * 0.6, pts[0][1] * 0.6, A_text,
                        fontsize=fs - 1, color="#2e7d32",
                        fontweight="bold", bbox=lbl_bbox)
            if B_text:
                ax.text(pts[1][0] * 0.6, pts[1][1] * 0.6, B_text,
                        fontsize=fs - 1, color="#2e7d32",
                        fontweight="bold", bbox=lbl_bbox)
            if C_text:
                ax.text(pts[2][0] * 0.6, pts[2][1] * 0.6, C_text,
                        fontsize=fs - 1, color="#2e7d32",
                        fontweight="bold", bbox=lbl_bbox)
            if "ABD" in givens:
                ax.text((pts[1][0] + pts[3][0]) * 0.25,
                        (pts[1][1] + pts[3][1]) * 0.25 - 0.3,
                        f"∠ABD = {fmt(givens['ABD'])}°",
                        fontsize=fs - 1, color="#2e7d32",
                        fontweight="bold", bbox=lbl_bbox)
            if "arc_BC" in givens:
                midA = (base[1] + base[2]) / 2
                ann_p = self._point_on_circle(cx, cy, R + 0.5, midA)
                ax.text(ann_p[0], ann_p[1],
                        f"arc BC = {fmt(givens['arc_BC'])}°",
                        fontsize=fs - 1, color="#1565c0", fontweight="bold",
                        ha="center")
            if "tc_angle" in givens:
                ax.text(pts[0][0] + 0.15, pts[0][1] - 0.4,
                        f"tc = {fmt(givens['tc_angle'])}°",
                        fontsize=fs - 1, color="#c0392b", fontweight="bold")
            # Unknown indicator — wrap in white bbox so it doesn't collide
            # with vertex letters when pulled toward centre.
            unk_bbox = dict(boxstyle="round,pad=0.18", facecolor="white",
                            edgecolor="#c0392b", alpha=0.9)
            if variant == "opposite_angles":
                ax.text(pts[3][0] * 0.6, pts[3][1] * 0.6, "∠D = x°",
                        fontsize=fs + 1, color="#c0392b",
                        fontweight="bold", bbox=unk_bbox)
            elif variant == "inscribed_on_quad":
                ax.text(pts[3][0] * 0.5, pts[3][1] * 0.5, "∠BDC = x°",
                        fontsize=fs + 1, color="#c0392b",
                        fontweight="bold", bbox=unk_bbox)
            elif variant == "tangent_chord_quad":
                # arc AB annotation
                midAB = (base[0] + base[1]) / 2
                ann_p = self._point_on_circle(cx, cy, R + 0.55, midAB)
                ax.text(ann_p[0], ann_p[1], "arc AB = x°",
                        fontsize=fs + 1, color="#c0392b",
                        fontweight="bold", ha="center", bbox=unk_bbox)
            elif variant == "diagonal_chain":
                # Pull further toward centre so it doesn't overlap D label
                ax.text(pts[3][0] * 0.5, pts[3][1] * 0.5 + 0.3, "∠ADB = x°",
                        fontsize=fs + 1, color="#c0392b",
                        fontweight="bold", bbox=unk_bbox)
        elif variant == "external_secant":
            # External point P outside circle; tangent from P to T, secant
            # from P through two points of the circle.
            P = (cx + R * 2.2, cy - 0.3)
            # tangent point
            T = self._point_on_circle(cx, cy, R, 135)
            # secant: near pt (close to P), far pt
            near = self._point_on_circle(cx, cy, R, 185)
            far = self._point_on_circle(cx, cy, R, 270)
            ax.plot([P[0], T[0]], [P[1], T[1]],
                    color=palette[4 % len(palette)], linewidth=lw)
            ax.plot([P[0], far[0]], [P[1], far[1]],
                    color=palette[5 % len(palette)], linewidth=lw)
            ax.plot(P[0], P[1], "o", color=palette[0], markersize=6)
            ax.text(P[0] + 0.1, P[1], "P",
                    fontsize=fs + 1, fontweight="bold",
                    family=ff, color=line_color)
            for p, l in [(T, "T"), (near, "B"), (far, "D")]:
                ax.plot(p[0], p[1], "o", color=palette[0], markersize=5)
                ax.text(p[0] * 1.1, p[1] * 1.1, l,
                        fontsize=fs + 1, fontweight="bold",
                        family=ff, color=line_color)

            def fmt(v):
                if isinstance(v, float) and cfg["non_integer_angles"]:
                    return f"{v:.1f}"
                return f"{int(round(v))}"

            ax.text(0, R + 0.3,
                    f"far arc = {fmt(givens['far_arc'])}°",
                    fontsize=fs - 1, color="#1565c0",
                    ha="center", fontweight="bold")
            ax.text(0, -R - 0.4,
                    f"near arc = {fmt(givens['near_arc'])}°",
                    fontsize=fs - 1, color="#1565c0",
                    ha="center", fontweight="bold")
            ax.text(P[0] + 0.4, P[1] + 0.35, "∠P = x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")

        pad = 1.5
        ax.set_xlim(-R - pad, R + pad + 2.0)
        ax.set_ylim(-R - pad, R + pad)
        ax.set_title(self._rng.choice(self._TITLE_VARIANTS),
                     fontsize=fs + 2, fontweight="bold", pad=8, family=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
