"""
Angle Bisector Chain QA environment.

Goal: targeted fix for Angle and Plane
Geometry reasoning Property. Triangle with multiple angle bisectors
(and/or perpendicular bisector / median / altitude), requiring chaining
through triangle-sum + bisector halving + supplementary/complementary.

Difficulty schedule (continuous level 0..9):
  Axis 1: n_construction_lines = 1 + level // 2     -> 1..5
  Axis 2: n_given_values = max(1, 4 - level // 3)   -> 4..1
  Axis 3: triangle_type (L0=isosceles, L5=right, L9=scalene w/ fractional)

Output: single integer (degrees).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class AngleBisectorChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "angle_bisector_chain"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_VARIANTS = [
        "Angle bisector chain",
        "Find angle x",
        "Bisector geometry",
        "Triangle with bisectors",
        "Angle chase (bisector)",
        "Incenter geometry",
        "Bisector + Altitude",
        "Chain of constructions",
        "Triangle cevians",
    ]

    # L0 starts with simple isosceles AND ensures low difficulty; L9 uses
    # scalene + multi-line + harder question variants that require chaining.
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            tri_families = ["isosceles", "equilateral_like"]
        elif level <= 5:
            tri_families = ["isosceles", "right", "scalene"]
        elif level <= 7:
            tri_families = ["right", "scalene", "obtuse"]
        else:
            tri_families = ["scalene", "obtuse", "narrow_acute"]
        return {
            "n_construction_lines": 1 + level // 2,           # 1..5
            "n_given_values":       max(1, 4 - level // 3),   # 4..1
            "triangle_families":    tri_families,
            "use_fractional":       level >= 7,
            "tight_distractors":    level >= 5,                # was 4 — give L4 looser opts
            "show_formula":         level <= 3,                # was 2 — keep hint through L3
            "level":                level,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 313)
        self._primary_complexity_feature = cfg["n_construction_lines"]

        # Problem pool scales with construction lines — structurally different
        # at each level tier, not just parameter-tweaked.
        n_lines = cfg["n_construction_lines"]
        level = cfg["level"]
        # Iter 3 (2026-04-17): L3 spiked to 0.50 while L0=0.25 — the
        # L2-L3 additions (bisect_base_angle, bisector_chain_sum) were
        # easier than the L0 baseline (bisect_one with construction lines
        # only). Push those easier variants earlier (L1+) so L3 is not
        # a sudden easier task.
        # Iter 4 (2026-04-17): L3=0.70 spike persisted — bisect_base_angle
        # and bisector_chain_sum at L3 involve a single 180-C/2 calculation,
        # much easier than L0's bisect_one with small-angle triangle
        # rendering. Make L3 use bisect_two_angles (90+A/2) which requires
        # the incenter-bisector theorem (genuinely 2-step).
        if level <= 0:
            ptypes = ["bisect_one"]
        elif level == 1:
            ptypes = ["bisect_one", "bisect_base_angle"]
        elif level == 2:
            ptypes = ["bisect_base_angle", "bisector_chain_sum"]
        elif level <= 4:
            # L3-L4: require incenter-angle theorem (harder than L2 chain).
            ptypes = ["bisect_two_angles", "external_bisector"]
        elif level <= 6:
            ptypes = ["bisect_two_angles", "external_bisector",
                      "bisector_plus_altitude"]
        elif level <= 7:
            ptypes = ["bisector_plus_altitude", "external_bisector"]
        else:
            ptypes = ["bisector_plus_altitude",
                      "bisector_plus_median_altitude"]

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
        if ptype == "bisect_one":
            return self._bisect_one(rng, cfg)
        if ptype == "bisect_base_angle":
            return self._bisect_base_angle(rng, cfg)
        if ptype == "bisect_two_angles":
            return self._bisect_two_angles(rng, cfg)
        if ptype == "bisector_plus_altitude":
            return self._bisector_plus_altitude(rng, cfg)
        if ptype == "bisector_plus_median_altitude":
            return self._bisector_plus_median_altitude(rng, cfg)
        if ptype == "external_bisector":
            return self._external_bisector(rng, cfg)
        if ptype == "bisector_chain_sum":
            return self._bisector_chain_sum(rng, cfg)
        return None

    # ------------------------------------------------------------------ #
    # Problem variants
    # ------------------------------------------------------------------ #
    @staticmethod
    def _angles_ok(A, B, C) -> bool:
        """Reject triangles that would render too narrow for clear labels.
        We require each angle >= 22 deg so no vertex label drifts through
        another side, AND min/max angle ratio reasonable."""
        mn = min(A, B, C)
        if mn < 22.0:
            return False
        mx = max(A, B, C)
        if mx >= 160.0:
            return False
        return True

    def _pick_triangle(self, rng, cfg) -> Tuple[float, float, float]:
        ttype = rng.choice(cfg["triangle_families"])
        if ttype == "isosceles":
            apex = rng.randint(30, 100)
            base = (180 - apex) / 2
            if cfg["use_fractional"]:
                if self._angles_ok(apex, base, base):
                    return apex, base, base
            apex = int(apex)
            base = (180 - apex) // 2
            apex = 180 - 2 * base
            if self._angles_ok(apex, base, base):
                return apex, base, base
            return 60.0, 60.0, 60.0
        if ttype == "equilateral_like":
            # 3 angles near 60
            a = rng.randint(55, 65)
            b = rng.randint(55, 65)
            c = 180 - a - b
            return float(a), float(b), float(c)
        if ttype == "right":
            a = rng.randint(25, 65)
            b = 90 - a
            return 90.0, float(a), float(b)
        if ttype == "obtuse":
            # Bounded so min angle stays >= 25.
            a = rng.randint(100, 125)
            min_other = 25
            max_other = 180 - a - min_other
            if max_other < min_other:
                return 120.0, 30.0, 30.0
            b = rng.randint(min_other, max_other)
            c = 180 - a - b
            if self._angles_ok(a, b, c):
                return float(a), float(b), float(c)
            return 120.0, 30.0, 30.0
        if ttype == "narrow_acute":
            # Redesigned: "narrow_acute" previously produced triangles with
            # one angle as low as ~7 deg, which caused severe label drift.
            # Now: all angles in [25, 85], avoiding extremes.
            for _ in range(10):
                a = rng.randint(55, 85)
                b = rng.randint(35, 80)
                c = 180 - a - b
                if self._angles_ok(a, b, c):
                    return float(a), float(b), float(c)
            return 75.0, 60.0, 45.0
        # scalene
        for _ in range(10):
            a = rng.randint(40, 80)
            b = rng.randint(40, 80)
            c = 180 - a - b
            if self._angles_ok(a, b, c):
                return float(a), float(b), float(c)
        return 60.0, 55.0, 65.0

    def _bisect_one(self, rng, cfg):
        """L0-ish: bisect one vertex angle. Given that angle, find the half."""
        if cfg["level"] <= 1:
            # Force an EVEN-angle bisection so the half is a whole integer
            # and the labeled value is the one the model bisects.
            apex = rng.choice([40, 60, 80, 100, 120])
            base = (180 - apex) // 2
            A, B, C = float(apex), float(base), float(base)
            ang = apex
            which = "A"
        else:
            A, B, C = self._pick_triangle(rng, cfg)
            which = rng.choice(["A", "B", "C"])
            ang = {"A": A, "B": B, "C": C}[which]
        half = ang / 2.0
        if half < 8 or half > 88:
            return None
        given = {"vertex": which, "ang": ang}
        unknown = {"type": "half_at_" + which, "val": half}
        return self._finalize(rng, cfg, given, unknown,
                              (A, B, C), variant="bisect_one")

    def _bisect_base_angle(self, rng, cfg):
        """Bisect base angle of an isosceles/right triangle. Given apex
        (or base+another), find half-angle at base-vertex."""
        A, B, C = self._pick_triangle(rng, cfg)
        # bisect B
        half_B = B / 2.0
        if half_B < 8 or half_B > 88:
            return None
        given = {"vertex": "B", "ang": B, "apex": A}
        unknown = {"type": "half_at_B", "val": half_B}
        return self._finalize(rng, cfg, given, unknown, (A, B, C),
                              variant="bisect_base_angle")

    def _bisect_two_angles(self, rng, cfg):
        """Two bisectors meeting inside the triangle. Given one full angle,
        find the angle between bisectors. Uses: angle between bisectors of
        B and C = 90 + A/2."""
        A, B, C = self._pick_triangle(rng, cfg)
        # angle B'I C = 90 + A/2 where I is incenter
        ans = 90.0 + A / 2.0
        if ans <= 95 or ans >= 180:
            return None
        given = {"vertex": "A", "ang": A}
        unknown = {"type": "angle_between_bisectors_B_C", "val": ans}
        return self._finalize(rng, cfg, given, unknown, (A, B, C),
                              variant="bisect_two_angles")

    def _bisector_plus_altitude(self, rng, cfg):
        """Bisector from A + altitude from A to BC. Angle between them
        = |B - C| / 2. Given B and C, find the angle."""
        A, B, C = self._pick_triangle(rng, cfg)
        gap = abs(B - C)
        ans = gap / 2.0
        if ans < 1 or ans > 80:
            return None
        given = {"vertex_B": "B", "ang_B": B, "vertex_C": "C", "ang_C": C}
        unknown = {"type": "angle_bisector_vs_altitude_from_A", "val": ans}
        return self._finalize(rng, cfg, given, unknown, (A, B, C),
                              variant="bisector_plus_altitude")

    def _bisector_plus_median_altitude(self, rng, cfg):
        """3-step chain: triangle with bisector + altitude + median. Find the
        angle between the bisector of A and the altitude from A,
        knowing only A and one other angle (derive B,C first)."""
        A, B, C = self._pick_triangle(rng, cfg)
        if A <= 20 or A >= 160:
            return None
        gap = abs(B - C)
        ans = gap / 2.0
        if ans < 1 or ans > 80:
            return None
        given = {"vertex_A": "A", "ang_A": A, "vertex_B": "B", "ang_B": B}
        unknown = {"type": "chain_bisector_altitude", "val": ans}
        return self._finalize(rng, cfg, given, unknown, (A, B, C),
                              variant="bisector_plus_median_altitude")

    def _external_bisector(self, rng, cfg):
        """The external bisector of ∠A is perpendicular to the internal
        bisector of ∠A. Answer is (180 - A)/2, the angle between the
        external bisector and side AB extended. Chaining required."""
        A, B, C = self._pick_triangle(rng, cfg)
        ext_half = (180 - A) / 2.0
        if ext_half < 5 or ext_half > 85:
            return None
        given = {"vertex": "A", "ang": A}
        unknown = {"type": "external_bisector_half", "val": ext_half}
        return self._finalize(rng, cfg, given, unknown, (A, B, C),
                              variant="external_bisector")

    def _bisector_chain_sum(self, rng, cfg):
        """Sum of two bisected half-angles. ∠A and ∠B each bisected, answer
        is (A+B)/2 = (180-C)/2. Requires chaining via triangle sum."""
        A, B, C = self._pick_triangle(rng, cfg)
        sum_half = (A + B) / 2.0
        if sum_half < 10 or sum_half > 170:
            return None
        given = {"vertex": "C", "ang": C}
        unknown = {"type": "sum_half_A_B", "val": sum_half}
        return self._finalize(rng, cfg, given, unknown, (A, B, C),
                              variant="bisector_chain_sum")

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    def _finalize(self, rng, cfg, given, unknown, tri_angles, variant):
        val = unknown["val"]
        if cfg["use_fractional"] and abs(val - round(val)) > 0.05:
            answer = f"{val:.1f}"
            ans_val = round(val, 1)
        else:
            ans_val = int(round(val))
            answer = str(ans_val)
        if ans_val <= 0 or ans_val >= 180:
            return None

        # BUGFIX 2026-04-24: pass ans_val and answer into question text so the
        # prompt can say 'round to 1 decimal' instead of 'single integer' when
        # the answer is fractional.
        question = self._question_text(given, unknown, cfg, variant,
                                        ans_val=ans_val, answer_str=answer)
        image = self._render(tri_angles, given, unknown, cfg, variant)
        return question, answer, image

    def _question_text(self, given, unknown, cfg, variant, ans_val=None,
                        answer_str=None):
        # Angle values are drawn on the image (see label_angle_at in _render).
        # Do NOT restate them in the question — the model must READ the image.
        # 4+ phrasings per variant
        rng = random.Random((self.seed or 0) * 1000 + cfg["level"] * 37 + 707)
        vtemps = {
            "bisect_one": [
                ("In triangle ABC (shown), angle {v} is bisected. Using "
                 "the angle value labeled in the figure, find the measure "
                 "of one half of angle {v} in degrees."),
                ("A ray from {v} bisects ∠{v} in triangle ABC. Based on the "
                 "angle labeled in the figure, what is the measure of each "
                 "half (in degrees)?"),
                ("Triangle ABC is shown with the bisector at vertex {v}. "
                 "From the labeled angle, compute half of ∠{v} (degrees)."),
                ("The bisector from {v} splits ∠{v} into two equal halves. "
                 "Using the figure's angle label, give the measure of one "
                 "half in degrees."),
            ],
            "bisect_base_angle": [
                ("In triangle ABC (shown), angle B is bisected by a ray from "
                 "B. Using the angles labeled in the figure, find the "
                 "measure of each half of angle B in degrees."),
                ("The ray from B bisects ∠B. Read the angle labels on the "
                 "figure and report half of ∠B (degrees)."),
                ("Given the angle labels in the figure, determine the "
                 "measure of one half of ∠B after it is bisected."),
                ("Half of ∠B (the base angle) — compute from the angles "
                 "shown in the image. Answer in degrees."),
            ],
            "bisect_two_angles": [
                ("In triangle ABC (shown), the bisectors of angles B and C "
                 "meet at the incenter I. Using the angle labeled in the "
                 "figure, find the measure of angle BIC in degrees."),
                ("Two bisectors from B and C meet at I. From the angle "
                 "labeled in the image, find ∠BIC (degrees)."),
                ("Using the image, compute ∠BIC where I is the incenter "
                 "formed by the bisectors of ∠B and ∠C."),
                ("Bisectors of ∠B and ∠C meet at I; using the angle label, "
                 "find ∠BIC."),
            ],
            "bisector_plus_altitude": [
                ("In triangle ABC (shown), the dashed line from A is the "
                 "angle bisector and the dotted line from A is the altitude "
                 "to BC. Using the angles labeled in the figure, find the "
                 "angle (in degrees) between the bisector and the altitude."),
                ("Find the angle between the bisector (dashed) and altitude "
                 "(dotted) from A, using the angle values labeled in the "
                 "figure."),
                ("A triangle is shown with bisector + altitude from A. From "
                 "the labels, determine the angle between them (degrees)."),
                ("The difference between the bisector's and altitude's "
                 "direction from A — compute from the figure labels."),
            ],
            "bisector_plus_median_altitude": [
                ("In triangle ABC (shown), the dashed line from A is the "
                 "angle bisector and the dotted line from A is the altitude "
                 "to BC. Using the angles labeled in the figure, find the "
                 "angle (in degrees) between the bisector and the altitude."),
                ("From the labels, compute the angle between the bisector "
                 "(dashed) and the altitude (dotted) from A."),
                ("The dashed and dotted rays from A are the bisector and "
                 "altitude; use the labeled angles to find the angle "
                 "between them."),
                ("Based on the triangle's angle labels, what is the angle "
                 "between the bisector and altitude from A?"),
            ],
            "external_bisector": [
                ("In triangle ABC (shown), the external bisector of ∠A is "
                 "drawn. Using the labeled angle, find the angle (in "
                 "degrees) between the external bisector and side AB "
                 "extended."),
                ("Based on the angle labels, compute (180° − ∠A)/2 — the "
                 "angle between the external bisector at A and side AB."),
                ("From the figure, determine the angle made by the external "
                 "bisector of ∠A with line AB."),
                ("Read ∠A from the figure and compute half of the exterior "
                 "angle at A, in degrees."),
            ],
            "bisector_chain_sum": [
                ("In triangle ABC (shown), the bisectors from A and from B "
                 "each bisect their respective vertex angles. Based on the "
                 "labeled angle, find (∠A + ∠B)/2 in degrees."),
                ("Using only the angle at C (labeled), compute (180° − "
                 "∠C)/2, which equals the sum of the half-angles at A "
                 "and B."),
                ("The image labels ∠C. Find the combined half-measure of "
                 "∠A + ∠B (i.e. (A+B)/2)."),
                ("After bisecting ∠A and ∠B, their two halves sum to "
                 "(A+B)/2. Compute this from the angle labeled at C."),
            ],
        }
        vv = given.get("vertex", "A")
        pool = vtemps.get(variant, vtemps["bisect_one"])
        q = rng.choice(pool).format(v=vv)
        # BUGFIX 2026-04-24: choose 'round to 1 decimal' when the answer is
        # fractional; otherwise say 'single integer'. Previously said 'single
        # integer' unconditionally even when use_fractional produced 0.5's.
        is_fractional = False
        if ans_val is not None:
            try:
                is_fractional = abs(float(ans_val) - round(float(ans_val))) > 0.05
            except Exception:
                is_fractional = False
        if is_fractional:
            q += " Round to 1 decimal."
        else:
            q += " Answer with a single integer."
        if cfg.get("show_formula") and variant == "bisect_one":
            q += " (Hint: half of an angle = angle / 2.)"
        return q

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, tri_angles, given, unknown, cfg, variant):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.3 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        palette = style["palette"]
        line_color = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        A_ang, B_ang, C_ang = tri_angles
        # Place triangle: B at origin, C on x-axis at (a,0), A derived from angles
        # Use Law of Sines with an arbitrary scale
        rot = self._rng.uniform(-0.3, 0.3)
        a_side = 4.0
        # compute A position given B and C and angles at B, C
        # B = (0,0), C = (a_side, 0), angle at B = B_ang, angle at C = C_ang
        b_r = math.radians(B_ang)
        Bx, By = 0.0, 0.0
        Cx, Cy = a_side, 0.0
        # direction from B: slope angle = B_ang
        # line from B: y = tan(B_ang) * x
        # line from C: y = -tan(C_ang) * (x - a_side)
        if B_ang + C_ang >= 179:
            return None
        mB = math.tan(b_r)
        mC = -math.tan(math.radians(C_ang))
        # intersect
        try:
            Ax = (a_side * mC) / (mC - mB)
            Ay = mB * Ax
        except ZeroDivisionError:
            return None
        # apply rotation + translation
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        def rt(px, py):
            return (cos_r * px - sin_r * py, sin_r * px + cos_r * py)
        B = rt(Bx, By)
        C = rt(Cx, Cy)
        A = rt(Ax, Ay)
        xs = [A[0], B[0], C[0]]
        ys = [A[1], B[1], C[1]]
        pad = 1.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad + 0.4)

        # Draw triangle
        tri_xs = [A[0], B[0], C[0], A[0]]
        tri_ys = [A[1], B[1], C[1], A[1]]
        ax.plot(tri_xs, tri_ys, color=line_color, linewidth=lw + 0.5)

        # Label vertices
        ax.plot(A[0], A[1], "o", color=palette[0], markersize=5)
        ax.plot(B[0], B[1], "o", color=palette[0], markersize=5)
        ax.plot(C[0], C[1], "o", color=palette[0], markersize=5)
        ax.text(A[0] - 0.1, A[1] + 0.25, "A", fontsize=fs + 2,
                fontweight="bold", family=ff, color=line_color)
        ax.text(B[0] - 0.35, B[1] - 0.35, "B", fontsize=fs + 2,
                fontweight="bold", family=ff, color=line_color)
        ax.text(C[0] + 0.1, C[1] - 0.35, "C", fontsize=fs + 2,
                fontweight="bold", family=ff, color=line_color)

        # Draw construction lines depending on variant
        bisector_color = palette[3 % len(palette)]
        altitude_color = palette[4 % len(palette)]
        median_color = palette[5 % len(palette)]

        def midpoint(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        def foot_of_perp(A, B, C):
            # Foot of perpendicular from A onto line BC
            bc = (C[0] - B[0], C[1] - B[1])
            ab = (A[0] - B[0], A[1] - B[1])
            denom = bc[0]**2 + bc[1]**2
            t = (ab[0] * bc[0] + ab[1] * bc[1]) / denom
            return (B[0] + t * bc[0], B[1] + t * bc[1])

        def bisector_foot_on_opposite(V, U1, U2):
            """Foot of the internal angle bisector from vertex V onto the
            opposite side U1U2. Uses: divides in ratio of adjacent sides."""
            d1 = math.hypot(V[0] - U1[0], V[1] - U1[1])
            d2 = math.hypot(V[0] - U2[0], V[1] - U2[1])
            total = d1 + d2
            # the bisector from V meets U1U2 at point X with
            # U1X : XU2 = d(V,U1) : d(V,U2)? No — opposite sides:
            # U1X/XU2 = (adj to V, i.e. VU1) / (adj, VU2) actually
            # Law: U1X/XU2 = VU1_to_U1? Classical: ratio = VU2 / VU1? — no.
            # By the angle bisector theorem: U1X / XU2 = VU1 / VU2.
            # Wait — the foot divides OPPOSITE side in ratio equal to the
            # RATIO OF ADJACENT SIDES (i.e. sides meeting at V).
            # VU1 and VU2 are the adjacent sides; U1X/XU2 = VU1/VU2.
            return (U1[0] + (d1 / total) * (U2[0] - U1[0]),
                    U1[1] + (d1 / total) * (U2[1] - U1[1]))

        n_lines = cfg["n_construction_lines"]

        if variant == "bisect_one":
            # Bisect one vertex angle
            vertex = given["vertex"]
            if vertex == "A":
                V, U1, U2 = A, B, C
            elif vertex == "B":
                V, U1, U2 = B, A, C
            else:
                V, U1, U2 = C, A, B
            foot = bisector_foot_on_opposite(V, U1, U2)
            ax.plot([V[0], foot[0]], [V[1], foot[1]],
                    color=bisector_color, linewidth=lw, linestyle="--")
            ax.text(foot[0], foot[1] - 0.25, "D",
                    fontsize=fs, fontweight="bold", color=bisector_color)
        elif variant == "bisect_base_angle":
            # Bisect B
            foot = bisector_foot_on_opposite(B, A, C)
            ax.plot([B[0], foot[0]], [B[1], foot[1]],
                    color=bisector_color, linewidth=lw, linestyle="--")
            ax.text(foot[0], foot[1] + 0.15, "D",
                    fontsize=fs, fontweight="bold", color=bisector_color)
        elif variant == "bisect_two_angles":
            # Two bisectors, from B and C, meeting at incenter I
            f_b = bisector_foot_on_opposite(B, A, C)
            f_c = bisector_foot_on_opposite(C, A, B)
            # Incenter: intersection of bisectors.
            # Compute by solving line-line intersection.
            def intersect(p1, p2, p3, p4):
                x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
                d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(d) < 1e-9:
                    return None
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            I = intersect(B, f_b, C, f_c)
            if I is not None:
                ax.plot([B[0], f_b[0]], [B[1], f_b[1]],
                        color=bisector_color, linewidth=lw, linestyle="--")
                ax.plot([C[0], f_c[0]], [C[1], f_c[1]],
                        color=bisector_color, linewidth=lw, linestyle="--")
                ax.plot(I[0], I[1], "o", color=bisector_color, markersize=5)
                ax.text(I[0] + 0.1, I[1] + 0.1, "I",
                        fontsize=fs + 1, fontweight="bold",
                        color=bisector_color)
        elif variant == "bisector_plus_altitude":
            # Bisector from A + altitude from A to BC
            f_bis = bisector_foot_on_opposite(A, B, C)
            f_alt = foot_of_perp(A, B, C)
            ax.plot([A[0], f_bis[0]], [A[1], f_bis[1]],
                    color=bisector_color, linewidth=lw, linestyle="--")
            ax.plot([A[0], f_alt[0]], [A[1], f_alt[1]],
                    color=altitude_color, linewidth=lw, linestyle=":")
            ax.text(f_bis[0] - 0.1, f_bis[1] - 0.3, "D",
                    fontsize=fs, fontweight="bold", color=bisector_color)
            ax.text(f_alt[0] + 0.05, f_alt[1] - 0.3, "H",
                    fontsize=fs, fontweight="bold", color=altitude_color)
        elif variant == "bisector_plus_median_altitude":
            f_bis = bisector_foot_on_opposite(A, B, C)
            f_alt = foot_of_perp(A, B, C)
            f_med = midpoint(B, C)
            ax.plot([A[0], f_bis[0]], [A[1], f_bis[1]],
                    color=bisector_color, linewidth=lw, linestyle="--")
            ax.plot([A[0], f_alt[0]], [A[1], f_alt[1]],
                    color=altitude_color, linewidth=lw, linestyle=":")
            if n_lines >= 5:
                ax.plot([A[0], f_med[0]], [A[1], f_med[1]],
                        color=median_color, linewidth=lw, linestyle="-.")
                ax.text(f_med[0] + 0.05, f_med[1] - 0.35, "M",
                        fontsize=fs, fontweight="bold", color=median_color)
            ax.text(f_bis[0] - 0.05, f_bis[1] - 0.3, "D",
                    fontsize=fs, fontweight="bold", color=bisector_color)
            ax.text(f_alt[0] + 0.05, f_alt[1] - 0.3, "H",
                    fontsize=fs, fontweight="bold", color=altitude_color)
        elif variant == "external_bisector":
            # Draw both internal and external bisector from A
            f_int = bisector_foot_on_opposite(A, B, C)
            ax.plot([A[0], f_int[0]], [A[1], f_int[1]],
                    color=bisector_color, linewidth=lw, linestyle="--")
            # External bisector is perpendicular to internal bisector at A.
            dx = f_int[0] - A[0]
            dy = f_int[1] - A[1]
            n = math.hypot(dx, dy) + 1e-9
            # Perpendicular direction
            ex, ey = -dy / n, dx / n
            ext_len = max(n * 1.3, 3.0)
            ex_pt = (A[0] + ex * ext_len, A[1] + ey * ext_len)
            ex_pt2 = (A[0] - ex * ext_len, A[1] - ey * ext_len)
            ax.plot([ex_pt[0], ex_pt2[0]], [ex_pt[1], ex_pt2[1]],
                    color=altitude_color, linewidth=lw, linestyle="-.")
            ax.text(ex_pt[0] + 0.1, ex_pt[1] + 0.1, "E",
                    fontsize=fs, fontweight="bold", color=altitude_color)
        elif variant == "bisector_chain_sum":
            # Draw bisectors from A and B
            fa = bisector_foot_on_opposite(A, B, C)
            fb = bisector_foot_on_opposite(B, A, C)
            ax.plot([A[0], fa[0]], [A[1], fa[1]],
                    color=bisector_color, linewidth=lw, linestyle="--")
            ax.plot([B[0], fb[0]], [B[1], fb[1]],
                    color=altitude_color, linewidth=lw, linestyle="--")
            ax.text(fa[0] + 0.05, fa[1] - 0.3, "D",
                    fontsize=fs, fontweight="bold", color=bisector_color)
            ax.text(fb[0] + 0.05, fb[1] + 0.2, "E",
                    fontsize=fs, fontweight="bold", color=altitude_color)
        else:  # fallback — no additional construction
            pass

        # Label given angles inside the triangle
        def label_angle_at(V, other1, other2, text, color):
            # position label along the angle bisector direction from V.
            # For narrow triangles, the default 0.55-unit offset can push
            # the label past the opposite side. Cap the offset at 25% of
            # the shorter adjacent edge so the label stays near the vertex.
            v1 = ((other1[0] - V[0]), (other1[1] - V[1]))
            v2 = ((other2[0] - V[0]), (other2[1] - V[1]))
            n1 = math.hypot(*v1) + 1e-9
            n2 = math.hypot(*v2) + 1e-9
            min_edge = min(n1, n2)
            offset = min(0.55, 0.25 * min_edge)
            # Unit bisector direction
            ux = v1[0] / n1 + v2[0] / n2
            uy = v1[1] / n1 + v2[1] / n2
            un = math.hypot(ux, uy) + 1e-9
            bx = V[0] + offset * (ux / un)
            by = V[1] + offset * (uy / un)
            ax.text(bx, by, text, fontsize=fs - 1, color=color,
                    fontweight="bold", ha="center", va="center")

        def fmt(v):
            return f"{v:.1f}" if cfg["use_fractional"] else f"{int(round(v))}"

        # Only label the angles needed by this variant
        n_given = cfg["n_given_values"]
        givens_map = []
        if variant == "bisect_one":
            givens_map = [(given["vertex"], given["ang"])]
        elif variant == "bisect_base_angle":
            givens_map = [("A", given["apex"]), ("B", given["ang"])]
        elif variant == "bisect_two_angles":
            givens_map = [("A", given["ang"])]
        elif variant == "bisector_plus_altitude":
            givens_map = [("B", given["ang_B"]), ("C", given["ang_C"])]
        elif variant == "external_bisector":
            givens_map = [(given["vertex"], given["ang"])]
        elif variant == "bisector_chain_sum":
            givens_map = [(given["vertex"], given["ang"])]
        else:
            givens_map = [("A", given["ang_A"]), ("B", given["ang_B"])]

        # Image must show enough labels to solve the problem — values are NOT
        # in the question text anymore. Keep at least the full given set.
        givens_map = givens_map[:len(givens_map)]

        for vert, val in givens_map:
            if vert == "A":
                V, o1, o2 = A, B, C
            elif vert == "B":
                V, o1, o2 = B, A, C
            else:
                V, o1, o2 = C, A, B
            # Use "∠A = NN°" (rather than just "NN°") so the model can't
            # mistake the value for a sub-angle when a bisector is present.
            label_angle_at(V, o1, o2, f"∠{vert} = {fmt(val)}°", "#2e7d32")

        # x-label on the unknown
        if variant == "bisect_one":
            # label on half-angle: position between V and foot, offset to one side
            vv = given["vertex"]
            if vv == "A":
                V, U1, U2 = A, B, C
            elif vv == "B":
                V, U1, U2 = B, A, C
            else:
                V, U1, U2 = C, A, B
            foot = bisector_foot_on_opposite(V, U1, U2)
            mx = V[0] * 0.7 + foot[0] * 0.3
            my = V[1] * 0.7 + foot[1] * 0.3
            ax.text(mx + 0.4, my + 0.05, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        elif variant == "bisect_base_angle":
            ax.text(B[0] + 0.35, B[1] + 0.15, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        elif variant == "bisect_two_angles":
            # x is angle BIC — put near incenter area
            ax.text((B[0] + C[0]) / 2 - 0.2,
                    (B[1] + C[1]) / 2 + 0.5,
                    "x° (∠BIC)",
                    fontsize=fs, color="#c0392b", fontweight="bold")
        elif variant == "bisector_plus_altitude":
            # Place x° between altitude foot H and bisector foot D, not at vertex A
            f_alt = (A[0], B[1] if abs(B[1] - C[1]) < 1e-9 else (B[1] + C[1]) / 2)
            f_bis = bisector_foot_on_opposite(A, B, C)
            mx = (f_alt[0] + f_bis[0]) / 2
            my = (f_alt[1] + f_bis[1]) / 2 + 0.3
            ax.text(mx, my, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        elif variant == "external_bisector":
            ax.text(A[0] + 0.4, A[1] - 0.45, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        elif variant == "bisector_chain_sum":
            ax.text((A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0 + 0.3,
                    "x° = (∠A + ∠B)/2",
                    fontsize=fs, color="#c0392b", fontweight="bold")
        else:
            # bisector_plus_median_altitude: place x° between bisector foot D
            # and altitude foot H, well below the angle label at A.
            f_alt = (A[0], B[1] if abs(B[1] - C[1]) < 1e-9 else (B[1] + C[1]) / 2)
            f_bis = bisector_foot_on_opposite(A, B, C)
            mx = (f_alt[0] + f_bis[0]) / 2 - 0.5
            my = (f_alt[1] + f_bis[1]) / 2 + 0.6
            ax.text(mx, my, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")

        ax.set_title(self._rng.choice(self._TITLE_VARIANTS),
                     fontsize=fs + 2, fontweight="bold", pad=8, family=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
