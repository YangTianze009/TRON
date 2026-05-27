"""
Arc-Angle Relationship QA environment.

Goal: targeted fix for Angle and multi-math Circle.
A circle with inscribed angles, central angles, chords, secants, tangents
and arc measures labeled in degrees. Problems exercise multiple
arc-angle theorems.

Difficulty schedule:
  Axis 1: n_lines = 2 + level // 2                 -> 2..6 (chords/secants/tangents)
  Axis 2: theorem_type scales:
           L0 -> inscribed angle
           L3 -> tangent-chord
           L6 -> secant-secant from external point
           L9 -> mixed with arc addition

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

class ArcAngleRelationshipQA(StandaloneVisualEnv):
    ENV_NAME = "arc_angle_relationship"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_VARIANTS = [
        "Arc-angle relationship",
        "Find angle x",
        "Arc and angle",
        "Circle: arcs and angles",
        "Angle from arcs",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            theorem = "inscribed"
        elif level <= 4:
            theorem = "tangent_chord"
        elif level <= 7:
            theorem = "secant_secant"
        else:
            theorem = "mixed"
        return {
            "n_lines":           2 + level // 2,      # 2..6
            "theorem_type":      theorem,
            "use_fractional":    level >= 7,
            "tight_distractors": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 401)
        self._primary_complexity_feature = cfg["n_lines"]

        for _ in range(25):
            try:
                th = cfg["theorem_type"]
                if th == "inscribed":
                    r = self._inscribed_central(sub_rng, cfg)
                elif th == "tangent_chord":
                    r = self._tangent_chord(sub_rng, cfg)
                elif th == "secant_secant":
                    r = self._secant_secant(sub_rng, cfg)
                else:  # mixed
                    r = self._mixed_arc_addition(sub_rng, cfg)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    def _inscribed_central(self, rng, cfg):
        """Central angle = 2 × inscribed OR inscribed = arc/2.

        All angles are kept EVEN so that halving produces a clean integer
        answer — the env grader requires an integer.
        """
        direction = rng.choice(["central_to_inscribed", "inscribed_to_central",
                                "arc_to_inscribed"])
        if direction == "central_to_inscribed":
            central = rng.randrange(40, 161, 2)
            inscribed = central // 2
            given = {"central_angle": central}
            ans = inscribed
            qvar = "find inscribed angle"
        elif direction == "inscribed_to_central":
            inscribed = rng.randint(20, 80)
            central = 2 * inscribed
            given = {"inscribed_angle": inscribed}
            ans = central
            qvar = "find central angle"
        else:
            arc = rng.randrange(40, 161, 2)
            inscribed = arc // 2
            given = {"arc": arc}
            ans = inscribed
            qvar = "find inscribed angle from arc"
        return self._finalize(rng, cfg, answer=ans, variant="inscribed",
                              given=given, qvar=qvar)

    def _tangent_chord(self, rng, cfg):
        """Tangent-chord angle = half the intercepted arc."""
        arc = rng.randrange(40, 181, 2)
        angle = arc // 2
        direction = rng.choice(["arc_to_angle", "angle_to_arc"])
        if direction == "arc_to_angle":
            given = {"arc": arc}
            ans = angle
            qvar = "find tangent-chord angle"
        else:
            given = {"tc_angle": angle}
            ans = arc
            qvar = "find intercepted arc"
        return self._finalize(rng, cfg, answer=ans, variant="tangent_chord",
                              given=given, qvar=qvar)

    def _secant_secant(self, rng, cfg):
        """Two secants from external point: angle = (far arc - near arc) / 2."""
        # Ensure (far - near) is even so the halving gives an integer answer.
        far = rng.randrange(100, 201, 2)
        near = rng.randrange(20, 81, 2)
        if far <= near + 10:
            return None
        angle = (far - near) // 2
        if angle <= 5:
            return None
        given = {"far_arc": far, "near_arc": near}
        ans = angle
        return self._finalize(rng, cfg, answer=ans, variant="secant_secant",
                              given=given, qvar="find external angle")

    def _mixed_arc_addition(self, rng, cfg):
        """Two secants + arc addition: 3 arcs labeled, find arc4 = 360 - sum,
        then apply secant-secant formula. All arcs kept even AND <=180 so the
        figure is geometrically valid (no reflex arcs)."""
        for _ in range(40):
            a1 = rng.randrange(50, 121, 2)
            a2 = rng.randrange(30, 71, 2)
            a4 = rng.randrange(50, 121, 2)
            a3 = 360 - a1 - a2 - a4
            if a3 < 20 or a3 > 130 or a3 % 2 != 0:
                continue
            far, near = a1, a3
            if far <= near + 10:
                continue
            angle = (far - near) // 2
            if angle <= 5:
                continue
            given = {"a1": a1, "a2": a2, "a4": a4}
            return self._finalize(rng, cfg, answer=angle, variant="mixed",
                                  given=given,
                                  qvar="chain arc addition + secant")
        return None

    # ------------------------------------------------------------------ #
    def _finalize(self, rng, cfg, answer, variant, given, qvar):
        if cfg["use_fractional"] and isinstance(answer, float) and \
                abs(answer - round(answer)) > 0.05:
            ans_str = f"{answer:.1f}"
        else:
            ans_str = str(int(round(answer)))
        try:
            v = float(ans_str)
            if v <= 0 or v >= 360:
                return None
        except ValueError:
            return None

        # Question text — numeric values are labeled on the image.
        if variant == "inscribed":
            if "central_angle" in given:
                q = ("In the circle shown, a central angle and inscribed "
                     "angle subtend the same arc. Using the central angle "
                     "value labeled in the figure, find the inscribed "
                     "angle in degrees.")
            elif "inscribed_angle" in given:
                q = ("In the circle shown, an inscribed angle is labeled. "
                     "Using its value labeled in the figure, find the "
                     "central angle subtending the same arc in degrees.")
            else:
                q = ("In the circle shown, an arc is labeled with its "
                     "measure. Using the arc value labeled in the figure, "
                     "find the inscribed angle that subtends this arc in "
                     "degrees.")
        elif variant == "tangent_chord":
            if "arc" in given:
                q = ("A tangent meets a chord at the point of tangency, "
                     "as shown. Using the intercepted arc value labeled "
                     "in the figure, find the tangent-chord angle in "
                     "degrees.")
            else:
                q = ("A tangent meets a chord at the point of tangency, "
                     "as shown. Using the tangent-chord angle labeled in "
                     "the figure, find the intercepted arc in degrees.")
        elif variant == "secant_secant":
            q = ("Two secants from an external point P intersect the "
                 "circle, as shown. Using the far arc and near arc "
                 "values labeled in the figure, find the angle at P "
                 "(external angle = (far - near) / 2) in degrees.")
        else:  # mixed
            q = ("Two secants from external point P cut the circle into "
                 "4 arcs (a1, a2, a3, a4), as shown. Using the three arc "
                 "values labeled in the figure, find the external angle "
                 "at P (use arc-sum a1+a2+a3+a4 = 360° to get a3, then "
                 "the secant-secant formula).")
        # Answer may be decimal (.5) when use_fractional is on; only say
        # "integer" when we know the answer is integer-valued.
        if isinstance(answer, float) and abs(answer - round(answer)) > 0.05 and cfg["use_fractional"]:
            q += " Answer with a number rounded to 1 decimal place."
        else:
            q += " Answer with a single integer."

        img = self._render(variant, given, cfg)
        return q, ans_str, img

    # ------------------------------------------------------------------ #
    def _pt(self, cx, cy, r, deg):
        rad = math.radians(deg)
        return (cx + r * math.cos(rad), cy + r * math.sin(rad))

    def _render(self, variant, given, cfg):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.3 * sc, 6.0 * sc))
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

        n_lines = cfg["n_lines"]

        def fmt(v):
            return f"{v:.1f}" if cfg["use_fractional"] else f"{int(round(v))}"

        if variant == "inscribed":
            # inscribed angle at C subtending arc AB.
            a_deg = rot_off
            b_deg = rot_off + 110
            c_deg = rot_off + 240
            A = self._pt(cx, cy, R, a_deg)
            B = self._pt(cx, cy, R, b_deg)
            C = self._pt(cx, cy, R, c_deg)
            ax.plot([A[0], C[0]], [A[1], C[1]],
                    color=palette[3 % len(palette)], linewidth=lw)
            ax.plot([B[0], C[0]], [B[1], C[1]],
                    color=palette[3 % len(palette)], linewidth=lw)
            if "central_angle" in given or "arc" in given:
                # Show radii OA and OB (central angle)
                ax.plot([cx, A[0]], [cy, A[1]],
                        color=palette[4 % len(palette)], linewidth=lw,
                        linestyle="--")
                ax.plot([cx, B[0]], [cy, B[1]],
                        color=palette[4 % len(palette)], linewidth=lw,
                        linestyle="--")
            for p, l in [(A, "A"), (B, "B"), (C, "C")]:
                ax.plot(p[0], p[1], "o", color=palette[0], markersize=5)
                ax.text(p[0] * 1.12, p[1] * 1.12, l,
                        fontsize=fs + 1, fontweight="bold",
                        family=ff, color=line_color)
            if "central_angle" in given:
                ax.text(cx + 0.3, cy + 0.2,
                        f"{fmt(given['central_angle'])}°",
                        fontsize=fs - 1, color="#2e7d32",
                        fontweight="bold")
                ax.text(C[0] * 0.6, C[1] * 0.6, "x°",
                        fontsize=fs + 1, color="#c0392b", fontweight="bold")
            elif "arc" in given:
                arc_patch = mpatches.Arc((cx, cy), 2 * R, 2 * R, angle=0,
                                         theta1=a_deg, theta2=b_deg,
                                         color="red", linewidth=3, alpha=0.55)
                ax.add_patch(arc_patch)
                mp = self._pt(cx, cy, R + 0.5, (a_deg + b_deg) / 2)
                ax.text(mp[0], mp[1], f"arc = {fmt(given['arc'])}°",
                        fontsize=fs - 1, color="#1565c0",
                        fontweight="bold", ha="center")
                ax.text(C[0] * 0.6, C[1] * 0.6, "x°",
                        fontsize=fs + 1, color="#c0392b", fontweight="bold")
            else:  # inscribed given, find central
                ax.text(C[0] * 0.6, C[1] * 0.6,
                        f"{fmt(given['inscribed_angle'])}°",
                        fontsize=fs - 1, color="#2e7d32",
                        fontweight="bold")
                ax.text(cx + 0.3, cy + 0.2, "x°",
                        fontsize=fs + 1, color="#c0392b", fontweight="bold")
        elif variant == "tangent_chord":
            # tangent at P, chord PQ
            P_deg = rot_off
            Q_deg = rot_off + 120
            P = self._pt(cx, cy, R, P_deg)
            Q = self._pt(cx, cy, R, Q_deg)
            # tangent at P: perpendicular to OP
            tx = -P[1] + cy
            ty = P[0] - cx
            nrm = math.hypot(tx, ty) + 1e-9
            tx, ty = tx / nrm, ty / nrm
            T1 = (P[0] - 2.2 * tx, P[1] - 2.2 * ty)
            T2 = (P[0] + 2.2 * tx, P[1] + 2.2 * ty)
            ax.plot([T1[0], T2[0]], [T1[1], T2[1]],
                    color=palette[4 % len(palette)], linewidth=lw)
            ax.plot([P[0], Q[0]], [P[1], Q[1]],
                    color=palette[3 % len(palette)], linewidth=lw)
            for p, l in [(P, "P"), (Q, "Q")]:
                ax.plot(p[0], p[1], "o", color=palette[0], markersize=5)
                ax.text(p[0] * 1.12, p[1] * 1.12, l,
                        fontsize=fs + 1, fontweight="bold",
                        family=ff, color=line_color)
            if "arc" in given:
                arc_patch = mpatches.Arc((cx, cy), 2 * R, 2 * R, angle=0,
                                         theta1=P_deg, theta2=Q_deg,
                                         color="red", linewidth=3, alpha=0.55)
                ax.add_patch(arc_patch)
                mp = self._pt(cx, cy, R + 0.5, (P_deg + Q_deg) / 2)
                ax.text(mp[0], mp[1], f"arc = {fmt(given['arc'])}°",
                        fontsize=fs - 1, color="#1565c0",
                        fontweight="bold", ha="center")
                ax.text(P[0] - 0.5, P[1] - 0.3, "x°",
                        fontsize=fs + 1, color="#c0392b", fontweight="bold")
            else:  # tc_angle given, find arc
                ax.text(P[0] - 0.35, P[1] - 0.3,
                        f"{fmt(given['tc_angle'])}°",
                        fontsize=fs - 1, color="#2e7d32", fontweight="bold")
                mp = self._pt(cx, cy, R + 0.5, (P_deg + Q_deg) / 2)
                ax.text(mp[0], mp[1], "arc = x°",
                        fontsize=fs + 1, color="#c0392b",
                        fontweight="bold", ha="center")
        elif variant == "secant_secant":
            # External point P, two secants through circle
            # Use deg for near/far arc positioning
            near_arc = given["near_arc"]
            far_arc = given["far_arc"]
            # Layout: P to the right of circle, secants cross through
            # circle hitting two chords.
            P = (cx + R * 2.4, cy)
            # upper secant: enters at top-right region, exits upper-left
            # lower secant: enters at bottom-right, exits lower-left
            enter1 = self._pt(cx, cy, R, 40)
            exit1 = self._pt(cx, cy, R, 140)
            enter2 = self._pt(cx, cy, R, -40)
            exit2 = self._pt(cx, cy, R, -140)
            ax.plot([P[0], exit1[0]], [P[1], exit1[1]],
                    color=palette[3 % len(palette)], linewidth=lw)
            ax.plot([P[0], exit2[0]], [P[1], exit2[1]],
                    color=palette[5 % len(palette)], linewidth=lw)
            ax.plot(P[0], P[1], "o", color=palette[0], markersize=6)
            ax.text(P[0] + 0.1, P[1] + 0.2, "P",
                    fontsize=fs + 1, fontweight="bold",
                    family=ff, color=line_color)
            # far arc between exit1 and exit2 (going through top/far side)
            far_mid = self._pt(cx, cy, R + 0.55, 180)
            ax.text(far_mid[0], far_mid[1], f"far arc = {fmt(far_arc)}°",
                    fontsize=fs - 1, color="#1565c0", fontweight="bold",
                    ha="center")
            near_mid = self._pt(cx, cy, R + 0.55, 0)
            ax.text(near_mid[0] + 0.3, near_mid[1],
                    f"near = {fmt(near_arc)}°",
                    fontsize=fs - 1, color="#1565c0", fontweight="bold",
                    ha="left")
            ax.text(P[0] + 0.25, P[1] + 0.55, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        elif variant == "mixed":
            # Show external angle P with three arcs labeled
            P = (cx + R * 2.4, cy)
            exit1 = self._pt(cx, cy, R, 140)
            exit2 = self._pt(cx, cy, R, -140)
            enter1 = self._pt(cx, cy, R, 40)
            enter2 = self._pt(cx, cy, R, -40)
            ax.plot([P[0], exit1[0]], [P[1], exit1[1]],
                    color=palette[3 % len(palette)], linewidth=lw)
            ax.plot([P[0], exit2[0]], [P[1], exit2[1]],
                    color=palette[5 % len(palette)], linewidth=lw)
            ax.plot(P[0], P[1], "o", color=palette[0], markersize=6)
            ax.text(P[0] + 0.1, P[1] + 0.2, "P",
                    fontsize=fs + 1, fontweight="bold",
                    family=ff, color=line_color)
            # a1 (far) arc, a2 (side), a4 (other side), a3=near (unknown)
            p_top = self._pt(cx, cy, R + 0.55, 180)
            p_side1 = self._pt(cx, cy, R + 0.55, 90)
            p_side2 = self._pt(cx, cy, R + 0.55, -90)
            p_near = self._pt(cx, cy, R + 0.55, 0)
            ax.text(p_top[0], p_top[1], f"a1 = {fmt(given['a1'])}°",
                    fontsize=fs - 1, color="#1565c0", fontweight="bold",
                    ha="center")
            ax.text(p_side1[0], p_side1[1], f"a2 = {fmt(given['a2'])}°",
                    fontsize=fs - 1, color="#1565c0", fontweight="bold",
                    ha="center")
            ax.text(p_side2[0], p_side2[1], f"a4 = {fmt(given['a4'])}°",
                    fontsize=fs - 1, color="#1565c0", fontweight="bold",
                    ha="center")
            ax.text(p_near[0] + 0.4, p_near[1], "a3 = ?",
                    fontsize=fs - 1, color="#c0392b", fontweight="bold",
                    ha="left")
            ax.text(P[0] + 0.25, P[1] + 0.55, "x°",
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")

        # NOTE: previously drew `extras_needed` dotted decoration chords here
        # for "visual clutter" at high levels. They were not part of the
        # geometry and confused the model (tried to reason about them).
        # Removed 2026-04-16 — difficulty comes from the theorem chain, not
        # irrelevant visual clutter.

        pad = 1.5
        ax.set_xlim(-R - pad, R + pad + 2.2)
        ax.set_ylim(-R - pad, R + pad)
        ax.set_title(self._rng.choice(self._TITLE_VARIANTS),
                     fontsize=fs + 2, fontweight="bold", pad=8, family=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _count_drawn_lines(variant):
        if variant == "inscribed":
            return 2
        if variant == "tangent_chord":
            return 2
        if variant == "secant_secant":
            return 2
        return 2
