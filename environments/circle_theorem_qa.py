"""
Circle theorem QA — inscribed / central angles, cyclic quads, tangent-chord,
power-of-a-point, secant-secant, tangent-secant.

Round 2 fixes:
  - Values on IMAGE not question text: the question refers to "as labeled"
    but the question itself does not restate the numeric givens.
  - L0 is structurally simple: pick an inscribed angle (answer = arc / 2)
    or central-vs-inscribed (answer = 2 × labeled).
  - L9 is multi-step: two-chord systems, tangent-secant, arc chain, cyclic
    with three unknowns.
  - Per-seed jitter: circle position, point angles, colors shuffled.
  - 4+ question phrasings per variant.
"""
import random
import math
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class CircleTheoremQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "circle_theorem"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    # L9: three-step chain titles and questions — added 2026-04-17
    _L9_TITLES = ["Three-Step Chain", "Cyclic & Tangent Chain",
                  "Compound Circle Theorem"]
    _L9_QUESTIONS = [
        "Read the labeled arc and tangent-chord angle from the figure. "
        "Step 1: compute the inscribed angle corresponding to the "
        "labeled arc. Step 2: use the cyclic-quad opposite-angle relation "
        "to find the opposite angle. Step 3: the tangent-chord angle "
        "subtracted from the step-2 result gives the target x.",
        "Three steps: (1) inscribed angle from labeled arc; (2) its "
        "cyclic-quad opposite; (3) subtract the labeled tangent-chord "
        "angle. Report x in degrees.",
    ]

    _TITLE_POOL = {
        "inscribed_angle": ["Inscribed Angle Theorem", "Inscribed Angle",
                            "Arc-Inscribed Angle"],
        "central_vs_inscribed": ["Central vs Inscribed Angle",
                                  "Central Angle", "Same-Arc Angle Relationship"],
        "cyclic_quad": ["Cyclic Quadrilateral", "Opposite-Angle Sum in a Cyclic Quad"],
        "tangent_right_angle": ["Tangent-Radius Right Angle", "Tangent & Radius"],
        "chord_perpendicular": ["Perpendicular from Center to Chord",
                                 "Chord–Radius–Distance"],
        "cyclic_quad_two_unknowns": ["Cyclic Quad — Two Unknowns",
                                      "System of Angles"],
        "chord_chain": ["Intersecting Chords — Power of a Point",
                         "Power of a Point (Chord Form)"],
        "inscribed_plus_tangent": ["Tangent-Chord + Inscribed Chain",
                                    "Two-Step: Tangent-Chord → Inscribed"],
        "secant_secant": ["Two Secants — Power of a Point",
                           "External Secant-Secant"],
        "tangent_secant": ["Tangent-Secant — Power of a Point",
                            "Tangent-Secant External Angle"],
        "arc_chain": ["Arc Chain", "Adding Arcs"],
    }

    _QUESTION_POOL = {
        "inscribed_angle": [
            "Use the arc labeled in the figure. Find the inscribed angle x at point C (in degrees).",
            "The image labels an arc. What is the inscribed angle at C intercepting that arc?",
            "Based on the labeled arc, compute the inscribed angle x (degrees).",
            "The inscribed angle at C intercepts the labeled arc — find its measure in degrees.",
        ],
        "central_vs_inscribed": [
            "Use the inscribed angle labeled in the figure. Find the central angle x (AOB) in degrees.",
            "The image shows an inscribed angle. What is the corresponding central angle (degrees)?",
            "Read the labeled inscribed angle on the figure. Compute the central angle.",
            "The central angle is twice the inscribed angle on the same arc. Use the image label to find x.",
        ],
        "cyclic_quad": [
            "ABCD is cyclic. Use the labeled angle at A to find ∠C (degrees).",
            "Using the labeled angle in the figure, find the opposite angle in this cyclic quadrilateral.",
            "Opposite angles in a cyclic quadrilateral sum to 180°. Find the missing angle from the labeled one.",
            "Based on the labeled ∠A, compute the unknown ∠C.",
        ],
        "tangent_right_angle": [
            "The tangent meets the radius at 90°. Using the labeled angle in the figure, find x (degrees).",
            "Using the labeled acute angle in the image, find the complementary angle x.",
            "A tangent and a chord form a right angle with the radius; use the label to find x.",
            "Given the labeled angle, compute x = 90° − (labeled angle).",
        ],
        "chord_perpendicular": [
            "Use the radius and half-chord labeled in the figure. Find the distance from center to chord (round to 2 decimals).",
            "Compute d via Pythagoras: d = sqrt(r² − (half-chord)²), using the labeled values.",
            "The figure labels r and the half-chord. Find the perpendicular distance from the center to the chord.",
            "From the figure, read r and the half-chord, then compute the distance d.",
        ],
        "cyclic_quad_two_unknowns": [
            "ABCD is a cyclic quadrilateral. From the labeled angle and the relation shown, find ∠B (degrees).",
            "Solve the system using the labeled ∠A and the ∠B = k · ∠D relationship on the image.",
            "Using the labels in the figure, compute ∠B.",
            "Read ∠A and the B/D ratio from the image, then find ∠B.",
        ],
        "chord_chain": [
            "Use the lengths labeled in the figure and the intersecting-chords theorem (AP · PB = CP · PD) to find PD.",
            "From the labeled chord segments, compute the unknown length PD.",
            "Apply power-of-a-point: AP × PB = CP × PD, using the image labels.",
            "The image labels AP, PB, CP. Find PD.",
        ],
        "inscribed_plus_tangent": [
            "A tangent-chord angle is labeled. Using the resulting arc and the complementary arc, find the inscribed angle x on the MAJOR arc.",
            "Read the tangent-chord angle from the figure. Compute the inscribed angle on the other arc (degrees).",
            "Given the labeled tangent-chord angle, use two theorems to find x (inscribed angle on the supplementary arc).",
            "The labeled tangent-chord angle gives the intercepted arc. Find the inscribed angle intercepting the other (major) arc.",
        ],
        "secant_secant": [
            "Two secants meet at an external point P. Use the labeled outside + whole-lengths and the power-of-a-point relation to find the missing length.",
            "From the figure labels, compute the unknown external secant segment.",
            "Use PA · PB = PC · PD with the labeled values.",
            "The image labels three of the four secant segments; find the fourth.",
        ],
        "tangent_secant": [
            "A tangent and a secant meet at P. Use (tangent)² = PA · PB with the labeled values to find the tangent length.",
            "From the labels in the figure, compute the tangent length.",
            "Apply the tangent-secant power-of-a-point theorem using the labels.",
            "The tangent length satisfies t² = PA · PB; find t using the labeled numbers.",
        ],
        "arc_chain": [
            "The figure labels two arcs on the circle. Find the arc remaining to complete 360°.",
            "Subtract the labeled arcs from 360° to find the missing arc.",
            "Use the labeled arc values to find the unlabeled arc.",
            "The sum of all arcs is 360°; from the labels, compute the remaining arc.",
        ],
    }

    # ------------------------------------------------------------------ #
    def _point_on_circle(self, cx, cy, r, angle_deg):
        rad = math.radians(angle_deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesign 2026-04-17: previous L3 dipped to 0.6 because
        # tangent_right_angle + cyclic_quad both require a subtraction step.
        # L9 was 1.0 because chord_chain/secant_secant are direct arithmetic
        # once you apply the power-of-a-point formula. Reshuffle so that L9
        # uses the hardest multi-step variants (inscribed_plus_tangent,
        # cyclic_quad_two_unknowns).
        if level == 0:
            return {"ptypes": ["inscribed_angle", "arc_chain"]}
        if level == 1:
            return {"ptypes": ["inscribed_angle", "central_vs_inscribed",
                               "arc_chain"]}
        if level <= 3:
            return {"ptypes": ["central_vs_inscribed", "arc_chain",
                               "tangent_right_angle"]}
        if level <= 5:
            return {"ptypes": ["cyclic_quad", "tangent_right_angle",
                               "chord_perpendicular"]}
        if level <= 7:
            return {"ptypes": ["chord_perpendicular",
                               "chord_chain", "tangent_secant",
                               "secant_secant"]}
        if level == 8:
            return {"ptypes": ["inscribed_plus_tangent",
                               "cyclic_quad_two_unknowns"]}
        # Iter 3 (2026-04-17): L9 = three_step_chain overshot (0.05
        # pass-rate). Downgrade to 2-step variants (inscribed_plus_tangent
        # + cyclic_quad_two_unknowns) with slightly harder distractor
        # placement — same families as L8 but rng tightened implicitly
        # via level*37 seed.
        # Iter 4 (2026-04-17): iter-3 L9 (2-step pool) now at 0.95 (too
        # easy). Re-introduce three_step_chain BUT mix it with the 2-step
        # variants so the mean pass-rate lands near 0.5 rather than 0.05.
        return {"ptypes": ["inscribed_plus_tangent",
                           "cyclic_quad_two_unknowns",
                           "three_step_chain"]}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 929)
        for _ in range(25):
            ptype = rng.choice(cfg["ptypes"])
            try:
                r = self._dispatch(ptype, rng)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _dispatch(self, ptype, rng):
        if ptype == "inscribed_angle":
            return self._inscribed_angle(rng)
        if ptype == "central_vs_inscribed":
            return self._central_inscribed(rng)
        if ptype == "cyclic_quad":
            return self._cyclic_quad(rng)
        if ptype == "tangent_right_angle":
            return self._tangent_right_angle(rng)
        if ptype == "chord_perpendicular":
            return self._chord_perpendicular(rng)
        if ptype == "cyclic_quad_two_unknowns":
            return self._cyclic_quad_two_unknowns(rng)
        if ptype == "chord_chain":
            return self._chord_chain(rng)
        if ptype == "inscribed_plus_tangent":
            return self._inscribed_plus_tangent(rng)
        if ptype == "secant_secant":
            return self._secant_secant(rng)
        if ptype == "tangent_secant":
            return self._tangent_secant(rng)
        if ptype == "arc_chain":
            return self._arc_chain(rng)
        if ptype == "three_step_chain":
            return self._three_step_chain(rng)
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _make_fig(self, rng):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        fs = style["font_size_base"] + 1
        return fig, ax, style, palette, fs

    def _get_q(self, key, rng):
        return rng.choice(self._QUESTION_POOL[key])

    def _title(self, key, rng):
        return rng.choice(self._TITLE_POOL[key])

    # ------------------------------------------------------------------ #
    # Variants
    # ------------------------------------------------------------------ #
    def _inscribed_angle(self, rng):
        arc = rng.choice(list(range(40, 161, 4)))
        inscribed = arc // 2
        cx, cy, r = 3, 3, 2.4 + rng.uniform(-0.2, 0.3)
        a1 = rng.randint(10, 80)
        a2 = a1 + arc
        a3 = a2 + rng.randint(60, 150)
        fig, ax, style, palette, fs = self._make_fig(rng)
        circle = plt.Circle((cx, cy), r, fill=False, edgecolor=palette[0],
                            linewidth=2)
        ax.add_patch(circle)
        pA = self._point_on_circle(cx, cy, r, a1)
        pB = self._point_on_circle(cx, cy, r, a2)
        pC = self._point_on_circle(cx, cy, r, a3)
        ax.plot([pA[0], pC[0]], [pA[1], pC[1]], color=palette[2], linewidth=1.8)
        ax.plot([pB[0], pC[0]], [pB[1], pC[1]], color=palette[2], linewidth=1.8)
        arc_patch = mpatches.Arc((cx, cy), 2 * r, 2 * r, angle=0,
                                  theta1=a1, theta2=a2,
                                  color=palette[4], linewidth=3.5, alpha=0.7)
        ax.add_patch(arc_patch)
        ax.annotate(f"arc AB = {arc}°",
                    xy=self._point_on_circle(cx, cy, r + 0.45, (a1 + a2) / 2),
                    fontsize=fs, color="#c0392b", fontweight="bold", ha="center")
        ax.annotate("x = ?", xy=(pC[0] + 0.15, pC[1] + 0.15),
                    fontsize=fs + 2, color="#27ae60", fontweight="bold")
        for p, lbl in [(pA, "A"), (pB, "B"), (pC, "C")]:
            ax.plot(*p, "o", color=style["geo_line_color"], markersize=5)
            d = ((p[0] - cx) * 0.18, (p[1] - cy) * 0.18)
            ax.annotate(lbl, xy=p, xytext=(p[0] + d[0], p[1] + d[1]),
                        fontsize=fs + 1, fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("inscribed_angle", rng),
                      fontsize=fs + 2, fontweight="bold")
        q = self._get_q("inscribed_angle", rng)
        return q, str(inscribed), self.fig_to_pil(fig, dpi=style["dpi"])

    def _central_inscribed(self, rng):
        inscribed = rng.randint(20, 80)
        central = 2 * inscribed
        cx, cy, r = 3, 3, 2.4 + rng.uniform(-0.2, 0.3)
        fig, ax, style, palette, fs = self._make_fig(rng)
        circle = plt.Circle((cx, cy), r, fill=False, edgecolor=palette[0],
                            linewidth=2)
        ax.add_patch(circle)
        ax.plot(cx, cy, "k+", markersize=8)
        a1 = rng.randint(10, 60)
        pA = self._point_on_circle(cx, cy, r, a1)
        pB = self._point_on_circle(cx, cy, r, a1 + central)
        pC = self._point_on_circle(cx, cy, r, a1 + central + rng.randint(60, 150))
        ax.plot([cx, pA[0]], [cy, pA[1]], color=palette[4], linewidth=1.8)
        ax.plot([cx, pB[0]], [cy, pB[1]], color=palette[4], linewidth=1.8)
        ax.plot([pA[0], pC[0]], [pA[1], pC[1]], color=palette[2], linewidth=1.8)
        ax.plot([pB[0], pC[0]], [pB[1], pC[1]], color=palette[2], linewidth=1.8)
        ax.annotate(f"{inscribed}°", xy=(pC[0] + 0.15, pC[1] + 0.15),
                    fontsize=fs + 1, color="#1a5276", fontweight="bold")
        ax.annotate("x = ?", xy=(cx + 0.2, cy - 0.3),
                    fontsize=fs + 2, color="#c0392b", fontweight="bold")
        for p, lbl in [(pA, "A"), (pB, "B"), (pC, "C"), ((cx, cy), "O")]:
            ax.plot(*p, "o", color=style["geo_line_color"], markersize=5)
            ax.annotate(lbl, xy=p, xytext=(p[0] + 0.15, p[1] + 0.15),
                        fontsize=fs + 1, fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("central_vs_inscribed", rng),
                      fontsize=fs + 2, fontweight="bold")
        q = self._get_q("central_vs_inscribed", rng)
        return q, str(central), self.fig_to_pil(fig, dpi=style["dpi"])

    def _cyclic_quad(self, rng):
        angle_a = rng.randint(50, 130)
        angle_c = 180 - angle_a
        cx, cy, r = 3, 3, 2.3 + rng.uniform(-0.1, 0.4)
        angs = sorted(rng.sample(range(0, 360, 15), 4))
        fig, ax, style, palette, fs = self._make_fig(rng)
        circle = plt.Circle((cx, cy), r, fill=False, edgecolor=palette[0],
                            linewidth=2)
        ax.add_patch(circle)
        pts = [self._point_on_circle(cx, cy, r, a) for a in angs]
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=palette[2], linewidth=1.8)
        for p, lbl in zip(pts, "ABCD"):
            ax.plot(*p, "o", color=style["geo_line_color"], markersize=5)
            d = ((p[0] - cx) * 0.15, (p[1] - cy) * 0.15)
            ax.annotate(lbl, xy=p, xytext=(p[0] + d[0], p[1] + d[1]),
                        fontsize=fs + 2, fontweight="bold")
        ax.annotate(f"∠A = {angle_a}°", xy=(pts[0][0] - 0.5, pts[0][1]),
                    fontsize=fs, color="#1a5276", fontweight="bold")
        ax.annotate("∠C = ?", xy=(pts[2][0] + 0.2, pts[2][1]),
                    fontsize=fs + 2, color="#c0392b", fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("cyclic_quad", rng),
                      fontsize=fs + 2, fontweight="bold")
        q = self._get_q("cyclic_quad", rng)
        return q, str(angle_c), self.fig_to_pil(fig, dpi=style["dpi"])

    def _tangent_right_angle(self, rng):
        given = rng.randint(20, 70)
        answer = 90 - given
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 2.0
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        tp = (cx, cy - r)
        ax.plot(*tp, "o", color=style["geo_line_color"], markersize=6)
        ax.plot([cx - 2.5, cx + 2.5], [cy - r, cy - r],
                color=palette[4], linewidth=2)
        ax.plot([cx, tp[0]], [cy, tp[1]], color=palette[2], linewidth=1.8)
        ax.annotate(f"{given}°", xy=(cx + 0.3, cy - r + 0.3),
                    fontsize=fs + 1, color="#1a5276", fontweight="bold")
        ax.annotate("x = ?", xy=(cx - 0.8, cy - r + 0.5),
                    fontsize=fs + 2, color="#c0392b", fontweight="bold")
        ax.annotate("tangent", xy=(cx + 1.5, cy - r - 0.3),
                    fontsize=fs - 1, color="#c0392b")
        ax.annotate("90°", xy=(cx + 0.1, cy - r / 2 - 0.2),
                    fontsize=fs - 1, color="gray")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("tangent_right_angle", rng),
                      fontsize=fs + 2, fontweight="bold")
        q = self._get_q("tangent_right_angle", rng)
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    def _chord_perpendicular(self, rng):
        r_val = rng.randint(5, 13)
        half = rng.randint(2, r_val - 1)
        dist = round(math.sqrt(r_val ** 2 - half ** 2), 2)
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 2.2
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        chord_y = cy - 0.8
        half_len = math.sqrt(max(r ** 2 - (chord_y - cy) ** 2, 0.01))
        ax.plot([cx - half_len, cx + half_len],
                [chord_y, chord_y], color=palette[2], linewidth=2)
        ax.plot([cx, cx], [cy, chord_y], color="#c0392b",
                linewidth=1.8, linestyle="--")
        ax.plot([cx, cx + half_len], [cy, chord_y], color=palette[4],
                linewidth=1.8)
        ax.annotate(f"r = {r_val}", xy=(cx + 0.6, (cy + chord_y) / 2),
                    fontsize=fs, color="#1a5276", fontweight="bold")
        ax.annotate(f"half = {half}", xy=(cx + 0.2, chord_y - 0.3),
                    fontsize=fs - 1, color="#1a5276", fontweight="bold")
        ax.annotate("d = ?", xy=(cx - 0.8, (cy + chord_y) / 2),
                    fontsize=fs + 2, color="#c0392b", fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("chord_perpendicular", rng),
                      fontsize=fs + 1, fontweight="bold")
        q = self._get_q("chord_perpendicular", rng)
        return q, str(dist), self.fig_to_pil(fig, dpi=style["dpi"])

    def _cyclic_quad_two_unknowns(self, rng):
        angle_a = rng.randint(50, 110)
        k = rng.choice([2, 3])
        angle_d = 180 // (k + 1)
        angle_b = k * angle_d
        if angle_d <= 10 or angle_b >= 170:
            return None
        cx, cy, r = 3, 3, 2.4
        angs = sorted(rng.sample(range(0, 360, 15), 4))
        fig, ax, style, palette, fs = self._make_fig(rng)
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        pts = [self._point_on_circle(cx, cy, r, a) for a in angs]
        for i in range(4):
            j = (i + 1) % 4
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=palette[2], linewidth=1.8)
        for p, lbl in zip(pts, "ABCD"):
            ax.plot(*p, "o", color=style["geo_line_color"], markersize=5)
            d = ((p[0] - cx) * 0.15, (p[1] - cy) * 0.15)
            ax.annotate(lbl, xy=p, xytext=(p[0] + d[0], p[1] + d[1]),
                        fontsize=fs + 2, fontweight="bold")
        ax.annotate(f"∠A = {angle_a}°", xy=(pts[0][0] - 0.5, pts[0][1]),
                    fontsize=fs - 1, color="#1a5276", fontweight="bold")
        ax.annotate(f"∠B = {k} · ∠D", xy=(pts[1][0] + 0.2, pts[1][1] + 0.15),
                    fontsize=fs - 1, color="#1a5276")
        ax.annotate("∠B = ?", xy=(pts[1][0] - 0.6, pts[1][1] - 0.35),
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("cyclic_quad_two_unknowns", rng),
                      fontsize=fs + 1, fontweight="bold")
        q = self._get_q("cyclic_quad_two_unknowns", rng)
        return q, str(angle_b), self.fig_to_pil(fig, dpi=style["dpi"])

    def _chord_chain(self, rng):
        ap = rng.randint(2, 8)
        pb = rng.randint(2, 8)
        product = ap * pb
        # choose cp such that product % cp == 0
        divisors = [d for d in range(2, 11) if product % d == 0]
        if not divisors:
            return None
        cp = rng.choice(divisors)
        pd = product // cp
        if pd <= 0 or pd > 30:
            return None
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 2.4
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        # random chord orientation
        offs = rng.randint(0, 60)
        pA = self._point_on_circle(cx, cy, r, 30 + offs)
        pB = self._point_on_circle(cx, cy, r, 210 + offs)
        pC = self._point_on_circle(cx, cy, r, 120 + offs)
        pD = self._point_on_circle(cx, cy, r, 300 + offs)
        ax.plot([pA[0], pB[0]], [pA[1], pB[1]],
                color=palette[2], linewidth=1.8)
        ax.plot([pC[0], pD[0]], [pC[1], pD[1]],
                color=palette[4], linewidth=1.8)
        for p, lbl in [(pA, "A"), (pB, "B"), (pC, "C"), (pD, "D")]:
            ax.plot(*p, "o", color=style["geo_line_color"], markersize=5)
            d = ((p[0] - cx) * 0.18, (p[1] - cy) * 0.18)
            ax.annotate(lbl, xy=p, xytext=(p[0] + d[0], p[1] + d[1]),
                        fontsize=fs + 1, fontweight="bold")
        mid_x = (pA[0] + pB[0]) / 2
        mid_y = (pA[1] + pB[1]) / 2
        ax.plot(mid_x, mid_y, "o", color="#c0392b", markersize=5)
        ax.annotate("P", xy=(mid_x, mid_y),
                    xytext=(mid_x + 0.15, mid_y + 0.15),
                    fontsize=fs + 1, fontweight="bold", color="#c0392b")
        ax.annotate(f"AP={ap}", xy=(mid_x - 0.8, mid_y + 0.3),
                    fontsize=fs, color="#1a5276", fontweight="bold")
        ax.annotate(f"PB={pb}", xy=(mid_x + 0.2, mid_y - 0.3),
                    fontsize=fs, color="#1a5276", fontweight="bold")
        ax.annotate(f"CP={cp}", xy=(mid_x - 0.8, mid_y - 0.3),
                    fontsize=fs, color="#1a5276", fontweight="bold")
        ax.annotate("PD=?", xy=(mid_x + 0.3, mid_y + 0.3),
                    fontsize=fs + 2, color="#c0392b", fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("chord_chain", rng),
                      fontsize=fs + 1, fontweight="bold")
        q = self._get_q("chord_chain", rng)
        return q, str(pd), self.fig_to_pil(fig, dpi=style["dpi"])

    def _inscribed_plus_tangent(self, rng):
        arc = rng.choice(list(range(60, 201, 4)))
        tangent_chord_angle = arc // 2
        other_arc = 360 - arc
        other_inscribed = other_arc // 2
        if other_inscribed <= 0 or other_inscribed >= 180:
            return None
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 2.2
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        tp = (cx, cy - r)
        ax.plot(*tp, "o", color=style["geo_line_color"], markersize=6)
        ax.plot([cx - 2.5, cx + 2.5], [cy - r, cy - r],
                color=palette[4], linewidth=2)
        ax.annotate("tangent", xy=(cx + 1.5, cy - r - 0.3),
                    fontsize=fs - 1, color="#c0392b")
        ax.annotate(f"tangent-chord\nangle = {tangent_chord_angle}°",
                    xy=(cx + 0.3, cy - r + 0.2),
                    fontsize=fs - 1, color="#1a5276", fontweight="bold")
        ax.annotate("x = ?", xy=(cx - 1.5, cy + 0.5),
                    fontsize=fs + 2, color="#c0392b", fontweight="bold")
        # Removed the redundant "labeled arc = …" annotation: the tangent-chord
        # angle already determines the minor arc (arc = 2 * tangent-chord). The
        # solver must apply the tangent-chord theorem rather than read the arc
        # directly off the image.
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("inscribed_plus_tangent", rng),
                      fontsize=fs, fontweight="bold")
        q = self._get_q("inscribed_plus_tangent", rng)
        return q, str(other_inscribed), self.fig_to_pil(fig, dpi=style["dpi"])

    def _secant_secant(self, rng):
        # PA * PB = PC * PD  (A, C are on the near side; B, D on the far)
        pa = rng.randint(3, 8)
        pb = rng.randint(pa + 2, pa + 12)
        pc = rng.randint(3, 8)
        # choose pd so product equals pa * pb and pd > pc
        prod = pa * pb
        if pc == 0 or prod % pc != 0:
            # adjust pc to divide prod
            divisors = [d for d in range(3, 10) if prod % d == 0 and d < prod / d + 1]
            if not divisors:
                return None
            pc = rng.choice(divisors)
        pd = prod // pc
        if pd <= pc or pd > 30:
            return None
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 1.9
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        P = (cx - 2.5, cy - 0.4)
        ax.plot(*P, "o", color="#c0392b", markersize=6)
        ax.annotate("P", xy=P, xytext=(P[0] - 0.25, P[1] - 0.2),
                    fontsize=fs + 1, fontweight="bold", color="#c0392b")
        # Line through P, intersecting circle at A, B
        # Use direction
        th1 = math.radians(25)
        dir1 = (math.cos(th1), math.sin(th1))
        th2 = math.radians
        dir2 = (math.cos(th2), math.sin(th2))
        pA = (P[0] + pa * dir1[0], P[1] + pa * dir1[1])
        pB = (P[0] + pb * dir1[0], P[1] + pb * dir1[1])
        pC = (P[0] + pc * dir2[0], P[1] + pc * dir2[1])
        pD = (P[0] + pd * dir2[0], P[1] + pd * dir2[1])
        ax.plot([P[0], pB[0]], [P[1], pB[1]],
                color=palette[2], linewidth=1.7)
        ax.plot([P[0], pD[0]], [P[1], pD[1]],
                color=palette[4], linewidth=1.7)
        for pp, lbl in [(pA, "A"), (pB, "B"), (pC, "C"), (pD, "D")]:
            ax.plot(*pp, "o", color=style["geo_line_color"], markersize=4)
            ax.annotate(lbl, xy=pp, xytext=(pp[0] + 0.08, pp[1] + 0.08),
                        fontsize=fs, fontweight="bold")
        ax.annotate(f"PA = {pa}", xy=(P[0] + 0.2, P[1] + 0.35),
                    fontsize=fs - 1, color="#1a5276")
        ax.annotate(f"PB = {pb}", xy=(P[0] + 2.2, P[1] + 1.0),
                    fontsize=fs - 1, color="#1a5276")
        ax.annotate(f"PC = {pc}", xy=(P[0] + 0.2, P[1] - 0.35),
                    fontsize=fs - 1, color="#1a5276")
        ax.annotate("PD = ?", xy=(P[0] + 2.2, P[1] - 1.2),
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        ax.set_xlim(-0.3, 6.3); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("secant_secant", rng),
                      fontsize=fs + 1, fontweight="bold")
        q = self._get_q("secant_secant", rng)
        return q, str(pd), self.fig_to_pil(fig, dpi=style["dpi"])

    def _tangent_secant(self, rng):
        # t^2 = PA * PB
        # pick perfect square
        pa = rng.randint(2, 6)
        pb = rng.choice([i for i in range(pa + 2, pa + 14)
                          if int(math.isqrt(pa * i)) ** 2 == pa * i])
        t = int(math.isqrt(pa * pb))
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3.2, 3.2, 1.7
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        P = (cx - 2.5, cy - 0.4)
        ax.plot(*P, "o", color="#c0392b", markersize=6)
        ax.annotate("P", xy=P, xytext=(P[0] - 0.25, P[1] - 0.2),
                    fontsize=fs + 1, fontweight="bold", color="#c0392b")
        th1 = math.radians(20)
        dir1 = (math.cos(th1), math.sin(th1))
        pA = (P[0] + pa * dir1[0], P[1] + pa * dir1[1])
        pB = (P[0] + pb * dir1[0], P[1] + pb * dir1[1])
        ax.plot([P[0], pB[0]], [P[1], pB[1]],
                color=palette[2], linewidth=1.7)
        # tangent line
        th2 = math.radians
        dir2 = (math.cos(th2), math.sin(th2))
        T = (P[0] + t * dir2[0], P[1] + t * dir2[1])
        ax.plot([P[0], T[0]], [P[1], T[1]],
                color=palette[4], linewidth=1.7)
        for pp, lbl in [(pA, "A"), (pB, "B"), (T, "T")]:
            ax.plot(*pp, "o", color=style["geo_line_color"], markersize=4)
            ax.annotate(lbl, xy=pp, xytext=(pp[0] + 0.08, pp[1] + 0.08),
                        fontsize=fs, fontweight="bold")
        ax.annotate(f"PA = {pa}", xy=(P[0] + 0.2, P[1] + 0.35),
                    fontsize=fs - 1, color="#1a5276")
        ax.annotate(f"PB = {pb}", xy=(P[0] + 2.2, P[1] + 1.0),
                    fontsize=fs - 1, color="#1a5276")
        ax.annotate("PT = t = ?", xy=(P[0] + 1.8, P[1] - 1.2),
                    fontsize=fs + 1, color="#c0392b", fontweight="bold")
        ax.set_xlim(-0.3, 6.3); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("tangent_secant", rng),
                      fontsize=fs + 1, fontweight="bold")
        q = self._get_q("tangent_secant", rng)
        return q, str(t), self.fig_to_pil(fig, dpi=style["dpi"])

    def _three_step_chain(self, rng):
        """L9: three-step chain. Compute:
          A = arc / 2   (inscribed angle)
          B = 180 - A   (cyclic opposite)
          C = B - t     (subtract labeled tangent-chord angle)
        Answer = C.
        """
        # Constrain so all three intermediate values are well-defined and
        # the final answer is positive.
        for _ in range(20):
            arc = rng.choice(list(range(40, 181, 4)))
            t = rng.randint(20, 70)
            inscribed = arc // 2
            opp = 180 - inscribed
            target = opp - t
            if 10 <= target <= 170:
                break
        else:
            return None
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 2.3
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        # Arc AB (labeled) — an inscribed angle will be derived from this
        # as the first step.
        a1 = rng.randint(20, 60)
        a2 = a1 + arc
        arc_patch = mpatches.Arc((cx, cy), 2 * r, 2 * r, angle=0,
                                  theta1=a1, theta2=a2,
                                  color=palette[4], linewidth=3.5, alpha=0.8)
        ax.add_patch(arc_patch)
        # Two cyclic-quad vertices A, B on the arc; C is the opposite.
        pA = self._point_on_circle(cx, cy, r, a1)
        pB = self._point_on_circle(cx, cy, r, a2)
        # Put C and D on the other side
        pC = self._point_on_circle(cx, cy, r, a2 + rng.randint(60, 120))
        pD = self._point_on_circle(cx, cy, r, a2 + rng.randint(180, 260))
        for i in range(4):
            pts = [pA, pB, pC, pD]
            nxt = pts[(i + 1) % 4]
            cur = pts[i]
            ax.plot([cur[0], nxt[0]], [cur[1], nxt[1]],
                    color=palette[2], linewidth=1.5)
        for p, lbl in [(pA, "A"), (pB, "B"), (pC, "C"), (pD, "D")]:
            ax.plot(*p, "o", color=style["geo_line_color"], markersize=4)
            d = ((p[0] - cx) * 0.15, (p[1] - cy) * 0.15)
            ax.annotate(lbl, xy=p, xytext=(p[0] + d[0], p[1] + d[1]),
                        fontsize=fs, fontweight="bold")
        # Label the arc
        ax.annotate(f"arc AB = {arc}°",
                    xy=self._point_on_circle(cx, cy, r + 0.55, (a1 + a2) / 2),
                    fontsize=fs, color="#c0392b", fontweight="bold",
                    ha="center")
        # Draw a tangent at D with labeled tangent-chord angle t
        # Tangent direction perpendicular to OD at D.
        dx_ = pD[0] - cx
        dy_ = pD[1] - cy
        norm = math.hypot(dx_, dy_)
        tx, ty = -dy_ / norm, dx_ / norm
        t_end1 = (pD[0] + 1.8 * tx, pD[1] + 1.8 * ty)
        t_end2 = (pD[0] - 1.8 * tx, pD[1] - 1.8 * ty)
        ax.plot([t_end1[0], t_end2[0]], [t_end1[1], t_end2[1]],
                color=palette[3], linewidth=2)
        ax.annotate("tangent at D", xy=t_end2,
                    fontsize=fs - 1, color=palette[3])
        ax.annotate(f"tangent-chord = {t}°",
                    xy=(pD[0] + 0.15, pD[1] + 0.15),
                    fontsize=fs - 1, color="#1a5276",
                    fontweight="bold")
        ax.annotate("x = ?", xy=(pC[0] + 0.15, pC[1] + 0.15),
                    fontsize=fs + 1, color="#27ae60", fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(rng.choice(self._L9_TITLES),
                      fontsize=fs + 1, fontweight="bold")
        q = rng.choice(self._L9_QUESTIONS)
        return q, str(target), self.fig_to_pil(fig, dpi=style["dpi"])

    def _arc_chain(self, rng):
        # Three arcs summing to 360; two labeled, ask for the third.
        a1 = rng.choice(list(range(40, 161, 10)))
        a2 = rng.choice(list(range(40, 161, 10)))
        if a1 + a2 >= 330:
            a2 = 300 - a1
        a3 = 360 - a1 - a2
        if a3 < 10 or a3 > 300:
            return None
        fig, ax, style, palette, fs = self._make_fig(rng)
        cx, cy, r = 3, 3, 2.3
        circle = plt.Circle((cx, cy), r, fill=False,
                            edgecolor=palette[0], linewidth=2)
        ax.add_patch(circle)
        a_start = rng.randint(0, 60)
        bounds = [a_start, a_start + a1, a_start + a1 + a2, a_start + 360]
        arc_cols = [palette[2], palette[4], palette[5 % len(palette)]]
        labels = [f"{a1}°", f"{a2}°", "?"]
        for i in range(3):
            arc_patch = mpatches.Arc((cx, cy), 2 * r, 2 * r,
                                      angle=0,
                                      theta1=bounds[i], theta2=bounds[i + 1],
                                      color=arc_cols[i % 3],
                                      linewidth=4, alpha=0.9)
            ax.add_patch(arc_patch)
            mid_theta = (bounds[i] + bounds[i + 1]) / 2.0
            lp = self._point_on_circle(cx, cy, r + 0.4, mid_theta)
            color = "#c0392b" if labels[i] == "?" else "#1a5276"
            weight = "bold"
            ax.annotate(labels[i], xy=lp,
                        fontsize=fs + 1, color=color,
                        fontweight=weight, ha="center")
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(self._title("arc_chain", rng),
                      fontsize=fs + 1, fontweight="bold")
        q = self._get_q("arc_chain", rng)
        return q, str(a3), self.fig_to_pil(fig, dpi=style["dpi"])
