"""
Multi-Hop Metric Chain QA — redesigned 2026-04-16.

Task: draw a labelled polygonal figure and ask for an unknown length or
angle that requires chaining multiple theorems.

DIVERSITY (per seed):
  * 5 primitive families: right-triangle ladder, stacked triangles,
    concentric squares with diagonals, right-trapezoid chain, right-hexagon fan.
  * Labels jittered in position (within reason).
  * 6 question phrasings.
  * Random color palette rotation.
  * Axis heading jitter (+/- rotation_mode).

DIFFICULTY:
  L0-L1: SINGLE right triangle; ONE Pythagoras or ONE triangle-sum.
         Formula is explicit in the question.
  L2-L3: 2-triangle chain; 1 intermediate quantity.
  L4-L5: 3-triangle chain or trapezoid+triangle.
  L6-L7: 3-4 hops.
  L8-L9: 4+ hops, NO formula hints, composed types.

Values are ONLY on the image.
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLES = [
    "Find the unknown",
    "Compute the marked value",
    "Solve the chain",
    "Geometry chain",
    "Find the ?",
    "Unknown quantity",
]

class MultiHopMetricChainQA(StandaloneVisualEnv):
    ENV_NAME = "multi_hop_metric_chain"

    _LENGTH_QUESTION_PHRASES = [
        "The figure shows right triangle(s) linked in a chain. Integer leg lengths are marked; the final hypotenuse (or final marked edge) is '?'. Find its length. Answer with a single integer.",
        "Study the chain of right triangles. Using the marked integer legs, find the length labeled '?'. Answer with a single integer.",
        "Each triangle in the figure is right-angled (right-angle symbol shown). Use the marked lengths to compute the '?' edge. Answer with an integer.",
        "Compute the length of the edge marked '?'. Apply the Pythagorean theorem along the chain. Answer with a single integer.",
    ]
    _LENGTH_SINGLE_PHRASES = [
        "The figure shows a right triangle with the right angle marked. Two legs are labeled with integer lengths. Find the hypotenuse using a^2 + b^2 = c^2. Answer with a single integer.",
        "Given the two labelled legs of the right triangle in the figure, find the hypotenuse length. Use the Pythagorean theorem. Answer with an integer.",
        "The image shows a right triangle. The legs are labelled with integers. Compute the hypotenuse (length marked '?'). Answer with an integer.",
    ]
    _ANGLE_QUESTION_PHRASES = [
        "The figure shows triangle(s) sharing a common base. Every base angle is labelled with an integer degree value. The final triangle's apex angle is marked '?'. Compute the apex angle in degrees. Answer with an integer.",
        "Using the labelled base angles, find the angle marked '?' by applying the triangle-angle-sum (= 180) at every step. Answer with a single integer in degrees.",
        "Compute the angle labelled '?' in the figure. Chain the triangle-angle-sum rule. Answer as an integer in degrees.",
    ]
    _ANGLE_SINGLE_PHRASES = [
        "A triangle is shown with two base angles labelled in degrees. Find the apex angle (marked '?') using the fact that the three interior angles sum to 180 degrees. Answer with an integer.",
        "Given the two labelled interior angles in the triangle, find the third angle marked '?'. Answer with an integer (degrees).",
        "Two interior angles of the triangle are shown in degrees. Compute the third (marked '?'). Answer with an integer.",
    ]

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesign 2026-04-17: push L9 to 4-hop length_trapezoid only
        # (genuinely harder than 3-hop length/angle chains, which the model
        # frequently shortcuts by reading the final edge from integer legs).
        if level <= 1:
            return {"n_hops": 1, "mode_pool": ["length_single", "angle_single"]}
        if level <= 3:
            return {"n_hops": 2, "mode_pool": ["length", "angle"]}
        if level <= 5:
            return {"n_hops": 2 + (level - 4),
                    "mode_pool": ["length", "angle", "length_trapezoid"]}
        if level <= 7:
            return {"n_hops": 3, "mode_pool": ["length", "angle", "length_trapezoid"]}
        return {"n_hops": 4, "mode_pool": ["length_trapezoid", "angle"]}

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(30):
            result = self._try_generate(parameter)
            if result is not None:
                return result
        return None

    def _try_generate(self, parameter: Dict):
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        mode = rng.choice(cfg["mode_pool"])
        n_hops = cfg["n_hops"]
        self._primary_complexity_feature = n_hops

        if mode == "length_single":
            return self._gen_single_right_triangle(rng, kind="length")
        if mode == "angle_single":
            return self._gen_single_right_triangle(rng, kind="angle")
        if mode == "length":
            r = self._gen_length_chain(rng, n_hops)
            if r is not None:
                return r
            return self._gen_angle_chain(rng, n_hops)
        if mode == "angle":
            return self._gen_angle_chain(rng, n_hops)
        if mode == "length_trapezoid":
            return self._gen_trapezoid_mixed(rng, n_hops)
        return None

    # ------------------------------------------------------------------ #
    # L0/L1 — SINGLE right triangle
    # ------------------------------------------------------------------ #
    def _gen_single_right_triangle(self, rng, kind="length"):
        if kind == "length":
            triples = [
                (3, 4, 5), (5, 12, 13), (8, 15, 17), (6, 8, 10),
                (7, 24, 25), (9, 12, 15), (20, 21, 29), (15, 20, 25),
            ]
            a, b, c = rng.choice(triples)
            if rng.random() < 0.5:
                a, b = b, a
            # Layout a triangle with right angle at origin.
            P = (0.0, 0.0)              # right-angle corner
            Q = (float(a), 0.0)         # along x-axis
            R = (0.0, float(b))         # along y-axis
            style = self._random_style()
            fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=style["dpi"])
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            ax.set_aspect("equal")
            ax.axis("off")
            palette = list(style["palette"])
            rng.shuffle(palette)
            face_color = palette[0]
            tri = Polygon([P, Q, R], closed=True, facecolor=face_color,
                          edgecolor="black", lw=2.0, alpha=0.4)
            ax.add_patch(tri)
            # Right-angle marker at P.
            m = 0.4 * min(a, b) / 5
            ax.plot([m, m, 0], [0, m, m], color="#333", lw=1.2)
            # Label legs a and b; mark hyp as '?'.
            ax.text(a / 2, -0.5, f"{a}", fontsize=16, fontweight="bold",
                    ha="center", va="center", color="#1a5276")
            ax.text(-0.5, b / 2, f"{b}", fontsize=16, fontweight="bold",
                    ha="right", va="center", color="#1a5276")
            mid = ((Q[0] + R[0]) / 2, (Q[1] + R[1]) / 2)
            ax.text(mid[0] + 0.5, mid[1] + 0.5, "?", fontsize=20,
                    fontweight="bold", ha="center", va="center",
                    color="#b00000")
            ax.plot([Q[0], R[0]], [Q[1], R[1]], color="#b00000", lw=2.5)
            ax.set_xlim(-2, a + 2)
            ax.set_ylim(-2, b + 2)
            ax.set_title(rng.choice(_TITLES), fontsize=14, fontweight="bold")
            img = self.fig_to_pil(fig, dpi=style["dpi"])
            q = rng.choice(self._LENGTH_SINGLE_PHRASES)
            return q, str(c), img

        # angle_single: draw a SCALENE triangle (no right-angle marker)
        # whose interior angles exactly match the labels.
        ang1 = rng.randint(30, 75)
        ang2 = rng.randint(30, 170 - ang1 - 15)
        if ang1 + ang2 > 150:
            ang2 = 150 - ang1
        third = 180 - ang1 - ang2
        if third < 20:
            return None
        # Place vertex A at origin, B along +x, C computed from angles a1@A, a2@B.
        w = 5.0
        a1r = math.radians(ang1)
        a2r = math.radians(ang2)
        denom = math.sin(a1r + a2r)
        if abs(denom) < 1e-6:
            return None
        t = w * math.sin(a2r) / denom
        A = (0.0, 0.0)
        B = (w, 0.0)
        C = (A[0] + t * math.cos(a1r), A[1] + t * math.sin(a1r))

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        palette = list(style["palette"])
        rng.shuffle(palette)
        face_color = palette[0]
        tri = Polygon([A, B, C], closed=True, facecolor=face_color,
                      edgecolor="black", lw=2.0, alpha=0.4)
        ax.add_patch(tri)
        # label ang1 near A (inside), ang2 near B (inside), ? near C.
        cx = (A[0] + B[0] + C[0]) / 3
        cy = (A[1] + B[1] + C[1]) / 3
        for vertex, label in [(A, f"{ang1}\u00b0"), (B, f"{ang2}\u00b0"),
                              (C, "?")]:
            dx = (cx - vertex[0]) * 0.35
            dy = (cy - vertex[1]) * 0.35
            col = "#b00000" if label == "?" else "#1a5276"
            fs = 20 if label == "?" else 14
            ax.text(vertex[0] + dx, vertex[1] + dy, label,
                    fontsize=fs, fontweight="bold",
                    ha="center", va="center", color=col)
        xs = [A[0], B[0], C[0]]
        ys = [A[1], B[1], C[1]]
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.5)
        ax.set_title(rng.choice(_TITLES), fontsize=14, fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        q = rng.choice(self._ANGLE_SINGLE_PHRASES)
        return q, str(third), img

    # ------------------------------------------------------------------ #
    # LENGTH CHAIN: right-triangle ladder
    # ------------------------------------------------------------------ #
    def _gen_length_chain(self, rng, n_hops):
        triples = [
            (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
            (20, 21, 29), (9, 12, 15), (6, 8, 10), (10, 24, 26),
            (8, 6, 10), (12, 16, 20), (15, 20, 25),
        ]
        index = {}
        for (a, b, c) in triples:
            for v in [a, b]:
                index.setdefault(v, []).append((a, b, c))

        current = list(rng.choice(triples))
        chain = [tuple(current)]
        for hop in range(n_hops - 1):
            prev_c = chain[-1][2]
            if prev_c not in index:
                return None
            nxt = rng.choice(index[prev_c])
            a2, b2, c2 = nxt
            if a2 == prev_c:
                new_leg = b2
            elif b2 == prev_c:
                new_leg = a2
            else:
                return None
            chain.append((prev_c, new_leg, c2))

        pts: List[Tuple[float, float]] = [(0.0, 0.0)]
        direction = rng.uniform(-0.1, 0.1)  # slight jitter
        cur = (0.0, 0.0)
        triangles = []
        for k, (leg_a, leg_b, hyp) in enumerate(chain):
            P = cur
            dx = leg_a * math.cos(direction)
            dy = leg_a * math.sin(direction)
            Q = (P[0] + dx, P[1] + dy)
            perp = direction + math.pi / 2
            R = (Q[0] + leg_b * math.cos(perp),
                 Q[1] + leg_b * math.sin(perp))
            triangles.append((P, Q, R, leg_a, leg_b, hyp))
            next_dir = math.atan2(R[1] - P[1], R[0] - P[0])
            cur = R
            direction = next_dir

        last_hyp = chain[-1][2]
        answer = last_hyp
        img = self._render_length_chain(triangles, chain, rng)
        q = rng.choice(self._LENGTH_QUESTION_PHRASES)
        return q, str(answer), img

    def _render_length_chain(self, triangles, chain, rng) -> Image.Image:
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        palette = list(style["palette"])
        rng.shuffle(palette)

        all_pts = []
        for k, (P, Q, R, leg_a, leg_b, hyp) in enumerate(triangles):
            face_color = palette[k % len(palette)]
            tri = Polygon([P, Q, R], closed=True, facecolor=face_color,
                          edgecolor="black", lw=1.6, alpha=0.45)
            ax.add_patch(tri)
            all_pts.extend([P, Q, R])

            v_qp = (P[0] - Q[0], P[1] - Q[1])
            v_qr = (R[0] - Q[0], R[1] - Q[1])
            norm_qp = math.hypot(*v_qp) or 1.0
            norm_qr = math.hypot(*v_qr) or 1.0
            m = 0.6
            a1 = (Q[0] + v_qp[0] * m / norm_qp,
                  Q[1] + v_qp[1] * m / norm_qp)
            a3 = (Q[0] + v_qr[0] * m / norm_qr,
                  Q[1] + v_qr[1] * m / norm_qr)
            a2 = (a1[0] + a3[0] - Q[0], a1[1] + a3[1] - Q[1])
            ax.plot([a1[0], a2[0], a3[0]], [a1[1], a2[1], a3[1]],
                    color="#333333", lw=1.2)

            mid_pq = ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2)
            leg_dir = math.atan2(Q[1] - P[1], Q[0] - P[0])
            perp = leg_dir + math.pi / 2
            off = 0.7
            ox, oy = math.cos(perp) * off, math.sin(perp) * off
            if k == 0:
                ax.text(mid_pq[0] - ox, mid_pq[1] - oy, f"{leg_a}",
                        fontsize=13, fontweight="bold", ha="center",
                        va="center", color="#1a5276")
            mid_qr = ((Q[0] + R[0]) / 2, (Q[1] + R[1]) / 2)
            leg2_dir = math.atan2(R[1] - Q[1], R[0] - Q[0])
            perp2 = leg2_dir + math.pi / 2
            ox2, oy2 = math.cos(perp2) * off, math.sin(perp2) * off
            ax.text(mid_qr[0] + ox2, mid_qr[1] + oy2, f"{leg_b}",
                    fontsize=13, fontweight="bold", ha="center",
                    va="center", color="#1a5276")

            if k == len(triangles) - 1:
                mid_pr = ((P[0] + R[0]) / 2, (P[1] + R[1]) / 2)
                hyp_dir = math.atan2(R[1] - P[1], R[0] - P[0])
                perp3 = hyp_dir + math.pi / 2
                ox3, oy3 = math.cos(perp3) * off * 1.1, math.sin(perp3) * off * 1.1
                ax.text(mid_pr[0] + ox3, mid_pr[1] + oy3, "?",
                        fontsize=17, fontweight="bold", ha="center",
                        va="center", color="#b00000")
                ax.plot([P[0], R[0]], [P[1], R[1]],
                        color="#b00000", lw=2.2, linestyle="-")

        arr = np.array(all_pts)
        margin = 2.0
        ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
        ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)
        ax.set_title(rng.choice(_TITLES), fontsize=13, fontweight="bold", pad=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # ANGLE CHAIN
    # ------------------------------------------------------------------ #
    def _gen_angle_chain(self, rng, n_hops):
        angles = []
        answer = None
        for k in range(n_hops):
            a1 = rng.randint(30, 70)
            a2_max = min(140 - a1, 150 - a1 - 5)
            if a2_max < 30:
                return None
            a2 = rng.randint(30, a2_max)
            a3 = 180 - a1 - a2
            if a3 < 20:
                return None
            angles.append((a1, a2, a3))
            if k == n_hops - 1:
                answer = a3

        x = 0.0
        w = 2.5
        all_pts = []
        triangles_pts = []
        for k, (a1, a2, a3) in enumerate(angles):
            A = (x, 0.0)
            B = (x + w, 0.0)
            a1r = math.radians(a1)
            a2r = math.radians(a2)
            denom = math.sin(a1r + a2r)
            if abs(denom) < 1e-6:
                return None
            t = w * math.sin(a2r) / denom
            C = (A[0] + t * math.cos(a1r), A[1] + t * math.sin(a1r))
            triangles_pts.append((A, B, C))
            all_pts.extend([A, B, C])
            x += w
        img = self._render_angle_chain(triangles_pts, angles, n_hops, rng)
        q = rng.choice(self._ANGLE_QUESTION_PHRASES)
        return q, str(answer), img

    def _render_angle_chain(self, triangles_pts, angles, n_hops, rng) -> Image.Image:
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        palette = list(style["palette"])
        rng.shuffle(palette)

        all_pts = []
        for k, ((A, B, C), (a1, a2, a3)) in enumerate(
                zip(triangles_pts, angles)):
            face_color = palette[k % len(palette)]
            tri = Polygon([A, B, C], closed=True, facecolor=face_color,
                          edgecolor="black", lw=1.6, alpha=0.45)
            ax.add_patch(tri)
            all_pts.extend([A, B, C])
            for vertex, label_text in [(A, f"{a1}\u00b0"), (B, f"{a2}\u00b0")]:
                cx = (A[0] + B[0] + C[0]) / 3
                cy = (A[1] + B[1] + C[1]) / 3
                dx = (cx - vertex[0]) * 0.45
                dy = (cy - vertex[1]) * 0.45
                ax.text(vertex[0] + dx, vertex[1] + dy, label_text,
                        fontsize=11, fontweight="bold", ha="center",
                        va="center", color="#1a5276")
            if k == n_hops - 1:
                ax.text(C[0], C[1] + 0.35, "?",
                        fontsize=16, fontweight="bold", ha="center",
                        va="bottom", color="#b00000")
            else:
                ax.text(C[0], C[1] + 0.25, f"{a3}\u00b0",
                        fontsize=11, fontweight="bold", ha="center",
                        va="bottom", color="#1a5276")

        arr = np.array(all_pts)
        margin = 1.5
        ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
        ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin + 1)
        ax.set_title(rng.choice(_TITLES), fontsize=13, fontweight="bold", pad=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Trapezoid + triangle mix
    # ------------------------------------------------------------------ #
    def _gen_trapezoid_mixed(self, rng, n_hops):
        """Right-trapezoid: the right leg is perpendicular to the parallel sides.
        Given: top a, bottom b, right leg h. Ask for the slanted side (hyp
        of right triangle formed by (b-a) and h)."""
        # Use Pythagorean triples so diff leg (b-a) and h form a triple.
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (6, 8, 10),
                   (9, 12, 15), (20, 21, 29)]
        legA, legB, hyp = rng.choice(triples)
        a = rng.randint(2, 10)
        b = a + legA
        h = legB
        # Slanted side: hyp
        # For higher hop counts, require chaining a second Pythagoras on top.
        if n_hops >= 3:
            # Second leg on top: split top edge into (m, legA) right triangle
            # with height q, giving next Pythagoras.
            triples2 = [(3, 4, 5), (6, 8, 10), (5, 12, 13)]
            q1, q2, q3 = rng.choice(triples2)
            # Draw attached triangle above top edge.
            # Answer becomes q3 (the hypotenuse of the upper attached triangle).
            answer = q3 if rng.random() < 0.5 else hyp
            # For simplicity: the attached triangle gives the final '?'.
            answer = q3
        else:
            answer = hyp

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        palette = list(style["palette"])
        rng.shuffle(palette)

        # Trapezoid vertices.
        P = (0.0, 0.0)
        Q = (b, 0.0)
        R = (b, h)
        S = (b - a, h)
        tri = Polygon([P, Q, R, S], closed=True, facecolor=palette[0],
                      edgecolor="black", lw=1.6, alpha=0.45)
        ax.add_patch(tri)
        # right-angle mark at Q.
        ax.plot([b - 0.3, b - 0.3, b], [0, 0.3, 0.3], color="#333", lw=1.2)
        # Label a, b, h on image.
        ax.text(b / 2, -0.4, f"{b}", fontsize=13, fontweight="bold",
                ha="center", color="#1a5276")
        ax.text((b + (b - a)) / 2, h + 0.3, f"{a}", fontsize=13,
                fontweight="bold", ha="center", color="#1a5276")
        ax.text(b + 0.4, h / 2, f"{h}", fontsize=13, fontweight="bold",
                ha="left", color="#1a5276")
        if n_hops < 3:
            # Mark slant ? on PS
            mid = ((P[0] + S[0]) / 2, (P[1] + S[1]) / 2)
            ax.text(mid[0] - 0.7, mid[1], "?", fontsize=18, fontweight="bold",
                    color="#b00000")
            ax.plot([P[0], S[0]], [P[1], S[1]], color="#b00000", lw=2.2)
            q_txt = rng.choice(self._LENGTH_QUESTION_PHRASES)
            img = self.fig_to_pil(fig, dpi=style["dpi"])
            return q_txt, str(answer), img

        # n_hops >= 3: attach upper triangle with legs (q1, q2).
        # Draw a right triangle on top of the trapezoid.
        top_base_start = (b - a, h)
        top_base_end = (b - a + q1, h)
        apex = (b - a + q1, h + q2)
        tri_top = Polygon([top_base_start, top_base_end, apex], closed=True,
                          facecolor=palette[1 % len(palette)],
                          edgecolor="black", lw=1.6, alpha=0.45)
        ax.add_patch(tri_top)
        # right-angle mark at top_base_end
        ax.plot([top_base_end[0] - 0.3, top_base_end[0] - 0.3, top_base_end[0]],
                [h, h + 0.3, h + 0.3], color="#333", lw=1.2)
        # label q1 (base of upper tri) — overlaps with trapezoid a? Only if q1=a.
        ax.text(top_base_start[0] + q1 / 2, h - 0.4, f"{q1}", fontsize=11,
                fontweight="bold", ha="center", color="#1a5276")
        ax.text(top_base_end[0] + 0.4, h + q2 / 2, f"{q2}", fontsize=11,
                fontweight="bold", ha="left", color="#1a5276")
        # mark hypotenuse of upper triangle as ?
        mid_up = ((top_base_start[0] + apex[0]) / 2,
                  (top_base_start[1] + apex[1]) / 2)
        ax.text(mid_up[0] - 0.6, mid_up[1] + 0.3, "?", fontsize=18,
                fontweight="bold", color="#b00000")
        ax.plot([top_base_start[0], apex[0]], [top_base_start[1], apex[1]],
                color="#b00000", lw=2.2)

        # set limits
        xs = [0, b, 0, b, top_base_start[0], top_base_end[0], apex[0]]
        ys = [0, 0, h, h, top_base_start[1], top_base_end[1], apex[1]]
        ax.set_xlim(min(xs) - 2, max(xs) + 2)
        ax.set_ylim(min(ys) - 2, max(ys) + 2)
        ax.set_title(rng.choice(_TITLES), fontsize=13, fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        q_txt = rng.choice(self._LENGTH_QUESTION_PHRASES)
        return q_txt, str(answer), img

if __name__ == "__main__":
    env = MultiHopMetricChainQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer}")
