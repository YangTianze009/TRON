"""
Secant/tangent QA (redesigned 2026-04-16) — circle with secant
and/or tangent lines.

Questions: power of a point, tangent-chord angle, external segment
length, tangent length, two-secants angle.

Critical fix (vs Grade D baseline):
  * Old question text leaked ALL the given numbers: e.g. "Point P is 10
    units from the center of a circle with radius 3. Find the tangent
    length PT." — pure algebra solvable without the image.
  * Now: numbers are displayed on the image as labels; the question
    text says "as shown in the figure" and refers to labels (e.g.
    "find PT", "find the external-segment product"). The model MUST
    read the numbers from the image.
  * Diverse colors per seed.
  * Randomized positions (P location, angles, radius).
  * 5+ question templates per subtype.
  * L0/L9 structural shift in qtype mix.
"""
import math
import random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_NEUTRAL_TITLES = [
    "Circle geometry",
    "Figure",
    "Diagram",
    "Circle diagram",
    "Geometric figure",
    "Tangent and secant",
    "Circle problem",
]

def _rand_palette_choice(rng, palette, exclude=None):
    colors = [c for c in palette if c != exclude]
    return rng.choice(colors) if colors else palette[0]

class SecantTangentQA(StandaloneVisualEnv):
    ENV_NAME = "secant_tangent"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["tangent_length"], "qweights": [10]}
        if level <= 2:
            return {"qtypes": ["tangent_length", "tangent_chord_angle"],
                    "qweights": [5, 5]}
        if level <= 4:
            return {"qtypes": ["tangent_length", "secant_external",
                               "tangent_chord_angle"],
                    "qweights": [3, 4, 3]}
        if level <= 6:
            return {"qtypes": ["secant_external", "power_of_point",
                               "tangent_chord_angle"],
                    "qweights": [3, 4, 3]}
        if level <= 8:
            return {"qtypes": ["power_of_point", "two_secants",
                               "tangent_secant_chain"],
                    "qweights": [3, 4, 3]}
        # L9: multi-step combinations only (no single-step shortcuts).
        return {"qtypes": ["tangent_secant_chain", "two_secants",
                           "power_of_point"],
                "qweights": [5, 3, 2]}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 7701)
        vis_rng = random.Random(seed * 1000 + level * 37 + 4571)
        style = self._random_style()
        qtype = parameter.get("question_type")
        valid = {"tangent_length", "secant_external", "power_of_point",
                 "tangent_chord_angle", "two_secants",
                 "tangent_secant_chain"}
        if qtype not in valid:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"],
                                    k=1)[0]

        try:
            if qtype == "tangent_length":
                return self._tangent_length(rng, sub_rng, vis_rng, style)
            elif qtype == "secant_external":
                return self._secant_external(rng, sub_rng, vis_rng, style)
            elif qtype == "power_of_point":
                return self._power_of_point(rng, sub_rng, vis_rng, style)
            elif qtype == "tangent_chord_angle":
                return self._tangent_chord_angle(rng, sub_rng, vis_rng, style)
            elif qtype == "two_secants":
                return self._two_secants(rng, sub_rng, vis_rng, style)
            elif qtype == "tangent_secant_chain":
                return self._tangent_secant_chain(rng, sub_rng, vis_rng,
                                                  style)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _draw_circle(self, ax, style, cx, cy, r, palette):
        c = plt.Circle((cx, cy), r, fill=False, ec=palette[0],
                        linewidth=style["line_width"] + 0.5)
        ax.add_patch(c)
        ax.plot(cx, cy, "+", color="gray", markersize=6)
        return c

    # ------------------------------------------------------------------
    # Subproblems
    # ------------------------------------------------------------------

    def _tangent_length(self, rng, sub_rng, vis_rng, style):
        r = rng.randint(3, 8)
        d = rng.randint(r + 2, r + 10)
        t = round(math.sqrt(d ** 2 - r ** 2), 2)

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        cx, cy = 3.5, 3.0
        draw_r = 1.8
        scale = draw_r / r

        self._draw_circle(ax, style, cx, cy, draw_r, palette)

        px = cx + d * scale
        py = cy
        ax.plot(px, py, "o", color=palette[2 % len(palette)], markersize=8,
                zorder=5)
        ax.text(px + 0.15, py + 0.2, "P",
                fontsize=style["font_size_base"] + 1, fontweight="bold")

        tang_ang = math.acos(r / d)
        tx = cx + draw_r * math.cos(tang_ang)
        ty = cy + draw_r * math.sin(tang_ang)
        tx2 = cx + draw_r * math.cos(-tang_ang)
        ty2 = cy + draw_r * math.sin(-tang_ang)

        tangent_color = palette[3 % len(palette)]
        ax.plot([px, tx], [py, ty], color=tangent_color,
                linewidth=style["line_width"] + 0.5)
        ax.plot([px, tx2], [py, ty2], color=tangent_color,
                linewidth=style["line_width"] + 0.5, alpha=0.4)
        ax.plot([cx, px], [cy, py], color="gray", linewidth=1,
                linestyle=":")

        ax.plot(tx, ty, "o", color=tangent_color, markersize=6, zorder=5)
        ax.text(tx - 0.15, ty + 0.2, "T",
                fontsize=style["font_size_base"], fontweight="bold")
        ax.plot([cx, tx], [cy, ty], color="gray",
                linewidth=1, linestyle="--", alpha=0.5)

        # Numeric labels on the figure — question text will NOT restate these
        mid_x, mid_y = (cx + px) / 2, cy - 0.3
        ax.annotate(f"d = {d}", xy=(mid_x, mid_y),
                    fontsize=style["font_size_base"],
                    fontweight="bold", ha="center")
        ax.annotate(f"r = {r}",
                    xy=(cx - 0.3, cy + draw_r * 0.5 + 0.2),
                    fontsize=style["font_size_base"], fontweight="bold")

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "In the figure, P is an external point and PT is a tangent "
            "line to the circle. Use the labeled distance d and the "
            "labeled radius r to find the tangent length PT. Round to "
            "2 decimals.",
            "As shown in the figure, a tangent from point P touches the "
            "circle at T. The distance from P to the center and the "
            "circle's radius are labeled. Find PT. Round to 2 decimals.",
            "Using the values labeled in the diagram, compute the "
            "tangent length PT. Round to 2 decimals.",
        ])
        return q, str(t), img

    def _secant_external(self, rng, sub_rng, vis_rng, style):
        ext = rng.randint(2, 8)
        chord = rng.randint(3, 12)
        whole = ext + chord
        power = ext * whole

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        cx, cy, r = 3.5, 3.0, 1.8
        self._draw_circle(ax, style, cx, cy, r, palette)

        angle = vis_rng.uniform(-0.4, 0.4)
        px = cx + r + ext * 0.18 + 0.5
        py = cy + angle

        dx = cx - px
        dy = cy - py
        dist_pc = math.sqrt(dx ** 2 + dy ** 2)
        ux, uy = dx / dist_pc, dy / dist_pc

        a_coeff = 1
        b_coeff = 2 * ((px - cx) * ux + (py - cy) * uy)
        c_coeff = (px - cx) ** 2 + (py - cy) ** 2 - r ** 2
        disc = max(0, b_coeff ** 2 - 4 * a_coeff * c_coeff)
        t1 = (-b_coeff - math.sqrt(disc)) / (2 * a_coeff)
        t2 = (-b_coeff + math.sqrt(disc)) / (2 * a_coeff)

        ax_pt = (px + t1 * ux, py + t1 * uy)
        bx_pt = (px + t2 * ux, py + t2 * uy)

        secant_color = palette[2 % len(palette)]
        ax.plot([px, bx_pt[0] - 0.2 * ux], [py, bx_pt[1] - 0.2 * uy],
                color=secant_color,
                linewidth=style["line_width"] + 0.5, zorder=3)

        ax.plot(px, py, "o", color=secant_color, markersize=8, zorder=5)
        ax.text(px + 0.15, py + 0.2, "P",
                fontsize=style["font_size_base"] + 1, fontweight="bold")

        pt_color = palette[3 % len(palette)]
        ax.plot(*ax_pt, "o", color=pt_color, markersize=6, zorder=5)
        ax.text(ax_pt[0] + 0.15, ax_pt[1] + 0.2, "A",
                fontsize=style["font_size_base"], fontweight="bold")

        ax.plot(*bx_pt, "o", color=pt_color, markersize=6, zorder=5)
        ax.text(bx_pt[0] - 0.3, bx_pt[1] + 0.2, "B",
                fontsize=style["font_size_base"], fontweight="bold")

        # Labels on the figure ONLY
        mid_pa = ((px + ax_pt[0]) / 2, (py + ax_pt[1]) / 2)
        ax.annotate(f"PA = {ext}", xy=mid_pa,
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", ha="center",
                    xytext=(mid_pa[0], mid_pa[1] + 0.3))
        mid_ab = ((ax_pt[0] + bx_pt[0]) / 2, (ax_pt[1] + bx_pt[1]) / 2)
        ax.annotate(f"AB = {chord}", xy=mid_ab,
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", ha="center",
                    xytext=(mid_ab[0], mid_ab[1] - 0.4))

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "A secant from external point P passes through the circle "
            "at A (near) and B (far). Use the labeled lengths in the "
            "figure to compute the power of the point (PA times PB).",
            "As shown, the secant from P hits the circle at A and B. "
            "Read PA and AB from the figure and compute PA * PB.",
            "In the figure, the secant from P crosses the circle at A "
            "and B. Using the labeled segment lengths, find the product "
            "PA * PB (the power of point P).",
        ])
        return q, str(power), img

    def _power_of_point(self, rng, sub_rng, vis_rng, style):
        # Try several (e1,c1) combinations until we find an e2 that divides
        # cleanly and produces a valid c2>0.
        chosen = None
        for _ in range(50):
            e1 = rng.randint(2, 6)
            c1 = rng.randint(3, 10)
            w1 = e1 + c1
            product = e1 * w1
            for _ in range(40):
                e2 = rng.randint(2, 6)
                if e2 == e1:
                    continue
                if product % e2 != 0:
                    continue
                w2 = product // e2
                c2 = w2 - e2
                if c2 > 0 and w2 > e2:
                    chosen = (e1, c1, e2, c2)
                    break
            if chosen:
                break
        if chosen is None:
            return None
        e1, c1, e2, c2 = chosen

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        cx, cy, r = 3.2, 3.0, 1.8
        self._draw_circle(ax, style, cx, cy, r, palette)

        px, py = 6.5, 3.0
        ax.plot(px, py, "o", color=palette[2 % len(palette)],
                markersize=8, zorder=5)
        ax.text(px + 0.15, py + 0.15, "P",
                fontsize=style["font_size_base"] + 1, fontweight="bold")

        def _draw_secant(angle_offset, label1, label2, val_ext, val_chord,
                         color, y_text_off):
            target_x = cx + math.cos(math.pi + angle_offset) * 0.3
            target_y = cy + math.sin(math.pi + angle_offset) * 0.3
            dx_ = target_x - px
            dy_ = target_y - py
            dist = math.sqrt(dx_ ** 2 + dy_ ** 2)
            ux, uy = dx_ / dist, dy_ / dist
            a_c = 1
            b_c = 2 * ((px - cx) * ux + (py - cy) * uy)
            c_c = (px - cx) ** 2 + (py - cy) ** 2 - r ** 2
            disc = max(0, b_c ** 2 - 4 * a_c * c_c)
            t1 = (-b_c - math.sqrt(disc)) / 2
            t2 = (-b_c + math.sqrt(disc)) / 2
            a_pt = (px + t1 * ux, py + t1 * uy)
            b_pt = (px + t2 * ux, py + t2 * uy)
            ax.plot([px, b_pt[0]], [py, b_pt[1]], color=color,
                    linewidth=style["line_width"] + 0.3, zorder=3)
            ax.plot(*a_pt, "o", color=color, markersize=5, zorder=5)
            ax.plot(*b_pt, "o", color=color, markersize=5, zorder=5)
            ax.text(a_pt[0], a_pt[1] + y_text_off, label1,
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", ha="center")
            ax.text(b_pt[0], b_pt[1] + y_text_off, label2,
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", ha="center")
            mid_ext = ((px + a_pt[0]) / 2, (py + a_pt[1]) / 2)
            ax.text(mid_ext[0], mid_ext[1] + y_text_off * 0.7,
                    f"{val_ext}",
                    fontsize=style["font_size_base"] - 1, ha="center",
                    color=color)
            mid_chord = ((a_pt[0] + b_pt[0]) / 2,
                         (a_pt[1] + b_pt[1]) / 2)
            ax.text(mid_chord[0], mid_chord[1] + y_text_off * 0.7,
                    f"{val_chord}",
                    fontsize=style["font_size_base"] - 1, ha="center",
                    color=color)

        color1 = palette[3 % len(palette)]
        _draw_secant(0.35, "A", "B", e1, c1, color1, 0.25)
        _draw_secant(-0.35, "C", "D", e2, "?", "red", -0.35)

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "Two secants are drawn from the external point P in the "
            "figure. One secant has external segment PA and chord AB "
            "(values labeled). The other has external segment PC (value "
            "labeled) and an unknown chord CD. Find CD.",
            "As shown, P has two secants PAB and PCD. Using the labeled "
            "PA, AB, and PC, find CD.",
            "In the figure, apply the power of a point at P to determine "
            "the unknown chord CD. Use the labeled values of PA, AB, and "
            "PC.",
        ])
        return q, str(c2), img

    def _tangent_chord_angle(self, rng, sub_rng, vis_rng, style):
        arc = rng.choice(range(40, 161, 2))
        angle = arc // 2

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        cx, cy, r = 3, 3, 2
        self._draw_circle(ax, style, cx, cy, r, palette)

        tp = (cx, cy - r)
        tangent_color = palette[2 % len(palette)]
        ax.plot([cx - 2.5, cx + 2.5], [tp[1], tp[1]], color=tangent_color,
                linewidth=style["line_width"] + 0.5, label="tangent")
        ax.plot(*tp, "o", color=palette[3 % len(palette)],
                markersize=7, zorder=5)
        ax.text(tp[0] + 0.15, tp[1] - 0.3, "T",
                fontsize=style["font_size_base"], fontweight="bold")

        arc_rad = math.radians(arc)
        chord_angle = math.radians(270) + arc_rad
        chord_end = (cx + r * math.cos(chord_angle),
                     cy + r * math.sin(chord_angle))

        chord_color = palette[3 % len(palette)]
        ax.plot([tp[0], chord_end[0]], [tp[1], chord_end[1]],
                color=chord_color,
                linewidth=style["line_width"] + 0.5, zorder=3)
        ax.plot(*chord_end, "o", color=chord_color, markersize=6, zorder=5)
        ax.text(chord_end[0] - 0.3, chord_end[1] + 0.2, "C",
                fontsize=style["font_size_base"], fontweight="bold")

        arc_patch = mpatches.Arc(
            (cx, cy), 2 * r, 2 * r, angle=0, theta1=270,
            theta2=270 + arc, color="red", linewidth=3, alpha=0.7)
        ax.add_patch(arc_patch)

        label_angle = math.radians(270 + arc / 2)
        label_x = cx + (r + 0.4) * math.cos(label_angle)
        label_y = cy + (r + 0.4) * math.sin(label_angle)
        ax.annotate(f"arc = {arc}\u00b0", xy=(label_x, label_y),
                    fontsize=style["font_size_base"],
                    color="red", fontweight="bold", ha="center")

        ax.text(cx + 2.2, tp[1] - 0.25, "tangent",
                fontsize=style["font_size_base"] - 1,
                color=tangent_color, fontstyle="italic")

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "A tangent and a chord meet at point T on the circle (see "
            "figure). The intercepted arc is labeled. Find the angle "
            "between the tangent and the chord at T.",
            "As shown, a tangent touches the circle at T and a chord TC "
            "is drawn. Using the labeled arc, compute the angle between "
            "the tangent and the chord at T.",
            "In the figure, the intercepted arc is shown (labeled in "
            "degrees). Compute the tangent-chord angle at T.",
        ])
        return q, str(angle), img

    def _two_secants(self, rng, sub_rng, vis_rng, style):
        arc1 = rng.randint(60, 200)
        arc2 = rng.randint(20, arc1 - 10)
        angle = abs(arc1 - arc2) // 2

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        cx, cy, r = 3.0, 3.0, 1.8
        self._draw_circle(ax, style, cx, cy, r, palette)

        px, py = 6.5, 3.0
        ax.plot(px, py, "o", color=palette[2 % len(palette)],
                markersize=8, zorder=5)
        ax.text(px + 0.12, py + 0.15, "P",
                fontsize=style["font_size_base"] + 1, fontweight="bold")

        half_far = math.radians(arc1 / 2)
        half_near = math.radians(arc2 / 2)
        base_angle = math.radians(180)

        a_angle = base_angle + half_near
        c_angle = base_angle - half_near
        b_angle = base_angle + half_far
        d_angle = base_angle - half_far

        pts = {}
        for name, ang in [("A", a_angle), ("B", b_angle),
                          ("C", c_angle), ("D", d_angle)]:
            pts[name] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

        for name_pair, color, lbl_off in [
            (("A", "B"), palette[3 % len(palette)], 0.25),
            (("C", "D"), "green", -0.3)
        ]:
            n1, n2 = name_pair
            p1, p2 = pts[n1], pts[n2]
            ax.plot([px, p2[0]], [py, p2[1]], color=color,
                    linewidth=style["line_width"] + 0.3, zorder=3)
            ax.plot(*p1, "o", color=color, markersize=5, zorder=5)
            ax.plot(*p2, "o", color=color, markersize=5, zorder=5)
            ax.text(p1[0] - 0.25, p1[1] + lbl_off, n1,
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", color=color)
            ax.text(p2[0] - 0.25, p2[1] + lbl_off, n2,
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", color=color)

        theta1_deg = math.degrees(d_angle)
        theta2_deg = math.degrees(b_angle)
        if theta2_deg < theta1_deg:
            theta1_deg, theta2_deg = theta2_deg, theta1_deg
        arc_far = mpatches.Arc(
            (cx, cy), 2 * r + 0.15, 2 * r + 0.15, angle=0,
            theta1=theta1_deg, theta2=theta2_deg,
            color="blue", linewidth=2.5, alpha=0.7)
        ax.add_patch(arc_far)

        theta1n = math.degrees(c_angle)
        theta2n = math.degrees(a_angle)
        if theta2n < theta1n:
            theta1n, theta2n = theta2n, theta1n
        arc_near = mpatches.Arc(
            (cx, cy), 2 * r - 0.15, 2 * r - 0.15, angle=0,
            theta1=theta1n, theta2=theta2n,
            color="green", linewidth=2.5, alpha=0.7)
        ax.add_patch(arc_near)

        # Labels
        ax.text(0.3, 5.2, f"Far arc = {arc1}\u00b0",
                fontsize=style["font_size_base"],
                fontweight="bold", color="blue")
        ax.text(0.3, 4.7, f"Near arc = {arc2}\u00b0",
                fontsize=style["font_size_base"],
                fontweight="bold", color="green")

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "Two secants from external point P intercept the far arc and "
            "the near arc (labeled in the figure). Find the angle between "
            "the two secants at P.",
            "As shown in the diagram, two secants from P cut off a far "
            "arc and a near arc (both labeled). Compute the angle between "
            "the secants.",
            "Use the far-arc and near-arc values labeled in the figure to "
            "find the angle between the two secants at P.",
        ])
        return q, str(angle), img

    def _tangent_secant_chain(self, rng, sub_rng, vis_rng, style):
        """Multi-step L9 problem: combine tangent-length (Pythagoras) with
        power-of-a-point. From external point P we have a tangent PT (r,d
        labelled on the figure) AND a secant PAB with external segment PA
        labelled. Using PT^2 = PA * PB, solve for PB, then return the chord
        length AB = PB - PA. Requires TWO non-trivial steps and reading
        three numeric labels from the image.
        """
        # Choose (r, d) so PT^2 is a "nice" integer if possible.
        chosen = None
        for _ in range(60):
            r = rng.randint(3, 9)
            d = rng.randint(r + 2, r + 12)
            pt_sq = d * d - r * r  # = PT^2
            # Pick PA so that PB = pt_sq / PA is a positive integer and
            # PB > PA (so chord AB > 0).
            pa_candidates = [k for k in range(2, pt_sq)
                             if pt_sq % k == 0 and (pt_sq // k) > k]
            if not pa_candidates:
                continue
            pa = sub_rng.choice(pa_candidates)
            pb = pt_sq // pa
            chord = pb - pa
            if chord <= 0 or chord > 40:
                continue
            chosen = (r, d, pa, pb, chord)
            break
        if chosen is None:
            return None
        r, d, pa, pb, chord = chosen

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7.5 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        cx, cy, draw_r = 3.2, 3.0, 1.8
        scale = draw_r / r
        self._draw_circle(ax, style, cx, cy, draw_r, palette)

        px = cx + d * scale
        py = cy
        ax.plot(px, py, "o", color=palette[2 % len(palette)],
                markersize=8, zorder=5)
        ax.text(px + 0.15, py + 0.2, "P",
                fontsize=style["font_size_base"] + 1, fontweight="bold")

        # Tangent from P to T (upper).
        tang_ang = math.acos(r / d)
        tx = cx + draw_r * math.cos(tang_ang)
        ty = cy + draw_r * math.sin(tang_ang)
        tangent_color = palette[3 % len(palette)]
        ax.plot([px, tx], [py, ty], color=tangent_color,
                linewidth=style["line_width"] + 0.5)
        ax.plot(tx, ty, "o", color=tangent_color, markersize=6, zorder=5)
        ax.text(tx - 0.12, ty + 0.22, "T",
                fontsize=style["font_size_base"], fontweight="bold")
        # Radius guide (dashed) from center to T.
        ax.plot([cx, tx], [cy, ty], color="gray", linewidth=1,
                linestyle="--", alpha=0.5)

        # Secant from P going through circle at A (near) and B (far), drawn
        # below the horizontal so it doesn't overlap the tangent.
        # Place A at the near intersection using a downward-tilted line.
        angle_off = -0.55
        target_x = cx + math.cos(math.pi + angle_off) * 0.3
        target_y = cy + math.sin(math.pi + angle_off) * 0.3
        dx_ = target_x - px
        dy_ = target_y - py
        dist = math.sqrt(dx_ ** 2 + dy_ ** 2)
        ux, uy = dx_ / dist, dy_ / dist
        b_c = 2 * ((px - cx) * ux + (py - cy) * uy)
        c_c = (px - cx) ** 2 + (py - cy) ** 2 - draw_r ** 2
        disc = max(0, b_c ** 2 - 4 * c_c)
        t1 = (-b_c - math.sqrt(disc)) / 2
        t2 = (-b_c + math.sqrt(disc)) / 2
        a_pt = (px + t1 * ux, py + t1 * uy)
        b_pt = (px + t2 * ux, py + t2 * uy)
        secant_color = "teal"
        ax.plot([px, b_pt[0]], [py, b_pt[1]], color=secant_color,
                linewidth=style["line_width"] + 0.4, zorder=3)
        ax.plot(*a_pt, "o", color=secant_color, markersize=5, zorder=5)
        ax.plot(*b_pt, "o", color=secant_color, markersize=5, zorder=5)
        ax.text(a_pt[0] + 0.05, a_pt[1] - 0.3, "A",
                fontsize=style["font_size_base"] - 1, fontweight="bold",
                color=secant_color)
        ax.text(b_pt[0] - 0.3, b_pt[1] - 0.3, "B",
                fontsize=style["font_size_base"] - 1, fontweight="bold",
                color=secant_color)

        # Labels on the figure only.
        ax.annotate(f"d = {d}",
                    xy=((cx + px) / 2, cy + 0.25),
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold", ha="center")
        ax.annotate(f"r = {r}",
                    xy=(cx - 0.25, cy + draw_r * 0.55),
                    fontsize=style["font_size_base"] - 1,
                    fontweight="bold")
        mid_pa = ((px + a_pt[0]) / 2, (py + a_pt[1]) / 2)
        ax.text(mid_pa[0] + 0.1, mid_pa[1] - 0.25, f"PA = {pa}",
                fontsize=style["font_size_base"] - 1, fontweight="bold",
                color=secant_color, ha="center")

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "From external point P the tangent PT and a secant through A "
            "and B are drawn. Using the labelled d, r and PA, and the "
            "identity PT^2 = PA * PB, find the chord length AB.",
            "As shown in the figure, PT is tangent at T and PAB is a "
            "secant. Read d, r and PA from the labels; compute PT first "
            "(tangent length), then apply the power of the point to find "
            "AB.",
            "In the figure, use d and r to find the tangent PT, then use "
            "PT^2 = PA * PB with the labelled PA to determine the chord "
            "AB.",
        ])
        return q, str(chord), img

if __name__ == "__main__":
    env = SecantTangentQA()
    for lv in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": lv})
            print(f"L{lv} s{seed}: ok={ok}, A={env._answer}")
