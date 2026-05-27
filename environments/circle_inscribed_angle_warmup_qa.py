"""
Circle Inscribed Angle Warmup QA environment (v3 diversity redesign, 2026-04-16).

Targets Property reasoning / multi-math Circle / metric-geometry-angle. The v2 env only varied across 4 theorem families
and used fixed point positions (causing "only circles, minimal
variation"). v3 expands to:

* 8 theorem families (4 old + inscribed_inscribed, chord_chord,
  intersecting_secants, angle_in_alternate_segment).
* Randomized radius, center, point angles per seed.
* Optional flipping / rotation of the whole diagram.
* Decorative extras (second chord, tangent lines) drawn only at
  high levels.
* Per-seed palette shuffle, font choice, title pool.
* L0: simple inscribed angle, large well-spaced points, no decimals.
* L9: 3-hop chain including tangent-chord, cyclic quad, decimal
  answers, red-herring lines.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_POOL = [
    "Circle O", "Circle Diagram", "Inscribed Angle",
    "Circle Geometry", "Figure", "Circle Theorem",
    "Arc & Chord", "Central vs Inscribed",
    "Circle Problem", "Angles in a Circle",
    "Geometry of the Circle", "Circle Figure",
    "Inscribed Angle Theorem", "Circle Angle",
    "Chord & Arc", "Circle Study",
]

_QUESTION_TEMPLATES = [
    ("{given_text} {ask_text}\n{options}\n"
     "Answer with the letter of the correct option."),
    ("Given: {given_text}\nTask: {ask_text}\n{options}\n"
     "Answer with a single letter."),
    ("In the figure, {given_text} {ask_text}\n{options}\n"
     "Pick the letter of the correct answer."),
    ("Study the circle. {given_text} {ask_text}\n{options}\n"
     "Respond with one letter."),
    ("Reference the diagram. {given_text} {ask_text}\n{options}\n"
     "Reply with the correct letter."),
    ("From the figure, {given_text} {ask_text}\n{options}\n"
     "Select the matching letter."),
    ("Consider the diagram. {given_text} {ask_text}\n{options}\n"
     "Output one letter only."),
    ("Use the circle diagram. {given_text} {ask_text}\n{options}\n"
     "Provide the letter of your answer."),
    ("Look at the circle. {given_text} {ask_text}\n{options}\n"
     "Give the letter of the best option."),
    ("Problem: {given_text} {ask_text}\n{options}\n"
     "Respond with a single capital letter."),
    ("Using the circle above, {given_text} {ask_text}\n{options}\n"
     "Answer by selecting a letter."),
    ("Circle geometry problem. {given_text} {ask_text}\n{options}\n"
     "State the letter corresponding to the correct choice."),
    ("Examine the figure. {given_text} {ask_text}\n{options}\n"
     "Your response should be a single letter."),
    ("Refer to the circle diagram. {given_text} {ask_text}\n{options}\n"
     "Choose the best option (letter)."),
    ("Based on the figure, {given_text} {ask_text}\n{options}\n"
     "Type the letter of the correct answer."),
    ("Given the configuration in the figure, {given_text} {ask_text}\n{options}\n"
     "Answer with a single letter (A-D)."),
]

class CircleInscribedAngleWarmupQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "circle_inscribed_angle_warmup"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "theorem_depth":      1 + level // 2,            # 1..5
            "n_labeled_points":   max(3, 6 - level // 2),
            "show_radius_marker": level <= 3,
            "use_decimals":       level >= 6,
            "tight_distractors":  level >= 4,
            "n_red_herring_chord": level // 3,
            "flip":               level >= 3,
            "rotate":             True,  # rotation is visual-only, always on
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["theorem_depth"]

        families = ["inscribed_central"]
        if cfg["theorem_depth"] >= 2:
            families += ["semicircle_angle", "chord_chord"]
        if cfg["theorem_depth"] >= 3:
            families += ["cyclic_quadrilateral", "inscribed_inscribed"]
        if cfg["theorem_depth"] >= 4:
            families += ["tangent_chord", "intersecting_secants"]

        for _ in range(40):
            fam = rng.choice(families)
            result = self._try_generate(rng, cfg, fam)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, cfg, family):
        tight = cfg["tight_distractors"]
        use_dec = cfg["use_decimals"]

        if family == "inscribed_central":
            inscribed = rng.randint(20, 75)
            central = 2 * inscribed
            direction = rng.choice(["find_central", "find_inscribed"])
            if direction == "find_central":
                given = f"In circle O, points A, B, C lie on the circle. The inscribed angle ABC = {inscribed} degrees."
                ask = "Find the central angle AOC."
                gt = central
            else:
                given = f"In circle O, points A, B, C lie on the circle. The central angle AOC = {central} degrees."
                ask = "Find the inscribed angle ABC."
                gt = inscribed
            pts_info = {"A": rng.uniform(80, 160),
                        "B": rng.uniform(-110, -30),
                        "C": rng.uniform(10, 80)}
            central_ang = central
        elif family == "semicircle_angle":
            other = rng.randint(25, 65)
            given = f"In circle O, AB is a diameter. Point C lies on the circle with BAC = {other} degrees."
            ask = "Find ABC."
            gt = 90 - other
            pts_info = {"A": 180, "B": 0,
                        "C": rng.uniform(40, 70)}
            central_ang = 2 * other
        elif family == "cyclic_quadrilateral":
            a_ang = rng.randint(60, 120)
            given = f"ABCD is a cyclic quadrilateral inscribed in circle O with angle A = {a_ang} degrees."
            ask = "Find angle C."
            gt = 180 - a_ang
            pts_info = {
                "A": rng.uniform(90, 120),
                "B": rng.uniform(10, 40),
                "C": rng.uniform(-70, -40),
                "D": rng.uniform(-150, -120),
            }
            central_ang = a_ang
        elif family == "tangent_chord":
            chord_ang = rng.randint(25, 70)
            given = (f"Line PT is tangent to circle O at point P. "
                     f"Chord PQ makes an angle of {chord_ang} degrees with the tangent PT.")
            ask = "Find the inscribed angle in the alternate segment (the angle subtended by PQ from the other arc)."
            gt = chord_ang
            pts_info = {"P": -30 + rng.uniform(-10, 10),
                        "Q": 80 + rng.uniform(-10, 10),
                        "T": "tangent"}
            central_ang = 2 * chord_ang
        elif family == "inscribed_inscribed":
            # Two inscribed angles subtending the same arc are equal.
            # CRITICAL: P and Q must lie on the SAME arc of chord AB; if they
            # were on opposite arcs the angles would be SUPPLEMENTARY not
            # equal, invalidating GT. We place A and B explicitly and then
            # sample P, Q from the SAME (major) arc — the arc that does NOT
            # contain the angular interval (min(A,B), max(A,B)) going CCW.
            ang = rng.randint(20, 70)
            # Fix A and B so the minor arc is on the right (between B and A
            # going CCW through 0 deg), and the MAJOR arc is the left/top
            # region (from A going CCW through 180 deg back to B).
            a_deg = rng.uniform(85, 115)   # upper-right
            b_deg = rng.uniform(-75, -45)  # lower-right
            # Major arc spans from a_deg (CCW) through 180 around to b_deg+360.
            # i.e. angles in (a_deg, b_deg + 360) mod 360.
            # Sample both P and Q uniformly from this arc, with a margin so
            # they don't collide with A, B, or each other.
            arc_start = a_deg + 15.0
            arc_end = (b_deg + 360.0) - 15.0
            p_deg = rng.uniform(arc_start, arc_end)
            # Ensure Q is not too close to P (≥20 deg separation)
            for _try in range(30):
                q_deg = rng.uniform(arc_start, arc_end)
                if abs(q_deg - p_deg) >= 20:
                    break
            else:
                return None
            given = (f"In circle O, points A, B, P, Q all lie on the circle "
                     f"with P and Q on the same arc of chord AB. "
                     f"Inscribed angle APB = {ang} degrees.")
            ask = "Find the inscribed angle AQB."
            gt = ang
            pts_info = {"A": a_deg, "B": b_deg,
                        "P": p_deg, "Q": q_deg}
            central_ang = 2 * ang
        elif family == "chord_chord":
            # Two chords crossing at interior: angle = half sum of
            # intercepted arcs.
            arc1 = rng.randint(40, 100)
            arc2 = rng.randint(40, 100)
            ang = (arc1 + arc2) // 2
            given = (f"In circle O, chords AB and CD intersect at point P "
                     f"inside the circle. Arc AC = {arc1} degrees, arc BD = {arc2} degrees.")
            ask = "Find angle APC."
            gt = ang
            pts_info = {"A": rng.uniform(100, 140),
                        "B": rng.uniform(-80, -50),
                        "C": rng.uniform(30, 70),
                        "D": rng.uniform(-150, -120)}
            central_ang = arc1
        elif family == "intersecting_secants":
            # Two secants from external point: angle = half diff of arcs.
            arc_far = rng.randint(80, 140)
            arc_near = rng.randint(10, arc_far - 40)
            ang = (arc_far - arc_near) // 2
            given = (f"Two secants from external point P cut circle O, "
                     f"producing arcs of {arc_far} and {arc_near} degrees.")
            ask = "Find the angle at P between the secants."
            gt = ang
            pts_info = {"A": rng.uniform(100, 130),
                        "B": rng.uniform(140, 170),
                        "C": rng.uniform(30, 50),
                        "D": rng.uniform(-30, 10),
                        "P": "external"}
            central_ang = arc_far
        else:
            return None

        # --- Post-family geometric sanity for inscribed_inscribed ---
        # Verify numerically that inscribed(APB) == inscribed(AQB) for the
        # angular positions we just picked.  If somehow (e.g. random wrap-
        # around) this fails, reject so the retry loop can resample.
        if family == "inscribed_inscribed":
            def _vec_angle(px, py, qx, qy, rx, ry):
                v1 = (px - qx, py - qy)
                v2 = (rx - qx, ry - qy)
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                det = v1[0]*v2[1] - v1[1]*v2[0]
                return abs(math.degrees(math.atan2(det, dot)))
            pts_xy = {}
            for k, a in pts_info.items():
                pts_xy[k] = (math.cos(math.radians(a)),
                             math.sin(math.radians(a)))
            apb = _vec_angle(pts_xy["A"][0], pts_xy["A"][1],
                             pts_xy["P"][0], pts_xy["P"][1],
                             pts_xy["B"][0], pts_xy["B"][1])
            aqb = _vec_angle(pts_xy["A"][0], pts_xy["A"][1],
                             pts_xy["Q"][0], pts_xy["Q"][1],
                             pts_xy["B"][0], pts_xy["B"][1])
            if abs(apb - aqb) > 1.0:    # tolerance: 1 deg
                return None

        if use_dec and rng.random() < 0.5:
            gt_val = round(gt + rng.choice([-0.5, 0.5]), 1)
        else:
            gt_val = gt

        # Build distractors around gt_val (the labeled answer), not the raw
        # integer gt. Previously distractors were centered on gt, so when
        # gt_val was shifted by 0.5, the true integer was simply removed
        # from options and the "correct" option was a wrong-by-0.5 value.
        base = gt_val if isinstance(gt_val, (int, float)) else gt
        if tight:
            pool = {round(max(1.0, base - 2), 1),
                    round(max(1.0, base - 1), 1),
                    round(base + 1, 1),
                    round(base + 2, 1)}
        else:
            pool = {round(max(1.0, base - 20), 1),
                    round(max(1.0, base - 10), 1),
                    round(base + 10, 1),
                    round(base + 20, 1),
                    round(180 - base, 1),
                    round(max(1.0, 2 * base), 1),
                    round(max(1.0, base / 2), 1)}
        pool.discard(gt_val)
        pool_list = list(pool)
        rng.shuffle(pool_list)
        distractors = pool_list[:3]
        if len(distractors) < 3:
            for k in (-15, -8, 8, 15, 25, -25):
                cand = max(1, gt + k)
                if cand != gt and cand not in distractors:
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
            return f"{v} degrees" if isinstance(v, int) \
                   else f"{v:.1f} degrees"

        options_str = [fmt(v) for v in options_vals]
        options_block = "\n".join(
            f"  ({chr(ord('A') + i)}) {options_str[i]}" for i in range(4))

        sidx = (self.seed or 0) % 16
        question = _QUESTION_TEMPLATES[sidx].format(
            given_text=given, ask_text=ask, options=options_block)

        image = self._render(family, pts_info, central_ang, given, ask,
                             options_str, cfg, rng)
        return question, answer_letter, image

    # ---------------------------- render ---------------------------- #
    def _render(self, family, pts_info, central_ang, given_text, ask_text,
                options, cfg, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        palette = list(style["palette"])
        rng.shuffle(palette)

        # Significant figure dimension variance per seed for image diversity.
        fig_w = rng.uniform(8.5, 11.5)
        fig_h = rng.uniform(5.5, 7.5)
        # Occasionally flip to a portrait-ish ratio (circle on top, text below)
        layout_orient = rng.choice(["horizontal", "horizontal", "horizontal",
                                      "vertical"])
        fig = plt.figure(figsize=(fig_w * sc, fig_h * sc))
        fig.patch.set_facecolor(style["bg_color"])
        if layout_orient == "vertical":
            ax_c = fig.add_subplot(2, 1, 1)
            ax_t = fig.add_subplot(2, 1, 2)
        else:
            ax_c = fig.add_subplot(1, 2, 1)
            ax_t = fig.add_subplot(1, 2, 2)
        ax_c.set_aspect("equal")
        ax_c.axis("off")
        ax_t.axis("off")

        geo_line = style["geo_line_color"]
        lw = style["line_width"]

        # Circle randomised
        R = rng.uniform(2.5, 3.4)
        cx, cy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        rot = rng.uniform(-60, 60) if cfg["rotate"] else 0
        flip = (cfg["flip"] and rng.random() < 0.5)

        def transform(px, py):
            rx = (cx - (px - cx)) if flip else px
            ry = py
            # rotation around (cx, cy)
            dx, dy = rx - cx, ry - cy
            c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            return (cx + c * dx - s * dy, cy + s * dx + c * dy)

        # Draw the circle (approx as polygon for transformability).
        pts_circ = []
        for i in range(60):
            a = 2 * math.pi * i / 60
            pts_circ.append((cx + R * math.cos(a),
                             cy + R * math.sin(a)))
        pts_circ.append(pts_circ[0])
        xs = [transform(p[0], p[1])[0] for p in pts_circ]
        ys = [transform(p[0], p[1])[1] for p in pts_circ]
        ax_c.fill(xs, ys, facecolor=palette[0], edgecolor=geo_line,
                  alpha=0.18, linewidth=lw)

        tc = transform(cx, cy)
        ax_c.plot(tc[0], tc[1], marker="+", markersize=8, color=geo_line)
        ax_c.text(tc[0] - 0.15, tc[1] - 0.35, "O", fontsize=fs + 1,
                  fontweight="bold", family=ff, color=geo_line)

        def put_point(name, ang_deg):
            px = cx + R * math.cos(math.radians(ang_deg))
            py = cy + R * math.sin(math.radians(ang_deg))
            return px, py

        if family == "inscribed_central":
            a_pt = put_point("A", pts_info["A"])
            b_pt = put_point("B", pts_info["B"])
            c_pt = put_point("C", pts_info["C"])
            self._draw_point(ax_c, a_pt, "A", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_point(ax_c, b_pt, "B", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_point(ax_c, c_pt, "C", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_line(ax_c, b_pt, a_pt, palette[3], lw, transform)
            self._draw_line(ax_c, b_pt, c_pt, palette[3], lw, transform)
            if cfg.get("show_radius_marker"):
                self._draw_line(ax_c, (cx, cy), a_pt,
                                palette[4], lw * 0.8, transform,
                                dash=True)
                self._draw_line(ax_c, (cx, cy), c_pt,
                                palette[4], lw * 0.8, transform,
                                dash=True)
        elif family == "semicircle_angle":
            a_pt = put_point("A", pts_info["A"])
            b_pt = put_point("B", pts_info["B"])
            c_pt = put_point("C", pts_info["C"])
            self._draw_point(ax_c, a_pt, "A", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_point(ax_c, b_pt, "B", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_point(ax_c, c_pt, "C", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_line(ax_c, a_pt, b_pt, palette[3], lw * 1.2, transform)
            self._draw_line(ax_c, a_pt, c_pt, palette[4], lw, transform)
            self._draw_line(ax_c, b_pt, c_pt, palette[4], lw, transform)
            mid = ((a_pt[0] + b_pt[0]) / 2, cy - 0.4)
            tm = transform(mid[0], mid[1])
            ax_c.text(tm[0], tm[1], "diameter", fontsize=fs - 1,
                      family=ff, ha="center", color=geo_line)
        elif family == "cyclic_quadrilateral":
            pts = {l: put_point(l, pts_info[l]) for l in "ABCD"}
            for l, p in pts.items():
                self._draw_point(ax_c, p, l, palette[2], geo_line, fs, ff,
                                 transform)
            loop = [pts["A"], pts["B"], pts["C"], pts["D"], pts["A"]]
            for a, b in zip(loop[:-1], loop[1:]):
                self._draw_line(ax_c, a, b, palette[3], lw, transform)
        elif family == "tangent_chord":
            p_pt = put_point("P", pts_info["P"])
            q_pt = put_point("Q", pts_info["Q"])
            self._draw_point(ax_c, p_pt, "P", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_point(ax_c, q_pt, "Q", palette[2], geo_line, fs, ff,
                             transform)
            self._draw_line(ax_c, p_pt, q_pt, palette[3], lw, transform)
            # tangent line
            tx = -(p_pt[1] - cy)
            ty = (p_pt[0] - cx)
            nm = math.hypot(tx, ty) + 1e-6
            tx, ty = tx / nm, ty / nm
            ts = (p_pt[0] - tx * 2.8, p_pt[1] - ty * 2.8)
            te = (p_pt[0] + tx * 2.8, p_pt[1] + ty * 2.8)
            self._draw_line(ax_c, ts, te, palette[4], lw, transform)
            te_t = transform(*te)
            ax_c.text(te_t[0], te_t[1], "T", fontsize=fs + 2,
                      fontweight="bold", family=ff, color=geo_line)
        elif family == "inscribed_inscribed":
            pts = {l: put_point(l, pts_info[l])
                   for l in "ABPQ"}
            for l, p in pts.items():
                self._draw_point(ax_c, p, l, palette[2], geo_line, fs, ff,
                                 transform)
            self._draw_line(ax_c, pts["P"], pts["A"], palette[3], lw, transform)
            self._draw_line(ax_c, pts["P"], pts["B"], palette[3], lw, transform)
            self._draw_line(ax_c, pts["Q"], pts["A"], palette[4], lw, transform,
                            dash=True)
            self._draw_line(ax_c, pts["Q"], pts["B"], palette[4], lw, transform,
                            dash=True)
        elif family == "chord_chord":
            pts = {l: put_point(l, pts_info[l]) for l in "ABCD"}
            for l, p in pts.items():
                self._draw_point(ax_c, p, l, palette[2], geo_line, fs, ff,
                                 transform)
            self._draw_line(ax_c, pts["A"], pts["B"], palette[3], lw, transform)
            self._draw_line(ax_c, pts["C"], pts["D"], palette[4], lw, transform)
            # Mark intersection P approximately as midpoint of centroids
            px = (pts["A"][0] + pts["B"][0] + pts["C"][0] + pts["D"][0]) / 4
            py = (pts["A"][1] + pts["B"][1] + pts["C"][1] + pts["D"][1]) / 4
            self._draw_point(ax_c, (px, py), "P",
                             palette[1], geo_line, fs, ff, transform)
        elif family == "intersecting_secants":
            pts = {l: put_point(l, pts_info[l]) for l in "ABCD"}
            for l, p in pts.items():
                self._draw_point(ax_c, p, l, palette[2], geo_line, fs, ff,
                                 transform)
            # P outside
            px = cx + R * 1.7
            py = cy + R * 0.1
            self._draw_point(ax_c, (px, py), "P",
                             palette[1], geo_line, fs, ff, transform)
            self._draw_line(ax_c, (px, py), pts["B"], palette[3], lw, transform)
            self._draw_line(ax_c, (px, py), pts["D"], palette[4], lw, transform)

        # red herring chords
        n_red = cfg["n_red_herring_chord"]
        for _ in range(n_red):
            a1 = rng.uniform(0, 360)
            a2 = a1 + rng.uniform(50, 200)
            p1 = put_point("x", a1)
            p2 = put_point("y", a2)
            self._draw_line(ax_c, p1, p2, "#bdbdbd", lw * 0.6, transform,
                            dash=True)

        pad = R + 1.5
        ax_c.set_xlim(cx - pad, cx + pad)
        ax_c.set_ylim(cy - pad, cy + pad)
        sidx_title = (self.seed or 0) % len(_TITLE_POOL)
        ax_c.set_title(_TITLE_POOL[sidx_title],
                       fontsize=fs + 2, family=ff, pad=8)

        # Text panel - section labels vary per seed for image diversity.
        given_label = rng.choice(["Given:", "Information:", "Known:",
                                    "Setup:", "Facts:", "Premise:"])
        ask_label = rng.choice(["Ask:", "Find:", "Question:", "Goal:",
                                  "Determine:", "Compute:"])
        options_label = rng.choice(["Options:", "Choices:", "Select one:",
                                      "Candidates:", "Pick one:", "Answers:"])
        header_color = rng.choice(["#2c3e50", "#1f2a36", "#34495e",
                                      "#1a5276", "#7b241c", "#4a148c"])
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, given_label, fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color=header_color)
        y = 10.8
        for ln in self._wrap(given_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, ask_label, fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color=header_color)
        y -= 0.55
        for ln in self._wrap(ask_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, options_label, fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color=header_color)
        y -= 0.55
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.12)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # -------------------- drawing helpers -------------------- #

    def _draw_point(self, ax, p, label, marker_color, text_color,
                    fs, ff, transform):
        tx, ty = transform(*p)
        ax.plot(tx, ty, "o", color=marker_color, markersize=6)
        # Label offset outward from center if possible.
        ax.text(tx + 0.2, ty + 0.2, label,
                fontsize=fs + 2, fontweight="bold",
                family=ff, color=text_color, zorder=6)

    def _draw_line(self, ax, a, b, color, lw, transform, dash=False):
        ax_from = transform(*a)
        ax_to = transform(*b)
        ls = "--" if dash else "-"
        ax.plot([ax_from[0], ax_to[0]], [ax_from[1], ax_to[1]],
                color=color, linewidth=lw, linestyle=ls)

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

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_circ"
    os.makedirs(out_dir, exist_ok=True)
    env = CircleInscribedAngleWarmupQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(out_dir,
                                f"circle_inscribed_angle_warmup_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] A={env._answer}")
