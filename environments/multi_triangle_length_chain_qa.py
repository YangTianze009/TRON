"""
Multi-Triangle Length Chain QA environment.

Goal: find a missing length ``x`` in a figure composed of multiple
triangles that share sides. At low levels only Pythagoras is needed
(across 2 triangles). At higher levels the chain grows to 3-5 triangles
and requires similar-triangle ratios and/or the law of cosines.

Targets metric-geometry-length, Plane
Geometry reasoning.

Difficulty axes:
  A) n_triangles = 2 + level // 3  (2..5).
  B) theorem_mix: L0 = Pythagoras only, L5 = Pythagoras + similar,
     L9 = Pythagoras + cosine rule + similar.

Format: integer or decimal (length value).
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
from ._template_lib import measure_templates
from ._render_modes import pick_render_mode, sketch_context

# 16 phrasings of "find the length of x" (integer) — deterministic pick
# by seed to guarantee template diversity at L0 passes the audit floor.
_ASK_X_INT = [
    "Find the length of x.",
    "What is the length of x in the figure?",
    "In the figure, what is the value of x?",
    "Determine the length labeled x.",
    "Compute the length of side x.",
    "Find x. Answer with a single number.",
    "What is x in the figure shown?",
    "Report the length of x.",
    "The length of side x is ___. Fill in.",
    "Let x denote the length labeled in the figure. Find x.",
    "Inspect the figure and compute the length of x.",
    "Calculate the length labeled x.",
    "Using the labeled sides in the figure, find x.",
    "Based on the figure, what is the length of x?",
    "Identify the length x in the figure.",
    "Determine x based on the figure.",
]

_ASK_X_DEC = [
    "Find x (round to 2 decimals).",
    "Compute the length of x, rounded to 2 decimals.",
    "What is the length of x in the figure? Round to 2 decimals.",
    "Determine x. Round to 2 decimals.",
    "Find the length labeled x. Round to 2 decimals.",
    "Calculate x (round answer to 2 decimals).",
    "Report x, rounded to 2 decimal places.",
    "Based on the figure, compute x. Round to 2 decimals.",
    "The length of x is ___ (round to 2 decimals).",
    "In the figure, what is x to 2 decimal places?",
    "Using the figure, find x. Give your answer to 2 decimal places.",
    "Determine the length x (2-decimal precision).",
    "What is the length x shown? Provide 2 decimals.",
    "Inspect the figure and calculate x, rounded to 2 decimals.",
    "Find x (answer rounded to 2 decimals).",
    "Compute the labeled length x; round to 2 decimals.",
]

class MultiTriangleLengthChainQA(StandaloneVisualEnv):
    ENV_NAME = "multi_triangle_length_chain"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # Pythagorean triples for clean integer steps.
    _TRIPLES = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25),
                (8, 15, 17), (9, 12, 15), (9, 40, 41), (12, 16, 20),
                (20, 21, 29)]

    def _pick_ask_text(self, cfg) -> str:
        """Pick ask_text deterministically by seed from 16-template pool."""
        pool = _ASK_X_DEC if cfg.get("allow_decimal") else _ASK_X_INT
        return pool[(self.seed or 0) % len(pool)]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_triangles = 2 + level // 3  # 2,2,2,3,3,3,4,4,4,5
        if level <= 2:
            theorem_mix = ["pythag"]
        elif level <= 5:
            theorem_mix = ["pythag", "similar"]
        elif level <= 7:
            theorem_mix = ["pythag", "similar", "cosine"]
        else:
            theorem_mix = ["pythag", "cosine", "similar"]
        return {
            "n_triangles": n_triangles,
            "theorem_mix": theorem_mix,
            "label_hide_frac": 0.1 + 0.05 * level,
            "allow_decimal": level >= 2,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 977)
        self._primary_complexity_feature = cfg["n_triangles"]

        for _ in range(25):
            try:
                r = self._try_generate(sub_rng, cfg, level)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n = cfg["n_triangles"]
        theorems = cfg["theorem_mix"]

        if n == 2:
            return self._two_triangle(rng, theorems, cfg)
        if n == 3:
            return self._three_triangle(rng, theorems, cfg)
        if n == 4:
            return self._four_triangle(rng, theorems, cfg)
        return self._five_triangle(rng, theorems, cfg)

    # -------------------------------------------------- #
    # Problem families
    # -------------------------------------------------- #

    def _two_triangle(self, rng, theorems, cfg):
        """Two right triangles sharing common leg BD (horizontal).
        Triangle 1 = ABD (A above D, right angle at D).
        Triangle 2 = BDC (C below D, right angle at D).
        Find the diagonal BC of triangle 2."""
        t1 = rng.choice(self._TRIPLES)
        a1, b1, c1 = t1  # A=(0,a1), D=(0,0), B=(b1,0); AD=a1, BD=b1, AB=c1
        # Second triangle: C below D, DC vertical, BDC right-angled at D.
        # At L0/L1 (integer mode), pick a2 such that (b1, a2) forms a
        # Pythagorean pair from _TRIPLES — otherwise BC is irrational and
        # the integer-rounded GT can disagree with a numerically correct
        # model answer (e.g. sqrt(65) ≈ 8.06 rounds to 8, model writes 8.06).
        # At L2+ decimals are allowed so any a2 works.
        if cfg.get("allow_decimal"):
            a2 = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
            gt = math.sqrt(b1 ** 2 + a2 ** 2)
        else:
            pair_candidates = []
            for p, q, r in self._TRIPLES:
                if p == b1:
                    pair_candidates.append((q, r))
                if q == b1:
                    pair_candidates.append((p, r))
            if not pair_candidates:
                # Fallback: mirror the first triangle for an integer hyp.
                a2, hyp_bc = a1, c1
            else:
                a2, hyp_bc = rng.choice(pair_candidates)
            gt = float(hyp_bc)
        gt = self._round_answer(gt, cfg)

        given_text = (
            "Two right triangles share leg BD. Both right angles are at D. "
            "Triangle ABD is above, triangle BDC is below. "
            "Using the side lengths labeled in the figure, side x = BC."
        )
        ask_text = self._pick_ask_text(cfg)
        shape = {
            "family": "two_triangle",
            "A": (0, a1), "D": (0, 0), "B": (b1, 0), "C": (0, -a2),
            "bd": b1,
            "labels": {"AD": str(a1), "AB": str(c1), "BD": str(b1),
                       "DC": str(a2), "BC": "x"},
        }
        image = self._render(shape, given_text, ask_text, cfg)
        return self._build_qa(gt, given_text, ask_text, image, cfg)

    def _three_triangle(self, rng, theorems, cfg):
        """Three right triangles stacked like a staircase along the x-axis.
        Ask length of the outermost hypotenuse after 3 Pythagoras applications
        (or a similar-triangle ratio at L>=3)."""
        use_similar = "similar" in theorems and rng.random() < 0.5

        if use_similar:
            # Three similar right triangles with ratio k.
            base = rng.choice([(3, 4, 5), (5, 12, 13), (6, 8, 10)])
            a, b, c = base
            k1 = rng.choice([1, 2])
            k2 = k1 + rng.choice([1, 2])
            k3 = k2 + rng.choice([1, 2])
            gt = c * k3
            gt = self._round_answer(gt, cfg)
            given_text = (
                "Three similar right triangles are arranged in sequence, "
                "as shown. Using the side lengths labeled in the figure, "
                "side x is the hypotenuse of Triangle 3."
            )
            ask_text = self._pick_ask_text(cfg)
            shape = {
                "family": "chain_row", "n": 3,
                "tris": [(a*k1, b*k1, c*k1),
                         (a*k2, b*k2, c*k2),
                         (a*k3, b*k3, c*k3)],
                "ask_tri": 2,
            }
        else:
            # Chain of 3 right triangles stacked: each shares a leg with the next.
            a = rng.randint(3, 8)
            b = rng.randint(3, 8)
            c = rng.randint(3, 8)
            d = rng.randint(3, 8)
            # Build: triangle 1 legs (a, b); hyp = sqrt(a^2 + b^2)
            # triangle 2 uses that hyp as leg, plus new leg c;
            # triangle 3 uses triangle 2's hyp as leg, plus new leg d.
            h1 = math.sqrt(a * a + b * b)
            h2 = math.sqrt(h1 * h1 + c * c)
            h3 = math.sqrt(h2 * h2 + d * d)
            gt = self._round_answer(h3, cfg)
            given_text = (
                "A chain of 3 right triangles, each one's hypotenuse "
                "becoming a leg of the next, as shown. Using the leg "
                "lengths labeled in the figure, side x is the "
                "hypotenuse of Triangle 3."
            )
            ask_text = self._pick_ask_text(cfg)
            shape = {
                "family": "spiral_chain", "n": 3,
                "legs": [a, b, c, d],
            }

        image = self._render(shape, given_text, ask_text, cfg)
        return self._build_qa(gt, given_text, ask_text, image, cfg)

    def _four_triangle(self, rng, theorems, cfg):
        """Four right triangles in a spiral (Theodorus-like)."""
        legs = [rng.randint(3, 7) for _ in range(5)]
        vals = [float(legs[0])]  # first hypotenuse baseline = legs[0]
        for i in range(1, 5):
            vals.append(math.sqrt(vals[-1] ** 2 + legs[i] ** 2))
        gt = self._round_answer(vals[-1], cfg)
        given_text = (
            "A spiral of 4 right triangles, as shown. Each subsequent "
            "triangle adds a new leg perpendicular to the previous "
            "hypotenuse. Using the leg lengths labeled in the figure, "
            "side x is the final hypotenuse."
        )
        ask_text = self._pick_ask_text(cfg)
        shape = {"family": "spiral_chain", "n": 4, "legs": legs}
        image = self._render(shape, given_text, ask_text, cfg)
        return self._build_qa(gt, given_text, ask_text, image, cfg)

    def _five_triangle(self, rng, theorems, cfg):
        """Five triangles: 3 right triangles plus 2 non-right (law of cosines)."""
        use_cosine = "cosine" in theorems and rng.random() < 0.7
        if use_cosine:
            a = rng.randint(4, 10)
            b = rng.randint(4, 10)
            angle_deg = rng.choice([60, 75, 90, 105, 120])
            gt = math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(angle_deg)))
            # Add one more step using another Pythagoras with another triangle.
            extra = rng.randint(3, 8)
            gt = math.sqrt(gt * gt + extra * extra)
            gt = self._round_answer(gt, cfg)
            given_text = (
                "Consider a chain of triangles, as shown. Triangle P has "
                "two sides meeting at a marked angle; its third side is "
                "shared with Triangle Q (a right triangle). Using the "
                "side lengths and angle labeled in the figure, side x is "
                "Triangle Q's hypotenuse."
            )
            # Cosine-chain: keep the method hint but vary the outer phrasing.
            cosine_pool = [
                "Find x. Use law of cosines then Pythagoras (round to 2 decimals).",
                "Compute x using the law of cosines followed by Pythagoras (round to 2 decimals).",
                "Determine x via law of cosines and Pythagoras (round to 2 decimals).",
                "Apply the law of cosines, then Pythagoras, to find x (round to 2 decimals).",
                "Find the length of x; use law of cosines then Pythagoras (round to 2 decimals).",
                "Using cosine rule + Pythagoras, compute x (round to 2 decimals).",
                "Calculate x by applying the law of cosines and then Pythagoras (round to 2 decimals).",
                "With the law of cosines and Pythagoras, find x (round to 2 decimals).",
                "Solve for x using cosine rule and Pythagoras (round to 2 decimals).",
                "Determine the length x via law of cosines + Pythagoras (round to 2 decimals).",
                "Find x. Start with the law of cosines, finish with Pythagoras (round to 2 decimals).",
                "Compute the length labeled x using cosine rule, then Pythagoras (round to 2 decimals).",
                "Apply cosine rule + Pythagoras in sequence to find x (round to 2 decimals).",
                "Using the cosine rule and Pythagorean theorem, find x (round to 2 decimals).",
                "What is the length of x? Use law of cosines + Pythagoras (round to 2 decimals).",
                "Find x by combining cosine rule and Pythagoras (round to 2 decimals).",
            ]
            ask_text = cosine_pool[(self.seed or 0) % len(cosine_pool)]
            shape = {
                "family": "cosine_chain", "a": a, "b": b,
                "angle": angle_deg, "extra": extra,
            }
        else:
            legs = [rng.randint(3, 7) for _ in range(6)]
            vals = [float(legs[0])]
            for i in range(1, 6):
                vals.append(math.sqrt(vals[-1] ** 2 + legs[i] ** 2))
            gt = self._round_answer(vals[-1], cfg)
            given_text = (
                "A spiral of 5 right triangles, as shown. Each "
                "subsequent triangle shares a leg with the prior "
                "hypotenuse and adds a new perpendicular leg. Using "
                "the leg lengths labeled in the figure, side x is the "
                "final hypotenuse."
            )
            ask_text = self._pick_ask_text(cfg)
            shape = {"family": "spiral_chain", "n": 5, "legs": legs}

        image = self._render(shape, given_text, ask_text, cfg)
        return self._build_qa(gt, given_text, ask_text, image, cfg)

    # -------------------------------------------------- #
    # Helpers
    # -------------------------------------------------- #

    def _round_answer(self, v, cfg):
        if cfg.get("allow_decimal"):
            rv = round(v, 2)
            if abs(rv - round(rv)) < 1e-6:
                return int(round(rv))
            return rv
        return int(round(v))

    def _build_qa(self, gt, given, ask, image, cfg):
        if isinstance(gt, int):
            ans_str = str(gt)
            tail = "Answer with a single integer."
        else:
            ans_str = f"{gt:.2f}"
            tail = "Answer with a single decimal."
        q = f"{given} {ask} {tail}"
        return q, ans_str, image

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _render(self, shape, given_text, ask_text, cfg) -> Image.Image:
        # Dispatch on render mode; sketch wraps the full inner render in an
        # xkcd context so the lines & labels pick up hand-drawn wobble.
        mode = pick_render_mode(self._rng)
        if mode == "sketch":
            with sketch_context(scale=1.0, length=80, randomness=1.5):
                return self._render_impl(shape, given_text, ask_text, cfg, mode)
        return self._render_impl(shape, given_text, ask_text, cfg, mode)

    def _render_impl(self, shape, given_text, ask_text, cfg, mode) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        palette = style["palette"]
        lw = style["line_width"]
        geo_line = style["geo_line_color"]

        # Mode-specific overrides for background + geo line color.
        if mode == "textbook":
            bg_override = self._rng.choice(["#ffffff", "#fefefe", "#fdfdfd"])
            geo_line = self._rng.choice(["#111111", "#222222", "#1a1a1a"])
        elif mode == "sketch":
            bg_override = self._rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            geo_line = "#1a1a1a"
        else:
            bg_override = style["bg_color"]

        rng = self._rng
        rotate = rng.uniform(-0.15, 0.15) if rng.random() < 0.5 else 0.0

        fig = plt.figure(figsize=(9.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(bg_override)
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        fam = shape["family"]
        def R(pt):
            c, s = math.cos(rotate), math.sin(rotate)
            return (pt[0] * c - pt[1] * s, pt[0] * s + pt[1] * c)

        if fam == "two_triangle":
            A, D, B, C = shape["A"], shape["D"], shape["B"], shape["C"]
            pts = {"A": R(A), "D": R(D), "B": R(B), "C": R(C)}
            poly_abd = mpatches.Polygon(
                [pts["A"], pts["B"], pts["D"]], closed=True,
                facecolor=palette[0], edgecolor=geo_line,
                linewidth=lw, alpha=0.30)
            poly_bdc = mpatches.Polygon(
                [pts["B"], pts["D"], pts["C"]], closed=True,
                facecolor=palette[2], edgecolor=geo_line,
                linewidth=lw, alpha=0.30)
            ax_f.add_patch(poly_abd)
            ax_f.add_patch(poly_bdc)
            for k, p in pts.items():
                ax_f.text(p[0] + 0.05, p[1] + 0.15, k,
                          fontsize=fs + 2, fontweight="bold",
                          family=ff, color=geo_line)
            labs = shape["labels"]
            mid = lambda p1, p2: ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            lbl_color = "#c0392b"
            for pair, name in [(("A", "D"), "AD"), (("A", "B"), "AB"),
                               (("B", "D"), "BD"), (("D", "C"), "DC"),
                               (("B", "C"), "BC")]:
                if name in labs:
                    m = mid(pts[pair[0]], pts[pair[1]])
                    ax_f.text(m[0], m[1] + 0.15, labs[name],
                              fontsize=fs + 1, family=ff, ha="center",
                              color=lbl_color, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.15",
                                        facecolor="white", edgecolor="#ccc",
                                        alpha=0.90))
            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            pad = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.15 + 0.8
            ax_f.set_xlim(min(xs) - pad, max(xs) + pad)
            ax_f.set_ylim(min(ys) - pad, max(ys) + pad)

        elif fam == "spiral_chain":
            # Draw concentric spiral with n triangles.
            n = shape["n"]
            legs = shape["legs"]
            # Start with triangle 1 in xy-plane.
            # p0 = origin, p1 = (legs[0], 0), p2 = (legs[0], legs[1]) right angle at p1
            verts = [(0.0, 0.0), (legs[0], 0.0),
                     (legs[0], legs[1])]
            tris = [verts[:]]
            # Track which edge of each triangle is the newly-added leg.
            # Triangle 0: new leg is (p1 -> p2) with length legs[1].
            new_leg_edges = [(verts[1], verts[2], legs[1])]
            # Also label the initial base leg (p0 -> p1) length legs[0]:
            base_leg_edge = (verts[0], verts[1], legs[0])
            # Each subsequent triangle: hypotenuse of previous becomes one leg;
            # new leg perpendicular to previous hypotenuse.
            for i in range(1, n):
                p_start = verts[0]
                p_end = verts[-1]
                dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
                L = math.hypot(dx, dy)
                # Perpendicular direction (rotated +90 from hypotenuse)
                px, py = -dy / L, dx / L
                leg_len = legs[i + 1]
                new_pt = (p_end[0] + px * leg_len, p_end[1] + py * leg_len)
                tris.append([p_start, p_end, new_pt])
                new_leg_edges.append((p_end, new_pt, leg_len))
                verts.append(new_pt)
            # Rotate all points.
            for i, tri in enumerate(tris):
                rot = [R(p) for p in tri]
                col = palette[(i + 1) % len(palette)]
                poly = mpatches.Polygon(rot, closed=True, facecolor=col,
                                         edgecolor=geo_line, linewidth=lw,
                                         alpha=0.28)
                ax_f.add_patch(poly)
                cx = sum(p[0] for p in rot) / 3
                cy = sum(p[1] for p in rot) / 3
                ax_f.text(cx, cy, f"T{i+1}", fontsize=fs - 1, family=ff,
                          ha="center", va="center", color=geo_line,
                          fontweight="bold")
            # Label the base leg and each new leg with its numeric length.
            lbl_color = "#c0392b"
            def _lbl_edge(p1, p2, txt):
                p1r, p2r = R(p1), R(p2)
                mx = (p1r[0] + p2r[0]) / 2
                my = (p1r[1] + p2r[1]) / 2
                # small offset perpendicular to edge
                dx, dy = p2r[0] - p1r[0], p2r[1] - p1r[1]
                L = math.hypot(dx, dy) + 1e-9
                ox, oy = -dy / L * 0.25, dx / L * 0.25
                ax_f.text(mx + ox, my + oy, txt,
                          fontsize=fs, family=ff, ha="center",
                          color=lbl_color, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                    facecolor="white", edgecolor="#ccc",
                                    alpha=0.90))

            _lbl_edge(base_leg_edge[0], base_leg_edge[1],
                      str(base_leg_edge[2]))
            for (a, b, v) in new_leg_edges:
                _lbl_edge(a, b, str(v))
            # Label x at the outermost added point
            last = R(verts[-1])
            ax_f.text(last[0] + 0.3, last[1] + 0.3, "x",
                      fontsize=fs + 3, fontweight="bold",
                      family=ff, color=palette[5])
            xs = [R(v)[0] for v in verts]
            ys = [R(v)[1] for v in verts]
            pad = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.2 + 1.0
            ax_f.set_xlim(min(xs) - pad, max(xs) + pad)
            ax_f.set_ylim(min(ys) - pad, max(ys) + pad)

        elif fam == "chain_row":
            tris = shape["tris"]
            offset_x = 0
            pts_all = []
            last_idx = len(tris) - 1
            for i, (a, b, c) in enumerate(tris):
                v = [(offset_x, 0), (offset_x + a, 0), (offset_x, b)]
                pts_all.extend(v)
                rot_v = [R(p) for p in v]
                poly = mpatches.Polygon(rot_v, closed=True,
                                         facecolor=palette[(i + 1) % len(palette)],
                                         edgecolor=geo_line, linewidth=lw,
                                         alpha=0.3)
                ax_f.add_patch(poly)
                cx = sum(p[0] for p in rot_v) / 3
                cy = sum(p[1] for p in rot_v) / 3
                ax_f.text(cx, cy, f"T{i+1}", fontsize=fs - 1, family=ff,
                          ha="center", va="center", color=geo_line,
                          fontweight="bold")
                # label legs a and b for every triangle; only label hyp of
                # the final one as "x" (handled below) — others get numeric.
                lbl_color = "#c0392b"
                mx_a = (rot_v[0][0] + rot_v[1][0]) / 2
                my_a = (rot_v[0][1] + rot_v[1][1]) / 2
                ax_f.text(mx_a, my_a - 0.5, f"{a}",
                          fontsize=fs, family=ff, ha="center",
                          color=lbl_color, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                    facecolor="white", edgecolor="#ccc",
                                    alpha=0.9))
                mx_b = (rot_v[0][0] + rot_v[2][0]) / 2
                my_b = (rot_v[0][1] + rot_v[2][1]) / 2
                ax_f.text(mx_b - 0.5, my_b, f"{b}",
                          fontsize=fs, family=ff, ha="center",
                          color=lbl_color, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.15",
                                    facecolor="white", edgecolor="#ccc",
                                    alpha=0.9))
                if i != last_idx:
                    # hyp numeric label for non-final triangles
                    mx_c = (rot_v[1][0] + rot_v[2][0]) / 2
                    my_c = (rot_v[1][1] + rot_v[2][1]) / 2
                    ax_f.text(mx_c + 0.3, my_c + 0.2, f"{c}",
                              fontsize=fs, family=ff, ha="center",
                              color=lbl_color, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.15",
                                        facecolor="white", edgecolor="#ccc",
                                        alpha=0.9))
                offset_x += a + 1.5
            # Mark x at hypotenuse of last triangle
            ax_f.text(pts_all[-3][0] + 0.3, pts_all[-1][1] * 0.6,
                      "x", fontsize=fs + 3, fontweight="bold",
                      family=ff, color=palette[5])
            xs = [R(p)[0] for p in pts_all]
            ys = [R(p)[1] for p in pts_all]
            pad = 1.0
            ax_f.set_xlim(min(xs) - pad, max(xs) + pad)
            ax_f.set_ylim(min(ys) - pad, max(ys) + pad)

        elif fam == "cosine_chain":
            a = shape["a"]
            b = shape["b"]
            ang = math.radians(shape["angle"])
            extra = shape["extra"]
            # Triangle P: (0,0), (a,0), (b*cos, b*sin)
            p0 = (0.0, 0.0)
            p1 = (float(a), 0.0)
            p2 = (b * math.cos(ang), b * math.sin(ang))
            P = [p0, p1, p2]
            # Side s length from p1 to p2
            sx = p2[0] - p1[0]
            sy = p2[1] - p1[1]
            sL = math.hypot(sx, sy)
            # Perpendicular from p2 going right, length extra
            px, py = sy / sL, -sx / sL
            q_outer = (p2[0] + px * extra, p2[1] + py * extra)
            Q = [p1, p2, q_outer]
            for tri, col in [(P, palette[0]), (Q, palette[2])]:
                rot = [R(p) for p in tri]
                poly = mpatches.Polygon(rot, closed=True, facecolor=col,
                                         edgecolor=geo_line, linewidth=lw,
                                         alpha=0.32)
                ax_f.add_patch(poly)
            ax_f.text(*R(((p0[0]+p1[0])/2, -0.35)),
                      s=f"{a}", fontsize=fs, family=ff,
                      ha="center", color=palette[4],
                      fontweight="bold")
            ax_f.text(*R((p2[0]/2 - 0.3, p2[1]/2 + 0.2)),
                      s=f"{b}", fontsize=fs, family=ff,
                      ha="center", color=palette[4],
                      fontweight="bold")
            # Label the 'extra' leg in Triangle Q (p2 -> q_outer)
            emx = (p2[0] + q_outer[0]) / 2
            emy = (p2[1] + q_outer[1]) / 2
            ax_f.text(*R((emx + 0.25, emy + 0.25)),
                      s=f"{extra}", fontsize=fs, family=ff,
                      ha="center", color=palette[4],
                      fontweight="bold")
            # x label near outer hypotenuse of Q
            mx = (p1[0] + q_outer[0]) / 2
            my = (p1[1] + q_outer[1]) / 2
            ax_f.text(*R((mx + 0.3, my)),
                      s="x", fontsize=fs + 3, fontweight="bold",
                      family=ff, color=palette[5])
            # Angle marker
            ax_f.text(*R((0.4, 0.25)),
                      s=f"{shape['angle']} deg", fontsize=fs - 1,
                      family=ff, ha="center", color=palette[1],
                      fontweight="bold")
            xs_all = [R(p)[0] for p in P + Q]
            ys_all = [R(p)[1] for p in P + Q]
            pad = 1.5
            ax_f.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
            ax_f.set_ylim(min(ys_all) - pad, max(ys_all) + pad)

        title_pool = ["Multi-Triangle Chain", "Length Chain",
                      "Triangle Network", "Geometry Chain", "Linked Triangles"]
        ax_f.set_title(rng.choice(title_pool),
                       fontsize=fs + 2, family=ff, pad=8)

        # Text panel
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Given:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        for ln in self._wrap(given_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax_t.text(0.3, y, "Ask:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for ln in self._wrap(ask_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.12)
        return self.fig_to_pil(fig, dpi=style["dpi"])

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
