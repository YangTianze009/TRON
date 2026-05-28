"""
Quadrilateral Angle Sum QA environment (redesigned 2026-04-16).

Goal: angle chasing in quadrilaterals — general quads, parallelograms,
trapezoids, kites, cyclic quadrilaterals.

Critical fix (vs Grade D baseline):
  * The old render drew ALL FOUR angle values on the figure, including
    the unknown that the question asks about — the answer was visible.
  * Now: only the given angles are drawn on the image (and only those
    angles appear in the question text). The queried vertex shows "?"
    instead of its numeric value.
  * 5+ question templates for each kind
  * Diverse colors, line styles per seed
  * Randomized vertex perturbation for variety
  * L0/L9 structural shift: L0 is a single-hop general quad;
    L9 is multi-hop on cyclic / kite with near-distractors.
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

_FIGURE_TITLES = [
    "Quadrilateral",
    "Figure",
    "Angle diagram",
    "Geometry figure",
    "Diagram",
    "Shape",
    "ABCD",
]

class QuadrilateralAngleSumQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "quadrilateral_angle_sum"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    # 2026-05-04 R4: full-gradient redesign per a math benchmark quadrilateral.
    # L0-L1: trivial general quad with 3 angles given (1-step subtract from 360)
    # L2-L3: parallelogram (use opp-equal / supp rule, 1-step)
    # L4-L5: trapezoid + kite (1-step with shape-specific rule)
    # L6-L7: MULTI-QUAD CHAIN — two adjacent quads sharing edge BC; given
    #        3 angles in quad ABCD + offset to quad BCEF, find angle at E
    #        (2-step: derive shared angle, then apply offset)
    # L8-L9: cyclic chain (existing) + cyclic triple unknown (existing 4-eq system)
    _KIND_BY_LEVEL = [
        "general", "general",
        "parallelogram", "parallelogram",
        "trapezoid", "kite",
        "multi_quad_chain", "multi_quad_chain",
        "cyclic_chain", "cyclic_triple_unknown",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "kind": self._KIND_BY_LEVEL[level],
            "n_hops": 2 + level // 2,
            "tight_distractors": level >= 2,
            "visible_label_frac": max(0.30, 0.85 - 0.08 * level),
            # From L3 onwards, keep the given-angle numeric values off
            # the question text so the learner must read them from the
            # image (prevents text-only solvability).
            "hide_given_values_in_text": level >= 3,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        vis_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 643)
        self._primary_complexity_feature = cfg["n_hops"]

        for _ in range(40):
            r = self._try_generate(rng, vis_rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, vis_rng, cfg
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        kind = cfg["kind"]

        # angles[i] corresponds to vertex ABCD[i]
        # given_indices are the vertices whose values are given in problem
        # queried_index is the vertex the question asks about
        if kind == "general":
            a = rng.randint(50, 120)
            b = rng.randint(50, 120)
            c = rng.randint(50, 120)
            d = 360 - a - b - c
            if d < 25 or d > 170:
                return None
            angles = [a, b, c, d]
            given_indices = [0, 1, 2]
            queried_index = 3
            gt = d
        elif kind == "parallelogram":
            ab = rng.randint(50, 130)
            angles = [ab, 180 - ab, ab, 180 - ab]
            # Choose which vertex to ask
            which = rng.choice(["B", "C", "D"])
            queried_index = {"B": 1, "C": 2, "D": 3}[which]
            gt = angles[queried_index]
            given_indices = [0]  # only A is given
        elif kind == "trapezoid":
            a = rng.randint(50, 120)
            b = rng.randint(50, 120)
            c = 180 - b
            d = 180 - a
            angles = [a, b, c, d]
            which = rng.choice(["C", "D"])
            queried_index = {"C": 2, "D": 3}[which]
            gt = angles[queried_index]
            given_indices = [0, 1]
        elif kind == "kite":
            a_ang = rng.randint(60, 120)
            bd = rng.randint(40, 90)
            c_ang = 360 - a_ang - 2 * bd
            if c_ang < 30 or c_ang > 200:
                return None
            angles = [a_ang, bd, c_ang, bd]
            which = rng.choice(["C", "D"])
            queried_index = {"C": 2, "D": 3}[which]
            gt = angles[queried_index]
            given_indices = [0, 1]
        elif kind == "cyclic":
            a = rng.randint(60, 130)
            b = rng.randint(60, 130)
            c = 180 - a
            d = 180 - b
            angles = [a, b, c, d]
            which = rng.choice(["C", "D"])
            queried_index = {"C": 2, "D": 3}[which]
            gt = angles[queried_index]
            given_indices = [0, 1]
        elif kind == "multi_quad_chain":
            # 2026-05-04 R4: Two adjacent quadrilaterals ABCD and BCEF
            # sharing edge BC. Given ∠A, ∠B (in ABCD), ∠D and the
            # additional fact ∠F = ∠B + k (offset constraint between the
            # two quads). Find ∠E. (Quad sum = 360°, so ∠C in BCEF =
            # 360 - ∠B - ∠F - ∠E. We tell model the OFFSET k and ask ∠E
            # given ∠C in second quad, derived from a labeled relationship.)
            # For simplicity: render TWO quads side-by-side. Quad ABCD has
            # 3 angles labeled; quad BCEF has 2 angles labeled. Ask the
            # missing 4th angle in quad BCEF.
            # We render this as a single "general" quadrilateral on the
            # image (the second quad), but the question text describes the
            # chain, with all numeric inputs in TEXT.
            for _ in range(20):
                a1 = rng.randint(60, 120)
                b1 = rng.randint(60, 120)
                d1 = rng.randint(60, 120)
                c1 = 360 - a1 - b1 - d1
                if not (40 <= c1 <= 170):
                    continue
                # Second quad shares edge BC. Angles at B' and C' on the
                # second quad relate to (180-b1) and (180-c1) respectively
                # (supplementary on a straight line). Pick small offset k
                # for E and ask F (the 4th vertex of second quad).
                b2_angle = 180 - b1
                c2_angle = 180 - c1
                e_angle = rng.randint(50, 130)
                f_angle = 360 - b2_angle - c2_angle - e_angle
                if not (40 <= f_angle <= 170):
                    continue
                break
            else:
                return None
            # We use the second quadrilateral as the rendered figure so
            # vertex names match BCEF -> B,C,E,F (mapped to A,B,C,D for
            # render). queried = F position (index 3), other 3 given.
            angles = [b2_angle, c2_angle, e_angle, f_angle]
            given_indices = [0, 1, 2]
            queried_index = 3
            gt = f_angle
            # Stash chain-context for intro text
            self._mq_a1 = a1
            self._mq_b1 = b1
            self._mq_d1 = d1
            self._mq_c1 = c1
        elif kind == "cyclic_chain":
            # L9: cyclic quad but we're given ∠C and a second fact relating
            # ∠B and ∠D (e.g., ∠B = ∠D + k). Ask for the SUM ∠A + ∠B
            # (two-step: first find ∠A=180-∠C via cyclic, then solve the
            # system for ∠B, then add).
            a = rng.randint(60, 130)
            # pick k as the B-D offset; ensure both end up in (30,150) and
            # both sum to 180 (since cyclic)
            # B + D = 180, B = D + k → D = (180-k)/2, B = (180+k)/2
            k = rng.choice([-40, -20, -10, 10, 20, 40])
            if (180 - k) % 2 != 0:
                k += 1
            d_ang = (180 - k) // 2
            b_ang = (180 + k) // 2
            if not (30 <= d_ang <= 150 and 30 <= b_ang <= 150):
                return None
            c_ang = 180 - a
            angles = [a, b_ang, c_ang, d_ang]
            # Give ∠C on image + constraint ∠B = ∠D + k on image text.
            given_indices = [2]  # only ∠C visible
            # query: sum ∠A + ∠B. Mark A as queried_index so "?" appears at A.
            # (B will simply not render a value — the _render loop sets
            # val_text=None for non-given, non-queried vertices.)
            queried_index = 0
            gt = a + b_ang
            # Stash the B-D offset so the intro can mention it.
            self._cyclic_chain_k = k
        elif kind == "cyclic_triple_unknown":
            # L9 iter-3: cyclic quad inscribed in a circle with THREE
            # simultaneous constraints that pin all four angles uniquely.
            # System of equations (all printed on the image as text):
            #   (1) ∠A + ∠C = 180        (cyclic opposite-angle)
            #   (2) ∠B + ∠D = 180        (cyclic opposite-angle)
            #   (3) ∠A = 2·∠D − m        (given on image)
            #   (4) ∠B = ∠C + k          (given on image)
            # Combine: A = 2D - m, so C = 180-A = 180-2D+m
            # From (4): B = C + k = 180-2D+m+k, and from (2) B = 180-D.
            # ⇒ 180-2D+m+k = 180-D  ⇒  D = m+k.
            # So D = m+k, A = 2(m+k)-m = m+2k, C=180-m-2k, B=180-m-k.
            # 2026-05-04: bumped L9 difficulty (was 100% saturated → wider
            # m,k integer pools so model must do real arithmetic on the linear
            # system rather than recognize a small set of canonical patterns).
            for _ in range(60):
                m = rng.choice([13, 17, 23, 28, 32, 37, 43, 47, 52, 58])
                k = rng.choice([7, 11, 14, 19, 22, 26, 29, 33, 36])
                d_ang = m + k
                a_ang = m + 2 * k
                c_ang = 180 - m - 2 * k
                b_ang = 180 - m - k
                if not (25 <= d_ang <= 155 and 25 <= b_ang <= 155 and
                        25 <= c_ang <= 155 and 25 <= a_ang <= 155):
                    continue
                break
            else:
                return None
            angles = [a_ang, b_ang, c_ang, d_ang]
            # No angle labeled; constraints in text.
            given_indices = []  # nothing shown at vertices
            queried_index = 0   # mark A as queried
            gt = a_ang + c_ang + d_ang  # three-angle sum
            self._triple_m = m
            self._triple_k = k
        else:
            return None

        # Build question text — only mentions given angles, not answer
        vertex_names = ["A", "B", "C", "D"]
        if cfg.get("hide_given_values_in_text"):
            # Generic form — values only on image.
            names_only = ", ".join(f"\u2220{vertex_names[i]}"
                                   for i in given_indices)
            given_str = f"angles at {names_only} are labeled on the figure"
        else:
            given_str_list = [f"\u2220{vertex_names[i]} = {angles[i]}\u00b0"
                              for i in given_indices]
            given_str = ", ".join(given_str_list)

        if kind == "general":
            intro = rng.choice([
                f"In quadrilateral ABCD, {given_str}.",
                f"Quadrilateral ABCD has {given_str}.",
                f"In the quadrilateral ABCD shown, {given_str}.",
            ])
        elif kind == "parallelogram":
            intro = rng.choice([
                f"ABCD is a parallelogram with {given_str}.",
                f"In parallelogram ABCD, {given_str}.",
            ])
        elif kind == "trapezoid":
            intro = rng.choice([
                f"ABCD is a trapezoid with AB parallel to CD. {given_str}.",
                f"In trapezoid ABCD (AB parallel to CD), {given_str}.",
            ])
        elif kind == "kite":
            intro = rng.choice([
                f"ABCD is a kite (AB = AD, CB = CD). {given_str}.",
                f"In kite ABCD with AB = AD and CB = CD, {given_str}.",
            ])
        elif kind == "multi_quad_chain":
            a1 = getattr(self, "_mq_a1", 90)
            b1 = getattr(self, "_mq_b1", 90)
            d1 = getattr(self, "_mq_d1", 90)
            c1 = getattr(self, "_mq_c1", 90)
            # Re-read renamed labels: image shows BCEF (mapped to A,B,C,D
            # in render order). Tell model in text.
            intro = rng.choice([
                f"Two adjacent quadrilaterals share edge BC. In quad ABCD "
                f"(not shown), ∠A = {a1}°, ∠B = {b1}°, ∠D = {d1}°. The "
                f"second quad BCEF (shown in the figure, with vertices "
                f"labeled A,B,C,D corresponding to B,C,E,F) shares the "
                f"side BC. The angles at B and C of the second quad are "
                f"supplementary to ∠B and ∠C of ABCD respectively (linear "
                f"pair across the shared edge). The angle at E (vertex C "
                f"in the figure) and the unknown ∠F (vertex D in the "
                f"figure, marked '?') are the other two angles of BCEF. "
                f"The labeled angles on the figure are the three given "
                f"angles of the second quad.",
            ])
        elif kind == "cyclic_chain":
            k = getattr(self, "_cyclic_chain_k", 0)
            k_str = f"+ {k}" if k >= 0 else f"- {abs(k)}"
            intro = rng.choice([
                f"ABCD is a cyclic quadrilateral. The measure of ∠C is labeled "
                f"on the figure, and additionally ∠B = ∠D {k_str}°.",
                f"ABCD is inscribed in a circle. ∠C is labeled on the image, "
                f"and a side-note gives ∠B = ∠D {k_str}°.",
            ])
        elif kind == "cyclic_triple_unknown":
            m = getattr(self, "_triple_m", 0)
            k_off = getattr(self, "_triple_k", 0)
            intro = rng.choice([
                f"ABCD is a cyclic quadrilateral (inscribed in the dashed "
                f"circle in the figure). TWO additional constraints are "
                f"printed on the image as side-notes: "
                f"(i) ∠A = 2·∠D − {m}° and "
                f"(ii) ∠B = ∠C + {k_off}°. "
                f"Combined with the cyclic-quad rules (∠A+∠C=180°, "
                f"∠B+∠D=180°), these 4 equations uniquely determine all "
                f"four angles.",
                f"ABCD is inscribed in a circle. Two linear side-notes are "
                f"printed in the figure: ∠A = 2·∠D − {m}° and "
                f"∠B = ∠C + {k_off}°. Use these together with ∠A+∠C=180° "
                f"and ∠B+∠D=180° to solve the 4-equation system.",
            ])
        else:  # cyclic
            intro = rng.choice([
                f"ABCD is a cyclic quadrilateral (inscribed in a circle). "
                f"{given_str}.",
                f"ABCD is inscribed in a circle. {given_str}.",
            ])

        if kind == "multi_quad_chain":
            ask_vertex = "D"  # queried_index = 3 -> letter D
            ask_text = rng.choice([
                f"Find ∠{ask_vertex} (the unknown 4th angle of the "
                f"figure-shown quadrilateral). Apply the quadrilateral "
                f"angle-sum rule.",
                f"Compute the missing angle ∠{ask_vertex} using ∠A + ∠B + "
                f"∠C + ∠D = 360°.",
            ])
        elif kind == "cyclic_chain":
            ask_text = rng.choice([
                "Find the sum ∠A + ∠B (two-step: use the cyclic property to "
                "get ∠A from ∠C, then solve for ∠B from the offset).",
                "Compute ∠A + ∠B using first the opposite-angle relation "
                "(∠A + ∠C = 180°) then the B-D offset.",
            ])
        elif kind == "cyclic_triple_unknown":
            ask_text = rng.choice([
                "Compute the sum ∠A + ∠C + ∠D in degrees. Answer must be "
                "an integer.",
                "Solve the 4-equation system (two cyclic rules + two "
                "offset constraints) for the 4 angles and return "
                "∠A + ∠C + ∠D as an integer.",
            ])
        else:
            ask_vertex = vertex_names[queried_index]
            ask_text = rng.choice([
                f"Find \u2220{ask_vertex}.",
                f"What is the measure of \u2220{ask_vertex}?",
                f"Determine \u2220{ask_vertex}.",
                f"Compute \u2220{ask_vertex}.",
            ])

        # Distractors
        tight = cfg["tight_distractors"]
        if tight:
            pool = {max(1, gt - 2), max(1, gt - 1), gt + 1, gt + 2,
                    max(1, gt - 5), gt + 5}
        else:
            pool = {max(1, gt - 15), max(1, gt - 8), gt + 8, gt + 15,
                    180 - gt, 360 - gt}
        pool.discard(gt)
        pool_list = [p for p in pool if 0 < p < 360 and p != gt]
        rng.shuffle(pool_list)
        distractors = pool_list[:3]
        if len(distractors) < 3:
            for k in (-20, -10, 10, 20, 30, 45):
                cand = gt + k
                if 0 < cand < 360 and cand != gt and cand not in distractors:
                    distractors.append(cand)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            return None

        options_vals = [gt] + distractors[:3]
        rng.shuffle(options_vals)
        if options_vals.count(gt) > 1:
            return None
        answer_letter = chr(ord("A") + options_vals.index(gt))
        options_str = [f"{v}\u00b0" for v in options_vals]

        question = (
            f"{intro} {ask_text}\n"
            + "\n".join(f"  ({chr(ord('A') + i)}) {options_str[i]}"
                        for i in range(4))
            + "\nAnswer with the single letter of the correct option."
        )

        image = self._render(kind, angles, given_indices, queried_index,
                             intro, ask_text, options_str, cfg, vis_rng)
        return question, answer_letter, image

    def _render(self, kind, angles, given_indices, queried_index,
                intro, ask_text, options, cfg, vis_rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(9.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        lw = style["line_width"]

        # Build vertices — add a small amount of random perturbation
        jitter = lambda: vis_rng.uniform(-0.15, 0.15)
        if kind == "parallelogram":
            verts = [(0 + jitter(), 0 + jitter()),
                     (5 + jitter(), 0 + jitter()),
                     (6 + jitter(), 3 + jitter()),
                     (1 + jitter(), 3 + jitter())]
        elif kind == "trapezoid":
            verts = [(1 + jitter(), 0 + jitter()),
                     (5 + jitter(), 0 + jitter()),
                     (6 + jitter(), 3 + jitter()),
                     (0 + jitter(), 3 + jitter())]
        elif kind == "kite":
            verts = [(3 + jitter(), 4 + jitter()),
                     (5 + jitter(), 1.5 + jitter()),
                     (3 + jitter(), 0 + jitter()),
                     (1 + jitter(), 1.5 + jitter())]
        elif kind in ("cyclic", "cyclic_chain", "cyclic_triple_unknown"):
            # Place on actual circle for visual realism
            base_angles = [30, 110, 200, 310]
            jitter_angles = [a + vis_rng.uniform(-10, 10)
                             for a in base_angles]
            verts = [(math.cos(math.radians(a)) * 3 + 3,
                      math.sin(math.radians(a)) * 3 + 3)
                     for a in jitter_angles]
            c = mpatches.Circle((3, 3), 3.0, facecolor="none",
                                 edgecolor="#95a5a6", linewidth=1,
                                 linestyle="--")
            ax_f.add_patch(c)
        else:
            verts = [(0 + jitter(), 0 + jitter()),
                     (5 + jitter(), 0 + jitter()),
                     (6 + jitter(), 3 + jitter()),
                     (1 + jitter(), 4 + jitter())]

        # Draw polygon
        poly = mpatches.Polygon(verts, closed=True,
                                 facecolor=palette[0],
                                 edgecolor=palette[2] if len(palette) > 2
                                 else "#333",
                                 linewidth=lw, alpha=0.3)
        ax_f.add_patch(poly)

        cx = sum(v[0] for v in verts) / 4
        cy = sum(v[1] for v in verts) / 4
        vertex_names = ["A", "B", "C", "D"]

        # Vertex and angle labels
        # IMPORTANT: only display the angle values for given_indices.
        # For the queried_index, show "?" (or nothing) — the answer must NOT
        # be visible on the image.
        # Vertex letters sit OUTSIDE the polygon; angle values sit just
        # INSIDE the polygon so they never overlap the letter.
        for i, (vx, vy) in enumerate(verts):
            label = vertex_names[i]
            ox, oy = vx - cx, vy - cy
            n = math.hypot(ox, oy) + 1e-6
            # Vertex letter well outside the polygon
            ax_f.text(vx + 0.70 * ox / n, vy + 0.70 * oy / n, label,
                      fontsize=fs + 3, fontweight="bold",
                      family=ff, ha="center", va="center",
                      color="#1a1a1a",
                      bbox=dict(boxstyle="circle,pad=0.2",
                                facecolor="#ffffff",
                                edgecolor="#555555", linewidth=0.8,
                                alpha=0.9))
            # Angle value or "?"
            if i in given_indices:
                val_text = f"{angles[i]}\u00b0"
                val_color = "#1b5e20"
                box_fc = "#e8f5e9"
                box_ec = "#2e7d32"
            elif i == queried_index:
                val_text = "?"
                val_color = "#b71c1c"
                box_fc = "#ffebee"
                box_ec = "#c62828"
            else:
                val_text = None
                val_color = None
            if val_text is not None:
                ax_f.text(vx - 0.55 * ox / n, vy - 0.55 * oy / n,
                          val_text,
                          fontsize=fs + (2 if i == queried_index else 0),
                          family=ff,
                          fontweight="bold",
                          ha="center", va="center", color=val_color,
                          bbox=dict(boxstyle="round,pad=0.22",
                                    facecolor=box_fc,
                                    edgecolor=box_ec, linewidth=1.0,
                                    alpha=0.95))

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        pad = 2
        ax_f.set_xlim(min(xs) - pad, max(xs) + pad)
        ax_f.set_ylim(min(ys) - pad, max(ys) + pad)
        quad_title_pool = list(_FIGURE_TITLES) + [f"{kind.title()}"]
        ax_f.set_title(vis_rng.choice(quad_title_pool),
                       fontsize=fs + 1, family=ff, pad=6)

        # Right-side text column
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Given:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        for ln in self._wrap(intro, 42):
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
        y -= 0.3
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.15)
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

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b2b"
    os.makedirs(out_dir, exist_ok=True)
    env = QuadrilateralAngleSumQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed} FAILED")
                continue
            path = os.path.join(
                out_dir, f"quadrilateral_angle_sum_s{seed}_L{level}.png")
            env.render().save(path)
            print(f"L{level} s{seed} A={env._answer}")
