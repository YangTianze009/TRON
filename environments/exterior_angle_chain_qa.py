"""
Exterior Angle Chain QA environment.

Goal: targeted fix for Angle. Two or more
triangles sharing sides/vertices, with one or more exterior angles marked.
Uses the exterior-angle theorem (ext = sum of remote interior angles)
repeatedly, plus supplementary trap at high levels.

Difficulty schedule:
  Axis 1: n_triangles = 1 + level // 3               -> 1..4
  Axis 2: n_given_angles = max(2, 5 - level // 2)    -> 5..2
  Axis 3: use_supplementary_trap = level >= 5

Output: MCQ (A/B/C/D).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class ExteriorAngleChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "exterior_angle_chain"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_VARIANTS = [
        "Exterior angle chain",
        "Find angle x",
        "Exterior angles",
        "Chain of triangles",
        "Angle puzzle (exterior)",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_triangles":       1 + level // 3,             # 1..4
            # ALL necessary givens are always shown on the image;
            # difficulty comes from number of triangles & traps, NOT
            # from hiding values.
            "n_given_angles":    99,
            "supp_trap":         level >= 5,
            "tight_distractors": level >= 4,
            "use_fractional":    level >= 7,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 347)
        self._primary_complexity_feature = cfg["n_triangles"]

        # Problem family depends on n_triangles
        n_tri = cfg["n_triangles"]
        for _ in range(25):
            try:
                if n_tri == 1:
                    r = self._one_triangle(sub_rng, cfg)
                elif n_tri == 2:
                    r = self._two_triangles(sub_rng, cfg)
                elif n_tri == 3:
                    r = self._three_triangles(sub_rng, cfg)
                else:
                    r = self._four_triangles(sub_rng, cfg)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # Problem variants
    # ------------------------------------------------------------------ #
    def _one_triangle(self, rng, cfg):
        """Single triangle + exterior angle. ext = sum of remote interior."""
        a = rng.randint(30, 80)
        b = rng.randint(30, 80)
        if a + b >= 170 or a + b < 30:
            return None
        ext = a + b
        # Optionally wrap with a supp trap: give (180 - a) instead of a
        given_a = a
        given_b = b
        trap = False
        if cfg["supp_trap"] and rng.random() < 0.5:
            given_a = 180 - a  # supplementary to the remote interior
            trap = True
        return self._finalize(rng, cfg, answer_val=ext, n_tri=1,
                              givens=[given_a, given_b],
                              interiors=[a, b, 180 - a - b],
                              trap=trap, which_is_trap=0 if trap else None,
                              variant="one_tri")

    def _two_triangles(self, rng, cfg):
        """Two triangles sharing an edge. Exterior angle at shared vertex
        chain. a,b -> shared = 180-a-b; second tri has shared + d + x = 180
        OR ext at shared = a + b -> second tri uses ext. Use simpler
        version: given two interior of tri1 + one interior of tri2, find
        the exterior at shared vertex of tri2."""
        a = rng.randint(30, 70)
        b = rng.randint(30, 70)
        if a + b >= 160:
            return None
        shared = 180 - a - b
        d = rng.randint(30, 80)
        if shared + d >= 170:
            return None
        # x = exterior at third vertex of tri2 = shared + d
        ext2 = shared + d
        if ext2 >= 175 or ext2 <= 15:
            return None
        trap = False
        given_d = d
        if cfg["supp_trap"] and rng.random() < 0.5:
            given_d = 180 - d
            trap = True
        return self._finalize(rng, cfg, answer_val=ext2, n_tri=2,
                              givens=[a, b, given_d],
                              interiors=[a, b, shared, d,
                                         180 - shared - d],
                              trap=trap, which_is_trap=2 if trap else None,
                              variant="two_tri")

    def _three_triangles(self, rng, cfg):
        """Three triangles connected in a strip. Given 3 angles, chain
        through to find exterior angle on the far side."""
        a = rng.randint(35, 65)
        b = rng.randint(35, 65)
        c = rng.randint(35, 65)
        if a + b >= 155 or a + c >= 155:
            return None
        # tri1: a, b, 180-a-b
        # tri2 shares the (180-a-b) angle; c is another angle of tri2
        shared1 = 180 - a - b
        if shared1 + c >= 170:
            return None
        shared2 = 180 - shared1 - c
        # tri3 shares shared2; ask for exterior of tri3 at the far vertex
        # if tri3 is just a triangle with one known angle d=shared2, and
        # another angle e; ext = shared2 + e
        e = rng.randint(30, 70)
        if shared2 + e >= 170:
            return None
        ext3 = shared2 + e
        if ext3 >= 175 or ext3 <= 15:
            return None
        given_e = e
        trap = False
        if cfg["supp_trap"] and rng.random() < 0.5:
            given_e = 180 - e
            trap = True
        return self._finalize(rng, cfg, answer_val=ext3, n_tri=3,
                              givens=[a, b, c, given_e],
                              interiors=[a, b, shared1, c, shared2, e,
                                         180 - shared2 - e],
                              trap=trap, which_is_trap=3 if trap else None,
                              variant="three_tri")

    def _four_triangles(self, rng, cfg):
        """Four triangles connected in a chain. Longer hop but compact.

        Chain of 4 triangles sharing bottom vertices B,C,D.
          tri0 verts: A, B, P (apex);  interior angles a=@A, b=@P, s1=@B
          tri1 verts: B, C, Q (apex);  interior angles s1=@B, c=@Q, s2=@C
          tri2 verts: C, D, R (apex);  interior angles s2=@C, d=@R, s3=@D
          tri3 verts: D, E, S (apex);  interior angles s3=@D, e=@S, s4=@E
        x = exterior at E along baseline extension = 180 - s4 = s3 + e.
        """
        a = rng.randint(35, 60)
        b = rng.randint(35, 60)
        c = rng.randint(35, 60)
        d = rng.randint(35, 60)
        e = rng.randint(35, 60)
        if a + b >= 150 or a + c >= 150 or a + d >= 150:
            return None
        s1 = 180 - a - b
        if s1 + c >= 170:
            return None
        s2 = 180 - s1 - c
        if s2 + d >= 170:
            return None
        s3 = 180 - s2 - d
        if s3 + e >= 170:
            return None
        ext4 = s3 + e
        if ext4 >= 175 or ext4 <= 15:
            return None
        given_d = d
        given_e = e
        trap = False
        trap_idx = None
        if cfg["supp_trap"] and rng.random() < 0.5:
            # randomly trap d or e
            if rng.random() < 0.5:
                given_d = 180 - d
                trap_idx = 3
            else:
                given_e = 180 - e
                trap_idx = 4
            trap = True
        return self._finalize(rng, cfg, answer_val=ext4, n_tri=4,
                              givens=[a, b, c, given_d, given_e],
                              interiors=[a, b, s1, c, s2, d, s3, e,
                                         180 - s3 - e],
                              trap=trap, which_is_trap=trap_idx,
                              variant="four_tri")

    # ------------------------------------------------------------------ #
    # Finalization: MCQ + render
    # ------------------------------------------------------------------ #
    def _finalize(self, rng, cfg, answer_val, n_tri, givens, interiors,
                  trap, which_is_trap, variant):
        gt = int(round(answer_val))
        if gt <= 0 or gt >= 180:
            return None
        # Build distractors
        if cfg["tight_distractors"]:
            pool = [gt - 5, gt - 3, gt + 3, gt + 7, 180 - gt, gt // 2 + 10]
        else:
            pool = [gt - 25, gt - 12, gt + 12, gt + 25, 180 - gt, gt // 2,
                    gt * 2 % 180]
        pool = [p for p in pool if 10 < p < 175 and p != gt]
        rng.shuffle(pool)
        distractors = []
        for p in pool:
            if p not in distractors:
                distractors.append(p)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            return None
        options = [gt] + distractors[:3]
        rng.shuffle(options)
        idx = options.index(gt)
        answer_letter = chr(ord("A") + idx)
        opt_text = " ".join(
            f"({chr(ord('A') + i)}) {v}°" for i, v in enumerate(options)
        )

        # Question text — DO NOT embed numeric givens; read the figure.
        supp_note = ""
        if trap and which_is_trap is not None:
            supp_note = (" (Note: at least one marked angle is supplementary "
                         "to the interior angle shown; read carefully.)")

        if variant == "one_tri":
            _POOL = [
                "In the diagram, a triangle has interior angles labeled. Using the angles labeled in the figure, find the exterior angle x at the vertex opposite the two given interior angles.",
                "A single triangle is shown with two interior angles marked. What is the exterior angle x at the remaining vertex?",
                "The figure shows a triangle and one of its exterior angles x. Using the labeled interior angles, determine x.",
                "Based on the triangle diagram, apply the exterior angle theorem to compute x.",
                "Given the triangle with its interior angles marked, find the exterior angle x shown in the figure.",
                "From the diagram, use the exterior-angle theorem (ext = sum of remote interior angles) to find x.",
                "A triangle with labeled angles is depicted. Compute the exterior angle x at the indicated vertex.",
                "Find x in the figure: it is the exterior angle of the triangle at the vertex opposite the two given angles.",
                "Using the labeled triangle, determine the value of the exterior angle x.",
                "The diagram shows a triangle with interior angles labeled and an exterior angle x to be found. Compute x.",
                "Apply the exterior angle theorem to the triangle in the figure. What is x?",
                "In the figure, a triangle has two marked interior angles; find the exterior angle x at the remaining vertex.",
                "Given the triangle's interior angles shown, find x, the exterior angle marked in the diagram.",
                "Compute x using the triangle's labeled angles and the exterior angle theorem.",
                "Identify the exterior angle x in the triangle diagram using the marked interior angles.",
                "The triangle shown has known interior angles and a marked exterior angle x. Find x.",
            ]
        elif variant == "two_tri":
            _POOL = [
                "Two triangles share an edge, as shown. Using the angles labeled in the figure, apply the exterior angle theorem to find x at the far vertex.",
                "The figure shows two triangles joined along an edge. Using the marked angles, determine x.",
                "Two triangles meet at a shared side in the diagram. Find the angle x labeled at the far vertex.",
                "Given the two-triangle configuration, apply the exterior angle theorem to compute x.",
                "In the figure, two triangles share an edge; use the labeled angles to find x.",
                "From the two-triangle diagram, chain the exterior angle theorem to determine x at the indicated vertex.",
                "Two adjacent triangles are shown with angle labels. What is the value of x?",
                "Using the two-triangle figure, find the angle x using the exterior angle theorem.",
                "The diagram shows two triangles sharing a common side. Compute the marked angle x.",
                "Apply the exterior angle theorem twice (once per triangle) to the figure to find x.",
                "Find x in the two-triangle diagram using the labeled interior angles.",
                "Given two triangles joined at an edge, determine x from the angle labels in the figure.",
                "In the two-triangle figure, use the exterior angle property to compute x.",
                "Two triangles connected by an edge are shown. Find angle x based on the labels.",
                "From the figure of two joined triangles, compute x using the exterior angle chain.",
                "Using the marked angles in the two-triangle diagram, determine the value of x.",
            ]
        elif variant == "three_tri":
            _POOL = [
                "Three triangles are connected by shared edges, as shown. Using the angles labeled in the figure, chain the exterior angle theorem to find x at the far vertex of the third triangle.",
                "The figure shows three triangles linked by shared edges. Compute the angle x using the exterior angle chain.",
                "Three triangles share edges in a chain. Using the marked angles, determine x.",
                "Given the three-triangle diagram, apply the exterior angle theorem across the chain to find x.",
                "Three triangles are joined along successive edges. Find x at the far vertex using the labeled angles.",
                "From the three-triangle figure, chain the exterior angle property to compute x.",
                "Using the three-triangle configuration, determine x from the marked angles.",
                "The diagram shows a chain of three triangles sharing edges. Find x.",
                "Apply the exterior angle theorem repeatedly to the three-triangle chain to find x.",
                "In the figure of three connected triangles, compute the value of x.",
                "Three triangles in a chain are shown with angle labels. What is x at the end?",
                "From the three-triangle diagram, use the exterior angle theorem to find the marked angle x.",
                "Determine x in the three-triangle chain using the labeled interior angles.",
                "Given three triangles sharing edges, compute x via exterior angle chaining.",
                "Using the labels on the three-triangle figure, find the exterior angle x at the far vertex.",
                "Chain the exterior angle theorem through the three triangles in the figure to obtain x.",
            ]
        else:
            _POOL = [
                "Four triangles form a chain, as shown. Using the angles labeled in the figure, chain the exterior angle theorem to find x at the final vertex.",
                "The figure shows four triangles linked by shared edges. Compute x at the last vertex.",
                "Four triangles are connected in a chain. Using the marked angles, determine x.",
                "Given the four-triangle diagram, apply the exterior angle theorem across the chain to find x.",
                "Four triangles share edges in succession. Find x at the final vertex based on the labels.",
                "From the four-triangle figure, chain the exterior angle property to compute x.",
                "Using the four-triangle configuration, determine x from the marked interior angles.",
                "The diagram shows a chain of four triangles sharing edges. Find x.",
                "Apply the exterior angle theorem repeatedly to the four-triangle chain to find x.",
                "In the figure of four connected triangles, compute the value of x at the end.",
                "Four triangles in a chain are shown with angle labels. What is x at the final vertex?",
                "From the four-triangle diagram, use the exterior angle theorem to find the marked angle x.",
                "Determine x in the four-triangle chain using the labeled interior angles.",
                "Given four triangles sharing edges, compute x via exterior angle chaining.",
                "Using the labels on the four-triangle figure, find the exterior angle x at the far vertex.",
                "Chain the exterior angle theorem through the four triangles in the figure to obtain x.",
            ]
        sidx = (self.seed or 0) % 16
        stem = _POOL[sidx]
        question = (f"{stem}{supp_note} Options: {opt_text}. "
                    f"Answer with a single letter.")

        img = self._render(n_tri, givens, interiors, cfg, trap, variant)
        return question, answer_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, n_tri, givens, interiors, cfg, trap, variant):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7.0 * sc, 5.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        palette = style["palette"]
        line_color = style["geo_line_color"]
        lw = style["line_width"] + 0.3
        fs = style["font_size_base"]
        ff = style["font_family"]

        # Place n_tri triangles in a strip
        base_y = 0.0
        edge = 2.6
        apex_h = 2.2
        rot = self._rng.uniform(-0.15, 0.15)

        # build chain of vertices along the bottom
        bottom_pts = []
        for i in range(n_tri + 1):
            x = i * edge * 0.9 + self._rng.uniform(-0.2, 0.2)
            bottom_pts.append((x, base_y))
        apex_pts = []
        for i in range(n_tri):
            ax_x = (bottom_pts[i][0] + bottom_pts[i + 1][0]) / 2 + \
                   self._rng.uniform(-0.25, 0.25)
            ax_y = apex_h + self._rng.uniform(-0.3, 0.3)
            apex_pts.append((ax_x, ax_y))

        # apply rotation
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        def rt(p):
            return (cos_r * p[0] - sin_r * p[1],
                    sin_r * p[0] + cos_r * p[1])
        bottom_pts = [rt(p) for p in bottom_pts]
        apex_pts = [rt(p) for p in apex_pts]

        # Draw each triangle
        for i in range(n_tri):
            p1 = bottom_pts[i]
            p2 = bottom_pts[i + 1]
            p3 = apex_pts[i]
            tri = [p1, p2, p3, p1]
            ax.plot([t[0] for t in tri], [t[1] for t in tri],
                    color=line_color, linewidth=lw)

        # Extend the last bottom edge past to show an exterior angle
        last_left = bottom_pts[-2]
        last_right = bottom_pts[-1]
        dx = last_right[0] - last_left[0]
        dy = last_right[1] - last_left[1]
        n = math.hypot(dx, dy) + 1e-9
        ext_tip = (last_right[0] + 1.5 * dx / n,
                   last_right[1] + 1.5 * dy / n)
        ax.plot([last_right[0], ext_tip[0]],
                [last_right[1], ext_tip[1]],
                color=line_color, linewidth=lw, linestyle="--")

        # Label vertices
        for i, p in enumerate(bottom_pts):
            ax.plot(p[0], p[1], "o", color=palette[0], markersize=4)
            ax.text(p[0] - 0.15, p[1] - 0.4, chr(ord("A") + i),
                    fontsize=fs, fontweight="bold", family=ff,
                    color=line_color)
        for i, p in enumerate(apex_pts):
            ax.plot(p[0], p[1], "o", color=palette[0], markersize=4)
            ax.text(p[0] - 0.1, p[1] + 0.15, chr(ord("P") + i),
                    fontsize=fs, fontweight="bold", family=ff,
                    color=line_color)

        # Label a few given angles.
        # Show first n_given_angles of the givens list near the correct
        # vertex/angle position.
        n_show = cfg["n_given_angles"]
        # Strategy: put givens[0] at bottom-left vertex of tri0, givens[1] at
        # apex of tri0; additional givens at apex of subsequent tris.
        def label_angle_at_vertex(V, oa, ob, text, color):
            """Place text near V inside the angle formed by oa-V-ob."""
            va = (oa[0] - V[0], oa[1] - V[1])
            vb = (ob[0] - V[0], ob[1] - V[1])
            na = math.hypot(*va) + 1e-9
            nb = math.hypot(*vb) + 1e-9
            bx = V[0] + 0.55 * (va[0] / na + vb[0] / nb)
            by = V[1] + 0.55 * (va[1] / na + vb[1] / nb)
            ax.text(bx, by, text, fontsize=fs - 1, color=color,
                    fontweight="bold", ha="center", va="center")

        # Use interiors to actually label accurate-looking values on the
        # figure. The "givens" list contains what is shown textually —
        # match positions roughly.
        positions = []  # list of (V, o1, o2)
        # tri0 bottom-left
        positions.append((bottom_pts[0], bottom_pts[1], apex_pts[0]))
        # apex tri0
        positions.append((apex_pts[0], bottom_pts[0], bottom_pts[1]))
        for i in range(1, n_tri):
            positions.append((apex_pts[i], bottom_pts[i], bottom_pts[i + 1]))

        label_color = "#2e7d32"
        for k in range(min(n_show, len(givens), len(positions))):
            V, o1, o2 = positions[k]
            val = givens[k]
            label_angle_at_vertex(V, o1, o2, f"{int(round(val))}°",
                                   label_color)

        # Mark the unknown x at the far vertex (exterior)
        last_vertex = bottom_pts[-1]
        ax.text(last_vertex[0] + 0.3, last_vertex[1] + 0.2, "x°",
                fontsize=fs + 2, color="#c0392b", fontweight="bold")

        pad = 1.2
        all_x = [p[0] for p in bottom_pts + apex_pts + [ext_tip]]
        all_y = [p[1] for p in bottom_pts + apex_pts + [ext_tip]]
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad + 0.3)

        ax.set_title(self._rng.choice(self._TITLE_VARIANTS),
                     fontsize=fs + 2, fontweight="bold", pad=8, family=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
