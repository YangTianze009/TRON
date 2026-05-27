"""
Polygon Interior Angle Advanced QA environment.

Goal: targeted fix for Angle and metric-geometry-angle. Irregular convex polygon (5-8 sides) with some
diagonals; some interior angles and sub-triangle angles partially
labeled. Requires combining the polygon interior angle sum theorem
( (n-2)*180 ) with triangle-sum in sub-triangles.

Difficulty schedule:
  Axis 1: n_sides = 5 + level // 3                            -> 5..8
  Axis 2: n_diagonals = level // 2                            -> 0..4
  Axis 3: n_given_angles = max(2, n_sides - 1 - level // 2)

Output: single integer (degrees).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class PolygonInteriorAngleAdvancedQA(StandaloneVisualEnv):
    ENV_NAME = "polygon_interior_angle_advanced"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _TITLE_VARIANTS = [
        "Polygon interior angle",
        "Find angle x (polygon)",
        "Polygon + diagonals",
        "Polygon angle chain",
        "Polygon geometry",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_sides = 5 + level // 3
        n_sides = min(n_sides, 8)
        n_diag = level // 2
        n_given = max(2, n_sides - 1 - level // 2)
        return {
            "n_sides":           n_sides,
            "n_diagonals":       n_diag,
            "n_given_angles":    n_given,
            "tight_distractors": level >= 4,
            "use_sub_triangle":  level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 379)
        self._primary_complexity_feature = cfg["n_sides"]

        for _ in range(20):
            try:
                if cfg["use_sub_triangle"] and cfg["n_diagonals"] >= 1:
                    r = self._sub_triangle_chain(sub_rng, cfg)
                else:
                    r = self._basic_polygon_sum(sub_rng, cfg)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    def _random_convex_angles(self, rng, n_sides, min_a=60, max_a=150):
        """Generate n angles summing to (n-2)*180 that look convex.

        At small n the default [60, 150] window works; at n=7,8 the average
        approaches the upper bound so we widen the window around the mean to
        keep sampling feasible (mean ± 30° clamped to [40, 165]).
        """
        target = (n_sides - 2) * 180
        mean = target / n_sides
        # Auto-widen for large n while staying convex (<180).
        lo = max(40, int(mean - 30))
        hi = min(165, int(mean + 30))
        if lo >= hi:
            lo, hi = max(40, int(mean) - 5), min(170, int(mean) + 5)
        for _ in range(400):
            angles = [rng.randint(lo, hi) for _ in range(n_sides - 1)]
            last = target - sum(angles)
            if lo <= last <= hi and 0 < last < 180:
                angles.append(last)
                regular = target / n_sides
                if any(abs(a - regular) >= 5 for a in angles):
                    return angles
        return None

    def _basic_polygon_sum(self, rng, cfg):
        """Give n-1 angles, ask for the last one."""
        n = cfg["n_sides"]
        angles = self._random_convex_angles(rng, n)
        if angles is None:
            return None
        unknown_idx = rng.randint(0, n - 1)
        unknown_val = angles[unknown_idx]
        given = [(i, a) for i, a in enumerate(angles) if i != unknown_idx]
        # drop some to reach n_given
        n_keep = min(len(given), cfg["n_given_angles"])
        # We cannot drop too many — answer wouldn't be derivable.
        # For the basic version keep n-1 givens (so still solvable).
        n_keep = n - 1
        given = given[:n_keep]
        return self._finalize(rng, cfg, answer=unknown_val, n_sides=n,
                              angles=angles, given_indices=[g[0] for g in given],
                              unknown_idx=unknown_idx, variant="polygon_sum",
                              subangles=None)

    def _sub_triangle_chain(self, rng, cfg):
        """Draw diagonal(s) to create sub-triangle(s). Label some interior
        angles AND some sub-triangle angles. Ask for one polygon interior
        angle.

        Strategy: pick the unknown at vertex k. Create a sub-triangle
        involving vertex k by drawing a diagonal from k to another
        non-adjacent vertex. Label enough information to determine the
        unknown via triangle-sum + polygon-sum combined."""
        n = cfg["n_sides"]
        angles = self._random_convex_angles(rng, n)
        if angles is None:
            return None
        unknown_idx = rng.randint(0, n - 1)
        unknown_val = angles[unknown_idx]

        # Build a sub-triangle at the unknown vertex k.
        # Diagonal from unknown_idx to (unknown_idx+2) % n
        # This creates a triangle with vertices k, k+1, k+2.
        k = unknown_idx
        a_k1 = rng.randint(25, 75)  # portion of angle at k in sub-triangle
        a_k2_full = angles[(k + 1) % n]
        # triangle sub-angle at k+1 is the full angle (the diagonal from k
        # to k+2 cuts only angle at k and k+2, not at k+1).
        # Actually the diagonal from k to k+2 passes through side (k+1, k+2),
        # no — a diagonal from k to k+2 is just a chord inside the polygon
        # that cuts off triangle k,k+1,k+2. In that triangle the angle at
        # vertex k+1 equals the FULL interior angle at k+1 (since k+1 is
        # incident to both edges of that triangle).
        # The angle at k in that triangle is part of the polygon's
        # interior angle at k (a portion split by the diagonal). Similarly
        # angle at k+2 is a portion of polygon interior angle at k+2.
        # So: a_k1 + a_k2_full + a_k3 = 180, giving a_k3 = 180 - a_k1 - a_k2.
        if a_k2_full >= 170:
            return None
        a_k3 = 180 - a_k1 - a_k2_full
        if a_k3 <= 10 or a_k3 >= 150:
            return None
        # The labelled information: we show a_k1 (sub-angle at k in the
        # sub-triangle), and the full interior angle at k+1 (which is
        # a_k2_full). And the other polygon interior angles except
        # at position k. Then:
        # interior_at_k (unknown) can be computed from polygon_sum:
        # sum = (n-2)*180, subtract all other known interiors.
        return self._finalize(rng, cfg, answer=unknown_val, n_sides=n,
                              angles=angles, given_indices=None,
                              unknown_idx=unknown_idx,
                              variant="sub_triangle",
                              subangles={"k": k,
                                         "a_k1": a_k1,
                                         "a_k2_full": a_k2_full,
                                         "a_k3": a_k3})

    # ------------------------------------------------------------------ #
    def _finalize(self, rng, cfg, answer, n_sides, angles, given_indices,
                  unknown_idx, variant, subangles):
        ans = int(round(answer))
        if ans <= 0 or ans >= 180:
            return None

        # Build question — numeric values are labeled on the image; do NOT
        # restate them in the question text.
        letters = [chr(ord("A") + i) for i in range(n_sides)]
        unknown_letter = letters[unknown_idx]
        sum_int = (n_sides - 2) * 180
        if variant == "polygon_sum":
            q = (f"In the convex {n_sides}-gon shown, the interior angles "
                 f"sum to {sum_int}°. Using the angles labeled in the "
                 f"figure, find the missing interior angle "
                 f"∠{unknown_letter} in degrees.")
        else:
            q = (f"In the convex {n_sides}-gon shown, diagonals have been "
                 f"drawn to split it into sub-triangles; interior angles "
                 f"sum to {sum_int}°. Using the angles labeled in the "
                 f"figure (including any sub-angle formed by a diagonal), "
                 f"find the full interior angle ∠{unknown_letter} in "
                 f"degrees.")

        q += " Answer with a single integer."
        img = self._render(n_sides, angles, unknown_idx, variant, subangles,
                           given_indices, cfg)
        return q, str(ans), img

    # ------------------------------------------------------------------ #
    def _render(self, n_sides, angles, unknown_idx, variant, subangles,
                given_indices, cfg):
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

        # Place vertices on a jittered circle
        R = 3.0
        rot_off = self._rng.uniform(0, 2 * math.pi)
        pts = []
        for i in range(n_sides):
            angle = rot_off + 2 * math.pi * i / n_sides
            r = R + self._rng.uniform(-0.25, 0.25)
            pts.append((r * math.cos(angle), r * math.sin(angle)))
        # close polygon
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, color=line_color, linewidth=lw)

        # Draw diagonals per config
        n_diag = cfg["n_diagonals"]
        diag_color = palette[5 % len(palette)]
        diag_drawn = 0
        if variant == "sub_triangle" and subangles is not None:
            k = subangles["k"]
            k2 = (k + 2) % n_sides
            ax.plot([pts[k][0], pts[k2][0]],
                    [pts[k][1], pts[k2][1]],
                    color=diag_color, linewidth=lw * 0.9, linestyle=":")
            diag_drawn += 1
        # Fill in additional decoration diagonals
        extra_needed = max(0, n_diag - diag_drawn)
        pool = [(i, (i + 2) % n_sides) for i in range(n_sides)]
        random.Random(self._rng.random()).shuffle(pool)
        for (i, j) in pool:
            if extra_needed <= 0:
                break
            if variant == "sub_triangle" and {i, j} == {subangles["k"], (subangles["k"] + 2) % n_sides}:
                continue
            ax.plot([pts[i][0], pts[j][0]],
                    [pts[i][1], pts[j][1]],
                    color="#888888", linewidth=lw * 0.6, linestyle=":")
            extra_needed -= 1

        # Label vertices
        letters = [chr(ord("A") + i) for i in range(n_sides)]
        cx_poly = sum(p[0] for p in pts) / n_sides
        cy_poly = sum(p[1] for p in pts) / n_sides
        for i, p in enumerate(pts):
            ax.plot(p[0], p[1], "o", color=palette[0], markersize=5)
            # push label outward
            dx, dy = p[0] - cx_poly, p[1] - cy_poly
            dn = math.hypot(dx, dy) + 1e-9
            ox = p[0] + 0.35 * dx / dn
            oy = p[1] + 0.35 * dy / dn
            ax.text(ox, oy, letters[i], fontsize=fs + 1,
                    fontweight="bold", family=ff, color=line_color)

        # Label given angles inside the polygon, near each vertex
        label_color = "#2e7d32"
        unknown_color = "#c0392b"

        def inward_pt(V, d=0.55):
            dx, dy = cx_poly - V[0], cy_poly - V[1]
            dn = math.hypot(dx, dy) + 1e-9
            return (V[0] + d * dx / dn, V[1] + d * dy / dn)

        if variant == "polygon_sum":
            # Show ALL n-1 given angles on the image (needed for solution).
            shown_idx = list(given_indices)
            for i in shown_idx:
                tx, ty = inward_pt(pts[i])
                ax.text(tx, ty, f"{angles[i]}°",
                        fontsize=fs - 1, color=label_color,
                        fontweight="bold", ha="center")
            # unknown
            tx, ty = inward_pt(pts[unknown_idx])
            ax.text(tx, ty, "x°",
                    fontsize=fs + 1, color=unknown_color,
                    fontweight="bold", ha="center")
        else:
            # Show angles at all vertices except unknown_idx + sub-angle at k
            for i in range(n_sides):
                if i == unknown_idx:
                    continue
                tx, ty = inward_pt(pts[i])
                ax.text(tx, ty, f"{angles[i]}°",
                        fontsize=fs - 1, color=label_color,
                        fontweight="bold", ha="center")
            # unknown
            tx, ty = inward_pt(pts[unknown_idx])
            ax.text(tx, ty, "x°",
                    fontsize=fs + 1, color=unknown_color,
                    fontweight="bold", ha="center")
            # sub-angle at k
            k = subangles["k"]
            tx, ty = inward_pt(pts[k], d=1.05)
            ax.text(tx, ty, f"sub={subangles['a_k1']}°",
                    fontsize=fs - 1, color="#1565c0",
                    fontweight="bold", ha="center")

        pad = 1.2
        all_x = [p[0] for p in pts]
        all_y = [p[1] for p in pts]
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        ax.set_title(self._rng.choice(self._TITLE_VARIANTS),
                     fontsize=fs + 2, fontweight="bold", pad=8, family=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
