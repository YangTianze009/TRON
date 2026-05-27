"""Similar Triangles Ratio QA environment.

Given two similar triangles (or nested / shared-vertex / altitude
configurations), find a missing side length. All numeric side lengths are
drawn ON the image — the question text refers to "the figure" only.

Difficulty:
  L0-1:  two_separate (clear side labels, integer ratio 2/3).
  L2-3:  nested_parallel (inner triangle similar to outer) — integer ratio.
  L4-5:  shared_vertex — non-integer ratio possible.
  L6-7:  nested_chain (3 chained similar triangles).
  L8-9:  altitude_to_hypotenuse (find subsegment) — hardest.

Diversity:
  - 5 configurations (primitives).
  - 4 question templates.
  - randomize vertex labels (ABC/DEF, PQR/XYZ, KLM/RST, etc.).
  - randomized orientation (flip, rotate).
  - palette shuffled per seed; random dashed/dashed-dot style.

No text leakage: the question never mentions a numeric side length; all
values are displayed as colored labels on the image.
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

_VERTEX_SETS = [
    ("A", "B", "C", "D", "E", "F"),
    ("P", "Q", "R", "X", "Y", "Z"),
    ("K", "L", "M", "R", "S", "T"),
    ("J", "N", "O", "U", "V", "W"),
    ("G", "H", "I", "M", "N", "O"),
]

_Q_TEMPLATES = [
    "From the figure above, find the length of {target}.",
    "Using the similar-triangle relationship shown, what is the length of {target}?",
    "The figure shows similar triangles with side lengths labeled. "
    "Find {target}.",
    "Determine {target} based on the similar triangles in the image.",
]

class SimilarTrianglesRatioQA(StandaloneVisualEnv):
    ENV_NAME = "similar_triangles_ratio"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesign 2026-04-17: monotonic difficulty.
        # Previous L3 (nested_parallel) dipped to 0.5 — probably because the
        # parallel-line side mapping confused the model. Previous L9
        # (altitude_hypotenuse) was 1.0 because h² = p·q is such a clean
        # memorized formula. Now: two_separate → shared_vertex → nested_parallel
        # → altitude_hypotenuse → nested_chain (hardest: 3+ similar triangles).
        if level <= 1:
            config = "two_separate"
        elif level <= 3:
            config = "shared_vertex"
        elif level <= 5:
            config = "nested_parallel"
        elif level <= 7:
            config = "altitude_hypotenuse"
        else:
            config = "nested_chain"
        return {
            "config":            config,
            "integer_ratio":     level <= 3,
            "tight_distractors": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        config = cfg["config"]
        if cfg["integer_ratio"]:
            k = rng.choice([2, 3, 4])
        else:
            k = rng.choice([1.5, 2, 2.5, 3])

        vertset = rng.choice(_VERTEX_SETS)
        a1, b1, c1, a2, b2, c2 = vertset  # for two-triangle configs

        if config == "two_separate":
            return self._do_two_separate(rng, cfg, vertset, k)
        if config == "nested_parallel":
            return self._do_nested_parallel(rng, cfg, vertset, k)
        if config == "shared_vertex":
            return self._do_shared_vertex(rng, cfg, vertset, k)
        if config == "nested_chain":
            return self._do_nested_chain(rng, cfg, vertset, k)
        return self._do_altitude_hypotenuse(rng, cfg, vertset)

    # ------------------------------------------------------------------ #
    # Configurations
    # ------------------------------------------------------------------ #

    def _do_two_separate(self, rng, cfg, vertset, k):
        a1, b1, c1, a2, b2, c2 = vertset
        ab = rng.randint(3, 9)
        bc = rng.randint(3, 9)
        ca = rng.randint(3, 9)
        de = ab * k
        ef = bc * k
        fd = ca * k

        which = rng.choice(["EF", "DE", "FD", "AB"])
        if which == "EF":
            # Given AB, BC, DE. Find EF.
            shown = {
                "ABC": {"AB": ab, "BC": bc, "CA": None},
                "DEF": {"DE": de, "EF": "?", "FD": None},
            }
            target_side = "EF"
            gt = ef
        elif which == "DE":
            shown = {
                "ABC": {"AB": ab, "BC": bc, "CA": None},
                "DEF": {"DE": "?", "EF": ef, "FD": None},
            }
            target_side = "DE"
            gt = de
        elif which == "FD":
            shown = {
                "ABC": {"AB": ab, "BC": None, "CA": ca},
                "DEF": {"DE": de, "EF": None, "FD": "?"},
            }
            target_side = "FD"
            gt = fd
        else:
            shown = {
                "ABC": {"AB": "?", "BC": bc, "CA": None},
                "DEF": {"DE": de, "EF": ef, "FD": None},
            }
            target_side = "AB"
            gt = ab

        # Relabel with vertset letters.
        v_map = {"A": a1, "B": b1, "C": c1, "D": a2, "E": b2, "F": c2}
        shown_renamed = {}
        for tri_label, sides in shown.items():
            new_label = "".join(v_map[c] for c in tri_label)
            new_sides = {}
            for side, val in sides.items():
                new_sides["".join(v_map[c] for c in side)] = val
            shown_renamed[new_label] = new_sides
        target_side_r = "".join(v_map[c] for c in target_side)

        gt_norm = self._norm_gt(gt)
        options = self._make_distractors(rng, gt_norm, cfg["tight_distractors"], k)
        if options is None:
            return None
        ans_letter = options["letter"]
        image = self._render_two_separate(rng, shown_renamed, target_side_r,
                                          options["strs"])
        q = self._format_q(rng, target_side_r, options["strs"])
        return q, ans_letter, image

    def _do_nested_parallel(self, rng, cfg, vertset, k):
        # Outer triangle with vertex A, side BC at base; inner DE parallel to BC
        # through D on AB, E on AC. Given AD, AB maybe, find BC from DE or vice versa.
        a1, b1, c1, a2, b2, c2 = vertset
        ad = rng.randint(2, 6)
        de = rng.randint(3, 8)
        db = ad * (k - 1) if cfg["integer_ratio"] else ad * (k - 1)
        ab = ad * k
        bc = de * k
        # Options: given AD, DB, DE → find BC; or AD, AB, DE → find BC
        gt = bc if isinstance(bc, int) else round(bc, 2)
        gt_norm = self._norm_gt(gt)
        options = self._make_distractors(rng, gt_norm, cfg["tight_distractors"], k)
        if options is None:
            return None
        ans_letter = options["letter"]
        target_label = f"{b1}{c1}"  # outer base
        image = self._render_nested_parallel(rng, vertset, ad, db, de, bc,
                                             options["strs"],
                                             cfg["integer_ratio"])
        q = self._format_q(rng, target_label, options["strs"])
        return q, ans_letter, image

    def _do_shared_vertex(self, rng, cfg, vertset, k):
        a1, b1, c1, a2, b2, c2 = vertset
        ab = rng.randint(3, 8)
        bc = rng.randint(3, 8)
        ad = ab * k
        ae = bc * k
        # Two triangles sharing vertex A: ABС and ADE. DE ∥ BC. Given AB, AD, BC → find DE.
        gt = round(bc * k, 2)
        gt_norm = self._norm_gt(gt)
        options = self._make_distractors(rng, gt_norm, cfg["tight_distractors"], k)
        if options is None:
            return None
        ans_letter = options["letter"]
        target_label = f"{a2}{b2}"  # DE
        image = self._render_shared_vertex(rng, vertset, ab, ad, bc, gt,
                                           options["strs"])
        q = self._format_q(rng, target_label, options["strs"])
        return q, ans_letter, image

    def _do_nested_chain(self, rng, cfg, vertset, k):
        # Three similar triangles T1, T2, T3 with ratios k (T2:T1) and k2 (T3:T2).
        k2 = rng.choice([1.5, 2, 2.5]) if not cfg["integer_ratio"] else \
             rng.choice([2, 3])
        base_t1 = rng.randint(2, 5)
        base_t2 = round(base_t1 * k, 2)
        base_t3 = round(base_t2 * k2, 2)
        # Ask for base_t3 given base_t1 and the ratios on image.
        gt_norm = self._norm_gt(base_t3)
        options = self._make_distractors(rng, gt_norm, cfg["tight_distractors"], k)
        if options is None:
            return None
        ans_letter = options["letter"]
        target_label = "the labeled base of T3"
        image = self._render_nested_chain(rng, base_t1, base_t2, base_t3,
                                          k, k2, options["strs"])
        q = self._format_q(rng, "the labeled base of T3 (marked '?')",
                           options["strs"])
        return q, ans_letter, image

    def _do_altitude_hypotenuse(self, rng, cfg, vertset):
        a1, b1, c1, *_ = vertset
        # Right triangle with legs a, b. Altitude to hypotenuse from right angle.
        a = rng.randint(3, 9)
        b = rng.randint(3, 9)
        c_hyp = math.sqrt(a * a + b * b)
        # Subsegment adjacent to leg a is a^2 / c.
        gt = round(a * a / c_hyp, 2)
        gt_norm = self._norm_gt(gt)
        options = self._make_distractors(rng, gt_norm, cfg["tight_distractors"], 2)
        if options is None:
            return None
        ans_letter = options["letter"]
        target_label = "segment AH (marked '?')"
        image = self._render_altitude_hypotenuse(rng, a, b, round(c_hyp, 2),
                                                 gt, options["strs"])
        q = self._format_q(rng, target_label, options["strs"])
        return q, ans_letter, image

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _norm_gt(gt):
        if isinstance(gt, float) and abs(gt - round(gt)) < 1e-6:
            return int(round(gt))
        if isinstance(gt, float):
            return round(gt, 2)
        return gt

    def _make_distractors(self, rng, gt, tight, k_unused):
        try:
            if tight:
                deltas = [-0.5, 0.5, -1, 1, 1.5, -1.5, 2, -2]
            else:
                deltas = [-3, 3, -5, 5, 7, -7, 2, -2]
            rng.shuffle(deltas)
            distractors = []
            for d in deltas:
                cand = round(gt + d, 2) if isinstance(gt, float) else gt + int(d)
                if isinstance(cand, float) and abs(cand - round(cand)) < 1e-6:
                    cand = int(round(cand))
                if cand == gt or cand <= 0:
                    continue
                if cand in distractors:
                    continue
                distractors.append(cand)
                if len(distractors) == 3:
                    break
            if len(distractors) < 3:
                return None
            opts = [gt] + distractors
            rng.shuffle(opts)
            if opts.count(gt) > 1:
                return None
            def fmt(v):
                if isinstance(v, int):
                    return str(v)
                if abs(v - round(v)) < 1e-6:
                    return str(int(round(v)))
                return f"{v:.2f}"
            strs = [fmt(v) for v in opts]
            letter = chr(ord("A") + opts.index(gt))
            return {"vals": opts, "strs": strs, "letter": letter}
        except Exception:
            return None

    def _format_q(self, rng, target, options_strs):
        prefix = rng.choice(_Q_TEMPLATES).format(target=target)
        opt_str = "\n".join(f"  ({chr(ord('A')+i)}) {o}"
                            for i, o in enumerate(options_strs))
        return (f"{prefix}\n{opt_str}\n"
                f"Answer with a single letter.")

    # ------------------------------------------------------------------ #
    # Renderers — image shows all given values.
    # ------------------------------------------------------------------ #

    def _setup_fig(self):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(8.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax, style

    def _draw_tri(self, ax, verts, face, edge, lw, labels=None,
                  side_labels=None, label_color="#111"):
        """Draw triangle with optional vertex labels and side labels (dict
        mapping 'side_key' to value string, side_key = tuple of 2 indices)."""
        ax.add_patch(mpatches.Polygon(verts, closed=True,
                                      facecolor=face, edgecolor=edge,
                                      linewidth=lw, alpha=0.35))
        if labels:
            for (vx, vy), lbl in zip(verts, labels):
                # outward offset
                cx = sum(p[0] for p in verts) / 3
                cy = sum(p[1] for p in verts) / 3
                dx, dy = vx - cx, vy - cy
                norm = math.hypot(dx, dy) + 1e-6
                ox = vx + 0.28 * dx / norm
                oy = vy + 0.28 * dy / norm
                ax.text(ox, oy, lbl, fontsize=13,
                        color=label_color, fontweight="bold",
                        ha="center", va="center")
        if side_labels:
            for (i, j), val in side_labels.items():
                p1 = verts[i]
                p2 = verts[j]
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                # outward perpendicular offset
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # rotate 90deg
                nx = -dy
                ny = dx
                norm = math.hypot(nx, ny) + 1e-6
                # Push outward: choose direction away from centroid
                cx = sum(p[0] for p in verts) / 3
                cy = sum(p[1] for p in verts) / 3
                sign = 1 if ((mx + 0.4 * nx / norm - cx) ** 2 +
                             (my + 0.4 * ny / norm - cy) ** 2) > \
                            ((mx - 0.4 * nx / norm - cx) ** 2 +
                             (my - 0.4 * ny / norm - cy) ** 2) else -1
                ox = mx + sign * 0.42 * nx / norm
                oy = my + sign * 0.42 * ny / norm
                color = "#b00020" if val == "?" else "#0d3b66"
                ax.text(ox, oy, str(val), fontsize=12,
                        color=color, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white",
                                  edgecolor=color, linewidth=1.0,
                                  alpha=0.9))

    def _render_two_separate(self, rng, shown_renamed, target_side, opts_strs):
        fig, ax, style = self._setup_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        lw = max(1.5, style["line_width"])
        # two triangles side by side.
        tri_defs = list(shown_renamed.items())
        tri1_name, tri1_sides = tri_defs[0]
        tri2_name, tri2_sides = tri_defs[1]
        offset_x = rng.uniform(4.5, 5.5)
        v1_local = [(0, 0), (3, 0), (0.8, 2.5)]
        v2_local = [(offset_x, 0), (offset_x + 3.6, 0),
                    (offset_x + 0.9, 2.95)]
        self._draw_tri(ax, v1_local, palette[0], "#111", lw,
                       labels=list(tri1_name),
                       side_labels=self._sides_to_edges(tri1_sides, tri1_name))
        self._draw_tri(ax, v2_local, palette[2], "#111", lw,
                       labels=list(tri2_name),
                       side_labels=self._sides_to_edges(tri2_sides, tri2_name))
        ax.set_xlim(-1, offset_x + 5.2)
        ax.set_ylim(-1, 4)
        ax.set_title("Two similar triangles", fontsize=14,
                     fontweight="bold", pad=8)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _sides_to_edges(side_map, tri_name):
        """Convert a dict like {AB: 3, BC: 4, CA: 2} and tri_name 'ABC' to
        {(i,j): val} for drawing."""
        # Map vertex label → index in tri_name
        idx = {c: i for i, c in enumerate(tri_name)}
        out = {}
        for side, val in side_map.items():
            if val is None:
                continue
            a, b = side[0], side[1]
            ia, ib = idx[a], idx[b]
            out[(ia, ib)] = val
        return out

    def _render_nested_parallel(self, rng, vertset, ad, db, de, bc,
                                 opts_strs, integer):
        fig, ax, style = self._setup_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        a, b, c, d, e, _ = vertset
        lw = max(1.5, style["line_width"])
        # Outer ABC
        outer = [(0, 0), (6, 0), (1.8, 4.5)]
        # inner DE on AC and AB, parallel to BC
        # We place D and E so that AD:AB = 1/k ratio.
        # Approximate: D is on segment A→B at 1/k, E on A→C at 1/k.
        # Here A = outer[2], B = outer[0], C = outer[1]  (flipped so A is apex).
        A = outer[2]
        B = outer[0]
        C = outer[1]
        ratio = ad / (ad + (db if isinstance(db, int) else float(db)))
        D_pt = (A[0] + (B[0] - A[0]) * ratio,
                A[1] + (B[1] - A[1]) * ratio)
        E_pt = (A[0] + (C[0] - A[0]) * ratio,
                A[1] + (C[1] - A[1]) * ratio)
        # Outer triangle ABC: only label the base BC (the unknown).
        self._draw_tri(ax, [B, C, A], palette[0], "#111", lw,
                       labels=[b, c, a],
                       side_labels={(0, 1): "?"})
        # inner DE  (triangle ADE)
        self._draw_tri(ax, [D_pt, E_pt, A], palette[2], "#111", lw,
                       labels=[d, e, a],
                       side_labels={(0, 1): str(de),
                                     (2, 0): str(ad)})
        # Mark DB explicitly
        ax.plot([D_pt[0], B[0]], [D_pt[1], B[1]],
                color="#111", linewidth=lw, alpha=0.7)
        # label DB
        mx = (D_pt[0] + B[0]) / 2
        my = (D_pt[1] + B[1]) / 2
        ax.text(mx - 0.4, my, str(db if isinstance(db, int)
                                   else round(db, 2)),
                fontsize=11, fontweight="bold", color="#0d3b66",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#0d3b66", alpha=0.9))
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 5.5)
        ax.set_title("Nested Similar Triangles (DE ∥ BC)",
                     fontsize=13, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_shared_vertex(self, rng, vertset, ab, ad, bc, de_gt, opts):
        fig, ax, style = self._setup_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        a, b, c, d, e, _ = vertset
        lw = max(1.5, style["line_width"])
        # A at origin. ABC smaller triangle; ADE larger.
        A = (0, 0)
        B = (3, 0)
        C = (1, 2.2)
        scale = ad / ab
        D_pt = (B[0] * scale, B[1] * scale)
        E_pt = (C[0] * scale, C[1] * scale)
        self._draw_tri(ax, [A, D_pt, E_pt], palette[0], "#111", lw,
                       labels=[a, d, e],
                       side_labels={(0, 1): str(ad if isinstance(ad, int)
                                                else round(ad, 2)),
                                     (1, 2): "?"})
        self._draw_tri(ax, [A, B, C], palette[2], "#111", lw,
                       labels=[a, b, c],
                       side_labels={(0, 1): str(ab), (1, 2): str(bc)})
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 6)
        ax.set_title("Shared vertex — similar triangles",
                     fontsize=13, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_nested_chain(self, rng, base_t1, base_t2, base_t3,
                              k, k2, opts):
        fig, ax, style = self._setup_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        lw = max(1.5, style["line_width"])
        # Three triangles scaled up. Place side by side.
        tris_widths = [base_t1 * 0.8, base_t2 * 0.8, base_t3 * 0.8]
        positions = []
        x = 0
        for w in tris_widths:
            positions.append(x)
            x += w + 0.8
        labels = ["T1", "T2", "T3"]
        colors = [palette[0], palette[2], palette[4]]
        bases_disp = [str(base_t1) if isinstance(base_t1, int)
                      else str(base_t1),
                      str(base_t2) if isinstance(base_t2, int)
                      else str(base_t2),
                      "?"]
        for x0, w, lbl, col, base_disp in zip(positions, tris_widths,
                                              labels, colors, bases_disp):
            h = 0.7 * w
            verts = [(x0, 0), (x0 + w, 0), (x0 + w / 2, h)]
            self._draw_tri(ax, verts, col, "#111", lw,
                           labels=[lbl + "_1", lbl + "_2", lbl + "_3"][:0] or None,
                           side_labels={(0, 1): base_disp})
            # Triangle name centered
            ax.text(x0 + w / 2, h / 2, lbl,
                    fontsize=16, fontweight="bold", color="#111",
                    ha="center", va="center")
        # Annotate ratios near arrows between triangles
        for i, (pos, w) in enumerate(zip(positions[:-1], tris_widths[:-1])):
            ratio_text = f"× {k}" if i == 0 else f"× {k2}"
            arrow_x = pos + w + 0.2
            ax.annotate(ratio_text, xy=(arrow_x + 0.5, 0.1),
                         fontsize=12, fontweight="bold", color="#b22222",
                         ha="center")
        ax.set_xlim(-0.5, x + 0.5)
        max_h = max(0.7 * w for w in tris_widths)
        ax.set_ylim(-0.8, max_h + 1.2)
        ax.set_title("Chain of similar triangles",
                     fontsize=13, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_altitude_hypotenuse(self, rng, a, b, c_hyp, subseg, opts):
        fig, ax, style = self._setup_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        lw = max(1.5, style["line_width"])
        # Right triangle with legs a (vertical) and b (horizontal).
        # B at origin (right angle), C to the right (along b), A up (along a).
        B = (0, 0)
        C = (b, 0)
        A = (0, a)
        verts = [A, B, C]
        self._draw_tri(ax, verts, palette[0], "#111", lw,
                       labels=["A", "B", "C"],
                       side_labels={(0, 1): str(a), (1, 2): str(b),
                                     (2, 0): str(c_hyp)})
        # Altitude from B to hypotenuse AC.
        # Foot H = projection of B onto AC.
        AC = (C[0] - A[0], C[1] - A[1])
        AB = (B[0] - A[0], B[1] - A[1])
        dot = AB[0] * AC[0] + AB[1] * AC[1]
        mag2 = AC[0] ** 2 + AC[1] ** 2
        t = dot / mag2
        H = (A[0] + AC[0] * t, A[1] + AC[1] * t)
        ax.plot([B[0], H[0]], [B[1], H[1]], color="#b22222",
                linewidth=lw * 0.9, linestyle="--")
        ax.text(H[0] + 0.1, H[1] + 0.1, "H", fontsize=12, fontweight="bold",
                color="#b22222")
        # Label the two sub-segments: AH and HC. AH = a^2/c, HC = b^2/c.
        # we label AH as "?" since that is the answer (adjacent to leg a).
        mid_ah = ((A[0] + H[0]) / 2, (A[1] + H[1]) / 2)
        mid_hc = ((H[0] + C[0]) / 2, (H[1] + C[1]) / 2)
        ax.text(mid_ah[0] + 0.2, mid_ah[1], "?",
                fontsize=14, fontweight="bold", color="#b00020",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="#b00020"))
        ax.text(mid_hc[0] + 0.1, mid_hc[1] - 0.2,
                str(round(b * b / c_hyp, 2)),
                fontsize=11, fontweight="bold", color="#0d3b66",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="#0d3b66"))
        ax.set_xlim(-1, b + 2)
        ax.set_ylim(-1, a + 2)
        ax.set_title("Right triangle with altitude to hypotenuse",
                     fontsize=13, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = SimilarTrianglesRatioQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer if ok else 'X'}")
