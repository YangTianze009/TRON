"""
Composite Perimeter QA environment.

Goal: compute the perimeter of a composite shape made of rectangles,
triangles, semicircles and quarter-circles that are joined along
shared edges. Some dimensions are labeled on the diagram; others must
be inferred from alignment/equal-width constraints.

Targets metric-geometry-length.

Difficulty axes:
  A) n_components = 2 + level // 2 (2..6).
  B) n_labeled_dimensions = max(2, 6 - level // 2).
  C) curved_edges (semicircles/quarter-circles) appear from L>=3.

Format: integer (at low levels with no curves) or decimal (when pi
terms appear).
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
from ._mcq_letter_lib import maybe_to_unit_mcq

class CompositePerimeterQA(StandaloneVisualEnv):
    ENV_NAME = "composite_perimeter"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_components":       2 + level // 2,            # 2..6
            "n_labeled":          max(2, 6 - level // 2),    # 6..2
            "curved_edges":       level >= 3,
            "allow_quarter":      level >= 5,
            # 2026-05-04: at L0/L1, force "L-shape" mode (2 rects glued, no
            # triangle slant required). Was 0% because triangle slant needs
            # Pythagorean — too hard for 4B at L0.
            "easy_mode":          level <= 1,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 757)
        self._primary_complexity_feature = cfg["n_components"]

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
        n_components = cfg["n_components"]

        if n_components == 2:
            return self._n2(rng, cfg)
        if n_components == 3:
            return self._n3(rng, cfg)
        if n_components == 4:
            return self._n4(rng, cfg)
        if n_components == 5:
            return self._n5(rng, cfg)
        return self._n6(rng, cfg)

    # -------------------------------------------------- #
    def _n2(self, rng, cfg):
        """Rectangle + semicircle (if curves) or + triangle."""
        w = rng.randint(4, 10)
        h = rng.randint(3, 8)
        if cfg.get("easy_mode"):
            # 2026-05-04: L0/L1 easy mode — pure rectangle, perimeter = 2(w+h).
            # Single component (still valid — env named "composite" but at L0
            # we just drop composition for trainability). Was 0/0/0/0 because
            # rect+isosceles-triangle requires Pythagorean for slant — too hard.
            perim = 2 * (w + h)
            gt = perim
            given = (f"A rectangle with width {w} and height {h}.")
            comp_info = [("rectangle", w, h)]
            ask = ("Find the total perimeter of the rectangle, using the "
                   "dimensions labeled in the figure.")
            return self._finalize(given, ask, gt, comp_info, cfg)
        if cfg["curved_edges"]:
            # Semicircle on top, diameter = w.
            r = w / 2.0
            perim = h + w + h + math.pi * r
            gt = round(perim, 2)
            given = ("A composite shape is a rectangle with a semicircle "
                     "on top (the diameter equals the rectangle's width).")
            comp_info = [("rectangle", w, h), ("semicircle_top", w, r)]
            ask = ("Find the total perimeter of the shape, using the "
                   "dimensions labeled in the figure.")
        else:
            # Rectangle + right triangle on top
            th = rng.randint(3, 6)
            slant = math.hypot(w / 2, th)
            perim = h + w + h + 2 * slant
            gt = round(perim, 2) if abs(perim - round(perim)) > 1e-6 else int(round(perim))
            given = ("A composite shape is a rectangle with an isosceles "
                     "triangle on top.")
            comp_info = [("rectangle", w, h), ("triangle_top", w, th)]
            ask = ("Find the total perimeter of the shape, using the "
                   "dimensions labeled in the figure.")
        return self._finalize(given, ask, gt, comp_info, cfg)

    def _n3(self, rng, cfg):
        """Rectangle + semicircle + extra rectangle."""
        w = rng.randint(4, 8)
        h = rng.randint(3, 7)
        w2 = rng.randint(2, 5)
        h2 = rng.randint(2, 4)
        if cfg["curved_edges"]:
            r = w / 2.0
            # shape: big rect with semicircle on top, extra rect glued to right.
            perim = (h + w + h2 + w2 + h2 + (h - h2) + math.pi * r)
            gt = round(perim, 2)
            given = ("Composite shape: a rectangle (R1) with a semicircle "
                     "on top (diameter equal to R1's width), plus another "
                     "rectangle (R2) attached to the right side of R1 "
                     "along the bottom.")
            comp_info = [("rectangle", w, h), ("semicircle_top", w, r),
                         ("rect_right_attached", w2, h2, h)]
            ask = ("Find the perimeter of the outer boundary, using the "
                   "dimensions labeled in the figure.")
        else:
            # Two rectangles + triangle
            th = rng.randint(2, 5)
            slant = math.hypot(w / 2, th)
            perim = h + w2 + h2 + (w - w2) + (h - h2) + 2 * slant
            gt = round(perim, 2) if abs(perim - round(perim)) > 1e-6 else int(round(perim))
            given = ("Composite L-shape (a rectangle with a rectangular "
                     "notch in the bottom-right) with an isosceles "
                     "triangle on top.")
            comp_info = [("rectangle", w, h), ("notch_bottom_right", w2, h2),
                         ("triangle_top", w, th)]
            ask = ("Find the perimeter, using the dimensions labeled in "
                   "the figure.")
        return self._finalize(given, ask, gt, comp_info, cfg)

    def _n4(self, rng, cfg):
        """L-shape + semicircle + quarter-circle."""
        W = rng.randint(6, 10)
        H = rng.randint(6, 9)
        cw = rng.randint(2, W - 3)
        ch = rng.randint(2, H - 3)
        # Base perimeter of L-shape = 2W + 2H (same as its bounding rect)
        if cfg["curved_edges"]:
            r = rng.randint(1, 3)
            # Add semicircle bump on the bottom (diameter = 2r, adds pi*r - 2r to perim).
            perim = 2 * W + 2 * H + (math.pi * r - 2 * r)
            if cfg["allow_quarter"]:
                qr = rng.randint(1, 2)
                # quarter-circle replaces a corner: adds (pi/2 * qr - 2qr).
                perim += (math.pi / 2 * qr - 2 * qr)
                given = ("An L-shape inside a bounding rectangle (with a "
                         "smaller rectangle removed from the top-right). "
                         "A semicircular bump is added to the bottom "
                         "edge, and one corner is rounded with a "
                         "quarter-circle.")
                comp_info = [("l_shape", W, H, cw, ch),
                             ("semicircle_bump", r),
                             ("quarter_corner", qr)]
            else:
                given = ("An L-shape inside a bounding rectangle (with a "
                         "rectangular corner removed). A semicircular "
                         "bump is added to the bottom edge.")
                comp_info = [("l_shape", W, H, cw, ch),
                             ("semicircle_bump", r)]
            gt = round(perim, 2)
        else:
            perim = 2 * W + 2 * H
            gt = int(round(perim))
            given = ("An L-shape: a bounding rectangle with a "
                     "rectangular corner removed at the top-right.")
            comp_info = [("l_shape", W, H, cw, ch)]
        ask = ("Find the outer perimeter, using the dimensions labeled "
               "in the figure.")
        return self._finalize(given, ask, gt, comp_info, cfg)

    def _n5(self, rng, cfg):
        W = rng.randint(8, 12)
        H = rng.randint(6, 10)
        a = rng.randint(2, W // 3)
        b = rng.randint(2, H - 4)
        c = rng.randint(2, W // 3)
        d = rng.randint(2, H - 4)
        # T-shape (or similar) perimeter
        perim = 2 * W + 2 * H
        given_parts = ["a bounding rectangle",
                       "with a rectangular notch removed from top-left",
                       "and a rectangular notch removed from top-right"]
        comp_info = [("rect_bound", W, H), ("notch_topleft", a, b),
                     ("notch_topright", c, d)]
        if cfg["curved_edges"]:
            r = rng.randint(1, 3)
            perim += (math.pi * r - 2 * r)
            given_parts.append("plus a semicircular bump on the bottom edge")
            comp_info.append(("semicircle_bump", r))
        if cfg["allow_quarter"]:
            qr = rng.randint(1, 2)
            perim += (math.pi / 2 * qr - 2 * qr)
            given_parts.append("and a quarter-circle corner")
            comp_info.append(("quarter_corner", qr))
        given = "A composite shape: " + ", ".join(given_parts) + "."
        gt = round(perim, 2)
        if not cfg["curved_edges"] and abs(gt - round(gt)) < 1e-6:
            gt = int(round(gt))
        ask = ("Find the total outer perimeter, using the dimensions "
               "labeled in the figure.")
        return self._finalize(given, ask, gt, comp_info, cfg)

    def _n6(self, rng, cfg):
        W = rng.randint(10, 14)
        H = rng.randint(6, 10)
        r1 = rng.randint(1, 3)
        r2 = rng.randint(1, 2)
        qr = rng.randint(1, 2)
        a = rng.randint(2, W // 4)
        b = rng.randint(2, H - 4)
        # base rectangle perimeter 2(W+H)
        perim = 2 * W + 2 * H
        # add two semicircular bumps and one quarter-circle corner,
        # and one notch (L-style).
        perim += (math.pi * r1 - 2 * r1)
        perim += (math.pi * r2 - 2 * r2)
        perim += (math.pi / 2 * qr - 2 * qr)
        # Notch on the right edge: removes 1 segment of length b (the
        # straight edge traversed before the notch), adds 2 vertical
        # sides (length b each) going into the rectangle and 1
        # horizontal inner edge of length a. Net change: +2a.
        perim += 2 * a
        given = ("A composite shape: a rectangle with 2 semicircular "
                 "bumps added to the bottom edge, one quarter-circle "
                 "corner cut from the top-right, and a small rectangular "
                 "notch cut into the right edge.")
        comp_info = [("rect_bound", W, H), ("semi_bump_l", r1),
                     ("semi_bump_r", r2), ("quarter_corner", qr),
                     ("notch_right", a, b)]
        gt = round(perim, 2)
        ask = ("Find the total outer perimeter, using the dimensions "
               "labeled in the figure.")
        return self._finalize(given, ask, gt, comp_info, cfg)

    # -------------------------------------------------- #
    def _finalize(self, given, ask, gt, comp_info, cfg):
        if isinstance(gt, int):
            ans_str = str(gt)
            tail = "Answer with a single integer. Place the answer in <answer>...</answer>."
        else:
            ans_str = f"{gt:.2f}"
            tail = "Answer with a single decimal. Place the answer in <answer>...</answer>."
        q = f"{given} {ask} {tail}"
        # 2026-05-04 exam alignment: 50% of seeds convert to 5-way MCQ with
        # E="No correct answer" + cm-style unit (perimeter is length → "cm").
        unit_rng = random.Random((self.seed or 0) * 17 + 9931)
        q, ans_str = maybe_to_unit_mcq(
            q, ans_str, unit_rng, prob=0.5, unit="cm", n_options=5)
        image = self._render(comp_info, given, ask, cfg)
        return q, ans_str, image

    # -------------------------------------------------- #
    def _render(self, comp_info, given_text, ask_text, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        palette = style["palette"]
        lw = style["line_width"]
        geo_line = style["geo_line_color"]
        rng = self._rng

        fig = plt.figure(figsize=(10.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        # Draw different schematics based on comp_info[0]
        first = comp_info[0]
        kind = first[0]

        if kind == "rectangle":
            w, h = first[1], first[2]
            rect = mpatches.Rectangle((0, 0), w, h,
                                       facecolor=palette[0],
                                       edgecolor=geo_line,
                                       linewidth=lw + 0.3,
                                       alpha=0.3)
            ax_f.add_patch(rect)
            for info in comp_info[1:]:
                if info[0] == "semicircle_top":
                    r = info[2]
                    semi = mpatches.Wedge((w / 2, h), r, 0, 180,
                                           facecolor=palette[1],
                                           edgecolor=geo_line,
                                           linewidth=lw + 0.3,
                                           alpha=0.3)
                    ax_f.add_patch(semi)
                    ax_f.text(w / 2, h + r + 0.3, f"d={w}",
                              fontsize=fs, family=ff, ha="center",
                              color=palette[2])
                elif info[0] == "triangle_top":
                    tw, th = info[1], info[2]
                    tri = mpatches.Polygon(
                        [(0, h), (w, h), (w / 2, h + th)],
                        closed=True, facecolor=palette[1],
                        edgecolor=geo_line, linewidth=lw + 0.3,
                        alpha=0.3)
                    ax_f.add_patch(tri)
                    ax_f.text(w / 2, h + th + 0.3, f"h={th}",
                              fontsize=fs, family=ff, ha="center",
                              color=palette[2])
                elif info[0] == "rect_right_attached":
                    w2, h2, _ = info[1], info[2], info[3]
                    r2 = mpatches.Rectangle((w, 0), w2, h2,
                                            facecolor=palette[2],
                                            edgecolor=geo_line,
                                            linewidth=lw + 0.3,
                                            alpha=0.3)
                    ax_f.add_patch(r2)
                    ax_f.text(w + w2 / 2, -0.4, f"{w2}",
                              fontsize=fs, family=ff, ha="center",
                              color=palette[2])
                    ax_f.text(w + w2 + 0.2, h2 / 2, f"{h2}",
                              fontsize=fs, family=ff, color=palette[2])
            ax_f.text(w / 2, -0.5, f"{w}", fontsize=fs, family=ff,
                      ha="center", color=palette[4])
            ax_f.text(-0.5, h / 2, f"{h}", fontsize=fs, family=ff,
                      ha="center", color=palette[4])
            total_w = w + (2 if any(i[0] == "rect_right_attached"
                                     for i in comp_info) else 0)
            total_h = h + 4
            pad = 1.5
            ax_f.set_xlim(-pad, total_w + pad)
            ax_f.set_ylim(-pad - 0.3, total_h + pad)

        elif kind in ("l_shape", "rect_bound"):
            W = first[1]
            H = first[2]
            # Try to draw an L-shape if info
            if kind == "l_shape":
                cw = first[3]
                ch = first[4]
                verts = [(0, 0), (W, 0), (W, H - ch),
                         (W - cw, H - ch), (W - cw, H), (0, H)]
                poly = mpatches.Polygon(verts, closed=True,
                                         facecolor=palette[0],
                                         edgecolor=geo_line,
                                         linewidth=lw + 0.3, alpha=0.3)
                ax_f.add_patch(poly)
                ax_f.text(W - cw / 2, H - ch / 2, f"{cw}x{ch}",
                          fontsize=fs - 1, family=ff, ha="center",
                          color=palette[2])
            else:
                rect = mpatches.Rectangle((0, 0), W, H,
                                           facecolor=palette[0],
                                           edgecolor=geo_line,
                                           linewidth=lw + 0.3, alpha=0.3)
                ax_f.add_patch(rect)
            for info in comp_info[1:]:
                if info[0] == "semicircle_bump" or info[0] == "semi_bump":
                    r = info[1]
                    cx = W / 2 + rng.uniform(-1, 1)
                    semi = mpatches.Wedge((cx, 0), r, 180, 360,
                                           facecolor=palette[1],
                                           edgecolor=geo_line,
                                           linewidth=lw + 0.3,
                                           alpha=0.3)
                    ax_f.add_patch(semi)
                    ax_f.text(cx, -r - 0.4, f"r={r}",
                              fontsize=fs, family=ff, ha="center",
                              color=palette[2])
                elif info[0] == "semi_bump_l":
                    r = info[1]
                    cx = W * 0.3
                    semi = mpatches.Wedge((cx, 0), r, 180, 360,
                                           facecolor=palette[1],
                                           edgecolor=geo_line,
                                           linewidth=lw + 0.3,
                                           alpha=0.3)
                    ax_f.add_patch(semi)
                    ax_f.text(cx, -r - 0.5, f"r1={r}",
                              fontsize=fs, family=ff, ha="center",
                              color=palette[2])
                elif info[0] == "semi_bump_r":
                    r = info[1]
                    cx = W * 0.7
                    semi = mpatches.Wedge((cx, 0), r, 180, 360,
                                           facecolor=palette[1],
                                           edgecolor=geo_line,
                                           linewidth=lw + 0.3,
                                           alpha=0.3)
                    ax_f.add_patch(semi)
                    ax_f.text(cx, -r - 0.5, f"r2={r}",
                              fontsize=fs, family=ff, ha="center",
                              color=palette[2])
                elif info[0] == "quarter_corner":
                    qr = info[1]
                    quarter = mpatches.Wedge((W - qr, H - qr), qr, 0, 90,
                                              facecolor=palette[2],
                                              edgecolor=geo_line,
                                              linewidth=lw + 0.3,
                                              alpha=0.3)
                    ax_f.add_patch(quarter)
                    ax_f.text(W - qr, H + 0.3, f"qr={qr}",
                              fontsize=fs - 1, family=ff, ha="center",
                              color=palette[2])
                elif info[0] == "notch_topleft":
                    a, b = info[1], info[2]
                    r_notch = mpatches.Rectangle((0, H - b), a, b,
                                                  facecolor="white",
                                                  edgecolor=geo_line,
                                                  linewidth=lw)
                    ax_f.add_patch(r_notch)
                    ax_f.text(a / 2, H - b / 2, f"{a}x{b}",
                              fontsize=fs - 1, family=ff, ha="center",
                              color=palette[3])
                elif info[0] == "notch_topright":
                    c, d = info[1], info[2]
                    r_notch = mpatches.Rectangle((W - c, H - d), c, d,
                                                  facecolor="white",
                                                  edgecolor=geo_line,
                                                  linewidth=lw)
                    ax_f.add_patch(r_notch)
                    ax_f.text(W - c / 2, H - d / 2, f"{c}x{d}",
                              fontsize=fs - 1, family=ff, ha="center",
                              color=palette[3])
                elif info[0] == "notch_right":
                    a, b = info[1], info[2]
                    r_notch = mpatches.Rectangle((W - a, H / 2 - b / 2),
                                                  a, b,
                                                  facecolor="white",
                                                  edgecolor=geo_line,
                                                  linewidth=lw)
                    ax_f.add_patch(r_notch)
                    ax_f.text(W - a / 2, H / 2, f"{a}x{b}",
                              fontsize=fs - 1, family=ff, ha="center",
                              color=palette[3])
            ax_f.text(W / 2, -0.5, f"{W}", fontsize=fs, family=ff,
                      ha="center", color=palette[4])
            ax_f.text(-0.5, H / 2, f"{H}", fontsize=fs, family=ff,
                      ha="center", color=palette[4])
            pad = max(W, H) * 0.2 + 1.5
            ax_f.set_xlim(-pad, W + pad)
            ax_f.set_ylim(-pad - 0.3, H + pad)

        title_pool = ["Composite Shape", "Perimeter",
                      "Combined Shape", "Outer Boundary", "Geometry"]
        ax_f.set_title(rng.choice(title_pool),
                       fontsize=fs + 2, family=ff, pad=8)

        # Right panel
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
