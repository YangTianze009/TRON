"""
Scale Drawing Measurement QA environment.

Goal: read a drawing with a scale bar (e.g. ``1 cm = 2 m``) and a pair
of marked points A, B, and determine the real-world distance between
them. Low levels are simple rectangles with integer scales and a
straight-line path. High levels use irregular multi-room floor plans
with non-integer scales and multi-segment diagonal paths.

Targets geometry problem solving.

Difficulty axes:
  A) shape_complexity: rectangle -> L-shape -> irregular multi-room.
  B) scale_factor: integer -> non-integer.
  C) path_type: straight -> multi-segment diagonal.

Format: 4-way MCQ (single letter).
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

class ScaleDrawingMeasurementQA(StandaloneVisualEnv):
    ENV_NAME = "scale_drawing_measurement"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            shape = "rectangle"
        elif level <= 5:
            shape = "l_shape"
        else:
            shape = "irregular"

        if level <= 1:
            scale = rng_choice = ("int_simple", [1, 2])
        elif level <= 3:
            scale = ("int_simple", [1, 2, 3])
        elif level <= 6:
            scale = ("int_med", [2, 3, 4, 5])
        else:
            scale = ("decimal", [2.5, 3.5, 4.5])

        if level <= 1:
            path = "straight_axis_aligned"
        elif level <= 4:
            path = "straight_diagonal"
        elif level <= 7:
            path = "two_segment"
        else:
            path = "three_segment"

        return {
            "shape": shape,
            "scale_family": scale[0],
            "scale_options": scale[1],
            "path_type": path,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 883)
        self._primary_complexity_feature = level

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
        # Pick scale.
        scale = rng.choice(cfg["scale_options"])  # drawing cm -> real m
        unit_real = rng.choice(["m", "m", "km"]) if level >= 3 else "m"

        # Build floor plan polygon
        if cfg["shape"] == "rectangle":
            w = rng.randint(4, 9)
            h = rng.randint(3, 7)
            poly = [(0, 0), (w, 0), (w, h), (0, h)]
        elif cfg["shape"] == "l_shape":
            W = rng.randint(6, 10)
            H = rng.randint(5, 9)
            cw = rng.randint(2, W - 3)
            ch = rng.randint(2, H - 3)
            poly = [(0, 0), (W, 0), (W, H - ch), (W - cw, H - ch),
                    (W - cw, H), (0, H)]
        else:
            # Irregular multi-room (up to 7-8 vertices).
            W = rng.randint(8, 12)
            H = rng.randint(6, 10)
            a = rng.randint(2, W // 2 - 1) if W // 2 > 2 else 2
            b = rng.randint(2, H // 2 - 1) if H // 2 > 2 else 2
            c = rng.randint(2, W // 2 - 1) if W // 2 > 2 else 2
            d = rng.randint(2, H - 3)
            poly = [(0, 0), (W, 0), (W, d), (W - c, d),
                    (W - c, H), (a, H), (a, H - b), (0, H - b)]

        # Pick path points
        path_type = cfg["path_type"]
        pts_cm = []  # path points in drawing cm
        if path_type == "straight_axis_aligned":
            # Along one wall
            wall_edges = [(poly[i], poly[(i + 1) % len(poly)])
                          for i in range(len(poly))]
            wall_edges = [e for e in wall_edges
                          if e[0][0] == e[1][0] or e[0][1] == e[1][1]]
            if not wall_edges:
                return None
            e = rng.choice(wall_edges)
            # Place A,B as two points on this edge
            t1, t2 = sorted([rng.random(), rng.random()])
            if t2 - t1 < 0.4:
                return None
            A = (e[0][0] + (e[1][0] - e[0][0]) * t1,
                 e[0][1] + (e[1][1] - e[0][1]) * t1)
            B = (e[0][0] + (e[1][0] - e[0][0]) * t2,
                 e[0][1] + (e[1][1] - e[0][1]) * t2)
            pts_cm = [A, B]
        elif path_type == "straight_diagonal":
            A = (0.5 + rng.random() * 1.0,
                 0.5 + rng.random() * 1.0)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            B = (max(xs) - 0.5 - rng.random() * 1.0,
                 max(ys) - 0.5 - rng.random() * 1.0)
            pts_cm = [A, B]
        elif path_type == "two_segment":
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            A = (0.5 + rng.random() * 1.0, 0.5 + rng.random() * 1.0)
            mid = (max(xs) / 2 + rng.uniform(-0.5, 0.5),
                   max(ys) / 2 + rng.uniform(-0.5, 0.5))
            B = (max(xs) - 0.5 - rng.random() * 1.0,
                 max(ys) - 0.5 - rng.random() * 1.0)
            pts_cm = [A, mid, B]
        else:  # three_segment
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            A = (0.6, 0.6)
            m1 = (max(xs) * 0.3, max(ys) * 0.8)
            m2 = (max(xs) * 0.7, max(ys) * 0.2)
            B = (max(xs) - 0.6, max(ys) - 0.6)
            pts_cm = [A, m1, m2, B]

        # Drawing distance (cm)
        total_cm = 0.0
        for i in range(len(pts_cm) - 1):
            total_cm += math.hypot(pts_cm[i + 1][0] - pts_cm[i][0],
                                   pts_cm[i + 1][1] - pts_cm[i][1])
        real_dist = total_cm * scale
        # Round by precision
        if cfg["scale_family"] == "decimal":
            gt_val = round(real_dist, 1)
        else:
            gt_val = round(real_dist, 0) if cfg["path_type"] in (
                "straight_axis_aligned",) else round(real_dist, 1)

        # Make distractors: apply common misconceptions
        # 1) forgot to multiply (drawing distance as answer)
        # 2) divided by scale instead
        # 3) +/- small perturbation
        # Require distractors to differ from gt_val by at least 10% of gt_val
        # (min gap 1.0) so they are not visually indistinguishable.
        min_gap = max(1.0, 0.10 * abs(gt_val))
        d1 = round(total_cm, 1) if total_cm != gt_val else gt_val + min_gap
        d2 = round(total_cm / scale, 1) if scale > 0 else gt_val - min_gap
        # Use a perturbation large enough to respect the min gap
        perturb = max(min_gap, 2.0)
        d3 = round(gt_val + rng.choice([-perturb, perturb,
                                         -1.5 * perturb, 1.5 * perturb]), 1)
        distractors = []
        candidates = [d1, d2, d3, round(gt_val * 2, 1), round(gt_val / 2, 1),
                      round(gt_val + 2 * min_gap, 1),
                      round(gt_val - 2 * min_gap, 1),
                      round(gt_val + 3 * min_gap, 1),
                      round(gt_val - 3 * min_gap, 1)]
        for c in candidates:
            # Must be positive, >= min_gap from gt_val, AND at least min_gap
            # from all existing distractors.
            if c <= 0:
                continue
            if abs(c - gt_val) < min_gap:
                continue
            if any(abs(c - d) < min_gap for d in distractors):
                continue
            distractors.append(c)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            return None

        # 2026-05-04 WeMath alignment: 50% of seeds use the WeMath-style 5-way
        # MCQ with E="No correct answer" trailing option (matches reference's
        # dominant surface format: "A. ...; B. ...; C. ...; D. ...; E. No
        # correct answer"). The other 50% keep the legacy 4-way letter MCQ.
        wemath_style = rng.random() < 0.5
        opts_vals = [gt_val] + distractors[:3]
        rng.shuffle(opts_vals)
        if opts_vals.count(gt_val) > 1:
            return None
        answer_letter = chr(ord("A") + opts_vals.index(gt_val))

        def fmt(v):
            if isinstance(v, (int,)) or abs(v - round(v)) < 1e-6:
                return f"{int(round(v))} {unit_real}"
            return f"{v:.1f} {unit_real}"

        opt_strs = [fmt(v) for v in opts_vals]
        if wemath_style:
            # Append E="No correct answer" — gt_val is still in A-D so the
            # correct letter does not change.
            opt_strs = opt_strs + ["No correct answer"]

        q_head = ("A floor plan has been drawn to scale; the scale is "
                  "given by the scale bar labeled on the drawing. Points "
                  "A and B are marked on the drawing. Using the scale "
                  "shown on the drawing and measuring from the figure, "
                  "what is the real-world distance from A to B along "
                  "the shown path?")
        n_opts = len(opt_strs)
        q = (q_head + "\n" + "\n".join(
            f"  ({chr(ord('A') + i)}) {opt_strs[i]}" for i in range(n_opts))
            + "\nAnswer with a single letter.")

        image = self._render(poly, pts_cm, scale, unit_real, opt_strs, cfg)
        return q, answer_letter, image

    # -------------------------------------------------- #
    def _render(self, poly, pts_cm, scale, unit_real, opts, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]
        palette = style["palette"]
        lw = style["line_width"]
        geo_line = style["geo_line_color"]
        rng = self._rng

        fig = plt.figure(figsize=(10.0 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        Wmax = max(xs)
        Hmax = max(ys)

        # Floor plan polygon
        poly_patch = mpatches.Polygon(poly, closed=True,
                                      facecolor=palette[0],
                                      edgecolor=geo_line,
                                      linewidth=lw + 0.5,
                                      alpha=0.25)
        ax_f.add_patch(poly_patch)

        # Optional interior wall to suggest multi-room
        if cfg["shape"] == "irregular" and rng.random() < 0.7:
            y_split = Hmax * rng.uniform(0.35, 0.65)
            ax_f.plot([0, Wmax * 0.7], [y_split, y_split], "-",
                      color=geo_line, linewidth=lw * 0.6, alpha=0.5)

        # Path: A -> ... -> B
        path_xs = [p[0] for p in pts_cm]
        path_ys = [p[1] for p in pts_cm]
        ax_f.plot(path_xs, path_ys, "--", color=palette[4],
                  linewidth=lw + 0.6, zorder=4)
        for i, p in enumerate(pts_cm):
            label = "A" if i == 0 else ("B" if i == len(pts_cm) - 1 else f"P{i}")
            color = palette[2] if i == 0 else (palette[5] if i == len(pts_cm) - 1
                                               else palette[3])
            ax_f.scatter([p[0]], [p[1]], s=80, c=color, edgecolors="black",
                         linewidths=1.5, zorder=5)
            ax_f.text(p[0] + 0.15, p[1] + 0.15, label,
                      fontsize=fs + 2, fontweight="bold",
                      family=ff, color="#1a1a1a", zorder=6)

        # Scale bar below the drawing
        bar_y = -1.6
        bar_x0 = 0.2
        bar_x1 = bar_x0 + 2.0  # 2 cm bar
        ax_f.plot([bar_x0, bar_x1], [bar_y, bar_y],
                  "-", color="#333", linewidth=lw + 1)
        # Tick marks
        ticks = np.linspace(bar_x0, bar_x1, 5)
        for i, t in enumerate(ticks):
            ax_f.plot([t, t], [bar_y - 0.12, bar_y + 0.12], "-",
                      color="#333", linewidth=lw + 0.4)
        ax_f.text((bar_x0 + bar_x1) / 2, bar_y - 0.45,
                  f"Scale: 1 cm = {scale} {unit_real}",
                  fontsize=fs, family=ff, ha="center", color="#1a1a1a")

        # Ruler (faint) beside scale bar
        ruler_x0 = bar_x0 + 3.2
        ruler_x1 = ruler_x0 + 3.0
        ax_f.plot([ruler_x0, ruler_x1], [bar_y, bar_y],
                  "-", color="#555", linewidth=lw * 0.7)
        for i in range(7):
            tx = ruler_x0 + i * 0.5
            h = 0.08 if i % 2 else 0.14
            ax_f.plot([tx, tx], [bar_y - h, bar_y], "-",
                      color="#555", linewidth=lw * 0.5)
            if i % 2 == 0:
                ax_f.text(tx, bar_y - 0.35, f"{i // 2}",
                          fontsize=fs - 2, family=ff, ha="center",
                          color="#555")
        ax_f.text((ruler_x0 + ruler_x1) / 2, bar_y + 0.22,
                  "cm", fontsize=fs - 1, family=ff, ha="center",
                  color="#555")

        pad = 0.8
        ax_f.set_xlim(-pad, Wmax + pad + 0.5)
        ax_f.set_ylim(bar_y - 1.3, Hmax + pad)

        title_pool = ["Floor Plan", "Scale Drawing",
                      "Building Plan", "Map", "Architectural Plan"]
        ax_f.set_title(rng.choice(title_pool),
                       fontsize=fs + 2, family=ff, pad=8)

        # Right panel: options
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Question:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        head = ("Using the scale bar, find the real-world distance from A "
                "to B along the marked path.")
        for ln in self._wrap(head, 40):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.4
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(opts):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.55
            if y < 0.5:
                break

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
