"""
Area Decomposition QA environment (batch 2 Part B, 2026-04-14).

Goal: find the area of a compound polygon by decomposing it into
rectangles and right triangles. Targets Area reasoning, combinatorial geometry, geometry problem solving.

Difficulty axes:
  A) Pattern A — n_decomposition_parts 2..6 sub-rectangles.
  B) Pattern G — gridline visibility fades with level.
  C) Pattern H — include triangle cut at L≥3, semicircle at L≥6.

Format: 4-way MCQ (letter).
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
from ._mcq_letter_lib import maybe_to_unit_mcq

class AreaDecompositionQA(StandaloneVisualEnv):
    ENV_NAME = "area_decomposition"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_parts":          2 + level // 2,   # 2..6
            "include_triangle": level >= 3,
            "include_semicircle": level >= 6,
            "grid_alpha":       max(0.1, 1.0 - 0.08 * level),
            "tight_distractors": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_parts"]

        for _ in range(30):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random,
                      cfg: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        n_parts = cfg["n_parts"]

        # Generate an L-shape / staircase from unit rectangles on a grid.
        # Start with a base rectangle and stack additional rectangles.
        rects = []
        # Base rectangle
        w0 = rng.randint(3, 6)
        h0 = rng.randint(2, 5)
        rects.append((0, 0, w0, h0))

        # Helper: detect overlap between rectangles (treating them as
        # half-open intervals on the unit grid). Two rectangles overlap if
        # their x-ranges and y-ranges both strictly intersect.
        def _overlaps(cand, others):
            cx, cy, cw, ch = cand
            for (ox, oy, ow, oh) in others:
                if (cx < ox + ow and cx + cw > ox and
                        cy < oy + oh and cy + ch > oy):
                    return True
            return False

        # Add more rectangles adjacent to existing ones. Reject placements
        # that overlap any existing rectangle (the area formula sums
        # non-overlapping parts; overlapping layouts would under-report
        # the visible area and produce wrong ground truth).
        for _ in range(n_parts - 1):
            placed = False
            for _attempt in range(25):
                base = rng.choice(rects)
                bx, by, bw, bh = base
                side = rng.choice(["top", "right"])
                rw = rng.randint(1, 4)
                rh = rng.randint(1, 4)
                if side == "top":
                    nx = bx + rng.randint(0, max(0, bw - 1))
                    ny = by + bh
                else:
                    nx = bx + bw
                    ny = by + rng.randint(0, max(0, bh - 1))
                cand = (nx, ny, rw, rh)
                if not _overlaps(cand, rects):
                    rects.append(cand)
                    placed = True
                    break
            if not placed:
                # Could not place a non-overlapping rectangle. Abort this
                # problem seed so we don't render an incorrect layout.
                return None

        total_area = sum(w * h for (_, _, w, h) in rects)
        if cfg["include_triangle"]:
            # A triangular cut: a right triangle removed from a corner.
            tri_leg = rng.randint(1, 3)
            total_area -= 0.5 * tri_leg * tri_leg
            tri_info = tri_leg
        else:
            tri_info = None
        if cfg["include_semicircle"]:
            rad = rng.randint(1, 2)
            total_area -= 0.5 * math.pi * rad * rad
            semi_info = rad
        else:
            semi_info = None

        gt = round(total_area, 2)
        if abs(gt - round(gt)) < 1e-6:
            gt = int(round(gt))

        given_pool = [
            "Find the area of the shaded compound region shown in the figure. The grid spacing is 1 unit.",
            "The figure shows a compound region drawn on a grid (each cell is 1 unit). Determine its area.",
            "Compute the total area of the compound shaded region in the figure (grid spacing = 1 unit).",
            "Inspect the shaded compound shape on the unit grid. What is its area?",
        ]
        given_text = rng.choice(given_pool)
        # reference-style bare numeric answer (no MCQ letter).
        ask_text = "Provide just the numeric area value."
        question = f"{given_text} {ask_text}"

        # Render — left panel shows the shape; right panel previously held
        # MCQ options. We pass an empty option list so the side panel is
        # blank (or shows a brief prompt).
        image = self._render(rects, tri_info, semi_info, cfg,
                             given_text, ask_text, [])
        # Add <answer> tag to the bare-numeric form so maybe_to_unit_mcq can
        # cleanly strip the tail when converting.
        question = question + " Place the answer in <answer>...</answer>."
        ans_str = str(gt)
        # 2026-05-04 exam alignment: 50% → 5-way MCQ with E="No correct
        # answer" + "cm²" unit (area).
        unit_rng = random.Random((self.seed or 0) * 17 + 5071)
        question, ans_str = maybe_to_unit_mcq(
            question, ans_str, unit_rng, prob=0.5, unit="cm²", n_options=5)
        return question, ans_str, image

    def _render(self, rects, tri_info, semi_info, cfg,
                given_text, ask_text, options) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(9.5 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_t.axis("off")

        palette = style["palette"]
        lw = style["line_width"]

        # Compute bounds
        xs = []
        ys = []
        for (x, y, w, h) in rects:
            xs.extend([x, x + w])
            ys.extend([y, y + h])
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad = 1.5
        ax_f.set_xlim(xmin - pad, xmax + pad)
        ax_f.set_ylim(ymin - pad, ymax + pad)

        # Draw gridlines
        grid_alpha = cfg["grid_alpha"]
        if grid_alpha > 0.1:
            for gx in range(int(xmin) - 1, int(xmax) + 2):
                ax_f.plot([gx, gx], [ymin - 1, ymax + 1], "-",
                          color="#bdc3c7", linewidth=0.5, alpha=grid_alpha)
            for gy in range(int(ymin) - 1, int(ymax) + 2):
                ax_f.plot([xmin - 1, xmax + 1], [gy, gy], "-",
                          color="#bdc3c7", linewidth=0.5, alpha=grid_alpha)

        # Draw shaded compound region.
        for (x, y, w, h) in rects:
            rect = mpatches.Rectangle((x, y), w, h,
                                       facecolor=palette[0],
                                       edgecolor=style["geo_line_color"],
                                       linewidth=lw, alpha=0.4)
            ax_f.add_patch(rect)
            ax_f.text(x + w / 2, y + h / 2, f"{w}x{h}",
                      fontsize=fs - 1, family=ff, ha="center",
                      va="center", color=style["geo_line_color"])

        # Triangle cut
        if tri_info is not None:
            t = tri_info
            # Cut from the top-right corner of the last rectangle.
            lx, ly, lw_, lh_ = rects[-1]
            tri_verts = [(lx + lw_, ly + lh_),
                         (lx + lw_ - t, ly + lh_),
                         (lx + lw_, ly + lh_ - t)]
            tri_patch = mpatches.Polygon(tri_verts, closed=True,
                                          facecolor=style["bg_color"],
                                          edgecolor=palette[5],
                                          linewidth=lw * 1.2)
            ax_f.add_patch(tri_patch)
            ax_f.text(lx + lw_ - t * 0.35, ly + lh_ - t * 0.35,
                      f"cut {t}", fontsize=fs - 1, family=ff,
                      color=palette[5])

        # Semi-circle removal
        if semi_info is not None:
            r = semi_info
            bx, by, bw, bh = rects[0]
            circ = mpatches.Circle((bx + bw / 2, by + bh),
                                    r, facecolor=style["bg_color"],
                                    edgecolor=palette[4], linewidth=lw)
            ax_f.add_patch(circ)
            ax_f.text(bx + bw / 2, by + bh + 0.2, f"r={r}",
                      fontsize=fs - 1, family=ff, ha="center",
                      color=palette[4])

        area_title_pool = ["Compound Region", "Area Decomposition",
                           "Shaded Region", "Figure", "Grid Region"]
        ax_f.set_title(self._rng.choice(area_title_pool),
                       fontsize=fs + 1, family=ff, pad=6)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        for spine in ax_f.spines.values():
            spine.set_color("#7f8c8d")

        # Text panel
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Problem:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        for ln in self._wrap(given_text + " " + ask_text, 42):
            ax_t.text(0.3, y, ln, fontsize=fs, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        if options:
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
    env = AreaDecompositionQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(out_dir,
                                f"area_decomposition_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  Q (first 100): {env.get_instruction()[:100]}")
            print(f"  A: {env._answer}")
