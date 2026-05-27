"""
Inscribed polygon QA — difficulty redesign 2026-04-14.

L0: polygon area given n and R (was ~L3). R shown on image.
L9: two concentric inscribed polygons (n1-gon in outer circle, n2-gon in inner),
    find area of annular region between them. No R label on image. Target 5-15%.
"""
import math
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class InscribedPolygonQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "inscribed_polygon"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        return {
            # Question types by level
            "qtype": [
                "polygon_area",         # L0
                "perimeter",            # L1
                "gap_area",             # L2
                "gap_area",             # L3
                "apothem_then_area",    # L4: multi-step
                "sector_minus_triangle",# L5: sector area minus triangle area
                "ratio_areas",          # L6: ratio of polygon area to circle area
                "inverse_from_area",    # L7: given area, find n
                "two_polygon_gap",      # L8: two polygons, gap between them
                "two_polygon_gap",      # L9
            ][level],
            # n range grows
            "n_choices": list(range(3, 5 + level)),  # 3-5 at L0, 3-14 at L9
            # R range
            "R_choices": list(range(3, 6 + level // 2)),
            # Show radius label on image — must always be shown because the
            # question text no longer mentions R (text-leakage fix 2026-04-17).
            "show_R_label": True,
            # Show n label — must always be shown for the same reason.
            "show_n_label": True,
            # Show grid
            "show_grid": level <= 3,
        }

    # Phrasings — reference image via "as shown" to avoid text leakage of n, R.
    # n and R are rendered on the image (n via title when show_n_label, R via
    # radius label when show_R_label); we intentionally do NOT inject them into
    # the question string so the model must READ the image.
    _PHRASE_BANK = {
        "polygon_area": [
            "What is the area of the inscribed polygon shown? Round to 2 decimals.",
            "Find the area of the regular polygon inscribed in the circle shown. Round to 2 decimal places.",
            "Compute the area of the polygon shown in the figure (2 decimals).",
            "Determine the area of the shown regular polygon. Two decimals.",
        ],
        "perimeter": [
            "What is the perimeter of the inscribed polygon shown? Round to 2 decimals.",
            "Find the perimeter of the regular polygon inscribed in the circle shown (2 decimals).",
            "Compute the perimeter of the polygon in the figure. Two decimals.",
            "Determine the perimeter of the shown regular polygon. Round to 2 decimals.",
        ],
        "gap_area": [
            "What is the area between the circle and the inscribed polygon shown? Round to 2 decimals.",
            "Find the area of the region inside the circle but outside the inscribed polygon shown (2 decimals).",
            "How much area lies between the circle and the inscribed polygon in the figure? 2 decimals.",
            "Find the area of the crescent regions between the circle and the inscribed polygon shown. Round to 2 decimals.",
        ],
        "apothem_then_area": [
            "For the inscribed polygon shown, first find the apothem, then compute the polygon area as (1/2) * apothem * perimeter. Round to 2 decimals.",
            "For the regular polygon shown, compute its area using the formula (1/2) * apothem * perimeter. Two decimals.",
            "Find the area of the regular polygon in the figure using A = (1/2) * a * P where a is the apothem and P is the perimeter. Two decimals.",
        ],
        "sector_minus_triangle": [
            "For the inscribed polygon shown, find the area of ONE circular segment (sector minus triangle). Round to 2 decimals.",
            "Consider the regular polygon inscribed in the circle shown. Compute the area of a single circular segment (one sector minus the corresponding triangle). 2 decimals.",
            "For the inscribed regular polygon in the figure, what is the area of a single segment between one polygon side and the corresponding arc? Two decimals.",
        ],
        "ratio_areas": [
            "For the regular polygon inscribed in the circle shown, what fraction of the circle's area does the polygon cover? Round to 4 decimals.",
            "For the figure shown, what fraction of the circular area is occupied by the inscribed regular polygon? Four decimals.",
            "Compute the ratio of the area of the inscribed regular polygon shown to the area of its circumscribing circle. Round to 4 decimals.",
        ],
        "inverse_from_area": [
            "The regular polygon inscribed in the circle shown has area {poly_area}. How many sides does it have?",
            "The inscribed regular polygon in the figure has area {poly_area}. Find the number of sides.",
            "The regular n-gon inscribed in the circle shown has area {poly_area}. What is n?",
        ],
        "two_polygon_gap": [
            "Two concentric inscribed regular polygons are shown (outer and inner). What is the area between the two polygons? Round to 2 decimals.",
            "The figure shows two concentric inscribed regular polygons. Find the area between them. 2 decimals.",
            "For the two concentric inscribed regular polygons shown, compute the area difference between them. Round to 2 decimals.",
        ],
    }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Level-aware sub_rng for proper level-dependent diversity
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 631)
        # Style RNG also seeded by level so seeds at same level differ visually
        self._rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1117)
        style = self._random_style()

        qtype = cfg["qtype"]
        n = rng.choice(cfg["n_choices"])
        R = rng.choice(cfg["R_choices"])

        # Layout variant: rotation angle and figure orientation per seed
        layout = {
            "rot_offset": rng.choice([0, math.pi/8, math.pi/6, math.pi/4, math.pi/3, math.pi/2]),
            "show_center": rng.choice([True, False]),
            "use_dotted": rng.choice(["--", ":", "-."]),
        }

        # Pre-compute standard quantities
        side = round(2 * R * math.sin(math.pi / n), 2)
        poly_area = round(0.5 * n * R**2 * math.sin(2 * math.pi / n), 2)
        circle_area = round(math.pi * R**2, 2)
        gap = round(circle_area - poly_area, 2)
        perimeter = round(n * side, 2)
        apothem = round(R * math.cos(math.pi / n), 2)

        # -------- Build question + answer ----------
        if qtype == "polygon_area":
            q = rng.choice(self._PHRASE_BANK["polygon_area"])
            answer = str(poly_area)
        elif qtype == "perimeter":
            q = rng.choice(self._PHRASE_BANK["perimeter"])
            answer = str(perimeter)
        elif qtype == "gap_area":
            q = rng.choice(self._PHRASE_BANK["gap_area"])
            answer = str(gap)
        elif qtype == "apothem_then_area":
            area_via_apothem = round(0.5 * apothem * perimeter, 2)
            q = rng.choice(self._PHRASE_BANK["apothem_then_area"])
            answer = str(area_via_apothem)
        elif qtype == "sector_minus_triangle":
            sector_area = round(math.pi * R**2 / n, 2)
            triangle_area = round(0.5 * R**2 * math.sin(2 * math.pi / n), 2)
            diff = round(sector_area - triangle_area, 2)
            q = rng.choice(self._PHRASE_BANK["sector_minus_triangle"])
            answer = str(diff)
        elif qtype == "ratio_areas":
            ratio = round(poly_area / circle_area, 4)
            q = rng.choice(self._PHRASE_BANK["ratio_areas"])
            answer = str(ratio)
        elif qtype == "inverse_from_area":
            q = rng.choice(self._PHRASE_BANK["inverse_from_area"]).format(poly_area=poly_area)
            answer = str(n)
        elif qtype == "two_polygon_gap":
            n2 = rng.choice([k for k in cfg["n_choices"] if k != n] or [n + 1])
            r2 = max(1, R - rng.randint(1, 2))
            inner_area = round(0.5 * n2 * r2**2 * math.sin(2 * math.pi / n2), 2)
            diff = round(abs(poly_area - inner_area), 2)
            q = rng.choice(self._PHRASE_BANK["two_polygon_gap"])
            answer = str(diff)
            return q, answer, self._draw(n, R, style, cfg, n2=n2, r2=r2, layout=layout)
        else:
            return None

        return q, answer, self._draw(n, R, style, cfg, layout=layout)

    def _draw(self, n, R, style, cfg, n2=None, r2=None, layout=None):
        if layout is None:
            layout = {"rot_offset": math.pi/2, "show_center": True, "use_dotted": "--"}
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6*sc, 6*sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]

        rot = layout["rot_offset"]
        # Outer circle + polygon
        circle = plt.Circle((0, 0), R, fill=False, edgecolor=palette[0],
                             linewidth=style["line_width"]+0.5,
                             linestyle=layout["use_dotted"])
        ax.add_patch(circle)

        angles = [2*math.pi*i/n + rot for i in range(n)]
        verts = [(R*math.cos(a), R*math.sin(a)) for a in angles]
        poly = plt.Polygon(verts, fill=True, facecolor=palette[1],
                           edgecolor=style["geo_line_color"],
                           linewidth=style["line_width"],
                           alpha=style["geo_fill_alpha"])
        ax.add_patch(poly)

        if cfg["show_R_label"]:
            ax.plot([0, verts[0][0]], [0, verts[0][1]], color=palette[2],
                    linewidth=1.5)
            mid = (verts[0][0]/2, verts[0][1]/2)
            ax.annotate(f"R={R}", xy=mid, fontsize=style["font_size_base"],
                        fontweight="bold", color=palette[2],
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        if n2 is not None and r2 is not None:
            c2 = plt.Circle((0, 0), r2, fill=False, edgecolor=palette[3],
                             linewidth=style["line_width"]+0.5,
                             linestyle=layout["use_dotted"])
            ax.add_patch(c2)
            # Inner polygon at a different rotation for visual diversity
            inner_rot = rot + math.pi / max(n2, 4)
            a2 = [2*math.pi*i/n2 + inner_rot for i in range(n2)]
            v2 = [(r2*math.cos(a), r2*math.sin(a)) for a in a2]
            p2 = plt.Polygon(v2, fill=True, facecolor=palette[3],
                             edgecolor=style["geo_line_color"],
                             linewidth=style["line_width"], alpha=0.3)
            ax.add_patch(p2)
            if cfg["show_R_label"]:
                ax.annotate(f"r={r2}", xy=(v2[0][0]/2, v2[0][1]/2-0.5),
                            fontsize=style["font_size_base"]-1, color=palette[3])

        # BUGFIX 2026-04-24: when both outer and inner polygons are drawn
        # (two_polygon_gap), include the inner n in the title so students can
        # identify it without having to count dashed sides on a tiny polygon.
        if n2 is not None:
            title = (f"Outer: {n}-gon, Inner: {n2}-gon"
                     if cfg["show_n_label"] else "Two inscribed polygons")
        else:
            title = f"Regular {n}-gon" if cfg["show_n_label"] else "Inscribed polygon"
        ax.set_title(title, fontsize=style["font_size_base"]+2, fontweight="bold")
        if layout["show_center"]:
            ax.plot(0, 0, "ko", markersize=4)
        lim = R * 1.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        if cfg["show_grid"]:
            ax.grid(True, alpha=0.2)
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=style["dpi"])
