"""Fractal pattern QA — iteration counting, scale factor.

Diversity: multiple fractal types (Sierpinski triangle, Sierpinski carpet,
Vicsek fractal, T-square fractal). Iteration shown only on the image.
Multiple question templates and view orientations.
"""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._template_lib import count_templates
from ._render_modes import pick_render_mode, textbook_params, sketch_context

_FRACTAL_TITLES = {
    "sierpinski": [
        "Sierpinski Triangle", "Triangular Fractal", "Self-similar Triangle",
        "Triangle Fractal", "Recursive Triangle Pattern",
        "Fractal Triangle Structure", "Triangle-based Fractal",
        "Nested Triangle Pattern",
    ],
    "carpet": [
        "Sierpinski Carpet", "Square Carpet Fractal", "Carpet Fractal",
        "Square Fractal", "Perforated Square Fractal",
        "Recursive Square Carpet", "Hollow Square Fractal",
        "Nested Square Pattern",
    ],
    "vicsek": [
        "Vicsek Fractal", "Plus Fractal", "Cross Fractal",
        "Plus-shaped Fractal", "Vicsek-style Pattern",
        "Recursive Plus Pattern", "Cross-shaped Fractal Figure",
        "Star Fractal",
    ],
    "tsquare": [
        "T-square Fractal", "Square Fractal", "Quad Fractal",
        "T-square Pattern", "Recursive T-square",
        "Square-division Fractal", "Quadrant Fractal",
        "Nested Quad Pattern",
    ],
}

# Extra templates for count_iteration (how deep is this fractal drawn)
_COUNT_ITERATION_TEMPLATES = [
    # Direct interrogative (4)
    "Look at the fractal in the image. How many iterations (depth levels) does the construction show?",
    "The figure shows a fractal pattern. At what iteration depth is it drawn?",
    "In the image, what is the iteration depth of the fractal?",
    "How many recursion levels does the fractal in the figure go through?",
    # Imperative (4)
    "Count the iterations of the fractal shown. Answer with a single integer.",
    "Determine the iteration depth of the figure. Give one integer.",
    "Report the number of iteration steps visible in the fractal.",
    "Identify the recursion depth of the fractal pattern shown.",
    # Declarative / cloze (4)
    "The fractal in the image is drawn at iteration depth ___.",
    "The figure shows the fractal at iteration ___.",
    "Let n be the iteration depth of the fractal. Find n.",
    "The construction has completed ___ recursive subdivisions.",
    # Reasoning-prompt (4)
    "Examine the fractal carefully and determine how many iterations it depicts.",
    "Based on the subdivisions visible, state the iteration depth.",
    "Reason about the recursive structure to find the iteration count.",
    "Inspect the fractal to decide what iteration it corresponds to.",
]

_AREA_FRACTION_TEMPLATES = [
    # Direct interrogative (4)
    "At the iteration shown, what percentage of the original shape's area is still filled? Round to 2 decimals.",
    "In the figure, what fraction (as a percentage) of the original area remains filled? Round to 2 decimals.",
    "What percentage of the starting area is filled at the iteration depicted? Round to 2 decimals.",
    "At the depicted iteration, what share of the original area (as a percent) is still filled? Round to 2 decimals.",
    # Imperative (4)
    "Compute the percentage of the original area that remains filled at this iteration. Round to 2 decimals.",
    "Find the filled-area ratio (as a percentage) for the fractal shown. Round to 2 decimals.",
    "Determine what percent of the original area is still colored. Round to 2 decimals.",
    "Calculate the proportion of the initial area that is filled, expressed as a percent. Round to 2 decimals.",
    # Declarative / cloze (4)
    "At the iteration shown, ___ % of the original area is filled. Round to 2 decimals.",
    "The filled area is ___ % of the starting area. Round to 2 decimals.",
    "Let p be the percentage of the original area that is still filled. Find p (2 decimals).",
    "The fraction of the original area remaining, as a percent, is ___. Round to 2 decimals.",
    # Reasoning-prompt (4)
    "Using the iteration depth visible in the image, compute the filled-area percentage to 2 decimals.",
    "Reason about the fractal's construction to determine what percent of the original area remains. Round to 2 decimals.",
    "Infer the filled area percentage of the original shape at the depicted iteration. Round to 2 decimals.",
    "Based on the fractal's recursive rule, state the percent of the starting area that is filled. Round to 2 decimals.",
]

_PERIMETER_SEGMENTS_TEMPLATES = [
    # Direct (4)
    "At the iteration shown, how many line segments make up the total boundary of all filled pieces?",
    "In the figure, what is the total number of boundary edges across all filled pieces?",
    "How many edges in total form the boundaries of the filled pieces in the figure?",
    "The filled pieces have a combined boundary made of how many line segments?",
    # Imperative (4)
    "Count the total line segments that form the boundaries of all filled pieces.",
    "Enumerate every boundary segment of the filled pieces. State the total.",
    "Tally the edges belonging to the outlines of all filled pieces.",
    "Report the sum of the number of boundary segments over all filled pieces.",
    # Cloze (4)
    "The total number of boundary segments is ___.",
    "Summed across all filled pieces, the boundary comprises ___ segments.",
    "Let S be the total number of boundary segments; compute S.",
    "The filled pieces have N boundary segments in total; find N.",
    # Reasoning-prompt (4)
    "Determine how many boundary segments the filled pieces have in total.",
    "Reason about the structure to compute the sum of boundary segments.",
    "Based on the number of filled pieces at this iteration, compute the total boundary-segment count.",
    "Using the iteration visible in the image, find the total number of edges forming the outlines.",
]

_REMOVED_COUNT_TEMPLATES = [
    # Direct (4)
    "At the iteration depicted, how many pieces have been removed (are now empty) cumulatively since the start of the construction?",
    "In the figure, how many originally-present pieces have been carved out?",
    "How many pieces have been excluded (emptied) in total up to this iteration?",
    "Up to the iteration shown, how many pieces have been removed from the fractal?",
    # Imperative (4)
    "Count the pieces that have been removed from the construction cumulatively.",
    "Determine the number of pieces carved out across all iterations up to the current one.",
    "Enumerate the pieces that are now empty (removed) in total. Give one integer.",
    "Report the cumulative count of emptied pieces across iterations.",
    # Cloze (4)
    "The cumulative number of removed pieces at this iteration is ___.",
    "Up to the depicted iteration, ___ pieces have been carved out.",
    "Let R denote the total pieces removed so far. Find R.",
    "The fractal has had ___ pieces carved out cumulatively.",
    # Reasoning-prompt (4)
    "Reason about how many pieces are carved out at each iteration and sum them up to this stage.",
    "Using the iteration visible in the image, compute the cumulative number of removed pieces.",
    "Determine the total number of pieces that have been removed up to and including this iteration.",
    "Based on the fractal's construction rule, find the cumulative pieces removed.",
]

class FractalPatternQA(StandaloneVisualEnv):
    ENV_NAME = "fractal_pattern"

    def _level_config(self, level: int) -> Dict:
        # L0-1: only sierpinski with low iter. L9: all fractals, area questions.
        # Previous schedule was bumpy: L3=0.65 then L5=0.40 then L7=0.55 because
        # count_iteration re-appeared in L4-5 then vanished at L6-7.
        # Now: count_filled everywhere; count_iteration stays on from L4 up;
        # area added gradually at L7 instead of jumping in only at L8.
        # Iter 3 (2026-04-17): L0 collapsed to 0.15 because 3^3 = 27
        # triangles at iter=3 is too many to count visually. Drop L0-1
        # to iter 1-2 (3 or 9 triangles — tractable visual count).
        # Iter 4 (2026-04-17): L3=0.35 dip — carpet at iter=3 yields 512
        # filled small squares (8^3), visually impossible to count. Cap
        # carpet/vicsek to iter 2 at L3 so L0-L3 can all be solved by
        # counting. Shift count_iteration (which asks only depth, no
        # counting) to L3 so the level is solvable.
        # Iter 5 (2026-04-22): L0=0.40 vs L3=1.00 — count_iteration at L3 was
        # far easier than count_filled at L0-2, creating an inverted curriculum.
        # Fix: force L0 to Sierpinski iter=1 (exactly 3 triangles — trivial),
        # L1 to iter=2 (9 triangles, still easy), keep count_filled through L3
        # with count_iteration only as a *mix* at L3, so the capability under
        # training is consistent across L0-L3.
        if level == 0:
            # iter=1 across all fractal types → GT in {3,4,5,8}, model has to
            # actually look at the image rather than memorize "L0 == 3".
            return {"iter_range": (1, 1), "qtypes": ["count_filled"],
                    "ftypes": ["sierpinski", "carpet", "vicsek", "tsquare"]}
        if level == 1:
            return {"iter_range": (2, 2), "qtypes": ["count_filled"],
                    "ftypes": ["sierpinski"]}
        if level == 2:
            return {"iter_range": (2, 2), "qtypes": ["count_filled"],
                    "ftypes": ["sierpinski", "carpet"]}
        if level == 3:
            # Mix count_filled + count_iteration so L3 still tests the
            # counting skill built at L0-2, rather than replacing it.
            return {"iter_range": (2, 3),
                    "qtypes": ["count_filled", "count_iteration"],
                    "ftypes": ["sierpinski", "carpet", "vicsek"]}
        if level <= 5:
            return {"iter_range": (2, 3),
                    "qtypes": ["count_filled", "count_iteration"],
                    "ftypes": ["sierpinski", "carpet", "vicsek"]}
        if level <= 7:
            return {"iter_range": (3, 4),
                    "qtypes": ["count_filled", "count_iteration"],
                    "ftypes": ["sierpinski", "carpet", "vicsek", "tsquare"]}
        return {"iter_range": (3, 4),
                "qtypes": ["total_area_fraction", "count_filled"],
                "ftypes": ["sierpinski", "carpet", "vicsek"]}

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 713)

        ftype = sub_rng.choice(cfg["ftypes"])
        lo_i, hi_i = cfg["iter_range"]
        n_iter = sub_rng.randint(lo_i, hi_i)
        # Carpet/Vicsek/T-square at iteration 5 explode in count, cap them.
        if ftype != "sierpinski" and n_iter > 4:
            n_iter = 4

        # Compute structure stats per fractal type.
        if ftype == "sierpinski":
            # 3^n filled triangles, removed = (3^n - 1) // 2, fraction =(3/4)^n
            n_filled = 3 ** n_iter
            removed = (3 ** n_iter - 1) // 2
            area_frac = (3 / 4) ** n_iter * 100
            n_segments = 3 ** (n_iter + 1)
            base_unit = "small triangle"
        elif ftype == "carpet":
            n_filled = 8 ** n_iter
            removed = (8 ** n_iter - 1) // 7
            area_frac = (8 / 9) ** n_iter * 100
            n_segments = 4 * 8 ** n_iter
            base_unit = "small square"
        elif ftype == "vicsek":
            # Vicsek (plus): each square -> 5 squares. Filled count = 5^n.
            n_filled = 5 ** n_iter
            removed = (5 ** n_iter - 1) // 4 * 4
            area_frac = (5 / 9) ** n_iter * 100
            n_segments = 4 * 5 ** n_iter
            base_unit = "small square"
        else:  # tsquare
            # Filled count grows as 4^n.
            n_filled = 4 ** n_iter
            removed = (4 ** n_iter - 1) // 3
            area_frac = 100.0  # T-square has no area removal in canonical form
            n_segments = 4 * 4 ** n_iter
            base_unit = "small square"

        # Build figure.
        style = self._random_style()
        mode = pick_render_mode(sub_rng)  # clean / textbook / sketch
        sc = style["figsize_scale"]
        fig_w = sub_rng.uniform(5.6, 6.6) * sc

        if mode == "textbook":
            tbp = textbook_params(sub_rng)
            fig, ax = plt.subplots(figsize=(fig_w, fig_w))
            fig.patch.set_facecolor(tbp["bg"])
            ax.set_facecolor(tbp["bg"])
            # In textbook mode, use thin dark stroke + very light grey fill.
            color = tbp["fill_color"]
            edge_override = tbp["line_color"]
            edge_lw = tbp["line_width"]
            fill_alpha_override = tbp["fill_alpha"]
        elif mode == "sketch":
            fig, ax = plt.subplots(figsize=(fig_w, fig_w))
            fig.patch.set_facecolor("#fffdf7")
            ax.set_facecolor("#fffdf7")
            color = sub_rng.choice(style["palette"])
            edge_override = "#1a1a1a"
            edge_lw = 1.6
            fill_alpha_override = 0.7
        else:
            fig, ax = plt.subplots(figsize=(fig_w, fig_w))
            fig.patch.set_facecolor(style["bg_color"])
            ax.set_facecolor(style["bg_color"])
            color = sub_rng.choice(style["palette"])
            edge_override = None
            edge_lw = None
            fill_alpha_override = None

        # Draw (with optional sketch wobble).
        def _do_draw():
            if ftype == "sierpinski":
                orient = sub_rng.choice(["up", "down", "left", "right"])
                pts = self._tri_pts(orient)
                self._draw_sierpinski(ax, pts, n_iter, color,
                                       edge=edge_override, lw=edge_lw,
                                       alpha=fill_alpha_override)
            elif ftype == "carpet":
                # Carpet: optional rotation for diversity.
                self._draw_carpet(ax, 0, 0, 6, n_iter, color,
                                   edge=edge_override, lw=edge_lw,
                                   alpha=fill_alpha_override)
            elif ftype == "vicsek":
                self._draw_vicsek(ax, 0, 0, 6, n_iter, color,
                                   edge=edge_override, lw=edge_lw,
                                   alpha=fill_alpha_override)
            else:
                self._draw_tsquare(ax, 0, 0, 6, n_iter, color,
                                    edge=edge_override, lw=edge_lw,
                                    alpha=fill_alpha_override)

        if mode == "sketch":
            with sketch_context(scale=1.2, length=60, randomness=1.5):
                _do_draw()
        else:
            _do_draw()

        ax.set_xlim(-1, 7); ax.set_ylim(-1, 7)
        ax.set_aspect("equal"); ax.axis("off")

        qtype = sub_rng.choice(cfg["qtypes"])

        ttl_pool = _FRACTAL_TITLES[ftype]
        if qtype == "count_iteration":
            title = sub_rng.choice(ttl_pool)
        else:
            title = f"{sub_rng.choice(ttl_pool)} (iter={n_iter})"
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Template selection: deterministic seed-index so every L0 seed
        # hits a distinct template (guarantees audit floor).
        def _pick_tpl(pool):
            return pool[(self.seed or 0) % len(pool)]

        if qtype == "count_iteration":
            question = _pick_tpl(_COUNT_ITERATION_TEMPLATES)
            answer = n_iter
        elif qtype == "count_filled":
            question = _pick_tpl(count_templates(f"filled {base_unit}"))
            answer = n_filled
        elif qtype == "total_area_fraction":
            frac = round(area_frac, 2)
            question = _pick_tpl(_AREA_FRACTION_TEMPLATES)
            answer = str(frac)
        elif qtype == "perimeter_segments":
            question = _pick_tpl(_PERIMETER_SEGMENTS_TEMPLATES)
            answer = str(n_segments)
        else:  # removed_count
            question = _pick_tpl(_REMOVED_COUNT_TEMPLATES)
            answer = str(removed)

        question += "\n\nReply with the answer inside <answer>...</answer>. Example: <answer>5</answer>"
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # --- fractal drawing helpers -------------------------------------- #

    def _tri_pts(self, orient):
        if orient == "up":
            return [(0, 0), (6, 0), (3, 5.196)]
        if orient == "down":
            return [(0, 5.196), (6, 5.196), (3, 0)]
        if orient == "left":
            return [(0, 2.6), (5.196, 0), (5.196, 5.196)]
        return [(5.196, 2.6), (0, 0), (0, 5.196)]

    def _draw_sierpinski(self, ax, pts, depth, color,
                          edge=None, lw=None, alpha=None):
        if depth == 0:
            tri = plt.Polygon(pts,
                               facecolor=color,
                               edgecolor=edge if edge else "black",
                               linewidth=lw if lw else 0.5,
                               alpha=alpha if alpha else 0.8)
            ax.add_patch(tri)
            return
        mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        m01 = mid(pts[0], pts[1])
        m12 = mid(pts[1], pts[2])
        m02 = mid(pts[0], pts[2])
        self._draw_sierpinski(ax, [pts[0], m01, m02], depth - 1, color,
                               edge=edge, lw=lw, alpha=alpha)
        self._draw_sierpinski(ax, [m01, pts[1], m12], depth - 1, color,
                               edge=edge, lw=lw, alpha=alpha)
        self._draw_sierpinski(ax, [m02, m12, pts[2]], depth - 1, color,
                               edge=edge, lw=lw, alpha=alpha)

    def _draw_carpet(self, ax, x, y, size, depth, color,
                      edge=None, lw=None, alpha=None):
        if depth == 0:
            r = mpatches.Rectangle((x, y), size, size,
                                    facecolor=color,
                                    edgecolor=edge if edge else "black",
                                    linewidth=lw if lw else 0.4,
                                    alpha=alpha if alpha else 0.85)
            ax.add_patch(r)
            return
        sub = size / 3.0
        # 8 of 9 sub-squares; skip middle (1,1).
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                self._draw_carpet(ax, x + i * sub, y + j * sub, sub,
                                  depth - 1, color,
                                  edge=edge, lw=lw, alpha=alpha)

    def _draw_vicsek(self, ax, x, y, size, depth, color,
                      edge=None, lw=None, alpha=None):
        if depth == 0:
            r = mpatches.Rectangle((x, y), size, size,
                                    facecolor=color,
                                    edgecolor=edge if edge else "black",
                                    linewidth=lw if lw else 0.4,
                                    alpha=alpha if alpha else 0.85)
            ax.add_patch(r)
            return
        sub = size / 3.0
        # Plus pattern: center + 4 mid-edges.
        cells = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]
        for i, j in cells:
            self._draw_vicsek(ax, x + i * sub, y + j * sub, sub,
                              depth - 1, color,
                              edge=edge, lw=lw, alpha=alpha)

    def _draw_tsquare(self, ax, x, y, size, depth, color,
                       edge=None, lw=None, alpha=None):
        if depth == 0:
            r = mpatches.Rectangle((x, y), size, size,
                                    facecolor=color,
                                    edgecolor=edge if edge else "black",
                                    linewidth=lw if lw else 0.4,
                                    alpha=alpha if alpha else 0.8)
            ax.add_patch(r)
            return
        sub = size / 2.0
        for i in range(2):
            for j in range(2):
                self._draw_tsquare(ax, x + i * sub, y + j * sub, sub,
                                   depth - 1, color,
                                   edge=edge, lw=lw, alpha=alpha)
