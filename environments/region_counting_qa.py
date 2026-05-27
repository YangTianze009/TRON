"""
Region Counting QA environment.

Draws lines dividing a bounded plane region.
Questions about the number of regions, bounded regions, and
intersection points created by the arrangement.

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 1 line → 2 regions. `count_regions` only. Trivial.
L1: 2 lines (non-parallel) → 4 regions. `count_regions` only.
L2: 2-3 lines, no parallel. `count_regions`.
L3: 3 lines, mix parallel/non-parallel. `count_regions` + `count_intersection_points`.
L4: 3-4 lines. Full question mix adds `bounded_regions`.
L5: 4 lines. All question types.
L6: 4-5 lines. Tighter rendering.
L7: 5 lines. Occlusion-level line clutter.
L8: 5-6 lines. More intersections.
L9: 6-7 lines. Full complexity.

parameter = {"level": int in [0,9]}
"""
import math
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = [
    "Lines in a Plane",
    "Plane Divisions",
    "Line Arrangement",
    "Line Plane",
    "Planar Lines",
]

class RegionCountingQA(StandaloneVisualEnv):
    ENV_NAME = "region_counting"

    QUESTION_TYPES = [
        "count_regions", "count_bounded_regions", "count_intersection_points",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choices(cfg["qtypes"], weights=cfg["qtype_weights"])[0]

        for _ in range(25):
            result = self._try_generate(qtype, level, cfg)
            if result is not None:
                self._primary_complexity_feature = level * 3 + len(result[1])
                return result
        return None

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            # 1 line = 2 regions; 2 non-parallel = 4 regions; 2 parallel = 3 regions.
            # All trivially visible. Mixes across seeds so GT varies.
            return {"qtypes": ["count_regions"], "qtype_weights": [1],
                    "min_lines": 1, "max_lines": 2, "allow_parallel": True}
        if level == 1:
            return {"qtypes": ["count_regions"], "qtype_weights": [1],
                    "min_lines": 2, "max_lines": 2, "allow_parallel": False}
        if level == 2:
            return {"qtypes": ["count_regions"], "qtype_weights": [1],
                    "min_lines": 2, "max_lines": 3, "allow_parallel": False}
        if level == 3:
            return {"qtypes": ["count_regions", "count_intersection_points"],
                    "qtype_weights": [6, 4],
                    "min_lines": 3, "max_lines": 3, "allow_parallel": True}
        if level == 4:
            return {"qtypes": ["count_regions", "count_intersection_points",
                               "count_bounded_regions"],
                    "qtype_weights": [5, 3, 2],
                    "min_lines": 3, "max_lines": 4, "allow_parallel": True}
        if level == 5:
            return {"qtypes": ["count_regions", "count_intersection_points",
                               "count_bounded_regions"],
                    "qtype_weights": [4, 3, 3],
                    "min_lines": 4, "max_lines": 4, "allow_parallel": True}
        if level == 6:
            return {"qtypes": ["count_regions", "count_intersection_points",
                               "count_bounded_regions"],
                    "qtype_weights": [4, 3, 3],
                    "min_lines": 4, "max_lines": 5, "allow_parallel": True}
        if level == 7:
            return {"qtypes": ["count_regions", "count_intersection_points",
                               "count_bounded_regions"],
                    "qtype_weights": [3, 3, 4],
                    "min_lines": 5, "max_lines": 5, "allow_parallel": True}
        if level == 8:
            return {"qtypes": ["count_regions", "count_intersection_points",
                               "count_bounded_regions"],
                    "qtype_weights": [3, 3, 4],
                    "min_lines": 5, "max_lines": 6, "allow_parallel": True}
        return {  # level 9
            "qtypes": ["count_regions", "count_intersection_points",
                       "count_bounded_regions"],
            "qtype_weights": [3, 3, 4],
            "min_lines": 6, "max_lines": 7, "allow_parallel": True,
        }

    # -------------------------------------------------------------- #
    # Line / intersection helpers
    # -------------------------------------------------------------- #

    def _generate_lines(self, rng, n, allow_parallel=True):
        lines_abc = []
        lines_render = []
        bound = 5.0

        attempts = 0
        while len(lines_abc) < n and attempts < 100:
            attempts += 1
            angle = rng.uniform(0, math.pi)
            if not allow_parallel:
                too_close = False
                for existing_a, existing_b, _ in lines_abc:
                    ea = math.atan2(existing_a, existing_b)
                    na = math.atan2(math.sin(angle), math.cos(angle))
                    diff = abs(ea - na) % math.pi
                    if diff < 0.25 or abs(diff - math.pi) < 0.25:
                        too_close = True
                        break
                if too_close:
                    continue

            c = rng.uniform(-2, 2)
            a = math.sin(angle)
            b = -math.cos(angle)
            lines_abc.append((a, b, c))

            pts = self._clip_line_to_box(a, b, c, -bound, bound, -bound, bound)
            if pts is not None:
                lines_render.append(pts)
            else:
                lines_abc.pop()

        return lines_abc, lines_render

    def _clip_line_to_box(self, a, b, c, xmin, xmax, ymin, ymax):
        pts = []
        if abs(b) > 1e-9:
            for x in [xmin, xmax]:
                y = -(a * x + c) / b
                if ymin - 0.01 <= y <= ymax + 0.01:
                    pts.append((x, max(ymin, min(ymax, y))))
        if abs(a) > 1e-9:
            for y in [ymin, ymax]:
                x = -(b * y + c) / a
                if xmin - 0.01 <= x <= xmax + 0.01:
                    pts.append((max(xmin, min(xmax, x)), y))

        unique = []
        for p in pts:
            is_dup = False
            for q in unique:
                if abs(p[0] - q[0]) < 0.01 and abs(p[1] - q[1]) < 0.01:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(p)

        if len(unique) < 2:
            return None
        return (unique[0][0], unique[0][1], unique[1][0], unique[1][1])

    def _line_intersection(self, l1, l2):
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            return None
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        return (x, y)

    def _count_intersections_in_box(self, lines_abc, bound=5.0):
        points = set()
        for i, j in combinations(range(len(lines_abc)), 2):
            pt = self._line_intersection(lines_abc[i], lines_abc[j])
            if pt is not None:
                x, y = pt
                if -bound <= x <= bound and -bound <= y <= bound:
                    points.add((round(x, 4), round(y, 4)))
        return len(points), list(points)

    def _count_regions_lines(self, n_lines, n_intersections):
        return 1 + n_lines + n_intersections

    def _count_bounded_regions(self, n_lines, n_intersections):
        bounded = 1 + n_intersections - n_lines
        return max(0, bounded)

    # -------------------------------------------------------------- #

    def _try_generate(self, qtype: str, level: int, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        sub_rng = random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + rng.randint(0, 10)
        )

        n_lines = sub_rng.randint(cfg["min_lines"], cfg["max_lines"])
        allow_parallel = cfg["allow_parallel"] and sub_rng.random() < 0.4

        lines_abc, lines_render = self._generate_lines(
            sub_rng, n_lines, allow_parallel=allow_parallel)

        if len(lines_abc) < cfg["min_lines"]:
            return None

        n = len(lines_abc)
        n_int, int_points = self._count_intersections_in_box(lines_abc)

        if qtype == "count_regions":
            regions = self._count_regions_lines(n, n_int)
            img = self._render(lines_render, int_points, n, sub_rng)
            stem = sub_rng.choice([
                f"The image shows {n} straight line{'s' if n > 1 else ''} drawn in a plane. Into how many regions do these lines divide the plane?",
                f"How many regions are formed when these {n} line{'s' if n > 1 else ''} divide the plane?",
                f"Count the total number of regions created by the arrangement of {n} line{'s' if n > 1 else ''} shown.",
            ])
            q = stem + " Answer with a single integer."
            return q, str(regions), img

        elif qtype == "count_bounded_regions":
            if n < 3:
                return None
            bounded = self._count_bounded_regions(n, n_int)
            if bounded < 1:
                return None
            img = self._render(lines_render, int_points, n, sub_rng)
            stem = sub_rng.choice([
                f"These {n} lines create some bounded (enclosed) regions. How many bounded regions are there?",
                f"Count the number of fully enclosed (bounded) regions formed by these {n} lines.",
            ])
            q = stem + " Answer with a single integer."
            return q, str(bounded), img

        elif qtype == "count_intersection_points":
            img = self._render(lines_render, int_points, n, sub_rng,
                               show_intersections=True)
            stem = sub_rng.choice([
                f"How many points of intersection are formed by these {n} lines within the visible region?",
                f"Count the number of intersection points where the {n} lines cross each other inside the square.",
            ])
            q = stem + " Answer with a single integer."
            return q, str(n_int), img
        return None

    # -------------------------------------------------------------- #
    # Rendering
    # -------------------------------------------------------------- #

    def _render(self, lines_render, int_points, n_lines, sub_rng,
                show_intersections=False):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect('equal')
        bound = 5.0

        bg_rect = mpatches.Rectangle(
            (-bound, -bound), 2 * bound, 2 * bound,
            facecolor=style["bg_color"], edgecolor=style["geo_line_color"],
            linewidth=style["line_width"], zorder=0)
        ax.add_patch(bg_rect)

        line_colors = list(style["palette"])
        sub_rng.shuffle(line_colors)
        lw = style["line_width"] + 0.6
        for i, (x1, y1, x2, y2) in enumerate(lines_render):
            color = line_colors[i % len(line_colors)]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=2)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            norm = math.sqrt(dx ** 2 + dy ** 2) + 1e-9
            ox, oy = -dy / norm * 0.3, dx / norm * 0.3
            ax.text(mx + ox, my + oy, f'L{i + 1}', fontsize=10,
                    fontweight='bold', color=color, ha='center', va='center',
                    zorder=4, bbox=dict(boxstyle='round,pad=0.15',
                                        facecolor='white', alpha=0.8,
                                        edgecolor=color))

        if show_intersections and int_points:
            for x, y in int_points:
                ax.plot(x, y, 'ko', markersize=8, zorder=5)
                ax.plot(x, y, 'yo', markersize=5, zorder=6)

        ax.set_xlim(-bound - 0.5, bound + 0.5)
        ax.set_ylim(-bound - 0.5, bound + 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        title_base = sub_rng.choice(_TITLE_VARIANTS)
        ax.set_title(f'{title_base} ({n_lines} line{"s" if n_lines > 1 else ""})',
                     fontsize=style["font_size_base"] + 4,
                     fontweight='bold', pad=12)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = RegionCountingQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: {gt}")
