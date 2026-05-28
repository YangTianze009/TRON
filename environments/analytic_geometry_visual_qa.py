"""
Analytic Geometry Visual QA (Task C, Analytic reasoning gap, ).

Plot geometric shapes on coordinate axes, ask for intersection points,
distances, or slopes using coordinate geometry.

Difficulty axes:
  A) Shape complexity (line -> circle -> parabola + line)
  B) Number of shapes (1 -> 3)
  C) Computation steps (read -> compute slope -> find intersection)
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

_T_SLOPE = [
    "Two points A and B are plotted on the coordinate plane as shown in the image. What is the slope of line AB? Give an integer or decimal rounded to 2 places.",
    "Points A and B appear on the coordinate grid shown. Compute the slope of segment AB. Answer as an integer or a 2-decimal number.",
    "Given the two points A and B plotted in the figure, find the slope of line AB. Integer or decimal to 2 places.",
    "From the coordinate diagram, determine the slope of line AB through the two plotted points. Round to 2 decimal places if not integer.",
    "The coordinate plane shows two points A and B. Calculate the slope of AB. Give the value (integer, or 2-decimal).",
    "Based on the plot, compute the slope m of the line through A and B. Integer or decimal rounded to 2 places.",
    "Look at A and B in the image. What is the slope of the line they determine? Answer integer or 2-decimal.",
    "Two labeled points A and B are shown in the coordinate grid. Find the slope of the line AB. 2 decimal places if needed.",
    "Using the plotted points A and B, what is the slope of line AB? Integer or 2-decimal.",
    "From the figure, determine rise/run for line AB. Give the slope as an integer or 2-place decimal.",
    "Consider points A and B in the image. Find the slope of segment AB. Answer integer or decimal (2 d.p.).",
    "Based on the plotted coordinates of A and B, calculate the slope of AB. Integer or 2-decimal.",
    "The image shows two points A, B. Compute the slope of the line AB. Answer: integer or 2-decimal.",
    "What is the slope of the line joining the plotted points A and B? Integer or decimal rounded to 2 places.",
    "Find the slope of AB from the plotted points A and B in the image. Give integer or 2-decimal number.",
    "Given the image of A and B on the coordinate plane, find slope(AB). Answer as integer or 2-decimal.",
]

_T_DIST = [
    "Two points A and B are plotted on the coordinate plane as shown in the image. What is the distance between A and B? Round to 2 decimals.",
    "Points A and B appear on the coordinate grid. Compute the distance |AB|. Round to 2 decimal places.",
    "From the plot, determine the distance between the two points A and B. 2 decimal places.",
    "The figure shows points A and B on a coordinate plane. What is |AB|? Round to 2 decimals.",
    "Find the distance between A and B based on their plotted positions. Answer to 2 decimal places.",
    "Using the coordinates of A and B shown in the image, compute the distance AB. 2-decimal answer.",
    "Based on the image, what is the length of segment AB? Round to 2 decimals.",
    "Compute |AB| for the two labeled points shown on the coordinate grid. 2 decimal places.",
    "Given points A and B in the figure, find the distance between them. Report to 2 decimal places.",
    "What is the Euclidean distance from A to B in the shown figure? Round to 2 decimals.",
    "From the coordinate diagram, determine the length of AB. Answer rounded to 2 decimal places.",
    "The image shows two points. What is the distance between A and B? Give 2 decimal places.",
    "Calculate |AB| using the plotted positions of A and B. 2 decimals.",
    "Based on the plotted points A and B, compute the distance |AB|. Report to 2 decimals.",
    "Given A and B on the coordinate grid, find the distance between them. Round to 2 decimals.",
    "Look at points A and B in the figure and compute the distance between them. 2-decimal answer.",
]

_T_MID = [
    "Two points A and B are plotted on the coordinate plane as shown in the image. What is the x-coordinate of the midpoint of AB?",
    "Points A and B are plotted on the coordinate grid. Give the x-coordinate of the midpoint M of segment AB.",
    "From the figure showing A and B, what is the x-coordinate of the midpoint of AB?",
    "Given the plotted points A and B, compute the x-coordinate of the midpoint of AB.",
    "The image shows A and B on a coordinate plane. Find the x-coordinate of their midpoint.",
    "Using the coordinates of A and B, what is the x-coordinate of the midpoint of segment AB?",
    "Based on the figure, compute the x-coordinate of M = midpoint(A, B).",
    "Find the x-coordinate of the midpoint of line segment AB as plotted in the image.",
    "What is M_x, the x-coordinate of the midpoint of A and B shown in the plot?",
    "From the plotted points A and B, determine the x-coordinate of the midpoint.",
    "The figure shows two points A and B. What is the x-component of their midpoint?",
    "Calculate the x-coordinate of the midpoint of AB based on the image.",
    "Based on the plotted A and B, what is the x-coordinate of midpoint(A, B)?",
    "Given A, B in the coordinate figure, find M_x for M = midpoint(AB).",
    "What is the x-coordinate of the midpoint of A and B, as shown in the plot?",
    "Report the x-coordinate of the midpoint of segment AB shown in the figure.",
]

_T_INT = [
    "Two lines are plotted as shown in the image (their equations appear in the legend). What is the x-coordinate of their intersection point? Integer answer.",
    "Two lines are shown in the coordinate plane, with equations in the legend. What is the x-coordinate of their intersection? Integer.",
    "The figure shows two intersecting lines (equations in legend). Find the x-coordinate of their intersection point. Integer answer.",
    "Given the two lines plotted (see legend for equations), what is the x-coordinate of the intersection? Answer as integer.",
    "Two lines appear in the plot with equations in the legend. Determine the x-coordinate where they cross. Integer.",
    "From the coordinate figure, two lines are plotted. What is the x-coordinate of their intersection? Integer.",
    "Based on the plot of two lines (equations in legend), compute the x-coordinate of the intersection point. Integer.",
    "The image shows two lines crossing. Using their equations in the legend, find the x-coordinate of the intersection. Integer answer.",
    "Two lines are shown; their equations are in the legend. What is the x-value at which they intersect? Integer.",
    "Given two plotted lines (see legend), find the x-coordinate of their crossing point. Integer.",
    "From the figure, two lines meet at a single point. What is the x-coordinate of that point? Integer.",
    "Two lines (legend equations) are plotted in the figure. Find the x-coordinate of their intersection. Integer answer.",
    "The legend shows equations of two plotted lines. What is the x-coordinate of their point of intersection? Integer.",
    "Using the two lines plotted (equations in the legend), determine the x-coordinate of the intersection. Integer.",
    "Two lines are drawn on the plane (equations given in the legend). What is the x-coordinate where they meet? Integer answer.",
    "Compute the x-coordinate of the intersection of the two plotted lines (equations in legend). Integer.",
]

_T_TAN = [
    "A circle is centered at the origin and an external point P lies on the positive x-axis, as shown in the image (the radius and P's coordinates are labeled). What is the length of the tangent from P to the circle? Round to 2 decimals.",
    "In the figure, a circle centered at the origin and an external point P on the positive x-axis are shown (radius and P's coordinates labeled). Compute the tangent length from P to the circle. Round to 2 decimals.",
    "The image shows a circle centered at the origin with an external point P on the positive x-axis. Using the labeled radius and P, find the tangent length from P. 2 decimals.",
    "Given the circle and external point P (labeled in the figure), what is the length of the tangent from P to the circle? Round to 2 decimals.",
    "A circle at origin and an external point P (positive x-axis) are drawn. Radius r and P's coordinates are shown. Find tangent length from P. 2 decimals.",
    "From the image, a circle is centered at O and P is external on the positive x-axis (labels shown). Compute |PT| where T is tangent point. 2 decimals.",
    "Using the figure (circle at origin, external P on +x-axis), what is the tangent segment length from P? 2 decimal places.",
    "The plot shows a circle at the origin and an external point P (labeled). What is the tangent length from P to the circle? Round to 2 decimals.",
    "Based on the labeled radius and external point P's coordinates in the image, find the tangent length from P. 2 decimals.",
    "A circle (centered at origin) and external point P are shown with labels. Find the length of the tangent from P to the circle. Round to 2 decimals.",
    "In the figure, the circle at the origin and external P on the +x-axis have labeled radius and P's coordinates. Calculate the tangent length. 2 decimals.",
    "From the plot of a circle (at origin) and an external point P (labeled), compute the tangent length from P to the circle. 2-decimal answer.",
    "Given the circle centered at O and the external point P (coordinates labeled), determine the length of the tangent line from P to the circle. 2 decimals.",
    "Look at the circle and external point P in the figure. Compute |PT|, the tangent from P to the circle. Round to 2 decimal places.",
    "Using the labeled radius r and point P on the x-axis (external to circle), find the tangent length from P. 2 decimals.",
    "The figure shows an origin-centered circle and an external labeled point P. What is the tangent segment length from P? Round to 2 decimals.",
]

class AnalyticGeometryVisualQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "analytic_geometry_visual"

    # 2026-05-04 R4: full-gradient redesign per a math benchmark analytic geom samples.
    # Five qtypes (slope, distance, midpoint, intersection, tangent) but
    # gradient was just chunking them into level bands. Real progressive:
    #   - Each level adds one MORE qtype to mix
    #   - Coord range scales with level
    #   - At higher levels, label_points=False (model must read coords from grid)
    #   - At highest levels, grid_visible=False (need fine coordinate reading)
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"qtypes": ["slope"], "coord_range": 4,
                    "grid_visible": True, "label_points": True}
        if level == 1:
            return {"qtypes": ["slope", "distance"], "coord_range": 4,
                    "grid_visible": True, "label_points": True}
        if level == 2:
            return {"qtypes": ["slope", "distance"], "coord_range": 5,
                    "grid_visible": True, "label_points": True}
        if level == 3:
            return {"qtypes": ["slope", "distance", "midpoint"],
                    "coord_range": 5, "grid_visible": True,
                    "label_points": True}
        if level == 4:
            return {"qtypes": ["distance", "midpoint", "line_intersection"],
                    "coord_range": 6, "grid_visible": True,
                    "label_points": True}
        if level == 5:
            return {"qtypes": ["distance", "midpoint", "line_intersection",
                                "circle_tangent"], "coord_range": 6,
                    "grid_visible": True, "label_points": True}
        if level == 6:
            return {"qtypes": ["distance", "line_intersection",
                                "circle_tangent"], "coord_range": 7,
                    "grid_visible": True, "label_points": True}
        if level == 7:
            # remove explicit point coordinates — model must read grid
            return {"qtypes": ["distance", "line_intersection",
                                "circle_tangent"], "coord_range": 7,
                    "grid_visible": True, "label_points": False}
        if level == 8:
            return {"qtypes": ["line_intersection", "circle_tangent"],
                    "coord_range": 8, "grid_visible": True,
                    "label_points": False}
        return {"qtypes": ["line_intersection", "circle_tangent"],
                "coord_range": 9, "grid_visible": False,
                "label_points": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level

        # Try up to 10 qtype/rng combos before giving up so edge cases
        # (e.g. tangent geometry where discriminant < 0) don't leak null.
        dispatch = {
            "slope": self._gen_slope,
            "distance": self._gen_distance,
            "midpoint": self._gen_midpoint,
            "line_intersection": self._gen_intersection,
            "circle_tangent": self._gen_circle_tangent,
        }
        for attempt in range(10):
            qtype = rng.choice(cfg["qtypes"])
            gen = dispatch.get(qtype)
            if gen is None:
                continue
            result = gen(rng, cfg)
            if result is not None:
                return result
        return None

    def _gen_slope(self, rng, cfg):
        cr = cfg.get("coord_range", 4)
        x1 = rng.randint(-cr, cr)
        y1 = rng.randint(-cr, cr)
        x2 = rng.randint(-cr, cr)
        while x2 == x1:
            x2 = rng.randint(-cr, cr)
        y2 = rng.randint(-cr, cr)

        if (y2 - y1) % (x2 - x1) == 0:
            slope = (y2 - y1) // (x2 - x1)
            answer = str(slope)
        else:
            slope = round((y2 - y1) / (x2 - x1), 2)
            answer = str(slope)

        img = self._plot_line_segment(x1, y1, x2, y2, cfg)
        sidx = (self.seed or 0) % 16
        question = _T_SLOPE[sidx]
        return question, answer, img

    def _gen_distance(self, rng, cfg):
        cr = cfg.get("coord_range", 4)
        x1 = rng.randint(-cr, cr)
        y1 = rng.randint(-cr, cr)
        x2 = rng.randint(-cr, cr)
        y2 = rng.randint(-cr, cr)
        while (x1, y1) == (x2, y2):
            x2 = rng.randint(-cr, cr)
            y2 = rng.randint(-cr, cr)

        dist = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
        img = self._plot_line_segment(x1, y1, x2, y2, cfg)
        sidx = (self.seed or 0) % 16
        question = _T_DIST[sidx]
        return question, str(dist), img

    def _gen_midpoint(self, rng, cfg):
        cr = cfg.get("coord_range", 4)
        x1 = rng.randint(-cr, cr)
        y1 = rng.randint(-cr, cr)
        x2 = rng.randint(-cr, cr)
        y2 = rng.randint(-cr, cr)
        while (x1, y1) == (x2, y2):
            x2 = rng.randint(-cr, cr)

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        img = self._plot_line_segment(x1, y1, x2, y2, cfg)
        # Ask for x-coordinate of midpoint
        if mx == int(mx):
            answer = str(int(mx))
        else:
            answer = str(mx)
        sidx = (self.seed or 0) % 16
        question = _T_MID[sidx]
        return question, answer, img

    def _gen_intersection(self, rng, cfg):
        m1 = rng.choice([-2, -1, 1, 2])
        b1 = rng.randint(-3, 3)
        m2 = rng.choice([-2, -1, 1, 2])
        while m2 == m1:
            m2 = rng.choice([-2, -1, 1, 2])
        b2 = rng.randint(-3, 3)

        num = b2 - b1
        den = m1 - m2
        if num % den != 0:
            return None
        x_int = num // den
        y_int = m1 * x_int + b1
        if abs(x_int) > 6 or abs(y_int) > 10:
            return None

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        xs = np.linspace(-6, 6, 200)
        ax.plot(xs, m1 * xs + b1, color=style["palette"][0], linewidth=2,
                label=f"y = {m1}x + {b1}")
        ax.plot(xs, m2 * xs + b2, color=style["palette"][1], linewidth=2,
                label=f"y = {m2}x + {b2}")
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.axvline(0, color="#333", linewidth=0.8)
        if cfg["grid_visible"]:
            ax.grid(True, alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-8, 8)
        ax.legend(fontsize=10)
        ax.set_title("Two Lines", fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        sidx = (self.seed or 0) % 16
        question = _T_INT[sidx]
        return question, str(x_int), img

    def _gen_circle_tangent(self, rng, cfg):
        # Circle centered at origin with radius r, tangent from external point
        r = rng.randint(2, 4)
        # External point at (px, 0) where px > r
        px = r + rng.randint(1, 3)
        # Length of tangent = sqrt(px^2 - r^2)
        tang_len = round(math.sqrt(px ** 2 - r ** 2), 2)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        circle = plt.Circle((0, 0), r, fill=False, ec=style["palette"][0],
                            linewidth=2)
        ax.add_patch(circle)
        ax.plot(0, 0, "k+", markersize=8)
        ax.plot(px, 0, "ro", markersize=8)
        ax.annotate(f"P({px}, 0)", xy=(px, 0), xytext=(px + 0.3, 0.3),
                   fontsize=10, color="red", fontweight="bold")
        ax.annotate(f"r = {r}", xy=(r / 2, 0.3), fontsize=10, color=style["palette"][0])
        if cfg["grid_visible"]:
            ax.grid(True, alpha=0.3)
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlim(-r - 2, px + 2)
        ax.set_ylim(-r - 2, r + 2)
        ax.set_aspect("equal")
        ax.set_title("Circle and External Point", fontsize=12, fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        sidx = (self.seed or 0) % 16
        question = _T_TAN[sidx]
        return question, str(tang_len), img

    def _plot_line_segment(self, x1, y1, x2, y2, cfg):
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        ax.plot([x1, x2], [y1, y2], "o-", color=style["palette"][0],
                linewidth=2, markersize=8)

        if cfg["label_points"]:
            ax.annotate(f"A({x1},{y1})", (x1, y1), textcoords="offset points",
                       xytext=(8, 8), fontsize=10, fontweight="bold")
            ax.annotate(f"B({x2},{y2})", (x2, y2), textcoords="offset points",
                       xytext=(8, 8), fontsize=10, fontweight="bold")

        ax.axhline(0, color="#333", linewidth=0.8)
        ax.axvline(0, color="#333", linewidth=0.8)
        if cfg["grid_visible"]:
            ax.grid(True, alpha=0.3)
        margin = 2
        ax.set_xlim(min(x1, x2) - margin, max(x1, x2) + margin)
        ax.set_ylim(min(y1, y2) - margin, max(y1, y2) + margin)
        ax.set_aspect("equal")
        ax.set_title("Coordinate Plane", fontsize=12, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskc"
    os.makedirs(out_dir, exist_ok=True)
    env = AnalyticGeometryVisualQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[analytic_geometry_visual L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"analytic_geometry_visual_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[analytic_geometry_visual L{level} s{s}] A={env._answer}")
