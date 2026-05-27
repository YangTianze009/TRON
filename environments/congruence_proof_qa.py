"""
Congruence/similarity QA — identify congruence criterion, find unknown side/angle.
Targets: VO Property, geometry reasoning.
Capabilities: V1 (shape), V2 (labels), R2 (theorems), R5 (multi-step)
"""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class CongruenceProofQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "congruence_proof"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesigned: L0 starts with find_side + find_angle (old L3-5).
        # L9 requires double_similar + area_ratio multi-step.
        if level <= 1:
            return {"ptypes": ["find_side_sss", "find_angle_asa"]}
        if level <= 3:
            return {"ptypes": ["find_angle_asa", "similar_ratio"]}
        if level <= 5:
            return {"ptypes": ["similar_ratio", "similar_pythagoras"]}
        if level <= 7:
            return {"ptypes": ["similar_pythagoras", "similar_area_ratio"]}
        return {"ptypes": ["similar_area_ratio", "double_similar"]}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        problem_type = parameter.get("problem_type", rng.choice(cfg["ptypes"]))

        try:
            if problem_type == "identify_criterion":
                return self._identify(rng)
            elif problem_type == "find_side_sss":
                return self._find_side(rng)
            elif problem_type == "find_angle_asa":
                return self._find_angle(rng)
            elif problem_type == "similar_ratio":
                return self._similar_ratio(rng)
            elif problem_type == "similar_pythagoras":
                return self._similar_pythagoras(rng)
            elif problem_type == "similar_area_ratio":
                return self._similar_area_ratio(rng)
            elif problem_type == "double_similar":
                return self._double_similar(rng)
        except:
            return None
        return None

    def _draw_triangle(self, ax, pts, labels, sides, angles, color, offset_x=0):
        """Draw labeled triangle."""
        xs = [p[0] + offset_x for p in pts] + [pts[0][0] + offset_x]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, '-', color=color, linewidth=2)

        for i, (p, lbl) in enumerate(zip(pts, labels)):
            dx = (p[0] + offset_x - sum(pp[0] + offset_x for pp in pts) / 3) * 0.3
            dy = (p[1] - sum(pp[1] for pp in pts) / 3) * 0.3
            ax.annotate(lbl, xy=(p[0] + offset_x, p[1]),
                       xytext=(p[0] + offset_x + dx, p[1] + dy),
                       fontsize=12, fontweight="bold", color=color)

        # Side labels
        for i, (s_label, s_val) in enumerate(sides):
            j = (i + 1) % 3
            mx = (pts[i][0] + pts[j][0]) / 2 + offset_x
            my = (pts[i][1] + pts[j][1]) / 2
            ax.annotate(s_val, xy=(mx, my), fontsize=10, color=color,
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.8))

    def _identify(self, rng):
        """Two triangles shown with marks — identify congruence criterion."""
        criterion = rng.choice(["SSS", "SAS", "ASA", "AAS"])
        a = rng.randint(3, 8)
        b = rng.randint(3, 8)
        angle = rng.randint(30, 120)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        tri1 = [(0, 0), (a, 0), (b * 0.5, b * 0.7)]
        tri2 = [(0, 0), (a, 0), (b * 0.5, b * 0.7)]

        if criterion == "SSS":
            c = round(math.sqrt((tri1[2][0] - tri1[1][0])**2 + (tri1[2][1] - tri1[1][1])**2), 1)
            sides1 = [("AB", str(a)), ("BC", str(round(c, 1))), ("CA", str(b))]
            sides2 = [("DE", str(a)), ("EF", str(round(c, 1))), ("FD", str(b))]
            marks = "All three sides equal"
        elif criterion == "SAS":
            sides1 = [("AB", str(a)), ("", ""), ("CA", str(b))]
            sides2 = [("DE", str(a)), ("", ""), ("FD", str(b))]
            marks = f"Two sides equal + included angle {angle}°"
        elif criterion == "ASA":
            a2 = 180 - angle - rng.randint(30, 80)
            sides1 = [("AB", str(a)), ("", ""), ("", "")]
            sides2 = [("DE", str(a)), ("", ""), ("", "")]
            marks = f"Two angles ({angle}°, {a2}°) + included side"
        else:  # AAS
            a2 = rng.randint(30, 80)
            sides1 = [("", ""), ("BC", str(b)), ("", "")]
            sides2 = [("", ""), ("EF", str(b)), ("", "")]
            marks = f"Two angles ({angle}°, {a2}°) + non-included side"

        self._draw_triangle(ax, tri1, ["A", "B", "C"], sides1, [], palette[0], offset_x=0)
        self._draw_triangle(ax, tri2, ["D", "E", "F"], sides2, [], palette[1], offset_x=a + 2)

        ax.set_xlim(-1, 2 * a + 4)
        ax.set_ylim(-1, b * 0.7 + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Congruent Triangles — {marks}", fontsize=11, fontweight="bold")

        question = f"Based on the marked equal parts, which congruence criterion applies: SSS, SAS, ASA, or AAS?"
        return question, criterion, self.fig_to_pil(fig, dpi=style["dpi"])

    def _find_side(self, rng):
        """Given congruent triangles, find unknown side."""
        a = rng.randint(4, 10)
        b = rng.randint(4, 10)
        c = rng.randint(4, 10)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        tri1 = [(0, 0), (a, 0), (b * 0.4, b * 0.6)]
        sides1 = [("", str(a)), ("", str(c)), ("", str(b))]
        sides2 = [("", str(a)), ("", "x = ?"), ("", str(b))]

        self._draw_triangle(ax, tri1, ["A", "B", "C"], sides1, [], palette[0], offset_x=0)
        self._draw_triangle(ax, tri1, ["D", "E", "F"], sides2, [], palette[1], offset_x=a + 2)

        ax.text(a + 1, b * 0.6 + 0.5, "≅", fontsize=24, ha="center", va="center", color="gray")

        ax.set_xlim(-1, 2 * a + 4)
        ax.set_ylim(-1, b * 0.6 + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("△ABC ≅ △DEF — Find the unknown side", fontsize=12, fontweight="bold")

        question = (
            "△ABC ≅ △DEF as shown in the figure. The side labels are given "
            "in the image. Find x (the unknown side of △DEF).")
        return question, str(c), self.fig_to_pil(fig, dpi=style["dpi"])

    def _find_angle(self, rng):
        """Given congruent triangles, find unknown angle."""
        a1 = rng.randint(30, 80)
        a2 = rng.randint(30, 80)
        a3 = 180 - a1 - a2
        if a3 <= 10:
            return None

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        tri = [(0, 0), (5, 0), (2, 3)]

        self._draw_triangle(ax, tri, ["A", "B", "C"],
                          [("", ""), ("", ""), ("", "")], [], palette[0], offset_x=0)
        self._draw_triangle(ax, tri, ["D", "E", "F"],
                          [("", ""), ("", ""), ("", "")], [], palette[1], offset_x=9)

        ax.annotate(f"{a1}°", xy=(0.5, 0.3), fontsize=11, color=palette[0])
        ax.annotate(f"{a2}°", xy=(4.3, 0.3), fontsize=11, color=palette[0])
        ax.annotate(f"x=?", xy=(9.5 + 2, 2.5), fontsize=13, color="red", fontweight="bold")
        ax.annotate(f"{a1}°", xy=(9.5, 0.3), fontsize=11, color=palette[1])
        ax.annotate(f"{a2}°", xy=(9 + 4.3, 0.3), fontsize=11, color=palette[1])

        ax.text(7, 1.5, "≅", fontsize=24, ha="center", color="gray")

        ax.set_xlim(-1, 16); ax.set_ylim(-1, 4.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("△ABC ≅ △DEF — Find angle x", fontsize=12, fontweight="bold")

        question = (
            "△ABC ≅ △DEF as shown in the figure, with angles A and B "
            "labeled on △ABC. Find angle F (= angle C) in degrees.")
        return question, str(a3), self.fig_to_pil(fig, dpi=style["dpi"])

    def _similar_ratio(self, rng):
        """Similar triangles — find side using ratio."""
        ratio = rng.choice([2, 3, 1.5, 2.5])
        a = rng.randint(3, 8)
        b = rng.randint(3, 8)
        a2 = round(a * ratio, 1)
        b2 = round(b * ratio, 1)
        c = rng.randint(3, 8)
        c2 = round(c * ratio, 1)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]

        tri1 = [(0, 0), (a * 0.8, 0), (a * 0.3, a * 0.5)]
        tri2 = [(0, 0), (a2 * 0.8, 0), (a2 * 0.3, a2 * 0.5)]

        sides1 = [("", str(a)), ("", str(c)), ("", str(b))]
        sides2 = [("", str(a2)), ("", "x=?"), ("", str(b2))]

        self._draw_triangle(ax, tri1, ["P", "Q", "R"], sides1, [], palette[0], offset_x=0)
        self._draw_triangle(ax, tri2, ["X", "Y", "Z"], sides2, [], palette[1], offset_x=a + 3)

        ax.text(a + 1.5, a * 0.3, "~", fontsize=24, ha="center", color="gray")

        ax.set_xlim(-1, a + 3 + a2 + 1)
        ax.set_ylim(-1, max(a, a2) * 0.5 + 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Similar Triangles — Find x", fontsize=12, fontweight="bold")

        question = (
            f"△PQR ~ △XYZ with linear scale factor {ratio}, as shown in "
            "the figure with side lengths labeled. Find YZ (= x)."
        )
        return question, str(c2), self.fig_to_pil(fig, dpi=style["dpi"])

    def _similar_pythagoras(self, rng):
        """Similar triangles + Pythagorean theorem chain.
        Given similar ratio and one side of each, find the hypotenuse of the
        larger triangle using Pythagoras on the smaller then scaling."""
        # Small right triangle: legs a, b. Hyp c = sqrt(a^2+b^2).
        a = rng.choice([3, 5, 6, 8])
        b = rng.choice([4, 5, 8, 12])
        c = round(math.sqrt(a**2 + b**2), 2)
        ratio = rng.choice([2, 3, 1.5, 2.5])
        # Big triangle hypotenuse = c * ratio
        big_hyp = round(c * ratio, 2)
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        tri1 = [(0, 0), (a, 0), (0, b)]
        tri2 = [(0, 0), (a * ratio, 0), (0, b * ratio)]
        self._draw_triangle(ax, tri1, ["A", "B", "C"],
                          [("", str(a)), ("", "?"), ("", str(b))], [], palette[0], offset_x=0)
        self._draw_triangle(ax, tri2, ["D", "E", "F"],
                          [("", str(round(a * ratio, 1))), ("", "x=?"), ("", str(round(b * ratio, 1)))],
                          [], palette[1], offset_x=a + 3)
        ax.text(a + 1.5, b * 0.4, "~", fontsize=24, ha="center", color="gray")
        ax.set_xlim(-1, a + 3 + a * ratio + 1)
        ax.set_ylim(-1, max(b, b * ratio) + 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Similar Right Triangles — Find hypotenuse", fontsize=12, fontweight="bold")
        question = (f"△ABC ~ △DEF with linear scale factor {ratio}, as shown "
                    "in the figure. △ABC is a right triangle with legs "
                    "labeled on the image. Find the hypotenuse of △DEF. "
                    "Round to 2 decimals.")
        return question, str(big_hyp), self.fig_to_pil(fig, dpi=style["dpi"])

    def _similar_area_ratio(self, rng):
        """If linear ratio is k, area ratio is k^2. Given area of small and
        linear ratio, find area of large."""
        a_small = rng.randint(6, 30)
        k = rng.choice([2, 3, 4, 1.5, 2.5])
        a_large = round(a_small * k**2, 2)
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        tri1 = [(0, 0), (3, 0), (1.5, 2)]
        tri2 = [(0, 0), (3 * k, 0), (1.5 * k, 2 * k)]
        self._draw_triangle(ax, tri1, ["P", "Q", "R"], [("", ""), ("", ""), ("", "")], [], palette[0], offset_x=0)
        self._draw_triangle(ax, tri2, ["X", "Y", "Z"], [("", ""), ("", ""), ("", "")], [], palette[1], offset_x=5)
        ax.text(4, 1, "~", fontsize=24, ha="center", color="gray")
        ax.annotate(f"Area = {a_small}", xy=(1.2, 1.0), fontsize=10,
                     color=palette[0], ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec=palette[0], alpha=0.85))
        ax.annotate("Area = ?", xy=(5 + 1.5 * k, 1.0 * k), fontsize=12,
                     color="red", fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec="red", alpha=0.85))
        ax.set_xlim(-1, 5 + 3 * k + 1); ax.set_ylim(-1, max(2, 2 * k) + 1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Similar Triangles — Area Ratio", fontsize=12, fontweight="bold")
        question = (f"△PQR ~ △XYZ with linear scale factor {k}, as shown in "
                    "the figure with the area of △PQR labeled. Find the "
                    "area of △XYZ. (Area ratio = scale factor squared.) "
                    "Round to 2 decimals.")
        return question, str(a_large), self.fig_to_pil(fig, dpi=style["dpi"])

    def _double_similar(self, rng):
        """Chain of two similarity ratios. A ~ B with ratio k1, B ~ C with ratio k2.
        Given a side of A, find corresponding side of C."""
        k1 = rng.choice([2, 3, 1.5])
        k2 = rng.choice([2, 3, 1.5, 2.5])
        side_a = rng.randint(3, 8)
        side_c = round(side_a * k1 * k2, 2)
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(style["bg_color"])
        palette = style["palette"]
        tri = [(0, 0), (3, 0), (1, 2)]
        self._draw_triangle(ax, tri, ["A", "B", "C"],
                          [("", str(side_a)), ("", ""), ("", "")], [], palette[0], offset_x=0)
        ax.text(4, 1, f"~k1={k1}", fontsize=10, ha="center", color="gray")
        self._draw_triangle(ax, tri, ["D", "E", "F"],
                          [("", "?"), ("", ""), ("", "")], [], palette[1], offset_x=5)
        ax.text(9, 1, f"~k2={k2}", fontsize=10, ha="center", color="gray")
        self._draw_triangle(ax, tri, ["G", "H", "I"],
                          [("", "x=?"), ("", ""), ("", "")], [], palette[2], offset_x=10)
        ax.set_xlim(-1, 14); ax.set_ylim(-1, 3.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Chained Similarity: A ~ B ~ C", fontsize=12, fontweight="bold")
        question = (f"△ABC ~ △DEF with ratio {k1}, and △DEF ~ △GHI with "
                    f"ratio {k2}, as shown in the figure with AB labeled. "
                    "Find the corresponding side GH. Round to 2 decimals.")
        return question, str(side_c), self.fig_to_pil(fig, dpi=style["dpi"])
