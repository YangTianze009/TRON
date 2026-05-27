"""
Multi-angle chasing QA — angle chain reasoning with progressive difficulty.

Targets: image-only angle reasoning, geometry reasoning.
Capabilities: V1 (shape recognition), V2 (label reading), R2 (geometric theorems)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 1-step. Find the missing angle in a triangle given the other 2 angles
    (or 4th angle in a quadrilateral). Trivial — just sum subtraction.
L1: 1-step. Linear pair (180° - given) or vertical angles (= given).
L2: 1-step. Parallel lines + transversal: corresponding / alternate / co-interior.
L3: 2-step. Triangle exterior angle = sum of remote interior angles.
L4: 2-step. Parallel lines with triangle (one alternate interior + triangle sum).
L5: 2-step. Isosceles triangle + parallel line (base angles + alternate interior).
L6: 3-step. Polygon with parallel sides — combine corresponding angles + polygon sum.
L7: 3-step. Z-angle chain with extra split (alternate interior + triangle exterior).
L8: 4-step. Two parallel lines with two transversals + extra triangle.
L9: 4-step. Composite: parallel lines + multiple triangles + polygon sum.

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = [
    "Angle Chase",
    "Find the angle",
    "Geometry: angle reasoning",
    "Angles in figure",
    "Angle problem",
]

class AngleChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "angle_chain"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(20):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _dispatch(self, level: int):
        rng = self._rng
        sub_rng = self._sub_rng(level)

        if level == 0:
            return self._triangle_missing(sub_rng)
        if level == 1:
            mode = sub_rng.choice(["linear_pair", "vertical_angles"])
            return self._one_step_pair(sub_rng, mode)
        if level == 2:
            return self._parallel_single(sub_rng)
        if level == 3:
            return self._triangle_exterior_chain(sub_rng)
        if level == 4:
            return self._parallel_with_triangle(sub_rng)
        if level == 5:
            return self._isosceles_parallel(sub_rng)
        if level == 6:
            return self._polygon_interior_chain(sub_rng, n_known=4)
        if level == 7:
            return self._z_angle_chain(sub_rng)
        if level == 8:
            return self._parallel_double(sub_rng)
        return self._composite_hard(sub_rng)

    # ------------------------------------------------------------------ #
    # L0 — 1-step triangle sum
    # ------------------------------------------------------------------ #
    def _triangle_missing(self, rng):
        """L0: triangle with two known angles, find third = 180 - a - b."""
        a = rng.randint(30, 80)
        b = rng.randint(30, 80)
        third = 180 - a - b
        if third <= 10 or third >= 170:
            return None

        fig, ax = plt.subplots(figsize=(6.5, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#1d3557", "#2c3e50", "#0d6efd", "#198754"])
        lw = rng.choice([1.5, 1.8, 2.0, 2.5])

        # Random triangle position
        offset_x = rng.uniform(-0.3, 0.3)
        offset_y = rng.uniform(-0.2, 0.2)
        A = (0 + offset_x, 0 + offset_y)
        B = (5 + offset_x, 0 + offset_y)
        cx = rng.uniform(1.5, 3.5)
        cy = rng.uniform(2.5, 3.5)
        C = (cx + offset_x, cy + offset_y)

        for p1, p2 in [(A, B), (B, C), (C, A)]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge, linewidth=lw)

        ax.annotate(f"{a}°", xy=(A[0] + 0.4, A[1] + 0.3), fontsize=14,
                    color="#2e7d32", fontweight="bold")
        ax.annotate(f"{b}°", xy=(B[0] - 1.0, B[1] + 0.3), fontsize=14,
                    color="#2e7d32", fontweight="bold")
        ax.annotate("x = ?", xy=(C[0] - 0.3, C[1] + 0.2), fontsize=14,
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#c0392b"))

        ax.set_xlim(-1.5, 6.5)
        ax.set_ylim(-1, 4.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        stems = [
            "In the triangle shown, two interior angles are labeled and the "
            "third is marked x. Find the third interior angle x (in degrees).",
            "A triangle has three interior angles. Two of them are labeled in "
            "the figure and the third is marked x. What is x?",
            "Read the two labeled interior angles from the figure and compute "
            "the remaining interior angle x of the triangle.",
        ]
        return rng.choice(stems), str(third), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L1 — 1-step linear pair / vertical angles
    # ------------------------------------------------------------------ #
    def _one_step_pair(self, rng, mode):
        given = rng.randint(25, 155)
        if mode == "linear_pair":
            answer = 180 - given
        else:
            answer = given

        fig, ax = plt.subplots(figsize=(6, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        if mode == "linear_pair":
            # one straight horizontal line + ray
            ax.plot([-3, 3], [0, 0], color=edge, linewidth=2.2)
            theta = math.radians(given)
            ax.plot([0, 2 * math.cos(theta)],
                    [0, 2 * math.sin(theta)], color=edge, linewidth=2.2)
            ax.annotate(f"{given}°", xy=(0.3, 0.4), fontsize=13,
                        color="#2e7d32", fontweight="bold")
            ax.annotate("x = ?", xy=(-1.3, 0.3), fontsize=14,
                        color="#c0392b", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="lightyellow", ec="#c0392b"))
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-1.5, 2.5)
            stems = [
                "Two rays form a straight line in the figure. One of the "
                "two angles is labeled in the image. What is the "
                "supplementary angle x (in degrees)?",
                "In the diagram, the marked angle lies on a straight line. "
                "Using the labeled angle, find x (the angle adjacent to it "
                "on the same line), in degrees.",
            ]
        else:  # vertical_angles
            theta = math.radians(rng.randint(20, 70))
            ax.plot([-2.5 * math.cos(theta), 2.5 * math.cos(theta)],
                    [-2.5 * math.sin(theta), 2.5 * math.sin(theta)],
                    color=edge, linewidth=2.2)
            theta2 = math.radians(rng.randint(110, 160))
            ax.plot([-2.5 * math.cos(theta2), 2.5 * math.cos(theta2)],
                    [-2.5 * math.sin(theta2), 2.5 * math.sin(theta2)],
                    color=edge, linewidth=2.2)
            ax.annotate(f"{given}°", xy=(0.3, 0.6), fontsize=13,
                        color="#2e7d32", fontweight="bold")
            ax.annotate("x = ?", xy=(-1.0, -0.8), fontsize=14,
                        color="#c0392b", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="lightyellow", ec="#c0392b"))
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            stems = [
                "Two straight lines intersect as shown in the figure, with "
                "one angle labeled. Find the vertical (opposite) angle x, "
                "in degrees.",
                "At the intersection of two lines, one angle is labeled in "
                "the figure. What is the angle directly opposite (vertical "
                "to) it, in degrees?",
            ]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        return rng.choice(stems), str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L2 — 1-step parallel lines + transversal
    # ------------------------------------------------------------------ #
    def _parallel_single(self, rng):
        given_angle = rng.randint(30, 150)
        rel = rng.choice(["corresponding", "alternate", "co_interior"])
        if rel == "corresponding":
            answer = given_angle
            rel_text = "corresponding"
        elif rel == "alternate":
            answer = given_angle
            rel_text = "alternate interior"
        else:
            answer = 180 - given_angle
            rel_text = "co-interior"

        fig, ax = plt.subplots(figsize=(7, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        ax.plot([-3, 5], [2, 2], color=edge, linewidth=2)
        ax.plot([-3, 5], [-1, -1], color=edge, linewidth=2)
        slope = math.tan(math.radians(given_angle))
        if abs(slope) < 0.2:
            slope = 0.2
        x_int = 1
        ax.plot([x_int - 3 / slope, x_int + 3 / slope],
                [-1 - 3, -1 + 3], color="#c0392b", linewidth=1.6)

        ax.annotate(f"{given_angle}°", xy=(x_int - 0.5, -1 + 0.3),
                    fontsize=14, color="#2e7d32", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#2e7d32"))
        ax.annotate("x = ?", xy=(x_int + 0.3, 2 + 0.3),
                    fontsize=14, color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#c0392b"))
        ax.annotate("l₁ ∥ l₂", xy=(4, 0.5), fontsize=12, color="#1976d2",
                    bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))

        ax.set_xlim(-3, 5.5)
        ax.set_ylim(-3, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        stems = [
            f"In the figure, l₁ ∥ l₂ are cut by a transversal. x is the "
            f"{rel_text} angle of the labeled angle. Read the labeled angle "
            f"from the image and find x in degrees.",
            f"Two parallel lines are crossed by a transversal. Using the "
            f"labeled {rel_text} angle in the image, find x (in degrees).",
        ]
        return rng.choice(stems), str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L3 — 2-step exterior angle of triangle
    # ------------------------------------------------------------------ #
    def _triangle_exterior_chain(self, rng):
        a = rng.randint(30, 80)
        b = rng.randint(30, 80)
        exterior = a + b
        if exterior >= 180:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        A = (0, 0)
        B = (5, 0)
        cx = rng.uniform(1.5, 3.0)
        cy = rng.uniform(2.8, 3.8)
        C = (cx, cy)

        for p1, p2 in [(A, B), (B, C), (C, A)]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge, linewidth=2)
        ax.plot([B[0], B[0] + 2], [B[1], B[1]], color=edge, linewidth=1.5,
                linestyle="--")

        ax.annotate(f"{a}°", xy=(0.5, 0.3), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate(f"{b}°", xy=(C[0] - 0.3, C[1] - 0.7), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate("x = ?", xy=(5.3, 0.3), fontsize=14,
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#c0392b"))

        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        stems = [
            "In the triangle shown, two interior angles are labeled in the "
            "figure. The side AB is extended past B. Find the exterior "
            "angle x at vertex B (in degrees).",
            "A triangle has two labeled interior angles shown in the figure. "
            "Find the exterior angle x at the third vertex (formed by "
            "extending one side).",
        ]
        return rng.choice(stems), str(exterior), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L4 — 2-step parallel + triangle (alt int + triangle sum)
    # ------------------------------------------------------------------ #
    def _parallel_with_triangle(self, rng):
        base_a = rng.randint(35, 70)
        base_b = rng.randint(35, 70)
        apex = 180 - base_a - base_b
        if apex <= 15 or apex >= 130:
            return None
        # x = base_a (alternate interior with apex angle on the other side)
        answer = base_a

        fig, ax = plt.subplots(figsize=(7, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        ax.plot([-2, 6], [0, 0], color=edge, linewidth=2)
        ax.plot([-2, 6], [3, 3], color=edge, linewidth=2)
        ax.plot([1, 4], [0, 0], color="#c0392b", linewidth=2)
        ax.plot([1, 2.5], [0, 3], color="#c0392b", linewidth=2)
        ax.plot([4, 2.5], [0, 3], color="#c0392b", linewidth=2)

        ax.annotate(f"{base_a}°", xy=(1.3, 0.4), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate(f"{base_b}°", xy=(3.3, 0.4), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate("x = ?", xy=(2.8, 3.2), fontsize=14,
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#c0392b"))
        ax.annotate("l₁ ∥ l₂", xy=(5, 1.5), fontsize=12, color="#1976d2",
                    bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))

        ax.set_xlim(-2.5, 7)
        ax.set_ylim(-1, 4.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        question = (
            "In the figure, l\u2081 \u2225 l\u2082. A triangle has its base "
            "on l\u2081 with two labeled base angles, and apex on l\u2082. "
            "Find angle x at the apex, on the side of the LEFT base angle, "
            "in degrees."
        )
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L5 — 2-step isosceles + parallel
    # ------------------------------------------------------------------ #
    def _isosceles_parallel(self, rng):
        base_angle = rng.randint(35, 75)
        apex = 180 - 2 * base_angle
        # x is base_angle (alternate interior with the parallel line)
        answer = base_angle

        fig, ax = plt.subplots(figsize=(7, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        ax.plot([0, 5], [0, 0], color=edge, linewidth=2)
        ax.plot([0, 2.5], [0, 4], color=edge, linewidth=2)
        ax.plot([5, 2.5], [0, 4], color=edge, linewidth=2)
        ax.plot([0.5, 4.5], [2, 2], color="#c0392b", linewidth=1.5, linestyle="--")

        ax.annotate(f"{apex}°", xy=(2.2, 3.3), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate(f"{base_angle}°", xy=(0.4, 0.3), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate("x = ?", xy=(0.7, 1.7), fontsize=14,
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#c0392b"))
        ax.annotate("∥ base", xy=(3.5, 2.2), fontsize=10, color="#c0392b")

        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        question = (
            "An isosceles triangle is shown with its apex angle and base "
            "angles labeled in the figure. A line through the interior "
            "parallel to the base intersects the left side. Find angle x "
            "between this parallel line and the left side (in degrees)."
        )
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L6 — 3-step polygon
    # ------------------------------------------------------------------ #
    def _polygon_interior_chain(self, rng, n_known=4):
        # quadrilateral with 3 known
        angles = []
        for _ in range(3):
            angles.append(rng.randint(60, 120))
        fourth = 360 - sum(angles)
        if fourth <= 30 or fourth >= 170:
            return None

        fig, ax = plt.subplots(figsize=(6.5, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        pts = [(0, 0), (4, 0.5), (3.5, 3), (0.5, 3.5)]
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            ax.plot([x1, x2], [y1, y2], color=edge, linewidth=2)

        labels = [f"{angles[0]}°", f"{angles[1]}°", f"{angles[2]}°", "x = ?"]
        offsets = [(-0.5, -0.3), (4.3, 0.3), (3.8, 3.3), (-0.3, 3.5)]
        colors = ["#2e7d32", "#2e7d32", "#2e7d32", "#c0392b"]
        for lbl, off, col in zip(labels, offsets, colors):
            ax.annotate(lbl, xy=off, fontsize=13, color=col, fontweight="bold")

        ax.set_xlim(-1.5, 5.5)
        ax.set_ylim(-1, 5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        question = (
            "In the quadrilateral shown, three interior angles are labeled "
            "in the figure and the fourth is marked x. The interior angles "
            "of a quadrilateral sum to 360\u00b0. Find angle x (in degrees)."
        )
        return question, str(fourth), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L7 — 3-step Z-angle chain
    # ------------------------------------------------------------------ #
    def _z_angle_chain(self, rng):
        angle_a = rng.randint(25, 70)
        angle_b = rng.randint(25, 70)
        answer = angle_a + angle_b
        if answer >= 180:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        ax.plot([-2, 6], [3, 3], color=edge, linewidth=2)
        ax.plot([-2, 6], [0, 0], color=edge, linewidth=2)
        ax.plot([1, 4], [3, 0], color="#c0392b", linewidth=2)
        mid = (2.5, 1.5)
        ax.plot([mid[0], mid[0] + 2], [mid[1], mid[1] + 1.5],
                color="#e67e22", linewidth=1.5)

        ax.annotate(f"{angle_a}°", xy=(1.3, 2.5), fontsize=13,
                    color="#2e7d32", fontweight="bold")
        ax.annotate(f"{angle_b}°", xy=(2.8, 1.8), fontsize=13,
                    color="#e67e22", fontweight="bold")
        ax.annotate("x = ?", xy=(3.8, 0.3), fontsize=14,
                    color="#c0392b", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#c0392b"))
        ax.annotate("l₁ ∥ l₂", xy=(5, 1.5), fontsize=12, color="#1976d2",
                    bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))

        ax.set_xlim(-2.5, 7)
        ax.set_ylim(-1, 4.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        question = (
            "In the figure, l\u2081 \u2225 l\u2082. A Z-shaped line "
            "connects them with an additional branch creating two labeled "
            "angles (one between branch and l\u2081, one between the "
            "Z-line and the additional branch). Using the labeled angles, "
            "find angle x where the Z meets l\u2082, in degrees."
        )
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L8 — 4-step double-transversal
    # ------------------------------------------------------------------ #
    def _parallel_double(self, rng):
        a1 = rng.randint(30, 75)
        a2 = rng.randint(30, 75)
        answer = 180 - a1 - a2
        if answer <= 10 or answer >= 170:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        ax.plot([-3, 6], [2, 2], color=edge, linewidth=2)
        ax.plot([-3, 6], [-1, -1], color=edge, linewidth=2)
        ax.plot([0, 3], [-1, 2], color="#c0392b", linewidth=1.5)
        ax.plot([2, 5], [-1, 2], color="#e67e22", linewidth=1.5)

        ax.annotate(f"{a1}°", xy=(0.5, -0.5), fontsize=13,
                    color="#c0392b", fontweight="bold")
        ax.annotate(f"{a2}°", xy=(2.3, -0.5), fontsize=13,
                    color="#e67e22", fontweight="bold")
        ax.annotate("x = ?", xy=(2.5, 2.3), fontsize=14,
                    color="#2e7d32", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#2e7d32"))
        ax.annotate("l₁ ∥ l₂", xy=(5, 0.5), fontsize=12, color="#1976d2",
                    bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))

        ax.set_xlim(-3, 7)
        ax.set_ylim(-2.5, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        question = (
            "In the figure, l\u2081 \u2225 l\u2082 with two transversals "
            "forming a triangle between them. The two angles where the "
            "transversals meet l\u2081 are labeled in the figure. Using "
            "those labeled angles, find angle x where the two transversals "
            "meet at l\u2082."
        )
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # L9 — 4-step composite
    # ------------------------------------------------------------------ #
    def _composite_hard(self, rng):
        # combination: parallel lines + triangle + extension
        a1 = rng.randint(35, 75)
        a2 = rng.randint(35, 75)
        # angle x = a1 + a2 (exterior angle from inner triangle)
        answer = a1 + a2
        if answer >= 175:
            return None

        fig, ax = plt.subplots(figsize=(7, 6))
        style = self._random_style()
        self._apply_style(fig, ax, style)
        edge = rng.choice(["#1f3a93", "#2c3e50", "#0d6efd"])

        ax.plot([-2, 6], [0, 0], color=edge, linewidth=2)
        ax.plot([-2, 6], [4, 4], color=edge, linewidth=2)
        ax.plot([0, 4], [0, 4], color="#c0392b", linewidth=2)  # transversal 1
        ax.plot([2, 5], [0, 4], color="#e67e22", linewidth=2)  # transversal 2
        ax.plot([5, 6.5], [4, 4], color="#c0392b", linewidth=1.5, linestyle="--")  # extension

        ax.annotate(f"{a1}°", xy=(0.5, 0.4), fontsize=13,
                    color="#c0392b", fontweight="bold")
        ax.annotate(f"{a2}°", xy=(2.3, 0.4), fontsize=13,
                    color="#e67e22", fontweight="bold")
        ax.annotate("x = ?", xy=(5.0, 4.2), fontsize=14,
                    color="#2e7d32", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="lightyellow", ec="#2e7d32"))
        ax.annotate("l₁ ∥ l₂", xy=(5, 2.0), fontsize=12, color="#1976d2",
                    bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))

        ax.set_xlim(-2.5, 7.5)
        ax.set_ylim(-1, 5.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=10)

        question = (
            "In the figure, l\u2081 \u2225 l\u2082. Two transversals form "
            "a triangle between the lines. The two angles at l\u2081 are "
            "labeled in the figure. The right transversal meets l\u2082 "
            "and is extended (dashed). Using the labeled angles, find the "
            "exterior angle x at l\u2082 on the side of the extension."
        )
        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = AngleChainQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)} dist={dict(list(gt.items())[:8])}")
