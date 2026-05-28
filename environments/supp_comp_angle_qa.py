"""
Supplementary / Complementary Angle QA.

Two angles meet at a common vertex:
  - SUPPLEMENTARY case: two angles share a straight line. Given one,
    the other = 180 - given.
  - COMPLEMENTARY case: two angles share a right angle. Given one,
    the other = 90 - given.

The figure shows the vertex with two rays plus a third bounding line/ray
(the straight base or the perpendicular line), with one angle labeled
in degrees and the other marked with a "?".

Difficulty axes:
  - L0..L2: supplementary, given is a multiple of 10, between 30 and 150.
  - L3..L5: mix supplementary + complementary, given is a multiple of 5.
  - L6..L9: integer degrees not on multiples of 5; complementary up to 89.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_SUPP = [
    "The figure shows two angles that together form a straight line at the vertex. One angle is labeled. Find the other angle in degrees. Place the integer answer in <answer>...</answer>.",
    "Two angles share a vertex on a straight line; one is labeled, the other marked '?'. Compute the unknown angle (in degrees). Place the integer in <answer>...</answer>.",
    "At the vertex shown, two angles add up to a straight line (180 degrees). Given the labeled angle, find the unknown angle in degrees. Place the integer in <answer>...</answer>.",
    "The two angles in the figure are supplementary (they form a straight line). One is labeled; compute the other in degrees. Place the integer in <answer>...</answer>.",
    "From the figure, the two angles together make a straight line. Find the unknown angle in degrees. Place the integer in <answer>...</answer>.",
    "Two adjacent angles lie on a straight line. The labeled angle is shown; find the other one (in degrees). Place the integer in <answer>...</answer>.",
    "Given that the two angles in the figure are supplementary, compute the missing one in degrees. Place the integer in <answer>...</answer>.",
    "Find the angle marked '?' in the figure: it forms a straight line with the labeled angle. Answer in degrees, integer, in <answer>...</answer>.",
    "The labeled angle and the unknown angle make a straight line. Find the unknown in degrees and place the integer in <answer>...</answer>.",
]

_TEMPLATES_COMP = [
    "The figure shows two angles that together form a right angle (90 degrees) at the vertex. One angle is labeled. Find the other angle in degrees. Place the integer in <answer>...</answer>.",
    "Two angles share a vertex inside a right angle; one is labeled, the other marked '?'. Compute the unknown angle (in degrees). Place the integer in <answer>...</answer>.",
    "At the vertex shown, two angles add up to a right angle (90 degrees). Given the labeled angle, find the unknown angle in degrees. Place the integer in <answer>...</answer>.",
    "The two angles in the figure are complementary (they form a 90-degree corner). One is labeled; compute the other in degrees. Place the integer in <answer>...</answer>.",
    "From the figure, the two angles together make a right angle. Find the unknown angle in degrees. Place the integer in <answer>...</answer>.",
    "Two adjacent angles lie inside a right angle. The labeled angle is shown; find the other one (in degrees). Place the integer in <answer>...</answer>.",
    "Given that the two angles in the figure are complementary, compute the missing one in degrees. Place the integer in <answer>...</answer>.",
    "Find the angle marked '?' in the figure: it forms a 90-degree corner together with the labeled angle. Place the integer answer in <answer>...</answer>.",
]

# 2026-05-04 R4: full-gradient redesign per a math benchmark (supp+comp chain).
# COMPOUND mode (L6-L9): two-step angle chain.
# Forms (rendered as one labeled angle X°; model computes the chain answer):
#   "supp_of_comp"  : 180 - (90 - X) = 90 + X
#   "comp_of_supp"  : 90 - (180 - X)  → only valid when X >= 91 (so 90<=180-X<=89)
#                     equivalent: X - 90  for 91 <= X <= 179 (and result in 1..89)
#   "supp_minus_comp": (180 - X) - (90 - X) = 90  (constant; rejected)
# We use supp_of_comp and comp_of_supp.
_TEMPLATES_CHAIN = [
    "The figure labels an angle X = {given}°. Compute Y = supplement(complement(X)) = 180 - (90 - X). Place the integer Y in <answer>...</answer>.",
    "From the figure, X = {given}°. Find Y = the supplement of the complement of X (i.e. 180 − (90 − X)). Place the integer in <answer>...</answer>.",
    "Given X = {given}° (shown in the figure), compute the SUPPLEMENT of the COMPLEMENT of X. Place the integer in <answer>...</answer>.",
]
_TEMPLATES_CHAIN2 = [
    "The figure labels an angle X = {given}°. Compute Y = complement(supplement(X)) = 90 − (180 − X) = X − 90. Place the integer Y in <answer>...</answer>.",
    "From the figure, X = {given}°. Find Y = the complement of the supplement of X. Place the integer in <answer>...</answer>.",
    "Given X = {given}° (shown in the figure), compute the COMPLEMENT of the SUPPLEMENT of X. Place the integer in <answer>...</answer>.",
]


class SuppCompAngleQA(StandaloneVisualEnv):
    ENV_NAME = "supp_comp_angle"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per a math benchmark supp+comp chain.
        # L0-L1: SUPP only, multiples of 10 → trivial 1-step (180-X)
        # L2-L3: SUPP+COMP, multiples of 5
        # L4-L5: SUPP+COMP, integer non-5 (mid arithmetic)
        # L6-L7: COMPOUND chain "supp(comp(X)) = 90+X" (2-step, integer X 30-89)
        # L8-L9: COMPOUND chain "comp(supp(X)) = X-90" (X in 91..179) PLUS
        #        the L4-L5 mid-difficulty distribution to force model to
        #        recognize chain words like "supplement of complement".
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "modes": ["supp"],
                "given_pool": [30, 40, 50, 60, 70, 80, 100, 110, 120, 130, 150],
            }
        if level <= 3:
            return {
                "modes": ["supp", "comp"],
                "given_pool": [25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 145],
            }
        if level <= 5:
            return {
                "modes": ["supp", "comp"],
                "given_pool": [17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 67, 73,
                               79, 83, 89, 97, 103, 113, 127, 137, 149, 157,
                               163],
            }
        if level <= 7:
            return {
                "modes": ["chain_supp_of_comp"],
                "given_pool": [12, 17, 23, 28, 31, 37, 41, 43, 47, 53, 58,
                               61, 67, 73, 79, 83, 88],
            }
        # L8-L9: bigger chain mode mix
        return {
            "modes": ["chain_supp_of_comp", "chain_comp_of_supp"],
            "given_pool_supp_of_comp": [13, 19, 27, 33, 38, 44, 49, 56, 62,
                                        68, 71, 77, 81, 84, 87, 89],
            "given_pool_comp_of_supp": [91, 93, 97, 101, 107, 113, 119, 127,
                                        131, 137, 143, 151, 157, 163, 169,
                                        173, 177, 179],
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1303 + level * 89 + 23)

        for _ in range(40):
            mode = rng.choice(cfg["modes"])
            if mode == "chain_supp_of_comp":
                given = rng.choice(cfg.get("given_pool_supp_of_comp",
                                           cfg.get("given_pool", [])))
                if not (1 <= given <= 89):
                    continue
                # supp(comp(X)) = 90 + X
                other = 90 + given
                # Render the labeled angle X as a comp-arc (since it's
                # treated as the angle itself; the chain is purely textual).
                render_mode = "comp"
            elif mode == "chain_comp_of_supp":
                given = rng.choice(cfg.get("given_pool_comp_of_supp",
                                           cfg.get("given_pool", [])))
                if not (91 <= given <= 179):
                    continue
                # comp(supp(X)) = X - 90
                other = given - 90
                render_mode = "supp"
            elif mode == "supp":
                given = rng.choice(cfg["given_pool"])
                if not (10 <= given <= 170):
                    continue
                other = 180 - given
                render_mode = "supp"
            else:  # comp
                given = rng.choice(cfg["given_pool"])
                if not (5 <= given <= 85):
                    continue
                other = 90 - given
                render_mode = "comp"
            if other <= 0:
                continue
            break
        else:
            return None

        if mode == "chain_supp_of_comp":
            templates = _TEMPLATES_CHAIN
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx].format(given=given)
        elif mode == "chain_comp_of_supp":
            templates = _TEMPLATES_CHAIN2
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx].format(given=given)
        else:
            templates = _TEMPLATES_SUPP if mode == "supp" else _TEMPLATES_COMP
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx]
        answer = str(int(other))
        img = self._render(given, render_mode)
        return question, answer, img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render(self, given_deg, mode) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Vertex at origin.
        # mode="supp": straight line from (-1, 0) to (+1, 0) with a ray going
        #              up at angle = given (left half) and (180-given) right.
        #              Actually easier: ray goes up from origin at the
        #              boundary angle = given. The "given" angle is between
        #              the +x axis and the ray; the "?" angle is between the
        #              ray and the -x axis = 180 - given.
        # mode="comp": two rays inside a right angle. The right-angle base
        #              consists of the +x axis and the +y axis. Ray goes
        #              from origin at angle = given from +x axis. Given
        #              between +x and ray; ? between ray and +y = 90 - given.

        ray_len = 1.6
        # Convert given degrees to radians for the inner ray direction.
        ang_rad = math.radians(given_deg)
        rx = ray_len * math.cos(ang_rad)
        ry = ray_len * math.sin(ang_rad)

        edge_color = "#2c3e50"
        ray_color = "#1b4f72"
        arc_given = "#c0392b"
        arc_unknown = "#8e44ad"

        if mode == "supp":
            # Straight base: from (-ray_len, 0) to (+ray_len, 0).
            ax.plot([-ray_len, ray_len], [0, 0], color=edge_color, lw=2.0)
            # Inner ray from origin upward.
            ax.plot([0, rx], [0, ry], color=ray_color, lw=2.0)
            # "Given" arc: from +x axis (angle 0) to the inner ray.
            given_arc = patches.Arc(
                (0, 0), 0.7, 0.7, angle=0, theta1=0, theta2=given_deg,
                color=arc_given, lw=2.0)
            ax.add_patch(given_arc)
            # "?" arc: from inner ray to -x axis (180).
            unknown_arc = patches.Arc(
                (0, 0), 0.55, 0.55, angle=0, theta1=given_deg, theta2=180,
                color=arc_unknown, lw=2.0)
            ax.add_patch(unknown_arc)

            # Given label position: midway in given arc, slightly outside.
            given_mid = math.radians(given_deg / 2)
            ax.text(0.55 * math.cos(given_mid), 0.55 * math.sin(given_mid),
                    f"{given_deg}°",
                    ha="center", va="center", fontsize=13, color=arc_given,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="#fff", edgecolor=arc_given, lw=0.8))
            # ? label position
            unknown_mid = math.radians((given_deg + 180) / 2)
            ax.text(0.7 * math.cos(unknown_mid), 0.7 * math.sin(unknown_mid),
                    "?",
                    ha="center", va="center", fontsize=15, color=arc_unknown,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#fff", edgecolor=arc_unknown, lw=0.8))

            # Endpoint markers (small dots / arrows)
            ax.plot([-ray_len], [0], 'k>', markersize=8)
            ax.plot([ray_len], [0], 'k<', markersize=8)
            ax.plot([rx], [ry], 'ko', markersize=5)
            ax.text(-ray_len - 0.1, -0.15, "A", ha="right", va="top",
                    fontsize=11, color=edge_color, fontweight="bold")
            ax.text(ray_len + 0.1, -0.15, "B", ha="left", va="top",
                    fontsize=11, color=edge_color, fontweight="bold")
            ax.text(rx + 0.1, ry + 0.05, "C", ha="left", va="bottom",
                    fontsize=11, color=ray_color, fontweight="bold")
            ax.text(0, -0.35, "O", ha="center", va="top", fontsize=11,
                    color=edge_color, fontweight="bold")

            ax.set_xlim(-ray_len - 0.7, ray_len + 0.7)
            ax.set_ylim(-1.0, max(ry + 0.6, 1.4))

            ax.text(0, max(ry + 0.6, 1.3) + 0.05,
                    "Two angles together form a straight line.",
                    ha="center", va="bottom", fontsize=11,
                    color="#5d4037", fontstyle="italic")

        else:  # comp
            # Right angle base: +x axis from (0,0) to (ray_len, 0)
            #                  +y axis from (0,0) to (0, ray_len)
            ax.plot([0, ray_len], [0, 0], color=edge_color, lw=2.0)
            ax.plot([0, 0], [0, ray_len], color=edge_color, lw=2.0)
            # Right-angle marker (small square at origin)
            sq_size = 0.12
            ax.add_patch(patches.Rectangle(
                (0, 0), sq_size, sq_size, fill=False,
                edgecolor=edge_color, lw=1.4))
            # Inner ray from origin at angle = given from +x axis.
            ax.plot([0, rx], [0, ry], color=ray_color, lw=2.0)
            # "Given" arc: 0 to given.
            given_arc = patches.Arc(
                (0, 0), 0.7, 0.7, angle=0, theta1=0, theta2=given_deg,
                color=arc_given, lw=2.0)
            ax.add_patch(given_arc)
            # "?" arc: given to 90.
            unknown_arc = patches.Arc(
                (0, 0), 0.55, 0.55, angle=0, theta1=given_deg, theta2=90,
                color=arc_unknown, lw=2.0)
            ax.add_patch(unknown_arc)
            given_mid = math.radians(given_deg / 2)
            ax.text(0.55 * math.cos(given_mid), 0.55 * math.sin(given_mid),
                    f"{given_deg}°",
                    ha="center", va="center", fontsize=13, color=arc_given,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="#fff", edgecolor=arc_given, lw=0.8))
            unknown_mid = math.radians((given_deg + 90) / 2)
            ax.text(0.7 * math.cos(unknown_mid), 0.7 * math.sin(unknown_mid),
                    "?",
                    ha="center", va="center", fontsize=15, color=arc_unknown,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#fff", edgecolor=arc_unknown, lw=0.8))

            ax.plot([ray_len], [0], 'k<', markersize=8)
            ax.plot([0], [ray_len], 'kv', markersize=8)
            ax.plot([rx], [ry], 'ko', markersize=5)
            ax.text(ray_len + 0.1, -0.05, "A", ha="left", va="top",
                    fontsize=11, color=edge_color, fontweight="bold")
            ax.text(-0.1, ray_len + 0.1, "B", ha="right", va="bottom",
                    fontsize=11, color=edge_color, fontweight="bold")
            ax.text(rx + 0.1, ry + 0.05, "C", ha="left", va="bottom",
                    fontsize=11, color=ray_color, fontweight="bold")
            ax.text(-0.18, -0.18, "O", ha="right", va="top", fontsize=11,
                    color=edge_color, fontweight="bold")

            ax.set_xlim(-0.7, ray_len + 0.6)
            ax.set_ylim(-0.5, ray_len + 0.7)

            ax.text(ray_len / 2, ray_len + 0.5,
                    "Two angles together form a right angle.",
                    ha="center", va="bottom", fontsize=11,
                    color="#5d4037", fontstyle="italic")

        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
