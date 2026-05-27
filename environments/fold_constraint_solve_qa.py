"""
Fold Constraint Solve QA.

A rectangle ABCD (A top-left, B top-right, C bottom-right, D bottom-left)
is folded so vertex A lands on a point E on the side BC. The crease is the
line segment from D to a point F on side AB. The fold means triangle ADF
maps to triangle ED'F (with D'=D, F=F, A → E). Two key invariants from the
fold:
  - DE = DA  (because A maps to E and D is fixed)
  - FE = FA  (because A maps to E and F is fixed)

L0..L2 (TRIVIAL warm-up): rectangle is folded along its midline so AB lands
on CD; the fold halves the height. Trivially: half of AB. Answer is always
half of the labeled side AB.

L3..L5 (Pythagorean fold): given AB = c (CD = c since rectangle has equal
horizontal sides), AD = b (BC = b), and after the fold A lands on E on BC
with BE = e (labeled), find AF (the crease end on AB).
  Setup: in right triangle BEF, BE = e, BF = c - AF, FE = FA = AF.
  Pythagoras: (c - AF)^2 + e^2 = AF^2
  Expand: c^2 - 2c*AF + AF^2 + e^2 = AF^2
  Solve: AF = (c^2 + e^2) / (2c)
  Numbers are picked so AF is a clean rational (often half-integer).

L6..L9 (harder Pythagorean fold): same as above with messier numbers,
answer to one decimal.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_MIDFOLD = [
    "The rectangle ABCD shown is folded along the horizontal midline so that side AB lands on side DC. After folding, the resulting (smaller) rectangle has the same width AB but half the original height. Given AB = {c}, what is the length of the crease line (which is parallel to AB and bisects the height)? Place the numeric answer in <answer>...</answer>.",
    "The figure shows rectangle ABCD folded along its horizontal midline. Given AB = {c}, find the length of the crease line. Place the answer in <answer>...</answer>.",
    "Rectangle ABCD is folded along the horizontal line that goes through the midpoints of AD and BC, so that side AB coincides with side DC. The crease is parallel to AB. Given AB = {c}, find the crease length. Place the value in <answer>...</answer>.",
    "The rectangle in the figure is folded along its midline (parallel to AB and DC) so that AB lands on DC. The crease is parallel to AB. With AB = {c}, find the crease's length. Place the answer in <answer>...</answer>.",
    "Rectangle ABCD with AB = {c} is folded so the top edge AB lands exactly on the bottom edge DC. The crease is the horizontal midline. Compute its length. Place the value in <answer>...</answer>.",
    "Fold rectangle ABCD along the horizontal midline so AB and DC coincide. The crease is parallel to AB. Given AB = {c}, what is the crease's length? Place the answer in <answer>...</answer>.",
    "ABCD is a rectangle with AB = {c}. It is folded so the top side AB falls onto the bottom side DC. Find the length of the crease. Place the answer in <answer>...</answer>.",
    "The rectangle is folded so AB lands on DC (horizontal midline fold). Given AB = {c}, what is the length of the crease line? Place the value in <answer>...</answer>.",
    "Fold rectangle ABCD horizontally so the top edge meets the bottom edge. With AB = {c}, find the crease length. Place the answer in <answer>...</answer>.",
]

_TEMPLATES_PYTHAGOREAN = [
    "Rectangle ABCD has AB = {c} and AD = {b}. It is folded along a crease DF (with F on AB) so that vertex A lands on point E on side BC, and the labeled distance BE = {e}. Find the length of AF. Place the numeric answer in <answer>...</answer>.",
    "In rectangle ABCD, AB = {c}, AD = {b}. The corner A is folded over a crease passing through D and a point F on AB; after folding, A lands at E on BC with BE = {e}. Compute AF. Place the value in <answer>...</answer>.",
    "Rectangle ABCD: AB = {c}, AD = {b}. Folding along DF (F on AB) sends vertex A to point E on BC. Given BE = {e}, find AF. Place the numeric answer in <answer>...</answer>.",
    "ABCD is a rectangle with AB = {c} and AD = {b}. It is folded along the crease DF (F on AB) so A maps to E on BC with BE = {e}. Compute the length AF. Place the answer in <answer>...</answer>.",
    "In rectangle ABCD (AB = {c}, AD = {b}), a fold along crease DF (F on side AB) maps vertex A onto a point E on side BC, with BE = {e}. Find AF. Place the value in <answer>...</answer>.",
    "Rectangle ABCD, AB = {c}, AD = {b}, is folded so that corner A lands on side BC at point E with BE = {e}. The crease DF intersects AB at F. Find AF. Place the answer in <answer>...</answer>.",
    "Given rectangle ABCD with AB = {c} and AD = {b}, fold along crease DF so vertex A goes to point E on BC, and BE = {e}. Compute AF. Place the numeric answer in <answer>...</answer>.",
    "Rectangle ABCD: AB = {c}, AD = {b}. After folding along DF (F is on AB), A coincides with E on BC, where BE = {e}. Find AF. Place the value in <answer>...</answer>.",
]


class FoldConstraintSolveQA(StandaloneVisualEnv):
    ENV_NAME = "fold_constraint_solve"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "mode": "midfold",
                "c_pool": [4, 6, 8, 10, 12],
                "b_pool": [4, 6, 8, 10],  # only used for figure aspect
            }
        if level <= 5:
            # Pythagorean triples / cleanly soluble: pick c and e so that
            # (c^2 + e^2) is even AND divisible by 2c → AF is a clean
            # fraction (typically a half-integer).
            # Standard 3-4-5 type: c=8, e=4 → AF = (64+16)/16 = 5 ✓
            #                       c=8, e=6 → AF = (64+36)/16 = 6.25 (one decimal)
            #                       c=6, e=3 → AF = (36+9)/12 = 3.75
            # Pick (c, b, e) trios where AF is integer or .5.
            return {
                "mode": "pyth",
                # (c=AB, b=AD, e=BE): AF = (c^2+e^2)/(2c)
                "trios": [
                    (8, 6, 4),    # AF = (64+16)/16 = 5
                    (10, 8, 6),   # AF = (100+36)/20 = 6.8 → reject
                    (10, 7, 4),   # AF = (100+16)/20 = 5.8 → reject
                    (8, 5, 2),    # AF = (64+4)/16 = 4.25 → reject (need integer/.5)
                    (6, 4, 2),    # AF = (36+4)/12 = 10/3 ≈ 3.333 → reject
                    (4, 3, 2),    # AF = (16+4)/8 = 2.5 ✓
                    (10, 8, 4),   # AF = (100+16)/20 = 5.8 → reject
                    (8, 5, 6),    # AF = (64+36)/16 = 6.25 (1 dp)
                    (12, 8, 4),   # AF = (144+16)/24 = 6.667 → reject
                    (8, 6, 2),    # AF = (64+4)/16 = 4.25 (2 dp) → reject
                    (10, 6, 2),   # AF = (100+4)/20 = 5.2 (1 dp)
                    (12, 8, 6),   # AF = (144+36)/24 = 7.5 ✓
                    (5, 4, 1),    # AF = (25+1)/10 = 2.6 (1 dp)
                    (8, 6, 8),    # invalid: AF would > c, reject in solver
                ],
                # Acceptable AF must satisfy: 0 < AF <= c, and AF is integer or .5 step.
                "decimal_places": 1,
            }
        # Higher levels: allow 1-decimal answers.
        return {
            "mode": "pyth_hard",
            "trios": [
                (10, 8, 6),   # AF = 6.8
                (12, 9, 6),   # AF = (144+36)/24 = 7.5
                (10, 6, 2),   # AF = 5.2
                (12, 9, 8),   # AF = (144+64)/24 = 8.667 → reject
                (8, 5, 6),    # AF = 6.25 → reject (2 dp)
                (15, 10, 6),  # AF = (225+36)/30 = 8.7
                (10, 7, 4),   # AF = 5.8
                (12, 7, 5),   # AF = (144+25)/24 = 7.04 → reject (2 dp)
                (15, 12, 9),  # AF = (225+81)/30 = 10.2
                (20, 15, 12), # AF = (400+144)/40 = 13.6
                (12, 9, 4),   # AF = (144+16)/24 = 6.667 → reject
                (8, 6, 4),    # AF = (64+16)/16 = 5
                (10, 8, 8),   # AF = (100+64)/20 = 8.2
                (15, 10, 12), # AF = (225+144)/30 = 12.3
            ],
            "decimal_places": 1,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1597 + level * 113 + 37)

        if cfg["mode"] == "midfold":
            c = rng.choice(cfg["c_pool"])
            b = rng.choice(cfg["b_pool"])
            # Crease length = AB (since the fold is parallel to AB across
            # the rectangle, the crease spans the entire width).
            answer_val = c
            answer = str(int(c))
            templates = _TEMPLATES_MIDFOLD
            sidx = (self.seed or 0) % len(templates)
            question = templates[sidx].format(c=c)
            # 2026-05-04: simplified L0 (was 15% too-hard) — concise key insight.
            if level <= 1:
                question += (
                    " Be concise. The crease for a horizontal midline fold is "
                    "parallel to AB and spans the full width of the rectangle, "
                    "so its length equals the length of AB. Output only the "
                    "integer."
                )
            img = self._render_midfold(c, b)
            return question, answer, img

        # Pythagorean fold mode.
        for _ in range(40):
            c, b, e = rng.choice(cfg["trios"])
            # Constraint: BE <= BC = b, otherwise E isn't on BC.
            if e > b:
                continue
            # Compute AF = (c^2 + e^2) / (2c)
            af = (c * c + e * e) / (2 * c)
            # Need AF strictly less than c (else F coincides with B, which
            # is a degenerate fold).
            if af <= 0 or af >= c:
                continue
            # Check answer is integer or one-decimal.
            af_round = round(af, cfg.get("decimal_places", 1))
            if abs(af - af_round) > 1e-6:
                continue
            # Final sanity: triangle BEF must be non-degenerate
            # BF = c - AF must be positive
            if c - af_round <= 0:
                continue
            break
        else:
            return None

        templates = _TEMPLATES_PYTHAGOREAN
        sidx = (self.seed or 0) % len(templates)
        question = templates[sidx].format(c=c, b=b, e=e)
        if abs(af_round - round(af_round)) < 1e-9:
            answer = str(int(round(af_round)))
        else:
            answer = f"{af_round:g}"
        img = self._render_pythagorean(c, b, e, af_round)
        return question, answer, img

    def _render_midfold(self, c, b) -> Image.Image:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")

        # Left: BEFORE fold — rectangle ABCD with crease line indicated.
        ax = axes[0]
        ax.set_facecolor("#ffffff")
        # A=top-left, B=top-right, C=bottom-right, D=bottom-left.
        A = (0, b)
        B = (c, b)
        C = (c, 0)
        D = (0, 0)
        ax.add_patch(patches.Polygon([A, B, C, D], closed=True,
                     facecolor="#e7eef9", edgecolor="#2c3e50", lw=2.0))
        # Midline (horizontal crease).
        mid_y = b / 2
        ax.plot([0, c], [mid_y, mid_y], color="#7b241c", lw=2.0,
                linestyle="--")
        # Vertex labels
        ax.text(A[0] - 0.15, A[1] + 0.1, "A", ha="right", va="bottom",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(B[0] + 0.15, B[1] + 0.1, "B", ha="left", va="bottom",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(C[0] + 0.15, C[1] - 0.1, "C", ha="left", va="top",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(D[0] - 0.15, D[1] - 0.1, "D", ha="right", va="top",
                fontsize=12, fontweight="bold", color="#2c3e50")
        # AB = c label
        ax.annotate("", xy=B, xytext=A,
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(c / 2, b + 0.4, f"AB = {c}",
                ha="center", va="bottom", fontsize=12, color="#c0392b",
                fontweight="bold")
        ax.text(c / 2, mid_y + 0.15, "crease (midline)", ha="center",
                va="bottom", fontsize=10, color="#7b241c", fontweight="bold")
        ax.set_title("Before fold", fontsize=11, fontweight="bold")
        ax.set_xlim(-1.2, c + 1.2)
        ax.set_ylim(-1.0, b + 1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        # Right: AFTER fold — top half folded down.
        ax2 = axes[1]
        ax2.set_facecolor("#ffffff")
        # Bottom half (still in place)
        ax2.add_patch(patches.Polygon(
            [D, C, (c, mid_y), (0, mid_y)], closed=True,
            facecolor="#dbe5f5", edgecolor="#2c3e50", lw=2.0))
        # Top half folded down (now overlapping on top of bottom half)
        ax2.add_patch(patches.Polygon(
            [(0, mid_y), (c, mid_y), (c, 0), (0, 0)], closed=True,
            facecolor="#f5b7b1", edgecolor="#7b241c", lw=2.0,
            alpha=0.5, hatch="///"))
        ax2.text(c / 2, mid_y / 2, "(folded top half\non bottom half)",
                 ha="center", va="center", fontsize=10,
                 color="#7b241c", fontstyle="italic")
        ax2.text(D[0] - 0.15, D[1] - 0.1, "D", ha="right", va="top",
                 fontsize=12, fontweight="bold", color="#2c3e50")
        ax2.text(C[0] + 0.15, C[1] - 0.1, "C", ha="left", va="top",
                 fontsize=12, fontweight="bold", color="#2c3e50")
        ax2.set_title("After fold (AB lands on DC)", fontsize=11,
                      fontweight="bold")
        ax2.set_xlim(-1.2, c + 1.2)
        ax2.set_ylim(-1.0, b + 1.4)
        ax2.set_aspect("equal")
        ax2.axis("off")

        try:
            fig.tight_layout()
        except Exception:
            pass

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render_pythagorean(self, c, b, e, af) -> Image.Image:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")

        # A=top-left, B=top-right, C=bottom-right, D=bottom-left
        A = (0, b)
        B = (c, b)
        C = (c, 0)
        D = (0, 0)
        E = (c, b - e)  # On BC, distance BE measured from B downward.

        # Left: BEFORE fold — show rectangle ABCD, the crease DF, and mark E on BC.
        ax = axes[0]
        ax.set_facecolor("#ffffff")
        ax.add_patch(patches.Polygon([A, B, C, D], closed=True,
                     facecolor="#e7eef9", edgecolor="#2c3e50", lw=2.0))
        # F on AB at distance AF from A.
        F = (af, b)
        # Crease DF (line from D to F)
        ax.plot([D[0], F[0]], [D[1], F[1]], color="#7b241c", lw=2.2,
                linestyle="--")
        # Mark E on BC.
        ax.plot([E[0]], [E[1]], 'o', color="#1e8449", markersize=8)
        # Vertex labels.
        ax.text(A[0] - 0.2, A[1] + 0.15, "A", ha="right", va="bottom",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(B[0] + 0.2, B[1] + 0.15, "B", ha="left", va="bottom",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(C[0] + 0.2, C[1] - 0.15, "C", ha="left", va="top",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(D[0] - 0.2, D[1] - 0.15, "D", ha="right", va="top",
                fontsize=12, fontweight="bold", color="#2c3e50")
        ax.text(F[0], F[1] + 0.25, "F", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#7b241c")
        ax.text(E[0] + 0.25, E[1], "E", ha="left", va="center",
                fontsize=12, fontweight="bold", color="#1e8449")
        # AB and AD labels (placed well away from the F vertex on top edge).
        ax.text(c / 2, b + 0.85, f"AB = {c}",
                ha="center", va="bottom", fontsize=11, color="#c0392b",
                fontweight="bold")
        ax.text(-0.4, b / 2, f"AD = {b}",
                ha="right", va="center", fontsize=11, color="#c0392b",
                fontweight="bold", rotation=90)
        # BE label
        ax.text(c + 0.4, b - e / 2, f"BE = {e}",
                ha="left", va="center", fontsize=11, color="#1e8449",
                fontweight="bold")
        # Mark AF as the unknown — place below the top edge inside rect.
        ax.text(af / 2, b - 0.45, "AF = ?", ha="center", va="top",
                fontsize=10, color="#7b241c", fontweight="bold")
        ax.set_title("Before fold (crease DF shown)", fontsize=11,
                     fontweight="bold")
        ax.set_xlim(-1.5, c + 1.6)
        ax.set_ylim(-1.0, b + 1.8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Right: AFTER fold. Triangle ADF maps to triangle EDF (vertex A → E,
        # D fixed, F fixed). Show the original rectangle in light, and the
        # folded triangle (D, F, E) overlaid in red.
        ax2 = axes[1]
        ax2.set_facecolor("#ffffff")
        ax2.add_patch(patches.Polygon([A, B, C, D], closed=True,
                      facecolor="#f6f6f6", edgecolor="#bdc3c7", lw=1.5,
                      linestyle="--"))
        # Original triangle ADF (light red)
        ax2.add_patch(patches.Polygon([A, D, F], closed=True,
                      facecolor="#f5b7b1", edgecolor="#7b241c", lw=1.6,
                      alpha=0.4))
        # Folded triangle EDF (heavier outline)
        ax2.add_patch(patches.Polygon([E, D, F], closed=True,
                      facecolor="#e74c3c", edgecolor="#7b241c", lw=2.0,
                      alpha=0.55))
        # Crease DF
        ax2.plot([D[0], F[0]], [D[1], F[1]], color="#7b241c", lw=2.2)
        # Labels
        ax2.text(A[0] - 0.2, A[1] + 0.15, "A (orig)", ha="right", va="bottom",
                 fontsize=10, fontweight="bold", color="#aaaaaa")
        ax2.text(B[0] + 0.2, B[1] + 0.15, "B", ha="left", va="bottom",
                 fontsize=12, fontweight="bold", color="#2c3e50")
        ax2.text(C[0] + 0.2, C[1] - 0.15, "C", ha="left", va="top",
                 fontsize=12, fontweight="bold", color="#2c3e50")
        ax2.text(D[0] - 0.2, D[1] - 0.15, "D", ha="right", va="top",
                 fontsize=12, fontweight="bold", color="#2c3e50")
        ax2.text(F[0], F[1] + 0.25, "F", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="#7b241c")
        ax2.text(E[0] + 0.25, E[1], "E (image of A)", ha="left", va="center",
                 fontsize=10, fontweight="bold", color="#1e8449")
        ax2.text(c / 2, -0.7,
                 "After fold: A → E (so DA = DE, FA = FE)",
                 ha="center", va="top", fontsize=10, color="#5d4037",
                 fontstyle="italic")
        ax2.set_title("After fold (A maps to E on BC)", fontsize=11,
                      fontweight="bold")
        ax2.set_xlim(-1.5, c + 1.8)
        ax2.set_ylim(-1.4, b + 1.4)
        ax2.set_aspect("equal")
        ax2.axis("off")

        try:
            fig.tight_layout()
        except Exception:
            pass

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
