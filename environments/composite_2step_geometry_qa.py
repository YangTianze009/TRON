"""
Two-step composite geometry: a single-figure problem that requires the
solver to chain two atomic skills:

  Sub-step 1: read a primary measurement (diagonal, area, or perimeter
              of a square / rectangle / circle / equilateral triangle)
              and back-solve for the side length s.
  Sub-step 2: use s to compute a different quantity (perimeter from area,
              area from perimeter, or area from diagonal).

The model only outputs the final composite answer — sub-step 1 is implicit.
This pattern targets InadequateGeneralization in multi-step composite
benchmarks: each atomic step is easy in isolation, but combined performance
drops because the intermediate has to be carried forward correctly.

L0: square with diagonal 4√2 → side 4 → perimeter 16.
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_SQ_DIAG_TO_PERI = [
    "As shown in the diagram, square ABCD has diagonal AC = {d}. What is the perimeter of the square?",
    "As shown in the figure, a square has its diagonal labeled {d}. What is the perimeter of the square?",
    "As shown in the diagram, the square in the figure has diagonal {d}. Compute its perimeter.",
    "As shown in the figure, square ABCD has its diagonal AC = {d}. Find the perimeter of the square.",
    "As shown in the diagram, a square has diagonal length {d}. What is the perimeter?",
    "As shown in the figure, the square ABCD shown has diagonal {d}. What is the perimeter of the square?",
]

_TEMPLATES_SQ_AREA_TO_PERI = [
    "As shown in the diagram, a square has area = {A}. What is its perimeter?",
    "As shown in the figure, the square shown has area {A}. Compute the perimeter.",
    "As shown in the diagram, a square has area {A}. Find the perimeter.",
    "As shown in the figure, the square has area = {A}. What is its perimeter?",
    "As shown in the diagram, a square in the figure has area {A}. Find the perimeter.",
    "As shown in the figure, the square shown has area {A}. What is the perimeter of the square?",
]

_TEMPLATES_SQ_PERI_TO_AREA = [
    "As shown in the diagram, the square in the figure has perimeter {P}. What is its area?",
    "As shown in the figure, a square has perimeter {P}. Compute the area of the square.",
    "As shown in the diagram, the square has perimeter = {P}. Find its area.",
    "As shown in the figure, a square of perimeter {P} is shown. What is its area?",
    "As shown in the diagram, the square has perimeter {P}. What is the area?",
    "As shown in the figure, a square with perimeter {P} is shown. Compute the area.",
]

_TEMPLATES_RECT_DIAG_RATIO_TO_AREA = [
    "As shown in the diagram, a rectangle has diagonal {d} and length:width ratio {r1}:{r2}. What is the rectangle's area?",
    "As shown in the figure, a rectangle has diagonal {d} and length-to-width ratio {r1}:{r2}. Find its area.",
    "As shown in the diagram, the rectangle has diagonal = {d} and length:width = {r1}:{r2}. Compute its area.",
    "As shown in the figure, the rectangle shown has diagonal {d} and length:width ratio {r1}:{r2}. What is its area?",
]

# 2026-05-05 R5 B3B HARDEN: 3-step problem — back-solve side from diagonal,
# compute area in cm², then convert to dm² (÷100). Distractors trap models
# that forgot the conversion or used the wrong factor.
_TEMPLATES_SQ_DIAG_TO_AREA_DM2 = [
    "As shown in the diagram, square ABCD has diagonal AC = {d} cm. What is the area of the square in dm² (1 dm² = 100 cm²)?",
    "As shown in the figure, the square in the figure has diagonal {d} cm. What is its area expressed in dm²?",
    "As shown in the diagram, a square has diagonal {d} cm. Compute the area in dm² (recall 1 dm² = 100 cm²).",
    "As shown in the figure, square ABCD has diagonal AC = {d} cm. Express its area in dm².",
]

_TEMPLATES_SQ_PERI_TO_AREA_DM2 = [
    "As shown in the diagram, a square has perimeter {P} cm. What is the area of the square in dm² (1 dm² = 100 cm²)?",
    "As shown in the figure, the square shown has perimeter {P} cm. Compute the area in dm².",
    "As shown in the diagram, a square has perimeter {P} cm. Find its area in dm² (1 dm² = 100 cm²).",
    "As shown in the figure, a square with perimeter {P} cm is shown. Express the area in dm².",
]


def _format_sqrt_term(integer_part: int, root: int) -> str:
    """Render integer_part * sqrt(root) compactly: '4√2', '6√2', etc.
    If root == 1, return just the integer."""
    if root == 1 or root == 0:
        return f"{integer_part}"
    if integer_part == 1:
        return f"√{root}"
    if integer_part == -1:
        return f"-√{root}"
    return f"{integer_part}√{root}"


class Composite2StepGeometryQA(StandaloneVisualEnv):
    ENV_NAME = "composite_2step_geometry"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-05 R5 B3B HARDEN: bumped e_trap to 40% at L8/L9 (was 25%
        # in R3). At L8/L9 also enable a 3rd-step "convert to dm²/m²" variant
        # so the model must (1) back-solve side, (2) compute area, (3) apply
        # unit conversion. Distractors include forgot-conversion (×100/×10000).
        # 2026-05-05 Phase A: bump e_trap to 50% (R5 hardening was
        # insufficient — env still saturated 100%).
        return {"level": level, "e_trap": level >= 8,
                "e_trap_rate": 0.50 if level >= 8 else 0.0,
                "third_step_unit": level >= 8}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1373 + level * 47 + 19)

        # Choose the variant
        if level == 0:
            # Canonical L0 from task: square diag 4√2 → side 4 → perimeter 16
            variant = "sq_diag_to_peri"
            s = 4
        elif level <= 2:
            variant = rng.choice(["sq_diag_to_peri", "sq_peri_to_area",
                                   "sq_area_to_peri"])
            s = rng.choice([2, 3, 4, 5, 6, 8])
        elif level <= 4:
            variant = rng.choice(["sq_diag_to_peri", "sq_peri_to_area",
                                   "sq_area_to_peri"])
            s = rng.choice([4, 5, 6, 7, 8, 9, 10, 12])
        elif level <= 6:
            variant = rng.choice(["sq_diag_to_peri", "sq_peri_to_area",
                                   "sq_area_to_peri", "rect_diag_ratio"])
            s = rng.choice([5, 6, 7, 8, 9, 10, 12, 15])
        elif level == 7:
            # L7: still pure 2-step but bigger numbers + rect variants
            # (R3 behaviour). 3-step dm² conversion only fires at L8/L9.
            variant = rng.choice(["sq_diag_to_peri", "sq_area_to_peri",
                                   "rect_diag_ratio", "rect_diag_ratio"])
            s = rng.choice([12, 15, 16, 18, 20, 24, 25, 28, 30, 36, 40])
        else:
            # 2026-05-05 R5 B3B HARDEN: at L8/L9 add the 3-step
            # diag→side→area→dm² conversion variant (~43% mix). This forces a
            # 3rd reasoning step (unit conversion). The other 57% remains the
            # 2-step rect / sq variants from R3 with bigger numbers.
            variant = rng.choice([
                "sq_diag_to_area_dm2", "sq_diag_to_area_dm2",
                "sq_peri_to_area_dm2",
                "sq_diag_to_peri", "sq_area_to_peri",
                "rect_diag_ratio", "rect_diag_ratio",
            ])
            # For dm² variants, side must yield clean dm² (i.e. area divisible by
            # 100). Picks below give s² in {100, 400, 900, 1600, 2500, 3600, ...}.
            if variant.endswith("_dm2"):
                s = rng.choice([10, 20, 30, 40, 50, 60])
            else:
                s = rng.choice([12, 15, 16, 18, 20, 24, 25, 28, 30, 36, 40])

        if variant == "sq_diag_to_peri":
            # diagonal = s * sqrt(2)
            d_str = _format_sqrt_term(s, 2)
            perimeter = 4 * s
            sidx = (self.seed or 0) % len(_TEMPLATES_SQ_DIAG_TO_PERI)
            stem = _TEMPLATES_SQ_DIAG_TO_PERI[sidx].format(d=d_str)
            correct = perimeter
            distractors = [s * 2, s * s, 8 * s, 2 * s + 4]
            unit = "cm"
            img = self._render_square(s, label_kind="diagonal",
                                       label_str=d_str)

        elif variant == "sq_area_to_peri":
            area = s * s
            perimeter = 4 * s
            sidx = (self.seed or 0) % len(_TEMPLATES_SQ_AREA_TO_PERI)
            stem = _TEMPLATES_SQ_AREA_TO_PERI[sidx].format(A=area)
            correct = perimeter
            distractors = [area, 2 * area, 2 * s, 8 * s]
            unit = "cm"
            img = self._render_square(s, label_kind="area",
                                       label_str=f"area = {area}")

        elif variant == "sq_peri_to_area":
            perimeter = 4 * s
            area = s * s
            sidx = (self.seed or 0) % len(_TEMPLATES_SQ_PERI_TO_AREA)
            stem = _TEMPLATES_SQ_PERI_TO_AREA[sidx].format(P=perimeter)
            correct = area
            distractors = [perimeter, 2 * area, area + perimeter, perimeter // 2 if perimeter > 1 else 1]
            unit = "cm²"
            img = self._render_square(s, label_kind="perimeter",
                                       label_str=f"perimeter = {perimeter}")

        elif variant == "sq_diag_to_area_dm2":
            # 3-step: diagonal → side → area in cm² → convert to dm² (÷100).
            # s is chosen multiple of 10 so area is multiple of 100, dm² is
            # always an integer.
            d_str = _format_sqrt_term(s, 2)
            area_cm2 = s * s
            area_dm2 = area_cm2 // 100  # since s ∈ {10,20,30,...}
            sidx = (self.seed or 0) % len(_TEMPLATES_SQ_DIAG_TO_AREA_DM2)
            stem = _TEMPLATES_SQ_DIAG_TO_AREA_DM2[sidx].format(d=d_str)
            correct = area_dm2
            # Distractor pool: forgot conversion (gives cm²), wrong conversion
            # factor (×10 instead of ÷100), perimeter, side itself, ÷10000
            # (m² confusion).
            distractors = [
                area_cm2,                  # forgot to convert
                area_cm2 // 10 if area_cm2 % 10 == 0 else area_cm2 // 10 + 1,
                4 * s,                     # gave perimeter
                s,                          # gave side length
                area_cm2 * 10,              # wrong direction conversion
            ]
            distractors = [d for d in distractors if d != correct and d > 0]
            unit = "dm²"
            img = self._render_square(s, label_kind="diagonal",
                                       label_str=f"{d_str} cm")

        elif variant == "sq_peri_to_area_dm2":
            # 3-step: perimeter → side → area cm² → dm². s ∈ {10,20,30,...}.
            perimeter = 4 * s
            area_cm2 = s * s
            area_dm2 = area_cm2 // 100
            sidx = (self.seed or 0) % len(_TEMPLATES_SQ_PERI_TO_AREA_DM2)
            stem = _TEMPLATES_SQ_PERI_TO_AREA_DM2[sidx].format(P=perimeter)
            correct = area_dm2
            distractors = [
                area_cm2,                  # forgot conversion
                perimeter,                 # gave perimeter
                area_cm2 // 10 if area_cm2 % 10 == 0 else area_cm2 // 10 + 1,
                s,                         # gave side
                area_cm2 // 1000 if area_cm2 >= 1000 else area_cm2,
            ]
            distractors = [d for d in distractors if d != correct and d > 0]
            unit = "dm²"
            img = self._render_square(s, label_kind="perimeter",
                                       label_str=f"perimeter = {perimeter} cm")

        elif variant == "rect_diag_ratio":
            # length:width = r1:r2, diagonal d = sqrt(r1^2 + r2^2) * k
            r1, r2 = rng.choice([(3, 4), (4, 3), (5, 12), (12, 5), (6, 8), (8, 6),
                                 (8, 15), (15, 8)])
            k = rng.choice([1, 2, 3])
            length = r1 * k
            width = r2 * k
            diag = math.sqrt(length * length + width * width)
            d_int = int(round(diag))
            if abs(diag - d_int) > 1e-6:
                return None
            area = length * width
            sidx = (self.seed or 0) % len(_TEMPLATES_RECT_DIAG_RATIO_TO_AREA)
            stem = _TEMPLATES_RECT_DIAG_RATIO_TO_AREA[sidx].format(
                d=d_int, r1=r1, r2=r2)
            correct = area
            distractors = [length + width, 2 * (length + width),
                           d_int * d_int, length * d_int]
            unit = "cm²"
            img = self._render_rectangle(length, width,
                                          diag_label=f"d = {d_int}")
        else:
            return None

        # Build MCQ (4 options A-D + E. No correct answer)
        # 2026-05-05 R5 B3B HARDEN: bumped trap rate to 40% at L8/L9 (was 25%).
        # Multi-step style: every Q has option E and ~10-40% have E
        # correct in the multi-step subset where confident-wrong is common.
        if cfg.get("e_trap") and rng.random() < cfg.get("e_trap_rate", 0.40):
            wrong_vals = []
            seen_w = {correct}
            for d in distractors:
                if d != correct and d > 0 and d not in seen_w:
                    wrong_vals.append(d)
                    seen_w.add(d)
                if len(wrong_vals) >= 4:
                    break
            if len(wrong_vals) < 4:
                for delta in [1, 2, 3, 5, 7, 11, 13]:
                    cand = correct + delta
                    if cand not in seen_w and cand > 0:
                        wrong_vals.append(cand)
                        seen_w.add(cand)
                    if len(wrong_vals) >= 4:
                        break
            if len(wrong_vals) < 4:
                return None
            rng.shuffle(wrong_vals)
            opts_vals = wrong_vals[:4]
            answer = "E"
        else:
            opts_vals = [correct]
            seen = {correct}
            for d in distractors:
                if d != correct and d > 0 and d not in seen:
                    opts_vals.append(d)
                    seen.add(d)
                if len(opts_vals) >= 4:
                    break
            if len(opts_vals) < 4:
                # pad
                for delta in [1, 2, 3, 5, 7]:
                    cand = correct + delta
                    if cand not in seen:
                        opts_vals.append(cand)
                        seen.add(cand)
                    if len(opts_vals) >= 4:
                        break
            if len(opts_vals) < 4:
                return None
            rng.shuffle(opts_vals)
            correct_idx = opts_vals.index(correct)
            answer = "ABCD"[correct_idx]
        opts_lines = [f"{chr(ord('A')+i)}. {opts_vals[i]} {unit}"
                      for i in range(4)]
        opts_lines.append("E. No correct answer")
        question = (
            f"{stem}\n\n"
            + "\n".join(opts_lines)
            + "\n\nChoose the correct option (A, B, C, D, or E)."
        )
        return question, answer, img

    def _render_square(self, s, label_kind, label_str) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Square positioned at (0,0) -> (s,s)
        sq = patches.Rectangle((0, 0), s, s, facecolor="#cce5ff",
                                edgecolor="#1f4e79", linewidth=2)
        ax.add_patch(sq)
        # Vertex labels
        ax.annotate("A", (0, 0), textcoords="offset points", xytext=(-12, -14),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        ax.annotate("B", (s, 0), textcoords="offset points", xytext=(6, -14),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        ax.annotate("C", (s, s), textcoords="offset points", xytext=(6, 6),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        ax.annotate("D", (0, s), textcoords="offset points", xytext=(-12, 6),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        # Optional: label depending on label_kind
        if label_kind == "diagonal":
            # Draw the diagonal AC
            ax.plot([0, s], [0, s], color="#c0392b", linewidth=2)
            ax.text(s / 2 - s * 0.18, s / 2 + s * 0.05,
                    label_str, fontsize=13, color="#c0392b", fontweight="bold",
                    ha="right")
        elif label_kind == "area":
            ax.text(s / 2, s / 2, label_str,
                    ha="center", va="center", fontsize=13, color="#1f4e79",
                    fontweight="bold")
        elif label_kind == "perimeter":
            ax.text(s / 2, -s * 0.18, label_str,
                    ha="center", va="top", fontsize=13, color="#1f4e79",
                    fontweight="bold")

        margin = s * 0.4
        ax.set_xlim(-margin, s + margin)
        ax.set_ylim(-margin, s + margin)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render_rectangle(self, length, width, diag_label) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        rect = patches.Rectangle((0, 0), length, width,
                                  facecolor="#cce5ff", edgecolor="#1f4e79",
                                  linewidth=2)
        ax.add_patch(rect)
        # Vertex labels A,B,C,D
        ax.annotate("A", (0, 0), textcoords="offset points", xytext=(-12, -14),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        ax.annotate("B", (length, 0), textcoords="offset points", xytext=(6, -14),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        ax.annotate("C", (length, width), textcoords="offset points", xytext=(6, 6),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        ax.annotate("D", (0, width), textcoords="offset points", xytext=(-12, 6),
                    fontsize=12, fontweight="bold", color="#1f4e79")
        # Diagonal AC
        ax.plot([0, length], [0, width], color="#c0392b", linewidth=2)
        ax.text(length / 2 - length * 0.05, width / 2 + width * 0.08,
                diag_label, fontsize=12, color="#c0392b", fontweight="bold",
                ha="right")
        # Side ratio annotation along the bottom
        ax.text(length / 2, -width * 0.18,
                f"length : width = {length}:{width}",
                ha="center", va="top", fontsize=11, color="#1f4e79")

        margin_x = length * 0.25
        margin_y = width * 0.45
        ax.set_xlim(-margin_x, length + margin_x)
        ax.set_ylim(-margin_y, width + margin_y * 0.7)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
