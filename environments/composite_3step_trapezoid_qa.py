"""
3-step composite: identify perimeter formula for trapezoid, identify side
lengths from figure, compute sum. Targets multi-step composite reasoning
on plane figures.

Format: plane-figure MCQ with 4-5 options labeled A-E (last option
"No correct answer"), prefix "As shown in the figure" / "As shown in
the diagram", values include unit "cm" inline (e.g. "16 cm", "24").
Stem style: "...the trapezoid has top base = X cm, bottom base = Y cm,
equal legs Z cm. What is the perimeter? A. 24; B. 20; C. 18;
D. No correct answer."
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# 2026-05-05 R5 B3B HARDEN: right-trapezoid area templates for L8/L9.
# One leg is perpendicular to the parallel bases (so its length = height of
# the trapezoid). Forces (1) recognise the right-trapezoid shape, (2)
# compute area = (top + bot) * h / 2, (3) compare to options. Distractors:
# parallelogram-area (top * h or bot * h), forgot to halve, used wrong h.
_TEMPLATES_RT_AREA = [
    "As shown in the figure, the trapezoid ABCD is a right trapezoid: side AD is perpendicular to both parallel bases. The top base = {top} cm, bottom base = {bot} cm, and the vertical leg AD = {h} cm. What is the area of the trapezoid in cm²?",
    "As shown in the diagram, in right trapezoid ABCD, AD is perpendicular to AB and DC. The lengths are: top base DC = {top} cm, bottom base AB = {bot} cm, vertical leg AD = {h} cm. Compute its area in cm².",
    "As shown in the figure, the trapezoid in the figure has one vertical leg of {h} cm joining the two parallel bases of {top} cm (top) and {bot} cm (bottom). What is the area in cm²?",
    "As shown in the diagram, a right trapezoid has parallel bases {top} cm and {bot} cm and a vertical leg {h} cm. What is the area of the trapezoid (in cm²)?",
    "As shown in the figure, the right trapezoid ABCD has perpendicular leg AD = {h} cm joining bases AB = {bot} cm and DC = {top} cm. Find the area in cm².",
    "As shown in the diagram, the right trapezoid has top base {top} cm, bottom base {bot} cm, and the vertical leg between them is {h} cm. What is its area in cm²?",
    "As shown in the figure, in a right trapezoid the upper base is {top} cm, the lower base is {bot} cm, and the leg perpendicular to the bases is {h} cm. Compute its area (cm²).",
    "As shown in the diagram, a right trapezoid has its vertical leg labeled {h} cm and parallel bases of {top} cm and {bot} cm. What is the area in cm²?",
]

_TEMPLATES = [
    "As shown in the figure, an isosceles trapezoid has top base {top} cm, bottom base {bot} cm, and each leg {leg} cm. What is the perimeter of the trapezoid in cm?",
    "As shown in the diagram, the trapezoid has top base = {top} cm, bottom base = {bot} cm, and each leg = {leg} cm. What is the perimeter (in cm)?",
    "As shown in the figure, the isosceles trapezoid has upper base {top} cm, lower base {bot} cm, and slanted equal legs {leg} cm. What is the perimeter in cm?",
    "As shown in the diagram, an isosceles trapezoid has parallel bases {top} cm and {bot} cm, and the two legs each measure {leg} cm. Compute its perimeter (cm).",
    "As shown in the figure, in the isosceles trapezoid, top = {top} cm, bottom = {bot} cm, and equal slanted sides each = {leg} cm. What is the perimeter (cm)?",
    "As shown in the diagram, the trapezoid has bases of {top} cm (top) and {bot} cm (bottom), with equal legs of {leg} cm. Find the perimeter in cm.",
    "As shown in the figure, an isosceles trapezoid has top base of {top} cm, bottom base of {bot} cm, and each slanted side {leg} cm. What is the total perimeter in cm?",
    "As shown in the diagram, the trapezoid has top base {top} cm, bottom base {bot} cm, and two legs of {leg} cm each. What is the perimeter (in cm)?",
    "As shown in the figure, given an isosceles trapezoid with top {top} cm, bottom {bot} cm, and slanted equal legs {leg} cm, what is its perimeter (cm)?",
    "As shown in the diagram, the isosceles trapezoid has upper base = {top} cm, lower base = {bot} cm, and each leg = {leg} cm. What is the perimeter in cm?",
    "As shown in the figure, the trapezoid in the figure has parallel sides {top} cm (upper) and {bot} cm (lower), and legs each of {leg} cm. Compute the perimeter (cm).",
    "As shown in the diagram, in the isosceles trapezoid, the bases measure {top} cm and {bot} cm, and the equal legs measure {leg} cm. What is the perimeter (in cm)?",
    "As shown in the figure, an isosceles trapezoid has bases {top} cm and {bot} cm, with equal slanted sides {leg} cm long. What is the perimeter (cm)?",
    "As shown in the diagram, given an isosceles trapezoid with the parallel sides labeled {top} cm and {bot} cm and legs each {leg} cm, find the perimeter in cm.",
    "As shown in the figure, the trapezoid has top base {top} cm, bottom base {bot} cm, and each leg {leg} cm. What is the perimeter of the trapezoid in cm?",
    "As shown in the diagram, the isosceles trapezoid in the figure has upper base of {top} cm, lower base of {bot} cm, and equal legs each {leg} cm. What is the perimeter (cm)?",
]


class Composite3StepTrapezoidQA(StandaloneVisualEnv):
    ENV_NAME = "composite_3step_trapezoid"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04: was max_size=6+L*2 → 100% saturated even at L9.
        # Bumped to 4+L*4 (4 → 40) so L9 has 3-digit perimeters.
        max_size = 4 + level * 4
        # 2026-05-05 R5 B3B HARDEN: at L8/L9 swap from isosceles-perimeter to
        # right-trapezoid AREA computed via split into rectangle+triangle. One
        # leg is vertical (so its length is the height), the other is slanted
        # (must use Pythag to find foot of altitude, then split). Distractors:
        # parallelogram-formula instead of trapezoid; forgot to halve. Also
        # bump trap rate to 40% (was 25% in R3).
        # 2026-05-05 Phase A: bump trap rate to 50% at L8/L9 (R5 was 40%
        # but env saturated 100% — model still finds correct via elimination).
        return {"level": level, "max_size": max_size,
                "e_trap": level >= 8,
                "e_trap_rate": 0.50 if level >= 8 else 0.0,
                "right_trap_area": level >= 8}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1019 + level * 71 + 41)

        # 2026-05-05 R5 B3B HARDEN: at L8/L9, ~50% of the time use the
        # right-trapezoid AREA variant (split into rect + triangle when needed).
        if cfg.get("right_trap_area") and rng.random() < 0.50:
            return self._gen_right_trap_area(rng, cfg)
        return self._gen_isosceles_perimeter(rng, cfg)

    def _gen_isosceles_perimeter(self, rng, cfg
                                  ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(80):
            top = rng.randint(2, max(3, cfg["max_size"] // 2))
            bot = top + 2 * rng.randint(1, max(1, cfg["max_size"] // 3))
            leg = rng.randint(top, max(top + 1, cfg["max_size"] // 2 + top))
            # Need leg > (bot - top) / 2 for shape to close
            half_diff = (bot - top) / 2.0
            if leg <= half_diff + 0.5:
                continue
            perim = top + bot + 2 * leg
            sidx = (self.seed or 0) % len(_TEMPLATES)
            stem = _TEMPLATES[sidx].format(top=top, bot=bot, leg=leg)

            # Build MCQ options: correct + 3 distractors. Last option is
            # "No correct answer" (matches plane-figure MCQ samples).
            distractor_pool = [
                top + bot + leg,            # forgot 2nd leg
                2 * (top + bot),            # confused with parallelogram-style
                top + bot + 2 * (leg - 1),  # off-by-one on leg
                top + bot + 2 * (leg + 1),  # off-by-one on leg
                3 * leg + top,              # wrong combination
                2 * leg + bot,              # missed top
                2 * leg + top,              # missed bottom
            ]
            distractors = []
            for d in distractor_pool:
                if d != perim and d > 0 and d not in distractors:
                    distractors.append(d)
                if len(distractors) >= 3:
                    break
            if len(distractors) < 3:
                continue
            # 2026-05-05 R5 B3B HARDEN: bumped trap rate to 40% (was 25% R3)
            if cfg.get("e_trap") and rng.random() < cfg.get("e_trap_rate", 0.40):
                # Use 4 distractors (no correct value). Need a 4th distractor.
                if len(distractors) < 4:
                    distractors = distractors + [perim + 7, perim - 5,
                                                  perim * 2, perim + 13]
                    distractors = [d for d in distractors if d != perim and d > 0]
                opts_vals = distractors[:4]
                rng.shuffle(opts_vals)
                answer_letter = "E"
                opts_lines = [f"{chr(ord('A')+i)}. {opts_vals[i]}"
                              for i in range(4)]
            else:
                opts_vals = [perim] + distractors[:3]
                rng.shuffle(opts_vals)
                correct_idx = opts_vals.index(perim)
                answer_letter = "ABCD"[correct_idx]
                opts_lines = [f"{chr(ord('A')+i)}. {opts_vals[i]}"
                              for i in range(4)]
            opts_lines.append("E. No correct answer")
            question = (
                f"{stem}\n\n"
                + "\n".join(opts_lines)
                + "\n\nChoose the correct option (A, B, C, D, or E)."
            )
            img = self._render(top, bot, leg)
            return question, answer_letter, img
        return None

    def _gen_right_trap_area(self, rng, cfg
                              ) -> Optional[Tuple[str, str, Image.Image]]:
        """3-step right-trapezoid AREA. Three sub-skills:
         (1) recognise vertical leg = h,
         (2) apply trapezoid area formula = (top + bot) * h / 2,
         (3) avoid parallelogram-trap (top*h or bot*h, forgot to halve).
        """
        for _ in range(60):
            # Need top+bot to be even so area is an integer.
            top = rng.randint(2, max(3, cfg["max_size"] // 2))
            bot = top + 2 * rng.randint(1, max(1, cfg["max_size"] // 3))
            h = rng.choice([4, 6, 8, 10, 12, 14])  # even h → area always int
            # Sanity: avoid degenerate
            if top + bot < 4 or h < 2:
                continue
            sum_bases = top + bot
            area = sum_bases * h // 2  # integer because both top+bot even and *h
            if area <= 0:
                continue
            sidx = (self.seed or 0) % len(_TEMPLATES_RT_AREA)
            stem = _TEMPLATES_RT_AREA[sidx].format(top=top, bot=bot, h=h)

            # Distractors target classic mistakes
            distractor_pool = [
                sum_bases * h,            # forgot to halve (parallelogram-style)
                top * h,                   # used only top base
                bot * h,                   # used only bottom base
                (sum_bases - h),           # confused with perimeter add
                top * bot,                 # multiplied bases together
                top + bot + h,             # added all 3 (perimeter-shaped)
                area + h,                  # off by h
                area - h if area > h else area + 2 * h,
            ]
            distractors = []
            for d in distractor_pool:
                if d != area and d > 0 and d not in distractors:
                    distractors.append(d)
                if len(distractors) >= 4:
                    break
            if len(distractors) < 3:
                continue

            if cfg.get("e_trap") and rng.random() < cfg.get("e_trap_rate", 0.40):
                if len(distractors) < 4:
                    for delta in [3, 5, 7, 11]:
                        cand = area + delta
                        if cand not in distractors and cand > 0 and cand != area:
                            distractors.append(cand)
                        if len(distractors) >= 4:
                            break
                opts_vals = distractors[:4]
                rng.shuffle(opts_vals)
                answer_letter = "E"
            else:
                opts_vals = [area] + distractors[:3]
                rng.shuffle(opts_vals)
                correct_idx = opts_vals.index(area)
                answer_letter = "ABCD"[correct_idx]

            opts_lines = [f"{chr(ord('A')+i)}. {opts_vals[i]}"
                          for i in range(4)]
            opts_lines.append("E. No correct answer")
            question = (
                f"{stem}\n\n"
                + "\n".join(opts_lines)
                + "\n\nChoose the correct option (A, B, C, D, or E)."
            )
            img = self._render_right_trap(top, bot, h)
            return question, answer_letter, img
        return None

    def _render_right_trap(self, top, bot, h) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Right trapezoid: BL=(0,0), BR=(bot,0), TR=(top,h), TL=(0,h).
        # AD vertical leg = TL–BL.  BC slanted leg = BR–TR.
        BL = (0, 0); BR = (bot, 0); TR = (top, h); TL = (0, h)
        xs = [BL[0], BR[0], TR[0], TL[0], BL[0]]
        ys = [BL[1], BR[1], TR[1], TL[1], BL[1]]
        ax.plot(xs, ys, color="#1f4e79", linewidth=2.4)
        ax.fill(xs[:-1], ys[:-1], color="#cce4f0", alpha=0.6)
        # Right-angle markers at BL and TL (vertical leg)
        sq = 0.4
        ax.plot([BL[0], BL[0] + sq, BL[0] + sq, BL[0]],
                [BL[1] + sq, BL[1] + sq, BL[1], BL[1]],
                color="#000000", linewidth=1.2)
        ax.plot([TL[0], TL[0] + sq, TL[0] + sq, TL[0]],
                [TL[1] - sq, TL[1] - sq, TL[1], TL[1]],
                color="#000000", linewidth=1.2)
        # Vertex labels
        ax.text(BL[0] - 0.4, BL[1] - 0.4, "A", fontsize=11, fontweight="bold",
                color="#1f4e79", ha="right", va="top")
        ax.text(BR[0] + 0.3, BR[1] - 0.4, "B", fontsize=11, fontweight="bold",
                color="#1f4e79", ha="left", va="top")
        ax.text(TR[0] + 0.3, TR[1] + 0.3, "C", fontsize=11, fontweight="bold",
                color="#1f4e79", ha="left", va="bottom")
        ax.text(TL[0] - 0.4, TL[1] + 0.3, "D", fontsize=11, fontweight="bold",
                color="#1f4e79", ha="right", va="bottom")
        # Side labels
        ax.annotate(f"DC = {top} cm", ((TL[0] + TR[0]) / 2, TL[1] + 0.3),
                    ha="center", fontsize=11, color="#0d3a5c")
        ax.annotate(f"AB = {bot} cm", ((BL[0] + BR[0]) / 2, BL[1] - 0.6),
                    ha="center", fontsize=11, color="#0d3a5c")
        ax.annotate(f"AD = {h} cm", (BL[0] - 0.4, (BL[1] + TL[1]) / 2),
                    fontsize=11, color="#7c1f1f", ha="right",
                    va="center", rotation=90)
        ax.set_aspect("equal")
        ax.set_xlim(-3, bot + 2)
        ax.set_ylim(-2, max(h + 2, 5))
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render(self, top, bot, leg) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        half_diff = (bot - top) / 2.0
        height = math.sqrt(max(0.0, leg * leg - half_diff * half_diff))
        # bottom-left, bottom-right, top-right, top-left
        BL = (0, 0); BR = (bot, 0); TR = (bot - half_diff, height); TL = (half_diff, height)
        xs = [BL[0], BR[0], TR[0], TL[0], BL[0]]
        ys = [BL[1], BR[1], TR[1], TL[1], BL[1]]
        ax.plot(xs, ys, color="#1f4e79", linewidth=2.4)
        ax.fill(xs[:-1], ys[:-1], color="#cce4f0", alpha=0.6)
        # Side labels
        ax.annotate(f"{top} cm", ((TL[0] + TR[0]) / 2, TL[1] + 0.3),
                    ha="center", fontsize=11, color="#0d3a5c")
        ax.annotate(f"{bot} cm", ((BL[0] + BR[0]) / 2, BL[1] - 0.6),
                    ha="center", fontsize=11, color="#0d3a5c")
        ax.annotate(f"{leg} cm", ((BL[0] + TL[0]) / 2 - 0.6, (BL[1] + TL[1]) / 2),
                    fontsize=11, color="#7c1f1f", rotation=80)
        ax.annotate(f"{leg} cm", ((BR[0] + TR[0]) / 2 + 0.4, (BR[1] + TR[1]) / 2),
                    fontsize=11, color="#7c1f1f", rotation=-80)
        ax.set_aspect("equal")
        ax.set_xlim(-2, bot + 2)
        ax.set_ylim(-2, max(height + 2, 5))
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
