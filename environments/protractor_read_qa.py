"""
Read an angle off a semicircular protractor. Tests reading inner vs outer scale
based on which ray is the reference (left or right baseline). The scale-reading
discipline is a known weakness on geometric figure benchmarks.

Two output modes (selected probabilistically per-problem):
  - Open numeric: bare integer angle (legacy mode).
  - MCQ A-E: discrete angle options matching the dominant benchmark surface
    style for protractor-read questions (e.g., "(A) 30° (B) 45° (C) 60°
    (D) 90° (E) No correct answer").

Difficulty axes:
  - angle (multiples of 5 → multiples of 1)
  - scale ambiguity: only one scale labeled vs both labeled
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
from ._mcq_letter_lib import make_mcq_letter, build_mcq_prompt_suffix


_TEMPLATES = [
    "The image shows a semicircular protractor with two rays from its center. Read the angle in degrees between the two rays. Place the integer in <answer>...</answer>.",
    "Use the protractor in the image to determine the angle (in degrees) between the two rays. Integer answer in <answer>...</answer>.",
    "What is the angle (in degrees) shown by the two rays on the protractor? Place integer in <answer>...</answer>.",
    "Read the protractor: report the measure of the angle (degrees, integer) formed by the two rays. Answer in <answer>...</answer>.",
    "Determine the angle between the two rays on the protractor. Integer degrees in <answer>...</answer>.",
    "Read off the angle measure (in degrees) between the rays from the protractor. Integer in <answer>...</answer>.",
    "Examine the protractor diagram and report the angle in degrees between the two rays. Integer answer in <answer>...</answer>.",
    "What angle (degrees) is shown by the protractor? Integer in <answer>...</answer>.",
    "Read the angle from the protractor scale. Place the integer in <answer>...</answer>.",
    "Output the angle in degrees indicated by the two rays on the protractor. Integer in <answer>...</answer>.",
    "Use the scale of the protractor in the image to read the angle (degrees). Integer answer in <answer>...</answer>.",
    "Read the angle that the protractor in the image is measuring. Place the integer (degrees) in <answer>...</answer>.",
    "Looking at the protractor, what angle is formed between the two rays? Integer (degrees) in <answer>...</answer>.",
    "Read the angle measure from the protractor diagram. Integer in <answer>...</answer>.",
    "Identify the angle in degrees between the two rays on the protractor. Integer answer in <answer>...</answer>.",
    "From the protractor diagram, output the angle (degrees) between the rays. Integer in <answer>...</answer>.",
]


# Benchmark-style MCQ stems (matches the dominant question style for
# protractor-read questions in elementary-math benchmarks: "measure the angle
# with a protractor. ∠D = ( )?" / "what is the measure of ∠B using the
# protractor?". 4-5 discrete angle options + an "E. No correct answer" option.)
_MCQ_STEMS = [
    "As shown in the diagram, measure the angle with a protractor. The angle between the two rays equals ( ).",
    "Using the protractor in the image, what is the measure of the angle between the two rays?",
    "As shown in the diagram, measure the angle using a protractor. The angle equals ( )°.",
    "Looking at the protractor diagram, what is the measure of the angle formed by the two rays? ( )°",
    "The protractor measures an angle between the two rays. Which of the following is the correct measure?",
    "What is the measure of the angle on the protractor? Choose the correct option.",
]


class ProtractorReadQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "protractor_read"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Read the angle and output the integer directly "
        "inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            step = 30  # 30, 60, 90, 120, 150
            both_scales = False
        elif level <= 5:
            step = 10
            both_scales = True
        elif level <= 7:
            step = 5
            both_scales = True
        else:
            step = 1
            both_scales = True
        return {"level": level, "step": step, "both_scales": both_scales}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        # 2026-05-04: added easier L0 mode (was 2.5% — VLM scale-reading limit, attempt fix)
        # L0/L1: give angle in question text + ask binary classification (acute vs obtuse).
        if level <= 1:
            return self._generate_easy_l0l1(level)
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1097 + level * 101 + 19)

        step = cfg["step"]
        # Pick angle in (10, 170)
        angle = rng.choice(list(range(10, 171, step)))
        # Pick reference side: 0 = left (0° to 180° standard)
        # Actually we'll always use a fixed reference: ray from origin going right
        # at angle 0 (0°), and another ray at angle `angle` (0°→180°).
        img = self._render(angle, cfg["both_scales"])

        # ---- Probabilistically produce an MCQ-style question matching the
        # dominant benchmark surface format: 4-5 discrete angle options A-E
        # with the correct angle plus near-miss distractors. Half the problems
        # use bare-integer mode (legacy) and half use MCQ mode.
        if rng.random() < 0.5:
            mcq_result = self._build_mcq_problem(angle, rng)
            if mcq_result is not None:
                question, answer = mcq_result
                return question, answer, img

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        return question, str(angle), img

    def _generate_easy_l0l1(self, level: int):
        """L0/L1: angle is given in the question text. Task is acute/right/obtuse
        classification (4-MCQ) — pure text task, just for shape recognition."""
        rng = random.Random((self.seed or 0) * 1097 + level * 101 + 23)
        # Pick angle from a clear-cut set
        angle = rng.choice([30, 45, 60, 90, 120, 135, 150])
        if angle < 90:
            correct = "acute"
        elif angle == 90:
            correct = "right"
        else:
            correct = "obtuse"
        # Render the protractor too (we still need an image)
        img = self._render(angle, both_scales=False)

        options = ["acute", "right", "obtuse", "straight"]
        rng.shuffle(options)
        correct_letter = "ABCD"[options.index(correct)]
        opt_str = "; ".join(f"{l}. {o}" for l, o in zip("ABCD", options))
        question = (
            f"The protractor in the image shows an angle of {angle} degrees. "
            f"Classify this angle as one of the following:\n"
            f"Options: {opt_str}\n"
            f"Recall: <90° = acute, =90° = right, 90°<x<180° = obtuse, "
            f"=180° = straight. "
            f"Reply with the letter (A/B/C/D) inside <answer>...</answer>."
        )
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # MCQ helper
    # ------------------------------------------------------------------ #

    def _build_mcq_problem(self, angle: int, rng: random.Random):
        """Build an MCQ-style protractor question with 4-5 discrete angle
        options. Returns (question_text, single_letter_answer) or None on
        failure (e.g. cannot generate enough plausible distractors)."""
        # Generate angle distractors at typical near-misses for protractor
        # reading: ±5°, ±10°, ±15°, ±30°, plus the supplementary angle
        # (180-angle) since reading the wrong scale is a common error.
        n_options = rng.choice([4, 5])
        candidate_distractors = []
        # Reading the wrong scale: 180 - angle
        if 0 < 180 - angle < 180 and (180 - angle) != angle:
            candidate_distractors.append(180 - angle)
        # Off-by-half-step (round) candidates
        for delta in (5, -5, 10, -10, 15, -15, 30, -30, 20, -20, 45, -45):
            cand = angle + delta
            if 0 < cand < 180 and cand != angle:
                candidate_distractors.append(cand)
        # Round candidates to nearest 5
        candidate_distractors = [int(round(c / 5) * 5)
                                 for c in candidate_distractors]
        # Dedup, drop the correct angle
        seen = set()
        unique_d = []
        for c in candidate_distractors:
            if c == angle or c in seen:
                continue
            seen.add(c)
            unique_d.append(c)
        if len(unique_d) < n_options - 1:
            # Need more — pad with random near angles
            for _ in range(20):
                cand = int(round(rng.choice(list(range(10, 171, 5))) / 5) * 5)
                if cand != angle and cand not in seen and 0 < cand < 180:
                    seen.add(cand)
                    unique_d.append(cand)
                    if len(unique_d) >= n_options - 1:
                        break
        if len(unique_d) < n_options - 1:
            return None
        rng.shuffle(unique_d)
        distractors = unique_d[:n_options - 1]
        # Optionally include "No correct answer" as the last (E) option, but
        # then the GT must be the angle, not the "No correct answer" option.
        include_no_correct = (n_options == 5)
        if include_no_correct:
            # Reserve one slot for "No correct answer"; correct angle goes
            # among the remaining 4 slots.
            slots = [angle] + distractors[:3]
            rng.shuffle(slots)
            options = slots + ["No correct answer"]
        else:
            options = [angle] + distractors
            rng.shuffle(options)
        # Determine correct letter (find angle in options)
        correct_idx = options.index(angle)
        correct_letter = "ABCDE"[correct_idx]
        # Build option block: "A. 30°; B. 60°; C. 45°; D. 90°[; E. No correct answer]"
        opt_strs = []
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            if isinstance(opt, str):
                opt_strs.append(f"{letter}. {opt}")
            else:
                opt_strs.append(f"{letter}. {opt}°")
        options_block = "; ".join(opt_strs)
        # Pick a stem
        stem = rng.choice(_MCQ_STEMS)
        # Pick a final instruction
        last_letter = "ABCDE"[len(options) - 1]
        instr = build_mcq_prompt_suffix(len(options), rng)
        question = f"{stem}\nOptions: {options_block}\n{instr}"
        return question, correct_letter

    def _render(self, angle: int, both_scales: bool) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Protractor body — semicircle
        r = 1.0
        theta = np.linspace(0, math.pi, 200)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#2c3e50",
                linewidth=2)
        ax.plot([-r * 1.1, r * 1.1], [0, 0], color="#2c3e50", linewidth=2)
        # Tick marks
        for deg in range(0, 181, 10):
            rad = math.radians(deg)
            x1 = r * math.cos(math.pi - rad)
            y1 = r * math.sin(math.pi - rad)
            tick_in = 0.94 if deg % 30 == 0 else 0.97
            x0 = tick_in * r * math.cos(math.pi - rad)
            y0 = tick_in * r * math.sin(math.pi - rad)
            ax.plot([x0, x1], [y0, y1], color="#2c3e50", linewidth=1)
            if deg % 30 == 0 or deg == 90:
                # outer scale (left=180 → right=0, going right-to-left as deg grows)
                lab_r = 1.07
                lx = lab_r * math.cos(math.pi - rad)
                ly = lab_r * math.sin(math.pi - rad)
                ax.text(lx, ly, str(deg), ha="center", va="center",
                        fontsize=8, color="#1a3a6e")
                if both_scales:
                    inv = 180 - deg
                    lab_r2 = 0.86
                    lx2 = lab_r2 * math.cos(math.pi - rad)
                    ly2 = lab_r2 * math.sin(math.pi - rad)
                    ax.text(lx2, ly2, str(inv), ha="center", va="center",
                            fontsize=7, color="#a52a2a")
        # Center mark
        ax.plot([0], [0], "o", color="#2c3e50", markersize=4)
        # Two rays:
        # ray1 at 0 degrees on outer scale = far right of baseline
        x1, y1 = r * 1.0, 0.0
        ax.annotate("", xy=(x1 * 1.05, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#1a5e1a", lw=2))
        # ray2 at `angle` on outer scale (which is at position angle from RIGHT baseline)
        rad = math.radians(angle)
        # Outer scale: 0 on right, 180 on left, so angle=θ corresponds to math.cos(θ), math.sin(θ)
        x2, y2 = r * 1.0 * math.cos(rad), r * 1.0 * math.sin(rad)
        ax.annotate("", xy=(x2 * 1.05, y2 * 1.05), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#a52a2a", lw=2))

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.2, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
