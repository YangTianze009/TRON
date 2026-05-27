"""
Optical Illusion Lines QA (D61).

Reference task:
  qid 208 (HS MCQ): "Are three lines in figure the same length? choices: (A)
   Yes (B) No." Ans: A.
  qid 209 (HS MCQ): "Is the distance between the two orange lines as long as
   the distance between the two purple lines? choices: (A) Yes (B) No." Ans: A.
  qid 210 (HS MCQ): "Are the red line and the blue line the same length?
   choices: (A) Yes (B) No." Ans: A.

Renders an optical-illusion configuration where two or more lines look
visually different but are in fact the same length (Müller-Lyer / Ponzo /
Sander). Asks Yes/No whether the lines are the same length.

The variants are generated such that ground truth is mostly Yes (illusion
typical), but distractor seeds also produce true differences (genuinely
different lengths) so the model can't shortcut.

Verifier: Yes/No.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Are the two highlighted lines in the figure the same length? Answer Yes or No. Place answer in <answer>...</answer>.",
    "Look at the two colored lines in the image. Are they the same length? Answer Yes or No in <answer>...</answer>.",
    "Compare the lengths of the highlighted line segments. Are they the same? Answer Yes or No in <answer>...</answer>.",
    "In the diagram, are the {color1} line and the {color2} line of equal length? Answer Yes or No in <answer>...</answer>.",
    "Are the {color1} and {color2} segments shown in the image of identical length? Answer Yes or No in <answer>...</answer>.",
    "Determine whether the two highlighted line segments have the same length. Answer Yes or No in <answer>...</answer>.",
    "Is the {color1} line as long as the {color2} line? Answer Yes or No in <answer>...</answer>.",
    "Examine the two colored lines. Do they have the same length? Answer Yes or No in <answer>...</answer>.",
]


class OpticalIllusionLinesQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "optical_illusion_lines"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        return {"level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        rng = random.Random((self.seed or 0) * 6133 + level * 71 + 101)

        # Choose illusion type and ground-truth (same length or not)
        illusion = rng.choice(["muller_lyer", "ponzo", "sander"])
        same_length = rng.random() < 0.5

        # Two segment lengths (top/bottom or left/right)
        L1 = 4.0
        if same_length:
            L2 = 4.0
        else:
            # Use a clearly different length so it's not borderline
            delta = rng.choice([-1.5, -1.0, 1.0, 1.5])
            L2 = L1 + delta

        color1 = "red"
        color2 = "blue"
        if rng.random() < 0.5:
            color1, color2 = "orange", "purple"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(color1=color1, color2=color2)
        answer = "Yes" if same_length else "No"
        img = self._render(illusion, L1, L2, color1, color2, rng)
        return question, answer, img

    def _color_hex(self, name):
        return {
            "red": "#e53e3e", "blue": "#1d4ed8",
            "orange": "#ea7c1c", "purple": "#7e22ce",
        }.get(name, "#1a1a1a")

    def _render(self, illusion, L1, L2, color1, color2, rng) -> Image.Image:
        c1 = self._color_hex(color1)
        c2 = self._color_hex(color2)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)

        if illusion == "muller_lyer":
            # Two horizontal lines with arrowheads in opposite directions
            y1 = 5.0
            y2 = 2.0
            # First line at center 5
            x1l = 5 - L1 / 2
            x1r = 5 + L1 / 2
            ax.plot([x1l, x1r], [y1, y1], color=c1, linewidth=3)
            self._draw_arrowheads(ax, x1l, y1, x1r, y1, c1, "out")
            # Second line at center 5
            x2l = 5 - L2 / 2
            x2r = 5 + L2 / 2
            ax.plot([x2l, x2r], [y2, y2], color=c2, linewidth=3)
            self._draw_arrowheads(ax, x2l, y2, x2r, y2, c2, "in")
        elif illusion == "ponzo":
            # Two parallel converging lines (like railway tracks) and two
            # horizontal segments inside (one near top, one near bottom).
            ax.plot([2, 4], [0.5, 6.5], color="#888", linewidth=1.5)
            ax.plot([8, 6], [0.5, 6.5], color="#888", linewidth=1.5)
            # Segment 1 near top
            cx1 = 5
            ax.plot([cx1 - L1 / 2, cx1 + L1 / 2], [5.0, 5.0], color=c1,
                    linewidth=4)
            # Segment 2 near bottom
            cx2 = 5
            ax.plot([cx2 - L2 / 2, cx2 + L2 / 2], [2.0, 2.0], color=c2,
                    linewidth=4)
        else:  # sander parallelogram-ish
            # Parallelograms with diagonals
            # Two diagonals of the same length, drawn inside skewed rects
            cy = 3.5
            # First diagonal at left
            ax.plot([1, 1 + L1], [cy + 0.5, cy + 0.5 + 0.4], color=c1,
                    linewidth=3)
            # Second diagonal at right
            ax.plot([10 - L2, 10], [cy - 1.0, cy - 0.5], color=c2,
                    linewidth=3)
            # Reference frames
            ax.plot([0.7, 5.3, 5.3, 0.7, 0.7],
                    [cy + 1.5, cy + 1.0, cy - 0.5, cy + 0.0, cy + 1.5],
                    color="#888", linewidth=1)
            ax.plot([4.7, 9.3, 9.3, 4.7, 4.7],
                    [cy + 1.5, cy + 0.5, cy - 1.0, cy + 0.0, cy + 1.5],
                    color="#888", linewidth=1)

        ax.set_title("Compare the highlighted line lengths.",
                     fontsize=13, pad=8)
        return self.fig_to_pil(fig, dpi=120)

    def _draw_arrowheads(self, ax, x1, y1, x2, y2, color, direction):
        """Draw arrowheads at both ends of a horizontal line.
        direction: 'out' = arrowheads point outward (>--<);
                   'in'  = arrowheads point inward  (<-->).
        """
        h = 0.4
        # Left end
        if direction == "out":
            ax.plot([x1, x1 - h], [y1, y1 + h], color=color, linewidth=2)
            ax.plot([x1, x1 - h], [y1, y1 - h], color=color, linewidth=2)
            ax.plot([x2, x2 + h], [y2, y2 + h], color=color, linewidth=2)
            ax.plot([x2, x2 + h], [y2, y2 - h], color=color, linewidth=2)
        else:
            ax.plot([x1, x1 + h], [y1, y1 + h], color=color, linewidth=2)
            ax.plot([x1, x1 + h], [y1, y1 - h], color=color, linewidth=2)
            ax.plot([x2, x2 - h], [y2, y2 + h], color=color, linewidth=2)
            ax.plot([x2, x2 - h], [y2, y2 - h], color=color, linewidth=2)


if __name__ == "__main__":
    env = OpticalIllusionLinesQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        print(f"L{level}: yes={ans.count('Yes')} no={ans.count('No')}")
