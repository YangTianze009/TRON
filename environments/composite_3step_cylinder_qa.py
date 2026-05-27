"""
Composite 3-step geometry: cylinder volume from 3 sub-skills (read radius +
read height + apply formula), explicitly composed in one MCQ. Targets the
InadequateGeneralization gap on benchmarks that score per-composition.

Two presentation modes (controlled by level):
  - "atomic + composite": shows two sub-questions then asks the composite.
  - "composite only": single MCQ requiring all three sub-skills implicitly.
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


_TEMPLATES_STEM = [
    "As shown in the diagram, the cylinder has labeled radius and height. What is the volume of the cylinder in cm³? (Take π = 3.14)",
    "As shown in the figure, given the cylinder with the radius and height labeled, what is the volume of the cylinder in cm³? (π = 3.14)",
    "As shown in the diagram, the cylinder has the labeled base radius and height. What is the volume in cm³? (Take π = 3.14)",
    "As shown in the figure, what is the volume of the cylinder shown in cm³? (Take π as 3.14)",
    "As shown in the diagram, given the radius and height of the cylinder shown, what is the volume in cm³? (π = 3.14)",
    "As shown in the figure, the cylinder has the labeled dimensions. What is the volume of the cylinder in cm³? (π = 3.14)",
    "As shown in the diagram, what is the volume of the cylinder shown in cm³? (Use π = 3.14)",
    "As shown in the figure, the cylinder's radius and height are labeled. What is the volume in cm³? (Take π as 3.14)",
]


class Composite3StepCylinderQA(StandaloneVisualEnv):
    ENV_NAME = "composite_3step_cylinder"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # bigger numbers + decimals at higher level
        # 2026-05-04 R3: benchmark-sample-driven harden — at L8/L9 enable
        # E (No correct answer) trap in mcq mode (~25% GT=E).
        if level == 9:
            # 2026-05-04: L9 was 100% saturated even with open mode. Wider
            # decimal r + bigger h. R3: switch to mcq mode so E trap fires.
            r_pool = [4.25, 5.75, 7.25, 8.75, 10.25, 12.75, 15.25, 18.75]
            h_pool = list(range(15, 45))
            mode = "mcq"
        elif level <= 2:
            r_pool = [1, 2, 3, 4, 5]
            h_pool = [2, 3, 4, 5, 6, 8, 10]
            mode = "open" if level == 0 else "open"
        elif level <= 5:
            r_pool = [2, 3, 4, 5, 6, 7]
            h_pool = list(range(3, 16))
            mode = "open" if level % 2 == 0 else "mcq"
        else:
            r_pool = [2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8]
            h_pool = list(range(5, 21))
            mode = "mcq"
        return {"level": level, "r_pool": r_pool, "h_pool": h_pool,
                "mode": mode, "e_trap": level >= 8}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1031 + level * 79 + 5)

        r = rng.choice(cfg["r_pool"])
        h = rng.choice(cfg["h_pool"])
        V = 3.14 * r * r * h
        V_round = round(V, 2)

        if cfg["mode"] == "mcq":
            # build distractors
            d1 = round(2 * 3.14 * r * h, 2)  # circumference * h (common error)
            d2 = round(3.14 * r * h, 2)  # missing square
            d3 = round(3.14 * (r + 1) ** 2 * h, 2)  # off-by-one
            d4 = round(3.14 * (r - 0.5) ** 2 * h, 2)  # close near-miss
            # 2026-05-04 R3: benchmark-sample-driven harden — at L8/L9 with
            # ~25% chance, E (No correct answer) is GT. All visible options
            # are wrong distractors.
            if cfg.get("e_trap") and rng.random() < 0.25:
                wrong_opts = [d1, d2, d3, d4]
                wrong_opts = list(dict.fromkeys(wrong_opts))
                # remove any equal to V_round
                wrong_opts = [o for o in wrong_opts if abs(o - V_round) > 0.01]
                while len(wrong_opts) < 4:
                    fake = round(V_round * (1.3 + 0.1 * len(wrong_opts)), 2)
                    if fake != V_round and fake not in wrong_opts:
                        wrong_opts.append(fake)
                options = wrong_opts[:4]
                rng.shuffle(options)
                correct_letter = "E"
            else:
                options = [V_round, d1, d2, d3]
                # dedup, ensure all distinct
                options = list(dict.fromkeys(options))
                while len(options) < 4:
                    options.append(round(V_round + len(options) * 1.7, 2))
                options = options[:4]
                rng.shuffle(options)
                correct_letter = "ABCD"[options.index(V_round)]
            sidx = (self.seed or 0) % len(_TEMPLATES_STEM)
            stem = _TEMPLATES_STEM[sidx]
            opts_text = "\n".join(
                f"{chr(ord('A')+i)}. {options[i]}" for i in range(4)
            ) + "\nE. No correct answer"
            question = (
                f"{stem}\n\n{opts_text}\n\n"
                "Choose the correct option (A, B, C, D, or E)."
            )
            answer = correct_letter
        else:
            sidx = (self.seed or 0) % len(_TEMPLATES_STEM)
            question = _TEMPLATES_STEM[sidx]
            answer = f"{V_round:.2f}"

        img = self._render(r, h)
        return question, answer, img

    def _render(self, r, h) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Cylinder body
        # base ellipse
        from matplotlib.patches import Ellipse
        # scale for display
        cx = 0
        body_h = h * 0.4 + 0.5
        body_w = r * 0.5
        # back (top) ellipse
        e_top = Ellipse((cx, body_h), body_w * 2, body_w * 0.5,
                        facecolor="#e7eef9", edgecolor="#2c3e50", linewidth=1.5)
        # bottom ellipse (front-arc style)
        e_bot = Ellipse((cx, 0), body_w * 2, body_w * 0.5,
                        facecolor="#dbe5f5", edgecolor="#2c3e50", linewidth=1.5)
        # Side rectangle as fill
        ax.add_patch(plt.Rectangle((cx - body_w, 0), body_w * 2, body_h,
                                   facecolor="#e7eef9", edgecolor="none"))
        ax.add_patch(e_bot)
        ax.add_patch(e_top)
        # Side edges
        ax.plot([cx - body_w, cx - body_w], [0, body_h], color="#2c3e50", lw=1.5)
        ax.plot([cx + body_w, cx + body_w], [0, body_h], color="#2c3e50", lw=1.5)
        # Radius label
        ax.annotate("", xy=(cx + body_w, body_h), xytext=(cx, body_h),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(cx + body_w / 2, body_h + 0.18, f"r = {r}",
                ha="center", fontsize=12, color="#c0392b", fontweight="bold")
        # Height label
        ax.annotate("", xy=(cx + body_w + 0.5, 0),
                    xytext=(cx + body_w + 0.5, body_h),
                    arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
        ax.text(cx + body_w + 0.7, body_h / 2, f"h = {h}",
                ha="left", va="center", fontsize=12, color="#2c3e50",
                fontweight="bold")

        ax.set_xlim(-body_w - 1, body_w + 2)
        ax.set_ylim(-body_w * 0.5, body_h + 1)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
