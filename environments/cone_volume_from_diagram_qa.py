"""
Cone Volume From Diagram QA — pure single-letter MCQ A-D / A-E.

2026-05-05 R5 P3 HARDENED: complete rewrite. Was 100% saturated because
distractors were too loose (random multiples) and answer was numeric. New
design forces precise computation with π = 3.14 and adds tight off-by-
formula-coefficient distractors plus the textbook "E. No correct answer" trap.

Per-level design:
  L0/L1 — 4-MCQ. Right cone with r and h labeled directly. V = (1/3)πr²h.
          Use π = 3.14, integer dims, 2-decimal answer.
  L2/L3 — 5-MCQ + 10% trap. Adds answers that confuse V_cone with V_cyl.
  L4/L5 — 5-MCQ + 15% trap. Cone with slant height labeled (must derive h
          via h² = l² − r²) for half the problems.
  L6/L7 — 5-MCQ + 25% trap. Volume in mL (capacity unit conversion: 1 cm³
          = 1 mL). Decimal dimensions on some.
  L8/L9 — 5-MCQ + 40% trap. Decimal dimensions, ask volume in **liters**
          (cm³ → mL → L), tight ±5% distractors. Often "E. No correct".

Dimensions are labeled ON THE IMAGE (the question stem doesn't restate
them — the model must read from the figure).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


PI = 3.14
_TITLES = [
    "Right circular cone (dimensions labeled)",
    "Cone — read dimensions from figure",
    "Cone (3D view)",
    "Cone diagram",
]


def _fmt(v: float, decimals: int = 2) -> str:
    """Format a number — int when possible, else `decimals` decimal places."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


class ConeVolumeFromDiagramQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a labeled cone, pick the volume option."""

    ENV_NAME = "cone_volume_from_diagram"
    ALLOW_ROTATION = False  # dimension labels are orientation-sensitive

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "label_mode": "rh",
                "unit": "cm3",
                "decimals_in_dims": False,
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "label_mode": "rh",
                "unit": "cm3",
                "decimals_in_dims": False,
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "label_mode": "rl_or_rh",  # half slant, half height
                "unit": "cm3",
                "decimals_in_dims": False,
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": False,
            }
        if level <= 7:
            return {
                "label_mode": "rl_or_rh",
                "unit": "mL",  # capacity conversion (1 cm³ = 1 mL)
                "decimals_in_dims": True,
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        return {
            "label_mode": "rl_or_rh",
            "unit": "L",  # cm³ → mL → L (÷1000)
            "decimals_in_dims": True,
            "use_5_options": True,
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1009)

        for attempt in range(15):
            try:
                res = self._try_generate(sub_rng, cfg, level, attempt)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level, attempt):
        # Sample (r, h) — Pythagorean triples ensure integer slant if needed
        triples = [(3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10),
                   (5, 12, 13), (12, 5, 13), (9, 12, 15), (12, 9, 15),
                   (8, 15, 17), (15, 8, 17), (7, 24, 25)]

        if cfg["decimals_in_dims"]:
            r = rng.choice([2.5, 3.5, 4.5, 5.0, 6.0])
            h = rng.choice([4.5, 5.0, 6.0, 7.5, 9.0])
            slant = math.sqrt(r * r + h * h)
            label_h_or_l = "h"  # for decimal mode, always use h not slant
        else:
            if cfg["label_mode"] == "rl_or_rh" and rng.random() < 0.5:
                r, h, slant = rng.choice(triples)
                label_h_or_l = "l"
            else:
                r, h = rng.choice([(3, 4), (2, 5), (3, 6), (4, 3),
                                   (5, 6), (2, 9), (4, 9), (6, 5), (3, 8),
                                   (5, 9), (4, 6), (3, 9), (5, 8)])
                slant = math.sqrt(r * r + h * h)
                label_h_or_l = "h"

        # Compute volume in cm³ first
        v_cm3 = PI * r * r * h / 3
        # Convert based on unit
        if cfg["unit"] == "cm3":
            true_value = v_cm3
            unit_str = "cm³"
        elif cfg["unit"] == "mL":
            true_value = v_cm3  # 1 cm³ = 1 mL
            unit_str = "mL"
        elif cfg["unit"] == "L":
            true_value = v_cm3 / 1000.0  # cm³ → L
            unit_str = "L"
        else:
            return None

        if true_value <= 0:
            return None

        # Build distractors (compute in same unit as true_value)
        distractors = self._make_distractors(rng, r, h, slant, cfg, true_value)
        if distractors is None or len(distractors) < 4:
            return None

        # Format
        if cfg["unit"] == "L":
            true_str = _fmt(true_value, decimals=4)
        else:
            true_str = _fmt(true_value, decimals=2)

        distr_strs = []
        seen = {true_str}
        for d in distractors:
            ds = _fmt(d, decimals=(4 if cfg["unit"] == "L" else 2))
            if ds in seen:
                continue
            seen.add(ds)
            distr_strs.append(ds)

        if len(distr_strs) < 4:
            return None

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False
        if use_trap:
            chosen = distr_strs[:n_value]
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distr_strs[:n_distractor]
            options = [true_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_str)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]} {unit_str}"
            for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Build stem
        decimals = "4 decimal places" if cfg["unit"] == "L" else "2 decimal places"
        stems = [
            (
                f"As shown in the diagram, the figure is a right circular cone "
                f"with its dimensions labeled (in cm). Compute the volume of "
                f"the cone in {unit_str}. Use π = 3.14 and round to {decimals}. ( )"
            ),
            (
                f"From the diagram, read the dimensions of the cone (in cm) "
                f"and compute its volume in {unit_str}. "
                f"Use π = 3.14 and round to {decimals}. ( )"
            ),
            (
                f"The figure shows a right circular cone with its dimensions "
                f"labeled (in cm). What is the cone's volume in {unit_str}? "
                f"Use π = 3.14, round to {decimals}. ( )"
            ),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        # Render — decide whether to label slant or height
        image = self._render_cone(rng, r, h, slant, label_h_or_l)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _make_distractors(self, rng, r, h, slant, cfg, true_value):
        """Produce confusable wrong values, expressed in same unit as true_value."""
        unit_factor = 1.0 if cfg["unit"] in ("cm3", "mL") else 1 / 1000.0

        candidates = []
        # 1. Forgot the 1/3 (cylinder volume instead)
        candidates.append(PI * r * r * h * unit_factor)
        # 2. Used 2*pi*r*h (lateral SA mistake)
        candidates.append(2 * PI * r * h * unit_factor)
        # 3. Used pi*r²*l (slant instead of height)
        candidates.append(PI * r * r * slant / 3 * unit_factor)
        # 4. Used l² instead of r² (squared the wrong thing)
        candidates.append(PI * slant * slant * h / 3 * unit_factor)
        # 5. Forgot to square radius
        candidates.append(PI * r * h / 3 * unit_factor)
        # 6. Used pi=3 (common Chinese textbook approximation)
        candidates.append(3.0 * r * r * h / 3 * unit_factor)
        # 7. Off by unit conversion — wrong direction
        if cfg["unit"] == "L":
            # Forgot to divide by 1000
            candidates.append(PI * r * r * h / 3)
            # Divided by wrong amount
            candidates.append(PI * r * r * h / 3 / 100)
            candidates.append(PI * r * r * h / 3 / 10000)
        if cfg["unit"] == "mL":
            # Treated as L by mistake (×1000)
            candidates.append(PI * r * r * h / 3 / 1000)

        # 8. ±5% / ±10% perturbations for tight mode
        if cfg["tight_distractors"]:
            for delta in (0.95, 0.97, 1.03, 1.05, 0.93, 1.07):
                candidates.append(true_value * delta)
        else:
            for delta in (0.7, 0.5, 1.5, 2.0):
                candidates.append(true_value * delta)

        # Filter
        good = []
        for c in candidates:
            if c is None or c <= 0:
                continue
            if abs(c - true_value) < max(0.005, true_value * 0.005):
                continue
            if any(abs(c - g) < max(0.005, max(c, g) * 0.003) for g in good):
                continue
            good.append(c)

        rng.shuffle(good)
        return good[:6]

    # ------------------------------------------------------------------ #
    # Rendering — labeled axonometric cone
    # ------------------------------------------------------------------ #
    def _render_cone(self, rng, r, h, slant, label_h_or_l) -> Image.Image:
        style = self._random_style()
        fig = plt.figure(figsize=(6.8, 6.8))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(style["bg_color"])

        theta = np.linspace(0, 2 * np.pi, 60)
        zl = np.linspace(0, h, 30)
        theta_g, z_g = np.meshgrid(theta, zl)
        r_g = r * (1 - z_g / h)
        surf_color = rng.choice(style["palette"])
        ax.plot_surface(r_g * np.cos(theta_g), r_g * np.sin(theta_g), z_g,
                        alpha=0.25, color=surf_color)

        # Base outline
        ax.plot(r * np.cos(theta), r * np.sin(theta), 0,
                color="black", lw=1.5)
        # Apex
        ax.scatter([0], [0], [h], color="black", s=30)
        # Radius line in red
        ax.plot([0, r], [0, 0], [0, 0], color="red", lw=2)
        # Height line (axis) in green dashed
        ax.plot([0, 0], [0, 0], [0, h], color="green", lw=2, linestyle="--")
        # Slant line in blue
        ax.plot([r, 0], [0, 0], [0, h], color="blue", lw=2)

        # Labels
        ax.text(r + 0.4, 0, 0, f"r = {_fmt(r)}", fontsize=13,
                color="red", fontweight="bold")
        if label_h_or_l == "h":
            ax.text(0.4, 0, h / 2, f"h = {_fmt(h)}", fontsize=13,
                    color="green", fontweight="bold")
        else:
            ax.text(r / 2 + 0.3, 0, h / 2 + 0.4,
                    f"l = {_fmt(slant)}", fontsize=13,
                    color="blue", fontweight="bold")

        m = max(r, h) * 1.05
        ax.set_xlim([-m, m])
        ax.set_ylim([-m, m])
        ax.set_zlim([-1, h + 1])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.view_init(elev=rng.randint(15, 28),
                     azim=rng.choice([30, 40, 45, 55, 60]))
        ax.set_title(rng.choice(_TITLES),
                     fontsize=12, fontweight="bold", pad=10)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=130)


if __name__ == "__main__":
    env = ConeVolumeFromDiagramQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
