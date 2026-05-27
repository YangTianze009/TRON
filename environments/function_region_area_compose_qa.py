"""
Function-Region-Area Composition QA — X25 (reference reasoning_val).

Renders a multi-curve plot (3-4 labeled curves like ``L_GIC``, ``L_AIC``,
``L_BIC`` plus a horizontal reference line). The prompt explicitly defines
named areas ``S_1 / S_2 / S_3`` by curve combinations, then asks which
named region is largest. Output is the region label.

Verbatim sample style (design notes §X25):

    Q-IDX 904 (Line+Scatter, stat):
      "Define S_1, S_2, S_3 as regions enclosed by combinations of curves
       L_GIC, L_AIC and the line u=5; which area is largest?" GT `S_2`

reference judge expects bare region label like ``S_1``.

2026-05-03 extension: added a new `smallest` mode alongside `largest`.
Both require image reading (compare three computed areas). A previously
considered `between_curves` mode was REJECTED — the textual definitions
trivially encoded the answer (image read not required), violating the
"don't make easier" constraint.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CURVE_LABEL_FAMILIES = [
    [("$L_{GIC}$", "L_GIC"), ("$L_{AIC}$", "L_AIC"),
     ("$L_{BIC}$", "L_BIC")],
    [("$f_1$", "f_1"), ("$f_2$", "f_2"), ("$f_3$", "f_3")],
    [("$\\alpha$", "alpha"), ("$\\beta$", "beta"), ("$\\gamma$", "gamma")],
]


class FunctionRegionAreaComposeQA(StandaloneVisualEnv):
    ENV_NAME = "function_region_area_compose"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            return {"n_curves": 3, "n_regions": 3}
        if level <= 6:
            return {"n_curves": 3, "n_regions": 3}
        return {"n_curves": 3, "n_regions": 3}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1879 + level * 173 + 79)
        np_rng = np.random.RandomState(rng.randint(0, 1_000_000))

        for _ in range(15):
            res = self._try(rng, np_rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, np_rng, cfg):
        x = np.linspace(0, 10, 100)
        # 3 curves: each is base + slope*x + amp*sin(...) — well-separated.
        family = list(rng.choice(_CURVE_LABEL_FAMILIES))
        rng.shuffle(family)
        family = family[:3]
        ref_line = rng.uniform(4, 7)  # u=5 style horizontal line

        curves = []
        for i in range(3):
            base = rng.uniform(2, 8)
            slope = rng.uniform(-0.4, 0.4)
            amp = rng.uniform(1.0, 2.5)
            phase = rng.uniform(0, 6.28)
            y = base + slope * x + amp * np.sin(phase + 0.4 * x)
            curves.append(y)

        # Define 3 named regions by pairwise combinations of curves vs ref_line:
        # S_1 = area where curve[0] > ref_line and curve[0] > curve[1]
        # S_2 = area where curve[1] > ref_line and curve[1] > curve[2]
        # S_3 = area where curve[2] > ref_line and curve[2] > curve[0]
        # Use trapezoidal sum over the overlap mask.
        dx = x[1] - x[0]

        def region_area(condition_mask, height_curve):
            # Area = sum over masked-true x of (height_curve - ref_line) * dx
            heights = np.maximum(height_curve - ref_line, 0)
            return float(np.sum(heights[condition_mask]) * dx)

        s1_mask = (curves[0] > ref_line) & (curves[0] > curves[1])
        s2_mask = (curves[1] > ref_line) & (curves[1] > curves[2])
        s3_mask = (curves[2] > ref_line) & (curves[2] > curves[0])
        s1 = region_area(s1_mask, curves[0])
        s2 = region_area(s2_mask, curves[1])
        s3 = region_area(s3_mask, curves[2])
        areas = [s1, s2, s3]
        # Need clear separation (>15% gap to next).
        sorted_areas = sorted(areas, reverse=True)
        if sorted_areas[0] - sorted_areas[1] < 0.15 * sorted_areas[0]:
            return None
        if sorted_areas[0] < 0.5:
            return None  # All regions tiny — skip.

        labels = ["S_1", "S_2", "S_3"]
        # Build prompt with explicit region definitions.
        ref_str = round(ref_line, 1)
        c0_label = family[0][1]
        c1_label = family[1][1]
        c2_label = family[2][1]
        defs = (
            f"Three curves are plotted: {c0_label}, {c1_label}, {c2_label}, "
            f"along with a horizontal reference line at u={ref_str}.\n"
            f"Define S_1 = area where {c0_label} is above {c1_label} and "
            f"above the line u={ref_str}.\n"
            f"Define S_2 = area where {c1_label} is above {c2_label} and "
            f"above the line u={ref_str}.\n"
            f"Define S_3 = area where {c2_label} is above {c0_label} and "
            f"above the line u={ref_str}.\n"
        )
        # Pick mode (largest / smallest) — both require reading the image
        # to compare areas. The earlier `between_curves` branch was removed
        # 2026-05-03 because the question text trivially encoded the answer
        # via curve-pair → region mapping in the definitions.
        mode = rng.choice(["largest", "smallest"])
        if mode == "largest":
            max_idx = areas.index(max(areas))
            answer = labels[max_idx]
            prompt = defs + "Which area is largest? Answer with one of: S_1, S_2, S_3."
        else:  # smallest
            # Need separation on the other end too.
            if sorted_areas[1] - sorted_areas[2] < 0.15 * sorted_areas[1]:
                return None
            min_idx = areas.index(min(areas))
            answer = labels[min_idx]
            prompt = defs + "Which area is smallest? Answer with one of: S_1, S_2, S_3."

        image = self._render(x, curves, family, ref_line)
        return prompt, answer, image

    def _render(self, x, curves, family, ref_line):
        style = self._random_style()
        palette = list(style["palette"])
        fig, ax = plt.subplots(figsize=(7 * style["figsize_scale"],
                                        4.6 * style["figsize_scale"]))
        for i, y in enumerate(curves):
            ax.plot(x, y, color=palette[i % len(palette)],
                    linewidth=style["line_width"],
                    label=family[i][0])
        ax.axhline(ref_line, color="#444", linestyle="--",
                   linewidth=1.2, label=f"u={round(ref_line, 1)}")
        ax.set_xlabel("x", fontsize=style["font_size_base"])
        ax.set_ylabel("u", fontsize=style["font_size_base"])
        ax.legend(fontsize=style["font_size_base"], loc=style["legend_loc"])
        ax.set_title("Curves and reference line",
                     fontsize=style["font_size_base"] + 1, pad=10)
        self._apply_style(fig, ax, style)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
