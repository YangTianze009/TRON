"""
Contour Region Area QA — X24 (reference reasoning_val).

Renders 3 side-by-side contour plots (matplotlib `contourf`) over different
2D scalar fields with continuous colorbar. Asks "which subplot has the
most/least area with value < threshold?" Output is the position word
(`left` / `middle` / `right` or `first` / `second` / `third`).

Verbatim sample style (design notes Number-in-Chart §X24):

    Q-IDX 977 (Contour Plot, math):
      "Which subplot has the most area with a value less than 30 based on
       the continuous color bar legend, left, middle, or right?" GT `left`

    Q-IDX 576 (Contour Plot, eess):
      "Across the three charts, which one shows the highest spatial
       coverage of energy values above 25 MeV? Answer with first, second,
       or third" GT `Third`

reference judge expects bare position word.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_FIELD_KINDS = ["gaussian", "poly", "sin_cos", "exp_decay"]


def _field(kind: str, X: np.ndarray, Y: np.ndarray, params: Dict) -> np.ndarray:
    if kind == "gaussian":
        cx, cy = params["cx"], params["cy"]
        s = params["sigma"]
        amp = params["amp"]
        return amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    if kind == "poly":
        a, b = params["a"], params["b"]
        return a * (X * X) + b * (Y * Y) + params["c"]
    if kind == "sin_cos":
        a = params["a"]
        return a * (np.sin(X) + np.cos(Y) + 2)
    if kind == "exp_decay":
        return params["amp"] * np.exp(-(X * X + Y * Y) / params["s"])
    return X * 0


class ContourRegionAreaQA(StandaloneVisualEnv):
    ENV_NAME = "contour_region_area"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_panels": 3, "field_choices": ["gaussian", "exp_decay"]}
        if level <= 5:
            return {"n_panels": 3, "field_choices": ["gaussian", "exp_decay",
                                                     "sin_cos"]}
        return {"n_panels": 3, "field_choices": ["gaussian", "exp_decay",
                                                  "sin_cos", "poly"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1741 + level * 137 + 59)

        for _ in range(15):
            res = self._try(rng, cfg)
            if res is not None:
                return res
        return None

    def _try(self, rng, cfg):
        n_panels = cfg["n_panels"]
        # Generate three distinct scalar fields and compute area-under-T.
        x = np.linspace(-3, 3, 80)
        y = np.linspace(-3, 3, 80)
        X, Y = np.meshgrid(x, y)
        fields = []
        for _ in range(n_panels):
            kind = rng.choice(cfg["field_choices"])
            params = {
                "cx": rng.uniform(-1.5, 1.5),
                "cy": rng.uniform(-1.5, 1.5),
                "sigma": rng.uniform(0.7, 1.6),
                "amp": rng.uniform(20, 50),
                "a": rng.uniform(0.5, 2.0),
                "b": rng.uniform(0.5, 2.0),
                "c": rng.uniform(0, 10),
                "s": rng.uniform(2, 5),
            }
            Z = _field(kind, X, Y, params)
            fields.append(Z)

        # Choose threshold so areas-under-T are clearly different across
        # panels.
        all_vals = np.concatenate([f.ravel() for f in fields])
        T = float(np.percentile(all_vals, rng.choice([30, 40, 50, 60])))

        areas_under = [int(np.sum(f < T)) for f in fields]
        # Require clearly-separated areas (at least 15% gap to second-best).
        sorted_idx = sorted(range(n_panels), key=lambda i: areas_under[i])
        # most under T = highest area under T
        max_idx = sorted_idx[-1]
        # check gap to runner-up
        sorted_vals = sorted(areas_under)
        if sorted_vals[-1] - sorted_vals[-2] < 0.15 * sorted_vals[-1]:
            return None

        # a chart benchmark X24 uses BOTH `left/middle/right` (IDX 977) and
        # `first/second/third` (IDX 576). Alternate by seed bit so training
        # sees both vocabularies. (Subplots are letter-labelled (a)/(b)/(c)
        # to avoid leaking the position word in the title.)
        use_ordinal = rng.random() < 0.5
        position_words = (["first", "second", "third"] if use_ordinal
                          else ["left", "middle", "right"])
        vocab_phrase = ("first, second, or third" if use_ordinal
                        else "left, middle, or right")
        answer = position_words[max_idx]

        # Question template alternates between "most" and "least" + "less"/"more than".
        if rng.random() < 0.5:
            stem = (f"Which subplot has the most area with a value less than "
                    f"{int(T)} based on the continuous color bar?")
            ground_truth = answer
        else:
            # most area > T → which has highest area where val > T
            areas_over = [int(np.sum(f > T)) for f in fields]
            sorted_over = sorted(range(n_panels), key=lambda i: areas_over[i])
            sv = sorted([areas_over[i] for i in range(n_panels)])
            if sv[-1] - sv[-2] < 0.15 * sv[-1]:
                return None
            ground_truth = position_words[sorted_over[-1]]
            stem = (f"Across the three charts, which one shows the highest "
                    f"spatial coverage of values greater than {int(T)} based "
                    f"on the continuous color bar?")

        prompt = (
            f"{stem} Answer with one word: {vocab_phrase}."
        )
        image = self._render(fields, X, Y)
        return prompt, ground_truth, image

    def _render(self, fields, X, Y):
        style = self._random_style()
        n = len(fields)
        fig, axes = plt.subplots(
            1, n, figsize=(4.5 * n * style["figsize_scale"],
                           4.4 * style["figsize_scale"]))
        # Shared color scale across all panels for fair area comparison.
        vmin = min(float(f.min()) for f in fields)
        vmax = max(float(f.max()) for f in fields)
        # Letter labels only — DO NOT leak position word in title (model
        # would just read it).
        panel_letters = ["(a)", "(b)", "(c)"]
        for i, ax in enumerate(axes):
            cs = ax.contourf(X, Y, fields[i], levels=20,
                             cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(panel_letters[i],
                         fontsize=style["font_size_base"] + 1)
            ax.set_xlabel("x", fontsize=style["font_size_base"])
            ax.set_ylabel("y", fontsize=style["font_size_base"])
        cbar = fig.colorbar(cs, ax=axes, shrink=0.85)
        cbar.set_label("value")
        fig.patch.set_facecolor(style["bg_color"])
        return self.fig_to_pil(fig, dpi=style["dpi"])
