"""
Vector addition QA — 2 to 4 vectors on coordinate plane.

Diversity & difficulty redesign (2026-04-16):
- Level-aware sub_rng (not `self._rng`) controls vectors, qtype, colors.
- Text leakage fixed: at L>=4 vectors are drawn on image and NOT restated in
  question ("as shown in the diagram"). L0-L3 still include vector tuples in
  text for scaffolding.
- L0: scaffolded text, always 2 vectors, simple qtypes (x-component, larger
  component, magnitude).
- L9: 3-4 vectors, dot_product + direction + harder magnitudes; MCQ format.
- More qtypes: y-component, magnitude_comparison, perpendicular check, vector
  subtraction.
- 4+ phrasings per qtype.
"""
import math
import random
from typing import Dict, Optional, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_Q_PHRASINGS = {
    "resultant_magnitude": [
        "What is the magnitude of the resultant vector (shown as R on the diagram)? Round to 2 decimals.",
        "Compute |R| for the resultant of the plotted vectors. Round to 2 decimals.",
        "Find the length of the resultant vector R. Give the answer rounded to 2 decimals.",
        "Using the plotted vectors, what is the magnitude of their sum? Answer to 2 decimals.",
    ],
    "resultant_direction": [
        "What angle (in degrees) does the resultant R make with the positive x-axis? Round to 2 decimals.",
        "Find the direction angle of the resultant vector R measured from the +x axis, in degrees (2 decimals).",
        "Compute the angle (degrees, CCW from +x axis) of R shown on the diagram. 2 decimals.",
    ],
    "component_compare": [
        "Which component of the resultant R has larger absolute value: x or y? Answer with 'x' or 'y'.",
        "Does |Rx| or |Ry| have the greater magnitude? Answer with 'x' or 'y'.",
        "Is the resultant's x-component or y-component larger in absolute value? 'x' or 'y'.",
    ],
    "resultant_x": [
        "What is the x-component (Rx) of the resultant vector?",
        "Find Rx, the x-component of the plotted vectors' sum.",
        "Compute the x-component of the resultant vector.",
    ],
    "resultant_y": [
        "What is the y-component (Ry) of the resultant vector?",
        "Find Ry, the y-component of the plotted vectors' sum.",
        "Compute the y-component of the resultant vector.",
    ],
    "dot_product": [
        "Compute the dot product of the first two plotted vectors (u \u00b7 v).",
        "What is u \u00b7 v, using the vectors u and v shown in the diagram?",
        "Find the dot product u \u00b7 v from the plotted vectors.",
    ],
    "magnitude_first": [
        "What is the magnitude of vector u (the first plotted vector)? Round to 2 decimals.",
        "Find |u| for the first vector shown. Answer to 2 decimals.",
    ],
    "difference_x": [
        "What is the x-component of u - v, where u and v are the first two plotted vectors?",
        "Find (u - v)_x from the diagram.",
    ],
}

class VectorAdditionQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "vector_addition"

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return {"qtypes": ["resultant_x", "component_compare"],
                    "n_vec_range": (2, 2), "val_range": (-4, 4),
                    "scaffold_text": True, "is_mcq": False}
        if level <= 2:
            return {"qtypes": ["resultant_x", "resultant_y",
                               "component_compare", "magnitude_first"],
                    "n_vec_range": (2, 2), "val_range": (-5, 5),
                    "scaffold_text": True, "is_mcq": False}
        if level <= 3:
            return {"qtypes": ["resultant_magnitude", "resultant_x",
                               "resultant_y", "component_compare"],
                    "n_vec_range": (2, 3), "val_range": (-5, 5),
                    "scaffold_text": True, "is_mcq": False}
        if level <= 5:
            return {"qtypes": ["resultant_magnitude", "resultant_direction",
                               "resultant_x", "component_compare",
                               "difference_x"],
                    "n_vec_range": (2, 3), "val_range": (-6, 6),
                    "scaffold_text": False, "is_mcq": False}
        if level <= 7:
            return {"qtypes": ["resultant_magnitude", "resultant_direction",
                               "dot_product", "difference_x",
                               "component_compare"],
                    "n_vec_range": (3, 3), "val_range": (-7, 7),
                    "scaffold_text": False, "is_mcq": False}
        return {"qtypes": ["resultant_magnitude", "resultant_direction",
                           "dot_product", "difference_x",
                           "resultant_y"],
                "n_vec_range": (3, 4), "val_range": (-8, 8),
                "scaffold_text": False, "is_mcq": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in cfg["qtypes"]:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(20):
            r = self._try_generate(sub_rng, cfg, qtype, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, qtype, level):
        style = self._random_style()
        n_lo, n_hi = cfg["n_vec_range"]
        n_vec = rng.randint(n_lo, n_hi)
        vlo, vhi = cfg["val_range"]

        vectors = []
        for _ in range(n_vec):
            vx = rng.randint(vlo, vhi)
            vy = rng.randint(vlo, vhi)
            if vx == 0 and vy == 0:
                vx = 1
            vectors.append((vx, vy))

        rx = sum(v[0] for v in vectors)
        ry = sum(v[1] for v in vectors)
        mag = round(math.sqrt(rx ** 2 + ry ** 2), 2)
        direction = round(math.degrees(math.atan2(ry, rx)), 2)

        labels = ["u", "v", "w", "z"]
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        # Shuffle colors per-seed
        palette = list(style["palette"])
        rng.shuffle(palette)
        show_resultant = rng.choice([True, False]) or level >= 4

        # At L0-L3 vector components are in the question text already
        # (scaffold_text), so labels show coord tuple; at L4+ labels show
        # only the variable name to avoid answer leakage on the image.
        show_coords_in_label = cfg.get("scaffold_text", False)
        for i, (vx, vy) in enumerate(vectors):
            color = palette[i % len(palette)]
            ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->,head_width=0.35",
                                        color=color,
                                        linewidth=style["line_width"] + 0.6))
            # Label on arrow (near arrowhead to avoid clustering at origin)
            lx = vx * 0.72 + rng.uniform(-0.15, 0.15)
            ly = vy * 0.72 + rng.uniform(0.2, 0.45)
            if show_coords_in_label:
                label_text = f"{labels[i]}=({vx},{vy})"
            else:
                label_text = labels[i]
            ax.text(lx, ly, label_text,
                    fontsize=style["font_size_base"] + 1, color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=style["bg_color"],
                              ec=color, alpha=0.82))

        if show_resultant:
            resultant_color = palette[(len(vectors)) % len(palette)]
            ax.annotate("", xy=(rx, ry), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->,head_width=0.4",
                                        color=resultant_color,
                                        linewidth=style["line_width"] + 1.0,
                                        linestyle="--"))
            # Don't print the resultant coords on the image at L4+ — that
            # would leak the answer for Rx/Ry/magnitude/direction qtypes.
            # BUGFIX 2026-04-24: also gate on qtype — never leak R coords when the
            # question asks for Rx/Ry/|R|/direction/component_compare.
            _leak_qtypes = {"resultant_x", "resultant_y", "resultant_magnitude",
                            "resultant_direction", "component_compare"}
            if show_coords_in_label and qtype not in _leak_qtypes:
                r_label = f"R = ({rx},{ry})"
            else:
                r_label = "R"
            ax.text(rx * 0.72 + 0.3, ry * 0.72 - 0.7,
                    r_label,
                    fontsize=style["font_size_base"] + 1,
                    color=resultant_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff9c4",
                              ec=resultant_color, alpha=0.9))

        lim = max(abs(rx), abs(ry),
                  max(max(abs(v[0]), abs(v[1])) for v in vectors)) + 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_aspect("equal")
        self._apply_style(fig, ax, style)
        title = rng.choice(["Vector Addition", "Vector Diagram", "Plot of Vectors", "Vectors"])
        ax.set_title(title, fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        # Build question text
        vstr = ", ".join(f"{labels[i]}=({v[0]},{v[1]})"
                         for i, v in enumerate(vectors))
        phrasing = rng.choice(_Q_PHRASINGS.get(qtype,
                                               [f"Find the answer ({qtype})."]))

        scaffold = cfg.get("scaffold_text", False)
        if scaffold:
            prefix = f"Given vectors {vstr} (as plotted): "
        else:
            prefix = "The diagram shows several vectors. "

        q = prefix + phrasing

        # Compute answer
        if qtype == "resultant_magnitude":
            answer_val = mag
            answer_str = str(mag)
        elif qtype == "resultant_direction":
            answer_val = direction
            answer_str = str(direction)
        elif qtype == "component_compare":
            answer_str = "x" if abs(rx) >= abs(ry) else "y"
            answer_val = answer_str
        elif qtype == "resultant_x":
            answer_val = rx
            answer_str = str(rx)
        elif qtype == "resultant_y":
            answer_val = ry
            answer_str = str(ry)
        elif qtype == "dot_product":
            dp = vectors[0][0] * vectors[1][0] + vectors[0][1] * vectors[1][1]
            answer_val = dp
            answer_str = str(dp)
        elif qtype == "magnitude_first":
            m = round(math.sqrt(vectors[0][0] ** 2 + vectors[0][1] ** 2), 2)
            answer_val = m
            answer_str = str(m)
        elif qtype == "difference_x":
            dv = vectors[0][0] - vectors[1][0]
            answer_val = dv
            answer_str = str(dv)
        else:
            return None

        # MCQ wrapping
        if cfg.get("is_mcq", False) and qtype not in ("component_compare",):
            try:
                correct_num = float(answer_str)
            except ValueError:
                correct_num = None
            if correct_num is not None:
                distractors = set()
                deltas = [-5, -3, -2, -1, 1, 2, 3, 5, 7, -7]
                for d in rng.sample(deltas, k=len(deltas)):
                    cand = round(correct_num + d, 2)
                    if cand != correct_num:
                        distractors.add(cand)
                    if len(distractors) >= 3:
                        break
                if len(distractors) >= 3:
                    opts = [correct_num] + list(distractors)[:3]
                    rng.shuffle(opts)
                    correct_idx = opts.index(correct_num)
                    letter = chr(ord("A") + correct_idx)
                    def _fmt(v):
                        if v == int(v):
                            return str(int(v))
                        return f"{v:.2f}"
                    q = (q + "\n" + "\n".join(
                        f"  ({chr(ord('A')+i)}) {_fmt(opts[i])}" for i in range(4))
                        + "\nAnswer with a single letter.")
                    answer_str = letter

        return q, answer_str, img
