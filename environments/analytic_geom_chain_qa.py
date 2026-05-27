"""
Analytic Geometry Chain QA (v4 G7, for image-heavy analytic geometry).

Targets:

Task: coordinate-geometry chain problems on a cartesian plane — given 2-3
points, compute distance / midpoint / slope / line equation.

Reward: numeric within 1% relative tolerance, or exact match on
midpoint tuple / slope fraction.

Level axes:
  A) Number of operations: 1 at L0 -> 3 at L7+
  B) Includes symbolic answer at L6+ (like "y = 2x + 3" vs numeric)
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES_DISTANCE = [
    "Two points A{A} and B{B} are plotted on the cartesian plane. Compute the distance between A and B. Round to 2 decimal places. Put the numeric value in <answer>...</answer>.",
    "Find the distance between A{A} and B{B} on the plane. Round to 2 decimal places. Put in <answer>...</answer>.",
    "Given A{A} and B{B}, compute |AB|. Round to 2 decimal places; put in <answer>...</answer>.",
    "Calculate the Euclidean distance from A{A} to B{B}. Round to 2dp; put in <answer>...</answer>.",
    "Distance between A{A} and B{B}? Round to 2 decimal places. Put in <answer>...</answer>.",
    "Compute |AB| where A{A} and B{B}. Round to 2dp. Put in <answer>...</answer>.",
    "Find the segment length AB where A{A}, B{B}. Round to 2dp. Put in <answer>...</answer>.",
    "Given A{A}, B{B}, compute AB distance (2dp). Put in <answer>...</answer>.",
    "Distance from A{A} to B{B}? (2dp) Put in <answer>...</answer>.",
    "|AB| with A{A}, B{B}? 2dp in <answer>...</answer>.",
    "Compute the distance from A{A} to B{B}. Round. Put in <answer>...</answer>.",
    "A is at {A}, B at {B}. Compute distance (2dp) in <answer>...</answer>.",
    "Find distance AB where A{A}, B{B}. 2dp in <answer>...</answer>.",
    "The points are A{A} and B{B}. What is |AB|? 2dp in <answer>...</answer>.",
    "Given the points A{A} and B{B}, find the distance. Round to 2dp. Put in <answer>...</answer>.",
    "Distance AB for A{A}, B{B}? Round to 2dp in <answer>...</answer>.",
]

_TEMPLATES_MIDPOINT = [
    "Find the midpoint of segment AB where A{A} and B{B}. Put as '(x, y)' in <answer>...</answer>.",
    "Given A{A}, B{B}, compute the midpoint M. Format '(x, y)'. Put in <answer>...</answer>.",
    "Midpoint of A{A} and B{B}? Format '(x, y)' in <answer>...</answer>.",
    "Compute midpoint of AB: A{A}, B{B}. Put '(x, y)' in <answer>...</answer>.",
    "The midpoint of segment AB where A{A}, B{B}? Put '(x, y)' in <answer>...</answer>.",
    "Find M = midpoint(A, B) for A{A}, B{B}. Format '(x, y)'. Put in <answer>...</answer>.",
    "Given A{A}, B{B}, midpoint? '(x, y)' in <answer>...</answer>.",
    "Midpoint of {A} and {B}? Put '(x, y)' in <answer>...</answer>.",
    "Compute midpoint M of A{A} and B{B}. Put '(x, y)' in <answer>...</answer>.",
    "Midpoint (x, y) of AB, with A{A}, B{B}? Put in <answer>...</answer>.",
    "Find midpoint coordinates for A{A}, B{B}. Put '(x, y)' in <answer>...</answer>.",
    "Midpoint of segment between A{A} and B{B}? Format '(x, y)' in <answer>...</answer>.",
    "Compute mid-segment point of A{A}, B{B}. Put '(x, y)' in <answer>...</answer>.",
    "Find (x, y) midpoint of A{A}, B{B}. Put in <answer>...</answer>.",
    "M is midpoint of A{A} and B{B}. M = ? Put '(x, y)' in <answer>...</answer>.",
    "Midpoint of AB (A{A}, B{B})? '(x, y)' in <answer>...</answer>.",
]

_TEMPLATES_SLOPE = [
    "Given A{A} and B{B}, compute the slope of line AB. Express as a decimal (2dp) or fraction. Put in <answer>...</answer>.",
    "Find slope of line through A{A} and B{B}. Put decimal/fraction in <answer>...</answer>.",
    "Slope of AB where A{A}, B{B}? Put in <answer>...</answer>.",
    "Compute the slope (rise over run) of line AB: A{A}, B{B}. Put in <answer>...</answer>.",
    "Slope of line from A{A} to B{B}? Decimal or fraction in <answer>...</answer>.",
    "Line slope AB (A{A}, B{B})? Put in <answer>...</answer>.",
    "Given A{A}, B{B}, find line slope. Put in <answer>...</answer>.",
    "Compute slope AB: A{A}, B{B}. Put in <answer>...</answer>.",
    "Slope from A{A} to B{B}? Put in <answer>...</answer>.",
    "Line through A{A}, B{B} has what slope? Put in <answer>...</answer>.",
    "Slope of AB (A{A}, B{B})? Put in <answer>...</answer>.",
    "What is the slope? A{A}, B{B}. Put in <answer>...</answer>.",
    "Slope AB with A{A}, B{B}? Put in <answer>...</answer>.",
    "Find slope m of line AB: A{A}, B{B}. Put in <answer>...</answer>.",
    "m = slope(A, B) where A{A}, B{B}. Put value in <answer>...</answer>.",
    "Given A{A}, B{B}: slope m? Put in <answer>...</answer>.",
]

class AnalyticGeomChainQA(StandaloneVisualEnv):
    ENV_NAME = "analytic_geom_chain"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            qtypes = ["distance"]
        elif level <= 5:
            qtypes = ["distance", "midpoint"]
        elif level <= 7:
            qtypes = ["distance", "midpoint", "slope"]
        else:
            qtypes = ["distance", "midpoint", "slope", "triangle_area"]
        coord_range = min(10, 5 + level)
        # Hide coords from question text at higher levels — model must read
        # them off the labelled grid instead of arithmetic-only solving.
        # L0-L3: coords inlined in text (pure arithmetic)
        # L4-L6: coords NOT in text, labelled on image (forces image read)
        # L7-L9: coords NOT in text AND NOT in label (model estimates from
        #        gridline position).
        if level <= 3:
            coords_visibility = "text_and_label"
        elif level <= 6:
            coords_visibility = "label_only"
        else:
            coords_visibility = "estimate_from_grid"
        return {"qtypes": qtypes, "coord_range": coord_range,
                "coords_visibility": coords_visibility}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 149)
        self._primary_complexity_feature = level

        cr = cfg["coord_range"]
        A = (rng.randint(-cr, cr), rng.randint(-cr, cr))
        B = (rng.randint(-cr, cr), rng.randint(-cr, cr))
        while B == A:
            B = (rng.randint(-cr, cr), rng.randint(-cr, cr))
        # Optional third point for triangle_area at L8/L9
        C = (rng.randint(-cr, cr), rng.randint(-cr, cr))
        while C in (A, B):
            C = (rng.randint(-cr, cr), rng.randint(-cr, cr))

        qtype = rng.choice(cfg["qtypes"])
        sidx = (self.seed or 0) % 16
        vis = cfg["coords_visibility"]

        # _coord_str: what gets substituted into the template's "{A}" / "{B}"
        # slot, which appears IMMEDIATELY after a literal "A" / "B" character
        # (e.g. template "A{A}" becomes "A(2, 3)" or "A" depending on level).
        def _coord_str(p):
            if vis == "text_and_label":
                return f"{tuple(p)}"
            # Higher levels: leave the slot empty so the template just
            # mentions "A" / "B" without coords. Add a hint suffix in the
            # question wrapper below.
            return ""

        coord_hint = ""
        if vis != "text_and_label":
            coord_hint = (" (read the coordinates of A and B from the "
                          "labelled grid in the image)")
            if vis == "estimate_from_grid":
                coord_hint = (" (the points are labelled A, B on the grid; "
                              "estimate their coordinates from the gridlines)")

        if qtype == "distance":
            d = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
            answer = f"{round(d, 2)}"
            q = _TEMPLATES_DISTANCE[sidx].format(
                A=_coord_str(A), B=_coord_str(B)) + coord_hint
        elif qtype == "midpoint":
            mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
            mx_str = str(int(mx)) if mx == int(mx) else f"{mx}"
            my_str = str(int(my)) if my == int(my) else f"{my}"
            answer = f"({mx_str}, {my_str})"
            q = _TEMPLATES_MIDPOINT[sidx].format(
                A=_coord_str(A), B=_coord_str(B)) + coord_hint
        elif qtype == "slope":
            if B[0] == A[0]:
                answer = "undefined"
            else:
                m = (B[1] - A[1]) / (B[0] - A[0])
                answer = f"{round(m, 2)}"
            q = _TEMPLATES_SLOPE[sidx].format(
                A=_coord_str(A), B=_coord_str(B)) + coord_hint
        else:  # triangle_area (L8/L9)
            area = abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) +
                        C[0] * (A[1] - B[1])) / 2.0
            answer = f"{round(area, 2)}"
            q = (f"Three points A, B, C are plotted on the cartesian plane. "
                 f"Compute the area of triangle ABC. Round to 2 decimal "
                 f"places. Put in <answer>...</answer>." + coord_hint)
            img = self._render(A, B, rng, third=C, vis=vis)
            return q, answer, img

        img = self._render(A, B, rng, vis=vis)
        return q, answer, img

    def _render(self, A, B, rng, third=None, vis="text_and_label"):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        all_x = [A[0], B[0]]; all_y = [A[1], B[1]]
        if third is not None:
            all_x.append(third[0]); all_y.append(third[1])
        lim = max(abs(v) for v in (all_x + all_y)) + 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        # Axes + tick marks (essential at high levels for grid estimation)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xticks(range(-int(lim), int(lim) + 1, 1))
        ax.set_yticks(range(-int(lim), int(lim) + 1, 1))
        ax.grid(True, linestyle=":", alpha=0.5)

        def _label(pt, name, color):
            ax.plot(*pt, "o", color=color, markersize=10)
            if vis == "text_and_label":
                ax.text(pt[0] + 0.3, pt[1] + 0.3, f"{name}{tuple(pt)}",
                        fontsize=12, fontweight="bold", color=color)
            elif vis == "label_only":
                ax.text(pt[0] + 0.3, pt[1] + 0.3, f"{name}{tuple(pt)}",
                        fontsize=12, fontweight="bold", color=color)
            else:  # estimate_from_grid
                ax.text(pt[0] + 0.3, pt[1] + 0.3, name,
                        fontsize=14, fontweight="bold", color=color)

        _label(A, "A", "red")
        _label(B, "B", "blue")
        if third is not None:
            _label(third, "C", "green")
        # Connect points
        if third is None:
            ax.plot([A[0], B[0]], [A[1], B[1]], color="black", lw=1.5, linestyle="--")
        else:
            xs = [A[0], B[0], third[0], A[0]]
            ys = [A[1], B[1], third[1], A[1]]
            ax.plot(xs, ys, color="black", lw=1.5)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        if pred == gt:
            return True
        # tuple-style
        if gt.startswith("(") and gt.endswith(")"):
            try:
                pred_vals = [float(x.strip()) for x in
                             pred.strip("()").split(",")]
                gt_vals = [float(x.strip()) for x in
                           gt.strip("()").split(",")]
                if len(pred_vals) == len(gt_vals):
                    return all(abs(p - g) < 0.02 for p, g in
                               zip(pred_vals, gt_vals))
            except ValueError:
                pass
        # numeric
        try:
            return abs(float(pred) - float(gt)) < 0.02 or \
                abs(float(pred) - float(gt)) / max(abs(float(gt)), 1e-9) < 0.01
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_agc"
    os.makedirs(out_dir, exist_ok=True)
    env = AnalyticGeomChainQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 41
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[agc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/agc_s{s}_L{level}.png")
            print(f"[agc L{level} s{s}] A={env._answer}")
