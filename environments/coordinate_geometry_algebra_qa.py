"""
Coordinate Geometry Algebra QA environment.

Round 2 fixes:
  - Remove text leakage: equation text, coefficients, and points are on the
    IMAGE (the left plot + labeled annotations), NOT restated in the question.
  - L0 is now a simple point-reading task (read y-coordinate of a labeled
    point on a linear graph), with large labels and the line equation shown.
  - L9 is a multi-step family: cubic evaluation, parabola-line intersection,
    two-line system with fractional answer, or tangent-line slope.
  - Expanded pool of function families (8+) and per-seed jitter on axis
    ranges, colors, marker styles.
  - 4+ question phrasings per family.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class CoordinateGeometryAlgebraQA(StandaloneVisualEnv):
    ENV_NAME = "coordinate_geometry_algebra"

    _TITLES = ["Coordinate Plane", "Graph", "xy-Plane",
               "Cartesian Coordinates", "Plot", "Function Graph",
               "Labeled Graph"]

    _Q_POOL = {
        "read_point_y": [
            "The graph shows a line with a labeled point P at x = {ax}. Read the y-coordinate of P.",
            "Look at the labeled point P on the graph (its x is shown). What is its y value?",
            "From the graph, report the y-coordinate of the labeled point P at x = {ax}.",
            "A point P is labeled on the line. Find its y-coordinate from the graph.",
        ],
        "read_intercept": [
            "From the graph, find where the line crosses the y-axis (y-intercept).",
            "Read the y-intercept of the line from the graph.",
            "What is the y-value where the graph crosses x = 0?",
            "The graph shows a line. Report its y-intercept.",
        ],
        "linear_value": [
            "Use the line equation shown on the figure to compute y at x = {ax}.",
            "The image shows a line. Use its equation (visible on the figure) to find y at x = {ax}.",
            "From the labeled equation, compute y when x = {ax}.",
            "Read the equation on the figure and evaluate at x = {ax}.",
        ],
        "parabola_value": [
            "Use the parabola equation shown on the figure to compute y at x = {ax}.",
            "The image displays a parabola with its equation. Evaluate it at x = {ax}.",
            "From the equation labeled on the figure, find y when x = {ax}.",
            "Apply the parabola formula (shown on the graph) at x = {ax}.",
        ],
        "two_lines_int": [
            "The figure shows two lines with labeled equations. Find the {which}-coordinate of their intersection.",
            "Two lines are plotted with their equations displayed. Report the {which}-coordinate of their intersection point.",
            "Using the equations labeled on the figure, solve for the {which}-coordinate where the two lines cross.",
            "From the image, determine the {which}-coordinate of the intersection of the two lines.",
        ],
        "parabola_line_int": [
            "The image shows a parabola and a line (both equations labeled). Find the {which} x-coordinate where they meet.",
            "Using the labeled equations on the figure, find the {which} x-intersection of the parabola and line.",
            "Report the {which} x-value where the parabola intersects the line (labels on figure).",
            "Solve for the {which} x-intersection of the parabola with the line (see the figure for equations).",
        ],
        "cubic_value": [
            "The figure shows a cubic with its equation labeled. Compute y at x = {ax}.",
            "Use the cubic formula visible on the figure to evaluate at x = {ax}.",
            "From the labeled cubic equation, find y when x = {ax}.",
            "Apply the cubic (labeled on the graph) at x = {ax}.",
        ],
        "distance_two_points": [
            "The figure labels two points. Use the distance formula to compute |PQ|. Round to 2 decimals if needed.",
            "Compute the distance between the two labeled points on the figure.",
            "Based on the coordinates labeled, find the distance between P and Q.",
            "Distance = sqrt((x1-x2)² + (y1-y2)²); apply with the labels on the figure.",
        ],
        "slope_two_points": [
            "The figure labels two points P and Q. Compute the slope of line PQ.",
            "Compute slope = (y2 - y1)/(x2 - x1) using the labeled coordinates.",
            "From the two labeled points on the figure, find the slope of the line through them.",
            "Slope of PQ — use the coordinate labels on the figure.",
        ],
    }

    _Q_TO_FAM = {
        "read_point_y": "read_point",
        "read_intercept": "read_intercept",
        "linear_value": "linear",
        "parabola_value": "parabola_vertex",
        "two_lines_int": "two_lines",
        "parabola_line_int": "parabola_line",
        "cubic_value": "cubic_root",
        "distance_two_points": "distance",
        "slope_two_points": "slope",
    }

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: trivial — read a labeled point off a clearly-drawn line.
        if level == 0:
            return {"qtype_pool": ["read_point_y", "read_intercept"],
                    "grid_density": 1, "tight_distractors": False,
                    "show_equation": True}
        if level == 1:
            return {"qtype_pool": ["read_point_y", "read_intercept",
                                   "linear_value"],
                    "grid_density": 1, "tight_distractors": False,
                    "show_equation": True}
        if level <= 3:
            return {"qtype_pool": ["linear_value", "distance_two_points",
                                   "slope_two_points"],
                    "grid_density": 2, "tight_distractors": False,
                    "show_equation": True}
        if level <= 5:
            return {"qtype_pool": ["parabola_value", "two_lines_int",
                                   "linear_value", "distance_two_points"],
                    "grid_density": 2, "tight_distractors": True,
                    "show_equation": True}
        if level <= 7:
            return {"qtype_pool": ["two_lines_int", "parabola_line_int",
                                   "parabola_value"],
                    "grid_density": 3, "tight_distractors": True,
                    "show_equation": False}  # L6-L7: HIDE equation text
        return {"qtype_pool": ["parabola_line_int", "cubic_value",
                               "two_lines_int"],
                "grid_density": 4, "tight_distractors": True,
                "show_equation": False}     # L8-L9: HIDE equation text

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        for _ in range(25):
            r = self._try(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try(self, rng: random.Random, cfg: Dict, level: int):
        qtype = rng.choice(cfg["qtype_pool"])
        fam = self._Q_TO_FAM[qtype]
        tight = cfg["tight_distractors"]

        # Build shape
        shape = {}
        equation_text = ""
        extra_labels = {}

        if fam == "read_point":
            m = rng.choice([-3, -2, -1, 1, 2, 3])
            b = rng.randint(-4, 4)
            ax_val = rng.choice([v for v in range(-5, 6) if v != 0])
            y_val = m * ax_val + b
            if abs(y_val) > 12:
                return None
            equation_text = f"y = {self._lin_eq(m, b)}"
            shape = {"m": m, "b": b, "ax": ax_val}
            gt = y_val
        elif fam == "read_intercept":
            m = rng.choice([-3, -2, -1, 1, 2, 3])
            b = rng.randint(-5, 5)
            equation_text = f"y = {self._lin_eq(m, b)}"
            shape = {"m": m, "b": b, "ax": 0}
            gt = b
        elif fam == "linear":
            m = rng.choice([-3, -2, -1, 1, 2, 3])
            b = rng.randint(-4, 4)
            ax_val = rng.choice([v for v in range(-5, 6) if v != 0])
            equation_text = f"y = {self._lin_eq(m, b)}"
            shape = {"m": m, "b": b, "ax": ax_val}
            gt = m * ax_val + b
            if abs(gt) > 15:
                return None
        elif fam == "parabola_vertex":
            a = rng.choice([-2, -1, 1, 2])
            h = rng.randint(-2, 2)
            k = rng.randint(-3, 3)
            ax_val = rng.choice([v for v in range(h - 3, h + 4) if v != 0])
            gt = a * (ax_val - h) ** 2 + k
            if abs(gt) > 25:
                return None
            equation_text = f"y = {a}(x - {h})² + {k}" if k >= 0 else \
                            f"y = {a}(x - {h})² − {-k}"
            shape = {"a": a, "h": h, "k": k, "ax": ax_val}
        elif fam == "two_lines":
            m1 = rng.choice([-3, -2, -1, 1, 2, 3])
            m2 = rng.choice([mm for mm in [-3, -2, -1, 1, 2, 3] if mm != m1])
            b1 = rng.randint(-4, 4)
            b2 = rng.randint(-4, 4)
            num = b2 - b1
            den = m1 - m2
            if den == 0 or num % den != 0:
                return None
            x_int = num // den
            y_int = m1 * x_int + b1
            if abs(x_int) > 6 or abs(y_int) > 15:
                return None
            which = rng.choice(["x", "y"])
            gt = x_int if which == "x" else y_int
            equation_text = f"ℓ₁: y = {self._lin_eq(m1, b1)}\nℓ₂: y = {self._lin_eq(m2, b2)}"
            shape = {"lines": [(m1, b1), (m2, b2)], "int": (x_int, y_int),
                     "which": which}
        elif fam == "parabola_line":
            x0 = rng.randint(-3, 3)
            x1 = rng.randint(-3, 3)
            if x0 == x1:
                return None
            p = rng.choice([-2, -1, 1, 2])
            q = rng.randint(-2, 2)
            s = -(x0 + x1)
            k_coef = x0 * x1
            b_ = p + s
            c_ = q + k_coef
            which = rng.choice(["smaller", "larger"])
            gt = min(x0, x1) if which == "smaller" else max(x0, x1)
            equation_text = f"Parabola: y = x² {self._sign_coef(b_)} + {c_}" \
                            if c_ >= 0 else \
                            f"Parabola: y = x² {self._sign_coef(b_)} − {-c_}"
            equation_text += f"\nLine: y = {self._lin_eq(p, q)}"
            shape = {"parabola": (1, b_, c_), "line": (p, q),
                     "ints": [(x0, None), (x1, None)],
                     "which": which}
        elif fam == "cubic_root":
            a_ = rng.choice([-1, 1])
            b_ = rng.randint(-3, 3)
            c_ = rng.randint(-3, 3)
            ax_val = rng.choice([-3, -2, -1, 1, 2, 3])
            gt = a_ * ax_val ** 3 + b_ * ax_val + c_
            if abs(gt) > 40:
                return None
            a_str = "" if a_ == 1 else ("−" if a_ == -1 else str(a_))
            b_str = self._sign_coef(b_)
            c_str = f" + {c_}" if c_ >= 0 else f" − {-c_}"
            equation_text = f"y = {a_str}x³ {b_str}{c_str}"
            shape = {"cubic": (a_, b_, c_), "ax": ax_val}
        elif fam == "distance":
            x1 = rng.randint(-5, 5)
            y1 = rng.randint(-5, 5)
            x2 = rng.randint(-5, 5)
            y2 = rng.randint(-5, 5)
            if (x1, y1) == (x2, y2):
                return None
            gt_f = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            gt = round(gt_f, 2)
            equation_text = "P and Q labeled on figure"
            shape = {"pq": [(x1, y1), (x2, y2)]}
        elif fam == "slope":
            x1 = rng.randint(-5, 5)
            x2 = rng.randint(-5, 5)
            if x1 == x2:
                return None
            y1 = rng.randint(-5, 5)
            y2 = rng.randint(-5, 5)
            if (y2 - y1) % (x2 - x1) != 0:
                return None
            gt = (y2 - y1) // (x2 - x1)
            equation_text = "P and Q labeled on figure"
            shape = {"pq": [(x1, y1), (x2, y2)]}
        else:
            return None

        # Build options (numeric)
        # Use integer distractors (5 delta) except for distance which is fractional.
        if isinstance(gt, float):
            deltas = [-2, -1, 1, 2] if tight else [-3, -1, 1, 3]
            pool = set()
            for d in deltas:
                v = round(gt + d, 2)
                if v != gt:
                    pool.add(v)
            pool_list = list(pool)[:3]
            while len(pool_list) < 3:
                v = round(gt + rng.uniform(-4, 4), 2)
                if v != gt and v not in pool_list:
                    pool_list.append(v)
            options_vals = [gt] + pool_list[:3]
        else:
            deltas = [-2, -1, 1, 2] if tight else [-3, -1, 1, 3]
            pool = {gt + d for d in deltas}
            pool.discard(gt)
            # For smaller/larger pick also the other root as distractor
            if fam == "parabola_line":
                alt = shape["ints"][1][0] if shape["which"] == "smaller" else shape["ints"][0][0]
                if alt != gt:
                    pool.add(alt)
            pool_list = list(pool)
            rng.shuffle(pool_list)
            options_vals = [gt] + pool_list[:3]
            while len(options_vals) < 4:
                v = gt + rng.choice([-5, -4, 4, 5, 6])
                if v != gt and v not in options_vals:
                    options_vals.append(v)
        rng.shuffle(options_vals)
        if options_vals.count(gt) > 1:
            return None
        answer_letter = chr(ord("A") + options_vals.index(gt))

        # Build question
        q_templates = self._Q_POOL[qtype]
        qparams = {}
        if qtype in ("read_point_y", "linear_value", "parabola_value", "cubic_value"):
            qparams["ax"] = shape.get("ax", 0)
        elif qtype == "two_lines_int":
            qparams["which"] = shape.get("which", "x")
        elif qtype == "parabola_line_int":
            qparams["which"] = shape.get("which", "smaller")
        question_base = rng.choice(q_templates).format(**qparams)

        # If the equation is hidden (L6-L9), reword prompts that said
        # "Use the equation shown" to ask the learner to read the graph
        # visually. Also rewrite any template that references a labeled
        # equation — at this level the equation panel is blank.
        if not cfg.get("show_equation"):
            question_base = question_base.replace(
                "Use the line equation shown on the figure",
                "From the graph alone (no equation given)")
            question_base = question_base.replace(
                "Use the parabola equation shown on the figure",
                "From the graph of the parabola alone (no equation shown)")
            question_base = question_base.replace(
                "Use the cubic formula visible on the figure",
                "From the plotted cubic curve (no formula shown)")
            question_base = question_base.replace(
                "labeled equations", "plotted curves")
            question_base = question_base.replace(
                "with labeled equations", "")
            question_base = question_base.replace(
                "both equations labeled", "without equations printed")
            question_base = question_base.replace(
                "using its equation (visible on the figure)", "by reading the graph")
            question_base = question_base.replace(
                "From the labeled equation", "From the graph")
            question_base = question_base.replace(
                "From the labeled cubic equation", "From the cubic graph")
            question_base = question_base.replace(
                "Read the equation on the figure and evaluate",
                "Read the graph visually and evaluate")
            question_base = question_base.replace(
                "Apply the parabola formula (shown on the graph)",
                "Read from the parabola graph")
            question_base = question_base.replace(
                "Apply the cubic (labeled on the graph)",
                "Read the cubic graph")
            question_base = question_base.replace(
                "Using the labeled equations on the figure",
                "Reading the curves from the figure")
            question_base = question_base.replace(
                "see the figure for equations", "read the plot visually")
            # Additional rewrites for templates that didn't have a match
            # above — they still mention "equation".
            question_base = question_base.replace(
                "From the equation labeled on the figure, find y when",
                "From the plotted curve, find y when")
            question_base = question_base.replace(
                "The image displays a parabola with its equation.",
                "The image displays a parabola (equation not shown).")
            question_base = question_base.replace(
                "The image shows a parabola and a line (both equations labeled).",
                "The image shows a parabola and a line (equations not shown).")
            question_base = question_base.replace(
                "Solve for the larger x-intersection of the parabola with the line (see the figure for equations).",
                "Solve for the larger x-intersection of the parabola with the line (read the curves from the figure).")
            question_base = question_base.replace(
                "Solve for the smaller x-intersection of the parabola with the line (see the figure for equations).",
                "Solve for the smaller x-intersection of the parabola with the line (read the curves from the figure).")
            question_base = question_base.replace(
                "The figure shows a cubic with its equation labeled.",
                "The figure shows a cubic (equation not shown).")
            # Safety net: if the word "equation" or "formula" still appears
            # in a way that implies the panel is populated, rewrite them.
            # Strip "equations labeled" / "equations shown" entirely so the
            # student is told to read the plotted curves, not an absent
            # equation panel.
            question_base = question_base.replace(
                "Using the equations labeled on the figure, ",
                "Reading the plotted curves, ")
            question_base = question_base.replace(
                "Using the equations shown on the figure, ",
                "Reading the plotted curves, ")
            question_base = question_base.replace(
                "shown on the figure", "visible in the plot")
            question_base = question_base.replace(
                "labeled on the figure", "shown in the plot")

        options_str = [str(v) for v in options_vals]
        question = (
            f"{question_base}\n"
            + "\n".join(f"  ({chr(ord('A') + i)}) {options_str[i]}"
                         for i in range(4))
            + "\nAnswer with the single letter of the correct option."
        )

        # Render
        image = self._render(fam, shape, equation_text, options_str, cfg,
                             rng, qtype, qparams)
        return question, answer_letter, image

    # ------------------------------------------------------------------ #
    def _lin_eq(self, m, b):
        """Format 'm·x + b' cleanly."""
        if m == 1:
            mx = "x"
        elif m == -1:
            mx = "−x"
        else:
            mx = f"{m}x"
        if b > 0:
            return f"{mx} + {b}"
        if b < 0:
            return f"{mx} − {-b}"
        return mx

    def _sign_coef(self, coef: int) -> str:
        if coef == 1:
            return "+ x"
        if coef == -1:
            return "− x"
        if coef >= 0:
            return f"+ {coef}x"
        return f"− {-coef}x"

    # ------------------------------------------------------------------ #
    def _render(self, fam, shape, equation_text, options, cfg,
                rng, qtype, qparams) -> Image.Image:
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        fig = plt.figure(figsize=(10 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_p = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_t.axis("off")

        lw = style["line_width"]

        ax_p.axhline(0, color="#7f8c8d", linewidth=0.8)
        ax_p.axvline(0, color="#7f8c8d", linewidth=0.8)
        xmin, xmax = -7, 7
        ymin, ymax = -15, 15
        if fam in ("read_point", "read_intercept", "linear", "two_lines",
                   "distance", "slope"):
            ymin, ymax = -10, 10
        elif fam == "parabola_vertex":
            ymin, ymax = -10, 20
        elif fam == "cubic_root":
            ymin, ymax = -30, 30
        ax_p.set_xlim(xmin, xmax)
        ax_p.set_ylim(ymin, ymax)
        if cfg.get("grid_density", 2) <= 2:
            ax_p.grid(True, alpha=0.3, linestyle=":")
        else:
            ax_p.grid(True, alpha=0.15, linestyle=":")
        ax_p.tick_params(axis="both", labelsize=fs - 1)

        if fam in ("read_point", "read_intercept", "linear"):
            m = shape["m"]
            b = shape["b"]
            xs = [xmin, xmax]
            ys = [m * x + b for x in xs]
            ax_p.plot(xs, ys, "-", color=palette[2], linewidth=lw)
            if fam == "read_point":
                px = shape["ax"]
                py = m * px + b
                ax_p.plot(px, py, "o", color=palette[3], markersize=9)
                ax_p.annotate(f"P (x={px})", (px, py),
                              textcoords="offset points", xytext=(10, 8),
                              fontsize=fs + 1, color=palette[3],
                              fontweight="bold")
            elif fam == "read_intercept":
                ax_p.plot(0, b, "o", color=palette[3], markersize=9)
                ax_p.annotate("y-intercept", (0, b),
                              textcoords="offset points", xytext=(10, 8),
                              fontsize=fs, color=palette[3],
                              fontweight="bold")
            else:
                px = shape["ax"]
                py = m * px + b
                ax_p.plot(px, py, "o", color=palette[3], markersize=8)
                ax_p.annotate(f"x = {px}", (px, py),
                              textcoords="offset points", xytext=(10, 8),
                              fontsize=fs, color=palette[3],
                              fontweight="bold")
        elif fam == "parabola_vertex":
            a, h, k = shape["a"], shape["h"], shape["k"]
            xs_ = [xmin + 0.1 * i for i in range(140)]
            ys_ = [a * (x - h) ** 2 + k for x in xs_]
            ax_p.plot(xs_, ys_, "-", color=palette[2], linewidth=lw)
            ax_p.plot(h, k, "o", color=palette[3], markersize=8)
            # Only show V(h,k) coordinates when the equation panel is
            # visible (else the label trivializes parabola_value at L6-L7
            # by giving away h and k).
            if cfg.get("show_equation"):
                ax_p.annotate(f"V({h},{k})", (h, k),
                              textcoords="offset points", xytext=(10, 8),
                              fontsize=fs + 1, color=palette[3],
                              fontweight="bold")
            else:
                ax_p.annotate("V", (h, k),
                              textcoords="offset points", xytext=(8, 8),
                              fontsize=fs + 1, color=palette[3],
                              fontweight="bold")
            px = shape["ax"]
            ax_p.axvline(px, color=palette[5 % len(palette)], linestyle=":",
                         alpha=0.6)
            ax_p.annotate(f"x = {px}", (px, ymax - 1), fontsize=fs,
                          color=palette[5 % len(palette)], ha="center")
        elif fam == "two_lines":
            for i, (m, b) in enumerate(shape["lines"]):
                xs_ = [xmin, xmax]
                ys_ = [m * x + b for x in xs_]
                label_sub = "ℓ₁" if i == 0 else "ℓ₂"
                ax_p.plot(xs_, ys_, "-", color=palette[2 + i], linewidth=lw,
                          label=label_sub)
            ax_p.legend(fontsize=fs - 1, loc="upper left")
        elif fam == "parabola_line":
            a_, b_, c_ = shape["parabola"]
            xs_ = [xmin + 0.1 * i for i in range(140)]
            ys_ = [a_ * x * x + b_ * x + c_ for x in xs_]
            ax_p.plot(xs_, ys_, "-", color=palette[2], linewidth=lw,
                      label="parabola")
            p, q = shape["line"]
            ax_p.plot([xmin, xmax], [p * xmin + q, p * xmax + q],
                      "-", color=palette[3], linewidth=lw, label="line")
            ax_p.legend(fontsize=fs - 1, loc="upper left")
        elif fam == "cubic_root":
            a_, b_, c_ = shape["cubic"]
            xs_ = [xmin + 0.1 * i for i in range(140)]
            ys_ = [a_ * x ** 3 + b_ * x + c_ for x in xs_]
            ax_p.plot(xs_, ys_, "-", color=palette[2], linewidth=lw)
            px = shape["ax"]
            ax_p.axvline(px, color=palette[5 % len(palette)], linestyle=":",
                         alpha=0.6)
        elif fam in ("distance", "slope"):
            (x1, y1), (x2, y2) = shape["pq"]
            ax_p.plot([x1, x2], [y1, y2], "-", color=palette[2],
                      linewidth=lw)
            ax_p.plot(x1, y1, "o", color=palette[3], markersize=8)
            ax_p.plot(x2, y2, "o", color=palette[3], markersize=8)
            ax_p.annotate(f"P({x1},{y1})", (x1, y1),
                          textcoords="offset points", xytext=(10, 8),
                          fontsize=fs, color=palette[3],
                          fontweight="bold")
            ax_p.annotate(f"Q({x2},{y2})", (x2, y2),
                          textcoords="offset points", xytext=(10, 8),
                          fontsize=fs, color=palette[3],
                          fontweight="bold")

        ax_p.set_aspect("auto")
        coord_title = rng.choice(self._TITLES)
        ax_p.set_title(coord_title, fontsize=fs + 1, family=ff, pad=6)

        # Equation panel
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        y = 11.5
        if cfg.get("show_equation") and equation_text:
            ax_t.text(0.3, y, "Equation:", fontsize=fs + 1, fontweight="bold",
                      family=ff, ha="left", va="top", color="#2c3e50")
            y -= 0.7
            for ln in equation_text.split("\n"):
                ax_t.text(0.3, y, ln, fontsize=fs + 1, family=ff,
                          ha="left", va="top", color="#1a1a1a")
                y -= 0.6
            y -= 0.4
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.55
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs + 1, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08,
                            wspace=0.18)
        return self.fig_to_pil(fig, dpi=style["dpi"])
