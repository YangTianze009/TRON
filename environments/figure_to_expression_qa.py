"""
Figure to Expression QA (v4 G5, for symbolic-expression reasoning).

Targets:

Task: given a labeled figure (rectangle / triangle / circle with literal
variables a, b, h, r, θ), ask for a symbolic expression for a quantity
(perimeter, area, angle sum, etc.).

Reward: symbolic equivalence via SymPy — accepts `2*a + 2*b` = `2(a+b)` = `2a+2b`.

Level axes:
  A) Figure type complexity: simple at L0 -> compound at L6+
  B) Target complexity: perimeter at L0-3 -> area at L4-6 -> volume/surface at L7+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The figure has labels {labels}. Express the {target} of the figure as a symbolic formula in terms of those variables. Put the formula in <answer>...</answer>.",
    "Given the labeled figure with variables {labels}, write a symbolic expression for the {target}. Put it in <answer>...</answer>.",
    "Using the variables {labels} shown on the figure, derive the {target} as a symbolic formula. Put in <answer>...</answer>.",
    "The figure uses {labels} as variables. Express the {target} symbolically. Put the expression in <answer>...</answer>.",
    "Write a symbolic formula for the {target} of the figure (variables: {labels}). Put in <answer>...</answer>.",
    "Express {target} symbolically using {labels}. Put the formula in <answer>...</answer>.",
    "Given variables {labels}, derive a formula for {target}. Put it in <answer>...</answer>.",
    "Write the {target} formula using the variables {labels} from the figure. Put in <answer>...</answer>.",
    "Express the figure's {target} using symbols {labels}. Put in <answer>...</answer>.",
    "Formula for {target} in terms of {labels}? Put the expression in <answer>...</answer>.",
    "Derive the {target} symbolically. Variables: {labels}. Put in <answer>...</answer>.",
    "Express {target} as a formula using {labels}. Put in <answer>...</answer>.",
    "Formula for {target} using variables {labels}: Put in <answer>...</answer>.",
    "Express {target} symbolically, using the variables {labels} labeled on the figure. Put in <answer>...</answer>.",
    "Using variables {labels}, write a symbolic expression for the {target}. Put in <answer>...</answer>.",
    "Symbolic formula for {target} (variables: {labels})? Put in <answer>...</answer>.",
]

_PROBLEMS = [
    # (figure_type, labels, target, gt_expression, verifier)
    ("rectangle", "a (length), b (width)", "perimeter",
     "2*a + 2*b", lambda a, b: 2*a + 2*b),
    ("rectangle", "a (length), b (width)", "area",
     "a*b", lambda a, b: a*b),
    ("triangle", "a, b (two sides), θ (angle between them)", "area",
     "(1/2)*a*b*sin(θ)", lambda a, b, t: 0.5 * a * b),  # placeholder
    ("circle", "r (radius)", "circumference",
     "2*pi*r", lambda r: 2 * 3.14159 * r),
    ("circle", "r (radius)", "area",
     "pi*r**2", lambda r: 3.14159 * r * r),
    ("cylinder", "r (radius), h (height)", "volume",
     "pi*r**2*h", lambda r, h: 3.14159 * r * r * h),
    ("cone", "r (radius), h (height)", "volume",
     "(1/3)*pi*r**2*h", lambda r, h: (1.0/3.0) * 3.14159 * r * r * h),
    ("cube", "s (side length)", "volume",
     "s**3", lambda s: s * s * s),
    ("cube", "s (side length)", "surface_area",
     "6*s**2", lambda s: 6 * s * s),
    ("regular_hexagon", "s (side length)", "perimeter",
     "6*s", lambda s: 6 * s),
    ("parallelogram", "a (base), h (height)", "area",
     "a*h", lambda a, h: a*h),
    ("trapezoid", "a, b (parallel sides), h (height)", "area",
     "(1/2)*(a+b)*h", lambda a, b, h: 0.5 * (a + b) * h),
    ("rhombus", "d1, d2 (diagonals)", "area",
     "(1/2)*d1*d2", lambda d1, d2: 0.5 * d1 * d2),
    ("equilateral_triangle", "s (side length)", "area",
     "(sqrt(3)/4)*s**2", lambda s: 0.433 * s * s),
    ("square", "s (side length)", "diagonal",
     "s*sqrt(2)", lambda s: 1.414 * s),
    ("rectangle_prism", "a, b, c (edge lengths)", "volume",
     "a*b*c", lambda a, b, c: a*b*c),
]

class FigureToExpressionQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "figure_to_expression"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Slice the pool by level — each band uses a different difficulty
        # tier, NOT a strictly growing prefix. That guarantees the figure
        # actually changes across levels (otherwise the same low-index
        # problem keeps getting picked at every level for the same seed).
        # Tiers (rough complexity ordering — see _PROBLEMS):
        #   T1 simple 2D perimeter/area (idx 0,1)
        #   T2 2D area + circle (idx 2-4)
        #   T3 cylinder/cone vol, cube vol/SA (idx 5-8)
        #   T4 hexagon, parallelogram, trapezoid, rhombus (idx 9-12)
        #   T5 equilateral, sqrt diagonal, rect-prism (idx 13-15)
        if level <= 1:
            problem_pool = _PROBLEMS[0:2]            # T1 only
        elif level <= 3:
            problem_pool = _PROBLEMS[0:5]            # T1 + T2
        elif level <= 5:
            problem_pool = _PROBLEMS[2:9]            # T2-T3 (skip easiest)
        elif level <= 7:
            problem_pool = _PROBLEMS[5:13]           # T3-T4
        else:
            problem_pool = _PROBLEMS[8:]             # T4-T5 (hardest)
        return {"pool": problem_pool}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 211)
        self._primary_complexity_feature = level

        prob = rng.choice(cfg["pool"])
        fig_type, labels, target, gt_expr, _ = prob

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(labels=labels, target=target)

        img = self._render_schematic(fig_type, labels, rng)
        return q, gt_expr, img

    def _render_schematic(self, fig_type, labels, rng):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.axis("off")

        if fig_type == "rectangle":
            ax.add_patch(mpatches.Rectangle((1, 1), 4, 2, fc="none",
                                             ec="black", lw=2.0))
            ax.text(3, 0.6, "a", fontsize=16, fontstyle="italic", ha="center")
            ax.text(0.6, 2, "b", fontsize=16, fontstyle="italic", va="center")
        elif fig_type == "triangle":
            pts = [(1, 0.5), (5, 0.5), (3, 3)]
            ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
            ax.text(3, 0.1, "a", fontsize=16, fontstyle="italic", ha="center")
            ax.text(1.8, 1.8, "b", fontsize=16, fontstyle="italic")
            ax.text(1.8, 0.9, "θ", fontsize=14, fontstyle="italic")
        elif fig_type == "circle":
            ax.add_patch(mpatches.Circle((3, 2), 1.3, fc="none",
                                          ec="black", lw=2.0))
            ax.plot([3, 3 + 1.3], [2, 2], color="black", lw=1.5)
            ax.text(3.6, 2.2, "r", fontsize=16, fontstyle="italic")
        elif fig_type == "cylinder":
            # simple cylinder schematic
            ax.add_patch(mpatches.Ellipse((3, 3.2), 2, 0.5, fc="none",
                                            ec="black", lw=2.0))
            ax.plot([2, 2], [1, 3.2], color="black", lw=2.0)
            ax.plot([4, 4], [1, 3.2], color="black", lw=2.0)
            ax.add_patch(mpatches.Ellipse((3, 1), 2, 0.5, fc="none",
                                            ec="black", lw=2.0))
            ax.text(3.1, 1, "r", fontsize=14, fontstyle="italic")
            ax.text(4.3, 2.1, "h", fontsize=14, fontstyle="italic")
        elif fig_type == "cone":
            ax.plot([2, 3, 4], [1, 3.5, 1], color="black", lw=2.0)
            ax.add_patch(mpatches.Ellipse((3, 1), 2, 0.4, fc="none",
                                            ec="black", lw=2.0))
            ax.text(3.1, 1.1, "r", fontsize=14, fontstyle="italic")
            ax.text(4.1, 2.3, "h", fontsize=14, fontstyle="italic")
        elif fig_type == "cube":
            # simple oblique projection
            ax.add_patch(mpatches.Rectangle((1.2, 1), 2.5, 2.5, fc="none",
                                             ec="black", lw=2.0))
            # back face offset
            offs = 0.5
            ax.plot([1.2 + offs, 1.2 + offs, 3.7 + offs, 3.7 + offs, 1.2 + offs],
                    [1 + offs, 3.5 + offs, 3.5 + offs, 1 + offs, 1 + offs],
                    color="black", lw=1.5)
            ax.plot([1.2, 1.2 + offs], [1, 1 + offs], color="black", lw=1.0)
            ax.plot([3.7, 3.7 + offs], [1, 1 + offs], color="black", lw=1.0)
            ax.plot([3.7, 3.7 + offs], [3.5, 3.5 + offs], color="black", lw=1.0)
            ax.plot([1.2, 1.2 + offs], [3.5, 3.5 + offs], color="black", lw=1.0)
            ax.text(2.3, 0.6, "s", fontsize=16, fontstyle="italic", ha="center")
        # BUGFIX 2026-04-24: add renderers for previously-missing shapes so
        # image never falls into literal-text placeholder branch.
        elif fig_type == "regular_hexagon":
            import math as _m
            cx, cy, R = 3, 2, 1.2
            pts = [(cx + R * _m.cos(_m.pi / 3 * i + _m.pi / 6),
                    cy + R * _m.sin(_m.pi / 3 * i + _m.pi / 6)) for i in range(6)]
            ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
            ax.text(cx, cy - R - 0.35, "s", fontsize=16, fontstyle="italic",
                    ha="center")
        elif fig_type == "parallelogram":
            pts = [(1, 1), (4, 1), (5, 3), (2, 3)]
            ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
            ax.text(2.5, 0.5, "a", fontsize=16, fontstyle="italic", ha="center")
            ax.plot([2, 2], [1, 3], color="gray", lw=1.0, linestyle="--")
            ax.text(1.6, 2, "h", fontsize=14, fontstyle="italic", ha="right")
        elif fig_type == "trapezoid":
            pts = [(1, 1), (5, 1), (4, 3), (2, 3)]
            ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
            ax.text(3, 0.5, "a", fontsize=16, fontstyle="italic", ha="center")
            ax.text(3, 3.3, "b", fontsize=16, fontstyle="italic", ha="center")
            ax.plot([2, 2], [1, 3], color="gray", lw=1.0, linestyle="--")
            ax.text(1.6, 2, "h", fontsize=14, fontstyle="italic", ha="right")
        elif fig_type == "rhombus":
            pts = [(3, 0.6), (4.5, 2), (3, 3.4), (1.5, 2)]
            ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
            ax.plot([1.5, 4.5], [2, 2], color="gray", lw=1.0, linestyle="--")
            ax.plot([3, 3], [0.6, 3.4], color="gray", lw=1.0, linestyle="--")
            ax.text(0.95, 1.9, "d1", fontsize=13, fontstyle="italic")
            ax.text(3.15, 3.45, "d2", fontsize=13, fontstyle="italic")
        elif fig_type == "equilateral_triangle":
            import math as _m
            s = 2.5
            h = s * _m.sqrt(3) / 2
            pts = [(3 - s / 2, 0.8), (3 + s / 2, 0.8), (3, 0.8 + h)]
            ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
            ax.text(3, 0.4, "s", fontsize=16, fontstyle="italic", ha="center")
        elif fig_type == "square":
            ax.add_patch(mpatches.Rectangle((1.8, 0.9), 2.2, 2.2, fc="none",
                                             ec="black", lw=2.0))
            ax.plot([1.8, 4.0], [0.9, 3.1], color="gray", lw=1.0, linestyle="--")
            ax.text(2.9, 0.5, "s", fontsize=16, fontstyle="italic", ha="center")
        elif fig_type == "rectangle_prism":
            # 3D rectangular prism: front face + offset back face
            ax.add_patch(mpatches.Rectangle((1.2, 1), 2.6, 1.6, fc="none",
                                             ec="black", lw=2.0))
            offs = 0.5
            ax.plot([1.2 + offs, 1.2 + offs, 3.8 + offs, 3.8 + offs, 1.2 + offs],
                    [1 + offs, 2.6 + offs, 2.6 + offs, 1 + offs, 1 + offs],
                    color="black", lw=1.5)
            ax.plot([1.2, 1.2 + offs], [1, 1 + offs], color="black", lw=1.0)
            ax.plot([3.8, 3.8 + offs], [1, 1 + offs], color="black", lw=1.0)
            ax.plot([3.8, 3.8 + offs], [2.6, 2.6 + offs], color="black", lw=1.0)
            ax.plot([1.2, 1.2 + offs], [2.6, 2.6 + offs], color="black", lw=1.0)
            ax.text(2.5, 0.6, "a", fontsize=14, fontstyle="italic", ha="center")
            ax.text(0.9, 1.8, "b", fontsize=14, fontstyle="italic", ha="right")
            ax.text(4.1, 1.3, "c", fontsize=14, fontstyle="italic")
        else:
            # Fallback: simple generic quad instead of literal shape name text
            ax.add_patch(mpatches.Polygon([(1.5, 1), (4.5, 1), (4.5, 3), (1.5, 3)],
                                            fc="none", ec="black", lw=2.0))
        # Caption
        ax.text(3, 3.75, f"Variables: {labels}", fontsize=10, ha="center",
                bbox=dict(facecolor="lightyellow", edgecolor="gray"))

        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import re as _re
        pred = predicted.strip().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().rstrip(".").rstrip(",").rstrip()

        def norm(s):
            s = s.replace(" ", "").replace("·", "*")
            s = s.replace("×", "*").replace("⋅", "*")
            s = s.replace("^", "**")
            s = s.replace("π", "pi").replace("Π", "pi")
            s = s.replace("\\pi", "pi").replace("\\Pi", "pi")
            s = s.replace("\\cdot", "*").replace("\\times", "*")
            s = s.replace("\\sqrt", "sqrt")
            s = s.replace("\\frac", "frac")
            s = s.replace("{", "(").replace("}", ")")
            s = s.replace("²", "**2").replace("³", "**3")
            # strip leading "A=", "P=", "f(...)=", etc. (single uppercase var
            # followed by "=" before the expression)
            s = _re.sub(r'^[A-Za-z][A-Za-z0-9_]*\s*=\s*', '', s)
            # implicit multiplication like 2a → 2*a, pir → pi*r,
            # ab → a*b only when between letter-letter that look variable-y
            s = _re.sub(r'(\d)([A-Za-z])', r'\1*\2', s)
            s = _re.sub(r'([A-Za-z])(\d)', r'\1*\2', s)
            # pi followed by letter or digit
            s = _re.sub(r'\bpi([A-Za-z\d])', r'pi*\1', s)
            # collapse double-stars from prior sub
            s = s.replace("****", "**")
            return s.lower()

        if pred == gt or norm(pred) == norm(gt):
            return True
        # Try SymPy equivalence (more forgiving)
        try:
            from sympy import sympify, simplify, symbols, pi, sqrt
            local = {v: symbols(v) for v in
                     ["a", "b", "c", "d", "d1", "d2", "e", "f", "g",
                      "h", "n", "r", "s", "t", "x", "y", "z", "L", "W"]}
            local["pi"] = pi
            local["sqrt"] = sqrt
            p_str = norm(pred); g_str = norm(gt)
            p = sympify(p_str, locals=local)
            g = sympify(g_str, locals=local)
            return simplify(p - g) == 0
        except Exception:
            return False

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_fte"
    os.makedirs(out_dir, exist_ok=True)
    env = FigureToExpressionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 101
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[fte L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/fte_s{s}_L{level}.png")
            print(f"[fte L{level} s{s}] A={env._answer}")
