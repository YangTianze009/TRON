"""
Expression Substitute Evaluate QA environment.

Round 2 fixes:
  - L0 now emits a trivial linear expression with integer coefficients and
    small substitution value; expression displayed in BIG font. Formula
    rendering is clear.
  - L9 includes nested rational, trivariate, and piecewise expressions.
  - Expanded visual diversity: randomized background color, randomized
    boxed frame, random font family, larger variety of function names and
    variable pools.
  - 4+ question phrasings ("compute f(x)", "evaluate", "find the value of f
    when x=...", etc.).
  - Expression values on IMAGE; question text does not restate the formula.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class ExpressionSubstituteEvaluateQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "expression_substitute_evaluate"

    _VAR_POOL_SINGLE = ["x", "t", "u", "z", "w", "m", "k", "r", "n"]
    _VAR_POOL_DOUBLE = [("x", "y"), ("t", "s"), ("u", "v"), ("a", "b"),
                         ("p", "q"), ("m", "n"), ("r", "s")]
    _VAR_POOL_TRIPLE = [("x", "y", "z"), ("a", "b", "c"), ("p", "q", "r"),
                         ("u", "v", "w")]
    _FN_POOL = ["f", "g", "h", "F", "G", "P", "Q", "H", "S", "T"]

    _Q_POOL = [
        "The image shows the definition of {fn_sig}. Compute {fn_val}. Answer with a single integer.",
        "Read the formula for {fn_sig} from the image and evaluate {fn_val}. Return a single integer.",
        "Using the expression shown in the figure, find the value of {fn_val}. Answer with one integer.",
        "From the image, substitute to compute {fn_val}. Provide a single integer.",
        "The figure defines {fn_sig}. What is {fn_val}? Single integer answer.",
        "Given {fn_sig} as defined in the figure, evaluate {fn_val}. Output a single integer.",
        "Substitute into the formula for {fn_sig} shown in the image to find {fn_val}. Integer answer only.",
        "Look at the definition of {fn_sig} in the figure. Compute {fn_val} and give one integer.",
        "The image defines {fn_sig}. Using that definition, calculate {fn_val} (integer answer).",
        "Apply the formula for {fn_sig} from the figure to find {fn_val}. Respond with a single integer.",
        "Refer to the image's definition of {fn_sig}. Evaluate {fn_val} and return an integer.",
        "With {fn_sig} given in the figure, compute {fn_val}. Provide only the integer value.",
        "The figure shows {fn_sig}. Determine {fn_val} by substitution. Single integer.",
        "Use the formula for {fn_sig} from the image to obtain {fn_val}. Integer output.",
        "Consult the figure's definition of {fn_sig} and compute {fn_val}. Reply with one integer.",
        "Per the formula for {fn_sig} shown in the image, find {fn_val}. Integer only.",
    ]

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: simple linear with small coefs and subs. Show BIG formula.
        if level == 0:
            kinds = ["linear"]
            coef_hi = 5
            sub_lo = 0
            sub_hi = 5
            fontsize = 24
        elif level == 1:
            kinds = ["linear", "quadratic"]
            coef_hi = 6
            sub_lo = -2
            sub_hi = 6
            fontsize = 22
        elif level <= 3:
            kinds = ["quadratic", "rational"]
            coef_hi = 8
            sub_lo = -3
            sub_hi = 7
            fontsize = 20
        elif level <= 5:
            kinds = ["rational", "cubic"]
            coef_hi = 10
            sub_lo = -4
            sub_hi = 8
            fontsize = 18
        elif level <= 7:
            kinds = ["multivar", "multivar_quad"]
            coef_hi = 11
            sub_lo = -5
            sub_hi = 10
            fontsize = 17
            puzzle_pieces = True   # split equation across visual boxes
            distractor_count = 2
        else:
            # L9 iter-3 (2026-04-17): ONLY `deep_chain` composition. Four
            # functions chained: f(g(h(k(x0)))) with distinct affine pieces.
            # Student must perform 4 substitutions.
            kinds = ["deep_chain"]
            coef_hi = 10
            sub_lo = -5
            sub_hi = 9
            fontsize = 16
            puzzle_pieces = True
            distractor_count = 3
        if level <= 5:
            puzzle_pieces = False
            distractor_count = 0
        return {
            "kind_pool": kinds,
            "coef_hi": coef_hi,
            "sub_lo": sub_lo,
            "sub_hi": sub_hi,
            "fontsize": fontsize,
            "puzzle_pieces": puzzle_pieces,
            "distractor_count": distractor_count,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(25):
            result = self._try_generate(parameter)
            if result is not None:
                return result
        return None

    def _try_generate(self, parameter: Dict):
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 997)

        kind = rng.choice(cfg["kind_pool"])
        fn = rng.choice(self._FN_POOL)

        if kind == "linear":
            var = rng.choice(self._VAR_POOL_SINGLE)
            a = rng.randint(1, min(5, cfg["coef_hi"]))
            b = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            if rng.random() < 0.3 and level > 0:
                a = -a
            x_val = rng.randint(cfg["sub_lo"], cfg["sub_hi"])
            if x_val == 0 and level > 0:
                x_val = 1
            ans = a * x_val + b
            expr_tex = self._tex_linear(fn, var, a, b)
            fn_sig = f"{fn}({var})"
            fn_val = f"{fn}({x_val})"

        elif kind == "quadratic":
            var = rng.choice(self._VAR_POOL_SINGLE)
            a = rng.randint(1, min(5, cfg["coef_hi"]))
            b = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            if rng.random() < 0.3:
                a = -a
            x_val = rng.randint(cfg["sub_lo"], cfg["sub_hi"])
            if x_val == 0:
                x_val = 2
            ans = a * x_val ** 2 + b * x_val + c
            expr_tex = self._tex_quadratic(fn, var, a, b, c)
            fn_sig = f"{fn}({var})"
            fn_val = f"{fn}({x_val})"

        elif kind == "cubic":
            var = rng.choice(self._VAR_POOL_SINGLE)
            a = rng.randint(1, 3)
            b = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x_val = rng.randint(-3, 4)
            if x_val == 0:
                x_val = 2
            ans = a * x_val ** 3 + b * x_val + c
            expr_tex = self._tex_cubic(fn, var, a, b, c)
            fn_sig = f"{fn}({var})"
            fn_val = f"{fn}({x_val})"

        elif kind == "rational":
            var = rng.choice(self._VAR_POOL_SINGLE)
            a = rng.randint(1, 4)
            candidates = [v for v in range(cfg["sub_lo"], cfg["sub_hi"] + 1)
                          if v != a and v != -a and v != 0]
            if not candidates:
                return None
            x_val = rng.choice(candidates)
            form = rng.choice(["diff_a_minus", "diff_a_plus", "poly_div"])
            if form == "diff_a_minus":
                num_val = x_val ** 2 - a ** 2
                den_val = x_val - a
                if den_val == 0:
                    return None
                ans = num_val // den_val
                expr_tex = (rf"${fn}({var}) = \dfrac{{{var}^{{2}} - {a * a}}}"
                            rf"{{{var} - {a}}}$")
            elif form == "diff_a_plus":
                num_val = x_val ** 2 - a ** 2
                den_val = x_val + a
                if den_val == 0:
                    return None
                ans = num_val // den_val
                expr_tex = (rf"${fn}({var}) = \dfrac{{{var}^{{2}} - {a * a}}}"
                            rf"{{{var} + {a}}}$")
            else:
                if x_val == 0:
                    return None
                coef_a = rng.randint(1, 4)
                coef_b = rng.randint(1, cfg["coef_hi"])
                num_val = coef_a * x_val ** 2 + coef_b * x_val
                den_val = x_val
                ans = num_val // den_val
                expr_tex = (rf"${fn}({var}) = "
                            rf"\dfrac{{{coef_a}{var}^{{2}} + {coef_b}{var}}}"
                            rf"{{{var}}}$")
            fn_sig = f"{fn}({var})"
            fn_val = f"{fn}({x_val})"

        elif kind == "multivar":
            v1, v2 = rng.choice(self._VAR_POOL_DOUBLE)
            a = rng.randint(1, 5)
            b = rng.randint(1, 5)
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            if rng.random() < 0.4:
                a = -a
            x1 = rng.randint(1, cfg["sub_hi"])
            x2 = rng.randint(1, cfg["sub_hi"])
            ans = a * x1 + b * x2 + c
            expr_tex = self._tex_multivar_linear(fn, v1, v2, a, b, c)
            fn_sig = f"{fn}({v1}, {v2})"
            fn_val = f"{fn}({x1}, {x2})"

        elif kind == "multivar_quad":
            v1, v2 = rng.choice(self._VAR_POOL_DOUBLE)
            a = rng.randint(1, 4)
            b = rng.randint(1, 4)
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x1 = rng.randint(-2, 4)
            x2 = rng.randint(-2, 4)
            if x1 == 0 and x2 == 0:
                x1 = 1
            ans = a * x1 ** 2 + b * x2 + c
            expr_tex = self._tex_multivar_quad(fn, v1, v2, a, b, c)
            fn_sig = f"{fn}({v1}, {v2})"
            fn_val = f"{fn}({x1}, {x2})"

        elif kind == "trivar":
            v1, v2, v3 = rng.choice(self._VAR_POOL_TRIPLE)
            a = rng.randint(1, 4)
            b = rng.randint(1, 3)
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x1 = rng.randint(-3, 4)
            x2 = rng.randint(-3, 4)
            x3 = rng.randint(-3, 4)
            if x1 == 0: x1 = 1
            ans = a * x1 * x2 + b * x3 ** 2 + c
            a_s = "" if a == 1 else f"{a}"
            b_s = "" if b == 1 else f"{b}"
            expr_tex = (rf"${fn}({v1}, {v2}, {v3}) = "
                        rf"{a_s}{v1}{v2} + {b_s}{v3}^{{2}}{self._signed(c)}$")
            fn_sig = f"{fn}({v1}, {v2}, {v3})"
            fn_val = f"{fn}({x1}, {x2}, {x3})"

        elif kind == "nested_rational":
            c_ = rng.randint(1, 4)
            x_val = rng.choice([v for v in range(-4, 5)
                                if v != -c_ and v != 0])
            den = x_val + c_
            k = rng.randint(-8, 8)
            if k == 0:
                k = rng.choice([-3, -2, -1, 1, 2, 3])
            a = rng.randint(1, 3)
            d = k * den - a * x_val ** 2
            ans = k
            var = "x"
            expr_tex = (rf"${fn}(x) = "
                        rf"\dfrac{{{a}x^{{2}}{self._signed(d)}}}"
                        rf"{{x + {c_}}}$")
            fn_sig = f"{fn}(x)"
            fn_val = f"{fn}({x_val})"

        elif kind == "multivar_cubic":
            v1, v2 = rng.choice(self._VAR_POOL_DOUBLE)
            a = rng.choice([1, -1, 2])
            b = rng.randint(1, 4)
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x1 = rng.randint(-2, 3)
            x2 = rng.randint(-2, 3)
            if x1 == 0: x1 = 1
            ans = a * x1 ** 3 + b * x1 * x2 + c
            # suppress coefficient "1" / render "-1" as "-"
            a_s = "" if a == 1 else ("-" if a == -1 else f"{a}")
            b_s = "" if b == 1 else f"{b}"
            expr_tex = (rf"${fn}({v1}, {v2}) = "
                        rf"{a_s}{v1}^{{3}} + {b_s}{v1}{v2}{self._signed(c)}$")
            fn_sig = f"{fn}({v1}, {v2})"
            fn_val = f"{fn}({x1}, {x2})"

        elif kind == "piecewise":
            # Inline two-line piecewise (matplotlib mathtext compatible).
            var = rng.choice(self._VAR_POOL_SINGLE)
            a = rng.randint(1, 5)
            b = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            c = rng.randint(1, 5)
            d = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x_val = rng.choice([v for v in range(-5, 6) if v != 0])
            if x_val >= 0:
                ans = a * x_val + b
            else:
                ans = c * x_val + d
            # Use plain-text piecewise rendering (not LaTeX cases block).
            part_pos = f"{a}{var}{self._signed(b).strip()}"
            part_neg = f"{c}{var}{self._signed(d).strip()}"
            expr_tex = (f"{fn}({var}) = {part_pos}   if {var} ≥ 0\n"
                        f"{' ' * len(fn + '(' + var + ')')}"
                        f" = {part_neg}   if {var} < 0")
            fn_sig = f"{fn}({var})"
            fn_val = f"{fn}({x_val})"

        elif kind == "composite":
            # Chained substitution: f(x) = ax+b, g(x) = cx^2+d. Ask for
            # f(g(x0)). Requires student to compute g(x0) first, then feed
            # into f.
            var = "x"
            g_name = rng.choice([c for c in self._FN_POOL if c != fn])
            a = rng.randint(1, 5)
            b = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            c_ = rng.randint(1, 4)
            d_ = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x_val = rng.choice([v for v in range(-3, 4) if v != 0])
            g_val = c_ * x_val * x_val + d_
            ans = a * g_val + b
            # TeX: f(x) = ax + b,  g(x) = cx^2 + d — shown together.
            b_sig = self._signed(b).strip()
            d_sig = self._signed(d_).strip()
            expr_tex = (rf"${fn}(x) = {a}x {b_sig};  "
                        rf"{g_name}(x) = {c_}x^{{2}} {d_sig}$")
            fn_sig = f"{fn}(x) and {g_name}(x)"
            fn_val = f"{fn}({g_name}({x_val}))"

        elif kind == "deep_chain":
            # L9 iter-3: 4-level composition of affine/quadratic pieces.
            # f(x) = a1*x + b1
            # g(x) = a2*x + b2
            # h(x) = a3*x^2 + b3
            # k(x) = a4*x + b4
            # Ask: compute f(g(h(k(x0)))) — 4 nested substitutions.
            # Pick small coefficients so the final answer fits.
            for _ in range(20):
                a1 = rng.choice([-2, -1, 1, 2])
                b1 = rng.randint(-5, 5)
                a2 = rng.choice([-2, -1, 1, 2])
                b2 = rng.randint(-5, 5)
                a3 = rng.choice([1, 2])
                b3 = rng.randint(-4, 4)
                a4 = rng.choice([-2, -1, 1, 2])
                b4 = rng.randint(-4, 4)
                x0 = rng.randint(-3, 3)
                if x0 == 0:
                    x0 = 1
                k_val = a4 * x0 + b4
                h_val = a3 * k_val * k_val + b3
                g_val = a2 * h_val + b2
                f_val = a1 * g_val + b1
                if -5000 < f_val < 5000:
                    ans = f_val
                    break
            else:
                return None
            g_name = rng.choice([c for c in self._FN_POOL if c != fn])
            h_name = rng.choice([c for c in self._FN_POOL
                                  if c != fn and c != g_name])
            k_name = rng.choice([c for c in self._FN_POOL
                                  if c not in (fn, g_name, h_name)])
            b1_s = self._signed(b1).strip()
            b2_s = self._signed(b2).strip()
            b3_s = self._signed(b3).strip()
            b4_s = self._signed(b4).strip()
            expr_tex = (
                rf"${fn}(x) = {a1}x {b1_s};  "
                rf"{g_name}(x) = {a2}x {b2_s};  "
                rf"{h_name}(x) = {a3}x^{{2}} {b3_s};  "
                rf"{k_name}(x) = {a4}x {b4_s}$"
            )
            fn_sig = (f"{fn}(x), {g_name}(x), {h_name}(x), {k_name}(x)")
            fn_val = f"{fn}({g_name}({h_name}({k_name}({x0}))))"

        elif kind == "quartic_trivar":
            # Fourth-degree trivariate: f(x,y,z) = a*x^4 + b*y*z + c.
            v1, v2, v3 = rng.choice(self._VAR_POOL_TRIPLE)
            a = rng.choice([1, 2])
            b = rng.randint(1, 4)
            c = rng.randint(-cfg["coef_hi"], cfg["coef_hi"])
            x1 = rng.randint(-2, 3)
            x2 = rng.randint(-3, 4)
            x3 = rng.randint(-3, 4)
            if x1 == 0: x1 = 1
            ans = a * (x1 ** 4) + b * x2 * x3 + c
            a_s = "" if a == 1 else f"{a}"
            b_s = "" if b == 1 else f"{b}"
            expr_tex = (rf"${fn}({v1}, {v2}, {v3}) = "
                        rf"{a_s}{v1}^{{4}} + {b_s}{v2}{v3}{self._signed(c)}$")
            fn_sig = f"{fn}({v1}, {v2}, {v3})"
            fn_val = f"{fn}({x1}, {x2}, {x3})"

        else:
            return None

        if abs(ans) > 5000:
            return None

        # Question
        sidx = (self.seed or 0) % 16
        q = self._Q_POOL[sidx].format(fn_sig=fn_sig, fn_val=fn_val)

        if cfg.get("puzzle_pieces"):
            q = (q + " NOTE: the formula is split across multiple labeled "
                 "pieces in the image (each marked with [*]) — you must "
                 "mentally re-assemble them, and you may see decoy "
                 "definitions using OTHER function names (ignore those).")

        img = self._render_expression(
            expr_tex, cfg["fontsize"], rng,
            puzzle=cfg.get("puzzle_pieces", False),
            n_distractors=cfg.get("distractor_count", 0),
            primary_fn=fn)
        return q, str(int(ans)), img

    # ------------------------------------------------------------------ #
    @staticmethod
    def _signed(coef: int, is_first: bool = False) -> str:
        if is_first:
            return f"{coef}"
        return f" - {abs(coef)}" if coef < 0 else f" + {coef}"

    def _tex_linear(self, fn, var, a, b):
        parts = []
        if a == 1:
            parts.append(f"{var}")
        elif a == -1:
            parts.append(f"-{var}")
        else:
            parts.append(f"{a}{var}")
        if b != 0:
            parts.append(self._signed(b))
        return rf"${fn}({var}) = {''.join(parts)}$"

    def _tex_quadratic(self, fn, var, a, b, c):
        parts = []
        if a == 1:
            parts.append(f"{var}^{{2}}")
        elif a == -1:
            parts.append(f"-{var}^{{2}}")
        else:
            parts.append(f"{a}{var}^{{2}}")
        if b != 0:
            if b == 1:
                parts.append(f" + {var}")
            elif b == -1:
                parts.append(f" - {var}")
            else:
                parts.append(self._signed(b) + f"{var}")
        if c != 0:
            parts.append(self._signed(c))
        return rf"${fn}({var}) = {''.join(parts)}$"

    def _tex_cubic(self, fn, var, a, b, c):
        parts = []
        if a == 1:
            parts.append(f"{var}^{{3}}")
        elif a == -1:
            parts.append(f"-{var}^{{3}}")
        else:
            parts.append(f"{a}{var}^{{3}}")
        if b != 0:
            if b == 1:
                parts.append(f" + {var}")
            elif b == -1:
                parts.append(f" - {var}")
            else:
                parts.append(self._signed(b) + f"{var}")
        if c != 0:
            parts.append(self._signed(c))
        return rf"${fn}({var}) = {''.join(parts)}$"

    def _tex_multivar_linear(self, fn, v1, v2, a, b, c):
        parts = []
        if a == 1:
            parts.append(f"{v1}")
        elif a == -1:
            parts.append(f"-{v1}")
        else:
            parts.append(f"{a}{v1}")
        if b == 1:
            parts.append(f" + {v2}")
        elif b == -1:
            parts.append(f" - {v2}")
        else:
            parts.append(self._signed(b) + f"{v2}")
        if c != 0:
            parts.append(self._signed(c))
        return rf"${fn}({v1}, {v2}) = {''.join(parts)}$"

    def _tex_multivar_quad(self, fn, v1, v2, a, b, c):
        parts = [f"{a}{v1}^{{2}}"]
        if b == 1:
            parts.append(f" + {v2}")
        elif b == -1:
            parts.append(f" - {v2}")
        else:
            parts.append(self._signed(b) + f"{v2}")
        if c != 0:
            parts.append(self._signed(c))
        return rf"${fn}({v1}, {v2}) = {''.join(parts)}$"

    # ------------------------------------------------------------------ #
    def _render_expression(self, tex: str, fontsize: int, rng,
                           puzzle: bool = False,
                           n_distractors: int = 0,
                           primary_fn: str = "f") -> Image.Image:
        style = self._random_style()
        bg_choices = ["#ffffff", "#fefefe", "#fffdf7", "#f7fbff",
                      "#f6fff6", "#fdfaf6", "#fafdff", "#fffaf0",
                      "#f5f5f5", "#fcfbf7"]
        bg = rng.choice(bg_choices)
        frame_style = rng.choice(["round", "simple", "boxed", "shadow",
                                   "dashed", "thick"])
        font_family = rng.choice(["serif", "DejaVu Sans", "STIXGeneral",
                                   "monospace", "sans-serif"])
        fig_w = rng.uniform(6.2, 8.2)
        fig_h = (rng.uniform(2.1, 2.9) if not puzzle
                 else rng.uniform(3.6, 4.4))
        dpi_val = rng.choice([100, 105, 110, 115, 120, 125])
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi_val)
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        text_color = rng.choice(["#1a1a1a", "#0d3b66", "#3d2b1f",
                                  "#0b2545", "#1b4332", "#2d1b4e"])
        box_face = rng.choice(["#fffacd", "#fff2cc", "#fef6e4",
                                "#e0f7fa", "#e7f5e0", "#f2e7fa", "#ffe9e3"])
        box_edge = rng.choice(["#666", "#333", "#8b4513", "#2e4057",
                                "#4b0082", "#184e77"])

        if frame_style == "round":
            bbox = dict(boxstyle="round,pad=0.6", facecolor=box_face,
                        edgecolor=box_edge, linewidth=1.5)
        elif frame_style == "boxed":
            bbox = dict(boxstyle="square,pad=0.4", facecolor=box_face,
                        edgecolor=box_edge, linewidth=1.5)
        elif frame_style == "shadow":
            bbox = dict(boxstyle="round,pad=0.5", facecolor=box_face,
                        edgecolor=box_edge, linewidth=1.2)
        elif frame_style == "dashed":
            bbox = dict(boxstyle="round,pad=0.5", facecolor=box_face,
                        edgecolor=box_edge, linewidth=1.4,
                        linestyle="--")
        elif frame_style == "thick":
            bbox = dict(boxstyle="square,pad=0.5", facecolor=box_face,
                        edgecolor=box_edge, linewidth=2.5)
        else:
            bbox = None

        is_mathtext = tex.startswith("$") and tex.endswith("$")

        if not puzzle:
            # Vary text position and fontsize per seed for image diversity.
            px = 0.5 + rng.uniform(-0.06, 0.06)
            py = 0.5 + rng.uniform(-0.08, 0.08)
            fs_jitter = rng.uniform(-2, 2)
            ax.text(
                px, py, tex,
                ha="center", va="center",
                fontsize=(fontsize + fs_jitter) if is_mathtext
                         else max(14, fontsize + fs_jitter - 3),
                family=font_family if is_mathtext else "monospace",
                color=text_color, bbox=bbox,
            )
            # Optional decorative banner/underline.
            deco = rng.choice(["none", "none", "underline", "top_band",
                                "corner"])
            if deco == "underline":
                ax.plot([0.25, 0.75], [0.15, 0.15], color=box_edge, lw=1.8)
            elif deco == "top_band":
                ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.06,
                                            facecolor=box_edge, alpha=0.25,
                                            edgecolor="none"))
            elif deco == "corner":
                ax.add_patch(plt.Rectangle((0.02, 0.85), 0.10, 0.10,
                                            facecolor=box_edge, alpha=0.35,
                                            edgecolor="none"))
            return self.fig_to_pil(fig, dpi=dpi_val)

        # Puzzle-piece layout: split expr into 2-3 chunks placed around,
        # plus n_distractors decoy function defs also scattered.
        chunks = self._split_for_puzzle(tex, rng)
        positions = [(0.25, 0.75), (0.72, 0.82), (0.18, 0.3),
                     (0.78, 0.28), (0.5, 0.55)]
        rng.shuffle(positions)
        # Primary (correct) chunks first, each highlighted with a red star
        # so learner knows these compose the real formula.
        for i, ch in enumerate(chunks[:len(positions)]):
            px, py = positions[i]
            ax.text(px, py, "[*] " + ch,
                    ha="center", va="center",
                    fontsize=max(12, fontsize - 2),
                    family=font_family if ch.startswith("$") else "monospace",
                    color="#b71c1c",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#fff8dc",
                              edgecolor="#b71c1c", linewidth=1.2))
        # Decoys: similar-looking definitions with OTHER function names.
        decoy_fns = [c for c in self._FN_POOL if c != primary_fn]
        rng.shuffle(decoy_fns)
        decoy_positions = [(0.08, 0.9), (0.92, 0.5), (0.5, 0.08),
                           (0.1, 0.55), (0.9, 0.9)]
        rng.shuffle(decoy_positions)
        for i in range(min(n_distractors, len(decoy_positions), len(decoy_fns))):
            dfn = decoy_fns[i]
            # Make a random linear-ish decoy formula
            a_d = rng.randint(-7, 7) or 2
            b_d = rng.randint(-9, 9)
            dtex = f"${dfn}(x) = {a_d}x {'+ ' + str(b_d) if b_d >= 0 else '- ' + str(abs(b_d))}$"
            dx, dy = decoy_positions[i]
            ax.text(dx, dy, dtex, ha="center", va="center",
                    fontsize=max(10, fontsize - 4),
                    family=font_family, color="#666666",
                    alpha=0.85)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _split_for_puzzle(tex: str, rng):
        """Split a LaTeX-ish expression into 2-3 visual chunks.
        For multi-line piecewise we split by line; for single-line we
        split at a plus/minus boundary."""
        if "\n" in tex:
            return [ln.strip() for ln in tex.split("\n") if ln.strip()]
        # strip leading/trailing $
        inner = tex.strip("$").strip()
        parts = []
        # Try to split at ' + ' or ' - ' boundaries
        # Keep LaTeX \dfrac{...}{...} atomic
        buf = ""
        depth = 0
        for ch in inner:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            buf += ch
            if depth == 0 and ch in ("+", "-") and len(buf) > 4:
                # Break here - leading operator included in next chunk
                pass
        # Simple strategy: split by ' = ' first, then halve the RHS.
        if "=" in inner:
            lhs, _, rhs = inner.partition("=")
            parts.append(f"${lhs.strip()} =$")
            # Split rhs roughly in half at a '+' or '-' BUT only at brace-depth 0
            # so we don't cut through a \dfrac{...}{...}.
            rhs = rhs.strip()
            # Compute brace depth at each index.
            depth_at = [0] * (len(rhs) + 1)
            d = 0
            for i, ch in enumerate(rhs):
                if ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
                depth_at[i + 1] = d
            mid = len(rhs) // 2
            best = -1
            for i in range(max(1, mid - 8), min(len(rhs), mid + 8)):
                if rhs[i] in ("+", "-") and i > 0 and depth_at[i] == 0:
                    best = i
                    break
            if best <= 0:
                # Fallback: search wider for a depth-0 +/-
                for i in range(1, len(rhs)):
                    if rhs[i] in ("+", "-") and depth_at[i] == 0:
                        best = i
                        break
            if best > 0:
                parts.append(f"${rhs[:best].strip()}$")
                parts.append(f"${rhs[best:].strip()}$")
            else:
                parts.append(f"${rhs}$")
        else:
            parts = [tex]
        return parts
