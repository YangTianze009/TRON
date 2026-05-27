"""
Algebra Robustness QA (v3 diversity redesign, 2026-04-16).

Task C — dynamic-math algebra gap. Renders algebraic equations with visual
variation (different fonts, positioning, decorations) and asks the model
to solve.

v3 diversity redesign:
  * 8 equation families: linear (3 forms), quadratic (2 forms),
    system of 2 equations, absolute-value, proportion, linear
    inequality.
  * Layouts: box, banner, scroll, margin note, circle badge.
  * Random font choice, font size, rotation jitter, background noise,
    framing decorations.
  * L0 = simple linear with small integer coefficients, no rotation, no
    noise, readable font.
  * L9 = system / absolute value with rotation, noise, decorative
    frames.
  * Question phrasing variants: 6 per operation type.
  * Variables can be x, y, t, m, n (per seed).
  * Constants are chosen from wide ranges so answers vary per seed.

No text leakage: problem asks "solve for X as shown", and the equation
appears in the image (never in question text).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_VARIABLES = ["x", "y", "t", "m", "n", "z", "w"]

_FONT_FAMILIES = ["serif", "DejaVu Sans", "monospace", "sans-serif"]

_PALETTE_POOL = [
    ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"],
    ["#4285f4", "#ea4335", "#fbbc05", "#34a853", "#46bdc6"],
    ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],
    ["#1d3557", "#457b9d", "#a8dadc", "#e63946", "#2b2d42"],
]

_LINEAR_STEMS = [
    "Solve for {var} in the equation shown in the image.",
    "The image shows an equation. Find {var}.",
    "Read the equation from the image and compute {var}.",
    "An equation appears in the image. Determine the value of {var}.",
    "Using the equation shown, solve for {var}.",
    "What is {var} in the equation displayed? Answer with a single integer.",
]

_QUAD_STEMS = [
    "The image shows a quadratic equation. What is the LARGER root? Answer with an integer.",
    "A quadratic equation is displayed. Compute the GREATER root.",
    "The image shows a quadratic. What is the maximum solution?",
    "Solve the quadratic shown and report the larger root (integer).",
    "The image shows a quadratic. Find the bigger root.",
    "Study the quadratic in the image. What is the LARGER value of the variable that satisfies it?",
]

_SYS_STEMS = [
    "The image shows a system of two equations. Solve for {var}. Answer with a single integer.",
    "Two equations appear in the image. Find {var}.",
    "Using the system of equations shown, determine {var}.",
    "Solve the system shown in the image for {var}.",
    "The image shows two simultaneous equations. What is {var}?",
    "Find {var} by solving the system of equations in the image.",
]

_ABS_STEMS = [
    "The image shows an absolute-value equation. Solve for {var} (report the POSITIVE solution).",
    "An absolute-value expression is shown. Find the positive value of {var}.",
    "Compute the positive {var} that satisfies the equation in the image.",
    "The image shows |a{var} + b| = c. Find the positive solution for {var}.",
]

_PROP_STEMS = [
    "The image shows a proportion a/b = c/{var}. Find {var}.",
    "Solve the proportion displayed for {var}.",
    "A cross-multiplication problem appears in the image. Find {var}.",
]

_INEQ_STEMS = [
    "The image shows a linear inequality: a{var} + b > c. Find the SMALLEST integer {var} that satisfies it.",
    "An inequality appears in the image. What is the smallest integer {var} that makes it true?",
    "The image shows an inequality in {var}. Compute the smallest integer value of {var} for which it holds.",
]

def _coeff_str(coef, var, first=False):
    """Render 'a*x' with sign."""
    if coef == 0:
        return ""
    sign = "" if (coef > 0 and first) else ("+" if coef > 0 else "-")
    mag = abs(coef)
    if mag == 1:
        body = var
    else:
        body = f"{mag}{var}"
    if first and sign == "":
        return body
    return f"{sign} {body}"

def _const_str(v, first=False):
    if v == 0 and first:
        return "0"
    if v == 0:
        return ""
    sign = "" if (v > 0 and first) else ("+" if v > 0 else "-")
    return (f"{abs(v)}" if first and sign == "" else
            f"{sign} {abs(v)}")

class AlgebraRobustnessQA(StandaloneVisualEnv):
    ENV_NAME = "algebra_robustness"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "eq_types": ["linear_ax_b_c"],
                "rotation_range": 0,
                "noise_level": 0.0,
                "font_variety": True,
                "coef_range": (1, 6),
                "decorations": True,
                "layout_pool": ["box", "banner", "scroll"],
            }
        if level <= 3:
            return {
                "eq_types": ["linear_ax_b_c", "linear_ab_c"],
                "rotation_range": 4,
                "noise_level": 0.05,
                "font_variety": False,
                "coef_range": (1, 9),
                "decorations": False,
                "layout_pool": ["box", "banner"],
            }
        if level <= 5:
            return {
                "eq_types": ["linear_ax_b_c", "linear_ab_c", "proportion",
                              "quadratic_std"],
                "rotation_range": 10,
                "noise_level": 0.12,
                "font_variety": True,
                "coef_range": (1, 12),
                "decorations": True,
                "layout_pool": ["box", "banner", "scroll"],
            }
        if level <= 7:
            return {
                "eq_types": ["quadratic_std", "quadratic_leading",
                              "abs_value", "proportion", "inequality"],
                "rotation_range": 15,
                "noise_level": 0.2,
                "font_variety": True,
                "coef_range": (1, 14),
                "decorations": True,
                "layout_pool": ["box", "banner", "scroll", "circle_badge"],
                "n_distractors": 2,   # L6-L7: show 2 similar distractor eqs
                "distractor_rotation": 25,
            }
        return {
            "eq_types": ["system", "quadratic_leading", "abs_value",
                          "inequality"],
            "rotation_range": 18,
            "noise_level": 0.28,
            "font_variety": True,
            "coef_range": (1, 15),
            "decorations": True,
            "layout_pool": ["box", "banner", "scroll", "circle_badge",
                            "margin_note"],
            "n_distractors": 3,   # L8-L9: 3 distractor equations
            "distractor_rotation": 35,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level

        eq_type = rng.choice(cfg["eq_types"])
        var = rng.choice(_VARIABLES)
        return self._build(eq_type, var, cfg, rng)

    # ---------------------------- builders ---------------------------- #

    def _build(self, eq_type, var, cfg, rng):
        lo, hi = cfg["coef_range"]

        if eq_type == "linear_ax_b_c":
            a = rng.choice([-hi, -hi // 2, -3, -2, 2, 3, hi // 2, hi])
            if a == 0:
                a = 2
            x = rng.randint(-8, 8)
            b = rng.randint(-10, 10)
            c = a * x + b
            eq = f"${_coeff_str(a, var, first=True)} {_const_str(b)} = {c}$"
            stem = rng.choice(_LINEAR_STEMS).format(var=var)
            question = f"{stem} Answer with a single integer."
            answer = str(x)

        elif eq_type == "linear_ab_c":
            # a(x + b) = c
            a = rng.choice([-5, -3, -2, 2, 3, 4, 5])
            x = rng.randint(-7, 7)
            b = rng.randint(-8, 8)
            while b == 0:
                b = rng.randint(-8, 8)
            c = a * (x + b)
            sign_b = "+" if b > 0 else "-"
            eq = f"${a}({var} {sign_b} {abs(b)}) = {c}$"
            stem = rng.choice(_LINEAR_STEMS).format(var=var)
            question = f"{stem} Answer with a single integer."
            answer = str(x)

        elif eq_type == "quadratic_std":
            r1 = rng.randint(-5, 5)
            r2 = rng.randint(-5, 5)
            b = -(r1 + r2)
            c = r1 * r2
            bx = _coeff_str(b, var) if b != 0 else ""
            cc = _const_str(c) if c != 0 else ""
            eq = f"${var}^2 {bx} {cc} = 0$"
            stem = rng.choice(_QUAD_STEMS)
            question = stem
            answer = str(max(r1, r2))

        elif eq_type == "quadratic_leading":
            a = rng.choice([2, 3, -2, -3])
            r1 = rng.randint(-4, 4)
            r2 = rng.randint(-4, 4)
            # a(x-r1)(x-r2) = 0
            b = -a * (r1 + r2)
            c = a * r1 * r2
            bx = _coeff_str(b, var) if b != 0 else ""
            cc = _const_str(c) if c != 0 else ""
            eq = f"${_coeff_str(a, f'{var}^2', first=True)} {bx} {cc} = 0$"
            stem = rng.choice(_QUAD_STEMS)
            question = stem
            answer = str(max(r1, r2))

        elif eq_type == "system":
            var2 = rng.choice([v for v in _VARIABLES if v != var])
            x = rng.randint(-5, 5)
            y = rng.randint(-5, 5)
            s = x + y
            a2 = rng.choice([2, 3])
            b2 = rng.choice([-2, -1, 1, 2])
            while a2 == b2:
                b2 = rng.choice([-3, -2, -1, 1, 2, 3])
            t = a2 * x + b2 * y
            sign_b2 = "+" if b2 >= 0 else "-"
            eq = (f"${var} + {var2} = {s}$\n"
                  f"${a2}{var} {sign_b2} {abs(b2)}{var2} = {t}$")
            stem = rng.choice(_SYS_STEMS).format(var=var)
            question = f"{stem}"
            answer = str(x)

        elif eq_type == "abs_value":
            a = rng.choice([1, 2, 3, -1, -2, -3])
            if a == 0:
                a = 1
            x = rng.randint(1, 8)
            b = rng.randint(-5, 5)
            c_rhs = abs(a * x + b)
            if c_rhs == 0:
                c_rhs = 1
            if b == 0:
                b_body = ""
            else:
                sign_b = "+" if b > 0 else "-"
                b_body = f" {sign_b} {abs(b)}"
            eq = f"$|{_coeff_str(a, var, first=True)}{b_body}| = {c_rhs}$"
            stem = rng.choice(_ABS_STEMS).format(var=var)
            question = f"{stem} Answer with a single integer."
            answer = str(x)

        elif eq_type == "proportion":
            # a/b = c/var
            b = rng.randint(2, 10)
            x = rng.randint(2, 10)
            # pick k so that a*x/b = c is integer
            k = rng.randint(2, 6)
            a = k * b
            c = k * x
            eq = f"$\\dfrac{{{a}}}{{{b}}} = \\dfrac{{{c}}}{{{var}}}$"
            stem = rng.choice(_PROP_STEMS).format(var=var)
            question = f"{stem} Answer with a single integer."
            answer = str(x)

        elif eq_type == "inequality":
            a = rng.choice([1, 2, 3, 4, 5])
            b = rng.randint(-10, 10)
            c = rng.randint(-10, 10)
            # ax + b > c -> x > (c - b) / a; smallest int x is
            # floor((c - b)/a) + 1
            import math as _m
            threshold = (c - b) / a
            smallest = int(_m.floor(threshold)) + 1
            if b == 0:
                b_body = ""
            else:
                sign_b = "+" if b > 0 else "-"
                b_body = f" {sign_b} {abs(b)}"
            eq = f"${_coeff_str(a, var, first=True)}{b_body} > {c}$"
            stem = rng.choice(_INEQ_STEMS).format(var=var)
            question = f"{stem} Answer with a single integer."
            answer = str(smallest)
        else:
            return None

        layout = rng.choice(cfg["layout_pool"])
        # Build decoy equations (visually similar but irrelevant) at L6+.
        n_decoys = cfg.get("n_distractors", 0)
        decoys = []
        if n_decoys > 0:
            decoys = self._build_decoys(eq, eq_type, var, cfg, rng, n_decoys)
            # Wrap primary equation with a marker so the question can
            # reference THE equation (we use a small arrow/star badge
            # drawn in the render; question text mentions it).
            question = (question + " The equation to use is the one "
                        "with a red star (\u2605) next to it; the others "
                        "are decoys.")
        img = self._render(eq, cfg, rng, layout, decoys=decoys)
        return question, answer, img

    def _build_decoys(self, main_eq, eq_type, var, cfg, rng, n):
        """Create similar-looking decoy equations with small perturbations."""
        decoys = []
        tries = 0
        while len(decoys) < n and tries < 25:
            tries += 1
            other_var = rng.choice([v for v in _VARIABLES if v != var])
            hi = cfg["coef_range"][1]
            a2 = rng.choice([-hi, -3, -2, 2, 3, hi // 2, hi])
            b2 = rng.randint(-10, 10)
            c2 = rng.randint(-15, 15)
            if eq_type in ("linear_ax_b_c", "linear_ab_c"):
                cand = f"${_coeff_str(a2, other_var, first=True)} {_const_str(b2)} = {c2}$"
            elif eq_type in ("quadratic_std", "quadratic_leading"):
                bx = _coeff_str(b2, other_var) if b2 != 0 else ""
                cc = _const_str(c2) if c2 != 0 else ""
                cand = f"${other_var}^2 {bx} {cc} = 0$"
            elif eq_type == "abs_value":
                cand = f"$|{_coeff_str(a2, other_var, first=True)} {_const_str(b2)}| = {abs(c2) + 1}$"
            elif eq_type == "proportion":
                d = rng.randint(2, 9)
                cand = f"$\\dfrac{{{abs(a2)}}}{{{d}}} = \\dfrac{{{abs(c2) + 1}}}{{{other_var}}}$"
            elif eq_type == "inequality":
                cand = f"${_coeff_str(a2, other_var, first=True)} {_const_str(b2)} > {c2}$"
            elif eq_type == "system":
                cand = (f"${other_var} + x = {b2 + c2}$\n"
                        f"${abs(a2)}{other_var} - x = {c2}$")
            else:
                cand = f"${_coeff_str(a2, other_var, first=True)} = {c2}$"
            if cand != main_eq and cand not in decoys:
                decoys.append(cand)
        return decoys

    # ---------------------------- rendering ---------------------------- #

    def _render(self, eq_str, cfg, rng, layout, decoys=None):
        style = self._random_style()
        palette = rng.choice(_PALETTE_POOL)
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(7.5 * sc, 4 * sc))
        bg = style["bg_color"]
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        font_family = (rng.choice(_FONT_FAMILIES) if cfg["font_variety"]
                       else "serif")

        # Noise dots
        if cfg["noise_level"] > 0:
            for _ in range(int(cfg["noise_level"] * 100)):
                nx = rng.uniform(0, 1)
                ny = rng.uniform(0, 1)
                gray = rng.randint(200, 255)
                col = f"#{gray:02x}{gray:02x}{gray:02x}"
                ax.plot(nx, ny, ".", color=col,
                        markersize=rng.uniform(1, 4))

        rotation = rng.uniform(-cfg["rotation_range"],
                               cfg["rotation_range"])
        lines = eq_str.split("\n")
        base_fs = rng.choice([18, 20, 22, 24])

        if layout == "box":
            self._layout_box(ax, lines, rotation, font_family, base_fs,
                              palette, cfg, rng)
        elif layout == "banner":
            self._layout_banner(ax, lines, rotation, font_family, base_fs,
                                 palette, cfg, rng)
        elif layout == "scroll":
            self._layout_scroll(ax, lines, rotation, font_family, base_fs,
                                 palette, cfg, rng)
        elif layout == "circle_badge":
            self._layout_badge(ax, lines, rotation, font_family, base_fs,
                                palette, cfg, rng)
        elif layout == "margin_note":
            self._layout_margin(ax, lines, rotation, font_family, base_fs,
                                 palette, cfg, rng)
        else:
            self._layout_box(ax, lines, rotation, font_family, base_fs,
                              palette, cfg, rng)

        # Mark the primary equation with a red star (reliable visual anchor).
        if decoys:
            ax.text(0.06, 0.55, "\u2605",
                    fontsize=28, color="#d62728",
                    ha="center", va="center", fontweight="bold")

        # Scatter decoy equations around the primary one, with stronger
        # rotation and slightly smaller font so the learner must visually
        # identify which eq is labeled.
        if decoys:
            decoy_rot_range = cfg.get("distractor_rotation", 25)
            decoy_fs = max(12, base_fs - 6)
            # Positions spread around the edges
            decoy_positions = [(0.18, 0.88), (0.8, 0.88), (0.15, 0.12),
                               (0.82, 0.15), (0.5, 0.90), (0.5, 0.08)]
            rng.shuffle(decoy_positions)
            for i, d_eq in enumerate(decoys[:len(decoy_positions)]):
                dx, dy = decoy_positions[i]
                drot = rng.uniform(-decoy_rot_range, decoy_rot_range)
                for li, d_line in enumerate(d_eq.split("\n")):
                    ax.text(dx, dy - li * 0.05, d_line,
                            fontsize=decoy_fs, ha="center", va="center",
                            family=font_family, color="#555",
                            rotation=drot, alpha=0.85)

        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _layout_box(self, ax, lines, rot, ff, fs, palette, cfg, rng):
        if cfg["decorations"]:
            rect = mpatches.FancyBboxPatch(
                (0.15, 0.25), 0.7, 0.5,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor="#ffffff",
                edgecolor=palette[rng.randint(0, len(palette) - 1)],
                linewidth=1.5)
            ax.add_patch(rect)
        n = len(lines)
        for i, line in enumerate(lines):
            y_pos = 0.55 + (n - 1) * 0.12 - i * 0.24
            ax.text(0.5, y_pos, line, ha="center", va="center",
                    fontsize=fs, family=ff, color="#1a1a1a",
                    rotation=rot)

    def _layout_banner(self, ax, lines, rot, ff, fs, palette, cfg, rng):
        col = palette[rng.randint(0, len(palette) - 1)]
        if cfg["decorations"]:
            rect = mpatches.FancyBboxPatch(
                (0.03, 0.4), 0.94, 0.2,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=col, edgecolor=col, alpha=0.18, linewidth=0)
            ax.add_patch(rect)
        n = len(lines)
        for i, line in enumerate(lines):
            y_pos = 0.55 + (n - 1) * 0.1 - i * 0.18
            ax.text(0.5, y_pos, line, ha="center", va="center",
                    fontsize=fs, family=ff, color="#2c3e50", rotation=rot)

    def _layout_scroll(self, ax, lines, rot, ff, fs, palette, cfg, rng):
        col = palette[rng.randint(0, len(palette) - 1)]
        if cfg["decorations"]:
            # Draw a rounded scroll shape
            rect = mpatches.FancyBboxPatch(
                (0.12, 0.28), 0.76, 0.5,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor="#fdf6e3", edgecolor=col, linewidth=1.6)
            ax.add_patch(rect)
            # decorative dots
            for _ in range(8):
                dx = rng.uniform(0.15, 0.85)
                dy = rng.uniform(0.30, 0.76)
                ax.plot(dx, dy, ".", color=col, alpha=0.15, markersize=2)
        n = len(lines)
        for i, line in enumerate(lines):
            y_pos = 0.58 + (n - 1) * 0.1 - i * 0.2
            ax.text(0.5, y_pos, line, ha="center", va="center",
                    fontsize=fs, family=ff, color="#2c3e50", rotation=rot)

    def _layout_badge(self, ax, lines, rot, ff, fs, palette, cfg, rng):
        col = palette[rng.randint(0, len(palette) - 1)]
        ax.add_patch(mpatches.Circle((0.5, 0.5), 0.38,
                                      facecolor="#ffffff",
                                      edgecolor=col, linewidth=2.5))
        ax.add_patch(mpatches.Circle((0.5, 0.5), 0.32,
                                      facecolor="none",
                                      edgecolor=col, linewidth=0.8,
                                      linestyle="--", alpha=0.6))
        n = len(lines)
        sub_fs = max(10, fs - 4)
        for i, line in enumerate(lines):
            y_pos = 0.55 + (n - 1) * 0.1 - i * 0.18
            ax.text(0.5, y_pos, line, ha="center", va="center",
                    fontsize=sub_fs, family=ff, color="#2c3e50",
                    rotation=rot)

    def _layout_margin(self, ax, lines, rot, ff, fs, palette, cfg, rng):
        col = palette[rng.randint(0, len(palette) - 1)]
        ax.plot([0.12, 0.12], [0.18, 0.88], color=col, linewidth=2.4)
        for y in np.linspace(0.2, 0.85, 6):
            ax.plot([0.08, 0.16], [y, y], color=col,
                     linewidth=0.8, alpha=0.4)
        n = len(lines)
        sub_fs = max(14, fs - 2)
        for i, line in enumerate(lines):
            y_pos = 0.6 + (n - 1) * 0.1 - i * 0.2
            ax.text(0.55, y_pos, line, ha="center", va="center",
                    fontsize=sub_fs, family=ff, color="#1a1a1a",
                    rotation=rot)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_alg"
    os.makedirs(out_dir, exist_ok=True)
    env = AlgebraRobustnessQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[algebra_robustness L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"algebra_robustness_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[algebra_robustness L{level} s{s}] A={env._answer}")
