"""
Implicit Function Level Set QA environment.

Shows a contour plot (level curves) of a scalar function f(x, y). Contour
lines are annotated with their f-values. A specific query point P is
marked on the plane, and the model must read/interpolate the value of f
at P.

Difficulty axes:
  - function_type: radial (circles) -> elliptic -> saddle -> periodic
  - contour_density (# labeled contours) and point_on_contour vs between
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class ImplicitFunctionLevelSetQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "implicit_function_level_set"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # function families expand with level
        if level <= 1:
            fams = ["radial"]
        elif level <= 3:
            fams = ["radial", "elliptic"]
        elif level <= 5:
            fams = ["elliptic", "saddle"]
        elif level <= 7:
            fams = ["saddle", "periodic"]
        else:
            fams = ["periodic"]
        # n_contours shrinks with level (fewer labeled references)
        n_contours = max(3, 8 - level // 2)      # 8 -> 3
        # interpolation required above L5
        require_interpolation = level >= 5
        return {
            "fams": fams,
            "n_contours": n_contours,
            "require_interpolation": require_interpolation,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1093)

        for _ in range(20):
            result = self._try_generate(sub_rng, cfg)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    def _build_func(self, fam: str, sub_rng: random.Random):
        """Return (f(x,y), label, grid_range). More variant subtypes per family."""
        if fam == "radial":
            a = sub_rng.choice([1, 2])
            b = sub_rng.choice([1, 2])
            shift_x = sub_rng.choice([0, 0, -1, 1])
            shift_y = sub_rng.choice([0, 0, -1, 1])
            f = lambda x, y, _a=a, _b=b, _sx=shift_x, _sy=shift_y: \
                _a * (x - _sx) ** 2 + _b * (y - _sy) ** 2
            sx = "" if shift_x == 0 else (f"-{shift_x}" if shift_x > 0 else f"+{-shift_x}")
            sy = "" if shift_y == 0 else (f"-{shift_y}" if shift_y > 0 else f"+{-shift_y}")
            ax_str = "x" if shift_x == 0 else f"(x{sx})"
            ay_str = "y" if shift_y == 0 else f"(y{sy})"
            label = f"f(x, y) = {a}{ax_str}² + {b}{ay_str}²"
            rng_val = 3.5
        elif fam == "elliptic":
            a = sub_rng.choice([1, 2])
            b = sub_rng.choice([x for x in [2, 3, 4] if x != a])
            f = lambda x, y, _a=a, _b=b: _a * x * x + _b * y * y
            label = f"f(x, y) = {a}x² + {b}y²"
            rng_val = 3.0
        elif fam == "saddle":
            a = sub_rng.choice([1, 2])
            b = sub_rng.choice([1, 2])
            sub = sub_rng.choice(["sub", "xy"])
            if sub == "sub":
                f = lambda x, y, _a=a, _b=b: _a * x * x - _b * y * y
                label = f"f(x, y) = {a}x² - {b}y²"
            else:
                f = lambda x, y: x * y
                label = "f(x, y) = x · y"
            rng_val = 3.0
        else:  # periodic
            mode = sub_rng.choice(["sincos", "doubsin", "cospx"])
            if mode == "sincos":
                f = lambda x, y: math.sin(x) * math.cos(y)
                label = "f(x, y) = sin(x) · cos(y)"
            elif mode == "doubsin":
                f = lambda x, y: math.sin(x) + math.sin(y)
                label = "f(x, y) = sin(x) + sin(y)"
            else:
                f = lambda x, y: math.cos(x + y)
                label = "f(x, y) = cos(x + y)"
            rng_val = 3.0
        return f, label, rng_val

    # ------------------------------------------------------------------ #
    def _try_generate(self, sub_rng: random.Random, cfg: Dict):
        fam = sub_rng.choice(cfg["fams"])
        f, label, rng_val = self._build_func(fam, sub_rng)

        # sample query point P
        P = None
        for _ in range(30):
            px = round(sub_rng.uniform(-rng_val + 0.5, rng_val - 0.5), 1)
            py = round(sub_rng.uniform(-rng_val + 0.5, rng_val - 0.5), 1)
            v = f(px, py)
            # avoid points too close to zero/degenerate
            if abs(v) < 0.05 and fam != "periodic":
                continue
            if abs(px) < 0.4 and abs(py) < 0.4:
                continue
            P = (px, py, v)
            break
        if P is None:
            return None
        px, py, v_true = P
        v_true_r = round(v_true, 2)

        # choose contour levels to show (include nearby levels to query point)
        # generate candidate contour levels spanning range
        n_c = cfg["n_contours"]
        # Sample f values on grid to pick reasonable contours
        grid = np.linspace(-rng_val, rng_val, 80)
        XX, YY = np.meshgrid(grid, grid)
        ZZ = np.vectorize(f)(XX, YY)
        z_min, z_max = float(ZZ.min()), float(ZZ.max())
        # uniform contour spacing
        levels = np.linspace(z_min + 0.1 * (z_max - z_min),
                             z_max - 0.1 * (z_max - z_min), n_c)
        # round to 1 decimal
        levels = sorted(set(round(float(l), 1) for l in levels))
        if len(levels) < 3:
            return None

        # Decide: point on contour (exact) or between contours (interpolation)
        if cfg["require_interpolation"]:
            # place contour levels so P is between two
            # pick two bracketing levels offset from v_true
            levels_arr = np.array(levels)
            # Ensure none equals v_true_r
            levels = [l for l in levels if abs(l - v_true_r) > 0.1]
            if len(levels) < 3:
                return None
        else:
            # make sure one of the levels equals v_true_r
            # adjust: replace closest level with v_true_r
            if len(levels) >= 3:
                closest_i = int(np.argmin(np.abs(np.array(levels) - v_true_r)))
                new_levels = list(levels)
                new_levels[closest_i] = round(v_true_r, 1)
                levels = sorted(set(new_levels))

        # Build MCQ options — 4 numeric options, one is ≈ v_true
        correct = v_true_r
        distractors = set()
        tries = 0
        while len(distractors) < 3 and tries < 60:
            tries += 1
            # pick from nearby contour levels or perturbations
            base = sub_rng.choice(levels)
            dv = sub_rng.choice([-0.5, -0.3, 0.3, 0.5, 0.8, -0.8, 1.0, -1.0])
            cand = round(base + dv, 2)
            if abs(cand - correct) < 0.2:
                continue
            if cand in distractors:
                continue
            # avoid near-duplicate distractors
            if any(abs(cand - d) < 0.2 for d in distractors):
                continue
            distractors.add(cand)
        if len(distractors) < 3:
            return None

        options = list(distractors) + [correct]
        sub_rng.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = chr(ord("A") + correct_idx)
        opt_str = ", ".join(f"{chr(ord('A') + i)}) {v:.2f}" for i, v in enumerate(options))

        # Render
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.2 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        cmap = sub_rng.choice(["viridis", "plasma", "coolwarm", "RdYlBu", "magma"])
        # draw filled contours as background (very subtle)
        ax.contourf(XX, YY, ZZ, levels=20, cmap=cmap, alpha=0.25)
        # draw labeled contour lines
        cs = ax.contour(XX, YY, ZZ, levels=levels, colors="black", linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")

        # mark P
        ax.plot(px, py, marker="*", markersize=18, color="#e60000",
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.annotate(f"P({px}, {py})", xy=(px, py),
                    xytext=(px + 0.15, py + 0.18),
                    fontsize=11, color="#1a1a1a", fontweight="bold")

        ax.set_xlim(-rng_val, rng_val)
        ax.set_ylim(-rng_val, rng_val)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.axvline(0, color="#555555", linewidth=0.7)
        ax.grid(True, color="#dddddd", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        # Title must not reveal the analytic formula — only labelled contours
        # may be used. (Text-leakage fix 2026-04-17: previously the formula
        # e.g. "f(x,y)=2x²+2(y-1)²" was printed, allowing direct computation.)
        ax.set_title("Contour plot of f(x, y)", fontsize=11, fontweight="bold")

        q_pool = [
            (f"The contour plot shows level curves of a function f(x, y), with "
             f"each curve labeled by its value. Point P (marked with a red star) "
             f"is plotted on the figure. What is the approximate value of f(P)? "
             f"{opt_str}. Answer with a single letter."),
            (f"A contour plot of f(x, y) is shown with labelled level curves. "
             f"The query point P appears as a red star. Estimate f(P) and select "
             f"the closest option. {opt_str}. Answer with a single letter."),
            (f"Use the labelled contour lines to read off the function value at "
             f"the marked red-star point P. {opt_str}. Answer with a single letter."),
        ]
        question = sub_rng.choice(q_pool)
        # 2026-05-04: simplified L0 (was 5% too-hard) — at low levels a hint
        # tells the model exactly how to read the answer off the labels.
        # require_interpolation=False at L0/L1 so P sits on a labeled curve.
        if not cfg["require_interpolation"]:
            question += (
                " Be concise. The red star P lies on (or extremely near) one "
                "of the labelled black contour lines; read that label and "
                "match it to the closest option."
            )
        answer = correct_letter

        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question, answer, img
