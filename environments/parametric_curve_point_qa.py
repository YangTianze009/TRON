"""
Parametric Curve Point QA environment.

Shows a parametric curve (x(t), y(t)) on the Cartesian plane with gridlines
and t-values annotated at several points along the curve. The model is
asked to identify the approximate coordinates (x, y) at a specific t value.

Difficulty axes:
  - curve_complexity: circle -> ellipse -> Lissajous -> epicycloid
  - n_labeled_t_points and grid_spacing (fewer/coarser reference points)
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

class ParametricCurvePointQA(StandaloneVisualEnv):
    ENV_NAME = "parametric_curve_point"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        """Two axes: curve family + reference-point / grid density."""
        level = max(0, min(9, level))
        # curve pool expands with level
        if level <= 1:
            curves = ["circle"]
        elif level <= 3:
            curves = ["circle", "ellipse"]
        elif level <= 5:
            curves = ["ellipse", "lissajous_simple"]
        elif level <= 7:
            curves = ["lissajous_simple", "lissajous"]
        else:
            curves = ["lissajous", "epicycloid"]
        n_labeled = max(2, 6 - level // 2)                 # 6 at L0 -> 2 at L8-9
        grid_spacing = round(0.25 + 0.1 * level, 2)         # 0.25 -> 1.15
        if grid_spacing > 1.0:
            grid_spacing = 1.0
        # t-query interpolation only starting at higher levels
        interpolate_t = level >= 4
        # L6+: force decimal/odd t values that cannot be evaluated symbolically
        odd_t_query = level >= 6
        # L7+: hide the explicit formula label so the model must use the plot
        hide_formula = level >= 7
        # L8+: tighter distractors near the correct point
        tight_distractors = level >= 8
        return {
            "curves": curves,
            "n_labeled": n_labeled,
            "grid_spacing": grid_spacing,
            "interpolate_t": interpolate_t,
            "odd_t_query": odd_t_query,
            "hide_formula": hide_formula,
            "tight_distractors": tight_distractors,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 911)

        for _ in range(20):
            result = self._try_generate(sub_rng, level, cfg)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Curve generation
    # ------------------------------------------------------------------ #
    def _build_curve(self, curve_type: str, sub_rng: random.Random):
        """Return (x_func, y_func, label, t_reference_marks, t_query_candidates)."""
        if curve_type == "circle":
            a = sub_rng.choice([1, 2, 3])
            b = a
            x_func = lambda t: a * math.cos(t)
            y_func = lambda t: b * math.sin(t)
            label = f"x = {a}cos(t),  y = {a}sin(t)"
            t_ref_base = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
            query_pool = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
        elif curve_type == "ellipse":
            a = sub_rng.choice([2, 3, 4])
            b = sub_rng.choice([x for x in [1, 2, 3] if x != a])
            x_func = lambda t, _a=a: _a * math.cos(t)
            y_func = lambda t, _b=b: _b * math.sin(t)
            label = f"x = {a}cos(t),  y = {b}sin(t)"
            t_ref_base = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
            query_pool = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
        elif curve_type == "lissajous_simple":
            a = sub_rng.choice([2, 3])
            b = sub_rng.choice([1, 2])
            x_func = lambda t, _a=a: _a * math.sin(t)
            y_func = lambda t, _b=b: _b * math.cos(t)
            label = f"x = {a}sin(t),  y = {b}cos(t)"
            t_ref_base = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
            query_pool = [math.pi / 4, 3 * math.pi / 4, math.pi / 3, 2 * math.pi / 3]
        elif curve_type == "lissajous":
            nx = sub_rng.choice([2, 3])
            ny = sub_rng.choice([1, 2, 3])
            # avoid nx == ny (degenerate circle)
            if nx == ny:
                ny = (ny % 3) + 1
            x_func = lambda t, _n=nx: math.sin(_n * t)
            y_func = lambda t, _n=ny: math.sin(_n * t + math.pi / 6)
            label = f"x = sin({nx}t),  y = sin({ny}t + π/6)"
            t_ref_base = [0.0, math.pi / 2, math.pi]
            query_pool = [math.pi / 4, math.pi / 3, 3 * math.pi / 4]
        else:  # epicycloid
            R = sub_rng.choice([2, 3])
            r = 1
            x_func = lambda t, _R=R, _r=r: (_R + _r) * math.cos(t) - _r * math.cos(((_R + _r) / _r) * t)
            y_func = lambda t, _R=R, _r=r: (_R + _r) * math.sin(t) - _r * math.sin(((_R + _r) / _r) * t)
            label = f"Epicycloid (R={R}, r={r})"
            t_ref_base = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
            query_pool = [math.pi / 4, math.pi / 3, 2 * math.pi / 3]
        return x_func, y_func, label, t_ref_base, query_pool

    # ------------------------------------------------------------------ #
    def _try_generate(self, sub_rng: random.Random, level: int, cfg: Dict):
        curve_type = sub_rng.choice(cfg["curves"])
        x_func, y_func, label, t_ref_base, query_pool = self._build_curve(curve_type, sub_rng)

        # reference t marks visible on the curve
        n_labeled = cfg["n_labeled"]
        if n_labeled <= len(t_ref_base):
            t_refs = sorted(sub_rng.sample(t_ref_base, n_labeled))
        else:
            extra = [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]
            pool = sorted(set(t_ref_base + extra))
            t_refs = sorted(sub_rng.sample(pool, min(n_labeled, len(pool))))

        # choose a query t value
        if cfg.get("odd_t_query"):
            # Decimal t values: cannot be evaluated symbolically, must read graph
            odd_pool = [0.6, 0.85, 1.1, 1.35, 1.6, 1.85, 2.1, 2.35,
                        2.6, 2.85, 3.4, 3.65, 3.9, 4.2, 4.5, 4.85,
                        5.1, 5.4, 5.65]
            t_q = sub_rng.choice(odd_pool)
        elif cfg["interpolate_t"] or level >= 4:
            t_q = sub_rng.choice(query_pool)
        else:
            # Easy levels: use a reference-mark t
            t_q = sub_rng.choice([t for t in t_refs if t > 0]) if any(t > 0 for t in t_refs) else t_refs[0]

        x_q = x_func(t_q)
        y_q = y_func(t_q)

        # Generate 4 coordinate-pair options (MCQ)
        correct = (round(x_q, 2), round(y_q, 2))

        def _fmt(p):
            return (round(p[0], 2), round(p[1], 2))

        # distractor candidates — perturb coordinates in semantically plausible ways
        tight = cfg.get("tight_distractors", False)
        distractors = set()
        tries = 0
        while len(distractors) < 3 and tries < 80:
            tries += 1
            # swap x/y, negate, shift, or use nearby point on curve
            if tight:
                mode = sub_rng.choice(["nearby_t", "nearby_t", "small_shift", "swap"])
            else:
                mode = sub_rng.choice(["swap", "negx", "negy", "nearby_t", "shift"])
            if mode == "swap":
                cand = (correct[1], correct[0])
            elif mode == "negx":
                cand = (-correct[0], correct[1])
            elif mode == "negy":
                cand = (correct[0], -correct[1])
            elif mode == "nearby_t":
                # Tight: small offset (< π/8) -> visually adjacent to correct point
                if tight:
                    delta = sub_rng.choice([math.pi / 12, math.pi / 10, math.pi / 8])
                else:
                    delta = sub_rng.choice([math.pi / 6, math.pi / 4, math.pi / 3])
                t2 = t_q + sub_rng.choice([-1, 1]) * delta
                cand = (x_func(t2), y_func(t2))
            elif mode == "small_shift":
                dx = sub_rng.choice([-0.3, -0.2, 0.2, 0.3])
                dy = sub_rng.choice([-0.3, -0.2, 0.2, 0.3])
                cand = (correct[0] + dx, correct[1] + dy)
            else:  # shift
                dx = sub_rng.choice([-1.0, -0.5, 0.5, 1.0])
                dy = sub_rng.choice([-1.0, -0.5, 0.5, 1.0])
                cand = (correct[0] + dx, correct[1] + dy)
            cand = _fmt(cand)
            if cand == correct:
                continue
            # avoid near-duplicates: tighter threshold when tight
            min_dup = 0.08 if tight else 0.15
            dup = False
            for d in distractors:
                if abs(d[0] - cand[0]) < min_dup and abs(d[1] - cand[1]) < min_dup:
                    dup = True
                    break
            if dup:
                continue
            distractors.add(cand)
        if len(distractors) < 3:
            return None

        options = list(distractors) + [correct]
        sub_rng.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = chr(ord("A") + correct_idx)

        def _fmt_opt(p):
            return f"({p[0]:.2f}, {p[1]:.2f})"

        opt_str = ", ".join(f"{chr(ord('A') + i)}) {_fmt_opt(p)}" for i, p in enumerate(options))

        # Render
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6.0 * sc, 6.0 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        # plot whole curve
        ts = np.linspace(0, 2 * math.pi, 600)
        xs = np.array([x_func(t) for t in ts])
        ys = np.array([y_func(t) for t in ts])
        color = style["palette"][0]
        ax.plot(xs, ys, color=color, linewidth=2.2)

        # Mark the QUERY point with a bold "?" marker so the student can
        # locate where on the curve they are asked to read coordinates.
        # 2026-04-17 fix: prior version only marked reference t's; the
        # query point was unmarked, making L7+ (hide_formula + odd_t) often
        # under-determined especially on Lissajous/epicycloid (non-monotonic).
        xq_pt = x_func(t_q)
        yq_pt = y_func(t_q)
        ax.plot(xq_pt, yq_pt, marker="*", color="#e74c3c",
                markersize=18, zorder=10,
                markeredgecolor="black", markeredgewidth=1.2)
        ax.annotate("? (query)", xy=(xq_pt, yq_pt),
                    xytext=(xq_pt + 0.35, yq_pt + 0.35),
                    fontsize=10, fontweight="bold", color="#c0392b",
                    zorder=11)

        # annotate reference t marks on curve
        for t_mark in t_refs:
            xm = x_func(t_mark)
            ym = y_func(t_mark)
            ax.plot(xm, ym, "o", color=color, markersize=6, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.8)
            # Use nice labels
            frac = t_mark / math.pi
            if abs(frac - round(frac)) < 0.01:
                nm = int(round(frac))
                lab = "0" if nm == 0 else (f"{nm}π" if nm != 1 else "π")
            else:
                # Fractional π
                num, den = frac.as_integer_ratio() if False else (None, None)
                # simpler: use common fractions
                common = {0.25: "π/4", 0.5: "π/2", 0.75: "3π/4", 1.25: "5π/4",
                          1.5: "3π/2", 1.75: "7π/4", 1/3: "π/3", 2/3: "2π/3"}
                lab = None
                for val, sym in common.items():
                    if abs(frac - val) < 0.02:
                        lab = sym
                        break
                if lab is None:
                    lab = f"t={t_mark:.2f}"
                else:
                    lab = f"t={lab}"
            if not lab.startswith("t="):
                lab = f"t={lab}"
            ax.annotate(lab, xy=(xm, ym),
                        xytext=(xm + 0.15, ym + 0.15),
                        fontsize=10, color="#1a1a1a")

        # determine plot bounds with some padding
        x_min = min(xs.min(), min(o[0] for o in options)) - 0.8
        x_max = max(xs.max(), max(o[0] for o in options)) + 0.8
        y_min = min(ys.min(), min(o[1] for o in options)) - 0.8
        y_max = max(ys.max(), max(o[1] for o in options)) + 0.8
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # gridlines at grid_spacing
        gs = cfg["grid_spacing"]
        ax.set_xticks(np.arange(math.floor(x_min), math.ceil(x_max) + gs, gs))
        ax.set_yticks(np.arange(math.floor(y_min), math.ceil(y_max) + gs, gs))
        ax.grid(True, which="major", color="#bdbdbd", linewidth=0.6, alpha=0.7)
        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.axvline(0, color="#555555", linewidth=0.9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="datalim")

        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        if cfg.get("hide_formula"):
            # Hide the explicit parametric formula; use generic title instead
            ax.set_title("Parametric curve", fontsize=11, fontweight="bold")
        else:
            ax.set_title(label, fontsize=11, fontweight="bold")

        # human-readable t_q string
        frac = t_q / math.pi
        common_q = {0.25: "π/4", 0.5: "π/2", 0.75: "3π/4", 1.25: "5π/4",
                    1.5: "3π/2", 1.75: "7π/4", 1/3: "π/3", 2/3: "2π/3",
                    0.0: "0", 1.0: "π"}
        t_q_str = None
        for val, sym in common_q.items():
            if abs(frac - val) < 0.02:
                t_q_str = sym
                break
        if t_q_str is None:
            t_q_str = f"{t_q:.2f}"

        question = (
            f"The graph shows a parametric curve. At t = {t_q_str}, what are "
            f"the approximate coordinates (x, y)? "
            f"{opt_str}. Answer with a single letter."
        )
        answer = correct_letter

        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return question, answer, img
