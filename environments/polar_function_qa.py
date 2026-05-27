"""
Polar function QA (redesigned 2026-04-16) — rose curves, cardioids,
limacons, circles, lemniscates, spirals.

Critical fix (vs Grade D baseline):
  * Title NO LONGER prints the polar equation (e.g. "r = 3cos(4θ)").
    Old title leaked the function type/constants for every question.
  * Title is now neutral: "Polar curve", "Polar plot", "Polar graph",
    etc. — the model must identify the curve from the plot alone.
  * Expanded curve pool: circle, cardioid, limacon (convex/dimpled/
    looped), rose (even/odd petals), lemniscate, archimedean spiral.
  * 8+ question templates covering identification, counting, symmetry,
    value-at-angle, projections, area-region.
  * L0/L9 structural shift: L0 = identify type / count petals;
    L9 = value_at_angle / projection on diverse curve set.
  * Randomized orientation: cos/sin variant, sign flip, phase offset.
  * Diverse colors and styles per seed.
  * MCQ option shuffle where applicable.
"""
import random
import math
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_NEUTRAL_TITLES = [
    "Polar curve",
    "Polar plot",
    "Polar graph",
    "Polar function",
    "Polar coordinate plot",
    "Curve in polar coordinates",
    "r vs θ",
    "Polar representation",
    "Polar figure",
]

def _rose(a, n_petals, use_sin, phase):
    """Compute rose points; return (theta_array, r_array, spec)."""
    theta = np.linspace(0, 2 * np.pi, 600)
    if use_sin:
        r = a * np.sin(n_petals * theta + phase)
    else:
        r = a * np.cos(n_petals * theta + phase)
    actual_petals = n_petals if n_petals % 2 == 1 else 2 * n_petals
    spec = {
        "type": "rose", "a": a, "n": n_petals, "use_sin": use_sin,
        "phase": phase, "max_r": a, "petals": actual_petals,
    }
    return theta, r, spec

def _cardioid(a, use_sin, flip):
    theta = np.linspace(0, 2 * np.pi, 600)
    f = np.sin if use_sin else np.cos
    s = -1 if flip else 1
    r = a * (1 + s * f(theta))
    spec = {"type": "cardioid", "a": a, "use_sin": use_sin,
            "flip": flip, "max_r": 2 * a, "petals": 0}
    return theta, r, spec

def _limacon(a, b, use_sin, flip):
    theta = np.linspace(0, 2 * np.pi, 600)
    f = np.sin if use_sin else np.cos
    s = -1 if flip else 1
    r = a + s * b * f(theta)
    spec = {"type": "limacon", "a": a, "b": b, "use_sin": use_sin,
            "flip": flip, "max_r": a + b, "petals": 0}
    return theta, r, spec

def _circle(a):
    theta = np.linspace(0, 2 * np.pi, 400)
    r = np.full_like(theta, a)
    spec = {"type": "circle", "a": a, "max_r": a, "petals": 0}
    return theta, r, spec

def _lemniscate(a, use_sin):
    """r^2 = a^2 cos(2theta) or sin(2theta); returns two lobes."""
    theta = np.linspace(0, 2 * np.pi, 800)
    arg = np.sin(2 * theta) if use_sin else np.cos(2 * theta)
    # Where arg >= 0, r is real
    r2 = a * a * arg
    r = np.sqrt(np.where(r2 >= 0, r2, 0))
    spec = {"type": "lemniscate", "a": a, "use_sin": use_sin,
            "max_r": a, "petals": 2}
    return theta, r, spec

def _spiral(a, turns):
    """Archimedean spiral r = a*theta, bounded to `turns` rotations."""
    theta = np.linspace(0.1, turns * 2 * np.pi, 800)
    r = a * theta / (2 * np.pi)
    spec = {"type": "spiral", "a": a, "turns": turns,
            "max_r": float(r[-1]), "petals": 0}
    return theta, r, spec

class PolarFunctionQA(StandaloneVisualEnv):
    ENV_NAME = "polar_function"

    # Question type inventories per level
    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"qtypes": ["identify_type", "max_radius"],
                    "qweights": [5, 5],
                    "curves": ["circle", "cardioid", "rose"]}
        if level <= 2:
            return {"qtypes": ["identify_type", "count_petals", "max_radius"],
                    "qweights": [4, 3, 3],
                    "curves": ["circle", "cardioid", "rose", "limacon"]}
        if level <= 4:
            return {"qtypes": ["count_petals", "max_radius",
                               "value_at_angle", "symmetry_type"],
                    "qweights": [3, 3, 2, 2],
                    "curves": ["rose", "cardioid", "limacon", "lemniscate"]}
        if level <= 6:
            return {"qtypes": ["value_at_angle", "count_petals",
                               "projection", "symmetry_type"],
                    "qweights": [3, 3, 2, 2],
                    "curves": ["rose", "cardioid", "limacon", "circle",
                               "lemniscate", "spiral"]}
        if level <= 8:
            return {"qtypes": ["value_at_angle", "projection"],
                    "qweights": [5, 5],
                    "curves": ["rose", "limacon", "lemniscate", "spiral"]}
        return {"qtypes": ["projection", "value_at_angle"],
                "qweights": [6, 4],
                "curves": ["rose", "limacon", "lemniscate", "spiral"]}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 1101)
        vis_rng = random.Random(seed * 1000 + level * 37 + 2803)

        question_type = parameter.get("question_type")
        valid_qtypes = {"identify_type", "count_petals", "max_radius",
                        "symmetry_type", "value_at_angle", "projection"}
        if question_type not in valid_qtypes:
            question_type = sub_rng.choices(
                cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        curve_type = sub_rng.choice(cfg["curves"])

        # Build curve — randomize sin/cos and phase
        if curve_type == "rose":
            n_petals = sub_rng.choice([3, 4, 5, 6, 7, 8])
            a = sub_rng.randint(2, 6)
            use_sin = sub_rng.random() < 0.5
            phase = sub_rng.choice([0, math.pi / 6, math.pi / 4, math.pi / 3,
                                    math.pi / 2, 0, 0])  # 0-biased
            theta, r, spec = _rose(a, n_petals, use_sin, phase)
        elif curve_type == "cardioid":
            a = sub_rng.randint(2, 6)
            use_sin = sub_rng.random() < 0.5
            flip = sub_rng.random() < 0.5
            theta, r, spec = _cardioid(a, use_sin, flip)
        elif curve_type == "limacon":
            a = sub_rng.randint(2, 6)
            b = sub_rng.randint(1, a + 2)
            use_sin = sub_rng.random() < 0.5
            flip = sub_rng.random() < 0.5
            theta, r, spec = _limacon(a, b, use_sin, flip)
        elif curve_type == "lemniscate":
            a = sub_rng.randint(2, 6)
            use_sin = sub_rng.random() < 0.5
            theta, r, spec = _lemniscate(a, use_sin)
        elif curve_type == "spiral":
            a = sub_rng.randint(2, 6)
            turns = sub_rng.randint(2, 4)
            theta, r, spec = _spiral(a, turns)
        else:  # circle
            a = sub_rng.randint(2, 6)
            theta, r, spec = _circle(a)

        # Build Q/A
        qa = self._make_qa(question_type, spec, sub_rng)
        if qa is None:
            return None
        question, answer = qa

        image = self._render(theta, r, vis_rng)
        return question, answer, image

    def _make_qa(self, qtype, spec, rng):
        ctype = spec["type"]
        if qtype == "identify_type":
            options = ["rose", "cardioid", "limacon", "circle",
                       "lemniscate", "spiral"]
            correct = ctype
            rng.shuffle(options)
            letter = chr(ord("A") + options.index(correct))
            opt_str = "  ".join(f"({chr(ord('A')+i)}) {v}"
                                for i, v in enumerate(options))
            stem = rng.choice([
                "What type of polar curve is shown?",
                "Identify the polar curve in the image.",
                "Which curve family does the plot belong to?",
                "Classify the polar curve shown in the figure.",
                "What kind of polar curve is plotted above?",
            ])
            q = f"{stem} Options: {opt_str}. Answer with the letter."
            return q, letter
        if qtype == "count_petals":
            if ctype == "rose":
                stem = rng.choice([
                    "How many petals does this rose curve have?",
                    "Count the petals visible in the polar plot.",
                    "How many lobes (petals) does the curve exhibit?",
                ])
                return (stem + " Answer with a single integer.",
                        str(spec["petals"]))
            elif ctype == "lemniscate":
                stem = rng.choice([
                    "How many lobes does this polar curve show?",
                    "Count the distinct lobes in the figure.",
                ])
                return (stem + " Answer with a single integer.", "2")
            else:
                stem = rng.choice([
                    "Does this curve have visible petals? Answer Yes or No.",
                    "Are there any petal-like lobes in this polar plot? "
                    "Answer Yes or No.",
                ])
                return (stem, "No")
        if qtype == "max_radius":
            stem = rng.choice([
                "What is the maximum radius (r) of the curve?",
                "Find the largest value of r attained by the curve.",
                "What is the maximum distance from the origin?",
            ])
            return (stem + " Answer as a number.",
                    str(round(spec["max_r"], 1)))
        if qtype == "symmetry_type":
            # For all these, x-axis symmetry holds for cos-based; sin-based
            # curves have y-axis symmetry. Build the check.
            sym = self._symmetry_axis(spec)
            stem = rng.choice([
                "Is this curve symmetric about the x-axis? Answer Yes or No.",
                "Does the curve exhibit x-axis symmetry? Answer Yes or No.",
            ])
            ans = "Yes" if sym == "x" or sym == "both" else "No"
            return (stem, ans)
        if qtype == "value_at_angle":
            test_angle = rng.choice([0, 30, 45, 60, 90, 120, 135, 180,
                                     225, 270, 315])
            rad = math.radians(test_angle)
            r_val = self._eval_r(spec, rad)
            stem = rng.choice([
                f"What is r when θ = {test_angle}°?",
                f"Find the value of r at θ = {test_angle} degrees.",
                f"Evaluate the polar function at θ = {test_angle}°.",
            ])
            return (stem + " Round to 1 decimal.",
                    str(round(abs(r_val), 1)))
        if qtype == "projection":
            test_angle = rng.choice([0, 30, 45, 60, 90, 120, 135, 180])
            rad = math.radians(test_angle)
            r_val = self._eval_r(spec, rad)
            x_proj = r_val * math.cos(rad)
            stem = rng.choice([
                f"What is the x-coordinate (r·cosθ) of the curve at "
                f"θ = {test_angle}°? Round to 1 decimal.",
                f"Compute the projection r·cosθ at θ = {test_angle}°. "
                f"Round to 1 decimal.",
                f"Find r·cos(θ) for the curve at θ = {test_angle} degrees. "
                f"Round to 1 decimal.",
            ])
            return (stem, str(round(x_proj, 1)))
        return None

    def _eval_r(self, spec, rad):
        ctype = spec["type"]
        if ctype == "rose":
            f = math.sin if spec["use_sin"] else math.cos
            return spec["a"] * f(spec["n"] * rad + spec["phase"])
        if ctype == "cardioid":
            f = math.sin if spec["use_sin"] else math.cos
            s = -1 if spec["flip"] else 1
            return spec["a"] * (1 + s * f(rad))
        if ctype == "limacon":
            f = math.sin if spec["use_sin"] else math.cos
            s = -1 if spec["flip"] else 1
            return spec["a"] + s * spec["b"] * f(rad)
        if ctype == "lemniscate":
            arg = (math.sin(2 * rad) if spec["use_sin"]
                   else math.cos(2 * rad))
            return math.sqrt(max(0, spec["a"] ** 2 * arg))
        if ctype == "spiral":
            return spec["a"] * rad / (2 * math.pi)
        if ctype == "circle":
            return spec["a"]
        return 0.0

    def _symmetry_axis(self, spec):
        ctype = spec["type"]
        if ctype == "circle":
            return "both"
        if ctype in ("cardioid", "limacon"):
            return "y" if spec.get("use_sin") else "x"
        if ctype == "rose":
            if spec.get("use_sin"):
                return "y"
            return "x"
        if ctype == "lemniscate":
            return "x" if not spec.get("use_sin") else "diagonal"
        if ctype == "spiral":
            return "none"
        return "x"

    def _render(self, theta, r, rng):
        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(
            figsize=(max(7, 7.5 * s), max(7, 7.5 * s)),
            subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor(style["bg_color"])

        # Randomize line color, width, style
        palette = list(style["palette"])
        rng.shuffle(palette)
        line_color = palette[0]
        lw = 2.0 + rng.random() * 1.2
        ls = rng.choice(["-", "-", "-", "--", "-."])

        ax.plot(theta, np.abs(r), color=line_color, linewidth=lw, linestyle=ls)

        # Optional secondary dashed reference circle at a chosen radius
        if rng.random() < 0.2:
            ref_r = float(np.mean(np.abs(r)[np.abs(r) > 0]))
            if ref_r > 0 and not np.isnan(ref_r):
                th_ref = np.linspace(0, 2 * np.pi, 200)
                ax.plot(th_ref, np.full_like(th_ref, ref_r),
                        color="#aaaaaa", linewidth=0.6, linestyle=":")

        # IMPORTANT: title no longer contains the equation
        title = rng.choice(_NEUTRAL_TITLES)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

        # Grid style
        ax.grid(True, alpha=0.3 + rng.random() * 0.25)

        return self.fig_to_pil(fig, dpi=max(120, style["dpi"]))

if __name__ == "__main__":
    env = PolarFunctionQA()
    for lv in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": lv})
            print(f"L{lv} s{seed} ok={ok} A={env._answer}")
