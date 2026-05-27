"""
Curve property identification: render a multi-line chart with 3-5 labeled
curves, then ask which curve has a stated visual property:
  - smallest variance / smoothest
  - steepest slope at a specific x
  - most-linear behavior
  - closest to a reference y at a stated x

The question targets the Number-in-Chart subtask "Identify the parameter value
(curve label) of the line that meets a stated condition" (IDX 531
"Which J/U ratio curve has the smallest variance...", IDX 526 "Which I value
exhibits the most linear increase...", IDX 409 "...lowest rate of increase",
IDX 460 "...completes its line path first... by the N value").

Output: the curve label as a numeric string (e.g. "0.95", "1.5"). Labels are
generated as well-separated values so numeric tolerance (5% rel / 0.5 abs)
does not let close distractors slip through.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Wording mirrors verbatim phrasings from reference samples.
_TEMPLATES_VARIANCE = [
    "Which {param} curve has the smallest variance in the y-values across all points shown? Reply with the curve's label (a number) in <answer>...</answer>.",
    "Among the labelled {param} curves, which one shows the smallest variance? Label answer in <answer>...</answer>.",
    "Identify the curve with the smallest variance over the displayed range. Reply with its {param} value in <answer>...</answer>.",
    "Which curve, identified by its {param} value, has the smallest variance? In <answer>...</answer>.",
    "From the {param} curves shown, which has the smallest variance in y-values? Place its {param} value in <answer>...</answer>.",
]

_TEMPLATES_LINEAR = [
    "Which {param} value exhibits the most linear behavior in the chart? Reply with the {param} value in <answer>...</answer>.",
    "Which curve, labelled by {param}, increases the most linearly? Label answer in <answer>...</answer>.",
    "Among the {param} curves, identify the one whose growth is most linear. Reply with its {param} value in <answer>...</answer>.",
    "Identify the {param} curve that is closest to a straight line. Label in <answer>...</answer>.",
    "Which {param} value yields the most linear curve in the plot? Label answer in <answer>...</answer>.",
]

_TEMPLATES_STEEPEST = [
    "Which {param} curve is the steepest at x = {x0}? Reply with the {param} value in <answer>...</answer>.",
    "At x = {x0}, which {param} curve has the largest slope (steepest)? Label answer in <answer>...</answer>.",
    "Among the {param} curves, which one rises (or falls) most steeply at x = {x0}? Label in <answer>...</answer>.",
    "Identify the steepest {param} curve at x = {x0}. Reply with its {param} value in <answer>...</answer>.",
    "Which {param} value's curve has the largest gradient at x = {x0}? Label in <answer>...</answer>.",
]

_TEMPLATES_CLOSEST = [
    "Which curve is closest to y = {ref} at x = {x0}? Reply with the curve's {param} value in <answer>...</answer>.",
    "At x = {x0}, which {param} curve has y closest to {ref}? Label in <answer>...</answer>.",
    "Identify the {param} curve whose value at x = {x0} is closest to {ref}. Label answer in <answer>...</answer>.",
    "Among the curves shown, which one (labelled by {param}) sits closest to y = {ref} at x = {x0}? Label in <answer>...</answer>.",
    "Which {param} value's curve passes nearest to y = {ref} at x = {x0}? Label in <answer>...</answer>.",
]

# Combined for diversity check (>= 16 wordings overall).
ALL_TEMPLATES = (_TEMPLATES_VARIANCE + _TEMPLATES_LINEAR
                 + _TEMPLATES_STEEPEST + _TEMPLATES_CLOSEST)
assert len(ALL_TEMPLATES) >= 16


def _format_label(v: float) -> str:
    """Format curve labels for display + answer."""
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _make_curve(rng, kind, x, amp, slope, noise, drift_freq, drift_phase, drift_amp):
    """Generate a curve.
    kind ∈ {"linear", "noisy", "wavy", "smooth_curve"}
    """
    if kind == "linear":
        y = slope * x + amp
    elif kind == "noisy":
        y = slope * x + amp
        y = y + drift_amp * np.sin(drift_freq * x + drift_phase)
        y = y + noise * np.random.RandomState(rng.randint(0, 1 << 24)).randn(len(x))
    elif kind == "wavy":
        y = slope * x + amp + drift_amp * np.sin(drift_freq * x + drift_phase)
    else:  # smooth_curve (slight curvature)
        y = slope * x + 0.05 * x ** 2 + amp
    return y


class CurvePropertyIdentifyQA(StandaloneVisualEnv):
    ENV_NAME = "curve_property_identify"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_curves = 2
            modes = ["variance"]
        elif level <= 2:
            n_curves = 3
            modes = ["variance", "linear"]
        elif level <= 4:
            n_curves = 3
            modes = ["variance", "linear", "steepest", "closest"]
        elif level <= 6:
            n_curves = 4
            modes = ["variance", "linear", "steepest", "closest"]
        elif level <= 8:
            n_curves = 5
            modes = ["variance", "linear", "steepest", "closest"]
        else:
            n_curves = 5
            modes = ["variance", "linear", "steepest", "closest"]
        return {"level": level, "n_curves": n_curves, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        import hashlib
        sd = int(hashlib.md5(f"cpi|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        n = cfg["n_curves"]
        mode = rng.choice(cfg["modes"])

        # Pick parameter symbol
        param_sym = rng.choice(["I", "J/U", "K_p", "N", "$\\beta$", "$\\gamma$"])

        # Generate well-separated label values
        # Use distinct numerics so labels don't collide under 5% / 0.5 tolerance
        candidate_labels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        rng.shuffle(candidate_labels)
        labels = sorted(candidate_labels[:n])
        # Verify minimum spacing > 0.5 absolute and > 5% relative
        labels = sorted(labels)
        ok = True
        for i in range(len(labels) - 1):
            d = labels[i + 1] - labels[i]
            r = d / max(abs(labels[i]), abs(labels[i + 1]))
            if d < 0.5 + 1e-6 or r < 0.05 + 1e-6:
                ok = False
                break
        if not ok:
            # fall back to integers far apart
            labels = sorted(rng.sample([1, 3, 7, 15, 31], n))
            labels = [float(v) for v in labels]

        x = np.linspace(0.0, 10.0, 200)

        # Build the curves & answer per mode
        if mode == "variance":
            # Make one curve flat (low variance), others noisy/wavy.
            target_idx = rng.randrange(n)
            curves = []
            for i in range(n):
                amp = rng.uniform(-0.5, 0.5)
                slope = rng.uniform(-0.05, 0.05)
                if i == target_idx:
                    # very low variance
                    y = amp * np.ones_like(x) + slope * x
                else:
                    drift_freq = rng.uniform(0.5, 1.5)
                    drift_phase = rng.uniform(0, 2 * math.pi)
                    drift_amp = rng.uniform(1.5, 3.0)
                    y = slope * x + amp + drift_amp * np.sin(drift_freq * x + drift_phase)
                curves.append(y)
            answer = _format_label(labels[target_idx])
            sidx = (self.seed or 0) % len(_TEMPLATES_VARIANCE)
            question = _TEMPLATES_VARIANCE[sidx].format(param=param_sym)
        elif mode == "linear":
            target_idx = rng.randrange(n)
            curves = []
            for i in range(n):
                amp = rng.uniform(-1.0, 1.0)
                slope = rng.uniform(0.4, 1.2)
                if i == target_idx:
                    y = slope * x + amp   # purely linear
                else:
                    # add curvature via quadratic / sinusoidal bend
                    bend = rng.choice(["quad", "sin"])
                    if bend == "quad":
                        coef = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
                        y = slope * x + coef * x ** 2 + amp
                    else:
                        drift_freq = rng.uniform(0.4, 0.8)
                        drift_phase = rng.uniform(0, 2 * math.pi)
                        drift_amp = rng.uniform(1.5, 3.0)
                        y = slope * x + amp + drift_amp * np.sin(drift_freq * x + drift_phase)
                curves.append(y)
            answer = _format_label(labels[target_idx])
            sidx = (self.seed or 0) % len(_TEMPLATES_LINEAR)
            question = _TEMPLATES_LINEAR[sidx].format(param=param_sym)
        elif mode == "steepest":
            x0 = round(rng.uniform(2.0, 8.0), 1)
            target_idx = rng.randrange(n)
            curves = []
            slopes = []
            for i in range(n):
                if i == target_idx:
                    s = rng.uniform(2.5, 4.0)   # well above
                else:
                    s = rng.uniform(0.3, 1.0)
                slopes.append(s)
                amp = rng.uniform(-2.0, 2.0)
                # Use a smooth curve so slope varies a bit but x0 is dominant
                y = s * x + amp + 0.3 * np.sin(0.5 * x + rng.uniform(0, 2 * math.pi))
                curves.append(y)
            answer = _format_label(labels[target_idx])
            sidx = (self.seed or 0) % len(_TEMPLATES_STEEPEST)
            question = _TEMPLATES_STEEPEST[sidx].format(param=param_sym, x0=x0)
        else:   # closest
            x0 = round(rng.uniform(2.0, 8.0), 1)
            ref = round(rng.uniform(2.0, 8.0), 1)
            target_idx = rng.randrange(n)
            curves = []
            for i in range(n):
                slope = rng.uniform(-0.4, 0.4)
                amp = rng.uniform(-2.0, 2.0)
                if i == target_idx:
                    # arrange so y(x0) ≈ ref
                    amp = ref - slope * x0
                else:
                    # arrange so y(x0) is at least 1.5 away from ref
                    while True:
                        amp = rng.uniform(-3.0, 3.0)
                        y_at_x0 = slope * x0 + amp
                        if abs(y_at_x0 - ref) > 1.5:
                            break
                drift_amp = rng.uniform(0.2, 0.6)
                drift_freq = rng.uniform(0.4, 0.9)
                drift_phase = rng.uniform(0, 2 * math.pi)
                y = slope * x + amp + drift_amp * np.sin(drift_freq * x + drift_phase)
                curves.append(y)
            answer = _format_label(labels[target_idx])
            sidx = (self.seed or 0) % len(_TEMPLATES_CLOSEST)
            question = _TEMPLATES_CLOSEST[sidx].format(param=param_sym, x0=x0, ref=ref)

        img = self._render(x, curves, labels, param_sym, rng)
        return question, answer, img

    def _render(self, x, curves, labels, param_sym, rng):
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        palette = ["#1f4e79", "#a6192e", "#2e7d32", "#6a1b9a", "#ef6c00",
                    "#00838f", "#5d4037"]
        styles = ["-", "--", "-.", ":", "-"]
        for i, (y, lbl) in enumerate(zip(curves, labels)):
            ax.plot(x, y, color=palette[i % len(palette)],
                    linestyle=styles[i % len(styles)], linewidth=2.0,
                    label=f"{param_sym} = {_format_label(lbl)}")
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
