"""
Curve functional form classification: render a single curve on a linear or
log scale (or mixed log-log), then ask whether y depends on x linearly,
logarithmically, or exponentially.

Targets the Text-in-General subtask "trend / direction word" answers (IDX 524
"Based on plot (a), should we say that the spin dimer correlations depend on r
linearly, logarithmically, or exponentially?" — GT: "exponentially").

Output: a single word in {linearly, logarithmically, exponentially}. The
verifier accepts common short forms (linear, log, logarithmic, exp,
exponential) via _check_answer.
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


# Phrasing mirrors IDX 524 verbatim plus paraphrases.
_TEMPLATES = [
    "Based on the plot, does y depend on x linearly, logarithmically, or exponentially? Reply with one word in <answer>...</answer>.",
    "Looking at the chart, does y vary with x linearly, logarithmically, or exponentially? Single word in <answer>...</answer>.",
    "Should we say that y depends on x linearly, logarithmically, or exponentially? One word in <answer>...</answer>.",
    "From the curve shown, classify the dependence of y on x: linearly, logarithmically, or exponentially? Word in <answer>...</answer>.",
    "Based on the chart, what is the functional form of y as a function of x? Reply with linearly, logarithmically, or exponentially in <answer>...</answer>.",
    "The plot shows y vs. x. Is the dependence linear, logarithmic, or exponential? One word in <answer>...</answer>.",
    "Examine the curve. Does it grow linearly, logarithmically, or exponentially? Single word in <answer>...</answer>.",
    "Identify the functional form of the curve: linearly, logarithmically, or exponentially? In <answer>...</answer>.",
    "Looking at the displayed curve, is y a linear, logarithmic, or exponential function of x? Reply with linearly / logarithmically / exponentially in <answer>...</answer>.",
    "Classify the curve as linearly, logarithmically, or exponentially varying. Single word in <answer>...</answer>.",
    "Does y rise linearly, logarithmically, or exponentially with x? Word in <answer>...</answer>.",
    "What is the shape of y(x) — linearly, logarithmically, or exponentially? In <answer>...</answer>.",
    "From the visual trend, is the relation between x and y linearly, logarithmically, or exponentially shaped? Word in <answer>...</answer>.",
    "Inspecting the curve shape, would you say y depends on x linearly, logarithmically, or exponentially? In <answer>...</answer>.",
    "Looking at the chart, classify the trend: linearly, logarithmically, or exponentially? In <answer>...</answer>.",
    "Identify whether the curve grows linearly, logarithmically, or exponentially. One word in <answer>...</answer>.",
]


def _make_data(rng, kind, level):
    """Generate (x, y) data of the requested functional form, plus appropriate
    axis-scale advice (linear/log) so the visual cue is honest.

    L0: huge separation between forms (all linear scale; exponential is very
    fast-growing, log very slow). Higher L: smaller separations and use
    log-scale axes that flatten the difference (so the model has to use shape
    cues, not naive read).
    """
    if level <= 1:
        x = np.linspace(1.0, 10.0, 100)
        if kind == "linearly":
            a, b = rng.uniform(1.0, 2.0), rng.uniform(0.0, 2.0)
            y = a * x + b
        elif kind == "logarithmically":
            a, b = rng.uniform(2.0, 4.0), rng.uniform(0.0, 1.0)
            y = a * np.log(x) + b
        else:   # exponentially
            a, b = rng.uniform(0.4, 0.7), rng.uniform(0.5, 1.0)
            y = b * np.exp(a * x)   # fast growth — hugely curved
        return x, y, "linear", "linear"
    elif level <= 4:
        x = np.linspace(1.0, 12.0, 100)
        if kind == "linearly":
            a, b = rng.uniform(0.6, 1.5), rng.uniform(-1.0, 2.0)
            y = a * x + b + rng.uniform(-0.05, 0.05) * np.random.RandomState(rng.randint(0, 1 << 24)).randn(len(x))
        elif kind == "logarithmically":
            a, b = rng.uniform(2.0, 4.0), rng.uniform(0.0, 2.0)
            y = a * np.log(x) + b
        else:
            a, b = rng.uniform(0.25, 0.45), rng.uniform(0.5, 1.5)
            y = b * np.exp(a * x)
        return x, y, "linear", "linear"
    else:
        # Use log-scale axes — model has to combine axis-scale + shape cues.
        x = np.linspace(1.0, 100.0, 200)
        if kind == "linearly":
            a, b = rng.uniform(0.5, 1.2), rng.uniform(0.0, 2.0)
            y = a * x + b
            xs, ys = "linear", "linear"
        elif kind == "logarithmically":
            a, b = rng.uniform(2.0, 4.0), rng.uniform(0.0, 1.0)
            y = a * np.log(x) + b
            # Log-x: a logarithmic curve appears as a straight line.
            # That's a recognisable cue for the trained reader.
            xs, ys = "log", "linear"
        else:   # exponential
            a, b = rng.uniform(0.04, 0.08), rng.uniform(0.5, 1.0)
            y = b * np.exp(a * x)
            # Log-y: exponential becomes a straight line — recognisable cue.
            xs, ys = "linear", "log"
        return x, y, xs, ys


# Synonym sets per ground-truth word
_SYNONYMS = {
    "linearly":         ["linear", "linearly", "linearity"],
    "logarithmically":  ["log", "logarithmic", "logarithmically", "log-linear", "logarithmical"],
    "exponentially":    ["exp", "exponential", "exponentially", "exponentiation"],
}


class CurveFunctionalFormQA(StandaloneVisualEnv):
    ENV_NAME = "curve_functional_form"

    def _level_config(self, level: int) -> Dict:
        return {"level": max(0, min(level, 9))}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        import hashlib
        sd = int(hashlib.md5(f"cff|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        # L0: only "linearly" — y=x, dead simple
        if level == 0:
            kind = "linearly"
        else:
            kind = rng.choice(["linearly", "logarithmically", "exponentially"])

        x, y, xs, ys = _make_data(rng, kind, level)
        answer = kind
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(x, y, xs, ys, rng)
        return question, answer, img

    def _render(self, x, y, xs, ys, rng):
        fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.plot(x, y, color="#1f4e79", linewidth=2.2)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        if xs == "log":
            ax.set_xscale("log")
        if ys == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", which="both")
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

    # Override to accept synonyms ("linear" ↔ "linearly")
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        p = predicted.strip().lower().rstrip(".")
        g = ground_truth.strip().lower().rstrip(".")
        if g not in _SYNONYMS:
            return super()._check_answer(predicted, ground_truth)
        synonyms = _SYNONYMS[g]
        # Must match a synonym as a whole token. Reject if any *other*
        # ground-truth's synonym is also present (mixed answer).
        import re
        matched_self = any(re.search(rf"\b{re.escape(s)}\b", p) for s in synonyms)
        if not matched_self:
            return False
        for other_g, other_syn in _SYNONYMS.items():
            if other_g == g:
                continue
            for s in other_syn:
                # Skip very short tokens that would over-match (e.g. "log" vs
                # "logarithm" in same family — handled above by `other_g != g`).
                if re.search(rf"\b{re.escape(s)}\b", p):
                    return False
        return True
