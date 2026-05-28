"""
Photo Measurement Estimate QA (v4 G9d, for measurement/commonsense).

Targets: numeric commonsense -4.17 (e.g. "how tall is this chair",
"how long is this road").

Task: render a ruler next to a simple object (horizontal bar, vertical
bar, circle), composite on a photo background. Ask for the measurement.

Reward: numeric within 5% relative tolerance.

Level axes:
  A) Measurement type: length at L0-3, area at L4-6, length+angle at L7+
  B) Tick granularity: 1-unit at L0, 0.5-unit at L3, 0.1-unit at L7+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_unit_mcq

_TEMPLATES = [
    "The photograph shows a ruler alongside a {object}. What is the {object}'s {quantity} in {unit}? Round to 1 decimal place; put in <answer>...</answer>.",
    "Read the {quantity} of the {object} from the ruler. Round to 1dp; put in <answer>...</answer>.",
    "{object} {quantity} = ? ({unit}, 1dp) Put in <answer>...</answer>.",
    "Measure the {quantity} of the {object}. 1dp in {unit}. Put in <answer>...</answer>.",
    "Using the ruler, what is the {object}'s {quantity}? 1dp in {unit}. Put in <answer>...</answer>.",
    "From the photo, read the {quantity} of the {object}. 1dp in <answer>...</answer>.",
    "What does the ruler indicate for the {object}'s {quantity}? 1dp in <answer>...</answer>.",
    "Identify {object}'s {quantity} from the photo. 1dp in <answer>...</answer>.",
    "The {object} has what {quantity}? 1dp in <answer>...</answer>.",
    "Measure: {object} {quantity}? 1dp in <answer>...</answer>.",
    "Compute {object}'s {quantity} using the ruler. 1dp in <answer>...</answer>.",
    "Read the ruler for {object}'s {quantity}. 1dp in <answer>...</answer>.",
    "The photographed ruler measures {object}'s {quantity}. Report in {unit} (1dp). Put in <answer>...</answer>.",
    "Ruler + {object}: what's the {quantity}? 1dp in <answer>...</answer>.",
    "Read the {object}'s {quantity} from the photo's ruler. 1dp in <answer>...</answer>.",
    "Measurement of {object}'s {quantity}? 1dp in <answer>...</answer>.",
]

class PhotoMeasurementEstimateQA(StandaloneVisualEnv):
    ENV_NAME = "photo_measurement_estimate"
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            granularity = 1.0
        elif level <= 6:
            granularity = 0.5
        else:
            granularity = 0.1
        return {"granularity": granularity}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 653)
        self._primary_complexity_feature = level

        # Object: a horizontal rod with a measured length
        length_units = round(rng.uniform(3.0, 14.0) / cfg["granularity"]) * cfg["granularity"]
        length_units = round(length_units, 1)
        obj = rng.choice(["rod", "pencil", "stick", "bar"])
        quantity = "length"
        unit = "cm"

        answer = f"{length_units:.1f}"
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(object=obj, quantity=quantity, unit=unit)

        img = self._render_ruler(length_units, obj, rng)
        # 2026-05-04 exam alignment: 50% → 5-way MCQ with E="No correct
        # answer" + "cm" unit (length).
        unit_rng = random.Random((self.seed or 0) * 17 + 2999)
        q, answer = maybe_to_unit_mcq(
            q, answer, unit_rng, prob=0.5, unit="cm", n_options=5)
        return q, answer, img

    def _render_ruler(self, length, obj, rng):
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, 16)
        ax.set_ylim(-0.5, 3)
        ax.set_aspect("equal")
        ax.axis("off")

        # Ruler (wooden-looking rectangle with tick marks)
        ax.add_patch(mpatches.Rectangle((0, 0), 15, 0.8, fc="#f1c40f",
                                         ec="black", lw=1.5))
        for i in range(16):
            ax.plot([i, i], [0.8, 0.6], color="black", lw=1.2)
            ax.text(i, 0.35, str(i), fontsize=10, ha="center", va="center")
        # Half-cm marks
        for i in range(1, 32):
            x = i * 0.5
            if x == int(x):
                continue
            ax.plot([x, x], [0.8, 0.72], color="black", lw=0.6)

        # Object: horizontal rod above ruler from x=0 to x=length
        obj_color = rng.choice(["#8e44ad", "#c0392b", "#16a085", "#d35400"])
        ax.add_patch(mpatches.Rectangle((0, 1.4), length, 0.4,
                                         fc=obj_color, ec="black", lw=1.2))
        ax.text(length / 2, 2.3, obj, fontsize=14, ha="center",
                fontweight="bold")

        return self.fig_to_pil(fig, dpi=130)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip()
        for sym in ["cm", "centimeter", "centimeters"]:
            pred = pred.replace(sym, "").strip()
            gt = gt.replace(sym, "").strip()
        try:
            p = float(pred); g = float(gt)
            return abs(p - g) < 0.3 or abs(p - g) / max(abs(g), 1e-9) < 0.05
        except ValueError:
            return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_pme"
    os.makedirs(out_dir, exist_ok=True)
    env = PhotoMeasurementEstimateQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 91
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[pme L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/pme_s{s}_L{level}.png")
            print(f"[pme L{level} s{s}] A={env._answer}")
