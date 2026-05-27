"""
Photo Chart Reading QA (v4 G9a, for scientific reasoning / textbook QA).

Targets:

Task: render a bar/line chart (using matplotlib), composite it on a
photographed-looking background (desk/paper/wall). Ask a standard
chart-reading question.

Reward: numeric within 2% relative tolerance.

Level axes:
  A) Chart type: bar at L0-3, line at L4-6, pie at L7+
  B) Number of bars/points: 4 at L0, 8 at L9
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The photograph shows a chart. What is the value of the {bar_label} bar/point? Put the integer in <answer>...</answer>.",
    "From the chart in the photo, read off the value for {bar_label}. Integer in <answer>...</answer>.",
    "Identify the value of {bar_label} from the chart. Integer in <answer>...</answer>.",
    "Value of {bar_label}? Integer in <answer>...</answer>.",
    "What does {bar_label} read on this chart? Integer in <answer>...</answer>.",
    "Photo chart: value of {bar_label}? Integer in <answer>...</answer>.",
    "Read {bar_label} from the chart. Integer in <answer>...</answer>.",
    "The {bar_label} bar shows what value? Integer in <answer>...</answer>.",
    "Chart in photo: what is {bar_label}'s value? Integer in <answer>...</answer>.",
    "From the photographed chart, extract the value of {bar_label}. Integer in <answer>...</answer>.",
    "Read the value of {bar_label} shown in the chart. Put integer in <answer>...</answer>.",
    "Identify {bar_label}'s value from the chart image. Integer in <answer>...</answer>.",
    "{bar_label} = ? (from chart). Integer in <answer>...</answer>.",
    "What value is {bar_label}? Integer in <answer>...</answer>.",
    "Report {bar_label}'s value from the chart. Integer in <answer>...</answer>.",
    "From the chart, what's {bar_label}? Integer in <answer>...</answer>.",
]

class PhotoChartReadingQA(StandaloneVisualEnv):
    ENV_NAME = "photo_chart_reading"
    # Rev 2 (2026-04-24): user feedback — dropped Qwen-background composite;
    # general VQA is either real photos (can't generate) or clean vector
    # charts (textbook style). Apply B1 textbook-scan filter via class flag.
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_bars = 4 + level // 2
        if level <= 3:
            chart_type = "bar"
        elif level <= 6:
            chart_type = "line"
        else:
            chart_type = "bar"
        return {"n_bars": n_bars, "chart_type": chart_type}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 643)
        self._primary_complexity_feature = level

        n = cfg["n_bars"]
        labels = [chr(ord("A") + i) for i in range(n)]
        values = [rng.randint(10, 100) for _ in range(n)]
        target_idx = rng.randint(0, n - 1)
        target_label = labels[target_idx]
        answer = str(values[target_idx])

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(bar_label=target_label)

        # Render chart (clean, textbook-style). The optional B1 textbook
        # postprocess is applied by the base class `fig_to_pil` when
        # TEXTBOOK_POSTPROCESS is True (~30% of rollouts).
        img = self._render_chart(labels, values, cfg["chart_type"], rng)
        return q, answer, img

    def _render_chart(self, labels, values, chart_type, rng):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        if chart_type == "bar":
            colors = rng.choice([
                ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
                ["#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
            ])
            ax.bar(labels, values,
                    color=[colors[i % len(colors)] for i in range(len(labels))],
                    edgecolor="black")
            ax.set_ylabel("Value", fontsize=12)
        else:  # line
            color = rng.choice(["#3498db", "#2ecc71", "#e74c3c"])
            ax.plot(labels, values, marker="o", color=color, lw=2, markersize=8)
            ax.set_ylabel("Value", fontsize=12)
        ax.set_title(rng.choice(["Quarterly Report", "Monthly Data",
                                   "Department Results", "Performance Summary"]),
                      fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        return self.fig_to_pil(fig, dpi=120)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_pcr"
    os.makedirs(out_dir, exist_ok=True)
    env = PhotoChartReadingQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 61
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[pcr L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/pcr_s{s}_L{level}.png")
            print(f"[pcr L{level} s{s}] A={env._answer}")
