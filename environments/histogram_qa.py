"""
Histogram Visual QA Environment.

Generates histograms with frequency data and asks questions about
bin heights, percentages, and summary statistics.

Capabilities: V3 (chart extraction), V2 (label reading), R1 (arithmetic), R4 (statistical)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 3 bins with VERY distinct integer counts. Ask "which bin is tallest".
L1: 3 bins, ask "what is the count in bin X" (read off label).
L2: 4 bins, distinct counts, ask tallest.
L3: 4 bins, ask "total count across all bins".
L4: 5 bins, ask "tallest bin".
L5: 5 bins, ask "which is shortest".
L6: 6 bins, ask "total count".
L7: 7 bins with subtler differences, ask tallest.
L8: 8 bins, normal-shaped, ask "median bin".
L9: 10 bins, mixed distribution, ask mean estimate or median.

parameter = {"level": int in [0, 9]}
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

from .base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

# ------------------------------------------------------------------ #
# Label pools
# ------------------------------------------------------------------ #

_DATA_CONTEXTS = [
    {"title": "Distribution of Exam Scores", "x": "Score", "y": "Number of Students",
     "low": 30, "high": 100},
    {"title": "Employee Age Distribution", "x": "Age (years)", "y": "Frequency",
     "low": 20, "high": 65},
    {"title": "Response Time Distribution", "x": "Response Time (ms)", "y": "Count",
     "low": 50, "high": 500},
    {"title": "Daily Step Count Distribution", "x": "Steps", "y": "Number of Days",
     "low": 2000, "high": 15000},
    {"title": "Product Weight Distribution", "x": "Weight (g)", "y": "Frequency",
     "low": 100, "high": 500},
    {"title": "House Price Distribution", "x": "Price ($K)", "y": "Number of Houses",
     "low": 150, "high": 800},
    {"title": "Commute Time Distribution", "x": "Time (minutes)", "y": "Number of People",
     "low": 5, "high": 90},
    {"title": "Monthly Rainfall Distribution", "x": "Rainfall (mm)", "y": "Frequency",
     "low": 10, "high": 250},
    {"title": "Customer Spending Distribution", "x": "Amount ($)", "y": "Number of Customers",
     "low": 10, "high": 300},
    {"title": "Task Completion Time", "x": "Duration (sec)", "y": "Count",
     "low": 5, "high": 120},
]

class HistogramQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.5
    ENV_NAME = "histogram"

    QUESTION_TYPES = [
        "tallest_bin", "shortest_bin", "count_in_bin", "total_frequency",
        "percentage_in_bin", "median_bin", "mean_estimate", "mode_bin",
    ]

    @staticmethod
    def _normalize_bin_label(s: str) -> str:
        """Accept en-dash, em-dash, 'X to Y', etc. as bin range separators."""
        if s is None:
            return ""
        s = str(s).strip()
        s = s.replace("–", "-").replace("—", "-")  # en/em dash → hyphen
        import re as _re
        s = _re.sub(r"\s*to\s*", "-", s, flags=_re.IGNORECASE)
        s = _re.sub(r"\s*-\s*", "-", s)  # collapse spaces around hyphen
        return s.lower()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Normalize bin-range answers (e.g. "86-100" vs "86–100" vs "86 to 100")
        if "-" in str(ground_truth) or "–" in str(ground_truth):
            return self._normalize_bin_label(predicted) == self._normalize_bin_label(ground_truth)
        return super()._check_answer(predicted, ground_truth)

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(15):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            # L0: read a single clearly-labelled bar height. 3 bins, huge spread.
            return {"n_bins": 3, "spread": "huge", "qtype": "count_in_bin"}
        if level == 1:
            return {"n_bins": 3, "spread": "huge", "qtype": "tallest_bin"}
        if level == 2:
            return {"n_bins": 4, "spread": "huge", "qtype": "count_in_bin"}
        if level == 3:
            return {"n_bins": 4, "spread": "wide", "qtype": "tallest_bin"}
        if level == 4:
            return {"n_bins": 5, "spread": "wide", "qtype": "total_frequency"}
        if level == 5:
            return {"n_bins": 5, "spread": "wide", "qtype": "shortest_bin"}
        if level == 6:
            return {"n_bins": 6, "spread": "wide", "qtype": "total_frequency"}
        if level == 7:
            return {"n_bins": 7, "spread": "narrow", "qtype": "tallest_bin"}
        if level == 8:
            return {"n_bins": 8, "spread": "narrow", "qtype": "median_bin"}
        return {"n_bins": 10, "spread": "narrow", "qtype": "mean_estimate"}

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)

        ctx = rng.choice(_DATA_CONTEXTS)
        n_bins = cfg["n_bins"]

        low = ctx["low"]
        high = ctx["high"]
        bin_width = max(1, round((high - low) / n_bins))
        bin_edges = [low + i * bin_width for i in range(n_bins + 1)]

        freqs = self._gen_freqs(rng, n_bins, cfg["spread"])

        bin_labels = [f"{bin_edges[i]}-{bin_edges[i + 1]}" for i in range(n_bins)]

        question, answer = self._make_qa(
            rng, cfg["qtype"], bin_edges, bin_labels, freqs)
        if question is None:
            return None
        image = self._render_histogram(rng, bin_edges, freqs, ctx)
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], image
        return question, str(answer), image

    def _gen_freqs(self, rng, n_bins, spread):
        if spread == "huge":
            # Force counts: small / medium / huge
            base = [3, 12, 28, 45, 60, 75]
            rng.shuffle(base)
            return [base[i % len(base)] + rng.randint(-1, 1)
                    for i in range(n_bins)]
        if spread == "wide":
            # Distinct values spread by ~10
            counts = [5 + i * 8 + rng.randint(-2, 2) for i in range(n_bins)]
            rng.shuffle(counts)
            return [max(2, c) for c in counts]
        # narrow: normal-ish distribution
        center = n_bins // 2
        out = []
        for i in range(n_bins):
            v = max(3, int(40 * math.exp(-0.3 * (i - center) ** 2)))
            out.append(v + rng.randint(0, 3))
        return out

    def _make_qa(self, rng, qtype, bin_edges, bin_labels, freqs):
        n_bins = len(freqs)
        total = sum(freqs)

        if qtype == "tallest_bin":
            idx = freqs.index(max(freqs))
            stems = [
                "Which bin (range) has the highest frequency? "
                "Answer with the bin range like 'low-high'.",
                "Which bin has the tallest bar?",
            ]
            return rng.choice(stems), bin_labels[idx]

        if qtype == "shortest_bin":
            idx = freqs.index(min(freqs))
            stems = [
                "Which bin (range) has the lowest frequency? "
                "Answer with the bin range.",
                "Which bin has the shortest bar?",
            ]
            return rng.choice(stems), bin_labels[idx]

        if qtype == "count_in_bin":
            idx = rng.randint(0, n_bins - 1)
            stems = [
                f"What is the frequency (count) of the bin {bin_labels[idx]}? "
                f"Answer with a single integer.",
                f"How many observations are in the bin {bin_labels[idx]}?",
            ]
            return rng.choice(stems), freqs[idx]

        if qtype == "total_frequency":
            stems = [
                "What is the total frequency across all bins? "
                "Answer with a single integer.",
                "What is the sum of all bar heights in the histogram?",
            ]
            return rng.choice(stems), total

        if qtype == "percentage_in_bin":
            idx = rng.randint(0, n_bins - 1)
            pct = round(100 * freqs[idx] / total)
            return (f"What integer percentage of the total does the bin "
                    f"{bin_labels[idx]} represent?", pct)

        if qtype == "median_bin":
            cumsum = 0
            target = total / 2
            median_idx = 0
            for i, f in enumerate(freqs):
                cumsum += f
                if cumsum >= target:
                    median_idx = i
                    break
            return ("Which bin (range) contains the median observation?",
                    bin_labels[median_idx])

        if qtype == "mean_estimate":
            midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2
                         for i in range(n_bins)]
            weighted = sum(m * f for m, f in zip(midpoints, freqs))
            mean_est = round(weighted / total)
            return ("Estimate the mean by using bin midpoints. "
                    "Answer with a single integer.", mean_est)

        if qtype == "mode_bin":
            idx = freqs.index(max(freqs))
            return ("Which bin (range) is the mode (most frequent)?",
                    bin_labels[idx])

        return None, None

    def _render_histogram(self, rng, bin_edges, freqs, ctx):
        vs = self._random_style()
        palette = list(vs["palette"])
        rng.shuffle(palette)
        color = palette[0]
        n_bins = len(freqs)

        fig_w = max(6, n_bins * 0.7 + 2) * vs["figsize_scale"]
        fig_h = rng.uniform(4.5, 6.0) * vs["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        label_fs = vs["font_size_base"]
        title_fs = label_fs + 3
        font_family = vs["font_family"]

        widths = [bin_edges[i + 1] - bin_edges[i] for i in range(n_bins)]
        lefts = bin_edges[:-1]

        edge_color = "white" if rng.random() < 0.5 else "black"
        bars = ax.bar(lefts, freqs, width=widths, align="edge",
                      color=color, edgecolor=edge_color, linewidth=0.8,
                      alpha=rng.uniform(0.7, 1.0))

        # Always show labels at low levels for clarity
        show_labels = rng.random() < 0.7 or n_bins <= 5
        if show_labels:
            for bar, f in zip(bars, freqs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(f),
                        ha="center", va="bottom",
                        fontsize=label_fs - 1, fontfamily=font_family)

        ax.set_xlabel(ctx["x"], fontsize=label_fs, fontfamily=font_family)
        ax.set_ylabel(ctx["y"], fontsize=label_fs, fontfamily=font_family)
        ax.set_title(ctx["title"], fontsize=title_fs, pad=10, fontfamily=font_family)

        if n_bins <= 8:
            ax.set_xticks(bin_edges)
        else:
            ax.set_xticks(bin_edges[::2])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        self._apply_style(fig, ax, vs)
        ax.set_axisbelow(True)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vs["dpi"])

if __name__ == "__main__":
    env = HistogramQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
