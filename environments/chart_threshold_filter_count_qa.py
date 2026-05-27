"""
Threshold-filter counting on a bar chart: render N=4-12 labeled bars, then ask
"how many items have value < T / > T / between A and B?".

Mirrors reference Factoid threshold-filter-count style (e.g. IDX 679 "how many
project phase stage requires less than 60 hours?", IDX 13 "which department had
the FY2017 fund budget value between 40,000,000 and 60,000,000?").

Output: a single non-negative integer (the count), e.g. "3".
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter


# Phrasing mirrors reference: "how many X are above/below T", "how many fall
# between A and B", optionally "exceeds" / "less than" wording variants.
_TEMPLATES_ABOVE = [
    "How many bars have a value greater than {T}? Reply with a non-negative integer in <answer>...</answer>.",
    "Count the number of categories whose value exceeds {T}. Integer answer in <answer>...</answer>.",
    "How many items in the chart show a value above {T}? Integer count in <answer>...</answer>.",
    "Count how many bars are higher than {T}. Place the integer in <answer>...</answer>.",
    "How many of the bars in the chart are greater than {T}? Integer in <answer>...</answer>.",
    "Determine the number of categories whose value is strictly above {T}. Integer in <answer>...</answer>.",
    "How many bar values exceed {T}? Reply with the integer count in <answer>...</answer>.",
    "Read the chart and count the bars taller than {T}. Integer in <answer>...</answer>.",
    "How many entries in the chart have a value larger than {T}? Integer in <answer>...</answer>.",
    "From the chart, how many bars exceed {T}? Integer count in <answer>...</answer>.",
    "Count the bars whose height is greater than {T}. Integer answer in <answer>...</answer>.",
    "How many of the categories shown have a value above {T}? Integer in <answer>...</answer>.",
    "Tally the number of bars exceeding {T}. Integer count in <answer>...</answer>.",
    "Find how many bars in the chart are above the value {T}. Integer in <answer>...</answer>.",
    "Inspect the chart: how many bars have a value greater than {T}? Integer in <answer>...</answer>.",
    "How many categories in the bar chart show a value strictly greater than {T}? Integer in <answer>...</answer>.",
]

_TEMPLATES_BELOW = [
    "How many bars have a value less than {T}? Reply with a non-negative integer in <answer>...</answer>.",
    "Count the number of categories whose value is below {T}. Integer answer in <answer>...</answer>.",
    "How many items in the chart show a value under {T}? Integer count in <answer>...</answer>.",
    "Count how many bars are lower than {T}. Place the integer in <answer>...</answer>.",
    "How many of the bars in the chart are less than {T}? Integer in <answer>...</answer>.",
    "Determine the number of categories whose value is strictly below {T}. Integer in <answer>...</answer>.",
    "How many bar values are less than {T}? Reply with the integer count in <answer>...</answer>.",
    "Read the chart and count the bars shorter than {T}. Integer in <answer>...</answer>.",
    "How many entries in the chart have a value smaller than {T}? Integer in <answer>...</answer>.",
    "From the chart, how many bars are below {T}? Integer count in <answer>...</answer>.",
    "Count the bars whose height is less than {T}. Integer answer in <answer>...</answer>.",
    "How many of the categories shown have a value under {T}? Integer in <answer>...</answer>.",
    "Tally the number of bars under {T}. Integer count in <answer>...</answer>.",
    "Find how many bars in the chart are below the value {T}. Integer in <answer>...</answer>.",
    "Inspect the chart: how many bars have a value less than {T}? Integer in <answer>...</answer>.",
    "How many categories in the bar chart show a value strictly less than {T}? Integer in <answer>...</answer>.",
]

_TEMPLATES_BETWEEN = [
    "How many bars have a value between {A} and {B} (inclusive)? Integer in <answer>...</answer>.",
    "Count the categories whose value falls between {A} and {B} (inclusive). Integer in <answer>...</answer>.",
    "How many items in the chart fall in the range [{A}, {B}]? Integer count in <answer>...</answer>.",
    "How many bars have a value of at least {A} and at most {B}? Integer in <answer>...</answer>.",
    "From the chart, how many categories have value within {A} to {B} inclusive? Integer in <answer>...</answer>.",
    "Count the number of bars whose value lies between {A} and {B} (both endpoints included). Integer in <answer>...</answer>.",
    "How many of the bars are between {A} and {B} (inclusive)? Integer answer in <answer>...</answer>.",
    "Tally the bars whose value sits in [{A}, {B}]. Integer in <answer>...</answer>.",
    "Determine how many categories show a value from {A} to {B}, inclusive. Integer in <answer>...</answer>.",
    "Read the chart and count the bars with value between {A} and {B} inclusive. Integer in <answer>...</answer>.",
    "How many bars lie inside the range {A} to {B} (endpoints included)? Integer in <answer>...</answer>.",
    "Count the bars whose value satisfies {A} <= value <= {B}. Integer answer in <answer>...</answer>.",
    "How many categories have a value at least {A} and not exceeding {B}? Integer in <answer>...</answer>.",
    "From the chart: how many bars are within {A} and {B} inclusive? Integer in <answer>...</answer>.",
    "Find the number of bars whose value falls between {A} and {B}, inclusive. Integer in <answer>...</answer>.",
    "How many entries in the chart show a value between {A} and {B} (inclusive)? Integer count in <answer>...</answer>.",
]


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches", "Pears", "Cherries", "Plums", "Kiwis", "Berries", "Limes"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil", "France", "UK", "Italy", "Spain", "Canada", "Mexico"],
    ["Sales", "Eng", "Mktg", "Finance", "HR", "Legal", "Support", "R&D", "Ops", "Design", "QA", "PR"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["NY", "LA", "CHI", "HOU", "PHX", "PHL", "DAL", "AUS", "SD", "SF", "BOS", "SEA"],
    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
]

_Y_LABELS = ["Revenue", "Units Sold", "Visitors", "Score", "Customers", "Hours", "Sales", "Count"]


class ChartThresholdFilterCountQA(StandaloneVisualEnv):
    ENV_NAME = "chart_threshold_filter_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # n_cat: 4 → 12
        if level == 0:
            n_cat = 4
        elif level <= 2:
            n_cat = 5
        elif level <= 4:
            n_cat = 6 + (level - 3)  # 7,8
        elif level <= 6:
            n_cat = 8
        elif level <= 8:
            n_cat = 10
        else:
            n_cat = 12
        # mode pool — at L0 force 'above' for trivial deterministic case
        if level == 0:
            modes = ["above"]
        elif level <= 2:
            modes = ["above", "below"]
        elif level <= 5:
            modes = ["above", "below", "between"]
        else:
            modes = ["above", "below", "between"]
        return {"level": level, "n_cat": n_cat, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1061 + level * 79 + 13)

        n = cfg["n_cat"]
        mode = rng.choice(cfg["modes"])
        cats_pool = rng.choice(_CATEGORY_POOLS)
        cats = rng.sample(cats_pool, n)
        y_label = rng.choice(_Y_LABELS)

        # L0 trivial: 4 bars [10,20,30,40], T=25 → above=2 (matches plan spec)
        if level == 0:
            base_vals = [10, 20, 30, 40]
            rng.shuffle(base_vals)
            values = base_vals
            T = 25
            count = sum(1 for v in values if v > T)
            sidx = (self.seed or 0) % len(_TEMPLATES_ABOVE)
            question = _TEMPLATES_ABOVE[sidx].format(T=T)
            answer = str(count)
            img = self._render_bars(cats, values, y_label, rng, hline=None)
            n_opts = rng.choice([4, 5])
            question, answer = maybe_to_mcq_letter(
                question, answer, rng, prob=0.5, n_options=n_opts)
            return question, answer, img

        # General path: pick clean integer values, then choose threshold(s)
        # such that no bar value equals the threshold (to keep < / > unambiguous).
        # Use multiples of 5 at lower levels for legibility.
        if level <= 3:
            step = 5
            v_min, v_max = 10, 100
        elif level <= 6:
            step = 5
            v_min, v_max = 10, 200
        else:
            step = 1
            v_min, v_max = 5, 250

        # Generate distinct-ish values (allow dups) but guarantee at least 2
        # are above and 2 below the eventual threshold for non-trivial counts.
        for _ in range(60):
            values = sorted(rng.choice(list(range(v_min, v_max + 1, step)))
                            for _ in range(n))
            # Need spread
            if values[-1] - values[0] >= 3 * step:
                break
        else:
            return None

        # Shuffle order so chart isn't sorted
        order = list(range(n))
        rng.shuffle(order)
        values = [values[i] for i in sorted(range(n), key=lambda k: order[k])]

        if mode in ("above", "below"):
            # Pick T strictly inside the (min, max) range so count >= 1 and <= n-1.
            sorted_v = sorted(set(values))
            if len(sorted_v) < 2:
                return None
            # Pick a midpoint between two consecutive distinct values, rounded
            # to int if step >= 2 to keep T clean. Using midpoint guarantees
            # T != any value.
            i = rng.randrange(len(sorted_v) - 1)
            lo, hi = sorted_v[i], sorted_v[i + 1]
            mid = (lo + hi) / 2.0
            # Prefer integer T when both ends are multiples of 5 and mid lies
            # on .5 — round either down or up but ensure T != any value.
            T_int = int(round(mid))
            if T_int in values:
                # Bump up by 1 (or down) to escape; recompute
                T_int = T_int + 1 if T_int + 1 not in values and T_int + 1 <= sorted_v[-1] else T_int - 1
            if T_int in values or T_int <= sorted_v[0] or T_int >= sorted_v[-1]:
                # Use the exact float mid (e.g. 22.5) — display as float in Q
                T_display = mid if mid != int(mid) else int(mid)
            else:
                T_display = T_int
            # Recompute count with chosen T
            T_use = T_display if isinstance(T_display, (int, float)) else float(T_display)
            if mode == "above":
                count = sum(1 for v in values if v > T_use)
                tpl = _TEMPLATES_ABOVE
            else:
                count = sum(1 for v in values if v < T_use)
                tpl = _TEMPLATES_BELOW
            if count == 0 or count == n:
                return None  # degenerate
            sidx = (self.seed or 0) % len(tpl)
            question = tpl[sidx].format(T=T_display)
            answer = str(count)
        else:  # between A B inclusive
            sorted_v = sorted(set(values))
            if len(sorted_v) < 3:
                return None
            # Pick A in [min, mid], B in (A, max] — both equal to actual values
            # so the count is meaningful (inclusive bounds match real bars).
            A_idx = rng.randrange(0, len(sorted_v) - 1)
            B_idx = rng.randrange(A_idx + 1, len(sorted_v))
            A, B = sorted_v[A_idx], sorted_v[B_idx]
            count = sum(1 for v in values if A <= v <= B)
            if count == 0 or count == n:
                return None
            sidx = (self.seed or 0) % len(_TEMPLATES_BETWEEN)
            question = _TEMPLATES_BETWEEN[sidx].format(A=A, B=B)
            answer = str(count)

        img = self._render_bars(cats, values, y_label, rng, hline=None)
        n_opts = rng.choice([4, 5])
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts)
        return question, answer, img

    def _render_bars(self, cats: List[str], values: List[float], y_label: str,
                     rng: random.Random, hline: Optional[float]) -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        bg_color = "#ffffff"
        n = len(cats)
        fig_w = max(6.5, 0.7 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        bar_colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(range(n), values, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.015,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        if hline is not None:
            ax.axhline(hline, color="#c0392b", linewidth=1.6, linestyle="--",
                       label=f"y = {hline}")
            ax.legend(fontsize=10)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.18 + 2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg_color)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
