"""
Composite infographic with multiple distinct viz elements.

Single image containing 2-3 sub-elements (donut + bar + KPI text annotation).
Question requires either:
  - identifying a specific value from one element ("what is the value of bar X?",
    "what percentage does the donut slice X take?", "what KPI is shown?")
  - computing across two elements (e.g. donut percent of a KPI total)

Mirrors Conversational reference samples that probe dashboard composites:
  - IDX 1288 ("dashboard ... can you list all chart types ... peak weekly visitors?")
  - IDX 1256 ("what is this dashboard about? overall percentage change...?")
  - IDX 1353 ("what are the three subjects? how many more cases...?")
  - IDX 1077 (multi-bar dashboard with multiple categories)
  - IDX 1203 (multi-city, multi-metric dashboard with cross-element reasoning)

Output: numeric value or short label string.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter


_TEMPLATES_BAR_VALUE = [
    "Look at the infographic. In the bar chart panel, what value does the bar labeled '{label}' show? Numeric answer in <answer>...</answer>.",
    "From the infographic, read the bar '{label}' in the bar panel. Place the numeric value in <answer>...</answer>.",
    "Inspect the bar panel of the infographic. What is the value of '{label}'? Numeric in <answer>...</answer>.",
    "In the infographic, the bar chart shows several categories. What value is shown for '{label}'? Numeric in <answer>...</answer>.",
]

_TEMPLATES_DONUT_VALUE = [
    "Look at the infographic. In the donut/pie chart panel, what percentage does '{label}' represent? Integer percent (no % sign) in <answer>...</answer>.",
    "From the infographic, read the donut chart slice for '{label}'. Place the percentage (no % sign) in <answer>...</answer>.",
    "Inspect the pie panel of the infographic. What percentage does '{label}' occupy? Integer percent (no % sign) in <answer>...</answer>.",
    "In the infographic, the donut chart shows shares. What percent does '{label}' have? Numeric (no % sign) in <answer>...</answer>.",
]

_TEMPLATES_KPI = [
    "Look at the infographic. What is the headline KPI value displayed (the large bold number)? Numeric answer in <answer>...</answer>.",
    "From the infographic, read the large KPI number shown at the top. Place the numeric value in <answer>...</answer>.",
    "Inspect the infographic. What number is shown as the headline KPI? Numeric in <answer>...</answer>.",
    "What is the headline number (the large highlighted figure) in the infographic? Numeric in <answer>...</answer>.",
]

_TEMPLATES_BAR_MAX_LABEL = [
    "Look at the infographic. In the bar chart panel, which category has the largest bar? Reply with its label in <answer>...</answer>.",
    "From the infographic's bar chart, which category is the highest? Place the label in <answer>...</answer>.",
    "In the bar panel of the infographic, which bar is the tallest? Reply with the label in <answer>...</answer>.",
]

_TEMPLATES_BAR_MIN_LABEL = [
    "Look at the infographic. In the bar chart panel, which category has the smallest bar? Reply with its label in <answer>...</answer>.",
    "From the infographic's bar chart, which category is the lowest? Place the label in <answer>...</answer>.",
    "In the bar panel of the infographic, which bar is the shortest? Reply with the label in <answer>...</answer>.",
]

_TEMPLATES_BAR_SUM = [
    "Look at the infographic. Sum the values across all bars in the bar chart panel. Numeric answer in <answer>...</answer>.",
    "From the infographic's bar chart, what is the total of all bar values? Numeric in <answer>...</answer>.",
    "Add up every bar in the infographic's bar chart panel. Numeric answer in <answer>...</answer>.",
]

_TEMPLATES_KPI_TIMES_DONUT = [
    "Look at the infographic. The KPI value (large bold number) is split by the donut chart shares. What numeric value corresponds to '{label}'s share of the KPI? Numeric answer in <answer>...</answer>.",
    "From the infographic, multiply the headline KPI by '{label}'s donut share to get '{label}'s contribution. Numeric in <answer>...</answer>.",
    "If the donut chart represents shares of the KPI total shown in the infographic, what is the value of '{label}'? Numeric in <answer>...</answer>.",
    "Take the KPI total in the infographic and apply '{label}'s percentage from the donut chart. What is the resulting value? Numeric in <answer>...</answer>.",
]


_BAR_LABEL_POOLS = [
    ["Alpha", "Bravo", "Charlie", "Delta"],
    ["North", "South", "East", "West"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["A", "B", "C", "D", "E"],
    ["Mon", "Tue", "Wed", "Thu", "Fri"],
]

_DONUT_LABEL_POOLS = [
    ["Web", "Mobile", "Desktop", "Other"],
    ["P1", "P2", "P3", "P4"],
    ["Group A", "Group B", "Group C", "Group D"],
    ["Type 1", "Type 2", "Type 3"],
]

_KPI_LABELS = ["Total Visitors", "Total Sales", "Total Users", "Total Revenue",
                "Total Customers", "Total Units"]
_BAR_TITLES = ["Monthly Trend", "By Category", "Per Region", "Performance"]
_DONUT_TITLES = ["Channel Mix", "Segment Share", "Distribution", "Allocation"]


def _format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"


class InfographicCompositeQA(StandaloneVisualEnv):
    ENV_NAME = "infographic_composite"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_bars": 3, "n_donut": 3, "include_donut": False,
                    "include_kpi": True, "modes": ["kpi"], "trivial_l0": True}
        if level <= 2:
            return {"n_bars": 4, "n_donut": 4, "include_donut": True,
                    "include_kpi": True,
                    "modes": ["bar_value", "donut_value", "kpi"],
                    "trivial_l0": False}
        if level <= 4:
            return {"n_bars": 4, "n_donut": 4, "include_donut": True,
                    "include_kpi": True,
                    "modes": ["bar_value", "donut_value", "kpi",
                              "bar_max_label", "bar_min_label"],
                    "trivial_l0": False}
        if level <= 6:
            return {"n_bars": 5, "n_donut": 4, "include_donut": True,
                    "include_kpi": True,
                    "modes": ["bar_value", "donut_value", "bar_max_label",
                              "bar_min_label", "bar_sum"],
                    "trivial_l0": False}
        if level <= 8:
            return {"n_bars": 5, "n_donut": 4, "include_donut": True,
                    "include_kpi": True,
                    "modes": ["bar_sum", "kpi_times_donut", "donut_value"],
                    "trivial_l0": False}
        # 2026-05-04: bumped L9 difficulty — was 95% saturated.
        # More bars/donut slices + only hardest modes.
        return {"n_bars": 7, "n_donut": 6, "include_donut": True,
                "include_kpi": True,
                "modes": ["kpi_times_donut"],
                "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1879 + level * 191 + 101)

        # Pick a bar pool large enough; fallback to letters if needed.
        sufficient_bar = [p for p in _BAR_LABEL_POOLS if len(p) >= cfg["n_bars"]]
        if not sufficient_bar:
            bar_pool = list("ABCDEFGHI")
        else:
            bar_pool = rng.choice(sufficient_bar)
        bar_labels = bar_pool[:cfg["n_bars"]]
        sufficient_donut = [p for p in _DONUT_LABEL_POOLS if len(p) >= cfg["n_donut"]]
        if not sufficient_donut:
            donut_pool = [f"D{i+1}" for i in range(cfg["n_donut"])]
        else:
            donut_pool = rng.choice(sufficient_donut)
        donut_labels = donut_pool[:cfg["n_donut"]]

        if cfg["trivial_l0"]:
            # Trivial: 3 bars, KPI is large bold = 100. Ask for KPI directly.
            bar_values = [10, 20, 30]
            kpi_value = 100
            kpi_label = "Total Visitors"
            mode = "kpi"
            answer = str(kpi_value)
            tlist = _TEMPLATES_KPI
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx]
            donut_values = None
            img = self._render(bar_labels, bar_values, donut_labels,
                                donut_values, kpi_label, kpi_value,
                                cfg["include_donut"], rng,
                                bar_title=rng.choice(_BAR_TITLES),
                                donut_title=rng.choice(_DONUT_TITLES))
            return question, answer, img

        # Non-trivial
        n_bars = cfg["n_bars"]
        bar_values = [rng.randint(15, 95) for _ in range(n_bars)]
        # Ensure unique max & min so argmax/argmin questions are unambiguous
        max_v = max(bar_values)
        if bar_values.count(max_v) > 1:
            idx = bar_values.index(max_v)
            bar_values[idx] = max_v + rng.randint(5, 15)
        min_v = min(bar_values)
        if bar_values.count(min_v) > 1:
            idx = bar_values.index(min_v)
            bar_values[idx] = max(5, min_v - rng.randint(3, 10))

        # Donut shares (sum to 100, snap to 5%)
        n_donut = cfg["n_donut"]
        snap = 5
        steps = 100 // snap
        for _attempt in range(20):
            cuts = sorted(rng.sample(range(1, steps), n_donut - 1))
            chunks = []
            prev = 0
            for c in cuts:
                chunks.append((c - prev) * snap)
                prev = c
            chunks.append((steps - prev) * snap)
            if all(v >= 5 for v in chunks):
                break
        donut_values = chunks
        if sum(donut_values) != 100:
            donut_values[0] += 100 - sum(donut_values)
        if min(donut_values) < 5:
            return None
        # Ensure distinct shares so donut argmin/argmax queries are clean
        # (we don't ask those, but distinctness reduces ambiguity)
        if len(set(donut_values)) < len(donut_values):
            # Nudge by ±5
            for i in range(len(donut_values)):
                if donut_values.count(donut_values[i]) > 1:
                    j = (i + 1) % len(donut_values)
                    if donut_values[j] - 5 >= 5:
                        donut_values[j] -= 5
                        donut_values[i] += 5
                        break

        kpi_label = rng.choice(_KPI_LABELS)
        # KPI value: pick a round number so kpi_times_donut gives clean answer
        kpi_value = rng.choice([100, 200, 250, 400, 500, 800, 1000, 2000, 5000])

        mode = rng.choice(cfg["modes"])

        if mode == "bar_value":
            target_idx = rng.randrange(n_bars)
            answer = _format_num(bar_values[target_idx])
            tlist = _TEMPLATES_BAR_VALUE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=bar_labels[target_idx])
        elif mode == "donut_value":
            target_idx = rng.randrange(n_donut)
            answer = str(int(donut_values[target_idx]))
            tlist = _TEMPLATES_DONUT_VALUE
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=donut_labels[target_idx])
        elif mode == "kpi":
            answer = str(kpi_value)
            tlist = _TEMPLATES_KPI
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx]
        elif mode == "bar_max_label":
            target_idx = bar_values.index(max(bar_values))
            answer = bar_labels[target_idx]
            tlist = _TEMPLATES_BAR_MAX_LABEL
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx]
        elif mode == "bar_min_label":
            target_idx = bar_values.index(min(bar_values))
            answer = bar_labels[target_idx]
            tlist = _TEMPLATES_BAR_MIN_LABEL
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx]
        elif mode == "bar_sum":
            answer = _format_num(sum(bar_values))
            tlist = _TEMPLATES_BAR_SUM
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx]
        elif mode == "kpi_times_donut":
            target_idx = rng.randrange(n_donut)
            answer_val = kpi_value * donut_values[target_idx] / 100.0
            answer = _format_num(answer_val)
            tlist = _TEMPLATES_KPI_TIMES_DONUT
            sidx = (self.seed or 0) % len(tlist)
            question = tlist[sidx].format(label=donut_labels[target_idx])
        else:
            return None

        img = self._render(bar_labels, bar_values, donut_labels, donut_values,
                            kpi_label, kpi_value, cfg["include_donut"], rng,
                            bar_title=rng.choice(_BAR_TITLES),
                            donut_title=rng.choice(_DONUT_TITLES))

        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        # For label-style answers, use other bar/donut labels as the
        # candidate pool.
        n_opts = rng.choice([4, 5])
        if mode in ("bar_max_label", "bar_min_label"):
            cand_pool = list(bar_labels)
        else:
            cand_pool = None
        question, answer = maybe_to_mcq_letter(
            question, answer, rng, prob=0.5, n_options=n_opts,
            candidate_pool=cand_pool)
        return question, answer, img

    def _render(self, bar_labels: List[str], bar_values: List[float],
                donut_labels: List[str], donut_values: Optional[List[int]],
                kpi_label: str, kpi_value: int,
                include_donut: bool, rng: random.Random,
                bar_title: str = "By Category",
                donut_title: str = "Distribution") -> Image.Image:
        palette = rng.choice(self._COLOR_PALETTES)
        # Layout: top row = KPI banner; bottom row = bar | donut
        if include_donut and donut_values is not None:
            fig = plt.figure(figsize=(11.5, 6.0), dpi=120)
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 3], hspace=0.45,
                                  wspace=0.35)
            ax_kpi = fig.add_subplot(gs[0, :])
            ax_bar = fig.add_subplot(gs[1, 0])
            ax_donut = fig.add_subplot(gs[1, 1])
        else:
            fig = plt.figure(figsize=(8.0, 6.0), dpi=120)
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.45)
            ax_kpi = fig.add_subplot(gs[0, 0])
            ax_bar = fig.add_subplot(gs[1, 0])
            ax_donut = None
        fig.patch.set_facecolor("#ffffff")

        # KPI banner
        ax_kpi.set_facecolor("#f5f8fc")
        ax_kpi.set_xticks([])
        ax_kpi.set_yticks([])
        for spine in ax_kpi.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)
        ax_kpi.text(0.5, 0.78, kpi_label, fontsize=13, ha="center",
                    va="center", color="#555555", weight="normal")
        ax_kpi.text(0.5, 0.32, f"{int(kpi_value):,}", fontsize=28, ha="center",
                    va="center", color=palette[0], weight="bold")

        # Bar panel
        ax_bar.set_facecolor("#ffffff")
        n_b = len(bar_labels)
        bar_colors = [palette[i % len(palette)] for i in range(n_b)]
        bars = ax_bar.bar(range(n_b), bar_values, color=bar_colors,
                            edgecolor="#1a1a1a", linewidth=0.6)
        max_bv = max(bar_values)
        for b, v in zip(bars, bar_values):
            ax_bar.text(b.get_x() + b.get_width() / 2, v + max_bv * 0.025,
                        str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                        ha="center", va="bottom", fontsize=10, color="#111",
                        fontweight="bold")
        ax_bar.set_xticks(range(n_b))
        ax_bar.set_xticklabels(bar_labels, fontsize=10)
        ax_bar.set_ylim(0, max_bv * 1.22 + 4)
        ax_bar.set_title(bar_title, fontsize=11, loc="left")
        ax_bar.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)

        # Donut panel
        if ax_donut is not None and donut_values is not None:
            ax_donut.set_facecolor("#ffffff")
            pal2 = list(palette)
            rng.shuffle(pal2)
            d_colors = [pal2[i % len(pal2)] for i in range(len(donut_values))]
            ax_donut.pie(donut_values, labels=donut_labels, colors=d_colors,
                         autopct="%1.0f%%", startangle=90,
                         wedgeprops={"width": 0.45, "edgecolor": "#000",
                                     "linewidth": 1.0},
                         textprops={"fontsize": 10, "fontweight": "bold"})
            ax_donut.set_title(donut_title, fontsize=11, loc="left")

        # tight_layout warns when pie axes are present; suppress it.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Allow base class to handle most. Add label-substring tolerance for
        # string answers (categories like "Alpha", "North", etc.) without
        # bleeding into numeric strings.
        # First try base class behavior
        if super()._check_answer(predicted, ground_truth):
            return True
        # If GT is alphabetic-only label, allow case-insensitive substring
        g = ground_truth.strip().lower().rstrip(".")
        p = predicted.strip().lower().rstrip(".")
        if g.replace(" ", "").isalpha() and len(g) >= 2:
            if g in p and len(p) <= len(g) * 4:
                return True
        return False
