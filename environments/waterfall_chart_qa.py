"""
Waterfall chart QA — running total, incremental changes, bridge reasoning.
Targets: chart-reading, statistical reasoning.

Diversity & difficulty redesign (2026-04-16):
- Level-aware sub_rng controls data (previously _rng -> L0=L9 same seed).
- Structural L0 vs L9 differences:
   * L0: 4-5 bars, only simplest qtypes (final_value, net_change), values labeled.
   * L9: 7-9 bars, advanced qtypes (running_total_at, cumulative_at_step,
     identify_largest_drop), MCQ format, values only visible as bar heights.
- 4+ question phrasings per qtype; title/colors randomized per seed.
"""
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_CAT_POOLS = [
    ["Starting", "Sales", "Returns", "Marketing", "Operations", "Tax",
     "Bonus", "Other", "Final"],
    ["Q1 Start", "Revenue", "COGS", "SG&A", "R&D", "Other", "Fx", "Q1 End"],
    ["Opening", "Item 1", "Item 2", "Item 3", "Item 4",
     "Item 5", "Item 6", "Closing"],
    ["Open", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Close"],
    ["Start", "Prod", "Dist", "Ret", "Ref", "Fin", "Disc", "End"],
]

_TITLES = [
    "Waterfall Chart", "Financial Bridge", "Running Total Chart",
    "Stepwise Change Chart", "Period Bridge",
]

_PHRASINGS = {
    "final_value": [
        "What is the final (closing) value in the waterfall chart?",
        "Read the final bar — what value does the chart end at?",
        "What is the ending total shown in the diagram?",
    ],
    "net_change": [
        "What is the net change from start to end?",
        "How much did the value change overall (end - start)?",
        "Compute the net change across the waterfall.",
    ],
    "largest_increase": [
        "Which category contributes the largest positive change?",
        "Which step shows the biggest upward increment?",
        "Name the category with the greatest positive bar.",
    ],
    "largest_decrease": [
        "Which category contributes the largest negative change?",
        "Which step has the greatest downward increment?",
        "Name the category with the largest drop.",
    ],
    "running_total_at": [
        "What is the running total after the '{cat}' step?",
        "What value is reached immediately after applying '{cat}'?",
        "Compute the cumulative value after the '{cat}' bar.",
    ],
    "count_positive": [
        "How many categories show a positive (upward) change?",
        "Count the upward bars in the waterfall.",
        "How many positive increments are there?",
    ],
    "count_negative": [
        "How many categories show a negative (downward) change?",
        "Count the downward bars.",
    ],
    "identify_largest_drop": [
        "Which category has the largest negative change, and what is the "
        "absolute magnitude of that drop? Answer as 'Category, magnitude'.",
        "Find the biggest drop: name the category and its size (magnitude). "
        "Answer as 'Category, magnitude'.",
    ],
    "cumulative_at_step": [
        "What is the running total (cumulative value) after the '{cat}' step?",
        "After '{cat}' is applied, what is the cumulative value?",
    ],
}

class WaterfallChartQA(StandaloneVisualEnv):
    ENV_NAME = "waterfall_chart"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        if level == 0:
            return {"qtypes": ["final_value", "net_change"],
                    "n_min": 4, "n_max": 5, "show_labels": True,
                    "is_mcq": False, "inc_range": (-25, 40)}
        if level <= 2:
            return {"qtypes": ["final_value", "net_change",
                               "largest_increase"],
                    "n_min": 5, "n_max": 6, "show_labels": True,
                    "is_mcq": False, "inc_range": (-30, 45)}
        if level <= 4:
            return {"qtypes": ["largest_increase", "largest_decrease",
                               "net_change", "count_positive"],
                    "n_min": 5, "n_max": 6, "show_labels": True,
                    "is_mcq": False, "inc_range": (-40, 55)}
        if level <= 6:
            return {"qtypes": ["largest_increase", "largest_decrease",
                               "count_positive", "count_negative",
                               "running_total_at"],
                    "n_min": 6, "n_max": 7, "show_labels": False,
                    "is_mcq": False, "inc_range": (-45, 65)}
        if level <= 7:
            return {"qtypes": ["running_total_at", "cumulative_at_step",
                               "identify_largest_drop",
                               "largest_decrease"],
                    "n_min": 7, "n_max": 8, "show_labels": False,
                    "is_mcq": True, "inc_range": (-50, 70)}
        return {"qtypes": ["running_total_at", "cumulative_at_step",
                           "identify_largest_drop",
                           "count_negative", "largest_decrease"],
                "n_min": 7, "n_max": 9, "show_labels": False,
                "is_mcq": True, "inc_range": (-55, 80)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        question_type = parameter.get("question_type",
                                      sub_rng.choice(cfg["qtypes"]))
        if question_type not in cfg["qtypes"]:
            question_type = sub_rng.choice(cfg["qtypes"])

        for _ in range(25):
            r = self._try_generate(sub_rng, cfg, question_type, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, qtype, level):
        # Pick category pool big enough
        pool = rng.choice(_CAT_POOLS)
        n_max = min(cfg["n_max"], len(pool))
        n_min = min(cfg["n_min"], n_max)
        n = rng.randint(n_min, n_max)
        cats = list(pool[:n])

        start_val = rng.randint(40, 220)
        lo_inc, hi_inc = cfg["inc_range"]
        increments = [rng.randint(lo_inc, hi_inc) for _ in range(n - 2)]
        final_val = start_val + sum(increments)
        values = [start_val] + increments + [final_val]
        is_total = [True] + [False] * (n - 2) + [True]

        # Compute answer
        answer = None
        q_text = None

        if qtype == "final_value":
            answer = final_val
            q_text = rng.choice(_PHRASINGS[qtype])
        elif qtype == "net_change":
            answer = final_val - start_val
            q_text = rng.choice(_PHRASINGS[qtype])
        elif qtype == "largest_increase":
            pos = [(cats[i + 1], increments[i])
                   for i in range(len(increments)) if increments[i] > 0]
            if not pos:
                return None
            best = max(pos, key=lambda x: x[1])
            answer = best[0]
            q_text = rng.choice(_PHRASINGS[qtype])
        elif qtype == "largest_decrease":
            neg = [(cats[i + 1], increments[i])
                   for i in range(len(increments)) if increments[i] < 0]
            if not neg:
                return None
            worst = min(neg, key=lambda x: x[1])
            answer = worst[0]
            q_text = rng.choice(_PHRASINGS[qtype])
        elif qtype == "running_total_at":
            idx = rng.randint(1, n - 2)
            running = start_val + sum(increments[:idx])
            answer = running
            q_text = rng.choice(_PHRASINGS[qtype]).format(cat=cats[idx])
        elif qtype == "count_positive":
            answer = sum(1 for inc in increments if inc > 0)
            q_text = rng.choice(_PHRASINGS[qtype])
        elif qtype == "count_negative":
            answer = sum(1 for inc in increments if inc < 0)
            q_text = rng.choice(_PHRASINGS[qtype])
        elif qtype == "cumulative_at_step":
            idx = rng.randint(1, n - 2)
            running = start_val + sum(increments[:idx])
            answer = running
            q_text = rng.choice(_PHRASINGS[qtype]).format(cat=cats[idx])
        elif qtype == "identify_largest_drop":
            neg = [(cats[i + 1], increments[i])
                   for i in range(len(increments)) if increments[i] < 0]
            if not neg:
                return None
            worst = min(neg, key=lambda x: x[1])
            answer = f"{worst[0]}, {abs(worst[1])}"
            q_text = rng.choice(_PHRASINGS[qtype])
        else:
            return None

        # MCQ wrapping (for numeric / category)
        if cfg.get("is_mcq", False) and isinstance(answer, (int, float)):
            is_count = qtype in ("count_positive", "count_negative")
            distractors = set()
            tries = 0
            while len(distractors) < 3 and tries < 60:
                delta = rng.choice([-40, -25, -15, -10, -5, 5, 10, 15, 25, 40])
                cand = answer + delta
                if is_count:
                    # Counts must be non-negative and <= n_bars
                    if cand < 0 or cand > n:
                        tries += 1
                        continue
                if cand != answer:
                    distractors.add(cand)
                tries += 1
            if len(distractors) >= 3:
                opts = [answer] + list(distractors)[:3]
                rng.shuffle(opts)
                letter = chr(ord("A") + opts.index(answer))
                q_text = (q_text + "\n" + "\n".join(
                    f"  ({chr(ord('A')+i)}) {opts[i]}" for i in range(4))
                    + "\nAnswer with a single letter.")
                answer = letter

        # Render chart
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        fig, ax = plt.subplots(figsize=(max(8, n * 1.15), 5))
        self._apply_style(fig, ax, style)

        x = np.arange(n)
        show_labels = cfg.get("show_labels", False)
        for i in range(n):
            if is_total[i]:
                ax.bar(x[i], values[i], color=palette[0], edgecolor="black",
                       linewidth=1)
                if show_labels:
                    ax.text(x[i], values[i] + 2, str(values[i]), ha="center",
                            fontsize=10, fontweight="bold")
            else:
                bottom = start_val + sum(increments[:i - 1])
                val = increments[i - 1]
                color = palette[3] if val >= 0 else palette[4]
                if val >= 0:
                    ax.bar(x[i], val, bottom=bottom, color=color,
                           edgecolor="black", linewidth=1, alpha=0.85)
                    if show_labels:
                        ax.text(x[i], bottom + val + 2, f"+{val}",
                                ha="center", fontsize=9, color="#155724")
                else:
                    ax.bar(x[i], val, bottom=bottom, color=color,
                           edgecolor="black", linewidth=1, alpha=0.85)
                    if show_labels:
                        ax.text(x[i], bottom + val - 4, str(val),
                                ha="center", fontsize=9, color="#721c24")

        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=style["font_size_base"] - 1,
                           rotation=15, ha="right")
        ax.set_ylabel("Value", fontsize=style["font_size_base"])
        title = rng.choice(_TITLES)
        ax.set_title(title, fontsize=style["font_size_base"] + 2,
                     fontweight="bold")

        return q_text, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
