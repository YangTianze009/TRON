"""
Chart Aggregate Claim QA — pure single-letter MCQ.

2026-05-05 R5 P5 (Batch 2 Agent B): REWRITTEN to align with real-world
data-chart benchmark style. Render a labeled bar chart and pose claims
about chart-wide aggregates (sum, average, max-min spread, group
sub-aggregates). Pure 5-MCQ output A/B/C/D/E.

Per-level design:
  L0/L1 — 4-MCQ. 4-5 bars, single aggregate (avg or sum) at very loose
          margin. No trap.
  L2/L3 — 5-MCQ ("E. Cannot be determined" 10% trap). 5-6 bars, mixed
          avg/sum/spread aggregates. Looser margin.
  L4/L5 — 5-MCQ. 6-7 bars. avg/sum/spread/max_twice_min mix. 25% trap.
  L6/L7 — 5-MCQ. 8 bars. Group-aggregate (color split) + spread + sum.
          30% trap.
  L8/L9 — 5-MCQ. 8-10 bars with TIGHT clustered values (5%/abs 3 margin).
          Subset-vs-total ratio claim or aggregate-vs-aggregate
          (sum_subset > avg_total*N). 40% trap.

Each option is a self-contained statement; correct = the TRUE one.
Distractors are off-by-N or wrong-direction perturbations.
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


_CATEGORY_POOLS = [
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches",
     "Pears", "Cherries", "Plums", "Berries"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil", "France",
     "UK", "Italy", "Spain"],
    ["Sales", "Eng", "Mktg", "Finance", "HR", "Legal", "Support", "RnD",
     "Ops", "Design"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
     "Oct", "Nov", "Dec"],
    # NB: don't use bare single-letter category labels (e.g. "A".."J")
    # because they would collide with the verifier's MCQ regex, which can
    # match "(A)" inside option text and mis-extract that as the answer.
    ["Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5", "Cat 6", "Cat 7",
     "Cat 8", "Cat 9", "Cat 10"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Dallas", "Austin",
     "Boston"],
]

_Y_LABELS = ["Revenue ($M)", "Units sold", "Visitors (K)", "Score",
             "Customers (K)", "Hours", "Sales", "Count", "Profit (%)"]

_NAMED_COLORS = [
    ("blue", "#2980b9"),
    ("orange", "#e67e22"),
    ("green", "#27ae60"),
    ("purple", "#8e44ad"),
    ("red", "#c0392b"),
]


class ChartAggregateClaimQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — aggregate claims (sum/avg/spread) on a chart."""

    ALLOW_ROTATION = False
    ENV_NAME = "chart_aggregate_claim"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "kinds": ["avg_compare", "sum_compare"],
                "n": 5,
                "use_5_options": False, "trap_rate": 0.0,
                "tight": False,
                "use_groups": False,
            }
        if level <= 3:
            return {
                "kinds": ["avg_compare", "sum_compare", "spread_compare"],
                "n": 6,
                "use_5_options": True, "trap_rate": 0.10,
                "tight": False,
                "use_groups": False,
            }
        if level <= 5:
            return {
                "kinds": ["avg_compare", "sum_compare", "spread_compare",
                          "max_twice_min"],
                "n": 7,
                "use_5_options": True, "trap_rate": 0.25,
                "tight": False,
                "use_groups": False,
            }
        if level <= 7:
            return {
                "kinds": ["avg_compare", "sum_compare", "spread_compare",
                          "group_diff"],
                "n": 8,
                "use_5_options": True, "trap_rate": 0.30,
                "tight": False,
                "use_groups": True,
            }
        # L8 / L9 — tight values
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (R5 was 0.40, env still
        # saturated 97.5%).
        return {
            "kinds": ["sum_subset_vs_total", "avg_compare",
                      "subset_sum_compare"],
            "n": 9,
            "use_5_options": True, "trap_rate": 0.50,
            "tight": True,
            "use_groups": False,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1117 + level * 67 + 31)

        for _ in range(20):
            res = self._try(cfg, sub_rng)
            if res is not None:
                return res
        return None

    # ------------------------------------------------------------------ #
    def _setup_chart(self, cfg, rng):
        n = cfg["n"]
        pool = list(rng.choice(_CATEGORY_POOLS))
        if len(pool) < n:
            return None
        rng.shuffle(pool)
        cats = pool[:n]
        if cfg.get("tight"):
            base = rng.randint(40, 90)
            vals = [base + rng.randint(-15, 15) for _ in range(n)]
        else:
            vals = [rng.randint(20, 200) for _ in range(n)]
        seen = set()
        for i in range(n):
            while vals[i] in seen:
                vals[i] += 1
            seen.add(vals[i])
        y_label = rng.choice(_Y_LABELS)
        return cats, vals, y_label

    def _try(self, cfg, rng):
        kind = rng.choice(cfg["kinds"])
        if kind == "avg_compare":
            return self._try_aggregate_compare(cfg, rng, "average")
        if kind == "sum_compare":
            return self._try_aggregate_compare(cfg, rng, "sum")
        if kind == "spread_compare":
            return self._try_aggregate_compare(cfg, rng, "spread")
        if kind == "max_twice_min":
            return self._try_max_twice_min(cfg, rng)
        if kind == "group_diff":
            return self._try_group_diff(cfg, rng)
        if kind == "sum_subset_vs_total":
            return self._try_sum_subset_vs_total(cfg, rng)
        if kind == "subset_sum_compare":
            return self._try_subset_sum_compare(cfg, rng)
        return None

    # ------------------------------------------------------------------ #
    # Generic aggregate_compare: 4 candidate values for the aggregate.
    # ------------------------------------------------------------------ #
    def _try_aggregate_compare(self, cfg, rng, agg_kind):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(vals)

        if agg_kind == "average":
            actual = sum(vals) / n
            phrase = f"The average {y_label.lower()} across all bars"
        elif agg_kind == "sum":
            actual = float(sum(vals))
            phrase = f"The sum of all bar values in {y_label.lower()}"
        elif agg_kind == "spread":
            actual = float(max(vals) - min(vals))
            phrase = (f"The spread (max - min) of {y_label.lower()} "
                      "across all bars")
        else:
            return None

        actual_int = int(round(actual))
        if actual_int <= 0:
            return None

        # 4 options: 1 correct value + 3 wrong values with sufficient margin.
        if cfg.get("tight"):
            # Tight: distractors at ±5% relative or ±3 abs (whichever bigger)
            step = max(int(round(actual * 0.05)), 3)
        else:
            step = max(int(round(actual * 0.20)), 10)

        offsets = [-2, -1, 1, 2]
        rng.shuffle(offsets)
        wrong_vals = []
        seen = {actual_int}
        for off in offsets:
            c = actual_int + off * step
            if c <= 0 or c in seen:
                continue
            wrong_vals.append(c)
            seen.add(c)
            if len(wrong_vals) == 3:
                break
        if len(wrong_vals) < 3:
            return None

        all_vals = [actual_int] + wrong_vals[:3]
        rng.shuffle(all_vals)
        opts = [f"{phrase} is approximately {v}." for v in all_vals]
        true_pos = all_vals.index(actual_int)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            for off in [-3, 3, -4, 4]:
                c = actual_int + off * step
                if c > 0 and c not in seen:
                    opts[true_pos] = f"{phrase} is approximately {c}."
                    seen.add(c)
                    correct_letter = "E"
                    break
            else:
                return None
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        phrase_lc = phrase[0].lower() + phrase[1:]
        stems = [
            f"According to the bar chart, which statement is TRUE? ( )",
            f"Inspect the chart and select the correct aggregate "
            f"claim. ( )",
            f"From the bars shown, identify the TRUE statement about "
            f"{phrase_lc}. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}. Be concise."
        )
        image = self._render_bars(cats, vals, y_label, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # max_twice_min: "max > 2 * min" — 4 candidate truth statements.
    # ------------------------------------------------------------------ #
    def _try_max_twice_min(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        mx = max(vals)
        mn = min(vals)
        actual = mx > 2 * mn
        # Force enough margin so the answer is unambiguous.
        margin = abs(mx - 2 * mn)
        if margin < 10:
            return None

        i_max = vals.index(mx)
        i_min = vals.index(mn)
        cat_max = cats[i_max]
        cat_min = cats[i_min]

        # 4 candidate statements about the "max-vs-min" relationship:
        #   correct + 3 perturbations
        true_stmt = (f"The maximum bar ({cat_max}) is more than "
                     f"twice the minimum bar ({cat_min}).") \
            if actual else \
            (f"The maximum bar ({cat_max}) is at most twice the "
             f"minimum bar ({cat_min}).")
        # Wrong perturbations
        false_choices = [
            (f"The maximum bar ({cat_min}) is more than twice the "
             f"minimum bar ({cat_max})."),  # swap names
            (f"The minimum bar ({cat_min}) is greater than the "
             f"maximum bar ({cat_max})."),  # nonsensical
            (f"All bars in the chart have equal {y_label.lower()}."),
        ]
        if actual:
            # Add another false: opposite direction
            false_choices.append(
                f"The maximum bar ({cat_max}) is at most twice the "
                f"minimum bar ({cat_min}).")
        else:
            false_choices.append(
                f"The maximum bar ({cat_max}) is more than twice the "
                f"minimum bar ({cat_min}).")
        false_choices = false_choices[:3]
        rng.shuffle(false_choices)

        all_opts = [true_stmt] + false_choices
        rng.shuffle(all_opts)
        true_pos = all_opts.index(true_stmt)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true_stmt with another wrong distractor
            extra_wrong = (f"The bar with the highest value cannot be "
                           f"determined from the chart.")
            all_opts[true_pos] = extra_wrong
            correct_letter = "E"
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {all_opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the bar chart, which statement is TRUE? ( )",
            f"Inspect the bars and pick the correct max-vs-min "
            f"statement. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render_bars(cats, vals, y_label, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # group_diff: 2-color partition; claim about which group avg is higher.
    # ------------------------------------------------------------------ #
    def _try_group_diff(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(vals)
        # Random partition with at least 2 in each group.
        for _ in range(40):
            assignments = [rng.randint(0, 1) for _ in range(n)]
            a_count = sum(1 for x in assignments if x == 0)
            if 2 <= a_count <= n - 2:
                break
        else:
            return None

        cA, cB = rng.sample(_NAMED_COLORS, 2)
        nameA, hexA = cA
        nameB, hexB = cB

        a_vals = [v for v, x in zip(vals, assignments) if x == 0]
        b_vals = [v for v, x in zip(vals, assignments) if x == 1]
        avgA = sum(a_vals) / len(a_vals)
        avgB = sum(b_vals) / len(b_vals)
        if avgA == 0 or avgB == 0:
            return None
        rel = abs(avgA - avgB) / max(avgA, avgB)
        if rel < 0.20:  # require ≥20% margin
            return None

        if avgA > avgB:
            higher_name, lower_name = nameA, nameB
        else:
            higher_name, lower_name = nameB, nameA

        # 4 options: TRUE direction + 3 perturbations
        true_stmt = (f"The average of {higher_name} bars is higher "
                     f"than the average of {lower_name} bars.")
        false_stmts = [
            (f"The average of {lower_name} bars is higher than the "
             f"average of {higher_name} bars."),  # swap
            (f"The average of {higher_name} bars equals the average "
             f"of {lower_name} bars."),  # equal
            (f"All {higher_name} bars are below all {lower_name} "
             f"bars."),  # too strong
        ]
        rng.shuffle(false_stmts)
        all_opts = [true_stmt] + false_stmts[:3]
        rng.shuffle(all_opts)
        true_pos = all_opts.index(true_stmt)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            all_opts[true_pos] = (f"The number of {higher_name} bars "
                                  f"and {lower_name} bars is identical.")
            correct_letter = "E"
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {all_opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the chart (each bar is colored {nameA} or "
            f"{nameB}), which statement is TRUE? ( )",
            f"Inspect the {nameA} vs {nameB} bars. Which group-average "
            f"claim is correct? ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render_color_bars(cats, vals, assignments, y_label,
                                        rng, nameA, hexA, nameB, hexB)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # sum_subset_vs_total (L8/L9): "sum of TOP-3 bars exceeds N% of total"
    # ------------------------------------------------------------------ #
    def _try_sum_subset_vs_total(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(vals)
        if n < 5:
            return None

        sorted_idx = sorted(range(n), key=lambda i: -vals[i])
        top_k = 3
        sum_top = sum(vals[i] for i in sorted_idx[:top_k])
        total = sum(vals)
        if total == 0:
            return None
        actual_pct = round(sum_top * 100 / total)
        if actual_pct < 20 or actual_pct > 90:
            return None

        # 4 options: 1 true + 3 off by N pct points.
        offsets = [-3, -2, 2, 3, -5, 5, -8, 8]
        wrong_pcts = []
        seen = {actual_pct}
        for off in offsets:
            c = actual_pct + off
            if c <= 0 or c >= 100 or c in seen:
                continue
            wrong_pcts.append(c)
            seen.add(c)
            if len(wrong_pcts) == 3:
                break
        if len(wrong_pcts) < 3:
            return None

        all_pcts = [actual_pct] + wrong_pcts[:3]
        rng.shuffle(all_pcts)
        opts = [(f"The top {top_k} bars together account for "
                 f"approximately {p}% of the total {y_label.lower()}.")
                for p in all_pcts]
        true_pos = all_pcts.index(actual_pct)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            for off in [-10, 10, -12, 12]:
                c = actual_pct + off
                if 0 < c < 100 and c not in seen:
                    opts[true_pos] = (f"The top {top_k} bars together "
                                      f"account for approximately {c}% "
                                      f"of the total {y_label.lower()}.")
                    seen.add(c)
                    correct_letter = "E"
                    break
            else:
                return None
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the bar chart, which statement is TRUE? ( ) "
            f"(Compute the sum of the {top_k} largest bars relative to "
            f"the total of all bars.)",
            f"Inspect the chart and pick the TRUE claim about the share "
            f"of the top {top_k} bars in the total. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}. Be concise."
        )
        image = self._render_bars(cats, vals, y_label, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # subset_sum_compare (L8/L9): "sum of {X, Y, Z} > Z+W" type compound.
    # ------------------------------------------------------------------ #
    def _try_subset_sum_compare(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(vals)
        if n < 5:
            return None

        # Pick a single index "X" and 2-3 "others" to be combined.
        idxs = list(range(n))
        rng.shuffle(idxs)
        x = idxs[0]
        n_combined = rng.choice([2, 3])
        others = idxs[1:1 + n_combined]
        sum_others = sum(vals[k] for k in others)
        if sum_others == 0:
            return None
        actually_greater = vals[x] > sum_others
        gap = abs(vals[x] - sum_others)
        if gap < max(5, int(0.05 * sum_others)):
            return None

        others_str = " + ".join(cats[k] for k in others)
        true_stmt = (f"{cats[x]} is greater than {others_str} combined.") \
            if actually_greater else \
            (f"{cats[x]} is less than {others_str} combined.")
        false_stmts = [
            (f"{cats[x]} is less than {others_str} combined."
             if actually_greater else
             f"{cats[x]} is greater than {others_str} combined."),
            f"{cats[x]} equals {others_str} combined.",
            f"{cats[x]} is more than twice {others_str} combined.",
        ]
        rng.shuffle(false_stmts)
        all_opts = [true_stmt] + false_stmts[:3]
        rng.shuffle(all_opts)
        true_pos = all_opts.index(true_stmt)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            all_opts[true_pos] = (f"None of the listed bars in the "
                                  f"chart contribute to {cats[x]}.")
            correct_letter = "E"
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {all_opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the bar chart, which statement is TRUE? ( ) "
            f"(Sum the values of the named categories before "
            f"comparing.)",
            f"Inspect the chart and identify the correct sum-comparison "
            f"claim. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}. Be concise."
        )
        image = self._render_bars(cats, vals, y_label, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_bars(self, cats, vals, y_label, rng):
        palette = list(rng.choice(self._COLOR_PALETTES))
        rng.shuffle(palette)
        bar_colors = [palette[i % len(palette)] for i in range(len(cats))]
        n = len(cats)
        fig_w = max(6.5, 0.85 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bars = ax.bar(range(n), vals, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.015,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _render_color_bars(self, cats, vals, assignments, y_label, rng,
                           nameA, hexA, nameB, hexB):
        bar_colors = [hexA if a == 0 else hexB for a in assignments]
        n = len(cats)
        fig_w = max(6.5, 0.85 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bars = ax.bar(range(n), vals, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6,
                      label=None)
        max_v = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.015,
                    str(int(v) if abs(v - int(v)) < 1e-6 else f"{v:.1f}"),
                    ha="center", va="bottom", fontsize=11, color="#111",
                    fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Manual legend handles
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=hexA, label=nameA),
                           Patch(facecolor=hexB, label=nameB)],
                  loc="upper right", fontsize=10)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
