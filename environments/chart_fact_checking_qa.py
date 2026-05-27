"""
Chart Fact Checking QA — pure single-letter MCQ.

2026-05-05 R5 P5 (Batch 2 Agent B): REWRITTEN to align with real-world
data-chart benchmark style. Render a chart and pose a declarative claim
about the chart contents (argmax/min, ranking, threshold-crossing count,
percentage difference, multi-bar comparison). Pure 5-MCQ output A/B/C/D/E.

Per-level design:
  L0/L1 — 4-MCQ. Single-bar lookup ("Bar X exceeds T — True / False").
          5-7 bars, no trap.
  L2/L3 — 5-MCQ ("E. Cannot be determined" 10% trap). 6-8 bars,
          two-bar comparison ("A is greater than B").
  L4/L5 — 5-MCQ. Multi-step argmax/argmin claims ("X is the highest").
          25% trap.
  L6/L7 — 5-MCQ. Threshold-crossing count over time series. 30% trap.
  L8/L9 — 5-MCQ. Percentage difference between two bars (off-by-1pp
          distractors), tight values, 40% trap.

Each option is a self-contained statement; correct = the TRUE one. Wrong
options are subtle perturbations — wrong category named, off-by-N count,
wrong direction (greater vs less), wrong percentage.

Question phrasing (textbook chart-question style, generic data charts):
  "According to the chart, which statement is TRUE? ( )"
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
    ["USA", "China", "Germany", "Japan", "India", "Brazil",
     "France", "UK", "Italy", "Spain"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Dallas",
     "Austin", "Boston"],
    ["Sales", "Engineering", "Marketing", "Finance", "HR", "Legal",
     "Support", "RnD"],
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches",
     "Pears", "Cherries"],
    ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
     "Dec"],
    ["Q1", "Q2", "Q3", "Q4"],
]

_Y_LABELS = ["Revenue ($M)", "Visitors (K)", "Sales (units)", "Score",
             "Customers (K)", "Population (K)", "Profit (%)", "Hours"]


class ChartFactCheckingQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — fact-checking declarative claims on a chart."""

    ALLOW_ROTATION = False
    ENV_NAME = "chart_fact_checking"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "claim_kind": "single_threshold", "n_cat": 5,
                "use_5_options": False, "trap_rate": 0.0,
                "tight": False,
            }
        if level <= 3:
            return {
                "claim_kind": "two_compare", "n_cat": 7,
                "use_5_options": True, "trap_rate": 0.10,
                "tight": False,
            }
        if level <= 5:
            return {
                "claim_kind": "argmax_argmin", "n_cat": 7,
                "use_5_options": True, "trap_rate": 0.25,
                "tight": False,
            }
        if level <= 7:
            return {
                "claim_kind": "threshold_count", "n_cat": 8,
                "use_5_options": True, "trap_rate": 0.30,
                "tight": False,
            }
        # L8/L9
        return {
            "claim_kind": "pct_diff_near", "n_cat": 8,
            "use_5_options": True, "trap_rate": 0.40,
            "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1607 + level * 109 + 47)

        for _ in range(20):
            res = self._try(cfg, sub_rng)
            if res is not None:
                return res
        return None

    # ------------------------------------------------------------------ #
    def _try(self, cfg, rng):
        kind = cfg["claim_kind"]
        if kind == "single_threshold":
            return self._try_single_threshold(cfg, rng)
        if kind == "two_compare":
            return self._try_two_compare(cfg, rng)
        if kind == "argmax_argmin":
            return self._try_argmax_argmin(cfg, rng)
        if kind == "threshold_count":
            return self._try_threshold_count(cfg, rng)
        if kind == "pct_diff_near":
            return self._try_pct_diff_near(cfg, rng)
        return None

    # ------------------------------------------------------------------ #
    def _setup_chart(self, cfg, rng):
        n = cfg["n_cat"]
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
        # Make all values unique to avoid argmax ambiguity.
        seen = set()
        for i in range(n):
            while vals[i] in seen:
                vals[i] += 1
            seen.add(vals[i])
        y_label = rng.choice(_Y_LABELS)
        return cats, vals, y_label

    # ------------------------------------------------------------------ #
    # single_threshold (L0/L1): "Bar X exceeds T units" — 4 candidate
    # statements, exactly 1 true.
    # ------------------------------------------------------------------ #
    def _try_single_threshold(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(cats)

        # Pick T cleanly between two values; build claims like "X exceeds T"
        sv = sorted(set(vals))
        if len(sv) < 2:
            return None
        split = rng.randrange(0, len(sv) - 1)
        T = (sv[split] + sv[split + 1]) // 2
        if T == sv[split]:
            T = sv[split] + 1

        above = [i for i, v in enumerate(vals) if v > T]
        below = [i for i, v in enumerate(vals) if v <= T]
        if len(above) < 1 or len(below) < 3:
            return None

        # 4 options: 1 true (a category that exceeds T), 3 false (don't exceed T).
        true_i = rng.choice(above)
        false_i = rng.sample(below, 3)
        idxs = [true_i] + false_i
        rng.shuffle(idxs)

        opts = [f"{cats[i]} exceeds {T} {y_label.lower()}." for i in idxs]
        true_pos = idxs.index(true_i)
        correct_letter = "ABCD"[true_pos]

        opt_lines = "\n".join(f"{c}. {opts[k]}"
                              for k, c in enumerate("ABCD"))
        stems = [
            f"According to the chart (threshold line T = {T} drawn as a "
            f"red dashed line), which statement is TRUE? ( )",
            f"Look at the bar chart with threshold T = {T} marked as a "
            f"red dashed line. Which one of the following claims is "
            f"correct? ( )",
            f"From the chart, identify the TRUE statement about the "
            f"{y_label.lower()} compared to the threshold T = {T}. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: A, B, C, or D."
        )
        image = self._render_bars(cats, vals, y_label, T=T,
                                  show_threshold=True, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # two_compare (L2/L3): "X is greater than Y" — 4 candidate statements
    # of pair comparisons, exactly 1 true.
    # ------------------------------------------------------------------ #
    def _try_two_compare(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(cats)
        if n < 4:
            return None

        # Pick TRUE pair (i, j) where vals[i] > vals[j].
        attempts = 0
        chosen_pairs = []
        seen_pairs = set()
        while len(chosen_pairs) < 4 and attempts < 100:
            attempts += 1
            i, j = rng.sample(range(n), 2)
            key = (i, j)
            if key in seen_pairs or (j, i) in seen_pairs:
                continue
            seen_pairs.add(key)
            chosen_pairs.append((i, j))
        if len(chosen_pairs) < 4:
            return None

        # Make 1 true claim "X > Y" and 3 false claims "X > Y" where X < Y.
        true_pair = None
        wrong_pairs = []
        for i, j in chosen_pairs:
            if vals[i] > vals[j]:
                if true_pair is None:
                    true_pair = (i, j)
                else:
                    # Flip to make wrong: claim "j > i" but actually i > j
                    wrong_pairs.append((j, i))
            else:
                # Already wrong if claim "i > j" but i < j
                wrong_pairs.append((i, j))
        if true_pair is None or len(wrong_pairs) < 3:
            return None
        wrong_pairs = wrong_pairs[:3]

        all_pairs = [true_pair] + wrong_pairs
        rng.shuffle(all_pairs)
        opts = [f"{cats[a]} is greater than {cats[b]} in {y_label.lower()}."
                for a, b in all_pairs]
        true_pos = all_pairs.index(true_pair)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true claim with another wrong claim — all 4 wrong → E.
            attempts2 = 0
            replacement = None
            while attempts2 < 80:
                attempts2 += 1
                i, j = rng.sample(range(n), 2)
                if vals[i] < vals[j]:
                    replacement = (i, j)
                    break
            if replacement is None:
                return None
            opts[true_pos] = (f"{cats[replacement[0]]} is greater than "
                              f"{cats[replacement[1]]} in "
                              f"{y_label.lower()}.")
            correct_letter = "E"
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the bar chart, which one of the following "
            f"comparisons is TRUE? ( )",
            f"Inspect the bars and find the correct comparison. Which "
            f"statement holds? ( )",
            f"Read the chart values and identify the TRUE comparison "
            f"of {y_label.lower()}. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render_bars(cats, vals, y_label, T=None,
                                  show_threshold=False, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # argmax_argmin (L4/L5): identify which is highest or lowest.
    # ------------------------------------------------------------------ #
    def _try_argmax_argmin(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(cats)
        if n < 5:
            return None

        kind = rng.choice(["argmax", "argmin"])
        if kind == "argmax":
            true_i = max(range(n), key=lambda k: vals[k])
            other = sorted(range(n), key=lambda k: -vals[k])[1:]
            stem_word = "highest"
        else:
            true_i = min(range(n), key=lambda k: vals[k])
            other = sorted(range(n), key=lambda k: vals[k])[1:]
            stem_word = "lowest"

        # 3 wrong choices = the next 3 closest.
        wrong = other[:3]
        idxs = [true_i] + wrong
        rng.shuffle(idxs)
        opts = [(f"{cats[i]} has the {stem_word} {y_label.lower()} in "
                 f"the chart.") for i in idxs]
        true_pos = idxs.index(true_i)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true with a different non-true index → all 4 wrong.
            possible = [k for k in range(n) if k != true_i and k not in wrong]
            if not possible:
                return None
            replacement = rng.choice(possible)
            opts[true_pos] = (f"{cats[replacement]} has the {stem_word} "
                              f"{y_label.lower()} in the chart.")
            correct_letter = "E"
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {opts[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the bar chart, which statement is TRUE? ( )",
            f"From the chart, identify the correct claim about the "
            f"{stem_word} {y_label.lower()}. ( )",
            f"Inspect the chart and pick the TRUE rank claim. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render_bars(cats, vals, y_label, T=None,
                                  show_threshold=False, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # threshold_count (L6/L7): "exactly N of the bars exceed T".
    # ------------------------------------------------------------------ #
    def _try_threshold_count(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(cats)

        # Pick T such that the count of vals exceeding T is between 2 and n-2.
        sv = sorted(set(vals))
        if len(sv) < 4:
            return None
        # Try a few placements.
        attempts = 0
        T = None
        actual_count = None
        while attempts < 30:
            attempts += 1
            split = rng.randrange(1, len(sv) - 2)
            T_candidate = (sv[split] + sv[split + 1]) // 2
            if T_candidate == sv[split]:
                T_candidate = sv[split] + 1
            cnt = sum(1 for v in vals if v > T_candidate)
            if 2 <= cnt <= n - 2:
                T = T_candidate
                actual_count = cnt
                break
        if T is None:
            return None

        # 4 options each stating a specific count: 1 true + 3 off-by-N.
        offsets = [-2, -1, 1, 2]
        rng.shuffle(offsets)
        wrong_counts = []
        seen_counts = {actual_count}
        for off in offsets:
            c = actual_count + off
            if c < 0 or c > n or c in seen_counts:
                continue
            wrong_counts.append(c)
            seen_counts.add(c)
            if len(wrong_counts) == 3:
                break
        # Try wider offsets if we don't have enough.
        for off in [-3, 3, -4, 4]:
            if len(wrong_counts) >= 3:
                break
            c = actual_count + off
            if c < 0 or c > n or c in seen_counts:
                continue
            wrong_counts.append(c)
            seen_counts.add(c)
        if len(wrong_counts) < 3:
            return None

        all_counts = [actual_count] + wrong_counts[:3]
        rng.shuffle(all_counts)
        opts = [f"Exactly {c} categories exceed {T} {y_label.lower()}."
                for c in all_counts]
        true_pos = all_counts.index(actual_count)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true count with another wrong count (further offset).
            for off in [-3, 3, -4, 4, -5, 5]:
                c = actual_count + off
                if 0 <= c <= n and c not in seen_counts:
                    opts[true_pos] = (f"Exactly {c} categories exceed "
                                      f"{T} {y_label.lower()}.")
                    seen_counts.add(c)
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
            f"According to the chart (threshold line T = {T} drawn as a "
            f"red dashed line), which statement is TRUE? ( )",
            f"Inspect each bar against the threshold T = {T}. Which "
            f"count statement is correct? ( )",
            f"Look at the bars and the threshold T = {T}. Pick the TRUE "
            f"count claim. ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render_bars(cats, vals, y_label, T=T,
                                  show_threshold=True, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # pct_diff_near (L8/L9): "percentage difference between A and B is N%"
    # with off-by-1pp distractors.
    # ------------------------------------------------------------------ #
    def _try_pct_diff_near(self, cfg, rng):
        setup = self._setup_chart(cfg, rng)
        if setup is None:
            return None
        cats, vals, y_label = setup
        n = len(cats)
        if n < 4:
            return None

        i, j = rng.sample(range(n), 2)
        if vals[j] == 0:
            return None
        actual_pct = round(abs(vals[i] - vals[j]) * 100 / vals[j])
        if actual_pct < 5 or actual_pct > 200:
            return None

        # 4 options with different pct values: 1 correct + 3 off by 1-3 pp.
        offsets = [-2, -1, 1, 2, -3, 3]
        rng.shuffle(offsets)
        wrong_pcts = []
        seen = {actual_pct}
        for off in offsets:
            c = actual_pct + off
            if c <= 0 or c in seen:
                continue
            wrong_pcts.append(c)
            seen.add(c)
            if len(wrong_pcts) == 3:
                break
        if len(wrong_pcts) < 3:
            return None

        all_pcts = [actual_pct] + wrong_pcts[:3]
        rng.shuffle(all_pcts)
        opts = [(f"The percentage difference between {cats[i]} and "
                 f"{cats[j]} in {y_label.lower()} is {p}%.") for p in all_pcts]
        true_pos = all_pcts.index(actual_pct)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true pct with another distant value.
            for off in [-4, 4, -5, 5, -6, 6]:
                c = actual_pct + off
                if c > 0 and c not in seen:
                    opts[true_pos] = (f"The percentage difference between "
                                      f"{cats[i]} and {cats[j]} in "
                                      f"{y_label.lower()} is {c}%.")
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
            f"According to the chart, which statement about the "
            f"percentage difference is TRUE? ( )",
            f"Inspect the bars for {cats[i]} and {cats[j]}. Which "
            f"percentage-difference claim is correct? ( ) "
            f"(Hint: difference relative to {cats[j]}.)",
            f"Read the bar chart and identify the TRUE pairwise "
            f"percentage-difference claim. ( ) "
            f"(Hint: difference relative to {cats[j]}.)",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}. Be concise."
        )
        image = self._render_bars(cats, vals, y_label, T=None,
                                  show_threshold=False, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_bars(self, cats, vals, y_label, T=None,
                     show_threshold=False, rng=None):
        palette = list(rng.choice(self._COLOR_PALETTES))
        rng.shuffle(palette)
        n = len(cats)
        fig_w = max(6.5, 0.85 * n + 2.2)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bar_colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(range(n), vals, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max_v * 0.015, str(int(v)),
                    ha="center", va="bottom", fontsize=11,
                    color="#111", fontweight="bold")
        if show_threshold and T is not None:
            ax.axhline(T, color="#c0392b", linestyle="--",
                       linewidth=1.6, label=f"T = {T}")
            ax.legend(loc="upper right", fontsize=10)
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
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
