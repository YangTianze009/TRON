"""
Chart Set-Membership Claim QA — pure single-letter MCQ.

2026-05-05 R5 P5 (Batch 2 Agent B): REWRITTEN to align with real-world
data-chart benchmark style. Render a labeled bar chart (with optional
threshold line at higher levels), then ask whether a stated set-membership
claim about specific named bars is true or false. Pure 5-MCQ output —
A/B/C/D/E.

Per-level design (set-membership benchmark style):
  L0/L1 — 4-MCQ. 4 bars, single-bar lookup ("Does X exceed T?"),
          dashed threshold line drawn, no trap.
  L2/L3 — 5-MCQ ("E. No correct answer" 10% trap). 5 bars,
          two-bar set membership ("X and Y both exceed T").
  L4-L5 — 5-MCQ. 6-7 bars, three-bar set membership ("X, Y, Z all exceed T").
          25% trap rate.
  L6/L7 — 5-MCQ. 8 bars. Set-intersection across two conditions
          ("which is in BOTH top-3 of group A AND top-5 of group B"). 30% trap.
  L8/L9 — 5-MCQ. 8-10 cluttered bars with tight values. Two-condition
          set intersection + ranking. 40% trap. Distractor includes
          "off by one" subset choices.

Question phrasing (textbook chart-question style, generic data charts):
  Q-style 1: "According to the chart, which of the following statements
              is TRUE? ( )"  with 4 candidate set-claims as options.
  Q-style 2: "Which categories satisfy <condition>? ( )" with sets of
              category names as options.
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
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
     "Iota", "Kappa"],
    ["Apples", "Bananas", "Oranges", "Grapes", "Mangoes", "Peaches",
     "Pears", "Cherries", "Plums", "Berries"],
    ["USA", "China", "Germany", "Japan", "India", "Brazil", "France",
     "UK", "Italy", "Spain"],
    ["Sales", "Eng", "Mktg", "Finance", "HR", "Legal", "Support", "RnD",
     "Ops", "Design"],
    ["NY", "LA", "Chicago", "Houston", "Phoenix", "Dallas", "Austin",
     "Boston", "Denver", "Seattle"],
    ["Region 1", "Region 2", "Region 3", "Region 4", "Region 5",
     "Region 6", "Region 7", "Region 8", "Region 9", "Region 10"],
    ["Q1 2020", "Q2 2020", "Q3 2020", "Q4 2020", "Q1 2021", "Q2 2021",
     "Q3 2021", "Q4 2021", "Q1 2022", "Q2 2022"],
    ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
     "Dec"],
]

_Y_LABELS = ["Revenue ($M)", "Units sold", "Visitors (K)", "Score",
             "Customers (K)", "Hours", "Sales", "Count", "Population (K)",
             "Profit (%)"]


class ChartSetMembershipClaimQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — set-membership claims on a bar chart."""

    ALLOW_ROTATION = False  # numeric labels orientation-sensitive
    ENV_NAME = "chart_set_membership_claim"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n_cat": 4, "set_size": 1, "use_5_options": False,
                "trap_rate": 0.0, "tight": False,
                "qmode": "exceed_t",
                "show_threshold": True,
            }
        if level <= 3:
            return {
                "n_cat": 5, "set_size": 2, "use_5_options": True,
                "trap_rate": 0.10, "tight": False,
                "qmode": "exceed_t",
                "show_threshold": True,
            }
        if level <= 5:
            return {
                "n_cat": 7, "set_size": 3, "use_5_options": True,
                "trap_rate": 0.25, "tight": False,
                "qmode": "exceed_t",
                "show_threshold": True,
            }
        if level <= 7:
            return {
                "n_cat": 8, "set_size": 3, "use_5_options": True,
                "trap_rate": 0.30, "tight": False,
                "qmode": "rank_topk",
                "show_threshold": False,
            }
        # L8 / L9 — set-intersection across 2 grouped series, tight.
        # 2026-05-05 Phase A: bump trap_rate to 0.5 (env still saturated 95%).
        return {
            "n_cat": 8, "set_size": 3, "use_5_options": True,
            "trap_rate": 0.50, "tight": True,
            "qmode": "intersect_topk_two_groups",
            "show_threshold": False,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1063 + level * 79 + 23)

        for _ in range(20):
            res = self._try(cfg, sub_rng)
            if res is not None:
                return res
        return None

    # ------------------------------------------------------------------ #
    def _try(self, cfg, rng):
        qmode = cfg["qmode"]
        if qmode == "exceed_t":
            return self._try_exceed_t(cfg, rng)
        if qmode == "rank_topk":
            return self._try_rank_topk(cfg, rng)
        if qmode == "intersect_topk_two_groups":
            return self._try_intersect_two_groups(cfg, rng)
        return None

    # ------------------------------------------------------------------ #
    # exceed_t: single-condition set membership against threshold line T.
    # ------------------------------------------------------------------ #
    def _try_exceed_t(self, cfg, rng):
        n = cfg["n_cat"]
        set_size = cfg["set_size"]
        cats_pool = list(rng.choice(_CATEGORY_POOLS))
        if len(cats_pool) < n:
            return None
        cats = rng.sample(cats_pool, n)
        y_label = rng.choice(_Y_LABELS)

        # Generate values; place T cleanly between sorted values.
        values = [rng.randint(20, 200) for _ in range(n)]
        sv = sorted(set(values))
        if len(sv) < 2:
            values = [v + i for i, v in enumerate(values)]
            sv = sorted(set(values))
        split = rng.randrange(0, len(sv) - 1)
        T = (sv[split] + sv[split + 1]) // 2
        if T == sv[split]:
            T = sv[split] + 1

        above_idx = sorted([i for i, v in enumerate(values) if v > T])
        below_idx = sorted([i for i, v in enumerate(values) if v <= T])
        if len(above_idx) < 1 or len(below_idx) < 1:
            return None

        # Build the TRUE candidate: a set of `set_size` categories all above T.
        if len(above_idx) < set_size:
            return None
        true_above = sorted(rng.sample(above_idx, set_size))

        # Build candidate set-claims (each is a tuple of indices).
        candidate_sets = self._build_candidate_sets(
            true_above, above_idx, below_idx, set_size, rng,
            n_distractors=4, target_total=4)
        if candidate_sets is None:
            return None

        # Each option is a candidate (sorted-tuple-of-indices) string-claim.
        option_strs = []
        for cand in candidate_sets:
            names = [cats[i] for i in cand]
            option_strs.append(self._fmt_set_claim(names, T, y_label))

        # The TRUE option is the one whose set is exactly true_above.
        true_idx = candidate_sets.index(tuple(true_above))

        # Decide trap (E. No correct answer) — replace true option with a
        # distractor mix so all 4 are wrong.
        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true option with another distractor (at least 1 below T).
            replacement = self._build_one_distractor(
                true_above, above_idx, below_idx, set_size, rng,
                forbid={tuple(s) for s in candidate_sets})
            if replacement is None:
                return None
            candidate_sets[true_idx] = replacement
            option_strs[true_idx] = self._fmt_set_claim(
                [cats[i] for i in replacement], T, y_label)
            correct_letter = "E"
        else:
            # Letter mapping: shuffle option order and pick correct letter.
            order = list(range(len(option_strs)))
            rng.shuffle(order)
            option_strs = [option_strs[i] for i in order]
            old_to_new = {old: new for new, old in enumerate(order)}
            correct_letter = "ABCD"[old_to_new[true_idx]]

        if not use_trap:
            # already shuffled above
            pass
        else:
            # also shuffle so trap doesn't always sit in slot of original true
            order = list(range(len(option_strs)))
            rng.shuffle(order)
            option_strs = [option_strs[i] for i in order]

        # Compose question.
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {option_strs[i]}" for i in range(4)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            f"According to the chart (threshold line T = {T}), which of "
            f"the following statements is TRUE? ( )",
            f"Look at the bar chart and the dashed threshold line at "
            f"T = {T} {y_label.lower()}. Which statement is correct? ( )",
            f"Inspect each bar against the threshold T = {T} "
            f"{y_label.lower()}. Which set-membership claim holds? ( )",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render(cats, values, y_label, T,
                             show_threshold=cfg["show_threshold"], rng=rng)
        return question, correct_letter, image

    @staticmethod
    def _fmt_set_claim(names: List[str], T: int, y_label: str) -> str:
        """Format a set-claim like 'X, Y, and Z all exceed T units.'."""
        if len(names) == 1:
            return f"{names[0]} exceeds {T} {y_label.lower()}."
        if len(names) == 2:
            return f"{names[0]} and {names[1]} both exceed {T} {y_label.lower()}."
        return (", ".join(names[:-1])
                + f", and {names[-1]} all exceed {T} {y_label.lower()}.")

    def _build_candidate_sets(self, true_set, above_idx, below_idx,
                              set_size, rng, n_distractors=4, target_total=4):
        """Build a list of candidate sets including the TRUE one + distractors.

        Each candidate is a sorted tuple of indices of size `set_size`.
        Distractors must include at least one index from below_idx so
        their claim is FALSE. The TRUE one is `true_set` (all above).
        """
        candidates = [tuple(sorted(true_set))]
        seen = {candidates[0]}
        attempts = 0
        while len(candidates) < target_total and attempts < 200:
            attempts += 1
            d = self._build_one_distractor(
                true_set, above_idx, below_idx, set_size, rng,
                forbid=seen)
            if d is None:
                continue
            candidates.append(d)
            seen.add(d)
        if len(candidates) < target_total:
            return None
        return candidates

    @staticmethod
    def _build_one_distractor(true_set, above_idx, below_idx, set_size,
                              rng, forbid):
        """Pick a set of `set_size` indices that includes at least one below_idx
        (so claim is FALSE), and is not in `forbid`."""
        for _ in range(80):
            # Mix: at least 1 from below.
            n_below = rng.randint(1, min(set_size, len(below_idx)))
            n_above = set_size - n_below
            if n_above > len(above_idx):
                continue
            sub_below = rng.sample(below_idx, n_below)
            sub_above = rng.sample(above_idx, n_above)
            cand = tuple(sorted(sub_below + sub_above))
            if cand in forbid:
                continue
            return cand
        return None

    # ------------------------------------------------------------------ #
    # rank_topk: "which of the following sets is exactly the TOP-K bars?"
    # ------------------------------------------------------------------ #
    def _try_rank_topk(self, cfg, rng):
        n = cfg["n_cat"]
        k = cfg["set_size"]
        cats_pool = list(rng.choice(_CATEGORY_POOLS))
        if len(cats_pool) < n:
            return None
        cats = rng.sample(cats_pool, n)
        y_label = rng.choice(_Y_LABELS)

        # Distinct values to ensure unique top-k.
        values = rng.sample(range(20, 250), n)

        sorted_idx = sorted(range(n), key=lambda i: -values[i])
        true_topk = sorted(sorted_idx[:k])

        # Distractors: shift one element by one rank.
        distractors = []
        seen = {tuple(true_topk)}
        attempts = 0
        while len(distractors) < 3 and attempts < 100:
            attempts += 1
            # swap one of the top-k with one slightly outside
            top_pos = rng.randrange(k)
            outside_pos = rng.randrange(k, n)
            new_set = sorted([sorted_idx[i] for i in range(n)
                              if i != top_pos and i < k]
                             + [sorted_idx[outside_pos]])
            cand = tuple(new_set)
            if cand in seen or len(cand) != k:
                continue
            seen.add(cand)
            distractors.append(cand)

        if len(distractors) < 3:
            return None

        candidates = [tuple(true_topk)] + distractors
        true_idx = 0

        use_trap = cfg["use_5_options"] and (rng.random() < cfg["trap_rate"])

        # Format options
        option_strs = [
            ", ".join(cats[i] for i in cand) for cand in candidates
        ]

        if use_trap:
            # Replace true option with another distractor.
            attempts = 0
            replacement = None
            while attempts < 80:
                attempts += 1
                top_pos = rng.randrange(k)
                outside_pos = rng.randrange(k, n)
                new_set = sorted([sorted_idx[i] for i in range(n)
                                  if i != top_pos and i < k]
                                 + [sorted_idx[outside_pos]])
                cand = tuple(new_set)
                if cand not in seen and len(cand) == k:
                    replacement = cand
                    seen.add(cand)
                    break
            if replacement is None:
                return None
            candidates[true_idx] = replacement
            option_strs[true_idx] = ", ".join(cats[i] for i in replacement)
            correct_letter = "E"
        else:
            order = list(range(4))
            rng.shuffle(order)
            option_strs = [option_strs[i] for i in order]
            old_to_new = {old: new for new, old in enumerate(order)}
            correct_letter = "ABCD"[old_to_new[true_idx]]

        opt_letters = ["A", "B", "C", "D", "E"]
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {option_strs[i]}" for i in range(4)
        )
        opt_lines += f"\nE. No correct answer"
        letter_str = "A, B, C, D, or E"

        stems = [
            f"According to the chart, which set of categories has the "
            f"top {k} highest {y_label.lower()}? ( )",
            f"Inspect the bars and identify the {k} categories with the "
            f"largest {y_label.lower()}. Which set is correct? ( )",
            f"Looking at the chart, which group lists exactly the top "
            f"{k} bars by {y_label.lower()}? ( )",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render(cats, values, y_label, T=None,
                             show_threshold=False, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # intersect_topk_two_groups: 2 grouped series; "which categories
    # are in TOP-3 of series A AND TOP-5 of series B?".
    # ------------------------------------------------------------------ #
    def _try_intersect_two_groups(self, cfg, rng):
        n = cfg["n_cat"]
        cats_pool = list(rng.choice(_CATEGORY_POOLS))
        if len(cats_pool) < n:
            return None
        cats = rng.sample(cats_pool, n)
        y_label = rng.choice(_Y_LABELS)
        seriesA_name = rng.choice(["2020", "2022", "Q1", "Group A",
                                   "Region North"])
        seriesB_name = rng.choice(["2024", "Q4", "Group B",
                                   "Region South"])
        if seriesA_name == seriesB_name:
            seriesB_name = "Group B"

        # Independent values.
        if cfg.get("tight"):
            base = rng.randint(40, 80)
            valsA = [base + rng.randint(-12, 12) for _ in range(n)]
            valsB = [base + rng.randint(-12, 12) for _ in range(n)]
            # Ensure unique within each series.
            valsA = self._make_unique(valsA, rng)
            valsB = self._make_unique(valsB, rng)
        else:
            valsA = rng.sample(range(20, 250), n)
            valsB = rng.sample(range(20, 250), n)

        kA = 3
        kB = 5
        topA = set(sorted(range(n), key=lambda i: -valsA[i])[:kA])
        topB = set(sorted(range(n), key=lambda i: -valsB[i])[:kB])
        intersection = sorted(topA & topB)
        if len(intersection) < 2 or len(intersection) > 3:
            return None
        true_set = tuple(intersection)

        # Distractors: subsets of (topA only) or (topB only).
        only_A = sorted(topA - topB)
        only_B = sorted(topB - topA)
        distractors = []
        seen = {true_set}
        attempts = 0
        while len(distractors) < 3 and attempts < 200:
            attempts += 1
            # Mix true-element + a wrong-only element
            mix_size = len(true_set)
            if not only_A or not only_B or len(intersection) < 1:
                # fallback: pick 2 from topA, 1 from topB-only
                pool_A = sorted(topA)
                pool_B = sorted(topB)
                if len(pool_A) >= mix_size:
                    cand = tuple(sorted(rng.sample(pool_A, mix_size)))
                else:
                    continue
            else:
                # 1 true + (mix_size-1) from A_or_B-only
                pool_extra = list(only_A) + list(only_B)
                if len(pool_extra) < mix_size - 1:
                    continue
                base_kept = rng.sample(intersection,
                                       max(1, mix_size - 2))
                rest = rng.sample(pool_extra,
                                  mix_size - len(base_kept))
                cand = tuple(sorted(base_kept + rest))
            if cand in seen or len(cand) != mix_size:
                continue
            # Confirm cand is NOT a subset of intersection of size mix_size.
            if set(cand) == set(intersection):
                continue
            seen.add(cand)
            distractors.append(cand)

        if len(distractors) < 3:
            return None

        candidates = [true_set] + distractors[:3]
        true_idx = 0

        use_trap = rng.random() < cfg["trap_rate"]

        option_strs = [
            ", ".join(cats[i] for i in cand) for cand in candidates
        ]

        if use_trap:
            # Replace true with another distractor.
            replacement = None
            attempts = 0
            while attempts < 80:
                attempts += 1
                # build a totally false set
                pool_extra = list(only_A) + list(only_B)
                base_kept = []
                if len(pool_extra) < len(true_set):
                    continue
                rest = rng.sample(pool_extra, len(true_set))
                cand = tuple(sorted(rest))
                if cand in seen or set(cand) == set(intersection):
                    continue
                replacement = cand
                seen.add(cand)
                break
            if replacement is None:
                return None
            candidates[true_idx] = replacement
            option_strs[true_idx] = ", ".join(cats[i] for i in replacement)
            correct_letter = "E"
        else:
            order = list(range(4))
            rng.shuffle(order)
            option_strs = [option_strs[i] for i in order]
            old_to_new = {old: new for new, old in enumerate(order)}
            correct_letter = "ABCD"[old_to_new[true_idx]]

        opt_letters = ["A", "B", "C", "D", "E"]
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {option_strs[i]}" for i in range(4)
        )
        opt_lines += f"\nE. No correct answer"
        letter_str = "A, B, C, D, or E"

        stems = [
            f"According to the grouped chart, which categories appear in "
            f"BOTH the top {kA} of {seriesA_name} AND the top {kB} of "
            f"{seriesB_name}? ( )",
            f"Look at the two-series chart and identify the categories "
            f"that are simultaneously in the top {kA} of {seriesA_name} "
            f"and the top {kB} of {seriesB_name}. Which option is "
            f"correct? ( )",
            f"From the chart, which set of categories belongs to BOTH "
            f"the top {kA} of {seriesA_name} AND the top {kB} of "
            f"{seriesB_name}? ( )",
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}."
        )
        image = self._render_grouped(cats, valsA, valsB, y_label,
                                     seriesA_name, seriesB_name, rng)
        return question, correct_letter, image

    @staticmethod
    def _make_unique(vals, rng):
        seen = set()
        out = []
        for v in vals:
            while v in seen:
                v += rng.choice([-1, 1])
            seen.add(v)
            out.append(v)
        return out

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, cats: List[str], values: List[int], y_label: str,
                T: Optional[int], show_threshold: bool,
                rng: random.Random) -> Image.Image:
        palette = list(rng.choice(self._COLOR_PALETTES))
        rng.shuffle(palette)
        n = len(cats)
        fig_w = max(6.5, 0.8 * n + 2.2)
        fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        bar_colors = [palette[i % len(palette)] for i in range(n)]
        bars = ax.bar(range(n), values, color=bar_colors,
                      edgecolor="#1a1a1a", linewidth=0.6)
        max_v = max(values)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max_v * 0.015, str(int(v)),
                    ha="center", va="bottom",
                    fontsize=11, color="#111", fontweight="bold")
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

    def _render_grouped(self, cats: List[str], valsA: List[int],
                        valsB: List[int], y_label: str,
                        nameA: str, nameB: str,
                        rng: random.Random) -> Image.Image:
        palette = list(rng.choice(self._COLOR_PALETTES))
        rng.shuffle(palette)
        colorA = palette[0]
        colorB = palette[1] if len(palette) > 1 else palette[0]
        n = len(cats)
        fig_w = max(7.5, 0.95 * n + 2.4)
        fig, ax = plt.subplots(figsize=(fig_w, 4.8), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = list(range(n))
        bw = 0.4
        bars_a = ax.bar([xi - bw / 2 for xi in x], valsA, bw,
                        color=colorA, label=nameA,
                        edgecolor="#1a1a1a", linewidth=0.5)
        bars_b = ax.bar([xi + bw / 2 for xi in x], valsB, bw,
                        color=colorB, label=nameB,
                        edgecolor="#1a1a1a", linewidth=0.5)
        max_v = max(valsA + valsB)
        for b, v in zip(bars_a, valsA):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max_v * 0.015, str(int(v)),
                    ha="center", va="bottom", fontsize=9, color="#111",
                    fontweight="bold")
        for b, v in zip(bars_b, valsB):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max_v * 0.015, str(int(v)),
                    ha="center", va="bottom", fontsize=9, color="#111",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20 if n > 6 else 0,
                           ha="right" if n > 6 else "center", fontsize=10)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(0, max_v * 1.20 + 2)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
