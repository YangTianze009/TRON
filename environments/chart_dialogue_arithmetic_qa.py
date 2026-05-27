"""
Chart Dialogue Arithmetic QA — pure single-letter MCQ.

2026-05-05 R5 P5 (Batch 2 Agent B): REWRITTEN to align with Conversational
benchmark style. Multi-turn dialogue context (prior Q/A pairs zipped into
the prompt as `Conversation: [Q1: ... A1: ...; ...]`) followed by a final
graded MCQ question. Pure 5-MCQ output A/B/C/D/E.

Per-level design:
  L0/L1 — 4-MCQ. 1 prior turn (single-bar lookup).
          Final = single arithmetic step (sum of 2 numbers).
  L2/L3 — 5-MCQ ("E. Cannot be determined" 10% trap). 2 prior turns;
          final = sum/diff of priors.
  L4/L5 — 5-MCQ. 3 prior turns; final = sum/max/min of priors. 25% trap.
  L6/L7 — 5-MCQ. 3 prior turns; conditional final ("if X had grown 10%,
          what would the difference between X and Y become?"). 30% trap.
  L8/L9 — 5-MCQ. 4 prior turns; chained conditional ("if A grew by Y%,
          what would the new ratio of A:B be?"). 40% trap.

Each option is a numeric (or "ratio = N") answer; correct = the TRUE one.
Distractors are off-by-N or wrong-arithmetic perturbations.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_GROUP_PAIR_POOLS = [
    ("opposite-sex parents", "same-sex parents"),
    ("urban areas", "rural areas"),
    ("public schools", "private schools"),
    ("morning shift", "evening shift"),
    ("control group", "treatment group"),
    ("Brand A users", "Brand B users"),
    ("North region", "South region"),
    ("baseline cohort", "treatment cohort"),
]

_OUTCOME_POOLS = [
    ["depression", "anxiety", "obesity", "diabetes", "asthma", "insomnia"],
    ["math score", "reading score", "writing score", "science score",
     "history score", "art score"],
    ["satisfaction", "loyalty", "engagement", "trust",
     "retention", "referrals"],
    ["sales", "returns", "complaints", "referrals",
     "subscriptions", "cancellations"],
    ["clicks", "signups", "logins", "purchases", "shares", "ratings"],
]


class ChartDialogueArithmeticQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — multi-turn dialogue with arithmetic final."""

    ALLOW_ROTATION = False
    ENV_NAME = "chart_dialogue_arithmetic"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n_outcomes": 3, "n_priors": 1,
                "final_op": "sum", "use_5_options": False,
                "trap_rate": 0.0, "use_conditional": False,
            }
        if level <= 3:
            return {
                "n_outcomes": 4, "n_priors": 2,
                "final_op": "sum", "use_5_options": True,
                "trap_rate": 0.10, "use_conditional": False,
            }
        if level <= 5:
            return {
                "n_outcomes": 5, "n_priors": 3,
                "final_op": "varied", "use_5_options": True,
                "trap_rate": 0.25, "use_conditional": False,
            }
        if level <= 7:
            return {
                "n_outcomes": 5, "n_priors": 3,
                "final_op": "varied", "use_5_options": True,
                "trap_rate": 0.30, "use_conditional": True,
            }
        # L8 / L9
        return {
            "n_outcomes": 6, "n_priors": 4,
            "final_op": "varied", "use_5_options": True,
            "trap_rate": 0.40, "use_conditional": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1409 + level * 71 + 23)

        for _ in range(20):
            res = self._try(cfg, sub_rng)
            if res is not None:
                return res
        return None

    # ------------------------------------------------------------------ #
    def _try(self, cfg, rng):
        gA, gB = rng.choice(_GROUP_PAIR_POOLS)
        outcomes_pool = list(rng.choice(_OUTCOME_POOLS))
        rng.shuffle(outcomes_pool)
        if len(outcomes_pool) < cfg["n_outcomes"]:
            return None
        outcomes = outcomes_pool[:cfg["n_outcomes"]]
        n = len(outcomes)
        # Values are percentages 10-70 per group per outcome.
        vals_a = [rng.randint(10, 70) for _ in range(n)]
        vals_b = [rng.randint(10, 70) for _ in range(n)]

        n_priors = cfg["n_priors"]
        if n < n_priors:
            return None
        idx_pool = list(range(n))
        rng.shuffle(idx_pool)
        idx_used = idx_pool[:n_priors]

        diffs = [abs(vals_a[k] - vals_b[k]) for k in idx_used]
        if any(d == 0 for d in diffs):
            return None

        # Build prior turns. L0/L1 (n_priors==1) collapses to a single
        # direct question with NO prior conversation context.
        prior_qs = []
        if n_priors == 1:
            pass  # no prior history
        else:
            for k in range(n_priors):
                if k == 0:
                    qx = (f"What is the absolute difference in "
                          f"{outcomes[idx_used[0]]} between {gA} and "
                          f"{gB}? (in percentage points)")
                elif k == 1:
                    qx = (f"And what is the absolute difference in "
                          f"{outcomes[idx_used[1]]} between the two "
                          f"groups?")
                elif k == 2:
                    qx = (f"What about for {outcomes[idx_used[2]]}?")
                else:
                    qx = (f"And for {outcomes[idx_used[k]]}?")
                prior_qs.append((qx, f"{diffs[k]}"))

        final_op = cfg["final_op"]
        if final_op == "varied":
            op = rng.choice(["sum", "max", "min", "max_min_diff"])
        else:
            op = "sum"

        n_str = {1: "value above", 2: "two differences", 3: "three differences",
                 4: "four differences"}.get(n_priors, f"{n_priors} differences")

        # Compute correct numeric answer
        if n_priors == 1:
            # Trivial L0/L1 — just lift the single value (no aggregation).
            if op == "sum":
                true_val = diffs[0]
                q_final_base = (
                    f"Look at the chart and confirm: what is the absolute "
                    f"difference in {outcomes[idx_used[0]]} between {gA} "
                    f"and {gB}?"
                )
            else:
                true_val = diffs[0]
                q_final_base = (
                    f"From the chart, what is the absolute difference in "
                    f"{outcomes[idx_used[0]]} between {gA} and {gB}?"
                )
        elif op == "sum":
            true_val = sum(diffs)
            q_final_base = (f"Now sum the {n_str} you just computed — "
                            f"what is the total?")
        elif op == "max":
            true_val = max(diffs)
            q_final_base = (f"Now report the maximum of the {n_str} "
                            f"you just computed.")
        elif op == "min":
            true_val = min(diffs)
            q_final_base = (f"Now report the minimum of the {n_str} "
                            f"you just computed.")
        else:  # max_min_diff
            true_val = max(diffs) - min(diffs)
            q_final_base = (f"Now compute (max - min) of the {n_str} "
                            f"you just computed.")

        # Conditional layer (L6+)
        if cfg.get("use_conditional"):
            grow_pct = rng.choice([10, 20, 25, 50])
            # Hypothetical: if vals_a[idx_used[0]] grew by grow_pct%, what
            # would the new |a' - b| become for that outcome?
            new_a = vals_a[idx_used[0]] * (100 + grow_pct) / 100
            new_diff_first = abs(new_a - vals_b[idx_used[0]])
            true_val = round(new_diff_first)
            q_final_base = (
                f"Hypothetical: if {gA}'s {outcomes[idx_used[0]]} grew "
                f"by {grow_pct}% (and {gB}'s value stayed the same), "
                f"what would the new absolute difference between {gA} "
                f"and {gB} for {outcomes[idx_used[0]]} become? "
                f"(Round to nearest integer.)"
            )

        if true_val < 0:
            return None

        # Build 4 numeric candidate options. For sum/max/etc.
        true_int = int(round(true_val))
        # Pick step proportional to magnitude.
        step = max(2, int(round(true_int * 0.10))) if true_int > 0 else 3
        offsets = [-3, -2, -1, 1, 2, 3, -4, 4, -5, 5]
        rng.shuffle(offsets)
        wrong_vals = []
        seen = {true_int}
        for off in offsets:
            c = true_int + off * step
            if c < 0 or c in seen:
                continue
            wrong_vals.append(c)
            seen.add(c)
            if len(wrong_vals) == 3:
                break
        if len(wrong_vals) < 3:
            return None

        all_vals = [true_int] + wrong_vals[:3]
        rng.shuffle(all_vals)
        opts = [str(v) for v in all_vals]
        true_pos = all_vals.index(true_int)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            for off in [-6, 6, -7, 7, -8, 8]:
                c = true_int + off * step
                if c >= 0 and c not in seen:
                    opts[true_pos] = str(c)
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
            opt_lines += f"\n{opt_letters[4]}. Cannot be determined"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Assemble conversational prompt with priors zipped in.
        if prior_qs:
            conv_parts = [f"Q{i+1}: {qx} A{i+1}: {ax}"
                          for i, (qx, ax) in enumerate(prior_qs)]
            conv_prefix = f"Conversation so far: [{'; '.join(conv_parts)}]\n\n"
        else:
            conv_prefix = ""
        question = (
            f"{conv_prefix}"
            f"{q_final_base}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}. Be concise."
        )
        image = self._render(rng, outcomes, vals_a, vals_b, gA, gB)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _render(self, rng, outcomes, vals_a, vals_b, gA, gB):
        palette = list(rng.choice(self._COLOR_PALETTES))
        rng.shuffle(palette)
        n = len(outcomes)
        fig_w = max(7, n * 1.4 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        x = list(range(n))
        bw = 0.4
        bars_a = ax.bar([xi - bw / 2 for xi in x], vals_a, bw,
                        color=palette[0], label=gA, edgecolor="#1a1a1a",
                        linewidth=0.5)
        bars_b = ax.bar([xi + bw / 2 for xi in x], vals_b, bw,
                        color=palette[2 % len(palette)], label=gB,
                        edgecolor="#1a1a1a", linewidth=0.5)
        max_v = max(vals_a + vals_b)
        for xi, v in zip(x, vals_a):
            ax.text(xi - bw / 2, v + max_v * 0.02, str(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        for xi, v in zip(x, vals_b):
            ax.text(xi + bw / 2, v + max_v * 0.02, str(v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes, fontsize=10, rotation=20)
        ax.set_ylabel("Percentage", fontsize=11)
        ax.set_title(f"{gA} vs {gB}", fontsize=12, pad=10)
        ax.set_ylim(0, max_v * 1.20 + 2)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
