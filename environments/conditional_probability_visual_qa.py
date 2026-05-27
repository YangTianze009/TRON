"""
Conditional Probability Visual QA environment (v3 planning, env #62).

Goal: compute a conditional probability P(A=a | B=b) from a joint frequency
table rendered in the image. Targets statistics, dynamic-math
statistics, and M4 (probabilistic reasoning) + X5 (deductive reasoning).

Difficulty axes (per spec):
  A) `table_size`: L0=2x2, L5=2x3, L9=3x3
  B) `distractor_type`: L0=unrelated fractions, L5=P(B|A) swap, L9=all four
     of P(A|B), P(B|A), P(A,B), P(A). Also `count_simplicity`: L0=round
     numbers, L9=prime-heavy.

Format: 4-way MCQ (letter) — options are decimals rounded to 3 places.
"""
import random
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]

_ROW_POOLS = [
    ("Colour", ["red", "blue", "green"]),
    ("Size", ["small", "medium", "large"]),
    ("Group", ["alpha", "beta", "gamma"]),
    ("Type", ["A", "B", "C"]),
    ("Condition", ["low", "medium", "high"]),
]

_COL_POOLS = [
    ("Outcome", ["pass", "fail"]),
    ("Result", ["win", "lose"]),
    ("Label", ["X", "Y"]),
    ("Status", ["on", "off"]),
    ("Answer", ["yes", "no", "maybe"]),
    ("Level", ["L1", "L2", "L3"]),
]

_TITLE_VARIANTS = [
    "Joint Frequency Table",
    "Contingency Table",
    "Joint Distribution",
    "Cross-Tabulation",
    "Frequency Table",
]

class ConditionalProbabilityVisualQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "conditional_probability_visual"

    # -------------------------------------------------- #
    # Level configuration (2 axes from spec)
    # -------------------------------------------------- #

    # 2026-05-04 R4: full-gradient redesign. Each level adds a new dimension:
    #   table_size, count_mode, distractor_mode, target_ptype, show_formula.
    #   - L0: 2x2 round + formula + unrelated distractors (trivial)
    #   - L1: 2x2 round + formula + simple distractors
    #   - L2: 2x2 mixed + formula + swap distractor (intro reverse-cond trap)
    #   - L3: 2x3 mixed + formula
    #   - L4: 2x3 mixed, no formula (must remember definition)
    #   - L5: 2x3 mixed, swap+joint distractor (multi-trap)
    #   - L6: 2x3 prime, all-four distractor
    #   - L7: 3x3 prime, all-four + 1 decoy table
    #   - L8: 3x3 prime, joint_or_reverse, near_miss distractor
    #   - L9: 3x3 prime, total-prob Bayes (compound multi-step)
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            cfg = dict(table_size=(2, 2), distractor_mode="unrelated",
                       count_mode="round10", target_ptype="simple_cond",
                       show_formula=True, n_decoy_tables=0)
        elif level == 1:
            cfg = dict(table_size=(2, 2), distractor_mode="unrelated",
                       count_mode="round10", target_ptype="simple_cond",
                       show_formula=True, n_decoy_tables=0)
        elif level == 2:
            cfg = dict(table_size=(2, 2), distractor_mode="swap",
                       count_mode="mixed", target_ptype="simple_cond",
                       show_formula=True, n_decoy_tables=0)
        elif level == 3:
            cfg = dict(table_size=(2, 3), distractor_mode="swap",
                       count_mode="mixed", target_ptype="simple_cond",
                       show_formula=True, n_decoy_tables=0)
        elif level == 4:
            cfg = dict(table_size=(2, 3), distractor_mode="swap_plus_joint",
                       count_mode="mixed", target_ptype="simple_cond",
                       show_formula=False, n_decoy_tables=0)
        elif level == 5:
            cfg = dict(table_size=(2, 3), distractor_mode="swap_plus_joint",
                       count_mode="mixed", target_ptype="joint_or_reverse",
                       show_formula=False, n_decoy_tables=0)
        elif level == 6:
            cfg = dict(table_size=(2, 3), distractor_mode="all_four",
                       count_mode="prime", target_ptype="joint_or_reverse",
                       show_formula=False, n_decoy_tables=0)
        elif level == 7:
            cfg = dict(table_size=(3, 3), distractor_mode="all_four",
                       count_mode="prime", target_ptype="joint_or_reverse",
                       show_formula=False, n_decoy_tables=1)
        elif level == 8:
            cfg = dict(table_size=(3, 3), distractor_mode="near_miss",
                       count_mode="prime", target_ptype="joint_or_reverse",
                       show_formula=False, n_decoy_tables=1)
        else:
            cfg = dict(table_size=(3, 3), distractor_mode="near_miss",
                       count_mode="prime",
                       target_ptype="total_prob_bayes_three_level",
                       show_formula=False, n_decoy_tables=1)
        return cfg

    # -------------------------------------------------- #
    # Problem generation
    # -------------------------------------------------- #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        # Unique prime 1423 for this env
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1423)
        self._primary_complexity_feature = cfg["table_size"][0] * cfg["table_size"][1]

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        self._bayes_hide_totals = False
        self._three_level_alpha = None
        nr, nc = cfg["table_size"]
        # Select row/column metadata
        _r = [p for p in _ROW_POOLS if len(p[1]) >= nr]
        _c = [p for p in _COL_POOLS if len(p[1]) >= nc]
        if not _r or not _c:
            return None
        r_meta = rng.choice(_r)
        c_meta = rng.choice(_c)
        row_label, row_vals = r_meta[0], r_meta[1][:nr]
        col_label, col_vals = c_meta[0], c_meta[1][:nc]

        # Counts
        count_mode = cfg["count_mode"]
        data = []
        for r in range(nr):
            row = []
            for c in range(nc):
                if count_mode == "round10":
                    v = rng.choice([10, 20, 30, 40, 50, 60])
                elif count_mode == "mixed":
                    v = rng.randint(5, 40)
                else:
                    v = rng.choice(_PRIMES)
                row.append(v)
            data.append(row)

        row_totals = [sum(r) for r in data]
        col_totals = [sum(data[r][c] for r in range(nr)) for c in range(nc)]
        grand_total = sum(row_totals)
        if grand_total == 0:
            return None

        # Pick the target cell (row r0, col c0) — answer P(row | col)
        # i.e. P(A=row | B=col) = cell / col_total
        r0 = rng.randrange(nr)
        c0 = rng.randrange(nc)
        if col_totals[c0] == 0 or row_totals[r0] == 0:
            return None

        cond_rc = Fraction(data[r0][c0], col_totals[c0])  # P(row | col)
        cond_cr = Fraction(data[r0][c0], row_totals[r0])  # P(col | row)
        joint = Fraction(data[r0][c0], grand_total)        # P(row, col)
        marg_row = Fraction(row_totals[r0], grand_total)   # P(row)
        marg_col = Fraction(col_totals[c0], grand_total)   # P(col)

        # At higher levels, structural twist: ask for joint or reverse cond,
        # or chained (product/diff). Question text still refers to labels.
        target_ptype = cfg.get("target_ptype", "simple_cond")
        q_kind = "cond_rc"
        correct_prob = cond_rc
        if target_ptype == "joint_or_reverse":
            q_kind = rng.choice(["cond_rc", "cond_cr", "joint"])
            correct_prob = {"cond_rc": cond_rc, "cond_cr": cond_cr,
                            "joint": joint}[q_kind]
        elif target_ptype == "total_prob_bayes_three_level":
            # L9 iter-3: a pre-event X with two levels, each giving its own
            # conditional P(row|col). Student must compute
            #   P(row|col) = P(X=1|col) * P(row|col,X=1)
            #              + P(X=0|col) * P(row|col,X=0)
            # The image shows a single table; a side-note lists the
            # "Stage-2 weights" for X: e.g., "when col=A, P(X=1)=0.6; when
            # col=B, P(X=1)=0.4; ..." and the CELL COUNTS represent
            # P(row|col,X=1) scaled by col. To make this soluble from the
            # image alone we use this simplified form: answer equals
            #   P(row|col) computed from the (hidden-total) table,
            #   TIMES a "mixing coefficient" α printed on the image,
            #   PLUS (1-α) * P(row).   (law of total probability)
            # We hide totals so the student computes all marginals.
            alpha = rng.choice([Fraction(1, 4), Fraction(3, 8),
                                Fraction(5, 8), Fraction(3, 4)])
            self._three_level_alpha = alpha
            self._bayes_hide_totals = True
            q_kind = "total_prob"
            correct_prob = alpha * cond_rc + (Fraction(1, 1) - alpha) * marg_row
        elif target_ptype == "bayes_inverted":
            # P(B|A) via Bayes: P(B|A) = P(A|B) * P(B) / P(A). We pose as
            # "compute P(col|row) via Bayes" but the student must compute
            # marginals themselves since totals are hidden. Choose whether
            # to ask the direct cell-based answer or a diff/ratio.
            q_kind = rng.choice(["bayes_cond_cr", "bayes_cond_diff",
                                 "cond_rc"])
            if q_kind == "bayes_cond_cr":
                # We literally return cond_cr, but we need to HIDE the row
                # totals on the image so the student has to compute them.
                correct_prob = cond_cr
            elif q_kind == "bayes_cond_diff":
                # P(B|A) - P(B): tests Bayes + marginal.
                correct_prob = cond_cr - marg_col
            else:
                correct_prob = cond_rc
            # Mark bayes so rendering hides totals.
            self._bayes_hide_totals = True
        elif target_ptype == "chained":
            q_kind = rng.choice(["cond_diff", "cond_ratio",
                                 "cond_rc", "cond_cr"])
            if q_kind == "cond_diff":
                # P(row|col) - P(row)
                correct_prob = cond_rc - marg_row
            elif q_kind == "cond_ratio":
                # P(row|col) / P(row)  (an independence-check ratio)
                if float(marg_row) < 1e-6:
                    return None
                correct_prob = Fraction(
                    int(round(float(cond_rc) * 1000000)),
                    max(1, int(round(float(marg_row) * 1000000))))
            elif q_kind == "cond_rc":
                correct_prob = cond_rc
            else:
                correct_prob = cond_cr
        correct_val = round(float(correct_prob), 3)
        if not (-1.5 <= correct_val <= 1.5):
            return None

        # Build distractors
        distractor_mode = cfg["distractor_mode"]
        distractor_values = []

        def _add(x):
            v = round(float(x), 3)
            if v != correct_val and v not in distractor_values:
                distractor_values.append(v)

        if distractor_mode == "unrelated":
            # Unrelated fractions near the correct value
            base = float(correct_prob)
            candidates = [base + 0.1, base - 0.15, base + 0.2,
                           0.25, 0.5, 0.75, base / 2]
            for c in candidates:
                cv = round(c, 3)
                if cv != correct_val:
                    _add(Fraction(int(round(cv * 1000)), 1000))
                if len(distractor_values) >= 3:
                    break
        elif distractor_mode == "swap":
            # Swap P(A|B) -> P(B|A)
            _add(cond_cr)
            # plus one joint, one marginal
            _add(joint)
            _add(marg_row)
            # pad if needed
            for base_delta in [0.1, -0.1, 0.05, -0.05]:
                if len(distractor_values) >= 3:
                    break
                cv = round(float(cond_rc) + base_delta, 3)
                if 0 < cv < 1:
                    _add(Fraction(int(round(cv * 1000)), 1000))
        elif distractor_mode == "swap_plus_joint":
            _add(cond_cr)
            _add(joint)
            _add(marg_col)
            for base_delta in [0.07, -0.08, 0.04]:
                if len(distractor_values) >= 3:
                    break
                cv = round(float(cond_rc) + base_delta, 3)
                if 0 < cv < 1:
                    _add(Fraction(int(round(cv * 1000)), 1000))
        elif distractor_mode == "all_four":
            _add(cond_cr)
            _add(joint)
            _add(marg_row)
            _add(marg_col)
            # keep first 3
        else:  # near_miss — L9 iter-4: tight distractors that match
            # semantically-plausible wrong answers for the total-probability
            # setup (α swap, α complement, reversed conditional, etc.).
            base = float(correct_prob)
            alpha_val = getattr(self, "_three_level_alpha", None)
            if alpha_val is not None:
                alpha_f = float(alpha_val)
                # Wrong: used (1-α) as the weight for cond.
                wrong1 = (1 - alpha_f) * float(cond_rc) + alpha_f * float(marg_row)
                _add(Fraction(int(round(wrong1 * 1000000)), 1000000))
                # Wrong: mixed with marg_col instead of marg_row.
                wrong2 = alpha_f * float(cond_rc) + (1 - alpha_f) * float(marg_col)
                _add(Fraction(int(round(wrong2 * 1000000)), 1000000))
                # Wrong: used cond_cr instead of cond_rc.
                wrong3 = alpha_f * float(cond_cr) + (1 - alpha_f) * float(marg_row)
                _add(Fraction(int(round(wrong3 * 1000000)), 1000000))
            cc = float(cond_cr)
            if abs(cc - base) < 0.12 and cc != base:
                _add(cond_cr)
            # Also keep the simple ±δ distractors — tighter (0.02-0.04).
            for delta in [-0.02, 0.02, -0.04, 0.04, -0.03, 0.03]:
                if len(distractor_values) >= 3:
                    break
                cv = round(base + delta, 3)
                if cv != correct_val and cv not in distractor_values:
                    distractor_values.append(cv)
        distractor_values = distractor_values[:3]
        while len(distractor_values) < 3:
            base_delta = rng.choice([0.03, -0.03, 0.05, -0.05, 0.08])
            cv = round(float(correct_prob) + base_delta, 3)
            if cv != correct_val and cv not in distractor_values:
                distractor_values.append(cv)
            else:
                r_fallback = round(rng.uniform(-0.5, 1.5), 3)
                if r_fallback != correct_val and r_fallback not in distractor_values:
                    distractor_values.append(r_fallback)

        rng.shuffle(distractor_values)
        distractor_strs = [f"{v:.3f}" for v in distractor_values]
        correct_str = f"{correct_val:.3f}"

        # 2026-05-05 R5 B3B HARDEN: 4-MCQ → 5-MCQ. Slot E is fixed
        # "No correct answer". L8/L9 e_trap rate 40%; L6/L7 = 25%; ≤L5 = 0%.
        # 2026-05-05 Phase A: bump L8/L9 to 50% (env still saturated 97.5%).
        if level >= 8:
            e_trap_rate = 0.50
        elif level >= 6:
            e_trap_rate = 0.25
        else:
            e_trap_rate = 0.0
        use_etrap = (rng.random() < e_trap_rate) if e_trap_rate > 0.0 else False

        if use_etrap:
            # All 4 visible options must be wrong. Need 4 distractor strings.
            wrong_pool = [s for s in distractor_strs if s != correct_str]
            # Pad with deltas if needed
            extras = [round(correct_val + 0.06, 3),
                      round(correct_val - 0.06, 3),
                      round(correct_val + 0.09, 3),
                      round(correct_val - 0.09, 3),
                      round(correct_val + 0.12, 3),
                      round(correct_val - 0.12, 3)]
            for ex in extras:
                ex_s = f"{ex:.3f}"
                if ex_s != correct_str and ex_s not in wrong_pool:
                    wrong_pool.append(ex_s)
                if len(wrong_pool) >= 4:
                    break
            if len(wrong_pool) < 4:
                return None
            options = wrong_pool[:4] + ["No correct answer"]
            answer_letter = "E"
        else:
            insert_idx = rng.randint(0, 3)
            options = distractor_strs[:insert_idx] + [correct_str] + distractor_strs[insert_idx:]
            options = options[:4]
            if correct_str not in options:
                options[0] = correct_str
            if options.count(correct_str) > 1:
                seen = False
                for i, o in enumerate(options):
                    if o == correct_str:
                        if seen:
                            options[i] = f"{round(correct_val + 0.07, 3):.3f}"
                        seen = True
            options.append("No correct answer")
            answer_letter = chr(ord("A") + options.index(correct_str))

        title = rng.choice(_TITLE_VARIANTS)

        # Build decoy tables for L6-L9 (same schema with random counts).
        n_decoy_tables = cfg.get("n_decoy_tables", 0)
        decoy_tables = []
        target_table_label = None
        if n_decoy_tables > 0:
            # Generate N extra tables that share row/col labels but have
            # different counts. Choose a unique label for the target table.
            labels = ["Table I", "Table II", "Table III", "Table IV"]
            rng.shuffle(labels)
            target_table_label = labels[0]
            for i in range(n_decoy_tables):
                d_data = []
                for rr in range(nr):
                    drow = []
                    for cc in range(nc):
                        if cfg["count_mode"] == "round10":
                            dv = rng.choice([10, 20, 30, 40, 50, 60])
                        elif cfg["count_mode"] == "mixed":
                            dv = rng.randint(5, 40)
                        else:
                            dv = rng.choice(_PRIMES)
                        drow.append(dv)
                    d_data.append(drow)
                d_row_tot = [sum(r) for r in d_data]
                d_col_tot = [sum(d_data[r][c] for r in range(nr))
                             for c in range(nc)]
                d_grand = sum(d_row_tot)
                decoy_tables.append({
                    "data": d_data,
                    "row_totals": d_row_tot,
                    "col_totals": d_col_tot,
                    "grand_total": d_grand,
                    "label": labels[i + 1],
                })

        image = self._render_table(
            row_label, row_vals, col_label, col_vals,
            data, row_totals, col_totals, grand_total,
            options, title, cfg.get("show_formula", False),
            decoy_tables=decoy_tables,
            target_label=target_table_label,
            hide_totals=getattr(self, "_bayes_hide_totals", False))

        # Question text per q_kind
        r_expr = f"{row_label}={row_vals[r0]}"
        c_expr = f"{col_label}={col_vals[c0]}"
        if q_kind == "cond_rc":
            q_templates = [
                f"Using the joint frequency table in the image, what is P({r_expr} | {c_expr})? Round to 3 decimal places.",
                f"From the contingency table shown, compute the conditional probability P({r_expr} | {c_expr}). Round to 3 decimal places.",
                f"The image shows counts for each combination of {row_label} and {col_label}. Given {c_expr}, what is the probability that {r_expr}? Round to 3 decimals.",
                f"Consult the frequency table in the image. What is the value of P({r_expr} | {c_expr})? Round to 3 decimal places.",
            ]
        elif q_kind == "cond_cr":
            q_templates = [
                f"Using the table in the image, what is P({c_expr} | {r_expr})? Round to 3 decimal places.",
                f"Compute the conditional probability P({c_expr} | {r_expr}) from the table. Round to 3 decimals.",
                f"Given {r_expr}, what is the probability that {c_expr}? Round to 3 decimals.",
                f"Find P({c_expr} | {r_expr}) using the figure. Round to 3 decimals.",
            ]
        elif q_kind == "joint":
            q_templates = [
                f"Using the table in the image, compute the joint probability P({r_expr}, {c_expr}). Round to 3 decimal places.",
                f"What fraction of the grand total is the cell ({r_expr}, {c_expr})? Round to 3 decimals.",
                f"From the figure, compute P({r_expr} AND {c_expr}). Round to 3 decimals.",
                f"Find the joint probability P({r_expr}, {c_expr}) in the figure. Round to 3 decimals.",
            ]
        elif q_kind == "cond_diff":
            q_templates = [
                f"Using the table in the image, compute P({r_expr} | {c_expr}) − P({r_expr}). Round to 3 decimal places.",
                f"Find the difference P({r_expr} | {c_expr}) − P({r_expr}) using the table. Round to 3 decimals.",
                f"What is [P({r_expr} | {c_expr}) minus P({r_expr})]? Round to 3 decimals.",
                f"Compute the conditional−marginal difference for {r_expr} given {c_expr}. Round to 3 decimals.",
            ]
        elif q_kind == "cond_ratio":
            q_templates = [
                f"Using the table in the image, compute P({r_expr} | {c_expr}) / P({r_expr}). Round to 3 decimal places.",
                f"Find the ratio P({r_expr} | {c_expr}) / P({r_expr}) (this equals 1 when {row_label} and {col_label} are independent). Round to 3 decimals.",
                f"From the table, compute the lift = P({r_expr} | {c_expr}) / P({r_expr}). Round to 3 decimals.",
                f"Compute the ratio of conditional to marginal probability for {r_expr} given {c_expr}. Round to 3 decimals.",
            ]
        elif q_kind == "bayes_cond_cr":
            q_templates = [
                f"ROW and COLUMN totals are NOT printed on the table — you must compute them from the cell counts yourself. Using Bayes' relation, find P({c_expr} | {r_expr}). Round to 3 decimals.",
                f"The table shows cell counts only (no marginal totals). Compute P({c_expr} | {r_expr}) by first computing the row total for {r_expr}. Round to 3 decimals.",
                f"No marginal totals are given. Reconstruct them from cell counts, then compute P({c_expr} | {r_expr}). Round to 3 decimals.",
            ]
        elif q_kind == "bayes_cond_diff":
            q_templates = [
                f"Row/column totals are hidden; reconstruct them from cells. Then compute P({c_expr} | {r_expr}) − P({c_expr}). Round to 3 decimals.",
                f"Using only cell counts (marginals are NOT shown), compute P({c_expr} | {r_expr}) minus P({c_expr}). Round to 3 decimals.",
            ]
        elif q_kind == "total_prob":
            alpha = getattr(self, "_three_level_alpha", Fraction(1, 2))
            alpha_str = f"{float(alpha):.3f}"
            q_templates = [
                (f"Use the LAW OF TOTAL PROBABILITY with mixing coefficient "
                 f"α = {alpha_str} (printed on the image). Compute "
                 f"α · P({r_expr} | {c_expr}) + (1 − α) · P({r_expr}). "
                 f"Row/column totals are HIDDEN — reconstruct them from the "
                 f"cell counts first. Round to 3 decimals."),
                (f"A mixing coefficient α = {alpha_str} is given. Compute "
                 f"the linear combination α · P({r_expr} | {c_expr}) + "
                 f"(1 − α) · P({r_expr}). No marginal totals are printed; "
                 f"you must sum the cell counts to get them. Round to 3 "
                 f"decimals."),
                (f"The weighting α = {alpha_str} is shown. Return "
                 f"α · P({r_expr} | {c_expr}) + (1 − α) · P({r_expr}). "
                 f"Reconstruct row and column totals from cells (they are "
                 f"NOT drawn on the table). Round to 3 decimals."),
            ]
        else:
            q_templates = [
                f"Using the table, find the indicated probability. Round to 3 decimals."]
        question = (f"{rng.choice(q_templates)} Answer with a single letter "
                    f"(A, B, C, D, or E). Note: option E means 'no correct "
                    f"answer is shown — none of A-D match the true value'.")
        if target_table_label is not None:
            question = question + (
                f" IMPORTANT: the image shows MULTIPLE tables; use ONLY the "
                f"one labeled '{target_table_label}' (highlighted in the "
                f"image) — the others are decoys with different numbers.")
        return question, answer_letter, image

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _draw_one_table(self, ax, row_label, row_vals, col_label, col_vals,
                         data, row_totals, col_totals, grand_total,
                         label, ff, base_fs, palette, style,
                         highlight=False, hide_totals=False):
        """Render a single contingency table inside the given axes.

        When `hide_totals` is True (L9 Bayes mode), Row/Col total cells are
        rendered as "?" so the student must compute marginals themselves.
        """
        nr = len(row_vals)
        nc = len(col_vals)
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")
        title_color = "#b71c1c" if highlight else "#2c3e50"
        title_weight = "bold"
        ax.set_title(label, fontsize=base_fs + 3,
                     fontweight=title_weight,
                     fontfamily=ff, pad=12, color=title_color)

        full_col_labels = [f"{col_label}={v}" for v in col_vals] + ["Total"]
        full_row_labels = [f"{row_label}={v}" for v in row_vals] + ["Total"]

        cell_text = []
        for r in range(nr):
            if hide_totals:
                row = [str(data[r][c]) for c in range(nc)] + ["?"]
            else:
                row = [str(data[r][c]) for c in range(nc)] + [str(row_totals[r])]
            cell_text.append(row)
        if hide_totals:
            total_row = ["?"] * nc + ["?"]
        else:
            total_row = [str(col_totals[c]) for c in range(nc)] + [str(grand_total)]
        cell_text.append(total_row)

        table = ax.table(
            cellText=cell_text,
            rowLabels=full_row_labels,
            colLabels=full_col_labels,
            cellLoc="center", rowLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(base_fs + 1)
        table.scale(1.0, 1.7)

        header_color = palette[0]
        row_label_color = palette[1 % len(palette)]
        total_bg = palette[2 % len(palette)]
        try:
            from matplotlib.colors import to_rgba
            rr_, gg_, bb_, _ = to_rgba(total_bg)
            total_bg_light = (
                f"#{int((rr_ * 0.3 + 0.7) * 255):02x}"
                f"{int((gg_ * 0.3 + 0.7) * 255):02x}"
                f"{int((bb_ * 0.3 + 0.7) * 255):02x}"
            )
        except Exception:
            total_bg_light = "#d4e6f1"

        for (row, col), cell in table.get_celld().items():
            if highlight:
                cell.set_edgecolor("#b71c1c")
                cell.set_linewidth(2.0 if (row == 0 or col == -1) else 1.5)
            else:
                cell.set_edgecolor("#7f8c8d")
                cell.set_linewidth(1.0)
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color="white", fontweight="bold",
                                     fontsize=base_fs + 1, fontfamily=ff)
            elif col == -1:
                cell.set_facecolor(row_label_color)
                cell.set_text_props(color="white", fontweight="bold",
                                     fontsize=base_fs, fontfamily=ff)
            elif row == nr + 1 or col == nc:
                cell.set_facecolor(total_bg_light)
                cell.set_text_props(fontweight="bold", fontsize=base_fs + 1,
                                     fontfamily=ff)
            else:
                cell.set_facecolor("#fdfefe" if row % 2 == 0 else "#ecf0f1")
                cell.set_text_props(fontsize=base_fs + 1, fontfamily=ff)

    def _render_table(self, row_label, row_vals, col_label, col_vals,
                       data, row_totals, col_totals, grand_total,
                       options, title,
                       show_formula: bool = False,
                       decoy_tables=None,
                       target_label=None,
                       hide_totals: bool = False) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans", "monospace"])
        base_fs = self._rng.choice([11, 12, 13])
        palette = style["palette"]

        nr = len(row_vals)
        nc = len(col_vals)

        decoy_tables = decoy_tables or []
        n_tables_total = 1 + len(decoy_tables)

        # Multi-table layout: tables vertically stacked on left, options on right.
        fig_w = (8.0 + 1.2 * nc) * sc
        fig_h = (7.0 + 3.0 * (n_tables_total - 1)) * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(max(1, n_tables_total), 2,
                              width_ratios=[1.4, 1.0], wspace=0.25)
        ax_opts = fig.add_subplot(gs[:, 1])
        if n_tables_total == 1:
            ax_table = fig.add_subplot(gs[0, 0])
        else:
            # Decide ordering: target table placed at a random vertical slot
            target_slot = self._rng.randint(0, n_tables_total - 1)
            ax_table = fig.add_subplot(gs[target_slot, 0])
            # Draw decoy tables in other slots
            decoy_axes = []
            for si in range(n_tables_total):
                if si == target_slot:
                    continue
                decoy_axes.append(fig.add_subplot(gs[si, 0]))
            # Render decoy tables
            for dax, dt in zip(decoy_axes, decoy_tables):
                self._draw_one_table(dax, row_label, row_vals, col_label,
                                     col_vals, dt["data"], dt["row_totals"],
                                     dt["col_totals"], dt["grand_total"],
                                     dt["label"], ff, base_fs, palette,
                                     style, highlight=False,
                                     hide_totals=hide_totals)

        # Draw the main target table
        main_table_label = target_label if target_label else title
        self._draw_one_table(ax_table, row_label, row_vals, col_label,
                             col_vals, data, row_totals, col_totals,
                             grand_total, main_table_label, ff, base_fs,
                             palette, style,
                             highlight=(target_label is not None),
                             hide_totals=hide_totals)

        # --- Options panel ---
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, 10)
        ax_opts.set_ylim(0, 10)
        ax_opts.axis("off")
        ax_opts.set_title("Choose the correct probability:",
                           fontsize=base_fs + 2, fontweight="bold",
                           fontfamily=ff, pad=10)
        y = 8.0
        dy = 1.2
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            ax_opts.text(0.4, y, f"({letter}) {opt}",
                          fontsize=base_fs + 1, ha="left", va="top",
                          fontfamily=ff, color="#1a1a1a")
            y -= dy
        # Optional formula hint (L0-L3)
        if show_formula:
            y -= 0.4
            ax_opts.text(0.4, y, "Hint: P(A|B) = count(A,B) / count(B)",
                          fontsize=base_fs, ha="left", va="top",
                          fontfamily=ff, color="#555", style="italic")
        # Print alpha (three-level / total-prob mode) in a red-bordered box
        # so the student sees it is a required input.
        alpha = getattr(self, "_three_level_alpha", None)
        if alpha is not None:
            y -= 0.6
            ax_opts.text(0.4, y,
                          f"MIXING COEFFICIENT  α = {float(alpha):.3f}",
                          fontsize=base_fs + 1, ha="left", va="top",
                          fontfamily=ff, fontweight="bold", color="#b71c1c",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="#fff3cd",
                                    edgecolor="#b71c1c", linewidth=1.5))

        fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.05,
                             wspace=0.25)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Local smoke test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import collections
    env = ConditionalProbabilityVisualQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED")
                continue
            print(f"[seed={seed} L{level}] A={env._answer}  "
                  f"Q={env.get_instruction()[:80]}")
    for level in (0, 3, 6, 9):
        letters = collections.Counter()
        for s in range(20):
            e = ConditionalProbabilityVisualQA()
            if e.generate(seed=s * 1000 + level * 37 + 17,
                          parameter={"level": level}):
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
