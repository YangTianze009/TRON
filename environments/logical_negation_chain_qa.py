"""
Logical Negation Chain QA environment (v3 planning, env #60).

Goal: track truth values through a chain of implications with optional
negation operators, rendered as a flow diagram of labelled boxes and
arrows. Targets reference deductive, VisualPuzzles
deductive and X5 (deductive symbolic reasoning).

Difficulty axes (per spec):
  A) `chain_length = 2 + level // 2`         (2..6 implications)
  B) `n_negations = level // 2`              (0..4 negation arrows)
     Also `branching = level >= 6`           (one node feeds two paths)

Format: 4-way MCQ (letter). The conclusion box is marked '?'; options
include True, False, Ambiguous, and a mis-tracked value.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PROP_NAME_POOLS = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["P", "Q", "R", "S", "T", "U", "V", "W"],
    ["X", "Y", "Z", "W", "V", "U", "T", "S"],
    ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
]

_TITLE_VARIANTS = [
    "Logic Flow Diagram",
    "Negation Chain",
    "Logical Implication Chain",
    "Truth-Value Propagation",
    "Propositional Flow",
]

_QUESTION_TEMPLATES = [
    ("Study the logic flow diagram in the image. Starting from the given fact "
     "and following the arrows (some of which negate the truth value), what "
     "is the value of the conclusion marked '?'?"),
    ("The image shows a chain of propositions connected by arrows. Some "
     "arrows carry a negation ('~') that flips the truth value. Starting "
     "from the given fact, determine the value of the '?' box."),
    ("Examine the logical flow diagram. Following the arrows and applying "
     "any negations shown, what is the truth value of the final box marked "
     "'?'?"),
    ("Trace the truth value from the initial box through the arrows in the "
     "diagram (dashed arrows indicate negation). What is the value of the "
     "box labelled '?'?"),
]

class LogicalNegationChainQA(StandaloneVisualEnv):
    ENV_NAME = "logical_negation_chain"

    # -------------------------------------------------- #
    # Level configuration (2 axes from spec)
    # -------------------------------------------------- #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        chain_length = 2 + level // 2           # 2,2,3,3,4,4,5,5,6,6
        n_negations = min(chain_length, level // 2)  # 0,0,1,1,2,2,3,3,4,4
        branching = level >= 6
        return {
            "chain_length": chain_length,
            "n_negations": n_negations,
            "branching": branching,
        }

    # -------------------------------------------------- #
    # Problem generation
    # -------------------------------------------------- #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        # Unique prime 1399 for this env
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1399)
        self._primary_complexity_feature = cfg["chain_length"]

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n = cfg["chain_length"]
        n_neg = cfg["n_negations"]
        branching = cfg["branching"]

        pool = rng.choice(_PROP_NAME_POOLS)
        if n + 1 > len(pool):
            return None
        names = pool[: n + 1]

        # Build chain: names[0] -> names[1] -> ... -> names[n]
        # Each arrow may or may not carry negation.
        # Decide which arrows negate.
        negate_positions = sorted(rng.sample(range(n), n_neg)) if n_neg > 0 else []
        negate_arrows = [i in negate_positions for i in range(n)]

        # Starting value
        start_value = rng.choice([True, False])

        # Compute truth of each node
        values = [start_value]
        for i in range(n):
            prev = values[-1]
            nxt = (not prev) if negate_arrows[i] else prev
            values.append(nxt)
        conclusion_value = values[-1]

        # Optional branching: at level >= 6, add a second branch from the
        # midpoint that merges back via AND/OR before the conclusion.
        branch_info = None
        if branching and n >= 3:
            branch_start = rng.randint(1, n - 2)
            extra_name = None
            # pick a fresh name
            remaining = [p for p in pool if p not in names]
            if remaining:
                extra_name = rng.choice(remaining)
            if extra_name is not None:
                # The branch starts from names[branch_start] with an
                # optional negation.
                branch_neg = rng.choice([True, False])
                branch_end_value = (not values[branch_start]) if branch_neg else values[branch_start]
                # Merge into the FINAL conclusion via AND or OR.
                merge_op = rng.choice(["AND", "OR"])
                # Recompute conclusion: existing conclusion_value merged
                # with branch_end_value
                if merge_op == "AND":
                    conclusion_value = values[-1] and branch_end_value
                else:
                    conclusion_value = values[-1] or branch_end_value
                branch_info = {
                    "from_idx": branch_start,
                    "extra_name": extra_name,
                    "branch_neg": branch_neg,
                    "branch_end_value": branch_end_value,
                    "merge_op": merge_op,
                }

        correct_str = "True" if conclusion_value else "False"
        wrong_str = "False" if conclusion_value else "True"
        # MCQ: T, F, Ambiguous, Missed-negation result
        flipped = "False" if conclusion_value else "True"
        distractors = [
            wrong_str,
            "Cannot be determined",
            f"Depends on the starting value",
        ]
        rng.shuffle(distractors)
        insert_idx = rng.randint(0, 3)
        options = distractors[:insert_idx] + [correct_str] + distractors[insert_idx:]
        options = options[:4]
        if correct_str not in options:
            options[0] = correct_str
        if options.count(correct_str) > 1:
            # dedupe
            seen = False
            for i, o in enumerate(options):
                if o == correct_str:
                    if seen:
                        options[i] = "Undetermined"
                    seen = True
        answer_letter = chr(ord("A") + options.index(correct_str))

        # --- Build the question text --- #
        fact_str = f"{names[0]} is {'True' if start_value else 'False'}"
        extra_msg = ""
        if branch_info is not None:
            extra_msg = (f" A side branch from {names[branch_info['from_idx']]} "
                          f"also feeds into the conclusion (merged by "
                          f"{branch_info['merge_op']}).")
        q_stem = rng.choice(_QUESTION_TEMPLATES)
        question = (
            f"{q_stem} The starting fact is: {fact_str}.{extra_msg} "
            f"Answer with a single letter."
        )

        title = rng.choice(_TITLE_VARIANTS)
        image = self._render(
            names, values, negate_arrows, branch_info,
            start_value, options, title, cfg)
        return question, answer_letter, image

    # -------------------------------------------------- #
    # Rendering — flow diagram of boxes and arrows
    # -------------------------------------------------- #

    def _render(self, names, values, negate_arrows, branch_info,
                start_value, options, title, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans"])
        base_fs = self._rng.choice([11, 12, 13])
        n = len(names)

        fig_w = 12.0 * sc
        fig_h = 8.0 * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.2)
        ax_diag = fig.add_subplot(gs[0])
        ax_opts = fig.add_subplot(gs[1])

        ax_diag.set_facecolor(style["bg_color"])
        ax_diag.axis("off")
        ax_diag.set_title(title, fontsize=base_fs + 3,
                           fontweight="bold", fontfamily=ff, pad=10)

        palette = style["palette"]

        # Box layout: horizontal chain
        box_w = 1.8
        box_h = 1.0
        spacing = 0.9
        total_w = n * box_w + (n - 1) * spacing
        y_main = 4.0
        x_start = 0.5

        positions = []
        for i in range(n):
            cx = x_start + i * (box_w + spacing) + box_w / 2
            positions.append((cx, y_main))

        # Optional branching: place branch box below the main chain
        branch_pos = None
        if branch_info is not None:
            bfrom = branch_info["from_idx"]
            # branch box is placed below the midpoint between from and
            # conclusion
            target_x = (positions[bfrom][0] + positions[-1][0]) / 2
            branch_pos = (target_x, y_main - 2.5)

        # Draw arrows between sequential boxes
        for i in range(n - 1):
            (x1, y1) = positions[i]
            (x2, y2) = positions[i + 1]
            self._draw_arrow(ax_diag, x1 + box_w / 2, y1,
                              x2 - box_w / 2, y2,
                              negate=negate_arrows[i], palette=palette)

        # Draw boxes
        for i, (cx, cy) in enumerate(positions):
            is_conclusion = (i == n - 1)
            color = palette[0] if not is_conclusion else palette[3 % len(palette)]
            box = mpatches.FancyBboxPatch(
                (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.06,rounding_size=0.18",
                facecolor="white", edgecolor=color, linewidth=2.0,
                zorder=3)
            ax_diag.add_patch(box)
            label = names[i]
            if i == 0:
                val_str = f"{label}\n= {'T' if start_value else 'F'}"
            elif is_conclusion:
                val_str = f"{label}\n= ?"
            else:
                val_str = f"{label}"
            ax_diag.text(cx, cy, val_str, ha="center", va="center",
                          fontsize=base_fs, fontfamily=ff,
                          fontweight="bold", zorder=4)

        # Draw branch box & arrows if any
        if branch_info is not None and branch_pos is not None:
            (bx, by) = branch_pos
            bname = branch_info["extra_name"]
            b_color = palette[4 % len(palette)]
            bbox = mpatches.FancyBboxPatch(
                (bx - box_w / 2, by - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.06,rounding_size=0.18",
                facecolor="white", edgecolor=b_color, linewidth=2.0,
                zorder=3)
            ax_diag.add_patch(bbox)
            ax_diag.text(bx, by, bname, ha="center", va="center",
                          fontsize=base_fs, fontfamily=ff,
                          fontweight="bold", zorder=4)
            # arrow from branch source down to branch box
            (sx, sy) = positions[branch_info["from_idx"]]
            self._draw_arrow(ax_diag, sx, sy - box_h / 2,
                              bx, by + box_h / 2,
                              negate=branch_info["branch_neg"],
                              palette=palette)
            # arrow from branch box to conclusion (with merge_op label)
            (cx, cy) = positions[-1]
            self._draw_arrow(ax_diag, bx + box_w / 2, by,
                              cx - box_w / 2, cy - box_h / 2 - 0.05,
                              negate=False, palette=palette,
                              label=branch_info["merge_op"])

        # Legend
        ax_diag.text(0.5, 1.2,
                      "Solid arrow = same value | Dashed arrow with '~' = negate",
                      fontsize=base_fs - 1, fontfamily=ff, color="#34495e",
                      ha="left")
        ax_diag.set_xlim(-0.5, total_w + 1.5)
        ax_diag.set_ylim(0, 7.0)

        # --- Options panel ---
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, 10)
        ax_opts.set_ylim(0, 10)
        ax_opts.axis("off")
        ax_opts.set_title(
            "What is the value of the '?' box?",
            fontsize=base_fs + 2, fontweight="bold",
            fontfamily=ff, pad=10)
        y = 8.0
        dy = 1.1
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            ax_opts.text(0.4, y, f"({letter}) {opt}",
                          fontsize=base_fs + 1, ha="left", va="top",
                          fontfamily=ff, color="#1a1a1a")
            y -= dy

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                             wspace=0.2)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_arrow(self, ax, x1, y1, x2, y2, negate: bool,
                     palette, label: str = None):
        line_style = "--" if negate else "-"
        color = "#c0392b" if negate else "#2c3e50"
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", linestyle=line_style,
                             color=color, linewidth=2.0,
                             mutation_scale=16), zorder=2)
        if negate:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.18, "~",
                     fontsize=16, fontweight="bold",
                     color=color, ha="center", va="center",
                     zorder=5,
                     bbox=dict(boxstyle="circle,pad=0.15",
                               facecolor="white", edgecolor=color))
        if label is not None:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.25, label,
                     fontsize=11, fontweight="bold",
                     color="#2c3e50", ha="center", va="center",
                     zorder=5,
                     bbox=dict(boxstyle="round,pad=0.15",
                               facecolor="white", edgecolor="#555"))

# ---------------------------------------------------------------------- #
# Local smoke test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import collections
    env = LogicalNegationChainQA()
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
            e = LogicalNegationChainQA()
            if e.generate(seed=s * 1000 + level * 37 + 17,
                          parameter={"level": level}):
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
