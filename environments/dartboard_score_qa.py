"""
Dartboard Score QA.

2026-05-05 R5 B5A: REWRITTEN — pure single-letter MCQ A-D / A-E.

Renders a dartboard with concentric rings (each ring labeled with its
point value), and several darts (red dots) hitting various rings. The
model must compute the total score and pick the matching MCQ option.

Per-level layout
  L0/L1 — 4-MCQ. 1-2 darts on a 5-ring "easy" board (values 1/5/10/20/50).
          Big figure, large ring labels, large dart markers. Concise hint.
  L2/L3 — 4-MCQ. 2-3 darts on the easy board. Distractor sums close.
  L4/L5 — 5-MCQ + 15% trap. 3 darts, easy board. "E. No correct answer".
  L6/L7 — 5-MCQ + 25% trap. 3-4 darts, easy or medium board.
  L8/L9 — 5-MCQ + 30/40% trap. 4-5 darts, medium or hard pool.

Distractors are constructed by:
  (a) off-by-one ring substitutions
  (b) sum/concat tricks (sum of (n-1) darts, sum + extra ring value)
  (c) common arithmetic slips (10x, double, half).

ALLOW_ROTATION = False — rotation reorders ring positions visually
relative to dart marks and breaks the score reading.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Ring point values, from outside in (so values[0] = outermost)
_RING_VALUES = {
    "easy":   [1, 5, 10, 20, 50],
    "medium": [1, 3, 7, 15, 25, 50],
    "hard":   [2, 4, 8, 16, 32, 64],
}


_STEM_TEMPLATES = [
    "The result of a shooter on a dartboard is shown in the image. Each red dot is a dart that hit the board. What is the total score ( )?",
    "A dartboard with concentric rings labeled by point value is shown. The red dots indicate where the darts landed. What is the sum of the scores ( )?",
    "The image shows a dartboard. Each red dot is a dart hitting one of the labeled rings. Calculate the total score ( ).",
    "Look at the dartboard. Each ring is labeled with its score, and the red dots show where the darts landed. What is the total score ( )?",
]


class DartboardScoreQA(StandaloneVisualEnv):
    ENV_NAME = "dartboard_score"
    ALLOW_ROTATION = False  # rotating board reorders ring positions visually

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. For each red dot, read the ring label. Sum them. "
        "Final answer in the format this prompt requested."
    )

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"value_pool": "easy", "n_darts": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 1:
            return {"value_pool": "easy", "n_darts": 2,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 3:
            return {"value_pool": "easy", "n_darts": 2 + (level - 2),
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 5:
            return {"value_pool": "easy", "n_darts": 3,
                    "use_5_options": True, "trap_rate": 0.15}
        if level == 6:
            return {"value_pool": "easy", "n_darts": 4,
                    "use_5_options": True, "trap_rate": 0.20}
        if level == 7:
            return {"value_pool": "medium", "n_darts": 4,
                    "use_5_options": True, "trap_rate": 0.25}
        if level == 8:
            return {"value_pool": "medium", "n_darts": 5,
                    "use_5_options": True, "trap_rate": 0.30}
        # L9
        return {"value_pool": "hard", "n_darts": 5,
                "use_5_options": True, "trap_rate": 0.40}

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5851 + level * 71 + 19)

        for _ in range(20):
            res = self._try_generate(cfg, level, rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        # values[0] = innermost ring (highest score for easy/medium/hard pools).
        # The original env used reversed(); we keep that convention.
        ring_values = list(reversed(_RING_VALUES[cfg["value_pool"]]))
        n_rings = len(ring_values)

        darts: List[Tuple[float, float, int]] = []
        total_score = 0
        for _ in range(cfg["n_darts"]):
            ring_idx = rng.randint(0, n_rings - 1)
            theta = rng.uniform(0, 2 * math.pi)
            r_inner, r_outer = ring_idx, ring_idx + 1
            r = r_inner + (r_outer - r_inner) * rng.uniform(0.25, 0.75)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            darts.append((x, y, ring_idx))
            total_score += ring_values[ring_idx]

        # Build distractor sums
        all_values_set = set(ring_values)

        def _candidates() -> List[int]:
            cands: List[int] = []
            # Off-by-one ring substitution (one dart shifted up/down 1 ring)
            for i in range(len(darts)):
                for shift in (-1, 1):
                    new_idx = darts[i][2] + shift
                    if 0 <= new_idx < n_rings:
                        s = total_score - ring_values[darts[i][2]] + ring_values[new_idx]
                        if s != total_score and s > 0:
                            cands.append(s)
            # Drop a dart
            for i in range(len(darts)):
                s = total_score - ring_values[darts[i][2]]
                if s != total_score and s > 0:
                    cands.append(s)
            # Add a dart of a random other value
            for v in ring_values:
                s = total_score + v
                if s != total_score:
                    cands.append(s)
            # Common slip: double / half
            cands.append(2 * total_score)
            cands.append(max(1, total_score // 2))
            cands.append(total_score + 1)
            cands.append(max(1, total_score - 1))
            return cands

        raw_cands = _candidates()
        # Dedupe & remove correct
        uniq = []
        seen = {total_score}
        for c in raw_cands:
            if c <= 0:
                continue
            if c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        rng.shuffle(uniq)

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])

        if use_trap:
            chosen = uniq[:n_value]
            if len(chosen) < n_value:
                return None
            options_int = list(chosen)
            rng.shuffle(options_int)
            correct_letter = opt_letters[-1]
            opt_lines = [f"{opt_letters[i]}. {options_int[i]}" for i in range(n_value)]
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        else:
            n_distractor = n_value - 1
            chosen = uniq[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options_int = [total_score] + chosen
            rng.shuffle(options_int)
            correct_letter = opt_letters[options_int.index(total_score)]
            opt_lines = [f"{opt_letters[i]}. {options_int[i]}" for i in range(n_value)]
            if use_5:
                opt_lines.append(f"{opt_letters[-1]}. No correct answer")

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        sidx = (self.seed or 0) % len(_STEM_TEMPLATES)
        stem = _STEM_TEMPLATES[sidx]
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )

        if level <= 1:
            question += (
                "\nHint: read the ring score for each red dot, then add. "
                "Pick the option matching that sum."
            )
        elif level <= 3:
            question += (
                "\nHint: identify the ring containing each red dot; sum the "
                "ring values."
            )

        img = self._render(ring_values, darts)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering — bigger and cleaner than the prior version
    # ------------------------------------------------------------------ #
    def _render(self, values, darts) -> Image.Image:
        n_rings = len(values)
        fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        ring_colors = ["#fce8e6", "#fdf2c5", "#d4f1c5", "#c8e6f7",
                       "#e1d4f7", "#f7d4d4", "#fff2cc"][:n_rings]

        # Draw rings outermost first so inner overlay
        for r in range(n_rings, 0, -1):
            circle = plt.Circle((0, 0), r, color=ring_colors[r - 1],
                                ec="#2c3e50", linewidth=1.6)
            ax.add_patch(circle)
        # Bullseye dot (visual center marker)
        ax.add_patch(plt.Circle((0, 0), 0.05, color="#222", zorder=4))

        # Ring value labels — placed at top of each ring with a clear bg box
        for i, val in enumerate(values):
            ax.text(0, i + 0.5, str(val),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="#ffffff",
                              edgecolor="#2c3e50",
                              linewidth=0.8,
                              alpha=0.92),
                    zorder=5)

        # Darts as larger, clearly visible markers
        for x, y, _ in darts:
            ax.scatter([x], [y], color="#d62728", s=220,
                       marker="o", edgecolor="black", linewidth=2,
                       zorder=10)

        ax.set_xlim(-n_rings - 0.6, n_rings + 0.6)
        ax.set_ylim(-n_rings - 0.6, n_rings + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Dartboard with hits (red)", fontsize=13,
                     fontweight="bold", color="#2c3e50", pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = DartboardScoreQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   pos={v_pos['accuracy']} neg={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
