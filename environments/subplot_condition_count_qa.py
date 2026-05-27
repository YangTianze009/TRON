"""
Subplot Condition Count QA — pure single-letter MCQ.

2026-05-05 R5 P5 (Batch 2 Agent B): REWRITTEN. Render an arXiv-style figure
with N subplots labeled (a), (b), ..., each containing a small chart
(curve or bar). Pose a per-subplot condition (e.g. "max value > T") and
ask the model to count how many subplots satisfy it. Pure 5-MCQ output
A/B/C/D/E.

Previous version (17.5% R3 passrate) was too hard partly because:
  - L0 had a literal "the answer is 1" leak in the prompt (now removed).
  - L0/L1 used 4 sinusoidal panels (visually noisy).
  - Required free-form integer answer (most failures were format).

Per-level design (much easier L0/L1, harder L8/L9):
  L0/L1 — 4-MCQ. 2-3 simple bar subplots (single bar each), threshold
          dashed line drawn. Count how many bars cross the line.
  L2/L3 — 5-MCQ. 4 subplots, simple bars. 10% trap.
  L4/L5 — 5-MCQ. 4 subplots with sinusoidal curves. 25% trap.
  L6/L7 — 5-MCQ. 6 subplots (mix of bars and curves). 30% trap.
  L8/L9 — 5-MCQ. 8-9 subplots, varied conditions per subplot. 40% trap.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _make_curve(rng: random.Random, peak_y: float, n_peaks: int = 2):
    x = np.linspace(0, 10, 200)
    freq = max(0.5, n_peaks * math.pi / 10.0)
    phase = rng.uniform(0, 2 * math.pi)
    y = np.sin(freq * x + phase)
    drift = 0.2 * np.sin(0.2 * x + rng.uniform(0, 2 * math.pi))
    y = y + drift
    cur_max = y.max()
    if cur_max != 0:
        y = y / cur_max * peak_y
    return x, y


class SubplotConditionCountQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — count subplots satisfying a condition."""

    ALLOW_ROTATION = False
    ENV_NAME = "subplot_condition_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {
                "n_sub": 2, "panel_kind": "bar",
                "use_5_options": False, "trap_rate": 0.0,
            }
        if level == 1:
            return {
                "n_sub": 3, "panel_kind": "bar",
                "use_5_options": False, "trap_rate": 0.0,
            }
        if level <= 3:
            return {
                "n_sub": 4, "panel_kind": "bar",
                "use_5_options": True, "trap_rate": 0.10,
            }
        if level <= 5:
            return {
                "n_sub": 4, "panel_kind": "curve",
                "use_5_options": True, "trap_rate": 0.25,
            }
        if level <= 7:
            return {
                "n_sub": 6, "panel_kind": "mixed",
                "use_5_options": True, "trap_rate": 0.30,
            }
        if level == 8:
            return {
                "n_sub": 8, "panel_kind": "mixed",
                "use_5_options": True, "trap_rate": 0.40,
            }
        # L9
        return {
            "n_sub": 9, "panel_kind": "mixed",
            "use_5_options": True, "trap_rate": 0.40,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1097 + level * 47 + 53)

        for _ in range(20):
            res = self._try(cfg, sub_rng)
            if res is not None:
                return res
        return None

    def _try(self, cfg, rng):
        n_sub = cfg["n_sub"]
        panel_kind = cfg["panel_kind"]

        # Decide condition type and threshold.
        cond_type = rng.choice(["max_above", "max_below"])
        T = round(rng.uniform(4.0, 6.0), 1)
        if cond_type == "max_below":
            T = round(rng.uniform(2.0, 4.0), 1)

        # Pick how many panels satisfy the condition.
        n_satisfy = rng.randint(1, max(1, n_sub - 1))
        # Build panels.
        panels = []
        for i in range(n_sub):
            satisfies = i < n_satisfy
            if panel_kind == "bar":
                kind = "bar"
            elif panel_kind == "curve":
                kind = "curve"
            else:
                kind = rng.choice(["bar", "curve"])
            if cond_type == "max_above":
                if satisfies:
                    h = rng.uniform(T + 1.5, T + 4.0)
                else:
                    h = rng.uniform(0.5, max(0.6, T - 1.5))
            else:
                if satisfies:
                    h = rng.uniform(0.3, max(0.4, T - 1.5))
                else:
                    h = rng.uniform(T + 1.5, T + 4.0)
            if kind == "bar":
                # 1-3 bars
                k = rng.randint(1, 3)
                vals = [rng.uniform(0.3 * h, h) for _ in range(k - 1)] + [h]
                rng.shuffle(vals)
                panels.append(("bar", vals))
            else:
                npks = rng.choice([2, 3])
                xx, yy = _make_curve(rng, peak_y=h, n_peaks=npks)
                panels.append(("curve", (xx, yy)))
        rng.shuffle(panels)

        # Compute actual count.
        actual_count = 0
        for kind, data in panels:
            if kind == "bar":
                vals = data
                if cond_type == "max_above":
                    if max(vals) > T:
                        actual_count += 1
                else:
                    if max(vals) < T:
                        actual_count += 1
            else:
                xx, yy = data
                if cond_type == "max_above":
                    if yy.max() > T:
                        actual_count += 1
                else:
                    if yy.max() < T:
                        actual_count += 1

        # Build 4 candidate counts: 1 correct + 3 off-by-N.
        # If n_sub < 4 we widen options past [0, n_sub] (e.g. include
        # n_sub + 1) so we can always present 4 distinct numeric choices.
        candidates = [actual_count]
        offsets = [-2, -1, 1, 2, -3, 3, 4, 5]
        rng.shuffle(offsets)
        for off in offsets:
            c = actual_count + off
            if c < 0 or c in candidates:
                continue
            candidates.append(c)
            if len(candidates) == 4:
                break
        if len(candidates) < 4:
            return None

        rng.shuffle(candidates)
        true_pos = candidates.index(actual_count)

        use_5 = cfg["use_5_options"]
        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            # Replace true count with a different distractor → all 4 wrong
            for off in [-3, 3, -4, 4, -5, 5, 6]:
                c = actual_count + off
                if c >= 0 and c not in candidates:
                    candidates[true_pos] = c
                    correct_letter = "E"
                    break
            else:
                return None
        else:
            correct_letter = "ABCD"[true_pos]

        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        opt_lines = "\n".join(f"{opt_letters[k]}. {candidates[k]}"
                              for k in range(4))
        if use_5:
            opt_lines += f"\n{opt_letters[4]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        T_str = int(T) if abs(T - int(T)) < 1e-6 else f"{T:.1f}"
        if cond_type == "max_above":
            cond_phrase = f"contains a value above the dashed line at y = {T_str}"
        else:
            cond_phrase = f"has all values below the dashed line at y = {T_str}"

        stems = [
            f"How many of the lettered subplots satisfy this condition: "
            f"the panel {cond_phrase}? ( )",
            f"Inspect each of the lettered subplots and count how many "
            f"{cond_phrase}. ( )",
            f"Look at all the lettered subplots. How many of them "
            f"{cond_phrase}? ( )",
        ]
        question = (
            f"{rng.choice(stems)}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter: {letter_str}. Be concise."
        )
        image = self._render_subplots(panels, threshold=T, rng=rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _render_subplots(self, panels, threshold=None, rng=None):
        n = len(panels)
        if n == 1:
            rows, cols = 1, 1
        elif n == 2:
            rows, cols = 1, 2
        elif n == 3:
            rows, cols = 1, 3
        elif n == 4:
            rows, cols = 2, 2
        elif n in (5, 6):
            rows, cols = 2, 3
        elif n in (7, 8):
            rows, cols = 2, 4
        elif n == 9:
            rows, cols = 3, 3
        else:
            cols = 3
            rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(2.6 * cols, 2.2 * rows), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[a] for a in axes]
        else:
            axes = [list(row) for row in axes]
        palette = ["#1f4e79", "#a6192e", "#2e7d32", "#6a1b9a",
                   "#ef6c00", "#00838f", "#5d4037", "#283593",
                   "#37474f", "#7b1fa2", "#1565c0", "#bf360c"]
        for k, (kind, data) in enumerate(panels):
            r, c = divmod(k, cols)
            ax = axes[r][c]
            color = palette[k % len(palette)]
            if kind == "bar":
                vals = data
                xs = list(range(len(vals)))
                ax.bar(xs, vals, color=color, edgecolor="#222", linewidth=0.6)
            else:
                xx, yy = data
                ax.plot(xx, yy, color=color, linewidth=1.6)
            if threshold is not None:
                ax.axhline(threshold, color="#c0392b", linewidth=1.2,
                           linestyle="--")
            letter = chr(ord('a') + k)
            ax.set_title(f"({letter})", fontsize=12, loc="left",
                         fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if kind == "bar":
                ax.set_ylim(0, 12)
            else:
                ax.set_ylim(-12, 12)
        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            ax = axes[r][c]
            ax.axis("off")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
