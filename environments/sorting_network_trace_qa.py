"""
Sorting Network Trace QA (v3 planning batch, 2026-04-16).

Target: reference algorithm_problems / a puzzle benchmark algorithmic.
Renders a sorting network (N horizontal wires + vertical comparators).
The model is given the input values on the left of the wires and is asked
to report the value on a specific wire after all comparators have been
applied (in left-to-right order).

Format: integer short-answer ("Answer with a single integer.").

Difficulty axes:
  1. n_wires = 3 + level // 2 (3..7)
  2. n_comparators = 2 + level (2..11)
  3. parallel_stages = level >= 4 (comparators may fire simultaneously in
     a stage, so the diagram becomes visually denser and the trace order
     is less obvious).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_QUESTION_TEMPLATES = [
    "The image shows a sorting network with {n} horizontal wires (wire 1 "
    "at the top) and a sequence of comparators drawn as vertical segments "
    "connecting pairs of wires. Each comparator compares the two values on "
    "its wires and places the smaller one on the upper wire and the larger "
    "one on the lower wire. The input values on the left are shown on each "
    "wire. Comparators are applied in left-to-right order. What value is "
    "on wire {q} after the last comparator? Answer with a single integer.",
    "A sorting network with {n} wires is drawn. Input values appear on "
    "the left side of each wire (wire 1 is the topmost). Each vertical "
    "comparator sorts its two wires so the smaller value ends up on top. "
    "Apply the comparators from left to right. What is the final value on "
    "wire {q}? Answer with a single integer.",
    "Look at the sorting network. Starting from the input values shown on "
    "the left, trace through every comparator (each makes the upper wire "
    "hold the minimum and the lower wire hold the maximum of the two "
    "wires it touches). Comparators are processed left-to-right. Report "
    "the value on wire {q} at the output. Answer with a single integer.",
    "The diagram is a sorting network with {n} parallel wires. Each "
    "vertical bar is a comparator that swaps the two connected wires if "
    "the upper value is greater than the lower value. Process the "
    "comparators from left to right. What value is on wire {q} at the "
    "right end of the network? Answer with a single integer.",
]

_TITLE_VARIANTS = [
    "Sorting Network", "Comparator Network", "Sort Network",
    "Sorting Circuit", "Comparison Network",
]

class SortingNetworkTraceQA(StandaloneVisualEnv):
    ENV_NAME = "sorting_network_trace"

    # ------------------------------------------------------------------ #
    # Level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_wires": 3 + level // 2,             # 3..7
            "n_comparators": 2 + level,            # 2..11
            "parallel_stages": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1427)
        self._primary_complexity_feature = (cfg["n_wires"] * 100
                                            + cfg["n_comparators"] * 10)

        for _ in range(25):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Problem construction
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n = cfg["n_wires"]
        n_comp = cfg["n_comparators"]

        # Draw n distinct integer inputs in range 1..9 (or 1..(n*2) if n > 5)
        hi = max(9, n * 2)
        inputs = rng.sample(range(1, hi + 1), n)

        # Build comparator list: list of (i, j) pairs, i<j (wires 0-indexed).
        # For parallel_stages = True we group comparators into stages with
        # non-overlapping wires. Each stage still draws left-to-right and is
        # processed left-to-right; parallelism is a visual-density property.
        comparators: List[Tuple[int, int]] = []
        if cfg["parallel_stages"]:
            while len(comparators) < n_comp:
                stage_size = rng.randint(1, max(1, n // 2))
                stage: List[Tuple[int, int]] = []
                used: set = set()
                attempts = 0
                while len(stage) < stage_size and attempts < 20:
                    attempts += 1
                    i, j = sorted(rng.sample(range(n), 2))
                    if i in used or j in used:
                        continue
                    stage.append((i, j))
                    used.add(i); used.add(j)
                for c in stage:
                    if len(comparators) < n_comp:
                        comparators.append(c)
        else:
            # Sequential comparators; enforce adjacent pairs are not identical
            prev = None
            for _ in range(n_comp):
                tries = 0
                while tries < 20:
                    tries += 1
                    i, j = sorted(rng.sample(range(n), 2))
                    if (i, j) != prev:
                        break
                comparators.append((i, j))
                prev = (i, j)

        # Simulate comparators left-to-right
        values = list(inputs)
        for (i, j) in comparators:
            if values[i] > values[j]:
                values[i], values[j] = values[j], values[i]

        # Choose a wire to ask about (1-indexed in the question)
        q_wire = rng.randint(0, n - 1)
        answer_val = values[q_wire]

        # Reject if the answer is trivially the same as the input on that
        # wire (no change at all) AND there were multiple comparators —
        # keeps problems meaningful.
        if answer_val == inputs[q_wire] and n_comp >= 3:
            # Still allow some fraction so harder levels can have stable
            # wires, but skip if diversity is low.
            if rng.random() < 0.5:
                return None

        # Question
        template = rng.choice(_QUESTION_TEMPLATES)
        question = template.format(n=n, q=q_wire + 1)

        image = self._render(inputs, comparators, n, rng)
        return question, str(answer_val), image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, inputs: List[int], comparators: List[Tuple[int, int]],
                n: int, rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        n_comp = len(comparators)
        # Figure width scales with comparator count
        width = max(6.0, 1.4 + 0.7 * n_comp)
        height = max(3.2, 0.7 * n + 1.4)

        fig, ax = plt.subplots(figsize=(width * sc, height * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        # Wire y-coordinates: wire 1 at the top
        y_of = lambda i: (n - 1 - i)

        # x-positions of comparators
        left_pad = 1.4
        right_pad = 1.0
        x_span = max(1.0, n_comp * 0.8)
        if n_comp == 1:
            x_positions = [left_pad + x_span * 0.5]
        else:
            x_positions = [left_pad + x_span * (k / max(1, n_comp - 1))
                            for k in range(n_comp)]

        total_x = left_pad + x_span + right_pad

        # Horizontal wires
        line_color = style.get("geo_line_color", "#2c3e50")
        lw = style.get("line_width", 1.5)
        for i in range(n):
            ax.plot([0.4, total_x], [y_of(i)] * 2,
                    color=line_color, linewidth=lw + 0.5, zorder=1)
            # Wire label on far left
            ax.text(0.25, y_of(i), f"w{i+1}",
                    ha="right", va="center",
                    fontsize=fs, fontweight="bold",
                    color=line_color)
            # Input value, just to the right of the label
            ax.text(0.65, y_of(i) + 0.18, str(inputs[i]),
                    ha="left", va="bottom",
                    fontsize=fs + 1, fontweight="bold",
                    color=palette[i % len(palette)])

        # Comparators
        for idx, ((i, j), x) in enumerate(zip(comparators, x_positions)):
            y1 = y_of(i)
            y2 = y_of(j)
            c = palette[(idx + 3) % len(palette)]
            ax.plot([x, x], [y1, y2],
                    color=c, linewidth=lw + 1.2, zorder=2)
            # Endpoint dots
            ax.add_patch(mpatches.Circle((x, y1), 0.10,
                                          facecolor=c, edgecolor="#000",
                                          linewidth=0.8, zorder=3))
            ax.add_patch(mpatches.Circle((x, y2), 0.10,
                                          facecolor=c, edgecolor="#000",
                                          linewidth=0.8, zorder=3))
            # Order label under the comparator
            ax.text(x, -0.55, f"{idx+1}", ha="center", va="top",
                    fontsize=fs - 1, color="#555")

        # Axis bounds
        ax.set_xlim(-0.2, total_x + 0.2)
        ax.set_ylim(-1.0, n - 0.2)

        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=fs + 2, fontweight="bold")

        # Annotation legend
        ax.text(total_x * 0.5, n - 0.55,
                "(each comparator: upper = min, lower = max; "
                "applied left-to-right)",
                ha="center", va="top", fontsize=fs - 2, color="#666",
                style="italic")

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = SortingNetworkTraceQA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
