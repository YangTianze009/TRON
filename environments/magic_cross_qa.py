"""
Magic-cross / plus-shape number puzzle.

A "+" / cross shape has N circles arranged horizontally and N circles arranged
vertically, sharing one center circle. The sum of all circles on each arm
(or each line) is given. Some peripheral circles are revealed; the model finds
the missing center value (or, occasionally, a missing peripheral value).

Reference qid 115:
  "The sum of the three numbers on each of the two lines of the cross is 83.
   Find the number in the center." Ans: 8.

L0: 5-circle plus (3 horizontal + 3 vertical, sharing center). All 4
peripheral circles labeled, one line-sum given. Answer = sum - peripheral_a -
peripheral_b. Trivial subtraction.
L9: 9-circle plus (5 horizontal + 5 vertical, sharing center). Multiple
unknowns + a constraint chain.

Answer: integer (or float) for the missing center.
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The cross is made of two lines (horizontal and vertical) sharing the center circle. The sum of the numbers on each line equals {S}. Find the number in the center circle. Place the integer in <answer>...</answer>.",
    "In the plus-shaped figure, the numbers on each line (horizontal and vertical) sum to {S}. What is the number in the center? Integer answer in <answer>...</answer>.",
    "Cross puzzle: every line of circles (horizontal and vertical) sums to {S}. Determine the value at the center circle. Place integer in <answer>...</answer>.",
    "The two lines (horizontal, vertical) of the cross each sum to {S}. Find the center number and put the integer in <answer>...</answer>.",
    "Each arm of the plus shape sums to {S} when the center is included. What number belongs in the center circle? Integer in <answer>...</answer>.",
    "Look at the cross of circles. The sum of all circles on the horizontal line equals {S}, and the same for the vertical line. Find the center value. Integer answer in <answer>...</answer>.",
    "Number cross puzzle: both lines of the cross have the same total {S}. What is the missing center value? Place integer in <answer>...</answer>.",
    "Examine the cross. Each line of circles (horizontal and vertical) sums to {S}. Find the integer at the center. Answer in <answer>...</answer>.",
    "The plus-shape puzzle: every straight line through the center sums to {S}. Determine the central number. Integer answer in <answer>...</answer>.",
    "Cross of {N} circles: both the horizontal and vertical lines sum to {S}. Find the center. Integer in <answer>...</answer>.",
    "Find the center number of the cross. Each line through the center totals {S}. Place the integer in <answer>...</answer>.",
    "Sum-of-cross puzzle: each line equals {S}. Compute the center value. Integer answer in <answer>...</answer>.",
    "Solve the cross puzzle: numbers on each arm-line sum to {S}. What is the center? Integer in <answer>...</answer>.",
    "The figure shows a cross (plus shape). Each line of circles sums to {S}. Find the missing center number. Integer answer in <answer>...</answer>.",
    "Cross-sum puzzle. Both lines of circles total {S}. Determine the center entry and put the integer in <answer>...</answer>.",
    "Identify the center value of the cross such that each line (horizontal and vertical) sums to {S}. Integer in <answer>...</answer>.",
]


class MagicCrossQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "magic_cross"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # arm_len = number of cells on each side of the center on one arm
        # Total circles per line = 2*arm_len + 1
        if level == 0:
            arm_len = 1
        elif level <= 2:
            arm_len = 1
        elif level <= 5:
            arm_len = 2
        else:
            arm_len = 2
        # n_unknowns: how many circles to leave blank (besides the question target)
        if level == 0:
            n_extra_unknowns = 0
        elif level <= 3:
            n_extra_unknowns = 0
        elif level <= 6:
            n_extra_unknowns = 1
        else:
            n_extra_unknowns = 2
        return {"level": level, "arm_len": arm_len,
                "n_extra_unknowns": n_extra_unknowns}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        arm = cfg["arm_len"]
        rng = random.Random((self.seed or 0) * 1493 + level * 47 + 7)

        # Build positions: horizontal arm cells [-arm..arm], vertical arm cells
        # [-arm..arm], sharing the center (0,0).
        h_positions = [(i, 0) for i in range(-arm, arm + 1)]   # left .. right
        v_positions = [(0, j) for j in range(-arm, arm + 1)]   # bottom .. top
        all_positions = list(set(h_positions) | set(v_positions))
        center = (0, 0)

        # Pick small distinct integers for the cross
        hi = 12 if level <= 3 else 25

        # Strategy: randomly assign peripherals on horizontal line and vertical
        # line so they share the same total target sum S. We pick S first.
        # Choose center value c.
        # Then horizontal peripherals must sum to S - c; vertical too.
        # Generate two disjoint sets of `2*arm` distinct values summing to (S-c).
        max_attempts = 500
        center_value = None
        h_peri = None
        v_peri = None
        target_S = None
        for _attempt in range(max_attempts):
            # Pick fresh target_S each attempt so we explore the space.
            target_S = rng.randint(2 * (arm + 1) + 1, 8 * (arm + 1) + 5)
            c = rng.randint(1, hi)
            target_arm_sum = target_S - c
            # Need 2*arm distinct positive integers summing to target_arm_sum
            # (excluding c). Try to draw them.
            def _draw(n_vals, total, exclude):
                if n_vals <= 0:
                    return [] if total == 0 else None
                # Random partition
                for _ in range(60):
                    pool = [x for x in range(1, hi + 1) if x not in exclude]
                    rng.shuffle(pool)
                    chosen = pool[:n_vals]
                    diff = total - sum(chosen)
                    if diff == 0 and len(set(chosen)) == n_vals:
                        return chosen
                    # Try local fix: replace one element to absorb diff
                    for i in range(n_vals):
                        new_v = chosen[i] + diff
                        if 1 <= new_v <= hi and new_v not in chosen and \
                                new_v not in exclude:
                            chosen[i] = new_v
                            return chosen
                return None
            n_arm = 2 * arm
            h_vals = _draw(n_arm, target_arm_sum, exclude={c})
            if h_vals is None:
                continue
            v_vals = _draw(n_arm, target_arm_sum, exclude=set(h_vals) | {c})
            if v_vals is None:
                continue
            center_value = c
            h_peri = h_vals
            v_peri = v_vals
            break
        if center_value is None:
            return None

        # Layout values on positions.
        # Horizontal peripheral cells (in order left → right, skipping center):
        h_cells = [(i, 0) for i in range(-arm, arm + 1) if (i, 0) != center]
        v_cells = [(0, j) for j in range(-arm, arm + 1) if (0, j) != center]
        cell_value: Dict[Tuple[int, int], int] = {}
        cell_value[center] = center_value
        for cell, v in zip(h_cells, h_peri):
            cell_value[cell] = v
        for cell, v in zip(v_cells, v_peri):
            cell_value[cell] = v

        # Decide which cells to hide.
        # Always hide the center (the question target).
        hidden = {center}
        # Optionally hide a few peripherals at higher levels.
        n_extra = cfg["n_extra_unknowns"]
        peripheral_cells = list((set(h_cells) | set(v_cells)) - {center})
        rng.shuffle(peripheral_cells)
        # Even when we hide peripherals, the model can still solve for the
        # center because the OTHER line is fully revealed. So center remains
        # uniquely determinable from the line sum and revealed peripherals.
        # We hide peripherals from the SAME line as a single one (so that line
        # has 2 unknowns total: center + 1 peri), keeping the other line
        # fully revealed for resolution.
        # To keep it solvable, only hide cells from one arm at a time:
        if n_extra > 0:
            # Hide n_extra cells from one arm (say horizontal). Then the model
            # uses the vertical line to recover center first.
            hide_arm = rng.choice(["h", "v"])
            cells_to_hide_pool = h_cells if hide_arm == "h" else v_cells
            cells_to_hide_pool = [c for c in cells_to_hide_pool if c != center]
            rng.shuffle(cells_to_hide_pool)
            for c in cells_to_hide_pool[:n_extra]:
                hidden.add(c)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        N_circles = len(all_positions)
        question = _TEMPLATES[sidx].format(S=target_S, N=N_circles)
        answer = str(center_value)
        img = self._render(arm, cell_value, hidden, target_S)
        return question, answer, img

    def _render(self, arm: int, cell_value: Dict[Tuple[int, int], int],
                hidden: set, target_S: int) -> Image.Image:
        rng = self._rng
        bg = "#ffffff"
        cell_color = "#dbe9f7"
        unknown_color = "#fde2e4"
        center_color = "#fff3b0"
        edge_color = "#1d3557"

        size = 2 * arm + 1
        fig_w = max(4.5, size * 1.0 + 1.5)
        fig_h = max(4.5, size * 1.0 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Draw connecting lines for the cross
        ax.plot([-arm - 0.1, arm + 0.1], [0, 0], color=edge_color,
                linewidth=2.0, zorder=1)
        ax.plot([0, 0], [-arm - 0.1, arm + 0.1], color=edge_color,
                linewidth=2.0, zorder=1)

        # Draw all circles
        for (i, j), v in cell_value.items():
            is_hidden = (i, j) in hidden
            is_center = (i == 0 and j == 0)
            if is_hidden:
                fc = unknown_color
            elif is_center:
                fc = center_color
            else:
                fc = cell_color
            circ = plt.Circle((i, j), 0.36, facecolor=fc,
                              edgecolor=edge_color, linewidth=1.6, zorder=3)
            ax.add_patch(circ)
            label = "?" if is_hidden else str(v)
            color = "#b00020" if is_hidden else "#1a1a1a"
            ax.text(i, j, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color, zorder=4)

        ax.set_xlim(-arm - 1.0, arm + 1.0)
        ax.set_ylim(-arm - 1.0, arm + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Each line of the cross sums to {target_S}",
                     fontsize=11, pad=6)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg,
                    dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
