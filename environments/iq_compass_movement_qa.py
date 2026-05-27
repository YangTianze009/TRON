"""
Compass-position movement IQ test. A small element (a colored dot/triangle/
arrow) sits in one of K positions arranged on a circle (a 4-, 6-, or 8-
position 'compass'). Each step the element moves K-1 → next slot in a
fixed direction (CW or CCW). Given 3-4 frames showing the element's
positions, pick the next frame from 4 candidates labeled A-D.

Difficulty axes:
  - K (number of compass positions): 4 → 8
  - direction (CW vs CCW)
  - step size (1 vs 2 slots per turn)
  - presence of a second element with its own rule
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_QUESTION_STEMS = [
    "An element moves around a fixed set of positions arranged on a circle. Which of the boxes comes next in the sequence? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Each frame shows the position of an element on a circular track. Which option (A/B/C/D) is the next frame? Letter in <answer>...</answer>.",
    "Study the movement of the element around the circle in the sequence. Which frame is next? Letter (A/B/C/D) in <answer>...</answer>.",
    "An element moves a fixed number of slots around a circle each step. Which of the four options is the next frame? Letter (A/B/C/D) in <answer>...</answer>.",
    "Identify the rule of motion (clockwise / counter-clockwise / step size) and pick the next frame. Letter (A/B/C/D) in <answer>...</answer>.",
    "Look at how the element circulates among the positions. What comes next? Letter (A/B/C/D) in <answer>...</answer>.",
    "The element advances to a new position each turn. Choose the next position from A-D. Letter in <answer>...</answer>.",
    "Each step in the sequence applies a consistent compass-style move. Which of A-D extends the sequence? Letter in <answer>...</answer>.",
    "Determine the step size and direction of motion. Which option is the next frame? Letter (A/B/C/D) in <answer>...</answer>.",
    "Inspect the frames and decide which option (A/B/C/D) shows the element in its next position. Letter in <answer>...</answer>.",
    "What choice (A, B, C, or D) should be in place of the question mark that fits the pattern? Letter in <answer>...</answer>.",
    "Which of the boxes comes next? Reply with one letter from A-D. Letter in <answer>...</answer>.",
    "Decode the pattern of the moving element and pick the next frame. Letter (A/B/C/D) in <answer>...</answer>.",
    "Which option from A-D should replace the question mark in the sequence? Letter in <answer>...</answer>.",
    "The element rotates around the circle by a fixed amount each step. Which of A-D continues the pattern? Letter in <answer>...</answer>.",
    "Choose the option that completes the sequence. Reply with one letter (A/B/C/D) in <answer>...</answer>.",
]


def _slot_xy(K, slot, R=1.0):
    """Position of compass slot index on a circle of radius R, angle 0 = top."""
    a = math.pi / 2 - 2 * math.pi * slot / K
    return R * math.cos(a), R * math.sin(a)


class IQCompassMovementQA(StandaloneVisualEnv):
    ENV_NAME = "iq_compass_movement"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {"K": 4, "n_seq": 2, "step_pool": [1], "two_elements": False}
        if level <= 3:
            return {"K": 4, "n_seq": 3, "step_pool": [1], "two_elements": False}
        if level <= 5:
            return {"K": 6, "n_seq": 3, "step_pool": [1], "two_elements": False}
        if level <= 7:
            return {"K": 6, "n_seq": 4, "step_pool": [1, 2], "two_elements": False}
        return {"K": 8, "n_seq": 4, "step_pool": [1, 2], "two_elements": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 8467 + level * 31 + 257)

        K = cfg["K"]
        n_seq = cfg["n_seq"]
        step = rng.choice(cfg["step_pool"])
        direction = rng.choice([+1, -1])  # +1 CW, -1 CCW
        start = rng.randrange(0, K)

        # Element 1 path
        path1 = [(start + i * step * direction) % K for i in range(n_seq + 1)]
        # Element 2 path (if any)
        if cfg["two_elements"]:
            start2 = (start + rng.randrange(1, K)) % K
            step2 = rng.choice(cfg["step_pool"])
            dir2 = rng.choice([+1, -1])
            path2 = [(start2 + i * step2 * dir2) % K for i in range(n_seq + 1)]
        else:
            path2 = None

        correct_state = (path1[-1], path2[-1] if path2 else None)

        # Build distractors
        distractors = []
        # Wrong direction
        d_dir = (start + n_seq * step * (-direction)) % K
        distractors.append((d_dir, path2[-1] if path2 else None))
        # Wrong step (off by one slot)
        d_step = (path1[-2] + (step + 1) * direction) % K
        distractors.append((d_step, path2[-1] if path2 else None))
        # No step (stay where it was)
        distractors.append((path1[-2], path2[-1] if path2 else None))
        # Apply step twice
        d_double = (path1[-1] + step * direction) % K
        distractors.append((d_double, path2[-1] if path2 else None))
        # Random different slot
        for _ in range(8):
            r_slot = rng.randrange(0, K)
            if r_slot != correct_state[0]:
                distractors.append((r_slot, path2[-1] if path2 else None))
                break
        # If two elements: also vary element 2
        if path2:
            distractors.append((correct_state[0], path2[-2]))
            d2_dir = (path2[-1] + step * (-direction)) % K
            distractors.append((correct_state[0], d2_dir))

        seen = {correct_state}
        opts = [correct_state]
        for d in distractors:
            if d not in seen:
                seen.add(d)
                opts.append(d)
            if len(opts) == 4:
                break
        # Top up if needed
        tries = 0
        while len(opts) < 4 and tries < 30:
            tries += 1
            d_slot = rng.randrange(0, K)
            d_state = (d_slot, path2[-1] if path2 else None)
            if d_state not in seen:
                seen.add(d_state)
                opts.append(d_state)
        if len(opts) < 4:
            return None
        opts = opts[:4]
        rng.shuffle(opts)
        ans_idx = opts.index(correct_state)
        answer = chr(ord("A") + ans_idx)

        q = rng.choice(_QUESTION_STEMS)
        if level <= 1:
            q = q + (
                " Hint: trace the element from one frame to the next. Note "
                "the direction (clockwise or counter-clockwise) and the number "
                "of positions it advances each step. Apply the same step once "
                "more to find the next frame."
            )
        elif level <= 4:
            q = q + " Hint: identify direction (CW/CCW) and step size, then advance once more."

        img = self._render(K, path1[:n_seq], opts, path2[:n_seq] if path2 else None, rng)
        return q, answer, img

    # -------------------- rendering -------------------- #
    def _draw_compass(self, ax, cx, cy, K, slot1, slot2, R=0.8):
        # Draw K dots around the circle, marker at the active slot.
        # All slots: small white circles
        for i in range(K):
            sx, sy = _slot_xy(K, i, R)
            sx += cx; sy += cy
            ax.add_patch(mpatches.Circle((sx, sy), 0.07,
                                          facecolor="#ffffff",
                                          edgecolor="#7f8c8d", linewidth=1.0))
        # Outer guide circle (faint)
        ax.add_patch(mpatches.Circle((cx, cy), R, facecolor="none",
                                       edgecolor="#bdbdbd", linewidth=0.8,
                                       linestyle="--"))
        # Active element 1
        sx, sy = _slot_xy(K, slot1, R)
        sx += cx; sy += cy
        ax.add_patch(mpatches.Circle((sx, sy), 0.13,
                                       facecolor="#212121",
                                       edgecolor="#212121", linewidth=1.4,
                                       zorder=3))
        # Active element 2 (different shape)
        if slot2 is not None:
            sx2, sy2 = _slot_xy(K, slot2, R)
            sx2 += cx; sy2 += cy
            tri = mpatches.Polygon(
                [(sx2, sy2 + 0.12), (sx2 - 0.10, sy2 - 0.07),
                 (sx2 + 0.10, sy2 - 0.07)],
                closed=True, facecolor="#c0392b", edgecolor="#c0392b",
                linewidth=1.4, zorder=3)
            ax.add_patch(tri)

    def _render(self, K, path1, opts, path2, rng) -> Image.Image:
        n_seq = len(path1)
        cell = 2.0
        fig = plt.figure(figsize=(2 + 1.4 * (n_seq + 1), 5.8))
        fig.patch.set_facecolor("#ffffff")
        ax_top = fig.add_subplot(2, 1, 1); ax_top.set_facecolor("#ffffff")
        ax_bot = fig.add_subplot(2, 1, 2); ax_bot.set_facecolor("#ffffff")
        ax_top.set_aspect("equal"); ax_bot.set_aspect("equal")
        ax_top.axis("off"); ax_bot.axis("off")

        # Top: sequence frames
        positions = [(i * cell, 0) for i in range(n_seq + 1)]
        for i, (slot, (px, py)) in enumerate(zip(path1, positions[:n_seq])):
            rect = mpatches.Rectangle(
                (px - cell / 2 + 0.05, py - cell / 2 + 0.05),
                cell - 0.1, cell - 0.1,
                facecolor="#ffffff", edgecolor="#7f8c8d", linewidth=1)
            ax_top.add_patch(rect)
            slot2 = path2[i] if path2 else None
            self._draw_compass(ax_top, px, py, K, slot, slot2)
            ax_top.text(px, py - cell / 2 - 0.18, f"{i + 1}",
                        fontsize=10, ha="center", va="top")
        # ? cell
        qx, qy = positions[n_seq]
        rect = mpatches.Rectangle(
            (qx - cell / 2 + 0.05, qy - cell / 2 + 0.05),
            cell - 0.1, cell - 0.1,
            facecolor="#ffeaa7", edgecolor="#c0392b", linewidth=1.5)
        ax_top.add_patch(rect)
        ax_top.text(qx, qy, "?", fontsize=22, fontweight="bold",
                    ha="center", va="center", color="#c0392b")
        xs_all = [p[0] for p in positions]
        ax_top.set_xlim(min(xs_all) - 1.0, max(xs_all) + 1.0)
        ax_top.set_ylim(-cell, cell)

        # Bottom: options
        for i, (slot, slot2) in enumerate(opts):
            ox = i * cell
            oy = 0
            rect = mpatches.Rectangle(
                (ox - cell / 2 + 0.05, oy - cell / 2 + 0.05),
                cell - 0.1, cell - 0.1,
                facecolor="#ffffff", edgecolor="#34495e", linewidth=1)
            ax_bot.add_patch(rect)
            self._draw_compass(ax_bot, ox, oy, K, slot, slot2)
            ax_bot.text(ox, oy - cell / 2 - 0.18, f"({chr(ord('A') + i)})",
                        fontsize=12, fontweight="bold",
                        ha="center", va="top")
        ax_bot.set_xlim(-1.0, (len(opts) - 1) * cell + 1.0)
        ax_bot.set_ylim(-cell, cell)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, hspace=0.35)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_compass"
    os.makedirs(out_dir, exist_ok=True)
    env = IQCompassMovementQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']}")
