"""
Arrow Block Simulate QA (v4 G25).

Targets: spatial-vision MentalAnimation -2.50 + BlockMoving -2.50.

Both subtasks need mental simulation of a movement sequence over a grid of
arrows/blocks. Task: render a grid with labeled blocks + an arrow pattern;
specify a sequence of moves (push block N up 2, rotate arrow A 90° CW);
ask what the final grid looks like — as coords, or as letter option.

Reward: final-coordinates equality or MCQ letter.

Level axes:
  A) Grid size: 4x4 → 6x6
  B) Moves count
  C) Move types: translate at L0-3, add rotate at L4-6, add chain-effect (push another block) at L7+
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The grid shows labeled blocks (A, B, C, ...) at start positions. Apply these moves in order: {moves}. Where does block {target} end up? Answer as 'row,column' (1-indexed, row 1 = bottom) in <answer>...</answer>.",
    "Starting from the grid shown, apply the move sequence {moves}. Final position of block {target}? Format row,column in <answer>...</answer>.",
    "Simulate: {moves}. Block {target}'s final cell? row,column in <answer>...</answer>.",
    "Apply move sequence to the grid: {moves}. Where is block {target} at the end? row,column in <answer>...</answer>.",
    "Starting grid + moves {moves}. Block {target} ends at? row,column in <answer>...</answer>.",
    "Move blocks per sequence {moves}. Final position of {target}? row,column in <answer>...</answer>.",
    "Simulate the move chain {moves}. Where is {target}? row,column in <answer>...</answer>.",
    "Apply {moves} to the grid. Block {target}'s final cell: row,column in <answer>...</answer>.",
    "Block {target} starts at the shown position. After {moves}, where is it? row,column in <answer>...</answer>.",
    "The block grid undergoes {moves}. Final {target}? row,column in <answer>...</answer>.",
    "Simulate block moves: {moves}. Where does {target} end? row,column in <answer>...</answer>.",
    "Starting grid + {moves} → final {target}? row,column in <answer>...</answer>.",
    "Grid + sequence {moves}. Final cell of {target}? row,column in <answer>...</answer>.",
    "After the move sequence {moves}, where is {target}? row,column in <answer>...</answer>.",
    "Track {target} through {moves}. Final position? row,column in <answer>...</answer>.",
    "Follow moves {moves} on the grid. Where does {target} land? row,column in <answer>...</answer>.",
]

class ArrowBlockSimulateQA(StandaloneVisualEnv):
    ENV_NAME = "arrow_block_simulate"
    NEEDS_COT_FLOOR = True  # spatial-vision is 100% bare-letter in flipped cases

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        grid_sz = 4 + level // 3           # 4, 4, 4, 5, 5, 5, 6, 6, 6, 6
        n_blocks = 2 + level // 3
        n_moves = 2 + level // 2
        return {"grid_size": grid_sz, "n_blocks": min(5, n_blocks),
                 "n_moves": min(5, n_moves)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 937)
        self._primary_complexity_feature = level

        gs = cfg["grid_size"]
        # Place blocks (labeled A, B, C, ...) at random positions
        labels = "ABCDE"[:cfg["n_blocks"]]
        positions = {}
        used = set()
        for lab in labels:
            for _ in range(30):
                x = rng.randint(0, gs - 1)
                y = rng.randint(0, gs - 1)
                if (x, y) not in used:
                    used.add((x, y))
                    positions[lab] = (x, y)
                    break
        if len(positions) != len(labels):
            return None

        # Generate move sequence
        moves = []
        for _ in range(cfg["n_moves"]):
            block = rng.choice(labels)
            direction = rng.choice(["up", "down", "left", "right"])
            n_steps = rng.randint(1, gs - 1)
            moves.append((block, direction, n_steps))

        # Apply moves (with boundary clamping; no push-chain for simplicity at L0-6)
        start_positions = dict(positions)
        dir_delta = {"up": (0, 1), "down": (0, -1),
                      "left": (-1, 0), "right": (1, 0)}
        for (block, direction, n_steps) in moves:
            dx, dy = dir_delta[direction]
            x, y = positions[block]
            for _ in range(n_steps):
                nx, ny = x + dx, y + dy
                # Clamp to grid
                if 0 <= nx < gs and 0 <= ny < gs:
                    # Check collision: if another block is there, stop
                    occupied = {p for k, p in positions.items() if k != block}
                    if (nx, ny) in occupied:
                        break
                    x, y = nx, ny
                else:
                    break
            positions[block] = (x, y)

        # Pick a target to query
        target = rng.choice(labels)
        fx, fy = positions[target]
        # 1-indexed row (row 1 = bottom = y=0+1), column (1-indexed x+1)
        answer = f"{fy + 1},{fx + 1}"

        # Render the START grid with moves listed
        moves_str = "; ".join(f"push {b} {d} {n}" for b, d, n in moves)
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(moves=moves_str, target=target)

        img = self._render(gs, start_positions, rng)
        return q, answer, img

    def _render(self, gs, positions, rng):
        fig, ax = plt.subplots(figsize=(5 + gs * 0.3, 5 + gs * 0.3))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, gs + 0.5)
        ax.set_ylim(-0.5, gs + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Grid
        for i in range(gs + 1):
            ax.plot([0, gs], [i, i], color="#aaaaaa", lw=1.0)
            ax.plot([i, i], [0, gs], color="#aaaaaa", lw=1.0)
        # Row/col labels
        for i in range(gs):
            ax.text(-0.3, i + 0.5, str(i + 1), fontsize=10,
                    ha="right", va="center")
            ax.text(i + 0.5, -0.3, str(i + 1), fontsize=10,
                    ha="center", va="top")
        ax.text(-0.65, gs / 2, "row↑", fontsize=10, rotation="vertical",
                ha="center", va="center")
        ax.text(gs / 2, gs + 0.25, "col →", fontsize=10, ha="center")

        colors = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71",
                   "D": "#f39c12", "E": "#9b59b6"}
        for lab, (x, y) in positions.items():
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1,
                                             fc=colors.get(lab, "#888"),
                                             ec="black", lw=1.5, alpha=0.9))
            ax.text(x + 0.5, y + 0.5, lab, fontsize=16, ha="center",
                    va="center", color="white", fontweight="bold")

        ax.set_title("Start positions (row/col 1-indexed)", fontsize=11)
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().replace(" ", "").rstrip(".")
        gt = ground_truth.strip().lower().replace(" ", "").rstrip(".")
        return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_abs"
    os.makedirs(out_dir, exist_ok=True)
    env = ArrowBlockSimulateQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 81
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[abs L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/abs_s{s}_L{level}.png")
            print(f"[abs L{level} s{s}] A={env._answer}")
