"""
Maze Turn Sequence QA.

2026-05-05 R5 B5B (FINAL): REWRITTEN as pure single-letter MCQ.

Question style: given a small grid maze with a marked starting cell, a
direction arrow, and a drawn path, count the number of 90° turns along the
path. Pure 5-MCQ (or 4-MCQ at L0/L1).

Per-level design:
  L0/L1 — 4-MCQ. 4x4 grid. Path has 0-3 turns (one of A/B/C/D).
  L2-L4 — 5-MCQ + "No correct answer". 5x5 grid. 1-5 turns.
  L5-L7 — 5-MCQ + "No correct answer". 6x6 grid. 1-7 turns. 15% trap.
  L8/L9 — 5-MCQ. 7x7 grid. 1-9 turns. 30/40% trap. Tighter distractors.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
_DIR_SEQ = ["N", "E", "S", "W"]


def _turn(d: str, t: str) -> str:
    idx = _DIR_SEQ.index(d)
    if t == "L":
        return _DIR_SEQ[(idx - 1) % 4]
    if t == "R":
        return _DIR_SEQ[(idx + 1) % 4]
    return d


class MazeTurnSequenceQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive (path geometry)
    ENV_NAME = "maze_turn_sequence"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "grid_size": 4,
                "n_steps_min": 3, "n_steps_max": 5,
                "n_options": 4,
                "trap_rate": 0.0,
                "tight": False,
            }
        if level <= 4:
            return {
                "grid_size": 5,
                "n_steps_min": 4, "n_steps_max": 7,
                "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.20,
                "tight": False,
            }
        if level <= 7:
            return {
                "grid_size": 6,
                "n_steps_min": 5, "n_steps_max": 9,
                "n_options": 5,
                "trap_rate": 0.20,
                "tight": False,
            }
        # L8/L9
        return {
            "grid_size": 7,
            "n_steps_min": 7, "n_steps_max": 11,
            "n_options": 5,
            "trap_rate": 0.40,
            "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 741)

        for _ in range(30):
            res = self._try_generate(cfg, rng, level)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, rng, level):
        gs = cfg["grid_size"]
        # Pick start cell near center
        sx = rng.randint(1, gs - 2)
        sy = rng.randint(1, gs - 2)
        # Initial direction
        init_dir = rng.choice(_DIR_SEQ)

        # Build a random self-avoiding path that stays inside the grid.
        # We pick step directions; the change of direction counts as a turn.
        n_steps = rng.randint(cfg["n_steps_min"], cfg["n_steps_max"])
        path = [(sx, sy)]
        cur_dir = init_dir
        x, y = sx, sy
        actual_dirs = [init_dir]
        for _ in range(n_steps):
            # candidate next directions: keep, turn left, turn right (no backtrack)
            cand = [cur_dir, _turn(cur_dir, "L"), _turn(cur_dir, "R")]
            rng.shuffle(cand)
            placed = False
            for nd in cand:
                dx, dy = _DIRS[nd]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < gs and 0 <= ny < gs):
                    continue
                if (nx, ny) in path:
                    continue
                x, y = nx, ny
                cur_dir = nd
                path.append((x, y))
                actual_dirs.append(nd)
                placed = True
                break
            if not placed:
                break
        if len(path) < 3:
            return None

        # Count 90° turns: consecutive direction changes (excluding the
        # initial heading itself; only turns between segments count).
        turns = 0
        for i in range(1, len(actual_dirs)):
            if actual_dirs[i] != actual_dirs[i - 1]:
                turns += 1
        # We want at least 0 (rare) or 1 turn; for L0/L1 keep 0-3 turns.
        if level <= 1:
            if turns > 3:
                return None

        # Build options
        n_options = cfg["n_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]
        # numeric distractor pool
        distractors = self._build_numeric_distractors(turns, cfg, rng)
        if cfg["n_options"] == 4:
            # No "No correct answer"; need 3 distractors, all numeric
            if len(distractors) < 3:
                return None
            options = [str(turns)] + [str(d) for d in distractors[:3]]
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(str(turns))]
        else:
            # 5 options including "No correct answer" as last
            n_numeric_slots = 4
            use_trap = (rng.random() < cfg["trap_rate"])
            if use_trap and len(distractors) >= n_numeric_slots:
                chosen = distractors[:n_numeric_slots]
                rng.shuffle(chosen)
                options = [str(d) for d in chosen]
                options.append("No correct answer")
                correct_letter = last_letter
            else:
                if len(distractors) < n_numeric_slots - 1:
                    return None
                chosen = [turns] + distractors[:n_numeric_slots - 1]
                rng.shuffle(chosen)
                options = [str(d) for d in chosen]
                options.append("No correct answer")
                correct_letter = opt_letters[options.index(str(turns))]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            ("The diagram shows a walker's path on a grid maze, starting "
             "from the marked START cell with the initial heading shown by "
             "the arrow, and ending at the END cell. How many 90° turns "
             "are in the drawn path?"),
            ("As shown in the figure, the walker follows the colored path "
             "from START to END on the grid. Count the number of 90° turns "
             "(direction changes) along the path."),
            ("In the figure below, a path is drawn from START to END on a "
             "grid maze. How many times does the path turn 90° (either "
             "left or right)?"),
            ("Look at the marked path on the maze (from START at the arrow "
             "to END). What is the total number of 90° turns in the path?"),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        img = self._render(gs, sx, sy, init_dir, path, rng)
        return question, correct_letter, img

    def _build_numeric_distractors(
        self, true_turns: int, cfg: Dict, rng: random.Random
    ) -> List[int]:
        """Numeric distractor pool around the true count."""
        if cfg["tight"]:
            offsets = [-2, -1, 1, 2, -3, 3]
        else:
            offsets = [-3, -2, -1, 1, 2, 3, -4, 4]
        cand = []
        for o in offsets:
            v = true_turns + o
            if v >= 0 and v != true_turns and v not in cand:
                cand.append(v)
        # Add 0 / large numbers
        for v in [0, true_turns + 5, true_turns + 1, max(0, true_turns - 1)]:
            if v != true_turns and v not in cand:
                cand.append(v)
        rng.shuffle(cand)
        return cand

    # ------------------------------------------------------------------ #
    def _render(self, gs, sx, sy, init_dir, path, rng):
        fig, ax = plt.subplots(figsize=(5 + gs * 0.25, 5 + gs * 0.25))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, gs + 0.5)
        ax.set_ylim(-0.5, gs + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        for i in range(gs + 1):
            ax.plot([0, gs], [i, i], color="#aaaaaa", lw=1.0)
            ax.plot([i, i], [0, gs], color="#aaaaaa", lw=1.0)

        for i in range(gs):
            ax.text(-0.3, i + 0.5, str(i + 1), fontsize=10, va="center",
                    ha="right")
            ax.text(i + 0.5, -0.3, str(i + 1), fontsize=10, va="top",
                    ha="center")

        # START cell
        ax.add_patch(mpatches.Rectangle((sx, sy), 1, 1,
                                         fc="#d9ead3", ec="black", lw=1.2))
        ax.text(sx + 0.5, sy + 0.5, "START", fontsize=10, ha="center",
                va="center", fontweight="bold")
        # Arrow
        dx, dy = _DIRS[init_dir]
        ax.annotate("", xy=(sx + 0.5 + dx * 0.4, sy + 0.5 + dy * 0.4),
                    xytext=(sx + 0.5, sy + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))

        # END cell
        ex, ey = path[-1]
        ax.add_patch(mpatches.Rectangle((ex, ey), 1, 1,
                                         fc="#fde2e2", ec="black", lw=1.2))
        ax.text(ex + 0.5, ey + 0.5, "END", fontsize=10, ha="center",
                va="center", fontweight="bold")

        # Draw path centerline
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]
        path_color = rng.choice(["#1f77b4", "#2ca02c", "#9467bd", "#17becf"])
        ax.plot(xs, ys, color=path_color, lw=3.0, alpha=0.85, zorder=3)
        # Mark each waypoint
        for p in path[1:-1]:
            ax.plot(p[0] + 0.5, p[1] + 0.5, "o",
                    markerfacecolor=path_color, markeredgecolor="white",
                    markersize=6, zorder=4)

        fig.tight_layout()
        return self.fig_to_pil(fig)


if __name__ == "__main__":
    env = MazeTurnSequenceQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed}: ok={ok} A={env._answer}")
