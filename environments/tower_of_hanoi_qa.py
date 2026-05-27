"""
Tower of Hanoi QA.

2026-05-05 R5 B5B (FINAL): REWRITTEN as pure single-letter MCQ. Given a
Tower of Hanoi state and a small number of moves to apply, the model picks
the correct (peg location of a specific disk) or (final state) from a
multiple-choice list.

Per-level design:
  L0/L1 — 4-MCQ. 2 disks. Apply 1 optimal move; ask which peg the smallest
          disk ends up on.
  L2-L4 — 5-MCQ. 3 disks. Apply 2-3 optimal moves; ask which peg the smallest
          / specified disk ends up on. 10/15% trap.
  L5-L7 — 5-MCQ. 4 disks. Apply 4-6 optimal moves; ask peg of disk K. 20% trap.
  L8/L9 — 5-MCQ. 5 disks. Apply 6-10 optimal moves. 30/40% trap.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _solve_hanoi_general(state: List[List[int]],
                         goal_peg: int) -> List[Tuple[int, int]]:
    """BFS solution from arbitrary state to all-on-goal-peg. Returns
    list of (disk_id, dest_peg)."""
    from collections import deque

    def canon(s):
        return tuple(tuple(p) for p in s)

    n_total = sum(len(p) for p in state)
    n_pegs = len(state)
    goal_state = [[] for _ in range(n_pegs)]
    goal_state[goal_peg] = list(range(n_total, 0, -1))
    target = canon(goal_state)
    start = canon(state)
    if start == target:
        return []
    visited = {start}
    parent: Dict = {}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == target:
            break
        for src in range(n_pegs):
            if not cur[src]:
                continue
            top = cur[src][-1]
            for dst in range(n_pegs):
                if dst == src:
                    continue
                if cur[dst] and cur[dst][-1] < top:
                    continue
                new = [list(p) for p in cur]
                disk = new[src].pop()
                new[dst].append(disk)
                ns = canon(new)
                if ns not in visited:
                    visited.add(ns)
                    parent[ns] = (cur, (disk, dst))
                    if ns == target:
                        path = []
                        x = ns
                        while x in parent:
                            prev, mv = parent[x]
                            path.append(mv)
                            x = prev
                        return list(reversed(path))
                    q.append(ns)
    return []


def _apply_moves(state: List[List[int]],
                 moves: List[Tuple[int, int]]) -> List[List[int]]:
    """Apply a list of (disk, dest_peg) moves and return the new state."""
    s = [list(p) for p in state]
    for disk, dst in moves:
        # find disk on top
        src = None
        for pi, peg in enumerate(s):
            if peg and peg[-1] == disk:
                src = pi
                break
        if src is None:
            continue
        s[src].pop()
        s[dst].append(disk)
    return s


def _disk_peg(state: List[List[int]], disk: int) -> int:
    for pi, peg in enumerate(state):
        if disk in peg:
            return pi
    return -1


class TowerOfHanoiQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # peg-position is orientation-sensitive
    ENV_NAME = "tower_of_hanoi"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n": 2, "n_moves_min": 1, "n_moves_max": 1,
                "n_options": 4, "trap_rate": 0.0, "tight": False,
            }
        if level <= 4:
            return {
                "n": 3, "n_moves_min": 2, "n_moves_max": 3,
                "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.15,
                "tight": False,
            }
        if level <= 7:
            return {
                "n": 4, "n_moves_min": 4, "n_moves_max": 6,
                "n_options": 5,
                "trap_rate": 0.20, "tight": False,
            }
        # L8/L9
        return {
            "n": 5, "n_moves_min": 6, "n_moves_max": 10,
            "n_options": 5,
            "trap_rate": 0.40, "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 991 + level * 31 + 7)
        for _ in range(20):
            res = self._try_generate(cfg, rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, rng):
        n = cfg["n"]
        # Build a legal random initial state
        disks = list(range(n, 0, -1))
        state = [[], [], []]
        for d in disks:
            choices = [p for p in range(3) if not state[p] or state[p][-1] > d]
            peg = rng.choice(choices)
            state[peg].append(d)

        goal_peg = 2  # C
        # Avoid initial = goal
        is_solved = (state[goal_peg] == list(range(n, 0, -1))
                     and all(not state[p] for p in range(3) if p != goal_peg))
        if is_solved:
            for p in range(3):
                if state[p] and state[p][-1] == 1:
                    state[p].pop()
                    state[(p + 1) % 3].append(1)
                    break

        try:
            solution = _solve_hanoi_general(
                [list(p) for p in state], goal_peg)
        except Exception:
            return None
        if not solution:
            return None

        # Apply k optimal moves
        k = rng.randint(cfg["n_moves_min"],
                        min(cfg["n_moves_max"], len(solution)))
        applied = solution[:k]
        new_state = _apply_moves(state, applied)

        # Choose disk to ask about (smallest at L0/L1, any at higher levels)
        if cfg["n"] <= 2:
            query_disk = 1
        else:
            query_disk = rng.randint(1, cfg["n"])
        query_peg = _disk_peg(new_state, query_disk)
        if query_peg < 0:
            return None
        peg_letters = ["A", "B", "C"]

        # Build options
        n_options = cfg["n_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]
        true_value = peg_letters[query_peg]

        # Distractors: the other peg names (only 2 wrong pegs!)
        wrong_pegs = [pl for pl in peg_letters if pl != true_value]
        # We'll pad with the same wrong pegs / "Cannot be determined"
        # The 4-MCQ at L0/L1 has 4 options total: but only 3 pegs A/B/C plus
        # "Cannot be determined" placeholder.
        if n_options == 4:
            options = [true_value] + wrong_pegs[:2] + ["Cannot be determined"]
            # Shuffle but keep "Cannot be determined" as the LAST item to match
            # the 4-MCQ design pattern (always D=Cannot)
            head = options[:3]
            rng.shuffle(head)
            options = head + ["Cannot be determined"]
            correct_letter = opt_letters[options.index(true_value)]
        else:
            # 5-MCQ: A/B/C peg labels + "None of the above (disk has been removed)"
            # plus an extra distractor. We'll make the 4 numeric/letter slots
            # = [A, B, C, "Off the board"] and the 5th = "No correct answer"
            use_trap = (rng.random() < cfg["trap_rate"])
            if use_trap:
                # All 4 visible options are wrong; correct = "No correct answer"
                # We use {wrong_pegs (2) + "None of the above" + "Off the board"}
                # but that's only 4 wrong; we need correct to be E label.
                # The wrong_pegs alone cover only 2 pegs -- so "Cannot be"
                # tokens are stand-ins. Force trap by leaving out true peg:
                chosen = wrong_pegs[:2] + ["None of the above",
                                            "Cannot be determined"]
                rng.shuffle(chosen)
                options = list(chosen)
                options.append("No correct answer")
                correct_letter = last_letter
            else:
                # 1 truth + 3 distractors + "No correct answer"
                chosen = ([true_value] + wrong_pegs[:2]
                          + ["Cannot be determined"])
                rng.shuffle(chosen)
                options = list(chosen)
                options.append("No correct answer")
                correct_letter = opt_letters[options.index(true_value)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Stash for later
        self._initial_state = [list(p) for p in state]
        self._n = n

        # Format applied moves as text
        moves_text = ", ".join(
            f"({d},{peg_letters[p]})" for d, p in applied
        )

        stems = [
            (f"The diagram shows the initial state of a Tower of Hanoi puzzle "
             f"with {n} disks. Apply the following {len(applied)} optimal "
             f"move(s) toward the goal of stacking all disks on peg C: "
             f"{moves_text}. After these moves, on which peg does disk "
             f"{query_disk} end up?"),
            (f"As shown in the figure, a {n}-disk Tower of Hanoi has the "
             f"initial state pictured. The optimal-solution moves "
             f"{moves_text} are applied (each move is `(disk, dest_peg)`). "
             f"After these moves, where is disk {query_disk}?"),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        img = self._render(state, n)
        return question, correct_letter, img

    def _render(self, state: List[List[int]], n: int) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        peg_x = [1.5, 4.0, 6.5]
        peg_h = max(2.5, n * 0.45 + 0.5)
        for x in peg_x:
            ax.plot([x, x], [0.2, peg_h], color="#5d4434", linewidth=4)
            ax.plot([x - 1.2, x + 1.2], [0.2, 0.2],
                    color="#5d4434", linewidth=4)
        cmap = plt.get_cmap("tab10")
        for pi, peg in enumerate(state):
            for k, d in enumerate(peg):
                w = 0.25 + d * 0.12
                rect = patches.Rectangle((peg_x[pi] - w, 0.2 + k * 0.4),
                                         2 * w, 0.36,
                                         facecolor=cmap((d - 1) % 10),
                                         edgecolor="#222", linewidth=1.0)
                ax.add_patch(rect)
                ax.text(peg_x[pi], 0.2 + k * 0.4 + 0.18, str(d),
                        ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold")
        for pi, label in enumerate(["A", "B", "C"]):
            ax.text(peg_x[pi], -0.25, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#2c3e50")

        ax.set_xlim(0, 8)
        ax.set_ylim(-0.6, peg_h + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()


if __name__ == "__main__":
    env = TowerOfHanoiQA()
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer}")
