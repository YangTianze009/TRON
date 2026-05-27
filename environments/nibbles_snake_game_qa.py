"""
Nibbles snake game QA.

A snake on an N×N grid must eat one or more apples without colliding with
walls or itself. The snake has an initial body (head plus tail cells), an
initial direction, and one or more apples placed on the grid. The model
outputs a sequence of moves (up/right/down/left, space-separated) that eats
all apples. Snake grows by 1 cell each time an apple is eaten.

Difficulty axes:
  - grid size N (5 -> 10)
  - number of apples (1 -> 5)
  - snake initial length (2 -> 3)
"""
import random
import re
from collections import deque
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the Nibbles snake puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Control the snake one move at a time: each move is `up`, `down`, `left`, or `right` (the direction the head moves).\n"
    "2. The snake cannot move into the wall (off-grid) or into its own body.\n"
    "3. When the head enters a cell containing an apple, the snake eats it and grows by one segment.\n"
    "4. The puzzle is solved when every apple has been eaten.\n\n"
    "### Coordinate System:\n"
    "- Cells are addressed (row, col), 0-indexed; row 0 is at the top, col 0 at the left.\n\n"
    "### Current Puzzle State:\n"
    "- Grid: {n}x{n}\n"
    "- Snake (head first): {snake_str}\n"
    "- Direction: {direction}\n"
    "- Apples: {apples_str}\n"
    "- Goal: Eat {n_apples} apple(s).\n\n"
    "### Output Format:\n"
    "Output the move sequence as space-separated direction words (`up`/`down`/`left`/`right`) inside <answer>...</answer>.\n"
    "Example: <answer>right up up right right right</answer>",

    "Solve the Nibbles snake game below.\n\n"
    "### Game Rules:\n"
    "- Each move is one cell in direction `up`, `down`, `left`, or `right`.\n"
    "- No wall collisions; no self-collisions.\n"
    "- Eating an apple grows the snake by one segment.\n"
    "- Goal: eat every apple.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Grid: {n}x{n}\n"
    "Snake (head first): {snake_str}\n"
    "Direction: {direction}\n"
    "Apples: {apples_str}\n"
    "Goal: Eat {n_apples} apple(s).\n\n"
    "### Output Format:\n"
    "Output the move sequence as space-separated direction words inside <answer>...</answer>.",

    "Your task is to solve the snake (Nibbles) game described below.\n\n"
    "### Game Rules:\n"
    "Standard Nibbles: control the head with up/down/left/right; avoid walls and self; eat all apples.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Grid: {n}x{n}\n"
    "Snake: {snake_str} (head first), facing {direction}.\n"
    "Apples: {apples_str}\n\n"
    "### Output Format:\n"
    "Output the move sequence inside <answer>...</answer>, space-separated direction words.",
]

_DIR_DELTAS = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


# L0-L2 probe templates: ask for the FIRST move only.
_PROBE_TEMPLATES = [
    "Your task is to identify the first move in the Nibbles snake puzzle below:\n\n"
    "### Game Rules:\n"
    "1. Each move is `up`, `down`, `left`, or `right` (the direction the head moves).\n"
    "2. The snake cannot move into the wall (off-grid) or into its own body.\n"
    "3. The head moves toward apples to eat them.\n"
    "4. The snake cannot reverse direction (no 180-degree turns).\n\n"
    "### Coordinate System:\n"
    "- Cells are addressed (row, col), 0-indexed; row 0 is at the top, col 0 at the left.\n\n"
    "### Current Puzzle State:\n"
    "- Grid: {n}x{n}\n"
    "- Snake (head first): {snake_str}\n"
    "- Direction: {direction}\n"
    "- Apples: {apples_str}\n\n"
    "### Output Format:\n"
    "Output ONLY the first move (one direction word: `up`, `down`, `left`, or `right`) inside <answer>...</answer>.\n"
    "Example: <answer>right</answer>",

    "Look at the Nibbles snake state below.\n\n"
    "### Game Rules:\n"
    "- Each move is one cell in direction `up`, `down`, `left`, or `right`.\n"
    "- No wall collisions; no self-collisions; no reversal.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Grid: {n}x{n}\n"
    "Snake (head first): {snake_str}\n"
    "Direction: {direction}\n"
    "Apples: {apples_str}\n\n"
    "### Output Format:\n"
    "What is the first move toward the apple? Output one direction word (up/down/left/right) inside <answer>...</answer>.",

    "Snake (Nibbles) game.\n\n"
    "### Game Rules:\n"
    "Standard Nibbles: control the head with up/down/left/right; avoid walls and self.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Grid: {n}x{n}\n"
    "Snake: {snake_str} (head first), facing {direction}.\n"
    "Apples: {apples_str}\n\n"
    "### Output Format:\n"
    "Provide just the first direction word (up/down/left/right) inside <answer>...</answer>.",
]


def _bfs_path(start_r, start_c, target_r, target_c, n, blocked,
                forbid_first_dir=None):
    """BFS shortest path to target avoiding blocked cells. Returns list of
    (dr, dc) directions or None.

    `blocked` is a set of cells the snake currently occupies (we treat them
    as obstacles for simplicity; this is a slight over-approximation since
    the tail moves out of the way). For our generator's purposes this is
    fine because we generate puzzles around BFS-solvable layouts.

    `forbid_first_dir` (dr, dc) — if set, the path's first step cannot be
    in this direction (used to enforce no-reverse-on-first-move).
    """
    visited = {(start_r, start_c)}
    queue = deque([(start_r, start_c, [])])
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == (target_r, target_c):
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if not path and forbid_first_dir is not None \
                    and (dr, dc) == forbid_first_dir:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if (nr, nc) in visited:
                continue
            if (nr, nc) in blocked and (nr, nc) != (target_r, target_c):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, path + [(dr, dc)]))
    return None


def _delta_to_dirname(dr, dc):
    for name, (sr, sc) in _DIR_DELTAS.items():
        if (sr, sc) == (dr, dc):
            return name
    return None


def _simulate_nibbles(snake, direction, apples, n, moves):
    """Simulate snake game with given moves. snake is a list of (r, c) with
    head first. direction is the initial direction string. apples is a list
    of (r, c). Returns (success, final_apples_eaten).
    success means: all apples eaten, no wall/self collision.
    Convention: snake[0] = head. After move, head shifts; if head lands on
    apple, snake grows (tail not popped). Else tail is popped before
    collision check (so the tail cell is freed in same step).
    """
    body = deque(snake)
    apples_remaining = set(apples)
    cur_dir = direction
    for mv in moves:
        if mv not in _DIR_DELTAS:
            return False, len(apples) - len(apples_remaining)
        # Disallow instant reversal (180-degree turn)
        if mv == _OPPOSITE.get(cur_dir):
            return False, len(apples) - len(apples_remaining)
        dr, dc = _DIR_DELTAS[mv]
        head_r, head_c = body[0]
        new_r, new_c = head_r + dr, head_c + dc
        # Wall collision
        if not (0 <= new_r < n and 0 <= new_c < n):
            return False, len(apples) - len(apples_remaining)
        # Eats apple?
        ate = (new_r, new_c) in apples_remaining
        # Determine new body: prepend head, pop tail unless ate
        if ate:
            apples_remaining.remove((new_r, new_c))
            body.appendleft((new_r, new_c))
            # No tail pop on apple
        else:
            # Pop tail first, then check self-collision
            body.pop()
            if (new_r, new_c) in body:
                return False, len(apples) - len(apples_remaining)
            body.appendleft((new_r, new_c))
        cur_dir = mv
    return len(apples_remaining) == 0, len(apples) - len(apples_remaining)


def _solve_with_order(snake, direction, apples_order, n):
    """Given a fixed apple-eating order, greedily BFS each leg. Returns
    (moves, final_snake, final_dir) on success or None on failure."""
    cur_snake = list(snake)
    cur_dir = direction
    all_moves = []
    for target in apples_order:
        head = cur_snake[0]
        blocked = set(cur_snake)
        blocked.discard(target)
        forbid = _DIR_DELTAS[_OPPOSITE[cur_dir]]
        path = _bfs_path(head[0], head[1], target[0], target[1], n,
                          blocked, forbid_first_dir=forbid)
        if path is None:
            return None
        move_names = [_delta_to_dirname(dr, dc) for dr, dc in path]
        # Simulate to update snake state (and verify no self-collision)
        body = deque(cur_snake)
        cd = cur_dir
        # Track all remaining apples (so we don't accidentally eat next-up
        # apple before its turn). For our solver, target is the only apple
        # in this leg.
        apples_rem = {target}
        ok = True
        for mv in move_names:
            if mv == _OPPOSITE.get(cd):
                ok = False
                break
            dr, dc = _DIR_DELTAS[mv]
            hr, hc = body[0]
            nr, nc = hr + dr, hc + dc
            if not (0 <= nr < n and 0 <= nc < n):
                ok = False
                break
            ate = (nr, nc) in apples_rem
            if ate:
                apples_rem.remove((nr, nc))
                body.appendleft((nr, nc))
            else:
                body.pop()
                if (nr, nc) in body:
                    ok = False
                    break
                body.appendleft((nr, nc))
            cd = mv
        if not ok or apples_rem:
            return None
        cur_snake = list(body)
        cur_dir = cd
        all_moves.extend(move_names)
    return all_moves, cur_snake, cur_dir


def _gen_solvable_puzzle(n, n_apples, snake_len, rng,
                            max_attempts=400):
    """Generate a snake/apple configuration plus a solution move sequence.

    Strategy: repeatedly pick random snake position+direction and apple
    locations; for each layout, try a handful of apple orderings to find
    one that the greedy BFS solver can complete.
    """
    from itertools import permutations
    for attempt in range(max_attempts):
        # Place initial snake: pick random position + direction; lay tail
        head_r = rng.randint(1, n - 2)
        head_c = rng.randint(1, n - 2)
        direction = rng.choice(list(_DIR_DELTAS.keys()))
        opp_dr, opp_dc = _DIR_DELTAS[_OPPOSITE[direction]]
        snake = [(head_r, head_c)]
        valid = True
        for k in range(1, snake_len):
            tr = head_r + opp_dr * k
            tc = head_c + opp_dc * k
            if not (0 <= tr < n and 0 <= tc < n):
                valid = False
                break
            snake.append((tr, tc))
        if not valid:
            continue

        # Place apples
        free = [(r, c) for r in range(n) for c in range(n)
                 if (r, c) not in snake]
        if len(free) < n_apples:
            continue
        rng.shuffle(free)
        apples = free[:n_apples]

        # Try multiple orderings: full perms if small, else random samples
        if n_apples <= 4:
            orderings = list(permutations(apples))
            rng.shuffle(orderings)
        else:
            orderings = []
            for _ in range(40):
                ord_ = list(apples)
                rng.shuffle(ord_)
                orderings.append(tuple(ord_))

        for ordering in orderings[:30]:
            r = _solve_with_order(snake, direction, ordering, n)
            if r is None:
                continue
            all_moves, _, _ = r
            # Sanity: re-simulate full sequence on original
            ok, _ = _simulate_nibbles(snake, direction, apples, n, all_moves)
            if ok:
                return snake, direction, apples, all_moves
    return None


class NibblesSnakeGameQA(StandaloneVisualEnv):
    ENV_NAME = "nibbles_snake_game"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the move sequence (lower-case directions, "
        "space-separated) directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode — small grid, 1 apple, ask for first move only.
        if level == 0:
            return {"n": 5, "n_apples": 1, "snake_len": 2, "probe": True}
        if level == 1:
            return {"n": 5, "n_apples": 1, "snake_len": 2, "probe": True}
        if level == 2:
            return {"n": 6, "n_apples": 1, "snake_len": 2, "probe": True}
        if level <= 4:
            return {"n": 7, "n_apples": 2, "snake_len": 2, "probe": False}
        if level <= 6:
            return {"n": 8, "n_apples": 3, "snake_len": 2, "probe": False}
        if level <= 8:
            return {"n": 9, "n_apples": 4, "snake_len": 3, "probe": False}
        return {"n": 10, "n_apples": 5, "snake_len": 3, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 6079 + level * 47 + 31)

        result = _gen_solvable_puzzle(n, cfg["n_apples"], cfg["snake_len"],
                                          rng)
        if result is None:
            return None
        snake, direction, apples, moves = result

        # Stash for verify
        self._n = n
        self._snake = snake
        self._direction = direction
        self._apples = apples
        self._n_total_apples = len(apples)
        self._probe_mode = cfg.get("probe", False)

        snake_str = " ".join(f"({r},{c})" for r, c in snake)
        apples_str = " ".join(f"({r},{c})" for r, c in apples)

        if self._probe_mode:
            # PROBE: first move from full solution. Compute set of equivalent
            # first moves: any direction D s.t. there exists ANY shortest-
            # length valid solution starting with D.
            self._probe_first_dirs = self._compute_first_optimal_dirs(
                snake, direction, apples, n, len(moves))
            # Fallback: at minimum, the actual first move from the canonical
            # solution must work.
            if not self._probe_first_dirs:
                self._probe_first_dirs = {moves[0]}
            valid_dirs = sorted(self._probe_first_dirs)
            ans_str = rng.choice(valid_dirs)
            sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
            question = _PROBE_TEMPLATES[sidx].format(
                n=n, snake_str=snake_str, direction=direction,
                apples_str=apples_str,
            )
        else:
            ans_str = " ".join(moves)
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(
                n=n, snake_str=snake_str, direction=direction,
                apples_str=apples_str, n_apples=len(apples),
            )

        img = self._render(snake, direction, apples, n, rng)
        return question, ans_str, img

    def _compute_first_optimal_dirs(self, snake, direction, apples, n, optimal_len):
        """For each candidate first direction D (not the reverse of `direction`),
        check if there's a solution of length `optimal_len` starting with D.
        Returns set of valid first directions.
        """
        good = set()
        for first_d in _DIR_DELTAS.keys():
            if first_d == _OPPOSITE.get(direction):
                continue
            # Try this first move; if it ate an apple immediately, that may
            # also be valid. We just BFS: simulate one step, then check if the
            # remaining (apples - already eaten) can be eaten in optimal_len-1
            # more moves using greedy BFS solver. Reuse _solve_with_order.
            dr, dc = _DIR_DELTAS[first_d]
            head_r, head_c = snake[0]
            new_r, new_c = head_r + dr, head_c + dc
            if not (0 <= new_r < n and 0 <= new_c < n):
                continue
            # check no body collision after step
            body = list(snake)
            apples_set = set(apples)
            new_apples = set(apples_set)
            ate = (new_r, new_c) in apples_set
            if ate:
                new_apples.remove((new_r, new_c))
                new_body = [(new_r, new_c)] + body
            else:
                new_body = [(new_r, new_c)] + body[:-1]
                if (new_r, new_c) in new_body[1:]:
                    continue
            # Try greedy ordering for remaining apples.
            # If no apples remaining, we're done — first_d is good.
            if not new_apples:
                good.add(first_d)
                continue
            # Try each ordering of remaining apples
            from itertools import permutations
            remaining = list(new_apples)
            tried = 0
            for ordering in permutations(remaining):
                tried += 1
                if tried > 20:
                    break
                r = _solve_with_order(new_body, first_d, ordering, n)
                if r is not None:
                    rest_moves, _, _ = r
                    if 1 + len(rest_moves) <= optimal_len + 2:  # allow some slack
                        good.add(first_d)
                        break
        return good

    # ----------------------------------------------------------------- #
    def _render(self, snake, direction, apples, n, rng) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0"])
        fig, ax = plt.subplots(figsize=(0.65 * (n + 2), 0.65 * (n + 2)),
                                dpi=140)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        # Grid cells with row/col labels
        for r in range(n):
            for c in range(n):
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                              facecolor="#fafafa",
                                              edgecolor="#bbb",
                                              linewidth=0.6))
                ax.text(c + 0.05, n - 0.05 - r, f"{r},{c}",
                         ha="left", va="top", fontsize=6.5, color="#bbb")
        # Apples
        for (ar, ac) in apples:
            ax.add_patch(plt.Circle((ac + 0.5, n - 0.5 - ar), 0.32,
                                       facecolor="#e74c3c",
                                       edgecolor="#922b21", linewidth=1.2,
                                       zorder=4))
            ax.text(ac + 0.5, n - 0.5 - ar, "A", ha="center", va="center",
                     fontsize=14, fontweight="bold", color="#ffffff",
                     zorder=5)
        # Snake body (tail to head)
        for idx, (sr, sc) in enumerate(snake[::-1]):
            color = "#27ae60" if idx < len(snake) - 1 else "#196f3d"
            ax.add_patch(plt.Rectangle((sc + 0.1, n - 0.9 - sr), 0.8, 0.8,
                                          facecolor=color, edgecolor="#0e3517",
                                          linewidth=1.5, zorder=3))
        # Mark head with H + arrow showing direction
        head_r, head_c = snake[0]
        ax.text(head_c + 0.5, n - 0.5 - head_r, "H",
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 color="#ffffff", zorder=6)
        dr, dc = _DIR_DELTAS[direction]
        ax.annotate("", xy=(head_c + 0.5 + dc * 0.45,
                              n - 0.5 - head_r - dr * 0.45),
                     xytext=(head_c + 0.5, n - 0.5 - head_r),
                     arrowprops=dict(arrowstyle="->", color="#fff",
                                       lw=2.0), zorder=7)
        # Title
        ax.set_title(
            f"Nibbles  •  dir={direction}  •  {len(apples)} apple(s)",
            fontsize=11, color="#1d3557", pad=4)
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                     facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ----------------------------------------------------------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Normalize: lowercase, take all direction tokens
        text = predicted.lower()
        tokens = re.findall(r"\b(up|down|left|right)\b", text)
        if not tokens:
            return False
        # PROBE MODE: check first direction is in valid first-move set.
        if getattr(self, "_probe_mode", False):
            return tokens[0] in getattr(self, "_probe_first_dirs", set())
        ok, _ = _simulate_nibbles(list(self._snake), self._direction,
                                      list(self._apples), self._n, tokens)
        return ok


if __name__ == "__main__":
    env = NibblesSnakeGameQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer!r}")
