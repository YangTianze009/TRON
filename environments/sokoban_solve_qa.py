"""
Sokoban (structured-puzzle style).

reference qids studied (sokoban, total 30): idx=1021, 1022, 1027, 1028, 1033,
1034, 1039, 1040, 1045, 1046 (verbatim per `design notes` §sokoban,
lines 817-871). TSV format confirmed via reference dataset:
  - question_text: "Your task is to solve the Sokoban puzzle...\n
                    Red circles=player, Blue squares=boxes, Yellow circles=targets..."
  - answer: lowercase direction words space-separated, e.g. "up up right down down"

Self-contained: inlines a reverse-walk Sokoban generator + BFS solver (adapted
from `RLVE_Visual/environments/sokoban_visual.py`, which itself was already
free of RLVE/Gym deps). NO `from RLVE`, `from Gym`, `sys.path.insert`.

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 5x5 grid, 1 box, ≤ 5 moves
  L9 within solvable range: 8x8 grid, 2 boxes, ≤ 25 moves
"""
import random
from collections import deque
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_DIRS = {
    "up":    (-1, 0),
    "down":  (+1, 0),
    "left":  (0, -1),
    "right": (0, +1),
}


# 2026-05-04: simplified L0/L1 (was 5% too-hard).
# L0/L1 PROBE templates: ask for FIRST MOVE only (not full sequence).
_PROBE_TEMPLATES = [
    "Your task is to determine the FIRST MOVE in the {N}x{N} Sokoban puzzle below:\n\n"
    "### Game Rules:\n"
    "1. The player (`@`) can move one cell at a time: up, down, left, or right.\n"
    "2. Walls (`#`) block movement.\n"
    "3. The player can PUSH a box (`$`) to the next cell if that cell is empty (or a goal). The player CANNOT pull boxes.\n"
    "4. The puzzle is solved when every box rests on a goal (`.`).\n\n"
    "### Coordinate System:\n"
    "- Symbols: `#`=wall, `@`=player, `$`=box, `.`=goal, ` `=empty floor.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "What is the FIRST move that begins the optimal solution? Output a single direction word (`up`, `down`, `left`, or `right`) inside <answer>...</answer>.",

    "Look at the Sokoban puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Sokoban — player (`@`) pushes boxes (`$`) onto goals (`.`). Walls (`#`) block. No pulling.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Determine the FIRST move of a solution. Output one word: up / down / left / right, inside <answer>...</answer>.",

    "Sokoban puzzle ({N}x{N}).\n\n"
    "### Game Rules:\n"
    "Standard Sokoban rules.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "What direction is the FIRST move of a valid solution? Output `up`, `down`, `left`, or `right` inside <answer>...</answer>.",
]


# structured 4-section templates. ASCII state in code-block style.
_TEMPLATES = [
    "Your task is to solve the {N}x{N} Sokoban puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. The player (`@`) can move one cell at a time: up, down, left, or right.\n"
    "2. Walls (`#`) block movement.\n"
    "3. The player can PUSH a box (`$`) to the next cell if that cell is empty (or a goal). The player CANNOT pull boxes.\n"
    "4. The puzzle is solved when every box rests on a goal (`.`).\n\n"
    "### Coordinate System:\n"
    "- Each row is a line; characters represent cells.\n"
    "- Symbols: `#`=wall, `@`=player, `$`=box, `.`=goal, ` `=empty floor.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Provide the move sequence as space-separated direction words (`up`, `down`, `left`, `right`) inside <answer>...</answer>.\n"
    "Example: <answer>up up right down down</answer>",

    "Solve the Sokoban puzzle described below.\n\n"
    "### Game Rules:\n"
    "- Move the player (`@`) one cell per step.\n"
    "- Pushing a box (`$`) into the next square is allowed if that square is empty or a goal; pulling is forbidden.\n"
    "- Walls (`#`) cannot be entered.\n"
    "- Goal: every box must end on a goal cell (`.`).\n\n"
    "### Coordinate System:\n"
    "- The grid is a rectangle of characters. Each character is one cell.\n"
    "- Direction names refer to grid rows (up = row -1, down = row +1) and columns (left = col -1, right = col +1).\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Output a sequence of moves as lowercase direction words separated by spaces, wrapped in <answer>...</answer>.",

    "Your task is to solve the Sokoban puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Sokoban — push boxes (`$`) onto goals (`.`). Walls (`#`) block. The player (`@`) can push but not pull. All boxes must rest on goals.\n\n"
    "### Coordinate System:\n"
    "- Rows top-to-bottom, columns left-to-right.\n"
    "- Symbols: `#` wall, `@` player, `$` box, `.` goal, ` ` empty.\n\n"
    "### Current Puzzle State:\n"
    "```\n{state}\n```\n\n"
    "### Output Format:\n"
    "Return the solution as space-separated direction words (`up`/`down`/`left`/`right`) in <answer>...</answer>.",
]


def _grid_to_ascii(grid, N, player, boxes, targets):
    """Render the sokoban state as ASCII."""
    box_set = set(boxes)
    target_set = set(targets)
    rows = []
    for i in range(N):
        line = []
        for j in range(N):
            ch = " "
            if grid[i][j] == "#":
                ch = "#"
            elif (i, j) == player:
                ch = "@"
            elif (i, j) in box_set:
                ch = "$"
            elif (i, j) in target_set:
                ch = "."
            line.append(ch)
        rows.append("".join(line))
    return "\n".join(rows)


def _bfs_solve(grid: List[List[str]], N: int, player: Tuple[int, int],
               boxes: List[Tuple[int, int]],
               targets: List[Tuple[int, int]],
               max_states: int = 60_000) -> Optional[List[str]]:
    """Standard Sokoban BFS over (player, frozenset(boxes)) states.
    Returns a list of direction words or None. Adapted from sokoban_visual.py."""
    targets_set = frozenset(targets)
    boxes_frozen = frozenset(boxes)
    start = (player, boxes_frozen)
    if boxes_frozen == targets_set:
        return []

    visited = {start}
    queue = deque([(start, [])])
    states = 0

    while queue and states < max_states:
        (pr, bxs), path = queue.popleft()
        states += 1
        for d, (dr, dc) in _DIRS.items():
            nr, nc = pr[0] + dr, pr[1] + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            if grid[nr][nc] == "#":
                continue
            new_boxes = bxs
            if (nr, nc) in bxs:
                br, bc = nr + dr, nc + dc
                if not (0 <= br < N and 0 <= bc < N):
                    continue
                if grid[br][bc] == "#":
                    continue
                if (br, bc) in bxs:
                    continue
                new_boxes = frozenset(b if b != (nr, nc) else (br, bc)
                                      for b in bxs)
            state = ((nr, nc), new_boxes)
            if state in visited:
                continue
            visited.add(state)
            new_path = path + [d]
            if new_boxes == targets_set:
                return new_path
            queue.append((state, new_path))
    return None


def _generate_sokoban(N: int, num_boxes: int, max_solution_len: int,
                      rng: random.Random,
                      retry: int = 60):
    """Generate a solvable Sokoban puzzle by backwards-pull random walk.

    Returns (grid, player_pos, box_positions, target_positions, solution_words)
    or None on failure.
    """
    for _ in range(retry):
        grid = [["#"] * N for _ in range(N)]
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                grid[i][j] = "."
        interior = [(i, j) for i in range(1, N - 1) for j in range(1, N - 1)]
        # Sparse internal walls
        n_walls = rng.randint(0, max(0, len(interior) // 6))
        wall_candidates = rng.sample(interior, min(n_walls, len(interior)))
        remaining = [p for p in interior if p not in wall_candidates]
        if len(remaining) < 1 + num_boxes:
            continue
        rng.shuffle(remaining)
        player = remaining[0]
        target_positions = remaining[1:1 + num_boxes]
        box_positions = list(target_positions)
        for w in wall_candidates:
            if w != player and w not in box_positions:
                grid[w[0]][w[1]] = "#"

        # Reverse walk to scramble
        p = list(player)
        bxs = [list(b) for b in box_positions]
        for _ in range(max_solution_len * 3):
            d = rng.choice(list(_DIRS.keys()))
            dr, dc = _DIRS[d]
            nr, nc = p[0] + dr, p[1] + dc
            if not (0 < nr < N - 1 and 0 < nc < N - 1):
                continue
            if grid[nr][nc] == "#":
                continue
            if [nr, nc] in bxs:
                continue
            behind_r, behind_c = p[0] - dr, p[1] - dc
            for bi, b in enumerate(bxs):
                if b == [behind_r, behind_c]:
                    bxs[bi] = list(p)
                    break
            p = [nr, nc]

        final_player = tuple(p)
        final_boxes = [tuple(b) for b in bxs]
        # Skip if start == solved
        if frozenset(final_boxes) == frozenset(target_positions):
            continue
        sol = _bfs_solve(grid, N, final_player, final_boxes, target_positions)
        if sol is not None and 1 <= len(sol) <= max_solution_len:
            return grid, final_player, final_boxes, target_positions, sol
    return None


class SokobanSolveQA(StandaloneVisualEnv):
    ENV_NAME = "sokoban_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Skip narrative. Output the move sequence directly inside "
        "`<answer>...</answer>` tags as space-separated direction words."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04: simplified L0/L1 (was 5% too-hard) — PROBE mode (first
        # move only) at L0/L1 with very short solutions.
        # L0..L1: 5x5, 1 box, ≤3 moves; PROBE = first move only.
        # L2..L3: 6x6, 1 box, ≤8
        # L4..L5: 6x6, 2 boxes, ≤12
        # L6..L7: 7x7, 2 boxes, ≤16
        # L8..L9: 8x8, 2 boxes, ≤22
        if level <= 1:
            N, nb, maxlen = 5, 1, 3
            return {"N": N, "num_boxes": nb, "max_solution_len": maxlen,
                    "level": level, "probe": True}
        elif level <= 3:
            N, nb, maxlen = 6, 1, 7 + level
        elif level <= 5:
            N, nb, maxlen = 6, 2, 10 + level
        elif level <= 7:
            N, nb, maxlen = 7, 2, 13 + level
        else:
            N, nb, maxlen = 8, 2, 18 + level - 8
        return {"N": N, "num_boxes": nb, "max_solution_len": maxlen,
                "level": level, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((seed or 0) * 991 + level * 31 + 13)

        result = _generate_sokoban(cfg["N"], cfg["num_boxes"],
                                   cfg["max_solution_len"], rng)
        if result is None:
            return None
        grid, player, boxes, targets, sol = result

        self._grid = grid
        self._N = cfg["N"]
        self._player = player
        self._boxes = boxes
        self._targets = targets
        self._sol_len = len(sol)
        self._probe_mode = cfg.get("probe", False)

        # 2026-05-04: simplified L0/L1 (was 5% too-hard) — PROBE: first move.
        if self._probe_mode:
            ans_str = sol[0]  # first direction word
            self._probe_first_move = ans_str
            sidx = (seed or 0) % len(_PROBE_TEMPLATES)
            state_ascii = _grid_to_ascii(grid, cfg["N"], player, boxes, targets)
            question = _PROBE_TEMPLATES[sidx].format(
                N=cfg["N"], state=state_ascii,
            )
            img = self._render(grid, cfg["N"], player, boxes, targets)
            return question, ans_str, img

        sidx = (seed or 0) % len(_TEMPLATES)
        state_ascii = _grid_to_ascii(grid, cfg["N"], player, boxes, targets)
        question = _TEMPLATES[sidx].format(N=cfg["N"], state=state_ascii)
        ans_str = " ".join(sol)
        img = self._render(grid, cfg["N"], player, boxes, targets)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, grid, N, player, boxes, targets) -> Image.Image:
        cell_size = max(40, 400 // N)
        fig_w = N * cell_size / 80 + 1.2
        fig_h = N * cell_size / 80 + 1.2
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=100)

        target_set = set(targets)
        box_set = set(boxes)

        for i in range(N):
            for j in range(N):
                if grid[i][j] == "#":
                    color = "#4a4a4a"
                elif (i, j) in target_set:
                    color = "#ffe0e0"
                else:
                    color = "#f5f0e1"
                rect = patches.Rectangle(
                    (j, N - 1 - i), 1, 1,
                    linewidth=0.5, edgecolor="#d4c5a0", facecolor=color,
                )
                ax.add_patch(rect)
                if (i, j) in target_set:
                    diamond = patches.RegularPolygon(
                        (j + 0.5, N - 0.5 - i), 4, radius=0.2,
                        facecolor="#ff6b6b", edgecolor="white",
                        linewidth=1, alpha=0.7,
                    )
                    ax.add_patch(diamond)
                if (i, j) in box_set:
                    on_target = (i, j) in target_set
                    box_color = "#2ecc71" if on_target else "#8b6914"
                    bx = patches.FancyBboxPatch(
                        (j + 0.15, N - 0.85 - i), 0.7, 0.7,
                        boxstyle="round,pad=0.05",
                        facecolor=box_color, edgecolor="#2c3e50", linewidth=1.5,
                    )
                    ax.add_patch(bx)
                    ax.text(j + 0.5, N - 0.5 - i, "B",
                            ha="center", va="center",
                            fontsize=max(8, 16 - N), fontweight="bold",
                            color="white")

        pi, pj = player
        circle = patches.Circle(
            (pj + 0.5, N - 0.5 - pi), 0.32,
            facecolor="#3498db", edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(circle)
        ax.text(pj + 0.5, N - 0.5 - pi, "P", ha="center", va="center",
                fontsize=max(8, 16 - N), fontweight="bold", color="white")

        label_font = max(7, 14 - N // 3)
        for i in range(N):
            ax.text(-0.25, N - 0.5 - i, str(i), ha="right", va="center",
                    fontsize=label_font, color="#555555")
        for j in range(N):
            ax.text(j + 0.5, N + 0.25, str(j), ha="center", va="bottom",
                    fontsize=label_font, color="#555555")

        ax.set_xlim(-0.5, N + 0.3)
        ax.set_ylim(-0.5, N + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import re
        words = re.findall(r"\b(up|down|left|right|u|d|l|r)\b",
                           predicted.lower())
        if not words:
            return False
        letter_map = {"u": "up", "d": "down", "l": "left", "r": "right"}
        moves = [letter_map.get(w, w) for w in words]

        N = self._N
        grid = self._grid
        player = list(self._player)
        boxes = [list(b) for b in self._boxes]
        targets = set(self._targets)

        # 2026-05-04: simplified L0/L1 (was 5% too-hard) — PROBE first move.
        # Accept any first move that begins SOME valid solution (not just the
        # one we cached). Simulate one step and BFS-check from that state.
        if getattr(self, "_probe_mode", False):
            first = moves[0]
            dr, dc = _DIRS[first]
            nr, nc = player[0] + dr, player[1] + dc
            if not (0 <= nr < N and 0 <= nc < N) or grid[nr][nc] == "#":
                return False
            new_boxes = [list(b) for b in boxes]
            box_idx = None
            for i, b in enumerate(new_boxes):
                if b == [nr, nc]:
                    box_idx = i
                    break
            if box_idx is not None:
                br, bc = nr + dr, nc + dc
                if not (0 <= br < N and 0 <= bc < N) or grid[br][bc] == "#":
                    return False
                if any(b == [br, bc] for b in new_boxes):
                    return False
                new_boxes[box_idx] = [br, bc]
            new_player = (nr, nc)
            new_box_tuples = [tuple(b) for b in new_boxes]
            # Check: is there any solution from (new_player, new_boxes)?
            sol = _bfs_solve(grid, N, new_player, new_box_tuples,
                             list(targets))
            return sol is not None

        for m in moves:
            dr, dc = _DIRS[m]
            nr, nc = player[0] + dr, player[1] + dc
            if not (0 <= nr < N and 0 <= nc < N) or grid[nr][nc] == "#":
                return False
            box_idx = None
            for i, b in enumerate(boxes):
                if b == [nr, nc]:
                    box_idx = i
                    break
            if box_idx is not None:
                br, bc = nr + dr, nc + dc
                if not (0 <= br < N and 0 <= bc < N) or grid[br][bc] == "#":
                    return False
                if any(b == [br, bc] for b in boxes):
                    return False
                boxes[box_idx] = [br, bc]
            player = [nr, nc]
        box_set = set(tuple(b) for b in boxes)
        return box_set == targets
