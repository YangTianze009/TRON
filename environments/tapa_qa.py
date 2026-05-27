"""
Tapa-style ring-clue puzzle.

2026-05-05 R5 P4: REWRITTEN with single-cell probe MCQ at L0-L4 to give
4B training a learnable signal. Full solve at L5-L9 constrained to small
4x4 grid (D=1 sample style).

Rules (standard Tapa):
  - NxN grid; some cells are CLUE cells (containing one or more digits),
    others are EMPTY (`.`).
  - Fill each empty cell with B (filled/black) or leave unfilled (white).
    Clue cells are never filled.
  - Around each CLUE cell (its 8-cell ring of king-move neighbors, ignoring
    out-of-grid), the contiguous runs of B-cells must match the clue's
    multiset of digits. The ring is treated as a CYCLIC sequence of up to
    8 positions but only positions that lie inside the grid are considered.
  - Globally:
      * No 2x2 sub-grid is entirely B.
      * All B cells form a single 4-connected (orthogonal) region.

Per-level layout:
  L0/L1: single-cell probe MCQ — small 3x3 grid with most cells decided
         (visible as black/white), ask whether ONE highlighted (?) cell is
         Black or White. 4 options (2 contentful + 2 trap-style).
  L2/L3: 4x4 grid, single ? cell, 4-MCQ (B/W + Cannot determine + No correct).
  L4   : 4x4 grid, 2 unknown ? cells, ask one specific. 5-MCQ with 15% trap.
  L5-L7: 4x4 D=1 full-solve text answer (existing format).
  L8/L9: 4x4 D=1 full-solve as MCQ candidate-grid + raised trap rate (30-40%).
"""
import random
import re
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ----------------------------------------------------------------------- #
# Probe MCQ templates (L0-L4)
# ----------------------------------------------------------------------- #
_PROBE_TEMPLATES = [
    "Look at the {n}x{n} ring-clue puzzle below.\n\n"
    "### Game Rules:\n"
    "1. Fill some empty cells black (B); leave the rest white (W).\n"
    "2. Each clue cell's digits = run-lengths of consecutive black cells in its 8-cell ring (in-grid only).\n"
    "3. No 2x2 sub-grid is entirely black.\n"
    "4. All black cells must form one orthogonally-connected region.\n"
    "5. Clue cells themselves are never filled.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Already-decided cells: {known}\n\n"
    "### Question:\n"
    "Should the highlighted (?) cell at (row {r}, col {c}) be Black or White?\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "{n}x{n} ring-clue puzzle.\n\n"
    "### Game Rules:\n"
    "- Fill some empty cells black (B); the rest stay white.\n"
    "- Each clue's digits = lengths of consecutive black runs in its 8-neighbour ring.\n"
    "- No 2x2 fully black; black cells form one connected region.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Known cells: {known}\n\n"
    "### Question:\n"
    "What is the correct color of the (?) cell at (row {r}, col {c})?\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "Ring-clue puzzle ({n}x{n}). Each clue cell's digits give the run-lengths of black cells in its 8-cell ring; no 2x2 fully black; black cells connected.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Already known: {known}\n\n"
    "### Question:\n"
    "Decide the color of the highlighted (?) cell at (row {r}, col {c}).\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",
]


# ----------------------------------------------------------------------- #
# Full-solve text templates (L5-L7)
# ----------------------------------------------------------------------- #
_TEMPLATES = [
    "Your task is to solve the {n}x{n} ring-clue puzzle below.\n\n"
    "### Game Rules:\n"
    "1. Fill some empty cells black; leave the rest white.\n"
    "2. Each clue cell's digits = run-lengths of consecutive black cells in the 8-cell ring around it.\n"
    "3. No 2x2 region may be entirely filled.\n"
    "4. All filled cells must form a single orthogonally-connected region.\n"
    "5. Clue cells themselves are never filled.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the filled (black) cells as comma-separated `(row,col)` pairs.\n"
    "Example: (0,3),(1,0),(1,1),(1,3),(2,1),(2,2),(2,3),(3,2)",

    "Solve the ring-clue puzzle below.\n\n"
    "### Game Rules:\n"
    "- Fill some empty cells black.\n"
    "- Each clue's digits = run-lengths of black cells in its 8-cell ring.\n"
    "- No 2x2 fully filled; all black cells must be connected orthogonally.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n}, 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output black cells as `(r,c),(r,c),...`.",

    "Your task is to solve the {n}x{n} ring-clue puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard ring-clue rules: clue digits are 8-ring run-lengths of black cells; no 2x2 fully black; black region is connected.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output black cells.",
]


def _tapa_state_text(n, clues):
    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            if (r, c) in clues:
                row.append("".join(str(d) for d in clues[(r, c)]))
            else:
                row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)


_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _ring_runs(grid, n, r, c):
    seq = []
    for dr, dc in _RING:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            seq.append("X")
        else:
            seq.append("B" if grid[nr][nc] == "B" else ".")
    if "X" in seq:
        runs = []
        i = 0
        while i < len(seq):
            if seq[i] == "B":
                j = i
                while j < len(seq) and seq[j] == "B":
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
        return sorted(runs)
    else:
        if "B" not in seq:
            return [0]
        if "." in seq:
            start = seq.index(".")
            seq2 = seq[start:] + seq[:start]
            runs = []
            i = 0
            while i < len(seq2):
                if seq2[i] == "B":
                    j = i
                    while j < len(seq2) and seq2[j] == "B":
                        j += 1
                    runs.append(j - i)
                    i = j
                else:
                    i += 1
            return sorted(runs)
        else:
            return [8]


def _all_b_connected(grid, n):
    bs = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == "B"]
    if not bs:
        return True
    from collections import deque
    seen = {bs[0]}
    q = deque([bs[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in seen \
                    and grid[nr][nc] == "B":
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(bs)


def _no_2x2_all_b(grid, n):
    for r in range(n - 1):
        for c in range(n - 1):
            if grid[r][c] == "B" and grid[r + 1][c] == "B" \
                    and grid[r][c + 1] == "B" and grid[r + 1][c + 1] == "B":
                return False
    return True


def _validate_tapa(grid, clues, n):
    for (r, c), expected in clues.items():
        runs = _ring_runs(grid, n, r, c)
        if runs != sorted(expected):
            return False
    if not _no_2x2_all_b(grid, n):
        return False
    if not _all_b_connected(grid, n):
        return False
    return True


def _gen_tapa_problem(n, num_clues, rng, max_attempts=200):
    for _ in range(max_attempts):
        grid = [["."] * n for _ in range(n)]
        bs = []
        for r in range(n):
            for c in range(n):
                if rng.random() < 0.4:
                    grid[r][c] = "B"
                    bs.append((r, c))
        if not bs:
            continue
        if not _no_2x2_all_b(grid, n):
            continue
        if not _all_b_connected(grid, n):
            continue
        non_b = [(r, c) for r in range(n) for c in range(n) if grid[r][c] != "B"]
        if len(non_b) < num_clues:
            continue
        rng.shuffle(non_b)
        clues = {}
        for cell in non_b[:num_clues]:
            r, c = cell
            runs = _ring_runs(grid, n, r, c)
            clues[(r, c)] = runs
        return clues, grid
    return None


def _solve_tapa(n, clues, limit=1):
    """Backtracking solver returning up to `limit` solutions as a list."""
    clue_set = set(clues.keys())
    cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in clue_set]
    grid = [["." for _ in range(n)] for _ in range(n)]
    found = []

    def partial_2x2_ok(r, c):
        for dr in (-1, 0):
            for dc in (-1, 0):
                tr, tc = r + dr, c + dc
                if 0 <= tr < n - 1 and 0 <= tc < n - 1:
                    cells2 = [grid[tr][tc], grid[tr + 1][tc],
                              grid[tr][tc + 1], grid[tr + 1][tc + 1]]
                    if all(x == "B" for x in cells2):
                        return False
        return True

    def backtrack(idx):
        if len(found) >= limit:
            return
        if idx == len(cells):
            for cl, exp in clues.items():
                if _ring_runs(grid, n, cl[0], cl[1]) != sorted(exp):
                    return
            if not _all_b_connected(grid, n):
                return
            found.append([row[:] for row in grid])
            return
        r, c = cells[idx]
        for v in ("B", "."):
            grid[r][c] = v
            ok = True
            if v == "B" and not partial_2x2_ok(r, c):
                ok = False
            if ok:
                backtrack(idx + 1)
            grid[r][c] = "."
            if len(found) >= limit:
                return

    backtrack(0)
    return found


class TapaQA(StandaloneVisualEnv):
    ENV_NAME = "tapa"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0/L1: single ?, 3x3 grid with most cells decided
        if level <= 1:
            return {"level": level, "n": 3, "num_clues": 3,
                    "mode": "probe", "n_unknown": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        # L2/L3: 4x4 grid, single ? probe (need more clues for uniqueness)
        if level <= 3:
            return {"level": level, "n": 4, "num_clues": 5,
                    "mode": "probe", "n_unknown": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        # L4: 4x4 grid, 2 ?, 5-MCQ + 15% trap
        if level == 4:
            return {"level": level, "n": 4, "num_clues": 5,
                    "mode": "probe", "n_unknown": 2,
                    "use_5_options": True, "trap_rate": 0.15}
        # L5-L7: full-solve 4x4 D=1 text
        if level <= 7:
            return {"level": level, "n": 4, "num_clues": 5,
                    "mode": "full", "use_5_options": False,
                    "trap_rate": 0.0}
        # L8/L9: full-solve 4x4 with raised trap, MCQ wrapper
        return {"level": level, "n": 4, "num_clues": 5,
                "mode": "full", "use_5_options": True,
                "trap_rate": 0.40 if level == 9 else 0.30}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        num_clues = cfg["num_clues"]
        rng = random.Random((self.seed or 0) * 6133 + level * 47 + 13)
        mode = cfg["mode"]

        for _ in range(80):
            res = _gen_tapa_problem(n, num_clues, rng, max_attempts=100)
            if res is None:
                continue
            clues, full_grid = res
            # Need UNIQUE solution so probe answer / full-solve answer is
            # unambiguous.
            solutions = _solve_tapa(n, clues, limit=2)
            if not solutions:
                continue
            if len(solutions) > 1:
                continue
            sol = solutions[0]
            self._n = n
            self._clues = clues
            self._solution = sol
            self._mcq_mode = (mode == "probe") or cfg.get("use_5_options", False)

            if mode == "probe":
                return self._build_probe_question(cfg, rng, sol, clues, n, level)
            else:
                return self._build_full_question(cfg, rng, sol, clues, n, level)
        return None

    # ------------------------------------------------------------------ #
    # L0-L4 probe MCQ
    # ------------------------------------------------------------------ #
    def _build_probe_question(self, cfg, rng, sol, clues, n, level):
        # Pick non-clue cells to hide (?)
        non_clue = [(r, c) for r in range(n) for c in range(n)
                    if (r, c) not in clues]
        if not non_clue:
            return None
        n_unknown = cfg["n_unknown"]
        if len(non_clue) < n_unknown:
            return None
        hidden = list(rng.sample(non_clue, n_unknown))
        target = hidden[0]
        target_r, target_c = target
        is_black = sol[target_r][target_c] == "B"
        true_color = "Black" if is_black else "White"

        # Show known cells (color of all decided cells except hidden)
        known_pairs = []
        for (r, c) in non_clue:
            if (r, c) in hidden:
                continue
            color = "B" if sol[r][c] == "B" else "W"
            known_pairs.append(((r, c), color))
        known_str = (", ".join(f"({r},{c})={col}" for (r, c), col in known_pairs)
                     or "(none)")

        # Build option set: always include Black, White; "Cannot be determined"
        # is informational distractor (always wrong since we ensure uniqueness).
        # 4-option layout: A. Black B. White C. Cannot be determined D. No correct
        # 5-option: A. Black B. White C. Cannot be determined D. <other> E. No correct
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]

        # Build options list (text)
        if use_5:
            # 5-options: include both colors + "Cannot be determined" + a duplicate or "Both" + "No correct"
            # Make 4 informational + 1 "No correct"
            opt_texts_pool = ["Black", "White", "Cannot be determined",
                              "Both Black and White are valid"]
            rng.shuffle(opt_texts_pool)
            opt_texts = opt_texts_pool[:4]
            opt_texts.append("No correct answer")
            true_text = true_color
            use_trap = (cfg.get("trap_rate", 0.0) > 0
                        and rng.random() < cfg["trap_rate"])
            if use_trap:
                # Remove correct from options
                opt_texts = [t for t in opt_texts[:4] if t != true_text]
                # Pad if needed
                pad_pool = ["Either Black or White",
                            "Depends on neighbors",
                            "Insufficient information"]
                rng.shuffle(pad_pool)
                pad_idx = 0
                while len(opt_texts) < 4:
                    if pad_pool[pad_idx] not in opt_texts:
                        opt_texts.append(pad_pool[pad_idx])
                    pad_idx = (pad_idx + 1) % len(pad_pool)
                rng.shuffle(opt_texts)
                opt_texts.append("No correct answer")
                correct_letter = opt_letters[-1]
            else:
                # Ensure correct (Black or White) is in opt_texts[:4]
                if true_text not in opt_texts:
                    opt_texts[0] = true_text  # force-include
                # Find letter
                correct_letter = opt_letters[opt_texts.index(true_text)]
        else:
            # 4-option: A. Black, B. White, C. Cannot be determined, D. No correct
            opt_texts = ["Black", "White", "Cannot be determined",
                         "No correct answer"]
            # randomize order of first 3
            first3 = opt_texts[:3]
            rng.shuffle(first3)
            opt_texts = first3 + ["No correct answer"]
            correct_letter = opt_letters[opt_texts.index(true_color)]

        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
        state_text = _tapa_state_text(n, clues)
        question = _PROBE_TEMPLATES[sidx].format(
            n=n, state=state_text, known=known_str,
            r=target_r, c=target_c,
            options=opt_lines, letter_str=letter_str,
        )
        self._probe_target = target
        self._probe_color = true_color
        self._known_cells = {(r, c): ("B" if sol[r][c] == "B" else "W")
                              for (r, c) in non_clue if (r, c) not in hidden}
        img = self._render(n, clues,
                           known_cells=self._known_cells,
                           highlight_cell=target)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # L5-L9 full solve / MCQ wrapper
    # ------------------------------------------------------------------ #
    def _build_full_question(self, cfg, rng, sol, clues, n, level):
        if cfg.get("use_5_options", False):
            return self._build_full_mcq_question(cfg, rng, sol, clues, n, level)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_text = _tapa_state_text(n, clues)
        filled = [(r, c) for r in range(n) for c in range(n) if sol[r][c] == "B"]
        ans_str = ",".join(f"({r},{c})" for r, c in sorted(filled))
        question = _TEMPLATES[sidx].format(n=n, state=state_text)
        img = self._render(n, clues)
        self._mcq_mode = False
        return question, ans_str, img

    def _build_full_mcq_question(self, cfg, rng, sol, clues, n, level):
        """Build an MCQ where each option is a candidate full-grid solution
        (compact text representation)."""
        # Canonical answer: list of black cells sorted
        true_filled = sorted([(r, c) for r in range(n) for c in range(n)
                              if sol[r][c] == "B"])

        def _seq_to_str(filled):
            return ",".join(f"({r},{c})" for r, c in sorted(filled))

        # Build distractors by mutating one or two cells in the solution
        # (toggle non-clue cell's status). We avoid producing a valid solution
        # since the puzzle is unique.
        clue_set = set(clues.keys())
        non_clue = [(r, c) for r in range(n) for c in range(n)
                    if (r, c) not in clue_set]

        distractors_set = set()
        attempts = 0
        while len(distractors_set) < 6 and attempts < 200:
            attempts += 1
            new_grid = [row[:] for row in sol]
            # Mutate 1 or 2 cells
            n_mutations = rng.choice([1, 2])
            mutate_cells = rng.sample(non_clue, n_mutations)
            for (r, c) in mutate_cells:
                new_grid[r][c] = "." if new_grid[r][c] == "B" else "B"
            new_filled = tuple(sorted([(r, c) for r in range(n) for c in range(n)
                                        if new_grid[r][c] == "B"]))
            if new_filled == tuple(true_filled):
                continue
            distractors_set.add(new_filled)

        distractors = [list(d) for d in distractors_set]
        if len(distractors) < 3:
            return None

        use_5 = cfg["use_5_options"]
        n_distractor = 3 if use_5 else 2
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        use_trap = (cfg.get("trap_rate", 0.0) > 0
                    and rng.random() < cfg["trap_rate"])
        if use_trap and len(distractors) >= n_distractor + 1:
            options_data = distractors[: n_distractor + 1]
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            options_data = [true_filled] + distractors[:n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(true_filled)]

        opt_texts = [_seq_to_str(s) for s in options_data]
        opt_texts.append("No correct answer")
        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        state_text = _tapa_state_text(n, clues)
        stem = (
            f"Look at the {n}x{n} ring-clue puzzle.\n\n"
            "### Game Rules:\n"
            "1. Fill some empty cells black; the rest stay white.\n"
            "2. Each clue cell's digits = run-lengths of consecutive black cells in its 8-cell ring.\n"
            "3. No 2x2 fully black; black cells form one connected region.\n"
            "4. Clue cells are never filled.\n\n"
            "### Coordinate System:\n"
            "- (row, col), 0-indexed.\n\n"
            "### Current Puzzle State:\n"
            f"{state_text}\n\n"
            "### Question:\n"
            "Which option lists the black cells in the unique solution?\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        img = self._render(n, clues)
        self._mcq_mode = True
        return stem, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, n, clues, known_cells=None, highlight_cell=None) -> Image.Image:
        fig, ax = plt.subplots(figsize=(0.95 * (n + 1), 0.95 * (n + 1)), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for r in range(n):
            for c in range(n):
                if (r, c) in clues:
                    ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                               facecolor="#ffe4b5", edgecolor="#444",
                                               linewidth=1.2))
                    label = ",".join(str(d) for d in clues[(r, c)])
                    ax.text(c + 0.5, n - 0.5 - r, label,
                            ha="center", va="center", fontsize=12,
                            fontweight="bold", color="#222")
                elif highlight_cell is not None and (r, c) == highlight_cell:
                    ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                               facecolor="#fff5b1",
                                               edgecolor="#cc6600",
                                               linewidth=2.5))
                    ax.text(c + 0.5, n - 0.5 - r, "?",
                            ha="center", va="center", fontsize=22,
                            fontweight="bold", color="#cc6600")
                elif known_cells is not None and (r, c) in known_cells:
                    if known_cells[(r, c)] == "B":
                        ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                                   facecolor="#222",
                                                   edgecolor="#333",
                                                   linewidth=1.0))
                    else:
                        ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                                   facecolor="#ffffff",
                                                   edgecolor="#888",
                                                   linewidth=1.0))
                        ax.text(c + 0.5, n - 0.5 - r, "W",
                                ha="center", va="center", fontsize=10,
                                color="#888", fontweight="bold")
                else:
                    ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                               facecolor="#ffffff",
                                               edgecolor="#888",
                                               linewidth=0.8))
                # small coord text
                if (r, c) not in clues:
                    ax.text(c + 0.06, n - 0.06 - r, f"{r},{c}",
                            ha="left", va="top", fontsize=6, color="#bbb")
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ------------------------------------------------------------------ #
    # Answer checking — dispatches based on mode
    # ------------------------------------------------------------------ #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        if getattr(self, "_mcq_mode", False):
            return super()._check_answer(predicted, ground_truth)
        # Full-solve text mode: parse `(r,c)` pairs, validate
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", predicted)
        try:
            filled = set((int(r), int(c)) for r, c in pairs)
        except ValueError:
            return False
        n = self._n
        grid = [["." for _ in range(n)] for _ in range(n)]
        for r, c in filled:
            if not (0 <= r < n and 0 <= c < n):
                return False
            if (r, c) in self._clues:
                return False
            grid[r][c] = "B"
        return _validate_tapa(grid, self._clues, n)
