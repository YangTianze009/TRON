"""
Kakuro-style cross-sums puzzle.

2026-05-05 R5 P4: REWRITTEN with single-cell probe MCQ at L0-L4 to give 4B
training a learnable signal. Full solve at L5-L9 constrained to D=1 trivial
3x3 puzzles per the benchmark D=1 sample style.

Rules (standard Kakuro):
  - The grid contains BLACK and WHITE cells.
  - A "run" is a maximal horizontal/vertical strip of WHITE cells.
  - A BLACK cell may carry up to two clue numbers:
      * (down, right) — the down clue is the required sum for the white run
        going downward starting from the cell directly below.
      * The right clue is the required sum for the white run going right,
        starting from the cell directly to the right.
  - Fill each WHITE cell with a digit 1..9 such that:
      * each run sums to its labelled total, and
      * no digit repeats within a single run.

Per-level layout:
  L0/L1: single-cell probe MCQ — small 3x3 grid with all-but-one white cell
         filled in; ask for the digit in the highlighted (?) cell. 4 options
         (digits) including "No correct answer".
  L2/L3: 2-cell probe MCQ — two unknown white cells, ask one. Same MCQ form.
  L4   : 2-cell probe with 5-option MCQ (added "No correct answer" trap 15%).
  L5-L7: full-solve mode — 3x3 trivial grid (D=1 style), text answer.
  L8/L9: full-solve 3x3 D=1, but with 30-40% E="No correct answer" trap
         in MCQ companion. (Uses MCQ wrapper because full-solve has near-zero
         signal at 4B.)
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
    "Look at the {n}x{n} cross-sums puzzle below.\n\n"
    "### Game Rules:\n"
    "1. Fill every white cell with a digit 1-9.\n"
    "2. Each maximal horizontal/vertical run of white cells must sum to its labelled clue.\n"
    "3. No digit may repeat within a single run.\n"
    "4. Black cells with `right=R` constrain the run of white cells to the right; `down=D` constrain the run below.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed; row 0 at top.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Already-filled white cells: {known}\n\n"
    "### Question:\n"
    "What digit goes in the highlighted (?) cell at (row {r}, col {c})?\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "{n}x{n} cross-sums puzzle.\n\n"
    "### Game Rules:\n"
    "- White cells hold digits 1-9.\n"
    "- Each white run sums to its clue total without repeating digits.\n"
    "- Black cells carry `right` clues (sum of the run to the right) and/or `down` clues (sum of the run below).\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Known white-cell values: {known}\n\n"
    "### Question:\n"
    "Which digit must be placed in the (?) cell at (row {r}, col {c})?\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "Cross-sums puzzle ({n}x{n}). Each white-cell run sums to the labelled clue and contains no repeated digit (1-9).\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Already known: {known}\n\n"
    "### Question:\n"
    "Determine the digit in the highlighted (?) cell at (row {r}, col {c}).\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",
]


# ----------------------------------------------------------------------- #
# Full-solve templates (L5-L9): solve the whole small puzzle
# ----------------------------------------------------------------------- #
_TEMPLATES = [
    "Your task is to solve the {n}x{n} cross-sums puzzle described below.\n\n"
    "### Game Rules:\n"
    "1. Fill every white cell with a digit 1-9.\n"
    "2. Each maximal horizontal/vertical run of white cells must sum to its labelled clue.\n"
    "3. No digit may repeat within a single run.\n"
    "4. Black cells with `right=R` constrain the run of white cells to the right; `down=D` constrain the run below.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Provide white-cell assignments as space-separated `(row,col):digit` pairs (0-indexed).\n"
    "Example: (0,2):5 (1,0):1 (1,1):2 (1,2):8 (2,2):5",

    "Solve the cross-sums puzzle below.\n\n"
    "### Game Rules:\n"
    "- White cells hold digits 1-9.\n"
    "- Each white run sums to its clue total without repeating digits.\n"
    "- Black cells carry `right` clues (sum of the run to the right) and/or `down` clues (sum of the run below).\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output `(r,c):d` pairs separated by spaces.",

    "Your task is to solve the {n}x{n} cross-sums puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard cross-sums: white runs sum to their clues, no digit repeats within a run, digits are 1-9.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output digit assignments as `(r,c):d` pairs.",
]


def _kakuro_state_text(n, black, h_runs, v_runs, h_targets, v_targets):
    right_at = {}
    down_at = {}
    for run, target in zip(h_runs, h_targets):
        clue = run["clue_at"]
        right_at[clue] = target
    for run, target in zip(v_runs, v_targets):
        clue = run["clue_at"]
        down_at[clue] = target
    rows = []
    for r in range(n):
        cells = []
        for c in range(n):
            if not black[r][c]:
                cells.append("[W]")
            else:
                parts = []
                if (r, c) in right_at:
                    parts.append(f"right={right_at[(r, c)]}")
                if (r, c) in down_at:
                    parts.append(f"down={down_at[(r, c)]}")
                if parts:
                    cells.append(f"[B:{','.join(parts)}]")
                else:
                    cells.append("[.]")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _build_runs(black: List[List[bool]], n: int) -> Tuple[List[Dict], List[Dict]]:
    """Identify horizontal and vertical runs in the grid."""
    h_runs = []
    v_runs = []
    for r in range(n):
        c = 0
        while c < n:
            if not black[r][c]:
                start = c
                while c < n and not black[r][c]:
                    c += 1
                length = c - start
                if length >= 2:
                    if start == 0:
                        return None, None
                    h_runs.append({
                        "cells": [(r, k) for k in range(start, c)],
                        "clue_at": (r, start - 1),
                        "dir": "right",
                    })
            else:
                c += 1
    for c in range(n):
        r = 0
        while r < n:
            if not black[r][c]:
                start = r
                while r < n and not black[r][c]:
                    r += 1
                length = r - start
                if length >= 2:
                    if start == 0:
                        return None, None
                    v_runs.append({
                        "cells": [(k, c) for k in range(start, r)],
                        "clue_at": (start - 1, c),
                        "dir": "down",
                    })
            else:
                r += 1
    return h_runs, v_runs


def _try_fill_runs(n, black, h_runs, v_runs, rng) -> Optional[Dict[Tuple[int, int], int]]:
    cells = [(r, c) for r in range(n) for c in range(n) if not black[r][c]]
    cell_to_h = {}
    cell_to_v = {}
    for ri, run in enumerate(h_runs):
        for cell in run["cells"]:
            cell_to_h[cell] = ri
    for vi, run in enumerate(v_runs):
        for cell in run["cells"]:
            cell_to_v[cell] = vi
    used_h = [set() for _ in h_runs]
    used_v = [set() for _ in v_runs]
    assign = {}

    def backtrack(idx):
        if idx == len(cells):
            return True
        r, c = cells[idx]
        digits = list(range(1, 10))
        rng.shuffle(digits)
        hi = cell_to_h.get((r, c))
        vi = cell_to_v.get((r, c))
        for d in digits:
            if hi is not None and d in used_h[hi]:
                continue
            if vi is not None and d in used_v[vi]:
                continue
            assign[(r, c)] = d
            if hi is not None:
                used_h[hi].add(d)
            if vi is not None:
                used_v[vi].add(d)
            if backtrack(idx + 1):
                return True
            del assign[(r, c)]
            if hi is not None:
                used_h[hi].discard(d)
            if vi is not None:
                used_v[vi].discard(d)
        return False

    if backtrack(0):
        return assign
    return None


def _count_solutions(n, black, h_runs, v_runs,
                     run_targets_h, run_targets_v, limit=2):
    cells = [(r, c) for r in range(n) for c in range(n) if not black[r][c]]
    cell_to_h = {}
    cell_to_v = {}
    for ri, run in enumerate(h_runs):
        for cell in run["cells"]:
            cell_to_h[cell] = ri
    for vi, run in enumerate(v_runs):
        for cell in run["cells"]:
            cell_to_v[cell] = vi
    used_h = [set() for _ in h_runs]
    used_v = [set() for _ in v_runs]
    sum_h = [0 for _ in h_runs]
    sum_v = [0 for _ in v_runs]
    remain_h = [len(r["cells"]) for r in h_runs]
    remain_v = [len(r["cells"]) for r in v_runs]
    found = [0]

    def backtrack(idx):
        if found[0] >= limit:
            return
        if idx == len(cells):
            for ri in range(len(h_runs)):
                if sum_h[ri] != run_targets_h[ri]:
                    return
            for vi in range(len(v_runs)):
                if sum_v[vi] != run_targets_v[vi]:
                    return
            found[0] += 1
            return
        r, c = cells[idx]
        hi = cell_to_h.get((r, c))
        vi = cell_to_v.get((r, c))
        for d in range(1, 10):
            if hi is not None and d in used_h[hi]:
                continue
            if vi is not None and d in used_v[vi]:
                continue
            ok = True
            if hi is not None:
                avail = [x for x in range(1, 10) if x not in used_h[hi] and x != d]
                target = run_targets_h[hi]
                need = target - sum_h[hi] - d
                rem = remain_h[hi] - 1
                if rem == 0:
                    if need != 0:
                        ok = False
                else:
                    if len(avail) < rem:
                        ok = False
                    else:
                        avail_sorted = sorted(avail)
                        min_p = sum(avail_sorted[:rem])
                        max_p = sum(avail_sorted[-rem:])
                        if not (min_p <= need <= max_p):
                            ok = False
            if ok and vi is not None:
                target = run_targets_v[vi]
                need = target - sum_v[vi] - d
                rem = remain_v[vi] - 1
                if rem == 0:
                    if need != 0:
                        ok = False
                else:
                    avail = [x for x in range(1, 10) if x not in used_v[vi] and x != d]
                    if len(avail) < rem:
                        ok = False
                    else:
                        avail_sorted = sorted(avail)
                        min_p = sum(avail_sorted[:rem])
                        max_p = sum(avail_sorted[-rem:])
                        if not (min_p <= need <= max_p):
                            ok = False
            if not ok:
                continue
            if hi is not None:
                used_h[hi].add(d)
                sum_h[hi] += d
                remain_h[hi] -= 1
            if vi is not None:
                used_v[vi].add(d)
                sum_v[vi] += d
                remain_v[vi] -= 1
            backtrack(idx + 1)
            if hi is not None:
                used_h[hi].discard(d)
                sum_h[hi] -= d
                remain_h[hi] += 1
            if vi is not None:
                used_v[vi].discard(d)
                sum_v[vi] -= d
                remain_v[vi] += 1
            if found[0] >= limit:
                return

    backtrack(0)
    return found[0]


class KakuroQA(StandaloneVisualEnv):
    ENV_NAME = "kakuro"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0/L1: single-cell probe (one ?), 4-MCQ
        if level <= 1:
            return {"level": level, "n": 3, "black_density": 0.0,
                    "mode": "probe", "n_unknown": 1, "use_5_options": False,
                    "trap_rate": 0.0}
        # L2/L3: 2-cell probe, 4-MCQ
        if level <= 3:
            return {"level": level, "n": 3, "black_density": 0.0,
                    "mode": "probe", "n_unknown": 2, "use_5_options": False,
                    "trap_rate": 0.0}
        # L4: 2-cell probe, 5-MCQ with 15% trap
        if level == 4:
            return {"level": level, "n": 3, "black_density": 0.0,
                    "mode": "probe", "n_unknown": 2, "use_5_options": True,
                    "trap_rate": 0.15}
        # L5-L7: full-solve, small 3x3 D=1 puzzle (text format)
        if level <= 7:
            return {"level": level, "n": 3, "black_density": 0.0,
                    "mode": "full", "use_5_options": False,
                    "trap_rate": 0.0}
        # L8/L9: full-solve, but raised trap rate via MCQ-companion mode
        # (we keep full-solve text but optionally make it a 5-MCQ).
        return {"level": level, "n": 3, "black_density": 0.0,
                "mode": "full", "use_5_options": True,
                "trap_rate": 0.40 if level == 9 else 0.30}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 7919 + level * 23 + 5)
        mode = cfg["mode"]

        # We use a small constrained grid layout matching the D=1 sample style:
        # for n=3, top row + left col are black (clue holders). This gives
        # a consistent, easily-renderable, easily-solvable puzzle.
        for attempt in range(150):
            black = [[False] * n for _ in range(n)]
            # Top row & left col always black
            for c in range(n):
                black[0][c] = True
            for r in range(n):
                black[r][0] = True
            # Optionally: extra black cells (only at higher difficulty)
            if cfg.get("black_density", 0.0) > 0.0:
                for r in range(1, n):
                    for c in range(1, n):
                        if rng.random() < cfg["black_density"]:
                            black[r][c] = True
            white_cells = [(r, c) for r in range(n) for c in range(n)
                           if not black[r][c]]
            if len(white_cells) < 2:
                continue
            h_runs, v_runs = _build_runs(black, n)
            if h_runs is None:
                continue
            cells_in_runs = set()
            for run in h_runs + v_runs:
                for cell in run["cells"]:
                    cells_in_runs.add(cell)
            if any(cell not in cells_in_runs for cell in white_cells):
                continue
            if not h_runs and not v_runs:
                continue
            label_locs = set()
            for run in h_runs + v_runs:
                label_locs.add(run["clue_at"])
            if not all(black[r][c] for r, c in label_locs):
                continue
            assign = _try_fill_runs(n, black, h_runs, v_runs, rng)
            if assign is None:
                continue
            run_targets_h = [sum(assign[c] for c in run["cells"]) for run in h_runs]
            run_targets_v = [sum(assign[c] for c in run["cells"]) for run in v_runs]
            # Probe and full modes need unique solution so the queried
            # digit/full-grid is unambiguous.
            n_sols = _count_solutions(
                n, black, h_runs, v_runs,
                run_targets_h, run_targets_v, limit=2,
            )
            if n_sols != 1:
                continue
            # Save state
            self._n = n
            self._black = black
            self._h_runs = h_runs
            self._v_runs = v_runs
            self._h_targets = run_targets_h
            self._v_targets = run_targets_v
            self._solution = assign
            self._mcq_mode = (mode == "probe") or cfg.get("use_5_options", False)

            if mode == "probe":
                return self._build_probe_question(
                    cfg, rng, assign, h_runs, v_runs,
                    run_targets_h, run_targets_v, black, n, level,
                )
            else:
                return self._build_full_question(
                    cfg, rng, assign, h_runs, v_runs,
                    run_targets_h, run_targets_v, black, n, level,
                )
        return None

    # ------------------------------------------------------------------ #
    # L0-L4 probe MCQ
    # ------------------------------------------------------------------ #
    def _build_probe_question(self, cfg, rng, assign, h_runs, v_runs,
                              run_targets_h, run_targets_v, black, n, level):
        white_cells = list(assign.keys())
        if not white_cells:
            return None
        n_unknown = cfg["n_unknown"]
        if len(white_cells) < n_unknown:
            return None
        # Pick the cells that will be hidden ("?"). Pick deterministically
        # by sampling from rng.
        hidden = list(rng.sample(white_cells, n_unknown))
        target = hidden[0]  # Q asks specifically about this one
        target_r, target_c = target
        target_d = assign[target]
        known_pairs = sorted((k, v) for k, v in assign.items() if k not in hidden)
        known_str = (", ".join(f"({r},{c})={d}" for (r, c), d in known_pairs)
                     or "(none)")

        # Build distractors from digits 1-9 that respect each run that contains
        # the target (sum upper bound + can't equal an already-known digit in
        # that run).
        plausible = self._distractor_candidates(
            target, h_runs, v_runs, run_targets_h, run_targets_v, assign, hidden,
        )
        # Exclude the true digit
        plausible = [d for d in plausible if d != target_d]
        rng.shuffle(plausible)

        # Build option set
        use_5 = cfg["use_5_options"]
        n_distractor = 3 if use_5 else 2
        # Pad distractor pool with random other digits if needed
        all_digits = [d for d in range(1, 10) if d != target_d]
        rng.shuffle(all_digits)
        for d in all_digits:
            if len(plausible) >= n_distractor:
                break
            if d not in plausible:
                plausible.append(d)
        distractor_digits = plausible[:n_distractor]

        # Letters
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        # Trap path: correct = last letter ("No correct answer")
        use_trap = (cfg.get("trap_rate", 0.0) > 0
                    and rng.random() < cfg["trap_rate"])
        if use_trap and len(plausible) >= n_distractor + 1:
            # All numeric options are wrong; "No correct answer" wins
            options_data = plausible[: n_distractor + 1]
            options_data = options_data[: (4 if use_5 else 3)]
            rng.shuffle(options_data)
            correct_letter = opt_letters[-1]
        else:
            options_data = [target_d] + distractor_digits
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(target_d)]

        # Format option lines: digit options first, then "No correct answer"
        opt_texts = [str(d) for d in options_data]
        opt_texts.append("No correct answer")
        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
        state_text = _kakuro_state_text(n, black, h_runs, v_runs,
                                        run_targets_h, run_targets_v)
        question = _PROBE_TEMPLATES[sidx].format(
            n=n, state=state_text, known=known_str,
            r=target_r, c=target_c,
            options=opt_lines, letter_str=letter_str,
        )
        self._probe_target = target
        self._probe_target_digit = target_d
        self._hidden_cells = hidden
        self._known_cells = {k: v for k, v in assign.items() if k not in hidden}

        img = self._render(n, black, h_runs, v_runs,
                            run_targets_h, run_targets_v,
                            known_assign=self._known_cells,
                            highlight_cell=target)
        return question, correct_letter, img

    def _distractor_candidates(self, target, h_runs, v_runs,
                                run_targets_h, run_targets_v, assign, hidden):
        """Return digits 1-9 that look plausible — i.e. would NOT instantly
        violate the run constraints. We're generous: any digit that doesn't
        repeat with a known same-run digit AND keeps the partial sum within
        achievable range is included."""
        candidates = []
        for d in range(1, 10):
            ok = True
            # Check all horiz runs containing target
            for ri, run in enumerate(h_runs):
                if target in run["cells"]:
                    used = []
                    for cell in run["cells"]:
                        if cell != target and cell not in hidden:
                            used.append(assign[cell])
                    if d in used:
                        ok = False
                        break
                    # Sum check (very loose)
                    fixed = sum(used)
                    other_unknowns = [c for c in run["cells"]
                                      if c != target and c in hidden]
                    n_other = len(other_unknowns)
                    target_sum = run_targets_h[ri]
                    need = target_sum - fixed - d
                    if n_other == 0:
                        if need != 0:
                            ok = False
                            break
                    else:
                        # need at least 1 + (n_other-1)*1 = n_other; max 9*n_other
                        if not (n_other <= need <= 9 * n_other):
                            ok = False
                            break
            if not ok:
                continue
            for vi, run in enumerate(v_runs):
                if target in run["cells"]:
                    used = []
                    for cell in run["cells"]:
                        if cell != target and cell not in hidden:
                            used.append(assign[cell])
                    if d in used:
                        ok = False
                        break
                    fixed = sum(used)
                    other_unknowns = [c for c in run["cells"]
                                      if c != target and c in hidden]
                    n_other = len(other_unknowns)
                    target_sum = run_targets_v[vi]
                    need = target_sum - fixed - d
                    if n_other == 0:
                        if need != 0:
                            ok = False
                            break
                    else:
                        if not (n_other <= need <= 9 * n_other):
                            ok = False
                            break
            if ok:
                candidates.append(d)
        return candidates

    # ------------------------------------------------------------------ #
    # L5-L9 full-solve / MCQ wrapper
    # ------------------------------------------------------------------ #
    def _build_full_question(self, cfg, rng, assign, h_runs, v_runs,
                             run_targets_h, run_targets_v, black, n, level):
        # L8/L9 use MCQ "candidate full-grid" wrapper with E="No correct"
        # trap. L5-L7 use plain text answer.
        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_text = _kakuro_state_text(n, black, h_runs, v_runs,
                                        run_targets_h, run_targets_v)
        # Generate "true" assign as canonical answer
        true_assign_str = " ".join(f"({r},{c}):{d}"
                                   for (r, c), d in sorted(assign.items()))

        if cfg.get("use_5_options", False):
            return self._build_full_mcq_question(
                cfg, rng, assign, h_runs, v_runs,
                run_targets_h, run_targets_v, black, n, level,
            )

        question = _TEMPLATES[sidx].format(n=n, state=state_text)
        img = self._render(n, black, h_runs, v_runs,
                            run_targets_h, run_targets_v)
        # full-solve text mode — store full assignment as ground truth
        self._mcq_mode = False
        return question, true_assign_str, img

    def _build_full_mcq_question(self, cfg, rng, assign, h_runs, v_runs,
                                 run_targets_h, run_targets_v, black, n, level):
        """Build a 5-option MCQ where each option is a candidate full-grid
        assignment summary (sequence of digits). Trap rate ~30-40%."""
        # Build the canonical answer summary as digits in row-major white-cell
        # order. This is a compact representation the model can compare.
        white_cells = sorted([(r, c) for r in range(n) for c in range(n)
                              if not black[r][c]])
        true_seq = [assign[c] for c in white_cells]

        def _seq_to_str(seq):
            # Render as "(r,c)=d, ..." in white-cell order
            return ", ".join(f"({r},{c})={d}"
                             for (r, c), d in zip(white_cells, seq))

        # Build distractors by mutating one or two cells of true_seq.
        distractors = []
        attempts = 0
        while len(distractors) < 5 and attempts < 100:
            attempts += 1
            new_seq = true_seq[:]
            # Swap two cells' digits OR change one cell's digit
            if rng.random() < 0.5 and len(new_seq) >= 2:
                i, j = rng.sample(range(len(new_seq)), 2)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            else:
                i = rng.randrange(len(new_seq))
                d_new = rng.randint(1, 9)
                while d_new == new_seq[i]:
                    d_new = rng.randint(1, 9)
                new_seq[i] = d_new
            if new_seq == true_seq:
                continue
            if new_seq in distractors:
                continue
            distractors.append(new_seq)

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
            options_data = [true_seq] + distractors[:n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(true_seq)]

        opt_texts = [_seq_to_str(s) for s in options_data]
        opt_texts.append("No correct answer")
        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_text = _kakuro_state_text(n, black, h_runs, v_runs,
                                        run_targets_h, run_targets_v)
        stem = (
            f"Look at the {n}x{n} cross-sums puzzle.\n\n"
            "### Game Rules:\n"
            "1. Fill every white cell with a digit 1-9.\n"
            "2. Each white run sums to its labelled clue.\n"
            "3. No digit may repeat within a single run.\n\n"
            "### Coordinate System:\n"
            "- (row, col), 0-indexed.\n\n"
            "### Current Puzzle State:\n"
            f"{state_text}\n\n"
            "### Question:\n"
            "Which option correctly solves the puzzle?\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        img = self._render(n, black, h_runs, v_runs,
                            run_targets_h, run_targets_v)
        self._mcq_mode = True
        return stem, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, n, black, h_runs, v_runs, h_targets, v_targets,
                known_assign=None, highlight_cell=None) -> Image.Image:
        fig, ax = plt.subplots(figsize=(0.95 * n + 1, 0.95 * n + 1), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        right_labels = {}
        down_labels = {}
        for run, tgt in zip(h_runs, h_targets):
            right_labels[run["clue_at"]] = tgt
        for run, tgt in zip(v_runs, v_targets):
            down_labels[run["clue_at"]] = tgt

        for r in range(n):
            for c in range(n):
                if black[r][c]:
                    ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                               facecolor="#222", edgecolor="#222",
                                               linewidth=1.0))
                    rt = right_labels.get((r, c))
                    dn = down_labels.get((r, c))
                    if rt is not None or dn is not None:
                        ax.plot([c, c + 1], [n - r, n - 1 - r],
                                color="#fff", linewidth=1.0)
                        if rt is not None:
                            ax.text(c + 0.78, n - 0.25 - r, str(rt),
                                    ha="center", va="center", color="white",
                                    fontsize=10, fontweight="bold")
                        if dn is not None:
                            ax.text(c + 0.22, n - 0.75 - r, str(dn),
                                    ha="center", va="center", color="white",
                                    fontsize=10, fontweight="bold")
                else:
                    is_highlight = (highlight_cell is not None
                                    and (r, c) == highlight_cell)
                    if is_highlight:
                        ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                                   facecolor="#fff5b1",
                                                   edgecolor="#cc6600",
                                                   linewidth=2.5))
                        ax.text(c + 0.5, n - 0.5 - r, "?",
                                ha="center", va="center", fontsize=22,
                                fontweight="bold", color="#cc6600")
                    elif known_assign is not None and (r, c) in known_assign:
                        ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                                   facecolor="#e8f4ff",
                                                   edgecolor="#888",
                                                   linewidth=1.0))
                        ax.text(c + 0.5, n - 0.5 - r,
                                str(known_assign[(r, c)]),
                                ha="center", va="center", fontsize=18,
                                fontweight="bold", color="#0d3b66")
                    else:
                        ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                                   facecolor="#fafafa",
                                                   edgecolor="#888",
                                                   linewidth=1.0))
                    # small coord text in any case (top-left corner)
                    ax.text(c + 0.05, n - 0.05 - r, f"{r},{c}",
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
        # Fast-path guard: compute_score's fast-path sets _answer but skips
        # generate(), so _n / _black / _mcq_mode etc. aren't set. Raising
        # AttributeError triggers reward_function.py's slow-path fallback
        # (re-run generate to populate state, then retry verify).
        if not hasattr(self, "_n"):
            raise AttributeError("kakuro state not initialized; need slow-path generate")
        # MCQ mode: ground truth is a single letter A-E
        if getattr(self, "_mcq_mode", False):
            return super()._check_answer(predicted, ground_truth)
        # Full-solve text mode: parse pairs (r,c):d
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\d+)", predicted)
        if not pairs:
            return False
        try:
            assign = {(int(r), int(c)): int(d) for r, c, d in pairs}
        except ValueError:
            return False
        n = self._n
        white_cells = [(r, c) for r in range(n) for c in range(n)
                       if not self._black[r][c]]
        for cell in white_cells:
            if cell not in assign:
                return False
            v = assign[cell]
            if not (1 <= v <= 9):
                return False
        for cell in assign:
            if cell not in white_cells:
                return False
        for run, tgt in zip(self._h_runs, self._h_targets):
            digits = [assign[c] for c in run["cells"]]
            if sum(digits) != tgt:
                return False
            if len(set(digits)) != len(digits):
                return False
        for run, tgt in zip(self._v_runs, self._v_targets):
            digits = [assign[c] for c in run["cells"]]
            if sum(digits) != tgt:
                return False
            if len(set(digits)) != len(digits):
                return False
        return True
