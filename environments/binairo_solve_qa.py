"""
Binairo (Takuzu) puzzle.

2026-05-05 R5 B5A: REWRITTEN — pure single-letter MCQ A-D / A-E.

Rules:
  * NxN grid (N even). Each row and column must contain equal numbers of
    0s and 1s.
  * No three consecutive identical digits in any row or column.
  * All rows are unique; all columns are unique.

Two task modes (selected by level):

  PROBE mode (L0-L4) — small partially-filled board with EXACTLY ONE
  highlighted blank cell (?). Ask: "what is the digit at the highlighted
  cell?". MCQ over {0, 1, Cannot be determined, No correct answer}.

  VALID mode (L5-L9) — small board with a "candidate completion" shown.
  Ask: "is the completion valid?". MCQ over {Valid, Invalid (row/col
  imbalance), Invalid (3-in-a-row), Invalid (duplicate row/col),
  No correct answer}. The candidate is either correct or violates exactly
  one of the listed rules (chosen so the wrong-rule label is the trap).

Per-level layout
  L0/L1 — PROBE, 4-MCQ. 4x4 grid, almost-full, single ? cell. Hint.
  L2/L3 — PROBE, 4-MCQ. 4x4 grid, 2 blanks but only 1 highlighted ?.
  L4    — PROBE, 4-MCQ. 4x4 grid, 3 blanks, ? cell.
  L5/L6 — VALID, 4-MCQ. 4x4 candidate completion to validate.
  L7    — VALID, 5-MCQ + 25% trap.
  L8    — VALID, 5-MCQ + 30% trap. 6x6.
  L9    — VALID, 5-MCQ + 40% trap. 6x6.
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ---------------------------------------------------------------------- #
# Solver / generator helpers
# ---------------------------------------------------------------------- #
def _is_partial_ok(grid: List[List[Optional[int]]], n: int) -> bool:
    half = n // 2
    for r in range(n):
        ones = sum(1 for x in grid[r] if x == 1)
        zeros = sum(1 for x in grid[r] if x == 0)
        if ones > half or zeros > half:
            return False
        for c in range(n - 2):
            a, b, d = grid[r][c], grid[r][c + 1], grid[r][c + 2]
            if a is not None and a == b == d:
                return False
    for c in range(n):
        col = [grid[r][c] for r in range(n)]
        ones = sum(1 for x in col if x == 1)
        zeros = sum(1 for x in col if x == 0)
        if ones > half or zeros > half:
            return False
        for r in range(n - 2):
            a, b, d = col[r], col[r + 1], col[r + 2]
            if a is not None and a == b == d:
                return False
    completed_rows = [tuple(grid[r]) for r in range(n)
                      if all(x is not None for x in grid[r])]
    if len(completed_rows) != len(set(completed_rows)):
        return False
    completed_cols = []
    for c in range(n):
        col = [grid[r][c] for r in range(n)]
        if all(x is not None for x in col):
            completed_cols.append(tuple(col))
    if len(completed_cols) != len(set(completed_cols)):
        return False
    return True


def _is_full_valid(grid: List[List[int]], n: int) -> bool:
    half = n // 2
    for r in range(n):
        if sum(1 for x in grid[r] if x == 1) != half:
            return False
        for c in range(n - 2):
            if grid[r][c] == grid[r][c + 1] == grid[r][c + 2]:
                return False
    for c in range(n):
        col = [grid[r][c] for r in range(n)]
        if sum(1 for x in col if x == 1) != half:
            return False
        for r in range(n - 2):
            if col[r] == col[r + 1] == col[r + 2]:
                return False
    rows = [tuple(grid[r]) for r in range(n)]
    cols = [tuple(grid[r][c] for r in range(n)) for c in range(n)]
    if len(set(rows)) != n:
        return False
    if len(set(cols)) != n:
        return False
    return True


def _check_violation(grid: List[List[int]], n: int) -> str:
    """Return one of: 'valid', 'imbalance', 'three_in_row', 'duplicate'."""
    half = n // 2
    for r in range(n):
        if sum(1 for x in grid[r] if x == 1) != half:
            return "imbalance"
    for c in range(n):
        col = [grid[r][c] for r in range(n)]
        if sum(1 for x in col if x == 1) != half:
            return "imbalance"
    for r in range(n):
        for c in range(n - 2):
            if grid[r][c] == grid[r][c + 1] == grid[r][c + 2]:
                return "three_in_row"
    for c in range(n):
        for r in range(n - 2):
            v0 = grid[r][c]
            v1 = grid[r + 1][c]
            v2 = grid[r + 2][c]
            if v0 == v1 == v2:
                return "three_in_row"
    rows = [tuple(grid[r]) for r in range(n)]
    cols = [tuple(grid[r][c] for r in range(n)) for c in range(n)]
    if len(set(rows)) != n:
        return "duplicate"
    if len(set(cols)) != n:
        return "duplicate"
    return "valid"


def _solve_binairo(givens: List[List[Optional[int]]], n: int,
                   limit: int = 2,
                   max_steps: int = 200000) -> List[List[List[int]]]:
    grid = [row[:] for row in givens]
    found = []
    steps = [0]

    def backtrack(idx):
        if len(found) >= limit:
            return
        steps[0] += 1
        if steps[0] > max_steps:
            return
        if idx == n * n:
            full = [[grid[r][c] for c in range(n)] for r in range(n)]
            if _is_full_valid(full, n):
                found.append(full)
            return
        r, c = divmod(idx, n)
        if grid[r][c] is not None:
            backtrack(idx + 1)
            return
        for v in [0, 1]:
            grid[r][c] = v
            if _is_partial_ok(grid, n):
                backtrack(idx + 1)
            grid[r][c] = None
            if len(found) >= limit:
                return

    backtrack(0)
    return found


def _gen_random_solution(n: int, rng: random.Random) -> Optional[List[List[int]]]:
    grid: List[List[Optional[int]]] = [[None] * n for _ in range(n)]

    def fill(idx):
        if idx == n * n:
            return _is_full_valid(grid, n)
        r, c = divmod(idx, n)
        vals = [0, 1]
        rng.shuffle(vals)
        for v in vals:
            grid[r][c] = v
            if _is_partial_ok(grid, n):
                if fill(idx + 1):
                    return True
            grid[r][c] = None
        return False

    if fill(0):
        return [row[:] for row in grid]
    return None


def _gen_unique_puzzle(n: int, target_blank: int,
                       rng: random.Random,
                       max_attempts: int = 30
                       ) -> Optional[Tuple[List[List[Optional[int]]],
                                            List[List[int]]]]:
    """Generate a Binairo puzzle with `target_blank` blanks; returns
    (givens, solution) or None."""
    for _ in range(max_attempts):
        sol = _gen_random_solution(n, rng)
        if sol is None:
            continue
        cells = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(cells)
        givens = [row[:] for row in sol]
        blank_count = 0
        for (r, c) in cells:
            if blank_count >= target_blank:
                break
            saved = givens[r][c]
            givens[r][c] = None
            sols = _solve_binairo(givens, n, limit=2, max_steps=80000)
            if len(sols) == 1:
                blank_count += 1
            else:
                givens[r][c] = saved
        if blank_count == target_blank:
            return givens, sol
    return None


def _make_violating_completion(sol: List[List[int]],
                                n: int,
                                violation_type: str,
                                rng: random.Random
                                ) -> Optional[List[List[int]]]:
    """Mutate `sol` minimally to create a violation of the requested type.
    Returns a new grid or None."""
    grid = [row[:] for row in sol]
    if violation_type == "imbalance":
        # Flip TWO cells in the same row that have different values to
        # create row imbalance (and possibly col imbalance too). One flip
        # always breaks both row & col counts.
        for _ in range(40):
            r = rng.randrange(n)
            cols = [c for c in range(n) if grid[r][c] == 0]
            if not cols:
                continue
            c = rng.choice(cols)
            grid[r][c] = 1
            # Confirm the resulting violation is imbalance (not something
            # else like 3-in-a-row).
            v = _check_violation(grid, n)
            if v == "imbalance":
                return grid
            # Revert and try again
            grid[r][c] = 0
        return None

    if violation_type == "three_in_row":
        # Find 2-in-a-row, then flip the next cell to match.
        for _ in range(60):
            r = rng.randrange(n)
            for c in range(n - 2):
                if grid[r][c] == grid[r][c + 1]:
                    target = grid[r][c]
                    if grid[r][c + 2] != target:
                        grid[r][c + 2] = target
                        # also need to adjust to keep row balance — flip
                        # a different cell of the row to compensate.
                        for cc in range(n):
                            if cc != c + 2 and grid[r][cc] != target:
                                grid[r][cc] = target ^ 1  # already opposite
                                # Wait: need to flip a cell from `1-target`
                                # to `target` to restore... Actually we
                                # already added one of `target` so now have
                                # one extra. Flip a different cell from
                                # `target` → `1 - target`.
                                break
                        # Compensate row balance
                        ones = sum(grid[r])
                        zeros = n - ones
                        if ones > n // 2:
                            # Flip a 1 (not at c+2) to 0
                            for cc in range(n):
                                if cc != c + 2 and grid[r][cc] == 1:
                                    grid[r][cc] = 0
                                    break
                        elif zeros > n // 2:
                            for cc in range(n):
                                if cc != c + 2 and grid[r][cc] == 0:
                                    grid[r][cc] = 1
                                    break
                        # Confirm violation_type is three_in_row
                        v = _check_violation(grid, n)
                        if v == "three_in_row":
                            return grid
                        # Revert
                        grid = [row[:] for row in sol]
        return None

    if violation_type == "duplicate":
        # Force two rows equal — copy row r0 into r1, then re-balance r1
        # by flipping enough cells to keep balance (won't be possible to
        # avoid all violations, so check).
        for _ in range(40):
            r0, r1 = rng.sample(range(n), 2)
            grid = [row[:] for row in sol]
            grid[r1] = list(grid[r0])
            # Also need to keep column balance — usually breaks. Accept
            # whatever violation is reported as long as it's "duplicate".
            v = _check_violation(grid, n)
            if v == "duplicate":
                return grid
        return None

    return None


# ---------------------------------------------------------------------- #
# Question stems
# ---------------------------------------------------------------------- #
_PROBE_STEM = [
    "The image shows a partially-filled {n}x{n} Binairo (Takuzu) puzzle. The cell marked with a red ? is blank. Following the Binairo rules (each row/column has equal 0s and 1s; no three consecutive identical digits; rows unique, columns unique), what digit goes in the highlighted (?) cell ( )?",
    "Look at the {n}x{n} Binairo puzzle. The red ? cell is blank. By the rules (balanced 0/1 per row & column, no 3-in-a-row, unique rows & columns), what value must the highlighted cell hold ( )?",
    "{n}x{n} Binairo puzzle below. The cell highlighted with a red ? must be filled. Following the standard rules (balanced rows/cols, no 3 same in a row, unique rows & columns), what is the correct digit ( )?",
]

_VALID_STEM = [
    "The image shows a {n}x{n} grid filled with 0s and 1s as a candidate completed Binairo (Takuzu) board. Following the rules (each row/column has equal 0s and 1s; no three consecutive identical digits in any row or column; all rows unique; all columns unique), is the candidate valid, and if not, which rule does it violate ( )?",
    "Look at the {n}x{n} candidate Binairo solution shown in the image. Is it valid by the rules (balanced 0/1 per row & column; no 3-in-a-row; unique rows; unique columns)? If invalid, which rule does it break ( )?",
    "{n}x{n} candidate Binairo board below. By the rules (row/col balance, no 3-in-a-row, unique rows, unique columns), classify the board ( ).",
]


class BinairoSolveQA(StandaloneVisualEnv):
    ENV_NAME = "binairo_solve"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Apply the rules cell-by-cell or row-by-row. "
        "Final answer in the format this prompt requested."
    )

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"mode": "probe", "n": 4, "blanks": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 3:
            return {"mode": "probe", "n": 4, "blanks": 2,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 4:
            return {"mode": "probe", "n": 4, "blanks": 3,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 6:
            return {"mode": "valid", "n": 4,
                    "use_5_options": False, "trap_rate": 0.0}
        if level == 7:
            return {"mode": "valid", "n": 4,
                    "use_5_options": True, "trap_rate": 0.25}
        # 2026-05-05 Phase A: bump L8/L9 trap_rate to 0.5 (env still saturated 95%).
        if level == 8:
            return {"mode": "valid", "n": 6,
                    "use_5_options": True, "trap_rate": 0.50}
        # L9
        return {"mode": "valid", "n": 6,
                "use_5_options": True, "trap_rate": 0.50}

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5099 + level * 47 + 13)
        for _ in range(15):
            try:
                if cfg["mode"] == "probe":
                    res = self._try_generate_probe(cfg, level, rng)
                else:
                    res = self._try_generate_valid(cfg, level, rng)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # PROBE mode
    # ------------------------------------------------------------------ #
    def _try_generate_probe(self, cfg: Dict, level: int,
                            rng: random.Random):
        n = cfg["n"]
        blanks = cfg["blanks"]
        result = _gen_unique_puzzle(n, blanks, rng)
        if result is None:
            return None
        givens, sol = result

        blank_cells = [(r, c) for r in range(n) for c in range(n)
                       if givens[r][c] is None]
        if not blank_cells:
            return None
        # Pick one to highlight
        target = rng.choice(blank_cells)
        target_value = sol[target[0]][target[1]]
        true_label = str(target_value)  # "0" or "1"

        # 4-MCQ option pool: 0 / 1 / Cannot be determined / Both possible
        # 5-MCQ adds "No correct answer" trap. Note: probe mode never uses
        # 5-MCQ in current configs (L0-L4 only), so wrong_pool size is OK.
        base_options = ["0", "1", "Cannot be determined",
                        "Both 0 and 1 are valid"]
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            wrong_pool = [o for o in base_options if o != true_label]
            rng.shuffle(wrong_pool)
            chosen = wrong_pool[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            wrong_pool = [o for o in base_options if o != true_label]
            rng.shuffle(wrong_pool)
            n_distractor = n_value - 1
            chosen = wrong_pool[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [true_label] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_label)]

        opt_lines = [f"{opt_letters[i]}. {options[i]}" for i in range(n_value)]
        if use_5:
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(_PROBE_STEM)
        stem = _PROBE_STEM[sidx].format(n=n)
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )
        if level <= 1:
            question += (
                "\nHint: count the 0s and 1s already in the highlighted "
                "cell's row (and column). The correct digit is whichever "
                "still has fewer than half — or whichever avoids a "
                "3-in-a-row run."
            )
        elif level <= 3:
            question += (
                "\nHint: each row/column must end with two 0s and two 1s "
                "(for 4x4). Use this and the no-3-in-a-row rule to deduce "
                "the highlighted cell."
            )

        img = self._render_grid(n, givens, highlight_cell=target)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # VALID mode
    # ------------------------------------------------------------------ #
    def _try_generate_valid(self, cfg: Dict, level: int,
                            rng: random.Random):
        n = cfg["n"]
        sol = _gen_random_solution(n, rng)
        if sol is None:
            return None

        # Decide what to display: 50% valid, 50% invalid (random rule)
        is_valid_choice = rng.random() < 0.5
        if is_valid_choice:
            displayed = [row[:] for row in sol]
            ground_truth = "Valid"
        else:
            violation = rng.choice(["imbalance", "three_in_row", "duplicate"])
            mutated = _make_violating_completion(sol, n, violation, rng)
            if mutated is None:
                return None
            displayed = mutated
            ground_truth_map = {
                "imbalance": "Invalid: row or column does not have equal 0s and 1s",
                "three_in_row": "Invalid: three consecutive identical digits in a row or column",
                "duplicate": "Invalid: a row or column is duplicated",
            }
            ground_truth = ground_truth_map[violation]

        # 4-MCQ base option pool (enough distinct labels to support 5-MCQ trap
        # mode where 4 wrongs must be drawn from `base_options - {ground_truth}`)
        base_options = [
            "Valid",
            "Invalid: row or column does not have equal 0s and 1s",
            "Invalid: three consecutive identical digits in a row or column",
            "Invalid: a row or column is duplicated",
            "Invalid: contains a digit other than 0 or 1",
            "Invalid: not square / wrong dimensions",
        ]
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = use_5 and (rng.random() < cfg["trap_rate"])
        if use_trap:
            wrong_pool = [o for o in base_options if o != ground_truth]
            rng.shuffle(wrong_pool)
            chosen = wrong_pool[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            wrong_pool = [o for o in base_options if o != ground_truth]
            rng.shuffle(wrong_pool)
            n_distractor = n_value - 1
            chosen = wrong_pool[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [ground_truth] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(ground_truth)]

        opt_lines = [f"{opt_letters[i]}. {options[i]}" for i in range(n_value)]
        if use_5:
            opt_lines.append(f"{opt_letters[-1]}. No correct answer")
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        sidx = (self.seed or 0) % len(_VALID_STEM)
        stem = _VALID_STEM[sidx].format(n=n)
        question = (
            f"{stem}\n"
            + "\n".join(opt_lines)
            + f"\nAnswer with a single letter {letter_str}."
        )

        # No level <=1 hint here (VALID mode starts at L5).
        # Display as a fully-filled grid (no highlight).
        givens = [[v for v in row] for row in displayed]
        img = self._render_grid(n, givens, highlight_cell=None,
                                style_fill=True)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_grid(self, n: int,
                     grid: List[List[Optional[int]]],
                     highlight_cell: Optional[Tuple[int, int]] = None,
                     style_fill: bool = False) -> Image.Image:
        # Larger figure for L0/L1 readability
        cell_in = 0.65
        fig_w = (n + 0.6) * cell_in + 0.5
        fig_h = (n + 0.6) * cell_in + 0.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for r in range(n):
            for c in range(n):
                v = grid[r][c]
                if v is None:
                    bg = "#f7f9fc"
                    txt = ""
                elif v == 0:
                    bg = "#dfe6e9"
                    txt = "0"
                else:
                    bg = "#74b9ff"
                    txt = "1"
                ax.add_patch(patches.Rectangle((c, n - 1 - r), 1, 1,
                                               linewidth=1.4,
                                               edgecolor="#2c3e50",
                                               facecolor=bg))
                if txt:
                    ax.text(c + 0.5, n - 1 - r + 0.5, txt,
                            ha="center", va="center",
                            fontsize=max(13, 26 - n * 2),
                            fontweight="bold", color="#2c3e50")

        # Highlight target with a red ? and red border
        if highlight_cell is not None:
            tr, tc = highlight_cell
            ax.add_patch(patches.Rectangle((tc, n - 1 - tr), 1, 1,
                                           linewidth=3.5,
                                           edgecolor="#c0392b",
                                           facecolor="#ffeaa7",
                                           alpha=0.85))
            ax.text(tc + 0.5, n - 1 - tr + 0.5, "?",
                    ha="center", va="center",
                    fontsize=max(18, 30 - n * 2),
                    fontweight="bold", color="#c0392b")

        ax.add_patch(patches.Rectangle((0, 0), n, n, linewidth=2.8,
                                        edgecolor="#222", facecolor="none"))
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_ylim(-0.05, n + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=130)


if __name__ == "__main__":
    env = BinairoSolveQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   pos={v_pos['accuracy']} neg={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
