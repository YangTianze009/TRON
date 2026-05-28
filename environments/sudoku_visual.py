"""
Sudoku Visual QA (D67, P0 — port from RLVE_Visual, adapted for reference).

Reference an external reference:
  "Fill in the white space to make it like a 5x5 sudoku."  Ans: 4

This env renders a partial Sudoku-style grid where every row and every
column must contain each digit exactly once. One cell is highlighted as
the missing target; the model must deduce the integer that fills it.

Note on grid sizes:
- 5×5 is the canonical reference size. Since 5 is prime, a 5×5 cannot
  be subdivided into N×M subgrids — so a 5x5 puzzle uses **rows-and-
  columns only** (i.e., a Latin square, no subgrid lines). This matches
  the an external reference figure.
- 4×4 (2×2 subgrids) and 6×6 (2×3 subgrids) are also offered as
  warm-up / variant levels.

Verifier: integer answer (`\\boxed{N}` or bare N).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _generate_sudoku_grid(N: int, M: int, rng: random.Random) -> List[List[int]]:
    """Generate a valid completed Sudoku grid of size NM x NM with N*M
    subgrid blocks of size N x M."""
    NM = N * M
    base = [
        [(M * (row % N) + row // N + column) % NM + 1 for column in range(NM)]
        for row in range(NM)
    ]
    perm = list(range(1, NM + 1))
    rng.shuffle(perm)
    grid = [[perm[base[r][c] - 1] for c in range(NM)] for r in range(NM)]

    def shuffle_groups(data, group_size):
        G = len(data) // group_size
        for g in range(G):
            start = g * group_size
            slice_ = data[start:start + group_size]
            rng.shuffle(slice_)
            data[start:start + group_size] = slice_
        groups = [data[g * group_size:(g + 1) * group_size] for g in range(G)]
        rng.shuffle(groups)
        data[:] = [row for group in groups for row in group]

    shuffle_groups(grid, N)
    grid_t = list(map(list, zip(*grid)))
    shuffle_groups(grid_t, M)
    grid = list(map(list, zip(*grid_t)))
    return grid


def _generate_latin_square(NM: int, rng: random.Random) -> List[List[int]]:
    """Generate an NM x NM Latin square (each row, each column is a
    permutation of 1..NM). Used when NM is prime (e.g. 5)."""
    base = [[((r + c) % NM) + 1 for c in range(NM)] for r in range(NM)]
    # Random row permutation
    rng.shuffle(base)
    # Random column permutation
    cols = list(range(NM))
    rng.shuffle(cols)
    grid = [[base[r][cols[c]] for c in range(NM)] for r in range(NM)]
    # Random label permutation
    perm = list(range(1, NM + 1))
    rng.shuffle(perm)
    grid = [[perm[grid[r][c] - 1] for c in range(NM)] for r in range(NM)]
    return grid


class SudokuVisualQA(StandaloneVisualEnv):
    ENV_NAME = "sudoku_visual"

    def _level_config(self, level: int) -> Dict:
        """Difficulty schedule.

        a math benchmark an external reference uses a 5×5 Latin-square (no subgrids). Match that
        format across all levels; difficulty is varied by adding extra
        blanks (still uniquely deducible from the target's row/col).
        """
        level = max(0, min(level, 9))
        if level <= 3:
            # 5×5 Latin square, single blank — canonical reference D67
            return {"NM": 5, "N": 1, "M": 1, "use_subgrids": False, "n_blanks": 1}
        if level <= 6:
            # 5×5 Latin square, single blank + 2 extra blanks elsewhere
            return {"NM": 5, "N": 1, "M": 1, "use_subgrids": False,
                    "n_blanks": 1, "extra_blanks": 2}
        # L7-L9: 5×5 Latin square, single blank + 4 extra blanks
        return {"NM": 5, "N": 1, "M": 1, "use_subgrids": False,
                "n_blanks": 1, "extra_blanks": 4}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4111 + level * 77 + 53)
        NM = cfg["NM"]
        use_subgrids = cfg["use_subgrids"]

        if use_subgrids:
            grid = _generate_sudoku_grid(cfg["N"], cfg["M"], rng)
        else:
            grid = _generate_latin_square(NM, rng)

        # Pick blanks
        cells = [(i, j) for i in range(NM) for j in range(NM)]
        rng.shuffle(cells)
        target = cells[0]
        target_val = grid[target[0]][target[1]]

        # Build display grid (0 for blank cells)
        display = [row[:] for row in grid]
        display[target[0]][target[1]] = 0

        # Optional extra blanks (must NOT be in the same row or column as
        # target, so the target stays uniquely deducible by row/col rule).
        extra_blanks_n = int(cfg.get("extra_blanks", 0))
        if extra_blanks_n > 0:
            extras = [(i, j) for (i, j) in cells[1:]
                      if i != target[0] and j != target[1]]
            for (i, j) in extras[:extra_blanks_n]:
                display[i][j] = 0

        # 2026-05-04: was trimmed to verbatim a math benchmark wording → 3% passrate
        # (model can't find target without explicit guidance). Adds rule
        # explanation + cell location hint at L0-L4 for learning signal;
        # L5+ revert to verbatim benchmark wording.
        if level <= 4:
            ti, tj = target
            question = (
                f"Fill in the highlighted (red-shaded) cell at row {ti+1}, "
                f"column {tj+1} to make a valid {NM}x{NM} Latin square. "
                f"Rules: every row and every column must contain each digit "
                f"from 1 to {NM} exactly once. Output the missing digit."
            )
        else:
            question = (
                f"Fill in the white space to make it like a {NM}x{NM} sudoku."
            )

        answer = str(target_val)
        img = self._render(display, cfg, target)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, grid, cfg, target) -> Image.Image:
        NM = cfg["NM"]
        N, M = cfg["N"], cfg["M"]
        use_subgrids = cfg["use_subgrids"]
        cell_size = max(40, 360 // NM)
        fig_size = (NM * cell_size / 70 + 1.0, NM * cell_size / 70 + 1.0)
        fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        ti, tj = target
        for i in range(NM):
            for j in range(NM):
                val = grid[i][j]
                if (i, j) == target:
                    bg = "#fce8e6"
                    edge = "#d62728"
                    edge_lw = 2.5
                elif val == 0:
                    bg = "#f7f9fc"
                    edge = "#bdc3c7"
                    edge_lw = 0.6
                else:
                    bg = "#ffffff"
                    edge = "#bdc3c7"
                    edge_lw = 0.6
                rect = patches.Rectangle(
                    (j, NM - 1 - i), 1, 1,
                    linewidth=edge_lw, edgecolor=edge, facecolor=bg
                )
                ax.add_patch(rect)
                if val != 0:
                    ax.text(j + 0.5, NM - 0.5 - i, str(val),
                            ha="center", va="center",
                            fontsize=max(11, 22 - NM),
                            fontweight="bold", color="#2c3e50")

        # Lines: thick subgrid borders only when use_subgrids; otherwise
        # all internal lines are thin (Latin square style, like an external reference).
        for i in range(0, NM + 1):
            if use_subgrids:
                lw = 2.5 if i % N == 0 else 0.5
            else:
                lw = 2.5 if (i == 0 or i == NM) else 0.5
            ax.plot([0, NM], [i, i], color="#2c3e50", linewidth=lw)
        for j in range(0, NM + 1):
            if use_subgrids:
                lw = 2.5 if j % M == 0 else 0.5
            else:
                lw = 2.5 if (j == 0 or j == NM) else 0.5
            ax.plot([j, j], [0, NM], color="#2c3e50", linewidth=lw)

        ax.set_xlim(-0.1, NM + 0.1)
        ax.set_ylim(-0.1, NM + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = SudokuVisualQA()
    pass_count = 0
    total = 0
    # 2026-05-04 strict format: wrapper is determined by seed % 3, so each
    # seed must be tested with the wrapper its prompt requested.
    wrappers_for = lambda a: [
        f"<answer>{a}</answer>",        # idx 0
        f"\\boxed{{{a}}}",              # idx 1
        f"Final answer: {a}",           # idx 2
    ]
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} q[:60]={env._question[:60] if ok else '...'}; A={env._answer}")
            if ok:
                wrapped = wrappers_for(env._answer)[s % 3]
                v_correct = env.verify(wrapped)
                v_wrong = env.verify(wrappers_for("definitely_wrong_xyz")[s % 3])
                print(f"   correct={v_correct['accuracy']} wrong={v_wrong['accuracy']}")
                if v_correct['accuracy'] == 1 and v_wrong['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
