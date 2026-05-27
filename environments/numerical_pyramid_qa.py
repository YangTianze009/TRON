"""
Numerical pyramid puzzle.

Each cell of an N-row pyramid equals the sum of the two cells directly below it.
Some cells are revealed in the image; the model must compute a single specified
empty cell (labeled with a letter A, B, C, ...).

Difficulty axes:
  - rows N: 3 (L0) -> 6 (L9)
  - number of unknowns (more = harder; bigger search if you don't propagate)

Answer format: a single integer placed in <answer>...</answer>.
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "In the number pyramid shown, each cell is the sum of the two cells directly below it. What is the value of the cell labelled {L}? Place the integer in <answer>...</answer>.",
    "Look at the number pyramid. Each block equals the sum of the two blocks beneath it. Find the number that belongs in cell {L}. Integer answer in <answer>...</answer>.",
    "The pyramid follows the rule: each upper cell equals the sum of the two directly below it. Compute the missing value at position {L}. Place the integer in <answer>...</answer>.",
    "Each cell in this pyramid is the sum of the two cells immediately under it. What number replaces {L}? Integer answer in <answer>...</answer>.",
    "Solve the number pyramid (each cell = sum of the two below). What is {L}? Place integer answer in <answer>...</answer>.",
    "The image shows a partially-filled number pyramid. Cells satisfy: upper = lower-left + lower-right. Find {L}. Integer answer in <answer>...</answer>.",
    "Pyramid arithmetic: each cell equals the sum of the two cells directly underneath. Determine the value of {L}. Integer in <answer>...</answer>.",
    "Compute the missing value labelled {L} in the number pyramid. Each block equals the sum of the two below it. Integer answer in <answer>...</answer>.",
    "Number pyramid puzzle: every cell is the sum of the two cells directly below. What is the integer at {L}? Place it in <answer>...</answer>.",
    "Inspect the pyramid. The rule is: parent = left child + right child. What value goes at position {L}? Place integer in <answer>...</answer>.",
    "Find the number missing at cell {L} in the pyramid. Each cell equals the sum of the two beneath it. Integer answer in <answer>...</answer>.",
    "Solve for {L} in the pyramid. Rule: each cell = sum of two cells directly below it. Place integer in <answer>...</answer>.",
    "The pyramid below obeys: every cell is the sum of the two cells immediately under it. Calculate {L}. Integer answer in <answer>...</answer>.",
    "Read the number pyramid carefully. Each cell equals the sum of the two cells directly below. What number belongs at {L}? Integer in <answer>...</answer>.",
    "Pyramid puzzle (each cell = sum of two below). Compute the value at {L} and place the integer in <answer>...</answer>.",
    "Determine the missing pyramid cell labelled {L} (parent = sum of two children). Integer answer in <answer>...</answer>.",
]


def _solve_pyramid(rows: List[List[Optional[int]]]) -> Optional[List[List[int]]]:
    """Try to fully solve the pyramid by repeatedly applying parent = L+R, or
    L = parent - R, or R = parent - L. Returns full pyramid or None."""
    n = len(rows)
    rows = [r[:] for r in rows]
    for _ in range(n * n + 5):
        changed = False
        for i in range(n - 1):
            for j in range(len(rows[i])):
                par = rows[i][j]
                lc = rows[i + 1][j]
                rc = rows[i + 1][j + 1]
                if par is None and lc is not None and rc is not None:
                    rows[i][j] = lc + rc
                    changed = True
                elif par is not None and lc is None and rc is not None:
                    rows[i + 1][j] = par - rc
                    changed = True
                elif par is not None and lc is not None and rc is None:
                    rows[i + 1][j + 1] = par - lc
                    changed = True
        if not changed:
            break
    if any(c is None for r in rows for c in r):
        return None
    return rows


class NumericalPyramidQA(StandaloneVisualEnv):
    ENV_NAME = "numerical_pyramid"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_rows = 3
        elif level <= 2:
            n_rows = 3
        elif level <= 4:
            n_rows = 4
        elif level <= 6:
            n_rows = 5
        else:
            n_rows = 6
        # number of revealed cells: more for low level
        total = n_rows * (n_rows + 1) // 2
        if level == 0:
            n_hidden = 1
        elif level <= 2:
            n_hidden = 2
        elif level <= 4:
            n_hidden = max(2, total // 3)
        elif level <= 6:
            n_hidden = max(3, total // 2)
        else:
            n_hidden = max(4, (total * 3) // 5)
        return {"level": level, "n_rows": n_rows, "n_hidden": n_hidden}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n_rows"]
        rng = random.Random((self.seed or 0) * 1493 + level * 47 + 7)

        # Build a valid pyramid: bottom row = small ints, propagate up.
        if level == 0:
            base = [rng.randint(1, 6) for _ in range(n)]
        elif level <= 4:
            base = [rng.randint(1, 9) for _ in range(n)]
        else:
            base = [rng.randint(1, 12) for _ in range(n)]
        # Build bottom-up: rows[0] = base, rows[1] = next up, ..., rows[n-1] = apex
        rows_bu: List[List[int]] = [base]
        for _ in range(1, n):
            below = rows_bu[-1]
            row = [below[j] + below[j + 1] for j in range(len(below) - 1)]
            rows_bu.append(row)
        # Reverse so full[0] = apex, full[n-1] = base
        full: List[List[int]] = list(reversed(rows_bu))

        # Try several hide patterns until one is solvable + the target cell is among the hidden.
        positions = [(i, j) for i in range(n) for j in range(len(full[i]))]
        success = None
        for attempt in range(40):
            rng2 = random.Random(rng.random())
            hide_pool = positions[:]
            rng2.shuffle(hide_pool)
            n_hide = min(cfg["n_hidden"], len(positions))
            hidden = set(hide_pool[:n_hide])
            partial: List[List[Optional[int]]] = [
                [None if (i, j) in hidden else full[i][j] for j in range(len(full[i]))]
                for i in range(n)
            ]
            sol = _solve_pyramid(partial)
            if sol is None:
                continue
            # Check uniqueness: every hidden cell should be deterministically derivable
            # (since base sums up uniquely, this holds when we got a complete fill).
            # Pick the target = a hidden cell that requires at least 1 inference step.
            # Prefer cells higher up (more "interesting"). At L0 we have only 1 hidden.
            target_candidates = list(hidden)
            # Prefer non-base hidden cells when possible
            non_base = [p for p in target_candidates if p[0] != n - 1]
            if non_base:
                target_candidates = non_base
            target = rng2.choice(target_candidates)
            success = (partial, target)
            break
        if success is None:
            return None
        partial, target = success
        ti, tj = target
        target_value = full[ti][tj]

        # Letter label for the target cell
        label = chr(ord("A") + (ti * 10 + tj) % 26)

        # Render: show every revealed cell with its number; show '?' or letter on
        # hidden cells. The target cell carries the letter, other hidden cells get
        # other letters (so model can refer to them but doesn't strictly need to).
        labels: Dict[Tuple[int, int], str] = {}
        # Assign labels to each hidden cell for visual reference
        used = set()
        # Make sure target gets the chosen label
        labels[(ti, tj)] = label
        used.add(label)
        next_letter = 0
        all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(n):
            for j in range(len(full[i])):
                if partial[i][j] is None and (i, j) != (ti, tj):
                    while next_letter < len(all_letters) and all_letters[next_letter] in used:
                        next_letter += 1
                    if next_letter >= len(all_letters):
                        labels[(i, j)] = "?"
                    else:
                        labels[(i, j)] = all_letters[next_letter]
                        used.add(all_letters[next_letter])
                        next_letter += 1

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(L=label)
        answer = str(target_value)
        img = self._render(partial, labels, n)
        return question, answer, img

    def _render(self, partial: List[List[Optional[int]]],
                labels: Dict[Tuple[int, int], str], n: int) -> Image.Image:
        rng = self._rng
        palette = rng.choice(self._COLOR_PALETTES) if rng else None
        cell_color = "#dbe9f7"
        unknown_color = "#fde2e4"
        bg = "#ffffff"
        cell_w = 1.0
        cell_h = 0.85
        fig_w = max(4.5, n * cell_w + 1.5)
        fig_h = max(3.5, n * cell_h + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        for i in range(n):
            row_len = len(partial[i])
            # Center the row horizontally
            x_offset = (n - row_len) * cell_w / 2.0
            y = (n - 1 - i) * cell_h
            for j in range(row_len):
                x = x_offset + j * cell_w
                v = partial[i][j]
                fc = cell_color if v is not None else unknown_color
                ax.add_patch(plt.Rectangle((x, y), cell_w * 0.95, cell_h * 0.95,
                                           facecolor=fc, edgecolor="#2c3e50",
                                           linewidth=1.3))
                cx = x + cell_w * 0.475
                cy = y + cell_h * 0.475
                if v is not None:
                    ax.text(cx, cy, str(v), ha="center", va="center",
                            fontsize=14, color="#1a1a1a", fontweight="bold")
                else:
                    lbl = labels.get((i, j), "?")
                    ax.text(cx, cy, lbl, ha="center", va="center",
                            fontsize=15, color="#b00020", fontweight="bold")

        ax.set_xlim(-0.2, n * cell_w + 0.2)
        ax.set_ylim(-0.3, n * cell_h + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Each cell = sum of the two cells below", fontsize=10, pad=6)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg, dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
