"""
Arrow-grid missing-arrow MCQ.

A small grid (typically 2x2 or 3x3) of arrow icons is drawn following a
visual rule: arrows in a row rotate consistently, columns alternate, all
arrows point the same way, etc. One cell is left blank; the model picks
the missing arrow from {↗, ↖, ↘, ↙} or {N, S, E, W}.

Reference qid 340:
  "Determine the missing part. Choice: (A) ↗ (B) ↖ (C) ↘ (D) ↙." Ans: A.

Output: single MCQ letter A/B/C/D.

L0: 2x2 grid where all four arrows point in the same direction (one
missing). Trivial pick.
L9: 4x4 grid following a rotation/permutation rule.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Direction vectors for diagonal arrows (NE, NW, SE, SW)
_DIAG_DIRS = {
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
    "SW": (-1, -1),
}
_DIAG_ARROWS = {"NE": "↗", "NW": "↖", "SE": "↘", "SW": "↙"}
_DIAG_LIST = ["NE", "NW", "SE", "SW"]

# Direction vectors for orthogonal arrows (N, S, E, W)
_ORTH_DIRS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}
_ORTH_ARROWS = {"N": "↑", "S": "↓", "E": "→", "W": "←"}
_ORTH_LIST = ["N", "S", "E", "W"]


_TEMPLATES = [
    "The grid below shows arrows that follow a pattern. Determine the missing arrow (marked '?'). Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Place the letter in <answer>...</answer>.",
    "Examine the arrows in the grid. One cell is empty (marked '?'). Pick the arrow that fits the pattern. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter answer in <answer>...</answer>.",
    "The arrow grid follows a rule. Find the missing arrow at '?'. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Place the letter in <answer>...</answer>.",
    "Look at the grid of arrows. Choose the missing arrow. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter answer in <answer>...</answer>.",
    "Determine the missing part of the arrow grid. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Place the letter in <answer>...</answer>.",
    "Identify which arrow belongs at the '?' cell. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter in <answer>...</answer>.",
    "Pattern of arrows in a grid: find the missing one. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter answer in <answer>...</answer>.",
    "The arrows in this grid follow a visual rule. Pick the missing arrow at '?'. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Place letter in <answer>...</answer>.",
    "What arrow goes in the empty cell ('?')? Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter in <answer>...</answer>.",
    "Arrow grid puzzle: select the arrow that fits the missing position. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter answer in <answer>...</answer>.",
    "Find the missing arrow in the displayed arrow grid. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Place the letter in <answer>...</answer>.",
    "Choose the arrow that completes the pattern at '?'. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter in <answer>...</answer>.",
    "Inspect the arrow grid. Determine the arrow at the empty cell. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter answer in <answer>...</answer>.",
    "Pick the arrow that should go at the '?' position to fit the grid pattern. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter in <answer>...</answer>.",
    "The arrow grid follows a directional rule. Identify the missing arrow at '?'. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Place letter in <answer>...</answer>.",
    "Determine the arrow that completes the grid pattern. Choices: (A) {a} (B) {b} (C) {c} (D) {d}. Letter answer in <answer>...</answer>.",
]


def _generate_pattern(pattern_type: str, rows: int, cols: int,
                      rng: random.Random, dir_set: List[str]) -> List[List[str]]:
    """Generate a rows x cols arrow grid following the named pattern.
    Returns 2D list of direction names."""
    if pattern_type == "all_same":
        d = rng.choice(dir_set)
        return [[d] * cols for _ in range(rows)]
    if pattern_type == "rows_alternate":
        # Even rows = d1, odd rows = d2
        d1, d2 = rng.sample(dir_set, 2)
        return [[d1 if r % 2 == 0 else d2] * cols for r in range(rows)]
    if pattern_type == "cols_alternate":
        d1, d2 = rng.sample(dir_set, 2)
        return [[d1 if c % 2 == 0 else d2 for c in range(cols)]
                for _ in range(rows)]
    if pattern_type == "row_uniform":
        # Each row has its own direction
        ds = []
        avail = dir_set[:]
        for _ in range(rows):
            ds.append(rng.choice(avail))
        return [[ds[r]] * cols for r in range(rows)]
    if pattern_type == "checker":
        d1, d2 = rng.sample(dir_set, 2)
        return [[d1 if (r + c) % 2 == 0 else d2 for c in range(cols)]
                for r in range(rows)]
    if pattern_type == "rotate_cw":
        # Sequence rotates clockwise; only valid for diag set of length 4.
        # Use NE → SE → SW → NW → NE ...
        seq = ["NE", "SE", "SW", "NW"]
        # Skip if dir_set doesn't match
        if not all(s in dir_set for s in seq):
            seq = dir_set[:]
        # Walk in row-major order
        start = rng.randint(0, len(seq) - 1)
        out = [[None] * cols for _ in range(rows)]
        for k in range(rows * cols):
            r, c = divmod(k, cols)
            out[r][c] = seq[(start + k) % len(seq)]
        return out
    raise ValueError(pattern_type)


class ArrowGridMissingQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "arrow_grid_missing"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            rows, cols = 2, 2
            patterns = ["all_same"]
        elif level <= 2:
            rows, cols = 2, 2
            patterns = ["all_same", "rows_alternate", "cols_alternate"]
        elif level <= 4:
            rows, cols = 3, 3
            patterns = ["rows_alternate", "cols_alternate", "row_uniform"]
        elif level <= 6:
            rows, cols = 3, 3
            patterns = ["row_uniform", "checker", "rotate_cw"]
        else:
            rows, cols = 4, 4
            patterns = ["checker", "rotate_cw", "row_uniform"]
        return {"level": level, "rows": rows, "cols": cols,
                "patterns": patterns}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1973 + level * 67 + 31)

        # Pick diagonal vs orthogonal arrow set
        use_diagonal = rng.random() < 0.7
        dir_set = _DIAG_LIST[:] if use_diagonal else _ORTH_LIST[:]
        arrow_lookup = _DIAG_ARROWS if use_diagonal else _ORTH_ARROWS

        pattern = rng.choice(cfg["patterns"])
        # Some patterns require specific dir sets
        if pattern == "rotate_cw":
            dir_set = _DIAG_LIST[:]
            arrow_lookup = _DIAG_ARROWS
            use_diagonal = True

        grid_dirs = _generate_pattern(pattern, cfg["rows"], cfg["cols"],
                                      rng, dir_set)
        # Pick missing cell
        mr = rng.randint(0, cfg["rows"] - 1)
        mc = rng.randint(0, cfg["cols"] - 1)
        true_dir = grid_dirs[mr][mc]

        # Build choices: shuffled list of 4 directions; ensure correct one is in
        # there, the other 3 are distinct (use the full dir set, sample if >4).
        choices_pool = dir_set[:]
        if len(choices_pool) < 4:
            # Pad with duplicates from arrow_lookup keys
            extra = [d for d in (_DIAG_LIST + _ORTH_LIST) if d not in choices_pool]
            choices_pool.extend(extra[:4 - len(choices_pool)])
        # Pick 4 distinct including the true one
        if true_dir not in choices_pool:
            choices_pool.insert(0, true_dir)
        # Take 4
        if len(choices_pool) > 4:
            others = [d for d in choices_pool if d != true_dir]
            rng.shuffle(others)
            choices_dirs = [true_dir] + others[:3]
        else:
            choices_dirs = choices_pool[:4]
        rng.shuffle(choices_dirs)
        # Assign letters A B C D
        letters = ["A", "B", "C", "D"]
        choice_letter = letters[choices_dirs.index(true_dir)]
        # Map for templates
        # Use a single arrow lookup dict that covers both sets
        full_arrow = {**_DIAG_ARROWS, **_ORTH_ARROWS}
        arrow_chars = [full_arrow[d] for d in choices_dirs]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            a=arrow_chars[0], b=arrow_chars[1], c=arrow_chars[2],
            d=arrow_chars[3])
        answer = choice_letter

        img = self._render(grid_dirs, mr, mc, full_arrow)
        return question, answer, img

    def _render(self, grid_dirs: List[List[str]], mr: int, mc: int,
                arrow_lookup: Dict[str, str]) -> Image.Image:
        rows = len(grid_dirs)
        cols = len(grid_dirs[0])
        bg = "#ffffff"
        cell_color = "#f0f4f8"
        missing_color = "#fde2e4"
        edge_color = "#1d3557"

        cell_size = 1.0
        fig_w = max(4.5, cols * cell_size + 1.0)
        fig_h = max(4.5, rows * cell_size + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        for r in range(rows):
            for c in range(cols):
                x = c * cell_size
                y = (rows - 1 - r) * cell_size
                is_missing = (r == mr and c == mc)
                fc = missing_color if is_missing else cell_color
                ax.add_patch(plt.Rectangle((x, y), cell_size, cell_size,
                                           facecolor=fc, edgecolor=edge_color,
                                           linewidth=1.5))
                cx = x + cell_size / 2
                cy = y + cell_size / 2
                if is_missing:
                    ax.text(cx, cy, "?", ha="center", va="center",
                            fontsize=24, fontweight="bold", color="#b00020")
                else:
                    arr = arrow_lookup[grid_dirs[r][c]]
                    ax.text(cx, cy, arr, ha="center", va="center",
                            fontsize=24, color="#1a1a1a")

        ax.set_xlim(-0.2, cols * cell_size + 0.2)
        ax.set_ylim(-0.2, rows * cell_size + 0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Find the missing arrow", fontsize=11, pad=6)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg,
                    dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
