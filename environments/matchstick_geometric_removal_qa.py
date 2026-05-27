"""
Matchstick geometric removal puzzle (reference R14).

Family: "Use N matches (labeled A..Z) to form K shapes. Remove M matches to
leave (K - delta) shapes." Multiple valid removals may exist; we accept any
removal that satisfies the constraint.

Self-contained. NO `from RLVE`, `from Gym`, `sys.path.insert`.

We use a grid-of-unit-squares pattern (2x2, 3x2, 3x3) where each match is a
horizontal or vertical edge of the unit cells. We label matches with letters
A..Z and ask the model to remove a small number of matches such that exactly
the target number of unit squares remain (we count only 1x1 unit squares for
simplicity — extending to all square sizes is also valid but harder to
verify deterministically; we keep this version simple and well-defined).

Difficulty levels (10 levels 0..9):
  L0 trivially solvable: 2x2 grid (12 matches, 4 squares), remove 1 to leave 3 squares
  L9: 3x3 grid (24 matches, 9 squares), remove 4 matches to leave several squares
"""
import ast
import random
import re
from io import BytesIO
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the matchstick removal puzzle described below.\n\n"
    "### Game Rules:\n"
    "1. The figure is built from {n_matches} matchsticks labeled with letters {labels}.\n"
    "2. The figure currently contains {orig_squares} unit squares (1x1).\n"
    "3. Remove exactly {n_remove} matchsticks so that the remaining figure contains exactly {target_squares} unit squares.\n"
    "4. Multiple valid removals may exist; output any one of them.\n\n"
    "### Coordinate System:\n"
    "- Each matchstick is one edge of a {grid_w}x{grid_h} grid of unit cells.\n"
    "- Match labels are uppercase letters as shown in the image.\n\n"
    "### Current Puzzle State:\n"
    "- Grid: {grid_w}x{grid_h}\n"
    "- Match labels (in order): {labels}\n"
    "- Number of matches to remove: {n_remove}\n"
    "- Target number of remaining unit squares: {target_squares}\n\n"
    "### Output Format:\n"
    "Output a Python list of letter labels (e.g. ['A', 'C']) in the format the prompt asked for.",

    "Solve the matchstick removal puzzle below.\n\n"
    "### Game Rules:\n"
    "- Remove exactly {n_remove} matchsticks from the figure (labeled by letters).\n"
    "- The remaining figure must contain exactly {target_squares} unit squares.\n"
    "- Output any valid removal.\n\n"
    "### Coordinate System:\n"
    "- {grid_w}x{grid_h} unit-cell grid; matches are edges, labeled A..\n\n"
    "### Current Puzzle State:\n"
    "Total matches: {n_matches}; original unit squares: {orig_squares}\n"
    "Remove: {n_remove}; target unit squares: {target_squares}\n\n"
    "### Output Format:\n"
    "Output a Python list of letter labels (e.g. ['A', 'C']) in the format the prompt asked for.",

    "Your task is to solve the matchstick puzzle described below.\n\n"
    "### Game Rules:\n"
    "Remove the indicated number of matches so the remaining figure has the target unit-square count.\n\n"
    "### Coordinate System:\n"
    "- Letter-labeled matches; {grid_w}x{grid_h} grid.\n\n"
    "### Current Puzzle State:\n"
    "Remove {n_remove} matches; target {target_squares} unit squares.\n\n"
    "### Output Format:\n"
    "Output a Python list of letter labels (e.g. ['A', 'C']) in the format the prompt asked for.",
]


def _build_grid_matches(grid_w: int, grid_h: int):
    """Return (matches, squares) where:
       - matches is a list of edges, each (kind, r, c). 'h' = horizontal edge
         at row r, col c; 'v' = vertical edge at row r, col c.
       - squares is a list of (r, c, size) — currently size=1 unit squares.
    Match indices map to letters A..Z in order.
    Numbering convention: horizontal edges first row-by-row, then vertical
    edges row-by-row.
    """
    matches = []
    # Horizontal edges: (h, r, c) for r in 0..grid_h, c in 0..grid_w-1
    for r in range(grid_h + 1):
        for c in range(grid_w):
            matches.append(("h", r, c))
    # Vertical edges: (v, r, c) for r in 0..grid_h-1, c in 0..grid_w
    for r in range(grid_h):
        for c in range(grid_w + 1):
            matches.append(("v", r, c))
    # Unit squares: (r, c) for r in 0..grid_h-1, c in 0..grid_w-1
    squares = [(r, c, 1) for r in range(grid_h) for c in range(grid_w)]
    return matches, squares


def _square_edges(r: int, c: int, size: int = 1):
    """Return the 4 edges (kind, r, c) of the unit square at (r, c)."""
    edges = []
    # Top (h, r, c..c+size-1)
    for dc in range(size):
        edges.append(("h", r, c + dc))
    # Bottom (h, r+size, c..c+size-1)
    for dc in range(size):
        edges.append(("h", r + size, c + dc))
    # Left (v, r..r+size-1, c)
    for dr in range(size):
        edges.append(("v", r + dr, c))
    # Right (v, r..r+size-1, c+size)
    for dr in range(size):
        edges.append(("v", r + dr, c + size))
    return edges


def _count_unit_squares(matches, squares, removed_set):
    """Count squares that have all 4 edges still present."""
    present = set(matches) - removed_set
    count = 0
    for (r, c, size) in squares:
        edges = _square_edges(r, c, size)
        if all(e in present for e in edges):
            count += 1
    return count


def _generate_problem_for_level(level: int, rng: random.Random):
    """Choose grid size + removal target so that a valid solution exists."""
    if level <= 1:
        # 2026-05-04: simplified L0/L1 (was 0% too-hard).
        # Hardcode 2x2 grid + n_remove=1 + answer that removes ONE OUTER edge,
        # so target_squares == orig_squares - 1 (= 3). Outer edges are easy
        # to identify visually + eliminate by exclusion.
        grid_w, grid_h = 2, 2
        matches, squares = _build_grid_matches(grid_w, grid_h)
        # Outer edges of 2x2 grid: top row (h, 0, *), bottom row (h, 2, *),
        # left col (v, *, 0), right col (v, *, 2). Inner edges: (h, 1, *) and
        # (v, *, 1). Pick a deterministic outer edge per seed.
        outer_idxs = [i for i, (k, r, c) in enumerate(matches)
                      if (k == "h" and (r == 0 or r == grid_h)) or
                         (k == "v" and (c == 0 or c == grid_w))]
        # Use the rng to pick one outer edge to remove.
        ref_idx = rng.choice(outer_idxs)
        ref_solution = [ref_idx]
        n_remove = 1
        orig_squares = _count_unit_squares(matches, squares, set())
        removed_set = {matches[ref_idx]}
        target_squares = _count_unit_squares(matches, squares, removed_set)
        return {
            "grid_w": grid_w,
            "grid_h": grid_h,
            "matches": matches,
            "squares": squares,
            "n_matches": len(matches),
            "orig_squares": orig_squares,
            "n_remove": n_remove,
            "target_squares": target_squares,
            "ref_solution_idxs": ref_solution,
        }
    if level <= 3:
        grid_w, grid_h = 2, 2
        n_remove = 2
    elif level <= 5:
        grid_w, grid_h = 3, 2
        n_remove = 2
    elif level <= 7:
        grid_w, grid_h = 3, 2
        n_remove = 3
    else:
        grid_w, grid_h = 3, 3
        n_remove = rng.randint(3, 4)
    matches, squares = _build_grid_matches(grid_w, grid_h)
    n_matches = len(matches)
    if n_matches > 26:
        return None
    orig_squares = _count_unit_squares(matches, squares, set())
    # Try a random removal candidate, target = whatever it produces, but we
    # need at least `1 <= target_squares < orig_squares`. We pick target by
    # sampling until valid.
    for _ in range(80):
        candidate_idxs = rng.sample(range(n_matches), n_remove)
        removed_set = set(matches[i] for i in candidate_idxs)
        target_squares = _count_unit_squares(matches, squares, removed_set)
        if 0 < target_squares < orig_squares:
            # Also confirm there's at least one valid solution we can find via
            # exhaustive enumeration — if too many combinations, just trust the
            # sampled removal.
            ref_solution = sorted(candidate_idxs)
            return {
                "grid_w": grid_w,
                "grid_h": grid_h,
                "matches": matches,
                "squares": squares,
                "n_matches": n_matches,
                "orig_squares": orig_squares,
                "n_remove": n_remove,
                "target_squares": target_squares,
                "ref_solution_idxs": ref_solution,
            }
    return None


class MatchstickGeometricRemovalQA(StandaloneVisualEnv):
    ENV_NAME = "matchstick_geometric_removal"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the list of removed letter labels in the format the prompt asked for."
    )

    def _level_config(self, level: int) -> Dict:
        return {"level": max(0, min(level, 9))}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        rng = random.Random((seed or 0) * 5051 + level * 71 + 31)

        result = _generate_problem_for_level(level, rng)
        if result is None:
            return None

        # Cache state
        self._matches = result["matches"]
        self._squares = result["squares"]
        self._n_matches = result["n_matches"]
        self._n_remove = result["n_remove"]
        self._target_squares = result["target_squares"]

        # Letter labels A..Z
        labels = [chr(ord("A") + i) for i in range(self._n_matches)]
        self._labels = labels
        ref = result["ref_solution_idxs"]
        ans_str = str([labels[i] for i in ref])

        sidx = (seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            n_matches=self._n_matches,
            labels=", ".join(labels),
            orig_squares=result["orig_squares"],
            n_remove=self._n_remove,
            target_squares=self._target_squares,
            grid_w=result["grid_w"],
            grid_h=result["grid_h"],
        )
        # 2026-05-04: simplified L0/L1 (was 0% too-hard).
        # At L0/L1 the puzzle is hardcoded to a 2x2 grid where target = 3
        # (= 4 - 1). The answer is ALWAYS to remove ONE outer (perimeter)
        # matchstick, which destroys exactly one unit square (the one it
        # borders). Inner edges border 2 squares so they'd destroy both.
        if level <= 1:
            question += (
                "\n\nHint: this puzzle has a 2x2 grid with 4 unit squares. "
                "Target = 3 squares means destroy EXACTLY ONE square. "
                "An OUTER perimeter matchstick borders only 1 square; "
                "removing it destroys that 1 square (target reached). "
                "An INNER matchstick borders 2 squares; removing it destroys "
                "BOTH (count drops to 2, wrong). So pick ANY single OUTER "
                "perimeter matchstick. Example outer edges: those on the very "
                "top, bottom, leftmost, or rightmost border of the grid."
            )
        img = self._render(result["grid_w"], result["grid_h"],
                           self._matches, labels)
        return question, ans_str, img

    # ---------------------------------------------------------------- render
    def _render(self, grid_w, grid_h, matches, labels) -> Image.Image:
        fig_w = max(4.5, grid_w * 1.5 + 0.5)
        fig_h = max(3.5, grid_h * 1.5 + 0.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for i, (kind, r, c) in enumerate(matches):
            label = labels[i]
            if kind == "h":
                # Horizontal edge at row r, columns c..c+1; coords use (col, -row)
                x0, x1 = c, c + 1
                y0, y1 = -r, -r
                ax.plot([x0, x1], [y0, y1], color="#8b4513", linewidth=4,
                        solid_capstyle="round")
                ax.text((x0 + x1) / 2, y0 + 0.15, label,
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="#1a3a6e")
            else:
                # Vertical edge at column c, rows r..r+1
                x0, x1 = c, c
                y0, y1 = -r, -(r + 1)
                ax.plot([x0, x1], [y0, y1], color="#8b4513", linewidth=4,
                        solid_capstyle="round")
                ax.text(x0 - 0.15, (y0 + y1) / 2, label,
                        ha="right", va="center",
                        fontsize=10, fontweight="bold", color="#1a3a6e")
        ax.set_xlim(-0.6, grid_w + 0.5)
        ax.set_ylim(-grid_h - 0.5, 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # --------------------------------------------------------- answer check
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        s = predicted.strip()
        s = re.sub(r"```[^\n]*\n", "", s).replace("```", "").strip()
        # Try parsing list of letters
        letters = None
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                letters = [str(x).strip().upper() for x in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        if letters is None:
            # Extract single uppercase letter tokens
            letters = re.findall(r"\b([A-Z])\b", s.upper())
        if not letters:
            return False
        # Drop duplicates while preserving order
        seen = set()
        unique = []
        for L in letters:
            if L not in seen:
                seen.add(L)
                unique.append(L)
        # Must remove exactly n_remove distinct matches
        if len(unique) != self._n_remove:
            return False
        # Convert letters to match indices
        label_to_idx = {L: i for i, L in enumerate(self._labels)}
        try:
            removed_idxs = [label_to_idx[L] for L in unique]
        except KeyError:
            return False
        removed_set = set(self._matches[i] for i in removed_idxs)
        # Count remaining unit squares
        remaining = _count_unit_squares(self._matches, self._squares, removed_set)
        return remaining == self._target_squares
