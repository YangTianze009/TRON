"""
Campsite (Tents-and-Trees) puzzle — structured-puzzle style.

Rules:
  - N x N grid where some cells contain Trees (T); empty cells (X) may
    receive a Tent.
  - Each tent must be orthogonally adjacent to at least one tree.
  - No two tents may be adjacent (including diagonally).
  - Per-row and per-column tent counts must match the given clues.
  - There is a one-to-one matching between tents and trees (each tree
    has exactly one adjacent tent assigned to it). We do not enforce the
    matching directly in `_check_answer`; the orthogonal-adjacency rule
    plus row/col counts is enough for a small puzzle to be uniquely
    determined when the puzzle has a unique solution.

Answer format (mirrors reference): list of `[row, col]` 1-indexed tent
positions. Examples from reference `idx=181-206`:
  - idx=181  4x4 → `[[1,3],[2,1],[4,4]]`
  - idx=182  4x4 → `[[2,1],[2,3],[4,3]]`
  - idx=187  5x5 → `[[1,3],[3,4],[5,3]]`
  - idx=200  7x7 → `[[3,2],[4,7],[6,6],[7,3]]`

Studied reference qids (all from design notes): 181, 182, 187,
188, 193, 194, 199, 200, 205, 206 (see file lines 379-388).

Difficulty axes:
  - N: grid size 4 -> 8
  - tent_density: relative # of tents (low = easy, high = hard)
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

from .standalone_base import StandaloneVisualEnv


# structured 4-section templates.
_TEMPLATES = [
    "Your task is to solve the {n}x{n} Campsite (Tents and Trees) puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Place tents on empty cells such that each tent is orthogonally adjacent to at least one tree.\n"
    "2. No two tents may be adjacent to each other (orthogonally or diagonally).\n"
    "3. The number of tents in each row must match the row clue; same for each column.\n"
    "4. Trees do not move. Tents may not be placed on tree cells.\n\n"
    "### Coordinate System:\n"
    "- Grid is {n}x{n}, indexed by (row, col) with rows 1..{n} top-to-bottom and columns 1..{n} left-to-right (1-indexed).\n"
    "- `T` = tree, `X` = empty cell.\n\n"
    "### Current Puzzle State:\n"
    "- Input grid:\n{state}\n"
    "- Row clues (top-to-bottom): {row_clues}\n"
    "- Column clues (left-to-right): {col_clues}\n\n"
    "### Output Format:\n"
    "Provide tent positions as a list of `[row, col]` 1-indexed pairs inside <answer>...</answer>.\n"
    "Example: <answer>[[1,3],[2,1],[4,4]]</answer>",

    "Solve the Campsite (Tents and Trees) puzzle below.\n\n"
    "### Game Rules:\n"
    "- Each tent must be orthogonally adjacent to at least one tree.\n"
    "- No two tents may be adjacent (8-directional) to each other.\n"
    "- Row and column tent counts must match the given clues.\n\n"
    "### Coordinate System:\n"
    "- Cells are 1-indexed by (row, col).\n"
    "- `T` denotes a tree, `X` an empty cell where a tent could go.\n\n"
    "### Current Puzzle State:\n"
    "Grid ({n}x{n}):\n{state}\n"
    "Row counts: {row_clues}\n"
    "Column counts: {col_clues}\n\n"
    "### Output Format:\n"
    "Return a list of `[r, c]` 1-indexed tent positions inside <answer>...</answer>.",

    "Your task is to solve the {n}x{n} Tents-and-Trees puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard rules: every tent touches a tree orthogonally; no two tents touch each other (incl. diagonally); per-row and per-column tent counts match the clues.\n\n"
    "### Coordinate System:\n"
    "- The grid uses (row, col) coordinates, 1-indexed.\n"
    "- `T` = tree cell; `X` = empty cell.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Row clues: {row_clues}; Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "Output the tent positions as a Python list of `[row, col]` pairs (1-indexed) inside <answer>...</answer>.",
]


# L0-L2 probe: ask if a specific empty cell holds a tent. YES/NO.
_PROBE_TEMPLATES = [
    "Your task is to determine if one cell holds a tent in the {n}x{n} Campsite puzzle below:\n\n"
    "### Game Rules:\n"
    "1. Each tent is orthogonally adjacent to at least one tree.\n"
    "2. No two tents may be adjacent (including diagonally).\n"
    "3. Per-row and per-column tent counts match the clues.\n"
    "4. Tents are never placed on tree cells.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 1-indexed.\n"
    "- `T` = tree, `X` = empty.\n\n"
    "### Current Puzzle State:\n"
    "- Input grid:\n{state}\n"
    "- Row clues: {row_clues}\n"
    "- Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "In the unique solution, does cell (row {r}, col {c}) hold a tent? Answer YES or NO inside <answer>...</answer>.",

    "Look at the Campsite (Tents and Trees) puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Tents-and-Trees rules.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 1-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Grid ({n}x{n}):\n{state}\n"
    "Row counts: {row_clues}\n"
    "Column counts: {col_clues}\n\n"
    "### Output Format:\n"
    "Should there be a tent at cell (row {r}, col {c})? Answer YES or NO inside <answer>...</answer>.",

    "Campsite puzzle ({n}x{n}).\n\n"
    "### Game Rules:\n"
    "Standard rules.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 1-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Row clues: {row_clues}; Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "Is there a tent at (row {r}, col {c})? Output YES or NO inside <answer>...</answer>.",
]


def _grid_to_ascii(n, trees):
    """Render the campsite state as ASCII (T = tree, X = empty)."""
    tree_set = set(trees)
    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append("T" if (r, c) in tree_set else "X")
        rows.append(" ".join(row))
    return "\n".join(rows)


# ----------------------------------------------------------------------- #
# Puzzle generation
# ----------------------------------------------------------------------- #

def _is_tent_placement_valid(
    n: int,
    trees: List[Tuple[int, int]],
    tents: List[Tuple[int, int]],
    row_counts: List[int],
    col_counts: List[int],
) -> bool:
    """Check all 4 tent rules: in-grid; on empty cell; orthogonal-tree;
    no tent-tent 8-adjacency; row/col counts match."""
    tree_set = set(trees)
    tent_set = set(tents)
    if len(tent_set) != len(tents):
        return False
    for (r, c) in tents:
        if not (0 <= r < n and 0 <= c < n):
            return False
        if (r, c) in tree_set:
            return False
        # Orthogonal-tree adjacency
        if not any((r + dr, c + dc) in tree_set
                   for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            return False
    # No two tents 8-adjacent (or coincident)
    for i, (r, c) in enumerate(tents):
        for j in range(i + 1, len(tents)):
            r2, c2 = tents[j]
            if abs(r - r2) <= 1 and abs(c - c2) <= 1:
                return False
    # Row/col counts
    rcounts = [0] * n
    ccounts = [0] * n
    for (r, c) in tents:
        rcounts[r] += 1
        ccounts[c] += 1
    if rcounts != row_counts:
        return False
    if ccounts != col_counts:
        return False
    return True


def _gen_campsite_puzzle(
    n: int,
    target_tents: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]],
                    List[int], List[int]]]:
    """Build (trees, tents, row_counts, col_counts) such that a one-to-one
    tree-tent matching exists and all tent rules are satisfied. Strategy:
    randomly place tents first (respecting no 8-adjacency), then place a
    tree orthogonally adjacent to each tent (preferring cells that are
    not already occupied by a tent).
    """
    for _ in range(max_attempts):
        # Step 1: shuffle cells, greedily place tents respecting the
        # no-tent-adjacency rule.
        cells = [(r, c) for r in range(n) for c in range(n)]
        rng.shuffle(cells)
        tents: List[Tuple[int, int]] = []
        for (r, c) in cells:
            if len(tents) >= target_tents:
                break
            ok = True
            for (r2, c2) in tents:
                if abs(r - r2) <= 1 and abs(c - c2) <= 1:
                    ok = False
                    break
            if ok:
                tents.append((r, c))
        if len(tents) < target_tents:
            continue

        # Step 2: assign each tent a unique adjacent tree
        used_for_tree = set(tents)
        trees: List[Tuple[int, int]] = []
        ok = True
        for (r, c) in tents:
            options = [(r + dr, c + dc)
                       for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                       if 0 <= r + dr < n and 0 <= c + dc < n
                       and (r + dr, c + dc) not in used_for_tree]
            if not options:
                ok = False
                break
            tree = rng.choice(options)
            used_for_tree.add(tree)
            trees.append(tree)
        if not ok:
            continue

        row_counts = [0] * n
        col_counts = [0] * n
        for (r, c) in tents:
            row_counts[r] += 1
            col_counts[c] += 1

        if _is_tent_placement_valid(n, trees, tents, row_counts, col_counts):
            return trees, tents, row_counts, col_counts
    return None


class CampsiteSolveQA(StandaloneVisualEnv):
    ENV_NAME = "campsite_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the tent list directly inside `<answer>...</answer>` as "
        "`[[r,c],[r,c],...]` 1-indexed."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode — small grid, ask YES/NO about one cell.
        if level == 0:
            return {"level": level, "n": 4, "target_tents": 2, "probe": True}
        if level == 1:
            return {"level": level, "n": 4, "target_tents": 3, "probe": True}
        if level == 2:
            return {"level": level, "n": 5, "target_tents": 3, "probe": True}
        if level == 3:
            n, tents = 5, 4
        elif level == 4:
            n, tents = 6, 4
        elif level == 5:
            n, tents = 6, 5
        elif level == 6:
            n, tents = 7, 5
        elif level == 7:
            n, tents = 7, 6
        elif level == 8:
            n, tents = 8, 6
        else:
            n, tents = 8, 7
        return {"level": level, "n": n, "target_tents": tents, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        target_tents = cfg["target_tents"]
        rng = random.Random((self.seed or 0) * 8009 + level * 53 + 11)

        result = _gen_campsite_puzzle(n, target_tents, rng)
        if result is None:
            return None
        trees, tents, row_counts, col_counts = result

        # Cache for verifier
        self._n = n
        self._trees = trees
        self._row_counts = row_counts
        self._col_counts = col_counts
        self._probe_mode = cfg.get("probe", False)
        self._tents_solution = set(tents)

        if self._probe_mode:
            # Pick a non-tree cell as probe target. Bias 50/50 between
            # tent and non-tent cells so YES/NO is balanced.
            tree_set = set(trees)
            tent_set = set(tents)
            non_tree = [(r, c) for r in range(n) for c in range(n)
                         if (r, c) not in tree_set]
            if not non_tree:
                return None
            tent_cells = [c for c in non_tree if c in tent_set]
            non_tent = [c for c in non_tree if c not in tent_set]
            if rng.random() < 0.5 and tent_cells:
                target = rng.choice(tent_cells)
                ans = "YES"
            elif non_tent:
                target = rng.choice(non_tent)
                ans = "NO"
            elif tent_cells:
                target = rng.choice(tent_cells)
                ans = "YES"
            else:
                return None
            self._probe_target = target
            self._probe_answer = ans
            sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
            state_ascii = _grid_to_ascii(n, trees)
            # 1-indexed in question
            question = _PROBE_TEMPLATES[sidx].format(
                n=n,
                state=state_ascii,
                row_clues=str(row_counts),
                col_clues=str(col_counts),
                r=target[0] + 1,
                c=target[1] + 1,
            )
            img = self._render(n, trees, row_counts, col_counts)
            return question, ans, img

        # Format ground truth as a list of 1-indexed [r, c] pairs sorted by (r, c)
        tents_sorted = sorted(tents, key=lambda rc: (rc[0], rc[1]))
        gt = "[" + ",".join(f"[{r + 1},{c + 1}]" for (r, c) in tents_sorted) + "]"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_ascii = _grid_to_ascii(n, trees)
        question = _TEMPLATES[sidx].format(
            n=n,
            state=state_ascii,
            row_clues=str(row_counts),
            col_clues=str(col_counts),
        )
        img = self._render(n, trees, row_counts, col_counts)
        return question, gt, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, n, trees, row_counts, col_counts) -> Image.Image:
        cell_size_in = 0.55
        fig_w = (n + 1.6) * cell_size_in + 0.4
        fig_h = (n + 1.6) * cell_size_in + 0.4
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        tree_set = set(trees)
        for r in range(n):
            for c in range(n):
                # Use cell coords (col, n-1-row)
                bg = "#f7f9fc"
                rect = patches.Rectangle((c, n - 1 - r), 1, 1,
                                         linewidth=1.0, edgecolor="#445",
                                         facecolor=bg)
                ax.add_patch(rect)
                if (r, c) in tree_set:
                    # Draw a stylized tree: green triangle + brown trunk
                    ax.add_patch(patches.Polygon(
                        [(c + 0.5, n - 1 - r + 0.85),
                         (c + 0.18, n - 1 - r + 0.28),
                         (c + 0.82, n - 1 - r + 0.28)],
                        facecolor="#1f8a3a", edgecolor="#0e5e22", linewidth=1.0,
                    ))
                    ax.add_patch(patches.Rectangle(
                        (c + 0.42, n - 1 - r + 0.10), 0.16, 0.20,
                        facecolor="#5b3a1a", edgecolor="#3a2510", linewidth=0.8,
                    ))
                    # Letter T overlay (small) — keep structured-puzzle style salience
                    ax.text(c + 0.5, n - 1 - r + 0.55, "T",
                            ha="center", va="center", fontsize=9,
                            fontweight="bold", color="#ffffff")
        # Outer border
        ax.add_patch(patches.Rectangle((0, 0), n, n, linewidth=2.5,
                                        edgecolor="#222", facecolor="none"))
        # Row clues on the right
        for r in range(n):
            ax.text(n + 0.4, n - 1 - r + 0.5, str(row_counts[r]),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="#c0392b")
        # Col clues on the bottom
        for c in range(n):
            ax.text(c + 0.5, -0.4, str(col_counts[c]),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="#c0392b")
        # Coordinate labels (1-indexed) along top + left to help model
        for r in range(n):
            ax.text(-0.4, n - 1 - r + 0.5, str(r + 1),
                    ha="center", va="center", fontsize=9, color="#666")
        for c in range(n):
            ax.text(c + 0.5, n + 0.4, str(c + 1),
                    ha="center", va="center", fontsize=9, color="#666")
        ax.set_xlim(-0.8, n + 0.85)
        ax.set_ylim(-0.85, n + 0.85)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ------------------------------------------------------------------ #
    # Answer checking — multiple valid tent placements possible; only
    # require constraints to be satisfied.
    # ------------------------------------------------------------------ #

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Fast-path guard: compute_score's fast-path sets _answer but skips
        # generate(), so _n / _trees / _probe_mode etc. aren't set. Raising
        # AttributeError triggers reward_function.py's slow-path fallback
        # (re-run generate to populate state, then retry verify).
        if not hasattr(self, "_n"):
            raise AttributeError("campsite state not initialized; need slow-path generate")
        # Try to parse predicted as a list of [r, c] pairs (1-indexed)
        import re
        # PROBE MODE: yes/no
        if getattr(self, "_probe_mode", False):
            text = predicted.strip().lower()
            m = re.search(r"\b(yes|no|y|n)\b", text)
            if m:
                w = m.group(1)
                pred = "YES" if w.startswith("y") else "NO"
                return pred == self._probe_answer
            return False
        s = predicted.strip()
        # Strip markdown / code fences
        s = re.sub(r"```[^\n]*\n", "", s)
        s = s.replace("```", "").strip()
        # Find all integer pairs in the form [r,c] or (r,c)
        pairs = re.findall(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]", s)
        if not pairs:
            # Also try plain comma-separated pairs without brackets per-pair
            return False
        try:
            tents_1idx = [(int(a), int(b)) for a, b in pairs]
        except ValueError:
            return False
        n = self._n
        # Convert to 0-indexed
        tents_0idx = [(r - 1, c - 1) for (r, c) in tents_1idx]
        # Drop duplicates while preserving order
        seen = set()
        tents_unique = []
        for rc in tents_0idx:
            if rc in seen:
                continue
            seen.add(rc)
            tents_unique.append(rc)
        return _is_tent_placement_valid(
            n, self._trees, tents_unique, self._row_counts, self._col_counts,
        )
