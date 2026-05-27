"""
Skyscrapers puzzle — structured-puzzle style.

Rules:
  - N x N Latin square: each row and each column contains the heights
    1..N exactly once.
  - Clues around the grid (top, bottom, left, right) give the COUNT OF
    SKYSCRAPERS VISIBLE from that side, where a taller skyscraper blocks
    shorter ones behind it.

Studied reference qids (design notes lines 269-281):
  - idx=691 (D=1): 3x3, top=[1,3,2] bottom=[2,1,2] left=[1,3,2]
    right=[2,1,2] -> [[3,1,2],[1,2,3],[2,3,1]]
  - idx=692 (D=1): 3x3, top=[1,2,2] bottom=[3,2,1] left=[1,2,3]
    right=[2,2,1] -> [[3,1,2],[2,3,1],[1,2,3]]
  - idx=710 (D=3): 4x4
  - idx=720 (D=5): 5x5

Answer format (mirrors reference): 2D list of heights `[[3,1,2],[1,2,3],[2,3,1]]`.

Difficulty axis: N (3 -> 5).
"""
import random
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO

from .standalone_base import StandaloneVisualEnv


# 2026-05-04: simplified L0/L1 (was 7.5% too-hard).
# L0/L1 PROBE templates: ask single cell value with most others revealed.
_PROBE_TEMPLATES = [
    "Your task is to find ONE cell value in the {n}x{n} Skyscrapers puzzle below:\n\n"
    "### Game Rules:\n"
    "1. Each row contains heights 1..{n} exactly once.\n"
    "2. Each column contains heights 1..{n} exactly once.\n"
    "3. Each edge clue counts how many skyscrapers are visible from that side; taller skyscrapers block shorter ones behind them.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "- Top: {top}; Bottom: {bottom}; Left: {left}; Right: {right}\n"
    "- Already-known cell values: {known}\n\n"
    "### Output Format:\n"
    "What height (1-{n}) goes in cell (row {r}, col {c})? Output ONLY the single digit inside <answer>...</answer>.",

    "Look at the {n}x{n} Skyscrapers puzzle.\n\n"
    "### Game Rules:\n"
    "Standard Skyscrapers — Latin square + visibility clues.\n\n"
    "### Current Puzzle State:\n"
    "- Edge clues: top={top} bottom={bottom} left={left} right={right}\n"
    "- Known cell values: {known}\n\n"
    "### Output Format:\n"
    "Find the height (1-{n}) at cell (row {r}, col {c}). Output the digit inside <answer>...</answer>.",

    "Skyscrapers puzzle ({n}x{n}).\n\n"
    "### Game Rules:\n"
    "Latin square + visibility clues from each side.\n\n"
    "### Current Puzzle State:\n"
    "- top={top}; bottom={bottom}; left={left}; right={right}\n"
    "- Already known: {known}\n\n"
    "### Output Format:\n"
    "Provide the height (1-{n}) for cell (row {r}, col {c}) inside <answer>...</answer>.",
]


_TEMPLATES = [
    "Your task is to solve the {n}x{n} Skyscrapers puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each row contains heights 1..{n} exactly once.\n"
    "2. Each column contains heights 1..{n} exactly once.\n"
    "3. Each edge clue counts how many skyscrapers are visible from that side, where a taller skyscraper blocks all shorter ones behind it.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n"
    "- Top clues correspond to columns left-to-right; left clues correspond to rows top-to-bottom.\n\n"
    "### Current Puzzle State:\n"
    "- Top (left-to-right): {top}\n"
    "- Bottom (left-to-right): {bottom}\n"
    "- Left (top-to-bottom): {left}\n"
    "- Right (top-to-bottom): {right}\n\n"
    "### Output Format:\n"
    "Output the completed grid as a Python 2D list of heights inside <answer>...</answer>.\n"
    "Example: <answer>[[3,1,2],[1,2,3],[2,3,1]]</answer>",

    "Solve the {n}x{n} Skyscrapers puzzle below.\n\n"
    "### Game Rules:\n"
    "- Latin square: rows and columns are permutations of 1..{n}.\n"
    "- Edge clues = number of buildings visible from that side (taller blocks shorter).\n\n"
    "### Coordinate System:\n"
    "- {n}x{n} grid; 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Top: {top}\nBottom: {bottom}\nLeft: {left}\nRight: {right}\n\n"
    "### Output Format:\n"
    "Output the height grid as a Python 2D list inside <answer>...</answer>.",

    "Your task is to solve the {n}x{n} Skyscrapers puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard Skyscrapers: Latin square + visibility clues.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "Top: {top}; Bottom: {bottom}; Left: {left}; Right: {right}\n\n"
    "### Output Format:\n"
    "Output the heights grid inside <answer>...</answer>.",
]


def _visible_left_to_right(row: List[int]) -> int:
    """Count of skyscrapers visible looking from the left."""
    cnt = 0
    cur_max = 0
    for h in row:
        if h > cur_max:
            cnt += 1
            cur_max = h
    return cnt


def _visible_right_to_left(row: List[int]) -> int:
    return _visible_left_to_right(list(reversed(row)))


def _is_skyscraper_valid(grid: List[List[int]], n: int,
                          top: List[int], bottom: List[int],
                          left: List[int], right: List[int]) -> bool:
    if len(grid) != n:
        return False
    if any(len(row) != n for row in grid):
        return False
    expected = set(range(1, n + 1))
    for row in grid:
        if set(row) != expected:
            return False
    for c in range(n):
        col = [grid[r][c] for r in range(n)]
        if set(col) != expected:
            return False
    # Visibility
    for i in range(n):
        if _visible_left_to_right(grid[i]) != left[i]:
            return False
        if _visible_right_to_left(grid[i]) != right[i]:
            return False
    for j in range(n):
        col = [grid[r][j] for r in range(n)]
        if _visible_left_to_right(col) != top[j]:
            return False
        if _visible_right_to_left(col) != bottom[j]:
            return False
    return True


def _gen_skyscrapers(n: int, rng: random.Random,
                     max_attempts: int = 30) -> Optional[Tuple[List[List[int]],
                                                               List[int], List[int],
                                                               List[int], List[int]]]:
    """Sample a valid Latin square (heights 1..n, by shifting two random
    permutations) and compute clues."""
    for _ in range(max_attempts):
        perm_row = list(range(n))
        perm_col = list(range(n))
        rng.shuffle(perm_row)
        rng.shuffle(perm_col)
        # Cell (i, j) = ((perm_row[i] + perm_col[j]) % n) + 1
        grid = [[((perm_row[i] + perm_col[j]) % n) + 1 for j in range(n)]
                for i in range(n)]
        # Verify uniqueness in row/col
        ok = True
        for r in range(n):
            if set(grid[r]) != set(range(1, n + 1)):
                ok = False
                break
        if ok:
            for c in range(n):
                if set(grid[r][c] for r in range(n)) != set(range(1, n + 1)):
                    ok = False
                    break
        if not ok:
            continue
        left = [_visible_left_to_right(grid[i]) for i in range(n)]
        right = [_visible_right_to_left(grid[i]) for i in range(n)]
        top = []
        bottom = []
        for j in range(n):
            col = [grid[r][j] for r in range(n)]
            top.append(_visible_left_to_right(col))
            bottom.append(_visible_right_to_left(col))
        return grid, top, bottom, left, right
    return None


class SkyscrapersSolveQA(StandaloneVisualEnv):
    ENV_NAME = "skyscrapers_solve"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the completed grid directly as a 2D list "
        "`[[...],[...],...]` of heights inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — PROBE one cell
        # at L0/L1 with most other cells revealed.
        if level <= 1:
            return {"level": level, "n": 3, "probe": True}
        if level <= 2:
            n = 4
        elif level <= 4:
            n = 4
        elif level <= 6:
            n = 5
        else:
            n = 5
        return {"level": level, "n": n, "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 7901 + level * 67 + 19)

        result = _gen_skyscrapers(n, rng)
        if result is None:
            return None
        grid, top, bottom, left, right = result

        self._n = n
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right
        self._probe_mode = cfg.get("probe", False)

        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — PROBE one cell.
        if self._probe_mode:
            # Pick one cell to ask about; reveal the rest.
            target_r = rng.randrange(n)
            target_c = rng.randrange(n)
            target_val = grid[target_r][target_c]
            self._probe_target = (target_r, target_c)
            self._probe_target_val = target_val
            known_pairs = []
            for i in range(n):
                for j in range(n):
                    if (i, j) != (target_r, target_c):
                        known_pairs.append(f"({i},{j})={grid[i][j]}")
            known_str = " ".join(known_pairs)
            ans_str = str(target_val)
            sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
            question = _PROBE_TEMPLATES[sidx].format(
                n=n, top=str(top), bottom=str(bottom),
                left=str(left), right=str(right),
                known=known_str, r=target_r, c=target_c,
            )
            img = self._render(n, top, bottom, left, right)
            return question, ans_str, img

        gt = "[" + ",".join(
            "[" + ",".join(str(v) for v in row) + "]" for row in grid
        ) + "]"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            n=n,
            top=str(top), bottom=str(bottom),
            left=str(left), right=str(right),
        )
        img = self._render(n, top, bottom, left, right)
        return question, gt, img

    def _render(self, n, top, bottom, left, right) -> Image.Image:
        cell_in = 0.55
        fig_w = (n + 2.5) * cell_in + 0.4
        fig_h = (n + 2.5) * cell_in + 0.4
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Empty grid
        for r in range(n):
            for c in range(n):
                ax.add_patch(patches.Rectangle((c, n - 1 - r), 1, 1,
                                                linewidth=1.0,
                                                edgecolor="#445",
                                                facecolor="#f7f9fc"))
        ax.add_patch(patches.Rectangle((0, 0), n, n, linewidth=2.5,
                                        edgecolor="#222", facecolor="none"))
        # Top clues
        for j in range(n):
            ax.text(j + 0.5, n + 0.45, str(top[j]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1a3a6e")
        # Bottom clues
        for j in range(n):
            ax.text(j + 0.5, -0.45, str(bottom[j]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1a3a6e")
        # Left clues
        for i in range(n):
            ax.text(-0.45, n - 1 - i + 0.5, str(left[i]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1a3a6e")
        # Right clues
        for i in range(n):
            ax.text(n + 0.45, n - 1 - i + 0.5, str(right[i]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1a3a6e")
        ax.set_xlim(-0.85, n + 0.85)
        ax.set_ylim(-0.85, n + 0.85)
        ax.set_aspect("equal")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        n = self._n
        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — PROBE single digit.
        if getattr(self, "_probe_mode", False):
            text = predicted.strip()
            m = re.search(r"\b([1-9])\b", text)
            if m:
                return int(m.group(1)) == self._probe_target_val
            return False
        s = predicted.strip()
        s = re.sub(r"```[^\n]*\n", "", s)
        s = s.replace("```", "").strip()
        # Try Python-style 2D list parsing first
        rows_raw = re.findall(r"\[([^\[\]]+)\]", s)
        grid = None
        if len(rows_raw) >= n:
            try:
                cand = []
                for row_raw in rows_raw[:n]:
                    nums = re.findall(r"\d+", row_raw)
                    if len(nums) != n:
                        cand = None
                        break
                    cand.append([int(x) for x in nums])
                if cand and len(cand) == n:
                    grid = cand
            except ValueError:
                grid = None
        # Fallback: newline-separated rows of integers
        if grid is None:
            rows = [r for r in s.splitlines() if r.strip()]
            if len(rows) == n:
                try:
                    cand = []
                    for row in rows:
                        nums = re.findall(r"\d+", row)
                        if len(nums) != n:
                            cand = None
                            break
                        cand.append([int(x) for x in nums])
                    if cand and len(cand) == n:
                        grid = cand
                except ValueError:
                    grid = None
        if grid is None:
            return False
        return _is_skyscraper_valid(grid, n, self._top, self._bottom,
                                     self._left, self._right)
