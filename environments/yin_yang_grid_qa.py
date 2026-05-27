"""
Yin-Yang puzzle (small grid).

Rules:
  - Fill every cell with B(black) or W(white).
  - All B cells must be 4-orthogonally connected.
  - All W cells must be 4-orthogonally connected.
  - No 2x2 sub-grid can be all the same color.

Given a partial grid (some cells filled), output the completion.

Difficulty axes:
  - grid size (3 → 5)
  - # given clues (more clues = easier)
"""
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# 2026-05-04: simplified L0/L1 (was 7.5% too-hard).
# L0/L1 PROBE templates: ask whether one cell is B or W.
_PROBE_TEMPLATES = [
    "Your task is to determine the color of ONE cell in the {n}x{n} Yin-Yang puzzle below:\n\n"
    "### Game Rules:\n"
    "1. Each cell holds either B (black) or W (white).\n"
    "2. All B cells must form a single 4-orthogonally connected region.\n"
    "3. All W cells must form a single 4-orthogonally connected region.\n"
    "4. No 2x2 sub-grid may contain four cells of the same color.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "In the unique solution, what color is cell (row {r}, col {c})? Output `B` or `W` inside <answer>...</answer>.",

    "Look at the {n}x{n} Yin-Yang puzzle.\n\n"
    "### Game Rules:\n"
    "Standard Yin-Yang: each color forms one connected region; no 2x2 monochrome.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Determine whether cell (row {r}, col {c}) is B (black) or W (white). Output one letter inside <answer>...</answer>.",

    "Yin-Yang puzzle ({n}x{n}).\n\n"
    "### Game Rules:\n"
    "Each color is 4-connected; no 2x2 same color.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "What color goes in cell (row {r}, col {c})? Output `B` or `W` inside <answer>...</answer>.",
]


_TEMPLATES = [
    "Your task is to complete the {n}x{n} Yin-Yang puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Each cell holds either B (black) or W (white).\n"
    "2. All B cells must form a single 4-orthogonally connected region.\n"
    "3. All W cells must form a single 4-orthogonally connected region.\n"
    "4. No 2x2 sub-grid may contain four cells of the same color.\n"
    "5. Given (non-empty) cells must be preserved.\n\n"
    "### Coordinate System:\n"
    "- The grid is {n}x{n}, indexed (row, col); row 0 is the top, col 0 is the left.\n"
    "- A cell labelled `.` is currently empty and must be filled.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Provide the completed grid as {n} rows of B/W characters joined by `|` (single line; newlines also accepted), inside <answer>...</answer>.\n"
    "Example: <answer>{example}</answer>",

    "Solve the {n}x{n} Yin-Yang puzzle described below.\n\n"
    "### Game Rules:\n"
    "- Fill every empty cell with B (black) or W (white).\n"
    "- All B cells must be 4-connected; all W cells must be 4-connected.\n"
    "- No 2x2 sub-grid may be monochromatic (all B or all W).\n"
    "- Pre-filled cells are fixed.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n} grid; rows numbered 0..{n_minus_one} top-to-bottom, columns 0..{n_minus_one} left-to-right.\n"
    "- `.` marks an empty cell.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the solved grid as {n} rows of B/W characters joined by `|` (e.g., `BWB|WBW|BWB`), inside <answer>...</answer>.",

    "Complete the {n}x{n} Yin-Yang puzzle below.\n\n"
    "### Game Rules:\n"
    "Fill the grid with two symbols (B/W or equivalently 1/0) such that: (a) each color forms a single 4-connected region; (b) no 2x2 block is monochromatic; (c) given clues are preserved.\n\n"
    "### Coordinate System:\n"
    "- Rows 0..{n_minus_one} top-to-bottom; columns 0..{n_minus_one} left-to-right.\n"
    "- `.` = empty cell.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the completed grid as {n} rows of B/W (or 0/1) characters joined by `|`, inside <answer>...</answer>.",
]


def _check_solution(grid: List[List[str]]) -> bool:
    n = len(grid)
    m = len(grid[0]) if n else 0
    # No 2x2 monochrome
    for i in range(n - 1):
        for j in range(m - 1):
            cells = [grid[i][j], grid[i + 1][j], grid[i][j + 1], grid[i + 1][j + 1]]
            if len(set(cells)) == 1:
                return False
    # Connectivity per color
    for color in ("B", "W"):
        starts = [(i, j) for i in range(n) for j in range(m) if grid[i][j] == color]
        if not starts:
            continue
        from collections import deque
        seen = set()
        q = deque([starts[0]])
        seen.add(starts[0])
        while q:
            i, j = q.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < m and (ni, nj) not in seen \
                        and grid[ni][nj] == color:
                    seen.add((ni, nj))
                    q.append((ni, nj))
        if len(seen) != len(starts):
            return False
    return True


def _solve_yinyang(partial: List[List[str]]):
    n = len(partial)
    m = len(partial[0])
    cells = [(i, j) for i in range(n) for j in range(m) if partial[i][j] == "."]

    def backtrack(idx, grid):
        if idx == len(cells):
            return grid if _check_solution(grid) else None
        i, j = cells[idx]
        for color in ("B", "W"):
            grid[i][j] = color
            # Quick prune: no 2x2 monochrome partial check
            ok = True
            for di in (-1, 0):
                for dj in (-1, 0):
                    ti, tj = i + di, j + dj
                    if 0 <= ti < n - 1 and 0 <= tj < m - 1:
                        cs = [grid[ti][tj], grid[ti + 1][tj],
                              grid[ti][tj + 1], grid[ti + 1][tj + 1]]
                        if "." not in cs and len(set(cs)) == 1:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                res = backtrack(idx + 1, grid)
                if res is not None:
                    return res
            grid[i][j] = "."
        return None

    return backtrack(0, [row[:] for row in partial])


class YinYangGridQA(StandaloneVisualEnv):
    ENV_NAME = "yin_yang_grid"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the completed grid (rows of B/W chars, "
        "newline-separated) directly inside `<answer>...</answer>`."
    )

    def _random_valid_full(self, n: int, rng: random.Random):
        """Backtracking generator for a valid full Yin-Yang grid.
        Constraints enforced incrementally: no 2x2 monochrome (local check),
        and final connectivity check at the end.
        2026-05-04: bumped attempts_left budget 800->5000 to fix L6/L9 GEN flakes."""
        grid = [["." for _ in range(n)] for _ in range(n)]
        cells = [(i, j) for i in range(n) for j in range(n)]
        # Random ordering of color attempts at each cell
        def backtrack(idx, attempts_left=[5000]):
            if attempts_left[0] <= 0:
                return None
            attempts_left[0] -= 1
            if idx == len(cells):
                return [row[:] for row in grid] if _check_solution(grid) else None
            i, j = cells[idx]
            order = ["B", "W"]
            rng.shuffle(order)
            for c in order:
                grid[i][j] = c
                # local 2x2 prune: check the 2x2 ending at (i, j)
                ok = True
                if i >= 1 and j >= 1:
                    cs = [grid[i - 1][j - 1], grid[i - 1][j], grid[i][j - 1], grid[i][j]]
                    if len(set(cs)) == 1:
                        ok = False
                if ok:
                    res = backtrack(idx + 1, attempts_left)
                    if res is not None:
                        return res
                grid[i][j] = "."
            return None
        return backtrack(0)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — PROBE one cell
        # at L0/L1, with most cells revealed.
        # Cap n at 4: 5x5 random-backtracking generator is not reliable enough
        # to ship at curriculum-top (validated empirically — random Yin-Yang
        # validity rate is ~0.1% at 4x4 and ~0% at 5x5; backtracking succeeds
        # at 4x4 within attempt budget but flakes at 5x5).
        n = 3 if level <= 3 else 4
        # density of clues: more at low level, fewer at high level
        clue_density = max(0.35, 0.75 - level * 0.04)
        probe = level <= 1
        return {"level": level, "n": n, "clue_density": clue_density,
                "probe": probe}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 1117 + level * 113 + 31)

        # Build a valid full grid via constrained backtracking (random fill is
        # too sparse: only ~0.1% of 4x4 random grids satisfy all rules).
        full = self._random_valid_full(n, rng)
        if full is None:
            return None

        # Reveal random subset of cells
        partial = [["." for _ in range(n)] for _ in range(n)]
        n_reveal = max(2, int(n * n * cfg["clue_density"]))
        idxs = list(range(n * n))
        rng.shuffle(idxs)
        for k in idxs[:n_reveal]:
            i, j = divmod(k, n)
            partial[i][j] = full[i][j]

        # Verify the puzzle is solvable from partial — and store one valid solution
        sol = _solve_yinyang(partial)
        if sol is None:
            return None
        # If multiple solutions exist (puzzle not unique), accept first sol; check_answer will accept any valid completion.
        self._partial = partial
        self._n = n
        self._probe_mode = cfg.get("probe", False)
        self._solution = sol

        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — PROBE one cell.
        if self._probe_mode:
            empty_cells = [(i, j) for i in range(n) for j in range(n)
                           if partial[i][j] == "."]
            if not empty_cells:
                return None
            target_r, target_c = rng.choice(empty_cells)
            target_color = sol[target_r][target_c]
            self._probe_target = (target_r, target_c)
            self._probe_color = target_color
            sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
            state_str = self._format_state(partial)
            question = _PROBE_TEMPLATES[sidx].format(
                n=n, state=state_str, r=target_r, c=target_c,
            )
            img = self._render(partial, n)
            return question, target_color, img

        # Use `|` as row separator so the GT survives "Final answer:"
        # extraction (regex stops at \n). Verifier accepts both `|` and `\n`.
        ans_str = "|".join("".join(row) for row in sol)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_str = self._format_state(partial)
        # Tiny example for the first template — `|`-separated B/W rows
        # (matches the GT separator used for `Final answer:` extraction).
        example_rows = []
        for i in range(n):
            example_rows.append("".join("B" if (i + j) % 2 == 0 else "W"
                                        for j in range(n)))
        example = "|".join(example_rows)
        question = _TEMPLATES[sidx].format(
            n=n, n_minus_one=n - 1, state=state_str, example=example,
        )
        img = self._render(partial, n)
        return question, ans_str, img

    @staticmethod
    def _format_state(partial: List[List[str]]) -> str:
        """Render the partial grid as newline-joined B/W/. rows."""
        return "\n".join("".join(row) for row in partial)

    def _render(self, partial: List[List[str]], n: int) -> Image.Image:
        cell = 0.7
        fig, ax = plt.subplots(figsize=(0.7 * n + 1, 0.7 * n + 1), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for i in range(n):
            for j in range(n):
                v = partial[i][j]
                if v == "B":
                    fc = "#222"
                    txt_color = "white"
                elif v == "W":
                    fc = "#f5f5f5"
                    txt_color = "#222"
                else:
                    fc = "#e8eef5"
                    txt_color = "#888"
                ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                           facecolor=fc, edgecolor="#2c3e50",
                                           linewidth=1.0))
                if v != ".":
                    ax.text(j + 0.5, n - 0.5 - i, v, ha="center", va="center",
                            fontsize=14, color=txt_color, fontweight="bold")
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — PROBE single letter.
        if getattr(self, "_probe_mode", False):
            import re as _re
            text = predicted.strip().upper()
            m = _re.search(r"\b([BW])\b", text)
            if m:
                return m.group(1) == self._probe_color
            return False
        # Decode literal "\n" the model sometimes emits inside <answer>.
        if "\\n" in predicted:
            predicted = predicted.replace("\\n", "\n").replace("\\t", "\t")
        # Accept `|` and `/` as alternate row separators (so single-line
        # predictions like `WBB|WBW|WWW` work — important for the
        # `Final answer: ...` extractor which stops at \n).
        predicted = predicted.replace("|", "\n").replace("/", "\n")
        # Parse predicted as rows of chars (strip optional intra-line spaces +
        # commas — the benchmark format uses 0/1 sometimes space- or
        # comma-separated, e.g. "0 1 1 0 1 0").
        rows = [r.strip().replace(" ", "").replace(",", "")
                for r in predicted.strip().splitlines()
                if r.strip().replace(" ", "").replace(",", "")]
        if len(rows) != self._n:
            return False
        # Detect symbol set: B/W (env native) or 0/1 (benchmark — the puzzle is
        # color-symmetric so we try both 0→B,1→W and 0→W,1→B mappings and
        # accept either; this avoids guessing the canonical mapping).
        # Validates length of each row first.
        for r in rows:
            if len(r) != self._n:
                return False

        def _validate_with_symbols(parsed_rows):
            # Check each row consists of B/W only
            for r in parsed_rows:
                if any(ch not in "BW" for ch in r):
                    return False
            # Check it agrees with the partial clues
            for i in range(self._n):
                for j in range(self._n):
                    if self._partial[i][j] != "." \
                            and self._partial[i][j] != parsed_rows[i][j]:
                        return False
            # Check it satisfies the rules
            return _check_solution([list(r) for r in parsed_rows])

        # ---- Case A: B/W native env format
        if all(all(ch in "BW" for ch in r) for r in rows):
            return _validate_with_symbols(rows)

        # ---- Case B: 0/1 benchmark format. Try both mappings.
        if all(all(ch in "01" for ch in r) for r in rows):
            # Mapping 1: 1→B, 0→W
            mapped_a = ["".join("B" if ch == "1" else "W" for ch in r)
                        for r in rows]
            if _validate_with_symbols(mapped_a):
                return True
            # Mapping 2: 0→B, 1→W
            mapped_b = ["".join("B" if ch == "0" else "W" for ch in r)
                        for r in rows]
            if _validate_with_symbols(mapped_b):
                return True
            return False

        # Mixed or invalid characters — reject.
        return False
