"""
Aquarium-style region-fill puzzle.

2026-05-05 R5 P4: REWRITTEN with single-cell probe MCQ at L0-L4 to give
4B training a learnable signal. Full solve at L5-L9 constrained to small
4x4 D=1 puzzles per the benchmark D=1 sample style.

Rules (standard Aquarium):
  - NxN grid divided into REGIONS (each cell belongs to a region; regions
    are 4-connected sets, like aquarium tanks).
  - Each row has a count clue: # of W (water) cells in that row.
  - Each column has a count clue: # of W cells in that column.
  - Fill each cell with W (water) or A (air) such that:
      * row counts and column counts match the clues, and
      * within each region, water fills "from the bottom":
        if a cell is W, then every cell BELOW it in the SAME region
        (same column, lower row, and same region label) is also W.

Per-level layout:
  L0/L1: single-cell probe MCQ — 4x4 puzzle, most cells revealed (W or A),
         ask the color of ONE highlighted (?) cell. 4-MCQ (W / A / Cannot
         determine / No correct).
  L2/L3: 4x4 puzzle, single ? probe (no extra cells revealed beyond clues).
  L4   : 4x4 puzzle, 2 ? cells, ask one. 5-MCQ + 15% trap rate.
  L5-L7: full-solve 4x4 D=1 (text answer).
  L8/L9: full-solve 4x4 D=1 wrapped as MCQ candidate-grid + 30-40% trap.
"""
import random
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ----------------------------------------------------------------------- #
# Probe MCQ templates (L0-L4)
# ----------------------------------------------------------------------- #
_PROBE_TEMPLATES = [
    "Look at the {n}x{n} water-fill puzzle below.\n\n"
    "### Game Rules:\n"
    "1. The grid is divided into regions (\"tanks\"). Each cell is W (water) or A (air).\n"
    "2. Within any region, water settles to the bottom: if a cell is W, every cell directly below it in the same region must also be W.\n"
    "3. The number of W cells in each row equals the row clue; same for columns.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed; row 0 at top.\n"
    "- Each cell carries a region id integer.\n\n"
    "### Current Puzzle State:\n"
    "- Region grid:\n{state}\n"
    "- Row clues: {row_clues}\n"
    "- Column clues: {col_clues}\n"
    "- Already-decided cells: {known}\n\n"
    "### Question:\n"
    "Should the highlighted (?) cell at (row {r}, col {c}) contain Water (W) or Air (A)?\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "{n}x{n} water-fill puzzle.\n\n"
    "### Game Rules:\n"
    "- Each cell is W (water) or A (air).\n"
    "- Within a region, water fills from the bottom (no floating water).\n"
    "- Row/column water counts match the given clues.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Region map:\n{state}\n"
    "Row clues: {row_clues}\n"
    "Column clues: {col_clues}\n"
    "Already-decided: {known}\n\n"
    "### Question:\n"
    "What goes in the highlighted (?) cell at (row {r}, col {c})?\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",

    "Water-fill puzzle ({n}x{n}). Within each region water occupies a bottom-up filled level; row/column water counts match the clues.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Region map:\n{state}\n"
    "Row clues: {row_clues}; Column clues: {col_clues}\n"
    "Already known: {known}\n\n"
    "### Question:\n"
    "Determine the contents of the (?) cell at (row {r}, col {c}).\n"
    "{options}\n"
    "Answer with a single letter {letter_str}.",
]


# ----------------------------------------------------------------------- #
# Full-solve text templates (L5-L7)
# ----------------------------------------------------------------------- #
_TEMPLATES = [
    "Your task is to solve the {n}x{n} water-fill puzzle described below.\n\n"
    "### Game Rules:\n"
    "1. The grid is divided into regions (\"tanks\"). Each cell is W (water) or A (air).\n"
    "2. Within each region, water settles to the bottom: if a cell is W, then every cell directly below it in the same region must also be W.\n"
    "3. The number of W cells in each row equals the row clue; same for columns.\n\n"
    "### Coordinate System:\n"
    "- Grid is {n}x{n}, indexed (row, col) with row 0 at top. Cells given as `(x=column, y=row)` pairs.\n"
    "- Each cell carries a region id integer.\n\n"
    "### Current Puzzle State:\n"
    "- Region grid:\n{state}\n"
    "- Row clues: {row_clues}\n"
    "- Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "Output the list of water-cell coordinates as `(x, y)` tuples on one line (0-indexed; x = column, y = row).\n"
    "Example: [(0, 1), (1, 1), (3, 2)]",

    "Solve the water-fill puzzle below.\n\n"
    "### Game Rules:\n"
    "- Each cell is Water (W) or Air (A).\n"
    "- Within any region, water occupies a bottom-up filled level.\n"
    "- Row and column water counts must match the given clues.\n\n"
    "### Coordinate System:\n"
    "- {n}x{n}; coordinates as `(x=column, y=row)`, 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "Region map:\n{state}\n"
    "Row clues: {row_clues}\n"
    "Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "List the water cells as `[(x, y), (x, y), ...]` 0-indexed on one line.",

    "Your task is to solve the {n}x{n} water-fill puzzle described below.\n\n"
    "### Game Rules:\n"
    "Each region acts like a tank: water sits at the bottom.\n\n"
    "### Coordinate System:\n"
    "- (x = column, y = row), 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n"
    "Row clues: {row_clues}; Column clues: {col_clues}\n\n"
    "### Output Format:\n"
    "Output the water-cell coordinates as `[(x, y), ...]` on one line.",
]


def _region_to_text(region, n):
    rows = []
    for r in range(n):
        rows.append(" ".join(str(region[r][c]) for c in range(n)))
    return "\n".join(rows)


def _build_regions(n: int, num_regions: int, rng: random.Random) -> List[List[int]]:
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    seeds = cells[:num_regions]
    region = [[-1] * n for _ in range(n)]
    from collections import deque
    queues = [deque([s]) for s in seeds]
    for ri, s in enumerate(seeds):
        region[s[0]][s[1]] = ri
    while any(queues):
        for ri, q in enumerate(queues):
            if not q:
                continue
            r, c = q.popleft()
            neighbors = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if 0 <= nr < n and 0 <= nc < n and region[nr][nc] == -1:
                    region[nr][nc] = ri
                    q.append((nr, nc))
    return region


def _is_aquarium_valid(grid, region, row_counts, col_counts, n):
    for r in range(n):
        cnt = sum(1 for c in range(n) if grid[r][c] == "W")
        if cnt != row_counts[r]:
            return False
    for c in range(n):
        cnt = sum(1 for r in range(n) if grid[r][c] == "W")
        if cnt != col_counts[c]:
            return False
    for r in range(n):
        for c in range(n):
            if grid[r][c] == "W":
                for rp in range(r + 1, n):
                    if region[rp][c] == region[r][c] and grid[rp][c] != "W":
                        return False
    return True


def _gen_aquarium_solution(n, region, rng):
    grid = [["A"] * n for _ in range(n)]
    region_ids = set()
    for row in region:
        for v in row:
            region_ids.add(v)
    for ri in region_ids:
        cells = [(r, c) for r in range(n) for c in range(n) if region[r][c] == ri]
        rows_in = sorted(set(r for r, c in cells))
        choices = rows_in + [n]
        surface = rng.choice(choices)
        for r, c in cells:
            if r >= surface:
                grid[r][c] = "W"
    return grid


def _solve_aquarium(n, region, row_counts, col_counts, limit=1):
    region_ids = sorted(set(v for row in region for v in row))
    region_choices = {}
    for ri in region_ids:
        cells = [(r, c) for r in range(n) for c in range(n) if region[r][c] == ri]
        rows_in = sorted(set(r for r, c in cells))
        region_choices[ri] = rows_in + [n]

    grid = [["A"] * n for _ in range(n)]

    def fill_region(ri, surface):
        for r in range(n):
            for c in range(n):
                if region[r][c] == ri:
                    grid[r][c] = "W" if r >= surface else "A"

    found = []

    def backtrack(idx):
        if len(found) >= limit:
            return
        if idx == len(region_ids):
            if _is_aquarium_valid(grid, region, row_counts, col_counts, n):
                found.append([row[:] for row in grid])
            return
        ri = region_ids[idx]
        for surface in region_choices[ri]:
            fill_region(ri, surface)
            partial_row = [sum(1 for c in range(n) if grid[r][c] == "W")
                           for r in range(n)]
            partial_col = [sum(1 for r in range(n) if grid[r][c] == "W")
                           for c in range(n)]
            ok = all(partial_row[r] <= row_counts[r] for r in range(n)) \
                 and all(partial_col[c] <= col_counts[c] for c in range(n))
            if ok:
                backtrack(idx + 1)
            if len(found) >= limit:
                return

    backtrack(0)
    if found:
        return found
    return []


def _count_aquarium_solutions(n, region, row_counts, col_counts, limit=2):
    return len(_solve_aquarium(n, region, row_counts, col_counts, limit=limit))


class AquariumGridQA(StandaloneVisualEnv):
    ENV_NAME = "aquarium_grid"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0/L1: 4x4, 2 regions, single ? probe
        if level <= 1:
            return {"level": level, "n": 4, "num_regions": 2,
                    "mode": "probe", "n_unknown": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        # L2/L3: 4x4, 2 regions, single ? probe
        if level <= 3:
            return {"level": level, "n": 4, "num_regions": 2,
                    "mode": "probe", "n_unknown": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        # L4: 4x4 with 2 ? cells, 5-MCQ + 15% trap
        if level == 4:
            return {"level": level, "n": 4, "num_regions": 2,
                    "mode": "probe", "n_unknown": 2,
                    "use_5_options": True, "trap_rate": 0.15}
        # L5-L7: full-solve text 4x4 D=1
        if level <= 7:
            return {"level": level, "n": 4, "num_regions": 2,
                    "mode": "full", "use_5_options": False,
                    "trap_rate": 0.0}
        # L8/L9: full-solve as MCQ candidate-grid + raised trap
        return {"level": level, "n": 4, "num_regions": 2,
                "mode": "full", "use_5_options": True,
                "trap_rate": 0.40 if level == 9 else 0.30}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        num_regions = cfg["num_regions"]
        rng = random.Random((self.seed or 0) * 4001 + level * 37 + 7)
        mode = cfg["mode"]

        for attempt in range(120):
            region = _build_regions(n, num_regions, rng)
            grid = _gen_aquarium_solution(n, region, rng)
            row_counts = [sum(1 for c in range(n) if grid[r][c] == "W")
                          for r in range(n)]
            col_counts = [sum(1 for r in range(n) if grid[r][c] == "W")
                          for c in range(n)]
            total = sum(row_counts)
            if total == 0 or total == n * n:
                continue
            # Need uniqueness
            sols = _solve_aquarium(n, region, row_counts, col_counts, limit=2)
            if not sols:
                continue
            if len(sols) != 1:
                continue
            self._n = n
            self._region = region
            self._row_counts = row_counts
            self._col_counts = col_counts
            self._solution = grid
            self._mcq_mode = (mode == "probe") or cfg.get("use_5_options", False)

            if mode == "probe":
                return self._build_probe_question(
                    cfg, rng, grid, region, row_counts, col_counts, n, level,
                )
            else:
                return self._build_full_question(
                    cfg, rng, grid, region, row_counts, col_counts, n, level,
                )
        return None

    # ------------------------------------------------------------------ #
    # L0-L4 probe MCQ
    # ------------------------------------------------------------------ #
    def _build_probe_question(self, cfg, rng, grid, region, row_counts,
                              col_counts, n, level):
        # Pick `n_unknown` cells to hide
        all_cells = [(r, c) for r in range(n) for c in range(n)]
        n_unknown = cfg["n_unknown"]
        if len(all_cells) < n_unknown:
            return None
        hidden = list(rng.sample(all_cells, n_unknown))
        target = hidden[0]
        target_r, target_c = target
        true_color_letter = grid[target_r][target_c]  # 'W' or 'A'
        true_color = "Water (W)" if true_color_letter == "W" else "Air (A)"
        self._known_cells = {(r, c): grid[r][c]
                              for (r, c) in all_cells if (r, c) not in hidden}

        known_pairs = sorted((cell, val) for cell, val in self._known_cells.items())
        known_str = (", ".join(f"({r},{c})={v}" for (r, c), v in known_pairs)
                     or "(none)")

        # Options
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][: (5 if use_5 else 4)]

        # Standard 4-option layout: A. Water B. Air C. Cannot determined D. No correct
        # 5-option: A. Water B. Air C. Cannot determined D. Both possible E. No correct
        if use_5:
            base = ["Water (W)", "Air (A)",
                    "Cannot be determined", "Both Water and Air are valid"]
            rng.shuffle(base)
            opt_texts = base + ["No correct answer"]
            true_text = true_color
            use_trap = (cfg.get("trap_rate", 0.0) > 0
                        and rng.random() < cfg["trap_rate"])
            if use_trap:
                opt_texts = [t for t in opt_texts[:4] if t != true_text]
                pad = ["Either Water or Air",
                       "Insufficient clues",
                       "Depends on the queried column"]
                rng.shuffle(pad)
                pi = 0
                while len(opt_texts) < 4:
                    if pad[pi] not in opt_texts:
                        opt_texts.append(pad[pi])
                    pi = (pi + 1) % len(pad)
                rng.shuffle(opt_texts)
                opt_texts.append("No correct answer")
                correct_letter = opt_letters[-1]
            else:
                if true_text not in opt_texts[:4]:
                    opt_texts[0] = true_text
                correct_letter = opt_letters[opt_texts.index(true_text)]
        else:
            # 4-MCQ
            base = ["Water (W)", "Air (A)", "Cannot be determined"]
            rng.shuffle(base)
            opt_texts = base + ["No correct answer"]
            correct_letter = opt_letters[opt_texts.index(true_color)]

        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
        state_text = _region_to_text(region, n)
        question = _PROBE_TEMPLATES[sidx].format(
            n=n, state=state_text,
            row_clues=str(row_counts), col_clues=str(col_counts),
            known=known_str,
            r=target_r, c=target_c,
            options=opt_lines, letter_str=letter_str,
        )
        self._probe_target = target
        self._probe_color = true_color
        img = self._render(n, region, row_counts, col_counts,
                           known_cells=self._known_cells,
                           highlight_cell=target)
        return question, correct_letter, img

    # ------------------------------------------------------------------ #
    # L5-L9 full-solve / MCQ wrapper
    # ------------------------------------------------------------------ #
    def _build_full_question(self, cfg, rng, grid, region, row_counts,
                             col_counts, n, level):
        if cfg.get("use_5_options", False):
            return self._build_full_mcq_question(
                cfg, rng, grid, region, row_counts, col_counts, n, level,
            )
        # Single-line tuple-list format so the answer fits on one line.
        # The verifier accepts both (x=col, y=row) tuple-list and W/A grid
        # formats (see _check_answer below).
        water_cells = [(c, r) for r in range(n) for c in range(n)
                       if grid[r][c] == "W"]
        ans_str = "[" + ", ".join(f"({x}, {y})" for x, y in water_cells) + "]"
        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_text = _region_to_text(region, n)
        question = _TEMPLATES[sidx].format(
            n=n, state=state_text,
            row_clues=str(row_counts), col_clues=str(col_counts),
        )
        img = self._render(n, region, row_counts, col_counts)
        self._mcq_mode = False
        return question, ans_str, img

    def _build_full_mcq_question(self, cfg, rng, grid, region, row_counts,
                                 col_counts, n, level):
        # Single-line tuple-list representation
        def _grid_to_list(g):
            return tuple(sorted([(c, r) for r in range(n) for c in range(n)
                                  if g[r][c] == "W"]))

        def _list_to_str(lst):
            return "[" + ", ".join(f"({x}, {y})" for x, y in lst) + "]"

        true_list = _grid_to_list(grid)
        true_str = _list_to_str(true_list)

        distractor_set = set()
        attempts = 0
        all_cells = [(r, c) for r in range(n) for c in range(n)]
        while len(distractor_set) < 6 and attempts < 200:
            attempts += 1
            new_grid = [row[:] for row in grid]
            n_mut = rng.choice([1, 2])
            for (r, c) in rng.sample(all_cells, n_mut):
                new_grid[r][c] = "A" if new_grid[r][c] == "W" else "W"
            new_list = _grid_to_list(new_grid)
            if new_list == true_list:
                continue
            distractor_set.add(new_list)

        if len(distractor_set) < 3:
            return None

        distractors = list(distractor_set)
        rng.shuffle(distractors)

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
            options_data = [true_list] + distractors[:n_distractor]
            rng.shuffle(options_data)
            correct_letter = opt_letters[options_data.index(true_list)]

        opt_texts_full = [_list_to_str(s) for s in options_data]
        opt_texts_full.append("No correct answer")
        opt_lines = "\n".join(f"{opt_letters[i]}. {opt_texts_full[i]}"
                              for i in range(len(opt_letters)))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        state_text = _region_to_text(region, n)
        stem = (
            f"Look at the {n}x{n} water-fill puzzle.\n\n"
            "### Game Rules:\n"
            "1. Each cell is W (water) or A (air).\n"
            "2. Within a region, water settles at the bottom (no floating).\n"
            "3. Row/column water counts match the clues.\n\n"
            "### Coordinate System:\n"
            "- (x = column, y = row), 0-indexed.\n\n"
            "### Current Puzzle State:\n"
            f"Region grid:\n{state_text}\n"
            f"Row clues: {row_counts}\n"
            f"Column clues: {col_counts}\n\n"
            "### Question:\n"
            "Which option lists the water cells `(x, y)` in the unique solution?\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        img = self._render(n, region, row_counts, col_counts)
        self._mcq_mode = True
        return stem, correct_letter, img

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, n, region, row_counts, col_counts,
                known_cells=None, highlight_cell=None) -> Image.Image:
        fig, ax = plt.subplots(figsize=(0.95 * (n + 2), 0.95 * (n + 2)), dpi=140)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        palette = ["#FFE5E5", "#E5F3FF", "#E5FFE5", "#FFF8E5",
                   "#F0E5FF", "#FFE5F8", "#E0F7FA", "#FBE9E7"]
        for r in range(n):
            for c in range(n):
                ri = region[r][c]
                fc = palette[ri % len(palette)]
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                           facecolor=fc, edgecolor="#888",
                                           linewidth=0.6))
                # Overlay water/air or ? if applicable
                is_highlight = (highlight_cell is not None
                                and (r, c) == highlight_cell)
                if is_highlight:
                    # Draw thick orange border + ? overlay
                    ax.add_patch(plt.Rectangle(
                        (c + 0.05, n - 1 - r + 0.05), 0.9, 0.9,
                        facecolor="#fff5b1", edgecolor="#cc6600",
                        linewidth=3.0))
                    ax.text(c + 0.5, n - 0.5 - r, "?",
                            ha="center", va="center", fontsize=22,
                            fontweight="bold", color="#cc6600")
                elif known_cells is not None and (r, c) in known_cells:
                    val = known_cells[(r, c)]
                    if val == "W":
                        # Blue tint for water
                        ax.add_patch(plt.Rectangle(
                            (c + 0.1, n - 1 - r + 0.1), 0.8, 0.8,
                            facecolor="#3498db", edgecolor="#1d5d9b",
                            linewidth=1.2, alpha=0.85))
                        ax.text(c + 0.5, n - 0.5 - r, "W",
                                ha="center", va="center", fontsize=14,
                                fontweight="bold", color="white")
                    else:
                        # A: light gray text
                        ax.text(c + 0.5, n - 0.5 - r, "A",
                                ha="center", va="center", fontsize=12,
                                color="#777", fontweight="bold")

        # Bold borders between regions
        for r in range(n):
            for c in range(n):
                ri = region[r][c]
                if r == 0 or region[r - 1][c] != ri:
                    ax.plot([c, c + 1], [n - r, n - r], color="black", linewidth=2.2)
                if r == n - 1 or region[r + 1][c] != ri:
                    ax.plot([c, c + 1], [n - 1 - r, n - 1 - r], color="black", linewidth=2.2)
                if c == 0 or region[r][c - 1] != ri:
                    ax.plot([c, c], [n - 1 - r, n - r], color="black", linewidth=2.2)
                if c == n - 1 or region[r][c + 1] != ri:
                    ax.plot([c + 1, c + 1], [n - 1 - r, n - r], color="black", linewidth=2.2)
        # Column counts (bottom)
        for c in range(n):
            ax.text(c + 0.5, -0.25, str(col_counts[c]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1d3557")
        # Row counts (right)
        for r in range(n):
            ax.text(n + 0.25, n - 0.5 - r, str(row_counts[r]),
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#1d3557")
        ax.set_xlim(-0.05, n + 0.65)
        ax.set_ylim(-0.55, n + 0.05)
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
        # Full-solve text mode
        if "\\n" in predicted:
            predicted = predicted.replace("\\n", "\n").replace("\\t", "\t")
        n = self._n
        # First try benchmark coord-list format: list of `(x=col, y=row)`
        # tuples. This is the dominant format we emit; check it first because
        # the W/A-grid branch's tokenizer can mistakenly split a tuple-list
        # like `[(2, 3), (3, 3)]` into n tokens for small n.
        tuple_pairs = re.findall(
            r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", predicted
        )
        if tuple_pairs:
            water_cells = set()
            for x_str, y_str in tuple_pairs:
                try:
                    x = int(x_str)
                    y = int(y_str)
                except ValueError:
                    return False
                if not (0 <= x < n and 0 <= y < n):
                    return False
                water_cells.add((y, x))
            tuple_grid = [["A"] * n for _ in range(n)]
            for r in range(n):
                for c in range(n):
                    if (r, c) in water_cells:
                        tuple_grid[r][c] = "W"
            return _is_aquarium_valid(
                tuple_grid, self._region, self._row_counts,
                self._col_counts, n,
            )
        # Fall back to W/A grid format
        rows = [r.strip() for r in predicted.strip().splitlines() if r.strip()]
        if len(rows) != n:
            tokens = re.split(r"[\s,;]+", predicted.strip())
            tokens = [t.strip() for t in tokens if t.strip()]
            if len(tokens) == n:
                rows = tokens
            else:
                return False
        grid = []
        for row in rows:
            row = row.strip().upper().replace(",", "").replace(" ", "")
            if len(row) != n:
                return False
            if any(ch not in "WA" for ch in row):
                return False
            grid.append(list(row))
        return _is_aquarium_valid(grid, self._region, self._row_counts,
                                  self._col_counts, n)
