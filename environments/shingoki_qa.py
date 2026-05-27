"""
Shingoki-style loop puzzle on an NxN grid.

Each cell is one of:
  .       — empty (loop may or may not pass)
  W<n>    — white circle, the loop must pass STRAIGHT through this cell;
            sum of the lengths of the two segments before/after the cell
            (counted along the loop direction) equals n
  B<n>    — black circle, the loop must TURN 90° at this cell;
            sum of the lengths of the two segments meeting at the cell
            equals n

Goal: draw a single closed loop that visits all circle cells, satisfies
each circle's constraint, and uses no edge twice.

Output format (one edge per token): a list of grid edges
    (r1,c1)-(r2,c2) (r2,c2)-(r3,c3) ...
1-indexed. Order doesn't matter; only the set of edges matters.

L0: 3x3 grid, 2x2 loop in the upper-left corner with two B2 circles at
opposite corners — the unique trivial loop.
"""
import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to solve the Shingoki puzzle according to the rules and current state below:\n\n"
    "### Game Rules:\n"
    "1. Draw a single closed loop on the grid edges (orthogonal segments between adjacent cells).\n"
    "2. The loop passes through every circle cell.\n"
    "3. White circle `W n`: the loop passes STRAIGHT through this cell, and the sum of the two collinear straight segments through it equals n.\n"
    "4. Black circle `B n`: the loop TURNS 90 degrees at this cell, and the sum of the two perpendicular segments meeting at it equals n.\n\n"
    "### Coordinate System:\n"
    "- Cells are addressed (row, col), 1-indexed.\n"
    "- Each grid string entry is `.`, `Wn`, or `Bn`.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output the loop's edges as space-separated `(r1,c1)-(r2,c2)` (1-indexed adjacent cells) inside <answer>...</answer>.\n"
    "Example: <answer>(1,1)-(1,2) (1,2)-(1,3) ...</answer>",

    "Solve the Shingoki puzzle below.\n\n"
    "### Game Rules:\n"
    "- Draw one closed loop on grid edges.\n"
    "- White circles: straight pass-through; sum of the two collinear segments equals the number.\n"
    "- Black circles: 90 degree turn; sum of the two adjacent segments equals the number.\n\n"
    "### Coordinate System:\n"
    "- Cells (row, col), 1-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output `(r1,c1)-(r2,c2)` adjacent-cell edges separated by spaces inside <answer>...</answer>.",

    "Your task is to solve the Shingoki loop puzzle described below.\n\n"
    "### Game Rules:\n"
    "Standard Shingoki: one closed loop; W=straight (sum of collinear segments=n); B=90 degree turn (sum of segments=n).\n\n"
    "### Coordinate System:\n"
    "- 1-indexed (row, col).\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Output edges inside <answer>...</answer>.",
]


# L0-L2 probe: ask if a SPECIFIC cell is on the loop. YES/NO answer.
_PROBE_TEMPLATES = [
    "Your task is to determine if one cell lies on the Shingoki loop:\n\n"
    "### Game Rules:\n"
    "1. Draw a single closed loop on the grid edges between adjacent cells.\n"
    "2. The loop passes through every circle cell.\n"
    "3. White circle `W n`: loop goes STRAIGHT through, sum of two collinear segments = n.\n"
    "4. Black circle `B n`: loop TURNS 90 degrees, sum of two perpendicular segments = n.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 1-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "In the unique solution, does the loop pass through cell (row {r}, col {c})? Answer YES or NO inside <answer>...</answer>.",

    "Look at the Shingoki puzzle below.\n\n"
    "### Game Rules:\n"
    "Standard Shingoki rules.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 1-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Does cell (row {r}, col {c}) lie on the loop? Answer YES or NO inside <answer>...</answer>.",

    "Shingoki puzzle.\n\n"
    "### Game Rules:\n"
    "Standard Shingoki rules.\n\n"
    "### Coordinate System:\n"
    "- (row, col), 1-indexed.\n\n"
    "### Current Puzzle State:\n"
    "{state}\n\n"
    "### Output Format:\n"
    "Is cell (row {r}, col {c}) on the loop? Output YES or NO inside <answer>...</answer>.",
]


def _shingoki_state_text(cell_map, n):
    rows = []
    for r in range(n):
        rows.append(" ".join(cell_map[r]))
    return "\n".join(rows)


def _normalize_edge(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return edge as a sorted tuple so direction doesn't matter."""
    if a <= b:
        return (a, b)
    return (b, a)


def _gen_rectangular_loop(rng: random.Random, n: int) -> List[Tuple[int, int]]:
    """Generate a rectangular loop fitting in nxn grid.

    Returns the loop as a list of cell (row, col) coordinates in 0-indexed
    cyclic order — last cell adjacent to first.
    """
    # Pick top-left corner and dims s.t. loop fits.
    # Loop spans rows [r0..r1] and cols [c0..c1] with r1-r0 >= 1 and c1-c0 >= 1.
    max_h = n - 1
    max_w = n - 1
    if max_h < 1 or max_w < 1:
        return [(0, 0)]
    h = rng.randint(1, max_h)
    w = rng.randint(1, max_w)
    r0 = rng.randint(0, n - 1 - h)
    c0 = rng.randint(0, n - 1 - w)
    r1 = r0 + h
    c1 = c0 + w
    # Build perimeter cells in order.
    cells = []
    # Top row left -> right
    for c in range(c0, c1):
        cells.append((r0, c))
    # Right col top -> bottom
    for r in range(r0, r1):
        cells.append((r, c1))
    # Bottom row right -> left
    for c in range(c1, c0, -1):
        cells.append((r1, c))
    # Left col bottom -> top
    for r in range(r1, r0, -1):
        cells.append((r, c0))
    return cells


def _gen_l_shape_loop(rng: random.Random, n: int) -> Optional[List[Tuple[int, int]]]:
    """Generate an L-shaped loop (rectangle with a notch). Requires n >= 4."""
    if n < 4:
        return None
    for _ in range(20):
        h = rng.randint(2, n - 2)
        w = rng.randint(2, n - 2)
        r0 = rng.randint(0, n - 1 - h)
        c0 = rng.randint(0, n - 1 - w)
        r1 = r0 + h
        c1 = c0 + w
        # notch: chop off bottom-right block of size nh x nw
        nh = rng.randint(1, h - 1)
        nw = rng.randint(1, w - 1)
        # Build perimeter of the L shape:
        #   Top:  (r0, c0) → (r0, c1)
        #   Right top half: (r0, c1) → (r1-nh, c1)
        #   Notch top:      (r1-nh, c1) → (r1-nh, c1-nw)
        #   Notch right:    (r1-nh, c1-nw) → (r1, c1-nw)
        #   Bottom (left part): (r1, c1-nw) → (r1, c0)
        #   Left: (r1, c0) → (r0, c0)
        cells: List[Tuple[int, int]] = []
        # top edge
        for c in range(c0, c1):
            cells.append((r0, c))
        # right top
        for r in range(r0, r1 - nh):
            cells.append((r, c1))
        # notch top
        for c in range(c1, c1 - nw, -1):
            cells.append((r1 - nh, c))
        # notch right (going down)
        for r in range(r1 - nh, r1):
            cells.append((r, c1 - nw))
        # bottom
        for c in range(c1 - nw, c0, -1):
            cells.append((r1, c))
        # left
        for r in range(r1, r0, -1):
            cells.append((r, c0))
        # Validate uniqueness
        if len(set(cells)) == len(cells) and len(cells) >= 6:
            return cells
    return None


def _classify_loop(cells: List[Tuple[int, int]]) -> List[Dict]:
    """For each cell on the loop, return dict with:
       'cell': (r, c)
       'is_corner': bool (True ⇒ turn; False ⇒ straight pass-through)
       'sum': int — sum of straight-segment lengths in the two directions
              (if straight: lengths of two collinear segments adjacent to cell;
               if turn: lengths of two perpendicular segments adjacent to cell).
    """
    n_cells = len(cells)

    def dir_of(i, j):
        a = cells[i]
        b = cells[j]
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        # normalize to {-1, 0, 1}
        if dr != 0:
            dr = 1 if dr > 0 else -1
        if dc != 0:
            dc = 1 if dc > 0 else -1
        return (dr, dc)

    # For each cell index i, prev edge direction = (i-1 -> i),
    # next edge direction = (i -> i+1).
    classifications = []
    for i in range(n_cells):
        prev_d = dir_of((i - 1) % n_cells, i)
        next_d = dir_of(i, (i + 1) % n_cells)
        is_corner = (prev_d != next_d)
        # Compute "previous-direction straight run length" — the count of
        # consecutive edges before cell i sharing prev_d direction.
        len_prev = 0
        j = i
        while True:
            j_prev = (j - 1) % n_cells
            d = dir_of(j_prev, j)
            if d == prev_d:
                len_prev += 1
                j = j_prev
                if len_prev > n_cells:
                    break
            else:
                break
        # Compute "next-direction straight run length" forward from cell i.
        len_next = 0
        j = i
        while True:
            j_next = (j + 1) % n_cells
            d = dir_of(j, j_next)
            if d == next_d:
                len_next += 1
                j = j_next
                if len_next > n_cells:
                    break
            else:
                break
        classifications.append({
            "cell": cells[i],
            "is_corner": is_corner,
            "sum": len_prev + len_next,
        })
    return classifications


class ShingokiQA(StandaloneVisualEnv):
    ENV_NAME = "shingoki"
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the loop edges directly inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0..L2: PROBE mode (small puzzle, ask about one cell).
        if level == 0:
            return {"level": level, "n": 3, "trivial": True,
                    "shape": "rect", "n_circles": 2, "probe": True}
        if level == 1:
            return {"level": level, "n": 4, "trivial": False,
                    "shape": "rect", "n_circles": 3, "probe": True}
        if level == 2:
            return {"level": level, "n": 4, "trivial": False,
                    "shape": "rect", "n_circles": 3, "probe": True}
        if level <= 3:
            return {"level": level, "n": 4, "trivial": False,
                    "shape": "rect", "n_circles": 3 + level // 2, "probe": False}
        if level <= 6:
            return {"level": level, "n": 5, "trivial": False,
                    "shape": "rect" if level == 4 else "L",
                    "n_circles": 4 + (level - 4), "probe": False}
        return {"level": level, "n": 6, "trivial": False,
                "shape": "L", "n_circles": 5 + (level - 7), "probe": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1303 + level * 97 + 67)
        n = cfg["n"]

        if cfg.get("trivial"):
            # 3x3 grid, loop = perimeter of the upper-left 2x2 box.
            # Cells: (0,0), (0,1), (1,1), (1,0)
            cells = [(0, 0), (0, 1), (1, 1), (1, 0)]
        else:
            cells = None
            if cfg["shape"] == "rect":
                cells = _gen_rectangular_loop(rng, n)
            else:
                cells = _gen_l_shape_loop(rng, n)
                if cells is None:
                    cells = _gen_rectangular_loop(rng, n)
            if cells is None or len(cells) < 4:
                return None

        # Compute edges of the loop (consecutive cell pairs)
        loop_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for i in range(len(cells)):
            a = cells[i]
            b = cells[(i + 1) % len(cells)]
            loop_edges.append(_normalize_edge(a, b))

        classifications = _classify_loop(cells)

        # Pick which cells get circles.
        if cfg.get("trivial"):
            # Place B2 at two opposite corners on the loop. All four cells
            # are corners of a 2x2 loop, each with sum 1+1 = 2.
            # Choose (0,0) and (1,1).
            chosen_idx = []
            for i, info in enumerate(classifications):
                if info["cell"] in [(0, 0), (1, 1)]:
                    chosen_idx.append(i)
        else:
            # Random subset of cells. Make sure there's at least one corner
            # (B circle) so the puzzle is interesting.
            n_circles = min(cfg["n_circles"], len(classifications))
            indices = list(range(len(classifications)))
            rng.shuffle(indices)
            chosen_idx = indices[:n_circles]
            # ensure at least one corner included
            if not any(classifications[i]["is_corner"] for i in chosen_idx):
                # find first corner index and add it
                for j, info in enumerate(classifications):
                    if info["is_corner"]:
                        if j not in chosen_idx:
                            chosen_idx[-1] = j
                        break

        # Build cell map for image
        cell_map: List[List[str]] = [["." for _ in range(n)] for _ in range(n)]
        circles: List[Dict] = []
        for i in chosen_idx:
            info = classifications[i]
            r, c = info["cell"]
            ch = "B" if info["is_corner"] else "W"
            num = info["sum"]
            cell_map[r][c] = f"{ch}{num}"
            circles.append({
                "row": r,
                "col": c,
                "kind": ch,
                "num": num,
            })

        # Cache for verify
        self._sg_n = n
        self._sg_grid = cell_map  # list of strings per cell
        self._sg_circles = circles
        self._sg_solution_edges = sorted(loop_edges)
        self._probe_mode = cfg.get("probe", False)

        # Build set of cells on the loop.
        loop_cells_set = set(cells)
        self._sg_loop_cells = loop_cells_set

        if self._probe_mode:
            # Balanced 50/50 sampling between on-loop (YES) and off-loop (NO).
            on_loop = list(loop_cells_set)
            off_loop = [(r, c) for r in range(n) for c in range(n)
                        if (r, c) not in loop_cells_set]
            # Use seed-derived coin so we don't always pick same side.
            coin = rng.random() < 0.5
            if coin and on_loop:
                target = rng.choice(on_loop)
                ans = "YES"
            elif off_loop:
                target = rng.choice(off_loop)
                ans = "NO"
            elif on_loop:
                target = rng.choice(on_loop)
                ans = "YES"
            else:
                return None
            self._probe_target = target
            self._probe_answer = ans
            sidx = (self.seed or 0) % len(_PROBE_TEMPLATES)
            state_text = _shingoki_state_text(cell_map, n)
            # 1-indexed in question
            question = _PROBE_TEMPLATES[sidx].format(
                state=state_text, r=target[0] + 1, c=target[1] + 1,
            )
            # 2026-05-04: at L0/L1, append HINT giving the loop cells in
            # 1-indexed form. Solving Shingoki from constraint dots is too
            # hard for 4B; once told the loop, the model just checks
            # membership.
            if cfg.get("trivial") or level <= 1:
                loop_cells_text = ", ".join(
                    f"(row {r+1}, col {c+1})" for (r, c) in sorted(loop_cells_set)
                )
                question += (
                    f"\n\nHint: the puzzle's loop visits exactly these cells: "
                    f"{loop_cells_text}. Check whether your target cell is in "
                    f"this set."
                )
            img = self._render(cell_map, circles, n, rng)
            return question, ans, img

        # Canonical answer string
        ans_tokens = [f"({a[0] + 1},{a[1] + 1})-({b[0] + 1},{b[1] + 1})"
                      for (a, b) in self._sg_solution_edges]
        ans_str = " ".join(ans_tokens)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        state_text = _shingoki_state_text(cell_map, n)
        question = _TEMPLATES[sidx].format(state=state_text)
        img = self._render(cell_map, circles, n, rng)
        return question, ans_str, img

    def _render(self, cell_map, circles, n, rng):
        cell_size = 0.9
        fig_w = max(5.0, n * cell_size + 1.5)
        fig_h = max(5.0, n * cell_size + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Draw cells. Use (col, n-1-row) → cartesian so row 1 is at top.
        for r in range(n):
            for c in range(n):
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                           facecolor="#fafafa",
                                           edgecolor="#888888",
                                           linewidth=1.0))
        # Draw circles + numbers
        for circ in circles:
            r = circ["row"]; c = circ["col"]
            kind = circ["kind"]; num = circ["num"]
            cx = c + 0.5
            cy = n - 1 - r + 0.5
            if kind == "W":
                fc = "#ffffff"
                ec = "#222222"
                txt_color = "#000000"
            else:
                fc = "#222222"
                ec = "#000000"
                txt_color = "#ffffff"
            ax.add_patch(plt.Circle((cx, cy), 0.30, facecolor=fc,
                                    edgecolor=ec, linewidth=2.0, zorder=5))
            ax.text(cx, cy, str(num), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txt_color, zorder=6)
        # Row / col labels
        for c in range(n):
            ax.text(c + 0.5, n + 0.05, str(c + 1), ha="center", va="bottom",
                    fontsize=10, color="#444444")
        for r in range(n):
            ax.text(-0.15, n - 1 - r + 0.5, str(r + 1), ha="right", va="center",
                    fontsize=10, color="#444444")
        ax.text(n / 2.0, -0.55,
                "row 1..N (top->bottom), col 1..N (left->right)",
                ha="center", va="center", fontsize=9, color="#666666",
                style="italic")
        ax.set_xlim(-0.5, n + 0.2)
        ax.set_ylim(-0.8, n + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Validate the predicted edge list represents a single closed loop
        satisfying every circle's constraint."""
        if not hasattr(self, "_sg_grid"):
            # Fast-path guard: compute_score's fast-path skips generate(),
            # so _sg_grid isn't set. Raising AttributeError triggers the
            # slow-path fallback in reward_function.py (regenerate + retry).
            raise AttributeError("shingoki state not initialized; need slow-path generate")
        # PROBE MODE: yes/no
        if getattr(self, "_probe_mode", False):
            text = predicted.strip().lower()
            m = re.search(r"\b(yes|no|y|n)\b", text)
            if m:
                w = m.group(1)
                pred = "YES" if w.startswith("y") else "NO"
                return pred == self._probe_answer
            return False
        # Parse edges of form (r,c)-(r,c)
        pat = re.compile(
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*-\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        matches = pat.findall(predicted)
        if not matches:
            return False
        n = self._sg_n
        edges_set: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        for r1s, c1s, r2s, c2s in matches:
            r1, c1, r2, c2 = int(r1s) - 1, int(c1s) - 1, int(r2s) - 1, int(c2s) - 1
            if not (0 <= r1 < n and 0 <= r2 < n and 0 <= c1 < n and 0 <= c2 < n):
                return False
            # Adjacent (orthogonal)
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                return False
            edges_set.add(_normalize_edge((r1, c1), (r2, c2)))
        if not edges_set:
            return False
        # Build cell-degree map
        degree: Dict[Tuple[int, int], int] = defaultdict(int)
        adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
        for (a, b) in edges_set:
            degree[a] += 1
            degree[b] += 1
            adj[a].append(b)
            adj[b].append(a)
        # Every cell on the loop has degree 2; cells off the loop have degree 0.
        for c, d in degree.items():
            if d != 2:
                return False
        # Connectivity: the edge set must form a single closed loop.
        loop_cells = list(degree.keys())
        if not loop_cells:
            return False
        seen = {loop_cells[0]}
        stack = [loop_cells[0]]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if seen != set(loop_cells):
            return False
        # Now traverse the loop in order — pick start = first loop cell, and
        # arbitrarily pick one of its two neighbors as the initial "prev" so
        # the walk proceeds the other way.
        start = loop_cells[0]
        # Pick one neighbor as prev (so we walk into the other neighbor).
        if len(adj[start]) != 2:
            return False
        prev = adj[start][0]
        cur = start
        order: List[Tuple[int, int]] = [cur]
        while True:
            nxts = [v for v in adj[cur] if v != prev]
            if len(nxts) != 1:
                return False
            nxt = nxts[0]
            if nxt == start:
                break
            order.append(nxt)
            prev = cur
            cur = nxt
            if len(order) > 4 * n * n:
                return False
        # Length sanity
        if len(order) < 4:
            return False
        # Each circle cell must lie on the loop and satisfy its constraint.
        loop_set = set(order)
        for circ in self._sg_circles:
            cell = (circ["row"], circ["col"])
            if cell not in loop_set:
                return False
        # Compute classification at each loop position.
        # Use the same algorithm as _classify_loop on `order` cyclically.
        cls = _classify_loop(order)
        cell_to_info = {info["cell"]: info for info in cls}
        for circ in self._sg_circles:
            cell = (circ["row"], circ["col"])
            info = cell_to_info[cell]
            if circ["kind"] == "W" and info["is_corner"]:
                return False
            if circ["kind"] == "B" and not info["is_corner"]:
                return False
            if info["sum"] != circ["num"]:
                return False
        return True
