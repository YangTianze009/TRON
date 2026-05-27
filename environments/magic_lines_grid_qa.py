"""
Magic figures: place 1..N integers (or a specified subset) at the marked
positions of a triangle / hexagon / star figure so that every line of
positions sums to a fixed target.

L0 = magic triangle, vertices A B C with edge midpoints D E F (6 positions
total) — but for L0 we use a simplified TRIVIAL 3-position triangle puzzle.
Higher levels use 6-position magic triangles, 7-pointed stars, hexagrams.

Answer format: comma-separated `id:value` pairs (e.g., `A:1, B:5, C:3, ...`).
Multiple valid solutions are accepted via override `_check_answer`.
"""
import math
import random
from io import BytesIO
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Place the integers from 1 to {N} (each used exactly once) into the labelled circles {labels} of the figure shown so that the {n_lines} lines all sum to {target}. Output your assignment as comma-separated `id:value` pairs in <answer>...</answer>.",
    "Magic puzzle: assign integers 1 to {N} to the positions {labels} (each integer used once). Each line of the figure must sum to {target}. Place the assignment as `A:1, B:2, ...` style pairs in <answer>...</answer>.",
    "Fill the labelled positions {labels} with the integers 1..{N} (no repeats) so every line of the figure sums to {target}. Comma-separated `id:value` pairs in <answer>...</answer>.",
    "The figure has {N} circles {labels}. Place the integers 1..{N} (each once) so that each line sums to {target}. Output as comma-separated `id:value` pairs in <answer>...</answer>.",
    "Place 1..{N} into the {N} circles such that every straight line in the figure sums to {target}. Use each number exactly once. Format your answer as `A:val, B:val, ...` in <answer>...</answer>.",
    "Solve the magic puzzle. Assign 1..{N} to the {N} labelled positions; every line must sum to {target}. Output `id:value` pairs (comma-separated) in <answer>...</answer>.",
    "Each labelled circle gets a unique integer from 1 to {N}. Lines must each sum to {target}. Provide `id:value` pairs in <answer>...</answer>.",
    "Magic figure: 1..{N} into {N} circles, lines sum to {target}. Output as `A:value, B:value, ...` in <answer>...</answer>.",
    "Assign the numbers 1 through {N} to the circles {labels} so that the sum along each line equals {target}. Place the comma-separated `id:value` pairs in <answer>...</answer>.",
    "Distribute 1..{N} (each once) over the labelled circles. The sum along every line of the figure must equal {target}. Output `id:val` pairs in <answer>...</answer>.",
    "Solve: place the integers 1..{N} at positions {labels}, with every figure line summing to {target}. Comma-separated `id:value` pairs in <answer>...</answer>.",
    "Magic-figure puzzle ({N} positions). Use each integer 1..{N} exactly once; lines sum to {target}. Format `A:val, B:val,...` in <answer>...</answer>.",
    "Find an assignment of 1..{N} to {labels} such that all {n_lines} lines sum to {target}. Use comma-separated `id:value` pairs. Place answer in <answer>...</answer>.",
    "Fill the figure with 1..{N} (one per circle) so that every line of the figure totals {target}. Output `id:value` pairs in <answer>...</answer>.",
    "Magic puzzle: assign {N} distinct integers from 1..{N} to the labelled circles. Each line sums to {target}. Format `A:val, B:val, ...` in <answer>...</answer>.",
    "Place integers 1..{N} so that the {n_lines} lines of the figure shown sum to {target}. Comma-separated `id:value` pairs in <answer>...</answer>.",
]


def _solve_magic(positions: List[str], lines: List[List[str]], target: int,
                 N: int, max_attempts: int = 100000) -> Optional[Dict[str, int]]:
    """Find an assignment 1..N to `positions` such that every line sums to target.
    Brute-force permutations with line-pruning. Returns None if not solvable in
    budget."""
    pos_idx = {p: i for i, p in enumerate(positions)}
    line_idxs = [[pos_idx[p] for p in line] for line in lines]
    n_pos = len(positions)
    if n_pos != N:
        return None
    nums = list(range(1, N + 1))
    count = [0]

    # Order positions by frequency in lines (most-constrained first)
    freq = [0] * n_pos
    for li in line_idxs:
        for i in li:
            freq[i] += 1
    order = sorted(range(n_pos), key=lambda i: -freq[i])
    inv_order = [0] * n_pos
    for k, i in enumerate(order):
        inv_order[i] = k

    # Re-index lines so that we can incrementally check sums
    # Compute, for each line, the position in `order` of its highest-indexed cell
    line_finalized_at = []
    for li in line_idxs:
        line_finalized_at.append(max(inv_order[c] for c in li))

    # When a line is finalized at step k, we need its sum to equal target.
    # Group lines by their finalize-step.
    lines_at_step = [[] for _ in range(n_pos)]
    for li_i, li in enumerate(line_idxs):
        lines_at_step[line_finalized_at[li_i]].append(li_i)

    assignment = [0] * n_pos
    used = [False] * (N + 1)

    def backtrack(step):
        count[0] += 1
        if count[0] > max_attempts:
            return False
        if step == n_pos:
            return True
        pos_i = order[step]
        # Determine plausible values: try shuffled candidates? deterministic is fine
        for v in nums:
            if used[v]:
                continue
            assignment[pos_i] = v
            used[v] = True
            # Check finalized lines
            ok = True
            for li_i in lines_at_step[step]:
                s = sum(assignment[c] for c in line_idxs[li_i])
                if s != target:
                    ok = False
                    break
            if ok and backtrack(step + 1):
                return True
            used[v] = False
            assignment[pos_i] = 0
        return False

    if backtrack(0):
        return {positions[i]: assignment[i] for i in range(n_pos)}
    return None


# -------------------------------------------------------- figure definitions
def _triangle_3() -> Tuple[List[str], List[List[str]], List[Tuple[float, float]]]:
    """L0 trivial: 3 vertices A,B,C; one 'line' is the perimeter? Actually for
    a true magic puzzle we need a meaningful constraint. Use: 3 vertices,
    1..3 placed; constraint: A+B+C must equal 6 (which is automatic since
    1+2+3=6). To make it non-trivial we add: A < B < C OR a specific given.

    We make this trivial by REVEALING two of three values; the model just
    fills the third. So lines = [[A,B,C]], target=6, 2 of 3 are GIVEN.

    But the framework here doesn't have GIVEN cells — every cell is variable.
    So for L0 we use a different shape: 3 circles A,B,C and target=6 (any
    perm works). _check_answer only checks the constraint.
    """
    pts = [(-1.0, -0.6), (1.0, -0.6), (0.0, 1.0)]
    return ["A", "B", "C"], [["A", "B", "C"]], pts


def _magic_triangle_6() -> Tuple[List[str], List[List[str]], List[Tuple[float, float]]]:
    """6 positions: 3 vertices (A,B,C) + 3 midpoints (D,E,F).
    Lines: 3 sides each containing 3 cells (vertex-mid-vertex).
    Standard magic triangle: 1..6, each side sums to 9, 10, 11, or 12.
    """
    # A = top vertex, B = bottom-left, C = bottom-right
    # D = mid AB (left side), E = mid BC (bottom), F = mid AC (right side)
    pts = [
        (0.0, 1.5),     # A
        (-1.5, -1.0),   # B
        (1.5, -1.0),    # C
        (-0.75, 0.25),  # D (mid AB)
        (0.0, -1.0),    # E (mid BC)
        (0.75, 0.25),   # F (mid AC)
    ]
    labels = ["A", "B", "C", "D", "E", "F"]
    lines = [
        ["A", "D", "B"],   # left side
        ["B", "E", "C"],   # bottom side
        ["A", "F", "C"],   # right side
    ]
    return labels, lines, pts


def _hexagram_12() -> Tuple[List[str], List[List[str]], List[Tuple[float, float]]]:
    """6-pointed star (Star of David): 12 outer/inner points (A..L) on the
    perimeter. Each of 6 lines (a row of the star) contains 4 points and
    must sum to 26 with 1..12.

    Layout: triangle 1 vertices: A B C; triangle 2 vertices: G I K (rotated 60°).
    Intersections: D between A-G's line, etc. We use a simpler 6-line
    structure based on classic hexagram puzzle.

    Convention from MME idx=416:
      "Place 1..12 in A..L of 6-pointed star such that each of 6 rows
       (e.g., A-D-G-K) sums to 26."

    We define explicit 6 lines of 4 points each.
    """
    # Place 12 points on a star: 6 "outer points" + 6 "inner points"
    # We define them on a circle for outer, and a smaller circle for inner.
    import math
    R_out = 2.0
    R_in = 1.0
    pts = []
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    # Alternate outer-inner around the perimeter starting at top
    for i in range(12):
        ang = math.pi / 2 - i * (math.pi / 6)  # step of 30 deg, clockwise
        r = R_out if i % 2 == 0 else R_in
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    # 6 lines of 4 points crossing the star
    # Each line connects an outer point to the opposite outer point through
    # 2 inner points.
    # Outer indices: 0, 2, 4, 6, 8, 10
    # The lines are: 0-3-9-6, 0-7-5-... actually we need a verified set.
    # Use the known classic mapping: each "row" passes through 2 outer & 2 inner.
    # We'll use 6 lines of 4 cells defined by the structure of the hexagram:
    lines = [
        ["A", "D", "G", "J"],  # 0,3,6,9 - through center
        ["B", "C", "H", "I"],  # 1,2,7,8 - one of three rows
        ["B", "E", "F", "K"],  # 1,4,5,10
        ["L", "C", "F", "I"],  # 11,2,5,8 - alt row
        ["L", "E", "H", "K"],  # 11,4,7,10
        ["A", "K", "F", "G"],  # 0,10,5,6 — won't sum cleanly
    ]
    # NOTE: arbitrary line picks may not have any 1..12 assignment summing to 26.
    # We'll instead define a smaller, well-known magic figure.
    return labels, lines, pts


def _hexagon_6() -> Tuple[List[str], List[List[str]], List[Tuple[float, float]]]:
    """Hexagon 6 vertices A..F; 3 long diagonals A-D, B-E, C-F (each a "line"
    of 2 cells). For 1..6 with each diagonal pair summing to target=7
    (since 1+2+3+4+5+6 = 21, three pairs sum to 21 → each pair = 7),
    valid pairs: (1,6),(2,5),(3,4).
    """
    import math
    pts = []
    labels = ["A", "B", "C", "D", "E", "F"]
    for i in range(6):
        ang = math.pi / 2 - i * (math.pi / 3)
        pts.append((1.5 * math.cos(ang), 1.5 * math.sin(ang)))
    lines = [
        ["A", "D"], ["B", "E"], ["C", "F"],
    ]
    return labels, lines, pts


def _pentagram_5() -> Tuple[List[str], List[List[str]], List[Tuple[float, float]]]:
    """5 vertices A..E placed at pentagon corners; the lines are the 5 edges
    of the pentagon. With 1..5 each edge of length 2 should sum to a target.
    Sum of all = 15; each cell appears in 2 edges; so 5*target = 2*15 = 30 →
    target = 6. Valid pairs summing to 6: (1,5), (2,4), (3,3 = invalid).
    Need a Hamiltonian cycle on K5 with all pair-sums equal 6 — only 3 pairs
    sum to 6 (1+5, 2+4) but we need 5 edges. Impossible. Skip pentagram.
    """
    # Instead, use a simple line of 4 cells with target.
    # We'll skip pentagram; use only triangles, hexagons, and a custom 7-pos.
    pass


class MagicLinesGridQA(StandaloneVisualEnv):
    ENV_NAME = "magic_lines_grid"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # NOTE: triangle3 is intentionally NOT used at L0. With 1..3 placed
        # into A,B,C and the only line being A+B+C, the constraint
        # 1+2+3=6 holds for every permutation — vacuously trivial. Models
        # interpret the prompt as ill-posed and ramble. Use the hexagon
        # diagonals (1..6, three pairs each summing to 7) as a real but
        # easy puzzle at L0.
        if level == 0:
            shapes = ["hexagon6"]
        elif level <= 2:
            shapes = ["hexagon6"]
        elif level <= 4:
            shapes = ["hexagon6", "magic_tri6"]
        elif level <= 6:
            shapes = ["magic_tri6", "magic_tri6"]
        else:
            shapes = ["magic_tri6"]
        return {"level": level, "shapes": shapes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1789 + level * 71 + 23)
        shape = rng.choice(cfg["shapes"])

        if shape == "triangle3":
            labels, lines, pts = _triangle_3()
            N = 3
            # Sum of 1..3 = 6, the only line constraint is A+B+C=6 which always holds.
            target = 6
            # Get any solution as reference (used only by check_answer for format demo)
            sol = {"A": 1, "B": 2, "C": 3}
        elif shape == "hexagon6":
            labels, lines, pts = _hexagon_6()
            N = 6
            target = 7
            sol = _solve_magic(labels, lines, target, N)
            if sol is None:
                return None
        else:  # magic_tri6
            labels, lines, pts = _magic_triangle_6()
            N = 6
            # For the magic triangle, valid targets are 9, 10, 11, 12.
            # Pick one that has solutions.
            target = rng.choice([9, 10, 11, 12])
            sol = _solve_magic(labels, lines, target, N)
            if sol is None:
                return None

        # Save constraint info for _check_answer
        self._magic_labels = labels
        self._magic_lines = lines
        self._magic_target = target
        self._magic_N = N

        sidx = (self.seed or 0) % len(_TEMPLATES)
        labels_str = ",".join(labels)
        question = _TEMPLATES[sidx].format(
            N=N, labels=labels_str, target=target, n_lines=len(lines))
        # Format reference answer as id:value pairs in label order
        ans_str = ", ".join(f"{lab}:{sol[lab]}" for lab in labels)
        img = self._render_figure(labels, lines, pts, target, N)
        return question, ans_str, img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Override: parse predicted as id:value pairs; verify (a) it's a
        permutation of 1..N over `_magic_labels`, (b) every line sums to
        `_magic_target`."""
        if not hasattr(self, "_magic_labels"):
            return super()._check_answer(predicted, ground_truth)

        import re
        # Normalize: remove curly braces, single quotes, etc.
        s = predicted.replace("{", " ").replace("}", " ")
        s = s.replace("'", " ").replace('"', " ")
        # Match patterns like A:1 or A: 1 or A=1 or A -> 1
        pattern = re.compile(r"\b([A-Za-z])\s*[:=\-]+\s*(\d+)")
        matches = pattern.findall(s)
        if not matches:
            return False
        assignment = {}
        for k, v in matches:
            kk = k.upper()
            if kk in assignment:
                # Duplicate keys → keep first
                continue
            try:
                assignment[kk] = int(v)
            except ValueError:
                return False
        # Must cover every label exactly
        labels = self._magic_labels
        N = self._magic_N
        if set(assignment.keys()) != set(labels):
            return False
        # Values must be permutation of 1..N
        if sorted(assignment.values()) != list(range(1, N + 1)):
            return False
        # Every line must sum to target
        for line in self._magic_lines:
            s = sum(assignment[lab] for lab in line)
            if s != self._magic_target:
                return False
        return True

    def _render_figure(self, labels, lines, pts, target, N) -> Image.Image:
        rng = self._rng
        bg = "#ffffff"
        fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Map label -> point
        pos = {lab: pts[i] for i, lab in enumerate(labels)}

        # Draw lines
        line_color = "#1d3557"
        for line in lines:
            xs = [pos[c][0] for c in line]
            ys = [pos[c][1] for c in line]
            ax.plot(xs, ys, color=line_color, linewidth=2.4, zorder=1)

        # Draw circles with labels
        node_color = "#fff8e7"
        node_edge = "#1d3557"
        node_text = "#1d3557"
        for lab, (x, y) in pos.items():
            circ = plt.Circle((x, y), 0.32, facecolor=node_color,
                              edgecolor=node_edge, linewidth=2.0, zorder=3)
            ax.add_patch(circ)
            ax.text(x, y, lab, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=node_text, zorder=4)

        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Place 1..{N} so each line sums to {target}",
                     fontsize=12, pad=8)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg, dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
