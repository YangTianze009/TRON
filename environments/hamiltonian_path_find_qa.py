"""
Hamiltonian path finding (reference H31): given a small undirected graph + a
designated start vertex, find a path visiting every vertex exactly once
starting at that vertex; output `No` if no such path exists.

structured-puzzle format: structured 4-section text+image dual-mode prompt; answer is
a Python list of integer vertex IDs (no repeat at end since it's a path, not
a cycle).

Difficulty: vertex count grows from 4 to 10. L0 is K_4 with start = 0
(trivial: any permutation starting at 0 is a Hamiltonian path).
"""
import math
import random
import re
from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to find a Hamiltonian path in the undirected graph below.\n\n"
    "### Game Rules:\n"
    "1. A Hamiltonian path is a sequence of vertices that starts at the given start vertex, visits every vertex exactly once, and follows graph edges.\n"
    "2. If no such path exists, output the string `No`.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers labeled 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "- start = {start}\n"
    "- adj = {adj_text}\n\n"
    "### Output Format:\n"
    "Provide the Hamiltonian path as a Python list of vertex IDs inside <answer>...</answer>.\n"
    "Example: <answer>[2, 0, 1, 3]</answer> or <answer>No</answer>",

    "Find a Hamiltonian path in the graph below starting from a specified vertex.\n\n"
    "### Game Rules:\n"
    "- Path must start at the given start vertex.\n"
    "- Each vertex visited exactly once.\n"
    "- Output `No` if no such path exists.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "start = {start}\n"
    "adj = {adj_text}\n\n"
    "### Output Format:\n"
    "Output a Python list of integers (or `No`) inside <answer>...</answer>.",

    "Your task is to find a Hamiltonian path described below.\n\n"
    "### Game Rules:\n"
    "Path visiting every vertex exactly once, starting at the designated start vertex.\n\n"
    "### Coordinate System:\n"
    "- Integer labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "start = {start}\n"
    "adj = {adj_text}\n\n"
    "### Output Format:\n"
    "Output the path as a Python integer list inside <answer>...</answer>.",
]


def _has_hamiltonian_path(n: int, adj: Dict[int, Set[int]],
                          start: int) -> Optional[List[int]]:
    """DFS to find any Hamiltonian path starting at `start`. Returns the path
    or None."""
    visited = [False] * n
    visited[start] = True
    path = [start]

    def dfs(u: int) -> bool:
        if len(path) == n:
            return True
        for v in sorted(adj.get(u, ())):
            if not visited[v]:
                visited[v] = True
                path.append(v)
                if dfs(v):
                    return True
                visited[v] = False
                path.pop()
        return False

    if dfs(start):
        return path[:]
    return None


def _sample_graph_with_path(rng: random.Random, n: int, density: float,
                             force_yes: bool, max_attempts: int = 80):
    """Generate (adj, start) such that a Hamiltonian path from start
    either does or doesn't exist (per force_yes)."""
    for _ in range(max_attempts):
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i].add(j)
                    adj[j].add(i)
        # Connectivity check
        visited = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < n:
            continue
        # Pick a start randomly
        start = rng.randint(0, n - 1)
        path = _has_hamiltonian_path(n, adj, start)
        if force_yes:
            if path is not None:
                return adj, start, path
        else:
            if path is None:
                return adj, start, None
    return None


class HamiltonianPathFindQA(StandaloneVisualEnv):
    ENV_NAME = "hamiltonian_path_find"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            # Small but DIVERSE graphs (n=4, randomized edges); no longer K_4 always
            return {"level": level, "n": 4, "density": 0.65, "trivial": False}
        if level <= 2:
            n = 4 + level         # 5..6
        elif level <= 5:
            n = 6 + (level - 2)   # 7..9
        else:
            n = 10
        density = max(0.4, 0.85 - level * 0.04)
        return {"level": level, "n": n, "density": density,
                "trivial": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 2243 + level * 71 + 13)

        if cfg.get("trivial"):
            n = 4
            adj = {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}}  # K_4
            start = 0
            path = _has_hamiltonian_path(n, adj, start)
        else:
            n = cfg["n"]
            yes_case = (rng.random() < 0.6)
            result = _sample_graph_with_path(rng, n, cfg["density"], force_yes=yes_case)
            if result is None:
                # A11 (2026-05-03) bug fix: at moderate densities (~0.73 for L3,
                # n=7), random graphs almost always have a Hamiltonian path,
                # so force_yes=False can fail to find a NO-instance in 80
                # attempts (e.g., L3/seed=100 reproduced the GEN-FAIL).
                # Fallback: flip yes_case and retry — guarantees a generated
                # instance (the YES case is easy to satisfy at this density).
                result = _sample_graph_with_path(
                    rng, n, cfg["density"], force_yes=not yes_case)
            if result is None:
                # Last-resort fallback: lower density to guarantee
                # YES-instance success.
                result = _sample_graph_with_path(
                    rng, n, max(0.5, cfg["density"]), force_yes=True)
            if result is None:
                return None
            adj, start, path = result

        # Cache for verifier
        self._ham_n = n
        self._ham_adj = {k: set(v) for k, v in adj.items()}
        self._ham_start = start

        if path is None:
            ans_str = "No"
        else:
            ans_str = str(path)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        adj_dict = {i: sorted(adj[i]) for i in range(n)}
        question = _TEMPLATES[sidx].format(
            n=n, n_minus_one=n - 1, start=start, adj_text=str(adj_dict),
        )
        labels = [str(i) for i in range(n)]
        img = self._render(labels, adj, start, rng)
        return question, ans_str, img

    def _render(self, labels, adj, start, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        # Edges
        for u in range(n):
            for v in adj.get(u, ()):
                if u < v:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], color="#566573",
                            linewidth=1.7, zorder=1)
        # Vertices (highlight start)
        for i in range(n):
            x, y = pos[i]
            fc = "#ffe0b2" if i == start else "#c8e6c9"
            ec = "#bf360c" if i == start else "#1b5e20"
            ax.add_patch(plt.Circle((x, y), 0.17, facecolor=fc,
                                    edgecolor=ec, linewidth=1.5,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=ec, zorder=6)
        # Mark start vertex with a label
        sx, sy = pos[start]
        ax.text(sx, sy + 0.30, "START", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#bf360c")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Validate any Hamiltonian path starting at the designated vertex."""
        import ast
        gt = ground_truth.strip().lower()
        pred_clean = predicted.strip()
        is_no_pred = bool(re.match(r"^\s*(no|n)\s*$", pred_clean.lower()))
        if gt == "no":
            return is_no_pred
        if is_no_pred:
            return False

        s = re.sub(r"```[^\n]*\n", "", pred_clean).replace("```", "").strip()
        seq_idx = None
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                seq_idx = [int(x) for x in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        if seq_idx is None:
            try:
                seq_idx = [int(x) for x in re.findall(r"-?\d+", s)]
            except ValueError:
                return False
        if not seq_idx:
            return False
        n = self._ham_n
        if len(seq_idx) != n:
            return False
        if seq_idx[0] != self._ham_start:
            return False
        if len(set(seq_idx)) != n:
            return False
        if set(seq_idx) != set(range(n)):
            return False
        # Check edges
        for i in range(n - 1):
            u, v = seq_idx[i], seq_idx[i + 1]
            if v not in self._ham_adj.get(u, set()):
                return False
        return True
