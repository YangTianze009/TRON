"""
Hamiltonian cycle finding: given a small undirected graph, output a cycle
that visits every vertex exactly once and returns to the start, or output
"No" if no such cycle exists.

Difficulty: vertex count grows from 4 to 9. L0 is the trivially solvable
4-vertex complete graph K_4 (any rotation of A,B,C,D,A is a valid cycle).
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
    "Your task is to find a Hamiltonian cycle in the undirected graph below.\n\n"
    "### Game Rules:\n"
    "1. A Hamiltonian cycle visits every vertex exactly once and returns to its starting vertex.\n"
    "2. Output the sequence of vertices with the starting vertex appearing at both ends.\n"
    "3. If no Hamiltonian cycle exists, output the string `No`.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers labeled 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Provide the Hamiltonian cycle as a Python list of vertex IDs (start at both ends) inside <answer>...</answer>.\n"
    "Example: <answer>[0, 1, 5, 4, 2, 3, 0]</answer> or <answer>No</answer>",

    "Find a Hamiltonian cycle in the graph below.\n\n"
    "### Game Rules:\n"
    "- Visit every vertex exactly once and return to start.\n"
    "- Output `No` if no such cycle exists.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output a Python list of integers (start vertex repeated at end), or `No`, inside <answer>...</answer>.",

    "Your task is to find a Hamiltonian cycle in the graph described below.\n\n"
    "### Game Rules:\n"
    "Cycle visiting every vertex once, returning to start; output `No` if impossible.\n\n"
    "### Coordinate System:\n"
    "- Integer labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output the cycle as a Python integer list inside <answer>...</answer>.",
]


def _has_hamiltonian_cycle(n: int, adj: Dict[str, Set[str]],
                           labels: List[str]) -> Optional[Tuple[str, ...]]:
    """Brute-force check: returns one Hamiltonian cycle (tuple of labels with
    start at both ends) or None."""
    if n < 3:
        return None
    start = labels[0]
    for perm in permutations(labels[1:]):
        seq = (start,) + perm
        ok = True
        for i in range(n):
            u, v = seq[i], seq[(i + 1) % n]
            if v not in adj[u]:
                ok = False
                break
        if ok:
            return seq + (start,)
    return None


class HamiltonianCycleFindQA(StandaloneVisualEnv):
    ENV_NAME = "hamiltonian_cycle_find"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "n": 4, "force_k_complete": True}
        # Vertex count grows; allow some chance of "No".
        n = min(4 + level // 2 + (1 if level >= 4 else 0), 9)
        # density: lower density at higher levels makes it harder + more "No" cases.
        density = max(0.45, 0.85 - level * 0.04)
        return {"level": level, "n": n, "density": density,
                "force_k_complete": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1721 + level * 71 + 47)
        n = cfg["n"]
        labels = [str(i) for i in range(n)]  # integer labels (structured-puzzle format)

        if cfg.get("force_k_complete"):
            # K_n with all edges; Hamiltonian cycle trivially exists.
            edges = [(labels[i], labels[j]) for i in range(n)
                     for j in range(i + 1, n)]
            adj: Dict[str, Set[str]] = {l: set() for l in labels}
            for u, v in edges:
                adj[u].add(v); adj[v].add(u)
        else:
            edges, adj = self._sample_graph(rng, labels, cfg["density"], level)

        cycle = _has_hamiltonian_cycle(n, adj, labels)

        # Cache the adj for verify-time check.
        self._ham_adj = {k: set(v) for k, v in adj.items()}
        self._ham_labels = list(labels)
        self._ham_n = n
        self._ham_has_cycle = cycle is not None

        if cycle is None:
            ans_str = "No"
        else:
            # Convert string labels back to ints for output
            ans_str = str([int(c) for c in cycle])

        sidx = (self.seed or 0) % len(_TEMPLATES)
        # Build adjacency dict for prompt
        adj_dict = {i: sorted(int(x) for x in adj[labels[i]]) for i in range(n)}
        question = _TEMPLATES[sidx].format(
            n=n, n_minus_one=n - 1, adj_text=str(adj_dict),
        )
        img = self._render(labels, edges, rng)
        return question, ans_str, img

    def _sample_graph(self, rng: random.Random, labels: List[str],
                      density: float, level: int):
        """Sample a graph with the requested density. Try a few times so the
        problem isn't trivially disconnected."""
        n = len(labels)
        for _ in range(8):
            edges = []
            adj: Dict[str, Set[str]] = {l: set() for l in labels}
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        edges.append((labels[i], labels[j]))
                        adj[labels[i]].add(labels[j])
                        adj[labels[j]].add(labels[i])
            # Avoid trivially disconnected (would force No too easily).
            visited = {labels[0]}
            stack = [labels[0]]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            if len(visited) == n and len(edges) >= n:
                return edges, adj
        # Fallback: ensure connected by adding a path skeleton, then density
        # extras.
        edges = []
        adj = {l: set() for l in labels}
        order = list(labels)
        rng.shuffle(order)
        for i in range(1, n):
            u, v = order[i], order[i - 1]
            edges.append((u, v))
            adj[u].add(v); adj[v].add(u)
        for i in range(n):
            for j in range(i + 1, n):
                if labels[j] in adj[labels[i]]:
                    continue
                if rng.random() < density:
                    edges.append((labels[i], labels[j]))
                    adj[labels[i]].add(labels[j])
                    adj[labels[j]].add(labels[i])
        return edges, adj

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i, lbl in enumerate(labels):
            angle = 2 * math.pi * i / n + rng.uniform(-0.05, 0.05)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[lbl] = (r * math.cos(angle), r * math.sin(angle))
        for u, v in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#566573", linewidth=1.7, zorder=1)
        for lbl in labels:
            x, y = pos[lbl]
            ax.add_patch(plt.Circle((x, y), 0.17, facecolor="#c8e6c9",
                                    edgecolor="#1b5e20", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#1b5e20",
                    zorder=6)
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
        """Validate a Hamiltonian cycle (integer-list with start at both ends)
        OR `No` when no cycle exists. Also accepts cycles without start repeat.
        """
        import ast
        gt = ground_truth.strip().lower()
        pred_clean = predicted.strip()
        is_no_pred = bool(re.match(r"^\s*(no|n)\s*$", pred_clean.lower()))
        if gt == "no":
            return is_no_pred
        if is_no_pred:
            return False

        if not hasattr(self, "_ham_adj"):
            # Fast-path guard: compute_score's fast-path skips generate(),
            # so _ham_adj isn't set. Raising AttributeError triggers the
            # slow-path fallback in reward_function.py (regenerate + retry).
            raise AttributeError("hamiltonian state not initialized; need slow-path generate")
        # Try parsing as Python list of integers
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
        # Accept either with or without start-repeated form
        if len(seq_idx) == n + 1:
            if seq_idx[0] != seq_idx[-1]:
                return False
            seq_full = seq_idx
        elif len(seq_idx) == n:
            # No start-repeat; treat as cycle (close with start)
            seq_full = seq_idx + [seq_idx[0]]
        else:
            return False
        body = seq_full[:-1]
        # Each vertex visited exactly once
        if len(set(body)) != n:
            return False
        if set(body) != set(range(n)):
            return False
        # Check edges
        adj = self._ham_adj
        for i in range(n):
            u, v = str(seq_full[i]), str(seq_full[i + 1])
            if v not in adj.get(u, set()):
                return False
        return True
