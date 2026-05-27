"""
Topological sort: given a small DAG drawn in the image, output any valid
topological ordering of the vertices.

Difficulty axes:
  - # vertices (4 → 9)
  - density of edges (sparse → dense)
"""
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to perform a topological sort on the directed acyclic graph (DAG) below.\n\n"
    "### Game Rules:\n"
    "1. A topological ordering is a linear ordering of vertices such that for every directed edge u→v, vertex u appears before v in the ordering.\n"
    "2. Any valid topological ordering is accepted.\n\n"
    "### Coordinate System:\n"
    "- Vertices are labeled 0..{n_minus_one} as integers.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Adjacency list (out-edges per vertex): {adj_text}\n\n"
    "### Output Format:\n"
    "Provide one valid topological ordering as a Python list of vertex IDs inside <answer>...</answer>.\n"
    "Example: <answer>[4, 5, 2, 0, 1, 3]</answer>",

    "Find a topological ordering of the DAG below.\n\n"
    "### Game Rules:\n"
    "- For every directed edge u→v, u must appear before v in the ordering.\n"
    "- Output any valid ordering.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output a Python list of integers (vertex IDs in topological order) inside <answer>...</answer>.",

    "Your task is to topologically sort the DAG described below.\n\n"
    "### Game Rules:\n"
    "Any linear ordering of the {n} vertices that respects all directed edges (u before v whenever u→v exists) is correct.\n\n"
    "### Coordinate System:\n"
    "- Vertices: integers 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Out-edges per vertex: {adj_text}\n\n"
    "### Output Format:\n"
    "Output the ordering as a Python list of integers inside <answer>...</answer>.",
]


def _kahn_topo(n: int, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    indeg = [0] * n
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return out if len(out) == n else None


def _all_topo_orders(n: int, edges: List[Tuple[int, int]], cap: int = 1024):
    """Return up to `cap` valid topological orderings (used to verify)."""
    indeg = [0] * n
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    res = []
    cur = []

    def dfs(indeg_local):
        if len(res) >= cap:
            return
        if len(cur) == n:
            res.append(tuple(cur))
            return
        # nodes with indeg 0
        for u in range(n):
            if indeg_local[u] == 0 and u not in cur:
                cur.append(u)
                new_indeg = list(indeg_local)
                for v in adj[u]:
                    new_indeg[v] -= 1
                new_indeg[u] = -1
                dfs(new_indeg)
                cur.pop()
                if len(res) >= cap:
                    return

    dfs(list(indeg))
    return res


class TopologicalSortQA(StandaloneVisualEnv):
    ENV_NAME = "topological_sort"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n = 4 + level // 2  # 4 → 9
        density = 0.3 + 0.05 * level
        return {"n": n, "density": min(0.7, density), "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 1013 + level * 67 + 13)

        # Build random DAG: pick a random permutation as topo order, then add
        # forward edges only.
        perm = list(range(n))
        rng.shuffle(perm)
        order_pos = {v: i for i, v in enumerate(perm)}
        edges = []
        for u in range(n):
            for v in range(n):
                if order_pos[u] < order_pos[v]:
                    if rng.random() < cfg["density"]:
                        edges.append((u, v))
        if not edges:
            # at least one edge
            edges.append((perm[0], perm[1]))

        sol = _kahn_topo(n, edges)
        if sol is None:
            return None
        # reference uses integer labels
        labels = [str(i) for i in range(n)]
        ans_str = str(list(sol))
        # All valid orderings (for checking) — store as tuples of integers
        self._all_orders = set(
            tuple(perm) for perm in _all_topo_orders(n, edges)
        )
        self._n = n

        sidx = (self.seed or 0) % len(_TEMPLATES)
        # Build adjacency dict for prompt
        adj_dict = {i: [] for i in range(n)}
        for u, v in edges:
            adj_dict[u].append(v)
        for k in adj_dict:
            adj_dict[k].sort()
        adj_text = str(adj_dict)
        question = _TEMPLATES[sidx].format(
            n=n, n_minus_one=n - 1, adj_text=adj_text,
        )
        img = self._render(n, edges, labels, rng)
        return question, ans_str, img

    def _render(self, n, edges, labels, rng):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Place vertices roughly on a circle
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n + rng.uniform(-0.05, 0.05)
            r = 1.0 + rng.uniform(-0.1, 0.1)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        # Draw edges as arrows
        for u, v in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if d == 0:
                continue
            ux, uy = dx / d, dy / d
            sx, sy = x1 + ux * 0.13, y1 + uy * 0.13
            ex, ey = x2 - ux * 0.16, y2 - uy * 0.16
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color="#555",
                                        lw=1.4))
        # Draw vertices
        for i, (x, y) in pos.items():
            ax.add_patch(plt.Circle((x, y), 0.13, facecolor="#b8d4f0",
                                    edgecolor="#1a3a6e", linewidth=1.5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#1a3a6e")

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Override: any valid topological ordering counts as correct.

        Accepts integer-list answers (structured-puzzle format) as Python list literal,
        comma-separated, or space-separated integers.
        """
        import re, ast
        if not hasattr(self, "_all_orders"):
            # Fast-path guard: compute_score's fast-path skips generate(),
            # so _all_orders isn't set. Raising AttributeError triggers the
            # slow-path fallback in reward_function.py (regenerate + retry).
            raise AttributeError("topological_sort state not initialized; need slow-path generate")
        if not self._all_orders:
            return False
        s = predicted.strip()
        s = re.sub(r"```[^\n]*\n", "", s).replace("```", "").strip()
        # Try Python literal first
        ints = None
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                ints = [int(x) for x in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        if ints is None:
            try:
                ints = [int(x) for x in re.findall(r"-?\d+", s)]
            except ValueError:
                return False
        if not ints or len(ints) != self._n:
            return False
        return tuple(ints) in self._all_orders
