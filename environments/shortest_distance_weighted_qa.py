"""
Shortest Distance (Weighted, Undirected) — structured-puzzle style (graph
problems H34).

Rule: Compute the shortest distance from node 0 to node N-1 in a
weighted UNDIRECTED graph.

Studied reference qids (design notes lines 772-783):
  - (D=1) 6 nodes; from 0 to 5 -> 5
  - (D=1) 6 nodes; from 0 to 5 -> 10
  - (D=3) 8 nodes
  - (D=5) 10+ nodes

Answer format: number (integer or short decimal).

Note: This is DIFFERENT from `shortest_path_directed_weighted_qa.py`
which already exists. That env: directed, asks for integer; this env:
undirected (every edge usable in both directions), asks for integer.
We deliberately keep them as separate envs since reference evaluates
them separately and we want training signal on both.
"""
import heapq
import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the shortest distance in the weighted undirected graph below.\n\n"
    "### Game Rules:\n"
    "1. The graph is undirected; each edge has a positive integer weight.\n"
    "2. Compute the minimum total weight of any path from node 0 to node {m}.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers labeled 0..{m}.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Edges (each as `(u, v, w)`):\n{edges_text}\n\n"
    "### Output Format:\n"
    "Output the shortest distance (integer or short decimal) inside <answer>...</answer>.\n"
    "Example: <answer>10</answer>",

    "Compute the shortest distance from 0 to {m} in the weighted undirected graph below.\n\n"
    "### Game Rules:\n"
    "- Edges are undirected; weights are positive integers.\n"
    "- Output the minimum sum of weights along any path from 0 to {m}.\n\n"
    "### Coordinate System:\n"
    "- Vertices: integers 0..{m}.\n\n"
    "### Current Puzzle State:\n"
    "Edges (u, v, w):\n{edges_text}\n\n"
    "### Output Format:\n"
    "Output the integer distance inside <answer>...</answer>.",

    "Your task is to find the shortest path length in the graph described below.\n\n"
    "### Game Rules:\n"
    "Standard Dijkstra: shortest distance 0 -> {m}.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{m}.\n\n"
    "### Current Puzzle State:\n"
    "Edges (u, v, w):\n{edges_text}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _dijkstra_undirected(n: int, edges: List[Tuple[int, int, float]],
                         src: int, dst: int) -> Optional[float]:
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = [math.inf] * n
    dist[src] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == dst:
            return d
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return None if dist[dst] == math.inf else dist[dst]


def _gen_undirected_graph(n: int, density: float, wmax: int,
                          rng: random.Random,
                          max_attempts: int = 30) -> Optional[List[Tuple[int, int, float]]]:
    for _ in range(max_attempts):
        edges: List[Tuple[int, int, float]] = []
        # Spanning chain to ensure connectivity from 0 to n-1
        perm = list(range(1, n))
        rng.shuffle(perm)
        chain = [0] + perm
        if chain[-1] != n - 1:
            # Force the chain to reach n-1 at the end
            chain.remove(n - 1)
            chain.append(n - 1)
        for u, v in zip(chain, chain[1:]):
            w = rng.randint(1, wmax)
            edges.append((min(u, v), max(u, v), float(w)))
        # Add extra edges by density
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        existing = set((u, v) for (u, v, _) in edges)
        for (u, v) in all_pairs:
            if (u, v) in existing:
                continue
            if rng.random() < density:
                w = rng.randint(1, wmax)
                edges.append((u, v, float(w)))
        d = _dijkstra_undirected(n, edges, 0, n - 1)
        if d is None or math.isinf(d):
            continue
        return edges
    return None


class ShortestDistanceWeightedQA(StandaloneVisualEnv):
    ENV_NAME = "shortest_distance_weighted"
    shape_strategy = "min_over_max"
    shape_beta = 2.0

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n = 4
            density = 0.4
            wmax = 5
        elif level <= 2:
            n = 5
            density = 0.4
            wmax = 7
        elif level <= 4:
            n = 6
            density = 0.4
            wmax = 9
        elif level <= 6:
            n = 7
            density = 0.45
            wmax = 11
        elif level <= 8:
            n = 8
            density = 0.45
            wmax = 12
        else:
            n = 9
            density = 0.5
            wmax = 14
        return {"level": level, "n": n, "density": density, "wmax": wmax}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 1879 + level * 79 + 29)

        edges = _gen_undirected_graph(n, cfg["density"], cfg["wmax"], rng)
        if edges is None:
            return None
        ans = _dijkstra_undirected(n, edges, 0, n - 1)
        if ans is None or math.isinf(ans):
            return None
        ans_value = int(ans) if float(ans).is_integer() else round(ans, 2)
        ans_str = str(int(ans_value)) if float(ans_value).is_integer() \
            else f"{ans_value:.2f}"

        edges_text = "\n".join(f"({u}, {v}, {int(w)})" for (u, v, w) in edges)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            n=n, m=n - 1, edges_text=edges_text,
        )
        img = self._render(n, edges, rng)
        return question, ans_str, img

    def _render(self, n, edges, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Circular layout (with small jitter)
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2 \
                + rng.uniform(-0.03, 0.03)
            r = 1.2
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        # Edges
        for (u, v, w) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#566573", linewidth=1.5,
                    zorder=2)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            wstr = str(int(w)) if float(w).is_integer() else f"{w:.1f}"
            ax.text(mx, my, wstr, ha="center", va="center",
                    fontsize=11, color="#1a3a6e",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="#fdf6e3",
                              edgecolor="#8b8000", linewidth=0.8),
                    zorder=4)
        # Vertices
        for i in range(n):
            x, y = pos[i]
            if i == 0:
                face, edge_c, txt_c = "#a8e6a3", "#1f6f1f", "#1f6f1f"
            elif i == n - 1:
                face, edge_c, txt_c = "#f5b6b6", "#7a1c1c", "#7a1c1c"
            else:
                face, edge_c, txt_c = "#b8d4f0", "#1a3a6e", "#1a3a6e"
            ax.add_patch(plt.Circle((x, y), 0.18, facecolor=face,
                                    edgecolor=edge_c, linewidth=1.5,
                                    zorder=5))
            ax.text(x, y, str(i), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_c,
                    zorder=6)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Defer to base numeric matching.
        return super()._check_answer(predicted, ground_truth)
