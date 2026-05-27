"""
Maximum number of edges that can be removed from a connected undirected graph
while keeping it connected. For a connected graph this equals E - (V - 1),
i.e. the number of edges beyond a spanning tree.
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
    "The image shows a connected undirected graph. What is the maximum number of edges you can remove while keeping the graph connected? Integer in <answer>...</answer>.",
    "Compute the largest number of edges removable from the depicted graph such that the remainder stays connected. Integer in <answer>...</answer>.",
    "From the connected graph shown, output the max number of edges you can delete and still leave it connected. Integer in <answer>...</answer>.",
    "How many edges of the graph in the image can be removed while preserving connectivity? Maximum value, integer in <answer>...</answer>.",
    "The graph shown is connected. Find the maximum number of edges you can drop and keep it connected. Integer in <answer>...</answer>.",
    "Output the count of redundant edges in the connected graph (edges removable while keeping it connected). Integer in <answer>...</answer>.",
    "From the depicted connected graph, find max k such that some k edges may be removed leaving a connected subgraph. Integer in <answer>...</answer>.",
    "Given the connected graph in the image, compute the largest set of edges that can be deleted with the remainder still connected; output its size. Integer in <answer>...</answer>.",
    "Find the maximum number of edges removable while keeping the depicted graph connected. Integer in <answer>...</answer>.",
    "Output |E| minus (|V| - 1) for the connected graph shown. Integer in <answer>...</answer>.",
    "How many extra (cycle) edges does the connected graph in the image have beyond a spanning tree? Integer in <answer>...</answer>.",
    "Compute the number of edges that may be removed from the graph (which is connected) without disconnecting it. Integer in <answer>...</answer>.",
    "From the connected graph shown, find the maximal count of removable edges that preserves connectivity. Integer in <answer>...</answer>.",
    "Maximum removable-edge count keeping the graph connected. Integer in <answer>...</answer>.",
    "Given the connected graph in the image, output the answer to: 'how many edges can I remove and still have the graph connected?' Integer in <answer>...</answer>.",
    "Output the cyclomatic complexity (E - V + 1) of the connected graph shown. Integer in <answer>...</answer>.",
]


class EdgesRemovableCountQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "edges_removable_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "n": 3, "extra": 0}
        # Larger graphs and more extra edges with higher levels.
        if level <= 2:
            return {"level": level, "n": 4 + level - 1, "extra_lo": 1, "extra_hi": 2}
        if level <= 5:
            return {"level": level, "n": 5 + (level - 3), "extra_lo": 1, "extra_hi": 3}
        if level <= 7:
            return {"level": level, "n": 7, "extra_lo": 2, "extra_hi": 4}
        return {"level": level, "n": 8, "extra_lo": 3, "extra_hi": 5}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1471 + level * 59 + 31)
        n = cfg["n"]

        if level == 0:
            # Triangle: 3 vertices, 3 edges, answer = 3 - 2 = 1.
            labels = ["A", "B", "C"]
            edges = [("A", "B"), ("B", "C"), ("A", "C")]
        else:
            labels, edges = self._build_connected(rng, n,
                                                   cfg["extra_lo"], cfg["extra_hi"])

        ans = len(edges) - (n - 1)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(labels, edges, rng)
        return question, str(ans), img

    def _build_connected(
        self, rng: random.Random, n: int, extra_lo: int, extra_hi: int
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        labels = [chr(ord("A") + i) for i in range(n)]
        adj: Dict[str, Set[str]] = {l: set() for l in labels}
        edges: List[Tuple[str, str]] = []

        # Random spanning tree.
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(1, n):
            parent_idx = rng.randint(0, i - 1)
            u = labels[perm[i]]
            v = labels[perm[parent_idx]]
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)

        # Add extra edges (avoid duplicates, avoid self-loops).
        max_extra = (n * (n - 1) // 2) - (n - 1)
        target_extra = min(rng.randint(extra_lo, extra_hi), max_extra)
        attempts = 0
        added = 0
        while added < target_extra and attempts < target_extra * 20 + 30:
            i, j = rng.sample(range(n), 2)
            u, v = labels[i], labels[j]
            if v not in adj[u]:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                added += 1
            attempts += 1
        return labels, edges

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Place vertices on a circle (small jitter for visual variety).
        pos = {}
        for i, lbl in enumerate(labels):
            angle = 2 * math.pi * i / n + rng.uniform(-0.05, 0.05)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[lbl] = (r * math.cos(angle), r * math.sin(angle))
        # Draw edges.
        for u, v in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#7f8c8d", linewidth=1.6, zorder=1)
        # Draw nodes.
        for lbl in labels:
            x, y = pos[lbl]
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor="#b8d4f0",
                                    edgecolor="#1a3a6e", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#1a3a6e",
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
