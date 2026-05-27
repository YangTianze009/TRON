"""
Chromatic index χ'(G): given a small undirected graph (4–8 vertices),
output the minimum number of colors needed to color the EDGES such
that no two adjacent edges share a color.

Mirrors qid 233 / 234 / 235 wording from the source distribution:
  "Find the chromatic index of the following graph.  Ans: 5 / 2 / 3."

By Vizing's theorem, χ'(G) is either Δ(G) (max degree, "class 1") or
Δ(G)+1 ("class 2"). So we try edge-coloring with Δ colors via a small
backtracking search; if that succeeds, χ'(G) = Δ; otherwise Δ+1.
With ≤8 vertices and ≤28 edges this terminates instantly.

Output: integer.
"""
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Find the chromatic index of the graph in the image (the minimum number of colors needed to color edges so that no two adjacent edges share a color). Output the integer in <answer>...</answer>.",
    "Compute the chromatic index χ'(G) for the depicted graph. The chromatic index is the minimum number of edge colors so adjacent edges differ. Answer in <answer>...</answer>.",
    "What is the chromatic index of the graph in the image? Output the integer answer in <answer>...</answer>.",
    "Determine χ'(G), the edge chromatic number, for the graph shown. Output as an integer in <answer>...</answer>.",
    "Find the minimum number of colors needed to color the edges of the depicted graph so that no two adjacent edges share a color. In <answer>...</answer>.",
    "Output the chromatic index (edge chromatic number) of the graph in the image. Answer in <answer>...</answer>.",
    "What is the smallest number of colors needed for a proper edge-coloring of the graph shown? Place the integer in <answer>...</answer>.",
    "Compute the chromatic index of the graph in the image. The chromatic index of a graph is the minimum number of colors required to color its edges so adjacent edges receive different colors. Answer in <answer>...</answer>.",
    "Find χ'(G) for the graph drawn. Output the integer in <answer>...</answer>.",
    "The chromatic index of a graph is the minimum number of edge colors needed so that adjacent edges (edges sharing a vertex) get different colors. Find this value for the graph in the image. In <answer>...</answer>.",
    "From the graph in the image, output its chromatic index χ'(G) as an integer in <answer>...</answer>.",
    "Determine the chromatic index (edge chromatic number) of the depicted graph. Place the integer in <answer>...</answer>.",
    "How many colors are needed at minimum to color the edges of the graph in the image so that no two edges sharing a vertex have the same color? Output the integer in <answer>...</answer>.",
    "Compute the chromatic index of the graph shown. Output the answer as a single integer in <answer>...</answer>.",
    "What is the chromatic index of the graph drawn in the image? Output an integer in <answer>...</answer>.",
    "Find χ'(G), the minimum number of colors required to properly color the edges of the graph in the image, in <answer>...</answer>.",
]


def _max_degree(n: int, edges: List[Tuple[int, int]]) -> int:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return max(deg) if deg else 0


def _can_edge_color_with(n: int, edges: List[Tuple[int, int]],
                          k: int) -> bool:
    """Try to k-edge-color the graph via backtracking on edge order.
    True iff a proper k-edge-coloring exists. Edges with no incident
    edges (isolated edge) trivially color with 1 color, so handle k>=1.
    """
    m = len(edges)
    if m == 0:
        return k >= 0
    if k <= 0:
        return False
    # Adjacency from vertex to list of edge indices
    inc: Dict[int, List[int]] = defaultdict(list)
    for ei, (u, v) in enumerate(edges):
        inc[u].append(ei)
        inc[v].append(ei)
    # For each edge ei, list of edges it conflicts with (sharing a vertex).
    conflicts = [set() for _ in range(m)]
    for ei, (u, v) in enumerate(edges):
        for ej in inc[u]:
            if ej != ei:
                conflicts[ei].add(ej)
        for ej in inc[v]:
            if ej != ei:
                conflicts[ei].add(ej)

    # Order edges by descending conflict count (most constrained first)
    order = sorted(range(m), key=lambda ei: -len(conflicts[ei]))
    color = [-1] * m

    def backtrack(idx: int) -> bool:
        if idx == m:
            return True
        ei = order[idx]
        used = set()
        for ej in conflicts[ei]:
            if color[ej] >= 0:
                used.add(color[ej])
        for c in range(k):
            if c in used:
                continue
            color[ei] = c
            if backtrack(idx + 1):
                return True
            color[ei] = -1
        return False

    return backtrack(0)


def _chromatic_index(n: int, edges: List[Tuple[int, int]]) -> int:
    """χ'(G) by Vizing: in {Δ, Δ+1}. Search Δ first."""
    if not edges:
        return 0
    delta = _max_degree(n, edges)
    if _can_edge_color_with(n, edges, delta):
        return delta
    return delta + 1


class ChromaticIndexQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "chromatic_index"

    shape_strategy = "min_over_max"
    shape_beta = 2.0

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial": True}
        if level <= 2:
            n = 4
        elif level <= 4:
            n = 5
        elif level <= 6:
            n = 6
        elif level <= 8:
            n = 7
        else:
            n = 8
        density = 0.30 + 0.04 * level
        return {"level": level, "n": n, "density": min(0.7, density),
                "trivial": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1733 + level * 91 + 23)

        if cfg.get("trivial"):
            # Two trivial cases: triangle K_3 (χ'=3) or path P_4 (χ'=2).
            choice = rng.choice(["k3", "p4"])
            if choice == "k3":
                n = 3
                labels = ["A", "B", "C"]
                edges = [(0, 1), (1, 2), (0, 2)]
            else:
                n = 4
                labels = ["A", "B", "C", "D"]
                edges = [(0, 1), (1, 2), (2, 3)]
        else:
            n = cfg["n"]
            labels = [chr(ord("A") + i) for i in range(n)]
            edges = self._sample_graph(rng, n, cfg)

        ans_value = _chromatic_index(n, edges)
        ans_str = str(ans_value)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(labels, edges, rng)
        return question, ans_str, img

    def _sample_graph(
        self, rng: random.Random, n: int, cfg: Dict
    ) -> List[Tuple[int, int]]:
        density = cfg["density"]
        for _ in range(15):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            if len(edges) >= 2:
                return edges
        # Fallback: a star
        return [(0, j) for j in range(1, n)]

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2 + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        for (u, v) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#2c3e50",
                    linewidth=1.8, zorder=1)
        for i in range(n):
            x, y = pos[i]
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor="#dcd0ff",
                                    edgecolor="#3a1a6e", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#3a1a6e",
                    zorder=6)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
