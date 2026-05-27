"""
Shortest path on a directed weighted graph: given a graph drawn with
arrows + edge weight labels, output the length of the shortest path
from a labelled source S to a labelled target T.

Mirrors qid 136 wording from the source distribution:
  "What is the shortest distance from node S to node G in this directed
   graph?  Ans: 29."

Output: integer (or short decimal). Standard `_check_answer` numeric
tolerance handles small float wobble.
"""
import heapq
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
    "What is the shortest distance from node {src} to node {dst} in the directed weighted graph shown? Output the integer (or short decimal) distance in <answer>...</answer>.",
    "Find the length of the shortest path from {src} to {dst} in the directed graph in the image. Place the numeric distance in <answer>...</answer>.",
    "From the directed weighted graph in the image, compute the shortest-path distance from {src} to {dst}. Answer in <answer>...</answer>.",
    "In the depicted directed graph, what is the shortest distance from node {src} to node {dst}? Output the number in <answer>...</answer>.",
    "Run Dijkstra on the directed weighted graph shown. What is the shortest distance from {src} to {dst}? Answer in <answer>...</answer>.",
    "What is the minimum total edge weight of any path from {src} to {dst} in the directed graph in the image? Place the answer in <answer>...</answer>.",
    "Compute the shortest path distance from {src} to {dst} in the directed weighted graph depicted. Output a single number in <answer>...</answer>.",
    "Output the shortest distance from node {src} to node {dst} in the directed graph (arrows show direction, labels show weights). Answer in <answer>...</answer>.",
    "From source {src} to target {dst}, what is the shortest path length in the depicted directed weighted graph? Place the numeric answer in <answer>...</answer>.",
    "Inspect the directed weighted graph in the image. Find the minimum-cost directed path from {src} to {dst} and output its total weight in <answer>...</answer>.",
    "What is the shortest path distance from {src} to {dst} in the directed graph drawn? Output a number in <answer>...</answer>.",
    "Determine the shortest distance (sum of edge weights) from node {src} to node {dst} in the directed weighted graph in the image. In <answer>...</answer>.",
    "On the directed graph shown, compute the shortest distance from {src} to {dst}. Place the answer in <answer>...</answer>.",
    "Apply Dijkstra's shortest-path algorithm to the directed weighted graph in the image, source {src}, target {dst}. Output the resulting distance in <answer>...</answer>.",
    "What is the cost of the shortest directed path from {src} to {dst} in the depicted graph? Output as a number in <answer>...</answer>.",
    "From the directed weighted graph in the image, output the length of the shortest path from {src} to {dst}. Numeric answer in <answer>...</answer>.",
]


def _dijkstra(n: int, edges: List[Tuple[int, int, float]],
              src: int, dst: int) -> Optional[float]:
    """Standard Dijkstra on a directed weighted graph."""
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
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


class ShortestPathDirectedWeightedQA(StandaloneVisualEnv):
    ENV_NAME = "shortest_path_directed_weighted"

    # Numeric → shape with min/max ratio. Same convention as e.g. integer
    # answer envs in this codebase.
    shape_strategy = "min_over_max"
    shape_beta = 2.0

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial": True, "n": 3}
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
        wmax = 8 + 2 * level
        return {"level": level, "n": n, "density": min(0.7, density),
                "wmax": wmax, "trivial": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1217 + level * 83 + 7)

        if cfg.get("trivial"):
            # 3-vertex direct path S -> M -> T with explicit weights.
            # Direct S -> T edge optionally exists with strictly larger
            # weight so the answer is the sum across S -> M -> T.
            n = 3
            labels = ["S", "M", "T"]
            w1 = rng.randint(2, 6)
            w2 = rng.randint(2, 6)
            edges: List[Tuple[int, int, float]] = [
                (0, 1, w1),
                (1, 2, w2),
            ]
            # ~50% include a longer S -> T direct edge so the picture
            # actually has alternative paths to inspect.
            if rng.random() < 0.5:
                edges.append((0, 2, w1 + w2 + rng.randint(1, 4)))
            src, dst = 0, 2
            ans_value = w1 + w2
        else:
            n = cfg["n"]
            labels = [chr(ord("A") + i) for i in range(n)]
            # Replace first vertex name with S, last with T or G to
            # match the original question style.
            labels[0] = "S"
            labels[-1] = rng.choice(["T", "G"])
            edges, src, dst = self._sample_directed_graph(rng, n, cfg)
            if edges is None:
                return None
            ans_value = _dijkstra(n, edges, src, dst)
            if ans_value is None or math.isinf(ans_value):
                return None

        ans_str = str(int(ans_value)) if float(ans_value).is_integer() \
            else f"{ans_value:.2f}"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(src=labels[src], dst=labels[dst])
        img = self._render(labels, edges, src, dst, rng)
        return question, ans_str, img

    def _sample_directed_graph(
        self, rng: random.Random, n: int, cfg: Dict
    ) -> Tuple[Optional[List[Tuple[int, int, float]]], int, int]:
        wmax = cfg["wmax"]
        density = cfg["density"]
        # Build a random DAG so a path exists. Pick a permutation,
        # only add edges from earlier in the permutation to later.
        for _attempt in range(20):
            perm = list(range(n))
            rng.shuffle(perm)
            order_pos = {v: i for i, v in enumerate(perm)}
            edges: List[Tuple[int, int, float]] = []
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    if order_pos[u] >= order_pos[v]:
                        continue
                    if rng.random() < density:
                        edges.append((u, v, float(rng.randint(1, wmax))))
            # Force at least the spanning chain along perm so a path
            # source-target exists.
            chain_set = set((perm[i], perm[i + 1]) for i in range(n - 1))
            existing = set((u, v) for (u, v, _) in edges)
            for (u, v) in chain_set:
                if (u, v) not in existing:
                    edges.append((u, v, float(rng.randint(1, wmax))))
            # source = perm[0], target = perm[-1]
            src = perm[0]
            dst = perm[-1]
            d = _dijkstra(n, edges, src, dst)
            if d is None:
                continue
            return edges, src, dst
        return None, 0, n - 1

    def _render(self, labels, edges, src, dst, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2 + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        # Edges (directed) — slight curvature for parallel pairs (u→v
        # and v→u) so they don't overlap.
        edge_pair = defaultdict(list)
        for idx, (u, v, w) in enumerate(edges):
            edge_pair[(min(u, v), max(u, v))].append(idx)
        for (u, v, w) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            pair_idx = edge_pair[(min(u, v), max(u, v))]
            has_pair = len(pair_idx) > 1
            rad = 0.0 if not has_pair else (0.18 if u < v else -0.18)
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="#566573",
                                        lw=1.4,
                                        shrinkA=14, shrinkB=14,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=2)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / d * 0.13, dx / d * 0.13
            if rad != 0.0:
                # offset more in the direction of the curve
                ox += -dy / d * (0.15 if rad > 0 else -0.15)
                oy += dx / d * (0.15 if rad > 0 else -0.15)
            wstr = str(int(w)) if float(w).is_integer() else f"{w:.1f}"
            ax.text(mx + ox, my + oy, wstr, ha="center", va="center",
                    fontsize=11, color="#1a3a6e",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="#fdf6e3",
                              edgecolor="#8b8000", linewidth=0.8),
                    zorder=4)
        # Vertices: highlight src and dst
        for i in range(n):
            x, y = pos[i]
            if i == src:
                face, edge_c, txt_c = "#a8e6a3", "#1f6f1f", "#1f6f1f"
            elif i == dst:
                face, edge_c, txt_c = "#f5b6b6", "#7a1c1c", "#7a1c1c"
            else:
                face, edge_c, txt_c = "#b8d4f0", "#1a3a6e", "#1a3a6e"
            ax.add_patch(plt.Circle((x, y), 0.18, facecolor=face,
                                    edgecolor=edge_c, linewidth=1.5,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_c,
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
