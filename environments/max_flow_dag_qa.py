"""
Max-flow on a small layered DAG. Source S, sink T, integer capacities on
edges. Output the integer max flow value.

Difficulty: layer count and width grow with level. L0 is a 3-node single
edge whose capacity equals the answer (trivially correct).
"""
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the maximum flow in the network below.\n\n"
    "### Game Rules:\n"
    "1. The graph is directed; each edge has an integer capacity (the maximum flow it can carry).\n"
    "2. Compute the value of the maximum flow from the source vertex {source_id} to the sink vertex {sink_id}.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers labeled 0..{n_minus_one}.\n"
    "- Edge `(u, c)` in the adjacency list of v means there is a directed edge v->u with capacity c.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Source: {source_id}; Sink: {sink_id}\n"
    "- Adjacency list with capacities: {adj_text}\n\n"
    "### Output Format:\n"
    "Output the integer max flow value inside <answer>...</answer>.\n"
    "Example: <answer>14</answer>",

    "Compute the max flow in the network below.\n\n"
    "### Game Rules:\n"
    "- Directed graph with edge capacities.\n"
    "- Output the integer max flow from source to sink.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Source: {source_id}, Sink: {sink_id}\n"
    "Edges (with capacities): {adj_text}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",

    "Your task is to compute max flow in the directed capacitated network below.\n\n"
    "### Game Rules:\n"
    "Standard max-flow / min-cut. Output the integer max flow.\n\n"
    "### Coordinate System:\n"
    "- Integer labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Source: {source_id}, Sink: {sink_id}\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output the max flow integer inside <answer>...</answer>.",
]


def _max_flow_edmonds_karp(n: int, capacity: List[List[int]],
                            source: int, sink: int) -> int:
    """Edmonds-Karp BFS max flow on adjacency-matrix capacity. Returns max flow."""
    flow = 0
    cap = [row[:] for row in capacity]
    while True:
        parent = [-1] * n
        parent[source] = source
        q = deque([source])
        while q and parent[sink] == -1:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if parent[sink] == -1:
            break
        # bottleneck
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, cap[u][v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= path_flow
            cap[v][u] += path_flow
            v = u
        flow += int(path_flow)
    return flow


class MaxFlowDAGQA(StandaloneVisualEnv):
    ENV_NAME = "max_flow_dag"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial": True}
        # Layer count and width.
        layers = 2 + level // 3       # 2..5 internal layers between S and T
        width = 1 + (level // 2)      # 1..5 nodes per layer
        cap_max = 4 + level           # capacities up to ~13
        return {"level": level, "trivial": False, "layers": layers,
                "width": width, "cap_max": cap_max}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1399 + level * 53 + 19)

        if cfg.get("trivial"):
            # 3-node line: S -> M -> T. Both capacities equal so answer is
            # immediate. Use small distinct caps so the visible bottleneck
            # answer is unambiguous.
            cap_se = rng.randint(2, 5)
            cap_et = rng.randint(2, 5)
            ans = min(cap_se, cap_et)
            # Build problem
            n = 3
            S, M, T = 0, 1, 2
            capacity = [[0] * n for _ in range(n)]
            capacity[S][M] = cap_se
            capacity[M][T] = cap_et
            labels = ["S", "M", "T"]
            edges = [(S, M, cap_se), (M, T, cap_et)]
            # Layered positions
            pos = {S: (0, 0.0), M: (1, 0.0), T: (2, 0.0)}
        else:
            layers, width, cap_max = cfg["layers"], cfg["width"], cfg["cap_max"]
            # Build a layered DAG: layer 0 = {S}, layers 1..L = inner nodes,
            # layer L+1 = {T}. Edges only go layer i -> layer i+1.
            # Each inner layer has 1..width nodes (random).
            layer_sizes = [1]  # source
            for _ in range(layers):
                layer_sizes.append(rng.randint(max(1, width - 1), width))
            layer_sizes.append(1)  # sink
            # Assign global indices
            node_layer = []
            layer_nodes: List[List[int]] = []
            idx = 0
            for sz in layer_sizes:
                lst = []
                for _ in range(sz):
                    node_layer.append(len(layer_nodes))
                    lst.append(idx); idx += 1
                layer_nodes.append(lst)
            n = idx
            S = layer_nodes[0][0]
            T = layer_nodes[-1][0]
            # Add edges between consecutive layers; each pair with prob ~0.6,
            # plus ensure each non-source/sink node has at least one in and
            # one out edge.
            capacity = [[0] * n for _ in range(n)]
            edges: List[Tuple[int, int, int]] = []
            for li in range(len(layer_nodes) - 1):
                A = layer_nodes[li]
                B = layer_nodes[li + 1]
                for u in A:
                    for v in B:
                        if rng.random() < 0.65:
                            c = rng.randint(1, cap_max)
                            capacity[u][v] = c
                            edges.append((u, v, c))
                # ensure connectivity: every node in layer li has >=1 outgoing
                for u in A:
                    if all(capacity[u][v] == 0 for v in B):
                        v = rng.choice(B)
                        c = rng.randint(1, cap_max)
                        capacity[u][v] = c
                        edges.append((u, v, c))
                # every node in layer li+1 has >=1 incoming
                for v in B:
                    if all(capacity[u][v] == 0 for u in A):
                        u = rng.choice(A)
                        c = rng.randint(1, cap_max)
                        capacity[u][v] = c
                        edges.append((u, v, c))
            # Labels: S, T, then A1, A2 ... for inner nodes.
            labels = [None] * n
            labels[S] = "S"
            labels[T] = "T"
            inner_idx = 0
            inner_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for li in range(1, len(layer_nodes) - 1):
                for u in layer_nodes[li]:
                    labels[u] = inner_letters[inner_idx]
                    inner_idx += 1
                    if inner_idx >= len(inner_letters):
                        return None
            ans = _max_flow_edmonds_karp(n, capacity, S, T)
            # Avoid degenerate ans=0 (can happen if some layer has no
            # path-through). Just retry by returning None.
            if ans <= 0:
                return None
            # Positions for rendering
            pos: Dict[int, Tuple[float, float]] = {}
            for li, lst in enumerate(layer_nodes):
                # vertical centering
                m = len(lst)
                for k, u in enumerate(lst):
                    y = (k - (m - 1) / 2.0) * 1.2
                    pos[u] = (li * 1.6, y)

        # Pick template
        sidx = (self.seed or 0) % len(_TEMPLATES)
        # Build adjacency dict for prompt: {u: [(v, cap), ...]}
        adj_dict = {i: [] for i in range(n)}
        for u, v, c in edges:
            adj_dict[u].append((v, c))
        for k in adj_dict:
            adj_dict[k].sort()
        question = _TEMPLATES[sidx].format(
            n=n, n_minus_one=n - 1,
            source_id=S, sink_id=T,
            adj_text=str(adj_dict),
        )
        ans_str = str(int(ans))
        img = self._render(labels, edges, pos, rng)
        return question, ans_str, img

    def _render(self, labels, edges, pos, rng):
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_lo, x_hi = min(xs) - 0.6, max(xs) + 0.6
        y_lo, y_hi = min(ys) - 0.8, max(ys) + 0.8
        fig_w = max(5.0, (x_hi - x_lo) * 1.2)
        fig_h = max(3.5, (y_hi - y_lo) * 0.9)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Draw edges with capacities
        for (u, v, c) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if d == 0:
                continue
            ux, uy = dx / d, dy / d
            sx, sy = x1 + ux * 0.20, y1 + uy * 0.20
            ex, ey = x2 - ux * 0.22, y2 - uy * 0.22
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color="#333",
                                        lw=1.5))
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # offset perpendicular to edge so label doesn't overlap line
            perp_x, perp_y = -uy, ux
            offset = 0.18
            ax.text(mx + perp_x * offset, my + perp_y * offset,
                    f"cap:{c}", fontsize=10, ha="center", va="center",
                    color="#a0522d",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#fff8e6",
                              ec="#bbbbbb", lw=0.6))

        # Draw nodes
        for u, (x, y) in pos.items():
            lbl = labels[u]
            color = "#ffd1a4" if lbl in ("S", "T") else "#b8d4f0"
            edge = "#7e3a00" if lbl in ("S", "T") else "#1a3a6e"
            ax.add_patch(plt.Circle((x, y), 0.18, facecolor=color,
                                    edgecolor=edge, linewidth=1.6, zorder=5))
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=edge, zorder=6)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
