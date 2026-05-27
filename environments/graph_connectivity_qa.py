"""
Graph Connectivity QA environment.

Renders an undirected graph and asks about connectivity properties:
is_connected, count_components, shortest_path_length, bridge_edge,
articulation_point.
"""
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class GraphConnectivityQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "graph_connectivity"

    QUESTION_TYPES = [
        "is_connected", "count_components", "shortest_path_length",
        "bridge_edge", "articulation_point",
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_nodes": (5, 6), "qtypes": ["is_connected", "count_components"]}
        if level == 1:
            return {"n_nodes": (5, 6), "qtypes": ["is_connected", "count_components"]}
        if level == 2:
            return {"n_nodes": (5, 7), "qtypes": ["count_components"]}
        if level == 3:
            return {"n_nodes": (5, 7), "qtypes": ["count_components", "shortest_path_length"]}
        if level == 4:
            return {"n_nodes": (6, 8), "qtypes": ["shortest_path_length"]}
        if level == 5:
            return {"n_nodes": (6, 8), "qtypes": ["shortest_path_length", "bridge_edge"]}
        if level == 6:
            return {"n_nodes": (6, 9), "qtypes": ["bridge_edge", "articulation_point"]}
        if level == 7:
            return {"n_nodes": (7, 9), "qtypes": ["articulation_point"]}
        if level == 8:
            return {"n_nodes": (7, 9), "qtypes": ["articulation_point", "bridge_edge"]}
        return {"n_nodes": (7, 9), "qtypes": ["articulation_point"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 719)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(30):
            result = self._try_generate(qtype, cfg, sub_rng)
            if result is not None:
                return result
        return None

    # -------------------------------------------------------------- #
    # Graph generation helpers
    # -------------------------------------------------------------- #

    def _gen_graph(self, rng, n, density=0.3):
        labels = [chr(65 + i) for i in range(n)]
        adj = {l: set() for l in labels}
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    edges.append((labels[i], labels[j]))
                    adj[labels[i]].add(labels[j])
                    adj[labels[j]].add(labels[i])
        return labels, adj, edges

    def _gen_connected_graph(self, rng, n, extra_edges=3):
        """Generate a connected graph by building a spanning tree then adding extras."""
        labels = [chr(65 + i) for i in range(n)]
        adj = {l: set() for l in labels}
        edges = []
        # Random spanning tree via random permutation
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(1, n):
            parent = rng.randint(0, i - 1)
            u, v = labels[perm[i]], labels[perm[parent]]
            if v not in adj[u]:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
        # Add extra edges
        for _ in range(extra_edges):
            i, j = rng.sample(range(n), 2)
            u, v = labels[i], labels[j]
            if v not in adj[u]:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
        return labels, adj, edges

    def _gen_disconnected_graph(self, rng, n, num_components=None):
        """Generate a disconnected graph with specified number of components."""
        if num_components is None:
            num_components = rng.randint(2, min(4, n - 1))
        labels = [chr(65 + i) for i in range(n)]
        adj = {l: set() for l in labels}
        edges = []

        # Partition nodes into components
        sizes = []
        remaining = n
        for c in range(num_components - 1):
            sz = rng.randint(1, remaining - (num_components - c - 1))
            sizes.append(sz)
            remaining -= sz
        sizes.append(remaining)

        idx = 0
        for sz in sizes:
            comp_labels = labels[idx:idx + sz]
            # Connect within component
            if sz >= 2:
                perm = list(range(sz))
                rng.shuffle(perm)
                for i in range(1, sz):
                    u = comp_labels[perm[i]]
                    v = comp_labels[perm[rng.randint(0, i - 1)]]
                    if v not in adj[u]:
                        edges.append((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                # Add a few extra within component
                for _ in range(rng.randint(0, max(0, sz - 2))):
                    a, b = rng.sample(comp_labels, 2)
                    if b not in adj[a]:
                        edges.append((a, b))
                        adj[a].add(b)
                        adj[b].add(a)
            idx += sz
        return labels, adj, edges, num_components

    # -------------------------------------------------------------- #
    # Graph algorithms
    # -------------------------------------------------------------- #

    def _bfs(self, adj, start):
        visited = {start}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return visited

    def _count_components(self, labels, adj):
        visited = set()
        count = 0
        for l in labels:
            if l not in visited:
                visited |= self._bfs(adj, l)
                count += 1
        return count

    def _shortest_path(self, adj, start, end):
        if start == end:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            u, d = queue.popleft()
            for v in adj[u]:
                if v == end:
                    return d + 1
                if v not in visited:
                    visited.add(v)
                    queue.append((v, d + 1))
        return -1

    def _find_bridges(self, labels, adj):
        """Find bridge edges using DFS."""
        disc = {}
        low = {}
        bridges = []
        timer = [0]

        def dfs(u, parent):
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            for v in adj[u]:
                if v not in disc:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent:
                    low[u] = min(low[u], disc[v])

        for l in labels:
            if l not in disc:
                dfs(l, None)
        return bridges

    def _find_articulation_points(self, labels, adj):
        """Find articulation points using DFS."""
        disc = {}
        low = {}
        ap = set()
        timer = [0]

        def dfs(u, parent):
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            children = 0
            for v in adj[u]:
                if v not in disc:
                    children += 1
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if parent is None and children > 1:
                        ap.add(u)
                    if parent is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent:
                    low[u] = min(low[u], disc[v])

        for l in labels:
            if l not in disc:
                dfs(l, None)
        return ap

    # -------------------------------------------------------------- #
    # Problem generation
    # -------------------------------------------------------------- #

    def _try_generate(self, qtype: str, cfg=None, sub_rng=None) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        sr = sub_rng or rng
        lo_n, hi_n = (cfg or {}).get("n_nodes", (5, 9))
        num_nodes = sr.randint(lo_n, hi_n)

        if qtype == "is_connected":
            if rng.random() < 0.5:
                labels, adj, edges = self._gen_connected_graph(
                    rng, num_nodes, rng.randint(1, 4))
                answer = "Yes"
            else:
                labels, adj, edges, _ = self._gen_disconnected_graph(rng, num_nodes)
                answer = "No"
            if len(edges) < 3:
                return None
            img = self._render(labels, edges, num_nodes)
            q = "Is this graph connected? Answer Yes or No."
            return q, answer, img

        elif qtype == "count_components":
            if rng.random() < 0.3:
                labels, adj, edges = self._gen_connected_graph(
                    rng, num_nodes, rng.randint(1, 3))
                nc = 1
            else:
                labels, adj, edges, nc = self._gen_disconnected_graph(rng, num_nodes)
            if len(edges) < 2:
                return None
            img = self._render(labels, edges, num_nodes)
            q = "How many connected components does this graph have?"
            return q, str(nc), img

        elif qtype == "shortest_path_length":
            labels, adj, edges = self._gen_connected_graph(
                rng, num_nodes, rng.randint(2, 5))
            if len(edges) < 4:
                return None
            # Pick two nodes with path length > 1
            pairs = []
            for u in labels:
                for v in labels:
                    if u < v:
                        d = self._shortest_path(adj, u, v)
                        if d > 1:
                            pairs.append((u, v, d))
            if not pairs:
                return None
            src, dst, dist = rng.choice(pairs)
            img = self._render(labels, edges, num_nodes,
                               highlight_nodes={src, dst})
            q = (f"What is the shortest path length (in edges) from "
                 f"node {src} to node {dst}?")
            return q, str(dist), img

        elif qtype == "bridge_edge":
            labels, adj, edges = self._gen_connected_graph(
                rng, num_nodes, rng.randint(1, 3))
            bridges = self._find_bridges(labels, adj)
            if not bridges:
                # No bridges — ask "how many bridge edges"
                img = self._render(labels, edges, num_nodes)
                q = "How many bridge edges does this graph have? (A bridge edge is one whose removal disconnects the graph.)"
                return q, "0", img
            # Ask about a specific edge
            if rng.random() < 0.5 and bridges:
                u, v = rng.choice(bridges)
                answer = "Yes"
            else:
                non_bridges = [(a, b) for a, b in edges
                               if (a, b) not in bridges and (b, a) not in bridges]
                if non_bridges:
                    u, v = rng.choice(non_bridges)
                    answer = "No"
                else:
                    u, v = rng.choice(bridges)
                    answer = "Yes"
            img = self._render(labels, edges, num_nodes,
                               highlight_edges={(u, v)})
            q = (f"Is the edge between {u} and {v} (shown in red) a bridge? "
                 f"(Removing it disconnects the graph.) Answer Yes or No.")
            return q, answer, img

        elif qtype == "articulation_point":
            labels, adj, edges = self._gen_connected_graph(
                rng, num_nodes, rng.randint(1, 3))
            aps = self._find_articulation_points(labels, adj)
            if not aps:
                img = self._render(labels, edges, num_nodes)
                q = ("How many articulation points does this graph have? "
                     "(A vertex whose removal disconnects the graph.)")
                return q, "0", img
            # Ask if a specific node is an articulation point
            if rng.random() < 0.5:
                target = rng.choice(list(aps))
                answer = "Yes"
            else:
                non_aps = [l for l in labels if l not in aps]
                if non_aps:
                    target = rng.choice(non_aps)
                    answer = "No"
                else:
                    target = rng.choice(list(aps))
                    answer = "Yes"
            img = self._render(labels, edges, num_nodes,
                               highlight_nodes={target})
            q = (f"Is vertex {target} (highlighted) an articulation point? "
                 f"(Removing it and its edges disconnects the graph.) "
                 f"Answer Yes or No.")
            return q, answer, img

        return None

    # -------------------------------------------------------------- #
    # Rendering
    # -------------------------------------------------------------- #

    def _render(self, labels, edges, num_nodes,
                highlight_nodes=None, highlight_edges=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        base_lw = style["line_width"]
        ax.set_aspect('equal')
        ax.axis('off')

        radius = 2.5
        cx, cy = 3.5, 3.5
        positions = {}
        for i, label in enumerate(labels):
            angle = 2 * math.pi * i / num_nodes - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[label] = (x, y)

        highlight_nodes = highlight_nodes or set()
        highlight_edges = highlight_edges or set()

        # Draw edges
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            is_hl = ((u, v) in highlight_edges or (v, u) in highlight_edges)
            color = '#e74c3c' if is_hl else '#7f8c8d'
            lw = base_lw + 1.5 if is_hl else base_lw
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)

        # Draw nodes
        for i, label in enumerate(labels):
            x, y = positions[label]
            is_hl = label in highlight_nodes
            fc = '#f1c40f' if is_hl else palette[i % len(palette)]
            ec = '#e74c3c' if is_hl else 'black'
            lw = base_lw + 1 if is_hl else base_lw
            circle = plt.Circle((x, y), 0.35, facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=fs + 2, fontweight='bold',
                    color='black' if is_hl else 'white', zorder=6,
                    fontfamily=ff)

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_title('Undirected Graph', fontsize=fs + 3, fontweight='bold',
                     pad=12, fontfamily=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
