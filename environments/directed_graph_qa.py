"""
Directed Graph QA environment.

Renders a directed graph with 4-8 nodes and arrows between them.
Nodes as circles with labels, edges as arrows.
Questions about in-degree, out-degree, path existence, shortest path,
edge count, and cycle detection.
"""
import math
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class DirectedGraphQA(StandaloneVisualEnv):
    ENV_NAME = "directed_graph"

    QUESTION_TYPES = [
        "in_degree", "out_degree", "path_exists",
        "shortest_path_length", "count_edges", "is_cyclic",
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_nodes_range": (4, 5), "density": (0.2, 0.3), "qtypes": ["count_edges"]}
        if level == 1:
            return {"n_nodes_range": (4, 5), "density": (0.2, 0.35), "qtypes": ["count_edges", "out_degree"]}
        if level == 2:
            return {"n_nodes_range": (4, 6), "density": (0.2, 0.35), "qtypes": ["in_degree", "out_degree"]}
        if level == 3:
            return {"n_nodes_range": (5, 6), "density": (0.25, 0.4), "qtypes": ["in_degree", "out_degree", "path_exists"]}
        if level == 4:
            return {"n_nodes_range": (5, 6), "density": (0.25, 0.4), "qtypes": ["path_exists", "is_cyclic"]}
        if level == 5:
            return {"n_nodes_range": (5, 7), "density": (0.25, 0.4), "qtypes": ["path_exists", "shortest_path_length"]}
        if level == 6:
            return {"n_nodes_range": (5, 7), "density": (0.25, 0.45), "qtypes": ["shortest_path_length", "is_cyclic"]}
        if level == 7:
            return {"n_nodes_range": (6, 7), "density": (0.25, 0.45), "qtypes": ["shortest_path_length", "is_cyclic"]}
        if level == 8:
            return {"n_nodes_range": (6, 7), "density": (0.3, 0.45), "qtypes": ["shortest_path_length"]}
        return {"n_nodes_range": (6, 7), "density": (0.3, 0.45), "qtypes": ["shortest_path_length", "is_cyclic"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 705)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(30):
            result = self._try_generate(qtype, cfg, sub_rng)
            if result is not None:
                return result
        return None

    def _generate_graph(self, rng, num_nodes, edge_density=0.3, allow_cycles=True):
        """Generate a directed graph.

        Returns:
            labels: list of node labels
            adj: dict node_label -> list of neighbor labels
            edges: list of (from, to)
        """
        labels = [chr(65 + i) for i in range(num_nodes)]  # A, B, C, ...
        edges = []
        adj = {l: [] for l in labels}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if not allow_cycles and j <= i:
                    continue
                if rng.random() < edge_density:
                    edges.append((labels[i], labels[j]))
                    adj[labels[i]].append(labels[j])

        return labels, adj, edges

    def _bfs_shortest(self, adj, start, end):
        """BFS shortest path length. Returns -1 if no path."""
        if start == end:
            return 0
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in adj.get(node, []):
                if nb == end:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1

    def _has_cycle(self, labels, adj):
        """Detect cycle using DFS coloring."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {l: WHITE for l in labels}

        def dfs(node):
            color[node] = GRAY
            for nb in adj.get(node, []):
                if color[nb] == GRAY:
                    return True
                if color[nb] == WHITE and dfs(nb):
                    return True
            color[node] = BLACK
            return False

        for l in labels:
            if color[l] == WHITE:
                if dfs(l):
                    return True
        return False

    def _try_generate(self, qtype: str, cfg=None, sub_rng=None) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        sr = sub_rng or rng
        lo_n, hi_n = (cfg or {}).get("n_nodes_range", (4, 7))
        lo_d, hi_d = (cfg or {}).get("density", (0.2, 0.45))
        num_nodes = sr.randint(lo_n, hi_n)
        density = sr.uniform(lo_d, hi_d)

        if qtype == "is_cyclic":
            # Generate with/without cycles with 50/50 chance
            allow_cycles = rng.random() < 0.5
            if not allow_cycles:
                labels, adj, edges = self._generate_graph(
                    rng, num_nodes, density, allow_cycles=False)
            else:
                labels, adj, edges = self._generate_graph(
                    rng, num_nodes, density, allow_cycles=True)
        else:
            labels, adj, edges = self._generate_graph(
                rng, num_nodes, density, allow_cycles=True)

        if len(edges) < 3:
            return None

        img = self._render_graph(labels, edges, num_nodes)

        sidx = (self.seed or 0) % 16
        if qtype == "in_degree":
            target = rng.choice(labels)
            in_deg = sum(1 for _, t in edges if t == target)
            _P = [f'What is the in-degree of node {target}?',
                  f'Count the in-degree of node {target}.',
                  f'How many directed edges enter node {target}?',
                  f'Determine the in-degree of vertex {target} in the graph.',
                  f'Number of edges pointing into node {target}?',
                  f'Find the in-degree (incoming edge count) of node {target}.',
                  f'How many arcs have node {target} as their target?',
                  f'Compute the in-degree of {target} from the graph.',
                  f'In the directed graph, what is the in-degree of {target}?',
                  f'How many edges terminate at node {target}?',
                  f'Report the number of incoming edges at node {target}.',
                  f'What is deg-in({target}) for this directed graph?',
                  f'Count incoming directed edges at node {target}.',
                  f'From the graph, find the in-degree of {target}.',
                  f'How many edges enter vertex {target}?',
                  f'Determine the in-degree count at node {target}.']
            q = _P[sidx]
            return q, str(in_deg), img

        elif qtype == "out_degree":
            target = rng.choice(labels)
            out_deg = len(adj[target])
            _P = [f'What is the out-degree of node {target}?',
                  f'Count the out-degree of node {target}.',
                  f'How many directed edges leave node {target}?',
                  f'Determine the out-degree of vertex {target}.',
                  f'Number of edges leaving {target}?',
                  f'Find the out-degree (outgoing edge count) of {target}.',
                  f'How many arcs originate at {target}?',
                  f'Compute the out-degree of node {target} in the graph.',
                  f'In this directed graph, what is the out-degree of {target}?',
                  f'How many edges start at node {target}?',
                  f'Report the number of outgoing edges from {target}.',
                  f'What is deg-out({target}) in this directed graph?',
                  f'Count outgoing directed edges at {target}.',
                  f'From the graph, find the out-degree of node {target}.',
                  f'How many edges exit vertex {target}?',
                  f'Determine the out-degree count at node {target}.']
            q = _P[sidx]
            return q, str(out_deg), img

        elif qtype == "path_exists":
            src, dst = rng.sample(labels, 2)
            path_len = self._bfs_shortest(adj, src, dst)
            exists = "Yes" if path_len >= 0 else "No"
            _P = [f'Is there a directed path from node {src} to node {dst}? Answer Yes or No.',
                  f'Does a directed path exist from {src} to {dst}? Yes or No.',
                  f'Can you reach {dst} starting from {src} via directed edges? Yes/No.',
                  f'Is {dst} reachable from {src} in the directed graph? Answer Yes or No.',
                  f'From node {src}, can we follow directed edges to reach {dst}? Yes or No.',
                  f'Is there any directed path connecting {src} to {dst}? Yes/No answer.',
                  f'Determine whether a directed path exists between {src} and {dst}. Yes or No.',
                  f'Reachability: does a directed path from {src} to {dst} exist? Answer Yes or No.',
                  f'Can {dst} be reached from {src} along directed arcs? Yes or No.',
                  f'Starting at {src}, is there a sequence of directed edges ending at {dst}? Yes/No.',
                  f'In the directed graph, is {dst} reachable from {src}? Yes or No.',
                  f'Is node {dst} reachable from node {src} via directed edges? Yes/No.',
                  f'Given the graph, does there exist a directed path {src} → ... → {dst}? Yes or No.',
                  f'Is there a forward path from {src} to {dst} in the directed graph? Yes/No.',
                  f'Can we travel from {src} to {dst} following directed edges? Yes or No.',
                  f'Check: is {dst} directed-reachable from {src}? Yes/No answer.']
            q = _P[sidx]
            return q, exists, img

        elif qtype == "shortest_path_length":
            pairs_with_path = []
            for s in labels:
                for t in labels:
                    if s != t:
                        d = self._bfs_shortest(adj, s, t)
                        if d > 0:
                            pairs_with_path.append((s, t, d))
            if not pairs_with_path:
                return None
            src, dst, dist = rng.choice(pairs_with_path)
            _P = [f'What is the shortest path length (number of edges) from node {src} to node {dst}?',
                  f'Find the length (edge count) of the shortest directed path from {src} to {dst}.',
                  f'Shortest directed distance (in edges) between {src} and {dst}?',
                  f'How many edges does the shortest path from {src} to {dst} have?',
                  f'Determine the minimum number of edges in any directed path from {src} to {dst}.',
                  f'What is the fewest edges needed to go from {src} to {dst} along directed arcs?',
                  f'Compute the shortest-path distance (edges) from {src} to {dst}.',
                  f'In this directed graph, what is the BFS distance from {src} to {dst}?',
                  f'Length of shortest directed path {src} → {dst} (count edges)?',
                  f'How many edges are in the shortest directed path from node {src} to node {dst}?',
                  f'Find the minimum path length (edges) from {src} to {dst}.',
                  f'What is the shortest-hop count from {src} to {dst} in the directed graph?',
                  f'Determine the shortest directed-path edge count starting at {src} ending at {dst}.',
                  f'Count edges in the shortest directed path {src} to {dst}.',
                  f'Minimum edges along any directed path from {src} to {dst}?',
                  f'From the graph, compute the shortest directed path length (edges) from {src} to {dst}.']
            q = _P[sidx]
            return q, str(dist), img

        elif qtype == "count_edges":
            _P = ["How many directed edges are in this graph?",
                  "Count the total number of directed edges in the graph.",
                  "What is the edge count of this directed graph?",
                  "Determine the number of directed arcs in the graph.",
                  "How many arrows does this directed graph contain?",
                  "Total directed edges in the graph?",
                  "Report the number of edges in the directed graph.",
                  "What is |E| for this directed graph?",
                  "How many directed-edge pairs does the graph have?",
                  "Give the total count of directed edges.",
                  "Count arrows in this directed graph.",
                  "How many edges (directed) appear in this graph?",
                  "Determine the cardinality of the edge set in the directed graph.",
                  "How many directed links does this graph have?",
                  "Number of directed edges present in the graph?",
                  "Report |E| (edge count) for the directed graph."]
            q = _P[sidx]
            return q, str(len(edges)), img

        elif qtype == "is_cyclic":
            cyclic = self._has_cycle(labels, adj)
            answer = "Yes" if cyclic else "No"
            _P = ["Does this directed graph contain a cycle? Answer Yes or No.",
                  "Is there a cycle in the directed graph? Yes or No.",
                  "Does the directed graph have any cycles? Answer Yes/No.",
                  "Can we find a directed cycle in the graph? Yes or No.",
                  "Is this directed graph cyclic? Yes or No.",
                  "Check for cycles: does the directed graph contain one? Yes/No.",
                  "Does the graph contain at least one directed cycle? Yes or No.",
                  "Is there a closed directed path (cycle) in this graph? Yes or No.",
                  "Determine if the directed graph has a cycle. Yes/No.",
                  "Is the directed graph not a DAG (i.e., has a cycle)? Yes or No.",
                  "Does a directed cycle exist in the graph? Yes/No answer.",
                  "Is the graph cyclic (has at least one directed cycle)? Yes or No.",
                  "Check: does this directed graph contain a cycle? Answer Yes/No.",
                  "Is there at least one directed loop/cycle in the graph? Yes or No.",
                  "Is a cycle present in the directed graph? Answer Yes or No.",
                  "Does any cycle exist in this directed graph? Answer Yes or No."]
            q = _P[sidx]
            return q, answer, img

        return None

    def _render_graph(self, labels, edges, num_nodes):
        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * s, 8 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect('equal')
        ax.axis('off')

        palette = style["palette"]
        edge_color = style["geo_line_color"]
        lw = style["line_width"]

        # Place nodes in a circle
        radius = 2.5
        cx, cy = 3.5, 3.5
        positions = {}
        for i, label in enumerate(labels):
            angle = 2 * math.pi * i / num_nodes - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[label] = (x, y)

        # Draw edges
        drawn_edges = set()
        for src, dst in edges:
            if (src, dst) in drawn_edges:
                continue
            drawn_edges.add((src, dst))

            sx, sy = positions[src]
            dx, dy = positions[dst]

            has_reverse = (dst, src) in set(edges)
            rad = 0.2 if has_reverse else 0.0

            vec_x = dx - sx
            vec_y = dy - sy
            dist = math.sqrt(vec_x ** 2 + vec_y ** 2)
            if dist < 0.01:
                continue
            shrink = 0.4 / dist
            ax.annotate(
                '', xy=(dx - vec_x * shrink, dy - vec_y * shrink),
                xytext=(sx + vec_x * shrink, sy + vec_y * shrink),
                arrowprops=dict(
                    arrowstyle='->', color=edge_color, lw=lw + 0.5,
                    connectionstyle=f'arc3,rad={rad}'
                ),
                zorder=2,
            )

        # Draw nodes
        for i, label in enumerate(labels):
            x, y = positions[label]
            circle = plt.Circle((x, y), 0.35, facecolor=palette[i % len(palette)],
                                 edgecolor='black', linewidth=lw + 1, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=style["font_size_base"] + 2, fontweight='bold',
                    color='white', zorder=6, fontfamily=style["font_family"])

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_title('Directed Graph', fontsize=style["font_size_base"] + 3,
                     fontweight='bold', pad=12, fontfamily=style["font_family"])
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
