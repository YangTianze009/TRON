"""
Graph Coloring QA environment.

Renders an undirected graph with 5-10 vertices, some pre-colored.
Questions about chromatic number, k-colorability, vertex coloring, bipartiteness.
"""
import math
import random
from collections import deque
from itertools import combinations, product
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_NODE_COLORS = {
    0: "#95a5a6",   # uncolored (gray)
    1: "#e74c3c",   # red
    2: "#3498db",   # blue
    3: "#27ae60",   # green
    4: "#f39c12",   # yellow
    5: "#9b59b6",   # purple
    6: "#1abc9c",   # teal
}
_COLOR_NAMES = {
    0: "uncolored", 1: "red", 2: "blue", 3: "green",
    4: "yellow", 5: "purple", 6: "teal",
}

class GraphColoringQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "graph_coloring"

    QUESTION_TYPES = [
        "min_colors", "can_color_with_k", "color_vertex", "is_bipartite",
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_nodes": (5, 6), "density": (0.25, 0.35), "qtypes": ["min_colors", "color_vertex"]}
        if level == 1:
            return {"n_nodes": (5, 6), "density": (0.25, 0.35), "qtypes": ["min_colors", "is_bipartite", "can_color_with_k"]}
        if level == 2:
            return {"n_nodes": (5, 6), "density": (0.25, 0.4), "qtypes": ["can_color_with_k"]}
        if level == 3:
            return {"n_nodes": (5, 7), "density": (0.25, 0.4), "qtypes": ["can_color_with_k", "color_vertex"]}
        if level == 4:
            return {"n_nodes": (5, 7), "density": (0.3, 0.4), "qtypes": ["color_vertex", "min_colors"]}
        if level == 5:
            return {"n_nodes": (6, 7), "density": (0.3, 0.45), "qtypes": ["min_colors", "color_vertex"]}
        if level == 6:
            return {"n_nodes": (6, 8), "density": (0.3, 0.45), "qtypes": ["min_colors"]}
        if level == 7:
            return {"n_nodes": (6, 8), "density": (0.3, 0.5), "qtypes": ["min_colors", "color_vertex"]}
        if level == 8:
            return {"n_nodes": (7, 8), "density": (0.35, 0.5), "qtypes": ["min_colors"]}
        return {"n_nodes": (7, 8), "density": (0.35, 0.5), "qtypes": ["min_colors", "color_vertex"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 718)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(30):
            result = self._try_generate(qtype, cfg, sub_rng)
            if result is not None:
                return result
        return None

    # -------------------------------------------------------------- #
    # Graph generation
    # -------------------------------------------------------------- #

    def _generate_undirected_graph(self, rng, n, density=0.35):
        """Return labels, adjacency dict, edge list for undirected graph."""
        labels = [chr(65 + i) for i in range(n)]
        edges = []
        adj = {l: set() for l in labels}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    edges.append((labels[i], labels[j]))
                    adj[labels[i]].add(labels[j])
                    adj[labels[j]].add(labels[i])
        return labels, adj, edges

    # -------------------------------------------------------------- #
    # Graph algorithms
    # -------------------------------------------------------------- #

    def _chromatic_number(self, labels, adj):
        """Brute-force chromatic number for small graphs."""
        n = len(labels)
        idx = {l: i for i, l in enumerate(labels)}
        edge_pairs = []
        for u in labels:
            for v in adj[u]:
                if idx[u] < idx[v]:
                    edge_pairs.append((idx[u], idx[v]))

        for k in range(1, n + 1):
            # Try all k-colorings
            for coloring in product(range(k), repeat=n):
                valid = True
                for u, v in edge_pairs:
                    if coloring[u] == coloring[v]:
                        valid = False
                        break
                if valid:
                    return k
        return n

    def _can_color_k(self, labels, adj, k):
        """Check if graph is k-colorable (backtracking)."""
        n = len(labels)
        idx = {l: i for i, l in enumerate(labels)}
        color = [-1] * n

        def backtrack(node):
            if node == n:
                return True
            label = labels[node]
            neighbor_colors = {color[idx[nb]] for nb in adj[label] if color[idx[nb]] >= 0}
            for c in range(k):
                if c not in neighbor_colors:
                    color[node] = c
                    if backtrack(node + 1):
                        return True
                    color[node] = -1
            return False

        return backtrack(0)

    def _is_bipartite(self, labels, adj):
        """BFS 2-coloring check."""
        color = {}
        for start in labels:
            if start in color:
                continue
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    def _greedy_color_vertex(self, labels, adj, pre_colors, target):
        """Find valid color for target given pre-colored neighbors."""
        neighbor_colors = set()
        for nb in adj[target]:
            if nb in pre_colors and pre_colors[nb] > 0:
                neighbor_colors.add(pre_colors[nb])
        for c in range(1, 7):
            if c not in neighbor_colors:
                return c
        return 1

    # -------------------------------------------------------------- #
    # Question generation
    # -------------------------------------------------------------- #

    def _try_generate(self, qtype: str, cfg=None, sub_rng=None) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        sr = sub_rng or rng
        lo_n, hi_n = (cfg or {}).get("n_nodes", (5, 8))
        lo_d, hi_d = (cfg or {}).get("density", (0.25, 0.5))
        num_nodes = sr.randint(lo_n, hi_n)
        density = sr.uniform(lo_d, hi_d)
        labels, adj, edges = self._generate_undirected_graph(rng, num_nodes, density)

        if len(edges) < 3:
            return None

        if qtype == "min_colors":
            chi = self._chromatic_number(labels, adj)
            # Pre-color some vertices to give partial info
            pre_colors = {}
            num_precolored = rng.randint(1, max(1, num_nodes // 3))
            precolored_nodes = rng.sample(labels, num_precolored)
            used_c = 0
            for node in precolored_nodes:
                used_c += 1
                pre_colors[node] = min(used_c, 6)

            img = self._render_graph(labels, edges, num_nodes, pre_colors)
            q = ("What is the minimum number of colors needed to color all "
                 "vertices of this graph so that no two adjacent vertices "
                 "share the same color (chromatic number)?")
            return q, str(chi), img

        elif qtype == "can_color_with_k":
            chi = self._chromatic_number(labels, adj)
            # Ask about k = chi-1 (No) or chi (Yes) or chi+1 (Yes)
            options = []
            if chi > 1:
                options.append((chi - 1, "No"))
            options.append((chi, "Yes"))
            if chi < num_nodes:
                options.append((chi + 1, "Yes"))
            k, answer = rng.choice(options)

            pre_colors = {}
            img = self._render_graph(labels, edges, num_nodes, pre_colors)
            q = (f"Can this graph be properly colored using at most {k} colors "
                 f"(no two adjacent vertices share a color)? Answer Yes or No.")
            return q, answer, img

        elif qtype == "color_vertex":
            # Pre-color all but one vertex, ask what color it should be
            target = rng.choice(labels)
            pre_colors = {}
            # Assign valid coloring to all other nodes first
            idx = {l: i for i, l in enumerate(labels)}
            coloring = [0] * len(labels)
            for i, l in enumerate(labels):
                if l == target:
                    continue
                nb_colors = {coloring[idx[nb]] for nb in adj[l]
                             if coloring[idx[nb]] > 0}
                for c in range(1, 7):
                    if c not in nb_colors:
                        coloring[i] = c
                        break
            for i, l in enumerate(labels):
                if l != target and coloring[i] > 0:
                    pre_colors[l] = coloring[i]

            valid_color = self._greedy_color_vertex(labels, adj, pre_colors, target)
            answer = _COLOR_NAMES[valid_color]

            img = self._render_graph(labels, edges, num_nodes, pre_colors,
                                     highlight_node=target)
            nb_colors_str = []
            for nb in sorted(adj[target]):
                if nb in pre_colors and pre_colors[nb] > 0:
                    nb_colors_str.append(f"{nb}={_COLOR_NAMES[pre_colors[nb]]}")
            q = (f"Vertex {target} (marked with '?') needs to be colored. "
                 f"Given the current coloring of its neighbors, what is the "
                 f"lowest-indexed valid color for vertex {target}? "
                 f"Colors in order: red, blue, green, yellow, purple, teal.")
            return q, answer, img

        elif qtype == "is_bipartite":
            # 50/50: generate bipartite or non-bipartite
            if rng.random() < 0.5:
                # Force bipartite: partition nodes, only cross-edges
                half = num_nodes // 2
                setA = labels[:half]
                setB = labels[half:]
                edges = []
                adj = {l: set() for l in labels}
                for a in setA:
                    for b in setB:
                        if rng.random() < density:
                            edges.append((a, b))
                            adj[a].add(b)
                            adj[b].add(a)
                if len(edges) < 3:
                    return None
            bip = self._is_bipartite(labels, adj)
            answer = "Yes" if bip else "No"
            img = self._render_graph(labels, edges, num_nodes, {})
            q = "Is this graph bipartite? Answer Yes or No."
            return q, answer, img

        return None

    # -------------------------------------------------------------- #
    # Rendering
    # -------------------------------------------------------------- #

    def _render_graph(self, labels, edges, num_nodes, pre_colors,
                      highlight_node=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        fs = style["font_size_base"]
        ff = style["font_family"]
        lw = style["line_width"]
        ax.set_aspect('equal')
        ax.axis('off')

        # Layout: circular
        radius = 2.5
        cx, cy = 3.5, 3.5
        positions = {}
        for i, label in enumerate(labels):
            angle = 2 * math.pi * i / num_nodes - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            positions[label] = (x, y)

        # Draw edges
        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=lw, zorder=1)

        # Draw nodes
        for label in labels:
            x, y = positions[label]
            c_idx = pre_colors.get(label, 0)
            fcolor = _NODE_COLORS.get(c_idx, _NODE_COLORS[0])

            if label == highlight_node:
                fcolor = "#ffffff"
                ec = "#e74c3c"
                node_lw = lw + 1
            else:
                ec = "black"
                node_lw = lw

            circle = plt.Circle((x, y), 0.35, facecolor=fcolor,
                                edgecolor=ec, linewidth=node_lw, zorder=5)
            ax.add_patch(circle)
            txt = "?" if label == highlight_node else label
            tcolor = 'black' if (label == highlight_node or c_idx == 0) else 'white'
            ax.text(x, y, txt, ha='center', va='center',
                    fontsize=fs + 2, fontweight='bold', color=tcolor,
                    zorder=6, fontfamily=ff)
            # Label outside circle
            lx = cx + (radius + 0.6) * math.cos(
                2 * math.pi * labels.index(label) / num_nodes - math.pi / 2)
            ly = cy + (radius + 0.6) * math.sin(
                2 * math.pi * labels.index(label) / num_nodes - math.pi / 2)
            if label == highlight_node:
                ax.text(lx, ly, label, ha='center', va='center',
                        fontsize=fs - 1, fontweight='bold', color='#e74c3c',
                        zorder=6, fontfamily=ff)

        # Color legend
        used_colors = set(pre_colors.values()) - {0}
        if used_colors:
            handles = [mpatches.Patch(color=_NODE_COLORS[c],
                       label=_COLOR_NAMES[c]) for c in sorted(used_colors)]
            ax.legend(handles=handles, loc='upper right', fontsize=fs - 3,
                      framealpha=0.9, title='Colors')

        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_title('Graph Coloring', fontsize=fs + 3, fontweight='bold',
                     pad=12, fontfamily=ff)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
