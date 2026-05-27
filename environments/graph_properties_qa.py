"""
Graph Properties QA environment.

Capabilities: V3 (chart extraction), V2 (label reading), R5 (multi-step reasoning)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 4-node graph, ask "degree of vertex A" (highlighted).
L1: 4-node graph, ask "total edges".
L2: 5-node graph, ask "degree of vertex A".
L3: 5-node graph, ask "is this a tree" (Yes/No).
L4: 6-node graph, ask "total edges".
L5: 8-node graph, ask "connected components count".
L6: 7-node graph, ask "Euler path" (Yes/No).
L7: 6-node graph, ask "diameter".
L8: 7-node graph, ask "diameter".
L9: 8-node graph, ask "Hamiltonian path" (Yes/No).

parameter = {"level": int in [0, 9]}
"""
import math
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = ["Undirected Graph", "Graph", "Network", "Vertex-edge Diagram"]

class GraphPropertiesQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "graph_properties"

    QUESTION_TYPES = [
        "degree_of_vertex", "total_edges", "is_tree",
        "diameter", "has_euler_path", "has_hamilton_path",
        "connected_components",
        # reference D129/D130/D132 extensions:
        "is_cyclic", "is_complete", "edges_to_complete",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(30):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _level_config(self, level: int) -> Dict:
        # Reordered so L3 is not the "is_tree" yes/no cakewalk (0.80 spike).
        # is_tree moves to L1; L3 gets degree on a larger graph; diameter
        # stays at L7-8; Hamilton at L9.
        # Iter 3 (2026-04-17): L3=0.40 (dropped below L1=0.50) and L6=0.55
        # (above L3) — still non-monotonic. Degree-of-vertex on n=6 was
        # harder than L6's has_euler_path (yes/no). Keep L3 inside the
        # easy-counting band (total_edges) and push the degree-on-6 task
        # to L4.
        # Iter 4 (2026-04-17): L3=0.05 collapsed completely. total_edges
        # on n=6 becomes genuinely hard because densely-laid edges in
        # spring/grid layouts create visual ambiguity. Solution: keep L3
        # on is_tree (yes/no is easier) and move total_edges later.
        # reference additions: a fraction of seeds at higher levels rotate to
        # is_cyclic / is_complete / edges_to_complete (D129/D130/D132).
        seed_mod = (self.seed or 0) % 5
        if level == 0:
            return {"n": 4, "qtype": "degree_of_vertex"}
        if level == 1:
            return {"n": 4, "qtype": "total_edges"}
        if level == 2:
            return {"n": 5, "qtype": "is_tree"}
        if level == 3:
            return {"n": 5, "qtype": "is_tree"}  # yes/no — more forgiving
        if level == 4:
            # 1/5 of seeds: is_cyclic
            if seed_mod == 0:
                return {"n": 5, "qtype": "is_cyclic"}
            return {"n": 5, "qtype": "total_edges"}
        if level == 5:
            # 1/5 seeds: is_complete
            if seed_mod == 0:
                return {"n": 5, "qtype": "is_complete"}
            return {"n": 6, "qtype": "degree_of_vertex"}
        if level == 6:
            # 1/5 seeds: edges_to_complete
            if seed_mod == 0:
                return {"n": 6, "qtype": "edges_to_complete"}
            return {"n": 7, "qtype": "connected_components"}
        if level == 7:
            return {"n": 7, "qtype": "has_euler_path"}
        if level == 8:
            # 1/5 seeds: is_cyclic on larger graph
            if seed_mod == 0:
                return {"n": 7, "qtype": "is_cyclic"}
            return {"n": 6, "qtype": "diameter"}
        return {"n": 7, "qtype": "has_hamilton_path"}

    def _gen_tree(self, rng, n):
        labels = [chr(65 + i) for i in range(n)]
        adj = {l: set() for l in labels}
        edges = []
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(1, n):
            parent = rng.randint(0, i - 1)
            u, v = labels[perm[i]], labels[perm[parent]]
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
        return labels, adj, edges

    def _gen_connected(self, rng, n, extra=2):
        labels, adj, edges = self._gen_tree(rng, n)
        for _ in range(extra):
            i, j = rng.sample(range(n), 2)
            u, v = labels[i], labels[j]
            if v not in adj[u]:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
        return labels, adj, edges

    def _gen_disconnected(self, rng, n):
        # Two components
        a = rng.randint(2, n - 2)
        labels1, adj1, edges1 = self._gen_tree(rng, a)
        # rebuild for second part
        labels2 = [chr(65 + a + i) for i in range(n - a)]
        adj2 = {l: set() for l in labels2}
        edges2 = []
        for i in range(1, n - a):
            parent = rng.randint(0, i - 1)
            edges2.append((labels2[i], labels2[parent]))
            adj2[labels2[i]].add(labels2[parent])
            adj2[labels2[parent]].add(labels2[i])
        labels = labels1 + labels2
        adj = {**adj1, **adj2}
        edges = edges1 + edges2
        return labels, adj, edges

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

    def _components(self, labels, adj):
        seen = set()
        comps = 0
        for l in labels:
            if l not in seen:
                seen.update(self._bfs(adj, l))
                comps += 1
        return comps

    def _is_connected(self, labels, adj):
        if not labels:
            return True
        return len(self._bfs(adj, labels[0])) == len(labels)

    def _diameter(self, labels, adj):
        if not self._is_connected(labels, adj):
            return -1
        max_d = 0
        for u in labels:
            dist = {u: 0}
            queue = deque([u])
            while queue:
                node = queue.popleft()
                for v in adj[node]:
                    if v not in dist:
                        dist[v] = dist[node] + 1
                        queue.append(v)
            max_d = max(max_d, max(dist.values()))
        return max_d

    def _has_euler_path(self, labels, adj):
        if not self._is_connected(labels, adj):
            return False
        odd = sum(1 for l in labels if len(adj[l]) % 2 == 1)
        return odd == 0 or odd == 2

    def _has_hamilton_path(self, labels, adj):
        n = len(labels)
        if n > 10:
            return False
        visited = set()

        def backtrack(node, count):
            if count == n:
                return True
            visited.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    if backtrack(nb, count + 1):
                        return True
            visited.discard(node)
            return False

        for start in labels:
            visited.clear()
            if backtrack(start, 1):
                return True
        return False

    _Q_TEMPLATES = {
        "is_cyclic": [
            "Is this graph cyclic (does it contain a cycle)? Answer Yes or No.",
            "Determine whether the graph above contains at least one cycle. Answer Yes or No.",
            "Does this graph contain a cycle? Answer Yes or No.",
        ],
        "is_complete": [
            "Is this graph complete (every pair of distinct vertices connected by an edge)? Answer Yes or No.",
            "Determine whether the displayed graph is complete (all pairs of vertices joined). Answer Yes or No.",
            "Is the graph above a complete graph? Answer Yes or No.",
        ],
        "edges_to_complete": [
            "How many additional edges should be added to this graph so that it becomes complete? Answer with a single integer.",
            "Compute the number of edges to add so the graph becomes complete (every pair connected). Answer with a single integer.",
            "How many edges are missing for the displayed graph to become complete? Single integer answer.",
        ],
        "connected_components": [
            "How many connected components does this graph have? Answer with a single integer.",
            "Count the connected components in the graph above. Answer with a single integer.",
            "Examine the graph and report the number of connected components as a single integer.",
        ],
        "total_edges": [
            "How many edges does this graph have? Answer with a single integer.",
            "Count all edges in the graph and answer with a single integer.",
            "Report the total number of edges in the depicted graph (single integer).",
        ],
        "is_tree": [
            "Is this graph a tree? (Connected with no cycles.) Answer Yes or No.",
            "Determine whether the graph above is a tree (connected, acyclic). Answer Yes or No.",
            "Is the graph an (undirected) tree? Answer Yes or No.",
        ],
        "diameter": [
            "What is the diameter of this graph? (The longest shortest path between any pair of vertices.) Answer with a single integer.",
            "Compute the diameter (longest shortest-path distance) of the graph. Answer with a single integer.",
            "Find the diameter of the graph and answer with a single integer.",
        ],
        "has_euler_path": [
            "Does this graph have an Euler path? (A path visiting every edge exactly once.) Answer Yes or No.",
            "Determine whether the graph admits an Euler path (each edge traversed exactly once). Answer Yes or No.",
            "Does an Euler path exist in the graph above? Answer Yes or No.",
        ],
        "has_hamilton_path": [
            "Does this graph have a Hamiltonian path? (A path visiting every vertex exactly once.) Answer Yes or No.",
            "Determine whether the graph admits a Hamiltonian path. Answer Yes or No.",
            "Does the graph contain a Hamiltonian path? Answer Yes or No.",
        ],
        "degree_of_vertex_template": [
            "What is the degree of vertex {target} (highlighted in yellow)? Answer with a single integer.",
            "Count the number of edges incident to vertex {target} (highlighted). Answer with a single integer.",
            "Find the degree of the highlighted vertex {target}. Answer with a single integer.",
        ],
    }

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)
        n = cfg["n"]
        qtype = cfg["qtype"]
        # Layout choice from sub_rng so seed has visible effect.
        layout = rng.choice(["circle", "grid", "spring"])

        if qtype == "connected_components":
            if rng.random() < 0.5:
                labels, adj, edges = self._gen_connected(rng, n,
                                                          rng.randint(1, 3))
            else:
                labels, adj, edges = self._gen_disconnected(rng, n)
            answer = self._components(labels, adj)
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["connected_components"]),
                    str(answer), img)

        if qtype == "degree_of_vertex":
            labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 2))
            target = rng.choice(labels)
            deg = len(adj[target])
            img = self._render(rng, labels, edges, n, highlight={target},
                                layout=layout)
            tmpl = rng.choice(self._Q_TEMPLATES["degree_of_vertex_template"])
            return (tmpl.format(target=target), str(deg), img)

        if qtype == "total_edges":
            labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 3))
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["total_edges"]),
                    str(len(edges)), img)

        if qtype == "is_tree":
            if rng.random() < 0.5:
                labels, adj, edges = self._gen_tree(rng, n)
                ans = "Yes"
            else:
                labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 3))
                is_tree = (self._is_connected(labels, adj) and
                           len(edges) == n - 1)
                ans = "Yes" if is_tree else "No"
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["is_tree"]), ans, img)

        if qtype == "diameter":
            labels, adj, edges = self._gen_connected(rng, n, rng.randint(0, 2))
            d = self._diameter(labels, adj)
            if d < 1:
                return None
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["diameter"]), str(d), img)

        if qtype == "has_euler_path":
            labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 3))
            ans = "Yes" if self._has_euler_path(labels, adj) else "No"
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["has_euler_path"]), ans, img)

        if qtype == "has_hamilton_path":
            labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 4))
            ans = "Yes" if self._has_hamilton_path(labels, adj) else "No"
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["has_hamilton_path"]),
                    ans, img)

        # reference D129: cyclic Yes/No.
        if qtype == "is_cyclic":
            # 50/50 tree (no cycle) vs add edges (cycle).
            if rng.random() < 0.5:
                labels, adj, edges = self._gen_tree(rng, n)
                ans = "No"
            else:
                labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 3))
                # connected with extra edges => cycle definitely
                ans = "Yes" if len(edges) >= n else "No"
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["is_cyclic"]), ans, img)

        # reference D130: complete Yes/No.
        if qtype == "is_complete":
            if rng.random() < 0.5:
                # Construct complete graph K_n
                labels = [chr(65 + i) for i in range(n)]
                adj = {l: set() for l in labels}
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        u, v = labels[i], labels[j]
                        edges.append((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                ans = "Yes"
            else:
                labels, adj, edges = self._gen_connected(rng, n, rng.randint(1, 4))
                full = n * (n - 1) // 2
                ans = "Yes" if len(edges) == full else "No"
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["is_complete"]), ans, img)

        # reference D132: how many edges to add to be complete.
        if qtype == "edges_to_complete":
            labels, adj, edges = self._gen_connected(rng, n, rng.randint(0, 3))
            full = n * (n - 1) // 2
            answer = full - len(edges)
            img = self._render(rng, labels, edges, n, layout=layout)
            return (rng.choice(self._Q_TEMPLATES["edges_to_complete"]),
                    str(answer), img)

        return None

    def _render(self, rng, labels, edges, n, highlight=None, layout="circle"):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)
        fs = style["font_size_base"]
        ff = style["font_family"]
        base_lw = style["line_width"]
        ax.set_aspect("equal")
        ax.axis("off")

        cx, cy = 3.5, 3.5
        positions = {}
        if layout == "circle":
            radius = 2.5
            phase = rng.uniform(0, 2 * math.pi)
            for i, label in enumerate(labels):
                angle = 2 * math.pi * i / n - math.pi / 2 + phase
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                positions[label] = (x, y)
        elif layout == "grid":
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            spacing = 1.4
            for i, label in enumerate(labels):
                r = i // cols
                c = i % cols
                jitter_x = rng.uniform(-0.15, 0.15)
                jitter_y = rng.uniform(-0.15, 0.15)
                x = cx - (cols - 1) * spacing / 2 + c * spacing + jitter_x
                y = cy + (rows - 1) * spacing / 2 - r * spacing + jitter_y
                positions[label] = (x, y)
        else:  # spring-like simulated annealing (cheap)
            # Random initial positions then mild relaxation toward circle.
            radius = 2.3
            base_positions = []
            for i in range(n):
                a = 2 * math.pi * i / n + rng.uniform(-0.4, 0.4)
                rr = radius + rng.uniform(-0.5, 0.5)
                base_positions.append((cx + rr * math.cos(a),
                                        cy + rr * math.sin(a)))
            for i, label in enumerate(labels):
                positions[label] = base_positions[i]

        highlight = highlight or set()

        for u, v in edges:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], color="#7f8c8d",
                    linewidth=base_lw, zorder=1)

        for i, label in enumerate(labels):
            x, y = positions[label]
            hl = label in highlight
            fc = "#f1c40f" if hl else palette[i % len(palette)]
            ec = "#e74c3c" if hl else "black"
            lw = base_lw + 1 if hl else base_lw
            circle = plt.Circle((x, y), 0.35, facecolor=fc,
                                edgecolor=ec, linewidth=lw, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=fs + 2, fontweight="bold",
                    color="black" if hl else "white", zorder=6,
                    fontfamily=ff)

        # Compute bounds tightly so all layouts fit.
        all_xs = [p[0] for p in positions.values()]
        all_ys = [p[1] for p in positions.values()]
        pad = 0.7
        ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
        ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=fs + 3, fontweight="bold", pad=12)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = GraphPropertiesQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
