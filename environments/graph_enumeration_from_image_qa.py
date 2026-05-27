"""
Graph Enumeration From Image QA (v4 G8c, for graph theory).

Targets: graph theory -5.56 (complement to G8a maze_turn_sequence
and G8b eulerian_path_count).

Task: render a small graph, ask structural enumeration questions:
  - "How many connected components?"
  - "How many triangles in the graph?"
  - "How many simple paths of length k between A and B?"
  - "Does vertex X have a cycle of length 3 through it?"

Reward: exact integer or yes/no.

Level axes:
  A) Vertices: 4 at L0 -> 10 at L9
  B) Question type: component-count at L0, triangle-count at L3-5, path-count at L6+
"""
import random
import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES_COMP = [
    "How many connected components does this graph have? Put the integer in <answer>...</answer>.",
    "Count the number of connected components in the graph. Integer in <answer>...</answer>.",
    "Enumerate connected components; report the count as an integer in <answer>...</answer>.",
    "How many disjoint groups of connected vertices are in this graph? Integer in <answer>...</answer>.",
    "The graph has how many connected components? Put integer in <answer>...</answer>.",
    "Count the disjoint connected subgraphs. Integer in <answer>...</answer>.",
    "How many separate components? Integer in <answer>...</answer>.",
    "The graph's connected component count is? Put integer in <answer>...</answer>.",
    "Count the number of connected components. Put integer in <answer>...</answer>.",
    "How many connected parts? Integer in <answer>...</answer>.",
    "Number of connected components? Integer in <answer>...</answer>.",
    "Count the connected component subgraphs. Put integer in <answer>...</answer>.",
    "How many disjoint connected sets of vertices? Integer in <answer>...</answer>.",
    "Compute the number of connected components. Put integer in <answer>...</answer>.",
    "Integer: number of connected components. Put in <answer>...</answer>.",
    "Count connected components. Integer answer in <answer>...</answer>.",
]

_TEMPLATES_TRI = [
    "How many triangles (cycles of length 3) are in this graph? Put the integer in <answer>...</answer>.",
    "Count the number of triangles in the graph. Integer in <answer>...</answer>.",
    "Enumerate triangles (3-cycles); report count as integer in <answer>...</answer>.",
    "Number of triangles in the graph? Put integer in <answer>...</answer>.",
    "Triangle count? Integer in <answer>...</answer>.",
    "How many 3-cycles does the graph contain? Put integer in <answer>...</answer>.",
    "Count 3-vertex cycles. Integer in <answer>...</answer>.",
    "How many triangles? Put integer in <answer>...</answer>.",
    "Compute triangle count in the graph. Integer in <answer>...</answer>.",
    "Number of distinct triangles? Put integer in <answer>...</answer>.",
    "Report the triangle count in the graph. Integer in <answer>...</answer>.",
    "Count the triangles (length-3 cycles). Put integer in <answer>...</answer>.",
    "How many 3-cliques does this graph have? Integer in <answer>...</answer>.",
    "Integer: triangle count. Put in <answer>...</answer>.",
    "Count the triangles in the graph. Put integer in <answer>...</answer>.",
    "Triangles (3-cycles): count? Integer in <answer>...</answer>.",
]

_TEMPLATES_EDGE = [
    "How many edges does this graph have? Put the integer in <answer>...</answer>.",
    "Count the edges. Integer in <answer>...</answer>.",
    "Number of edges? Put integer in <answer>...</answer>.",
    "Edge count? Integer in <answer>...</answer>.",
    "How many line segments connecting vertex pairs? Integer in <answer>...</answer>.",
    "Count all edges in the graph. Put integer in <answer>...</answer>.",
    "Report edge count. Integer in <answer>...</answer>.",
    "Total edges? Put integer in <answer>...</answer>.",
    "How many edges? Put integer in <answer>...</answer>.",
    "Number of graph edges? Put integer in <answer>...</answer>.",
    "Edges total? Integer in <answer>...</answer>.",
    "Count the edges between distinct vertices. Put integer in <answer>...</answer>.",
    "Integer count of edges? Put in <answer>...</answer>.",
    "How many vertex-to-vertex connections? Integer in <answer>...</answer>.",
    "Compute the edge count. Integer in <answer>...</answer>.",
    "Edge count in the graph? Put integer in <answer>...</answer>.",
]

class GraphEnumerationFromImageQA(StandaloneVisualEnv):
    ENV_NAME = "graph_enumeration_from_image"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n = 4 + level
        if level <= 2:
            qtype = "components"
        elif level <= 5:
            qtype = "triangles"
        else:
            qtype = "edges"
        density = 0.35 + 0.03 * level
        return {"n": n, "density": min(0.7, density), "qtype": qtype}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 833)
        self._primary_complexity_feature = level

        n = cfg["n"]
        edges = set()
        target_edges = int(n * (n - 1) / 2 * cfg["density"])
        attempts = 0
        while len(edges) < target_edges and attempts < 100:
            attempts += 1
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            if a == b:
                continue
            edges.add(tuple(sorted((a, b))))

        # Adjacency
        adj = {i: set() for i in range(n)}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)

        # Compute answer
        if cfg["qtype"] == "components":
            visited = set()
            n_comp = 0
            for start in range(n):
                if start in visited:
                    continue
                n_comp += 1
                queue = [start]
                while queue:
                    u = queue.pop(0)
                    if u in visited:
                        continue
                    visited.add(u)
                    queue.extend(adj[u])
            answer = str(n_comp)
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_COMP[sidx]
        elif cfg["qtype"] == "triangles":
            # count triangles: iterate over each edge, find common neighbors
            tri = 0
            for a, b in edges:
                common = adj[a] & adj[b]
                tri += len(common)
            tri //= 3  # each triangle counted 3 times
            answer = str(tri)
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_TRI[sidx]
        else:  # edges
            answer = str(len(edges))
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_EDGE[sidx]

        img = self._render(n, edges, rng)
        return q, answer, img

    def _render(self, n, edges, rng):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        positions = {}
        for i in range(n):
            theta = 2 * math.pi * i / n
            positions[i] = (math.cos(theta), math.sin(theta))

        for a, b in edges:
            xa, ya = positions[a]
            xb, yb = positions[b]
            ax.plot([xa, xb], [ya, yb], color="black", lw=1.8)

        for i, (x, y) in positions.items():
            ax.scatter(x, y, s=500, color="#e74c3c", zorder=5,
                       edgecolors="black", linewidths=1.5)
            ax.text(x, y, str(i + 1), fontsize=13, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_gei"
    os.makedirs(out_dir, exist_ok=True)
    env = GraphEnumerationFromImageQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 67
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[gei L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/gei_s{s}_L{level}.png")
            print(f"[gei L{level} s{s}] A={env._answer}")
