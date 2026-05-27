"""
Eulerian Path Count QA (v4 G8b, for graph theory).

Targets: graph theory -5.56 (idx=272 counts Eulerian paths directly).

Failure mode: 50% of flipped cases commit a specific wrong number, 25% use
"let me assume" heuristic. v3 doesn't do exhaustive enumeration — just
applies the formula loosely or guesses.

Task: render a small graph (≤10 nodes), ask either:
- "Can this graph be drawn in one stroke (Eulerian path/circuit)?"
- "How many odd-degree vertices does this graph have?"
- "How many distinct shapes (connected components) are shown?"

Reward: exact integer or yes/no.

Level axes:
  A) n_vertices: 4 at L0 -> 9 at L9
  B) density: sparse (L0) -> dense (L9)
  C) disconnected components possible at L6+
"""
import random
import math
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES_EULERIAN = [
    "Decide whether the {n}-vertex undirected graph below admits an Eulerian walk (a single continuous trail that uses every edge exactly once).\n\n"
    "### Game Rules:\n"
    "1. An Eulerian walk traverses every edge of the graph exactly once.\n"
    "2. The walk is one-stroke: vertices may be revisited, but edges may not.\n"
    "3. A connected graph admits such a walk iff it has 0 or 2 vertices of odd degree.\n\n"
    "### Coordinate System:\n"
    "- Vertices are labelled with integers 1..{n}, matching the labels shown in the image.\n"
    "- Edges are undirected pairs `i-j`.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Edges ({n_edges}): {edges}\n\n"
    "### Output Format:\n"
    "Output `yes` or `no` (lowercase) inside <answer>...</answer>.\n"
    "Example: <answer>yes</answer>",

    "Decide whether the graph below can be drawn in one continuous stroke without lifting the pen and without re-using any edge.\n\n"
    "### Game Rules:\n"
    "- Each edge may be used at most once.\n"
    "- Vertices may be re-visited.\n"
    "- A connected graph is one-stroke drawable iff it has 0 or 2 vertices of odd degree.\n\n"
    "### Coordinate System:\n"
    "- Vertex IDs are integers 1..{n}.\n"
    "- Edges are undirected.\n\n"
    "### Current Puzzle State:\n"
    "- Vertices: {n}\n"
    "- Edges: {edges}\n\n"
    "### Output Format:\n"
    "Answer `yes` or `no` inside <answer>...</answer>.",

    "Determine whether this graph has an Eulerian path.\n\n"
    "### Game Rules:\n"
    "An Eulerian path traverses every edge exactly once. A connected graph admits one iff it has either 0 or exactly 2 vertices of odd degree.\n\n"
    "### Coordinate System:\n"
    "- Vertices labelled 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "- Edge list ({n_edges}): {edges}\n\n"
    "### Output Format:\n"
    "Output `yes` or `no` inside <answer>...</answer>.",
]

_TEMPLATES_ODD = [
    "Count the number of vertices with odd degree in the {n}-vertex undirected graph below.\n\n"
    "### Game Rules:\n"
    "1. The degree of a vertex is the number of edges incident to it.\n"
    "2. A vertex has odd degree if its degree is an odd integer.\n"
    "3. Multi-edges and self-loops are not present.\n\n"
    "### Coordinate System:\n"
    "- Vertices are labelled 1..{n}, matching the labels in the image.\n"
    "- Edges are undirected pairs `i-j`.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Edges ({n_edges}): {edges}\n\n"
    "### Output Format:\n"
    "Output a single integer (the count of odd-degree vertices) inside <answer>...</answer>.\n"
    "Example: <answer>2</answer>",

    "How many vertices of the graph below have odd degree?\n\n"
    "### Game Rules:\n"
    "- Degree = number of edges incident to the vertex.\n"
    "- A vertex is odd-degree if its degree is an odd integer.\n\n"
    "### Coordinate System:\n"
    "- Vertices labelled 1..{n}.\n"
    "- Edges undirected.\n\n"
    "### Current Puzzle State:\n"
    "- Vertices: {n}\n"
    "- Edges: {edges}\n\n"
    "### Output Format:\n"
    "Output the integer count inside <answer>...</answer>.",

    "Count the odd-degree vertices in this graph.\n\n"
    "### Game Rules:\n"
    "Degree of a vertex is its incident-edge count; odd-degree means degree is odd.\n\n"
    "### Coordinate System:\n"
    "- Vertices labelled 1..{n}.\n\n"
    "### Current Puzzle State:\n"
    "- Edge list ({n_edges}): {edges}\n\n"
    "### Output Format:\n"
    "Output the integer count inside <answer>...</answer>.",
]

class EulerianPathCountQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "eulerian_path_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n = 4 + level // 2
        density = 0.5 + 0.05 * level  # edges roughly density * n*(n-1)/2
        qtype = "eulerian" if level % 2 == 0 else "odd_count"
        return {"n": n, "density": min(0.85, density), "qtype": qtype, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 221)
        self._primary_complexity_feature = level

        n = cfg["n"]
        # sample edges
        edges = set()
        target_edges = int(n * (n - 1) / 2 * cfg["density"])
        attempts = 0
        while len(edges) < target_edges and attempts < 50:
            attempts += 1
            a = rng.randint(0, n - 1)
            b = rng.randint(0, n - 1)
            if a == b:
                continue
            edges.add(tuple(sorted((a, b))))
        # ensure connected (BFS)
        adj = {i: set() for i in range(n)}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        visited = set()
        start = next(iter(adj))
        queue = [start]
        while queue:
            u = queue.pop(0)
            if u in visited:
                continue
            visited.add(u)
            for v in adj[u]:
                queue.append(v)
        unvisited = set(range(n)) - visited
        for v in unvisited:
            # connect to nearest visited node
            u = rng.choice(list(visited))
            edge = tuple(sorted((u, v)))
            edges.add(edge)
            adj[u].add(v)
            adj[v].add(u)
            visited.add(v)

        # compute answer
        degrees = [len(adj[i]) for i in range(n)]
        odd_count = sum(1 for d in degrees if d % 2 == 1)
        # eulerian if odd_count == 0 or 2
        is_eulerian = odd_count in (0, 2)

        # Format edges as "1-2, 1-3, ..." (1-indexed to match image labels).
        edges_sorted = sorted(edges)
        edges_str = ", ".join(f"{a + 1}-{b + 1}" for a, b in edges_sorted)
        n_edges = len(edges_sorted)

        if cfg["qtype"] == "eulerian":
            answer = "yes" if is_eulerian else "no"
            sidx = (self.seed or 0) % len(_TEMPLATES_EULERIAN)
            q = _TEMPLATES_EULERIAN[sidx].format(
                n=n, edges=edges_str, n_edges=n_edges,
            )
        else:
            answer = str(odd_count)
            sidx = (self.seed or 0) % len(_TEMPLATES_ODD)
            q = _TEMPLATES_ODD[sidx].format(
                n=n, edges=edges_str, n_edges=n_edges,
            )

        img = self._render(n, edges, rng)
        return q, answer, img

    @staticmethod
    def _format_state(n, edges) -> str:
        """Render the graph state (vertex count + edge list) as text. Used
        for diagnostics; the prompt embeds the same info via .format()."""
        edges_sorted = sorted(edges)
        edges_str = ", ".join(f"{a + 1}-{b + 1}" for a, b in edges_sorted)
        return f"Vertices: 1..{n}\nEdges ({len(edges_sorted)}): {edges_str}"

    def _render(self, n, edges, rng):
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        # Layout nodes on circle
        positions = {}
        for i in range(n):
            theta = 2 * math.pi * i / n
            positions[i] = (math.cos(theta), math.sin(theta))

        # Draw edges
        for a, b in edges:
            xa, ya = positions[a]
            xb, yb = positions[b]
            ax.plot([xa, xb], [ya, yb], color="black", lw=1.8)

        # Draw nodes
        for i, (x, y) in positions.items():
            ax.scatter(x, y, s=500, color="#3498db", zorder=5,
                       edgecolors="black", linewidths=1.5)
            ax.text(x, y, str(i + 1), fontsize=13, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_epc"
    os.makedirs(out_dir, exist_ok=True)
    env = EulerianPathCountQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 71
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[epc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/epc_s{s}_L{level}.png")
            print(f"[epc L{level} s{s}] A={env._answer}")
