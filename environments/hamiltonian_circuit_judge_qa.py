"""
Yes/No Hamiltonian circuit detection on a small undirected graph.
Reference an external reference: "Is the following graph containing a Hamiltonian
circuit? choice: (A) Yes (B) No."

Mapping: D136 (reference gt — Hamiltonian circuit Y/N).
Studied qids (≥10): 227, 226 (Euler), 228, 220, 142, 120, 42, 221, 224, 223,
222, 225, 231, 232. Sample-derived design choice: reference an external reference phrases
the question as MCQ "(A) Yes (B) No" — so my answer is letter A or B (no
free-form). Graph rendering uses circular vertex layout to mirror reference's
graph theory image style (vertices as labeled circles, edges as straight
lines).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Decide whether the {n}-vertex undirected graph below contains a Hamiltonian circuit (a cycle that visits every vertex exactly once and returns to the start).\n\n"
    "### Game Rules:\n"
    "1. A Hamiltonian circuit visits each vertex exactly once and returns to the starting vertex.\n"
    "2. Each edge may be traversed at most once.\n"
    "3. Loops and multi-edges are not present in this graph.\n\n"
    "### Coordinate System:\n"
    "- Vertices are labelled with uppercase letters A, B, C, ... in the order shown above.\n"
    "- Edges are undirected (`A-B` means the edge between A and B).\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Vertices: {vertices}\n"
    "- Edge list: {edges}\n\n"
    "### Output Format:\n"
    "Provide just the choice letter (A or B) inside <answer>...</answer>.\n"
    "Choice: (A) Yes (B) No\n"
    "Example: <answer>A</answer>",

    "Examine the {n}-vertex graph and decide whether it admits a Hamiltonian circuit.\n\n"
    "### Game Rules:\n"
    "- A Hamiltonian circuit visits every vertex exactly once and ends back at the start.\n"
    "- Each edge may be used at most once.\n"
    "- The graph is undirected and simple.\n\n"
    "### Coordinate System:\n"
    "- Vertices are labelled A..{last_label}, matching the labels shown in the image.\n\n"
    "### Current Puzzle State:\n"
    "- Vertices: {vertices}\n"
    "- Edges: {edges}\n\n"
    "### Output Format:\n"
    "Output only the letter inside <answer>...</answer>. Choice: (A) Yes (B) No.",

    "Determine whether the graph shown contains a Hamiltonian circuit.\n\n"
    "### Game Rules:\n"
    "A Hamiltonian circuit traverses every vertex exactly once and returns to its origin. The graph is undirected and simple.\n\n"
    "### Coordinate System:\n"
    "- Vertices labelled A..{last_label}.\n"
    "- Edges are unordered pairs `X-Y`.\n\n"
    "### Current Puzzle State:\n"
    "- Vertex count: {n}\n"
    "- Edges ({n_edges}): {edges}\n\n"
    "### Output Format:\n"
    "Choice: (A) Yes (B) No. Put the single letter inside <answer>...</answer>.",
]


def _has_hamiltonian_circuit(adj: List[List[int]]) -> bool:
    n = len(adj)
    if n < 3:
        return False
    # Backtracking: starting at vertex 0, try to visit all and return to 0.
    visited = [False] * n
    visited[0] = True

    def bt(v: int, count: int) -> bool:
        if count == n:
            return adj[v][0] == 1
        for u in range(n):
            if not visited[u] and adj[v][u] == 1:
                visited[u] = True
                if bt(u, count + 1):
                    return True
                visited[u] = False
        return False
    return bt(0, 1)


class HamiltonianCircuitJudgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "hamiltonian_circuit_judge"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n = 4 + level // 3  # 4 → 7
        return {"level": level, "n": n}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1019 + level * 71 + 53)
        n = cfg["n"]
        for _ in range(60):
            # Decide whether to produce YES or NO instance (alternate via rng)
            target_yes = rng.random() < 0.5
            adj = [[0] * n for _ in range(n)]
            if target_yes:
                # Construct a Hamiltonian cycle then add a couple random extra edges
                perm = list(range(n))
                rng.shuffle(perm)
                for i in range(n):
                    a = perm[i]; b = perm[(i + 1) % n]
                    adj[a][b] = adj[b][a] = 1
                # add random extra edges
                extras = rng.randint(0, max(1, n // 2))
                for _ in range(extras):
                    a = rng.randint(0, n - 1); b = rng.randint(0, n - 1)
                    if a != b and adj[a][b] == 0:
                        adj[a][b] = adj[b][a] = 1
            else:
                # Random sparse graph; may or may not have Ham circuit. Verify it doesn't.
                edge_p = 0.35 + rng.random() * 0.15
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < edge_p:
                            adj[i][j] = adj[j][i] = 1
                if _has_hamiltonian_circuit(adj):
                    # remove a random edge to try to break it (best-effort)
                    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
                    if not edges:
                        continue
                    a, b = rng.choice(edges)
                    adj[a][b] = adj[b][a] = 0
                    if _has_hamiltonian_circuit(adj):
                        continue
                # also avoid isolated vertex (looks weird)
                if any(sum(adj[i]) == 0 for i in range(n)):
                    continue
            ans_letter = "A" if _has_hamiltonian_circuit(adj) else "B"
            sidx = (self.seed or 0) % len(_TEMPLATES)
            labels = "ABCDEFGHIJ"
            vertices = ", ".join(labels[i] for i in range(n))
            edges_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges_list.append(f"{labels[i]}-{labels[j]}")
            edges_str = ", ".join(edges_list)
            question = _TEMPLATES[sidx].format(
                n=n,
                last_label=labels[n - 1],
                vertices=vertices,
                edges=edges_str,
                n_edges=len(edges_list),
            )
            img = self._render(adj, n)
            return question, ans_letter, img
        return None

    @staticmethod
    def _format_state(adj, n) -> str:
        """Render the graph state as a list of A-B edges (mirrors the
        Current Puzzle State block in the prompt). Used for diagnostics."""
        labels = "ABCDEFGHIJ"
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append(f"{labels[i]}-{labels[j]}")
        return f"Vertices: {', '.join(labels[i] for i in range(n))}\n" \
               f"Edges: {', '.join(edges)}"

    def _render(self, adj, n) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        coords = []
        labels = "ABCDEFGHIJ"
        for i in range(n):
            angle = 2 * math.pi * i / n
            coords.append((math.cos(angle), math.sin(angle)))
        # Draw edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    ax.plot([coords[i][0], coords[j][0]],
                            [coords[i][1], coords[j][1]],
                            color="#1976d2", linewidth=1.6, zorder=1)
        # Draw vertices
        for i in range(n):
            x, y = coords[i]
            ax.scatter([x], [y], s=600, color="#fff",
                       edgecolors="#222", linewidths=1.8, zorder=3)
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=14, color="#222", zorder=4)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
