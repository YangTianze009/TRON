"""
Find an Eulerian circuit in a small undirected graph: a closed walk that
traverses every edge exactly once. Output the vertex sequence (start
vertex appears at both ends), or No if no Eulerian circuit exists.

Difficulty: vertex count grows from 3 to 9. L0 is K_3 (a triangle), where
the only Eulerian circuit is A,B,C,A (or rotations).
"""
import math
import random
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to find an Eulerian cycle in the undirected graph below.\n\n"
    "### Game Rules:\n"
    "1. An Eulerian cycle is a closed walk that traverses every edge exactly once and returns to its start vertex.\n"
    "2. An Eulerian cycle exists if and only if the graph is connected and every vertex has even degree.\n"
    "3. If no Eulerian cycle exists, output the string `No`.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers labeled 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Provide the Eulerian cycle as a Python list of vertex IDs (start vertex appears at both ends) inside <answer>...</answer>.\n"
    "Example: <answer>[0, 7, 6, 3, 4, 2, 7, 5, 2, 1, 0]</answer> or <answer>No</answer>",

    "Find an Eulerian circuit in the graph below.\n\n"
    "### Game Rules:\n"
    "- A closed walk that uses every edge exactly once.\n"
    "- Output `No` if no such walk exists.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output a Python list of integer vertex IDs (start at both ends), or `No`, inside <answer>...</answer>.",

    "Your task is to find an Eulerian cycle in the graph described below.\n\n"
    "### Game Rules:\n"
    "Closed walk visiting every edge once; output `No` if impossible.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output a Python integer list inside <answer>...</answer>, or `No`.",
]


def _connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    """Check graph is connected over vertices that have any edges. Isolated
    vertices are ignored — Eulerian circuit requires the edge-induced
    subgraph to be connected."""
    if not edges:
        return n <= 1
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
    start = next(iter(nodes))
    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return visited == nodes


def _has_eulerian_circuit(n: int, edges: List[Tuple[int, int]]) -> bool:
    if not edges:
        return False
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d % 2 != 0 for d in deg):
        return False
    return _connected(n, edges)


def _hierholzer_circuit(n: int, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Run Hierholzer's algorithm. Returns list of vertices forming the
    Eulerian circuit (start appears at both ends), or None if not Eulerian.
    """
    if not _has_eulerian_circuit(n, edges):
        return None
    # Multigraph adjacency with edge ids for removal.
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # u -> [(v, edge_id), ...]
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    used = [False] * len(edges)
    # Start at vertex with smallest label that has edges
    start = min(u for u in range(n) if adj[u])
    stack = [start]
    path: List[int] = []
    while stack:
        u = stack[-1]
        # find unused edge
        while adj[u] and used[adj[u][-1][1]]:
            adj[u].pop()
        if adj[u]:
            v, eid = adj[u][-1]
            used[eid] = True
            adj[u].pop()
            stack.append(v)
        else:
            path.append(stack.pop())
    path.reverse()
    if len(path) != len(edges) + 1:
        return None
    return path


class EulerianCycleFindQA(StandaloneVisualEnv):
    ENV_NAME = "eulerian_cycle_find"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial_k3": True}
        n = min(4 + level // 2, 9)
        density = max(0.4, 0.75 - level * 0.03)
        return {"level": level, "n": n, "density": density,
                "trivial_k3": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1543 + level * 73 + 31)

        if cfg.get("trivial_k3"):
            n = 3
            edges = [(0, 1), (1, 2), (0, 2)]
        else:
            n = cfg["n"]
            # Generate random graph; with prob ~0.5 force Eulerian by fixing
            # all odd-degree vertices via pairing them with extra edges; else
            # leave as-is (likely No).
            force_yes = (rng.random() < 0.55)
            edges = self._sample_graph(rng, n, cfg["density"], force_yes,
                                        force_no=not force_yes)
            if edges is None:
                return None
        # Integer labels (structured-puzzle format)
        labels = [str(i) for i in range(n)]

        # Run check
        circuit = _hierholzer_circuit(n, edges)
        # Cache for verify-time check
        self._eul_n = n
        self._eul_edges_multiset = Counter()
        for (u, v) in edges:
            key = (min(u, v), max(u, v))
            self._eul_edges_multiset[key] += 1
        self._eul_total_edges = len(edges)

        if circuit is None:
            ans_str = "No"
        else:
            ans_str = str(circuit)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        # Build adjacency dict for prompt
        adj_dict = {i: [] for i in range(n)}
        for u, v in edges:
            adj_dict[u].append(v)
            adj_dict[v].append(u)
        for k in adj_dict:
            adj_dict[k].sort()
        question = _TEMPLATES[sidx].format(
            n=n, n_minus_one=n - 1, adj_text=str(adj_dict),
        )
        img = self._render(labels, edges, rng)
        return question, ans_str, img

    def _sample_graph(self, rng, n: int, density: float,
                       force_yes: bool, force_no: bool):
        for _ in range(15):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            if not edges:
                continue
            if not _connected(n, edges):
                # Try patching by adding one edge between components.
                comp = [-1] * n
                cid = 0
                for u, v in edges:
                    pass
                # Build adjacency for component finding
                adj = defaultdict(list)
                for u, v in edges:
                    adj[u].append(v); adj[v].append(u)
                used_nodes = set()
                for u, v in edges:
                    used_nodes.add(u); used_nodes.add(v)
                if not used_nodes:
                    continue
                # Find components among used_nodes
                seen = {}
                ccs = []
                for s in used_nodes:
                    if s in seen:
                        continue
                    stk = [s]
                    cur = []
                    while stk:
                        u = stk.pop()
                        if u in seen:
                            continue
                        seen[u] = len(ccs)
                        cur.append(u)
                        for w in adj[u]:
                            if w not in seen:
                                stk.append(w)
                    ccs.append(cur)
                # Add edges to connect ccs via lowest-index nodes
                for i in range(1, len(ccs)):
                    edges.append((min(ccs[0]), min(ccs[i])))
                if not _connected(n, edges):
                    continue
            # Compute degrees
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            odds = [u for u in range(n) if deg[u] % 2 == 1]
            if force_yes:
                # Pair up odd-degree vertices to make Eulerian.
                if len(odds) % 2 != 0:
                    continue  # shouldn't happen
                # Random pairing of odds.
                rng.shuffle(odds)
                added = []
                ok = True
                for i in range(0, len(odds), 2):
                    a, b = odds[i], odds[i + 1]
                    if a == b:
                        ok = False
                        break
                    added.append((min(a, b), max(a, b)))
                if not ok:
                    continue
                edges = edges + added
                if _hierholzer_circuit(n, edges) is None:
                    continue
                return edges
            elif force_no:
                # Want NOT Eulerian. If already not Eulerian, accept.
                if _hierholzer_circuit(n, edges) is None:
                    return edges
                # else keep trying
                continue
            else:
                return edges
        # Last-resort fallback: even ring (Eulerian) for force_yes;
        # path (not Eulerian) for force_no.
        if force_yes:
            edges = [(i, (i + 1) % n) for i in range(n)]
            return edges
        else:
            return [(i, i + 1) for i in range(n - 1)]

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        # Edges (parallel edges drawn with slight curvature)
        edge_count = Counter()
        for (u, v) in edges:
            key = (min(u, v), max(u, v))
            edge_count[key] += 1
        drawn = Counter()
        for (u, v) in edges:
            key = (min(u, v), max(u, v))
            mult = edge_count[key]
            k = drawn[key]
            drawn[key] += 1
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            if mult == 1:
                ax.plot([x1, x2], [y1, y2], color="#566573",
                        linewidth=1.7, zorder=1)
            else:
                # parallel: arc with rad offset
                rad = 0.15 * (k - (mult - 1) / 2.0) * 2
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-",
                                            color="#566573", lw=1.6,
                                            connectionstyle=f"arc3,rad={rad}"),
                            zorder=1)
        for i in range(n):
            x, y = pos[i]
            ax.add_patch(plt.Circle((x, y), 0.17, facecolor="#ffe0b2",
                                    edgecolor="#bf360c", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#bf360c",
                    zorder=6)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import ast
        gt = ground_truth.strip().lower()
        pred = predicted.strip()
        # Handle "No" case
        is_no_pred = bool(re.match(r"^\s*(no|n)\s*$", pred.lower()))
        if gt == "no":
            return is_no_pred
        if is_no_pred:
            return False
        # Try to parse Python list of integers
        seq_idx = None
        s = re.sub(r"```[^\n]*\n", "", pred).replace("```", "").strip()
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                seq_idx = [int(x) for x in obj]
        except (ValueError, SyntaxError, TypeError):
            pass
        if seq_idx is None:
            try:
                seq_idx = [int(x) for x in re.findall(r"-?\d+", s)]
            except ValueError:
                return False
        if not seq_idx:
            return False
        # Need exactly total_edges + 1 vertices
        need = self._eul_total_edges + 1
        if len(seq_idx) != need:
            return False
        if seq_idx[0] != seq_idx[-1]:
            return False
        for v in seq_idx:
            if not (0 <= v < self._eul_n):
                return False
        # Check edges
        used = Counter()
        for i in range(len(seq_idx) - 1):
            a, b = seq_idx[i], seq_idx[i + 1]
            key = (min(a, b), max(a, b))
            used[key] += 1
        return used == self._eul_edges_multiset
