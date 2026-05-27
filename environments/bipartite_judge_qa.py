"""
Bipartite check: given a small undirected graph, decide whether it is
bipartite (vertices can be 2-colored such that every edge crosses the
partition). Output Yes/No.
"""
import math
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows an undirected graph. Decide whether it is bipartite (vertices split into two groups with all edges between groups). Answer Yes or No in <answer>...</answer>.",
    "Is the graph in the image bipartite? Answer Yes or No in <answer>...</answer>.",
    "Determine whether the depicted graph is bipartite. Output Yes or No in <answer>...</answer>.",
    "Bipartiteness check: is the graph shown bipartite (can its vertices be 2-colored without any monochromatic edge)? Yes or No in <answer>...</answer>.",
    "Decide if the graph in the image admits a 2-coloring with no edge between same-colored vertices. Yes/No in <answer>...</answer>.",
    "From the depicted graph, output whether it is bipartite. Yes or No in <answer>...</answer>.",
    "Is the undirected graph shown free of odd-length cycles (i.e., bipartite)? Answer Yes or No in <answer>...</answer>.",
    "Output Yes if the graph in the image is bipartite, No otherwise. Place answer in <answer>...</answer>.",
    "Tell me whether the graph shown is bipartite. Yes/No in <answer>...</answer>.",
    "Bipartite or not? Inspect the graph in the image and answer Yes or No in <answer>...</answer>.",
    "Can the vertices of the depicted graph be partitioned into two sets so every edge goes between sets? Yes or No in <answer>...</answer>.",
    "Decide bipartiteness for the graph in the image. Answer Yes or No in <answer>...</answer>.",
    "Output Yes if the graph is bipartite, otherwise No. Place answer in <answer>...</answer>.",
    "Is the undirected graph drawn in the image bipartite? Output Yes or No in <answer>...</answer>.",
    "From the image, judge whether the graph is bipartite. Yes/No in <answer>...</answer>.",
    "Yes or No: is the depicted undirected graph bipartite? Place answer in <answer>...</answer>.",
]


def _is_bipartite(labels: List[str], adj: Dict[str, Set[str]]) -> bool:
    color: Dict[str, int] = {}
    for start in labels:
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def _bfs_visited(start: str, adj: Dict[str, Set[str]]) -> Set[str]:
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def _is_connected(labels: List[str], adj: Dict[str, Set[str]]) -> bool:
    if not labels:
        return True
    return len(_bfs_visited(labels[0], adj)) == len(labels)


class BipartiteJudgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "bipartite_judge"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "n": 4}
        if level <= 2:
            return {"level": level, "n": 4 + (level - 1)}
        if level <= 5:
            return {"level": level, "n": 6 + (level - 3) // 2}
        if level <= 7:
            return {"level": level, "n": 7}
        return {"level": level, "n": 8}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1597 + level * 67 + 41)
        n = cfg["n"]

        if level == 0:
            # Two trivial cases:
            #   bipartite: 4-vertex path A-B-C-D  (Yes)
            #   non-bipartite: triangle A-B-C with isolated D? Triangle alone (3 vertices)
            #   has an odd cycle. To keep n=4 with a clear odd cycle, use triangle
            #   A-B-C plus attached D-A (still odd cycle from A-B-C).
            want_yes = bool(rng.randint(0, 1))
            labels = ["A", "B", "C", "D"]
            if want_yes:
                # path A-B-C-D
                edges = [("A", "B"), ("B", "C"), ("C", "D")]
            else:
                # triangle ABC + pendant D from A (still triangle present -> odd cycle)
                edges = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")]
            adj: Dict[str, Set[str]] = {l: set() for l in labels}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            ans = "Yes" if _is_bipartite(labels, adj) else "No"
        else:
            # Pick a target answer; build a graph that satisfies it.
            want_yes = bool(rng.randint(0, 1))
            labels, edges, adj = self._build_graph(rng, n, want_yes, level)
            ans = "Yes" if _is_bipartite(labels, adj) else "No"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(labels, edges, rng)
        return question, ans, img

    def _build_graph(
        self, rng: random.Random, n: int, want_yes: bool, level: int
    ) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, Set[str]]]:
        labels = [chr(ord("A") + i) for i in range(n)]

        # Density / extra-edge count grows with level.
        extra = 1 + level // 3

        for attempt in range(40):
            adj: Dict[str, Set[str]] = {l: set() for l in labels}
            edges: List[Tuple[str, str]] = []
            if want_yes:
                # Build a guaranteed bipartite graph: random 2-partition,
                # then add only cross-edges.
                idxs = list(range(n))
                rng.shuffle(idxs)
                a_size = max(1, n // 2 + rng.randint(-1, 1))
                a_size = min(max(a_size, 1), n - 1)
                set_a = {labels[idxs[i]] for i in range(a_size)}
                set_b = {labels[idxs[i]] for i in range(a_size, n)}

                # Spanning-tree-ish: connect via crossing edges.
                a_list = list(set_a)
                b_list = list(set_b)
                rng.shuffle(a_list)
                rng.shuffle(b_list)
                # Connect every node by walking back and forth.
                merged = []
                i = j = 0
                while i < len(a_list) or j < len(b_list):
                    if i < len(a_list):
                        merged.append(a_list[i]); i += 1
                    if j < len(b_list):
                        merged.append(b_list[j]); j += 1
                for k in range(1, len(merged)):
                    u = merged[k]
                    v = merged[k - 1]
                    if (u in set_a) == (v in set_a):
                        # would not cross — find a partner that does
                        cands = [x for x in merged[:k]
                                 if (x in set_a) != (u in set_a)]
                        if not cands:
                            continue
                        v = rng.choice(cands)
                    if v not in adj[u]:
                        edges.append((u, v))
                        adj[u].add(v)
                        adj[v].add(u)

                # Add extras (still cross only).
                added = 0
                tries = 0
                want_extra = max(0, extra)
                while added < want_extra and tries < 60:
                    a = rng.choice(a_list) if a_list else None
                    b = rng.choice(b_list) if b_list else None
                    if a is None or b is None:
                        break
                    if b not in adj[a]:
                        edges.append((a, b))
                        adj[a].add(b)
                        adj[b].add(a)
                        added += 1
                    tries += 1
            else:
                # Build a non-bipartite graph: include at least one odd cycle.
                # Approach: pick 3 vertices for a triangle, then add a connected
                # spanning skeleton, then a few extras (any).
                tri = rng.sample(range(n), 3)
                a, b, c = labels[tri[0]], labels[tri[1]], labels[tri[2]]
                edges.extend([(a, b), (b, c), (c, a)])
                for u, v in [(a, b), (b, c), (c, a)]:
                    adj[u].add(v); adj[v].add(u)
                # Connect remaining vertices via spanning tree to triangle nodes.
                connected_nodes = {a, b, c}
                others = [labels[i] for i in range(n) if labels[i] not in connected_nodes]
                rng.shuffle(others)
                for u in others:
                    v = rng.choice(list(connected_nodes))
                    edges.append((u, v))
                    adj[u].add(v); adj[v].add(u)
                    connected_nodes.add(u)
                # Add extras anywhere.
                added = 0
                tries = 0
                want_extra = max(0, extra - 1)
                while added < want_extra and tries < 60:
                    i, j = rng.sample(range(n), 2)
                    u, v = labels[i], labels[j]
                    if v not in adj[u]:
                        edges.append((u, v))
                        adj[u].add(v); adj[v].add(u)
                        added += 1
                    tries += 1

            if not _is_connected(labels, adj):
                continue
            actually_bipartite = _is_bipartite(labels, adj)
            if actually_bipartite == want_yes:
                return labels, edges, adj

        # Fallback: return whatever we built last attempt; the answer is
        # whatever it actually is (still well-defined).
        return labels, edges, adj

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Place vertices on a circle.
        pos = {}
        for i, lbl in enumerate(labels):
            angle = 2 * math.pi * i / n + rng.uniform(-0.05, 0.05)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[lbl] = (r * math.cos(angle), r * math.sin(angle))
        for u, v in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#7f8c8d", linewidth=1.6, zorder=1)
        for lbl in labels:
            x, y = pos[lbl]
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor="#ffd6a5",
                                    edgecolor="#7a4f10", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#5a3300",
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
