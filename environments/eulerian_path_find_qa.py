"""
Find an Eulerian path in a small undirected graph: a walk that traverses
every edge exactly once (start and end vertices may differ). Output the
vertex sequence, or No if no Eulerian path exists.

Difficulty: vertex count grows from 4 to 9. L0 is the 4-vertex path graph
A-B-C-D, which has the trivial Eulerian path A,B,C,D.
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
    "Your task is to find an Eulerian path in the undirected graph below.\n\n"
    "### Game Rules:\n"
    "1. An Eulerian path is a walk that traverses every edge exactly once. The start and end vertices may differ.\n"
    "2. An Eulerian path exists if the graph is connected and has 0 or exactly 2 vertices of odd degree.\n"
    "3. If no Eulerian path exists, output the string `No`.\n\n"
    "### Coordinate System:\n"
    "- Vertices are integers labeled 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "- Number of vertices: {n}\n"
    "- Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Provide the Eulerian path as a Python list of vertex IDs inside <answer>...</answer>.\n"
    "Example: <answer>[0, 5, 4, 3, 2, 4, 0, 1, 2, 0]</answer> or <answer>No</answer>",

    "Find an Eulerian path in the graph below.\n\n"
    "### Game Rules:\n"
    "- A walk that uses every edge exactly once.\n"
    "- Output `No` if no such walk exists.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output a Python list of integers, or `No`, inside <answer>...</answer>.",

    "Your task is to find an Eulerian path in the graph described below.\n\n"
    "### Game Rules:\n"
    "Walk visiting every edge once; output `No` if impossible.\n\n"
    "### Coordinate System:\n"
    "- Integer labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "Adjacency list: {adj_text}\n\n"
    "### Output Format:\n"
    "Output the path as a Python list inside <answer>...</answer>, or `No`.",
]


def _connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    if not edges:
        return n <= 1
    adj = defaultdict(list)
    used_nodes = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        used_nodes.add(u); used_nodes.add(v)
    start = next(iter(used_nodes))
    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return visited == used_nodes


def _has_eulerian_path(n: int, edges: List[Tuple[int, int]]) -> bool:
    if not edges:
        return False
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odd = sum(1 for d in deg if d % 2 == 1)
    if odd not in (0, 2):
        return False
    return _connected(n, edges)


def _hierholzer_path(n: int, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    if not _has_eulerian_path(n, edges):
        return None
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    odds = [u for u in range(n) if deg[u] % 2 == 1]
    # Start at an odd-degree vertex if any; else lowest-label vertex with edges.
    if odds:
        start = min(odds)
    else:
        start = min(u for u in range(n) if deg[u] > 0)
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    used = [False] * len(edges)
    stack = [start]
    path: List[int] = []
    while stack:
        u = stack[-1]
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


class EulerianPathFindQA(StandaloneVisualEnv):
    ENV_NAME = "eulerian_path_find"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial_path": True}
        n = min(4 + level // 2, 9)
        density = max(0.4, 0.75 - level * 0.03)
        return {"level": level, "n": n, "density": density,
                "trivial_path": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1487 + level * 79 + 41)

        if cfg.get("trivial_path"):
            n = 4
            edges = [(0, 1), (1, 2), (2, 3)]   # path 0-1-2-3
        else:
            n = cfg["n"]
            force_yes = (rng.random() < 0.6)
            edges = self._sample_graph(rng, n, cfg["density"], force_yes)
            if edges is None:
                return None
        labels = [str(i) for i in range(n)]

        path = _hierholzer_path(n, edges)
        # Cache for verify
        self._eul_n = n
        self._eul_edges_multiset = Counter()
        for (u, v) in edges:
            key = (min(u, v), max(u, v))
            self._eul_edges_multiset[key] += 1
        self._eul_total_edges = len(edges)

        if path is None:
            ans_str = "No"
        else:
            ans_str = str(path)

        sidx = (self.seed or 0) % len(_TEMPLATES)
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

    def _sample_graph(self, rng, n: int, density: float, force_yes: bool):
        for _ in range(15):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        edges.append((i, j))
            if not edges:
                continue
            # Ensure connected by patching
            if not _connected(n, edges):
                adj = defaultdict(list)
                for u, v in edges:
                    adj[u].append(v); adj[v].append(u)
                used_nodes = set()
                for u, v in edges:
                    used_nodes.add(u); used_nodes.add(v)
                if not used_nodes:
                    continue
                seen: Dict[int, int] = {}
                ccs: List[List[int]] = []
                for s in used_nodes:
                    if s in seen:
                        continue
                    stk = [s]; cur: List[int] = []
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
                for i in range(1, len(ccs)):
                    edges.append((min(ccs[0]), min(ccs[i])))
                if not _connected(n, edges):
                    continue
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            odds = [u for u in range(n) if deg[u] % 2 == 1]
            has_path = (len(odds) in (0, 2))
            if force_yes:
                if has_path:
                    return edges
                # Pair up extra odds (keep 2 unpaired) by adding edges among
                # them — walk them in order, pair (odds[0], odds[1]) skipped,
                # then (odds[2], odds[3]), (odds[4], odds[5]) ...
                if len(odds) <= 2:
                    return edges
                rng.shuffle(odds)
                # keep two as endpoints, pair the rest
                keep = odds[:2]
                rest = odds[2:]
                added = []
                ok = True
                for i in range(0, len(rest), 2):
                    if i + 1 >= len(rest):
                        ok = False; break
                    a, b = rest[i], rest[i + 1]
                    if a == b:
                        ok = False; break
                    added.append((min(a, b), max(a, b)))
                if not ok:
                    continue
                edges = edges + added
                if _hierholzer_path(n, edges) is None:
                    continue
                return edges
            else:
                if not has_path:
                    return edges
                continue
        # Fallback: if force_yes, return a path graph; else a star with 4
        # leaves (>=4 odd vertices → No).
        if force_yes:
            return [(i, i + 1) for i in range(n - 1)]
        else:
            if n >= 5:
                return [(0, i) for i in range(1, n)]
            return [(0, 1), (0, 2), (0, 3)]

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
        edge_count = Counter()
        for (u, v) in edges:
            edge_count[(min(u, v), max(u, v))] += 1
        drawn = Counter()
        for (u, v) in edges:
            key = (min(u, v), max(u, v))
            mult = edge_count[key]
            k = drawn[key]
            drawn[key] += 1
            x1, y1 = pos[u]; x2, y2 = pos[v]
            if mult == 1:
                ax.plot([x1, x2], [y1, y2], color="#37474f",
                        linewidth=1.7, zorder=1)
            else:
                rad = 0.15 * (k - (mult - 1) / 2.0) * 2
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-",
                                            color="#37474f", lw=1.6,
                                            connectionstyle=f"arc3,rad={rad}"),
                            zorder=1)
        for i in range(n):
            x, y = pos[i]
            ax.add_patch(plt.Circle((x, y), 0.17, facecolor="#bbdefb",
                                    edgecolor="#0d47a1", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#0d47a1",
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
        is_no_pred = bool(re.match(r"^\s*(no|n)\s*$", pred.lower()))
        if gt == "no":
            return is_no_pred
        if is_no_pred:
            return False
        s = re.sub(r"```[^\n]*\n", "", pred).replace("```", "").strip()
        seq_idx = None
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
        need = self._eul_total_edges + 1
        if len(seq_idx) != need:
            return False
        for v in seq_idx:
            if not (0 <= v < self._eul_n):
                return False
        used = Counter()
        for i in range(len(seq_idx) - 1):
            a, b = seq_idx[i], seq_idx[i + 1]
            used[(min(a, b), max(a, b))] += 1
        return used == self._eul_edges_multiset
