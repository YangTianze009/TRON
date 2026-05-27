"""
Decide whether two small undirected graphs G1, G2 (drawn side by side) are
isomorphic. Output Yes / No.

Difficulty: vertex count grows from 3 to 10. For levels >= 3, "No" cases
are filtered to share the same degree sequence so the model cannot win by
counting degrees alone.
"""
import math
import random
from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to decide whether the two undirected graphs G1 and G2 below are isomorphic.\n\n"
    "### Game Rules:\n"
    "1. Two graphs are isomorphic if there exists a bijection between their vertex sets that preserves all edges.\n"
    "2. Answer `Yes` if such a bijection exists, otherwise answer `No`.\n\n"
    "### Coordinate System:\n"
    "- Vertices in each graph are integers labeled 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "- G1 adjacency list: {adj1_text}\n"
    "- G2 adjacency list: {adj2_text}\n\n"
    "### Output Format:\n"
    "Output `Yes` or `No` inside <answer>...</answer>.\n"
    "Example: <answer>Yes</answer> or <answer>No</answer>",

    "Decide whether the two graphs G1 and G2 below are isomorphic.\n\n"
    "### Game Rules:\n"
    "- Isomorphic = there is a vertex bijection that preserves edges.\n"
    "- Output `Yes` or `No`.\n\n"
    "### Coordinate System:\n"
    "- Integer vertex labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "G1: {adj1_text}\n"
    "G2: {adj2_text}\n\n"
    "### Output Format:\n"
    "Output `Yes` or `No` inside <answer>...</answer>.",

    "Your task is to determine isomorphism for the two graphs described below.\n\n"
    "### Game Rules:\n"
    "Standard graph isomorphism. Output `Yes` or `No`.\n\n"
    "### Coordinate System:\n"
    "- Integer labels 0..{n_minus_one}.\n\n"
    "### Current Puzzle State:\n"
    "G1: {adj1_text}\n"
    "G2: {adj2_text}\n\n"
    "### Output Format:\n"
    "Output `Yes` or `No` inside <answer>...</answer>.",
]


def _adj_to_text(adj):
    n = len(adj)
    d = {i: sorted(adj[i]) for i in range(n)}
    return str(d)


def _are_isomorphic(n: int, adj1: List[Set[int]], adj2: List[Set[int]]) -> bool:
    """Brute-force isomorphism via permutation. n must be small (<=10)."""
    deg1 = sorted(len(a) for a in adj1)
    deg2 = sorted(len(a) for a in adj2)
    if deg1 != deg2:
        return False
    # Edge sets for fast membership test
    e1 = set()
    for u in range(n):
        for v in adj1[u]:
            if u < v:
                e1.add((u, v))
    e2 = set()
    for u in range(n):
        for v in adj2[u]:
            if u < v:
                e2.add((u, v))
    if len(e1) != len(e2):
        return False
    # Search permutations, prune by degree
    deg2_list = [len(a) for a in adj2]
    nodes_by_deg2: Dict[int, List[int]] = {}
    for v, d in enumerate(deg2_list):
        nodes_by_deg2.setdefault(d, []).append(v)
    deg1_list = [len(a) for a in adj1]
    candidates = [list(nodes_by_deg2.get(deg1_list[u], [])) for u in range(n)]
    used = [False] * n
    mapping = [-1] * n

    def dfs(u: int) -> bool:
        if u == n:
            # check edges
            for a, b in e1:
                ma, mb = mapping[a], mapping[b]
                if ma > mb:
                    ma, mb = mb, ma
                if (ma, mb) not in e2:
                    return False
            return True
        for v in candidates[u]:
            if used[v]:
                continue
            # partial check: every edge from u to assigned x must map to (v, mapping[x])
            ok = True
            for x in adj1[u]:
                if x < u:
                    mx = mapping[x]
                    a, b = (mx, v) if mx < v else (v, mx)
                    if (a, b) not in e2:
                        ok = False
                        break
            if not ok:
                continue
            used[v] = True
            mapping[u] = v
            if dfs(u + 1):
                return True
            used[v] = False
            mapping[u] = -1
        return False

    return dfs(0)


def _sample_graph(n: int, density: float, rng: random.Random) -> List[Set[int]]:
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _permute(adj: List[Set[int]], perm: List[int]) -> List[Set[int]]:
    n = len(adj)
    new_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            new_adj[perm[u]].add(perm[v])
    return new_adj


class GraphIsomorphismQA(StandaloneVisualEnv):
    ENV_NAME = "graph_isomorphism"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial": True}
        # Vertex count.
        if level <= 2:
            n = 4 + level         # 5..6
        elif level <= 5:
            n = 6 + (level - 2)   # 7..9
        else:
            n = 9 + min(1, level - 6)  # 9..10
            n = min(n, 10)
        density = 0.4 + 0.04 * level
        density = min(0.7, density)
        return {"level": level, "trivial": False, "n": n, "density": density,
                "match_degree_seq": level >= 3}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1117 + level * 41 + 23)

        if cfg.get("trivial"):
            # 50/50: identical K_3 (Yes) or K_3 vs P_3 (No)
            yes_case = (rng.random() < 0.5)
            if yes_case:
                # K_3 on both sides
                adj1 = [{1, 2}, {0, 2}, {0, 1}]
                adj2 = [{1, 2}, {0, 2}, {0, 1}]
                ans = "Yes"
            else:
                adj1 = [{1, 2}, {0, 2}, {0, 1}]   # K_3
                adj2 = [{1}, {0, 2}, {1}]         # P_3 (path A-B-C)
                ans = "No"
            n = 3
            labels1 = ["A", "B", "C"]
            labels2 = ["A", "B", "C"]
        else:
            n = cfg["n"]
            density = cfg["density"]
            yes_case = (rng.random() < 0.5)
            attempts = 0
            adj1 = adj2 = None
            while attempts < 24:
                attempts += 1
                a1 = _sample_graph(n, density, rng)
                # avoid trivial isolated-vertex graphs
                if any(len(a) == 0 for a in a1):
                    continue
                if yes_case:
                    perm = list(range(n))
                    rng.shuffle(perm)
                    a2 = _permute(a1, perm)
                    adj1, adj2 = a1, a2
                    break
                else:
                    # No case
                    a2 = _sample_graph(n, density, rng)
                    if any(len(a) == 0 for a in a2):
                        continue
                    deg1 = sorted(len(a) for a in a1)
                    deg2 = sorted(len(a) for a in a2)
                    if cfg.get("match_degree_seq"):
                        if deg1 != deg2:
                            continue
                    # Check NOT isomorphic
                    if _are_isomorphic(n, a1, a2):
                        continue
                    adj1, adj2 = a1, a2
                    break
            if adj1 is None:
                # Fallback: easier no-case (different edge count)
                a1 = _sample_graph(n, density, rng)
                a2 = _sample_graph(n, max(0.15, density - 0.2), rng)
                if _are_isomorphic(n, a1, a2):
                    return None
                adj1, adj2 = a1, a2
                yes_case = False
            ans = "Yes" if yes_case else "No"
            labels1 = [chr(ord("A") + i) for i in range(n)]
            labels2 = [chr(ord("a") + i) for i in range(n)]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            n_minus_one=n - 1,
            adj1_text=_adj_to_text(adj1),
            adj2_text=_adj_to_text(adj2),
        )
        img = self._render(adj1, labels1, adj2, labels2, rng)
        return question, ans, img

    def _render(self, adj1, labels1, adj2, labels2, rng):
        n = len(labels1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        for ax, adj, labels, title, fc, ec in (
            (axes[0], adj1, labels1, "G1", "#cfe2ff", "#0b3d91"),
            (axes[1], adj2, labels2, "G2", "#d1f0d1", "#1b5e20"),
        ):
            ax.set_facecolor("#ffffff")
            pos = {}
            for i in range(n):
                angle = 2 * math.pi * i / n + rng.uniform(-0.04, 0.04)
                r = 1.1 + rng.uniform(-0.05, 0.05)
                pos[i] = (r * math.cos(angle), r * math.sin(angle))
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        x1, y1 = pos[u]
                        x2, y2 = pos[v]
                        ax.plot([x1, x2], [y1, y2], color="#566573",
                                linewidth=1.6, zorder=1)
            for i in range(n):
                x, y = pos[i]
                ax.add_patch(plt.Circle((x, y), 0.16, facecolor=fc,
                                        edgecolor=ec, linewidth=1.4,
                                        zorder=5))
                ax.text(x, y, labels[i], ha="center", va="center",
                        fontsize=12, fontweight="bold", color=ec, zorder=6)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.axis("off")

        plt.tight_layout()
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
