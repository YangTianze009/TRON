"""
Prim's algorithm first edge: given a small undirected weighted graph and
a designated start vertex, output the first edge added to the minimum
spanning tree by Prim's algorithm. Ties are broken in alphabetical order
on the edge label (each label is its two endpoint letters in
alphabetical order).

Mirrors qid 152 wording from the source distribution:
  "what is the first edge added to the MST when running Prim's
   Algorithm from node A? In the case of a tie, choose the edge which
   comes first in alphabetical order."

Output format: two letters (e.g. AG). Endpoints accepted in either order.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows an undirected weighted graph. What is the first edge added to the MST when running Prim's Algorithm from node {start}? In the case of a tie, choose the edge which comes first in alphabetical order. Output the edge as two letters in <answer>...</answer>.",
    "Run Prim's algorithm starting at node {start} on the weighted graph in the image. What is the first edge added to the spanning tree? Break ties alphabetically on edge labels. Answer in <answer>...</answer>.",
    "From the weighted graph in the image, output the first edge picked by Prim's algorithm starting at vertex {start}. Tie-break: alphabetical edge label. Answer in <answer>...</answer> as two letters.",
    "What is the first edge that Prim's algorithm adds to the MST when started from node {start} on the depicted graph? Use alphabetical order to break ties. Place answer (two letters) in <answer>...</answer>.",
    "Inspect the weighted graph in the image. Prim's algorithm starts at node {start}. What is the very first edge it adds to the MST? Tie-break: alphabetical. Answer in <answer>...</answer>.",
    "Apply Prim's algorithm beginning at vertex {start} to the depicted weighted graph. Output the first edge selected. Ties: alphabetical edge label. In <answer>...</answer>.",
    "Trace Prim's algorithm from start vertex {start} on the weighted graph shown. What is the first edge added to the MST? Alphabetical tie-break. Answer in <answer>...</answer>.",
    "Find the first edge picked by Prim's algorithm starting at node {start}, applied to the graph in the image. Tie rule: alphabetical edge label. Output two letters in <answer>...</answer>.",
    "Prim's algorithm grows the MST one edge at a time starting at a given node. For start node {start} and the depicted graph, what is the first edge added? Break ties alphabetically. In <answer>...</answer>.",
    "What edge does Prim's algorithm pick first when started at node {start} on the weighted graph in the image? Alphabetical tie-break. Two-letter edge in <answer>...</answer>.",
    "Trace Prim's algorithm with start node {start} on the depicted weighted graph and output the first edge added. Ties broken alphabetically. Place answer in <answer>...</answer>.",
    "Run Prim's MST algorithm from vertex {start} on the graph shown. What is the first edge added? Alphabetical tie-break. Answer in <answer>...</answer> as two endpoint letters.",
    "The weighted graph in the image is processed by Prim's algorithm starting at {start}. What is the first edge added to the MST? Use alphabetical order on edge labels for ties. Answer in <answer>...</answer>.",
    "Output the first edge selected by Prim's algorithm starting at node {start} on the depicted graph. Tie-break: alphabetical. Answer (two letters) in <answer>...</answer>.",
    "Prim's algorithm from start vertex {start}: what is the first MST edge in the graph in the image? Ties broken alphabetically by edge label. Place answer in <answer>...</answer>.",
    "Begin Prim's algorithm at vertex {start} on the weighted graph shown. Output the first edge added (alphabetical tie-break). Answer in <answer>...</answer> as two letters (e.g. AG).",
]


def _prim_first_edge(
    n: int, edges: List[Tuple[int, int, float]],
    labels: List[str], start: int
) -> str:
    """First edge picked by Prim's algorithm starting at `start`.

    The candidate set initially is every edge incident to `start`. We
    pick the minimum-weight edge; ties broken by alphabetical edge
    label (each label is endpoints in alphabetical order).
    """
    def edge_label(u: int, v: int) -> str:
        a, b = labels[u], labels[v]
        return a + b if a < b else b + a

    candidates = []
    for u, v, w in edges:
        if u == start or v == start:
            other = v if u == start else u
            candidates.append((w, edge_label(u, v), other))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][1]


class PrimFirstEdgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "prim_first_edge"

    shape_strategy = "binary"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "n": 3, "trivial": True}
        if level <= 2:
            n = 4
        elif level <= 4:
            n = 5
        elif level <= 6:
            n = 6
        elif level <= 8:
            n = 7
        else:
            n = 8
        density = 0.30 + 0.05 * level
        tie_prob = 0.0 if level == 0 else min(0.4, 0.05 + 0.04 * level)
        wmax = 5 + level
        return {"level": level, "n": n, "density": min(0.7, density),
                "tie_prob": tie_prob, "wmax": wmax, "trivial": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1373 + level * 71 + 19)

        if cfg.get("trivial"):
            # 3-vertex graph with weights 1, 2, 3 — start at a random vertex,
            # the answer is whichever incident edge has the smaller weight.
            n = 3
            labels = ["A", "B", "C"]
            pairs = [(0, 1), (0, 2), (1, 2)]
            weights = [1, 2, 3]
            rng.shuffle(weights)
            edges: List[Tuple[int, int, float]] = [
                (u, v, w) for (u, v), w in zip(pairs, weights)
            ]
            start = rng.randint(0, 2)
        else:
            n = cfg["n"]
            labels = [chr(ord("A") + i) for i in range(n)]
            edges = self._sample_weighted_graph(rng, n, cfg)
            # Start vertex: any vertex with at least one incident edge.
            incident = sorted({u for (u, v, w) in edges} |
                              {v for (u, v, w) in edges})
            start = rng.choice(incident)

        ans = _prim_first_edge(n, edges, labels, start)
        if not ans:
            return None

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(start=labels[start])
        img = self._render(labels, edges, start, rng)
        return question, ans, img

    def _sample_weighted_graph(
        self, rng: random.Random, n: int, cfg: Dict
    ) -> List[Tuple[int, int, float]]:
        wmax = cfg["wmax"]
        tie_prob = cfg["tie_prob"]
        density = cfg["density"]

        perm = list(range(n))
        rng.shuffle(perm)
        edges_set: Dict[Tuple[int, int], int] = {}
        for i in range(1, n):
            u = perm[i]
            v = perm[rng.randint(0, i - 1)]
            a, b = (u, v) if u < v else (v, u)
            edges_set[(a, b)] = rng.randint(1, wmax)
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edges_set:
                    continue
                if rng.random() < density:
                    edges_set[(u, v)] = rng.randint(1, wmax)
        if tie_prob > 0 and len(edges_set) >= 2 and rng.random() < tie_prob:
            keys = list(edges_set.keys())
            rng.shuffle(keys)
            min_w = min(edges_set.values())
            edges_set[keys[0]] = min_w

        return [(u, v, w) for (u, v), w in edges_set.items()]

    def _render(self, labels, edges, start, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2 + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        for (u, v, w) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#566573",
                    linewidth=1.5, zorder=1)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / d * 0.10, dx / d * 0.10
            wstr = str(int(w)) if float(w).is_integer() else f"{w:.1f}"
            ax.text(mx + ox, my + oy, wstr, ha="center", va="center",
                    fontsize=11, color="#1a3a6e",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="#fdf6e3",
                              edgecolor="#8b8000", linewidth=0.8),
                    zorder=4)
        for i in range(n):
            x, y = pos[i]
            # Highlight start vertex
            if i == start:
                face = "#a8e6a3"
                edge_c = "#1f6f1f"
                txt_c = "#1f6f1f"
            else:
                face = "#b8d4f0"
                edge_c = "#1a3a6e"
                txt_c = "#1a3a6e"
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor=face,
                                    edgecolor=edge_c, linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_c,
                    zorder=6)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Accept the edge label in either order (AB or BA)."""
        import re
        gt = ground_truth.strip().upper()
        pred = predicted.strip().upper()
        m = re.search(r"\b([A-Z])[\s,\-]*([A-Z])\b", pred)
        if not m:
            letters = re.findall(r"[A-Z]", pred)
            if len(letters) < 2:
                return False
            a, b = letters[0], letters[1]
        else:
            a, b = m.group(1), m.group(2)
        if a == b:
            return False
        canon = a + b if a < b else b + a
        return canon == gt
