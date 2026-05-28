"""
Kruskal's algorithm first edge: given a small undirected weighted graph,
output the first edge added to the minimum spanning tree by Kruskal's
algorithm. Ties are broken in alphabetical order of the edge label,
where each edge label is its two endpoint letters in alphabetical
order (e.g. AS comes before BT, AE comes before AS).

Mirrors an external reference wording from the source distribution:
  "what is the first edge added to the MST when running Kruskal's
   Algorithm? In the case of a tie, choose the edge which comes first
   in alphabetical order i.e. if you had to choose between AS and AE,
   then you would choose AE first."

Output format: two letters concatenated (e.g. AB or BS). Accepts the
endpoints in either order at verify time.
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
    "The image shows an undirected weighted graph. What is the first edge added to the MST when running Kruskal's Algorithm? In the case of a tie, choose the edge which comes first in alphabetical order, i.e. if you had to choose between AS and AE, then you would choose AE first. Output the edge as two letters (e.g. AB) in <answer>...</answer>.",
    "Run Kruskal's algorithm on the weighted graph in the image. What is the first edge added to the minimum spanning tree? Break ties alphabetically by edge label (AE before AS). Answer in <answer>...</answer> as a two-letter edge name.",
    "Given the weighted graph shown, output the first edge that Kruskal's algorithm picks for the MST. Ties: alphabetical order on edge labels. Answer like AB in <answer>...</answer>.",
    "What is the first edge added to the MST by Kruskal's Algorithm on the depicted weighted graph? Tie-break: alphabetical order (AD comes before AG). Place the two-letter edge label in <answer>...</answer>.",
    "Inspect the weighted graph in the image. Identify the first edge Kruskal's algorithm adds to the spanning tree (lowest weight, ties broken alphabetically). Output the edge in <answer>...</answer> as two letters.",
    "Apply Kruskal's algorithm to the graph shown. What is the very first edge picked? Use alphabetical order to break weight ties. Answer in <answer>...</answer> as two endpoint letters.",
    "From the weighted graph in the image, output the first edge selected by Kruskal's algorithm. Ties broken in alphabetical order of the edge label. Place answer in <answer>...</answer>.",
    "Find the first edge added to the MST by Kruskal's algorithm on the weighted graph depicted. Alphabetical tie-break. Output the two-letter edge label in <answer>...</answer>.",
    "Kruskal's algorithm starts by adding the minimum-weight edge. For the graph in the image, which edge is that? Break ties alphabetically. Answer in <answer>...</answer>.",
    "What edge does Kruskal's algorithm add first when run on the depicted weighted graph? In case of equal weights, the edge whose label comes first alphabetically wins. Answer (two-letter edge) in <answer>...</answer>.",
    "Trace Kruskal's algorithm on the weighted graph in the image and output the first MST edge. Tie-break: alphabetical edge label (so AE beats AS). Place answer in <answer>...</answer>.",
    "Output the first edge that Kruskal's algorithm chooses when constructing the MST of the graph shown. Ties broken alphabetically. Answer in <answer>...</answer> as two letters.",
    "The weighted graph in the image is processed by Kruskal's algorithm. What is the first edge picked? Use alphabetical order on edge labels for tie-breaking. Place answer in <answer>...</answer>.",
    "From the depicted graph, list the first edge added to the MST under Kruskal's algorithm. Tie rule: alphabetical (AD before AG). Output two letters in <answer>...</answer>.",
    "Kruskal's algorithm sorts edges by weight and picks the smallest first (ties broken by alphabetical edge label). For the graph in the image, what is that first edge? Place answer in <answer>...</answer>.",
    "Run Kruskal's MST algorithm on the weighted graph in the image. Output the first edge added (alphabetical tie-break). Answer in <answer>...</answer> as two endpoint letters (e.g. BS).",
]


def _kruskal_first_edge(
    n: int, edges: List[Tuple[int, int, float]], labels: List[str]
) -> str:
    """Return the first edge picked by Kruskal's algorithm.

    Edges are sorted by (weight, alphabetical edge label). The label of
    edge (u, v) is the two endpoint letters in alphabetical order.
    """
    def edge_label(u: int, v: int) -> str:
        a, b = labels[u], labels[v]
        return a + b if a < b else b + a

    keyed = []
    for u, v, w in edges:
        keyed.append((w, edge_label(u, v), u, v))
    keyed.sort()
    w, lbl, u, v = keyed[0]
    return lbl


class KruskalFirstEdgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "kruskal_first_edge"

    # Integer-like answers, but as 2-letter strings → keep binary.
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
        # density of extra edges past spanning tree
        density = 0.25 + 0.05 * level
        # When level is high, allow ties in weights more often.
        tie_prob = 0.0 if level == 0 else min(0.4, 0.05 + 0.04 * level)
        # Weight range
        wmax = 5 + level
        return {"level": level, "n": n, "density": min(0.7, density),
                "tie_prob": tie_prob, "wmax": wmax, "trivial": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1289 + level * 79 + 11)

        if cfg.get("trivial"):
            # 3-vertex graph with weights 1, 2, 3 — first edge is the
            # weight-1 edge unambiguously. Randomize which pair gets which
            # weight so the answer varies but is deterministic per seed.
            n = 3
            labels = ["A", "B", "C"]
            pairs = [(0, 1), (0, 2), (1, 2)]
            weights = [1, 2, 3]
            rng.shuffle(weights)
            edges: List[Tuple[int, int, float]] = [
                (u, v, w) for (u, v), w in zip(pairs, weights)
            ]
        else:
            n = cfg["n"]
            labels = [chr(ord("A") + i) for i in range(n)]
            edges = self._sample_weighted_graph(rng, n, cfg)

        ans = _kruskal_first_edge(n, edges, labels)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(labels, edges, rng)
        return question, ans, img

    def _sample_weighted_graph(
        self, rng: random.Random, n: int, cfg: Dict
    ) -> List[Tuple[int, int, float]]:
        wmax = cfg["wmax"]
        tie_prob = cfg["tie_prob"]
        density = cfg["density"]

        # Build a random spanning tree first to guarantee connectedness.
        perm = list(range(n))
        rng.shuffle(perm)
        edges_set: Dict[Tuple[int, int], int] = {}
        # pool of weights to allow tying
        for i in range(1, n):
            u = perm[i]
            v = perm[rng.randint(0, i - 1)]
            a, b = (u, v) if u < v else (v, u)
            edges_set[(a, b)] = rng.randint(1, wmax)
        # Add extra edges
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edges_set:
                    continue
                if rng.random() < density:
                    edges_set[(u, v)] = rng.randint(1, wmax)
        # Inject ties at the smallest weight to make the alphabetical
        # tie-break actually meaningful at higher levels.
        if tie_prob > 0 and len(edges_set) >= 2 and rng.random() < tie_prob:
            keys = list(edges_set.keys())
            rng.shuffle(keys)
            min_w = min(edges_set.values())
            # pick a random edge and demote its weight to min_w
            target = keys[0]
            edges_set[target] = min_w

        return [(u, v, w) for (u, v), w in edges_set.items()]

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Place vertices on a circle.
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2 + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        # Edges + weight labels
        for (u, v, w) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#566573",
                    linewidth=1.5, zorder=1)
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            # offset perpendicular to edge so the label doesn't sit on
            # the line
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
        # Vertices
        for i in range(n):
            x, y = pos[i]
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor="#b8d4f0",
                                    edgecolor="#1a3a6e", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#1a3a6e",
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
        """Override: accept the edge label in either order (AB or BA)."""
        import re
        gt = ground_truth.strip().upper()
        pred = predicted.strip().upper()
        # Pull two letters from the prediction; first occurrence wins.
        # Allow surrounding parens / dashes / commas / spaces.
        m = re.search(r"\b([A-Z])[\s,\-]*([A-Z])\b", pred)
        if not m:
            # fall back: any two letters in the string
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
