"""
Prüfer code: given a small labelled tree (5–7 vertices), output the
Prüfer code as a comma-separated bracketed list of vertex labels.

Mirrors qid 258 wording from the source distribution:
  "Give the Prüfer code of the following graph. Please answer in the
   format like '[1, 2, 3, 4, 5]'.  Ans: [3, 2, 3, 3, 6]."

Internally we use letter labels (A, B, C, ...) consistent with the rest
of the graph-theory env family, and the answer format is
`[X, Y, Z, ...]`. The classical Prüfer encoding of an n-vertex tree
produces a sequence of length n-2.
"""
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Give the Prüfer code of the tree shown in the image. Use vertex labels and answer in the format like [A, B, C, D, E] inside <answer>...</answer>.",
    "Compute the Prüfer code of the labeled tree in the image. Output as a bracketed comma-separated list of vertex labels in <answer>...</answer>.",
    "What is the Prüfer code of the tree depicted? Format the answer like [X, Y, Z] in <answer>...</answer>.",
    "Encode the labeled tree in the image as a Prüfer code. Output as [A, B, ...] in <answer>...</answer>.",
    "Output the Prüfer code (sequence of n-2 vertex labels) for the tree in the image. Use the bracketed list format like [A, B, C] in <answer>...</answer>.",
    "From the labeled tree drawn in the image, derive its Prüfer code. Answer in the format [A, B, C, ...] in <answer>...</answer>.",
    "What is the Prüfer encoding of the tree shown? Output as a bracketed comma-separated list of labels in <answer>...</answer>.",
    "Determine the Prüfer code for the tree in the image. Format like [B, C, D] in <answer>...</answer>.",
    "Produce the Prüfer code of the labeled tree depicted. Answer as [A, B, ...] in <answer>...</answer>.",
    "Trace the Prüfer encoding (repeatedly delete the smallest-labeled leaf and record its neighbor) on the tree in the image. Output the resulting code in [A, B, C] form in <answer>...</answer>.",
    "Give the Prüfer code of the displayed tree as a bracketed comma-separated list of vertex labels. Place answer in <answer>...</answer>.",
    "Output the Prüfer code of the tree in the image, formatted like [X, Y, Z], in <answer>...</answer>.",
    "From the depicted labeled tree, output the Prüfer code as [A, B, C, ...] in <answer>...</answer>.",
    "Encode the tree shown using the Prüfer sequence. Format the answer as [A, B, ...] inside <answer>...</answer>.",
    "What is the Prüfer code corresponding to the labeled tree in the image? Output in [A, B, C] format in <answer>...</answer>.",
    "Compute and output the Prüfer code of the tree in the image. Answer as a bracketed comma-separated label list in <answer>...</answer>.",
]


def _prufer_code(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Compute the classical Prüfer code of a labelled tree.

    Algorithm: repeatedly find the smallest-labelled leaf, append its
    neighbour to the code, delete the leaf. Stop when 2 vertices remain.
    """
    deg = [0] * n
    adj: Dict[int, set] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    code: List[int] = []
    remaining = set(range(n))
    for _ in range(n - 2):
        # smallest-label leaf
        leaf = min(v for v in remaining if deg[v] == 1)
        # its single neighbour
        neighbor = next(iter(adj[leaf]))
        code.append(neighbor)
        # remove leaf
        adj[neighbor].remove(leaf)
        adj[leaf].clear()
        deg[neighbor] -= 1
        deg[leaf] = 0
        remaining.remove(leaf)
    return code


class PruferCodeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "prufer_code"

    shape_strategy = "list_mean"
    shape_beta = 2.0

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
            n = 7
        return {"level": level, "n": n, "trivial": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1493 + level * 89 + 17)

        if cfg.get("trivial"):
            # 3-vertex path A - B - C: smallest leaf is A, its neighbour
            # is B → code = [B]. Single-element answer.
            n = 3
            labels = ["A", "B", "C"]
            edges = [(0, 1), (1, 2)]
        else:
            n = cfg["n"]
            labels = [chr(ord("A") + i) for i in range(n)]
            edges = self._sample_random_tree(rng, n)

        code = _prufer_code(n, edges)
        ans_str = "[" + ", ".join(labels[v] for v in code) + "]"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(labels, edges, rng)
        return question, ans_str, img

    def _sample_random_tree(
        self, rng: random.Random, n: int
    ) -> List[Tuple[int, int]]:
        """Build a uniformly random labeled tree by sampling a Prüfer
        sequence (any length n-2 sequence over {0..n-1} maps to a tree)
        and decoding it. This guarantees variety: paths, stars, brooms,
        caterpillars all appear.
        """
        if n == 2:
            return [(0, 1)]
        seq = [rng.randint(0, n - 1) for _ in range(n - 2)]
        # Decode Prüfer → tree.
        deg = [1] * n
        for v in seq:
            deg[v] += 1
        edges: List[Tuple[int, int]] = []
        for v in seq:
            # smallest-label leaf
            for u in range(n):
                if deg[u] == 1:
                    edges.append((u, v) if u < v else (v, u))
                    deg[u] -= 1
                    deg[v] -= 1
                    break
        # Two remaining vertices with degree 1
        last = [u for u in range(n) if deg[u] == 1]
        if len(last) >= 2:
            u, v = last[0], last[1]
            edges.append((u, v) if u < v else (v, u))
        return edges

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Use a layered tree-ish layout. Pick the most central vertex
        # (highest degree) as a root; do BFS to assign levels; for very
        # small trees just use circular layout, which still reads well.
        if n <= 4:
            pos = {}
            for i in range(n):
                angle = 2 * math.pi * i / n - math.pi / 2 + rng.uniform(-0.04, 0.04)
                r = 1.1 + rng.uniform(-0.05, 0.05)
                pos[i] = (r * math.cos(angle), r * math.sin(angle))
        else:
            # Tree layout: pick highest-degree vertex as root.
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            root = max(range(n), key=lambda v: len(adj[v]))
            level = {root: 0}
            order = [root]
            seen = {root}
            head = 0
            while head < len(order):
                u = order[head]
                head += 1
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        level[v] = level[u] + 1
                        order.append(v)
            # Group nodes by level
            by_level = defaultdict(list)
            for v, lv in level.items():
                by_level[lv].append(v)
            max_level = max(by_level.keys())
            pos = {}
            for lv in range(max_level + 1):
                nodes = by_level[lv]
                # spread horizontally
                count = len(nodes)
                for i, v in enumerate(nodes):
                    if count == 1:
                        x = 0.0
                    else:
                        x = (i / (count - 1) - 0.5) * 2.4
                    y = 1.2 - lv * (2.4 / max(max_level, 1))
                    pos[v] = (x + rng.uniform(-0.06, 0.06),
                              y + rng.uniform(-0.04, 0.04))
        # Draw edges
        for (u, v) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#566573",
                    linewidth=1.6, zorder=1)
        # Vertices
        for i in range(n):
            x, y = pos[i]
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor="#ffe0b2",
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
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Parse a bracketed comma-separated label sequence.

        Accepts both letter form (A, B, C) AND integer form (1-indexed:
        A=1, B=2, ...). DynaMath qid 258 uses integer Prüfer codes; we
        normalize predicted tokens against the ground-truth letter set
        by mapping integers→letters before comparison."""
        import re
        gt = ground_truth.strip()
        pred = predicted.strip()
        def parse(s):
            m = re.search(r"\[(.*?)\]", s)
            inner = m.group(1) if m else s
            toks = [t.strip().upper() for t in inner.split(",") if t.strip()]
            return toks
        def to_letter(tok: str) -> str:
            # If integer, treat as 1-indexed → letter A=1, B=2, ...
            if re.fullmatch(r"\d+", tok):
                idx = int(tok) - 1
                if 0 <= idx < 26:
                    return chr(ord("A") + idx)
            return tok
        p_tokens = [to_letter(t) for t in parse(pred)]
        g_tokens = [to_letter(t) for t in parse(gt)]
        if len(p_tokens) != len(g_tokens):
            return False
        return p_tokens == g_tokens
