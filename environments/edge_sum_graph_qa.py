"""
Edge-sum graph puzzle.

A small undirected graph is drawn with integer labels at each vertex and
labels on the edges. Each edge label = sum of the two endpoint vertex
values. One edge label is replaced with `?`. Find the missing edge label.

Reference an external reference:
  "The sum of the numbers at the two vertices at the ends of each edge is
   equal to the number written on that edge. What number should he write
   on the edge marked with the question mark?" Ans: 3.

L0: triangle (3 vertices, 3 edges). All vertices labeled, one edge missing.
Answer = sum of the two endpoints. Trivial 1-step.
L9: 7+ vertices, denser graph, more edges; some vertex labels also missing
(must be solved together with edges).
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "In the graph shown, each edge is labeled with the sum of its two endpoint values. The edge marked '?' has no label. What number belongs on the '?' edge? Place the integer in <answer>...</answer>.",
    "Each edge label equals the sum of the values at its two endpoints. One edge is marked with a '?'. Find the missing edge value. Integer in <answer>...</answer>.",
    "The graph satisfies: edge label = sum of its two endpoint vertex values. Determine the value on the edge labeled '?'. Integer answer in <answer>...</answer>.",
    "Edge-sum graph: every edge label = sum of its endpoint vertices. Find the number that should replace '?' on the marked edge. Place the integer in <answer>...</answer>.",
    "Look at the graph. Each edge label is the sum of the two vertices it connects. Compute the value of the edge marked '?'. Integer in <answer>...</answer>.",
    "In the depicted graph, each edge has a label equal to the sum of its two endpoint values. What integer goes on the edge marked '?'. Place it in <answer>...</answer>.",
    "The numbers on each edge are the sums of the values at the two endpoint vertices. Find the missing edge value (marked '?'). Integer answer in <answer>...</answer>.",
    "Determine the integer on the edge marked '?' given that every edge label equals the sum of its endpoint vertex values. Place answer in <answer>...</answer>.",
    "Edge sums equal vertex pair sums. What number replaces '?' on the marked edge? Integer in <answer>...</answer>.",
    "Each edge of the graph is labeled with the sum of its two endpoint values. What is the value of the missing label '?'. Integer answer in <answer>...</answer>.",
    "The graph has integer-labeled vertices and edges. Each edge label = sum of two endpoint values. Find the missing edge value (marked '?'). Place integer in <answer>...</answer>.",
    "Compute the missing edge label '?' given the rule: edge label = sum of vertex pair. Integer in <answer>...</answer>.",
    "In this graph, edge label = sum of endpoint values. One edge is missing its label, marked '?'. Determine the integer. Place in <answer>...</answer>.",
    "The image shows a graph where every edge label equals the sum of its two endpoint vertex values. What integer goes on the edge with '?'? Integer answer in <answer>...</answer>.",
    "Edge-sum puzzle: edges are labeled with the sums of endpoint pairs. Find the missing label '?'. Integer answer in <answer>...</answer>.",
    "Find the integer on the edge marked '?' (rule: each edge = sum of its two vertex endpoints). Place answer in <answer>...</answer>.",
]


class EdgeSumGraphQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "edge_sum_graph"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n = 3
            extra_edges = 0
            n_hidden_vertices = 0
        elif level <= 2:
            n = 4
            extra_edges = 1
            n_hidden_vertices = 0
        elif level <= 4:
            n = 5
            extra_edges = 2
            n_hidden_vertices = 0
        elif level <= 6:
            n = 6
            extra_edges = 3
            n_hidden_vertices = 1
        else:
            n = 7
            extra_edges = 4
            n_hidden_vertices = 2
        return {"level": level, "n": n, "extra_edges": extra_edges,
                "n_hidden_vertices": n_hidden_vertices}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 1879 + level * 71 + 23)

        # Vertex values: small distinct positives
        hi = 9 if level <= 4 else 15
        pool = list(range(1, hi + 1))
        rng.shuffle(pool)
        vertex_values = pool[:n]

        # Build graph: spanning ring + extra random edges
        edges = []
        # Ring through vertices in random order
        order = list(range(n))
        rng.shuffle(order)
        for i in range(n):
            u = order[i]
            v = order[(i + 1) % n]
            edges.append(tuple(sorted((u, v))))
        # Add extra edges
        possible = [(i, j) for i in range(n) for j in range(i + 1, n)
                    if (i, j) not in edges]
        rng.shuffle(possible)
        for (i, j) in possible[:cfg["extra_edges"]]:
            edges.append((i, j))
        # Dedup
        edges = list(set(edges))

        # Compute edge labels
        edge_labels = {(u, v): vertex_values[u] + vertex_values[v]
                       for (u, v) in edges}

        # Pick one edge to hide.
        hidden_edge = rng.choice(edges)
        # Optionally hide some vertex values, but ensure the hidden edge is
        # still uniquely determinable from the remaining info.
        n_hide_v = cfg["n_hidden_vertices"]
        hidden_vertices = set()
        if n_hide_v > 0:
            # We need: vertex values can be solved from edge_labels - {hidden_edge}
            # Try several random hidings.
            for _attempt in range(20):
                cand = list(range(n))
                rng.shuffle(cand)
                # Don't hide BOTH endpoints of the hidden_edge — that makes
                # the hidden edge under-determined.
                cand = [c for c in cand if c not in hidden_edge or
                        rng.random() < 0.3]
                hv = set(cand[:n_hide_v])
                # The two endpoints of hidden_edge must NOT both be hidden.
                if all(e in hv for e in hidden_edge):
                    continue
                # Check: with edge_labels for known edges, can we solve for
                # all hidden vertex values uniquely?
                if self._can_solve(n, edges, edge_labels, hidden_edge, hv,
                                   vertex_values):
                    hidden_vertices = hv
                    break
            if not hidden_vertices and n_hide_v > 0:
                # Fall back to no-hidden if can't solve
                hidden_vertices = set()

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer = str(edge_labels[hidden_edge])

        labels = [chr(ord("A") + i) for i in range(n)]
        img = self._render(n, edges, vertex_values, labels, edge_labels,
                           hidden_edge, hidden_vertices, rng)
        return question, answer, img

    def _can_solve(self, n: int, edges: List[Tuple[int, int]],
                   edge_labels: Dict, hidden_edge: Tuple[int, int],
                   hidden_vertices: set, true_vertex_values: List[int]) -> bool:
        """Can we recover hidden vertex values uniquely from the visible info?
        Use Gaussian elimination on the linear system over the known edges.
        """
        # Known edges (excluding hidden_edge)
        # Variables: hidden vertex indices. Equations: for each known edge
        # (u, v), x_u + x_v = label - sum(known endpoints).
        var_idx = {v: i for i, v in enumerate(sorted(hidden_vertices))}
        k = len(var_idx)
        if k == 0:
            return True
        # Build A and b
        rows = []
        for (u, v), lab in edge_labels.items():
            if (u, v) == hidden_edge:
                continue
            row = [0.0] * k
            rhs = lab
            if u in var_idx:
                row[var_idx[u]] += 1
            else:
                rhs -= true_vertex_values[u]
            if v in var_idx:
                row[var_idx[v]] += 1
            else:
                rhs -= true_vertex_values[v]
            rows.append((row, rhs))
        # Augmented matrix
        A = [r + [rhs] for r, rhs in rows]
        # Row reduce
        m = len(A)
        col = 0
        r = 0
        while r < m and col < k:
            # find pivot
            piv = -1
            for rr in range(r, m):
                if abs(A[rr][col]) > 1e-9:
                    piv = rr
                    break
            if piv < 0:
                col += 1
                continue
            A[r], A[piv] = A[piv], A[r]
            for rr in range(m):
                if rr == r or abs(A[rr][col]) < 1e-9:
                    continue
                f = A[rr][col] / A[r][col]
                for cc in range(col, k + 1):
                    A[rr][cc] -= f * A[r][cc]
            r += 1
            col += 1
        # Check rank == k (unique solution)
        rank = 0
        for row in A:
            if any(abs(row[c]) > 1e-9 for c in range(k)):
                rank += 1
        return rank >= k

    def _render(self, n: int, edges: List[Tuple[int, int]],
                vertex_values: List[int], labels: List[str],
                edge_labels: Dict, hidden_edge: Tuple[int, int],
                hidden_vertices: set, rng: random.Random) -> Image.Image:
        bg = "#ffffff"
        edge_color = "#1d3557"
        node_color_known = "#dbe9f7"
        node_color_unknown = "#fde2e4"

        # Place vertices on a circle
        positions = {}
        for i in range(n):
            ang = 2 * math.pi * i / n + math.pi / 2
            positions[i] = (math.cos(ang), math.sin(ang))

        fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Edges
        for (u, v), lab in edge_labels.items():
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=1.6,
                    zorder=1)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # offset perpendicular to edge for label placement
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy) or 1
            px, py = -dy / d * 0.12, dx / d * 0.12
            text = "?" if (u, v) == hidden_edge else str(lab)
            color = "#b00020" if (u, v) == hidden_edge else "#0d4f3c"
            ax.text(mx + px, my + py, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color,
                    bbox=dict(facecolor="#ffffff", edgecolor="none",
                              boxstyle="round,pad=0.15"),
                    zorder=4)

        # Vertices
        for i in range(n):
            x, y = positions[i]
            is_hidden = i in hidden_vertices
            fc = node_color_unknown if is_hidden else node_color_known
            circ = plt.Circle((x, y), 0.18, facecolor=fc,
                              edgecolor=edge_color, linewidth=1.6, zorder=3)
            ax.add_patch(circ)
            txt = labels[i] if is_hidden else str(vertex_values[i])
            color = "#b00020" if is_hidden else "#1a1a1a"
            ax.text(x, y, txt, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color, zorder=5)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Each edge label = sum of its two endpoint values",
                     fontsize=11, pad=6)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg,
                    dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
