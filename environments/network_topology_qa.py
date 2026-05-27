"""Network topology QA — count/property questions over a drawn graph.

Fixes:
  - L0 and L9 no longer pixel-identical: layout, node count, and question type diverge.
  - Diverse layouts (circular, grid, random, star, chain, bipartite).
  - Diverse node shapes (circle, square, diamond, hexagon, triangle).
  - Diverse edge styles (straight, curved, bold).
  - Expanded question pool (8+ question types, scaling with level).
  - Sub-RNG seeded on (seed, level) so structural changes across level.
"""
import random
import math
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon, FancyArrowPatch
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ----------------------------------------------------------------------
# Layout generators
# ----------------------------------------------------------------------
def _circular_layout(n: int, sub: random.Random,
                     cx: float = 3.0, cy: float = 3.0,
                     r: float = 2.0) -> List[Tuple[float, float]]:
    phase = sub.uniform(0, 2 * math.pi)
    pos = []
    for i in range(n):
        theta = 2 * math.pi * i / n + phase
        pos.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return pos

def _grid_layout(n: int, sub: random.Random) -> List[Tuple[float, float]]:
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cell_w = 5.0 / max(1, cols - 1) if cols > 1 else 0
    cell_h = 5.0 / max(1, rows - 1) if rows > 1 else 0
    jitter = sub.uniform(0.0, 0.25)
    pos = []
    for i in range(n):
        r = i // cols
        c = i % cols
        x = 0.5 + c * cell_w + sub.uniform(-jitter, jitter)
        y = 0.5 + r * cell_h + sub.uniform(-jitter, jitter)
        pos.append((x, y))
    return pos

def _random_layout(n: int, sub: random.Random,
                   min_dist: float = 0.9) -> List[Tuple[float, float]]:
    pos = []
    for _ in range(200):
        if len(pos) == n:
            break
        cand = (sub.uniform(0.5, 5.5), sub.uniform(0.5, 5.5))
        if all(math.hypot(cand[0] - p[0], cand[1] - p[1]) >= min_dist
               for p in pos):
            pos.append(cand)
    while len(pos) < n:
        # Fallback
        pos.append((sub.uniform(0.5, 5.5), sub.uniform(0.5, 5.5)))
    return pos

def _star_layout(n: int, sub: random.Random) -> List[Tuple[float, float]]:
    # Node 0 is centre, rest on a circle around it.
    center = (3.0, 3.0)
    r = 2.0
    pos = [center]
    phase = sub.uniform(0, 2 * math.pi)
    for i in range(1, n):
        theta = 2 * math.pi * (i - 1) / max(1, (n - 1)) + phase
        pos.append((center[0] + r * math.cos(theta),
                    center[1] + r * math.sin(theta)))
    return pos

def _chain_layout(n: int, sub: random.Random) -> List[Tuple[float, float]]:
    # Left to right chain
    xs = np.linspace(0.7, 5.3, n)
    y_base = sub.uniform(2.5, 3.5)
    pos = [(float(x), y_base + sub.uniform(-0.3, 0.3)) for x in xs]
    return pos

def _bipartite_layout(n: int, sub: random.Random) -> List[Tuple[float, float]]:
    split = n // 2
    # Evenly-spaced x positions along each row so nodes never overlap.
    top_xs = list(np.linspace(0.7, 5.3, max(split, 1)))
    bot_xs = list(np.linspace(0.7, 5.3, max(n - split, 1)))
    top = [(float(x), 5.0 + sub.uniform(-0.15, 0.15)) for x in top_xs]
    bot = [(float(x), 1.0 + sub.uniform(-0.15, 0.15)) for x in bot_xs]
    return top + bot

# ----------------------------------------------------------------------
# Graph structure generators
# ----------------------------------------------------------------------
def _random_edges(n: int, sub: random.Random,
                  density: float = 0.4) -> set:
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sub.random() < density:
                edges.add((i, j))
    # Ensure connected-ish: add a spanning path
    perm = list(range(n))
    sub.shuffle(perm)
    for k in range(n - 1):
        a, b = perm[k], perm[k + 1]
        edges.add((min(a, b), max(a, b)))
    return edges

def _star_edges(n: int) -> set:
    """Edges for star graph: node 0 connects to all others."""
    return {(0, i) for i in range(1, n)}

def _chain_edges(n: int) -> set:
    return {(i, i + 1) for i in range(n - 1)}

def _cycle_edges(n: int) -> set:
    edges = {(i, (i + 1) % n) for i in range(n)}
    # Normalize order
    return {(min(a, b), max(a, b)) for a, b in edges}

def _bipartite_edges(n: int, sub: random.Random) -> set:
    split = n // 2
    left = list(range(split))
    right = list(range(split, n))
    edges = set()
    # Each right node connects to at least 1 left
    for rn in right:
        k = sub.randint(1, max(1, len(left)))
        for ln in sub.sample(left, k):
            edges.add((min(ln, rn), max(ln, rn)))
    # And each left connects to at least 1 right
    for ln in left:
        if not any(ln in e for e in edges):
            rn = sub.choice(right)
            edges.add((min(ln, rn), max(ln, rn)))
    return edges

# ----------------------------------------------------------------------
# Node shape drawing
# ----------------------------------------------------------------------
def _draw_node(ax, x, y, shape, color, size, label, label_color="white",
               font_family="DejaVu Sans", font_size=10, edgecolor="black"):
    if shape == "circle":
        p = plt.Circle((x, y), size, facecolor=color,
                       edgecolor=edgecolor, linewidth=1.5, zorder=5)
        ax.add_patch(p)
    elif shape == "square":
        p = mpatches.Rectangle((x - size, y - size), 2 * size, 2 * size,
                               facecolor=color, edgecolor=edgecolor,
                               linewidth=1.5, zorder=5)
        ax.add_patch(p)
    elif shape == "diamond":
        p = RegularPolygon((x, y), numVertices=4, radius=size * 1.15,
                           orientation=math.pi / 4, facecolor=color,
                           edgecolor=edgecolor, linewidth=1.5, zorder=5)
        ax.add_patch(p)
    elif shape == "hexagon":
        p = RegularPolygon((x, y), numVertices=6, radius=size * 1.1,
                           facecolor=color, edgecolor=edgecolor,
                           linewidth=1.5, zorder=5)
        ax.add_patch(p)
    elif shape == "triangle":
        p = RegularPolygon((x, y), numVertices=3, radius=size * 1.15,
                           orientation=math.pi / 2,
                           facecolor=color, edgecolor=edgecolor,
                           linewidth=1.5, zorder=5)
        ax.add_patch(p)
    else:
        p = plt.Circle((x, y), size, facecolor=color,
                       edgecolor=edgecolor, linewidth=1.5, zorder=5)
        ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=font_size, fontweight="bold",
            color=label_color, fontfamily=font_family, zorder=6)

class NetworkTopologyQA(StandaloneVisualEnv):
    ENV_NAME = "network_topology"

    _QUESTION_TEMPLATES = {
        "count_nodes": [
            "How many nodes are in this network?",
            "Count the total number of nodes (vertices) in the graph shown.",
            "How many vertices does this network have?",
        ],
        "count_edges": [
            "How many connections (edges) are in this network?",
            "Count the total number of edges in the graph shown.",
            "How many connections are drawn in this network?",
        ],
        "max_degree": [
            "What is the maximum degree (most connections) of any node?",
            "Which number equals the degree of the most-connected node?",
            "Find the highest degree among all nodes in the graph.",
        ],
        "min_degree": [
            "What is the minimum degree among the nodes?",
            "Find the smallest number of edges meeting at any node.",
        ],
        "count_isolated": [
            "How many nodes have degree 0 (no connections)?",
            "Count the isolated nodes in the graph.",
        ],
        "count_leaf": [
            "How many leaf nodes (degree 1) does this graph have?",
            "Count the nodes that are connected to exactly one other node.",
        ],
        "sum_degrees": [
            "What is the sum of degrees over all nodes?",
            "Compute the total degree (sum over nodes).",
        ],
        "triangle_count": [
            "How many triangles (3-cycles) are in this graph?",
            "Count the number of triangles formed by the edges.",
        ],
    }

    _LAYOUT_GEN = {
        "circular": (_circular_layout, _random_edges),
        "grid": (_grid_layout, _random_edges),
        "random": (_random_layout, _random_edges),
        "star": (_star_layout, _star_edges),
        "chain": (_chain_layout, _chain_edges),
        "cycle": (_circular_layout, _cycle_edges),
        "bipartite": (_bipartite_layout, _bipartite_edges),
    }

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # L0-L1 : count_nodes  (trivial, use chain/star/cycle)
        # L2-L3 : count_edges, count_leaf
        # L4-L5 : max_degree, min_degree, count_isolated
        # L6-L7 : sum_degrees, max_degree
        # L8-L9 : triangle_count, max_degree on dense graphs
        configs = {
            0: {"qtypes": ["count_nodes"],
                "layouts": ["chain", "star"],
                "n_range": (3, 5), "density": 0.0},
            1: {"qtypes": ["count_nodes"],
                "layouts": ["star", "cycle", "chain"],
                "n_range": (4, 6), "density": 0.0},
            2: {"qtypes": ["count_edges", "count_leaf"],
                "layouts": ["circular", "star", "chain"],
                "n_range": (4, 6), "density": 0.25},
            3: {"qtypes": ["count_edges", "count_leaf"],
                "layouts": ["circular", "grid", "bipartite"],
                "n_range": (5, 7), "density": 0.3},
            4: {"qtypes": ["max_degree", "count_isolated"],
                "layouts": ["circular", "random", "grid"],
                "n_range": (5, 7), "density": 0.3},
            5: {"qtypes": ["max_degree", "min_degree"],
                "layouts": ["random", "bipartite", "circular"],
                "n_range": (5, 8), "density": 0.35},
            6: {"qtypes": ["sum_degrees", "max_degree"],
                "layouts": ["random", "bipartite"],
                "n_range": (6, 8), "density": 0.4},
            7: {"qtypes": ["sum_degrees", "max_degree", "min_degree"],
                "layouts": ["random", "grid"],
                "n_range": (6, 9), "density": 0.45},
            8: {"qtypes": ["triangle_count", "max_degree"],
                "layouts": ["circular", "circular"],
                "n_range": (5, 6), "density": 0.5},
            9: {"qtypes": ["max_degree", "sum_degrees"],
                "layouts": ["circular"],
                "n_range": (6, 7), "density": 0.45},
        }
        return configs[level]

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Sub-RNG seeded by (seed, level) so different levels look different
        sub = random.Random((self.seed or 0) * 1000 + level * 37 + 2053)
        style = self._random_style()

        n = sub.randint(*cfg["n_range"])
        layout_name = sub.choice(cfg["layouts"])
        layout_fn, edge_fn = self._LAYOUT_GEN[layout_name]

        pos = layout_fn(n, sub)
        if edge_fn is _random_edges:
            edges = edge_fn(n, sub, density=cfg["density"])
        elif edge_fn is _bipartite_edges:
            edges = edge_fn(n, sub)
        else:
            edges = edge_fn(n)

        # Compute graph stats
        degrees = [0] * n
        for i, j in edges:
            degrees[i] += 1
            degrees[j] += 1

        max_d = max(degrees) if degrees else 0
        min_d = min(degrees) if degrees else 0
        n_isolated = sum(1 for d in degrees if d == 0)
        n_leaf = sum(1 for d in degrees if d == 1)
        sum_deg = sum(degrees)

        # Triangle count
        tri = 0
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j); adj[j].add(i)
        for a in range(n):
            for b in adj[a]:
                if b > a:
                    for c in adj[b]:
                        if c > b and c in adj[a]:
                            tri += 1

        # ---------------- Rendering ----------------
        sc = style.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        ax.set_aspect("equal"); ax.axis("off")

        palette = list(style["palette"])
        sub.shuffle(palette)
        edge_color = sub.choice([style["geo_line_color"], "#7f8c8d", "#34495e",
                                 "#2c3e50", "#636e72", "#555"])
        edge_style = sub.choice(["straight", "straight", "curved"])
        edge_lw = sub.uniform(1.2, 2.3)
        edge_alpha = sub.uniform(0.7, 1.0)
        node_shape = sub.choice(["circle", "circle", "square",
                                 "diamond", "hexagon", "triangle"])
        node_size = sub.uniform(0.22, 0.34)
        use_letters = sub.random() < 0.5
        label_offset = sub.choice([0.0, 0.0])
        # Node labels: A-Z or N0,N1,... or 1,2,3
        if use_letters:
            labels = [chr(ord("A") + i) for i in range(n)]
        else:
            mode = sub.choice(["N{i}", "{i}", "v{i}"])
            labels = [mode.format(i=i) for i in range(n)]

        # Edges
        for i, j in edges:
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            if edge_style == "curved":
                rad = sub.uniform(-0.2, 0.2)
                arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-", color=edge_color,
                    linewidth=edge_lw, alpha=edge_alpha, zorder=1)
                ax.add_patch(arrow)
            else:
                ax.plot([x1, x2], [y1, y2], "-",
                        color=edge_color, linewidth=edge_lw,
                        alpha=edge_alpha, zorder=1)

        # Nodes
        for i in range(n):
            x, y = pos[i]
            color = palette[i % len(palette)]
            label_color = "white" if _is_dark(color) else "#1a1a1a"
            _draw_node(ax, x, y, node_shape, color, node_size,
                       labels[i], label_color=label_color,
                       font_family=style["font_family"],
                       font_size=style["font_size_base"] - 1,
                       edgecolor=style["geo_line_color"])

        titles = ["Network Topology", "Graph", "Connectivity Diagram",
                  "Network", "Nodes and Edges", "Graph Structure"]
        ax.set_title(sub.choice(titles),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold",
                     fontfamily=style["font_family"])

        img = self.fig_to_pil(fig, dpi=style["dpi"])

        # ---------------- Question ----------------
        qtype = sub.choice(cfg["qtypes"])
        q = sub.choice(self._QUESTION_TEMPLATES[qtype])

        if qtype == "count_nodes":
            ans = n
        elif qtype == "count_edges":
            ans = len(edges)
        elif qtype == "max_degree":
            ans = max_d
        elif qtype == "min_degree":
            ans = min_d
        elif qtype == "count_isolated":
            ans = n_isolated
        elif qtype == "count_leaf":
            ans = n_leaf
        elif qtype == "sum_degrees":
            ans = sum_deg
        elif qtype == "triangle_count":
            ans = tri
        else:
            ans = len(edges)

        return q, str(ans), img

def _is_dark(hex_color: str) -> bool:
    try:
        s = hex_color.lstrip("#")
        if len(s) != 6:
            return False
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return (0.299 * r + 0.587 * g + 0.114 * b) < 140
    except Exception:
        return False

if __name__ == "__main__":
    env = NetworkTopologyQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, a={env._answer}, q={env._question[:50]}")
