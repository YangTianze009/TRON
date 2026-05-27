"""
Map Distance QA environment.

Capabilities: V3 (chart extraction), V2 (label reading), R1 (arithmetic), R5 (multi-step)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: Grid map with 2 points, ask Manhattan distance.
L1: 3-node graph, ask "direct distance from A to B" (read off label).
L2: 3-node triangle, ask "shortest path from A to B".
L3: 4-node graph, ask "shortest path from A to B".
L4: 4-node graph, ask "edge count".
L5: 5-node graph, ask "shortest path".
L6: 5-node graph, ask "farthest from A".
L7: 6-node graph, ask "shortest path".
L8: 7-node graph, ask "total route through 3 waypoints".
L9: 8-node graph, ask "total route through 3 waypoints".

parameter = {"level": int in [0, 9]}
"""
import math
import random
from collections import defaultdict
import heapq
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PLACE_NAMES = [
    "Town", "Village", "Market", "Harbor", "School", "Park",
    "Station", "Library", "Hospital", "Airport", "Mall", "Fort",
]

_TITLE_VARIANTS = [
    "Map with Distances",
    "Road Network",
    "City Map",
    "Travel Routes",
]

def _dijkstra(adj, start, end=None):
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    if end is not None:
        return dist.get(end, -1)
    return dist

class MapDistanceQA(StandaloneVisualEnv):
    ENV_NAME = "map_distance"

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(15):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            return {"mode": "grid_manhattan"}
        if level == 1:
            return {"mode": "graph", "n": 3, "qtype": "direct_distance"}
        if level == 2:
            return {"mode": "graph", "n": 3, "qtype": "shortest_path"}
        if level == 3:
            return {"mode": "graph", "n": 4, "qtype": "shortest_path"}
        if level == 4:
            return {"mode": "graph", "n": 4, "qtype": "edge_count"}
        if level == 5:
            return {"mode": "graph", "n": 5, "qtype": "shortest_path"}
        if level == 6:
            return {"mode": "graph", "n": 5, "qtype": "farthest_from"}
        if level == 7:
            return {"mode": "graph", "n": 6, "qtype": "shortest_path"}
        if level == 8:
            return {"mode": "graph", "n": 7, "qtype": "total_route"}
        return {"mode": "graph", "n": 8, "qtype": "total_route"}

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)

        if cfg["mode"] == "grid_manhattan":
            return self._grid_manhattan(rng)
        return self._graph_problem(rng, cfg)

    def _grid_manhattan(self, rng):
        # Grid 5x5 with 2 marked points
        grid_size = rng.choice([5, 6])
        x1 = rng.randint(0, grid_size - 1)
        y1 = rng.randint(0, grid_size - 1)
        x2 = rng.randint(0, grid_size - 1)
        y2 = rng.randint(0, grid_size - 1)
        if (x1, y1) == (x2, y2):
            x2 = (x2 + 1) % grid_size
        manhattan = abs(x1 - x2) + abs(y1 - y2)

        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * s, 7 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, grid_size + 0.5)
        ax.set_ylim(-0.5, grid_size + 0.5)

        # Grid lines
        for i in range(grid_size + 1):
            ax.axhline(i, color="#bbb", lw=0.8, zorder=1)
            ax.axvline(i, color="#bbb", lw=0.8, zorder=1)

        # Coordinate labels on axes
        for i in range(grid_size):
            ax.text(i + 0.5, -0.3, str(i), ha="center", fontsize=10)
            ax.text(-0.3, i + 0.5, str(i), va="center", fontsize=10)

        palette = list(style["palette"])
        rng.shuffle(palette)
        c1 = palette[0]
        c2 = palette[1] if len(palette) > 1 else palette[0]
        ax.scatter([x1 + 0.5], [y1 + 0.5], s=400, c=c1, edgecolors="black",
                   linewidths=2, zorder=5)
        ax.scatter([x2 + 0.5], [y2 + 0.5], s=400, c=c2, edgecolors="black",
                   linewidths=2, zorder=5)
        ax.text(x1 + 0.5, y1 + 0.5, "A", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white", zorder=6)
        ax.text(x2 + 0.5, y2 + 0.5, "B", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white", zorder=6)
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        question = (f"On the grid, point A is at ({x1}, {y1}) and point B is at "
                    f"({x2}, {y2}). What is the Manhattan (taxi-cab) distance "
                    f"between A and B? Answer with a single integer.")
        return question, str(manhattan), self.fig_to_pil(fig, dpi=style["dpi"])

    def _build_graph(self, rng, n_nodes):
        names = rng.sample(_PLACE_NAMES, n_nodes)
        adj = defaultdict(list)
        edges = []
        perm = list(range(n_nodes))
        rng.shuffle(perm)
        for i in range(n_nodes - 1):
            u, v = perm[i], perm[i + 1]
            w = rng.randint(2, 20)
            adj[names[u]].append((names[v], w))
            adj[names[v]].append((names[u], w))
            edges.append((names[u], names[v], w))

        for _ in range(rng.randint(0, max(1, n_nodes // 3))):
            u, v = rng.sample(range(n_nodes), 2)
            existing = {(e[0], e[1]) for e in edges} | {(e[1], e[0]) for e in edges}
            if (names[u], names[v]) not in existing:
                w = rng.randint(2, 20)
                adj[names[u]].append((names[v], w))
                adj[names[v]].append((names[u], w))
                edges.append((names[u], names[v], w))

        positions = {}
        for i, name in enumerate(names):
            angle = 2 * math.pi * i / n_nodes + rng.uniform(-0.2, 0.2)
            r = 2.5 + rng.uniform(-0.4, 0.4)
            positions[name] = (4 + r * math.cos(angle), 4 + r * math.sin(angle))
        return names, adj, edges, positions

    def _graph_problem(self, rng, cfg):
        names, adj, edges, positions = self._build_graph(rng, cfg["n"])
        img = self._render(rng, names, edges, positions)
        qtype = cfg["qtype"]

        if qtype == "shortest_path":
            src, dst = rng.sample(names, 2)
            d = _dijkstra(adj, src, dst)
            return (f"What is the shortest distance from {src} to {dst}? "
                    f"Answer with a single integer.", str(d), img)
        if qtype == "direct_distance":
            e = rng.choice(edges)
            return (f"What is the direct (single-edge) distance between {e[0]} "
                    f"and {e[1]}? Answer with a single integer.",
                    str(e[2]), img)
        if qtype == "edge_count":
            return ("How many roads (edges) are shown on the map? "
                    "Answer with a single integer.", str(len(edges)), img)
        if qtype == "farthest_from":
            src = rng.choice(names)
            dists = _dijkstra(adj, src)
            farthest = max(dists, key=dists.get)
            return (f"Which location is farthest from {src} (by shortest path)? "
                    f"Answer with the location name.", farthest, img)
        if qtype == "total_route":
            route = rng.sample(names, min(3, len(names)))
            total = 0
            for i in range(len(route) - 1):
                d = _dijkstra(adj, route[i], route[i + 1])
                if d < 0:
                    return None
                total += d
            return (f"What is the total shortest distance for the route "
                    f"{' -> '.join(route)}? Answer with a single integer.",
                    str(total), img)
        return None

    def _render(self, rng, names, edges, positions):
        style = self._random_style()
        s = style["figsize_scale"]
        # Enlarge figure at higher edge density to avoid label clutter.
        size_boost = 1.0 + min(0.6, max(0, len(edges) - 5) * 0.12)
        fig, ax = plt.subplots(figsize=(9 * s * size_boost, 9 * s * size_boost))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        palette = list(style["palette"])
        rng.shuffle(palette)
        fs = style["font_size_base"]
        ff = style["font_family"]
        lw = style["line_width"]

        # Stagger label positions along each edge to reduce overlap with
        # labels on crossing edges. We distribute labels along t in [0.3, 0.7].
        n_edges = len(edges)
        for idx, (u, v, w) in enumerate(edges):
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            ax.plot([x1, x2], [y1, y2], color=style["geo_line_color"],
                    linewidth=lw, zorder=1, alpha=0.7)
            # Place label along edge at staggered t (0.35, 0.5, 0.65, ...)
            t_options = [0.35, 0.5, 0.65, 0.42, 0.58]
            t = t_options[idx % len(t_options)]
            mx = x1 + t * (x2 - x1)
            my = y1 + t * (y2 - y1)
            # Offset perpendicular to the edge to avoid sitting on top of line.
            edx, edy = x2 - x1, y2 - y1
            elen = math.hypot(edx, edy) or 1.0
            perp_x, perp_y = -edy / elen, edx / elen
            off = 0.22
            mx += perp_x * off
            my += perp_y * off
            ax.text(mx, my, str(w), ha="center", va="center",
                    fontsize=fs, fontweight="bold", fontfamily=ff,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#777", alpha=0.95), zorder=4)

        for i, name in enumerate(names):
            x, y = positions[name]
            circle = plt.Circle((x, y), 0.35, facecolor=palette[i % len(palette)],
                                edgecolor="#333", linewidth=lw + 1, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y - 0.6, name, ha="center", va="top",
                    fontsize=fs, fontweight="bold", fontfamily=ff, zorder=6)

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_title(rng.choice(_TITLE_VARIANTS), fontsize=fs + 3,
                     fontweight="bold", fontfamily=ff, pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = MapDistanceQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
