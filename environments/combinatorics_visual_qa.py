"""
Combinatorics Visual QA environment.

Visual counting problems: paths on grids, arrangements of objects,
selections from displayed items, colored region counting.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class CombinatoricsVisualQA(StandaloneVisualEnv):
    ENV_NAME = "combinatorics_visual"

    QUESTION_TYPES = [
        "count_paths", "count_arrangements", "count_selections",
        "count_colored_regions", "grid_paths",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"qtypes": ["count_colored_regions",
                               "count_arrangements"]}
        if level <= 5:
            return {"qtypes": ["count_colored_regions", "grid_paths",
                               "count_arrangements"]}
        if level <= 7:
            return {"qtypes": ["grid_paths", "count_selections",
                               "count_paths"]}
        return {"qtypes": ["count_paths", "count_selections"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type", self._rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choice(cfg["qtypes"])

        for _ in range(20):
            result = self._try_generate(qtype)
            if result is not None:
                return result
        return None

    def _try_generate(self, qtype: str) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng

        if qtype == "grid_paths":
            return self._grid_paths(rng)
        elif qtype == "count_paths":
            return self._count_paths_graph(rng)
        elif qtype == "count_arrangements":
            return self._count_arrangements(rng)
        elif qtype == "count_selections":
            return self._count_selections(rng)
        elif qtype == "count_colored_regions":
            return self._count_colored_regions(rng)
        return None

    def _grid_paths(self, rng):
        """Count shortest paths from top-left to bottom-right on a grid."""
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        # C(rows+cols, rows) = number of shortest paths (right and down moves)
        answer = math.comb(rows + cols, rows)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        lw = style["line_width"]
        fs = style["font_size_base"]
        line_color = style["geo_line_color"]

        # Draw grid
        for i in range(cols + 1):
            ax.plot([i, i], [0, rows], color=line_color, linewidth=lw)
        for j in range(rows + 1):
            ax.plot([0, cols], [j, j], color=line_color, linewidth=lw)

        # Mark start and end
        palette = style["palette"]
        ax.plot(0, rows, 'o', color=palette[0], markersize=18, zorder=5)
        ax.text(0, rows, 'S', ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white', zorder=6)
        ax.plot(cols, 0, 's', color=palette[1 % len(palette)], markersize=18, zorder=5)
        ax.text(cols, 0, 'E', ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white', zorder=6)

        # Label grid dimensions
        ax.text(cols / 2, rows + 0.4, f'{cols} columns', ha='center',
                fontsize=fs + 2, fontweight='bold')
        ax.text(-0.5, rows / 2, f'{rows}\nrows', ha='center', va='center',
                fontsize=fs + 2, fontweight='bold', rotation=0)

        ax.set_xlim(-0.8, cols + 0.8)
        ax.set_ylim(-0.8, rows + 0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Grid Paths (move only right or down)', fontsize=fs + 4,
                      fontweight='bold', pad=15)
        fig.tight_layout()

        q = (f"How many shortest paths are there from S (top-left) to E "
             f"(bottom-right), moving only right or down on this "
             f"{rows}x{cols} grid?")
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    def _count_paths_graph(self, rng):
        """Count paths from A to B in a small directed acyclic graph."""
        # Simple layered graph: 3 layers
        layers = [1, rng.randint(2, 3), rng.randint(2, 3), 1]
        # Build adjacency: each node connects to some nodes in next layer
        nodes_per_layer = []
        node_id = 0
        for n in layers:
            layer = list(range(node_id, node_id + n))
            nodes_per_layer.append(layer)
            node_id += n

        edges = []
        for li in range(len(layers) - 1):
            for src in nodes_per_layer[li]:
                # Connect to at least one node in next layer
                targets = list(nodes_per_layer[li + 1])
                rng.shuffle(targets)
                num_conn = rng.randint(1, len(targets))
                for t in targets[:num_conn]:
                    edges.append((src, t))

        # Count paths from first node to last node using DP
        total_nodes = node_id
        start = nodes_per_layer[0][0]
        end = nodes_per_layer[-1][0]

        # Build adjacency list
        adj = {i: [] for i in range(total_nodes)}
        for s, t in edges:
            adj[s].append(t)

        # DP: count paths
        path_count = [0] * total_nodes
        path_count[start] = 1
        for li in range(len(layers) - 1):
            for src in nodes_per_layer[li]:
                for t in adj[src]:
                    path_count[t] += path_count[src]

        answer = path_count[end]
        if answer == 0:
            return None

        # Draw the graph
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        positions = {}
        labels = {}
        label_names = [chr(65 + i) for i in range(total_nodes)]  # A, B, C, ...

        for li, layer in enumerate(nodes_per_layer):
            n = len(layer)
            for ni, nid in enumerate(layer):
                x = li * 2.5
                y = (n - 1) / 2 - ni
                positions[nid] = (x, y)
                labels[nid] = label_names[nid]

        # Draw edges
        for s, t in edges:
            sx, sy = positions[s]
            tx, ty = positions[t]
            ax.annotate('', xy=(tx, ty), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=style["geo_line_color"],
                                        lw=style["line_width"], connectionstyle='arc3,rad=0.1'))

        # Draw nodes
        palette = style["palette"]
        for nid, (x, y) in positions.items():
            color = palette[0] if nid == start else (palette[1 % len(palette)] if nid == end else palette[2 % len(palette)])
            circle = plt.Circle((x, y), 0.3, color=color, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, labels[nid], ha='center', va='center',
                    fontsize=style["font_size_base"] + 1, fontweight='bold', color='white', zorder=6)

        ax.set_xlim(-1, (len(layers) - 1) * 2.5 + 1)
        y_max = max(len(l) for l in nodes_per_layer)
        ax.set_ylim(-y_max, y_max)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Directed Graph: Count paths from {label_names[start]} to {label_names[end]}',
                      fontsize=style["font_size_base"] + 4, fontweight='bold', pad=15)
        fig.tight_layout()

        q = (f"How many distinct paths are there from node {label_names[start]} "
             f"to node {label_names[end]} following the arrows?")
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    def _count_arrangements(self, rng):
        """How many ways to arrange n distinct colored objects in a row."""
        n = rng.randint(3, 5)
        style = self._random_style()
        palette = style["palette"]
        colors = [palette[i % len(palette)] for i in range(n)]
        color_names = ['Red', 'Blue', 'Green', 'Orange', 'Purple',
                        'Teal', 'Brown'][:n]
        answer = math.factorial(n)

        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 4 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        fs = style["font_size_base"]
        for i in range(n):
            circle = plt.Circle((i * 1.5 + 1, 1.5), 0.5, color=colors[i],
                                 zorder=5, ec='black', lw=style["line_width"])
            ax.add_patch(circle)
            ax.text(i * 1.5 + 1, 0.7, color_names[i], ha='center',
                    fontsize=fs, fontweight='bold')

        ax.set_xlim(-0.5, n * 1.5 + 0.5)
        ax.set_ylim(0, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{n} Distinct Objects', fontsize=fs + 4, fontweight='bold', pad=15)
        fig.tight_layout()

        q = (f"How many different ways can these {n} distinct objects "
             f"be arranged in a row?")
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    def _count_selections(self, rng):
        """How many ways to choose k items from n displayed items."""
        n = rng.randint(4, 7)
        k = rng.randint(2, min(4, n - 1))
        answer = math.comb(n, k)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 4 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]
        colors = palette

        fs = style["font_size_base"]
        for i in range(n):
            x = (i % 5) * 1.5 + 1
            y = 2.5 - (i // 5) * 1.5
            rect = patches.FancyBboxPatch((x - 0.4, y - 0.4), 0.8, 0.8,
                                           boxstyle="round,pad=0.1",
                                           facecolor=colors[i % len(colors)],
                                           edgecolor='black', linewidth=style["line_width"])
            ax.add_patch(rect)
            ax.text(x, y, str(i + 1), ha='center', va='center',
                    fontsize=fs + 2, fontweight='bold', color='white')

        ax.set_xlim(-0.2, 8.5)
        ax.set_ylim(0, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{n} Items', fontsize=fs + 4, fontweight='bold', pad=15)
        fig.tight_layout()

        q = (f"How many ways can you choose {k} items from these {n} "
             f"distinct items? (Order does not matter.)")
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    def _count_colored_regions(self, rng):
        """Count regions of a specific color in a grid."""
        rows = rng.randint(3, 5)
        cols = rng.randint(3, 5)
        color_map = {0: '#3498db', 1: '#e74c3c', 2: '#27ae60'}
        color_names = {0: 'Blue', 1: 'Red', 2: 'Green'}

        grid = [[rng.randint(0, 2) for _ in range(cols)] for _ in range(rows)]
        target_color = rng.randint(0, 2)
        count = sum(1 for r in range(rows) for c in range(cols)
                    if grid[r][c] == target_color)

        if count == 0:
            # Ensure at least one
            r, c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
            grid[r][c] = target_color
            count = sum(1 for r in range(rows) for c in range(cols)
                        if grid[r][c] == target_color)

        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        for r in range(rows):
            for c in range(cols):
                rect = patches.Rectangle((c, rows - 1 - r), 1, 1,
                                          facecolor=color_map[grid[r][c]],
                                          edgecolor='black', linewidth=style["line_width"])
                ax.add_patch(rect)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Colored Grid', fontsize=style["font_size_base"] + 4, fontweight='bold', pad=15)
        fig.tight_layout()

        q = (f"How many {color_names[target_color]} cells are in the grid?")
        return q, str(count), self.fig_to_pil(fig, dpi=style["dpi"])
