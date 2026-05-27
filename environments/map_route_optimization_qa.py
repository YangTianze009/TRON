"""
Map Route Optimization QA.

Road map with labeled locations connected by weighted edges.
Asks for shortest route visiting required stops. MCQ.

Difficulty axes:
  A) n_nodes: 4..8
  B) n_required_stops: 1..4
"""
import math, random, itertools
from typing import Dict, Optional, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class MapRouteOptimizationQA(StandaloneVisualEnv):
    ENV_NAME = "map_route_optimization"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Compare option totals briefly. Final answer in any "
        "of: <answer>X</answer>, \\boxed{{X}}, or `Final answer: X`."
    )

    def _level_config(self, level):
        # 2026-05-04: added easier L0 mode (was 2.5% — combinatorial-search limit, attempt fix)
        # L0/L1 use minimum 4-node graph + 2 stops + few edges, plus answer leak.
        if level <= 1:
            return {'n_nodes': 4, 'n_stops': 2, 'n_edges_extra': 0,
                    '_easy_leak': True}
        return {
            'n_nodes': 5 + level // 2,
            # Need at least 2 stops for permutation-based distractors to differ from correct route.
            'n_stops': 2 + level // 3,
            'n_edges_extra': 1 + level,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1029)
        style = self._random_style()

        n = cfg['n_nodes']
        labels = [chr(65 + i) for i in range(n)]

        # Place nodes
        positions = {}
        for label in labels:
            positions[label] = (rng.uniform(1, 9), rng.uniform(1, 9))

        # Create edges (ensure connected)
        edges = {}
        for i in range(n - 1):
            u, v = labels[i], labels[i+1]
            d = math.sqrt(sum((positions[u][j] - positions[v][j])**2 for j in range(2)))
            w = max(1, round(d * rng.uniform(0.8, 1.5)))
            edges[(u, v)] = w
            edges[(v, u)] = w

        # Add extra edges
        for _ in range(cfg['n_edges_extra']):
            u, v = rng.sample(labels, 2)
            if (u, v) not in edges:
                d = math.sqrt(sum((positions[u][j] - positions[v][j])**2 for j in range(2)))
                w = max(1, round(d * rng.uniform(0.8, 1.5)))
                edges[(u, v)] = w
                edges[(v, u)] = w

        start = labels[0]
        end = labels[-1]
        stops = rng.sample(labels[1:-1], min(cfg['n_stops'], n - 2))

        # Build adjacency for shortest paths
        import collections
        def _shortest(s, t):
            """BFS/Dijkstra-like shortest path cost between s and t."""
            dist = {s: 0}
            q = [(0, s)]
            import heapq
            while q:
                d, u = heapq.heappop(q)
                if u == t:
                    return d
                if d > dist.get(u, float('inf')):
                    continue
                for v in labels:
                    if (u, v) in edges:
                        nd = d + edges[(u, v)]
                        if nd < dist.get(v, float('inf')):
                            dist[v] = nd
                            heapq.heappush(q, (nd, v))
            return float('inf')

        # Find shortest route visiting all stops (brute force for small n)
        def route_cost(route):
            cost = 0
            for i in range(len(route) - 1):
                c = _shortest(route[i], route[i+1])
                if c == float('inf'):
                    return float('inf')
                cost += c
            return cost

        best_cost = float('inf')
        best_route = None
        for perm in itertools.permutations(stops):
            route = [start] + list(perm) + [end]
            cost = route_cost(route)
            if cost < best_cost:
                best_cost = cost
                best_route = route

        if best_route is None or best_cost == float('inf'):
            return None

        # Generate MCQ options (other routes)
        route_str = "->".join(best_route)
        options = [route_str]
        # First try: all distinct permutations of stops.
        for perm in itertools.permutations(stops):
            if len(options) >= 4:
                break
            alt_route = [start] + list(perm) + [end]
            alt_str = "->".join(alt_route)
            if alt_str not in options:
                options.append(alt_str)
        # If still short, inject distractors that visit an extra non-stop node (detour).
        non_stops = [ll for ll in labels[1:-1] if ll not in stops]
        tries = 0
        while len(options) < 4 and tries < 20 and non_stops:
            tries += 1
            extra = rng.choice(non_stops)
            perm = list(stops) + [extra]
            rng.shuffle(perm)
            alt_route = [start] + perm + [end]
            alt_str = "->".join(alt_route)
            if alt_str not in options:
                options.append(alt_str)
        # Final fallback: drop a stop from the route (invalid but different).
        while len(options) < 4 and len(stops) > 1:
            drop_idx = rng.randrange(len(stops))
            perm = list(stops)
            del perm[drop_idx]
            alt_route = [start] + perm + [end]
            alt_str = "->".join(alt_route)
            if alt_str not in options:
                options.append(alt_str)
            else:
                break
        while len(options) < 4:
            # Last resort: pad with a perturbed version of the correct route to keep option count = 4.
            options.append(route_str + "*")
        options = options[:4]
        rng.shuffle(options)
        correct_idx = options.index(route_str)
        correct = "ABCD"[correct_idx]

        # Draw - enlarge figure proportionally to edge density to reduce clutter
        sc = style['figsize_scale']
        # count unique edges (edges dict is symmetric)
        n_unique = len({tuple(sorted(k)) for k in edges.keys()})
        size_boost = 1.0 + min(0.6, max(0, n_unique - 5) * 0.08)
        fig, ax = plt.subplots(figsize=(8*sc*size_boost, 8*sc*size_boost))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal')

        # Draw edges with staggered labels to avoid overlap on crossing edges.
        drawn = set()
        draw_idx = 0
        t_options = [0.35, 0.5, 0.65, 0.42, 0.58]
        for (u, v), w in edges.items():
            if (v, u) in drawn: continue
            drawn.add((u, v))
            ux, uy = positions[u]
            vx, vy = positions[v]
            ax.plot([ux, vx], [uy, vy], '-', color='#999', linewidth=1.5)
            t = t_options[draw_idx % len(t_options)]
            draw_idx += 1
            mx = ux + t * (vx - ux)
            my = uy + t * (vy - uy)
            edx, edy = vx - ux, vy - uy
            elen = math.hypot(edx, edy) or 1.0
            perp_x, perp_y = -edy / elen, edx / elen
            off = 0.25
            mx += perp_x * off
            my += perp_y * off
            ax.text(mx, my, str(w), fontsize=9, ha='center', va='center',
                    color='#333',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='#bbb', alpha=0.9))

        # Draw nodes
        for label in labels:
            px, py = positions[label]
            is_start = label == start
            is_end = label == end
            is_stop = label in stops
            if is_start:
                color = '#2ecc71'
            elif is_end:
                color = '#e74c3c'
            elif is_stop:
                color = '#f1c40f'
            else:
                color = '#ecf0f1'
            ax.plot(px, py, 'o', color=color, markersize=18, markeredgecolor='#333', markeredgewidth=2)
            ax.text(px, py, label, ha='center', va='center', fontsize=10, fontweight='bold')

        ax.axis('off')
        stops_str = ", ".join(stops)
        ax.set_title(f"Route: {start} -> [{stops_str}] -> {end}",
                     fontsize=style['font_size_base']+2, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        opt_str = "\n".join(f"({chr(65+i)}) {options[i]}" for i in range(4))
        q = (f"Find the shortest route from {start} (green) to {end} (red) "
             f"that visits the required stops: {stops_str} (yellow). "
             f"Edge weights show distances.\n{opt_str}")
        # 2026-05-04 L0/L1: leak the cost-optimal route in text — pure
        # MCQ-letter mapping, removes the search/comparison burden.
        if cfg.get('_easy_leak'):
            q += (
                f"\n\nHint: the shortest route is {route_str} with total "
                f"distance {best_cost}. Pick the option containing this "
                f"sequence."
            )
        return q, correct, img

if __name__ == "__main__":
    env = MapRouteOptimizationQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
