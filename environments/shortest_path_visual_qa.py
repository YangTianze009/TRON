"""
Shortest Path Visual QA.

2026-05-05 R5 B5B (FINAL): REWRITTEN as pure single-letter MCQ. Given a
weighted directed graph rendered on the canvas, the model picks the value of
the shortest path length from S (source) to G (target) from a multiple-choice
list.

Per-level design:
  L0/L1 — 4-MCQ. 4 nodes, simple connectivity, small integer weights.
  L2-L4 — 5-MCQ. 5-6 nodes. 10/15% trap (correct = "Cannot reach"/"No correct").
  L5-L7 — 5-MCQ. 7-8 nodes. 20% trap. Tighter distractors.
  L8/L9 — 5-MCQ. 9-10 nodes. 30/40% trap. Off-by-1/2 distractors.
"""
import math
import random
import string
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _dijkstra(N: int, edges: List[Tuple[int, int, int]],
              src: int, dst: int) -> Optional[int]:
    import heapq
    adj = [[] for _ in range(N)]
    for s, t, w in edges:
        adj[s].append((t, w))
    dist = [float("inf")] * N
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            return d
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist[dst] if dist[dst] != float("inf") else None


class ShortestPathVisualQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # graph layout is orientation-sensitive
    ENV_NAME = "shortest_path_visual"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "N": 4, "max_w": 4, "edge_density": 0.45,
                "n_options": 4, "trap_rate": 0.0, "tight": False,
            }
        if level <= 4:
            return {
                "N": 5 + (level - 2) // 2, "max_w": 6,
                "edge_density": 0.35,
                "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.15,
                "tight": False,
            }
        if level <= 7:
            return {
                "N": 7 + (level - 5) // 2, "max_w": 8,
                "edge_density": 0.30,
                "n_options": 5,
                "trap_rate": 0.20,
                "tight": False,
            }
        # L8/L9
        return {
            "N": 9 + (level - 8), "max_w": 10,
            "edge_density": 0.25,
            "n_options": 5,
            "trap_rate": 0.40,
            "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 8191 + level * 211 + 17)

        for _ in range(20):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg):
        N = cfg["N"]
        max_w = cfg["max_w"]
        edge_density = cfg["edge_density"]
        # Build a guaranteed S->...->G path
        constructed_path = list(range(1, N - 1))
        rng.shuffle(constructed_path)
        constructed_path = [0] + constructed_path + [N - 1]
        edges = []
        for s, t in zip(constructed_path, constructed_path[1:]):
            w = rng.randint(1, max(1, N // 2))
            edges.append((s, t, w))

        num_edges = max(len(edges), int(edge_density * N * (N - 1)))
        existing = set((s, t) for s, t, w in edges)
        candidates = [(s, t) for s in range(N) for t in range(N)
                      if s != t and (s, t) not in existing
                      and not (s == 0 and t == N - 1)]
        rng.shuffle(candidates)
        for s, t in candidates[: max(0, num_edges - len(edges))]:
            w = rng.randint(max(2, N // 2), max_w)
            edges.append((s, t, w))

        # remove direct S->G edge to avoid trivial
        edges = [(s, t, w) for s, t, w in edges
                 if not (s == 0 and t == N - 1)]
        rng.shuffle(edges)

        dist = _dijkstra(N, edges, 0, N - 1)
        if dist is None or dist <= 0 or dist < 3:
            return None

        labels = self._make_labels(N)

        # Build options
        n_options = cfg["n_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]

        # Numeric distractor pool around true distance
        if cfg["tight"]:
            offsets = [-3, -2, -1, 1, 2, 3, -4, 4]
        else:
            offsets = [-4, -3, -2, -1, 1, 2, 3, 4, 5, -5]
        cand = []
        for o in offsets:
            v = dist + o
            if v > 0 and v != dist and v not in cand:
                cand.append(v)
        rng.shuffle(cand)

        if n_options == 4:
            need = 3
            if len(cand) < need:
                return None
            options = [str(dist)] + [str(d) for d in cand[:need]]
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(str(dist))]
        else:
            n_value_slots = 4
            use_trap = (rng.random() < cfg["trap_rate"])
            if use_trap and len(cand) >= n_value_slots:
                chosen = cand[:n_value_slots]
                rng.shuffle(chosen)
                options = [str(d) for d in chosen]
                options.append("Cannot be determined / no correct answer")
                correct_letter = last_letter
            else:
                if len(cand) < n_value_slots - 1:
                    return None
                chosen = [dist] + cand[:n_value_slots - 1]
                rng.shuffle(chosen)
                options = [str(d) for d in chosen]
                options.append("Cannot be determined / no correct answer")
                correct_letter = opt_letters[options.index(str(dist))]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stems = [
            ("In the directed weighted graph shown, what is the shortest "
             "distance from node S to node G (sum of edge weights along "
             "the shortest directed path)?"),
            ("As shown in the diagram, find the length of the shortest "
             "directed path from S to G."),
            ("The figure shows a directed graph where each edge is "
             "labeled with a positive weight. What is the total weight of "
             "the shortest path from S to G?"),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        img = self._render(N, edges, labels)
        return question, correct_letter, img

    @staticmethod
    def _make_labels(N: int) -> List[str]:
        labs = ["S"]
        letters = [c for c in string.ascii_uppercase if c not in ("S", "G")]
        for i in range(1, N - 1):
            labs.append(letters[(i - 1) % len(letters)])
        labs.append("G")
        return labs

    # ------------------------------------------------------------------ #
    def _render(self, N, edges, labels) -> Image.Image:
        fig_size = max(6, 2 + N * 0.7)
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=100)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        node_positions = {}
        for i in range(N):
            angle = 2 * math.pi * i / N - math.pi / 2
            radius = max(2.5, N * 0.5)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_positions[i] = (x, y)

        edge_set = set((s, t) for s, t, w in edges)
        for s, t, w in edges:
            sx, sy = node_positions[s]
            tx, ty = node_positions[t]
            has_reverse = (t, s) in edge_set
            rad = 0.2 if has_reverse else 0.0
            ax.annotate("",
                        xy=(tx, ty), xytext=(sx, sy),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="#7f8c8d",
                            lw=1.5,
                            connectionstyle=f"arc3,rad={rad}",
                            shrinkA=18, shrinkB=18,
                        ))
            mx = (sx + tx) / 2
            my = (sy + ty) / 2
            if has_reverse:
                dx = tx - sx
                dy = ty - sy
                length = math.sqrt(dx * dx + dy * dy) + 1e-9
                nx_off = -dy / length * 0.35
                ny_off = dx / length * 0.35
                mx += nx_off
                my += ny_off
            ax.text(mx, my, str(w),
                    ha="center", va="center",
                    fontsize=max(8, 12 - N // 3),
                    fontweight="bold", color="#e74c3c",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85))

        node_size = max(11, 18 - N // 2)
        for i in range(N):
            x, y = node_positions[i]
            if i == 0:
                color = "#27ae60"
            elif i == N - 1:
                color = "#e74c3c"
            else:
                color = "#3498db"
            circle = plt.Circle((x, y), 0.4, facecolor=color,
                                edgecolor="#2c3e50", linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, labels[i],
                    ha="center", va="center",
                    fontsize=node_size, fontweight="bold",
                    color="white", zorder=6)

        margin = max(3.5, N * 0.6 + 0.5)
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        ax.axis("off")

        return self.fig_to_pil(fig, dpi=100)


if __name__ == "__main__":
    env = ShortestPathVisualQA()
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer}")
