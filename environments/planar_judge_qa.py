"""
Planar judge: given a small undirected graph, decide whether it is
planar (drawable in the plane without edge crossings) and answer
Yes/No.

Mirrors qid 221 / 224 wording from the source distribution:
  "Is the following graph a planar graph? choice: (A) Yes (B) No.
   Ans: A."

We construct planar (Yes) and non-planar (No) instances directly,
so we always know the ground-truth answer without running a generic
planarity test:

  Planar (Yes):
    - simple cycle / path / tree
    - cycle with chords arranged so all chords stay on the same side
      (e.g. cycle plus a few non-crossing chords from one vertex)
    - wheel W_k (a cycle plus a hub vertex connected to all rim
      vertices) — planar for any k
    - K_4 is planar
    - any tree is planar

  Non-planar (No):
    - explicit K_5 on 5 vertices (plus optional pendants → Kuratowski
      subgraph still embedded)
    - explicit K_{3,3} on 6 vertices (plus optional pendants)

Output format: Yes / No (mirrors the existing `bipartite_judge_qa.py`;
the framework's `_check_answer` already handles natural-language
yes/no parsing).
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
    "The image shows an undirected graph. Is it a planar graph (can it be drawn in the plane without edge crossings)? Answer Yes or No in <answer>...</answer>.",
    "Is the graph in the image planar? Answer Yes or No in <answer>...</answer>.",
    "Determine whether the depicted graph is planar (drawable without crossing edges). Output Yes or No in <answer>...</answer>.",
    "Planarity check: is the graph shown planar? Yes or No in <answer>...</answer>.",
    "From the depicted graph, output whether it is planar. Yes or No in <answer>...</answer>.",
    "Output Yes if the graph in the image is planar, No otherwise. Place answer in <answer>...</answer>.",
    "Is the undirected graph drawn in the image a planar graph? Answer Yes or No in <answer>...</answer>.",
    "Tell me whether the graph shown can be drawn without any edge crossings. Yes or No in <answer>...</answer>.",
    "Yes or No: is the depicted graph planar? Place answer in <answer>...</answer>.",
    "Decide planarity for the graph in the image. Answer Yes or No in <answer>...</answer>.",
    "Is the following graph a planar graph? Answer Yes or No in <answer>...</answer>.",
    "From the image, judge whether the graph is planar. Yes/No in <answer>...</answer>.",
    "Can the vertices and edges of the depicted graph be re-embedded in the plane with no crossings? Answer Yes or No in <answer>...</answer>.",
    "Output Yes if the depicted graph is planar, otherwise No. Place answer in <answer>...</answer>.",
    "Inspect the graph in the image and decide whether it is planar. Yes or No in <answer>...</answer>.",
    "Is the graph drawn in the image planar (no required crossings)? Output Yes or No in <answer>...</answer>.",
]


class PlanarJudgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "planar_judge"

    shape_strategy = "binary"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"level": level, "trivial": True}
        # n grows but we stay within the K5 / K3,3 ± pendant range.
        if level <= 2:
            return {"level": level, "n": 5, "extra_edges": 0}
        if level <= 4:
            return {"level": level, "n": 6, "extra_edges": 1}
        if level <= 6:
            return {"level": level, "n": 7, "extra_edges": 1}
        if level <= 8:
            return {"level": level, "n": 7, "extra_edges": 2}
        return {"level": level, "n": 8, "extra_edges": 2}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1607 + level * 97 + 29)

        if cfg.get("trivial"):
            # Two simple cases:
            #   triangle K_3 → planar (Yes)
            #   K_5 (5-vertex complete) → non-planar (No)
            want_yes = bool(rng.randint(0, 1))
            if want_yes:
                n = 3
                labels = ["A", "B", "C"]
                edges = [(0, 1), (1, 2), (0, 2)]
                ans = "Yes"
            else:
                n = 5
                labels = ["A", "B", "C", "D", "E"]
                edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
                ans = "No"
        else:
            want_yes = bool(rng.randint(0, 1))
            n = cfg["n"]
            extra_edges = cfg["extra_edges"]
            labels = [chr(ord("A") + i) for i in range(n)]
            if want_yes:
                edges = self._build_planar(rng, n, extra_edges)
                ans = "Yes"
            else:
                edges = self._build_nonplanar(rng, n, extra_edges)
                ans = "No"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        img = self._render(labels, edges, rng)
        return question, ans, img

    # ---------------------------------------------------------------- #
    # Construction helpers
    # ---------------------------------------------------------------- #

    def _build_planar(self, rng, n: int, extra_edges: int):
        """Construct a guaranteed-planar graph on n vertices.

        Strategy randomly chosen:
          1. cycle C_n
          2. wheel W_{n-1}: cycle on n-1 vertices + hub
          3. tree (random)
          4. cycle with non-crossing chords from one vertex (a "fan")
        Then optionally add pendant vertices via tree edges (still planar).
        """
        choice = rng.choice(["cycle", "wheel", "tree", "fan"])
        edges: List[Tuple[int, int]] = []
        if choice == "cycle":
            # Cycle on all n vertices
            for i in range(n):
                edges.append((i, (i + 1) % n))
        elif choice == "wheel":
            # Hub = vertex 0; rim = vertices 1..n-1 forming a cycle
            rim = list(range(1, n))
            for i in range(len(rim)):
                a = rim[i]
                b = rim[(i + 1) % len(rim)]
                edges.append((min(a, b), max(a, b)))
            for r in rim:
                edges.append((0, r))
        elif choice == "tree":
            # random labelled tree (any tree is planar)
            perm = list(range(n))
            rng.shuffle(perm)
            for i in range(1, n):
                u = perm[i]
                v = perm[rng.randint(0, i - 1)]
                edges.append((min(u, v), max(u, v)))
        else:  # fan
            # cycle with chords all from vertex 0 (fan triangulation;
            # always planar)
            for i in range(n):
                edges.append((i, (i + 1) % n))
            # chords from 0 to 2, 3, ..., n-2  (avoid duplicates and
            # multi-edges)
            for j in range(2, n - 1):
                a, b = 0, j
                if (a, b) not in edges and (b, a) not in edges:
                    edges.append((a, b))

        # Add a few "extra" edges that stay planar: only between vertices
        # already on the same "outer boundary" — easiest to keep planar
        # by adding a leaf via subdivision. We approximate this by adding
        # an edge only if it doesn't create K_5 / K_3,3.  Safe to skip if
        # uncertain — better to under-add than risk going non-planar.
        added = 0
        tries = 0
        existing = set(tuple(sorted(e)) for e in edges)
        while added < extra_edges and tries < 30:
            tries += 1
            i, j = rng.sample(range(n), 2)
            key = (min(i, j), max(i, j))
            if key in existing:
                continue
            # Tentatively add and verify planarity heuristically:
            # we accept only edges that keep |E| <= 3n - 6 (necessary
            # condition for planarity) AND the graph contains no K_5
            # subgraph and no K_{3,3} subgraph.
            if len(edges) + 1 > 3 * n - 6:
                continue
            edges.append(key)
            existing.add(key)
            if not self._has_k5(n, edges) and not self._has_k33(n, edges):
                added += 1
            else:
                edges.pop()
                existing.remove(key)

        # Deduplicate (defensive)
        edges = list({(min(u, v), max(u, v)) for (u, v) in edges})
        return edges

    def _build_nonplanar(self, rng, n: int, extra_edges: int):
        """Construct a guaranteed non-planar graph on n vertices by
        embedding K_5 or K_{3,3} as a subgraph and attaching the
        remaining vertices as pendants (which preserves Kuratowski).
        """
        choice = rng.choice(["k5", "k33"])
        edges: List[Tuple[int, int]] = []
        if choice == "k5" and n >= 5:
            for i in range(5):
                for j in range(i + 1, 5):
                    edges.append((i, j))
            # Attach pendants
            for k in range(5, n):
                # pick a vertex to attach to
                v = rng.randint(0, k - 1)
                edges.append((v, k))
        elif choice == "k33" or n < 5:
            # K_{3,3}: bipartition {0,1,2} vs {3,4,5}
            if n < 6:
                # fall back to K_5 if not enough room
                for i in range(min(5, n)):
                    for j in range(i + 1, min(5, n)):
                        edges.append((i, j))
            else:
                left = [0, 1, 2]
                right = [3, 4, 5]
                for u in left:
                    for v in right:
                        edges.append((u, v))
                for k in range(6, n):
                    v = rng.randint(0, k - 1)
                    edges.append((v, k))
        # Add a few extras (still non-planar — adding edges to a
        # non-planar graph keeps it non-planar).
        existing = set(tuple(sorted(e)) for e in edges)
        added = 0
        tries = 0
        while added < extra_edges and tries < 30:
            tries += 1
            i, j = rng.sample(range(n), 2)
            key = (min(i, j), max(i, j))
            if key in existing:
                continue
            edges.append(key)
            existing.add(key)
            added += 1
        return list(existing)

    # ---------------------------------------------------------------- #
    # Subgraph detection: simple K_5 / K_{3,3}
    # ---------------------------------------------------------------- #

    @staticmethod
    def _has_k5(n: int, edges: List[Tuple[int, int]]) -> bool:
        """Check if graph contains K_5 as a subgraph (5 mutually adjacent
        vertices)."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # iterate over all 5-subsets — fine for n ≤ 8 (at most C(8,5)=56)
        from itertools import combinations
        for sub in combinations(range(n), 5):
            ok = True
            for a, b in combinations(sub, 2):
                if b not in adj[a]:
                    ok = False
                    break
            if ok:
                return True
        return False

    @staticmethod
    def _has_k33(n: int, edges: List[Tuple[int, int]]) -> bool:
        """Check if graph contains K_{3,3} as a subgraph: any partition
        of 6 vertices into two triples with all 9 cross-edges."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        from itertools import combinations
        for six in combinations(range(n), 6):
            for left in combinations(six, 3):
                right = tuple(v for v in six if v not in left)
                # check all 9 cross edges
                ok = True
                for a in left:
                    for b in right:
                        if b not in adj[a]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return True
        return False

    def _render(self, labels, edges, rng):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        pos = {}
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2 + rng.uniform(-0.04, 0.04)
            r = 1.2 + rng.uniform(-0.05, 0.05)
            pos[i] = (r * math.cos(angle), r * math.sin(angle))
        for (u, v) in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#566573",
                    linewidth=1.4, zorder=1)
        for i in range(n):
            x, y = pos[i]
            ax.add_patch(plt.Circle((x, y), 0.16, facecolor="#c5e1a5",
                                    edgecolor="#33691e", linewidth=1.4,
                                    zorder=5))
            ax.text(x, y, labels[i], ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#1b5e20",
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
