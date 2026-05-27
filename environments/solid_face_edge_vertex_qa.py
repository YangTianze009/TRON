"""
Solid Face / Edge / Vertex Count QA — pure single-letter MCQ A-D / A-E.

2026-05-05 R5 P3 HARDENED: complete rewrite. Was 100% saturated because
single-property questions on cube/cuboid are memorizable. New design:

  - L0/L1: single-property F/E/V on basic solids (cube, cuboid, sq pyramid,
           tri prism). 4-MCQ. Trivial floor preserved.
  - L2/L3: F/E/V on more solids (penta/hexagonal prism, tetrahedron).
           5-MCQ + 10% trap.
  - L4/L5: TUPLE question — "(faces, edges, vertices) of this solid is ( )".
           4 candidate tuples + "No correct". Forces all 3 to be right.
  - L6/L7: Two-solid combined F/E/V counts (compound 2-step). Tuple options.
  - L8/L9: TRUNCATED solid (Euler V−E+F=2 must hold). Tuple options + 40%
           trap. E.g. "cube with one corner cut off" = (7, 15, 10) per Euler.

The tuple option style is MUCH harder than single-prop because all 3 must
be right, and distractors include "swapped F and V" + Euler-violating tuples
that look reasonable.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Solid catalog: faces, edges, vertices
_SOLIDS = {
    "cube":              {"F": 6,  "E": 12, "V": 8},
    "cuboid":            {"F": 6,  "E": 12, "V": 8},
    "triangular_prism":  {"F": 5,  "E": 9,  "V": 6},
    "pentagonal_prism":  {"F": 7,  "E": 15, "V": 10},
    "hexagonal_prism":   {"F": 8,  "E": 18, "V": 12},
    "heptagonal_prism":  {"F": 9,  "E": 21, "V": 14},
    "octagonal_prism":   {"F": 10, "E": 24, "V": 16},
    "nonagonal_prism":   {"F": 11, "E": 27, "V": 18},
    "square_pyramid":    {"F": 5,  "E": 8,  "V": 5},
    "pentagonal_pyramid":{"F": 6,  "E": 10, "V": 6},
    "hexagonal_pyramid": {"F": 7,  "E": 12, "V": 7},
    "triangular_pyramid":{"F": 4,  "E": 6,  "V": 4},  # tetrahedron
    # Truncated solids — apply Euler V-E+F=2 to verify
    # "cube with one corner cut off" (vertex-truncated cube, 1 vertex):
    #   removes 1 vertex (V=8→7), adds 1 triangular face (F=6→7),
    #   adds 3 edges (E=12→15). Euler: 7-15+10=2 ✓ (V=10? no — removing 1
    #   vertex, but the cut introduces 3 NEW vertices. Actual: V=8-1+3=10.
    "cube_corner_cut":   {"F": 7,  "E": 15, "V": 10},  # Euler 10-15+7=2 ✓
    # "cube with all 8 corners cut" (truncated cube):
    #   F=6+8=14, V=8*3=24, E=12+8*3=36. Euler 24-36+14=2 ✓
    "truncated_cube":    {"F": 14, "E": 36, "V": 24},
    # "tetrahedron with one corner cut off":
    #   removes 1 vertex (V=4→3), adds 1 triangular face (F=4→5),
    #   adds 3 vertices (V=3+3=6), adds 3 edges (E=6+3=9).
    #   Euler: 6-9+5=2 ✓
    "tetra_corner_cut":  {"F": 5,  "E": 9,  "V": 6},
}


_SHAPE_NAME = {
    "cube": "cube",
    "cuboid": "rectangular cuboid",
    "triangular_prism": "triangular prism",
    "pentagonal_prism": "pentagonal prism",
    "hexagonal_prism": "hexagonal prism",
    "heptagonal_prism": "heptagonal prism",
    "octagonal_prism": "octagonal prism",
    "nonagonal_prism": "nonagonal prism",
    "square_pyramid": "square pyramid",
    "pentagonal_pyramid": "pentagonal pyramid",
    "hexagonal_pyramid": "hexagonal pyramid",
    "triangular_pyramid": "triangular pyramid (tetrahedron)",
    "cube_corner_cut": "cube with one corner cut off (vertex-truncated)",
    "truncated_cube": "truncated cube (all 8 corners cut)",
    "tetra_corner_cut": "tetrahedron with one corner cut off",
}


class SolidFaceEdgeVertexQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a 3D solid, pick (F, E, V) tuple."""

    ENV_NAME = "solid_face_edge_vertex"
    ALLOW_ROTATION = False  # 3D orientation labels affect interpretation

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "shapes": ["cube", "cuboid", "square_pyramid",
                           "triangular_prism", "triangular_pyramid"],
                "qmode": "single",  # ask just F or E or V
                "n_solids": 1,
                "use_5_options": False,
                "trap_rate": 0.0,
            }
        if level <= 3:
            return {
                "shapes": ["cube", "cuboid", "square_pyramid",
                           "triangular_prism", "triangular_pyramid",
                           "pentagonal_prism", "hexagonal_prism"],
                "qmode": "single",
                "n_solids": 1,
                "use_5_options": True,
                "trap_rate": 0.10,
            }
        if level <= 5:
            return {
                "shapes": ["cube", "cuboid", "square_pyramid",
                           "triangular_prism", "triangular_pyramid",
                           "pentagonal_prism", "hexagonal_prism",
                           "pentagonal_pyramid", "hexagonal_pyramid"],
                "qmode": "tuple",  # ask (F, E, V) — all 3 must be right
                "n_solids": 1,
                "use_5_options": True,
                "trap_rate": 0.15,
            }
        if level <= 7:
            return {
                "shapes": ["cube", "cuboid", "triangular_prism",
                           "pentagonal_prism", "hexagonal_prism",
                           "square_pyramid", "triangular_pyramid",
                           "pentagonal_pyramid", "hexagonal_pyramid"],
                "qmode": "tuple",
                "n_solids": 2,  # combined sums
                "use_5_options": True,
                "trap_rate": 0.25,
            }
        # L8/L9
        return {
            "shapes": ["cube_corner_cut", "truncated_cube", "tetra_corner_cut",
                       "heptagonal_prism", "octagonal_prism", "nonagonal_prism",
                       "hexagonal_pyramid"],
            "qmode": "tuple",
            "n_solids": 1,
            "use_5_options": True,
            "trap_rate": 0.40,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5839 + level * 67 + 19)

        for attempt in range(15):
            try:
                res = self._try_generate(rng, cfg, level, attempt)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level, attempt):
        if cfg["n_solids"] == 1:
            shape = rng.choice(cfg["shapes"])
            sd = _SOLIDS[shape]
            F, E, V = sd["F"], sd["E"], sd["V"]
            shape_label = _SHAPE_NAME[shape]
        else:
            shape1 = rng.choice(cfg["shapes"])
            shape2 = rng.choice(cfg["shapes"])
            sd1, sd2 = _SOLIDS[shape1], _SOLIDS[shape2]
            F = sd1["F"] + sd2["F"]
            E = sd1["E"] + sd2["E"]
            V = sd1["V"] + sd2["V"]
            shape_label = (f"two solids — Solid 1: {_SHAPE_NAME[shape1]}, "
                           f"Solid 2: {_SHAPE_NAME[shape2]}")

        if cfg["qmode"] == "single":
            mode = rng.choice(["F", "E", "V"])
            ask_word = {"F": "faces", "E": "edges", "V": "vertices"}[mode]
            true_value = {"F": F, "E": E, "V": V}[mode]
            true_str = str(true_value)

            distractors = self._make_single_distractors(
                rng, mode, F, E, V, true_value)
        else:
            true_value = (F, E, V)
            true_str = f"({F}, {E}, {V})"
            distractors = self._make_tuple_distractors(rng, F, E, V)

        if not distractors:
            return None

        distr_strs = []
        seen = {true_str}
        for d in distractors:
            if cfg["qmode"] == "single":
                ds = str(d)
            else:
                ds = f"({d[0]}, {d[1]}, {d[2]})"
            if ds in seen:
                continue
            seen.add(ds)
            distr_strs.append(ds)

        if len(distr_strs) < 4:
            return None

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False
        if use_trap:
            chosen = distr_strs[:n_value]
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distr_strs[:n_distractor]
            options = [true_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_str)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Build stem
        if cfg["qmode"] == "single":
            ask_word = {"F": "faces", "E": "edges", "V": "vertices"}[mode]
            stems = [
                (f"As shown in the diagram, the solid is a {shape_label}. "
                 f"The total number of {ask_word} is ( )."),
                (f"The figure shows a {shape_label}. How many {ask_word} "
                 f"does it have ( )?"),
            ]
        else:
            if cfg["n_solids"] == 1:
                stems = [
                    (f"As shown in the diagram, the solid is a {shape_label}. "
                     f"Its (number of faces, number of edges, number of "
                     f"vertices) is ( )."),
                    (f"The figure shows a {shape_label}. Express its "
                     f"(F, E, V) as a tuple ( )."),
                ]
            else:
                stems = [
                    (f"As shown in the diagram, two solids are displayed "
                     f"side-by-side. {shape_label}. Compute the COMBINED "
                     f"(faces, edges, vertices) summed across both solids ( )."),
                    (f"The diagram shows two solids: {shape_label}. "
                     f"What is (F1+F2, E1+E2, V1+V2) ( )?"),
                ]

        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        # Render
        if cfg["n_solids"] == 1:
            image = self._render_one(shape)
        else:
            image = self._render_two(shape1, shape2)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _make_single_distractors(self, rng, mode, F, E, V, true_value):
        # Common errors
        candidates = []
        # Swap with another property
        if mode != "F":
            candidates.append(F)
        if mode != "E":
            candidates.append(E)
        if mode != "V":
            candidates.append(V)
        # ±2, ±4, ±1
        for d in (-2, -1, 1, 2, 3, -3, 4, -4):
            v = true_value + d
            if v > 0:
                candidates.append(v)
        # Off by Euler relation
        # If asked for E, common error is V+F-2 ... but that = E.
        # If asked for F: V-E+2
        if mode == "F":
            candidates.append(V - E + 2 + 1)  # off-by-1 Euler
        if mode == "V":
            candidates.append(E - F + 2 + 1)
        if mode == "E":
            candidates.append(V + F - 2 + 1)

        # Filter
        seen = {true_value}
        good = []
        for c in candidates:
            if c in seen or c <= 0:
                continue
            seen.add(c)
            good.append(c)
        rng.shuffle(good)
        return good[:6]

    def _make_tuple_distractors(self, rng, F, E, V):
        true_t = (F, E, V)
        # Generate confusable tuples
        candidates = []
        # 1. Swap F and V (very common error — students confuse face & vertex)
        candidates.append((V, E, F))
        # 2. Off by one in F
        candidates.append((F + 1, E, V))
        candidates.append((F - 1, E, V))
        # 3. Off by one in V
        candidates.append((F, E, V + 1))
        candidates.append((F, E, V - 1))
        # 4. Off by 2 in E (most-likely miscount)
        candidates.append((F, E + 2, V))
        candidates.append((F, E - 2, V))
        candidates.append((F, E + 1, V))
        candidates.append((F, E - 1, V))
        # 5. Swap E and V
        candidates.append((F, V, E))
        # 6. All shifted by 1
        candidates.append((F + 1, E + 1, V + 1))
        candidates.append((F - 1, E - 1, V - 1))
        # 7. Euler-violating but plausible
        candidates.append((F + 1, E + 2, V))
        candidates.append((F, E - 1, V + 1))

        seen = {true_t}
        good = []
        for c in candidates:
            if any(x <= 0 for x in c):
                continue
            if c in seen:
                continue
            seen.add(c)
            good.append(c)
        rng.shuffle(good)
        return good[:6]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_one(self, shape) -> Image.Image:
        fig = plt.figure(figsize=(5.5, 5.5), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        verts, edges = self._get_geometry(shape)
        for i, j in edges:
            x = [verts[i][0], verts[j][0]]
            y = [verts[i][1], verts[j][1]]
            z = [verts[i][2], verts[j][2]]
            ax.plot(x, y, z, color="#2c3e50", linewidth=2)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        ax.scatter(xs, ys, zs, color="#d62728", s=30, zorder=5)
        ax.view_init(elev=20, azim=35)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(_SHAPE_NAME.get(shape, shape),
                     fontsize=11, fontweight="bold")
        return self.fig_to_pil(fig, dpi=120)

    def _render_two(self, shape1, shape2) -> Image.Image:
        fig = plt.figure(figsize=(9, 5), dpi=120)
        for idx, sh in enumerate([shape1, shape2], 1):
            ax = fig.add_subplot(1, 2, idx, projection="3d")
            ax.set_axis_off()
            verts, edges = self._get_geometry(sh)
            for i, j in edges:
                x = [verts[i][0], verts[j][0]]
                y = [verts[i][1], verts[j][1]]
                z = [verts[i][2], verts[j][2]]
                ax.plot(x, y, z, color="#2c3e50", linewidth=2)
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            ax.scatter(xs, ys, zs, color="#d62728", s=30, zorder=5)
            ax.view_init(elev=20, azim=35)
            ax.set_box_aspect([1, 1, 1])
            ax.set_title(f"Solid {idx}: {_SHAPE_NAME.get(sh, sh)}",
                         fontsize=10, fontweight="bold")
        return self.fig_to_pil(fig, dpi=120)

    # ------------------------------------------------------------------ #
    def _get_geometry(self, shape):
        """Return (vertices_list, edges_list) for the named solid."""
        if shape == "cube":
            s = 1.0
            verts = [
                (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
                (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
            return verts, edges
        if shape == "cuboid":
            l, w, h = 1.5, 1.0, 0.7
            verts = [
                (0, 0, 0), (l, 0, 0), (l, w, 0), (0, w, 0),
                (0, 0, h), (l, 0, h), (l, w, h), (0, w, h)
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]
            return verts, edges
        if shape == "triangular_prism":
            verts = [
                (0, 0, 0), (1, 0, 0), (0.5, 0.866, 0),
                (0, 0, 1.2), (1, 0, 1.2), (0.5, 0.866, 1.2)
            ]
            edges = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
            return verts, edges
        if shape in ("pentagonal_prism", "hexagonal_prism", "heptagonal_prism",
                     "octagonal_prism", "nonagonal_prism"):
            n_map = {"pentagonal_prism": 5, "hexagonal_prism": 6,
                     "heptagonal_prism": 7, "octagonal_prism": 8,
                     "nonagonal_prism": 9}
            n = n_map[shape]
            verts = []
            for h in [0, 1.2]:
                for k in range(n):
                    a = 2 * math.pi * k / n
                    verts.append((math.cos(a), math.sin(a), h))
            edges = []
            for k in range(n):
                edges.append((k, (k + 1) % n))
                edges.append((n + k, n + (k + 1) % n))
                edges.append((k, n + k))
            return verts, edges
        if shape == "square_pyramid":
            verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                     (0.5, 0.5, 1.0)]
            edges = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]
            return verts, edges
        if shape == "triangular_pyramid":
            verts = [(0, 0, 0), (1, 0, 0), (0.5, 0.866, 0),
                     (0.5, 0.289, 0.816)]
            edges = [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)]
            return verts, edges
        if shape in ("pentagonal_pyramid", "hexagonal_pyramid"):
            n = 5 if shape == "pentagonal_pyramid" else 6
            verts = []
            for k in range(n):
                a = 2 * math.pi * k / n
                verts.append((math.cos(a), math.sin(a), 0))
            verts.append((0, 0, 1.2))  # apex
            edges = []
            for k in range(n):
                edges.append((k, (k + 1) % n))
                edges.append((k, n))  # apex
            return verts, edges
        if shape == "cube_corner_cut":
            # cube with corner at (1,1,1) cut by plane through nearby points
            s = 1.0
            cut = 0.4
            # Original cube vertices (vertex 6 = (1,1,1) is cut)
            verts = [
                (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
                (0, 0, s), (s, 0, s), (0, s, s),  # vertex (1,1,1) removed
                # Three new vertices near where (1,1,1) was
                (s, s, s - cut),  # along bottom-edge of cut, on z-face
                (s, s - cut, s),  # along x-edge of cut, on y-face
                (s - cut, s, s),  # along y-edge of cut, on x-face
            ]
            edges = [
                (0,1),(1,2),(2,3),(3,0),  # bottom square
                (4,5),(6,4),  # top square (partial)
                (0,4),(1,5),(2,7),(3,6),  # vertical edges
                (5,8),(6,9),  # remaining top edges
                (5,8),  # connects (1,0,1)→(1,1-cut,1) — wait let me redo
            ]
            # Let me re-derive clean indices:
            # 0: (0,0,0)
            # 1: (1,0,0)
            # 2: (1,1,0)
            # 3: (0,1,0)
            # 4: (0,0,1)
            # 5: (1,0,1)
            # 6: (0,1,1)
            # 7: (1,1,1-cut) — on edge (1,1,0)→(1,1,1)
            # 8: (1,1-cut,1) — on edge (1,0,1)→(1,1,1)
            # 9: (1-cut,1,1) — on edge (0,1,1)→(1,1,1)
            verts = [
                (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
                (0, 0, s), (s, 0, s), (0, s, s),
                (s, s, s - cut),
                (s, s - cut, s),
                (s - cut, s, s),
            ]
            edges = [
                # bottom square
                (0,1),(1,2),(2,3),(3,0),
                # vertical edges (4 of them; the (1,1,0)→(1,1,1) edge is now (1,1,0)→(1,1,1-cut)=(2,7))
                (0,4),(1,5),(2,7),(3,6),
                # top edges (originally 4: (4,5)(5,6)(6,7)(7,4) but corner removed)
                (4,5),
                (5,8),  # (1,0,1) to (1,1-cut,1)
                (6,9),  # (0,1,1) to (1-cut,1,1)
                (4,6),  # (0,0,1) to (0,1,1) — yes still there
                # Triangle of cut
                (7,8),(8,9),(9,7),
            ]
            return verts, edges
        if shape == "truncated_cube":
            # Cube with all 8 corners cut. Each original vertex becomes 3
            # new vertices; each face becomes octagon.
            s = 1.0
            cut = 0.3
            # 24 vertices: for each original vertex (x,y,z), 3 new vertices
            # close to it along the 3 edges.
            cube_v = [
                (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
                (0, 0, s), (s, 0, s), (s, s, s), (0, s, s),
            ]
            verts = []
            # Map: for each cube vertex, the 3 displacements along its 3 edges
            for (x, y, z) in cube_v:
                dx = -cut if x > 0.5 else cut
                dy = -cut if y > 0.5 else cut
                dz = -cut if z > 0.5 else cut
                verts.append((x + dx, y, z))
                verts.append((x, y + dy, z))
                verts.append((x, y, z + dz))
            # Edges: (a) triangles around each truncated corner (8 triangles, 24 edges)
            #        (b) the remaining 12 edges between corner-clusters
            edges = []
            for c in range(8):
                # triangle within cluster c
                a, b, d = c * 3, c * 3 + 1, c * 3 + 2
                edges.append((a, b))
                edges.append((b, d))
                edges.append((d, a))
            # Inter-cluster edges (replacing original 12 edges)
            # original edge (0,1) between cube vertices 0 and 1 — connects
            # the displacement-along-x of vertex 0 to displacement-along-x of vertex 1
            cube_edges = [(0,1),(1,2),(2,3),(3,0),
                          (4,5),(5,6),(6,7),(7,4),
                          (0,4),(1,5),(2,6),(3,7)]
            # for each cube edge (i,j), find the displacement direction (which axis)
            for (i, j) in cube_edges:
                v_i, v_j = cube_v[i], cube_v[j]
                # axis differs
                axis = None
                for k in range(3):
                    if v_i[k] != v_j[k]:
                        axis = k
                        break
                # i's displacement-axis = axis is verts[i*3 + axis]
                edges.append((i * 3 + axis, j * 3 + axis))
            return verts, edges
        if shape == "tetra_corner_cut":
            # tetrahedron with one corner cut
            cut = 0.3
            # original tetra: vertices at (0,0,0), (1,0,0), (0.5,0.866,0), (0.5,0.289,0.816)
            # cut the apex (vertex 3); 3 new vertices replace it.
            v0 = (0, 0, 0)
            v1 = (1, 0, 0)
            v2 = (0.5, 0.866, 0)
            v3 = (0.5, 0.289, 0.816)  # apex (cut)
            # New vertices: along edges (0,3), (1,3), (2,3), at fraction (1-cut)
            def along(a, b, t):
                return tuple(a[k] * (1 - t) + b[k] * t for k in range(3))
            n0 = along(v0, v3, 1 - cut)
            n1 = along(v1, v3, 1 - cut)
            n2 = along(v2, v3, 1 - cut)
            verts = [v0, v1, v2, n0, n1, n2]
            edges = [
                (0, 1), (1, 2), (2, 0),  # base triangle
                (0, 3), (1, 4), (2, 5),  # edges from base to truncation
                (3, 4), (4, 5), (5, 3),  # truncation triangle
            ]
            return verts, edges
        return [(0, 0, 0)], []


if __name__ == "__main__":
    env = SolidFaceEdgeVertexQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
