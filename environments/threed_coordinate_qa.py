"""3D Coordinate QA environment.

Task families:
  - distance: distance between two labeled points P and Q
  - midpoint: coordinates of midpoint M of segment PQ
  - vector_length: magnitude of vector PQ
  - dot_product: dot product of vectors OP and OQ
  - farthest: among 3 labeled points, which is farthest from origin
  - closest: among 3 labeled points, which is closest to reference point
  - collinear_check (L8+): given 3 points A,B,C, are they collinear?
  - dominant_axis (L8+): along which axis does |PQ| change the most?

Difficulty axes:
  * L0-L2: small coord range, only distance/midpoint questions, axis-aligned
  * L3-L5: moderate range, add vector/dot product
  * L6-L9: wide range, 3-4 point configurations, add collinearity and axis questions

All numeric values used by the question appear AS LABELS ON THE IMAGE.
Question text refers to points by letter (P/Q/etc.), never embeds coordinates.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_MARKER_CHOICES = ["o", "s", "^", "D", "P", "*", "X", "v"]
_VIEW_PRESETS = [
    (20, 45), (25, 60), (30, 30), (35, 75), (15, 110),
    (40, 20), (25, 135), (30, 160), (18, 200), (28, 245),
]

class ThreeDCoordinateQA(StandaloneVisualEnv):
    ENV_NAME = "threed_coordinate"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    QUESTION_TYPES = [
        "distance", "midpoint", "vector_length",
        "dot_product", "farthest", "closest",
        "collinear_check", "dominant_axis",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {
                "qtypes": ["distance", "midpoint"],
                "coord_range": 4,
                "n_points": 2,
                "axis_aligned": True,
                "integer_answer": True,
            }
        if level <= 3:
            return {
                "qtypes": ["distance", "midpoint", "vector_length"],
                "coord_range": 6,
                "n_points": 2,
                "axis_aligned": False,
                "integer_answer": False,
            }
        if level <= 5:
            return {
                "qtypes": ["distance", "midpoint", "vector_length", "dot_product"],
                "coord_range": 9,
                "n_points": 2,
                "axis_aligned": False,
                "integer_answer": False,
            }
        if level <= 7:
            return {
                "qtypes": ["distance", "vector_length", "dot_product",
                           "farthest", "closest"],
                "coord_range": 12,
                "n_points": 3,
                "axis_aligned": False,
                "integer_answer": False,
            }
        return {
            "qtypes": ["dot_product", "farthest", "closest",
                       "collinear_check", "dominant_axis"],
            "coord_range": 15,
            "n_points": 3,
            "axis_aligned": False,
            "integer_answer": False,
        }

    # ------------------------------------------------------------------ #
    # Main generator
    # ------------------------------------------------------------------ #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Sub-RNG explicitly includes level so L0 and L9 get different seeds.
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2609)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for attempt in range(25):
            result = self._try_generate(sub_rng, cfg, qtype, level)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, cfg, qtype, level):
        cr = cfg["coord_range"]
        n_pts = cfg["n_points"]
        axis_aligned = cfg["axis_aligned"]

        # Generate point positions
        def _rand_point():
            if axis_aligned:
                # axis-aligned: fix one coordinate to 0 for simpler spatial reasoning
                a = rng.randint(1, cr)
                b = rng.randint(1, cr)
                c = 0
                pts_list = [a, b, c]
                rng.shuffle(pts_list)
                return tuple(pts_list)
            return (rng.randint(-cr, cr),
                    rng.randint(-cr, cr),
                    rng.randint(-cr, cr))

        # Point letters randomized from a pool
        letter_pool = ["P", "Q", "R", "S", "A", "B", "C", "M", "N", "K"]
        rng.shuffle(letter_pool)
        letters = letter_pool[:n_pts]
        points = {L: _rand_point() for L in letters}
        # ensure pairwise distinct
        while len(set(points.values())) < len(points):
            points = {L: _rand_point() for L in letters}

        # Dispatch by qtype
        if qtype == "distance":
            if n_pts < 2:
                return None
            L1, L2 = letters[0], letters[1]
            p1, p2 = points[L1], points[L2]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            if cfg.get("integer_answer") and axis_aligned:
                # pick configurations where distance is integer-ish
                d_rounded = round(d, 2)
                answer = f"{d_rounded:.2f}"
            else:
                answer = f"{d:.2f}"
            q = (f"The figure shows labeled points in 3D space with their "
                 f"coordinates written beside each point. "
                 f"Find the Euclidean distance between point {L1} and point {L2}. "
                 f"Round your answer to 2 decimal places.")

        elif qtype == "midpoint":
            if n_pts < 2:
                return None
            L1, L2 = letters[0], letters[1]
            p1, p2 = points[L1], points[L2]
            mx = round((p1[0] + p2[0]) / 2, 2)
            my = round((p1[1] + p2[1]) / 2, 2)
            mz = round((p1[2] + p2[2]) / 2, 2)
            answer = f"({self._fmt(mx)}, {self._fmt(my)}, {self._fmt(mz)})"
            q = (f"The figure shows labeled points in 3D space with their "
                 f"coordinates written on the image. "
                 f"What are the coordinates of the midpoint of segment {L1}{L2}? "
                 f"Answer as (x, y, z). Round each coordinate to 2 decimals.")

        elif qtype == "vector_length":
            if n_pts < 2:
                return None
            L1, L2 = letters[0], letters[1]
            p1, p2 = points[L1], points[L2]
            v = tuple(b - a for a, b in zip(p1, p2))
            mag = math.sqrt(sum(c * c for c in v))
            answer = f"{mag:.2f}"
            q = (f"The figure shows labeled points in 3D space. The vector {L1}{L2} "
                 f"goes from {L1} to {L2}. "
                 f"Using the coordinates shown on the image, what is the magnitude "
                 f"(length) of vector {L1}{L2}? Round to 2 decimals.")

        elif qtype == "dot_product":
            if n_pts < 2:
                return None
            L1, L2 = letters[0], letters[1]
            p1, p2 = points[L1], points[L2]
            dp = sum(a * b for a, b in zip(p1, p2))
            answer = str(int(dp))
            q = (f"The figure shows labeled points in 3D space with their "
                 f"coordinate values. "
                 f"Treat O{L1} and O{L2} as position vectors from the origin. "
                 f"Compute the dot product O{L1} · O{L2}. "
                 f"Answer with an integer.")

        elif qtype == "farthest":
            if n_pts < 3:
                return None
            # Which of the 3 points is farthest from origin?
            dists = {L: math.sqrt(sum(c * c for c in p)) for L, p in points.items()}
            sorted_ds = sorted(dists.items(), key=lambda kv: kv[1])
            # Need clear separation between closest two and farthest
            if sorted_ds[-1][1] - sorted_ds[-2][1] < 1.0:
                return None
            answer = sorted_ds[-1][0]
            letters_list = ", ".join(letters)
            q = (f"The figure shows three labeled points ({letters_list}) in 3D space "
                 f"with their coordinates written beside each. "
                 f"Which point is farthest from the origin (0,0,0)? "
                 f"Answer with just the letter.")

        elif qtype == "closest":
            if n_pts < 3:
                return None
            # Which of the 3 points is closest to a 4th reference point?
            # The reference point is also drawn and labeled 'O_ref' on the image.
            ref = _rand_point()
            while ref in points.values():
                ref = _rand_point()
            dists = {L: math.sqrt(sum((c - r) ** 2 for c, r in zip(p, ref)))
                     for L, p in points.items()}
            sorted_ds = sorted(dists.items(), key=lambda kv: kv[1])
            if sorted_ds[1][1] - sorted_ds[0][1] < 1.0:
                return None
            answer = sorted_ds[0][0]
            letters_list = ", ".join(letters)
            points["T"] = ref  # target reference point on image
            q = (f"The figure shows three candidate points ({letters_list}) and "
                 f"a target point labeled T, all with coordinates written on the image. "
                 f"Which of {letters_list} is the closest to T? "
                 f"Answer with just the letter.")

        elif qtype == "collinear_check":
            if n_pts < 3:
                return None
            # Randomly decide yes/no then construct accordingly
            collinear = rng.random() < 0.5
            L1, L2, L3 = letters[0], letters[1], letters[2]
            if collinear:
                # pick a line direction + two offsets
                dirv = (rng.randint(-3, 3), rng.randint(-3, 3), rng.randint(-3, 3))
                if all(c == 0 for c in dirv):
                    dirv = (1, 2, -1)
                base = (rng.randint(-cr, cr), rng.randint(-cr, cr), rng.randint(-cr, cr))
                t1 = rng.choice([-2, -1, 1, 2])
                t2 = rng.choice([-3, -2, 2, 3])
                while t2 == t1:
                    t2 = rng.choice([-3, -2, 2, 3])
                points[L1] = base
                points[L2] = tuple(base[i] + t1 * dirv[i] for i in range(3))
                points[L3] = tuple(base[i] + t2 * dirv[i] for i in range(3))
                answer = "yes"
            else:
                # just use random and verify non-collinear
                for _try in range(10):
                    p1 = _rand_point()
                    p2 = _rand_point()
                    p3 = _rand_point()
                    v1 = tuple(p2[i] - p1[i] for i in range(3))
                    v2 = tuple(p3[i] - p1[i] for i in range(3))
                    # cross product magnitude > 0
                    cx = v1[1] * v2[2] - v1[2] * v2[1]
                    cy = v1[2] * v2[0] - v1[0] * v2[2]
                    cz = v1[0] * v2[1] - v1[1] * v2[0]
                    if cx * cx + cy * cy + cz * cz > 4:
                        points[L1], points[L2], points[L3] = p1, p2, p3
                        break
                else:
                    return None
                answer = "no"
            # Ensure within view bounds
            for L in (L1, L2, L3):
                p = points[L]
                if any(abs(c) > cr + 6 for c in p):
                    return None
            q = (f"The figure shows three points {L1}, {L2}, {L3} in 3D space with "
                 f"their coordinates written beside them. "
                 f"Are these three points collinear (do they lie on one single line)? "
                 f"Answer yes or no.")

        elif qtype == "dominant_axis":
            if n_pts < 2:
                return None
            L1, L2 = letters[0], letters[1]
            p1, p2 = points[L1], points[L2]
            diffs = [abs(p2[i] - p1[i]) for i in range(3)]
            sorted_diffs = sorted(enumerate(diffs), key=lambda kv: kv[1], reverse=True)
            if sorted_diffs[0][1] - sorted_diffs[1][1] < 2:
                return None
            axis_names = ["x", "y", "z"]
            answer = axis_names[sorted_diffs[0][0]]
            q = (f"The figure shows two labeled points {L1} and {L2} in 3D space. "
                 f"From {L1} to {L2}, along which coordinate axis does the change "
                 f"|Δ| have the largest magnitude? Answer with a single letter: x, y, or z.")
        else:
            return None

        img = self._render(points, letters, qtype, rng)
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #

    def _render(self, points, letters, qtype, rng):
        style = self._random_style()
        palette = style["palette"]
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(style["bg_color"])

        # Random view angle for diversity
        elev, azim = rng.choice(_VIEW_PRESETS)
        ax.view_init(elev=elev, azim=azim)

        # Axes
        all_coords = list(points.values())
        if not all_coords:
            return None
        xs = [p[0] for p in all_coords] + [0]
        ys = [p[1] for p in all_coords] + [0]
        zs = [p[2] for p in all_coords] + [0]
        margin = 2
        ax_lim = max(abs(min(xs) - margin), abs(max(xs) + margin),
                     abs(min(ys) - margin), abs(max(ys) + margin),
                     abs(min(zs) - margin), abs(max(zs) + margin))
        ax.set_xlim(-ax_lim, ax_lim)
        ax.set_ylim(-ax_lim, ax_lim)
        ax.set_zlim(-ax_lim, ax_lim)

        # Draw origin marker
        ax.scatter([0], [0], [0], color="#444", s=45, marker="+",
                   linewidths=2, zorder=6)
        ax.text(0, 0, -0.3, "O(0,0,0)", fontsize=9, color="#444", zorder=7)

        # Plot each point with a unique color + marker
        for idx, L in enumerate(letters + ["T"] if "T" in points else letters):
            if L not in points:
                continue
            p = points[L]
            color = palette[idx % len(palette)]
            marker = _MARKER_CHOICES[(idx + hash(L)) % len(_MARKER_CHOICES)]
            size = 110 if L == "T" else 100
            ax.scatter(*p, color=color, s=size, marker=marker,
                       edgecolor="black", linewidth=1.0, zorder=5)
            label_txt = f"{L}({p[0]},{p[1]},{p[2]})"
            ax.text(p[0] + 0.3, p[1] + 0.3, p[2] + 0.4,
                    label_txt, fontsize=10, color=color,
                    fontweight="bold", zorder=6)

        # Connect first two points with dashed line if question involves pair
        if qtype in ("distance", "midpoint", "vector_length", "dominant_axis"):
            if len(letters) >= 2:
                L1, L2 = letters[0], letters[1]
                p1, p2 = points[L1], points[L2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color="black", linestyle="--", linewidth=1.5, zorder=3)

        # Collinearity: connect all 3 with thin line
        if qtype == "collinear_check" and len(letters) >= 3:
            pts_sorted = [points[L] for L in letters[:3]]
            xs2 = [p[0] for p in pts_sorted]
            ys2 = [p[1] for p in pts_sorted]
            zs2 = [p[2] for p in pts_sorted]
            ax.plot(xs2, ys2, zs2, color="gray", linestyle=":",
                    linewidth=1.2, alpha=0.6, zorder=2)

        # dot product: draw OP and OQ vectors
        if qtype == "dot_product" and len(letters) >= 2:
            for i, L in enumerate(letters[:2]):
                p = points[L]
                col = palette[i % len(palette)]
                ax.quiver(0, 0, 0, p[0], p[1], p[2], color=col,
                          arrow_length_ratio=0.08, linewidth=2, alpha=0.8)

        # farthest: draw radius to farthest point
        if qtype == "farthest":
            for L in letters[:3]:
                p = points[L]
                ax.plot([0, p[0]], [0, p[1]], [0, p[2]],
                        color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)
        ax.set_zlabel("Z", fontsize=11)
        title_variants = [
            "3D Coordinate Problem",
            "Points in 3D Space",
            "3D Geometry Figure",
            "Coordinate Diagram (3D)",
        ]
        ax.set_title(title_variants[rng.randrange(len(title_variants))],
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fmt(v):
        """Format number: drop .0 for integer-valued floats."""
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:.2f}"

    # ------------------------------------------------------------------ #
    # Answer checking override for coordinate tuples
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import re
        p = predicted.strip().lower().rstrip(".")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        # Tuple comparison
        tup_pat = r'\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?'
        pm = re.match(tup_pat, p)
        gm = re.match(tup_pat, g)
        if pm and gm:
            try:
                pv = [float(pm.group(i)) for i in (1, 2, 3)]
                gv = [float(gm.group(i)) for i in (1, 2, 3)]
                return all(abs(a - b) < 0.1 for a, b in zip(pv, gv))
            except Exception:
                pass
        # Scalar numeric with tolerance
        try:
            pn = float(p)
            gn = float(g)
            return abs(pn - gn) < 0.1
        except ValueError:
            pass
        return super()._check_answer(predicted, ground_truth)
