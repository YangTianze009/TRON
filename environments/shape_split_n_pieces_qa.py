"""
Shape Split into N Pieces QA (W35, P1).

Renders a rectangle / equilateral triangle / regular polygon with 1-3 cut
lines (labelled EF, GH, etc.) overlaid. Question modes:
  (a) "additional edge length compared to the original perimeter is
      equivalent to which expression?" (Q1, Q4 style — answer expression)
  (b) "after cutting along the labelled lines, what are the dimensions of
      the largest remaining piece?" (Q2 style — cut largest square)
  (c) "the perimeter has decreased by how many cm" when N small congruent
      pieces are tiled to form a larger shape (Q18 style)

reference verbatim samples:
  Q1: "rectangular cake 5×3 cm, divided ... the total length of the
       additional edge length ... A. 4HG+4EF; B. 2HG+2EF; C. HG+EF; D. NCA"
       Ground truth: B
  Q2: "rectangular paper 8×5, cutting out the largest square ... remaining
       figure ... A. 5,3; B. 5,5; C. 3,3; D. NCA"   Ground truth: A

MCQ format: A/B/C/D (sometimes E. NCA). Verifier accepts \\boxed{X}.
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class ShapeSplitNPiecesQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "shape_split_n_pieces"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"modes": ["largest_square_remainder"]}
        if level <= 5:
            return {"modes": ["largest_square_remainder", "additional_edge"]}
        return {"modes": ["additional_edge", "perimeter_decrease",
                          "largest_square_remainder"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7331 + level * 53 + 5)

        mode = rng.choice(cfg["modes"])
        for _ in range(20):
            if mode == "largest_square_remainder":
                r = self._gen_largest_square(rng, level)
            elif mode == "additional_edge":
                r = self._gen_additional_edge(rng, level)
            else:
                r = self._gen_perimeter_decrease(rng, level)
            if r is not None:
                return r
        return None

    def _gen_largest_square(self, rng, level):
        # Pick rectangle l x w with l > w; cut out largest square of side w
        w = rng.randint(2, 6)
        l = w + rng.randint(1, 5)
        # Remaining piece: w x (l - w)
        rem_w = w
        rem_l = l - w
        correct = f"{rem_w}, {rem_l}" if rem_w > rem_l else f"{rem_l}, {rem_w}"
        # Sort dims for canonical form: smaller first or larger first
        dims_pair = tuple(sorted([rem_w, rem_l], reverse=True))
        correct = f"{dims_pair[0]}, {dims_pair[1]}"

        # Make MCQ options
        opts = [correct]
        opts.append(f"{w}, {w}")
        opts.append(f"{rem_l}, {rem_l}")
        opts.append(f"{l}, {w}")
        # Dedup and pad
        opts = list(dict.fromkeys(opts))[:4]
        while len(opts) < 4:
            extra = f"{w}, {rng.randint(1, 8)}"
            if extra not in opts:
                opts.append(extra)

        rng.shuffle(opts)
        labels = ["A", "B", "C", "D"]
        ans = labels[opts.index(correct)]

        question = (
            f"A rectangular piece of paper is {l} cm long and {w} cm wide. "
            f"After cutting out the largest possible square from it, the "
            f"dimensions of the remaining figure are (___) cm.\n\n"
            + "\n".join(f"{L}. {opt}" for L, opt in zip(labels, opts))
            + f"\n\nChoose the correct option (A, B, C, D)."
        )
        img = self._render_rect_cut_square(l, w)
        return question, ans, img

    def _gen_additional_edge(self, rng, level):
        # Q1 style: rectangle cut by 2 lines EF and GH, each cut adds
        # 2 * length cuts (one per side of the cut). For a rectangle cut
        # by horizontal line EF and vertical line GH:
        # additional edges = 2*EF + 2*GH (each cut creates two new edges
        # on the inner boundary).
        l = rng.randint(4, 10)
        w = rng.randint(3, 6)

        # Build options
        correct = f"2HG + 2EF"
        opts = [correct, "4HG + 4EF", "HG + EF", "3HG + 3EF"]
        rng.shuffle(opts)
        labels = ["A", "B", "C", "D"]
        ans = labels[opts.index(correct)]

        question = (
            f"As shown in the diagram, there is a rectangular cake measuring "
            f"{l} cm in length and {w} cm in width. If it is divided into four "
            f"parts along lines EF and GH, the total length of the additional "
            f"edge created compared to the original perimeter of the cake can "
            f"be equivalent to which of the following expressions?\n\n"
            + "\n".join(f"{L}. {opt}" for L, opt in zip(labels, opts))
            + f"\n\nChoose the correct option (A, B, C, D)."
        )
        img = self._render_rect_cut_cross(l, w)
        return question, ans, img

    def _gen_perimeter_decrease(self, rng, level):
        # Q18 style: equilateral triangle composed of N² small triangles tiled.
        # Perimeter of large = 3 * N * s; perimeter of all small if separated
        # = N² * 3 * s. Decrease = N² * 3s - 3Ns = 3s * (N² - N) = 3s * N * (N-1).
        # Actually small triangles before tiling all separate is (#small) * 3s
        # and after tiling forms big triangle 3*N*s. Decrease = 3sN² - 3sN
        # = 3sN(N-1).
        N = rng.choice([3, 4])
        n_small = N * N
        s = rng.choice([3, 4, 6])
        decrease = 3 * s * N * (N - 1)

        opts_set = set()
        opts_set.add(decrease)
        opts_set.add(decrease // 2)
        opts_set.add(decrease + 12)
        opts_set.add(int(decrease * 1.5))
        opts = list(opts_set)[:4]
        while len(opts) < 4:
            opts.append(decrease + rng.randint(5, 30))
        rng.shuffle(opts)
        labels = ["A", "B", "C", "D"]
        ans = labels[opts.index(decrease)]

        question = (
            f"As shown in the figure, the equilateral triangle ABC is composed "
            f"of {n_small} smaller equilateral triangles tiled together, each "
            f"with side length {s} cm. Compared to the total perimeter of all "
            f"the small triangles separately (before tiling), by how many cm "
            f"has the total perimeter decreased after tiling?\n\n"
            + "\n".join(f"{L}. {opt}" for L, opt in zip(labels, opts))
            + f"\n\nChoose the correct option (A, B, C, D)."
        )
        img = self._render_tri_grid(N)
        return question, ans, img

    # ------------------------------------------------------------------ #
    def _render_rect_cut_square(self, l, w) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Rectangle
        ax.add_patch(patches.Rectangle((0, 0), l, w,
                                       edgecolor="#2c3e50", facecolor="#ecf0f1",
                                       linewidth=2))
        # Largest square (w x w) on the left
        ax.add_patch(patches.Rectangle((0, 0), w, w,
                                       edgecolor="#d62728", facecolor="#fce8e6",
                                       linewidth=2, hatch="//"))
        # Remaining piece
        if l - w > 0:
            ax.add_patch(patches.Rectangle((w, 0), l - w, w,
                                           edgecolor="#27ae60",
                                           facecolor="#d4edda",
                                           linewidth=2))
        # Labels — only NAMES, not numeric dims (the model must compute
        # remaining dimensions from outer-rectangle dims labeled below).
        ax.text(w / 2, w / 2, "square", ha="center", va="center",
                fontsize=11, color="#2c3e50")
        ax.text(w + (l - w) / 2, w / 2, "remaining",
                ha="center", va="center", fontsize=11, color="#2c3e50")
        # Dim
        ax.annotate("", xy=(0, -0.5), xytext=(l, -0.5),
                    arrowprops=dict(arrowstyle="<->"))
        ax.text(l / 2, -0.9, f"{l} cm", ha="center", va="top", fontsize=10)
        ax.annotate("", xy=(-0.5, 0), xytext=(-0.5, w),
                    arrowprops=dict(arrowstyle="<->"))
        ax.text(-0.9, w / 2, f"{w} cm", ha="right", va="center", fontsize=10)

        ax.set_xlim(-2, l + 1)
        ax.set_ylim(-2, w + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)

    def _render_rect_cut_cross(self, l, w) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.add_patch(patches.Rectangle((0, 0), l, w,
                                       edgecolor="#2c3e50", facecolor="#ecf0f1",
                                       linewidth=2))
        # Vertical line GH
        gx = l / 2 + 0.5
        ax.plot([gx, gx], [0, w], color="#d62728", linewidth=2.4)
        ax.text(gx, w + 0.2, "G", ha="center", fontsize=12, color="#d62728",
                fontweight="bold")
        ax.text(gx, -0.4, "H", ha="center", fontsize=12, color="#d62728",
                fontweight="bold")
        # Horizontal line EF
        ey = w / 2
        ax.plot([0, l], [ey, ey], color="#27ae60", linewidth=2.4)
        ax.text(-0.4, ey, "E", ha="right", fontsize=12, color="#27ae60",
                fontweight="bold")
        ax.text(l + 0.4, ey, "F", ha="left", fontsize=12, color="#27ae60",
                fontweight="bold")

        ax.set_xlim(-1.5, l + 1.5)
        ax.set_ylim(-1, w + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)

    def _render_tri_grid(self, N) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Draw grid of small triangles
        # Use barycentric-ish coords. Just draw N rows of triangles.
        h = math.sqrt(3) / 2
        for row in range(N):
            for col in range(N - row):
                # Up-pointing triangle
                x0 = col + row / 2
                y0 = row * h
                pts_up = [(x0, y0), (x0 + 1, y0), (x0 + 0.5, y0 + h)]
                ax.add_patch(patches.Polygon(
                    pts_up, edgecolor="#2c3e50", facecolor="#ecf0f1",
                    linewidth=1.2))
                # Down-pointing if not the last column
                if col < N - row - 1:
                    pts_dn = [(x0 + 1, y0), (x0 + 1.5, y0 + h),
                              (x0 + 0.5, y0 + h)]
                    ax.add_patch(patches.Polygon(
                        pts_dn, edgecolor="#2c3e50", facecolor="#dfe6e9",
                        linewidth=1.2))
        # ABC labels
        ax.text(-0.3, -0.3, "B", ha="center", fontsize=14, fontweight="bold",
                color="#d62728")
        ax.text(N + 0.3, -0.3, "C", ha="center", fontsize=14, fontweight="bold",
                color="#d62728")
        ax.text(N / 2, N * h + 0.3, "A", ha="center", fontsize=14,
                fontweight="bold", color="#d62728")
        ax.set_xlim(-1, N + 1)
        ax.set_ylim(-1, N * h + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = ShapeSplitNPiecesQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
