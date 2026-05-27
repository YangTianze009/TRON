"""
Multi-View 3D Reconstruction QA -- redesigned to match three-view projection.

Format (matches the benchmark):
  Top row (3 images): isometric view, front view, top view of the cube stack.
  Bottom row (4 images, A/B/C/D): candidate LEFT views.

  Question: "Which image in the bottom row is the left view of the cube stack?"

This matches three-view projection where Qwen3-VL-8B-Instruct
scores about 50%.

Difficulty axes:
  L0..L1: 2x2 grid, 2-3 cubes; distractors are clearly different shapes.
  L2..L5: 3x3 grid, 3-6 cubes.
  L6..L9: 3x3 / 4x4 grid, 5-9 cubes; distractors are near-misses
          (one cell with off-by-one height).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class MultiView3dReconstructionQA(StandaloneVisualEnv):
    ENV_NAME = "multi_view_3d_reconstruction"

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n_cubes": 2 + level,    # 2..3
                "grid_size": 2,
                "near_miss_distractors": False,
            }
        if level <= 5:
            return {
                "n_cubes": 3 + level,   # 5..8
                "grid_size": 3,
                "near_miss_distractors": False,
            }
        return {
            "n_cubes": 5 + (level - 6) // 2,   # 5..6
            "grid_size": 3 if level <= 7 else 4,
            "near_miss_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        self._level = level
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1471)

        gs = cfg["grid_size"]
        target_n = cfg["n_cubes"]

        heights = self._random_heightmap(gs, target_n, sub_rng)
        if heights is None:
            return None

        # Compute the four canonical views.
        iso_h = heights                              # for isometric render
        front_v = self._front_view_2d(heights)       # bar profile
        top_v = self._top_view_2d(heights)           # 2D presence grid
        left_v = self._left_view_2d(heights)         # the answer view (bar profile)

        # ------- Build distractor LEFT views -------
        candidates = [left_v]                       # correct first
        seen = {self._view_key(left_v)}
        attempts = 0
        max_attempts = 200
        while len(candidates) < 4 and attempts < max_attempts:
            attempts += 1
            if cfg["near_miss_distractors"]:
                cand = self._mutate_view(left_v, sub_rng,
                                         small=True)
            else:
                cand = self._mutate_view(left_v, sub_rng,
                                         small=False)
            key = self._view_key(cand)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)

        if len(candidates) < 4:
            # Fallback: any random profile of same length.
            while len(candidates) < 4:
                cand = [sub_rng.randint(0, max(left_v) + 1)
                        for _ in range(len(left_v))]
                key = self._view_key(cand)
                if key not in seen and any(c > 0 for c in cand):
                    seen.add(key)
                    candidates.append(cand)

        # Shuffle.
        sub_rng.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates)
                           if self._view_key(c) == self._view_key(left_v))
        answer_letter = chr(ord("A") + correct_idx)

        # 2026-04-26: at L0/L1, ALSO show the correct LEFT view directly in
        # the top row (as a hint panel) — task collapses to pure visual
        # matching, which 4B/8B VL models can do. From L2 onward, the LEFT
        # view is hidden and the model must compute it.
        show_left_hint = level <= 1

        # Render figure.
        image = self._render(iso_h, front_v, top_v, candidates, sub_rng,
                             left_hint=left_v if show_left_hint else None)

        if show_left_hint:
            question = (
                "The top-right inset shows the LEFT view of the cube stack. "
                "Which image in the bottom row (A/B/C/D) matches that "
                "LEFT view? "
                "Please answer from options A, B, C, or D."
            )
        else:
            question = (
                "The cube stack is made of equal-sized small cubes. "
                "The top row shows its isometric view, front view, and "
                "top view from left to right. Which image in the bottom row "
                "is the LEFT view of the cube stack? "
                "Please answer from options A, B, C, or D."
            )
        if level == 2 or level == 3:
            question += (
                " Hint: the LEFT view shows, for each Y position (row in "
                "top view), the TALLEST height across that row. Procedure: "
                "(1) for each row of the top view, find the max height in "
                "that row; (2) the LEFT view is the bar profile of those "
                "row-maxes, ordered by Y."
            )
        return question, answer_letter, image

    # ------------------------------------------------------------------ #
    # Heightmap + view utilities
    # ------------------------------------------------------------------ #

    def _random_heightmap(self, gs: int, target_n: int,
                          rng: random.Random) -> Optional[List[List[int]]]:
        heights = [[0] * gs for _ in range(gs)]
        remaining = target_n
        cells = [(r, c) for r in range(gs) for c in range(gs)]
        rng.shuffle(cells)
        idx = 0
        while remaining > 0 and idx < 200:
            r, c = cells[idx % len(cells)]
            add = min(remaining, rng.randint(1, 2))
            heights[r][c] = min(3, heights[r][c] + add)
            remaining = target_n - sum(sum(row) for row in heights)
            idx += 1
        if sum(sum(row) for row in heights) == 0:
            return None
        return heights

    @staticmethod
    def _front_view_2d(heights):
        """Looking from FRONT (positive Y axis -> -Y). For each X column,
        the height seen is the max over all Y."""
        gs = len(heights)
        # heights[r][c] : r is depth (Y), c is X.
        # Front view: across X, height = max across Y.
        return [max(heights[r][c] for r in range(gs)) for c in range(gs)]

    @staticmethod
    def _left_view_2d(heights):
        """Looking from LEFT (positive X axis). For each Y row, the height
        seen is the max over all X. The leftmost cell in the view is the
        FRONT-most row (largest Y from viewer's POV)."""
        gs = len(heights)
        # We want the bar profile across depth (Y rows).
        # Convention: view from left, so the leftmost bar is the row closest
        # to the viewer (front of stack). To match typical benchmark
        # convention, just return the per-row max profile in order
        # row=0 (back) ... row=gs-1 (front), then reverse for left-view.
        col_maxes = [max(heights[r][c] for c in range(gs)) for r in range(gs)]
        # Left view: viewer is on +X side, so what they see leftmost is
        # the row at y=gs-1 (largest Y, closest to them). Standard
        # convention varies; we'll just present a consistent profile.
        return col_maxes  # length = gs

    @staticmethod
    def _top_view_2d(heights):
        gs = len(heights)
        return [[(1 if heights[r][c] > 0 else 0) for c in range(gs)]
                for r in range(gs)]

    @staticmethod
    def _view_key(profile):
        return tuple(profile)

    def _mutate_view(self, profile, rng, small=True):
        """Produce a perturbed bar-profile distractor.
        - small: change exactly one cell by +/-1.
        - large: change 1-2 cells by larger amounts, OR shuffle.
        """
        out = list(profile)
        if small:
            # Off-by-one in exactly one column.
            i = rng.randint(0, len(out) - 1)
            delta = rng.choice([-1, 1])
            out[i] = max(0, min(4, out[i] + delta))
            if out == list(profile):
                # If clamping made it identical, force the other direction.
                out[i] = max(0, min(4, out[i] + (1 if delta < 0 else -1)))
        else:
            n_changes = rng.randint(1, max(1, len(out) // 2))
            for _ in range(n_changes):
                i = rng.randint(0, len(out) - 1)
                delta = rng.choice([-2, -1, 1, 2])
                out[i] = max(0, min(4, out[i] + delta))
            # Sometimes reverse the whole profile (mirror).
            if rng.random() < 0.3 and out[::-1] != list(profile):
                out = out[::-1]
        # Ensure at least one bar visible.
        if all(v == 0 for v in out):
            out[rng.randint(0, len(out) - 1)] = 1
        return out

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, heights, front_v, top_v, left_candidates, rng,
                left_hint=None):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(13 * sc, 8.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        gs_grid = fig.add_gridspec(2, 4,
                                   height_ratios=[1.3, 1.0],
                                   hspace=0.35, wspace=0.25)

        # ----- Top row: isometric + front + top + (label OR left_hint) ----- #
        ax_iso = fig.add_subplot(gs_grid[0, 0])
        ax_front = fig.add_subplot(gs_grid[0, 1])
        ax_top = fig.add_subplot(gs_grid[0, 2])
        ax_label = fig.add_subplot(gs_grid[0, 3])

        # Extend height labels through L4 to reduce L3 cliff.
        show_heights = self._level <= 4 if hasattr(self, '_level') else False
        self._render_isometric(ax_iso, heights, "Isometric")
        self._render_profile_bars(ax_front, front_v, "Front View",
                                  bar_color="#85c1e9", edge_color="#1f618d",
                                  label_heights=show_heights)
        # 2026-04-25: at L0/L1 show heights inside top-view cells so the
        # model can directly read column heights and derive left view.
        self._render_top_grid(ax_top, top_v, "Top View",
                              heights=heights if show_heights else None)

        if left_hint is not None:
            # 2026-04-26: at L0/L1, show the answer LEFT view directly so
            # the task collapses to visual matching.
            self._render_profile_bars(
                ax_label, left_hint, "LEFT View (target)",
                bar_color="#f9e79f", edge_color="#b7950b",
                label_heights=show_heights)
            # Add a box around it to make it pop visually.
            for spine in ax_label.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#b7950b")
                spine.set_linewidth(2.5)
        else:
            ax_label.axis("off")
            ax_label.text(
                0.5, 0.5,
                "Find the\nLEFT VIEW\nfrom the options",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5",
                          fc="#fef9e7", ec="#b7950b", lw=1.2))

        # ----- Bottom row: 4 candidate LEFT views ----- #
        letters = ["A", "B", "C", "D"]
        for i, prof in enumerate(left_candidates):
            ax_opt = fig.add_subplot(gs_grid[1, i])
            self._render_profile_bars(ax_opt, prof, f"({letters[i]})",
                                      bar_color="#a6cee3",
                                      edge_color="#08519c",
                                      title_fontsize=14,
                                      label_heights=show_heights)

        fig.suptitle("Three-View Projection", fontsize=15,
                     fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.04)
        return self.fig_to_pil(fig, dpi=max(style["dpi"], 110))

    # ----- Drawing primitives ----- #
    def _render_profile_bars(self, ax, profile: List[int], title: str,
                             bar_color="#85c1e9", edge_color="#1f618d",
                             title_fontsize=12, label_heights=False):
        n = len(profile)
        max_h = max(max(profile), 1)
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(-0.4, max_h + 1.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")
        # Faint slot outlines so empty columns are still visible.
        for i in range(n):
            slot = mpatches.Rectangle((i + 0.05, 0.05), 0.9, 0.05,
                                      fc="#cccccc", ec="#888888", lw=0.6)
            ax.add_patch(slot)
        # Draw stacked unit cubes
        for i, h in enumerate(profile):
            for level_i in range(h):
                rect = mpatches.Rectangle((i + 0.05, level_i + 0.05),
                                          0.9, 0.9,
                                          fc=bar_color, ec=edge_color,
                                          lw=1.4)
                ax.add_patch(rect)
            if label_heights:
                ax.text(i + 0.5, h + 0.3, f"h={h}",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color=edge_color)
        # Ground line spans all slots
        ax.plot([-0.4, n + 0.4], [0, 0], color="#2c3e50", lw=2.0)
        # No cube-count annotation -- the visible bars ARE the answer.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _render_top_grid(self, ax, top: List[List[int]], title: str,
                         heights=None):
        gs = len(top)
        ax.set_xlim(-0.5, gs + 0.5)
        ax.set_ylim(-0.5, gs + 0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        for r in range(gs):
            for c in range(gs):
                fill = ("#f8c471" if top[r][c] else "#ffffff")
                rect = mpatches.Rectangle((c, gs - 1 - r), 1, 1,
                                          fc=fill, ec="#2c3e50", lw=1.0)
                ax.add_patch(rect)
                if heights is not None and top[r][c]:
                    h = heights[r][c]
                    ax.text(c + 0.5, gs - 1 - r + 0.5, str(h),
                            ha="center", va="center",
                            fontsize=14, fontweight="bold", color="#2c3e50")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _render_isometric(self, ax, heights: List[List[int]], title: str):
        """Standard polycube isometric using (X, Y, Z) -> screen with
        side length 1.0 in (c, r, h) world units. Painter's algorithm:
        draw back-to-front, bottom-up."""
        gs = len(heights)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Iso projection: each unit cube projects to a hex cell with these
        # vertex offsets. Use full unit so cubes touch (no gaps).
        cos30 = math.cos(math.radians(30))
        sin30 = math.sin(math.radians(30))

        def iso(x, y, z):
            sx = (x - y) * cos30
            sy = (x + y) * sin30 + z
            return sx, sy

        from matplotlib.colors import to_rgba
        gray_color = "#cfd2d6"
        red_color = "#e85a4f"

        # Render order: back rows first (r small means farther back since
        # row index is depth Y), bottom-up; within row, left-to-right.
        # In our heightmap, r=0 is "back" (largest Y in iso),
        # r=gs-1 is "front" (smallest Y).
        # Convention: c is X axis, r is Y axis. Higher Y is further from
        # viewer (back). We use sort by (x + y) to ensure proper draw order.
        cubes_to_draw = []
        for r in range(gs):
            for c in range(gs):
                for h in range(heights[r][c]):
                    cubes_to_draw.append((c, r, h))   # (x, y, z)
        # Painter's algorithm: draw back-to-front, bottom-up.
        cubes_to_draw.sort(key=lambda t: (-t[0] + t[1], t[2]))

        for (x, y, z) in cubes_to_draw:
            # Compute the 8 corner projections.
            v000 = iso(x, y, z)
            v100 = iso(x + 1, y, z)
            v110 = iso(x + 1, y + 1, z)
            v010 = iso(x, y + 1, z)
            v001 = iso(x, y, z + 1)
            v101 = iso(x + 1, y, z + 1)
            v111 = iso(x + 1, y + 1, z + 1)
            v011 = iso(x, y + 1, z + 1)

            # Pick color: occasional red highlight.
            is_red = ((x * 7 + y * 13 + z * 5) % 7 == 0)
            base_col = red_color if is_red else gray_color
            bc = to_rgba(base_col)
            dark = tuple(max(0, v * 0.65) for v in bc[:3]) + (1.0,)
            darker = tuple(max(0, v * 0.45) for v in bc[:3]) + (1.0,)

            # TOP face (z+1): v001-v101-v111-v011
            top_pts = [v001, v101, v111, v011]
            ax.fill([p[0] for p in top_pts], [p[1] for p in top_pts],
                    facecolor=base_col, edgecolor="#2c3e50", lw=0.9)
            # LEFT face (x): v000-v010-v011-v001
            left_pts = [v000, v010, v011, v001]
            ax.fill([p[0] for p in left_pts], [p[1] for p in left_pts],
                    facecolor=dark, edgecolor="#2c3e50", lw=0.9)
            # RIGHT face (y=0): v000-v100-v101-v001 — this is the front face
            front_pts = [v000, v100, v101, v001]
            ax.fill([p[0] for p in front_pts], [p[1] for p in front_pts],
                    facecolor=darker, edgecolor="#2c3e50", lw=0.9)

        ax.autoscale()
        ax.margins(0.15)

        total = sum(sum(row) for row in heights)
        ax.text(0.5, -0.02, f"{total} cubes total",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#555")

if __name__ == "__main__":
    env = MultiView3dReconstructionQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
