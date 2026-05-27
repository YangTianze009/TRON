"""
Colored Grid Rotation QA (v4 G17).

Targets: spatial-vision task.2DRotation -3.75.

Task form (directly mirrors the spatial-vision 2DRotation prompt style):
  Left image shows a colored grid with a red square marking a corner.
  Which of 3 candidate grids is obtained by rotating the left grid only
  (no flipping or other changes)?

Reward: MCQ letter exact match.

Level axes:
  A) Grid size: 2x2 at L0 -> 5x5 at L9
  B) Rotation angle: 90 at L0-3, 180 at L4-6, 270 at L7+ mixed with all
  C) Distractor complexity: random rotation vs actual flip-and-rotate at L5+
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_COLOR_POOL = [
    "#e74c3c",  # red (reserved for marker)
    "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#16a085", "#c0392b",
    "#8e44ad", "#2980b9", "#27ae60", "#d35400",
    "#f1c40f", "#7f8c8d",
]
# Visually-distinct primary colors used at L0-L2 — avoids the model
# confusing "red" with "orange/dark-red" when the random pool draws
# similar-hue cells.
_DISTINCT_PRIMARY = [
    "#e74c3c",  # red
    "#3498db",  # blue
    "#2ecc71",  # green
    "#f1c40f",  # yellow
]

def _rotate_grid(grid: List[List[str]], deg: int) -> List[List[str]]:
    n = len(grid)
    if deg == 0:
        return [row[:] for row in grid]
    if deg == 90:
        # CCW 90: new[r][c] = old[c][n-1-r]
        return [[grid[c][n - 1 - r] for c in range(n)] for r in range(n)]
    if deg == 180:
        return [[grid[n - 1 - r][n - 1 - c] for c in range(n)] for r in range(n)]
    if deg == 270:
        return [[grid[n - 1 - c][r] for c in range(n)] for r in range(n)]
    return grid

def _flip_grid(grid: List[List[str]], axis: str) -> List[List[str]]:
    n = len(grid)
    if axis == "h":
        return [row[::-1] for row in grid]
    if axis == "v":
        return grid[::-1]
    return [row[:] for row in grid]

def _grids_equal(g1, g2) -> bool:
    return all(g1[r][c] == g2[r][c] for r in range(len(g1)) for c in range(len(g1)))

_TEMPLATES = [
    "The left grid is a colored {n}x{n} pattern with a red square marking one corner. After rotating the left grid {rot}° {dir}, which option (A, B, or C) is the result? Put the letter in <answer>...</answer>.",
    "A colored {n}x{n} grid (left, red corner marker). Rotate it {rot}° {dir} — which option matches? A, B, or C in <answer>...</answer>.",
    "Left: colored {n}x{n} grid with red corner. Rotate it {rot}° {dir}. Which of A, B, C is the result? Letter in <answer>...</answer>.",
    "Reference grid has red corner. Apply a {rot}° {dir} rotation. Which option (A/B/C) matches the rotated grid? Letter in <answer>...</answer>.",
    "Rotate the left grid {rot}° {dir}. Which of A, B, or C is the rotated result? Put A/B/C in <answer>...</answer>.",
    "Left grid has a red corner marker. After rotating {rot}° {dir}, which one is the result? A/B/C in <answer>...</answer>.",
    "Apply a rotation of {rot}° {dir} to the colored grid (left). Which candidate (A, B, C) is the result? Letter in <answer>...</answer>.",
    "Among A, B, C, which is the left grid rotated by {rot}° {dir}? Put the letter in <answer>...</answer>.",
    "The left grid rotated {rot}° {dir} is one of A, B, C. Which letter? <answer>...</answer>.",
    "Which of A, B, C is the left grid after a {rot}° {dir} rotation? Letter in <answer>...</answer>.",
    "Left: reference colored grid with red corner. Apply {rot}° {dir} rotation. Pick the result among A, B, C. Letter in <answer>...</answer>.",
    "Find the result of rotating the left grid {rot}° {dir} among options A, B, C. Letter in <answer>...</answer>.",
    "The reference grid (left) rotated {rot}° {dir} = which option A/B/C? Letter in <answer>...</answer>.",
    "Among options A, B, and C, which is the left grid after a {rot}° {dir} rotation? Letter in <answer>...</answer>.",
    "Left is a reference grid. Rotate {rot}° {dir}. Which of A, B, C matches? Letter in <answer>...</answer>.",
    "Pick the {rot}° {dir} rotation of the left grid from A, B, C. Letter in <answer>...</answer>.",
]

class ColoredGridRotationQA(StandaloneVisualEnv):
    ENV_NAME = "colored_grid_rotation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        grid_sz = 2 + level // 2
        if level <= 3:
            rotations = [90]
        elif level <= 6:
            rotations = [90, 180]
        else:
            rotations = [90, 180, 270]
        use_flip_distractor = level >= 5
        return {"grid_size": grid_sz, "rotations": rotations,
                 "use_flip_distractor": use_flip_distractor}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 307)
        self._primary_complexity_feature = level

        n = cfg["grid_size"]
        # Pick a marker corner
        marker_pos = rng.choice([(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)])

        # At L0-L2, draw from 4 visually-distinct primaries so the model
        # can read cell colors unambiguously.
        if level <= 2:
            color_pool = list(_DISTINCT_PRIMARY)
            non_red_colors = color_pool[1:]
        else:
            non_red_colors = _COLOR_POOL[1:]
        rng.shuffle(non_red_colors)
        grid = [["" for _ in range(n)] for _ in range(n)]
        k = 0
        for r in range(n):
            for c in range(n):
                if (r, c) == marker_pos:
                    grid[r][c] = _COLOR_POOL[0]  # red marker
                else:
                    grid[r][c] = non_red_colors[k % len(non_red_colors)]
                    k += 1

        # Correct answer: rotate by a random angle from cfg
        rot_deg = rng.choice(cfg["rotations"])
        correct = _rotate_grid(grid, rot_deg)

        # Distractors: at L0 always use a flip-based distractor (visually
        # different, unambiguously not a rotation). Higher levels mix flip
        # + wrong-rotation distractors. Identity (0°) excluded — it would
        # show the same grid as the reference and is a degenerate option.
        distractors = []
        attempts = 0
        while len(distractors) < 2 and attempts < 30:
            attempts += 1
            if cfg["use_flip_distractor"] and rng.random() < 0.5:
                flipped = _flip_grid(grid, rng.choice(["h", "v"]))
                d_grid = _rotate_grid(flipped, rng.choice([0, 90, 180, 270]))
            elif level <= 2:
                # always a flip distractor — clearly not a rotation
                flipped = _flip_grid(grid, rng.choice(["h", "v"]))
                d_grid = _rotate_grid(flipped, rng.choice([0, 90, 180, 270]))
            else:
                # wrong rotation (excluding the correct one and 0°)
                other_rots = [r for r in [90, 180, 270] if r != rot_deg]
                d_grid = _rotate_grid(grid, rng.choice(other_rots))
            if _grids_equal(d_grid, correct):
                continue
            if _grids_equal(d_grid, grid):
                continue   # identical to ref — degenerate
            if any(_grids_equal(d_grid, d) for d in distractors):
                continue
            distractors.append(d_grid)
        if len(distractors) < 2:
            return None

        options = [correct] + distractors
        rng.shuffle(options)
        correct_idx = next(i for i, g in enumerate(options) if _grids_equal(g, correct))
        letter = "ABC"[correct_idx]

        sidx = (self.seed or 0) % 16
        # Display angle/direction in the question (CCW only — easier to verbalize)
        q = _TEMPLATES[sidx].format(n=n, rot=rot_deg, dir="counterclockwise")
        # 2026-04-25: at low levels, add a tracking hint pointing to the
        # red corner marker as the reference for rotation verification.
        if level <= 2:
            q += (
                " Hint: the red corner cell in the reference is the easiest "
                "to track. Determine which corner the red cell ends up at "
                "after rotation, then check which option has the red cell "
                "at that destination corner. (90° CCW: top-right → top-left → "
                "bottom-left → bottom-right; 180°: opposite corner; 270° CCW: "
                "reverse of 90° CCW.)"
            )

        img = self._render(grid, options, marker_pos, n, rng)
        return q, letter, img

    def _render(self, ref_grid, options, marker_pos, n, rng):
        # Dimensions scale with grid size. Each grid has n*cell_size width.
        cell_size = 0.7
        grid_w = n * cell_size
        gap = 1.0
        total_w = grid_w * 4 + gap * 4 + 0.5
        fig, ax = plt.subplots(figsize=(total_w * 0.9, max(4, n * cell_size + 1.5)))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, total_w)
        ax.set_ylim(-1.2, n * cell_size + 0.8)
        ax.set_aspect("equal")
        ax.axis("off")

        def draw_grid(grid, cx, cy, label, is_reference=False):
            for r in range(n):
                for c in range(n):
                    color = grid[r][c]
                    # Marker cell (the red one) — thicker black outline for emphasis
                    is_marker = (color == _COLOR_POOL[0])
                    rect = mpatches.Rectangle(
                        (cx + c * cell_size, cy + (n - 1 - r) * cell_size),
                        cell_size, cell_size,
                        fc=color,
                        ec="black",
                        lw=2.2 if is_marker else 1.0)
                    ax.add_patch(rect)
            # Label in a separated box below
            label_y = cy - 0.5
            ax.text(cx + n * cell_size / 2, label_y, label,
                    fontsize=14, ha="center", fontweight="bold")

        # Reference at left
        draw_grid(ref_grid, 0.4, 0.3, "Reference", is_reference=True)
        # A, B, C options
        base_x = grid_w + gap * 1.5
        for i, opt in enumerate(options):
            draw_grid(opt, base_x + i * (grid_w + gap), 0.3, "ABC"[i])

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_cgr"
    os.makedirs(out_dir, exist_ok=True)
    env = ColoredGridRotationQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 147
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[cgr L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/cgr_s{s}_L{level}.png")
            print(f"[cgr L{level} s{s}] A={env._answer}")
