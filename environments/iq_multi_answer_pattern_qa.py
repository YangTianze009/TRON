"""
Single-answer rule-induction IQ pattern test (R5 B4B rewrite, 2026-05-05).

Two example grids on the left illustrate a hidden rule. Among 4
candidate grids on the right, exactly ONE follows the same rule.
Output: a single letter A/B/C/D (or A/B/C/D/E with "No correct" trap
at L6+).

Each cell is a small 3x3 sub-grid that may contain colored shapes
(triangles / circles / squares). Rule families:
  - "diagonal cells contain triangles"
  - "all cells in the top row are circles"
  - "each row has at least one circle"
  - "exactly N cells are colored"
  - "all four corners contain circles"
  - "center cell is colored"
  - "left column is all squares"
  - "no squares anywhere"
  - "bottom rows have a triangle each"

Per-level config:
  L0: 4 options (1 correct, 3 weak distractors), simplest rules only.
  L1-L5: 4 options, medium-strength distractors.
  L6-L7: 5 options (E = "No correct"), 30% trap, harder rules.
  L8-L9: 5 options, 40% trap, all rules + tighter distractors.

Verifier: standalone_base built-in (single MCQ letter, parent _check_answer).
"""
import math
import random
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_QUESTION_STEMS = [
    "The two grids on the left both follow the same hidden rule. Which ONE of the four options on the right also follows that rule?",
    "Two example grids share a common rule. Find the ONE grid among the options that follows the same rule.",
    "The two grids on the left obey one rule. Identify which option obeys the same rule.",
    "Look at the two example grids on the left. Among the options on the right, which ONE follows the same rule?",
    "Based on the two examples on the left, which option matches the same hidden pattern?",
    "Two example grids on the left define a rule. Pick the ONE option that satisfies the same rule.",
    "Identify the rule the two left-hand grids share. Which option also follows this rule?",
    "Find the rule from the two example grids and pick the option that matches.",
    "Two example grids share a common pattern. Choose the option that matches the pattern.",
    "The pair of example grids defines a hidden rule. Which option satisfies the rule?",
    "Examine the two example grids carefully. One of the options follows the same pattern. Which one?",
    "The left pair of grids both share a rule. Which option shares it as well?",
    "Determine the rule between the two left grids; choose the right-hand grid that obeys it.",
    "Two grids show a common rule. Among the options, which one also satisfies it?",
    "Inspect the example grids. Pick the option that follows the same rule.",
    "The two example grids illustrate a rule. Find the option that follows it.",
]


# Cell content: each cell is a 3x3 sub-grid with shapes at some positions.
# We model a 'grid' as a 3x3 tensor of shape strings or '' (empty).
SHAPE_OPTS = ["triangle", "circle", "square"]
GRID_SIZE = 3


def _empty_grid():
    return [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


# --------- Rule definitions: each rule is (sample fn, check fn). ---------- #
# sample(rng) -> grid satisfying the rule
# check(grid) -> bool

def _rule_diagonal_triangles():
    """Main diagonal cells contain triangles; off-diagonal varies."""
    def sample(rng):
        g = _empty_grid()
        for i in range(GRID_SIZE):
            g[i][i] = "triangle"
        # Add a few random off-diagonal shapes, but NOT triangles
        non_tri = ["circle", "square"]
        for _ in range(rng.randint(0, 2)):
            r, c = rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE)
            if r != c:
                g[r][c] = rng.choice(non_tri)
        return g
    def check(g):
        for i in range(GRID_SIZE):
            if g[i][i] != "triangle":
                return False
        return True
    return ("diagonal_triangles", sample, check)


def _rule_top_row_circles():
    """All cells in the top row are circles."""
    def sample(rng):
        g = _empty_grid()
        for c in range(GRID_SIZE):
            g[0][c] = "circle"
        # Other rows random
        non_circ = ["triangle", "square"]
        for _ in range(rng.randint(0, 3)):
            r = rng.randint(1, GRID_SIZE - 1)
            c = rng.randrange(GRID_SIZE)
            g[r][c] = rng.choice(non_circ)
        return g
    def check(g):
        for c in range(GRID_SIZE):
            if g[0][c] != "circle":
                return False
        return True
    return ("top_row_circles", sample, check)


def _rule_blue_triangles_each_bottom_lines():
    """Each of the bottom two rows contains at least one triangle.
    (Mirrors qid 5 reasoning: 'Blue triangles in each of the two bottom lines.')
    """
    def sample(rng):
        g = _empty_grid()
        for r in (1, 2):
            c = rng.randrange(GRID_SIZE)
            g[r][c] = "triangle"
            # Add other shapes randomly
            for _ in range(rng.randint(0, 1)):
                cc = rng.randrange(GRID_SIZE)
                if g[r][cc] == "":
                    g[r][cc] = rng.choice(["circle", "square"])
        # Top row maybe empty or has some shapes
        for _ in range(rng.randint(0, 2)):
            cc = rng.randrange(GRID_SIZE)
            g[0][cc] = rng.choice(SHAPE_OPTS)
        return g
    def check(g):
        for r in (1, 2):
            if not any(c == "triangle" for c in g[r]):
                return False
        return True
    return ("triangles_in_bottom_rows", sample, check)


def _rule_exact_n_squares(N):
    def sample(rng):
        g = _empty_grid()
        cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
        rng.shuffle(cells)
        for k in range(N):
            r, c = cells[k]
            g[r][c] = "square"
        # Add a couple of non-squares for visual richness
        non_sq = ["triangle", "circle"]
        for _ in range(rng.randint(0, 2)):
            r, c = rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE)
            if g[r][c] == "":
                g[r][c] = rng.choice(non_sq)
        return g
    def check(g):
        cnt = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if g[r][c] == "square")
        return cnt == N
    return (f"exact_{N}_squares", sample, check)


def _rule_each_row_has_circle():
    def sample(rng):
        g = _empty_grid()
        for r in range(GRID_SIZE):
            c = rng.randrange(GRID_SIZE)
            g[r][c] = "circle"
            for _ in range(rng.randint(0, 1)):
                cc = rng.randrange(GRID_SIZE)
                if g[r][cc] == "":
                    g[r][cc] = rng.choice(["triangle", "square"])
        return g
    def check(g):
        for r in range(GRID_SIZE):
            if not any(g[r][c] == "circle" for c in range(GRID_SIZE)):
                return False
        return True
    return ("each_row_has_circle", sample, check)


def _rule_no_squares():
    def sample(rng):
        g = _empty_grid()
        non_sq = ["triangle", "circle"]
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE)
            g[r][c] = rng.choice(non_sq)
        return g
    def check(g):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if g[r][c] == "square":
                    return False
        return True
    return ("no_squares", sample, check)


def _rule_corner_circles():
    """All four corners contain circles."""
    def sample(rng):
        g = _empty_grid()
        for (r, c) in [(0, 0), (0, GRID_SIZE - 1),
                        (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]:
            g[r][c] = "circle"
        # Random middles (not circles)
        for _ in range(rng.randint(0, 2)):
            r, c = rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE)
            if (r, c) not in [(0, 0), (0, GRID_SIZE - 1),
                               (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]:
                g[r][c] = rng.choice(["triangle", "square"])
        return g
    def check(g):
        for (r, c) in [(0, 0), (0, GRID_SIZE - 1),
                        (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]:
            if g[r][c] != "circle":
                return False
        return True
    return ("corner_circles", sample, check)


def _rule_center_filled():
    """Center cell is colored (any shape)."""
    def sample(rng):
        g = _empty_grid()
        g[1][1] = rng.choice(SHAPE_OPTS)
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE)
            if (r, c) != (1, 1):
                g[r][c] = rng.choice(SHAPE_OPTS)
        return g
    def check(g):
        return g[1][1] != ""
    return ("center_filled", sample, check)


def _rule_left_col_squares():
    def sample(rng):
        g = _empty_grid()
        for r in range(GRID_SIZE):
            g[r][0] = "square"
        for _ in range(rng.randint(0, 3)):
            r = rng.randrange(GRID_SIZE)
            c = rng.randint(1, GRID_SIZE - 1)
            g[r][c] = rng.choice(["triangle", "circle"])
        return g
    def check(g):
        for r in range(GRID_SIZE):
            if g[r][0] != "square":
                return False
        return True
    return ("left_col_squares", sample, check)


_ALL_RULES = [
    _rule_diagonal_triangles,
    _rule_top_row_circles,
    _rule_blue_triangles_each_bottom_lines,
    _rule_each_row_has_circle,
    _rule_corner_circles,
    _rule_center_filled,
    _rule_left_col_squares,
    _rule_no_squares,
    lambda: _rule_exact_n_squares(2),
    lambda: _rule_exact_n_squares(3),
    lambda: _rule_exact_n_squares(4),
]


class IQMultiAnswerPatternQA(StandaloneVisualEnv):
    ENV_NAME = "iq_multi_answer_pattern"

    # 2026-05-05 R5 B4B: orientation-sensitive grid layout.
    ALLOW_ROTATION = False

    # Simple rules (visually obvious for low levels)
    _SIMPLE_RULES = [
        _rule_diagonal_triangles,
        _rule_top_row_circles,
        _rule_corner_circles,
        _rule_left_col_squares,
        _rule_no_squares,
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            # L0/L1: easy rules only, 4-MCQ, 1 correct, weak distractors
            return {
                "n_options": 4, "rules": self._SIMPLE_RULES,
                "use_5opt": False, "trap_rate": 0.0,
                "distractor_strength": "weak",
            }
        if level <= 3:
            return {
                "n_options": 4, "rules": _ALL_RULES,
                "use_5opt": False, "trap_rate": 0.0,
                "distractor_strength": "medium",
            }
        if level <= 5:
            return {
                "n_options": 4, "rules": _ALL_RULES,
                "use_5opt": False, "trap_rate": 0.0,
                "distractor_strength": "medium",
            }
        if level <= 7:
            return {
                "n_options": 5, "rules": _ALL_RULES,
                "use_5opt": True, "trap_rate": 0.30,
                "distractor_strength": "medium",
            }
        return {
            "n_options": 5, "rules": _ALL_RULES,
            "use_5opt": True, "trap_rate": 0.40,
            "distractor_strength": "strong",
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 9001 + level * 13 + 71)

        for attempt in range(40):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        rule_factory = rng.choice(cfg["rules"])
        rule_name, sample_fn, check_fn = rule_factory()

        # Two example grids that satisfy the rule
        examples = [sample_fn(rng), sample_fn(rng)]

        # 2026-05-05 R5 B4B: trap mode — all 4 visible options violate the rule;
        # correct answer is "E. No correct".
        use_trap = cfg["use_5opt"] and (rng.random() < cfg["trap_rate"])

        if use_trap:
            distractors = []
            tries = 0
            while len(distractors) < 4 and tries < 60:
                tries += 1
                g = self._random_grid(rng)
                if not check_fn(g) and g not in distractors and g not in examples:
                    distractors.append(g)
            if len(distractors) < 4:
                return None
            all_opts = distractors[:4]
            rng.shuffle(all_opts)
            answer = "E"
        else:
            # Pick 1 correct option (different from examples)
            correct_g = None
            for _ in range(40):
                g = sample_fn(rng)
                if g not in examples:
                    correct_g = g
                    break
            if correct_g is None:
                return None

            # Pick 3 distractors that violate the rule
            distractors = []
            tries = 0
            while len(distractors) < 3 and tries < 60:
                tries += 1
                g = self._random_grid(rng)
                if not check_fn(g):
                    if g not in distractors and g != correct_g and g not in examples:
                        distractors.append(g)
            if len(distractors) < 3:
                return None

            all_opts = [correct_g] + distractors
            rng.shuffle(all_opts)
            for i, g in enumerate(all_opts):
                if g == correct_g:
                    answer = chr(ord("A") + i)
                    break

        # Build question
        letter_set = "A/B/C/D/E" if cfg["use_5opt"] else "A/B/C/D"
        q = rng.choice(_QUESTION_STEMS)
        q += f" Reply with the single letter ({letter_set}) inside <answer>...</answer>. Example: <answer>B</answer>"
        if cfg["use_5opt"]:
            q += " (Option E means none of A-D follows the rule.)"
        if level == 0:
            q += f" Hint: rule = '{rule_name}'. Answer: {answer}."
        elif level == 1:
            q += " Hint: identify the property both example grids share, then check options. Be concise."
        elif level <= 3:
            q += " Hint: find the common visual property of the examples (rows/cols/corners/counts/colors). Be concise."

        img = self._render(examples, all_opts, rng, has_E_label=cfg["use_5opt"])
        return q, answer, img

    def _random_grid(self, rng):
        g = _empty_grid()
        for _ in range(rng.randint(2, 5)):
            r = rng.randrange(GRID_SIZE)
            c = rng.randrange(GRID_SIZE)
            g[r][c] = rng.choice(SHAPE_OPTS + [""])
        return g

    # -------------------- rendering -------------------- #
    def _draw_grid(self, ax, cx, cy, grid, cell_size=0.4):
        # cx,cy is the center of the small 3x3 grid
        # Each sub-cell is cell_size; total grid spans 3*cell_size.
        total = GRID_SIZE * cell_size
        x0 = cx - total / 2
        y0 = cy - total / 2
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                gx = x0 + c * cell_size + cell_size / 2
                gy = y0 + (GRID_SIZE - 1 - r) * cell_size + cell_size / 2
                # Cell background
                rect = mpatches.Rectangle(
                    (gx - cell_size / 2 + 0.01, gy - cell_size / 2 + 0.01),
                    cell_size - 0.02, cell_size - 0.02,
                    facecolor="#fafafa", edgecolor="#9e9e9e", linewidth=0.6)
                ax.add_patch(rect)
                shape = grid[r][c]
                if shape:
                    s = cell_size * 0.30
                    sx, sy = gx, gy
                    if shape == "triangle":
                        pts = [(sx, sy + s), (sx - s, sy - s * 0.85),
                               (sx + s, sy - s * 0.85)]
                        ax.add_patch(mpatches.Polygon(
                            pts, closed=True, facecolor="#1a5276",
                            edgecolor="#212121", linewidth=0.8))
                    elif shape == "circle":
                        ax.add_patch(mpatches.Circle(
                            (sx, sy), s, facecolor="#c0392b",
                            edgecolor="#212121", linewidth=0.8))
                    elif shape == "square":
                        ax.add_patch(mpatches.Rectangle(
                            (sx - s, sy - s), 2 * s, 2 * s,
                            facecolor="#196f3d", edgecolor="#212121", linewidth=0.8))

    def _render(self, examples, options, rng, has_E_label=False) -> Image.Image:
        fig = plt.figure(figsize=(10, 5.5))
        fig.patch.set_facecolor("#ffffff")
        # Layout: top row = 2 examples (left), 4 options (right) split into top
        # and bottom of options for clarity. Use 2 axes (top: examples; bottom:
        # 4 options as a row).
        ax_top = fig.add_subplot(2, 1, 1); ax_bot = fig.add_subplot(2, 1, 2)
        for ax in (ax_top, ax_bot):
            ax.set_facecolor("#ffffff"); ax.set_aspect("equal"); ax.axis("off")
        # Examples: 2 cells horizontally
        cell = 1.7
        for i, g in enumerate(examples):
            cx = i * cell
            cy = 0
            # Box around the grid
            ax_top.add_patch(mpatches.Rectangle(
                (cx - cell / 2 + 0.05, cy - cell / 2 + 0.05),
                cell - 0.1, cell - 0.1,
                facecolor="#ffffff", edgecolor="#34495e", linewidth=1.2))
            self._draw_grid(ax_top, cx, cy, g, cell_size=0.4)
            ax_top.text(cx, cy - cell / 2 - 0.18, f"Example {i + 1}",
                        fontsize=11, ha="center", va="top")
        ax_top.set_xlim(-cell, cell * 2)
        ax_top.set_ylim(-cell, cell)
        ax_top.set_title("Examples (both follow the same rule)", fontsize=12)

        # Options: 4 cells horizontally + optional E label
        for i, g in enumerate(options):
            cx = i * cell
            cy = 0
            ax_bot.add_patch(mpatches.Rectangle(
                (cx - cell / 2 + 0.05, cy - cell / 2 + 0.05),
                cell - 0.1, cell - 0.1,
                facecolor="#ffffff", edgecolor="#34495e", linewidth=1.2))
            self._draw_grid(ax_bot, cx, cy, g, cell_size=0.4)
            ax_bot.text(cx, cy - cell / 2 - 0.18, f"({chr(ord('A') + i)})",
                        fontsize=12, fontweight="bold",
                        ha="center", va="top")
        n_opts = len(options)
        if has_E_label:
            cx = n_opts * cell
            cy = 0
            ax_bot.add_patch(mpatches.Rectangle(
                (cx - cell / 2 + 0.05, cy - cell / 2 + 0.05),
                cell - 0.1, cell - 0.1,
                facecolor="#fafafa", edgecolor="#34495e", linewidth=1.2,
                linestyle="--"))
            ax_bot.text(cx, cy, "No correct\nanswer",
                        fontsize=10, ha="center", va="center",
                        style="italic", color="#1a1a1a")
            ax_bot.text(cx, cy - cell / 2 - 0.18, "(E)",
                        fontsize=12, fontweight="bold",
                        ha="center", va="top")
            ax_bot.set_xlim(-cell, cell * (n_opts + 1))
        else:
            ax_bot.set_xlim(-cell, cell * n_opts)
        ax_bot.set_ylim(-cell, cell)
        ax_bot.set_title("Options (pick the one that follows the same rule)",
                          fontsize=12)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, hspace=0.4)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_iq_multi"
    os.makedirs(out_dir, exist_ok=True)
    env = IQMultiAnswerPatternQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']}")
