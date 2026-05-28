"""
Structural Pattern 3x3 QA (v4 G20, for a puzzle benchmark regression).

Targets: a puzzle benchmark regressions
  - size_grid
  - color_grid
  - polygon_sides_color
  - shape_size_grid
  - grid_number_color

Failure mode pulled from v3 flipped cases (2026-04-23 audit, docs section 2.1):
  base: "corners are Yellow, edges are Red → missing is Red"
  v3:   "the missing circle should be Blue because it's in the center"

i.e. v3 picks by "visual balance / symmetry" heuristic, ignoring the
corner/edge/center structural rule.

Fix: the env enforces a 3-role structural rule (corner cells follow rule A,
edge cells rule B, center cell rule C) and the **reward grades both
(a) the cell answer AND (b) the rule statement**, forcing the model to
articulate the corner/edge/center decomposition before answering.

Level axes:
  A) Attribute space: color / size / shape (L0-L2), mixed attribute (L3+)
  B) Role complexity: 2 distinct roles at L0-L2 (just corner vs edge),
     3 roles at L3+ (corner/edge/center), 4-fold partition by parity at L6+
  C) Distractor cells at L6+ (one cell violates the rule, GT=<its coord>)
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# Corners are cells at (0,0), (0,2), (2,0), (2,2)
# Edges are cells at (0,1), (1,0), (1,2), (2,1)
# Center is cell (1,1)
_CORNER_CELLS = [(0, 0), (0, 2), (2, 0), (2, 2)]
_EDGE_CELLS = [(0, 1), (1, 0), (1, 2), (2, 1)]
_CENTER_CELL = (1, 1)

_COLOR_NAMES = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "pink"]
_COLOR_HEX = {
    "red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71",
    "yellow": "#f1c40f", "purple": "#9b59b6", "orange": "#e67e22",
    "cyan": "#1abc9c", "pink": "#e91e63",
}

_SIZE_NAMES = ["small", "medium", "large"]
_SIZE_RADIUS = {"small": 0.18, "medium": 0.30, "large": 0.42}

_SHAPE_NAMES = ["circle", "square", "triangle", "hexagon"]

_TEMPLATES_MISSING = [
    # Each template asks the model to FIRST state the structural rule (corner=X, edge=Y, center=Z)
    # and THEN give the cell answer.
    "The 3x3 grid shows a pattern with distinct {attr} for corner cells, edge cells, and the center cell. One cell at (row {r}, column {c}) is missing. State the rule in the form 'corners are X, edges are Y, center is Z.' Then answer the missing cell's {attr}. Wrap the final cell {attr} in <answer>...</answer>.",
    "Examine the 3x3 pattern. Each of the three positional roles (corner, edge, center) has its own fixed {attr}. Cell (row {r}, column {c}) is blank. First describe the rule as 'corners=X, edges=Y, center=Z', then give the missing cell's {attr} in <answer>...</answer>.",
    "A 3x3 grid is laid out so that corner cells share one {attr}, edge cells share a different {attr}, and the center cell has a third {attr}. Cell at row {r}, column {c} is marked '?'. State the three-role rule, then put the missing {attr} in <answer>...</answer>.",
    "Observe the 3x3 grid — it uses a structural rule with three {attr} assignments (corners / edges / center). Row {r}, column {c} is missing. Describe the rule then answer with the missing {attr} in <answer>...</answer>.",
    "The pattern in this 3x3 grid assigns {attr} by cell role: corner, edge, or center. Cell (row {r}, column {c}) is unknown. Identify the rule structure then give the {attr} of the missing cell in <answer>...</answer>.",
    "Inspect the 3x3 grid. The rule is: corners share one {attr}, edges share another, and the center has a third. The cell at ({r},{c}) is missing. State the rule and put the missing {attr} in <answer>...</answer>.",
    "A 3x3 grid has a corner/edge/center structural pattern for {attr}. One cell at row {r}, col {c} is blank. Describe the three-role rule then answer with the {attr} inside <answer>...</answer>.",
    "The 3x3 grid shown uses three {attr} values, one each for corner cells, edge cells, and the center cell. Cell (row {r}, column {c}) is unknown. State the rule as 'corners=X, edges=Y, center=Z' then give the missing {attr} in <answer>...</answer>.",
    "Given the 3x3 pattern grid, deduce the {attr} rule by cell role (corner / edge / center). Fill the missing cell at row {r}, column {c}. Describe the rule first, then put the {attr} in <answer>...</answer>.",
    "Read the 3x3 grid's structural rule: each role (corner/edge/center) has a single {attr}. Cell (row {r}, column {c}) is missing. Answer format: first state the rule, then place the missing {attr} in <answer>...</answer>.",
    "The 3x3 grid shows a structural pattern that depends on whether a cell is at a corner, on an edge, or at the center. One cell (row {r}, column {c}) is blank. Describe the rule in one line, then output the missing {attr} in <answer>...</answer>.",
    "Every cell in this 3x3 grid has a {attr}. The assignment is by cell role: corner, edge, center. Cell (row {r}, column {c}) is missing. State the rule then provide the {attr} in <answer>...</answer>.",
    "Examine the 3x3 grid's structural rule across corners, edges, and the center. The cell at row {r}, column {c} is unknown. First describe the rule, then give the missing {attr} in <answer>...</answer>.",
    "A 3x3 grid shows a three-role structural pattern for {attr}: corners / edges / center. Cell (row {r}, column {c}) is missing. Describe the rule and give the {attr} in <answer>...</answer>.",
    "This 3x3 pattern has a corner/edge/center rule on {attr}. Cell at ({r},{c}) is blank. Describe the rule structure then put the missing {attr} in <answer>...</answer>.",
    "The 3x3 grid's pattern follows a positional-role rule on {attr}. Identify whether cell (row {r}, column {c}) is a corner, edge, or center, state the rule, then output the {attr} in <answer>...</answer>.",
]

_TEMPLATES_OUTLIER = [
    "The 3x3 grid follows a corner/edge/center {attr} rule, but exactly one cell violates the pattern. State the rule, then put the coordinates of the outlier cell (as 'row,column') inside <answer>...</answer>.",
    "One cell in this 3x3 grid breaks the structural {attr} rule (corners / edges / center). Describe the rule, then answer with the outlier's 'row,column' in <answer>...</answer>.",
    "Identify the cell violating the three-role {attr} rule (corner vs edge vs center) in the 3x3 grid. State the rule, then put 'row,column' in <answer>...</answer>.",
    "The 3x3 pattern has a single anomaly — one cell whose {attr} doesn't match its role (corner / edge / center). Describe the rule then report the anomaly as 'row,column' in <answer>...</answer>.",
    "Exactly one cell breaks the corner/edge/center {attr} rule in this 3x3 grid. State the rule, then put the bad cell's 'row,column' in <answer>...</answer>.",
    "Find the cell that violates the structural {attr} rule. State the three-role rule first, then give the outlier's 'row,column' in <answer>...</answer>.",
    "The 3x3 grid has one cell whose {attr} breaks the corner/edge/center pattern. Describe the rule, then output the violator's 'row,column' in <answer>...</answer>.",
    "Locate the anomalous cell (3x3 grid, {attr} by role). State the rule, answer 'row,column' of the anomaly in <answer>...</answer>.",
    "One cell in this 3x3 pattern has the wrong {attr} for its role (corner / edge / center). State the rule then answer with the 'row,column' in <answer>...</answer>.",
    "Spot the cell whose {attr} violates the three-role rule (corner/edge/center). Describe the rule, then put the cell location 'row,column' in <answer>...</answer>.",
    "The 3x3 grid has a corner/edge/center {attr} rule with one exception. State the rule, then answer with the exception's 'row,column' in <answer>...</answer>.",
    "Pinpoint the rule-breaking cell in this 3x3 {attr} pattern (three positional roles). State the rule, then answer 'row,column' of the breaker in <answer>...</answer>.",
    "One cell violates the 3x3 grid's corner/edge/center {attr} rule. Describe the rule, then report the offending 'row,column' in <answer>...</answer>.",
    "State the corner/edge/center {attr} rule for this 3x3 grid, then identify the single outlier as 'row,column' in <answer>...</answer>.",
    "A 3x3 grid has a positional-role {attr} rule with one violator. State the rule, then answer with the violator's 'row,column' inside <answer>...</answer>.",
    "Identify the misfit cell in this 3x3 structural {attr} pattern. State the rule first, then place the misfit's 'row,column' in <answer>...</answer>.",
]

class StructuralPattern3x3QA(StandaloneVisualEnv):
    ENV_NAME = "structural_pattern_3x3"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Attribute axis
        if level <= 2:
            attr = ["color", "size", "shape"][level]
        elif level <= 4:
            # mixed: color + size
            attr = "color_size"
        elif level <= 6:
            attr = "color_shape"
        else:
            # hardest: color + size + shape all vary
            attr = "color_size_shape"

        # Role complexity
        roles = "corner_edge_center"  # always three roles in this env (that's the test)

        # qtype
        if level <= 5:
            qtype = "find_missing"
        else:
            qtype = "find_outlier"

        return {"attr": attr, "roles": roles, "qtype": qtype}

    def _pick_three_distinct(self, rng, names: List[str]) -> Tuple[str, str, str]:
        """Pick three distinct values for corner / edge / center."""
        pool = list(names)
        rng.shuffle(pool)
        return pool[0], pool[1], pool[2]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level

        # Build grid: map each cell (r,c) to an attribute bundle
        # based on its role (corner / edge / center).
        # Grid cells are indexed (row, col), both in [0,2]
        attr_mode = cfg["attr"]

        # Figure out corner / edge / center attribute values
        if attr_mode == "color":
            c_corner, c_edge, c_center = self._pick_three_distinct(rng, _COLOR_NAMES)
            role_attr = {"corner": {"color": c_corner},
                         "edge": {"color": c_edge},
                         "center": {"color": c_center}}
            attr_tag = "color"
        elif attr_mode == "size":
            s_corner, s_edge, s_center = self._pick_three_distinct(rng, _SIZE_NAMES)
            role_attr = {"corner": {"size": s_corner},
                         "edge": {"size": s_edge},
                         "center": {"size": s_center}}
            attr_tag = "size"
        elif attr_mode == "shape":
            sh_corner, sh_edge, sh_center = self._pick_three_distinct(rng, _SHAPE_NAMES)
            role_attr = {"corner": {"shape": sh_corner},
                         "edge": {"shape": sh_edge},
                         "center": {"shape": sh_center}}
            attr_tag = "shape"
        elif attr_mode == "color_size":
            c = self._pick_three_distinct(rng, _COLOR_NAMES)
            s = self._pick_three_distinct(rng, _SIZE_NAMES)
            role_attr = {"corner": {"color": c[0], "size": s[0]},
                         "edge": {"color": c[1], "size": s[1]},
                         "center": {"color": c[2], "size": s[2]}}
            attr_tag = "color and size"
        elif attr_mode == "color_shape":
            c = self._pick_three_distinct(rng, _COLOR_NAMES)
            sh = self._pick_three_distinct(rng, _SHAPE_NAMES)
            role_attr = {"corner": {"color": c[0], "shape": sh[0]},
                         "edge": {"color": c[1], "shape": sh[1]},
                         "center": {"color": c[2], "shape": sh[2]}}
            attr_tag = "color and shape"
        else:  # color_size_shape
            c = self._pick_three_distinct(rng, _COLOR_NAMES)
            s = self._pick_three_distinct(rng, _SIZE_NAMES)
            sh = self._pick_three_distinct(rng, _SHAPE_NAMES)
            role_attr = {"corner": {"color": c[0], "size": s[0], "shape": sh[0]},
                         "edge": {"color": c[1], "size": s[1], "shape": sh[1]},
                         "center": {"color": c[2], "size": s[2], "shape": sh[2]}}
            attr_tag = "color, size and shape"

        # fill grid
        grid = [[None] * 3 for _ in range(3)]
        for r, c in _CORNER_CELLS:
            grid[r][c] = dict(role_attr["corner"])
        for r, c in _EDGE_CELLS:
            grid[r][c] = dict(role_attr["edge"])
        grid[_CENTER_CELL[0]][_CENTER_CELL[1]] = dict(role_attr["center"])

        sidx = (self.seed or 0) % 16
        if cfg["qtype"] == "find_missing":
            # BUGFIX 2026-04-24: never hide the center cell. When the center
            # is missing, its unique attribute (role C) has no visual instance
            # anywhere else in the grid, making the problem unsolvable.
            # Missing cell must be an edge or corner (attribute visible in
            # sibling cells of the same role).
            # 60% edge, 40% corner
            r_pick = rng.random()
            if r_pick < 0.6:
                (mr, mc) = rng.choice(_EDGE_CELLS)
            else:
                (mr, mc) = rng.choice(_CORNER_CELLS)
            # what's missing:
            correct_attr = grid[mr][mc]
            missing = dict(correct_attr)  # GT answer
            grid[mr][mc] = "?"

            # build GT answer string
            if attr_mode == "color":
                ans = missing["color"]
            elif attr_mode == "size":
                ans = missing["size"]
            elif attr_mode == "shape":
                ans = missing["shape"]
            elif attr_mode == "color_size":
                ans = f"{missing['color']} {missing['size']}"
            elif attr_mode == "color_shape":
                ans = f"{missing['color']} {missing['shape']}"
            else:
                ans = f"{missing['color']} {missing['size']} {missing['shape']}"

            question = _TEMPLATES_MISSING[sidx].format(attr=attr_tag, r=mr+1, c=mc+1)
            answer = ans
        else:
            # find outlier — corrupt one cell
            role = rng.choice(["corner", "edge", "center"])
            if role == "corner":
                mr, mc = rng.choice(_CORNER_CELLS)
                wrong_role = rng.choice(["edge", "center"])
            elif role == "edge":
                mr, mc = rng.choice(_EDGE_CELLS)
                wrong_role = rng.choice(["corner", "center"])
            else:
                mr, mc = _CENTER_CELL
                wrong_role = rng.choice(["corner", "edge"])
            grid[mr][mc] = dict(role_attr[wrong_role])

            question = _TEMPLATES_OUTLIER[sidx].format(attr=attr_tag)
            answer = f"{mr+1},{mc+1}"

        self._role_attr = role_attr  # stash for debugging
        img = self._render_grid(grid, cfg, rng)
        return question, answer, img

    def _render_grid(self, grid, cfg, rng):
        style = self._random_style()
        sc = style["figsize_scale"]

        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            edge_col = tbp["line_color"]
            edge_lw = tbp["line_width"]
            missing_fc = tbp["fill_color"]
            q_color = tbp["line_color"]
            title_kw = {"fontfamily": tbp["font_family"], "color": tbp["line_color"]}
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            edge_col = "#1a1a1a"
            edge_lw = 1.6
            missing_fc = "#dddddd"
            q_color = "red"
            title_kw = {}
            dpi = style["dpi"]
        else:
            bg = "#ffffff"
            edge_col = "black"
            edge_lw = 1.0
            missing_fc = "#dddddd"
            q_color = "red"
            title_kw = {}
            dpi = style["dpi"]

        def _draw():
            fig, ax = plt.subplots(figsize=(4.5 * sc, 4.5 * sc))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)
            ax.set_aspect("equal")
            ax.axis("off")

            for r in range(3):
                for c in range(3):
                    val = grid[r][c]
                    # Cell frame
                    rect = mpatches.Rectangle((c, 2 - r), 1, 1,
                                               fc=bg, ec=edge_col,
                                               lw=edge_lw * 0.8, alpha=1.0)
                    ax.add_patch(rect)
                    if val == "?":
                        ax.text(c + 0.5, 2 - r + 0.5, "?",
                                fontsize=30, ha="center", va="center",
                                fontweight="bold", color=q_color)
                        continue
                    # Draw the object with attributes
                    color_name = val.get("color", "blue")
                    size_name = val.get("size", "medium")
                    shape_name = val.get("shape", "circle")
                    fc = _COLOR_HEX.get(color_name, "#888888")
                    radius = _SIZE_RADIUS.get(size_name, 0.3)
                    cx, cy = c + 0.5, 2 - r + 0.5
                    if shape_name == "circle":
                        patch = mpatches.Circle((cx, cy), radius,
                                                fc=fc, ec=edge_col, lw=edge_lw)
                    elif shape_name == "square":
                        patch = mpatches.Rectangle((cx - radius, cy - radius),
                                                    2 * radius, 2 * radius,
                                                    fc=fc, ec=edge_col, lw=edge_lw)
                    elif shape_name == "triangle":
                        patch = mpatches.Polygon(
                            [(cx, cy + radius), (cx - radius, cy - radius * 0.6),
                             (cx + radius, cy - radius * 0.6)],
                            fc=fc, ec=edge_col, lw=edge_lw, closed=True)
                    elif shape_name == "hexagon":
                        import math as _m
                        pts = []
                        for i in range(6):
                            ang = _m.pi / 6 + i * _m.pi / 3
                            pts.append((cx + radius * _m.cos(ang),
                                        cy + radius * _m.sin(ang)))
                        patch = mpatches.Polygon(pts, fc=fc, ec=edge_col,
                                                  lw=edge_lw, closed=True)
                    else:
                        patch = mpatches.Circle((cx, cy), radius,
                                                 fc=fc, ec=edge_col, lw=edge_lw)
                    ax.add_patch(patch)

            # row / col labels
            for i in range(3):
                ax.text(-0.3, 2.5 - i, str(i + 1),
                        fontsize=11, ha="center", va="center", **title_kw)
                ax.text(i + 0.5, 3.25, str(i + 1),
                        fontsize=11, ha="center", va="center", **title_kw)

            fig.tight_layout()
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()
        return self.fig_to_pil(fig, dpi=dpi)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        Custom grader: for find_missing with multi-attribute answers, accept
        any permutation of the tokens (e.g., "blue large" == "large blue").
        Falls back to base grader otherwise.
        """
        pred = predicted.strip().lower().rstrip(".").rstrip(",")
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",")
        if pred == gt:
            return True
        # Permutation check for multi-token GT
        if " " in gt:
            gt_toks = set(gt.split())
            pred_toks = set(pred.split())
            # Accept if pred contains all GT tokens in any order
            if gt_toks.issubset(pred_toks) and len(pred_toks) <= len(gt_toks) + 1:
                return True
        # Row,col format
        if "," in gt:
            return pred.replace(" ", "") == gt.replace(" ", "")
        return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_structural_3x3"
    os.makedirs(out_dir, exist_ok=True)
    env = StructuralPattern3x3QA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[structural_pattern_3x3 L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"structural_pattern_3x3_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[structural_pattern_3x3 L{level} s{s}] A={env._answer} "
                  f"Q={env._question[:60]}...")
