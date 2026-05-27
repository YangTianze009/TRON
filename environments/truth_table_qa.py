"""
Truth Table QA environment.

Round 2 diversity + difficulty fix (2026-04-16):
- 4+ question templates per qtype
- 4 variable-name pools (P/Q, A/B, X/Y, p/q)
- Level-gated operator sets (L0: only AND/OR; L9: all ops + 3-var chains)
- Render: grid style jitter, header color cycling, font size jitter
- L0 vs L9 structurally: L0 is 2-var with simple op; L9 is 3-var nested.
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_OPS = {
    "AND": lambda a, b: a and b,
    "OR": lambda a, b: a or b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: not (a and b),
    "NOR": lambda a, b: not (a or b),
    "IMPLIES": lambda a, b: (not a) or b,
}

_VAR_POOLS = [
    ["P", "Q", "R"],
    ["A", "B", "C"],
    ["X", "Y", "Z"],
    ["p", "q", "r"],
]

_TITLE_VARIANTS = ["Truth Table", "Logic Truth Table", "Boolean Truth Table",
                    "Propositional Truth Table"]

class TruthTableQA(StandaloneVisualEnv):
    ENV_NAME = "truth_table"

    QUESTION_TYPES = [
        "fill_missing", "which_row_true", "count_true",
        "count_false", "evaluate_row", "identify_expression",
        "is_tautology",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration (L0 simpler than L9 structurally)
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 1:
            # TOO_EASY fix: at L0-L1 the question text is the ONLY task prompt,
            # so qtypes that inline the operator + variable values (like
            # `evaluate_row` -> "Evaluate p AND q with p=1, q=0") are pure text
            # tasks and bypass the image. Restrict to `fill_missing`
            # (must read the '?' cell) and `count_true` (must count the
            # result column). Both force looking at the table.
            return {"qtypes": ["fill_missing", "count_true"],
                    "n_vars_choices": [2],
                    "ops": ["AND", "OR"],
                    "gate_header": False}
        if level <= 3:
            # Same concern as L0-L1: evaluate_row / which_row_true inline the
            # operator and values in text. Keep them out until gate_header is
            # enabled (L6+), so the model must use the image.
            return {"qtypes": ["fill_missing", "count_true", "count_false"],
                    "n_vars_choices": [2],
                    "ops": ["AND", "OR", "XOR"],
                    "gate_header": False}
        if level <= 5:
            return {"qtypes": ["fill_missing", "count_true", "count_false"],
                    "n_vars_choices": [2, 3],
                    "ops": ["AND", "OR", "XOR", "NAND"],
                    "gate_header": False}
        if level <= 7:
            # L6-L7: replace operator text with a graphical gate symbol
            # in the header; identify_expression MUST be visual.
            return {"qtypes": ["fill_missing", "count_true",
                                "count_false", "identify_expression",
                                "is_tautology"],
                    "n_vars_choices": [2, 3],
                    "ops": list(_OPS.keys()),
                    "gate_header": True}
        # L8-L9: hardest — 3-var, full op set, gate header (no op text)
        return {"qtypes": self.QUESTION_TYPES,
                "n_vars_choices": [3],
                "ops": list(_OPS.keys()),
                "gate_header": True}

    # ------------------------------------------------------------------ #

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 937)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        n_vars = sub_rng.choice(cfg["n_vars_choices"])
        op_name = sub_rng.choice(cfg["ops"])
        op_fn = _OPS[op_name]
        var_pool = sub_rng.choice(_VAR_POOLS)

        if n_vars == 2:
            var_names = var_pool[:2]
            rows = [(p, q) for p in [0, 1] for q in [0, 1]]
        else:
            var_names = var_pool[:3]
            rows = [(p, q, r) for p in [0, 1] for q in [0, 1] for r in [0, 1]]

        # Compute results
        if n_vars == 2:
            expr_label = f"{var_names[0]} {op_name} {var_names[1]}"
            results = [int(op_fn(bool(r[0]), bool(r[1]))) for r in rows]
        else:
            expr_label = (f"({var_names[0]} {op_name} {var_names[1]}) "
                           f"{op_name} {var_names[2]}")
            results = [int(op_fn(op_fn(bool(r[0]), bool(r[1])),
                                  bool(r[2]))) for r in rows]

        # Pick a hidden cell
        hidden_idx = sub_rng.randint(0, len(rows) - 1)
        hidden_val = results[hidden_idx]

        # Decide whether to hide the operator from the question text.
        hide_op_in_text = cfg.get("gate_header", False)
        expr_label_for_q = ("the operation shown in the header"
                            if hide_op_in_text else expr_label)
        if qtype == "fill_missing":
            tmpls = [
                "In the truth table shown, one result cell is marked '?'. What value (0 or 1) should replace the '?'?",
                "The truth table in the image has one missing result. What value belongs there (0 or 1)?",
                "Fill in the '?' cell shown in the truth table. Answer 0 or 1.",
                "What is the result value for the row with '?' in the image? Answer 0 or 1.",
            ]
            q = sub_rng.choice(tmpls)
            a = str(hidden_val)

        elif qtype == "which_row_true":
            true_rows = [i for i, r in enumerate(results) if r == 1]
            if not true_rows:
                q = f"How many rows make the expression {expr_label} true?"
                a = "0"
            else:
                target = sub_rng.choice(true_rows)
                row_vals = ", ".join(
                    f"{var_names[j]}={rows[target][j]}" for j in range(n_vars))
                tmpls = [
                    f"Is the expression {expr_label_for_q} true when {row_vals}? Answer 1 for true, 0 for false.",
                    f"Given {row_vals}, is {expr_label_for_q} true (1) or false (0)?",
                    f"Evaluate {expr_label_for_q} when {row_vals} — answer 1 or 0.",
                ]
                q = sub_rng.choice(tmpls)
                a = "1"
                hidden_idx = -1

        elif qtype == "count_true":
            ct = sum(results)
            tmpls = [
                f"How many rows in the truth table produce a result of 1 (true)?",
                f"Count the rows in the table with output 1.",
                f"From the truth table shown, how many rows have a true result?",
                f"Report the number of rows where {expr_label_for_q} is true.",
            ]
            q = sub_rng.choice(tmpls)
            a = str(ct)
            hidden_idx = -1

        elif qtype == "count_false":
            cf = sum(1 for r in results if r == 0)
            tmpls = [
                f"How many rows in the truth table produce a result of 0 (false)?",
                f"Count the rows in the table where {expr_label_for_q} is false.",
                f"From the truth table shown, how many rows output 0?",
                f"Report the number of false rows for {expr_label_for_q}.",
            ]
            q = sub_rng.choice(tmpls)
            a = str(cf)
            hidden_idx = -1

        elif qtype == "evaluate_row":
            idx = sub_rng.randint(0, len(rows) - 1)
            row_vals = ", ".join(
                f"{var_names[j]}={rows[idx][j]}" for j in range(n_vars))
            tmpls = [
                f"What is the result of {expr_label_for_q} when {row_vals}?",
                f"Evaluate {expr_label_for_q} with {row_vals}.",
                f"For {row_vals}, compute {expr_label_for_q}.",
                f"Given {row_vals}, what does {expr_label_for_q} equal (0 or 1)?",
            ]
            q = sub_rng.choice(tmpls)
            a = str(results[idx])
            hidden_idx = -1

        elif qtype == "identify_expression":
            # Force 2 vars for identify
            if n_vars != 2:
                n_vars = 2
                var_names = var_pool[:2]
                rows = [(p, q) for p in [0, 1] for q in [0, 1]]
                results = [int(op_fn(bool(r[0]), bool(r[1]))) for r in rows]
            expr_label = f"{var_names[0]} ? {var_names[1]}"
            tmpls = [
                (f"The truth table shows the results for an unknown "
                  f"operation between {var_names[0]} and {var_names[1]}. "
                  f"Which logical operation produces these results? "
                  f"Choose from: AND, OR, XOR, NAND, NOR, IMPLIES."),
                (f"Identify the operation shown between {var_names[0]} "
                  f"and {var_names[1]}. Options: AND, OR, XOR, NAND, NOR, IMPLIES."),
                (f"From the results column, determine the logical operation. "
                  f"Reply with one of: AND, OR, XOR, NAND, NOR, IMPLIES."),
            ]
            q = sub_rng.choice(tmpls)
            a = op_name
            hidden_idx = -1

        elif qtype == "is_tautology":
            all_true = all(r == 1 for r in results)
            tmpls = [
                (f"Is the expression {expr_label_for_q} a tautology (true for all "
                  f"possible input combinations)? Answer yes or no."),
                (f"Does {expr_label_for_q} produce 1 for every row? Answer yes or no."),
                (f"Is {expr_label_for_q} a tautology? (yes/no)"),
            ]
            q = sub_rng.choice(tmpls)
            a = "yes" if all_true else "no"
            hidden_idx = -1

        else:
            return None

        img = self._render(sub_rng, var_names, expr_label, rows, results,
                           hidden_idx, gate_op=op_name,
                           gate_header=cfg.get("gate_header", False))
        return q, a, img

    # ------------------------------------------------------------------ #
    # Renderer
    # ------------------------------------------------------------------ #

    def _draw_gate_symbol(self, ax, cx, cy, op_name, color, fs):
        """Draw a small logic-gate symbol at (cx, cy).
        Simple silhouette shapes: each op gets a distinct silhouette."""
        import matplotlib.patches as mpatches_local
        # All shapes inscribed in a box of size 0.85 x 0.5 centered at (cx, cy).
        w = 0.85
        h = 0.5
        if op_name in ("AND", "NAND"):
            # D-shape (flat-left, semicircle right)
            rect = mpatches_local.FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w * 0.55, h,
                boxstyle="square,pad=0", facecolor="none",
                edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            arc = mpatches_local.Wedge(
                (cx - w / 2 + w * 0.55, cy), h / 2,
                -90, 90, facecolor="none", edgecolor=color, linewidth=2)
            ax.add_patch(arc)
        elif op_name in ("OR", "NOR", "XOR"):
            # Shield-style OR shape
            from matplotlib.path import Path
            verts = [
                (cx - w / 2, cy - h / 2),
                (cx - w / 4, cy),
                (cx - w / 2, cy + h / 2),
                (cx, cy + h / 2),
                (cx + w / 2, cy),
                (cx, cy - h / 2),
                (cx - w / 2, cy - h / 2),
            ]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3,
                     Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            pth = Path(verts, codes)
            ax.add_patch(mpatches_local.PathPatch(
                pth, facecolor="none", edgecolor=color, linewidth=2))
            if op_name == "XOR":
                # Add an extra curve on the left
                verts2 = [(cx - w / 2 - 0.08, cy - h / 2),
                          (cx - w / 4 - 0.08, cy),
                          (cx - w / 2 - 0.08, cy + h / 2)]
                codes2 = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                pth2 = Path(verts2, codes2)
                ax.add_patch(mpatches_local.PathPatch(
                    pth2, facecolor="none", edgecolor=color, linewidth=2))
        elif op_name == "IMPLIES":
            # Arrow symbol
            ax.annotate("", xy=(cx + w / 2 - 0.05, cy),
                        xytext=(cx - w / 2 + 0.05, cy),
                        arrowprops=dict(arrowstyle="->",
                                        color=color, lw=2.2))
        else:
            ax.text(cx, cy, "?", ha="center", va="center",
                    fontsize=fs + 4, fontweight="bold", color=color)
        # NAND / NOR add a small bubble on output
        if op_name in ("NAND", "NOR"):
            ax.add_patch(mpatches_local.Circle(
                (cx + w / 2 + 0.06, cy), 0.06,
                facecolor="white", edgecolor=color, linewidth=1.8))

    def _render(self, sub_rng, var_names, expr_label, rows, results,
                hidden_idx, gate_op=None, gate_header=False):
        style = self._random_style()
        n_rows = len(rows)
        n_cols = len(var_names) + 1

        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(4.5, n_cols * 1.6) * s,
                                         max(3.2, (n_rows + 2) * 0.65) * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        palette = style["palette"]
        fs = max(style["font_size_base"], 11)
        ff = style["font_family"]

        # Header (gate symbol + var headers)
        headers = var_names + [expr_label]
        header_color = sub_rng.choice([palette[0], palette[1], palette[5]])
        # For variable columns, render text as usual
        for j, h in enumerate(var_names):
            ax.text(j + 0.5, n_rows + 0.5, h, ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", fontfamily=ff,
                    color=header_color)
        # Last column: either text or a gate shape
        if gate_header:
            self._draw_gate_symbol(ax, len(var_names) + 0.5,
                                    n_rows + 0.55, gate_op,
                                    header_color, fs)
        else:
            ax.text(len(var_names) + 0.5, n_rows + 0.5, expr_label,
                    ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", fontfamily=ff,
                    color=header_color)

        # Data rows
        row_alt = sub_rng.choice([True, False])
        for i, row in enumerate(rows):
            y = n_rows - 1 - i
            if row_alt and i % 2 == 1:
                ax.add_patch(mpatches.Rectangle(
                    (0, y), n_cols, 1,
                    facecolor=palette[6], alpha=0.12, zorder=0))
            for j, val in enumerate(row):
                ax.text(j + 0.5, y + 0.5, str(val), ha="center", va="center",
                        fontsize=fs, fontfamily=ff)
            display = "?" if i == hidden_idx else str(results[i])
            color = palette[2] if display == "?" else "black"
            ax.text(len(var_names) + 0.5, y + 0.5, display,
                    ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", color=color,
                    fontfamily=ff)
            if i == hidden_idx:
                # highlight the unknown row
                ax.add_patch(mpatches.Rectangle(
                    (0, y), n_cols, 1,
                    fill=False, edgecolor=palette[2], linewidth=2.0,
                    zorder=5))

        # Grid lines
        for i in range(n_rows + 2):
            ax.axhline(y=i, color="#cccccc", linewidth=0.9)
        for j in range(n_cols + 1):
            ax.axvline(x=j, color="#cccccc", linewidth=0.9)
        ax.axhline(y=n_rows, color="#333333", linewidth=2)
        # Separator between var columns and result
        ax.axvline(x=len(var_names), color="#333333", linewidth=1.8)

        ax.set_xlim(0, n_cols)
        ax.set_ylim(-0.3, n_rows + 1.2)
        ax.set_title(sub_rng.choice(_TITLE_VARIANTS),
                     fontsize=fs + 3, fontweight="bold",
                     fontfamily=ff, pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
