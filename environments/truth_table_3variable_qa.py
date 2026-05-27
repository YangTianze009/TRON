"""
Truth Table 3-variable QA environment (v3 planning, env #59).

Goal: evaluate compound Boolean expressions on 3 variables (P, Q, R) where
the learner must reason about the value of specific rows marked '?'.
Targets reference deductive and X5 (deductive symbolic
reasoning) + X1 (multi-step rule execution).

Difficulty axes (per spec):
  A) `expression_depth = 2 + level // 2`      (2..6 operators)
  B) `operator_set`: L0={AND,OR}, L3 adds NOT, L6 adds XOR, L9 adds IMPLIES
     Also `n_missing_cells = 1 + level // 3`  (visual distractor count)

Format is constant 4-way MCQ (letters A-D; options are T/F variants or
row descriptors).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ---------------------------------------------------------------------- #
# Boolean operators
# ---------------------------------------------------------------------- #

def _op_and(a, b): return a and b
def _op_or(a, b): return a or b
def _op_xor(a, b): return a ^ b
def _op_impl(a, b): return (not a) or b

_OPS_BIN = {
    "AND":     (_op_and,  " AND ",  True),
    "OR":      (_op_or,   " OR ",   True),
    "XOR":     (_op_xor,  " XOR ",  True),
    "->":      (_op_impl, " -> ",   True),
}

_VAR_NAME_POOLS = [
    ["P", "Q", "R"],
    ["A", "B", "C"],
    ["X", "Y", "Z"],
    ["p", "q", "r"],
]

_TITLE_VARIANTS = [
    "Truth Table",
    "Boolean Truth Table",
    "Logic Table",
    "Propositional Truth Table",
    "Truth Assignment Table",
]

# ---------------------------------------------------------------------- #
# Expression tree: nodes are dicts
#   {"var": idx}   leaf
#   {"not": child}
#   {"op": name, "l": child, "r": child}
# ---------------------------------------------------------------------- #

def _evaluate(node, vals) -> bool:
    if "var" in node:
        return bool(vals[node["var"]])
    if "not" in node:
        return not _evaluate(node["not"], vals)
    op = node["op"]
    l = _evaluate(node["l"], vals)
    r = _evaluate(node["r"], vals)
    fn = _OPS_BIN[op][0]
    return fn(l, r)

def _render_expr(node, var_names) -> str:
    if "var" in node:
        return var_names[node["var"]]
    if "not" in node:
        sub = _render_expr(node["not"], var_names)
        if "var" in node["not"] or "not" in node["not"]:
            return f"NOT {sub}"
        return f"NOT ({sub})"
    l = _render_expr(node["l"], var_names)
    r = _render_expr(node["r"], var_names)
    glyph = _OPS_BIN[node["op"]][1]
    return f"({l}{glyph}{r})"

def _count_ops(node) -> int:
    if "var" in node:
        return 0
    if "not" in node:
        return 1 + _count_ops(node["not"])
    return 1 + _count_ops(node["l"]) + _count_ops(node["r"])

def _build_expr(rng: random.Random, depth: int,
                ops_binary: List[str], allow_not: bool,
                n_vars: int, r_usage: bool) -> Dict:
    """Build a random expression tree with `depth` operators.

    Keeps variable indices in [0, n_vars). If n_vars == 3 and `r_usage` is
    False (L0 spec), we only draw P or Q.
    """
    if depth <= 0:
        if n_vars == 3 and not r_usage:
            vi = rng.choice([0, 1])
        else:
            vi = rng.randrange(n_vars)
        return {"var": vi}

    # Occasionally wrap NOT when allowed
    if allow_not and rng.random() < 0.28 and depth >= 1:
        return {"not": _build_expr(rng, depth - 1, ops_binary, allow_not,
                                    n_vars, r_usage)}

    op = rng.choice(ops_binary)
    # Split depth between two children
    left_depth = rng.randint(0, depth - 1)
    right_depth = depth - 1 - left_depth
    l = _build_expr(rng, left_depth, ops_binary, allow_not, n_vars, r_usage)
    r = _build_expr(rng, right_depth, ops_binary, allow_not, n_vars, r_usage)
    return {"op": op, "l": l, "r": r}

class TruthTable3variableQA(StandaloneVisualEnv):
    ENV_NAME = "truth_table_3variable"

    # -------------------------------------------------- #
    # Level configuration (2 axes from spec)
    # -------------------------------------------------- #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Axis A: number of operators (boosted at high levels)
        if level <= 5:
            expression_depth = 2 + level // 2       # 2,2,3,3,4,4
        elif level <= 7:
            expression_depth = 5 + (level - 6)      # 5, 6
        else:
            expression_depth = 7 + (level - 8) * 2  # 7, 9 -- much deeper at L9
        # Axis B: operator set
        if level <= 2:
            ops = ["AND", "OR"]
            allow_not = False
        elif level <= 4:
            ops = ["AND", "OR"]
            allow_not = True
        elif level <= 6:
            ops = ["AND", "OR", "XOR"]
            allow_not = True
        else:
            ops = ["AND", "OR", "XOR", "->"]
            allow_not = True

        # Visual distractor: number of rows marked '?'
        n_missing_cells = 1 + level // 3            # 1,1,1,2,2,2,3,3,3,4

        # L0 spec: R is unused (distractor column) for the expression
        r_in_expr = level >= 1

        # L6-L9: hide the expression from the QUESTION TEXT so the model
        # must read it from the image's banner, and apply visual noise
        # (rotation, partial occlusion) to the banner text.
        hide_expr_in_question = level >= 6
        obscure_banner = level >= 6
        return {
            "expression_depth": expression_depth,
            "ops": ops,
            "allow_not": allow_not,
            "n_missing_cells": n_missing_cells,
            "r_in_expr": r_in_expr,
            "hide_expr_in_question": hide_expr_in_question,
            "obscure_banner": obscure_banner,
        }

    # -------------------------------------------------- #
    # Problem generation
    # -------------------------------------------------- #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        # Unique prime 1381 for this env
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1381)
        self._primary_complexity_feature = cfg["expression_depth"]

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        var_names = rng.choice(_VAR_NAME_POOLS)

        # Build expression tree with exact (or close) op count
        expr = None
        for _ in range(40):
            expr = _build_expr(rng, cfg["expression_depth"],
                               cfg["ops"], cfg["allow_not"],
                               n_vars=3, r_usage=cfg["r_in_expr"])
            n_ops = _count_ops(expr)
            if n_ops >= cfg["expression_depth"]:
                break
        if expr is None:
            return None

        expr_text = _render_expr(expr, var_names)

        # Compute all 8 rows (3 variables)
        rows = [(p, q, r) for p in [0, 1] for q in [0, 1] for r in [0, 1]]
        results = []
        for row in rows:
            results.append(int(_evaluate(expr, row)))

        # Pick n_missing_cells rows to mark '?' (avoid degenerate: at
        # least one '?' must be askable)
        n_missing = min(cfg["n_missing_cells"], 4)
        missing_idx = rng.sample(range(8), n_missing)
        # Pick the asked cell — first of missing
        asked_idx = missing_idx[0]

        # Avoid constant-result expressions (all 0 or all 1): the '?' is
        # trivially guessable from looking at other rows.
        if all(r == results[asked_idx] for r in results):
            return None
        # Also ensure results vary across the 8 rows (otherwise any "?" is
        # trivially guessed from any other row).
        if len(set(results)) < 2:
            return None

        answer_letter, options = self._build_mcq_options(
            rng, results[asked_idx], level=level)

        # Phrasing pool
        asked_row = rows[asked_idx]
        row_desc = ", ".join(
            f"{var_names[i]}={asked_row[i]}" for i in range(3))

        if cfg.get("hide_expr_in_question"):
            # Refer to expression only as "the expression shown in the image".
            q_stems = [
                (f"The truth table in the image evaluates an expression "
                 f"over 3 Boolean variables (the expression itself is "
                 f"written in the banner ABOVE the table — you must read "
                 f"it from the image). One cell is marked '?'. What is "
                 f"the result when {row_desc}?"),
                (f"Read the Boolean expression from the image's banner. "
                 f"The truth table has '?' cells. What value belongs in "
                 f"the '?' cell for the row with {row_desc}?"),
                (f"The image shows a truth table for a Boolean expression "
                 f"(the expression is ONLY visible in the image). "
                 f"Determine the output for the row where {row_desc}."),
                (f"The expression is shown at the top of the image (not "
                 f"given in this question). What does the '?' cell equal "
                 f"for the row with {row_desc}?"),
            ]
        else:
            q_stems = [
                (f"The truth table in the image evaluates the expression "
                 f"{expr_text} for 3 Boolean variables. One cell is marked "
                 f"'?'. Using the row values shown, what is the result when "
                 f"{row_desc}?"),
                (f"Examine the partially filled truth table for {expr_text}. "
                 f"What value belongs in the '?' cell of the row with "
                 f"{row_desc}?"),
                (f"For the expression {expr_text}, compute the result when "
                 f"{row_desc}. The truth table in the image shows the other "
                 f"rows for reference."),
                (f"The image shows a truth table with {n_missing} '?' cells. "
                 f"Determine the output of {expr_text} for the row where "
                 f"{row_desc}."),
            ]
        question = (
            f"{rng.choice(q_stems)} Answer with a single letter."
        )
        # 2026-05-04: simplified L0 (was 10% too-hard) — concise + step hint.
        if level <= 1:
            question += (
                " Be concise. Substitute the given variable values into the "
                "expression, evaluate AND/OR per Boolean rules (1=T, 0=F), "
                "and pick T or F. Output only the letter."
            )

        title = rng.choice(_TITLE_VARIANTS)
        image = self._render(var_names, expr_text, rows, results,
                             missing_idx, asked_idx, options, title, cfg)
        return question, answer_letter, image

    def _build_mcq_options(self, rng: random.Random,
                           correct_val: int,
                           level: int = 0) -> Tuple[str, List[str]]:
        # For truth-table values, options T/F + distractors.
        # At low levels we keep two "obvious" non-Boolean distractors.
        # At L6+, use only real T/F options so every option is plausible
        # (model must actually evaluate the expression).
        correct_str = "T" if correct_val == 1 else "F"
        wrong_str = "F" if correct_val == 1 else "T"
        if level >= 6:
            # Use legitimate logic terms instead of obvious fake answers.
            # "Tautology" and "Contradiction" are real concepts but wrong
            # for a single-cell evaluation (the question asks for a value,
            # not a property of the expression).
            distractors = [
                wrong_str,
                "Tautology (T for all rows)",
                "Contradiction (F for all rows)",
            ]
        else:
            distractors = [
                wrong_str,
                "Both T and F are valid",
                "Neither T nor F (undefined)",
            ]
        rng.shuffle(distractors)
        insert_idx = rng.randint(0, 3)
        options = distractors[:insert_idx] + [correct_str] + distractors[insert_idx:]
        options = options[:4]
        if correct_str not in options:
            options[0] = correct_str
        if options.count(correct_str) > 1:
            # Fix duplicates
            for i, o in enumerate(options):
                if o == correct_str and i != insert_idx:
                    options[i] = "Undetermined"
                    break
        answer_letter = chr(ord("A") + options.index(correct_str))
        return answer_letter, options

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _render(self, var_names, expr_text, rows, results,
                missing_idx, asked_idx, options, title, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans", "monospace"])
        base_fs = self._rng.choice([11, 12, 13])

        n_rows = len(rows)
        n_cols = len(var_names) + 1

        fig_w = 10.5 * sc
        fig_h = 8.0 * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.25)
        ax_table = fig.add_subplot(gs[0])
        ax_opts = fig.add_subplot(gs[1])

        ax_table.set_facecolor(style["bg_color"])
        ax_table.axis("off")
        ax_table.set_title(title, fontsize=base_fs + 3, fontweight="bold",
                            fontfamily=ff, pad=14)

        palette = style["palette"]

        # --- Expression banner above the table ---
        # Always render the expression in a wrapped banner ABOVE the table
        # rather than as the result-column header: even short expressions like
        # "(Q OR (Q AND P))" (16 chars) are wider than the column (1.2 units)
        # and spill into neighboring columns (e.g. the R column), making the
        # header row look like "R (Q OR (Q AND P))" merged. Short "Result"
        # header keeps the column clean; banner is legible and sits outside
        # the grid cells. (L6+ also applies obscure_banner on top of this.)
        use_short_header = True
        result_header = "Result"

        # --- Table layout: draw as matplotlib cells manually ---
        headers = var_names + [result_header]
        col_w = 1.2
        row_h = 0.8
        x_off = 0.5
        y_off = 0.5

        # Header row
        for j, h in enumerate(headers):
            ax_table.text(x_off + (j + 0.5) * col_w,
                          y_off + (n_rows + 0.5) * row_h,
                          h, ha="center", va="center",
                          fontsize=base_fs + 1, fontweight="bold",
                          fontfamily=ff, color=palette[0])

        if use_short_header:
            # Draw wrapped formula banner ABOVE the table
            wrapped = self._wrap_expr(expr_text, width=34)
            n_lines = len(wrapped)
            banner_y = y_off + (n_rows + 1 + n_lines * 0.6) * row_h
            obscure = cfg.get("obscure_banner", False)
            banner_rot = self._rng.uniform(-8, 8) if obscure else 0
            banner_color = "#3a3a3a" if obscure else "#1a1a1a"
            ax_table.text(x_off + (n_cols / 2) * col_w, banner_y,
                          "Expression:",
                          ha="center", va="top",
                          fontsize=base_fs, fontweight="bold",
                          fontfamily=ff, color=palette[0])
            for li, ln in enumerate(wrapped):
                ax_table.text(x_off + (n_cols / 2) * col_w,
                              banner_y - (li + 1) * 0.45 * row_h,
                              ln, ha="center", va="top",
                              fontsize=max(base_fs - 1, 10),
                              fontfamily="monospace", color=banner_color,
                              rotation=banner_rot)
            # Add partial occlusion smudges on banner at high levels.
            if obscure:
                import matplotlib.patches as mpatches_local
                for _ in range(2):
                    ox = x_off + self._rng.uniform(0.5, n_cols * col_w - 0.5)
                    oy = banner_y - self._rng.uniform(0, n_lines * 0.45 * row_h)
                    ax_table.add_patch(mpatches_local.Ellipse(
                        (ox, oy),
                        width=self._rng.uniform(0.6, 1.2),
                        height=self._rng.uniform(0.15, 0.35),
                        angle=self._rng.uniform(-30, 30),
                        facecolor="#c4b99a",
                        alpha=self._rng.uniform(0.25, 0.40),
                        zorder=6))
        # Data rows
        for i, row in enumerate(rows):
            y_cell = y_off + (n_rows - 1 - i + 0.5) * row_h
            for j, val in enumerate(row):
                ax_table.text(x_off + (j + 0.5) * col_w, y_cell,
                              str(val), ha="center", va="center",
                              fontsize=base_fs, fontfamily=ff)
            # Result column
            if i in missing_idx:
                display = "?"
                color = "#e74c3c"
            else:
                display = "T" if results[i] == 1 else "F"
                color = "#1a1a1a"
            ax_table.text(x_off + (n_cols - 0.5) * col_w, y_cell,
                          display, ha="center", va="center",
                          fontsize=base_fs + 1, fontweight="bold",
                          color=color, fontfamily=ff)

            # Highlight the asked row
            if i == asked_idx:
                from matplotlib.patches import Rectangle
                ax_table.add_patch(Rectangle(
                    (x_off, y_off + (n_rows - 1 - i) * row_h),
                    n_cols * col_w, row_h,
                    fill=False, edgecolor="#e74c3c",
                    linewidth=2.0, zorder=5))

        # Grid lines
        total_w = n_cols * col_w
        total_h = (n_rows + 1) * row_h
        for i in range(n_rows + 2):
            ax_table.plot([x_off, x_off + total_w],
                          [y_off + i * row_h, y_off + i * row_h],
                          color="#bbbbbb", linewidth=0.8, zorder=1)
        for j in range(n_cols + 1):
            ax_table.plot([x_off + j * col_w, x_off + j * col_w],
                          [y_off, y_off + total_h],
                          color="#bbbbbb", linewidth=0.8, zorder=1)
        # Header separator
        ax_table.plot([x_off, x_off + total_w],
                      [y_off + n_rows * row_h, y_off + n_rows * row_h],
                      color="#333333", linewidth=1.8, zorder=2)

        # Adjust ylim to account for banner height (if any)
        ax_table.set_xlim(0, total_w + 1.0)
        if use_short_header:
            # Estimate banner lines
            wrapped = self._wrap_expr(expr_text, width=34)
            banner_h = (len(wrapped) + 1) * 0.6 * row_h
            ax_table.set_ylim(0, total_h + 1.2 + banner_h)
        else:
            ax_table.set_ylim(0, total_h + 1.2)

        # --- Options panel ---
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, 10)
        ax_opts.set_ylim(0, 10)
        ax_opts.axis("off")
        ax_opts.set_title(
            f"What value belongs in the '?' cell?",
            fontsize=base_fs + 2, fontweight="bold",
            fontfamily=ff, pad=10)
        y = 8.0
        dy = 1.1
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            ax_opts.text(0.4, y, f"({letter}) {opt}",
                          fontsize=base_fs + 1, ha="left", va="top",
                          fontfamily=ff, color="#1a1a1a")
            y -= dy

        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05,
                             wspace=0.25)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # -------------------------------------------------- #
    # Helpers
    # -------------------------------------------------- #
    @staticmethod
    def _wrap_expr(expr: str, width: int = 34) -> List[str]:
        """Wrap a long Boolean expression into lines of <=width chars.
        Breaks at operator boundaries (AND / OR / XOR / -> / NOT / parens).
        """
        tokens = []
        cur = ""
        i = 0
        while i < len(expr):
            ch = expr[i]
            if ch == ' ':
                if cur:
                    tokens.append(cur)
                    cur = ""
                i += 1
                continue
            cur += ch
            i += 1
        if cur:
            tokens.append(cur)
        lines = []
        cur = ""
        for tok in tokens:
            if len(cur) + len(tok) + 1 > width and cur:
                lines.append(cur)
                cur = tok
            else:
                cur = (cur + " " + tok).strip()
        if cur:
            lines.append(cur)
        return lines

# ---------------------------------------------------------------------- #
# Local smoke test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import collections
    env = TruthTable3variableQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED")
                continue
            print(f"[seed={seed} L{level}] A={env._answer}  "
                  f"Q={env.get_instruction()[:80]}")
    for level in (0, 3, 6, 9):
        letters = collections.Counter()
        for s in range(20):
            e = TruthTable3variableQA()
            if e.generate(seed=s * 1000 + level * 37 + 17,
                          parameter={"level": level}):
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
