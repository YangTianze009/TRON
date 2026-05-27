"""
Text Render Math QA (redesigned 2026-04-16).

Target: ocr-bench text reading + targeted-geometry Vision-Only (P2 OCR capability).

Redesign goal: the v2 was just text on a white canvas — trivially text-only.
The new version renders the arithmetic expression as a TRULY VISUAL artifact:
  * Operands are drawn inside boxes / circles / clouds.
  * Operators are drawn as visual symbols (e.g. coloured arrows, labelled
    boxes, handwritten glyphs).
  * Answer slots are visually emphasised (double-box, dashed border).
  * Various visual layouts:
      - Equation line (like original but heavily embellished)
      - Boxed fraction stack (numerator / denominator)
      - Column addition / subtraction layout
      - Stacked multiplication table
      - Flow-chart where each operand is a node connected by arrows
      - Visual counting: operand N represented by N drawn dots / tally marks
        (L0 only).
  * Visual markers: arrows pointing to where the answer goes, boxes around
    sub-expressions, highlighted operator, scribble background.
  * Handwriting style (italic + rotation jitter + size jitter).
  * L0 vs L9 structural shift:
      - L0: 2 small integers 1-9 shown as dots + a + operator; direct
        counting possible.
      - L9: 5-6 terms, mixed ops, parentheses, rotated text, noisy
        background — requires visual parsing.
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

_FONTS = ["serif", "sans-serif", "DejaVu Sans", "monospace"]

_BOX_FACES = [
    "#ffffff", "#fef9e7", "#e8f8f5", "#fef5e7", "#eaeded",
    "#fdebd0", "#ebf5fb", "#f4ecf7", "#fff3cd",
]

_BOX_EDGES = [
    "#2c3e50", "#34495e", "#7f8c8d", "#1a5276", "#117864",
    "#7d6608", "#4d5656", "#283747",
]

_OPERATOR_COLORS = [
    "#c0392b", "#d35400", "#b03a2e", "#7d3c98", "#8e44ad",
    "#922b21", "#196f3d", "#1b4f72",
]

def _rotate_pt(cx, cy, x, y, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c

class TextRenderMathQA(StandaloneVisualEnv):
    ENV_NAME = "text_render_math"

    _QUESTION_TEMPLATES = [
        ("The image shows an arithmetic expression laid out visually. "
         "Read the expression and compute the result. Answer with an "
         "integer."),
        ("Parse the rendered arithmetic problem in the image. Evaluate "
         "the expression and give the final value. Answer with an "
         "integer."),
        ("Read the numbers and operators drawn in the figure. Compute "
         "the final value of the expression. Answer with an integer."),
        ("The diagram presents a math expression as a visual layout. "
         "Compute the result. Answer with an integer."),
        ("Carefully read the arithmetic expression shown in the image "
         "and evaluate it. Answer with an integer."),
        ("Evaluate the arithmetic expression drawn in the image and "
         "report the numeric result. Answer with an integer."),
    ]

    _TITLE_VARIANTS = [
        "Evaluate the expression", "Compute the value", "Arithmetic puzzle",
        "Read and compute", "Math expression", "Find the result",
        "Expression evaluation", "Solve the equation",
    ]

    _LAYOUT_POOL_BY_LEVEL = {
        "L0": ["counting_dots", "boxed_line", "boxed_line"],
        "L1": ["counting_dots", "boxed_line", "column_stack",
               "boxed_line"],
        "L3": ["boxed_line", "column_stack", "boxed_line",
               "flow_chart"],
        "L5": ["boxed_line", "column_stack", "flow_chart",
               "fraction_stack"],
        "L7": ["boxed_line", "flow_chart", "fraction_stack",
               "column_stack"],
        "L9": ["boxed_line", "flow_chart", "fraction_stack",
               "column_stack"],
    }

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 0:
            return {
                "n_ops": 1,
                "num_min": 1, "num_max": 9,
                "allow_parens": False, "allow_mul": False,
                "allow_div": False,
                "allow_negative_intermediate": False,
                "bg_noise": 0.0, "rotation_max": 1.0,
                "layouts": ["counting_dots", "boxed_line"],
                "font_jitter": 0.0,
                "handwriting": False,
            }
        if level <= 1:
            return {
                "n_ops": 2, "num_min": 1, "num_max": 12,
                "allow_parens": False, "allow_mul": False,
                "allow_div": False,
                "allow_negative_intermediate": False,
                "bg_noise": 0.02, "rotation_max": 2.0,
                "layouts": ["boxed_line", "counting_dots", "column_stack"],
                "font_jitter": 0.1, "handwriting": False,
            }
        if level <= 3:
            return {
                "n_ops": 3, "num_min": 1, "num_max": 25,
                "allow_parens": False, "allow_mul": True,
                "allow_div": False,
                "allow_negative_intermediate": False,
                "bg_noise": 0.05, "rotation_max": 3.0,
                "layouts": ["boxed_line", "column_stack", "flow_chart"],
                "font_jitter": 0.15, "handwriting": False,
            }
        if level <= 5:
            return {
                "n_ops": 4, "num_min": 1, "num_max": 40,
                "allow_parens": True, "allow_mul": True,
                "allow_div": False,
                "allow_negative_intermediate": True,
                "bg_noise": 0.08, "rotation_max": 4.0,
                "layouts": ["boxed_line", "column_stack",
                            "flow_chart", "fraction_stack"],
                "font_jitter": 0.2, "handwriting": True,
            }
        if level <= 7:
            return {
                "n_ops": 5, "num_min": 2, "num_max": 60,
                "allow_parens": True, "allow_mul": True,
                "allow_div": False,
                "allow_negative_intermediate": True,
                "bg_noise": 0.12, "rotation_max": 6.0,
                "layouts": ["boxed_line", "flow_chart",
                            "fraction_stack", "column_stack"],
                "font_jitter": 0.25, "handwriting": True,
            }
        # L8-9
        return {
            "n_ops": 5 + level // 5,
            "num_min": 2, "num_max": 80,
            "allow_parens": True, "allow_mul": True,
            "allow_div": False,
            "allow_negative_intermediate": True,
            "bg_noise": 0.15, "rotation_max": 8.0,
            "layouts": ["boxed_line", "flow_chart",
                        "fraction_stack", "column_stack"],
            "font_jitter": 0.3, "handwriting": True,
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_ops"] * 20 + cfg["num_max"]

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n_terms = cfg["n_ops"] + 1
        ops_allowed = ["+", "-"]
        if cfg["allow_mul"]:
            ops_allowed.append("*")

        terms = [rng.randint(cfg["num_min"], cfg["num_max"])
                 for _ in range(n_terms)]
        ops = [rng.choice(ops_allowed) for _ in range(n_terms - 1)]

        expr_tokens = []
        for i, t in enumerate(terms):
            expr_tokens.append(str(t))
            if i < len(ops):
                expr_tokens.append(ops[i])
        expr = " ".join(expr_tokens)

        # Insert parens
        if cfg["allow_parens"] and n_terms >= 3:
            lo = rng.randint(0, n_terms - 2)
            hi = lo + 1
            expr_tokens2 = []
            for i, t in enumerate(terms):
                if i == lo:
                    expr_tokens2.append("(")
                expr_tokens2.append(str(t))
                if i == hi:
                    expr_tokens2.append(")")
                if i < len(ops):
                    expr_tokens2.append(ops[i])
            expr = " ".join(expr_tokens2)

        try:
            result = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None
        if not isinstance(result, (int, float)):
            return None
        if abs(result) > 10000:
            return None
        if isinstance(result, float) and not result.is_integer():
            return None
        answer = str(int(result))

        # Choose layout (but consider that some layouts require specific
        # structure — e.g. column_stack wants only + or -, fraction_stack
        # wants a / or just 2 terms).
        layout = self._choose_layout(rng, cfg, terms, ops)
        image = self._render(terms, ops, expr, layout, cfg, rng)
        q = rng.choice(self._QUESTION_TEMPLATES)
        return q, answer, image

    def _choose_layout(self, rng, cfg, terms, ops):
        pool = list(cfg["layouts"])
        # Counting dots only works for small operands + single op
        if "counting_dots" in pool:
            if len(ops) > 1 or any(o == "*" for o in ops) or \
                    any(abs(t) > 10 for t in terms):
                pool = [p for p in pool if p != "counting_dots"]
        # Column stack only works for + / - chains
        if "column_stack" in pool:
            if any(o == "*" for o in ops):
                pool = [p for p in pool if p != "column_stack"]
            if len(terms) > 5:
                pool = [p for p in pool if p != "column_stack"]
        # Fraction stack arbitrarily splits tokens at the midpoint; disable it
        # when the split would leave unbalanced parentheses or land in the
        # middle of a sub-expression, because the displayed layout would no
        # longer represent the evaluated expression.
        if "fraction_stack" in pool:
            pool = [p for p in pool if p != "fraction_stack"]
        return rng.choice(pool) if pool else "boxed_line"

    # ------------------------------------------------ #
    # Rendering dispatcher
    # ------------------------------------------------ #

    def _render(self, terms, ops, expr, layout, cfg, rng) -> Image.Image:
        style = self._random_style()
        if layout == "boxed_line":
            return self._render_boxed_line(terms, ops, expr, cfg, rng,
                                           style)
        if layout == "column_stack":
            return self._render_column_stack(terms, ops, expr, cfg, rng,
                                             style)
        if layout == "counting_dots":
            return self._render_counting_dots(terms, ops, expr, cfg, rng,
                                              style)
        if layout == "flow_chart":
            return self._render_flow_chart(terms, ops, expr, cfg, rng,
                                           style)
        if layout == "fraction_stack":
            return self._render_fraction_stack(terms, ops, expr, cfg, rng,
                                               style)
        return self._render_boxed_line(terms, ops, expr, cfg, rng, style)

    # ------------------------------------------------ #
    # Layout implementations
    # ------------------------------------------------ #

    def _add_bg_noise(self, ax, cfg, rng):
        if cfg["bg_noise"] > 0:
            n_dots = int(cfg["bg_noise"] * 200)
            xs = [rng.random() for _ in range(n_dots)]
            ys = [rng.random() for _ in range(n_dots)]
            ax.scatter(xs, ys, s=4, c="#cccccc", alpha=0.45, zorder=1,
                       transform=ax.transAxes)

    def _add_visual_markers(self, ax, fig, cfg, rng):
        # Arrow pointing to "=" symbol
        # etc.
        pass

    def _render_boxed_line(self, terms, ops, expr, cfg, rng, style):
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(9 * sc, 3.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)

        self._add_bg_noise(ax, cfg, rng)

        # Draw each token in its own box
        # Need to handle parens too.
        tokens = expr.split()
        n = len(tokens)
        gap = 0.4  # extra spacing so operator glyphs don't clip into number boxes
        # Box widths estimated
        box_w_map = []
        for tok in tokens:
            if tok in "+-*":
                # operator: circle/diamond has radius ~0.32, needs width >= 0.75
                box_w_map.append(0.85)
            elif tok in "()":
                box_w_map.append(0.45)
            else:
                box_w_map.append(0.32 * len(tok) + 0.55)
        total_w = sum(box_w_map) + gap * (n - 1)
        # Scale to fit. Leave ~2.0 units for the "= ?" on the right.
        avail_w = 7.6
        scale = min(1.0, avail_w / max(total_w, 0.1))
        box_w_map = [w * scale for w in box_w_map]
        total_w = sum(box_w_map) + gap * (n - 1)
        x_cursor = max(0.3, (10 - total_w - 1.8) / 2 + 0.3)
        cy = 1.5

        eq_edge = rng.choice(_BOX_EDGES)
        num_face = rng.choice(_BOX_FACES)
        op_color = rng.choice(_OPERATOR_COLORS)
        for i, tok in enumerate(tokens):
            w = box_w_map[i]
            font = rng.choice(_FONTS)
            rot = rng.uniform(-cfg["rotation_max"], cfg["rotation_max"])
            base_fs = 24 + int(10 * cfg["font_jitter"] * rng.uniform(-1, 1))
            if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
                box_style = rng.choice([
                    "round,pad=0.05", "round4,pad=0.08",
                    "round,pad=0.1", "square,pad=0.08"])
                rect = mpatches.FancyBboxPatch(
                    (x_cursor, cy - 0.55), w, 1.1,
                    boxstyle=box_style, facecolor=num_face,
                    edgecolor=eq_edge, linewidth=1.5, zorder=3)
                ax.add_patch(rect)
                ax.text(x_cursor + w / 2, cy, tok,
                        fontsize=max(14, base_fs),
                        fontweight="bold", ha="center", va="center",
                        family=font, rotation=rot,
                        color="#1a1a1a", zorder=10)
            elif tok in "+-*":
                # Draw operator as a coloured diamond / circle
                shape = rng.choice(["circle", "diamond", "plain"])
                if shape == "circle":
                    circ = mpatches.Circle(
                        (x_cursor + w / 2, cy), 0.32,
                        facecolor="#ffffff", edgecolor=op_color,
                        linewidth=2, zorder=3)
                    ax.add_patch(circ)
                elif shape == "diamond":
                    dia = mpatches.Polygon(
                        [(x_cursor + w / 2, cy + 0.35),
                         (x_cursor + w / 2 + 0.35, cy),
                         (x_cursor + w / 2, cy - 0.35),
                         (x_cursor + w / 2 - 0.35, cy)],
                        facecolor="#ffffff", edgecolor=op_color,
                        linewidth=2, zorder=3)
                    ax.add_patch(dia)
                display_op = "\u00d7" if tok == "*" else tok
                ax.text(x_cursor + w / 2, cy, display_op,
                        fontsize=max(22, base_fs + 4),
                        fontweight="bold", ha="center", va="center",
                        family=font, color=op_color, zorder=10)
            else:  # paren
                ax.text(x_cursor + w / 2, cy, tok,
                        fontsize=max(20, base_fs),
                        fontweight="bold", ha="center", va="center",
                        family=font, color="#2c3e50", zorder=10)
            x_cursor += w + gap

        # "=" and "?" box
        ax.text(x_cursor + 0.2, cy, "=",
                fontsize=28, fontweight="bold",
                ha="center", va="center", color="#1a1a1a", zorder=10)
        qx = x_cursor + 0.6
        rect = mpatches.FancyBboxPatch(
            (qx, cy - 0.55), 0.9, 1.1,
            boxstyle="round,pad=0.08",
            facecolor="#fff3cd", edgecolor="#c0392b",
            linewidth=2.5, linestyle="--", zorder=3)
        ax.add_patch(rect)
        ax.text(qx + 0.45, cy, "?",
                fontsize=30, fontweight="bold",
                ha="center", va="center",
                color="#c0392b", zorder=10)

        title = rng.choice(self._TITLE_VARIANTS)
        # Note: data coords (xlim 0..10, ylim 0..3); do NOT use transAxes here.
        ax.text(5.0, 2.7, title, fontsize=13, fontweight="bold",
                ha="center", color="#2c3e50", zorder=10)

        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_column_stack(self, terms, ops, expr, cfg, rng, style):
        # Draw addition/subtraction in vertical column
        sc = style["figsize_scale"]
        n = len(terms)
        fig_h = 1.5 + 0.7 * n
        fig, ax = plt.subplots(figsize=(6 * sc, fig_h * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, fig_h)
        self._add_bg_noise(ax, cfg, rng)

        # Right-align digits of each term
        max_digits = max(len(str(t)) for t in terms)
        right_x = 6.5
        y_top = fig_h - 0.6
        font = rng.choice(_FONTS)
        line_color = rng.choice(_BOX_EDGES)
        for i, t in enumerate(terms):
            y = y_top - (i + 1) * 0.7
            text = str(t)
            rot = rng.uniform(-cfg["rotation_max"], cfg["rotation_max"])
            if i > 0:
                op_display = {"+": "+", "-": "\u2212", "*": "\u00d7"}.get(
                    ops[i - 1], ops[i - 1])
                ax.text(right_x - 0.5 * max_digits - 1.2, y,
                        op_display, fontsize=28,
                        fontweight="bold", ha="left", va="center",
                        family=font, color=rng.choice(_OPERATOR_COLORS),
                        rotation=rot, zorder=10)
            for j, ch in enumerate(text[::-1]):
                ax.text(right_x - 0.5 * j, y, ch, fontsize=26,
                        fontweight="bold", ha="center", va="center",
                        family=font, color="#1a1a1a", rotation=rot,
                        zorder=10)
        # Divider line and = ?
        div_y = y_top - (n + 0.3) * 0.7
        left_x = right_x - 0.5 * (max_digits + 1) - 0.7
        ax.plot([left_x, right_x + 0.4], [div_y, div_y],
                color=line_color, linewidth=2.5, zorder=5)
        ax.text(right_x + 1.3, div_y - 0.4, "= ?", fontsize=32,
                fontweight="bold", ha="center", va="center",
                family=font, color="#c0392b", zorder=10)

        title = rng.choice(self._TITLE_VARIANTS)
        ax.text(5, fig_h - 0.25, title,
                fontsize=13, fontweight="bold", ha="center",
                color="#2c3e50", zorder=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_counting_dots(self, terms, ops, expr, cfg, rng, style):
        # Each small operand rendered as N dots; op is + / -
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(9 * sc, 3.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 3)
        self._add_bg_noise(ax, cfg, rng)

        def draw_dots(cx, cy, n, color):
            # Arrange in row, then wrap
            per_row = 5
            rows = (n + per_row - 1) // per_row
            total_w = per_row * 0.22
            for i in range(n):
                r = i // per_row
                c = i % per_row
                actual_col = c if rows == 1 and n < per_row else c
                ox = cx - total_w / 2 + actual_col * 0.22 + 0.11
                oy = cy - (rows - 1) * 0.22 / 2 + (rows - 1 - r) * 0.22
                dot = mpatches.Circle((ox, oy), 0.08,
                                      facecolor=color,
                                      edgecolor="#2c3e50",
                                      linewidth=0.8, zorder=5)
                ax.add_patch(dot)

        # For each operand + op
        n_tokens = len(terms) * 2 - 1
        cx = 1.0
        gap = 1.8
        op_color = rng.choice(_OPERATOR_COLORS)
        dot_color = rng.choice(["#3498db", "#27ae60", "#9b59b6",
                                "#e67e22", "#16a085"])
        for i, t in enumerate(terms):
            # Draw dots
            n_val = abs(t)
            if n_val > 12:
                n_val = 12  # clamp for display
            box = mpatches.FancyBboxPatch(
                (cx - 0.7, 1.0), 1.4, 1.0,
                boxstyle="round,pad=0.04",
                facecolor="#ffffff", edgecolor="#2c3e50",
                linewidth=1.5, zorder=3)
            ax.add_patch(box)
            draw_dots(cx, 1.5, n_val, dot_color)
            # Also label with number below
            ax.text(cx, 0.7, f"({t})", fontsize=14, ha="center",
                    va="center", color="#7f8c8d", zorder=10)
            if i < len(terms) - 1:
                opdisplay = {"+": "+", "-": "\u2212",
                              "*": "\u00d7"}.get(ops[i], ops[i])
                ax.text(cx + 0.9, 1.5, opdisplay, fontsize=32,
                        fontweight="bold", ha="center", va="center",
                        color=op_color, zorder=10)
            cx += gap
        # = ?
        ax.text(cx - 0.2, 1.5, "= ?", fontsize=28, fontweight="bold",
                ha="center", va="center", color="#c0392b", zorder=10)
        title = rng.choice(self._TITLE_VARIANTS)
        # Use data coords (matches xlim 0..10, ylim 0..3) — transAxes bug fix.
        ax.text(5, 2.7, title, fontsize=13, fontweight="bold",
                ha="center", color="#2c3e50", zorder=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_flow_chart(self, terms, ops, expr, cfg, rng, style):
        sc = style["figsize_scale"]
        # Arrange operands as boxes connected by arrows with op labels
        fig, ax = plt.subplots(figsize=(10 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5)
        self._add_bg_noise(ax, cfg, rng)

        # Choose a polyline path for the flow: straight, zigzag, or step.
        path_shape = rng.choice(["straight", "zigzag"])
        n = len(terms)
        xs = []
        ys = []
        base_y = 2.5
        if path_shape == "straight":
            for i in range(n):
                xs.append(1 + i * (10 / max(1, n - 1 if n > 1 else 1)))
                ys.append(base_y)
        else:
            for i in range(n):
                xs.append(1 + i * (10 / max(1, n - 1 if n > 1 else 1)))
                ys.append(base_y + (1 if i % 2 == 0 else -0.5))

        edge = rng.choice(_BOX_EDGES)
        face = rng.choice(_BOX_FACES)
        op_color = rng.choice(_OPERATOR_COLORS)
        font = rng.choice(_FONTS)
        for i, (xx, yy) in enumerate(zip(xs, ys)):
            rect = mpatches.FancyBboxPatch(
                (xx - 0.55, yy - 0.4), 1.1, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=face, edgecolor=edge, linewidth=1.8,
                zorder=5)
            ax.add_patch(rect)
            rot = rng.uniform(-cfg["rotation_max"], cfg["rotation_max"])
            ax.text(xx, yy, str(terms[i]), fontsize=22, fontweight="bold",
                    ha="center", va="center", family=font,
                    color="#1a1a1a", rotation=rot, zorder=10)
            if i < n - 1:
                x2 = xs[i + 1]
                y2 = ys[i + 1]
                ax.annotate(
                    "", xy=(x2 - 0.55, y2), xytext=(xx + 0.55, yy),
                    arrowprops=dict(arrowstyle="->", lw=2.0,
                                    color=op_color),
                    zorder=4)
                mx = (xx + x2) / 2
                my = (yy + y2) / 2 + 0.3
                op_text = {"+": "+", "-": "\u2212", "*": "\u00d7"}.get(
                    ops[i], ops[i])
                ax.text(mx, my, op_text, fontsize=22,
                        fontweight="bold", ha="center", va="center",
                        color=op_color, family=font, zorder=11)
        # Final = ?
        last_x = xs[-1] + 1.0
        last_y = ys[-1]
        rect_q = mpatches.FancyBboxPatch(
            (last_x - 0.55, last_y - 0.45), 1.1, 0.9,
            boxstyle="round,pad=0.05",
            facecolor="#fff3cd", edgecolor="#c0392b",
            linewidth=2.5, linestyle="--", zorder=5)
        ax.add_patch(rect_q)
        ax.text(last_x, last_y, "?", fontsize=30, fontweight="bold",
                ha="center", va="center", color="#c0392b", zorder=10)
        ax.annotate("", xy=(last_x - 0.55, last_y),
                    xytext=(xs[-1] + 0.55, ys[-1]),
                    arrowprops=dict(arrowstyle="->", lw=2.0,
                                    color="#c0392b"),
                    zorder=4)
        ax.text((xs[-1] + last_x) / 2, last_y + 0.3, "=",
                fontsize=22, fontweight="bold", ha="center",
                va="center", color="#c0392b", zorder=11)

        title = rng.choice(self._TITLE_VARIANTS)
        ax.text(6, 4.6, title, fontsize=13, fontweight="bold",
                ha="center", color="#2c3e50", zorder=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_fraction_stack(self, terms, ops, expr, cfg, rng, style):
        # Split expression into two sub-expressions, render as a stacked
        # fraction with an = ?  Even when there's no real division, this
        # layout visually suggests stacking for variety.
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(9 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        self._add_bg_noise(ax, cfg, rng)

        # Split tokens into two halves
        tokens = expr.split()
        half = len(tokens) // 2
        top_tokens = tokens[:half]
        bottom_tokens = tokens[half:]
        # Pick display op between halves
        top_str = self._tok_display(top_tokens)
        bot_str = self._tok_display(bottom_tokens)
        font = rng.choice(_FONTS)
        top_y = 3.5
        bot_y = 1.5
        # Draw top
        rot = rng.uniform(-cfg["rotation_max"], cfg["rotation_max"])
        ax.text(3.5, top_y, top_str, fontsize=28, fontweight="bold",
                ha="center", va="center", family=font,
                color="#1a1a1a", rotation=rot, zorder=10)
        # Draw bottom
        rot = rng.uniform(-cfg["rotation_max"], cfg["rotation_max"])
        ax.text(3.5, bot_y, bot_str, fontsize=28, fontweight="bold",
                ha="center", va="center", family=font,
                color="#1a1a1a", rotation=rot, zorder=10)
        # Horizontal divider
        edge = rng.choice(_BOX_EDGES)
        ax.plot([1.0, 6.0], [2.5, 2.5], color=edge, linewidth=2.5,
                zorder=5)
        ax.text(1.2, 2.2, "combine vertically", fontsize=9,
                ha="left", va="center", color="#7f8c8d", zorder=10,
                style="italic")
        # = ?
        ax.text(7.3, 2.5, "=", fontsize=32, fontweight="bold",
                ha="center", va="center", color="#1a1a1a", zorder=10)
        rect_q = mpatches.FancyBboxPatch(
            (8.0, 2.0), 1.3, 1.0, boxstyle="round,pad=0.05",
            facecolor="#fff3cd", edgecolor="#c0392b",
            linewidth=2.5, linestyle="--", zorder=3)
        ax.add_patch(rect_q)
        ax.text(8.65, 2.5, "?", fontsize=30, fontweight="bold",
                ha="center", va="center", color="#c0392b", zorder=10)

        title = rng.choice(self._TITLE_VARIANTS)
        ax.text(5, 4.6, title, fontsize=13, fontweight="bold",
                ha="center", color="#2c3e50", zorder=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _tok_display(tokens):
        out = []
        for t in tokens:
            if t == "*":
                out.append("\u00d7")
            elif t == "-":
                out.append("\u2212")
            else:
                out.append(t)
        return " ".join(out)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = TextRenderMathQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[text_render_math L{level} s{s}] FAILED")
                continue
            print(f"[text_render_math L{level} s{s}] A={env._answer}")
