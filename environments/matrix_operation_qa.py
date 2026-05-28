"""Matrix Operation QA — diversity + difficulty redesign 2026-04-16.

Round-2 fix: pass rate was near-saturated → introduce structurally different
operations per level and substantial visual diversity (layout, colours, font,
bracket style).

Level map (structural, not parametric, differences):
  L0: 2x2 trace (literally sum of two values)
  L1: 2x2 determinant
  L2: 3x3 transpose element (index lookup)
  L3: 3x3 determinant
  L4: 2x2 multiply element (single dot product)
  L5: 2x2 A*B multiplication — sum of full product matrix
  L6: 3x3 trace after B = A^T, i.e. trace(A * A^T)
  L7: 3x3 A*B — specific element
  L8: 3x3 A*B — diagonal sum (trace of product)
  L9: 3x3 A*B*C triple multiply — specific element (hardest)
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _mat_mul(A, B, n):
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]

def _det2(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]

def _det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
          - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
          + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))

def _transpose(M, n):
    return [[M[j][i] for j in range(n)] for i in range(n)]

class MatrixOperationQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "matrix_operation"

    QUESTION_TYPES = [
        "trace_2", "determinant_2", "transpose_element_3", "determinant_3",
        "multiply_element_2", "product_sum_2",
        "trace_AAT_3", "multiply_element_3", "diag_sum_3",
        "triple_product_element_3",
        # reference D9 extension
        "sum_all_3",
    ]

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, int(level)))
        # reference D9 (sum of all elements): rotated in at L2, L4, L6
        # for 1/4 of seeds.
        seed_mod = (self.seed or 0) % 4
        base = {
            0: {"qt": "trace_2", "n": 2},
            1: {"qt": "determinant_2", "n": 2},
            2: {"qt": "transpose_element_3", "n": 3},
            3: {"qt": "determinant_3", "n": 3},
            4: {"qt": "multiply_element_2", "n": 2},
            5: {"qt": "product_sum_2", "n": 2},
            6: {"qt": "trace_AAT_3", "n": 3},
            7: {"qt": "multiply_element_3", "n": 3},
            8: {"qt": "diag_sum_3", "n": 3},
            9: {"qt": "triple_product_element_3", "n": 3},
        }[level]
        if level in (2, 4, 6) and seed_mod == 0:
            return {"qt": "sum_all_3", "n": 3}
        return base

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 719)
        qt = lcfg["qt"]
        n = lcfg["n"]
        # Remember level for rendering
        self._mat_level = level

        # Random value range per-level
        lo, hi = self._value_range(level)

        A = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        B = [[rng.randint(lo // 2, hi // 2 + 1) for _ in range(n)] for _ in range(n)]

        q = a = None
        imgs = None

        if qt == "trace_2":
            tr = sum(A[i][i] for i in range(n))
            q = self._phrase_trace(n, rng)
            a = str(tr)
            imgs = ("single", A)
        elif qt == "determinant_2":
            det = _det2(A)
            q = self._phrase_det(n, rng)
            a = str(det)
            imgs = ("single", A)
        elif qt == "transpose_element_3":
            i, j = rng.randint(0, n-1), rng.randint(0, n-1)
            val = A[j][i]
            q = self._phrase_transpose(i, j, rng)
            a = str(val)
            imgs = ("single", A)
        elif qt == "determinant_3":
            det = _det3(A)
            q = self._phrase_det(n, rng)
            a = str(det)
            imgs = ("single", A)
        elif qt == "multiply_element_2":
            ri, ci = rng.randint(0, n-1), rng.randint(0, n-1)
            val = sum(A[ri][k] * B[k][ci] for k in range(n))
            q = self._phrase_multiply_element(ri, ci, rng)
            a = str(val)
            imgs = ("pair", A, B)
        elif qt == "product_sum_2":
            P = _mat_mul(A, B, n)
            s = sum(P[i][j] for i in range(n) for j in range(n))
            q = self._phrase_product_sum(rng)
            a = str(s)
            imgs = ("pair", A, B)
        elif qt == "trace_AAT_3":
            AAT = _mat_mul(A, _transpose(A, n), n)
            tr = sum(AAT[i][i] for i in range(n))
            q = ("Compute the trace (sum of main-diagonal entries) of the product "
                 "A * A-transpose, where A is the 3x3 matrix shown.")
            a = str(tr)
            imgs = ("single", A)
        elif qt == "multiply_element_3":
            ri, ci = rng.randint(0, n-1), rng.randint(0, n-1)
            val = sum(A[ri][k] * B[k][ci] for k in range(n))
            q = self._phrase_multiply_element(ri, ci, rng)
            a = str(val)
            imgs = ("pair", A, B)
        elif qt == "diag_sum_3":
            P = _mat_mul(A, B, n)
            ds = sum(P[i][i] for i in range(n))
            q = ("Compute the product A * B of the two 3x3 matrices shown, then "
                 "give the trace (sum of the main diagonal) of the result.")
            a = str(ds)
            imgs = ("pair", A, B)
        elif qt == "triple_product_element_3":
            C = [[rng.randint(-3, 3) for _ in range(n)] for _ in range(n)]
            AB = _mat_mul(A, B, n)
            ABC = _mat_mul(AB, C, n)
            ri, ci = rng.randint(0, n-1), rng.randint(0, n-1)
            val = ABC[ri][ci]
            q = (f"Three 3x3 matrices A, B, C are shown. Compute the product "
                 f"A * B * C and give the element at row {ri+1}, column {ci+1}.")
            a = str(val)
            imgs = ("triple", A, B, C)
        elif qt == "sum_all_3":
            # reference D9 (an external reference, 477): "What is the sum of all elements
            # in the matrix A?"  Integer answer (3x3).
            s = sum(A[i][j] for i in range(n) for j in range(n))
            q = rng.choice([
                f"What is the sum of all elements in the {n}x{n} matrix A shown?",
                f"Compute the sum of every entry of the {n}x{n} matrix A "
                f"displayed. Integer.",
                f"Add up all entries of A (the {n}x{n} matrix in the image). "
                f"What is the sum?",
            ])
            a = str(s)
            imgs = ("single", A)
        else:
            return None

        # Render based on imgs
        if imgs[0] == "single":
            img = self._render_matrices([("A", imgs[1])], rng)
        elif imgs[0] == "pair":
            img = self._render_matrices([("A", imgs[1]), ("B", imgs[2])], rng)
        else:
            img = self._render_matrices([("A", imgs[1]), ("B", imgs[2]), ("C", imgs[3])], rng)
        return q, a, img

    @staticmethod
    def _value_range(level: int) -> Tuple[int, int]:
        if level <= 1:
            return (-5, 5)
        if level <= 3:
            return (-6, 6)
        if level <= 5:
            return (-7, 7)
        if level <= 7:
            return (-9, 9)
        # 2026-05-04: bumped L9 difficulty (also L8) — was 92.5% saturated.
        if level == 8:
            return (-12, 12)
        return (-15, 15)

    # ------------------------------------------------------------------ #
    # Phrasing
    # ------------------------------------------------------------------ #

    def _phrase_trace(self, n, rng):
        return rng.choice([
            f"What is the TRACE (sum of main-diagonal entries) of the {n}x{n} matrix A shown?",
            f"Compute tr(A) for the {n}x{n} matrix A displayed. Integer.",
            f"Add the diagonal entries of the {n}x{n} matrix A shown. What is the sum?",
            f"Find the sum A[1,1] + A[2,2]" + (f" + A[3,3]" if n == 3 else "") + ". Integer.",
        ])

    def _phrase_det(self, n, rng):
        return rng.choice([
            f"What is the DETERMINANT of the {n}x{n} matrix A shown?",
            f"Compute det(A) for the {n}x{n} matrix A displayed. Integer.",
            f"Evaluate the determinant of the {n}x{n} matrix A (as shown). Integer.",
            f"Find |A| for the {n}x{n} matrix A displayed.",
        ])

    def _phrase_transpose(self, i, j, rng):
        return rng.choice([
            f"What value appears at row {i+1}, column {j+1} of A-transpose, "
            f"where A is the matrix shown?",
            f"Compute A^T for the shown matrix A; what is the entry at "
            f"position ({i+1}, {j+1})?",
            f"Given the matrix A in the image, give the element at row "
            f"{i+1}, column {j+1} of the transpose.",
            f"The transpose of A has which value at (row {i+1}, col {j+1})?",
        ])

    def _phrase_multiply_element(self, ri, ci, rng):
        return rng.choice([
            f"Compute A * B for the matrices shown. What is the entry at row {ri+1}, column {ci+1}?",
            f"Let P = A * B. Give P[{ri+1}, {ci+1}] where A and B are the matrices displayed.",
            f"Multiply the matrices A and B shown; report the value at position ({ri+1}, {ci+1}) of the product.",
            f"What is element ({ri+1}, {ci+1}) of the matrix product A * B (matrices shown)?",
        ])

    def _phrase_product_sum(self, rng):
        return rng.choice([
            "Compute A * B for the two matrices shown, and give the sum of ALL entries of the product.",
            "Multiply the matrices A and B (as shown), then add every entry of the result. Integer.",
            "Let P = A * B (matrices displayed). Report the sum of all elements of P.",
            "The two matrices are A and B. Compute A*B and give the total sum of its entries.",
        ])

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_matrices(self, mats: List[Tuple[str, List[List[int]]]], rng) -> Image.Image:
        style = self._random_style()
        n_mats = len(mats)
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        palette = list(style["palette"])
        rng.shuffle(palette)

        # Layout: horizontal row, unless 3 matrices and random chooses vertical stack
        layout = rng.choice(["horizontal", "horizontal", "grid"])
        if n_mats <= 2 or layout == "horizontal":
            fig_w = (3.6 * n_mats + 0.4) * sc
            fig_h = 4.2 * sc
            fig, axes = plt.subplots(1, n_mats, figsize=(fig_w, fig_h))
            if n_mats == 1:
                axes = [axes]
        else:  # 3 matrices, 2-row layout
            fig_w = 8 * sc; fig_h = 7.8 * sc
            fig = plt.figure(figsize=(fig_w, fig_h))
            axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 3)]

        fig.patch.set_facecolor(style["bg_color"])

        for ax_i, (name, M) in enumerate(mats):
            ax = axes[ax_i]
            n = len(M)
            self._draw_matrix(ax, M, n, f"Matrix {name}", style, palette, fs, ff, rng)
            ax.set_facecolor(style["bg_color"])

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_matrix(self, ax, M, n, title, style, palette, fs, ff, rng):
        ax.set_title(title, fontsize=fs + 4, fontweight="bold", fontfamily=ff, pad=10)
        ax.set_aspect("equal")
        ax.axis("off")

        level = getattr(self, "_mat_level", 0)
        # At L6+ render as a heatmap with small (non-dominant) numbers so
        # values can't be OCR'd cleanly — the model must read magnitudes
        # from colour intensity.
        heatmap_mode = level >= 6

        # Bracket style: square brackets or parentheses
        bracket_style = rng.choice(["square", "paren"])
        bx_min, bx_max = -0.5, n - 0.5
        by_min, by_max = -0.5, n - 0.5

        if bracket_style == "square":
            lw = 2.5
            ax.plot([bx_min - 0.25, bx_min - 0.25], [by_min - 0.25, by_max + 0.25],
                    color="#222", linewidth=lw)
            ax.plot([bx_min - 0.25, bx_min - 0.05], [by_min - 0.25, by_min - 0.25],
                    color="#222", linewidth=lw)
            ax.plot([bx_min - 0.25, bx_min - 0.05], [by_max + 0.25, by_max + 0.25],
                    color="#222", linewidth=lw)
            ax.plot([bx_max + 0.25, bx_max + 0.25], [by_min - 0.25, by_max + 0.25],
                    color="#222", linewidth=lw)
            ax.plot([bx_max + 0.05, bx_max + 0.25], [by_min - 0.25, by_min - 0.25],
                    color="#222", linewidth=lw)
            ax.plot([bx_max + 0.05, bx_max + 0.25], [by_max + 0.25, by_max + 0.25],
                    color="#222", linewidth=lw)
        else:
            # Parentheses: approximate arcs
            import matplotlib.patches as mpatches
            ax.add_patch(mpatches.Arc(
                ((bx_min + bx_max) / 2, (by_min + by_max) / 2),
                width=(bx_max - bx_min) + 1.0, height=(by_max - by_min) + 1.2,
                theta1=90, theta2=270, edgecolor="#222", linewidth=2.2))
            ax.add_patch(mpatches.Arc(
                ((bx_min + bx_max) / 2, (by_min + by_max) / 2),
                width=(bx_max - bx_min) + 1.0, height=(by_max - by_min) + 1.2,
                theta1=-90, theta2=90, edgecolor="#222", linewidth=2.2))

        if heatmap_mode:
            import matplotlib.patches as mpatches_local
            flat = [v for row in M for v in row]
            lo = min(flat)
            hi = max(flat)
            span = hi - lo if hi != lo else 1
            for i in range(n):
                for j in range(n):
                    v = M[i][j]
                    t = (v - lo) / span  # 0..1
                    # red-to-blue diverging colour; near zero = white
                    if v >= 0:
                        r = 1.0 - t * 0.35
                        g = 1.0 - t * 0.55
                        b = 1.0 - t * 0.25
                    else:
                        r = 1.0 + t * 0.05
                        g = 1.0 - t * 0.15
                        b = 1.0 - t * 0.05
                    r = max(0.2, min(1.0, r))
                    g = max(0.2, min(1.0, g))
                    b = max(0.2, min(1.0, b))
                    y_plot = (n - 1 - i)
                    rect = mpatches_local.Rectangle(
                        (j - 0.45, y_plot - 0.45), 0.9, 0.9,
                        facecolor=(r, g, b), edgecolor="#888",
                        linewidth=0.8)
                    ax.add_patch(rect)
                    # Values rendered clearly but non-dominant; use
                    # dark/light contrast based on cell brightness so
                    # numbers stay legible. (Previously alpha=0.55 fs-3
                    # made values unreadable — rendered the problem
                    # unsolvable at L6+.)
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b
                    txt_color = "#111" if brightness > 0.55 else "#ffffff"
                    ax.text(j, y_plot, str(v),
                            ha="center", va="center",
                            fontsize=fs + 2, fontweight="bold",
                            color=txt_color,
                            fontfamily=ff)
        else:
            # Cell fonts per-seed: bold/italic mix.
            # Bugfix 2026-04-17: prior palette used very pale pastel hues
            # so numbers were nearly illegible on white bg. Use a dark,
            # saturated set independent of style palette.
            is_bold = rng.random() < 0.7
            dark_palette = ["#1b1b1b", "#1f3a5f", "#4a148c", "#1b5e20",
                            "#b71c1c", "#37474f", "#4e342e"]
            for i in range(n):
                for j in range(n):
                    col = dark_palette[(i * n + j) % len(dark_palette)]
                    # Matrix is conventionally shown with row 1 at top — flip y
                    y_plot = (n - 1 - i)
                    ax.text(j, y_plot, str(M[i][j]),
                            ha="center", va="center",
                            fontsize=fs + 6, fontweight="bold" if is_bold else "normal",
                            color=col, fontfamily=ff)

        pad = 0.8
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)
