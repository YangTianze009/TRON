"""
Table Cell Lookup QA.

Target: ocr-bench KIE / Doc-VQA. A small table is rendered as an
image; the model must locate a cell by row/column name and report its
value, or perform a small operation (max/min/sum/row-sum) on a row/column.

Difficulty axes:
  L0-L1: 3x3 table, direct cell lookup, small numeric values (< 50)
  L2-L3: 4x4 table, direct lookup + single-row maximum
  L4-L5: 5x5 table, row sum / column sum
  L6-L7: 6x6 table, max/min across entire table, which row has highest total
  L8-L9: 6-7 col table, compound operations: max of one column minus min of
         another column; or "how many cells > threshold T (shown as label)"

Question types:
  - direct_lookup  : "what is row R, column C?"
  - row_max        : "what is the maximum value in row R?"
  - col_max        : "what is the maximum value in column C?"
  - row_sum        : "what is the sum of values in row R?"
  - col_sum        : "what is the sum of values in column C?"
  - table_max      : "what is the maximum value in the entire table?"
  - which_row_max  : "which row has the highest total?" (answer: row label)
  - which_col_max  : "which column has the highest total?"
  - count_above    : "how many cells have value > T?" (T rendered on image)
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
from ._mcq_letter_lib import maybe_to_mcq_letter

_ROW_LABELS = ["Apple", "Pear", "Peach", "Grape", "Mango", "Lemon",
               "Berry", "Kiwi", "Plum", "Papaya",
               "Ruby", "Onyx", "Opal", "Jade", "Coral"]
_COL_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
               "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL"]
_HEADER_COLORS = ["#dde6ef", "#eef2e3", "#f8e6cc", "#e0d8f0",
                  "#d9efe6", "#fceecd", "#dbe9f2"]

class TableCellLookupQA(StandaloneVisualEnv):
    ENV_NAME = "table_cell_lookup"

    QUESTION_TYPES = [
        "direct_lookup", "row_max", "col_max", "row_sum", "col_sum",
        "table_max", "which_row_max", "which_col_max", "count_above",
        # Harder multi-step qtypes (L8-L9): require locating >1 cell and
        # combining them with arithmetic.
        "col_max_minus_col_min",   # max(col_A) - min(col_B)
        "second_largest",          # 2nd largest value in the whole table
        "row_sum_gt_threshold",    # how many rows have sum > T
    ]

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per ocr-bench/doc-vqa.
        # L0: 3x3 direct lookup (1-step)
        # L1: 3x3 + row_max (1-step single-row)
        # L2: 4x4 direct/row_max/col_max
        # L3: 4x4 row_sum/col_sum (1-step aggregation)
        # L4: 5x5 row_sum/col_sum + table_max
        # L5: 5x5 which_row_max/which_col_max (compare aggregates)
        # L6: 6x6 with handwritten_nums (OCR difficulty)
        # L7: 6x6 with similar_values clusters (close-value detection)
        # L8: 7x7 multi-step compound (col_max_minus_col_min)
        # L9: 7-8x7-8 second_largest / row_sum_gt_threshold (heavy DP)
        level = max(0, min(level, 9))
        if level == 0:
            return {
                "n_rows_range": (3, 3), "n_cols_range": (3, 3),
                "val_max": 30, "qtypes": ["direct_lookup"],
                "header_noise": False,
            }
        if level == 1:
            return {
                "n_rows_range": (3, 3), "n_cols_range": (3, 3),
                "val_max": 50, "qtypes": ["direct_lookup", "row_max"],
                "header_noise": False,
            }
        if level == 2:
            return {
                "n_rows_range": (4, 4), "n_cols_range": (4, 4),
                "val_max": 80,
                "qtypes": ["direct_lookup", "row_max", "col_max"],
                "header_noise": False,
            }
        if level == 3:
            return {
                "n_rows_range": (4, 4), "n_cols_range": (4, 4),
                "val_max": 100,
                "qtypes": ["row_sum", "col_sum"],
                "header_noise": False,
            }
        if level == 4:
            return {
                "n_rows_range": (5, 5), "n_cols_range": (5, 5),
                "val_max": 150,
                "qtypes": ["row_sum", "col_sum", "table_max"],
                "header_noise": False,
            }
        if level == 5:
            return {
                "n_rows_range": (5, 5), "n_cols_range": (5, 5),
                "val_max": 180,
                "qtypes": ["which_row_max", "which_col_max"],
                "header_noise": True,
            }
        if level == 6:
            return {
                "n_rows_range": (6, 6), "n_cols_range": (6, 6),
                "val_max": 200,
                "qtypes": ["row_sum", "col_sum", "which_row_max",
                           "which_col_max"],
                "header_noise": True,
                "handwritten_nums": True,
            }
        if level == 7:
            return {
                "n_rows_range": (6, 6), "n_cols_range": (6, 6),
                "val_max": 220,
                "qtypes": ["table_max", "which_row_max", "which_col_max"],
                "header_noise": True,
                "similar_values": True,
                "handwritten_nums": True,
            }
        if level == 8:
            return {
                "n_rows_range": (7, 7), "n_cols_range": (7, 7),
                "val_max": 350,
                "qtypes": ["col_max_minus_col_min", "second_largest"],
                "header_noise": True,
                "similar_values": True,
                "handwritten_nums": True,
            }
        # L9
        return {
            "n_rows_range": (7, 8), "n_cols_range": (7, 8),
            "val_max": 500,
            "qtypes": ["col_max_minus_col_min", "second_largest",
                       "row_sum_gt_threshold"],
            "header_noise": True,
            "similar_values": True,
            "handwritten_nums": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = rng.choice(cfg["qtypes"])

        for _ in range(25):
            r = self._try_generate(rng, cfg, qtype)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, qtype):
        n_r = rng.randint(*cfg["n_rows_range"])
        n_c = rng.randint(*cfg["n_cols_range"])
        if n_r > len(_ROW_LABELS) or n_c > len(_COL_LABELS):
            return None
        rows = rng.sample(_ROW_LABELS, n_r)
        cols = rng.sample(_COL_LABELS, n_c)

        # At L6-L9 generate clusters of close-valued cells so OCR errors
        # or slight misreads lead to wrong answers.
        if cfg.get("similar_values"):
            # Pick a base and generate values in a tight window with
            # occasional outliers.
            base = rng.randint(cfg["val_max"] // 2, cfg["val_max"])
            data = []
            for _ in range(n_r):
                row = []
                for _ in range(n_c):
                    if rng.random() < 0.15:
                        # occasional outlier
                        row.append(rng.randint(1, cfg["val_max"]))
                    else:
                        row.append(max(1, base + rng.randint(-20, 20)))
                data.append(row)
        else:
            data = [[rng.randint(1, cfg["val_max"]) for _ in range(n_c)]
                    for _ in range(n_r)]

        # Dispatch
        threshold = None
        target_cell = None

        if qtype == "direct_lookup":
            ri = rng.randint(0, n_r - 1)
            ci = rng.randint(0, n_c - 1)
            answer = str(data[ri][ci])
            target_cell = (ri, ci)
            q = (f"In the table shown, what is the value at row "
                 f"'{rows[ri]}' and column '{cols[ci]}'? Answer with an integer.")

        elif qtype == "row_max":
            ri = rng.randint(0, n_r - 1)
            # Ensure a clear maximum (diff >=2)
            vals = data[ri]
            s = sorted(vals, reverse=True)
            if s[0] - s[1] < 2:
                return None
            answer = str(max(vals))
            q = (f"In the table shown, what is the MAXIMUM value in row "
                 f"'{rows[ri]}'? Answer with an integer.")

        elif qtype == "col_max":
            ci = rng.randint(0, n_c - 1)
            vals = [data[i][ci] for i in range(n_r)]
            s = sorted(vals, reverse=True)
            if s[0] - s[1] < 2:
                return None
            answer = str(max(vals))
            q = (f"In the table shown, what is the MAXIMUM value in column "
                 f"'{cols[ci]}'? Answer with an integer.")

        elif qtype == "row_sum":
            ri = rng.randint(0, n_r - 1)
            answer = str(sum(data[ri]))
            q = (f"In the table shown, what is the SUM of all values "
                 f"in row '{rows[ri]}'? Answer with an integer.")

        elif qtype == "col_sum":
            ci = rng.randint(0, n_c - 1)
            answer = str(sum(data[i][ci] for i in range(n_r)))
            q = (f"In the table shown, what is the SUM of all values "
                 f"in column '{cols[ci]}'? Answer with an integer.")

        elif qtype == "table_max":
            flat = [v for row in data for v in row]
            s = sorted(flat, reverse=True)
            if s[0] - s[1] < 2:
                return None
            answer = str(max(flat))
            q = ("In the table shown, what is the MAXIMUM value in the "
                 "entire table? Answer with an integer.")

        elif qtype == "which_row_max":
            totals = [sum(row) for row in data]
            s = sorted(totals, reverse=True)
            if s[0] - s[1] < 5:
                return None
            idx = totals.index(max(totals))
            answer = rows[idx]
            q = ("In the table shown, which row has the HIGHEST total "
                 "(sum of its values)? Answer with the row label.")

        elif qtype == "which_col_max":
            totals = [sum(data[i][c] for i in range(n_r)) for c in range(n_c)]
            s = sorted(totals, reverse=True)
            if s[0] - s[1] < 5:
                return None
            idx = totals.index(max(totals))
            answer = cols[idx]
            q = ("In the table shown, which column has the HIGHEST total "
                 "(sum of its values)? Answer with the column label.")

        elif qtype == "count_above":
            # Threshold is placed ON the image
            mid = cfg["val_max"] // 2
            t = rng.randint(int(mid * 0.7), int(mid * 1.2))
            cnt = sum(1 for row in data for v in row if v > t)
            # Ensure the count is meaningful (not 0 or n_r*n_c)
            if cnt == 0 or cnt == n_r * n_c:
                return None
            answer = str(cnt)
            threshold = t
            q = ("The table is shown with a threshold value T labeled on "
                 "the image. How many cells in the table have a value "
                 "STRICTLY GREATER than T? Answer with an integer.")

        elif qtype == "col_max_minus_col_min":
            # Multi-step: max of column A minus min of column B.
            if n_c < 2:
                return None
            ci_a, ci_b = rng.sample(range(n_c), 2)
            col_a = [data[i][ci_a] for i in range(n_r)]
            col_b = [data[i][ci_b] for i in range(n_r)]
            # Disambiguate each extremum (diff >=2 from runner-up).
            sa = sorted(col_a, reverse=True)
            sb = sorted(col_b)
            if sa[0] - sa[1] < 2 or sb[1] - sb[0] < 2:
                return None
            answer = str(max(col_a) - min(col_b))
            q = (f"In the table shown, compute: (MAXIMUM value in column "
                 f"'{cols[ci_a]}') minus (MINIMUM value in column "
                 f"'{cols[ci_b]}'). Answer with an integer.")

        elif qtype == "second_largest":
            # Multi-step: 2nd largest value across the entire table.
            flat = sorted([v for row in data for v in row], reverse=True)
            # Disambiguate 1st, 2nd, 3rd: want flat[0] > flat[1] > flat[2].
            if not (flat[0] - flat[1] >= 2 and flat[1] - flat[2] >= 2):
                return None
            answer = str(flat[1])
            q = ("In the table shown, what is the SECOND LARGEST value in "
                 "the entire table (the largest value that is strictly "
                 "less than the overall maximum)? Answer with an integer.")

        elif qtype == "row_sum_gt_threshold":
            # Multi-step: how many rows have sum > T? Threshold on image.
            sums = [sum(row) for row in data]
            # Choose T between two row-sums so count > 0 and < n_r.
            avg = sum(sums) // len(sums)
            t = rng.randint(int(avg * 0.85), int(avg * 1.10))
            cnt = sum(1 for s in sums if s > t)
            if cnt == 0 or cnt == n_r:
                return None
            # Make sure T is not equal to any row sum (avoids boundary).
            if any(s == t for s in sums):
                return None
            answer = str(cnt)
            threshold = t
            q = ("The table is shown with a threshold value T labeled on "
                 "the image. For each row, compute its SUM. How many rows "
                 "have a sum STRICTLY GREATER than T? Answer with an "
                 "integer.")

        else:
            return None

        image = self._render(rows, cols, data, cfg, rng,
                             target=target_cell, threshold=threshold)

        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        # which_row_max / which_col_max return labels — use rows / cols as
        # candidate pool. Other qtypes return numeric strings.
        n_opts = rng.choice([4, 5])
        if qtype == "which_row_max":
            cand_pool = list(rows)
        elif qtype == "which_col_max":
            cand_pool = list(cols)
        else:
            cand_pool = None
        q, answer = maybe_to_mcq_letter(
            q, answer, rng, prob=1.0, n_options=n_opts,
            candidate_pool=cand_pool)
        return q, answer, image

    # -------------------------------------------------------------- #
    def _render(self, rows, cols, data, cfg, rng, target=None,
                threshold=None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]

        n_r = len(rows)
        n_c = len(cols)

        fig_w = 1.4 + 1.2 * n_c
        fig_h = 1.0 + 0.7 * n_r + 0.8
        if threshold is not None:
            fig_h += 0.5
        fig, ax = plt.subplots(figsize=(fig_w * sc, fig_h * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        cell_w = 1.0
        cell_h = 0.7
        header_color = rng.choice(_HEADER_COLORS)

        # Column headers
        for j in range(n_c):
            rect = mpatches.Rectangle(
                ((j + 1) * cell_w, n_r * cell_h),
                cell_w, cell_h,
                facecolor=header_color, edgecolor="#333", linewidth=1.2)
            ax.add_patch(rect)
            ax.text((j + 1) * cell_w + cell_w / 2,
                    n_r * cell_h + cell_h / 2,
                    cols[j], ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold")

        # Row headers
        for i in range(n_r):
            rect = mpatches.Rectangle(
                (0, (n_r - 1 - i) * cell_h),
                cell_w, cell_h,
                facecolor=header_color, edgecolor="#333", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(cell_w / 2,
                    (n_r - 1 - i) * cell_h + cell_h / 2,
                    rows[i], ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold")

        # Data cells — at L6-L9, use handwritten-style rotated numbers
        # with font/size jitter per cell.
        handwritten = cfg.get("handwritten_nums", False)
        for i in range(n_r):
            for j in range(n_c):
                fc = "#ffffff"
                rect = mpatches.Rectangle(
                    ((j + 1) * cell_w, (n_r - 1 - i) * cell_h),
                    cell_w, cell_h,
                    facecolor=fc, edgecolor="#333", linewidth=1.0)
                ax.add_patch(rect)
                if handwritten:
                    rot = rng.uniform(-15, 15)
                    fj = rng.choice(["serif", "DejaVu Sans", "monospace"])
                    fs_jit = fs + 1 + rng.randint(-2, 2)
                    ax.text((j + 1) * cell_w + cell_w / 2,
                            (n_r - 1 - i) * cell_h + cell_h / 2,
                            str(data[i][j]),
                            ha="center", va="center",
                            fontsize=fs_jit, rotation=rot,
                            family=fj, fontweight="bold",
                            color=rng.choice(["#1a1a2a", "#2a1a1a",
                                              "#1a2a1a", "#333"]))
                else:
                    ax.text((j + 1) * cell_w + cell_w / 2,
                            (n_r - 1 - i) * cell_h + cell_h / 2,
                            str(data[i][j]),
                            ha="center", va="center",
                            fontsize=fs + 1)

        # Optional title/threshold label
        y_top = (n_r + 1) * cell_h + 0.2
        if threshold is not None:
            ax.text((n_c + 1) * cell_w / 2, y_top,
                    f"T = {threshold}",
                    ha="center", va="center", fontsize=fs + 3,
                    fontweight="bold", color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#fff3cd", edgecolor="#c0392b"))
            y_top += 0.5

        ax.set_xlim(-0.1, (n_c + 1) * cell_w + 0.1)
        ax.set_ylim(-0.1, y_top + 0.1)
        ax.set_aspect("equal")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # -------------------------------------------------------------- #
    def _check_answer(self, predicted, ground_truth):
        p = predicted.strip().lower().rstrip(".")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        # Numeric tolerance
        try:
            return abs(float(p) - float(g)) < 0.5
        except ValueError:
            pass
        return super()._check_answer(predicted, ground_truth)
