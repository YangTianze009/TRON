"""Pivot table QA — cross-tabulation, row/column aggregation."""
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_lib import maybe_to_mcq_letter

class PivotTableQA(StandaloneVisualEnv):
    ENV_NAME = "pivot_table"
    
    _ROW_POOLS = [
        ["Alpha", "Beta", "Gamma", "Delta"],
        ["North", "South", "East", "West"],
        ["Dept A", "Dept B", "Dept C"],
    ]
    _COL_POOLS = [
        ["Q1", "Q2", "Q3", "Q4"],
        ["2021", "2022", "2023"],
        ["Jan", "Feb", "Mar"],
    ]
    # Iter-3: large 8x8 row/col pools for L9 ONLY. Attempt: 8 rows × 8 cols
    # (64 cells) so the model must track many items. Using 8 keeps rendering
    # clean; experiment showed 10x10 causes rendering issues.
    _L9_ROW_POOL = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    _L9_COL_POOL = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    def _level_config(self, level: int) -> Dict:
        # Difficulty redesign 2026-04-14. L0 = row_sum + col_sum (was L2-L4).
        if level <= 0:
            return {"qtypes": ["row_sum", "col_sum"],
                    "qweights": [5, 5], "val_range": (10, 70)}
        if level <= 2:
            return {"qtypes": ["grand_total", "which_row_highest", "col_average"],
                    "qweights": [3, 4, 3], "val_range": (10, 99)}
        if level <= 4:
            return {"qtypes": ["col_average", "which_row_highest", "conditional_average"],
                    "qweights": [3, 3, 4], "val_range": (10, 99)}
        if level <= 6:
            return {"qtypes": ["conditional_average", "rank_by_column"],
                    "qweights": [5, 5], "val_range": (10, 99)}
        if level <= 8:
            return {"qtypes": ["conditional_average", "rank_by_column",
                               "chained_lookup", "row_col_max_diff"],
                    "qweights": [2, 3, 3, 2], "val_range": (10, 99)}
        # L9 iter-3 (2026-04-17): large 8×8 grid + THREE-stage nested
        # aggregations. Model must track 64 cells AND combine filters with
        # per-row thresholds.
        return {"qtypes": ["three_stage_filter_sum",
                           "weighted_diag_conditional",
                           "col_mean_exceeds_row_count",
                           "triple_filter_product"],
                "qweights": [3, 3, 2, 2],
                "val_range": (10, 99), "big_grid": True}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 9901)

        question_type = parameter.get("question_type")
        _allowed = ["cell_value", "row_sum", "col_sum", "grand_total",
                    "row_max", "col_average", "which_row_highest",
                    "conditional_average", "rank_by_column",
                    "chained_lookup", "row_col_max_diff",
                    "rank_by_column_conditional",
                    "nested_sum_conditional", "two_stage_filter_avg",
                    "range_conditional",
                    "three_stage_filter_sum",
                    "weighted_diag_conditional",
                    "col_mean_exceeds_row_count",
                    "triple_filter_product"]
        if question_type not in _allowed:
            question_type = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        if cfg.get("big_grid"):
            rows = list(self._L9_ROW_POOL)
            cols = list(self._L9_COL_POOL)
        else:
            rows = sub_rng.choice(self._ROW_POOLS)
            cols = sub_rng.choice(self._COL_POOLS)
        nr, nc = len(rows), len(cols)
        vr = cfg["val_range"]
        data = [[sub_rng.randint(*vr) for _ in cols] for _ in rows]
        
        if question_type == "cell_value":
            r, c = rng.randint(0, nr-1), rng.randint(0, nc-1)
            question = f"What is the value in row '{rows[r]}', column '{cols[c]}'?"
            answer = data[r][c]
        elif question_type == "row_sum":
            r = rng.randint(0, nr-1)
            question = f"What is the sum of all values in row '{rows[r]}'?"
            answer = sum(data[r])
        elif question_type == "col_sum":
            c = rng.randint(0, nc-1)
            question = f"What is the sum of all values in column '{cols[c]}'?"
            answer = sum(data[r][c] for r in range(nr))
        elif question_type == "grand_total":
            question = "What is the grand total (sum of all values in the table)?"
            answer = sum(sum(row) for row in data)
        elif question_type == "row_max":
            r = rng.randint(0, nr-1)
            question = f"What is the maximum value in row '{rows[r]}'?"
            answer = max(data[r])
        elif question_type == "col_average":
            c = rng.randint(0, nc-1)
            avg = round(sum(data[r][c] for r in range(nr)) / nr, 1)
            question = f"What is the average of column '{cols[c]}'? Round to 1 decimal."
            answer = avg
        elif question_type == "which_row_highest":
            row_sums = [sum(data[r]) for r in range(nr)]
            best = rows[row_sums.index(max(row_sums))]
            question = "Which row has the highest total sum?"
            answer = best
        elif question_type == "conditional_average":
            # Average of cells > threshold in a specific column
            c = rng.randint(0, nc-1)
            thresh = rng.choice([30, 40, 50, 60])
            vals_above = [data[r][c] for r in range(nr) if data[r][c] > thresh]
            if not vals_above:
                # Lower threshold
                thresh = min(data[r][c] for r in range(nr))
                vals_above = [data[r][c] for r in range(nr) if data[r][c] > thresh]
            if not vals_above:
                return None
            avg = round(sum(vals_above) / len(vals_above), 1)
            question = (f"What is the average of values greater than {thresh} "
                       f"in column '{cols[c]}'? Round to 1 decimal.")
            answer = avg
        elif question_type == "rank_by_column":
            # Rank rows by a specific column, return the 2nd highest
            c = rng.randint(0, nc-1)
            col_vals = [(data[r][c], rows[r]) for r in range(nr)]
            col_vals.sort(reverse=True)
            if nr < 2:
                return None
            question = (f"If rows are ranked by column '{cols[c]}' from highest to lowest, "
                       f"which row is ranked 2nd?")
            answer = col_vals[1][1]
        elif question_type == "chained_lookup":
            # Step 1: find the row whose SUM is highest.
            # Step 2: in that row, report the VALUE in column c.
            # Requires reading every cell of every row.
            c = rng.randint(0, nc - 1)
            row_sums = [sum(data[r]) for r in range(nr)]
            best_r = row_sums.index(max(row_sums))
            question = (
                f"First identify the row with the highest total sum, "
                f"then report the value in column '{cols[c]}' for that row."
            )
            answer = data[best_r][c]
        elif question_type == "row_col_max_diff":
            # Difference between the highest and lowest row-totals.
            row_sums = [sum(data[r]) for r in range(nr)]
            answer = max(row_sums) - min(row_sums)
            question = (
                "Compute the row totals, then report the difference between "
                "the largest row total and the smallest row total."
            )
        elif question_type == "rank_by_column_conditional":
            # Among rows whose SUM is above the median, which has the
            # highest value in column c? Multi-step.
            c = rng.randint(0, nc - 1)
            row_sums = sorted([sum(data[r]) for r in range(nr)])
            if nr < 3:
                return None
            median = row_sums[nr // 2]
            cands = [(data[r][c], rows[r], sum(data[r]))
                     for r in range(nr) if sum(data[r]) > median]
            if not cands:
                # Fallback to >= median
                cands = [(data[r][c], rows[r], sum(data[r]))
                         for r in range(nr) if sum(data[r]) >= median]
            if not cands:
                return None
            cands.sort(reverse=True)
            answer = cands[0][1]
            question = (
                f"Consider only rows whose total sum is strictly greater than the median "
                f"of all row totals. Among those rows, which has the highest value "
                f"in column '{cols[c]}'?"
            )
        elif question_type == "nested_sum_conditional":
            # For rows whose value in column c_filter exceeds threshold,
            # compute the total SUM of their row totals.
            c_filter = rng.randint(0, nc - 1)
            thr = rng.choice([30, 40, 50, 60])
            qualifying = [r for r in range(nr) if data[r][c_filter] > thr]
            if not qualifying:
                # Relax threshold
                thr = min(data[r][c_filter] for r in range(nr))
                qualifying = [r for r in range(nr) if data[r][c_filter] > thr]
            if not qualifying:
                return None
            total_rs = sum(sum(data[r]) for r in qualifying)
            answer = total_rs
            question = (
                f"Compute row totals. For every row whose value in column "
                f"'{cols[c_filter]}' is strictly greater than {thr}, add its "
                f"row total to a running sum. What is the final sum?"
            )
        elif question_type == "two_stage_filter_avg":
            # Rows whose col_a > thr_a: compute average of their col_b values.
            if nc < 2:
                return None
            c_a, c_b = rng.sample(range(nc), 2)
            thr = rng.choice([30, 40, 50, 60])
            quals = [r for r in range(nr) if data[r][c_a] > thr]
            if not quals:
                thr = min(data[r][c_a] for r in range(nr))
                quals = [r for r in range(nr) if data[r][c_a] > thr]
            if not quals:
                return None
            avg_b = sum(data[r][c_b] for r in quals) / len(quals)
            answer = round(avg_b, 1)
            question = (
                f"Select only rows whose value in column '{cols[c_a]}' is "
                f"strictly greater than {thr}. For those rows, compute the "
                f"average of the values in column '{cols[c_b]}'. Round to "
                f"1 decimal."
            )
        elif question_type == "range_conditional":
            # For rows where col_a is in [lo, hi], sum col_b values.
            if nc < 2:
                return None
            c_a, c_b = rng.sample(range(nc), 2)
            lo, hi = rng.choice([(25, 55), (30, 70), (40, 80), (20, 60)])
            quals = [r for r in range(nr)
                     if lo <= data[r][c_a] <= hi]
            if not quals:
                # Force: find range that captures at least one row.
                vals = sorted(data[r][c_a] for r in range(nr))
                lo = vals[0]
                hi = vals[len(vals) // 2]
                quals = [r for r in range(nr)
                         if lo <= data[r][c_a] <= hi]
            if not quals:
                return None
            answer = sum(data[r][c_b] for r in quals)
            question = (
                f"For every row whose column '{cols[c_a]}' value is between "
                f"{lo} and {hi} inclusive, sum that row's column "
                f"'{cols[c_b]}' value. Report the total."
            )
        elif question_type == "three_stage_filter_sum":
            # L9 big-grid (8x8): 3-stage filter
            # (1) pick rows where sum > row-sum median
            # (2) among them, pick columns where col_mean > threshold
            # (3) sum cell values at the intersection
            if nc < 3 or nr < 3:
                return None
            row_sums = [sum(data[r]) for r in range(nr)]
            sorted_sums = sorted(row_sums)
            n = len(sorted_sums)
            if n % 2 == 0:
                row_median = (sorted_sums[n // 2 - 1] + sorted_sums[n // 2]) / 2
            else:
                row_median = sorted_sums[n // 2]
            kept_rows = [r for r in range(nr) if row_sums[r] > row_median]
            if not kept_rows:
                kept_rows = list(range(nr))
            col_thr = rng.choice([45, 50, 55, 60])
            col_means = [sum(data[r][c] for r in range(nr)) / nr
                         for c in range(nc)]
            kept_cols = [c for c in range(nc) if col_means[c] > col_thr]
            if not kept_cols:
                # relax threshold
                col_thr = min(col_means) - 1
                kept_cols = [c for c in range(nc) if col_means[c] > col_thr]
            if not kept_cols:
                return None
            total = sum(data[r][c] for r in kept_rows for c in kept_cols)
            answer = total
            question = (
                f"STEP 1: compute each row's total sum. Let M be the MEDIAN "
                f"of these row totals. STEP 2: identify rows whose total "
                f"is STRICTLY GREATER than M. STEP 3: compute each column's "
                f"MEAN across ALL {nr} rows; keep only columns whose mean "
                f"is strictly greater than {col_thr}. STEP 4: sum the cell "
                f"values at the intersection of the kept rows and kept "
                f"columns. Return the total as an integer."
            )
        elif question_type == "weighted_diag_conditional":
            # For each row r, take its MAX column value; weight by (r+1);
            # conditionally include the term ONLY if the row's SUM is > 300.
            # Sum weighted-max-products over surviving rows. Integer answer.
            row_sums = [sum(data[r]) for r in range(nr)]
            thr = rng.choice([250, 300, 350])
            # If too few rows pass threshold, relax
            while sum(1 for s in row_sums if s > thr) < 2:
                thr -= 50
                if thr < 100:
                    break
            total = 0
            for r in range(nr):
                if row_sums[r] > thr:
                    total += max(data[r]) * (r + 1)
            answer = total
            question = (
                f"For each of the {nr} rows, compute the row SUM. Consider "
                f"only the rows whose sum is STRICTLY GREATER than {thr}. "
                f"For each such row r (1-indexed from the TOP), find the "
                f"MAXIMUM cell value in that row, and multiply it by r "
                f"(the row index, starting at 1). Sum these products over "
                f"all surviving rows. Report an integer."
            )
        elif question_type == "col_mean_exceeds_row_count":
            # Count the number of (row, col) cells such that the cell
            # STRICTLY exceeds the mean of its column. Answer is integer
            # count.
            col_means = [sum(data[r][c] for r in range(nr)) / nr
                         for c in range(nc)]
            count = 0
            for r in range(nr):
                for c in range(nc):
                    if data[r][c] > col_means[c]:
                        count += 1
            answer = count
            question = (
                f"For each column, compute the MEAN of its {nr} values. "
                f"Then count how many cells (across the ENTIRE {nr}×{nc} "
                f"table) have a value STRICTLY GREATER than their own "
                f"column's mean. Return the COUNT as an integer."
            )
        elif question_type == "triple_filter_product":
            # Compute: product of cell values where
            #   (a) cell > row_mean
            #   (b) cell > col_mean
            #   (c) cell ends in an EVEN digit (units place is even).
            # If fewer than 2 cells satisfy, relax. Product too large → mod.
            row_means = [sum(row) / nc for row in data]
            col_means = [sum(data[r][c] for r in range(nr)) / nr
                         for c in range(nc)]
            eligible = []
            for r in range(nr):
                for c in range(nc):
                    v = data[r][c]
                    if (v > row_means[r] and v > col_means[c]
                            and (v % 10) % 2 == 0):
                        eligible.append(v)
            if len(eligible) < 2:
                return None
            # SUM (not product) to keep answer bounded.
            answer = sum(eligible)
            question = (
                f"Consider every cell in the {nr}×{nc} table. Compute each "
                f"row's MEAN and each column's MEAN. Find all cells that "
                f"satisfy ALL THREE conditions simultaneously: "
                f"(1) cell value > its row mean, "
                f"(2) cell value > its column mean, "
                f"(3) the cell's units digit is even (i.e. the value mod "
                f"10 is in {{0,2,4,6,8}}). SUM all such cells and report "
                f"the integer total."
            )
        else:
            return None
        
        # Render table
        style = self._random_style()
        # Bigger figure for big 8×8 grid so font stays readable.
        fig_w = max(5, nc * 1.0 + 2)
        fig_h = max(3, nr * 0.6 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.axis("off")

        cell_text = [row[:] for row in data]
        table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        # Scale down font for larger grids.
        fs = style["font_size_base"] if nr <= 4 else max(8, style["font_size_base"] - 2)
        table.set_fontsize(fs)
        table.scale(1, 1.2 if nr > 6 else 1.5)
        
        # Style
        palette = style["palette"]
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(palette[0])
                cell.set_text_props(color='white', fontweight='bold')
            elif c == -1:
                cell.set_facecolor(palette[1])
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#f8f9fa')
            cell.set_edgecolor('#7f8c8d')
        
        ax.set_title("Pivot Table", fontsize=style["font_size_base"] + 2, fontweight="bold", pad=20)

        # MCQ-letter style MCQ-letter mode: with prob 0.5, convert to MCQ.
        # which_row_highest / rank_by_column / rank_by_column_conditional return
        # row labels — use rows as candidate pool.
        ans_str = str(answer)
        n_opts = sub_rng.choice([4, 5])
        if question_type in ("which_row_highest", "rank_by_column",
                             "rank_by_column_conditional"):
            cand_pool = list(rows)
        else:
            cand_pool = None
        question, ans_str = maybe_to_mcq_letter(
            question, ans_str, sub_rng, prob=1.0, n_options=n_opts,
            candidate_pool=cand_pool)
        return question, ans_str, self.fig_to_pil(fig, dpi=style["dpi"])
