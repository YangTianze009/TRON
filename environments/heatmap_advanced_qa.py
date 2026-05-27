"""Heatmap Visual QA Environment."""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_ROW_POOLS = [
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Dept A", "Dept B", "Dept C", "Dept D", "Dept E"],
    ["Region 1", "Region 2", "Region 3", "Region 4", "Region 5", "Region 6"],
]
_COL_POOLS = [
    ["9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM", "4PM", "5PM"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    ["Product X", "Product Y", "Product Z", "Product W"],
    ["Metric A", "Metric B", "Metric C", "Metric D", "Metric E"],
]
_CMAPS = ["YlOrRd", "Blues", "Greens", "RdYlGn", "viridis", "plasma", "coolwarm"]

class HeatmapAdvancedQA(StandaloneVisualEnv):
    ENV_NAME = "heatmap_advanced"

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Accept short cell labels: GT "Dept B, Product Z" should also match "B, Z".
        # Strip leading prefix words (e.g. "Dept ", "Product ", "Region ", "Item ").
        import re as _re
        def _strip_prefix(s: str) -> str:
            s = str(s).strip()
            parts = [p.strip() for p in s.split(",")]
            stripped = []
            for p in parts:
                # Take last whitespace-separated token if multi-word (e.g. "Dept B" → "B")
                toks = p.split()
                stripped.append(toks[-1] if toks else p)
            return ",".join(stripped).lower()
        return _strip_prefix(predicted) == _strip_prefix(ground_truth) or super()._check_answer(predicted, ground_truth)

    def _level_config(self, level: int) -> dict:
        """Return difficulty config for a given level (0-9)."""
        configs = {
            0: {'question_type': 'max_cell'},
            1: {'question_type': 'row_highest_avg'},
            2: {'question_type': 'row_sum'},
            3: {'question_type': 'cell_difference'},
            4: {'question_type': 'count_above_threshold'},
            5: {'question_type': 'col_lowest_avg'},
            6: {'question_type': 'row_with_max_variance'},
            7: {'question_type': 'row_with_max_variance'},
            8: {'question_type': 'count_local_maxima'},
            9: {'question_type': 'count_local_maxima'},
        }
        return configs.get(max(0, min(9, level)), configs[0])

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:

        # ── Level routing (auto-generated) ──
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)
        rng = self._rng
        np_rng = np.random.RandomState(seed)
        qtype = parameter.get("question_type", "max_cell")

        rows = list(rng.choice(_ROW_POOLS))
        cols = list(rng.choice(_COL_POOLS))
        nr, nc = len(rows), len(cols)

        # Try up to 10 matrices so qtypes that reject ambiguous data (e.g.
        # max_cell with tied maxima) get a valid sample.
        question, answer, data = None, None, None
        for attempt in range(10):
            data = np_rng.randint(1, 100, size=(nr, nc)).tolist()
            question, answer = self._make_qa(rng, qtype, rows, cols, data)
            if question is not None:
                break
        if question is None:
            return None

        style = self._random_style()
        cmap = rng.choice(_CMAPS)
        fig, ax = plt.subplots(figsize=(max(5, nc * 0.9) * style["figsize_scale"],
                                        max(4, nr * 0.7) * style["figsize_scale"]))
        arr = np.array(data)
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
        ax.set_xticks(range(nc))
        ax.set_xticklabels(cols, fontsize=style["font_size_base"] - 1, rotation=45, ha="right")
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rows, fontsize=style["font_size_base"] - 1)
        # Annotate cells
        for i in range(nr):
            for j in range(nc):
                val = data[i][j]
                color = "white" if val > arr.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=style["font_size_base"] - 2, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Heatmap", fontsize=style["font_size_base"] + 2, fontfamily=style["font_family"])
        fig.patch.set_facecolor(style["bg_color"])
        fig.tight_layout()
        return question, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    def _make_qa(self, rng, qtype, rows, cols, data):
        nr, nc = len(rows), len(cols)
        if qtype == "max_cell":
            mx = -1
            max_positions = []
            for i in range(nr):
                for j in range(nc):
                    if data[i][j] > mx:
                        mx = data[i][j]
                        max_positions = [(i, j)]
                    elif data[i][j] == mx:
                        max_positions.append((i, j))
            # Reject when the global maximum is not unique — otherwise the
            # learner cannot pick a single cell from the image alone.
            if len(max_positions) >= 2:
                return None, None
            mi, mj = max_positions[0]
            return f"Which cell has the maximum value? Give as 'Row, Column'.", f"{rows[mi]}, {cols[mj]}"

        elif qtype == "row_highest_avg":
            avgs = [sum(data[i]) / nc for i in range(nr)]
            idx = avgs.index(max(avgs))
            return "Which row has the highest average value?", rows[idx]

        elif qtype == "cell_difference":
            r1, r2 = rng.sample(range(nr), 2)
            c1, c2 = rng.sample(range(nc), 2)
            diff = abs(data[r1][c1] - data[r2][c2])
            return (f"What is the absolute difference between the cell at "
                    f"({rows[r1]}, {cols[c1]}) and ({rows[r2]}, {cols[c2]})?"), str(diff)

        elif qtype == "count_above_threshold":
            thresh = rng.choice([30, 40, 50, 60, 70])
            cnt = sum(1 for i in range(nr) for j in range(nc) if data[i][j] > thresh)
            return f"How many cells have a value greater than {thresh}?", str(cnt)

        elif qtype == "col_lowest_avg":
            avgs = [sum(data[i][j] for i in range(nr)) / nr for j in range(nc)]
            idx = avgs.index(min(avgs))
            return "Which column has the lowest average value?", cols[idx]

        elif qtype == "row_sum":
            ri = rng.randint(0, nr - 1)
            return f"What is the sum of all values in the '{rows[ri]}' row?", str(sum(data[ri]))

        elif qtype == "row_with_max_variance":
            # Compute variance for each row, find which row has the highest
            variances = []
            for i in range(nr):
                row_mean = sum(data[i]) / nc
                var = sum((data[i][j] - row_mean) ** 2 for j in range(nc)) / nc
                variances.append(round(var, 2))
            max_var_idx = variances.index(max(variances))
            return (f"Compute the variance of each row's values "
                    f"(variance = mean of squared deviations from row mean). "
                    f"Which row has the highest variance?"), rows[max_var_idx]

        elif qtype == "count_local_maxima":
            # Count cells that are strictly greater than all 4 neighbors (up/down/left/right)
            count = 0
            for i in range(nr):
                for j in range(nc):
                    val = data[i][j]
                    neighbors = []
                    if i > 0: neighbors.append(data[i-1][j])
                    if i < nr - 1: neighbors.append(data[i+1][j])
                    if j > 0: neighbors.append(data[i][j-1])
                    if j < nc - 1: neighbors.append(data[i][j+1])
                    if neighbors and val > max(neighbors):
                        count += 1
            return (f"A local maximum is a cell whose value is strictly greater than "
                    f"all of its adjacent cells (up, down, left, right — not diagonal). "
                    f"Edge/corner cells have fewer neighbors. "
                    f"How many local maxima are in this heatmap?"), str(count)

        else:
            mx, mi, mj = -1, 0, 0
            for i in range(nr):
                for j in range(nc):
                    if data[i][j] > mx:
                        mx, mi, mj = data[i][j], i, j
            return f"What is the maximum value in the heatmap?", str(mx)
