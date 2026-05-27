"""
Frequency Table QA environment.

Renders a frequency distribution table (value ranges + frequencies)
and asks questions about mean (using midpoints), median class, mode,
cumulative frequency, percentages, and totals.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

class FrequencyTableQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "frequency_table"

    QUESTION_TYPES = [
        "total_frequency", "mean", "median_class", "mode_class",
        "percentage", "cumulative",
        # Hard types
        "variance_estimate",
        "relative_cumulative",
        "interquartile_class",
        "weighted_mean_two_tables",
        # Ultra-hard types
        "ultra_hard_std_deviation",
        "ultra_hard_percentile_value",
        "ultra_hard_combine_two_tables",
        # Harder computation types
        "weighted_mean",
        "interquartile_range",
    ]

    CONTEXT_POOLS = [
        {"title": "Test Scores", "unit": "Score"},
        {"title": "Ages of Participants", "unit": "Age"},
        {"title": "Heights of Plants (cm)", "unit": "Height (cm)"},
        {"title": "Daily Sales ($)", "unit": "Sales ($)"},
        {"title": "Weights of Packages (kg)", "unit": "Weight (kg)"},
        {"title": "Time Spent (minutes)", "unit": "Time (min)"},
    ]

    def _level_config(self, level: int) -> Dict:
        # Redesigned: L0 starts with mean + cumulative (old L3-4).
        # L9 requires percentile interpolation + combined tables.
        if level <= 1:
            return {"n_classes": (5, 6), "qtypes": ["mean", "cumulative"]}
        if level <= 3:
            return {"n_classes": (5, 7), "qtypes": ["variance_estimate", "relative_cumulative"]}
        if level <= 5:
            return {"n_classes": (5, 7), "qtypes": ["interquartile_class", "weighted_mean"]}
        if level <= 7:
            return {"n_classes": (5, 7), "qtypes": ["ultra_hard_std_deviation", "interquartile_range"]}
        return {"n_classes": (6, 8), "qtypes": ["ultra_hard_percentile_value",
                "ultra_hard_combine_two_tables", "ultra_hard_std_deviation",
                "interquartile_range"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 714)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        for _ in range(20):
            result = self._try_generate(qtype, cfg, sub_rng)
            if result is not None:
                return result
        return None

    def _try_generate(self, qtype: str, cfg=None, sub_rng=None) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        sr = sub_rng or rng
        context = sr.choice(self.CONTEXT_POOLS)

        # Generate class intervals
        lo_c, hi_c = (cfg or {}).get("n_classes", (4, 7))
        num_classes = sr.randint(lo_c, hi_c)
        class_width = rng.choice([5, 10, 20])
        start = rng.choice([0, 10, 20, 50, 100])

        intervals = []
        for i in range(num_classes):
            lo = start + i * class_width
            hi = lo + class_width - 1
            intervals.append((lo, hi))

        # Generate frequencies (ensure total is reasonable)
        freqs = [rng.randint(2, 30) for _ in range(num_classes)]
        # Make sure not all equal for mode_class
        if qtype == "mode_class":
            # Ensure a unique max
            max_idx = rng.randint(0, num_classes - 1)
            freqs[max_idx] = max(freqs) + rng.randint(3, 10)

        total_freq = sum(freqs)

        # Compute midpoints
        midpoints = [(lo + hi) / 2 for lo, hi in intervals]

        question, answer = self._build_qa(
            qtype, intervals, freqs, midpoints, total_freq, context
        )
        if question is None:
            return None

        img = self._render_table(intervals, freqs, context)
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(self._rng, question, answer, rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], img
        return question, answer, img

    def _build_qa(self, qtype, intervals, freqs, midpoints, total_freq, context):
        num_classes = len(intervals)

        if qtype == "total_frequency":
            q = "What is the total frequency (sum of all frequencies) in the table?"
            return q, str(total_freq)

        elif qtype == "mean":
            # Mean using midpoints: sum(f*m) / sum(f)
            fx_sum = sum(f * m for f, m in zip(freqs, midpoints))
            mean_val = fx_sum / total_freq
            q = ("Calculate the estimated mean using the midpoint of each class interval. "
                 "Round your answer to 2 decimal places.")
            return q, str(round(mean_val, 2))

        elif qtype == "median_class":
            # Median class: class containing the (total/2)th value
            median_pos = total_freq / 2
            cum = 0
            for i, f in enumerate(freqs):
                cum += f
                if cum >= median_pos:
                    lo, hi = intervals[i]
                    q = ("Which class interval contains the median? "
                         "Give your answer as the interval (e.g., '20-29').")
                    return q, f"{lo}-{hi}"
            return None, None

        elif qtype == "mode_class":
            max_freq = max(freqs)
            max_idx = freqs.index(max_freq)
            lo, hi = intervals[max_idx]
            q = ("Which class interval is the modal class (has the highest frequency)? "
                 "Give your answer as the interval (e.g., '20-29').")
            return q, f"{lo}-{hi}"

        elif qtype == "percentage":
            idx = self._rng.randint(0, num_classes - 1)
            lo, hi = intervals[idx]
            pct = round(freqs[idx] / total_freq * 100, 1)
            q = (f"What percentage of the total does the class interval "
                 f"{lo}-{hi} represent? Round to 1 decimal place.")
            return q, str(pct)

        elif qtype == "cumulative":
            idx = self._rng.randint(1, num_classes - 1)
            cum_freq = sum(freqs[:idx + 1])
            lo, hi = intervals[idx]
            q = (f"What is the cumulative frequency up to and including "
                 f"the class interval {lo}-{hi}?")
            return q, str(cum_freq)

        # ---- Hard question types ----

        elif qtype == "variance_estimate":
            # Estimated variance using midpoints:
            # var = sum(f*(m-mean)^2) / sum(f)
            fx_sum = sum(f * m for f, m in zip(freqs, midpoints))
            mean_val = fx_sum / total_freq
            var = sum(f * (m - mean_val) ** 2 for f, m in zip(freqs, midpoints)) / total_freq
            q = ("Estimate the variance of the distribution using midpoints. "
                 "First compute the mean using midpoints, then compute "
                 "sum(f * (midpoint - mean)^2) / total_frequency. "
                 "Round to 2 decimal places.")
            return q, str(round(var, 2))

        elif qtype == "relative_cumulative":
            # Relative cumulative frequency (percentage) at a given class
            idx = self._rng.randint(1, num_classes - 1)
            cum_freq = sum(freqs[:idx + 1])
            rel_cum = round(cum_freq / total_freq * 100, 1)
            lo, hi = intervals[idx]
            q = (f"What is the relative cumulative frequency (as a percentage) "
                 f"up to and including the class interval {lo}-{hi}? "
                 f"Round to 1 decimal place.")
            return q, str(rel_cum)

        elif qtype == "interquartile_class":
            # Find the class containing Q1 (25th percentile) and Q3 (75th percentile)
            # Report Q3 class
            q3_pos = total_freq * 0.75
            cum = 0
            for i, f in enumerate(freqs):
                cum += f
                if cum >= q3_pos:
                    lo, hi = intervals[i]
                    q = ("Which class interval contains the third quartile (Q3)? "
                         "Q3 is at the 75th percentile position. "
                         "Give your answer as the interval (e.g., '20-29').")
                    return q, f"{lo}-{hi}"
            return None, None

        elif qtype == "weighted_mean_two_tables":
            # Multi-step: given overall mean and this table's mean + freq,
            # find what the other group's mean must be
            fx_sum = sum(f * m for f, m in zip(freqs, midpoints))
            this_mean = fx_sum / total_freq
            # Generate a second group
            other_freq = self._rng.randint(total_freq // 2, total_freq * 2)
            other_mean = round(self._rng.uniform(this_mean * 0.7, this_mean * 1.3), 2)
            combined_mean = round(
                (this_mean * total_freq + other_mean * other_freq) /
                (total_freq + other_freq), 2
            )
            # Ask: given combined mean and this table, what is the other group's mean?
            q = (f"This frequency table represents Group A with {total_freq} observations. "
                 f"Group B has {other_freq} observations. "
                 f"The combined mean of both groups is {combined_mean}. "
                 f"Using midpoints to estimate Group A's mean, "
                 f"find Group B's mean. Round to 2 decimal places.")
            return q, str(other_mean)

        # ---- Ultra-hard question types ----

        elif qtype == "ultra_hard_std_deviation":
            # Standard deviation using midpoints:
            # 1. Compute mean = sum(f*m)/N
            # 2. Compute variance = sum(f*(m-mean)^2)/N
            # 3. std = sqrt(variance)
            fx_sum = sum(f * m for f, m in zip(freqs, midpoints))
            mean_val = fx_sum / total_freq
            var = sum(f * (m - mean_val) ** 2 for f, m in zip(freqs, midpoints)) / total_freq
            std_val = round(math.sqrt(var), 2)
            q = ("Estimate the standard deviation of the distribution using midpoints. "
                 "First compute the mean, then the variance "
                 "(sum of f*(midpoint-mean)^2 divided by total frequency), "
                 "then take the square root. "
                 "Round to 2 decimal places.")
            return q, str(std_val)

        elif qtype == "ultra_hard_percentile_value":
            # Interpolate to find the actual estimated value at a given percentile.
            # Use linear interpolation within the median class.
            # P-th percentile value = L + ((P*N/100 - CF) / f) * w
            # where L = lower boundary, CF = cumulative freq before class,
            # f = freq of class, w = class width
            p = self._rng.choice([25, 50, 75, 90])
            target_pos = p * total_freq / 100
            cum = 0
            for i, freq in enumerate(freqs):
                if cum + freq >= target_pos:
                    lo, hi = intervals[i]
                    class_width = hi - lo + 1
                    cf_before = cum
                    # Linear interpolation
                    pval = lo + ((target_pos - cf_before) / freq) * class_width
                    pval = round(pval, 2)
                    q = (f"Estimate the {p}th percentile value using linear "
                         f"interpolation within the appropriate class interval. "
                         f"Use the formula: L + ((P*N/100 - CF) / f) * w, where "
                         f"L is the lower boundary, CF is cumulative frequency before "
                         f"the class, f is the class frequency, and w is the class width. "
                         f"Round to 2 decimal places.")
                    return q, str(pval)
                cum += freq
            return None, None

        elif qtype == "weighted_mean":
            # Compute the weighted mean using frequency as weight for midpoints
            # Then also compute the simple average of midpoints and find the difference
            fx_sum = sum(f * m for f, m in zip(freqs, midpoints))
            weighted_mean = fx_sum / total_freq
            simple_mean = sum(midpoints) / len(midpoints)
            diff = round(abs(weighted_mean - simple_mean), 2)
            q = ("Compute the weighted mean using frequencies as weights "
                 "(sum of frequency*midpoint / total frequency). "
                 "Then compute the simple (unweighted) average of the midpoints. "
                 "What is the absolute difference between these two means? "
                 "Round to 2 decimal places.")
            return q, str(diff)

        elif qtype == "interquartile_range":
            # Estimate IQR using linear interpolation:
            # Q1 and Q3 positions, then interpolate within their classes
            for p, label in [(25, "Q1"), (75, "Q3")]:
                pass  # just structuring
            results = {}
            for p in [25, 75]:
                target_pos = p * total_freq / 100
                cum = 0
                for i, freq in enumerate(freqs):
                    if cum + freq >= target_pos:
                        lo, hi = intervals[i]
                        class_width = hi - lo + 1
                        cf_before = cum
                        pval = lo + ((target_pos - cf_before) / freq) * class_width
                        results[p] = pval
                        break
                    cum += freq
            if 25 not in results or 75 not in results:
                return None, None
            iqr = round(results[75] - results[25], 2)
            q = ("Estimate the interquartile range (IQR = Q3 - Q1) using "
                 "linear interpolation within each class interval. "
                 "Use the formula: L + ((P*N/100 - CF) / f) * w for each quartile, "
                 "where L is lower boundary, CF is cumulative frequency before the class, "
                 "f is class frequency, w is class width. "
                 "Round to 2 decimal places.")
            return q, str(iqr)

        elif qtype == "ultra_hard_combine_two_tables":
            # Given this table and a second table (described in text),
            # compute the combined mean of both tables.
            fx_sum = sum(f * m for f, m in zip(freqs, midpoints))
            mean_a = fx_sum / total_freq

            # Generate second table stats
            rng2 = self._rng
            num_classes_b = rng2.randint(3, 5)
            class_width_b = intervals[0][1] - intervals[0][0] + 1
            start_b = intervals[0][0] + rng2.choice([-10, 0, 10])
            freqs_b = [rng2.randint(3, 25) for _ in range(num_classes_b)]
            total_b = sum(freqs_b)
            midpoints_b = [start_b + i * class_width_b + class_width_b / 2
                           for i in range(num_classes_b)]
            fx_sum_b = sum(f * m for f, m in zip(freqs_b, midpoints_b))
            mean_b = fx_sum_b / total_b

            combined_mean = round(
                (fx_sum + fx_sum_b) / (total_freq + total_b), 2
            )

            # Describe table B in the question
            q = (f"Table A is shown in the image with {total_freq} observations. "
                 f"Table B (not shown) has {total_b} observations with an "
                 f"estimated mean of {round(mean_b, 2)} (using midpoints). "
                 f"Compute the combined mean of both tables together. "
                 f"First find Table A's mean using midpoints, then compute "
                 f"the weighted average of the two means. "
                 f"Round to 2 decimal places.")
            return q, str(combined_mean)

        return None, None

    def _render_table(self, intervals, freqs, context):
        vs = self._random_style()
        palette = vs["palette"]
        label_fs = vs["font_size_base"]
        title_fs = label_fs + 3
        font_family = vs["font_family"]

        num_classes = len(intervals)

        fig_height = max(3.5, 1.0 + num_classes * 0.6) * vs["figsize_scale"]
        fig_width = 7 * vs["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")
        fig.patch.set_facecolor(vs["bg_color"])

        col_labels = [context["unit"] + " Range", "Frequency"]
        cell_text = []
        for i in range(num_classes):
            lo, hi = intervals[i]
            cell_text.append([f"{lo} - {hi}", str(freqs[i])])

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(label_fs + 1)
        table.scale(1.0, 1.6)

        header_color = palette[0]

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#7f8c8d")
            cell.set_linewidth(1.2)
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color="white", fontweight="bold",
                                    fontsize=label_fs + 1, fontfamily=font_family)
            else:
                if row % 2 == 1:
                    cell.set_facecolor("#ecf0f1")
                else:
                    cell.set_facecolor("#fdfefe")
                cell.set_text_props(fontsize=label_fs + 1, fontfamily=font_family)

        ax.set_title(context["title"], fontsize=title_fs + 1, fontweight="bold",
                     pad=15, fontfamily=font_family)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vs["dpi"])
