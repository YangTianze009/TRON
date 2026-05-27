"""
Multi-Chart QA environment — redesign 2026-04-16.

DIVERSITY AXES:
  1. Chart type pair pool: bar_bar, bar_line, line_line, bar_pie,
     line_area, bar_scatter, pie_bar, bar_bar_stacked.
  2. Scenario pool: 8 scenarios (Revenue/Expenses, Visits/Conversion, etc).
  3. X-axis label pool: quarters, months, products, regions, years,
     weekdays, departments.
  4. Question templates: 6 phrasings per qtype.
  5. Color palettes shuffled per seed.
  6. Value ranges vary with level.

DIFFICULTY:
  L0: "which chart has higher total?" — ONE number comparison, categorical answer.
  L1: "which category has highest in chart A" — single-chart lookup.
  L2: compare totals across both charts (diff).
  L3: combined max (identify argmax of A+B).
  L4: correlation peaks.
  L5: ratio between charts at one category.
  L6: find the category where A - B is largest.
  L7: multi-hop: "exclude category X, then find biggest in A".
  L8: compute weighted average of A using values of B as weights.
  L9: find category where ratio A/B exceeds threshold AND A is > median.

VALUES are ONLY on the image. Questions refer to "as shown" or categories.

2026-05-03 extension (X32 / reference composite): added optional 3+ chart
type panel mode. When `n_charts=3` (or `parameter['n_charts']=3`), the
render shows three side-by-side panels of DIFFERENT chart types
(bar / line / scatter — or bar / pie / area), and questions reference
the third panel by position word (e.g. "in the right panel", "in the
middle subplot"). New question types `which_panel_higher_total` and
`panel_argmax` mirror reference IDX 781 / IDX 935 / IDX 947 phrasings.
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

_PRODUCTS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
             "Theta", "Lambda", "Sigma", "Omega"]
_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_REGIONS = ["North", "South", "East", "West", "Central"]
_YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
_DEPARTMENTS = ["Sales", "R&D", "Ops", "HR", "IT", "Finance"]

_SCENARIOS = [
    ("Revenue ($K)", "Expenses ($K)"),
    ("Units Sold", "Profit ($K)"),
    ("Website Visits (K)", "Conversion (%)"),
    ("Production (units)", "Defects"),
    ("Orders", "Returns"),
    ("Downloads (K)", "Bug Reports"),
    ("Trainees", "Pass Rate (%)"),
    ("Calls Received", "Avg Duration (s)"),
]

_TITLE_PREFIXES = ["Dashboard", "Comparison", "Report", "Summary",
                   "Overview", "Metrics", "Analytics", "Snapshot"]

class MultiChartQA(StandaloneVisualEnv):
    ENV_NAME = "multi_chart"

    QUESTION_TYPES = [
        "which_chart_higher_total",  # L0
        "chart_a_argmax",            # L1
        "compare_totals",            # L2
        "combined_max",              # L3
        "correlation",               # L4
        "ratio_between_charts",      # L5
        "diff_argmax",               # L6
        "exclude_and_argmax",        # L7
        "weighted_average",          # L8
        "threshold_filter",          # L9
        # X32 (3-panel composite) modes:
        "tri_panel_argmax_in_right",  # which category is the argmax in the RIGHT panel
        "tri_panel_higher_total",     # left/middle/right — which panel sums highest
    ]

    _QUESTION_TEMPLATES = {
        "which_chart_higher_total": [
            "Which chart has the larger total across all categories: the left chart or the right chart? Answer 'left' or 'right'.",
            "Sum all the values in the LEFT chart, and sum all values in the RIGHT chart. Which sum is larger? Answer 'left' or 'right'.",
            "Looking at both charts, which one has the higher grand total across categories? Answer 'left' or 'right'.",
            "Compare the total of the LEFT chart to the total of the RIGHT chart. Which is larger? Answer 'left' or 'right'.",
            "Which side (left/right) has the greater sum of all values shown? Answer 'left' or 'right'.",
            "Add up every value in each chart. Which chart sums to more? Answer 'left' or 'right'.",
            "Which chart wins on grand total? Answer 'left' or 'right'.",
            "Between the LEFT chart and the RIGHT chart, which has the higher overall total? Answer 'left' or 'right'.",
            "Total everything in the left chart; total everything in the right. Which is larger? Answer 'left' or 'right'.",
            "Is the sum of the left chart greater than the sum of the right chart? Answer 'left' if yes, 'right' if no.",
            "Which chart's values sum to the larger number? Answer 'left' or 'right'.",
            "Compare cumulative totals of the two charts. Which exceeds? Answer 'left' or 'right'.",
            "Sum the LEFT chart across all categories. Do the same for the RIGHT. Which total is bigger? Answer 'left' or 'right'.",
            "Determine which chart has a higher sum overall. Answer 'left' or 'right'.",
            "If you summed each chart's values, which would come out higher? Answer 'left' or 'right'.",
            "Which chart (left or right) totals more across all categories? Answer 'left' or 'right'.",
        ],
        "chart_a_argmax": [
            "In the LEFT chart, which category has the largest value? Answer with the category name as shown.",
            "Which category in the LEFT chart reaches the highest value? Answer with the category name.",
            "Read the LEFT chart only. Which x-axis category has the largest bar/point? Answer with the name.",
            "Identify the category with the greatest value in the LEFT chart. Answer with the category name.",
            "Find the argmax of the LEFT chart. Answer with the category label.",
            "Where does the LEFT chart peak? Answer with the x-axis category name.",
            "Which x-axis category is the maximum in the LEFT chart? Answer with the name.",
            "In the LEFT chart, which category shows the largest value? Answer with the category name exactly.",
            "The LEFT chart reaches its max at which category? Answer with the category name.",
            "Looking only at the LEFT chart, which category has the tallest bar (or highest point)? Answer with the name.",
            "Which category dominates the LEFT chart? Answer with the category label.",
            "Among the x-axis categories in the LEFT chart, which has the highest value? Answer with the name.",
            "What is the argmax of the left chart? Use the category name as the answer.",
            "Which category is highest in the LEFT chart? Answer with the name as printed on the axis.",
            "Give the category name where the LEFT chart attains its maximum.",
            "Which LEFT-chart category has the largest reading? Answer with the x-axis label.",
        ],
        "compare_totals": [
            "Looking at both charts, what is the absolute difference between the sum of values in the LEFT chart and the sum in the RIGHT chart? Answer with an integer.",
            "Compute total(left chart) minus total(right chart) across all categories. Answer with the absolute difference as an integer.",
            "What is |total_left - total_right| using all categories shown? Answer with an integer.",
            "Compute the absolute difference between the two chart totals. Report an integer.",
            "Sum the LEFT chart and sum the RIGHT chart. Report |left_total - right_total| as an integer.",
            "Return |sum(left) - sum(right)| as an integer.",
            "What is the magnitude of the difference between the two chart totals? Integer answer.",
            "Compute the gap (absolute) between left-total and right-total. Integer.",
            "How big is the difference between the two grand totals? Answer with the absolute integer difference.",
            "Find |sum(LEFT) - sum(RIGHT)| across all categories. Report as an integer.",
            "Report the absolute-value difference of the two chart totals (integer).",
            "What integer |A_total - B_total| do the charts have?",
            "Compute absolute(total_left - total_right) using all categories. Integer.",
            "Take the totals of the left and right charts; what is the absolute difference? Integer.",
            "Answer with the integer |left_sum - right_sum| across all categories.",
            "Return |sum_L - sum_R| as an integer, using all visible categories.",
        ],
        "combined_max": [
            "For which category is the SUM of the value in the left chart and the value in the right chart the largest? Answer with the category name.",
            "Add the left-chart value and the right-chart value for each category. Which category has the largest sum? Answer with the category name.",
            "Which x-axis category has the highest combined (left + right) value? Answer with the name.",
            "Compute left+right for every category; return the category with the largest combined total.",
            "Which category maximizes (LEFT value + RIGHT value)? Answer with the category name.",
            "Identify the category whose combined (A+B) value is greatest. Answer with its name.",
            "The sum of left + right peaks at which x-axis category? Answer with the name.",
            "Sum each category's two values. The argmax category is the answer (use its name).",
            "For every category, compute A_i + B_i; report the argmax's category label.",
            "Which category's combined total (left chart + right chart) is highest? Answer with the name.",
            "For which x-axis label is (left + right) largest? Return the label.",
            "Find the category with the maximum of (LEFT value + RIGHT value). Answer with its name.",
            "Compute the per-category sum across the two charts; return the argmax category.",
            "Which category adds up to the biggest (left + right) number? Answer with the name.",
            "Summing the two charts per category, which category totals the most? Answer with the name.",
            "Which category has the highest A+B combined value? Answer with the category name.",
        ],
        "correlation": [
            "Do the two charts both reach their maximum at the same category? Answer yes or no.",
            "Do the left chart and the right chart peak at the same x-axis category? Answer yes or no.",
            "Is the argmax of the left chart the same as the argmax of the right chart? Answer yes or no.",
            "Are the peaks of the two charts aligned (same category)? Answer yes or no.",
            "Does the left chart's maximum occur at the same category as the right chart's maximum? Answer yes or no.",
            "Compare argmax(left) with argmax(right). Are they the same category? Answer yes or no.",
            "Do the two charts share a common peak category? Answer yes or no.",
            "Is there a single category that is the argmax of BOTH charts? Answer yes or no.",
            "Left and right charts — do they peak at the same x-axis label? Answer yes or no.",
            "Do both charts' maxima coincide at one category? Answer yes or no.",
            "Are the two argmax categories identical? Answer yes or no.",
            "Same peak category for both? Answer yes or no.",
            "Does left-peak = right-peak (same category)? Answer yes or no.",
            "Is the highest-value category the same in both charts? Answer yes or no.",
            "Compare the peaks of both charts — are they on the same category? Answer yes or no.",
            "Are both charts maximized at the same x-axis category? Answer yes or no.",
        ],
        "ratio_between_charts": [
            "For the category marked with an asterisk (*) on the x-axis, what is the ratio of the LEFT chart value to the RIGHT chart value? Round to 2 decimals.",
            "Take the starred (*) category. Divide its left-chart value by its right-chart value. Round to 2 decimals.",
            "The (*) x-axis label indicates the target category. Compute left_value / right_value for that category, rounded to 2 decimals.",
            "At the starred category, compute LEFT / RIGHT. Round to 2 dp.",
            "For the x-axis label with a (*), report left_value / right_value to 2 decimal places.",
            "Using the (*)-marked category, divide left chart value by right chart value (2 dp).",
            "The asterisk marks one category. Give LEFT/RIGHT at that category, 2 decimals.",
            "Compute the ratio LEFT_value / RIGHT_value for the starred x-axis category. 2 dp.",
            "At the (*) category, what is A/B (left-to-right ratio)? Two decimals.",
            "For the starred category, find left-chart value divided by right-chart value (2 dp).",
            "Report a_star / b_star where 'star' is the (*)-marked category, rounded to 2 dp.",
            "Which ratio? Left value over right value at the starred category (round to 2 decimals).",
            "Find LEFT / RIGHT at the category indicated by the asterisk (*). 2 dp.",
            "At the (*)-labeled category: left / right (rounded to 2 decimals).",
            "Compute the left-to-right ratio at the starred category. Round to 2 dp.",
            "Target the (*)-marked category; divide its left value by its right value. 2 dp.",
        ],
        "diff_argmax": [
            "For which category is (left chart value - right chart value) the LARGEST (most positive)? Answer with the category name.",
            "Compute left - right for each category. Which category gives the biggest positive value? Answer with the name.",
            "At which category does the left chart most exceed the right chart? Answer with the category name.",
            "Find the category with the largest (LEFT - RIGHT). Answer with the name.",
            "Where is LEFT - RIGHT maximized? Answer with the category label.",
            "Which category has the greatest positive A - B difference? Answer with the name.",
            "Identify the category where left exceeds right by the most. Answer with the name.",
            "Compute A_i - B_i for each category; return the argmax category's name.",
            "For each x-axis label, compute (left - right). Which label yields the largest positive value?",
            "Which category shows the biggest positive gap between left-chart and right-chart values? Answer with the name.",
            "Return the category label where (LEFT value - RIGHT value) is largest.",
            "The maximum of (A - B) across categories occurs at which label? Answer with the name.",
            "At which category does LEFT most dominate RIGHT? Answer with the category name.",
            "Find argmax over categories of (LEFT_i - RIGHT_i). Return the category name.",
            "Which category has the greatest surplus of left over right? Answer with the name.",
            "Compute per-category diff (LEFT - RIGHT). Return the category with the highest value.",
        ],
        "exclude_and_argmax": [
            "Ignoring the category marked with (*) on the x-axis, which category has the highest value in the LEFT chart? Answer with the name.",
            "Exclude the (*) starred category. Among the remaining categories in the LEFT chart, which has the largest value? Answer with the category name.",
            "Skip the (*) column. In the LEFT chart, which other category peaks? Answer with the name.",
            "Drop the (*)-marked category. What is the argmax of the LEFT chart among the remaining? Answer with the name.",
            "Excluding the starred category, which LEFT-chart category is highest? Answer with the name.",
            "Remove the (*) column; report the argmax of the LEFT chart over the rest.",
            "Leaving out the (*) category, which LEFT-chart category has the biggest value? Answer with the name.",
            "Skip the starred category and find the max LEFT category. Answer with the category name.",
            "After removing the asterisked category, what is the LEFT chart's argmax? Answer with the name.",
            "Without the (*) category, which category in the LEFT chart has the largest value? Name it.",
            "Argmax(LEFT) excluding the starred category — which category? Answer with the name.",
            "Ignore the (*)-marked x-axis label; which remaining category tops the LEFT chart?",
            "Ignoring the starred category, find the LEFT-chart's highest category. Answer with the name.",
            "Exclude starred; which category wins in the LEFT chart? Answer with the category name.",
            "Drop the asterisked category. Which remaining LEFT-chart category is largest? Answer with the name.",
            "After excluding (*) from consideration, what is the LEFT-chart argmax category?",
        ],
        "weighted_average": [
            "Compute the weighted average of the LEFT chart values using the RIGHT chart values as weights (sum(a_i * b_i) / sum(b_i)). Round to the nearest integer.",
            "Treat the RIGHT chart values as weights. Compute the weighted average of the LEFT chart values. Round to the nearest integer.",
            "For each category, weight the left-chart value by the right-chart value. Output the weighted average, rounded to integer.",
            "Compute sum(a_i * b_i) / sum(b_i) using LEFT as a_i and RIGHT as b_i. Round to the nearest integer.",
            "Right chart provides weights. Left chart provides values. Report the weighted mean (integer).",
            "Take a weighted mean of LEFT using RIGHT as weights. Report integer.",
            "Compute weighted average of A weighted by B. Round to nearest integer.",
            "Weighted mean = sum(left_i * right_i) / sum(right_i). Round to integer.",
            "Using right-chart values as weights, compute the weighted mean of left-chart values (integer).",
            "Weighted average of LEFT (weights = RIGHT). Report nearest-integer value.",
            "Calculate sum(a*b)/sum(b) with a=left, b=right. Round to integer.",
            "Weight each LEFT value by the corresponding RIGHT value and return the weighted mean, rounded to the nearest integer.",
            "Compute the RIGHT-weighted average of the LEFT chart. Nearest integer.",
            "The weighted mean of the LEFT chart using the RIGHT chart as weights? Rounded to integer.",
            "Given LEFT as values and RIGHT as weights, return the weighted mean (integer).",
            "Report the RIGHT-weighted mean of LEFT, rounded to the nearest integer.",
        ],
        "threshold_filter": [
            "Among categories whose RIGHT-chart value is strictly greater than the RIGHT-chart median, which has the largest LEFT-chart value? Answer with the category name.",
            "Filter categories to those with right-chart value above the right-chart median. Among those, pick the one with the largest left-chart value. Answer with the category name.",
            "Consider only categories whose right-chart value exceeds the right-chart median. Which of these has the highest left-chart value? Answer with the name.",
            "Take the subset of categories where RIGHT > median(RIGHT). Among this subset, which has the highest LEFT value? Answer with the name.",
            "Restrict to categories with RIGHT-chart value greater than the RIGHT-chart median. Report the argmax of LEFT over this subset.",
            "Keep only categories whose RIGHT value is strictly above median(RIGHT). Which of these has the largest LEFT value?",
            "Among categories with above-median RIGHT value, pick the one with the largest LEFT value. Answer with the name.",
            "After filtering for RIGHT > median(RIGHT), find the LEFT argmax. Answer with the category name.",
            "Select categories where the right-chart value exceeds the median of the right chart. Which has the max left value?",
            "Filter by RIGHT > median(RIGHT); within those, find argmax of LEFT. Answer with the name.",
            "Consider only categories with RIGHT above its own median. Within those, return the LEFT argmax.",
            "Among categories with above-median RIGHT-chart reading, which has the highest LEFT-chart reading? Answer with the name.",
            "Identify categories whose RIGHT value > median(RIGHT). In that subset, return the LEFT-chart argmax.",
            "Filter on RIGHT value > median(RIGHT); in the filtered subset, return the LEFT argmax category name.",
            "Consider only above-median RIGHT categories. Which has the biggest LEFT value? Answer with the name.",
            "Among the RIGHT-above-median subset, find the category with the largest LEFT value. Answer with its name.",
        ],
        # X32 (reference composite multi-type panels) phrasings.
        "tri_panel_argmax_in_right": [
            "The figure shows three panels (left, middle, right). In the RIGHT panel, which category has the largest value? Answer with the category name.",
            "Three panels are shown side by side. Look only at the RIGHT-most panel. Which x-axis category has the highest value? Answer with the category name.",
            "Among the three subplots, focus on the RIGHT one. Which category in that panel has the largest value? Answer with the category name.",
            "The figure has 3 panels (left, middle, right). Identify the argmax category in the RIGHT panel. Answer with the category name.",
            "In the rightmost panel of the figure, which category reaches the largest value? Answer with the name as printed on the axis.",
        ],
        "tri_panel_higher_total": [
            "The figure shows three panels (left, middle, right). Sum the values in each panel. Which panel has the highest total? Answer 'left', 'middle', or 'right'.",
            "Among the three panels in the figure, which one has the largest grand total? Answer 'left', 'middle', or 'right'.",
            "Sum the values across all categories in each of the three panels. Report which panel sums to the most. Answer 'left', 'middle', or 'right'.",
            "Compare the totals of the three panels (left, middle, right). Which panel's grand total is the largest? Answer with the position word.",
            "The figure shows 3 subplots. Which subplot (left, middle, or right) has the highest sum across categories?",
        ],
    }

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        qtypes = [
            "which_chart_higher_total",   # L0
            "chart_a_argmax",             # L1
            "compare_totals",             # L2
            "combined_max",               # L3
            "correlation",                # L4
            "ratio_between_charts",       # L5
            "diff_argmax",                # L6
            "exclude_and_argmax",         # L7
            "weighted_average",           # L8
            "threshold_filter",           # L9
        ]
        return {"question_type": qtypes[level]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = self._rng.choice(self.QUESTION_TYPES)

        # sub-rng
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 4441)
        for _ in range(30):
            result = self._try_generate(qtype, level, sub_rng)
            if result is not None:
                return result
        return None

    def _try_generate(self, qtype: str, level: int,
                      sub_rng: random.Random) -> Optional[Tuple[str, str, Image.Image]]:
        rng = sub_rng
        # X32 (3-panel composite) routing: dispatch before 2-panel logic.
        if qtype in ("tri_panel_argmax_in_right", "tri_panel_higher_total"):
            return self._try_generate_tri_panel(qtype, level, rng)
        # Layout pool expands with level.
        if level <= 2:
            layouts = ["bar_bar", "bar_line"]
        elif level <= 5:
            layouts = ["bar_bar", "bar_line", "line_line", "bar_pie"]
        else:
            layouts = ["bar_bar", "bar_line", "line_line", "bar_pie",
                       "line_area", "bar_scatter"]
        layout = rng.choice(layouts)

        # X-axis labels.
        x_types = ["quarters", "months", "products", "regions",
                   "years", "weekdays", "departments"]
        # Smaller N at L0.
        x_type = rng.choice(x_types)
        if x_type == "quarters":
            labels = list(_QUARTERS)
        elif x_type == "months":
            num = rng.randint(4, 6) if level <= 3 else rng.randint(5, 8)
            labels = _MONTHS[:num]
        elif x_type == "products":
            num = rng.randint(3, 5) if level <= 3 else rng.randint(4, 6)
            labels = list(_PRODUCTS[:num])
            rng.shuffle(labels)
        elif x_type == "regions":
            labels = list(_REGIONS[:4])
        elif x_type == "years":
            num = rng.randint(3, 5)
            start = rng.randint(0, 2)
            labels = _YEARS[start:start + num]
        elif x_type == "weekdays":
            labels = list(_WEEKDAYS)
        else:  # departments
            num = rng.randint(4, 6)
            labels = list(_DEPARTMENTS[:num])
            rng.shuffle(labels)

        n = len(labels)
        scenario = rng.choice(_SCENARIOS)
        title_a, title_b = scenario

        # Difficulty-dependent value ranges.
        if level <= 1:
            vmin, vmax = 10, 100
        elif level <= 5:
            vmin, vmax = 10, 200
        else:
            vmin, vmax = 20, 400

        values_a = [rng.randint(vmin, vmax) for _ in range(n)]
        values_b = [rng.randint(vmin, vmax) for _ in range(n)]

        # Enforce interesting answers.
        star_idx = None  # used by ratio/exclude qtypes
        if qtype == "which_chart_higher_total":
            # Ensure not a tie.
            if sum(values_a) == sum(values_b):
                values_a[0] += 5
            answer = "left" if sum(values_a) > sum(values_b) else "right"

        elif qtype == "chart_a_argmax":
            # Ensure unique max in A.
            max_a = max(values_a)
            if values_a.count(max_a) > 1:
                best = rng.randint(0, n - 1)
                values_a[best] = max_a + 7
            answer = labels[values_a.index(max(values_a))]

        elif qtype == "compare_totals":
            # Avoid zero-diff trivial case.
            if sum(values_a) == sum(values_b):
                values_a[0] += 5
            answer = str(abs(sum(values_a) - sum(values_b)))

        elif qtype == "combined_max":
            combined = [a + b for a, b in zip(values_a, values_b)]
            if combined.count(max(combined)) > 1:
                idx = rng.randint(0, n - 1)
                values_a[idx] += 10
                combined = [a + b for a, b in zip(values_a, values_b)]
            answer = labels[combined.index(max(combined))]

        elif qtype == "correlation":
            max_a_idx = values_a.index(max(values_a))
            max_b_idx = values_b.index(max(values_b))
            # Force 50/50 yes/no via coin flip.
            want_yes = rng.random() < 0.5
            if want_yes and max_a_idx != max_b_idx:
                values_b[max_a_idx] = max(values_b) + 5
            elif (not want_yes) and max_a_idx == max_b_idx:
                other = [i for i in range(n) if i != max_a_idx]
                j = rng.choice(other)
                values_b[j] = max(values_b) + 5
                values_b[max_a_idx] = rng.randint(vmin, vmax // 2)
            max_a_idx = values_a.index(max(values_a))
            max_b_idx = values_b.index(max(values_b))
            answer = "yes" if max_a_idx == max_b_idx else "no"

        elif qtype == "ratio_between_charts":
            star_idx = rng.randint(0, n - 1)
            if values_b[star_idx] == 0:
                values_b[star_idx] = rng.randint(5, 20)
            # Make it a "nice" ratio when possible.
            # (not enforced — just randomize)
            ratio = round(values_a[star_idx] / values_b[star_idx], 2)
            answer = str(ratio)

        elif qtype == "diff_argmax":
            diffs = [a - b for a, b in zip(values_a, values_b)]
            if diffs.count(max(diffs)) > 1:
                idx = rng.randint(0, n - 1)
                values_a[idx] += 15
            diffs = [a - b for a, b in zip(values_a, values_b)]
            answer = labels[diffs.index(max(diffs))]

        elif qtype == "exclude_and_argmax":
            # Force the starred-category to be the argmax of A, so exclusion
            # materially changes the answer.
            orig_max_idx = values_a.index(max(values_a))
            star_idx = orig_max_idx
            remaining_vals = [v if i != star_idx else -1 for i, v in enumerate(values_a)]
            if max(remaining_vals) < 0:
                return None
            # ensure unique argmax among remaining
            sec_max = max(remaining_vals)
            if remaining_vals.count(sec_max) > 1:
                idx = [i for i, v in enumerate(remaining_vals) if v == sec_max][0]
                values_a[idx] += 7
                remaining_vals = [v if i != star_idx else -1 for i, v in enumerate(values_a)]
            answer = labels[remaining_vals.index(max(remaining_vals))]

        elif qtype == "weighted_average":
            denom = sum(values_b)
            if denom == 0:
                values_b = [v + 1 for v in values_b]
                denom = sum(values_b)
            num = sum(a * b for a, b in zip(values_a, values_b))
            answer = str(round(num / denom))

        elif qtype == "threshold_filter":
            # BUGFIX 2026-04-24: use standard median (avg of two mid values for
            # even n). Previous code used sorted[n//2] which is the upper-middle
            # value — disagrees with standard math/stat convention and produced
            # wrong GT (~half of even-n samples) for models applying textbook
            # definitions. "strictly greater than" is unaffected by the change.
            svals = sorted(values_b)
            if n % 2 == 0:
                med = (svals[n // 2 - 1] + svals[n // 2]) / 2
            else:
                med = svals[n // 2]
            above = [i for i, v in enumerate(values_b) if v > med]
            if len(above) < 2:
                # promote a few values_b
                for _ in range(3):
                    values_b[rng.randint(0, n - 1)] = max(values_b) + 10
                svals = sorted(values_b)
                if n % 2 == 0:
                    med = (svals[n // 2 - 1] + svals[n // 2]) / 2
                else:
                    med = svals[n // 2]
                above = [i for i, v in enumerate(values_b) if v > med]
                if len(above) < 1:
                    return None
            sub_values = [values_a[i] for i in above]
            if sub_values.count(max(sub_values)) > 1:
                # bump one
                idx = above[0]
                values_a[idx] = max(sub_values) + 15
                sub_values = [values_a[i] for i in above]
            best_i = above[sub_values.index(max(sub_values))]
            answer = labels[best_i]

        else:
            return None

        # Mark the star on x-axis for ratio / exclude questions.
        display_labels = list(labels)
        if star_idx is not None:
            display_labels[star_idx] = f"{labels[star_idx]}*"

        img = self._draw_multi_chart(layout, display_labels, values_a, values_b,
                                     title_a, title_b, rng, level)
        sidx = (self.seed or 0) % 16
        templates = self._QUESTION_TEMPLATES[qtype]
        q = templates[sidx % len(templates)]
        return q, answer, img

    # ------------------------------------------------------------------ #
    # X32 3-panel composite generator (reference style)
    # ------------------------------------------------------------------ #
    def _try_generate_tri_panel(self, qtype: str, level: int, rng: random.Random):
        """Render 3 panels of DIFFERENT chart types and ask a panel-aware question."""
        # X-axis labels (small-N for legibility in 3 panels)
        x_pool = [list(_QUARTERS), list(_REGIONS[:4])]
        labels = list(rng.choice(x_pool))
        if level >= 5:
            # add a 5th category for larger panels at higher levels
            extra = rng.choice([_PRODUCTS[0], _DEPARTMENTS[0]])
            if extra not in labels:
                labels.append(extra)
        n = len(labels)

        # Pick 3 chart types from a curated set (each panel a different type).
        chart_types = rng.sample(["bar", "line", "scatter", "area", "pie"], 3)

        # Three independent value series.
        if level <= 3:
            vmin, vmax = 10, 100
        elif level <= 6:
            vmin, vmax = 10, 200
        else:
            vmin, vmax = 20, 400
        v_left = [rng.randint(vmin, vmax) for _ in range(n)]
        v_mid = [rng.randint(vmin, vmax) for _ in range(n)]
        v_right = [rng.randint(vmin, vmax) for _ in range(n)]

        # Compute answer per qtype, with tie-break safeguards.
        if qtype == "tri_panel_argmax_in_right":
            mx = max(v_right)
            if v_right.count(mx) > 1:
                idx = rng.randint(0, n - 1)
                v_right[idx] = mx + 11
            answer = labels[v_right.index(max(v_right))]
        elif qtype == "tri_panel_higher_total":
            totals = [sum(v_left), sum(v_mid), sum(v_right)]
            # Force a unique max by bumping if a tie.
            tries = 0
            while totals.count(max(totals)) > 1 and tries < 5:
                bump_panel = rng.randint(0, 2)
                if bump_panel == 0:
                    v_left[0] += 30
                elif bump_panel == 1:
                    v_mid[0] += 30
                else:
                    v_right[0] += 30
                totals = [sum(v_left), sum(v_mid), sum(v_right)]
                tries += 1
            if totals.count(max(totals)) > 1:
                return None
            pos_words = ["left", "middle", "right"]
            answer = pos_words[totals.index(max(totals))]
        else:
            return None

        # Render 3-panel image.
        scenario = rng.choice(_SCENARIOS)
        title_left, title_mid = scenario
        # Pick a third title from the second-element pool of another scenario.
        other = rng.choice([s for s in _SCENARIOS if s != scenario])
        title_right = other[0]

        img = self._draw_tri_panel(chart_types, labels, v_left, v_mid, v_right,
                                   title_left, title_mid, title_right, rng)
        sidx = (self.seed or 0) % 5
        templates = self._QUESTION_TEMPLATES[qtype]
        q = templates[sidx % len(templates)]
        return q, answer, img

    def _draw_tri_panel(self, chart_types, labels, v_left, v_mid, v_right,
                        t_left, t_mid, t_right, rng):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, axes = plt.subplots(1, 3, figsize=(15 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)
        fs = max(10, style["font_size_base"])
        n = len(labels)
        x = np.arange(n)

        for ax, ctype, vals, title in zip(axes,
                                          chart_types,
                                          [v_left, v_mid, v_right],
                                          [t_left, t_mid, t_right]):
            cols = [palette[i % len(palette)] for i in range(n)]
            if ctype == "bar":
                ax.bar(x, vals, color=cols, edgecolor='black', lw=1)
            elif ctype == "line":
                ax.plot(x, vals, 'o-', color=palette[0], linewidth=2.0,
                        markersize=8, markerfacecolor=palette[1 % len(palette)])
            elif ctype == "scatter":
                sizes = [max(40, v * 2) for v in vals]
                ax.scatter(x, vals, s=sizes, c=cols, edgecolor='black', linewidth=1.0)
            elif ctype == "area":
                ax.fill_between(x, 0, vals, color=palette[2 % len(palette)],
                                alpha=0.4)
                ax.plot(x, vals, 'o-', color=palette[2 % len(palette)],
                        linewidth=2.0, markersize=6)
            elif ctype == "pie":
                ax.pie(vals, labels=labels, colors=cols, autopct="%1.0f",
                       startangle=rng.randint(0, 359),
                       textprops={'fontsize': fs - 2})

            ax.set_facecolor(style["bg_color"])
            ax.set_title(title, fontsize=fs + 2, fontweight='bold', pad=10)
            if ctype != "pie":
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=fs)
                ax.set_ylabel("Value", fontsize=fs)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                # Show value labels on image (questions reference values).
                for i, v in enumerate(vals):
                    ax.text(i, v + max(vals) * 0.02, str(v),
                            ha='center', va='bottom', fontsize=fs - 1,
                            fontweight='bold')

        suptitle = f"{rng.choice(_TITLE_PREFIXES)}: 3-Panel View"
        fig.suptitle(suptitle, fontsize=fs + 3, fontweight='bold', y=1.02)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    def _draw_multi_chart(self, layout, labels, values_a, values_b,
                          title_a, title_b, rng, level):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)
        fs = max(11, style["font_size_base"])
        lw = style["line_width"]

        n = len(labels)
        x = np.arange(n)
        colors_a = [palette[i % len(palette)] for i in range(n)]
        colors_b = [palette[(i + len(palette) // 2) % len(palette)]
                    for i in range(n)]

        def render_bar(ax, vals, cols):
            ax.bar(x, vals, color=cols, edgecolor='black', lw=1)

        def render_line(ax, vals, c1, c2):
            ax.plot(x, vals, 'o-', color=c1, linewidth=lw,
                    markersize=8, markerfacecolor=c2)

        def render_area(ax, vals, c1):
            ax.fill_between(x, 0, vals, color=c1, alpha=0.35)
            ax.plot(x, vals, 'o-', color=c1, linewidth=lw, markersize=6)

        def render_pie(ax, vals, cols, lbls):
            ax.pie(vals, labels=lbls, colors=cols, autopct="%1.0f",
                   startangle=rng.randint(0, 359),
                   textprops={'fontsize': fs - 2})

        def render_scatter(ax, vals, cols):
            sizes = [max(40, v * 2) for v in vals]
            ax.scatter(x, vals, s=sizes, c=cols, edgecolor='black', linewidth=1.2)
            for i, v in enumerate(vals):
                ax.text(i, v + max(vals) * 0.02, str(v),
                        ha='center', va='bottom', fontsize=fs - 1,
                        fontweight='bold')

        # Draw chart A
        pie_a = False
        pie_b = False
        if layout == "bar_bar":
            render_bar(ax1, values_a, colors_a)
            render_bar(ax2, values_b, colors_b)
        elif layout == "bar_line":
            render_bar(ax1, values_a, colors_a)
            render_line(ax2, values_b, palette[0], palette[1 % len(palette)])
        elif layout == "line_line":
            render_line(ax1, values_a, palette[0], palette[1 % len(palette)])
            render_line(ax2, values_b, palette[2 % len(palette)],
                        palette[3 % len(palette)])
        elif layout == "bar_pie":
            render_bar(ax1, values_a, colors_a)
            render_pie(ax2, values_b, colors_b, labels)
            pie_b = True
        elif layout == "line_area":
            render_line(ax1, values_a, palette[0], palette[1 % len(palette)])
            render_area(ax2, values_b, palette[2 % len(palette)])
        elif layout == "bar_scatter":
            render_bar(ax1, values_a, colors_a)
            render_scatter(ax2, values_b, colors_b)

        for ax, title, values, is_pie in [(ax1, title_a, values_a, pie_a),
                                          (ax2, title_b, values_b, pie_b)]:
            if is_pie:
                ax.set_title(title, fontsize=fs + 2, fontweight='bold', pad=10)
                continue
            ax.set_facecolor(style["bg_color"])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=fs)
            ax.set_title(title, fontsize=fs + 2, fontweight='bold', pad=10)
            ax.set_ylabel("Value", fontsize=fs)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            # Label every bar/point with value (values on IMAGE only).
            for i, v in enumerate(values):
                ax.text(i, v + max(values) * 0.02, str(v),
                        ha='center', va='bottom',
                        fontsize=fs - 1, fontweight='bold')

        suptitle = f"{rng.choice(_TITLE_PREFIXES)}: {title_a} vs {title_b}"
        fig.suptitle(suptitle, fontsize=fs + 3, fontweight='bold', y=1.02)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = MultiChartQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer}")
