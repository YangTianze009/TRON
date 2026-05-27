"""
Hexagon Size Comparison QA (Task C, PuzzleVQA shape_size_hexagon gap, ).

Show hexagons of different sizes with labels, ask comparison or ordering.

Difficulty axes:
  A) Number of hexagons (3 -> 8)
  B) Size difference (large gap -> small gap)
  C) Question type (largest -> ordering -> ratio)
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

class HexagonSizeComparisonQA(StandaloneVisualEnv):
    ENV_NAME = "hexagon_size_comparison"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Iter-3 hardening 2026-04-17: L9 now uses 12 hexagons and a
        # FILTERED multi-step computation (e.g. "median area of top-5 by
        # size after filtering by parity") — this forces tracking 12 items
        # AND multiple arithmetic stages AND area-squaring.
        if level <= 2:
            qtypes = ["largest"]
        elif level <= 5:
            qtypes = ["largest", "smallest", "ordering"]
        elif level <= 8:
            qtypes = ["ordering", "ratio", "count_larger_than"]
        else:
            # L9: compound filter+aggregate qtypes with 12 hexagons.
            qtypes = ["filtered_area_median_top5",
                      "area_stddev_top5",
                      "filtered_area_sum_parity",
                      "quartile_area_diff"]
        n_hex = 3 + level * 2 // 3  # original 3..8
        if level == 9:
            n_hex = 12  # bumped from 9 → 12 hexagons
        gap = max(0.04, 0.5 - level * 0.055)
        return {
            "n_hexagons": n_hex,
            "size_gap": gap,
            "qtypes": qtypes,
            # Labels shown on image at ALL levels so questions that reference
            # numeric sizes (ratio) are answerable from visual input alone.
            "show_size_labels": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_hexagons"]

        n = cfg["n_hexagons"]
        gap = cfg["size_gap"]
        qtype = rng.choice(cfg["qtypes"])

        # Generate hexagon sizes
        labels = [chr(ord("A") + i) for i in range(n)]
        base = 0.5
        sizes = []
        for i in range(n):
            s = base + rng.uniform(0, 2.0)
            sizes.append(round(s, 2))

        # Ensure distinct sizes with minimum gap
        sizes.sort()
        for i in range(1, len(sizes)):
            if sizes[i] - sizes[i - 1] < gap:
                sizes[i] = sizes[i - 1] + gap

        # Shuffle for display
        display_order = list(range(n))
        rng.shuffle(display_order)
        display_sizes = [sizes[display_order[i]] for i in range(n)]
        display_labels = [labels[i] for i in range(n)]

        # Build Q/A
        if qtype == "largest":
            max_idx = display_sizes.index(max(display_sizes))
            question = ("Several hexagons of different sizes are shown, each labeled. "
                       "Which hexagon is the LARGEST? Answer with its letter.")
            answer = display_labels[max_idx]

        elif qtype == "smallest":
            min_idx = display_sizes.index(min(display_sizes))
            question = ("Several hexagons of different sizes are shown, each labeled. "
                       "Which hexagon is the SMALLEST? Answer with its letter.")
            answer = display_labels[min_idx]

        elif qtype == "ordering":
            # Order from smallest to largest
            indexed = list(zip(display_sizes, display_labels))
            indexed.sort(key=lambda x: x[0])
            ordering = ",".join(lbl for _, lbl in indexed)
            question = ("Several hexagons are shown with labels. "
                       "Order them from SMALLEST to LARGEST. "
                       "Give your answer as comma-separated letters (e.g., 'C,A,B').")
            answer = ordering

        elif qtype == "ratio":
            i, j = rng.sample(range(n), 2)
            ratio = round(display_sizes[i] / display_sizes[j], 2)
            question = (f"What is the ratio of hexagon {display_labels[i]}'s size "
                       f"to hexagon {display_labels[j]}'s size? Round to 2 decimals.")
            answer = str(ratio)

        elif qtype == "count_larger_than":
            threshold_idx = rng.randint(0, n - 1)
            threshold = display_sizes[threshold_idx]
            count = sum(1 for s in display_sizes if s > threshold)
            question = (f"How many hexagons are LARGER than hexagon "
                       f"{display_labels[threshold_idx]}? Answer with an integer.")
            answer = str(count)

        elif qtype == "area_ratio":
            # Ratio of AREA (not size) = (size_i / size_j)^2. Requires the
            # student to square the ratio — a common error is answering the
            # size ratio itself.
            i, j = rng.sample(range(n), 2)
            area_ratio = round((display_sizes[i] / display_sizes[j]) ** 2, 2)
            question = (f"What is the ratio of the AREA of hexagon "
                       f"{display_labels[i]} to the AREA of hexagon "
                       f"{display_labels[j]}? (Hint: area scales with the "
                       f"SQUARE of the linear size.) Round to 2 decimals.")
            answer = str(area_ratio)

        elif qtype == "top3_sum":
            # Sum of the top-3 sizes, rounded to 2 decimals.
            top3 = sorted(display_sizes, reverse=True)[:3]
            answer = str(round(sum(top3), 2))
            question = ("Sum the sizes (r values) of the THREE LARGEST "
                       "hexagons. Report the total, rounded to 2 decimals.")

        elif qtype == "median_value":
            # Median r value across all hexagons, rounded to 2 decimals.
            sorted_sizes = sorted(display_sizes)
            m = len(sorted_sizes)
            if m % 2 == 1:
                med = sorted_sizes[m // 2]
            else:
                med = (sorted_sizes[m // 2 - 1] + sorted_sizes[m // 2]) / 2
            answer = str(round(med, 2))
            question = ("Consider all hexagons' r values shown. What is "
                       "their median (the middle value after sorting)? "
                       "Round to 2 decimals.")

        elif qtype == "filtered_area_median_top5":
            # Multi-step: (1) pick hexagons whose r rounded-to-2dp has EVEN
            # first decimal (after dp) (arbitrary parity filter), (2) compute
            # areas ~ r^2 * 3*sqrt(3)/2 for each, (3) take top 5 by area,
            # (4) report the MEDIAN of those top 5 areas, to 2 decimals.
            # Model has to filter 12 items, square each, sort, pick 5, median.
            import math as _m
            # parity filter: first decimal (tenths digit) is even (0,2,4,6,8)
            area_factor = 3 * _m.sqrt(3) / 2
            filtered = []
            for lbl, s in zip(display_labels, display_sizes):
                tenths = int(round(s * 10)) % 10
                if tenths % 2 == 0:
                    filtered.append((lbl, s, area_factor * s * s))
            if len(filtered) < 5:
                # Relax filter: keep all
                filtered = [(lbl, s, area_factor * s * s)
                            for lbl, s in zip(display_labels, display_sizes)]
            filtered.sort(key=lambda t: -t[2])
            top5 = filtered[:5]
            top5_areas = sorted(t[2] for t in top5)
            med_area = top5_areas[2]  # middle of 5
            answer = f"{round(med_area, 2)}"
            question = (
                "Use the r values labeled beneath each hexagon. "
                "STEP 1: keep only hexagons whose r value (rounded to 2 "
                "decimals) has an EVEN tenths digit (0, 2, 4, 6, or 8 in "
                "the first decimal place). "
                "STEP 2: for each kept hexagon compute its area = "
                "(3 * sqrt(3) / 2) * r^2. "
                "STEP 3: take the TOP 5 largest areas. "
                "STEP 4: report the MEDIAN of those 5 areas. "
                "Round the final answer to 2 decimals."
            )

        elif qtype == "area_stddev_top5":
            # sample stddev of top-5 areas (population stdev).
            import math as _m
            area_factor = 3 * _m.sqrt(3) / 2
            areas = sorted([area_factor * s * s for s in display_sizes],
                           reverse=True)[:5]
            mu = sum(areas) / 5.0
            var = sum((a - mu) ** 2 for a in areas) / 5.0
            sd = _m.sqrt(var)
            answer = f"{round(sd, 2)}"
            question = (
                "Compute the AREA of each hexagon (area = "
                "(3*sqrt(3)/2) * r^2). Take the 5 hexagons with the "
                "LARGEST areas. Compute the POPULATION standard "
                "deviation of those 5 areas (divide by 5, NOT by 4). "
                "Round to 2 decimals."
            )

        elif qtype == "filtered_area_sum_parity":
            # SUM of areas where r's units digit (integer part parity) is odd.
            # This is non-trivial with 12 hexagons.
            import math as _m
            area_factor = 3 * _m.sqrt(3) / 2
            total = 0.0
            question_parity = "odd"
            for s in display_sizes:
                units = int(s)
                if units % 2 == 1:  # odd integer part
                    total += area_factor * s * s
            if total == 0.0:
                # No odd units → fallback: use even
                for s in display_sizes:
                    units = int(s)
                    if units % 2 == 0:
                        total += area_factor * s * s
                question_parity = "even"
            answer = f"{round(total, 2)}"
            question = (
                f"For each hexagon compute area = (3*sqrt(3)/2) * r^2. "
                f"Keep only the hexagons whose integer part of r (i.e. "
                f"floor(r)) is {question_parity}. Sum those areas. "
                f"Round the final total to 2 decimals."
            )

        elif qtype == "quartile_area_diff":
            # Difference: area of the 3rd-largest hexagon minus area of the
            # 3rd-smallest hexagon (not trivial to order 12 items by eye).
            import math as _m
            area_factor = 3 * _m.sqrt(3) / 2
            areas = sorted([area_factor * s * s for s in display_sizes])
            # 3rd-smallest = index 2; 3rd-largest = index -3
            diff = areas[-3] - areas[2]
            answer = f"{round(diff, 2)}"
            question = (
                "Compute the AREA of each hexagon (area = "
                "(3*sqrt(3)/2) * r^2). Sort the 12 areas ascending. "
                "Report (3rd LARGEST area) − (3rd SMALLEST area). "
                "Round to 2 decimals."
            )

        else:
            return None

        img = self._render_hexagons(display_labels, display_sizes, cfg, rng)
        return question, answer, img

    def _render_hexagons(self, labels, sizes, cfg, rng):
        style = self._random_style()
        n = len(labels)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3 * style["figsize_scale"],
                                                       rows * 3 * style["figsize_scale"]))
        fig.patch.set_facecolor(style["bg_color"])
        if rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        # Compute axis limit so the largest hexagon fits without overflow
        # and a label below doesn't overlap the shape.
        max_size = max(sizes) if sizes else 2.5
        axis_lim = max_size + 1.1  # room below for 'r=...' label
        for i in range(len(axes)):
            ax = axes[i]
            ax.set_xlim(-axis_lim, axis_lim)
            ax.set_ylim(-axis_lim, axis_lim)
            ax.set_aspect("equal")
            ax.axis("off")

            if i < n:
                s = sizes[i]
                color = style["palette"][i % len(style["palette"])]
                pts = [(s * math.cos(math.radians(60 * k + 30)),
                        s * math.sin(math.radians(60 * k + 30))) for k in range(6)]
                hex_patch = plt.Polygon(pts, fc=color, ec="black",
                                       lw=1.5, alpha=0.8)
                ax.add_patch(hex_patch)
                ax.text(0, 0, labels[i], fontsize=14, ha="center", va="center",
                       fontweight="bold", color="white")

                if cfg.get("show_size_labels"):
                    # Place label below the hexagon's bottom vertex at y=-s
                    # with enough padding that it never overlaps.
                    ax.text(0, -(s + 0.55), f"r={s:.2f}", fontsize=9,
                           ha="center", va="center", color="#333")

        fig.suptitle("Hexagon Size Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskc"
    os.makedirs(out_dir, exist_ok=True)
    env = HexagonSizeComparisonQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[hexagon_size_comparison L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"hexagon_size_comparison_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[hexagon_size_comparison L{level} s{s}] A={env._answer}")
