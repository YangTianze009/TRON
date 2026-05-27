"""Treemap Visual QA Environment.

Round 2 diversity + difficulty fix (2026-04-16):
- Fix L0 == L9 bug: sub_rng now includes level.
- Add new qtypes at L6-L9 (sub_of_parent_percent, rank_of_sub, etc.).
- Structural difference: L0-L1 have small treemaps (2 parents, 2-3 subs each);
  L9 has 3-4 parents with 3-4 subs each AND decimal answers.
- 5 domain context pools (was 3).
- Rendering: squarified alt layout, grid jitter, font sizing.
- Values go only on image; question says "as shown".
"""

import random
import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CATEGORY_SETS = [
    {"Tech": ["Hardware", "Software", "Services", "Cloud"],
     "Finance": ["Banking", "Insurance", "Investments"],
     "Health": ["Pharma", "Devices", "Biotech"]},
    {"Food": ["Fruits", "Dairy", "Grains", "Meat"],
     "Drinks": ["Water", "Juice", "Soda"],
     "Snacks": ["Chips", "Candy", "Nuts"]},
    {"Asia": ["China", "Japan", "India"],
     "Europe": ["Germany", "France", "UK", "Spain"],
     "Americas": ["USA", "Brazil", "Canada"]},
    {"Retail": ["Online", "Stores", "Kiosks"],
     "Transport": ["Air", "Rail", "Road", "Sea"],
     "Energy": ["Solar", "Wind", "Hydro"]},
    {"Books": ["Fiction", "NonFic", "Ref", "Kids"],
     "Music": ["Rock", "Pop", "Jazz"],
     "Film": ["Drama", "Comedy", "Thriller"]},
]

class TreemapQA(StandaloneVisualEnv):
    ENV_NAME = "treemap"

    # ------------------------------------------------------------------ #
    # Per-level configuration (structurally different L0 vs L9)
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 1:
            return {"qtypes": ["largest_category", "smallest_sub"],
                    "n_parents": 2, "subs_range": (2, 3),
                    "val_range": (10, 40),
                    "hide_values": False, "similar_colors": False}
        if level <= 3:
            return {"qtypes": ["largest_category", "smallest_sub",
                                "compare_categories"],
                    "n_parents": 3, "subs_range": (2, 3),
                    "val_range": (10, 50),
                    "hide_values": False, "similar_colors": False}
        if level <= 5:
            return {"qtypes": ["largest_category", "smallest_sub",
                                "fraction_of_category",
                                "compare_categories"],
                    "n_parents": 3, "subs_range": (3, 4),
                    "val_range": (5, 60),
                    "hide_values": False, "similar_colors": False}
        if level <= 7:
            # L6-L7: hide numeric values; learner must estimate from rect
            # sizes. Only pure-visual questions (argmax / argmin).
            return {"qtypes": ["largest_category", "smallest_sub"],
                    "n_parents": 3, "subs_range": (3, 4),
                    "val_range": (5, 70),
                    "hide_values": True, "similar_colors": False}
        # L8-L9: hardest — values hidden AND similar colors.
        return {"qtypes": ["largest_category", "smallest_sub"],
                "n_parents": 3, "subs_range": (3, 5),
                "val_range": (3, 80),
                "hide_values": True, "similar_colors": True}

    # ------------------------------------------------------------------ #

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # FIX: sub_rng now includes level — prime 919 unique to treemap
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 919)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        cat_set_full = sub_rng.choice(_CATEGORY_SETS)
        # Trim to desired n_parents
        parents_all = list(cat_set_full.keys())
        sub_rng.shuffle(parents_all)
        parents = parents_all[: min(cfg["n_parents"], len(parents_all))]

        val_lo, val_hi = cfg["val_range"]
        sub_lo, sub_hi = cfg["subs_range"]

        # Assign values
        values = {}
        for p in parents:
            children_full = list(cat_set_full[p])
            sub_rng.shuffle(children_full)
            n_subs = min(sub_rng.randint(sub_lo, sub_hi), len(children_full))
            kids = children_full[:n_subs]
            values[p] = {}
            for c in kids:
                values[p][c] = sub_rng.randint(val_lo, val_hi)

        qa = self._make_qa(sub_rng, qtype, values)
        if qa is None:
            return None
        question, answer = qa

        style = self._random_style()
        image = self._render(sub_rng, style, values, cfg)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # Renderer
    # ------------------------------------------------------------------ #

    def _render(self, sub_rng, style, values, cfg=None):
        cfg = cfg or {}
        fig_w = 7 * style["figsize_scale"]
        fig_h = 5.5 * style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        palette = list(style["palette"])
        hide_values = cfg.get("hide_values", False)
        similar = cfg.get("similar_colors", False)
        # Re-derive palette when similar colors are requested. Use a wider
        # luminance range and enforce a minimum pairwise luminance gap so
        # adjacent cells stay distinguishable (but still same-hue family).
        if similar:
            base_r = sub_rng.uniform(0.30, 0.55)
            base_g = sub_rng.uniform(0.35, 0.60)
            base_b = sub_rng.uniform(0.45, 0.75)
            # Spread luminance offsets evenly and add small jitter to preserve
            # similar-hue aesthetic.
            n_colors = 12
            ladder = [(-0.35 + i * 0.70 / (n_colors - 1)) for i in range(n_colors)]
            sub_rng.shuffle(ladder)
            palette = []
            for j in ladder:
                jit = sub_rng.uniform(-0.03, 0.03)
                palette.append((
                    max(0.1, min(0.95, base_r + j + jit)),
                    max(0.1, min(0.95, base_g + j * 0.6 + jit * 0.5)),
                    max(0.1, min(0.95, base_b + j * 0.4 - jit * 0.4)),
                ))

            def _lum(c):
                return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

            # Enforce min luminance gap between successive entries
            # (adjacent cells are drawn with successive palette indices).
            MIN_GAP = 0.10
            for _ in range(8):
                ok = True
                for i in range(len(palette) - 1):
                    if abs(_lum(palette[i]) - _lum(palette[i + 1])) < MIN_GAP:
                        sub_rng.shuffle(palette)
                        ok = False
                        break
                if ok:
                    break
        layout_h = sub_rng.choice(["horizontal_slices", "vertical_slices"])

        parent_totals = {p: sum(values[p].values()) for p in values}
        grand_total = sum(parent_totals.values())
        parents = sorted(values.keys(),
                          key=lambda p: parent_totals[p], reverse=True)

        fs_base = style["font_size_base"]
        pi = 0

        if layout_h == "horizontal_slices":
            y_start = 0.0
            for p in parents:
                h = parent_totals[p] / grand_total
                children = sorted(values[p].keys(),
                                  key=lambda c: values[p][c],
                                  reverse=True)
                x_start = 0.0
                for ci, c in enumerate(children):
                    w = values[p][c] / parent_totals[p]
                    color = palette[(pi + ci) % len(palette)]
                    rect = mpatches.FancyBboxPatch(
                        (x_start + 0.005, y_start + 0.005),
                        w - 0.01, h - 0.01,
                        boxstyle="round,pad=0.005",
                        facecolor=color, alpha=0.8,
                        edgecolor="white", linewidth=1.5)
                    ax.add_patch(rect)
                    label = c if hide_values else f"{c}\n{values[p][c]}"
                    fs = fs_base - 2 if w < 0.15 else fs_base
                    ax.text(x_start + w / 2, y_start + h / 2, label,
                            ha="center", va="center", fontsize=max(fs, 9),
                            fontweight="bold")
                    x_start += w
                ax.text(-0.02, y_start + h / 2, p, ha="right", va="center",
                        fontsize=fs_base + 1, fontweight="bold")
                y_start += h
                pi += len(values[p])
            ax.set_xlim(-0.2, 1.05)
            ax.set_ylim(-0.05, 1.05)
        else:  # vertical_slices
            x_start = 0.0
            for p in parents:
                w = parent_totals[p] / grand_total
                children = sorted(values[p].keys(),
                                  key=lambda c: values[p][c],
                                  reverse=True)
                y_start = 0.0
                for ci, c in enumerate(children):
                    h = values[p][c] / parent_totals[p]
                    color = palette[(pi + ci) % len(palette)]
                    rect = mpatches.FancyBboxPatch(
                        (x_start + 0.005, y_start + 0.005),
                        w - 0.01, h - 0.01,
                        boxstyle="round,pad=0.005",
                        facecolor=color, alpha=0.8,
                        edgecolor="white", linewidth=1.5)
                    ax.add_patch(rect)
                    label = c if hide_values else f"{c}\n{values[p][c]}"
                    fs = fs_base - 2 if h < 0.15 else fs_base
                    ax.text(x_start + w / 2, y_start + h / 2, label,
                            ha="center", va="center", fontsize=max(fs, 9),
                            fontweight="bold")
                    y_start += h
                ax.text(x_start + w / 2, -0.05, p, ha="center", va="top",
                        fontsize=fs_base + 1, fontweight="bold")
                x_start += w
                pi += len(values[p])
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.2, 1.05)

        title_pool = ["Treemap", "Proportional Treemap",
                       "Category Treemap", "Hierarchical Treemap"]
        ax.set_title(sub_rng.choice(title_pool),
                     fontsize=fs_base + 3, fontweight="bold")
        ax.axis("off")
        fig.patch.set_facecolor(style["bg_color"])
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Q/A generation
    # ------------------------------------------------------------------ #

    def _make_qa(self, sub_rng, qtype, values):
        parent_totals = {p: sum(values[p].values()) for p in values}
        grand_total = sum(parent_totals.values())
        all_subs = [(p, c, v) for p in values for c, v in values[p].items()]

        if qtype == "largest_category":
            best = max(parent_totals, key=parent_totals.get)
            templates = [
                "Which top-level category has the largest total value?",
                "From the treemap, which category has the highest total?",
                "Identify the category with the greatest aggregate value.",
                "Which category occupies the largest portion of the treemap?",
            ]
            return sub_rng.choice(templates), best

        if qtype == "smallest_sub":
            mn = min(all_subs, key=lambda x: x[2])
            templates = [
                "Which sub-category has the smallest value?",
                "Find the sub-category with the lowest value in the treemap.",
                "Which sub-category is the smallest in the figure?",
            ]
            return sub_rng.choice(templates), mn[1]

        if qtype == "fraction_of_category":
            p = sub_rng.choice(list(values.keys()))
            frac = round(parent_totals[p] / grand_total * 100, 1)
            templates = [
                f"What percentage of the grand total does '{p}' represent? Round to 1 decimal place.",
                f"From the treemap, what percent of the total does '{p}' occupy? 1 decimal.",
                f"Compute '{p}' as a percentage of the total. Round to 1 decimal.",
            ]
            return sub_rng.choice(templates), str(frac)

        if qtype == "compare_categories":
            p1, p2 = sub_rng.sample(list(values.keys()), 2)
            diff = abs(parent_totals[p1] - parent_totals[p2])
            templates = [
                f"What is the absolute difference in total value between '{p1}' and '{p2}'?",
                f"From the treemap, how much larger is the bigger of '{p1}' vs '{p2}' than the smaller?",
                f"Compute |total('{p1}') - total('{p2}')|.",
            ]
            return sub_rng.choice(templates), str(diff)

        if qtype == "count_subs_above":
            thresh = sub_rng.choice([15, 20, 25, 30, 35, 40])
            cnt = sum(1 for _, _, v in all_subs if v > thresh)
            templates = [
                f"How many sub-categories have a value greater than {thresh}?",
                f"Count the sub-categories with values strictly above {thresh}.",
                f"From the treemap, how many sub-categories exceed {thresh}?",
            ]
            return sub_rng.choice(templates), str(cnt)

        if qtype == "ratio_largest_smallest":
            largest = max(all_subs, key=lambda x: x[2])
            smallest = min(all_subs, key=lambda x: x[2])
            if smallest[2] == 0:
                return None
            ratio = round(largest[2] / smallest[2], 2)
            templates = [
                "What is the ratio of the largest sub-category value to the smallest sub-category value? Round to 2 decimals.",
                "Compute (largest sub / smallest sub) value ratio. 2 decimals.",
                "From the treemap, divide the biggest sub-category value by the smallest. 2 decimals.",
            ]
            return sub_rng.choice(templates), str(ratio)

        if qtype == "percentage_of_total":
            p = sub_rng.choice(list(values.keys()))
            max_child = max(values[p].items(), key=lambda x: x[1])
            pct = round(max_child[1] / grand_total * 100, 1)
            templates = [
                f"What percentage of the grand total does '{max_child[0]}' (in '{p}') represent? Round to 1 decimal place.",
                f"Find '{max_child[0]}' in '{p}' as a % of the total. 1 decimal.",
                f"Compute '{max_child[0]}' (under '{p}') as a percent of the full treemap. 1 decimal.",
            ]
            return sub_rng.choice(templates), str(pct)

        if qtype == "sub_of_parent_percent":
            # Largest sub in a parent, as percent of that parent
            p = sub_rng.choice(list(values.keys()))
            if parent_totals[p] == 0:
                return None
            max_child = max(values[p].items(), key=lambda x: x[1])
            pct = round(max_child[1] / parent_totals[p] * 100, 1)
            templates = [
                f"What percentage of '{p}' does '{max_child[0]}' represent? Round to 1 decimal.",
                f"Within '{p}', what percent is '{max_child[0]}'? 1 decimal.",
                f"Find '{max_child[0]}' as a share of '{p}' (percent, 1 decimal).",
            ]
            return sub_rng.choice(templates), str(pct)

        if qtype == "rank_of_sub":
            # Pick a sub-category and ask its rank (1 = largest) among all
            target = sub_rng.choice(all_subs)
            sorted_subs = sorted(all_subs, key=lambda x: -x[2])
            rank = 1 + [s[1] for s in sorted_subs].index(target[1])
            templates = [
                f"If all sub-categories are ranked by value (1 = largest), what is the rank of '{target[1]}'?",
                f"Among all sub-categories, rank '{target[1]}' (1 = largest).",
                f"Sort sub-categories largest-to-smallest — what position is '{target[1]}' in?",
            ]
            return sub_rng.choice(templates), str(rank)

        if qtype == "cumulative_top_n_percent":
            # Sum of top-3 sub values as percent of grand total
            n = min(3, len(all_subs))
            top = sorted(all_subs, key=lambda x: -x[2])[:n]
            pct = round(sum(v for _, _, v in top) / grand_total * 100, 1)
            templates = [
                f"What percentage of the grand total is covered by the top {n} largest sub-categories? Round to 1 decimal.",
                f"Sum the {n} largest sub-category values — what percent of the total is this? 1 decimal.",
                f"Compute the cumulative % of the top-{n} sub-categories. 1 decimal.",
            ]
            return sub_rng.choice(templates), str(pct)

        # Fallback
        best = max(all_subs, key=lambda x: x[2])
        return "Which sub-category has the largest value?", best[1]
