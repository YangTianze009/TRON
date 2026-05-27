"""
Proportion/ratio QA (redesigned 2026-04-16) — shaded vs unshaded grid,
fraction bars, circle sectors, dot clusters.

Critical fix (vs Grade D baseline):
  * Old text leaked the numerator/denominator in the question stem
    (e.g. "The ratio of shaded to unshaded cells is 7:13. Simplify...").
    This made the problem text-solvable with no image needed.
  * Now: question uses "as shown" / "in the image"; numbers reside on
    the image only. Title is neutral ("Grid", "Bars", "Fractions").
  * Expanded representation pool: grid, bar, circle-sector (pie),
    dot-cluster, number-line-segment.
  * Diverse colors per seed; L0/L9 structural shift.
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

_NEUTRAL_TITLES_GRID = [
    "Grid", "Shaded Grid", "Shaded cells", "Grid pattern",
    "Shading", "Grid figure", "Squares",
]

_NEUTRAL_TITLES_BARS = [
    "Bars", "Fraction bars", "Strip chart", "Segmented bars",
    "Horizontal bars", "Bar display",
]

_NEUTRAL_TITLES_PIE = [
    "Pie chart", "Circle", "Sector chart", "Wedges",
    "Circular diagram",
]

_NEUTRAL_TITLES_DOTS = [
    "Dot collection", "Dots", "Markers", "Scattered dots",
    "Dot group",
]

class ProportionRatioQA(StandaloneVisualEnv):
    ENV_NAME = "proportion_ratio"

    @staticmethod
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {
                "qtypes": ["simplify_ratio", "compare_fractions",
                           "fraction_shaded"],
                "qweights": [4, 4, 2],
                "reps": ["grid", "bar", "dots"],
                "grid_max": 5,
            }
        if level <= 2:
            return {
                "qtypes": ["compare_fractions", "equivalent_fraction",
                           "ratio_to_percentage", "fraction_shaded"],
                "qweights": [3, 3, 2, 2],
                "reps": ["grid", "bar", "pie", "dots"],
                "grid_max": 6,
            }
        if level <= 4:
            return {
                "qtypes": ["equivalent_fraction", "ratio_to_percentage",
                           "decimal_to_fraction_simplified",
                           "compare_fractions"],
                "qweights": [3, 3, 3, 1],
                "reps": ["grid", "bar", "pie", "dots"],
                "grid_max": 6,
            }
        if level <= 6:
            return {
                "qtypes": ["ratio_to_percentage",
                           "decimal_to_fraction_simplified",
                           "equivalent_fraction"],
                "qweights": [3, 5, 2],
                "reps": ["grid", "bar", "pie"],
                "grid_max": 7,
            }
        if level <= 8:
            return {
                "qtypes": ["decimal_to_fraction_simplified",
                           "ratio_to_percentage"],
                "qweights": [6, 4],
                "reps": ["grid", "bar", "pie"],
                "grid_max": 8,
            }
        return {
            # L9: harder multi-part ratio chains.
            "qtypes": ["three_bar_sum",
                       "decimal_to_fraction_simplified"],
            "qweights": [7, 3],
            "reps": ["grid", "bar"],
            "grid_max": 10,
        }

    def _generate_problem(self, seed, parameter
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 2201)
        vis_rng = random.Random(seed * 1000 + level * 37 + 9973)
        style = self._random_style()
        qtype = parameter.get("question_type")
        valid = {"fraction_shaded", "simplify_ratio", "compare_fractions",
                 "equivalent_fraction", "shaded_count",
                 "ratio_to_percentage", "decimal_to_fraction_simplified",
                 "three_bar_sum"}
        if qtype not in valid:
            qtype = sub_rng.choices(cfg["qtypes"],
                                    weights=cfg["qweights"], k=1)[0]

        if qtype in ("fraction_shaded", "simplify_ratio", "shaded_count"):
            return self._single_shape_problem(rng, sub_rng, vis_rng,
                                              style, qtype, cfg)
        elif qtype == "compare_fractions":
            return self._bar_compare(rng, sub_rng, vis_rng, style)
        elif qtype == "equivalent_fraction":
            return self._equivalent(rng, sub_rng, vis_rng, style)
        elif qtype == "ratio_to_percentage":
            return self._ratio_to_percentage(rng, sub_rng, vis_rng,
                                             style, cfg)
        elif qtype == "decimal_to_fraction_simplified":
            return self._decimal_to_fraction(rng, sub_rng, vis_rng, style)
        elif qtype == "three_bar_sum":
            return self._three_bar_sum(rng, sub_rng, vis_rng, style)
        return None

    # ------------------------------------------------------------------
    # Problem builders
    # ------------------------------------------------------------------

    def _single_shape_problem(self, rng, sub_rng, vis_rng, style, qtype, cfg):
        """Show one shape (grid / pie / bar / dots); ask about the shaded
        fraction, ratio, or count — WITHOUT writing the numbers in text."""
        rep = vis_rng.choice(cfg["reps"])
        if rep == "grid":
            rows = rng.randint(2, cfg["grid_max"])
            cols = rng.randint(2, cfg["grid_max"])
        elif rep == "dots":
            rows, cols = rng.randint(3, 6), rng.randint(3, 6)
        elif rep == "pie":
            rows, cols = 1, rng.randint(4, 10)  # wedges
        else:
            rows, cols = 1, rng.randint(4, 10)  # bar segments
        total = rows * cols
        shaded = rng.randint(1, max(1, total - 1))

        g = self._gcd(shaded, total)
        simp_num, simp_den = shaded // g, total // g
        unshaded = total - shaded
        g2 = self._gcd(shaded, unshaded) if unshaded > 0 else 1

        img = self._render_shape(vis_rng, rep, rows, cols, shaded, style)

        if qtype == "fraction_shaded":
            q = vis_rng.choice([
                "What fraction of the figure is shaded? "
                "Give a simplified fraction (e.g. 2/3).",
                "As shown in the image, what simplified fraction of "
                "the figure is shaded?",
                "Express the shaded portion of the image as a simplified "
                "fraction.",
            ])
            return q, f"{simp_num}/{simp_den}", img
        if qtype == "simplify_ratio":
            if unshaded == 0:
                return None
            q = vis_rng.choice([
                "The figure shows shaded and unshaded parts. "
                "Give the ratio of shaded to unshaded in simplified form "
                "(e.g. 2:3).",
                "As shown, what is the simplified ratio of shaded to "
                "unshaded parts?",
                "Express the ratio of shaded to unshaded regions (in the "
                "image) in simplest form.",
            ])
            return (q, f"{shaded // g2}:{unshaded // g2}", img)
        if qtype == "shaded_count":
            q = vis_rng.choice([
                "How many parts are shaded in the figure?",
                "Count the shaded parts shown in the image.",
                "How many shaded segments does the figure contain?",
            ])
            return q, str(shaded), img
        return None

    def _bar_compare(self, rng, sub_rng, vis_rng, style):
        """Show two bars, ask which is larger — by LETTER, not by fraction."""
        n1, d1 = rng.randint(1, 7), rng.randint(2, 8)
        n2, d2 = rng.randint(1, 7), rng.randint(2, 8)
        attempts = 0
        while n1 / d1 == n2 / d2 and attempts < 10:
            n2 = rng.randint(1, 7)
            attempts += 1
        if n1 / d1 == n2 / d2:
            return None

        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 3 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        bar_w = 5
        for idx, (n, d, y, lbl) in enumerate(
                [(n1, d1, 1.5, "A"), (n2, d2, 0.3, "B")]):
            seg = bar_w / d
            for k in range(d):
                # BUGFIX 2026-04-24: unfilled cells use visible light-gray
                # fill at full opacity (was white at alpha 0.2, invisible).
                fc = palette[idx] if k < n else "#d0d0d0"
                alpha = 0.85 if k < n else 0.9
                rect = mpatches.Rectangle(
                    (k * seg, y), seg - 0.05, 0.8,
                    fc=fc, ec=style["geo_line_color"],
                    linewidth=style["line_width"], alpha=alpha)
                ax.add_patch(rect)
            # Only the letter label (not the fraction) — the count of
            # shaded vs unshaded segments is the ONLY source of info
            ax.text(-0.7, y + 0.35, f"{lbl}:",
                    fontsize=style["font_size_base"] + 1,
                    fontweight="bold", va="center")

        ax.set_xlim(-1.5, bar_w + 0.5)
        ax.set_ylim(-0.2, 2.8)
        ax.axis("off")
        ax.set_title(vis_rng.choice(_NEUTRAL_TITLES_BARS),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        ans = "A" if n1 / d1 > n2 / d2 else "B"
        q = vis_rng.choice([
            "Which bar represents the larger fraction, A or B? Answer A or B.",
            "Compare the two bars. Which shows the larger shaded fraction, "
            "A or B? Answer A or B.",
            "As shown, which fraction is larger: A or B? Answer A or B.",
        ])
        return q, ans, img

    def _equivalent(self, rng, sub_rng, vis_rng, style):
        """Show two bars that represent equivalent fractions; ask the
        simplified fraction the second bar shows."""
        n = rng.randint(1, 6)
        d = rng.randint(n + 1, 12)
        g = self._gcd(n, d)
        n, d = n // g, d // g
        mult = rng.randint(2, 5)
        nd, dd = n * mult, d * mult

        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6 * sc, 3 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        vis_rng.shuffle(palette)

        bar_w = 4
        seg = bar_w / d
        for k in range(d):
            # BUGFIX 2026-04-24: unfilled cells visible light-gray.
            fc = palette[0] if k < n else "#d0d0d0"
            rect = mpatches.Rectangle(
                (k * seg, 1.5), seg - 0.04, 0.8,
                fc=fc, ec=style["geo_line_color"],
                linewidth=style["line_width"],
                alpha=0.85 if k < n else 0.9)
            ax.add_patch(rect)
        ax.text(-0.8, 1.85, "Top",
                fontsize=style["font_size_base"] + 1, fontweight="bold")

        seg2 = bar_w / dd
        for k in range(dd):
            # BUGFIX 2026-04-24: unfilled cells visible light-gray.
            fc = palette[1 % len(palette)] if k < nd else "#d0d0d0"
            rect = mpatches.Rectangle(
                (k * seg2, 0.3), seg2 - 0.02, 0.8,
                fc=fc, ec=style["geo_line_color"],
                linewidth=style["line_width"],
                alpha=0.85 if k < nd else 0.9)
            ax.add_patch(rect)
        ax.text(-0.8, 0.65, "Bottom",
                fontsize=style["font_size_base"] + 1, fontweight="bold",
                color=palette[2 % len(palette)])

        ax.set_xlim(-1.7, bar_w + 0.5)
        ax.set_ylim(-0.2, 2.8)
        ax.axis("off")
        ax.set_title(vis_rng.choice(["Two fraction bars",
                                      "Equivalent bars",
                                      "Top and bottom bars"]),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "The top bar and bottom bar show equivalent fractions. "
            "What simplified fraction do they represent?",
            "Both bars (top and bottom) represent the same simplified "
            "fraction. What is it? Give your answer in lowest terms.",
            "The two bars in the image represent the same value. "
            "Give that value as a simplified fraction.",
        ])
        return q, f"{n}/{d}", img

    def _ratio_to_percentage(self, rng, sub_rng, vis_rng, style, cfg):
        """Show a grid/pie/bar; ask the percentage shaded (no text leak)."""
        rep = vis_rng.choice(cfg["reps"])
        if rep == "grid":
            rows = rng.randint(3, cfg["grid_max"])
            cols = rng.randint(3, cfg["grid_max"])
        elif rep == "pie":
            rows, cols = 1, rng.randint(5, 12)
        else:
            rows, cols = 1, rng.randint(5, 12)
        total = rows * cols
        shaded = rng.randint(1, max(1, total - 1))

        img = self._render_shape(vis_rng, rep, rows, cols, shaded, style)

        pct = round(shaded / total * 100, 2)
        q = vis_rng.choice([
            "What percentage of the figure is shaded? Round to 2 decimal "
            "places.",
            "As shown in the image, what percent of the figure is shaded? "
            "Round to 2 decimal places.",
            "Compute the percentage of the figure that is shaded. "
            "Round to 2 decimals.",
        ])
        return q, str(pct), img

    def _decimal_to_fraction(self, rng, sub_rng, vis_rng, style):
        """Two bars A and B; ask sum as simplified fraction, reading
        values off the image."""
        n1 = rng.randint(1, 5)
        d1 = rng.randint(n1 + 1, 10)
        n2 = rng.randint(1, 5)
        d2 = rng.randint(n2 + 1, 10)
        g1 = self._gcd(n1, d1)
        n1, d1 = n1 // g1, d1 // g1
        g2 = self._gcd(n2, d2)
        n2, d2 = n2 // g2, d2 // g2
        sum_num = n1 * d2 + n2 * d1
        sum_den = d1 * d2
        g = self._gcd(sum_num, sum_den)
        sum_num, sum_den = sum_num // g, sum_den // g

        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 3 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        bar_w = 5
        for idx, (n, d, y, lbl) in enumerate(
                [(n1, d1, 1.5, "A"), (n2, d2, 0.3, "B")]):
            seg = bar_w / d
            for k in range(d):
                # BUGFIX 2026-04-24: unfilled cells use visible light-gray
                # fill at full opacity (was white at alpha 0.2, invisible).
                fc = palette[idx] if k < n else "#d0d0d0"
                alpha = 0.85 if k < n else 0.9
                rect = mpatches.Rectangle(
                    (k * seg, y), seg - 0.05, 0.8,
                    fc=fc, ec=style["geo_line_color"],
                    linewidth=style["line_width"], alpha=alpha)
                ax.add_patch(rect)
            ax.text(-0.7, y + 0.35, f"{lbl}:",
                    fontsize=style["font_size_base"] + 1,
                    fontweight="bold", va="center")
        ax.set_xlim(-1.5, bar_w + 0.5)
        ax.set_ylim(-0.2, 2.8)
        ax.axis("off")
        ax.set_title(vis_rng.choice(["Add two fractions",
                                      "Bars A and B",
                                      "Two bars"]),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        q = vis_rng.choice([
            "The image shows two fraction bars A and B. What is A + B as a "
            "simplified fraction?",
            "As shown, bar A and bar B represent two fractions. "
            "Compute their sum as a simplified fraction (e.g. 3/4).",
            "Read the values of bar A and bar B from the image, then give "
            "A + B as a simplified fraction.",
        ])
        return q, f"{sum_num}/{sum_den}", img

    def _three_bar_sum(self, rng, sub_rng, vis_rng, style):
        """L9 chain: three bars A, B, C with distinct denominators.
        Question asks (A + B) - C, or similar chain, as simplified
        fraction. Values live ONLY on the image."""
        def _pick():
            n = rng.randint(1, 6)
            d = rng.randint(max(3, n + 1), 9)
            g = self._gcd(n, d)
            return n // g, d // g

        # Resample until the chain produces a positive numerator. With
        # a + b - c there is always a non-trivial chance of a non-positive
        # result; resampling rather than returning None keeps every seed live.
        n1 = d1 = n2 = d2 = n3 = d3 = 0
        shape = vis_rng.choice(["a_plus_b_minus_c", "a_plus_b_plus_c"])
        num = 0
        for _ in range(50):
            n1, d1 = _pick()
            n2, d2 = _pick()
            n3, d3 = _pick()
            if shape == "a_plus_b_minus_c":
                num = n1 * d2 * d3 + n2 * d1 * d3 - n3 * d1 * d2
            else:
                num = n1 * d2 * d3 + n2 * d1 * d3 + n3 * d1 * d2
            if num > 0:
                break
        if num <= 0:
            # Fall back to plus shape (always positive).
            shape = "a_plus_b_plus_c"
            num = n1 * d2 * d3 + n2 * d1 * d3 + n3 * d1 * d2
        if shape == "a_plus_b_minus_c":
            prompt = vis_rng.choice([
                "The image shows three fraction bars A, B, C. "
                "Compute A + B - C and give the result as a simplified "
                "fraction.",
                "Read the three bars (A, B, C) in the image. "
                "Evaluate A + B - C as a simplified fraction (e.g. 3/4).",
            ])
        else:
            prompt = vis_rng.choice([
                "The image shows three fraction bars A, B, C. "
                "Compute A + B + C and give the result as a simplified "
                "fraction.",
                "Read the three bars (A, B, C) from the image. "
                "Evaluate A + B + C as a simplified fraction.",
            ])
        den = d1 * d2 * d3
        g = self._gcd(num, den)
        num_s, den_s = num // g, den // g

        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7.5 * sc, 4 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        bar_w = 5
        rows = [(n1, d1, 2.6, "A"),
                (n2, d2, 1.4, "B"),
                (n3, d3, 0.2, "C")]
        for idx, (n, d, y, lbl) in enumerate(rows):
            seg = bar_w / d
            for k in range(d):
                # BUGFIX 2026-04-24: unfilled cells visible light-gray.
                fc = palette[idx % len(palette)] if k < n else "#d0d0d0"
                alpha = 0.85 if k < n else 0.9
                rect = mpatches.Rectangle(
                    (k * seg, y), seg - 0.05, 0.8,
                    fc=fc, ec=style["geo_line_color"],
                    linewidth=style["line_width"], alpha=alpha)
                ax.add_patch(rect)
            ax.text(-0.7, y + 0.4, f"{lbl}:",
                    fontsize=style["font_size_base"] + 1,
                    fontweight="bold", va="center")
        ax.set_xlim(-1.5, bar_w + 0.5)
        ax.set_ylim(-0.2, 3.8)
        ax.axis("off")
        ax.set_title(vis_rng.choice(["Three fraction bars",
                                      "Bars A, B and C",
                                      "A, B, C"]),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])
        return prompt, f"{num_s}/{den_s}", img

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_shape(self, vis_rng, rep, rows, cols, shaded, style):
        """Render a single shape (grid/bar/pie/dots) with `shaded` cells
        filled in. All labels are neutral — no counts in titles."""
        palette = list(style["palette"])
        vis_rng.shuffle(palette)
        color_main = palette[0]

        total = rows * cols
        shade_mask = [1] * shaded + [0] * (total - shaded)
        vis_rng.shuffle(shade_mask)

        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(5 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])

        if rep == "grid":
            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    fc = color_main if shade_mask[idx] else "white"
                    alpha = 0.8 if shade_mask[idx] else 0.3
                    rect = mpatches.FancyBboxPatch(
                        (j, rows - 1 - i), 0.9, 0.9,
                        boxstyle="round,pad=0.02",
                        fc=fc, ec=style["geo_line_color"],
                        linewidth=style["line_width"], alpha=alpha)
                    ax.add_patch(rect)
            ax.set_xlim(-0.3, cols + 0.3)
            ax.set_ylim(-0.3, rows + 0.3)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(vis_rng.choice(_NEUTRAL_TITLES_GRID),
                         fontsize=style["font_size_base"] + 2,
                         fontweight="bold")
        elif rep == "bar":
            bar_w = 6
            seg = bar_w / cols
            for k in range(cols):
                fc = color_main if shade_mask[k] else "white"
                alpha = 0.85 if shade_mask[k] else 0.25
                rect = mpatches.Rectangle(
                    (k * seg, 1.0), seg - 0.04, 1.0,
                    fc=fc, ec=style["geo_line_color"],
                    linewidth=style["line_width"], alpha=alpha)
                ax.add_patch(rect)
            ax.set_xlim(-0.4, bar_w + 0.4)
            ax.set_ylim(-0.2, 3.2)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(vis_rng.choice(_NEUTRAL_TITLES_BARS),
                         fontsize=style["font_size_base"] + 2,
                         fontweight="bold")
        elif rep == "pie":
            sector_count = cols
            radii = 1.6
            start = vis_rng.uniform(0, 2 * math.pi)
            for k in range(sector_count):
                theta1 = math.degrees(start + 2 * math.pi * k / sector_count)
                theta2 = math.degrees(start + 2 * math.pi * (k + 1)
                                       / sector_count)
                fc = color_main if shade_mask[k] else "white"
                wedge = mpatches.Wedge(
                    (0, 0), radii, theta1, theta2,
                    facecolor=fc, edgecolor=style["geo_line_color"],
                    linewidth=style["line_width"],
                    alpha=0.85 if shade_mask[k] else 0.25)
                ax.add_patch(wedge)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(vis_rng.choice(_NEUTRAL_TITLES_PIE),
                         fontsize=style["font_size_base"] + 2,
                         fontweight="bold")
        else:  # dots
            dots_per_row = cols
            n_rows = rows
            for i in range(n_rows):
                for j in range(dots_per_row):
                    idx = i * dots_per_row + j
                    fc = color_main if shade_mask[idx] else "#cccccc"
                    dot = mpatches.Circle(
                        (j + 0.5, n_rows - 1 - i + 0.5),
                        0.32,
                        facecolor=fc, edgecolor=style["geo_line_color"],
                        linewidth=0.8,
                        alpha=0.9 if shade_mask[idx] else 0.5)
                    ax.add_patch(dot)
            ax.set_xlim(-0.3, dots_per_row + 0.3)
            ax.set_ylim(-0.3, n_rows + 0.3)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(vis_rng.choice(_NEUTRAL_TITLES_DOTS),
                         fontsize=style["font_size_base"] + 2,
                         fontweight="bold")

        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = ProportionRatioQA()
    for lv in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": lv})
            print(f"L{lv} s{seed} ok={ok} A={env._answer}")
