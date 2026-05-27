"""
Visual Word Problem Warmup QA environment.

Target regression: targeted-geometry Applied -4.59. Combine a small visual scene
with a short word problem and ask for a numeric answer.

Diversity & difficulty redesign (2026-04-16):
- Level-aware sub_rng.
- Text leakage reduced: at L>=4, numeric values for the primary quantities
  (dimensions, counts, prices, totals) are ONLY on the image; the question
  says "as shown". L0-L3 still repeat numbers in text for scaffolding.
- More visual kinds: shape-row divide, pool-volume, percentage-bar, coin-&-item,
  area-to-square, plus new kinds: scaled-length, pie-fraction.
- 3-4 phrasings per kind.
- Colors, scene offsets jittered per seed.

All answers are integers.
"""
import math
import random
import textwrap
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, Wedge
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class VisualWordProblemWarmupQA(StandaloneVisualEnv):
    ENV_NAME = "visual_word_problem_warmup"

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"kinds": ["divide_equal", "pie_fraction"],
                    "mag_scale": 1, "scaffold_text": True}
        if level <= 2:
            return {"kinds": ["divide_equal", "pie_fraction",
                              "rate_volume"],
                    "mag_scale": 1, "scaffold_text": True}
        if level <= 4:
            return {"kinds": ["rate_volume", "percentage_bar",
                              "scaled_length"],
                    "mag_scale": 2, "scaffold_text": False}
        if level <= 6:
            return {"kinds": ["percentage_bar", "rate_cost",
                              "scaled_length"],
                    "mag_scale": 3, "scaffold_text": False}
        if level <= 7:
            return {"kinds": ["rate_cost", "rect_area_to_square",
                              "percentage_bar"],
                    "mag_scale": 3, "scaffold_text": False}
        # L8-L9: larger magnitudes and only the harder kinds (rect_area_to_square
        # requires factor-inference + sqrt; rate_cost requires integer floor-div
        # at multi-digit prices; percentage_bar uses larger, non-round totals).
        return {"kinds": ["rect_area_to_square", "rate_cost",
                          "percentage_bar"],
                "mag_scale": 5, "scaffold_text": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(30):
            r = self._try_generate(parameter)
            if r is not None:
                return r
        return None

    # ------------------------------------------------------------------ #
    def _try_generate(self, parameter: Dict):
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        scaffold = cfg.get("scaffold_text", False)

        kind = rng.choice(cfg["kinds"])
        scale = cfg["mag_scale"]

        if kind == "divide_equal":
            primary = 1
            n = rng.randint(2, 5)
            per_item = rng.randint(2 * scale, 7 * scale)
            total = n * per_item
            noun = rng.choice(
                ["circles", "triangles", "squares", "stars", "diamonds"]
            )
            phrasings = [
                f"The {noun} below cost {total} coins in total, with each "
                f"{noun[:-1]} costing the same. How many coins does one cost?",
                f"All {noun} shown cost the same. Their combined price is "
                f"{total} coins. What is the cost of one {noun[:-1]}?",
                f"The total cost of the {noun} shown is {total} coins. "
                f"Each costs the same amount. How much is one?",
            ]
            nonleak = [
                f"The {noun} shown cost the same amount each. The diagram "
                f"labels the total cost. How many coins does one "
                f"{noun[:-1]} cost? Integer only.",
                f"The diagram labels the total price for all {noun}. Each "
                f"costs the same. What is the cost per {noun[:-1]}? Integer.",
            ]
            if scaffold:
                story = rng.choice(phrasings) + " Answer with a single integer."
                img_total = None
            else:
                story = rng.choice(nonleak)
                img_total = total
            answer = per_item
            img = self._render_shape_row(n, noun, story, img_total=total if not scaffold else None, rng=rng)

        elif kind == "rate_volume":
            primary = 2
            length = rng.randint(2, 4 + scale)
            width = rng.randint(2, 4 + scale)
            rate = rng.randint(2, 4 + scale)
            minutes = rng.randint(2, 4 + scale)
            added = rate * minutes
            if scaffold:
                story = (
                    f"The rectangle below represents a swimming pool of "
                    f"{length} m by {width} m. Water is pumped in at "
                    f"{rate} cubic metres per minute for {minutes} minutes. "
                    f"How many cubic metres were added? Answer with a single "
                    f"integer."
                )
            else:
                story = (
                    "The rectangle is a swimming pool (dimensions labeled "
                    "in the diagram). Water is pumped in at the rate shown "
                    "for the number of minutes shown. How many cubic metres "
                    "of water were added? Integer only."
                )
            answer = added
            img = self._render_rect_pool(length, width, rate, minutes, story, scaffold, rng)

        elif kind == "percentage_bar":
            primary = 3
            for _ in range(25):
                a = rng.randint(2, 9 * scale)
                b = rng.randint(2, 9 * scale)
                tot = a + b
                if tot == 0:
                    continue
                if (a * 100) % tot == 0:
                    break
            else:
                return None
            pct_a = (a * 100) // tot
            if scaffold:
                story = (
                    f"The bar chart shows two values: A = {a} and B = {b}. "
                    f"What percentage of the total (A + B) is A? "
                    f"Answer with a single integer."
                )
            else:
                story = (
                    "The bar chart shows two values, A and B, with their "
                    "magnitudes labeled. What percentage of the total "
                    "(A + B) is A? Integer only."
                )
            answer = pct_a
            img = self._render_bar_two(a, b, story, scaffold, rng)

        elif kind == "rate_cost":
            primary = 4
            p = rng.randint(2, 4 + scale)
            max_items = rng.randint(3, 6 + scale)
            d = p * rng.randint(2, max_items) + rng.randint(0, p - 1)
            answer = d // p
            items = rng.choice(["muffin", "pencil", "apple", "sticker",
                                "cookie", "ticket"])
            if scaffold:
                story = (
                    f"A {items} costs {p} coins each. The buyer has {d} coins. "
                    f"How many {items}s can the buyer afford at most? Integer."
                )
            else:
                story = (
                    f"The diagram labels the price of one {items} and the "
                    f"buyer's total coins. How many {items}s can they afford "
                    f"at most? Integer only."
                )
            img = self._render_coins_and_item(p, d, items, story, scaffold, rng)

        elif kind == "rect_area_to_square":
            primary = 5
            for _ in range(25):
                s = rng.randint(3, 4 + scale)
                area = s * s
                factors = [
                    (i, area // i) for i in range(2, area)
                    if area % i == 0 and i != area // i
                ]
                if not factors:
                    continue
                a, b = rng.choice(factors)
                break
            else:
                return None
            answer = s
            if scaffold:
                story = (
                    f"The rectangle below has dimensions {a} by {b}. Its area "
                    f"is reshaped into a square of equal area. What is the "
                    f"side length of the square? Integer."
                )
            else:
                story = (
                    "The rectangle's dimensions are labeled in the diagram. "
                    "If its area is reshaped into a square of equal area, "
                    "what is the square's side length? Integer."
                )
            img = self._render_rect_to_square(a, b, s, story, scaffold, rng)

        elif kind == "pie_fraction":
            primary = 1
            slices = rng.randint(3, 6 + scale // 2)
            total = rng.randint(2, 5 * scale) * slices  # divisible
            per_slice = total // slices
            if scaffold:
                story = (
                    f"A pie chart has {slices} equal slices. The total is "
                    f"{total}. What value does one slice represent? Integer."
                )
            else:
                story = (
                    "The pie chart has equal-sized slices. The total (sum "
                    "of all slices) is labeled in the diagram. What value "
                    "does a single slice represent? Integer."
                )
            answer = per_slice
            img = self._render_pie_chart(slices, total, story, scaffold, rng)

        elif kind == "scaled_length":
            primary = 2
            scale_factor = rng.choice([2, 3, 4, 5])
            small_len = rng.randint(2, 4 + scale)
            large_len = small_len * scale_factor
            if scaffold:
                story = (
                    f"The figure shows a small bar of length {small_len} and "
                    f"a large bar that is {scale_factor}x the small. What is "
                    f"the large bar's length? Integer."
                )
            else:
                story = (
                    "The figure shows a small bar whose length is labeled, "
                    "and a larger bar labeled with its scale factor (e.g., "
                    "'3x'). What is the large bar's length? Integer."
                )
            answer = large_len
            img = self._render_scaled_bars(small_len, scale_factor, story, scaffold, rng)

        else:
            return None

        self._primary_complexity_feature = primary
        if abs(answer) > 3000:
            return None
        return story, str(int(answer)), img

    # ------------------------------------------------------------------ #
    def _new_fig(self, seed_style=None):
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax, style

    def _add_story(self, fig, story):
        fig.text(
            0.5, 0.94,
            "\n".join(textwrap.wrap(story, 60)),
            ha="center", va="top", fontsize=10, color="#1a1a1a",
        )

    def _render_shape_row(self, n, noun, story, img_total, rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        color = palette[0]
        for i in range(n):
            if "circles" in noun.lower():
                ax.add_patch(Circle((i * 1.4, 0), 0.5,
                                    facecolor=color, edgecolor="black",
                                    lw=1.4))
            elif "triangles" in noun.lower():
                pts = [(i * 1.4 - 0.5, -0.4), (i * 1.4 + 0.5, -0.4),
                       (i * 1.4, 0.5)]
                ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                                     edgecolor="black", lw=1.4))
            elif "squares" in noun.lower():
                ax.add_patch(Rectangle((i * 1.4 - 0.45, -0.45), 0.9, 0.9,
                                       facecolor=color, edgecolor="black",
                                       lw=1.4))
            elif "stars" in noun.lower():
                theta = np.linspace(0, 2 * math.pi, 11)[:-1]
                radii = [0.5 if k % 2 == 0 else 0.2 for k in range(10)]
                pts = [(i * 1.4 + r * math.cos(t + math.pi / 2),
                        r * math.sin(t + math.pi / 2))
                       for t, r in zip(theta, radii)]
                ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                                     edgecolor="black", lw=1.4))
            else:  # diamonds
                pts = [(i * 1.4, -0.5), (i * 1.4 + 0.4, 0),
                       (i * 1.4, 0.5), (i * 1.4 - 0.4, 0)]
                ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                                     edgecolor="black", lw=1.4))

        # Always show total on image (regardless of scaffold) — image is truth
        if img_total is not None:
            ax.text((n - 1) * 0.7, 1.4,
                    f"Total cost: {img_total} coins",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#fff9c4", edgecolor="#e67e22"))

        ax.set_xlim(-1, n * 1.4 + 0.5)
        ax.set_ylim(-2, 3)
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_rect_pool(self, length, width, rate, minutes, story,
                          scaffold, rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        rect = Rectangle((0, 0), length, width,
                         facecolor=palette[2], edgecolor="black", lw=2.0)
        ax.add_patch(rect)
        ax.text(length / 2, -0.5, f"{length} m",
                ha="center", va="top", fontsize=13, fontweight="bold")
        ax.text(-0.3, width / 2, f"{width} m",
                ha="right", va="center", fontsize=13, fontweight="bold")

        if not scaffold:
            # Embed rate & minutes in image
            ax.text(length / 2, width + 0.6,
                    f"Rate: {rate} m\u00b3/min  |  Time: {minutes} min",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#fff9c4", edgecolor="#e67e22"))
        ax.set_xlim(-2, length + 2)
        ax.set_ylim(-2, width + 2.5)
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_bar_two(self, a, b, story, scaffold, rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        width = 0.5
        ax.add_patch(Rectangle((0.3, 0), width, a,
                               facecolor=palette[0], edgecolor="black", lw=1.5))
        ax.add_patch(Rectangle((1.3, 0), width, b,
                               facecolor=palette[1], edgecolor="black", lw=1.5))
        ax.text(0.55, -0.3, "A", ha="center", va="top",
                fontsize=13, fontweight="bold")
        ax.text(1.55, -0.3, "B", ha="center", va="top",
                fontsize=13, fontweight="bold")
        # Values always on image (this env needs them to solve)
        ax.text(0.55, a + 0.3, f"{a}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.text(1.55, b + 0.3, f"{b}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.5, max(a, b) + 2)
        ax.axhline(0, color="black", lw=1.0)
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_coins_and_item(self, p, d, items, story, scaffold,
                                rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        ax.add_patch(FancyBboxPatch((0, 0), 1.5, 1.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=palette[0],
                                    edgecolor="black", lw=1.8))
        ax.text(0.75, 0.75, items, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1a1a1a")
        ax.text(0.75, -0.3, f"price: {p} coins",
                ha="center", va="top", fontsize=12, fontweight="bold")
        ax.add_patch(Rectangle((3.0, 0), 2.2, 1.0,
                               facecolor=palette[2],
                               edgecolor="black", lw=1.8))
        ax.text(4.1, 0.5, f"{d} coins", ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(4.1, -0.3, "wallet", ha="center", va="top",
                fontsize=10, style="italic")
        ax.set_xlim(-0.5, 6.0)
        ax.set_ylim(-1.5, 2.0)
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_rect_to_square(self, a, b, s, story, scaffold,
                                rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        ax.add_patch(Rectangle((0, 0), a, b,
                               facecolor=palette[1],
                               edgecolor="black", lw=1.8))
        ax.text(a / 2, -0.4, f"{a}", ha="center", va="top",
                fontsize=12, fontweight="bold")
        ax.text(-0.3, b / 2, f"{b}", ha="right", va="center",
                fontsize=12, fontweight="bold")
        ax.text(a / 2, b / 2, "Area = ?", ha="center", va="center",
                fontsize=12, fontweight="bold", color="#ffffff")
        ax.set_xlim(-1.5, a + 2)
        ax.set_ylim(-1.5, b + 2)
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_pie_chart(self, slices, total, story, scaffold,
                           rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        cx, cy = 0, 0
        r = 1.3
        angle_per = 360.0 / slices
        for i in range(slices):
            w = Wedge((cx, cy), r,
                      i * angle_per, (i + 1) * angle_per,
                      facecolor=palette[i % len(palette)],
                      edgecolor="black", linewidth=1.4)
            ax.add_patch(w)
        # Total label (visible on image always)
        ax.text(cx, cy - 1.6, f"Total = {total}",
                ha="center", va="top", fontsize=13, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#fff9c4", edgecolor="#e67e22"))
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_scaled_bars(self, small_len, scale_factor, story, scaffold,
                             rng) -> Image.Image:
        fig, ax, style = self._new_fig()
        palette = list(style["palette"])
        rng.shuffle(palette)
        bar_h = 0.5
        large_len = small_len * scale_factor
        # Small bar
        ax.add_patch(Rectangle((0, 0.2), small_len, bar_h,
                               facecolor=palette[0], edgecolor="black", lw=1.5))
        ax.text(small_len / 2, 0.45, f"{small_len}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="#ffffff")
        ax.text(small_len + 0.3, 0.45, "small",
                ha="left", va="center", fontsize=11, style="italic")
        # Large bar (displayed without number, just scale label)
        ax.add_patch(Rectangle((0, 1.4), large_len, bar_h,
                               facecolor=palette[2], edgecolor="black", lw=1.5))
        ax.text(large_len / 2, 1.65, f"{scale_factor}x small",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="#ffffff")
        ax.text(large_len + 0.3, 1.65, "large = ?",
                ha="left", va="center", fontsize=11, style="italic",
                color="#c62828")
        ax.set_xlim(-0.5, large_len + 3)
        ax.set_ylim(-0.5, 2.6)
        self._add_story(fig, story)
        fig.tight_layout(rect=[0, 0, 1, 0.80])
        return self.fig_to_pil(fig, dpi=style["dpi"])
