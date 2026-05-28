"""
Attribute Ordering QA environment (v3 planning env 56).

Targets a logic benchmark Attribute Reasoning. Shows 4-8 figures
(labeled A, B, C, ...) that vary in a quantifiable attribute: height, side
count, fill shade, or internal-line count. The model must order the figures
by the asked attribute from least to most, picking the correct ordering
from a 4-way MCQ.

Difficulty:
  - n_figures: 4 + level // 2 (clamped to 8)
  - attribute_confusability: L0 = only target attribute varies; L5 = one
    extra attribute varies as noise; L9 = multiple extra attributes and
    adjacent values (within 10%).
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

_ATTR_BASE_PALETTE = [
    "#3498db", "#e74c3c", "#27ae60", "#e67e22",
    "#8e44ad", "#16a085", "#f39c12", "#2c3e50",
]

def _shade_to_hex(shade: float, base: str = "#34495e") -> str:
    """Blend a base color with white based on shade (0=white, 1=full)."""
    shade = max(0.05, min(0.98, shade))
    r = int(base[1:3], 16)
    g = int(base[3:5], 16)
    b = int(base[5:7], 16)
    # white-lerp: result = white * (1-shade) + base*shade
    rr = int(255 * (1 - shade) + r * shade)
    gg = int(255 * (1 - shade) + g * shade)
    bb = int(255 * (1 - shade) + b * shade)
    return f"#{rr:02x}{gg:02x}{bb:02x}"

class AttributeOrderingQA(StandaloneVisualEnv):
    """Order figures by a quantitative attribute (A4 + P3)."""

    ENV_NAME = "attribute_ordering"

    _ATTR_CHOICES = ["height", "sides", "shade", "lines"]

    _TITLE_VARIANTS = [
        "Order the figures",
        "Figures labeled A-{last}",
        "Attribute ordering",
        "Rank the shapes",
        "Sort by attribute",
    ]

    _QUESTION_STEMS = {
        "height": [
            "Order the figures from shortest to tallest (least to most {attr}).",
            "Rank the shapes by height, from shortest to tallest.",
            "Sort the figures in ascending order of height.",
            "Arrange the shapes from smallest to largest {attr}.",
            "Put the figures in order of increasing height.",
            "Order the shapes from shortest to tallest.",
            "Rank the figures by overall size ({attr}), least to most.",
            "Which ordering lists the figures by increasing height?",
            "Order A-{last} from the shortest figure to the tallest.",
            "Sort the shapes by height ascending (least {attr} first).",
            "List the figures in order of growing height.",
            "From shortest to tallest, how should the figures be ordered?",
            "Order the figures so {attr} increases from left to right in the answer.",
            "Rank the figures least-to-most by {attr}.",
            "Arrange by ascending height: what is the correct order?",
            "Sort the shapes by vertical size (shortest first, tallest last).",
        ],
        "sides": [
            "Order the figures from fewest to most {attr} of the outer polygon.",
            "Sort the shapes by side count, fewest sides first.",
            "Rank the polygons by number of sides (ascending).",
            "Order A-{last} from the polygon with fewest sides to most.",
            "Arrange by increasing side count of the outer polygon.",
            "Put the figures in order of growing side count.",
            "Which ordering lists the shapes by increasing number of sides?",
            "Order the polygons from triangle-like (few sides) to many-sided.",
            "Sort by side count ascending.",
            "From fewest to most {attr}, how should the shapes be ordered?",
            "Rank the figures by number of outer-polygon sides, least to most.",
            "Order the shapes so the side count increases across the sequence.",
            "List the figures in ascending order of side count.",
            "Arrange the polygons by sides: least sides to most.",
            "Which option ranks the shapes by increasing {attr}?",
            "Order by number of sides (ascending).",
        ],
        "shade": [
            "Order the figures from lightest to darkest fill {attr}.",
            "Rank the shapes by fill darkness, lightest first.",
            "Sort the figures in ascending order of fill {attr}.",
            "Arrange A-{last} from the lightest fill to the darkest.",
            "Order the figures by increasing fill darkness.",
            "Which ordering goes from lightest fill to darkest?",
            "Rank the shapes by how dark their fill is, least to most.",
            "Order by fill saturation/darkness (light to dark).",
            "Sort the figures from palest to deepest fill {attr}.",
            "Arrange the shapes so fill darkness grows across the answer.",
            "Put the figures in order of increasing fill {attr}.",
            "List A-{last} from lightest to darkest fill.",
            "Rank by fill color depth, lightest first.",
            "Order the shapes from light fill to dark fill.",
            "Which option orders the figures by increasing {attr}?",
            "Sort by fill {attr}: pale first, dark last.",
        ],
        "lines": [
            "Order the figures from the fewest to the most internal {attr}.",
            "Rank the shapes by internal-line count, fewest first.",
            "Sort the figures in ascending order of internal {attr}.",
            "Arrange A-{last} from the shape with fewest internal lines to the most.",
            "Order the figures by increasing internal-line count.",
            "Which ordering lists the shapes by increasing internal {attr}?",
            "Rank the figures by number of internal lines, least to most.",
            "Sort by internal stripe count ascending.",
            "Order the shapes from no-lines to many-lines inside.",
            "From fewest to most internal {attr}, how should the shapes be ordered?",
            "List the figures in order of growing internal-line count.",
            "Arrange by increasing number of internal stripes.",
            "Order the shapes so interior-line count grows across the answer.",
            "Rank by interior {attr} count (ascending).",
            "Which option orders the figures by increasing interior-line count?",
            "Sort the shapes from fewest interior lines to most.",
        ],
    }

    _ATTR_WORD = {
        "height": "height",
        "sides": "sides",
        "shade": "shade",
        "lines": "lines",
    }

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        """Difficulty axes:
          1) n_figures: 4 + level // 2 (4..8)
          2) confusability: L0 = 1 varying attribute, L>=5 = 2, L>=8 = 3
             value_spacing: L0 = wide, L9 = narrow (~10%)
          3) attr_pool: easier at low levels
        """
        level = max(0, min(9, int(level)))
        n_figures = min(8, 4 + level // 2)
        # How many attributes vary in addition to the asked one
        if level <= 2:
            n_noise = 0
        elif level <= 6:
            n_noise = 1
        else:
            n_noise = 2
        # Spacing between adjacent values (fraction of the full range).
        if level <= 1:
            spacing = 0.22   # very wide
        elif level <= 4:
            spacing = 0.16
        elif level <= 7:
            spacing = 0.11
        else:
            spacing = 0.07   # adjacent values ~7-10%
        if level <= 2:
            attr_pool = ["height", "sides"]
        elif level <= 5:
            attr_pool = ["height", "sides", "shade"]
        else:
            attr_pool = ["height", "sides", "shade", "lines"]
        return {
            "n_figures": n_figures,
            "n_noise": n_noise,
            "spacing": spacing,
            "attr_pool": attr_pool,
        }

    # ------------------------------------------------------------------ #
    # Problem generation
    # ------------------------------------------------------------------ #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1361)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 43 + 1361)
        self._primary_complexity_feature = cfg["n_figures"] + level * 2

        for _ in range(40):
            r = self._try_generate(rng, sub_rng, level, cfg)
            if r is not None:
                return r
        return None

    def _sample_values(self, rng, cfg, attr: str, n: int) -> Optional[List[float]]:
        """Sample n distinct values for an attribute.
        The sampled values span the valid range and are separated by `spacing`.
        Returns None if not enough distinct values can be produced."""
        # Value range per attribute:
        if attr == "height":
            lo, hi = 0.25, 0.95
        elif attr == "sides":
            # integer values 3..8 (6 options). Need at least n distinct options.
            options = list(range(3, 9))
            if len(options) < n:
                return None
            rng.shuffle(options)
            return sorted(options[:n])
        elif attr == "shade":
            lo, hi = 0.10, 0.95
        elif attr == "lines":
            options = list(range(0, 8))
            if len(options) < n:
                return None
            rng.shuffle(options)
            return sorted(options[:n])
        else:
            lo, hi = 0.1, 0.9

        # Continuous: evenly spaced with small jitter, then shuffled later
        spacing = max(cfg["spacing"], (hi - lo) / (2 * n))
        # Try a layout where each consecutive value differs by at least `spacing`.
        base = np.linspace(lo, hi, n)
        jitter = spacing / 3.0
        values = [float(b + rng.uniform(-jitter, jitter)) for b in base]
        # Ensure sorted and spacing preserved
        values.sort()
        for i in range(1, n):
            if values[i] - values[i - 1] < spacing * 0.8:
                values[i] = values[i - 1] + spacing * 0.8
        values = [round(min(hi, max(lo, v)), 4) for v in values]
        if len(set(values)) < n:
            return None
        return values

    def _try_generate(self, rng, sub_rng, level, cfg):
        target_attr = rng.choice(cfg["attr_pool"])
        n_figs = cfg["n_figures"]
        values = self._sample_values(rng, cfg, target_attr, n_figs)
        if values is None:
            return None
        # Shuffle positions: position -> value; GT is the ordering of positions
        # after sorting by value ascending.
        pos = list(range(n_figs))
        rng.shuffle(pos)
        values_at_pos = [None] * n_figs
        for p, v in zip(pos, values):
            values_at_pos[p] = v

        # Build noise attributes (random but do not collide with target).
        noise_attrs = [a for a in self._ATTR_CHOICES if a != target_attr]
        rng.shuffle(noise_attrs)
        noise_selected = noise_attrs[: cfg["n_noise"]]

        # Per-position attribute values
        fig_attrs: List[Dict] = []
        for i in range(n_figs):
            entry = {target_attr: values_at_pos[i]}
            for a in noise_selected:
                # Random noise values (independent of position).
                if a == "height":
                    entry[a] = round(rng.uniform(0.3, 0.85), 3)
                elif a == "sides":
                    entry[a] = rng.randint(3, 6)
                elif a == "shade":
                    entry[a] = round(rng.uniform(0.2, 0.85), 3)
                elif a == "lines":
                    entry[a] = rng.randint(0, 4)
            # Fill in defaults for unused attributes so rendering has something.
            for a in self._ATTR_CHOICES:
                if a not in entry:
                    if a == "height":
                        entry[a] = 0.55
                    elif a == "sides":
                        entry[a] = 4
                    elif a == "shade":
                        entry[a] = 0.55
                    elif a == "lines":
                        entry[a] = 0
            fig_attrs.append(entry)

        # Compute correct ordering as labels A, B, ...
        target_sort_key = lambda i: fig_attrs[i][target_attr]
        correct_order = sorted(range(n_figs), key=target_sort_key)
        correct_labels = [chr(ord("A") + i) for i in correct_order]
        correct_text = ",".join(correct_labels)

        # Distractors: swap adjacent pairs, reverse slices, shuffle slightly.
        distractors = self._build_distractors(rng, correct_order, n_figs)
        if len(distractors) < 3:
            return None
        distractor_texts = [
            ",".join(chr(ord("A") + i) for i in d) for d in distractors[:3]
        ]
        if correct_text in distractor_texts:
            return None

        options = [correct_text] + distractor_texts
        rng.shuffle(options)
        answer_letter = chr(ord("A") + options.index(correct_text))

        # Render
        title = sub_rng.choice(self._TITLE_VARIANTS).format(
            last=chr(ord("A") + n_figs - 1))
        image = self._render(fig_attrs, options, title=title,
                             sub_rng=sub_rng)

        opts_block = "\n".join(
            f"  ({chr(ord('A') + i)}) {opt}"
            for i, opt in enumerate(options))
        sidx = (self.seed or 0) % 16
        stems = self._QUESTION_STEMS[target_attr]
        stem = stems[sidx].format(
            attr=self._ATTR_WORD[target_attr],
            last=chr(ord("A") + n_figs - 1))
        question = (
            f"{stem}\n{opts_block}\n"
            "Answer with a single letter."
        )
        return question, answer_letter, image

    def _build_distractors(self, rng, correct: List[int],
                           n: int) -> List[List[int]]:
        out = []
        seen = {tuple(correct)}
        # 1) Reverse order
        cand = list(reversed(correct))
        if tuple(cand) not in seen:
            seen.add(tuple(cand))
            out.append(cand)
        # 2) Single adjacent swap
        attempts = 0
        while len(out) < 3 and attempts < 60:
            attempts += 1
            cand = list(correct)
            i = rng.randint(0, n - 2)
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            if tuple(cand) in seen:
                continue
            seen.add(tuple(cand))
            out.append(cand)
        # 3) Broader shuffle fallback
        attempts = 0
        while len(out) < 3 and attempts < 80:
            attempts += 1
            cand = list(correct)
            rng.shuffle(cand)
            if tuple(cand) in seen:
                continue
            seen.add(tuple(cand))
            out.append(cand)
        return out

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, fig_attrs: List[Dict], options: List[str],
                title: str = "Order the figures",
                sub_rng: Optional[random.Random] = None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        rng_local = sub_rng or self._rng
        base_col = rng_local.choice(_ATTR_BASE_PALETTE)

        n = len(fig_attrs)
        fig = plt.figure(figsize=(max(9.0, 1.5 * n + 3.0) * sc, 6.2 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.4], hspace=0.20)
        ax_img = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        ax_img.set_aspect("equal")
        ax_img.axis("off")
        ax_img.set_xlim(0, n * 1.7 + 0.2)
        ax_img.set_ylim(-0.2, 2.2)
        ax_img.set_title(title, fontsize=fs + 2, fontweight="bold",
                         fontfamily=ff, pad=6)

        cell_w = 1.5
        spacing = 1.7
        for i, attrs in enumerate(fig_attrs):
            cx = (i + 0.5) * spacing
            cy = 1.0
            height = attrs["height"]
            sides = int(attrs["sides"])
            shade = float(attrs["shade"])
            lines = int(attrs["lines"])

            fill_hex = _shade_to_hex(shade, base=base_col)
            # Draw regular polygon with given number of sides scaled by height
            size = 0.42 + 0.35 * height
            orient = math.pi / 2
            p = mpatches.RegularPolygon((cx, cy), sides, radius=size,
                                        orientation=orient,
                                        facecolor=fill_hex,
                                        edgecolor="#2c3e50",
                                        linewidth=1.6, zorder=3)
            ax_img.add_patch(p)

            # Internal lines: horizontal stripes inside polygon (symbolic)
            if lines > 0:
                for k in range(lines):
                    y_line = (cy - size * 0.6) + (k + 1) * (size * 1.2) / (lines + 1)
                    ax_img.plot([cx - size * 0.55, cx + size * 0.55],
                                [y_line, y_line],
                                color="#2c3e50", linewidth=0.8, zorder=4)

            # Label
            ax_img.text(cx, -0.05, chr(ord("A") + i),
                        fontsize=fs + 2, fontweight="bold",
                        ha="center", va="top", color="#2c3e50",
                        fontfamily=ff)

        # Options panel
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 10)
        ax_txt.set_ylim(0, 6)
        ax_txt.text(0.3, 5.5, "Ordering options (least to most):",
                    fontsize=fs + 1, fontweight="bold",
                    fontfamily=ff, color="#2c3e50",
                    ha="left", va="top")
        y = 4.6
        for i, opt in enumerate(options):
            ax_txt.text(0.5, y, f"({chr(ord('A') + i)}) {opt}",
                        fontsize=fs + 1, fontfamily=ff,
                        color="#1a1a1a", ha="left", va="top")
            y -= 0.9

        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.05)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = AttributeOrderingQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed * 7 + level, parameter={"level": level})
            print(f"L{level} seed={seed} ok={ok} A={env._answer}")
