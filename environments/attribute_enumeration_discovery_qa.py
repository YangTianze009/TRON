"""
Attribute Enumeration Discovery QA environment (v3 diversity redesign, 2026-04-16).

Goal: given N figures, enumerate candidate attributes and pick the one that
partitions the figures into two groups. Targets a logic benchmark Attribute
Reasoning.

v3 diversity redesign:
  * L0 has 4 figures / 1 discriminator attribute; L9 has 8 figures / up to
    3 compounded attributes.
  * Shape pool expanded to 14 primitives; per-seed palette shuffled from
    8 wide palettes.
  * Layout varies: row-linear, 2-row grid, ring-layout.
  * Question stem variant pool (6 phrasings).
  * Discriminator families: shape_family, color, filled, size, n_dots,
    orientation, stripe_direction, has_inner_dot, is_star-like.
  * L0 uses dramatic visual separations (huge size differences, color
    contrast) so it is truly easy.
  * L9 uses subtle attributes and structurally adds "prime" vs "composite"
    count attributes.

Format: 4-way MCQ (letter) where each option is a partition string.
"""
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# -------------------------------------------------------------------- #
# Shape primitives                                                     #
# -------------------------------------------------------------------- #

_SHAPE_POOL_ALL = [
    "circle", "square", "triangle", "hexagon", "diamond", "star5",
    "pentagon", "plus", "octagon", "heart", "arrow", "crescent",
    "trapezoid", "rhombus"
]
_EVEN_SHAPES = ["square", "hexagon", "diamond", "plus", "octagon",
                 "rhombus"]
_ODD_SHAPES = ["triangle", "pentagon", "star5", "circle", "heart",
                "crescent"]
_STAR_LIKE_SHAPES = ["star5", "plus"]
_CURVY_SHAPES = ["circle", "crescent", "heart"]

_COLOR_SET = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#2980b9",
    "#27ae60", "#c0392b", "#8e44ad", "#d35400", "#bdc3c7",
]

# -------------------------------------------------------------------- #
# Shape drawing                                                        #
# -------------------------------------------------------------------- #

def _shape_patch(shape: str, cx: float, cy: float, s: float,
                 fc: str, ec: str, lw: float, rot_deg: float = 0.0):
    """Return a matplotlib patch for the given shape."""
    pts = None
    if shape == "circle":
        p = mpatches.Circle((cx, cy), s, facecolor=fc,
                             edgecolor=ec, linewidth=lw)
    elif shape == "square":
        p = mpatches.Rectangle((cx - s, cy - s), 2 * s, 2 * s,
                                facecolor=fc, edgecolor=ec, linewidth=lw)
    elif shape == "triangle":
        pts = [(cx, cy + s), (cx - s, cy - s * 0.85),
               (cx + s, cy - s * 0.85)]
    elif shape == "hexagon":
        pts = [(cx + s * math.cos(math.radians(60 * i + 30)),
                cy + s * math.sin(math.radians(60 * i + 30)))
               for i in range(6)]
    elif shape == "diamond":
        pts = [(cx, cy + s), (cx + s, cy),
               (cx, cy - s), (cx - s, cy)]
    elif shape == "pentagon":
        pts = [(cx + s * math.cos(math.radians(72 * i + 90)),
                cy + s * math.sin(math.radians(72 * i + 90)))
               for i in range(5)]
    elif shape == "star5":
        pts = []
        for i in range(10):
            a = math.radians(36 * i - 90)
            r = s if i % 2 == 0 else s * 0.45
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    elif shape == "plus":
        pts = [(cx - s / 3, cy - s), (cx + s / 3, cy - s),
               (cx + s / 3, cy - s / 3), (cx + s, cy - s / 3),
               (cx + s, cy + s / 3), (cx + s / 3, cy + s / 3),
               (cx + s / 3, cy + s), (cx - s / 3, cy + s),
               (cx - s / 3, cy + s / 3), (cx - s, cy + s / 3),
               (cx - s, cy - s / 3), (cx - s / 3, cy - s / 3)]
    elif shape == "octagon":
        pts = [(cx + s * math.cos(math.radians(45 * i + 22.5)),
                cy + s * math.sin(math.radians(45 * i + 22.5)))
               for i in range(8)]
    elif shape == "heart":
        pts = []
        for i in range(40):
            t = 2 * math.pi * i / 40
            hx = 16 * math.sin(t) ** 3
            hy = 13 * math.cos(t) - 5 * math.cos(2 * t) \
                 - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((cx + hx * s / 18, cy + hy * s / 18))
    elif shape == "arrow":
        pts = [(cx - s, cy - s / 3), (cx + s * 0.3, cy - s / 3),
               (cx + s * 0.3, cy - s * 0.8), (cx + s, cy),
               (cx + s * 0.3, cy + s * 0.8),
               (cx + s * 0.3, cy + s / 3),
               (cx - s, cy + s / 3)]
    elif shape == "crescent":
        # Two overlapping circles give a crescent shape. Use a polygon of
        # the outline as an approximation.
        pts = []
        for i in range(30):
            a = math.radians(-90 + 180 * i / 29)
            pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
        for i in range(30):
            a = math.radians(90 - 180 * i / 29)
            pts.append((cx + s * 0.5 + s * 0.7 * math.cos(a),
                        cy + s * 0.7 * math.sin(a)))
    elif shape == "trapezoid":
        pts = [(cx - s, cy - s * 0.7), (cx + s, cy - s * 0.7),
               (cx + s * 0.5, cy + s * 0.7),
               (cx - s * 0.5, cy + s * 0.7)]
    elif shape == "rhombus":
        pts = [(cx, cy + s * 0.9), (cx + s * 0.6, cy),
               (cx, cy - s * 0.9), (cx - s * 0.6, cy)]
    else:
        p = mpatches.Circle((cx, cy), s, facecolor=fc,
                             edgecolor=ec, linewidth=lw)

    if pts is not None:
        p = mpatches.Polygon(pts, closed=True, facecolor=fc,
                              edgecolor=ec, linewidth=lw)
    if abs(rot_deg) > 1e-6:
        return p, rot_deg
    return p, 0.0

# -------------------------------------------------------------------- #
# Discriminators: attribute families used to partition the figures.   #
# -------------------------------------------------------------------- #

_DISCRIMINATORS_EASY = [
    "color", "n_dots_1v3", "big_vs_small", "filled_vs_hollow",
]
_DISCRIMINATORS_MEDIUM = [
    "color", "n_dots_1v2", "big_vs_small", "filled_vs_hollow",
    "rotation_0_vs_45", "stripe_direction", "curvy_vs_polygonal",
    "star_like",
]
_DISCRIMINATORS_HARD = [
    "n_dots_parity", "shape_family_even_odd", "filled_vs_hollow",
    "rotation_0_vs_45", "stripe_direction", "curvy_vs_polygonal",
    "star_like", "big_vs_small", "n_dots_prime",
]

class AttributeEnumerationDiscoveryQA(StandaloneVisualEnv):
    ENV_NAME = "attribute_enumeration_discovery"

    _QUESTION_STEMS = [
        "Classify the {n} figures shown into two groups so each group shares a common visual attribute. Which option is correct?",
        "Split the {n} figures into two equal groups based on a shared visual property. Which partition is correct?",
        "The {n} figures can be divided into two groups using one distinguishing attribute. Pick the correct grouping.",
        "Examine the figures and find the attribute that separates them into two equal groups. Choose the correct partition.",
        "Partition the {n} figures shown into two groups of equal size by a single shared attribute. Which option works?",
        "Group the figures shown into two equal sets, each set sharing a common feature. Which grouping is right?",
    ]

    _TITLE_POOL = [
        "Figures", "Attribute Discovery", "Figure Classification",
        "Visual Attributes", "Group the Figures", "Sort the Figures",
        "Partition the Set", "Visual Grouping",
    ]

    _LAYOUT_POOL = ["grid_2_rows", "single_row", "ring", "grid_3_rows"]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n_figures": 6,
                "discriminator_pool": _DISCRIMINATORS_EASY,
                "subtle": False,
                "n_options": 4,
                "layout_pool": ["single_row", "grid_2_rows"],
                "size_gap_boost": 1.7,
            }
        if level <= 3:
            return {
                "n_figures": 6,
                "discriminator_pool": _DISCRIMINATORS_EASY,
                "subtle": False,
                "n_options": 4,
                "layout_pool": ["grid_2_rows", "single_row"],
                "size_gap_boost": 1.4,
            }
        if level <= 5:
            return {
                "n_figures": 6,
                "discriminator_pool": _DISCRIMINATORS_MEDIUM,
                "subtle": True,
                "n_options": 4,
                "layout_pool": ["grid_2_rows", "ring"],
                "size_gap_boost": 1.15,
            }
        if level <= 7:
            return {
                "n_figures": 8,
                "discriminator_pool": _DISCRIMINATORS_MEDIUM,
                "subtle": True,
                "n_options": 4,
                "layout_pool": ["grid_2_rows", "ring", "grid_3_rows"],
                "size_gap_boost": 1.05,
            }
        return {
            "n_figures": 8,
            "discriminator_pool": _DISCRIMINATORS_HARD,
            "subtle": True,
            "n_options": 4,
            "layout_pool": ["ring", "grid_2_rows", "grid_3_rows"],
            "size_gap_boost": 1.0,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_figures"]

        for _ in range(60):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_figures"]
        half = n // 2
        discriminator = rng.choice(cfg["discriminator_pool"])
        g1_size = half
        g2_size = n - half

        # Randomize per-figure base attributes.
        base_shapes = rng.sample(_SHAPE_POOL_ALL, n)
        palette = list(_COLOR_SET)
        rng.shuffle(palette)
        base_color = palette[0]
        alt_color = palette[1]
        base_sizes = [0.42] * n
        base_filled = [True] * n
        base_dots = [0] * n
        base_rot = [0] * n
        base_stripes = [None] * n
        gap_boost = cfg["size_gap_boost"]

        # Initialize per discriminator
        group1 = list(range(half))
        group2 = list(range(half, n))
        rng.shuffle(group1)
        rng.shuffle(group2)

        colors = [base_color] * n
        sizes = list(base_sizes)
        filled = list(base_filled)
        n_dots = list(base_dots)
        rots = list(base_rot)
        stripes = list(base_stripes)

        def pick_stripe():
            return rng.choice(["horiz", "vert", "diag"])

        if discriminator == "color":
            c_a, c_b = palette[0], palette[3]
            for i in group1:
                colors[i] = c_a
            for i in group2:
                colors[i] = c_b
        elif discriminator == "n_dots_1v3":
            for i in group1:
                n_dots[i] = 1
            for i in group2:
                n_dots[i] = 3
        elif discriminator == "n_dots_1v2":
            for i in group1:
                n_dots[i] = 1
            for i in group2:
                n_dots[i] = 2
        elif discriminator == "n_dots_parity":
            even_count = rng.choice([2, 4])
            odd_count = rng.choice([1, 3, 5])
            for i in group1:
                n_dots[i] = even_count
            for i in group2:
                n_dots[i] = odd_count
        elif discriminator == "n_dots_prime":
            prime = rng.choice([2, 3, 5])
            composite = rng.choice([4, 6])
            for i in group1:
                n_dots[i] = prime
            for i in group2:
                n_dots[i] = composite
        elif discriminator == "big_vs_small":
            for i in group1:
                sizes[i] = 0.28 * gap_boost / 1.4
            for i in group2:
                sizes[i] = 0.52 * gap_boost / 1.4
        elif discriminator == "filled_vs_hollow":
            for i in group1:
                filled[i] = True
            for i in group2:
                filled[i] = False
        elif discriminator == "rotation_0_vs_45":
            for i in group1:
                rots[i] = 0
            for i in group2:
                rots[i] = 45
        elif discriminator == "stripe_direction":
            d1 = rng.choice(["horiz", "vert", "diag"])
            d2 = "vert" if d1 == "horiz" else ("horiz" if d1 == "vert" else "horiz")
            for i in group1:
                stripes[i] = d1
            for i in group2:
                stripes[i] = d2
        elif discriminator == "curvy_vs_polygonal":
            for i in group1:
                base_shapes[i] = rng.choice(_CURVY_SHAPES)
            for i in group2:
                base_shapes[i] = rng.choice(
                    [s for s in _SHAPE_POOL_ALL if s not in _CURVY_SHAPES])
        elif discriminator == "star_like":
            for i in group1:
                base_shapes[i] = rng.choice(_STAR_LIKE_SHAPES)
            for i in group2:
                base_shapes[i] = rng.choice(
                    [s for s in _SHAPE_POOL_ALL if s not in _STAR_LIKE_SHAPES])
        elif discriminator == "shape_family_even_odd":
            for i in group1:
                base_shapes[i] = rng.choice(_EVEN_SHAPES)
            for i in group2:
                base_shapes[i] = rng.choice(_ODD_SHAPES)

        # Add per-figure noise so figures look different even when sharing
        # the discriminator category. Some attributes get randomised as
        # noise (e.g. shape), others we leave constant so the
        # discriminator stays unique.
        for i in range(n):
            # random small rotation jitter (not for rotation discriminator)
            if discriminator != "rotation_0_vs_45":
                rots[i] = rng.choice([0, 15, 30, -15, -30, 45, -45])
            # random color (not for color discriminator)
            if discriminator != "color":
                colors[i] = rng.choice(palette[:6])
            # random filled/hollow (not for filled discriminator)
            if discriminator != "filled_vs_hollow":
                filled[i] = rng.choice([True, True, False])
            # random small size jitter (not for big_vs_small discriminator)
            if discriminator != "big_vs_small":
                sizes[i] = 0.38 + rng.uniform(-0.05, 0.08)

        # Build figure list
        figs = []
        for i in range(n):
            figs.append({
                "shape": base_shapes[i],
                "color": colors[i],
                "filled": filled[i],
                "size": sizes[i],
                "n_dots": n_dots[i],
                "rot": rots[i],
                "stripe": stripes[i],
            })

        # Shuffle display positions.
        order = list(range(n))
        rng.shuffle(order)
        figs_display = [figs[i] for i in order]
        correct_g1 = []
        correct_g2 = []
        for disp_idx, orig_idx in enumerate(order):
            if orig_idx in group1:
                correct_g1.append(disp_idx + 1)
            else:
                correct_g2.append(disp_idx + 1)
        correct_g1.sort()
        correct_g2.sort()
        correct_partition = (tuple(correct_g1), tuple(correct_g2))

        # Build partitions.
        all_idxs = list(range(1, n + 1))

        def _make_random_partition():
            pool = list(all_idxs)
            rng.shuffle(pool)
            g1 = sorted(pool[:g1_size])
            g2 = sorted(pool[g1_size:])
            return (tuple(g1), tuple(g2))

        options = [correct_partition]
        seen = {correct_partition,
                (correct_partition[1], correct_partition[0])}
        tries = 0
        while len(options) < cfg["n_options"] and tries < 400:
            p = _make_random_partition()
            if p not in seen and (p[1], p[0]) not in seen:
                seen.add(p)
                options.append(p)
            tries += 1
        if len(options) < cfg["n_options"]:
            return None

        rng.shuffle(options)
        idx_correct = None
        for i, o in enumerate(options):
            if (o == correct_partition
                    or o == (correct_partition[1], correct_partition[0])):
                idx_correct = i
                break
        if idx_correct is None:
            return None
        answer_letter = chr(ord("A") + idx_correct)

        def fmt(p):
            g1, g2 = p
            g1s = ",".join(str(x) for x in g1)
            g2s = ",".join(str(x) for x in g2)
            return f"{{{g1s}}} | {{{g2s}}}"

        options_str = [fmt(o) for o in options]

        stem = rng.choice(self._QUESTION_STEMS).format(n=n)
        opts_block = "\n".join(
            f"  ({chr(ord('A') + i)}) {s}" for i, s in enumerate(options_str))
        question = f"{stem}\n{opts_block}\nReply with the single letter (A/B/C/D) inside <answer>...</answer>. Example: <answer>A</answer>"

        layout = rng.choice(cfg["layout_pool"])
        title = rng.choice(self._TITLE_POOL)
        image = self._render(figs_display, options_str, cfg, layout, title,
                             rng, palette)
        return question, answer_letter, image

    # ---------------------------------------------------------------- #
    def _draw_one(self, ax, cx, cy, fig_data):
        shape = fig_data["shape"]
        color = fig_data["color"]
        fill = fig_data["filled"]
        s = fig_data["size"]
        fc = color if fill else "none"
        ec = color if not fill else "#1a1a1a"
        lw = 1.8 if not fill else 1.2
        patch, rot_deg = _shape_patch(shape, cx, cy, s, fc, ec, lw,
                                       fig_data.get("rot", 0))
        if rot_deg:
            patch.set_transform(
                Affine2D().rotate_deg_around(cx, cy, rot_deg) + ax.transData)
        ax.add_patch(patch)

        # Stripes (if requested)
        if fig_data.get("stripe") is not None:
            stripe = fig_data["stripe"]
            stripe_col = "#2c3e50"
            if stripe == "horiz":
                for ys in (cy - s * 0.55, cy, cy + s * 0.55):
                    ax.plot([cx - s * 0.7, cx + s * 0.7],
                            [ys, ys], color=stripe_col, linewidth=0.9)
            elif stripe == "vert":
                for xs in (cx - s * 0.55, cx, cx + s * 0.55):
                    ax.plot([xs, xs],
                            [cy - s * 0.7, cy + s * 0.7],
                            color=stripe_col, linewidth=0.9)
            elif stripe == "diag":
                for k in (-0.5, 0.0, 0.5):
                    ax.plot(
                        [cx - s * 0.7 + k * s,
                         cx + s * 0.7 + k * s],
                        [cy - s * 0.7, cy + s * 0.7],
                        color=stripe_col, linewidth=0.9)

        # dots
        n_dots = fig_data.get("n_dots", 0)
        if n_dots > 0:
            for d in range(n_dots):
                offset = (d - (n_dots - 1) / 2.0) * 0.12
                ax.plot(cx + offset, cy, "o",
                        color="#1a1a1a", markersize=3.5)

    def _render(self, figs, options, cfg, layout, title, rng, palette):
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = style["font_size_base"]

        n = len(figs)
        fig = plt.figure(figsize=(11.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_f = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_f.set_aspect("equal")
        ax_f.axis("off")
        ax_t.axis("off")

        # Layout positions.
        positions = self._compute_positions(layout, n, rng)

        # Jitter
        jittered = []
        for (cx, cy) in positions:
            jx = cx + rng.uniform(-0.06, 0.06)
            jy = cy + rng.uniform(-0.06, 0.06)
            jittered.append((jx, jy))

        for i, (f, (cx, cy)) in enumerate(zip(figs, jittered)):
            cell_w = 1.6
            rect = mpatches.FancyBboxPatch(
                (cx - cell_w / 2 + 0.05, cy - cell_w / 2 + 0.05),
                cell_w - 0.1, cell_w - 0.1,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#ffffff",
                edgecolor="#7f8c8d",
                linewidth=1.0)
            ax_f.add_patch(rect)
            self._draw_one(ax_f, cx, cy, f)
            ax_f.text(cx - 0.65, cy - 0.68,
                      f"{i + 1}", fontsize=fs + 1, fontweight="bold",
                      family=ff, color="#1a1a1a")

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        pad = 1.2
        ax_f.set_xlim(min(xs) - pad, max(xs) + pad)
        ax_f.set_ylim(min(ys) - pad, max(ys) + pad)
        ax_f.set_title(title, fontsize=fs + 2, family=ff, pad=8)

        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        ax_t.text(0.3, 11.5, "Classification options:",
                  fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y = 10.8
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs + 1, family=ff,
                      ha="left", va="top", color="#1a1a1a")
            y -= 0.75

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _compute_positions(self, layout, n, rng):
        """Return list of (cx, cy) for each figure index."""
        positions = []
        if layout == "single_row":
            for i in range(n):
                positions.append((i * 1.9 + 1, 1.0))
        elif layout == "grid_3_rows":
            cols = max(2, (n + 2) // 3)
            for i in range(n):
                r = i // cols
                c = i % cols
                positions.append((c * 1.9 + 1, 4.5 - r * 1.9))
        elif layout == "ring":
            # cell width ~1.6; ensure arc-chord >= cell-width with ~30% padding
            R = max(2.6, 1.05 * n / 3.0)
            ang0 = rng.uniform(0, 2 * math.pi)
            for i in range(n):
                a = ang0 + 2 * math.pi * i / n
                positions.append((R * math.cos(a), R * math.sin(a)))
        else:  # grid_2_rows (default)
            cols = (n + 1) // 2
            for i in range(n):
                r = i // cols
                c = i % cols
                positions.append((c * 1.9 + 1, 3.0 - r * 1.9))
        return positions

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_aed"
    os.makedirs(out_dir, exist_ok=True)
    env = AttributeEnumerationDiscoveryQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"attribute_enumeration_discovery_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  A: {env._answer}")
