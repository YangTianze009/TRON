"""
Quantifier Logic QA environment (v3, diversity redesign 2026-04-16).

Goal: train reasoning with universal/existential/negative quantifiers using
visual set diagrams. Targets reference / deductive.

v3 diversity redesign:
  * 4 distinct visual diagram families picked per seed:
      - Venn circle layout (2-4 circles)
      - Bubble-cluster (free-form circles arranged by containment)
      - Matrix/grid showing memberships
      - Eulerian set outline (trimmed curves)
  * Category pool expanded to 50 diverse nouns.
  * Question template pool (8 phrasings) + title variants (10).
  * Sub-RNG per level, per-seed palette shuffle.
  * L0: 2-set Venn with 1 shaded + 1 empty region, clean statement.
  * L9: 4-set Eulerian or matrix with multiple shadings and tight
    near-miss distractor quantifiers.
  * Numeric indices / markers randomised per seed.

Format is 4-way MCQ at every level.
"""
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_CATEGORY_POOL = [
    "Animals", "Mammals", "Pets", "Birds", "Fish", "Reptiles", "Insects",
    "Amphibians", "Plants", "Trees", "Flowers", "Fruits", "Vegetables",
    "Berries", "Grains", "Vehicles", "Cars", "Trucks", "Boats", "Planes",
    "Bikes", "Trains", "Students", "Athletes", "Artists", "Doctors",
    "Teachers", "Engineers", "Writers", "Chefs", "Metals", "Liquids",
    "Solids", "Gases", "Crystals", "Cities", "Countries", "Rivers",
    "Mountains", "Islands", "Stars", "Planets", "Galaxies", "Moons",
    "Books", "Movies", "Games", "Songs", "Poems", "Paintings",
]

_TITLE_POOL = [
    "Set Diagram", "Venn Diagram", "Set Relations", "Quantifier Logic",
    "Sets & Membership", "Logical Sets", "Category Diagram",
    "Membership Diagram", "Set Overlap Diagram", "Visual Logic",
]

_QUESTION_TEMPLATES = [
    "Based on the diagram (shaded regions = non-empty sets; X-marked regions = empty sets), which statement must be TRUE? Answer A, B, C, or D.",
    "The diagram shows set relations. Shaded = non-empty; X = empty. Which conclusion follows necessarily? Answer with a single letter.",
    "Study the diagram. Shaded areas are known to be non-empty; crossed areas are known to be empty. Which statement is logically guaranteed?",
    "Given the shaded and crossed regions in the diagram, which option must hold? Pick a single letter.",
    "Using only what the diagram tells us (shading = non-empty, X = empty), which claim is necessarily true? Answer with the letter.",
    "From the marked regions in the set diagram, which of the statements below is guaranteed? Answer with a single letter A, B, C, or D.",
    "Inspect the diagram. Which of the following statements logically follows from the shaded/empty markings?",
    "The set diagram marks certain regions as non-empty (shaded) or empty (X). Choose the statement that must be true.",
]

# Region utilities
def _all_regions(n_sets: int) -> List[frozenset]:
    out = []
    for mask in range(1, 1 << n_sets):
        s = frozenset(i for i in range(n_sets) if mask & (1 << i))
        out.append(s)
    return out

class QuantifierLogicQA(StandaloneVisualEnv):
    ENV_NAME = "quantifier_logic"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n_sets": 2,
                "n_shaded": 1,
                "n_empty": 1,
                "diagram_styles": ["venn"],
                "tight_distractor": False,
            }
        if level <= 3:
            return {
                "n_sets": 2,
                "n_shaded": 1,
                "n_empty": 1,
                "diagram_styles": ["venn", "bubble"],
                "tight_distractor": False,
            }
        if level <= 5:
            return {
                "n_sets": 3,
                "n_shaded": 1,
                "n_empty": 2,
                "diagram_styles": ["venn", "bubble", "grid"],
                "tight_distractor": True,
            }
        if level <= 7:
            return {
                "n_sets": 3,
                "n_shaded": 2,
                "n_empty": 2,
                "diagram_styles": ["venn", "grid", "euler"],
                "tight_distractor": True,
            }
        return {
            "n_sets": 4,
            "n_shaded": 2,
            "n_empty": 3,
            "diagram_styles": ["grid", "venn", "euler"],
            "tight_distractor": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)

        rng = random.Random((self.seed or 0) * 31 + level * 17 + 5)
        self._primary_complexity_feature = cfg["n_sets"] * 2 + cfg["n_shaded"]

        for _ in range(40):
            result = self._try_generate(rng, level, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, level, cfg):
        n_sets = cfg["n_sets"]
        labels = rng.sample(_CATEGORY_POOL, n_sets)
        regions = _all_regions(n_sets)

        max_marks = len(regions) - 1
        n_shaded = min(cfg["n_shaded"], max_marks)
        n_empty = min(cfg["n_empty"], max_marks)
        if n_shaded + n_empty > len(regions):
            return None

        shuffled = list(regions)
        rng.shuffle(shuffled)
        shaded_regions = set(shuffled[:n_shaded])
        empty_regions = set(shuffled[n_shaded:n_shaded + n_empty])

        def eval_statement(t, x, y):
            x_not_y = [r for r in regions if x in r and y not in r]
            x_and_y = [r for r in regions if x in r and y in r]
            if t == "all":
                if any(r in shaded_regions for r in x_not_y):
                    return False
                if x_not_y and all(r in empty_regions for r in x_not_y) \
                        and any(r in shaded_regions for r in x_and_y):
                    return True
                return None
            if t == "some":
                if any(r in shaded_regions for r in x_and_y):
                    return True
                if x_and_y and all(r in empty_regions for r in x_and_y):
                    return False
                return None
            if t == "no":
                if any(r in shaded_regions for r in x_and_y):
                    return False
                if x_and_y and all(r in empty_regions for r in x_and_y):
                    return True
                return None
            if t == "not_all":
                if any(r in shaded_regions for r in x_not_y):
                    return True
                if x_not_y and all(r in empty_regions for r in x_not_y):
                    return False
                return None
            if t == "some_not":
                return eval_statement("not_all", x, y)
            return None

        trues, falses = [], []
        for t in ["all", "some", "no", "not_all", "some_not"]:
            for x in range(n_sets):
                for y in range(n_sets):
                    if x == y:
                        continue
                    v = eval_statement(t, x, y)
                    if v is True:
                        trues.append((t, x, y))
                    elif v is False:
                        falses.append((t, x, y))

        if not trues or len(falses) < 3:
            return None

        correct = rng.choice(trues)
        if cfg["tight_distractor"]:
            # Prefer distractors that share an endpoint AND use a near-miss
            # quantifier.
            near_miss = {
                "all": ["no", "not_all", "some"],
                "some": ["no", "not_all", "all"],
                "no": ["some", "all", "not_all"],
                "not_all": ["all", "no", "some"],
                "some_not": ["all", "no", "some"],
            }.get(correct[0], [])
            pool = [c for c in falses
                    if (c[1] == correct[1] or c[2] == correct[2])
                    and c[0] in near_miss]
            if len(pool) < 3:
                pool = [c for c in falses
                        if c[1] == correct[1] or c[2] == correct[2]]
            if len(pool) < 3:
                pool = list(falses)
        else:
            pool = [c for c in falses
                    if c[1] == correct[1] or c[2] == correct[2]]
            if len(pool) < 3:
                pool = list(falses)

        rng.shuffle(pool)
        distractors = pool[:3]
        if len(distractors) < 3:
            return None

        options = [correct] + distractors
        rng.shuffle(options)
        answer = chr(ord("A") + options.index(correct))

        def humanise(t, x, y):
            X = labels[x]
            Y = labels[y]
            if t == "all":
                return f"All {X} are {Y}."
            if t == "some":
                return f"Some {X} are {Y}."
            if t == "no":
                return f"No {X} are {Y}."
            if t == "not_all":
                return f"Not all {X} are {Y}."
            if t == "some_not":
                return f"Some {X} are not {Y}."
            return ""

        option_strs = [humanise(*o) for o in options]

        style_choice = rng.choice(cfg["diagram_styles"])
        title = rng.choice(_TITLE_POOL)
        image = self._render(n_sets, labels, shaded_regions, empty_regions,
                             option_strs, title=title, style=style_choice,
                             rng=rng)
        question = rng.choice(_QUESTION_TEMPLATES)
        return question, answer, image

    # -------------------------------------------------- #
    def _render(self, n_sets, labels, shaded_regions, empty_regions,
                option_strs, title, style, rng) -> Image.Image:
        vstyle = self._random_style()
        sc = vstyle["figsize_scale"]
        ff = rng.choice(["serif", "DejaVu Sans", "sans-serif"])
        base_fs = rng.choice([11, 12, 13])

        fig = plt.figure(figsize=(12.0 * sc, 7.8 * sc))
        fig.patch.set_facecolor(vstyle["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.1], wspace=0.15)
        ax_d = fig.add_subplot(gs[0])
        ax_o = fig.add_subplot(gs[1])

        ax_d.set_aspect("equal")
        ax_d.axis("off")
        ax_d.set_title(title, fontsize=base_fs + 2, fontweight="bold",
                       family=ff)

        palette = list(vstyle["palette"])
        rng.shuffle(palette)

        # Dispatch to diagram style. For n_sets=4, bubble/euler/venn
        # geometries cannot reliably surface every region; fall back to
        # grid (which guarantees each shaded/empty region is shown).
        if n_sets >= 4:
            self._render_grid(ax_d, n_sets, labels, shaded_regions,
                              empty_regions, palette, base_fs, ff, rng)
        elif style == "venn":
            self._render_venn(ax_d, n_sets, labels, shaded_regions,
                              empty_regions, palette, base_fs, ff, rng)
        elif style == "bubble":
            self._render_bubble(ax_d, n_sets, labels, shaded_regions,
                                empty_regions, palette, base_fs, ff, rng)
        elif style == "grid":
            self._render_grid(ax_d, n_sets, labels, shaded_regions,
                              empty_regions, palette, base_fs, ff, rng)
        elif style == "euler":
            self._render_euler(ax_d, n_sets, labels, shaded_regions,
                                empty_regions, palette, base_fs, ff, rng)
        else:
            self._render_venn(ax_d, n_sets, labels, shaded_regions,
                              empty_regions, palette, base_fs, ff, rng)

        # Options panel
        ax_o.set_xlim(0, 10)
        ax_o.set_ylim(0, 10)
        ax_o.axis("off")
        stem_pool = [
            "Which statement must be TRUE?",
            "Which option is logically true?",
            "Pick the true conclusion:",
            "True statement?",
        ]
        ax_o.set_title(rng.choice(stem_pool),
                       fontsize=base_fs + 2, fontweight="bold", family=ff)
        y = 8.6
        dy = 1.35
        for i, opt in enumerate(option_strs):
            letter = chr(ord("A") + i)
            ax_o.text(0.4, y, f"({letter}) {opt}",
                      fontsize=base_fs + 1, ha="left", va="top",
                      family=ff, color="#1a1a1a")
            y -= dy

        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=vstyle["dpi"])

    # ------------------ Render helpers --------------------- #

    def _mark_region(self, ax, rx, ry, is_shaded, is_empty):
        if is_shaded:
            for dxy in [(-0.1, 0.03), (0.1, 0.03), (0, -0.08),
                        (-0.15, -0.06), (0.15, -0.06)]:
                dot = mpatches.Circle(
                    (rx + dxy[0], ry + dxy[1]), 0.055,
                    facecolor="#1a1a1a", zorder=10)
                ax.add_patch(dot)
        elif is_empty:
            ax.plot([rx - 0.15, rx + 0.15], [ry - 0.15, ry + 0.15],
                    color="#c0392b", linewidth=2.6, zorder=10)
            ax.plot([rx - 0.15, rx + 0.15], [ry + 0.15, ry - 0.15],
                    color="#c0392b", linewidth=2.6, zorder=10)

    def _add_legend(self, ax, base_fs, ff, cx=0.0, cy=-4.6):
        ax.text(cx, cy, "Legend:", fontsize=base_fs - 1,
                fontweight="bold", family=ff, color="#34495e",
                ha="center", va="center")
        for dxy in [(-2.4, cy - 0.6), (-2.28, cy - 0.6),
                    (-2.34, cy - 0.75)]:
            ax.add_patch(mpatches.Circle(dxy, 0.05,
                                          facecolor="#1a1a1a"))
        ax.text(-2.18, cy - 0.6, "= non-empty",
                fontsize=base_fs - 2, family=ff, va="center",
                ha="left")
        ax.plot([0.80, 1.05], [cy - 0.72, cy - 0.48],
                color="#c0392b", linewidth=2.0)
        ax.plot([0.80, 1.05], [cy - 0.48, cy - 0.72],
                color="#c0392b", linewidth=2.0)
        ax.text(1.18, cy - 0.6, "= empty",
                fontsize=base_fs - 2, family=ff, va="center",
                ha="left")

    def _render_venn(self, ax, n_sets, labels, shaded, empty,
                     palette, base_fs, ff, rng):
        ax.set_xlim(-4.4, 4.4)
        ax.set_ylim(-5.6, 4.3)
        if n_sets == 2:
            jitter = rng.uniform(-0.15, 0.15)
            circles = [((-0.9 + jitter, 0.0), 1.7),
                       ((0.9 - jitter, 0.0), 1.7)]
            label_pos = [(-1.8, 2.2), (1.8, 2.2)]
            region_centers = {
                frozenset({0}): (-1.85, 0.0),
                frozenset({1}): (1.85, 0.0),
                frozenset({0, 1}): (0.0, 0.0),
            }
        elif n_sets == 3:
            jitter = rng.uniform(-0.1, 0.1)
            circles = [((-0.95 + jitter, 0.55), 1.7),
                       ((0.95 - jitter, 0.55), 1.7),
                       ((0.0, -0.95), 1.7)]
            label_pos = [(-2.2, 2.55), (2.2, 2.55), (0.0, -3.0)]
            region_centers = {
                frozenset({0}):         (-2.0, 1.15),
                frozenset({1}):         (2.0, 1.15),
                frozenset({2}):         (0.0, -2.05),
                frozenset({0, 1}):      (0.0, 1.3),
                frozenset({0, 2}):      (-1.25, -0.65),
                frozenset({1, 2}):      (1.25, -0.65),
                frozenset({0, 1, 2}):   (0.0, -0.05),
            }
        else:
            return self._render_grid(ax, n_sets, labels, shaded, empty,
                                      palette, base_fs, ff, rng)

        for i, ((cx, cy), r) in enumerate(circles):
            circ = mpatches.Circle(
                (cx, cy), r,
                facecolor=palette[i % len(palette)], edgecolor="#2c3e50",
                linewidth=1.8, alpha=0.22, zorder=3)
            ax.add_patch(circ)
        for i, (lx, ly) in enumerate(label_pos):
            ax.text(lx, ly, labels[i],
                    fontsize=base_fs + 1, fontweight="bold",
                    ha="center", va="center", family=ff,
                    color="#2c3e50", zorder=5)

        for region, (rx, ry) in region_centers.items():
            self._mark_region(ax, rx, ry,
                              region in shaded, region in empty)
        self._add_legend(ax, base_fs, ff)

    def _render_bubble(self, ax, n_sets, labels, shaded, empty,
                       palette, base_fs, ff, rng):
        """Alternative visual: free-positioned circles with explicit
        region markers placed via a grid."""
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-5.5, 4.5)
        if n_sets == 2:
            centers = [(-1.2, 0.4), (1.0, -0.1)]
            radii = [1.7, 1.9]
        elif n_sets == 3:
            centers = [(-1.5, 0.9), (1.3, 1.2), (0.1, -1.3)]
            radii = [1.7, 1.6, 1.7]
        else:
            centers = [(-1.6, 1.0), (1.2, 1.3), (-1.0, -1.1), (1.3, -0.8)]
            radii = [1.55] * 4

        for i, ((cx, cy), r) in enumerate(zip(centers, radii)):
            ax.add_patch(mpatches.Circle(
                (cx, cy), r,
                facecolor=palette[i % len(palette)], edgecolor="#2c3e50",
                linewidth=1.6, alpha=0.22, zorder=3))
            ax.text(cx + r * 0.9, cy + r * 0.65, labels[i],
                    fontsize=base_fs + 1, fontweight="bold",
                    family=ff, color="#2c3e50", zorder=5)

        # Compute grid-based region centres.
        region_centers = {}
        for xi in range(-30, 31, 3):
            for yi in range(-30, 31, 3):
                px, py = xi * 0.15, yi * 0.15
                mem = frozenset(
                    i for i, ((cx, cy), r) in enumerate(zip(centers, radii))
                    if (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2
                )
                if not mem:
                    continue
                if mem not in region_centers:
                    region_centers[mem] = []
                region_centers[mem].append((px, py))
        for mem, pts in region_centers.items():
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            self._mark_region(ax, cx, cy,
                              mem in shaded, mem in empty)
        self._add_legend(ax, base_fs, ff, cy=-4.7)

    def _render_grid(self, ax, n_sets, labels, shaded, empty,
                     palette, base_fs, ff, rng):
        """Matrix-style: rows = regions, columns = sets; shading markers
        in the rightmost cell."""
        regions = _all_regions(n_sets)

        # CRITICAL: make sure every shaded/empty region is actually shown
        # in the grid (otherwise the problem is unanswerable from the image).
        rngs = list(regions)
        rng.shuffle(rngs)
        must_include = list(shaded) + list(empty)
        for r in must_include:
            if r in rngs:
                rngs.remove(r)
        rngs = must_include + rngs
        rows = min(len(rngs), max(9, len(must_include) + 4))

        # Adjust row spacing so larger region counts fit the axis.
        dy = 0.9 if rows <= 9 else 8.1 / rows
        ax.set_xlim(-1, 7)
        ax.set_ylim(-0.5, 10)
        # Header. Column spacing scales with label length so 8-char labels
        # like "Vehicles" don't crash into each other (previously 0.85
        # was too tight — "VehiclesIslandsRivers" squished).
        max_label_len = max((len(labels[i][:8]) for i in range(n_sets)), default=4)
        col_step = max(1.2, 0.18 * max_label_len)  # ≥ 1.2 always
        ax.text(0, 9.3, "Region", fontsize=base_fs, fontweight="bold",
                family=ff)
        for i in range(n_sets):
            ax.text(1.4 + i * col_step, 9.3, labels[i][:8],
                    fontsize=base_fs, fontweight="bold", family=ff)
        ax.text(1.4 + n_sets * col_step + 0.3, 9.3, "Status",
                fontsize=base_fs, fontweight="bold", family=ff)

        for row, region in enumerate(rngs[:rows]):
            y = 8.4 - row * dy
            rect = mpatches.Rectangle(
                (-0.5, y - 0.3), 6.5, 0.75,
                facecolor=("#f5f5f5" if row % 2 == 0 else "#ffffff"),
                edgecolor="none", zorder=1)
            ax.add_patch(rect)
            ax.text(0, y, "R" + str(row + 1),
                    fontsize=base_fs - 1, family=ff, color="#2c3e50")
            for i in range(n_sets):
                cx = 1.4 + i * col_step
                if i in region:
                    ax.add_patch(mpatches.Circle(
                        (cx, y + 0.1), 0.13,
                        facecolor=palette[i % len(palette)],
                        edgecolor="#2c3e50", linewidth=1.0, zorder=4))
                else:
                    ax.plot([cx - 0.12, cx + 0.12], [y - 0.02, y + 0.22],
                            color="#95a5a6", linewidth=1.1, zorder=4)
            stat_x = 1.4 + n_sets * col_step + 0.3
            if region in shaded:
                self._mark_region(ax, stat_x, y + 0.1, True, False)
            elif region in empty:
                self._mark_region(ax, stat_x, y + 0.1, False, True)
            else:
                ax.text(stat_x, y + 0.1, "?",
                        fontsize=base_fs + 2, family=ff, ha="center",
                        va="center", color="#7f8c8d")

        # Legend
        ax.text(-0.5, -0.2, "● = present in set; / = not present;"
                " dots = region non-empty; X = region empty",
                fontsize=base_fs - 2, family=ff, color="#34495e")

    def _render_euler(self, ax, n_sets, labels, shaded, empty,
                      palette, base_fs, ff, rng):
        """Euler-style: draw outlined shapes that may be nested or
        disjoint; shading markers per region."""
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-5.5, 4.5)
        # Place "containers" from larger to smaller, mixing nesting and
        # overlap. This is a loose approximation; we still mark region
        # centres the same way.
        if n_sets == 2:
            centers = [(-1.3, 0.4), (0.9, -0.2)]
            radii = [2.1, 1.6]
        elif n_sets == 3:
            centers = [(-1.7, 1.0), (1.5, 1.1), (-0.1, -1.3)]
            radii = [1.9, 1.6, 1.7]
        else:
            centers = [(-1.8, 1.0), (1.5, 1.1), (-1.0, -1.1), (1.2, -1.0)]
            radii = [1.55, 1.45, 1.5, 1.4]

        # Draw outlines (no fill) to mimic Euler-style.
        for i, ((cx, cy), r) in enumerate(zip(centers, radii)):
            ax.add_patch(mpatches.Circle(
                (cx, cy), r, facecolor="none",
                edgecolor=palette[i % len(palette)], linewidth=2.4,
                zorder=3))
            ax.text(cx + r * 0.95, cy + r * 0.65, labels[i],
                    fontsize=base_fs + 1, fontweight="bold",
                    family=ff, color=palette[i % len(palette)], zorder=5)

        region_centers = {}
        for xi in range(-30, 31, 3):
            for yi in range(-30, 31, 3):
                px, py = xi * 0.15, yi * 0.15
                mem = frozenset(
                    i for i, ((cx, cy), r) in enumerate(zip(centers, radii))
                    if (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2
                )
                if not mem:
                    continue
                if mem not in region_centers:
                    region_centers[mem] = []
                region_centers[mem].append((px, py))
        for mem, pts in region_centers.items():
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            self._mark_region(ax, cx, cy,
                              mem in shaded, mem in empty)
        self._add_legend(ax, base_fs, ff, cy=-4.7)

if __name__ == "__main__":
    import os, collections
    out_dir = "/tmp/env_check_ql"
    os.makedirs(out_dir, exist_ok=True)
    env = QuantifierLogicQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED to generate")
                continue
            img = env.render()
            path = os.path.join(
                out_dir, f"quantifier_logic_seed{seed}_L{level}.png")
            img.save(path)
            print(f"[seed={seed} L{level}] A={env._answer}")

    for level in (0, 3, 6, 9):
        letters = collections.Counter()
        for s in range(20):
            e = QuantifierLogicQA()
            ok = e.generate(seed=s * 1000 + level * 37 + 17,
                            parameter={"level": level})
            if ok:
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
