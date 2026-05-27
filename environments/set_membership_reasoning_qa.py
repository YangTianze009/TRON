"""
Set Membership Reasoning QA environment (v3 planning, env #61).

Goal: perform set-theoretic reasoning on a Venn diagram (2, 3, or 4 sets).
Each region is annotated with an integer count; at higher levels some
regions are unknown ('?') and must be derived from universe totals. The
learner is asked a compound set-operation question. Targets dynamic-math Venn
+ reference deductive and X5 + X4 (scientific diagram).

Difficulty axes (per spec):
  A) `n_sets = 2 + level // 3`              (2..4)
  B) `n_unknown_regions = level // 2`        (regions derived from totals)
     Also `question_complexity`: single op -> compound (A intersect B
     minus C union D).

Format: 4-way MCQ (letter) — options are integer counts.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ---------------------------------------------------------------------- #
# Region enumeration per n_sets.  A region is a frozenset of set indices.
# ---------------------------------------------------------------------- #

def _regions(n_sets: int) -> List[frozenset]:
    out = []
    for mask in range(1, 1 << n_sets):
        s = frozenset(i for i in range(n_sets) if mask & (1 << i))
        out.append(s)
    return out

_SET_NAME_POOLS = [
    ["A", "B", "C", "D"],
    ["P", "Q", "R", "S"],
    ["X", "Y", "Z", "W"],
    ["S1", "S2", "S3", "S4"],
]

_TITLE_VARIANTS = [
    "Venn Diagram",
    "Set Membership Diagram",
    "Overlapping Sets",
    "Set Counts",
    "Set Relations",
    "Element Distribution",
    "Set Intersection Diagram",
]

_PALETTES_VENN = [
    ["#3498db", "#e74c3c", "#27ae60", "#f39c12"],
    ["#9b59b6", "#1abc9c", "#e67e22", "#34495e"],
    ["#f1c40f", "#16a085", "#c0392b", "#2980b9"],
    ["#8e44ad", "#d35400", "#27ae60", "#2c3e50"],
    ["#e84393", "#0984e3", "#fdcb6e", "#00b894"],
]

_QUESTION_INTROS_VENN = [
    "Using the Venn diagram in the image,",
    "Based on the set diagram shown,",
    "Refer to the overlapping sets in the figure;",
    "From the diagram of intersecting sets,",
]

class SetMembershipReasoningQA(StandaloneVisualEnv):
    ENV_NAME = "set_membership_reasoning"

    # -------------------------------------------------- #
    # Level configuration (2 axes from spec)
    # -------------------------------------------------- #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_sets = min(4, 2 + level // 3)                  # 2,2,2,3,3,3,4,4,4,4
        n_unknown_regions = level // 2                    # 0..4
        if level <= 2:
            q_complexity = "simple"
        elif level <= 5:
            q_complexity = "pair"
        else:
            q_complexity = "compound"
        return {
            "n_sets": n_sets,
            "n_unknown_regions": n_unknown_regions,
            "q_complexity": q_complexity,
        }

    # -------------------------------------------------- #
    # Problem generation
    # -------------------------------------------------- #

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        # Unique prime 1409 for this env
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1409)
        self._primary_complexity_feature = cfg["n_sets"] * 2 + cfg["n_unknown_regions"]

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n_sets = cfg["n_sets"]
        set_names = rng.choice(_SET_NAME_POOLS)[:n_sets]
        regions = _regions(n_sets)

        # Assign counts to each region
        region_count = {}
        for r in regions:
            if len(r) == 1:
                region_count[r] = rng.randint(4, 20)
            elif len(r) == 2:
                region_count[r] = rng.randint(2, 10)
            elif len(r) == 3:
                region_count[r] = rng.randint(1, 6)
            else:
                region_count[r] = rng.randint(0, 4)

        # Pick unknown regions (to be derived from totals)
        n_unknown = min(cfg["n_unknown_regions"], max(0, len(regions) - 2))
        unknown_regions = set(rng.sample(regions, n_unknown)) if n_unknown > 0 else set()

        # For each set, we will display its total count (sum over all
        # regions containing that set), so unknown regions can be solved.
        set_totals = {}
        for i in range(n_sets):
            total = 0
            for r, cnt in region_count.items():
                if i in r:
                    total += cnt
            set_totals[i] = total

        # Build the question
        result_answer = self._build_question(
            rng, n_sets, set_names, region_count, cfg["q_complexity"])
        if result_answer is None:
            return None
        question_text, correct_count = result_answer

        # MCQ 4-way (all integer)
        answer_letter, options = self._build_mcq_options(rng, correct_count)

        title = rng.choice(_TITLE_VARIANTS)
        palette_pick = rng.choice(_PALETTES_VENN)
        image = self._render(
            n_sets, set_names, region_count, unknown_regions,
            set_totals, options, title, palette=palette_pick)
        intro = rng.choice(_QUESTION_INTROS_VENN)
        full_question = (
            f"{intro} {question_text} "
            f"Answer with a single letter."
        )
        return full_question, answer_letter, image

    def _build_question(self, rng, n_sets, set_names, region_count,
                         q_complexity: str):
        regions = list(region_count.keys())
        names = set_names
        if q_complexity == "simple":
            # Simple questions: intersection, only_a, union etc.
            qtype = rng.choice(["intersection", "only_a", "union", "exactly_k"])
            if qtype == "intersection":
                # A intersect B (if n_sets >= 2)
                a, b = rng.sample(range(n_sets), 2)
                target = [r for r in regions if a in r and b in r]
                count = sum(region_count[r] for r in target)
                q = (f"how many elements are in {names[a]} intersect "
                     f"{names[b]}?")
                return q, count
            if qtype == "only_a":
                a = rng.randint(0, n_sets - 1)
                target = [r for r in regions if r == frozenset({a})]
                count = sum(region_count[r] for r in target)
                q = (f"how many elements are only in {names[a]} (and not in "
                     f"any other set)?")
                return q, count
            if qtype == "union":
                a, b = rng.sample(range(n_sets), 2)
                target = [r for r in regions if a in r or b in r]
                count = sum(region_count[r] for r in target)
                q = f"how many elements are in {names[a]} union {names[b]}?"
                return q, count
            if qtype == "exactly_k":
                target = [r for r in regions if len(r) == 1]
                count = sum(region_count[r] for r in target)
                q = "how many elements are in exactly one of the sets?"
                return q, count
        elif q_complexity == "pair":
            # A intersect B but not C, or (A intersect B) minus C
            if n_sets < 3:
                # fall back to simple for 2-set diagrams
                a, b = rng.sample(range(n_sets), 2)
                target = [r for r in regions if a in r and b in r]
                count = sum(region_count[r] for r in target)
                q = (f"how many elements are in {names[a]} intersect "
                     f"{names[b]}?")
                return q, count
            a, b, c = rng.sample(range(n_sets), 3)
            target = [r for r in regions if (a in r and b in r) and c not in r]
            count = sum(region_count[r] for r in target)
            q = (f"how many elements are in {names[a]} intersect {names[b]} "
                 f"but NOT in {names[c]}?")
            return q, count
        else:  # compound
            # (A intersect B) minus (C union D) or similar
            if n_sets < 3:
                a, b = rng.sample(range(n_sets), 2)
                target = [r for r in regions if a in r and b in r]
                count = sum(region_count[r] for r in target)
                q = (f"how many elements are in {names[a]} intersect "
                     f"{names[b]}?")
                return q, count
            # pick compound
            if n_sets >= 4 and rng.random() < 0.5:
                # (A ∩ B) minus (C ∪ D)
                a, b, c, d = rng.sample(range(n_sets), 4)
                target = [r for r in regions
                          if (a in r and b in r) and (c not in r and d not in r)]
                count = sum(region_count[r] for r in target)
                q = (f"how many elements are in {names[a]} intersect "
                     f"{names[b]} but NOT in ({names[c]} union "
                     f"{names[d]})?")
                return q, count
            else:
                # (A ∪ B) minus C  (= elements in A or B but not C)
                a, b, c = rng.sample(range(n_sets), 3)
                target = [r for r in regions if (a in r or b in r) and c not in r]
                count = sum(region_count[r] for r in target)
                q = (f"how many elements are in ({names[a]} union "
                     f"{names[b]}) but NOT in {names[c]}?")
                return q, count
        return None

    def _build_mcq_options(self, rng: random.Random,
                            correct_count: int) -> Tuple[str, List[str]]:
        # 4-way MCQ: correct + 3 plausible integer distractors near correct
        correct_str = str(correct_count)
        distractors = set()
        # nearby integers
        for delta in [-2, -1, 1, 2, 3, -3]:
            v = correct_count + delta
            if v >= 0 and v != correct_count:
                distractors.add(str(v))
            if len(distractors) >= 6:
                break
        distractors = list(distractors)
        rng.shuffle(distractors)
        # pick 3
        distractors = distractors[:3]
        while len(distractors) < 3:
            distractors.append(str(correct_count + len(distractors) + 5))
        insert_idx = rng.randint(0, 3)
        options = distractors[:insert_idx] + [correct_str] + distractors[insert_idx:]
        options = options[:4]
        if correct_str not in options:
            options[0] = correct_str
        if options.count(correct_str) > 1:
            seen = False
            for i, o in enumerate(options):
                if o == correct_str:
                    if seen:
                        options[i] = str(correct_count + 7)
                    seen = True
        answer_letter = chr(ord("A") + options.index(correct_str))
        return answer_letter, options

    # -------------------------------------------------- #
    # Rendering: Venn diagram
    # -------------------------------------------------- #

    def _render(self, n_sets, set_names, region_count, unknown_regions,
                 set_totals, options, title, palette=None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans"])
        base_fs = self._rng.choice([11, 12, 13])

        fig_w = 12.0 * sc
        fig_h = 8.0 * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)
        ax_diag = fig.add_subplot(gs[0])
        ax_opts = fig.add_subplot(gs[1])

        ax_diag.set_facecolor(style["bg_color"])
        ax_diag.set_aspect("equal")
        ax_diag.axis("off")
        ax_diag.set_title(title, fontsize=base_fs + 3, fontweight="bold",
                           fontfamily=ff, pad=10)

        if palette is None:
            palette = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

        # Configure centers, radii, and region-label centers per n_sets
        if n_sets == 2:
            circles = [((-0.85, 0.0), 1.7), ((0.85, 0.0), 1.7)]
            label_pos = [(-2.1, 1.85), (2.1, 1.85)]
            region_centers = {
                frozenset({0}): (-1.85, 0.0),
                frozenset({1}): (1.85, 0.0),
                frozenset({0, 1}): (0.0, 0.0),
            }
            ax_diag.set_xlim(-4.0, 4.0)
            ax_diag.set_ylim(-3.0, 3.2)
        elif n_sets == 3:
            circles = [((-0.95, 0.55), 1.7), ((0.95, 0.55), 1.7),
                        ((0.0, -0.95), 1.7)]
            label_pos = [(-2.35, 2.3), (2.35, 2.3), (0.0, -2.95)]
            region_centers = {
                frozenset({0}):         (-1.9, 1.1),
                frozenset({1}):         (1.9, 1.1),
                frozenset({2}):         (0.0, -2.05),
                frozenset({0, 1}):      (0.0, 1.3),
                frozenset({0, 2}):      (-1.2, -0.65),
                frozenset({1, 2}):      (1.2, -0.65),
                frozenset({0, 1, 2}):   (0.0, -0.1),
            }
            ax_diag.set_xlim(-4.0, 4.0)
            ax_diag.set_ylim(-4.0, 3.5)
        else:
            # 4-set: 4 overlapping circles; approximate region centers by
            # sampling grid of points and averaging per membership mask.
            centers4 = [(-0.95, 0.4), (0.95, 0.4),
                         (-0.4, -0.6), (0.4, -0.6)]
            rad = 1.75
            circles = [(c, rad) for c in centers4]
            label_pos = [(-2.85, 2.1), (2.85, 2.1),
                          (-2.85, -2.6), (2.85, -2.6)]
            region_centers = {}
            grid_pts = []
            for xi in range(-40, 41, 2):
                for yi in range(-40, 41, 2):
                    px, py = xi * 0.1, yi * 0.1
                    mem = frozenset(i for i, (cx, cy) in enumerate(centers4)
                                     if (px - cx) ** 2 + (py - cy) ** 2 <= rad ** 2)
                    if not mem:
                        continue
                    grid_pts.append((mem, px, py))
            for mem in set(m for m, _, _ in grid_pts):
                pts = [(px, py) for (m, px, py) in grid_pts if m == mem]
                if not pts:
                    continue
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                region_centers[mem] = (cx, cy)
            ax_diag.set_xlim(-4.3, 4.3)
            ax_diag.set_ylim(-4.0, 3.5)

        # Draw circles
        for i, ((cx, cy), r) in enumerate(circles):
            circ = mpatches.Circle(
                (cx, cy), r,
                facecolor=palette[i % len(palette)],
                edgecolor="#2c3e50", linewidth=1.8,
                alpha=0.22, zorder=3)
            ax_diag.add_patch(circ)

        # Draw set labels
        for i, (lx, ly) in enumerate(label_pos):
            ax_diag.text(lx, ly, set_names[i],
                          fontsize=base_fs + 2, fontweight="bold",
                          ha="center", va="center", fontfamily=ff,
                          color="#2c3e50", zorder=5)

        # Draw region counts
        for region, (rx, ry) in region_centers.items():
            if region in unknown_regions:
                text = "?"
                color = "#e74c3c"
            else:
                text = str(region_count.get(region, "?"))
                color = "#1a1a1a"
            ax_diag.text(rx, ry, text,
                          fontsize=base_fs, fontweight="bold",
                          ha="center", va="center", fontfamily=ff,
                          color=color, zorder=6)

        # Totals (only shown if there are unknowns — so they can be solved)
        if unknown_regions:
            y0 = ax_diag.get_ylim()[0] + 0.2
            tot_str = "  ".join(
                f"|{set_names[i]}| = {set_totals[i]}" for i in range(n_sets))
            ax_diag.text(0.0, y0, tot_str,
                          fontsize=base_fs, fontfamily=ff, ha="center",
                          va="bottom", color="#34495e",
                          bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="white", edgecolor="#888"))

        # --- Options panel ---
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, 10)
        ax_opts.set_ylim(0, 10)
        ax_opts.axis("off")
        ax_opts.set_title("Choose the correct count:",
                           fontsize=base_fs + 2, fontweight="bold",
                           fontfamily=ff, pad=10)
        y = 8.0
        dy = 1.2
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            ax_opts.text(0.4, y, f"({letter}) {opt}",
                          fontsize=base_fs + 1, ha="left", va="top",
                          fontfamily=ff, color="#1a1a1a")
            y -= dy

        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05,
                             wspace=0.25)
        return self.fig_to_pil(fig, dpi=style["dpi"])

# ---------------------------------------------------------------------- #
# Local smoke test
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import collections
    env = SetMembershipReasoningQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED")
                continue
            print(f"[seed={seed} L{level}] A={env._answer}  "
                  f"Q={env.get_instruction()[:80]}")
    for level in (0, 3, 6, 9):
        letters = collections.Counter()
        for s in range(20):
            e = SetMembershipReasoningQA()
            if e.generate(seed=s * 1000 + level * 37 + 17,
                          parameter={"level": level}):
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
