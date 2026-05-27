"""
Visual Similarity Ranking QA (multi-image P5 capability + P3 fine-grained).

A reference image (R) and 4 comparison images (A, B, C, D) are shown,
each composed of simple geometric shapes. The model must pick the option
MOST similar to R (fewest attribute mismatches).

Difficulty axes:
  A) n_attributes (2 -> 5) -- attributes per shape varying across options
  B) similarity_gradient -- L0: one option is identical; L9: all differ,
                            must count mismatches precisely.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon", "star", "diamond"]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#f1c40f", "#e91e63"]
_SIZES = [0.25, 0.4, 0.55]  # small, medium, large
_POSITIONS = ["left", "center", "right", "top", "bottom"]

_QUESTION_TEMPLATES_VSR = [
    "A reference image R is shown at the top, and 4 comparison images (A, B, C, D) are shown below. Each image contains simple shapes with specific colors and sizes. Which comparison image is MOST similar to R (i.e., has the fewest differing attributes)? Answer with a single letter.",
    "Image R at top is the reference. Pick the option (A-D) below that most closely matches R in shape/color/size. Answer with a single letter.",
    "Compare each option (A, B, C, D) to the reference R. Which one has the fewest attribute mismatches with R? Answer with a single letter.",
    "Looking at the reference R and the 4 candidates below, identify the candidate that differs from R in the fewest attributes. Answer with a single letter.",
    "Of the four images A-D, which is closest to the reference R (matches it in the most attributes)? Answer with a single letter.",
]
_TITLE_VARIANTS_VSR = [
    "Visual Similarity Ranking",
    "Pick the Closest Match",
    "Reference vs Candidates",
    "Similarity Comparison",
    "Visual Match Test",
]

class VisualSimilarityRankingQA(StandaloneVisualEnv):
    ENV_NAME = "visual_similarity_ranking"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_attributes": 2 + level // 3,        # 2..5
            "n_shapes": 2 + level // 3,            # 2..5
            "similarity_gradient": level,           # 0..9
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1439)
        self._primary_complexity_feature = cfg["n_attributes"] * 3 + level

        n_shapes = cfg["n_shapes"]

        # Build reference image R: list of shapes with varying attributes.
        reference = []
        for _ in range(n_shapes):
            reference.append({
                "shape": sub_rng.choice(_SHAPES),
                "color": sub_rng.choice(_COLORS),
                "size_idx": sub_rng.randint(0, len(_SIZES) - 1),
            })

        # Generate 4 options with increasing number of attribute mismatches.
        # Correct = fewest mismatches. At L0 = identical (0 mismatches).
        # At higher levels, all options differ but correct has fewest.
        options = []
        mismatch_counts = []

        # Correct option: mismatches based on level
        correct_mismatch = min(level // 3, n_shapes)  # 0..3
        correct_opt = self._make_variant(reference, correct_mismatch, sub_rng)
        options.append(correct_opt)
        mismatch_counts.append(correct_mismatch)

        # Distractor options: strictly more mismatches
        attempts = 0
        while len(options) < 4 and attempts < 100:
            attempts += 1
            extra = sub_rng.randint(1, max(2, n_shapes))
            target = correct_mismatch + extra
            target = min(target, n_shapes * 3)  # cap
            target = max(target, correct_mismatch + 1)
            distractor = self._make_variant(reference, target, sub_rng)
            # Check distractor not accidentally closer than correct
            actual = self._count_mismatches(reference, distractor)
            if actual > correct_mismatch:
                options.append(distractor)
                mismatch_counts.append(actual)

        if len(options) < 4:
            return None

        # Shuffle options and compute answer letter
        paired = list(zip(options, mismatch_counts))
        sub_rng.shuffle(paired)
        # Find min mismatch index
        min_mis = min(p[1] for p in paired)
        # In case of tie, pick first (shouldn't happen with strict logic above)
        correct_idx = next(i for i, p in enumerate(paired) if p[1] == min_mis)
        answer_letter = chr(ord("A") + correct_idx)
        shuffled_options = [p[0] for p in paired]

        image = self._render(reference, shuffled_options, sub_rng)
        question = sub_rng.choice(_QUESTION_TEMPLATES_VSR)
        return question, answer_letter, image

    def _make_variant(self, reference: List[Dict], n_mismatches: int,
                      rng: random.Random) -> List[Dict]:
        """Build a copy of the reference with exactly n_mismatches attribute
        differences distributed across shapes."""
        variant = [dict(s) for s in reference]
        n_shapes = len(reference)
        # Each attribute change counts as one mismatch
        total_available = n_shapes * 3  # shape, color, size
        k = min(n_mismatches, total_available)
        # Enumerate all (shape_idx, attribute) slots
        slots = [(i, a) for i in range(n_shapes) for a in ("shape", "color", "size_idx")]
        rng.shuffle(slots)
        for i, attr in slots[:k]:
            if attr == "shape":
                others = [s for s in _SHAPES if s != variant[i]["shape"]]
                variant[i]["shape"] = rng.choice(others)
            elif attr == "color":
                others = [c for c in _COLORS if c != variant[i]["color"]]
                variant[i]["color"] = rng.choice(others)
            elif attr == "size_idx":
                others = [s for s in range(len(_SIZES)) if s != variant[i]["size_idx"]]
                variant[i]["size_idx"] = rng.choice(others)
        return variant

    @staticmethod
    def _count_mismatches(ref: List[Dict], other: List[Dict]) -> int:
        n = 0
        for a, b in zip(ref, other):
            if a["shape"] != b["shape"]:
                n += 1
            if a["color"] != b["color"]:
                n += 1
            if a["size_idx"] != b["size_idx"]:
                n += 1
        return n

    def _draw_shape(self, ax, shape: str, color: str, size: float, cx: float, cy: float):
        if shape == "circle":
            p = plt.Circle((cx, cy), size, fc=color, ec="black", lw=1.2, alpha=0.9)
            ax.add_patch(p)
        elif shape == "square":
            p = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size,
                                   fc=color, ec="black", lw=1.2, alpha=0.9)
            ax.add_patch(p)
        elif shape == "triangle":
            p = RegularPolygon((cx, cy), 3, radius=size * 1.1,
                               orientation=math.pi / 2,
                               fc=color, ec="black", lw=1.2, alpha=0.9)
            ax.add_patch(p)
        elif shape == "pentagon":
            p = RegularPolygon((cx, cy), 5, radius=size * 1.1,
                               orientation=math.pi / 2,
                               fc=color, ec="black", lw=1.2, alpha=0.9)
            ax.add_patch(p)
        elif shape == "hexagon":
            p = RegularPolygon((cx, cy), 6, radius=size * 1.1,
                               fc=color, ec="black", lw=1.2, alpha=0.9)
            ax.add_patch(p)
        elif shape == "star":
            angles = [math.radians(90 + i * 72) for i in range(5)]
            inner = [math.radians(90 + 36 + i * 72) for i in range(5)]
            pts = []
            for a, ia in zip(angles, inner):
                pts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
                pts.append((cx + size * 0.4 * math.cos(ia),
                           cy + size * 0.4 * math.sin(ia)))
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=1.2, alpha=0.9))
        elif shape == "diamond":
            pts = [(cx, cy + size), (cx + size * 0.7, cy),
                   (cx, cy - size), (cx - size * 0.7, cy)]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=1.2, alpha=0.9))

    def _render_panel(self, ax, shapes: List[Dict], title: str,
                      title_color: str = "#2c3e50", highlight: bool = False):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        # Background box
        bg_color = "#fff8e7" if highlight else "#f8f9fa"
        rect = mpatches.FancyBboxPatch(
            (0.08, 0.08), 5.84, 3.84,
            boxstyle="round,pad=0.05",
            fc=bg_color,
            ec=("#e67e22" if highlight else "#95a5a6"),
            lw=(2.2 if highlight else 1.2)
        )
        ax.add_patch(rect)
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color,
                     pad=4)
        # Place shapes horizontally
        n = len(shapes)
        if n == 0:
            return
        x_positions = [(6.0 / (n + 1)) * (i + 1) for i in range(n)]
        for x, s in zip(x_positions, shapes):
            size = _SIZES[s["size_idx"]]
            self._draw_shape(ax, s["shape"], s["color"], size, x, 2.0)

    def _render(self, reference: List[Dict], options: List[List[Dict]],
                sub_rng: random.Random) -> Image.Image:
        style = self._random_style()
        fig = plt.figure(figsize=(11, 8))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 0.2, 1.3], hspace=0.15,
                              wspace=0.12)

        # Top row: Reference spanning all 4 cols
        ax_ref = fig.add_subplot(gs[0, :])
        self._render_panel(ax_ref, reference, "Reference R", title_color="#c0392b",
                           highlight=True)

        # Separator
        ax_sep = fig.add_subplot(gs[1, :])
        ax_sep.axis("off")
        ax_sep.text(0.5, 0.5, "--- Comparison Images ---", ha="center",
                    va="center", fontsize=11, color="#7f8c8d",
                    style="italic", transform=ax_sep.transAxes)

        # Bottom row: 4 options
        letters = ["A", "B", "C", "D"]
        for i, (letter, opt) in enumerate(zip(letters, options)):
            ax_opt = fig.add_subplot(gs[2, i])
            self._render_panel(ax_opt, opt, f"({letter})")

        fig.suptitle(sub_rng.choice(_TITLE_VARIANTS_VSR),
                     fontsize=14, fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = VisualSimilarityRankingQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
