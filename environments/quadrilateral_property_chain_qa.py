"""
Quadrilateral Property Chain QA (v4 G4c, for quadrilateral-property).

Complement to parallel_rhombus_property_qa — this one asks multi-hop
property reasoning: "Given that diagonals AC and BD bisect each other
AND are perpendicular AND equal — is this a rhombus, rectangle, or square?"

Reward: exact string match on quadrilateral type.

Level axes:
  A) Number of properties given: 2 at L0, 3 at L3+, full 4 at L6+
  B) Target: type-id at L0-5, then also a derived quantity (area/diagonal) at L6+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# Property -> set of quadrilateral types that have this property
_PROPS = {
    "diagonals_bisect": {"parallelogram", "rhombus", "rectangle", "square"},
    "diagonals_perpendicular": {"rhombus", "square"},
    "diagonals_equal": {"rectangle", "square"},
    "four_sides_equal": {"rhombus", "square"},
    "four_angles_90": {"rectangle", "square"},
    "opposite_sides_parallel": {"parallelogram", "rhombus", "rectangle", "square"},
    "opposite_sides_equal": {"parallelogram", "rhombus", "rectangle", "square"},
    "exactly_one_pair_parallel": {"trapezoid"},
}


def _most_general(candidates):
    """Return the unique most-general type consistent with `candidates`.

    Hierarchy: parallelogram ⊃ rhombus ⊃ square, parallelogram ⊃ rectangle ⊃
    square; trapezoid is separate. Returns None when the candidate set is
    incomparable (e.g., {rhombus, rectangle} alone) or empty.
    """
    if not candidates:
        return None
    if candidates == {"trapezoid"}:
        return "trapezoid"
    if "trapezoid" in candidates:
        return None
    if "parallelogram" in candidates:
        return "parallelogram"
    if "rhombus" in candidates and "rectangle" in candidates:
        return None
    if "rhombus" in candidates:
        return "rhombus"
    if "rectangle" in candidates:
        return "rectangle"
    if "square" in candidates:
        return "square"
    return None

_READABLE = {
    "diagonals_bisect": "The diagonals bisect each other",
    "diagonals_perpendicular": "The diagonals are perpendicular",
    "diagonals_equal": "The diagonals are equal in length",
    "four_sides_equal": "All four sides are equal",
    "four_angles_90": "All four angles are 90°",
    "opposite_sides_parallel": "Opposite sides are parallel",
    "opposite_sides_equal": "Opposite sides are equal",
    "exactly_one_pair_parallel": "Exactly one pair of opposite sides is parallel",
}

_OPTIONS_BLURB = (
    "Options: A. parallelogram  B. rhombus  C. rectangle  D. square  E. trapezoid"
)

_TEMPLATES = [
    "Quadrilateral ABCD is given. Known properties: {props}. What is the most general type ABCD must be (the weakest classification justified by the given properties)? "
    + _OPTIONS_BLURB
    + ". Put the letter in <answer>...</answer>.",
    "Properties of quadrilateral ABCD: {props}. Choose the most general classification consistent with these properties. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "From the given properties of ABCD ({props}), what is the most general quadrilateral type that must hold? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Given {props}, classify ABCD using the weakest definite type from "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "ABCD satisfies: {props}. Identify the broadest type that ABCD is guaranteed to be. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Suppose quadrilateral ABCD has these properties: {props}. Choose the most general guaranteed type. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Properties: {props}. What is the most general type consistent with all of them for ABCD? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "If ABCD has {props}, what is the broadest classification justified by these facts? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Quadrilateral ABCD: {props}. Pick the most general guaranteed type. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "ABCD has properties {props}. Select the weakest definite classification. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    # 2026-05-03 (M54 / SM-T7 additional reference style phrasings):
    "In quadrilateral ABCD with diagonals AC and BD, the following hold: {props}. Identify the most general type of ABCD. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Suppose in ABCD, the following are observed about its sides, angles, or diagonals: {props}. What is the broadest quadrilateral type guaranteed by these conditions? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "ABCD is a quadrilateral. Given that {props}, name the most general (weakest) classification of ABCD that must hold. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "{props}. Choose the most general type ABCD must be. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Given the listed properties {props} of ABCD, what is the broadest guaranteed classification? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "ABCD has {props}. Select the most general consistent type. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Properties of ABCD: {props}. Pick the broadest type that all of ABCD-like quadrilaterals satisfying them belong to. "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "ABCD: {props}. Most general guaranteed classification? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
    "Given that ABCD has {props}, what is the most general quadrilateral class implied? "
    + _OPTIONS_BLURB
    + ". Put letter in <answer>...</answer>.",
]

class QuadrilateralPropertyChainQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "quadrilateral_property_chain"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_props = min(4, 2 + level // 3)  # 2..4
        return {"n_props": n_props, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 239)
        self._primary_complexity_feature = level

        # Pick a target quadrilateral type
        types = ["parallelogram", "rhombus", "rectangle", "square", "trapezoid"]
        target = rng.choice(types)

        valid_props = [p for p, ts in _PROPS.items() if target in ts]
        if not valid_props:
            return None

        # Find a prop subset whose "most-general consistent type" equals target.
        # The task asks the model for the most-general type consistent with the
        # listed properties (parallelogram is the most general within the
        # parallelogram lattice; trapezoid is separate). This is well-defined
        # for any non-incomparable candidate set, so all 5 targets are
        # reachable without negative properties.
        from itertools import combinations
        n_props = min(cfg["n_props"], len(valid_props))
        all_subsets = list(combinations(valid_props, n_props))
        rng.shuffle(all_subsets)

        chosen_props = None
        for subset in all_subsets:
            candidates = set(types)
            for p in subset:
                candidates &= _PROPS[p]
            if _most_general(candidates) == target:
                chosen_props = list(subset)
                break
        if chosen_props is None:
            # Drop n_props by 1 and retry — at small n_props every target
            # has a valid subset (e.g., parallelogram with any single
            # parallelogram-only prop, square with the right pair).
            for k in range(max(1, n_props - 1), 0, -1):
                subsets = list(combinations(valid_props, k))
                rng.shuffle(subsets)
                for subset in subsets:
                    candidates = set(types)
                    for p in subset:
                        candidates &= _PROPS[p]
                    if _most_general(candidates) == target:
                        chosen_props = list(subset)
                        break
                if chosen_props is not None:
                    break
        if chosen_props is None:
            return None

        best = target
        props_txt = "; ".join(_READABLE[p] for p in chosen_props)

        letter_map = {"parallelogram": "A", "rhombus": "B",
                      "rectangle": "C", "square": "D", "trapezoid": "E"}
        letter = letter_map[best]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        q = _TEMPLATES[sidx].format(props=props_txt)

        # BUGFIX 2026-04-24 (bug 1, image leakage): draw a neutral generic
        # quadrilateral instead of the answer shape so visual recognition
        # doesn't short-circuit the property-matching task.
        img = self._render("generic_quad", rng)
        return q, letter, img

    def _render(self, quad_type, rng):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 4)
        ax.set_aspect("equal")
        ax.axis("off")

        # BUGFIX 2026-04-24: generic_quad case — neutral irregular quadrilateral
        # to avoid visual leakage of the answer shape.
        if quad_type == "generic_quad":
            pts = [(0.3, 0.6), (4.2, 0.3), (4.5, 2.7), (1.0, 2.9)]
        elif quad_type == "parallelogram":
            pts = [(0.5, 0.5), (4, 0.5), (4.5, 2.5), (1, 2.5)]
        elif quad_type == "rhombus":
            pts = [(0.5, 1.5), (2.5, 0), (4.5, 1.5), (2.5, 3)]
        elif quad_type == "rectangle":
            pts = [(0.5, 0.5), (4.5, 0.5), (4.5, 2.5), (0.5, 2.5)]
        elif quad_type == "square":
            pts = [(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]
        else:  # trapezoid
            pts = [(0.2, 0.5), (4.5, 0.5), (3.5, 2.5), (1.5, 2.5)]

        ax.add_patch(mpatches.Polygon(pts, fc="none", ec="black", lw=2.0))
        labels = ["A", "B", "C", "D"]
        for i, (x, y) in enumerate(pts):
            ax.text(x + (-0.2 if i in (0, 3) else 0.15),
                    y + (-0.25 if i < 2 else 0.15),
                    labels[i], fontsize=14, fontweight="bold")
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_qpc"
    os.makedirs(out_dir, exist_ok=True)
    env = QuadrilateralPropertyChainQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 71
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[qpc L{level} s{s}] FAILED (retry)")
                continue
            env.render().save(f"{out_dir}/qpc_s{s}_L{level}.png")
            print(f"[qpc L{level} s{s}] A={env._answer}")
