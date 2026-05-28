"""
Spinner Probability QA (D116, P2).

Renders a circular spinner divided into colored sectors of different sizes.
Asks: "What is the probability that the spinner lands on the {color} sector?"
or "Which color is the spinner more likely to land on?"

Verifier: numeric fraction or color name.
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_COLOR_NAMES = {
    "#e74c3c": "red",
    "#3498db": "blue",
    "#2ecc71": "green",
    "#f1c40f": "yellow",
    "#9b59b6": "purple",
    "#e67e22": "orange",
}


class SpinnerProbabilityQA(StandaloneVisualEnv):
    ENV_NAME = "spinner_probability"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    def _level_config(self, level: int) -> Dict:
        # Aligned with a math benchmark D116/161/162: ask "more likely color/number".
        # "probability_value" mode (which asked for a simplified fraction) was
        # invented and not present in the source benchmark — removed.
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_sectors": 4, "modes": ["most_likely"]}
        if level <= 5:
            return {"n_sectors": 5, "modes": ["most_likely", "most_likely_number"]}
        return {"n_sectors": 6, "modes": ["most_likely", "most_likely_number"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4831 + level * 79 + 17)

        n = cfg["n_sectors"]
        # Generate sector sizes as ints summing to 12 (so probabilities are nice fractions)
        total = 12
        sizes = []
        for _ in range(40):
            sizes = [rng.randint(1, 5) for _ in range(n)]
            if sum(sizes) == 0:
                continue
            # Normalize to sum to total
            scale = total / sum(sizes)
            sizes = [max(1, int(round(s * scale))) for s in sizes]
            if sum(sizes) != total:
                continue
            # Require unique max so "most likely" question has a unique answer.
            sorted_sizes = sorted(sizes, reverse=True)
            has_unique_max = sorted_sizes[0] > sorted_sizes[1]
            if has_unique_max and len(set(sizes)) >= 2:
                break
        else:
            return None
        # Use distinct sizes ideally
        colors = list(_COLOR_NAMES.keys())[:n]
        rng.shuffle(colors)
        color_names = [_COLOR_NAMES[c] for c in colors]

        mode = rng.choice(cfg["modes"])
        # In number mode, label sectors with integer numerals 1..n instead of
        # colors (matches D161 "On which number is the spinner more likely to
        # land?").
        if mode == "most_likely_number":
            max_idx = sizes.index(max(sizes))
            number_labels = list(range(1, n + 1))
            rng.shuffle(number_labels)
            answer = str(number_labels[max_idx])
            question = (
                f"The figure shows a spinner divided into {n} numbered "
                f"sectors of different sizes. On which number is the spinner "
                f"more likely to land?"
            )
            img = self._render(sizes, colors, [str(x) for x in number_labels])
            return question, answer, img

        # most_likely (color)
        max_idx = sizes.index(max(sizes))
        answer = color_names[max_idx]
        question = (
            f"The figure shows a spinner divided into {n} colored sectors "
            f"of different sizes. On which color is the spinner more likely "
            f"to land?"
        )
        img = self._render(sizes, colors, color_names)
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, sizes, colors, names) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        total = sum(sizes)
        cur_angle = 0
        for sz, c, name in zip(sizes, colors, names):
            sweep = sz / total * 360
            wedge = patches.Wedge((0, 0), 1.0, cur_angle, cur_angle + sweep,
                                   facecolor=c, edgecolor="white",
                                   linewidth=2)
            ax.add_patch(wedge)
            # Label
            mid_angle = math.radians(cur_angle + sweep / 2)
            lx = 0.6 * math.cos(mid_angle)
            ly = 0.6 * math.sin(mid_angle)
            ax.text(lx, ly, name, ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
            cur_angle += sweep

        # Pointer
        ax.annotate("", xy=(1.1, 0), xytext=(0.0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2.5,
                                    color="#2c3e50"))

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = SpinnerProbabilityQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
