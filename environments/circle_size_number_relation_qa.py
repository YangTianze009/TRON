"""
Circle Size-Number Relation QA.

2026-05-05 R5 B5B (FINAL): REWRITTEN as pure single-letter MCQ. Image shows
N circles labeled with numbers; circle radius follows a hidden rule based on
the number. Pick the rule from a list of candidate rules.

Per-level design:
  L0/L1 — 4-MCQ. 3 circles labeled e.g. 1/4/9. Rule = "n²".
          Options: A. n²  B. 2n  C. n+3  D. n
  L2-L4 — 5-MCQ. 4 circles. Various rules (linear / quadratic). 10/15% trap.
  L5-L7 — 5-MCQ. 5 circles. More rules + 20% trap.
  L8/L9 — 5-MCQ. 5+ circles, nonlinear / fibonacci-like rules. 30/40% trap.

Rule pool (each maps int n -> radius):
  - n            (linear)
  - 2n           (linear scaled)
  - n²/2         (quadratic, halved to keep size reasonable)
  - n + c        (offset)
  - sqrt(n)*k    (sqrt)
  - Fibonacci    (n-th Fibonacci)
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


def _radius_linear(n: int) -> float:
    return 0.05 + 0.05 * n


def _radius_2n(n: int) -> float:
    return 0.04 + 0.025 * 2 * n


def _radius_quad(n: int) -> float:
    return 0.05 + 0.018 * (n * n)


def _radius_offset(n: int) -> float:
    return 0.04 + 0.04 * (n + 2)


def _radius_sqrt(n: int) -> float:
    return 0.06 + 0.10 * math.sqrt(max(0, n))


_RULES_BASIC = [
    ("n",     "Radius is proportional to n",     _radius_linear),
    ("2n",    "Radius is proportional to 2n",    _radius_2n),
    ("n^2",   "Radius is proportional to n²",    _radius_quad),
    ("n+c",   "Radius is proportional to (n + constant)", _radius_offset),
]

_RULES_ADV = [
    ("n",      "Radius is proportional to n",                       _radius_linear),
    ("2n",     "Radius is proportional to 2n",                      _radius_2n),
    ("n^2",    "Radius is proportional to n²",                      _radius_quad),
    ("n+c",    "Radius is proportional to (n + constant)",          _radius_offset),
    ("sqrt",   "Radius is proportional to √n",                       _radius_sqrt),
]


class CircleSizeNumberRelationQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # circle layout is orientation-sensitive
    ENV_NAME = "circle_size_number_relation"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            return {
                "n_circles": 3, "n_options": 4,
                "trap_rate": 0.0, "rules": _RULES_BASIC,
                "tight": False,
            }
        if level <= 4:
            return {
                "n_circles": 4, "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.15,
                "rules": _RULES_BASIC,
                "tight": False,
            }
        if level <= 7:
            return {
                "n_circles": 5, "n_options": 5,
                "trap_rate": 0.20,
                "rules": _RULES_ADV,
                "tight": False,
            }
        # L8/L9
        return {
            "n_circles": 6, "n_options": 5,
            "trap_rate": 0.40,
            "rules": _RULES_ADV,
            "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 941)
        for _ in range(20):
            res = self._try_generate(cfg, rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, rng):
        rules = cfg["rules"]
        # Pick true rule
        true_rule_idx = rng.randint(0, len(rules) - 1)
        true_rule_id, true_rule_text, true_radius_fn = rules[true_rule_idx]

        # Pick distinct numbers per circle. For 'n^2' and 'sqrt' we want to
        # keep n small so radii are reasonable.
        n_circles = cfg["n_circles"]
        if true_rule_id in ("n^2",):
            pool = list(range(1, 5))
        elif true_rule_id in ("sqrt",):
            pool = list(range(1, 10))
        else:
            pool = list(range(1, 8))
        if len(pool) < n_circles:
            return None
        rng.shuffle(pool)
        numbers = sorted(pool[:n_circles])
        radii = [true_radius_fn(n) for n in numbers]

        # Build options: which rule does the figure follow?
        n_options = cfg["n_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]

        # Distractor rules (other than the true one)
        wrong_rules = [r for r in rules if r[0] != true_rule_id]
        rng.shuffle(wrong_rules)
        wrong_texts = [r[1] for r in wrong_rules]

        if n_options == 4:
            need = 3
            if len(wrong_texts) < need:
                return None
            options = [true_rule_text] + wrong_texts[:need]
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_rule_text)]
        else:
            # 5-MCQ with "No correct answer"
            n_value_slots = 4
            use_trap = (rng.random() < cfg["trap_rate"])
            if len(wrong_texts) < n_value_slots:
                # Pad with rules from advanced pool
                extra = [r for r in _RULES_ADV
                         if r[0] != true_rule_id
                         and r[1] not in wrong_texts]
                rng.shuffle(extra)
                wrong_texts.extend([r[1] for r in extra])
            if use_trap and len(wrong_texts) >= n_value_slots:
                chosen = wrong_texts[:n_value_slots]
                rng.shuffle(chosen)
                options = list(chosen)
                options.append("No correct answer")
                correct_letter = last_letter
            else:
                if len(wrong_texts) < n_value_slots - 1:
                    return None
                chosen = [true_rule_text] + wrong_texts[:n_value_slots - 1]
                rng.shuffle(chosen)
                options = list(chosen)
                options.append("No correct answer")
                correct_letter = opt_letters[options.index(true_rule_text)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        stems = [
            (f"The figure shows {n_circles} circles, each labeled with a "
             f"number. The radius of each circle follows a fixed rule "
             f"based on its number. Which rule is being used?"),
            (f"As shown in the diagram, {n_circles} circles are drawn with "
             f"different sizes. Each circle is labeled with a positive "
             f"integer, and the radius is determined by a rule that "
             f"depends on the integer. Identify the rule."),
            (f"In the figure, {n_circles} circles of varying sizes are "
             f"each labeled with a number. The size (radius) follows a "
             f"hidden function of the number. Which of the following "
             f"functions matches the data?"),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        img = self._render(numbers, radii, rng)
        return question, correct_letter, img

    def _render(self, numbers: List[int], radii: List[float],
                rng: random.Random) -> Image.Image:
        n = len(numbers)
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(6, n * 1.3) * sc, 2.4 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        fill_color = rng.choice([
            "#3498db", "#2ecc71", "#9b59b6", "#e74c3c",
            "#f39c12", "#1abc9c", "#e67e22", "#16a085"])

        # Position the circles in a row, scale radii to fit row height
        max_r = max(radii)
        scale = 0.35 / max(max_r, 0.001)
        for i, (num, r) in enumerate(zip(numbers, radii)):
            x = i + 0.5
            y = 0.6
            disp_r = r * scale
            patch = mpatches.Circle((x, y), disp_r,
                                    fc=fill_color, ec="black",
                                    lw=1.2, alpha=0.85)
            ax.add_patch(patch)
            # Label inside if big enough, else above
            font_size = 14 if disp_r >= 0.18 else 11
            text_color = "white" if disp_r >= 0.18 else "black"
            label_y = y if disp_r >= 0.18 else y + disp_r + 0.1
            ax.text(x, label_y, str(num),
                    fontsize=font_size, ha="center", va="center",
                    fontweight="bold", color=text_color)

        ax.set_title("Circles labeled with numbers (radius follows a rule)",
                     fontsize=11)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = CircleSizeNumberRelationQA()
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer}")
