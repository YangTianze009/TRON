"""
Water Jug Multistep QA (round-2 fix 2026-04-16).

Target: reference puzzles / multi-step rule execution (X1). Shows two labelled
water jugs with capacities and a target amount, all rendered ON the image.
The question text asks for the minimum number of pour operations.

Text leakage fix: Capacity and target values moved to the image only (no
numeric leakage in text at L>=3). L0-L2 still include numbers for scaffolding.

Diversity:
- 5+ jug label pairs; 4+ title variants; 4+ question phrasings.
- Colors randomized per seed.
- Capacity axis, target banner, jug width all jittered.
- L0-L2: 2-3 step puzzles, small caps, scaffolded text.
- L3-L6: 3-6 step puzzles, medium caps, pure visual.
- L7-L9: 6-12 step puzzles, larger caps, pure visual, MCQ-style options.
"""
import math
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _water_jug_min_steps(a: int, b: int, target: int) -> Optional[int]:
    """Return minimum pours to get `target` into either jug; None if infeasible."""
    if target > max(a, b):
        return None
    if target == 0:
        return 0
    start = (0, 0)
    visited = {start}
    q = deque([(start, 0)])
    while q:
        (x, y), steps = q.popleft()
        if x == target or y == target:
            return steps
        nxt: List[Tuple[int, int]] = []
        nxt.append((a, y))
        nxt.append((x, b))
        nxt.append((0, y))
        nxt.append((x, 0))
        pour = min(x, b - y)
        nxt.append((x - pour, y + pour))
        pour = min(y, a - x)
        nxt.append((x + pour, y - pour))
        for s in nxt:
            if s not in visited:
                visited.add(s)
                q.append((s, steps + 1))
    return None

_JUG_LABEL_POOLS = [
    ("Jug A", "Jug B"),
    ("Left", "Right"),
    ("Jug 1", "Jug 2"),
    ("Small", "Large"),
    ("Red Jug", "Blue Jug"),
    ("X", "Y"),
    ("Container 1", "Container 2"),
    ("Pitcher A", "Pitcher B"),
]

_NONLEAK_QUESTIONS = [
    "Two empty water jugs with labeled capacities are shown. The target amount to "
    "measure is written on the diagram. On each move you may: fill a jug from the "
    "tap, empty a jug, or pour from one jug to another until the source empties or "
    "the destination fills. What is the minimum number of moves to measure the "
    "target exactly? Answer with an integer.",
    "The diagram shows two empty jugs (capacities marked on each) and a target "
    "amount. Allowed operations per move: fill, empty, or pour. Find the fewest "
    "moves to achieve the target in either jug. Integer only.",
    "Read the two jug capacities and the target from the image. Using the "
    "fill / empty / pour operations, what is the minimum number of moves to "
    "measure exactly the target volume? Answer with an integer.",
    "Given two empty containers whose capacities are labeled in the figure and a "
    "target volume (also shown), compute the minimum number of pour operations "
    "required to measure exactly the target. Answer with an integer.",
]

_LEAK_QUESTIONS = [
    "You have two empty water jugs with capacities {a} L and {b} L. On each "
    "move you may: fill a jug from the tap, empty a jug down the drain, or "
    "pour water from one jug to the other until the source jug is empty or "
    "the target jug is full. What is the minimum number of moves required to "
    "measure exactly {target} L in one of the jugs? Answer with an integer.",
    "Two jugs hold {a} L and {b} L respectively, both initially empty. "
    "Available operations: fill, empty, or pour between jugs. How many moves "
    "at minimum does it take to get exactly {target} L? Answer with an integer.",
    "Given two empty containers of capacity {a} L and {b} L, find the fewest "
    "operations (fill / empty / pour) needed to obtain exactly {target} L in "
    "either container. Answer with an integer.",
    "The diagram shows two jugs ({a} L and {b} L, both empty). Using fill, "
    "drain, and pour operations, what is the minimum number of steps to "
    "measure {target} L? Answer with an integer.",
]

_TITLE_VARIANTS = [
    "Water Jug Puzzle", "Jug Problem", "Measuring Jugs",
    "Water Puzzle", "Jug Challenge", "Pour Planning",
    "Measure the Target",
]

class WaterJugMultistepQA(StandaloneVisualEnv):
    ENV_NAME = "water_jug_multistep"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"cap_min": 2, "cap_max": 5, "min_steps": 2,
                    "max_steps": 4, "scaffold_text": True, "is_mcq": False}
        if level <= 2:
            return {"cap_min": 3, "cap_max": 7, "min_steps": 2,
                    "max_steps": 6, "scaffold_text": True, "is_mcq": False}
        if level <= 4:
            return {"cap_min": 3, "cap_max": 9, "min_steps": 3,
                    "max_steps": 8, "scaffold_text": False, "is_mcq": False}
        if level <= 6:
            return {"cap_min": 4, "cap_max": 12, "min_steps": 4,
                    "max_steps": 10, "scaffold_text": False, "is_mcq": False}
        if level <= 7:
            return {"cap_min": 5, "cap_max": 14, "min_steps": 5,
                    "max_steps": 12, "scaffold_text": False, "is_mcq": True}
        return {"cap_min": 5, "cap_max": 16, "min_steps": 6,
                "max_steps": 14, "scaffold_text": False, "is_mcq": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["cap_max"] + cfg["min_steps"] * 10

        for _ in range(40):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        a = rng.randint(cfg["cap_min"], cfg["cap_max"])
        b = rng.randint(cfg["cap_min"], cfg["cap_max"])
        if a == b:
            return None
        target = rng.randint(1, max(a, b) - 1)
        steps = _water_jug_min_steps(a, b, target)
        if steps is None or steps < cfg["min_steps"]:
            return None
        if steps > cfg["max_steps"]:
            return None

        answer = str(steps)
        if cfg.get("scaffold_text", False):
            template = rng.choice(_LEAK_QUESTIONS)
            q = template.format(a=a, b=b, target=target)
        else:
            q = rng.choice(_NONLEAK_QUESTIONS)

        if cfg.get("is_mcq", False):
            # Build MCQ around correct step count
            correct = steps
            distractors = set()
            deltas = [-3, -2, -1, 1, 2, 3, 4, 5]
            tries = 0
            while len(distractors) < 3 and tries < 30:
                d = rng.choice(deltas)
                cand = correct + d
                if cand >= 1 and cand != correct:
                    distractors.add(cand)
                tries += 1
            if len(distractors) >= 3:
                opts = [correct] + list(distractors)[:3]
                rng.shuffle(opts)
                letter = chr(ord("A") + opts.index(correct))
                # Strip any prior "Answer with an integer" instruction before
                # appending the MCQ format so we don't contradict ourselves.
                q = q.rsplit("Answer with", 1)[0].strip()
                q = (q + "\n" + "\n".join(
                    f"  ({chr(ord('A')+i)}) {opts[i]}" for i in range(4))
                    + "\nAnswer with a single letter.")
                answer = letter

        jug_labels = rng.choice(_JUG_LABEL_POOLS)
        title = rng.choice(_TITLE_VARIANTS)
        image = self._render(a, b, target, cfg, rng, jug_labels, title)
        return q, answer, image

    def _render(self, a, b, target, cfg, sub_rng, jug_labels, title) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = list(style["palette"])
        sub_rng.shuffle(palette)

        fig, ax = plt.subplots(figsize=(6.5 * sc, 5.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        max_cap = max(a, b) + 2
        jug_w = sub_rng.uniform(1.1, 1.5)
        left_cx = sub_rng.uniform(1.3, 1.8)
        right_cx = left_cx + sub_rng.uniform(2.7, 3.2)

        self._draw_jug(ax, cx=left_cx, w=jug_w, cap=a, max_cap=max_cap,
                       color=palette[0 % len(palette)],
                       label=jug_labels[0], fs=fs)
        self._draw_jug(ax, cx=right_cx, w=jug_w, cap=b, max_cap=max_cap,
                       color=palette[2 % len(palette)],
                       label=jug_labels[1], fs=fs)

        ax.set_title(title, fontsize=fs + 2, fontweight="bold", pad=8)
        # Target banner — always on image
        target_text = f"Target: {target} L (in either jug)"
        ax.text((left_cx + right_cx) / 2, -0.8, target_text,
                fontsize=fs + 2, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#fff9c4",
                          edgecolor="#e67e22", linewidth=1.4))

        ax.set_xlim(-0.5, right_cx + 1.5)
        ax.set_ylim(-1.8, max_cap + 1.2)
        ax.set_aspect("equal")
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_jug(self, ax, cx, w, cap, max_cap, color, label, fs):
        ax.add_patch(mpatches.Rectangle(
            (cx - w / 2, 0), w, cap,
            facecolor="#dff1fb", edgecolor="#1a5276", linewidth=1.8))
        # Tick marks
        for i in range(1, cap + 1):
            ax.plot([cx - w / 2, cx - w / 2 + 0.2], [i, i],
                    color="#1a5276", linewidth=1.2)
            ax.text(cx - w / 2 - 0.2, i, f"{i}",
                    fontsize=max(fs - 1, 9), ha="right", va="center",
                    color="#1a5276")
        # Label with capacity
        ax.text(cx, cap + 0.4, f"{label}\n({cap} L)",
                fontsize=fs + 1, fontweight="bold", ha="center",
                va="bottom", color=color)
