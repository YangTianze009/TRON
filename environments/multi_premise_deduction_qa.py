"""
Multi-Premise Deduction QA — redesign 2026-04-16.

DIFFICULTY LADDER (structurally different L0 vs L9):
  L0: DIRECT rule. "All red boxes contain the cat." "The cat is in a red box."
       -> Conclude: red box has the cat. NO CHAIN. Factual match only.
  L1: 2 boxes, 1-hop rule (nested: Box A is red; Box B is in A; item in B).
       Rule: red boxes contain X.
  L2-L3: 3-4 box chain, 0-1 distractor rules.
  L4-L5: 4-5 boxes, with 1-2 distractor rules.
  L6-L7: 5-6 boxes, 2 distractors, 1 red-herring attribute fact.
  L8-L9: 6-7 boxes, 3-4 distractors, contradictory rule that must be skipped.

DIVERSITY:
  1. Scenario pool: boxes/containers, rooms/people, trees/fruits,
     pools/fish, shelves/books, sacks/coins, tanks/gases, vaults/relics
     (8 scenarios).
  2. 5 attribute pairs (warm/cold, heavy/light, etc.), rotated per seed.
  3. 14 item names, item chosen per seed.
  4. 5 question templates, picked per seed.
  5. 5 title variants.
  6. Diagram variants: nested-boxes, side-by-side cards, tree diagram.
  7. Font family rotation.
  8. Box color shuffle per seed.

Values/facts remain in image and text body. No numeric leakage.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_BOX_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_BOX_COLORS = {
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "green":  "#27ae60",
    "yellow": "#f1c40f",
    "purple": "#8e44ad",
    "orange": "#e67e22",
    "pink":   "#e91e8f",
    "teal":   "#1abc9c",
}
_COLOR_LIST = list(_BOX_COLORS.keys())

_ATTR_PAIRS = [
    ("warm", "cold"),
    ("heavy", "light"),
    ("fragile", "sturdy"),
    ("expensive", "cheap"),
    ("shiny", "dull"),
    ("valuable", "ordinary"),
    ("noisy", "silent"),
    ("sharp", "blunt"),
    ("wet", "dry"),
    ("old", "new"),
]
_ALL_ATTRIBUTES = [a for pair in _ATTR_PAIRS for a in pair]
_OPPOSITE = {}
for a, b in _ATTR_PAIRS:
    _OPPOSITE[a] = b
    _OPPOSITE[b] = a

_ITEMS = ["pen", "coin", "gem", "key", "ring", "toy", "bell", "cup",
          "lamp", "book", "clock", "brush", "vase", "dish"]

# Scenario: (container_singular, container_plural, verb)
_SCENARIOS = [
    ("box", "boxes", "inside"),
    ("chest", "chests", "inside"),
    ("crate", "crates", "inside"),
    ("drawer", "drawers", "inside"),
    ("cabinet", "cabinets", "inside"),
    ("vault", "vaults", "inside"),
    ("bag", "bags", "inside"),
    ("bin", "bins", "inside"),
]

class MultiPremiseDeductionQA(StandaloneVisualEnv):
    ENV_NAME = "multi_premise_deduction"

    _QUESTION_TEMPLATES = [
        "Read the facts shown in the image. Which of the following statements about the {item} must be TRUE? Answer with a single letter (A, B, C, or D).",
        "The image lists facts about {container_plural} and their contents. Based solely on these facts, which statement about the {item} is necessarily true? Answer A, B, C, or D.",
        "Study the rules and containment facts in the image. What can you deduce about the {item}? Pick the correct conclusion. Answer with a single letter.",
        "Given the premises in the image, which conclusion about the {item} follows logically? Answer with a single letter A-D.",
        "Use only the facts listed in the image. Which conclusion about the {item} is entailed? Answer with one letter A/B/C/D.",
    ]

    _TITLE_VARIANTS = [
        "Deduction puzzle",
        "Inheritance logic",
        "Nested container rules",
        "Logic rules",
        "Attribute inheritance",
        "Rule chain",
    ]

    # 2026-05-04 R4: full-gradient redesign. Existing L0/L1 are well-designed
    # (direct rule vs. 2-box chain). Saturation came at L4-L7 where adding
    # 1 distractor rule did little. Now each level adds a NEW reasoning load
    # (more boxes, more distractors, red-herrings, or contradictions to skip).
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0 unchanged (direct rule baseline) — works as designed.
        if level == 0:
            return {"n_boxes": 1, "n_distractor_rules": 0,
                    "n_red_herring_attrs": 0, "use_contradictory_subset": False,
                    "direct_rule": True}
        # L1 chain begins
        if level == 1:
            return {"n_boxes": 2, "n_distractor_rules": 0,
                    "n_red_herring_attrs": 0, "use_contradictory_subset": False,
                    "direct_rule": False}
        if level == 2:
            return {"n_boxes": 3, "n_distractor_rules": 1,
                    "n_red_herring_attrs": 0, "use_contradictory_subset": False,
                    "direct_rule": False}
        if level == 3:
            return {"n_boxes": 3, "n_distractor_rules": 2,
                    "n_red_herring_attrs": 1, "use_contradictory_subset": False,
                    "direct_rule": False}
        if level == 4:
            return {"n_boxes": 4, "n_distractor_rules": 2,
                    "n_red_herring_attrs": 1, "use_contradictory_subset": False,
                    "direct_rule": False}
        if level == 5:
            return {"n_boxes": 4, "n_distractor_rules": 3,
                    "n_red_herring_attrs": 2, "use_contradictory_subset": False,
                    "direct_rule": False}
        if level == 6:
            return {"n_boxes": 5, "n_distractor_rules": 3,
                    "n_red_herring_attrs": 2, "use_contradictory_subset": True,
                    "direct_rule": False}
        if level == 7:
            return {"n_boxes": 6, "n_distractor_rules": 4,
                    "n_red_herring_attrs": 3, "use_contradictory_subset": True,
                    "direct_rule": False}
        if level == 8:
            return {"n_boxes": 7, "n_distractor_rules": 4,
                    "n_red_herring_attrs": 3, "use_contradictory_subset": True,
                    "direct_rule": False}
        return {"n_boxes": 8, "n_distractor_rules": 5,
                "n_red_herring_attrs": 4, "use_contradictory_subset": True,
                "direct_rule": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 31 + level * 13 + 11)
        self._primary_complexity_feature = cfg["n_boxes"]

        if cfg.get("direct_rule"):
            return self._gen_direct(rng, level, cfg)

        for attempt in range(30):
            result = self._try_generate(rng, level, cfg)
            if result is not None:
                return result
        return None

    # ---------------------------------------------------------------- #
    # L0: DIRECT rule. 1 box, 1 color rule, 1 item.
    # ---------------------------------------------------------------- #
    def _gen_direct(self, rng, level, cfg):
        container_s, container_p, verb = rng.choice(_SCENARIOS)
        color = rng.choice(_COLOR_LIST)
        attr = rng.choice(_ALL_ATTRIBUTES)
        item = rng.choice(_ITEMS)
        label = rng.choice(_BOX_LABELS)
        facts = [
            f"{container_s.capitalize()} {label} is {color}.",
            f"The {item} is {verb} {container_s} {label}.",
            f"All items {verb} a {color} {container_s} are {attr}.",
        ]
        rng.shuffle(facts)
        correct = attr
        distractor_pool = [_OPPOSITE[correct]]
        fillers = [a for a in _ALL_ATTRIBUTES if a not in (correct, _OPPOSITE[correct])]
        rng.shuffle(fillers)
        distractor_pool.extend(fillers[:2])
        options = list(distractor_pool[:3])
        insert_idx = rng.randint(0, 3)
        options.insert(insert_idx, correct)
        answer_letter = chr(ord("A") + insert_idx)
        option_strs = [f"The {item} is {opt}." for opt in options]
        title = rng.choice(self._TITLE_VARIANTS)
        image = self._render(facts, [label], [color], item, option_strs,
                             level=level, title=title,
                             container_plural=container_p)
        q = rng.choice(self._QUESTION_TEMPLATES).format(
            item=item, container_plural=container_p)
        return q, answer_letter, image

    # ---------------------------------------------------------------- #
    # L1+: chained rule.
    # ---------------------------------------------------------------- #
    def _try_generate(self, rng, level, cfg):
        n_boxes = cfg["n_boxes"]
        if n_boxes > len(_COLOR_LIST):
            return None
        labels = rng.sample(_BOX_LABELS[:max(6, n_boxes + 2)], n_boxes)
        colors = rng.sample(_COLOR_LIST, n_boxes)
        container_s, container_p, verb = rng.choice(_SCENARIOS)

        outer_color = colors[0]
        outer_attr = rng.choice(_ALL_ATTRIBUTES)

        facts: List[str] = []
        for lbl, col in zip(labels, colors):
            facts.append(f"{container_s.capitalize()} {lbl} is {col}.")
        for i in range(n_boxes - 1):
            facts.append(
                f"{container_s.capitalize()} {labels[i + 1]} is {verb} "
                f"{container_s} {labels[i]}.")

        item = rng.choice(_ITEMS)
        if n_boxes == 1:
            facts.append(f"The {item} is {verb} {container_s} {labels[-1]}.")
        else:
            facts.append(f"The {item} is {verb} {container_s} {labels[-1]}.")
        facts.append(
            f"All items {verb} a {outer_color} {container_s} are {outer_attr}.")

        used_attrs = {outer_attr}
        used_colors = set(colors)
        unused_colors = [c for c in _COLOR_LIST if c not in used_colors]
        for _ in range(cfg["n_distractor_rules"]):
            if not unused_colors:
                break
            dc = rng.choice(unused_colors)
            unused_colors.remove(dc)
            available = [a for a in _ALL_ATTRIBUTES
                         if a not in used_attrs and _OPPOSITE[a] not in used_attrs]
            if not available:
                break
            da = rng.choice(available)
            used_attrs.add(da)
            facts.append(
                f"All items {verb} a {dc} {container_s} are {da}.")

        for _ in range(cfg["n_red_herring_attrs"]):
            template = rng.choice([
                "No {0} {c_s} is {1}.",
                "Some {0} {c_p} are {1}.",
                "The {2} is {1}.",
            ])
            pool_colors = unused_colors + [c for c in colors if c != outer_color]
            if not pool_colors:
                continue
            c = rng.choice(pool_colors)
            available_attrs = [a for a in _ALL_ATTRIBUTES
                               if a != outer_attr and a != _OPPOSITE[outer_attr]]
            if not available_attrs:
                continue
            a = rng.choice(available_attrs)
            other_item = rng.choice([x for x in _ITEMS if x != item])
            facts.append(template.format(
                c, a, other_item, c_s=container_s, c_p=container_p))

        if cfg["use_contradictory_subset"]:
            alt_color = rng.choice(unused_colors) if unused_colors else None
            if alt_color is not None:
                facts.append(
                    f"All items {verb} a {alt_color} {container_s} are "
                    f"{_OPPOSITE[outer_attr]}.")

        rng.shuffle(facts)

        correct = outer_attr
        distractor_pool = [_OPPOSITE[correct]]
        for fact in facts:
            for a in _ALL_ATTRIBUTES:
                if f" are {a}." in fact and a != correct and a not in distractor_pool:
                    distractor_pool.append(a)
        fillers = [a for a in _ALL_ATTRIBUTES
                   if a != correct and a not in distractor_pool]
        rng.shuffle(fillers)
        distractor_pool.extend(fillers)
        if len(distractor_pool) < 3:
            return None
        distractors = distractor_pool[:3]
        options = list(distractors)
        insert_idx = rng.randint(0, 3)
        options.insert(insert_idx, correct)
        answer_letter = chr(ord("A") + insert_idx)
        option_strs = [f"The {item} is {opt}." for opt in options]
        title = rng.choice(self._TITLE_VARIANTS)
        image = self._render(facts, labels, colors, item, option_strs,
                             level=level, title=title,
                             container_plural=container_p)
        q = rng.choice(self._QUESTION_TEMPLATES).format(
            item=item, container_plural=container_p)
        return q, answer_letter, image

    # ---------------------------------------------------------------- #
    def _render(self, facts, labels, colors, item, options,
                level=0, title="Deduction puzzle",
                container_plural="boxes") -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans", "monospace"])
        base_fs = self._rng.choice([12, 13, 14])

        n_premises = len(facts)
        n_options = len(options)
        n_lines = n_premises + n_options + 4
        panel_height = max(7.0, n_lines * 0.58 + 1.0)

        fig_w = 11.5 * sc
        fig_h = (panel_height + 1.0) * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.08)
        ax_text = fig.add_subplot(gs[0])
        ax_diag = fig.add_subplot(gs[1])

        ax_text.set_xlim(0, 10)
        ax_text.set_ylim(0, panel_height + 0.5)
        ax_text.set_axis_off()

        border_bottom = 0.2
        border_top = panel_height + 0.3
        border = mpatches.FancyBboxPatch(
            (0.2, border_bottom), 9.6, border_top - border_bottom,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor="#ffffff" if style["bg_color"] == "#ffffff" else style["bg_color"],
            edgecolor="#7f8c8d", linewidth=1.2, zorder=1)
        ax_text.add_patch(border)

        title_y = panel_height - 0.05
        ax_text.text(5.0, title_y, title, fontsize=base_fs + 3,
                     fontweight="bold", ha="center", va="top",
                     family=ff, color="#2c3e50", zorder=2)

        facts_header_y = title_y - 0.75
        ax_text.text(0.6, facts_header_y, "Facts:", fontsize=base_fs + 1,
                     fontweight="bold", ha="left", va="top",
                     family=ff, color="#34495e", zorder=2)

        y = facts_header_y - 0.55
        dy = 0.53
        for i, fact in enumerate(facts):
            ax_text.text(0.9, y, f"{i + 1}. {fact}",
                         fontsize=base_fs, ha="left", va="top",
                         family=ff, color="#1a1a1a", zorder=2)
            y -= dy

        y -= 0.2
        ax_text.plot([0.8, 9.2], [y + 0.15, y + 0.15], color="#95a5a6",
                     linewidth=1.0, zorder=2)
        y -= 0.35
        ax_text.text(0.6, y, "Possible conclusions:",
                     fontsize=base_fs + 1, fontweight="bold",
                     ha="left", va="top", family=ff, color="#34495e",
                     zorder=2)
        y -= dy
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            ax_text.text(0.9, y, f"({letter}) {opt}",
                         fontsize=base_fs, ha="left", va="top",
                         family=ff, color="#1a1a1a", zorder=2)
            y -= dy

        ax_diag.set_xlim(0, 10)
        ax_diag.set_ylim(0, 10)
        ax_diag.set_aspect("equal")
        ax_diag.axis("off")
        ax_diag.set_title("Containment", fontsize=base_fs + 1,
                          fontweight="bold", family=ff)

        n_boxes = len(labels)
        outer_size = 9.2
        step = outer_size * 0.9 / max(n_boxes, 1) * 0.5
        for i in range(n_boxes):
            size = outer_size - i * step
            lx = (10 - size) / 2
            ly = (10 - size) / 2
            col = _BOX_COLORS[colors[i]]
            rect = mpatches.Rectangle(
                (lx, ly), size, size,
                facecolor=col, edgecolor="#2c3e50",
                linewidth=1.5, alpha=0.20 + 0.07 * i, zorder=2 + i)
            ax_diag.add_patch(rect)
            ax_diag.text(lx + 0.15, ly + size - 0.1,
                         f"{labels[i]}",
                         fontsize=base_fs - 1, fontweight="bold",
                         ha="left", va="top", family=ff,
                         color="#1a1a1a", zorder=5 + i)

        ax_diag.scatter([5.0], [5.0], s=70, color="#1a1a1a",
                        zorder=20)
        ax_diag.text(5.0, 4.5, item, fontsize=base_fs,
                     ha="center", va="top", family=ff,
                     color="#1a1a1a", zorder=20)

        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03,
                            wspace=0.08)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = MultiPremiseDeductionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer}")
