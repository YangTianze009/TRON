"""
Food Web QA environment.

Capabilities: V3 (chart extraction), V2 (label reading), R5 (multi-step reasoning)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 3-node chain (plant → herbivore → carnivore). Ask "what does X eat".
L1: 3-node chain. Ask "what eats X".
L2: 4-node web (one branch). Ask "what does X eat".
L3: 4-5 node web. Ask "count predators of X".
L4: 5-node web. Ask "top predators".
L5: 5-6 node web. Ask "longest food chain length".
L6: 6-7 node web. Ask "count producers".
L7: 7 node web. Ask "if X removed, who loses food".
L8: 7-8 node web. Ask "top predators".
L9: 8-node web. Ask "longest food chain length".

parameter = {"level": int in [0, 9]}
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

# Tiny pools for L0-L1 (3-node chain)
_TINY_WEBS = [
    {"organisms": ["Grass", "Rabbit", "Fox"],
     "edges": [("Grass", "Rabbit"), ("Rabbit", "Fox")],
     "producers": ["Grass"]},
    {"organisms": ["Algae", "Fish", "Heron"],
     "edges": [("Algae", "Fish"), ("Fish", "Heron")],
     "producers": ["Algae"]},
    {"organisms": ["Seeds", "Mouse", "Owl"],
     "edges": [("Seeds", "Mouse"), ("Mouse", "Owl")],
     "producers": ["Seeds"]},
    {"organisms": ["Leaves", "Caterpillar", "Robin"],
     "edges": [("Leaves", "Caterpillar"), ("Caterpillar", "Robin")],
     "producers": ["Leaves"]},
    {"organisms": ["Plankton", "Krill", "Penguin"],
     "edges": [("Plankton", "Krill"), ("Krill", "Penguin")],
     "producers": ["Plankton"]},
]

# Small webs for L2-L3 (4-5 nodes)
_SMALL_WEBS = [
    {"organisms": ["Grass", "Rabbit", "Mouse", "Fox"],
     "edges": [("Grass", "Rabbit"), ("Grass", "Mouse"),
               ("Rabbit", "Fox"), ("Mouse", "Fox")],
     "producers": ["Grass"]},
    {"organisms": ["Algae", "Fish", "Snail", "Heron"],
     "edges": [("Algae", "Fish"), ("Algae", "Snail"),
               ("Fish", "Heron"), ("Snail", "Heron")],
     "producers": ["Algae"]},
    {"organisms": ["Seeds", "Bird", "Squirrel", "Cat", "Hawk"],
     "edges": [("Seeds", "Bird"), ("Seeds", "Squirrel"),
               ("Bird", "Cat"), ("Bird", "Hawk"),
               ("Squirrel", "Hawk")],
     "producers": ["Seeds"]},
    {"organisms": ["Plankton", "Krill", "Squid", "Penguin"],
     "edges": [("Plankton", "Krill"), ("Krill", "Squid"),
               ("Krill", "Penguin"), ("Squid", "Penguin")],
     "producers": ["Plankton"]},
]

# Large webs for L4+ (6-8 nodes)
_LARGE_WEBS = [
    {"organisms": ["Grass", "Rabbit", "Mouse", "Snake", "Hawk", "Fox", "Frog"],
     "edges": [("Grass", "Rabbit"), ("Grass", "Mouse"),
               ("Rabbit", "Fox"), ("Rabbit", "Hawk"),
               ("Mouse", "Snake"), ("Mouse", "Fox"),
               ("Snake", "Hawk"), ("Frog", "Snake"),
               ("Grass", "Frog")],
     "producers": ["Grass"]},
    {"organisms": ["Algae", "Zooplankton", "Small Fish", "Big Fish",
                   "Heron", "Snail", "Frog"],
     "edges": [("Algae", "Zooplankton"), ("Algae", "Snail"),
               ("Zooplankton", "Small Fish"), ("Small Fish", "Big Fish"),
               ("Small Fish", "Heron"), ("Big Fish", "Heron"),
               ("Snail", "Frog"), ("Frog", "Heron")],
     "producers": ["Algae"]},
    {"organisms": ["Seeds", "Berries", "Squirrel", "Bird", "Cat",
                   "Owl", "Snake", "Hawk"],
     "edges": [("Seeds", "Squirrel"), ("Seeds", "Bird"),
               ("Berries", "Bird"), ("Berries", "Squirrel"),
               ("Squirrel", "Cat"), ("Squirrel", "Hawk"),
               ("Bird", "Cat"), ("Bird", "Hawk"),
               ("Bird", "Snake"), ("Snake", "Owl"),
               ("Snake", "Hawk")],
     "producers": ["Seeds", "Berries"]},
    {"organisms": ["Phytoplankton", "Krill", "Penguin", "Seal",
                   "Orca", "Squid", "Fish"],
     "edges": [("Phytoplankton", "Krill"),
               ("Krill", "Penguin"), ("Krill", "Fish"), ("Krill", "Squid"),
               ("Fish", "Penguin"), ("Fish", "Seal"),
               ("Squid", "Seal"), ("Squid", "Penguin"),
               ("Penguin", "Orca"), ("Seal", "Orca")],
     "producers": ["Phytoplankton"]},
    {"organisms": ["Oak Tree", "Caterpillar", "Deer", "Robin",
                   "Spider", "Wolf", "Bear"],
     "edges": [("Oak Tree", "Caterpillar"), ("Oak Tree", "Deer"),
               ("Caterpillar", "Robin"), ("Caterpillar", "Spider"),
               ("Robin", "Wolf"), ("Deer", "Wolf"),
               ("Deer", "Bear"), ("Spider", "Robin")],
     "producers": ["Oak Tree"]},
]

_TITLE_VARIANTS = ["Food Web", "Ecosystem", "Trophic Network", "Species Web"]

class FoodWebQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "food_web"

    QUESTION_TYPES = [
        "what_does_x_eat",
        "what_eats_x",
        "count_prey",
        "top_predator",
        "food_chain_length",
        "what_happens_if_removed",
        "count_producers",
    ]

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(15):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _level_config(self, level: int) -> Dict:
        if level == 0:
            return {"pool": "tiny", "qtype": "what_does_x_eat"}
        if level == 1:
            return {"pool": "tiny", "qtype": "what_eats_x"}
        if level == 2:
            return {"pool": "small", "qtype": "what_does_x_eat"}
        if level == 3:
            return {"pool": "small", "qtype": "what_eats_x"}
        if level == 4:
            return {"pool": "small", "qtype": "top_predator"}
        if level == 5:
            return {"pool": "large", "qtype": "food_chain_length"}
        if level == 6:
            return {"pool": "large", "qtype": "what_eats_x"}
        if level == 7:
            return {"pool": "large", "qtype": "what_happens_if_removed"}
        if level == 8:
            return {"pool": "large", "qtype": "top_predator"}
        return {"pool": "large", "qtype": "what_does_x_eat"}

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)
        if cfg["pool"] == "tiny":
            web = rng.choice(_TINY_WEBS)
        elif cfg["pool"] == "small":
            web = rng.choice(_SMALL_WEBS)
        else:
            web = rng.choice(_LARGE_WEBS)

        organisms = list(web["organisms"])
        edges = list(web["edges"])
        producers = set(web["producers"])

        prey_to_pred = {o: set() for o in organisms}
        pred_to_prey = {o: set() for o in organisms}
        for prey, pred in edges:
            prey_to_pred[prey].add(pred)
            pred_to_prey[pred].add(prey)

        question, answer = self._make_qa(
            rng, cfg["qtype"], organisms, prey_to_pred, pred_to_prey,
            producers, edges)
        if question is None:
            return None
        image = self._render_food_web(rng, organisms, edges, producers)
        return question, str(answer), image

    def _make_qa(self, rng, qtype, organisms, prey_to_pred, pred_to_prey,
                 producers, edges):
        if qtype == "what_does_x_eat":
            cands = [o for o in organisms if pred_to_prey[o]]
            if not cands:
                return None, None
            tgt = rng.choice(cands)
            prey = sorted(pred_to_prey[tgt])
            return (f'According to the food web, what does "{tgt}" eat? '
                    f'Answer with the prey name(s), comma separated.',
                    ", ".join(prey))

        if qtype == "what_eats_x":
            cands = [o for o in organisms if prey_to_pred[o]]
            if not cands:
                return None, None
            tgt = rng.choice(cands)
            preds = sorted(prey_to_pred[tgt])
            return (f'According to the food web, what eats "{tgt}"? '
                    f'Answer with predator name(s), comma separated.',
                    ", ".join(preds))

        if qtype == "count_prey":
            cands = [o for o in organisms if pred_to_prey[o]]
            if not cands:
                return None, None
            tgt = rng.choice(cands)
            return (f'How many different organisms does "{tgt}" eat? '
                    f'Answer with a single integer.',
                    len(pred_to_prey[tgt]))

        if qtype == "top_predator":
            tops = sorted([o for o in organisms if not prey_to_pred[o]])
            if not tops:
                return None, None
            return ("Which organism(s) in the food web have no predators "
                    "(top predators)? List them, comma separated.",
                    ", ".join(tops))

        if qtype == "food_chain_length":
            adj = {o: [] for o in organisms}
            for prey, pred in edges:
                adj[prey].append(pred)
            max_len = 0
            for start in producers:
                stack = [(start, 0)]
                while stack:
                    node, depth = stack.pop()
                    if depth > max_len:
                        max_len = depth
                    for nxt in adj[node]:
                        stack.append((nxt, depth + 1))
            return ("What is the length of the longest food chain in the web? "
                    "Count the number of arrows from a producer to a top predator. "
                    "Answer with a single integer.", max_len)

        if qtype == "count_producers":
            return ("How many producers (organisms with no prey) are in the food web? "
                    "Answer with a single integer.", len(producers))

        if qtype == "what_happens_if_removed":
            mids = [o for o in organisms
                    if pred_to_prey[o] and prey_to_pred[o]]
            if not mids:
                return None, None
            tgt = rng.choice(mids)
            affected = sorted([o for o in prey_to_pred[tgt]
                                if pred_to_prey[o] == {tgt}])
            if not affected:
                return (f'If "{tgt}" were removed from the food web, would any predator '
                        f'lose its ONLY food source? Answer Yes or No.', "No")
            return (f'If "{tgt}" were removed, which predator(s) would lose their '
                    f'ONLY food source? List them comma separated.',
                    ", ".join(affected))
        return None, None

    def _render_food_web(self, rng, organisms, edges, producers):
        levels = self._compute_trophic_levels(organisms, edges, producers)
        max_level = max(levels.values()) if levels else 0

        by_level = {}
        for org, lv in levels.items():
            by_level.setdefault(lv, []).append(org)

        positions = {}
        for lv, orgs in by_level.items():
            y = 0.1 + 0.8 * lv / max(max_level, 1)
            n_at_level = len(orgs)
            for i, org in enumerate(orgs):
                x = 0.15 + 0.7 * i / max(n_at_level - 1, 1)
                x += rng.uniform(-0.04, 0.04)
                y_j = y + rng.uniform(-0.03, 0.03)
                positions[org] = (x, y_j)

        style = self._random_style()
        s = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * s, 7 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")

        palette = list(style["palette"])
        rng.shuffle(palette)
        edge_color = style["geo_line_color"]
        lw = style["line_width"]

        for prey, pred in edges:
            x1, y1 = positions[prey]
            x2, y2 = positions[pred]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                continue
            shrink = 0.045
            ax.annotate(
                "", xy=(x2 - shrink * dx / dist, y2 - shrink * dy / dist),
                xytext=(x1 + shrink * dx / dist, y1 + shrink * dy / dist),
                arrowprops=dict(arrowstyle="->", color=edge_color,
                                lw=lw, connectionstyle="arc3,rad=0.08"),
            )

        for i, org in enumerate(organisms):
            x, y = positions[org]
            color = palette[i % len(palette)]
            if org in producers:
                color = palette[0]
            circle = mpatches.Circle((x, y), 0.04, facecolor=color,
                                     edgecolor=edge_color, linewidth=lw, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y - 0.06, org, fontsize=style["font_size_base"] - 1,
                    ha="center", va="top",
                    fontweight="bold", color=edge_color,
                    fontfamily=style["font_family"],
                    bbox=dict(boxstyle="round,pad=0.15", fc=style["bg_color"],
                              ec="#cccccc", alpha=0.85),
                    zorder=6)

        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold", pad=10)
        ax.text(0.02, -0.02, "Arrow: prey → predator (eaten by)",
                fontsize=style["font_size_base"] - 3, color="#888888",
                transform=ax.transAxes)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    @staticmethod
    def _compute_trophic_levels(organisms, edges, producers):
        levels = {o: 0 for o in organisms}
        pred_to_prey = {o: set() for o in organisms}
        for prey, pred in edges:
            pred_to_prey[pred].add(prey)

        changed = True
        for _ in range(len(organisms)):
            if not changed:
                break
            changed = False
            for org in organisms:
                if org in producers:
                    continue
                if pred_to_prey[org]:
                    new_level = 1 + max(levels[p] for p in pred_to_prey[org])
                    if new_level != levels[org]:
                        levels[org] = new_level
                        changed = True
        return levels

if __name__ == "__main__":
    env = FoodWebQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
