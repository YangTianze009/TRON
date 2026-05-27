"""
Price Tag QA (v4 G9c, for VQA / arithmetic reasoning on photos).

Targets: general VQA -4.47 (idx=49 blank sign, arithmetic-on-sign).

Task: render a grid of price tags (item name + price), composite on a
photo background (shelf/store). Ask for sum of 2-3 specific items or
change from a payment.

Reward: numeric within 0.01.

Level axes:
  A) Number of items: 3 at L0-3, 5 at L4-6, 7 at L7+
  B) Question type: single lookup at L0, sum at L3, sum+change at L6+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_ITEMS_POOL = ["Apple", "Banana", "Bread", "Butter", "Cheese", "Coffee",
                 "Cookies", "Cream", "Donut", "Eggs", "Fish", "Grapes",
                 "Honey", "Juice", "Ketchup", "Lemon", "Milk", "Noodles",
                 "Orange", "Pasta", "Pizza", "Rice", "Salad", "Soup",
                 "Tea", "Tomato", "Water", "Yogurt"]

_TEMPLATES_SINGLE = [
    "The photograph shows price tags. What is the price of {item}? Put the number in <answer>...</answer>.",
    "Read the price of {item} from the tags. Number in <answer>...</answer>.",
    "Price of {item}? Put in <answer>...</answer>.",
    "What does {item} cost? Put the price in <answer>...</answer>.",
    "Identify {item}'s price from the tags. Put in <answer>...</answer>.",
    "How much is {item}? Put the price in <answer>...</answer>.",
    "{item} costs how much? Put in <answer>...</answer>.",
    "From the price tags, read {item}'s price. Put in <answer>...</answer>.",
    "What's the price of {item}? Put in <answer>...</answer>.",
    "Read {item}'s price. Put in <answer>...</answer>.",
    "Price tag: {item} = ? Put in <answer>...</answer>.",
    "Find {item}'s price from the tags. Put in <answer>...</answer>.",
    "Identify the price of {item}. Put in <answer>...</answer>.",
    "{item} price? Put in <answer>...</answer>.",
    "Read off {item}'s price from the tag display. Put in <answer>...</answer>.",
    "Look at the price tags and give {item}'s price. Put in <answer>...</answer>.",
]

_TEMPLATES_SUM = [
    "The photograph shows price tags. What is the total cost of {items}? Put the sum (numeric) in <answer>...</answer>.",
    "Compute the total price for {items}. Put sum in <answer>...</answer>.",
    "Add up prices of {items}. Put total in <answer>...</answer>.",
    "Total for {items}? Put in <answer>...</answer>.",
    "Sum the prices of {items}. Put in <answer>...</answer>.",
    "What's the total of {items}? Put in <answer>...</answer>.",
    "Combined cost of {items}? Put in <answer>...</answer>.",
    "Compute sum: {items}. Put in <answer>...</answer>.",
    "Add the prices of {items}. Put in <answer>...</answer>.",
    "Total price for {items}? Put in <answer>...</answer>.",
    "Sum of {items} prices? Put in <answer>...</answer>.",
    "{items} together cost? Put in <answer>...</answer>.",
    "What's {items} combined? Put in <answer>...</answer>.",
    "Price sum of {items}? Put in <answer>...</answer>.",
    "Compute total: {items}. Put in <answer>...</answer>.",
    "Sum the prices of the items {items}. Put in <answer>...</answer>.",
]

# 2026-05-04 R4: full-gradient redesign per general-VQA arithmetic-on-sign.
# Compound mode (L6-L9): buy a list of items, pay $X, return CHANGE.
# Forces: (1) read N price tags, (2) sum them, (3) subtract from payment.
_TEMPLATES_CHANGE = [
    "From the price tags, you buy {items} and pay ${pay:.2f}. What is the change you receive? Put the numeric change in <answer>...</answer>.",
    "You purchase {items} (read prices from the tags) and hand the cashier ${pay:.2f}. Compute the change. Put it in <answer>...</answer>.",
    "Buying {items} from the displayed price tags and paying ${pay:.2f}, what change do you get back? Put the value in <answer>...</answer>.",
    "Items bought: {items}. Payment: ${pay:.2f}. Compute the change. Put in <answer>...</answer>.",
    "After buying {items} from the tags shown and paying ${pay:.2f}, what is the change? Put the numeric answer in <answer>...</answer>.",
]

class PriceTagQA(StandaloneVisualEnv):
    ENV_NAME = "price_tag"
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R4: full-gradient redesign per general-VQA arithmetic.
        # L0-L1: 3 items, single lookup
        # L2-L3: 4-5 items, single lookup
        # L4-L5: 5-6 items, sum 2-3 items
        # L6-L7: 7 items, sum-then-change-from-payment (3-step)
        # L8-L9: 9 items, sum 4-5 items then change
        level = max(0, min(level, 9))
        if level <= 1:
            return {"n_items": 3, "qtype": "single", "level": level}
        if level <= 3:
            return {"n_items": 4 + (level - 2),  # L2=4, L3=5
                    "qtype": "single", "level": level}
        if level <= 5:
            return {"n_items": 5 + (level - 4),  # L4=5, L5=6
                    "qtype": "sum", "level": level}
        if level <= 7:
            return {"n_items": 7, "qtype": "change", "level": level}
        # L8-L9: more items, more sum-terms before change
        return {"n_items": 9, "qtype": "change", "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 523)
        self._primary_complexity_feature = level

        n = cfg["n_items"]
        items = rng.sample(_ITEMS_POOL, n)
        prices = [round(rng.uniform(1.50, 12.99), 2) for _ in range(n)]

        if cfg["qtype"] == "single":
            target_idx = rng.randint(0, n - 1)
            answer = f"{prices[target_idx]:.2f}"
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_SINGLE[sidx].format(item=items[target_idx])
        elif cfg["qtype"] == "sum":
            # 2026-05-04 R4: L4-L5 sum 2-3 items
            k = rng.randint(2, min(3, n))
            targets = rng.sample(range(n), k)
            target_items = ", ".join(items[i] for i in targets)
            total = round(sum(prices[i] for i in targets), 2)
            answer = f"{total:.2f}"
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_SUM[sidx].format(items=target_items)
        else:
            # 2026-05-04 R4: change-from-payment. L6-L7: 3 items; L8-L9: 4-5 items.
            if cfg["level"] >= 8:
                k = rng.randint(4, min(5, n))
            else:
                k = 3
            targets = rng.sample(range(n), k)
            target_items = ", ".join(items[i] for i in targets)
            total = round(sum(prices[i] for i in targets), 2)
            # Pick payment >= total, rounded up to nearest $5
            import math as _m
            pay = max(_m.ceil(total / 5.0) * 5, total + 1)
            change = round(pay - total, 2)
            answer = f"{change:.2f}"
            sidx = (self.seed or 0) % len(_TEMPLATES_CHANGE)
            q = _TEMPLATES_CHANGE[sidx].format(items=target_items, pay=pay)

        img = self._render_tags(items, prices, rng)
        return q, answer, img

    def _render_tags(self, items, prices, rng):
        n = len(items)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, ax = plt.subplots(figsize=(cols * 2.5, rows * 1.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(0, cols); ax.set_ylim(0, rows)
        ax.set_aspect("equal"); ax.axis("off")

        for i, (item, price) in enumerate(zip(items, prices)):
            r = i // cols
            c = i % cols
            x, y = c, rows - 1 - r
            # Tag background
            ax.add_patch(mpatches.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8,
                                             fc="#fffddd", ec="black", lw=1.5))
            # item name
            ax.text(x + 0.5, y + 0.7, item, fontsize=12, ha="center",
                    fontweight="bold")
            # price
            ax.text(x + 0.5, y + 0.35, f"${price:.2f}", fontsize=16,
                    ha="center", fontweight="bold", color="#c0392b")
        return self.fig_to_pil(fig, dpi=130)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().replace("$", "").rstrip(".").rstrip()
        gt = ground_truth.strip().lower().replace("$", "").rstrip(".").rstrip()
        try:
            return abs(float(pred) - float(gt)) < 0.02
        except ValueError:
            return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_pt"
    os.makedirs(out_dir, exist_ok=True)
    env = PriceTagQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 83
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[pt L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/pt_s{s}_L{level}.png")
            print(f"[pt L{level} s{s}] A={env._answer}")
