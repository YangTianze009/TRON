"""
Multi-table Join QA — redesign 2026-04-16.

Two tables shown side-by-side; task = lookup + compute across tables.

DIVERSITY:
  1. 5 SCENARIO FAMILIES (fruits, electronics, books, pets, stationery).
  2. 8 item-name pools rotated per seed.
  3. 5 column schemas (price+qty, price+stock, cost+units,
     revenue+count, unit_cost+quantity).
  4. Per-seed color palette shuffle and header style rotation.
  5. 6 question phrasings per qtype.
  6. Row count varies 3-6 with level.
  7. Table styling (row-striped, header-colored, compact vs. spread).

DIFFICULTY:
  L0: "Which item has highest price?" — ONE-table lookup (easy).
  L1: "Which item has the highest quantity?" — ONE-table lookup.
  L2: sum of one column.
  L3: total spending = sum(price * qty).  — JOIN row-wise.
  L4: most_expensive_total = argmax(price*qty).
  L5: median filter (price above median).
  L6: filter_and_sum (qty > threshold).
  L7: double filter + argmax.
  L8: weighted avg.
  L9: complex multi-filter.
"""
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SCENARIOS = {
    "fruits": {
        "items": ["Apple", "Banana", "Cherry", "Date", "Fig", "Grape",
                  "Kiwi", "Lemon", "Mango", "Peach"],
        "col1": ("Price", "Price ($)"),
        "col2": ("Quantity", "Quantity"),
        "unit": "$",
    },
    "electronics": {
        "items": ["Laptop", "Tablet", "Phone", "Camera", "Router",
                  "Speaker", "Monitor", "Keyboard", "Mouse", "Headset"],
        "col1": ("UnitCost", "Unit Cost ($)"),
        "col2": ("Stock", "Stock"),
        "unit": "$",
    },
    "books": {
        "items": ["Atlas", "Novel", "Manual", "Textbook", "Journal",
                  "Magazine", "Dictionary", "Cookbook", "Biography", "Anthology"],
        "col1": ("Price", "Price ($)"),
        "col2": ("Copies", "Copies"),
        "unit": "$",
    },
    "pets": {
        "items": ["Dog", "Cat", "Parrot", "Rabbit", "Hamster",
                  "Fish", "Turtle", "Snake", "Lizard", "Ferret"],
        "col1": ("Care", "Care ($/mo)"),
        "col2": ("Count", "Count"),
        "unit": "$",
    },
    "stationery": {
        "items": ["Pen", "Pencil", "Eraser", "Ruler", "Notebook",
                  "Marker", "Folder", "Stapler", "Tape", "Glue"],
        "col1": ("Cost", "Cost ($)"),
        "col2": ("Units", "Units"),
        "unit": "$",
    },
}

_THREE_TABLE_QUESTION_TEMPLATES = {
    "three_table_filter_sum": [
        "Three stores are shown. Among items whose {col1_name} is strictly above the threshold printed in the image, sum ({col1_name} * {col2_name}) ACROSS ALL THREE stores. Answer with an integer.",
        "Three stores A/B/C each have an inventory table. For items whose {col1_name} exceeds the threshold, compute sum({col1_name}*{col2_name}) aggregated over all three stores. Answer with an integer.",
    ],
    "three_table_argmax_store": [
        "Three stores are shown. For each store compute sum({col1_name}*{col2_name}) over items whose {col2_name} > threshold (shown in image). Which store has the LARGEST total? Answer with just the store label ('A', 'B', or 'C').",
        "Three stores displayed. Per store, sum ({col1_name}*{col2_name}) over items with {col2_name} > threshold. Return the label of the store whose total is the largest.",
    ],
    "five_table_having": [
        "Five stores A/B/C/D/E are shown. Some items are missing from "
        "specific stores (cells marked '—'); missing entries contribute 0. "
        "STEP 1: for each store compute sum({col1_name}*{col2_name}) across "
        "ALL items that appear in that store. STEP 2: filter to stores "
        "whose STEP-1 total is STRICTLY GREATER than the HAVING threshold "
        "shown on the image. STEP 3: among the remaining stores, SUM the "
        "STEP-1 totals. Report the final integer.",
        "Five stores are displayed, each with a partial inventory (items "
        "absent from a store are shown as '—' and count as 0). For each "
        "store, compute sum({col1_name}*{col2_name}). Keep only the stores "
        "whose computed total is STRICTLY ABOVE the HAVING threshold "
        "printed in the image. Sum the surviving stores' totals. Return "
        "the integer.",
    ],
}

_QUESTION_TEMPLATES = {
    **_THREE_TABLE_QUESTION_TEMPLATES,
    "max_price": [
        "Which item has the highest {col1_name}? Answer with the item name.",
        "Read the left table. Which item has the largest {col1_name}? Answer with the name.",
        "According to the left (price) table, which item is the most expensive? Answer with the item name.",
    ],
    "max_qty": [
        "Which item has the highest {col2_name}? Answer with the item name.",
        "In the right table, which item has the largest {col2_name}? Answer with the name.",
        "Which item appears in greatest quantity? Answer with the item name.",
    ],
    "total_spending": [
        "What is the grand total spending across all items (sum of {col1_name} times {col2_name} for every item)? Answer with an integer.",
        "Compute the sum of ({col1_name} * {col2_name}) over all items. Answer as a single integer.",
        "Grand total = sum over all items of ({col1_name} x {col2_name}). Compute and answer as an integer.",
    ],
    "most_expensive_total": [
        "Which item has the highest total ({col1_name} * {col2_name})? Answer with the item name.",
        "For each item, compute {col1_name} * {col2_name}. Which item has the largest product? Answer with the item name.",
        "Pick the item whose {col1_name}*{col2_name} product is the largest. Answer with the name.",
    ],
    "total_col1": [
        "What is the sum of the {col1_name} column across all items? Answer with an integer.",
        "Add up every value in the LEFT table. What is the sum? Answer with an integer.",
        "Compute total {col1_name} across all items. Answer with a single integer.",
    ],
    "most_expensive_by_quantity": [
        "Among items whose {col1_name} is at or above the median {col1_name}, which has the highest {col2_name}? Answer with the item name.",
        "Restrict to items with {col1_name} >= median. Among those, which has the largest {col2_name}? Answer with the name.",
        "Filter: {col1_name} >= median value. Pick argmax of {col2_name}. Answer with the item name.",
    ],
    "filter_and_sum": [
        "What is the total ({col1_name} * {col2_name}) for items whose {col2_name} is strictly greater than the threshold shown? Answer with an integer.",
        "For items with {col2_name} > threshold (shown in image), compute sum({col1_name}*{col2_name}). Answer with an integer.",
        "Among items with {col2_name} exceeding the threshold, sum the product of the two columns. Answer with an integer.",
    ],
    "double_filter_argmax": [
        "Among items whose {col1_name} is above the {col1_name} median AND {col2_name} exceeds the threshold shown, which has the highest {col2_name}? Answer with the item name.",
        "Apply two filters: {col1_name} > median AND {col2_name} > threshold. Among surviving items, pick the one with the largest {col2_name}. Answer with the name.",
    ],
    "weighted_avg": [
        "Compute the weighted average of {col1_name} using {col2_name} as weights (i.e., sum(p*q)/sum(q)). Round to the nearest integer.",
        "Take {col2_name} as weights. Compute the weighted mean of {col1_name}, rounded to the nearest integer.",
    ],
    "complex_multi_filter": [
        "Among items whose {col1_name} is strictly above the threshold shown AND {col2_name} is strictly below the upper cap shown, compute the total ({col1_name} * {col2_name}). Answer with an integer.",
        "Keep only items with {col1_name} > threshold AND {col2_name} < cap (both shown in image). For those, compute sum({col1_name}*{col2_name}). Answer with an integer.",
    ],
}

class MultiTableJoinQA(StandaloneVisualEnv):
    ENV_NAME = "multi_table_join"

    # 2026-05-04 R4: full-gradient redesign per chartqapro multi-table samples.
    # Original gradient was monotonic qtype change but n_items barely moved
    # (3,3,4,4,4,5,5,6,6,4) — saturated because L7 (6 items + double_filter)
    # is no harder than L4 (4 items + product). Real progressive gradient:
    #   L0/L1: 1-table lookup (max in column)
    #   L2: 1-table sum
    #   L3: 2-table join row-wise sum
    #   L4: 2-table argmax of products
    #   L5: median filter
    #   L6: filter+sum with threshold
    #   L7: double filter + argmax (more items + decoy tables)
    #   L8: weighted avg
    #   L9: 5-table HAVING with NULL handling
    # Each level adds either ITEMS or DECOY TABLES or new operator.
    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        mapping = [
            {"qtype": "max_price",           "n_items": 3, "n_decoys": 0},  # L0
            {"qtype": "max_qty",             "n_items": 4, "n_decoys": 0},  # L1
            {"qtype": "total_col1",          "n_items": 5, "n_decoys": 0},  # L2
            {"qtype": "total_spending",      "n_items": 5, "n_decoys": 0},  # L3
            {"qtype": "most_expensive_total", "n_items": 6, "n_decoys": 0}, # L4
            {"qtype": "most_expensive_by_quantity", "n_items": 6,
                                                       "n_decoys": 1},     # L5
            {"qtype": "filter_and_sum",      "n_items": 7, "n_decoys": 1},  # L6
            {"qtype": "double_filter_argmax", "n_items": 7, "n_decoys": 2}, # L7
            {"qtype": "weighted_avg",        "n_items": 8, "n_decoys": 2},  # L8
            # L9: 5 stores with LEFT-JOIN nullability + HAVING clause.
            {"qtype": "five_table_having", "n_items": 6, "n_decoys": 3},   # L9
        ]
        cfg = mapping[level]
        cfg["three_table_mode"] = False
        cfg["five_table_mode"] = (level == 9)
        return cfg

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        qtype = parameter.get("question_type") or lcfg["qtype"]
        n_items = lcfg["n_items"]

        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 5553)
        self._n_decoys = lcfg.get("n_decoys", 0)
        for _ in range(30):
            result = self._try_generate(qtype, n_items, rng, level)
            if result is not None:
                return result
        return None

    def _try_generate(self, qtype: str, n_items: int,
                      rng: random.Random, level: int):
        scenario_name = rng.choice(list(_SCENARIOS.keys()))
        scenario = _SCENARIOS[scenario_name]
        pool = list(scenario["items"])
        rng.shuffle(pool)
        items = pool[:n_items]
        col1_key, col1_display = scenario["col1"]
        col2_key, col2_display = scenario["col2"]

        if level <= 2:
            p_lo, p_hi, q_lo, q_hi = 2, 15, 1, 20
        elif level <= 5:
            p_lo, p_hi, q_lo, q_hi = 2, 25, 1, 30
        else:
            p_lo, p_hi, q_lo, q_hi = 3, 40, 1, 40

        prices = {it: rng.randint(p_lo, p_hi) for it in items}
        qtys = {it: rng.randint(q_lo, q_hi) for it in items}

        threshold = None
        cap = None

        lcfg = self._level_config(level)
        # L9 iter-3: 5-table path with nullable joins + HAVING clause.
        if lcfg.get("five_table_mode") and qtype == "five_table_having":
            return self._five_table_having_path(
                scenario_name, scenario, col1_display, col2_display,
                n_items, p_lo, p_hi, q_lo, q_hi, rng)
        # L9 three-table path: build three independent stores A/B/C and
        # aggregate across them.
        three_table_mode = False
        if lcfg.get("three_table_mode") and qtype in (
                "three_table_filter_sum", "three_table_argmax_store"):
            three_table_mode = True
            qtype_chosen = rng.choice(["three_table_filter_sum",
                                        "three_table_argmax_store"])
            # Build three distinct stores with distinct items each
            stores = {}
            pool2 = list(scenario["items"])
            rng.shuffle(pool2)
            # Ensure each store gets n_items (may reuse items across stores).
            for idx, label in enumerate(["A", "B", "C"]):
                st_items = (pool2 + pool2)[idx * n_items:(idx + 1) * n_items]
                st_prices = {it: rng.randint(p_lo, p_hi) for it in st_items}
                st_qtys = {it: rng.randint(q_lo, q_hi) for it in st_items}
                stores[label] = {
                    "items": st_items,
                    "prices": st_prices,
                    "qtys": st_qtys,
                }
            # filter threshold
            if qtype_chosen == "three_table_filter_sum":
                thr = rng.choice([5, 8, 10, 12, 15])
                # Ensure filter is non-trivial for at least one store
                total = 0
                any_match = False
                for st in stores.values():
                    for it in st["items"]:
                        if st["prices"][it] > thr:
                            total += st["prices"][it] * st["qtys"][it]
                            any_match = True
                if not any_match:
                    # bump some prices
                    for st in stores.values():
                        some_it = rng.choice(st["items"])
                        st["prices"][some_it] = thr + rng.randint(2, 10)
                    total = 0
                    for st in stores.values():
                        for it in st["items"]:
                            if st["prices"][it] > thr:
                                total += st["prices"][it] * st["qtys"][it]
                answer = str(total)
                threshold = thr
                # Use col1 for threshold wording (price threshold).
                q_template = rng.choice(
                    _QUESTION_TEMPLATES["three_table_filter_sum"])
                question_text = q_template.format(
                    col1_name=col1_display, col2_name=col2_display)
            else:
                thr = rng.choice([5, 8, 10, 12, 15])
                # Per-store totals over qty-filter
                per_store = {}
                for lbl, st in stores.items():
                    per_store[lbl] = sum(
                        st["prices"][it] * st["qtys"][it]
                        for it in st["items"] if st["qtys"][it] > thr)
                if all(v == 0 for v in per_store.values()):
                    # ensure at least one store matches; adjust
                    for lbl, st in stores.items():
                        some_it = rng.choice(st["items"])
                        st["qtys"][some_it] = thr + rng.randint(2, 10)
                    per_store = {lbl: sum(
                        st["prices"][it] * st["qtys"][it]
                        for it in st["items"] if st["qtys"][it] > thr)
                        for lbl, st in stores.items()}
                # ensure unique max
                if list(per_store.values()).count(max(per_store.values())) > 1:
                    # bump one store
                    top_lbl = max(per_store, key=per_store.get)
                    some_it = rng.choice(stores[top_lbl]["items"])
                    stores[top_lbl]["prices"][some_it] += 15
                    per_store = {lbl: sum(
                        st["prices"][it] * st["qtys"][it]
                        for it in st["items"] if st["qtys"][it] > thr)
                        for lbl, st in stores.items()}
                answer = max(per_store, key=per_store.get)
                threshold = thr
                q_template = rng.choice(
                    _QUESTION_TEMPLATES["three_table_argmax_store"])
                question_text = q_template.format(
                    col1_name=col1_display, col2_name=col2_display)
            # Render three tables side-by-side (as decoy_tables but all targets)
            decoy_tables_for_render = [
                {"label": f"Store {lbl}",
                 "items": stores[lbl]["items"],
                 "prices": stores[lbl]["prices"],
                 "qtys": stores[lbl]["qtys"]}
                for lbl in ["A", "B", "C"]
            ]
            # Render all 3 tables — we use the "Store A" slot as main
            # (unhighlighted) and push B, C as decoys.
            img = self._render_three_stores(
                stores, col1_display, col2_display, scenario_name, rng,
                threshold=threshold)
            question_text = (
                f"{question_text} Three stores are shown in the image, "
                f"labeled 'Store A', 'Store B', 'Store C'; threshold "
                f"is {threshold}. All three tables are REAL data.")
            return question_text, answer, img

        if qtype == "max_price":
            best = max(prices, key=prices.get)
            if [prices[it] for it in items].count(prices[best]) > 1:
                prices[best] += 3
                best = max(prices, key=prices.get)
            answer = best

        elif qtype == "max_qty":
            best = max(qtys, key=qtys.get)
            if [qtys[it] for it in items].count(qtys[best]) > 1:
                qtys[best] += 3
                best = max(qtys, key=qtys.get)
            answer = best

        elif qtype == "total_col1":
            answer = str(sum(prices.values()))

        elif qtype == "total_spending":
            answer = str(sum(prices[i] * qtys[i] for i in items))

        elif qtype == "most_expensive_total":
            totals = {i: prices[i] * qtys[i] for i in items}
            if list(totals.values()).count(max(totals.values())) > 1:
                idx_item = rng.choice(items)
                prices[idx_item] += 5
                totals = {i: prices[i] * qtys[i] for i in items}
            best = max(totals, key=totals.get)
            answer = best

        elif qtype == "most_expensive_by_quantity":
            sorted_prices = sorted(prices.values())
            median_price = sorted_prices[len(sorted_prices) // 2]
            expensive_items = {i: qtys[i] for i in items if prices[i] >= median_price}
            if not expensive_items:
                return None
            vals = list(expensive_items.values())
            if vals.count(max(vals)) > 1:
                # bump one
                key = list(expensive_items.keys())[0]
                qtys[key] += 5
                expensive_items = {i: qtys[i] for i in items if prices[i] >= median_price}
            best = max(expensive_items, key=expensive_items.get)
            answer = best

        elif qtype == "filter_and_sum":
            threshold = rng.choice([5, 8, 10, 12])
            filtered = [i for i in items if qtys[i] > threshold]
            if len(filtered) < 1:
                # bump some qtys so the filter is non-trivial
                for _ in range(2):
                    key = rng.choice(items)
                    qtys[key] = threshold + rng.randint(1, 10)
                filtered = [i for i in items if qtys[i] > threshold]
                if not filtered:
                    return None
            answer = str(sum(prices[i] * qtys[i] for i in filtered))

        elif qtype == "double_filter_argmax":
            median_price = sorted(prices.values())[len(items) // 2]
            threshold = rng.choice([5, 8, 10])
            both = [i for i in items if prices[i] > median_price and qtys[i] > threshold]
            if len(both) < 1:
                for _ in range(3):
                    k = rng.choice(items)
                    prices[k] = max(prices[k], median_price + 5)
                    qtys[k] = max(qtys[k], threshold + 5)
                median_price = sorted(prices.values())[len(items) // 2]
                both = [i for i in items if prices[i] > median_price and qtys[i] > threshold]
                if len(both) < 1:
                    return None
            sub_q = {i: qtys[i] for i in both}
            if list(sub_q.values()).count(max(sub_q.values())) > 1:
                k = list(sub_q.keys())[0]
                qtys[k] += 5
                sub_q = {i: qtys[i] for i in both}
            answer = max(sub_q, key=sub_q.get)

        elif qtype == "weighted_avg":
            denom = sum(qtys.values())
            if denom == 0:
                return None
            num = sum(prices[i] * qtys[i] for i in items)
            answer = str(round(num / denom))

        elif qtype == "complex_multi_filter":
            threshold = rng.choice([3, 5, 8])
            cap = rng.choice([25, 30, 35])
            survivors = [i for i in items
                         if prices[i] > threshold and qtys[i] < cap]
            if len(survivors) < 1:
                # adjust
                for _ in range(3):
                    k = rng.choice(items)
                    prices[k] = max(prices[k], threshold + 2)
                    qtys[k] = min(qtys[k], cap - 2)
                survivors = [i for i in items
                             if prices[i] > threshold and qtys[i] < cap]
                if not survivors:
                    return None
            answer = str(sum(prices[i] * qtys[i] for i in survivors))
        else:
            return None

        # Build decoy tables (same schema, random values, different items)
        n_decoys = getattr(self, "_n_decoys", 0)
        decoy_tables = []
        target_label = None
        if n_decoys > 0:
            # Use a label like "Store A" (target), "Store B" (decoy), ...
            labels = ["Store A", "Store B", "Store C", "Store D"]
            rng.shuffle(labels)
            target_label = labels[0]
            # Decoys reuse the item pool but pick different items where
            # possible, with different price/qty distributions.
            for i in range(n_decoys):
                d_pool = [it for it in scenario["items"] if it not in items]
                rng.shuffle(d_pool)
                d_items = d_pool[:n_items] if len(d_pool) >= n_items \
                          else (d_pool + items)[:n_items]
                d_prices = {it: rng.randint(p_lo, p_hi) for it in d_items}
                d_qtys = {it: rng.randint(q_lo, q_hi) for it in d_items}
                decoy_tables.append({
                    "label": labels[i + 1],
                    "items": d_items,
                    "prices": d_prices,
                    "qtys": d_qtys,
                })

        # Render tables
        img = self._render_tables(items, prices, qtys, col1_display, col2_display,
                                  scenario_name, rng, threshold=threshold, cap=cap,
                                  decoy_tables=decoy_tables,
                                  target_label=target_label)
        q_template = rng.choice(_QUESTION_TEMPLATES[qtype])
        q = q_template.format(col1_name=col1_display, col2_name=col2_display)
        if target_label is not None:
            q = (q + f" IMPORTANT: the image contains MULTIPLE store tables; "
                 f"use ONLY '{target_label}' (highlighted in red) — the "
                 f"others are decoy stores.")
        return q, str(answer), img

    def _five_table_having_path(self, scenario_name, scenario,
                                 col1_display, col2_display,
                                 n_items, p_lo, p_hi, q_lo, q_hi, rng):
        """L9 iter-4 (2026-04-17): five stores A/B/C/D/E with nullable
        entries. Task now requires FIVE operations: (1) compute each
        store's sum(col1*col2) ignoring nulls; (2) filter stores by
        HAVING threshold; (3) filter items-within-surviving-stores by a
        PER-ITEM price threshold; (4) compute a RANKED top-K contribution
        over the remaining cells; (5) report the TOP-K sum. This is
        materially harder than the iter-3 HAVING-only task."""
        master_pool = list(scenario["items"])
        rng.shuffle(master_pool)
        master_items = master_pool[:n_items]
        store_labels = ["A", "B", "C", "D", "E"]
        # For each store, decide which items it carries (each present with
        # independent probability 0.65) and fill prices/qtys.
        stores = {}
        for lbl in store_labels:
            present = {}
            prices = {}
            qtys = {}
            for it in master_items:
                if rng.random() < 0.7:
                    present[it] = True
                    prices[it] = rng.randint(p_lo, p_hi)
                    qtys[it] = rng.randint(q_lo, q_hi)
                else:
                    present[it] = False
                    prices[it] = None
                    qtys[it] = None
            # Ensure every store has AT LEAST 1 item
            if not any(present.values()):
                it = rng.choice(master_items)
                present[it] = True
                prices[it] = rng.randint(p_lo, p_hi)
                qtys[it] = rng.randint(q_lo, q_hi)
            stores[lbl] = {
                "present": present,
                "prices": prices,
                "qtys": qtys,
            }
        # Compute per-store totals
        store_totals = {}
        for lbl in store_labels:
            st = stores[lbl]
            t = 0
            for it in master_items:
                if st["present"][it]:
                    t += st["prices"][it] * st["qtys"][it]
            store_totals[lbl] = t
        # Pick HAVING threshold near MEDIAN of totals, so some stores survive
        sorted_t = sorted(store_totals.values())
        median_t = sorted_t[len(sorted_t) // 2]
        having = median_t - rng.randint(0, 30)
        # Compute surviving stores
        survivors = [lbl for lbl, t in store_totals.items() if t > having]
        if not survivors:
            having = sorted_t[0] - 1
            survivors = [lbl for lbl, t in store_totals.items() if t > having]
        if not survivors:
            return None

        # Iter-4 hardening: add per-ITEM price threshold AND top-K sum.
        # Step 3: within surviving stores, keep only cells whose price
        #         strictly exceeds `item_price_threshold`.
        # Step 4: take the top-K cell revenues (price*qty) among
        #         those surviving cells, ACROSS all surviving stores.
        # Step 5: sum those top-K revenues and report.
        # 2026-05-04 R3: tighter L9 — bumped item_price_threshold and top_k
        # (was 100% saturated; previous R2 "n_decoys=3" was dead code since
        # five_table_having_path never renders decoys).
        item_price_threshold = rng.choice([14, 16, 18, 20])
        top_k = 5
        cell_revs = []
        for lbl in survivors:
            st = stores[lbl]
            for it in master_items:
                if (st["present"][it]
                        and st["prices"][it] > item_price_threshold):
                    cell_revs.append(st["prices"][it] * st["qtys"][it])
        if len(cell_revs) < top_k:
            # Too few cells pass — lower price threshold until we have enough.
            for fallback in (8, 5, 3, 1):
                item_price_threshold = fallback
                cell_revs = []
                for lbl in survivors:
                    st = stores[lbl]
                    for it in master_items:
                        if (st["present"][it]
                                and st["prices"][it]
                                > item_price_threshold):
                            cell_revs.append(
                                st["prices"][it] * st["qtys"][it])
                if len(cell_revs) >= top_k:
                    break
            if len(cell_revs) < top_k:
                return None
        cell_revs.sort(reverse=True)
        top_k_sum = sum(cell_revs[:top_k])
        final_sum = top_k_sum

        # Render
        img = self._render_five_stores(
            stores, master_items, col1_display, col2_display,
            scenario_name, rng, having=having,
            item_price_threshold=item_price_threshold, top_k=top_k)
        question_text = (
            f"Five stores labeled A/B/C/D/E are shown; missing cells are "
            f"drawn as '—' (treat as 0). Execute this FIVE-step pipeline:\n"
            f"(1) For each store, compute sum({col1_display}*{col2_display}) "
            f"over items present in that store.\n"
            f"(2) Keep only stores whose step-1 total is STRICTLY GREATER "
            f"than the HAVING threshold = {having}.\n"
            f"(3) Within surviving stores, keep only CELLS where "
            f"{col1_display} > {item_price_threshold}.\n"
            f"(4) Compute revenue = {col1_display}*{col2_display} for every "
            f"surviving cell, then rank them.\n"
            f"(5) SUM the top {top_k} revenues. Return this integer."
        )
        return question_text, str(final_sum), img

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Strict integer match (P6.6 audit). Base 5% tolerance falsely accepts
        e.g. 31.2 for GT=30 on this env's discrete-integer answers; override
        strict for integer GT, fall back to base for non-integer GT (e.g. when
        L0 returns a string label)."""
        import re as _re
        p = predicted.strip().lower().rstrip(".").replace(",", "")
        g = ground_truth.strip().lower().rstrip(".")
        if p == g:
            return True
        if not _re.match(r"^-?\d+$", g):
            return super()._check_answer(predicted, ground_truth)
        m = _re.search(r"-?\d+(?:\.\d+)?", p)
        if not m:
            return False
        try:
            v = float(m.group())
            if v != int(v):
                return False
            return int(v) == int(g)
        except (ValueError, TypeError):
            return False

    def _render_five_stores(self, stores, master_items, col1_display,
                             col2_display, scenario_name, rng, having=None,
                             item_price_threshold=None, top_k=None):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        store_labels = ["A", "B", "C", "D", "E"]
        fig, axes_arr = plt.subplots(
            5, 1, figsize=(9, 2.0 * 5))
        fig.patch.set_facecolor(style["bg_color"])
        for gi, lbl in enumerate(store_labels):
            ax = axes_arr[gi]
            ax.axis("off")
            st = stores[lbl]
            rows_text = []
            for it in master_items:
                if st["present"][it]:
                    rows_text.append([it,
                                       f"${st['prices'][it]}",
                                       str(st["qtys"][it])])
                else:
                    rows_text.append([it, "—", "—"])
            t = ax.table(
                cellText=rows_text,
                colLabels=["Item", col1_display, col2_display],
                cellLoc='center', loc='center')
            t.auto_set_font_size(False)
            t.set_fontsize(10)
            t.scale(1, 1.3)
            for (r, c), cell in t.get_celld().items():
                if r == 0:
                    cell.set_facecolor(palette[0])
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#eef2f7')
                cell.set_edgecolor('#7f8c8d')
                cell.set_linewidth(1.0)
            ax.set_title(f"Store {lbl}", fontsize=11,
                         fontweight="bold", color="#1a1a1a", pad=6)
        subtitle = f"Scenario: {scenario_name}"
        if having is not None:
            subtitle += f"    (HAVING > {having})"
        if item_price_threshold is not None:
            subtitle += f"   cell-price > {item_price_threshold}"
        if top_k is not None:
            subtitle += f"   TOP-{top_k}"
        fig.suptitle(subtitle, fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_three_stores(self, stores, col1_display, col2_display,
                              scenario_name, rng, threshold=None):
        """L9: render three real stores without decoy styling."""
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        labels = ["A", "B", "C"]
        max_items = max(len(stores[l]["items"]) for l in labels)
        fig, axes_arr = plt.subplots(
            3, 2, figsize=(10, max(3, max_items * 0.55 + 1.2) * 3))
        fig.patch.set_facecolor(style["bg_color"])
        for gi, lbl in enumerate(labels):
            ax1, ax2 = axes_arr[gi]
            ax1.axis("off"); ax2.axis("off")
            st = stores[lbl]
            t1 = ax1.table(
                cellText=[[it, f"${st['prices'][it]}"]
                           for it in st["items"]],
                colLabels=["Item", col1_display],
                cellLoc='center', loc='center')
            t1.auto_set_font_size(False)
            t1.set_fontsize(11); t1.scale(1, 1.5)
            for (r, c), cell in t1.get_celld().items():
                if r == 0:
                    cell.set_facecolor(palette[0])
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#eef2f7')
                cell.set_edgecolor('#7f8c8d')
                cell.set_linewidth(1.0)
            ax1.set_title(f"Store {lbl} — {col1_display}",
                          fontsize=12, fontweight="bold",
                          color="#1a1a1a", pad=15)
            t2 = ax2.table(
                cellText=[[it, str(st["qtys"][it])] for it in st["items"]],
                colLabels=["Item", col2_display],
                cellLoc='center', loc='center')
            t2.auto_set_font_size(False)
            t2.set_fontsize(11); t2.scale(1, 1.5)
            for (r, c), cell in t2.get_celld().items():
                if r == 0:
                    cell.set_facecolor(palette[1 % len(palette)])
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#eef2f7')
                cell.set_edgecolor('#7f8c8d')
                cell.set_linewidth(1.0)
            ax2.set_title(f"Store {lbl} — {col2_display}",
                          fontsize=12, fontweight="bold",
                          color="#1a1a1a", pad=15)
        subtitle = f"Scenario: {scenario_name}"
        if threshold is not None:
            subtitle += f"    (threshold = {threshold})"
        fig.suptitle(subtitle, fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _render_tables(self, items, prices, qtys, col1_display, col2_display,
                       scenario_name, rng, threshold=None, cap=None,
                       decoy_tables=None, target_label=None):
        style = self._random_style()
        palette = list(style["palette"])
        rng.shuffle(palette)
        n_extra_rows = 0
        if threshold is not None:
            n_extra_rows += 1
        if cap is not None:
            n_extra_rows += 1
        decoy_tables = decoy_tables or []
        n_groups = 1 + len(decoy_tables)
        fig_height = max(3, len(items) * 0.6 + 1 + 0.5 * n_extra_rows) * n_groups
        fig, axes_arr = plt.subplots(
            n_groups, 2, figsize=(10, fig_height))
        if n_groups == 1:
            axes_arr = [axes_arr]
        fig.patch.set_facecolor(style["bg_color"])

        # Randomize placement of the target group
        target_row_idx = rng.randint(0, n_groups - 1)
        group_data = []
        # Target group
        group_data.append({
            "label": target_label or "Main",
            "items": items,
            "prices": prices,
            "qtys": qtys,
            "is_target": True,
        })
        for dt in decoy_tables:
            group_data.append({
                "label": dt["label"],
                "items": dt["items"],
                "prices": dt["prices"],
                "qtys": dt["qtys"],
                "is_target": False,
            })
        # Reorder so target sits at target_row_idx
        decoy_only = group_data[1:]
        ordered = [None] * n_groups
        ordered[target_row_idx] = group_data[0]
        dec_iter = iter(decoy_only)
        for si in range(n_groups):
            if ordered[si] is None:
                ordered[si] = next(dec_iter)

        for gi, grp in enumerate(ordered):
            ax1, ax2 = axes_arr[gi]
            ax1.axis("off")
            ax2.axis("off")
            its = grp["items"]
            pr = grp["prices"]
            qt = grp["qtys"]
            hl = grp["is_target"] and target_label is not None
            edge_c = "#b71c1c" if hl else "#7f8c8d"
            edge_lw = 2.4 if hl else 1.0
            title_c = "#b71c1c" if hl else "#1a1a1a"

            t1 = ax1.table(
                cellText=[[it, f"${pr[it]}"] for it in its],
                colLabels=["Item", col1_display],
                cellLoc='center', loc='center')
            t1.auto_set_font_size(False)
            t1.set_fontsize(11)
            t1.scale(1, 1.5)
            for (r, c), cell in t1.get_celld().items():
                if r == 0:
                    cell.set_facecolor(palette[0])
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#eef2f7')
                cell.set_edgecolor(edge_c)
                cell.set_linewidth(edge_lw)
            ax1.set_title(f"{grp['label']} — {col1_display}",
                          fontsize=12, fontweight="bold",
                          color=title_c, pad=15)

            t2 = ax2.table(
                cellText=[[it, str(qt[it])] for it in its],
                colLabels=["Item", col2_display],
                cellLoc='center', loc='center')
            t2.auto_set_font_size(False)
            t2.set_fontsize(11)
            t2.scale(1, 1.5)
            for (r, c), cell in t2.get_celld().items():
                if r == 0:
                    cell.set_facecolor(palette[1 % len(palette)])
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#f8f9fa' if r % 2 == 0 else '#eef2f7')
                cell.set_edgecolor(edge_c)
                cell.set_linewidth(edge_lw)
            ax2.set_title(f"{grp['label']} — {col2_display}",
                          fontsize=12, fontweight="bold",
                          color=title_c, pad=15)

        extra_txt = []
        if threshold is not None:
            extra_txt.append(f"threshold = {threshold}")
        if cap is not None:
            extra_txt.append(f"cap = {cap}")
        subtitle = f"Scenario: {scenario_name}"
        if extra_txt:
            subtitle += f"    ({', '.join(extra_txt)})"
        fig.suptitle(subtitle, fontsize=12, fontweight="bold", y=0.995)
        plt.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = MultiTableJoinQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer}")
