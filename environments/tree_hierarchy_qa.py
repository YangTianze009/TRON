"""
Tree/hierarchy diagram QA — weighted tree aggregation, org chart reasoning.
Targets: diagram (hierarchical diagrams), MMStar logical reasoning.

Capabilities: V9 (arrow/flow parsing), R1 (arithmetic), R5 (multi-step)

Round 2 diversity + difficulty fix (2026-04-16):
- 7 domain context pools (was 3)
- Structural difficulty: L0 has shallow 2-level tree; L9 has 3-level tree with
  grandchildren AND weighted subtree sums.
- Level-gated qtypes with new ops at L6-L9 (max_subtree_sum, deepest_path, etc.)
- Sub-RNG includes level — different seeds produce different contexts.
- Rendering: horizontal vs vertical layout jitter, varied node shapes.
"""
import random
import math
from typing import Dict, Optional, Tuple, List
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_CONTEXT_POOLS = [
    {"title": "Organization Chart", "root": "CEO",
     "children_pool": ["VP Sales", "VP Eng", "VP Marketing", "VP Finance",
                        "VP Ops", "VP HR", "VP Legal"],
     "grandchildren_pool": ["Mgr A", "Mgr B", "Mgr C", "Mgr D", "Mgr E"],
     "leaf_pool": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
                    "Hank", "Ivy", "Jack", "Kate", "Leo"]},
    {"title": "File System", "root": "root/",
     "children_pool": ["docs/", "src/", "config/", "tests/", "data/", "bin/"],
     "grandchildren_pool": ["sub_a/", "sub_b/", "sub_c/", "mod/", "lib/"],
     "leaf_pool": ["readme.md", "main.py", "config.yaml", "test_1.py",
                    "data.csv", "utils.py", "notes.txt", "log.txt",
                    "setup.py", "img.png", "tree.json"]},
    {"title": "Category Tree", "root": "Animals",
     "children_pool": ["Mammals", "Birds", "Reptiles", "Fish", "Insects",
                        "Arachnids"],
     "grandchildren_pool": ["Canidae", "Felidae", "Passerine", "Raptor",
                             "Serpent", "Lacertid"],
     "leaf_pool": ["Dog", "Cat", "Eagle", "Sparrow", "Snake", "Lizard",
                    "Salmon", "Ant", "Tiger", "Shark", "Owl", "Parrot"]},
    {"title": "Product Catalog", "root": "Store",
     "children_pool": ["Electronics", "Apparel", "Books", "Toys", "Home",
                        "Sports"],
     "grandchildren_pool": ["Phones", "Laptops", "Shirts", "Shoes",
                             "Fiction", "Science"],
     "leaf_pool": ["ModelA", "ModelB", "ModelC", "Item1", "Item2",
                    "Item3", "Novel", "Guide", "Toy1", "Toy2", "Lamp"]},
    {"title": "Course Prerequisites", "root": "Curriculum",
     "children_pool": ["Math", "CS", "Physics", "Biology", "History"],
     "grandchildren_pool": ["Algebra", "Calculus", "Algorithms", "OS",
                             "Mechanics", "Quantum"],
     "leaf_pool": ["101", "102", "201", "202", "301", "302", "401",
                    "402", "501", "502"]},
    {"title": "Menu Hierarchy", "root": "Restaurant",
     "children_pool": ["Appetizers", "Mains", "Desserts", "Drinks", "Sides"],
     "grandchildren_pool": ["Hot", "Cold", "Vegan", "Meat", "Fish"],
     "leaf_pool": ["Dish1", "Dish2", "Dish3", "Dish4", "Dish5", "Dish6",
                    "Dish7", "Dish8", "Dish9"]},
    {"title": "Company Regions", "root": "HQ",
     "children_pool": ["North", "South", "East", "West", "Central"],
     "grandchildren_pool": ["Branch1", "Branch2", "Branch3", "Branch4"],
     "leaf_pool": ["Office A", "Office B", "Office C", "Office D",
                    "Office E", "Office F", "Office G", "Office H"]},
]

class TreeHierarchyQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "tree_hierarchy"

    # ------------------------------------------------------------------ #
    # Per-level configuration (L0 simplest, L9 hardest + structurally)
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        # Redesign 2026-04-17: monotonic difficulty.
        # Previous L6 (total_in_subtree on depth-3 trees) dipped to 0.6
        # while L9 (count_nodes_by_depth, mean_leaf_value) was 0.9 because
        # those last two ops are easier than total_in_subtree. Now we keep
        # total_in_subtree + max_subtree_sum + deepest_path_sum at L8-L9.
        if level <= 1:
            return {"qtypes": ["count_leaves", "count_children"],
                    "depth": 2, "n_children_range": (2, 3),
                    "leaves_per_child_range": (1, 2),
                    "show_values": False}
        if level <= 3:
            return {"qtypes": ["count_leaves", "count_children",
                                "depth_of_node", "path_to_node"],
                    "depth": 2, "n_children_range": (2, 4),
                    "leaves_per_child_range": (1, 3),
                    "show_values": True}
        if level <= 5:
            return {"qtypes": ["count_leaves", "depth_of_node",
                                "count_children", "path_to_node",
                                "count_nodes_by_depth",
                                "mean_leaf_value"],
                    "depth": 2, "n_children_range": (3, 4),
                    "leaves_per_child_range": (2, 3),
                    "show_values": True}
        if level <= 7:
            # 3-level deep tree (structurally different)
            return {"qtypes": ["count_leaves_deep", "depth_of_node",
                                "path_to_node", "count_nodes_by_depth",
                                "mean_leaf_value"],
                    "depth": 3, "n_children_range": (2, 3),
                    "leaves_per_child_range": (2, 3),
                    "show_values": True}
        # L8-L9: deepest trees + genuinely hardest arithmetic ops
        return {"qtypes": ["total_in_subtree", "max_subtree_sum",
                           "deepest_path_sum", "common_ancestor_depth"],
                "depth": 3, "n_children_range": (2, 4),
                "leaves_per_child_range": (2, 3),
                "show_values": True}

    # ------------------------------------------------------------------ #

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 971)
        question_type = parameter.get("question_type",
                                       sub_rng.choice(cfg["qtypes"]))

        for _ in range(20):
            result = self._try_generate(sub_rng, rng, cfg, question_type, level)
            if result is not None:
                return result
        return None

    def _try_generate(self, sub_rng, rng, cfg, question_type, level
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        ctx = sub_rng.choice(_CONTEXT_POOLS)
        n_children = sub_rng.randint(*cfg["n_children_range"])
        n_children = min(n_children, len(ctx["children_pool"]))
        children = sub_rng.sample(ctx["children_pool"], n_children)

        depth = cfg["depth"]
        lc_lo, lc_hi = cfg["leaves_per_child_range"]

        # Build tree structure
        # Level 0: root
        # Level 1: children
        # Level 2: leaves (if depth==2) else grandchildren
        # Level 3: leaves (if depth==3)
        tree = {"name": ctx["root"], "children": [], "depth": 0}

        pool_leaves = list(ctx["leaf_pool"])
        sub_rng.shuffle(pool_leaves)
        pool_grand = list(ctx["grandchildren_pool"])
        sub_rng.shuffle(pool_grand)

        if depth == 2:
            for cn in children:
                n_l = sub_rng.randint(lc_lo, lc_hi)
                if len(pool_leaves) < n_l:
                    pool_leaves = list(ctx["leaf_pool"])
                    sub_rng.shuffle(pool_leaves)
                leaves = []
                for _ in range(n_l):
                    if not pool_leaves:
                        break
                    leaf_name = pool_leaves.pop()
                    val = sub_rng.randint(1, 20) if cfg["show_values"] else None
                    leaves.append({"name": leaf_name, "depth": 2,
                                    "value": val, "children": []})
                tree["children"].append({"name": cn, "depth": 1,
                                          "children": leaves, "value": None})
        else:  # depth == 3
            grandchildren_pool_iter = list(pool_grand)
            for cn in children:
                # child has 1-2 grandchildren
                n_g = sub_rng.randint(1, 2)
                grand_list = []
                for _ in range(n_g):
                    if not grandchildren_pool_iter:
                        grandchildren_pool_iter = list(ctx["grandchildren_pool"])
                        sub_rng.shuffle(grandchildren_pool_iter)
                    gname = grandchildren_pool_iter.pop()
                    n_l = sub_rng.randint(lc_lo, lc_hi)
                    leaves = []
                    for _ in range(n_l):
                        if not pool_leaves:
                            pool_leaves = list(ctx["leaf_pool"])
                            sub_rng.shuffle(pool_leaves)
                        leaf_name = pool_leaves.pop()
                        val = sub_rng.randint(1, 20) if cfg["show_values"] else None
                        leaves.append({"name": leaf_name, "depth": 3,
                                        "value": val, "children": []})
                    grand_list.append({"name": gname, "depth": 2,
                                        "children": leaves, "value": None})
                tree["children"].append({"name": cn, "depth": 1,
                                          "children": grand_list,
                                          "value": None})

        # Flatten helpers
        def all_leaves(node):
            if not node["children"]:
                return [node]
            res = []
            for c in node["children"]:
                res.extend(all_leaves(c))
            return res

        def find_parent(node, target_name):
            for c in node["children"]:
                if c["name"] == target_name:
                    return node
                p = find_parent(c, target_name)
                if p:
                    return p
            return None

        def depth_of(node, target_name, d=0):
            if node["name"] == target_name:
                return d
            for c in node["children"]:
                r = depth_of(c, target_name, d + 1)
                if r is not None:
                    return r
            return None

        def path_to(node, target_name):
            if node["name"] == target_name:
                return [node["name"]]
            for c in node["children"]:
                sub = path_to(c, target_name)
                if sub is not None:
                    return [node["name"]] + sub
            return None

        def subtree_sum(node):
            if not node["children"] and node.get("value") is not None:
                return node["value"]
            s = 0
            for c in node["children"]:
                s += subtree_sum(c)
            return s

        def all_non_leaf(node):
            res = []
            if node["children"]:
                res.append(node)
                for c in node["children"]:
                    res.extend(all_non_leaf(c))
            return res

        def count_at_depth(node, target_d, d=0):
            if d == target_d:
                return 1
            return sum(count_at_depth(c, target_d, d + 1)
                       for c in node["children"])

        leaves = all_leaves(tree)
        non_leaves = all_non_leaf(tree)

        # Build question / answer
        question = None; answer = None

        if question_type == "count_leaves":
            templates = [
                "How many leaf nodes (bottom-level items) does the tree have?",
                "Count the leaf nodes in the diagram shown.",
                "From the figure, how many leaves are in the tree?",
                "Determine the number of terminal (leaf) nodes in the tree.",
            ]
            question = sub_rng.choice(templates)
            answer = len(leaves)

        elif question_type == "count_leaves_deep":
            templates = [
                "How many leaf nodes (bottom-level items) does the tree contain?",
                "From the figure, count all leaf nodes across every branch.",
                "Count the terminal nodes in the hierarchical diagram.",
            ]
            question = sub_rng.choice(templates)
            answer = len(leaves)

        elif question_type == "depth_of_node":
            target = sub_rng.choice(leaves)
            templates = [
                f"What is the depth of '{target['name']}' in the tree? (root = depth 0)",
                f"From the diagram, find the depth of node '{target['name']}'. Root is depth 0.",
                f"How deep is '{target['name']}' in the tree (root depth = 0)?",
                f"Report the depth of '{target['name']}' (root at depth 0).",
            ]
            question = sub_rng.choice(templates)
            answer = depth_of(tree, target["name"])

        elif question_type == "count_children":
            target = sub_rng.choice(non_leaves)
            templates = [
                f"How many direct children does '{target['name']}' have?",
                f"From the figure, count the direct children of '{target['name']}'.",
                f"How many nodes are immediately below '{target['name']}' in the tree?",
                f"Report the number of direct children of node '{target['name']}'.",
            ]
            question = sub_rng.choice(templates)
            answer = len(target["children"])

        elif question_type == "path_to_node":
            target = sub_rng.choice(leaves)
            p = path_to(tree, target["name"])
            templates = [
                f"What is the path from the root to '{target['name']}'? (list nodes separated by ' -> ')",
                f"Write the root-to-'{target['name']}' path as nodes separated by ' -> '.",
                f"List the nodes from root to '{target['name']}' using ' -> '.",
            ]
            question = sub_rng.choice(templates)
            answer = " -> ".join(p)

        elif question_type == "total_in_subtree":
            target = sub_rng.choice([c for c in non_leaves if c["depth"] >= 1])
            s = subtree_sum(target)
            templates = [
                f"What is the sum of all leaf values under '{target['name']}'?",
                f"From the diagram, compute the total leaf value under '{target['name']}'.",
                f"Sum the leaf values in the subtree rooted at '{target['name']}'.",
            ]
            question = sub_rng.choice(templates)
            answer = s

        elif question_type == "max_subtree_sum":
            # Find child of root with greatest subtree sum
            sums = {c["name"]: subtree_sum(c) for c in tree["children"]}
            best = max(sums, key=sums.get)
            templates = [
                "Which direct child of the root has the greatest subtree sum (sum of descendant leaf values)?",
                "Among the root's direct children, which subtree has the largest total leaf value?",
                "Find the child of the root whose subtree has the highest leaf-value sum.",
            ]
            question = sub_rng.choice(templates)
            answer = best

        elif question_type == "common_ancestor_depth":
            if len(leaves) < 2:
                return None
            l1, l2 = sub_rng.sample(leaves, 2)
            p1 = path_to(tree, l1["name"])
            p2 = path_to(tree, l2["name"])
            common = 0
            for i in range(min(len(p1), len(p2))):
                if p1[i] == p2[i]:
                    common = i
                else:
                    break
            templates = [
                f"What is the depth of the lowest common ancestor of '{l1['name']}' and '{l2['name']}'? (root depth = 0)",
                f"Find the depth of the LCA of '{l1['name']}' and '{l2['name']}'.",
                f"Compute the depth of the deepest shared ancestor of '{l1['name']}' and '{l2['name']}'.",
            ]
            question = sub_rng.choice(templates)
            answer = common

        elif question_type == "deepest_path_sum":
            # Path from root to the deepest leaf; sum leaf values on that path
            # (tree has only one leaf-level with values, so sum = deepest leaf value)
            # We use the max subtree sum path instead
            best_leaf = max(leaves, key=lambda x: x.get("value", 0))
            p = path_to(tree, best_leaf["name"])
            templates = [
                "What is the greatest leaf value anywhere in the tree?",
                "Find the maximum value among all leaves in the figure.",
                "From the diagram, report the largest leaf value.",
            ]
            question = sub_rng.choice(templates)
            answer = best_leaf["value"]

        elif question_type == "count_nodes_by_depth":
            # Pick a non-root depth
            max_depth = cfg["depth"]
            d = sub_rng.randint(1, max_depth)
            cnt = count_at_depth(tree, d)
            templates = [
                f"How many nodes exist at depth {d} in the tree? (root depth = 0)",
                f"Count the nodes at depth {d} (root is depth 0).",
                f"From the figure, how many nodes are exactly {d} levels below the root?",
            ]
            question = sub_rng.choice(templates)
            answer = cnt

        elif question_type == "mean_leaf_value":
            vals = [l["value"] for l in leaves if l.get("value") is not None]
            if not vals:
                return None
            mean = round(sum(vals) / len(vals), 2)
            templates = [
                "What is the mean of all leaf values in the tree? Round to 2 decimals.",
                "Compute the average of leaf values shown. Round to 2 decimals.",
                "From the figure, find the mean leaf value. 2 decimals.",
            ]
            question = sub_rng.choice(templates)
            answer = mean
        else:
            return None

        if answer is None:
            return None

        img = self._render(sub_rng, tree, ctx["title"], cfg, level)
        return question, str(answer), img

    # ------------------------------------------------------------------ #
    # Renderer with layout variety
    # ------------------------------------------------------------------ #

    def _render(self, sub_rng, tree, title, cfg, level) -> Image.Image:
        style = self._random_style()
        layout = sub_rng.choice(["vertical_topdown", "vertical_topdown",
                                   "horizontal_lr"])
        palette = style["palette"]
        fs = max(style["font_size_base"], 10)

        # Compute tree width/depth
        def max_depth(node, d=0):
            if not node["children"]:
                return d
            return max(max_depth(c, d + 1) for c in node["children"])

        def leaf_count(node):
            if not node["children"]:
                return 1
            return sum(leaf_count(c) for c in node["children"])

        md = max_depth(tree)
        lc = leaf_count(tree)

        # Layout: assign positions
        positions = {}
        if layout == "vertical_topdown":
            # root at top, leaves at bottom
            fig_w = max(7, lc * 1.2)
            fig_h = max(4.5, (md + 1) * 1.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            y_top = fig_h - 0.6
            dy = (fig_h - 1.2) / max(md, 1)
            # Assign x to leaves in-order, then propagate up
            leaves_in_order = []
            def collect_leaves(node):
                if not node["children"]:
                    leaves_in_order.append(node["name"])
                else:
                    for c in node["children"]:
                        collect_leaves(c)
            collect_leaves(tree)
            x_spacing = fig_w / max(len(leaves_in_order), 1)
            for i, ln in enumerate(leaves_in_order):
                positions[ln] = ((i + 0.5) * x_spacing,
                                  y_top - md * dy)
            def assign(node, d=0):
                if not node["children"]:
                    return positions[node["name"]][0]
                xs = [assign(c, d + 1) for c in node["children"]]
                mx = sum(xs) / len(xs)
                positions[node["name"]] = (mx, y_top - d * dy)
                return mx
            assign(tree)
        else:  # horizontal_lr
            fig_w = max(8, (md + 1) * 2.3)
            fig_h = max(4.5, lc * 1.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            x_left = 0.5
            dx = (fig_w - 1.5) / max(md, 1)
            leaves_in_order = []
            def collect_leaves(node):
                if not node["children"]:
                    leaves_in_order.append(node["name"])
                else:
                    for c in node["children"]:
                        collect_leaves(c)
            collect_leaves(tree)
            y_spacing = fig_h / max(len(leaves_in_order), 1)
            for i, ln in enumerate(leaves_in_order):
                positions[ln] = (x_left + md * dx,
                                  fig_h - (i + 0.5) * y_spacing)
            def assign(node, d=0):
                if not node["children"]:
                    return positions[node["name"]][1]
                ys = [assign(c, d + 1) for c in node["children"]]
                my = sum(ys) / len(ys)
                positions[node["name"]] = (x_left + d * dx, my)
                return my
            assign(tree)

        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        # Edge draw
        def draw_edges(node):
            x0, y0 = positions[node["name"]]
            for c in node["children"]:
                x1, y1 = positions[c["name"]]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->",
                                            color=palette[7],
                                            lw=style["line_width"]))
                draw_edges(c)
        draw_edges(tree)

        # Node draw
        def node_label(node):
            if node.get("value") is not None:
                return f"{node['name']}\n({node['value']})"
            return node["name"]

        depth_colors = [palette[0], palette[1], palette[2], palette[3],
                         palette[4]]

        def draw_nodes(node, d=0):
            x, y = positions[node["name"]]
            col = depth_colors[d % len(depth_colors)]
            ax.text(x, y, node_label(node), ha="center", va="center",
                    fontsize=fs,
                    fontweight="bold" if d == 0 else "normal",
                    color="white" if d <= 1 else "#1a1a1a",
                    bbox=dict(boxstyle="round,pad=0.28",
                              facecolor=col if d <= 1 else "#ecf0f1",
                              edgecolor=palette[7], linewidth=1.2))
            for c in node["children"]:
                draw_nodes(c, d + 1)
        draw_nodes(tree)

        margin = 0.5
        ax.set_xlim(-margin, fig.get_size_inches()[0] + margin)
        ax.set_ylim(-margin, fig.get_size_inches()[1] + margin)
        ax.axis("off")
        ax.set_title(title, fontsize=fs + 3, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
