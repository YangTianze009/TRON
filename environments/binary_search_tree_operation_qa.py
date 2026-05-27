"""
Binary Search Tree Operation QA (v3 planning batch, 2026-04-16).

Target: reference graph_problems / VisualPuzzles algorithmic.
Renders a BST as a node-link diagram. Questions ask about the result of
an operation (search / insert / delete) on the tree — specifically:
  - L0-2: search (how many comparisons until a value is found)
  - L3-5: insert (which existing node becomes the parent of the new value)
  - L6-8: delete-leaf (after deleting a leaf, which node is the new
           parent of a given value, or root of a stable subtree)
  - L9:   delete-internal (which node replaces the deleted node — the
           in-order successor's value)

Answer type:
  - Integer for "comparisons needed" queries (Answer with a single integer.)
  - MCQ letter (A-D) for "which node" queries (Answer with a single letter.)

Difficulty axes:
  1. n_nodes = 5 + level // 2 (5..9)
  2. operation_type progression (search -> insert -> delete-leaf
     -> delete-internal)
  3. tree_balance: balanced at L0, gradually skewed by L9.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

_SEARCH_TEMPLATES = [
    "The image shows a binary search tree (BST). In a BST search, we start at the root and at each node compare the target with the node's value: if equal we stop, if smaller we go left, if larger we go right. How many node comparisons are needed to find the value {target}? Count each visited node (including the match) as one comparison. Answer with a single integer.",
    "In the BST shown, counting each visited node as a comparison, how many comparisons does a standard BST search take to locate {target}? Give a single integer.",
    "Perform a BST search for {target} in the tree depicted. Report the number of node comparisons (including the one that matches). Answer: single integer.",
    "For the BST above, how many nodes are visited while searching for {target}? Each visited node counts as one comparison. Integer answer only.",
    "Trace the BST search for {target} from the root down. How many nodes are compared? Provide the integer.",
    "How many comparisons does it take to find {target} in this BST using the standard go-left-if-smaller/right-if-larger rule? Single integer.",
    "Count the number of nodes visited by a BST search for value {target} in the tree shown. Integer answer.",
    "Starting at the root, walk the BST looking for {target}. How many node comparisons are performed (including the terminal one)? Answer as an integer.",
    "In the given BST, determine the search path length to reach {target}. Report it as the number of comparisons (integer).",
    "Given the binary search tree in the image, what is the number of comparisons a standard BST lookup of {target} performs? Single integer.",
    "The tree above is a BST. Trace the search for {target}. How many comparisons occur along the way? Answer with a single integer.",
    "Find {target} in the BST using standard lookup. How many node values were compared? Integer only.",
    "Using the BST shown, state the number of comparisons made to locate {target}. Each visited node is one comparison. Integer answer.",
    "Walk the BST from root toward {target}. How many comparisons (visited nodes, including the match) occur? Answer as a single integer.",
    "Search the depicted BST for value {target}. Report the count of node-value comparisons as an integer.",
    "For the BST in the image, how many nodes must be examined in order to find {target}? Answer: single integer.",
]

_INSERT_TEMPLATES = [
    "The image shows a binary search tree. If we insert the value {new_val} using the standard BST insertion (walk from the root, going left when the new value is smaller and right when larger, until we reach an empty child), which existing node will become the parent of {new_val}?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST pictured, perform a standard BST insertion of {new_val}. Which node ends up as its parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Insert {new_val} into the BST using the usual left-if-smaller / right-if-larger walk. Which existing node becomes its parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "For the tree shown, which node will be the parent of {new_val} after a standard BST insertion?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Apply BST insertion of the value {new_val} to the depicted tree. Identify the parent of the new node.\nOptions: {opt_text}\nAnswer with a single letter.",
    "When {new_val} is inserted into this BST following the standard rule, which node gains it as a child?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Where does the standard BST insertion put {new_val} in the tree above? Name the parent node.\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST shown, insertion of {new_val} produces a new leaf. Which node is its parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Perform a BST insert of {new_val} on the given tree. Which node attaches to the new leaf?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Walk down the BST to insert {new_val}. Where does it attach — which node is the parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "If {new_val} is inserted into the BST via the standard algorithm, which existing node becomes its parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Trace the BST insertion of {new_val} in the tree above; which node's empty child slot receives it?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Inserting {new_val} into the depicted BST, the new node is attached under which existing node?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Using standard BST insertion, after inserting {new_val}, the parent of {new_val} is which node?\nOptions: {opt_text}\nAnswer with a single letter.",
    "The image shows a BST. Insert {new_val}. Which existing node gets {new_val} as a new child?\nOptions: {opt_text}\nAnswer with a single letter.",
    "After a standard BST insert of {new_val}, report the parent of the newly added node.\nOptions: {opt_text}\nAnswer with a single letter.",
]

_DELETE_LEAF_TEMPLATES = [
    "The image shows a binary search tree. We want to delete the leaf node with value {leaf_val}. Which existing node will directly lose a child (i.e., the parent of {leaf_val}) as a result of this deletion?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST above, a leaf with value {leaf_val} is removed. Which node loses a child in the process?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Delete the leaf {leaf_val} from the depicted BST. Which existing node was the parent of {leaf_val}?\nOptions: {opt_text}\nAnswer with a single letter.",
    "For the BST shown, removing the leaf {leaf_val} disconnects it from which parent node?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Which node is the parent of the leaf {leaf_val} in the BST above (i.e., the one losing a child when {leaf_val} is deleted)?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Remove the leaf node {leaf_val} from the BST. Which existing node is directly affected (loses a child)?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In this BST, leaf {leaf_val} is about to be deleted. Name its parent.\nOptions: {opt_text}\nAnswer with a single letter.",
    "Identify the parent of leaf {leaf_val} in the tree above (the node that ends up with one fewer child after deletion).\nOptions: {opt_text}\nAnswer with a single letter.",
    "A BST leaf with value {leaf_val} is deleted. Which node had {leaf_val} as its child?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the depicted BST, find the parent of the leaf {leaf_val} — the node that loses a child upon deletion.\nOptions: {opt_text}\nAnswer with a single letter.",
    "Delete leaf {leaf_val} from the tree. Which existing node's child-list shrinks by one?\nOptions: {opt_text}\nAnswer with a single letter.",
    "For the BST above, when leaf {leaf_val} is removed, which node is directly impacted as its parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Point out the parent of leaf {leaf_val} in the given BST — i.e., the node directly connected to {leaf_val} above it.\nOptions: {opt_text}\nAnswer with a single letter.",
    "Leaf {leaf_val} is deleted from this BST. Which node loses {leaf_val} as a child?\nOptions: {opt_text}\nAnswer with a single letter.",
    "If we delete the leaf {leaf_val} from the BST shown, which existing node is its parent?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST, the leaf {leaf_val} will be removed. Which node is the parent node that loses a child?\nOptions: {opt_text}\nAnswer with a single letter.",
]

_DELETE_INTERNAL_TEMPLATES = [
    "The image shows a binary search tree. We want to delete the node with value {target}, which has two children. Under the standard BST deletion that replaces the deleted node with its in-order successor (the smallest value in its right subtree), which value will take the place of {target}?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST shown, delete the internal node {target} (which has two children). Using the in-order successor rule, which value replaces it?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Delete node {target} from the depicted BST. The in-order successor (smallest value in {target}'s right subtree) takes its place. Which value is that?\nOptions: {opt_text}\nAnswer with a single letter.",
    "For the BST above, deleting {target} replaces it with its in-order successor. Which value becomes the new {target}'s slot occupant?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Which value replaces {target} when we delete it from the BST using the in-order-successor rule?\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST, delete the two-child node {target} by substituting its in-order successor. Identify that successor value.\nOptions: {opt_text}\nAnswer with a single letter.",
    "Find the in-order successor of {target} in the BST shown — the value that replaces {target} on deletion.\nOptions: {opt_text}\nAnswer with a single letter.",
    "Given the BST above, what value does standard BST deletion put in place of {target} (using the in-order-successor rule)?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Delete internal node {target} using the in-order-successor strategy. The replacement value is which of the options?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Name the value that would replace {target} in the BST after a standard deletion (in-order successor).\nOptions: {opt_text}\nAnswer with a single letter.",
    "In the BST, identify the in-order successor of {target} — i.e., the value that overwrites {target} when it's deleted.\nOptions: {opt_text}\nAnswer with a single letter.",
    "Which value, when deleting {target} from the BST using the in-order-successor rule, will occupy {target}'s position?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Perform BST deletion of {target} with the standard in-order-successor substitution. Which value takes {target}'s place?\nOptions: {opt_text}\nAnswer with a single letter.",
    "Delete {target} from the BST by replacing it with the smallest value in its right subtree. What is that value?\nOptions: {opt_text}\nAnswer with a single letter.",
    "For the BST depicted, the in-order-successor replacement of {target} is which value?\nOptions: {opt_text}\nAnswer with a single letter.",
    "After deleting {target} and substituting its in-order successor, which value ends up in {target}'s former location?\nOptions: {opt_text}\nAnswer with a single letter.",
]

class _BSTNode:
    __slots__ = ("val", "left", "right")
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def _bst_insert(root: Optional[_BSTNode], val: int) -> _BSTNode:
    if root is None:
        return _BSTNode(val)
    if val < root.val:
        root.left = _bst_insert(root.left, val)
    elif val > root.val:
        root.right = _bst_insert(root.right, val)
    return root

def _collect_vals(root: Optional[_BSTNode]) -> List[int]:
    if root is None:
        return []
    return _collect_vals(root.left) + [root.val] + _collect_vals(root.right)

def _find_parent(root: Optional[_BSTNode], val: int,
                  parent: Optional[_BSTNode] = None) -> Optional[_BSTNode]:
    if root is None:
        return None
    if root.val == val:
        return parent
    if val < root.val:
        return _find_parent(root.left, val, root)
    return _find_parent(root.right, val, root)

def _search_comparisons(root: Optional[_BSTNode], val: int) -> int:
    """Number of node comparisons needed to find val in the BST."""
    cmps = 0
    cur = root
    while cur is not None:
        cmps += 1
        if cur.val == val:
            return cmps
        cur = cur.left if val < cur.val else cur.right
    return cmps  # not found: returns the full unsuccessful path length

def _insert_parent_value(root: _BSTNode, new_val: int) -> int:
    """Return the value of the parent that the newly-inserted node would
    attach to. Assumes new_val is not already in the tree."""
    cur = root
    while True:
        if new_val < cur.val:
            if cur.left is None:
                return cur.val
            cur = cur.left
        else:
            if cur.right is None:
                return cur.val
            cur = cur.right

def _inorder_successor_val(root: _BSTNode, target_val: int) -> Optional[int]:
    """Find the in-order successor's value for a node with two children
    in a BST. Returns None if such a successor cannot be found."""
    # Locate the target
    cur = root
    while cur is not None and cur.val != target_val:
        cur = cur.left if target_val < cur.val else cur.right
    if cur is None or cur.right is None:
        return None
    nxt = cur.right
    while nxt.left is not None:
        nxt = nxt.left
    return nxt.val

def _leaves(root: Optional[_BSTNode]) -> List[int]:
    if root is None:
        return []
    if root.left is None and root.right is None:
        return [root.val]
    return _leaves(root.left) + _leaves(root.right)

def _compute_positions(root: _BSTNode) -> Dict[int, Tuple[float, float]]:
    """Place BST using an x-based spread that scales with subtree width.
    Root at (0,0), y decreases by 1.2 per depth.

    Single-child nodes (only-left or only-right) would otherwise land at the
    same x as the parent, making left vs right ambiguous. We offset them
    horizontally so the edge clearly slopes in the correct direction.
    """
    positions: Dict[int, Tuple[float, float]] = {}

    def subtree_width(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return subtree_width(node.left) + subtree_width(node.right)

    def layout(node, x0, x1, depth):
        if node is None:
            return
        lw = subtree_width(node.left)
        rw = subtree_width(node.right)
        total_sub = lw + rw
        if total_sub > 0:
            mid = x0 + (x1 - x0) * lw / total_sub
        else:
            # leaf: place in the middle of its slot
            mid = (x0 + x1) / 2
        positions[node.val] = (mid, -depth * 1.2)
        layout(node.left, x0, mid, depth + 1)
        layout(node.right, mid, x1, depth + 1)

    total_w = max(subtree_width(root), 1)
    layout(root, -total_w * 0.8, total_w * 0.8, 0)

    # Post-pass: shift only-left/only-right children so edges slope clearly.
    OFFSET = 0.55

    def shift_only_child(node):
        if node is None:
            return
        px, py = positions[node.val]
        if node.left is not None and node.right is None:
            cx, cy = positions[node.left.val]
            if abs(cx - px) < 0.05:
                dx = cx - (px + OFFSET)
                # shift entire left subtree leftward by OFFSET
                _shift_subtree(node.left, -OFFSET)
        elif node.right is not None and node.left is None:
            cx, cy = positions[node.right.val]
            if abs(cx - px) < 0.05:
                _shift_subtree(node.right, +OFFSET)
        shift_only_child(node.left)
        shift_only_child(node.right)

    def _shift_subtree(node, dx):
        if node is None:
            return
        x, y = positions[node.val]
        positions[node.val] = (x + dx, y)
        _shift_subtree(node.left, dx)
        _shift_subtree(node.right, dx)

    shift_only_child(root)
    return positions

class BinarySearchTreeOperationQA(StandaloneVisualEnv):
    ENV_NAME = "binary_search_tree_operation"

    # ------------------------------------------------------------------ #
    # Level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            op = "search"
        elif level <= 5:
            op = "insert"
        elif level <= 8:
            op = "delete_leaf"
        else:
            op = "delete_internal"
        return {
            "n_nodes": 5 + level // 2,          # 5..9
            "op": op,
            # 0 = balanced, ~0.6 = noticeably skewed. Cap at 0.6 so a
            # node with two children always exists even at L9.
            "skew": min(0.6, level / 15.0),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1429)
        self._primary_complexity_feature = cfg["n_nodes"] * 10 + level

        for _ in range(30):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Tree construction
    # ------------------------------------------------------------------ #

    def _build_bst(self, rng: random.Random, cfg: Dict) -> Tuple[_BSTNode, List[int]]:
        n = cfg["n_nodes"]
        skew = cfg["skew"]
        values = rng.sample(range(10, 90), n)
        # Insertion order controls balance. Sorted order -> fully skewed.
        sorted_vals = sorted(values)
        # Blend between a balanced permutation (BFS-style from a balanced
        # ordering) and a sorted order.
        balanced_order = self._balanced_insert_order(sorted_vals)
        skewed_order = list(sorted_vals)
        rng.shuffle(skewed_order[:0])   # no-op; keep deterministic

        # Pick each slot from balanced or skewed with probability = skew.
        order: List[int] = []
        used = set()
        for k in range(n):
            if rng.random() < skew:
                # pick first unused from skewed_order
                for v in skewed_order:
                    if v not in used:
                        order.append(v); used.add(v); break
            else:
                for v in balanced_order:
                    if v not in used:
                        order.append(v); used.add(v); break

        # Fall back: fill any missing
        for v in values:
            if v not in used:
                order.append(v); used.add(v)

        root = None
        for v in order:
            root = _bst_insert(root, v)
        return root, sorted_vals

    @staticmethod
    def _balanced_insert_order(sorted_vals: List[int]) -> List[int]:
        """Return an insertion order that produces a balanced BST."""
        order: List[int] = []

        def rec(lo, hi):
            if lo > hi:
                return
            mid = (lo + hi) // 2
            order.append(sorted_vals[mid])
            rec(lo, mid - 1)
            rec(mid + 1, hi)

        rec(0, len(sorted_vals) - 1)
        return order

    # ------------------------------------------------------------------ #
    # Operation handlers
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        root, sorted_vals = self._build_bst(rng, cfg)
        if root is None:
            return None
        op = cfg["op"]
        image = self._render(root, rng)

        if op == "search":
            return self._gen_search(root, sorted_vals, rng, image)
        if op == "insert":
            return self._gen_insert(root, sorted_vals, rng, image)
        if op == "delete_leaf":
            return self._gen_delete_leaf(root, sorted_vals, rng, image)
        if op == "delete_internal":
            return self._gen_delete_internal(root, sorted_vals, rng, image)
        return None

    def _gen_search(self, root, sorted_vals, rng, image):
        # Ask for number of comparisons to find a target value.
        target = rng.choice(sorted_vals)
        cmps = _search_comparisons(root, target)
        if cmps < 1:
            return None
        sidx = (self.seed or 0) % 16
        question = _SEARCH_TEMPLATES[sidx].format(target=target)
        return question, str(cmps), image

    def _gen_insert(self, root, sorted_vals, rng, image):
        # Pick a value not in the tree; the standard BST insert would
        # attach the new node to some existing node — ask which one.
        present = set(sorted_vals)
        candidates = [v for v in range(10, 90) if v not in present]
        if not candidates:
            return None
        rng.shuffle(candidates)
        new_val = candidates[0]
        parent_val = _insert_parent_value(root, new_val)

        # Build MCQ options
        other_vals = [v for v in sorted_vals if v != parent_val]
        if len(other_vals) < 3:
            return None
        distractors = rng.sample(other_vals, 3)
        options = list(distractors)
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, parent_val)
        answer_letter = chr(ord("A") + correct_idx)

        opt_text = "  ".join(f"{chr(ord('A')+i)}) {v}"
                             for i, v in enumerate(options))
        sidx = (self.seed or 0) % 16
        question = _INSERT_TEMPLATES[sidx].format(new_val=new_val, opt_text=opt_text)
        return question, answer_letter, image

    def _gen_delete_leaf(self, root, sorted_vals, rng, image):
        # Delete a leaf, then ask either (a) which node becomes a new leaf
        # (if parent has no children left) or (b) which node was the
        # parent of the deleted leaf. We'll ask (b) because it has a
        # stable unique answer.
        leaves = _leaves(root)
        if len(leaves) < 2:
            return None
        leaf_val = rng.choice(leaves)
        parent = _find_parent(root, leaf_val)
        if parent is None:
            return None
        parent_val = parent.val

        other_vals = [v for v in sorted_vals if v != parent_val and v != leaf_val]
        if len(other_vals) < 3:
            return None
        distractors = rng.sample(other_vals, 3)
        options = list(distractors)
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, parent_val)
        answer_letter = chr(ord("A") + correct_idx)

        opt_text = "  ".join(f"{chr(ord('A')+i)}) {v}"
                             for i, v in enumerate(options))
        sidx = (self.seed or 0) % 16
        question = _DELETE_LEAF_TEMPLATES[sidx].format(leaf_val=leaf_val, opt_text=opt_text)
        return question, answer_letter, image

    def _gen_delete_internal(self, root, sorted_vals, rng, image):
        # Find an internal node with two children; ask for the in-order
        # successor's value (the value that replaces the deleted node
        # under the standard "replace with in-order successor" rule).
        two_children: List[int] = []

        def walk(node):
            if node is None:
                return
            if node.left is not None and node.right is not None:
                two_children.append(node.val)
            walk(node.left); walk(node.right)

        walk(root)
        if not two_children:
            return None
        target = rng.choice(two_children)
        succ = _inorder_successor_val(root, target)
        if succ is None:
            return None

        other_vals = [v for v in sorted_vals if v != succ and v != target]
        if len(other_vals) < 3:
            return None
        distractors = rng.sample(other_vals, 3)
        options = list(distractors)
        correct_idx = rng.randint(0, 3)
        options.insert(correct_idx, succ)
        answer_letter = chr(ord("A") + correct_idx)

        opt_text = "  ".join(f"{chr(ord('A')+i)}) {v}"
                             for i, v in enumerate(options))
        sidx = (self.seed or 0) % 16
        question = _DELETE_INTERNAL_TEMPLATES[sidx].format(target=target, opt_text=opt_text)
        return question, answer_letter, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, root: _BSTNode, rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        positions = _compute_positions(root)
        if not positions:
            return None
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]

        mode = pick_render_mode(rng)
        if mode == "textbook":
            tbp = textbook_params(rng)
            bg = tbp["bg"]
            edge_col = tbp["line_color"]
            edge_lw = tbp["line_width"] + 0.5
            node_edge = tbp["line_color"]
            text_color = "white"
            font_kw = {"fontfamily": tbp["font_family"]}
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            edge_col = "#1a1a1a"
            edge_lw = style.get("line_width", 1.4) + 0.8
            node_edge = "#1a1a1a"
            text_color = "white"
            font_kw = {}
            dpi = style["dpi"]
        else:
            bg = style["bg_color"]
            edge_col = style.get("geo_line_color", "#2c3e50")
            edge_lw = style.get("line_width", 1.4) + 0.5
            node_edge = "#2c3e50"
            text_color = "white"
            font_kw = {}
            dpi = style["dpi"]

        def _draw():
            fig, ax = plt.subplots(figsize=(7.5 * sc, 5.5 * sc))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
            ax.axis("off")

            def _draw_edges(node):
                if node is None:
                    return
                px, py = positions[node.val]
                for child in (node.left, node.right):
                    if child is None:
                        continue
                    cx, cy = positions[child.val]
                    ax.plot([px, cx], [py, cy], color=edge_col,
                            linewidth=edge_lw, zorder=1)
                    _draw_edges(child)

            _draw_edges(root)

            radius = 0.38
            for i, (val, (x, y)) in enumerate(positions.items()):
                circ = plt.Circle((x, y), radius,
                                  facecolor=palette[i % len(palette)],
                                  edgecolor=node_edge,
                                  linewidth=style.get("line_width", 1.5) + 0.5,
                                  zorder=5)
                ax.add_patch(circ)
                ax.text(x, y, str(val),
                        ha="center", va="center",
                        fontsize=fs + 1, fontweight="bold",
                        color=text_color, zorder=6, **font_kw)

            ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
            ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)

            titles = ["Binary Search Tree", "BST", "Binary Search Tree Diagram",
                      "BST Diagram"]
            ax.set_title(rng.choice(titles),
                         fontsize=fs + 3, fontweight="bold",
                         **font_kw)

            fig.tight_layout()
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()
        return self.fig_to_pil(fig, dpi=dpi)

if __name__ == "__main__":
    env = BinarySearchTreeOperationQA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
