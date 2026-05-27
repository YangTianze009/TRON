"""
Number-tree inverse puzzle.

A binary or ternary tree is drawn with values at some nodes. A combination
rule is given (parent = sum / product / max of children). The model must
compute the value at a marked target node (root, an interior node, or a
leaf), reasoning either forward (compute upward) or inverse (one child
missing, parent given → child = inverse-op).

Mirrors reference qids 155, 156, 179, 180, 285, 350:
  "Find the missing number in the last node." Ans: e.g. 233 / 160 / 37 /
  38 / 529 / 170. Various ops including sum and powers.

L0: 3-leaf tree (depth 2). 2 leaves filled, 1 leaf missing AND parent given;
or all leaves filled, find root. Trivial 1-step compute.
L9: depth-4 tree (15+ nodes), several unknowns scattered, with non-trivial
op (e.g. parent = product, parent = max, parent = abs-diff).
"""
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "In the tree shown, each parent node equals the {op_desc} of its children. Find the value of the node marked {L} and place the integer in <answer>...</answer>.",
    "The tree obeys: parent = {op_desc} of its children. Determine the value of node {L}. Integer answer in <answer>...</answer>.",
    "Each non-leaf node equals the {op_desc} of its child nodes. What is the value at node {L}? Integer in <answer>...</answer>.",
    "Tree puzzle: every parent node is the {op_desc} of its children. Compute the missing value at node {L} and place the integer in <answer>...</answer>.",
    "The displayed tree follows the rule: parent = {op_desc} of children. Find the value of node {L}. Integer in <answer>...</answer>.",
    "Look at the number tree. The rule: each parent = {op_desc} of its children. What number belongs at node {L}? Integer answer in <answer>...</answer>.",
    "Each interior node of the tree is the {op_desc} of its children. Determine the value of {L}. Integer in <answer>...</answer>.",
    "Compute node {L} of the tree, given that each parent equals the {op_desc} of its children. Integer answer in <answer>...</answer>.",
    "In this number tree, parent = {op_desc} of children. Find the value at the node marked {L}. Place integer in <answer>...</answer>.",
    "The number tree's rule: parent = {op_desc} of its children. What is the missing value at node {L}? Integer in <answer>...</answer>.",
    "Find the value at node {L}. Each parent in the tree equals the {op_desc} of its children. Integer answer in <answer>...</answer>.",
    "Tree-arithmetic puzzle: parent = {op_desc} of children. Determine the integer at node {L}. Place it in <answer>...</answer>.",
    "Solve the number tree (parent = {op_desc} of its children). What value is at node {L}? Integer in <answer>...</answer>.",
    "The image shows a number tree with rule: each parent = {op_desc} of children. Compute the value at node {L}. Integer answer in <answer>...</answer>.",
    "Number-tree question: parent = {op_desc} of children. Find {L}. Place the integer in <answer>...</answer>.",
    "Determine the missing number at node {L} of the tree (parent = {op_desc} of children). Integer answer in <answer>...</answer>.",
]


_OP_DESCRIPTIONS = {
    "sum": "sum",
    "product": "product",
    "max": "maximum",
    "min": "minimum",
    "abs_diff": "absolute difference",
}


def _apply_op(op: str, children: List[int]) -> int:
    if op == "sum":
        return sum(children)
    if op == "product":
        p = 1
        for c in children:
            p *= c
        return p
    if op == "max":
        return max(children)
    if op == "min":
        return min(children)
    if op == "abs_diff":
        # For 2 children only.
        return abs(children[0] - children[1])
    raise ValueError(op)


class _Node:
    __slots__ = ("val", "children", "label")

    def __init__(self, val=None, children=None, label=None):
        self.val = val
        self.children = children or []
        self.label = label


def _build_tree(rng: random.Random, depth: int, branching: int,
                leaf_max: int) -> _Node:
    """Build a tree of given depth where every leaf is at depth `depth`."""
    if depth == 0:
        return _Node(val=rng.randint(1, leaf_max))
    children = [_build_tree(rng, depth - 1, branching, leaf_max)
                for _ in range(branching)]
    return _Node(children=children)


def _propagate(node: _Node, op: str) -> int:
    if not node.children:
        return node.val
    vals = [_propagate(c, op) for c in node.children]
    node.val = _apply_op(op, vals)
    return node.val


def _all_nodes(node: _Node, out: List[_Node]):
    out.append(node)
    for c in node.children:
        _all_nodes(c, out)


class NumberTreeInverseQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "number_tree_inverse"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            depth = 1
            branching = 3
            n_hide = 1
            ops = ["sum"]
            leaf_max = 9
        elif level <= 2:
            depth = 2
            branching = 2
            n_hide = 1
            ops = ["sum"]
            leaf_max = 9
        elif level <= 4:
            depth = 2
            branching = 2
            n_hide = 2
            ops = ["sum", "max"]
            leaf_max = 12
        elif level <= 6:
            depth = 3
            branching = 2
            n_hide = 2
            ops = ["sum", "max", "min"]
            leaf_max = 12
        elif level <= 8:
            depth = 3
            branching = 2
            n_hide = 3
            ops = ["sum", "max", "product", "abs_diff"]
            leaf_max = 9
        else:
            depth = 4
            branching = 2
            n_hide = 4
            ops = ["sum", "product", "max", "abs_diff"]
            leaf_max = 8
        return {"level": level, "depth": depth, "branching": branching,
                "n_hide": n_hide, "ops": ops, "leaf_max": leaf_max}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1789 + level * 53 + 19)

        op = rng.choice(cfg["ops"])
        # abs_diff requires branching == 2
        if op == "abs_diff" and cfg["branching"] != 2:
            op = "sum"
        # product can blow up; if depth is high cap leaf_max
        leaf_max = cfg["leaf_max"]
        if op == "product" and cfg["depth"] >= 3:
            leaf_max = min(leaf_max, 5)

        # Build tree + propagate
        for _attempt in range(10):
            tree = _build_tree(rng, cfg["depth"], cfg["branching"], leaf_max)
            _propagate(tree, op)
            nodes: List[_Node] = []
            _all_nodes(tree, nodes)
            # Hide n_hide values; the marked target is one of the hidden ones.
            # Make sure each hidden value is uniquely determinable from the rest.
            n_hide = min(cfg["n_hide"], len(nodes) - 1)
            indices = list(range(len(nodes)))
            rng.shuffle(indices)
            hide_set = set(indices[:n_hide])
            # Assign labels A, B, C, ... to all hidden nodes (target gets first)
            target_idx = list(hide_set)[0] if hide_set else 0
            target_node = nodes[target_idx]

            # Solve: try to recover all hidden values uniquely.
            # Save originals.
            originals = [n.val for n in nodes]
            # Mark hidden as None
            for k in hide_set:
                nodes[k].val = None
            # Iterate inference
            ok = self._solve_tree(tree, op)
            if not ok:
                # Restore + try again
                for n, v in zip(nodes, originals):
                    n.val = v
                continue
            # All values now reconstructed. Verify they match originals (sanity).
            mismatch = False
            for n, v in zip(nodes, originals):
                if n.val != v:
                    mismatch = True
                    break
            if mismatch:
                continue
            # Restore mode: hide them again for rendering.
            shown_value: List[Optional[int]] = []
            for k, n in enumerate(nodes):
                if k in hide_set:
                    shown_value.append(None)
                else:
                    shown_value.append(n.val)
            # Assign labels to hidden nodes
            label_map: Dict[int, str] = {}
            target_label = "x"
            label_map[target_idx] = target_label
            other_letters = "ABCDEFGHIJKL"
            li = 0
            for k in hide_set:
                if k == target_idx:
                    continue
                if li >= len(other_letters):
                    break
                label_map[k] = other_letters[li]
                li += 1

            # Restore so we know answers
            for n, v in zip(nodes, originals):
                n.val = v
            answer = str(target_node.val)

            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(
                op_desc=_OP_DESCRIPTIONS[op], L=target_label)
            img = self._render(tree, nodes, shown_value, label_map, op)
            return question, answer, img
        return None

    def _solve_tree(self, tree: _Node, op: str) -> bool:
        """Iteratively try to fill in None values using parent/child relations.
        Returns True iff all values recovered."""
        # Collect parent-relations
        nodes: List[_Node] = []
        _all_nodes(tree, nodes)
        for _ in range(len(nodes) * 3 + 5):
            changed = False
            # Forward: parent unknown, all children known
            for n in nodes:
                if n.val is None and n.children and all(
                        c.val is not None for c in n.children):
                    n.val = _apply_op(op, [c.val for c in n.children])
                    changed = True
            # Inverse: parent known, exactly one child unknown (only for sum,
            # abs_diff). For max/min/product, inverse is ambiguous.
            if op in ("sum", "abs_diff"):
                for n in nodes:
                    if n.val is not None and n.children:
                        unk = [c for c in n.children if c.val is None]
                        kn = [c for c in n.children if c.val is not None]
                        if len(unk) == 1 and len(kn) == len(n.children) - 1:
                            if op == "sum":
                                unk[0].val = n.val - sum(c.val for c in kn)
                                changed = True
                            elif op == "abs_diff":
                                # parent = |a - b|; only solvable if we know
                                # which is bigger. Make assumption: keep a >= b
                                # implicitly by noting that abs_diff inverse
                                # gives 2 candidates → ambiguous. Skip.
                                pass
            if not changed:
                break
        return all(n.val is not None for n in nodes)

    def _render(self, tree: _Node, nodes: List[_Node],
                shown_value: List[Optional[int]],
                label_map: Dict[int, str], op: str) -> Image.Image:
        # Compute layout: assign x-position to each leaf, then back up.
        leaves: List[_Node] = []

        def collect_leaves(n: _Node):
            if not n.children:
                leaves.append(n)
            else:
                for c in n.children:
                    collect_leaves(c)
        collect_leaves(tree)

        # Build node-index lookup
        node_idx = {id(n): k for k, n in enumerate(nodes)}
        positions: Dict[int, Tuple[float, float]] = {}

        # Assign x to leaves spaced evenly
        for i, lf in enumerate(leaves):
            positions[id(lf)] = (float(i), 0.0)

        # Assign depth via BFS
        depth_of: Dict[int, int] = {}

        def depth(n: _Node, d=0):
            depth_of[id(n)] = d
            for c in n.children:
                depth(c, d + 1)
        depth(tree)
        max_depth = max(depth_of.values())

        # Assign x to interior nodes = mean of children's x
        def assign_x(n: _Node):
            if not n.children:
                return
            for c in n.children:
                assign_x(c)
            xs = [positions[id(c)][0] for c in n.children]
            x = sum(xs) / len(xs)
            y = (max_depth - depth_of[id(n)]) * 1.2
            positions[id(n)] = (x, y)
        assign_x(tree)
        # Ensure all leaves get correct y (depth 0 means leaf)
        for lf in leaves:
            x = positions[id(lf)][0]
            positions[id(lf)] = (x, (max_depth - depth_of[id(lf)]) * 1.2)

        bg = "#ffffff"
        node_color_known = "#dbe9f7"
        node_color_unknown = "#fde2e4"
        edge_color = "#1d3557"

        n_leaves = len(leaves)
        fig_w = max(5.0, n_leaves * 1.3 + 1.0)
        fig_h = max(3.5, (max_depth + 1) * 1.4 + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # Draw edges
        for n in nodes:
            x1, y1 = positions[id(n)]
            for c in n.children:
                x2, y2 = positions[id(c)]
                ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=1.4,
                        zorder=1)

        # Draw nodes
        for k, n in enumerate(nodes):
            x, y = positions[id(n)]
            v = shown_value[k]
            if v is None:
                fc = node_color_unknown
                label = label_map.get(k, "?")
                color = "#b00020"
            else:
                fc = node_color_known
                label = str(v)
                color = "#1a1a1a"
            circ = plt.Circle((x, y), 0.32, facecolor=fc,
                              edgecolor=edge_color, linewidth=1.4, zorder=2)
            ax.add_patch(circ)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color, zorder=3)

        ax.set_xlim(-0.7, n_leaves - 0.3)
        ax.set_ylim(-0.7, max_depth * 1.2 + 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        op_desc = _OP_DESCRIPTIONS[op]
        ax.set_title(f"Number tree (parent = {op_desc} of children)",
                     fontsize=11, pad=6)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=bg,
                    dpi=130)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
