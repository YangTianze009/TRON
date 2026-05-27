"""
Tree Traversal Visit QA (D139, P1).

Reference task:
  qid 241: "Which node will be visited first if using pre-order DFS on the
           following tree graph?"  Ans: D
  qid 243: "Which node will be secondly visited if using post-order DFS
           on the following tree graph?"  Ans: E

Generates a small labeled rooted tree (3-8 vertices, letter labels A-H).
Asks for the n-th visited node under {pre, in, post}-order DFS.

Verifier: single-letter label (`\\boxed{D}` or bare `D`).
"""
import math
import random
import string
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


class TreeNode:
    __slots__ = ("label", "children", "x", "y")
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []
        self.x = 0.0
        self.y = 0.0


def _preorder(node, out):
    out.append(node.label)
    for c in node.children:
        _preorder(c, out)


def _inorder(node, out):
    """Inorder for general trees: visit first child(ren), then node, then rest.
    For binary-like trees: left, node, right. For >2 children: split children
    in half: visit first half, then node, then second half. We use this
    convention which is standard for reference problems."""
    n_children = len(node.children)
    if n_children == 0:
        out.append(node.label)
        return
    if n_children == 1:
        _inorder(node.children[0], out)
        out.append(node.label)
        return
    # Binary or more: left = first child, then node, then rest
    _inorder(node.children[0], out)
    out.append(node.label)
    for c in node.children[1:]:
        _inorder(c, out)


def _postorder(node, out):
    for c in node.children:
        _postorder(c, out)
    out.append(node.label)


def _build_random_tree(rng: random.Random, n: int) -> TreeNode:
    """Build a random rooted tree with n nodes."""
    labels = list(string.ascii_uppercase[:n])
    rng.shuffle(labels)
    nodes = [TreeNode(lab) for lab in labels]
    # Build parent connections: node i (i>=1) attaches to a random earlier node
    for i in range(1, n):
        parent_idx = rng.randint(0, i - 1)
        # cap children at 3
        if len(nodes[parent_idx].children) >= 3:
            # find another parent
            for p in range(i):
                if len(nodes[p].children) < 3:
                    parent_idx = p
                    break
        nodes[parent_idx].children.append(nodes[i])
    return nodes[0]


def _layout(root, x_lo, x_hi, y):
    """Compute (x, y) layout with leaves spread evenly across [x_lo, x_hi]."""
    leaves = []
    def collect_leaves(n):
        if not n.children:
            leaves.append(n)
        for c in n.children:
            collect_leaves(c)
    collect_leaves(root)
    if not leaves:
        return
    if len(leaves) == 1:
        leaves[0].x = (x_lo + x_hi) / 2
    else:
        for i, lf in enumerate(leaves):
            lf.x = x_lo + (x_hi - x_lo) * i / (len(leaves) - 1)
    # Internal nodes positioned at average of children
    def assign_x(n, depth=0):
        n.y = -depth
        for c in n.children:
            assign_x(c, depth + 1)
        if n.children:
            n.x = sum(c.x for c in n.children) / len(n.children)
    assign_x(root, 0)


class TreeTraversalVisitQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "tree_traversal_visit"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_nodes": 4, "ordinals": [1]}
        if level <= 4:
            return {"n_nodes": 5, "ordinals": [1, 2]}
        if level <= 6:
            return {"n_nodes": 6, "ordinals": [1, 2, 3]}
        if level <= 8:
            return {"n_nodes": 7, "ordinals": [2, 3, 4]}
        return {"n_nodes": 8, "ordinals": [3, 4, 5]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4099 + level * 41 + 19)

        n = cfg["n_nodes"]
        root = _build_random_tree(rng, n)

        order_type = rng.choice(["pre", "in", "post"])
        ordinal = rng.choice(cfg["ordinals"])
        if ordinal > n:
            ordinal = n

        out = []
        if order_type == "pre":
            _preorder(root, out)
            order_word = "pre-order"
        elif order_type == "in":
            _inorder(root, out)
            order_word = "in-order"
        else:
            _postorder(root, out)
            order_word = "post-order"

        if len(out) < ordinal:
            return None
        target_label = out[ordinal - 1]

        ord_word = {1: "first", 2: "second", 3: "third", 4: "fourth",
                    5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}.get(ordinal, f"{ordinal}-th")

        question = (
            f"Which node will be visited {ord_word} if using {order_word} DFS "
            f"on the rooted tree shown in the figure? Give a single letter "
            f"label."
        )

        img = self._render(root)
        return question, target_label, img

    # ------------------------------------------------------------------ #
    def _render(self, root) -> Image.Image:
        # Find depth and breadth for canvas size
        def max_depth(n):
            if not n.children:
                return 0
            return 1 + max(max_depth(c) for c in n.children)
        def count_leaves(n):
            if not n.children:
                return 1
            return sum(count_leaves(c) for c in n.children)
        d = max_depth(root)
        leaves = count_leaves(root)
        w = max(6, leaves * 1.5)
        h = max(4, (d + 1) * 1.0)

        _layout(root, 0, w, 0)

        fig, ax = plt.subplots(figsize=(w, h), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Edges
        def draw_edges(n):
            for c in n.children:
                ax.plot([n.x, c.x], [n.y, c.y], color="#7f8c8d", linewidth=1.5)
                draw_edges(c)
        draw_edges(root)

        # Nodes
        def draw_nodes(n):
            ax.scatter([n.x], [n.y], s=600, color="#3498db",
                       edgecolor="#2c3e50", linewidth=2, zorder=5)
            ax.text(n.x, n.y, n.label, ha="center", va="center",
                    fontsize=14, color="white", fontweight="bold", zorder=6)
            for c in n.children:
                draw_nodes(c)
        draw_nodes(root)

        ax.set_xlim(-0.5, w + 0.5)
        ax.set_ylim(-d - 1, 1)
        ax.axis("off")
        ax.set_aspect("auto")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = TreeTraversalVisitQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
