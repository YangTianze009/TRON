"""Follow cause-effect arrows in branching causal diagrams."""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class CausalDiagramQA(StandaloneVisualEnv):
    ENV_NAME = "causal_diagram"

    # Real-world themed node labels per domain
    _DOMAINS = [
        # Economics
        ["Inflation", "Interest Rate", "Unemployment", "GDP Growth", "Consumer Spending",
         "Tax Revenue", "Imports", "Exports"],
        # Health
        ["Smoking", "Lung Disease", "Exercise", "Heart Health", "Stress",
         "Sleep Quality", "Blood Pressure", "Obesity"],
        # Environment
        ["CO2 Emissions", "Temperature", "Ice Melt", "Sea Level", "Deforestation",
         "Biodiversity Loss", "Drought", "Crop Yield"],
        # Technology
        ["R&D Spending", "Innovation", "Productivity", "Market Share", "Revenue",
         "User Growth", "Data Volume", "AI Accuracy"],
    ]

    _QUESTION_TYPES = [
        "direct_cause",       # What directly causes X?
        "direct_effect",      # What does X directly cause?
        "chain_length",       # How many steps from X to Y?
        "all_ancestors",      # List all causes (direct+indirect) of X
        "common_cause",       # What is the common cause of X and Y?
        "total_effects",      # How many nodes does X affect (directly or indirectly)?
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"qtypes": ["direct_cause", "direct_effect"],
                    "n_nodes": (4, 5)}
        if level <= 5:
            return {"qtypes": ["direct_cause", "direct_effect",
                               "chain_length", "total_effects"],
                    "n_nodes": (5, 6)}
        if level <= 7:
            return {"qtypes": ["chain_length", "all_ancestors",
                               "total_effects", "common_cause"],
                    "n_nodes": (6, 7)}
        return {"qtypes": ["all_ancestors", "common_cause",
                           "chain_length"],
                "n_nodes": (7, 8)}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        style = self._random_style()
        palette = style["palette"]

        q_type = parameter.get("question_type", rng.choice(cfg["qtypes"]))
        lo, hi = cfg["n_nodes"]
        n_nodes = parameter.get("n_nodes", rng.randint(lo, hi))

        # Pick a domain and select node labels
        domain = rng.choice(self._DOMAINS)
        labels = rng.sample(domain, min(n_nodes, len(domain)))
        while len(labels) < n_nodes:
            labels.append(f"Factor {len(labels)+1}")

        # Build graph: either chain or tree (with possible branches)
        # edges: list of (src_idx, dst_idx)
        edges = []
        if q_type in ("common_cause", "all_ancestors") and n_nodes >= 5:
            # Tree structure: root → branches
            # Root connects to 2-3 children, each child may have 1-2 children
            root = 0
            children = list(range(1, min(4, n_nodes)))
            for c in children:
                edges.append((root, c))
            remaining = list(range(len(children) + 1, n_nodes))
            for r in remaining:
                parent = rng.choice(children)
                edges.append((parent, r))
        else:
            # Linear chain with possible side branches
            for i in range(n_nodes - 1):
                edges.append((i, i + 1))
            # Optionally add a branch
            if n_nodes >= 5 and rng.random() < 0.4:
                src = rng.randint(0, n_nodes - 3)
                dst = rng.randint(src + 2, n_nodes - 1)
                if (src, dst) not in edges:
                    edges.append((src, dst))

        # Compute adjacency for question answering
        children_of = {i: [] for i in range(n_nodes)}
        parents_of = {i: [] for i in range(n_nodes)}
        for s, d in edges:
            children_of[s].append(d)
            parents_of[d].append(s)

        def all_descendants(node):
            visited = set()
            stack = [node]
            while stack:
                n = stack.pop()
                for c in children_of[n]:
                    if c not in visited:
                        visited.add(c)
                        stack.append(c)
            return visited

        def all_ancestors(node):
            visited = set()
            stack = [node]
            while stack:
                n = stack.pop()
                for p in parents_of[n]:
                    if p not in visited:
                        visited.add(p)
                        stack.append(p)
            return visited

        def shortest_path_len(src, dst):
            from collections import deque
            visited = {src}
            queue = deque([(src, 0)])
            while queue:
                n, d = queue.popleft()
                if n == dst:
                    return d
                for c in children_of[n]:
                    if c not in visited:
                        visited.add(c)
                        queue.append((c, d + 1))
            return -1

        # Generate question and answer
        if q_type == "direct_cause":
            # Pick a node with at least one parent
            candidates = [i for i in range(n_nodes) if parents_of[i]]
            if not candidates:
                return None
            target = rng.choice(candidates)
            if len(parents_of[target]) == 1:
                question = f"What directly causes {labels[target]}?"
                answer = labels[parents_of[target][0]]
            else:
                question = f"How many factors directly cause {labels[target]}?"
                answer = str(len(parents_of[target]))

        elif q_type == "direct_effect":
            candidates = [i for i in range(n_nodes) if children_of[i]]
            if not candidates:
                return None
            target = rng.choice(candidates)
            if len(children_of[target]) == 1:
                question = f"What does {labels[target]} directly cause?"
                answer = labels[children_of[target][0]]
            else:
                question = f"How many factors does {labels[target]} directly affect?"
                answer = str(len(children_of[target]))

        elif q_type == "chain_length":
            # Find two nodes with a path
            src = rng.randint(0, n_nodes - 2)
            candidates = list(all_descendants(src))
            if not candidates:
                return None
            dst = rng.choice(list(candidates))
            d = shortest_path_len(src, dst)
            question = f"How many causal steps are there from {labels[src]} to {labels[dst]}?"
            answer = str(d)

        elif q_type == "all_ancestors":
            candidates = [i for i in range(n_nodes) if all_ancestors(i)]
            if not candidates:
                return None
            target = rng.choice(candidates)
            anc = all_ancestors(target)
            question = f"How many factors (directly or indirectly) cause {labels[target]}?"
            answer = str(len(anc))

        elif q_type == "common_cause":
            # Find two nodes with a common ancestor
            leaf_nodes = [i for i in range(n_nodes) if not children_of[i]]
            if len(leaf_nodes) < 2:
                leaf_nodes = list(range(1, n_nodes))
            if len(leaf_nodes) < 2:
                return None
            pair = rng.sample(leaf_nodes, 2)
            anc0 = all_ancestors(pair[0]) | {pair[0]}
            anc1 = all_ancestors(pair[1]) | {pair[1]}
            common = anc0 & anc1
            if not common:
                # Fall back to direct_cause
                target = pair[0] if parents_of[pair[0]] else pair[1]
                if not parents_of[target]:
                    return None
                question = f"What directly causes {labels[target]}?"
                answer = labels[parents_of[target][0]]
            else:
                question = f"What is the common ancestor (root cause) that affects both {labels[pair[0]]} and {labels[pair[1]]}?"
                # Pick the deepest common ancestor
                best = max(common, key=lambda x: len(all_ancestors(x)))
                answer = labels[best]

        elif q_type == "total_effects":
            candidates = [i for i in range(n_nodes) if all_descendants(i)]
            if not candidates:
                return None
            target = rng.choice(candidates)
            desc = all_descendants(target)
            question = f"How many nodes does {labels[target]} affect (directly or indirectly)?"
            answer = str(len(desc))
        else:
            return None

        # Render diagram
        # Layout: use layered positioning
        # Compute depth for each node
        depth = [0] * n_nodes
        for _ in range(n_nodes):
            for s, d in edges:
                depth[d] = max(depth[d], depth[s] + 1)
        max_depth = max(depth) if depth else 0

        # Group by depth
        layers = {}
        for i in range(n_nodes):
            layers.setdefault(depth[i], []).append(i)

        # Assign positions
        pos = {}
        for d, nodes_at_d in layers.items():
            n_at_d = len(nodes_at_d)
            for j, node_idx in enumerate(nodes_at_d):
                x = d * 3.0 + 1.5
                y = (j - (n_at_d - 1) / 2) * 2.0 + 2.0
                pos[node_idx] = (x, y)

        fig_w = max(8, (max_depth + 1) * 3 + 2)
        fig_h = max(4, max(len(v) for v in layers.values()) * 2 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w * style["figsize_scale"],
                                        fig_h * style["figsize_scale"]))
        fig.patch.set_facecolor(style["bg_color"])

        # Draw edges
        for s, d in edges:
            sx, sy = pos[s]
            dx, dy = pos[d]
            ax.annotate("", xy=(dx - 0.55, dy), xytext=(sx + 0.55, sy),
                        arrowprops=dict(arrowstyle="-|>", color=style["geo_line_color"],
                                       lw=style["line_width"] + 0.5,
                                       connectionstyle="arc3,rad=0.1" if abs(sy - dy) > 0.5 else "arc3,rad=0"))

        # Draw nodes
        for i in range(n_nodes):
            x, y = pos[i]
            color = palette[i % len(palette)]
            if style["node_shape"] == "rounded_rect":
                bbox = mpatches.FancyBboxPatch((x - 0.55, y - 0.35), 1.1, 0.7,
                                               boxstyle="round,pad=0.1",
                                               facecolor=color, edgecolor="black",
                                               linewidth=1.5, alpha=0.85)
                ax.add_patch(bbox)
            else:
                circle = plt.Circle((x, y), 0.5, facecolor=color,
                                    edgecolor="black", linewidth=1.5, alpha=0.85)
                ax.add_patch(circle)
            # Label: use short name for readability
            label = labels[i]
            if len(label) > 12:
                # Split into two lines
                words = label.split()
                mid = len(words) // 2
                label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
            import matplotlib.patheffects as path_effects
            txt = ax.text(x, y, label, ha="center", va="center",
                    fontsize=max(7, style["font_size_base"] - 3),
                    fontweight="bold", color="#111111",
                    fontfamily=style["font_family"])
            txt.set_path_effects([path_effects.Stroke(linewidth=2.2, foreground="white"),
                                   path_effects.Normal()])

        margin = 1.0
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Causal Diagram", fontsize=style["font_size_base"] + 2,
                      fontweight="bold", fontfamily=style["font_family"])

        return question, answer, self.fig_to_pil(fig, dpi=style["dpi"])

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Override: more flexible matching for causal diagram answers."""
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()

        # Exact match
        if predicted == ground_truth:
            return True

        # Numeric match (for count questions)
        try:
            p_val = float(predicted)
            g_val = float(ground_truth)
            return abs(p_val - g_val) < 0.5
        except ValueError:
            pass

        # Check if ground_truth is contained in predicted (flexible matching)
        if ground_truth in predicted and len(predicted) < len(ground_truth) * 4:
            return True

        # Check if predicted is a key part of ground_truth
        # e.g., predicted="inflation" matches ground_truth="inflation"
        # but also predicted="co2 emissions" matches "co2 emissions"
        if predicted in ground_truth and len(predicted) > 2:
            return True

        return False
