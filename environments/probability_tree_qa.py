"""
Probability Tree QA environment.

Capabilities: V3 (chart extraction), R1 (arithmetic), R4 (statistical), R5 (multi-step)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2-branch single-stage tree, integer numerators (e.g. 3/10), ask
    "probability of outcome X" (which is just the leaf prob).
L1: 2-branch single-stage tree, ask which outcome is "most likely".
L2: 2-branch single-stage tree, conditional on first branch.
L3: 2-branch 2-stage tree, ask "single path probability".
L4: 2-branch 2-stage tree, ask "event probability" (sum paths).
L5: 2-branch 2-stage tree, ask "most likely outcome".
L6: 3-branch 2-stage tree, ask "single path probability".
L7: 3-branch 2-stage tree, ask "conditional probability".
L8: 2-branch 3-stage tree, ask "expected value".
L9: 3-branch 3-stage tree, ask "expected value difference" (two payoff schemes).

parameter = {"level": int in [0, 9]}
"""
import math
import random
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_OUTCOME_POOLS = [
    ["Win", "Lose"],
    ["Pass", "Fail"],
    ["Red", "Blue"],
    ["Heads", "Tails"],
    ["Yes", "No"],
    ["A", "B"],
]

_OUTCOME_POOLS_3 = [
    ["Red", "Blue", "Green"],
    ["Win", "Lose", "Draw"],
    ["A", "B", "C"],
    ["Gold", "Silver", "Bronze"],
]

_TITLE_VARIANTS = [
    "Probability Tree",
    "Branching Probability",
    "Decision Tree",
    "Outcome Tree",
    "Event Tree",
    "Probabilistic Branching",
    "Stage Tree",
    "Sequential Probability",
    "Branch Diagram",
    "Likelihood Tree",
    "Decision Branches",
    "Tree of Outcomes",
    "Multi-stage Probability",
    "Probability Diagram",
    "Branching Outcomes",
    "Probability Branches",
]

_TEMPLATES_SINGLE_PATH = [
    "Using the probability tree shown, find the probability of following the path: {path_str}. Give your answer as a decimal rounded to 4 decimal places.",
    "From the probability tree, compute the probability that the outcome follows the path {path_str}. Report as a decimal rounded to 4 decimal places.",
    "What is the probability of tracing the path {path_str} through the probability tree? Answer as a decimal rounded to 4 decimal places.",
    "Given the probability tree above, determine the probability of the path {path_str}. Provide a decimal rounded to 4 decimal places.",
    "In the tree diagram, find P({path_str}). Round to 4 decimal places and report as a decimal.",
    "Using the tree shown, what is the probability of the sequence {path_str}? Answer as a decimal rounded to 4 decimal places.",
    "Compute the probability of the path {path_str} from the probability tree. Give your answer as a decimal with 4 decimal places.",
    "The tree diagram represents a sequence of events. Find the probability of the path {path_str}. Round to 4 decimal places.",
    "Refer to the probability tree. Calculate the joint probability along the path {path_str}. Answer as a 4-decimal decimal.",
    "Trace the path {path_str} through the tree. What is its probability? Use 4 decimal places.",
    "Based on the probability tree, what is the chance of following {path_str}? Give a decimal rounded to 4 decimal places.",
    "From the diagram, determine the probability associated with the path {path_str}. Report as a decimal (4 d.p.).",
    "Using the tree, find P(path = {path_str}). Answer as a decimal rounded to 4 decimal places.",
    "The probability tree depicts multiple branches. Compute P({path_str}). Give the answer as a decimal, 4 decimal places.",
    "Calculate the product of probabilities along the path {path_str} in the tree. Round to 4 decimal places.",
    "From the probability tree, find the probability of this exact branch sequence: {path_str}. Answer as a decimal to 4 places.",
]

_TEMPLATES_EVENT_PROB = [
    "Using the probability tree shown, find the total probability of the outcome \"{target}\". Give your answer as a decimal rounded to 4 decimal places.",
    "From the tree diagram above, compute P({target}). Give a decimal with 4 decimal places.",
    "What is the overall probability of outcome \"{target}\" based on the probability tree? Answer as a decimal, rounded to 4 decimal places.",
    "Using the tree, sum the probabilities of all paths leading to \"{target}\". Report as a decimal to 4 decimal places.",
    "Find the total probability of the outcome labeled \"{target}\" in the probability tree. Give a decimal (4 d.p.).",
    "Refer to the probability tree. Calculate P(outcome = {target}). Round to 4 decimal places.",
    "Based on the diagram, determine the probability of outcome \"{target}\". Answer as a decimal rounded to 4 decimal places.",
    "In the tree shown, compute the probability of event {target}. Round to 4 decimal places.",
    "Using the probability tree, what is the total probability mass for outcome \"{target}\"? Give as a decimal (4 d.p.).",
    "The probability tree shows multiple branches. Compute P({target}) by summing relevant paths. Report as a decimal to 4 decimal places.",
    "From the tree diagram, find the aggregate probability of \"{target}\". Answer as a decimal, 4 decimal places.",
    "What fraction of all paths in the tree yield outcome \"{target}\"? Express as a decimal to 4 decimal places.",
    "Based on the probability tree, find the marginal probability of \"{target}\". Round to 4 decimal places.",
    "Using the tree, sum up all branches ending in \"{target}\" to get its probability. Answer as a decimal to 4 places.",
    "Determine the total probability of outcome \"{target}\" from the shown probability tree. Give a decimal with 4 decimal places.",
    "From the tree, compute the likelihood of \"{target}\" across all paths. Answer as a decimal rounded to 4 decimal places.",
]

_TEMPLATES_MOST_LIKELY = [
    "Using the probability tree, determine which outcome has the highest total probability. Give just the outcome name as your answer.",
    "Based on the probability tree, which outcome is most likely? Answer with the outcome name only.",
    "From the tree diagram, identify the outcome with the greatest total probability. Give the outcome name.",
    "Which outcome in the probability tree has the highest combined probability? Answer with the name.",
    "Looking at the tree, which outcome is the most probable overall? Provide the outcome name.",
    "In the probability tree, find the outcome with the maximum total probability. Answer with the outcome label.",
    "Using the tree, determine the single most likely outcome. Just give the outcome's name.",
    "From the probability tree, name the outcome that has the largest probability. Output only the name.",
    "Which outcome dominates in total probability across the tree? Give the outcome name only.",
    "Based on the tree, which outcome has the greatest chance of occurring? Answer with the label.",
    "In the tree diagram, which outcome carries the highest aggregate probability? Respond with the name.",
    "From the probability tree shown, which outcome is most likely overall? Give the outcome's name.",
    "Identify the outcome with maximum marginal probability in the probability tree. Answer with the name.",
    "Which outcome is most favored by the probability tree? Provide just the outcome name.",
    "Using the probability tree, name the outcome that occurs with the highest probability. Give only the label.",
    "Based on the diagram, which outcome has the best overall chance? Respond with just the outcome name.",
]

_TEMPLATES_CONDITIONAL = [
    "Using the probability tree, given that the first branch taken is \"{fb_label}\", what is the probability that the final outcome is \"{target}\"? Give your answer as a decimal rounded to 4 decimal places.",
    "From the probability tree, suppose the first branch is \"{fb_label}\". Find P(outcome = {target} | first = {fb_label}). Answer as a decimal to 4 decimal places.",
    "Given that the first branch in the tree is \"{fb_label}\", compute the conditional probability of outcome \"{target}\". Round to 4 decimal places.",
    "In the probability tree, if the first stage is \"{fb_label}\", what is the probability of finally reaching \"{target}\"? Answer as a decimal (4 d.p.).",
    "Refer to the tree. Conditional on \"{fb_label}\" being taken first, find P({target}). Give a decimal rounded to 4 decimal places.",
    "Assume the first branch is \"{fb_label}\" in the probability tree. Compute the probability of reaching outcome \"{target}\". Round to 4 decimal places.",
    "Using the tree, what is P({target} | first branch = {fb_label})? Answer as a decimal to 4 decimal places.",
    "From the probability tree, given first branch \"{fb_label}\", compute the chance the final outcome is \"{target}\". Give 4 decimal places.",
    "In the tree, condition on the first branch being \"{fb_label}\" and find P(final = {target}). Answer as a decimal (4 d.p.).",
    "Given the first branch \"{fb_label}\" in the probability tree, what is the conditional probability of outcome \"{target}\"? Round to 4 places.",
    "Using the probability tree, calculate P(outcome = {target} | first = {fb_label}). Report as a decimal rounded to 4 decimal places.",
    "Suppose in the tree we took \"{fb_label}\" first. What is the probability the end outcome is \"{target}\"? Answer as a 4-decimal decimal.",
    "Refer to the tree diagram. If the first stage outcome is \"{fb_label}\", compute the conditional probability of \"{target}\". Round to 4 decimal places.",
    "From the tree, given first = {fb_label}, find the probability that the outcome is \"{target}\". Give a decimal to 4 decimal places.",
    "In the probability tree, if we know the first branch is \"{fb_label}\", what is P(final outcome = {target})? Answer as a decimal (4 d.p.).",
    "Using the tree, conditional on \"{fb_label}\" being the first branch, determine the probability of ending at \"{target}\". Round to 4 decimal places.",
]

_TEMPLATES_EV = [
    "Using the probability tree shown, compute the expected value given these payoffs: {payoff_str}. Round your answer to 2 decimal places.",
    "From the probability tree, calculate the expected value with payoffs: {payoff_str}. Give your answer to 2 decimal places.",
    "Given the payoffs {payoff_str}, compute the expected value using the probability tree. Round to 2 decimal places.",
    "Using the tree and the payoffs {payoff_str}, what is the expected value? Report to 2 decimal places.",
    "Based on the probability tree and the payoff schedule {payoff_str}, determine the expected value. Round to 2 decimal places.",
    "From the tree, compute E[payoff] where payoffs are: {payoff_str}. Give the answer to 2 decimal places.",
    "Using the probability tree shown, and payoffs {payoff_str}, find the expected value. Round to 2 decimal places.",
    "Given payoffs {payoff_str}, use the probability tree to compute E[X]. Round to 2 decimal places.",
    "Refer to the tree. With the following payoffs: {payoff_str}, what is the expected value? Answer to 2 decimal places.",
    "From the probability tree, calculate the expected payoff under {payoff_str}. Report to 2 decimal places.",
    "Compute the expected value of the outcome using the probability tree and payoffs {payoff_str}. Round to 2 decimal places.",
    "Using the tree, combine the payoffs {payoff_str} to find the expected value. Give your answer to 2 decimal places.",
    "Based on the probability tree, determine the expected value for payoffs: {payoff_str}. Round to 2 decimal places.",
    "From the probability tree shown, compute the expected value given the payoff scheme: {payoff_str}. Report to 2 decimal places.",
    "Using the tree and the payoff set {payoff_str}, calculate E[value]. Answer to 2 decimal places.",
    "Given the probability tree and payoffs {payoff_str}, compute the expected value. Round your final answer to 2 decimal places.",
]

class ProbabilityTreeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "probability_tree"

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(20):
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
            return {"depth": 1, "branching": 2, "qtype": "event_probability"}
        if level == 1:
            return {"depth": 1, "branching": 2, "qtype": "most_likely"}
        if level == 2:
            return {"depth": 1, "branching": 2, "qtype": "single_path"}
        if level == 3:
            return {"depth": 2, "branching": 2, "qtype": "single_path"}
        if level == 4:
            return {"depth": 2, "branching": 2, "qtype": "event_probability"}
        if level == 5:
            return {"depth": 2, "branching": 2, "qtype": "most_likely"}
        if level == 6:
            return {"depth": 2, "branching": 3, "qtype": "single_path"}
        if level == 7:
            return {"depth": 2, "branching": 3, "qtype": "conditional"}
        if level == 8:
            return {"depth": 3, "branching": 2, "qtype": "expected_value"}
        return {"depth": 3, "branching": 3, "qtype": "expected_value"}

    def _make_probs(self, rng, n):
        denom = rng.choice([4, 5, 6, 8, 10])
        nums = [rng.randint(1, denom - n + 1) for _ in range(n - 1)]
        total = sum(nums)
        if total >= denom:
            return None
        nums.append(denom - total)
        rng.shuffle(nums)
        return [Fraction(x, denom) for x in nums]

    def _make_tree(self, rng, depth, branching):
        if branching == 2:
            pool = rng.choice(_OUTCOME_POOLS)
        else:
            pool = rng.choice(_OUTCOME_POOLS_3)
        outcome_labels = pool[:branching]

        def _build(d):
            if d == 0:
                return None
            probs = None
            for _ in range(20):
                probs = self._make_probs(rng, branching)
                if probs is not None:
                    break
            if probs is None:
                return None
            children = []
            if d == 1:
                for i in range(branching):
                    children.append({
                        "prob": probs[i],
                        "label": outcome_labels[i],
                        "outcome": outcome_labels[i],
                    })
            else:
                stage_labels = outcome_labels
                for i in range(branching):
                    sub = _build(d - 1)
                    if sub is None:
                        return None
                    children.append({
                        "prob": probs[i],
                        "label": stage_labels[i],
                        "children": sub,
                    })
            return children

        tree = _build(depth)
        if tree is None:
            return None, None

        leaves = []

        def _collect(node_list, path, cum_prob):
            for child in node_list:
                p = cum_prob * child["prob"]
                new_path = path + [child["label"]]
                if "outcome" in child:
                    leaves.append((new_path, p, child["outcome"]))
                else:
                    _collect(child["children"], new_path, p)

        _collect(tree, [], Fraction(1))
        return tree, leaves

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)
        tree, leaves = self._make_tree(rng, cfg["depth"], cfg["branching"])
        if tree is None or len(leaves) == 0:
            return None

        qtype = cfg["qtype"]
        sidx = (self.seed or 0) % 16

        if qtype == "single_path":
            leaf = rng.choice(leaves)
            path_labels, prob, outcome = leaf
            answer = round(float(prob), 4)
            path_str = " -> ".join(path_labels)
            question = _TEMPLATES_SINGLE_PATH[sidx].format(path_str=path_str)
            return question, str(answer), self._draw_tree(tree, leaves)

        if qtype == "event_probability":
            outcomes = list({l[2] for l in leaves})
            target = rng.choice(outcomes)
            total = sum(l[1] for l in leaves if l[2] == target)
            answer = round(float(total), 4)
            question = _TEMPLATES_EVENT_PROB[sidx].format(target=target)
            return question, str(answer), self._draw_tree(tree, leaves)

        if qtype == "most_likely":
            outcome_probs = {}
            for _, prob, outcome in leaves:
                outcome_probs[outcome] = outcome_probs.get(outcome, Fraction(0)) + prob
            best = max(outcome_probs, key=lambda k: outcome_probs[k])
            question = _TEMPLATES_MOST_LIKELY[sidx]
            return question, best, self._draw_tree(tree, leaves)

        if qtype == "conditional":
            first_branches = {}
            for path, prob, outcome in leaves:
                fb = path[0]
                first_branches.setdefault(fb, []).append((path, prob, outcome))
            fb_label = rng.choice(list(first_branches.keys()))
            fb_leaves = first_branches[fb_label]
            outcomes_in_branch = list({l[2] for l in fb_leaves})
            target = rng.choice(outcomes_in_branch)
            branch_total = sum(l[1] for l in fb_leaves)
            target_total = sum(l[1] for l in fb_leaves if l[2] == target)
            cond = target_total / branch_total
            answer = round(float(cond), 4)
            question = _TEMPLATES_CONDITIONAL[sidx].format(
                fb_label=fb_label, target=target)
            return question, str(answer), self._draw_tree(tree, leaves)

        if qtype == "expected_value":
            outcomes = list({l[2] for l in leaves})
            payoffs = {o: rng.randint(1, 15) for o in outcomes}
            ev = sum(float(prob) * payoffs[outcome] for _, prob, outcome in leaves)
            answer = round(ev, 2)
            payoff_str = ", ".join(f'"{k}" = ${v}' for k, v in payoffs.items())
            question = _TEMPLATES_EV[sidx].format(payoff_str=payoff_str)
            return question, str(answer), self._draw_tree(tree, leaves, payoffs)
        return None

    def _draw_tree(self, tree, leaves, payoffs=None):
        style = self._random_style()
        srng = random.Random((self.seed or 0) * 8831 + 421)
        bg_override_pool = ["#ffffff", "#fdfdfb", "#fafafa", "#f7f9fc",
                            "#f4f6fb", "#fffdf6", "#fff8f0", "#eef6ff",
                            "#fef6f8", "#f5fbf4", "#f8fafc", "#fdfaf0"]
        style_bg = srng.choice(bg_override_pool)
        s = style["figsize_scale"] * srng.uniform(0.92, 1.08)
        fw_jitter = srng.uniform(0.9, 1.1)
        fh_jitter = srng.uniform(0.9, 1.1)
        dpi_val = srng.choice([96, 100, 110, 115, 120])
        # Scale figure width with leaf count so dense trees don't overlap.
        n_leaves = len(leaves)
        fig_w = max(11.0, n_leaves * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w * s * fw_jitter, 7 * s * fh_jitter),
                               dpi=dpi_val)
        fig.patch.set_facecolor(style_bg)
        ax.set_facecolor(style_bg)
        ax.axis("off")
        palette = list(style["palette"])
        edge_color = srng.choice(["#2c3e50", "#34495e", "#1a1a1a",
                                   "#222831", "#2d3436", style["geo_line_color"]])
        lw = style["line_width"] * srng.uniform(0.85, 1.2)
        # Shrink font on dense trees so labels don't overlap.
        fs = style["font_size_base"] + srng.randint(-1, 1)
        if n_leaves >= 18:
            fs = max(7, fs - 3)
        elif n_leaves >= 12:
            fs = max(8, fs - 2)
        ff = style["font_family"]
        leaf_marker = srng.choice(["s", "o", "D", "P", "h"])
        interior_marker = srng.choice(["o", "s", "D", "h"])

        positions = {}
        labels_map = {}
        node_id = [0]

        def _count_leaves(node_list):
            c = 0
            for child in node_list:
                if "outcome" in child:
                    c += 1
                else:
                    c += _count_leaves(child["children"])
            return c

        total_leaves = _count_leaves(tree)
        y_step = 2.2

        def _layout(node_list, parent_id, depth, x_start, x_end, y):
            subtree_sizes = []
            for child in node_list:
                if "outcome" in child:
                    subtree_sizes.append(1)
                else:
                    subtree_sizes.append(_count_leaves(child["children"]))
            total_s = sum(subtree_sizes)
            x_cur = x_start

            for i, child in enumerate(node_list):
                nid = node_id[0]
                node_id[0] += 1
                span = (x_end - x_start) * subtree_sizes[i] / total_s
                cx = x_cur + span / 2
                cy = y - y_step
                positions[nid] = (cx, cy)

                f = child["prob"]
                if f.denominator <= 10:
                    prob_str = f"{f.numerator}/{f.denominator}"
                else:
                    prob_str = f"{float(f):.3f}"

                labels_map[(parent_id, i)] = (prob_str, child["label"], nid)

                if "children" in child:
                    _layout(child["children"], nid, depth + 1,
                            x_cur, x_cur + span, cy)
                x_cur += span

        root_id = node_id[0]
        node_id[0] += 1
        root_x = total_leaves / 2
        positions[root_id] = (root_x, 0)
        _layout(tree, root_id, 0, 0, total_leaves, 0)

        for (pid, cidx), (prob_str, label, nid) in labels_map.items():
            px, py = positions[pid]
            cx, cy = positions[nid]
            ax.plot([px, cx], [py, cy], "-", color=edge_color, linewidth=lw, zorder=1)

            # Stagger label position along edge when there are many children
            # to avoid horizontal collisions between sibling labels.
            t = 0.5
            if n_leaves >= 9:
                t = 0.42 + 0.08 * ((cidx % 3))
            mx = px + t * (cx - px)
            my = py + t * (cy - py)
            ax.text(mx, my, prob_str, fontsize=fs, ha="center", va="center",
                    color="#1a1a1a", fontweight="bold", zorder=5,
                    fontfamily=ff,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#888", alpha=0.95))

        root_marker = srng.choice(["o", "D", "s", "h", "p"])
        root_size = srng.choice([12, 14, 16, 18])
        leaf_size = srng.choice([12, 14, 16])
        int_size = srng.choice([10, 12, 14])
        for nid, (x, y) in positions.items():
            if nid == root_id:
                ax.plot(x, y, root_marker, markersize=root_size,
                        color=palette[7 % len(palette)], zorder=6)
                ax.text(x, y, "S", fontsize=fs - 2, ha="center", va="center",
                        color="white", fontweight="bold", zorder=7)
            else:
                is_leaf = not any(pid == nid for (pid, _) in labels_map)
                if is_leaf:
                    lbl = None
                    for (pid2, cidx2), (ps, lb, cid) in labels_map.items():
                        if cid == nid:
                            lbl = lb
                            break
                    ax.plot(x, y, leaf_marker, markersize=leaf_size,
                            color=palette[0], zorder=6)
                    txt = lbl if lbl else ""
                    if payoffs and lbl in payoffs:
                        txt += f"\n${payoffs[lbl]}"
                    ax.text(x, y - 0.6, txt, fontsize=fs, ha="center",
                            va="top", fontweight="bold", color=edge_color,
                            zorder=7, fontfamily=ff)
                else:
                    ax.plot(x, y, interior_marker, markersize=int_size,
                            color=palette[1 % len(palette)], zorder=6)
                    # Add intermediate label
                    lbl = None
                    for (pid2, cidx2), (ps, lb, cid) in labels_map.items():
                        if cid == nid:
                            lbl = lb
                            break
                    if lbl:
                        ax.text(x, y + 0.35, lbl, fontsize=fs - 1, ha="center",
                                va="bottom", color=edge_color, fontweight="bold",
                                zorder=7)

        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
        ax.set_ylim(min(all_y) - 1.8, max(all_y) + 0.6)
        title_idx = (self.seed or 0) % len(_TITLE_VARIANTS)
        ax.set_title(_TITLE_VARIANTS[title_idx], fontsize=fs + 3, fontweight="bold",
                     pad=10, fontfamily=ff)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = ProbabilityTreeQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
