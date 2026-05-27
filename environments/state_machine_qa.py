"""
Finite State Machine QA — trace input sequences, identify accepting states.

Capabilities: V9 (arrow/flow parsing), R5 (multi-step reasoning), R2 (logic)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 3 states, alphabet {a,b}, trace 2-symbol input, ask "final state".
L1: 3 states, trace 2-symbol input, ask "is accepting".
L2: 3 states, trace 3-symbol input.
L3: 4 states, trace 3-symbol input.
L4: 4 states, trace 4-symbol input.
L5: 5 states, trace 4-symbol input.
L6: 5 states, trace 5-symbol input.
L7: 6 states, trace 5-symbol input.
L8: 6 states, trace 6-symbol input, accept/reject question.
L9: 6 states, trace 6-symbol input, "find final state".

parameter = {"level": int in [0, 9]}
"""
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = ["Finite State Machine", "FSM", "State Diagram", "Automaton"]

class StateMachineQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "state_machine"

    QUESTION_TYPES = ["trace_input", "is_accepting"]

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
            return {"n_states": 3, "input_len": 2, "qtype": "trace_input"}
        if level == 1:
            return {"n_states": 3, "input_len": 2, "qtype": "is_accepting"}
        if level == 2:
            return {"n_states": 3, "input_len": 3, "qtype": "trace_input"}
        if level == 3:
            return {"n_states": 4, "input_len": 3, "qtype": "trace_input"}
        if level == 4:
            return {"n_states": 4, "input_len": 4, "qtype": "trace_input"}
        if level == 5:
            return {"n_states": 5, "input_len": 4, "qtype": "trace_input"}
        if level == 6:
            return {"n_states": 5, "input_len": 5, "qtype": "trace_input"}
        if level == 7:
            return {"n_states": 6, "input_len": 5, "qtype": "trace_input"}
        if level == 8:
            return {"n_states": 6, "input_len": 6, "qtype": "is_accepting"}
        return {"n_states": 6, "input_len": 6, "qtype": "trace_input"}

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)
        n_states = cfg["n_states"]
        alphabet = ["a", "b"]

        states = [f"S{i}" for i in range(n_states)]
        start = states[0]
        accepting = set(rng.sample(states[1:], rng.randint(1, max(1, n_states // 2))))

        # Generate transitions deterministically per seed
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in alphabet:
                transitions[s][a] = rng.choice(states)

        input_len = cfg["input_len"]
        input_str = "".join(rng.choice(alphabet) for _ in range(input_len))

        current = start
        for ch in input_str:
            current = transitions[current][ch]

        if cfg["qtype"] == "trace_input":
            stems = [
                f"Starting from {start}, what state do you end up in after processing "
                f"input '{input_str}'? Answer with the state name.",
                f"Process the input string '{input_str}' from start state {start}. "
                f"Which state is reached?",
            ]
            answer = current
        else:  # is_accepting
            is_acc = current in accepting
            stems = [
                f"Does the input '{input_str}' get accepted by this state machine "
                f"(starting from {start})? Answer Yes or No.",
                f"Trace '{input_str}' from {start}. Is the final state an accepting "
                f"state? Answer Yes or No.",
            ]
            answer = "Yes" if is_acc else "No"

        question = rng.choice(stems)
        if level <= 2:
            question += (
                " Hint: trace the input one character at a time. Start at "
                f"{start}. For each character, follow the labeled arrow "
                "from the current state to find the next state. Repeat for "
                f"all {len(input_str)} characters. The final state is "
                "the answer (or check if it's accepting for yes/no)."
            )
        image = self._render(rng, states, transitions, accepting, start, alphabet)
        return question, str(answer), image

    def _render(self, rng, states, transitions, accepting, start, alphabet):
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor(style["bg_color"])

        palette = list(style["palette"])
        rng.shuffle(palette)
        n = len(states)
        angles = [2 * np.pi * i / n - np.pi / 2 for i in range(n)]
        positions = {s: (2.5 + 1.8 * np.cos(a), 2.5 + 1.8 * np.sin(a))
                     for s, a in zip(states, angles)}

        for s in states:
            x, y = positions[s]
            color = "#27ae60" if s in accepting else palette[0]
            circle = plt.Circle((x, y), 0.35, facecolor=color, edgecolor="black",
                                linewidth=2, alpha=0.85)
            ax.add_patch(circle)
            if s in accepting:
                inner = plt.Circle((x, y), 0.28, fill=False, edgecolor="black",
                                   linewidth=1)
                ax.add_patch(inner)
            ax.text(x, y, s, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white")
            if s == start:
                ax.annotate("", xy=(x - 0.35, y), xytext=(x - 0.8, y),
                            arrowprops=dict(arrowstyle="->",
                                            color="black", lw=1.5))

        # Draw transitions: group by (s, t) so multiple symbols on same edge merge
        edges = {}
        for s in states:
            for a in alphabet:
                t = transitions[s][a]
                edges.setdefault((s, t), []).append(a)

        for (s, t), syms in edges.items():
            sx, sy = positions[s]
            tx, ty = positions[t]
            label = ",".join(syms)
            if s == t:
                ax.annotate(label, xy=(sx, sy + 0.55), fontsize=10,
                            ha="center", color="#e74c3c", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="lightyellow"))
            else:
                ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="->", color="#7f8c8d",
                                            lw=1.2,
                                            connectionstyle="arc3,rad=0.18"))
                mid_x = (sx + tx) / 2 + 0.18 * (ty - sy)
                mid_y = (sy + ty) / 2 - 0.18 * (tx - sx)
                ax.text(mid_x, mid_y, label, fontsize=10, ha="center",
                        color="#e74c3c", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="lightyellow", alpha=0.9))

        ax.text(0.1, 4.9, f"Start: {start}", fontsize=9, color="#2c3e50")
        ax.text(0.1, 4.65, f"Accepting: {', '.join(sorted(accepting))}",
                fontsize=9, color="#27ae60")
        ax.text(0.1, 4.4, "Alphabet: {a, b}", fontsize=9, color="#7f8c8d")

        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=14, fontweight="bold")
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = StateMachineQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
