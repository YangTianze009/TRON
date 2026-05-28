"""
Stack / Queue Trace QA (v3 planning batch, 2026-04-16).

Target: reference algorithm_problems / a puzzle benchmark algorithmic.
Renders a stack (vertical LIFO) or a queue (horizontal FIFO), or both,
with a sequence of operations (push / pop / enqueue / dequeue) drawn as
a numbered list alongside the structure.

Questions (integer short-answer):
  - "After all operations, what is the value at the TOP of the stack?"
  - "What value was the k-th to be OUTPUT (by any pop/dequeue)?"

Difficulty axes:
  1. n_operations = 3 + level (3..12)
  2. structure_type:
       level <= 4  -> stack only
       level <= 7  -> queue only
       level <= 8  -> mixed stack + queue (ask about mixed output order)
       level == 9  -> mixed stack + queue, longer op list
  3. operation_mix: mostly pushes at L0, balanced push/pop by L9.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = {
    "stack": ["Stack Trace", "LIFO Stack", "Stack Operations",
              "Stack Diagram"],
    "queue": ["Queue Trace", "FIFO Queue", "Queue Operations",
              "Queue Diagram"],
    "mixed": ["Stack + Queue Trace", "Mixed LIFO/FIFO Operations",
              "Stack and Queue", "Combined Stack/Queue"],
}

class StackQueueTraceQA(StandaloneVisualEnv):
    ENV_NAME = "stack_queue_trace"

    # ------------------------------------------------------------------ #
    # Level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 4:
            structure = "stack"
        elif level <= 7:
            structure = "queue"
        else:
            structure = "mixed"
        # Fraction of ops that are "remove" (pop/dequeue); rest are "add".
        # L0: 10%, L9: 40%.
        remove_frac = 0.1 + 0.03 * level
        return {
            "n_ops": 3 + level,                 # 3..12
            "structure": structure,
            "remove_frac": remove_frac,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1433)
        self._primary_complexity_feature = cfg["n_ops"] * 10 + level

        for _ in range(30):
            result = self._try_generate(cfg, sub_rng, level)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------ #
    # Problem construction
    # ------------------------------------------------------------------ #

    def _try_generate(self, cfg: Dict, rng: random.Random,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        structure = cfg["structure"]
        n_ops = cfg["n_ops"]
        rfrac = cfg["remove_frac"]

        ops: List[Tuple[str, Optional[int]]] = []
        stack: List[int] = []
        queue: List[int] = []
        outputs: List[int] = []  # in order of emission
        next_val = rng.randint(1, 5)

        def pick_add_structure() -> str:
            if structure == "stack":
                return "stack"
            if structure == "queue":
                return "queue"
            # mixed: choose based on which is smaller (keeps both active)
            return "stack" if rng.random() < 0.5 else "queue"

        def pick_remove_structure() -> Optional[str]:
            # Return which structure to remove from (must be non-empty)
            if structure == "stack":
                return "stack" if stack else None
            if structure == "queue":
                return "queue" if queue else None
            choices = []
            if stack: choices.append("stack")
            if queue: choices.append("queue")
            if not choices:
                return None
            return rng.choice(choices)

        # Build operations
        for _ in range(n_ops):
            want_remove = (rng.random() < rfrac)
            if want_remove:
                tgt = pick_remove_structure()
                if tgt is None:
                    want_remove = False
            if not want_remove:
                tgt = pick_add_structure()
                val = next_val
                next_val = rng.randint(next_val + 1, next_val + 4)
                if tgt == "stack":
                    stack.append(val)
                    ops.append(("push", val))
                else:
                    queue.append(val)
                    ops.append(("enqueue", val))
            else:
                if tgt == "stack":
                    v = stack.pop()
                    outputs.append(v)
                    ops.append(("pop", None))
                else:
                    v = queue.pop(0)
                    outputs.append(v)
                    ops.append(("dequeue", None))

        # --- Decide question type -----------------------------------------
        # Always ask an integer. For stack-only: prefer "top of stack".
        # For queue-only: prefer "front of queue". For mixed or when the
        # stack/queue is empty at the end: ask about the k-th output.
        final_stack_top = stack[-1] if stack else None
        final_queue_front = queue[0] if queue else None
        n_outputs = len(outputs)

        candidates = []
        if structure == "stack" and final_stack_top is not None:
            candidates.append("stack_top")
        if structure == "queue" and final_queue_front is not None:
            candidates.append("queue_front")
        if n_outputs >= 1:
            candidates.append("kth_output")
        # mixed: prefer kth_output to truly require cross-structure tracking
        if structure == "mixed" and n_outputs >= 2:
            candidates = ["kth_output"]

        if not candidates:
            return None
        qtype = rng.choice(candidates)

        struct_desc = self._structure_description(structure)

        if qtype == "stack_top":
            question = (
                f"{struct_desc} The list of operations shown in the image "
                f"is applied in order from top to bottom. (Recall: 'push x' "
                f"places x on top of the stack; 'pop' removes and returns "
                f"the top element.) After all operations have been "
                f"performed, what integer value is at the TOP of the "
                f"stack? Answer with a single integer."
            )
            answer = final_stack_top
        elif qtype == "queue_front":
            question = (
                f"{struct_desc} The operations shown in the image are "
                f"applied in order from top to bottom. (Recall: 'enqueue x' "
                f"adds x to the back of the queue; 'dequeue' removes and "
                f"returns the front element.) After all operations, what "
                f"integer value is at the FRONT of the queue? Answer with "
                f"a single integer."
            )
            answer = final_queue_front
        else:  # kth_output
            k = rng.randint(1, n_outputs)
            ordinal = self._ordinal(k)
            if structure == "mixed":
                rule = (
                    "'push x' pushes x onto the stack; 'pop' removes and "
                    "outputs the top of the stack; 'enqueue x' adds x to "
                    "the back of the queue; 'dequeue' removes and outputs "
                    "the front of the queue. Treat every pop and every "
                    "dequeue as producing one output value, in the order "
                    "they occur in the operation list."
                )
            elif structure == "stack":
                rule = (
                    "'push x' places x on top of the stack; 'pop' removes "
                    "and outputs the top element. Every pop produces one "
                    "output value in the order it occurs."
                )
            else:
                rule = (
                    "'enqueue x' adds x to the back of the queue; "
                    "'dequeue' removes and outputs the front element. "
                    "Every dequeue produces one output value in the order "
                    "it occurs."
                )
            question = (
                f"{struct_desc} Apply the operations in order. {rule} "
                f"What is the {ordinal} value that is output? Answer with "
                f"a single integer."
            )
            answer = outputs[k - 1]

        if answer is None:
            return None

        # BUGFIX 2026-04-24: answer leakage for stack_top / queue_front.
        # The renderer draws the FINAL stack/queue state with "top"/"front"
        # pointer arrows. For those two qtypes the pointer points to the
        # answer cell, trivializing the task. Pass empty stack/queue so the
        # student must trace the ops.
        if qtype == "stack_top":
            render_stack = []
            render_queue = queue
        elif qtype == "queue_front":
            render_stack = stack
            render_queue = []
        else:
            render_stack = stack
            render_queue = queue
        image = self._render(ops, structure, render_stack, render_queue,
                             outputs, rng)
        return question, str(answer), image

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ordinal(k: int) -> str:
        if 10 <= (k % 100) <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
        return f"{k}{suf}"

    @staticmethod
    def _structure_description(structure: str) -> str:
        if structure == "stack":
            return ("The image shows a stack (a LIFO data structure drawn "
                    "vertically, with the top of the stack at the top).")
        if structure == "queue":
            return ("The image shows a queue (a FIFO data structure drawn "
                    "horizontally, with the front of the queue on the "
                    "left and the back on the right).")
        return ("The image shows BOTH a stack (LIFO, vertical) and a "
                "queue (FIFO, horizontal). Each operation acts on exactly "
                "one of them.")

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, ops: List[Tuple[str, Optional[int]]], structure: str,
                final_stack: List[int], final_queue: List[int],
                outputs: List[int], rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        # Layout: left = structure diagram(s), right = operation list
        fig = plt.figure(figsize=(10 * sc, 6.8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.3], wspace=0.2)
        ax_d = fig.add_subplot(gs[0])
        ax_o = fig.add_subplot(gs[1])
        ax_d.set_facecolor(style["bg_color"])
        ax_o.set_facecolor(style["bg_color"])
        ax_d.axis("off")
        ax_o.axis("off")

        # Titles
        title = rng.choice(_TITLE_VARIANTS[structure])
        ax_d.set_title(title, fontsize=fs + 3, fontweight="bold")

        # Draw structure
        if structure == "stack":
            self._draw_stack(ax_d, final_stack, palette, fs, x_center=3.0,
                              y_bottom=0.5)
            ax_d.set_xlim(0, 6); ax_d.set_ylim(-0.2, 7)
        elif structure == "queue":
            self._draw_queue(ax_d, final_queue, palette, fs, y_center=3.0,
                              x_left=0.5)
            ax_d.set_xlim(-0.2, 10); ax_d.set_ylim(0, 6)
        else:
            # mixed
            self._draw_stack(ax_d, final_stack, palette, fs, x_center=1.8,
                              y_bottom=0.5, label="Stack",
                              arrow_on_left=True)
            self._draw_queue(ax_d, final_queue, palette, fs, y_center=1.5,
                              x_left=3.5, label="Queue")
            ax_d.set_xlim(0, 10); ax_d.set_ylim(-0.2, 7)

        # Operations panel
        ax_o.set_xlim(0, 1)
        ax_o.set_ylim(0, 1)
        ax_o.text(0.5, 0.98, "Operations (top -> bottom):",
                   ha="center", va="top",
                   fontsize=fs + 1, fontweight="bold",
                   transform=ax_o.transAxes)

        n_ops = len(ops)
        # Vertical space from 0.02 .. 0.92
        top = 0.92
        bottom = 0.04
        step = (top - bottom) / max(n_ops, 1)
        for i, (op, val) in enumerate(ops):
            y = top - (i + 0.5) * step
            if op == "push":
                text = f"{i+1}. push {val}"
                c = "#2ecc71"
            elif op == "pop":
                text = f"{i+1}. pop"
                c = "#e74c3c"
            elif op == "enqueue":
                text = f"{i+1}. enqueue {val}"
                c = "#3498db"
            else:
                text = f"{i+1}. dequeue"
                c = "#9b59b6"
            ax_o.text(0.08, y, text,
                       ha="left", va="center",
                       fontsize=max(8, fs),
                       color=c, fontweight="bold",
                       transform=ax_o.transAxes,
                       bbox=dict(facecolor="#f6f6f6",
                                 edgecolor="#bbbbbb",
                                 boxstyle="round,pad=0.18"))

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_stack(self, ax, stack_vals: List[int], palette, fs,
                    x_center: float, y_bottom: float,
                    label: Optional[str] = None,
                    arrow_on_left: bool = False):
        """Draw a stack with 'top' labeled; show cells for *visual* purposes
        (they show the final state so the model knows the structure type)."""
        cell_w = 1.4
        cell_h = 0.7
        # Draw a container outline tall enough to visualize max ~6 cells
        n_show = max(1, len(stack_vals))
        n_show = min(n_show, 6)
        container_h = cell_h * max(n_show + 1, 3)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_center - cell_w / 2 - 0.1, y_bottom - 0.1),
            cell_w + 0.2, container_h + 0.2,
            boxstyle="round,pad=0.02",
            facecolor="#ffffff", edgecolor="#2c3e50",
            linewidth=1.4))

        # Show only the top few cells of the final stack
        show_vals = stack_vals[-n_show:]
        for k, v in enumerate(show_vals):
            y = y_bottom + k * cell_h + 0.05
            color = palette[(v + k) % len(palette)]
            ax.add_patch(mpatches.Rectangle(
                (x_center - cell_w / 2, y),
                cell_w, cell_h - 0.05,
                facecolor=color, edgecolor="#2c3e50", linewidth=1.2))
            ax.text(x_center, y + (cell_h - 0.05) / 2, str(v),
                    ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", color="white")
        # Top arrow — place on the left in the mixed layout so it doesn't
        # overlap the queue drawn to the right of the stack.
        top_y = y_bottom + max(len(show_vals), 1) * cell_h + 0.05
        if arrow_on_left:
            ax.annotate("top",
                        xy=(x_center - cell_w / 2 - 0.05, top_y - cell_h / 2),
                        xytext=(x_center - cell_w / 2 - 1.0, top_y - cell_h / 2),
                        ha="right", va="center",
                        fontsize=fs, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#2c3e50"))
        else:
            ax.annotate("top",
                        xy=(x_center + cell_w / 2 + 0.05, top_y - cell_h / 2),
                        xytext=(x_center + cell_w / 2 + 1.0, top_y - cell_h / 2),
                        ha="left", va="center",
                        fontsize=fs, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#2c3e50"))
        if label:
            ax.text(x_center, y_bottom + container_h + 0.45, label,
                    ha="center", va="bottom",
                    fontsize=fs + 1, fontweight="bold", color="#2c3e50")
        else:
            ax.text(x_center, y_bottom + container_h + 0.45, "Stack",
                    ha="center", va="bottom",
                    fontsize=fs + 1, fontweight="bold", color="#2c3e50")

    def _draw_queue(self, ax, queue_vals: List[int], palette, fs,
                    y_center: float, x_left: float,
                    label: Optional[str] = None):
        cell_w = 0.9
        cell_h = 0.9
        n_show = max(1, len(queue_vals))
        n_show = min(n_show, 8)
        container_w = cell_w * max(n_show + 1, 3)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_left - 0.1, y_center - cell_h / 2 - 0.1),
            container_w + 0.2, cell_h + 0.2,
            boxstyle="round,pad=0.02",
            facecolor="#ffffff", edgecolor="#2c3e50",
            linewidth=1.4))
        show_vals = queue_vals[:n_show]
        for k, v in enumerate(show_vals):
            x = x_left + k * cell_w
            color = palette[(v + k) % len(palette)]
            ax.add_patch(mpatches.Rectangle(
                (x, y_center - cell_h / 2 + 0.05),
                cell_w - 0.05, cell_h - 0.1,
                facecolor=color, edgecolor="#2c3e50", linewidth=1.2))
            ax.text(x + (cell_w - 0.05) / 2,
                    y_center, str(v),
                    ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", color="white")
        # Arrows: front, back
        ax.annotate("front",
                    xy=(x_left, y_center + cell_h / 2 + 0.1),
                    xytext=(x_left - 0.2, y_center + cell_h / 2 + 0.85),
                    ha="center", va="bottom",
                    fontsize=fs, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2c3e50"))
        ax.annotate("back",
                    xy=(x_left + container_w - 0.2,
                        y_center + cell_h / 2 + 0.1),
                    xytext=(x_left + container_w - 0.2,
                            y_center + cell_h / 2 + 0.85),
                    ha="center", va="bottom",
                    fontsize=fs, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2c3e50"))
        lbl = label or "Queue"
        ax.text(x_left + container_w / 2, y_center - cell_h / 2 - 0.45,
                lbl, ha="center", va="top",
                fontsize=fs + 1, fontweight="bold", color="#2c3e50")

if __name__ == "__main__":
    env = StackQueueTraceQA()
    for lv in (0, 3, 6, 9):
        for s in range(3):
            ok = env.generate(seed=s, parameter={"level": lv})
            print(f"L{lv} s{s} ok={ok} A={env._answer}")
