"""
Circuit Logic QA environment.

Renders a logic gate circuit (AND, OR, NOT, XOR).
Questions: output for given inputs, which input flips output, gate count.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_GATE_FNS = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
}

def _eval_circuit(gates, input_vals):
    """Evaluate a layered circuit. gates is list of (gate_type, in1, in2).
    Inputs are 'A','B','C','D'; intermediates are 'G0','G1',..."""
    vals = dict(input_vals)
    for i, (gtype, in1, in2) in enumerate(gates):
        if gtype == "NOT":
            vals[f"G{i}"] = 1 - vals[in1]
        else:
            vals[f"G{i}"] = _GATE_FNS[gtype](vals[in1], vals[in2])
    # Last gate output
    return vals[f"G{len(gates)-1}"], vals

class CircuitLogicQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "circuit_logic"

    QUESTION_TYPES = [
        "compute_output", "flip_input", "gate_count",
        "intermediate_value", "output_if_all_ones",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Redesigned: L0 starts with 3 inputs + intermediate_value (old L5-6).
        # L9 has 5 inputs, truth_table_count, and flip_input.
        if level <= 1:
            return {"qtypes": ["compute_output", "intermediate_value",
                               "output_if_all_ones"],
                    "n_inputs": 3}
        if level <= 3:
            return {"qtypes": ["compute_output", "flip_input",
                               "intermediate_value"],
                    "n_inputs": 3}
        if level <= 5:
            return {"qtypes": ["flip_input", "truth_table_count",
                               "compute_output"],
                    "n_inputs": 4, "n_gate_layers": 3}
        if level <= 7:
            return {"qtypes": ["truth_table_count", "flip_input"],
                    "n_inputs": 4, "n_gate_layers": 3}
        return {"qtypes": ["truth_table_count", "flip_input"],
                "n_inputs": 5, "n_gate_layers": 4}

    def _generate_problem(self, seed, parameter):
        rng = self._rng
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        qtype = parameter.get("question_type", rng.choice(cfg["qtypes"]))

        # Build a small circuit
        n_inputs = cfg["n_inputs"]
        input_names = ["A", "B", "C", "D", "E"][:n_inputs]
        input_vals = {name: rng.randint(0, 1) for name in input_names}

        # Layer 1: combine inputs
        gate_types_pool = ["AND", "OR", "XOR", "NAND"]
        gates = []
        if n_inputs == 2:
            g1 = rng.choice(gate_types_pool)
            gates.append((g1, "A", "B"))
            # Optionally add a NOT
            if rng.random() < 0.5:
                gates.append(("NOT", "G0", "G0"))
        elif n_inputs == 3:
            g1 = rng.choice(gate_types_pool)
            g2 = rng.choice(gate_types_pool)
            gates.append((g1, "A", "B"))
            gates.append((g2, "G0", "C"))
        elif n_inputs == 4:
            # 4 inputs: 3 gate layers
            g1 = rng.choice(gate_types_pool)
            g2 = rng.choice(gate_types_pool)
            g3 = rng.choice(gate_types_pool)
            gates.append((g1, "A", "B"))       # G0
            gates.append((g2, "C", "D"))       # G1
            gates.append((g3, "G0", "G1"))     # G2
        else:
            # 5 inputs: 4 gate layers
            g1 = rng.choice(gate_types_pool)
            g2 = rng.choice(gate_types_pool)
            g3 = rng.choice(gate_types_pool)
            g4 = rng.choice(gate_types_pool)
            gates.append((g1, "A", "B"))       # G0
            gates.append((g2, "C", "D"))       # G1
            gates.append((g3, "G0", "E"))      # G2
            gates.append((g4, "G1", "G2"))     # G3

        output, all_vals = _eval_circuit(gates, input_vals)

        # Pick qtype BEFORE rendering so we can hide the final output when the
        # task asks the solver to compute the output itself. Otherwise the
        # rendered "Out=<value>" label leaks the answer.
        hide_out = qtype in ("compute_output", "output_if_all_ones",
                             "truth_table_count", "flip_input")
        img = self._render(gates, input_names, input_vals, all_vals,
                           hide_out=hide_out)

        if qtype == "compute_output":
            iv_str = ", ".join(f"{k}={v}" for k, v in input_vals.items())
            q = f"Given inputs {iv_str}, what is the circuit output (0 or 1)?"
            a = str(output)

        elif qtype == "flip_input":
            # Try flipping each input, see which changes output
            flippers = []
            for name in input_names:
                flipped = dict(input_vals)
                flipped[name] = 1 - flipped[name]
                new_out, _ = _eval_circuit(gates, flipped)
                if new_out != output:
                    flippers.append(name)
            if not flippers:
                q = ("Which single input, if flipped, changes the output? "
                     "Answer 'none' if no single flip changes it.")
                a = "none"
            else:
                target = rng.choice(flippers)
                q = (f"If input {target} is flipped (from {input_vals[target]} to "
                     f"{1-input_vals[target]}), does the output change? Answer Yes or No.")
                a = "Yes"

        elif qtype == "gate_count":
            q = "How many logic gates are in this circuit?"
            a = str(len(gates))

        elif qtype == "intermediate_value":
            if len(gates) > 1:
                g_idx = 0
                q = (f"What is the output of the first gate ({gates[0][0]}) "
                     f"in the circuit?")
                a = str(all_vals["G0"])
            else:
                q = f"What is the output of the {gates[0][0]} gate?"
                a = str(output)

        elif qtype == "output_if_all_ones":
            all_ones = {name: 1 for name in input_names}
            out_ones, _ = _eval_circuit(gates, all_ones)
            q = "If all inputs are set to 1, what is the circuit output?"
            a = str(out_ones)

        elif qtype == "truth_table_count":
            # Count how many input combinations produce output=1
            count = 0
            for bits in range(2 ** n_inputs):
                test_vals = {}
                for idx_b, name in enumerate(input_names):
                    test_vals[name] = (bits >> idx_b) & 1
                out_val, _ = _eval_circuit(gates, test_vals)
                if out_val == 1:
                    count += 1
            q = (f"This circuit has {n_inputs} inputs. Out of all "
                 f"{2**n_inputs} possible input combinations, how many "
                 f"produce an output of 1?")
            a = str(count)
        else:
            return None

        return q, a, img

    def _render(self, gates, input_names, input_vals, all_vals,
                hide_out=False):
        style = self._random_style()
        s = style["figsize_scale"]

        # Dynamic layout: enough vertical room for all inputs, and extra
        # horizontal room for larger circuits.
        n_in = len(input_names)
        n_gates = len(gates)
        # Evenly space inputs in [y_bot, y_top]; pad the figure height so even
        # n_in=5 leaves space for title and routing.
        y_top_plot = 1.5 + n_in * 1.2
        y_bot_plot = 0.5
        fig_w = 8 * s if n_in <= 4 else 10 * s
        fig_h = 5 * s if n_in <= 4 else 6 * s
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.axis("off")

        palette = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        lw = style["line_width"]

        # Spread inputs vertically so they all fit on-canvas.
        # Top input at y_top, bottom input at y_bot (evenly spaced).
        y_top = y_top_plot - 0.5
        y_bot = y_bot_plot + 0.8
        if n_in == 1:
            ys = [(y_top + y_bot) / 2]
        else:
            step = (y_top - y_bot) / (n_in - 1)
            ys = [y_top - i * step for i in range(n_in)]

        input_pos = {}
        for i, name in enumerate(input_names):
            y = ys[i]
            input_pos[name] = (1.0, y)
            ax.text(0.3, y, f"{name}={input_vals[name]}",
                    ha="center", va="center",
                    fontsize=fs + 1, fontweight="bold", fontfamily=ff,
                    color=palette[i % len(palette)])
            ax.plot([0.6, 1.0], [y, y], color="#333", linewidth=lw)

        # Gate positions (centered vertically in the plot area)
        gate_pos = {}
        gy_center = (y_top + y_bot) / 2
        # Compute gate x spacing so larger circuits have room
        gate_x0 = 3.2
        gate_step = 2.5
        for i, (gtype, in1, in2) in enumerate(gates):
            gx = gate_x0 + i * gate_step
            # Stagger gate y slightly for deeper layers to reduce wire cross
            gy = gy_center
            gate_pos[f"G{i}"] = (gx, gy)

            color = palette[(i + 3) % len(palette)]
            rect = FancyBboxPatch((gx - 0.6, gy - 0.5), 1.2, 1.0,
                                   boxstyle="round,pad=0.1", facecolor=color,
                                   edgecolor="#333", linewidth=lw + 1,
                                   zorder=5)
            ax.add_patch(rect)
            ax.text(gx, gy, gtype, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="white",
                    fontfamily=ff, zorder=6)

            # Draw input wires to gate with orthogonal (L-shaped) routing.
            # For each input: go from source to a bus x just left of the
            # gate, then drop/rise to the correct input pin y.
            input_list = [in1, in2] if gtype != "NOT" else [in1]
            for j, inp in enumerate(input_list):
                if inp in input_pos:
                    sx, sy = input_pos[inp]
                elif inp in gate_pos:
                    sx, sy = gate_pos[inp]
                    sx += 0.6
                else:
                    continue
                if gtype == "NOT":
                    ty = gy
                else:
                    ty = gy + 0.25 - j * 0.5
                # Orthogonal L-shape: horizontal from source to bus_x,
                # then vertical to ty, then horizontal to gate input pin.
                # Use per-gate bus_x offset (closer to the gate) and small
                # per-input wiggle so wires don't overlap exactly.
                bus_x = gx - 0.85 - 0.12 * j
                ax.plot([sx, bus_x], [sy, sy], color="#333",
                        linewidth=lw, zorder=2)
                ax.plot([bus_x, bus_x], [sy, ty], color="#333",
                        linewidth=lw, zorder=2)
                ax.plot([bus_x, gx - 0.6], [ty, ty], color="#333",
                        linewidth=lw, zorder=2)

        # Output wire
        last_gx, last_gy = gate_pos[f"G{n_gates-1}"]
        out_val = all_vals[f"G{n_gates-1}"]
        ax.plot([last_gx + 0.6, last_gx + 1.5], [last_gy, last_gy],
                color="#333", linewidth=lw)
        out_label = "Out=?" if hide_out else f"Out={out_val}"
        ax.text(last_gx + 1.8, last_gy, out_label, ha="center", va="center",
                fontsize=fs + 1, fontweight="bold", fontfamily=ff,
                color=palette[0])

        ax.set_xlim(-0.5, last_gx + 3)
        ax.set_ylim(y_bot_plot - 0.2, y_top_plot + 0.3)
        ax.set_title("Logic Gate Circuit", fontsize=fs + 3, fontweight="bold",
                      fontfamily=ff, pad=10)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
