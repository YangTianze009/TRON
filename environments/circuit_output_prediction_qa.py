"""
Circuit Output Prediction QA environment.

Capabilities: X3 (physical/mechanical reasoning) + X1 (multi-step)
Target regression: reference mechanical, MMMU Engineering.

A simple electrical circuit diagram: resistors (and, at higher levels,
an ideal op-amp follower). Component values are labeled, and an output
node is marked with "V_out = ?". A 4-option MCQ asks for the numeric
V_out.

Difficulty schedule (0..9):
  Axis 1: n_components = 2 + level       -> 2..11 resistors
  Axis 2: topology  L<=2: voltage divider (series)
          L3..L5: series-parallel
          L6..L8: Wheatstone bridge
          L9: bridge + extra op-amp-follower branch

Output: (question_str, answer_letter, PIL_Image)
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

class CircuitOutputPredictionQA(StandaloneVisualEnv):
    ENV_NAME = "circuit_output_prediction"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        n_components = 2 + level
        if level <= 2:
            topology = "divider"
        elif level <= 5:
            topology = "series_parallel"
        elif level <= 8:
            topology = "bridge"
        else:
            topology = "bridge_with_buffer"
        return {
            "n_components": n_components,
            "topology": topology,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2281)

        for _ in range(10):
            try:
                result = self._try_generate(sub_rng, cfg, level)
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level):
        topology = cfg["topology"]
        v_in = rng.choice([5, 9, 10, 12, 15, 18, 20, 24])

        if topology == "divider":
            # Simple 2-resistor series voltage divider: Vout across R2
            r1 = rng.choice([100, 220, 330, 470, 680, 1000, 1500, 2200, 3300,
                              4700])
            r2 = rng.choice([100, 220, 330, 470, 680, 1000, 1500, 2200, 3300,
                              4700])
            v_out = v_in * r2 / (r1 + r2)
            image = self._draw_divider(rng, v_in, r1, r2)
            topology_desc = "simple voltage divider"

        elif topology == "series_parallel":
            # R1 in series with (R2 || R3); V_out across the parallel pair
            r1 = rng.choice([220, 330, 470, 680, 1000, 1500])
            r2 = rng.choice([220, 330, 470, 680, 1000, 1500, 2200])
            r3 = rng.choice([220, 330, 470, 680, 1000, 1500, 2200])
            r_par = (r2 * r3) / (r2 + r3)
            v_out = v_in * r_par / (r1 + r_par)
            image = self._draw_series_parallel(rng, v_in, r1, r2, r3)
            topology_desc = "R1 in series with R2 || R3"

        elif topology == "bridge":
            # Balanced-or-unbalanced Wheatstone bridge; V_out = V_node_B - V_node_D
            r1 = rng.choice([100, 220, 330, 470, 680, 1000])
            r2 = rng.choice([100, 220, 330, 470, 680, 1000])
            r3 = rng.choice([100, 220, 330, 470, 680, 1000])
            r4 = rng.choice([100, 220, 330, 470, 680, 1000])
            v_b = v_in * r3 / (r1 + r3)
            v_d = v_in * r4 / (r2 + r4)
            v_out = v_b - v_d
            image = self._draw_bridge(rng, v_in, r1, r2, r3, r4,
                                       with_buffer=False)
            topology_desc = "Wheatstone bridge"

        else:  # bridge_with_buffer
            # Same bridge, but output routed through an ideal op-amp
            # differential amplifier (unity-gain subtractor) that outputs
            # V_B - V_D onto V_out. Behavior matches the plain bridge V_out.
            r1 = rng.choice([100, 220, 330, 470, 680, 1000])
            r2 = rng.choice([100, 220, 330, 470, 680, 1000])
            r3 = rng.choice([100, 220, 330, 470, 680, 1000])
            r4 = rng.choice([100, 220, 330, 470, 680, 1000])
            v_b = v_in * r3 / (r1 + r3)
            v_d = v_in * r4 / (r2 + r4)
            v_out = v_b - v_d
            image = self._draw_bridge(rng, v_in, r1, r2, r3, r4,
                                       with_buffer=True)
            topology_desc = ("Wheatstone bridge followed by an ideal "
                             "unity-gain differential amplifier "
                             "(V_out = V_B - V_D)")

        correct_disp = round(float(v_out), 2)
        # Build 3 numeric distractors
        base = max(0.3, abs(correct_disp) * 0.3)
        offsets = [-base, base, -2 * base, 2 * base, -0.6 * base, 0.6 * base,
                   base * 1.4, -base * 1.4]
        rng.shuffle(offsets)
        distractors = []
        for off in offsets:
            cand = round(correct_disp + off, 2)
            if abs(cand - correct_disp) < 0.05:
                continue
            if any(abs(cand - d) < 0.05 for d in distractors):
                continue
            distractors.append(cand)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            return None
        options = [correct_disp] + distractors
        rng.shuffle(options)
        correct_idx = options.index(correct_disp)
        letter = chr(ord("A") + correct_idx)
        opts_str = " ".join(
            f"({chr(ord('A')+i)}) {v} V" for i, v in enumerate(options)
        )
        q_pool = [
            (f"The circuit shown is a {topology_desc}. Using the labeled "
             f"component values and V_in, what is the output voltage V_out "
             f"(in volts)? Options: {opts_str}. Answer with a single letter."),
            (f"For the {topology_desc} depicted, calculate V_out using the "
             f"resistor labels and supply voltage shown. Pick the closest "
             f"option. Options: {opts_str}. Answer with a single letter."),
            (f"Apply circuit-analysis rules to the {topology_desc} in the "
             f"figure and find V_out (in volts). Options: {opts_str}. "
             f"Answer with a single letter."),
            (f"From the {topology_desc} schematic, determine V_out using "
             f"the displayed component values. Options: {opts_str}. "
             f"Answer with a single letter."),
        ]
        question = rng.choice(q_pool)
        return question, letter, image

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _draw_zigzag(ax, x_start, x_end, y, amplitude=0.12, n_teeth=5, **kwargs):
        xs = np.linspace(x_start, x_end, n_teeth * 2 + 1)
        ys = [y]
        for i in range(1, len(xs)):
            if i % 2 == 1:
                ys.append(y + amplitude)
            else:
                ys.append(y - amplitude)
        ys[-1] = y
        ys[0] = y
        ax.plot(xs, ys, color=kwargs.get("color", "#2c3e50"),
                linewidth=kwargs.get("lw", 2))

    @staticmethod
    def _draw_vertical_zigzag(ax, x, y_start, y_end, amplitude=0.12,
                               n_teeth=5, **kwargs):
        ys = np.linspace(y_start, y_end, n_teeth * 2 + 1)
        xs = [x]
        for i in range(1, len(ys)):
            if i % 2 == 1:
                xs.append(x + amplitude)
            else:
                xs.append(x - amplitude)
        xs[-1] = x
        xs[0] = x
        ax.plot(xs, ys, color=kwargs.get("color", "#2c3e50"),
                linewidth=kwargs.get("lw", 2))

    @staticmethod
    def _draw_battery(ax, x, y_bot, y_top, voltage, **kwargs):
        mid = (y_bot + y_top) / 2
        gap = 0.1
        color = kwargs.get("color", "#2c3e50")
        lw = kwargs.get("lw", 2)
        ax.plot([x - 0.2, x + 0.2], [mid + gap, mid + gap], color=color,
                lw=lw + 0.5)
        ax.plot([x - 0.1, x + 0.1], [mid - gap, mid - gap], color=color,
                lw=lw + 0.5)
        ax.text(x + 0.25, mid + gap, "+", fontsize=10, color=color,
                va="center")
        ax.text(x + 0.25, mid - gap, "-", fontsize=10, color=color,
                va="center")
        ax.plot([x, x], [y_bot, mid - gap], color=color, lw=lw)
        ax.plot([x, x], [mid + gap, y_top], color=color, lw=lw)
        ax.text(x - 0.35, mid, f"V_in={voltage}V", fontsize=10,
                fontweight="bold", color="#c0392b", ha="right", va="center")

    def _draw_divider(self, rng, v_in, r1, r2):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        color = style["geo_line_color"]
        lw = style["line_width"]

        # Layout: battery on left, vertical stack of R1 (top) and R2 (bottom)
        x_bat = 0.6
        y_bot = 0.5
        y_top = 4.0
        self._draw_battery(ax, x_bat, y_bot, y_top, v_in, color=color, lw=lw)
        # Top wire
        ax.plot([x_bat, 4.2], [y_top, y_top], color=color, lw=lw)
        # Vertical column at x=4.2
        x_col = 4.2
        # R1 top half: y 3.5..2.4
        self._draw_vertical_zigzag(ax, x_col, 3.6, 2.6, color=color, lw=lw)
        ax.plot([x_col, x_col], [y_top, 3.6], color=color, lw=lw)
        ax.text(x_col + 0.35, 3.1, f"R1={r1}Ω", fontsize=11,
                fontweight="bold", color="#8e44ad")
        # mid node (V_out)
        y_mid = 2.4
        ax.plot([x_col, x_col], [2.6, y_mid], color=color, lw=lw)
        ax.plot(x_col, y_mid, "o", color="black", markersize=6, zorder=5)
        ax.text(x_col + 0.25, y_mid + 0.05,
                "V_out = ?", fontsize=12, fontweight="bold", color="#e74c3c")

        # R2 bottom half
        self._draw_vertical_zigzag(ax, x_col, 2.2, 1.2, color=color, lw=lw)
        ax.plot([x_col, x_col], [y_mid, 2.2], color=color, lw=lw)
        ax.plot([x_col, x_col], [1.2, y_bot], color=color, lw=lw)
        ax.text(x_col + 0.35, 1.7, f"R2={r2}Ω", fontsize=11,
                fontweight="bold", color="#2980b9")

        # Bottom wire back to battery
        ax.plot([x_bat, x_col], [y_bot, y_bot], color=color, lw=lw)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4.8)
        ax.set_title("Voltage Divider",
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_series_parallel(self, rng, v_in, r1, r2, r3):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(9 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        color = style["geo_line_color"]
        lw = style["line_width"]

        y_top = 3.5
        y_bot = 0.5
        x_bat = 0.6
        self._draw_battery(ax, x_bat, y_bot, y_top, v_in, color=color, lw=lw)
        # Top wire to R1
        ax.plot([x_bat, 1.5], [y_top, y_top], color=color, lw=lw)
        self._draw_zigzag(ax, 1.5, 3.0, y_top, color=color, lw=lw)
        ax.text(2.25, y_top + 0.35, f"R1={r1}Ω", fontsize=11,
                ha="center", fontweight="bold", color="#8e44ad")

        # After R1 split into parallel
        ax.plot([3.0, 3.8], [y_top, y_top], color=color, lw=lw)
        # V_out node at parallel entry
        ax.plot(3.8, y_top, "o", color="black", markersize=6, zorder=5)
        ax.text(3.85, y_top + 0.25, "V_out = ?", fontsize=12,
                fontweight="bold", color="#e74c3c")

        y_upper = y_top - 0.4
        y_lower = y_top - 1.8
        # Vertical rail from V_out (at y_top) down through both branch tees.
        # Previously started at y_upper, leaving V_out visually disconnected.
        ax.plot([3.8, 3.8], [y_top, y_lower], color=color, lw=lw)

        # R2 branch (upper)
        ax.plot([3.8, 4.3], [y_upper, y_upper], color=color, lw=lw)
        self._draw_zigzag(ax, 4.3, 5.8, y_upper, color=color, lw=lw)
        ax.text(5.05, y_upper + 0.25, f"R2={r2}Ω", fontsize=11,
                ha="center", fontweight="bold", color="#2980b9")
        ax.plot([5.8, 6.3], [y_upper, y_upper], color=color, lw=lw)

        # R3 branch (lower)
        ax.plot([3.8, 4.3], [y_lower, y_lower], color=color, lw=lw)
        self._draw_zigzag(ax, 4.3, 5.8, y_lower, color=color, lw=lw)
        ax.text(5.05, y_lower - 0.35, f"R3={r3}Ω", fontsize=11,
                ha="center", fontweight="bold", color="#16a085")
        ax.plot([5.8, 6.3], [y_lower, y_lower], color=color, lw=lw)

        # Rejoin
        ax.plot([6.3, 6.3], [y_upper, y_lower], color=color, lw=lw)
        # Down to bottom
        ax.plot([6.3, 6.3], [y_lower, y_bot], color=color, lw=lw)
        ax.plot([x_bat, 6.3], [y_bot, y_bot], color=color, lw=lw)

        ax.set_xlim(-0.2, 7.2)
        ax.set_ylim(-0.2, 4.3)
        ax.set_title("Series-Parallel Circuit",
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_bridge(self, rng, v_in, r1, r2, r3, r4, with_buffer=False):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")
        color = style["geo_line_color"]
        lw = style["line_width"]

        # Diamond layout: A top, B right, C bottom, D left
        A = (3.5, 5.0)
        B = (6.0, 3.0)
        C = (3.5, 1.0)
        D = (1.0, 3.0)

        # R1: A -> B (top-right)
        ax.plot([A[0], B[0]], [A[1], B[1]], color=color, lw=lw)
        mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        ax.text(mx + 0.3, my + 0.3, f"R1={r1}Ω", fontsize=10,
                fontweight="bold", color="#8e44ad")

        # R2: A -> D (top-left)
        ax.plot([A[0], D[0]], [A[1], D[1]], color=color, lw=lw)
        mx, my = (A[0] + D[0]) / 2, (A[1] + D[1]) / 2
        ax.text(mx - 0.8, my + 0.3, f"R2={r2}Ω", fontsize=10,
                fontweight="bold", color="#2980b9")

        # R3: B -> C (bottom-right)
        ax.plot([B[0], C[0]], [B[1], C[1]], color=color, lw=lw)
        mx, my = (B[0] + C[0]) / 2, (B[1] + C[1]) / 2
        ax.text(mx + 0.3, my - 0.3, f"R3={r3}Ω", fontsize=10,
                fontweight="bold", color="#16a085")

        # R4: D -> C (bottom-left)
        ax.plot([D[0], C[0]], [D[1], C[1]], color=color, lw=lw)
        mx, my = (D[0] + C[0]) / 2, (D[1] + C[1]) / 2
        ax.text(mx - 0.8, my - 0.3, f"R4={r4}Ω", fontsize=10,
                fontweight="bold", color="#e67e22")

        # Battery: between A and C via an offset wire at x=-0.2.
        # _draw_battery itself draws vertical wires from y_bot to (mid-gap)
        # and from (mid+gap) to y_top at x=-0.2, so we only supply the short
        # stubs from the bridge nodes to the battery rail endpoints
        # (avoids drawing a continuous line that shorts the battery plates).
        ax.plot([A[0], -0.2], [A[1], A[1]], color=color, lw=lw)
        ax.plot([-0.2, -0.2], [A[1], 4.8], color=color, lw=lw)
        # Battery symbol at (-0.2, 3)
        self._draw_battery(ax, -0.2, 1.2, 4.8, v_in, color=color, lw=lw)
        ax.plot([-0.2, -0.2], [1.2, C[1]], color=color, lw=lw)
        ax.plot([-0.2, C[0]], [C[1], C[1]], color=color, lw=lw)

        # Node labels
        for pt, lbl in [(A, "A"), (B, "B"), (C, "C"), (D, "D")]:
            ax.plot(pt[0], pt[1], "ko", markersize=6, zorder=5)
            ax.text(pt[0] - 0.15, pt[1] + 0.35, lbl, fontsize=12,
                    fontweight="bold")

        if with_buffer:
            # Draw an op-amp (differential amp) between (B, D) and V_out.
            # Op-amp triangle at (7.0..8.4, 2.4..3.6); "+" top input, "-" bottom.
            tri_xs = [7.0, 8.4, 7.0, 7.0]
            tri_ys = [3.6, 3.0, 2.4, 3.6]
            ax.plot(tri_xs, tri_ys, color=color, lw=lw)
            ax.text(7.3, 3.2, "+", fontsize=10, fontweight="bold",
                    ha="center", va="center")
            ax.text(7.3, 2.8, "-", fontsize=10, fontweight="bold",
                    ha="center", va="center")
            # Wire B -> op-amp +input (top). Short horizontal, then small
            # step down so the connection lands on the triangle edge.
            ax.plot([B[0], 7.0], [B[1], 3.2], color=color, lw=lw)
            # Wire D -> op-amp -input (bottom). Route UP from D, across
            # above the bridge, then down just left of the op-amp triangle,
            # and in to the "-" input at (7.0, 2.8). Avoids crossing the
            # triangle edges and the battery-to-C wire at y=1.0.
            ax.plot([D[0], D[0]], [D[1], 5.5], color=color, lw=lw)
            ax.plot([D[0], 6.8], [5.5, 5.5], color=color, lw=lw)
            ax.plot([6.8, 6.8], [5.5, 2.8], color=color, lw=lw)
            ax.plot([6.8, 7.0], [2.8, 2.8], color=color, lw=lw)
            # Output wire
            ax.plot([8.4, 9.2], [3.0, 3.0], color=color, lw=lw)
            ax.plot(9.2, 3.0, "o", color="black", markersize=6, zorder=5)
            ax.text(9.3, 3.15, "V_out = ?", fontsize=12,
                    fontweight="bold", color="#e74c3c")
            ax.text(7.7, 3.85, "Diff\nAmp", fontsize=9, ha="center",
                    color="#555555")
            ax.set_xlim(-1.5, 10.2)
        else:
            # V_out between B and D
            ax.text(3.5, 3.1, "V_out\n= V_B - V_D", fontsize=11,
                    fontweight="bold", color="#e74c3c", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffeaa7",
                              ec="#e74c3c", alpha=0.9))
            # Mark V_B and V_D labels
            ax.text(B[0] + 0.25, B[1] - 0.2, "V_B", fontsize=10,
                    fontweight="bold", color="#c0392b")
            ax.text(D[0] - 0.5, D[1] - 0.2, "V_D", fontsize=10,
                    fontweight="bold", color="#c0392b")
            ax.set_xlim(-1.5, 7.5)

        ax.set_ylim(0, 6)
        title = "Wheatstone Bridge" + (
            " + Differential Amplifier" if with_buffer else "")
        ax.set_title(title, fontsize=style["font_size_base"] + 3,
                     fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
