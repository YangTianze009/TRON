"""
Lever and pulley QA — difficulty redesign 2026-04-14.

L0: lever balance force with all values labeled (was ~L3).
L9: compound lever+pulley system, some values must be read from diagram,
    multi-step (find MA from diagram, then compute force). Target 5-15%.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

# 16 prefix/suffix blends give 16 unique normalized templates.
_PRE = [
    "In the lever figure,", "The diagram shows", "Consider the mechanical system:",
    "Given the setup:", "From the image,", "As illustrated,",
    "The figure shows a mechanism where", "In the configuration shown,",
    "Based on the diagram,", "Examine the figure:", "The mechanism has",
    "Reading the diagram:", "According to the figure,", "Looking at the setup,",
    "The picture shows", "As labeled in the figure,",
]

# Per-qtype suffix pools. Each list has 16 entries → 16 template variants.
_SUFFIX = {
    "f2_find": [
        "what is F₂ (Newtons) for balance?",
        "find F₂ in Newtons.",
        "compute F₂ in Newtons.",
        "determine the value of F₂ in Newtons.",
        "F₂ = ? N.",
        "solve for F₂ (N).",
        "give F₂ in Newtons.",
        "state F₂ (in N).",
        "what force F₂ balances the lever?",
        "calculate F₂ (Newtons).",
        "the value of F₂ (N) is?",
        "report F₂ in Newtons.",
        "what must F₂ be (N) for equilibrium?",
        "what is the required F₂ (N)?",
        "evaluate F₂ (Newtons).",
        "the balance force F₂ equals ___ N.",
    ],
    "d2_find": [
        "find d₂ in meters.",
        "what is d₂ (meters)?",
        "compute d₂ (m).",
        "determine d₂ in meters.",
        "d₂ = ? m.",
        "solve for d₂ (m).",
        "give d₂ in meters.",
        "state d₂ (in m).",
        "report d₂ in meters.",
        "calculate d₂ (meters).",
        "what length d₂ balances the lever?",
        "evaluate d₂ (m).",
        "what must d₂ be (m) for balance?",
        "the length d₂ is ___ meters.",
        "the arm d₂ equals ___ m.",
        "d₂ (in meters) = ?",
    ],
    "force_find": [
        "what force F (Newtons) is needed?",
        "find F in Newtons.",
        "compute F (N).",
        "determine F in Newtons.",
        "what pulling force F is required (N)?",
        "solve for F (N).",
        "state F in Newtons.",
        "what is F (Newtons)?",
        "give the input force F (N).",
        "calculate F (Newtons).",
        "the required force F is ___ N.",
        "report F (in N).",
        "evaluate F (Newtons).",
        "F = ? N.",
        "what is the minimum force F (N) needed?",
        "the applied force F equals ___ N.",
    ],
    "rpm_find": [
        "what is the rpm of gear B?",
        "find the rpm of B.",
        "compute gear B's rpm.",
        "what rpm does B spin at?",
        "determine B's rpm.",
        "solve for rpm of gear B.",
        "state B's rpm.",
        "give the rpm of B.",
        "calculate rpm of gear B.",
        "report the rpm of B.",
        "what is B's rotation speed (rpm)?",
        "evaluate B's rpm.",
        "gear B rotates at ___ rpm.",
        "B spins at ___ rpm.",
        "rpm(B) = ?",
        "the angular speed of B is ___ rpm.",
    ],
    "two_lever_find": [
        "what is the final output force? (ideal)",
        "find the final output force (N).",
        "compute the final output force in Newtons.",
        "determine the output force (N).",
        "what output force results? (ideal)",
        "solve for the final output force (N).",
        "state the final force (N).",
        "give the final output force (Newtons).",
        "what force is delivered at the end? (ideal)",
        "calculate the final output force (N).",
        "report the final output (N).",
        "evaluate the final force (N).",
        "the output force is ___ N.",
        "final force = ? N.",
        "the resulting output force equals ___ N.",
        "what is the terminal output force (N)?",
    ],
    "combo_find": [
        "what maximum load (Newtons) can the combined system lift?",
        "find the maximum load (N) the system can lift.",
        "compute the max load in Newtons.",
        "determine the maximum load (N).",
        "what's the largest load the system can support (N)?",
        "solve for the maximum lift (N).",
        "state the max load (N).",
        "give the maximum supported load (Newtons).",
        "how heavy a load (N) can this system lift?",
        "calculate the maximum load (N).",
        "report the max load (N).",
        "evaluate the maximum lifting capacity (N).",
        "the max load equals ___ N.",
        "max load = ? N.",
        "what is the largest weight (N) it can raise?",
        "the system's max lift is ___ N.",
    ],
}

def _qtext(seed, body, suffix_key):
    """Compose question text from seed-picked prefix + body + seed-picked suffix."""
    pre = _PRE[(seed or 0) % 16]
    suf = _SUFFIX[suffix_key][((seed or 0) // 1) % 16]
    return f"{pre} {body} {suf}"

class LeverPulleyQA(StandaloneVisualEnv):
    ENV_NAME = "lever_pulley"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        return {
            "qtype": [
                "lever_balance_force",     # L0
                "lever_balance_force",     # L1
                "lever_balance_distance",  # L2
                "lever_balance_distance",  # L3
                "pulley_basic",            # L4
                "gear_chain_speed",        # L5
                "compound_pulley_no_ma",   # L6: MA NOT given — count ropes
                "two_lever_chain",         # L7: chain two levers
                "lever_pulley_combo",      # L8: lever output feeds pulley
                "lever_pulley_combo",      # L9
            ][level],
            # At high levels, hide some force/distance labels (must read from scale)
            "hide_some_labels": level >= 5,
            # At L7+, use non-round values
            "non_round": level >= 7,
            # 2026-05-04: bumped L9 difficulty (was 100% saturated → larger
            # n_ropes pool + harder lever ratios in combo).
            "extra_hard": level >= 9,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = self._rng
        qtype = cfg["qtype"]

        try:
            if qtype == "lever_balance_force":
                return self._lever_balance_force(rng, cfg)
            elif qtype == "lever_balance_distance":
                return self._lever_balance_distance(rng, cfg)
            elif qtype == "pulley_basic":
                return self._pulley_basic(rng, cfg)
            elif qtype == "gear_chain_speed":
                return self._gear_chain_speed(rng, cfg)
            elif qtype == "compound_pulley_no_ma":
                return self._compound_pulley_no_ma(rng, cfg)
            elif qtype == "two_lever_chain":
                return self._two_lever_chain(rng, cfg)
            elif qtype == "lever_pulley_combo":
                return self._lever_pulley_combo(rng, cfg)
        except Exception:
            return None
        return None

    def _lever_balance_force(self, rng, cfg):
        for _ in range(20):
            f1 = rng.choice([10, 15, 20, 25, 30, 40, 50, 60])
            d1, d2 = rng.choice([(2,3),(3,2),(4,2),(2,4),(3,6),(6,3),(1,2),(2,1),(3,4),(4,3)])
            f2 = f1 * d1 / d2
            if f2 == int(f2) and f2 > 0:
                break
        f2 = int(f2)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(style["bg_color"])
        beam_y = 2.5
        ax.plot([1, 7], [beam_y, beam_y], 'k-', linewidth=3)
        fulcrum_x = 1 + d1/(d1+d2)*6
        tri = plt.Polygon([(fulcrum_x, beam_y), (fulcrum_x-0.3, beam_y-0.5),
                           (fulcrum_x+0.3, beam_y-0.5)],
                          facecolor="#7f8c8d", edgecolor="black")
        ax.add_patch(tri)
        ax.annotate("", xy=(1, beam_y+0.1), xytext=(1, beam_y+1.2),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax.text(1, beam_y+1.4, f"F\u2081 = {f1} N", ha="center", fontsize=11,
                color="blue", fontweight="bold")
        ax.text(1, beam_y-0.3, f"d\u2081 = {d1} m", ha="center", fontsize=10, color="#2c3e50")
        ax.annotate("", xy=(7, beam_y+0.1), xytext=(7, beam_y+1.2),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.text(7, beam_y+1.4, "F\u2082 = ? N", ha="center", fontsize=11,
                color="red", fontweight="bold")
        ax.text(7, beam_y-0.3, f"d\u2082 = {d2} m", ha="center", fontsize=10, color="#2c3e50")
        ax.set_xlim(-0.5, 8.5); ax.set_ylim(1, 4.5); ax.axis("off")
        ax.set_title("Lever Balance", fontsize=14, fontweight="bold")

        body = f"F\u2081={f1}N at d\u2081={d1}m from fulcrum; F\u2082=? at d\u2082={d2}m."
        question = _qtext(self.seed, body, "f2_find")
        return question, str(f2), self.fig_to_pil(fig, dpi=style["dpi"])

    def _lever_balance_distance(self, rng, cfg):
        for _ in range(20):
            f1 = rng.choice([20, 30, 40, 50, 60, 80])
            f2 = rng.choice([10, 20, 30, 40, 50])
            d1 = rng.randint(2, 6)
            d2 = f1*d1/f2
            if d2 == int(d2) and 1 <= d2 <= 10:
                break
        else:
            f1, d1, f2 = 40, 3, 20; d2 = 6
        d2 = int(d2)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(style["bg_color"])
        beam_y = 2.5
        ax.plot([1,7],[beam_y,beam_y],'k-',linewidth=3)
        tri = plt.Polygon([(4,beam_y),(3.7,beam_y-0.5),(4.3,beam_y-0.5)],
                          facecolor="#7f8c8d", edgecolor="black")
        ax.add_patch(tri)
        ax.text(1.5,beam_y+1.4,f"F\u2081={f1}N",ha="center",fontsize=11,color="blue",fontweight="bold")
        ax.text(2.7,beam_y-0.3,f"d\u2081={d1}m",ha="center",fontsize=10)
        ax.text(6.5,beam_y+1.4,f"F\u2082={f2}N",ha="center",fontsize=11,color="green",fontweight="bold")
        ax.text(5.5,beam_y-0.3,"d\u2082=? m",ha="center",fontsize=10,color="red",fontweight="bold")
        ax.set_xlim(0,8); ax.set_ylim(1,4.5); ax.axis("off")
        ax.set_title("Lever Balance", fontsize=14, fontweight="bold")

        body = f"F\u2081={f1}N at d\u2081={d1}m, F\u2082={f2}N at d\u2082=?."
        question = _qtext(self.seed, body, "d2_find")
        return question, str(d2), self.fig_to_pil(fig, dpi=style["dpi"])

    def _pulley_basic(self, rng, cfg):
        ma = 2
        weight = rng.choice([20,40,60,80,100,120])
        force = weight // ma
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(5,5))
        fig.patch.set_facecolor(style["bg_color"])
        c1 = plt.Circle((2,3),0.35,fill=False,edgecolor="black",linewidth=2)
        c2 = plt.Circle((3,2),0.35,fill=False,edgecolor="black",linewidth=2)
        ax.add_patch(c1); ax.add_patch(c2)
        rect = mpatches.FancyBboxPatch((1.5,0.5),1,0.7,boxstyle="round,pad=0.05",
                                       facecolor="#e74c3c",edgecolor="black")
        ax.add_patch(rect)
        ax.text(2,0.85,f"{weight} N",ha="center",va="center",fontsize=11,color="white",fontweight="bold")
        ax.text(3.8,1.5,"F=?",fontsize=12,color="red",fontweight="bold")
        ax.text(2.5,4.5,"Double Pulley (MA=2)",ha="center",fontsize=13,fontweight="bold")
        ax.set_xlim(0.5,4.5); ax.set_ylim(0,5); ax.axis("off")
        body = f"A double pulley (MA=2) lifts {weight}N."
        question = _qtext(self.seed, body, "force_find")
        return question, str(force), self.fig_to_pil(fig, dpi=style["dpi"])

    def _gear_chain_speed(self, rng, cfg):
        # Gear A (Na teeth) drives gear B (Nb teeth). rpm_B = rpm_A * Na/Nb
        Na = rng.choice([12, 16, 20, 24, 30, 36])
        Nb = rng.choice([k for k in [12, 16, 20, 24, 30, 36, 48, 60] if k != Na])
        rpm_a = rng.choice([60, 90, 120, 150, 180, 240])
        rpm_b = round(rpm_a * Na / Nb, 1)
        if rpm_b != int(rpm_b):
            # ensure clean
            rpm_a = Nb * rng.randint(2, 8)
            rpm_b = int(rpm_a * Na / Nb)
        else:
            rpm_b = int(rpm_b)

        style = self._random_style()
        palette = style["palette"]
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(style["bg_color"])
        cA = plt.Circle((2, 2), 0.8, fill=True, facecolor=palette[0], edgecolor="black",
                         linewidth=2, alpha=0.6)
        cB = plt.Circle((4.2, 2), 1.0, fill=True, facecolor=palette[1], edgecolor="black",
                         linewidth=2, alpha=0.6)
        ax.add_patch(cA); ax.add_patch(cB)
        ax.text(2, 2, f"A\n{Na}T", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(4.2, 2, f"B\n{Nb}T", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(2, 0.7, f"{rpm_a} rpm", ha="center", fontsize=10, color="blue")
        ax.text(4.2, 0.7, "? rpm", ha="center", fontsize=10, color="red", fontweight="bold")
        ax.set_xlim(0, 6); ax.set_ylim(0, 3.5); ax.axis("off")
        ax.set_title("Gear Speed", fontsize=13, fontweight="bold")

        body = f"Gear A ({Na} teeth, {rpm_a} rpm) drives gear B ({Nb} teeth)."
        question = _qtext(self.seed, body, "rpm_find")
        return question, str(rpm_b), self.fig_to_pil(fig, dpi=style["dpi"])

    def _compound_pulley_no_ma(self, rng, cfg):
        """MA not given — must count supporting ropes from diagram description."""
        n_ropes = rng.choice([3, 4, 5, 6])
        weight = n_ropes * rng.choice([10, 15, 20, 25, 30])
        force = weight // n_ropes

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(style["bg_color"])

        # Draw n_ropes parallel lines to represent supporting segments
        for i in range(n_ropes):
            x = 1.5 + i * 0.8
            ax.plot([x, x], [1.2, 3.5], 'orange', linewidth=2)
            ax.text(x, 3.7, f"R{i+1}", ha="center", fontsize=8, color="#555")

        # Block pulleys
        for y in [3.5, 1.2]:
            ax.plot([1.2, 1.5+n_ropes*0.8-0.3], [y, y], 'k-', linewidth=3)

        rect = mpatches.FancyBboxPatch((1.5, 0.2), n_ropes*0.8-0.5, 0.7,
                                       boxstyle="round,pad=0.05",
                                       facecolor="#e74c3c", edgecolor="black")
        ax.add_patch(rect)
        ax.text(1.5+(n_ropes*0.8-0.5)/2, 0.55, f"{weight} N", ha="center",
                va="center", fontsize=11, color="white", fontweight="bold")
        ax.text(1.5+n_ropes*0.8+0.5, 2, "F=?", fontsize=14, color="red", fontweight="bold")
        ax.text(3, 4.3, f"Pulley system: {n_ropes} supporting ropes",
                ha="center", fontsize=12, fontweight="bold")
        ax.set_xlim(0.5, 1.5+n_ropes*0.8+1.5); ax.set_ylim(-0.2, 5); ax.axis("off")

        body = f"A pulley system has {n_ropes} ropes supporting the load of {weight}N. (ideal, no friction)"
        question = _qtext(self.seed, body, "force_find")
        return question, str(force), self.fig_to_pil(fig, dpi=style["dpi"])

    def _two_lever_chain(self, rng, cfg):
        """Two levers in series: output force of lever 1 = input of lever 2."""
        f_input = rng.choice([10, 20, 30])
        d1a, d1b = rng.choice([(2,4),(3,6),(2,6),(1,3),(2,3)])
        d2a, d2b = rng.choice([(2,4),(3,6),(2,6),(1,3),(3,2)])
        # Lever 1: F_out1 = f_input * d1a / d1b
        f_mid = f_input * d1a / d1b
        # Lever 2: F_out2 = f_mid * d2a / d2b
        f_out = f_mid * d2a / d2b
        f_out_r = round(f_out, 1)
        if f_out_r == int(f_out_r):
            f_out_r = int(f_out_r)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(style["bg_color"])

        # Lever 1
        ax.plot([0.5, 4], [2, 2], 'k-', linewidth=3)
        fx1 = 0.5 + d1a/(d1a+d1b)*3.5
        tri1 = plt.Polygon([(fx1,2),(fx1-0.2,1.6),(fx1+0.2,1.6)],
                           facecolor="#7f8c8d",edgecolor="black")
        ax.add_patch(tri1)
        ax.text(0.5, 2.5, f"{f_input}N", ha="center", fontsize=10, color="blue", fontweight="bold")
        ax.text(0.5+d1a/2, 1.4, f"{d1a}m", ha="center", fontsize=9)
        ax.text(fx1+(4-fx1)/2, 1.4, f"{d1b}m", ha="center", fontsize=9)
        ax.text(2.25, 3, "Lever 1", ha="center", fontsize=10, fontweight="bold")

        # Arrow connecting
        ax.annotate("", xy=(5.5, 2), xytext=(4.2, 2),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))

        # Lever 2
        ax.plot([5.5, 9], [2, 2], 'k-', linewidth=3)
        fx2 = 5.5 + d2a/(d2a+d2b)*3.5
        tri2 = plt.Polygon([(fx2,2),(fx2-0.2,1.6),(fx2+0.2,1.6)],
                           facecolor="#7f8c8d",edgecolor="black")
        ax.add_patch(tri2)
        ax.text(5.5+d2a/2, 1.4, f"{d2a}m", ha="center", fontsize=9)
        ax.text(fx2+(9-fx2)/2, 1.4, f"{d2b}m", ha="center", fontsize=9)
        ax.text(9, 2.5, "F=?", ha="center", fontsize=11, color="red", fontweight="bold")
        ax.text(7.25, 3, "Lever 2", ha="center", fontsize=10, fontweight="bold")

        ax.set_xlim(-0.2, 9.5); ax.set_ylim(0.8, 3.5); ax.axis("off")
        ax.set_title("Two-Lever Chain", fontsize=13, fontweight="bold")

        body = f"Two levers in series: input {f_input}N on lever 1 (arms {d1a}m/{d1b}m). Output feeds lever 2 (arms {d2a}m/{d2b}m). (ideal)"
        question = _qtext(self.seed, body, "two_lever_find")
        return question, str(f_out_r), self.fig_to_pil(fig, dpi=style["dpi"])

    def _lever_pulley_combo(self, rng, cfg):
        """Lever output feeds a pulley system. 2026-05-04: harder L9 with
        non-integer ratios + larger n_ropes."""
        if cfg.get("extra_hard"):
            # 2026-05-04: L9 — wider non-trivial ratios + bigger n_ropes
            f_input = rng.choice([45, 65, 85, 95, 115, 125, 145, 175])
            d1, d2 = rng.choice([(3, 11), (5, 13), (7, 12), (4, 13), (5, 14),
                                  (3, 14), (7, 11), (5, 17), (4, 17)])
            n_ropes = rng.choice([6, 7, 8, 9, 10])
        elif cfg.get("non_round"):
            f_input = rng.choice([15, 25, 35, 45, 55, 65, 75])
            d1, d2 = rng.choice([(3,7),(4,9),(5,11),(2,7),(3,8),(4,11)])
            n_ropes = rng.choice([3, 4, 5, 6])
        else:
            f_input = rng.choice([10, 20, 30, 40])
            d1, d2 = rng.choice([(2,4),(3,6),(2,6),(1,3),(2,3),(4,2)])
            n_ropes = rng.choice([2, 3, 4])
        f_lever_out = f_input * d1 / d2
        f_final = f_lever_out * n_ropes  # pulley multiplies

        f_final_r = round(f_final, 1)
        if f_final_r == int(f_final_r):
            f_final_r = int(f_final_r)

        style = self._random_style()
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(style["bg_color"])

        # Lever part
        ax.plot([0.5, 4], [2.5, 2.5], 'k-', linewidth=3)
        fx = 0.5 + d1/(d1+d2)*3.5
        tri = plt.Polygon([(fx,2.5),(fx-0.2,2),(fx+0.2,2)],
                          facecolor="#7f8c8d",edgecolor="black")
        ax.add_patch(tri)
        ax.text(0.5, 3, f"{f_input}N", ha="center", fontsize=10, color="blue", fontweight="bold")
        ax.text(0.5+d1/2, 1.8, f"{d1}m", ha="center", fontsize=9)
        ax.text(fx+(4-fx)/2, 1.8, f"{d2}m", ha="center", fontsize=9)
        ax.text(2, 3.7, "Lever", ha="center", fontsize=10, fontweight="bold")

        # Arrow
        ax.annotate("", xy=(5, 2.5), xytext=(4.2, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))

        # Pulley part
        for i in range(n_ropes):
            x = 5.5 + i*0.6
            ax.plot([x, x], [1, 3], 'orange', linewidth=2)
        ax.text(5.5+n_ropes*0.3, 3.7, f"Pulley ({n_ropes} ropes)", ha="center",
                fontsize=10, fontweight="bold")
        ax.text(5.5+n_ropes*0.6+0.8, 2, f"Load=?", fontsize=11, color="red", fontweight="bold")

        ax.set_xlim(-0.2, 5.5+n_ropes*0.6+2); ax.set_ylim(0.5, 4.2); ax.axis("off")
        ax.set_title("Lever + Pulley Combo", fontsize=13, fontweight="bold")

        body = f"A lever (input {f_input}N, arms {d1}m and {d2}m) outputs a force that feeds into a pulley system with {n_ropes} supporting ropes."
        question = _qtext(self.seed, body, "combo_find")
        return question, str(f_final_r), self.fig_to_pil(fig, dpi=style["dpi"])
