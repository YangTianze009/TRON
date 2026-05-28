"""
Phase Diagram QA environment.

Draws P-T phase diagrams with solid/liquid/gas regions, triple point,
critical point, and phase boundary curves. Questions about phase
identification, transitions, and special points.
Targets: a multimodal benchmark science diagrams, MMMU chemistry/physics.
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

# ------------------------------------------------------------------ #
# Substance presets — realistic-ish phase diagram parameters
# ------------------------------------------------------------------ #

_SUBSTANCES = [
    {
        "name": "Water",
        "triple_T": 0.01, "triple_P": 0.006,
        "critical_T": 374, "critical_P": 218,
        "normal_bp": 100, "normal_mp": 0,
        "sublimation_possible": True,
        "solid_liquid_slope": "negative",
    },
    {
        "name": "Carbon Dioxide",
        "triple_T": -56.6, "triple_P": 5.11,
        "critical_T": 31.0, "critical_P": 72.8,
        "normal_bp": -78.5, "normal_mp": -56.6,
        "sublimation_possible": True,
        "solid_liquid_slope": "positive",
    },
    {
        "name": "Substance X",
        "triple_T": 20, "triple_P": 0.5,
        "critical_T": 250, "critical_P": 80,
        "normal_bp": 120, "normal_mp": 25,
        "sublimation_possible": True,
        "solid_liquid_slope": "positive",
    },
    {
        "name": "Substance Y",
        "triple_T": -40, "triple_P": 0.1,
        "critical_T": 180, "critical_P": 50,
        "normal_bp": 80, "normal_mp": -35,
        "sublimation_possible": True,
        "solid_liquid_slope": "positive",
    },
    {
        "name": "Substance Z",
        "triple_T": 50, "triple_P": 2.0,
        "critical_T": 320, "critical_P": 120,
        "normal_bp": 180, "normal_mp": 55,
        "sublimation_possible": True,
        "solid_liquid_slope": "negative",
    },
]

def _phase_at_point(T, P, sub):
    """Determine the phase at a given (T, P) for the substance."""
    tT, tP = sub["triple_T"], sub["triple_P"]
    cT, cP = sub["critical_T"], sub["critical_P"]

    # Below triple point pressure: solid or gas (sublimation line)
    if P < tP:
        if T < tT:
            return "solid"
        else:
            return "gas"

    # Above critical point: supercritical (treat as gas for simplicity)
    if T > cT and P > cP:
        return "supercritical fluid"

    # Between triple and critical: use approximate boundaries
    # Solid-liquid boundary: roughly vertical (or slightly tilted)
    sl_T_at_P = tT + (P - tP) * (0.05 if sub["solid_liquid_slope"] == "positive" else -0.05)

    # Liquid-gas boundary: curve from triple point to critical point
    # Approximate with log interpolation
    if P <= cP and T >= tT:
        frac = (T - tT) / max(cT - tT, 1)
        boundary_P = tP + (cP - tP) * (frac ** 0.7)
        if T < sl_T_at_P:
            return "solid"
        elif P > boundary_P:
            return "liquid"
        else:
            return "gas"

    if T < sl_T_at_P:
        return "solid"

    return "liquid"

class PhaseDiagramQA(StandaloneVisualEnv):
    ENV_NAME = "phase_diagram"

    QUESTION_TYPES = [
        "identify_phase", "triple_point_temp", "boiling_point_at_pressure",
        "sublimation_possible", "phase_transition_name",
    ]

    def _level_config(self, level: int) -> Dict:
        # Note: `phase_transition_name` is pure text-knowledge (independent of
        # image). Weight reduced so most questions still require reading the
        # diagram.
        if level <= 0:
            return {"qtypes": ["triple_point_temp", "sublimation_possible",
                               "phase_transition_name"],
                    "qweights": [4, 4, 2], "substances": _SUBSTANCES[:2]}
        if level <= 2:
            return {"qtypes": ["triple_point_temp", "sublimation_possible",
                               "phase_transition_name"],
                    "qweights": [5, 3, 2], "substances": _SUBSTANCES[:3]}
        if level <= 4:
            return {"qtypes": ["triple_point_temp", "identify_phase", "phase_transition_name"],
                    "qweights": [3, 4, 3], "substances": _SUBSTANCES[:4]}
        if level <= 6:
            return {"qtypes": ["identify_phase", "boiling_point_at_pressure", "triple_point_temp"],
                    "qweights": [4, 3, 3], "substances": _SUBSTANCES}
        if level <= 8:
            return {"qtypes": ["identify_phase", "boiling_point_at_pressure"],
                    "qweights": [5, 5], "substances": _SUBSTANCES}
        return {"qtypes": ["boiling_point_at_pressure", "identify_phase"],
                "qweights": [6, 4], "substances": _SUBSTANCES}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 6601)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        sub = sub_rng.choice(cfg["substances"])

        question, answer, marker = self._make_qa(rng, qtype, sub)
        if question is None:
            return None

        image = self._render(rng, sub, marker)
        return question, answer, image

    def _make_qa(
        self, rng: random.Random, qtype: str, sub: dict
    ) -> Tuple[Optional[str], Optional[str], Optional[Tuple[float, float]]]:
        tT, tP = sub["triple_T"], sub["triple_P"]
        cT, cP = sub["critical_T"], sub["critical_P"]
        marker = None

        if qtype == "identify_phase":
            # Pick a random point and ask what phase it is
            T_range = cT - tT
            P_range = cP - tP
            T = round(tT + rng.uniform(-T_range * 0.3, T_range * 1.2), 1)
            P = round(max(0.001, tP + rng.uniform(-tP * 0.5, P_range * 1.1)), 1)
            phase = _phase_at_point(T, P, sub)
            marker = (T, P)
            q = (f"The phase diagram for {sub['name']} is shown. "
                 f"A point is marked at T = {T} and P = {P} atm. "
                 f"What phase is the substance in at this point?")
            return q, phase, marker

        elif qtype == "triple_point_temp":
            q = (f"From the phase diagram of {sub['name']}, "
                 f"what is the temperature at the triple point?")
            return q, str(tT), None

        elif qtype == "boiling_point_at_pressure":
            # At 1 atm (or some pressure between triple and critical)
            P_query = round(rng.uniform(tP * 1.5, cP * 0.5), 1)
            # Approximate boiling point: interpolate along liquid-gas boundary
            frac = (P_query - tP) / max(cP - tP, 1e-9)
            frac = max(0, min(1, frac))
            bp = round(tT + (cT - tT) * (frac ** (1 / 0.7)), 1)
            q = (f"From the phase diagram, estimate the boiling point of "
                 f"{sub['name']} at a pressure of {P_query} atm. "
                 f"Round to 1 decimal place.")
            return q, str(bp), None

        elif qtype == "sublimation_possible":
            q = (f"Based on the phase diagram, is sublimation possible "
                 f"for {sub['name']}? Answer yes or no.")
            ans = "yes" if sub["sublimation_possible"] else "no"
            return q, ans, None

        elif qtype == "phase_transition_name":
            transitions = [
                ("solid", "liquid", "melting"),
                ("liquid", "solid", "freezing"),
                ("liquid", "gas", "vaporization"),
                ("gas", "liquid", "condensation"),
                ("solid", "gas", "sublimation"),
                ("gas", "solid", "deposition"),
            ]
            from_phase, to_phase, name = rng.choice(transitions)
            q = (f"What is the name of the phase transition from "
                 f"{from_phase} to {to_phase}?")
            return q, name, None

        return None, None, None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(
        self, rng: random.Random, sub: dict,
        marker: Optional[Tuple[float, float]] = None
    ) -> Image.Image:
        style = self._random_style()
        base_w, base_h = 8, 6
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(base_w * sc, base_h * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        tT, tP = sub["triple_T"], sub["triple_P"]
        cT, cP = sub["critical_T"], sub["critical_P"]

        # Temperature range
        T_min = tT - abs(tT) * 0.5 - 20
        T_max = cT + abs(cT) * 0.2 + 20
        P_min = 0
        P_max = cP * 1.3

        # Solid-Liquid boundary (nearly vertical)
        sl_T = np.linspace(tT, tT + (10 if sub["solid_liquid_slope"] == "positive"
                                      else -10), 50)
        sl_P = np.linspace(tP, P_max, 50)
        if sub["solid_liquid_slope"] == "negative":
            sl_T = tT - np.linspace(0, 10, 50)
        ax.plot(sl_T, sl_P, color=palette[0], linewidth=lw, label="Solid-Liquid")

        # Liquid-Gas boundary (triple to critical)
        t_vals = np.linspace(0, 1, 100)
        lg_T = tT + (cT - tT) * t_vals
        lg_P = tP + (cP - tP) * (t_vals ** 0.7)
        ax.plot(lg_T, lg_P, color=palette[1], linewidth=lw, label="Liquid-Gas")

        # Solid-Gas boundary (sublimation curve, below triple point)
        sg_T = np.linspace(T_min, tT, 50)
        sg_P = tP * np.exp(-2.0 * (tT - sg_T) / max(abs(tT - T_min), 1))
        ax.plot(sg_T, sg_P, color=palette[2], linewidth=lw, label="Solid-Gas")

        # Triple point
        ax.plot(tT, tP, "ko", markersize=10, zorder=5)
        ax.annotate(f"Triple Point\n({tT}, {tP} atm)",
                     xy=(tT, tP), xytext=(tT + (T_max - T_min) * 0.08,
                                          tP + (P_max - P_min) * 0.08),
                     fontsize=fs - 3, fontweight="bold", fontfamily=ff,
                     arrowprops=dict(arrowstyle="->", color="black"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                               edgecolor="black"))

        # Critical point
        ax.plot(cT, cP, "k^", markersize=12, zorder=5)
        ax.annotate(f"Critical Point\n({cT}, {cP} atm)",
                     xy=(cT, cP), xytext=(cT - (T_max - T_min) * 0.15,
                                          cP + (P_max - P_min) * 0.08),
                     fontsize=fs - 3, fontweight="bold", fontfamily=ff,
                     arrowprops=dict(arrowstyle="->", color="black"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                               edgecolor="black"))

        # Region labels
        # Solid region: left of solid-liquid and solid-gas boundaries
        solid_T = T_min + (tT - T_min) * 0.3
        solid_P = P_max * 0.6
        ax.text(solid_T, solid_P, "SOLID", fontsize=fs + 4, fontweight="bold",
                ha="center", va="center", color=palette[0], alpha=0.7,
                fontfamily=ff)

        # Liquid region: between solid-liquid and liquid-gas
        liq_T = (tT + cT) / 2
        liq_P = (tP + cP) / 2 + P_max * 0.15
        ax.text(liq_T, liq_P, "LIQUID", fontsize=fs + 4, fontweight="bold",
                ha="center", va="center", color=palette[1], alpha=0.7,
                fontfamily=ff)

        # Gas region: right of liquid-gas and below
        gas_T = cT * 0.7
        gas_P = tP * 0.3 + P_max * 0.05
        ax.text(gas_T, gas_P, "GAS", fontsize=fs + 4, fontweight="bold",
                ha="center", va="center", color=palette[2], alpha=0.7,
                fontfamily=ff)

        # Marker point if provided
        if marker is not None:
            mT, mP = marker
            ax.plot(mT, mP, "r*", markersize=18, zorder=10,
                    markeredgecolor="black", markeredgewidth=1)
            ax.annotate(f"({mT}, {mP})", xy=(mT, mP),
                         xytext=(mT + (T_max - T_min) * 0.05,
                                 mP - (P_max - P_min) * 0.05),
                         fontsize=fs - 3, color=palette[1], fontweight="bold",
                         fontfamily=ff)

        ax.set_xlabel("Temperature (°C)", fontsize=fs, fontfamily=ff)
        ax.set_ylabel("Pressure (atm)", fontsize=fs, fontfamily=ff)
        ax.set_title(f"Phase Diagram: {sub['name']}", fontsize=fs + 2,
                     fontweight="bold", fontfamily=ff)
        ax.set_xlim(T_min, T_max)
        ax.set_ylim(P_min, P_max)
        ax.legend(loc="upper left", fontsize=fs - 3)
        if style["show_grid"]:
            ax.grid(True, alpha=style["grid_alpha"],
                    linestyle=style["grid_style"])

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
