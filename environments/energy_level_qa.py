"""
Energy Level Diagram QA environment.

Draws horizontal energy levels with transition arrows between them.
Questions about energy differences, transition types, counting
possible transitions. Covers electron shells, molecular orbitals,
and reaction energy profiles.
Targets: a multimodal benchmark science diagrams, MMMU physics/chemistry.
"""

import math
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Presets
# ------------------------------------------------------------------ #

_LEVEL_LABELS_ATOM = ["n=1", "n=2", "n=3", "n=4", "n=5", "n=6"]
_LEVEL_LABELS_ORBITAL = ["1s", "2s", "2p", "3s", "3p", "3d", "4s"]
_LEVEL_LABELS_GENERIC = ["E₁", "E₂", "E₃", "E₄", "E₅", "E₆"]

_ENERGY_UNIT_SETS = [
    ("eV", [(-13.6, -3.4, -1.51, -0.85, -0.54, -0.38), "Hydrogen-like"]),
    ("kJ/mol", [(-500, -300, -180, -100, -60, -30), "Generic Atom"]),
    ("eV", [(-10.0, -6.5, -4.2, -2.8, -1.5, -0.5), "Multi-electron Atom"]),
]

class EnergyLevelQA(StandaloneVisualEnv):
    ENV_NAME = "energy_level"

    QUESTION_TYPES = [
        "energy_difference", "highest_energy_transition",
        "absorption_vs_emission", "ground_state_energy",
        "count_possible_transitions",
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_levels": (4, 5), "qtypes": ["energy_difference", "count_possible_transitions"]}
        if level == 1:
            return {"n_levels": (4, 5), "qtypes": ["ground_state_energy", "count_possible_transitions"]}
        if level == 2:
            return {"n_levels": (4, 5), "qtypes": ["energy_difference", "count_possible_transitions"]}
        if level == 3:
            return {"n_levels": (4, 5), "qtypes": ["energy_difference", "absorption_vs_emission"]}
        if level == 4:
            return {"n_levels": (4, 6), "qtypes": ["energy_difference", "absorption_vs_emission"]}
        if level == 5:
            return {"n_levels": (5, 6), "qtypes": ["absorption_vs_emission", "highest_energy_transition"]}
        if level == 6:
            return {"n_levels": (5, 6), "qtypes": ["highest_energy_transition", "energy_difference"]}
        if level == 7:
            return {"n_levels": (5, 6), "qtypes": ["highest_energy_transition"]}
        if level == 8:
            return {"n_levels": (5, 6), "qtypes": ["highest_energy_transition", "energy_difference"]}
        return {"n_levels": (5, 6), "qtypes": ["highest_energy_transition"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 709)
        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])

        lo_n, hi_n = cfg["n_levels"]
        num_levels = sub_rng.randint(lo_n, hi_n)
        diagram_type = sub_rng.choice(["atom", "generic"])

        # Generate energy levels (sorted, lowest first)
        if diagram_type == "atom":
            unit_set = rng.choice(_ENERGY_UNIT_SETS)
            unit = unit_set[0]
            base_energies = list(unit_set[1][0][:num_levels])
            system_name = unit_set[1][1]
            labels = _LEVEL_LABELS_ATOM[:num_levels]
        else:
            unit = "eV"
            system_name = "Energy System"
            # Random increasing energies
            base = rng.uniform(-15, -8)
            energies_raw = [base]
            for _ in range(num_levels - 1):
                step = rng.uniform(1.0, 4.0)
                energies_raw.append(energies_raw[-1] + step)
            base_energies = [round(e, 2) for e in energies_raw]
            labels = _LEVEL_LABELS_GENERIC[:num_levels]

        energies = base_energies

        # Generate some transitions (arrows)
        all_pairs = list(combinations(range(num_levels), 2))
        num_transitions = rng.randint(2, min(5, len(all_pairs)))
        rng.shuffle(all_pairs)
        transitions = []
        for lower, upper in all_pairs[:num_transitions]:
            is_absorption = rng.choice([True, False])
            transitions.append({
                "from_level": lower if is_absorption else upper,
                "to_level": upper if is_absorption else lower,
                "is_absorption": is_absorption,
            })

        question, answer = self._make_qa(
            rng, qtype, energies, labels, transitions, unit, num_levels
        )
        if question is None:
            return None

        image = self._render(
            rng, energies, labels, transitions, unit, system_name
        )
        return question, answer, image

    def _make_qa(
        self, rng, qtype, energies, labels, transitions,
        unit, num_levels
    ):
        if qtype == "energy_difference":
            # Pick two random levels
            i, j = sorted(rng.sample(range(num_levels), 2))
            diff = round(abs(energies[j] - energies[i]), 2)
            q = (f"What is the energy difference between level {labels[i]} "
                 f"and level {labels[j]}? Give the absolute value in {unit}.")
            return q, str(diff)

        elif qtype == "highest_energy_transition":
            if not transitions:
                return None, None
            max_t = max(transitions,
                        key=lambda t: abs(energies[t["to_level"]] - energies[t["from_level"]]))
            from_l = labels[max_t["from_level"]]
            to_l = labels[max_t["to_level"]]
            q = ("Which transition shown in the diagram has the largest "
                 "energy change? Express as 'X to Y' using the level labels.")
            return q, f"{from_l} to {to_l}"

        elif qtype == "absorption_vs_emission":
            if not transitions:
                return None, None
            t = rng.choice(transitions)
            from_l = labels[t["from_level"]]
            to_l = labels[t["to_level"]]
            ans = "absorption" if t["is_absorption"] else "emission"
            q = (f"The transition from {from_l} to {to_l} shown in the "
                 f"diagram — is this absorption or emission?")
            return q, ans

        elif qtype == "ground_state_energy":
            q = (f"What is the energy of the ground state (lowest level) "
                 f"in {unit}?")
            return q, str(energies[0])

        elif qtype == "count_possible_transitions":
            n = num_levels
            total = n * (n - 1) // 2
            q = (f"The diagram shows {n} energy levels. How many distinct "
                 f"transitions are possible between any two levels? "
                 f"(Count each pair once, regardless of direction.)")
            return q, str(total)

        return None, None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(
        self, rng, energies, labels, transitions, unit, system_name
    ) -> Image.Image:
        vstyle = self._random_style()
        sc = vstyle["figsize_scale"]
        fig, ax = plt.subplots(figsize=(8 * sc, 7 * sc))
        fig.patch.set_facecolor(vstyle["bg_color"])
        ax.set_facecolor(vstyle["bg_color"])

        palette = vstyle["palette"]
        lw = vstyle["line_width"]
        fs = vstyle["font_size_base"]
        ff = vstyle["font_family"]

        num_levels = len(energies)
        e_min, e_max = min(energies), max(energies)
        e_range = max(e_max - e_min, 1e-9)

        level_width = 3.0
        x_center = 2.5
        x_left = x_center - level_width / 2
        x_right = x_center + level_width / 2

        def e_to_y(e):
            return 1.0 + 8.0 * (e - e_min) / e_range

        # Draw energy levels. Do NOT print numeric energy values next to each
        # level — that would leak answers for energy_difference and
        # ground_state_energy. The model must read positions from the y-axis.
        for i, (e, label) in enumerate(zip(energies, labels)):
            y = e_to_y(e)
            color = palette[i % len(palette)]
            ax.plot([x_left, x_right], [y, y], color=color, linewidth=lw + 1,
                    solid_capstyle="round")
            ax.text(x_left - 0.3, y, label, fontsize=fs - 1, fontweight="bold",
                    ha="right", va="center", color=color, fontfamily=ff)

        # Draw transitions as arrows
        for idx, t in enumerate(transitions):
            from_y = e_to_y(energies[t["from_level"]])
            to_y = e_to_y(energies[t["to_level"]])
            x_offset = x_center + (idx - len(transitions) / 2) * 0.35

            if t["is_absorption"]:
                color = palette[2 % len(palette)]
                astyle = "-|>"
            else:
                color = palette[1 % len(palette)]
                astyle = "-|>"

            ax.annotate(
                "", xy=(x_offset, to_y), xytext=(x_offset, from_y),
                arrowprops=dict(arrowstyle=astyle, color=color, lw=lw,
                                mutation_scale=15,
                                linestyle="--" if t["is_absorption"] else "-"))
            # Do NOT print the numeric delta-E next to each arrow: it would
            # leak the answer for highest_energy_transition.

        # Legend
        abs_patch = mpatches.Patch(color=palette[2 % len(palette)], label="Absorption (up)")
        emi_patch = mpatches.Patch(color=palette[1 % len(palette)], label="Emission (down)")
        ax.legend(handles=[abs_patch, emi_patch], loc=vstyle["legend_loc"], fontsize=fs - 2)

        ax.set_ylabel(f"Energy ({unit})", fontsize=fs, fontfamily=ff)
        ax.set_title(f"Energy Level Diagram: {system_name}",
                     fontsize=fs + 2, fontweight="bold", fontfamily=ff)
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 10)
        ax.set_xticks([])

        # Add a regular numeric y-axis on the right with evenly-spaced round
        # gridlines (NOT aligned to level energies), so the model must
        # estimate each level's energy from its vertical position.
        span = e_max - e_min
        pad = max(0.1 * span, 0.5)
        lo = e_min - pad
        hi = e_max + pad
        # Choose a round step yielding 5..7 ticks
        raw_step = (hi - lo) / 6
        mag = 10 ** math.floor(math.log10(max(raw_step, 1e-6)))
        for m in (1, 2, 2.5, 5, 10):
            step = m * mag
            if step >= raw_step:
                break
        first = math.ceil(lo / step) * step
        ticks = []
        t = first
        while t <= hi + 1e-9:
            ticks.append(round(t, 4))
            t += step
        ax2 = ax.twinx()
        ax2.set_yticks([e_to_y(e) for e in ticks])
        ax2.set_yticklabels([f"{int(e) if abs(e - round(e)) < 1e-6 else round(e, 2)}"
                             for e in ticks], fontsize=fs - 4)
        ax2.set_ylim(0, 10)
        ax2.set_ylabel(f"Energy ({unit})", fontsize=fs - 2, fontfamily=ff)

        ax.spines["top"].set_visible(vstyle.get("spine_top", False))
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vstyle["dpi"])
