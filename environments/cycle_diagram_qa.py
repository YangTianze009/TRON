"""
Cycle Diagram QA environment.

Draws cyclical processes (water cycle, rock cycle, abstract process cycles)
as stages arranged in a circle with directional arrows.  Questions cover
stage ordering, next/previous stages, and counting.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class CycleDiagramQA(StandaloneVisualEnv):
    ENV_NAME = "cycle_diagram"

    QUESTION_TYPES = [
        "next_stage",
        "count_stages",
        "find_stage_after_n",
        "reverse_order",
        "which_stage_before",
    ]

    # Preset cycles
    _CYCLES = [
        {
            "name": "Water Cycle",
            "stages": ["Evaporation", "Condensation", "Precipitation",
                       "Collection", "Infiltration"],
        },
        {
            "name": "Rock Cycle",
            "stages": ["Magma", "Igneous Rock", "Sediment",
                       "Sedimentary Rock", "Metamorphic Rock"],
        },
        {
            "name": "Carbon Cycle",
            "stages": ["Atmosphere CO2", "Photosynthesis", "Plant Biomass",
                       "Decomposition", "Fossil Fuels", "Combustion"],
        },
        {
            "name": "Cell Cycle",
            "stages": ["G1 Phase", "S Phase", "G2 Phase", "Mitosis"],
        },
        {
            "name": "Day-Night Cycle",
            "stages": ["Dawn", "Morning", "Noon", "Afternoon",
                       "Dusk", "Night"],
        },
        {
            "name": "Butterfly Life Cycle",
            "stages": ["Egg", "Larva", "Pupa", "Adult"],
        },
        {
            "name": "Nitrogen Cycle",
            "stages": ["Nitrogen Gas", "Fixation", "Ammonium",
                       "Nitrification", "Nitrates", "Denitrification"],
        },
        {
            "name": "Product Life Cycle",
            "stages": ["Introduction", "Growth", "Maturity", "Decline"],
        },
        {
            "name": "Seasons",
            "stages": ["Spring", "Summer", "Autumn", "Winter"],
        },
        {
            "name": "Software Development Cycle",
            "stages": ["Requirements", "Design", "Implementation",
                       "Testing", "Deployment", "Maintenance"],
        },
    ]

    _STAGE_COLORS = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60",
    ]

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_stages_range": (4, 5), "qtypes": ["count_stages", "next_stage"]}
        if level == 1:
            return {"n_stages_range": (4, 5), "qtypes": ["count_stages", "next_stage"]}
        if level == 2:
            return {"n_stages_range": (4, 5), "qtypes": ["next_stage", "which_stage_before"]}
        if level == 3:
            return {"n_stages_range": (4, 6), "qtypes": ["next_stage", "which_stage_before"]}
        if level == 4:
            return {"n_stages_range": (4, 6), "qtypes": ["next_stage", "which_stage_before", "find_stage_after_n"]}
        if level == 5:
            return {"n_stages_range": (5, 6), "qtypes": ["find_stage_after_n", "which_stage_before"]}
        if level == 6:
            return {"n_stages_range": (5, 6), "qtypes": ["find_stage_after_n", "reverse_order"]}
        if level == 7:
            return {"n_stages_range": (5, 6), "qtypes": ["reverse_order", "find_stage_after_n"]}
        if level == 8:
            return {"n_stages_range": (5, 6), "qtypes": ["reverse_order", "find_stage_after_n"]}
        return {"n_stages_range": (5, 6), "qtypes": ["reverse_order", "find_stage_after_n"]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 702)
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choice(cfg["qtypes"])
        # Filter cycles by stage count range
        lo, hi = cfg["n_stages_range"]
        valid_cycles = [c for c in self._CYCLES if lo <= len(c["stages"]) <= hi]
        if not valid_cycles:
            valid_cycles = self._CYCLES

        for _ in range(20):
            result = self._try_generate(qtype, valid_cycles, sub_rng)
            if result is not None:
                return result
        return None

    def _try_generate(self, qtype: str, cycle_pool=None, sub_rng=None) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        pool = cycle_pool if cycle_pool else self._CYCLES
        cycle = (sub_rng or rng).choice(pool)
        name = cycle["name"]
        stages = list(cycle["stages"])
        n = len(stages)

        img = self._render_cycle(rng, name, stages)

        if qtype == "next_stage":
            idx = rng.randint(0, n - 1)
            nxt = stages[(idx + 1) % n]
            q = (f'In the "{name}" diagram, what stage comes immediately '
                 f'after "{stages[idx]}"?')
            return q, nxt, img

        elif qtype == "count_stages":
            q = f'How many stages are shown in the "{name}" cycle diagram?'
            return q, str(n), img

        elif qtype == "find_stage_after_n":
            start_idx = rng.randint(0, n - 1)
            steps = rng.randint(2, min(n + 2, 8))
            target_idx = (start_idx + steps) % n
            q = (f'Starting from "{stages[start_idx]}" in the "{name}" cycle, '
                 f"what stage do you reach after moving forward {steps} steps?")
            return q, stages[target_idx], img

        elif qtype == "reverse_order":
            # Ask for the stage that is k steps BACKWARD
            idx = rng.randint(0, n - 1)
            k = rng.randint(1, 3)
            prev_idx = (idx - k) % n
            direction_word = "step" if k == 1 else "steps"
            q = (f'In the "{name}" cycle, what stage is {k} {direction_word} '
                 f'BEFORE "{stages[idx]}" (going backward/counter-clockwise)?')
            return q, stages[prev_idx], img

        elif qtype == "which_stage_before":
            idx = rng.randint(0, n - 1)
            prev_idx = (idx - 1) % n
            q = (f'In the "{name}" diagram, what stage comes immediately '
                 f'before "{stages[idx]}"?')
            return q, stages[prev_idx], img

        return None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_cycle(self, rng, cycle_name: str,
                      stages: List[str]) -> Image.Image:
        style = self._random_style()
        s = style["figsize_scale"]
        n = len(stages)
        fig, ax = plt.subplots(figsize=(7 * s, 7 * s))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        palette = style["palette"]
        edge_color = style["geo_line_color"]
        lw = style["line_width"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        radius = 1.1
        angles = [math.pi / 2 - i * 2 * math.pi / n for i in range(n)]
        positions = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]

        box_w = 0.32
        box_h = 0.16

        # Draw arrows between consecutive stages
        for i in range(n):
            x1, y1 = positions[i]
            x2, y2 = positions[(i + 1) % n]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.01:
                continue
            # Shrink more aggressively so arrowhead sits clear of the
            # destination box (boxes have zorder=5; we now set arrow
            # zorder=7 so arrowheads render above boxes for visibility).
            shrink_frac = 0.28
            sx = x1 + shrink_frac * dx
            sy = y1 + shrink_frac * dy
            ex = x2 - shrink_frac * dx
            ey = y2 - shrink_frac * dy
            ax.annotate(
                "", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.35,head_length=0.28",
                    color=edge_color, lw=lw + 0.5,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=7,
            )

        # Draw stage boxes
        for i, (stage, (x, y)) in enumerate(zip(stages, positions)):
            color = palette[i % len(palette)]
            rect = FancyBboxPatch(
                (x - box_w, y - box_h), 2 * box_w, 2 * box_h,
                boxstyle="round,pad=0.06", facecolor=color,
                edgecolor=edge_color, linewidth=lw, zorder=5,
            )
            ax.add_patch(rect)
            fontsize = fs - 2 if len(stage) > 12 else fs - 1
            ax.text(x, y, stage, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="white",
                    zorder=6, wrap=True, fontfamily=ff)

        ax.set_title(cycle_name, fontsize=fs + 3, fontweight="bold", pad=12,
                     fontfamily=ff)

        if rng.random() < 0.5:
            ax.text(0, 0, "\u21bb", fontsize=28, ha="center", va="center",
                    color="#cccccc", alpha=0.4)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
