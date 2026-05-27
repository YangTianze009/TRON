"""
Periodic Table (Partial) QA environment.

Draws a portion of the periodic table (2-3 rows x 4-6 columns) with
atomic number, symbol, and atomic mass. Questions about element properties,
comparisons, groups, electron counts.
Targets: MMStar science diagrams, MMMU chemistry.
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
# Element data (subset)
# ------------------------------------------------------------------ #

# (atomic_number, symbol, name, atomic_mass, category)
# category: metal, nonmetal, metalloid, noble_gas, halogen, alkali, alkaline_earth
_ELEMENTS = [
    (1, "H", "Hydrogen", 1.008, "nonmetal"),
    (2, "He", "Helium", 4.003, "noble_gas"),
    (3, "Li", "Lithium", 6.941, "alkali"),
    (4, "Be", "Beryllium", 9.012, "alkaline_earth"),
    (5, "B", "Boron", 10.81, "metalloid"),
    (6, "C", "Carbon", 12.01, "nonmetal"),
    (7, "N", "Nitrogen", 14.01, "nonmetal"),
    (8, "O", "Oxygen", 16.00, "nonmetal"),
    (9, "F", "Fluorine", 19.00, "halogen"),
    (10, "Ne", "Neon", 20.18, "noble_gas"),
    (11, "Na", "Sodium", 22.99, "alkali"),
    (12, "Mg", "Magnesium", 24.31, "alkaline_earth"),
    (13, "Al", "Aluminum", 26.98, "metal"),
    (14, "Si", "Silicon", 28.09, "metalloid"),
    (15, "P", "Phosphorus", 30.97, "nonmetal"),
    (16, "S", "Sulfur", 32.07, "nonmetal"),
    (17, "Cl", "Chlorine", 35.45, "halogen"),
    (18, "Ar", "Argon", 39.95, "noble_gas"),
    (19, "K", "Potassium", 39.10, "alkali"),
    (20, "Ca", "Calcium", 40.08, "alkaline_earth"),
    (21, "Sc", "Scandium", 44.96, "metal"),
    (22, "Ti", "Titanium", 47.87, "metal"),
    (23, "V", "Vanadium", 50.94, "metal"),
    (24, "Cr", "Chromium", 52.00, "metal"),
    (25, "Mn", "Manganese", 54.94, "metal"),
    (26, "Fe", "Iron", 55.85, "metal"),
    (27, "Co", "Cobalt", 58.93, "metal"),
    (28, "Ni", "Nickel", 58.69, "metal"),
    (29, "Cu", "Copper", 63.55, "metal"),
    (30, "Zn", "Zinc", 65.38, "metal"),
    (31, "Ga", "Gallium", 69.72, "metal"),
    (32, "Ge", "Germanium", 72.63, "metalloid"),
    (33, "As", "Arsenic", 74.92, "metalloid"),
    (34, "Se", "Selenium", 78.97, "nonmetal"),
    (35, "Br", "Bromine", 79.90, "halogen"),
    (36, "Kr", "Krypton", 83.80, "noble_gas"),
]

_CATEGORY_COLORS = {
    "metal": "#AED6F1",
    "nonmetal": "#ABEBC6",
    "metalloid": "#F9E79F",
    "noble_gas": "#D7BDE2",
    "halogen": "#F5CBA7",
    "alkali": "#F1948A",
    "alkaline_earth": "#FAD7A0",
}

# Standard periodic table layout: (period, group) -> atomic number
# We only need rows 1-4 for our subset
_PT_LAYOUT = {}
# Row 1
_PT_LAYOUT[(1, 1)] = 1
_PT_LAYOUT[(1, 18)] = 2
# Row 2
for i, z in enumerate(range(3, 11), start=0):
    col = [1, 2, 13, 14, 15, 16, 17, 18][i]
    _PT_LAYOUT[(2, col)] = z
# Row 3
for i, z in enumerate(range(11, 19), start=0):
    col = [1, 2, 13, 14, 15, 16, 17, 18][i]
    _PT_LAYOUT[(3, col)] = z
# Row 4
for i, z in enumerate(range(19, 37), start=0):
    col = i + 1
    _PT_LAYOUT[(4, col)] = z

_ELEM_BY_Z = {e[0]: e for e in _ELEMENTS}

class PeriodicTableQA(StandaloneVisualEnv):
    ENV_NAME = "periodic_table"

    QUESTION_TYPES = [
        "find_atomic_mass", "compare_elements", "identify_group",
        "predict_property", "count_electrons", "which_heavier",
        "electron_configuration_shorthand", "ionization_trend",
        # Visual-reasoning types added 2026-04-17. These all require reading
        # the element's position (period/group) from the rendered table
        # rather than relying on memorised chemistry facts.
        "position_to_group", "position_to_period", "element_at_cell",
    ]

    def _level_config(self, level: int) -> Dict:
        # Difficulty redesign 2026-04-14. L0 = which_heavier (was L2-L3).
        if level <= 0:
            return {"qtypes": ["which_heavier", "compare_elements"],
                    "qweights": [5, 5], "rows_range": (1, 3), "num_rows": (2, 2), "num_cols": (4, 5)}
        if level == 1:
            return {"qtypes": ["compare_elements", "predict_property", "which_heavier"],
                    "qweights": [4, 3, 3], "rows_range": (1, 3), "num_rows": (2, 3), "num_cols": (4, 6)}
        if level == 2:
            return {"qtypes": ["compare_elements", "predict_property", "identify_group"],
                    "qweights": [3, 4, 3], "rows_range": (1, 3), "num_rows": (2, 3), "num_cols": (4, 6)}
        if level == 3:
            return {"qtypes": ["predict_property", "ionization_trend", "compare_elements"],
                    "qweights": [3, 4, 3], "rows_range": (1, 3), "num_rows": (2, 3), "num_cols": (4, 6)}
        if level == 4:
            return {"qtypes": ["ionization_trend", "compare_elements", "predict_property"],
                    "qweights": [4, 3, 3], "rows_range": (2, 4), "num_rows": (2, 3), "num_cols": (5, 6)}
        if level == 5:
            return {"qtypes": ["compare_elements", "predict_property", "ionization_trend"],
                    "qweights": [3, 3, 4], "rows_range": (1, 3), "num_rows": (2, 3), "num_cols": (5, 6)}
        if level == 6:
            return {"qtypes": ["compare_elements", "ionization_trend", "electron_configuration_shorthand"],
                    "qweights": [3, 3, 4], "rows_range": (2, 4), "num_rows": (2, 3), "num_cols": (5, 6)}
        if level == 7:
            return {"qtypes": ["ionization_trend", "electron_configuration_shorthand", "compare_elements"],
                    "qweights": [4, 4, 2], "rows_range": (2, 4), "num_rows": (2, 3), "num_cols": (5, 6)}
        if level == 8:
            # Visual-reasoning mix at L8: primarily position-based reading
            # (removes knowledge-leak of pure ionization/electron-config
            # recall).
            return {"qtypes": ["position_to_group", "position_to_period",
                               "element_at_cell", "ionization_trend"],
                    "qweights": [4, 3, 3, 2],
                    "rows_range": (1, 4), "num_rows": (2, 3), "num_cols": (5, 6)}
        # L9: pure visual-position reasoning with one ionisation-trend
        # distractor qtype mixed in. The table itself is the source of truth
        # for group (column) and period (row); the answer is not derivable
        # from memorised Z->element mapping alone.
        return {"qtypes": ["position_to_group", "position_to_period",
                           "element_at_cell"],
                "qweights": [4, 4, 3],
                "rows_range": (1, 4), "num_rows": (2, 3), "num_cols": (5, 6)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random(seed * 1000 + level * 37 + 5501)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        # Choose a rectangular region of the periodic table
        # Pick a starting row and column range
        start_row = sub_rng.randint(*cfg["rows_range"])
        num_rows = sub_rng.randint(*cfg["num_rows"])
        num_rows = min(num_rows, 5 - start_row)
        # Pick contiguous columns that have elements
        possible_cols = sorted(set(c for (r, c) in _PT_LAYOUT
                                   if start_row <= r < start_row + num_rows))
        if len(possible_cols) < 4:
            possible_cols = list(range(1, 19))

        # Pick 4-6 contiguous columns
        num_cols = rng.randint(4, min(6, len(possible_cols)))
        start_idx = rng.randint(0, max(0, len(possible_cols) - num_cols))
        selected_cols = possible_cols[start_idx:start_idx + num_cols]

        # Collect elements in the region
        elements_in_view = []
        grid_positions = {}  # (row_idx, col_idx) -> element tuple
        for ri, row in enumerate(range(start_row, start_row + num_rows)):
            for ci, col in enumerate(selected_cols):
                z = _PT_LAYOUT.get((row, col))
                if z and z in _ELEM_BY_Z:
                    elem = _ELEM_BY_Z[z]
                    elements_in_view.append(elem)
                    grid_positions[(ri, ci)] = elem

        if len(elements_in_view) < 3:
            return None

        question, answer = self._make_qa(rng, qtype, elements_in_view,
                                         grid_positions, selected_cols,
                                         start_row, num_rows)
        if question is None:
            return None

        image = self._render(rng, grid_positions, num_rows, len(selected_cols),
                             selected_cols, start_row)
        return question, answer, image

    def _make_qa(self, rng, qtype, elements,
                 grid_positions=None, selected_cols=None,
                 start_row=None, num_rows=None):
        if qtype == "find_atomic_mass":
            elem = rng.choice(elements)
            q = (f"From the periodic table shown, what is the atomic mass "
                 f"of {elem[2]} ({elem[1]})?")
            return q, str(elem[3])

        elif qtype == "compare_elements":
            e1, e2 = rng.sample(elements, 2)
            diff = round(abs(e1[3] - e2[3]), 3)
            q = (f"What is the difference in atomic mass between "
                 f"{e1[2]} ({e1[1]}) and {e2[2]} ({e2[1]})? "
                 f"Give the absolute value.")
            return q, str(diff)

        elif qtype == "identify_group":
            elem = rng.choice(elements)
            cat = elem[4]
            cat_display = cat.replace("_", " ")
            q = (f"Looking at the periodic table, what category does "
                 f"{elem[2]} ({elem[1]}) belong to? Choose from: "
                 f"metal, nonmetal, metalloid, noble gas, halogen, "
                 f"alkali, alkaline earth.")
            return q, cat_display

        elif qtype == "predict_property":
            # Which element has the highest atomic number?
            max_elem = max(elements, key=lambda e: e[0])
            q = ("Which element shown in the table has the highest "
                 "atomic number? Give the element symbol.")
            return q, max_elem[1]

        elif qtype == "count_electrons":
            elem = rng.choice(elements)
            q = (f"How many electrons does a neutral atom of "
                 f"{elem[2]} ({elem[1]}) have?")
            return q, str(elem[0])

        elif qtype == "which_heavier":
            e1, e2 = rng.sample(elements, 2)
            heavier = e1 if e1[3] > e2[3] else e2
            q = (f"Which element is heavier: {e1[2]} ({e1[1]}) or "
                 f"{e2[2]} ({e2[1]})? Give the element symbol.")
            return q, heavier[1]

        elif qtype == "electron_configuration_shorthand":
            # Electron configuration using shell filling
            elem = rng.choice(elements)
            z = elem[0]
            # Build configuration
            orbitals = [(1, 's', 2), (2, 's', 2), (2, 'p', 6), (3, 's', 2),
                        (3, 'p', 6), (4, 's', 2), (3, 'd', 10), (4, 'p', 6)]
            config_parts = []
            remaining = z
            for n_shell, orbital, max_e in orbitals:
                if remaining <= 0:
                    break
                fill = min(remaining, max_e)
                config_parts.append(f"{n_shell}{orbital}{fill}")
                remaining -= fill
            config = " ".join(config_parts)
            q = (f"What is the electron configuration of {elem[2]} ({elem[1]}, "
                 f"Z={z})? Write in standard notation (e.g., 1s2 2s2 2p6).")
            return q, config

        elif qtype == "ionization_trend":
            # Compare two elements: which has higher first ionization energy?
            # General trend: increases left->right, decreases top->bottom
            if len(elements) < 2:
                return None, None
            e1, e2 = rng.sample(elements, 2)
            # Higher Z in same period = higher IE (simplified)
            # Noble gases > halogens > others; nonmetals > metals
            ie_rank = {"noble_gas": 6, "halogen": 5, "nonmetal": 4,
                       "metalloid": 3, "alkaline_earth": 2, "metal": 1, "alkali": 0}
            ie1 = ie_rank.get(e1[4], 2) * 10 + (e1[0] % 18)  # rough ranking
            ie2 = ie_rank.get(e2[4], 2) * 10 + (e2[0] % 18)
            higher = e1 if ie1 > ie2 else e2
            q = (f"Based on periodic trends, which element has a higher "
                 f"first ionization energy: {e1[2]} ({e1[1]}) or "
                 f"{e2[2]} ({e2[1]})? Give the element symbol.")
            return q, higher[1]

        elif qtype == "position_to_group":
            # Read the group number (column header) for the given element
            # from the table. The row/column labels are printed in the
            # rendered image (`Grp 13`, `P3`, etc.), so the answer comes
            # from spatial reading, not chemistry knowledge.
            if grid_positions is None or selected_cols is None:
                return None, None
            elem = rng.choice(list(grid_positions.values()))
            for (ri, ci), e in grid_positions.items():
                if e == elem:
                    col = selected_cols[ci]
                    q = (f"Looking at the periodic table shown, which GROUP "
                         f"(column number) does {elem[2]} ({elem[1]}) "
                         f"occupy in the table? Answer with a single integer.")
                    return q, str(col)
            return None, None

        elif qtype == "position_to_period":
            # Read the period number (row header) for the given element.
            if grid_positions is None or start_row is None:
                return None, None
            elem = rng.choice(list(grid_positions.values()))
            for (ri, ci), e in grid_positions.items():
                if e == elem:
                    period = start_row + ri
                    q = (f"Based on the periodic table shown, which PERIOD "
                         f"(row number) does {elem[2]} ({elem[1]}) "
                         f"occupy? Answer with a single integer.")
                    return q, str(period)
            return None, None

        elif qtype == "element_at_cell":
            # Given a (period, group) pair, ask for the element symbol
            # located there in the table. Purely positional reading.
            if grid_positions is None or selected_cols is None \
                    or start_row is None:
                return None, None
            (ri, ci) = rng.choice(list(grid_positions.keys()))
            elem = grid_positions[(ri, ci)]
            period = start_row + ri
            group = selected_cols[ci]
            q = (f"From the periodic table shown, which element occupies "
                 f"period {period}, group {group}? Give the element symbol.")
            return q, elem[1]

        return None, None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, rng, grid_positions, num_rows, num_cols,
                selected_cols, start_row) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        ff = style["font_family"]
        palette = style["palette"]

        cell_w, cell_h = 1.6, 1.8
        fig_w = (num_cols * cell_w + 1.5) * sc
        fig_h = (num_rows * cell_h + 2.0) * sc
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_xlim(-0.5, num_cols * cell_w + 0.5)
        ax.set_ylim(-num_rows * cell_h - 1.0, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.text(num_cols * cell_w / 2, 1.2, "Periodic Table (Partial)",
                ha="center", fontsize=fs + 2, fontweight="bold", fontfamily=ff)

        # Column headers (group numbers)
        for ci, col in enumerate(selected_cols):
            ax.text(ci * cell_w + cell_w / 2, 0.5, f"Grp {col}",
                    ha="center", fontsize=fs - 4, color="gray", fontfamily=ff)

        # Row headers (period numbers)
        for ri in range(num_rows):
            ax.text(-0.3, -ri * cell_h - cell_h / 2,
                    f"P{start_row + ri}", ha="center", fontsize=fs - 4,
                    color="gray", va="center", fontfamily=ff)

        # Map categories to palette colors. Skip any dark palette colors that
        # would render black text invisible (letters are black; cell must be
        # light enough for contrast). Use stable category colors as fallback.
        def _is_light(hex_str):
            try:
                h = hex_str.lstrip('#')
                if len(h) != 6: return True
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                # perceived luminance
                lum = (0.299 * r + 0.587 * g + 0.114 * b)
                return lum > 160  # require clearly light
            except Exception:
                return True
        cats_all = sorted(set(elem[4] for elem in grid_positions.values()))
        light_palette = [c for c in palette if _is_light(c)]
        # Fall back to stable _CATEGORY_COLORS if palette is all too dark.
        if len(light_palette) < len(cats_all):
            cat_color_map = {cat: _CATEGORY_COLORS.get(cat, "#EAECEE") for cat in cats_all}
        else:
            cat_color_map = {cat: light_palette[i % len(light_palette)] for i, cat in enumerate(cats_all)}

        for (ri, ci), elem in grid_positions.items():
            z, sym, name, mass, cat = elem
            x0 = ci * cell_w
            y0 = -ri * cell_h

            color = cat_color_map.get(cat, "#EAECEE")
            rect = mpatches.FancyBboxPatch(
                (x0 + 0.05, y0 - cell_h + 0.05),
                cell_w - 0.1, cell_h - 0.1,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="black", linewidth=style["line_width"])
            ax.add_patch(rect)

            # Atomic number (top-left)
            ax.text(x0 + 0.15, y0 - 0.15, str(z),
                    fontsize=fs - 5, ha="left", va="top", color="#555555",
                    fontfamily=ff)

            # Symbol (center, large)
            ax.text(x0 + cell_w / 2, y0 - cell_h / 2 + 0.1,
                    sym, fontsize=fs + 4, fontweight="bold",
                    ha="center", va="center", color="black", fontfamily=ff)

            # Atomic mass (bottom)
            ax.text(x0 + cell_w / 2, y0 - cell_h + 0.25,
                    str(mass), fontsize=fs - 5, ha="center", va="center",
                    color="#333333", fontfamily=ff)

        # Legend
        legend_y = -num_rows * cell_h - 0.5
        lx = 0
        for cat in cats_all:
            color = cat_color_map.get(cat, "#EAECEE")
            ax.add_patch(mpatches.Rectangle((lx, legend_y), 0.3, 0.3,
                         facecolor=color, edgecolor="black"))
            label = cat.replace("_", " ").title()
            ax.text(lx + 0.4, legend_y + 0.15, label, fontsize=fs - 5,
                    va="center", fontfamily=ff)
            lx += len(label) * 0.1 + 1.0

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
