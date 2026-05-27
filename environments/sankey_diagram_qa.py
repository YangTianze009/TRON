"""Sankey Diagram Visual QA Environment.

Draws a simplified Sankey-like flow diagram with labeled flows.

Capabilities: V3 (chart extraction), V2 (label reading), R1 (arithmetic)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 2 sources, 2 destinations, integer labels. Ask "flow from A to B".
L1: 2x2, ask "total outflow from source A".
L2: 2x3, ask "flow from A to B".
L3: 3x2, ask "total inflow to dest B".
L4: 3x3, ask "flow from A to B".
L5: 3x3, ask "max single path".
L6: 3x3, ask "compare two source totals" (which is bigger).
L7: 4x3, ask "flow from A to B".
L8: 3x4 (with intermediate-stage style), ask total inflow to dest.
L9: 4x4, ask "max single path".

parameter = {"level": int in [0, 9]}
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

_SOURCE_NAME_POOLS = [
    ["Solar", "Wind", "Hydro", "Nuclear"],
    ["East", "West", "North", "South"],
    ["Factory A", "Factory B", "Factory C", "Factory D"],
    ["Budget A", "Budget B", "Budget C", "Budget D"],
    ["Coal", "Oil", "Gas", "Bio"],
    ["Plant 1", "Plant 2", "Plant 3", "Plant 4"],
]
_DEST_NAME_POOLS = [
    ["Residential", "Commercial", "Industrial", "Other"],
    ["Sales", "Marketing", "R&D", "Support"],
    ["Product X", "Product Y", "Product Z", "Product W"],
    ["Region 1", "Region 2", "Region 3", "Region 4"],
    ["NA", "EU", "AS", "AF"],
    ["Inv", "Save", "Spend", "Donate"],
]
_TITLE_VARIANTS = [
    "Flow Diagram",
    "Resource Flows",
    "Sankey Flow",
    "Allocation Chart",
    "Source-to-Destination",
]

class SankeyDiagramQA(StandaloneVisualEnv):
    ENV_NAME = "sankey_diagram"

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
            # L0: 2 sources, 2 dests, read a single labelled flow value
            return {"ns": 2, "nd": 2, "qtype": "flow_between"}
        if level == 1:
            return {"ns": 2, "nd": 2, "qtype": "total_from_source"}
        if level == 2:
            return {"ns": 2, "nd": 3, "qtype": "flow_between"}
        if level == 3:
            return {"ns": 3, "nd": 2, "qtype": "net_flow_to_dest"}
        if level == 4:
            return {"ns": 3, "nd": 3, "qtype": "flow_between"}
        if level == 5:
            return {"ns": 3, "nd": 3, "qtype": "compare_sources"}
        if level == 6:
            return {"ns": 3, "nd": 3, "qtype": "net_flow_to_dest"}
        if level == 7:
            return {"ns": 4, "nd": 3, "qtype": "flow_between"}
        if level == 8:
            return {"ns": 3, "nd": 4, "qtype": "max_flow_path"}
        return {"ns": 4, "nd": 4, "qtype": "max_flow_path"}

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)
        ns, nd = cfg["ns"], cfg["nd"]

        sources = list(rng.choice(_SOURCE_NAME_POOLS))[:ns]
        dests = list(rng.choice(_DEST_NAME_POOLS))[:nd]

        flows = [[rng.randint(5, 50) for _ in range(nd)] for _ in range(ns)]

        question, answer = self._make_qa(rng, cfg["qtype"], sources, dests, flows)
        if question is None:
            return None

        style = self._random_style()
        image = self._render(rng, style, sources, dests, flows)
        return question, str(answer), image

    def _make_qa(self, rng, qtype, sources, dests, flows):
        ns, nd = len(sources), len(dests)
        if qtype == "flow_between":
            s = rng.randint(0, ns - 1)
            d = rng.randint(0, nd - 1)
            stems = [
                f"What is the flow from '{sources[s]}' to '{dests[d]}'? "
                f"Answer with a single integer.",
                f"Read the value labeled on the band from '{sources[s]}' to "
                f"'{dests[d]}'.",
            ]
            return rng.choice(stems), flows[s][d]
        if qtype == "total_from_source":
            s = rng.randint(0, ns - 1)
            return (f"What is the total outflow from '{sources[s]}' (sum across all "
                    f"destinations)? Answer with a single integer.", sum(flows[s]))
        if qtype == "net_flow_to_dest":
            d = rng.randint(0, nd - 1)
            return (f"What is the total inflow to '{dests[d]}' (sum across all "
                    f"sources)? Answer with a single integer.",
                    sum(flows[s][d] for s in range(ns)))
        if qtype == "max_flow_path":
            mx, ms, md = -1, 0, 0
            for s in range(ns):
                for d in range(nd):
                    if flows[s][d] > mx:
                        mx, ms, md = flows[s][d], s, d
            return ("Which single path carries the most flow? Answer as "
                    "'Source -> Destination'.", f"{sources[ms]} -> {dests[md]}")
        if qtype == "compare_sources":
            s1, s2 = rng.sample(range(ns), 2)
            t1, t2 = sum(flows[s1]), sum(flows[s2])
            bigger = sources[s1] if t1 >= t2 else sources[s2]
            return (f"Which source has the larger total outflow: '{sources[s1]}' "
                    f"or '{sources[s2]}'? Answer with the source name.", bigger)
        return None, None

    def _render(self, rng, style, sources, dests, flows):
        ns, nd = len(sources), len(dests)
        fig, ax = plt.subplots(figsize=(8 * style["figsize_scale"],
                                        5 * style["figsize_scale"]))
        palette = list(style["palette"])
        rng.shuffle(palette)

        total_src = [sum(flows[s]) for s in range(ns)]
        total_dst = [sum(flows[s][d] for s in range(ns)) for d in range(nd)]
        max_h = max(max(total_src), max(total_dst), 1)

        # NOTE: Do NOT display source/dest totals as labels — that would
        # leak the answer for total_from_source (L1) and net_flow_to_dest
        # (L3/L6/L8) question types.
        src_y = []
        y = 0.0
        for s in range(ns):
            h = total_src[s] / max_h * 4 + 0.3
            src_y.append((y, h))
            rect = mpatches.FancyBboxPatch(
                (0.5, y), 0.4, h, boxstyle="round,pad=0.05",
                facecolor=palette[s % len(palette)], alpha=0.85)
            ax.add_patch(rect)
            ax.text(0.3, y + h / 2, f"{sources[s]}",
                    ha="right", va="center", fontsize=style["font_size_base"] - 1)
            y += h + 0.4
        max_y = y

        dst_y = []
        y = 0.0
        for d in range(nd):
            h = total_dst[d] / max_h * 4 + 0.3
            dst_y.append((y, h))
            rect = mpatches.FancyBboxPatch(
                (5.1, y), 0.4, h, boxstyle="round,pad=0.05",
                facecolor=palette[(ns + d) % len(palette)], alpha=0.85)
            ax.add_patch(rect)
            ax.text(5.7, y + h / 2, f"{dests[d]}",
                    ha="left", va="center", fontsize=style["font_size_base"] - 1)
            y += h + 0.4
        max_y = max(max_y, y)

        # Flow bands.  Stagger label x-positions across a band so that
        # labels from different flows (which often share similar vertical
        # midpoints) do not collide.  Also track placed label positions and
        # push new ones away vertically if they would overlap.
        src_offsets = [0.0] * ns
        dst_offsets = [0.0] * nd
        # Pre-compute flat index list for staggering so neighbouring flows
        # get distinct x offsets.
        flat_idx = [(s, d) for s in range(ns) for d in range(nd)]
        total_flows = len(flat_idx)
        # Stagger over 5 columns between x=1.8 and x=4.2
        n_cols = min(5, max(3, total_flows // 3))
        col_xs = np.linspace(1.8, 4.2, n_cols)
        placed = []  # list of (x, y) of label centers
        min_dy = 0.28  # min vertical gap between labels in same column
        for k, (s, d) in enumerate(flat_idx):
            fv = flows[s][d]
            fh = fv / max_h * 4
            sy = src_y[s][0] + src_offsets[s]
            dy = dst_y[d][0] + dst_offsets[d]
            verts = [(0.9, sy), (0.9, sy + fh),
                     (5.1, dy + fh), (5.1, dy)]
            poly = plt.Polygon(verts, alpha=0.30,
                               color=palette[s % len(palette)])
            ax.add_patch(poly)
            mid_x = float(col_xs[k % n_cols])
            # Interpolate along the band at fraction f_mid
            f_mid = (mid_x - 0.9) / (5.1 - 0.9)
            mid_y = (sy + fh / 2) * (1.0 - f_mid) + (dy + fh / 2) * f_mid
            # Nudge if overlapping a previously placed label in the same col
            for (px, py) in placed:
                if abs(px - mid_x) < 0.35 and abs(py - mid_y) < min_dy:
                    mid_y = py + (min_dy if mid_y >= py else -min_dy)
            placed.append((mid_x, mid_y))
            ax.text(mid_x, mid_y, str(fv), ha="center", va="center",
                    fontsize=style["font_size_base"] - 2, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="#888", alpha=0.85))
            src_offsets[s] += fh
            dst_offsets[d] += fh

        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, max_y + 0.5)
        ax.set_title(rng.choice(_TITLE_VARIANTS),
                     fontsize=style["font_size_base"] + 2)
        ax.axis("off")
        fig.patch.set_facecolor(style["bg_color"])
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = SankeyDiagramQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
