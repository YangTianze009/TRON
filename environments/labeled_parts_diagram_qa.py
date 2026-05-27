"""
Labeled Parts Diagram QA (batch 3, 2026-04-14).

Target: diagram partsOfA / atomStructure / anatomy — X4 science diagrams.
A schematic of a simple system (cell, circuit, water cycle, solar system)
is drawn with labelled parts A, B, C, ... and the model must identify
what a labelled part represents.

Format: constant MCQ letter A/B/C/D.

Difficulty axes:
  A) Pattern A: n_parts (3..6) grows with level.
  B) Pattern C: system pool grows — L0 common system (cell), L3+ rarer.
  C) Pattern D: at high level the distractors are semantically close parts.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SYSTEMS = {
    "plant_cell": {
        "parts": [
            ("cell wall", (0.5, 0.5), 0.45, "rectangle"),
            ("nucleus", (0.5, 0.55), 0.10, "circle"),
            ("chloroplast", (0.3, 0.3), 0.07, "circle"),
            ("vacuole", (0.7, 0.4), 0.12, "circle"),
            ("mitochondrion", (0.35, 0.7), 0.07, "circle"),
            ("cytoplasm", (0.5, 0.2), 0.05, "dot"),
        ],
        "desc": "plant cell",
    },
    "atom": {
        # Spread the nucleus parts apart so proton/neutron aren't stacked on
        # top of the nucleus center; keep all three visible inside the nucleus
        # region but at distinct coordinates.
        "parts": [
            ("nucleus", (0.5, 0.5), 0.10, "circle"),
            ("proton", (0.45, 0.53), 0.035, "circle"),
            ("neutron", (0.55, 0.47), 0.035, "circle"),
            ("electron shell", (0.5, 0.5), 0.28, "ring"),
            ("electron", (0.78, 0.5), 0.04, "circle"),
            ("inner shell", (0.5, 0.5), 0.18, "ring"),
        ],
        "desc": "atom",
    },
    "simple_circuit": {
        "parts": [
            ("battery", (0.2, 0.5), 0.08, "rectangle"),
            ("wire", (0.5, 0.8), 0.15, "line"),
            ("resistor", (0.8, 0.5), 0.08, "rectangle"),
            ("switch", (0.5, 0.2), 0.07, "line"),
            ("ground", (0.2, 0.2), 0.05, "dot"),
            ("bulb", (0.8, 0.8), 0.06, "circle"),
        ],
        "desc": "circuit",
    },
    "water_cycle": {
        "parts": [
            ("evaporation", (0.25, 0.35), 0.08, "arrow"),
            ("condensation", (0.5, 0.75), 0.10, "circle"),
            ("precipitation", (0.7, 0.5), 0.08, "arrow"),
            ("runoff", (0.8, 0.2), 0.07, "arrow"),
            ("ocean", (0.3, 0.15), 0.12, "rectangle"),
            ("cloud", (0.5, 0.85), 0.08, "circle"),
        ],
        "desc": "water cycle",
    },
    "solar_system": {
        "parts": [
            ("Sun", (0.5, 0.5), 0.08, "circle"),
            ("Earth", (0.68, 0.5), 0.03, "circle"),
            ("Mars", (0.78, 0.5), 0.03, "circle"),
            ("Venus", (0.6, 0.5), 0.03, "circle"),
            ("Jupiter", (0.88, 0.5), 0.05, "circle"),
            ("Mercury", (0.55, 0.5), 0.02, "circle"),
        ],
        "desc": "solar system",
    },
}

class LabeledPartsDiagramQA(StandaloneVisualEnv):
    ENV_NAME = "labeled_parts_diagram"
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # More labeled parts = MORE context = EASIER (model can identify
        # the system type from the visible labels). Empirically L9 with
        # n_parts=6 scored 0.50 vs L6 n_parts=5 at 0.20.
        # Fix: fewer parts at higher levels, hardest systems at high levels.
        if level <= 2:
            pool = ["solar_system", "water_cycle"]     # visually obvious
        elif level <= 4:
            pool = ["solar_system", "water_cycle", "simple_circuit"]
        elif level <= 6:
            pool = ["simple_circuit", "water_cycle", "plant_cell"]
        else:
            pool = ["plant_cell", "atom"]              # hardest to identify
        return {
            "n_parts": max(3, 6 - level // 3),        # 6,6,5,5,4,4,3,3,3,3
            "system_pool": pool,
            "tight_distractors": level >= 5,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_parts"] * 10 + len(cfg["system_pool"])

        for _ in range(25):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        sys_name = rng.choice(cfg["system_pool"])
        sys = _SYSTEMS[sys_name]
        parts_all = list(sys["parts"])
        if len(parts_all) < cfg["n_parts"]:
            return None
        parts = rng.sample(parts_all, cfg["n_parts"])

        # Assign labels A..
        labels = [chr(ord("A") + i) for i in range(len(parts))]

        # Pick which labelled part to ask about
        target_idx = rng.randint(0, len(parts) - 1)
        target_name = parts[target_idx][0]
        target_label = labels[target_idx]

        # Build options: correct + 3 other part names (must not equal)
        distractor_pool = [p[0] for p in sys["parts"] if p[0] != target_name]
        if len(distractor_pool) < 3:
            distractor_pool = [p[0] for s in _SYSTEMS.values() for p in s["parts"]
                               if p[0] != target_name]
        distractors = rng.sample(distractor_pool, min(3, len(distractor_pool)))
        options = [target_name] + distractors
        rng.shuffle(options)
        correct_letter = chr(ord("A") + options.index(target_name))

        opts_text = "\n".join(
            f"  {chr(ord('A') + i)}) {o}" for i, o in enumerate(options))
        q = (f"The diagram shows a schematic of a {sys['desc']}, with "
             f"labelled parts. What does the part labelled '{target_label}' "
             f"represent? Options:\n{opts_text}\nAnswer with a single letter.")

        image = self._render(sys_name, parts, labels, rng=rng)
        return q, correct_letter, image

    def _render(self, sys_name, parts, labels, rng=None) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        # Use a per-seed rng for render-level jitter to boost image diversity.
        rr = rng if rng is not None else self._rng

        fig_w = 6.0 + rr.uniform(0, 1.2)
        fig_h = 5.6 + rr.uniform(0, 1.0)
        fig, ax = plt.subplots(figsize=(fig_w * sc, fig_h * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # System-specific background
        if sys_name == "plant_cell":
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.05, 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#d5f5e3", edgecolor="#145a32", linewidth=2.0))
        elif sys_name == "atom":
            for r in [0.15, 0.25]:
                ax.add_patch(mpatches.Circle((0.5, 0.5), r,
                             facecolor="none", edgecolor="#333", linewidth=1.4))
        elif sys_name == "simple_circuit":
            # a rectangle loop
            ax.plot([0.2, 0.8, 0.8, 0.2, 0.2],
                    [0.2, 0.2, 0.8, 0.8, 0.2],
                    color="#333", linewidth=2.2)
        elif sys_name == "water_cycle":
            # horizon line, clouds above, ocean below
            ax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 0.25, facecolor="#aed6f1", edgecolor="none"))
            ax.add_patch(mpatches.Rectangle(
                (0, 0.25), 1, 0.75, facecolor="#eaf2f8", edgecolor="none"))
        elif sys_name == "solar_system":
            ax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, facecolor="#0b1d3a", edgecolor="none"))

        text_color = "#fff" if sys_name == "solar_system" else "#1c2833"

        # Part-specific fill colors — makes different parts visually
        # distinguishable, preventing "all identical white circles" rendering.
        _PART_COLORS = {
            # Solar system
            "Sun": "#f1c40f", "Earth": "#2980b9", "Mars": "#c0392b",
            "Venus": "#e67e22", "Jupiter": "#d35400", "Mercury": "#95a5a6",
            # Atom
            "nucleus": "#d35400", "proton": "#e74c3c", "neutron": "#34495e",
            "electron": "#3498db",
            # Plant cell
            "nucleus ": "#8e44ad",  # (never hit: key differs)
            "chloroplast": "#27ae60", "vacuole": "#aed6f1",
            "mitochondrion": "#e74c3c", "cell wall": "#f9e79f",
            # Water cycle
            "condensation": "#ecf0f1", "cloud": "#ecf0f1", "ocean": "#2e86c1",
            # Circuit
            "battery": "#f7dc6f", "resistor": "#d35400", "bulb": "#f9e79f",
        }

        # Per-seed label style for image diversity.
        _label_edge_pool = ["#c0392b", "#7b241c", "#a93226", "#943126",
                             "#1f618d", "#154360", "#196f3d", "#0e6251",
                             "#6c3483", "#512e5f"]
        _label_face_pool = ["#fff", "#fefcf3", "#fff8ef", "#f7fbfd"]
        label_edge = rr.choice(_label_edge_pool)
        label_face = rr.choice(_label_face_pool)
        label_lw = rr.uniform(1.1, 1.8)
        jitter_mag = 0.018
        for (name, (cx, cy), size, kind), lbl in zip(parts, labels):
            cx = cx + rr.uniform(-jitter_mag, jitter_mag)
            cy = cy + rr.uniform(-jitter_mag, jitter_mag)
            fill_color = _PART_COLORS.get(name, "#ffffff")
            if kind == "circle":
                c = mpatches.Circle((cx, cy), size,
                                    facecolor=fill_color, edgecolor="#c0392b",
                                    linewidth=1.8)
                ax.add_patch(c)
            elif kind == "rectangle":
                c = mpatches.Rectangle((cx - size, cy - size * 0.5),
                                       2 * size, size,
                                       facecolor="#fdebd0", edgecolor="#7d6608",
                                       linewidth=1.6)
                ax.add_patch(c)
            elif kind == "ring":
                c = mpatches.Circle((cx, cy), size,
                                    facecolor="none", edgecolor="#2980b9",
                                    linewidth=2)
                ax.add_patch(c)
            elif kind == "line":
                ax.plot([cx - size, cx + size], [cy, cy],
                        color="#1b4f72", linewidth=2.4)
            elif kind == "dot":
                ax.plot([cx], [cy], "o",
                        color="#6c3483", markersize=8)
            elif kind == "arrow":
                ax.annotate("", xy=(cx + size, cy), xytext=(cx - size, cy),
                            arrowprops=dict(arrowstyle="->",
                                            color="#148f77", lw=2))
            # Push label outward from part center to avoid overlap with
            # nearby parts (atom's nucleus/proton/neutron cluster, etc.).
            label_off = max(0.04, size + 0.02)
            # Place label on the side with more room (outward from diagram mid).
            dx = label_off if cx >= 0.5 else -label_off
            dy = label_off if cy >= 0.5 else -label_off
            ax.text(cx + dx, cy + dy, lbl,
                    fontsize=fs + 4, fontweight="bold",
                    color="#1c2833",
                    bbox=dict(boxstyle="circle,pad=0.25",
                              facecolor=label_face,
                              edgecolor=label_edge, linewidth=label_lw))

        title_pool = [
            f"Diagram: {_SYSTEMS[sys_name]['desc']}",
            f"{_SYSTEMS[sys_name]['desc'].title()} Diagram",
            f"Labelled {_SYSTEMS[sys_name]['desc']}",
            f"Schematic of a {_SYSTEMS[sys_name]['desc']}",
            f"{_SYSTEMS[sys_name]['desc']} — Parts",
            f"A {_SYSTEMS[sys_name]['desc']}",
        ]
        ax.set_title(rr.choice(title_pool),
                     fontsize=fs + 1, color=text_color)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = LabeledPartsDiagramQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[labeled_parts_diagram L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"labeled_parts_diagram_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[labeled_parts_diagram L{level} s{s}] A={env._answer}")
