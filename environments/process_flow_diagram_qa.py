"""
Process Flow Diagram QA (batch 3, 2026-04-14).

Target: diagram lifeCycles / process flowcharts / X1 multi-step rule
execution. A process flowchart is drawn with labelled stages (rectangles)
and arrows between them. Question asks: "What comes after stage N?" or
"Which stage is directly before X?".

Format: constant short-answer (stage name string).

Difficulty axes:
  A) Pattern A: n_stages (3..7) grows with level.
  B) Pattern E: chain depth — "what comes two steps after X"
     (follow-up at L>=4).
  C) Pattern C: branching at L>=6 (two outgoing edges, conditional label).
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

_STAGE_NAMES = [
    "egg", "larva", "pupa", "adult", "juvenile",
    "seed", "seedling", "sapling", "tree", "flower",
    "cloud", "rain", "river", "ocean", "steam",
    "Heat", "Melt", "Cool", "Solid", "Liquid",
    "input", "parse", "analyze", "store", "output",
]

class ProcessFlowDiagramQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "process_flow_diagram"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_stages": 4 + level // 2,               # 4..8 (was 3..7)
            "hops": 1 + level // 3,                   # 1..4 (was 1..2)
            "branching": level >= 4,                   # was >= 6
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_stages"] * 10 + cfg["hops"]

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_stages"]
        stages = rng.sample(_STAGE_NAMES, n)

        # Build adjacency: simple linear chain
        # Optionally with a branch out of one node
        next_map: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n - 1):
            next_map[i].append(i + 1)

        branch_cond = None
        if cfg["branching"] and n >= 5:
            # Pick branching node idx
            bi = rng.randint(1, n - 3)
            target_alt = rng.randint(bi + 2, n - 1)
            if target_alt != bi + 1:
                next_map[bi].append(target_alt)
                branch_cond = (bi, "if cold", target_alt)

        # Question: pick a stage and ask for next stage after 'hops' steps
        start_idx = rng.randint(0, n - 1 - cfg["hops"])
        cur = start_idx
        for _ in range(cfg["hops"]):
            nxts = next_map[cur]
            if not nxts:
                return None
            cur = nxts[0]  # main chain
        answer = stages[cur]

        branch_note = (
            " Ignore any conditional branch arrows (labelled with 'if …'); "
            "follow only the main (unlabelled) sequence of arrows."
            if branch_cond is not None else ""
        )
        if cfg["hops"] == 1:
            templates_1 = [
                f"The image shows a process flow diagram with labelled stages connected by arrows. What is the next stage after '{stages[start_idx]}' on the main path?{branch_note} Answer with the exact stage name as written in the diagram.",
                f"In the process flow diagram, which stage immediately follows '{stages[start_idx]}' on the main path?{branch_note} Provide the exact stage name shown.",
                f"Look at the flowchart. Which stage does the main arrow lead to from '{stages[start_idx]}'?{branch_note} Reply with the exact stage label.",
                f"Following the main arrows in the flow diagram, identify the stage directly after '{stages[start_idx]}'.{branch_note} Answer with the exact label text.",
            ]
            q = rng.choice(templates_1)
        else:
            templates_n = [
                f"The image shows a process flow diagram with labelled stages connected by arrows. Following the main (unlabelled) arrows forward from '{stages[start_idx]}', what stage do you reach after {cfg['hops']} steps?{branch_note} Answer with the exact stage name as written in the diagram.",
                f"Starting from '{stages[start_idx]}' and following the main (unlabelled) arrows of the flow diagram for {cfg['hops']} steps, which stage do you arrive at?{branch_note} Provide the exact label.",
                f"From '{stages[start_idx]}', traverse the main (unlabelled) arrows of the flowchart {cfg['hops']} steps forward. What is the destination stage label (exact text)?{branch_note}",
                f"Trace {cfg['hops']} main (unlabelled) arrows beginning at '{stages[start_idx]}' in the flow diagram. Name the resulting stage (exact label).{branch_note}",
            ]
            q = rng.choice(templates_n)
        image = self._render(stages, next_map, branch_cond, rng)
        return q, answer, image

    def _render(self, stages, next_map, branch_cond, rng=None) -> Image.Image:
        if rng is None:
            rng = random.Random(0)
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = list(style["palette"])
        rng.shuffle(palette)

        n = len(stages)
        # Vary node shape and orientation
        orientation = rng.choice(["horizontal", "horizontal", "wave"])
        node_shape = rng.choice(["round", "round", "square"])

        fig_w = max(6.5, 1.8 * n)
        fig, ax = plt.subplots(figsize=(fig_w * sc, 4.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        # Position nodes in a row (or sinusoidal wave)
        positions = []
        box_w, box_h = 1.2, 0.7
        wave_amp = rng.uniform(0.0, 0.4) if orientation == "wave" else 0.0
        for i in range(n):
            cx = (i + 0.5) * (box_w + 0.6)
            cy = 0.5 + wave_amp * math.sin(i * math.pi / max(2, n - 1))
            positions.append((cx, cy))
            if node_shape == "square":
                ax.add_patch(mpatches.Rectangle(
                    (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                    facecolor=palette[i % len(palette)],
                    edgecolor="#000", linewidth=1.3))
            else:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                    boxstyle="round,pad=0.04",
                    facecolor=palette[i % len(palette)],
                    edgecolor="#000", linewidth=1.3))
            ax.text(cx, cy, stages[i],
                    fontsize=fs + 1, fontweight="bold",
                    ha="center", va="center", color="#fff")

        # Draw arrows (color picked per render)
        arrow_color = rng.choice(["#333", "#1f4e79", "#3a3a3a", "#22272e"])
        for src, nxts in next_map.items():
            sx, sy = positions[src]
            for tgt in nxts:
                tx, ty = positions[tgt]
                ax.annotate("", xy=(tx - box_w / 2 - 0.05, ty),
                            xytext=(sx + box_w / 2 + 0.05, sy),
                            arrowprops=dict(arrowstyle="->", lw=1.6,
                                            color=arrow_color,
                                            connectionstyle="arc3,rad=0.0"))

        # Optional branch
        if branch_cond is not None:
            bi, cond, alt = branch_cond
            bx, by = positions[bi]
            ax_x = positions[alt][0]
            ax_y = positions[alt][1]
            ax.annotate("", xy=(ax_x - box_w / 2 - 0.05, ax_y - 0.5),
                        xytext=(bx + box_w / 2 - 0.1, by - 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.4,
                                        color="#c0392b",
                                        connectionstyle="arc3,rad=-0.3"))
            ax.text((bx + ax_x) / 2, by - 0.9, cond,
                    fontsize=fs - 1, color="#c0392b", ha="center")

        ax.set_xlim(0, n * (box_w + 0.6) + 0.5)
        ax.set_ylim(-1.2, 1.8)
        title_pool = [
            "Process flow", "Process Flow Diagram", "Stage Diagram",
            "Flow Chart", "Sequential Process",
        ]
        ax.set_title(rng.choice(title_pool), fontsize=fs + 1)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = ProcessFlowDiagramQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[process_flow_diagram L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"process_flow_diagram_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[process_flow_diagram L{level} s{s}] A={env._answer}")
