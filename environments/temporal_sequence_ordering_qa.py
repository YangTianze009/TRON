"""
Temporal Sequence Ordering QA (multi-image P5 + L3 temporal reasoning).

4 images (labeled A, B, C, D) show the same scene at different stages of
a process (e.g., blocks stacking, graph being drawn). The images are
presented in shuffled order. Ask: what is the correct chronological order?
MCQ with 4 ordering permutations.

Difficulty axes:
  A) process_type (additive -> mixed -> add+delete+modify)
  B) n_changes_per_step and visual similarity between adjacent frames.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PROCESS_KINDS = ["stacking", "graph_grow", "shape_add"]
_SHAPES = ["circle", "square", "triangle", "hexagon", "diamond"]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e"]

class TemporalSequenceOrderingQA(StandaloneVisualEnv):
    ENV_NAME = "temporal_sequence_ordering"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            process_mode = "additive"
        else:
            # Previously add_delete_modify at L7+, but delete+stacking
            # produces floating blocks (physically unreachable) making the
            # sequence under-determined. Keep it to additive+modify so states
            # remain monotonically reachable.
            process_mode = "additive_modify"
        return {
            "process_mode": process_mode,
            "n_changes_per_step": 1 + level // 3,   # 1..4
            # visual_similarity L0 = frames very different; L9 = similar
            "small_change_magnitude": min(1.0, 0.25 + level * 0.08),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1451)
        self._primary_complexity_feature = level + cfg["n_changes_per_step"]

        kind = sub_rng.choice(_PROCESS_KINDS)
        frames = self._generate_process(kind, cfg, sub_rng)  # list of 4 frames
        if frames is None or len(frames) != 4:
            return None

        # Shuffle frame indices for presentation order (labeled A, B, C, D)
        perm = [0, 1, 2, 3]
        sub_rng.shuffle(perm)
        labeled_frames = [frames[perm[i]] for i in range(4)]  # display[i] = frame at perm[i]
        # Find true ordering in terms of letters
        # perm maps display_idx -> original chronological_idx
        # We want chronological order expressed as list of letters
        # If perm[display] = chrono, then the display whose chrono == k is letters[display_k]
        letters = "ABCD"
        order_letters = ["?"] * 4
        for display_idx, chrono_idx in enumerate(perm):
            order_letters[chrono_idx] = letters[display_idx]
        correct_order = "".join(order_letters)

        # Build 4 MCQ options
        options = [correct_order]
        tries = 0
        while len(options) < 4 and tries < 200:
            tries += 1
            perm2 = list(letters)
            sub_rng.shuffle(perm2)
            cand = "".join(perm2)
            if cand not in options:
                options.append(cand)

        # Shuffle options
        sub_rng.shuffle(options)
        correct_idx = options.index(correct_order)
        answer_letter = chr(ord("A") + correct_idx)

        image = self._render(labeled_frames, kind, options, sub_rng)
        option_lines = "\n".join(f"  ({chr(ord('A') + i)}) {o[0]} -> {o[1]} -> "
                                 f"{o[2]} -> {o[3]}"
                                 for i, o in enumerate(options))
        question = (
            "Four images labeled A, B, C, D are shown. They depict the same "
            "scene at different stages of a process, but are presented in a "
            "shuffled order. Determine the correct chronological order (earliest "
            "to latest).\n" + option_lines +
            "\nAnswer with a single letter."
        )
        return question, answer_letter, image

    def _generate_process(self, kind: str, cfg: Dict,
                          rng: random.Random) -> Optional[List[Dict]]:
        """Return a list of 4 frame states in chronological order."""
        mode = cfg["process_mode"]
        n_changes = cfg["n_changes_per_step"]

        if kind == "stacking":
            # Build a 4-frame stacking sequence.
            # Each frame: list of blocks (col, row, color).
            # We guarantee a monotonically-reachable sequence by only adding
            # blocks on top of existing columns (no floating blocks, no
            # deletions). Color-modify is still allowed at L4+.
            col_count = 4
            state = []
            frames = []
            # Frame 0: 1 block
            first_col = rng.randint(0, col_count - 1)
            state.append({"col": first_col, "row": 0,
                          "color": rng.choice(_COLORS)})
            frames.append([dict(b) for b in state])
            for step in range(3):
                for _ in range(max(1, n_changes)):
                    if mode != "additive" and state and rng.random() < 0.25:
                        # Modify color (doesn't break physics)
                        idx = rng.randint(0, len(state) - 1)
                        state[idx]["color"] = rng.choice(_COLORS)
                    else:
                        # Add a new block strictly atop existing column.
                        col_heights = [0] * col_count
                        for b in state:
                            col_heights[b["col"]] = max(col_heights[b["col"]],
                                                        b["row"] + 1)
                        col = rng.randint(0, col_count - 1)
                        state.append({
                            "col": col,
                            "row": col_heights[col],
                            "color": rng.choice(_COLORS),
                        })
                frames.append([dict(b) for b in state])
            # Validate: every frame must have no floating blocks.
            for f in frames:
                col_set = {}
                for b in f:
                    col_set.setdefault(b["col"], set()).add(b["row"])
                for col, rows in col_set.items():
                    if 0 not in rows:
                        return None
                    # Require contiguous rows starting at 0
                    if max(rows) >= len(rows):
                        return None
            return [{"kind": "stacking", "blocks": f} for f in frames]

        elif kind == "graph_grow":
            # Graph: nodes on a ring, edges add incrementally.
            n_nodes = rng.randint(4, 6)
            node_positions = []
            for i in range(n_nodes):
                a = 2 * math.pi * i / n_nodes
                node_positions.append((math.cos(a) * 1.4, math.sin(a) * 1.4))

            edges = set()
            frames = []
            # Frame 0: a seed edge
            e0 = tuple(sorted(rng.sample(range(n_nodes), 2)))
            edges.add(e0)
            frames.append({"nodes": list(node_positions),
                           "edges": list(edges)})
            for step in range(3):
                for _ in range(max(1, n_changes)):
                    # Strictly additive: always add a new edge (no deletion).
                    for _ in range(10):
                        a, b = sorted(rng.sample(range(n_nodes), 2))
                        if (a, b) not in edges:
                            edges.add((a, b))
                            break
                frames.append({"nodes": list(node_positions),
                               "edges": list(edges)})
            return [{"kind": "graph_grow", **f} for f in frames]

        else:  # shape_add
            # Canvas with 10x10 coords, add/modify shapes across 4 frames.
            shapes_state = []
            frames = []
            # Seed initial shape
            shapes_state.append({
                "shape": rng.choice(_SHAPES),
                "color": rng.choice(_COLORS),
                "x": rng.uniform(1.5, 8.5),
                "y": rng.uniform(1.5, 8.5),
                "size": rng.uniform(0.5, 0.8),
            })
            frames.append([dict(s) for s in shapes_state])
            for step in range(3):
                for _ in range(max(1, n_changes)):
                    if mode != "additive" and shapes_state and rng.random() < 0.25:
                        # Modify existing shape color (still monotonic add +
                        # reversible color change).
                        i = rng.randint(0, len(shapes_state) - 1)
                        shapes_state[i]["color"] = rng.choice(_COLORS)
                    else:
                        # Strictly add a new shape (no deletion).
                        shapes_state.append({
                            "shape": rng.choice(_SHAPES),
                            "color": rng.choice(_COLORS),
                            "x": rng.uniform(1.5, 8.5),
                            "y": rng.uniform(1.5, 8.5),
                            "size": rng.uniform(0.4, 0.8),
                        })
                frames.append([dict(s) for s in shapes_state])
            return [{"kind": "shape_add", "shapes": f} for f in frames]

    def _render_frame(self, ax, frame: Dict):
        ax.set_aspect("equal")
        ax.axis("off")
        kind = frame["kind"]
        if kind == "stacking":
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-0.5, 5.0)
            for b in frame["blocks"]:
                rect = mpatches.Rectangle((b["col"], b["row"]), 1, 1,
                                          fc=b["color"], ec="black", lw=1.0,
                                          alpha=0.9)
                ax.add_patch(rect)
            # ground line
            ax.plot([-0.3, 4.3], [0, 0], color="#2c3e50", lw=1.5)
        elif kind == "graph_grow":
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            for i, (x, y) in enumerate(frame["nodes"]):
                ax.add_patch(plt.Circle((x, y), 0.22, fc="#ecf0f1",
                                        ec="#2c3e50", lw=1.3))
                ax.text(x, y, str(i + 1), ha="center", va="center",
                        fontsize=9, fontweight="bold")
            for (a, b) in frame["edges"]:
                xa, ya = frame["nodes"][a]
                xb, yb = frame["nodes"][b]
                ax.plot([xa, xb], [ya, yb], color="#34495e", lw=1.6,
                        zorder=0.5)
        else:  # shape_add
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            for s in frame["shapes"]:
                self._draw_plain_shape(ax, s)

    def _draw_plain_shape(self, ax, s: Dict):
        shape = s["shape"]
        x, y, size = s["x"], s["y"], s["size"]
        color = s["color"]
        if shape == "circle":
            ax.add_patch(plt.Circle((x, y), size, fc=color, ec="black", lw=1.0,
                                    alpha=0.9))
        elif shape == "square":
            ax.add_patch(mpatches.Rectangle((x - size, y - size),
                                            2 * size, 2 * size,
                                            fc=color, ec="black", lw=1.0,
                                            alpha=0.9))
        elif shape == "triangle":
            ax.add_patch(RegularPolygon((x, y), 3, radius=size * 1.1,
                                        orientation=math.pi / 2,
                                        fc=color, ec="black", lw=1.0,
                                        alpha=0.9))
        elif shape == "hexagon":
            ax.add_patch(RegularPolygon((x, y), 6, radius=size * 1.1,
                                        fc=color, ec="black", lw=1.0,
                                        alpha=0.9))
        elif shape == "diamond":
            pts = [(x, y + size), (x + size * 0.7, y),
                   (x, y - size), (x - size * 0.7, y)]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=1.0,
                                     alpha=0.9))

    def _render(self, labeled_frames: List[Dict], kind: str,
                options: List[str], rng: random.Random) -> Image.Image:
        style = self._random_style()
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        fig.patch.set_facecolor(style["bg_color"])
        letters = "ABCD"
        for i, (ax, frame) in enumerate(zip(axes, labeled_frames)):
            self._render_frame(ax, frame)
            ax.set_title(f"({letters[i]})", fontsize=13, fontweight="bold",
                         pad=4)

        fig.suptitle("Temporal Sequence Ordering",
                     fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = TemporalSequenceOrderingQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
