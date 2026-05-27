"""
Shape-merge sequence completion QA.

A 1-row strip of 3-4 frames shows a "moving" shape (e.g., pentagon) that
translates one step per frame toward a stationary "container" shape (e.g.,
rectangle). On the final step it merges into / overlaps the container. A
secondary rule (a small dot toggles on/off) runs in parallel. The model
picks the next frame from 4 candidates A-D.

Difficulty levels 0-9:
  L0-L1: 3 frames, only translation rule, dot rule disabled.
  L2-L3: 3 frames + dot toggle.
  L4-L5: 4 frames + dot toggle + bigger movement step.
  L6-L7: 4-5 frames + dot toggle + variable container shape.
  L8-L9: 5 frames + harder distractors (off-by-one merge step).
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


_MOVERS = ["pentagon", "triangle", "diamond", "star", "hexagon"]
_CONTAINERS = ["square", "circle"]

_QUESTION_TEMPLATES = [
    "The sequence shows a moving shape gradually approaching and merging into a stationary shape. Which option (A, B, C, or D) comes next? Answer with one letter.",
    "Examine the sequence: one shape translates each step toward another. Which of A, B, C, D is the next frame?",
    "Identify the rule: the moving shape translates and eventually overlaps the container. Pick the next frame: A, B, C, or D.",
    "Each frame the moving shape steps closer to the stationary shape; the dot may toggle. Which option (A-D) continues the sequence?",
    "Determine the rule and select the next frame from A, B, C, D. Reply with one letter.",
    "Look at the sequence carefully. Which of the four options (A, B, C, D) should be the next frame?",
    "After working out the rule (translation toward container + optional dot toggle), pick A, B, C, or D as the next frame.",
    "The sequence ends with one frame missing. Which candidate completes it? A, B, C, or D — single letter answer.",
    "Pick the option that matches the rule's next state: (A) (B) (C) (D). Single letter only.",
    "Each step applies the same rule. Choose the option (A, B, C, D) that should come after the last shown frame.",
    "Inspect the strip of frames. The next frame is one of A, B, C, D — which letter?",
    "Determine which option among A-D continues the merging-shape pattern. Provide one letter.",
    "Study the sequence and the four candidates below. Which letter (A, B, C, D) is the correct next frame?",
    "Choose the next frame in the sequence: A / B / C / D.",
    "What comes next in the sequence? Pick from A, B, C, D and answer with a single letter.",
    "The sequence follows a translate-and-merge rule. Pick the next frame: A, B, C, or D.",
]


def _draw_shape(ax, shape: str, cx: float, cy: float, size: float,
                color: str, edge: str = "#222", linewidth: float = 1.4,
                fill: bool = True, zorder: int = 3):
    fc = color if fill else "none"
    if shape == "circle":
        ax.add_patch(mpatches.Circle((cx, cy), size, facecolor=fc,
                                     edgecolor=edge, linewidth=linewidth,
                                     zorder=zorder))
    elif shape == "square":
        ax.add_patch(mpatches.Rectangle(
            (cx - size, cy - size), 2 * size, 2 * size, facecolor=fc,
            edgecolor=edge, linewidth=linewidth, zorder=zorder))
    elif shape == "triangle":
        verts = [(cx, cy + size), (cx - size, cy - size * 0.85),
                 (cx + size, cy - size * 0.85)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "diamond":
        verts = [(cx, cy + size), (cx + size, cy),
                 (cx, cy - size), (cx - size, cy)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "pentagon":
        verts = [(cx + size * math.cos(math.radians(72 * i + 90)),
                  cy + size * math.sin(math.radians(72 * i + 90)))
                 for i in range(5)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "hexagon":
        verts = [(cx + size * math.cos(math.radians(60 * i + 30)),
                  cy + size * math.sin(math.radians(60 * i + 30)))
                 for i in range(6)]
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))
    elif shape == "star":
        verts = []
        for i in range(10):
            a = math.pi / 2 + 2 * math.pi * i / 10
            r = size if i % 2 == 0 else size * 0.45
            verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        ax.add_patch(mpatches.Polygon(verts, facecolor=fc, edgecolor=edge,
                                      linewidth=linewidth, zorder=zorder))


class ShapeMergeSequenceQA(StandaloneVisualEnv):
    ENV_NAME = "shape_merge_sequence"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — reduce to 2
            # options (binary choice) so model has a real signal even with
            # weak visual reasoning.
            return {"n_frames": 3, "use_dot": False, "step": 0.45,
                    "n_options": 2}
        if level <= 3:
            return {"n_frames": 3, "use_dot": True, "step": 0.45}
        if level <= 5:
            return {"n_frames": 4, "use_dot": True, "step": 0.5}
        if level <= 7:
            return {"n_frames": 4, "use_dot": True, "step": 0.55}
        return {"n_frames": 5, "use_dot": True, "step": 0.6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1502)

        for _ in range(20):
            result = self._try_generate(sub_rng, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        mover_shape = rng.choice(_MOVERS)
        container_shape = rng.choice(_CONTAINERS)
        n_frames = cfg["n_frames"]
        step = cfg["step"]
        use_dot = cfg["use_dot"]

        # Container is at fixed position (right side). Mover starts to the
        # left and steps right each frame, with merge happening on the next
        # frame (the answer). We compute positions so that:
        #   - frame 0 mover is at x ≈ 0.15
        #   - the answer (next) frame has mover at container_x (merged)
        #   - intermediate frames evenly distribute toward container
        container_x = 0.70
        start_x = 0.15
        # We want n_frames+1 positions total (frames 0..n_frames-1 displayed
        # + answer at index n_frames). The answer is at container_x.
        # Linearly interpolate so that each step is uniform.
        total_steps = n_frames  # gaps: from start to merged
        step_size = (container_x - start_x) / total_steps
        mover_xs = [start_x + i * step_size for i in range(n_frames)]
        next_mover_x = container_x

        # Dot toggle rule: starts at frame 0 = ON, alternates.
        # If use_dot=False, dot is always off.
        if use_dot:
            dot_states = [(i % 2 == 0) for i in range(n_frames + 1)]
        else:
            dot_states = [False] * (n_frames + 1)

        next_dot = dot_states[n_frames]

        # Generate 4 candidates: 1 correct + 3 distractors.
        # Distractor strategies:
        #   - wrong mover_x (still partially merged or stopped early)
        #   - flipped dot state
        #   - wrong container shape
        #   - mover at container_x but dot wrong
        candidates = []
        candidates.append({
            "mover_x": next_mover_x, "dot_on": next_dot,
            "container": container_shape,
            "mover_shape": mover_shape, "is_correct": True,
        })

        distractor_pool = []
        # Stopped early (no merge): use the last displayed frame's position
        # (one step short of merge).
        distractor_pool.append({
            "mover_x": mover_xs[-1], "dot_on": next_dot,
            "container": container_shape,
            "mover_shape": mover_shape, "is_correct": False,
        })
        # Flipped dot
        distractor_pool.append({
            "mover_x": next_mover_x, "dot_on": not next_dot,
            "container": container_shape,
            "mover_shape": mover_shape, "is_correct": False,
        })
        # Wrong container shape (only if there's a different container)
        other_container = "square" if container_shape == "circle" else "circle"
        distractor_pool.append({
            "mover_x": next_mover_x, "dot_on": next_dot,
            "container": other_container,
            "mover_shape": mover_shape, "is_correct": False,
        })
        # Mover gone past container (off-by-one over-step)
        distractor_pool.append({
            "mover_x": min(0.85, container_x + 0.10), "dot_on": next_dot,
            "container": container_shape,
            "mover_shape": mover_shape, "is_correct": False,
        })
        # Wrong mover shape (subtle distractor)
        other_mover = rng.choice([m for m in _MOVERS if m != mover_shape])
        distractor_pool.append({
            "mover_x": next_mover_x, "dot_on": next_dot,
            "container": container_shape,
            "mover_shape": other_mover, "is_correct": False,
        })

        # If dot rule disabled, drop the "flipped dot" distractor (it's same as correct)
        if not use_dot:
            distractor_pool = [d for d in distractor_pool if d["dot_on"] == next_dot]

        rng.shuffle(distractor_pool)

        # Filter distinctness from correct
        # 2026-05-04: simplified L0/L1 (was 7.5% too-hard) — n_options=2 at L0/L1.
        n_options = cfg.get("n_options", 4)
        n_distractors_needed = n_options - 1
        chosen_distractors = []
        for d in distractor_pool:
            if any(self._cells_equal(d, c) for c in candidates + chosen_distractors):
                continue
            chosen_distractors.append(d)
            if len(chosen_distractors) >= n_distractors_needed:
                break
        if len(chosen_distractors) < n_distractors_needed:
            return None

        candidates.extend(chosen_distractors)
        rng.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates) if c["is_correct"])
        answer_letter = chr(ord("A") + correct_idx)

        img = self._render(mover_shape, container_shape, mover_xs,
                           dot_states[:n_frames], candidates,
                           container_x, rng)
        sidx = (self.seed or 0) % len(_QUESTION_TEMPLATES)
        question = _QUESTION_TEMPLATES[sidx]
        if n_options == 2:
            # Override question for binary choice
            question = (
                "The sequence shows a moving shape stepping toward a "
                "stationary container shape. The next frame shows the moving "
                "shape MERGING/OVERLAPPING the container. Which option "
                "(A or B) is the correct next frame? Reply with one letter."
            )
        return question, answer_letter, img

    @staticmethod
    def _cells_equal(a: Dict, b: Dict) -> bool:
        return (
            abs(a["mover_x"] - b["mover_x"]) < 0.02
            and a["dot_on"] == b["dot_on"]
            and a["container"] == b["container"]
            and a["mover_shape"] == b["mover_shape"]
        )

    def _render(self, mover_shape: str, container_shape: str,
                mover_xs: List[float], dot_states: List[bool],
                candidates: List[Dict], container_x: float, rng):
        style = self._random_style()
        n_frames = len(mover_xs)
        n_cands = len(candidates)
        # Top row: n_frames cells + ? cell. Bottom row: candidate cells.
        n_top = n_frames + 1
        fig_w = max(8.0, 1.4 * n_top)
        fig_h = 6.0
        fig, (ax_seq, ax_opts) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1.0, 1.0]})
        fig.patch.set_facecolor(style["bg_color"])

        ax_seq.set_facecolor(style["bg_color"])
        ax_seq.set_xlim(0, n_top)
        ax_seq.set_ylim(0, 1)
        ax_seq.set_aspect("equal")
        ax_seq.axis("off")
        ax_seq.set_title("Sequence", fontsize=12, fontweight="bold")

        mover_color = style["palette"][2]
        container_color = "#ffffff"
        container_edge = style["palette"][1]

        for i in range(n_frames):
            cx = i + 0.5
            cy = 0.5
            ax_seq.add_patch(mpatches.Rectangle(
                (i, 0), 1, 1, facecolor="#ffffff", edgecolor="#222",
                linewidth=1.2, zorder=1))
            # Container
            cont_cx = cx - 0.5 + container_x
            _draw_shape(ax_seq, container_shape, cont_cx, cy, 0.18,
                        container_color, edge=container_edge,
                        linewidth=1.4, zorder=2)
            # Mover
            mover_cx = cx - 0.5 + mover_xs[i]
            _draw_shape(ax_seq, mover_shape, mover_cx, cy, 0.10,
                        mover_color, edge="#222", zorder=3)
            # Dot in upper-left
            if dot_states[i]:
                ax_seq.add_patch(mpatches.Circle(
                    (i + 0.13, 0.85), 0.04, facecolor="#000",
                    edgecolor="#000", zorder=4))

        # Question mark cell
        qi = n_frames
        ax_seq.add_patch(mpatches.Rectangle(
            (qi, 0), 1, 1, facecolor="#fff8f0", edgecolor="#b00",
            linewidth=2.0, linestyle="--", zorder=1))
        ax_seq.text(qi + 0.5, 0.5, "?", fontsize=36, fontweight="bold",
                    color="#b00", ha="center", va="center", zorder=5)

        # Options row
        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, n_cands)
        ax_opts.set_ylim(0, 1.3)
        ax_opts.set_aspect("equal")
        ax_opts.axis("off")
        ax_opts.set_title("Options", fontsize=11, fontweight="bold")

        for i, cand in enumerate(candidates):
            cx = i + 0.5
            cy = 0.5
            ax_opts.add_patch(mpatches.Rectangle(
                (i + 0.05, 0.05), 0.9, 0.9, facecolor="#ffffff",
                edgecolor="#222", linewidth=1.2, zorder=1))
            cont_cx = cx - 0.5 + container_x
            _draw_shape(ax_opts, cand["container"], cont_cx, cy, 0.18,
                        container_color, edge=container_edge,
                        linewidth=1.4, zorder=2)
            mover_cx = cx - 0.5 + cand["mover_x"]
            _draw_shape(ax_opts, cand["mover_shape"], mover_cx, cy, 0.10,
                        mover_color, edge="#222", zorder=3)
            if cand["dot_on"]:
                ax_opts.add_patch(mpatches.Circle(
                    (i + 0.18, 0.85), 0.04, facecolor="#000",
                    edgecolor="#000", zorder=4))
            label = chr(ord("A") + i)
            ax_opts.text(i + 0.5, 1.05, label, fontsize=14, fontweight="bold",
                          ha="center", va="center", color="#222")

        return self.fig_to_pil(fig, dpi=style["dpi"])
