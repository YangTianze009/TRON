"""
Object Tracking Across Frames QA (multi-image P5 + L3 motion tracking).

3 side-by-side "frames" showing a scene with multiple objects. Objects
move between frames. One object is circled in Frame 1. Ask: where is that
object in Frame 3? MCQ with 4 positions marked in Frame 3.

Difficulty axes:
  A) n_objects (3..7)
  B) movement_complexity (1 moves -> all move + swaps, optional slight
     color change of tracked object at level >= 5).
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

_SHAPES = ["circle", "square", "triangle", "hexagon", "diamond", "pentagon"]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e"]

def _color_tweak(hex_color: str) -> str:
    """Return a slightly perturbed version of a hex color."""
    from matplotlib.colors import to_rgba, rgb2hex
    r, g, b, _ = to_rgba(hex_color)
    r = max(0, min(1, r + (random.random() - 0.5) * 0.15))
    g = max(0, min(1, g + (random.random() - 0.5) * 0.15))
    b = max(0, min(1, b + (random.random() - 0.5) * 0.15))
    return rgb2hex((r, g, b))

class ObjectTrackingAcrossFramesQA(StandaloneVisualEnv):
    ENV_NAME = "object_tracking_across_frames"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_objects": 3 + level // 2,            # 3..7
            "n_moving": 1 if level == 0 else
                        2 if level <= 2 else
                        3 if level <= 4 else
                        5 if level <= 7 else 99,    # 99 => all move
            "allow_swap": level >= 5,
            "appearance_change": level >= 5,
            "similar_colors": level >= 4,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1453)
        self._primary_complexity_feature = cfg["n_objects"] * 2 + level

        n_obj = cfg["n_objects"]

        # Build initial layout: distinct colors if possible
        if cfg["similar_colors"]:
            base_color = sub_rng.choice(_COLORS)
            colors = [base_color] * n_obj  # same color base
        else:
            colors = sub_rng.sample(_COLORS, min(n_obj, len(_COLORS)))
            while len(colors) < n_obj:
                colors.append(sub_rng.choice(_COLORS))

        # Frame 1 positions: distinct grid slots
        grid_x = [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 1.5]
        grid_y = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.5]
        slots = list(zip(grid_x, grid_y))
        if n_obj > len(slots):
            return None
        chosen_slots = slots[:n_obj]
        # Shuffle slot order slightly
        sub_rng.shuffle(chosen_slots)

        objects = []
        for i in range(n_obj):
            objects.append({
                "id": i,
                "shape": sub_rng.choice(_SHAPES),
                "color": colors[i],
                "x": chosen_slots[i][0] + sub_rng.uniform(-0.2, 0.2),
                "y": chosen_slots[i][1] + sub_rng.uniform(-0.2, 0.2),
                "size": sub_rng.uniform(0.3, 0.4),
            })

        # Frame 2 / Frame 3: apply movements.
        frames = [objects]
        n_moving = cfg["n_moving"]
        if n_moving >= 99:
            n_moving = n_obj
        n_moving = min(n_moving, n_obj)

        for f_idx in range(2):
            prev = frames[-1]
            new_state = [dict(o) for o in prev]

            # Optionally swap two objects
            if cfg["allow_swap"] and n_obj >= 2 and sub_rng.random() < 0.5:
                i, j = sub_rng.sample(range(n_obj), 2)
                new_state[i]["x"], new_state[j]["x"] = new_state[j]["x"], new_state[i]["x"]
                new_state[i]["y"], new_state[j]["y"] = new_state[j]["y"], new_state[i]["y"]

            # Move a subset
            moving_ids = sub_rng.sample(range(n_obj), n_moving)
            for idx in moving_ids:
                new_state[idx]["x"] = max(0.8, min(9.2,
                                                    new_state[idx]["x"]
                                                    + sub_rng.uniform(-1.5, 1.5)))
                new_state[idx]["y"] = max(0.8, min(5.5,
                                                    new_state[idx]["y"]
                                                    + sub_rng.uniform(-1.2, 1.2)))
            frames.append(new_state)

        # Pick tracked object (must exist in all 3 frames with consistent id)
        tracked_id = sub_rng.randint(0, n_obj - 1)

        # Apply appearance change in later frames if enabled
        if cfg["appearance_change"]:
            for f_idx in (1, 2):
                # tweak color slightly, keep shape consistent for tracking
                frames[f_idx][tracked_id]["color"] = _color_tweak(
                    frames[f_idx][tracked_id]["color"]
                )
                frames[f_idx][tracked_id]["size"] *= sub_rng.uniform(0.9, 1.1)

        # Frame 3 candidate positions: 4 positions include the tracked + 3 distractors
        final_frame = frames[2]
        tracked_obj = final_frame[tracked_id]

        # Pick 3 distractor positions (other objects in frame 3)
        other_ids = [i for i in range(n_obj) if i != tracked_id]
        sub_rng.shuffle(other_ids)
        distractor_ids = other_ids[:3]
        if len(distractor_ids) < 3:
            # Add synthetic marker positions if not enough objects
            while len(distractor_ids) < 3:
                distractor_ids.append(tracked_id)  # fallback (will dedupe)

        option_positions: List[Tuple[float, float]] = [
            (tracked_obj["x"], tracked_obj["y"])
        ]
        used_obj_ids = [tracked_id]
        # Enforce a minimum separation between option boxes so labels/boxes
        # do not visually overlap. If a distractor is too close to an existing
        # option, skip it; we'll fall back to synthetic placement below.
        MIN_OPT_SEP = 1.5
        for did in distractor_ids:
            if did in used_obj_ids:
                continue
            cand = (final_frame[did]["x"], final_frame[did]["y"])
            if all(math.hypot(cand[0] - p[0], cand[1] - p[1]) >= MIN_OPT_SEP
                   for p in option_positions):
                option_positions.append(cand)
                used_obj_ids.append(did)

        # If fewer than 4, add synthetic boxes that don't overlap existing
        # option boxes or any frame-3 object. Each option is drawn as a 1.1
        # square with a label above, so enforce >=1.5 separation.
        min_sep = 1.5
        all_obj_pts = [(o["x"], o["y"]) for o in final_frame]
        attempts = 0
        while len(option_positions) < 4 and attempts < 120:
            attempts += 1
            cand = (sub_rng.uniform(1.0, 9.0), sub_rng.uniform(1.0, 5.2))
            def _far_enough(pt, others, sep):
                return all(math.hypot(pt[0] - o[0], pt[1] - o[1]) >= sep
                           for o in others)
            if (_far_enough(cand, option_positions, min_sep)
                    and _far_enough(cand, all_obj_pts, min_sep * 0.7)):
                option_positions.append(cand)
        while len(option_positions) < 4:
            option_positions.append((tracked_obj["x"] + sub_rng.uniform(-2, 2),
                                     tracked_obj["y"] + sub_rng.uniform(-1.5, 1.5)))

        # Shuffle; correct letter is where tracked pos is
        letters = ["A", "B", "C", "D"]
        indexed = list(enumerate(option_positions))
        sub_rng.shuffle(indexed)
        new_positions = [p for _, p in indexed]
        correct_orig = 0  # tracked was first
        correct_idx = next(i for i, (orig, _) in enumerate(indexed) if orig == correct_orig)
        answer_letter = letters[correct_idx]

        image = self._render(frames, tracked_id, new_positions, sub_rng)
        templates = [
            "Three frames of a scene are shown side by side (Frame 1, Frame 2, "
            "Frame 3). Objects move between frames. The object circled in red "
            "in Frame 1 needs to be tracked. Four candidate positions in Frame "
            "3 are labeled A, B, C, D with boxes. At which position is the "
            "tracked object in Frame 3? Answer with a single letter.",
            "The image presents three sequential frames (Frame 1, Frame 2, "
            "Frame 3). The target object marked with a red circle in Frame 1 "
            "moves between frames. Four labelled boxes (A-D) mark candidate "
            "positions in Frame 3. Which letter corresponds to the tracked "
            "object's final position? Single letter answer.",
            "Across three frames shown side by side, objects shift positions. "
            "Track the object circled in red in Frame 1 and identify which of "
            "the four labelled boxes (A, B, C, D) in Frame 3 contains it. "
            "Reply with a single letter.",
            "Inspect the three frames. The red-circled object in Frame 1 must "
            "be located in Frame 3 among four candidate boxes (A-D). Provide "
            "the correct letter only.",
        ]
        question = sub_rng.choice(templates)
        return question, answer_letter, image

    def _draw_object(self, ax, obj: Dict):
        cx, cy, size = obj["x"], obj["y"], obj["size"]
        shape = obj["shape"]
        color = obj["color"]
        if shape == "circle":
            ax.add_patch(plt.Circle((cx, cy), size, fc=color, ec="black",
                                    lw=1.0, alpha=0.9))
        elif shape == "square":
            ax.add_patch(mpatches.Rectangle((cx - size, cy - size),
                                            2 * size, 2 * size,
                                            fc=color, ec="black", lw=1.0,
                                            alpha=0.9))
        elif shape == "triangle":
            ax.add_patch(RegularPolygon((cx, cy), 3, radius=size * 1.15,
                                        orientation=math.pi / 2,
                                        fc=color, ec="black", lw=1.0,
                                        alpha=0.9))
        elif shape == "pentagon":
            ax.add_patch(RegularPolygon((cx, cy), 5, radius=size * 1.15,
                                        orientation=math.pi / 2,
                                        fc=color, ec="black", lw=1.0,
                                        alpha=0.9))
        elif shape == "hexagon":
            ax.add_patch(RegularPolygon((cx, cy), 6, radius=size * 1.15,
                                        fc=color, ec="black", lw=1.0,
                                        alpha=0.9))
        elif shape == "diamond":
            pts = [(cx, cy + size), (cx + size * 0.7, cy),
                   (cx, cy - size), (cx - size * 0.7, cy)]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=1.0,
                                     alpha=0.9))

    def _render(self, frames: List[List[Dict]], tracked_id: int,
                option_positions: List[Tuple[float, float]],
                rng: random.Random) -> Image.Image:
        style = self._random_style()
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        fig.patch.set_facecolor(style["bg_color"])
        letters = ["A", "B", "C", "D"]
        for i, (ax, frame) in enumerate(zip(axes, frames)):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.set_aspect("equal")
            ax.axis("off")
            # Frame border
            ax.add_patch(mpatches.Rectangle((0.1, 0.1), 9.8, 5.8,
                                            fc="#f8f9fa", ec="#7f8c8d",
                                            lw=1.2, zorder=0))
            for obj in frame:
                self._draw_object(ax, obj)
            ax.set_title(f"Frame {i + 1}", fontsize=12, fontweight="bold",
                         pad=4)

        # Circle tracked object in Frame 1
        t = frames[0][tracked_id]
        axes[0].add_patch(plt.Circle((t["x"], t["y"]), t["size"] + 0.25,
                                     fc="none", ec="#e74c3c", lw=2.5,
                                     zorder=10))
        axes[0].annotate("target", xy=(t["x"], t["y"] + t["size"] + 0.35),
                         ha="center", va="bottom", fontsize=10,
                         color="#e74c3c", fontweight="bold")

        # Label 4 option positions in Frame 3
        for (x, y), letter in zip(option_positions, letters):
            axes[2].add_patch(mpatches.Rectangle(
                (x - 0.55, y - 0.55), 1.1, 1.1,
                fc="none", ec="#2c3e50", lw=1.6, zorder=9))
            axes[2].text(x, y + 0.72, letter, ha="center", va="bottom",
                         fontsize=12, fontweight="bold", color="#2c3e50",
                         bbox=dict(boxstyle="round,pad=0.2", fc="#f9e79f",
                                   ec="#2c3e50", lw=1.0), zorder=10)

        fig.suptitle("Object Tracking Across Frames",
                     fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.05,
                            wspace=0.1)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = ObjectTrackingAcrossFramesQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
