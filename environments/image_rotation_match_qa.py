"""
Image Rotation Match QA (multi-image P5 + S2.1 2D rotation).

Two side-by-side images (A and B). Image B is a rotated (or rotated+reflected,
or unrelated) version of Image A. MCQ asks the transformation.

Difficulty axes:
  A) pattern_complexity (grid size or number of shapes in the pattern).
  B) rotation_granularity (L0: 90/180 only; L9: every 30 deg).
  C) include_reflection at level >= 3.
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

_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#34495e", "#f1c40f", "#e91e63"]

class ImageRotationMatchQA(StandaloneVisualEnv):
    ENV_NAME = "image_rotation_match"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 deep-redesign: was 20/10/0/30 — model can't handle
        # rotation matching with random complexity-3 patterns at L0. Simplify
        # L0 to use complexity-2 (very few cells) + only 90/180 rotations
        # + clear distractors.
        level = max(0, min(level, 9))
        if level == 0:
            angles = [90, 180]
            pattern_complexity = 2  # was 3
        elif level <= 3:
            angles = [90, 180, 270]
            pattern_complexity = 3
        elif level <= 6:
            angles = [45, 90, 135, 180, 225, 270, 315]
            pattern_complexity = 4 + (level - 4) // 2
        else:
            angles = [30, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270,
                      300, 315, 330]
            pattern_complexity = 5 + (level - 7) // 2
        return {
            "pattern_complexity": pattern_complexity,
            "angles": angles,
            "include_reflection": level >= 3,
            "include_unrelated": level >= 5,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1481)
        self._primary_complexity_feature = cfg["pattern_complexity"] + level

        # Generate pattern A
        pattern_a = self._make_pattern(cfg["pattern_complexity"], sub_rng)

        # Choose transformation type
        types = ["rotation"]
        if cfg["include_reflection"]:
            types.append("reflection")
        if cfg["include_unrelated"]:
            types.append("unrelated")
        true_type = sub_rng.choice(types)

        if true_type == "rotation":
            angle = sub_rng.choice(cfg["angles"])
            pattern_b = self._transform_rotate(pattern_a, angle)
            true_answer_desc = f"Rotated {angle} CCW"
        elif true_type == "reflection":
            axis = sub_rng.choice(["horizontal", "vertical"])
            pattern_b = self._transform_reflect(pattern_a, axis)
            true_answer_desc = f"Reflected {axis}ly"
        else:
            # Unrelated: different pattern
            pattern_b = self._make_pattern(cfg["pattern_complexity"], sub_rng)
            true_answer_desc = "Not a rotation/reflection of A"

        # Build 4 options - selection of (rotations, reflections, unrelated)
        option_set: List[str] = set()
        option_set.add(true_answer_desc)
        # Add other rotations from the active angle pool
        angle_pool = list(cfg["angles"])
        sub_rng.shuffle(angle_pool)
        for a in angle_pool:
            if len(option_set) >= 4:
                break
            d = f"Rotated {a} CCW"
            if d != true_answer_desc:
                option_set.add(d)
        if cfg["include_reflection"]:
            for axis in ("horizontal", "vertical"):
                if len(option_set) >= 4:
                    break
                d = f"Reflected {axis}ly"
                if d != true_answer_desc:
                    option_set.add(d)
        if cfg["include_unrelated"]:
            if len(option_set) < 4 and "Not a rotation/reflection of A" != true_answer_desc:
                option_set.add("Not a rotation/reflection of A")
        # Pad with extra rotations if still short
        extra_angles = [15, 75, 105, 165, 255, 345]
        for a in extra_angles:
            if len(option_set) >= 4:
                break
            option_set.add(f"Rotated {a} CCW")

        options = list(option_set)[:4]
        # Make sure the true answer is included
        if true_answer_desc not in options:
            options[-1] = true_answer_desc
        sub_rng.shuffle(options)
        correct_idx = options.index(true_answer_desc)
        answer_letter = chr(ord("A") + correct_idx)

        image = self._render(pattern_a, pattern_b, sub_rng)
        opt_lines = "\n".join(f"  ({chr(ord('A') + i)}) {o}"
                              for i, o in enumerate(options))
        question = (
            "Two images are shown side by side: Image A (left) and Image B "
            "(right). Determine how Image B relates to Image A.\n" +
            opt_lines +
            "\nAnswer with a single letter."
        )
        return question, answer_letter, image

    def _make_pattern(self, complexity: int,
                      rng: random.Random) -> List[Dict]:
        """Return list of shape primitives {shape, color, x, y, size}."""
        n_items = complexity + rng.randint(-1, 2)
        n_items = max(2, min(n_items, 10))
        items = []
        # Place shapes at distinct positions within a square [-1, 1]
        placed = []
        tries = 0
        while len(items) < n_items and tries < 100:
            tries += 1
            x = rng.uniform(-0.8, 0.8)
            y = rng.uniform(-0.8, 0.8)
            if any((x - px) ** 2 + (y - py) ** 2 < 0.12 for px, py in placed):
                continue
            placed.append((x, y))
            items.append({
                "shape": rng.choice(["circle", "square", "triangle"]),
                "color": rng.choice(_COLORS),
                "x": x,
                "y": y,
                "size": rng.uniform(0.1, 0.18),
            })
        return items

    @staticmethod
    def _transform_rotate(pattern, angle_deg):
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = []
        for it in pattern:
            x, y = it["x"], it["y"]
            nx = x * cos_a - y * sin_a
            ny = x * sin_a + y * cos_a
            out.append({**it, "x": nx, "y": ny, "rot": angle_deg})
        return out

    @staticmethod
    def _transform_reflect(pattern, axis):
        out = []
        for it in pattern:
            if axis == "horizontal":
                out.append({**it, "y": -it["y"]})
            else:
                out.append({**it, "x": -it["x"]})
        return out

    def _draw_pattern(self, ax, pattern: List[Dict], title: str):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        # Frame
        ax.add_patch(mpatches.Rectangle((-1.1, -1.1), 2.2, 2.2,
                                        fc="#f8f9fa", ec="#2c3e50", lw=1.3))
        # Axes reference (crosshair)
        ax.axhline(0, color="#bdc3c7", lw=0.6, zorder=0.5)
        ax.axvline(0, color="#bdc3c7", lw=0.6, zorder=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for it in pattern:
            self._draw_item(ax, it)

    def _draw_item(self, ax, it: Dict):
        x, y, size = it["x"], it["y"], it["size"]
        color = it["color"]
        shape = it["shape"]
        # Apply rotation to shape orientation if available
        rot = it.get("rot", 0)
        if shape == "circle":
            ax.add_patch(plt.Circle((x, y), size, fc=color, ec="black",
                                    lw=0.8, alpha=0.9))
        elif shape == "square":
            # Rotate square corners if rot specified
            rad = math.radians(rot)
            pts = [(-size, -size), (size, -size), (size, size), (-size, size)]
            pts = [(x + px * math.cos(rad) - py * math.sin(rad),
                    y + px * math.sin(rad) + py * math.cos(rad))
                   for px, py in pts]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=0.8,
                                     alpha=0.9))
        elif shape == "triangle":
            rad = math.radians(rot)
            raw = [(0, size), (-size, -size * 0.8), (size, -size * 0.8)]
            pts = [(x + px * math.cos(rad) - py * math.sin(rad),
                    y + px * math.sin(rad) + py * math.cos(rad))
                   for px, py in raw]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black", lw=0.8,
                                     alpha=0.9))

    def _render(self, pattern_a: List[Dict], pattern_b: List[Dict],
                rng: random.Random) -> Image.Image:
        style = self._random_style()
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 5.5))
        fig.patch.set_facecolor(style["bg_color"])
        self._draw_pattern(ax_a, pattern_a, "Image A")
        self._draw_pattern(ax_b, pattern_b, "Image B")
        fig.suptitle("Image Rotation / Transformation Match",
                     fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.05,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = ImageRotationMatchQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
