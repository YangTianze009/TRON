"""
Coordinate Transform QA.

2026-05-05 R5 P1 polish — REWRITTEN as pure single-letter MCQ in textbook
style. Triangle ABC is drawn on a labeled grid; the question describes a
geometric transformation (translation, rotation 90°, reflection-then-
translation, or 3-step compound) and asks for the new coordinates of one
labeled vertex.

Per-level design:
  L0/L1 — 5-MCQ. Translate by (dx, dy). Loose distractors. No trap.
  L2-L4 — 5-MCQ. Rotation by 90° about origin. Loose distractors. 10/20% trap.
  L5-L7 — 5-MCQ. Reflection across an axis (x-axis or y-axis) followed by a
          small translation. 25% trap.
  L8/L9 — 5-MCQ. Three-step compound (translate + rotate + translate). Tight
          off-by-one distractors. 40% trap.

Question phrasing examples:
  - "Triangle ABC is translated 3 units to the right and 1 unit up. What are
     the coordinates of point A' (the image of vertex A)?"
  - "Triangle ABC is rotated 90° clockwise about the origin O. What are the
     coordinates of point A'?"
  - Options always end with "No correct answer".
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _fmt_coord(x: int, y: int) -> str:
    return f"({x}, {y})"


def _generate_distractors(
    correct: Tuple[int, int],
    rng: random.Random,
    k: int,
    tight: bool = False,
) -> List[Tuple[int, int]]:
    """Generate k unique distractor coord tuples."""
    cx, cy = correct
    if tight:
        candidates = [
            (cx + 1, cy), (cx - 1, cy),
            (cx, cy + 1), (cx, cy - 1),
            (cx + 1, cy + 1), (cx - 1, cy - 1),
            (cx + 1, cy - 1), (cx - 1, cy + 1),
            (-cx, cy), (cx, -cy), (-cx, -cy),
            (cy, cx), (-cy, cx), (cy, -cx), (-cy, -cx),
            (cx + 2, cy), (cx - 2, cy),
            (cx, cy + 2), (cx, cy - 2),
        ]
    else:
        candidates = [
            (cx + 2, cy), (cx - 2, cy),
            (cx, cy + 2), (cx, cy - 2),
            (-cx, cy), (cx, -cy), (-cx, -cy),
            (cy, cx), (-cy, cx), (cy, -cx),
            (cx + 1, cy - 2), (cx - 1, cy + 2),
            (cx + 3, cy + 1), (cx - 3, cy - 1),
        ]
    rng.shuffle(candidates)
    seen = {(cx, cy)}
    out = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _step_phrase(dx: int, dy: int) -> str:
    """Convert (dx, dy) → 'right 2 units and up 1 unit' style."""
    parts = []
    if dx != 0:
        if dx < 0:
            parts.append(f"left {abs(dx)} unit{'s' if abs(dx) != 1 else ''}")
        else:
            parts.append(f"right {dx} unit{'s' if dx != 1 else ''}")
    if dy != 0:
        if dy < 0:
            parts.append(f"down {abs(dy)} unit{'s' if abs(dy) != 1 else ''}")
        else:
            parts.append(f"up {dy} unit{'s' if dy != 1 else ''}")
    if not parts:
        return "0 units"
    if len(parts) == 1:
        return parts[0]
    return parts[0] + " and " + parts[1]


def _rotate90(p: Tuple[int, int], cw: bool) -> Tuple[int, int]:
    """Rotate (x, y) by 90 degrees about origin. CW or CCW."""
    x, y = p
    if cw:
        return (y, -x)
    return (-y, x)


def _rotate180(p: Tuple[int, int]) -> Tuple[int, int]:
    x, y = p
    return (-x, -y)


def _reflect(p: Tuple[int, int], axis: str) -> Tuple[int, int]:
    x, y = p
    if axis == "x":
        return (x, -y)
    if axis == "y":
        return (-x, y)
    return p


class CoordinateTransformQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive
    ENV_NAME = "coordinate_transform"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "qtype": "translate",
                "n_options": 5,
                "trap_rate": 0.0,
                "tight": False,
            }
        if level <= 4:
            return {
                "qtype": "rotate90",
                "n_options": 5,
                "trap_rate": 0.10 if level <= 3 else 0.20,
                "tight": False,
            }
        if level <= 7:
            return {
                "qtype": "reflect_translate",
                "n_options": 5,
                "trap_rate": 0.25,
                "tight": False,
            }
        # L8/L9
        return {
            "qtype": "compound_3step",
            "n_options": 5,
            "trap_rate": 0.40,
            "tight": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1030)
        qtype = cfg["qtype"]
        for _ in range(20):
            if qtype == "translate":
                res = self._try_generate_translate(cfg, rng)
            elif qtype == "rotate90":
                res = self._try_generate_rotate90(cfg, rng)
            elif qtype == "reflect_translate":
                res = self._try_generate_reflect_translate(cfg, rng)
            else:
                res = self._try_generate_compound_3step(cfg, rng)
            if res is not None:
                return res
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sample_triangle(self, rng) -> List[Tuple[int, int]]:
        for _ in range(30):
            pts = [(rng.randint(-3, 3), rng.randint(-3, 3))
                   for _ in range(3)]
            (x1, y1), (x2, y2), (x3, y3) = pts
            area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if area2 >= 4:
                return pts
        return [(-2, 0), (2, 0), (0, 3)]

    def _build_options(self, true_pt: Tuple[int, int],
                       cfg: Dict, rng: random.Random
                       ) -> Tuple[List[str], str]:
        n_options = cfg["n_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        last_letter = opt_letters[-1]

        n_distractor_pool = n_options + 2
        distractor_pool = _generate_distractors(
            true_pt, rng, k=n_distractor_pool, tight=cfg["tight"],
        )
        if len(distractor_pool) < n_options - 1:
            return None, None

        use_trap = (rng.random() < cfg["trap_rate"])
        if use_trap:
            n_numeric = n_options - 1
            chosen = distractor_pool[:n_numeric]
            rng.shuffle(chosen)
            opt_texts = [_fmt_coord(*c) for c in chosen]
            opt_texts.append("No correct answer")
            correct_letter = last_letter
        else:
            n_distractor = n_options - 2
            chosen = [true_pt] + distractor_pool[:n_distractor]
            rng.shuffle(chosen)
            opt_texts = [_fmt_coord(*c) for c in chosen]
            opt_texts.append("No correct answer")
            correct_letter = opt_letters[chosen.index(true_pt)]
        return opt_texts, correct_letter

    def _format_question(self, stem: str, opt_texts: List[str],
                         n_options: int) -> str:
        opt_letters = ["A", "B", "C", "D", "E"][:n_options]
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {opt_texts[i]}" for i in range(n_options)
        )
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        return (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

    # ------------------------------------------------------------------ #
    # L0/L1 — translation
    # ------------------------------------------------------------------ #
    def _try_generate_translate(self, cfg, rng):
        pts = self._sample_triangle(rng)
        labels = ["A", "B", "C"]
        target_idx = rng.randint(0, 2)
        target_label = labels[target_idx]

        # Translation step (small for L0/L1)
        max_step = 3
        for _ in range(20):
            dx = rng.randint(-max_step, max_step)
            dy = rng.randint(-max_step, max_step)
            if (dx, dy) != (0, 0):
                break
        sx, sy = pts[target_idx]
        true_pt = (sx + dx, sy + dy)

        opt_texts, correct_letter = self._build_options(true_pt, cfg, rng)
        if opt_texts is None:
            return None

        # Standard textbook phrasing
        stems = [
            (f"As shown in the diagram, triangle ABC is translated "
             f"{_step_phrase(dx, dy)}. What are the coordinates of point "
             f"{target_label}' (the image of vertex {target_label}) after "
             f"the translation?"),
            (f"As shown in the figure, if point {target_label} of triangle "
             f"ABC is translated {_step_phrase(dx, dy)}, the resulting "
             f"point {target_label}' has coordinates (    )."),
            (f"In the diagram below, triangle ABC is translated "
             f"{_step_phrase(dx, dy)}. The new coordinates of point "
             f"{target_label}' are (    )."),
        ]
        stem = rng.choice(stems)
        question = self._format_question(stem, opt_texts, cfg["n_options"])
        image = self._draw_triangle(pts, labels, target_idx, rng,
                                    title="Triangle ABC on coordinate grid")
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L2-L4 — rotation by 90° about origin
    # ------------------------------------------------------------------ #
    def _try_generate_rotate90(self, cfg, rng):
        pts = self._sample_triangle(rng)
        labels = ["A", "B", "C"]
        target_idx = rng.randint(0, 2)
        target_label = labels[target_idx]
        cw = rng.choice([True, False])
        true_pt = _rotate90(pts[target_idx], cw)

        opt_texts, correct_letter = self._build_options(true_pt, cfg, rng)
        if opt_texts is None:
            return None

        dir_word = "clockwise" if cw else "counterclockwise"
        stems = [
            (f"As shown in the diagram, triangle ABC is rotated 90° "
             f"{dir_word} about the origin O. What are the coordinates of "
             f"point {target_label}' (the image of vertex {target_label})?"),
            (f"As shown in the figure, if triangle ABC is rotated 90° "
             f"{dir_word} about the origin, the new coordinates of point "
             f"{target_label}' are (    )."),
            (f"The diagram shows triangle ABC. After rotating it 90° "
             f"{dir_word} around the origin O, the coordinates of point "
             f"{target_label}' are (    )."),
        ]
        stem = rng.choice(stems)
        question = self._format_question(stem, opt_texts, cfg["n_options"])
        image = self._draw_triangle(pts, labels, target_idx, rng,
                                    title="Triangle ABC on coordinate grid",
                                    show_origin=True)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L5-L7 — reflection across an axis followed by translation
    # ------------------------------------------------------------------ #
    def _try_generate_reflect_translate(self, cfg, rng):
        pts = self._sample_triangle(rng)
        labels = ["A", "B", "C"]
        target_idx = rng.randint(0, 2)
        target_label = labels[target_idx]
        axis = rng.choice(["x", "y"])
        for _ in range(20):
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            if (dx, dy) != (0, 0):
                break
        sx, sy = pts[target_idx]
        rx, ry = _reflect((sx, sy), axis)
        true_pt = (rx + dx, ry + dy)

        opt_texts, correct_letter = self._build_options(true_pt, cfg, rng)
        if opt_texts is None:
            return None

        axis_word = "x-axis" if axis == "x" else "y-axis"
        stems = [
            (f"As shown in the diagram, triangle ABC is first reflected "
             f"across the {axis_word}, and then translated "
             f"{_step_phrase(dx, dy)}. What are the coordinates of point "
             f"{target_label}' (the image of vertex {target_label})?"),
            (f"As shown in the figure, triangle ABC undergoes a reflection "
             f"across the {axis_word} followed by a translation of "
             f"{_step_phrase(dx, dy)}. The new coordinates of point "
             f"{target_label}' are (    )."),
        ]
        stem = rng.choice(stems)
        question = self._format_question(stem, opt_texts, cfg["n_options"])
        image = self._draw_triangle(pts, labels, target_idx, rng,
                                    title="Triangle ABC on coordinate grid",
                                    show_origin=True)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # L8/L9 — compound 3-step (translate + rotate + translate)
    # ------------------------------------------------------------------ #
    def _try_generate_compound_3step(self, cfg, rng):
        pts = self._sample_triangle(rng)
        labels = ["A", "B", "C"]
        target_idx = rng.randint(0, 2)
        target_label = labels[target_idx]

        for _ in range(20):
            dx1 = rng.randint(-2, 2)
            dy1 = rng.randint(-2, 2)
            if (dx1, dy1) != (0, 0):
                break
        rotation_choice = rng.choice(["cw90", "ccw90", "180"])
        for _ in range(20):
            dx2 = rng.randint(-2, 2)
            dy2 = rng.randint(-2, 2)
            if (dx2, dy2) != (0, 0):
                break
        sx, sy = pts[target_idx]
        # Step 1: translate
        x1, y1 = sx + dx1, sy + dy1
        # Step 2: rotate about origin
        if rotation_choice == "cw90":
            x2, y2 = _rotate90((x1, y1), cw=True)
            rot_text = "rotated 90° clockwise about the origin"
        elif rotation_choice == "ccw90":
            x2, y2 = _rotate90((x1, y1), cw=False)
            rot_text = "rotated 90° counterclockwise about the origin"
        else:
            x2, y2 = _rotate180((x1, y1))
            rot_text = "rotated 180° about the origin"
        # Step 3: translate
        true_pt = (x2 + dx2, y2 + dy2)

        opt_texts, correct_letter = self._build_options(true_pt, cfg, rng)
        if opt_texts is None:
            return None

        stems = [
            (f"As shown in the diagram, triangle ABC is first translated "
             f"{_step_phrase(dx1, dy1)}, then {rot_text}, and finally "
             f"translated {_step_phrase(dx2, dy2)}. What are the "
             f"coordinates of point {target_label}' (the image of vertex "
             f"{target_label}) after all three transformations?"),
            (f"In the figure, triangle ABC undergoes three transformations "
             f"in order: (1) translate {_step_phrase(dx1, dy1)}, (2) "
             f"{rot_text}, (3) translate {_step_phrase(dx2, dy2)}. The "
             f"final coordinates of point {target_label}' are (    )."),
        ]
        stem = rng.choice(stems)
        question = self._format_question(stem, opt_texts, cfg["n_options"])
        image = self._draw_triangle(pts, labels, target_idx, rng,
                                    title="Triangle ABC on coordinate grid",
                                    show_origin=True)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw_triangle(self, pts: List[Tuple[int, int]],
                       labels: List[str], target_idx: int,
                       rng: random.Random,
                       title: str = "Triangle ABC on coordinate grid",
                       show_origin: bool = False) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = list(style["palette"])
        rng.shuffle(palette)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        margin = 2
        xmin = min(xs + [0]) - margin
        xmax = max(xs + [0]) + margin
        ymin = min(ys + [0]) - margin
        ymax = max(ys + [0]) + margin

        ax.set_xticks(range(xmin, xmax + 1))
        ax.set_yticks(range(ymin, ymax + 1))
        ax.grid(True, alpha=0.40, linestyle="--", linewidth=0.5)
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.5, ymax + 0.5)
        ax.set_aspect("equal")
        ax.axhline(0, color="#444", linewidth=0.7, alpha=0.7)
        ax.axvline(0, color="#444", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        poly = MplPolygon(
            pts, closed=True, facecolor=palette[0], alpha=0.30,
            edgecolor=palette[0], linewidth=2, zorder=3,
        )
        ax.add_patch(poly)

        for i, (x, y) in enumerate(pts):
            is_target = (i == target_idx)
            ax.plot(x, y, "o", color=palette[0],
                    markersize=12 if is_target else 7,
                    markerfacecolor=palette[0],
                    markeredgecolor="black",
                    markeredgewidth=1.2, zorder=5)
            label_text = labels[i]
            if is_target:
                label_text = f"{labels[i]} (target)"
            ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(x + 0.30, y + 0.30),
                fontsize=14 if not is_target else 13,
                fontweight="bold",
                color="black",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=palette[0], alpha=0.85)
                if is_target else None,
            )

        if show_origin:
            ax.plot(0, 0, "o", color="black", markersize=8, zorder=5)
            ax.annotate("O", xy=(0, 0), xytext=(0.3, 0.3),
                        fontsize=14, fontweight="bold", color="red",
                        zorder=6)

        ax.tick_params(axis="both", labelsize=9)
        ax.set_title(title, fontsize=style["font_size_base"] + 1)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        return super()._check_answer(predicted, ground_truth)


if __name__ == "__main__":
    env = CoordinateTransformQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed}: ok={ok} A={env._answer}")
