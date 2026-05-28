"""
Rotation Composition MCQ QA environment.

# 2026-05-05 R5 P1: rewrite to match a math benchmark Q4, Q7, Q9, Q13 verbatim
# (Basic Transformations of Figures topic, base 72.11% step175 59.18%
# delta -12.93). Stem opener "As shown in the diagram, ...", trailing
# "(    )" marker, options separated by "; ", "No correct answer" as
# the LAST option (and sometimes the actual ground truth).

Goal: teach 2D rotation-composition skill (spatial-vision 2DRotation
+ transformation geometry).

Difficulty schedule (multi-axis, continuous):
  L0/L1: single_vertex_coord algebra MCQ (KEEP from R3 — works for 4B)
  L2-L4: a math benchmark Q7-style — show shape A + 4 candidate rotated figures
         labeled A,B,C,D in the diagram, options "A. B; B. C; C. D;
         D. No correct answer" (model picks which figure-letter is the
         correct rotation).
  L5-L7: a math benchmark Q13-style — compose 2 transformations (rotate then
         translate, etc.). Same option format.
  L8-L9: a math benchmark Q4/Q9-style — rotation about NAMED point O (marked on
         both before and after panels), options name the rotation
         directly: "Rotate 90° clockwise; Rotate 90° counterclockwise; ...".
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Asymmetric shapes on a small grid (set of filled cells + red corner)
# ------------------------------------------------------------------ #

def _l_shape():
    """L-tromino fitting inside 3x3."""
    return [(0, 0), (0, 1), (0, 2), (1, 0)]

def _arrow_shape():
    return [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2), (2, 2)]

def _flag_shape():
    return [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (1, 2)]

def _f_shape():
    return [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def _s_shape():
    return [(0, 0), (1, 0), (1, 1), (2, 1)]

_SHAPES = [
    ("L", _l_shape),
    ("arrow", _arrow_shape),
    ("flag", _flag_shape),
    ("F", _f_shape),
    ("S", _s_shape),
]

def _rotate_cells(cells: List[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
    """Rotate cell coordinates 90° CW k times."""
    k = k % 4
    out = list(cells)
    for _ in range(k):
        out = [(y, -x) for (x, y) in out]
    minx = min(c[0] for c in out)
    miny = min(c[1] for c in out)
    return sorted([(x - minx, y - miny) for (x, y) in out])


def _translate_cells(cells, dx, dy):
    out = [(x + dx, y + dy) for (x, y) in cells]
    minx = min(c[0] for c in out)
    miny = min(c[1] for c in out)
    return sorted([(x - minx, y - miny) for (x, y) in out])


def _canonical_key(cells):
    return tuple(sorted(cells))


class RotationCompositionMcqQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "rotation_composition_mcq"

    # ------------------------------------------------------------------ #
    # exam-style stems and option templates
    # ------------------------------------------------------------------ #

    # Q7-style stems (figure-MCQ): "What is the shape obtained by ...?"
    _Q7_STEMS = [
        "What is the shape obtained by rotating Shape A {ang_text} around the center? (    )",
        "As shown in the diagram, what figure is obtained by rotating Shape A {ang_text} around the center? (    )",
        "Looking at the diagram, the shape obtained by rotating Shape A {ang_text} around the center is (    ).",
        "In the figure, after rotating Shape A {ang_text} around the center, the resulting shape is (    ).",
    ]

    # Q13-style stems (composite): "After rotating ... and then translating ..."
    _Q13_STEMS = [
        "After rotating Shape A {ang_text} around the center and then translating it {trans_text}, which figure is obtained? (    )",
        "As shown in the diagram, Shape A is rotated {ang_text} around the center, then translated {trans_text}. Which figure is the result? (    )",
        "Shape A is first rotated {ang_text} around the center, then translated {trans_text}. The resulting figure is (    ).",
    ]

    # Q4/Q9-style stems (rotation about named point O):
    _Q4_STEMS = [
        "As shown in the diagram, by rotating around the endpoint O, how can shape A be transformed into shape B? (    )",
        "As shown in the diagram, shape A is transformed into shape B by ______. (    )",
        "Looking at the diagram, what rotation around point O maps shape A onto shape B? (    )",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # KEEP single_vertex_coord algebra at L0/L1 (per task spec)
        if level == 0:
            return {"mode": "single_vertex_coord",
                    "force_angles": [180]}
        if level == 1:
            return {"mode": "single_vertex_coord"}
        # L2-L4: figure-MCQ (Q7-style), 1 rotation, increasing angle pool
        if level == 2:
            return {"mode": "figure_mcq", "n_rotations": 1,
                    "angle_pool": [90, 180, 270],
                    "show_grid": True}
        if level == 3:
            return {"mode": "figure_mcq", "n_rotations": 1,
                    "angle_pool": [90, 180, 270],
                    "show_grid": True}
        if level == 4:
            return {"mode": "figure_mcq", "n_rotations": 1,
                    "angle_pool": [90, 180, 270],
                    "show_grid": False}
        # L5-L7: composite (Q13-style)
        if level == 5:
            return {"mode": "composite_mcq", "n_rotations": 1,
                    "translate": True,
                    "angle_pool": [90, 180, 270],
                    "show_grid": True}
        if level == 6:
            return {"mode": "composite_mcq", "n_rotations": 1,
                    "translate": True,
                    "angle_pool": [90, 180, 270],
                    "show_grid": False}
        if level == 7:
            return {"mode": "composite_mcq", "n_rotations": 2,
                    "translate": False,
                    "angle_pool": [90, 180, 270],
                    "show_grid": False}
        # L8-L9: rotation about named point O (Q4/Q9-style)
        if level == 8:
            return {"mode": "named_point_rotation",
                    "angle_pool": [90, 180, 270]}
        return {"mode": "named_point_rotation",
                "angle_pool": [90, 180, 270]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level

        for _ in range(20):
            try:
                if cfg["mode"] == "single_vertex_coord":
                    result = self._gen_single_vertex_coord(sub_rng, cfg)
                elif cfg["mode"] == "figure_mcq":
                    result = self._gen_figure_mcq(sub_rng, cfg, level)
                elif cfg["mode"] == "composite_mcq":
                    result = self._gen_composite_mcq(sub_rng, cfg, level)
                elif cfg["mode"] == "named_point_rotation":
                    result = self._gen_named_point(sub_rng, cfg, level)
                else:
                    return None
                if result is not None:
                    return result
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # L0/L1: single-vertex coordinate algebra (KEEP from R3)
    # ------------------------------------------------------------------ #
    def _gen_single_vertex_coord(self, rng, cfg):
        if "force_angles" in cfg:
            angles_pool = cfg["force_angles"]
        else:
            angles_pool = [90, 180, 270]
        ang = rng.choice(angles_pool)
        x0 = rng.randint(1, 4)
        y0 = rng.randint(1, 4)
        if ang == 90:
            new_pt = (y0, -x0); op_text = "rotate 90° clockwise about origin"
        elif ang == 180:
            new_pt = (-x0, -y0); op_text = "rotate 180° about origin"
        else:
            new_pt = (-y0, x0); op_text = "rotate 270° clockwise about origin"
        all_results = {
            90: (y0, -x0), 180: (-x0, -y0), 270: (-y0, x0),
            "reflect_h": (-x0, y0), "reflect_v": (x0, -y0),
        }
        distractors = [v for k, v in all_results.items() if v != new_pt]
        rng.shuffle(distractors)
        options = [new_pt] + distractors[:3]
        rng.shuffle(options)
        idx = options.index(new_pt)
        answer = chr(ord("A") + idx)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
        ax.plot([x0], [y0], 'o', color="#c0392b", markersize=10)
        ax.text(x0 + 0.2, y0 + 0.2, f"({x0}, {y0})", fontsize=12,
                fontweight="bold", color="#c0392b")
        ax.axhline(0, color="#2c3e50", lw=1.0)
        ax.axvline(0, color="#2c3e50", lw=1.0)
        ax.grid(True, alpha=0.3)
        max_v = max(x0, y0) + 2
        ax.set_xlim(-max_v, max_v); ax.set_ylim(-max_v, max_v)
        ax.set_aspect("equal")
        ax.set_title("Original point (red dot)", fontsize=11)
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        img = Image.open(buf).copy()

        opt_text = "; ".join(f"{chr(ord('A')+i)}. ({options[i][0]}, {options[i][1]})"
                              for i in range(4))
        q = (
            f"As shown in the diagram, the original point ({x0}, {y0}) is "
            f"marked with a red dot. Apply the transformation: {op_text}. "
            f"What is the new coordinate? (    )\n"
            f"Options: {opt_text}\n"
            "Answer with a single letter A, B, C, or D."
        )
        return q, answer, img

    # ------------------------------------------------------------------ #
    # L2-L4: a math benchmark Q7-style figure MCQ
    # Shape A on left, 3 candidate figures labeled B, C, D on the right.
    # Options say "A. B; B. C; C. D; D. No correct answer" — meaning
    # option-letter A maps to figure-letter B, etc.
    # ------------------------------------------------------------------ #
    def _gen_figure_mcq(self, rng, cfg, level):
        # Pick shape
        if level <= 3:
            small_shapes = [(n, fn) for n, fn in _SHAPES if n in ("L", "S")]
            name, fn = rng.choice(small_shapes)
        else:
            name, fn = rng.choice(_SHAPES)
        base = fn()
        ang = rng.choice(cfg["angle_pool"])

        ang_text = f"{ang} degrees clockwise"
        correct_shape = _rotate_cells(base, ang // 90)
        correct_key = _canonical_key(correct_shape)

        # Build 3 distinct candidate figures (B, C, D) — one is correct,
        # OR (with prob ~0.25) all 3 are wrong and answer is "No correct
        # answer" (option D in the option list).
        no_correct = (rng.random() < 0.25)

        candidates = []
        seen = {_canonical_key(base)}

        if not no_correct:
            candidates.append(correct_shape)
            seen.add(correct_key)

        # Generate distractors (other rotation angles)
        wrong_angs = [a for a in [90, 180, 270] if a != ang]
        rng.shuffle(wrong_angs)
        for a in wrong_angs:
            cand = _rotate_cells(base, a // 90)
            ck = _canonical_key(cand)
            if ck not in seen:
                candidates.append(cand)
                seen.add(ck)
            if len(candidates) >= 3:
                break
        # Mirror as fallback
        tries = 0
        while len(candidates) < 3 and tries < 20:
            tries += 1
            mirrored = [(-x, y) for (x, y) in base]
            mx = min(c[0] for c in mirrored); my = min(c[1] for c in mirrored)
            mirrored = sorted([(x - mx, y - my) for (x, y) in mirrored])
            mk = _canonical_key(mirrored)
            if mk not in seen:
                candidates.append(mirrored)
                seen.add(mk)
                continue
            # Shift
            extra = _translate_cells(_rotate_cells(base, rng.randint(1, 3)),
                                     rng.randint(-1, 1), rng.randint(-1, 1))
            ek = _canonical_key(extra)
            if ek not in seen:
                candidates.append(extra)
                seen.add(ek)
        if len(candidates) < 3:
            return None
        candidates = candidates[:3]
        rng.shuffle(candidates)

        # Determine option-letter correctness
        # Option list: "A. B; B. C; C. D; D. No correct answer"
        # If correct_shape is in candidates at index 0 → option A (figure B)
        # If correct_shape is in candidates at index 1 → option B (figure C)
        # If correct_shape is in candidates at index 2 → option C (figure D)
        # If no_correct → option D
        if no_correct:
            answer = "D"
        else:
            for i, cand in enumerate(candidates):
                if _canonical_key(cand) == correct_key:
                    answer = chr(ord("A") + i)
                    break
            else:
                # Shouldn't happen, but safety
                return None

        # Build stem
        stem = self._Q7_STEMS[(self.seed or 0) % len(self._Q7_STEMS)]
        question_stem = stem.format(ang_text=ang_text)
        opts = "Options: A. B; B. C; C. D; D. No correct answer"
        question = (
            f"{question_stem}\n{opts}\n"
            "Answer with a single letter A, B, C, or D."
        )

        # Render: Shape A on left, then 3 candidates labeled B, C, D
        image = self._render_figure_mcq(base, candidates, cfg, "A",
                                        ["B", "C", "D"])
        return question, answer, image

    # ------------------------------------------------------------------ #
    # L5-L7: a math benchmark Q13-style composite (rotate + translate)
    # ------------------------------------------------------------------ #
    def _gen_composite_mcq(self, rng, cfg, level):
        name, fn = rng.choice(_SHAPES)
        base = fn()
        ang = rng.choice(cfg["angle_pool"])

        if cfg.get("translate"):
            # rotate then translate
            dy = rng.choice([-2, -1, 1, 2])
            dir_text = "downward" if dy < 0 else "upward"
            trans_text = f"{dir_text} by {abs(dy)} units"
            rotated = _rotate_cells(base, ang // 90)
            correct = _translate_cells(rotated, 0, dy)
            ang_text = f"clockwise by {ang} degrees"
            stem = self._Q13_STEMS[(self.seed or 0) % len(self._Q13_STEMS)]
            question_stem = stem.format(ang_text=ang_text, trans_text=trans_text)
        else:
            # 2 rotations
            ang2 = rng.choice(cfg["angle_pool"])
            ang_text = f"clockwise by {ang} degrees and then clockwise by {ang2} degrees"
            r1 = _rotate_cells(base, ang // 90)
            correct = _rotate_cells(r1, ang2 // 90)
            stem = ("As shown in the diagram, Shape A is rotated {ang_text} "
                    "around the center. Which figure is the result? (    )")
            question_stem = stem.format(ang_text=ang_text)

        # Build 3 candidates
        no_correct = (rng.random() < 0.25)
        candidates = []
        seen = {_canonical_key(base)}
        correct_key = _canonical_key(correct)

        if not no_correct:
            candidates.append(correct)
            seen.add(correct_key)

        # Distractors: wrong angles, opposite translate
        wrong_angs = [a for a in [90, 180, 270] if a != ang]
        rng.shuffle(wrong_angs)
        for a in wrong_angs:
            if cfg.get("translate"):
                cand = _translate_cells(_rotate_cells(base, a // 90), 0, dy)
            else:
                cand = _rotate_cells(_rotate_cells(base, a // 90), ang2 // 90)
            ck = _canonical_key(cand)
            if ck not in seen:
                candidates.append(cand)
                seen.add(ck)
            if len(candidates) >= 3:
                break
        # Wrong translate sign
        if cfg.get("translate") and len(candidates) < 3:
            cand = _translate_cells(_rotate_cells(base, ang // 90), 0, -dy)
            ck = _canonical_key(cand)
            if ck not in seen:
                candidates.append(cand)
                seen.add(ck)
        # Translation perpendicular fallback
        tries = 0
        while len(candidates) < 3 and tries < 20:
            tries += 1
            extra = _translate_cells(_rotate_cells(base, rng.randint(1, 3)),
                                     rng.randint(-1, 1), rng.randint(-1, 1))
            ek = _canonical_key(extra)
            if ek not in seen:
                candidates.append(extra)
                seen.add(ek)
        if len(candidates) < 3:
            return None
        candidates = candidates[:3]
        rng.shuffle(candidates)

        if no_correct:
            answer = "D"
        else:
            for i, cand in enumerate(candidates):
                if _canonical_key(cand) == correct_key:
                    answer = chr(ord("A") + i)
                    break
            else:
                return None

        opts = "Options: A. B; B. C; C. D; D. No correct answer"
        question = (
            f"{question_stem}\n{opts}\n"
            "Answer with a single letter A, B, C, or D."
        )
        image = self._render_figure_mcq(base, candidates, cfg, "A",
                                        ["B", "C", "D"])
        return question, answer, image

    # ------------------------------------------------------------------ #
    # L8-L9: a math benchmark Q4/Q9-style — rotation about NAMED point O
    # ------------------------------------------------------------------ #
    def _gen_named_point(self, rng, cfg, level):
        name, fn = rng.choice(_SHAPES)
        base = fn()
        ang = rng.choice(cfg["angle_pool"])
        # Rotate CW or CCW
        cw = rng.random() < 0.5
        if cw:
            after = _rotate_cells(base, ang // 90)
            correct_text = f"Rotate {ang}° clockwise"
        else:
            # CCW = 360 - ang CW
            after = _rotate_cells(base, (360 - ang) // 90)
            correct_text = f"Rotate {ang}° counterclockwise"

        # 4 options: CW @ ang, CCW @ ang, CW @ other, plus "No correct answer"
        other_ang = rng.choice([a for a in cfg["angle_pool"] if a != ang])
        no_correct = (rng.random() < 0.20)

        all_opts = [
            f"Rotate {ang}° clockwise",
            f"Rotate {ang}° counterclockwise",
            f"Rotate {other_ang}° clockwise",
            f"Rotate {other_ang}° counterclockwise",
        ]
        if no_correct:
            # Remove correct from pool
            all_opts = [o for o in all_opts if o != correct_text]
            rng.shuffle(all_opts)
            options = all_opts[:3] + ["No correct answer"]
            answer = "D"
        else:
            distractors = [o for o in all_opts if o != correct_text]
            rng.shuffle(distractors)
            options = [correct_text] + distractors[:2] + ["No correct answer"]
            rng.shuffle(options[:3])
            # Re-find idx (shuffled within first 3 only is wrong; shuffle all
            # 4 except keep "No correct" last as in exam-style)
            # Actually the cleaner pattern: shuffle first 3 properly
            non_no = options[:3]
            rng.shuffle(non_no)
            options = non_no + ["No correct answer"]
            for i, o in enumerate(options):
                if o == correct_text:
                    answer = chr(ord("A") + i)
                    break
            else:
                return None

        stem = self._Q4_STEMS[(self.seed or 0) % len(self._Q4_STEMS)]
        opts_str = "; ".join(f"{chr(ord('A')+i)}. {o}" for i, o in enumerate(options))
        question = (
            f"{stem}\n"
            f"Options: {opts_str}.\n"
            "Answer with a single letter A, B, C, or D."
        )
        # Render with named point O at the rotation center
        image = self._render_named_point(base, after, cfg)
        return question, answer, image

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_figure_mcq(self, base, candidates, cfg, base_label, cand_labels):
        # 4-panel: shape A + 3 candidates labeled B, C, D
        sc = 1.0
        fig, axes = plt.subplots(1, 4, figsize=(13 * sc, 3.6 * sc))
        fig.patch.set_facecolor("#ffffff")
        fig.suptitle("Rotation problem", fontsize=14, fontweight="bold", y=0.98)

        palette = ["#3498db", "#e67e22", "#27ae60", "#9b59b6"]
        edge_col = "#1a1a1a"

        ax0 = axes[0]
        ax0.set_facecolor("#ffffff")
        self._draw_cells(ax0, base, palette[0], "#ffffff", edge_col, 1.5,
                         label=f"Shape {base_label}",
                         show_grid=cfg.get("show_grid", False))

        for i, cand in enumerate(candidates):
            ax = axes[i + 1]
            ax.set_facecolor("#ffffff")
            self._draw_cells(ax, cand, palette[(i + 1) % 4], "#ffffff",
                             edge_col, 1.5,
                             label=f"Shape {cand_labels[i]}",
                             show_grid=cfg.get("show_grid", False))
        try: fig.tight_layout()
        except Exception: pass
        return self.fig_to_pil(fig, dpi=120)

    def _render_named_point(self, base, after, cfg):
        sc = 1.0
        fig, axes = plt.subplots(1, 2, figsize=(8 * sc, 4.0 * sc))
        fig.patch.set_facecolor("#ffffff")
        fig.suptitle("Rotation about point O", fontsize=14, fontweight="bold")

        palette = ["#3498db", "#e67e22"]
        edge_col = "#1a1a1a"

        for ax, cells, lab, col in [
            (axes[0], base, "Shape A", palette[0]),
            (axes[1], after, "Shape B", palette[1])]:
            ax.set_facecolor("#ffffff")
            self._draw_cells(ax, cells, col, "#ffffff", edge_col, 1.5,
                             label=lab, show_grid=True)
            # Mark rotation center O at origin (geometric center of the
            # bounding box). Offset to be visible.
            xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
            cx = (min(xs) + max(xs) + 1) / 2.0
            cy = (min(ys) + max(ys) + 1) / 2.0
            ax.plot(cx, cy, "o", color="#c0392b", markersize=8,
                    markeredgecolor="#000000")
            ax.text(cx + 0.15, cy + 0.15, "O", fontsize=14,
                    fontweight="bold", color="#c0392b")
        try: fig.tight_layout()
        except Exception: pass
        return self.fig_to_pil(fig, dpi=120)

    def _draw_cells(self, ax, cells, color, bg, edge_col, edge_lw,
                    label="", show_grid=False):
        if not cells:
            ax.axis("off"); return
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        w = max(xs) + 1
        h = max(ys) + 1
        pad = 1.0
        for (cx, cy) in cells:
            rect = plt.Rectangle((cx, cy), 1, 1,
                                 facecolor=color, edgecolor=edge_col,
                                 linewidth=edge_lw)
            ax.add_patch(rect)
        # Mark first cell with red square so chirality is visible
        fc = cells[0]
        ax.plot(fc[0] + 0.18, fc[1] + 0.18, "s", color="#e74c3c",
                markersize=6, markeredgecolor="#000000")
        ax.set_xlim(-pad, w + pad)
        ax.set_ylim(-pad, h + pad)
        ax.set_aspect("equal")
        if show_grid:
            ax.set_xticks(range(int(-pad), int(w + pad) + 1))
            ax.set_yticks(range(int(-pad), int(h + pad) + 1))
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.tick_params(axis="both", labelsize=6)
        else:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax.set_title(label, fontsize=11, fontweight="bold")


if __name__ == "__main__":
    import os
    os.makedirs("/tmp/env_check_b2a", exist_ok=True)
    env = RotationCompositionMcqQA()
    for level in [0, 3, 6, 9]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[rotation_composition_mcq] L{level} s{seed} FAILED")
                continue
            out = f"/tmp/env_check_b2a/rotation_composition_mcq_s{seed}_L{level}.png"
            env.render().save(out)
            print(f"saved {out} | A={env._answer}")
