"""
Cuboid Into Cubes QA.

A cuboid with length l, width w, height h is split into N identical small
cubes of side s. Two question modes:
  - "find_n":    given l, w, h, and the small cube side s, find N = (l*w*h) / s^3.
  - "find_side": given l, w, h, and N, find the small cube side s.

The figure shows the cuboid with grid lines indicating how it is partitioned
into the small cubes (e.g. an l/s × w/s × h/s grid).

Difficulty axes:
  - L0..L2: integer cube side, integer N <= 64; small cuboid dims.
  - L3..L5: bigger cuboid dims, N up to a few hundred.
  - L6..L9: integer side >= 2 + larger N (up to several thousand).
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_FIND_N = [
    "As shown in the diagram, a cuboid is split into identical small cubes of the labeled side. How many small cubes are produced?",
    "As shown in the figure, a cuboid with the labeled dimensions is divided into identical small cubes of side s. What is the total number of small cubes?",
    "As shown in the diagram, the cuboid in the figure is partitioned into identical small cubes (side s shown). How many small cubes are there in total?",
    "As shown in the figure, using the labeled length, width, height, and cube side s, how many small cubes fill the cuboid?",
    "As shown in the diagram, the cuboid is split into identical small cubes of the labeled side. What is the count of small cubes?",
    "As shown in the figure, the cuboid is divided into small cubes as labeled. How many small cubes are produced in total?",
    "As shown in the diagram, the cuboid splits into identical cubes of side s shown in the figure. What is the total number of small cubes?",
    "As shown in the figure, a rectangular block is split into identical unit cubes of side s. How many cubes are produced from the block?",
]

_TEMPLATES_FIND_SIDE = [
    "As shown in the diagram, the cuboid is split into N identical small cubes (N labeled). Using the labeled l, w, h, what is the side length of one small cube?",
    "As shown in the figure, a cuboid with the labeled dimensions is divided into N identical cubes (N shown). What is the side length of one small cube?",
    "As shown in the diagram, the cuboid in the figure splits into N identical small cubes (N labeled). What is the side length of each small cube?",
    "As shown in the figure, given the labeled cuboid (l, w, h) and the count N of identical small cubes, what is the side length s of one small cube?",
    "As shown in the diagram, the cuboid is partitioned into N identical small cubes. Using the labeled dimensions, what is the cube's side length?",
    "As shown in the figure, the cuboid divides into N identical small cubes (N labeled). What is the side length of one small cube?",
]


class CuboidIntoCubesQA(StandaloneVisualEnv):
    ENV_NAME = "cuboid_into_cubes"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {
                "side_pool": [1, 2],
                "factor_pool": [(1, 2, 2), (2, 2, 2), (1, 2, 3), (2, 3, 3),
                                (1, 1, 2), (1, 1, 1)],
                "modes": ["find_n"],
            }
        if level <= 5:
            return {
                "side_pool": [1, 2, 3],
                "factor_pool": [(2, 2, 3), (2, 3, 3), (2, 3, 4), (3, 3, 3),
                                (2, 2, 4), (1, 2, 4), (3, 3, 4), (2, 4, 4)],
                "modes": ["find_n", "find_side"],
            }
        if level <= 7:
            return {
                "side_pool": [2, 3, 4, 5],
                "factor_pool": [(2, 3, 4), (3, 3, 4), (3, 4, 5), (4, 4, 4),
                                (2, 4, 5), (3, 4, 4), (4, 4, 5), (3, 5, 5)],
                "modes": ["find_n", "find_side"],
            }
        # 2026-05-04: L9 was 100% saturated. Larger cuboids and mixed sides.
        # 2026-05-04: bumped L9 difficulty further (was still 100% saturated
        # → much larger factor products N forcing real multiplication, plus
        # bigger side pool).
        # 2026-05-04 R3: benchmark-sample-driven harden — enable
        # "E. No correct answer" trap (~25% chance E is GT) at L8/L9.
        return {
            "side_pool": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "factor_pool": [(7, 9, 11), (8, 9, 11), (7, 11, 13), (9, 11, 13),
                            (8, 11, 13), (10, 11, 13), (9, 12, 13),
                            (11, 12, 13), (8, 13, 15), (9, 13, 15),
                            (11, 13, 15)],
            "modes": ["find_n", "find_side"],
            "e_trap": True,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1213 + level * 83 + 17)

        s = rng.choice(cfg["side_pool"])
        nx, ny, nz = rng.choice(cfg["factor_pool"])
        l = nx * s
        w = ny * s
        h = nz * s
        N = nx * ny * nz

        mode = rng.choice(cfg["modes"])
        if mode == "find_n":
            correct = N
            templates = _TEMPLATES_FIND_N
            unit = ""
            distractors = [
                nx + ny + nz,        # added instead of multiplied
                l * w * h,           # used cuboid volume
                2 * N,
                N // 2 if N > 1 else N + 1,
                nx * ny,             # missed one factor
                N + 4,
            ]
        else:
            correct = s
            templates = _TEMPLATES_FIND_SIDE
            unit = ""
            distractors = [
                s * 2,
                max(1, s - 1),
                s + 1,
                l // (nx + 1) if (nx + 1) > 0 else s + 2,
                N // (s if s > 0 else 1),
            ]

        sidx = (self.seed or 0) % len(templates)
        stem = templates[sidx]
        # Build MCQ A-D + E. No correct answer
        # 2026-05-04 R3: benchmark-sample-driven harden — at L8/L9 with
        # ~25% chance, E is GT and all visible options are wrong.
        if cfg.get("e_trap") and rng.random() < 0.25:
            wrong_vals = []
            seen_w = {correct}
            for d in distractors:
                if d != correct and d > 0 and d not in seen_w:
                    wrong_vals.append(d)
                    seen_w.add(d)
                if len(wrong_vals) >= 4:
                    break
            if len(wrong_vals) < 4:
                for delta in [1, 2, 3, 5, 7, 9, 11, 13, 17]:
                    cand = correct + delta
                    if cand not in seen_w and cand > 0:
                        wrong_vals.append(cand)
                        seen_w.add(cand)
                    if len(wrong_vals) >= 4:
                        break
            if len(wrong_vals) < 4:
                return None
            rng.shuffle(wrong_vals)
            opts_vals = wrong_vals[:4]
            answer = "E"
        else:
            opts_vals = [correct]
            seen = {correct}
            for d in distractors:
                if d != correct and d > 0 and d not in seen:
                    opts_vals.append(d)
                    seen.add(d)
                if len(opts_vals) >= 4:
                    break
            if len(opts_vals) < 4:
                for delta in [1, 2, 3, 5, 7, 9]:
                    cand = correct + delta
                    if cand not in seen and cand > 0:
                        opts_vals.append(cand)
                        seen.add(cand)
                    if len(opts_vals) >= 4:
                        break
            if len(opts_vals) < 4:
                return None
            rng.shuffle(opts_vals)
            correct_idx = opts_vals.index(correct)
            answer = "ABCD"[correct_idx]
        opts_lines = [f"{chr(ord('A')+i)}. {opts_vals[i]}{(' ' + unit) if unit else ''}"
                      for i in range(4)]
        opts_lines.append("E. No correct answer")
        question = (
            f"{stem}\n\n"
            + "\n".join(opts_lines)
            + "\n\nChoose the correct option (A, B, C, D, or E)."
        )
        img = self._render(l, w, h, s, nx, ny, nz, N, mode)
        return question, answer, img

    def _render(self, l, w, h, s, nx, ny, nz, N, mode) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        scale = 1.0
        # Axonometric projection.
        dx, dy = 0.55, 0.55
        L = l * scale
        W = w * scale
        H = h * scale

        FBL = (0.0, 0.0)
        FBR = (L, 0.0)
        BBR = (L + W * dx, W * dy)
        BBL = (W * dx, W * dy)
        FTL = (0.0, H)
        FTR = (L, H)
        BTR = (L + W * dx, H + W * dy)
        BTL = (W * dx, H + W * dy)

        edge_color = "#2c3e50"
        face_top = "#e7eef9"
        face_front = "#dbe5f5"
        face_side = "#cdd9eb"
        grid_color = "#34495e"

        # Top face
        ax.add_patch(patches.Polygon(
            [FTL, FTR, BTR, BTL], closed=True,
            facecolor=face_top, edgecolor=edge_color, linewidth=1.8))
        # Front face
        ax.add_patch(patches.Polygon(
            [FBL, FBR, FTR, FTL], closed=True,
            facecolor=face_front, edgecolor=edge_color, linewidth=1.8))
        # Right face
        ax.add_patch(patches.Polygon(
            [FBR, BBR, BTR, FTR], closed=True,
            facecolor=face_side, edgecolor=edge_color, linewidth=1.8))

        # Grid lines on top face (showing nx columns × ny rows of cubes)
        for i in range(1, nx):
            x = i * s
            ax.plot([x, x + W * dx], [H, H + W * dy], color=grid_color,
                    lw=0.7, alpha=0.7)
        for j in range(1, ny):
            x_off = j * s * dx
            y_off = j * s * dy
            ax.plot([x_off, L + x_off], [H + y_off, H + y_off],
                    color=grid_color, lw=0.7, alpha=0.7)

        # Grid lines on front face (nx columns × nz rows of cubes)
        for i in range(1, nx):
            x = i * s
            ax.plot([x, x], [0, H], color=grid_color, lw=0.7, alpha=0.7)
        for k in range(1, nz):
            y = k * s
            ax.plot([0, L], [y, y], color=grid_color, lw=0.7, alpha=0.7)

        # Grid lines on right face (ny columns × nz rows of cubes)
        for j in range(1, ny):
            x_off = j * s * dx
            y_off = j * s * dy
            ax.plot([L + x_off, L + x_off], [y_off, H + y_off],
                    color=grid_color, lw=0.7, alpha=0.7)
        for k in range(1, nz):
            y = k * s
            ax.plot([L, L + W * dx], [y, y + W * dy],
                    color=grid_color, lw=0.7, alpha=0.7)

        # Length / width / height labels
        ax.annotate("", xy=(L, -0.4), xytext=(0, -0.4),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(L / 2, -0.65, f"l = {self._fmt(l)}",
                ha="center", va="top", fontsize=12, color="#c0392b",
                fontweight="bold")
        ax.annotate("", xy=BBR, xytext=FBR,
                    arrowprops=dict(arrowstyle="<->", color="#1e8449", lw=1.4))
        ax.text(L + W * dx / 2 + 0.45, W * dy / 2 - 0.1,
                f"w = {self._fmt(w)}",
                ha="left", va="center", fontsize=12, color="#1e8449",
                fontweight="bold")
        ax.annotate("", xy=(L + 0.4, H), xytext=(L + 0.4, 0),
                    arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.4))
        ax.text(L + 0.6, H / 2, f"h = {self._fmt(h)}",
                ha="left", va="center", fontsize=12, color="#2c3e50",
                fontweight="bold")

        # Mode-specific extra label
        if mode == "find_n":
            label_text = f"small cube side  s = {self._fmt(s)}"
        else:
            label_text = f"split into  N = {N}  identical cubes"
        ax.text((L + W * dx) / 2, H + W * dy + 0.5, label_text,
                ha="center", va="bottom", fontsize=11, color="#7b241c",
                fontweight="bold")

        ax.set_xlim(-1, L + W * dx + 1.5)
        ax.set_ylim(-1.4, H + W * dy + 1.6)
        ax.set_aspect("equal")
        ax.axis("off")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    @staticmethod
    def _fmt(v):
        if isinstance(v, int):
            return str(v)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:g}"
