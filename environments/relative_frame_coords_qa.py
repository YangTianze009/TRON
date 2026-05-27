"""
Relative Frame Coordinates QA (W44, P2).

reference W44 spec: "given a labelled point B has coordinates (x_B, y_B), what
coordinates does point A have?" with an unspecified-origin grid (no axes
visible). The model must use the grid distance from B to A to deduce A's
coordinates.

This is a 2D grid problem: render a square grid (no axis labels), mark two
points A and B with B's coordinates labelled. Ask for A's coordinates as a
pair (x, y).

Answer format: "(x, y)" — handled by custom _check_answer pair parser.
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

from .standalone_base import StandaloneVisualEnv


class RelativeFrameCoordsQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_REL = 0.01
    ENV_NAME = "relative_frame_coords"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"grid_size": 6, "delta_range": 2}
        if level <= 5:
            return {"grid_size": 8, "delta_range": 4}
        return {"grid_size": 10, "delta_range": 6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        # 2026-05-04: added easier L0 mode (was 10% — VLM mental-rotation limit, attempt fix)
        # L0/L1 give the offset in text — pure addition, no visual estimation.
        if level <= 1:
            return self._generate_easy_l0l1(level)
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 6047 + level * 79 + 11)

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _generate_easy_l0l1(self, level: int):
        """Easy L0/L1: state offset in text. Pure (Bx+dx, By+dy) addition."""
        rng = random.Random((self.seed or 0) * 6047 + level * 79 + 17)
        Bx_label = rng.randint(-3, 5)
        By_label = rng.randint(-3, 5)
        d_max = 1 if level == 0 else 2
        dx = rng.randint(-d_max, d_max)
        dy = rng.randint(-d_max, d_max)
        if dx == 0 and dy == 0:
            dx = 1
        Ax_label = Bx_label + dx
        Ay_label = By_label + dy

        # render a simple grid with both labelled; question states the offset
        gs = 6
        Bx_world = 2
        By_world = 3
        Ax_world = max(0, min(gs - 1, Bx_world + dx))
        Ay_world = max(0, min(gs - 1, By_world + dy))
        img = self._render(gs, (Ax_world, Ay_world, "A"),
                           (Bx_world, By_world, "B", Bx_label, By_label))
        sx = "right" if dx > 0 else ("left" if dx < 0 else "none")
        sy = "up" if dy > 0 else ("down" if dy < 0 else "none")
        question = (
            f"Point B has coordinates ({Bx_label}, {By_label}). "
            f"Point A is offset from B by {dx} units in x and {dy} units in y "
            f"(x-direction: {sx}, y-direction: {sy}). "
            f"What are the coordinates of point A? "
            f"Compute (Bx + {dx}, By + {dy}). "
            f"Format the answer as `(x, y)`."
        )
        answer = f"({Ax_label}, {Ay_label})"
        return question, answer, img

    def _try_generate(self, rng, cfg, level):
        gs = cfg["grid_size"]
        # Pick B's coordinates (visible to model) and A's coordinates (unknown).
        Bx_world = rng.randint(1, gs - 2)
        By_world = rng.randint(1, gs - 2)
        # B's labelled coords (in some frame)
        Bx_label = rng.randint(-3, 5)
        By_label = rng.randint(-3, 5)
        # A's grid position
        d = cfg["delta_range"]
        Ax_world = max(0, min(gs - 1, Bx_world + rng.randint(-d, d)))
        Ay_world = max(0, min(gs - 1, By_world + rng.randint(-d, d)))
        if Ax_world == Bx_world and Ay_world == By_world:
            return None
        # A's labelled coords are inferred from offset
        Ax_label = Bx_label + (Ax_world - Bx_world)
        Ay_label = By_label + (Ay_world - By_world)

        question = (
            f"In the figure, a square grid is shown with two labelled points A "
            f"and B. The coordinates of point B are ({Bx_label}, {By_label}). "
            f"Using the grid spacing as the unit, determine the coordinates of "
            f"point A. Format the answer as `(x, y)`."
        )
        answer = f"({Ax_label}, {Ay_label})"
        img = self._render(gs, (Ax_world, Ay_world, "A"),
                            (Bx_world, By_world, "B", Bx_label, By_label))
        return question, answer, img

    # ------------------------------------------------------------------ #
    def _render(self, gs, A, B) -> Image.Image:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Grid (no numeric labels)
        for i in range(gs + 1):
            ax.plot([0, gs], [i, i], color="#bdc3c7", linewidth=0.8)
            ax.plot([i, i], [0, gs], color="#bdc3c7", linewidth=0.8)

        Ax, Ay, Alab = A
        Bx, By, Blab, Bxl, Byl = B
        ax.scatter([Ax], [Ay], s=120, color="#d62728", zorder=5,
                   edgecolor="black", linewidth=1.2)
        ax.text(Ax + 0.2, Ay + 0.2, Alab, fontsize=14, fontweight="bold",
                color="#d62728")

        ax.scatter([Bx], [By], s=120, color="#2ecc71", zorder=5,
                   edgecolor="black", linewidth=1.2)
        ax.text(Bx + 0.2, By + 0.2, f"{Blab}({Bxl}, {Byl})",
                fontsize=12, fontweight="bold", color="#27ae60")

        ax.set_xlim(-0.5, gs + 0.5)
        ax.set_ylim(-0.5, gs + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)

    # ------------------------------------------------------------------ #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        gt = self._parse_pair(ground_truth)
        pr = self._parse_pair(predicted)
        if gt is None:
            return super()._check_answer(predicted, ground_truth)
        if pr is None:
            return False
        return abs(pr[0] - gt[0]) < 0.5 and abs(pr[1] - gt[1]) < 0.5

    @staticmethod
    def _parse_pair(s):
        if s is None:
            return None
        t = s.strip().lower().strip("()[]{}")
        t = re.sub(r"\bx\s*=\s*", "", t)
        t = re.sub(r"\by\s*=\s*", "", t)
        parts = re.split(r"[,;]|\band\b", t)
        vals = []
        for p in parts:
            p = p.strip().strip("()[]{}")
            if not p:
                continue
            try:
                vals.append(float(p))
            except ValueError:
                m = re.match(r"^(-?\d+)\s*/\s*(\d+)$", p)
                if m:
                    vals.append(int(m.group(1)) / int(m.group(2)))
                else:
                    return None
        if len(vals) != 2:
            return None
        return tuple(vals)


if __name__ == "__main__":
    env = RelativeFrameCoordsQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
