"""
Parabola Vertex QA (batch 3, 2026-04-14).

Target: Coordinate . A parabola y = a(x-h)^2 + k is plotted
on a labelled coordinate grid. The model must read the vertex, axis of
symmetry, or a root.

Format: numeric short answer (x-coordinate value or single integer/decimal).

Difficulty axes:
  A) Pattern H: vertex range expansion (0..3 L0 -> -5..5 L9)
  B) Pattern H: |a| goes 1 (L0) → up to 3 (L9), and sign flips at L>=3
  C) Pattern G: grid-label thinning at higher levels
  D) Pattern A (question variant): L0 asks vertex-x; L3 alternates root/x; L6 axis
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class ParabolaVertexQA(StandaloneVisualEnv):
    ENV_NAME = "parabola_vertex"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Difficulty redesign 2026-04-14. L0 starts harder (negative a, bigger range).
        return {
            "h_range": 2 + level,            # 2..11 (was 1..5)
            "k_range": 3 + level,            # 3..12
            "allow_negative_a": level >= 1,  # was >= 3
            "a_max": 1 + level // 2,         # 1,1,2,2,3,3,4,4,5,5
            "show_grid": level <= 3,          # was <= 5
            "show_vertex_dot": level <= 0,   # was <= 2
            "axis_pad": 2 + level // 2,
            # L6+: hide axis tick labels — force reading from grid
            "hide_ticks": level >= 6,
            # L8+: ask for discriminant or root product instead of vertex
            "hard_qtype": "roots_product" if level >= 8 else ("axis_of_sym" if level >= 6 else "vertex"),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["h_range"] + cfg["k_range"] + cfg["a_max"]

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        # Vertex (h, k)
        h = rng.randint(-cfg["h_range"], cfg["h_range"])
        k = rng.randint(-cfg["k_range"], cfg["k_range"])
        # Leading coefficient a
        a_abs = rng.randint(1, cfg["a_max"])
        sign = 1
        if cfg["allow_negative_a"] and rng.random() < 0.5:
            sign = -1
        a = sign * a_abs

        # Ask type: vertex_x, vertex_y, axis, or (if roots real) smaller root.
        # At L6+ add axis_of_sym; at L8+ add roots_product (both need reading
        # the curve to compute, not just spotting the vertex dot).
        # 2026-05-03 (M5 Vieta extension): add `roots_sum` and `discriminant`
        # asks alongside roots_product so the env mirrors reference QF-T10
        # (Vieta x_1+x_2 = -b/a, x_1*x_2 = c/a) and QF-T6 (discriminant).
        ask_options = ["vertex_x", "vertex_y", "axis"]
        disc_pos = (a > 0 and k <= 0) or (a < 0 and k >= 0)
        if disc_pos and abs(k) >= 1:
            ask_options.append("smaller_root")
        # Engage hard_qtype schedule
        hq = cfg.get("hard_qtype", "vertex")
        if hq == "axis_of_sym":
            ask = rng.choice(["axis"] + (["smaller_root"] if "smaller_root" in ask_options else []))
        elif hq == "roots_product" and disc_pos and abs(k) >= 1:
            # Mix in roots_sum and discriminant so Vieta coverage is more than
            # just product. roots_sum = 2h (always integer); discriminant
            # for y = a(x-h)²+k is sign-determined by sign(-k/a).
            ask = rng.choice(["roots_product", "roots_sum",
                              "discriminant_count"])
            ask_options.append("roots_product")
            ask_options.append("roots_sum")
            ask_options.append("discriminant_count")
        else:
            ask = rng.choice(ask_options)

        if ask == "vertex_x":
            answer = str(h)
            q = "What is the x-coordinate of the vertex of the parabola shown?"
        elif ask == "vertex_y":
            answer = str(k)
            q = "What is the y-coordinate of the vertex of the parabola shown?"
        elif ask == "axis":
            answer = str(h)
            q = ("What is the value of x for the axis of symmetry of the "
                 "parabola shown? Give an integer value.")
        elif ask == "roots_product":
            # Product of roots = h² - k/a (Vieta's for y = a(x-h)²+k = 0)
            # => x = h ± √(-k/a); product = h² - (-k/a) = h² + k/a
            val = -k / a
            if val < 0:
                return None
            root_offset = math.sqrt(val)
            r1 = h + root_offset
            r2 = h - root_offset
            prod = r1 * r2
            if abs(prod - round(prod)) > 1e-6:
                return None
            answer = str(int(round(prod)))
            q = ("What is the product of the two x-intercepts of the parabola "
                 "shown? Give an integer value.")
        elif ask == "roots_sum":
            # By symmetry / Vieta: x_1 + x_2 = 2h.
            val = -k / a
            if val < 0:
                return None
            root_offset = math.sqrt(val)
            if abs(root_offset - round(root_offset)) > 1e-6:
                return None
            answer = str(int(2 * h))
            q = ("What is the SUM of the two x-intercepts of the parabola "
                 "shown? Give an integer value.")
        elif ask == "discriminant_count":
            # Count real roots: # = 2 if disc>0, 1 if disc=0, 0 if disc<0.
            val = -k / a
            if val > 1e-6:
                answer = "2"
            elif val < -1e-6:
                answer = "0"
            else:
                answer = "1"
            q = ("How many real x-intercepts does the parabola shown have? "
                 "Give an integer (0, 1, or 2).")
        else:
            # smaller root: x = h +/- sqrt(-k/a)
            val = -k / a
            if val < 0:
                return None
            root_offset = math.sqrt(val)
            if abs(root_offset - round(root_offset)) > 1e-6:
                return None
            rt = int(round(root_offset))
            smaller = h - rt
            answer = str(smaller)
            q = ("What is the smaller of the two x-intercepts of the parabola "
                 "shown? Give an integer value.")

        # Axis window
        pad = cfg["axis_pad"]
        x_min = h - 6 - pad
        x_max = h + 6 + pad
        y_span = max(6 + cfg["k_range"], 8)
        y_min = min(k - y_span, -6)
        y_max = max(k + y_span, 6)

        image = self._render(a, h, k, cfg, x_min, x_max, y_min, y_max)
        return q, answer, image

    def _render(self, a, h, k, cfg, x_min, x_max, y_min, y_max) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        fig, ax = plt.subplots(figsize=(6.2 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        if cfg["show_grid"]:
            ax.grid(True, which="major", alpha=0.3, linestyle="--")

        ax.axhline(0, color="#333", linewidth=1.2, zorder=2)
        ax.axvline(0, color="#333", linewidth=1.2, zorder=2)

        # Parabola
        xs = np.linspace(x_min, x_max, 400)
        ys = a * (xs - h) ** 2 + k
        mask = (ys >= y_min - 1) & (ys <= y_max + 1)
        ax.plot(xs[mask], ys[mask], color=palette[0 % len(palette)],
                linewidth=2.4, zorder=3)

        if cfg["show_vertex_dot"]:
            # Show the vertex point as a dot but WITHOUT its (h,k) coord label
            # (the coord label is the answer — OCR would let model bypass
            # actually reading the graph). Dot alone is enough scaffolding
            # at L0.
            ax.plot([h], [k], "o", color=palette[1 % len(palette)],
                    markersize=7, zorder=4,
                    markeredgecolor="#000", markeredgewidth=0.8)
            ax.annotate("vertex", (h, k),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=fs - 1)

        ax.set_xticks(range(int(x_min) + 1, int(x_max) + 1))
        ax.set_yticks(range(int(y_min) + 1, int(y_max) + 1))
        ax.tick_params(labelsize=fs - 2)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x", fontsize=fs + 1)
        ax.set_ylabel("y", fontsize=fs + 1)
        ax.set_title("Parabola y = a(x - h)² + k", fontsize=fs + 1)
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = ParabolaVertexQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[parabola_vertex L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"parabola_vertex_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[parabola_vertex L{level} s{s}] saved | A={env._answer}")
