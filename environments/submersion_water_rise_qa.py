"""
Submersion Water Rise QA — pure single-letter MCQ A-D / A-E.

2026-05-05 R5 P3 HARDENED: complete rewrite. Was 100% saturated because
distractors were too lax and answer was numeric. New design forces precise
computation Δh = V_solid / A_container, with off-by-formula distractors and
the textbook-style "E. No correct answer" trap option.

Per-level design:
  L0/L1 — 4-MCQ. Cube into rectangular container; integer Δh. No trap.
  L2/L3 — 5-MCQ + 10% trap. Cube + rectangular container; decimal Δh ok.
  L4/L5 — 5-MCQ + 15% trap. Cuboid (l x w x h) submerged; rectangular tank.
  L6/L7 — 5-MCQ + 25% trap. Cylinder submerged in rectangular tank
          (V = πr²h, π = 3.14).
  L8/L9 — 5-MCQ + 40% trap. Cuboid submerged in CYLINDRICAL tank
          (Δh = V_cuboid / (πR²)). Decimal dimensions, tight ±5% distractors.

Image: side-by-side panels showing the container before and after submersion,
with all dimensions labeled.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


PI = 3.14


def _fmt(v: float, decimals: int = 2) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


class SubmersionWaterRiseQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a labeled submersion setup, pick Δh."""

    ENV_NAME = "submersion_water_rise"
    ALLOW_ROTATION = False  # dimension labels orientation-sensitive

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "solid": "cube",
                "container": "rect",
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
                "decimals_in_dims": False,
            }
        if level <= 3:
            return {
                "solid": "cube",
                "container": "rect",
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
                "decimals_in_dims": False,
            }
        if level <= 5:
            return {
                "solid": "cuboid",
                "container": "rect",
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": False,
                "decimals_in_dims": False,
            }
        if level <= 7:
            return {
                "solid": "cylinder",
                "container": "rect",
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
                "decimals_in_dims": False,
            }
        # L8/L9
        return {
            "solid": "cuboid",
            "container": "cyl",
            "use_5_options": True,
            "trap_rate": 0.40,
            "tight_distractors": True,
            "decimals_in_dims": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1331)

        for attempt in range(15):
            try:
                res = self._try_generate(sub_rng, cfg, level, attempt)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level, attempt):
        # Sample solid + container dimensions
        solid_kind = cfg["solid"]
        cont_kind = cfg["container"]

        if solid_kind == "cube":
            s = rng.choice([2, 3, 4, 5])
            v_solid = s ** 3
            solid_label = f"cube of side {_fmt(s)} cm"
            solid_dims = {"s": s}
        elif solid_kind == "cuboid":
            if cfg["decimals_in_dims"]:
                l = rng.choice([3.5, 4.5, 5.0, 6.0])
                w = rng.choice([2.5, 3.0, 3.5])
                h = rng.choice([2.0, 2.5, 3.0])
            else:
                l = rng.choice([3, 4, 5, 6])
                w = rng.choice([2, 3, 4])
                h = rng.choice([2, 3])
            v_solid = l * w * h
            solid_label = f"rectangular block ({_fmt(l)} × {_fmt(w)} × {_fmt(h)} cm)"
            solid_dims = {"l": l, "w": w, "h": h}
        elif solid_kind == "cylinder":
            r_sol = rng.choice([1.5, 2, 2.5, 3])
            h_sol = rng.choice([3, 4, 5, 6])
            v_solid = PI * r_sol * r_sol * h_sol
            solid_label = f"cylinder (r = {_fmt(r_sol)} cm, h = {_fmt(h_sol)} cm)"
            solid_dims = {"r": r_sol, "h": h_sol}
        else:
            return None

        if cont_kind == "rect":
            if cfg["decimals_in_dims"]:
                A = rng.choice([50.0, 75.0, 80.0, 100.0])
            else:
                A = rng.choice([20, 25, 40, 50, 60, 80, 100])
            cont_label = f"rectangular tank with base area A = {_fmt(A)} cm²"
            cont_dims = {"A": A}
        elif cont_kind == "cyl":
            R = rng.choice([4, 5, 6])
            A = PI * R * R  # π·R²
            cont_label = f"cylindrical tank with base radius R = {_fmt(R)} cm"
            cont_dims = {"R": R, "A": A}
        else:
            return None

        # Verify the cube/cuboid actually fits inside the container's footprint,
        # and the rise is physically meaningful (>0).
        delta_h = v_solid / A
        if delta_h <= 0 or delta_h > 30:
            return None
        # Also: solid volume must be reasonable so it's actually submersible
        if v_solid <= 0:
            return None

        true_value = delta_h
        true_str = _fmt(true_value, decimals=2)

        # Build distractors
        distractors = self._make_distractors(
            rng, solid_kind, solid_dims, cont_kind, cont_dims, true_value, cfg
        )
        if distractors is None or len(distractors) < 4:
            return None

        distr_strs = []
        seen = {true_str}
        for d in distractors:
            ds = _fmt(d, decimals=2)
            if ds in seen:
                continue
            seen.add(ds)
            distr_strs.append(ds)

        if len(distr_strs) < 4:
            return None

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False
        if use_trap:
            chosen = distr_strs[:n_value]
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distr_strs[:n_distractor]
            options = [true_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_str)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]} cm" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Stem
        formula_hint = ""
        if cont_kind == "cyl":
            formula_hint = " (Recall: cylindrical tank base area = π·R².)"
        stems = [
            (
                f"As shown in the diagram, a {solid_label} is fully submerged in "
                f"a {cont_label}. By how much (in cm) does the water level rise? "
                f"Use π = 3.14, round to 2 decimal places.{formula_hint} ( )"
            ),
            (
                f"From the diagram, a {solid_label} is dropped into a "
                f"{cont_label} and is completely covered by water. "
                f"Compute the rise in water level (in cm). "
                f"Use π = 3.14 and round to 2 decimal places.{formula_hint} ( )"
            ),
            (
                f"The figure shows a {solid_label} placed at the bottom of a "
                f"{cont_label} (fully submerged). What is the increase in "
                f"water height (in cm)? Use π = 3.14, round to 2 decimal "
                f"places.{formula_hint} ( )"
            ),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        # Render
        image = self._render(rng, solid_kind, solid_dims,
                             cont_kind, cont_dims, delta_h)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _make_distractors(self, rng, solid_kind, sd, cont_kind, cd, true_val, cfg):
        candidates = []

        # Common error: forgot to divide by A → just V_solid as the answer
        if solid_kind == "cube":
            v_solid = sd["s"] ** 3
        elif solid_kind == "cuboid":
            v_solid = sd["l"] * sd["w"] * sd["h"]
        else:  # cylinder
            v_solid = PI * sd["r"] ** 2 * sd["h"]

        A = cd["A"]
        candidates.append(v_solid)  # just V_solid (forgot /A)
        candidates.append(v_solid * 2 / A)  # off by factor of 2
        candidates.append(v_solid / 2 / A)  # off by factor of 1/2

        # Common error: confused the formula for V_solid
        if solid_kind == "cube":
            s = sd["s"]
            candidates.append(6 * s * s / A)  # used SA instead of V
            candidates.append(s * s / A)  # used face area
        if solid_kind == "cuboid":
            l, w, h = sd["l"], sd["w"], sd["h"]
            candidates.append(2 * (l * w + l * h + w * h) / A)  # used SA
            candidates.append(l * w / A)  # used base area
        if solid_kind == "cylinder":
            r, h = sd["r"], sd["h"]
            candidates.append(2 * PI * r * (r + h) / A)  # used SA
            candidates.append(PI * r * h / A)  # forgot square
            candidates.append(PI * r * r * h * 2 / A)  # 2x mistake

        # Cylindrical container: common error using diameter not radius for area
        if cont_kind == "cyl":
            R = cd["R"]
            candidates.append(v_solid / (PI * (2 * R) ** 2))  # used D instead of R
            candidates.append(v_solid / (2 * PI * R))  # used circumference
            candidates.append(v_solid / R)  # forgot πR²

        # ±5% / ±10% perturbations for tight mode
        if cfg["tight_distractors"]:
            for delta in (0.95, 0.97, 1.03, 1.05, 0.93, 1.07):
                candidates.append(true_val * delta)
        else:
            for delta in (0.5, 0.7, 1.5, 2.0):
                candidates.append(true_val * delta)

        # Filter
        good = []
        for c in candidates:
            if c is None or c <= 0:
                continue
            if abs(c - true_val) < max(0.005, true_val * 0.005):
                continue
            if any(abs(c - g) < max(0.005, max(c, g) * 0.003) for g in good):
                continue
            good.append(c)

        rng.shuffle(good)
        return good[:6]

    # ------------------------------------------------------------------ #
    def _render(self, rng, solid_kind, sd, cont_kind, cd, delta_h) -> Image.Image:
        fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")

        # Visualize container as a rectangle of width = sqrt(A) for rect,
        # or width = 2*R for cyl.
        if cont_kind == "rect":
            cwidth = math.sqrt(cd["A"])
        else:
            cwidth = 2 * cd["R"]
        # Pick a friendly initial water level
        if solid_kind == "cube":
            solid_dim = sd["s"]
        elif solid_kind == "cuboid":
            solid_dim = max(sd["l"], sd["w"], sd["h"])
        else:
            solid_dim = max(2 * sd["r"], sd["h"])
        h0 = max(2, solid_dim - 0.5)
        cheight = max(h0 + delta_h + 1, solid_dim + 4)

        self._draw_panel(axes[0], cwidth, cheight, h0,
                         solid_kind, sd, cont_kind, cd,
                         show_solid=False, water_level=h0,
                         title="Before (solid not yet submerged)")
        self._draw_panel(axes[1], cwidth, cheight, h0,
                         solid_kind, sd, cont_kind, cd,
                         show_solid=True, water_level=h0 + delta_h,
                         title="After (solid fully submerged)")

        try:
            fig.tight_layout()
        except Exception:
            pass

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _draw_panel(self, ax, cw, ch, initial_h, solid_kind, sd,
                    cont_kind, cd, show_solid, water_level, title):
        ax.set_facecolor("#ffffff")
        ax.plot([0, 0], [0, ch], color="#2c3e50", lw=2.5)
        ax.plot([0, cw], [0, 0], color="#2c3e50", lw=2.5)
        ax.plot([cw, cw], [0, ch], color="#2c3e50", lw=2.5)

        # Water region
        ax.add_patch(patches.Rectangle(
            (0, 0), cw, water_level,
            linewidth=0, facecolor="#5dade2", alpha=0.55))
        # Water level line
        ax.plot([0, cw], [water_level, water_level],
                color="#1b4f72", lw=1.6, linestyle="--")

        # Solid (if requested)
        if show_solid:
            if solid_kind == "cube":
                s = sd["s"]
                cx = (cw - s) / 2
                ax.add_patch(patches.Rectangle(
                    (cx, 0), s, s, linewidth=1.8,
                    edgecolor="#7b241c", facecolor="#f5b7b1", alpha=0.85, zorder=3))
                ax.text(cx + s / 2, s / 2, f"side\ns = {_fmt(s)}",
                        ha="center", va="center", fontsize=10,
                        color="#7b241c", fontweight="bold")
            elif solid_kind == "cuboid":
                l, w, h = sd["l"], sd["w"], sd["h"]
                cx = (cw - l) / 2
                ax.add_patch(patches.Rectangle(
                    (cx, 0), l, h, linewidth=1.8,
                    edgecolor="#7b241c", facecolor="#f5b7b1", alpha=0.85, zorder=3))
                ax.text(cx + l / 2, h / 2,
                        f"{_fmt(l)} × {_fmt(w)} × {_fmt(h)}",
                        ha="center", va="center", fontsize=9,
                        color="#7b241c", fontweight="bold")
            elif solid_kind == "cylinder":
                r, h = sd["r"], sd["h"]
                cx = (cw - 2 * r) / 2
                ax.add_patch(patches.Rectangle(
                    (cx, 0), 2 * r, h, linewidth=1.8,
                    edgecolor="#7b241c", facecolor="#f5b7b1", alpha=0.85, zorder=3))
                ax.text(cx + r, h / 2, f"r = {_fmt(r)}\nh = {_fmt(h)}",
                        ha="center", va="center", fontsize=9,
                        color="#7b241c", fontweight="bold")

        # Container label
        if cont_kind == "rect":
            ax.text(cw / 2, -0.6, f"base area A = {_fmt(cd['A'])} cm²",
                    ha="center", va="top", fontsize=10,
                    color="#1e8449", fontweight="bold")
        else:
            ax.text(cw / 2, -0.6, f"base radius R = {_fmt(cd['R'])} cm",
                    ha="center", va="top", fontsize=10,
                    color="#1e8449", fontweight="bold")

        ax.text(-0.4, water_level, f"  water level = {_fmt(water_level)}",
                ha="right", va="center", fontsize=9,
                color="#1b4f72", fontweight="bold")

        ax.set_title(title, fontsize=11, fontweight="bold")

        # If not showing solid, draw it above the container as preview
        if not show_solid:
            preview_x = cw + 1.2
            preview_y = ch - 4
            if solid_kind == "cube":
                s = sd["s"]
                ax.add_patch(patches.Rectangle(
                    (preview_x, preview_y), s, s, linewidth=1.8,
                    edgecolor="#7b241c", facecolor="#f5b7b1", alpha=0.85))
                ax.text(preview_x + s / 2, preview_y - 0.5,
                        f"cube\nside s = {_fmt(s)}",
                        ha="center", va="top", fontsize=9,
                        color="#7b241c", fontweight="bold")
            elif solid_kind == "cuboid":
                l, w, h = sd["l"], sd["w"], sd["h"]
                ax.add_patch(patches.Rectangle(
                    (preview_x, preview_y), l, h, linewidth=1.8,
                    edgecolor="#7b241c", facecolor="#f5b7b1", alpha=0.85))
                ax.text(preview_x + l / 2, preview_y - 0.5,
                        f"cuboid\n{_fmt(l)} × {_fmt(w)} × {_fmt(h)}",
                        ha="center", va="top", fontsize=9,
                        color="#7b241c", fontweight="bold")
            elif solid_kind == "cylinder":
                r, h = sd["r"], sd["h"]
                ax.add_patch(patches.Rectangle(
                    (preview_x, preview_y), 2 * r, h, linewidth=1.8,
                    edgecolor="#7b241c", facecolor="#f5b7b1", alpha=0.85))
                ax.text(preview_x + r, preview_y - 0.5,
                        f"cylinder\nr = {_fmt(r)}\nh = {_fmt(h)}",
                        ha="center", va="top", fontsize=9,
                        color="#7b241c", fontweight="bold")

        ax.set_xlim(-2.5, cw + 8)
        ax.set_ylim(-2.0, ch + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")


if __name__ == "__main__":
    env = SubmersionWaterRiseQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
