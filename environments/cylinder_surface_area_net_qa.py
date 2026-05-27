"""
Cylinder Surface Area Net QA — pure single-letter MCQ A-D / A-E.

2026-05-05 R5 P3 HARDENED: complete rewrite. Was 97.5% saturated. New
design forces the model to:

  1. Read r and h from the labeled cylinder + its 2D net.
  2. Apply the correct SA formula:
       - total SA          = 2πr² + 2πrh
       - lateral SA        = 2πrh        (rectangle area only)
       - open-top SA       = πr² + 2πrh  (one base + lateral)
       - lateral-rect width = 2πr        (circumference)
       - back out r from rect width:  r = width / (2π)
  3. Compute with π = 3.14 (NOT exact π).
  4. Pick from 4 tight numerical distractors with off-by-formula traps.

Per-level design:
  L0/L1 — 4-MCQ. Just total SA, integer r, h. No trap.
  L2/L3 — 5-MCQ + 10% trap. Adds lateral SA. Rect-width question style.
  L4/L5 — 5-MCQ + 15% trap. Adds open-top SA (one base + lateral).
  L6/L7 — 5-MCQ + 25% trap. Decimal r, h. All four question types.
  L8/L9 — 5-MCQ + 40% trap. Decimal dims, "back-out r from width" question
          (if rect width = W, then r = W/(2·3.14)) + tight ±5% distractors.

Image: cylinder side-by-side with its 2D net (rectangle + 2 circles).
Labels: r and h on the cylinder; net may show width or be unlabeled.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


PI = 3.14
_TITLES = [
    "Cylinder and its net",
    "Cylinder with 2D net",
    "Surface net of a cylinder",
    "Cylinder unfolded view",
]


def _fmt(v: float, decimals: int = 2) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


class CylinderSurfaceAreaNetQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a labeled cylinder + net."""

    ENV_NAME = "cylinder_surface_area_net"
    ALLOW_ROTATION = False  # dimension labels orientation-sensitive

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "qtypes": ["total_sa"],
                "decimals_in_dims": False,
                "use_5_options": False,
                "trap_rate": 0.0,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "qtypes": ["total_sa", "lateral_sa", "rect_width"],
                "decimals_in_dims": False,
                "use_5_options": True,
                "trap_rate": 0.10,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "qtypes": ["total_sa", "lateral_sa", "open_top_sa"],
                "decimals_in_dims": False,
                "use_5_options": True,
                "trap_rate": 0.15,
                "tight_distractors": False,
            }
        if level <= 7:
            return {
                "qtypes": ["total_sa", "lateral_sa", "open_top_sa", "rect_width"],
                "decimals_in_dims": True,
                "use_5_options": True,
                "trap_rate": 0.25,
                "tight_distractors": True,
            }
        # L8/L9
        return {
            "qtypes": ["total_sa", "open_top_sa", "back_out_r"],
            "decimals_in_dims": True,
            "use_5_options": True,
            "trap_rate": 0.40,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1019)

        for attempt in range(15):
            try:
                res = self._try_generate(sub_rng, cfg, level, attempt)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level, attempt):
        if cfg["decimals_in_dims"]:
            r = rng.choice([2.5, 3, 3.5, 4, 5])
            h = rng.choice([4, 5.5, 6, 7.5, 8])
        else:
            r = rng.choice([2, 3, 4, 5])
            h = rng.choice([3, 4, 5, 6, 7, 8])

        qtype = rng.choice(cfg["qtypes"])

        # Compute true value based on qtype
        if qtype == "total_sa":
            true_value = 2 * PI * r * r + 2 * PI * r * h
            unit = "cm²"
            decimals = 2
        elif qtype == "lateral_sa":
            true_value = 2 * PI * r * h
            unit = "cm²"
            decimals = 2
        elif qtype == "open_top_sa":
            # one base + lateral (open at top)
            true_value = PI * r * r + 2 * PI * r * h
            unit = "cm²"
            decimals = 2
        elif qtype == "rect_width":
            true_value = 2 * PI * r  # circumference
            unit = "cm"
            decimals = 2
        elif qtype == "back_out_r":
            # Show only width on net; ask for r.
            # We'll set the question so the student computes r = W/(2π).
            true_value = r
            unit = "cm"
            decimals = 2
        else:
            return None

        if true_value <= 0:
            return None

        # Build distractors
        distractors = self._make_distractors(rng, qtype, r, h, true_value, cfg)
        if distractors is None or len(distractors) < 4:
            return None

        true_str = _fmt(true_value, decimals=decimals)
        distr_strs = []
        seen = {true_str}
        for d in distractors:
            ds = _fmt(d, decimals=decimals)
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
            f"{opt_letters[i]}. {options[i]} {unit}" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Stem per qtype
        stems_map = {
            "total_sa": [
                ("As shown in the diagram, the figure on the left is a "
                 "cylinder and on the right is its unfolded net (rectangle "
                 "+ two circles). Read r and h from the figure and compute "
                 "the TOTAL surface area of the cylinder. "
                 "Use π = 3.14, round to 2 decimal places. ( )"),
                ("From the diagram, identify the cylinder's r and h. "
                 "Compute its TOTAL surface area (lateral + 2 caps). "
                 "Use π = 3.14, round to 2 decimal places. ( )"),
            ],
            "lateral_sa": [
                ("The figure on the left shows a cylinder; on the right is "
                 "its 2D net. Compute ONLY the LATERAL surface area "
                 "(the rectangle area, no caps). Use π = 3.14, round to 2 "
                 "decimal places. ( )"),
                ("Read r and h from the diagram. The lateral surface area "
                 "of a cylinder equals 2πrh. Compute it (cm²). "
                 "Use π = 3.14, round to 2 decimal places. ( )"),
            ],
            "open_top_sa": [
                ("As shown, the cylinder is an OPEN-TOP container (no top "
                 "cap). Compute its outer surface area (one base circle "
                 "+ lateral). Use π = 3.14, round to 2 decimal places. ( )"),
                ("The figure shows an open-top cylindrical container "
                 "(the top is open). Compute the surface area of its "
                 "wall + bottom (one circular base + lateral surface). "
                 "Use π = 3.14, round to 2 decimal places. ( )"),
            ],
            "rect_width": [
                ("The right diagram shows the unfolded lateral surface "
                 "(rectangle) of the cylinder. Compute the WIDTH of the "
                 "rectangle (which equals the circumference of the "
                 "cylinder's base). Use π = 3.14, round to 2 decimal "
                 "places. ( )"),
                ("In the unfolded net (right), the rectangle's width "
                 "matches the base circumference. Compute that width "
                 "(cm). Use π = 3.14, round to 2 decimal places. ( )"),
            ],
            "back_out_r": [
                ("The right diagram shows the unfolded lateral surface "
                 "(rectangle). The rectangle's width equals the circumference "
                 "of the cylinder's base, so r = width / (2π). "
                 "Read width from the diagram and compute r in cm. "
                 "Use π = 3.14, round to 2 decimal places. ( )"),
                ("In the unfolded net, the rectangle width equals 2πr. "
                 "Read the labeled width and back-solve for r. "
                 "Use π = 3.14, round to 2 decimal places. ( )"),
            ],
        }
        stems = stems_map[qtype]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

        # Render — for back_out_r, we show width on net BUT NOT r on cylinder
        image = self._render(rng, r, h, qtype)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _make_distractors(self, rng, qtype, r, h, true_value, cfg):
        candidates = []

        # Common errors per qtype
        if qtype == "total_sa":
            # Forgot one cap
            candidates.append(PI * r * r + 2 * PI * r * h)
            # Forgot lateral
            candidates.append(2 * PI * r * r)
            # Used 1 cap only
            candidates.append(2 * PI * r * h + PI * r * r)
            # Used radius squared instead of r in lateral
            candidates.append(2 * PI * r * r + 2 * PI * r * r * h)
            # 2x mistake
            candidates.append(2 * (2 * PI * r * r + 2 * PI * r * h))
        elif qtype == "lateral_sa":
            # Used total SA
            candidates.append(2 * PI * r * r + 2 * PI * r * h)
            # Used 1 cap only
            candidates.append(PI * r * r)
            # 2x mistake
            candidates.append(4 * PI * r * h)
            # Forgot to multiply by 2
            candidates.append(PI * r * h)
            # Used h^2
            candidates.append(2 * PI * r * h * h)
        elif qtype == "open_top_sa":
            # Used total SA
            candidates.append(2 * PI * r * r + 2 * PI * r * h)
            # Used both caps
            candidates.append(2 * PI * r * r + 2 * PI * r * h - PI * r * r)
            # Forgot the base
            candidates.append(2 * PI * r * h)
            # 2 bases + lateral
            candidates.append(2 * PI * r * r + 2 * PI * r * h)
            # base only
            candidates.append(PI * r * r)
        elif qtype == "rect_width":
            # Used diameter only
            candidates.append(2 * r)
            # Used pi*r (forgot 2)
            candidates.append(PI * r)
            # Used pi*r^2 (area of circle)
            candidates.append(PI * r * r)
            # Used 4*pi*r (×2 mistake)
            candidates.append(4 * PI * r)
        elif qtype == "back_out_r":
            # Forgot to divide
            candidates.append(2 * PI * r)
            # Divided by pi only
            candidates.append(2 * r)
            # Divided by 2 only
            candidates.append(PI * r)
            # Used wrong formula r = W/π
            candidates.append(2 * r)

        # Generic ±5%/10% perturbations
        if cfg["tight_distractors"]:
            for delta in (0.95, 0.97, 1.03, 1.05, 0.93, 1.07):
                candidates.append(true_value * delta)
        else:
            for delta in (0.5, 0.7, 1.5, 2.0):
                candidates.append(true_value * delta)

        # Filter
        good = []
        for c in candidates:
            if c is None or c <= 0:
                continue
            if abs(c - true_value) < max(0.005, true_value * 0.005):
                continue
            if any(abs(c - g) < max(0.005, max(c, g) * 0.003) for g in good):
                continue
            good.append(c)

        rng.shuffle(good)
        return good[:6]

    # ------------------------------------------------------------------ #
    # Rendering: cylinder + 2D net
    # ------------------------------------------------------------------ #
    def _render(self, rng, r, h, qtype) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, axes = plt.subplots(1, 2, figsize=(11 * sc, 6 * sc),
                                 gridspec_kw={"width_ratios": [1, 1.3]})
        fig.patch.set_facecolor(style["bg_color"])

        # --- LEFT: 2D-projected cylinder --- #
        ax3d = axes[0]
        ax3d.set_facecolor(style["bg_color"])
        ax3d.set_aspect("equal")
        ax3d.axis("off")

        palette = style["palette"]
        body_color = palette[0]
        top_color = palette[1]
        cx = 0.0
        rx = float(r)
        ry = rx * 0.32
        cy_bot = 0.0
        cy_top = float(h)

        ax3d.add_patch(Polygon(
            [(cx - rx, cy_bot), (cx + rx, cy_bot),
             (cx + rx, cy_top), (cx - rx, cy_top)],
            closed=True, facecolor=body_color, alpha=0.55,
            edgecolor="black", lw=1.6))
        theta = np.linspace(0, 2 * math.pi, 120)
        top_x = cx + rx * np.cos(theta)
        top_y = cy_top + ry * np.sin(theta)
        # For open_top_sa, draw top opening as dashed (no fill)
        if qtype == "open_top_sa":
            ax3d.plot(top_x, top_y, color="black", lw=1.4, linestyle="--")
        else:
            ax3d.fill(top_x, top_y, color=top_color, edgecolor="black", lw=1.6,
                      alpha=0.85)
        # Bottom outline
        theta_f = np.linspace(math.pi, 2 * math.pi, 60)
        theta_b = np.linspace(0, math.pi, 60)
        ax3d.plot(cx + rx * np.cos(theta_f), cy_bot + ry * np.sin(theta_f),
                  color="black", lw=1.6)
        ax3d.plot(cx + rx * np.cos(theta_b), cy_bot + ry * np.sin(theta_b),
                  color="black", lw=1.0, linestyle="--", alpha=0.55)

        # Label r unless qtype is back_out_r
        if qtype != "back_out_r":
            ax3d.annotate(
                "", xy=(cx + rx, cy_top), xytext=(cx, cy_top),
                arrowprops=dict(arrowstyle="->", color="#b00000", lw=2))
            ax3d.text(cx + rx * 0.5, cy_top + 0.35, f"r = {_fmt(r)}",
                      fontsize=13, fontweight="bold", color="#b00000")
        # Always label h
        ax3d.annotate(
            "", xy=(cx - rx - 0.5, cy_top),
            xytext=(cx - rx - 0.5, cy_bot),
            arrowprops=dict(arrowstyle="<->", color="#1b3b6f", lw=2))
        ax3d.text(cx - rx - 0.8, (cy_top + cy_bot) / 2, f"h = {_fmt(h)}",
                  fontsize=13, fontweight="bold", color="#1b3b6f",
                  ha="right", va="center")

        ax3d.set_xlim(cx - rx - 2.5, cx + rx + 1.5)
        ax3d.set_ylim(cy_bot - ry - 1.4, cy_top + ry + 1.4)
        ax3d.set_title("Cylinder",
                       fontsize=style["font_size_base"] + 2,
                       fontweight="bold", pad=8)

        # --- RIGHT: net (rectangle + caps) --- #
        ax2 = axes[1]
        ax2.set_facecolor(style["bg_color"])
        ax2.set_aspect("equal")
        ax2.axis("off")

        # Scale net so rectangle visually fits
        scale = 1.0 / math.pi
        rect_w = 2 * math.pi * r * scale  # = 2r visually
        rect_h = float(h)
        x0 = 0.0
        y0 = 0.0
        ax2.add_patch(Polygon(
            [(x0, y0), (x0 + rect_w, y0),
             (x0 + rect_w, y0 + rect_h), (x0, y0 + rect_h)],
            closed=True, facecolor=palette[2], alpha=0.45,
            edgecolor="black", lw=1.6))
        cr = float(r)
        # For open_top_sa, only show ONE cap (the bottom)
        if qtype != "open_top_sa":
            top_c = (x0 + rect_w / 2.0, y0 + rect_h + cr + 0.3)
            ax2.add_patch(Circle(top_c, cr, facecolor=palette[3], alpha=0.6,
                                 edgecolor="black", lw=1.6))
        bot_c = (x0 + rect_w / 2.0, y0 - cr - 0.3)
        ax2.add_patch(Circle(bot_c, cr, facecolor=palette[3], alpha=0.6,
                             edgecolor="black", lw=1.6))

        # Net labels
        if qtype == "back_out_r":
            # Label width with NUMERIC value; that's all the student gets
            actual_width = 2 * PI * r
            ax2.annotate(
                "", xy=(x0, y0 - 0.5), xytext=(x0 + rect_w, y0 - 0.5),
                arrowprops=dict(arrowstyle="<->", color="#b00000", lw=2))
            ax2.text(x0 + rect_w / 2.0, y0 - 1.0,
                     f"width = {_fmt(actual_width)} cm",
                     fontsize=12, fontweight="bold",
                     color="#b00000", ha="center", va="top")
            ax2.annotate(
                "", xy=(x0 + rect_w + 0.4, y0),
                xytext=(x0 + rect_w + 0.4, y0 + rect_h),
                arrowprops=dict(arrowstyle="<->", color="#1b3b6f", lw=2))
            ax2.text(x0 + rect_w + 0.6, y0 + rect_h / 2,
                     f"h = {_fmt(h)}", fontsize=12, fontweight="bold",
                     color="#1b3b6f", ha="left", va="center")
        else:
            # Generic labels indicating layout
            ax2.text(x0 + rect_w / 2.0, y0 - 0.8, "width = 2πr",
                     fontsize=11, ha="center", va="top", color="#444")
            ax2.text(x0 + rect_w + 0.4, y0 + rect_h / 2,
                     f"h = {_fmt(h)}", fontsize=11, color="#1b3b6f",
                     ha="left", va="center", fontweight="bold")

        # Dashed fold lines
        for yv in [y0, y0 + rect_h]:
            ax2.plot([x0, x0 + rect_w], [yv, yv],
                     color="black", lw=0.8, linestyle=":")

        pad = 1.5
        x_min = x0 - pad
        x_max = x0 + rect_w + pad
        y_min = bot_c[1] - cr - pad
        if qtype == "open_top_sa":
            y_max = y0 + rect_h + pad
        else:
            y_max = y0 + rect_h + cr + pad + 0.5
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_title("Unfolded net",
                      fontsize=style["font_size_base"] + 2,
                      fontweight="bold", pad=8)

        fig.suptitle(rng.choice(_TITLES),
                     fontsize=style["font_size_base"] + 4,
                     fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=130)


if __name__ == "__main__":
    env = CylinderSurfaceAreaNetQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
