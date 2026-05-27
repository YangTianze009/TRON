"""
Prism Volume Warmup QA environment.

Target regression: Solid reasoning Geometry -2.52. Train the model on
clean "given dimensions, compute volume / surface area" tasks for
rectangular prisms, triangular prisms, cylinders, and compound solids.

Two independent difficulty axes scale with level:
  * solid complexity  (rectangular -> triangular prism -> cylinder ->
    compound solid)
  * dimension magnitude range

All questions ask for a single integer answer, format constant.
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

def _iso(x, y, z):
    """Simple isometric projection (same as in other env files)."""
    sx = (x - y) * math.cos(math.radians(30))
    sy = (x + y) * math.sin(math.radians(30)) + z
    return sx, sy

class PrismVolumeWarmupQA(StandaloneVisualEnv):
    ENV_NAME = "prism_volume_warmup"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            kind_pool = ["rect_volume"]
        elif level <= 3:
            kind_pool = ["rect_volume", "tri_prism_volume"]
        elif level <= 5:
            kind_pool = ["tri_prism_volume", "cylinder_volume"]
        elif level <= 7:
            kind_pool = ["cylinder_surface", "compound_volume"]
        else:
            kind_pool = ["compound_volume"]
        # Dimension range grows with level
        dim_lo = 2
        dim_hi = 4 + level               # 4 -> 13
        return {
            "kind_pool": kind_pool,
            "dim_lo": dim_lo,
            "dim_hi": dim_hi,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        for _ in range(20):
            result = self._try_generate(parameter)
            if result is not None:
                return result
        return None

    def _try_generate(self, parameter: Dict):
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        kind = rng.choice(cfg["kind_pool"])

        if kind == "rect_volume":
            primary = 1
            a = rng.randint(cfg["dim_lo"], cfg["dim_hi"])
            b = rng.randint(cfg["dim_lo"], cfg["dim_hi"])
            c = rng.randint(cfg["dim_lo"], cfg["dim_hi"])
            volume = a * b * c
            img = self._draw_rect_prism(a, b, c)
            q = ("The image shows a rectangular prism with the labelled "
                 "edge lengths. What is the volume of the prism? "
                 "Answer with a single integer.")
            answer = str(volume)

        elif kind == "tri_prism_volume":
            primary = 2
            # Right triangle cross section with legs (base, height),
            # length is the depth of the prism.
            base = rng.randint(cfg["dim_lo"], cfg["dim_hi"])
            height = rng.randint(cfg["dim_lo"], cfg["dim_hi"])
            length = rng.randint(cfg["dim_lo"], cfg["dim_hi"])
            # Use even values so (1/2 * base * height) is an integer
            if (base * height) % 2 != 0:
                # force one of them even
                if base % 2 != 0:
                    base += 1
                else:
                    height += 1
            volume = (base * height * length) // 2
            img = self._draw_tri_prism(base, height, length)
            q = ("The image shows a triangular prism. The triangular cross "
                 "section has a right angle with legs labelled 'base' and "
                 "'height'. The depth of the prism is labelled 'length'. "
                 "What is the volume of the prism? Answer with a single "
                 "integer.")
            answer = str(volume)

        elif kind == "cylinder_volume":
            primary = 3
            r = rng.randint(2, cfg["dim_hi"])
            h = rng.randint(2, cfg["dim_hi"])
            # Use V = pi * r^2 * h and round to nearest integer.
            # To keep answers clean, ask the user for volume in terms of pi,
            # actually let's give an integer using pi ~= 3.
            # Cleaner: ask for "the volume divided by pi".
            volume_over_pi = r * r * h
            img = self._draw_cylinder(r, h)
            q = (
                "The image shows a right circular cylinder with radius "
                f"r = {r} and height h = {h}. What is V/pi (the volume "
                "divided by pi)? Answer with a single integer."
            )
            answer = str(volume_over_pi)

        elif kind == "cylinder_surface":
            primary = 4
            r = rng.randint(2, cfg["dim_hi"])
            h = rng.randint(2, cfg["dim_hi"])
            # Total surface area S = 2*pi*r*h + 2*pi*r^2 = 2*pi*r*(h + r)
            # S/pi = 2r(h + r)
            s_over_pi = 2 * r * (h + r)
            img = self._draw_cylinder(r, h)
            q = (
                "The image shows a right circular cylinder with radius "
                f"r = {r} and height h = {h}. What is the total surface "
                "area S divided by pi (S/pi)? Answer with a single integer."
            )
            answer = str(s_over_pi)

        else:  # compound_volume
            primary = 5
            # Compound = rectangular prism with a smaller rectangular block
            # on top (like a step). Volumes add.
            a1 = rng.randint(3, cfg["dim_hi"])
            b1 = rng.randint(3, cfg["dim_hi"])
            c1 = rng.randint(2, cfg["dim_hi"])
            a2 = rng.randint(2, max(3, a1 - 1))
            b2 = rng.randint(2, max(3, b1 - 1))
            c2 = rng.randint(2, cfg["dim_hi"])
            total = a1 * b1 * c1 + a2 * b2 * c2
            img = self._draw_compound_step(a1, b1, c1, a2, b2, c2)
            q = ("The image shows a compound solid made of a larger "
                 "rectangular prism with a smaller rectangular prism on "
                 "top. The labelled dimensions are shown. What is the "
                 "total volume of the compound solid? Answer with a single "
                 "integer.")
            answer = str(total)

        # Sanity clamp
        if int(answer) > 8000:
            return None

        self._primary_complexity_feature = primary
        return q, answer, img

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #
    def _new_axes(self):
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=style["dpi"])
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax, style

    def _draw_rect_prism(self, a, b, c) -> Image.Image:
        """Draw an oblique view of a labelled rectangular prism.

        Uses a flat-front projection: front face is an axis-aligned
        rectangle with width = a, height = c; depth edges go up-right
        at 30 degrees with length proportional to b.
        """
        fig, ax, style = self._new_axes()
        palette = style["palette"]
        front_color = palette[0]
        top_color = palette[1]
        right_color = palette[2]

        # Depth offset for one unit of b
        d = 0.55
        dx = d * math.cos(math.radians(30))
        dy = d * math.sin(math.radians(30))

        # Front face vertices (x, z plane, at y=0)
        f_bl = (0.0, 0.0)           # bottom-left
        f_br = (a,   0.0)           # bottom-right
        f_tr = (a,   c)             # top-right
        f_tl = (0.0, c)             # top-left
        # Back face offset by (b*dx, b*dy)
        ox, oy = b * dx, b * dy
        r_bl = (f_bl[0] + ox, f_bl[1] + oy)
        r_br = (f_br[0] + ox, f_br[1] + oy)
        r_tr = (f_tr[0] + ox, f_tr[1] + oy)
        r_tl = (f_tl[0] + ox, f_tl[1] + oy)

        # Draw back-first: back rectangle (mostly hidden, we draw its
        # visible top edge and right edge via the side faces).
        # Right (depth) side face
        ax.add_patch(Polygon([f_br, r_br, r_tr, f_tr], closed=True,
                             facecolor=right_color, edgecolor="black", lw=1.6))
        # Top face
        ax.add_patch(Polygon([f_tl, f_tr, r_tr, r_tl], closed=True,
                             facecolor=top_color, edgecolor="black", lw=1.6))
        # Front face
        ax.add_patch(Polygon([f_bl, f_br, f_tr, f_tl], closed=True,
                             facecolor=front_color, edgecolor="black", lw=1.6))

        # Dashed hidden back edges (back-left vertical, back-bottom edge)
        hidden_lines = [
            (r_bl, r_br),
            (r_bl, r_tl),
            (f_bl, r_bl),
        ]
        for (p, q) in hidden_lines:
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color="black", lw=1.0, linestyle="--", alpha=0.55)

        # Labels: a on bottom front edge, c on left front edge, b on
        # top-right depth edge
        ax.text((f_bl[0] + f_br[0]) / 2, f_bl[1] - 0.35, f"{a}",
                fontsize=16, fontweight="bold",
                ha="center", va="top", color="#222222")
        ax.text(f_bl[0] - 0.35, (f_bl[1] + f_tl[1]) / 2, f"{c}",
                fontsize=16, fontweight="bold",
                ha="right", va="center", color="#222222")
        ax.text((f_tr[0] + r_tr[0]) / 2 + 0.15,
                (f_tr[1] + r_tr[1]) / 2 + 0.15, f"{b}",
                fontsize=16, fontweight="bold",
                ha="left", va="bottom", color="#222222")

        pts = [f_bl, f_br, f_tr, f_tl, r_bl, r_br, r_tr, r_tl]
        self._autoscale_pts(ax, pts)
        ax.set_title("Rectangular prism",
                     fontsize=14, fontweight="bold", pad=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_tri_prism(self, base, height, length) -> Image.Image:
        """Draw a triangular prism with right-triangle cross-section.

        The right triangle has legs 'base' (horizontal) and 'height'
        (vertical); the prism extends back with depth 'length'.
        """
        fig, ax, style = self._new_axes()
        palette = style["palette"]

        d = 0.6
        depth_x = length * d * math.cos(math.radians(30))
        depth_y = length * d * math.sin(math.radians(30))

        # Front (right) triangle vertices - this is the face facing the viewer
        f0 = (0.0, 0.0)             # right angle corner (origin)
        f1 = (base, 0.0)            # bottom-right
        f2 = (0.0, height)          # top-left (above origin)
        # Back triangle vertices (shifted by (depth_x, depth_y))
        b0 = (f0[0] + depth_x, f0[1] + depth_y)
        b1 = (f1[0] + depth_x, f1[1] + depth_y)
        b2 = (f2[0] + depth_x, f2[1] + depth_y)

        # Bottom face (f0 - f1 - b1 - b0)
        ax.add_patch(Polygon([f0, f1, b1, b0], closed=True,
                             facecolor=palette[2], edgecolor="black", lw=1.6))
        # Back vertical face (f0 - f2 - b2 - b0)
        ax.add_patch(Polygon([f0, f2, b2, b0], closed=True,
                             facecolor=palette[1], edgecolor="black", lw=1.6))
        # Slanted top face (f1 - f2 - b2 - b1)
        ax.add_patch(Polygon([f1, f2, b2, b1], closed=True,
                             facecolor=palette[0], edgecolor="black", lw=1.6))
        # Front triangle
        ax.add_patch(Polygon([f0, f1, f2], closed=True,
                             facecolor=palette[4], edgecolor="black", lw=1.6))

        # Right-angle marker at f0
        rs = min(base, height) * 0.12
        ax.add_patch(Polygon(
            [(f0[0] + rs, f0[1]), (f0[0] + rs, f0[1] + rs),
             (f0[0], f0[1] + rs)],
            closed=False, facecolor="none", edgecolor="#222222", lw=1.2,
        ))

        # Labels
        ax.text((f0[0] + f1[0]) / 2, f0[1] - 0.5, f"base = {base}",
                fontsize=14, fontweight="bold", ha="center", va="top",
                color="#222222")
        ax.text(f0[0] - 0.4, (f0[1] + f2[1]) / 2, f"height = {height}",
                fontsize=14, fontweight="bold", ha="right", va="center",
                color="#222222")
        ax.text((f1[0] + b1[0]) / 2 + 0.15,
                (f1[1] + b1[1]) / 2 - 0.25,
                f"length = {length}", fontsize=14, fontweight="bold",
                ha="left", va="top", color="#222222")

        pts = [f0, f1, f2, b0, b1, b2]
        self._autoscale_pts(ax, pts)
        ax.set_title("Triangular prism",
                     fontsize=14, fontweight="bold", pad=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_cylinder(self, r, h) -> Image.Image:
        fig, ax, style = self._new_axes()
        palette = style["palette"]

        # We draw the cylinder with aspect ratio r : h, but the matplotlib
        # aspect is 'equal' so we use linear units for both. Use rx as the
        # ellipse horizontal radius (== r), ry as the ellipse vertical
        # radius (smaller to give the 'looking slightly down' look).
        cx = 0.0
        cy_bot = 0.0
        cy_top = float(h)
        rx = float(r)
        ry = min(rx * 0.32, 1.2)       # ellipse vertical half-axis

        # 1. Body rectangle (side of cylinder).  Avoid black.
        body_fill = palette[1]
        if body_fill.lower() in ("#000000", "#202020"):
            body_fill = "#c7d6e5"
        ax.add_patch(Polygon(
            [(cx - rx, cy_bot), (cx + rx, cy_bot),
             (cx + rx, cy_top), (cx - rx, cy_top)],
            closed=True, facecolor=body_fill,
            edgecolor="none",
        ))
        # 2. Vertical outline (black lines on each side)
        ax.plot([cx - rx, cx - rx], [cy_bot, cy_top],
                color="black", lw=1.6)
        ax.plot([cx + rx, cx + rx], [cy_bot, cy_top],
                color="black", lw=1.6)

        # 3. Top ellipse (fully visible). Use a non-black fill so the
        # red radius arrow remains visible.
        top_fill = palette[0]
        if top_fill.lower() in ("#000000", "#202020", "#34495e", "#1d3557"):
            top_fill = palette[2] if len(palette) > 2 else "#aac9e0"
        theta = np.linspace(0, 2 * math.pi, 120)
        top_x = cx + rx * np.cos(theta)
        top_y = cy_top + ry * np.sin(theta)
        ax.fill(top_x, top_y, color=top_fill, edgecolor="black", lw=1.6)

        # 4. Bottom ellipse: front half visible (lower arc), back half
        # dashed (upper arc). Using the standard ellipse parameterization
        # theta in [0, 2pi], the lower half y <= cy_bot corresponds to
        # theta in [pi, 2pi].
        theta_front = np.linspace(math.pi, 2 * math.pi, 60)
        theta_back = np.linspace(0, math.pi, 60)
        front_x = cx + rx * np.cos(theta_front)
        front_y = cy_bot + ry * np.sin(theta_front)
        back_x = cx + rx * np.cos(theta_back)
        back_y = cy_bot + ry * np.sin(theta_back)
        ax.plot(front_x, front_y, color="black", lw=1.6)
        ax.plot(back_x, back_y, color="black", lw=1.2, linestyle="--",
                alpha=0.6)

        # 5. Labels
        # Radius arrow on top ellipse: from center to right edge
        ax.annotate(
            "", xy=(cx + rx, cy_top), xytext=(cx, cy_top),
            arrowprops=dict(arrowstyle="->", color="#b00000", lw=2),
        )
        ax.text(cx + rx * 0.5, cy_top + 0.3, f"r = {r}",
                fontsize=14, fontweight="bold", color="#b00000")
        # Height arrow on the left side
        ax.annotate(
            "", xy=(cx - rx - 0.5, cy_top),
            xytext=(cx - rx - 0.5, cy_bot),
            arrowprops=dict(arrowstyle="<->", color="#222222", lw=2),
        )
        ax.text(cx - rx - 0.8, (cy_top + cy_bot) / 2, f"h = {h}",
                fontsize=14, fontweight="bold", color="#222222",
                ha="right", va="center")

        margin_x = 1.8
        margin_y = 1.5
        ax.set_xlim(cx - rx - margin_x, cx + rx + margin_x)
        ax.set_ylim(cy_bot - ry - margin_y, cy_top + ry + margin_y)
        ax.set_title("Right circular cylinder",
                     fontsize=14, fontweight="bold", pad=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _draw_compound_step(self, a1, b1, c1, a2, b2, c2) -> Image.Image:
        """Compound solid: smaller prism centered on top of larger prism."""
        fig, ax, style = self._new_axes()
        palette = style["palette"]

        d = 0.55
        dx = d * math.cos(math.radians(30))
        dy = d * math.sin(math.radians(30))

        def box(x0, z0, a, b, c, colors):
            """Add a box with front-bottom-left at (x0, z0), x-axis=a,
            z-axis=c, depth=b. Returns outer point list."""
            f_bl = (x0, z0)
            f_br = (x0 + a, z0)
            f_tr = (x0 + a, z0 + c)
            f_tl = (x0, z0 + c)
            ox, oy = b * dx, b * dy
            r_bl = (f_bl[0] + ox, f_bl[1] + oy)
            r_br = (f_br[0] + ox, f_br[1] + oy)
            r_tr = (f_tr[0] + ox, f_tr[1] + oy)
            r_tl = (f_tl[0] + ox, f_tl[1] + oy)
            # Right side
            ax.add_patch(Polygon([f_br, r_br, r_tr, f_tr], closed=True,
                                 facecolor=colors[0], edgecolor="black",
                                 lw=1.6))
            # Top
            ax.add_patch(Polygon([f_tl, f_tr, r_tr, r_tl], closed=True,
                                 facecolor=colors[1], edgecolor="black",
                                 lw=1.6))
            # Front
            ax.add_patch(Polygon([f_bl, f_br, f_tr, f_tl], closed=True,
                                 facecolor=colors[2], edgecolor="black",
                                 lw=1.6))
            return [f_bl, f_br, f_tr, f_tl, r_bl, r_br, r_tr, r_tl]

        big_pts = box(0, 0, a1, b1, c1,
                      [palette[2], palette[1], palette[0]])
        # Small block centered on top of big block
        ox_s = (a1 - a2) / 2
        # Depth-center too: offset the small block's y by (b1-b2)/2
        z0_s = c1
        # Also shift its front origin inward in depth
        # To shift in depth, we fake it by adding (depth_offset * dx, dy)
        small_offset_y = (b1 - b2) / 2
        shift = (small_offset_y * dx, small_offset_y * dy)
        ax_shift_x, ax_shift_y = shift
        # Draw small box with adjusted front-bottom-left
        small_bl_x = ox_s + ax_shift_x
        small_bl_y = z0_s + ax_shift_y
        small_pts = box(small_bl_x, small_bl_y, a2, b2, c2,
                        [palette[5], palette[4], palette[3]])

        # Labels
        big_f_bl = big_pts[0]
        big_f_br = big_pts[1]
        big_f_tl = big_pts[3]
        big_f_tr = big_pts[2]
        big_r_tr = big_pts[6]
        ax.text((big_f_bl[0] + big_f_br[0]) / 2,
                big_f_bl[1] - 0.35, f"{a1}",
                fontsize=13, fontweight="bold", ha="center", va="top",
                color="#222222")
        ax.text(big_f_bl[0] - 0.35,
                (big_f_bl[1] + big_f_tl[1]) / 2, f"{c1}",
                fontsize=13, fontweight="bold", ha="right", va="center",
                color="#222222")
        ax.text((big_f_tr[0] + big_r_tr[0]) / 2 + 0.15,
                (big_f_tr[1] + big_r_tr[1]) / 2 + 0.15, f"{b1}",
                fontsize=13, fontweight="bold",
                ha="left", va="bottom", color="#222222")

        # Small labels
        small_f_bl = small_pts[0]
        small_f_br = small_pts[1]
        small_f_tl = small_pts[3]
        small_f_tr = small_pts[2]
        small_r_tr = small_pts[6]
        ax.text((small_f_bl[0] + small_f_br[0]) / 2,
                small_f_bl[1] + (small_f_tl[1] - small_f_bl[1]) / 2, f"{c2}",
                fontsize=12, fontweight="bold", ha="right", va="center",
                color="#1a5276")
        ax.text((small_f_bl[0] + small_f_br[0]) / 2,
                small_f_bl[1] - 0.1, f"{a2}",
                fontsize=12, fontweight="bold", ha="center", va="top",
                color="#1a5276")
        ax.text((small_f_tr[0] + small_r_tr[0]) / 2 + 0.12,
                (small_f_tr[1] + small_r_tr[1]) / 2 + 0.1, f"{b2}",
                fontsize=12, fontweight="bold",
                ha="left", va="bottom", color="#1a5276")

        self._autoscale_pts(ax, big_pts + small_pts)
        ax.set_title("Compound solid",
                     fontsize=14, fontweight="bold", pad=10)
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _autoscale_pts(self, ax, pts):
        arr = np.array(pts)
        margin = 1.0
        ax.set_xlim(arr[:, 0].min() - margin, arr[:, 0].max() + margin)
        ax.set_ylim(arr[:, 1].min() - margin, arr[:, 1].max() + margin)

# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b4"
    os.makedirs(out_dir, exist_ok=True)
    env = PrismVolumeWarmupQA()
    for level in [0, 3, 6]:
        for seed in [1, 2, 3]:
            ok = env.generate(seed=seed, parameter={"level": level})
            status = "OK" if ok else "FAIL"
            print(f"[prism_volume_warmup] L{level} seed{seed}: {status} "
                  f"ans={env._answer!r}")
            if ok:
                env._image.save(
                    f"{out_dir}/prism_volume_warmup_s{seed}_L{level}.png")
