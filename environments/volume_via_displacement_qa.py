"""
Volume-via-Displacement QA.

A container (rectangular cuboid or cylinder) holds water at some level h_init.
A solid object is fully submerged, raising the water to h_final. The model
computes the object's volume as V = (h_final - h_init) * base_area.

Difficulty axes:
  - container shape: cuboid at L<=4, cylinder at L>=5 (uses pi=3.14)
  - magnitude of dimensions
  - decimal vs integer water rise
  - unit conversion (cm³ -> mL or L) at L>=8
"""
import math
import random
import re
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_CUBOID_CM3 = [
    "A rectangular tank with base length {L} cm and width {W} cm contains water at a height of {h0} cm. After a solid object is fully submerged, the water rises to {h1} cm. What is the volume of the object in cubic centimeters? Output the number in <answer>...</answer>.",
    "The image shows a rectangular cuboid container ({L} cm by {W} cm base) holding water at level {h0} cm. A solid object is dropped in and the water level rises to {h1} cm. Find the volume of the object (cm³). Place the answer in <answer>...</answer>.",
    "Look at the figure. A {L}-cm by {W}-cm rectangular tank has water at height {h0} cm. After fully submerging a solid object, the water level reaches {h1} cm. Compute the object's volume in cm³. Answer in <answer>...</answer>.",
    "A rectangular box of base {L} cm × {W} cm contains water at {h0} cm. When an object is sunk in it, the water rises to {h1} cm. What is the volume of the object in cm³? Output in <answer>...</answer>.",
    "A cuboid container ({L} cm long, {W} cm wide) holds water to height {h0} cm. After submerging a solid, the water level becomes {h1} cm. Find the object's volume in cubic centimeters and place it in <answer>...</answer>.",
    "Examine the image: a rectangular tank with base {L} × {W} cm has water at {h0} cm; an object is fully submerged and water rises to {h1} cm. Output the object's volume in cm³ inside <answer>...</answer>.",
    "Given the rectangular cuboid container shown ({L} cm × {W} cm base, water from {h0} cm to {h1} cm after submerging a solid), compute the object's volume in cubic centimeters. Place answer in <answer>...</answer>.",
    "A rectangular water tank ({L} cm by {W} cm) has water level {h0} cm. After a stone is sunk into it, the water rises to {h1} cm. The volume of the stone is __ cm³. Place answer in <answer>...</answer>.",
]

_TEMPLATES_CYL_CM3 = [
    "A cylindrical container with base radius {r} cm holds water at {h0} cm. After a solid object is submerged, the water level reaches {h1} cm. What is the volume of the object in cm³? Use π = 3.14. Place answer in <answer>...</answer>.",
    "The image shows a cylindrical tank, radius {r} cm. Water level is {h0} cm; after submerging an object, water rises to {h1} cm. Compute the object's volume (cm³, π = 3.14). Output in <answer>...</answer>.",
    "Look at the figure. A cylinder with radius {r} cm contains water at {h0} cm. A solid object is dropped in and the water level becomes {h1} cm. Find the volume of the object in cm³ (π = 3.14). Place answer in <answer>...</answer>.",
    "A cylindrical bucket has base radius {r} cm. The water is initially at {h0} cm; after fully submerging a solid, the water level rises to {h1} cm. Use π = 3.14 to compute the object's volume in cm³. Output in <answer>...</answer>.",
    "Given a cylindrical container of radius {r} cm with water from {h0} cm to {h1} cm after a solid is submerged, compute the volume of the solid in cm³ (π = 3.14). Place answer in <answer>...</answer>.",
    "A cylinder, radius {r} cm, contains water at level {h0} cm. After submerging an object, the water level is {h1} cm. The object's volume is __ cm³ (π = 3.14). Answer in <answer>...</answer>.",
    "From the image: cylindrical tank radius {r} cm, water rises from {h0} cm to {h1} cm when a stone is fully submerged. What is the stone's volume in cm³? Use π = 3.14. Output in <answer>...</answer>.",
    "A cylindrical glass container of radius {r} cm has water at {h0} cm. After a solid object is dropped in (fully submerged), water reaches {h1} cm. Volume of object in cm³? (π = 3.14). Place in <answer>...</answer>.",
]

_TEMPLATES_ML_CONV = [
    "A rectangular tank with base {L} cm × {W} cm holds water at {h0} cm. A submerged solid raises the water to {h1} cm. What is the volume of the solid in milliliters (mL)? Note: 1 mL = 1 cm³. Place answer in <answer>...</answer>.",
    "A cylindrical container, radius {r} cm, has water at {h0} cm. After submerging an object, water rises to {h1} cm. Output the object's volume in mL (use π = 3.14, and 1 mL = 1 cm³). Place answer in <answer>...</answer>.",
    "From the figure: water level rises from {h0} cm to {h1} cm in a {L} cm × {W} cm cuboid container after a solid is fully submerged. What is the object's volume in mL? Place answer in <answer>...</answer>.",
    "A cylinder of radius {r} cm holds water at {h0} cm. A submerged solid raises the level to {h1} cm. Find the object's volume in mL (π = 3.14, 1 mL = 1 cm³). Output in <answer>...</answer>.",
]

_TEMPLATES_L_CONV = [
    "A large rectangular tank with base {L} cm × {W} cm is filled with water to {h0} cm. A submerged solid raises the level to {h1} cm. What is the volume of the solid in liters (L)? Note: 1 L = 1000 cm³. Output in <answer>...</answer>.",
    "From the image: a cuboid container ({L} cm × {W} cm base) has water rising from {h0} cm to {h1} cm after submerging an object. Volume of the object in liters (1 L = 1000 cm³)? Place answer in <answer>...</answer>.",
    "A cylindrical tank, base radius {r} cm, has water at level {h0} cm. A submerged solid raises the level to {h1} cm. What is the object's volume in liters? (π = 3.14, 1 L = 1000 cm³). Output in <answer>...</answer>.",
    "A {L} cm × {W} cm rectangular pool has water at {h0} cm. After submerging a solid, water reaches {h1} cm. What is the object's volume in liters? (1 L = 1000 cm³). Place answer in <answer>...</answer>.",
]


class VolumeViaDisplacementQA(StandaloneVisualEnv):
    ENV_NAME = "volume_via_displacement"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output the numeric answer (no units) inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # mode is one of "cuboid_int", "cuboid", "cylinder", "convert_ml", "convert_l"
        if level == 0:
            return {"mode": "cuboid_int", "decimal_rise": False}
        if level <= 2:
            return {"mode": "cuboid_int", "decimal_rise": False}
        if level <= 4:
            return {"mode": "cuboid", "decimal_rise": True}
        if level <= 6:
            return {"mode": "cylinder", "decimal_rise": False}
        if level <= 7:
            return {"mode": "cylinder", "decimal_rise": True}
        if level == 8:
            return {"mode": "convert_ml", "decimal_rise": True}
        # 2026-05-04: bumped L9 difficulty — cylinder + L conversion (3-step:
        # area=π·r², ΔV=area·rise, then divide by 1000).
        return {"mode": "convert_l_cylinder", "decimal_rise": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4099 + level * 23 + 13)

        mode = cfg["mode"]

        if mode in ("cuboid_int", "cuboid", "convert_ml", "convert_l"):
            # Pick container base dims and water levels
            if mode == "convert_l":
                # large tank for liter scale
                L = rng.randint(20, 40)
                W = rng.randint(20, 30)
            else:
                L = rng.randint(5, 15)
                W = rng.randint(4, 12)
            h0 = rng.randint(2, 9)
            if cfg["decimal_rise"]:
                # rise in 0.5 increments
                rise_units = rng.randint(2, 8)
                rise = rise_units * 0.5
            else:
                rise = float(rng.randint(1, 5))
            h1 = h0 + rise
            base_area = L * W
            vol_cm3 = base_area * rise

            if mode == "cuboid_int":
                template = rng.choice(_TEMPLATES_CUBOID_CM3)
                question = template.format(L=L, W=W, h0=h0, h1=h1)
                ans = self._fmt_num(vol_cm3)
            elif mode == "cuboid":
                template = rng.choice(_TEMPLATES_CUBOID_CM3)
                question = template.format(L=L, W=W, h0=h0,
                                            h1=self._fmt_num(h1))
                ans = self._fmt_num(vol_cm3)
            elif mode == "convert_ml":
                template = rng.choice(_TEMPLATES_ML_CONV[::2])  # cuboid ones
                question = template.format(L=L, W=W, h0=h0,
                                            h1=self._fmt_num(h1))
                ans = self._fmt_num(vol_cm3)  # 1 mL = 1 cm³
            else:  # convert_l
                template = rng.choice([t for t in _TEMPLATES_L_CONV
                                        if "{L}" in t and "{r}" not in t])
                question = template.format(L=L, W=W, h0=h0,
                                            h1=self._fmt_num(h1))
                ans = self._fmt_num(vol_cm3 / 1000.0)

            self._gt_value = float(ans)
            img = self._render_cuboid(L, W, h0, h1, rng)
            return question, ans, img

        elif mode == "cylinder":
            # Pick radius and water levels; use π = 3.14 to keep answers clean
            r = rng.choice([2, 3, 4, 5, 6, 8, 10])
            h0 = rng.randint(3, 10)
            if cfg["decimal_rise"]:
                rise = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5])
            else:
                rise = float(rng.randint(1, 4))
            h1 = h0 + rise
            base_area = 3.14 * r * r
            vol = base_area * rise

            template = rng.choice(_TEMPLATES_CYL_CM3)
            question = template.format(r=r, h0=h0, h1=self._fmt_num(h1))
            ans = self._fmt_num(vol)
            self._gt_value = float(ans)
            img = self._render_cylinder(r, h0, h1, rng)
            return question, ans, img

        elif mode == "convert_l_cylinder":
            # 2026-05-04: L9 — cylindrical tank + L conversion (3-step problem).
            r = rng.choice([10, 12, 15, 20, 25])  # bigger radius for L scale
            h0 = rng.randint(5, 15)
            rise = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
            h1 = h0 + rise
            base_area = 3.14 * r * r
            vol = base_area * rise  # cm³
            template = next(t for t in _TEMPLATES_L_CONV if "{r}" in t)
            question = template.format(r=r, h0=h0, h1=self._fmt_num(h1))
            ans = self._fmt_num(vol / 1000.0)
            self._gt_value = float(ans)
            img = self._render_cylinder(r, h0, h1, rng)
            return question, ans, img

        return None

    @staticmethod
    def _fmt_num(v: float) -> str:
        if abs(v - round(v)) < 1e-7:
            return str(int(round(v)))
        # Round to 2 decimals max, strip trailing zeros
        s = f"{v:.4f}"
        s = s.rstrip("0").rstrip(".")
        return s

    # ----------------------------------------------------------------- #
    # Renderers
    # ----------------------------------------------------------------- #
    def _render_cuboid(self, L, W, h0, h1, rng) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0"])
        fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        # Container side view: width L (horizontal), height = h1 + 2 cm headroom
        H_max = max(h0, h1) + 3
        # Outer container outline
        wall = mpatches.Rectangle((0, 0), L, H_max, linewidth=2.0,
                                    facecolor="none", edgecolor="#37474f")
        ax.add_patch(wall)
        # Initial water (light blue, dotted top line shows initial level)
        water_init = mpatches.Rectangle(
            (0, 0), L, h0, linewidth=0,
            facecolor="#aed6f1", alpha=0.5)
        ax.add_patch(water_init)
        # Final water (slightly darker on top)
        water_final = mpatches.Rectangle(
            (0, h0), L, h1 - h0, linewidth=0,
            facecolor="#5dade2", alpha=0.6)
        ax.add_patch(water_final)
        # Object inside: small rounded rect
        ow = min(L * 0.4, 4)
        oh = min((h1 - h0) * 1.4, h1 * 0.5)
        ox = (L - ow) / 2
        oy = max(0.4, (h1 - oh) * 0.45)
        obj = mpatches.FancyBboxPatch(
            (ox, oy), ow, oh,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="#a1887f", edgecolor="#5d4037", linewidth=1.4,
            zorder=5)
        ax.add_patch(obj)
        ax.text(ox + ow / 2, oy + oh / 2, "object", fontsize=9,
                 ha="center", va="center", color="#3e2723", zorder=6,
                 fontweight="bold")
        # Initial water level dashed line
        ax.plot([0, L], [h0, h0], color="#1565c0", linewidth=1.5,
                 linestyle="--", zorder=4)
        # Final water level solid line
        ax.plot([0, L], [h1, h1], color="#0d47a1", linewidth=2.0, zorder=4)
        # Labels for water levels
        ax.annotate(f"h0 = {self._fmt_num(h0)} cm", xy=(L * 0.55, h0),
                     xytext=(L + 0.4, h0 - 0.05), fontsize=10,
                     color="#1565c0",
                     arrowprops=dict(arrowstyle="-", color="#1565c0",
                                       lw=0.8))
        ax.annotate(f"h1 = {self._fmt_num(h1)} cm", xy=(L * 0.55, h1),
                     xytext=(L + 0.4, h1 + 0.4), fontsize=10,
                     color="#0d47a1",
                     arrowprops=dict(arrowstyle="-", color="#0d47a1",
                                       lw=0.8))
        # Width labels (base dimension shown explicitly)
        ax.annotate("", xy=(L, -0.7), xytext=(0, -0.7),
                     arrowprops=dict(arrowstyle="<->", color="#555"))
        ax.text(L / 2, -1.2, f"length = {L} cm  (base width = {W} cm)",
                 ha="center", va="top", fontsize=10, color="#37474f")
        # Set view limits
        ax.set_xlim(-1.0, L + 5.5)
        ax.set_ylim(-2.0, H_max + 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Submerged object — water rise", fontsize=11,
                     color="#1d3557", pad=4)
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                     facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _render_cylinder(self, r, h0, h1, rng) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0"])
        fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=130)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        diameter = 2 * r
        H_max = max(h0, h1) + 3.0

        # Cylinder body: rectangular outline + top/bottom ellipses for 3D feel
        ax.plot([0, 0], [0, H_max], color="#37474f", linewidth=2.0)
        ax.plot([diameter, diameter], [0, H_max], color="#37474f", linewidth=2.0)
        # Bottom ellipse
        bot_ell = mpatches.Ellipse((diameter / 2, 0), diameter, diameter / 2.5,
                                     facecolor="#cfd8dc", edgecolor="#37474f",
                                     linewidth=2.0)
        ax.add_patch(bot_ell)
        # Top ellipse (open top)
        top_ell = mpatches.Ellipse((diameter / 2, H_max), diameter,
                                     diameter / 2.5,
                                     facecolor="#ffffff", edgecolor="#37474f",
                                     linewidth=2.0)
        ax.add_patch(top_ell)

        # Initial water rect + top ellipse
        water_rect_init = mpatches.Rectangle(
            (0, 0), diameter, h0, facecolor="#aed6f1", alpha=0.5,
            zorder=2)
        ax.add_patch(water_rect_init)
        # Final water rect (light fill above initial)
        water_rect_final = mpatches.Rectangle(
            (0, h0), diameter, h1 - h0, facecolor="#5dade2", alpha=0.6,
            zorder=3)
        ax.add_patch(water_rect_final)
        # Final water surface ellipse
        water_top_ell = mpatches.Ellipse(
            (diameter / 2, h1), diameter, diameter / 2.5,
            facecolor="#5dade2", edgecolor="#0d47a1", linewidth=2.0,
            alpha=0.85, zorder=4)
        ax.add_patch(water_top_ell)
        # Initial water level (dashed ellipse trace)
        init_ell = mpatches.Ellipse(
            (diameter / 2, h0), diameter, diameter / 2.5,
            facecolor="none", edgecolor="#1565c0", linewidth=1.4,
            linestyle="--", zorder=5)
        ax.add_patch(init_ell)
        # Submerged object marker
        obj = mpatches.FancyBboxPatch(
            (diameter / 2 - r * 0.4, h0 * 0.3),
            r * 0.8, max(0.6, (h1 - h0) * 0.7),
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#a1887f", edgecolor="#5d4037", linewidth=1.4,
            zorder=6)
        ax.add_patch(obj)
        ax.text(diameter / 2, h0 * 0.3 + max(0.3, (h1 - h0) * 0.35),
                 "object", fontsize=8, ha="center", va="center",
                 color="#3e2723", zorder=7, fontweight="bold")

        # Labels
        ax.annotate(f"h0 = {self._fmt_num(h0)} cm",
                     xy=(diameter, h0),
                     xytext=(diameter + 1.0, h0 - 0.4),
                     fontsize=10, color="#1565c0",
                     arrowprops=dict(arrowstyle="-", color="#1565c0",
                                       lw=0.8))
        ax.annotate(f"h1 = {self._fmt_num(h1)} cm",
                     xy=(diameter, h1),
                     xytext=(diameter + 1.0, h1 + 0.4),
                     fontsize=10, color="#0d47a1",
                     arrowprops=dict(arrowstyle="-", color="#0d47a1",
                                       lw=0.8))
        ax.annotate("", xy=(diameter, -1.1), xytext=(0, -1.1),
                     arrowprops=dict(arrowstyle="<->", color="#555"))
        ax.text(diameter / 2, -1.7, f"radius = {r} cm",
                 ha="center", va="top", fontsize=10, color="#37474f")

        ax.set_xlim(-1.0, diameter + 5.0)
        ax.set_ylim(-2.5, H_max + 2.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Cylindrical container — water rise", fontsize=11,
                     color="#1d3557", pad=4)
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                     facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ----------------------------------------------------------------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Extract first signed decimal number
        m = re.search(r"-?\d+(?:\.\d+)?", predicted.replace(",", ""))
        if not m:
            return False
        try:
            p = float(m.group())
        except ValueError:
            return False
        g = self._gt_value
        if abs(g) < 1e-9:
            return abs(p) < 1e-3
        # Within 2% tolerance, or absolute < 0.5
        return abs(p - g) / abs(g) < 0.02 or abs(p - g) < 0.5


if __name__ == "__main__":
    env = VolumeViaDisplacementQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer}")
