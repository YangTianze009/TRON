"""
Optics diagram QA — lens/mirror ray tracing, image position/magnification.

Fixes:
  - Previously all physics values were in the question text, making the
    diagram redundant. Now the question says "as shown in the diagram"
    and the values are shown on the IMAGE with clearly labeled markers.
  - Diverse lens/mirror styles (thin lens glyph variants, concave/convex).
  - Randomized layout: object and lens at varied x-positions; random
    scale; random colors; random title.
  - Additional question templates (3-5 per qtype).
  - MCQ for real/virtual with shuffled options.
  - L0 vs L9 differ structurally (image position vs focal length).

NOTE on text-leakage: For thin-lens problems the learner needs to know
BOTH f and do (if asked for di), OR do and di (if asked for f). We keep
the two *given* values as labels on the IMAGE (near the object /
focal marker), and the question simply refers to the picture.
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

class OpticsDiagramQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "optics_diagram"

    _Q_TEMPLATES = {
        "converging_image_position": [
            "A converging lens and object are shown. Using the thin lens equation "
            "1/f = 1/do + 1/di with the values labeled on the figure, find the "
            "image distance di (cm). Round to 1 decimal.",
            "Using the focal length f and object distance do labeled in the diagram, "
            "compute the image distance di in cm (round to 1 decimal).",
            "The figure shows a converging lens with the focal length and object distance "
            "marked. Apply the thin lens equation to find di (in cm, 1 decimal).",
        ],
        "magnification": [
            "From the focal length and object distance labeled on the diagram, "
            "compute the absolute magnification |m| = |di/do|. Round to 2 decimals.",
            "Using the values shown, find the absolute magnification of the image. "
            "Round to 2 decimals.",
        ],
        "real_vs_virtual": [
            "For the converging lens in the diagram with the labeled f and do, is the "
            "image real and inverted, or virtual and upright?",
            "The figure shows a converging lens with a given focal length and object "
            "distance. Decide: is the resulting image (A) real and inverted, or "
            "(B) virtual and upright?",
        ],
        "focal_length_from_image": [
            "From the object distance do and image distance di labeled in the diagram, "
            "find the focal length f of the lens (in cm).",
            "Using the values shown in the diagram and 1/f = 1/do + 1/di, compute f "
            "(in cm).",
        ],
        "two_lens_chain": [
            "Two thin converging lenses are placed along the optical axis. The object "
            "sits to the left of Lens 1 (focal length f1). Lens 2 (focal length f2) is "
            "placed at distance L to the right of Lens 1. Using the values labeled in "
            "the diagram, compute the FINAL image distance (cm) measured from Lens 2. "
            "Round to 1 decimal.",
            "A two-lens system is shown. Apply the thin lens equation twice: image from "
            "Lens 1 becomes the object for Lens 2 (with sign based on its position). "
            "From the labeled f1, f2, do, and L, find the final image distance from "
            "Lens 2 (cm, 1 decimal).",
        ],
        "mirror_reflection_angle": [
            "A ray hits a flat mirror at the angle shown in the diagram. What is the "
            "angle of reflection (degrees)?",
            "From the diagram, what is the angle between the reflected ray and the "
            "normal (in degrees)?",
        ],
        "converging_image_type": [
            "Based on the object and focal length shown in the diagram, is the image "
            "(A) real or (B) virtual?",
        ],
    }

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        configs = {
            0: {"question_type": "converging_image_position"},
            1: {"question_type": "converging_image_position"},
            2: {"question_type": "magnification"},
            3: {"question_type": "magnification"},
            4: {"question_type": "real_vs_virtual"},
            5: {"question_type": "real_vs_virtual"},
            6: {"question_type": "focal_length_from_image"},
            7: {"question_type": "focal_length_from_image"},
            # Harden L8-L9 with two-lens chain (requires thin-lens eq twice)
            8: {"question_type": "two_lens_chain"},
            9: {"question_type": "two_lens_chain"},
        }
        return configs[level]

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        lcfg = self._level_config(level)
        if "question_type" not in parameter or parameter.get("question_type") is None:
            parameter = dict(parameter, **lcfg)
        sub = random.Random((self.seed or 0) * 1000 + level * 37 + 5107)
        qtype = parameter.get("question_type", sub.choice(list(self._Q_TEMPLATES.keys())))

        try:
            if qtype == "converging_image_position":
                return self._converging_lens(sub)
            if qtype == "converging_image_type":
                return self._converging_type(sub)
            if qtype == "mirror_reflection_angle":
                return self._mirror_angle(sub)
            if qtype == "magnification":
                return self._magnification(sub)
            if qtype == "real_vs_virtual":
                return self._real_vs_virtual(sub)
            if qtype == "focal_length_from_image":
                return self._focal_length_from_image(sub)
            if qtype == "two_lens_chain":
                return self._two_lens_chain(sub)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _draw_lens(self, ax, style_choice, color):
        """Draw one of several converging-lens glyphs."""
        if style_choice == "thin":
            ax.plot([0, 0], [-2, 2], color=color, linewidth=3.5)
            # Ends with arrows
            ax.plot(0, 2, marker="^", color=color, markersize=10)
            ax.plot(0, -2, marker="v", color=color, markersize=10)
        elif style_choice == "biconvex":
            # Two arcs forming biconvex lens body
            theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
            x1 = -0.4 + 0.6 * np.cos(theta); y1 = 2 * np.sin(theta)
            x2 = 0.4 - 0.6 * np.cos(theta); y2 = 2 * np.sin(theta)
            xs = list(x1) + list(x2[::-1])
            ys = list(y1) + list(y2[::-1])
            ax.fill(xs, ys, facecolor=color, alpha=0.25,
                    edgecolor=color, linewidth=2)
        else:
            # Elliptical lens
            theta = np.linspace(0, 2 * np.pi, 60)
            xs = 0.35 * np.cos(theta); ys = 2 * np.sin(theta)
            ax.plot(xs, ys, color=color, linewidth=2.5)

    def _setup_ax(self, sub):
        style = self._random_style()
        sc = style["figsize_scale"]
        fig_w = sub.uniform(8.0, 10.0) * sc
        fig_h = sub.uniform(4.8, 5.6) * sc
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        return fig, ax, style

    # ------------------------------------------------------------------
    # Problems
    # ------------------------------------------------------------------
    def _converging_lens(self, sub):
        """Find image distance given f and do labeled on image."""
        f = sub.choice([5, 8, 10, 12, 15, 18, 20, 25])
        do = sub.choice([v for v in [10, 12, 14, 15, 18, 20, 22, 25, 30, 35, 40, 45, 50]
                         if v > f + 2])
        di = round(1 / (1 / f - 1 / do), 1)

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)
        lens_c = palette[0]
        obj_c = palette[1]
        focal_c = palette[2]

        lens_style = sub.choice(["thin", "biconvex", "elliptical"])
        self._draw_lens(ax, lens_style, lens_c)

        # Optical axis
        axis_xmin = -do - max(5, f + 3)
        axis_xmax = max(di, do) + max(5, f + 3)
        ax.axhline(0, color="#888", linewidth=1)

        # Focal points — labeled with value
        focal_label_style = sub.choice(["F", "f"])
        ax.plot(f, 0, "o", color=focal_c, markersize=8)
        ax.plot(-f, 0, "o", color=focal_c, markersize=8)
        ax.text(f, -0.55, f"{focal_label_style}={f} cm",
                ha="center", fontsize=style["font_size_base"],
                color=focal_c, fontweight="bold")
        ax.text(-f, -0.55, f"{focal_label_style}={f} cm",
                ha="center", fontsize=style["font_size_base"],
                color=focal_c, fontweight="bold")

        # Object
        obj_h = sub.uniform(1.0, 1.8)
        ax.annotate("", xy=(-do, obj_h), xytext=(-do, 0),
                    arrowprops=dict(arrowstyle="->", color=obj_c, lw=2.5))
        ax.text(-do, obj_h + 0.25, f"Object\ndo={do} cm",
                ha="center", fontsize=style["font_size_base"], color=obj_c)

        # Image position marker "?"
        ax.text(di if di < axis_xmax - 2 else axis_xmax - 3, -0.2,
                "di = ?", fontsize=style["font_size_base"] + 4,
                color="red", fontweight="bold", ha="center", va="top")

        ax.set_xlim(axis_xmin, axis_xmax)
        ax.set_ylim(-3, 3.5)
        ax.axis("off")

        titles = ["Converging Lens — Find Image Position", "Ray Diagram",
                  "Thin Lens Setup", "Find di"]
        ax.set_title(sub.choice(titles),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")

        q = sub.choice(self._Q_TEMPLATES["converging_image_position"])
        return q, str(di), self.fig_to_pil(fig, dpi=style["dpi"])

    def _converging_type(self, sub):
        f = sub.choice([10, 12, 15, 18, 20])
        inside = sub.choice([True, False])
        if inside:
            do = sub.randint(3, f - 1)
            answer = "virtual"
        else:
            do = sub.randint(f + 2, 3 * f)
            answer = "real"

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)
        self._draw_lens(ax, sub.choice(["thin", "biconvex", "elliptical"]),
                        palette[0])
        ax.axhline(0, color="#888", linewidth=1)
        ax.plot(f, 0, "o", color=palette[2], markersize=8)
        ax.text(f, -0.55, f"F = {f} cm",
                ha="center", fontsize=style["font_size_base"],
                color=palette[2], fontweight="bold")
        ax.annotate("", xy=(-do, 1.2), xytext=(-do, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[1], lw=2.5))
        ax.text(-do, 1.45, f"Object\ndo = {do} cm",
                ha="center", fontsize=style["font_size_base"], color=palette[1])
        ax.set_xlim(-do - 6, f + 10)
        ax.set_ylim(-2.5, 2.5)
        ax.axis("off")
        ax.set_title(sub.choice(["Converging Lens — Image Type", "Lens Setup"]),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")

        # MCQ
        options = ["real", "virtual"]
        sub.shuffle(options)
        idx = options.index(answer)
        letter = chr(ord("A") + idx)
        q = (sub.choice(self._Q_TEMPLATES["converging_image_type"])
             + "  " + "  ".join(f"({chr(ord('A')+i)}) {o}"
                                for i, o in enumerate(options)))
        return q, letter, self.fig_to_pil(fig, dpi=style["dpi"])

    def _mirror_angle(self, sub):
        angle_i = sub.randint(15, 75)
        answer = angle_i  # reflection angle equals incidence

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)

        # Mirror: vertical line w/ hatch behind
        ax.plot([0, 0], [-2.5, 2.5], color="#7f8c8d", linewidth=5)
        ax.fill_betweenx([-2.5, 2.5], 0, 0.35, color="#bdc3c7", alpha=0.4)
        # Normal
        ax.plot([-3.5, 0], [0, 0], "k--", linewidth=1.2, alpha=0.55)
        ax.text(-1.8, 0.18, "Normal",
                fontsize=style["font_size_base"] - 1, color="gray")

        rad = math.radians(angle_i)
        ix, iy = -3 * math.cos(rad), 3 * math.sin(rad)

        # Incident ray
        ax.annotate("", xy=(0, 0), xytext=(ix, iy),
                    arrowprops=dict(arrowstyle="->", color=palette[0], lw=2.3))
        ax.text(ix * 0.55, iy * 0.55 + 0.3,
                f"{angle_i}°",
                fontsize=style["font_size_base"] + 2,
                color=palette[0], fontweight="bold")
        # Reflected ray (angle unknown)
        ax.annotate("", xy=(ix, -iy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[1], lw=2.3))
        ax.text(ix * 0.55, -iy * 0.55 - 0.45,
                "θ = ?", fontsize=style["font_size_base"] + 2,
                color="red", fontweight="bold")

        ax.set_xlim(-3.5, 1.2)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(sub.choice(["Mirror Reflection", "Flat Mirror"]),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = sub.choice(self._Q_TEMPLATES["mirror_reflection_angle"])
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])

    def _magnification(self, sub):
        f = sub.choice([8, 10, 12, 15, 18, 20])
        do = sub.choice([v for v in [15, 18, 20, 24, 25, 30, 35, 40, 45]
                         if v > f + 2])
        di = 1 / (1 / f - 1 / do)
        m = abs(round(-di / do, 2))

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)
        self._draw_lens(ax, sub.choice(["thin", "biconvex", "elliptical"]),
                        palette[0])
        ax.axhline(0, color="#888", linewidth=1)
        ax.plot(f, 0, "o", color=palette[2], markersize=8)
        ax.plot(-f, 0, "o", color=palette[2], markersize=8)
        ax.text(0, 2.2, f"f = {f} cm",
                ha="center", fontsize=style["font_size_base"] + 1,
                color=palette[2], fontweight="bold")
        obj_h = sub.uniform(1.0, 1.6)
        ax.annotate("", xy=(-do, obj_h), xytext=(-do, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[1], lw=2.3))
        ax.text(-do, obj_h + 0.25, f"do = {do} cm",
                ha="center", fontsize=style["font_size_base"], color=palette[1])
        ax.set_xlim(-do - 5, do + 5)
        ax.set_ylim(-2.5, 3)
        ax.axis("off")
        ax.set_title(sub.choice(["Find Magnification |m|", "Lens Setup — Magnification"]),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = sub.choice(self._Q_TEMPLATES["magnification"])
        return q, str(m), self.fig_to_pil(fig, dpi=style["dpi"])

    def _real_vs_virtual(self, sub):
        f = sub.choice([8, 10, 12, 15, 18, 20])
        inside = sub.choice([True, False])
        if inside:
            do = sub.randint(3, f - 1)
            answer = "virtual and upright"
        else:
            do = sub.randint(f + 2, 3 * f)
            answer = "real and inverted"

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)
        self._draw_lens(ax, sub.choice(["thin", "biconvex", "elliptical"]),
                        palette[0])
        ax.axhline(0, color="#888", linewidth=1)
        ax.plot(f, 0, "o", color=palette[2], markersize=8)
        ax.text(f, -0.55, f"F = {f} cm",
                ha="center", fontsize=style["font_size_base"],
                color=palette[2], fontweight="bold")
        obj_h = sub.uniform(1.0, 1.6)
        ax.annotate("", xy=(-do, obj_h), xytext=(-do, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[1], lw=2.3))
        ax.text(-do, obj_h + 0.25, f"do = {do} cm",
                ha="center", fontsize=style["font_size_base"], color=palette[1])
        ax.set_xlim(-do - 5, f + 12)
        ax.set_ylim(-2.5, 2.5)
        ax.axis("off")
        ax.set_title(sub.choice(["Image Properties", "Converging Lens — Image"]),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")

        options = ["real and inverted", "virtual and upright"]
        sub.shuffle(options)
        idx = options.index(answer)
        letter = chr(ord("A") + idx)
        q = (sub.choice(self._Q_TEMPLATES["real_vs_virtual"])
             + "  " + "  ".join(f"({chr(ord('A')+i)}) {o}"
                                for i, o in enumerate(options)))
        return q, letter, self.fig_to_pil(fig, dpi=style["dpi"])

    def _focal_length_from_image(self, sub):
        f = sub.choice([5, 8, 10, 12, 15, 18, 20, 25])
        do = sub.choice([v for v in [10, 12, 14, 15, 18, 20, 25, 30, 40, 50]
                         if v > f + 2])
        di = round(1 / (1 / f - 1 / do), 1)

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)
        self._draw_lens(ax, sub.choice(["thin", "biconvex", "elliptical"]),
                        palette[0])
        ax.axhline(0, color="#888", linewidth=1)

        obj_h = sub.uniform(1.1, 1.6)
        ax.annotate("", xy=(-do, obj_h), xytext=(-do, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[1], lw=2.3))
        ax.text(-do, obj_h + 0.3, f"do = {do} cm",
                ha="center", fontsize=style["font_size_base"], color=palette[1])

        ax.annotate("", xy=(di, -1.0), xytext=(di, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[3], lw=2.3))
        ax.text(di, -1.3, f"di = {round(di, 1)} cm",
                ha="center", fontsize=style["font_size_base"], color=palette[3])

        ax.text(0, -2.3, "f = ?",
                fontsize=style["font_size_base"] + 4,
                color="red", fontweight="bold", ha="center")
        ax.set_xlim(-do - 5, max(di, do) + 5)
        ax.set_ylim(-3, 3)
        ax.axis("off")
        ax.set_title(sub.choice(["Find the Focal Length", "Solve for f"]),
                     fontsize=style["font_size_base"] + 2, fontweight="bold")
        q = sub.choice(self._Q_TEMPLATES["focal_length_from_image"])
        return q, str(f), self.fig_to_pil(fig, dpi=style["dpi"])

    def _two_lens_chain(self, sub):
        """Two converging lenses in series. Solve thin lens eq twice.

        Lens 1 at x=0 with focal length f1; Lens 2 at x=L with focal length f2.
        Object at x=-do1 (so do1>0). Find final image distance from Lens 2.

        Sign convention: distances positive to the right of each lens.
        """
        answer = None
        for _ in range(80):
            f1 = sub.choice([8, 10, 12, 15, 18, 20])
            f2 = sub.choice([8, 10, 12, 15, 18, 20])
            do1 = sub.choice([v for v in [20, 24, 30, 36, 40, 45, 50]
                              if v > f1 + 3])
            di1 = 1.0 / (1.0 / f1 - 1.0 / do1)  # image from lens 1 (>0)
            # L candidates: require L > di1 + 3 so image forms between lenses
            L_candidates = [v for v in [25, 30, 35, 40, 45, 50, 55, 60, 70]
                            if v > di1 + 3 and v < di1 + 45]
            if not L_candidates:
                continue
            L = sub.choice(L_candidates)
            do2 = L - di1
            if do2 <= 2 or abs(do2 - f2) < 1.5:
                continue
            try:
                di2 = 1.0 / (1.0 / f2 - 1.0 / do2)
            except ZeroDivisionError:
                continue
            if abs(di2) > 500:
                continue
            answer = round(di2, 1)
            break
        if answer is None:
            return None

        fig, ax, style = self._setup_ax(sub)
        palette = list(style["palette"]); sub.shuffle(palette)
        obj_c = palette[0]
        lens1_c = palette[1]
        lens2_c = palette[2 % len(palette)]
        label_c = palette[3 % len(palette)] if len(palette) >= 4 else "#444"

        # Optical axis
        axis_xmin = -do1 - 6
        axis_xmax = L + 8
        ax.axhline(0, color="#888", linewidth=1, zorder=1)

        # Lens 1 at x=0
        ax.plot([0, 0], [-1.8, 1.8], color=lens1_c, linewidth=3.2, zorder=3)
        ax.plot(0, 1.8, marker="^", color=lens1_c, markersize=9)
        ax.plot(0, -1.8, marker="v", color=lens1_c, markersize=9)
        ax.text(0, 2.1, f"Lens 1\nf1 = {f1} cm",
                ha="center", fontsize=style["font_size_base"],
                color=lens1_c, fontweight="bold")

        # Lens 2 at x=L
        ax.plot([L, L], [-1.8, 1.8], color=lens2_c, linewidth=3.2, zorder=3)
        ax.plot(L, 1.8, marker="^", color=lens2_c, markersize=9)
        ax.plot(L, -1.8, marker="v", color=lens2_c, markersize=9)
        ax.text(L, 2.1, f"Lens 2\nf2 = {f2} cm",
                ha="center", fontsize=style["font_size_base"],
                color=lens2_c, fontweight="bold")

        # Object
        obj_h = sub.uniform(1.0, 1.5)
        ax.annotate("", xy=(-do1, obj_h), xytext=(-do1, 0),
                    arrowprops=dict(arrowstyle="->", color=obj_c, lw=2.3))
        ax.text(-do1, obj_h + 0.25,
                f"Object\ndo = {do1} cm",
                ha="center", fontsize=style["font_size_base"], color=obj_c)

        # Distance label between lenses
        ax.annotate("", xy=(L, -2.4), xytext=(0, -2.4),
                    arrowprops=dict(arrowstyle="<->", color=label_c, lw=1.3))
        ax.text(L / 2.0, -2.7, f"L = {L} cm",
                ha="center", va="top", fontsize=style["font_size_base"],
                color=label_c, fontweight="bold")

        # Question marker past lens 2
        ax.text(L + 4.5, -0.2, "final image\ndi = ?",
                fontsize=style["font_size_base"] + 1,
                color="red", fontweight="bold", ha="center", va="top")

        ax.set_xlim(axis_xmin, axis_xmax)
        ax.set_ylim(-3.5, 3.5)
        ax.axis("off")

        titles = ["Two-Lens System — Find Final Image",
                  "Compound Lens Setup",
                  "Chain of Two Converging Lenses"]
        ax.set_title(sub.choice(titles),
                     fontsize=style["font_size_base"] + 2,
                     fontweight="bold")
        q = sub.choice(self._Q_TEMPLATES["two_lens_chain"])
        return q, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
