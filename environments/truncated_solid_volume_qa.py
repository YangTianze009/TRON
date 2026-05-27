"""
Truncated Solid Volume QA.

Targets: Solid Geometry, Solid_Figures.
Capabilities: M2 (solid geometry).

Visual: a frustum (truncated pyramid or cone) drawn isometrically with
top/bottom dimensions and height labelled. MCQ asking for the volume.

Difficulty axes:
  1. solid_type: L0 = square pyramid frustum; L5 = triangular pyramid
     frustum; L9 = cone frustum.
  2. given_dimensions: L0 = height + both radii/edges; L5 = slant height
     instead of h; L9 = slant height + only one radius (other derived from
     similar triangles, i.e. from overall height of the original solid
     before truncation).

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_SQUARE = ["Frustum of square pyramid", "Truncated square pyramid"]
_TITLE_TRI = ["Frustum of triangular pyramid", "Truncated triangular pyramid"]
_TITLE_CONE = ["Frustum of cone", "Conical frustum", "Truncated cone"]

class TruncatedSolidVolumeQA(StandaloneVisualEnv):
    ENV_NAME = "truncated_solid_volume"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 3:
            solid_type = "square_frustum"
        elif level <= 6:
            solid_type = "tri_frustum"
        else:
            solid_type = "cone_frustum"
        # Smoother ramp: keep ALL three dimensions given through L6 (the
        # change from squarefrustum to trianglefrustum at L4 is already a
        # significant difficulty bump). Slant-derived problems are reserved
        # for L7+. Likewise the formula stays printed on the figure through
        # L5 — at L4-5 the model still has the formula but must apply it to
        # a new solid type. At L6+ no formula and slant must be used.
        if level <= 6:
            given = "all"
        else:
            given = "derived"
        return {
            "solid_type": solid_type,
            "given_dimensions": given,
            "formula_shown": level <= 5,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1021)

        # Choose R, r, h with R > r, r > 0, h > 0
        R_choices = [6, 7, 8, 9, 10, 12]
        R = sub_rng.choice(R_choices)
        r_frac = sub_rng.choice([1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0])
        r = max(1, int(round(R * r_frac)))
        if r >= R:
            r = R - 2
        # Prefer integer Pythagorean consistency so slant is clean
        # slant^2 = (R - r)^2 + h^2
        # pick h from a set that gives a nice Pythagorean triple if possible
        # e.g. R-r=3, h=4 -> slant=5; R-r=6, h=8 -> slant=10; etc.
        diff = R - r
        pyth_h = {3: [4], 4: [3], 6: [8], 8: [6], 5: [12], 12: [5]}
        if diff in pyth_h:
            h = sub_rng.choice(pyth_h[diff])
        else:
            h = sub_rng.choice([4, 5, 6, 8, 10])
        slant = math.sqrt(diff * diff + h * h)

        solid_type = cfg["solid_type"]
        if solid_type == "square_frustum":
            # Edges (not radii). Let R = bottom half-diagonal? Use R & r
            # directly as the edge lengths for simplicity.
            bottom_edge = R
            top_edge = r
            A1 = bottom_edge * bottom_edge
            A2 = top_edge * top_edge
            vol_exact = (h / 3.0) * (A1 + A2 + math.sqrt(A1 * A2))
        elif solid_type == "tri_frustum":
            bottom_edge = R
            top_edge = r
            A1 = (math.sqrt(3) / 4.0) * bottom_edge * bottom_edge
            A2 = (math.sqrt(3) / 4.0) * top_edge * top_edge
            vol_exact = (h / 3.0) * (A1 + A2 + math.sqrt(A1 * A2))
        else:
            # Cone frustum
            vol_exact = (math.pi * h / 3.0) * (R * R + R * r + r * r)

        vol_exact_round = round(vol_exact, 1)
        correct_str = f"{vol_exact_round}"

        # Distractors - common mistakes
        cand = []
        if solid_type == "cone_frustum":
            # Forget the 1/3 factor
            cand.append(math.pi * h * (R * R + R * r + r * r))
            # Use (R^2 + r^2) (miss cross term)
            cand.append(math.pi * h / 3.0 * (R * R + r * r))
            # Cone with average radius ((R+r)/2) full formula
            rm = (R + r) / 2.0
            cand.append(math.pi * rm * rm * h)
            # Treat as cylinder with average
            cand.append(math.pi * rm * rm * h * 2)
            # Swap h with slant
            cand.append(math.pi * slant / 3.0 * (R * R + R * r + r * r))
        else:
            # Forget 1/3 factor
            cand.append(h * (A1 + A2 + math.sqrt(A1 * A2)))
            # Omit cross term
            cand.append((h / 3.0) * (A1 + A2))
            # Average of bases times h
            cand.append(((A1 + A2) / 2.0) * h)
            # Use slant instead of h
            cand.append((slant / 3.0) * (A1 + A2 + math.sqrt(A1 * A2)))
            # Use R for both bases
            cand.append((h / 3.0) * (3 * A1))

        cand_rounded = []
        seen = {round(vol_exact, 1)}
        for c in cand:
            cv = round(c, 1)
            if cv in seen or cv <= 0:
                continue
            seen.add(cv)
            cand_rounded.append(f"{cv}")

        distractors = cand_rounded[:3]
        while len(distractors) < 3:
            fake = vol_exact_round * sub_rng.uniform(1.3, 2.0)
            fake_str = f"{round(fake, 1)}"
            if fake_str != correct_str and fake_str not in distractors:
                distractors.append(fake_str)
            else:
                distractors.append(f"{round(vol_exact_round * 0.5, 1)}")

        options = [correct_str] + distractors
        sub_rng.shuffle(options)
        correct_letter = "ABCD"[options.index(correct_str)]

        # Draw the image
        if solid_type == "square_frustum":
            img = self._draw_square_frustum(sub_rng, R, r, h, slant, cfg)
        elif solid_type == "tri_frustum":
            img = self._draw_tri_frustum(sub_rng, R, r, h, slant, cfg)
        else:
            img = self._draw_cone_frustum(sub_rng, R, r, h, slant, cfg)

        opt_str = "  ".join(f"({chr(65+i)}) {options[i]}" for i in range(4))

        # The volume formula must NOT appear in the question text (hotzone
        # requirement: dims visible, formula NOT in question). The model is
        # expected to recall the frustum formula from its math knowledge.
        formula_note = ""

        if cfg["given_dimensions"] == "derived":
            derived_hint = (
                " The figure shows only the slant height and one radius; "
                "use similar triangles to find the missing dimension."
            )
        else:
            derived_hint = ""

        stems = [
            (f"The figure shows a frustum. Compute its volume.{formula_note}"
             f"{derived_hint} Round to 1 decimal place.\n{opt_str}\n"
             f"Answer with a single letter."),
            (f"What is the volume of the truncated solid in the diagram?"
             f"{formula_note}{derived_hint} Round your answer to 1 decimal.\n"
             f"{opt_str}\nAnswer with a single letter."),
        ]
        q = sub_rng.choice(stems)
        return q, correct_letter, img

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def _draw_square_frustum(self, sub_rng, R, r, h, slant, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(style["bg_color"])

        bR = R / 2.0
        br = r / 2.0
        bottom = np.array([[-bR, -bR, 0], [bR, -bR, 0],
                           [bR, bR, 0], [-bR, bR, 0]])
        top = np.array([[-br, -br, h], [br, -br, h],
                        [br, br, h], [-br, br, h]])

        faces = [list(bottom), list(top)]
        for i in range(4):
            faces.append([bottom[i], bottom[(i + 1) % 4],
                          top[(i + 1) % 4], top[i]])
        color = sub_rng.choice(style["palette"])
        poly = Poly3DCollection(faces, alpha=0.25,
                                facecolors=[color] * len(faces),
                                edgecolors="black", linewidths=1.5)
        ax.add_collection3d(poly)

        # Labels
        mode = cfg["given_dimensions"]
        if mode == "all":
            ax.text(bR + 0.5, 0, 0, f"bottom = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(br + 0.3, 0, h + 0.4, f"top = {r}", fontsize=12,
                    color="blue", fontweight="bold")
            ax.text(-bR - 1.0, 0, h / 2, f"h = {h}", fontsize=12,
                    color="green", fontweight="bold")
        elif mode == "slant_hrr":
            ax.text(bR + 0.5, 0, 0, f"bottom = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(br + 0.3, 0, h + 0.4, f"top = {r}", fontsize=12,
                    color="blue", fontweight="bold")
            slant_lbl = int(slant) if slant == int(slant) else round(slant, 2)
            ax.text(bR - 0.5, -bR - 0.5, h / 2,
                    f"slant = {slant_lbl}", fontsize=12,
                    color="purple", fontweight="bold")
        else:
            ax.text(bR + 0.5, 0, 0, f"bottom = {R}", fontsize=12,
                    color="red", fontweight="bold")
            slant_lbl = int(slant) if slant == int(slant) else round(slant, 2)
            ax.text(bR - 0.5, -bR - 0.5, h / 2,
                    f"slant = {slant_lbl}", fontsize=12,
                    color="purple", fontweight="bold")
            ax.text(br + 0.3, 0, h + 0.4, f"top edge = ?", fontsize=12,
                    color="#444", fontweight="bold")

        m = max(R / 2.0, h) * 1.3
        ax.set_xlim([-m, m]); ax.set_ylim([-m, m]); ax.set_zlim([-1, h + 2])
        ax.view_init(elev=sub_rng.randint(18, 28),
                     azim=sub_rng.choice([30, 40, 55, 65]))
        ax.set_title(sub_rng.choice(_TITLE_SQUARE),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=12)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=max(style["dpi"], 130))

    def _draw_tri_frustum(self, sub_rng, R, r, h, slant, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(style["bg_color"])

        def tri_points(edge, z):
            # equilateral triangle with side length `edge`, centered
            pts = []
            for i in range(3):
                theta = math.radians(90 + 120 * i)
                rr = edge / math.sqrt(3)
                pts.append([rr * math.cos(theta), rr * math.sin(theta), z])
            return np.array(pts)

        bottom = tri_points(R, 0)
        top = tri_points(r, h)
        faces = [list(bottom), list(top)]
        for i in range(3):
            faces.append([bottom[i], bottom[(i + 1) % 3],
                          top[(i + 1) % 3], top[i]])
        color = sub_rng.choice(style["palette"])
        poly = Poly3DCollection(faces, alpha=0.25,
                                facecolors=[color] * len(faces),
                                edgecolors="black", linewidths=1.5)
        ax.add_collection3d(poly)

        mode = cfg["given_dimensions"]
        slant_lbl = int(slant) if slant == int(slant) else round(slant, 2)
        if mode == "all":
            ax.text(0, -R / 1.5, 0, f"bottom = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(0, -r / 1.5, h + 0.4, f"top = {r}", fontsize=12,
                    color="blue", fontweight="bold")
            ax.text(-R / 2 - 0.6, 0, h / 2, f"h = {h}", fontsize=12,
                    color="green", fontweight="bold")
        elif mode == "slant_hrr":
            ax.text(0, -R / 1.5, 0, f"bottom = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(0, -r / 1.5, h + 0.4, f"top = {r}", fontsize=12,
                    color="blue", fontweight="bold")
            ax.text(-R / 2, -R / 2, h / 2, f"slant = {slant_lbl}",
                    fontsize=12, color="purple", fontweight="bold")
        else:
            ax.text(0, -R / 1.5, 0, f"bottom = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(-R / 2, -R / 2, h / 2, f"slant = {slant_lbl}",
                    fontsize=12, color="purple", fontweight="bold")
            ax.text(0, -r / 1.5, h + 0.4, "top = ?", fontsize=12,
                    color="#444", fontweight="bold")

        m = max(R, h) * 1.1
        ax.set_xlim([-m, m]); ax.set_ylim([-m, m]); ax.set_zlim([-1, h + 2])
        ax.view_init(elev=sub_rng.randint(18, 28),
                     azim=sub_rng.choice([30, 40, 55, 65]))
        ax.set_title(sub_rng.choice(_TITLE_TRI),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=12)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=max(style["dpi"], 130))

    def _draw_cone_frustum(self, sub_rng, R, r, h, slant, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(7 * sc, 7 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(style["bg_color"])

        theta = np.linspace(0, 2 * np.pi, 60)
        # Bottom circle
        ax.plot(R * np.cos(theta), R * np.sin(theta), 0,
                color="black", lw=1.5)
        # Top circle
        ax.plot(r * np.cos(theta), r * np.sin(theta), h,
                color="black", lw=1.5)

        # Lateral surface
        zl = np.linspace(0, h, 20)
        theta_g, z_g = np.meshgrid(theta, zl)
        r_g = R + (r - R) * (z_g / h)
        color = sub_rng.choice(style["palette"])
        ax.plot_surface(r_g * np.cos(theta_g), r_g * np.sin(theta_g), z_g,
                        alpha=0.22, color=color)

        # Slant edges for visual reference
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ax.plot([R * np.cos(angle), r * np.cos(angle)],
                    [R * np.sin(angle), r * np.sin(angle)],
                    [0, h], color="black", lw=1, alpha=0.5)

        mode = cfg["given_dimensions"]
        slant_lbl = int(slant) if slant == int(slant) else round(slant, 2)
        if mode == "all":
            ax.text(R + 0.5, 0, 0, f"R = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(r + 0.3, 0, h + 0.3, f"r = {r}", fontsize=12,
                    color="blue", fontweight="bold")
            ax.text(0.3, 0, h / 2, f"h = {h}", fontsize=12,
                    color="green", fontweight="bold")
        elif mode == "slant_hrr":
            ax.text(R + 0.5, 0, 0, f"R = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(r + 0.3, 0, h + 0.3, f"r = {r}", fontsize=12,
                    color="blue", fontweight="bold")
            ax.text(R / 2 + 0.3, -R / 2 + 0.2, h / 2 + 0.4,
                    f"slant = {slant_lbl}",
                    fontsize=12, color="purple", fontweight="bold")
        else:
            ax.text(R + 0.5, 0, 0, f"R = {R}", fontsize=12,
                    color="red", fontweight="bold")
            ax.text(R / 2 + 0.3, -R / 2 + 0.2, h / 2 + 0.4,
                    f"slant = {slant_lbl}",
                    fontsize=12, color="purple", fontweight="bold")
            ax.text(r + 0.3, 0, h + 0.3, "r = ?", fontsize=12,
                    color="#444", fontweight="bold")

        m = max(R, h) * 1.15
        ax.set_xlim([-m, m]); ax.set_ylim([-m, m]); ax.set_zlim([-1, h + 2])
        ax.view_init(elev=sub_rng.randint(18, 28),
                     azim=sub_rng.choice([30, 40, 55, 65]))
        ax.set_title(sub_rng.choice(_TITLE_CONE),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=12)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=max(style["dpi"], 130))

if __name__ == "__main__":
    env = TruncatedSolidVolumeQA()
    for lv in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": lv}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{lv}: {gt}")
