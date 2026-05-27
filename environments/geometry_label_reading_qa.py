"""
Geometry Label Reading QA (v4 G1, for Angles/Length and Property reasoning).

Targets (from bug doc rev-2 Cluster 4 + case pulls):

Failure mode (from bug rev-2 idx=48 cylinder, idx=115 rect, idx=116 cylinder unfold,
idx=229 triangle cases):
  The model can do the geometry math correctly, but it misidentifies WHICH label
  in the figure corresponds to which geometric quantity. E.g.:
  - labels 6 cm and 4 cm on a cylinder — v3 swaps diameter vs height
  - label '12.56 cm' on a rectangle — v3 hallucinates a second length '4 cm'
  - triangle with two tick marks showing AB=AC — v3 says "insufficient info"

Fix: isolate the label-reading step. Generate a figure with numeric labels;
ask what geometric role each label plays (radius / diameter / chord /
height / side length / angle / etc.). No arithmetic at all.

Level axes:
  A) Figure type: circle at L0 -> triangle/rectangle at L1-3 -> cylinder/cone at L4-6 -> compound at L7+
  B) Label ambiguity: unambiguous at L0-2 -> placed-inside at L3-5 -> adjacent-to-2-objects at L6+
  C) Number of labels: 1 at L0, 2-3 at L3-5, 4+ at L6+
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_ROLE_TEMPLATES_SINGLE = [
    "In the figure, a single numeric label is shown. What geometric quantity does it indicate? Options: {opts}. Put the role name in <answer>...</answer>.",
    "Identify what the labeled number represents in the figure. Options: {opts}. Put the role in <answer>...</answer>.",
    "Look at the figure; one numeric label is placed on a specific geometric element. Which of {opts} does it label? Answer in <answer>...</answer>.",
    "The figure has exactly one numeric label. Which geometric feature does it measure? Options: {opts}. Put the answer in <answer>...</answer>.",
    "The number on the figure labels which feature? Options: {opts}. Answer in <answer>...</answer>.",
    "Identify the geometric role of the single label in the figure. Options: {opts}. Put the role in <answer>...</answer>.",
    "Which of {opts} is the label on the figure indicating? Answer in <answer>...</answer>.",
    "Look at the figure and identify what geometric measurement the label represents. Options: {opts}. Put it in <answer>...</answer>.",
    "The numeric label in the figure corresponds to which geometric quantity? Options: {opts}. Answer in <answer>...</answer>.",
    "Name the geometric role of the label shown in the figure. Options: {opts}. Put the role name in <answer>...</answer>.",
    "From the figure, what does the labeled number represent? Options: {opts}. Put it in <answer>...</answer>.",
    "What is the geometric role of the single label? Options: {opts}. Answer in <answer>...</answer>.",
    "State which of {opts} the label is measuring. Put the role name in <answer>...</answer>.",
    "Identify what the number labels in the figure. Choose from {opts}. Answer in <answer>...</answer>.",
    "The figure has one number; identify the geometric element it labels. Options: {opts}. Put the answer in <answer>...</answer>.",
    "In the figure, a single measurement is shown. Which of {opts} does it correspond to? Answer in <answer>...</answer>.",
]

_ROLE_TEMPLATES_TABLE = [
    "The figure has several numeric labels. For each label, state what geometric quantity it represents. Output a Python-style dict mapping each label's value to its role, e.g. {{'5': 'radius', '30': 'angle_A'}}. Put the full dict in <answer>...</answer>.",
    "Read every labeled value in the figure and identify its geometric role. Output a dict like {{value: role}}. Put the dict in <answer>...</answer>.",
    "Produce a mapping from each numeric label in the figure to its geometric role (e.g. 'radius', 'height', 'angle_B'). Format: {{value: role}}. Put in <answer>...</answer>.",
    "For every number shown in the figure, give its corresponding geometric role. Output a dict {{value: role}}. Answer in <answer>...</answer>.",
    "List every numeric label and its geometric role as a dict. Put the dict in <answer>...</answer>.",
    "Map each number in the figure to its geometric role (e.g. 'diameter', 'angle_A', 'side_AB'). Output a single dict. Put in <answer>...</answer>.",
    "Output a dict {{'label_value': 'role'}} covering all numeric labels in the figure. Put in <answer>...</answer>.",
    "Enumerate all labels in the figure with their roles. Format: dict mapping value to role. Put in <answer>...</answer>.",
    "Parse every numeric label and produce a dict of value-to-role. Put in <answer>...</answer>.",
    "Build a dict that contains every label's value and its geometric role. Put in <answer>...</answer>.",
    "Identify the geometric role of each numeric label in the figure. Output as a dict mapping value to role. Put in <answer>...</answer>.",
    "Output a key-value mapping from each label value in the figure to the role it labels. Put in <answer>...</answer>.",
    "Give a dictionary from label values to their geometric roles for every number shown. Put in <answer>...</answer>.",
    "Provide a full mapping of each label to its role for every numeric annotation in the figure. Put the dict in <answer>...</answer>.",
    "List all numeric labels in the figure and their geometric roles as a dict. Put in <answer>...</answer>.",
    "Return a dict {{value: role}} covering every labeled quantity in the figure. Put in <answer>...</answer>.",
]

# Predefined figure generators for each level band
# Each returns (image, roles_dict: {value: role}, figure_name)

class GeometryLabelReadingQA(StandaloneVisualEnv):
    ENV_NAME = "geometry_label_reading"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            fig_types = ["circle", "rectangle", "triangle"]
            n_labels = 1
        elif level <= 4:
            fig_types = ["circle", "rectangle", "triangle", "cylinder_net"]
            n_labels = 2
        elif level <= 6:
            fig_types = ["cylinder_net", "cone_net", "trapezoid", "rect_in_circle"]
            n_labels = 3
        else:
            fig_types = ["compound", "cylinder_net", "cone_net", "rect_in_circle"]
            n_labels = 4
        return {"fig_types": fig_types, "n_labels": n_labels, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 131)
        self._primary_complexity_feature = level

        fig_type = rng.choice(cfg["fig_types"])
        renderer = {
            "circle": self._draw_circle,
            "rectangle": self._draw_rectangle,
            "triangle": self._draw_triangle,
            "cylinder_net": self._draw_cylinder_net,
            "cone_net": self._draw_cone_net,
            "trapezoid": self._draw_trapezoid,
            "rect_in_circle": self._draw_rect_in_circle,
            "compound": self._draw_compound,
        }.get(fig_type, self._draw_circle)
        # BUGFIX 2026-04-24: renderers now sample distinct numeric values
        # (see _draw_compound etc.). Single call is sufficient.
        img, roles = renderer(rng, cfg)

        # Decide question type
        sidx = (self.seed or 0) % 16
        if cfg["n_labels"] == 1:
            # Single label → multi-choice role naming
            key, gt_role = next(iter(roles.items()))
            # Distractors from a pool of plausible other roles
            other_pool = ["radius", "diameter", "chord", "height", "width",
                          "side_AB", "side_BC", "angle_A", "angle_B",
                          "perimeter", "area", "slant_height", "circumference",
                          "arc_length"]
            distractors = [r for r in rng.sample(other_pool, 6) if r != gt_role][:3]
            options = [gt_role] + distractors
            rng.shuffle(options)
            q = _ROLE_TEMPLATES_SINGLE[sidx].format(
                opts=", ".join(options))
            answer = gt_role
        else:
            # Multi-label → dict mapping
            q = _ROLE_TEMPLATES_TABLE[sidx]
            # canonical GT string
            answer = str(roles)

        return q, answer, img

    # -------------------------- Figure renderers --------------------------

    def _figure_setup(self, rng, xlim, ylim):
        bg = rng.choice(["#ffffff", "#fdfdfd", "#fafafa", "#f8f8f8"])
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    def _draw_circle(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        fig, ax = self._figure_setup(rng, (-2.5, 2.5), (-2.5, 2.5))
        r_val = rng.randint(2, 8)
        d_val = r_val * 2
        circle = mpatches.Circle((0, 0), 1.8, fc="none", ec="black", lw=2.0)
        ax.add_patch(circle)
        if cfg["n_labels"] == 1:
            which = rng.choice(["radius", "diameter", "chord"])
            if which == "radius":
                ax.plot([0, 1.8], [0, 0], color="black", lw=1.3)
                ax.annotate(f"{r_val}", xy=(0.9, 0.2), fontsize=18,
                            ha="center", fontweight="bold")
                roles = {str(r_val): "radius"}
            elif which == "diameter":
                ax.plot([-1.8, 1.8], [0, 0], color="black", lw=1.3)
                ax.annotate(f"{d_val}", xy=(0, 0.25), fontsize=18,
                            ha="center", fontweight="bold")
                roles = {str(d_val): "diameter"}
            else:  # chord
                ax.plot([-1.5, 1.5], [-1, -1], color="black", lw=1.3)
                ch = 6
                ax.annotate(f"{ch}", xy=(0, -1.4), fontsize=18,
                            ha="center", fontweight="bold")
                roles = {str(ch): "chord"}
            return self.fig_to_pil(fig), roles
        # Multi-label — always produce exactly n_labels
        roles = {}
        # always add radius
        ax.plot([0, 1.8], [0, 0], color="black", lw=1.3)
        ax.annotate(f"{r_val}", xy=(0.9, 0.22), fontsize=16,
                    ha="center", fontweight="bold")
        roles[str(r_val)] = "radius"
        # chord — BUGFIX 2026-04-24: avoid collision with r_val and d_val(=2r)
        ch_pool = [x for x in range(3, 13) if x != r_val and x != 2 * r_val]
        ch = rng.choice(ch_pool) if ch_pool else 3
        ax.plot([-1.5, 1.5], [-1, -1], color="black", lw=1.3)
        ax.annotate(f"{ch}", xy=(0, -1.4), fontsize=16,
                    ha="center", fontweight="bold")
        roles[str(ch)] = "chord"
        if cfg["n_labels"] >= 3:
            # add a diameter
            ax.plot([-1.8, 1.8], [0.9, 0.9], color="black", lw=1.0)
            ax.annotate(f"{d_val}", xy=(0, 1.05), fontsize=14,
                        ha="center", fontweight="bold")
            roles[str(d_val)] = "diameter"
        if cfg["n_labels"] >= 4:
            # Add an arc marker at a point
            ax.annotate("θ=30°", xy=(1.4, -0.7), fontsize=12,
                        ha="center", fontweight="bold")
            roles["30°"] = "arc_angle"
        return self.fig_to_pil(fig), roles

    def _draw_rectangle(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        fig, ax = self._figure_setup(rng, (-0.5, 5.5), (-0.5, 3.5))
        W_val = rng.randint(6, 15)
        H_val = rng.randint(2, 5)
        rect = mpatches.Rectangle((0.5, 0.5), 4, 2,
                                   fc="none", ec="black", lw=2.0)
        ax.add_patch(rect)
        if cfg["n_labels"] == 1:
            which = rng.choice(["width", "height"])
            if which == "width":
                ax.annotate(f"{W_val}", xy=(2.5, 0.2), fontsize=18,
                            ha="center", fontweight="bold")
                roles = {str(W_val): "width"}
            else:
                ax.annotate(f"{H_val}", xy=(0.2, 1.5), fontsize=18,
                            ha="center", fontweight="bold")
                roles = {str(H_val): "height"}
        else:
            ax.annotate(f"{W_val}", xy=(2.5, 0.2), fontsize=16,
                        ha="center", fontweight="bold")
            ax.annotate(f"{H_val}", xy=(0.2, 1.5), fontsize=16,
                        ha="center", fontweight="bold")
            roles = {str(W_val): "width", str(H_val): "height"}
        return self.fig_to_pil(fig), roles

    def _draw_triangle(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        fig, ax = self._figure_setup(rng, (-1, 5), (-1, 4))
        # vertices A (bottom left), B (bottom right), C (top)
        A, B, C = (0, 0), (4, 0), (2, 2.8)
        tri = mpatches.Polygon([A, B, C], fc="none", ec="black", lw=2.0)
        ax.add_patch(tri)
        ax.text(A[0] - 0.25, A[1] - 0.25, "A", fontsize=16, fontweight="bold")
        ax.text(B[0] + 0.1, B[1] - 0.25, "B", fontsize=16, fontweight="bold")
        ax.text(C[0], C[1] + 0.2, "C", fontsize=16, fontweight="bold",
                ha="center")
        side_AB = rng.randint(4, 12)
        angle_A = rng.randint(30, 80)
        if cfg["n_labels"] == 1:
            which = rng.choice(["side_AB", "angle_A"])
            if which == "side_AB":
                ax.annotate(f"{side_AB}", xy=(2, -0.3), fontsize=18,
                            ha="center", fontweight="bold")
                roles = {str(side_AB): "side_AB"}
            else:
                # draw small arc at A
                arc = mpatches.Arc((A[0], A[1]), 1.2, 1.2, angle=0,
                                   theta1=0, theta2=54)
                ax.add_patch(arc)
                ax.annotate(f"{angle_A}°", xy=(0.7, 0.3), fontsize=16,
                            ha="left", fontweight="bold")
                roles = {f"{angle_A}°": "angle_A"}
        else:
            ax.annotate(f"{side_AB}", xy=(2, -0.3), fontsize=16,
                        ha="center", fontweight="bold")
            arc = mpatches.Arc((A[0], A[1]), 1.2, 1.2, angle=0,
                               theta1=0, theta2=54)
            ax.add_patch(arc)
            ax.annotate(f"{angle_A}°", xy=(0.7, 0.3), fontsize=14,
                        ha="left", fontweight="bold")
            roles = {str(side_AB): "side_AB",
                     f"{angle_A}°": "angle_A"}
            if cfg["n_labels"] >= 3:
                # BUGFIX 2026-04-24: side_AC must differ from side_AB to avoid
                # colliding keys in the dict.
                side_AC_pool = [x for x in range(3, 9) if x != side_AB]
                side_AC = rng.choice(side_AC_pool) if side_AC_pool else side_AB + 1
                ax.annotate(f"{side_AC}", xy=(0.6, 1.6), fontsize=14,
                            ha="center", fontweight="bold")
                roles[str(side_AC)] = "side_AC"
        return self.fig_to_pil(fig), roles

    def _draw_cylinder_net(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        """Cylinder unfolded: 2 circles (top/bottom) + rectangle (side)."""
        # BUGFIX 2026-04-24: ensure r_val, h_val, and d_val=2*r_val are all
        # distinct (resample if h_val collides with r_val or d_val).
        fig, ax = self._figure_setup(rng, (-2, 7), (-2, 6))
        r_val = rng.randint(2, 6)
        h_candidates = [x for x in range(5, 13) if x != r_val and x != 2 * r_val]
        h_val = rng.choice(h_candidates)
        # rectangle (side face): width = 2πr (not labeled), height = h_val
        ax.add_patch(mpatches.Rectangle((0, 0), 5, 3, fc="none",
                                         ec="black", lw=2.0))
        # top circle
        ax.add_patch(mpatches.Circle((0.8, 4), 0.7, fc="none",
                                      ec="black", lw=1.8))
        # Show labels
        # radius (on top circle)
        ax.plot([0.8, 1.5], [4, 4], color="black", lw=1.0)
        ax.annotate(f"{r_val}", xy=(1.15, 4.1), fontsize=15,
                    ha="center", fontweight="bold")
        # height of rectangle
        ax.annotate(f"{h_val}", xy=(-0.3, 1.5), fontsize=15,
                    ha="center", fontweight="bold")
        roles = {str(r_val): "radius", str(h_val): "height"}
        if cfg["n_labels"] >= 3:
            d_val = r_val * 2
            # Second circle with diameter instead
            ax.add_patch(mpatches.Circle((3.8, 4), 0.7, fc="none",
                                          ec="black", lw=1.8))
            ax.plot([3.1, 4.5], [4, 4], color="black", lw=1.0)
            ax.annotate(f"{d_val}", xy=(3.8, 4.15), fontsize=15,
                        ha="center", fontweight="bold")
            roles[str(d_val)] = "diameter"
        return self.fig_to_pil(fig), roles

    def _draw_cone_net(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        # BUGFIX 2026-04-24: ensure r_val != slant_val (they're always distinct
        # here since slant >= r+2, but keep defensive).
        fig, ax = self._figure_setup(rng, (-4, 4), (-2, 5))
        r_val = rng.randint(2, 5)
        slant_val = rng.randint(r_val + 2, r_val + 7)
        # Sector (lateral surface)
        sector = mpatches.Wedge((0, 0), 2.5, 30, 150, fc="none",
                                  ec="black", lw=2.0)
        ax.add_patch(sector)
        ax.annotate(f"{slant_val}", xy=(1.0, 1.6), fontsize=15,
                    ha="center", fontweight="bold")
        # Small circle (base)
        ax.add_patch(mpatches.Circle((0, -1.2), 0.8, fc="none",
                                      ec="black", lw=1.8))
        ax.plot([0, 0.8], [-1.2, -1.2], color="black", lw=1.0)
        ax.annotate(f"{r_val}", xy=(0.4, -1.1), fontsize=15,
                    ha="center", fontweight="bold")
        roles = {str(slant_val): "slant_height", str(r_val): "radius"}
        if cfg["n_labels"] >= 3:
            # BUGFIX 2026-04-24: pick h_val distinct from r_val and slant_val.
            h_pool = [v for v in range(r_val + 1, slant_val)
                      if v != r_val and v != slant_val]
            h_val = rng.choice(h_pool) if h_pool else r_val + 1
            ax.plot([-3.0, -3.0], [-1, 2], color="black", lw=1.0, linestyle=":")
            ax.annotate(f"{h_val}", xy=(-3.3, 0.5), fontsize=14,
                        ha="center", fontweight="bold")
            roles[str(h_val)] = "height"
        return self.fig_to_pil(fig), roles

    def _draw_trapezoid(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        # BUGFIX 2026-04-24: ensure top, bottom, height distinct so dict keys
        # don't collide at n_labels>=3.
        fig, ax = self._figure_setup(rng, (-1, 6), (-1, 4))
        top = rng.randint(3, 8)
        bottom = top + rng.randint(2, 6)  # bottom > top by construction
        h_pool = [x for x in range(2, 6) if x != top and x != bottom]
        height = rng.choice(h_pool) if h_pool else 2
        # trapezoid
        pts = [(1, 0), (5, 0), (4, 2.5), (2, 2.5)]
        trap = mpatches.Polygon(pts, fc="none", ec="black", lw=2.0)
        ax.add_patch(trap)
        ax.annotate(f"{bottom}", xy=(3, -0.3), fontsize=15,
                    ha="center", fontweight="bold")
        ax.annotate(f"{top}", xy=(3, 2.75), fontsize=15,
                    ha="center", fontweight="bold")
        roles = {str(bottom): "bottom_base", str(top): "top_base"}
        if cfg["n_labels"] >= 3:
            ax.plot([2, 2], [0, 2.5], color="black", lw=1.0, linestyle=":")
            ax.annotate(f"{height}", xy=(1.7, 1.25), fontsize=14,
                        ha="center", fontweight="bold")
            roles[str(height)] = "height"
        return self.fig_to_pil(fig), roles

    def _draw_rect_in_circle(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        # BUGFIX 2026-04-24: sample distinct values so {radius, rectangle_width,
        # rectangle_height} never collide.
        fig, ax = self._figure_setup(rng, (-3.5, 3.5), (-3.5, 3.5))
        picks = rng.sample([3, 4, 5, 6, 7, 8, 9, 10], 3)
        r_val, rect_w, rect_h = picks[0], picks[1], picks[2]
        ax.add_patch(mpatches.Circle((0, 0), 2.5, fc="none",
                                      ec="black", lw=1.8))
        ax.add_patch(mpatches.Rectangle((-1.5, -1), 3, 2, fc="none",
                                         ec="black", lw=2.0))
        # radius label (diagonal)
        ax.plot([0, 2.5 * math.cos(math.radians(40))],
                [0, 2.5 * math.sin(math.radians(40))], color="black", lw=1.0)
        ax.annotate(f"{r_val}", xy=(1.1, 0.9), fontsize=15,
                    ha="center", fontweight="bold")
        ax.annotate(f"{rect_w}", xy=(0, -1.3), fontsize=15,
                    ha="center", fontweight="bold")
        roles = {str(r_val): "radius", str(rect_w): "rectangle_width"}
        if cfg["n_labels"] >= 3:
            ax.annotate(f"{rect_h}", xy=(-1.85, 0), fontsize=14,
                        ha="center", fontweight="bold")
            roles[str(rect_h)] = "rectangle_height"
        return self.fig_to_pil(fig), roles

    def _draw_compound(self, rng, cfg) -> Tuple[Image.Image, Dict]:
        """Compound figure: rectangle + semicircle on top."""
        # BUGFIX 2026-04-24: sample w,h,r_val_label so all three numeric
        # labels are distinct (previously colliding values silently overwrote
        # dict entries, producing unsolvable samples).
        fig, ax = self._figure_setup(rng, (-1, 6), (-1, 5))
        picks = rng.sample([2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        w, h, r_val_label = picks[0], picks[1], picks[2]
        ax.add_patch(mpatches.Rectangle((0, 0), 4, 2, fc="none",
                                         ec="black", lw=2.0))
        ax.add_patch(mpatches.Wedge((2, 2), 2, 0, 180, fc="none",
                                     ec="black", lw=2.0))
        ax.annotate(f"{w}", xy=(2, -0.3), fontsize=15,
                    ha="center", fontweight="bold")
        ax.annotate(f"{h}", xy=(-0.3, 1), fontsize=15,
                    ha="center", fontweight="bold")
        ax.plot([2, 3.5], [2, 2.7], color="black", lw=1.0)
        ax.annotate(f"{r_val_label}", xy=(2.8, 2.45), fontsize=15,
                    ha="center", fontweight="bold")
        roles = {str(w): "rectangle_width",
                 str(h): "rectangle_height",
                 str(r_val_label): "semicircle_radius"}
        return self.fig_to_pil(fig), roles

    # ---- Answer grading ----

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        if pred == gt:
            return True
        # If GT is a dict, try dict comparison
        if gt.startswith("{"):
            import re
            try:
                import ast
                gt_d = ast.literal_eval(ground_truth)
                # extract dict-like structure from pred
                m = re.search(r"\{[^{}]*\}", predicted)
                if m:
                    pred_d = ast.literal_eval(m.group(0))
                    # normalize keys/values to lower strings
                    def norm(d):
                        return {str(k).strip().lower(): str(v).strip().lower() for k, v in d.items()}
                    return norm(gt_d) == norm(pred_d)
            except Exception:
                return False
        return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_glr"
    os.makedirs(out_dir, exist_ok=True)
    env = GeometryLabelReadingQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 33
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[glr L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/glr_s{s}_L{level}.png")
            print(f"[glr L{level} s{s}] A={env._answer}")
