"""
Orthographic projection QA — given the three orthographic views (top, front,
side) of a 3D solid, identify the solid via single-letter MCQ.

2026-05-05 R5 P3: REWRITTEN to align with the Understanding of Solid Figures
benchmark style. Old design rendered a 3D block structure and asked
comma-separated profile / row-sum / col-sum questions; that diverged from
the benchmark which is "given views, identify the solid" MCQ.

Pure single-letter MCQ A-D / A-E with "No correct answer" trap.

Per-level design:
  L0/L1 — 4-MCQ. Single basic solid (cube/cuboid/cylinder/cone) with all 3
    views drawn clearly.
  L2/L3 — 5-MCQ with "E. No correct answer" (10% trap). Adds sphere /
    pyramid / triangular_prism.
  L4-L6 — 5-MCQ. 15% trap. Trickier dims (cuboid that resembles cube).
  L7 — 5-MCQ tighter distractors (cylinder vs cone share top circle), 25% trap.
  L8/L9 — 5-MCQ + 40% trap.

Image: matplotlib panel with 3 sub-axes labeled "Top view", "Front view",
"Side view" — each shows the silhouette of that orthographic projection.

Question phrasing matches standard textbook style:
  Q2: "The cuboid shown in the diagram below has a total of ___ faces. ( )"
  Q17 phrasing inverse: "The 3 views shown are of which of the following
                          solids?"
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Map solid -> (top_view_shape, front_view_shape, side_view_shape)
def _three_views(solid: str, dims: dict) -> Tuple[str, str, str]:
    if solid == "cube":
        return ("square", "square", "square")
    if solid == "cuboid":
        l, w, h = dims["l"], dims["w"], dims["h"]
        top = "square" if l == w else "rectangle"
        front = "square" if l == h else "rectangle"
        side = "square" if w == h else "rectangle"
        return (top, front, side)
    if solid == "cylinder":
        return ("circle", "rectangle", "rectangle")
    if solid == "cone":
        return ("circle", "isosceles triangle", "isosceles triangle")
    if solid == "sphere":
        return ("circle", "circle", "circle")
    if solid == "pyramid":  # square pyramid
        return ("square", "isosceles triangle", "isosceles triangle")
    if solid == "triangular_prism":
        return ("rectangle", "isosceles triangle", "rectangle")
    return ("unknown", "unknown", "unknown")


def _all_solid_names() -> List[str]:
    return ["cube", "cuboid", "cylinder", "cone", "sphere",
            "pyramid", "triangular_prism"]


def _solid_label(solid: str) -> str:
    return {
        "cube": "Cube",
        "cuboid": "Rectangular cuboid",
        "cylinder": "Cylinder",
        "cone": "Cone",
        "sphere": "Sphere",
        "pyramid": "Square pyramid",
        "triangular_prism": "Triangular prism",
    }[solid]


def _solid_pool_for_level(level):
    if level <= 1:
        return ["cube", "cuboid", "cylinder", "cone"]
    if level <= 3:
        return ["cube", "cuboid", "cylinder", "cone", "sphere"]
    if level <= 6:
        return ["cube", "cuboid", "cylinder", "cone", "sphere",
                "pyramid", "triangular_prism"]
    return ["cuboid", "cylinder", "cone", "sphere",
            "pyramid", "triangular_prism"]


class OrthographicProjectionQA(StandaloneVisualEnv):
    """Given top/front/side views, pick which 3D solid produced them."""

    ALLOW_ROTATION = False  # orientation-sensitive
    ENV_NAME = "orthographic_projection"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"use_5_options": False, "trap_rate": 0.0,
                    "tight_distractors": False}
        if level <= 3:
            return {"use_5_options": True, "trap_rate": 0.10,
                    "tight_distractors": False}
        if level <= 6:
            return {"use_5_options": True, "trap_rate": 0.15,
                    "tight_distractors": False}
        if level == 7:
            return {"use_5_options": True, "trap_rate": 0.25,
                    "tight_distractors": True}
        return {"use_5_options": True, "trap_rate": 0.40,
                "tight_distractors": True}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 6133)

        for _ in range(20):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        solid_pool = _solid_pool_for_level(level)
        solid = rng.choice(solid_pool)
        dims = self._gen_dims(solid, rng)
        true_views = _three_views(solid, dims)
        if "unknown" in true_views:
            return None

        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        all_solids = _all_solid_names()
        distractor_pool = [s for s in all_solids if s != solid]
        rng.shuffle(distractor_pool)

        if cfg["tight_distractors"]:
            similar = {
                "cube": ["cuboid"],
                "cuboid": ["cube"],
                "cylinder": ["cone"],
                "cone": ["pyramid", "cylinder"],
                "pyramid": ["cone"],
                "triangular_prism": ["pyramid"],
                "sphere": ["cylinder"],
            }
            tight = [s for s in distractor_pool if s in similar.get(solid, [])]
            other = [s for s in distractor_pool if s not in tight]
            distractor_pool = tight + other

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            chosen = distractor_pool[:n_value]
            if len(chosen) < n_value:
                return None
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distractor_pool[:n_distractor]
            if len(chosen) < n_distractor:
                return None
            options = [solid] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(solid)]

        opt_lines = "\n".join(
            f"{opt_letters[i]}. {_solid_label(options[i])}"
            for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        # Standard textbook phrasing
        stems = [
            (
                "As shown in the diagram, the top view, front view, and "
                "side view of a three-dimensional solid are given. The "
                "solid is which of the following ( )?"
            ),
            (
                "The diagram below shows the three orthographic views "
                "(top / front / side) of a 3D solid. Based on these "
                "views, identify the solid ( )."
            ),
            (
                "As shown in the figure, the three views (top, front, "
                "side) of a solid figure are given. Which of the "
                "following solids do these views correspond to ( )?"
            ),
        ]
        stem = rng.choice(stems)
        question = (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )
        image = self._render_three_views(solid, dims, true_views, rng)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    def _gen_dims(self, solid: str, rng: random.Random) -> dict:
        if solid == "cube":
            return {"s": rng.randint(2, 5)}
        if solid == "cuboid":
            l = rng.randint(2, 6)
            w = rng.randint(2, 6)
            h = rng.randint(2, 6)
            if l == w == h:
                w = w + 1
            return {"l": l, "w": w, "h": h}
        if solid == "cylinder":
            return {"r": rng.randint(2, 4), "h": rng.randint(3, 6)}
        if solid == "cone":
            return {"r": rng.randint(2, 4), "h": rng.randint(3, 6)}
        if solid == "sphere":
            return {"r": rng.randint(2, 4)}
        if solid == "pyramid":
            return {"b": rng.randint(2, 4), "h": rng.randint(3, 5)}
        if solid == "triangular_prism":
            return {"b": rng.randint(2, 4), "h_tri": rng.randint(2, 4),
                    "length": rng.randint(3, 6)}
        return {}

    # ------------------------------------------------------------------ #
    # Render the 3 view panels (top / front / side)
    # ------------------------------------------------------------------ #
    def _render_three_views(self, solid: str, dims: dict,
                            views: Tuple[str, str, str],
                            rng: random.Random) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        bg = style["bg_color"]

        fig, axes = plt.subplots(1, 3, figsize=(11 * sc, 4.5 * sc))
        fig.patch.set_facecolor(bg)
        for ax in axes:
            ax.set_facecolor(bg)
            ax.set_aspect("equal")
            ax.axis("off")

        labels = ["Top view", "Front view", "Side view"]
        # Compute 2D shapes parameters for each view
        shape_params = self._view_dims(solid, dims)
        for i, (ax, lbl, view_type) in enumerate(zip(axes, labels,
                                                     ["top", "front", "side"])):
            self._draw_2d_shape(ax, views[i], shape_params[view_type], rng, style)
            ax.set_title(lbl, fontsize=style["font_size_base"] + 2,
                         fontweight="bold")
        suptitle = "Three orthographic views of a solid"
        fig.suptitle(suptitle,
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", y=0.98)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

    def _view_dims(self, solid: str, dims: dict) -> dict:
        """For each view, compute (width, height) of the silhouette."""
        out = {}
        if solid == "cube":
            s = dims["s"]
            out["top"] = {"w": s, "h": s}
            out["front"] = {"w": s, "h": s}
            out["side"] = {"w": s, "h": s}
        elif solid == "cuboid":
            l, w, h = dims["l"], dims["w"], dims["h"]
            out["top"] = {"w": l, "h": w}      # looking down: x-y rect
            out["front"] = {"w": l, "h": h}    # looking from -y: x-z rect
            out["side"] = {"w": w, "h": h}     # looking from +x: y-z rect
        elif solid == "cylinder":
            r, h = dims["r"], dims["h"]
            out["top"] = {"r": r}
            out["front"] = {"w": 2 * r, "h": h}
            out["side"] = {"w": 2 * r, "h": h}
        elif solid == "cone":
            r, h = dims["r"], dims["h"]
            out["top"] = {"r": r}
            out["front"] = {"base": 2 * r, "h": h}
            out["side"] = {"base": 2 * r, "h": h}
        elif solid == "sphere":
            r = dims["r"]
            out["top"] = {"r": r}
            out["front"] = {"r": r}
            out["side"] = {"r": r}
        elif solid == "pyramid":
            b, h = dims["b"], dims["h"]
            out["top"] = {"w": b, "h": b}     # square outline
            out["front"] = {"base": b, "h": h}
            out["side"] = {"base": b, "h": h}
        elif solid == "triangular_prism":
            b, h_tri, length = dims["b"], dims["h_tri"], dims["length"]
            out["top"] = {"w": b, "h": length}     # rectangle (footprint)
            out["front"] = {"base": b, "h": h_tri}  # triangle (cross-section)
            out["side"] = {"w": length, "h": h_tri}  # rectangle
        return out

    def _draw_2d_shape(self, ax, shape: str, params: dict,
                       rng: random.Random, style: dict):
        palette = list(style["palette"])
        rng.shuffle(palette)
        fill = palette[0]
        edge = "#222"

        # Use bounded units -- pick max dim for axis.
        if shape == "square":
            s = params.get("w") or params.get("h") or 2
            sq = Rectangle((-s / 2, -s / 2), s, s,
                           facecolor=fill, edgecolor=edge,
                           linewidth=2.0, alpha=0.55)
            ax.add_patch(sq)
            mx = s * 0.9
            ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx)
        elif shape == "rectangle":
            w = params["w"]; h = params["h"]
            r = Rectangle((-w / 2, -h / 2), w, h,
                          facecolor=fill, edgecolor=edge,
                          linewidth=2.0, alpha=0.55)
            ax.add_patch(r)
            mx = max(w, h) * 0.9
            ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx)
        elif shape == "circle":
            r = params["r"]
            c = Circle((0, 0), r, facecolor=fill, edgecolor=edge,
                       linewidth=2.0, alpha=0.55)
            ax.add_patch(c)
            ax.set_xlim(-r * 1.5, r * 1.5)
            ax.set_ylim(-r * 1.5, r * 1.5)
        elif shape == "isosceles triangle":
            base = params.get("base") or params.get("w") or 2
            height = params.get("h") or 2
            pts = [(-base / 2, -height / 2),
                   (base / 2, -height / 2),
                   (0, height / 2)]
            t = Polygon(pts, closed=True, facecolor=fill,
                        edgecolor=edge, linewidth=2.0, alpha=0.55)
            ax.add_patch(t)
            mx = max(base, height) * 0.9
            ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx)
        elif shape == "ellipse":
            a = params.get("w", 2) / 2
            b = params.get("h", 1) / 2
            e = Ellipse((0, 0), 2 * a, 2 * b, facecolor=fill,
                        edgecolor=edge, linewidth=2.0, alpha=0.55)
            ax.add_patch(e)
            ax.set_xlim(-a * 1.5, a * 1.5)
            ax.set_ylim(-a * 1.5, a * 1.5)
        else:
            ax.text(0.5, 0.5, "?", ha="center", va="center")

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        return super()._check_answer(predicted, ground_truth)


if __name__ == "__main__":
    env = OrthographicProjectionQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
