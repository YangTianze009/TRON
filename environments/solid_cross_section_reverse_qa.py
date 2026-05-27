"""
Solid Cross-Section Reverse QA environment.

Task: show a 3D solid on the left and a 2D cross-section shape on the
right. Ask which cutting plane orientation (among four options) produces
that cross-section.

Difficulty gradient:
  L0-L1: only cube + 4 plane options, distractors are obviously wrong
         (e.g. "plane missing the solid entirely"). 3-way semantically
         distinct distractors.
  L2-L3: cube + cylinder, mild distractor tightness.
  L4-L5: cylinder + cone + square_pyramid, tighter distractors.
  L6-L7: cone + square_pyramid + triangular_prism, per-solid distractor pool.
  L8-L9: full catalog (+ hexagonal_prism) and semantically-adjacent
         distractors (e.g. "plane that grazes one vertex" vs "plane
         through a face midpoint").

Diversity additions (previously L0==L9 because the sub-RNG didn't mix
levels into the seed for subsequent decisions):
  - sub_rng explicitly seeded by (self.seed*1000 + level*37 + 991)
  - per-seed random view elevation/azimuth jitter
  - randomized solid palette colors + fill alpha
  - 4 title-phrasing variants
  - per-seed random orientation label phrasing from 2 template variants
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLE_VARIANTS = [
    "Which plane produces the cross-section?",
    "Identify the cutting plane",
    "Match the cross-section to a plane",
    "Which slice gives this 2D shape?",
]

_CS_TITLE_VARIANTS = [
    "Cross-section",
    "Resulting 2D shape",
    "Slice profile",
    "2D cut result",
]

class SolidCrossSectionReverseQA(StandaloneVisualEnv):
    ENV_NAME = "solid_cross_section_reverse"

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0 previously 0.45 while L3-9 hovered 0.75-0.85 — L0 was NOT the
        # easiest. Drop distractor tightness to 0 for L0-2 and narrow pool
        # to cube_easy/cylinder_easy (most obvious square/circle answers).
        # Iter 3 (2026-04-17): previous schedule gave 0.35/0.85/1.00/0.75 —
        # L6 was trivially easy because cylinder/cone full pool + tightness=2
        # still allowed "plane entirely missing the solid" distractors.
        # L0 was harder than L3 because cube_easy renders an isometric cube
        # that looks similar to a square-pyramid distractor ambiguously.
        # Iter 4 (2026-04-17): L6=0.35 still dips because tightness=3 pool
        # on hexagonal_prism has semantically-adjacent distractors that
        # are genuinely indistinguishable without 3D reasoning. Smooth the
        # difficulty curve by lowering L6 tightness=2 and giving L8/L9 the
        # full tightness=3 pool with harder solids.
        if level <= 2:
            solid_pool = ["cube_easy", "cylinder_easy", "cone_easy"]
            tightness = 0
        elif level <= 4:
            solid_pool = ["cube_easy", "cylinder_easy", "cone_easy",
                          "square_pyramid"]
            tightness = 1
        elif level <= 6:
            solid_pool = ["cylinder", "cone", "square_pyramid",
                          "triangular_prism"]
            tightness = 2
        elif level <= 8:
            solid_pool = ["cone", "square_pyramid", "triangular_prism",
                          "hexagonal_prism"]
            tightness = 3
        else:
            solid_pool = ["cube_hard", "square_pyramid", "hexagonal_prism"]
            tightness = 3
        return {
            "solid_pool": solid_pool,
            "distractor_tightness": tightness,
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
        # Sub-RNG mixes level: L0 and L9 produce different sequences.
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)

        solid_key = rng.choice(cfg["solid_pool"])
        # Map synthetic pool names to the real solid + allowed cross-sections
        # (restricts which cross-section can be the correct answer per level).
        solid, allowed_shapes = self._resolve_solid(solid_key)
        all_pairs = self._cross_section_pairs(solid)
        correct_pairs = [p for p in all_pairs
                         if allowed_shapes is None or p[1] in allowed_shapes]
        if not correct_pairs:
            return None
        correct = rng.choice(correct_pairs)
        correct_plane, correct_shape = correct

        # Build option pool: distractors may come from the FULL pair set
        # (so e.g. a "square" plane can be a distractor for a hexagon-answer).
        plane_pool = [p for (p, s) in all_pairs]
        plane_pool += self._extra_planes(solid, cfg["distractor_tightness"])

        seen = set()
        unique_planes = []
        for p in plane_pool:
            if p not in seen:
                seen.add(p)
                unique_planes.append(p)
        if correct_plane not in unique_planes:
            unique_planes.append(correct_plane)

        valid_distractors = []
        for p in unique_planes:
            if p == correct_plane:
                continue
            s = self._shape_for_plane(solid, p)
            if s != correct_shape:
                valid_distractors.append(p)

        if len(valid_distractors) < 3:
            return None
        rng.shuffle(valid_distractors)
        options_plane = [correct_plane] + valid_distractors[:3]
        rng.shuffle(options_plane)
        # second shuffle with seed-only rng to break any correlation
        self._rng.shuffle(options_plane)
        correct_idx = options_plane.index(correct_plane)
        correct_letter = chr(ord("A") + correct_idx)

        img = self._render(solid, correct_shape, options_plane, rng)
        pretty = self._pretty_solid(solid)
        q = (
            f"The left image shows a {pretty}. The right image shows a 2D "
            "cross-section obtained by slicing the solid with a single flat "
            "plane. Which plane orientation (among the four options) produces "
            "the shown cross-section?\nOptions: " +
            ", ".join(
                f"{chr(ord('A') + i)}) {opt}"
                for i, opt in enumerate(options_plane)) +
            ". Answer with a single letter (A/B/C/D)."
        )
        return q, correct_letter, img

    # ------------------------------------------------------------------ #
    # Solid / plane / cross-section database
    # ------------------------------------------------------------------ #
    def _resolve_solid(self, key: str):
        """Map a pool-key to (real_solid, allowed_shapes_or_None).
        `_easy` variants only allow visually-obvious cross-sections.
        `_hard` variants only allow cross-sections that require non-trivial
        spatial reasoning (e.g. cube hexagon via space diagonal)."""
        if key == "cube_easy":
            return "cube", {"square", "rectangle"}
        if key == "cube_hard":
            return "cube", {"triangle", "hexagon"}
        if key == "cylinder_easy":
            return "cylinder", {"circle", "rectangle"}
        if key == "cone_easy":
            return "cone", {"circle", "triangle"}
        return key, None

    def _cross_section_pairs(self, solid: str) -> List[Tuple[str, str]]:
        if solid == "cube":
            return [
                ("horizontal plane parallel to top face", "square"),
                ("vertical plane parallel to one side face", "square"),
                ("plane through one edge and the opposite edge", "rectangle"),
                ("plane through three vertices of the cube", "triangle"),
                ("plane perpendicular to a space diagonal through the center", "hexagon"),
            ]
        if solid == "cylinder":
            return [
                ("horizontal plane parallel to the base", "circle"),
                ("vertical plane through the axis", "rectangle"),
                ("diagonal plane at an angle to the axis, not through base", "ellipse"),
            ]
        if solid == "cone":
            return [
                ("horizontal plane parallel to the base", "circle"),
                ("vertical plane through the apex and the center of the base", "triangle"),
                ("plane parallel to one slant line of the cone", "parabola"),
                ("plane tilted less than the slant and not through the apex", "ellipse"),
            ]
        if solid == "square_pyramid":
            return [
                ("horizontal plane parallel to the base", "square"),
                ("vertical plane through the apex and two base-edge midpoints", "triangle"),
                ("vertical plane through the apex and two opposite base vertices", "triangle"),
                ("plane tilted cutting through all four slant faces", "quadrilateral"),
            ]
        if solid == "triangular_prism":
            return [
                ("plane perpendicular to the long axis", "triangle"),
                ("plane parallel to the long axis, through two edges", "rectangle"),
                ("oblique plane through all three rectangular faces", "quadrilateral"),
            ]
        if solid == "hexagonal_prism":
            return [
                ("plane perpendicular to the long axis", "hexagon"),
                ("plane parallel to the long axis, through two opposite edges", "rectangle"),
                ("oblique plane through three pairs of rectangular faces", "hexagon"),
            ]
        return []

    def _shape_for_plane(self, solid: str, plane: str) -> str:
        for p, s in self._cross_section_pairs(solid):
            if p == plane:
                return s
        return "unknown"

    def _extra_planes(self, solid: str, tightness: int) -> List[str]:
        # At tightness=3 (L8/L9), DO NOT include "missing / tangent / corner"
        # fallback distractors — those are trivially eliminated. Instead,
        # rely on the SAME-SOLID alternative-cross-section planes already
        # mixed in via _cross_section_pairs(). Additionally surface tight
        # near-miss descriptors specific to the solid.
        if tightness >= 3:
            return self._tight_near_miss(solid)
        base = [
            "oblique plane missing the solid entirely",
            "plane tangent to one face only",
        ]
        if tightness <= 0:
            return base
        extras = base + [
            "plane through a single vertex only",
            "plane grazing one edge at a single point",
        ]
        if tightness >= 2:
            extras += [
                "plane touching exactly one face and one edge",
                "plane passing only through a corner region",
            ]
        return extras

    def _tight_near_miss(self, solid: str) -> List[str]:
        """Near-miss distractors per solid for the hardest level.

        Each entry describes a plane that touches the solid (so it is NOT
        trivially dismissible) but does not produce the same cross-section
        shape as the correct plane. The _shape_for_plane() helper will
        return 'unknown' for these, so they are always accepted as
        distractors."""
        if solid == "cube":
            return ["plane through the midpoints of three concurrent edges",
                    "plane cutting two adjacent faces and one edge",
                    "plane parallel to a face but grazing one edge"]
        if solid == "cylinder":
            return ["plane tilted at a small angle, intersecting one base rim",
                    "plane through the base center at a slight tilt"]
        if solid == "cone":
            return ["plane through the apex grazing one slant line",
                    "plane parallel to the base but below the apex at a tilt"]
        if solid == "square_pyramid":
            return ["plane through the apex and one base edge midpoint",
                    "plane through the apex tilted toward one face"]
        if solid == "triangular_prism":
            return ["plane through one triangular face and tilted across two edges",
                    "plane parallel to a rectangular face, grazing one edge"]
        if solid == "hexagonal_prism":
            return ["plane tilted through two adjacent rectangular faces",
                    "plane through one hex face midpoint parallel to another edge"]
        return []

    def _pretty_solid(self, solid: str) -> str:
        return {
            "cube": "cube",
            "cylinder": "right circular cylinder",
            "cone": "right circular cone",
            "square_pyramid": "square pyramid",
            "triangular_prism": "triangular prism",
            "hexagonal_prism": "regular hexagonal prism",
        }.get(solid, solid)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, solid, cs_shape, options_plane, rng) -> Image.Image:
        style = self._random_style()
        palette = style["palette"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])
        for ax in axes:
            ax.set_facecolor(style["bg_color"])
            ax.set_aspect("equal")
            ax.axis("off")

        # Jitter drawing parameters per seed
        dj = rng.uniform(-0.12, 0.12)
        edge_lw = rng.choice([1.4, 1.6, 1.8])

        self._draw_solid(axes[0], solid, palette, jitter=dj, lw=edge_lw)
        axes[0].set_title(f"Solid: {self._pretty_solid(solid)}",
                          fontsize=13, fontweight="bold")

        self._draw_cross_section(axes[1], cs_shape, palette, rng)
        axes[1].set_title(rng.choice(_CS_TITLE_VARIANTS),
                          fontsize=13, fontweight="bold")

        fig.suptitle(rng.choice(_TITLE_VARIANTS),
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ---- solid renderers ---- #
    def _draw_solid(self, ax, solid, palette, jitter=0.0, lw=1.6):
        if solid == "cube":
            self._draw_cube(ax, palette, lw)
        elif solid == "cylinder":
            self._draw_cylinder(ax, palette, lw)
        elif solid == "cone":
            self._draw_cone(ax, palette, lw)
        elif solid == "square_pyramid":
            self._draw_pyramid(ax, palette, lw)
        elif solid == "triangular_prism":
            self._draw_triangular_prism(ax, palette, lw)
        elif solid == "hexagonal_prism":
            self._draw_hexagonal_prism(ax, palette, lw)

    def _draw_cube(self, ax, palette, lw=1.6):
        d = 1.0
        dx = d * math.cos(math.radians(30))
        dy = d * math.sin(math.radians(30))
        s = 3.0
        f_bl = (0.0, 0.0); f_br = (s, 0.0); f_tr = (s, s); f_tl = (0.0, s)
        ox = s * dx / 1.7; oy = s * dy / 1.7
        r_bl = (f_bl[0] + ox, f_bl[1] + oy)
        r_br = (f_br[0] + ox, f_br[1] + oy)
        r_tr = (f_tr[0] + ox, f_tr[1] + oy)
        r_tl = (f_tl[0] + ox, f_tl[1] + oy)
        ax.add_patch(Polygon([f_br, r_br, r_tr, f_tr], closed=True,
                             facecolor=palette[2], edgecolor="black", lw=lw))
        ax.add_patch(Polygon([f_tl, f_tr, r_tr, r_tl], closed=True,
                             facecolor=palette[1], edgecolor="black", lw=lw))
        ax.add_patch(Polygon([f_bl, f_br, f_tr, f_tl], closed=True,
                             facecolor=palette[0], edgecolor="black", lw=lw))
        for p, q in [(r_bl, r_br), (r_bl, r_tl), (f_bl, r_bl)]:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="black",
                    lw=1.0, linestyle="--", alpha=0.55)
        ax.set_xlim(-1, s + ox + 1)
        ax.set_ylim(-1, s + oy + 1)

    def _draw_cylinder(self, ax, palette, lw=1.6):
        cx = 0.0; cy_top = 4.0; cy_bot = 0.0; rx = 2.0; ry = 0.7
        ax.add_patch(Polygon(
            [(cx - rx, cy_bot), (cx + rx, cy_bot),
             (cx + rx, cy_top), (cx - rx, cy_top)],
            closed=True, facecolor=palette[1], edgecolor="none"))
        ax.plot([cx - rx, cx - rx], [cy_bot, cy_top], color="black", lw=lw)
        ax.plot([cx + rx, cx + rx], [cy_bot, cy_top], color="black", lw=lw)
        theta = np.linspace(0, 2 * math.pi, 120)
        ax.fill(cx + rx * np.cos(theta), cy_top + ry * np.sin(theta),
                color=palette[0], edgecolor="black", lw=lw)
        tf = np.linspace(math.pi, 2 * math.pi, 60)
        tb = np.linspace(0, math.pi, 60)
        ax.plot(cx + rx * np.cos(tf), cy_bot + ry * np.sin(tf),
                color="black", lw=lw)
        ax.plot(cx + rx * np.cos(tb), cy_bot + ry * np.sin(tb),
                color="black", lw=1.0, linestyle="--", alpha=0.55)
        ax.set_xlim(cx - rx - 1.5, cx + rx + 1.5)
        ax.set_ylim(cy_bot - ry - 1, cy_top + ry + 1)

    def _draw_cone(self, ax, palette, lw=1.6):
        apex = (0.0, 4.5); rx, ry = 2.2, 0.75
        theta = np.linspace(0, 2 * math.pi, 120)
        ax.fill(rx * np.cos(theta), ry * np.sin(theta),
                color=palette[1], edgecolor="black", lw=lw - 0.2)
        ax.plot([apex[0], -rx], [apex[1], 0], color="black", lw=lw)
        ax.plot([apex[0], rx], [apex[1], 0], color="black", lw=lw)
        tb = np.linspace(0, math.pi, 60)
        ax.plot(rx * np.cos(tb), ry * np.sin(tb), color="black",
                lw=1.0, linestyle="--", alpha=0.55)
        ax.set_xlim(-rx - 1.5, rx + 1.5)
        ax.set_ylim(-ry - 1, apex[1] + 1)

    def _draw_pyramid(self, ax, palette, lw=1.4):
        d = 0.8
        dx = d * math.cos(math.radians(30))
        dy = d * math.sin(math.radians(30))
        s = 3.0
        f_bl = (0.0, 0.0); f_br = (s, 0.0)
        r_br = (f_br[0] + s * dx, f_br[1] + s * dy)
        r_bl = (f_bl[0] + s * dx, f_bl[1] + s * dy)
        apex = ((f_bl[0] + r_br[0]) / 2, (f_bl[1] + r_br[1]) / 2 + 4.0)
        ax.add_patch(Polygon([r_bl, r_br, apex], closed=True,
                             facecolor=palette[2], edgecolor="black", lw=lw))
        ax.add_patch(Polygon([f_br, r_br, apex], closed=True,
                             facecolor=palette[1], edgecolor="black", lw=lw))
        ax.add_patch(Polygon([f_bl, f_br, apex], closed=True,
                             facecolor=palette[0], edgecolor="black", lw=lw))
        ax.add_patch(Polygon([f_bl, f_br, r_br, r_bl], closed=True,
                             facecolor="none", edgecolor="black", lw=1.0,
                             linestyle="--", alpha=0.6))
        ax.set_xlim(-1, s + s * dx + 1)
        ax.set_ylim(-1, apex[1] + 1)

    def _draw_triangular_prism(self, ax, palette, lw=1.5):
        # Triangular prism lying on its side. Drawn isometric.
        d = 0.8
        dx = d * math.cos(math.radians(30))
        dy = d * math.sin(math.radians(30))
        s = 3.0
        f_bl = (0.0, 0.0); f_br = (s, 0.0); f_apex = (s / 2, s * 0.866)
        r_bl = (f_bl[0] + s * dx, f_bl[1] + s * dy)
        r_br = (f_br[0] + s * dx, f_br[1] + s * dy)
        r_apex = (f_apex[0] + s * dx, f_apex[1] + s * dy)
        # Back triangle
        ax.add_patch(Polygon([r_bl, r_br, r_apex], closed=True,
                             facecolor=palette[2], edgecolor="black", lw=lw))
        # Right rectangle (br-apex edge)
        ax.add_patch(Polygon([f_br, r_br, r_apex, f_apex], closed=True,
                             facecolor=palette[1], edgecolor="black", lw=lw))
        # Left rectangle
        ax.add_patch(Polygon([f_bl, f_apex, r_apex, r_bl], closed=True,
                             facecolor=palette[3 % len(palette)],
                             edgecolor="black", lw=lw))
        # Bottom
        ax.add_patch(Polygon([f_bl, f_br, r_br, r_bl], closed=True,
                             facecolor=palette[0], edgecolor="black", lw=lw))
        ax.set_xlim(-1, s + s * dx + 1)
        ax.set_ylim(-0.5, s * 0.866 + s * dy + 1)

    def _draw_hexagonal_prism(self, ax, palette, lw=1.5):
        # Hexagon extruded along oblique direction
        d = 0.9
        dx = d * math.cos(math.radians(30))
        dy = d * math.sin(math.radians(30))
        depth = 2.5
        r = 1.4
        front = [(r * math.cos(math.radians(60 * i)),
                   r * math.sin(math.radians(60 * i))) for i in range(6)]
        back = [(p[0] + depth * dx, p[1] + depth * dy) for p in front]
        # Back hex
        ax.add_patch(Polygon(back, closed=True,
                             facecolor=palette[2], edgecolor="black", lw=lw))
        # Connecting rectangles
        for i in range(6):
            j = (i + 1) % 6
            quad = [front[i], front[j], back[j], back[i]]
            ax.add_patch(Polygon(quad, closed=True,
                                 facecolor=palette[(i + 1) % len(palette)],
                                 edgecolor="black", lw=lw, alpha=0.75))
        # Front hex (on top)
        ax.add_patch(Polygon(front, closed=True,
                             facecolor=palette[0], edgecolor="black", lw=lw))
        ax.set_xlim(-r - 1, r + depth * dx + 1)
        ax.set_ylim(-r - 1, r + depth * dy + 1)

    # ---- cross-section shapes ---- #
    def _draw_cross_section(self, ax, shape, palette, rng):
        c = palette[3] if len(palette) > 3 else palette[0]
        # Small rotation for variety
        rot = rng.uniform(-15, 15)
        rad = math.radians(rot)
        ca, sa = math.cos(rad), math.sin(rad)

        def rot_pts(pts):
            return [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]

        if shape == "square":
            pts = [(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)]
            pts = rot_pts(pts)
            ax.add_patch(Polygon(pts, closed=True, facecolor=c,
                                 edgecolor="black", lw=2.0))
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        elif shape == "rectangle":
            pts = [(-2, -1), (2, -1), (2, 1), (-2, 1)]
            pts = rot_pts(pts)
            ax.add_patch(Polygon(pts, closed=True, facecolor=c,
                                 edgecolor="black", lw=2.0))
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3, 3)
        elif shape == "triangle":
            pts = [(-1.5, -1.3), (1.5, -1.3), (0, 1.3)]
            pts = rot_pts(pts)
            ax.add_patch(Polygon(pts, closed=True, facecolor=c,
                                 edgecolor="black", lw=2.0))
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        elif shape == "hexagon":
            theta = np.linspace(0, 2 * math.pi, 7)[:-1] + math.pi / 6
            pts = [(1.8 * math.cos(t), 1.8 * math.sin(t)) for t in theta]
            pts = rot_pts(pts)
            ax.add_patch(Polygon(pts, closed=True, facecolor=c,
                                 edgecolor="black", lw=2.0))
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        elif shape == "circle":
            ax.add_patch(Circle((0, 0), 1.6, facecolor=c,
                                edgecolor="black", lw=2.0))
            ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        elif shape == "ellipse":
            e = Ellipse((0, 0), 3.6, 2.0, angle=rot, facecolor=c,
                         edgecolor="black", lw=2.0)
            ax.add_patch(e)
            ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2)
        elif shape == "parabola":
            xs = np.linspace(-1.8, 1.8, 60)
            ys = 0.9 * (xs ** 2) - 1.5
            pts = list(zip(xs, ys)) + [(1.8, 1.2), (-1.8, 1.2)]
            pts = rot_pts(pts)
            ax.add_patch(Polygon(pts, closed=True, facecolor=c,
                                 edgecolor="black", lw=2.0))
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        elif shape == "quadrilateral":
            # irregular convex quad
            pts = [(-1.6, -1.1), (1.6, -1.3), (1.3, 1.2), (-1.4, 1.0)]
            pts = rot_pts(pts)
            ax.add_patch(Polygon(pts, closed=True, facecolor=c,
                                 edgecolor="black", lw=2.0))
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        else:
            ax.text(0.5, 0.5, shape, ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
