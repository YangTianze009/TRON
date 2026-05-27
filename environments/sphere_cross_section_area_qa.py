"""
Cross-Section Identification QA -- redesigned to match cross-section.

Format (matches the benchmark):
  Top row: a 3D combined solid rendered from TWO different viewing angles.
  Bottom row: four candidate 2D shapes labeled A/B/C/D.

  Question:
    Either "Which of the following IS a cross-section of the shape?"
    or     "Which of the following CANNOT be a cross-section of the shape?"

This matches cross-section, where Qwen3-VL-8B-Instruct
scores about 76.7%.

Difficulty axes:
  L0..L1: simple solid (cone, cylinder, sphere, prism, pyramid, cube),
          axis-aligned cut, distractor obviously wrong shape.
  L2..L5: simple solid + a single oblique cut option as a distractor;
          OR a stacked combination of two basic solids.
  L6..L9: combined shape (two stacked solids) + subtle distractors
          (e.g., circle vs. ellipse, square vs. rectangle).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle, Circle
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ---------------------------------------------------------------------------- #
#  Shape catalog: (solid_name, possible_cross_section_set)                     #
# ---------------------------------------------------------------------------- #
# Possible cross-section shape vocab.
# We use a compact label so we can match easily.
ALL_SHAPES = [
    "circle", "ellipse", "square", "rectangle", "triangle",
    "trapezoid", "pentagon", "hexagon",
]

# Map: solid -> set of POSSIBLE cross-section shapes.
SOLID_CROSS_SECTIONS = {
    # Basic solids
    "sphere":            {"circle"},
    "cylinder":          {"circle", "ellipse", "rectangle"},
    "cone":              {"circle", "ellipse", "triangle"},
    "cube":              {"square", "rectangle", "triangle", "pentagon", "hexagon"},
    "square_pyramid":    {"square", "triangle", "trapezoid"},
    "triangular_prism":  {"triangle", "rectangle"},
    "rectangular_prism": {"rectangle", "triangle"},
    "square_frustum":    {"square", "trapezoid", "rectangle"},
    "cone_frustum":      {"circle", "ellipse", "trapezoid"},
}

# For combined (stacked) solids, the cross-section set is the UNION of the
# two parts' cross-sections (since you could cut through one, the other,
# or where they meet).
COMBINED_PAIRS = [
    ("cone", "square_frustum"),
    ("triangular_prism", "cylinder"),
    ("square_frustum", "rectangular_prism"),
    ("cone", "cylinder"),
    ("square_pyramid", "cube"),
    ("cone_frustum", "cylinder"),
    ("cone", "cube"),
    # NOTE: triangular_prism + rectangular_prism is excluded because the
    # union of cross-sections is only {triangle, rectangle} (size 2),
    # which is too small for a 3-distractor "cannot" question.
]

# Human-readable name for the description.
PRETTY_NAMES = {
    "sphere": "sphere",
    "cylinder": "cylinder",
    "cone": "cone",
    "cube": "cube",
    "square_pyramid": "square pyramid",
    "triangular_prism": "triangular prism",
    "rectangular_prism": "rectangular prism",
    "square_frustum": "square frustum",
    "cone_frustum": "(circular) frustum",
}

# ---------------------------------------------------------------------------- #
#  Environment                                                                 #
# ---------------------------------------------------------------------------- #

class SphereCrossSectionAreaQA(StandaloneVisualEnv):
    """Class name kept the same for registration; the task is now
    'cross-section shape identification' instead of 'compute area'."""
    ENV_NAME = "sphere_cross_section_area"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "combined": False,
                "candidate_solids": ["sphere", "cone", "cylinder", "cube"],
                "subtle_distractors": False,
                "question_form": "is",   # "Which IS a cross-section?"
            }
        if level <= 3:
            # Don't use sphere/triangular_prism with 'cannot' form: their
            # cross-section sets are too small (<3), so we can't construct
            # 3 valid distractors.
            return {
                "combined": False,
                "candidate_solids": ["cone", "cylinder", "cube",
                                      "square_pyramid"],
                "subtle_distractors": False,
                "question_form": "cannot",   # benchmark form
            }
        if level <= 5:
            # All these solids have |cross-sections| >= 3, allowing the
            # 'cannot' form to use 3 valid distractors + 1 impossible answer.
            return {
                "combined": False,
                "candidate_solids": ["cone", "cylinder", "cube",
                                      "square_pyramid",
                                      "square_frustum", "cone_frustum"],
                "subtle_distractors": True,
                "question_form": "cannot",
            }
        if level <= 7:
            return {
                "combined": True,
                "subtle_distractors": False,
                "question_form": "cannot",
            }
        return {
            "combined": True,
            "subtle_distractors": True,
            "question_form": "cannot",
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1013)

        # ----- Pick the solid (or combined pair) ----- #
        if cfg["combined"]:
            top_part, bottom_part = rng.choice(COMBINED_PAIRS)
            possible = (SOLID_CROSS_SECTIONS[top_part]
                        | SOLID_CROSS_SECTIONS[bottom_part])
            solid_desc = (f"{PRETTY_NAMES[top_part]} on top of "
                          f"{PRETTY_NAMES[bottom_part]}")
            solid_id = (top_part, bottom_part)
        else:
            solid_id = rng.choice(cfg["candidate_solids"])
            possible = SOLID_CROSS_SECTIONS[solid_id]
            solid_desc = PRETTY_NAMES[solid_id]

        impossible = [s for s in ALL_SHAPES if s not in possible]
        if not impossible or len(possible) < 1:
            return None

        # ----- Build options based on question form ----- #
        if cfg["question_form"] == "is":
            # Correct = a possible shape; 3 distractors are NOT possible.
            correct_shape = rng.choice(sorted(possible))
            n_distractors = 3
            if len(impossible) >= n_distractors:
                distractors = rng.sample(sorted(impossible), n_distractors)
            else:
                distractors = sorted(impossible) + rng.sample(
                    sorted(possible - {correct_shape}),
                    n_distractors - len(impossible))
            options = [correct_shape] + distractors
        else:
            # "Which CANNOT be a cross-section?"
            # Correct (answer) = an IMPOSSIBLE shape;
            # 3 distractors are POSSIBLE shapes (so the model needs
            # to reason about which ONE is the impostor).
            correct_shape = rng.choice(sorted(impossible))
            possible_list = sorted(possible)
            if len(possible_list) < 3:
                # Cannot construct a clean "cannot" question (would
                # require multiple impossible shapes among options,
                # which is ambiguous). Reject and let the caller retry.
                return None
            distractors = rng.sample(possible_list, 3)
            options = [correct_shape] + distractors

        # Optionally tighten distractors at higher levels: ensure that
        # the impossible/possible pair includes a near-miss
        # (e.g., circle vs. ellipse, square vs. rectangle).
        if cfg["subtle_distractors"]:
            options = self._maybe_swap_for_near_miss(
                options, possible, correct_shape, cfg["question_form"], rng)

        rng.shuffle(options)
        correct_idx = options.index(correct_shape)
        correct_letter = "ABCD"[correct_idx]

        # ----- Render figure ----- #
        img = self._render(solid_id, solid_desc, options, rng,
                           combined=cfg["combined"])

        sidx = (self.seed or 0) % 16
        if cfg["question_form"] == "is":
            is_templates = [
                "The top row shows the shape from two different angles. The shape is a {desc}. Which of the following IS a possible cross-section of the shape? Please answer from options A, B, C, or D.",
                "Two views of a {desc} appear in the top row. Which of the four 2D shapes below IS a valid cross-section of this solid? Answer with A, B, C, or D.",
                "You see a {desc} rendered from two perspectives at the top. Identify which of the candidate shapes (A-D) below IS a legitimate cross-section. Respond with a single letter.",
                "The figure's upper half shows a {desc} from two viewpoints. Among the four choices below, which shape IS achievable as a cross-section? Answer A, B, C, or D.",
                "Look at the {desc} drawn in two views at the top. Which of the shapes A-D below IS something you could obtain by slicing this solid? Reply with one letter.",
                "A {desc} is depicted twice (different angles) at the top. Select the shape from (A)-(D) below that IS a possible cross-section. Give only the letter.",
                "The top of the figure shows a {desc} from two angles. Which answer choice (A, B, C, or D) IS a valid cross-section of the solid?",
                "Two renderings of a {desc} appear in the upper row. Which 2D shape among options A-D IS a possible cross-section? Choose one letter.",
                "The shape shown in two views is a {desc}. Which labeled option below IS a cross-section that this solid can produce? Answer with a letter.",
                "Inspect the two views of the {desc}. Which of the answer candidates (A-D) IS a shape that can appear as a planar cross-section? Single letter answer.",
                "Top row: two views of a {desc}. Determine which option (A, B, C, or D) IS a possible cross-section and respond with that letter.",
                "The figure shows a {desc} from two angles. Among A-D, which IS an achievable cross-section? Answer with one capital letter.",
                "We render a {desc} from two perspectives. Pick the option (A-D) that IS a valid cross-section shape. Respond with the letter only.",
                "Presented above is a {desc} (two views). Which of the bottom-row shapes IS a cross-section of this solid? Give A, B, C, or D.",
                "The solid is a {desc}, shown in two views. Which labelled shape IS a possible cross-section? Reply with one of A/B/C/D.",
                "Study the two views of the {desc}. Which of the options A-D IS a shape you could obtain as a cross-section? Answer with a single letter.",
            ]
            q = is_templates[sidx].format(desc=solid_desc)
        else:
            if cfg["combined"]:
                desc_text = f"consists of a {solid_desc}"
                prefix = "combined "
            else:
                desc_text = f"is a {solid_desc}"
                prefix = ""
            cannot_templates = [
                "The top row shows the {prefix}shape viewed from two different angles. The shape {desc_text}. Which of the following images CANNOT be a cross-section of the shape? Please answer from options A, B, C, or D.",
                "Two views of the {prefix}solid appear above; the shape {desc_text}. Among options A-D, which CANNOT be a cross-section? Answer with a letter.",
                "The {prefix}shape {desc_text}, as seen in two views at the top. Which of the answer choices CANNOT be a valid cross-section? Respond with a single letter.",
                "Inspect the two views of the {prefix}shape ({desc_text}). One of the options A-D is impossible as a cross-section — which? Give A, B, C, or D.",
                "Above are two renderings of the {prefix}shape, which {desc_text}. Which option below CANNOT be obtained as a cross-section? Reply with a letter.",
                "Look at the two viewpoints of the {prefix}solid ({desc_text}). Which of the four labelled shapes CANNOT arise as a cross-section? Single-letter answer.",
                "The solid shown above ({desc_text}) is depicted from two angles. Identify the option (A-D) that CANNOT be a cross-section of it. Answer with one letter.",
                "Two views are provided of the {prefix}shape ({desc_text}). Which of the candidates A-D CANNOT be produced as a cross-section? Reply A, B, C, or D.",
                "The {prefix}figure {desc_text}. Two angles are shown. Which of the four shapes CANNOT be a cross-section? Give the letter.",
                "Given the two views of the {prefix}shape that {desc_text}, which option CANNOT be a cross-section? Answer A, B, C, or D.",
                "Above, the {prefix}shape ({desc_text}) is shown from two views. Which labelled shape CANNOT be obtained by slicing the solid? One letter.",
                "The top row displays a {prefix}solid that {desc_text}. Which candidate shape below CANNOT possibly be a cross-section? Answer with a letter.",
                "Two views of the {prefix}shape ({desc_text}) are provided. Which of the four options CANNOT be a cross-section of this solid? Respond with the letter.",
                "Study the {prefix}solid rendered in two views (it {desc_text}). Which of options A-D CANNOT be a cross-section? Single letter only.",
                "The {prefix}shape that {desc_text} is shown twice from different angles. Which of the four labelled shapes CANNOT be a cross-section? A, B, C, or D?",
                "Identify which of the candidate shapes A-D CANNOT be a cross-section of the {prefix}solid shown above (which {desc_text}). Answer with one letter.",
            ]
            q = cannot_templates[sidx].format(prefix=prefix, desc_text=desc_text)
        return q, correct_letter, img

    # ------------------------------------------------------------------ #
    def _maybe_swap_for_near_miss(self, options, possible, correct_shape,
                                  q_form, rng):
        """Try to ensure at least one near-miss pair (e.g., circle/ellipse
        or square/rectangle) appears among the options to reduce easy wins."""
        near_pairs = [("circle", "ellipse"),
                      ("square", "rectangle"),
                      ("triangle", "trapezoid")]
        for a, b in near_pairs:
            if a in options and b in options:
                return options
        # Try inserting one near-miss
        for a, b in rng.sample(near_pairs, len(near_pairs)):
            # If correct=a (impossible for this solid), make sure b is also
            # in distractors (b is possible).
            if a == correct_shape:
                continue
            if b == correct_shape:
                continue
            if q_form == "cannot":
                # Distractors must be POSSIBLE; if a in possible and b in possible,
                # try to include both.
                if a in possible and b in possible:
                    if a not in options and b in options:
                        # swap a non-correct distractor for a
                        for i, o in enumerate(options):
                            if o != correct_shape and o != b:
                                new_options = list(options)
                                new_options[i] = a
                                if len(set(new_options)) == 4:
                                    return new_options
                    if b not in options and a in options:
                        for i, o in enumerate(options):
                            if o != correct_shape and o != a:
                                new_options = list(options)
                                new_options[i] = b
                                if len(set(new_options)) == 4:
                                    return new_options
        return options

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, solid_id, solid_desc, options, rng, combined):
        style = self._random_style()
        sc = style["figsize_scale"]
        # Randomize viewing angles and solid colour per seed for diversity.
        elev1 = rng.uniform(15, 32)
        elev2 = rng.uniform(15, 32)
        azim1 = rng.uniform(10, 60)
        azim2 = rng.uniform(105, 160)
        self._render_color = rng.choice(
            ["#85c1e9", "#7fb3d5", "#a9cce3", "#aed6f1",
             "#f5b7b1", "#f1948a", "#f8c471", "#82e0aa",
             "#76d7c4", "#c39bd3", "#d7bde2", "#a3e4d7"])
        self._render_edge = rng.choice(
            ["#1f618d", "#21618c", "#1b4f72", "#154360",
             "#7b241c", "#943126", "#b7950b", "#196f3d",
             "#117a65", "#6c3483", "#7d3c98", "#0e6251"])
        self._render_alpha = rng.uniform(0.72, 0.92)
        suptitle_pool = [
            "Cross-Section Identification",
            "Identify the Cross-Section",
            "Cross-Section Problem",
            "Which is a Cross-Section?",
            "Cross-Section of the Solid",
            "Shape Cross-Section Task",
            "3D Solid Cross-Section",
            "Cross-Section Matching",
            "Solid and Cross-Section",
            "Cross-Section Question",
            "Possible Cross-Sections",
            "Slice the Solid",
            "Planar Cross-Section",
            "Cross-Section Identification Problem",
            "Find the Correct Cross-Section",
            "Cross-Section Exercise",
        ]
        sidx_title = (self.seed or 0) % len(suptitle_pool)
        suptitle = suptitle_pool[sidx_title]
        view_label_pool = [
            ("View 1", "View 2"), ("View A", "View B"),
            ("Angle 1", "Angle 2"), ("Perspective 1", "Perspective 2"),
            ("Side A", "Side B"), ("Front View", "Side View"),
            ("View I", "View II"), ("(1)", "(2)"),
        ]
        view_a, view_b = rng.choice(view_label_pool)
        fig_w = rng.uniform(10, 12)
        fig_h = rng.uniform(7.8, 9.2)
        fig = plt.figure(figsize=(fig_w * sc, fig_h * sc))
        fig.patch.set_facecolor(style["bg_color"])

        gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0],
                              hspace=rng.uniform(0.3, 0.42),
                              wspace=rng.uniform(0.22, 0.32))
        # Top row: 3D shape from two angles.
        ax_a = fig.add_subplot(gs[0, 0:2], projection="3d")
        ax_b = fig.add_subplot(gs[0, 2:4], projection="3d")
        for ax in (ax_a, ax_b):
            ax.set_facecolor(style["bg_color"])

        if combined:
            top_part, bottom_part = solid_id
            self._draw_combined(ax_a, top_part, bottom_part,
                                elev=elev1, azim=azim1)
            self._draw_combined(ax_b, top_part, bottom_part,
                                elev=elev2, azim=azim2)
        else:
            self._draw_solid(ax_a, solid_id, elev=elev1, azim=azim1)
            self._draw_solid(ax_b, solid_id, elev=elev2, azim=azim2)

        ax_a.set_title(view_a, fontsize=12, fontweight="bold")
        ax_b.set_title(view_b, fontsize=12, fontweight="bold")

        # Bottom row: 4 candidate 2D shapes.
        for i, shape in enumerate(options):
            ax_opt = fig.add_subplot(gs[1, i])
            ax_opt.set_facecolor(style["bg_color"])
            self._draw_2d_shape(ax_opt, shape)
            ax_opt.set_title(f"({chr(65 + i)})", fontsize=14,
                             fontweight="bold")
            ax_opt.set_xticks([])
            ax_opt.set_yticks([])
            ax_opt.set_aspect("equal")
            for spine in ax_opt.spines.values():
                spine.set_visible(False)

        fig.suptitle(suptitle, fontsize=15,
                     fontweight="bold", y=0.98)
        fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.04)
        return self.fig_to_pil(fig, dpi=max(style["dpi"], 110))

    # ------------------------------------------------------------------ #
    # 3D solid drawing
    # ------------------------------------------------------------------ #
    def _draw_solid(self, ax, solid, elev, azim, z_offset=0.0,
                    height_scale=1.0, base_radius=1.0):
        """Render a single primitive solid centered at origin (with given
        z_offset for stacking)."""
        color = getattr(self, "_render_color", "#85c1e9")
        edge_color = getattr(self, "_render_edge", "#1f618d")

        if solid == "sphere":
            self._draw_sphere(ax, R=base_radius, color=color, z=z_offset)
        elif solid == "cylinder":
            self._draw_cylinder(ax, R=base_radius, h=2 * height_scale,
                                color=color, z=z_offset)
        elif solid == "cone":
            self._draw_cone(ax, R=base_radius, h=2 * height_scale,
                            color=color, z=z_offset)
        elif solid == "cube":
            self._draw_cuboid(ax, dx=base_radius, dy=base_radius,
                              dz=base_radius * height_scale,
                              color=color, z=z_offset)
        elif solid == "square_pyramid":
            self._draw_square_pyramid(ax, side=base_radius * 2,
                                      h=2 * height_scale,
                                      color=color, z=z_offset)
        elif solid == "triangular_prism":
            self._draw_triangular_prism(ax, side=base_radius * 1.6,
                                        h=2 * height_scale,
                                        color=color, z=z_offset)
        elif solid == "rectangular_prism":
            self._draw_cuboid(ax, dx=base_radius * 1.5,
                              dy=base_radius * 0.9,
                              dz=base_radius * 1.4 * height_scale,
                              color=color, z=z_offset)
        elif solid == "square_frustum":
            self._draw_square_frustum(ax, top_side=base_radius,
                                      bot_side=base_radius * 1.6,
                                      h=2 * height_scale,
                                      color=color, z=z_offset)
        elif solid == "cone_frustum":
            self._draw_cone_frustum(ax, top_R=base_radius * 0.5,
                                    bot_R=base_radius,
                                    h=2 * height_scale,
                                    color=color, z=z_offset)
        # View setup
        m = base_radius * 2.5
        ax.set_xlim([-m, m]); ax.set_ylim([-m, m])
        ax.set_zlim([-m * 0.5, m * 1.5])
        ax.view_init(elev=elev, azim=azim)
        self._strip_3d_chrome(ax)

    def _draw_combined(self, ax, top_part, bottom_part, elev, azim):
        """Stack two solids: bottom_part at z=0..2, top_part at z=2..4."""
        # Bottom solid
        self._draw_solid(ax, bottom_part, elev=elev, azim=azim,
                         z_offset=0.0, base_radius=1.2, height_scale=1.0)
        # Top solid -- slightly larger than before so it's clearly visible.
        # Previously base_radius=1.0 height_scale=0.8 made a pyramid look
        # like a small bump on top of a cube.
        self._draw_solid(ax, top_part, elev=elev, azim=azim,
                         z_offset=2.4, base_radius=1.1, height_scale=1.1)
        # Set view limits relative to combined shape.
        m = 2.8
        ax.set_xlim([-m, m]); ax.set_ylim([-m, m])
        ax.set_zlim([-0.5, 5.5])
        ax.view_init(elev=elev, azim=azim)
        self._strip_3d_chrome(ax)

    @staticmethod
    def _strip_3d_chrome(ax):
        """Hide axis ticks, gridlines, and pane edges for clean rendering."""
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        # Make the panes transparent so the bounding box doesn't show.
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("white")
            ax.yaxis.pane.set_edgecolor("white")
            ax.zaxis.pane.set_edgecolor("white")
        except Exception:
            pass
        # Hide the axis lines (3D spines).
        try:
            ax.xaxis.line.set_color((1, 1, 1, 0))
            ax.yaxis.line.set_color((1, 1, 1, 0))
            ax.zaxis.line.set_color((1, 1, 1, 0))
        except Exception:
            pass
        # Hide the axis labels.
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    # ----- Primitive drawers ----- #
    def _draw_sphere(self, ax, R, color, z=0.0):
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 18)
        xs = R * np.outer(np.cos(u), np.sin(v))
        ys = R * np.outer(np.sin(u), np.sin(v))
        zs = R * np.outer(np.ones_like(u), np.cos(v)) + z + R
        ax.plot_surface(xs, ys, zs, color=color, alpha=0.85,
                        edgecolor=getattr(self, "_render_edge", "#1f618d"), linewidth=0.3)

    def _draw_cylinder(self, ax, R, h, color, z=0.0):
        u = np.linspace(0, 2 * np.pi, 24)
        zs = np.linspace(z, z + h, 12)
        U, Z = np.meshgrid(u, zs)
        X = R * np.cos(U)
        Y = R * np.sin(U)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.85,
                        edgecolor=getattr(self, "_render_edge", "#1f618d"), linewidth=0.3)
        # Caps
        theta = np.linspace(0, 2 * np.pi, 24)
        cx = R * np.cos(theta)
        cy = R * np.sin(theta)
        for cz in [z, z + h]:
            verts = [list(zip(cx, cy, [cz] * len(cx)))]
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection(verts, alpha=0.85,
                                    facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
            ax.add_collection3d(poly)

    def _draw_cone(self, ax, R, h, color, z=0.0):
        u = np.linspace(0, 2 * np.pi, 24)
        zs = np.linspace(z, z + h, 12)
        U, Z = np.meshgrid(u, zs)
        rr = R * (1 - (Z - z) / h)
        X = rr * np.cos(U)
        Y = rr * np.sin(U)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.85,
                        edgecolor=getattr(self, "_render_edge", "#1f618d"), linewidth=0.3)
        # Base cap
        theta = np.linspace(0, 2 * np.pi, 24)
        cx = R * np.cos(theta)
        cy = R * np.sin(theta)
        verts = [list(zip(cx, cy, [z] * len(cx)))]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        poly = Poly3DCollection(verts, alpha=0.85,
                                facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
        ax.add_collection3d(poly)

    def _draw_cuboid(self, ax, dx, dy, dz, color, z=0.0):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        x0, y0, z0 = -dx, -dy, z
        x1, y1, z1 = dx, dy, z + dz * 2
        verts = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        poly = Poly3DCollection(verts, alpha=0.8,
                                facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
        ax.add_collection3d(poly)

    def _draw_square_pyramid(self, ax, side, h, color, z=0.0):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        s = side / 2
        base = [(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)]
        apex = (0, 0, z + h)
        verts = [base,
                 [base[0], base[1], apex],
                 [base[1], base[2], apex],
                 [base[2], base[3], apex],
                 [base[3], base[0], apex]]
        poly = Poly3DCollection(verts, alpha=0.8,
                                facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
        ax.add_collection3d(poly)

    def _draw_triangular_prism(self, ax, side, h, color, z=0.0):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        s = side / 2
        bottom = [(-s, -s * 0.866, z), (s, -s * 0.866, z), (0, s * 0.866, z)]
        top = [(x, y, z + h) for (x, y, _) in bottom]
        verts = [bottom, top,
                 [bottom[0], bottom[1], top[1], top[0]],
                 [bottom[1], bottom[2], top[2], top[1]],
                 [bottom[2], bottom[0], top[0], top[2]]]
        poly = Poly3DCollection(verts, alpha=0.8,
                                facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
        ax.add_collection3d(poly)

    def _draw_square_frustum(self, ax, top_side, bot_side, h, color, z=0.0):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ts = top_side / 2
        bs = bot_side / 2
        bottom = [(-bs, -bs, z), (bs, -bs, z), (bs, bs, z), (-bs, bs, z)]
        top = [(-ts, -ts, z + h), (ts, -ts, z + h),
               (ts, ts, z + h), (-ts, ts, z + h)]
        verts = [bottom, top,
                 [bottom[0], bottom[1], top[1], top[0]],
                 [bottom[1], bottom[2], top[2], top[1]],
                 [bottom[2], bottom[3], top[3], top[2]],
                 [bottom[3], bottom[0], top[0], top[3]]]
        poly = Poly3DCollection(verts, alpha=0.8,
                                facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
        ax.add_collection3d(poly)

    def _draw_cone_frustum(self, ax, top_R, bot_R, h, color, z=0.0):
        u = np.linspace(0, 2 * np.pi, 24)
        zs = np.linspace(z, z + h, 12)
        U, Z = np.meshgrid(u, zs)
        rr = bot_R + (top_R - bot_R) * (Z - z) / h
        X = rr * np.cos(U)
        Y = rr * np.sin(U)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.85,
                        edgecolor=getattr(self, "_render_edge", "#1f618d"), linewidth=0.3)
        # Caps
        theta = np.linspace(0, 2 * np.pi, 24)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for r, cz in [(bot_R, z), (top_R, z + h)]:
            cx = r * np.cos(theta)
            cy = r * np.sin(theta)
            verts = [list(zip(cx, cy, [cz] * len(cx)))]
            poly = Poly3DCollection(verts, alpha=0.85,
                                    facecolor=color, edgecolor=getattr(self, "_render_edge", "#1f618d"))
            ax.add_collection3d(poly)

    # ------------------------------------------------------------------ #
    # 2D shape drawers
    # ------------------------------------------------------------------ #
    def _draw_2d_shape(self, ax, shape):
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        face = "#fdebd0"
        edge = "#a04000"
        lw = 2.0
        if shape == "circle":
            ax.add_patch(Circle((0, 0), 1.0, fc=face, ec=edge, lw=lw))
        elif shape == "ellipse":
            ax.add_patch(Ellipse((0, 0), 2.4, 1.4, fc=face, ec=edge, lw=lw))
        elif shape == "square":
            ax.add_patch(Rectangle((-0.9, -0.9), 1.8, 1.8,
                                   fc=face, ec=edge, lw=lw))
        elif shape == "rectangle":
            ax.add_patch(Rectangle((-1.2, -0.55), 2.4, 1.1,
                                   fc=face, ec=edge, lw=lw))
        elif shape == "triangle":
            ax.add_patch(Polygon([(-1.0, -0.7), (1.0, -0.7), (0.0, 1.0)],
                                 closed=True, fc=face, ec=edge, lw=lw))
        elif shape == "trapezoid":
            ax.add_patch(Polygon([(-1.1, -0.6), (1.1, -0.6),
                                  (0.6, 0.7), (-0.6, 0.7)],
                                 closed=True, fc=face, ec=edge, lw=lw))
        elif shape == "pentagon":
            pts = []
            for k in range(5):
                a = math.pi / 2 + k * 2 * math.pi / 5
                pts.append((math.cos(a), math.sin(a)))
            ax.add_patch(Polygon(pts, closed=True, fc=face, ec=edge, lw=lw))
        elif shape == "hexagon":
            pts = []
            for k in range(6):
                a = math.pi / 6 + k * 2 * math.pi / 6
                pts.append((math.cos(a), math.sin(a)))
            ax.add_patch(Polygon(pts, closed=True, fc=face, ec=edge, lw=lw))

if __name__ == "__main__":
    env = SphereCrossSectionAreaQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer}")
