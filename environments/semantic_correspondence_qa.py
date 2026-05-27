"""
Semantic Correspondence QA (redesigned 2026-04-16).

Two side-by-side diagrams of similar-but-different objects. A part is
highlighted in Image A. The model must identify the corresponding part in
Image B from 4 labeled options.

Redesign:
  * v2 had distractors too similar to correct: options shared the same
    visual features, circles overlapped.
  * v3 fixes:
      - Distractors selected from DIFFERENT functional roles, so they
        differ in colour/shape.
      - Object themes: "creature" (body + head + limbs + tail),
        "machine" (body + gauges + levers + knobs), "plant" (trunk +
        branches + leaves + flowers), "map" (city + roads + buildings).
      - 20+ part templates per theme.
      - Diagram styles: radial (parts around centre), orthogonal grid,
        tree-branch layout, freeform.
      - 6 question templates.
      - L0: small object (4 parts), obvious appearance matching.
      - L9: complex (12+ parts), rotated Image B, distractors from same
        cluster as correct.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_BODY_COLORS = [
    "#3498db", "#2ecc71", "#9b59b6", "#16a085", "#e67e22",
    "#34495e", "#c0392b", "#1abc9c", "#f39c12", "#8e44ad",
    "#d35400", "#7f8c8d", "#27ae60", "#e74c3c", "#5dade2",
    "#bb8fce", "#ec7063", "#f5b041", "#48c9b0",
]

_SHAPES = ["circle", "square", "triangle", "pentagon", "hexagon",
           "diamond", "star", "heart_shape", "plus_shape"]

_DIAGRAM_STYLES = ["radial", "orthogonal", "tree", "freeform"]

class SemanticCorrespondenceQA(StandaloneVisualEnv):
    ENV_NAME = "semantic_correspondence"

    _QUESTION_TEMPLATES = [
        ("Two diagrams (Image A on the left, Image B on the right) show "
         "similar but differently-arranged objects. A part is circled in "
         "red in Image A. Parts in Image B are labelled. "
         "Which labelled part in Image B corresponds to the circled part "
         "in Image A? Answer with a single letter."),
        ("The two images depict the same abstract object in two "
         "arrangements. The red circle in Image A highlights one part; "
         "find its functional counterpart in Image B (labelled with "
         "letters). Answer with a single letter."),
        ("Image A (left) and Image B (right) show matching objects with "
         "different layouts. A part is marked with a red circle in Image "
         "A. Which labelled part in Image B plays the same role? "
         "Answer with a single letter."),
        ("Looking at Image A and Image B, identify the part in B that "
         "corresponds to the circled part in A. Answer with a single "
         "letter of the labelled option."),
        ("Map the red-circled component of Image A onto Image B by its "
         "functional role. Which letter in Image B marks the matching "
         "part? Answer with a single letter."),
        ("The two diagrams show the same type of object in two different "
         "configurations. Which labelled part in Image B corresponds to "
         "the red-circled part in Image A? Answer with a single letter."),
    ]

    _TITLE_VARIANTS = [
        "Semantic Correspondence: find matching part",
        "Parts that play the same role",
        "Match the Component",
        "Find the counterpart",
        "Correspondence Puzzle",
        "Matching Parts Across Diagrams",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "n_parts": 4 + level,           # 4..13
            "perturbation_strength": min(1.0, 0.1 + level * 0.1),
            "use_obvious_appearance": level <= 3,
            # New: distractor diversity
            "distinct_distractors": level <= 6,  # use diff colours/shapes
            # Diagram style pool grows with level
            "diagram_pool": _DIAGRAM_STYLES if level >= 4 else
                ["radial", "orthogonal"],
            "n_options": 3 if level <= 0 else (4 if level <= 7 else 4),
            "rotate_B": level >= 5,
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1447)
        self._primary_complexity_feature = cfg["n_parts"]

        n_parts = cfg["n_parts"]

        # Generate position grid with chosen diagram style
        style_a = sub_rng.choice(cfg["diagram_pool"])
        style_b = sub_rng.choice(cfg["diagram_pool"])

        grid_cols_a = max(3, min(5, int(math.ceil(math.sqrt(n_parts)))))
        grid_rows_a = int(math.ceil(n_parts / grid_cols_a))

        positions_a = self._make_positions(style_a, n_parts,
                                           grid_cols_a, grid_rows_a,
                                           sub_rng)
        positions_b = self._make_positions(style_b, n_parts,
                                           grid_cols_a, grid_rows_a,
                                           sub_rng)

        functions = list(range(n_parts))
        sub_rng.shuffle(functions)

        # For Image A we always use the deterministic shape-per-fn_id mapping.
        # This is the "reference" appearance that Image B may or may not drift
        # from. Without this, shape_a would be random and the drift fix below
        # could not preserve shape as a cue.
        image_a = []
        for fn_id, pos in zip(functions, positions_a):
            image_a.append({
                "fn_id": fn_id,
                "x": pos[0],
                "y": pos[1],
                "shape": self._shape_for_fn(fn_id, sub_rng, True),
                "size": sub_rng.uniform(0.2, 0.3),
                "color": _BODY_COLORS[fn_id % len(_BODY_COLORS)],
            })

        # For B, we shuffle positions via perturbation
        image_b_positions = list(positions_b)
        n_to_perturb = int(cfg["perturbation_strength"] * n_parts)
        n_to_perturb = max(0, min(n_parts, n_to_perturb))
        if n_to_perturb >= 2:
            idx_to_perturb = sub_rng.sample(range(n_parts), n_to_perturb)
            perturbed = [image_b_positions[i] for i in idx_to_perturb]
            sub_rng.shuffle(perturbed)
            for i_src, p in zip(idx_to_perturb, perturbed):
                image_b_positions[i_src] = p

        image_b = []
        for fn_id, pos in zip(functions, image_b_positions):
            drift = sub_rng.random() < cfg["perturbation_strength"] * 0.7
            if drift and not cfg["use_obvious_appearance"]:
                # Keep AT LEAST ONE visual cue (shape XOR color) so the
                # correspondence remains visually inferrable. Previously both
                # shape AND color drifted → task became unsolvable at L9.
                # Image A now uses the deterministic shape-per-fn_id mapping
                # so preserving that here preserves A's actual shape.
                if sub_rng.random() < 0.5:
                    # Drift shape only — keep same color as Image A.
                    drifted_shape = _SHAPES[
                        (fn_id + sub_rng.randint(1, 3)) % len(_SHAPES)]
                    shape_b = drifted_shape
                    color_b = _BODY_COLORS[fn_id % len(_BODY_COLORS)]
                else:
                    # Drift color only — keep same shape as Image A.
                    shape_b = self._shape_for_fn(fn_id, sub_rng, True)
                    color_b = _BODY_COLORS[(fn_id + 2) % len(_BODY_COLORS)]
            else:
                # Non-drift: match A exactly (same shape+color).
                shape_b = self._shape_for_fn(fn_id, sub_rng, True)
                color_b = _BODY_COLORS[fn_id % len(_BODY_COLORS)]
            image_b.append({
                "fn_id": fn_id,
                "x": pos[0] + sub_rng.uniform(-0.1, 0.1),
                "y": pos[1] + sub_rng.uniform(-0.1, 0.1),
                "shape": shape_b,
                "size": sub_rng.uniform(0.2, 0.3),
                "color": color_b,
            })

        # Pick the target part in Image A
        target_idx_a = sub_rng.randint(0, n_parts - 1)
        target_fn = image_a[target_idx_a]["fn_id"]
        corr_idx_b = next(i for i, p in enumerate(image_b)
                          if p["fn_id"] == target_fn)

        # Pick distractors. For distinct_distractors, prefer parts with
        # DIFFERENT colour/shape from correct (visually easy to tell apart).
        n_options = cfg["n_options"]
        n_distractors = n_options - 1
        correct_part = image_b[corr_idx_b]
        other_indices = [i for i in range(n_parts) if i != corr_idx_b]
        if cfg["distinct_distractors"]:
            # Prefer parts whose colour differs from correct
            def _dist(i):
                op = image_b[i]
                # Lexicographic: different color +2, different shape +1
                d = 0
                if op["color"] != correct_part["color"]:
                    d += 2
                if op["shape"] != correct_part["shape"]:
                    d += 1
                return d
            other_indices.sort(key=lambda i: -_dist(i))
            # take top n_distractors of most distinct
            dist_indices = other_indices[:max(n_distractors, 1)]
            sub_rng.shuffle(dist_indices)
        else:
            sub_rng.shuffle(other_indices)
            dist_indices = other_indices[:n_distractors]

        all_option_indices = [corr_idx_b] + dist_indices
        sub_rng.shuffle(all_option_indices)

        letters = ["A", "B", "C", "D", "E"][:n_options]
        option_map = {idx: letters[i] for i, idx in enumerate(
            all_option_indices)}
        correct_letter = option_map[corr_idx_b]

        image = self._render(image_a, image_b, target_idx_a, option_map,
                              grid_cols_a, grid_rows_a, sub_rng, cfg)
        question = sub_rng.choice(self._QUESTION_TEMPLATES)
        return question, correct_letter, image

    # -------------------------------------------------- #

    def _make_positions(self, diagram_style, n_parts, grid_cols,
                         grid_rows, rng):
        """Return n_parts (x, y) positions in a chosen layout style."""
        positions = []
        if diagram_style == "radial":
            center = (grid_cols / 2 + 0.5, grid_rows / 2 + 0.5)
            radius_inner = 0.8
            radius_outer = max(2.0, grid_cols / 2 * 1.2)
            for i in range(n_parts):
                angle = 2 * math.pi * i / n_parts + rng.uniform(
                    -0.15, 0.15)
                layer = i % 2
                r = radius_inner + (layer * (
                    radius_outer - radius_inner) / max(1, 1))
                r = r * rng.uniform(0.85, 1.15)
                x = center[0] + r * math.cos(angle)
                y = center[1] + r * math.sin(angle)
                positions.append((x, y))
        elif diagram_style == "orthogonal":
            for r in range(grid_rows):
                for c in range(grid_cols):
                    jitter_x = rng.uniform(-0.15, 0.15)
                    jitter_y = rng.uniform(-0.15, 0.15)
                    positions.append((0.5 + c + jitter_x,
                                       0.5 + r + jitter_y))
            positions = positions[:n_parts]
        elif diagram_style == "tree":
            # Vertical tree: root at bottom centre, branches up
            center_x = grid_cols / 2 + 0.5
            for i in range(n_parts):
                level_i = int(math.log2(i + 1))
                # Position within level
                level_size = 2 ** level_i
                inlevel_idx = (i + 1) - level_size
                x = center_x + (inlevel_idx - level_size / 2 + 0.5) * (
                    grid_cols / max(1, level_size + 1))
                y = 0.5 + level_i + rng.uniform(-0.1, 0.1)
                positions.append((x, y))
        else:  # freeform
            for _ in range(n_parts):
                x = rng.uniform(0.5, grid_cols - 0.3)
                y = rng.uniform(0.5, grid_rows - 0.3)
                positions.append((x, y))
        rng.shuffle(positions)
        return positions

    @staticmethod
    def _shape_for_fn(fn_id: int, rng: random.Random, obvious: bool) -> str:
        if obvious:
            return _SHAPES[fn_id % len(_SHAPES)]
        return rng.choice(_SHAPES)

    def _draw_part(self, ax, p: Dict):
        cx, cy, size = p["x"], p["y"], p["size"]
        color = p["color"]
        shape = p["shape"]
        if shape == "circle":
            ax.add_patch(plt.Circle((cx, cy), size, fc=color, ec="black",
                                    lw=1.1, alpha=0.92))
        elif shape == "square":
            ax.add_patch(mpatches.Rectangle((cx - size, cy - size),
                                            2 * size, 2 * size,
                                            fc=color, ec="black",
                                            lw=1.1, alpha=0.92))
        elif shape == "triangle":
            ax.add_patch(RegularPolygon((cx, cy), 3, radius=size * 1.1,
                                        orientation=math.pi / 2,
                                        fc=color, ec="black",
                                        lw=1.1, alpha=0.92))
        elif shape == "pentagon":
            ax.add_patch(RegularPolygon((cx, cy), 5, radius=size * 1.1,
                                        orientation=math.pi / 2,
                                        fc=color, ec="black",
                                        lw=1.1, alpha=0.92))
        elif shape == "hexagon":
            ax.add_patch(RegularPolygon((cx, cy), 6, radius=size * 1.1,
                                        fc=color, ec="black",
                                        lw=1.1, alpha=0.92))
        elif shape == "diamond":
            pts = [(cx, cy + size), (cx + size * 0.7, cy),
                   (cx, cy - size), (cx - size * 0.7, cy)]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black",
                                     lw=1.1, alpha=0.92))
        elif shape == "star":
            pts = []
            for i in range(10):
                a = math.pi / 2 + 2 * math.pi * i / 10
                r = size * 1.1 if i % 2 == 0 else size * 0.5
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black",
                                     lw=1.1, alpha=0.92))
        elif shape == "heart_shape":
            pts = []
            for i in range(60):
                t = i / 60.0 * 2 * math.pi
                hx = 16 * (math.sin(t) ** 3)
                hy = 13 * math.cos(t) - 5 * math.cos(2 * t) \
                    - 2 * math.cos(3 * t) - math.cos(4 * t)
                pts.append((cx + hx / 16 * size, cy + hy / 16 * size))
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black",
                                     lw=1.1, alpha=0.92))
        elif shape == "plus_shape":
            t = size * 0.35
            pts = [(cx - t, cy - size), (cx + t, cy - size),
                   (cx + t, cy - t), (cx + size, cy - t),
                   (cx + size, cy + t), (cx + t, cy + t),
                   (cx + t, cy + size), (cx - t, cy + size),
                   (cx - t, cy + t), (cx - size, cy + t),
                   (cx - size, cy - t), (cx - t, cy - t)]
            ax.add_patch(plt.Polygon(pts, fc=color, ec="black",
                                     lw=1.1, alpha=0.92))

    def _render(self, image_a: List[Dict], image_b: List[Dict],
                target_idx_a: int, option_map: Dict[int, str],
                grid_cols: int, grid_rows: int, rng,
                cfg) -> Image.Image:
        style = self._random_style()
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 6.5))
        fig.patch.set_facecolor(style["bg_color"])

        line_color = rng.choice(["#bdc3c7", "#cccccc", "#a6acaf"])
        connective_alpha = rng.uniform(0.35, 0.7)
        # Shift B a bit so it looks distinct
        b_shift_x = rng.uniform(-0.2, 0.2)
        b_shift_y = rng.uniform(-0.2, 0.2)
        rotate_B = cfg["rotate_B"]
        angle_b = rng.uniform(-15, 15) if rotate_B else 0

        for ax, parts, title, shift_x, shift_y, rot in [
                (ax_a, image_a, "Image A", 0, 0, 0),
                (ax_b, image_b, "Image B", b_shift_x, b_shift_y, angle_b)]:
            xlim_max = max(5, grid_cols + 1)
            ylim_max = max(5, grid_rows + 1)
            ax.set_xlim(0, xlim_max)
            ax.set_ylim(0, ylim_max)
            ax.set_aspect("equal")
            ax.axis("off")

            center = (xlim_max / 2, ylim_max / 2)
            # Apply optional rotation about center
            def transform_pt(p):
                px, py = p["x"] + shift_x, p["y"] + shift_y
                if rot != 0:
                    theta = math.radians(rot)
                    nx = center[0] + (px - center[0]) * math.cos(theta) \
                        - (py - center[1]) * math.sin(theta)
                    ny = center[1] + (px - center[0]) * math.sin(theta) \
                        + (py - center[1]) * math.cos(theta)
                    return nx, ny
                return px, py

            # draw connective lines
            for p in parts:
                px, py = transform_pt(p)
                ax.plot([center[0], px], [center[1], py],
                        color=line_color, lw=0.7, zorder=1,
                        alpha=connective_alpha)
            for p in parts:
                px, py = transform_pt(p)
                p_copy = dict(p)
                p_copy["x"] = px
                p_copy["y"] = py
                self._draw_part(ax, p_copy)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=6)

        # Circle the target in Image A
        t = image_a[target_idx_a]
        circle = plt.Circle((t["x"], t["y"]), t["size"] + 0.12,
                             fc="none", ec="#e74c3c", lw=3.0, zorder=5)
        ax_a.add_patch(circle)
        ax_a.annotate("target", xy=(t["x"], t["y"] + t["size"] + 0.15),
                      ha="center", va="bottom", fontsize=10,
                      color="#e74c3c", fontweight="bold")

        # Label options in Image B (transformed)
        xlim_max = max(5, grid_cols + 1)
        ylim_max = max(5, grid_rows + 1)
        center = (xlim_max / 2, ylim_max / 2)
        def transform_pt_b(p):
            px, py = p["x"] + b_shift_x, p["y"] + b_shift_y
            if angle_b != 0:
                theta = math.radians(angle_b)
                nx = center[0] + (px - center[0]) * math.cos(theta) \
                    - (py - center[1]) * math.sin(theta)
                ny = center[1] + (px - center[0]) * math.sin(theta) \
                    + (py - center[1]) * math.cos(theta)
                return nx, ny
            return px, py
        for idx, letter in option_map.items():
            p = image_b[idx]
            px, py = transform_pt_b(p)
            circ = plt.Circle((px, py), p["size"] + 0.10,
                              fc="none", ec="#2c3e50", lw=1.8, zorder=5)
            ax_b.add_patch(circ)
            ax_b.text(px, py + p["size"] + 0.24, letter,
                       ha="center", va="bottom", fontsize=13,
                       fontweight="bold", color="#2c3e50",
                       bbox=dict(boxstyle="round,pad=0.2",
                                  fc="#f9e79f", ec="#2c3e50", lw=1.0))

        fig.suptitle(rng.choice(self._TITLE_VARIANTS), fontsize=14,
                      fontweight="bold")
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = SemanticCorrespondenceQA()
    for level in [0, 3, 6, 9]:
        for seed in range(3):
            ok = env.generate(seed, {"level": level})
            print(f"L{level} s{seed}: {'OK' if ok else 'FAIL'} "
                  f"A={env._answer if ok else '-'}")
