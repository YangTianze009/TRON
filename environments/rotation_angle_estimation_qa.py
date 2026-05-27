"""
Rotation Angle Estimation QA.

Shows a 2D shape in original (dashed) and rotated (solid) positions.
Asks to identify the rotation angle. MCQ format.

Difficulty axes:
  A) angle_granularity -> 30 -> 15 degree multiples
  B) shape_complexity: 3..12 vertices
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Arc
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

def _rotate_pts(pts, angle_deg, cx=0, cy=0):
    rad = math.radians(angle_deg)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        out.append((cx + dx*math.cos(rad) - dy*math.sin(rad),
                     cy + dx*math.sin(rad) + dy*math.cos(rad)))
    return out

def _make_shape(rng, n_verts, easy=False):
    """Generate an asymmetric polygon with n_verts vertices.

    easy=True (L0/L1): use a fixed L-tromino-like wedge — visually distinct
    under rotation, no irregular randomness that confuses 4B base model.
    """
    if easy:
        # Right-pointing arrow / wedge — 3 vertices, very asymmetric.
        # Original orientation: tip at (2, 0), base at (-1, ±1).
        return [(2.0, 0.0), (-1.0, 1.0), (-1.0, -1.0)]
    angles = sorted([rng.uniform(0, 2*math.pi) for _ in range(n_verts)])
    pts = []
    for a in angles:
        r = rng.uniform(1.0, 2.5)
        pts.append((r*math.cos(a), r*math.sin(a)))
    # Make asymmetric by shifting one point
    if len(pts) > 2:
        pts[0] = (pts[0][0] + 0.8, pts[0][1] + 0.3)
    return pts

class RotationAngleEstimationQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "rotation_angle_estimation"

    def _level_config(self, level):
        if level <= 1:
            # Only 90 (CCW quarter) and 180 (half) at L0/L1 — avoids the
            # 270°-CCW vs 90°-CW ambiguity that the model trips over.
            granularity = [90, 180]
        elif level <= 2:
            granularity = [90, 180, 270]
        elif level <= 4:
            granularity = [45, 90, 135, 180, 225, 270, 315]
        elif level <= 6:
            granularity = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        else:
            granularity = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                           195, 210, 225, 240, 255, 270, 285, 300, 315, 345]
        # Cap vertex count so the shape stays recognizable at L9 (was 3+level
        # = 12 at L9, which made rotation angle visually un-estimable).
        n_verts = 3 + level // 2  # L0=3 ... L9=7
        return {
            'n_verts': min(n_verts, 7),
            'angles': granularity,
            'center_offset': level >= 4,
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        # 2026-05-04: added easier L0 mode (was 7.5% — VLM mental-rotation limit, attempt fix)
        # L0/L1 short-circuit: pure coordinate algebra MCQ — no visual estimation.
        if level <= 1:
            return self._generate_easy_l0l1(level)
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1002)
        style = self._random_style()

        pts = _make_shape(rng, cfg['n_verts'], easy=(level <= 1))
        angle = rng.choice(cfg['angles'])

        cx, cy = 0.0, 0.0
        if cfg['center_offset']:
            cx = rng.uniform(-0.5, 0.5)
            cy = rng.uniform(-0.5, 0.5)

        rotated = _rotate_pts(pts, angle, cx, cy)

        # Build MCQ distractors. 2026-04-26: at L0/L1 use ONLY [0, 90, 180,
        # 270] so options are visually maximally distinct.
        options_set = set()
        options_set.add(angle)
        if level <= 1:
            for a in [0, 90, 180, 270]:
                if len(options_set) >= 4: break
                options_set.add(a)
        else:
            pool = [a for a in cfg['angles'] if a != angle]
            rng.shuffle(pool)
            for a in pool:
                if len(options_set) >= 4: break
                options_set.add(a)
            while len(options_set) < 4:
                options_set.add(rng.choice([30, 60, 90, 120, 150, 180, 210, 270]))
        options = sorted(options_set)
        correct_idx = options.index(angle)
        letters = "ABCD"
        correct = letters[correct_idx]

        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(6*sc, 6*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal')

        # Draw original (dashed)
        orig_patch = Polygon(pts, closed=True, fill=False,
                             edgecolor='gray', linestyle='--', linewidth=2)
        ax.add_patch(orig_patch)
        # Draw rotated (solid)
        rot_patch = Polygon(rotated, closed=True, fill=True,
                            facecolor=style['palette'][0], alpha=0.6,
                            edgecolor=style['palette'][1], linewidth=2.5)
        ax.add_patch(rot_patch)
        # Mark center
        ax.plot(cx, cy, 'ko', markersize=8, zorder=10)
        ax.annotate('Center', (cx, cy), textcoords='offset points',
                    xytext=(8, 8), fontsize=style['font_size_base'])

        # 2026-04-26: at L0/L1, draw a CCW arc with arrow indicating the
        # actual rotation direction & magnitude. Removes the CW/CCW
        # ambiguity that the model keeps tripping over.
        if level <= 1:
            # Pick a vertex of the original to anchor the arc on
            anchor_pt = pts[0]
            r = math.hypot(anchor_pt[0] - cx, anchor_pt[1] - cy) * 1.15
            start_ang_rad = math.atan2(anchor_pt[1] - cy, anchor_pt[0] - cx)
            start_ang_deg = math.degrees(start_ang_rad)
            # Counter-clockwise arc from start angle through `angle` deg.
            arc = Arc((cx, cy), 2 * r, 2 * r,
                      angle=0,
                      theta1=start_ang_deg,
                      theta2=start_ang_deg + angle,
                      color="#c0392b", linewidth=3.0, zorder=11)
            ax.add_patch(arc)
            # Arrowhead at the end of the arc (CCW means moving in increasing
            # angle, so end angle is start + angle)
            end_ang_rad = math.radians(start_ang_deg + angle)
            arrow_tip = (cx + r * math.cos(end_ang_rad),
                          cy + r * math.sin(end_ang_rad))
            tangent_ang = end_ang_rad + math.pi / 2  # CCW direction tangent
            tail_offset = 0.4
            arrow_tail = (arrow_tip[0] - tail_offset * math.cos(tangent_ang),
                          arrow_tip[1] - tail_offset * math.sin(tangent_ang))
            arrow = FancyArrowPatch(arrow_tail, arrow_tip,
                                     arrowstyle="-|>", mutation_scale=22,
                                     color="#c0392b", linewidth=3.0,
                                     zorder=12)
            ax.add_patch(arrow)
            # Direction-only label (does NOT leak the magnitude)
            mid_ang_rad = math.radians(start_ang_deg + angle / 2)
            label_xy = (cx + (r + 0.4) * math.cos(mid_ang_rad),
                        cy + (r + 0.4) * math.sin(mid_ang_rad))
            ax.text(label_xy[0], label_xy[1], "CCW",
                    color="#c0392b", fontsize=style['font_size_base'] + 2,
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="#c0392b", lw=1.5),
                    zorder=13)

        ax.autoscale_view()
        ax.margins(0.15)
        if level <= 4:
            ax.grid(True, alpha=0.2, linestyle='--')
        ax.axis('off')
        title_pool = [
            "Shape Rotation",
            "Identify the Rotation Angle",
            "Rotation Problem",
            "How much was it rotated?",
            "Polygon Rotation",
        ]
        ax.set_title(rng.choice(title_pool),
                     fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        opt_str = "  ".join(f"({letters[i]}) {options[i]} degrees" for i in range(4))
        convention = (
            "Convention: rotation is measured COUNTER-CLOCKWISE (CCW) — "
            "i.e., a 90° rotation moves a point at (1,0) to (0,1). "
            "If the rotation looks clockwise, convert: 90°CW = 270°CCW, "
            "180°CW = 180°CCW, 270°CW = 90°CCW."
        )
        templates = [
            f"A shape (dashed outline) was rotated COUNTER-CLOCKWISE around the marked center to produce the colored shape. What is the rotation angle?\n{opt_str}\n{convention}",
            f"The dashed polygon was rotated COUNTER-CLOCKWISE about the marked center to produce the filled (solid-colored) polygon. By how many degrees was it rotated?\n{opt_str}\n{convention}",
            f"Compare the dashed (original) and colored (rotated) shapes. The colored shape is the result of a COUNTER-CLOCKWISE rotation about the marked center. Identify the rotation angle.\n{opt_str}\n{convention}",
            f"The image shows a polygon and its rotated copy. The dashed outline is the original; the colored fill is the rotation. What is the COUNTER-CLOCKWISE rotation angle (in degrees) about the marked center?\n{opt_str}\n{convention}",
            f"By what COUNTER-CLOCKWISE angle (in degrees) was the dashed outline rotated about the marked center to produce the colored polygon?\n{opt_str}\n{convention}",
        ]
        q = rng.choice(templates)
        if level <= 2:
            q += (
                "\nHint: pick one corresponding vertex on both shapes. "
                "(a) If colored is the dashed flipped/rotated 180° → 180°. "
                "(b) If colored is dashed rotated 90° CCW (e.g. right→up) → 90°. "
                "(c) If colored is dashed rotated 90° CW (e.g. right→down) → 270°."
            )
        return q, correct, img

    def _generate_easy_l0l1(self, level: int):
        """L0/L1 trivial: 'Rotate point (x,y) by N degrees CCW. New coords?' MCQ."""
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2027)
        # pick simple integer point
        px, py = rng.choice([(2, 0), (0, 2), (3, 0), (0, 3), (2, 1), (1, 2),
                              (3, 1), (1, 3)])
        angle = rng.choice([90, 180, 270]) if level == 1 else rng.choice([90, 180])
        # CCW rotation about origin
        if angle == 90:
            new = (-py, px)
        elif angle == 180:
            new = (-px, -py)
        else:  # 270
            new = (py, -px)
        correct = f"({new[0]}, {new[1]})"
        # build distractor pool from other rotations of the point
        candidates = []
        for ang in (90, 180, 270):
            if ang == 90:
                candidates.append((-py, px))
            elif ang == 180:
                candidates.append((-px, -py))
            else:
                candidates.append((py, -px))
        # also add original and a sign-flip
        candidates.append((px, py))
        candidates.append((py, px))
        seen = set()
        opts = [correct]
        seen.add(correct)
        for c in candidates:
            s = f"({c[0]}, {c[1]})"
            if s not in seen:
                seen.add(s)
                opts.append(s)
            if len(opts) >= 4:
                break
        rng.shuffle(opts)
        correct_letter = "ABCD"[opts.index(correct)]
        opt_str = "  ".join(f"({l}) {o}" for l, o in zip("ABCD", opts))

        # Render a tiny image (the model needs an image)
        style = self._random_style()
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(5 * sc, 5 * sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axhline(0, color='#999', lw=0.8)
        ax.axvline(0, color='#999', lw=0.8)
        for i in range(-3, 4):
            ax.axhline(i, color='#ddd', lw=0.4)
            ax.axvline(i, color='#ddd', lw=0.4)
        ax.plot(px, py, 'o', color='#3498db', markersize=14)
        ax.annotate(f"P({px},{py})", (px, py), textcoords="offset points",
                    xytext=(8, 8), fontsize=12, fontweight='bold',
                    color='#2980b9')
        ax.set_title(f"Rotate point P by {angle}° CCW about origin",
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2)
        try:
            fig.tight_layout()
        except Exception:
            pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        question = (
            f"Apply a {angle}° COUNTER-CLOCKWISE rotation about the origin "
            f"to the point P = ({px}, {py}). "
            f"What are the new coordinates of P?\n{opt_str}\n"
            "Hint: 90°CCW: (x,y)→(-y,x); 180°: (x,y)→(-x,-y); "
            "270°CCW: (x,y)→(y,-x). Answer with a single letter (A/B/C/D)."
        )
        self._primary_complexity_feature = level
        return question, correct_letter, img


if __name__ == "__main__":
    env = RotationAngleEstimationQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
