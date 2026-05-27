"""
Multi-View Object Match QA (Task B, multi-image P5 capability).

Show 2 views of 3D objects from different angles as side-by-side images.
Ask "Are these the same object?" MCQ (A/B/C/D).
L0: wildly different objects. L9: same object type, subtle angle difference.

Difficulty axes:
  A) Object similarity (different type -> same type -> same with rotation)
  B) Angle difference (large -> small)
  C) Number of distractors / visual noise
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_SHAPE_TYPES = ["cube", "sphere", "cylinder", "cone", "pyramid", "torus"]
_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

class MultiViewObjectMatchQA(StandaloneVisualEnv):
    ENV_NAME = "multi_view_object_match"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "same_prob": 0.5,
            # At L0, different shapes are very different; at L9, very similar
            "similarity": min(1.0, 0.2 + level * 0.09),  # 0.2 -> 1.0
            "angle_diff": max(5, 90 - level * 9),          # 90 -> 9 degrees
            "n_detail_features": 1 + level // 2,            # 1 -> 5
            "visual_noise": level >= 5,
            "hide_shape_label": level >= 3,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = level

        # At high levels (L7+), switch to a genuine 3D cube-composite rendering
        # so that rotation is visually discriminable (2D polygons are trivial).
        use_3d = level >= 7

        is_same = rng.random() < cfg["same_prob"]

        if use_3d:
            # Construct a small composite of 1x1 cubes; both views show the
            # same object if is_same, else a mirrored/edited composite.
            composite_a = self._random_cube_composite(rng)
            if is_same:
                composite_b = [tuple(c) for c in composite_a]
            else:
                # Modify by mirroring or moving one cube (so it's a different
                # 3D shape), ensuring at least one cube differs.
                composite_b = self._mutate_cube_composite(composite_a, rng)
                if composite_b == composite_a:
                    composite_b = [(-x, y, z) for (x, y, z) in composite_a]
            color1 = color2 = rng.choice(_COLORS)
            # Use different camera yaw/pitch per view
            yaw1 = rng.uniform(0, 360)
            yaw2 = yaw1 + rng.uniform(-cfg["angle_diff"], cfg["angle_diff"])
            pitch1 = rng.uniform(20, 40)
            pitch2 = pitch1 + rng.uniform(-10, 10)
            img1 = self._render_cube_composite(
                composite_a, color1, yaw1, pitch1, rng, cfg)
            img2 = self._render_cube_composite(
                composite_b, color2, yaw2, pitch2, rng, cfg)
        else:
            shape1 = rng.choice(_SHAPE_TYPES)
            color1 = rng.choice(_COLORS)
            angle1 = rng.uniform(0, 360)

            if is_same:
                shape2 = shape1
                color2 = color1
                angle2 = angle1 + rng.uniform(-cfg["angle_diff"], cfg["angle_diff"])
            else:
                # Pick a visually similar distractor shape at high levels so
                # "different" is not trivially solvable by color or gross shape.
                visually_similar = {
                    "cube": ["pyramid"], "pyramid": ["cube", "cone"],
                    "sphere": ["torus"], "torus": ["sphere"],
                    "cylinder": ["cube"], "cone": ["pyramid"],
                }
                if cfg["similarity"] > 0.7:
                    # At L9, always pick a visually similar distractor +
                    # the SAME color so the model must use shape details.
                    near = visually_similar.get(shape1, [])
                    if near:
                        shape2 = rng.choice(near)
                    else:
                        shape2 = rng.choice([s for s in _SHAPE_TYPES if s != shape1])
                    color2 = color1  # keep color identical
                else:
                    shape2 = rng.choice([s for s in _SHAPE_TYPES if s != shape1])
                    color2 = rng.choice([c for c in _COLORS if c != color1])
                angle2 = rng.uniform(0, 360)

            # Render two side-by-side views
            img1 = self._render_shape(shape1, color1, angle1, rng, cfg)
            img2 = self._render_shape(shape2, color2, angle2, rng, cfg)

        # Combine side-by-side
        composite = self._combine_images(img1, img2)

        # Binary MCQ only: avoid "cannot determine" / "same type different"
        # distractors because they are never the GT and leak information.
        options = ["Yes, same object", "No, different objects"]
        rng.shuffle(options)

        target = "Yes, same object" if is_same else "No, different objects"
        answer_letter = chr(ord("A") + options.index(target))

        templates = [
            "Two views of objects are shown side by side (Image 1 on left, Image 2 on right). Are these two views of the SAME object?",
            "The image displays two object views side-by-side (Image 1 / Image 2). Decide whether they show the same object or two different objects.",
            "Compare Image 1 (left) and Image 2 (right). Is each side showing the SAME 3D object?",
            "Two object renderings are arranged side by side. Determine whether Image 1 and Image 2 depict identical objects.",
            "Inspect the two object views in the image. Are Image 1 and Image 2 representations of the SAME object?",
            "Look at the two renderings. Image 1 sits on the left; Image 2 sits on the right. Do they depict the same underlying 3D object?",
            "Given the two side-by-side views (Image 1 / Image 2), judge: are these the SAME object seen from two angles, or two different objects?",
            "Two 3D views appear next to each other. Decide if Image 1 and Image 2 correspond to one and the same object.",
            "You are presented with two object images (Image 1 at left, Image 2 at right). Are both panels views of a SINGLE object?",
            "Examine the paired views carefully. Does Image 1 show the same 3D object as Image 2, or are they different objects?",
            "The figure contains two panels: Image 1 and Image 2. Are they the same object (perhaps rotated), or genuinely different objects?",
            "Study the side-by-side rendering. Determine whether Image 1 and Image 2 are two different views of one object.",
            "Two views, placed left/right (Image 1 / Image 2). Question: same object, yes or no?",
            "Assess the two object views shown side by side. Are Image 1 and Image 2 the SAME 3D object?",
            "Two renderings are given (Image 1 = left, Image 2 = right). Answer whether both depict the identical object.",
            "Consider the pair of object images. Is the object in Image 1 the same as the object in Image 2?",
        ]
        sidx = (self.seed or 0) % len(templates)
        q_stem = templates[sidx]
        q = (q_stem + "\n"
             + "\n".join(f"  ({chr(ord('A') + i)}) {options[i]}"
                         for i in range(len(options)))
             + "\nAnswer with a single letter.")

        return q, answer_letter, composite

    # --------------------- 3D cube-composite helpers --------------------- #
    def _random_cube_composite(self, rng):
        """Build a random connected composite of 3-5 unit cubes."""
        n = rng.randint(3, 5)
        cubes = [(0, 0, 0)]
        while len(cubes) < n:
            base = rng.choice(cubes)
            dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                      (0, -1, 0), (0, 0, 1), (0, 0, -1)])
            cand = (base[0] + dx, base[1] + dy, base[2] + dz)
            if cand not in cubes:
                cubes.append(cand)
        return cubes

    def _mutate_cube_composite(self, cubes, rng):
        """Return a composite with the same cube count but a different shape."""
        for _ in range(10):
            new_cubes = [tuple(c) for c in cubes]
            # Move one cube to a new adjacent position
            idx = rng.randint(0, len(new_cubes) - 1)
            base = new_cubes[rng.randint(0, len(new_cubes) - 1)]
            for _ in range(10):
                dx, dy, dz = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                          (0, -1, 0), (0, 0, 1), (0, 0, -1)])
                cand = (base[0] + dx, base[1] + dy, base[2] + dz)
                if cand not in new_cubes:
                    new_cubes[idx] = cand
                    break
            # Test the two sets are genuinely different as multisets of offsets
            set_orig = tuple(sorted(cubes))
            set_new = tuple(sorted(new_cubes))
            if set_new != set_orig:
                return new_cubes
        return cubes

    def _render_cube_composite(self, cubes, color, yaw_deg, pitch_deg,
                               rng, cfg):
        """Render a 3D composite of unit cubes using isometric-ish projection."""
        srng = random.Random((self.seed or 0) * 7919 + 131)
        bg_pool = ["#ffffff", "#fdfdfb", "#fafafa", "#f6f8fa", "#f4f6fb",
                   "#f8fafc", "#fffdf6", "#fff8f0", "#fef7ec", "#f5fbf4",
                   "#eef6ff", "#fef6f8"]
        bg = srng.choice(bg_pool)
        fw = srng.uniform(3.7, 4.6)
        fh = srng.uniform(3.7, 4.6)
        dpi_val = srng.choice([96, 100, 110, 115, 120])
        fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi_val)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        xlim_pad = srng.uniform(2.7, 3.3)
        ax.set_xlim(-xlim_pad, xlim_pad)
        ax.set_ylim(-xlim_pad, xlim_pad)
        ax.set_aspect("equal")
        ax.axis("off")

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        def project(p):
            x, y, z = p
            # rotate about Z (yaw), then tilt so z goes "up" on screen
            xr = cy * x - sy * y
            yr = sy * x + cy * y
            # orthographic: project (xr, yr, z) with pitch so that
            # screen_y = z * cos(pitch) - yr * sin(pitch)
            sx = xr
            syp = z * cp - yr * sp
            return sx, syp

        # Center the composite
        cx = sum(c[0] for c in cubes) / len(cubes)
        ccy = sum(c[1] for c in cubes) / len(cubes)
        ccz = sum(c[2] for c in cubes) / len(cubes)
        centered = [(c[0] - cx, c[1] - ccy, c[2] - ccz) for c in cubes]

        from matplotlib.colors import to_rgba
        r_, g_, b_, _ = to_rgba(color)
        light = (min(1, r_ + 0.18), min(1, g_ + 0.18), min(1, b_ + 0.18))
        dark = (max(0, r_ - 0.18), max(0, g_ - 0.18), max(0, b_ - 0.18))

        # Draw each cube's 3 visible faces in painter's-algorithm order.
        # Sort cubes by depth (farther first) using yr + z as proxy.
        def depth_key(c):
            x, y, z = c
            return sy * x + cy * y + z * 0.5
        sorted_cubes = sorted(centered, key=depth_key)

        s = 0.5  # half-size of unit cube
        for c in sorted_cubes:
            x0, y0, z0 = c
            # 8 corners of this unit cube
            corners = {
                "lbb": (x0 - s, y0 - s, z0 - s),
                "rbb": (x0 + s, y0 - s, z0 - s),
                "rfb": (x0 + s, y0 + s, z0 - s),
                "lfb": (x0 - s, y0 + s, z0 - s),
                "lbt": (x0 - s, y0 - s, z0 + s),
                "rbt": (x0 + s, y0 - s, z0 + s),
                "rft": (x0 + s, y0 + s, z0 + s),
                "lft": (x0 - s, y0 + s, z0 + s),
            }
            pp = {k: project(v) for k, v in corners.items()}
            # top face (+z)
            ax.fill(
                [pp["lbt"][0], pp["rbt"][0], pp["rft"][0], pp["lft"][0]],
                [pp["lbt"][1], pp["rbt"][1], pp["rft"][1], pp["lft"][1]],
                color=light, edgecolor="black", linewidth=1.0, alpha=0.95)
            # front face (+y)
            ax.fill(
                [pp["lfb"][0], pp["rfb"][0], pp["rft"][0], pp["lft"][0]],
                [pp["lfb"][1], pp["rfb"][1], pp["rft"][1], pp["lft"][1]],
                color=color, edgecolor="black", linewidth=1.0, alpha=0.95)
            # right face (+x)
            ax.fill(
                [pp["rfb"][0], pp["rbb"][0], pp["rbt"][0], pp["rft"][0]],
                [pp["rfb"][1], pp["rbb"][1], pp["rbt"][1], pp["rft"][1]],
                color=dark, edgecolor="black", linewidth=1.0, alpha=0.95)

        if not cfg.get("hide_shape_label"):
            label_variants = [
                f"View (yaw ~{int(yaw_deg)}deg)",
                f"Angle {int(yaw_deg)}°",
                f"Camera yaw ≈ {int(yaw_deg)}°",
                f"view @ yaw {int(yaw_deg)}",
                f"yaw = {int(yaw_deg)}°",
                f"Render (yaw {int(yaw_deg)}°)",
            ]
            fs = srng.choice([8, 9, 10, 11])
            ax.set_title(srng.choice(label_variants), fontsize=fs)

        return self._fig_to_pil_raw(fig, dpi=dpi_val)

    def _render_shape(self, shape, color, angle, rng, cfg):
        style = self._random_style()
        srng = random.Random((self.seed or 0) * 7919 + 257)
        bg_pool = ["#ffffff", "#fdfdfb", "#fafafa", "#f6f8fa", "#f4f6fb",
                   "#f8fafc", "#fffdf6", "#fff8f0", "#fef7ec", "#f5fbf4",
                   "#eef6ff", "#fef6f8"]
        bg = srng.choice(bg_pool)
        fw = srng.uniform(3.7, 4.6)
        fh = srng.uniform(3.7, 4.6)
        dpi_val = srng.choice([96, 100, 110, 115, 120])
        fig, ax = plt.subplots(figsize=(fw, fh), dpi=dpi_val)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        xlim_pad = srng.uniform(1.8, 2.3)
        ax.set_xlim(-xlim_pad, xlim_pad)
        ax.set_ylim(-xlim_pad, xlim_pad)
        ax.set_aspect("equal")
        ax.axis("off")

        rad = math.radians(angle)
        # Draw shape representation
        if shape == "cube":
            # Isometric cube
            s = 0.8
            front = [(-s, -s), (s, -s), (s, s), (-s, s)]
            dx, dy = 0.4 * math.cos(rad * 0.1), 0.3
            back = [(x + dx, y + dy) for x, y in front]
            ax.fill([p[0] for p in front], [p[1] for p in front],
                    color=color, alpha=0.8, edgecolor="black", linewidth=1.5)
            top = [front[2], front[3], back[3], back[2]]
            from matplotlib.colors import to_rgba
            r_, g_, b_, _ = to_rgba(color)
            lighter = (min(1, r_ + 0.2), min(1, g_ + 0.2), min(1, b_ + 0.2))
            ax.fill([p[0] for p in top], [p[1] for p in top],
                    color=lighter, alpha=0.8, edgecolor="black", linewidth=1.5)
            side = [front[1], front[2], back[2], back[1]]
            darker = (max(0, r_ - 0.15), max(0, g_ - 0.15), max(0, b_ - 0.15))
            ax.fill([p[0] for p in side], [p[1] for p in side],
                    color=darker, alpha=0.8, edgecolor="black", linewidth=1.5)
        elif shape == "sphere":
            circle = plt.Circle((0, 0), 1.0, fc=color, ec="black", lw=1.5, alpha=0.8)
            ax.add_patch(circle)
            highlight = plt.Circle((-0.3, 0.3), 0.2, fc="white", alpha=0.4)
            ax.add_patch(highlight)
        elif shape == "cylinder":
            h = 1.2
            w = 0.8
            rect = mpatches.Rectangle((-w, -h / 2), 2 * w, h,
                                       fc=color, ec="black", lw=1.5, alpha=0.8)
            ax.add_patch(rect)
            top_e = mpatches.Ellipse((0, h / 2), 2 * w, 0.5,
                                      fc=color, ec="black", lw=1.5, alpha=0.9)
            ax.add_patch(top_e)
        elif shape == "cone":
            tri = plt.Polygon([(-0.8, -1.0), (0.8, -1.0), (0, 1.2)],
                              fc=color, ec="black", lw=1.5, alpha=0.8)
            ax.add_patch(tri)
        elif shape == "pyramid":
            tri = plt.Polygon([(-1.0, -0.8), (1.0, -0.8), (0.3, 1.0), (-0.3, 1.0)],
                              fc=color, ec="black", lw=1.5, alpha=0.8)
            ax.add_patch(tri)
        elif shape == "torus":
            outer = plt.Circle((0, 0), 1.2, fc=color, ec="black", lw=1.5, alpha=0.8)
            inner = plt.Circle((0, 0), 0.5, fc="#ffffff", ec="black", lw=1.5)
            ax.add_patch(outer)
            ax.add_patch(inner)

        # Add visual noise at high levels
        if cfg.get("visual_noise"):
            for _ in range(rng.randint(2, 5)):
                nx = rng.uniform(-1.8, 1.8)
                ny = rng.uniform(-1.8, 1.8)
                ns = rng.uniform(0.05, 0.15)
                ax.plot(nx, ny, ".", color="#aaaaaa", markersize=ns * 50)

        if not cfg.get("hide_shape_label"):
            label_variants = [
                f"View (angle ~{int(angle)}deg)",
                f"Angle {int(angle)}°",
                f"Rotation ≈ {int(angle)}°",
                f"angle @ {int(angle)}",
                f"θ = {int(angle)}°",
                f"View ({int(angle)}°)",
            ]
            fs = srng.choice([8, 9, 10, 11])
            ax.set_title(srng.choice(label_variants), fontsize=fs)

        buf = self._fig_to_pil_raw(fig, dpi=dpi_val)
        return buf

    def _fig_to_pil_raw(self, fig, dpi=100):
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _combine_images(self, img1, img2):
        """Combine two PIL images side by side with labels."""
        # Resize to same height
        h = max(img1.height, img2.height)
        w1 = int(img1.width * h / img1.height)
        w2 = int(img2.width * h / img2.height)
        img1 = img1.resize((w1, h))
        img2 = img2.resize((w2, h))

        gap = 20
        composite = Image.new("RGB", (w1 + w2 + gap, h + 30), "white")
        composite.paste(img1, (0, 30))
        composite.paste(img2, (w1 + gap, 30))

        # Add labels using matplotlib
        fig, ax = plt.subplots(figsize=(
            (w1 + w2 + gap) / 100, (h + 30) / 100), dpi=100)
        ax.imshow(np.array(composite))
        ax.text(w1 // 2, 15, "Image 1", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black")
        ax.text(w1 + gap + w2 // 2, 15, "Image 2", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=100)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_taskb"
    os.makedirs(out_dir, exist_ok=True)
    env = MultiViewObjectMatchQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[multi_view_object_match L{level} s{s}] FAILED")
                continue
            path = os.path.join(out_dir, f"multi_view_object_match_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[multi_view_object_match L{level} s{s}] A={env._answer}")
