"""
Solid Geometry QA — pure single-letter MCQ A-D / A-E with "No correct answer".

2026-05-05 R5 P3 HARDENED: complete rewrite. Was previously 95% saturated
because it accepted numeric answers in many formats and tolerated 5%/0.5
slop. New design forces the model to:

  1. Read dimensions from the IMAGE (labeled axonometric solid).
  2. Pick the correct V or SA formula for that solid.
  3. Compute with pi = 3.14 (NOT pi exact).
  4. Pick from 4-5 tight numerical distractors (off-by-unit-conversion,
     off-by-formula-coefficient, off-by-arithmetic — not random numbers).

Per-level design:
  L0/L1 — 4-MCQ. Cube / cuboid V or SA. Integer answers. No trap.
  L2/L3 — 5-MCQ + 10% trap. Adds cylinder.
  L4/L5 — 5-MCQ + 15% trap. Adds cone & sphere.
  L6/L7 — 5-MCQ + 25% trap. Adds composite (cylinder on cuboid).
  L8/L9 — 5-MCQ + 40% trap. Composite + decimal dimensions, all 4 wrong
          options very tight; ground truth is 'E' often.

Question phrasing matches Chinese-elementary-textbook style:
  "As shown in the diagram, this solid has its dimensions labeled on the
   figure. Compute its volume (using π = 3.14)."
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


PI = 3.14
_LABEL_PALETTE = ["#c0392b", "#2980b9", "#16a085", "#8e44ad",
                  "#d35400", "#27ae60", "#2c3e50", "#7f1d1d"]
_VIEW_PRESETS = [(18, 32), (22, 45), (25, 60), (28, 120),
                 (20, -30), (30, -60), (24, 140), (16, 200),
                 (32, 75), (18, 250)]


def _fmt(v: float, decimals: int = 2) -> str:
    """Format a number — int when possible, else `decimals` decimal places."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.{decimals}f}"


class SolidGeometryQA(StandaloneVisualEnv):
    """Pure single-letter MCQ — given a labeled 3D solid, pick the V/SA option."""

    ENV_NAME = "solid_geometry"
    ALLOW_ROTATION = False  # dimension labels are orientation-sensitive

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        if level <= 1:
            return {
                "shapes": ["cube", "cuboid"],
                "tasks": ["volume", "surface_area"],
                "use_5_options": False,
                "trap_rate": 0.0,
                "decimals_in_dims": False,
                "tight_distractors": False,
            }
        if level <= 3:
            return {
                "shapes": ["cube", "cuboid", "cylinder"],
                "tasks": ["volume", "surface_area"],
                "use_5_options": True,
                "trap_rate": 0.10,
                "decimals_in_dims": False,
                "tight_distractors": False,
            }
        if level <= 5:
            return {
                "shapes": ["cube", "cuboid", "cylinder", "cone", "sphere"],
                "tasks": ["volume", "surface_area"],
                "use_5_options": True,
                "trap_rate": 0.15,
                "decimals_in_dims": False,
                "tight_distractors": False,
            }
        if level <= 7:
            return {
                "shapes": ["cylinder", "cone", "sphere", "composite"],
                "tasks": ["volume", "surface_area"],
                "use_5_options": True,
                "trap_rate": 0.25,
                "decimals_in_dims": False,
                "tight_distractors": True,
            }
        # L8/L9
        return {
            "shapes": ["cylinder", "cone", "composite", "cuboid"],
            "tasks": ["volume", "surface_area"],
            "use_5_options": True,
            "trap_rate": 0.40,
            "decimals_in_dims": True,
            "tight_distractors": True,
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1201)

        for attempt in range(15):
            try:
                res = self._try_generate(sub_rng, cfg, level, attempt)
                if res is not None:
                    return res
            except Exception:
                continue
        return None

    def _try_generate(self, rng, cfg, level, attempt):
        shape = rng.choice(cfg["shapes"])
        task = rng.choice(cfg["tasks"])

        # Dispatch — each builder returns (true_value, dim_dict, image)
        if shape == "cube":
            params = self._sample_cube(rng, cfg)
            true_value = self._compute(shape, task, params)
            image = self._render_cube(rng, params)
        elif shape == "cuboid":
            params = self._sample_cuboid(rng, cfg)
            true_value = self._compute(shape, task, params)
            image = self._render_cuboid(rng, params)
        elif shape == "cylinder":
            params = self._sample_cylinder(rng, cfg)
            true_value = self._compute(shape, task, params)
            image = self._render_cylinder(rng, params)
        elif shape == "cone":
            params = self._sample_cone(rng, cfg)
            true_value = self._compute(shape, task, params)
            image = self._render_cone(rng, params)
        elif shape == "sphere":
            params = self._sample_sphere(rng, cfg)
            true_value = self._compute(shape, task, params)
            image = self._render_sphere(rng, params)
        elif shape == "composite":
            # Cylinder on cuboid base — composite volume or surface area
            params = self._sample_composite(rng, cfg)
            true_value = self._compute(shape, task, params)
            image = self._render_composite(rng, params)
        else:
            return None

        if true_value is None or true_value <= 0:
            return None

        # Build distractors — tight if cfg says so
        distractors = self._make_distractors(
            rng, shape, task, params, true_value, cfg["tight_distractors"]
        )
        if distractors is None or len(distractors) < 4:
            return None

        # Format the value strings for options
        decimals = 2 if (task == "volume" or task == "surface_area"
                         or cfg["decimals_in_dims"]) else 2
        # Always 2-decimal for cleaner π=3.14 outputs
        true_str = _fmt(true_value, decimals=2)

        # Map distractors to strings, ensure unique
        distr_strs = []
        seen_strs = {true_str}
        for d in distractors:
            ds = _fmt(d, decimals=2)
            if ds in seen_strs:
                continue
            seen_strs.add(ds)
            distr_strs.append(ds)

        if len(distr_strs) < 4:
            return None

        # Decide if we use the 'E. No correct' trap
        use_5 = cfg["use_5_options"]
        opt_letters = ["A", "B", "C", "D", "E"][:(5 if use_5 else 4)]
        n_value = len(opt_letters) - (1 if use_5 else 0)

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            # All 4 options are wrong; correct letter is 'E'
            chosen = distr_strs[:n_value]
            options = list(chosen)
            rng.shuffle(options)
            correct_letter = opt_letters[-1]
        else:
            n_distractor = n_value - 1
            chosen = distr_strs[:n_distractor]
            options = [true_str] + chosen
            rng.shuffle(options)
            correct_letter = opt_letters[options.index(true_str)]

        # Build option block
        opt_lines = "\n".join(
            f"{opt_letters[i]}. {options[i]}" for i in range(n_value)
        )
        if use_5:
            opt_lines += f"\n{opt_letters[-1]}. No correct answer"

        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"

        unit = self._get_unit(task)
        question = self._build_stem(rng, shape, task, params, opt_lines,
                                    letter_str, unit)
        return question, correct_letter, image

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _sample_cube(self, rng, cfg):
        if cfg["decimals_in_dims"]:
            a = rng.choice([3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        else:
            a = rng.randint(3, 9)
        return {"a": a}

    def _sample_cuboid(self, rng, cfg):
        if cfg["decimals_in_dims"]:
            l = rng.choice([3.5, 4.5, 5.5, 6.0, 7.5])
            w = rng.choice([2.5, 3.0, 4.5])
            h = rng.choice([3.0, 4.5, 5.5, 6.5])
        else:
            l = rng.randint(3, 10)
            w = rng.randint(2, 7)
            h = rng.randint(2, 8)
            # Make sure not all equal (would be a cube)
            while l == w == h:
                h = rng.randint(2, 8)
        return {"l": l, "w": w, "h": h}

    def _sample_cylinder(self, rng, cfg):
        if cfg["decimals_in_dims"]:
            r = rng.choice([2.5, 3.5, 4.5, 5.5])
            h = rng.choice([4.0, 5.5, 6.5, 8.0])
        else:
            r = rng.randint(2, 7)
            h = rng.randint(3, 10)
        return {"r": r, "h": h}

    def _sample_cone(self, rng, cfg):
        if cfg["decimals_in_dims"]:
            r = rng.choice([2.5, 3.5, 4.0, 5.5])
            h = rng.choice([4.5, 6.0, 7.5, 9.0])
        else:
            r = rng.randint(2, 7)
            h = rng.randint(4, 10)
        return {"r": r, "h": h}

    def _sample_sphere(self, rng, cfg):
        if cfg["decimals_in_dims"]:
            r = rng.choice([2.5, 3.5, 4.5, 5.5])
        else:
            r = rng.randint(2, 7)
        return {"r": r}

    def _sample_composite(self, rng, cfg):
        # Cylinder sitting on a cuboid base (l x w x h_base), cylinder
        # diameter < min(l, w). Composite V = V_cuboid + V_cyl.
        if cfg["decimals_in_dims"]:
            l = rng.choice([6.0, 7.0, 8.0])
            w = rng.choice([5.0, 6.0])
            h_base = rng.choice([2.0, 2.5, 3.0])
            r = rng.choice([1.5, 2.0, 2.5])
            h_cyl = rng.choice([3.5, 4.0, 4.5])
        else:
            l = rng.choice([6, 7, 8, 9, 10])
            w = rng.choice([5, 6, 7])
            h_base = rng.choice([2, 3])
            r = rng.choice([2, 3])
            h_cyl = rng.choice([3, 4, 5])
        # Force diameter ≤ width
        if 2 * r > min(l, w):
            r = (min(l, w) - 1) / 2
        return {"l": l, "w": w, "h_base": h_base, "r": r, "h_cyl": h_cyl}

    # ------------------------------------------------------------------ #
    # Compute (true value)
    # ------------------------------------------------------------------ #
    def _compute(self, shape, task, p):
        if shape == "cube":
            a = p["a"]
            return a ** 3 if task == "volume" else 6 * a ** 2
        if shape == "cuboid":
            l, w, h = p["l"], p["w"], p["h"]
            if task == "volume":
                return l * w * h
            return 2 * (l * w + l * h + w * h)
        if shape == "cylinder":
            r, h = p["r"], p["h"]
            if task == "volume":
                return PI * r * r * h
            return 2 * PI * r * r + 2 * PI * r * h  # total SA
        if shape == "cone":
            r, h = p["r"], p["h"]
            if task == "volume":
                return PI * r * r * h / 3
            slant = math.sqrt(r * r + h * h)
            return PI * r * slant + PI * r * r  # lateral + base
        if shape == "sphere":
            r = p["r"]
            if task == "volume":
                return 4 / 3 * PI * r ** 3
            return 4 * PI * r * r
        if shape == "composite":
            l, w = p["l"], p["w"]
            h_base, r, h_cyl = p["h_base"], p["r"], p["h_cyl"]
            v_cuboid = l * w * h_base
            v_cyl = PI * r * r * h_cyl
            if task == "volume":
                return v_cuboid + v_cyl
            # Surface area: top of cuboid (minus circle base of cyl)
            #              + 4 sides of cuboid + bottom of cuboid
            #              + lateral of cylinder + top of cylinder
            top_cuboid = l * w
            base_circle = PI * r * r
            sides_cuboid = 2 * (l * h_base + w * h_base)
            bottom_cuboid = l * w
            lat_cyl = 2 * PI * r * h_cyl
            top_cyl = PI * r * r
            return (top_cuboid - base_circle + sides_cuboid + bottom_cuboid
                    + lat_cyl + top_cyl)
        return None

    # ------------------------------------------------------------------ #
    # Distractors — 4 tight wrong numbers per problem
    # ------------------------------------------------------------------ #
    def _make_distractors(self, rng, shape, task, p, true_val, tight):
        """Generate up to 6 confusable wrong values, then return 4."""
        candidates = []

        # 1. Forgot the *coefficient* (used 1 instead of 1/3 / 4/3 / 6)
        if shape == "cone" and task == "volume":
            r, h = p["r"], p["h"]
            candidates.append(PI * r * r * h)  # forgot /3 (cylinder vol)
            candidates.append(2 * PI * r * r * h / 3)  # *2 mistake
        if shape == "sphere" and task == "volume":
            r = p["r"]
            candidates.append(PI * r ** 3)  # forgot 4/3
            candidates.append(4 * PI * r ** 3)  # forgot /3
        if shape == "cube" and task == "surface_area":
            a = p["a"]
            candidates.append(4 * a ** 2)  # forgot 6, used 4 (sides only)
            candidates.append(a ** 2)  # only one face
        if shape == "cuboid" and task == "surface_area":
            l, w, h = p["l"], p["w"], p["h"]
            candidates.append(l * w + l * h + w * h)  # missing factor 2
        if shape == "cylinder" and task == "surface_area":
            r, h = p["r"], p["h"]
            candidates.append(2 * PI * r * h)  # only lateral
            candidates.append(PI * r * r + 2 * PI * r * h)  # only one base
        if shape == "cylinder" and task == "volume":
            r, h = p["r"], p["h"]
            candidates.append(PI * r * h)  # forgot square
            candidates.append(2 * PI * r * r * h)  # *2 mistake
        if shape == "cone" and task == "surface_area":
            r, h = p["r"], p["h"]
            slant = math.sqrt(r * r + h * h)
            candidates.append(PI * r * slant)  # only lateral
            candidates.append(PI * r * h + PI * r * r)  # used h instead of slant

        # 2. Off-by-unit-conversion (×10, ÷10) — common when student forgot
        candidates.append(true_val * 10)
        candidates.append(true_val / 10)

        # 3. Used pi=3 or pi=3.14159 instead of 3.14
        if shape in ("cylinder", "cone", "sphere", "composite"):
            ratio = 3.0 / PI
            candidates.append(true_val * ratio)
            ratio2 = math.pi / PI
            candidates.append(true_val * ratio2)

        # 4. Tight perturbation: ±5% / ±10% — only if tight mode
        if tight:
            for delta in (0.95, 0.97, 1.03, 1.05, 0.93, 1.07):
                candidates.append(true_val * delta)
        else:
            for delta in (0.7, 0.5, 1.5, 2.0, 1.3):
                candidates.append(true_val * delta)

        # 5. Add a couple absolute-shift wrongs
        candidates.append(true_val + 1)
        candidates.append(max(0.1, true_val - 1))

        # Clean: filter out duplicates (within numeric tolerance) of true_val
        good = []
        for c in candidates:
            if c is None:
                continue
            if c <= 0:
                continue
            if abs(c - true_val) < max(0.01, true_val * 0.005):
                continue
            # Avoid duplicate distractors (within rounding)
            if any(abs(c - g) < max(0.01, max(c, g) * 0.003) for g in good):
                continue
            good.append(c)

        rng.shuffle(good)
        return good[:6]

    # ------------------------------------------------------------------ #
    # Question stem
    # ------------------------------------------------------------------ #
    def _get_unit(self, task):
        return "(in cubic units)" if task == "volume" else "(in square units)"

    def _build_stem(self, rng, shape, task, params, opt_lines, letter_str, unit):
        task_word = "volume" if task == "volume" else "total surface area"
        shape_name = {
            "cube": "cube",
            "cuboid": "rectangular cuboid",
            "cylinder": "right circular cylinder",
            "cone": "right circular cone",
            "sphere": "sphere",
            "composite": "composite solid (a cylinder mounted on a rectangular cuboid base)",
        }.get(shape, "solid")
        stems = [
            (
                f"As shown in the diagram, the figure is a {shape_name} with "
                f"its dimensions labeled. Compute its {task_word} {unit} "
                f"using π = 3.14. Round to 2 decimal places. ( )"
            ),
            (
                f"The figure shows a {shape_name} with all key dimensions "
                f"labeled on the image. What is its {task_word} {unit}? "
                f"(Use π = 3.14, round to 2 decimal places.) ( )"
            ),
            (
                f"From the diagram, read the dimensions of the {shape_name} "
                f"and compute the {task_word} {unit}. "
                f"Use π = 3.14, round to 2 decimal places. ( )"
            ),
        ]
        stem = rng.choice(stems)
        return (
            f"{stem}\n"
            f"{opt_lines}\n"
            f"Answer with a single letter {letter_str}."
        )

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #
    def _setup_fig(self, rng, title=None):
        style = self._random_style()
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor(style["bg_color"])
        elev, azim = rng.choice(_VIEW_PRESETS)
        ax.view_init(elev=elev, azim=azim)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        return style, fig, ax

    def _lc(self, rng, idx=0):
        return _LABEL_PALETTE[(rng.randrange(len(_LABEL_PALETTE)) + idx) % len(_LABEL_PALETTE)]

    def _render_cube(self, rng, p):
        a = p["a"]
        style, fig, ax = self._setup_fig(rng, title="Cube")
        col = style["palette"][0]
        for x in [0, a]:
            for y in [0, a]:
                ax.plot3D([x, x], [y, y], [0, a], color=col, linewidth=1.6)
        for x in [0, a]:
            for z in [0, a]:
                ax.plot3D([x, x], [0, a], [z, z], color=col, linewidth=1.6)
        for y in [0, a]:
            for z in [0, a]:
                ax.plot3D([0, a], [y, y], [z, z], color=col, linewidth=1.6)
        ax.text(a / 2, -0.7, 0, f"a = {_fmt(a)}", fontsize=13,
                color=self._lc(rng), fontweight="bold")
        return self.fig_to_pil(fig, dpi=130)

    def _render_cuboid(self, rng, p):
        l, w, h = p["l"], p["w"], p["h"]
        style, fig, ax = self._setup_fig(rng, title="Rectangular cuboid")
        col = style["palette"][0]
        for x in [0, l]:
            for y in [0, w]:
                ax.plot3D([x, x], [y, y], [0, h], color=col, linewidth=1.6)
        for x in [0, l]:
            for z in [0, h]:
                ax.plot3D([x, x], [0, w], [z, z], color=col, linewidth=1.6)
        for y in [0, w]:
            for z in [0, h]:
                ax.plot3D([0, l], [y, y], [z, z], color=col, linewidth=1.6)
        ax.text(l / 2, -0.5, 0, f"l = {_fmt(l)}", fontsize=12,
                color=self._lc(rng, 0), fontweight="bold")
        ax.text(l + 0.3, w / 2, 0, f"w = {_fmt(w)}", fontsize=12,
                color=self._lc(rng, 2), fontweight="bold")
        ax.text(0, 0, h / 2, f"h = {_fmt(h)}", fontsize=12,
                color=self._lc(rng, 4), fontweight="bold")
        return self.fig_to_pil(fig, dpi=130)

    def _render_cylinder(self, rng, p):
        r, h = p["r"], p["h"]
        style, fig, ax = self._setup_fig(rng, title="Right circular cylinder")
        col = style["palette"][0]
        theta = np.linspace(0, 2 * np.pi, 60)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, 0, color=col, linewidth=1.6)
        ax.plot(x, y, h, color=col, linewidth=1.6)
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ax.plot([r * np.cos(angle)] * 2, [r * np.sin(angle)] * 2,
                    [0, h], color=col, linewidth=1.2, alpha=0.6)
        ax.text(r + 0.3, 0, h / 2, f"h = {_fmt(h)}", fontsize=12,
                color=self._lc(rng, 0), fontweight="bold")
        ax.text(0, 0, -0.7, f"r = {_fmt(r)}", fontsize=12,
                color=self._lc(rng, 3), fontweight="bold")
        return self.fig_to_pil(fig, dpi=130)

    def _render_cone(self, rng, p):
        r, h = p["r"], p["h"]
        style, fig, ax = self._setup_fig(rng, title="Right circular cone")
        col = style["palette"][0]
        theta = np.linspace(0, 2 * np.pi, 60)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, 0, color=col, linewidth=1.6)
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ax.plot([r * np.cos(angle), 0], [r * np.sin(angle), 0],
                    [0, h], color=col, linewidth=1.2, alpha=0.7)
        ax.text(r + 0.3, 0, 0, f"r = {_fmt(r)}", fontsize=12,
                color=self._lc(rng, 0), fontweight="bold")
        ax.text(0.3, 0, h / 2, f"h = {_fmt(h)}", fontsize=12,
                color=self._lc(rng, 3), fontweight="bold")
        return self.fig_to_pil(fig, dpi=130)

    def _render_sphere(self, rng, p):
        r = p["r"]
        style, fig, ax = self._setup_fig(rng, title="Sphere")
        col = style["palette"][0]
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color=col, alpha=0.45, linewidth=0.8)
        # Show equator and great circle for clarity
        theta = np.linspace(0, 2 * np.pi, 60)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 0,
                color=col, linewidth=1.5)
        # Label radius
        ax.plot([0, r], [0, 0], [0, 0], color=self._lc(rng, 5),
                linewidth=2.0)
        ax.text(r / 2, 0.2, 0.3, f"r = {_fmt(r)}", fontsize=12,
                color=self._lc(rng, 0), fontweight="bold")
        return self.fig_to_pil(fig, dpi=130)

    def _render_composite(self, rng, p):
        l, w = p["l"], p["w"]
        h_base, r, h_cyl = p["h_base"], p["r"], p["h_cyl"]
        style, fig, ax = self._setup_fig(rng,
                                         title="Composite solid (cylinder on cuboid base)")
        col = style["palette"][0]
        col2 = style["palette"][2]
        # Cuboid base
        for x in [0, l]:
            for y in [0, w]:
                ax.plot3D([x, x], [y, y], [0, h_base], color=col, linewidth=1.6)
        for x in [0, l]:
            for z in [0, h_base]:
                ax.plot3D([x, x], [0, w], [z, z], color=col, linewidth=1.6)
        for y in [0, w]:
            for z in [0, h_base]:
                ax.plot3D([0, l], [y, y], [z, z], color=col, linewidth=1.6)
        # Cylinder centered on top of cuboid
        cx, cy = l / 2, w / 2
        theta = np.linspace(0, 2 * np.pi, 60)
        xc = cx + r * np.cos(theta)
        yc = cy + r * np.sin(theta)
        ax.plot(xc, yc, h_base, color=col2, linewidth=1.6)
        ax.plot(xc, yc, h_base + h_cyl, color=col2, linewidth=1.6)
        for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ax.plot([cx + r * np.cos(angle)] * 2,
                    [cy + r * np.sin(angle)] * 2,
                    [h_base, h_base + h_cyl],
                    color=col2, linewidth=1.2, alpha=0.6)
        # Labels
        ax.text(l / 2, -0.5, 0, f"l = {_fmt(l)}", fontsize=11,
                color=self._lc(rng, 0), fontweight="bold")
        ax.text(l + 0.3, w / 2, 0, f"w = {_fmt(w)}", fontsize=11,
                color=self._lc(rng, 2), fontweight="bold")
        ax.text(0, 0, h_base / 2, f"h_base = {_fmt(h_base)}", fontsize=11,
                color=self._lc(rng, 4), fontweight="bold")
        ax.text(cx + r + 0.3, cy, h_base + h_cyl / 2,
                f"h_cyl = {_fmt(h_cyl)}", fontsize=11,
                color=self._lc(rng, 5), fontweight="bold")
        ax.text(cx + r * 0.6, cy, h_base + 0.1, f"r = {_fmt(r)}",
                fontsize=11, color=self._lc(rng, 6), fontweight="bold")
        return self.fig_to_pil(fig, dpi=130)


if __name__ == "__main__":
    env = SolidGeometryQA()
    for level in (0, 3, 6, 9):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} a={env._answer}")
