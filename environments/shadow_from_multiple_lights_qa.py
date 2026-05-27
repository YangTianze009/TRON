"""
Shadow From Multiple Lights QA — pure single-letter MCQ.

2026-05-05 R5 Batch 4 (TOO-HARD ease + restructure):
Old design (197 LOC) had wrapper-mismatch bugs in question stems plus mixed
free-form + MCQ semantics. R3 = 20%.

REWRITE strategy:
  - L0/L1: single light + single cube, 4-MCQ over {N/E/S/W} direction the
            shadow falls. No special wrapper hint — framework handles wrapper.
  - L2-L4: 2 lights from perpendicular angles; ask which light's shadow is
            larger, 4-MCQ {Light 1 / Light 2 / Same area / Cannot determine}.
  - L5-L7: 3 lights, larger shapes; 5-MCQ with 15-25% trap.
  - L8/L9: 4 lights, 5-MCQ + 35-40% trap "E. No correct answer".

Pure single-letter MCQ A-D / A-E. ALLOW_ROTATION=False.
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


def _compute_shadow(cubes: List[Tuple[int, int, int]],
                    light_dir: Tuple[float, float, float]) -> set:
    """Compute shadow footprint on the ground (z=0) as a set of grid cells."""
    shadow = set()
    dx, dy, dz = light_dir
    for (cx, cy, cz) in cubes:
        for corner_x in [cx, cx + 1]:
            for corner_y in [cy, cy + 1]:
                top_z = cz + 1
                if dz != 0:
                    t = -top_z / dz
                    sx = corner_x + dx * t
                    sy = corner_y + dy * t
                    shadow.add((int(math.floor(sx)),
                                int(math.floor(sy))))
    return shadow


def _shadow_centroid_dir(cubes, shadow) -> str:
    """Determine the dominant direction (N/E/S/W) of the shadow centroid
    relative to the cube footprint centroid."""
    obj_cx = sum(c[0] for c in cubes) / len(cubes) + 0.5
    obj_cy = sum(c[1] for c in cubes) / len(cubes) + 0.5
    if not shadow:
        return "?"
    sh_cx = sum(c[0] for c in shadow) / len(shadow) + 0.5
    sh_cy = sum(c[1] for c in shadow) / len(shadow) + 0.5
    dx = sh_cx - obj_cx
    dy = sh_cy - obj_cy
    if abs(dx) >= abs(dy):
        return "East" if dx > 0 else "West"
    return "North" if dy > 0 else "South"


# ----------------------------------------------------------------------- #
# Env
# ----------------------------------------------------------------------- #

_DIRECTIONS = ["North", "East", "South", "West"]


class ShadowFromMultipleLightsQA(StandaloneVisualEnv):
    """Pure MCQ shadow direction / shadow comparison."""

    ALLOW_ROTATION = False
    ENV_NAME = "shadow_from_multiple_lights"

    def _level_config(self, level: int) -> dict:
        level = max(0, min(9, level))
        if level <= 1:
            return {"mode": "direction", "n_lights": 1, "n_cubes": 1,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 4:
            return {"mode": "compare", "n_lights": 2, "n_cubes": 2 + level,
                    "use_5_options": False, "trap_rate": 0.0}
        if level <= 6:
            return {"mode": "compare", "n_lights": 3, "n_cubes": 3 + level,
                    "use_5_options": True, "trap_rate": 0.15}
        if level == 7:
            return {"mode": "compare", "n_lights": 3, "n_cubes": 5,
                    "use_5_options": True, "trap_rate": 0.25}
        if level == 8:
            return {"mode": "compare", "n_lights": 4, "n_cubes": 6,
                    "use_5_options": True, "trap_rate": 0.35}
        # L9
        return {"mode": "compare", "n_lights": 4, "n_cubes": 7,
                "use_5_options": True, "trap_rate": 0.40}

    def _generate_problem(self, seed: int, parameter: Dict
                          ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1016)
        for _ in range(40):
            res = self._try_generate(cfg, level, sub_rng)
            if res is not None:
                return res
        return None

    def _try_generate(self, cfg, level, rng):
        if cfg["mode"] == "direction":
            return self._gen_direction(cfg, level, rng)
        return self._gen_compare(cfg, level, rng)

    # -------------------- L0/L1: shadow direction -------------------- #
    def _gen_direction(self, cfg, level, rng):
        # Single cube on ground at (0,0,0)
        cubes = [(0, 0, 0)]
        # Pick a strong cardinal direction and matching light position.
        # Note: light_dir is the direction light TRAVELS. Shadow is cast on the
        # ground in the SAME direction as horizontal component of light_dir.
        # So light_dir = (0, +1.5, -1) -> shadow extends to +y = North.
        dir_choice = rng.choice(_DIRECTIONS)
        if dir_choice == "North":
            light_dir = (0, 1.5, -1)   # shadow extends north
        elif dir_choice == "East":
            light_dir = (1.5, 0, -1)   # shadow extends east
        elif dir_choice == "South":
            light_dir = (0, -1.5, -1)  # shadow extends south
        else:  # West
            light_dir = (-1.5, 0, -1)

        shadow = _compute_shadow(cubes, light_dir)
        # Sanity: shadow must extend in the right direction
        actual_dir = _shadow_centroid_dir(cubes, shadow)
        if actual_dir != dir_choice:
            return None

        # 4-MCQ options = N/E/S/W (no trap at L0/L1)
        opts = list(_DIRECTIONS)
        rng.shuffle(opts)
        opt_letters = ["A", "B", "C", "D"]
        correct_letter = opt_letters[opts.index(dir_choice)]

        # Render
        img = self._render_scene(cubes, [(light_dir, "Light 1")],
                                 [shadow], rng, light_colors=["#c0392b"])

        opt_text = "\n".join(f"{opt_letters[i]}. {opts[i]}"
                             for i in range(4))
        stem = (
            "The diagram shows one cube and one light source on a top-down "
            "view. The shadow cast by the cube on the ground is shaded. "
            "In which compass direction (relative to the cube) does the "
            "shadow fall? Hint: the shadow falls AWAY from the light. "
            "Be concise."
        )
        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter A, B, C, or D."
        )
        return question, correct_letter, img

    # -------------------- L2+: shadow comparison -------------------- #
    def _gen_compare(self, cfg, level, rng):
        n_cubes = cfg["n_cubes"]
        cubes = [(0, 0, 0)]
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1)]
        for _ in range(n_cubes - 1):
            for _a in range(50):
                base = rng.choice(cubes)
                d = rng.choice(dirs)
                nc = (base[0] + d[0], base[1] + d[1], base[2] + d[2])
                if nc not in cubes and nc[2] >= 0:
                    cubes.append(nc)
                    break
        if len(cubes) < n_cubes:
            return None

        n_lights = cfg["n_lights"]
        light_dirs = []
        # Make light directions deterministic and varied
        candidate_xy = [(2.0, 0), (-2.0, 0), (0, 2.0), (0, -2.0),
                        (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5), (-1.5, -1.5),
                        (0, 0)]  # 0,0 = overhead (small shadow)
        rng.shuffle(candidate_xy)
        for i in range(n_lights):
            dx, dy = candidate_xy[i]
            light_dirs.append((dx, dy, -1))

        shadows = [_compute_shadow(cubes, ld) for ld in light_dirs]
        areas = [len(s) for s in shadows]

        # Find the largest-area light
        max_area = max(areas)
        max_count = sum(1 for a in areas if a == max_area)
        if max_count > 1:
            # Tie — answer would be "Same area"
            true_text = "Same / cannot determine"
        else:
            max_idx = areas.index(max_area)
            true_text = f"Light {max_idx + 1}"

        # Build option list. We always include all real lights + "Same/cannot
        # determine". If n_value > len(real options) we pad with extra
        # plausible but wrong "Light K" labels (where K > n_lights) — these
        # are guaranteed wrong because no such light exists.
        real_options = [f"Light {i + 1}" for i in range(n_lights)]
        real_options.append("Same / cannot determine")

        use_5 = cfg["use_5_options"]
        n_value = 5 if use_5 else 4
        opt_letters = ["A", "B", "C", "D", "E"][:n_value]

        # Build pool: include all real options
        pool = list(real_options)
        # Pad with extra "Light K" placeholders (always wrong) until len == n_value
        extra_idx = n_lights + 1
        while len(pool) < n_value:
            pool.append(f"Light {extra_idx}")
            extra_idx += 1
        if true_text not in pool:
            return None

        use_trap = (rng.random() < cfg["trap_rate"]) if use_5 else False

        if use_trap:
            wrong_pool = [o for o in pool if o != true_text]
            if len(wrong_pool) < n_value - 1:
                return None
            rng.shuffle(wrong_pool)
            chosen = wrong_pool[:n_value - 1]
            opt_lines = chosen + ["No correct answer"]
            correct_letter = opt_letters[-1]
        else:
            opt_lines = list(pool)
            rng.shuffle(opt_lines)
            correct_letter = opt_letters[opt_lines.index(true_text)]

        # Render
        light_colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
        labeled_lights = [(light_dirs[i], f"Light {i + 1}")
                          for i in range(n_lights)]
        img = self._render_scene(cubes, labeled_lights, shadows, rng,
                                 light_colors=light_colors[:n_lights])

        opt_text = "\n".join(f"{opt_letters[i]}. {opt_lines[i]}"
                             for i in range(n_value))
        letter_str = ", ".join(opt_letters[:-1]) + f", or {opt_letters[-1]}"
        stem = (
            f"The diagram shows {n_cubes} unit cubes (top-down view) and "
            f"{n_lights} light sources, each casting a shadow shaded in a "
            f"distinct color. Which light source casts the LARGEST-area "
            f"shadow on the ground?"
        )
        question = (
            f"{stem}\n"
            f"{opt_text}\n"
            f"Answer with a single letter {letter_str}."
        )
        return question, correct_letter, img

    # -------------------- Rendering -------------------- #
    def _render_scene(self, cubes, light_list, shadows, rng,
                      light_colors=None) -> Image.Image:
        if light_colors is None:
            light_colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 7 * sc), dpi=style["dpi"])
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw shadows (each light has its color)
        for sh, color in zip(shadows, light_colors):
            for cell in sh:
                rect = mpatches.Rectangle((cell[0], cell[1]), 1, 1,
                                          facecolor=color, alpha=0.30,
                                          edgecolor=color, linewidth=0.7)
                ax.add_patch(rect)

        # Draw cube footprints (top-down)
        for (cx, cy, cz) in cubes:
            rect = mpatches.Rectangle((cx, cy), 1, 1,
                                      facecolor="#34495e",
                                      edgecolor="#1c2833", linewidth=2.0)
            ax.add_patch(rect)
            # Label the cube z-level
            if cz > 0:
                ax.text(cx + 0.5, cy + 0.5, f"z={cz}",
                        ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

        # Light indicators with directional arrows
        if cubes:
            obj_cx = sum(c[0] for c in cubes) / len(cubes) + 0.5
            obj_cy = sum(c[1] for c in cubes) / len(cubes) + 0.5
        else:
            obj_cx = obj_cy = 0
        for i, ((dx, dy, dz), label) in enumerate(light_list):
            color = light_colors[i % len(light_colors)]
            # Light position is opposite to shadow direction; place marker
            # on the opposite side from the shadow.
            lx = obj_cx - dx * 2
            ly = obj_cy - dy * 2
            ax.plot(lx, ly, marker='*', color=color, markersize=24,
                    markeredgecolor='black', markeredgewidth=1.0)
            ax.text(lx, ly + 0.4, label, ha="center", va="bottom",
                    fontsize=11, color=color, fontweight="bold")

        ax.autoscale_view()
        ax.margins(0.3)
        ax.set_title("Top-down view: cubes and shadows",
                     fontsize=style["font_size_base"] + 3, fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])


if __name__ == "__main__":
    env = ShadowFromMultipleLightsQA()
    for lv in [0, 3, 6, 9]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
